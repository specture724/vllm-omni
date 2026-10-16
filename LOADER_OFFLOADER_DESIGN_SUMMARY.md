# Loader / Offloader / HWM 设计摘要

> 本文给出结论。机制和约束见 [总体设计](./LOADER_OFFLOADER_DESIGN.md)，RFC 正文见 [umbrella RFC](./rfcs/00-umbrella.md)。
>
> 范围：vllm-omni 的扩散加载代码（`vllm_omni/diffusion/model_loader/`、`vllm_omni/diffusion/offloader/`），以及下发部署标识要经过的编排代码。

## 1. 要解决的问题

| 需求 | 做法 |
|---|---|
| 一台机器上多个 worker 用同一份权重 | 同一个部署在一台机器上只占一份 Host 内存，跨 stage、跨副本共享；不同部署互不共享、互不牵连 |
| 多台小内存机器共同部署一个模型 | 每个 rank 只持有 1/N，运行时拼回完整的块 |
| 加速卡显存不足 | 权重按需在 Host 和加速卡之间搬运 |
| 退出后不留内存垃圾 | 正常退出、`kill -9`、OOM 都由内核回收 |
| 不破坏现有部署 | 新栈与现有实现并存，一个总开关切换；并存期间现有实现的行为和构造代码不变 |
| 机制通用 | 没有针对单个模型的加载或 offload 代码 |

## 2. 总体关系

```text
Checkpoint ──读取、融合、变换──▶ Loader ──写入──▶ host 存储（HWM 分配）
                                                   │
                                                   ├─ 本机后端：同一个部署在同一台机器上共享一份
                                                   └─ 分片后端：每个 rank 只持有自己的 1/N
                                                   │
                                   Offloader：Unit + Trigger + Transport
                                                   │
                                                   ▼
                                                加速卡
```

一句话：

> Loader 决定权重是什么并把最终字节写进去；HWM 决定这些字节放在 Host 内存的哪里、和谁共享；Offloader 在运行时把它们搬到加速卡。权重只存在于内存里，每次启动都从 checkpoint 构建。

## 3. 模块职责

| 模块 | 负责 | 不负责 |
|---|---|---|
| Runner | 构造各模块，推进启动流程，关闭时调用 `close()` | 解析 checkpoint，规划搬运，分配存储 |
| Loader | 组件声明、只建结构、checkpoint 映射、适配器融合、变换、写入、收尾 | 分配 host 存储，搬运 |
| HWM（Host Weight Manager） | 本机记录和共享计划；按规划分配 host 存储；同机共享；rank 分片 | 理解 checkpoint，锁页，加速卡资源 |
| Offloader | 规划 Unit / Trigger / Transport；锁页、内存登记、暂存区；加速卡资源；搬运和集合通信；提交后的守卫；关闭 | 读取 checkpoint，决定字节内容 |

## 4. 启动流程

```text
只建结构 → 拓扑收集 → Offloader 规划 → Loader 预检 → 全组核对、选举、容量准入
→ HWM 分配存储 → Offloader 准入 → Loader 写入 → Loader 收尾 → 提交 → 编译设置 → 服务
```

- 规划是纯计算；所有可预知的错误在第一次写模型之前报出。
- 权重物化只有一种机制：先分配、后写入。存储先分配好，各来源的写入入口逐个张量写进去；适配器在 checkpoint 读取器里融合，在线量化逐层进行；收尾时被替换掉的参数最后拷进存储。
- 新栈不调用模型的加载后钩子，`eval()` 是模型侧唯一的收尾入口。
- 提交之前的模型实例是可丢弃的，失败就丢弃。
- 提交之后基础权重不可变。

## 5. Host 存储的两个后端

后端由规划决定，没有配置项。

**本机后端**：部署标识相同、在同一台机器的同一个容器里、内容和规划完全相同的 worker 共享一份，不论属于哪个 stage、哪个副本。

- 部署标识是编排进程每次启动时生成的随机数，下发到这个部署的每个扩散 worker。不同部署的标识不同，即使权重相同也不共享。
- 存储是没有文件名的内存文件，由 Linux 的 `memfd_create` 创建。它不占用 `/dev/shm`，大小只受机器内存和容器内存限额约束。
- 内部格式是 safetensors，每个组件按数据类型分成若干内存文件。
- 每组同机 rank 的代表绑定一个包含部署标识的本机套接字名，绑定成功的是写入进程。写完后封存为不可写，再把只读的文件描述符交给其他 worker；所有 worker 都用只读的共享映射绑定。
- 后启动的副本、运行期才加入的副本直接拿到已封存的存储；写入进程已经退出时重新选举、重新构建。

**分片后端**：多台小内存机器共同持有一份。

- 每个重复块的字节拉平后按 rank 等分，每个 rank 持有一段，运行时用 AllGather 拼回完整的块。AllGather 组按组件确定：一个副本里这个组件内容身份相同的全部 rank。
- 冷启动时每个 rank 只保留自己那一段，构建过程中的额外占用有上限。
- 分摊的是重复块；其余部分必须单机放得下，放不下在规划阶段报错。

## 6. Offloader 策略

```text
offload 策略 = Unit + Trigger + Transport
```

- **Unit**：整组件、一个重复块、组件声明里一个命名组，或开头若干块组成的常驻层组。
- **Trigger**：块环（一块共享的缓冲区，按后继表边算当前块边准备下一块），或使用触发（某个组件被使用时装入，并让别的组件让出）。
- **Transport**：本 rank 直接 H2D，或分片 H2D 后 AllGather。

重复块按通用规则自动发现，模型可以在组件声明里用 `repeated_blocks` 覆盖。没被选中的 VAE 常驻加速卡；使用 DiT 时其他被选中的组件让出，编码器之间不互相让出。

模型里没有 offload 专用的代码。模型只提供组件声明和一组封闭的、带类型的结构字段（`repeated_blocks`、`unit_groups`、`shard_group`、`fixed_adapters`），驻留、设备和搬运全部由 Offloader 决定。

- **动态 LoRA**：包装在只建结构时完成，LoRA 槽位在启动时分配并常驻加速卡；提交后切换 adapter 只拷数据。
- **睡眠模式**：Offloader 的临时槽位在睡眠时直接释放；常驻加速卡的权重由睡眠模式搬出搬回；host 存储不受影响；唤醒后地址不变。
- **CUDA graph 与编译**：模型自己捕获的 CUDA graph 和 cudagraph 编译模式只用于全部常驻加速卡的组件，其他组件走 eager 或 `default` 编译。

## 7. 清理与恢复

- 权重只存在于内存里。
- 共享内存由内核按持有者计数，最后一个持有者退出后立即释放。
- 一个 worker 退出时，它所在副本的全部 worker 一起终止；其他副本照常服务，已绑定的共享内存不受影响。
- 服务重启后，每个副本从 checkpoint 构建，部署标识重新生成。

## 8. 配置与上线

- 总开关：`--diffusion-weight-stack {legacy,unified}`，默认 `legacy`。`unified` 同时启用新加载流程、HWM 和新 Offloader。
- offload 策略：`--diffusion-offload` 的四个预设 `none`、`module`、`layerwise`、`distributed-layerwise`，或 `--diffusion-offload-policy` 按角色或组件名给出的自定义配置。旧开关和旧的 `--diffusion-offload-config` 由兼容层翻译；要让 VAE 由使用触发管理，在自定义配置里选中它。
- host 策略：`--pin-cpu-memory` / `--no-pin-cpu-memory`，`--diffusion-host-registration-limit-gib`。
- 动态 LoRA 需要在启动时给出 `max_lora_rank`。
- `unified` 下不支持的组合在参数解析或规划阶段报错。
- 并存期间，构造期自己读张量的 pipeline 只在读张量的调用点分成 legacy、unified 两支，legacy 一支是原语句；每个迁移家族有以 legacy 为基准的逐张量对比测试。
- 上线门槛：现有每个有测试的“模型 × offload 策略”组合在 `unified` 下输出一致，内存峰值不高于 `legacy`。HSDP 管理的组件做 offload 的组合在 `unified` 下不再支持，不在门槛之内。
- 门槛达到、所有注册的 pipeline 都能在只建结构下构造之后，默认值翻转为 `unified`；`legacy` 保留一个版本，之后删除。

## 9. 核心不变量

1. 所有可预知的错误在第一次写模型之前报出。
2. 每个最终张量只做一次有语义的写入。
3. 提交后基础权重的字节、布局、共用关系和模块结构不可变。
4. 一个 rank 稳定后在 Host 上只持有规划指定的那一份。
5. 不同部署之间不共享 host 存储，也不互相连接。
6. 运行期让出只释放加速卡存储，不往 Host 回写。
7. 所有进程退出后，内存里不留任何权重。
8. 关闭不恢复 CPU 权重，不发起集合通信。

## 10. 最小验收标准

1. 同一个部署在一台机器上的 worker 数、副本数增加，Host 上的权重占用不增加。
2. 同一台机器上两个部署跑同一个模型，彼此不共享、不连接，一个部署退出不影响另一个。
3. 权重大于单机 Host 内存时，多台机器共同部署，冷启动的占用不超出上限。
4. 权重大于 `/dev/shm` 的大小时正常加载。
5. 对任意 worker 执行 `kill -9` 后，机器上不残留任何共享内存。
6. 新栈的最终权重与完整加载的基线逐张量一致，输出一致。
7. 总开关为 `legacy` 时，现有行为完全不变。
