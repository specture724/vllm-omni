# Loader / Offloader / HWM 设计摘要

> 本文给出结论。机制和约束见 [总体设计](./LOADER_OFFLOADER_DESIGN.md)，RFC 正文见 [umbrella RFC](./rfcs/00-umbrella.md)。
>
> 范围：vllm-omni 的扩散加载代码（`vllm_omni/diffusion/model_loader/`、`vllm_omni/diffusion/offloader/`）。

## 1. 要解决的问题

| 需求 | 做法 |
|---|---|
| 一台机器上多个 worker 用同一份权重 | 整台机器只占一份 Host 内存，worker 之间共享 |
| 多台小内存机器共同部署一个模型 | 每个 rank 只持有 1/N，运行时拼回完整的块 |
| 加速卡显存不足 | 权重按需在 Host 和加速卡之间搬运 |
| 退出后不留内存垃圾 | 正常退出、`kill -9`、OOM 都由内核回收 |
| 不破坏现有部署 | 新栈与现有实现并存，一个总开关切换 |
| 机制通用 | 没有针对单个模型的代码 |

## 2. 总体关系

```text
Checkpoint ──读取、变换──▶ Loader ──写入──▶ host 存储（HWM 分配）
                                              │
                                              ├─ 本机后端：同机器的 worker 共享一份
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
| Loader | 组件声明、只建结构、checkpoint 映射、变换、写入、收尾 | 分配 host 存储，搬运 |
| HWM（Host Weight Manager） | 按规划分配 host 存储；同机共享；rank 分片 | 理解 checkpoint，锁页，加速卡资源 |
| Offloader | 规划 Unit / Trigger / Transport；锁页、内存登记、暂存区；加速卡资源；搬运和集合通信；关闭 | 读取 checkpoint，决定字节内容 |

## 4. 启动流程

```text
只建结构 → Offloader 规划 → Loader 预检 → HWM 分配存储
→ Offloader 准入 → Loader 写入 → Loader 收尾 → 提交 → 服务
```

- 规划是纯计算；所有可预知的错误在第一次写模型之前报出。
- 权重物化只有一种机制：先分配、后写入。存储先分配好，`load_weights` 逐个张量写进去；收尾时被替换掉的参数最后拷进存储。
- 提交之前的模型实例是可丢弃的，失败就丢弃。
- 提交之后基础权重不可变。

## 5. Host 存储的两个后端

后端由规划决定，没有配置项。

**本机后端**：同一台机器的同一个容器里、内容和规划完全相同的 worker 共享一份。

- 存储是没有文件名的内存文件，由 Linux 的 `memfd_create` 创建。它不占用 `/dev/shm`，大小只受机器内存和容器内存限额约束。
- 内部格式是 safetensors，每个组件按数据类型分成若干内存文件。
- 第一个绑定本机套接字的 worker 负责写。写完后封存为不可写，再把只读的文件描述符交给其他 worker；所有 worker 都以只读方式映射。
- 整个部署只有一个 worker 时使用进程私有内存。

**分片后端**：多台小内存机器共同持有一份。

- 每个重复块的字节拉平后按 rank 等分，每个 rank 持有一段，运行时用 AllGather 拼回完整的块。
- 冷启动时每个 rank 只保留自己那一段，构建过程中的额外占用有上限。
- 分摊的是重复块；其余部分必须单机放得下，放不下在规划阶段报错。

## 6. Offloader 策略

```text
offload 策略 = Unit + Trigger + Transport
```

- **Unit**：整组件、一个重复块，或开头若干块组成的常驻层组。
- **Trigger**：块环（两个共享的加速卡槽位，边算当前块边准备下一块），或使用触发（某个组件被使用时装入，并让别的组件让出）。
- **Transport**：本 rank 直接 H2D，或分片 H2D 后 AllGather。

重复块自动发现，执行顺序在第一次前向时学到。模型里没有 offload 专用的代码。

## 7. 清理与恢复

- 权重只存在于内存里。
- 共享内存由内核按持有者计数，最后一个持有者退出后立即释放。
- 一个 worker 退出时，这个 stage 的全部 worker 一起终止。
- 服务重启后，每个 stage 从 checkpoint 构建。

## 8. 配置与上线

- 总开关：`--diffusion-weight-stack {legacy,unified}`，默认 `legacy`。`unified` 同时启用新加载流程、HWM 和新 Offloader。
- offload 策略：四个预设 `none`、`module`、`layerwise`、`distributed-layerwise`，或按角色给出的自定义配置。
- `unified` 下不支持的组合在参数解析或规划阶段报错。
- 上线门槛：现有每个有测试的“模型 × offload 策略”组合在 `unified` 下输出一致，内存峰值不高于 `legacy`。HSDP 管理的组件做 offload 的组合在 `unified` 下不再支持，不在门槛之内。
- 门槛达到后默认值翻转为 `unified`；`legacy` 保留一个版本，之后删除。

## 9. 核心不变量

1. 所有可预知的错误在第一次写模型之前报出。
2. 每个最终张量只做一次有语义的写入。
3. 提交后基础权重的字节、布局、共用关系不可变。
4. 一个 rank 稳定后在 Host 上只持有规划指定的那一份。
5. 运行期让出只释放加速卡存储，不往 Host 回写。
6. 所有进程退出后，内存里不留任何权重。
7. 关闭不恢复 CPU 权重，不发起集合通信。

## 10. 最小验收标准

1. 同一台机器上 worker 数增加，Host 上的权重占用不增加。
2. 权重大于单机 Host 内存时，多台机器共同部署，冷启动的占用不超出上限。
3. 权重大于 `/dev/shm` 的大小时正常加载。
4. 对任意 worker 执行 `kill -9` 后，机器上不残留任何共享内存。
5. 新栈的最终权重与完整加载的基线逐张量一致，输出一致。
6. 总开关为 `legacy` 时，现有行为完全不变。
