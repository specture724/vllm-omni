# [RFC]: 统一 Loader / Offloader / Host Weight Manager

> 范围：vllm-omni 的扩散加载代码（`vllm_omni/diffusion/model_loader/`、`vllm_omni/diffusion/offloader/`）。
>
> 基线：upstream `main@726b21a5e`
>
> 设计输入：[设计摘要](../LOADER_OFFLOADER_DESIGN_SUMMARY.md)、[总体设计](../LOADER_OFFLOADER_DESIGN.md)

## 摘要

本 RFC 把扩散模型的权重加载、Host 上的权重存放、运行期在 Host 和加速卡之间的搬运，整理成三个职责单一的模块：

- **Loader**：理解模型和 checkpoint，把最终正确的字节写进参数。
- **Host Weight Manager（HWM）**：决定这些字节放在 Host 内存的哪里、和谁共享。一个接口，两个后端：同一台机器共享一份；多台小内存机器各持有 1/N。
- **Offloader**：用 Unit、Trigger、Transport 三个互相独立的要素管理运行期搬运。

三个模块由一条启动流程串起来：

```text
只建结构 → Offloader 规划 → Loader 预检 → HWM 分配存储
→ Offloader 准入 → Loader 写入 → Loader 收尾 → 提交 → 服务
```

权重只存在于内存里。每次启动都从 checkpoint 构建；所有进程退出后，内存由内核回收。

这三个模块连同启动流程称为新栈。新栈作为一套独立的代码加入主干，由一个总开关启用，与现有实现并存；达到上线门槛后翻转默认值，随后删除现有实现。

## 背景与问题

当前实现里，同一件事有多条互不相通的路径：

1. **两条加载路径**。DiT 主要走 Loader 看得见的 `weights_sources → load_weights → weight_loader`。文本编码器、VAE 等组件在 pipeline 构造函数里直接 `from_pretrained(...).to(device)`，Loader 看不见，无法参与搬运规划、共享和内存核算。
2. **三套 offload 后端**。整组件、逐层、分布式逐层三个后端各自实现同一套生命周期，Host 上的表示、块的发现方式、钩子都不一样，共约 3100 行。`load_format="diffusers"` 还整条 pipeline 用 `from_pretrained` 加载，并直接调用 diffusers 自带的 offload。36 个模型文件声明了 offload 专用的元数据，3 个模型文件自己实现了整组件 offload 的生命周期。
3. **启动交接是隐式的**。Loader 把 `OffloadStartupState` 挂在 pipeline 对象上，分布式逐层后端之后再做 checkpoint 映射、重新绑定参数、调用 `post_load_weights`。映射和租约在什么时候归谁、收尾是否只做了一次，都难以证明。
4. **同机共享只在两条窄路径上成立**。一条是直接映射 checkpoint，只支持 TP 为 1、无量化、无 HSDP；另一条是 Host Weight Runtime 的最终布局，只支持两个模型。其余情况每个进程各有一份完整的私有副本。
5. **多台小内存机器无法部署**。AllGather 模式稳定后每个 rank 只留 1/N，但在多数加载路径上，冷启动时每个 rank 会先得到完整一份再切。
6. **退出后的清理没有保证**。内存文件系统里有名字的文件会熬过进程退出，被杀掉的进程来不及清理。
7. **配置分散**。大约十个字段散在命令行、Python 参数、YAML 里，旧的布尔开关靠优先级隐式选后端。

## 目标

### 权重加载

1. 扩散 pipeline 里所有带权重的组件走同一条加载流程。
2. Loader 是 checkpoint 和模型语义的唯一持有者；机制通用，没有模型专属代码。
3. 每个最终张量只做一次有语义的写入。
4. 所有可预知的错误在第一次写模型之前报出。
5. 动态 LoRA 在提交后不改模块结构和基础权重。

### Host 上的权重

1. 同一台机器上内容相同的 worker 共享一份。
2. 多台小内存机器共同部署一个模型，冷启动的占用也有上限。
3. 所有进程退出后内存里不留任何权重，包括被 `kill -9` 和被 OOM 杀掉的情况。

### Offload

1. 用 Unit、Trigger、Transport 表达全部 offload 策略，三者互相独立。
2. 锁页、内存登记、暂存区、加速卡槽位、流、事件、H2D、集合通信统一归 Offloader。
3. 模型里没有 offload 专用的代码。
4. 关闭时不把权重恢复回 CPU，不发起集合通信。
5. 与睡眠模式共存，唤醒后加速卡地址不变。

### 上线

1. 总开关关闭时，现有行为完全不变。
2. 新栈整体达到门槛后才替换现有路径。

## 非目标

- 自回归和语音 stage。它们使用 vLLM 上游的加载器。
- DP、SP、TP 等并行方式的计算语义和请求调度。
- 激活值和输出的搬运。
- 可变的基础权重。动态 LoRA 是不修改基础权重的叠加层。
- 睡眠 level 2 之后的唤醒。
- HSDP 的加载与运行。被 HSDP 管理的组件由 HSDP 自己加载、分片和拼接。
- Host Weight Runtime（`vllm_omni/host_weight_runtime/`）。这个包不改动，新栈不使用它。

## 总体架构

```text
Runner（组合各模块，推进启动流程）
  ├── Loader ──────── 读取、变换、写入、收尾
  │      │ 写入
  │      ▼
  ├── HWM ─────────── host 存储
  │      ├─ 本机后端：同机器的 worker 共享一份（没有文件名的共享内存）
  │      └─ 分片后端：每个 rank 只持有自己的 1/N
  │
  └── Offloader ───── Unit / Trigger / Transport
         │
         ▼
       加速卡
```

| 模块 | 负责 | 不负责 |
|---|---|---|
| Runner | 构造各模块，推进启动流程，关闭时调用 `close()` | 解析 checkpoint，规划 Unit，分配 host 存储 |
| Loader | 组件声明、只建结构、叠加层包装、checkpoint 映射、变换、写入、收尾、内容身份、最终清单 | 分配 host 存储，H2D，AllGather |
| HWM | 按规划分配 host 存储；同机共享；rank 分片 | 理解 checkpoint，锁页和内存登记，加速卡资源 |
| Offloader | 规划；锁页、内存登记、暂存区；加速卡资源；H2D 与 AllGather；钩子；关闭 | 读取 checkpoint，决定字节内容 |

依赖规则：

1. Loader 只通过接口调用 HWM 和 Offloader，不 import 具体实现。
2. Offloader 不 import Loader 和 safetensors 读取器，不调用模型的 `post_load_weights`。
3. HWM 不解释 checkpoint 的 key，不做任何变换。
4. 模型只向 Loader 提供组件声明和通用结构信息，不读 offload 或 host 配置。
5. 启动状态不挂在 pipeline 对象上传递。

## 名词

| 名词 | 含义 |
|---|---|
| stage | vllm-omni 里承担一个模型阶段的一组 worker 进程 |
| rank | 一个 worker 进程在分布式进程组里的编号，一个 rank 对应一块加速卡 |
| DP / SP / TP | 三种并行方式：数据并行（多个副本各自处理请求）、序列并行（一个请求的序列切给多个 rank）、张量并行（每个 rank 持有参数的一部分） |
| 组件 | Loader 声明的、带名字和角色的模型部分，例如 DiT、文本编码器、VAE |
| 重复块 | 组件里结构相同、依次执行的一组层，例如 DiT 的每一个 block |
| 内容身份 | 判断同一次部署里两个 worker 需要的最终权重是否完全相同的依据 |
| 最终清单 | 加载完成后每个张量的名字、形状、类型、在存储里的位置，以及哪些张量共用同一段存储 |
| host 存储 | HWM 分配的、存放最终权重字节的 Host 内存 |
| rank 分片 | 一个重复块的全部字节拉平后按 rank 等分，某个 rank 持有的那一段 |
| Unit | Offloader 一起驻留、一起搬运的一组权重 |
| Trigger | 决定某个 Unit 什么时候装上加速卡、什么时候让出 |
| Transport | 一个 Unit 怎么从 Host 到加速卡 |
| H2D | 从 Host 内存拷到加速卡 |
| AllGather | 各 rank 把自己手里的一段拼成完整一份的集合通信操作 |
| 锁页、内存登记、暂存区 | 让 Host 内存可以被异步拷到加速卡的三种手段：直接锁住；把已有内存交给驱动登记；先拷进一小块固定的锁页内存 |
| 文件描述符 | 进程手里指向一个已打开文件的编号，可以通过本机套接字交给别的进程 |
| 封存 | Linux 给内存文件加的不可撤销的限制，例如不许写、不许改大小 |

## 启动事务

### 接口

```python
class Offloader:
    def plan(catalog, topology, policy) -> PlacementPlan: ...
    def prepare_materialization(model, plan, storage) -> LoadTransaction: ...

class LoadTransaction:
    def commit(final_manifest) -> OffloadRuntime: ...

class OffloadRuntime:
    def close() -> None: ...

class HostWeightManager:
    def allocate(layout) -> HostStorage: ...

class HostStorage:
    def view(tensor_key) -> torch.Tensor: ...
    def ranges(unit) -> list[MemoryRange]: ...
    def seal(final_manifest) -> None: ...
    def close() -> None: ...
```

阶段、所有权的转移和失败时的行为是规范性的。

### 步骤

```text
1.  Runner 构造 Loader、HWM、Offloader
2.  Loader 只建结构，所有带权重的组件都没有真实数据
3.  Offloader.plan()：纯计算，得出 Unit、Trigger、Transport 和每个组件的 host 存储类型
4.  Loader 预检：来源、内容身份、预期清单
5.  HWM.allocate()：为需要 Host 驻留的张量分配存储
6.  Offloader.prepare_materialization()：写模型之前拿齐所有占容量的资源，返回启动事务
7.  Loader 写入：参数绑定到已分配的存储，load_weights 逐个张量写
8.  Loader 收尾；host 存储定稿
9.  commit()：校验最终清单，装上触发器，资源整体移交 OffloadRuntime
10. 开始服务
```

### 状态

```text
STRUCTURE_READY       结构建好
  → PLANNED           规划完成，模型未被修改
  → SOURCES_PREPARED  来源、内容身份、预期清单确定
  → TRANSACTION_READY 存储和运行期资源全部到位
  → WEIGHTS_READY     所有字节写入或绑定完成
  → FINALIZED         收尾完成，字节不再变化
  → COMMITTED         资源移交给 OffloadRuntime
  → IN_SERVICE
```

### 规则

- `plan()` 不分配大块资源，不改模型，不装钩子。
- 启动事务是提交前所有临时资源的唯一持有者；没提交就退出时自动释放本 worker 的资源。
- 提交成功后，全部资源一次性移交给 `OffloadRuntime`；同一个资源句柄只登记一次。
- 从开始写模型到提交成功，这个模型实例是可丢弃的；写入、收尾或提交失败时直接丢弃。
- 进入 `IN_SERVICE` 之后的失败是运行期故障。

### 多个 rank 的全组确认

规划里有跨 rank 的集合通信时，在三个点上做全组确认，走已有的 torch.distributed 进程组：

1. 规划之后：各 rank 的 Unit、内容身份、布局、Transport 的指纹一致。
2. `prepare_materialization()` 之后：所有 rank 都就绪，才返回启动事务。
3. `commit()` 时：所有 rank 都完成收尾，才一起激活。

任何一个 rank 在确认前失败，整组一起失败。

## Loader

### 组件声明

pipeline 的构造函数里，每个带权重的部分声明一次：

- 稳定的组件名和角色。角色是 `dit`、`text_encoder`、`image_encoder`、`audio_encoder`、`vae`、`other` 之一。
- 组件在 pipeline 里的路径。
- 只用配置、不用权重数值的结构构造方法。
- 权重来源。
- checkpoint 里的形式和内存里的形式不一样时，一张规则表。

### 构造函数只建结构

- vLLM 风格的组件用配置构造函数。
- transformers 的类按 `create_transformers_model` 的方式构造。
- diffusers 的类用 `from_config`。
- 只由配置算出来的普通 buffer 照常创建。
- 构造函数不读 checkpoint 的张量数值，不把带权重的模块搬到设备上。

`from_pretrained` 在加载时顺带做的事（改名、跳过某些部分、让两个参数共用存储、让某些层保持 fp32）由每个类一张规则表表达。每张表配一个测试，逐张量对比它和 `from_pretrained` 的结果。规则表是数据，由 Loader 统一消费。

`load_format="diffusers"` 按 `model_index.json` 声明组件：每个带权重的组件用 `from_config` 只建结构，由 Loader 写入，走同一条流程。diffusers 的 pipeline 对象只负责编排前向和去噪循环，不加载权重，也不调用 diffusers 自带的 offload。分词器、调度器和纯配置不经过 Loader。

### 先分配、后写入

这是唯一的权重物化机制：

1. 写入之前，每个参数的存储已经按规划分配好：留在加速卡上的在加速卡，Host 驻留的在 HWM 分配的 host 存储里。
2. Loader 把参数绑定到这些存储上，然后调用 `load_weights`。它从 checkpoint 逐个张量读出、变换、写入。TP 切分、合并参数、模型自己的映射都使用现有的 `weight_loader` 写法。
3. 一层的全部权重写完时，Loader 立即对这一层做逐层变换：先融合静态或蒸馏 LoRA，再做在线量化。量化在加速卡上进行，结果直接写进这一层的最终存储，量化前的数据随即释放。量化方法在写入之前声明每个输出张量的形状和类型，所以预期清单在写入之前就是最终的。
4. 每个最终张量只做一次有语义的写入。
5. 收尾过程中被整个替换掉的参数（例如会改参数的校准），在收尾的最后一步拷进 host 存储。

这个机制与模型无关。

### 收尾顺序

Loader 按固定顺序做一次：

1. 完成所有写入或绑定。
2. 做需要整个模型参与的、会改字节的工作，例如会改参数的校准。LoRA 融合和在线量化已在写入时逐层完成。
3. 建立共用存储的关系。
4. 校验必需的参数和持久 buffer 都有真实数据。
5. 运行模型的加载后校验，以及不改字节的运行期设置。这一步不改变任何张量的字节、类型或形状；某些层保持 fp32 由规则表表达。
6. 把第 2 步里被替换的参数拷进 host 存储，生成最终清单，host 存储定稿。
7. 交给启动事务 `commit()`。

第 6 步之后字节不再变化。提交之后，基础权重的字节、布局、共用关系都不可变。融合进基础权重的 LoRA 或量化方式要改变时，关闭运行时，重新走一次启动流程。动态 LoRA 不改基础权重，不受此限。

同机共享时的等待者从共享内存直接绑定：第 1 步是绑定，跳过第 2 步和第 6 步里的拷贝，其余照做。

### 内容身份

内容身份在一次部署内部使用：判断同一台机器上的两个 worker 需要的最终权重是否完全相同，以及多个 rank 在规划之后核对彼此是否一致。

它由配置和本 worker 的坐标算出：模型的标识和版本、组件边界、运行期表示（类型、量化方式）、会改变字节或形状的并行坐标（TP 的大小和编号）、静态融合的适配器、Loader 规则表的版本。

它不包含 offload 策略、叠加层、不改变字节的 DP 编号、设备地址和文件路径。

### 叠加层：动态 LoRA

动态 LoRA 是叠加在基础权重之上的叠加层，不修改基础权重。它的结构在启动时定死，提交之后只拷数据。

**启用**。启动配置给出 `max_lora_rank` 才启用动态 LoRA。不启用时，运行期的 `add_lora` 被拒绝。

**包装在只建结构时完成**。

- Loader 在只建结构阶段把目标层包成 LoRA 层。目标层由组件声明里的 DiT 和 pipeline 声明的 LoRA 组件决定。
- 包装只改结构，不分配真实数据。
- 包装后原来的参数路径多一级 `base_layer`。Loader 用一条通用规则把 `x.weight` 和 `x.base_layer.weight` 对应起来；checkpoint 映射、内容身份、最终清单都使用包装前的名字。是否启用动态 LoRA、加载了哪个 adapter，都不影响同机共享。
- 规划看到的是包装后的最终结构：包装层整体属于原来那一层所在的 Unit，钩子装在包装层上。

**叠加层槽位**。

- 每个包装层有一组 A/B 槽位，大小按 `max_lora_rank` 确定。
- 规划把槽位放进“其余部分”：常驻加速卡，不进任何 Unit，不经过 HWM，不进内容身份。
- `prepare_materialization()` 分配槽位，计入加速卡的容量准入。提交时槽位交给动态 LoRA 管理器，地址此后不变。

**运行期**。

- `add_lora` 把 adapter 读进 CPU 缓存，条数由 `max_cpu_loras` 限定。CPU 缓存归动态 LoRA 管理器，不经过 HWM。
- 激活一个 adapter 就是把它的 A/B 拷进槽位，rank 小于槽位时补零。rank 大于 `max_lora_rank` 的 adapter 被拒绝，报错提示调大配置。
- 包装层的前向是 `base(x) + scale · B(A(x))`。基础权重在哪由 Offloader 决定，包装层不关心。
- 动态 LoRA 管理器不读 offload 配置。

**静态和蒸馏 LoRA**。融合进基础权重的 LoRA 只在 Loader 的逐层变换里融合一次，进入内容身份。动态 LoRA 可以叠在融合后的权重上。

## HWM

### 后端的选择

后端由规划决定：

| 组件的 Transport | 后端 | 这个 rank 在 Host 上持有 |
|---|---|---|
| 本 rank 直接 H2D | 本机后端 | 完整一份，同机器共享 |
| 分片 H2D 后 AllGather | 分片后端 | 自己的 rank 分片 |
| 不做 offload，常驻加速卡 | 不经过 HWM | 无 |

### 与并行方式的关系

| 并行方式 | 各 rank 的权重内容 | host 存储 |
|---|---|---|
| DP | 相同 | 本机后端下同机共享；分片后端下各持 1/N |
| SP | 相同 | 同 DP |
| CFG 并行 | 相同 | 同 DP |
| TP | 每个编号不同 | 内容身份包含 TP 的大小和编号；相同编号的 worker 之间共享 |
| HSDP | 由 HSDP 管理 | 不经过 HWM |

### 本机后端：同一台机器共享一份

**共享的条件**。同时满足以下四条的 worker 共享一份：

- 属于同一次部署；
- 在同一台机器的同一个容器里；
- 整套权重的内容身份和规划完全相同；
- 写入之前就能确定每个张量的最终形状和类型。

不满足条件的 worker 使用进程私有内存，各自加载。整个部署只有一个 worker 时使用进程私有内存。

**内存文件**。共享的存储是没有文件名的内存文件，由 Linux 的 `memfd_create` 创建：

- 它位于内核内部的内存文件系统，文件系统里看不到它，只能通过文件描述符访问。
- 它不占用 `/dev/shm`，大小只受机器内存和容器内存限额约束。
- 大小在加载前按预期清单算出，一次定好；物理页在写入时才占用。
- 只要还有进程持有它的文件描述符或映射，它就存在；最后一个持有者退出，内核立即释放。

**格式**。内存文件的内部格式是 safetensors：

- 每个组件按数据类型分成若干内存文件，同一个文件里的张量类型相同，每个张量的起点都按它的类型对齐。
- 布局由张量的名字、形状、类型算出：按名字排序、连续摆放，同一个重复块的张量相邻。
- 最终清单就是 safetensors 的头部。
- 绑定用 safetensors 库的 `safe_open("/proc/self/fd/N")`。

**步骤**：

1. 每个 worker 用“部署标识 + 整套内容身份 + 规划指纹”算出一个本机套接字名。部署标识是执行器下发的分布式初始化地址。套接字使用 Linux 的抽象命名空间，不在文件系统里留文件，进程退出后名字自动消失。
2. 第一个绑定成功的 worker 是写入进程，其余 worker 连接上去等待。绑定是原子的。
3. 写入进程创建内存文件，以可写方式映射，把参数绑定到映射上的各个偏移，完成写入和收尾。
4. 定稿。写入进程撤掉可写映射，给内存文件加上封存：不许写、不许加长、不许缩短。此后任何进程都不能修改它。
5. 交接。写入进程以只读方式重新打开内存文件，用 `socket.send_fds` 把只读的文件描述符发给每个等待者。发送前校验对方和自己是同一个系统用户。
6. 绑定。所有 worker，包括写入进程，都以只读方式映射并按头部逐个名字绑定，走同一段代码。等待者在此之前完成 `prepare_materialization()`，绑定之后做不改字节的收尾。
7. 写入进程持有文件描述符，直到本次启动的所有 worker 都拿到。之后每个 worker 只保留映射，关闭文件描述符。

映射是私有的只读映射，带写时复制：某个 worker 对共享权重的写入只影响它自己。

写入进程在构建过程中退出时，等待者的连接收到结束信号，启动失败。

在线量化在加速卡上逐层进行，写入进程只短暂持有一层量化前的数据。

### 分片后端：多台小内存机器共同持有一份

**布局**。每个重复块的全部字节按类型分组拉平，按 rank 等分成 N 段，不足的部分补零；rank r 持有第 r 段。运行期每个 rank 把自己那段 H2D，再用 `all_gather_into_tensor` 拼成完整的块。

**分组**。AllGather 组是这个 stage 里内容身份相同的全部 rank：TP 编号相同的 DP、SP、CFG 并行 rank 都在同一组，TP 为 1 时就是整个 stage。一个组可以跨多台机器，每台机器持有“本机 rank 数 / N”的权重。内存大的机器放更多的 rank。

**存储**。每个 rank 的分片是它自己的私有内存。

**冷启动**。每个 rank 稳定后只占自己的分片。构建过程有两条路径，由规划选定并写入启动日志：

| 路径 | 条件 | 构建过程中每个 rank 的额外占用 |
|---|---|---|
| 瘦路径 | 写入之前就能确定每个张量的最终形状和类型 | 有上限：一个释放阈值（一个重复块的大小）加上最大的单个张量 |
| 先全量、后裁剪 | 最终形状或类型在收尾时才确定，例如会改形状的校准 | 完整一份，短暂持有 |

瘦路径的做法：

- 每个 rank 为整个组件预留一段地址空间，物理内存在写入时才占用。参数绑定到这段地址空间上，`load_weights` 照常写。
- 完全落在本 rank 范围之外的张量不分配存储，写入对它们是空操作，也不读 checkpoint 里对应的数据。
- 跨越范围边界的张量先完整写入，再把范围之外的页归还给内核。归还按已写字节数的阈值触发。

小内存多机部署使用瘦路径。

**分摊的范围**。跨 rank 分摊的是重复块：DiT 的块，以及各 rank 内容相同的文本编码器的块。DiT 里重复块以外的部分、VAE、图像编码器在每个 rank 上都是完整一份，必须单机放得下；放不下时规划阶段报错，并指出是哪个组件。

**约束**。AllGather 要求同组所有 rank 以相同顺序执行相同的块：

- DP 下同一波请求有相同的去噪步数和影响执行路径的参数。
- CFG 并行下各分支执行相同的块。
- DP 或 CFG 并行配 AllGather 时不能启用缓存加速；DP 配 AllGather 时不能启用 prompt embedding 缓存。

## Offloader

### 策略

```text
offload 策略 = Unit + Trigger + Transport
```

每个被选中的组件有三个互相独立的设置：

| 设置 | 取值 | 含义 |
|---|---|---|
| `mode` | `module` / `layer` | 整个组件一个 Unit，或每个重复块一个 Unit |
| `weight_transfer` | `rank-local` / `allgather` | 本 rank 直接 H2D，或分片 H2D 后 AllGather |
| `resident_layers` | 非负整数 | 开头若干个重复块组成一个常驻层组 |

没被选中的组件常驻加速卡。不做 offload 也走同一个 Offloader 接口，所有 Unit 都常驻加速卡。

### Unit 规划

规划是纯计算，输入是策略、Loader 的组件声明、模型结构、并行拓扑和平台能力，输出是互不重叠的 Unit：

| Unit | 平时在哪 | Trigger |
|---|---|---|
| 整组件 | Host | 使用触发 |
| 流式的重复块 | Host | 块环 |
| 常驻层组 | Host | 使用触发 |
| 其余部分 | 加速卡 | 无 |

- 每个参数和 buffer 只属于一个 Unit。
- 共用同一段存储的张量属于同一个 Unit。跨组件共用且生命周期不兼容时，规划阶段报错。
- 重复块自动发现：组件里每个最外层的、足够大的 `nn.ModuleList` 是一组重复块。通用的结构提示是 `_repeated_blocks` 和 Hugging Face 的 `_no_split_modules`。Offloader 不读任何 offload 专用的模型属性。
- 执行顺序在第一次前向时学到：记录实际执行顺序，第一次遇到的块同步装入，之后按学到的顺序预取。
- 叠加层张量不进任何 Unit，不经过 HWM，常驻加速卡。

### Trigger

**块环**。所有按块流式搬运都用两个共享的加速卡槽位，大小按最大的流式块确定。当前块计算时，下一个块在另一个槽位里准备。块环：

- 检测共享槽位被别的组覆盖。
- 缓存加速跳过预期的块时，同步补装。
- 推迟第一次预取，保证跨 rank 的集合通信顺序一致。
- 编译边界只包住块的计算，不包住钩子。

**使用触发**。

- 钩子装在每个组件的直接子模块、列表元素和重复块上。
- 使用某个组件时，装入它的整组件 Unit 或常驻层组，并让其他组件的这两类 Unit 让出加速卡。被使用的组件无论是否被选中做 offload，都触发让出。角色是 `other` 的组件不触发。
- 让出只释放加速卡上的存储，不往 Host 回写。
- 组件根上不属于任何子模块的小参数常驻加速卡。
- 模型代码绕过所有钩子直接读一个已让出的参数时，空占位直接报错。

**VAE**。启用 offload 后，VAE 按整组件由使用触发管理：被使用时装入，别的组件被使用时让出。VAE 不在可选的角色里。

### Transport

- `rank-local`：从 host 存储直接 H2D。存储所在的内存登记给加速卡驱动后直接异步拷贝；平台不支持登记或超出登记上限时，经过两个固定大小的暂存区。
- `allgather`：把本 rank 的分片 H2D 到一对分片槽位，再拼成完整的块放进块环的槽位。
- 任何策略都不生成完整的锁页副本。

### 资源与运行期约束

`prepare_materialization()` 在写模型之前拿齐：host 存储的内存登记或暂存区、加速卡上的 Unit 槽位、块环槽位、分片槽位、叠加层槽位、流、事件、集合通信的工作区。这些资源是真实分配的。提交时叠加层槽位交给动态 LoRA 管理器，其余交给 `OffloadRuntime`。

一个 `OffloadRuntime` 同一时刻只允许一次前向，可以是一个请求批次；重入或并发的前向被拒绝。出异常时先等拷贝和集合通信做完再回到空闲。集合通信的顺序不再可信时，运行时标记为不可用。

### 关闭

顺序固定：

```text
停止接受新的执行
→ 等加速卡上的拷贝和集合通信做完
→ 撤掉钩子和对模型的引用
→ 释放加速卡槽位、事件、流
→ 撤销内存登记，释放暂存区
→ 关闭 host 存储的本进程映射
```

`OffloadRuntime.close()` 是本地的、终止性的：不把权重恢复回 CPU，不发起集合通信。

### 睡眠模式

睡眠模式在服务暂停期间把加速卡上的权重全部腾出来；Offloader 在服务期间管理权重在加速卡上的短暂驻留。两者按显存的类别分工：

| 加速卡上的显存 | 睡眠时 | 唤醒时 |
|---|---|---|
| 常驻加速卡的权重和 buffer、叠加层槽位 | 由睡眠模式拷到 Host，释放加速卡显存 | 由睡眠模式拷回原地址 |
| Offloader 的临时槽位：块环槽位、分片槽位、整组件 Unit 和常驻层组的槽位 | 直接释放，不备份 | 重新映射到原地址，内容在下一次使用时装入 |
| host 存储 | 不受影响 | 不受影响 |

- 两类显存放在不同的内存池里：常驻的部分在睡眠模式的 `weights` 池；Offloader 的临时槽位在一个睡眠时只释放、不备份的池里。两个池都在 `prepare_materialization()` 里、写模型之前分配。
- 唤醒后地址不变，编译结果和 CUDA graph 继续有效。
- 进入睡眠前，`OffloadRuntime` 等拷贝和集合通信做完，把所有整组件 Unit 和常驻层组标记为已让出，清空块环。唤醒后的第一次前向按正常流程装入。
- 睡眠期间的前向被拒绝。

### 模型侧

模型里没有 offload 专用的代码。新栈只读 Loader 的组件声明和通用的结构提示。

## 清理与恢复

- 权重只存在于内存里：共享内存，或各 worker 的私有内存。
- 共享内存由内核按持有者计数，最后一个持有者退出后立即释放。正常退出、`kill -9`、OOM 的结果相同。
- 一个 worker 退出时，这个 stage 的全部 worker 一起终止。多进程执行器和 Ray 执行器都是这个行为。
- 服务重启后，每个 stage 从 checkpoint 构建。
- worker 进程不产生只复制自己、不换成别的程序的子进程。

## 内存模型

记一个组件的最终权重大小为 `W`，AllGather 组的大小为 `N`。

本机后端，每台机器：

```text
稳态   = W（整台机器一份，与 worker 数无关）+ 每个 worker 的暂存区
冷启动 = 稳态 + checkpoint 的页缓存（内核可回收）
              + 写入进程的瞬时占用（变换中的单个张量；在线量化时是一层量化前的数据）
```

分片后端，每个 rank：

```text
稳态                 = W / N（含等分时的补零）+ 暂存区
冷启动，瘦路径       = 稳态 + 释放阈值 + 最大的单个张量 + checkpoint 的页缓存（内核可回收）
冷启动，先全量后裁剪 = 稳态 + W（短暂）
```

每台机器的占用是本机各 rank 之和。

容量准入在写入之前完成，按整台机器的总内存计算；在容器里按容器的内存限额计算。

## 配置

### 总开关

```text
--diffusion-weight-stack {legacy,unified}
```

默认是 `legacy`，完全使用现有实现。`unified` 同时启用新加载流程、HWM 和新 Offloader。只有这两种组合。

### offload 策略

```text
--diffusion-offload {none,module,layerwise,distributed-layerwise}
--diffusion-offload-policy '<JSON>'
```

四个预设各自等价于一份自定义配置：

| 预设 | 展开 |
|---|---|
| `none` | 不做 offload |
| `module` | 角色为 `dit`、`text_encoder`、`image_encoder` 的组件按整组件 |
| `layerwise` | `dit` 按重复块，`rank-local` |
| `distributed-layerwise` | `dit` 按重复块，`allgather` |

自定义配置按角色选组件，每个组件各自设置 `mode`、`weight_transfer`、`resident_layers`。Python 参数和 YAML 里对应的字段是 `diffusion_offload_policy`：

```json
{
  "mode": "layer",
  "components": {
    "dit": {"weight_transfer": "allgather"},
    "text_encoder": {"mode": "module"}
  }
}
```

- 顶层的 `mode` 是组件的默认粒度，组件可以覆盖。
- 可选的角色是 `dit`、`text_encoder`、`image_encoder`。
- `weight_transfer` 默认是 `rank-local`；`resident_layers` 默认是 0，只用于 `layer`。
- 预设和自定义配置在同一个配置来源里互斥。
- 命令行、Python 参数、YAML 使用同一个解析器；多个来源之间，优先级高的整体覆盖。
- 预设里的角色在某个模型上不存在时忽略；自定义配置里显式选了不存在的角色时，规划阶段报错。
- 未知字段和类型错误报错。

### host 策略

| 设置 | 参数 | 含义 |
|---|---|---|
| 锁页与内存登记 | `pin_cpu_memory` | 开启时登记 host 存储或使用锁页的暂存区；关闭时不登记，暂存区也不锁页 |
| 登记上限 | `--dlo-host-registration-limit-gib` | 每个 worker 登记内存的上限，0 表示不设上限 |

同机共享没有配置项，满足条件即启用。

### 动态 LoRA

| 设置 | 参数 | 含义 |
|---|---|---|
| rank 上限 | `--diffusion-max-lora-rank` / `max_lora_rank` | 给出时启用动态 LoRA，按它分配叠加层槽位 |
| CPU 缓存条数 | `max_cpu_loras` | 已有字段，含义不变 |

`unified` 下没有给出 `max_lora_rank` 时，动态 LoRA 不启用，`add_lora` 被拒绝并提示这个参数。`legacy` 下行为不变。

### 旧开关

旧开关由一个兼容层翻译成策略。`legacy` 下它们驱动现有实现，行为不变；`unified` 下按下表翻译：

| 旧开关 | 翻译 |
|---|---|
| `--enable-cpu-offload` | `module`，组件范围与现有行为一致 |
| `--enable-layerwise-offload` | `layer`，`rank-local`，块的发现范围与现有行为一致，不流式搬运嵌套的子模块 |
| `--enable-distributed-layerwise-offload` | `layer`，DiT 用 `allgather` |
| `--dlo-no-use-allgather` | DiT 用 `rank-local` |
| `--dlo-resident-layers` | DiT 的 `resident_layers` |
| `--diffusion-offload-config` / `diffusion_offload_config` | `mode` 照搬；`components` 列表里的角色按 `mode` 选中；`layer_options` 里每个组件的 `weight_transfer`、`resident_layers` 照搬；`pin_memory` 翻成 `pin_cpu_memory` |

新旧两种写法同时出现且冲突时报错，并给出迁移提示。

`legacy` 下，被 HSDP 管理的组件可以做整组件 offload 或 `rank-local` 逐层 offload。`unified` 下这两种组合在规划阶段报错，报错信息给出两个选择：保留 HSDP、不对这个组件做 offload；或者去掉 HSDP，给这个组件用 `layer` 配 `allgather`。

### 不支持的组合

`unified` 下，以下组合在参数解析或规划阶段报错，并说明原因：

| 组合 | 报错阶段 |
|---|---|
| `module` 配 `allgather` | 参数解析 |
| 自定义配置选择 `dit`、`text_encoder`、`image_encoder` 以外的角色 | 参数解析 |
| `--host-weight-runtime-mode` 不是 `disabled` | 参数解析 |
| DP 或 CFG 并行大于 1 且使用 `allgather` 时启用缓存加速；DP 大于 1 且使用 `allgather` 时启用 prompt embedding 缓存 | 参数解析 |
| 被 HSDP 管理的组件被选中做 offload | 规划 |
| 分片后端下，重复块以外的组件单机放不下 | 规划 |

### 校验分层

| 阶段 | 校验内容 |
|---|---|
| 参数解析 | 类型、未知字段、互斥项、新旧冲突、不支持的组合 |
| Offloader 规划 | 选中的组件存在、重复块可发现、常驻层数合法、共用存储的生命周期一致、单机容量 |
| Loader 预检 | 来源可读、内容身份、冷启动路径 |
| `prepare_materialization()` | 加速卡和 Host 的运行期资源准入、内存登记或暂存区的选择、多 rank 的全组确认 |

每一层都在自己那一步的第一次不可逆修改之前完成。

## 上线方式

1. 新栈作为一套独立的代码加入主干，与现有的三个后端、现有加载路径并存。总开关默认是 `legacy`。
2. 上线门槛：现有每个有端到端测试或每晚例行测试的“模型 × offload 策略”组合，在 `unified` 下输出一致，Host 和加速卡的内存峰值不高于 `legacy`。被 HSDP 管理的组件做 offload 的组合在 `unified` 下不再支持，不在门槛之内。
3. 门槛全部达到后，默认值翻转为 `unified`。
4. 翻转后的一个版本保留 `legacy` 作为回退。
5. 再下一个版本删除 `legacy`：三个现有后端、diffusers 后端整条 pipeline 的 `from_pretrained` 加载和它对 diffusers 自带 offload 的调用、现有的非 HSDP 加载路径（包括其中对 Host Weight Runtime 的接入）、模型里的 offload 专用声明和生命周期代码。

模型里被删除的内容：

- `SupportsComponentDiscovery` 和 `_dit_modules` / `_encoder_modules` / `_vae_modules`，它们的使用方已改读组件声明；
- `_layerwise_offload_blocks_attrs`、模型声明的 `OffloadPlan`、常驻路径声明；
- `SupportsModelCpuOffload`；
- 各模型自己的驻留控制器、阶段上下文和搬运代码；
- 模型里按 offload 策略改变行为的分支；
- worker 里的第二个蒸馏 LoRA 融合点，以及 LoRA 管理器里读 offload 策略的代码。

每个现有的块声明在删除前都有一个测试，证明自动发现的结果与它一致。模型里按特定组合拒绝 offload 的检查由通用规则取代。

## 与既有 RFC 的关系

### #6648（Offloader 统一）

本 RFC 沿用 #6648 的这些结论：单一的拓扑解析；提前校验；省略组件范围时与历史行为一致；组件与 Transport 互相独立；关闭是终止性的，不发起集合通信。

本 RFC 取代 #6648 的这些部分：

- 模型侧的 `OffloadPlan` 由 Loader 的组件声明和重复块自动发现取代。
- 模型里的阶段声明由使用触发取代。
- 策略由 Unit、Trigger、Transport 的组合表达，不再由后端类表达。
- 运行时只有终止性的 `close()`。
- 迁移方式是并行新建，由总开关切换。

### #6414（Host Weight Runtime）

本 RFC 不修改 #6414 的设计和代码。新栈不使用 Host Weight Runtime：同机共享由 HWM 的无名共享内存承担；锁页、内存登记和暂存区归 Offloader，与 #6414 划定的边界一致。

### #7204（配置）

本 RFC 的“配置”一节取代 #7204。

### 子 RFC

本 RFC 之下有五份子 RFC：启动事务、统一加载、host 存储、统一 Offloader、配置。

## 实现拆分

### Job 1：启动事务与总开关

- `plan` / `prepare_materialization` / `commit` / `close` 的接口，以“不做 offload”这条路径落地。
- 总开关；不支持的组合在参数解析阶段报错。
- 依赖规则的静态测试。

### Job 2：Loader

- 组件声明；只建结构的构造方法。
- 每个 Hugging Face 类的规则表和逐张量对比测试。
- 先分配、后写入；逐层的 LoRA 融合和在线量化；固定的收尾顺序；内容身份和最终清单。
- 组件声明的查询接口；`torch.compile`、缓存加速、动态 LoRA、模型注册表改读组件声明。
- 叠加层包装；`load_format="diffusers"` 按组件声明加载。

### Job 3：host 存储

- HWM 接口。
- 本机后端：内存文件、封存、本机套接字交接、只读绑定。
- 分片后端：等分布局、瘦路径、先全量后裁剪。

### Job 4：新 Offloader

- Unit 规划；重复块自动发现和逐类对比测试。
- 块环；使用触发。
- 两种 Transport；内存登记和暂存区。
- 单次执行约束；关闭。
- 叠加层槽位；睡眠模式的两个内存池。

### Job 5：配置与可观测性

- 策略解析、预设、旧开关的兼容层（包括旧的 `--diffusion-offload-config`）、分层校验。
- `max_lora_rank`。
- 指标。
- 文档、recipe、支持矩阵。

### Job 6：达标、翻转、删除

- 跑完门槛矩阵。
- 翻转默认值；保留一个版本的回退。
- 删除 `legacy` 和模型侧代码。

### 依赖

```text
Job 1 ─┬─▶ Job 2 ─┐
       ├─▶ Job 3 ─┼─▶ Job 5 ─▶ Job 6
       └─▶ Job 4 ─┘
```

## 验收标准

### 静态结构

- Loader 不 import Offloader 的实现；Offloader 不 import Loader 和 checkpoint 读取器。
- HWM 在没有加速卡、没有初始化分布式环境的情况下可以测试。
- 模型不读 offload 或 host 配置，不调用运行期的生命周期方法。

### Loader

- 所有组件最终的张量名、形状、类型、步长、buffer、共用关系和字节与完整加载的基线一致。
- 量化、TP 布局、共用存储、收尾，每一项只有一个负责的地方。
- 有测试能发现构造期间读 checkpoint 或把带权重的模块搬到设备上。
- 在线量化时 Host 和加速卡的峰值不高于 `legacy` 的逐层量化。
- 加载、切换、卸载动态 LoRA 前后，模块结构和基础权重的地址不变。

### host 存储

- 同一台机器上 worker 数从 1 增加到 k，各 worker 按共享比例平摊后的内存占用之和不随 k 增长。
- 权重大于 `/dev/shm` 的大小时正常加载，`/dev/shm` 的用量不变。
- 定稿之后，任何进程对内存文件的写入和可写映射都被拒绝。
- 等待者绑定后，已绑定的字节和存储地址在不改字节的收尾前后不变。
- 分片后端稳定后每个 rank 占 `W / N`；瘦路径冷启动的额外占用不超过声明的上限。
- 对任意一个或全部 worker 执行 `kill -9` 后，机器的共享内存用量回到启动前。
- 写入进程在构建中被杀掉，等待者启动失败，没有残留。
- 重复块以外的组件单机放不下时，规划阶段报错并指出组件。

### Offloader

- 各种策略产生互不重叠的 Unit。
- `rank-local` 和 `allgather` 的输出与完整加载的基线一致。
- 块环的内存有固定上限，正确处理真实执行顺序、跳块和异常。
- 每个现有的块声明都有测试证明自动发现的结果与它一致。
- 每个支持的 pipeline 都有“全部组件做 offload”的冒烟测试，覆盖编译和异常路径。
- 整组件模式下没有反复的回写和反复的锁页分配。
- 关闭不恢复 CPU 权重，不发起集合通信；每个资源句柄只释放一次。
- 睡眠再唤醒后输出一致，加速卡地址不变，host 存储不变。

### 事务与故障

- 规划、分配、准入、写入、收尾、提交，每个失败点都有故障注入测试。
- 多 rank 下任何一个 rank 失败，整组一起失败，不死锁。

### 上线

- `legacy` 下现有测试全部不变。
- 门槛矩阵里每个组合保留输出对比和内存峰值的原始数据。

## 可观测性

报告以下五项，分别报告，不相加：

1. 每个 worker 按共享比例平摊后的内存占用，以及整台机器之和。
2. host 存储的字节数，按私有、共享、rank 分片区分。
3. 已登记内存的字节数。
4. 暂存区的字节数。
5. 加速卡槽位的字节数。

启动日志写明规划结果：每个组件的 Unit 划分、使用的后端、是否共享及原因、冷启动路径。

## 反馈期

发布后一周。

## CC

Loader、Offloader、分布式运行时、模型接入、CUDA 与 NPU 平台、配置的负责人。
