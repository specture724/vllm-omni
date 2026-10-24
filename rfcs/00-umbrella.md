# [RFC]: 统一 Loader / Offloader / Host Weight Manager

> 范围：vllm-omni 的扩散加载代码（`vllm_omni/diffusion/model_loader/`、`vllm_omni/diffusion/offloader/`），以及下发部署标识要经过的编排代码（`vllm_omni/engine/stage_runtime.py`、`stage_engine_startup.py`、`stage_init_utils.py`，`vllm_omni/diffusion/data.py`）。
>
> 基线：upstream `main@726b21a5e`
>
> 设计输入：[设计摘要](../LOADER_OFFLOADER_DESIGN_SUMMARY.md)、[总体设计](../LOADER_OFFLOADER_DESIGN.md)

## 摘要

本 RFC 把扩散模型的权重加载、Host 上的权重存放、运行期在 Host 和加速卡之间的搬运，整理成三个职责单一的模块：

- **Loader**：理解模型和 checkpoint，融合适配器，把最终正确的字节写进参数。
- **Host Weight Manager（HWM）**：决定这些字节放在 Host 内存的哪里、和谁共享。一个接口，两个后端：同一个部署在一台机器上共享一份，不同部署之间不共享；多台小内存机器各持有 1/N。
- **Offloader**：用 Unit、Trigger、Transport 三个互相独立的要素管理运行期搬运。

三个模块由一条启动流程串起来：

```text
只建结构 → 拓扑收集 → Offloader 规划 → Loader 预检 → 全组核对、选举、容量准入
→ HWM 分配存储 → Offloader 准入 → Loader 写入 → Loader 收尾 → 提交 → 编译设置 → 服务
```

权重只存在于内存里。每次启动都从 checkpoint 构建；所有进程退出后，内存由内核回收。

模型里没有 offload 专用的代码：模型只提供组件声明和一组封闭的、带类型的结构字段，驻留、设备和搬运全部由 Offloader 决定。

这三个模块连同启动流程称为新栈。新栈作为一套独立的代码加入主干，由一个总开关启用，与现有实现并存；并存期间现有实现的行为和构造代码不变。达到上线门槛后翻转默认值，随后删除现有实现。

## 背景与问题

当前实现里，同一件事有多条互不相通的路径：

1. **两条加载路径**。DiT 主要经 `weights_sources`、`load_weights`、`weight_loader` 这条 Loader 看得见的路径加载。文本编码器、VAE 等组件在 pipeline 构造函数里直接 `from_pretrained(...).to(device)`，Loader 看不见，无法参与搬运规划、共享和内存核算。
2. **三套 offload 后端**。整组件、逐层、分布式逐层三个后端各自实现同一套生命周期，Host 上的表示、块的发现方式、钩子都不一样，共约 3100 行。`load_format="diffusers"` 还整条 pipeline 用 `from_pretrained` 加载，并直接调用 diffusers 自带的 offload。37 个模型文件声明了 offload 专用的元数据；Cosmos3、MiniMax H3 两个 pipeline 实现了整组件 offload 的生命周期协议，MAGI-2 用自己的分阶段组件搬运。
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

1. 同一个部署在一台机器上内容相同的 worker 共享一份，不论属于哪个 stage、哪个副本；不同部署之间不共享、互不牵连。
2. 多台小内存机器共同部署一个模型，冷启动的占用也有上限。
3. 所有进程退出后内存里不留任何权重，包括被 `kill -9` 和被 OOM 杀掉的情况。

### Offload

1. 用 Unit、Trigger、Transport 表达全部 offload 策略，三者互相独立。
2. 锁页、内存登记、暂存区、加速卡槽位、流、事件、H2D、集合通信统一归 Offloader。
3. 模型里没有 offload 专用的代码。模型只提供组件声明和带类型的结构字段。
4. 关闭时不把权重恢复回 CPU，不发起集合通信。
5. 与睡眠模式共存，唤醒后加速卡地址不变。
6. 地址会变的权重不进 CUDA graph。

### 上线

1. 总开关为 `legacy` 时，现有行为完全不变；并存期间现有实现的构造代码不变。
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
  ├── Loader ──────── 读取、融合、变换、写入、收尾
  │      │ 写入
  │      ▼
  ├── HWM ─────────── host 存储
  │      ├─ 本机后端：同一个部署在同一台机器上的 worker 共享一份（没有文件名的共享内存）
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
| Loader | 组件声明、只建结构、叠加层包装、checkpoint 映射、适配器融合、变换、写入、收尾、内容身份、最终清单 | 分配 host 存储，H2D，AllGather |
| HWM | 本机记录和共享计划；按规划结果分配 host 存储；同机共享；rank 分片 | 理解 checkpoint，锁页和内存登记，加速卡资源 |
| Offloader | 规划；锁页、内存登记、暂存区；加速卡资源；H2D 与 AllGather；钩子；提交后的守卫；关闭 | 读取 checkpoint，决定字节内容 |

依赖规则：

1. Loader 只通过接口调用 HWM 和 Offloader，不 import 具体实现。
2. Offloader 不 import Loader 和 safetensors 读取器，不调用模型的任何加载后钩子。
3. HWM 不解释 checkpoint 的键名，不做改名、切分、合并、量化或收尾。
4. 模型只向 Loader 提供组件声明和通用结构信息，不读 offload 或 host 配置，不 import Offloader，不按名字读 offload 钩子，不调用运行期的生命周期方法。
5. 启动状态不挂在 pipeline 对象上传递。

## 名词

| 名词 | 含义 |
|---|---|
| 部署 | 一个编排进程（`Omni`、`AsyncOmni` 里的 `StageRuntime`，分布式模式下是它的子类 `DistStageRuntime`）从启动到关闭的一次运行，包括它直接启动的全部副本，以及在 `DistStageRuntime` 的 `OmniMasterServer` 上注册的 headless 副本 |
| 部署标识 | 编排进程每次启动时生成的 128 位随机数，下发到这个部署的每个扩散 worker。不同部署的标识不同；同一套配置先后启动两次是两个部署，标识也不同 |
| stage | vllm-omni 里承担一个模型阶段的逻辑单元，由一个或多个副本组成 |
| 副本 | 一个 stage 的一份独立运行的实例，有自己的执行器、worker 进程和 torch.distributed 进程组 |
| rank | 一个 worker 进程在分布式进程组里的编号，一个 rank 对应一块加速卡 |
| DP / SP / TP | 三种并行方式：数据并行（一个副本内按 `data_parallel_size` 分成的几组 rank 各自处理不同的请求；多个副本之间不算 DP）、序列并行（一个请求的序列切给多个 rank）、张量并行（每个 rank 持有参数的一部分） |
| HSDP | 一种把参数本身分片放在各 rank 的加速卡上、使用前拼接的并行方式 |
| 组件 | Loader 声明的、带名字和角色的模型部分，例如 DiT、文本编码器、VAE |
| 组件声明的字段 | 组件声明里一组封闭的、带类型的结构字段：`repeated_blocks`、`unit_groups`、`shard_group`、`fixed_adapters` |
| 重复块 | 组件里结构相同、依次执行的一组层，例如 DiT 的每一个块 |
| 规则表 | 每个类一张的数据表：类型、保持 fp32 的条目、改名合并、加载时忽略的键 |
| 适配器 | LoRA 类增量。用户给出、在 checkpoint 读取器里融合进基础权重的叫融合适配器；模型在 `fixed_adapters` 里自带的叫固定适配器；作为动态 LoRA 写进叠加层槽位的叫动态适配器 |
| 内容身份 | 判断同一个部署里两个 worker 需要的某个组件的最终权重是否完全相同的依据 |
| 最终清单 | 加载完成后每个张量的名字、形状、类型、在存储里的位置，以及哪些张量共用同一段存储 |
| host 存储 | HWM 分配的、存放最终权重字节的 Host 内存 |
| 本机键 | 判断两个 worker 能否共享内存文件的五项：boot_id、网络命名空间、PID 命名空间、内存 cgroup、有效 uid |
| 规划指纹 | 规划里决定本机后端存储内容和布局的那部分的摘要：哪些张量进本机后端、按类型怎样分文件、每个张量的偏移 |
| 共享集合 | 部署标识相同，本机键、内容身份、规划指纹也都相同的一组 worker，共用一份共享内存 |
| 写入进程、等待者 | 写入进程是一个共享集合在一台机器上构建并封存共享内存、之后一直给后来者提供文件描述符的 worker；其余 worker 是等待者 |
| rank 分片 | 一个重复块的全部字节拉平后按 rank 等分，某个 rank 持有的那一段 |
| Unit | Offloader 一起驻留、一起搬运的一组权重：整组件、一个重复块、一个命名组、一组常驻层，或 layer 组件里重复块以外的一个直接子模块 |
| Trigger | 决定某个 Unit 什么时候装上加速卡、什么时候让出 |
| Transport | 一个 Unit 怎么从 Host 到加速卡 |
| `fully_resident` | 组件全部常驻加速卡、地址到关闭前不变、不由 HSDP 管理时为真；由 `commit()` 填写 |
| H2D | 从 Host 内存拷到加速卡 |
| AllGather | 各 rank 把自己手里的一段拼成完整一份的集合通信操作 |
| 锁页、内存登记、暂存区 | 让 Host 内存可以被异步拷到加速卡的三种手段：直接锁住；把已有内存交给驱动登记；先拷进一小块固定的锁页内存 |
| 文件描述符 | 进程手里指向一个已打开文件的编号，可以通过本机套接字交给别的进程 |
| 封存 | Linux 给内存文件加的不可撤销的限制，例如不许写、不许改大小 |
| 叠加层 | 叠加在基础权重之上、不修改基础权重的张量，即动态 LoRA 的 A/B 槽位 |

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
    def local_record() -> LocalRecord: ...                           # 本 rank 的本机键、pid、本机可用内存
    def share_plan(topology, fingerprints) -> SharePlan: ...         # 纯函数，各 rank 结果相同
    def elect(share_plan) -> Election: ...                           # 本 rank 是代表时参选
    def admit(topology, share_plan, elections, layout) -> None: ...  # 纯函数，超出容量时报错
    def allocate(layout, share_plan, elections) -> HostStorage: ...

class HostStorage:
    def view(tensor_key) -> torch.Tensor: ...     # 写入目标，或只读视图
    def ranges(unit) -> list[MemoryRange]: ...    # 给 Offloader 做内存登记和 H2D
    def seal(final_manifest) -> None: ...         # 写入进程：定稿、封存、交出只读 fd；等待者：接收、核对、只读映射并绑定
    def close() -> None: ...                      # 只释放本进程的映射；写入进程同时关闭监听套接字和 fd

@dataclass(frozen=True)
class MemoryRange:
    address: int
    length: int
    mapping: Mapping      # 这段内存所在的映射
    read_only: bool       # 共享存储为真，私有存储和 rank 分片为假
```

- `Topology`：部署标识，以及按 rank 排列的 `LocalRecord`。副本内所有 rank 拿到同一份。
- `SharePlan`：本副本内每个共享集合在每台机器上的成员、代表（rank 最小者）和套接字名。
- `Election`：每个共享集合的写入进程在本副本还是在别处；在别处时，它是否已封存、还有多少字节没写入。还记录代表参选时读到的本机可用内存。本 rank 是写入进程时，`Election` 持有监听套接字和监听线程，`allocate()` 时移交给 `HostStorage`；在这之前失败时由 Runner 关闭。
- `ranges(unit)` 正好覆盖这个 Unit 在本进程里的字节，只在 `seal()` 之后（共享存储）或写入完成、页归还之后（私有存储、rank 分片）有效。

阶段、所有权的转移和失败时的行为是规范性的。

### 步骤

```text
1.  Runner 构造 Loader、HWM、Offloader
2.  Loader 只建结构：模型里所有带权重的组件都没有真实数据
3.  拓扑收集
4.  Offloader.plan()：纯计算，得出 Unit、Trigger、Transport，以及每个组件用哪种 host 存储
5.  Loader 预检：解析 checkpoint 和适配器的来源，算出内容身份和预期清单
6.  全组核对、选举、容量准入
7.  HWM.allocate()：为需要 Host 驻留的张量分配存储
8.  Offloader.prepare_materialization()：在第一次写模型之前，
    拿齐所有占容量的运行期资源，返回启动事务
9.  Loader 写入：把参数绑定到已分配的存储，各来源的写入入口从 checkpoint 读取器
    逐个张量写；适配器在读取器里融合，在线量化逐层进行
10. Loader 收尾：核对写入，seal() 里 host 存储定稿、共享存储交接和只读绑定，调用 eval()
11. 启动事务 commit()：校验最终清单，内存登记，填写 fully_resident，装上钩子，
    资源整体移交 OffloadRuntime
12. 编译设置
13. 开始服务
```

### 状态

```text
STRUCTURE_READY       结构建好
  → PLANNED           规划完成，模型未被修改
  → SOURCES_PREPARED  来源、内容身份、预期清单确定
  → ADMITTED          全组核对、选举、容量准入通过
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
- 进入 `IN_SERVICE` 之后的失败是运行期故障：这个副本的全部 worker 终止。

### 多个 rank 的全组确认

全组确认走本副本已有的 torch.distributed 进程组，用 world 组的 gloo 组收集，只覆盖本副本的 rank。每一轮收集都带上本 rank 是否出错；任何一个 rank 出错，所有 rank 一起报错。副本之间不经过 torch.distributed，只通过本机套接字交接。

0. **拓扑收集**：`init_distributed_environment` 之后、`plan()` 之前。各 rank 交出 `HWM.local_record()`，0 号 rank 另外交出部署标识。所有 rank 得到同一份 `Topology`。
1. **核对、选举、准入**：Loader 预检之后、`allocate()` 之前。
   - 各 rank 交出 Unit、内容身份、布局、Transport、规划的指纹。只核对参与集合通信的部分：AllGather 组里的 Unit、它们的内容身份和布局。`rank-local` 的 Unit 和只在本 rank 上声明的组件不参与核对。
   - AllGather 组由收集到的各 rank 组件内容身份确定。本副本全部 rank，包括没有声明这个组件的 rank，按组件名排序、以相同顺序创建这些进程组。组里只剩一个 rank 时在这里报错。
   - `share_plan()` 在每个 rank 上算出相同的共享集合和代表。
   - 每个 rank 都调用 `elect()`：代表参选，不是代表的 rank 返回空的参选结果；参选结果再收集一次。
   - `admit()` 在每个 rank 上得出相同的结论；超出容量时所有 rank 同时报错。这时还没有分配 host 存储，也没有锁页。
2. **`prepare_materialization()` 之后**：所有 rank 都就绪才返回启动事务。之后等待者连接写入进程。
3. **`commit()` 时**：所有 rank 都完成收尾和绑定，才一起激活。

任何一个 rank 在确认前失败，本副本整组一起失败。world size 为 1 时，以上各点都在本进程内完成。

## Loader

### 组件声明

pipeline 的构造函数里，每个带权重的部分声明一次。声明按实例、在配置解析之后生成：同一个类在不同配置下可以有不同的组件。一条声明包含：

- 稳定的组件名和角色。角色是 `dit`、`text_encoder`、`image_encoder`、`audio_encoder`、`vae`、`other` 之一。
- 组件在 pipeline 里的路径。路径可以是一个子模块，也可以是几个子模块的列表；路径嵌套时，内层从外层划出。
- 只用配置、不用权重数值的结构构造方法。
- 权重来源。
- checkpoint 里的形式和内存里的形式不一样时，一张规则表。
- 下表里的带类型字段，按需给出。

**完整性**。规划时，pipeline 下的每个参数和持久 buffer 都必须恰好属于一个组件，否则规划阶段报错并列出路径。只想常驻、不参与 offload 的部分显式声明为 `other`。同一个模块挂在两个属性上时，按模块身份去重。

**按 rank 声明**。声明可以按 rank 不同，没有张量的组件视为不存在。

**带类型字段**。字段是一个封闭的集合，只描述模型是什么，不描述怎么 offload；是否 offload、用哪种方式，由策略决定。每个字段都有一个通用的使用方，进入规划指纹，由期望表测试覆盖。新的需求要加新字段，经过设计评审，不开放任意键。

| 字段 | 含义 | 使用方 |
|---|---|---|
| `repeated_blocks` | 重复块容器的路径列表，取代这个组件的自动发现。列表顺序也是常驻层的计数顺序 | Offloader 规划；`torch.compile` 选择编译区域 |
| `unit_groups` | 命名的、可以单独流式搬运的子模块组，路径可以用通配符 | 策略的 `units` 按名字选择 |
| `shard_group` | TP、EP 以外切分这个组件权重的进程组 | 内容身份、AllGather 分组 |
| `fixed_adapters` | 模型自带的固定适配器：名字、来源、目标层、`apply`（`fuse` 或 `overlay`） | Loader |

**查询**。组件声明在提交之后仍然可以查询。`torch.compile`、缓存加速、动态 LoRA、模型注册表、序列并行钩子和 VAE 快路径都从组件声明读取，不读模型上的 `_dit_modules`、`_encoder_modules`、`_vae_modules`。`commit()` 之后，每个组件还有一个只读属性 `fully_resident`。

### 构造函数只建结构

`legacy` 下每个 pipeline 的构造函数照旧执行，行为不变。只建结构的构造上下文只由 `unified` 的 Loader 进入：

- 参数在 `register_parameter` 时转到 meta 设备上，与外层的 `torch.device` 上下文无关；buffer 照常在 CPU 上创建。
- 构造期间不做设备移动：`Module.to()`、`.cuda()` 对参数和 buffer 只应用类型和内存格式。
- vLLM 风格的组件用配置构造函数；transformers 的类按 `create_transformers_model` 的方式构造，上下文里跳过 `init_parameters`，`create_transformers_model_with_vllm_linears` 同样跳过 `init_parameters_preserving_vllm_linear`；diffusers 的类用 `from_config`。
- 读取陷阱从进入构造上下文一直开到收尾结束。构造期间拦截张量数据的读取并报错：`get_tensor` 和切片取值、`load_file`、`torch.load`、传入真实张量的 `load_state_dict`、`from_single_file`、非 Hugging Face 类自定义的 `from_pretrained`。写入和收尾期间只拦截文件读取；写入入口对已绑定存储的模块调用 `load_state_dict` 属于写入，不拦截。只读 safetensors 头部是允许的。陷阱抛出异常时同时把违例记在上下文里，模型代码用 `try/except` 吞掉了异常，构造或收尾结束时照样失败。
- 构造之后检查：所有参数都在 meta 上；除了经 `declare_weights` 登记了来源的非持久 buffer，buffer 和模块属性里没有 meta 张量。

每个张量的最终类型在只建结构时就定下来：先取构造出的参数类型，再按规则表覆盖。规划、HWM 布局和预期清单看到的都是最终类型。

`from_pretrained` 在加载时顺带做的事由每个类一张规则表表达：共用参数、加载时忽略的键、transformers v5 的权重转换、类型解析、保持 fp32 的条目、写入时的变换（条目可以标注“须在执行设备上做”），以及不在 checkpoint 里但初值有意义的参数。规则表是数据，由 Loader 统一消费；每张表配一个测试，逐张量对比它和 `from_pretrained` 的结果。运行时 Loader 只读入库的表，不读类上的 `_keep_in_fp32_modules`。

被 HSDP 管理的组件同样只建结构，物化时只为 meta 参数分配 CPU 存储，之后照常 `load_weights` 并分片，不经过 HWM。

`load_format="diffusers"` 按 `model_index.json` 声明组件：每个带权重的组件用 `from_config` 只建结构，由 Loader 写入，走同一条流程。diffusers 的 pipeline 对象只负责编排前向和去噪循环，不加载权重，也不调用 diffusers 自带的 offload。`unified` 不调用 diffusers 适配器的 `load_weights()`，也不经过 `build_component`；`legacy` 下适配器在 `load_weights()` 里整条 `from_pretrained` 的代码不改。`load_format="custom_pipeline"` 与注册的 pipeline 规则相同，首次加载在 `re_init_pipeline` 里完整运行启动流程；再次调用时先 `close()`，再完整走一遍启动流程，包括选举。

### 两个栈的构造代码

组件构造函数是两个栈在构造期唯一的分发点：

```python
# vllm_omni/diffusion/model_loader/construct.py
def build_component(*, legacy: Callable[[], T], unified: Callable[[], T]) -> T: ...
def declare_weights(module: nn.Module, *, sources, rules=None, writer=None) -> None: ...
```

- `legacy` 下，`build_component` 只调用 `legacy()`。`unified` 下，构造上下文里只调用 `unified()`；Loader 关闭上下文之后再调用 `build_component`、`declare_weights` 或 HF 组件入口直接报错，写明调用位置。按需构造的组件在 `unified` 一支里于构造函数中建出。
- 分发和读取陷阱看同一个进程级状态。除共存期必改清单里的分支（Job 1，随 `legacy` 删除）以外，模型不读总开关。
- `legacy` 是原来那几条语句原样移进的局部函数，参数、顺序和它调用的函数都不变。
- `unified` 只用配置建结构，返回与 `legacy` 同一个类的实例，模块树相同，并对建出的每个带权重的模块调用一次 `declare_weights`。它调用 legacy 的路径解析、配置解析和校验函数，不复制它们；只有读张量、拼 state dict 的部分另写，放在家族目录下的 `unified_build.py`。需要时可以给原有的类加新方法，原有方法不改。

HF 组件有两个入口，内部走同一个分发：

- `from_pretrained_with_prefetch`：上下文外的代码不变。上下文里读配置，用 `from_config` 在 meta 上建结构，登记 HF 来源和这个类的规则表。类自己覆写了 `from_pretrained` 的，这一步构造后处理写进规则表，在上下文里同样执行。
- `hf_component(factory, *args, **kwargs)`：用在构造函数里直接调用 `X.from_pretrained(...)` 的地方。上下文外就是 `factory(*args, **kwargs)`，不附加预取和重试；上下文里与上一条相同。

权重来源有两种写法：`declare_weights` 登记的来源由 Loader 直接写进那个模块，给了 `writer` 就调用它，模块有 `load_weights(weights)` 就调用它，否则按参数名写入；pipeline 的 `weights_sources` 按前缀归到组件，流交给 pipeline 的 `load_weights`；pipeline 没有 `weights_sources` 时，`unified` 不调用它的 `load_weights`。预先算好的数据张量登记为非持久 buffer 的来源；FastH3、VDN 登记为适配器来源。

构造函数里读张量的调用点以外的部分、`forward`、处理 `weights_sources` 的 `load_weights`、各模块类，两个栈共用一份。共用代码里 legacy 的放置和搬运代码在 `unified` 下不起作用：offload 的查询函数报告“不做 offload”，构造期间的 `.to()` 只改类型和内存格式，模型自己的分阶段和驻留代码按共存期必改清单（Job 1）处理。

新增的 pipeline 按 Z-Image、Kandinsky6 的写法：带权重的组件只用配置构造，或者经 HF 组件入口构造；权重全部经 `weights_sources` 和 `load_weights` 写入，两个栈共用一份构造代码。共存期间不合入构造期读张量的新 pipeline。未迁移的 pipeline 在 `unified` 下构造时启动报错，写明读取发生的位置，并提示改用 `--diffusion-weight-stack legacy`。

### 先分配、后写入

这是唯一的权重物化机制：

1. 写入之前，每个参数的存储已经按规划分配好：留在加速卡上的在加速卡，Host 驻留的在 HWM 分配的 host 存储里。
2. Loader 把参数绑定到这些存储上，然后把每个来源的流交给它的写入入口。流来自 Loader 的 checkpoint 读取器。TP 切分、合并参数、模型自己的映射都使用现有的 `weight_loader` 写法。适配器在 checkpoint 读取器里融合，`load_weights` 看到的已经是融合后的张量。
3. 一层的全部权重写完时，Loader 立即对这一层做在线量化。量化在加速卡上进行，结果直接写进这一层的最终存储，量化前的数据随即释放。量化方法在写入之前声明每个输出张量的形状和类型，所以预期清单在写入之前就是最终的。
4. 每个最终张量只做一次有语义的写入。
5. 收尾过程中被整个替换掉的参数（例如会改参数的校准），在收尾时拷进 host 存储。

这个机制与模型无关。

### 收尾顺序

Loader 按固定顺序做一次：

1. 完成所有写入或绑定。
2. 做需要整个模型参与的、会改字节的工作，例如会改参数的校准。
3. 建立共用存储的关系。
4. 按写入跟踪核对：必需的参数都已写到；从写入开始到这一步，除了第 2 步声明要替换的参数，每个参数和持久 buffer 都仍然绑定在写入前分配的存储上，类型和形状不变。这一步在每个 worker 上做，只核对本 worker 自己写入的张量；共享存储里的张量由写入进程核对。
5. 把第 2 步里被替换的参数拷进 host 存储，生成最终清单。
6. 调用 host 存储的 `seal()`：各个后端的 host 存储在这里定稿；共享存储在这里交接并做只读绑定。
7. 在每个 worker 上调用 pipeline 的 `eval()`。这是模型侧唯一的收尾入口：它只能读权重，可以建立不属于权重的运行期状态，不能改参数和持久 buffer。分片后端 Unit 的权重在本 rank 上只有一段，`eval()` 不能读它们；需要这些权重的派生状态在第一次使用时建立。
8. 交给启动事务 `commit()`。

新栈不调用模型的任何加载后钩子：`post_load_weights`、`validate_loaded_weights`、`validate_restored_host_weights` 都不调用。

第 5 步之后字节不再变化。提交之后，基础权重的字节、布局、共用关系都不可变。融合进基础权重的适配器或量化方式要改变时，关闭运行时，重新走一次启动流程。动态 LoRA 不改基础权重，不受此限。

同机共享时，等待者照常执行第 1 步的写入：写进共享存储的张量对它是空操作，其余张量（常驻加速卡的、私有内存的、rank 分片的）照常写，集合通信与写入进程的顺序相同。共享存储里的张量不做第 4、5 步，在第 6 步的 `seal()` 里接收只读文件描述符、核对并绑定。同机共享要求没有第 2 步的工作。

### 内容身份

内容身份在一个部署内部使用：判断同一台机器上的两个 worker 需要的最终权重是否完全相同，按组件确定 AllGather 组，以及多个 rank 在全组确认时核对彼此是否一致。它按组件计算。

它由配置和本 worker 的坐标算出：模型的标识和版本、组件边界、运行期表示（类型、量化方式）、Loader 规则表的版本；会改变字节或形状的并行坐标（TP 的大小和编号；启用专家并行时加 EP；组件声明了 `shard_group` 时加它的大小和编号）；融合的适配器和固定适配器的标识（用户给出的路径或仓库、版本、格式）。

它不包含 offload 策略、叠加层、不改变字节的 DP 编号、设备地址、解析出的本地缓存文件路径和部署标识。

### 叠加层：动态 LoRA

动态 LoRA 是叠加在基础权重之上的叠加层，不修改基础权重。它的结构在启动时定死，提交之后只拷数据。

**启用**。启动配置给出 `max_lora_rank` 时启用动态 LoRA，槽位按它分配。`lora_backend` 是 `peft` 并给了 `--lora-path`、但没给 `max_lora_rank` 时，槽位取这个适配器自己的 rank，向上取到 LoRA 层支持的最小合法 rank，运行期 `add_lora` 更大的 rank 被拒绝。都不给时，运行期的 `add_lora` 被拒绝。

**包装在只建结构时完成**。

- Loader 在只建结构阶段把目标层包成 LoRA 层。目标层由组件声明里的 DiT 和 pipeline 声明的 LoRA 组件决定。
- 包装只改结构，不分配真实数据。
- 包装后原来的参数路径多一级 `base_layer`。Loader 用一条通用规则把 `x.weight` 和 `x.base_layer.weight` 对应起来；checkpoint 映射、内容身份、最终清单都使用包装前的名字。是否启用动态 LoRA、加载了哪个动态适配器，都不影响同机共享。
- 规划看到的是包装后的最终结构：包装层整体属于原来那一层所在的 Unit，钩子装在包装层上。

**叠加层槽位**。

- 每个包装层有一组 A/B 槽位，大小按 `max_lora_rank` 确定。
- 规划把槽位放进“其余部分”：常驻加速卡，不进任何 Unit，不经过 HWM，不进内容身份。
- `prepare_materialization()` 分配槽位，计入加速卡的容量准入。提交时槽位交给动态 LoRA 管理器，地址此后不变。

**运行期**。

- `add_lora` 把动态适配器读进 CPU 缓存，条数由 `max_cpu_loras` 限定。CPU 缓存归动态 LoRA 管理器，不经过 HWM。
- 激活一个动态适配器就是把它的 A/B 拷进槽位，rank 小于槽位时补零。rank 大于槽位的适配器被拒绝，报错提示调大配置。
- 包装层的前向是 `base(x) + scale · B(A(x))`。基础权重在哪由 Offloader 决定，包装层不关心。
- 动态 LoRA 管理器不读 offload 配置。动态 LoRA 可以叠在融合了适配器的基础权重上。

### 适配器融合与固定适配器

- **融合点是 Loader 的 checkpoint 读取器**，普通路径和 HSDP 路径都从它读。之后的改名、TP 切分、合并参数、HSDP 分片、逐层在线量化看到的都是融合后的张量，所以“先融合、后量化”由数据流保证，TP 大于 1 时也正确。
- **数值**：Δ 在加速卡上计算；默认把 W 和 Δ 都先转成 stage 类型再相加，与现有实现逐位一致，适配器格式可以规定别的规则。融合不乘 `lora_scale`。
- **融不融合由适配器格式决定**：只能融合的格式（MiniMax H3 的 FastH3、VDN）总是融合；普通 A/B 格式在 `lora_backend=distill` 时融合，在 `peft` 时作为启动时激活的动态 LoRA。
- **固定适配器**在组件声明的 `fixed_adapters` 里给出。`apply: fuse` 与用户给出的适配器一样融合；`apply: overlay` 只建结构时包装目标层，A/B 是基础内容，由 Loader 写入，属于目标层所在的 Unit，进入内容身份，不占动态 LoRA 的槽位。
- **预检**在第一次写模型之前：属于已加载组件的键对不上时报错，属于没加载组件的键只警告；融合时 `lora_scale` 不等于 1 报错；checkpoint 已经量化的组件不能融合；HSDP 预分片配 LoRA 或量化在参数解析阶段报错。

## HWM

### 后端的选择

后端由规划决定：

| 组件的 Transport | 后端 | 这个 rank 在 Host 上持有 |
|---|---|---|
| 本 rank 直接 H2D | 本机后端 | 完整一份，同一部署里同一台机器上内容相同的 worker 共享 |
| 分片 H2D 后 AllGather | 分片后端 | 自己的 rank 分片 |
| 不做 offload，常驻加速卡 | 不经过 HWM | 无 |

### 与并行方式的关系

| 并行方式 | 各 rank 的权重内容 | host 存储 |
|---|---|---|
| DP | 相同 | 本机后端下同机共享；分片后端下各持 1/N |
| SP | 相同 | 同 DP |
| CFG 并行 | 相同 | 同 DP |
| TP | 每个编号不同 | 内容身份包含 TP 的大小和编号；相同编号的 worker 之间共享 |
| 专家并行 | 每个编号不同 | 内容身份包含 EP 的大小和编号 |
| 组件声明的 `shard_group` | 每个编号不同 | 内容身份包含切分组的大小和编号 |
| HSDP | 由 HSDP 管理 | 不经过 HWM |

### 本机后端：同一个部署在一台机器上共享一份

**共享的条件**。同时满足以下条件的 worker 组成一个共享集合，共用一份共享内存，不论它们属于哪个 stage、哪个副本：

- 部署标识相同；
- 本机键相同，也就是在同一台机器的同一个容器里，并且是同一个系统用户；
- 整套权重的内容身份和规划指纹相同；
- 写入之前就能确定每个张量的最终形状和类型；
- 收尾第 2 步为空：没有需要整个模型参与、会改字节的收尾工作。

不满足条件的 worker 使用进程私有内存，各自加载。不同部署的部署标识不同，即使权重完全相同也不共享、不互相连接。

**部署标识**。编排进程在构造 `StageRuntime` 时用 `secrets.token_hex(16)` 生成一次，每次启动都重新生成。本地副本在启动进程之前写入 `od_config.deployment_id`；headless 副本从 `OmniMasterServer` 的注册回复拿到同一个值。`deployment_id` 是 `init=False` 的内部字段，不能从配置或命令行设置，不进内容身份。`od_config` 不是由 `StageRuntime` 下发时（例如直接用 `OmniDiffusionConfig.from_kwargs` 构造配置的工具和测试），里面没有部署标识，0 号 rank 在拓扑收集时随机生成一个只在本副本内有效的标识，共享只发生在本副本内。

**本机键**由 `local_record()` 读取：boot_id、网络命名空间和 PID 命名空间的设备号与 inode、内存 cgroup 目录的设备号与 inode、有效 uid。任何一项读不到时，这个 worker 不参与共享。

**规划指纹**只由配置和并行布局决定，不依赖 `LocalRecord` 里的 pid 和本机可用内存，也不依赖每块卡当时的空闲显存。否则同一块卡上依次启动的副本会算出不同的规划，不再共享。

**内存文件**。共享的存储是没有文件名的内存文件，由 `memfd_create`（`MFD_CLOEXEC | MFD_ALLOW_SEALING`）创建：

- 它位于内核内部的内存文件系统，文件系统里看不到它，只能通过文件描述符访问。
- 它不占用 `/dev/shm`，大小只受机器内存和容器内存限额约束。它占用的内存记在写入进程所在的容器上。
- 大小在加载前按预期清单算出，一次定好；物理页在写入时才占用。
- 只要还有进程持有它的文件描述符或映射，它就存在；最后一个持有者退出，内核立即释放。

**格式**。内存文件的内部格式是 safetensors：每个组件按数据类型分成若干内存文件，每个张量的起点都按它的类型对齐；布局由张量的名字、形状、类型算出，按名字排序、连续摆放，同一个重复块的张量相邻；最终清单就是 safetensors 的头部。

**套接字名**是 `\0vllm-omni-hwm/<部署标识>/<共享指纹>`，位于 Linux 的抽象命名空间，持有它的进程退出后名字自动消失。共享指纹是本机键、内容身份、规划指纹和存储格式版本的 128 位摘要。

**选举**。`share_plan()` 在副本内把属于同一个共享集合、又在同一台机器上的 rank 分成一组，rank 最小的是代表。代表创建一个 `AF_UNIX`、`SOCK_SEQPACKET`、带 `SOCK_CLOEXEC` 的套接字并绑定这个名字，绑定是原子的：

- 绑定成功：代表是这个部署在这台机器上这个共享集合的写入进程，立即开始监听，本组其余 rank 是等待者。
- 名字已被占用：写入进程在别的副本里。代表连接它，用 `SO_PEERCRED` 核对对方的 uid 等于自己的有效 uid，本组全部 rank 都是等待者。uid 不一致时按其他错误处理。
- 其他错误：本组不共享，使用私有内存。

**交接**。

1. 写入进程在 `allocate()` 里创建内存文件，以可写方式映射，并对可写映射调用 `madvise(MADV_DONTFORK)`。它的监听线程从选举成功一直运行到 `HostStorage.close()`；每个连接先校验 `SO_PEERCRED` 拿到的 uid，以及对方的协议版本、部署标识、共享指纹、预期清单摘要。
2. 全组确认第 2 点之后，其余等待者连接写入进程，代表沿用选举时的连接。写入进程已经封存时，立即交出文件描述符。
3. 所有 rank 执行同一遍写入，写进共享存储的张量对等待者是空操作。
4. **定稿**：写入进程在 `seal()` 里以只读方式重新打开内存文件，用这个只读文件描述符建 `MAP_SHARED`、`PROT_READ` 映射，按最终清单把全部张量（包括共用存储的别名）重新绑定到只读映射上，然后撤掉可写映射；还有张量引用可写映射时启动失败。之后给内存文件加上封存：不许写、不许加长、不许缩短、不许再改封存。加封存失败时启动失败。
5. **交出**：写入进程用 `socket.send_fds` 把只读的文件描述符和最终清单摘要发给每个已连接的等待者；之后才连上来的等待者立即拿到。
6. **绑定**：等待者用 `SO_PEERCRED` 和 `fstat` 核对对方和文件的 uid 等于自己的有效 uid，再核对封存位、文件大小和清单摘要，然后用只读文件描述符建 `MAP_SHARED`、`PROT_READ` 映射，按名字逐个绑定，与写入进程定稿时走同一段代码，映射之后关闭文件描述符。

这种映射不是写时复制映射：长期只读的内存登记直接锁住共享页，不产生副本；任何对共享权重的写入都会触发段错误。所有映射都用只读打开的文件描述符建立：在 6.7 之前的内核上，对加了 `F_SEAL_WRITE` 的内存文件，用可写打开的描述符建 `MAP_SHARED` 映射，即使只读也会被拒绝。

**顺序**。`seal()` 之前，本副本收尾里的集合通信全部完成，`seal()` 里不做集合通信。写入进程封存并交出之前不等待任何套接字；等待者到 `seal()` 里才阻塞接收。因此副本之间的等待不会和副本内的集合通信互相卡住。

**后来的 worker**。后启动的副本和运行期才注册的副本照常选举。写入进程还在时，连接后立即拿到已封存的存储，容量准入不计这一份。写入进程已经退出时，名字随它一起消失，后来的 worker 成为新的写入进程，重新构建一份；已经绑定旧存储的 worker 不受影响。

**失败**。

| 情况 | 结果 |
|---|---|
| 写入进程在封存之前退出或报错 | 已连接的等待者在连接上收到结束信号，所在副本启动失败。报错给出写入进程的 pid 和它所在的 stage |
| 写入进程在封存之后退出 | 已绑定的 worker 不受影响。之后加入的 worker 重新选举 |
| 写入进程在封存之后、本组等待者都拿到文件描述符之前退出 | 已经收到文件描述符的等待者照常绑定；还没收到的等待者所在副本启动失败，不在启动中途改变角色 |
| 等待者失败 | 写入进程只关闭这个连接，继续构建和监听 |
| 写入进程停住不动 | 等待者一直等到 stage 的启动超时 |

在线量化在加速卡上逐层进行，写入进程只短暂持有一层量化前的数据。

### 分片后端：多台小内存机器共同持有一份

**布局**。每个重复块的全部字节按类型分组拉平，按 rank 等分成 N 段，不足的部分补零；rank r 持有第 r 段。运行期每个 rank 把自己那段 H2D，再用 `all_gather_into_tensor` 拼成完整的块。

**分组**。AllGather 组按组件确定：一个 stage 副本里这个组件内容身份相同的全部 rank。TP 编号相同的 DP、SP、CFG 并行 rank 都在同一组，TP 为 1、没有 `shard_group` 时就是整个副本。组件声明了 `shard_group` 时，组内的 rank 还要在切分组里编号相同。组在全组确认第 1 点创建，组里只剩一个 rank 时在那里报错。一个组可以跨多台机器，每台机器持有“本机 rank 数 / N”的权重。

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
- DP 或 CFG 并行配 AllGather 时不能启用缓存加速。
- 缓存后端必须在 SP 组上同步跳块的决定。
- DP 大于 1 且文本编码器用 AllGather 时不能启用 prompt embedding 缓存。

## Offloader

### 策略

```text
offload 策略 = Unit + Trigger + Transport
```

每个被选中的组件有四个互相独立的设置：

| 设置 | 取值 | 含义 |
|---|---|---|
| `mode` | `module` / `layer` | 整个组件一个 Unit，或每个重复块一个 Unit |
| `weight_transfer` | `rank-local` / `allgather` | 本 rank 直接 H2D，或分片 H2D 后 AllGather |
| `resident_layers` | 非负整数 | 开头若干个重复块组成一个常驻层组 |
| `units` | 组件声明里 `unit_groups` 的名字 | 只有这些组按块环流式搬运，组件其余部分常驻加速卡 |

没被选中的组件常驻加速卡。不做 offload 也走同一个 Offloader 接口，所有 Unit 都常驻加速卡。

### Unit 规划

规划是纯计算，输入是策略、Loader 的组件声明、模型结构、并行拓扑和平台能力，输出是互不重叠的 Unit：

| Unit | 平时在哪 | Trigger |
|---|---|---|
| 整组件 | Host | 使用触发 |
| 流式的重复块，或 `units` 选中的组 | Host | 块环 |
| 常驻层组 | Host | 使用触发 |
| layer 组件里重复块以外的部分（非 `dit` 角色） | Host | 使用触发 |
| 其余部分 | 加速卡 | 无 |

- 每个参数和 buffer 只属于一个 Unit。共用同一段存储的张量属于同一个 Unit；跨组件共用且生命周期不兼容时，规划阶段报错。
- 叠加层张量不进任何 Unit，不经过 HWM，常驻加速卡。
- 组件根上不属于任何子模块的小参数、按配置算出的非持久 buffer 常驻加速卡。
- 被选中的整组件 Unit 或常驻层组，如果本 rank 上没有任何组件能让它让出，规划把它当作常驻，并写进启动日志。

**重复块发现**是规划里的纯函数，只看模块树，不读任何模型属性：

- 候选列表是 `nn.ModuleList`（含子类），带基础权重的元素至少 2 个，Python 类型相同，并且这个类型自己定义了 `forward`。
- 组件里所有候选元素的最大值记为 B；候选列表的最大元素不小于 B/4 就合格。元素大小按本 rank 的最终字节数计算。
- 从组件根按注册顺序往下走，遇到合格列表就记为一组，不再进入它的元素；属于另一个组件的子树不进入。
- 组件声明给了 `repeated_blocks` 时，用它取代这个组件的自动发现。选中按 `layer` 处理的组件找不到重复块时，规划阶段报错。

**layer 组件里重复块以外的部分**：`dit` 角色常驻加速卡；其他角色按组件的直接子模块分成使用触发的 Unit。

**`units`**：选中的 `unit_groups` 里每个匹配的子模块是一个 Unit，按块环流式搬运；这个组件的自动发现不再进行，其余部分常驻加速卡。

**`resident_layers` 按角色计数**：同一角色下被选中的 layer 组件按声明顺序、组件内的重复块按 `repeated_blocks` 的列表顺序（自动发现时按注册顺序）排成一列，取前 N 个。

### Trigger

**块环**。

- 块环按 Transport 分成两个：`rank-local` 的流式 Unit 和 `allgather` 的流式 Unit 各用一块缓冲区、一张后继表，互不读写。没有某一类 Unit 时不分配那一块。
- 每块缓冲区的大小是这一类里最大的两个流式 Unit 的占用之和。占用按装入后的实际布局计算：`allgather` 的 Unit 按补零后的 N × 分片大小，每个张量按类型对齐，整个 Unit 向上取整到 256 字节。
- 后继表：每个 Unit 的后继初始为组件内注册顺序的下一个。`rank-local` 表里最后一个接回第一个；`allgather` 表里最后一个没有后继。进入一个 Unit 时预取它的后继。预测错时，进入的 Unit 不在缓冲区里就同步装入，并把同一张表里上一个 Unit 的后继改成它。
- 一个流式 Unit 的前向还没结束又进入另一个流式 Unit 时，直接报错。
- `allgather` 的 Unit 不跨前向保留：每次前向的第一个块在进入时拼接，之后按后继表预取。`allgather` 的后继表只按 `allgather` Unit 的进入顺序更新，所以是否拼接只取决于各 rank 相同的进入顺序。每次拼接带上 Unit 编号，在加速卡上断言。
- 编译边界只包住块的计算，不包住钩子。

**钩子的位置**。块环的 Unit 在块的根上，以及块里每个直接持有张量的模块上装钩子，哪个先被调用就从哪个进入，每次调用都核对驻留；使用触发的 Unit 在每个直接持有张量的模块上装钩子。钩子通过 HookRegistry 替换 `forward` 来安装。

**使用触发的让出关系**。

- 使用 `dit` 组件时，其他被选中组件的整组件 Unit、常驻层组和使用触发的 Unit 让出。
- 使用编码器角色的组件时，`dit` 的这些 Unit 和被选中的 `vae` 让出；编码器之间不互相让出。
- 使用被选中的 `vae` 时，`dit` 的这些 Unit 和其他被选中的 `vae` 让出。
- 编码器不管是否被选中都会触发让出。没被选中的 `vae` 和 `other` 角色不触发。

**让出**只释放加速卡上的存储，不往 Host 回写。让出后，参数指向读到就报错的空占位。

**VAE**。`vae` 没被选中时常驻加速卡、不装钩子；被选中时由使用触发管理。

**提交后的守卫**。对被管理的模块调用 `.to(执行设备)` 什么都不做；改到别的设备或改类型时报错。守卫覆盖 `Module._apply` 的全部入口。被管理参数在服务期间始终报告执行设备。

### Transport

- `rank-local`：从 host 存储直接 H2D。存储所在的内存登记给加速卡驱动后直接异步拷贝；平台不支持登记、超出登记上限或登记失败时，经过两个固定大小的暂存区。
- `allgather`：把本 rank 的分片 H2D 到一对分片槽位，再拼成完整的块放进块环的缓冲区。
- 任何策略都不生成完整的锁页副本。

**内存登记**。输入是 `HostStorage.ranges(unit)` 给出的 `MemoryRange`。同一个映射里所有决定登记的 Unit，区间先向外对齐到页，再合并成互不重叠的区间逐个登记；一个合并区间登记失败时，落在它里面的 Unit 都改走暂存区。撤销登记按合并区间进行。共享存储做只读登记。平台层提供 `host_registration_support() -> (writable, read_only)`，CUDA 查询设备属性，其他平台返回不支持。登记在 `commit()` 里做：共享存储在封存并只读绑定之后，私有存储和分片在写入、页归还之后；登记后确认这段内存被认作锁页。上限按每个 worker 计算，超出时按流式块、常驻层组、整组件 Unit 的顺序登记，一个 Unit 要么整个登记，要么整个走暂存区。

**暂存区**（2 × 256 MiB）。本 worker 有 Host 驻留的 Unit 时，Host 容量准入总是计入它；计划里有走暂存区的 Unit 时由 `prepare_materialization()` 分配；`commit()` 里登记失败、暂存区还没分配时，在这时分配。

### 资源与运行期约束

加速卡存储分两类：

- **固定槽位**：块环缓冲区、分片槽位、叠加层槽位。`prepare_materialization()` 在写模型之前分配，到 `close()` 前地址不变。
- **整组件 Unit 和常驻层组**：装入时向缓存分配器申请，让出时释放，地址不做承诺。

`prepare_materialization()` 还拿齐内存登记的计划、需要时的暂存区、流、事件、集合通信的工作区，并做加速卡的容量准入：

```text
常驻部分 + 叠加层槽位 + 块环缓冲区 + 分片槽位
  + max(写入期间的瞬时占用, 服务期间同时驻留的使用触发 Unit)
  ≤ 可用显存
```

- 写入期间的瞬时占用：最大的单个适配器 Δ；在线量化时，一层量化前的数据。
- 服务期间同时驻留的使用触发 Unit：按让出关系，取两者中较大的：最大的一个 `dit` 组件的使用触发 Unit 之和；全部被选中的编码器组件的使用触发 Unit 之和，加上最大的一个被选中 `vae` 的使用触发 Unit 之和。使用触发 Unit 指整组件 Unit、常驻层组，以及 layer 组件里重复块以外的使用触发 Unit。

提交时叠加层槽位交给动态 LoRA 管理器，其余交给 `OffloadRuntime`。`OffloadRuntime` 提供前向使用的梯度模式：有块环 Unit 时是 `no_grad`，否则是 `inference_mode`。runner 不读 offload 策略；HSDP 下 runner 照旧用 `no_grad`。

一个 `OffloadRuntime` 同一时刻只允许一次前向，可以是一个请求批次；重入或并发的前向被拒绝。出异常时先等拷贝和集合通信做完再回到空闲。集合通信的顺序不再可信时，运行时标记为不可用。

### 关闭

顺序固定：

```text
停止接受新的执行
→ 等加速卡上的拷贝和集合通信做完
→ 撤掉钩子和对模型的引用
→ 释放加速卡槽位、事件、流
→ 撤销内存登记，释放暂存区
→ 参数改指向占位
→ 关闭 host 存储的本进程映射
```

`OffloadRuntime.close()` 是本地的、终止性的：不把权重恢复回 CPU，不发起集合通信。

### 睡眠模式

睡眠模式在服务暂停期间把加速卡上的权重全部腾出来；Offloader 在服务期间管理权重在加速卡上的短暂驻留。两者按显存的类别分工：

| 加速卡上的显存 | 睡眠时 | 唤醒时 |
|---|---|---|
| 常驻加速卡的权重和 buffer、叠加层槽位 | 由睡眠模式拷到 Host，释放加速卡显存 | 由睡眠模式拷回原地址 |
| 块环缓冲区、分片槽位 | 直接释放，不备份 | 重新映射到原地址，内容在下一次使用时装入 |
| 整组件 Unit 和常驻层组 | 进入睡眠前让出 | 下一次使用时装入 |
| host 存储 | 不受影响 | 不受影响 |

- 常驻的部分在睡眠模式的 `weights` 池；块环缓冲区和分片槽位在一个睡眠时只释放、不备份的池里。块环缓冲区、分片槽位和叠加层槽位在 `prepare_materialization()` 里、写模型之前分配；常驻加速卡的权重和 buffer 由 Loader 在写入和收尾时分配在 `weights` 池里。启动驱动本身不进入任何池。
- 唤醒后常驻部分和固定槽位的地址不变，编译结果和只读取全部常驻组件的 CUDA graph 继续有效。
- 进入睡眠前，`OffloadRuntime` 等拷贝和集合通信做完，让出所有整组件 Unit 和常驻层组，清空块环。唤醒后的第一次前向按正常流程装入。
- 睡眠期间的前向被拒绝。

### 模型侧

模型只提供结构和组件声明，驻留、设备、搬运全部由 Offloader 决定。现有模型里与 offload 有关的代码，按下表由通用机制取代：

| 现有代码 | 取代它的机制 |
|---|---|
| 按 offload 策略选构造设备或加载设备 | 只建结构 |
| 模型自己搬运组件、自己维护锁页副本、自己在使用前后装卸 | 组件声明加使用触发 |
| 按 offload 策略关掉快路径或改变结构 | 结构和字节只由模型配置和并行方式决定，不随 offload 策略变化 |
| 模型自带的权重 offload（MiniMax H3 的 AdaLN 投影） | 组件声明的 `unit_groups`，策略的 `units` 按名字选择 |
| 按 offload 组合拒绝 | 现有后端的限制随新栈消失；集合通信顺序、容量、内容身份由通用规则表达；没有正确性问题、只是还没验证的组合不报错，写进支持矩阵 |
| offload 专用声明（`_layerwise_offload_blocks_attrs`、`OffloadPlan`、常驻路径） | 组件声明、重复块发现、`repeated_blocks`、`unit_groups` |
| 模型自己的 CUDA graph 按名字读钩子决定能否捕获 | `fully_resident` |

不属于 offload 机制、但在新栈下要改的模型代码（构造期间读张量、加载后改绑参数存储、用参数的设备判断执行设备、以存储地址或设备为键的缓存、自己捕获 CUDA graph 或选择 cudagraph 编译模式、在 `.to()` 之外改参数类型）逐项列在共存期必改清单里（Job 1）。

依赖规则由静态测试保证：`models/`、`lora/`、`cache/` 不 import Offloader，也不按字符串名读 offload 钩子。共存期间，现有的 28 个模型文件和 `lora/manager.py` 对 `vllm_omni.diffusion.offloader` 的 import、SeaCache 按字符串名读钩子，列入冻结的白名单；删除 `legacy` 时清空。

### CUDA graph 与编译

- `fully_resident` 由 `commit()` 填写：组件的全部权重和 buffer 常驻加速卡，地址到 `close()` 前不变，并且组件不由 HSDP 管理。
- 模型自己捕获的 CUDA graph 只能读取 `fully_resident` 组件的权重，不是的走 eager。模型通过一个通用函数读这个属性：`unified` 下读声明，`legacy` 下执行模型现有的判断。Offloader 的钩子在 graph 捕获期间被调用时直接报错。
- 编译设置在 `commit()` 之后运行，编译目标从组件声明读取。`reduce-overhead`、`max-autotune`、`triton.cudagraphs` 只用于 `fully_resident` 的组件；对其他组件，通用的编译函数自己降成 `default` 并记日志。
- 以权重为键的运行期缓存用参数身份加写入代数做签名，不用存储地址和设备。

## 清理与恢复

- 权重只存在于内存里：共享内存，或各 worker 的私有内存。
- 共享内存由内核按持有者计数，最后一个持有者退出后立即释放。正常退出、`kill -9`、OOM 的结果相同。写入进程持有只读文件描述符和监听套接字直到 `close()`；等待者只保留映射。
- 写入进程所在的副本关闭或被剔除时，其他副本的映射继续有效；之后加入的 worker 重新选举。
- 一个 worker 退出时，它所在副本的全部 worker 一起终止，多进程执行器和 Ray 执行器都是这样。其他副本照常服务。
- 服务重启后，每个副本从 checkpoint 构建，部署标识重新生成。
- fork：内存文件和监听套接字带 close-on-exec；写入期间的可写映射和之后的只读映射建立后都调用 `madvise(MADV_DONTFORK)`；HWM 用 `os.register_at_fork` 在子进程里关闭继承来的内存文件描述符、监听套接字和已接受的等待者连接。

## 内存模型

记一个组件的最终权重大小为 `W`，AllGather 组的大小为 `N`。

本机后端，每台机器、每个部署：

```text
稳态   = 每个共享集合一份 W（与 worker 数、副本数、stage 数无关）
       + 每个使用私有内存的 worker 一份 W
       + 需要时的暂存区
冷启动 = 稳态 + checkpoint 的页缓存（内核可回收）
              + 写入进程的瞬时占用（变换中的单个张量；在线量化时是一层量化前的数据）
```

同一台机器上的不同部署各占各的，不合并。

分片后端，每个 rank：

```text
稳态                 = W / N（含等分时的补零）+ 需要时的暂存区
冷启动，瘦路径       = 稳态 + 释放阈值 + 最大的单个张量 + checkpoint 的页缓存（内核可回收）
冷启动，先全量后裁剪 = 稳态 + W（短暂）
```

每台机器的占用是本机各 rank 之和。

**容量准入**。每个副本在全组核对时做，在写入之前：

```text
本副本在一台机器上的新增占用
  = 本副本担任写入进程的共享集合，各一份 W
  + 本副本使用私有内存的 worker，各一份 W
  + 本副本在这台机器上的 rank 分片
  + 这些 rank 分片的冷启动额外占用（瘦路径：释放阈值 + 最大的单个张量；先全量后裁剪：一份 W）
  + 暂存区
  + 写入进程和使用私有内存的 worker 各自的瞬时占用

可用内存
  = 同一台机器上，各 rank 在 local_record() 里读到的值、各代表参选时读到的值中的最小值
  − 写入进程在别处、还没写完的共享集合尚未写入的字节
```

每次读到的值：从本进程所在的内存 cgroup 到可见的最上层，逐级计算 `memory.max − memory.current + inactive_file + active_file`，取最小值。文件页取 `memory.stat` 的 `inactive_file`、`active_file`，不用 `file`，后者含内存文件和 tmpfs 的页。各级都没有限额时为 MemAvailable。

写入进程在别处的共享集合不计 W。新增占用超过可用内存时，本副本所有 rank 同时报错。同一台机器上并发启动的副本各自做准入，各副本私有的部分并发时可能合计超出，这时在写入阶段因内存不足而失败。

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

自定义配置按角色或组件名选组件，每个组件各自设置 `mode`、`weight_transfer`、`resident_layers`、`units`。Python 参数和 YAML 里对应的字段是 `diffusion_offload_policy`：

```json
{
  "mode": "layer",
  "components": {
    "dit": {"weight_transfer": "allgather"},
    "text_encoder": {"mode": "module"},
    "vae": {"mode": "module"},
    "latent_upscaler": null
  }
}
```

- `components` 的键是角色或组件名。组件名的设置覆盖它所属角色的设置；值为 `null` 表示不选这个组件。
- 可选的角色是 `dit`、`text_encoder`、`image_encoder`、`audio_encoder`、`vae`。`other` 不可选。
- 顶层的 `mode` 是组件的默认粒度，组件可以覆盖。
- `weight_transfer` 默认是 `rank-local`。
- `resident_layers` 默认是 0，可用于任何按 `layer` 处理的组件，只能配 `rank-local`。
- `units` 只用于 `layer`，取组件声明里 `unit_groups` 的名字。MiniMax H3 的 AdaLN 投影写成 `"dit": {"mode": "layer", "units": "adaln"}`。
- 预设和自定义配置在同一个配置来源里互斥。
- 命令行、Python 参数、YAML 使用同一个解析器；多个来源之间，优先级高的整体覆盖。
- 预设里的角色在某个模型上不存在时忽略；自定义配置里显式选了不存在的角色、组件名或 `unit_groups` 名字时，规划阶段报错。
- 未知字段和类型错误报错。

预设和旧开关都不选 `vae`。要让 VAE 由使用触发管理，在自定义配置里选中它。`unified` 不做与硬件相关的隐式 offload：`none` 就是不 offload。

### host 策略

| 设置 | 参数 | 含义 |
|---|---|---|
| 锁页与内存登记 | `--pin-cpu-memory` / `--no-pin-cpu-memory`，字段 `pin_cpu_memory`，默认开启 | 开启时登记 host 存储或使用锁页的暂存区；关闭时不查询平台能力，不登记，暂存区也不锁页 |
| 登记上限 | `--diffusion-host-registration-limit-gib`，字段 `host_registration_limit_gib` | 每个 worker 向加速卡驱动登记的 host 存储上限，0 表示不设上限。只有启用 offload 且 `pin_cpu_memory` 开启时才可以大于 0 |

同机共享没有配置项，满足条件即启用。

### 动态 LoRA

| 设置 | 参数 | 含义 |
|---|---|---|
| rank 上限 | `--diffusion-max-lora-rank` / `max_lora_rank` | 给出时启用动态 LoRA，按它分配叠加层槽位 |
| CPU 缓存条数 | `max_cpu_loras` | 已有字段，含义不变 |

`unified` 下没有给出 `max_lora_rank` 时：`peft` 加 `--lora-path` 时槽位取这个适配器的 rank；否则动态 LoRA 不启用，`add_lora` 被拒绝并提示这个参数。`legacy` 下行为不变。

### 旧开关

旧开关由一个兼容层翻译成策略。`legacy` 下它们驱动现有实现，行为不变；`unified` 下按下表翻译：

| 旧开关 | 翻译 |
|---|---|
| `--enable-cpu-offload` | `module`，选中现有实现发现的全部 DiT 和编码器，编码器名单里属于 `vae` 角色的除外（JoyImage 的编码器名单含 `vae`） |
| `--enable-layerwise-offload` | `dit` 用 `layer`、`rank-local`；有重复块的 `text_encoder` 也用 `layer`、`rank-local` |
| `--enable-distributed-layerwise-offload` | `dit` 用 `layer`、`allgather`；有重复块的 `text_encoder` 用 `layer`、`rank-local` |
| `--dlo-no-use-allgather` | `dit` 用 `rank-local` |
| `--dlo-resident-layers` | 与 `--enable-distributed-layerwise-offload` 同时出现、并且现有实现为这个模型声明了常驻路径（MiniMax H3、Kandinsky6）时，翻译成 `dit` 的 `resident_layers`；其他模型忽略并记一次警告，与现有行为一致 |
| `--diffusion-offload-config` / `diffusion_offload_config` | `mode` 照搬；`components` 列表里的角色按 `mode` 选中；`layer_options` 里每个组件的 `weight_transfer`、`resident_layers` 照搬；`pin_memory` 翻成 `pin_cpu_memory` |

- 旧开关从不选 `vae`。翻译测试对每个 pipeline 断言：翻译后选中的组件等于现有实现发现的 DiT 和编码器，`vae` 角色的组件除外。
- MiniMax H3、Wan2.2-S2V、MAGI-2、JoyImage 的现有代码自己在 VAE 使用前后挪走 DiT 或常驻层；LingBot-Video 按请求参数 `offload_vae_during_denoise` 在去噪期间把 VAE 挪到 CPU。这五个模型在 `unified` 下不选 `vae` 时，显存峰值高于现有实现。翻译测试对这五个模型豁免并写明原因；它们的 recipe 使用选中 `vae` 的自定义配置；用旧开关启动这些模型时，启动日志提示改用自定义配置。`offload_vae_during_denoise` 在 `unified` 下忽略并记一次警告。

`unified` 下以下旧设置报错，并给出迁移提示：

| 旧设置 | 处理 |
|---|---|
| `--dlo-host-registration-limit-gib` 大于 0 | 参数解析报错，提示改用 `--diffusion-host-registration-limit-gib` |
| 旧配置里的 `pin_memory` 与 `pin_cpu_memory` 冲突 | 参数解析报错。`legacy` 下保持旧配置的 `pin_memory` 优先 |
| Host Weight Runtime 的参数（mode 不是 `disabled`，或给了 root） | 参数解析报错 |
| MiniMax H3 的 `minimax_h3_adaln_offload` 显式开启 | 报错，提示改用策略 `"units": "adaln"` |

新旧两种写法同时出现且冲突时报错，并给出迁移提示。

`legacy` 下，被 HSDP 管理的组件可以做整组件 offload 或 `rank-local` 逐层 offload。`unified` 下这两种组合在规划阶段报错，报错信息给出两个选择：保留 HSDP、不对这个组件做 offload；或者去掉 HSDP，给这个组件用 `layer` 配 `allgather`。

### 不支持的组合

`unified` 下，以下组合报错，并说明原因：

| 组合 | 报错阶段 |
|---|---|
| `module` 配 `allgather`；`resident_layers` 大于 0 配 `allgather` | 参数解析 |
| 自定义配置选择 `other` 角色 | 参数解析 |
| 旧的登记上限、Host Weight Runtime 参数、`pin_memory` 冲突 | 参数解析 |
| DP 或 CFG 并行大于 1 且使用 `allgather` 时启用缓存加速 | 参数解析 |
| 缓存后端不在 SP 组上同步跳块的决定（Cache-DiT 按不同步处理），且使用 `allgather` | 参数解析 |
| DP 大于 1 且文本编码器使用 `allgather` 时启用 prompt embedding 缓存 | 参数解析 |
| HSDP 预分片配 LoRA 或量化 | 参数解析 |
| 被 HSDP 管理的组件被选中做 offload | 规划 |
| 选中按 `layer` 处理的组件找不到重复块；`units` 的名字不存在 | 规划 |
| 分片后端下，重复块以外的组件单机放不下 | 规划 |
| AllGather 组里只剩一个 rank | 全组确认 |
| 有张量不属于任何组件 | 规划 |
| 融合适配器的键在已加载的组件上对不上；融合时 `lora_scale` 不等于 1；对已量化的 checkpoint 融合 | Loader 预检 |
| pipeline 还没有迁移到只建结构 | 构造 |

环境变量 `OMNI_DIFFUSION_PROMPT_EMBED_CACHE` 在构建配置时折进配置并冻结，worker 不读它，组合检查只在参数解析时做一次。

### 校验分层

| 阶段 | 校验内容 |
|---|---|
| 参数解析 | 类型、未知字段、互斥项、新旧冲突、不支持的组合 |
| 构造 | 只建结构、读取陷阱、构造后没有残留的 meta 张量 |
| Offloader 规划 | 组件完整、保持 fp32 的层不量化、选中的组件和 `unit_groups` 存在、重复块可发现、常驻层数合法、共用存储的生命周期一致 |
| Loader 预检 | 来源可读、内容身份、冷启动路径、适配器的键 |
| 全组确认第 1 点 | 参与集合通信的部分各 rank 一致、AllGather 组的成员、单机 Host 容量 |
| `prepare_materialization()` | 加速卡和 Host 的运行期资源准入、内存登记或暂存区的计划、多 rank 的全组确认 |

每一层都在自己那一步的第一次不可逆修改之前完成。

## 上线方式

1. 新栈作为一套独立的代码加入主干，与现有的三个后端、现有加载路径并存。总开关默认是 `legacy`。
2. 并存期间，`legacy` 的行为不变。`unified` 下，`offloader/config.py` 里供模型查询的函数一律报告“不做 offload”，模型里读这些函数的分支自动不生效。少数在 `unified` 下会弄坏被管理权重的模型代码列在共存期必改清单里（Job 1），按清单逐项改：每一项只改变 `unified` 下的行为，`legacy` 下的行为和已有测试不变；按总开关或构造上下文分出的 legacy 一支随 `legacy` 一起删除。提交后的守卫让漏掉的地方在第一次使用时报错，不会静默出错。
3. 上线门槛：现有每个有端到端测试或每晚例行测试的“模型 × offload 策略”组合，在 `unified` 下输出一致，Host 和加速卡的内存峰值不高于 `legacy`。
   - 对比基准：现有实现里结构或类型随 offload 变化的组合（Flux2-Klein TP 大于 1 配 offload、DreamZero 的 VAE 类型），以 `legacy` 不做 offload 的输出为基准。按空闲显存选择分块大小的 VAE，对比时固定分块。融合适配器的组合逐张量对比融合结果；`legacy` 下结果本来就不对的组合（蒸馏 LoRA 配在线量化、蒸馏 LoRA 配 TP 大于 1）以“完整加载、融合、再量化”的参考结果为基准。
   - 不在门槛之内、写进迁移说明的组合：被 HSDP 管理的组件做 offload；MiniMax H3 在 SM120 上不加参数的 AdaLN offload（`unified` 下由 recipe 写明 `units`）；Qwen-Image-2.1 配 `--enable-cpu-offload` 时解码的 CUDA graph（`unified` 下 DiT 按整组件 offload，解码走 eager：显存更低，解码更慢）；旧开关一节列出的五个模型用旧开关时的显存峰值；适配器的键在已加载的组件上对不上、或融合时 `lora_scale` 不等于 1 的部署（`unified` 在预检报错），以及 `legacy` 静默丢掉增量、`unified` 融合进去的适配器；同一台机器上的两个部署跑同一个模型时的 Host 占用（`legacy` 的分布式逐层后端配 `--dlo-no-use-allgather` 时直接映射 checkpoint，页缓存跨部署共用；`unified` 下不同部署各占一份）。
4. 门槛全部达到，并且满足下面三条之后，默认值翻转为 `unified`：
   - 主干里注册的每个 pipeline 都能在只建结构的上下文里构造，支持矩阵里没有 `unified` 不支持的 pipeline；
   - 每个迁移家族的逐张量对比测试通过，对比例外表里的每一条都已列入门槛例外；
   - legacy 一支的指纹清单没有未处理的变化。
5. 翻转后的一个版本保留 `legacy` 作为回退，共存期的规则照旧执行。主干外没有迁移的 pipeline（`register_diffusion_model` 注册的、`load_format="custom_pipeline"` 的）在这个版本里要显式选 `legacy`。
6. 再下一个版本删除 `legacy`：三个现有后端；diffusers 后端整条 pipeline 的 `from_pretrained` 加载和它对 diffusers 自带 offload 的调用；现有的非 HSDP 加载路径（包括其中对 Host Weight Runtime 的接入）；模型里的 offload 专用声明和生命周期代码；共存期两套构造代码里的 legacy 一支。删除之前，逐张量对比测试把 legacy 的结果存成逐张量的哈希；删除之后，改为与这份哈希对比。

### 共存期的构造代码

构造期读张量的 pipeline 迁移时，legacy 的构造代码不改。读张量的调用点用 `build_component` 分成两支：legacy 一支是原语句，unified 一支是新写的结构构造和来源声明。两支并存到第 6 步。

- 迁移的 PR 不修改已有测试文件，`legacy` 下已有测试全部通过。
- 每个迁移的家族有一个以 legacy 为基准的逐张量对比测试，比较张量名、形状、类型、步长、逐字节内容、持久和非持久 buffer、模块外的数据张量、共用关系、模块树的路径和类名。结构可以配置的家族用合成的小 checkpoint，在改动对应家族目录或 Loader 的 PR 上运行；结构固定的家族用真实 checkpoint，每晚运行。
- legacy 一支调用的读取和转换函数登记在一份清单里，按限定名记录源码指纹。清单用来发现 legacy 一支的改动，也是第 6 步的删除清单。

### 共存期 legacy 修复的同步

| 修复改动的位置 | 怎么发现 | 怎么做 |
|---|---|---|
| 共用代码 | 不需要 | 两个栈同时生效 |
| legacy 一支，改变最终张量、类型、布局或模块树 | 逐张量对比测试失败 | 同一个 PR 里改 unified 一支 |
| legacy 一支，不改变最终张量（报错、路径解析、重试、校验） | 指纹测试失败 | 同一个 PR 里改 unified 一支，或在清单里写明 unified 不需要的原因，再更新指纹 |
| 只服务于 legacy 放置和搬运的代码 | 不需要 | 不同步，unified 由通用机制取代 |

有意让 unified 与 legacy 不同的张量，在对比例外表里逐条写明原因，并列入门槛例外。

### 删除的内容

模型里被删除的内容：

- `SupportsComponentDiscovery` 和 `_dit_modules` / `_encoder_modules` / `_vae_modules`，它们的使用方已改读组件声明；
- `_layerwise_offload_blocks_attrs`、模型声明的 `OffloadPlan`、常驻路径声明；
- `SupportsModelCpuOffload`；
- MiniMax H3 的驻留控制器、分配器缓存、AdaLN 的 Host 暂存和 SM120 默认、VAE 两半和放大器的搬运；Cosmos3 的阶段上下文；MAGI-2 的分阶段组件和内存窗口；其他模型自己的搬运代码；
- 模型里按 offload 策略改变行为的分支，以及只为现有后端存在的拒绝；
- 模型加载后钩子里只为现有实现存在的部分（`validate_restored_host_weights` 等）；
- `_maybe_fuse_distilled_lora`、pipeline 上会融合权重的 `load_lora_weights`、worker 里的第二个蒸馏 LoRA 融合点，以及 LoRA 管理器里读 offload 策略的代码；
- 每个 `build_component` 调用的 legacy 一支，以及 `build_component` 本身；
- 指纹清单登记的 legacy 读取和转换函数中没有其他使用方的；
- HF 组件入口里加载带权重组件的 legacy 一支，以及 `create_transformers_model` 里的 `init_parameters`、`create_transformers_model_with_vllm_linears` 里的 `init_parameters_preserving_vllm_linear`；
- 只服务于 legacy 的 `load_weights` 主体；
- 规则表取代的运行期改类型代码：LingBot-Video 和 InternVLA-A1 的 `.to()` 覆写、π0.5 的 `_set_inference_dtype`、LTX2 vocoder 的 `.float()` 往返；LTX2 在 `eval()` 里物化相位适配器的代码；MiniMax H3 pipeline 对 FastH3、VDN 的流包装；SeaCache 读分布式逐层 offload 钩子的代码；
- 指纹清单和指纹测试。

删除前的保证：

- 每个注册的扩散 pipeline 按测试覆盖的并行拓扑和量化配置分别在 meta 上构造，断言：现有声明里的块全部被发现；发现的组与期望表逐项一致，期望表写明多出或缺少的原因；容器本身不成为 Unit。用了 `repeated_blocks` 的组件同时记录自动发现的结果和覆盖的原因。每个模型另做一次 `layer` 冒烟测试。
- 只靠 `_layerwise_offload_blocks_attrs` 回退被编译的块类先补进 `_repeated_blocks`：`FluxSingleTransformerBlock`、`VaceWanTransformerBlock`、`MotionerAttentionBlock`、`MiniMaxH3TokenRefinerBlock`、`BooguImageDoubleStreamTransformerBlock`，以及 SDXL 的 `SDXLDownBlock2D`、`SDXLCrossAttnDownBlock2D`、`SDXLUpBlock2D`、`SDXLCrossAttnUpBlock2D`。一个测试对比删除前后被编译的模块集合。
- MAGI-2 的内存窗口由 recipe 里的配置给出：`dit` 选 `module`、`rank-local`。

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

本 RFC 不修改 `vllm_omni/host_weight_runtime/` 包，新栈不使用它：同机共享由 HWM 的无名共享内存承担；锁页、内存登记和暂存区归 Offloader，与 #6414 划定的边界一致。现有加载路径里对它的接入随 `legacy` 一起删除（上线方式第 6 步）。

### #7204（配置）

本 RFC 的“配置”一节取代 #7204。

### 子 RFC

本 RFC 之下有五份子 RFC：启动事务、统一加载、host 存储、统一 Offloader、配置。

## 实现拆分

### Job 1：启动事务、总开关与共存期

- `plan` / `prepare_materialization` / `commit` / `close` 的接口，以“不做 offload”这条路径落地。
- 总开关；不支持的组合在参数解析阶段报错。
- `unified` 下，`offloader/config.py` 里供模型查询的函数一律报告“不做 offload”。
- 共存期必改清单（下表）。每一项和启用对应策略的 Job 一起改，只改变 `unified` 下的行为，`legacy` 下的行为和已有测试不变；按总开关或构造上下文分出的 legacy 一支随 `legacy` 在 Job 6 删除。
- 依赖规则的静态测试：`models/`、`lora/`、`cache/` 不 import Offloader，也不按字符串名读 offload 钩子（SeaCache 按字符串名读，import 检查抓不到）。现有的 28 个模型文件和 `lora/manager.py` 的 import、SeaCache 的按名读取列入冻结的白名单，Job 6 清空。

| 位置 | 在 `unified` 下的问题 | 改法 |
|---|---|---|
| MAGI-2 pipeline 的两处 offload 校验、分阶段组件、内存窗口 | 内存窗口会对被管理的 DiT 执行 `.to("cpu")` | 直接判断总开关 |
| MAGI-2 MoE 加载后把 `W_gate`、`W_up` 改绑成 `_owned_w13` 的视图 | 违反先分配后写入，让出时约 2/3 的 MoE 权重释放不掉 | 只在只建结构的上下文里注册合并的参数 `W13`，`weight_loader` 按 shard id 写进交错的两半；上下文外保留现有的 `W_gate`、`W_up`、`_owned_w13` 和 `_apply` 覆写 |
| MiniMax H3 transformer 的 SM120 AdaLN offload 自动默认；放大器的搬运 | 模型自己把权重放到 Host 或搬设备 | 直接判断总开关 |
| MiniMax H3 的 AdaLN sidecar | 在 `load_weights` 里按写入流校验和播种；同机等待者写进共享存储的张量是空操作，sidecar 被拒 | `unified` 一支：sidecar 由 Loader 按 `declare_weights` 登记的非持久 buffer 来源写入；`eval()` 只用写好的 buffer 给 AdaLN 缓存播种，设备用执行设备，校验用 Loader 的内容身份 |
| MiniMax H3 视频 VAE 按 `next(params)` 判断设备，类型不是 fp32 时调用 `self.to(float32)` | VAE 被管理后撞上提交后的守卫 | 规则表给出这个 VAE 运行时的类型，`unified` 下这个条件不成立；设备取执行设备 |
| MiniMax H3 多输出时的 `_offload_model_cpu_stage_output` | `unified` 下不触发，解码出的视频在卡上累积 | `unified` 下改为只看 `num_outputs_per_prompt`，`legacy` 下保持原条件 |
| 精确投影缓存、AdaLN 缓存的签名 | 签名含存储地址和设备，有 forward hook 时整个停用 | `unified` 下签名改用参数身份加写入代数；`legacy` 下签名和停用条件不变 |
| LingBot-Video 的 `offload_vae_during_denoise` | 对被管理的 VAE 调 `.to("cpu")` | 直接判断总开关 |
| SANA-WM 每个请求的 `_ensure_*` | `.to(device, dtype)` 撞上提交后的守卫；`self.vae` 为空时在请求期按需 `from_pretrained`，绕过 Loader | 直接判断总开关；`unified` 一支在构造函数里经 HF 组件入口建出 VAE |
| HunyuanImage3 `load_weights` 开头（调用 `AutoWeightsLoader` 之前）对零散模块的 `.to(device)` | 搬动被管理的模块 | 只在 `legacy` 下执行 |
| Kandinsky6 的 `self.device` | 取第一个参数的设备，只建结构时是 meta | `unified` 下取执行设备，`legacy` 下不变 |
| Qwen-Image-2.1 的解码 CUDA graph；SenseNova-U1 的 AR 解码 CUDA graph；AuK、OmniVoice 的 CUDA graph；Ming-Image、InternVLA-A1、DreamZero 的 cudagraph 编译模式 | 捕获会被换出的权重地址 | 经通用函数按 `fully_resident` 门控，编译模式改走通用的编译函数；`legacy` 下通用函数执行模型现有的判断 |
| Wan VAE 快路径 | `.to(memory_format)` 之后用 `tensor.data=` 恢复 | `unified` 下内存格式由规则表表达，Loader 按步长分配，不做恢复；`legacy` 下不变 |
| Cosmos3 混合精度的 NVFP4 查找表 | 按加载后权重的设备放置，只做一次 | `unified` 下按调用时权重的设备取，或按设备缓存；`legacy` 下不变 |
| LTX2 vocoder 在 mps、xpu、cpu 上运行时 `.float()` 往返 | 撞上提交后的守卫 | `unified` 下由规则表在这些平台上保持 fp32、经 stage 类型舍入，跳过往返；`legacy` 下不变 |
| LoRA 管理器提交后才包装 PEFT 层；worker 提交后融合蒸馏 LoRA | 提交后改结构、改基础权重 | `unified` 下包装挪到只建结构（2f），融合挪到 checkpoint 读取器（2c）；`legacy` 下不变 |
| diffusers 适配器的 `DiffusersAdapterPipeline` 在构造函数里调用 `_raise_unsupported_features`，`diffusion_offload_config` 不是 `None` 时报 `NotImplementedError` | 旧的 `--diffusion-offload-config` 在 `unified` 下由兼容层翻译成策略，这项拒绝让 diffusers 格式在 `unified` 下无法用 offload 启动 | 只在 `legacy` 下执行；`unified` 下 offload 由策略和 Offloader 承担（2f） |
| runner 在分布式逐层 offload 下把 `inference_mode` 换成 `no_grad` | `unified` 下没有分布式逐层后端 | `unified` 下这一项改读 `OffloadRuntime` 给出的梯度模式，runner 不读 offload 策略；HSDP 下照旧用 `no_grad`；`legacy` 下不变 |

### Job 2：Loader

- **2a 构造上下文与组件构造函数**（约 1～1.5 周，依赖 Job 1）
  - 只建结构的构造上下文；`build_component`、`declare_weights`。
  - HF 组件入口：`from_pretrained_with_prefetch` 在内部分发；构造函数里直接调用 `X.from_pretrained` 的地方改成 `hf_component`，上下文外与原调用完全相同。
  - 读取陷阱（构造期间拦截张量读取，写入和收尾期间只拦截文件读取；被吞掉的违例照样失败）、构造后检查、写入跟踪、收尾的存储绑定检查；上下文关闭后调用组件构造函数直接报错。
  - 组件声明，包括按实例生成、路径列表和嵌套划出、完整性检查、按 rank 声明。
  - 每个注册的 pipeline 在上下文里构造一次，结果写进支持矩阵。
  - 构造函数里直接调用 `X.from_pretrained` 的 pipeline（Flux、Flux2、ERNIE-Image、Helios、GLM-Image、HiDream-Image、LingBot-Video、Cosmos3 的 VAE 等）改成 `hf_component`，用到的类各配一张规则表。LingBot-Video 的 `.to()` 覆写让 `_keep_in_fp32_modules` 保持 fp32，`unified` 下这一效果由它的规则表表达，`legacy` 下不变。
- **2b 规则表和对比测试**（约 2～3 周，可按家族并行）
  - 每个 Hugging Face 类一张规则表，配一个与 `from_pretrained` 逐张量对比的测试。
  - 以 legacy 为基准的 pipeline 级逐张量对比框架，以及对比例外表。
  - legacy 一支的指纹清单和指纹测试。
- **2c 先分配、后写入**：适配器在 checkpoint 读取器里融合，预检按格式核对键；逐层的在线量化；`fuse` 固定适配器；固定的收尾顺序，不调用模型的加载后钩子；内容身份和最终清单。
- **2d 迁移构造期读张量的家族**（依赖 2a、2c 和 2b 的对比框架；MiniMax H3 约 1～2 周，其余每个 1～3 天，可以并行；粗估）
  - 家族：Wan2.2-S2V、AuK、Ming-Flash-Omni、Ming-Image、π0、π0.5、InternVLA-A1、Gr00t、OmniVoice、LTX2、SeedVR2、MAGI-2、DreamZero、Anima、Lance、MiniMax H3、Cosmos3 声音 tokenizer、NextStep-1.1、LongCat-Video-Avatar。
  - 每个家族要做：
    - 读张量的调用点改成 `build_component`：legacy 一支是原语句，unified 一支放在家族目录下的 `unified_build.py`；
    - 读张量的代码改成 `declare_weights` 登记的来源，加上规则表；
    - 预先算好的数据张量登记为非持久 buffer 的来源；
    - FastH3、VDN 登记为适配器来源；
    - legacy 一支登记进指纹清单；
    - 一个以 legacy 为基准的逐张量对比测试。
  - 完成前，这些 pipeline 在 `unified` 下启动报错。
- **2e 组件声明的带类型字段和查询接口**：`repeated_blocks`、`unit_groups`、`shard_group`、`fixed_adapters`；`torch.compile`、缓存加速、动态 LoRA、模型注册表、序列并行钩子和 VAE 快路径改读组件声明。
- **2f 叠加层包装；`load_format="diffusers"` 按组件声明加载**。叠加层包装覆盖动态 LoRA 和 `overlay` 固定适配器。diffusers 适配器在 `load_weights()` 里调用 `DiffusionPipeline.from_pretrained` 构造整条 pipeline；`unified` 下由 Loader 按 `model_index.json` 声明组件、用 `from_config` 只建结构并构造 diffusers pipeline 对象，不调用适配器的 `load_weights()`，也不经过 `build_component`；`legacy` 一支的 `load_weights()` 不改。

### Job 3：host 存储

- HWM 接口：`local_record`、`share_plan`、`elect`、`admit`、`allocate`；`HostStorage` 和 `MemoryRange`。
- 部署标识：`StageRuntime` 生成，写进 `od_config.deployment_id`（`init=False` 的内部字段）；`OmniMasterServer` 的注册回复带上它，headless 启动器写进 `od_config`；没有时 0 号 rank 生成副本内的标识。
- 本机后端：本机键；内存文件、封存；选举，双方用 `SO_PEERCRED` 核对 uid；写入进程的监听线程和连接校验；定稿时先只读重绑定、再撤掉可写映射、再封存；只读文件描述符加 `MAP_SHARED`、`PROT_READ` 绑定；fork 时的处理。
- 容量准入。
- 分片后端：等分布局、按组件的 AllGather 分组（含 `shard_group`，在全组确认第 1 点由全部 rank 按相同顺序创建）、瘦路径、先全量后裁剪。

### Job 4：新 Offloader

- Unit 规划；重复块发现、`repeated_blocks`、`unit_groups` 和策略的 `units`；每个注册的 pipeline 按测试覆盖的拓扑和量化配置做发现结果的期望表测试。
- 块环（按 Transport 分开的两块缓冲区和两张后继表）；钩子的位置；使用触发和让出关系；提交后的守卫；前向的梯度模式。
- 两种 Transport；在 `commit()` 里按 `MemoryRange` 合并区间做内存登记，登记上限和暂存区。
- 单次执行约束；关闭。
- `fully_resident`；编译设置在提交后运行，cudagraph 模式按 `fully_resident` 降级；以权重为键的缓存改用参数身份加写入代数。
- 叠加层槽位；睡眠模式的两个内存池。

### Job 5：配置与可观测性

- 策略解析（角色和组件名、`units`、`vae` 可选）、预设、旧开关的兼容层（包括旧的 `--diffusion-offload-config`）、分层校验。
- `--pin-cpu-memory` / `--no-pin-cpu-memory`、`--diffusion-host-registration-limit-gib`；旧设置的迁移报错。
- MiniMax H3、Wan2.2-S2V、MAGI-2、JoyImage、LingBot-Video 的 recipe 改成选中 `vae` 的自定义配置；翻译测试对它们豁免。
- MiniMax H3 在 SM120 上依赖 AdaLN offload 的 recipe 写明 `"dit": {"mode": "layer", "units": "adaln"}`；MAGI-2 的 recipe 写明 `dit` 用 `module`、`rank-local`。
- `--diffusion-max-lora-rank`。
- 指标。
- 文档、recipe、支持矩阵。

### Job 6：达标、翻转、删除

- 跑完门槛矩阵。
- 翻转前确认三件事：
  - 主干里注册的 pipeline 全部能在只建结构的上下文里构造；
  - 迁移家族的逐张量对比全部通过，对比例外都已列入门槛例外；
  - 指纹清单没有未处理的变化。
- 翻转默认值；保留一个版本的回退，这个版本里共存期的同步规则照旧执行。
- 删除前，把逐张量对比的 legacy 结果存成逐张量哈希，对比测试改为与哈希对比。
- 删除 `legacy` 和模型侧代码。其中构造代码：
  - 每个注册的 pipeline 都显式调用 `declare_components`，从 `_dit_modules` 等属性推出组件声明的默认推导随之删除；
  - 每个 `build_component` 调用只留 unified 一支，`build_component` 删除；
  - 指纹清单登记的 legacy 读取和转换函数，只删没有其他使用方的。OmniVoice 的 generator、decoder 的 `load_weights(model_dir, device)` 还被自回归路径使用，保留。`LanceWanVAE` 的读取同时被 Lance 和 MAGI-2 使用，两个家族都只剩 unified 一支后才删；
  - HF 组件入口里加载带权重组件的 legacy 一支，以及 `create_transformers_model` 里的 `init_parameters`、`create_transformers_model_with_vllm_linears` 里的 `init_parameters_preserving_vllm_linear`；
  - 只服务于 legacy 的 `load_weights` 主体；
  - 指纹清单和指纹测试。

### 依赖

```text
Job 1 ─┬─▶ Job 2 ─┐
       ├─▶ Job 3 ─┼─▶ Job 5 ─▶ Job 6
       └─▶ Job 4 ─┘
```

Job 2 内部，2d 依赖 2a、2c 和 2b 的对比框架。

## 验收标准

### 静态结构

- Loader 不 import Offloader 的实现；Offloader 不 import Loader 和 checkpoint 读取器。
- HWM 在没有加速卡、没有初始化分布式环境的情况下可以测试。
- 模型不读 offload 或 host 配置，不调用运行期的生命周期方法；`models/`、`lora/`、`cache/` 不按字符串名读 offload 钩子。

### Loader

- 所有组件最终的张量名、形状、类型、步长、buffer、共用关系和字节与完整加载的基线一致。
- 量化、TP 布局、共用存储、收尾，每一项只有一个负责的地方。
- `legacy` 下 `build_component` 只调用 legacy 一支；`unified` 下构造上下文里只调用 unified 一支，上下文关闭后调用直接报错。
- 写入入口对已绑定存储的模块调用 `load_state_dict` 不触发读取陷阱。
- 每个迁移家族在上下文里构造时不触发读取陷阱，构造后全部参数都在 meta 上。
- 每个迁移家族的逐张量对比与 legacy 一致：张量名、形状、类型、步长、字节，持久和非持久 buffer，共用关系，模块树的路径和类名。
- 未迁移的 pipeline 在 `unified` 下启动报错，报错写明读取位置，并提示改用 `legacy`。
- 模型代码吞掉读取陷阱的异常时，构造或收尾照样失败。
- 融合适配器后再在线量化的结果，在 TP 为 1 和大于 1 时都与先融合、后量化的参考一致。
- 在线量化时 Host 和加速卡的峰值不高于 `legacy` 的逐层量化。
- 加载、切换、卸载动态 LoRA 前后，模块结构和基础权重的地址不变。

### host 存储

- 同一个部署在一台机器上的 worker 数从 1 增加到 k，各 worker 按共享比例平摊后的内存占用之和不随 k 增长。
- 同一台机器上两个单卡副本属于同一个部署时共用一份；headless 副本并发启动时只有一个写入进程。
- 同一个部署里两个 stage 的 worker 在同一台机器上、整套权重的内容身份和规划指纹都相同时，共用一份。
- 同一台机器上两个部署跑同一个模型，彼此不共享、不连接；一个部署退出，另一个不受影响。
- 部署标识、共享指纹或 uid 不一致的连接被写入进程拒绝；套接字名被其他 uid 的进程占用时，等待者不绑定。
- 权重大于 `/dev/shm` 的大小时正常加载，`/dev/shm` 的用量不变。
- 定稿之后，任何进程对内存文件的写入和可写映射都被拒绝。
- 等待者绑定后，已绑定的字节和存储地址在 `eval()` 前后不变。
- 分片后端稳定后每个 rank 占 `W / N`；瘦路径冷启动的额外占用不超过声明的上限。
- 对任意一个 worker 执行 `kill -9` 后，它所在副本的全部 worker 退出，其他副本的映射和服务不受影响；对一个部署的全部 worker 执行 `kill -9` 后，机器的共享内存用量回到这个部署启动前。
- 写入进程在封存前被杀掉，等待者所在的副本启动失败，没有残留。
- 写入进程所在的副本被剔除后，其他副本照常服务；之后加入的副本重新选举、重新构建，并能正常服务。
- worker 之后 fork 出的 Python 子进程不持有内存文件和监听套接字。
- 重复块以外的组件单机放不下时，规划阶段报错并指出组件。

### Offloader

- 各种策略产生互不重叠的 Unit。
- `rank-local` 和 `allgather` 的输出与完整加载的基线一致。
- 块环的内存有固定上限，正确处理真实执行顺序、跳块和异常。
- 每个注册的 pipeline 在测试覆盖的拓扑和量化配置下，重复块发现的结果与期望表一致，现有声明里的块全部被发现。
- 只在部分 rank 上声明的 `rank-local` 流式 Unit 不打乱 `allgather` Unit 的拼接顺序。
- 每个支持的 pipeline 都有“全部组件做 offload”的冒烟测试，覆盖编译和异常路径。
- 整组件模式下没有反复的回写和反复的锁页分配。
- 提交后对被管理的模块改设备或改类型（包括 `.float()`、`.half()`）报错。
- 不是 `fully_resident` 的组件不进 CUDA graph；钩子在 graph 捕获期间被调用时报错。
- 关闭不恢复 CPU 权重，不发起集合通信；每个资源句柄只释放一次。
- 睡眠再唤醒后输出一致，常驻部分和固定槽位的加速卡地址不变，host 存储不变。

### 事务与故障

- 规划、准入、分配、写入、收尾、提交，每个失败点都有故障注入测试。
- 多 rank 下任何一个 rank 失败，本副本整组一起失败，不死锁；其他副本不受影响，只有失败的 rank 是别的副本某个共享集合在封存之前的写入进程时，那些等待者所在的副本启动失败。
- 几个副本互为对方某个共享集合的等待者时，启动不死锁。

### 上线

- `legacy` 下现有测试全部不变；共存期合入的 PR 不修改已有测试文件。
- 门槛矩阵里每个组合保留输出对比和内存峰值的原始数据。
- 默认值翻转前，构造期读张量的家族全部迁移完成。

## 可观测性

报告以下五项，分别报告，不相加：

1. 每个 worker 按共享比例平摊后的内存占用，以及整台机器之和。按进程 RSS 求和会把共享页算进每个映射它的进程，不能用来统计；cgroup 和 `Shmem` 统计只算一次。
2. host 存储的字节数，按私有、共享、rank 分片区分。
3. 已登记内存的字节数。
4. 暂存区的字节数。
5. 加速卡槽位的字节数。

启动日志写明规划结果：每个组件的 Unit 划分、使用的后端、是否共享及原因、冷启动路径；共享时本 worker 的角色、写入进程的 pid 和套接字名；每个 Unit 是登记还是走暂存区；每个组件的 `fully_resident` 和编译模式的降级。

## 反馈期

发布后一周。

## CC

Loader、Offloader、分布式运行时、模型接入、CUDA 与 NPU 平台、配置的负责人。
