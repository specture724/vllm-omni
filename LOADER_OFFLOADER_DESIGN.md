# Loader / Offloader / HWM 重构总体设计

> 范围：vllm-omni 的扩散加载代码（`vllm_omni/diffusion/model_loader/`、`vllm_omni/diffusion/offloader/`），以及下发部署标识要经过的编排代码（`vllm_omni/engine/stage_runtime.py`、`stage_engine_startup.py`、`stage_init_utils.py`，`vllm_omni/diffusion/data.py`）。
>
> 基线：upstream `main@726b21a5e`。关联：#6648、#6414、#7204。
>
> 配套文档：[设计摘要](./LOADER_OFFLOADER_DESIGN_SUMMARY.md)、[umbrella RFC](./rfcs/00-umbrella.md)。

## 0. 设计结论

1. **Loader 管权重内容**：负责模型结构、checkpoint 到参数的映射、各种变换、适配器融合，以及把最终字节写进参数。机制是通用的“先分配、后写入”，没有模型专属代码。
2. **HWM（Host Weight Manager）管 Host 上的存储**：一个接口，两个后端。本机后端让同一个部署里、同一台机器上内容相同的 worker 共享一份，不同部署之间不共享；分片后端让多台小内存机器共同持有一份。
3. **Offloader 管运行期搬运**：策略由 Unit、Trigger、Transport 三个互相独立的要素组成。锁页、内存登记、暂存区、加速卡上的槽位和流都归它。
4. **权重只存在于内存里**：每次启动都从 checkpoint 构建。
5. **内存由内核回收**：共享内存没有文件名，不占用 `/dev/shm`；最后一个持有它的进程退出后立即释放，正常退出、`kill -9`、OOM 的结果相同。
6. **一个 worker 退出时，它所在副本的全部 worker 一起终止**：其他副本照常服务，已绑定的共享内存不受影响。服务重启后，每个副本从 checkpoint 构建。
7. **新栈与现有实现并存，由一个总开关切换**：并存期间现有实现的行为和构造代码不变；达到上线门槛后翻转默认值，随后删除现有实现。
8. **模型里没有 offload 专用的代码**：模型只提供组件声明和一组封闭的、带类型的结构字段；驻留、设备和搬运全部由 Offloader 决定。
9. **动态 LoRA 是叠加层**：包装在只建结构时完成，LoRA 槽位在启动时分配并常驻加速卡；提交后切换适配器只拷数据，不改结构。
10. **睡眠模式与 Offloader 按显存类别分工**：块环缓冲区和分片槽位在睡眠时直接释放，常驻加速卡的权重由睡眠模式搬出搬回，host 存储不受影响。
11. **地址会变的权重不进 CUDA graph**：模型自己捕获的 CUDA graph 和编译的 cudagraph 模式只用于全部常驻加速卡的组件。

Loader、HWM、Offloader 连同把它们串起来的启动流程，下文统称新栈。

---

## 1. 需求与范围

### 1.1 需求

| 需求 | 设计要求 |
|---|---|
| 一台机器上多个 worker 用同一份权重 | 同一个部署在一台机器上只占一份 Host 内存；不同部署互不共享、互不牵连 |
| 多台小内存机器共同部署一个模型 | 每个 rank 只持有 1/N，冷启动的占用也有上限 |
| 加速卡显存不足 | 权重按需在 Host 和加速卡之间搬运 |
| 部署退出后不留内存垃圾 | 正常退出、`kill -9`、OOM 三种情况下内存都被回收 |
| 不破坏现有部署 | 总开关为 `legacy` 时现有行为完全不变 |
| 机制通用 | 没有针对单个模型的加载或 offload 代码 |

### 1.2 范围之外

- 自回归和语音 stage。它们使用 vLLM 上游的加载器。
- DP、SP、TP 等并行方式的计算语义和请求调度。
- 激活值和输出的搬运。
- 可变的基础权重。动态 LoRA 是不修改基础权重的叠加层。
- 睡眠 level 2 之后的唤醒。
- HSDP 的加载与运行。被 HSDP 管理的组件由 HSDP 自己加载、分片和拼接。
- Host Weight Runtime（`vllm_omni/host_weight_runtime/`）。这个包不改动，新栈不使用它。

---

## 2. 名词

| 名词 | 含义 |
|---|---|
| 部署 | 一个编排进程（`Omni`、`AsyncOmni` 里的 `StageRuntime`，分布式模式下是它的子类 `DistStageRuntime`）从启动到关闭的一次运行，包括它直接启动的全部副本，以及在 `DistStageRuntime` 的 `OmniMasterServer` 上注册的 headless 副本 |
| 部署标识 | 编排进程每次启动时生成的 128 位随机数，下发到这个部署的每个扩散 worker。不同部署的标识不同；同一套配置先后启动两次是两个部署，标识也不同 |
| stage | vllm-omni 里承担一个模型阶段的逻辑单元，由一个或多个副本组成 |
| 副本 | 一个 stage 的一份独立运行的实例，有自己的执行器、worker 进程和 torch.distributed 进程组。数量由 `num_replicas` 或 `--omni-dp-size-local` 决定，headless 副本可以在运行期加入 |
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
| 锁页 | 让一段 Host 内存不被换出，这样它才能被异步拷到加速卡 |
| 内存登记 | 把一段已有的 Host 内存交给加速卡驱动登记，效果等同锁页，不另外分配 |
| 暂存区 | 一小块固定大小的内存，数据先拷进来再 H2D |
| 文件描述符 | 进程手里指向一个已打开文件的编号，可以通过本机套接字交给别的进程 |
| 封存 | Linux 给内存文件加的不可撤销的限制，例如不许写、不许改大小 |
| 页缓存 | 内核为磁盘文件内容保留在内存里的缓存，内存紧张时被回收 |
| 启动事务（`LoadTransaction`） | 提交之前唯一持有所有临时资源的对象 |
| `OffloadRuntime` | 提交之后管理搬运、钩子和资源关闭的运行时对象 |
| 叠加层 | 叠加在基础权重之上、不修改基础权重的张量，即动态 LoRA 的 A/B 槽位 |

---

## 3. 模块与依赖

### 3.1 模块职责

| 模块 | 负责 | 不负责 |
|---|---|---|
| Runner（扩散 worker 里的模型运行器） | 构造下面三个模块，按顺序推进启动流程，关闭时调用 `close()` | 解析 checkpoint，规划 Unit，分配 host 存储 |
| Loader | 组件声明、只建结构、叠加层包装、checkpoint 映射、适配器融合、变换、写入、收尾、内容身份、最终清单 | 分配 host 存储，H2D，AllGather |
| HWM | 本机记录和共享计划；按规划结果分配 host 存储；同机共享；rank 分片 | 理解 checkpoint，锁页和内存登记，加速卡资源 |
| Offloader | Unit / Trigger / Transport 规划；锁页、内存登记、暂存区；加速卡槽位、流、事件；H2D 与 AllGather；钩子；提交后的守卫；关闭 | 读取 checkpoint，决定字节内容 |

### 3.2 依赖规则

1. Loader 只通过接口调用 HWM 和 Offloader，不 import 它们的具体实现。
2. Offloader 不 import Loader 和 safetensors 读取器，不调用模型的任何加载后钩子。
3. HWM 不解释 checkpoint 的键名，不做改名、切分、合并、量化或收尾。
4. 模型只向 Loader 提供组件声明和通用结构信息，不读 offload 配置或 host 配置，不 import Offloader，不按名字读 offload 钩子，不调用运行期的生命周期方法。
5. 启动状态不挂在 pipeline 对象上传递。

---

## 4. 启动流程

### 4.1 接口

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

- `Topology`：部署标识，以及按 rank 排列的 `LocalRecord`。副本内所有 rank 拿到同一份。`plan()` 的单机容量检查、`share_plan()`、`admit()` 都用它。
- `SharePlan`：本副本内每个共享集合在每台机器上的成员、代表（rank 最小者）和套接字名。
- `Election`：每个共享集合的写入进程在本副本还是在别处；在别处时，它是否已封存、还有多少字节没写入。还记录代表参选时读到的本机可用内存。本 rank 是写入进程时，`Election` 持有监听套接字和监听线程，`allocate()` 时移交给 `HostStorage`；在这之前失败时由 Runner 关闭。
- `ranges(unit)` 正好覆盖这个 Unit 在本进程里的字节，只在 `seal()` 之后（共享存储）或写入完成、页归还之后（私有存储、rank 分片）有效。

阶段、所有权的转移和失败时的行为是规范性的。

### 4.2 步骤

```text
1.  Runner 构造 Loader、HWM、Offloader
2.  Loader 只建结构：模型里所有带权重的组件都没有真实数据（5.2）
3.  拓扑收集（4.5 第 0 点）
4.  Offloader.plan()：纯计算，得出 Unit、Trigger、Transport，以及每个组件用哪种 host 存储
5.  Loader 预检：解析 checkpoint 和适配器的来源，算出内容身份和预期清单
6.  全组核对、选举、容量准入（4.5 第 1 点）
7.  HWM.allocate()：为需要 Host 驻留的张量分配存储
8.  Offloader.prepare_materialization()：在第一次写模型之前，
    拿齐所有占容量的运行期资源，返回启动事务（4.5 第 2 点）
9.  Loader 写入：把参数绑定到已分配的存储，各来源的写入入口（5.3）从 checkpoint
    读取器逐个张量写；适配器在读取器里融合，在线量化逐层进行
10. Loader 收尾：核对写入，host 存储定稿，共享存储交接和只读绑定，调用 eval()（5.4）
11. 启动事务 commit()：校验最终清单，内存登记，填写 fully_resident，装上钩子，
    资源整体移交 OffloadRuntime（4.5 第 3 点）
12. 编译设置（7.8）
13. 开始服务
```

### 4.3 状态

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

### 4.4 规则

- `plan()` 不分配大块资源，不改模型，不装钩子。
- 所有可以预知的错误（配置、拓扑、容量）在第一次写模型之前报出。
- 启动事务是提交前所有临时资源的唯一持有者。没提交就退出时，它自动释放本 worker 的资源。
- 提交成功后，全部资源一次性移交给 `OffloadRuntime`。同一个资源句柄只登记一次。
- 从开始写模型到提交成功，这个模型实例是可丢弃的；写入、收尾或提交失败时直接丢弃。
- 进入 `IN_SERVICE` 之后的失败是运行期故障：这个副本的全部 worker 终止。

### 4.5 多个 rank 的全组确认

全组确认走本副本已有的 torch.distributed 进程组，用 world 组的 gloo 组收集，只覆盖本副本的 rank。每一轮收集都带上本 rank 是否出错；任何一个 rank 出错，所有 rank 一起报错。副本之间不经过 torch.distributed，只通过 6.3 的本机套接字交接。

0. **拓扑收集**：`init_distributed_environment` 之后、`plan()` 之前。各 rank 交出 `HWM.local_record()`，0 号 rank 另外交出部署标识（6.3）。所有 rank 得到同一份 `Topology`。
1. **核对、选举、准入**：Loader 预检之后、`allocate()` 之前。
   - 各 rank 交出 Unit、内容身份、布局、Transport、规划的指纹。只核对参与集合通信的部分：AllGather 组里的 Unit、它们的内容身份和布局。`rank-local` 的 Unit 和只在本 rank 上声明的组件不参与核对（5.1）。
   - AllGather 组由收集到的各 rank 组件内容身份确定（6.4）。本副本全部 rank，包括没有声明这个组件的 rank，按组件名排序、以相同顺序创建这些进程组。组里只剩一个 rank 时在这里报错。
   - `share_plan()` 在每个 rank 上算出相同的共享集合和代表。
   - 代表调用 `elect()`，参选结果再收集一次。
   - `admit()` 在每个 rank 上得出相同的结论；超出容量时所有 rank 同时报错。这时还没有分配 host 存储，也没有锁页。
2. **`prepare_materialization()` 之后**：所有 rank 都就绪才返回启动事务。之后等待者连接写入进程。
3. **`commit()` 时**：所有 rank 都完成收尾和绑定，才一起激活。

任何一个 rank 在确认前失败，本副本整组一起失败。world size 为 1 时，以上各点都在本进程内完成。

---

## 5. Loader

### 5.1 组件声明

pipeline 的构造函数里，每个带权重的部分声明一次。声明按实例、在配置解析之后生成：同一个类在不同配置下可以有不同的组件，例如 LTX2 只在对应的 profile 下声明 `diffusion_decoder`、`latent_upsampler`。一条声明包含：

- 稳定的组件名和角色。角色是 `dit`、`text_encoder`、`image_encoder`、`audio_encoder`、`vae`、`other` 之一。
- 组件在 pipeline 里的路径。路径可以是一个子模块，也可以是几个子模块的列表；路径嵌套时，内层从外层划出。例如 Cosmos3 的 `text_encoder` 是 `transformer.language_model`，`dit` 是 `transformer` 去掉 `language_model`；MiniMax H3 视频 VAE 的编码半和解码半各是一个 `vae` 组件。
- 只用配置、不用权重数值的结构构造方法。
- 权重来源。
- checkpoint 里的形式和内存里的形式不一样时，一张规则表（见 5.2）。
- 下表里的带类型字段，按需给出。

**完整性**。规划时，pipeline 下的每个参数和持久 buffer 都必须恰好属于一个组件，否则规划阶段报错并列出路径。只想常驻、不参与 offload 的部分显式声明为 `other`。同一个模块挂在两个属性上（别名）时，按模块身份去重。

**按 rank 声明**。声明可以按 rank 不同，没有张量的组件视为不存在。例如 MAGI-2 只在输出 rank 上声明文本编码器、图像 VAE 和音频解码器，TurboVAE 视频解码器在 `vae_patch_parallel_size` 大于 1 时每个 rank 都声明；MiniMax H3 的文本编码器只在它自己的 TP 组里的 rank 上声明。

**带类型字段**。字段是一个封闭的集合，只描述模型是什么，不描述怎么 offload；是否 offload、用哪种方式，由策略决定（7.1）。每个字段都有一个通用的使用方，进入规划指纹，由期望表测试覆盖。新的需求要加新字段，经过设计评审，不开放任意键。

| 字段 | 含义 | 使用方 |
|---|---|---|
| `repeated_blocks` | 重复块容器的路径列表，取代这个组件的自动发现。列表顺序也是常驻层的计数顺序 | Offloader 规划（7.2）；`torch.compile` 选择编译区域 |
| `unit_groups` | 命名的、可以单独流式搬运的子模块组，路径可以用通配符 | 策略的 `units` 按名字选择（7.1） |
| `shard_group` | TP、EP 以外切分这个组件权重的进程组 | 内容身份（5.5）、AllGather 分组（6.4） |
| `fixed_adapters` | 模型自带的固定适配器：名字、来源、目标层、`apply`（`fuse` 或 `overlay`） | Loader（5.7） |

MiniMax H3 的声明里用到其中三类（`fixed_adapters` 的例子是 5.7 的 LTX2 相位适配器和 LongCat-Video-Avatar 的 DMD LoRA）：

```python
text_encoder = Component(
    role="text_encoder",
    path="text_encoder",
    repeated_blocks=["vision.blocks", "text_model.layers"],
    shard_group="minimax_h3_text_encoder",
)
transformer = Component(
    role="dit",
    path="transformer",
    repeated_blocks=["blocks", "token_refiner.blocks"],
    unit_groups={"adaln": ["blocks.*.adaln_proj.linear", "final_layer.adaln_proj.linear"]},
)
```

**查询**。组件声明在提交之后仍然可以查询。`torch.compile` 选择要编译的模块、缓存加速选择 DiT、动态 LoRA 选择目标层、模型注册表登记组件、序列并行钩子和 VAE 快路径选择目标模块，都从组件声明读取，不读模型上的 `_dit_modules`、`_encoder_modules`、`_vae_modules`。`commit()` 之后，每个组件还有一个只读属性 `fully_resident`（7.8）。

### 5.2 构造函数只建结构

`legacy` 下每个 pipeline 的构造函数照旧执行，行为不变。只建结构的构造上下文只由 `unified` 的 Loader 进入，本节其余的要求只对 `unified` 成立。

Loader 在这个上下文里调用 pipeline 的构造函数：

- 参数在 `register_parameter` 时转到 meta 设备上，与外层的 `torch.device` 上下文无关。参数的子类和 `weight_loader` 等属性保留。buffer 照常在 CPU 上创建。
- 构造期间不做设备移动：`Module.to()`、`.cuda()` 对参数和 buffer 只应用类型和内存格式。
- vLLM 风格的组件用配置构造函数。transformers 的类按 `create_transformers_model`（`vllm_omni/diffusion/models/utils.py`）的方式构造，上下文里跳过 `init_parameters`；`create_transformers_model_with_vllm_linears` 同样跳过 `init_parameters_preserving_vllm_linear`。diffusers 的类用 `from_config`。
- Hugging Face 组件经过 HF 组件入口（见下文“两个栈的构造代码”）：读配置，用 `from_config` 在 meta 上建结构，登记权重来源（路径、子目录、版本）和这个类的规则表。
- 读取陷阱从进入构造上下文一直开到收尾结束。构造期间拦截张量数据的读取并报错：`get_tensor` 和切片取值、`load_file`、`torch.load`、传入真实张量的 `load_state_dict`、`from_single_file`、非 Hugging Face 类自定义的 `from_pretrained`。写入和收尾期间只拦截文件读取：`get_tensor` 和切片取值、`load_file`、`torch.load`、`from_single_file`、非 Hugging Face 类自定义的 `from_pretrained`；写入入口对已绑定存储的模块调用 `load_state_dict` 属于写入，不拦截。只读 safetensors 头部是允许的，有的模型靠头部推出结构。Loader 自己的 checkpoint 读取器不受影响。陷阱抛出异常时，同时把违例记在上下文里：即使模型代码用 `try/except` 吞掉了异常，构造或收尾结束时照样失败。
- 构造结束后执行 `empty_cache`。构造期间的分配不进睡眠模式的 `weights` 池，只有 Loader 分配的常驻存储进池。

构造之后检查：所有参数都在 meta 上；除了经 `declare_weights` 登记了来源的非持久 buffer，buffer 和模块属性里没有 meta 张量。出现别的 meta 张量，说明构造期间用权重数值做了计算。

**类型在只建结构时就定下来**。每个张量的最终类型先取构造出的参数类型（构造函数显式给的，或 stage 的类型），再按规则表覆盖。Loader 在规划之前把最终类型落到 meta 参数上，保留 `weight_loader` 等属性。规划、HWM 布局和预期清单看到的都是最终类型。

**规则表**。`from_pretrained` 在加载时顺带做的事由每个类一张规则表表达。规则表是数据，由 Loader 统一消费，覆盖：

- 共用参数；
- 加载时忽略的键；
- transformers v5 的权重转换，包括改名、合并、堆叠；
- 类型解析；
- 保持 fp32 的条目。每个条目写明数值从哪来：默认把 checkpoint 值直接转成存储类型；标为“经 stage 类型舍入”的条目，先写成 stage 类型再升到 fp32，与现有实现逐位一致；
- 折叠 weight norm 等写入时的变换。条目可以标注“须在执行设备上做”，因为有的变换在 CPU 上的结果差一个 ULP，解码后被放大；
- 不在 checkpoint 里、但初值有意义的参数，由量化方法或规则表声明初始化方式。

规则表的约束：

- 每张表配一个测试，逐张量对比它和 `from_pretrained` 的结果，固定 transformers 和 diffusers 的版本。
- 运行时 Loader 只读入库的表，不读类上的 `_keep_in_fp32_modules`。这个属性是否生效随库的版本和类型变化。
- 保持 fp32 的层，量化方法必须是“不量化”，否则规划阶段报错。
- `config.json` 自带 `quantization_config` 的 Hugging Face 组件报错，`from_config` 不会套用 Hugging Face 的量化器。
- 会在 `__call__` 里改组件类型的 diffusers pipeline（例如 SDXL fp16 的 `upcast_vae()`），由规则表把这类组件固定为运行时的类型。

**构造期间改结构的其他情况**。构造期间删掉的子树（例如 Qwen-Image 文生图删掉视觉塔），Loader 按构造结束时的结构跳过对应的键。构造期间改布局或类型（Wan VAE 的 channels_last、MiniMax H3 VAE 解码器转 fp16）由规则表表达，Loader 按参数的类型和步长分配存储、写入时转换。

**HSDP**。被 HSDP 管理的组件同样只建结构。物化时只为 meta 参数分配 CPU 存储，不用 `to_empty`，因为它会连 buffer 一起重新分配，清掉构造期间按配置算出的 RoPE 表等。之后照常 `load_weights` 并分片，不经过 HWM。

**`load_format="diffusers"`** 按 `model_index.json` 声明组件：每个带权重的组件用 `from_config` 只建结构，由 Loader 写入，走同一条流程。diffusers 的 pipeline 对象只负责编排前向和去噪循环，不加载权重，也不调用 diffusers 自带的 offload。分词器、调度器和纯配置不经过 Loader。`unified` 不调用 diffusers 适配器的 `load_weights()`，也不经过 `build_component`；`legacy` 下适配器在 `load_weights()` 里整条 `from_pretrained` 的代码不改。

**`load_format="custom_pipeline"`** 与注册的 pipeline 规则相同。它的首次加载在 `re_init_pipeline` 里，`unified` 的整个启动流程（包括 4.5 的全组确认）都在其中运行。再次调用时，先 `close()` 并关闭本进程的 host 存储映射，再完整走一遍启动流程，包括 6.3 的选举，所有 rank 一起调用。

**加载设备**。`unified` 下没有“加载设备”：每个参数的存储由规划决定（5.3）。

**两个栈的构造代码**。组件构造函数是两个栈在构造期唯一的分发点：

```python
# vllm_omni/diffusion/model_loader/construct.py
def build_component(*, legacy: Callable[[], T], unified: Callable[[], T]) -> T: ...
def declare_weights(module: nn.Module, *, sources, rules=None, writer=None) -> None: ...
```

- `legacy` 下，`build_component` 只调用 `legacy()`。`unified` 下，构造上下文里只调用 `unified()`；Loader 关闭上下文之后再调用 `build_component`、`declare_weights` 或 HF 组件入口直接报错，写明调用位置。按需构造的组件在 `unified` 一支里于构造函数中建出。
- 分发和读取陷阱看同一个进程级状态。除共存期必改清单里的分支（umbrella RFC Job 1，随 `legacy` 删除）以外，模型不读总开关。
- `legacy` 是原来那几条语句原样移进的局部函数，参数、顺序和它调用的函数都不变。
- `unified` 只用配置建结构，返回与 `legacy` 同一个类的实例，模块树相同。它对建出的每个带权重的模块调用一次 `declare_weights`，登记权重来源、规则表和写入入口。在上下文外调用 `declare_weights` 直接报错。
- `unified` 调用 legacy 的路径解析、配置解析和校验函数，不复制它们；只有读张量、拼 state dict 的部分另写，放在家族目录下的 `unified_build.py`。需要时可以给原有的类加新方法，原有方法不改。

HF 组件有两个入口，内部走同一个分发：

- `from_pretrained_with_prefetch`（`vllm_omni/diffusion/model_loader/hub_prefetch.py`）：上下文外的代码不变。上下文里读配置，用 `from_config` 在 meta 上建结构，登记 HF 来源和这个类的规则表。类自己覆写了 `from_pretrained` 的（例如 `DistributedAutoencoderKLWan` 在里面调用 `init_distributed()`），这一步构造后处理写进规则表，在上下文里同样执行。
- `hf_component(factory, *args, **kwargs)`：用在构造函数里直接调用 `X.from_pretrained(...)` 的地方。上下文外就是 `factory(*args, **kwargs)`，不附加预取和重试；上下文里与上一条相同。
- 分词器、processor、调度器和纯配置直接通过。

权重来源有两种写法：

- `declare_weights` 登记的来源，由 Loader 直接写进那个模块：给了 `writer` 就调用它；模块有 `load_weights(weights)` 就调用它；否则按参数名写入，规则与 `AutoWeightsLoader` 相同。这些流不经过 pipeline 的 `load_weights`。
- pipeline 的 `weights_sources` 按前缀归到组件，流交给 pipeline 的 `load_weights`。pipeline 没有 `weights_sources` 时，`unified` 不调用它的 `load_weights`。
- 预先算好的数据张量（SeedVR2 的 `pos_emb`、MiniMax H3 的 AdaLN sidecar）也用 `declare_weights` 登记，作为非持久 buffer 的来源。构造期得到的是 meta 张量，形状来自文件头或配置。
- FastH3、VDN 登记为适配器来源，由 Loader 在 checkpoint 读取器里融合（5.7）。模型侧的适配器对象在 `unified` 一支只读文件头和元数据，校验方法与 legacy 相同，不参与写入。

构造函数里读张量的调用点以外的部分、`forward`、处理 `weights_sources` 的 `load_weights`、各模块类，两个栈共用一份。共用代码里 legacy 的放置和搬运代码在 `unified` 下不起作用：

- offload 的查询函数报告“不做 offload”；
- 构造期间的 `.to()` 只改类型和内存格式；
- 模型自己的分阶段和驻留代码按共存期必改清单处理（umbrella RFC Job 1）。

共用的 `load_weights` 在 `unified` 下不改结构，由收尾第 4 步的存储绑定检查兜底。

新增的 pipeline 按 Z-Image、Kandinsky6 的写法：带权重的组件只用配置构造，或者经 HF 组件入口构造；权重全部经 `weights_sources` 和 `load_weights` 写入。这种写法两个栈共用一份构造代码，不需要 `build_component`。共存期间不合入构造期读张量的新 pipeline。

未迁移的 pipeline 在 `unified` 下构造时会触发读取陷阱或构造后检查，启动报错，写明读取发生的位置，并提示改用 `--diffusion-weight-stack legacy`。是否已迁移不另设标记。CI 为每个注册的 pipeline 在上下文里构造一次（只需要配置文件），结果写进支持矩阵。

### 5.3 先分配、后写入

这是唯一的权重物化机制：

1. 写入之前，每个参数的存储已经按规划分配好：留在加速卡上的在加速卡，Host 驻留的在 HWM 分配的 host 存储里。
2. Loader 把参数绑定到这些存储上，然后把每个来源的流交给它的写入入口（5.2）：`weights_sources` 的流交给 pipeline 的 `load_weights`，`declare_weights` 登记的流交给对应的模块。流来自 Loader 的 checkpoint 读取器，写入入口逐个张量读出、变换、写入。TP 切分、合并参数、模型自己的映射都使用现有的 `weight_loader` 写法。适配器在 checkpoint 读取器里融合（5.7），所以 `load_weights` 看到的已经是融合后的张量。
3. Loader 包装每个参数的 `weight_loader`，按参数统计已写入的元素数。一层的全部权重写完时，Loader 立即对这一层做在线量化。量化在加速卡上进行，结果直接写进这一层的最终存储，量化前的数据随即释放。量化方法在写入之前声明每个输出张量的形状和类型，所以预期清单在写入之前就是最终的。
4. 每个最终张量只做一次有语义的写入。
5. 收尾过程中被整个替换掉的参数（例如会改参数的校准），在收尾时拷进 host 存储。

这个机制与模型无关。

### 5.4 收尾顺序

Loader 按固定顺序做一次：

1. 完成所有写入或绑定。
2. 做需要整个模型参与的、会改字节的工作，例如会改参数的校准。
3. 建立共用存储的关系（例如输入输出嵌入共用一份）。
4. 按写入跟踪核对：
   - 带 `is_checkpoint_required` 的参数都已写到，量化方法声明可以缺少的张量放过；
   - 从写入开始到这一步，除了第 2 步声明要替换的参数，每个参数和持久 buffer 都仍然绑定在写入前分配的存储上，类型和形状不变。量化层按量化方法声明的输出张量核对。

   这一步在每个 worker 上做，只核对本 worker 自己写入的张量；共享存储里的张量由写入进程核对。瘦路径下，本 rank 范围之外的张量仍按名字经过 `load_weights`，写入是空操作，但记为已写。
5. 把第 2 步里被替换的参数拷进 host 存储，生成最终清单，host 存储定稿。
6. 共享存储的交接和只读绑定（6.3）。
7. 在每个 worker（包括等待者）上调用 pipeline 的 `eval()`。这是模型侧唯一的收尾入口：它只能读权重，可以建立不属于权重的运行期状态，例如派生缓存；不能改参数和持久 buffer。分片后端 Unit 的权重在本 rank 上只有一段，`eval()` 不能读它们；需要这些权重的派生状态在第一次使用时建立。
8. 交给启动事务 `commit()`。

新栈不调用模型的任何加载后钩子：`post_load_weights`、`validate_loaded_weights`、`validate_restored_host_weights` 都不调用。模型在自己 `load_weights` 里做的检查照常运行。

第 5 步之后字节不再变化。提交之后，基础权重的字节、布局、共用关系都不可变，由 7.3 的提交后守卫保证。融合进基础权重的适配器或量化方式要改变时，关闭运行时，重新走一次启动流程。动态 LoRA 不改基础权重，不受此限。

同机共享时，等待者照常执行第 1 步的写入：写进共享存储的张量对它是空操作，其余张量（常驻加速卡的、私有内存的、rank 分片的）照常写，集合通信与写入进程的顺序相同。共享存储里的张量不做第 4、5 步，在第 6 步的 `seal()` 里接收只读文件描述符、核对并绑定。同机共享要求没有第 2 步的工作（6.3）。

### 5.5 内容身份与最终清单

内容身份在一个部署内部使用：判断同一台机器上的两个 worker 需要的最终权重是否完全相同（6.3），按组件确定 AllGather 组（6.4），以及多个 rank 在全组确认时核对彼此是否一致（4.5）。它按组件计算。

它由配置和本 worker 的坐标算出：

- 模型的标识和版本、组件边界、运行期表示（类型、量化方式）、Loader 规则表的版本；
- 会改变字节或形状的并行坐标：TP 的大小和编号；启用专家并行时加 EP 的大小和编号；组件声明了 `shard_group` 时加它的大小和编号；
- 融合的适配器和固定适配器的标识：路径或仓库、版本、格式。

它不包含 offload 策略、叠加层、不改变字节的 DP 编号、设备地址、文件路径和部署标识，也不比较 checkpoint 或适配器文件的字节内容。

最终清单由 Loader 在收尾第 5 步生成，是同机共享时按名字绑定的唯一依据。

### 5.6 叠加层：动态 LoRA

动态 LoRA 是叠加在基础权重之上的叠加层，不修改基础权重。它的结构在启动时定死，提交之后只拷数据。

**启用**。启动配置给出 `max_lora_rank` 时启用动态 LoRA，槽位按它分配。`lora_backend` 是 `peft` 并给了 `--lora-path`、但没给 `max_lora_rank` 时，槽位取这个适配器自己的 rank，运行期 `add_lora` 更大的 rank 被拒绝。都不给时，运行期的 `add_lora` 被拒绝。

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

### 5.7 适配器融合与固定适配器

**融合点是 Loader 的 checkpoint 读取器**，普通路径和 HSDP 路径的 `load_weights` 都从它读。对每个张量 `(name, W)`，它产出 `(name, W ⊕ Δ[name])`，Δ 按 checkpoint 张量名匹配。之后的改名、TP 切分、合并参数、HSDP 分片、`weight_loader` 里的逐层在线量化，看到的都是融合后的张量，所以“先融合、后量化”由数据流保证，TP 大于 1 时也正确。

**数值**。

- Δ 在加速卡上计算，计算类型由适配器格式规定：diffusers 格式用文件里的类型，kohya 格式先转成 fp32 再折叠 alpha。
- 默认把 W 和 Δ 都先转成 stage 类型再相加，与现有实现逐位一致。适配器格式可以规定别的规则，例如 FastH3 在 fp32 里相加、结果取 checkpoint 的类型。
- 融合不乘 `lora_scale`。
- Δ 瞬时多出一个张量大小的显存，计入准入。

**融不融合由适配器格式决定**。

- 只能融合的格式（MiniMax H3 的 FastH3、VDN）总是融合。它们改 RMSNorm 和 bias、注入额外的结构，做不成叠加层。来源由用户经 `--lora-path` 和模型配置给出，格式转换器负责认领。VDN 的 `linear_branch` 是额外的权重来源。
- 普通 A/B 格式在 `lora_backend=distill` 时融合，在 `peft` 时作为启动时激活的动态 LoRA（5.6）。

**固定适配器**是模型自带的、在组件声明的 `fixed_adapters` 里给出的适配器。

- `apply: fuse`：和用户给出的适配器一样在 checkpoint 读取器里融合。
- `apply: overlay`：只建结构时包装目标层；A/B 是基础内容，由 Loader 在写入阶段写入，TP 切分沿用现有的布局函数；A/B 属于目标层所在的 Unit，进入内容身份；运行期模型只负责开关。它不占动态 LoRA 的槽位。LTX2 两阶段蒸馏的相位适配器、LongCat-Video-Avatar 在离线 int8 底座上的 DMD LoRA 都用这种方式。

**预检**，在第一次写模型之前：

- 每种格式有一个转换器，把适配器的键映射成 checkpoint 张量名（Wan2.2 的适配器按位置对应两个 transformer），再读 checkpoint 头部核对。
- 属于已加载组件的键对不上时报错；属于没加载组件的键只警告。
- 融合时 `lora_scale` 不等于 1 报错。
- checkpoint 已经量化的组件不能融合，报错。
- HSDP 预分片配 LoRA 或量化，在参数解析阶段报错。

**瘦路径和同机共享**。只为本 rank 实际读取的张量计算 Δ；由写入进程融合，等待者直接绑定。

---

## 6. HWM

### 6.1 后端的选择

后端由规划决定：

| 组件的 Transport | 后端 | 这个 rank 在 Host 上持有 |
|---|---|---|
| 本 rank 直接 H2D | 本机后端 | 完整一份，同一部署里同一台机器上内容相同的 worker 共享 |
| 分片 H2D 后 AllGather | 分片后端 | 自己的 rank 分片 |
| 不做 offload，常驻加速卡 | 不经过 HWM | 无 |

`allocate()` 的 `layout` 来自 Offloader 的规划和 Loader 的预期清单：有哪些张量、各多大、属于哪个 Unit。`share_plan` 和 `elections` 来自 4.5 第 1 点：前者给出共享集合、代表和套接字名，后者说明本 worker 是写入进程、等待者，还是使用私有内存。

### 6.2 与并行方式的关系

| 并行方式 | 各 rank 的权重内容 | host 存储 |
|---|---|---|
| DP | 相同 | 本机后端下同机共享；分片后端下各持 1/N |
| SP | 相同 | 同 DP |
| CFG 并行 | 相同 | 同 DP |
| TP | 每个编号不同 | 内容身份包含 TP 的大小和编号；相同编号的 worker 之间共享 |
| 专家并行 | 每个编号不同 | 内容身份包含 EP 的大小和编号 |
| 组件声明的 `shard_group` | 每个编号不同 | 内容身份包含切分组的大小和编号 |
| HSDP | 由 HSDP 管理 | 不经过 HWM |

### 6.3 本机后端：同一个部署在一台机器上共享一份

**共享的条件**。同时满足以下条件的 worker 组成一个共享集合，共用一份共享内存，不论它们属于哪个 stage、哪个副本：

- 部署标识相同；
- 本机键相同，也就是在同一台机器的同一个容器里，并且是同一个系统用户；
- 整套权重的内容身份和规划指纹相同；
- 写入之前就能确定每个张量的最终形状和类型；
- 收尾第 2 步为空：没有需要整个模型参与、会改字节的收尾工作。

不满足条件的 worker 使用进程私有内存，各自加载。不同部署的部署标识不同，即使权重完全相同也不共享、不互相连接。

**部署标识**。

- 编排进程在构造 `StageRuntime` 时用 `secrets.token_hex(16)` 生成一次，运行期间不变，每次启动都重新生成。
- 本地副本：编排进程建好每个副本的 `OmniDiffusionConfig` 之后、启动这个副本的进程之前，写入 `od_config.deployment_id`。
- headless 副本：`OmniMasterServer` 在注册回复里带上部署标识，headless 启动器在启动副本进程之前把它写进 `od_config`。运行期才注册的副本拿到同一个值。
- `deployment_id` 是 `init=False` 的内部字段，不能从配置或命令行设置，不进内容身份。它随 `od_config` 到达每个 worker：子进程由 spawn 启动时序列化 `od_config`；Ray 执行器把它作为 `init_worker` 的参数；单进程执行器和 inline 时用的是同一个对象。
- `od_config` 不是由 `StageRuntime` 下发时（例如直接用 `OmniDiffusionConfig.from_kwargs` 构造配置的工具和测试），里面没有部署标识，0 号 rank 在 4.5 第 0 点随机生成一个只在本副本内有效的标识，共享只发生在本副本内。这时副本只有一个 worker 的话，使用进程私有内存。

**本机键**。每个 worker 在 `local_record()` 里读取五项：

| 项 | 读取方式 | 作用 |
|---|---|---|
| boot_id | `/proc/sys/kernel/random/boot_id` | 区分机器 |
| 网络命名空间 | `/proc/self/ns/net` 的设备号和 inode | 抽象套接字只在同一个网络命名空间里可见 |
| PID 命名空间 | `/proc/self/ns/pid` 的设备号和 inode | 只有相同时，`SO_PEERCRED` 拿到的 pid 才能核对 |
| 内存 cgroup | 本进程所在内存 cgroup 目录的设备号和 inode，不用路径字符串 | 共享内存记在写入进程的 cgroup 上，cgroup 相同才是同一个容器 |
| 有效 uid | `os.geteuid()` | 交接时校验对方 |

任何一项读不到时，这个 worker 不参与共享。挂载命名空间不在其中，文件描述符跨挂载命名空间也能用。用宿主网络的容器共用网络命名空间、彼此能看到抽象套接字，但内存 cgroup 和 PID 命名空间不同，算出的套接字名不同，不会交接。

**规划指纹**只由配置和并行布局决定，不依赖 `LocalRecord` 里的 pid 和本机可用内存，也不依赖每块卡当时的空闲显存。否则同一块卡上依次启动的副本会算出不同的规划，不再共享。

**内存文件**。共享的存储是没有文件名的内存文件，由 `memfd_create`（`MFD_CLOEXEC | MFD_ALLOW_SEALING`）创建：

- 它位于内核内部的内存文件系统，文件系统里看不到它，只能通过文件描述符访问。
- 它不占用 `/dev/shm`，大小只受机器内存和容器内存限额约束。它占用的内存记在写入进程所在的容器上，同一个共享集合的 worker 都在这个容器里。
- 大小在加载前按预期清单算出，一次定好；物理页在写入时才占用。
- 只要还有进程持有它的文件描述符或映射，它就存在；最后一个持有者退出，内核立即释放。
- 它的页不被内核丢弃，所以共享的权重必须在机器内存里放得下。

**格式**。内存文件的内部格式是 safetensors：

- 每个组件按数据类型分成若干内存文件，同一个文件里的张量类型相同，每个张量的起点都按它的类型对齐。
- 布局由张量的名字、形状、类型算出：按名字排序、连续摆放，头部补齐到 8 字节。同一个重复块的张量相邻。
- 最终清单就是 safetensors 的头部。

**套接字名**。共享指纹是本机键、内容身份、规划指纹和存储格式版本的 128 位摘要。套接字名是 `\0vllm-omni-hwm/<部署标识>/<共享指纹>`，位于 Linux 的抽象命名空间：不在文件系统里留文件，持有它的进程退出后名字自动消失。部署标识每次启动都重新生成，新实例算不出旧实例的套接字名，旧实例的写入进程还没退出时也不会连到它。

**选举**。

1. `share_plan()` 在副本内把属于同一个共享集合、又在同一台机器上的 rank 分成一组，rank 最小的是这组的代表。
2. 代表调用 `elect()`：创建一个 `AF_UNIX`、`SOCK_SEQPACKET`、带 `SOCK_CLOEXEC` 的套接字，绑定这个名字。绑定是原子的：
   - **绑定成功**：代表就是这个部署在这台机器上这个共享集合的写入进程，立即开始监听。本组其余 rank 是等待者。
   - **名字已被占用**：写入进程在别的副本里。代表连接它，用 `SO_PEERCRED` 核对对方的 uid 等于自己的有效 uid，再问到它是否已封存、还有多少字节没写入。本组全部 rank 都是等待者。uid 不一致时按“其他错误”处理。连接被拒绝时短暂重试；名字在重试时空出，就重新参选。
   - **其他错误**：本组不共享，使用私有内存。
3. 参选结果在副本的进程组里收集一次（4.5 第 1 点），每个 rank 由此知道自己的角色。启动日志写明角色、写入进程的 pid 和套接字名。

**交接**。

1. 写入进程在 `allocate()` 里创建内存文件，以可写方式映射，并对可写映射调用 `madvise(MADV_DONTFORK)`。
2. 写入进程的监听线程从选举成功一直运行到它的 `HostStorage.close()`。每个连接先校验，没通过的直接关闭、不计数：`SO_PEERCRED` 拿到的 uid 等于自己的有效 uid；对方发来的协议版本、部署标识、共享指纹、预期清单摘要都与自己的一致。
3. 4.5 第 2 点之后，其余等待者连接写入进程，代表沿用选举时的连接。写入进程还没封存时，连接一直保持到封存；已经封存时，立即交出文件描述符。
4. **写入**：所有 rank 执行同一遍写入，张量顺序和集合通信与写入进程相同。写进共享存储的张量对等待者是空操作。常驻加速卡的部分、不经过 HWM 的组件，各自照常写。
5. **定稿**：写入进程在 `seal()` 里经 `/proc/self/fd/N` 以只读方式重新打开内存文件，用这个只读文件描述符建 `MAP_SHARED`、`PROT_READ` 映射，由 HWM 解析 safetensors 头部，按最终清单把全部张量（包括共用存储的别名）重新绑定到只读映射上，然后撤掉可写映射；还有张量引用可写映射时启动失败。之后给内存文件加上封存：不许写、不许加长、不许缩短、不许再改封存。加封存失败时启动失败。此后任何进程都不能修改它。
6. **交出**：写入进程用 `socket.send_fds` 把只读的文件描述符和最终清单摘要发给每个已连接的等待者。之后才连上来的等待者立即拿到。一条消息放不下全部文件描述符时分多条发。
7. **绑定**：等待者用 `SO_PEERCRED` 和 `fstat` 核对对方和文件的 uid 等于自己的有效 uid，再核对封存位、文件大小和清单摘要，然后用只读文件描述符建 `MAP_SHARED`、`PROT_READ` 映射，按名字逐个绑定，与写入进程第 5 步走同一段代码。
8. 等待者映射之后关闭文件描述符。写入进程保留只读文件描述符和监听套接字，直到 `close()`。

**只读映射**。这种映射不是写时复制映射：长期只读的内存登记直接锁住共享页，不产生副本；任何对共享权重的写入都会触发段错误。所有映射都用只读打开的文件描述符建立：在 6.7 之前的内核上，对加了 `F_SEAL_WRITE` 的内存文件，用可写打开的描述符建 `MAP_SHARED` 映射，即使只读也会被拒绝。

**顺序**。`seal()` 之前，本副本收尾里的集合通信全部完成，`seal()` 里不做集合通信。写入进程封存并交出之前不等待任何套接字；等待者到 `seal()` 里才阻塞接收。因此副本之间的等待不会和副本内的集合通信互相卡住；几个副本互为对方某个共享集合的等待者时，也不会死锁。

**后来的 worker**。后启动的副本和运行期才注册的副本照常选举。写入进程还在时，绑定名字会失败，连接后立即拿到已封存的存储。这个副本照常执行写入阶段（共享张量是空操作），容量准入不计这一份。写入进程已经退出时，名字随它一起消失，后来的 worker 绑定成功，成为新的写入进程，重新构建一份；已经绑定旧存储的 worker 不受影响。

**写入进程的瞬时占用**。在线量化在加速卡上逐层进行，写入进程只短暂持有一层量化前的数据。

**失败**。

| 情况 | 结果 |
|---|---|
| 写入进程在封存之前退出或报错 | 已连接的等待者在连接上收到结束信号，所在副本启动失败。报错给出写入进程的 pid 和它所在的 stage |
| 写入进程在封存之后退出 | 已绑定的 worker 不受影响。之后加入的 worker 重新选举 |
| 写入进程在封存之后、本组等待者都拿到文件描述符之前退出 | 已经收到文件描述符的等待者照常绑定；还没收到的等待者所在副本启动失败，不在启动中途改变角色 |
| 等待者失败 | 写入进程只关闭这个连接，继续构建和监听。副本内的失败由 4.5 和执行器的进程监控处理 |
| 写入进程停住不动 | 等待者一直等到 stage 的启动超时 |

本地扩散副本依次启动，第一个副本的代表是写入进程，它在封存前失败时整个部署启动失败。只有 head 的本地副本和同机的 headless 副本、或两个 headless 启动器并发启动时，一个启动单元里写入进程在封存前的失败才会让另一个启动单元启动失败。

**执行器与部署形态**。

- 单进程执行器：worker 自己就是代表。inline 时监听线程在编排进程里；私有内存、共享映射、监听套接字和文件描述符、内存登记、加速卡资源都由 `close()` 释放，不靠进程退出。
- 多进程执行器：一个副本的 worker 都在本机，构成一组。
- Ray：一个副本可以跨机器，每台机器上的 rank 各成一组、各自参选。不同副本的 actor 落在同一台机器上时照常共享。
- headless：与本地副本相同，部署标识来自注册回复。
- 自定义执行器：只要构造 `DiffusionWorker` 就自动适用。

### 6.4 分片后端：多台小内存机器共同持有一份

**布局**。每个重复块的全部字节按类型分组拉平，按 rank 等分成 N 段，不足的部分补零；rank r 持有第 r 段。运行期每个 rank 把自己那段 H2D，再用 `all_gather_into_tensor` 拼成完整的块。

**分组**。AllGather 组按组件确定：一个 stage 副本里这个组件内容身份相同的全部 rank。TP 编号相同的 DP、SP、CFG 并行 rank 都在同一组，TP 为 1、没有 `shard_group` 时就是整个副本。组件声明了 `shard_group` 时，组内的 rank 还要在切分组里编号相同。组的成员和创建方式见 4.5 第 1 点；组里只剩一个 rank 时在那里报错。一个组可以跨多台机器，每台机器持有“本机 rank 数 / N”的权重。内存大的机器放更多的 rank。

**存储**。每个 rank 的分片是它自己的私有内存，用匿名 mmap 分配。

**冷启动**。每个 rank 稳定后只占自己的分片。构建过程有两条路径，由规划选定并写入启动日志：

| 路径 | 条件 | 构建过程中每个 rank 的额外占用 |
|---|---|---|
| 瘦路径 | 写入之前就能确定每个张量的最终形状和类型 | 有上限：一个释放阈值（一个重复块的大小）加上最大的单个张量 |
| 先全量、后裁剪 | 最终形状或类型在收尾时才确定，例如会改形状的校准 | 完整一份，短暂持有 |

瘦路径的做法：

- 每个 rank 为整个组件预留一段地址空间，物理内存在写入时才占用。参数绑定到这段地址空间上，`load_weights` 照常写。
- 完全落在本 rank 范围之外的张量不分配存储，写入对它们是空操作，也不读 checkpoint 里对应的数据。
- 跨越范围边界的张量先完整写入，再把范围之外的页归还给内核。归还按已写字节数的阈值触发。
- 需要变换的边界张量由相邻的两个 rank 各处理一次。

小内存多机部署使用瘦路径。

**分摊的范围**。跨 rank 分摊的是重复块：DiT 的块，以及各 rank 内容相同的文本编码器的块。DiT 里重复块以外的部分、VAE、图像编码器在每个 rank 上都是完整一份，必须单机放得下；放不下时规划阶段报错，并指出是哪个组件。

**约束**。AllGather 要求同组所有 rank 以相同顺序执行相同的块：

- DP 下同一波请求有相同的去噪步数和影响执行路径的参数。
- CFG 并行下各分支执行相同的块。
- DP 或 CFG 并行配 AllGather 时不能启用缓存加速。
- 缓存后端必须在 SP 组上同步跳块的决定。
- DP 大于 1 且文本编码器用 AllGather 时不能启用 prompt embedding 缓存。

### 6.5 生命周期与清理

- 权重只存在于内存里：共享内存，或各 worker 的私有内存。
- 共享内存由内核按持有者计数。它在最后一个持有它的进程真正退出时释放，正常退出、`kill -9`、OOM 的结果相同。
- 分片后端的分片是进程私有内存，进程退出即释放。
- 共享内存的持有者是映射它的进程，以及持有它文件描述符的进程。写入进程在整个生命周期里持有只读文件描述符和监听套接字，直到 `HostStorage.close()`；等待者映射之后关闭文件描述符，只保留映射。
- 写入进程所在的副本关闭或被剔除时，其他副本的映射继续有效；之后加入的 worker 重新选举（6.3）。
- `HostStorage.close()` 只释放本进程的映射；写入进程同时关闭监听套接字和文件描述符。不影响其他进程已有的映射。
- 单进程执行器把 worker 放在编排进程里时，私有内存、共享映射、监听套接字、内存登记、加速卡资源都由 `close()` 释放，不靠进程退出。
- 一个 worker 退出时，它所在副本的全部 worker 一起终止，多进程执行器和 Ray 执行器都是这样。其他副本照常服务。
- 服务重启后，每个副本从 checkpoint 构建，部署标识重新生成。
- fork：内存文件用 `MFD_CLOEXEC` 创建，监听套接字带 `SOCK_CLOEXEC`；写入期间的可写映射和之后的只读映射建立后都调用 `madvise(MADV_DONTFORK)`；HWM 用 `os.register_at_fork(after_in_child=...)` 在子进程里关闭继承来的内存文件描述符、监听套接字和已接受的等待者连接。worker 之后产生的 Python 子进程因此不会延长共享内存的寿命，也不会占住套接字名。

---

## 7. Offloader

### 7.1 策略的要素

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

### 7.2 Unit 规划

规划是纯计算，输入是策略、Loader 的组件声明、模型结构、并行拓扑和平台能力，输出是互不重叠的 Unit：

| Unit | 平时在哪 | Trigger |
|---|---|---|
| 整组件 | Host | 使用触发 |
| 流式的重复块，或 `units` 选中的组 | Host | 块环 |
| 常驻层组 | Host | 使用触发 |
| layer 组件里重复块以外的部分（非 `dit` 角色） | Host | 使用触发 |
| 其余部分 | 加速卡 | 无 |

- 每个参数和 buffer 只属于一个 Unit。
- 共用同一段存储的张量属于同一个 Unit。跨组件共用且生命周期不兼容时，规划阶段报错。重复块里的张量和块外的张量共用存储时，规划阶段报错。
- 叠加层张量不进任何 Unit，不经过 HWM，常驻加速卡。
- 组件根上不属于任何子模块的小参数常驻加速卡；按配置算出的非持久 buffer（RoPE 表等）常驻，不是钩子的位置。
- 被选中的整组件 Unit 或常驻层组，如果本 rank 上没有任何组件能让它让出，规划把它当作常驻，不分配 host 存储，并写进启动日志。这个判断按本 rank 的组件集合做。

**重复块发现**是规划里的纯函数，只看模块树，不读任何模型属性：

- **候选列表**：`nn.ModuleList`（含子类），带基础权重的元素至少 2 个，Python 类型相同，并且这个类型自己定义了 `forward`。不带权重的元素不参与判断。`ModuleDict`、`ModuleList`、`ParameterList` 这类容器本身不能成为 Unit：容器自己从不经过 `__call__`，装在容器上的钩子永远不会触发。
- **合格**：组件里所有候选元素的最大值记为 B，属于另一个组件的子树不计入。候选列表的最大元素不小于 B/4 就合格。元素大小按本 rank 的最终字节数计算，已计入 TP 切分和量化，与块环、HWM 布局用同一个估算。
- **最外层**：从组件根按注册顺序往下走，遇到合格列表就记为一组，不再进入它的元素。其他模块继续往下，包括自定义的可迭代模块。属于另一个组件的子树不进入。
- 每组里每个带权重的元素是一个 Unit。
- 组件声明给了 `repeated_blocks` 时，用它取代这个组件的自动发现。
- 选中按 `layer` 处理的组件找不到重复块时，规划阶段报错，提示改用 `module` 或不选这个组件。

**layer 组件里重复块以外的部分**：

- `dit` 角色：常驻加速卡。
- 其他角色：按组件的直接子模块分成使用触发的 Unit，共用存储的张量并入同一个 Unit。

**`units`**：选中的 `unit_groups` 里每个匹配的子模块是一个 Unit，按块环流式搬运；这个组件的自动发现不再进行，其余部分常驻加速卡。模型命中缓存时不调用这些子模块，它们也就不装入。

**`resident_layers` 按角色计数**：同一角色下被选中的 layer 组件按声明顺序、组件内的重复块按 `repeated_blocks` 的列表顺序（自动发现时按注册顺序）排成一列，取前 N 个，每个组件里取到的部分组成一个常驻层组。N 不小于总块数时全部常驻并记日志。

### 7.3 Trigger

**块环**。

- 块环按 Transport 分成两个：`rank-local` 的流式 Unit 和 `allgather` 的流式 Unit 各用一块缓冲区、一张后继表，互不读写。没有某一类 Unit 时不分配那一块。
- 每块缓冲区的大小是这一类里最大的两个流式 Unit 的占用之和。占用按装入后的实际布局计算：`allgather` 的 Unit 按补零后的 N × 分片大小，每个张量按类型对齐，整个 Unit 向上取整到 256 字节。相邻的两个 Unit 一个放在开头，一个放在末尾。
- 后继表：每个 Unit 的后继初始为组件内注册顺序的下一个。`rank-local` 表里最后一个接回第一个；`allgather` 表里最后一个没有后继。进入一个 Unit 时预取它的后继。
- 预测错时（缓存跳块、切换 DiT 等），进入的 Unit 不在缓冲区里就同步装入，并把同一张表里上一个 Unit 的后继改成它。
- 一个流式 Unit 的前向还没结束又进入另一个流式 Unit 时，直接报错。
- `allgather` 的 Unit 不跨前向保留：每次前向的第一个块在进入时拼接，之后按后继表预取。`allgather` 的后继表只按 `allgather` Unit 的进入顺序更新，所以是否拼接只取决于各 rank 相同的进入顺序。每次拼接带上 Unit 编号，在加速卡上断言。
- 编译边界只包住块的计算，不包住钩子。

**钩子的位置**。

- 块环的 Unit：钩子装在块的根上，以及块里每个直接持有张量的模块上。哪个先被调用，就从哪个进入这个 Unit；每次调用都核对驻留。这覆盖了模型在块的 `forward` 里绕过子模块直接读 `.weight`，以及在块外直接调用子层的写法。
- 使用触发的 Unit：钩子装在 Unit 里每个直接持有张量的模块上，第一次被调用时装入。
- 钩子通过 HookRegistry 替换 `forward` 来安装，不用 `register_forward_hook`。

**使用触发的让出关系**。

- 使用 `dit` 组件时，其他被选中组件的整组件 Unit、常驻层组和使用触发的 Unit 让出。
- 使用编码器角色的组件时，`dit` 的这些 Unit 和被选中的 `vae` 让出；编码器之间不互相让出。
- 使用被选中的 `vae` 时，`dit` 的这些 Unit 和其他被选中的 `vae` 让出。
- 编码器不管是否被选中都会触发让出。没被选中的 `vae` 和 `other` 角色不触发。

**让出**。

- 让出只释放加速卡上的存储，不往 Host 回写。释放后执行 `empty_cache`。
- 让出后，参数指向加速卡上的空占位。占位是读到就报错的形式（零元素张量），模型代码绕过所有钩子直接读一个已让出的参数时直接报错。

**VAE**。`vae` 没被选中时常驻加速卡、不装钩子；被选中时由使用触发管理。

**提交后的守卫**。

- 对被管理的模块调用 `.to(执行设备)` 什么都不做；改到别的设备或改类型时报错。
- 守卫覆盖 `Module._apply` 的全部入口：`.float()`、`.half()`、`.bfloat16()`、`.type()` 与 `.to()` 一样报错。
- 被管理参数在服务期间始终报告执行设备。用 `next(params).device` 判断设备的模型代码看到的仍是执行设备。

### 7.4 Transport

- `rank-local`：从 host 存储直接 H2D。存储所在的内存登记给加速卡驱动后直接异步拷贝；平台不支持登记、超出登记上限或登记失败时，经过两个固定大小的暂存区。
- `allgather`：把本 rank 的分片 H2D 到一对分片槽位，再拼成完整的块放进块环的缓冲区。
- 任何策略都不生成完整的锁页副本。

**内存登记**。

- 输入是 `HostStorage.ranges(unit)` 给出的 `MemoryRange`，Offloader 只依赖这个接口。同一个映射里所有决定登记的 Unit，区间先向外对齐到页，再合并成互不重叠的区间逐个登记；一个合并区间登记失败时，落在它里面的 Unit 都改走暂存区。撤销登记按合并区间进行。
- `read_only` 为真（共享存储）时做只读登记，为假（私有存储、rank 分片）时做普通登记。
- 登记后用平台接口确认这段内存被认作锁页，确认不了就当作登记失败，否则异步拷贝会悄悄退化成同步。
- 平台层提供 `host_registration_support() -> (writable, read_only)`。CUDA 查询设备属性，其他平台返回不支持，这些平台的 Unit 走暂存区。
- 时机：`prepare_materialization()` 查询平台能力，在上限内逐个 Unit 决定登记还是走暂存区。`commit()` 时登记：共享存储在封存并只读绑定之后，私有存储和分片在写入、页归还之后；登记后做锁页确认。某个 Unit 登记失败时改走暂存区并记警告。
- 上限按每个 worker 计算，统计对齐、合并之后的字节数。超出时按流式块、常驻层组、整组件 Unit 的顺序登记，其余走暂存区。一个 Unit 要么整个登记，要么整个走暂存区，结果写进启动日志。
- 暂存区（2 × 256 MiB）：本 worker 有 Host 驻留的 Unit 时，Host 容量准入总是计入它；计划里有走暂存区的 Unit 时由 `prepare_materialization()` 分配；`commit()` 里登记失败、暂存区还没分配时，在这时分配。暂存区分配失败，或撤销登记失败时，提交失败。

### 7.5 资源、单次执行与关闭

加速卡存储分两类：

- **固定槽位**：块环缓冲区、分片槽位、叠加层槽位。`prepare_materialization()` 在写模型之前分配，到 `close()` 前地址不变。
- **整组件 Unit 和常驻层组**：装入时向缓存分配器申请，让出时释放，地址不做承诺。

`prepare_materialization()` 还拿齐：内存登记的计划、需要时的暂存区、流、事件、集合通信的工作区；并做加速卡的容量准入：

```text
常驻部分 + 叠加层槽位 + 块环缓冲区 + 分片槽位
  + max(写入期间的瞬时占用, 服务期间同时驻留的使用触发 Unit)
  ≤ 可用显存
```

- 写入期间的瞬时占用：最大的单个适配器 Δ；在线量化时，一层量化前的数据。
- 服务期间同时驻留的使用触发 Unit：按 7.3 的让出关系，取两者中较大的：最大的一个 `dit` 组件的使用触发 Unit 之和；全部被选中的编码器组件的使用触发 Unit 之和，加上最大的一个被选中 `vae` 的使用触发 Unit 之和。使用触发 Unit 指整组件 Unit、常驻层组，以及 layer 组件里重复块以外的使用触发 Unit。

装入时显存不足，按下面的异常规则处理。提交时叠加层槽位交给动态 LoRA 管理器，其余交给 `OffloadRuntime`。每个 rank 的显存峰值是上式左边加上正在使用的激活。

`OffloadRuntime` 提供前向使用的梯度模式：有块环 Unit 时是 `no_grad`，否则是 `inference_mode`。runner 不读 offload 策略；HSDP 下 runner 照旧用 `no_grad`。

一个 `OffloadRuntime` 同一时刻只允许一次前向，可以是一个请求批次；重入或并发的前向被拒绝。出异常时先等拷贝和集合通信做完再回到空闲。集合通信的顺序不再可信时，运行时标记为不可用。

关闭顺序固定：

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

### 7.6 睡眠模式

睡眠模式在服务暂停期间把加速卡上的权重全部腾出来；Offloader 在服务期间管理权重在加速卡上的短暂驻留。两者按显存的类别分工：

| 加速卡上的显存 | 睡眠时 | 唤醒时 |
|---|---|---|
| 常驻加速卡的权重和 buffer、叠加层槽位 | 由睡眠模式拷到 Host，释放加速卡显存 | 由睡眠模式拷回原地址 |
| 块环缓冲区、分片槽位 | 直接释放，不备份 | 重新映射到原地址，内容在下一次使用时装入 |
| 整组件 Unit 和常驻层组 | 进入睡眠前让出 | 下一次使用时装入 |
| host 存储 | 不受影响 | 不受影响 |

- 常驻的部分在睡眠模式的 `weights` 池；块环缓冲区和分片槽位在一个睡眠时只释放、不备份的池里。两个池都在 `prepare_materialization()` 里、写模型之前分配。
- 唤醒后常驻部分和固定槽位的地址不变，编译结果和只读取全部常驻组件的 CUDA graph 继续有效。
- 进入睡眠前，`OffloadRuntime` 等拷贝和集合通信做完，让出所有整组件 Unit 和常驻层组，清空块环。唤醒后的第一次前向按正常流程装入。
- 睡眠期间的前向被拒绝。

### 7.7 模型侧

模型只提供结构和组件声明，驻留、设备、搬运全部由 Offloader 决定。模型不 import Offloader，不读 offload 配置，不按名字读 offload 钩子，不调用 `OffloadRuntime` 的任何方法。

现有模型里与 offload 有关的代码，按下表由通用机制取代：

| 现有代码 | 取代它的机制 |
|---|---|
| 按 offload 策略选构造设备或加载设备 | 只建结构（5.2） |
| 模型自己搬运组件、自己维护锁页副本、自己在使用前后装卸 | 组件声明加使用触发（5.1、7.3） |
| 按 offload 策略关掉快路径或改变结构 | 结构和字节只由模型配置和并行方式决定，不随 offload 策略变化 |
| 模型自带的权重 offload（MiniMax H3 的 AdaLN 投影） | 组件声明的 `unit_groups`，策略的 `units` 按名字选择 |
| 按 offload 组合拒绝 | 现有后端的限制随新栈消失；集合通信顺序、容量、内容身份由通用规则表达（6.4、8.5）；没有正确性问题、只是还没验证的组合不报错，写进支持矩阵 |
| offload 专用声明（`_layerwise_offload_blocks_attrs`、`OffloadPlan`、常驻路径） | 组件声明、重复块发现、`repeated_blocks`、`unit_groups` |
| 模型自己的 CUDA graph 按名字读钩子决定能否捕获 | `fully_resident`（7.8） |

不属于 offload 机制、但在新栈下要改的模型代码主要有以下几类，逐项清单是共存期必改清单（umbrella RFC Job 1）：

- 构造期间读张量（5.2）；
- 加载后改绑参数存储（5.3）；
- 用参数的设备判断执行设备；
- 以存储地址或设备为键的缓存（7.8）；
- 自己捕获 CUDA graph 或选择 cudagraph 编译模式（7.8）：AuK、SenseNova-U1、OmniVoice、Qwen-Image-2.1 改为经通用函数读 `fully_resident`；Ming-Image、DreamZero、InternVLA-A1 的 cudagraph 编译模式改走通用的编译函数；
- 在 `.to()` 之外改参数类型（7.3）。

依赖规则由静态测试保证：`models/`、`lora/`、`cache/` 不 import Offloader，也不按字符串名读 offload 钩子。共存期间，现有的 28 个模型文件和 `lora/manager.py` 对 `vllm_omni.diffusion.offloader` 的 import、SeaCache 按字符串名读钩子，列入冻结的白名单；删除 `legacy` 时清空。

### 7.8 CUDA graph 与编译

**`fully_resident`**。组件声明的只读属性，由 `commit()` 填写。为真的条件：组件的全部权重和 buffer 常驻加速卡，地址到 `close()` 前不变，并且组件不由 HSDP 管理（HSDP 每次前向都把参数拼到新的缓冲区）。

- 模型自己捕获的 CUDA graph 只能读取 `fully_resident` 组件的权重。模型通过一个通用函数读这个属性：`unified` 下读声明，`legacy` 下执行模型现有的判断。
- 不是 `fully_resident` 的组件走 eager。
- 安全网：Offloader 的钩子在 graph 捕获期间被调用时直接报错，写明组件名。

**编译**。

- 编译设置（`setup_compile`）在 `commit()` 之后运行，编译目标从组件声明和 `repeated_blocks` 读取。
- 编译包住块的原始 `forward`，钩子留在编译图之外。
- `default` 模式可用于所有组件。`reduce-overhead`、`max-autotune`、`triton.cudagraphs` 只用于 `fully_resident` 的组件；对其他组件，通用的编译函数自己降成 `default` 并记日志。模型直接传入的编译模式同样经过这一步。

**缓存的身份**。以权重为键的运行期缓存（精确投影缓存、AdaLN 缓存）用参数身份加写入代数做签名，不用存储地址和设备。基础权重提交后不变，写入代数只在动态 LoRA 写入叠加层时递增。Offloader 的钩子不让这些缓存停用。

## 8. 配置

### 8.1 总开关

```text
--diffusion-weight-stack {legacy,unified}
```

默认是 `legacy`，完全使用现有实现。`unified` 同时启用新加载流程、HWM 和新 Offloader。只有这两种组合。

### 8.2 offload 策略

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
- `resident_layers` 默认是 0，可用于任何按 `layer` 处理的组件，只能配 `rank-local`。原则是使用触发的 Unit（整组件、常驻层组）只用 `rank-local`，`allgather` 只用于块环，因为使用触发在各 rank 上发生的时机不保证一致。
- `units` 只用于 `layer`，取组件声明里 `unit_groups` 的名字。MiniMax H3 的 AdaLN 投影写成 `"dit": {"mode": "layer", "units": "adaln"}`。
- 预设和自定义配置在同一个配置来源里互斥。
- 命令行、Python 参数、YAML 使用同一个解析器；多个来源之间，优先级高的整体覆盖。
- 预设里的角色在某个模型上不存在时忽略；自定义配置里显式选了不存在的角色、组件名或 `unit_groups` 名字时，规划阶段报错。
- 未知字段和类型错误报错。

预设和旧开关都不选 `vae`。要让 VAE 由使用触发管理，在自定义配置里选中它。

`unified` 不做与硬件相关的隐式 offload：`none` 就是不 offload。显存放不下时由容量准入报错，并提示 recipe 里的策略。

### 8.3 host 策略

| 设置 | 参数 | 含义 |
|---|---|---|
| 锁页与内存登记 | `--pin-cpu-memory` / `--no-pin-cpu-memory`，字段 `pin_cpu_memory`，默认开启 | 开启时按 7.4 登记 host 存储或使用锁页的暂存区；关闭时不查询平台能力，不登记，暂存区也不锁页 |
| 登记上限 | `--diffusion-host-registration-limit-gib`，字段 `host_registration_limit_gib` | 每个 worker 向加速卡驱动登记的 host 存储上限，覆盖所有 Unit 和分片；0 表示不设上限。只有启用 offload 且 `pin_cpu_memory` 开启时才可以大于 0 |

同机共享没有配置项，满足 6.3 的条件即启用。

### 8.4 旧开关

旧开关由一个兼容层翻译成策略。`legacy` 下它们驱动现有实现，行为不变；`unified` 下按下表翻译：

| 旧开关 | 翻译 |
|---|---|
| `--enable-cpu-offload` | `module`，选中现有实现发现的全部 DiT 和编码器，编码器名单里属于 `vae` 角色的除外 |
| `--enable-layerwise-offload` | `dit` 用 `layer`、`rank-local`；有重复块的 `text_encoder` 也用 `layer`、`rank-local` |
| `--enable-distributed-layerwise-offload` | `dit` 用 `layer`、`allgather`；有重复块的 `text_encoder` 用 `layer`、`rank-local` |
| `--dlo-no-use-allgather` | `dit` 用 `rank-local` |
| `--dlo-resident-layers` | 与 `--enable-distributed-layerwise-offload` 同时出现、并且现有实现为这个模型声明了常驻路径（MiniMax H3、Kandinsky6）时，翻译成 `dit` 的 `resident_layers`；其他模型忽略并记一次警告，与现有行为一致 |
| `--diffusion-offload-config` / `diffusion_offload_config` | `mode` 照搬；`components` 列表里的角色按 `mode` 选中；`layer_options` 里每个组件的 `weight_transfer`、`resident_layers` 照搬；`pin_memory` 翻成 `pin_cpu_memory` |

- 没有重复块的编码器不做 offload，与现有行为一致。
- 现有实现的编码器名单里的名字，在组件声明里按用途给角色：`mllm`、`connectors` 是 `text_encoder`；`vision_model`、`gen_image_condition_refiner`、`conditioning` 是 `image_encoder`；JoyImage 名单里的 `vae` 是 `vae`。
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

### 8.5 不支持的组合

`unified` 下，以下组合报错，并说明原因：

| 组合 | 报错阶段 |
|---|---|
| `module` 配 `allgather`；`resident_layers` 大于 0 配 `allgather` | 参数解析 |
| 自定义配置选择 `other` 角色 | 参数解析 |
| 旧的登记上限、Host Weight Runtime 参数、`pin_memory` 冲突（8.4） | 参数解析 |
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
| pipeline 还没有迁移到只建结构（5.2） | 构造 |

环境变量 `OMNI_DIFFUSION_PROMPT_EMBED_CACHE` 在构建配置时折进配置并冻结，worker 不读它，组合检查只在参数解析时做一次。

### 8.6 校验分层

| 阶段 | 校验内容 |
|---|---|
| 参数解析 | 类型、未知字段、互斥项、新旧冲突、不支持的组合 |
| 构造 | 只建结构、读取陷阱、构造后没有残留的 meta 张量 |
| Offloader 规划 | 组件完整、选中的组件和 `unit_groups` 存在、重复块可发现、常驻层数合法、共用存储的生命周期一致 |
| Loader 预检 | 来源可读、内容身份、冷启动路径、适配器的键 |
| 全组确认第 1 点 | 参与集合通信的部分各 rank 一致、AllGather 组的成员、单机 Host 容量 |
| `prepare_materialization()` | 加速卡和 Host 的运行期资源准入、内存登记或暂存区的计划、多 rank 的全组确认 |

每一层都在自己那一步的第一次不可逆修改之前完成。

### 8.7 动态 LoRA

| 设置 | 参数 | 含义 |
|---|---|---|
| rank 上限 | `--diffusion-max-lora-rank` / `max_lora_rank` | 给出时启用动态 LoRA，按它分配叠加层槽位 |
| CPU 缓存条数 | `max_cpu_loras` | 已有字段，含义不变 |

`unified` 下没有给出 `max_lora_rank` 时，按 5.6 处理：`peft` 加 `--lora-path` 时槽位取这个适配器的 rank；否则动态 LoRA 不启用，`add_lora` 被拒绝并提示这个参数。`legacy` 下行为不变。

---

## 9. 上线方式

1. 新栈作为一套独立的代码加入主干，与现有的三个后端、现有加载路径并存。总开关默认是 `legacy`。
2. 并存期间，`legacy` 的行为不变。`unified` 下，`offloader/config.py` 里供模型查询的函数一律报告“不做 offload”，模型里读这些函数的分支自动不生效。少数在 `unified` 下会弄坏被管理权重的模型代码列在共存期必改清单里（umbrella RFC Job 1），按清单逐项改：每一项只改变 `unified` 下的行为，`legacy` 下的行为和已有测试不变；按总开关或构造上下文分出的 legacy 一支随 `legacy` 一起删除。提交后的守卫让漏掉的地方在第一次使用时报错，不会静默出错。
3. 上线门槛：现有每个有端到端测试或每晚例行测试的“模型 × offload 策略”组合，在 `unified` 下输出一致，Host 和加速卡的内存峰值不高于 `legacy`。
   - 对比基准：现有实现里结构或类型随 offload 变化的组合（Flux2-Klein TP 大于 1 配 offload、DreamZero 的 VAE 类型），以 `legacy` 不做 offload 的输出为基准。按空闲显存选择分块大小的 VAE，对比时固定分块。融合适配器的组合逐张量对比融合结果；`legacy` 下结果本来就不对的组合（蒸馏 LoRA 配在线量化、蒸馏 LoRA 配 TP 大于 1）以“完整加载、融合、再量化”的参考结果为基准。
   - 不在门槛之内、写进迁移说明的组合：被 HSDP 管理的组件做 offload；MiniMax H3 在 SM120 上不加参数的 AdaLN offload（`unified` 下由 recipe 写明 `units`）；Qwen-Image-2.1 配 `--enable-cpu-offload` 时解码的 CUDA graph（`unified` 下 DiT 按整组件 offload，解码走 eager：显存更低，解码更慢）；8.4 列出的五个模型用旧开关时的显存峰值；适配器的键在已加载的组件上对不上、或融合时 `lora_scale` 不等于 1 的部署（`unified` 在预检报错），以及 `legacy` 静默丢掉增量、`unified` 融合进去的适配器；同一台机器上的两个部署跑同一个模型时的 Host 占用（`legacy` 的分布式逐层后端配 `--dlo-no-use-allgather` 时直接映射 checkpoint，页缓存跨部署共用；`unified` 下不同部署各占一份）。
4. 门槛全部达到，并且满足下面三条之后，默认值翻转为 `unified`：
   - 主干里注册的每个 pipeline 都能在只建结构的上下文里构造，支持矩阵里没有 `unified` 不支持的 pipeline；
   - 每个迁移家族的逐张量对比测试通过，对比例外表里的每一条都已列入门槛例外；
   - legacy 一支的指纹清单没有未处理的变化。
5. 翻转后的一个版本保留 `legacy` 作为回退，共存期的规则照旧执行。主干外没有迁移的 pipeline（`register_diffusion_model` 注册的、`load_format="custom_pipeline"` 的）在这个版本里要显式选 `legacy`。
6. 再下一个版本删除 `legacy`：三个现有后端；diffusers 后端整条 pipeline 的 `from_pretrained` 加载和它对 diffusers 自带 offload 的调用；现有的非 HSDP 加载路径（包括其中对 Host Weight Runtime 的接入）；模型里的 offload 专用声明和生命周期代码；共存期两套构造代码里的 legacy 一支。删除之前，逐张量对比测试把 legacy 的结果存成逐张量的哈希；删除之后，改为与这份哈希对比。

### 9.1 共存期的构造代码

构造期读张量的 pipeline 迁移时，legacy 的构造代码不改。读张量的调用点用 `build_component`（5.2）分成两支：legacy 一支是原语句，unified 一支是新写的结构构造和来源声明。两支并存到第 6 步。

- 迁移的 PR 不修改已有测试文件，`legacy` 下已有测试全部通过。
- 每个迁移的家族有一个以 legacy 为基准的逐张量对比测试：同一份 checkpoint，legacy 正常构造加加载，unified 走到收尾结束、规划为全部常驻，两边在同一类设备上运行。比较张量名、形状、类型、步长、逐字节内容、持久和非持久 buffer、模块外的数据张量、共用关系、模块树的路径和类名。结构可以配置的家族用合成的小 checkpoint，在改动对应家族目录或 Loader 的 PR 上运行；结构固定的家族用真实 checkpoint，每晚运行。
- legacy 一支调用的读取和转换函数登记在一份清单里，按限定名记录源码指纹。清单用来发现 legacy 一支的改动，也是第 6 步的删除清单。

### 9.2 共存期 legacy 修复的同步

| 修复改动的位置 | 怎么发现 | 怎么做 |
|---|---|---|
| 共用代码 | 不需要 | 两个栈同时生效 |
| legacy 一支，改变最终张量、类型、布局或模块树 | 逐张量对比测试失败 | 同一个 PR 里改 unified 一支 |
| legacy 一支，不改变最终张量（报错、路径解析、重试、校验） | 指纹测试失败 | 同一个 PR 里改 unified 一支，或在清单里写明 unified 不需要的原因，再更新指纹 |
| 只服务于 legacy 放置和搬运的代码 | 不需要 | 不同步，unified 由通用机制取代 |

有意让 unified 与 legacy 不同的张量，在对比例外表里逐条写明原因，并列入门槛例外。

### 9.3 删除的内容

模型里被删除的内容：

- `SupportsComponentDiscovery` 和 `_dit_modules` / `_encoder_modules` / `_vae_modules`，它们的使用方已改读组件声明；
- `_layerwise_offload_blocks_attrs`、模型声明的 `OffloadPlan`、常驻路径声明；
- `SupportsModelCpuOffload`；
- MiniMax H3 的驻留控制器、分配器缓存、AdaLN 的 Host 暂存和 SM120 默认、VAE 两半和放大器的搬运；Cosmos3 的阶段上下文；MAGI-2 的分阶段组件和内存窗口；其他模型自己的搬运代码；
- 模型里按 offload 策略改变行为的分支，以及只为现有后端存在的拒绝；
- 模型加载后钩子里只为现有实现存在的部分（`validate_restored_host_weights` 等）；
- `_maybe_fuse_distilled_lora`、pipeline 上会融合权重的 `load_lora_weights`、worker 里的第二个蒸馏 LoRA 融合点，以及 LoRA 管理器里读 offload 策略的代码；
- 每个 `build_component` 调用的 legacy 一支；调用改成直接调用 unified 一支，`build_component` 本身删除；
- 指纹清单登记的 legacy 读取和转换函数，只删没有其他使用方的。OmniVoice 的 generator、decoder 的 `load_weights(model_dir, device)` 还被自回归路径使用，保留；`LanceWanVAE` 的读取同时被 Lance 和 MAGI-2 使用，两个家族都只剩 unified 一支后才删；
- HF 组件入口里加载带权重组件的 legacy 一支，以及 `create_transformers_model` 里的 `init_parameters`、`create_transformers_model_with_vllm_linears` 里的 `init_parameters_preserving_vllm_linear`；
- 只服务于 legacy 的 `load_weights` 主体：自己读文件的、空操作的、替没有写入的组件声明“已加载”的；
- 规则表取代的运行期改类型代码：LingBot-Video 和 InternVLA-A1 的 `.to()` 覆写、π0.5 的 `_set_inference_dtype`、LTX2 vocoder 的 `.float()` 往返；LTX2 在 `eval()` 里物化相位适配器的代码；MiniMax H3 pipeline 对 FastH3、VDN 的流包装；SeaCache 读分布式逐层 offload 钩子的代码；
- 指纹清单和指纹测试。

删除前的保证：

- 每个注册的扩散 pipeline 按测试覆盖的并行拓扑和量化配置分别在 meta 上构造，断言：现有声明里的块全部被发现；发现的组与期望表逐项一致，期望表写明多出或缺少的原因；容器本身不成为 Unit。用了 `repeated_blocks` 的组件同时记录自动发现的结果和覆盖的原因。每个模型另做一次 `layer` 冒烟测试。
- 只靠 `_layerwise_offload_blocks_attrs` 回退被编译的块类先补进 `_repeated_blocks`：`FluxSingleTransformerBlock`、`VaceWanTransformerBlock`、`MotionerAttentionBlock`、`MiniMaxH3TokenRefinerBlock`、`BooguImageDoubleStreamTransformerBlock`，以及 SDXL 的 `SDXLDownBlock2D`、`SDXLCrossAttnDownBlock2D`、`SDXLUpBlock2D`、`SDXLCrossAttnUpBlock2D`。一个测试对比删除前后被编译的模块集合。
- MAGI-2 的内存窗口由 recipe 里的配置给出：`dit` 选 `module`、`rank-local`。
- 逐张量对比测试的 legacy 结果已存成逐张量的哈希（第 6 步）。

---

## 10. 内存模型

记一个组件的最终权重大小为 `W`，AllGather 组的大小为 `N`。

本机后端，每台机器、每个部署：

```text
稳态   = 每个共享集合一份 W（与 worker 数、副本数、stage 数无关）
       + 每个使用私有内存的 worker 一份 W
       + 需要时的暂存区
冷启动 = 稳态 + checkpoint 的页缓存（内核可回收）
              + 写入进程的瞬时占用（变换中的单个张量；在线量化时是一层量化前的数据）
```

写入进程退出后又有 worker 加入时，新的写入进程重建一份 W；旧的一份在映射它的进程全部退出后归还。同一台机器上的不同部署各占各的，不合并。

分片后端，每个 rank：

```text
稳态                 = W / N（含等分时的补零）+ 需要时的暂存区
冷启动，瘦路径       = 稳态 + 释放阈值 + 最大的单个张量 + checkpoint 的页缓存（内核可回收）
冷启动，先全量后裁剪 = 稳态 + W（短暂）
```

每台机器的占用是本机各 rank 之和。

**容量准入**。每个副本在 4.5 第 1 点做，在写入之前：

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

写入进程在别处的共享集合不计 W。新增占用超过可用内存时，本副本所有 rank 同时报错。同一台机器上并发启动的副本各自做准入：共享集合的 W 由选举保证只计一次；各副本私有的部分并发时可能合计超出，这时在写入阶段因内存不足而失败。本地扩散副本依次启动，不受这种情况影响。

加速卡，每个 rank：见 7.5 的容量准入式。

---

## 11. 资源所有权

| 资源 | 持有者 |
|---|---|
| checkpoint 读取器、变换中的临时张量、适配器的 Δ | Loader，仅在写入期间 |
| 组件声明、内容身份、最终清单 | Loader |
| 拓扑、共享计划 | Runner，仅在启动期间 |
| host 存储（私有内存、共享内存的本进程映射、rank 分片） | HWM 分配；启动事务持有，提交后转给 `OffloadRuntime` |
| 共享存储的监听套接字和监听线程（写入进程） | `elect()` 返回的 `Election` 持有，`allocate()` 时移交 `HostStorage`，到写入进程的 `close()` 为止；在这之前失败时由 Runner 关闭 |
| 内存登记 | 在 `commit()` 里建立，转给 `OffloadRuntime` |
| 暂存区 | 计划里有走暂存区的 Unit 时在 `prepare_materialization()` 里分配；`commit()` 里登记失败、暂存区还没分配时在这时分配；转给 `OffloadRuntime` |
| 叠加层槽位 | 启动事务持有，提交后转给动态 LoRA 管理器 |
| 动态适配器的 CPU 缓存 | 动态 LoRA 管理器 |
| 加速卡槽位、流、事件、集合通信工作区、钩子 | 启动事务持有，提交后转给 `OffloadRuntime` |

同一个句柄在资源图里只登记一次。Offloader 借用 host 存储，只关闭自己的映射。

---

## 12. 不变量

1. 所有可预知的配置、拓扑、容量错误在第一次写模型之前报出。同一台机器上并发启动的副本之间，只保证共享集合的 W 不被重复计入（第 10 节）。
2. 每个最终张量只做一次有语义的写入；会改字节的收尾只在一个地方做一次。
3. 收尾完成后字节不再变化；提交后基础权重的字节、布局、共用关系不可变。
4. 内容身份不包含 offload 策略、叠加层、不改字节的 DP 编号、设备地址、文件路径和部署标识。
5. 一个 rank 稳定后在 Host 上只持有规划指定的那一份：本机后端下是本 worker 所在共享集合的那一份，或私有的一份；分片后端下是自己的 rank 分片。
6. 不同部署之间不共享 host 存储，也不互相连接。套接字名包含部署标识，交接时还要再核对一次。
7. 一个共享集合在一台机器上同一时刻最多只有一个写入进程。共享内存的字节只由它写，封存后任何进程都不能修改。
8. `seal()` 里不做集合通信。写入进程封存并交出之前不等待任何套接字；等待者在本副本收尾的集合通信全部完成之后才开始等待。
9. 运行期让出只释放加速卡存储，不往 Host 回写。
10. 所有进程退出后，内存里不留任何权重。
11. 关闭不恢复 CPU 权重，不发起集合通信。
12. 集合通信失败后，这一组不再继续，这个副本的全部 worker 终止。
13. 提交后模块结构不变；切换动态 LoRA 只改叠加层槽位里的数据。
14. 模型自己捕获的 CUDA graph 只读取 `fully_resident` 组件的权重。
15. 睡眠时不备份的只有块环缓冲区和分片槽位；唤醒后常驻部分和固定槽位的加速卡地址不变。

---

## 13. 风险

| 风险 | 影响 | 处理 |
|---|---|---|
| NUMA：共享内存文件的物理页落在写入进程所在的 NUMA 节点上 | 另一路 CPU 上的加速卡只能跨路做 H2D；一个部署在这台机器上的所有副本共用这份存储，都受影响。现有实现每个 worker 的锁页副本都在本地节点 | 门槛矩阵包含双路 8 卡机器，记录跨路 H2D 的带宽 |
| 非 CUDA 平台（ROCm、NPU、XPU、MUSA）不做内存登记 | 每个流式块每一步都先 CPU 拷贝再经暂存区 H2D，比现有实现的锁页副本慢；门槛不比较性能 | 支持矩阵写明这些平台走暂存区；ROCm 的可写登记有验证测试后放开 |
| 内存登记的驱动开销 | 驱动按登记量占用内核内存，约 2～4 MiB/GiB | 门槛测试记录登记前后的内核内存占用 |
| Cache-DiT 按不在 SP 组上同步跳块处理 | 与 `allgather` 同时启用时各 rank 执行的块不一致，AllGather 错位 | 参数解析阶段拒绝这个组合（8.5） |
| diffusers pipeline 在 `__call__` 里改组件类型 | 被管理的组件撞上提交后的守卫 | 规则表固定类型（5.2），冒烟测试覆盖 |
| 封存前的失败跨启动单元传递 | head 的本地副本和同机 headless 副本、或两个 headless 启动器并发启动时，一方的写入进程在封存前失败，另一方启动失败 | 启动日志写明角色、写入进程 pid 和套接字名；测试覆盖两种角色 |
| 写入进程退出后有新副本加入 | 新的写入进程再建一份 W，内存紧的机器上替换副本准入失败 | 写进迁移说明 |
| 按进程 RSS 求和统计内存 | 共享页算进每个映射它的进程的 RSS，W 被算 N 次 | 可观测性按共享比例平摊；cgroup 和 `Shmem` 统计只算一次 |
| 只用 Python 的 at-fork 钩子 | C 层直接 fork 出的子进程会继承写入进程的文件描述符和监听套接字，延长内存寿命并占住名字 | 记为已知限制 |
