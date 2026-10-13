# Loader / Offloader / HWM 重构总体设计

> 范围：vllm-omni 的扩散加载代码（`vllm_omni/diffusion/model_loader/`、`vllm_omni/diffusion/offloader/`）。
>
> 基线：upstream `main@726b21a5e`。关联：#6648、#6414、#7204。
>
> 配套文档：[设计摘要](./LOADER_OFFLOADER_DESIGN_SUMMARY.md)、[umbrella RFC](./rfcs/00-umbrella.md)。

## 0. 设计结论

1. **Loader 管权重内容**：负责模型结构、checkpoint 到参数的映射、各种变换，以及把最终字节写进参数。机制是通用的“先分配、后写入”，没有模型专属代码。
2. **HWM（Host Weight Manager）管 Host 上的存储**：一个接口，两个后端。本机后端让同一台机器上的多个 worker 共享一份；分片后端让多台小内存机器共同持有一份。
3. **Offloader 管运行期搬运**：策略由 Unit、Trigger、Transport 三个互相独立的要素组成。锁页、内存登记、暂存区、加速卡上的槽位和流都归它。
4. **权重只存在于内存里**：每次启动都从 checkpoint 构建。
5. **内存由内核回收**：共享内存没有文件名，不占用 `/dev/shm`；最后一个持有它的进程退出后立即释放，正常退出、`kill -9`、OOM 的结果相同。
6. **一个 worker 退出时，这个 stage 的全部 worker 一起终止**：服务重启后，每个 stage 从 checkpoint 构建。
7. **新栈与现有实现并存，由一个总开关切换**：达到上线门槛后翻转默认值，随后删除现有实现。
8. **模型里没有 offload 专用的代码**：Offloader 只读 Loader 的组件声明和通用的结构信息。

Loader、HWM、Offloader 连同把它们串起来的启动流程，下文统称新栈。

---

## 1. 需求与范围

### 1.1 需求

| 需求 | 设计要求 |
|---|---|
| 一台机器上多个 worker 用同一份权重 | 整台机器只占一份 Host 内存 |
| 多台小内存机器共同部署一个模型 | 每个 rank 只持有 1/N，冷启动的占用也有上限 |
| 加速卡显存不足 | 权重按需在 Host 和加速卡之间搬运 |
| 部署退出后不留内存垃圾 | 正常退出、`kill -9`、OOM 三种情况下内存都被回收 |
| 不破坏现有部署 | 总开关关闭时现有行为完全不变 |
| 机制通用 | 没有针对单个模型的加载或 offload 代码 |

### 1.2 范围之外

- 自回归和语音 stage。它们使用 vLLM 上游的加载器。
- DP、SP、TP 等并行方式的计算语义和请求调度。
- 激活值和输出的搬运。
- 可变的基础权重。动态权重是不修改基础权重的叠加层。
- HSDP 的加载与运行。被 HSDP 管理的组件由 HSDP 自己加载、分片和拼接。
- Host Weight Runtime（`vllm_omni/host_weight_runtime/`）。这个包不改动，新栈不使用它。

---

## 2. 名词

| 名词 | 含义 |
|---|---|
| stage | vllm-omni 里承担一个模型阶段的一组 worker 进程 |
| rank | 一个 worker 进程在分布式进程组里的编号，一个 rank 对应一块加速卡 |
| DP / SP / TP | 三种并行方式：数据并行（多个副本各自处理请求）、序列并行（一个请求的序列切给多个 rank）、张量并行（每个 rank 持有参数的一部分） |
| HSDP | 一种把参数本身分片放在各 rank 的加速卡上、使用前拼接的并行方式 |
| 组件 | Loader 声明的、带名字和角色的模型部分，例如 DiT、文本编码器、VAE |
| 重复块 | 组件里结构相同、依次执行的一组层，例如 DiT 的每一个 block |
| 内容身份 | 判断同一次部署里两个 worker 需要的最终权重是否完全相同的依据 |
| 最终清单 | 加载完成后每个张量的名字、形状、类型、在存储里的位置，以及哪些张量共用同一段存储 |
| host 存储 | HWM 分配的、存放最终权重字节的 Host 内存 |
| rank 分片 | 一个重复块的全部字节拉平后按 rank 等分，某个 rank 持有的那一段 |
| Unit | Offloader 一起驻留、一起搬运的一组权重：整组件、一个重复块，或一组常驻层 |
| Trigger | 决定某个 Unit 什么时候装上加速卡、什么时候让出 |
| Transport | 一个 Unit 怎么从 Host 到加速卡 |
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

---

## 3. 模块与依赖

### 3.1 模块职责

| 模块 | 负责 | 不负责 |
|---|---|---|
| Runner（扩散 worker 里的模型运行器） | 构造下面三个模块，按顺序推进启动流程，关闭时调用 `close()` | 解析 checkpoint，规划 Unit，分配 host 存储 |
| Loader | 组件声明、只建结构、checkpoint 映射、变换、写入、收尾、内容身份、最终清单 | 分配 host 存储，H2D，AllGather |
| HWM | 按规划结果分配 host 存储；同机共享；rank 分片 | 理解 checkpoint，锁页和内存登记，加速卡资源 |
| Offloader | Unit / Trigger / Transport 规划；锁页、内存登记、暂存区；加速卡槽位、流、事件；H2D 与 AllGather；钩子；关闭 | 读取 checkpoint，决定字节内容 |

### 3.2 依赖规则

1. Loader 只通过接口调用 HWM 和 Offloader，不 import 它们的具体实现。
2. Offloader 不 import Loader 和 safetensors 读取器，不调用模型的 `post_load_weights`。
3. HWM 不解释 checkpoint 的 key，不做改名、切分、合并、量化或收尾。
4. 模型只向 Loader 提供组件声明和通用结构信息，不读 offload 配置或 host 配置。
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
    def allocate(layout) -> HostStorage: ...

class HostStorage:
    def view(tensor_key) -> torch.Tensor: ...     # 写入目标，或只读视图
    def ranges(unit) -> list[MemoryRange]: ...    # 给 Offloader 做内存登记和 H2D
    def seal(final_manifest) -> None: ...         # 定稿，之后只读
    def close() -> None: ...
```

阶段、所有权的转移和失败时的行为是规范性的。

### 4.2 步骤

```text
1.  Runner 构造 Loader、HWM、Offloader
2.  Loader 只建结构：模型里所有带权重的组件都没有真实数据
3.  Offloader.plan()：纯计算，得出 Unit、Trigger、Transport，以及每个组件用哪种 host 存储
4.  Loader 预检：解析 checkpoint 来源，算出内容身份和预期清单
5.  HWM.allocate()：为需要 Host 驻留的张量分配存储
6.  Offloader.prepare_materialization()：在第一次写模型之前，
    拿齐所有占容量的运行期资源，返回启动事务
7.  Loader 写入：把参数绑定到已分配的存储，load_weights 逐个张量写
8.  Loader 收尾：量化、LoRA 融合、共用存储、校验；host 存储定稿
9.  启动事务 commit()：校验最终清单，装上触发器，资源整体移交 OffloadRuntime
10. 开始服务
```

### 4.3 状态

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

### 4.4 规则

- `plan()` 不分配大块资源，不改模型，不装钩子。
- 所有可以预知的错误（配置、拓扑、容量）在第一次写模型之前报出。
- 启动事务是提交前所有临时资源的唯一持有者。没提交就退出时，它自动释放本 worker 的资源。
- 提交成功后，全部资源一次性移交给 `OffloadRuntime`。同一个资源句柄只登记一次。
- 从开始写模型到提交成功，这个模型实例是可丢弃的；写入、收尾或提交失败时直接丢弃。
- 进入 `IN_SERVICE` 之后的失败是运行期故障：这个 stage 的全部 worker 终止。

### 4.5 多个 rank 的全组确认

规划里有跨 rank 的集合通信时，在三个点上做全组确认，走已有的 torch.distributed 进程组：

1. 规划之后：各 rank 的 Unit、内容身份、布局、Transport 的指纹一致。
2. `prepare_materialization()` 之后：所有 rank 都就绪，才返回启动事务。
3. `commit()` 时：所有 rank 都完成收尾，才一起激活。

任何一个 rank 在确认前失败，整组一起失败。

---

## 5. Loader

### 5.1 组件声明

pipeline 的构造函数里，每个带权重的部分声明一次。一条声明包含：

- 稳定的组件名和角色。角色是 `dit`、`text_encoder`、`image_encoder`、`audio_encoder`、`vae`、`other` 之一。
- 组件在 pipeline 里的路径。
- 只用配置、不用权重数值的结构构造方法。
- 权重来源。
- checkpoint 里的形式和内存里的形式不一样时，一张规则表（见 5.2）。

Z-Image 的文本编码器、VAE、DiT 就是通过声明的权重来源让 Loader 看得见的（`pipeline_z_image.py` 的 `weights_sources`）。

### 5.2 构造函数只建结构

- vLLM 风格的组件用配置构造函数。
- transformers 的类按 `create_transformers_model`（`vllm_omni/diffusion/models/utils.py`）的方式构造。
- diffusers 的类用 `from_config`。
- 只由配置算出来的普通 buffer 照常创建。
- 构造函数不读 checkpoint 的张量数值，不把带权重的模块搬到设备上。

`from_pretrained` 在加载时顺带做的事（改名、跳过某些部分、让两个参数共用存储、让某些层保持 fp32）由每个类一张规则表表达。每张表配一个测试，逐张量对比它和 `from_pretrained` 的结果。规则表是数据，由 Loader 统一消费。

`load_format="diffusers"` 按 `model_index.json` 拆成组件，走同一条流程。分词器、调度器和纯配置不经过 Loader。

### 5.3 先分配、后写入

这是唯一的权重物化机制：

1. 写入之前，每个参数的存储已经按规划分配好：留在加速卡上的在加速卡，Host 驻留的在 HWM 分配的 host 存储里。
2. Loader 把参数绑定到这些存储上，然后调用 `load_weights`。它从 checkpoint 逐个张量读出、变换、写入。TP 切分、合并参数、模型自己的映射都使用现有的 `weight_loader` 写法。
3. 每个最终张量只做一次有语义的写入。
4. 收尾过程中被整个替换掉的参数（例如在线量化把权重换成另一种类型并新增缩放系数），在收尾的最后一步拷进 host 存储。

这个机制与模型无关。

### 5.4 收尾顺序

Loader 按固定顺序做一次：

1. 完成所有写入或绑定。
2. 做会改字节的工作：在线量化、静态 LoRA 或蒸馏 LoRA 融合、会改参数的校准。
3. 建立共用存储的关系（例如输入输出嵌入共用一份）。
4. 校验必需的参数和持久 buffer 都有真实数据。
5. 运行模型的加载后校验，以及不改字节的运行期设置。
6. 把第 2 步里被替换的参数拷进 host 存储，生成最终清单，host 存储定稿。
7. 交给启动事务 `commit()`。

第 6 步之后字节不再变化。提交之后，基础权重的字节、布局、共用关系都不可变。合并或拆分 LoRA、动态量化需要关闭运行时，重新走一次启动流程。

同机共享时的等待者从共享内存直接绑定：第 1 步是绑定，跳过第 2 步和第 6 步里的拷贝，其余照做。绑定的字节和存储地址在不改字节的收尾前后保持不变。

### 5.5 内容身份与最终清单

内容身份在一次部署内部使用：判断同一台机器上的两个 worker 需要的最终权重是否完全相同（6.3），以及多个 rank 在规划之后核对彼此是否一致（4.5）。

它由配置和本 worker 的坐标算出：模型的标识和版本、组件边界、运行期表示（类型、量化方式）、会改变字节或形状的并行坐标（TP 的大小和编号）、静态融合的适配器、Loader 规则表的版本。

它不包含 offload 策略、不改变字节的 DP 编号、设备地址和文件路径，也不比较 checkpoint 文件的字节内容。

最终清单由 Loader 在收尾第 6 步生成，是同机共享时按名字绑定的唯一依据。

---

## 6. HWM

### 6.1 后端的选择

后端由规划决定：

| 组件的 Transport | 后端 | 这个 rank 在 Host 上持有 |
|---|---|---|
| 本 rank 直接 H2D | 本机后端 | 完整一份，同机器共享 |
| 分片 H2D 后 AllGather | 分片后端 | 自己的 rank 分片 |
| 不做 offload，常驻加速卡 | 不经过 HWM | 无 |

`allocate()` 的 `layout` 来自 Offloader 的规划和 Loader 的预期清单：有哪些张量、各多大、属于哪个 Unit。

### 6.2 与并行方式的关系

| 并行方式 | 各 rank 的权重内容 | host 存储 |
|---|---|---|
| DP | 相同 | 本机后端下同机共享；分片后端下各持 1/N |
| SP | 相同 | 同 DP；DP 为 1 时 AllGather 的分组用 SP 组 |
| TP | 每个编号不同 | 内容身份包含 TP 的大小和编号；相同编号的 worker 之间共享 |
| HSDP | 由 HSDP 管理 | 不经过 HWM |

### 6.3 本机后端：同一台机器共享一份

**共享的条件**。同时满足以下四条的 worker 共享一份：

- 属于同一次部署；
- 在同一台机器的同一个容器里；
- 整套权重的内容身份和规划完全相同；
- 写入之前就能确定每个张量的最终形状和类型。

不满足条件的 worker 使用进程私有内存，各自加载。整个部署只有一个 worker 时使用进程私有内存。

**内存文件**。共享的存储是没有文件名的内存文件，由 Linux 的 `memfd_create`（Python 的 `os.memfd_create`）创建：

- 它位于内核内部的内存文件系统，文件系统里看不到它，只能通过文件描述符访问。
- 它不占用 `/dev/shm`，大小只受机器内存和容器内存限额约束。它占用的内存记在第一个写入它的进程所在的容器上。
- 大小在加载前按预期清单算出，一次定好；物理页在写入时才占用。
- 只要还有进程持有它的文件描述符或映射，它就存在；最后一个持有者退出，内核立即释放。
- 它的页不被内核丢弃，所以共享的权重必须在机器内存里放得下。

**格式**。内存文件的内部格式是 safetensors：

- 每个组件按数据类型分成若干内存文件，同一个文件里的张量类型相同，每个张量的起点都按它的类型对齐。
- 布局由张量的名字、形状、类型算出：按名字排序、连续摆放，头部补齐到 8 字节。同一个重复块的张量相邻。
- 最终清单就是 safetensors 的头部。
- 绑定用 safetensors 库的 `safe_open("/proc/self/fd/N")`。

**步骤**：

1. 每个 worker 用“部署标识 + 整套内容身份 + 规划指纹”算出一个本机套接字名。部署标识是执行器下发的分布式初始化地址，同一次部署的所有 worker 相同，不同部署不同。套接字使用 Linux 的抽象命名空间，不在文件系统里留文件，进程退出后名字自动消失。
2. 第一个绑定成功的 worker 是写入进程，其余 worker 连接上去等待。绑定是原子的。
3. 写入进程创建内存文件，以可写方式映射，把参数绑定到映射上的各个偏移，完成写入和收尾。
4. 定稿。写入进程撤掉可写映射，给内存文件加上封存：不许写、不许加长、不许缩短。此后任何进程都不能修改它。
5. 交接。写入进程以只读方式重新打开内存文件，用 `socket.send_fds` 把只读的文件描述符发给每个等待者。发送前校验对方和自己是同一个系统用户。
6. 绑定。所有 worker，包括写入进程，都以只读方式映射并按头部逐个名字绑定，走同一段代码。等待者在此之前完成 `prepare_materialization()`，绑定之后做不改字节的收尾。
7. 写入进程持有文件描述符，直到本次启动的所有 worker 都拿到。之后每个 worker 只保留映射，关闭文件描述符。

**只读映射**。封存“不许写”之后，内核拒绝通过可读写的文件描述符建立共享映射。绑定使用私有的只读映射，它带写时复制：某个 worker 对共享权重的写入只影响它自己。

**写入进程的瞬时占用**。在线量化在加速卡上进行，写入进程在量化期间短暂持有量化前的数据。

**失败**。写入进程在构建过程中退出时，等待者的连接收到结束信号，启动失败。

### 6.4 分片后端：多台小内存机器共同持有一份

**布局**。每个重复块的全部字节按类型分组拉平，按 rank 等分成 N 段，不足的部分补零；rank r 持有第 r 段。运行期每个 rank 把自己那段 H2D，再用 `all_gather_into_tensor` 拼成完整的块。

**分组**。DP 大于 1 时用 DP 组，否则用 SP 组。一个组可以跨多台机器，每台机器持有“本机 rank 数 / N”的权重。内存大的机器放更多的 rank。

**存储**。每个 rank 的分片是它自己的私有内存。

**冷启动**。每个 rank 稳定后只占自己的分片。构建过程有两条路径，由规划选定并写入启动日志：

| 路径 | 条件 | 构建过程中每个 rank 的额外占用 |
|---|---|---|
| 瘦路径 | 写入之前就能确定每个张量的最终形状和类型 | 有上限：一个释放阈值（一个重复块的大小）加上最大的单个张量 |
| 先全量、后裁剪 | 最终形状或类型在收尾时才确定，例如在线量化 | 完整一份，短暂持有 |

瘦路径的做法：

- 每个 rank 为整个组件预留一段地址空间，物理内存在写入时才占用。参数绑定到这段地址空间上，`load_weights` 照常写。
- 完全落在本 rank 范围之外的张量不分配存储，写入对它们是空操作，也不读 checkpoint 里对应的数据。
- 跨越范围边界的张量先完整写入，再把范围之外的页归还给内核。归还按已写字节数的阈值触发。
- 需要变换的边界张量由相邻的两个 rank 各处理一次。

小内存多机部署使用瘦路径。

**分摊的范围**。跨 rank 分摊的是重复块：DiT 的块，以及各 rank 内容相同的文本编码器的块。DiT 里重复块以外的部分、VAE、图像编码器在每个 rank 上都是完整一份，必须单机放得下；放不下时规划阶段报错，并指出是哪个组件。

**约束**。AllGather 要求同组所有 rank 以相同顺序执行相同的块：

- DP 下同一波请求有相同的去噪步数和影响执行路径的参数。
- DP 配 AllGather 时不能启用缓存加速和 prompt embedding 缓存。

### 6.5 生命周期与清理

- 权重只存在于内存里：共享内存，或各 worker 的私有内存。
- 共享内存由内核按持有者计数。它在最后一个持有它的进程真正退出时释放，正常退出、`kill -9`、OOM 的结果相同。
- 分片后端的分片是进程私有内存，进程退出即释放。
- `HostStorage.close()` 只释放本进程的映射，不影响其他进程。
- 一个 worker 退出时，这个 stage 的全部 worker 一起终止。多进程执行器和 Ray 执行器都是这个行为。
- 服务重启后，每个 stage 从 checkpoint 构建。
- worker 进程不产生只复制自己、不换成别的程序的子进程（只 fork 不 exec）；这种子进程会继承映射并延长共享内存的寿命。worker 由 spawn 启动。

---

## 7. Offloader

### 7.1 策略的三个要素

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

### 7.2 Unit 规划与重复块发现

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

### 7.3 Trigger

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

### 7.4 Transport

- `rank-local`：从 host 存储直接 H2D。存储所在的内存登记给加速卡驱动后直接异步拷贝；平台不支持登记或超出登记上限时，经过两个固定大小的暂存区。内存登记在 CUDA 上可用。
- `allgather`：把本 rank 的分片 H2D 到一对分片槽位，再拼成完整的块放进块环的槽位。
- 任何策略都不生成完整的锁页副本。

### 7.5 资源、单次执行与关闭

`prepare_materialization()` 在写模型之前拿齐：host 存储的内存登记或暂存区、加速卡上的 Unit 槽位、块环槽位、分片槽位、流、事件、集合通信的工作区。这些资源是真实分配的。

一个 `OffloadRuntime` 同一时刻只允许一次前向，可以是一个请求批次；重入或并发的前向被拒绝。出异常时先等拷贝和集合通信做完再回到空闲。集合通信的顺序不再可信时，运行时标记为不可用。

关闭顺序固定：

```text
停止接受新的执行
→ 等加速卡上的拷贝和集合通信做完
→ 撤掉钩子和对模型的引用
→ 释放加速卡槽位、事件、流
→ 撤销内存登记，释放暂存区
→ 关闭 host 存储的本进程映射
```

`OffloadRuntime.close()` 是本地的、终止性的：不把权重恢复回 CPU，不发起集合通信。

### 7.6 模型侧

模型里没有 offload 专用的代码。新栈只读 Loader 的组件声明和通用的结构提示。模型里按特定组合拒绝 offload 的检查由通用规则表达。给编译或上游库用的通用结构提示保留。

---

## 8. 配置

### 8.1 总开关

```text
--diffusion-weight-stack {legacy,unified}
```

默认是 `legacy`，完全使用现有实现。`unified` 同时启用新加载流程、HWM 和新 Offloader。只有这两种组合。

### 8.2 offload 策略

```text
--diffusion-offload {none,module,layerwise,distributed-layerwise}
--diffusion-offload-config '<JSON>'
```

四个预设各自等价于一份自定义配置：

| 预设 | 展开 |
|---|---|
| `none` | 不做 offload |
| `module` | 角色为 `dit`、`text_encoder`、`image_encoder` 的组件按整组件 |
| `layerwise` | `dit` 按重复块，`rank-local` |
| `distributed-layerwise` | `dit` 按重复块，`allgather` |

自定义配置按角色选组件，每个组件各自设置 `mode`、`weight_transfer`、`resident_layers`：

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

### 8.3 host 策略

| 设置 | 参数 | 含义 |
|---|---|---|
| 锁页与内存登记 | `pin_cpu_memory` | 开启时登记 host 存储或使用锁页的暂存区；关闭时不登记，暂存区也不锁页 |
| 登记上限 | `--dlo-host-registration-limit-gib` | 每个 worker 登记内存的上限，0 表示不设上限 |

同机共享没有配置项，满足 6.3 的条件即启用。

### 8.4 旧开关

旧开关由一个兼容层翻译成策略。`legacy` 下它们驱动现有实现，行为不变；`unified` 下按下表翻译：

| 旧开关 | 翻译 |
|---|---|
| `--enable-cpu-offload` | `module`，组件范围与现有行为一致 |
| `--enable-layerwise-offload` | `layer`，`rank-local`，块的发现范围与现有行为一致，不流式搬运嵌套的子模块 |
| `--enable-distributed-layerwise-offload` | `layer`，DiT 用 `allgather` |
| `--dlo-no-use-allgather` | DiT 用 `rank-local` |
| `--dlo-resident-layers` | DiT 的 `resident_layers` |

新旧两种写法同时出现且冲突时报错，并给出迁移提示。

`legacy` 下，被 HSDP 管理的组件可以做整组件 offload 或 `rank-local` 逐层 offload。`unified` 下这两种组合在规划阶段报错，报错信息提示在 HSDP 和 offload 之间二选一。

### 8.5 不支持的组合

`unified` 下，以下组合在参数解析或规划阶段报错，并说明原因：

| 组合 | 报错阶段 |
|---|---|
| `module` 配 `allgather` | 参数解析 |
| 自定义配置选择 `dit`、`text_encoder`、`image_encoder` 以外的角色 | 参数解析 |
| `--host-weight-runtime-mode` 不是 `disabled` | 参数解析 |
| DP 大于 1 且使用 `allgather` 时启用缓存加速或 prompt embedding 缓存 | 参数解析 |
| 被 HSDP 管理的组件被选中做 offload | 规划 |
| 分片后端下，重复块以外的组件单机放不下 | 规划 |

### 8.6 校验分层

| 阶段 | 校验内容 |
|---|---|
| 参数解析 | 类型、未知字段、互斥项、新旧冲突、不支持的组合 |
| Offloader 规划 | 选中的组件存在、重复块可发现、常驻层数合法、共用存储的生命周期一致、单机容量 |
| Loader 预检 | 来源可读、内容身份、冷启动路径 |
| `prepare_materialization()` | 加速卡和 Host 的运行期资源准入、内存登记或暂存区的选择、多 rank 的全组确认 |

每一层都在自己那一步的第一次不可逆修改之前完成。

---

## 9. 上线方式

1. 新栈作为一套独立的代码加入主干，与现有的三个后端、现有加载路径并存。总开关默认是 `legacy`。
2. 上线门槛：现有每个有端到端测试或每晚例行测试的“模型 × offload 策略”组合，在 `unified` 下输出一致，Host 和加速卡的内存峰值不高于 `legacy`。被 HSDP 管理的组件做 offload 的组合在 `unified` 下不再支持，不在门槛之内。
3. 门槛全部达到后，默认值翻转为 `unified`。
4. 翻转后的一个版本保留 `legacy` 作为回退。
5. 再下一个版本删除 `legacy`：三个现有后端、现有的非 HSDP 加载路径（包括其中对 Host Weight Runtime 的接入）、模型里的 offload 专用声明和生命周期代码。

模型里被删除的内容：

- `SupportsComponentDiscovery` 和 `_dit_modules` / `_encoder_modules` / `_vae_modules`；
- `_layerwise_offload_blocks_attrs`、模型声明的 `OffloadPlan`、常驻路径声明；
- `SupportsModelCpuOffload`；
- MiniMax H3 的驻留控制器和分配器缓存代码、Cosmos3 的阶段上下文、MAGI-2 等模型自己的搬运代码。

每个现有的块声明在删除前都有一个测试，证明自动发现的结果与它一致。MAGI-2 默认的内存窗口由 recipe 里的配置给出。

---

## 10. 内存模型

记一个组件的最终权重大小为 `W`，AllGather 组的大小为 `N`。

本机后端，每台机器：

```text
稳态   = W（整台机器一份，与 worker 数无关）+ 每个 worker 的暂存区
冷启动 = 稳态 + checkpoint 的页缓存（内核可回收）
              + 写入进程的瞬时占用（变换中的单个张量；在线量化时是量化前的数据）
```

分片后端，每个 rank：

```text
稳态                 = W / N（含等分时的补零）+ 暂存区
冷启动，瘦路径       = 稳态 + 释放阈值 + 最大的单个张量 + checkpoint 的页缓存（内核可回收）
冷启动，先全量后裁剪 = 稳态 + W（短暂）
```

每台机器的占用是本机各 rank 之和。

容量准入在写入之前完成，按整台机器的总内存计算；在容器里按容器的内存限额计算。

---

## 11. 资源所有权

| 资源 | 持有者 |
|---|---|
| checkpoint 读取器、变换中的临时张量 | Loader，仅在写入期间 |
| 组件声明、内容身份、最终清单 | Loader |
| host 存储（私有内存、共享内存的本进程映射、rank 分片） | HWM 分配；启动事务持有，提交后转给 `OffloadRuntime` |
| 内存登记、暂存区 | 启动事务 → `OffloadRuntime` |
| 加速卡槽位、流、事件、集合通信工作区、钩子 | 启动事务 → `OffloadRuntime` |

同一个句柄在资源图里只登记一次。Offloader 借用 host 存储，只关闭自己的映射。

---

## 12. 不变量

1. 所有可预知的配置、拓扑、容量错误在第一次写模型之前报出。
2. 每个最终张量只做一次有语义的写入；会改字节的收尾只在一个地方做一次。
3. 收尾完成后字节不再变化；提交后基础权重的字节、布局、共用关系不可变。
4. 内容身份不包含 offload 策略、不改字节的 DP 编号、设备地址和文件路径。
5. 一个 rank 稳定后在 Host 上只持有规划指定的那一份：完整一份（本机后端，整台机器共享）或自己的 rank 分片（分片后端）。
6. 运行期让出只释放加速卡存储，不往 Host 回写。
7. 所有进程退出后，内存里不留任何权重。
8. 关闭不恢复 CPU 权重，不发起集合通信。
9. 集合通信失败后，这一组不再继续，这个 stage 的全部 worker 终止。
