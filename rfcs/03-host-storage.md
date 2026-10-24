# [RFC]: Host 存储（HWM）

> 上级 RFC：[umbrella RFC](./00-umbrella.md)。设计依据：[总体设计](../LOADER_OFFLOADER_DESIGN.md) 第 6 节、第 10 节，以及第 4.1、4.5、9.3、11、12 节中与 host 存储有关的部分。
>
> 范围：`vllm_omni/diffusion/weight_stack/host/`；部署标识经过的 `vllm_omni/engine/stage_runtime.py`、`vllm_omni/engine/stage_engine_startup.py`、`vllm_omni/engine/stage_init_utils.py`；`vllm_omni/diffusion/data.py` 的 `OmniDiffusionConfig`，以及 `vllm_omni/config/omni_config.py` 的 `_DIFFUSION_INTERNAL_FIELDS`；`tests/diffusion/weight_stack/host/`。
>
> 依赖：[子 RFC 01](./01-startup-transaction.md)（`interfaces.py` 的 `HostWeightManager`、`HostStorage`、`Topology`、`RankFingerprints`、`Elections`、`AllGatherGroups`、`StartupContext`；`errors.py` 的错误类型；全组确认第 0、1、2 点的调用顺序；`FaultPlan`；测试替身）；[子 RFC 02](./02-unified-loader.md)（`ExpectedManifest`、`FinalManifest` 的条目和 `digest()`、`live()`；`Loader.write()` 对 meta 目标的空操作；`WriteTracker` 在每个张量写完时调用 `HostStorage.note_written()`；`finalize()` 第 6 步调用一次 `seal()`）；[子 RFC 04](./04-unified-offloader.md)（`PlacementPlan.host_request()`、`Unit.unit_id`；启动事务持有和关闭 host 存储；按 `MemoryRange` 做内存登记）；[子 RFC 05](./05-configuration.md)（`host_share`、`host_admission` 事件的字段，`observability.record_host_mapping()`；同机共享没有配置项）。

## 摘要

本 RFC 实现总体设计第 6 节的 HWM：一个 `HostWeightManager` 接口，本机后端和分片后端两个后端，以及进程私有内存。内容包括：

- `LocalRecord` 的每一项在 Linux 上的读取方式，以及按内存 cgroup 逐级计算本机可用内存的方法；
- 部署标识的生成（`StageRuntime`）、下发（`od_config.deployment_id`、`OmniMasterServer` 的注册回复）和没有下发时的副本内标识；
- `HostLayout`：本机后端按组件、按类型的 safetensors 布局，分片后端按类型拉平、补零的布局，规划指纹；
- 共享指纹、套接字名、`SharePlan` 的分组和代表、`elect()` 的原子绑定和重试；
- 写入进程和等待者之间的套接字协议：消息类型和字段、版本、双方的 `SO_PEERCRED` 核对、文件描述符的分批交出；
- 内存文件的创建、写入、`seal()` 的定稿顺序、等待者的核对和只读绑定、`close()`；
- 分片后端的地址预留、瘦路径的页归还、先全量后裁剪；
- 容量准入 `admit()`；
- fork 处理、各执行器下的行为、失败处理和测试。

HWM 不需要加速卡，也不调用 `torch.distributed`。它的全部测试都可以在 CPU 上、用多个进程完成。

## 动机

总体设计第 6 节和 umbrella RFC 的背景说明了要解决的问题。现有代码里与 host 存储有关的具体情况如下，都已在代码里核对：

- offload 后端为每个 worker 各自持有一份 Host 上的权重。`SequentialOffloadHook._move_params`（`vllm_omni/diffusion/offloader/sequential_backend.py`）对参数调用 `data.pin_memory()`；逐层后端的 `LayerwiseOffloadHook.initialize_hook`（`vllm_omni/diffusion/offloader/layerwise_backend.py`）经 `_to_cpu` 为每个块按类型 `torch.empty(..., device="cpu", pin_memory=pin_memory)`，再把块的参数拷进去。这些都是进程私有的锁页内存，同一台机器上的两个 worker 各占一份。可选的 Host Weight Runtime（`vllm_omni/host_weight_runtime/`）在 `root` 目录下保存权重文件，以 mmap 映射的只读张量交给加载路径（`vllm_omni/diffusion/model_loader/host_weight_loader.py`）；它是基于文件系统的存储，新栈不使用它（总体设计 1.2）。
- 分布式逐层后端的分片在加载完成之后才做。`DistributedLayerwiseOffloadHook._shard_and_pin`（`vllm_omni/diffusion/offloader/distributed_layerwise_backend.py`）从已加载的参数里按 `dp_size` 和 `rank` 拷出本 rank 的一段，所以冷启动时每个 rank 先持有完整的一份；分组按 DP 组，而不是按组件的内容身份。
- 同一个后端的 `rank_local_mmap` 方式直接映射 checkpoint，构造时在 `dp_size != 1` 时报 `ValueError`；它依赖 checkpoint 的页缓存，页缓存在内存紧张时被回收，也被同一台机器上的其他部署共用。
- 编排代码里没有“部署”的标识。`vllm_omni/` 里没有任何 `deployment_id`，`OmniDiffusionConfig` 里也没有能区分两次启动的字段，所以无法安全地只在同一个部署内部共享。
- `OmniDiffusionConfig` 已有一个由编排代码在运行期填写的内部字段 `ray_worker_env`（`init=False, repr=False`）：`build_diffusion_config`（`vllm_omni/engine/stage_init_utils.py`）在构造配置之后写入它，`vllm_omni/config/omni_config.py` 把它列在 `_DIFFUSION_INTERNAL_FIELDS` 里，已有测试 `test_diffusion_config_field_classification_covers_current_fields`（`tests/config/test_omni_config.py`）断言 `OmniDiffusionConfig` 的每个字段都被分类。部署标识沿用这个写法（第 3.4 节）。

## 设计

### 1. 模块与对象

| 对象 | 实现 | 状态 |
|---|---|---|
| `HostWeightManager` | `host/manager.py` 的 `HostWeightManagerImpl`，由 `create_host_weight_manager(ctx)` 构造 | 只缓存 `layout()` 的结果（数据），不持有任何系统资源（01 第 2 节） |
| `Election` | `host/election.py` | 写入进程的 `Election` 持有监听套接字和监听线程；代表在别处有写入进程时持有到写入进程的连接 |
| `HostStorage` | `host/storage.py` 的 `CompositeHostStorage`，组合本机部分（共享或私有）和分片部分 | 持有内存文件、映射、监听套接字和连接，直到 `close()` |

- HWM 只支持 Linux。`create_host_weight_manager(ctx)` 在非 Linux 平台上报 `StackUnavailableError`。
- HWM 不 import `loader/`、`offload/` 和 `vllm_omni.diffusion.model_loader`（01 的 T1）。它读 `ExpectedManifest`、`FinalManifest`、`PlacementPlan` 和 `Unit` 只经过它们的属性和方法，不 import 它们所在的模块。
- 所有系统调用经过 `host/sysapi.py` 的一个小类 `SysApi`：`/proc`、cgroup 文件系统的根目录，`geteuid`，`memfd_create`，`madvise`。测试替换这个对象来构造假的 cgroup 树、伪造 uid、记录页归还和注入错误。
- `CompositeHostStorage` 的每个方法按名字把调用转给张量所在的部分。`seal()` 按固定顺序依次处理分片部分（第 9 节）、私有内存（第 8 节）和共享部分（写入进程第 7.3 节，等待者第 7.4 节）。前两部分只做本地操作，所以等待者只在最后一步阻塞接收。没有任何 Host 驻留张量的 worker 同样调用一次 `seal()`，它什么都不做。

### 2. 本机记录

#### 2.1 数据类型

```python
# vllm_omni/diffusion/weight_stack/host/record.py
@dataclass(frozen=True)
class LocalKey:
    boot_id: str
    net_ns: tuple[int, int]            # /proc/self/ns/net 的 (st_dev, st_ino)
    pid_ns: tuple[int, int]            # /proc/self/ns/pid 的 (st_dev, st_ino)
    memory_cgroup: tuple[int, int]     # 内存 cgroup 目录的 (st_dev, st_ino)
    euid: int
    def canonical(self) -> list: ...   # 进入共享指纹的规范形式

@dataclass(frozen=True)
class LocalRecord:                     # 可以 pickle
    rank: int
    pid: int
    hostname: str                      # 只用于日志和报错
    machine: str                       # 容量准入按它分组：boot_id；读不到时为 "host:<hostname>"
    local_key: LocalKey | None         # 任何一项读不到或能力探测失败时为 None，这个 rank 不参与共享
    unreadable: tuple[str, ...]        # 读不到的项的名字和原因，写进启动日志
    available_bytes: int | None        # 本机可用内存（2.4）；读不到时为 None
    memory_source: str                 # "cgroup-v2:<目录>"、"cgroup-v1:<目录>" 或 "meminfo"
```

`local_record()` 每次调用都重新读取，不缓存。

#### 2.2 读取方式

| 项 | 读取方式 | 读不到的情况 |
|---|---|---|
| boot_id | 读 `/proc/sys/kernel/random/boot_id`，去掉首尾空白。 | 文件不存在或内容为空。 |
| 网络命名空间 | `os.stat("/proc/self/ns/net")` 的 `st_dev` 和 `st_ino`。 | `stat` 失败。 |
| PID 命名空间 | `os.stat("/proc/self/ns/pid")` 的 `st_dev` 和 `st_ino`。 | `stat` 失败。 |
| 内存 cgroup | 按 2.3 找到本进程所在的内存 cgroup 目录，取 `os.stat(目录)` 的 `st_dev` 和 `st_ino`。不使用路径字符串：不同的 cgroup 命名空间里同一个路径可以指向不同的 cgroup。 | 找不到挂载点，或者本进程的 cgroup 不在可见的挂载范围之内。 |
| 有效 uid | `os.geteuid()`。 | 不会读不到。 |
| 能力探测 | `os.memfd_create("vllm-omni-hwm-probe", MFD_CLOEXEC \| MFD_ALLOW_SEALING)`，对它 `F_ADD_SEALS` 加 `F_SEAL_SEAL`，然后关闭。 | 系统调用被拒绝（例如容器的 seccomp 策略）或内核不支持封存。 |

任何一项读不到时 `local_key` 为 `None`，`unreadable` 写明是哪一项和原因，这个 rank 使用私有内存。boot_id 读不到时 `machine` 退回 `"host:" + socket.gethostname()`，只用于容量准入的分组。

#### 2.3 内存 cgroup 的定位

```text
1. 读 /proc/self/cgroup
   有 "<id>:<控制器列表>:<路径>" 且控制器列表含 memory 的行     使用 cgroup v1 的这一行
   否则有 "0::<路径>" 的行                                      使用 cgroup v2
   都没有                                                       读不到
2. 读 /proc/self/mountinfo，找挂载点
   v2：文件系统类型为 cgroup2 的条目
   v1：文件系统类型为 cgroup、超级块选项含 memory 的条目
   取这个条目的 root（第 4 列）和挂载点（第 5 列）
3. 路径必须等于 root 或在 root 之下，否则读不到
   目录 = 挂载点 / relpath(路径, root)
```

同时存在 v1 的 memory 行和 v2 的 `0::` 行时（混合模式），内存控制器挂在 v1 上，使用 v1。

#### 2.4 本机可用内存

按总体设计第 10 节计算，从 2.3 得到的目录逐级向上，直到挂载点为止：

| 版本 | 限额 | 用量 | 文件页 |
|---|---|---|---|
| v2 | `memory.max`，内容为 `max` 时这一级没有限额 | `memory.current` | 这一级 `memory.stat` 的 `inactive_file` 与 `active_file` 之和 |
| v1 | `memory.limit_in_bytes` | `memory.usage_in_bytes` | 这一级 `memory.stat` 的 `total_inactive_file` 与 `total_active_file` 之和（v1 的用量含子 cgroup，文件页也取含子 cgroup 的值） |

- 每一级有限额时，这一级的可用值是 `限额 − 用量 + 文件页`，结果小于 0 时取 0。限额不小于 `/proc/meminfo` 的 `MemTotal` 时这一级按没有限额处理（v1 用一个接近 `2^63` 的值表示没有限额）。
- 可用内存取各级可用值的最小值。各级都没有限额时，取 `/proc/meminfo` 的 `MemAvailable`。
- 文件页只取 `inactive_file`、`active_file`，不用 `file`，因为 `file` 含内存文件和 tmpfs 的页，这些页回收不了。
- 读某一级的文件失败时跳过这一级；全部失败、`/proc/meminfo` 也读不到时，`available_bytes` 为 `None`，`memory_source` 写明原因。

`local_record()` 在第 0 点读一次；代表在 `elect()` 里再读一次，写进 `ElectionRecord.available_bytes`（第 5.5 节）。

### 3. 部署标识

#### 3.1 生成

`StageRuntime.__init__`（`vllm_omni/engine/stage_runtime.py`）里增加：

```python
self._deployment_id = secrets.token_hex(16)    # 32 个十六进制字符，128 位
```

`DistStageRuntime.__init__` 调用 `super().__init__()`，因此分布式模式由同一行生成。值在 `StageRuntime` 的整个生命期内不变，每次构造 `StageRuntime` 都重新生成。生成不读总开关，`legacy` 下这个值不被使用，行为不变。

#### 3.2 本地副本

| 路径 | 改动 |
|---|---|
| `StageRuntime._initialize_local_diffusion_replica` | 调用 `launch_diffusion_stage_replica` 时传入 `deployment_id=self._deployment_id`。 |
| `launch_diffusion_stage_replica`（`stage_engine_startup.py`）没有 `OmniMasterServer` 的一支 | 把 `deployment_id` 传给 `initialize_diffusion_stage`。 |
| `initialize_diffusion_stage`（`stage_init_utils.py`） | 增加关键字参数 `deployment_id: str \| None = None`；`build_diffusion_config()` 返回之后、`create_diffusion_client()` 之前执行 `od_config.deployment_id = deployment_id`。 |
| `launch_diffusion_stage_replica` 有 `OmniMasterServer` 的一支 | `build_diffusion_config()` 返回之后、构造 `StageDiffusionProcManager` 之前执行 `od_config.deployment_id = deployment_id`。 |

这几处都在启动副本的进程之前写入，满足总体设计 6.3 的顺序。

#### 3.3 headless 副本

- `OmniMasterServer.__init__` 增加关键字参数 `deployment_id: str | None = None`，保存为 `self._deployment_id`。`DistStageRuntime` 构造它时传入 `deployment_id=self._deployment_id`。
- `OmniMasterServer._handle_registration` 的回复字典增加 `"deployment_id": self._deployment_id`。LLM 和扩散副本收到同一个字段，LLM 一侧不读它。
- `StageRegistrationResponse` 增加字段 `deployment_id: str | None = None`；`register_stage_with_omni_master` 读 `response_msg.get("deployment_id")`。旧版本的 head 不发这个字段时为 `None`。
- `launch_headless_diffusion_replica` 在 `register_stage_with_omni_master` 返回之后、`StageDiffusionProcManager.launch_headless` 之前执行 `od_config.deployment_id = response.deployment_id`。`launch_headless_diffusion_replicas` 的几个副本共用一个 `od_config` 对象，它们从同一个 head 拿到同一个值；拿到的值与对象上已有的非空值不同时报 `RuntimeError`。
- 运行期才注册的副本走同一个注册流程，拿到同一个值。

#### 3.4 `OmniDiffusionConfig.deployment_id`

```python
# vllm_omni/diffusion/data.py，OmniDiffusionConfig 的最后一个字段
deployment_id: str | None = field(default=None, init=False, repr=False, compare=False)
```

- 写法与已有的内部字段 `ray_worker_env` 相同。`init=False` 使它不是构造函数的参数，任何经 `OmniDiffusionConfig(**kwargs)` 或 `OmniDiffusionConfig.from_kwargs` 传入 `deployment_id` 的配置都在构造时报 `TypeError`，与现在传入 `ray_worker_env` 相同；命令行没有对应的参数。字段只由第 3.2、3.3 节列出的编排代码写入。
- `vllm_omni/config/omni_config.py` 的 `_DIFFUSION_INTERNAL_FIELDS` 增加 `"deployment_id"`，已有测试 `test_diffusion_config_field_classification_covers_current_fields` 因此不需要修改就能通过。枚举 `fields(OmniDiffusionConfig)` 的几处（`OmniDiffusionConfig.normalize_init_kwargs`、`omni_config.py` 的 `enrich_config` 和两处 `diffusion_fields`、`config_factory.py` 的 `diffusion_config_fields`）不改，它们对 `ray_worker_env` 的处理照旧，`legacy` 下的行为不变。
- `compare=False`：两个只差部署标识的配置仍然相等。部署标识不进内容身份（总体设计 5.5）。

#### 3.5 到达 worker

| 执行器 | 传递方式 |
|---|---|
| 多进程执行器（`MultiprocDiffusionExecutor`） | 以 spawn 方式启动 `WorkerProc.worker_main`，`od_config` 作为参数被 pickle。dataclass 的默认 pickle 保存实例的 `__dict__`，`init=False` 的字段随之保留。 |
| Ray 执行器（`RayDiffusionExecutor`） | `od_config` 作为 `RayDiffusionWorkerWrapper.init_worker` 的参数序列化。 |
| 单进程执行器（`UniProcDiffusionExecutor`）和 inline（`InlineStageDiffusionClient`） | worker 用的就是编排进程或引擎进程里的同一个对象。 |
| `StageDiffusionProc` 子进程 | `StageDiffusionProcManager` 用 `get_mp_context()` 启动子进程，`od_config` 被 pickle。 |
| 自定义执行器 | 只要构造 `DiffusionWorker` 时传入同一个 `od_config`，就自动适用。 |

`dataclasses.replace` 不复制 `init=False` 的字段。非测试代码里对 `od_config` 调用它的只有 Qwen-Image-2.1 pipeline（`vllm_omni/diffusion/models/qwen_image_21/pipeline_qwen_image_21.py`）为构造文本编码器复制出的 `encoder_od_config`；这份副本只交给 `create_transformers_model`，HWM 读的是 `StartupContext.od_config`，不受影响。03-2 的测试对上表每一种传递方式断言字段保留。

#### 3.6 `deployment_id()`

按 01 第 2 节：`ctx.od_config.deployment_id` 不为 `None` 时返回（这个值，`True`）；否则返回（`secrets.token_hex(16)` 新生成的值，`False`）。只有 0 号 rank 的结果被采用；`Topology.deployment_scoped` 为假时，共享只发生在本副本之内（第 5.3 节）。值不是 32 个小写十六进制字符时报 `ConfigError`。

### 4. 布局与规划指纹

#### 4.1 输入

`layout(plan, expected_manifest)` 只读两样东西：

- `plan.host_request()` 返回的对象（04 实现，结构见下）；
- `expected_manifest.entries` 里每个 `ManifestEntry` 的 `name`、`component`、`shape`、`dtype`、`stride`、`nbytes`、`aliases`、`replaced_at_finalize`，以及 `expected_manifest.digest(names)`。

```python
# vllm_omni/diffusion/weight_stack/host/layout.py
class HostUnitRequest(Protocol):
    unit_id: str
    component: str
    backend: Literal["local", "sharded"]       # local：本机后端（rank-local）；sharded：分片后端（allgather 的重复块）
    tensors: tuple[str, ...]                   # 这个 Unit 的规范名，不含别名

class ShardedComponentRequest(Protocol):
    component: str
    group_size: int                            # 规划由坐标推算的 AllGather 组大小 N
    group_index: int                           # 本 rank 在组里的位置 r
    cold_start: Literal["thin", "full_then_trim"]

class HostRequest(Protocol):
    units: Sequence[HostUnitRequest]           # 规划顺序
    sharded: Mapping[str, ShardedComponentRequest]
    staging_bytes: int                         # 本 worker 有 Host 驻留的 Unit 时为暂存区大小（总体设计 7.4 的 2 × 256 MiB），否则为 0
    transient_bytes: int                       # 写入期间 Host 上的瞬时占用：本 worker Host 驻留张量里最大的单个张量的字节数，
                                               # 与 catalog.write_transient.quant_layer_bytes 中的较大者（总体设计 10）
    shapes_final: bool                         # 本机后端的每个组件在写入之前就能确定最终形状和类型
```

- `units` 里每个张量必须在预期清单里、恰好属于一个 Unit；不满足时报 `PreflightError`，写明张量名。
- `cold_start` 为 `thin` 的组件里不能有 `replaced_at_finalize` 为真的条目：瘦路径下本 rank 没有范围之外的字节，收尾第 2 步需要整个模型的工作无法执行（总体设计 6.4 的瘦路径条件）。不满足时 `layout()` 报 `PreflightError`，写明组件名和条目名。
- 别名不单独分配存储，跟随它的规范张量。

#### 4.2 本机后端的布局

本机后端的张量按（组件，类型）分成若干内存文件，文件键是 `"<组件名>/<类型名>"`，类型名是 `str(dtype)` 去掉 `torch.` 前缀。每个文件的格式是 safetensors：

```text
偏移 0          8 字节小端整数 H：头部长度
偏移 8          H 字节 JSON 头部，用空格补齐，使 8 + H 是 8 的倍数
偏移 8 + H      数据区：按名字排序的张量依次连续摆放
文件大小        8 + H + 数据区大小
```

- 张量按规范名的字符串顺序（码点顺序）排列。共用一个前缀的名字在这个顺序里是连续的一段，所以同一个重复块、同一个子模块的张量相邻（总体设计 6.3）。由几个子树组成的 Unit（常驻层组、路径列表声明的整组件）在文件里可以是几段。
- 同一个文件里的张量类型相同，数据区起点是 8 的倍数，所以每个张量的起点都按它的类型对齐。
- 每个张量占 `ManifestEntry.nbytes` 字节，即它的存储范围；步长不是行优先连续的张量（例如规则表要求 channels_last 的）按存储范围原样摆放。
- 头部的张量条目是 `{"dtype": <safetensors 类型名>, "shape": [...], "data_offsets": [起, 止]}`，偏移相对数据区起点。`__metadata__` 含：`format` 为 `vllm-omni-hwm`，`format_version`，`component`，`dtype`，`expected_digest` 为 `expected_manifest.digest(本文件的名字)`，`aliases` 为别名到规范名的 JSON 字符串，`strides` 为非连续张量的步长的 JSON 字符串。
- safetensors 类型名使用一张固定的对照表（`BF16`、`F16`、`F32`、`F64`、`I8`、`U8`、`I16`、`I32`、`I64`、`BOOL`、`F8_E4M3`、`F8_E5M2` 等）。有张量的类型不在表里时，这个 worker 的本机部分不参与共享，原因写进启动日志。
- 头部 JSON 用 `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=True)` 生成，同样的输入在每个进程里得到同样的字节。
- 头部在 `layout()` 里由预期清单算出，`layout()` 时就确定了 H 和每个张量的偏移；`seal()` 用最终清单重新生成头部，必须与它逐字节相同（第 7.3 节）。

#### 4.3 分片后端的布局

对 `sharded` 里的每个组件 c，取它 `backend="sharded"` 的 Unit（每个是一个重复块），按 `units` 的顺序排成一列：

```text
对每个 Unit u（flatten_shard_unit 的规则）：
  张量按类型分组；组按类型名排序；组内张量按规范名排序，依次连续摆放，记下组内偏移
  组 g 的元素数 E_g = 组内字节数 / 元素大小
  每段元素数 seg_g = ceil(E_g / N)，每段字节数 S_g = seg_g × 元素大小
  组 g 补零后的大小 P_g = N × S_g
  本 rank 在组里的范围 = [r × S_g, (r + 1) × S_g)
Unit 补零后的大小 = Σ_g P_g
主机上的预留区（分片后端自己的规则）：
  每个 Unit 的每个组放在一个起点按页大小对齐的位置，页大小取 max(4096, mmap.PAGESIZE)，
  所以一页只属于一个组
  组件预留区大小 = 各 Unit 各组的 P_g 向上取整到页大小之后的和
释放阈值     = 本 worker 全部瘦路径组件里最大的一个 Unit 补零后的大小
最大两个张量 = 本 worker 全部瘦路径组件的分片 Unit 里最大的两个张量的 nbytes 之和
```

- 按元素数取上整等分，与现有 `DistributedLayerwiseOffloadHook._shard_and_pin` 的切分方式一致（`(total_numel + dp_size - 1) // dp_size`）。
- `flatten_shard_unit(tensors, group_size) -> ShardUnitLayout` 是纯函数，从 `weight_stack.host` 导出，只给出组的划分、组内偏移、`E_g`、`seg_g`、`S_g` 和 `P_g`，不含任何对齐。04 用同一个函数决定块环缓冲区里 AllGather 结果的布局，再按自己的规则把组的起点对齐到 256 字节（04 第 3.8 节）；主机上按页对齐只属于分片后端。两边不各写一份切分规则。

#### 4.4 共享资格

本机部分满足以下全部条件时可以共享，否则这个 worker 的本机部分使用私有内存：

- `host_request().shapes_final` 为真；
- 预期清单里没有 `replaced_at_finalize` 为真的条目，也就是收尾第 2 步为空；
- 本机部分的每个类型都在 safetensors 对照表里；
- 本机部分不为空。

`local_key` 为 `None`、副本内的共享集合只有自己并且部署标识只在本副本内有效，这两种情况在 `share_plan()` 里判断（第 5.3 节）。

#### 4.5 规划指纹

```text
本机部分为空                plan_fingerprint = ""
本机部分不满足 4.4          plan_fingerprint = D + "/private"
否则                        plan_fingerprint = D
D = blake2b(规范 JSON, digest_size=16).hexdigest()
规范 JSON 覆盖：每个文件按文件键排序的（文件键，H，数据区大小，
               每个张量的（名字，形状，类型，步长，偏移，nbytes，别名））
```

- D 只由规划和预期清单决定，也就是只由配置和并行布局决定。它不读 `LocalRecord`，不读可用内存和空闲显存，04 的规划也不能让 Unit 的划分依赖这些值（见“与其他子 RFC 的接口”里对 04 的约定）。
- `share_plan()` 只把 `plan_fingerprint` 是 32 个十六进制字符的 rank 放进共享集合。

#### 4.6 `HostLayout`

```python
class HostLayout:
    plan_fingerprint: str
    local_files: tuple[LocalFileLayout, ...]           # 按文件键排序
    sharded: tuple[ShardComponentLayout, ...]          # 按组件名排序
    shareable: bool
    not_shareable_reason: str | None
    staging_bytes: int
    transient_bytes: int
    def demand(self) -> HostDemand: ...
    def bind_groups(self, groups: AllGatherGroups) -> None: ...

@dataclass(frozen=True)
class HostDemand:                                      # 可以 pickle，随 ElectionRecord 收集
    local_bytes: int          # 本机部分各文件大小之和，即一份 W
    shard_bytes: int          # 稳定后的 rank 分片：各组覆盖本 rank 范围的整页之和
    cold_extra_bytes: int     # 有瘦路径组件时：释放阈值 + 最大两个张量；
                              # 另加每个先全量后裁剪组件的预留区大小（总体设计 10）
    staging_bytes: int
    transient_bytes: int
```

`bind_groups(groups)`：对每个分片组件，在 `groups[component]` 里找到含本 rank 的组；组的大小不等于 N，或本 rank 在 `ranks` 里的位置不等于 r 时，报 `GroupConfirmationError`，写明组件名、推算值和实际值。`bind_groups` 之前调用 `allocate()` 报 `LifecycleError`。

### 5. 共享计划与选举

#### 5.1 共享指纹

```text
共享指纹 = blake2b(规范 JSON, digest_size=16).hexdigest()       32 个十六进制字符，128 位
规范 JSON = {"format_version": HWM_FORMAT_VERSION,               存储格式版本，取 1
             "local_key": LocalKey.canonical(),
             "content_identities": 按组件名排序的（组件名，内容身份摘要），取本 rank 的全部组件,
             "plan_fingerprint": plan_fingerprint}
```

内容身份取 `RankFingerprints.content_identities` 的全部条目，也就是总体设计 6.3 的“整套权重的内容身份”。部署标识不在共享指纹里，单独进套接字名。

#### 5.2 套接字名

```text
b"\0vllm-omni-hwm/" + 部署标识 + b"/" + 共享指纹             共 80 字节，小于 sun_path 的 108 字节
```

#### 5.3 `share_plan(topology, fingerprints)`

纯函数，每个 rank 用相同的输入得到相同的结果：

```text
1. 候选 rank：records[k].local_key 不为 None，并且 fingerprints[k].plan_fingerprint 是 32 个十六进制字符
2. 候选 rank 按（共享指纹）分组；共享指纹含本机键，所以一组一定在同一台机器、同一个容器里
3. 每组按 rank 升序排列，rank 最小者是代表
4. topology.deployment_scoped 为假，并且一组只有一个 rank 时，这一组不成立，这个 rank 使用私有内存
5. 其余有本机部分的 rank 使用私有内存，private_reasons 写明原因
```

```python
@dataclass(frozen=True)
class ShareSet:
    share_fingerprint: str
    socket_name: bytes
    machine: str
    members: tuple[int, ...]           # 本副本内的成员，升序
    representative: int                # members[0]

@dataclass(frozen=True)
class SharePlan:
    deployment_id: str
    deployment_scoped: bool
    sets: tuple[ShareSet, ...]                 # 按代表排序
    set_index: tuple[int | None, ...]          # 按 rank：所在共享集合在 sets 里的下标
    private_reasons: tuple[str | None, ...]    # 按 rank：有本机部分却不在共享集合里的原因
    def set_for(self, rank: int) -> ShareSet | None: ...
```

#### 5.4 `elect(share_plan)`

每个 rank 都调用。本 rank 不是任何共享集合的代表时返回 `outcome="none"` 的空 `Election`；代表执行：

```text
attempt = 0
loop:
  s = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC)
  bind(s, socket_name)
    成功                  listen(s, 128)；启动监听线程；outcome = "writer"；结束
    EADDRINUSE            关闭 s；连接 socket_name（第 6.4 节的客户端步骤，只到收到 STATUS 为止）
      连接成功并通过核对  outcome = "remote"，保存连接、写入进程的 pid、sealed、bytes_remaining；
                          sealed 为真时接着收齐 HANDOFF（第 6.1 节），收到的描述符由 Election 持有；结束
      ECONNREFUSED/EAGAIN attempt += 1；attempt < 20 时等 50 ms 回到 loop（先重新尝试绑定）
      其余失败、核对不通过、重试用完    outcome = "private"，记下原因；结束
    其他错误              outcome = "private"，记下原因；结束
代表另外读一次本机可用内存（2.4），写进 available_bytes
```

- 名字已被占用、连接被拒绝，说明绑定这个名字的套接字还没开始监听或正在关闭；重试时名字空出，代表在下一轮绑定成功，成为写入进程。
- 对方的 uid 与本进程的有效 uid 不同时按“其他错误”处理，本组使用私有内存，代表不连到这个名字上（总体设计 6.3）。
- 启动日志写明每个代表的结果、写入进程的 pid 和套接字名（套接字名里的 `\0` 写成 `@`）。

#### 5.5 `Election` 与 `ElectionRecord`

```python
@dataclass(frozen=True)
class ElectionRecord:                          # 可以 pickle
    rank: int
    outcome: Literal["none", "writer", "remote", "private"]
    share_fingerprint: str | None
    writer_pid: int | None                     # writer：本进程；remote：对方
    writer_stage_id: int | None
    remote_sealed: bool | None                 # 只在 remote 时给出
    remote_bytes_remaining: int | None         # 只在 remote 时给出
    available_bytes: int | None                # 代表参选时读到的本机可用内存
    demand: HostDemand                         # 本 rank 自己的 HostLayout.demand()
    reason: str | None                         # private 的原因

class Election:
    outcome: str
    share_set: ShareSet | None
    def record(self) -> ElectionRecord: ...
    def close(self) -> None: ...               # 幂等；allocate() 接管之后什么都不做
```

- 每个 rank 的 `ElectionRecord` 都带 `demand`，第 1 点第二轮收集之后，每个 rank 都知道本副本全部 rank 的需求，`admit()` 因此是纯函数。`elect()` 读 `layout()` 缓存的结果；还没调用 `layout()` 就调用 `elect()` 报 `LifecycleError`。
- 角色由纯函数 `resolve_roles(share_plan, records)` 得出，`admit()` 和 `allocate()` 都用它：

| 本 rank 所在共享集合的代表的 `outcome` | 代表的角色 | 其余成员的角色 |
|---|---|---|
| `writer` | 写入进程 | 等待者，写入进程在本副本 |
| `remote` | 等待者 | 等待者，写入进程在别处 |
| `private` | 私有内存 | 私有内存 |
| 不在共享集合里，有本机部分 | 私有内存 | 不适用 |
| 没有本机部分 | 无 | 不适用 |

- `Election.close()`：写入进程向已连接的客户端发送 `ABORT`，停止监听线程，关闭监听套接字（名字随之消失）；`remote` 关闭到写入进程的连接和已经收到的描述符。

### 6. 交接协议

#### 6.1 套接字与消息

- 套接字是 `AF_UNIX`、`SOCK_SEQPACKET`、带 `SOCK_CLOEXEC`，名字在抽象命名空间（5.2）。一条消息是一个 UTF-8 JSON 对象，不超过 64 KiB，含 `type` 和 `protocol` 两个字段。`HWM_PROTOCOL_VERSION` 取 1；协议版本与存储格式版本分开，格式版本进共享指纹，协议版本只在 `HELLO` 里核对。
- 文件描述符随消息经 `socket.send_fds` 发送，接收端用 `socket.recv_fds(..., flags=socket.MSG_CMSG_CLOEXEC)`，收到的描述符自动带 `FD_CLOEXEC`。

| 类型 | 方向 | 字段 | 说明 |
|---|---|---|---|
| `HELLO` | 等待者发给写入进程 | `protocol`、`deployment_id`、`share_fingerprint`、`expected_digest`、`pid`、`stage_id`、`rank` | 连接后的第一条消息。`expected_digest` 是 `expected_manifest.digest(本机共享部分的全部名字)`。 |
| `STATUS` | 写入进程发给等待者 | `protocol`、`writer_pid`、`writer_stage_id`、`writer_rank`、`sealed`、`bytes_remaining`、`total_bytes`、`file_count` | 对通过核对的 `HELLO` 的回复。`bytes_remaining` 在 `allocate()` 之前等于 `total_bytes`，之后是各内存文件大小减去 `fstat().st_blocks × 512` 之和（小于 0 时取 0），封存之后为 0。 |
| `HANDOFF` | 写入进程发给等待者 | `protocol`、`final_digest`、`seq`、`count`、`files`（本条消息所带描述符对应的 `[{"key": 文件键, "size": 字节数}, ...]`，与描述符一一对应） | 封存之后发送。一条消息最多带 64 个描述符，多于 64 个时按文件键顺序分成 `count` 条，`seq` 从 0 开始。全部发完后写入进程关闭这个连接。 |
| `ABORT` | 写入进程发给等待者 | `protocol`、`writer_pid`、`writer_stage_id`、`reason` | 写入进程在封存之前失败或关闭时尽力发送。进程被杀时没有这条消息，等待者看到的是连接结束。 |

#### 6.2 写入进程的监听线程

- 线程名 `vllm-omni-hwm-listener`，守护线程，从 `elect()` 绑定成功一直运行到 `HostStorage.close()`（`allocate()` 之前失败时到 `Election.close()`）。它用一个 `selectors.DefaultSelector` 处理监听套接字和全部连接，经一个 `socket.socketpair()` 接收主线程的通知（封存完成、关闭）。
- 每个新连接按顺序核对，任何一步不通过就直接关闭连接，不回复，也不计入已连接的等待者，并记一条警告：
  1. `getsockopt(SOL_SOCKET, SO_PEERCRED)` 的 uid 等于本进程的有效 uid，pid 不为 0；
  2. 60 秒内收到 `HELLO`；
  3. `HELLO` 的 `protocol`、`deployment_id`、`share_fingerprint`、`expected_digest` 与自己的相同，`pid` 等于 `SO_PEERCRED` 的 pid。
- 通过的连接立即收到 `STATUS`。已经封存时，紧接着收到 `HANDOFF`，然后连接关闭；还没封存时，连接保留到封存。
- 主线程在 `seal()` 里只做本地操作，最后通知监听线程；向等待者发送 `HANDOFF` 由监听线程完成，主线程不等待任何套接字（总体设计 6.3 的顺序）。发送用非阻塞方式，对方接收队列满时等待可写，不影响其他连接。

#### 6.3 等待者一侧

- 代表在 `elect()` 里建立连接，`allocate()` 时连接和已经收到的描述符从 `Election` 移交给 `HostStorage`。`connect_writer()` 对它做一次非阻塞检查：已经收齐 `HANDOFF` 时通过；否则连接上有 `ABORT`，或者连接在收齐 `HANDOFF` 之前已经结束，报 `AllocationError`。写入进程发完 `HANDOFF` 后关闭连接，所以收齐之后的连接结束是正常的。
- 其余等待者在 `connect_writer()` 里连接，套接字超时 30 秒：
  1. `connect(socket_name)`。`EAGAIN`（监听队列已满）时每 10 ms 重试，直到 30 秒用完。`ECONNREFUSED`（抽象命名空间里名字不存在时也返回它，说明写入进程已经退出）或重试用完时报 `AllocationError`，写明写入进程的 pid、stage 和套接字名。角色在第 1 点已经确定，等待者不在这里改为参选；
  2. `SO_PEERCRED` 的 uid 等于本进程的有效 uid，pid 等于代表的 `ElectionRecord.writer_pid`；不满足时报 `AllocationError`；
  3. 发送 `HELLO`，收到 `STATUS`，核对 `protocol` 和 `writer_pid`；
  4. 保持连接，到 `seal()` 里阻塞接收 `HANDOFF`。接收不设自己的超时，由 stage 的启动超时和 01 第 3 点的确认超时限定。
- 一个 rank 不是等待者时，`connect_writer()` 什么都不做。

#### 6.4 时序

```text
代表（写入进程）        其他副本的代表           本副本的等待者
elect：bind 成功，监听
                        elect：bind 失败，连接
                        HELLO / STATUS（未封存）
第 2 点
                                                 connect_writer：连接，HELLO / STATUS
写入（写共享存储）      写入（共享张量为空操作）  写入（共享张量为空操作）
收尾第 1 到 5 步        收尾第 1 到 5 步          收尾第 1 到 5 步
seal：定稿、封存、通知
监听线程：HANDOFF       seal：接收、核对、绑定    seal：接收、核对、绑定
第 3 点                 第 3 点                   第 3 点
```

### 7. 本机后端的存储

#### 7.1 `allocate(layout, share_plan, elections)`

| 角色 | 本机部分的分配 |
|---|---|
| 写入进程 | 每个文件 `os.memfd_create("vllm-omni-hwm:<文件键>", MFD_CLOEXEC \| MFD_ALLOW_SEALING)`，`os.ftruncate` 到文件大小，以 `MAP_SHARED`、`PROT_READ \| PROT_WRITE` 映射，对映射调用 `madvise(MADV_DONTFORK)`。物理页在写入时才占用。 |
| 等待者 | 不分配。 |
| 私有内存 | 每个文件一个匿名私有映射（`MAP_PRIVATE \| MAP_ANONYMOUS`），布局与内存文件相同，对映射调用 `madvise(MADV_DONTFORK)`。 |

- 每个映射上建一个 `uint8` 的基张量（`torch.frombuffer`），张量视图都是它的切片经 `view(dtype)` 和 `as_strided(shape, stride)` 得到。映射的寿命跟随引用它的张量。
- 全部分配成功之后，`allocate()` 才接管 `elections.local` 的监听套接字、监听线程，或者连接和已经收到的描述符，`Election` 随之变为空。任何一步失败时，已经分配的部分在 `allocate()` 里释放，报 `AllocationError`，`Election` 仍归调用方（01 第 2 节）。
- 写入进程的监听线程在 `allocate()` 之后用这些内存文件计算 `STATUS.bytes_remaining`。

#### 7.2 `view(tensor_key)`

| 张量所在 | 返回 |
|---|---|
| 本机部分，写入进程或私有内存 | 映射上的可写视图。 |
| 本机部分，等待者 | 形状、类型、步长正确的 meta 张量。Loader 对它的拷贝是空操作，记为已写（02 第 7.3、7.5 节）。 |
| 分片部分 | 见第 9.1 节。 |
| 不在布局里 | 报 `WriteError`，写明张量名。 |

`seal()` 之后或 `close()` 之后调用报 `LifecycleError`。

`note_written(tensor_key)` 由 Loader 在写入阶段对每个 Host 驻留张量调用一次（第 9.2 节）。本机部分和私有内存的张量只做名字检查，不做别的事；名字不在布局里时报 `WriteError`；`seal()` 之后调用报 `LifecycleError`。

#### 7.3 写入进程的 `seal(final_manifest)`

```text
1. 对每个文件：从 final_manifest.entries 里 residency 为 "host"、storage_key 在本文件里的条目
   生成头部；与 layout() 时的头部逐字节比较，不同时报 FinalizeError，写明不同的张量和字段
2. 把头部写进可写映射的偏移 0
3. ro_fd = os.open("/proc/self/fd/<fd>", O_RDONLY | O_CLOEXEC)
4. 用 ro_fd 建 MAP_SHARED、PROT_READ 映射，调用 madvise(MADV_DONTFORK)
5. 从只读映射解析头部，按头部的偏移把 final_manifest.live(name) 给出的每个张量对象
   （规范张量和别名）重新绑定到只读映射上（7.5）；之后 live() 里仍有存储地址落在可写映射里的
   对象时报 FinalizeError，写明张量名
6. 撤掉可写映射：丢掉 HWM 持有的可写视图和基张量，执行 gc.collect()，再关闭可写映射的 mmap 对象；
   关闭时报 BufferError（模型里还有别的张量引用可写映射，例如 load_weights 里留下的派生视图）
   或 /proc/self/maps 里仍有这个内存文件 inode 的可写映射时报 FinalizeError
7. fcntl(fd, F_ADD_SEALS, F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL)；失败时报 FinalizeError
8. 关闭可写的描述符 fd，保留 ro_fd
9. 通知监听线程：final_digest = final_manifest.digest(本机共享部分的全部名字)，文件键、大小和 ro_fd
```

- 第 3 步重新打开描述符是必须的：在 6.7 之前的内核上，对加了 `F_SEAL_WRITE` 的内存文件，用可写打开的描述符建 `MAP_SHARED` 映射即使只读也会被拒绝（总体设计 6.3）。写入进程和等待者的映射都用只读描述符建立，因此与内核版本无关。
- 第 7 步由内核再保证一次第 6 步：任何进程里还有这个内存文件的可写共享映射时，`F_SEAL_WRITE` 失败。
- `seal()` 只调用一次，不做集合通信，不等待任何套接字。

#### 7.4 等待者的 `seal(final_manifest)`

```text
1. 在连接上阻塞接收 HANDOFF，直到收齐 count 条（代表在 elect() 里已经收齐的直接使用）；
   收到 ABORT，或者收齐之前连接结束时报 FinalizeError，写明写入进程的 pid 和 stage
2. 对每个描述符核对：fstat().st_uid 等于本进程的有效 uid；fcntl(F_GET_SEALS) 含
   F_SEAL_WRITE、F_SEAL_GROW、F_SEAL_SHRINK、F_SEAL_SEAL；F_GETFL 的访问模式是 O_RDONLY；
   st_size 等于本进程布局里这个文件的大小；文件键与本进程的文件一一对应
3. final_digest 等于本进程的 final_manifest.digest(本机共享部分的全部名字)
4. 用描述符建 MAP_SHARED、PROT_READ 映射，调用 madvise(MADV_DONTFORK)，然后关闭描述符
5. 解析头部，与本进程 layout() 的头部逐字节比较；按名字重新绑定，与写入进程第 5 步是同一个函数
```

任何一步不通过都报 `FinalizeError`，写明哪一项不一致；已经建立的映射在报错前释放。

#### 7.5 重新绑定

`rebind_to_mapping(obj, base, offset)` 是写入进程和等待者共用的函数：

- 对 `final_manifest.live(name)` 里的每个对象，按它自己的大小、步长和相对规范张量的存储偏移，在只读基张量上建一个同类的张量；
- 复制对象的 `__dict__`（保留 `weight_loader` 等属性），用 `torch.utils.swap_tensors` 交换，参数对象的身份不变；
- `swap_tensors` 拒绝交换（对象有弱引用，或者还有别的张量以它为视图的基）时报 `FinalizeError`，写明张量名；
- 等待者的对象原来是 meta 张量，写入进程的对象原来在可写映射上，两种情况走同一段代码。
- 只读映射上的基张量由 `torch.frombuffer` 建立，HWM 屏蔽它对不可写缓冲区发出的警告；对这些张量的任何写入都触发段错误（总体设计 6.3）。

重新绑定之后，共享张量的存储变了；Loader 在 `eval()` 前后核对绑定快照时，以 `seal()` 返回之后的绑定为准（见“本 RFC 依赖的约定”）。

#### 7.6 `ranges(unit)`、`Mapping` 与 `MemoryRange`

```python
@dataclass(frozen=True)
class Mapping:
    mapping_id: str                    # "<kind>:<文件键或组件名>"
    address: int                       # 映射起点，按页对齐
    length: int                        # 映射长度，按页向上取整
    kind: Literal["shared", "private", "shard"]

@dataclass(frozen=True)
class MemoryRange:
    address: int
    length: int
    mapping: Mapping
    read_only: bool                    # shared 为真，private 和 shard 为假
```

- 本机部分的 Unit：取 `unit.unit_id` 在布局里的张量，按地址排序，相邻的合并成一段；同一个 Unit 的张量在布局里相邻（4.2），通常每个文件一段。
- 分片部分的 Unit：每个类型组一段，即本 rank 的范围（4.3）。
- `ranges()` 只在 `seal()` 之后有效，之前调用报 `LifecycleError`。04 把区间向外对齐到页并在同一个 `Mapping` 内合并（总体设计 7.4），对齐不会越出 `Mapping.length`。
- 05 的指标按 `Mapping.kind` 汇总 host 存储的字节数：私有、共享、rank 分片。

### 8. 私有内存

- 布局、视图和 `ranges()` 与共享存储相同，只是映射是匿名私有映射。
- `seal()` 定稿：逐条比较最终清单里落在私有部分的条目与布局（形状、类型、步长、存储键）。`replaced_at_finalize` 为真、最终形状或类型与布局不同的条目（02 第 9 节第 5 步把它留在原处），在新的匿名映射里按最终形状分配，从 `final_manifest.live(name)` 拷入，再重新绑定（7.5），完全落在原范围之内的整页归还给内核。搬入后的字节数大于布局里这个条目的 `nbytes` 时报 `FinalizeError`：容量准入按写入前的布局计算。其余不一致报 `FinalizeError`。
- 私有内存保持可写，`MemoryRange.read_only` 为假。

### 9. 分片后端

#### 9.1 预留与视图

- `allocate()` 为每个分片组件建一个匿名私有映射，大小为组件预留区大小（4.3），带 `MAP_NORESERVE`，调用 `madvise(MADV_DONTFORK)`。只是预留地址空间，物理页在写入时才占用，也只记在本进程的内存 cgroup 上。
- `view(name)`：

| 冷启动路径 | 张量与本 rank 范围的关系 | 返回 |
|---|---|---|
| 瘦路径 | 完全落在范围之外 | meta 张量。不分配，Loader 的写入是空操作，也不读 checkpoint 里对应的数据。 |
| 瘦路径 | 与范围相交 | 预留区上的完整视图。 |
| 先全量后裁剪 | 任意 | 预留区上的完整视图。 |

- 与范围相交但不完全落在范围之内的张量是边界张量。每个 Unit 的每个类型组最多有两个边界张量。需要变换的边界张量由相邻的两个 rank 各自完整处理一次（总体设计 6.4）。

#### 9.2 瘦路径的页归还

- 记号：瘦路径组件每个 Unit 的每个组 g 有本 rank 的范围 R_g（4.3）。与 R_g 相交、又不完全落在 R_g 之内的张量是边界张量。组的范围外区域 Z_g 是这个组的预留页里不含 R_g 任何字节的整页。完全落在范围之外的张量是 meta 张量，从不写入，所以 Z_g 里只有边界张量写进的字节，这些字节对本 rank 没有用处。
- 一个边界张量 t 的可归还页，是与 t 的范围外部分相交、又完全落在 Z_g 之内的整页。包含 R_g 起点或终点的页不在 Z_g 里，永远不归还，所以范围之内的字节不受影响。
- 归还由 `HostStorage.note_written(name)` 触发。02 的 `WriteTracker` 在一个张量的累计写入元素数第一次达到它的元素数时调用一次（02 第 7.5 节），所以触发点由写入的字节数决定，与时间无关（总体设计 6.4：归还按已写字节数的阈值触发）：

```text
note_written(name)：
  name 不是瘦路径的边界张量，或者已经记过：返回
  pending += t 的范围外字节数；t 的可归还页加进待归还列表
  pending ≥ 释放阈值：
    对待归还列表里的页调用 madvise(MADV_DONTNEED)，清空列表，pending = 0
```

- `note_written` 在 Loader 的写入线程里同步执行，不启动线程，不做集合通信，每个 rank 独立执行。归还的页在 `MAP_PRIVATE | MAP_ANONYMOUS` 映射里，再次读取时得到零。
- 一个边界张量只在全部写完之后才可能被归还（总体设计 6.4：跨越范围边界的张量先完整写入，再归还范围之外的页），所以写入入口在写的过程中读回目标时不会读到被归还的页。
- 上限：每次 `note_written` 返回时 `pending` 小于释放阈值。任何时刻，超出稳态（各组覆盖 R_g 的整页）的驻留字节，等于待归还列表里的字节（小于释放阈值）加上已经开始写、还没写完的边界张量的范围外字节。02 的写入入口对瘦路径组件按 Unit 的顺序产出张量（总体设计 6.4），每个 Unit 的每个类型组最多两个边界张量，所以同一时刻没写完的边界张量不超过两个，后一项不超过最大两个张量，冷启动的额外占用严格小于“释放阈值 + 最大两个张量”（总体设计 6.4、10）。合并参数或堆叠参数分几次写入时也在这个上限之内。
- `seal()` 对待归还列表里的页和所有边界张量的可归还页做最后一次归还；没有被 `note_written` 记过的边界张量（量化方法声明可以缺少的张量）也在这里归还。

#### 9.3 先全量后裁剪

- 写入期间预留区里的每个张量都完整写入，占用一份完整的组件。`note_written` 对这些张量只做名字检查。
- `seal()` 先按最终清单检查：`replaced_at_finalize` 为真、形状或类型改变的条目（02 第 9 节第 5 步把它留在原处）所在的 Unit，用 `flatten_shard_unit` 按最终形状重新计算布局；任何一个 Unit 重新计算后的补零大小超过写入前的大小时报 `FinalizeError`，因为容量准入和加速卡上的固定槽位都按写入前的布局计算。重新布局的 Unit 放进一个新的匿名映射，从 `live()` 拷入后重新绑定（7.5），原来的整页归还给内核。
- 然后对每个 Unit 的每个组归还 Z_g 的全部整页。之后的状态与瘦路径相同：预留区里只有覆盖本 rank 范围的页驻留。

#### 9.4 定稿后的读取

- `seal()` 之后，与本 rank 范围相交的分片张量仍绑定在预留区上，范围之外的字节读出来是零，不会触发段错误；瘦路径下完全落在范围之外的张量仍是 meta 张量。02 在 `eval()` 期间把全部分片 Unit 的参数换成零元素占位，`eval()` 读不到它们（02 第 9 节）。
- `ranges(unit)` 给出本 rank 的范围，每个组一段，供 04 登记和 H2D。

### 10. 容量准入

`admit(topology, share_plan, elections, layout)` 是纯函数，只读 `topology.records`、`share_plan`、`elections.records`：

```text
对每台机器 m（按 LocalRecord.machine 分组的本副本 rank）：
新增占用(m) = Σ 写入进程在本副本的共享集合，写入进程的 demand.local_bytes
           + Σ 角色为私有内存的 rank，demand.local_bytes
           + Σ 所有 rank，demand.shard_bytes + demand.cold_extra_bytes
           + Σ 所有 rank，demand.staging_bytes
           + Σ 写入进程和私有内存的 rank，demand.transient_bytes
可用内存(m) = min(m 上各 rank 的 LocalRecord.available_bytes，m 上各代表的 ElectionRecord.available_bytes)
           − Σ 写入进程在别处、remote_sealed 为假的共享集合，remote_bytes_remaining
新增占用(m) > 可用内存(m) 时报 HostCapacityError
```

- 写入进程在别处的共享集合不计 W；它已经写入的部分已经反映在读到的可用内存里，还没写入的部分从可用内存里扣除。
- 一台机器上所有读数都是 `None` 时，跳过这台机器并记一次警告。
- `HostCapacityError` 写明机器的 hostname、新增占用的每一项、可用内存和读数来源（`memory_source`），`fix` 给出 `--diffusion-offload distributed-layerwise` 或减少这台机器上的副本数。
- 所有 rank 的输入相同，所以同时报错（01 第 7.2 节）。这时还没有分配 host 存储，也没有锁页。

### 11. 生命周期、关闭与 fork

**`close()`**。幂等，只影响本进程：

| 角色 | `close()` 做的事 |
|---|---|
| 写入进程 | 封存前：向已连接的等待者发 `ABORT`。停止监听线程，关闭监听套接字（名字消失）和所有连接，关闭 `ro_fd`。 |
| 等待者 | 关闭连接和还没关闭的描述符。 |
| 分片 | 丢掉待归还列表，预留区随映射一起解除。 |
| 全部 | 丢掉 HWM 持有的基张量和视图，映射在最后一个引用它的张量释放时解除。共享内存文件在最后一个持有它的进程退出或解除映射时由内核释放。 |

- 04 的关闭顺序在关闭 host 存储之前把参数改指向占位（总体设计 7.5），所以 `close()` 之后的 `gc.collect()` 解除本进程的全部映射。单进程执行器和 inline 时，测试在 `close()` 和 `gc.collect()` 之后断言 `/proc/self/maps` 里没有 HWM 的映射，`/proc/self/fd` 里没有内存文件。
- 写入进程关闭不影响已经绑定的等待者：它们持有自己的只读映射。

**fork**。

- 内存文件用 `MFD_CLOEXEC` 创建，`ro_fd` 用 `O_CLOEXEC` 打开，套接字带 `SOCK_CLOEXEC`，收到的描述符带 `MSG_CMSG_CLOEXEC`，所以 fork 后 exec 的子进程不继承它们。
- 所有映射建立后都调用 `madvise(MADV_DONTFORK)`，fork 出的子进程里没有这些映射。
- 第一次调用 `create_host_weight_manager` 时，`host/forksafe.py` 调用一次 `os.register_at_fork(before=..., after_in_parent=..., after_in_child=...)`。进程级的登记表记下全部内存文件描述符、监听套接字、已接受的连接、等待者的连接和等待者已经收到的描述符。登记表的锁只在增删条目时短暂持有，不跨越任何阻塞的套接字操作，所以 `before` 不会因为监听线程而卡住 fork。`before` 取得登记表的锁，`after_in_parent` 释放；`after_in_child` 在子进程里关闭登记表里的全部描述符，把每个 host 存储对象标记为已失效（之后任何调用报 `LifecycleError`），再释放锁。子进程里没有监听线程。
- 不经过 Python 的 fork（C 扩展直接调用 `fork()`）不触发这个钩子，是已知限制（总体设计 13）。

### 12. 失败处理

| 情况 | 结果 | 错误 |
|---|---|---|
| 本机键某一项读不到 | 这个 rank 使用私有内存，启动日志写明原因 | 无 |
| 代表绑定时出现其他错误、uid 不一致、重试用完 | 本组使用私有内存 | 无 |
| 写入进程在封存之前失败或被杀 | 已连接的等待者收到 `ABORT` 或连接结束，所在副本启动失败 | 等待者在 `connect_writer` 报 `AllocationError`，或在 `seal()` 报 `FinalizeError`，写明写入进程的 pid 和 stage |
| 写入进程在封存之后退出 | 已绑定的 worker 不受影响；名字消失，之后加入的 worker 重新选举 | 无 |
| 写入进程在封存之后、部分等待者拿到描述符之前退出 | 已经收到描述符的等待者照常绑定；没收到的等待者所在副本启动失败，不改变角色 | `FinalizeError` |
| 等待者失败 | 写入进程只关闭这个连接，继续构建和监听 | 无 |
| 写入进程停住不动 | 等待者阻塞到 stage 的启动超时或 01 第 3 点的确认超时 | `ConfirmationTimeoutError` 或执行器的启动超时 |
| 连接的 uid、部署标识、共享指纹、预期清单摘要或协议版本不一致 | 写入进程关闭连接，不回复；代表在 `elect()` 里遇到时本组使用私有内存，其余等待者在 `connect_writer()` 里遇到时启动失败 | 其余等待者报 `AllocationError` |
| 交出的描述符的 uid、封存位、大小、清单摘要或头部不一致 | 等待者不绑定 | `FinalizeError` |
| 封存失败，或还有张量引用可写映射 | 写入进程启动失败，等待者随之失败 | `FinalizeError` |
| 内存文件或映射分配失败 | 本 rank 启动失败 | `AllocationError` |

本地扩散副本依次启动（`StageRuntime._init_group_key_override` 把本地扩散副本放进同一个顺序执行的组），第一个副本的代表成为写入进程，它在封存前失败时整个部署启动失败。只有 head 的本地副本和同机的 headless 副本、或两个 headless 启动器并发启动时，一个启动单元里写入进程在封存前的失败才会让另一个启动单元启动失败（总体设计 6.3）。

### 13. 执行器与部署形态

| 形态 | 行为 |
|---|---|
| 单进程执行器、inline | 只有一个 worker，它就是代表。inline 时监听线程运行在编排进程里。私有内存、共享映射、监听套接字和描述符都由 `close()` 释放，不依赖进程退出。 |
| 多进程执行器 | 一个副本的 worker 都在本机，内容相同的构成一个共享集合。 |
| Ray | 一个副本可以跨机器；共享指纹含本机键，每台机器上的 rank 各成一组、各自参选。不同副本的 actor 落在同一台机器上时照常共享。 |
| headless | 与本地副本相同，部署标识来自注册回复（3.3）。 |
| 自定义执行器 | 构造 `DiffusionWorker` 时自动适用。 |
| 工具和测试直接构造 `od_config` | 没有部署标识，0 号 rank 生成副本内的标识；单 worker 的副本使用私有内存。 |

### 14. 启动日志与故障注入点

- HWM 用 logger `vllm_omni.diffusion.weight_stack.host` 输出 05 第 15.2 节的两个事件，事件名和字段按 05 的表：
  - `host_share`：每个有本机部分的组件一条，字段 `component`、`shared`、`reason`、`role`（`writer`、`waiter`、`private`，由 `resolve_roles` 的结果映射）、`writer_pid`、`socket`（`\0` 写成 `@`）。
  - `host_admission`：每台机器一条，字段 `machine`、第 10 节新增占用的每一项、`available_bytes`、`memory_source`。
- 03 的 PR 先于 05-6 合入，期间用这个 logger 按上面的事件名和字段输出；05-6 把这些调用点改成 `observability.log_event`，并在 `allocate()` 和 `seal()` 返回之前对每个新建的 `Mapping` 调用 `observability.record_host_mapping(mapping)`、在 `HostStorage.close()` 里调用 `forget_host_mapping(mapping)`（05 第 16.2 节）。HWM 不读 host 策略（05 第 7 节）。
- 每个组件使用的后端和冷启动路径由 04 的 `component_plan` 事件输出，03 不重复。
- 03 在模块内部增加的故障注入点，命名规则和 `FaultPlan` 与 01 第 10 节相同：

| 注入点 | 位置 |
|---|---|
| `hwm_elect_bind` | `elect()` 绑定之前。 |
| `hwm_writer_listen` | 监听线程处理一个新连接之前。 |
| `hwm_allocate_memfd` | 创建第一个内存文件之前。 |
| `hwm_seal_reopen` | `seal()` 第 3 步之前。 |
| `hwm_seal_add_seals` | `seal()` 第 7 步之前。 |
| `hwm_handoff_send` | 监听线程发送第一条 `HANDOFF` 之前。 |
| `hwm_waiter_bind` | 等待者 `seal()` 第 4 步之前。 |
| `hwm_page_return` | `note_written` 第一次调用 `madvise(MADV_DONTNEED)` 之前。 |

## 与其他子 RFC 的接口

### 本 RFC 提供

| 提供的内容 | 使用方 | 约定 |
|---|---|---|
| `weight_stack.host.create_host_weight_manager(ctx: StartupContext) -> HostWeightManager` | 01 | 由 01 的 `EntryPointFactory` 经 `importlib` 取得。非 Linux 平台报 `StackUnavailableError`。 |
| `LocalRecord`、`SharePlan`、`Election`、`ElectionRecord`、`HostLayout`、`MemoryRange`、`Mapping` | 01、04 | 字段见设计第 2、4、5、7 节。`LocalRecord`、`ElectionRecord`、`HostDemand` 可以 pickle。`Election.record()`、`Election.close()` 按 01 第 2 节实现，`close()` 可以多次调用。 |
| `HostWeightManager` 的七个方法 | 01 | `deployment_id()` 按 3.6；`layout()` 只读预期清单的条目（4.1）；`plan_fingerprint` 只依赖配置和并行布局（4.5）；`share_plan()`、`admit()` 是纯函数；`elect()` 每个 rank 都调用，非代表返回空的 `Election`；`allocate()` 成功时接管 `Election`，失败时不接管；超出容量时 `admit()` 报 `HostCapacityError`。 |
| `HostLayout.bind_groups(groups)` | 01 | 第 1 点之后调用；组的大小或位置与规划的推算不同时报 `GroupConfirmationError`。 |
| `HostStorage` 的六个方法：01 第 2 节的五个，加上 `note_written(tensor_key: str) -> None` | 01、02、04 | `view()` 按 7.2 和 9.1，对等待者的共享张量和瘦路径的范围外张量返回形状、类型、步长正确的 meta 张量，Loader 对它的拷贝是空操作并记为已写；`note_written()` 在写入阶段由 Loader 对每个 Host 驻留张量在全部写完之后调用一次，不做集合通信，瘦路径据此同步归还范围之外的页（9.2），其他后端只做名字检查；`seal()` 在每个 worker、每种后端上被调用一次，不做集合通信，私有内存和分片在其中定稿，共享存储在其中封存、交接并用 `FinalManifest.entries` 和 `live()` 重新绑定到只读映射；`connect_writer()` 在第 2 点之后调用；`ranges()` 在 `seal()` 之后有效；`close()` 幂等，只影响本进程。 |
| `flatten_shard_unit(tensors, group_size) -> ShardUnitLayout` | 04 | 分片 Unit 的拉平和补零规则的唯一实现（4.3），04 用它决定 AllGather 结果在块环缓冲区里的布局。 |
| `OmniDiffusionConfig.deployment_id` | 01 | `init=False`，没有下发时为 `None`；`HostWeightManager.deployment_id()` 读它。 |

### 本 RFC 依赖的约定

| 子 RFC | 约定 |
|---|---|
| [01](./01-startup-transaction.md) | `interfaces.py` 的 Protocol 和数据类型；驱动按 01 第 6、7 节的顺序调用 `local_record()`、`deployment_id()`、`layout()`、`bind_groups()`、`share_plan()`、`elect()`、`admit()`、`allocate()`、`connect_writer()`；`allocate()` 之前失败时驱动调用 `Election.close()`；第 2 点成功之后才调用 `connect_writer()`；`StartupContext.faults` 带 `FaultPlan`；`run_unified_startup(ctx, factory)` 接受测试提供的工厂，03 的多进程测试用它把 01 的 Loader 和 Offloader 替身换成 `host/conftest.py` 里带 Host 驻留 Unit 和写入的替身。 |
| [02](./02-unified-loader.md) | `WriteTracker` 在一个 Host 驻留张量的累计写入元素数第一次达到它的元素数时，在写入线程里调用一次 `storage.note_written(name)`（目标为 meta 张量时同样调用）；`ManifestEntry` 带每个张量的最终形状、类型、步长和 `nbytes`；`ExpectedManifest.digest(names)` 和 `FinalManifest.digest(names)` 对同一组名字确定；`FinalEntry.storage_key` 等于 `view()` 用的规范名；`final_manifest.live(name)` 给出规范张量和别名的对象；收尾第 5 步只在最终形状和类型与预期条目相同时把替换的张量拷进 `view(name)`，否则把张量留在原处，由 `seal()` 搬入（第 8、9.3 节）；`eval()` 前后的绑定快照以 `seal()` 返回之后的绑定为准；`seal()` 之前完成收尾里的全部集合通信。 |
| [04](./04-unified-offloader.md) | `PlacementPlan.host_request()` 返回第 4.1 节结构的对象，`staging_bytes` 在本 worker 有 Host 驻留的 Unit 时为暂存区大小；`Unit.unit_id` 与 `host_request().units` 里的相同；规划的 Unit 划分不依赖 `LocalRecord` 的 pid、可用内存和空闲显存；启动事务把 host 存储登记在 `ResourceKind.HOST_STORAGE` 下，关闭时先把参数改指向占位再关闭 host 存储；内存登记按 `MemoryRange` 和 `read_only` 做；AllGather 结果的布局用 `flatten_shard_unit`，组的对齐由 04 自己决定；`cold_start` 只在组件的 `shapes_final` 为真并且 `finalize_replaced` 为空时取 `thin`（4.1）；先全量后裁剪的组件在收尾里重新布局时，`commit()` 按最终清单和 `flatten_shard_unit` 重建这个 Unit 在块环和分片槽位里的视图。 |
| [05](./05-configuration.md) | `host_share`、`host_admission` 事件的字段（05 第 15.2 节）；`observability.record_host_mapping()`、`forget_host_mapping()` 和 `log_event()` 由 05-6 提供并接到第 14 节的调用点；指标里 host 存储的字节数按 `Mapping.kind` 区分私有、共享和 rank 分片。 |

## 代码组织

新增文件：

```text
vllm_omni/diffusion/weight_stack/host/
  __init__.py        导出 create_host_weight_manager 和“本 RFC 提供”表里的类型
  sysapi.py          SysApi：/proc 与 cgroup 根目录、geteuid、memfd_create、madvise
  record.py          LocalKey、LocalRecord、本机键读取、cgroup 定位、可用内存
  layout.py          HostRequest 等 Protocol、HostLayout、LocalFileLayout、ShardComponentLayout、
                     HostDemand、flatten_shard_unit、safetensors 头部的生成和解析、规划指纹
  share.py           共享指纹、套接字名、ShareSet、SharePlan、share_plan()、resolve_roles()
  protocol.py        消息类型、编码、版本、SO_PEERCRED 读取、描述符分批
  election.py        Election、ElectionRecord、elect()、监听线程
  local_storage.py   内存文件、私有映射、view、写入进程和等待者的 seal、rebind_to_mapping、ranges
  shard_storage.py   预留区、瘦路径的 note_written 和页归还、先全量后裁剪、ranges
  storage.py         CompositeHostStorage
  admission.py       admit()
  forksafe.py        进程级登记表和 os.register_at_fork
  manager.py         HostWeightManagerImpl、create_host_weight_manager
tests/diffusion/weight_stack/host/
  conftest.py        假的 /proc 和 cgroup 树、SysApi 替身、多进程夹具
  test_record.py  test_deployment_id.py  test_layout.py  test_share_plan.py
  test_protocol.py  test_election_multiproc.py  test_local_storage.py
  test_shard_storage.py  test_admission.py  test_lifecycle_multiproc.py
  test_host_e2e.py
```

修改的文件：

| 文件 | 改动 |
|---|---|
| `vllm_omni/engine/stage_runtime.py` | `StageRuntime.__init__` 生成部署标识；`_initialize_local_diffusion_replica` 传给 `launch_diffusion_stage_replica`；`DistStageRuntime._start_omni_master_server` 构造 `OmniMasterServer` 时传入。 |
| `vllm_omni/engine/stage_engine_startup.py` | `OmniMasterServer` 保存并在注册回复里带上部署标识；`StageRegistrationResponse` 和 `register_stage_with_omni_master` 读它；`launch_diffusion_stage_replica` 和 `launch_headless_diffusion_replica` 写进 `od_config`。 |
| `vllm_omni/engine/stage_init_utils.py` | `initialize_diffusion_stage` 增加 `deployment_id` 参数并写进 `od_config`。 |
| `vllm_omni/diffusion/data.py` | 增加 `OmniDiffusionConfig.deployment_id`（3.4）。 |
| `vllm_omni/config/omni_config.py` | `_DIFFUSION_INTERNAL_FIELDS` 增加 `"deployment_id"`。 |

这些修改不读总开关。`legacy` 下没有代码读部署标识，注册回复多出的键不被旧的读取方使用，`legacy` 下行为不变，所以不需要登记表的位置文件。删除 `legacy` 时本 RFC 没有要删的文件（见“删除 legacy 时”）。

## 实现拆分

本 RFC 的 PR 都依赖 01-1 到 01-7 全部合入，下表不再重复。除 03-9 外，测试里的 Loader 和 Offloader 用 01 的测试替身，或 `host/conftest.py` 里在其基础上增加 Host 驻留 Unit 和写入的替身，不依赖 02、04 的 PR。01 的测试替身里的 host 存储实现 `note_written()`，什么都不做。每个 PR 只改变 `unified` 下的行为，`legacy` 下的行为和已有测试不变，不修改已有测试文件。本 RFC 不承担共存期必改清单的任何一行，也没有自己读总开关的分发点。

03-9 只含测试，是 Job 3 的收尾：它在 02-5 和 04 实现 `prepare_materialization()`、`commit()`、内存登记的 PR 合入之后合入，02、04 的任何 PR 都不依赖它，所以依赖图没有环；Job 3 的其余 PR 与 Job 2、Job 4 并行（umbrella 的依赖图）。04 用到的 `flatten_shard_unit` 和 `HostRequest` 协议在 03-3 里。

| PR | 内容 | 主要文件 | 测试 | 依赖 | 大小 | 必改行 |
|---|---|---|---|---|---|---|
| 03-1 | 包骨架、`SysApi`、`LocalKey`、`LocalRecord`、本机键读取、cgroup 定位、可用内存、`create_host_weight_manager` 和 `local_record()`。 | `host/__init__.py`、`sysapi.py`、`record.py`、`manager.py` | `test_record.py` | 无 | M | 无 |
| 03-2 | 部署标识的生成和下发，`OmniDiffusionConfig.deployment_id` 和 `_DIFFUSION_INTERNAL_FIELDS`，`deployment_id()`。 | `stage_runtime.py`、`stage_engine_startup.py`、`stage_init_utils.py`、`data.py`、`omni_config.py`、`manager.py` | `test_deployment_id.py`；已有的 `tests/config/test_omni_config.py` 不修改、照常通过 | 03-1 | S | 无 |
| 03-3 | `HostRequest` Protocol、本机后端和分片后端的布局、safetensors 头部、`flatten_shard_unit`、共享资格、规划指纹、`HostDemand`、`bind_groups`、`layout()`。 | `layout.py`、`manager.py` | `test_layout.py` | 03-1 | M | 无 |
| 03-4 | 共享指纹、套接字名、`share_plan()`、`resolve_roles()`、协议消息、`elect()`、监听线程、`Election`、`ElectionRecord`。 | `share.py`、`protocol.py`、`election.py` | `test_share_plan.py`、`test_protocol.py`、`test_election_multiproc.py` | 03-3 | L | 无 |
| 03-5 | 本机后端和私有内存：`allocate()` 的本机部分、`view()`、写入进程和等待者的 `seal()`、`rebind_to_mapping`、`connect_writer()`、`ranges()`、`close()`。 | `local_storage.py`、`storage.py`、`manager.py` | `test_local_storage.py` | 03-4 | L | 无 |
| 03-6 | 分片后端：预留区、瘦路径的视图、`note_written()` 和同步页归还、先全量后裁剪、`ranges()`。 | `shard_storage.py`、`storage.py` | `test_shard_storage.py` | 03-3 | M | 无 |
| 03-7 | 容量准入 `admit()`。 | `admission.py`、`manager.py` | `test_admission.py` | 03-4 | S | 无 |
| 03-8 | fork 处理、故障注入点，以及经 01 驱动的多进程生命周期和故障测试。 | `forksafe.py`，各模块的注入点 | `test_lifecycle_multiproc.py` | 03-5、03-6、03-7 | M | 无 |
| 03-9 | 与真实 Loader 和 Offloader 一起的 GPU 端到端测试。 | 只有测试 | `test_host_e2e.py` | 03-8、02-5，以及 04 里实现 `prepare_materialization()`、`commit()` 和内存登记的 PR | M | 无 |

```text
01-1..01-7
  03-1
    03-2
    03-3
      03-4
        03-5
        03-7
      03-6
          03-8（依赖 03-5、03-6、03-7）
            03-9（另依赖 02-5 和 04 的相应 PR）
```

## 测试计划

测试放在 `tests/diffusion/weight_stack/host/`。除特别写明的以外，都是 L1 测试，标记 `core_model`、`diffusion` 和 `cpu`，不需要加速卡，也不调用 `torch.distributed`。多进程测试用 `multiprocessing` 的 spawn 方式启动进程，标记同样是 L1。

**本机记录**（`test_record.py`）。在 `tmp_path` 里构造假的 `/proc/self/cgroup`、`/proc/self/mountinfo`、cgroup 目录和 `/proc/meminfo`，由 `SysApi` 指向它们：

- cgroup v2：多级限额取最小值；某一级 `memory.max` 为 `max`；全部没有限额时取 `MemAvailable`；`memory.stat` 里 `file` 很大而 `inactive_file`、`active_file` 很小时只计后两者；限额不小于 `MemTotal` 时按没有限额处理；用量大于限额时取 0。
- cgroup v1：`memory.limit_in_bytes` 的“没有限额”值、`total_inactive_file` 和 `total_active_file`；混合模式下选 v1。
- cgroup 命名空间：挂载条目的 root 不是 `/` 时路径换算正确；路径不在 root 之下时本机键读不到。
- 两个假容器里路径字符串相同、目录不同时，`memory_cgroup` 不同。
- boot_id、命名空间文件、`memfd_create` 任何一项失败时 `local_key` 为 `None`，`unreadable` 写明原因。

**部署标识**（`test_deployment_id.py`）：

- 两次构造 `StageRuntime` 得到不同的 32 字符标识；`DistStageRuntime` 与 `OmniMasterServer` 使用同一个值。
- `initialize_diffusion_stage` 和 `launch_diffusion_stage_replica` 的两支在启动进程之前写入 `od_config.deployment_id`：测试把 `create_diffusion_client` 和 `StageDiffusionProcManager` 换成记录参数的替身，断言收到的 `od_config` 已带部署标识。
- 注册回复带部署标识；旧版本的回复没有这个字段时为 `None`；`launch_headless_diffusion_replica` 写进 `od_config`。
- 经 `OmniDiffusionConfig.from_kwargs` 和 YAML 的 stage 配置传入 `deployment_id` 时，构造配置报 `TypeError`，与传入 `ray_worker_env` 相同；字段在 pickle 往返、`copy.deepcopy` 之后保留；已有的 `test_diffusion_config_field_classification_covers_current_fields` 不修改、照常通过。
- `deployment_id()` 的两种返回。

**布局**（`test_layout.py`）：头部可以被 safetensors 的格式规则解析，`8 + H` 是 8 的倍数，偏移按名字排序且连续；同一重复块的张量相邻；非连续步长写进元数据；规划指纹只随布局变化，改变 `LocalRecord` 的 pid 和可用内存时不变；三种 `plan_fingerprint` 形式；`flatten_shard_unit` 的补零、对齐和本 rank 范围；`bind_groups` 在组大小或位置不同时报错。

**共享计划与协议**（`test_share_plan.py`、`test_protocol.py`）：相同共享指纹的 rank 分到一组、代表是最小 rank；本机键不同或 `plan_fingerprint` 带 `/private` 的 rank 不进组；`deployment_scoped` 为假且组只有一个 rank 时使用私有内存；消息编码和长度上限；描述符超过 64 个时分批，接收端按 `seq` 收齐。

**选举与交接，多进程**（`test_election_multiproc.py`、`test_local_storage.py`）：

- 两个进程同时参选同一个名字，恰好一个成为写入进程。
- 写入进程封存之前和之后连接的等待者都拿到描述符，绑定后字节与写入进程相同，存储地址在“`eval()`”替身前后不变。
- 一个部署的 k 个 worker（k 取 1、2、4）共享一份：机器上 `/proc/meminfo` 的 `Shmem` 增量约为一份 W，不随 k 增长。
- 两个部署（两个不同的部署标识）跑相同的布局：各自有写入进程，互不连接；关闭一个部署，另一个的映射和字节不受影响。
- 部署标识、共享指纹、预期清单摘要或协议版本不一致的连接被写入进程关闭。
- uid 不一致：用 `SysApi` 让写入进程看到的有效 uid 不同，连接被拒绝；名字被“另一个 uid”占用时代表使用私有内存，不绑定。以 root 运行时另有一个用例在子进程里 `os.setuid` 成为真实的另一个用户。
- 封存之后，另一个进程经 `/proc/<pid>/fd/<n>` 以可写方式打开文件：`write` 和可写的 `MAP_SHARED` 映射都失败。
- 权重大于 `/dev/shm` 的容量时正常分配，`/dev/shm` 的用量不变（`os.statvfs`）。
- 模型里留下一个引用可写映射的派生视图时，`seal()` 报 `FinalizeError`，不加封存。
- 代表在 `elect()` 时写入进程已经封存：代表在 `elect()` 里收齐描述符，写入进程关闭连接之后 `connect_writer()` 照常通过，`seal()` 绑定成功。
- 监听队列已满时，等待者在 `connect_writer()` 里重试并连上；写入进程已经退出时立即报 `AllocationError`。

**分片后端**（`test_shard_storage.py`）：

- 瘦路径下范围外的张量得到 meta、范围内和边界张量得到视图；`flatten_shard_unit` 的结果与 `_shard_and_pin` 的段长一致。
- 按张量逐个写入一个合成组件，每写完一个张量调用 `note_written`，每次调用之后用 `/proc/self/smaps` 的驻留量断言额外占用严格小于“释放阈值 + 最大两个张量”；用记录调用的 `SysApi` 断言归还的页都在 Z_g 之内、不含范围起点和终点所在的页，范围之内的字节在 `seal()` 之后与写入的值相同。
- 以极快的速度写入（不让出线程）时上一条的断言同样成立：归还与时间无关。
- 合并参数分几次写入、期间写同一个组的另一个边界张量时，额外占用仍小于“释放阈值 + 最大两个张量”。
- `seal()` 之后只剩覆盖本 rank 范围的页；没有被 `note_written` 记过的边界张量在 `seal()` 里归还；`seal()` 之后调用 `note_written` 报 `LifecycleError`。
- 先全量后裁剪之后的驻留量与瘦路径相同；形状改变的替换张量重新布局，补零大小超过原大小时报 `FinalizeError`；瘦路径组件带 `replaced_at_finalize` 条目时 `layout()` 报 `PreflightError`。

**容量准入**（`test_admission.py`）：第 10 节公式的每一项；写入进程在别处时不计 W、扣除未写入的字节；多台机器分组；读数为 `None` 时跳过；所有 rank 对相同输入得到相同结论。

**生命周期与故障，多进程**（`test_lifecycle_multiproc.py`）。用 01 的 `run_unified_startup` 驱动真实的 HWM，Loader 和 Offloader 是 `host/conftest.py` 的替身：它们的规划把一个合成组件放在 Host 上，写入时把确定的字节写进 `view()` 返回的张量：

- 写入进程在封存之前被 `kill -9`：等待者所在副本启动失败，报错写明写入进程的 pid；所有进程退出后 `Shmem` 回到启动前。
- 写入进程在封存之后被 `kill -9`：已绑定的等待者照常读取；之后启动的副本重新选举、成为新的写入进程。
- 对一个部署的全部进程 `kill -9` 之后，`Shmem` 回到启动前。OOM 终止进程的方式同样是 SIGKILL；有可写的 cgroup 时，另有一个用例在内存限额很小的 cgroup 里触发真实的 OOM。
- 写入进程 fork 出的 Python 子进程里没有内存文件描述符，`/proc/<子进程>/maps` 里没有 HWM 的映射；父进程 `close()` 之后，子进程还活着时套接字名已经消失。
- 几个副本互为对方某个共享集合的等待者时启动不死锁。
- 每个 03 注入点的 `raise` 和 `hang`：失败在预期的阶段报出，不留下映射和描述符。
- 单进程执行器方式下，`close()` 和 `gc.collect()` 之后本进程没有 HWM 的映射、描述符和线程。

**端到端**（`test_host_e2e.py`，03-9）。L2，标记 `core_model`、`diffusion`，`@hardware_test(res={"cuda": "L4"}, num_cards=2)`：一个小 DiT 在两个单卡副本上以 `layerwise` 启动，两个副本共用一份共享内存，输出与只启动一个副本时相同；`distributed-layerwise` 在两卡上每个 rank 只持有 `W / 2`。

## 验收标准

- 同一个部署在一台机器上的 worker 数从 1 增加到 k，`Shmem` 的增量不随 k 增长；同一台机器上两个单卡副本属于同一个部署时共用一份，headless 副本并发启动时只有一个写入进程。
- 同一个部署里两个 stage 的 worker 在同一台机器上、整套权重的内容身份和规划指纹都相同时，共用一份。
- 同一台机器上两个部署跑同一个模型，彼此不共享、不连接；一个部署退出，另一个不受影响。
- 部署标识、共享指纹或 uid 不一致的连接被拒绝；套接字名被其他 uid 的进程占用时，等待者不绑定。
- 权重大于 `/dev/shm` 时正常加载，`/dev/shm` 的用量不变；定稿之后任何进程对内存文件的写入和可写映射都被拒绝。
- 等待者绑定后，已绑定的字节和存储地址在 `eval()` 前后不变。
- 分片后端稳定后每个 rank 占 `W / N`（含补零和页的取整）；瘦路径冷启动的额外占用在任何写入速度下都严格小于“释放阈值 + 最大两个张量”。
- 对任意一个 worker 执行 `kill -9` 后，其他副本的映射和服务不受影响；对一个部署的全部 worker 执行 `kill -9` 或 OOM 终止后，机器的共享内存用量回到这个部署启动前。
- 写入进程在封存前被杀，等待者所在副本启动失败，没有残留。
- 写入进程所在的副本退出后，已绑定的副本照常服务；之后加入的副本重新选举、重新构建，并能正常服务。
- 几个副本互为对方某个共享集合的等待者时，启动不死锁。
- 重复块以外的组件单机放不下时，规划阶段报错并指出组件；这项检查由 04 的规划承担，03 的测试不重复。
- worker 之后 fork 出的 Python 子进程不持有内存文件和监听套接字。
- 超出 Host 容量时，所有 rank 在分配之前同时报 `HostCapacityError`，报错写明每一项。
- `legacy` 下现有测试全部通过；03 的 PR 不修改已有测试文件。

## 删除 legacy 时

HWM、部署标识和 `OmniDiffusionConfig.deployment_id` 都是新栈的一部分，删除 `legacy` 时保留。现有实现里与 host 存储有关的代码随它们所在的后端和加载路径一起删除，由承担这些删除的子 RFC 执行：顺序、逐层、分布式逐层三个后端里的每 worker 锁页副本、`cpu_shards` 和 `rank_local_mmap` 方式（[04](./04-unified-offloader.md)），以及现有加载路径里对 Host Weight Runtime 的接入：`vllm_omni/diffusion/model_loader/host_weight_loader.py`、`host_weight_plan.py`、`host_weights/` 和 `diffusers_loader.py` 里对它们的调用（[02](./02-unified-loader.md)）。Host Weight Runtime 包本身不改动（总体设计 1.2）。本 RFC 自己没有 legacy 一支，也没有登记表里的行。

## 风险

| 风险 | 影响 | 处理 |
|---|---|---|
| 共享内存的物理页落在写入进程所在的 NUMA 节点上 | 另一路 CPU 上的加速卡跨路做 H2D | 按总体设计 13 处理：门槛矩阵包含双路 8 卡机器，记录跨路带宽。 |
| 容器的 seccomp 策略禁止 `memfd_create` 或封存 | 这台机器上不共享，每个 worker 各占一份 | 能力探测失败时本机键为 `None`，启动日志写明原因，容量准入按私有内存计算。 |
| 按进程 RSS 求和统计内存 | 共享页被算 N 次 | 指标按 `Mapping.kind` 和共享比例平摊（05）；测试用 `Shmem` 和 cgroup 统计。 |
| C 层直接 fork | 子进程继承描述符和监听套接字 | 记为已知限制（总体设计 13）。 |

## 反馈期

发布后一周。

## CC

HWM、分布式运行时、编排（`StageRuntime`、`OmniMasterServer`）、配置的负责人。
