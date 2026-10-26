# [RFC]: 统一 Offloader

> 上级 RFC：[umbrella RFC](./00-umbrella.md)。设计依据：[总体设计](../LOADER_OFFLOADER_DESIGN.md) 第 7 节（7.1～7.8），以及 4.1～4.5 里 Offloader 的步骤、8.5 和 8.6 里归规划和 `prepare_materialization()` 的校验、9.3、第 10 节的加速卡部分、第 11 节、第 12 节第 9、11、12、13、14、15 条。
>
> 范围：`vllm_omni/diffusion/weight_stack/offload/`；`vllm_omni/diffusion/compile.py`；`vllm_omni/platforms/` 的内存登记接口；`vllm_omni/diffusion/cache/exact_projection_cache.py`；`vllm_omni/diffusion/worker/diffusion_model_runner.py` 的前向梯度模式和执行作用域；共存期必改清单里由本 RFC 承担的 C01、C03、C06、C07、C08、C12、C14、C17 所在的模型文件。
>
> 依赖：[子 RFC 01](./01-startup-transaction.md)（`interfaces.py` 的 `Offloader`、`LoadTransaction`、`OffloadRuntime`、`Topology`、`RankCoordinates`、`AllGatherGroups`、`StartupContext`、`MemoryPools`；`errors.py`；`ResourceKind`、`ResourceLedger`、`FaultPlan`；第 1 点的 `Point1Payload.unmatched_policy_keys`；登记表和测试替身）；[子 RFC 02](./02-unified-loader.md)（`ComponentCatalog` 及其 `declaration_order`、`require_complete()` 和三个写方法、`CatalogEntry`、`TensorDecl`、`OverlaySlotDecl`、`TransientNeed`、`ShardGroupSpec`、`FinalManifest`、查询接口 `component_paths`、`repeated_block_paths`、`fully_resident`、`catalog`、`execution_device`、`is_unified_stack`、`weight_generation`，以及“本 RFC 依赖的约定”里要求 02 提供的 `pipeline_of(catalog)`）；[子 RFC 03](./03-host-storage.md)（`HostStorage.view`、`ranges`、`close`，`MemoryRange`、`Mapping`、`HostRequest` 协议、`flatten_shard_unit`、`LocalRecord.machine` 和 `available_bytes`）；[子 RFC 05](./05-configuration.md)（`OffloadPolicy`、`HostPolicy`、`resolve_selection`、`ComponentRef`、`Selection`、启动日志事件和 `record_device_resources`）。

## 摘要

本 RFC 实现总体设计第 7 节的新 Offloader。`plan()` 是纯函数，从组件目录、拓扑和策略算出互不重叠的 Unit，以及每个 Unit 的 Trigger 和 Transport；`prepare_materialization()` 在第一次写模型之前做加速卡容量准入，拿齐块环缓冲区、分片槽位、叠加层槽位、流、事件和暂存区，返回启动事务；`commit()` 校验最终清单，按 `MemoryRange` 合并区间做内存登记，填写 `fully_resident`，经 `HookRegistry` 装上钩子和提交后的守卫，把资源整体移交 `OffloadRuntime`。运行期由按 Transport 分开的两个块环、使用触发和让出关系、单次执行规则、固定顺序的关闭和按显存类别分工的睡眠模式组成。编译设置在提交后运行，cudagraph 模式和模型自己的 CUDA graph 按 `fully_resident` 门控，以权重为键的缓存改用参数身份加写入代数做签名。本 RFC 还承担共存期必改清单里与这些机制冲突的八行。

## 动机

umbrella RFC 的“背景与问题”已经说明总体问题，这里只列本 RFC 取代的现有代码事实：

1. 三个现有后端各自实现驻留和搬运：`vllm_omni/diffusion/offloader/` 下 `sequential_backend.py` 的 `SequentialOffloadHook`（钩子名 `sequential_offload`）、`layerwise_backend.py` 的 `LayerwiseOffloadHook`（`layerwise_offload`）、`distributed_layerwise_backend.py` 的 `DistributedLayerwiseOffloadHook`（`distributed_layerwise_offload`）。三者的锁页、登记和换绑各写一份，换绑经 `tensor_utils.set_tensor_storage` 写 `tensor.data`；内存登记只有 CUDA 上的只读登记（`cuda_host_registration.py` 的 `CudaHostRegistration`）。
2. 模型按名字读 offload 钩子来决定行为：Qwen-Image-2.1 的 `QwenImage21DecodeGraphManager._offload_reason` 读模型上的 `_hook_registry` 和 `SequentialOffloadHook._HOOK_NAME`，决定能否捕获解码 CUDA graph；SeaCache 的 `cache/seacache/hook.py` 按字符串 `"distributed_layerwise_offload"` 取钩子。
3. 模型自己搬运权重：MAGI-2 用 `_Magi2StagedComponent` 包住文本编码器和三个编解码组件，内部用 `PinnedModuleStager` 保留锁页副本，`Magi2Pipeline._conditioning_memory_window` 对 DiT 调用 `.to("cpu")`；MiniMax H3 的 `MiniMaxH3LatentUpscaler.upscale` 在 `resident` 为假时每次放大前后对 `resizer` 调用 `.to()`，`MiniMaxH3DiTModel` 的构造函数在 SM120 上默认把 AdaLN 投影留在 Host；`LingBotVideoPipeline._generate` 按请求参数 `offload_vae_during_denoise` 把 VAE `.to("cpu")`。
4. 编译目标和模式与驻留无关：`regionally_compile`（`vllm_omni/diffusion/compile.py`）按 `_repeated_blocks` 的类名和 `_layerwise_offload_blocks_attrs` 匹配块；Ming-Image 的 `setup_compile` 固定用 `mode="reduce-overhead"`，DreamZero 的 `setup_compile` 给文本、图像编码器和 VAE 用 `reduce-overhead`，InternVLA-A1 在 `InternVLAA1.__init__` 里按 `compile_mode`（默认 `max-autotune`）编译 `sample_actions`。权重被换出时这些 CUDA graph 读到的是过期地址。
5. runner 按 offload 策略选梯度模式：`DiffusionModelRunner._execute_request_list` 在 HSDP 或 `resolve_offload_strategy` 为分布式逐层时用 `no_grad`，否则用 `inference_mode`；`_execute_stepwise_core` 只在 HSDP 时用 `no_grad`。
6. 以权重为键的缓存用存储地址和设备做签名：`ExactProjectionCache._signature` 含每个参数的 `data_ptr()`、`_version` 和 `device`，`MiniMaxH3RuntimeAdalnCache` 继承它。块被换入换出后地址变化，缓存每次都不命中。
7. Cosmos3 的 `Nvfp4W4A4W4A16Strategy.validate_after_processing` 在 `process_weights_after_loading` 里按权重当时的设备放置一次 NVFP4 查找表 `kE2M1ToFloat_handle`；权重之后在 Host 和加速卡之间移动时，查找表留在原设备上。

## 设计

本节所有要求只对 `unified` 成立。`legacy` 下本 RFC 的新代码都不运行；本 RFC 修改的已有文件里，`legacy` 一支是原来的语句。

### 1. 模块与入口

`weight_stack/offload/__init__.py` 导出 `create_offloader(ctx: StartupContext) -> Offloader`、`PlacementPlan`、`Unit` 和 `HOOK_NAMES`，由 01 的 `EntryPointFactory` 按名字取用。`UnifiedOffloader` 实现 01 第 2 节的 `Offloader` 协议，对象本身不持有资源。

```text
启动步骤（总体设计 4.2）  调用                                                      结束时的状态
4   规划                  offloader.plan(catalog, topology, policy)                PLANNED
    第 1 点               plan.collective_signature()、plan.unmatched_policy_keys()、plan.bind_groups(groups)
8   运行期资源            offloader.prepare_materialization(model, plan, storage)  返回启动事务
9   写入（02）            transaction.target(tensor_key)
11  提交                  transaction.commit(final_manifest)                       COMMITTED
12  编译设置              offloader.setup_compile(model, catalog, runtime)
服务                      runtime.grad_mode()、runtime.execution()、before_sleep()、after_wake()
关闭                      runtime.close() 或 transaction.abort()
```

依赖规则（01 的 T1、T2 检查）：

- `offload/` 运行期只 import `weight_stack.interfaces`、`weight_stack.errors`、`weight_stack.startup`（`ResourceKind`、`ResourceLedger`、`FaultPlan`）、`weight_stack.components`（02 的数据类型和查询接口）、`weight_stack.policy`（`resolve_selection`、`ComponentRef`，只在 `plan()` 的函数体里 import）、`weight_stack.observability`（05-6 合入之后，第 9 节）、`weight_stack.host` 的 `flatten_shard_unit`、`vllm_omni.diffusion.hooks`、`vllm_omni.diffusion.compile`、`vllm_omni.platforms`。`MemoryRange`、`Mapping`、`HostRequest`、`OffloadPolicy`、`Selection` 只在 `if TYPE_CHECKING:` 里引用，运行期只读它们的属性。
- `offload/` 不 import `loader/`、`vllm_omni.diffusion.model_loader`、`safetensors`、`vllm.model_executor.model_loader`、`vllm_omni.diffusion.offloader`，也不 import `vllm_omni.diffusion.cache`：写入代数的绑定按方法名识别缓存对象（6.4）。
- import `weight_stack.host` 的 `flatten_shard_unit` 符合总体设计 3.2：第 1 条只约束 Loader 对 HWM 和 Offloader 的 import，第 2 条只禁止 Offloader import Loader 和 safetensors 读取器，第 3 条约束的是 HWM 的职责；01 的 T1 只禁止 `host/` 反向 import `offload/`，依赖方向无环。03 第 4.3 节规定 `flatten_shard_unit` 从 `weight_stack.host` 导出，是分片布局规则的唯一实现，所以这个函数不挪到别的模块。`host/` 不 import `safetensors`（03 第 4.2 节自己生成和解析头部），T2 对 `weight_stack.offload` 的运行期模块图检查会发现违例。
- Offloader 不调用模型的任何加载后钩子（总体设计 3.2 第 2 条）。

### 2. 数据结构

```python
# vllm_omni/diffusion/weight_stack/offload/plan.py
class UnitKind(enum.Enum):
    MODULE = "module"              # 整组件
    BLOCK = "block"                # 一个流式的重复块
    GROUP = "group"                # units 选中的 unit_groups 里匹配的一个模块
    RESIDENT_LAYERS = "resident"   # 一个组件里取到的常驻层组
    REMAINDER = "remainder"        # 非 dit 角色的 layer 组件里，重复块以外的一个直接子模块

class Trigger(enum.Enum):
    USE = "use"                    # 使用触发
    RING = "ring"                  # 块环

class Transport(enum.Enum):
    RANK_LOCAL = "rank-local"
    ALLGATHER = "allgather"

@dataclass(frozen=True)
class TensorExtent:                # 加速卡布局和 flatten_shard_unit 的输入
    name: str                      # 规范名
    shape: tuple[int, ...]
    dtype: torch.dtype
    stride: tuple[int, ...]
    nbytes: int                    # 存储范围，公式与 02 的 ManifestEntry.nbytes 相同

@dataclass(frozen=True)
class Unit:
    unit_id: str                   # 见下文的命名规则
    component: str
    kind: UnitKind
    trigger: Trigger
    transport: Transport
    roots: tuple[str, ...]         # Unit 覆盖的子树的规范模块路径
    hook_targets: tuple[str, ...]  # 装钩子的模块的规范路径，按注册顺序（5.1）
    tensors: tuple[str, ...]       # 规范名，不含别名，按（类型名，规范名）排序
    nbytes: int                    # tensors 的存储范围之和
    device_bytes: int              # 装入后在加速卡上的占用（3.8）
    order: int                     # 本组件流式序列里的位置；USE 的 Unit 为 -1

@dataclass(frozen=True)
class ComponentPlacement:
    name: str
    role: str
    selected: bool
    mode: Literal["module", "layer"] | None
    weight_transfer: Transport | None
    units: tuple[str, ...]                  # unit_id，规划顺序
    resident_tensors: tuple[str, ...]       # 常驻加速卡的规范名（“其余部分”）
    hsdp_managed: bool
    cold_start: Literal["thin", "full_then_trim"] | None   # 只对有 allgather Unit 的组件不为 None
    group_size: int | None                  # 推算的 AllGather 组大小 N
    group_index: int | None                 # 本 rank 在组里的位置 r
    fully_resident: bool                    # 3.9；commit() 原样写进目录
    discovered_blocks: tuple[str, ...]      # 重复块容器的路径，写法与 repeated_blocks 相同

@dataclass(frozen=True)
class RingPlan:
    transport: Transport
    buffer_bytes: int                       # 3.8
    units: tuple[str, ...]                  # 这一类的流式 Unit，规划顺序
    successors: Mapping[str, str | None]    # 初始后继表（5.2）
    shard_slot_bytes: int                   # allgather：一个分片槽位的大小；rank-local 为 0

@dataclass(frozen=True)
class DeviceDemand:                         # 加速卡准入式（4.1）里除可用显存以外的各项
    resident_bytes: int
    overlay_bytes: int
    ring_bytes: int
    shard_slot_bytes: int
    write_transient_bytes: int
    service_use_bytes: int

@dataclass(frozen=True)
class TensorPlacement:
    residency: Literal["host", "device", "hsdp"]
    unit_id: str | None                     # residency 为 host 时不为 None
    transport: Transport | None
    backend: Literal["local", "sharded"] | None

class PlacementPlan:
    components: Mapping[str, ComponentPlacement]  # 按组件名排序
    units: Mapping[str, Unit]                     # 规划顺序
    rings: Mapping[Transport, RingPlan]           # 没有某一类流式 Unit 时不含这一项
    yielders: Mapping[str, frozenset[str]]        # unit_id 到“使用时会让它让出的组件”（3.6）
    triggers: frozenset[str]                      # 只装触发钩子的组件（3.6）
    device_demand: DeviceDemand
    registration_order: tuple[str, ...]           # 内存登记的优先顺序（3.9）
    pin_cpu_memory: bool                          # policy.host.pin_cpu_memory；policy 为 None 时为真
    registration_limit_bytes: int                 # policy.host.registration_limit_bytes；0 表示不设上限
    def placement(self, tensor_name: str) -> TensorPlacement: ...
    def host_request(self) -> HostRequest: ...    # 03 第 4.1 节的结构
    def collective_signature(self) -> Mapping[str, str]: ...
    def unmatched_policy_keys(self) -> frozenset[str]: ...
    def bind_groups(self, groups: AllGatherGroups) -> None: ...
    def cold_start(self, component: str) -> Literal["thin", "full_then_trim"] | None: ...
    def sharded_tensor_names(self) -> frozenset[str]: ...
    def describe(self) -> tuple[PlanEvent, ...]: ...   # 05 第 15.2 节的 component_plan、component_skipped、unit_resident
```

- `unit_id` 只由组件目录和策略决定：`MODULE` 是 `"<组件名>/module"`，`BLOCK` 是 `"<组件名>/block/<块的规范路径>"`，`GROUP` 是 `"<组件名>/group/<模块的规范路径>"`，`RESIDENT_LAYERS` 是 `"<组件名>/resident"`，`REMAINDER` 是 `"<组件名>/remainder/<直接子模块的规范路径>"`。合并后的 Unit 取被合并者里规划顺序最前的那个的 `unit_id`（3.5）。
- `PlacementPlan` 在 `plan()` 返回之后不可变，只有 `bind_groups()` 写入一次进程组引用，第二次调用报 `LifecycleError`。
- `placement(name)` 对目录里的每个规范名和别名都有定义，别名返回它的规范张量的结果；不在目录里的名字报 `LifecycleError`。
- `host_request()` 实现 03 的 `HostRequest` 协议：`units` 按规划顺序为每个 Host 驻留的 Unit 给出一条满足 `HostUnitRequest` 的记录，`unit_id`、`component`、`tensors` 取自这个 Unit，`backend` 在 `allgather` 时为 `sharded`、否则为 `local`；`sharded` 给出每个有 `allgather` Unit 的组件的 `group_size`、`group_index`、`cold_start`；`staging_bytes` 在本 worker 有 Host 驻留的 Unit 时为 2 × 256 MiB，否则为 0；`transient_bytes` 取本 worker Host 驻留张量里最大的单个张量的字节数与 `catalog.write_transient.quant_layer_bytes` 中的较大者；`shapes_final` 是全部本机后端组件的 `CatalogEntry.shapes_final` 的与。
- `collective_signature()` 只含有 `allgather` Unit 的组件。每个组件的值是规范 JSON（键排序）的 blake2b 摘要（`digest_size=16`），覆盖：按 `order` 排列的每个 `allgather` Unit 的 `unit_id`、每个张量的（规范名、形状、类型、步长、存储范围）、`flatten_shard_unit` 给出的各类型组的元素数和段长、`group_size`、`Transport`。它不含 `rank-local` 的 Unit、`LocalRecord` 和任何地址。组件的内容身份不进这个摘要：01 在第 1 点按 `content_identities` 分组，同一个 AllGather 组里的内容身份按定义相同，01 第 2 节 `RankFingerprints.collective` 要求的“内容身份”由这个分组覆盖。
- `unmatched_policy_keys()` 返回 `frozenset(Selection.unmatched)`，`Selection` 是 `plan()` 里 `resolve_selection()` 在本 rank 上的结果；`policy` 为 `None` 时为空集合。01 在第 1 点把它放进 `Point1Payload.unmatched_policy_keys`，对各 rank 的结果取交集，交集不为空时报 `PlanningError`（05 的 L01）。
- `bind_groups(groups)`：对每个有 `allgather` Unit 的组件，在 `groups[component]` 里找含本 rank 的组；找不到、组的大小不等于 `group_size`、或本 rank 在 `ranks` 里的位置不等于 `group_index` 时报 `GroupConfirmationError`，写明组件名、推算值和实际值。找到时记下 `AllGatherGroup.group`，运行期的拼接用它。

### 3. 规划

#### 3.1 输入与纯度

`plan(catalog, topology, policy)` 只读：组件目录（每个 `CatalogEntry` 的声明、`roots`、`carved`、`tensors`、`hsdp_managed`、`shapes_final`、`finalize_replaced`；`overlay_slots`；`write_transient`；`declaration_order`）、模块树（经 02 的 `pipeline_of(catalog)` 取得以这个目录登记的 pipeline 对象，按规范路径解析出模块，只读类型、`type(m).forward` 是否被覆写、子模块的注册顺序）、`topology.coordinates`、`topology.records` 里的 `machine` 和 `available_bytes`、`ctx.rank`、`ctx.od_config.model_class_name`、05 解析出的策略。

- 它不分配大块资源，不改模型，不装钩子，不读空闲显存。
- Unit 的划分、`unit_id`、布局和 `collective_signature()` 不依赖 `LocalRecord` 的 pid、可用内存和空闲显存（03 第 4.5 节的约定）。读 `available_bytes` 的只有 3.3 第 5 条的检查，它只决定是否报错，不改变规划。
- 它对目录唯一的写操作是 `catalog.set_discovered_blocks()`（3.4）。
- `policy` 为 `None` 时按不做 offload 规划：不调用 `resolve_selection()`，没有 Unit，全部张量常驻加速卡（HSDP 管理的除外），`rings` 为空，`registration_limit_bytes` 为 0。

#### 3.2 步骤

```text
1  catalog.require_complete()        归属违例或保持 fp32 的层配了量化时报 PlanningError，列出全部路径
2  选择：resolve_selection()（3.3）
3  对每个不由 HSDP 管理的组件做重复块发现，记录结果（3.4）
4  按组件划分 Unit，处理共用存储和组件根上的参数（3.5）
5  定 Transport、后端，推算 AllGather 组，选冷启动路径（3.7）
6  建让出关系表；没有任何让出者的使用触发 Unit 按常驻处理（3.6）
7  计算 device_bytes、块环、分片槽位和准入的各项（3.8）
8  计算 fully_resident、登记顺序，生成规划说明（3.9）
```

#### 3.3 选择与校验

1. 选择只经 05 的 `resolve_selection(policy, refs, pipeline=ctx.od_config.model_class_name)`：`refs` 按 `catalog.declaration_order` 为每个组件构造一个 `ComponentRef(name, role, hsdp_managed)`（05 第 11 节）。L02（同一角色的非零 `resident_layers` 不同）、L03（被 HSDP 管理的组件被选中，报错给出保留 HSDP 或改用 `layer` 配 `allgather` 两个选择）、L04、L05 由它报 `PlanningError`。本 RFC 不重新解释角色键、组件名键和 `null`。选择函数由 `UnifiedOffloader` 的构造参数 `select` 注入，默认是 05 的 `resolve_selection`（见“代码组织”）。
2. `Selection.unmatched` 原样由 `unmatched_policy_keys()` 交给 01；单个 rank 上的不匹配不在规划阶段报错，因为组件可以按 rank 声明（总体设计 5.1）。
3. 选中组件的 `units` 里有组件声明的 `unit_groups` 没有的名字时报 `PlanningError`，写明组件名和名字；某个组的全部模式都没有匹配到带张量的模块时同样报错，写明组名和模式。
4. 选中、`mode` 为 `layer`、没有 `units` 的组件找不到重复块时，按 `SelectedComponent.if_no_blocks` 处理：`error` 报 `PlanningError`，提示改用 `module` 或不选这个组件；`skip` 不选这个组件，产生一条 `component_skipped` 事件（`reason=no_repeated_blocks`）。
5. 有组件使用 `allgather` 时，本 rank 上每个 Host 驻留、又不是分片 Unit 的组件部分，单个组件的字节数超过本 rank 所在机器（`LocalRecord.machine` 相同的 rank）上 `available_bytes` 的最小值时报 `PlanningError`，写明组件名（总体设计 6.4、8.5）。读数都为 `None` 时跳过。完整的 Host 容量准入由 03 在第 1 点做。

#### 3.4 重复块发现

发现只看模块树，不读模型属性，对每个不由 HSDP 管理、没有声明 `repeated_blocks` 的组件运行一次，不论组件是否被选中：

```text
属于组件 c 的模块：c 的根子树，去掉 carved 划出给其他组件的子树
size(m)：m 子树里属于 c 的最终张量（parameter、buffer、data、quant_output、fixed_overlay）的存储范围之和，
         按 TensorDecl 的形状、步长和类型计算，已含本 rank 的 TP 切分和量化；同一个张量的别名只计一次
候选列表 L：属于 c 的 nn.ModuleList（含子类）；L 里 size 大于 0 的元素至少 2 个、Python 类型相同，
            并且 type(元素).forward 不是 nn.Module.forward；size 为 0 的元素不参与判断
B：组件 c 里全部候选列表的全部元素 size 的最大值
合格：max(size(元素) for 元素 in L) >= B / 4
遍历：从每个根按注册顺序（named_children）递归；遇到合格列表记为一组，不进入它的元素；
      其他模块继续往下，包括自定义的可迭代模块；不进入属于其他组件的子树
结果：按遍历顺序的组列表；每组里 size 大于 0 的每个元素是一个块
```

- `ModuleDict`、`ModuleList`、`ParameterList` 本身从不成为 Unit，钩子只装在块和块里的模块上。
- 组件声明了 `repeated_blocks` 时不做自动发现，按列表顺序取每个容器里 size 大于 0 的元素；容器路径不存在，或者不是 `nn.ModuleList` 或 `nn.Sequential`（02 第 2.6 节的校验）时报 `PlanningError`。
- `plan()` 对每个做了自动发现的组件调用一次 `catalog.set_discovered_blocks(component, paths)`，`paths` 是发现的容器路径（相对组件根，写法与 `repeated_blocks` 相同）；声明了 `repeated_blocks` 的组件不调用。编译设置经 `repeated_block_paths` 读它（6.2）。

#### 3.5 Unit 的划分

每个被选中的组件按下表划分。表外的张量（没选中的组件、选中组件的“其余部分”）常驻加速卡。

| 设置 | Unit | Trigger |
|---|---|---|
| `mode="module"` | 整个组件一个 `MODULE` Unit。 | 使用触发。 |
| `mode="layer"`，没有 `units` | 每个块一个 `BLOCK` Unit；`resident_layers` 取到的块合成这个组件的一个 `RESIDENT_LAYERS` Unit。 | 块用块环，常驻层组用使用触发。 |
| `mode="layer"`，没有 `units`，角色不是 `dit` | 另外把组件根的每个直接子模块里、不属于任何块的张量组成一个 `REMAINDER` Unit。 | 使用触发。 |
| `mode="layer"`，有 `units` | 选中组里每个匹配的模块一个 `GROUP` Unit，不做块的划分。 | 块环。 |

- `units` 的模式按 02 的规则匹配：`*` 恰好匹配一段路径，模式相对组件根。匹配到的模块互相嵌套时报 `PlanningError`；不带张量的匹配忽略。`GROUP` Unit 按注册顺序排列。
- 常驻层计数：`Selection.resident_layers[role]` 为 N 时，这个角色下被选中、`mode` 为 `layer`、`rank-local`、没有 `units` 的组件按 `catalog.declaration_order`、组件内的块按 `repeated_blocks` 的列表顺序（自动发现时按遍历顺序）排成一列，取前 N 个，每个组件里取到的块组成它的 `RESIDENT_LAYERS` Unit。N 不小于总块数时这些块全部归入常驻层组，产生 `unit_resident` 事件（`reason=resident_layers_cover_all_blocks`）。
- `dit` 角色的 `layer` 组件里，块以外的张量常驻加速卡。
- 组件根模块自己直接持有的张量（不属于任何子模块）常驻加速卡，不进 Unit。
- 没有登记来源的非持久 buffer 不在目录里，常驻加速卡，由 `prepare_materialization()` 搬到执行设备（4.1 第 7 步）。`kind="data"` 的张量同样常驻。
- `fixed_overlay` 张量和量化输出属于它所在层的 Unit；量化层量化前的参数不在目录里。叠加层槽位不进任何 Unit，常驻加速卡。
- 动态 LoRA 和 `overlay` 固定适配器的包装层整体属于原来那一层所在的 Unit：目录里的模块路径是包装前的规范路径，解析得到的是包装层本身。
- **共用存储**：规范张量和它的别名属于同一个 Unit。别名落在另一个 Unit 时两个 Unit 合并；块里的张量与块外的张量共用时报 `PlanningError`；一组共用张量跨组件时，只有它们全部常驻加速卡才允许，否则报 `PlanningError`，写明两个组件和张量名。

#### 3.6 让出关系与按常驻处理

使用触发的 Unit 是 `MODULE`、`RESIDENT_LAYERS`、`REMAINDER`。一个组件的“使用”是它的任何一个钩子被调用（5.1）。`yielders[u]` 是使用时让 u 让出的组件集合，按 u 所属组件的角色确定：

```text
u 所属组件的角色      使用以下组件时 u 让出
dit                   任何编码器角色（text_encoder、image_encoder、audio_encoder）的组件，不论是否被选中；
                      被选中的 vae；本 rank 上的其他 dit 组件
编码器角色            本 rank 上的任何 dit 组件
vae（被选中）         任何 dit 组件；任何编码器角色的组件；其他被选中的 vae
```

- 编码器之间不互相让出。没被选中的 `vae` 和 `other` 角色的组件不在任何 `yielders` 里。
- 只考虑本 rank 目录里存在的组件。`yielders[u]` 为空的 Unit 按常驻处理：它的张量改为常驻加速卡，不分配 host 存储，产生 `unit_resident` 事件（`reason=no_yielding_component`，总体设计 7.2）。这一步按本 rank 的组件集合做，不同 rank 的结论可以不同。
- `triggers` 是在某个 `yielders` 里出现、本身又没有使用触发 Unit 或块环 Unit 的组件。这些组件只在根模块上装触发钩子（5.1），触发钩子不装入任何权重。

#### 3.7 Transport、后端与冷启动路径

- `BLOCK` 和 `GROUP` Unit 的 Transport 取组件的 `weight_transfer`；使用触发的 Unit 一律是 `rank-local`（05 的 P15、P16 在参数解析阶段拒绝 `module` 配 `allgather`、`resident_layers` 大于 0 配 `allgather`）。
- `rank-local` 的 Host Unit 用本机后端，`allgather` 的 Unit 用分片后端。
- **AllGather 组的推算**：候选 rank 是 `topology.coordinates` 里 TP 编号与本 rank 相同、启用专家并行时 EP 编号也相同的 rank，按 rank 升序。组件声明了 `shard_group`（`ShardGroupSpec.size` 为 s）时，候选数必须是 s 的倍数，候选列表里位置模 s 与本 rank 相同的 rank 是同组，本 rank 的位置模 s 必须等于 `ShardGroupSpec.rank`；不满足时报 `PlanningError`。`group_size` 是同组的数目，`group_index` 是本 rank 在其中的位置。推算假设同组的每个 rank 都声明了这个组件；实际的组由 01 在第 1 点按内容身份建立，`bind_groups()` 核对两者。组只有一个 rank 时规划不报错，01 在第 1 点报错。
- **冷启动路径**：有 `allgather` Unit 的组件在 `CatalogEntry.shapes_final` 为真、并且 `CatalogEntry.finalize_replaced` 为空时选 `thin`，否则选 `full_then_trim`（总体设计 6.4 的瘦路径条件，03 第 4.1 节）。瘦路径下每个 rank 的 Host 冷启动额外占用以释放阈值加最大的两个张量为上限，由 03 计入 Host 容量准入；加速卡准入不含这一项。`cold_start(component)` 给出它，02 的预检据此核对，`component_plan` 事件写明它。`sharded_tensor_names()` 返回全部 `allgather` Unit 的规范名，02 在收尾第 7 步把它们临时换成占位。

#### 3.8 占用估算

```text
rank-local Unit 的加速卡布局：张量按（类型名，规范名）排列；每个类型组的起点按 256 字节对齐，
  组内每个张量的起点按它的元素大小对齐
  device_bytes = 末尾张量的终点向上取整到 256 字节
allgather Unit：layout = flatten_shard_unit(TensorExtent 列表, group_size)（03 第 4.3 节）
  每个类型组 g 在块环里占 N × S_g，组的起点按 256 字节对齐
  device_bytes = 各组之和向上取整到 256 字节
  在分片槽位里占 Σ_g 向上取整到 256 字节(S_g)
块环 k 的 buffer_bytes = 这一类里最大的两个 device_bytes 之和；只有一个 Unit 时等于它的 device_bytes
shard_slot_bytes = 全部 allgather Unit 在分片槽位里的占用的最大值；分片槽位共两个
```

| `DeviceDemand` 字段 | 含义 |
|---|---|
| `resident_bytes` | 常驻加速卡的目录张量之和；HSDP 管理的组件按 `fs` 的大小均分后计入。没有登记来源的非持久 buffer 在 `prepare_materialization()` 里补上。 |
| `overlay_bytes` | `catalog.overlay_slots` 的字节数之和。 |
| `ring_bytes` | 两个块环的 `buffer_bytes` 之和。 |
| `shard_slot_bytes` | 两倍的 `RingPlan.shard_slot_bytes`，加上每个有 `allgather` Unit 的组件一个长度为 N 的 int64 编号缓冲区。 |
| `write_transient_bytes` | `catalog.write_transient.adapter_delta_bytes + catalog.write_transient.quant_layer_bytes`。两者在写入期间可能同时存在，所以相加。 |
| `service_use_bytes` | 按 3.6 的让出关系取两者中较大的：最大的一个 `dit` 组件的使用触发 Unit 的 `device_bytes` 之和；全部被选中的编码器组件的使用触发 Unit 之和，加上最大的一个被选中 `vae` 的使用触发 Unit 之和。 |

#### 3.9 `fully_resident`、登记顺序与规划说明

- 组件的 `fully_resident` 为真，当且仅当它不由 HSDP 管理、规划之后没有任何 Unit（总体设计 7.8）。只装触发钩子的组件仍然可以是 `fully_resident`。被 HSDP 管理的组件为假（02 第 13 节）。
- `registration_order`：`BLOCK` 和 `GROUP` Unit 按组件的规划顺序和 `order`，然后 `RESIDENT_LAYERS`，然后 `MODULE` 和 `REMAINDER`，后两类内部按 `unit_id` 排序（总体设计 7.4）。
- `describe()` 产生 05 第 15.2 节的 `component_plan`（每个组件一条；`backend` 按 05 的取值写 `local`、`shard` 或 `none`，`cold_path` 写 `thin`、`full_then_trim` 或 `none`）、`component_skipped`、`unit_resident` 事件，由 04 的 logger `vllm_omni.diffusion.weight_stack.offload` 在规划之后输出。

### 4. 启动事务

#### 4.1 `prepare_materialization(model, plan, storage)`

在第一次写模型之前按顺序执行。任何一步失败时，本调用已经分配的资源由它自己按逆序释放，`storage` 仍归调用方（01 第 2 节），报 `PrepareError` 或它的子类 `DeviceCapacityError`。

```text
1  核对：plan 已经 bind_groups()；plan.host_request() 的每个张量都能经 storage.view() 取到
2  登记计划：pin_cpu_memory 为真时调用 host_registration_support()；按 registration_order 和登记上限，
   为每个 Host Unit 选“登记候选”或“暂存区”（5.4）；pin_cpu_memory 为假时全部走暂存区，不查询平台
3  加速卡容量准入（下式）；不通过时报 DeviceCapacityError
4  在 ctx.memory_pools.slots() 里分配两个块环缓冲区、两个分片槽位和编号缓冲区
5  在 ctx.memory_pools.weights() 里分配叠加层槽位，按 OverlaySlotDecl 的形状和类型，填零
6  创建拷贝流、拼接流，每个块环位置和每个分片槽位的事件；计划里有走暂存区的 Unit 时分配两个 256 MiB 暂存区
   （pin_cpu_memory 为真时锁页）
7  把不由 HSDP 管理的组件里没有登记来源的非持久 buffer，在 ctx.memory_pools.weights() 里搬到执行设备
8  在任何梯度模式之外，为每个块环 Unit 在缓冲区的两个位置各建一组张量视图（5.2），为每种类型建零元素占位，
   在 ctx.memory_pools.weights() 里为每个 allgather Unit 在执行设备上建一个装着它的 order 的 int64 张量（5.4）；
   这些张量的内容在服务期间不变，所以放在睡眠时备份的池里
9  建启动事务：每个资源按 ResourceKind 登记进 ResourceLedger；storage 登记为 HOST_STORAGE；
   另登记一个 PLACEHOLDER 条目，关闭时按 TensorDecl 的规范名在 model 里取出每个 Host 驻留张量的对象（含别名），
   改指向占位，所以 abort() 在关闭 host 存储之前先让参数离开 host 存储（03 的约定）
```

加速卡容量准入（总体设计 7.5）：

```text
resident_bytes + 非持久 buffer + overlay_bytes + ring_bytes + shard_slot_bytes
  + max(write_transient_bytes, service_use_bytes)
  ≤ 可用显存
可用显存 = 第 3 步开始时调用 current_omni_platform.empty_cache() 之后
           current_omni_platform.get_free_memory(ctx.device) 的读数
```

- `DeviceCapacityError` 写明每一项的字节数和可用显存，`fix` 指向 recipe 里的 offload 策略。
- 准入只在本 rank 内做，不经过集合通信；各 rank 的结果在 01 的第 2 点汇总，任何一个 rank 失败时整组一起失败。
- 第 4、5 步分配的地址到 `close()` 前不变（总体设计 7.5 的固定槽位）。
- 第 7 步的 buffer 搬移不改变模块结构：按 `module._buffers[name] = buffer.to(ctx.device)` 替换，不经过 `Module.to()`。

#### 4.2 `LoadTransaction.target(tensor_key)`

| `plan.placement(tensor_key).residency` | 返回 |
|---|---|
| `host` | `storage.view(tensor_key)`：本机后端的写入目标；等待者的共享张量和瘦路径范围外的张量是 meta 张量（03 第 7.2、9.1 节）。 |
| `device` | `None`：Loader 在执行设备上、在 `weights()` 池里分配。 |
| `hsdp` | `None`：由 HSDP 的物化流程分配（02 第 13 节）。 |

- `tensor_key` 是预期清单里的规范名。不在规划里的名字报 `LifecycleError`。
- 返回张量的形状、类型、步长由 03 的布局保证与预期清单一致，`target()` 不再复制核对；Loader 绑定时核对（02 第 7.4 节）。
- `target()` 在提交之后或 `abort()` 之后调用报 `LifecycleError`。

#### 4.3 `commit(final_manifest)`

```text
1  校验最终清单（见下文）
2  内存登记：对登记候选取 storage.ranges(unit)，按 Mapping 合并后登记、确认锁页；
   失败的合并区间里的 Unit 改走暂存区，记 unit_transfer 事件；需要而还没分配暂存区时在这时分配（5.4）
3  建运行期的 Unit 记录：rank-local Unit 的每个张量从 final_manifest.live(name) 取模型张量对象，
   记下它此刻绑定的 Host 视图作为 H2D 源；allgather Unit 的 H2D 源取 storage.ranges(unit)（5.4）；
   然后把全部 Host 驻留的张量（含别名对象）改指向执行设备上的占位
4  叠加层槽位：对每个 OverlaySlotDecl 调用 catalog.bind_overlay_slot(name, slot)，再从账本 hand_over(slot)
5  对目录里的每个组件调用一次 catalog.set_fully_resident(component, value)
6  在 compile.py 的驻留表里登记每个组件（6.3）；给模型里的每个以权重为键的缓存绑定写入代数（6.4）
7  经 HookRegistry 装钩子（5.1），对每个不由 HSDP 管理的组件装提交后的守卫（5.5）
8  ledger.transfer() 移交给新建的 OffloadRuntime，启动事务变为空；
   输出 fully_resident 事件；05-6 合入之后在这里调用 observability.record_device_resources(...)（第 9 节）
```

第 1 步的校验：

- 每个 `FinalEntry` 的 `residency` 与 `plan.placement(name).residency` 相同；`host` 条目的 `storage_key` 等于规范名；每个 Host Unit 的每个张量都在清单里。
- 不在 `finalize_replaced` 里的张量，形状、类型、步长与 `TensorDecl` 相同。
- 在 `finalize_replaced` 里（预期清单里 `replaced_at_finalize` 为真）、最终形状或类型与预期条目不同的张量所在的 Unit，用最终清单的条目重新计算布局：`rank-local` Unit 按 3.8 的规则，`allgather` Unit 把这个 Unit 的 `FinalEntry` 做成 `TensorExtent` 交给 `flatten_shard_unit`（03 第 9.3 节在 `seal()` 里按同一个函数重新布局 Host 一侧）。块环 Unit 重新计算后，这个块环里最大的两个 `device_bytes` 之和大于 `buffer_bytes`，或者一个 `allgather` Unit 在分片槽位里的占用大于 `shard_slot_bytes` 时报 `CommitError`，因为这些固定槽位在写入之前按写入前的布局分配；使用触发 Unit 重新计算后的 `device_bytes` 大于规划值时同样报 `CommitError`，因为加速卡准入按规划值计算。放得下时，第 3 步按重新计算的布局重建这个 Unit 在块环两个位置上的视图和分片槽位里的区域，运行期用新的布局。
- 不一致时报 `CommitError`，写明张量名和不一致的字段。
- `commit()` 在收尾第 6 步的 `seal()` 之后运行，所以第 2、3 步调用 `storage.ranges()` 时它已经有效（03 第 7.6 节）。

其余规则：

- 第 2 步的暂存区分配失败或撤销登记失败时报 `CommitError`。
- 任何一步失败时，已经建立的登记、钩子、守卫都已登记在账本里，资源全部留在启动事务里，由 `abort()` 释放（01 第 2 节）。第 4 步之后失败时，已经 `hand_over` 的槽位只被模型的包装层引用，随模型丢弃（01 第 5.1 节）。
- `commit()` 只做本地操作，不发起集合通信；各 rank 在 01 的第 3 点一起激活。

#### 4.4 `abort()` 与所有权

- `abort()` 按 01 第 5.2 节的 `ResourceLedger.close()` 顺序释放启动事务持有的全部资源，包括关闭 host 存储；可以多次调用，提交之后什么都不做。
- 启动事务没提交就被丢弃时，`__del__` 不释放资源；释放只经 01 驱动的释放流程调用 `abort()`。
- 同一个句柄只登记一次：两个块环缓冲区、两个分片槽位、编号缓冲区、各 `allgather` Unit 的 order 张量、每个合并区间的登记、每个暂存区、每条流、每个事件组、第 9 步的占位条目各是一个句柄；`commit()` 第 3 步建好的 Unit 记录在 `transfer()` 时随占位条目一起移交，运行时的 `PLACEHOLDER` 关闭用这份记录里的张量对象；钩子按（模块，钩子名）登记一个句柄；守卫按组件登记一个句柄；驻留表的登记和写入代数的绑定按组件登记在 `HOOKS` 下。

### 5. 运行期

#### 5.1 钩子

钩子经 `vllm_omni/diffusion/hooks/base.py` 的 `HookRegistry` 安装：`HookRegistry.get_or_create(module)` 第一次调用时把 `module.forward` 换成分发器 `_WrappedForward`，原来的 `forward` 存在 `module._omni_original_forward`；`register_hook(name, hook)` 登记钩子，`remove_hook(name)` 撤掉。不用 `register_forward_hook`。

```python
# vllm_omni/diffusion/weight_stack/offload/hooks.py
HOOK_NAMES: frozenset[str] = frozenset({"_weight_stack_ring", "_weight_stack_use", "_weight_stack_trigger"})
```

| 钩子 | 装在哪里 | `pre_forward` | `post_forward` |
|---|---|---|---|
| `_weight_stack_ring` | 块环 Unit 的根模块，以及 Unit 里每个直接持有张量的模块。 | 记一次组件的使用；没有活动的流式 Unit 时从这个模块进入 Unit（5.2）；活动 Unit 就是它时只核对驻留；是另一个流式 Unit 时报 `RuntimeFault`。 | 进入点的模块返回时退出 Unit，在计算流上记录这个位置的完成事件。 |
| `_weight_stack_use` | 使用触发 Unit 里每个直接持有张量的模块。 | 记一次组件的使用；Unit 不在加速卡上时装入（5.3）；Unit 的使用深度加一。 | 使用深度减一。 |
| `_weight_stack_trigger` | `triggers` 里每个组件的每个根模块。 | 记一次组件的使用。 | 什么都不做。 |

- “直接持有张量的模块”是 `TensorDecl.module` 给出的规范路径解析出的模块；包装层解析成包装层本身，钩子装在包装层上。
- “记一次组件的使用”：组件与上一次使用的组件相同时什么都不做；否则按 `yielders` 让出这个组件的使用会让出的全部 Unit（5.3），再把它记为上一次使用的组件。
- 钩子只实现 `pre_forward` 和 `post_forward`，不覆写 `new_forward`，所以能与 TeaCache、MagCache 这类覆写 `new_forward` 的缓存钩子共存。`HookRegistry.update_sorted_hooks` 按钩子名排序执行 `pre_forward`，覆写 `new_forward` 的钩子排在最后，`post_forward` 按逆序执行。现有钩子名（`sequential_offload`、`teacache`、`mag_cache_block_hook`、`sea_cache`、`sp_input---<id>` 等）都以小写字母开头，三个名字以 `_` 开头，排在它们之前，所以 Offloader 的 `pre_forward` 最先执行、`post_forward` 最后执行。
- 两个方法都用 `torch.compiler.disable` 修饰：编译包住的区域里调用到被管理的模块时，钩子在编译图之外运行（总体设计 7.8）。
- CUDA graph 捕获期间（`torch.cuda.is_current_stream_capturing()` 为真）：`_weight_stack_ring` 和 `_weight_stack_use` 被调用时报 `RuntimeFault`，写明组件名，因为它们只装在不是 `fully_resident` 的组件上；`_weight_stack_trigger` 只在这次调用需要让出时报 `RuntimeFault`，不需要任何动作时放行，因为 `fully_resident` 的组件也可能带触发钩子。
- 钩子在 `execution()` 的作用域之外被调用时报 `RuntimeFault`，写明组件名和模块路径（5.6）。

#### 5.2 块环

两类流式 Unit 各用一块缓冲区和一张后继表，互不读写：

```text
位置：HEAD = [0, device_bytes(u))，TAIL = [buffer_bytes - device_bytes(u), buffer_bytes)
      任意两个 Unit 一个在 HEAD、一个在 TAIL 时不重叠，因为 buffer_bytes 是最大的两个占用之和
状态：slot[HEAD]、slot[TAIL] 各记一个 Unit 或空；每个位置有完成事件 done[p] 和就绪事件 ready[p]
后继表：succ[u] 初始为同一组件里 order 的下一个（RingPlan.successors）；组件里最后一个，rank-local 接回这个组件的第一个，
        allgather 没有后继

进入 Unit u（从它的某个钩子）：
  u 在某个位置 p 上：计算流等待 ready[p]
  u 不在缓冲区里（预测错）：
    p = 同一张表里上一个进入的 Unit 所在位置的另一侧；没有上一个时 p = HEAD
    同步装入 u 到 p（下文的“装入”），计算流等待 ready[p]
    同一张表里上一个进入的 Unit 存在时，把它的后继改成 u（allgather 表只在两者属于同一组件时改）
  u 的张量改指向位置 p 上的视图；u 记为活动 Unit
  w = succ[u]；w 存在且不在缓冲区里时，异步装入 w 到 p 的另一侧
退出 Unit u：在计算流上记录 done[p]；活动 Unit 清空

装入 Unit x 到位置 q：
  q 上原来的 Unit 的张量改指向占位
  拷贝流（allgather 时是拼接流）等待 done[q]
  rank-local：按 5.4 从 Host 拷到位置 q；allgather：H2D 到分片槽位，再拼接到位置 q
  在这条流上记录 ready[q]；slot[q] = x
```

- 每个块环 Unit 在两个位置上的视图在 `prepare_materialization()` 里建好，运行期只做 `tensor.data = view` 的换绑，与现有的 `set_tensor_storage` 是同一种写法。
- 一个流式 Unit 还没退出又从另一个流式 Unit 的钩子进入时报 `RuntimeFault`，写明两个 Unit。这条规则对两类块环都适用，也对两类之间适用。
- `allgather` 的块环不跨前向保留：每次 `execution()` 退出时，`allgather` 块环的两个位置清空，张量改指向占位。每次前向的第一个 `allgather` Unit 因此在进入时拼接，之后按后继表预取。
- `allgather` 的后继表只按 `allgather` Unit 的进入顺序更新，`rank-local` 的进入不改它；跨组件的进入不改它，所以预取只发生在同一个 AllGather 组里。同组各 rank 的进入顺序相同，所以每个 rank 发起拼接的次数和顺序相同；只在部分 rank 上声明的 `rank-local` 流式 Unit 不影响它。
- 拼接流上发起的集合通信按发起顺序完成，所以预测错时覆盖一个刚被预取的位置不需要额外等待。

#### 5.3 使用触发与让出

- 装入：在当前计算流上按 Unit 的 `device_bytes` 向缓存分配器申请一块存储，按 3.8 的布局建视图，从 Host 拷入（5.4），张量改指向视图，记为驻留。地址不做承诺。
- 装入时显存不足：释放这个 Unit 已经申请的存储，张量改回占位，然后按 5.6 的异常规则处理。
- 让出：Unit 的张量改指向占位，丢掉对存储的引用。让出不往 Host 回写（总体设计不变量 9）。一次“记使用”里让出的全部 Unit 释放完之后调用一次 `current_omni_platform.empty_cache()`。
- 要让出的 Unit 的使用深度大于 0 时（它自己的某个模块正在执行），报 `RuntimeFault`，写明两个组件。
- `vae` 没被选中时常驻加速卡，不装钩子；被选中时由使用触发管理。
- 模型绕过钩子直接读一个已让出的参数时读到零元素的占位，形状不符，直接报错。

#### 5.4 Transport 的实现

**rank-local**。H2D 源是提交时记下的 Host 视图。拷贝时把在 Host 上相邻、在加速卡布局上也相邻的连续张量合成一次拷贝；本机后端同一个 Unit 同一种类型的张量按规范名连续摆放（03 第 4.2 节），所以通常每种类型一次拷贝。

- 登记过的 Unit：块环在拷贝流上、使用触发在计算流上 `copy_(non_blocking=True)`。
- 走暂存区的 Unit：两个暂存区交替使用，数据按 256 MiB 分段。每段在 CPU 上拷进暂存区，然后异步 H2D，记录这个暂存区的事件；再次使用一个暂存区之前在 CPU 上等它的事件。`pin_cpu_memory` 为假时暂存区不锁页，H2D 是同步拷贝。

**allgather**。每个 `allgather` Unit 的 H2D 源是 `storage.ranges(unit)` 给出的本 rank 范围，按地址排序后与 `flatten_shard_unit` 的类型组一一对应，个数不符时在提交时报 `CommitError`。源张量由 `MemoryRange` 的地址和长度建成 uint8 视图，不拷贝。

```text
装入 allgather Unit x 到位置 q（拼接流上）：
  选两个分片槽位中的另一个 k；拼接流等待槽位 k 上一次拼接的完成事件
  拷贝流：本 rank 每个类型组的 S_g 字节 H2D 到槽位 k（登记或暂存区，规则同 rank-local）；记录事件 h
  拼接流：等待 h
          all_gather_into_tensor(ids, x 的 order 张量（4.1 第 8 步）)；torch._assert_async((ids == x.order).all())
          每个类型组一次 all_gather_into_tensor(位置 q 上这一组的 N × S_g 区域, 槽位 k 上这一组的 S_g 区域)
  记录 ready[q] 和槽位 k 的完成事件
```

- 拼接用 `bind_groups()` 记下的这个组件的 `AllGatherGroup.group`，各类型组都按 uint8 视图拼接。
- 编号不一致时设备侧断言失败，这次前向以异常结束，运行时按 5.6 标记为不可用。
- 张量视图建在每个类型组拼接结果里这个张量的偏移上，补零部分不被任何张量引用。

**内存登记**。

```text
commit() 第 2 步：
  对 registration_order 里每个登记候选 u：rs = storage.ranges(u)
  每个区间向外对齐到页（os.sysconf("SC_PAGE_SIZE")），截在所在 Mapping 的 [address, address + length) 之内
  按 Mapping.mapping_id 分组，组内合并成互不重叠的区间
  按 registration_order 依次把 u 的区间并入，合并后的总字节数超过上限时 u 改走暂存区（上限为 0 时不限）
  逐个合并区间调用 current_omni_platform.register_host_range(address, length, read_only=MemoryRange.read_only)
  登记后用 host_range_is_pinned 确认首页和末页被认作锁页；确认不了当作登记失败，撤销这个区间的登记
  一个合并区间失败时，落在它里面的 Unit 全部改走暂存区，记一次 WARNING
```

- `read_only` 为真（共享存储）时只读登记，需要平台支持只读登记；为假（私有存储、rank 分片）时普通登记，需要平台支持可写登记。平台不支持相应的登记时，这些 Unit 走暂存区，不尝试登记。
- 上限是 `plan.registration_limit_bytes`（05 的 `HostPolicy.registration_limit_bytes`），按每个 worker 统计对齐、合并之后的字节数，覆盖全部 Unit 和分片。`prepare_materialization()` 第 2 步还拿不到 `ranges()`，按每个 Unit 的字节数加上它涉及的每个映射两页估算对齐后的大小来选候选，并据此决定是否在准备阶段分配暂存区。提交时按 `ranges()` 的实际区间再按上面的规则判断一次：超出上限的 Unit 改走暂存区，暂存区还没分配时在这时分配（总体设计 7.4、第 11 节）。Host 容量准入总是计入暂存区（03 第 4.1 节的 `staging_bytes`），所以提交时分配暂存区不超出准入。
- 一个 Unit 要么整个登记，要么整个走暂存区。撤销登记按合并区间进行。每个 Unit 的结果写成一条 `unit_transfer` 事件（`path` 为 `registered` 或 `staging`，`reason` 为 `limit`、`unsupported`、`failed` 或 `pin_disabled`）。
- 任何策略都不生成完整的锁页副本：Host 上只有 HWM 分配的存储和两个固定大小的暂存区。

#### 5.5 提交后的守卫

`commit()` 对每个不由 HSDP 管理的组件的每个模块（根子树去掉划出的子树）把实例属性 `_apply` 换成守卫。`nn.Module` 的 `.to()`、`.cuda()`、`.cpu()`、`.float()`、`.half()`、`.bfloat16()`、`.type()`、`to_empty()` 都经过 `_apply`，递归时对子模块调用的也是子模块实例上的 `_apply`。

```text
守卫(fn, recurse=True)：
  对这个模块直接持有的张量的每种（类型，维数），在执行设备上建一个形状为 (2,) × 维数 的探针 t
  每个 fn(t) 的设备、类型、步长都与 t 相同：不对任何真实张量调用 fn，返回模块
  否则：报 RuntimeFault，写明组件名、模块路径和变化（设备、类型或内存格式）
```

- 守卫从不把 `fn` 用到被管理的张量上，`to_empty()` 因此也不重新分配。
- 被管理的参数在服务期间始终是执行设备上的张量（驻留时是视图，否则是占位），用 `next(params).device` 判断设备的模型代码看到的是执行设备。
- 守卫对全部组件生效，包括 `fully_resident` 的组件：提交后基础权重的字节、布局、设备都不可变（总体设计不变量 3）。
- `close()` 删除实例属性，恢复类上的 `_apply`。

#### 5.6 单次执行、梯度模式与异常

```python
# vllm_omni/diffusion/weight_stack/offload/runtime.py
class UnifiedOffloadRuntime:   # 实现 01 的 OffloadRuntime
    def grad_mode(self) -> AbstractContextManager[None]: ...   # 有块环 Unit 时 torch.no_grad()，否则 torch.inference_mode()
    def execution(self) -> AbstractContextManager[None]: ...
    def before_sleep(self) -> None: ...
    def after_wake(self) -> None: ...
    def close(self) -> None: ...
```

```text
进入 execution()：运行时不可用、处于睡眠、已经关闭，或已有一次执行在进行（任何线程）时报 RuntimeFault
正常退出：清空 allgather 块环；活动 Unit 应为空、每个使用触发 Unit 的使用深度应为 0，不满足时报 RuntimeFault
异常退出：同步拷贝流和拼接流；清空活动 Unit 和 allgather 块环；全部使用深度清零；
          这次执行里发起过 allgather 拼接时把运行时标记为不可用；然后重新抛出原异常
```

- 模型前向抛出异常时 `HookRegistry.dispatch` 不调用 `post_forward`，所以活动 Unit 和使用深度由 `execution()` 的异常退出统一清理，不依赖钩子的 `post_forward`。

- 并发和重入按非阻塞地获取一把锁判断，获取不到就报错，不等待。
- 运行时被标记为不可用之后，每次 `execution()` 都报 `RuntimeFault`，写明第一次失败的原因。集合通信的顺序已不可信，这个副本的 worker 由执行器终止（总体设计不变量 12）。
- runner 不读 offload 策略：`unified` 下前向的梯度模式取 `grad_mode()`，HSDP 下照旧用 `no_grad`；前向包在 `execution()` 里（C17，第 8 节）。

#### 5.7 关闭

`close()` 是本地的、终止性的，可以多次调用，按 01 第 5.2 节的 `ResourceKind` 顺序进行：

```text
EXECUTION     不再接受新的执行
INFLIGHT      同步拷贝流和拼接流
HOOKS         从每个模块的 HookRegistry 撤掉本 RFC 的钩子，撤掉守卫，清除驻留表的登记和写入代数的绑定，
              丢掉对模型的引用
DEVICE        释放块环缓冲区、分片槽位、编号缓冲区、事件、流
REGISTRATION  按合并区间调用 unregister_host_range
STAGING       释放暂存区
PLACEHOLDER   每个 Host 驻留的张量和块环 Unit 的张量改指向占位
HOST_STORAGE  storage.close()
```

- 关闭不把权重恢复回 CPU，不发起集合通信（总体设计不变量 11）。05-6 合入之后，`close()` 在最后一步之后调用 `record_device_resources` 把本 worker 的数值清零（第 9 节）。
- 运行时因为拼接失败被标记为不可用时，`INFLIGHT` 不等待拼接流，`DEVICE` 不释放块环缓冲区和分片槽位，它们留到进程退出：拼接流上可能还有永远不会完成的集合通信在写它们。
- 一个关闭函数抛出异常时其余照常执行，由账本汇总报 `LifecycleError`（01 第 5.2 节）。

#### 5.8 睡眠模式

`DiffusionWorker` 的睡眠用 CuMem 分配器：level 1 只把 `weights` 作为 `offload_tags`（`DiffusionWorker.sleep`），其余池释放、不备份；01 第 8.2 节让 `slots()` 进入标签为 `weight_slots` 的池。

| 加速卡上的显存 | 分配位置 | 睡眠时 | 唤醒时 |
|---|---|---|---|
| 常驻的权重和 buffer（Loader 分配）、本 RFC 搬到执行设备的非持久 buffer、叠加层槽位、`allgather` Unit 的 order 张量 | `memory_pools.weights()` | 由睡眠模式拷到 Host。 | 由睡眠模式拷回原地址。 |
| 块环缓冲区、分片槽位、编号缓冲区 | `memory_pools.slots()` | 直接释放，不备份。 | 重新映射到原地址，内容在下一次使用时装入。 |
| 使用触发 Unit | 缓存分配器 | `before_sleep()` 让出。 | 下一次使用时装入。 |
| host 存储、暂存区、内存登记 | Host | 不受影响。 | 不受影响。 |

```text
before_sleep()：非阻塞地获取 execution() 的锁（有执行在进行时报 RuntimeFault）；同步拷贝流和拼接流；
               让出全部使用触发 Unit；清空两个块环，张量改指向占位；记为睡眠；调用 empty_cache()；释放锁
after_wake()：清除睡眠标记；块环保持为空
```

- 01 第 8.4 节规定 worker 的调用顺序：`before_sleep()` 在 `allocator.sleep(...)` 之前，`after_wake()` 在唤醒 `weights` 和 `weight_slots` 之后。
- 唤醒后常驻部分和固定槽位的地址不变，编译结果和只读取 `fully_resident` 组件的 CUDA graph 继续有效。
- 睡眠期间的 `execution()` 报 `RuntimeFault`。

### 6. `fully_resident`、编译与 CUDA graph

#### 6.1 `fully_resident`

`commit()` 第 5 步按 3.9 的值对每个组件调用一次 `catalog.set_fully_resident()`。模型、`cache/`、`lora/` 只经 02 的 `fully_resident(pipeline, component, *, legacy)` 读它：`unified` 下读目录，`legacy` 下执行模型现有的判断。不是 `fully_resident` 的组件不进 CUDA graph，走 eager（总体设计 7.8、不变量 14）。

#### 6.2 编译设置 `setup_compile(model, catalog, runtime)`

01 在提交之后、只在 `ctx.compile_enabled` 为真时调用它（01 第 10 节的 `compile_setup` 注入点）。

```text
pipeline 定义了 setup_compile()：在 runtime.grad_mode() 和 runtime.execution() 里调用它，因为有的 setup_compile()
  会预热编译结果、调用到被管理的模块；模型里的编译经 6.3 的通用编译函数，模式按驻留降级
否则：对 component_paths(model, "dit", legacy=...) 的每个 dit 组件 c（组件名由 catalog.module_owner 给出）：
  compile_kwargs = {"dynamic": od_config.diffusion_compile_dynamic}
  组件根模块的 enable_cuda_graph_decode 为真时加
    options={"triton.cudagraphs": False, "triton.cudagraph_trees": False}（与 runner 的 _compile_transformer 相同）
  diffusion_compile_granularity 为 "full"：root.compile(**compile_kwargs)
  否则：对 repeated_block_paths(model, c, legacy=...) 每个容器里带张量的元素 b，编译 b 的原始 forward（6.3）
任何一个组件失败：记 WARNING，这个组件保持未编译，其余照常
```

- 编译目标只从组件目录读取，不读 `_repeated_blocks`、`_layerwise_offload_blocks_attrs` 和 `_dit_modules`（02 第 15 节）。
- `full` 粒度与 offload 互斥（05 的 P29），所以 `root.compile()` 时组件上没有 Offloader 的钩子。

#### 6.3 驻留表与通用编译函数

```python
# vllm_omni/diffusion/compile.py，新增；legacy 下不登记，也不读驻留表
def register_component_residency(root: nn.Module, *, component: str, fully_resident: bool,
                                 blocks: Sequence[nn.Module]) -> None: ...
def clear_component_residency(root: nn.Module) -> None: ...
def resolve_compile_mode(modules: Sequence[nn.Module], compile_kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]: ...
def compile_module_forward(module: nn.Module, **compile_kwargs: Any) -> nn.Module: ...
def compile_callable(fn: Callable[..., Any], *, reads: Sequence[nn.Module], **compile_kwargs: Any) -> Callable[..., Any]: ...
```

- 驻留表是以组件根模块为键的 `weakref.WeakKeyDictionary`，`commit()` 第 6 步对每个组件登记一次，`blocks` 是 `repeated_block_paths` 解析出的块模块；`close()` 在 `HOOKS` 阶段清除。
- `resolve_compile_mode`：涉及的组件是根模块与 `modules` 里某个模块互为祖先或相同的组件；全部涉及的组件都是 `fully_resident` 时视为常驻，没有涉及的组件时也视为常驻。`mode` 是 `reduce-overhead` 或 `max-autotune`、或 `options["triton.cudagraphs"]` 为真时视为 cudagraph 模式。cudagraph 模式遇到不常驻时，`mode` 改为 `default`，`options` 里的 `triton.cudagraphs` 和 `triton.cudagraph_trees` 改为假，返回原模式，并输出 `fully_resident` 事件（`downgraded_from` 为原模式）。模型直接传入的编译模式同样经过这一步（总体设计 7.8）。
- `regionally_compile(model, ...)` 的 `unified` 一支：参数经 `resolve_compile_mode([model], ...)` 处理；目标是驻留表里落在 `model` 子树内的块；编译每个块的方式与 `legacy` 一支相同（有 `HookRegistry` 时编译 `_omni_original_forward` 并更新钩子的 `fn_ref`）；没有目标时记同样的 WARNING 并返回。`legacy` 一支原样保留。
- `compile_module_forward(module, ...)`：`unified` 下参数经 `resolve_compile_mode([module], ...)`，模块有 `HookRegistry` 时编译 `_omni_original_forward`，钩子留在编译图之外；`legacy` 下执行 `module.forward = torch.compile(module.forward, **compile_kwargs)`。
- `compile_callable(fn, reads=...)`：`legacy` 下返回 `torch.compile(fn, **compile_kwargs)`；`unified` 下返回一个包装，第一次调用时按 `reads` 解析模式并编译，此后直接调用编译结果。第一次调用总在提交之后。
- `unified` 下，`regionally_compile`、`compile_module_forward` 在调用时、`compile_callable` 的包装在第一次调用时解析模式；这时驻留表为空（提交之前）报 `LifecycleError`。`compile_callable` 本身在构造函数里调用也不报错，它不读驻留表。这些函数经 `is_unified_stack()` 读总开关，位置登记在 C12 的位置文件里。`compile.py` 只在函数体里 import `weight_stack.components` 和 `weight_stack.errors`。

#### 6.4 以权重为键的缓存

```text
ExactProjectionCache（cache/exact_projection_cache.py），unified 一支：
  bind_weight_generation(source: Callable[[], int]) -> None      没有绑定时代数取 0
  _cache_signature(module)：legacy 下返回 _signature(module)（不变）；unified 下返回
    (id(module), id(quant_method), _diffusion_lora_active_slices, 代数,
     每个参数、buffer、lora_a_stacked、lora_b_stacked 张量的 (名字, id(张量)))
  project() 和 MiniMaxH3RuntimeAdalnCache.seed() 改调 _cache_signature
commit() 第 6 步：遍历模型每个模块的 __dict__，对带可调用的 bind_weight_generation 属性的对象
  （ExactProjectionCache 及其子类）调用 bind_weight_generation(functools.partial(weight_generation, model))
```

- 签名不含 `data_ptr()`、`_version`、设备，也不含张量当时的形状和类型：块被换入换出、参数指向占位或视图时，参数对象的身份不变，签名不变（总体设计 7.8）。形状不进签名，因为让出的参数指向零元素占位，`units` 选中的 AdaLN 投影在缓存命中时不被调用、不装入，按当时的形状计算会让签名在 `eval()` 和服务期间不同；提交后基础权重的形状和类型不变（总体设计不变量 3），身份和代数已经足够。
- 现有代码里 `ExactProjectionCache` 的实例只有 `MiniMaxH3DiTModel.adaln_cache`，它是模块 `__dict__` 里的普通属性，第 6 步能找到它。按方法名识别让 `offload/` 不 import `cache/`。基础权重提交后不变，代数只在动态 LoRA 管理器写叠加层槽位后由 `bump_weight_generation` 递增（02 第 11.5 节）。
- 不读 `_version`，所以 `inference_mode` 下的张量也能得到签名。
- `_has_forward_hooks` 的停用条件在两个栈下都不变：Offloader 的钩子经 `HookRegistry` 安装，不是 `nn.Module` 的 forward 钩子，所以不让这些缓存停用。
- 代数在 `eval()`（收尾第 7 步）和提交之后都取 0，`seed()` 在 `eval()` 里算的签名在服务期间仍然匹配。

### 7. 平台的内存登记接口

```python
# vllm_omni/platforms/interface.py，OmniPlatform 新增的类方法
def host_registration_support(cls) -> tuple[bool, bool]: ...    # （可写登记，只读登记）；默认 (False, False)
def register_host_range(cls, address: int, length: int, *, read_only: bool) -> None: ...  # 默认报 NotImplementedError
def unregister_host_range(cls, address: int) -> None: ...
def host_range_is_pinned(cls, address: int) -> bool: ...         # 默认 False
```

- CUDA 的实现放在 `vllm_omni/platforms/cuda/host_registration.py`，经 ctypes 调用 CUDA runtime：`cudaDeviceGetAttribute` 查询 `cudaDevAttrHostRegisterSupported` 和 `cudaDevAttrHostRegisterReadOnlySupported`，结果按进程缓存；`cudaHostRegister` 用 `cudaHostRegisterDefault` 或 `cudaHostRegisterReadOnly`；`cudaHostUnregister`；`cudaPointerGetAttributes` 的内存类型为 Host 时 `host_range_is_pinned` 为真。调用失败时读出并清除 CUDA 的上一个错误，再报 `HostRegistrationFailure`（`vllm_omni/platforms/interface.py` 新增的异常类型），`commit()` 捕获它并让这些 Unit 走暂存区。
- ROCm、NPU、XPU、MUSA 使用默认实现，Unit 走暂存区（总体设计 7.4）。
- 新的实现不 import `vllm_omni.diffusion.offloader`；`offloader/cuda_host_registration.py` 不改，`legacy` 的登记行为不变。

### 8. 共存期必改行

每一行只改变 `unified` 下的行为，`legacy` 一支是原来的语句、原样移进分支，`legacy` 下的行为和已有测试不变（01 第 9.3 节）。模型代码只经 `weight_stack.components.is_unified_stack()` 读总开关，每个 PR 新增 `tests/diffusion/weight_stack/coexistence/sites/<行编号>_<PR 编号>.py`。

| 行 | 位置 | `unified` 下的改动 | PR |
|---|---|---|---|
| C01 | `models/magi2/pipeline_magi2.py`：`_validate_native_topology`、`_Magi2StagedComponent`、`Magi2Pipeline._conditioning_memory_window` | `_validate_native_topology` 不执行由 `resolve_offload` 的结果决定的检查（拒绝 `MODEL_LEVEL`、逐层 offload 的 world size 和 HSDP 限制、分布式逐层与 HSDP 互斥、DP 大于 1 要求分布式逐层、AllGather 对 DP 和 TP 的要求），world size 为 1 视为合格拓扑，其余检查不变；`_Magi2StagedComponent` 不构造 `PinnedModuleStager`，`load_to_device()`、`offload_to_cpu()` 什么都不做，模块树和参数名不变；`_conditioning_memory_window` 不对 DiT 调用 `.to()`。内存窗口由 recipe 的策略（`dit` 用 `module`、`rank-local`）和使用触发给出（总体设计 9.3）。 | 04-10 |
| C03 | `models/minimax_h3/minimax_h3_transformer.py` 的 `MiniMaxH3DiTModel.__init__`；`latent_upscaler.py` 的 `MiniMaxH3LatentUpscaler.upscale` | `default_offload_adaln` 为假，SM120 上不再自动把 AdaLN 投影留在 Host（AdaLN offload 由策略 `"units": "adaln"` 表达，05 的 P23 拒绝显式开启旧开关）；`upscale()` 不调用 `self.resizer.to(...)`，放大器由 Offloader 放置。 | 04-11 |
| C06 | `models/minimax_h3/pipeline_minimax_h3.py` 的 `_offload_model_cpu_stage_output`，`pipeline_minimax_h3_decoder.py` 的 `forward` | `legacy` 下函数照旧看 `_model_cpu_offload_modules`。`unified` 下函数只看这次请求的输出个数，也就是 `num_outputs_per_prompt` 解析出的值：`MiniMaxH3Pipeline.forward` 在按种子循环之前把 `self._release_stage_outputs` 设为 `context["num_outputs"] > 1`（`context["num_outputs"]` 由 `_resolve_minimax_h3_num_outputs(sampling.num_outputs_per_prompt)` 得到）；`MiniMaxH3DecoderPipeline.forward` 按它收到的 latent 个数设置；`post_decode` 设为假，因为分步执行只允许一个输出。函数读 `getattr(self, "_release_stage_outputs", False)`，为真时按现有的方式释放（输出 rank 拷到 CPU，其余 rank 换成 meta 占位）。`_offload_model_cpu_stage_output` 本身只收到张量，拿不到请求参数，所以输出个数经这个属性传入；`execution()` 保证同一时刻只有一个请求在执行。 | 04-11 |
| C07 | `cache/exact_projection_cache.py` 的 `ExactProjectionCache`；`models/minimax_h3/adaln_cache.py` 的 `MiniMaxH3RuntimeAdalnCache.seed` | 签名按 6.4 改用参数身份加写入代数。 | 04-9 |
| C08 | `models/lingbot_video/pipeline_lingbot_video.py` 的 `LingBotVideoPipeline._generate` | `offload_vae_during_denoise` 为真时不执行挪走 VAE 的分支，用 `logger.warning_once` 记一次警告（总体设计 8.4）。 | 04-12 |
| C12 | 见下文 | 经 `fully_resident` 门控，编译模式改走 6.3 的通用编译函数。 | 04-8 |
| C14 | `models/cosmos3/mixed_precision/strategy.py` 的 `Nvfp4W4A4W4A16Strategy` | `validate_after_processing` 不搬动查找表；`materialize()` 在反量化之前，`kE2M1ToFloat_handle.val` 的设备与 `layer.weight` 的设备不同时，换成按设备缓存在模块级字典里的那一份。 | 04-9 |
| C17 | `worker/diffusion_model_runner.py` 的 `_execute_request_list` 和 `_execute_stepwise_core` | 不调用 `resolve_offload_strategy`；HSDP 时用 `no_grad`，否则用 `self.weight_stack.runtime.grad_mode()`；现有 `with grad_context:` 包住的整段（请求准备、缓存刷新、`pipeline.forward` 和分步执行的各步）同时包在 `runtime.execution()` 里；`execute_model`、`execute_model_batch`、`execute_stepwise`、两种执行模式下的 `profile_run` 和 `ARDiffusionModelRunner` 的预热都经这两个方法，所以走同一条路径（01 第 8.1 节）。没有运行时（`load_format` 为 `dummy`）时与 `legacy` 相同。 | 04-6 |

C12 的各个位置。只持有组件模块、没有 pipeline 引用的对象，由 pipeline 的构造函数在 `unified` 一支注入一个零参数的判断函数，函数持有 pipeline 的弱引用，第一次捕获之前求值，组件名由 `catalog(pipeline).module_owner(<模块路径>)` 给出：

- Qwen-Image-2.1：管理器由 `QwenImage21Transformer2DModel.__init__` 建出（`self._decode_graph_manager`），`QwenImage21Pipeline.__init__` 在 `unified` 一支、transformer 建好之后，管理器不为 `None` 时把判断函数（组件为 `transformer` 的所属组件）交给它。`QwenImage21DecodeGraphManager._offload_reason` 在 `unified` 下不 import `SequentialOffloadHook`，不读 `model_level_offload`：判断函数为假时返回原因；为真时，transformer 的 `HookRegistry` 里有覆写 `new_forward` 的钩子（`HookRegistry._new_fwd_impl_hook` 不为 `None`，现有的 TeaCache、MagCache、SeaCache 根钩子都覆写它）时返回原因，否则返回 `None`。Offloader 的钩子不覆写 `new_forward`（5.1），`fully_resident` 的 transformer 上的触发钩子因此不阻止捕获。`_check_static_eligibility` 里对块的钩子检查不变：`fully_resident` 组件的块上没有 Offloader 的钩子。
- SenseNova-U1：`_decode_context` 在 `paged_decode_supported` 之后，`language_model` 的所属组件不是 `fully_resident` 时返回 `None`，走 eager 解码。
- AuK：`AuKCUDAGraphWrapper` 和 `AuKVAEDecodeGraph` 在捕获之前查询注入的判断函数（`dit`、`vae` 的所属组件），为假时走 eager。
- OmniVoice：`OmniVoicePipeline.denoise_step` 的 `use_cuda_graph` 另外要求 `generator` 的所属组件是 `fully_resident`，结果在第一次求值后缓存。
- Ming-Image：`setup_compile` 照旧调用 `regionally_compile(..., mode="reduce-overhead")`，由 6.3 降级；`_uses_cudagraph_trees` 取 DiT 组件的 `fully_resident`。
- DreamZero：`setup_compile` 里 `self.text_encoder.forward`、`self.image_encoder.model.visual.forward` 和 `self.transformer.blocks` 里每个块的 `block.forward = torch.compile(...)` 改为 `compile_module_forward(module, ...)`；`self.vae._encode = torch.compile(self.vae._encode, ...)` 改为 `compile_callable(self.vae._encode, reads=(self.vae,), ...)`。块上有块环钩子时，`compile_module_forward` 编译 `_omni_original_forward`，钩子不进编译图。
- InternVLA-A1：`InternVLAA1.__init__` 在 `compile_model` 为真时，把 `self.sample_actions` 设为 `compile_callable(原方法, reads=(self,), mode=config.compile_mode)`，构造期间不编译。

### 9. 故障注入点与启动日志

- 本 RFC 在模块内部增加的注入点，命名规则和 `FaultPlan` 与 01 第 10 节相同：名字以 `offload_` 开头：`offload_prepare_slots`（4.1 第 4 步之前）、`offload_commit_register`（4.3 第 2 步之前）、`offload_commit_hooks`（4.3 第 7 步之前）、`offload_runtime_load`（5.3 的装入之前）、`offload_runtime_gather`（5.4 发起拼接之前）。`offload_runtime_*` 的 `raise` 让这次 `execution()` 走异常退出。
- 启动日志的事件名和字段按 05 第 15.2 节：`component_plan`、`component_skipped`、`unit_resident`、`unit_transfer`、`fully_resident`。05-6 合入之前，事件用 logger `vllm_omni.diffusion.weight_stack.offload` 按同样的事件名和字段输出；`record_device_resources` 的调用由 05-6 加进 `commit()` 的第 8 步和 `close()` 的最后一步之后（05 第 16.2 节）。

## 与其他子 RFC 的接口

### 本 RFC 提供

| 提供的内容 | 使用方 | 约定 |
|---|---|---|
| `weight_stack.offload.create_offloader(ctx)`、`PlacementPlan`、`Unit`、`HOOK_NAMES` | 01 | 第 1、2 节；`HOOK_NAMES` 供 01 的 T4。 |
| `PlacementPlan.collective_signature()`、`unmatched_policy_keys()`、`bind_groups(groups)` | 01 | 第 2 节；`bind_groups` 组大小或位置不符时报 `GroupConfirmationError`。 |
| `PlacementPlan.host_request()`、`Unit.unit_id` | 03 | 03 第 4.1 节的结构；Unit 划分不依赖 `LocalRecord` 的 pid、可用内存和空闲显存。 |
| `PlacementPlan.cold_start(component)`、`sharded_tensor_names()`、`placement(name)`、`units`，以及 `Unit.roots`、`Unit.transport` | 02 | 预检核对冷启动路径；收尾第 7 步换占位；瘦路径的读取器按 `units` 里 `allgather` Unit 的规划顺序产出（02 第 7.2 节）。 |
| `LoadTransaction.target()`、`commit()`、`abort()` | 01、02 | 4.2～4.4；`prepare_materialization()` 失败时不接管 `storage`，`commit()` 失败时资源全部留在启动事务里。 |
| `OffloadRuntime` 的 `grad_mode()`、`execution()`、`before_sleep()`、`after_wake()`、`close()` | 01 | 5.6～5.8。 |
| `Offloader.setup_compile()` | 01 | 6.2。 |
| `compile.py` 的驻留表和三个通用编译函数 | 模型代码 | 6.3。 |
| `ExactProjectionCache.bind_weight_generation()` | 04 自己的 `commit()` | 6.4。 |
| 平台的四个内存登记方法、`HostRegistrationFailure` | 04 自己 | 第 7 节。 |

### 本 RFC 依赖的约定

| 子 RFC | 约定 |
|---|---|
| [01](./01-startup-transaction.md) | 第 2 节的 Protocol 和数据类型；`ResourceLedger` 的 `register`、`hand_over`、`transfer`、`close`；驱动在第 1 点调用 `collective_signature()`、`unmatched_policy_keys()` 和 `bind_groups()`，在第 12 步调用 `setup_compile()`；`MemoryPools.slots()` 对应 `weight_slots` 池；worker 在睡眠前后调用 `before_sleep()`、`after_wake()`；`StartupContext.policy` 为 `None` 表示不做 offload；`StartupContext.faults`；05-3 合入之前 `unified` 下的 offload 配置在参数解析阶段被拒绝。 |
| [02](./02-unified-loader.md) | `ComponentCatalog` 的字段和三个写方法；`require_complete()` 报 `PlanningError`（归属违例和保持 fp32 的层配了量化）；`TensorDecl.module` 是包装前的规范路径、解析得到包装层；`ComponentCatalog.declaration_order` 给出组件的声明顺序（02 第 2.7 节），`resolve_selection()` 的 `refs` 和常驻层计数用它，`entries` 只用于查找；`weight_stack.components.pipeline_of(catalog) -> nn.Module` 返回以这个目录登记的 pipeline 对象，`plan()` 用它读模块树（读 02 第 2.8 节的进程级弱引用表），这个函数由 02-1 提供；`FinalManifest.live()` 给出规范张量和别名对象；`repeated_block_paths` 返回 `set_discovered_blocks()` 的记录；`fully_resident`、`component_paths`、`catalog`、`weight_generation` 的语义按 02 第 2.8 节；动态 LoRA 管理器从包装层取槽位，只用 `copy_` 写入，每次写入后调用 `bump_weight_generation`。 |
| [03](./03-host-storage.md) | `HostRequest` 的结构（03 第 4.1 节），`cold_start` 为 `thin` 的组件没有 `replaced_at_finalize` 的条目；`HostStorage.note_written()` 由 Loader 调用，04 不调用；先全量后裁剪的组件在 `seal()` 里按 `flatten_shard_unit` 重新布局，`commit()` 按最终清单重建块环和分片槽位的视图（4.3）；`view()` 对等待者的共享张量和瘦路径的范围外张量返回 meta 张量；`ranges(unit)` 在 `seal()` 之后有效，本机部分按地址合并相邻的张量、一个文件里可以有几段，分片部分每个类型组一段（03 第 7.6 节）；`MemoryRange.read_only` 与 `Mapping.kind` 一致；`flatten_shard_unit(tensors, group_size)` 接受带 `name`、`dtype`、`nbytes` 属性的对象，给出每个类型组的元素数、段长和组内偏移；`storage.close()` 可以多次调用。 |
| [05](./05-configuration.md) | `resolve_selection(policy, refs, *, pipeline)` 按 `declaration_order` 的 `refs` 调用，报 L02 到 L05，`Selection.unmatched` 按本 rank 判断；`HostPolicy.pin_cpu_memory` 和 `registration_limit_bytes`；P15 到 P18 和 P29 在参数解析阶段拒绝 `module` 配 `allgather`、`resident_layers` 大于 0 配 `allgather`、`units` 配 `module`、`units` 配 `resident_layers` 大于 0、全量编译配 offload；第 15.2 节的启动日志事件；05-6 在 `commit()` 末尾和 `OffloadRuntime.close()` 之后加入 `record_device_resources` 的调用（05 第 16.2 节）。 |

## 代码组织

新增文件：

```text
vllm_omni/diffusion/weight_stack/offload/
  __init__.py          create_offloader、PlacementPlan、Unit、HOOK_NAMES
  plan.py              第 2 节的数据类型和 PlacementPlan 的方法
  discovery.py         重复块发现（3.4）
  planner.py           UnifiedOffloader.plan()：选择、Unit 划分、让出关系、Transport、AllGather 组推算（3.2～3.9）
  layout.py            加速卡布局、device_bytes、块环和分片槽位的大小（3.8）
  transaction.py       prepare_materialization()、UnifiedLoadTransaction（第 4 节）
  registration.py      登记计划、区间对齐合并、登记与撤销（5.4）
  staging.py           两个暂存区
  hooks.py             HOOK_NAMES 和三个钩子（5.1）
  ring.py              两个块环（5.2）
  transport.py         rank-local 和 allgather 的装入（5.4）
  guard.py             提交后的守卫（5.5）
  runtime.py           UnifiedOffloadRuntime（5.6～5.8）
  compile_setup.py     setup_compile（6.2）
vllm_omni/platforms/cuda/host_registration.py
tests/diffusion/weight_stack/offload/
  conftest.py          合成模块树、HostStorage 替身、注入的选择函数
  test_plan_none.py  test_discovery.py  test_discovery_expectations.py  test_planner.py
  test_layout_device.py  test_collective_signature.py  test_transaction.py  test_registration.py
  test_guard.py  test_hooks.py  test_ring_rank_local.py  test_use_trigger.py  test_runtime_lifecycle.py
  test_sleep.py  test_ring_allgather_multiproc.py  test_compile_setup.py  test_compile_downgrade.py
  test_cache_signature_unified.py  test_offload_e2e.py
  expectations/discovery.yaml
```

`UnifiedOffloader(ctx, *, select=None)` 的 `select` 默认在 `plan()` 里、`policy` 不为 `None` 时从 `weight_stack.policy` 取 `resolve_selection`。05-3 合入之前，`unified` 下的 offload 配置在参数解析阶段被拒绝（01 第 1.3 节），生产路径上 `policy` 总是 `None`；本 RFC 的测试传入 `conftest.py` 里按 05 第 11 节字段写的选择函数替身，所以本 RFC 的 PR 不依赖 05 的 PR。

修改的文件：

| 文件 | 改动 |
|---|---|
| `vllm_omni/diffusion/compile.py` | 驻留表、三个通用编译函数、`regionally_compile` 的 `unified` 一支（6.3）。 |
| `vllm_omni/platforms/interface.py`、`vllm_omni/platforms/cuda/platform.py` | 第 7 节的四个方法和异常类型。 |
| `vllm_omni/diffusion/cache/exact_projection_cache.py` | 6.4。 |
| `vllm_omni/diffusion/worker/diffusion_model_runner.py` | C17。 |
| 第 8 节列出的模型文件 | C01、C03、C06、C07、C08、C12、C14。 |

## 实现拆分

本 RFC 的 PR 都依赖 01-1 到 01-7 全部合入，下表不再重复；都不依赖 05 的 PR。每个 PR 只改变 `unified` 下的行为，`legacy` 下的行为和已有测试不变，不修改已有测试文件；承担必改行的 PR 在描述里写明行编号和这一行的 `legacy_removal`。

| PR | 内容 | 主要文件 | 测试 | 依赖 | 大小 | 必改行 |
|---|---|---|---|---|---|---|
| 04-1 | 包骨架、第 2 节的数据类型、`create_offloader`、`HOOK_NAMES`；`policy` 为 `None` 的规划；`prepare_materialization()` 的准入、叠加层槽位、非持久 buffer；`target()`；`commit()` 的校验、槽位绑定、`set_fully_resident`、守卫；`abort()`；没有 Unit 的运行时。 | `__init__.py`、`plan.py`、`planner.py`、`transaction.py`、`guard.py`、`runtime.py` | `test_plan_none.py`、`test_transaction.py`、`test_guard.py` | 02-1 | M | 无 |
| 04-2 | 重复块发现、`repeated_blocks` 覆盖、`set_discovered_blocks`；每个注册的 pipeline 的发现期望表。 | `discovery.py` | `test_discovery.py`、`test_discovery_expectations.py` | 04-1、02-2 | M | 无 |
| 04-3 | 选择、Unit 划分、`units`、常驻层、共用存储、让出关系、Transport、AllGather 组推算、冷启动路径、占用估算、`host_request()`、`collective_signature()`、`unmatched_policy_keys()`、`bind_groups()`、`describe()`。 | `planner.py`、`layout.py`、`plan.py` | `test_planner.py`、`test_layout_device.py`、`test_collective_signature.py` | 04-2、03-3 | L | 无 |
| 04-4 | 平台的内存登记接口和 CUDA 实现。 | `platforms/interface.py`、`platforms/cuda/` | `test_registration.py` 里的平台用例 | 无 | S | 无 |
| 04-5 | 启动事务的完整实现：登记计划、准入的全部项、`slots()` 池里的固定槽位、流和事件、暂存区、块环视图、`commit()` 的内存登记、Unit 记录和占位。 | `transaction.py`、`registration.py`、`staging.py` | `test_transaction.py`、`test_registration.py` | 04-3、04-4 | L | 无 |
| 04-6 | 运行期：钩子、使用触发与让出、`rank-local` 块环、单次执行、梯度模式、关闭、睡眠；runner 改读 `grad_mode()` 并经过 `execution()`。 | `hooks.py`、`ring.py`、`transport.py`、`runtime.py`、`diffusion_model_runner.py` | `test_hooks.py`、`test_ring_rank_local.py`、`test_use_trigger.py`、`test_runtime_lifecycle.py`、`test_sleep.py` | 04-5 | L | C17 |
| 04-7 | `allgather`：分片槽位、编号断言、拼接流、不可用标记、跨组件的预取限制。 | `transport.py`、`ring.py` | `test_ring_allgather_multiproc.py` | 04-6 | M | 无 |
| 04-8 | 编译设置、驻留表、`resolve_compile_mode`、`regionally_compile` 的 `unified` 一支、`compile_module_forward`、`compile_callable`；C12 的七个位置。 | `compile_setup.py`、`compile.py`、第 8 节的 C12 文件 | `test_compile_setup.py`、`test_compile_downgrade.py`，每个位置一个 `unified` 用例 | 04-6、02-10、02-14、02-15、02-16 | M | C12 |
| 04-9 | 以权重为键的缓存签名和写入代数的绑定；Cosmos3 NVFP4 查找表。 | `exact_projection_cache.py`、`adaln_cache.py`、`cosmos3/mixed_precision/strategy.py`、`transaction.py` | `test_cache_signature_unified.py`，NVFP4 查找表的 `unified` 用例 | 04-5、02-10 | S | C07、C14 |
| 04-10 | MAGI-2。 | `magi2/pipeline_magi2.py` | `unified` 用例 | 04-6、02-12 | S | C01 |
| 04-11 | MiniMax H3。 | `minimax_h3_transformer.py`、`latent_upscaler.py`、`pipeline_minimax_h3.py`、`pipeline_minimax_h3_decoder.py` | `unified` 用例 | 04-6、02-11 | S | C03、C06 |
| 04-12 | LingBot-Video。 | `lingbot_video/pipeline_lingbot_video.py` | `unified` 用例 | 04-6、02-18 | S | C08 |
| 04-13 | 端到端和 GPU 测试：与完整加载的输出对比、全部组件 offload 的冒烟、睡眠唤醒、故障注入。 | 只有测试 | `test_offload_e2e.py` | 04-7、04-8、04-9、02-5、03-5、03-6 | M | 无 |

```text
01-1..01-7
  04-1（另依赖 02-1）
    04-2（另依赖 02-2）
      04-3（另依赖 03-3）
        04-5（另依赖 04-4）
          04-6
            04-7
            04-8（另依赖 02-10、02-14、02-15、02-16）
            04-10（另依赖 02-12）、04-11（另依赖 02-11）、04-12（另依赖 02-18）
          04-9（另依赖 02-10）
              04-13（依赖 04-7、04-8、04-9、02-5、03-5、03-6）
  04-4
```

- 04-8 依赖 04-6，因为 `setup_compile` 在 `runtime.execution()` 里调用 pipeline 的 `setup_compile()`；依赖 02-14、02-15、02-16，因为 DreamZero、Ming-Image、AuK、OmniVoice、InternVLA-A1 的 `unified` 用例要在只建结构的上下文里构造这些 pipeline。

- 03-9 依赖 04-5；05-3 依赖 04-3、04-5、04-6、04-7；05-5 依赖 04-2；05-6 依赖 04-5。本 RFC 的 PR 不依赖这些 PR，依赖图无环。

## 测试计划

CI 等级和标记按 `docs/contributing/ci/test_system_overview.md` 和 `pyproject.toml`：L1 是 `core_model` 加 `cpu`，L2 是 `core_model` 加 `hardware_test(res=...)`，L3 是 `advanced_model`，L4 是 `full_model`；本 RFC 的测试都带 `diffusion`，多卡测试带 `parallel` 和 `cards_<n>`。

**单元测试（L1）**

- 规划：`policy` 为 `None` 时没有 Unit、`fully_resident` 全为真（HSDP 组件为假）；五种 Unit 的划分表；`units` 的通配符匹配和嵌套报错；常驻层按角色计数、全部常驻时的事件；共用存储的合并和三种报错；让出关系表和按常驻处理；`triggers`；`shard_group` 的推算和自检报错；单机放不下时的 `PlanningError`；`plan()` 不改模型、只调用一次 `set_discovered_blocks`。
- 发现：合成模块树覆盖 B/4 规则、size 为 0 的元素、类型不同、没有覆写 `forward`、自定义可迭代模块、划出的子树、`repeated_blocks` 覆盖。
- 发现期望表：每个注册的 pipeline 按测试覆盖的拓扑和量化配置在只建结构的上下文里构造，发现结果与 `expectations/discovery.yaml` 逐项一致；现有声明里的块（`_repeated_blocks` 的类和 `_layerwise_offload_blocks_attrs`）全部被发现；容器本身不成为 Unit；用了 `repeated_blocks` 的组件同时记录自动发现的结果和覆盖的原因。
- 布局和签名：`device_bytes` 和 `buffer_bytes` 的公式；`collective_signature()` 改变 `LocalRecord` 的 pid 和可用内存时不变，改变 `group_size` 或张量形状时改变；`host_request()` 符合 03 的协议。
- 冷启动路径：`shapes_final` 为真且 `finalize_replaced` 为空时为 `thin`；`shapes_final` 为假，或 `finalize_replaced` 不为空时为 `full_then_trim`。
- 启动事务：用 01 的测试替身和 `HostStorage` 替身，覆盖准入的每一项、`DeviceCapacityError` 的内容、`prepare_materialization()` 失败时不接管 `storage`、`commit()` 第 1 步的每种不一致、`finalize_replaced` 的重新布局（放得进固定槽位时重建视图，放不进时报 `CommitError`）、`abort()` 的幂等和账本顺序（参数先改指向占位、再关闭 host 存储）、每个句柄只登记一次。
- 登记：区间的页对齐不越出 `Mapping`、同一映射内合并、上限按合并后的字节数、`read_only` 选择登记方式、平台不支持时走暂存区、确认失败时撤销。
- 守卫：`.to()`、`.cuda()`、`.cpu()`、`.float()`、`.half()`、`.bfloat16()`、`.type()`、`to_empty()` 和 `memory_format` 变化报错，`.to(执行设备)` 什么都不做，`close()` 后恢复。
- 钩子与运行时：钩子顺序与覆写 `new_forward` 的缓存钩子共存；执行作用域之外调用报错；并发和重入报错；前向抛出异常之后活动 Unit 和使用深度清零，下一次执行正常让出；pipeline 的 `setup_compile()` 在 `execution()` 里运行；关闭按 `ResourceKind` 顺序、可以多次调用，测试把 `torch.distributed` 的集合通信函数替换成报错的函数，断言关闭不调用它们。
- 编译：`resolve_compile_mode` 的降级表；`regionally_compile` 的两个分支；提交之前调用报错。
- 缓存签名：换绑参数存储、参数指向零元素占位时签名不变；`eval()` 里算的签名在提交之后仍然匹配；`bump_weight_generation` 之后签名改变；`legacy` 下签名不变；`offload/` 不 import `cache/`。

**GPU 测试（L2，`cards_1`）**

- 合成 DiT 上 `rank-local` 块环的输出与完整加载逐位一致，覆盖预测错（跳块、切换两个 DiT）和异常退出。
- 登记和暂存区两条路径的输出一致；`pin_cpu_memory` 为假时不登记、暂存区不锁页。
- 使用触发的装入和让出、`empty_cache` 次数、整组件模式下没有反复的回写和锁页分配。
- 捕获期间调用钩子报错；`fully_resident` 组件的 CUDA graph 捕获和重放正常。
- 睡眠再唤醒后输出一致，常驻部分和固定槽位的地址不变，host 存储不变。
- 每个必改行一个 `unified` 用例；对应的已有测试在 `legacy` 下不变。

**多进程测试（L2 和 L3，`parallel`，`cards_2`、`cards_4`）**

- `allgather` 块环的输出与完整加载一致，覆盖每次前向第一个块的拼接、预测错、两个组件各自的 AllGather 组。
- 只在部分 rank 上声明的 `rank-local` 流式 Unit 不打乱 `allgather` 的拼接顺序。
- 注入编号不一致时设备侧断言失败，运行时不可用，之后的 `execution()` 报 `RuntimeFault`；关闭不等待拼接流。
- `bind_groups()` 在组大小不符时报 `GroupConfirmationError`。
- 第 9 节的每个注入点一个用例，多 rank 下整组一起失败、不死锁。

**端到端测试（L3 和 L4）**

- 每个支持的 pipeline 一个“全部组件做 offload”的冒烟测试，覆盖编译和异常路径（L3，合成或小模型；L4，真实 checkpoint）。
- 有端到端或每晚测试的“模型 × offload 策略”组合在 `unified` 下与完整加载的输出一致，记录 Host 和加速卡的峰值（L4）。

## 验收标准

- 各种策略产生互不重叠的 Unit，每个张量恰好属于一个 Unit 或常驻加速卡。
- `rank-local` 和 `allgather` 的输出与完整加载的基线一致。
- 块环的内存有固定上限（`buffer_bytes` 和分片槽位），正确处理真实执行顺序、跳块和异常。
- 每个注册的 pipeline 在测试覆盖的拓扑和量化配置下，重复块发现的结果与期望表一致，现有声明里的块全部被发现。
- 只在部分 rank 上声明的 `rank-local` 流式 Unit 不打乱 `allgather` Unit 的拼接顺序。
- 每个支持的 pipeline 都有“全部组件做 offload”的冒烟测试，覆盖编译和异常路径。
- 整组件模式下没有反复的回写和反复的锁页分配；任何策略都不生成完整的锁页副本。
- 提交后对被管理的模块改设备或改类型（包括 `.float()`、`.half()`）报错。
- 不是 `fully_resident` 的组件不进 CUDA graph，cudagraph 编译模式降为 `default` 并记日志；钩子在 graph 捕获期间需要动作时报错。
- 关闭不恢复 CPU 权重，不发起集合通信；每个资源句柄只释放一次。
- 睡眠再唤醒后输出一致，常驻部分和固定槽位的加速卡地址不变，host 存储不变。
- C01、C03、C06、C07、C08、C12、C14、C17 都有位置文件和 `unified` 用例，`legacy` 下已有测试全部通过。

## 删除 legacy 时

umbrella RFC Job 6 删除 `legacy` 时，本 RFC 范围内删除：

- `vllm_omni/diffusion/offloader/` 整个包：三个后端、`block_discovery.py`、`component_utils.py`、`module_collector.py`、`module_residency.py`（`PinnedModuleStager`）、`offload_plan.py`（`OffloadPlan`）、`plan_resolver.py`、`startup.py`、`config.py` 的查询函数、`host_registration.py`、`cuda_host_registration.py`、`tensor_utils.py`；runner 里的 `enable_offload_backend` 和 `offload_backend.shutdown()`。
- `regionally_compile` 的 `legacy` 一支（按 `_repeated_blocks` 类名和 `_layerwise_offload_blocks_attrs` 匹配）。删除之前，只靠 `_layerwise_offload_blocks_attrs` 回退被编译的块类补进 `_repeated_blocks`（总体设计 9.3 列出的九个类），一个测试对比删除前后被编译的模块集合。
- `ExactProjectionCache._signature` 的地址签名；C07、C08、C12、C14、C17 和 C01、C03、C06 的 `legacy` 一支，按各行的 `legacy_removal`；`QwenImage21DecodeGraphManager._offload_reason` 对 `SequentialOffloadHook` 的读取。
- 模型里的 offload 专用声明：`_layerwise_offload_blocks_attrs`、模型声明的 `OffloadPlan`、常驻路径声明、`SupportsModelCpuOffload`；MAGI-2 的分阶段组件和内存窗口、MiniMax H3 的 AdaLN Host 暂存和 SM120 默认、放大器和 VAE 两半的搬运。
- 01 的 `HOOK_NAME_ALLOWLIST` 清空后，T4 只读 `HOOK_NAMES`。

## 风险

| 风险 | 影响 | 处理 |
|---|---|---|
| 覆写 `new_forward` 的缓存后端跳过块 | 跳块使块环频繁预测错，每次同步装入 | 后继表按真实进入顺序更新；端到端测试记录 TeaCache、MagCache 下的同步装入次数。 |
| 非 CUDA 平台不做内存登记 | 每个流式块每一步都经暂存区 H2D，比现有实现的锁页副本慢 | 支持矩阵写明这些平台走暂存区；平台实现第 7 节的方法并有验证测试后放开（总体设计 13）。 |
| 运行期 `tensor.data` 换绑与 `torch.compile` 的守卫 | 换绑触发重新编译 | 编译只包住块的原始 `forward`，块的形状不变；冒烟测试统计重新编译次数，与 `legacy` 的逐层 offload 相同。 |
| 内存登记的驱动开销 | 驱动按登记量占用内核内存 | 门槛测试记录登记前后的内核内存占用（总体设计 13）。 |
| `shard_group` 组件的 AllGather 组推算假设切分组编号等于候选位置模 s | 模型按别的方式给切分组编号时，`bind_groups()` 在第 1 点报 `GroupConfirmationError`，这个组件不能用 `allgather` | 推算只决定报错，不会静默错位；`test_planner.py` 和多进程测试覆盖 MiniMax H3 文本编码器的切分组布局。 |

## 反馈期

发布后一周。

## CC

Offloader、编译、分布式运行时、CUDA 与 NPU 平台、模型接入（MAGI-2、MiniMax H3、Qwen-Image-2.1、Cosmos3）的负责人。
