# [RFC]: 统一 Offloader

> 上级 RFC：[umbrella RFC](./00-umbrella.md)。设计依据：[总体设计](../LOADER_OFFLOADER_DESIGN.md) 第 7 节（7.1～7.8），以及 8.5、8.6 里归规划和 `prepare_materialization()` 的校验、9.3、第 10 节的加速卡部分、第 11 节、第 12 节第 9、11、13、14、15 条。
>
> 范围：`vllm_omni/diffusion/weight_stack/offload/`；`vllm_omni/diffusion/compile.py`；`vllm_omni/platforms/` 的内存登记接口；`vllm_omni/diffusion/cache/exact_projection_cache.py`；`vllm_omni/diffusion/worker/diffusion_model_runner.py` 的前向梯度模式；共存期必改清单里由本 RFC 承担的 C01、C03、C06、C07、C08、C12、C14、C17。
>
> 依赖：[子 RFC 01](./01-startup-transaction.md)（`interfaces.py` 的 `Offloader`、`LoadTransaction`、`OffloadRuntime`、`Topology`、`AllGatherGroups`、`StartupContext`、`MemoryPools`；`errors.py`；`ResourceKind` 和 `ResourceLedger`；`FaultPlan`；登记表和测试替身）；[子 RFC 02](./02-unified-loader.md)（`ComponentCatalog` 及其写方法、`TensorDecl`、`OverlaySlotDecl`、`FinalManifest`、查询接口 `component_paths`、`repeated_block_paths`、`fully_resident`、`weight_generation`）；[子 RFC 03](./03-host-storage.md)（`HostStorage.view`、`ranges`、`close`，`MemoryRange`、`Mapping`、`HostRequest` 协议、`flatten_shard_unit`、`LocalRecord`）；[子 RFC 05](./05-configuration.md)（`OffloadPolicy` 的解析结果、`pin_cpu_memory`、`host_registration_limit_gib`、参数解析阶段的组合检查）。

## 摘要

本 RFC 实现总体设计第 7 节的新 Offloader：`plan()` 是纯函数，从组件目录、拓扑和策略算出互不重叠的 Unit，以及每个 Unit 的 Trigger 和 Transport；`prepare_materialization()` 在第一次写模型之前做加速卡容量准入，拿齐块环缓冲区、分片槽位、叠加层槽位、流、事件和暂存区，返回启动事务；`commit()` 校验最终清单，按 `MemoryRange` 做内存登记，填写 `fully_resident`，经 `HookRegistry` 装上钩子和提交后的守卫，把资源整体移交 `OffloadRuntime`。运行期由两个按 Transport 分开的块环、使用触发和让出关系、单次执行规则、固定顺序的关闭和按显存类别分工的睡眠模式组成。编译设置在提交后运行，cudagraph 模式和模型自己的 CUDA graph 按 `fully_resident` 门控，以权重为键的缓存改用参数身份加写入代数。本 RFC 还承担共存期必改清单里与这些机制冲突的八行。

## 动机

umbrella RFC 的“背景与问题”已经说明总体问题，这里只列本 RFC 要取代的现有代码事实：

1. 现有的三个后端各自实现驻留和搬运：`sequential_backend.py` 的 `SequentialOffloadHook`（钩子名 `sequential_offload`）、`layerwise_backend.py` 的 `LayerwiseOffloadHook`（`layerwise_offload`）、`distributed_layerwise_backend.py` 的 `DistributedLayerwiseOffloadHook`（`distributed_layerwise_offload`），都在 `vllm_omni/diffusion/offloader/` 下。三者的锁页、登记和换绑各写一份，换绑经 `tensor_utils.set_tensor_storage` 写 `tensor.data`。
2. 模型按名字读 offload 钩子来决定行为：Qwen-Image-2.1 的 `QwenImage21DecodeGraphManager._offload_reason` 读 `SequentialOffloadHook._HOOK_NAME` 决定能否捕获解码 CUDA graph；SeaCache 的 `cache/seacache/hook.py` 按字符串 `"distributed_layerwise_offload"` 取钩子。
3. 模型自己搬运权重：MAGI-2 用 `_Magi2StagedComponent` 包住辅助组件、内部用 `PinnedModuleStager` 保留锁页副本，`Magi2Pipeline._conditioning_memory_window` 对 DiT 调用 `.to("cpu")`；MiniMax H3 的 `MiniMaxH3LatentUpscaler.upscale` 在每次放大前后 `.to()` 搬运，`MiniMaxH3DiTModel` 的构造函数在 SM120 上默认把 AdaLN 投影留在 Host；`LingBotVideoPipeline._generate` 按请求参数 `offload_vae_during_denoise` 把 VAE `.to("cpu")`。
4. 编译目标和模式与驻留无关：`regionally_compile`（`vllm_omni/diffusion/compile.py`）按 `_repeated_blocks` 的类名和 `_layerwise_offload_blocks_attrs` 匹配块；Ming-Image 的 `setup_compile` 固定用 `mode="reduce-overhead"`，DreamZero 的编码器和 VAE 用 `reduce-overhead`，InternVLA-A1 在 `InternVLAA1.__init__` 里按 `compile_mode`（默认 `max-autotune`）编译 `sample_actions`。权重被换出时这些 CUDA graph 读到的是过期地址。
5. runner 按 offload 策略选梯度模式：`DiffusionModelRunner._execute_request_list` 在 `resolve_offload_strategy` 为分布式逐层时用 `no_grad`，否则用 `inference_mode`（HSDP 除外）。
6. 以权重为键的缓存用存储地址和设备做签名：`ExactProjectionCache._signature` 含 `data_ptr()`、`_version` 和 `device`。块被换入换出后地址变化，缓存每次都不命中。
7. Cosmos3 的 `Nvfp4W4A4W4A16Strategy.validate_after_processing` 只在加载后按权重的设备放置一次 NVFP4 查找表；权重之后在 Host 和加速卡之间移动时，查找表留在原设备上。

## 设计

本节所有要求只对 `unified` 成立。`legacy` 下本 RFC 的新代码都不运行；本 RFC 修改的已有文件里，`legacy` 一支是原来的语句。

### 1. 模块与入口

`weight_stack/offload/__init__.py` 导出 `create_offloader(ctx: StartupContext) -> Offloader`、`PlacementPlan`、`Unit` 和 `HOOK_NAMES`，由 01 的 `EntryPointFactory` 按名字取用。`UnifiedOffloader` 实现 01 第 2 节的 `Offloader` 协议，对象本身不持有资源。

```text
启动步骤（总体设计 4.2）    调用                                                  结束时的状态
4   规划                    offloader.plan(catalog, topology, policy)            PLANNED
    第 1 点                 plan.collective_signature()、plan.bind_groups(groups)
8   运行期资源              offloader.prepare_materialization(model, plan, storage)  返回启动事务
9   写入（02）              transaction.target(tensor_key)
11  提交                    transaction.commit(final_manifest)                   COMMITTED
12  编译设置                offloader.setup_compile(model, catalog, runtime)
服务                        runtime.grad_mode()、runtime.execution()、before_sleep()、after_wake()
关闭                        runtime.close() 或 transaction.abort()
```

`offload/` 只 import `weight_stack.interfaces`、`weight_stack.errors`、`weight_stack.startup`（`ResourceKind`、`ResourceLedger`、`FaultPlan`）、`weight_stack.components`（02 的查询接口和数据类型）、`weight_stack.host` 的 `flatten_shard_unit`、`vllm_omni.diffusion.hooks`、`vllm_omni.diffusion.compile`、`vllm_omni.diffusion.cache.exact_projection_cache`、`vllm_omni.platforms`。它不 import `loader/`、`vllm_omni.diffusion.model_loader`、`safetensors`、`vllm.model_executor.model_loader`、`vllm_omni.diffusion.offloader`（01 的 T1）。Offloader 不调用模型的任何加载后钩子（总体设计 3.2 第 2 条）。

### 2. 数据结构

```python
# vllm_omni/diffusion/weight_stack/offload/plan.py
class UnitKind(enum.Enum):
    MODULE = "module"            # 整组件
    BLOCK = "block"              # 一个流式的重复块
    GROUP = "group"              # units 选中的 unit_groups 里匹配的一个子模块
    RESIDENT_LAYERS = "resident" # 一个组件里取到的常驻层组
    REMAINDER = "remainder"      # 非 dit 角色的 layer 组件里，重复块以外的一个直接子模块

class Trigger(enum.Enum):
    USE = "use"                  # 使用触发
    RING = "ring"                # 块环

class Transport(enum.Enum):
    RANK_LOCAL = "rank-local"
    ALLGATHER = "allgather"

@dataclass(frozen=True)
class Unit:
    unit_id: str                          # "<组件名>/<kind>/<规范路径或序号>"，只由组件目录和策略决定
    component: str
    kind: UnitKind
    trigger: Trigger
    transport: Transport
    roots: tuple[str, ...]                # Unit 覆盖的子树的规范模块路径
    hook_targets: tuple[str, ...]         # 装钩子的模块的规范路径，按注册顺序（5.1）
    tensors: tuple[str, ...]              # 规范名，不含别名，按（类型名，规范名）排序
    nbytes: int                           # tensors 的存储范围之和
    device_bytes: int                     # 装入后在加速卡上的占用（3.8）
    order: int                            # 在本 Transport 的流式序列里的位置；USE 的 Unit 为 -1

@dataclass(frozen=True)
class ComponentPlacement:
    name: str
    role: Role
    selected: bool                        # 策略选中了这个组件
    mode: Literal["module", "layer"] | None
    units: tuple[str, ...]                # unit_id，规划顺序
    resident_tensors: tuple[str, ...]     # 常驻加速卡的规范名（“其余部分”）
    hsdp_managed: bool
    cold_start: Literal["thin", "full_then_trim"] | None   # 只对有 allgather Unit 的组件不为 None
    group_size: int | None                # 推算的 AllGather 组大小 N
    group_index: int | None               # 本 rank 在组里的位置 r
    fully_resident: bool                  # 规划给出的值，commit() 原样写进目录（6.1）
    discovered_blocks: tuple[str, ...]    # 重复块容器的路径，写法与 repeated_blocks 相同
    notes: tuple[str, ...]                # 写进启动日志的规划说明（3.9）

@dataclass(frozen=True)
class RingPlan:
    transport: Transport
    buffer_bytes: int                     # 最大的两个 device_bytes 之和
    sequence: tuple[str, ...]             # 流式 Unit 的初始顺序，后继表由它生成（5.2）
    wraps: bool                           # rank-local 为真：最后一个接回第一个；allgather 为假
    shard_slot_bytes: int                 # allgather：一个分片槽位的大小；rank-local 为 0

class PlacementPlan:
    components: Mapping[str, ComponentPlacement]  # 按组件名排序
    units: Mapping[str, Unit]                     # 按规划顺序
    rings: Mapping[Transport, RingPlan]           # 没有某一类流式 Unit 时不含这一项
    yielders: Mapping[str, frozenset[str]]        # unit_id 到“使用时会让它让出的组件”（3.6）
    triggers: frozenset[str]                      # 需要装触发钩子的组件
    device_demand: DeviceDemand                   # 加速卡准入式的各项（4.1），可用显存除外
    registration_order: tuple[str, ...]           # 内存登记的优先顺序（5.4）
    def placement(self, tensor_name: str) -> TensorPlacement: ...
    def host_request(self) -> HostRequest: ...    # 03 第 4.1 节的结构
    def collective_signature(self) -> Mapping[str, str]: ...
    def bind_groups(self, groups: AllGatherGroups) -> None: ...
    def cold_start(self, component: str) -> Literal["thin", "full_then_trim"] | None: ...
    def sharded_tensor_names(self) -> frozenset[str]: ...
    def describe(self) -> PlanReport: ...         # 启动日志的规划部分，格式归 05

@dataclass(frozen=True)
class TensorPlacement:
    residency: Literal["host", "device", "hsdp"]
    unit_id: str | None                   # residency 为 host 时不为 None
    transport: Transport | None
    backend: Literal["local", "sharded"] | None
```

- `PlacementPlan` 在 `plan()` 返回之后不可变，只有 `bind_groups()` 写入一次进程组引用，第二次调用报 `LifecycleError`。
- `placement(name)` 对目录里的每个规范名和别名都有定义：别名返回它的规范张量的结果。不在目录里的名字报 `LifecycleError`。
- `host_request()` 实现 03 的 `HostRequest` 协议：`units` 是全部 `residency` 为 `host` 的 Unit，`backend` 在 `allgather` 时为 `sharded`、否则为 `local`；`sharded` 给出每个有 `allgather` Unit 的组件的 `group_size`、`group_index`、`cold_start`；`staging_bytes` 在本 worker 有 Host 驻留的 Unit 时为 2 × 256 MiB，否则为 0；`transient_bytes` 取本 worker Host 驻留张量里最大的单个张量的字节数与 `catalog.write_transient.quant_layer_bytes` 中的较大者；`shapes_final` 为全部本机后端组件的 `CatalogEntry.shapes_final` 的与。
- `collective_signature()` 只含有 `allgather` Unit 的组件。每个组件的值是规范 JSON 的 blake2b 摘要（`digest_size=16`），覆盖：按 `order` 排列的每个 `allgather` Unit 的 `unit_id`、张量的（规范名、形状、类型、步长、存储范围）、`flatten_shard_unit` 给出的各类型组的元素数和段长、`group_size`、`Transport`。它不含 `rank-local` 的 Unit、`LocalRecord` 和任何地址。
- `bind_groups(groups)`：对每个有 `allgather` Unit 的组件，在 `groups[component]` 里找含本 rank 的组；找不到、组的大小不等于 `group_size`、或本 rank 在 `ranks` 里的位置不等于 `group_index` 时报 `GroupConfirmationError`，写明组件名、推算值和实际值。找到时记下 `AllGatherGroup.group`，运行期的拼接用它。

### 3. 规划

#### 3.1 输入与纯度

`plan(catalog, topology, policy)` 只读：组件目录（每个 `CatalogEntry` 的声明、根、划出的子树、张量表、`hsdp_managed`、`shapes_final`；`overlay_slots`；`write_transient`）、经目录解析得到的模块对象（类型、`forward` 是否被定义、子模块的注册顺序）、`topology.coordinates` 和本 rank 编号、05 解析出的策略、`ctx.od_config` 里与平台无关的设置。

- 它不分配大块资源，不改模型，不装钩子，不读空闲显存。
- Unit 的划分、`unit_id`、布局和 `collective_signature()` 不依赖 `LocalRecord` 的 pid 和可用内存（03 的规划指纹约定）。唯一读 `LocalRecord.available_bytes` 的是 3.3 第 6 条的检查，它只决定是否报错，不改变规划。
- 它对目录唯一的写操作是 `catalog.set_discovered_blocks()`（3.4）。
- `policy` 为 `None` 时按不做 offload 规划：没有 Unit，全部张量常驻加速卡（HSDP 管理的除外），`rings` 为空。

#### 3.2 步骤

```text
1  catalog.require_complete()                      归属违例或保持 fp32 的层配了量化时报 PlanningError，列出全部路径
2  解析策略：对每个组件求（selected、mode、weight_transfer、resident_layers、units）（3.3）
3  对每个不由 HSDP 管理的组件做重复块发现，记录结果（3.4）
4  按组件划分 Unit，处理共用存储和组件根上的参数（3.5）
5  推算 AllGather 组，选冷启动路径（3.7）
6  建让出关系表；没有任何让出者的使用触发 Unit 按常驻处理（3.6）
7  计算 device_bytes、块环、分片槽位和加速卡准入的各项（3.8）
8  计算 fully_resident、登记顺序，生成规划说明（3.9）
```

#### 3.3 组件选择与校验

1. 05 的策略对每个组件给出生效的设置：组件名的设置覆盖角色的设置，值为 `null` 表示不选。本 RFC 只读解析后的结果，不重新解析。
2. 策略里显式写出、但在本 rank 的目录里不存在的角色、组件名或 `units` 的组名，在规划阶段报 `PlanningError`；来自预设的角色不存在时忽略（总体设计 8.2）。显式引用是否存在按全部 rank 的并集判断不了，所以只按本 rank 判断：在部分 rank 上声明的组件（总体设计 5.1）只在声明它的 rank 上参与这条检查，其他 rank 上来自显式写法的不存在不报错。
3. 被 HSDP 管理的组件被选中时报 `PlanningError`，报错给出两个选择：保留 HSDP，不对这个组件做 offload；或者去掉 HSDP，给这个组件用 `layer` 配 `allgather`（总体设计 8.4）。
4. `resident_layers` 按角色计数，同一角色下被选中的 `layer` 组件的 `resident_layers` 必须相同，不同时报 `PlanningError`。`resident_layers` 大于 0 与 `units` 同时出现在一个组件上时报 `PlanningError`。
5. 选中按 `layer` 处理的组件找不到重复块时报 `PlanningError`，提示改用 `module` 或不选这个组件；`units` 选中的组没有匹配到任何带张量的模块时报 `PlanningError`，写明组名和模式。
6. 有组件使用 `allgather` 时，每个不是分片 Unit、又在 Host 驻留的部分，如果单个组件的字节数超过本 rank 所在机器上 `LocalRecord.available_bytes` 的最小值，报 `PlanningError` 并写明组件名（总体设计 6.4、8.5）。读数为 `None` 时跳过。完整的 Host 容量准入由 03 在第 1 点做。

#### 3.4 重复块发现

发现只看模块树，不读模型属性，对每个不由 HSDP 管理、没有声明 `repeated_blocks` 的组件运行一次：

```text
属于组件 c 的模块：c 的根子树，去掉划出给其他组件的子树
元素大小 size(m)：m 子树里属于 c 的最终张量（parameter、buffer、quant_output、fixed_overlay）的存储范围之和，
                 按本 rank 的形状和类型计算，已含 TP 切分和量化
候选列表 L：属于 c 的 nn.ModuleList（含子类）；L 里 size 大于 0 的元素至少 2 个、Python 类型相同，
            并且 type(元素).forward 不是 nn.Module.forward；size 为 0 的元素不参与判断
B = 组件 c 里全部候选列表的全部元素 size 的最大值
合格：max(size(元素) for 元素 in L) >= B / 4
遍历：从每个根按注册顺序（named_children）递归；遇到合格列表记为一组，不进入它的元素；
      其他模块继续往下，包括自定义的可迭代模块；不进入属于其他组件的子树
结果：按遍历顺序的组列表；每组里 size 大于 0 的每个元素是一个块
```

- `ModuleDict`、`ModuleList`、`ParameterList` 本身从不成为 Unit：钩子只装在块和块里的模块上。
- 存储范围按 `TensorDecl` 的形状、步长和类型计算，公式与 02 的 `ManifestEntry.nbytes` 相同；同一个张量的别名只计一次。
- 组件声明了 `repeated_blocks` 时不做自动发现，按列表顺序取每个容器里 size 大于 0 的元素。
- `plan()` 对每个组件调用一次 `catalog.set_discovered_blocks(component, paths)`，`paths` 是发现的容器路径，写法与 `repeated_blocks` 相同；声明了 `repeated_blocks` 的组件不调用。没有选中这个组件也照样发现和记录，因为编译设置经 `repeated_block_paths` 读它（6.2）。

#### 3.5 Unit 的划分

每个被选中的组件按下表划分。表外的张量（没选中的组件、选中组件的“其余部分”）常驻加速卡。

| 设置 | Unit | Trigger |
|---|---|---|
| `mode="module"` | 整个组件一个 `MODULE` Unit。 | 使用触发。 |
| `mode="layer"`，没有 `units` | 每个块一个 `BLOCK` Unit；`resident_layers` 取到的块合成本组件的一个 `RESIDENT_LAYERS` Unit。 | 块用块环，常驻层组用使用触发。 |
| `mode="layer"`，非 `dit` 角色 | 另外把组件根的每个直接子模块里、不属于任何块的张量组成一个 `REMAINDER` Unit。 | 使用触发。 |
| `mode="layer"`，有 `units` | 选中组里每个匹配的模块一个 `GROUP` Unit，不做块的划分。 | 块环。 |

- `units` 的模式按 02 的规则匹配：`*` 恰好匹配一段路径，模式相对组件根。匹配到的模块互相嵌套时报 `PlanningError`；不带张量的匹配忽略。`GROUP` Unit 按注册顺序排列。
- `resident_layers` 为 N 时，同一角色下被选中的 `layer` 组件按组件名的声明顺序（`declare_components` 的顺序，推导出的声明按推导顺序）、组件内的块按 `repeated_blocks` 的列表顺序（自动发现时按遍历顺序）排成一列，取前 N 个。N 不小于总块数时这些块全部归入常驻层组，并写进规划说明。
- `dit` 角色的 `layer` 组件里，块以外的张量常驻加速卡。
- 组件根模块自己直接持有的张量（不属于任何子模块）常驻加速卡，不进 Unit。
- 没有登记来源的非持久 buffer 不在目录里，常驻加速卡，由 `prepare_materialization()` 搬到执行设备（4.1 第 7 步）。`kind="data"` 的张量（登记了来源的非持久 buffer）同样常驻。
- `fixed_overlay` 张量和量化输出属于它所在层的 Unit；量化层量化前的参数不在目录里，不进 Unit。叠加层槽位不进任何 Unit，常驻加速卡。
- 动态 LoRA 的包装层整体属于原来那一层所在的 Unit：目录里的模块路径是包装前的规范路径，解析得到的是包装层本身。
- **共用存储**：规范张量和它的别名属于同一个 Unit。别名落在另一个 Unit 时两个 Unit 合并，合并后的 Unit 取这一组里规划顺序最前的那个的 `unit_id`；块里的张量与块外的张量共用时报 `PlanningError`；一组共用张量跨组件时，只有它们全部常驻加速卡才允许，否则报 `PlanningError`，写明两个组件和张量名。

#### 3.6 让出关系与按常驻处理

使用触发的 Unit 是 `MODULE`、`RESIDENT_LAYERS`、`REMAINDER`。一个组件的“使用”是它任何一个钩子被调用（5.1）。`yielders[u]` 是使用时让 u 让出的组件集合，按 u 所属组件的角色确定：

```text
u 所属组件的角色      使用以下组件时 u 让出
dit                   任何编码器角色（text_encoder、image_encoder、audio_encoder）的组件，不论是否被选中；
                      被选中的 vae；本 rank 上的其他 dit 组件
编码器角色            本 rank 上的任何 dit 组件
vae（被选中）         任何 dit 组件；任何编码器角色的组件；其他被选中的 vae
```

- 编码器之间不互相让出。没被选中的 `vae` 和 `other` 角色的组件不在任何 `yielders` 里。
- 只考虑本 rank 目录里存在的组件。`yielders[u]` 为空的 Unit 按常驻处理：它的张量改为常驻加速卡，不分配 host 存储，写进规划说明（总体设计 7.2）。这一步按本 rank 的组件集合做，不同 rank 的结论可以不同。
- `triggers` 是在某个 `yielders` 里出现、本身又没有使用触发 Unit 或块环 Unit 的组件。这些组件只在根模块上装触发钩子（5.1），触发钩子不装入任何权重。

#### 3.7 Transport、后端与冷启动路径

- `BLOCK` Unit 的 Transport 取组件的 `weight_transfer`；`GROUP`、使用触发的 Unit 一律是 `rank-local`（05 在参数解析阶段拒绝 `module` 或 `resident_layers` 配 `allgather`）。`units` 配 `allgather` 时 `GROUP` Unit 用 `allgather`。
- `rank-local` 的 Host Unit 用本机后端，`allgather` 的 Unit 用分片后端。
- **AllGather 组的推算**：候选 rank 是 `topology.coordinates` 里 TP 编号与本 rank 相同、启用专家并行时 EP 编号也相同的 rank，按 rank 升序。组件声明了 `shard_group`（大小 s）时，候选 rank 里位置模 s 与本 rank 相同的 rank 是同组。`group_size` 是同组的数目，`group_index` 是本 rank 在其中的位置。推算假设本副本的每个 rank 都声明了这个组件；实际的组由 01 在第 1 点按内容身份建立，`bind_groups()` 核对两者（第 2 节）。组只有一个 rank 时规划不报错，01 在第 1 点报错。
- **冷启动路径**：有 `allgather` Unit 的组件在 `CatalogEntry.shapes_final` 为真时选 `thin`，否则选 `full_then_trim`，写进规划说明。`sharded_tensor_names()` 返回全部 `allgather` Unit 的规范名，02 在收尾第 7 步把它们换成占位。

#### 3.8 占用估算

```text
rank-local Unit 的加速卡布局：张量按（类型名，规范名）排列，每个张量的起点按它的元素大小对齐
  device_bytes = 末尾张量的终点向上取整到 256 字节
allgather Unit：layout = flatten_shard_unit(张量, group_size)
  每个类型组 g 在块环里占 N × S_g，组的起点按 256 字节对齐
  device_bytes = 各组之和向上取整到 256 字节
  分片槽位里占 Σ_g 向上取整到 256 字节(S_g)
块环 k 的 buffer_bytes = 这一类里最大的两个 device_bytes 之和；只有一个 Unit 时等于它的 device_bytes
分片槽位 shard_slot_bytes = 全部 allgather Unit 在分片槽位里的占用的最大值；共两个
```

`DeviceDemand` 记录准入式（4.1）里除可用显存以外的各项：

| 字段 | 含义 |
|---|---|
| `resident_bytes` | 常驻加速卡的目录张量之和；HSDP 管理的组件按 `fs` 的大小均分后计入。没有登记来源的非持久 buffer 的大小在 `prepare_materialization()` 里补上。 |
| `overlay_bytes` | `catalog.overlay_slots` 的字节数之和。 |
| `ring_bytes`、`shard_slot_bytes` | 两个块环的 `buffer_bytes` 之和；两倍的 `shard_slot_bytes`。 |
| `write_transient_bytes` | `catalog.write_transient.adapter_delta_bytes + catalog.write_transient.quant_layer_bytes`。两者在写入期间可能同时存在，所以相加。 |
| `service_use_bytes` | 按 3.6 的让出关系取两者中较大的：最大的一个 `dit` 组件的使用触发 Unit 的 `device_bytes` 之和；全部被选中的编码器组件的使用触发 Unit 之和，加上最大的一个被选中 `vae` 的使用触发 Unit 之和。 |

#### 3.9 `fully_resident`、登记顺序与规划说明

- 组件的 `fully_resident` 为真，当且仅当它不由 HSDP 管理、规划之后没有任何 Unit。只装触发钩子的组件仍然可以是 `fully_resident`。
- `registration_order`：`BLOCK` 和 `GROUP` Unit 按 `order`，然后 `RESIDENT_LAYERS`，然后 `MODULE` 和 `REMAINDER` Unit，各类内部按 `unit_id` 排序（总体设计 7.4）。
- 规划说明写进启动日志：每个组件的 Unit 划分、按常驻处理的 Unit 和原因、`resident_layers` 全部常驻、冷启动路径、自动发现的块和 `repeated_blocks` 覆盖。格式和日志通道归 05。

### 4. 启动事务

#### 4.1 `prepare_materialization(model, plan, storage)`

在第一次写模型之前按顺序执行。任何一步失败时，本步骤已经分配的资源由它自己释放，`storage` 仍归调用方（01 第 2 节），报 `PrepareError` 或它的子类 `DeviceCapacityError`。

```text
1  核对：plan.host_request() 的每个 Unit 都能在 storage 上找到；plan 已经 bind_groups()
2  登记计划：pin_cpu_memory 为真时查询 host_registration_support()，按 registration_order 和登记上限
             为每个 Host Unit 选“登记候选”或“暂存区”（5.4）；pin_cpu_memory 为假时全部走暂存区，不查询平台
3  加速卡容量准入（下式）；不通过时报 DeviceCapacityError
4  在 ctx.memory_pools.slots() 里分配两个块环缓冲区和两个分片槽位
5  在 ctx.memory_pools.weights() 里分配叠加层槽位，按 OverlaySlotDecl 的形状和类型，填零
6  创建拷贝流、拼接流和每个块环位置、每个分片槽位的事件；计划里有走暂存区的 Unit 时分配两个 256 MiB 暂存区
   （pin_cpu_memory 为真时锁页）
7  把没有登记来源的非持久 buffer（不由 HSDP 管理的组件里）在 ctx.memory_pools.weights() 里搬到执行设备
8  为每个块环 Unit 在缓冲区的两个位置各建一组张量视图（5.2），为每种类型建零元素占位
9  建启动事务：每个资源按 ResourceKind 登记进 ResourceLedger，storage 登记为 HOST_STORAGE
```

加速卡容量准入（总体设计 7.5）：

```text
resident_bytes + 非持久 buffer + overlay_bytes + ring_bytes + 2 × shard_slot_bytes
  + max(write_transient_bytes, service_use_bytes)
  ≤ 可用显存

可用显存 = 第 3 步开始时调用 empty_cache() 之后 current_omni_platform.get_free_memory(ctx.device) 的读数
```

- `DeviceCapacityError` 写明每一项的字节数和可用显存，`fix` 给出 recipe 里的 offload 策略（`--diffusion-offload-policy` 或预设）。
- 准入只在本 rank 内做，不经过集合通信；各 rank 的结果在 01 的第 2 点汇总，任何一个 rank 失败时整组一起失败。
- 第 4、5 步分配的地址到 `close()` 前不变（总体设计 7.5 的固定槽位）。

#### 4.2 `LoadTransaction.target(tensor_key)`

| `plan.placement(tensor_key).residency` | 返回 |
|---|---|
| `host` | `storage.view(tensor_key)`：本机后端的写入目标或只读视图，等待者的共享张量和瘦路径范围外的张量是 meta 张量（03）。 |
| `device` | `None`：Loader 在执行设备上、在 `weights()` 池里分配。 |
| `hsdp` | `None`：Loader 在 CPU 上分配，由 HSDP 分片（02 第 13 节）。 |

- `tensor_key` 是预期清单里的规范名。不在规划里的名字报 `LifecycleError`。
- 返回的张量的形状、类型、步长由 03 的布局保证与预期清单一致，`target()` 不再复制核对；Loader 绑定时核对（02 第 7.4 节）。
- `target()` 在提交之后或 `abort()` 之后调用报 `LifecycleError`。

#### 4.3 `commit(final_manifest)`

```text
1  校验最终清单：每个条目的形状、类型、步长与规划的张量一致；residency 与 placement() 一致；
   host 条目的 storage_key 等于规范名；每个 Host Unit 的张量都在清单里
2  内存登记：对登记候选取 storage.ranges(unit)，按 Mapping 合并后登记、确认锁页；
   失败的区间里的 Unit 改走暂存区并记警告；需要而还没分配暂存区时在这时分配（5.4）
3  建运行期的 Unit 记录：每个张量从 final_manifest.live(name) 取模型张量对象，记下它此刻绑定的 Host 视图作为 H2D 源；
   然后把全部 Host 驻留的张量（含别名对象）改指向执行设备上的占位
4  叠加层槽位：对每个 OverlaySlotDecl 调用 catalog.bind_overlay_slot(name, slot)，再从账本 hand_over(slot)
5  对目录里的每个组件调用一次 catalog.set_fully_resident(component, value)
6  在 compile.py 的驻留表里登记每个组件的模块和 fully_resident（6.3）；给模型里的每个 ExactProjectionCache 绑定写入代数（6.5）
7  经 HookRegistry 装钩子（5.1），对每个组件装提交后的守卫（5.5）
8  ledger.transfer() 移交给新建的 OffloadRuntime，启动事务变为空
```

- 第 1 步不一致时报 `CommitError`，写明张量名和不一致的字段。
- 第 2 步的暂存区分配失败或撤销登记失败时报 `CommitError`。
- 任何一步失败时，已经建立的登记、钩子、守卫都已登记在账本里，资源全部留在启动事务里，由 `abort()` 释放（01 第 2 节）。第 4 步之后失败时，已经 `hand_over` 的槽位只被模型的包装层引用，随模型丢弃（01 第 5.1 节）。
- `commit()` 只做本地操作，不发起集合通信；各 rank 在 01 的第 3 点一起激活。
- 本模块内的故障注入点：`commit_register` 在第 2 步之前，`commit_hooks` 在第 7 步之前，命名和 `FaultPlan` 与 01 第 10 节相同。

#### 4.4 `abort()` 与所有权

- `abort()` 按 01 第 5.2 节的 `ResourceLedger.close()` 顺序释放启动事务持有的全部资源，包括关闭 host 存储；可以多次调用，提交之后什么都不做。
- 启动事务没提交就被丢弃时，`__del__` 不释放资源；释放只经 01 驱动的 `release()` 调用 `abort()`。
- 同一个句柄只登记一次：块环缓冲区、分片槽位、每个合并区间的登记、暂存区、流和事件各是一个句柄；钩子按（模块，钩子名）登记一个句柄；守卫按组件登记一个句柄。

### 5. 运行期

#### 5.1 钩子

钩子经 `vllm_omni/diffusion/hooks/base.py` 的 `HookRegistry` 安装：`HookRegistry.get_or_create(module)` 第一次调用时把 `module.forward` 换成分发器 `_WrappedForward`，原来的 `forward` 存在 `module._omni_original_forward`；`register_hook(name, hook)` 登记钩子，`remove_hook(name)` 撤掉。不用 `register_forward_hook`。

```python
# vllm_omni/diffusion/weight_stack/offload/hooks.py
HOOK_NAMES: frozenset[str] = frozenset({"_weight_stack_ring", "_weight_stack_use", "_weight_stack_trigger"})
```

| 钩子 | 装在哪里 | `pre_forward` | `post_forward` |
|---|---|---|---|
| `_weight_stack_ring` | 块环 Unit 的根模块，以及 Unit 里每个直接持有张量的模块。 | 记一次组件的使用；没有活动的流式 Unit 时从这个模块进入 Unit（5.2）；有活动 Unit 且是同一个时只核对驻留；是另一个时报 `RuntimeFault`。 | 进入点的模块返回时退出 Unit，在计算流上记录这个位置的完成事件。 |
| `_weight_stack_use` | 使用触发 Unit 里每个直接持有张量的模块。 | 记一次组件的使用；Unit 不在加速卡上时装入（5.3）；Unit 的使用深度加一。 | 使用深度减一。 |
| `_weight_stack_trigger` | `triggers` 里每个组件的每个根模块。 | 记一次组件的使用。 | 什么都不做。 |

- “直接持有张量的模块”是 `TensorDecl.module` 给出的规范路径解析出的模块；动态 LoRA 和 `overlay` 固定适配器的包装层解析成包装层本身，钩子装在包装层上。
- “记一次组件的使用”：组件与上一次使用的组件相同时什么都不做；否则按 `yielders` 让出这个组件的使用会让出的全部 Unit（5.3），再把它记为上一次使用的组件。
- 钩子只实现 `pre_forward` 和 `post_forward`，不覆写 `new_forward`，所以能与 TeaCache、MagCache、SeaCache 这类覆写 `new_forward` 的缓存钩子共存。`HookRegistry` 按钩子名排序执行 `pre_forward`，按逆序执行 `post_forward`；三个名字以 `_` 开头，排在小写字母开头的现有钩子名之前，所以 Offloader 的 `pre_forward` 最先执行、`post_forward` 最后执行。
- 两个方法都用 `torch.compiler.disable` 修饰：编译包住的区域里调用到被管理的模块时，钩子在编译图之外运行（总体设计 7.8）。
- 钩子在 `torch.cuda.is_current_stream_capturing()` 为真时，凡是需要装入、让出或拼接，报 `RuntimeFault`，写明组件名；不需要任何动作的调用（Unit 已驻留、使用者没变）放行。
- 钩子在 `execution()` 的作用域之外被调用时报 `RuntimeFault`，写明组件名和模块路径（5.6）。

#### 5.2 块环

两类流式 Unit 各用一块缓冲区和一张后继表，互不读写：

```text
位置：HEAD = [0, device_bytes(u))，TAIL = [buffer_bytes − device_bytes(u), buffer_bytes)
      任意两个 Unit 一个在 HEAD、一个在 TAIL 时不重叠，因为 buffer_bytes 是最大的两个占用之和
状态：slot[HEAD]、slot[TAIL] 各记一个 Unit 或空；每个位置有完成事件 done[p] 和就绪事件 ready[p]
后继表：succ[u] 初始为 sequence 里的下一个；rank-local 最后一个接回第一个，allgather 最后一个没有后继

进入 Unit u（从它的某个钩子）：
  u 在某个位置 p 上：计算流等待 ready[p]
  u 不在缓冲区里（预测错）：
    p = 上一个进入的 Unit 所在位置的另一侧；没有上一个时 p = HEAD
    同步装入 u 到 p（下文的“装入”），计算流等待 ready[p]
    把同一张表里上一个进入的 Unit 的后继改成 u
  u 的张量改指向位置 p 上的视图；u 记为活动 Unit
  w = succ[u]；w 存在且不在缓冲区里时，异步装入 w 到 p 的另一侧
退出 Unit u：在计算流上记录 done[p]；活动 Unit 清空

装入 Unit x 到位置 q：
  q 上原来的 Unit 的张量改指向占位
  拷贝流（allgather 时是拼接流）等待 done[q]
  rank-local：按 5.4 从 Host 拷到位置 q；allgather：H2D 到分片槽位，再拼接到位置 q
  在这条流上记录 ready[q]；slot[q] = x
```

- 每个块环 Unit 在两个位置上的视图在 `prepare_materialization()` 里、在任何梯度模式之外建好，运行期只做 `tensor.data = view` 的换绑，与现有的 `set_tensor_storage` 是同一种写法。
- 一个流式 Unit 还没退出又从另一个流式 Unit 的钩子进入时报 `RuntimeFault`，写明两个 Unit。这条规则对两类块环都适用。
- `allgather` 的块环不跨前向保留：每次 `execution()` 退出时，`allgather` 块环的两个位置清空，张量改指向占位。每次前向的第一个 `allgather` Unit 因此在进入时拼接，之后按后继表预取。
- `allgather` 的后继表只按 `allgather` Unit 的进入顺序更新，`rank-local` 的进入不改它。同组各 rank 的进入顺序相同，所以每个 rank 发起拼接的次数和顺序相同；只在部分 rank 上声明的 `rank-local` 流式 Unit 不影响它。
- 在拼接流上发起的集合通信按发起顺序完成，所以预测错时覆盖一个刚被预取的位置不需要额外等待。

#### 5.3 使用触发与让出

- 装入：在当前计算流上按 Unit 的 `device_bytes` 向缓存分配器申请一块存储，按 3.8 的布局建视图，从 Host 拷入（5.4），张量改指向视图，记为驻留。地址不做承诺。
- 装入时显存不足，先把这个 Unit 已经分配的存储释放、张量改回占位，再按 5.6 的异常规则处理。
- 让出：Unit 的张量改指向占位，丢掉对存储的引用，然后调用 `empty_cache()`。让出不往 Host 回写（总体设计不变量 9）。
- 一次“记使用”里让出的全部 Unit 释放完之后只调用一次 `empty_cache()`。
- 要让出的 Unit 的使用深度大于 0 时（它自己的某个模块正在执行），报 `RuntimeFault`，写明两个组件。
- `vae` 没被选中时常驻加速卡，不装钩子；被选中时由使用触发管理。
- 模型绕过钩子直接读一个已让出的参数时读到零元素的占位，形状不符，直接报错。

#### 5.4 Transport 的实现

**rank-local**。H2D 源是提交时记下的 Host 视图。拷贝时把在 Host 上相邻、在加速卡布局上也相邻的连续张量合成一次拷贝；本机后端同一个 Unit 的张量在同一个文件里相邻（03 第 4.2 节），所以通常每种类型一次拷贝。

- 登记过的 Unit：在拷贝流（使用触发时在计算流）上 `copy_(non_blocking=True)`。
- 走暂存区的 Unit：两个暂存区交替使用，每块按 256 MiB 分段。每段先在 CPU 上拷进暂存区，再异步 H2D，记录这个暂存区的事件；再次使用一个暂存区之前在 CPU 上等它的事件。`pin_cpu_memory` 为假时暂存区不锁页，H2D 退化为同步拷贝。

**allgather**。每个 `allgather` Unit 的 H2D 源是 `storage.ranges(unit)` 给出的本 rank 范围，按地址排序后与 `flatten_shard_unit` 的类型组一一对应，个数不符时在提交时报 `CommitError`。源张量由 `MemoryRange` 的地址和长度经 `torch.frombuffer` 建成 uint8 视图，不拷贝。

```text
装入 allgather Unit x（拼接流上）：
  选两个分片槽位中的另一个 k；拼接流等待槽位 k 上一次拼接的完成事件
  拷贝流：本 rank 每个类型组的 S_g 字节 H2D 到槽位 k（登记或暂存区，规则同 rank-local）；记录事件 h
  拼接流：等待 h；
          all_gather_into_tensor(ids, 本 rank 的 x.order)，然后 torch._assert_async(ids == x.order)
          每个类型组一次 all_gather_into_tensor(块环位置上这一组的 N × S_g 区域, 槽位 k 上这一组的 S_g 区域)
  记录 ready[q] 和槽位 k 的完成事件
```

- 拼接用 `bind_groups()` 记下的这个组件的 `AllGatherGroup.group`。各类型组都按 uint8 视图拼接。
- 每次拼接先拼 Unit 编号并在加速卡上断言，编号不一致时设备侧断言失败，这次前向以异常结束，运行时按 5.6 标记为不可用。
- 张量视图建在每个类型组拼接结果的前 E_g 个元素上，补零部分不被任何张量引用。

**内存登记**。

```text
commit() 第 2 步：
  对 registration_order 里每个登记候选 u：rs = storage.ranges(u)
  按 Mapping.mapping_id 分组；每个区间向外对齐到页（os.sysconf("SC_PAGE_SIZE")），不越出 Mapping 的范围；
  组内合并成互不重叠的区间
  按 registration_order 依次把 u 的区间并入，合并后的字节数超过上限时 u 改走暂存区（上限为 0 时不限）
  逐个合并区间调用平台的 register_host_range(address, length, read_only=Mapping 的 read_only)
  登记后用 host_range_is_pinned 确认首页和末页被认作锁页；确认不了当作登记失败，撤销这个区间的登记
  一个合并区间失败时，落在它里面的 Unit 全部改走暂存区，记一次警告
```

- `read_only` 为真（共享存储）时只读登记，需要平台支持只读登记；为假时普通登记，需要平台支持可写登记。平台不支持相应的登记时，这些 Unit 走暂存区，不尝试登记。
- 上限是每个 worker 的 `host_registration_limit_gib`（05），统计对齐、合并之后的字节数，覆盖全部 Unit 和分片。`prepare_materialization()` 第 2 步按每个 Unit 的字节数加上它涉及的每个映射两页的上界选候选，这个上界不小于提交时的实际值，所以提交时一般不再剔除。
- 一个 Unit 要么整个登记，要么整个走暂存区。撤销登记按合并区间进行。登记结果写进启动日志。
- 任何策略都不生成完整的锁页副本：Host 上只有 HWM 分配的存储和两个固定大小的暂存区。

#### 5.5 提交后的守卫

`commit()` 对每个不由 HSDP 管理的组件的每个模块（根子树去掉划出的子树）把实例属性 `_apply` 换成守卫。`nn.Module` 的 `.to()`、`.cuda()`、`.cpu()`、`.float()`、`.half()`、`.bfloat16()`、`.type()`、`to_empty()` 都经过 `_apply`，递归时对子模块调用的也是子模块实例上的 `_apply`。

```text
守卫(fn)：对这个模块直接持有的张量的每种（类型，维数），在执行设备上建一个零元素探针 t
  fn(t) 的设备、类型、步长都与 t 相同：什么都不做，返回模块
  否则：报 RuntimeFault，写明组件名、模块路径和变化（设备、类型或内存格式）
```

- 被管理的参数在服务期间始终是执行设备上的张量（驻留时是视图，否则是占位），用 `next(params).device` 判断设备的模型代码看到的是执行设备。
- 守卫对全部组件生效，包括 `fully_resident` 的组件：提交后基础权重的字节、布局、设备都不可变（总体设计不变量 3）。
- `close()` 撤掉守卫，恢复类上的 `_apply`。

#### 5.6 单次执行、梯度模式与异常

```python
class UnifiedOffloadRuntime:   # 实现 01 的 OffloadRuntime
    def grad_mode(self) -> AbstractContextManager[None]: ...   # 有块环 Unit 时 torch.no_grad()，否则 torch.inference_mode()
    def execution(self) -> AbstractContextManager[None]: ...
    def before_sleep(self) -> None: ...
    def after_wake(self) -> None: ...
    def stats(self) -> OffloadStats: ...                       # 已登记字节、暂存区字节、加速卡槽位字节，供 05 的指标
    def close(self) -> None: ...
```

```text
进入 execution()：运行时不可用、处于睡眠、已经关闭，或已有一次执行在进行（任何线程）时报 RuntimeFault
正常退出：清空 allgather 块环；活动 Unit 应为空，不为空时报 RuntimeFault
异常退出：同步拷贝流和拼接流；清空活动 Unit 和 allgather 块环；
          这次执行里发起过 allgather 拼接时把运行时标记为不可用；然后重新抛出原异常
```

- 并发和重入按非阻塞地获取一把锁判断，获取不到就报错，不等待。
- 运行时被标记为不可用之后，每次 `execution()` 都报 `RuntimeFault`，写明第一次失败的原因。集合通信的顺序已不可信，这个副本的 worker 由执行器终止（总体设计不变量 12）。
- runner 不读 offload 策略：`unified` 下前向的梯度模式取 `grad_mode()`，HSDP 下照旧用 `no_grad`（C17，第 8 节）。

#### 5.7 关闭

`close()` 是本地的、终止性的，可以多次调用，按 01 第 5.2 节的 `ResourceKind` 顺序进行：

```text
EXECUTION     不再接受新的执行
INFLIGHT      同步拷贝流和拼接流
HOOKS         从每个模块的 HookRegistry 撤掉本 RFC 的钩子，撤掉守卫，丢掉对模型的引用
DEVICE        释放块环缓冲区、分片槽位、事件、流
REGISTRATION  按合并区间撤销登记
STAGING       释放暂存区
PLACEHOLDER   每个 Host 驻留的张量和块环 Unit 的张量改指向占位
HOST_STORAGE  storage.close()
```

- 关闭不把权重恢复回 CPU，不发起集合通信（总体设计不变量 11）。
- 运行时因为拼接失败被标记为不可用时，`INFLIGHT` 不等待拼接流，`DEVICE` 不释放块环缓冲区和分片槽位，它们留到进程退出：拼接流上可能还有永远不会完成的集合通信在写它们。
- 一个关闭函数抛出异常时其余照常执行，由账本汇总报 `LifecycleError`（01 第 5.2 节）。

#### 5.8 睡眠模式

| 加速卡上的显存 | 分配位置 | 睡眠时 | 唤醒时 |
|---|---|---|---|
| 常驻的权重和 buffer（Loader 分配）、本 RFC 搬到执行设备的非持久 buffer、叠加层槽位 | `memory_pools.weights()` | 由睡眠模式拷到 Host。 | 由睡眠模式拷回原地址。 |
| 块环缓冲区、分片槽位 | `memory_pools.slots()`（标签 `weight_slots`） | 直接释放，不备份。 | 重新映射到原地址，内容在下一次使用时装入。 |
| 使用触发 Unit | 缓存分配器 | `before_sleep()` 让出。 | 下一次使用时装入。 |
| host 存储、暂存区、内存登记 | Host | 不受影响。 | 不受影响。 |

```text
before_sleep()：进入 execution() 的锁（有执行在进行时报 RuntimeFault）；同步拷贝流和拼接流；
               让出全部使用触发 Unit；清空两个块环，张量改指向占位；记为睡眠；调用 empty_cache()；释放锁
after_wake()：清除睡眠标记；块环保持为空
```

- 01 第 8.4 节规定 worker 的调用顺序：`before_sleep()` 在 `allocator.sleep(...)` 之前，`after_wake()` 在唤醒 `weights` 和 `weight_slots` 之后。
- 唤醒后常驻部分和固定槽位的地址不变，编译结果和只读取 `fully_resident` 组件的 CUDA graph 继续有效。
- 睡眠期间的 `execution()` 报 `RuntimeFault`。
