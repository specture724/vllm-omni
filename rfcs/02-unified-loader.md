# [RFC]: 统一加载

> 上级 RFC：[umbrella RFC](./00-umbrella.md)。设计依据：[总体设计](../LOADER_OFFLOADER_DESIGN.md) 第 5 节（5.1～5.7）、8.7、9.1、9.2、9.3。
>
> 范围：`vllm_omni/diffusion/weight_stack/loader/`、`vllm_omni/diffusion/weight_stack/components.py`、`vllm_omni/diffusion/model_loader/construct.py`、`vllm_omni/diffusion/model_loader/hub_prefetch.py`，19 个构造期读张量的家族目录，以及共存期必改清单里由本 RFC 承担的各行。
>
> 依赖：[子 RFC 01](./01-startup-transaction.md)（总开关的进程级状态、`weight_stack/interfaces.py`、`weight_stack/errors.py`、启动驱动、`StartupContext.memory_pools`、共存期登记表（`tests/diffusion/weight_stack/coexistence/`）、测试替身）；[子 RFC 03](./03-host-storage.md)（`HostStorage.view` 与 `seal` 的语义、瘦路径的地址预留）；[子 RFC 04](./04-unified-offloader.md)（`PlacementPlan` 给出的冷启动路径和每个张量的 Transport、`LoadTransaction.target`、叠加层槽位的分配、`fully_resident` 的填写、编译设置）；[子 RFC 05](./05-configuration.md)（`--diffusion-max-lora-rank` 和 LoRA 相关组合的解析）。

## 摘要

本 RFC 是 umbrella Job 2 的实现设计，覆盖新栈里 Loader 的全部内容：

- 组件声明与组件目录，四个带类型字段的结构，组件查询接口；
- 只建结构的构造上下文、读取陷阱、构造后检查；
- 两个栈在构造期的唯一分发点：`build_component`、`declare_weights`、`hf_component`，以及 `from_pretrained_with_prefetch` 的内部分发；
- 规则表的格式、执行和逐表测试；
- checkpoint 读取器：来源解析、延迟张量、存储绑定、写入入口、写入跟踪、逐层在线量化、适配器融合和预检；
- 固定的收尾顺序，内容身份，预期清单和最终清单；
- 动态 LoRA 和 `overlay` 固定适配器的叠加层包装，动态 LoRA 管理器在 `unified` 下的运行期；
- `load_format="diffusers"`、`load_format="custom_pipeline"`、HSDP 组件的物化；
- 共存期的逐张量对比框架、对比例外表、legacy 一支的指纹清单；
- 19 个构造期读张量家族的迁移，构造函数里直接调用 `X.from_pretrained` 的 pipeline 改用 `hf_component`，以及共存期必改清单里由本 RFC 承担的 C02、C04、C05、C09、C10、C11、C13、C15、C16、C18。

本 RFC 的每一个 PR 只改变 `unified` 下的行为，`legacy` 下的行为和已有测试不变，不修改已有测试文件。

## 动机

umbrella 的“背景与问题”已经给出总体原因，这里只列本 RFC 直接替换的现有代码：

- `DiffusersPipelineLoader.load_model`（`vllm_omni/diffusion/model_loader/diffusers_loader.py`）先按加载设备完整构造 pipeline，再经 `load_weights` 写入，之后依次调用 `_maybe_fuse_distilled_lora`、`_process_weights_after_loading`、`_apply_skip_softmax_calibration`、`model.eval()`。offload 规划（`get_offload_plan`）、Host Weight Runtime 的接入（`_resolve_hwr`、`_publish_hwr_after_load`）和 HSDP 的两条路径（`_load_model_with_hsdp`、`_load_model_with_pre_sharded_hsdp`）都穿插在这个方法里。在线量化配 CPU offload 时，`_stream_online_quant_weights_to_cpu` 在量化完一层后对它调用 `module.to("cpu")`，写入之后还会整体 `model.to("cpu")`。
- 预分片 HSDP 路径的 `_materialize_pre_sharded_hsdp_state` 用 `to_empty` 物化，再保存、恢复非持久 buffer；`_finalize_pre_sharded_hsdp_loaded_weights` 调用模型的 `post_load_weights` 和 `validate_loaded_weights`。
- 蒸馏 LoRA 有两个融合点：`DiffusersPipelineLoader._maybe_fuse_distilled_lora` 和 `DiffusionWorker.init_lora_manager`。两处都由 pipeline 的 `load_lora_weights` 在写入之后对参数直接相加。
- `DiffusionLoRAManager._replace_layers_with_lora`（`vllm_omni/diffusion/lora/manager.py`）在第一次加载适配器时才包装层；`_ensure_max_lora_rank` 在运行期按更大的 rank 调用 `create_lora_weights` 重建槽位；管理器在构造时调用 `resolve_offload_strategy` 选择 LoRA buffer 的设备。
- `create_transformers_model` 和 `create_transformers_model_with_vllm_linears`（`vllm_omni/diffusion/models/utils.py`）在 meta 上用 `from_config` 建结构之后，立即用 `init_parameters`、`init_parameters_preserving_vllm_linear` 在默认设备上分配参数。
- 19 个家族在构造函数里，或在 pipeline 自己的 `load_weights` 里，直接读文件：`safe_open` 加 `get_tensor`、`load_file`、`torch.load`、`from_single_file`、自定义的 `from_pretrained`（第 16 节逐个列出）。
- `_dit_modules` 等发现属性被多处读取，直接读或经 `ModuleDiscovery.discover`（`vllm_omni/diffusion/offloader/module_collector.py`）读：`DiffusionModelRunner.load_model` 选择编译目标，`DiffusionLoRAManager._replace_layers_with_lora` 选择 LoRA 组件，`vllm_omni/diffusion/cache/cachedit/backend.py` 的 `_dit_module_names`，`vllm_omni/diffusion/registry.py` 的 `_apply_sequence_parallel_if_enabled` 和 `_apply_wan_vae_fastpath_if_enabled`。

## 设计

本节所有要求只对 `unified` 成立。`legacy` 下 Loader 的任何代码都不运行，`build_component` 只调用 legacy 一支，HF 组件入口与现有调用完全相同。

### 1. Loader 在启动流程中的位置

`UnifiedLoader`（`weight_stack/loader/loader.py`）实现 [子 RFC 01](./01-startup-transaction.md) 在 `weight_stack/interfaces.py` 里定义的 `Loader` 协议。`weight_stack/loader/__init__.py` 导出工厂函数 `create_loader(ctx: StartupContext) -> Loader`，由 01 的 `EntryPointFactory` 按名字取用。启动驱动按下表调用：

```text
启动步骤（总体设计 4.2）   调用                                              结束时的状态
2   只建结构               loader.build_structure()，返回 (model, catalog)    STRUCTURE_READY
4   Offloader.plan()       plan 读 catalog，只经 set_discovered_blocks 写   PLANNED
5   Loader 预检            loader.preflight(catalog, plan)，返回 Preflight    SOURCES_PREPARED
9   Loader 写入            loader.write(model, storage, transaction)         WEIGHTS_READY
10  Loader 收尾            loader.finalize(model, storage)，返回 FinalManifest FINALIZED
失败或关闭                 loader.close()，可以多次调用
```

```python
# weight_stack/loader/__init__.py 导出、01 的 interfaces.py 在 TYPE_CHECKING 里引用的类型
@dataclass(frozen=True)
class Preflight:
    content_identities: Mapping[str, str]                 # 组件名到内容身份摘要，即 RankFingerprints.content_identities
    identities: Mapping[str, ContentIdentity]             # 组件名到内容身份本身（10.1），用于日志和测试
    expected_manifest: ExpectedManifest                   # 10.2
    sources: Mapping[str, tuple[ResolvedSource, ...]]     # 组件名到解析后的来源（7.1）
    adapters: tuple[ResolvedAdapter, ...]                 # 融合适配器和固定适配器（第 8 节）
```

- `write()` 里每个张量的写入目标取 `transaction.target(tensor_key)`，`tensor_key` 是预期清单里的规范名。返回张量时绑定到它；返回 `None` 时这个张量常驻加速卡，Loader 在执行设备上、在 `ctx.memory_pools.weights()` 里分配，因此进入睡眠时备份的池。收尾里新建或替换的常驻加速卡张量（第 2 步的替换、第 1 步 HSDP 分片后的本地分片，第 9、13 节）同样由 Loader 在这个池里分配。只建结构期间的分配和写入期间的临时张量（量化前的参数、适配器的 Δ、`on_execution_device` 变换的中间结果）不进任何睡眠池。驱动自己不进入睡眠池（01 的 6.2）。
- `finalize()` 执行总体设计 5.4 的第 1 到第 7 步（第 9 节），第 6 步调用 `storage.seal(final_manifest)`。
- `close()` 关闭构造上下文、读取陷阱、checkpoint 读取器的文件句柄和临时张量。组件目录在 `close()` 之后仍然可以查询。
- Loader 从 HWM 和 Offloader 一侧只 import `weight_stack.interfaces`，不 import 它们的实现；在 `weight_stack` 里另外只 import `errors`、`components` 和 `startup`（只用 `WeightStack` 和 `process_weight_stack`）。复用 legacy 的来源解析时在函数内 import `model_loader.diffusers_loader`（7.1）。
- 每一步失败时抛出 01 在 `weight_stack/errors.py` 里定义的对应类型：`ConstructionError`、`PreflightError`、`WriteError`、`FinalizeError`。上下文关闭后调用组件构造函数报 `LifecycleError`。

### 2. 组件声明与组件目录

#### 2.1 模型侧的声明

模型侧的类型和函数放在 `weight_stack/components.py`。`models/`、`lora/`、`cache/` 可以 import 它；它不 import Loader、HWM、Offloader 的实现。

```python
Role = Literal["dit", "text_encoder", "image_encoder", "audio_encoder", "vae", "other"]

@dataclass(frozen=True)
class ShardGroupSpec:
    name: str            # 进程组的稳定名字，例如 "minimax_h3_text_encoder"
    size: int
    rank: int            # 本 rank 在这个组里的编号

@dataclass(frozen=True)
class AdapterLocation:
    location: str        # 本地路径或 Hub 仓库
    filename: str | None
    revision: str | None
    format: str          # 格式转换器的注册名（8.1）

@dataclass(frozen=True)
class FixedAdapter:
    name: str                          # 在组件内唯一
    source: AdapterLocation
    targets: tuple[str, ...]           # 相对组件根的模块路径，可以含通配符
    apply: Literal["fuse", "overlay"]

@dataclass(frozen=True)
class ComponentDecl:
    name: str                                       # 稳定的组件名
    role: Role
    path: str | tuple[str, ...]                     # 一个或几个根路径
    repeated_blocks: tuple[str, ...] | None = None
    unit_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    shard_group: ShardGroupSpec | None = None
    fixed_adapters: tuple[FixedAdapter, ...] = ()

def declare_components(pipeline: nn.Module, decls: Sequence[ComponentDecl]) -> None: ...
```

- pipeline 在构造函数里、配置解析之后调用一次 `declare_components`，传入本实例、本 rank 的全部声明。同一个实例调用第二次报错。`legacy` 下这个调用只把声明存在实例上，没有使用方读取它。
- 结构构造方法、权重来源和规则表不写在 `ComponentDecl` 里：结构就是构造上下文里运行的构造代码，来源和规则表由 `declare_weights`、HF 组件入口和 `weights_sources` 在构造期间登记（第 5 节），构造结束时 Loader 按模块路径把它们归到所属的组件。
- 动态 LoRA 的目标组件沿用 pipeline 现有的 `_lora_components` 属性（例如 `vllm_omni/diffusion/models/sensenova_u1/pipeline_sensenova_u1.py` 的 `_lora_components = ["language_model"]`），不进 `ComponentDecl`。

#### 2.2 默认推导

pipeline 没有调用 `declare_components` 时，Loader 在构造结束后从现有的发现属性推导声明，属性的读取方式与 `ModuleDiscovery.discover` 相同：

| 来源 | 推导出的声明 |
|---|---|
| `_dit_modules` 的每一项 | `role="dit"`，名字和路径都取这一项。 |
| `_encoder_modules` 的每一项 | 最后一段是 `image_encoder`、`vision_encoder`、`vision_model`、`gen_image_condition_refiner`、`conditioning` 时 `role="image_encoder"`；是 `vae` 时 `role="vae"`；其余 `role="text_encoder"`（总体设计 8.4）。 |
| `_vae_modules` 的每一项 | `role="vae"`。 |
| `_resident_modules` 的每一项 | `role="other"`。 |
| pipeline 不实现 `SupportsComponentDiscovery` | 按 `ModuleDiscovery._FALLBACK_DIT_ATTRS`、`_FALLBACK_ENCODER_ATTRS`、`_FALLBACK_VAE_ATTRS` 推导，角色规则同上。 |

推导出的声明不带任何带类型字段。显式声明完全取代推导，两者不合并。推导结果通不过完整性检查的 pipeline 必须显式声明，CI 的支持矩阵测试（5.7）列出它们。推导出的角色与总体设计不同的 pipeline 同样显式声明：Cosmos3 的 `_dit_modules` 是 `["transformer.language_model", "transformer"]`，显式声明把 `transformer.language_model` 记为 `text_encoder`（02-15）。

#### 2.3 路径解析、路径列表与嵌套划出

- 路径按属性路径解析，规则与 `ModuleDiscovery._collect_modules` 用的 `operator.attrgetter` 相同，可以经过不是 `nn.Module` 的对象。例如 Gr00t 的模型挂在普通对象 `Gr00tPolicy` 上（`vllm_omni/diffusion/models/gr00t/policy.py`），路径写 `policy.model`。这类组件的根不在 `pipeline.named_modules()` 里，完整性检查和对比测试把声明的根并入检查范围。
- 路径解析到 `None`，或解析到的子树里没有任何参数、持久 buffer 和登记过来源的数据张量时，这个组件视为不存在，不进目录。
- `path` 是列表时，组件由这几棵子树组成。例如 MiniMax H3 视频 VAE 的编码半和解码半各是一个 `vae` 组件，各自的 `path` 列出所属的几个子模块。
- 一个组件的根落在另一个组件的子树里时，内层从外层划出：外层组件的张量是外层子树去掉所有内层子树之后剩下的部分。内层的根必须恰好落在一个外层组件里。例如 Cosmos3 的 `text_encoder` 声明为 `transformer.language_model`，`dit` 声明为 `transformer`。
- 同一个模块挂在两个属性上时按模块身份去重，第一次出现的路径是它的规范路径。张量名、内容身份和清单都使用规范路径。
- 同一个张量对象出现在几个路径下（构造时就共用的参数，例如 `self.lm_head.weight = self.embed.weight`）按张量身份去重，记为一个张量和若干别名。

#### 2.4 完整性检查

构造结束、最终类型落到 meta 参数之后，Loader 遍历 pipeline 下的每个参数、每个持久 buffer 和每个登记过来源的非持久 buffer（加上 2.3 第一条并入的根），算出没有归属的路径和归属多于一个的路径，记在 `ComponentCatalog.ownership_violations` 里。总体设计 5.1、5.2 和 8.5 规定这类错误在规划阶段报出，所以 Loader 在构造阶段只记录、不报错：04 的 `plan()` 第一步调用 `catalog.require_complete()`，`ownership_violations` 或 6.2 的 `dtype_violations` 非空时报 `PlanningError`，列出全部路径。只想常驻、不参与 offload 的部分显式声明为 `other`。

#### 2.5 按 rank 声明

声明按 rank 生成，Loader 不要求各 rank 相同。没有张量的组件视为不存在，例如 MAGI-2 在非输出 rank 上把文本编码器构造成 `nn.Identity()`（`Magi2Pipeline.__init__`），这个 rank 的目录里就没有 `text_encoder`。哪些组件只在部分 rank 上存在、是否参与全组核对，由 01 在总体设计 4.5 第 1 点收集各 rank 的指纹后决定。

#### 2.6 带类型字段

字段集合是封闭的，加字段要经过设计评审。每个字段在构造结束时按下表校验，校验失败在构造阶段报 `ConstructionError`：

| 字段 | 类型 | 校验 | 使用方 |
|---|---|---|---|
| `repeated_blocks` | 相对组件根的路径元组 | 每个路径解析到一个 `nn.ModuleList` 或 `nn.Sequential`，元素的类相同；路径之间互不嵌套。元组顺序就是常驻层的计数顺序。 | Offloader 规划（04），编译区域选择（第 15 节）。 |
| `unit_groups` | 组名到路径模式的映射 | 组名是标识符；模式里的 `*` 匹配恰好一段路径。模式在构造时只做语法校验，选中的组是否匹配到模块由规划校验（04）。 | 策略的 `units`（04、05）。 |
| `shard_group` | `ShardGroupSpec` | `size >= 1`，`0 <= rank < size`；同一个名字在一个 rank 的所有声明里大小和编号相同。 | 内容身份（10.1），AllGather 分组（01、03）。 |
| `fixed_adapters` | `FixedAdapter` 元组 | 名字在组件内唯一；`targets` 每一项至少匹配一个支持的层（8.5）；`format` 已注册。 | Loader（第 8 节）。 |

每个字段的值进入规划指纹（03 计算），由各使用方的期望表测试覆盖。

#### 2.7 组件目录

目录的数据类型同样放在 `weight_stack/components.py`，构建逻辑放在 `weight_stack/loader/catalog.py`。Offloader 的 `plan()` 只读这个对象，不 import Loader。

```python
@dataclass(frozen=True)
class TensorDecl:
    name: str                     # 包装前的规范名，例如 "transformer.blocks.0.attn.to_q.weight"
    aliases: tuple[str, ...]      # 共用同一个张量的其他名字
    module: str                   # 所属模块的规范路径（包装前）
    kind: Literal["parameter", "buffer", "data", "quant_output", "fixed_overlay"]
    shape: tuple[int, ...]
    dtype: torch.dtype
    stride: tuple[int, ...]
    required: bool                # 必须由某个来源写入
    source: str | None            # 来源绑定的标识；weights_sources 的流记为 "weights_sources:<prefix>"

@dataclass
class CatalogEntry:
    decl: ComponentDecl
    roots: tuple[str, ...]               # 解析后的规范根路径
    carved: tuple[str, ...]              # 从本组件划出的内层组件的根
    tensors: tuple[TensorDecl, ...]
    rule_tables: tuple[str, ...]         # 用到的规则表：限定名加版本
    hsdp_managed: bool
    shapes_final: bool                   # 写入之前就能确定每个张量的最终形状和类型（总体设计 6.3、6.4）
    finalize_replaced: tuple[str, ...]   # 收尾第 2 步会整个替换的张量（第 9 节）
    fully_resident: bool | None          # commit() 之前为 None

@dataclass(frozen=True)
class TransientNeed:
    adapter_delta_bytes: int      # 最大的单个适配器 Δ（8.2）
    quant_layer_bytes: int        # 最大的一层量化前的数据（7.6）

@dataclass(frozen=True)
class OverlaySlotDecl:
    name: str                     # "<包装层的规范路径>.lora_a.<i>" 或 ".lora_b.<i>"
    layer: str                    # 包装层的规范路径（包装前的名字）
    shape: tuple[int, ...]
    dtype: torch.dtype

class ComponentCatalog:
    entries: Mapping[str, CatalogEntry]          # 按组件名排序
    declaration_order: tuple[str, ...]           # 组件名按声明顺序：declare_components 的顺序；默认推导时按推导顺序（2.2）
    overlay_slots: tuple[OverlaySlotDecl, ...]   # 动态 LoRA 槽位（11.4），不属于任何组件
    wrapped_layers: tuple[str, ...]              # 动态 LoRA 包装层的规范路径（11.2），供管理器使用
    dynamic_lora: DynamicLoraSpec | None         # 11.1；None 表示不启用
    write_transient: TransientNeed               # 写入期间加速卡上的瞬时需求，04 计入加速卡准入
    ownership_violations: tuple[str, ...]        # 2.4
    dtype_violations: tuple[str, ...]            # 6.2：保持 fp32 的层配了量化方法
    def require_complete(self) -> None: ...      # 两类违例任一非空时报 PlanningError
    def owner(self, tensor_name: str) -> str: ...
    def module_owner(self, module_path: str) -> str: ...
    def set_discovered_blocks(self, component: str, paths: tuple[str, ...]) -> None: ...   # 04 的 plan() 记录自动发现的重复块
    def bind_overlay_slot(self, name: str, tensor: torch.Tensor) -> None: ...
    def set_fully_resident(self, component: str, value: bool) -> None: ...
```

- `declaration_order` 给出组件的声明顺序。04 按这个顺序调用 05 的 `resolve_selection()`、清点 `resident_layers`；`entries` 按组件名排序，只用于查找。
- `tensors` 是最终张量表：参数、持久 buffer、登记过来源的数据张量、量化方法声明的输出张量（7.6）、`overlay` 固定适配器的 A/B（8.5）。量化层量化前的参数不在表里，它们只在写入期间以临时张量的形式存在于加速卡上。
- `kind="buffer"` 只指持久 buffer。没有登记来源的非持久 buffer（例如按配置算出的 RoPE 表）不在表里，它们跟随所在模块由 Offloader 放置。
- `bind_overlay_slot` 只给 04 用：04 在 `prepare_materialization()` 里分配槽位，在 `commit()` 里调用它把槽位装进包装层，然后交给动态 LoRA 管理器。每个槽位只能绑定一次，形状和类型必须与声明一致。
- `set_discovered_blocks` 只给 04 的 `plan()` 用，每个组件最多调用一次；`set_fully_resident` 只给 04 的 `commit()` 用，每个组件只能调用一次，之后属性只读。这三个方法是目录上仅有的写操作，重复调用报 `LifecycleError`。
- `shapes_final`、`finalize_replaced` 和 `write_transient` 都在只建结构结束时算出，规划据此选择共享条件和冷启动路径（03、04），不需要等到预检。`write_transient` 是上限：`adapter_delta_bytes` 取所有要融合适配器的组件里最大的单个张量按计算类型的字节数，`quant_layer_bytes` 取量化层里量化前参数之和的最大值。

#### 2.8 查询接口

`weight_stack/components.py` 提供以下只读函数，供 `torch.compile` 的目标选择、缓存加速、动态 LoRA、模型注册表、序列并行钩子和 VAE 快路径使用（第 15 节）。目录不挂在 pipeline 对象上（总体设计 3.2 第 5 条）：`build_structure()` 把它登记在 `components.py` 的进程级弱引用表里，以 pipeline 对象为键，pipeline 被回收时条目随之消失；`custom_pipeline` 重新初始化时，新 pipeline 登记新的目录。前三个函数和 `fully_resident` 带一个 `legacy` 参数：`legacy` 下函数只调用它并返回结果，使用方现有的判断原样移进这个局部函数，`legacy` 下行为不变。

```python
def component_paths(pipeline, role: Role, *, legacy: Callable[[], Sequence[str]]) -> tuple[str, ...]: ...
def lora_target_paths(pipeline, *, legacy: Callable[[], Sequence[str]]) -> tuple[str, ...]: ...
def repeated_block_paths(pipeline, component: str, *, legacy: Callable[[], Sequence[str]]) -> tuple[str, ...]: ...
def fully_resident(pipeline, component: str, *, legacy: Callable[[], bool]) -> bool: ...
def catalog(pipeline) -> ComponentCatalog: ...          # unified 下返回目录；legacy 下报 LifecycleError
def pipeline_of(catalog: ComponentCatalog) -> nn.Module: ...   # 以这个目录登记的 pipeline；找不到时报 LifecycleError
def execution_device() -> torch.device: ...
def is_unified_stack() -> bool: ...
def weight_generation(pipeline) -> int: ...
def bump_weight_generation(pipeline) -> None: ...
```

- `unified` 下前三个函数读目录：`component_paths` 返回这个角色的组件的规范根路径；`lora_target_paths` 返回 `dit` 组件和 `_lora_components` 的根路径；`repeated_block_paths` 返回组件声明的 `repeated_blocks`，没有声明时返回 04 经 `set_discovered_blocks()` 记录的自动发现结果。
- `fully_resident` 是读这个属性的通用函数：`unified` 下读目录里 `commit()` 填好的值，`commit()` 之前调用报 `LifecycleError`；`legacy` 下执行模型现有的判断。
- `pipeline_of(catalog)` 读同一张进程级弱引用表，返回以这个目录登记的 pipeline 对象。它只给 04 的 `plan()` 读模块树用：模块的类型、`forward` 是否被覆写、子模块的注册顺序（总体设计 4.1），由 02-1 提供。
- `execution_device()` 返回本 worker 的执行设备，与 `vllm_omni.diffusion.distributed.utils.get_local_device()` 相同。
- `is_unified_stack()` 返回 01 的 `process_weight_stack() is WeightStack.UNIFIED`。模型代码只在共存期必改清单登记的位置调用它（01 的静态测试 T6 检查调用位置）。
- `weight_generation` 和 `bump_weight_generation` 是以权重为键的缓存签名用的写入代数（总体设计 7.8）。基础权重提交后不变；动态 LoRA 管理器每次写叠加层槽位后调用一次 `bump_weight_generation`。签名的计算归 04。

### 3. 只建结构的构造上下文

#### 3.1 进入、阶段与退出

`StructureContext`（`weight_stack/loader/context.py`）是进程级的单例状态，与 01 的进程级总开关一起使用。同一时刻一个进程里最多有一个打开的上下文，第二个上下文在第一个关闭之前打开时报 `LifecycleError`，不排队等待：持有上下文的 worker 可能正在 `seal()` 里等待排队的那个 worker 作为写入进程完成封存，排队会死锁。单进程执行器把几个 worker 放在同一个进程里时，它们的 `unified` 启动必须依次进行。陷阱的补丁对进程里的所有线程生效，所以上下文打开期间，同一进程里的其他代码也不能读取 checkpoint。

```text
build_structure() 开始     phase = CONSTRUCT   参数落到 meta、设备移动受限、读取陷阱拦截张量读取
build_structure() 结束     phase = SEALED      组件构造函数和 declare_weights 不再可用；陷阱只拦截文件读取
write()、finalize()        phase = SEALED      同上
finalize() 结束            上下文关闭          陷阱撤除；违例在这里最后核对一次
loader.close()             上下文关闭          失败路径上同样撤除全部补丁
```

`model_loader/construct.py` 导出 `is_building_structure() -> bool`，`phase` 为 `CONSTRUCT` 时为真。模型代码对它的调用按 01 的 T6 登记。

#### 3.2 参数落到 meta

- `CONSTRUCT` 阶段，Loader 替换 `nn.Module.register_parameter`：传入的参数不在 meta 上时，按 `vllm_omni.diffusion.models.utils.release_module_parameters_to_meta` 的方式生成同类、同 `requires_grad`、`__dict__` 相同的 meta 参数，再调用原函数注册。参数的子类（vLLM 的 `BasevLLMParameter` 各子类）和 `weight_loader` 等属性因此保留。传入的参数已经在 meta 上时原样注册，构造时就共用的参数保持对象身份。
- `nn.Module.__setattr__` 对 `nn.Parameter` 的赋值经过 `register_parameter`，也被覆盖。`nn.ParameterList`、`nn.ParameterDict` 的元素同样经过 `register_parameter`。
- 这项转换与外层的 `torch.device` 上下文和 `set_default_torch_dtype` 无关：MiniMax H3 VAE 在 `with set_default_torch_dtype(torch.float32), torch.device("cpu")` 里构造，参数同样落到 meta，类型取构造出的 fp32。
- buffer 不转换，照常在 CPU 上创建。Loader 自己不进入任何设备上下文。

#### 3.3 设备移动的限制

`CONSTRUCT` 阶段替换 `nn.Module.to`、`cuda`、`cpu`、`xpu`、`to_empty`，以及平台插件在 `nn.Module` 上注册的同类设备方法：

- `to(*args, **kwargs)` 用 `torch._C._nn._parse_to` 解析出设备、类型、`non_blocking` 和内存格式，只把类型和内存格式交给原函数，设备丢弃。
- `cuda`、`cpu`、`xpu` 和平台插件的同类方法直接返回模块本身；`to_empty` 报 `ConstructionError`，因为它会重新分配 buffer。
- 对单个张量的 `Tensor.to(device)` 不拦截；对 meta 张量做这种调用由 PyTorch 自己报错，Loader 把这个异常包成 `ConstructionError`，写明调用位置。

#### 3.4 transformers 与 diffusers 的类

- vLLM 风格的组件直接运行它们的配置构造函数。
- `create_transformers_model` 和 `create_transformers_model_with_vllm_linears` 在 `is_building_structure()` 为真时跳过 `init_parameters`、`init_parameters_preserving_vllm_linear`，其余语句不变；这两个位置是本 RFC 的分发点，在登记表里用 `NEW_ROW` 声明为 D06（第 17 节）。
- diffusers 的类用 `from_config`；构造函数里直接调用 `X.from_pretrained` 的地方改成 `hf_component`（5.5）。

#### 3.5 构造后检查与最终类型

`build_structure()` 在 pipeline 构造函数返回之后依次做：

```text
1. 执行 empty_cache；构造期间的分配不进任何睡眠池
2. 默认推导或读取 declare_components 的声明（2.1、2.2），解析路径（2.3）
3. 动态 LoRA 和 overlay 固定适配器的包装（11.2、8.5）
4. 最终类型：每个张量先取构造出的类型，再按规则表覆盖（第 6 节），用 swap_tensors 落到 meta 参数上，保留参数对象和属性
5. 量化方法声明输出张量（7.6）
6. 构造后检查：所有参数都在 meta 上；除 declare_weights 登记了来源的非持久 buffer 以外，buffer 和模块的普通属性里没有 meta 张量
7. 核对读取陷阱的违例（第 4 节）
8. 构建组件目录：最终张量表、完整性违例、类型违例（6.2）、shapes_final、finalize_replaced、write_transient
```

- 第 6 步遍历每个模块的 `__dict__`，找出类型为 `torch.Tensor` 的普通属性和 `list`、`tuple`、`dict` 里一层深的张量。出现 meta 张量时报 `ConstructionError`，写明模块路径和属性名，说明构造期间用权重数值做了计算。第 3 步包装时 Loader 自己建出的 meta 张量（动态 LoRA 槽位、`overlay` 固定适配器的 A/B）记在目录上，不算违例。
- 第 6、7 步的任何错误都在构造阶段报出，报错信息的 `fix` 是 `use --diffusion-weight-stack legacy`。

### 4. 读取陷阱

`ReadTrap`（`weight_stack/loader/trap.py`）从 `build_structure()` 开始一直开到 `finalize()` 结束。

| 被拦截的调用 | `CONSTRUCT` 阶段 | `SEALED` 阶段 |
|---|---|---|
| `safe_open(...)` 返回的句柄的 `get_tensor`，`get_slice(...)` 的切片取值 | 报错 | 报错 |
| `safetensors.torch.load_file`、`safetensors.torch.load` | 报错 | 报错 |
| `torch.load` | 报错 | 报错 |
| diffusers `FromOriginalModelMixin.from_single_file` | 报错 | 报错 |
| transformers `PreTrainedModel.from_pretrained`、diffusers `ModelMixin.from_pretrained`、`DiffusionPipeline.from_pretrained`，不经过 HF 组件入口的直接调用 | 报错 | 报错 |
| `nn.Module.load_state_dict`，`state_dict` 里有不在 meta 上的张量 | 报错 | 放行 |
| 非 Hugging Face 类自定义的 `from_pretrained` | 由它内部的文件读取触发 | 同左 |

- 放行的头部读取：`safe_open` 句柄的 `keys()`、`metadata()`、`get_slice(name).get_shape()`、`get_slice(name).get_dtype()`，以及 JSON 配置文件的读取。
- 实现：陷阱用代理对象替换 `safetensors.safe_open`、`safetensors.torch.safe_open`、`safetensors.torch.load_file`、`safetensors.torch.load`、`torch.load`、`torch.serialization.load`，以及上面列出的类方法。模型模块常用 `from safetensors import safe_open` 把名字绑进自己的命名空间，`vllm`、`diffusers` 的辅助函数也这样绑定，所以陷阱在打开时扫描 `sys.modules` 里除 `torch`、`safetensors` 两个包以外的全部模块，把与原对象身份相同的全局名字换成代理，关闭时换回；上下文打开期间新 import 的模块由一个 `sys.meta_path` 查找器在执行完模块代码后同样处理。
- Loader 自己的 checkpoint 读取器不受影响：读取器只在 `trap.trusted()` 里调用读取函数，代理看到线程局部的信任标记时直接调用原函数。读取器的线程池在每个工作线程里进入 `trusted()`。
- 违例记录：代理抛出 `ConstructionError` 或 `WriteError` 之前，把一条 `Violation(kind, call, where)` 记进陷阱。`where` 取调用栈上第一个不属于 `torch`、`safetensors`、`transformers`、`diffusers` 和 `weight_stack` 的帧，写成文件路径和函数名。即使模型代码用 `try/except` 吞掉了异常，`build_structure()` 第 7 步和 `finalize()` 结束时照样报错，列出全部违例。
- `CONSTRUCT` 阶段违例的报错类型是 `ConstructionError`；`SEALED` 阶段是 `WriteError`（写入期间）或 `FinalizeError`（收尾期间）。

### 5. 构造期的分发

#### 5.1 进程级状态

分发和读取陷阱看同一个进程级状态：01 的 `process_weight_stack()` 和 3.1 的 `StructureContext.phase`。除共存期必改清单登记的分支以外，模型不读总开关。

#### 5.2 `build_component`

```python
# vllm_omni/diffusion/model_loader/construct.py
def build_component(*, legacy: Callable[[], T], unified: Callable[[], T]) -> T: ...
```

| 状态 | 行为 |
|---|---|
| `legacy` | 只调用 `legacy()`，返回它的结果。 |
| `unified`，`phase` 为 `CONSTRUCT` | 只调用 `unified()`，返回它的结果。 |
| `unified`，上下文没有打开或 `phase` 为 `SEALED` | 报 `LifecycleError`，写明调用位置（文件路径和函数名），说明组件必须在构造函数里建出。 |

- `legacy` 一支和 `unified` 一支的写法按总体设计 5.2“两个栈的构造代码”：`legacy` 是原语句原样移进的局部函数；`unified` 只用配置建结构，返回同一个类的实例和相同的模块树，对建出的每个带权重的模块调用一次 `declare_weights`，读张量、拼 state dict 的部分放在家族目录下的 `unified_build.py`。`unified` 调用 legacy 的路径解析、配置解析和校验函数，不复制它们；需要时可以给原有的类加新方法，原有方法不改。模块树相同由第 14 节的逐张量对比测试检查，`build_component` 在运行期不检查。
- 按需构造的组件（例如 SANA-WM 在请求期才建的 VAE）在 `unified` 一支里于构造函数中建出。

#### 5.3 `declare_weights`

```python
def declare_weights(
    module: nn.Module,
    *,
    sources: WeightSource | BufferSource | Sequence[WeightSource | BufferSource],
    rules: RuleTable | str | None = None,        # 规则表或它的限定名
    writer: Callable[[nn.Module, Iterable[tuple[str, torch.Tensor]]], Iterable[str] | None] | None = None,
) -> None: ...
```

- 只能在 `phase` 为 `CONSTRUCT` 时调用；其他任何时候（包括 `legacy` 下）报 `LifecycleError`，写明调用位置。
- 同一个模块登记两次报 `ConstructionError`。登记的模块嵌套时，内层模块的张量归内层的登记，外层得到剩下的部分，规则与 2.3 的划出相同。
- 写入入口的选择：给了 `writer` 就调用 `writer(module, stream)`；模块有 `load_weights(weights)` 就调用它；否则用 vLLM 的 `AutoWeightsLoader(module).load_weights(stream)` 按参数名写入。这些流不经过 pipeline 的 `load_weights`。
- 返回值是已写入的参数名集合时，Loader 用它补充写入跟踪的日志；写入是否完整只按写入跟踪判断（7.5）。

#### 5.4 来源的类型

```python
@dataclass(frozen=True)
class WeightSource:
    location: str                        # 本地目录、文件或 Hub 仓库
    files: tuple[str, ...] = ()          # 相对 location 的文件名或通配模式；空表示按 safetensors 索引或 *.safetensors
    subfolder: str | None = None
    revision: str | None = None
    format: Literal["safetensors", "torch"] = "safetensors"
    weights_only: bool = True            # torch 格式：与 legacy 一支的 torch.load 参数相同
    nested_key: str | None = None        # torch 格式里 state dict 所在的键，例如 "generator"
    strip_prefix: str = ""               # 去掉 checkpoint 键名的这个前缀
    add_prefix: str = ""                 # 再加上这个前缀，得到相对模块的名字
    include: tuple[str, ...] = ()        # 键名的通配模式；空表示全部
    exclude: tuple[str, ...] = ()

@dataclass(frozen=True)
class BufferSource:
    buffer: str                          # 相对模块的非持久 buffer 名
    location: str
    file: str
    key: str | None = None               # safetensors 键名或 torch 文件里的键；None 表示整个文件是一个张量
    format: Literal["safetensors", "torch"] = "safetensors"
    weights_only: bool = True
```

- `sources` 是列表时，列表里靠后的来源覆盖靠前来源里同名的键：读取器先按全部来源的键名算出每个最终键来自哪个来源，只产出这一个张量，每个最终张量因此只写一次。Ming-Flash-Omni 的 ByT5 编码器依次叠加三个文件就用这种写法。
- `BufferSource` 登记预先算好的数据张量。模型在构造函数里注册形状和类型正确的 meta 非持久 buffer，形状来自 safetensors 头部（陷阱放行）或配置；torch 格式没有头部时形状必须来自配置。Loader 在写入阶段按来源写入，类型不同时报 `WriteError`。
- 来源路径相对模型目录时，用 `od_config.model` 解析出的目录拼接；Hub 仓库由 7.1 的解析下载。

#### 5.5 HF 组件入口

两个入口内部走同一个分发 `_hf_dispatch(cls, model, subfolder, kwargs)`：

- `from_pretrained_with_prefetch(factory, model, *, subfolder, prefetch_list, local_files_only, max_attempts, **from_pretrained_kwargs)`（`vllm_omni/diffusion/model_loader/hub_prefetch.py`）：`legacy` 下的代码不变。`phase` 为 `CONSTRUCT` 时，从绑定的类方法 `factory.__self__` 取出类，进入分发；`factory` 不是绑定的类方法时报 `ConstructionError`。
- `hf_component(factory, *args, **kwargs)`（`construct.py`）：用在构造函数里直接调用 `X.from_pretrained(...)` 的地方。`legacy` 下就是 `factory(*args, **kwargs)`，不附加预取和重试；`phase` 为 `CONSTRUCT` 时进入分发。
- `unified` 下上下文没有打开（包括 Loader 关闭上下文之后）或 `phase` 为 `SEALED` 时，两个入口对带权重的类都报 `LifecycleError`，写明调用位置（总体设计 5.2）。
- 分词器、processor、调度器、特征提取器和纯配置类（不是 `PreTrainedModel` 或 `ModelMixin` 的子类）直接通过，照常调用 `from_pretrained`。transformers 的 `AutoModel` 一类先按配置解析出模型类，再按模型类判断，Gr00t 的 `AutoModel.from_pretrained` 因此进入分发。

分发的步骤：

```text
1. 读配置：transformers 用 AutoConfig 或 cls.config_class.from_pretrained，diffusers 用 cls.load_config；
   只读 JSON
2. config 自带 quantization_config 时报 ConstructionError（from_config 不会套用 Hugging Face 的量化器）
3. 查这个类的规则表；没有入库的表时报 ConstructionError，写明类名和生成表的命令
4. 在构造上下文里调用 cls.from_config(config, **允许的构造参数)；torch_dtype 作为构造出的类型
5. 执行规则表登记的构造后处理（例如 DistributedAutoencoderKLWan 的 init_distributed）
6. 对建出的模块登记 HF 来源：location 为 model，subfolder、revision 取调用参数，规则表取第 3 步的表
```

`from_pretrained` 的调用参数里改变加载方式的键（`device_map`、`low_cpu_mem_usage=False`、`quantization_config`、`offload_folder`、`variant` 以外的文件选择键）在第 4 步之前报 `ConstructionError`；`variant` 进入来源的文件选择；`torch_dtype` 和 `revision` 按上面的规则使用。

#### 5.6 `weights_sources` 与写入入口

- pipeline 的 `weights_sources` 按前缀归到组件：前缀是组件根路径加 `.` 的来源属于这个组件；前缀为空的来源覆盖 pipeline 下没有被 `declare_weights` 登记的全部张量。这些流交给 pipeline 的 `load_weights`。
- pipeline 没有 `weights_sources` 时，`unified` 不调用它的 `load_weights`。
- `declare_weights` 登记的模块里的张量只由这条登记写入。pipeline 的 `load_weights` 写进这些张量时，写入跟踪发现第二次写入，报 `WriteError`。
- pipeline 的 `load_weights` 自己读文件、空操作、或替没有写入的组件声明“已加载”的家族，在 `unified` 一支里让 `weights_sources` 不含这些前缀（`weights_sources` 的赋值本身经 `build_component` 分成两支），并用 `declare_weights` 登记对应的模块。

#### 5.7 未迁移的 pipeline 与支持矩阵

- 未迁移的 pipeline 在 `unified` 下构造时触发读取陷阱或构造后检查，启动报 `ConstructionError`，写明读取发生的位置，`fix` 是 `use --diffusion-weight-stack legacy`。是否已迁移不另设标记。
- `tests/diffusion/weight_stack/loader/test_structure_matrix.py` 为 `DiffusionModelRegistry` 里注册的每个 pipeline，用测试资源里的配置文件在上下文里调用一次 `build_structure()`（不需要权重），记录结果：成功、触发陷阱的位置、残留 meta 张量的位置、完整性违例、缺少的规则表，以及成功时目录里的组件名和角色。结果按 [05](./05-configuration.md) 第 17.3 节规定的格式写进 `structure_matrix.json`，由 05 渲染成 `docs/` 下的支持矩阵。
- 共存期间不合入构造期读张量的新 pipeline：这个测试对新注册的 pipeline 要求结果为成功。

### 6. 规则表

#### 6.1 格式

规则表是数据，放在 `weight_stack/loader/rules/`，每个类一个模块，模块里只有一个 `TABLE: RuleTable` 常量，由 `rules/__init__.py` 的注册表按类的限定名索引。

```python
@dataclass(frozen=True)
class RuleTable:
    cls: str                                 # 限定名，例如 "transformers.UMT5EncoderModel"
    version: int                             # 内容变化时加一，进入内容身份
    validated_with: Mapping[str, str]        # 验证时的库版本，例如 {"diffusers": "0.40.0"}
    tied: tuple[tuple[str, str], ...] = ()               # （别名，规范名）
    ignore_on_load: tuple[str, ...] = ()                 # 加载时忽略的 checkpoint 键，通配模式
    conversions: tuple[Conversion, ...] = ()             # transformers v5 的改名、合并、堆叠
    dtype: DtypeRule = DtypeRule()                       # 类型解析
    keep_fp32: tuple[KeepFp32, ...] = ()                 # 保持 fp32 的条目
    write_transforms: tuple[WriteTransform, ...] = ()    # 写入时的变换，例如折叠 weight norm
    initializers: tuple[Initializer, ...] = ()           # 不在 checkpoint 里、初值有意义的参数
    runtime_dtype: tuple[RuntimeDtype, ...] = ()         # 运行时类型固定的子模块，按平台区分
    memory_format: tuple[MemoryFormatRule, ...] = ()     # 例如 channels_last
    post_construct: tuple[str, ...] = ()                 # 构造后处理的限定名

@dataclass(frozen=True)
class KeepFp32:
    pattern: str                                         # 参数名的通配模式
    value: Literal["checkpoint", "stage_rounded"]        # 默认 checkpoint：直接转成 fp32；
                                                         # stage_rounded：先转成 stage 类型再升到 fp32

@dataclass(frozen=True)
class WriteTransform:
    kind: Literal["fold_weight_norm", "transpose", "reshape", "custom"]
    targets: tuple[str, ...]
    on_execution_device: bool = False                    # 须在执行设备上做
    fn: str | None = None                                # kind 为 custom 时的限定名

@dataclass(frozen=True)
class RuntimeDtype:
    module: str                                          # 相对组件根的模块路径
    dtype: torch.dtype
    value: Literal["checkpoint", "stage_rounded"] = "checkpoint"
    platforms: tuple[str, ...] = ()                      # 空表示全部平台，例如 ("mps", "xpu", "cpu")
```

`Conversion` 有三种：`rename(src_pattern, dst_pattern)`、`merge(srcs, dst, dim)`、`stack(src_pattern, dst, dim)`。`DtypeRule` 写明类型的来源：构造参数、`config.torch_dtype` 或固定类型。`Initializer` 写明参数名和初始化方式（`zeros`、`ones`、`constant`、`from_config` 加函数限定名）。`MemoryFormatRule` 写明目标模块的类型或路径、内存格式和生效条件；条件是 `od_config` 的键和取值，需要时再加平台，与 legacy 一支转换布局的条件相同。例如 Wan VAE 的卷积在 `vae_fast_path` 为 `channels_last` 时用 channels-last 布局，这一级别在 `install_wan_vae_fastpath` 里转换卷积权重的布局。

#### 6.2 执行

- 构造时：`dtype`、`keep_fp32`、`runtime_dtype`、`memory_format` 决定 3.5 第 4 步落到 meta 参数上的最终类型和步长；`post_construct` 在 5.5 第 5 步执行；`tied` 在 2.3 去重时合并别名。
- 写入时：读取器按 `ignore_on_load` 丢弃键，按 `conversions` 改名、合并、堆叠，再交给写入入口；`write_transforms` 在写进最终存储之前执行，`on_execution_device` 为真的变换在执行设备上做，结果再拷进最终存储；`keep_fp32` 的 `stage_rounded` 条目先转成 stage 类型再升到 fp32，与现有实现逐位一致。
- 收尾时：`tied` 在第 3 步核对；`initializers` 在第 1 步给没有写入、规则表声明了初始化方式的参数写初值，记为已写。
- 运行时 Loader 只读入库的表，不读类上的 `_keep_in_fp32_modules`。
- `keep_fp32` 或 `runtime_dtype` 为 fp32 的层，量化方法必须是“不量化”。Loader 在 3.5 第 8 步把不满足的层和它的量化方法记进 `ComponentCatalog.dtype_violations`，由 04 的 `plan()` 经 `require_complete()` 在规划阶段报 `PlanningError`（总体设计 5.2），报错写明层名和量化方法。
- `runtime_dtype` 表达会在 `__call__` 里改组件类型的 diffusers pipeline（例如 SDXL fp16 的 `upcast_vae()`）和模型自己改类型的代码（MiniMax H3 视频 VAE 的 fp32、LTX2 vocoder 在 mps、xpu、cpu 上的 fp32），组件因此从写入起就是运行时的类型。

#### 6.3 逐表测试

`tests/diffusion/weight_stack/loader/rules/test_rule_tables.py` 对注册表里的每张表各跑一个用例：

```text
1. 用表旁边的测试配置（tests/diffusion/weight_stack/loader/rules/configs/<cls>.json，缩小到几层）
   构造一个随机初始化的模型，save_pretrained 到临时目录
2. 参考：cls.from_pretrained(tmp, torch_dtype=stage 类型)
3. 被测：HF 组件入口在上下文里建结构，Loader 按规则表从同一目录写入（全部常驻）
4. 逐张量比较：名字、形状、类型、步长、字节、共用关系、buffer
5. 断言已安装的库版本等于 table.validated_with
```

- 库版本不等时测试失败，升级 transformers 或 diffusers 时每张表都要重新验证并更新 `validated_with`。
- 用 `from_single_file` 的类（Wan2.2-S2V 的 VAE），参考换成 `from_single_file`，被测用规则表的 `conversions` 表达单文件格式的键转换。
- `python -m vllm_omni.diffusion.weight_stack.loader.rules.generate <cls>` 从类的 `_tied_weights_keys`、`_keys_to_ignore_on_load_unexpected`、`_keep_in_fp32_modules` 和 transformers 的转换表生成表的初稿，初稿只有在上面的测试通过后才能入库。

### 7. checkpoint 读取器与写入

#### 7.1 来源解析

`preflight()` 把每条来源解析成 `ResolvedSource`：

```python
@dataclass(frozen=True)
class ResolvedSource:
    binding: str                         # TensorDecl.source 里的来源绑定标识
    component: str
    files: tuple[str, ...]               # 本地绝对路径，按 _natural_sort_key 排序
    format: Literal["safetensors", "torch"]
    headers: Mapping[str, TensorHeader]  # 键名到形状和类型；torch 格式在第一次打开时填写
    prefix: str
    quant_config: object | None          # 与 DiffusersPipelineLoader._get_source_quant_config 相同
    checkpoint_adapter: str | None       # get_checkpoint_adapter 选中的适配器
```

- `weights_sources` 的来源（`DiffusersPipelineLoader.ComponentSource`）用 `DiffusersPipelineLoader._prepare_weights` 解析文件，与 legacy 一致；读取器在函数内 import `diffusers_loader`，用同一份 `LoadConfig` 和 `od_config` 构造一个只用于解析的实例。
- `WeightSource`、`BufferSource` 和 HF 来源用 `prefetch_subfolders`（`hub_prefetch.py`）下载并校验缓存，再按 `files` 选择文件。
- 预检读每个 safetensors 文件的头部，核对来源可读。名字映射完全由 Loader 决定的来源（HF 来源和没有 `writer`、模块也没有 `load_weights` 的 `declare_weights` 来源）另外核对：每个 `required` 的张量在按规则表转换之后的键里出现。不满足时报 `PreflightError`，写明组件、来源和键名。经 pipeline 或模块 `load_weights`、`writer` 写入的张量，名字映射在模型代码里，预检不核对，是否写到由收尾第 4 步按写入跟踪核对（总体设计 5.4）。
- 预检同时核对规划选定的冷启动路径：组件的 `shapes_final` 为假时，规划不能选瘦路径，否则报 `PreflightError`。

#### 7.2 读取流水线

每条来源的流按下面的顺序产生：

```text
文件              safetensors 的 safe_open（trusted），torch 格式的 torch.load(mmap=True)
键名过滤          ignore_on_load、include、exclude
加前缀            ComponentSource.prefix 或 WeightSource 的 strip_prefix、add_prefix；得到 checkpoint 张量名
checkpoint 适配器 get_checkpoint_adapter 选中的格式转换（例如 ModelOpt）
适配器融合        W ⊕ Δ[checkpoint 张量名]（第 8 节）
规则表转换        conversions 的改名、合并、堆叠
写入入口          pipeline.load_weights、模块的 load_weights、writer 或 AutoWeightsLoader
```

- 格式和并行方式的选择规则与 `DiffusersPipelineLoader._get_weights_iterator` 相同：torchao 序列化的 checkpoint 按 torch 格式读；`enable_multithread_weight_load` 为真时，safetensors 文件用线程池按 `num_weight_load_threads` 并行打开，文件按 `_natural_sort_key` 排序。含在线量化层的组件不开线程池，按键名的自然顺序产出，以减少同时存在的量化前临时张量（风险表）。
- 规划给这个组件选了瘦路径（04 的冷启动路径）时，读取器同样不开线程池，按 Unit 的顺序产出：加前缀之后的 checkpoint 张量名按最长的 Unit 模块路径前缀归到 Unit，按 `PlacementPlan` 里分片 Unit 的顺序逐个 Unit 产出，Unit 内按键名的自然顺序，归不到 Unit 的键最后产出。每个 Unit 的每个类型组最多两个边界张量，所以同一时刻没写完的边界张量不超过两个，瘦路径的冷启动额外占用不超过释放阈值加最大的两个张量（总体设计 6.4，[03](./03-host-storage.md) 第 9.2 节）。
- 读取器产出的是延迟张量（7.3）；融合和规则表转换都作用在延迟张量上，真正的读取发生在写进最终存储的那一刻。

#### 7.3 延迟张量

`LazyTensor` 是用 `torch.Tensor._make_wrapper_subclass` 建的包装子类，记录文件、键名、形状、类型，以及叠加在它上面的视图操作、类型转换和待融合的 Δ。

- `narrow`、`chunk`、`split`、切片、`view`、`reshape`、`t`、`permute`、`to(dtype)` 返回新的延迟张量，不读数据。
- 目标是 meta 张量的 `copy_` 什么都不做，也不读数据：瘦路径下完全落在本 rank 范围之外的张量（03 的 `view()` 返回 meta 张量）和同机共享时等待者写进共享存储的张量（同样是 meta）走这一条。
- 其他任何操作先物化：safetensors 用 `get_slice` 只读视图覆盖的范围，torch 格式从 mmap 的张量取；有待融合的 Δ 时在这一步计算并融合（8.2）；然后执行原操作。
- 读取器在启动日志里报告目标为 meta 却被物化的字节数。瘦路径测试对门槛矩阵里的家族断言这个数为 0。

#### 7.4 存储绑定

`write()` 的第一步对目录里每个 `TensorDecl` 绑定存储：

```text
t = transaction.target(name)
t 为 None：在 ctx.memory_pools.weights() 里、在执行设备上 torch.empty_strided(shape, stride, dtype)
          HSDP 管理的组件例外：在 CPU 上分配（第 13 节）
t 不为 None：核对形状、类型、步长与 TensorDecl 相同，不同时报 WriteError
参数：torch.utils.swap_tensors(meta 参数, 与 t 共用存储的同类参数)，保留参数对象、__dict__ 和 weight_loader
buffer：module._buffers[name] = t；别名指向同一个对象
```

- 绑定后，Loader 为每个张量记录绑定快照：对象身份、`untyped_storage()._cdata`、`storage_offset`、形状、步长、类型。收尾第 4 步用它判断改绑。
- 量化层量化前的参数不在目录里，不在这里绑定（7.6）。

#### 7.5 写入入口与写入跟踪

- 写入入口按 5.3、5.6 的规则选出，每个入口在 `WriteTracker` 的 `TorchDispatchMode` 里调用。这个模式拦截 `copy_`、`index_copy_`、`index_put_`、`fill_`、`zero_`、`masked_fill_`、`scatter_` 等原地写操作，按目标的 `untyped_storage()._cdata` 和偏移找到所属的 `TensorDecl`，累加写入的元素数。目标是 meta 张量时同样计数，所以瘦路径和等待者的空操作“记为已写”。
- 绑定之后，Loader 包装每个参数的 `weight_loader`（总体设计 5.3）：没有 `weight_loader` 的参数以 `default_weight_loader` 为原函数，`getattr(param, "weight_loader", default_weight_loader)` 的结果因此不变。包装函数在调用原函数之前通知 `WriteTracker` 当前参数，原函数返回之后检查这个参数所在的量化层是否已经写完（7.6）。TP 切分、合并参数、模型自己的映射仍由原函数完成。
- 一个张量的累计数超过它的元素数时报 `WriteError`，写明张量名：每个最终张量只做一次有语义的写入。
- 一个张量的累计写入元素数第一次达到它的元素数时（目标为 meta 张量时同样计数），`WriteTracker` 在写入线程里调用一次 `storage.note_written(name)`，`name` 是规范名。只对 `transaction.target(name)` 不为 `None` 的张量调用，包括量化方法声明的输出张量；这个调用不做集合通信。分片后端的瘦路径据此按已写字节数同步归还范围之外的页（[03](./03-host-storage.md) 第 9.2 节）。
- 写入跟踪覆盖每条流的全部写入，包括不经过 `weight_loader` 的 `writer`、`load_state_dict` 和 buffer 的写入。

#### 7.6 逐层在线量化

- 构造结束时，Loader 对每个带 `quant_method` 的层调用 `quant_output_specs(layer)`（`weight_stack/loader/quant.py`），得到这一层量化后每个输出张量的名字、形状和类型，作为 `kind="quant_output"` 进入目录。没有登记输出声明的量化方法，在构造阶段报 `ConstructionError`。
- 写入阶段，包装后的 `weight_loader` 第一次被这一层的参数调用时，Loader 在执行设备上为这一层量化前的参数分配临时张量并绑定，然后才调用原函数；量化层量化前的参数被不经过 `weight_loader` 的写入命中、而临时张量还没有分配时报 `WriteError`。包装函数返回时，写入跟踪发现这一层量化前的参数全部写完，Loader 立即调用 `quant_method.process_weights_after_loading(layer)`，把每个输出张量拷进它的最终存储，把层的属性换成最终存储上的张量，然后释放临时张量。量化在加速卡上进行，结果直接写进最终存储。
- 输出目标全部是 meta 的量化层（等待者写进共享存储的层、瘦路径下完全落在本 rank 范围之外的层）：写入期间量化前的参数留在 meta 上，写入是空操作，不读 checkpoint；这一层写完时，Loader 分配不初始化的临时张量，照样调用 `process_weights_after_loading`，使层上不属于权重的运行期属性与写入进程相同，然后丢弃输出、释放临时张量。
- 输出的形状或类型与声明不同时报 `WriteError`。
- 离线量化的 checkpoint 同样经过这一步：`process_weights_after_loading` 改字节的方法（重新打包等）由它的输出声明给出最终张量。
- 收尾第 1 步对还没写完的量化层报 `FinalizeError`，量化方法声明可以缺少的张量放过，规则与 `DiffusersPipelineLoader._is_expected_quantized_weight` 列出的后缀相同。
- Loader 不使用 vLLM 的逐层物化钩子 `finalize_layerwise_processing`，也不做 `module.to(...)` 的搬运。

### 8. 适配器

#### 8.1 格式转换器

每种适配器格式有一个转换器，登记在 `weight_stack/loader/adapters/` 的注册表里：

```python
class AdapterFormat(Protocol):
    name: str                                                     # "diffusers"、"kohya"、"fasth3"、"vdn"、"ltx_phase"、"longcat_dmd"
    fusable: bool
    overlayable: bool
    def claim(self, location: AdapterLocation, catalog: ComponentCatalog) -> bool: ...
    def manifest(self, location: AdapterLocation, catalog: ComponentCatalog) -> AdapterManifest: ...   # 只读头部和元数据
    def delta(self, entry: AdapterEntry, reader: AdapterReader, device: torch.device) -> torch.Tensor: ...
    def combine(self, w: torch.Tensor, delta: torch.Tensor, stage_dtype: torch.dtype) -> torch.Tensor: ...
    def extra_sources(self, location: AdapterLocation) -> tuple[WeightSource, ...]: ...
```

- `manifest()` 把适配器的键映射成 checkpoint 张量名。Wan2.2 的适配器按位置对应 `transformer`、`transformer_2` 两个组件。
- 用户经 `--lora-path` 给出的适配器，以及模型配置里给出的 FastH3、VDN 路径，由第一个 `claim()` 为真的转换器认领；没有转换器认领时报 `PreflightError`。

#### 8.2 融合

- 读取器对每个 checkpoint 张量 `(name, W)`，有 `Δ[name]` 时产出 `(name, combine(W, Δ))`。融合点和数值规则按总体设计 5.7：Δ 在执行设备上按格式规定的类型计算；`combine` 默认把 W 和 Δ 都先转成 stage 类型再相加，FastH3 在 fp32 里相加、结果取 checkpoint 的类型；融合不乘 `lora_scale`。
- Δ 用完立即释放，峰值多出的显存由 `write_transient.adapter_delta_bytes` 交给 04 计入准入。
- Δ 挂在延迟张量上，只在物化时计算：瘦路径下只为本 rank 实际读取的张量计算，等待者对共享张量不物化，也不计算 Δ。

#### 8.3 FastH3 与 VDN

- 两者只能融合：它们改 RMSNorm 和 bias、注入额外的结构，做不成叠加层。
- `unified` 下，MiniMax H3 pipeline 的构造函数在 `unified` 一支里只读文件头和元数据：`FastH3WeightFusion` 和 `VDNCheckpoint` 各增加一个只读头部、生成适配器清单的新方法，原有方法不改，校验方法（`check_serving_contract` 等）与 legacy 相同；需要先建出的结构（`enable_vsa_gates`、`enable_vdn`）照常在构造函数里建。模型侧的这两个对象不参与写入。
- Loader 的 `fasth3`、`vdn` 转换器按同样的头部算出 Δ 的位置；VDN 的 `linear_branch` 由 `extra_sources()` 作为额外的权重来源，写进 `enable_vdn` 建出的模块。
- MiniMax H3 pipeline 的 `load_weights` 在 `unified` 下收到的 `transformer.` 流已经融合，`self._vdn.apply`、`self._fasth3.apply` 两层流包装只在 legacy 一支里执行；`validate_fully_applied` 一类的核对改由 Loader 的预检完成（8.6）。

#### 8.4 蒸馏 LoRA

- `lora_backend=distill` 加 `--lora-path` 时，适配器由 diffusers 或 kohya 转换器认领并融合。`unified` 下 `DiffusersPipelineLoader._maybe_fuse_distilled_lora` 不运行（legacy 加载路径整体不运行），`DiffusionWorker.init_lora_manager` 的 distill 分支在 `unified` 下只记一条日志，不调用 `pipeline.load_lora_weights`（C16）。
- `lora_backend=peft` 时，普通 A/B 格式作为启动时激活的动态 LoRA（第 11 节）。

#### 8.5 固定适配器

- `apply: fuse`：与用户给出的适配器一样在读取器里融合，进入内容身份。
- `apply: overlay`：只建结构时，Loader 把 `targets` 匹配的层包成 `FixedOverlayLinear`（`weight_stack/loader/overlay.py`）。A/B 以 `kind="fixed_overlay"` 进入目录，属于目标层所在的 Unit，进入内容身份；它们是基础内容，由 Loader 在写入阶段从适配器来源写入，TP 切分沿用目标层现有的布局函数。运行期模型只负责开关（`set_enabled(name, bool)`），不占动态 LoRA 的槽位。
- 支持的目标层是 vLLM 的线性层（`LinearBase` 的子类）和 `nn.Linear`；`targets` 匹配到其他类型时在构造阶段报错。
- LTX2 两阶段蒸馏的相位适配器、LongCat-Video-Avatar 的 DMD LoRA 用 `overlay`（第 16 节）。

#### 8.6 预检

在第一次写模型之前，`preflight()` 对每个适配器：

| 检查 | 结果 |
|---|---|
| 属于已加载组件的键对不上 checkpoint 张量名 | 报 `PreflightError`，列出键名 |
| 属于没加载组件的键 | 记一次警告 |
| 融合时 `lora_scale` 不等于 1 | 报 `PreflightError` |
| 要融合的组件的 checkpoint 已经量化 | 报 `PreflightError` |
| FastH3 的必需键没有全部对上（与 `validate_fully_applied` 的规则相同） | 报 `PreflightError` |

HSDP 预分片配 LoRA 或量化由 05 在参数解析阶段报错。

### 9. 收尾

`finalize(model, storage)` 按固定顺序做一次：

| 步骤 | 实现 |
|---|---|
| 1. 完成所有写入或绑定 | 排空所有流；对规则表声明了初始化方式、没有写入的参数写初值；没写完的量化层报 `FinalizeError`；HSDP 组件在这一步分片（第 13 节）。 |
| 2. 需要整个模型参与、会改字节的工作 | 这类工作只登记在 `loader/finalize.py` 的注册表里，每一项写明触发它的配置和它整个替换的张量名；构造结束时 Loader 按配置选出本模型的工作，把替换的张量记进 `finalize_replaced`。这一步执行选出的工作，新张量常驻加速卡时在 `ctx.memory_pools.weights()` 里分配，记为替换。注册表里没有命中的工作时跳过这一步。 |
| 3. 共用存储的关系 | 共用的张量在 2.3 已并成一个 `TensorDecl`，7.4 绑定时别名已经指向规范张量的同一个对象；这一步把第 2 步替换的张量的别名改指向替换后的张量，再按规则表的 `tied` 和 `TensorDecl.aliases` 核对每个别名与规范张量是同一个对象，不是时报 `FinalizeError`。 |
| 4. 按写入跟踪核对 | 带 `is_checkpoint_required` 和 `required` 的张量都已写到，量化方法声明可以缺少的张量放过；除第 2 步替换的张量以外，每个参数和持久 buffer 的绑定快照与 7.4 记录的相同。不满足时报 `FinalizeError`，写明张量名和变化的字段。 |
| 5. 替换的张量与最终清单 | 第 2 步替换的张量，最终形状和类型与预期清单的条目相同时拷进 `storage.view(name)`（目标为 meta 时不拷）；不同时留在原处，由 `seal()` 按最终清单搬入（03）。然后生成最终清单（10.3）。host 存储的定稿在第 6 步的 `seal()` 里完成（总体设计 4.1）。 |
| 6. host 存储定稿、交接和只读绑定 | 调用 `storage.seal(final_manifest)`；返回后 Loader 重新记录全部绑定快照，第 7 步用它核对。 |
| 7. `eval()` | 先调用 `apply_skip_softmax_calibration`（只给注意力层设置属性，不读写权重），再在每个 worker（包括等待者）上调用 `pipeline.eval()`。 |
| 8. 交给启动事务 | 返回最终清单，由驱动调用 `commit()`。 |

- 第 4 步只核对本 worker 自己写入的张量，等待者的共享张量不做第 4、5 步，在 `seal()` 里接收、核对并绑定（总体设计 5.4）。
- 收尾里的集合通信（HSDP 分片、量化方法的集合通信）全部在第 6 步之前完成，`seal()` 里不做集合通信。
- 第 7 步：调用 `eval()` 之前，Loader 把分片后端 Unit 的参数（`PlacementPlan` 给出这些张量名）临时换成零元素占位，`eval()` 读它们会报错；调用之后换回原来的绑定，再与第 6 步之后记录的快照核对一次，有变化时报 `FinalizeError`。需要这些权重的派生状态在第一次使用时建立（总体设计 5.4 第 7 步）。
- 新栈不调用 `post_load_weights`、`validate_loaded_weights`、`validate_restored_host_weights`。模型在自己 `load_weights` 里做的检查照常运行，例如 MiniMax H3 pipeline 的 `load_weights` 对 transformer 调用的 `post_load_weights`。
- 第 5 步之后字节不再变化。读取陷阱在第 7 步结束时撤除，并最后核对一次违例。

### 10. 内容身份与清单

#### 10.1 `ContentIdentity`

```python
@dataclass(frozen=True)
class ContentIdentity:
    schema: int                                   # 格式版本，取 1
    component: str
    model: str                                    # od_config.model 的原值（仓库名或给定的路径字符串）
    revision: str | None
    roots: tuple[str, ...]                        # 组件边界
    carved: tuple[str, ...]
    tensors_digest: str                           # 最终张量表（名字、形状、类型、步长、kind）的摘要
    stage_dtype: str
    quantization: str | None                      # 量化方法名加它对这个组件解析出的配置的摘要
    rule_tables: tuple[tuple[str, int], ...]      # （限定名，版本）
    loader_rules_version: int                     # Loader 自身的映射规则版本
    tp: tuple[int, int]                           # （大小，编号）
    ep: tuple[int, int] | None                    # 启用专家并行时
    shard_group: tuple[str, int, int] | None      # （名字，大小，编号）
    adapters: tuple[AdapterIdentity, ...]         # 融合的适配器和固定适配器
    def digest(self) -> str: ...                  # 规范 JSON（键排序）的 sha256 十六进制
```

- `AdapterIdentity` 是（格式、路径或仓库、文件名、版本、`apply`）。路径是用户给出的字符串，不是解析后的缓存路径。
- 不包含：offload 策略、叠加层和动态适配器、DP 编号、SP 和 CFG 并行的编号、设备地址、解析后的文件路径、部署标识、checkpoint 和适配器文件的字节内容。
- 测试（`test_content_identity.py`）对每一项断言：改变它时摘要改变；改变每个不包含的项时摘要不变。

#### 10.2 `ExpectedManifest`

```python
@dataclass(frozen=True)
class ManifestEntry:
    name: str                         # 规范名
    component: str
    kind: str                         # 与 TensorDecl.kind 相同
    shape: tuple[int, ...]
    dtype: torch.dtype
    stride: tuple[int, ...]
    nbytes: int
    aliases: tuple[str, ...]
    replaced_at_finalize: bool

@dataclass(frozen=True)
class ExpectedManifest:
    version: int
    entries: tuple[ManifestEntry, ...]          # 按 name 排序
    def component(self, name: str) -> tuple[ManifestEntry, ...]: ...
    def digest(self, names: Iterable[str] | None = None) -> str: ...   # names 为 None 时覆盖全部条目
```

预期清单由预检在写入之前生成，就是目录最终张量表的投影，包括量化输出和 `overlay` 固定适配器的 A/B，不包括叠加层槽位。03 的 `HostWeightManager.layout()` 用它算布局，`digest()` 是交接时核对的预期清单摘要。

#### 10.3 `FinalManifest`

```python
@dataclass(frozen=True)
class FinalEntry:
    name: str
    component: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    stride: tuple[int, ...]
    residency: Literal["host", "device", "hsdp"]
    storage_key: str | None           # host 存储里的张量键，等于规范名，即 view() 用的名字；residency 不是 host 时为 None
    aliases: tuple[str, ...]
    written: Literal["self", "shared", "out_of_range"]   # 不参与摘要

@dataclass(frozen=True)
class FinalManifest:
    version: int
    expected_digest: str
    entries: tuple[FinalEntry, ...]   # 按 name 排序
    def digest(self, names: Iterable[str] | None = None) -> str: ...
    def live(self, name: str) -> tuple[torch.Tensor, ...]: ...   # 绑定到这个名字的模型张量对象（规范张量和别名）
```

- 最终清单是同机共享时按名字绑定的唯一依据。03 的 `seal()` 用 `entries` 和 `HostLayout` 的偏移生成内存文件的 safetensors 头部，用 `live()` 把模型里的张量重新绑定到只读映射上；等待者用同一份代码绑定。
- `live()` 持有模型张量的引用，只在本进程内使用，不进入摘要，不可 pickle。04 的 `commit()` 用 `entries` 校验并做内存登记。
- 写入进程和等待者独立生成最终清单；共享部分的 `digest(names)` 必须相同，03 在交接时核对。

### 11. 动态 LoRA 叠加层

#### 11.1 启用与槽位的 rank

```python
@dataclass(frozen=True)
class DynamicLoraSpec:
    slot_rank: int                    # 叠加层槽位的 rank
    from_flag: bool                   # True：来自 --diffusion-max-lora-rank；False：来自启动适配器
    startup_adapter: str | None       # peft 加 --lora-path 时的适配器路径
```

| 配置 | `dynamic_lora` |
|---|---|
| 给出 `max_lora_rank` | `slot_rank = max_lora_rank`，`from_flag` 为真。字段由 05-2 增加，02-8 用 `getattr(od_config, "max_lora_rank", None)` 读取，05-2 合入之前按没给处理。 |
| 没给 `max_lora_rank`，`lora_backend=peft` 并给了 `--lora-path` | `slot_rank` 取这个适配器的 rank（只读它的 `adapter_config.json` 或 safetensors 头部），再按 `DiffusionLoRAManager._get_smallest_valid_max_rank` 向上取到 `_VALID_MAX_RANKS` 里最小的合法值。 |
| 其他 | `None`，不启用。 |

`max_lora_rank` 的解析和合法值检查归 05。

#### 11.2 包装

- 在 3.5 第 3 步，Loader 对 `lora_target_paths` 给出的组件逐层调用 `from_layer_diffusion`（`vllm_omni/diffusion/lora/utils.py`），`LoRAConfig` 的 `max_lora_rank` 取 `slot_rank`。包装的范围是管理器的 `_compute_supported_lora_modules` 给出的全部可替换线性层，不按单个适配器的 `target_modules` 筛选。包装不分配真实数据：`create_lora_weights` 返回之后，Loader 核对 `lora_a_stacked`、`lora_b_stacked` 都在 meta 上，不在 meta 上时报 `ConstructionError`。
- 包装后的层用 `replace_submodule` 装回原位，包装层的规范路径记在 `catalog.wrapped_layers` 里，管理器由它得到 `_lora_modules`。
- 规划看到包装后的最终结构：包装层整体属于原来那一层所在的 Unit，钩子装在包装层上（总体设计 5.6）。

#### 11.3 名字规则

- 包装后原来的参数路径多一级 `base_layer`。Loader 的通用规则：对 Loader 记录的包装层（动态 LoRA 包装层和 8.5 的 `FixedOverlayLinear`），名字里紧跟在它路径后面的 `base_layer` 段被去掉，得到规范名。目录、checkpoint 映射、内容身份、预期清单和最终清单都使用规范名。
- 写入阶段，Loader 把每个包装层临时换回它的 `base_layer`（同一个对象），pipeline 和模块的 `load_weights` 用 `named_parameters()` 查到的就是包装前的名字；`overlay` 固定适配器的 A/B 由 Loader 按规范名直接写进包装层持有的张量。收尾第 4 步之后、第 6 步之前，Loader 把包装层换回原位。参数对象和存储绑定在这个过程中不变。
- 是否启用动态 LoRA、加载了哪个动态适配器，都不影响内容身份和同机共享。

#### 11.4 槽位

- 每个包装层的每个切片有一对 A/B 槽位，形状由 `slot_rank` 和层的 TP 切分确定，类型是 stage 类型，以 `OverlaySlotDecl` 进入 `catalog.overlay_slots`。
- 规划把槽位放进“其余部分”：常驻加速卡，不进任何 Unit，不经过 HWM，不进内容身份。04 在 `prepare_materialization()` 里分配槽位并计入加速卡准入，在 `commit()` 里调用 `catalog.bind_overlay_slot` 装进包装层，再把槽位交给动态 LoRA 管理器，地址此后不变。

#### 11.5 管理器的运行期

`DiffusionLoRAManager` 增加 `unified` 一支（C16），由 `DiffusionWorker.init_lora_manager` 在 `unified` 下传入目录：

- 构造时不调用 `resolve_offload_strategy`，不设置 `_resident_lora_device`，不对包装层调用 `_set_diffusion_lora_buffer_device`；`_lora_modules` 由 `catalog.wrapped_layers` 得到，`_max_lora_rank` 取 `slot_rank`。槽位张量由 04 在 `commit()` 里绑定，管理器只用 `copy_` 往里写数据，不替换槽位张量对象，槽位地址因此到关闭前不变。
- `_replace_layers_with_lora` 在 `unified` 下不包装，只核对适配器的 `target_modules` 都落在已包装的层里，落在外面的报错。
- `_ensure_max_lora_rank` 在 `unified` 下不重建槽位：适配器的 rank 大于 `slot_rank` 时报错，提示调大 `--diffusion-max-lora-rank`。
- `add_adapter` 把动态适配器读进 CPU 缓存，条数由 `max_cpu_loras` 限定，缓存归管理器，不经过 HWM。激活一个动态适配器就是把它的 A/B 拷进槽位，rank 小于槽位时补零；每次写槽位之后调用 `bump_weight_generation(pipeline)`。
- `dynamic_lora` 为 `None` 时，`unified` 下 worker 不构造管理器，`DiffusionWorker.add_lora` 和带 `lora_request` 的请求报错，提示 `--diffusion-max-lora-rank`。
- `peft` 加 `--lora-path` 的启动适配器在管理器构造时激活，与现有的 `set_active_adapter(init_request, lora_scale)` 相同；这一步发生在提交之后，只拷数据。
- 包装层的前向和其余运行期规则按总体设计 5.6；包装层和管理器都不读 offload 配置。

### 12. load_format

#### 12.1 `diffusers`

`unified` 下 Loader 自己构造 diffusers 的 pipeline 对象，不调用 `DiffusersAdapterPipeline.load_weights()`，也不经过 `build_component`；`legacy` 下 `load_weights()` 整条 `from_pretrained` 的代码不改。

```text
1. 在构造上下文里构造 DiffusersAdapterPipeline(od_config=..., device=执行设备)
   unified 下 _raise_unsupported_features 不拒绝 diffusion_offload_config（C18），其余拒绝照旧
2. 按 legacy 的规则选 pipeline 类：resolve_diffusers_pipeline_class，
   不同于 od_config.diffusers_pipeline_cls 时用它，否则用 DiffusionPipeline 的动态解析
3. DiffusionPipeline.load_config(model) 读 model_index.json
4. 对每个组件 [library, class]：
   PreTrainedModel 或 ModelMixin 的子类：经 HF 组件入口的分发（5.5）from_config 建结构，subfolder 为组件名
   其他（分词器、调度器、特征提取器、纯配置）：照常 from_pretrained
5. pipeline = pipeline_cls(**components, **其余构造参数)；其余构造参数取 update_load_kwargs 给出的参数里
   属于 pipeline 构造函数签名的部分
6. adapter.attach_structure_only_pipeline(pipeline)：
   执行 load_weights() 里不读权重的部分（apply_post_load_updates、__call__ 参数表、VAE slicing 和 tiling、注意力后端），
   不执行 self._pipeline.to(self.device) 和 diffusers 自带的 offload
7. 声明组件：名字取 model_index.json 的键，路径为 "_pipeline.<键>"
```

- 第 7 步的角色：`transformer`、`transformer_2`、`unet`、`prior`、`decoder` 是 `dit`；`text_encoder` 开头的是 `text_encoder`；`image_encoder` 是 `image_encoder`；`vae`、`vqvae`、`movq` 是 `vae`；其余带权重的组件是 `other`。
- diffusers 的 pipeline 对象只负责编排前向和去噪循环，不加载权重，也不调用 diffusers 自带的 offload。
- `od_config.quantization_config` 不为 `None`，或 `diffusers_load_kwargs` 里有 `quantization_config`、`dduf_file`、`device_map` 时，在构造阶段报 `ConstructionError`：`from_config` 不会套用 diffusers 的量化器。
- `attach_structure_only_pipeline` 是 `DiffusersAdapterPipeline` 的新方法；`load_weights` 不改，并登记进指纹清单。

#### 12.2 `custom_pipeline`

`_resolve_custom_pipeline_cls(custom_pipeline_name)` 解析出的类与注册的 pipeline 规则相同：在构造上下文里用 `model_cls(od_config=od_config)` 构造。`unified` 下不做 legacy 的 `model.to(target_device)`。`CustomPipelineWorkerExtension.re_init_pipeline` 触发的重新初始化同样经 01 的驱动完成（01 的 D05）。

### 13. HSDP

- `parallel_config.use_hsdp` 为真时，HSDP 管理的组件是 legacy 选出的最外层 DiT（`PipelineModules.outermost_dits` 的规则），目录里 `hsdp_managed` 为真。它们同样只建结构。
- 物化：7.4 对这些组件的参数在 CPU 上分配，只为 meta 参数分配，不用 `to_empty`，buffer 保持构造时的值（包括按配置算出的 RoPE 表）。它们不经过 HWM，`transaction.target()` 对它们返回 `None` 时也在 CPU 上分配。
- 写入走同一个读取器和写入入口，适配器照常融合；量化层照常逐层量化。
- 收尾第 1 步：对每个 HSDP 组件依次调用 `prepare_fp8_layers_for_fsdp`（有量化配置时）和 `apply_hsdp_to_model(module, hsdp_config, target_device)`，`hsdp_config` 的构造与 `_load_model_with_hsdp` 相同。分片在 `ctx.memory_pools.weights()` 里进行，加速卡上的本地分片因此进入睡眠时备份的池。分片之后，这些张量的绑定快照不再核对，只核对写入完整。
- 最终清单里这些张量的 `residency` 为 `hsdp`；04 让它们的 `fully_resident` 为假。
- `hsdp_weight_load_strategy="pre_sharded"`：规划和在 meta 上的分片沿用 `_load_model_with_pre_sharded_hsdp` 里 `_prepare_pre_sharded_hsdp_load` 到 `finalize_hsdp_root` 的语句；物化不调用 `_materialize_pre_sharded_hsdp_state`，改为经 `nn.Module._apply` 只为 meta 张量在执行设备上分配本地分片，buffer 保持构造时的值，分配在 `ctx.memory_pools.weights()` 里；按 rank 读取切片沿用 legacy 的语句，在 `trap.trusted()` 里执行；不调用 `_finalize_pre_sharded_hsdp_loaded_weights`。这条路径不配 LoRA 和量化（总体设计 8.5，由 05 在参数解析阶段报错），所以不经过适配器融合。

### 14. 共存期：逐张量对比、例外表与指纹清单

#### 14.1 逐张量对比框架

`tests/diffusion/weight_stack/baseline/` 提供 `compare_against_legacy(family, checkpoint, od_overrides)`：

```text
1. 子进程 A：总开关 legacy，DiffusersPipelineLoader.load_model 正常构造加加载，不做 offload
2. 子进程 B：总开关 unified，01 的测试替身 Offloader 规划为全部常驻、HWM 为空，走到收尾结束
3. 两边在同一类设备上运行，各自导出快照：每个张量的名字、形状、类型、步长、sha256，
   持久和非持久 buffer，模块外的数据张量，共用关系，模块树的路径和类名
4. 逐项比较，例外表里的条目按表处理
```

- 结构可以配置的家族用合成的小 checkpoint（`tests/diffusion/weight_stack/baseline/fixtures/` 里的生成脚本），在改动对应家族目录或 `weight_stack/loader/` 的 PR 上运行，L2；结构固定的家族用真实 checkpoint，L4 每晚运行。
- 按空闲显存选择分块大小的 VAE，对比时固定分块。
- 融合适配器的组合逐张量对比融合结果；蒸馏 LoRA 配在线量化、蒸馏 LoRA 配 TP 大于 1 的组合，以“完整加载、融合、再量化”的参考结果为基准。
- 框架同时提供 `export_legacy_hashes(family, checkpoint, od_overrides)`，把子进程 A 的快照存成逐张量的哈希文件，比较一侧可以换成这份文件。删除 `legacy` 之前用它存下每个家族的结果（总体设计 9 第 6 步、9.3），之后对比测试改为与哈希对比。

#### 14.2 对比例外表

`tests/diffusion/weight_stack/baseline/exceptions.py` 里的 `EXCEPTIONS` 是数据：

```python
@dataclass(frozen=True)
class ComparisonException:
    family: str
    tensor_pattern: str           # 张量名或模块路径的通配模式
    field: str                    # 不同的字段：bytes、dtype、stride、presence、module_class
    reason: str                   # 有意不同的原因
    gate_exception: str           # 对应的门槛例外条目（umbrella 上线门槛）
```

有意让 unified 与 legacy 不同的张量逐条写明原因，并列入门槛例外。删除 `legacy` 之前，每一条都必须在门槛例外里有对应条目（总体设计 9 第 4 步）。

#### 14.3 指纹清单

`vllm_omni/diffusion/weight_stack/loader/fingerprints.py` 里的 `LEGACY_READERS` 是数据：

```python
@dataclass(frozen=True)
class LegacyReader:
    qualname: str                 # "vllm_omni.diffusion.models.auk.pipeline_auk._read_dit_weights"
    family: str
    digest: str                   # 源码指纹
    unified_counterpart: str | None   # 对应的 unified 实现；None 时 note 必须写明原因
    note: str
```

- 源码指纹是 `inspect.getsource` 的结果去掉注释、空行和行尾空白后再做 `ast.dump` 的 sha256，只对代码改动敏感。
- `tests/diffusion/weight_stack/loader/test_legacy_fingerprints.py` 对每一条重新计算指纹，与记录的不同时失败，报错按总体设计 9.2 的表给出处理方式：同一个 PR 里改 unified 一支，或在 `note` 里写明 unified 不需要的原因，再更新指纹。
- 清单同时是删除 legacy 时的删除清单（第 9.3 节的删除前保证在“删除 legacy 时”一节）。

### 15. 组件查询接口的使用方迁移

| 使用方 | 现有读取 | 迁移后 |
|---|---|---|
| 编译目标 | `DiffusionModelRunner.load_model` 读 `_dit_modules`，默认 `("transformer", "transformer_2")`；`regionally_compile`（`vllm_omni/diffusion/compile.py`）读 `_repeated_blocks` 和 `_layerwise_offload_blocks_attrs`。 | `unified` 下 runner 不运行这段代码，编译由 04 的 `setup_compile` 在提交后完成，它经 `component_paths(pipeline, "dit", ...)` 和 `repeated_block_paths` 选目标。legacy 不变。 |
| 缓存加速 | `cache/cachedit/backend.py` 的 `_dit_module_names` 读 `_dit_modules`，默认 `("transformer",)`。 | 改为 `component_paths(pipeline, "dit", legacy=<原函数体>)`。 |
| 动态 LoRA | `DiffusionLoRAManager._replace_layers_with_lora` 读默认组件、`_dit_modules`、`_lora_components`。 | 改为 `lora_target_paths(pipeline, legacy=<原表达式>)`；`unified` 下包装在 Loader 里完成（11.2）。 |
| 模型注册表 | `registry.py` 的 `initialize_model` 读 `model.vae` 设置 slicing、tiling 和 VAE 并行。 | `unified` 下对 `component_paths(pipeline, "vae", ...)` 的每个组件设置；这些设置只改属性，在构造上下文里执行。 |
| 序列并行钩子 | `_apply_sequence_parallel_if_enabled` 读 `_dit_modules`，默认 `("transformer", "transformer_2", "dit", "unet")`。 | 改为 `component_paths(pipeline, "dit", legacy=<原表达式>)`；钩子不读写权重，照常在构造上下文里安装。 |
| VAE 快路径 | `_apply_wan_vae_fastpath_if_enabled` 经 `ModuleDiscovery.discover` 取 VAE；`install_wan_vae_fastpath` 在 `channels_last` 级别用 `Module.to(memory_format=...)` 转换卷积权重，撤销（`_restore_module_layouts`）和回滚用 `tensor.data=` 换回原布局。 | 改为 `component_paths(pipeline, "vae", ...)`；`unified` 下布局由 Wan VAE 规则表的 `memory_format` 表达，Loader 按这个步长分配，安装时只绑定前向，不转换布局，也不登记布局的撤销和回滚（C13）。 |
| 模型代码里的组件发现 | `ltx2_components._place_aux_components`、`SanaWMPipeline` 的同类函数经 `ModuleDiscovery.discover` 搬运辅助组件。 | 只服务于 legacy 的放置；`unified` 下其中的 `.to(device)` 在构造上下文里只改类型，不迁移，随 legacy 删除。 |

- `fully_resident` 的使用方（模型自己的 CUDA graph 和编译模式，C12）由 04 迁移，它们经 2.8 的 `fully_resident` 函数读。
- 每个使用方的迁移都有一个测试：`legacy` 下结果与原表达式相同；`unified` 下结果与目录一致。

### 16. 家族迁移

每个家族的 PR 做同样的几件事：读张量的调用点改成 `build_component`，unified 一支放在家族目录的 `unified_build.py`；读张量的代码改成 `declare_weights` 登记的来源加规则表；预先算好的数据张量登记为 `BufferSource`；适配器登记为适配器来源；legacy 一支的读取函数登记进指纹清单；一个以 legacy 为基准的逐张量对比测试。下表列出代码里实际的读取点（文件相对 `vllm_omni/diffusion/models/`）。

| 家族 | 构造期或写入期的读取 | unified 一支 |
|---|---|---|
| Wan2.2-S2V | `wan2_2/pipeline_wan2_2_s2v.py`：`Wan22S2VPipeline._init_original_format` 调用 `_load_wan_t5_as_umt5`（`torch.load` 加 `load_state_dict(assign=True)`）、`DistributedAutoencoderKLWan.from_single_file`、`Wav2Vec2ForCTC.from_pretrained`；`_init_diffusers_format` 直接调用 `UMT5EncoderModel`、`DistributedAutoencoderKLWan`、`Wav2Vec2ForCTC` 的 `from_pretrained`。 | HF 类经 `hf_component`；T5 的 `.pth` 用 `WeightSource(format="torch")` 加 Wan T5 到 UMT5 键名转换的规则表；单文件 VAE 用 `from_config` 加单文件键转换的规则表，表的测试以 `from_single_file` 为参考；`.to(self.device)` 在上下文里只改类型。 |
| AuK | `auk/pipeline_auk.py`：`AuKPipeline.__init__` 调用 `self.dit.load_state_dict(_read_dit_weights(...))`（`safe_open` 加 `get_tensor`，`dit_state_dict` 改键），以及 `self.vae.load_weights(...)`（`auk/auk_vae.py` 的 `AuKVAE.load_weights` 用 `load_file`）。 | `declare_weights(self.dit, sources=..., writer=<dit_state_dict 的键映射>)`；`declare_weights(self.vae, sources=vae.safetensors)`，VAE 的 fp32 是构造出的类型。 |
| Ming-Flash-Omni | `ming_flash_omni/pipeline_ming_imagegen.py`：`MingImagePipeline.__init__` 调用 `MingConditionEncoder.load_from_checkpoint`（`Qwen2ForCausalLM.from_pretrained`，`_load_optional_mlp_weights` 用 `load_file` 或 `torch.load`）和 `MingByT5Encoder.from_checkpoint`（`T5ForConditionalGeneration.from_pretrained(...).get_encoder()`，再依次 `torch.load` `base.pt`、`byt5_model.pt` 经 `load_state_dict` 写进编码器，`byt5_mapper.pt` 经 mapper 的 `load_weights` 写入）。 | Qwen2、T5 经 `hf_component`；ByT5 编码器（`get_encoder()` 取出的子模块）用一次 `declare_weights` 登记三条有序的来源：T5 目录（键名按 T5 的规则表映射到编码器子模块）、`base.pt`（`nested_key="state_dict"`，`include=("module.text_tower.encoder.*",)`，`strip_prefix="module.text_tower.encoder."`）、`byt5_model.pt`，torch 格式都是 `weights_only=False`，与 legacy 相同，后面的覆盖前面的同名键；MLP 和 mapper 各用一次 `declare_weights`，mapper 由它自己的 `load_weights` 写入。 |
| Ming-Image | `ming_image/condition.py`：`MingImageConditioning.__init__` 调用 `query_encoder.load_from_checkpoint` 和 `_load_direct_weights`（`load_file` 加改键后的 `direct_projector.load_state_dict`）。 | 两处都改成 `declare_weights`，改键写进 `writer`。 |
| π0 | `pi0/pipeline_pi0.py`：`Pi0Pipeline._initialize_model` 调用 `_load_checkpoint`（`safetensors.torch.load_file`）。 | `declare_weights(model, sources=..., writer=<_load_checkpoint 的键映射>)`。 |
| π0.5 | `pi05/pipeline_pi05.py`：`Pi05Pipeline._initialize_model` 调用 `_load_checkpoint`（`load_file`）；`pi05/config.py`：`Pi05Config.from_pretrained` 调用 `load_lerobot_norm_stats`（`load_file`）；`_set_inference_dtype` 在加载后改类型。 | 模型同 π0；归一化统计量登记为 pipeline 上的非持久 buffer 来源，处理器在 `eval()` 里由它们建立 `NormStats`，这些 buffer 列入对比例外表；`_set_inference_dtype` 由规则表的类型取代。 |
| InternVLA-A1 | `internvla_a1/pipeline_internvla_a1.py`：`InternVLAA1Pipeline._initialize_policy` 调用 `internvla_a1/model_internvla_a1.py` 里自定义的 `InternVLAA1Policy.from_pretrained`（`safe_open` 加 `get_tensor` 交给带 `ignore_unexpected_prefixes` 的 `AutoWeightsLoader`）；`internvla_a1/model_cosmos.py` 的 `_load_native_cosmos_component`（`load_file` 加 `load_state_dict(strict=False)`）；`InternVLAA1Policy.to` 的覆写把 `model.cosmos` 改成 bf16、`model.action_out_proj` 改成 fp32。 | policy 只用配置构造，`declare_weights(policy, sources=model.safetensors, writer=<同样参数的 AutoWeightsLoader>)`；Cosmos 组件用 `declare_weights`；`.to` 覆写的类型效果由规则表的 `runtime_dtype` 表达。 |
| Gr00t | `gr00t/policy.py`：`Gr00tPolicy.__init__` 调用 `AutoModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16)`。 | 改成 `hf_component`，需要 `Gr00tN1d7` 的规则表；组件路径 `policy.model`（2.3）。 |
| OmniVoice | `omnivoice/pipeline_omnivoice.py`：`OmniVoicePipeline.__init__` 调用 `HiggsAudioV2TokenizerModel.from_pretrained`；`load_weights` 丢弃流，调用 `self.generator.load_weights(self.model_path, device)` 和 `self.decoder.load_weights(...)`（`vllm_omni/model_executor/models/omnivoice/` 里自己读文件）。 | 分词器模型经 `hf_component`；generator、decoder 用 `declare_weights` 登记模型目录的来源，键映射（`llm.*`、`audio_tokenizer.*`）写进 `writer`；`weights_sources` 的 unified 一支为空。`load_weights(model_dir, device)` 还被自回归路径使用，保留。 |
| LTX2 | `ltx2/ltx2_components.py`：`_load_ltx_latent_upsampler_single_file`（`safe_open` 读元数据，`load_file` 加 `load_state_dict`）、`_load_ltx25_native_diffusion_decoder`（经 `ltx2/vae/decoder.py` 的 `load_ltx25_native_diffusion_decoder_state_dict` 用 `get_tensor`）；`ltx2/ltx2_runtime.py` 的 `eval()` 调用 `phase_adapter.finalize()`，经 `ltx2_adapter_parser.iter_adapter_tensors` 读 A/B。 | upsampler、diffusion decoder 用 `declare_weights`；相位适配器声明为 `overlay` 固定适配器（`format="ltx_phase"`），A/B 在写入阶段写入，`eval()` 里的物化只在 legacy 一支执行；只在对应 profile 下声明 `diffusion_decoder`、`latent_upsampler`。 |
| SeedVR2 | `seedvr2/pipeline_seedvr2.py`：`SeedVR2Pipeline.__init__` 用 `torch.load` 读 `pos_emb.pt`。 | `pos_emb` 注册为 meta 非持久 buffer，形状来自配置，用 `BufferSource(format="torch")` 登记。 |
| MAGI-2 | `magi2/turbo_vae.py`：`Magi2TurboVAEDecoder.__init__` 用 `torch.load` 加 `load_state_dict`；`magi2/audio_decoder.py`：`Magi2AudioDecoder.__init__` 用 `safe_open` 加 `get_tensor`；`magi2/conditioning.py`：`Magi2Qwen35TextEncoder.__init__` 调用 `Qwen3_5TextModel.from_pretrained`；图像 VAE 用 Lance 的 `LanceWanVAE`。 | 两个解码器用 `declare_weights`；文本编码器经 `hf_component`；MoE 的 `W13` 见 C02（第 17 节）；按 rank 声明见 2.5。 |
| DreamZero | `dreamzero/pipeline_dreamzero.py`：`DreamZeroPipeline.__init__` 直接调用 `DistributedAutoencoderKLWan.from_pretrained`。 | 改成 `hf_component`，`init_distributed` 写进规则表的 `post_construct`；VAE 类型随 offload 变化的现象由对比基准处理（总体设计 9 第 3 步）。 |
| Anima | `anima/pipeline_anima.py`：`AnimaPipeline._load_native_denoiser_components` 用 `load_file` 读一个文件，拆给 `native_text_conditioner` 和 `native_transformer` 的 `load_state_dict`；`_load_outer_components` 直接调用 `Qwen3Model.from_pretrained` 和 `DistributedAutoencoderKLQwenImage.from_pretrained`。 | 两个原生组件各一条 `WeightSource`（同一文件，`include` 按前缀拆分）；外部组件经 `hf_component`。 |
| Lance | `lance/wan_vae.py`：`LanceWanVAE.__init__` 调用 `_ensure_built`（`torch.load` 加 `load_state_dict(strict=False)`）。 | VAE 只用配置构造，`declare_weights` 登记 `.pth`；`LanceWanVAE` 的读取同时被 Lance 和 MAGI-2 使用，两个家族都只剩 unified 一支后才删。 |
| MiniMax H3 | `minimax_h3/vae.py` 的 `_load_selected_state_dict`（`get_tensor` 加 `load_state_dict`，编码半和解码半在构造函数里加载）；`minimax_h3/latent_upscaler.py` 的 `load_minimax_h3_latent_upscaler`（`_read_state_dict` 用 `get_tensor` 或 `torch.load`）；`minimax_h3/adaln_cache.py` 的 `MiniMaxH3AdalnCache.__init__`（`get_tensor` 读 sidecar）；`minimax_h3/fasth3.py` 的 `FastH3WeightFusion.from_path` 和 `minimax_h3/vdnh3.py` 的 `VDNCheckpoint.__init__`（`get_tensor`）；pipeline 的 `load_weights` 替 `video_vae`、`audio_vae`、`latent_upscaler` 声明已加载。 | 视频 VAE 两半、音频 VAE、放大器用 `declare_weights`，`include` 按前缀选键；视频 VAE 两半各是一个 `vae` 组件；sidecar 见 C04；FastH3、VDN 见 8.3；组件声明带 `repeated_blocks`、`unit_groups={"adaln": ...}`、文本编码器的 `shard_group`；`weights_sources` 的 unified 一支不含 VAE 和放大器。 |
| Cosmos3 声音 tokenizer | `cosmos3/audio_tokenizer/avae.py`：构造函数调用模块级的 `_load_checkpoint`（`load_file` 或 `torch.load`）加 `load_state_dict(strict=False)`。 | `declare_weights` 登记 checkpoint，格式按文件后缀选择。 |
| NextStep-1.1 | `nextstep_1_1/pipeline_nextstep_1_1.py`：`NextStep11Pipeline.__init__` 调用 `nextstep_1_1/modeling_flux_vae.py` 里自定义的 `AutoencoderKL.from_pretrained`（`torch.load` 加 `load_state_dict(strict=False)`）。 | 用配置构造 `AutoencoderKL`，`declare_weights` 登记 checkpoint；自定义 `from_pretrained` 里的键处理写进 `writer`。 |
| LongCat-Video-Avatar | `longcat_video/pipeline_longcat_video_avatar.py`：`_load_components` 直接调用 `UMT5EncoderModel`、`DistributedAutoencoderKLWan`、`WhisperModel` 的 `from_pretrained`，并调用 `transformer.load_lora(...)`（`longcat_video/longcat_video_avatar_transformer.py` 的 `load_file` 加 `create_lora_network`），再 `enable_loras(["dmd"])`。 | HF 类经 `hf_component`；DMD LoRA 声明为 `overlay` 固定适配器（`format="longcat_dmd"`），A/B 在写入阶段写入，运行期由模型开关。 |

构造函数里直接调用 `X.from_pretrained` 加载带权重组件、但不在上表里的 pipeline，在 `unified` 下触发读取陷阱（第 4 节表里 Hugging Face `from_pretrained` 的直接调用），由 02-18 改成 `hf_component`（umbrella 2a），`legacy` 下与原调用完全相同。主干里的这类调用点是：`flux/pipeline_flux.py`、`flux/pipeline_flux_kontext.py`、`flux2/pipeline_flux2.py`、`ernie_image/pipeline_ernie_image.py`、`helios/pipeline_helios.py`、`glm_image/pipeline_glm_image.py`、`hidream_image/pipeline_hidream_image.py`、`lingbot_video/pipeline_lingbot_video.py`、`cosmos3/pipeline_cosmos3.py` 的 VAE；完整清单以支持矩阵测试（5.7）的结果为准。用到的类在 02-3 或 02-18 里有规则表。

LingBot-Video 的 `LingBotVideoTransformer3DModel.to` 覆写在改类型时让 `LINGBOT_VIDEO_FP32_MODULES` 匹配的参数保持 fp32（类上的 `_keep_in_fp32_modules` 取同一个列表），`LingBotVideoPipeline.to` 覆写在整体改类型之后再调用一次 transformer 自己的 `.to`。`unified` 下这一效果由 `LingBotVideoTransformer3DModel` 规则表的 `keep_fp32` 条目表达，条目的取值方式由逐表测试对 `from_pretrained` 的结果确定，LingBot-Video 另有一个以 legacy 为基准的逐张量对比测试（总体设计 9.3）；两个 `.to` 覆写不改，`legacy` 下行为不变，删除 `legacy` 时删除。

### 17. 本 RFC 承担的共存期必改行

每一行只改变 `unified` 下的行为，`legacy` 下的行为和已有测试不变（01 的 9.3）。承担一行的 PR 按 01 的 9.4 新增位置文件 `tests/diffusion/weight_stack/coexistence/sites/<行编号>_<PR 编号>.py`（例如 `C16_02_7.py`），在 `SITES` 里写出它新增的读总开关或构造上下文的位置，不修改 `registry.py`；一行拆给两个 PR 时各自新增一个位置文件。本 RFC 自己的分发点 D06 在 02-2 的位置文件 `D06_02_2.py` 里用 `NEW_ROW` 声明。模型代码只经 `is_unified_stack()` 或 `is_building_structure()` 判断。

| 编号 | 位置 | `unified` 一支 |
|---|---|---|
| C02 | `magi2/mh_moe.py` 的 `Magi2MultiHeadMoE` | `is_building_structure()` 为真时只注册合并的参数 `W13`，`weight_loader` 按 shard id 写进交错的两半；上下文外保留 `W_gate`、`W_up`、`_owned_w13` 和 `_apply` 覆写。 |
| C04 | MiniMax H3 的 AdaLN sidecar（`MiniMaxH3Pipeline._configure_adaln_sidecar`、`_verify_adaln_weights`、`_finish_adaln_sidecar`） | sidecar 的张量注册为 meta 非持久 buffer，由 `BufferSource` 写入；`eval()` 只用写好的 buffer 给 AdaLN 缓存播种，设备用 `execution_device()`，校验用 Loader 的内容身份。 |
| C05 | `minimax_h3/vae.py` 里按 `next(self.parameters())` 取设备和类型、类型不是 fp32 时先 `self.to(torch.float32)`、结束后 `self.to(previous_dtype)` 的两处 | 规则表的 `runtime_dtype` 给出这个 VAE 运行时的类型，`next(self.parameters())` 读到的类型是 fp32，`unified` 下往返的条件不成立；设备取 `execution_device()`。 |
| C09 | `sana_wm/pipeline_sana_wm.py` 的 `_ensure_stage1_text_encoder`、`_ensure_vae` | 不做 `.to(device, dtype)`；VAE 和第一阶段文本编码器在构造函数的 unified 一支里经 HF 组件入口建出，请求期不再 `from_pretrained`。 |
| C10 | `hunyuan_image3/pipeline_hunyuan_image3.py` 的 `HunyuanImage3Pipeline.load_weights` 在调用 `AutoWeightsLoader` 之前对零散模块的 `.to(device)` | 只在 `legacy` 下执行。 |
| C11 | `kandinsky6/pipeline_kandinsky6.py` 构造函数里由第一个参数的设备决定的 `self.device` | `unified` 下取 `execution_device()`。 |
| C13 | `registry.py` 的 `_apply_wan_vae_fastpath_if_enabled`；`vllm_omni/diffusion/distributed/autoencoders/wan_vae_fastpath/install.py` 的 `install_wan_vae_fastpath` 对卷积权重的布局转换和 `tensor.data=` 的撤销、回滚 | 布局由规则表表达，Loader 按步长分配；安装时只绑定前向，不转换布局，不登记布局的撤销和回滚（第 15 节）。 |
| C15 | `ltx2/ltx2_runtime.py` 的 `_run_ltx_vocoder` 对带 `bwe_generator` 的 vocoder 在 mps、xpu、cpu 上的 `vocoder.float()` 往返 | 规则表在这些平台上让 vocoder 保持 fp32，取值为 `stage_rounded`（与往返得到的字节相同），往返的条件不成立。 |
| C16 | `DiffusionLoRAManager` 的包装和 `_ensure_max_lora_rank`；`DiffusionWorker.init_lora_manager` 的 distill 融合；`DiffusionWorker.add_lora` | 融合挪到读取器（8.4，02-7）；包装挪到只建结构，管理器和 `add_lora` 的 `unified` 一支见 11.5（02-8）。 |
| C18 | `DiffusersAdapterPipeline._raise_unsupported_features` 对 `diffusion_offload_config` 的拒绝 | `unified` 下不执行这项拒绝，其余拒绝不变（12.1）。 |
| D06 | `models/utils.py` 的 `create_transformers_model`、`create_transformers_model_with_vllm_linears` | `is_building_structure()` 为真时跳过 `init_parameters`、`init_parameters_preserving_vllm_linear`（3.4）。 |

## 与其他子 RFC 的接口

### 本 RFC 提供

| 提供的内容 | 使用方 | 约定 |
|---|---|---|
| `weight_stack.loader.create_loader(ctx)`、`Preflight`、`ExpectedManifest`、`FinalManifest` | 01 | 签名见设计第 1 节和第 10 节；`Preflight.content_identities` 是组件名到摘要的映射，直接填进 `RankFingerprints.content_identities`。 |
| `Loader.write()` 的绑定规则 | 03、04 | 每个张量调用一次 `transaction.target(name)`，名字是预期清单的规范名；返回张量的形状、类型、步长必须与清单一致；返回 meta 张量表示写入是空操作。每个 Host 驻留张量全部写完时调用一次 `storage.note_written(name)`（7.5）。 |
| `Loader.finalize()` | 01、03 | 第 6 步之前完成收尾里的全部集合通信，第 6 步调用一次 `storage.seal(final_manifest)`；`final_manifest.live()` 给出要重新绑定的模型张量。 |
| `ComponentCatalog` 及其方法 | 04 | `plan()` 第一步调用 `require_complete()`，组件不完整或保持 fp32 的层配了量化时报 `PlanningError`；`plan()` 经 `set_discovered_blocks()` 记录自动发现的重复块；`commit()` 调用 `set_fully_resident()` 和 `bind_overlay_slot()`；`write_transient` 计入加速卡准入；`shapes_final`、`finalize_replaced` 用于冷启动路径和共享条件。 |
| `ExpectedManifest` | 03 | `layout(plan, expected_manifest)` 的输入；`digest(names)` 是交接时核对的预期清单摘要。 |
| `weight_stack.components` 的查询函数、`execution_device()`、`is_unified_stack()`、`weight_generation()`、`bump_weight_generation()` | 04、模型代码、`cache/`、`lora/`、`registry.py` | 第 2.8 节。`is_unified_stack()` 是模型代码读总开关的唯一入口（01 的 T6）。 |
| `model_loader.construct`：`build_component`、`declare_weights`、`hf_component`、`is_building_structure`、`WeightSource`、`BufferSource` | 模型代码 | 第 5 节；`declare_weights` 在构造阶段之外调用、`unified` 下 `build_component` 和 HF 组件入口（带权重的类）在构造阶段之外调用，都报 `LifecycleError`。 |
| `DynamicLoraSpec` 和包装层记录 | 04、`lora/manager.py` | 槽位形状来自 `overlay_slots`；管理器从目录取包装层。 |

### 本 RFC 依赖的约定

| 子 RFC | 约定 |
|---|---|
| [01](./01-startup-transaction.md) | `StartupContext`（含 `memory_pools`）、`Loader` 协议、错误类型；总开关取自 `weight_stack.startup` 的 `WeightStack` 和 `process_weight_stack()`；驱动不进入任何睡眠池，常驻加速卡的张量（包括收尾里新建、替换的张量和 HSDP 的本地分片）由 Loader 自己在 `ctx.memory_pools.weights()` 里分配；驱动在第 10 步成功之后立即调用 `loader.close()`；必改行和 D06 的位置文件放在 `tests/diffusion/weight_stack/coexistence/sites/` 下，D06 用 `NEW_ROW` 声明，不修改 `registry.py`；T6 不检查 `weight_stack/`、`model_loader/construct.py` 和 `model_loader/hub_prefetch.py` 内部；测试替身（全部常驻的规划、空的 host 存储）用于对比框架。单进程执行器在同一进程里依次进行各 worker 的 `unified` 启动（3.1）。 |
| [03](./03-host-storage.md) | `HostStorage.view(name)` 对瘦路径下本 rank 范围之外的张量、以及等待者的共享张量返回形状和类型正确的 meta 张量；`seal()` 在每个 worker、每种后端上都被调用一次：私有存储和分片在其中定稿，共享存储在其中封存、交接并用 `final_manifest.live()` 重新绑定；`layout()` 只读 `ExpectedManifest` 的条目。 |
| [04](./04-unified-offloader.md) | `plan()` 第一步调用 `catalog.require_complete()`，并经 `set_discovered_blocks()` 记录自动发现的重复块；`PlacementPlan` 给出每个组件的冷启动路径（供预检核对）和分片后端 Unit 的张量规范名（供收尾第 7 步换成占位）；`transaction.target()` 的语义同 01；`prepare_materialization()` 分配叠加层槽位，`commit()` 调用 `bind_overlay_slot()` 并把槽位交给动态 LoRA 管理器；`repeated_block_paths` 返回 `set_discovered_blocks()` 记录的结果；`setup_compile` 经查询接口选目标。 |
| [05](./05-configuration.md) | 解析 `--diffusion-max-lora-rank` 并检查合法值；HSDP 预分片配 LoRA 或量化在参数解析阶段报错；支持矩阵文档收录 5.7 的结果。 |

## 代码组织

```text
vllm_omni/diffusion/weight_stack/
  components.py            ComponentDecl、ComponentCatalog、TensorDecl、OverlaySlotDecl、查询函数、目录的弱引用表
  loader/
    __init__.py            create_loader、Preflight、ExpectedManifest、FinalManifest、ContentIdentity
    loader.py              UnifiedLoader：build_structure、preflight、write、finalize、close
    context.py             StructureContext、register_parameter 和 Module.to 的替换
    trap.py                ReadTrap、代理对象、违例记录、trusted()
    catalog.py             声明推导、路径解析、划出、去重、完整性、目录构建
    rules/                 RuleTable 及每个类一张表、generate 工具
    reader.py              ResolvedSource、读取流水线、LazyTensor
    writer.py              存储绑定、写入入口选择、WriteTracker
    quant.py               quant_output_specs、逐层量化
    adapters/              AdapterFormat 注册表；diffusers、kohya、fasth3、vdn、ltx_phase、longcat_dmd 转换器
    overlay.py             动态 LoRA 包装、FixedOverlayLinear、base_layer 名字规则
    finalize.py            收尾第 1 到第 7 步、第 2 步工作的注册表
    identity.py            ContentIdentity、ExpectedManifest、FinalManifest
    diffusers_format.py    load_format="diffusers"
    hsdp.py                HSDP 组件的物化和分片
    fingerprints.py        LEGACY_READERS
vllm_omni/diffusion/model_loader/construct.py   build_component、declare_weights、hf_component、is_building_structure、来源类型
vllm_omni/diffusion/models/<family>/unified_build.py   每个迁移家族的 unified 一支

tests/diffusion/weight_stack/loader/
  test_catalog.py  test_context.py  test_trap.py  test_construct_api.py  test_hf_entry.py
  test_reader.py  test_lazy_tensor.py  test_write_tracking.py  test_online_quant.py
  test_adapters.py  test_finalize.py  test_content_identity.py  test_manifests.py
  test_overlay.py  test_diffusers_format.py  test_hsdp.py  test_query_api.py
  test_structure_matrix.py  test_legacy_fingerprints.py
  rules/test_rule_tables.py  rules/configs/
tests/diffusion/weight_stack/baseline/
  compare.py  exceptions.py  fixtures/  test_<family>_baseline.py
```

修改的已有文件：

| 文件 | 改动 |
|---|---|
| `vllm_omni/diffusion/model_loader/hub_prefetch.py` | `from_pretrained_with_prefetch` 在构造上下文里进入分发。 |
| `vllm_omni/diffusion/models/utils.py` | D06。 |
| `vllm_omni/diffusion/lora/manager.py` | C16 的 `unified` 一支；`_replace_layers_with_lora` 的组件选择改用 `lora_target_paths`。 |
| `vllm_omni/diffusion/worker/diffusion_worker.py` | `init_lora_manager` 的 `unified` 一支（C16）。 |
| `vllm_omni/diffusion/cache/cachedit/backend.py`、`vllm_omni/diffusion/registry.py` | 第 15 节的查询接口迁移；C13。 |
| `vllm_omni/diffusion/models/diffusers_adapter/pipeline_diffusers_adapter.py` | C18；新方法 `attach_structure_only_pipeline`。 |
| 19 个家族目录、`magi2/mh_moe.py`、`sana_wm/`、`hunyuan_image3/`、`kandinsky6/`、`ltx2/ltx2_runtime.py` | 第 16、17 节。 |
| 第 16 节列出的直接调用 `X.from_pretrained` 的 pipeline（`flux/`、`flux2/`、`ernie_image/`、`helios/`、`glm_image/`、`hidream_image/`、`lingbot_video/`、`cosmos3/`） | 调用点改成 `hf_component`；LingBot-Video 的 `.to` 覆写不改，由规则表表达。 |

## 实现拆分

本 RFC 的 PR 都依赖 01-1 到 01-7 全部合入，下表不再重复。测试里的 HWM 和 Offloader 用 01 的测试替身，本 RFC 的 PR 不依赖 03、04 的 PR。

| PR | 组 | 内容 | 主要文件 | 测试 | 依赖 | 大小 | 必改行 |
|---|---|---|---|---|---|---|---|
| 02-1 | 2a | 组件声明的类型、`declare_components`、默认推导、路径解析和划出、去重、完整性违例、按 rank 声明、带类型字段的校验、目录构建、进程级弱引用表和 `pipeline_of` | `components.py`、`loader/catalog.py` | `test_catalog.py` | 无 | M | 无 |
| 02-2 | 2a | 构造上下文、读取陷阱、构造后检查、最终类型、`build_component`、`declare_weights`、`hf_component`、`from_pretrained_with_prefetch` 的分发、`load_format="default"` 和 `custom_pipeline` 的 `build_structure()`、D06、支持矩阵测试 | `loader/context.py`、`loader/trap.py`、`loader/loader.py`、`model_loader/construct.py`、`hub_prefetch.py`、`models/utils.py` | `test_context.py`、`test_trap.py`、`test_construct_api.py`、`test_hf_entry.py`、`test_structure_matrix.py` | 02-1 | L | D06 |
| 02-3 | 2b | 规则表的格式和执行、`generate` 工具、逐表测试框架，以及主干 pipeline 用到的 Hugging Face 类的第一批表 | `loader/rules/` | `rules/test_rule_tables.py` | 02-2 | L | 无 |
| 02-4 | 2c | 来源解析、读取流水线、延迟张量、预检的来源核对、内容身份、预期清单 | `loader/reader.py`、`loader/identity.py` | `test_reader.py`、`test_lazy_tensor.py`、`test_content_identity.py` | 02-2 | L | 无 |
| 02-5 | 2c | 存储绑定、写入入口选择、写入跟踪、逐层在线量化、收尾顺序、最终清单、HSDP 组件的物化和分片 | `loader/writer.py`、`loader/quant.py`、`loader/finalize.py`、`loader/hsdp.py` | `test_write_tracking.py`、`test_online_quant.py`、`test_finalize.py`、`test_manifests.py`、`test_hsdp.py` | 02-3、02-4 | L | 无 |
| 02-6 | 2b | 以 legacy 为基准的逐张量对比框架和 `export_legacy_hashes`、对比例外表、指纹清单和指纹测试 | `tests/diffusion/weight_stack/baseline/`、`loader/fingerprints.py` | 框架自测（用一个已按新写法构造的 pipeline，例如 Z-Image） | 02-5 | M | 无 |
| 02-7 | 2c | 适配器格式转换器（diffusers、kohya、fasth3、vdn）、读取器里的融合、预检的键核对、`fuse` 固定适配器、worker 的 distill 分支 | `loader/adapters/`、`diffusion_worker.py` | `test_adapters.py`；融合后在线量化在 TP 为 1 和 2 时与参考一致 | 02-5 | L | C16（融合） |
| 02-8 | 2f | 动态 LoRA 包装、`base_layer` 名字规则、槽位声明、`overlay` 固定适配器（ltx_phase、longcat_dmd 转换器）、管理器的 `unified` 一支 | `loader/overlay.py`、`lora/manager.py`、`diffusion_worker.py` | `test_overlay.py`；加载、切换、卸载动态 LoRA 前后结构和基础权重地址不变 | 02-5 | M | C16（包装） |
| 02-9 | 2f | `load_format="diffusers"` 的 `unified` 一支、`attach_structure_only_pipeline`、C18 | `loader/diffusers_format.py`、`pipeline_diffusers_adapter.py` | `test_diffusers_format.py`；SDXL fp16 的 `upcast_vae` 由规则表固定类型 | 02-3、02-5 | M | C18 |
| 02-10 | 2e | 查询接口和第 15 节的使用方迁移、Wan VAE 的内存格式规则表 | `components.py`、`cache/cachedit/backend.py`、`registry.py`、`distributed/autoencoders/wan_vae_fastpath/install.py`、`lora/manager.py`、`loader/rules/` | `test_query_api.py` | 02-2、02-3 | M | C13 |
| 02-11 | 2d | MiniMax H3 | `minimax_h3/` | 对比测试（合成 checkpoint，L2；真实 checkpoint，L4） | 02-6、02-7 | L | C04、C05 |
| 02-12 | 2d | MAGI-2、Lance | `magi2/`、`lance/` | 对比测试 | 02-6 | L | C02 |
| 02-13 | 2d | LTX2 | `ltx2/` | 对比测试；相位适配器开关前后的输出与 legacy 一致 | 02-6、02-8 | M | C15 |
| 02-14 | 2d | Wan2.2-S2V、DreamZero、LongCat-Video-Avatar | `wan2_2/`、`dreamzero/`、`longcat_video/` | 对比测试 | 02-6、02-8 | M | 无 |
| 02-15 | 2d | Ming-Flash-Omni、Ming-Image、Anima、NextStep-1.1、Cosmos3 声音 tokenizer 和 Cosmos3 的显式组件声明、SeedVR2、AuK、OmniVoice | 对应家族目录 | 对比测试 | 02-6 | L | 无 |
| 02-16 | 2d | π0、π0.5、InternVLA-A1、Gr00t | 对应家族目录 | 对比测试 | 02-6 | M | 无 |
| 02-17 | 2a | SANA-WM、HunyuanImage3、Kandinsky6 的必改行；SANA-WM 构造函数里的 HF 组件改用 `hf_component` | `sana_wm/`、`hunyuan_image3/`、`kandinsky6/` | 每行一个 `unified` 用例；`legacy` 下已有测试不变 | 02-3 | S | C09、C10、C11 |
| 02-18 | 2a | 第 16 节列出的直接调用 `X.from_pretrained` 的 pipeline 改用 `hf_component`；所需类的规则表，包括 LingBot-Video 的 `keep_fp32` 条目 | 第 16 节列出的 pipeline 文件、`loader/rules/` | 支持矩阵里这些 pipeline 构造成功；新增规则表的逐表测试；LingBot-Video 的逐张量对比测试 | 02-3、02-6 | M | 无 |

- 每个 PR 只改变 `unified` 下的行为，`legacy` 下的行为和已有测试不变，不修改已有测试文件；承担必改行的 PR 新增自己的位置文件（第 17 节），PR 描述写明行编号和这一行的 `legacy_removal`。
- 02-11 到 02-16、02-18 可以并行；它们合入之前，对应的 pipeline 在 `unified` 下启动报 `ConstructionError`。
- 本 RFC 的 PR 之间的依赖如下，都在 01 的全部 PR 之上，与 03、04 的 PR 互不依赖（umbrella 的依赖图）：

```text
01-1..01-7
  02-1
    02-2
      02-3
        02-10、02-17          依赖 02-3
      02-4
        02-5                  依赖 02-3、02-4
          02-6、02-7、02-8
          02-9                依赖 02-3、02-5
            02-11             依赖 02-6、02-7
            02-12、02-15、02-16  依赖 02-6
            02-13、02-14      依赖 02-6、02-8
            02-18             依赖 02-3、02-6
```

## 测试计划

除特别写明的以外，测试放在 `tests/diffusion/weight_stack/loader/`，是 L1，标记 `core_model`、`diffusion`、`cpu`，不需要加速卡。

**组件目录**（`test_catalog.py`）

- 默认推导对 `SupportsComponentDiscovery` 和回退属性各一个用例，角色按 2.2 的表；显式声明取代推导。
- 路径列表、嵌套划出（Cosmos3 式的 `transformer.language_model`）、别名模块去重、共用参数去重、非模块对象上的路径（Gr00t 式）。
- 完整性：缺归属和多归属各报出全部路径；保持 fp32 的层配了量化方法时记进 `dtype_violations`；两类违例都只在 `require_complete()` 里报 `PlanningError`，构造阶段不报错。
- 目录不挂在 pipeline 对象上；pipeline 被回收后弱引用表里的条目消失；`set_discovered_blocks`、`set_fully_resident`、`bind_overlay_slot` 重复调用报 `LifecycleError`。
- 带类型字段：每个字段的每条校验规则一个反例。

**构造上下文和读取陷阱**（`test_context.py`、`test_trap.py`）

- 在 `torch.device("cuda")` 和 `set_default_torch_dtype` 里构造，参数仍落到 meta，子类和 `weight_loader` 保留；buffer 在 CPU。
- `.to("cuda", torch.float16)` 只改类型；`.to(memory_format=torch.channels_last)` 改步长；`to_empty` 报错。
- 陷阱：第 4 节表里每个被拦截的调用在两个阶段各一个用例；头部读取放行；`from safetensors import safe_open` 绑进模型模块和 `vllm`、`diffusers` 模块命名空间的名字同样被拦截；上下文里新 import 的模块同样被拦截；Loader 读取器的线程池不被拦截；上下文关闭后所有名字恢复为原对象。
- 同一进程里第二个上下文在第一个关闭之前打开时报 `LifecycleError`；动态 LoRA 槽位和 `overlay` A/B 的 meta 张量不被构造后检查报出。
- 模型代码用 `try/except` 吞掉陷阱的异常，`build_structure()` 照样报错，信息里有调用位置。
- 构造期用权重数值算出的普通属性（meta 张量）被构造后检查报出。

**构造 API**（`test_construct_api.py`、`test_hf_entry.py`）

- `build_component` 在三种状态下的行为（5.2 的表）；`declare_weights` 在上下文外、重复登记、嵌套登记时的行为。
- HF 入口：`from_config` 只读 JSON；`config.json` 带 `quantization_config` 报错；没有规则表报错；`DistributedAutoencoderKLWan` 的 `init_distributed` 在上下文里执行；上下文外与原调用完全相同，`from_pretrained_with_prefetch` 的重试行为不变。

**读取、写入和收尾**（`test_reader.py`、`test_lazy_tensor.py`、`test_write_tracking.py`、`test_online_quant.py`、`test_finalize.py`）

- 来源解析与 `DiffusersPipelineLoader._prepare_weights` 的结果相同；有序来源只产出最后一个来源的张量。
- 延迟张量：视图操作不读数据；写进 meta 目标不读数据；`narrow` 之后物化只读对应的切片（统计读取字节数）。
- 写入跟踪：重复写入报 `WriteError`；`writer`、`AutoWeightsLoader`、`load_state_dict` 的写入都被计数；包装后的 `weight_loader` 与原函数对同一输入写出相同的字节，没有 `weight_loader` 的参数走 `default_weight_loader`；改绑参数（`param.data = ...`、替换 `Parameter` 对象）在收尾第 4 步被报出。
- 预检：名字映射由 Loader 决定的来源缺少 `required` 张量时报 `PreflightError`；经模型 `load_weights` 写入的来源不做这项核对，缺写的张量在收尾第 4 步报出。
- 逐层量化：用一个 fp8 在线量化的小模型，量化在每层写完时立即发生，量化前的临时张量随即释放，最终字节与 legacy 的 `_process_weights_after_loading` 一致；输出声明与实际不符时报错；不经过 `weight_loader` 写量化层量化前的参数时报 `WriteError`；输出目标全是 meta 的层不读 checkpoint，层上的运行期属性与目标为真实存储时相同。
- 收尾：第 2 步的注册表为空时跳过；第 2 步替换的张量的别名在第 3 步改指向新张量；`tied` 核对；常驻加速卡的新张量在 `weights()` 池里分配（用记录进入次数的替身 `MemoryPools`）；`eval()` 读分片后端 Unit 的参数时报错；`eval()` 改了参数时按第 6 步之后的快照报 `FinalizeError`，`seal()` 的只读重绑定本身不报错；不调用 `post_load_weights` 等钩子（用替身模型记录调用）；读取陷阱在收尾结束时最后核对一次违例。

**内容身份和清单**（`test_content_identity.py`、`test_manifests.py`）

- 10.1 里每个包含项改变时摘要改变，每个不包含项改变时摘要不变。
- 预期清单与最终清单的条目一致；写入进程和等待者的共享部分摘要相同；`live()` 返回的对象就是模型参数。

**适配器和叠加层**（`test_adapters.py`、`test_overlay.py`）

- 每个格式转换器：键映射、Δ 的数值规则（diffusers、kohya、FastH3 的 fp32 相加）、`combine` 与 legacy 的 `load_lora_weights` 逐位一致。
- 预检 8.6 表里的每一行。
- 融合后在线量化：TP 为 1 时在 CPU 上比较；TP 为 2 时是 L2，标记 `core_model`、`diffusion`、`parallel`，`hardware_test(res={"cuda": "L4"}, num_cards=2)`，与“完整加载、融合、再量化”的参考一致。
- 动态 LoRA：包装发生在 `build_structure()` 里；`base_layer` 名字规则；槽位声明的形状；管理器在 `unified` 下拒绝大于 `slot_rank` 的适配器并提示参数；没有启用时 `DiffusionWorker.add_lora` 报错。
- L2（`core_model`、`diffusion`，CUDA 单卡）：加载、切换、卸载动态 LoRA 前后，模块结构和基础权重的 `data_ptr` 不变，写入代数每次写槽位加一。

**load_format 和 HSDP**（`test_diffusers_format.py`、`test_hsdp.py`）

- 用测试资源里缩小到几层、由 `save_pretrained` 生成的 diffusers 格式模型目录：`unified` 下不调用 `load_weights()`，组件声明来自 `model_index.json`，前向输出与 legacy 一致（L2，CUDA 单卡）。
- HSDP：物化只为 meta 参数分配，RoPE 表等非持久 buffer 保持构造时的值，不调用 `to_empty`；分片在 `weights()` 池里进行；`pre_sharded` 不调用 `_materialize_pre_sharded_hsdp_state` 和 `_finalize_pre_sharded_hsdp_loaded_weights`；L2，`parallel`，2 卡，两种策略的输出都与 legacy 的 HSDP 一致。

**查询接口**（`test_query_api.py`）

- 第 15 节每个使用方：`legacy` 下结果与原表达式相同，`unified` 下结果与目录一致。

**支持矩阵和指纹**（`test_structure_matrix.py`、`test_legacy_fingerprints.py`）

- 支持矩阵测试每个 PR 运行，新注册的 pipeline 必须构造成功。
- 指纹测试每个 PR 运行。

**逐张量对比**（`tests/diffusion/weight_stack/baseline/`）

- 结构可以配置的家族：合成 checkpoint，L2，标记 `core_model`、`diffusion`，`hardware_test(res={"cuda": "L4"})`，在改动对应家族目录或 `weight_stack/loader/` 的 PR 上运行。
- 结构固定的家族（MiniMax H3、MAGI-2、LTX2 等真实 checkpoint）：L4，标记 `full_model`、`diffusion`，每晚运行，接入 `.buildkite/cuda/test-nightly.yml`。
- 规则表的逐表测试：L1，固定库版本。

## 验收标准

- `legacy` 下现有测试全部不变；本 RFC 的 PR 没有修改已有测试文件。
- `legacy` 下 `build_component` 只调用 legacy 一支；`unified` 下构造上下文里只调用 unified 一支，上下文关闭后调用 `build_component`、`declare_weights`、HF 组件入口报 `LifecycleError` 并写明调用位置。
- 每个迁移家族在上下文里构造时不触发读取陷阱，构造后全部参数都在 meta 上。
- 模型代码吞掉读取陷阱的异常时，构造或收尾照样失败。
- 写入入口对已绑定存储的模块调用 `load_state_dict` 不触发读取陷阱。
- 每个迁移家族的逐张量对比与 legacy 一致：张量名、形状、类型、步长、字节，持久和非持久 buffer，共用关系，模块树的路径和类名；例外表里的每一条都写明原因并对应一条门槛例外。
- 每张规则表的逐表测试通过，`validated_with` 与已安装的库版本相同。
- 未迁移的 pipeline 在 `unified` 下启动报 `ConstructionError`，写明读取位置并提示 `--diffusion-weight-stack legacy`；支持矩阵列出每个注册的 pipeline 的构造结果。
- 量化、TP 布局、共用存储、收尾，每一项只有一个负责的地方；每个最终张量只写一次，重复写入报错。
- 融合适配器后再在线量化的结果，在 TP 为 1 和大于 1 时都与先融合、后量化的参考一致。
- 在线量化时 Host 和加速卡的峰值不高于 `legacy` 的逐层量化。
- 内容身份的包含项和不包含项与总体设计 5.5 一致，由测试逐项断言。
- 组件不完整、保持 fp32 的层配了量化方法，都在规划阶段由 `require_complete()` 报 `PlanningError`。
- 常驻加速卡的张量（包括收尾里新建的和 HSDP 的本地分片）都在 `memory_pools.weights()` 里分配，只建结构期间的分配和写入期间的临时张量不在任何睡眠池里。
- 加载、切换、卸载动态 LoRA 前后，模块结构和基础权重的地址不变。
- `unified` 下 `load_format="diffusers"` 不调用适配器的 `load_weights()`；HSDP 组件的物化不调用 `to_empty`。
- 第 15 节的使用方在 `unified` 下都从组件目录读取，不读 `_dit_modules`、`_encoder_modules`、`_vae_modules`。
- 指纹清单覆盖每个 `build_component` 的 legacy 一支调用的读取和转换函数，指纹测试通过。

## 删除 legacy 时

- 每个 `build_component` 调用只留 unified 一支，`build_component` 删除；`is_building_structure()` 的调用点（C02、D06）只留上下文里的一支。
- 指纹清单登记的 legacy 读取和转换函数，只删没有其他使用方的。OmniVoice 的 generator、decoder 的 `load_weights(model_dir, device)` 还被自回归路径使用，保留；`LanceWanVAE` 的读取同时被 Lance 和 MAGI-2 使用，两个家族都只剩 unified 一支后才删。
- HF 组件入口里加载带权重组件的 legacy 一支；`create_transformers_model` 里的 `init_parameters`、`create_transformers_model_with_vllm_linears` 里的 `init_parameters_preserving_vllm_linear`。
- 只服务于 legacy 的 `load_weights` 主体：自己读文件的（OmniVoice 的 pipeline `load_weights`）、空操作的、替没有写入的组件声明“已加载”的（MiniMax H3 pipeline 对 VAE 和放大器的声明）。
- 规则表取代的运行期改类型代码：LingBot-Video 和 InternVLA-A1 的 `.to()` 覆写、π0.5 的 `_set_inference_dtype`、LTX2 vocoder 的 `.float()` 往返；LTX2 在 `eval()` 里物化相位适配器的代码；MiniMax H3 pipeline 对 FastH3、VDN 的流包装。
- `_maybe_fuse_distilled_lora`、pipeline 上会融合权重的 `load_lora_weights`、worker 里的第二个蒸馏 LoRA 融合点，以及 LoRA 管理器里读 offload 策略的代码和运行期重建槽位的 `_ensure_max_lora_rank` 分支。
- `SupportsComponentDiscovery`、`_dit_modules`、`_encoder_modules`、`_vae_modules` 和 2.2 的默认推导：删除之前，每个注册的 pipeline 都改成显式调用 `declare_components`；第 15 节查询函数的 `legacy` 参数一并删除。
- `DiffusersAdapterPipeline.load_weights()` 和它对 diffusers 自带 offload 的调用。
- 删除之前，逐张量对比测试的 legacy 结果存成逐张量的哈希；删除之后，对比测试改为与这份哈希对比，对比例外表保留。
- 指纹清单 `LEGACY_READERS` 和指纹测试。

总体设计 9.3 的删除前保证里与加载有关的一条由本 RFC 落实：逐张量对比的 legacy 结果已经用 `export_legacy_hashes` 存成逐张量的哈希（14.1）。重复块发现的期望表、只靠 `_layerwise_offload_blocks_attrs` 回退被编译的块类补进 `_repeated_blocks`、MAGI-2 内存窗口的 recipe 由 04 和 05 落实；02-10 的 `repeated_block_paths` 在 `unified` 下返回组件声明或 04 记录的结果，不读 `_layerwise_offload_blocks_attrs`。

## 风险

| 风险 | 影响 | 处理 |
|---|---|---|
| 读取陷阱靠替换模块命名空间里的名字实现，C 扩展或第三方库内部持有的原函数引用绕过它。 | 少数读取在构造期没有被拦截，构造后检查也没发现时，最终字节可能来自构造期而不是 Loader。 | 逐张量对比测试对每个家族比较字节；支持矩阵测试在 CI 里对每个注册的 pipeline 运行；构造后检查兜底构造期用权重算出的属性。 |
| 延迟张量遇到模型 `load_weights` 里不认识的操作时物化。 | 瘦路径下本 rank 范围之外的张量也被读取，冷启动占用页缓存和时间。 | 启动日志报告目标为 meta 却被物化的字节数；瘦路径测试对门槛矩阵里的家族断言为 0。 |
| 写入期间把包装层临时换回 `base_layer`。 | 模型 `load_weights` 持有包装层引用时找不到原来的层。 | 换回的是同一个 `base_layer` 对象；`test_overlay.py` 对启用动态 LoRA 的 pipeline 跑一遍写入，主干里 `_lora_components` 声明的 pipeline 都覆盖。 |
| 规则表跟不上 transformers 或 diffusers 的升级。 | 升级后 `from_pretrained` 的行为变化没有反映到表里，字节不一致。 | 逐表测试断言库版本，升级时全部表重新验证。 |
| 量化方法缺少输出声明。 | 使用这种量化方法的模型在 `unified` 下构造失败。 | 报错写明方法名并提示 `legacy`；支持矩阵按量化方法列出。 |
| checkpoint 里同一层的张量不相邻，或多线程读取让几层交错写入。 | 几层量化前的临时张量同时存在，写入期间的显存峰值超过 `write_transient.quant_layer_bytes`，准入通过后在写入阶段显存不足。 | 读取器对含量化层的组件按键名的自然顺序产出张量，不对这些组件开多线程；Loader 记录写入期间临时张量的峰值并写进启动日志，在线量化的测试断言峰值不超过 `quant_layer_bytes`。 |
| 陷阱的补丁对整个进程生效。 | 上下文打开期间同一进程里的其他加载（例如同进程里另一个 stage 的加载）被拦截。 | 同一进程同一时刻只有一个上下文（3.1）；报错写明调用位置，便于定位。 |
| `pre_sharded` 的物化经 `nn.Module._apply` 只替换 meta 张量（第 13 节）。 | 与 HSDP 在 meta 上分片之后的内部状态不配合时，本地分片没有被 HSDP 登记。 | `test_hsdp.py` 的 2 卡用例对 `pre_sharded` 比较输出和每个分片的字节，这个用例是 02-5 合入的条件。 |

## 反馈期

发布后一周。

## CC

Loader、模型接入、LoRA、量化、torch.compile 与缓存加速的负责人。
