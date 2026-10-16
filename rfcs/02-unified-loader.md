# [RFC]: 统一加载

> 上级 RFC：[umbrella RFC](./00-umbrella.md)。设计依据：[总体设计](../LOADER_OFFLOADER_DESIGN.md) 第 5 节（5.1～5.7）、8.7、9.1、9.2、9.3。
>
> 范围：`vllm_omni/diffusion/weight_stack/loader/`、`vllm_omni/diffusion/weight_stack/components.py`、`vllm_omni/diffusion/model_loader/construct.py`、`vllm_omni/diffusion/model_loader/hub_prefetch.py`，以及 19 个构造期读张量的家族目录。
>
> 依赖：[子 RFC 01](./01-startup-transaction.md)（总开关、`weight_stack/interfaces.py`、`weight_stack/errors.py`、启动驱动）；[子 RFC 03](./03-host-storage.md)（`HostStorage.view`、`HostLayout`）；[子 RFC 04](./04-unified-offloader.md)（`PlacementPlan`、`fully_resident` 的填写）；[子 RFC 05](./05-configuration.md)（`--diffusion-max-lora-rank` 的解析）。

## 摘要

本 RFC 是 umbrella Job 2 的实现设计，覆盖新栈里 Loader 的全部内容：

- 组件声明与组件目录，四个带类型字段的结构，组件查询接口；
- 只建结构的构造上下文、读取陷阱、构造后检查；
- 两个栈在构造期的唯一分发点：`build_component`、`declare_weights`、`hf_component`，以及 `from_pretrained_with_prefetch` 的内部分发；
- 规则表的格式、执行和逐表测试；
- checkpoint 读取器：来源解析、头部预检、存储绑定、写入入口、写入跟踪、逐层在线量化、适配器融合；
- 固定的收尾顺序，内容身份，预期清单和最终清单；
- 动态 LoRA 和 `overlay` 固定适配器的叠加层包装，动态 LoRA 管理器在 `unified` 下的运行期；
- `load_format="diffusers"`、`load_format="custom_pipeline"`、HSDP 组件的物化；
- 共存期的逐张量对比框架、对比例外表、legacy 一支的指纹清单；
- 19 个构造期读张量家族的迁移，以及共存期必改清单里由本 RFC 承担的九行。

本 RFC 的每一个 PR 只改变 `unified` 下的行为，`legacy` 下的行为和已有测试不变，不修改已有测试文件。

## 动机

umbrella 的“背景与问题”已经给出总体原因，这里只列本 RFC 直接替换的现有代码：

- `DiffusersPipelineLoader.load_model`（`vllm_omni/diffusion/model_loader/diffusers_loader.py`）先按加载设备完整构造 pipeline，再经 `load_weights` 写入，之后依次调用 `_maybe_fuse_distilled_lora`、`_process_weights_after_loading`、`_apply_skip_softmax_calibration`、`model.eval()`。offload 规划（`get_offload_plan`）、Host Weight Runtime 的接入（`_resolve_hwr`、`_publish_hwr_after_load`）和 HSDP 的两条路径（`_load_model_with_hsdp`、`_load_model_with_pre_sharded_hsdp`）都穿插在这个方法里。
- 预分片 HSDP 路径的 `_materialize_pre_sharded_hsdp_state` 用 `to_empty` 物化，再保存、恢复非持久 buffer；`_finalize_pre_sharded_hsdp_loaded_weights` 调用模型的 `post_load_weights` 和 `validate_loaded_weights`。
- 蒸馏 LoRA 有两个融合点：`DiffusersPipelineLoader._maybe_fuse_distilled_lora` 和 `DiffusionWorker.init_lora_manager`。融合由 pipeline 的 `load_lora_weights` 在写入之后对参数直接相加（`LoraLoaderMixin.load_lora_into_module`），对不上的键只记警告。
- `DiffusionLoRAManager._replace_layers_with_lora`（`vllm_omni/diffusion/lora/manager.py`）在第一次加载适配器时才包装层；`_ensure_max_lora_rank` 在运行期按更大的 rank 重建槽位；管理器在构造时调用 `resolve_offload_strategy` 选择 LoRA buffer 的设备。
- 19 个家族在构造函数里，或在 pipeline 自己的 `load_weights` 里，直接读文件：`safe_open` 加 `get_tensor`、`load_file`、`torch.load`、`from_single_file`、自定义的 `from_pretrained`。
- `_dit_modules` 等发现属性被多处读取，直接读或经 `ModuleDiscovery.discover`（`vllm_omni/diffusion/offloader/module_collector.py`）读：`DiffusionModelRunner.load_model` 选择编译目标，`DiffusionLoRAManager._replace_layers_with_lora` 选择 LoRA 组件，`vllm_omni/diffusion/cache/cachedit/backend.py` 的 `_dit_module_names`，`vllm_omni/diffusion/registry.py` 的 `_apply_sequence_parallel_if_enabled` 和 `_apply_wan_vae_fastpath_if_enabled`。

## 设计

本节所有要求只对 `unified` 成立。`legacy` 下 Loader 的任何代码都不运行，`build_component` 只调用 legacy 一支，HF 组件入口与现有调用完全相同。

### 1. Loader 在启动流程中的位置

`UnifiedLoader`（`vllm_omni/diffusion/weight_stack/loader/loader.py`）实现 [子 RFC 01](./01-startup-transaction.md) 在 `weight_stack/interfaces.py` 里定义的 `Loader` 协议。启动驱动用 `StartupContext` 构造它（`weight_stack/loader/__init__.py` 导出工厂函数 `make_loader(ctx)`），然后按下表调用：

```text
启动步骤（design 4.2）   调用                                                  结束时的状态
2   只建结构             loader.build_structure()，返回 (model, catalog)        STRUCTURE_READY
4   Offloader.plan()     plan 只读 catalog                                     PLANNED
5   Loader 预检          loader.preflight(catalog, plan)，返回 Preflight        SOURCES_PREPARED
9   Loader 写入          loader.write(model, storage, transaction)             WEIGHTS_READY
10  Loader 收尾          loader.finalize(model, storage)，返回 FinalManifest    FINALIZED
失败或关闭               loader.close()，可以多次调用
```

```python
# vllm_omni/diffusion/weight_stack/loader/__init__.py 导出的、01 的接口引用的类型
@dataclass(frozen=True)
class Preflight:
    content_identities: Mapping[str, ContentIdentity]     # 组件名到内容身份（10.1）
    expected_manifest: ExpectedManifest                    # 10.2
    sources: Mapping[str, tuple[ResolvedSource, ...]]      # 组件名到解析后的来源（7.1）

    def identity_digests(self) -> Mapping[str, str]: ...  # 即 RankFingerprints.content_identities
```

- `write()` 里每个张量的写入目标取 `transaction.target(tensor_key)`：返回张量时绑定到它；返回 `None` 时这个张量常驻加速卡，Loader 在 `ctx.memory_pools.weights()` 里分配（7.3）。
- `finalize()` 执行 design 5.4 的第 1 到第 7 步（第 9 节），第 6 步调用 `storage.seal()`。
- Loader 只 import `weight_stack.interfaces`、`weight_stack.errors`、`weight_stack.components` 和 `model_loader.construct`，不 import HWM 和 Offloader 的实现。
- 每一步失败时抛出 01 在 `weight_stack/errors.py` 里定义的对应类型：`ConstructionError`、`PreflightError`、`WriteError`、`FinalizeError`。上下文关闭之后再调用组件构造函数报 `LifecycleError`。

### 2. 组件声明与组件目录

#### 2.1 模型侧的声明

模型侧的类型和函数放在 `vllm_omni/diffusion/weight_stack/components.py`。`models/`、`lora/`、`cache/` 可以 import 它；它不 import Loader、HWM、Offloader 的实现。

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

pipeline 没有调用 `declare_components` 时，Loader 在构造结束后从现有的发现属性推导声明，推导规则与 `ModuleDiscovery.discover` 相同：

| 来源 | 推导出的声明 |
|---|---|
| `_dit_modules` 的每一项 | `role="dit"`，名字和路径都取这一项 |
| `_encoder_modules` 的每一项 | 名字是 `image_encoder`、`vision_encoder` 时 `role="image_encoder"`，否则 `role="text_encoder"` |
| `_vae_modules` 的每一项 | `role="vae"` |
| `_resident_modules` 的每一项 | `role="other"` |
| pipeline 不实现 `SupportsComponentDiscovery` | 按 `ModuleDiscovery._FALLBACK_DIT_ATTRS`、`_FALLBACK_ENCODER_ATTRS`、`_FALLBACK_VAE_ATTRS` 推导，角色同上 |

推导出的声明不带任何带类型字段。显式声明完全取代推导，两者不合并。推导结果通不过完整性检查的 pipeline 必须显式声明，CI 的支持矩阵测试（5.7）列出它们。

#### 2.3 路径解析、路径列表与嵌套划出

- 路径按属性路径解析，规则与 `ModuleDiscovery._collect_modules` 用的 `operator.attrgetter` 相同，可以经过不是 `nn.Module` 的对象。例如 Gr00t 的模型挂在普通对象 `Gr00tPolicy` 上（`vllm_omni/diffusion/models/gr00t/policy.py`），路径写 `policy.model`。这类组件的根不在 `pipeline.named_modules()` 里，完整性检查和对比测试把声明的根并入检查范围。
- 路径解析到 `None`，或解析到的子树里没有任何参数、持久 buffer 和登记过来源的数据张量时，这个组件视为不存在，不进目录。
- `path` 是列表时，组件由这几棵子树组成。例如 MiniMax H3 视频 VAE 的编码半和解码半各是一个 `vae` 组件，各自的 `path` 列出所属的几个子模块。
- 一个组件的根落在另一个组件的子树里时，内层从外层划出：外层组件的张量是外层子树去掉所有内层子树之后剩下的部分。内层的根必须恰好落在一个外层组件里。例如 Cosmos3 的 `text_encoder` 声明为 `transformer.language_model`，`dit` 声明为 `transformer`。
- 同一个模块挂在两个属性上时按模块身份去重，第一次出现的路径是它的规范路径。张量名、内容身份和清单都使用规范路径。
- 同一个张量对象出现在几个路径下（构造时就共用的参数，例如 `self.lm_head.weight = self.embed.weight`）按张量身份去重，记为一个张量和若干别名。

#### 2.4 完整性检查

构造结束、最终类型落到 meta 参数之后，Loader 遍历 pipeline 下的每个参数、每个持久 buffer 和每个登记过来源的非持久 buffer（加上 2.3 第一条并入的根），要求每一个恰好属于一个组件。不满足时在构造阶段报错，错误里列出没有归属的路径和归属多于一个的路径。只想常驻、不参与 offload 的部分显式声明为 `other`。design 8.6 把“组件完整”列在规划一层；Loader 在构造结束时先做一次，规划时 Offloader 只核对它拿到的目录里各组件互不重叠。

#### 2.5 按 rank 声明

声明按 rank 生成，Loader 不要求各 rank 相同。没有张量的组件视为不存在，例如 MAGI-2 在非输出 rank 上把文本编码器构造成 `nn.Identity()`（`Magi2Pipeline.__init__`），这个 rank 的目录里就没有 `text_encoder`。哪些组件只在部分 rank 上存在、是否参与全组核对，由 01 在 design 4.5 第 1 点收集各 rank 的目录后决定。

#### 2.6 带类型字段

字段集合是封闭的，加字段要经过设计评审。每个字段在构造结束时按下表校验，校验失败在构造阶段报错：

| 字段 | 类型 | 校验 | 使用方 |
|---|---|---|---|
| `repeated_blocks` | 相对组件根的路径元组 | 每个路径解析到一个 `nn.ModuleList` 或 `nn.Sequential`，子模块的类相同；路径之间互不嵌套。元组顺序就是常驻层的计数顺序 | Offloader 规划（04），编译区域选择（第 15 节） |
| `unit_groups` | 组名到路径模式的映射 | 组名是标识符；模式里的 `*` 匹配恰好一段路径。模式在构造时只做语法校验，选中的组是否匹配到模块由规划校验（04） | 策略的 `units`（04、05） |
| `shard_group` | `ShardGroupSpec` | `size >= 1`，`0 <= rank < size`；同一个名字在一个 rank 的所有声明里大小和编号相同 | 内容身份（10.1），AllGather 分组（03） |
| `fixed_adapters` | `FixedAdapter` 元组 | 名字在组件内唯一；`targets` 每一项至少匹配一个支持的层（8.5）；`format` 已注册 | Loader（第 8 节） |

每个字段的值进入规划指纹（04 计算），由各使用方的期望表测试覆盖。

#### 2.7 组件目录

目录的数据类型同样放在 `weight_stack/components.py`，构建逻辑放在 `weight_stack/loader/catalog.py`。Offloader 的 `plan()` 只读这个对象，不 import Loader。

```python
@dataclass(frozen=True)
class TensorDecl:
    name: str                     # 包装前的完整名字，例如 "transformer.blocks.0.attn.to_q.weight"
    aliases: tuple[str, ...]      # 共用同一个张量的其他名字
    module: str                   # 所属模块的规范路径（包装前）
    kind: Literal["parameter", "buffer", "data", "quant_output", "fixed_overlay"]
    shape: tuple[int, ...]
    dtype: torch.dtype
    stride: tuple[int, ...]
    required: bool                # 必须由某个来源写入
    source: str | None            # 来源标识；weights_sources 的流记为 "weights_sources"

@dataclass
class CatalogEntry:
    decl: ComponentDecl
    roots: tuple[str, ...]        # 解析后的规范根路径
    carved: tuple[str, ...]       # 从本组件划出的内层组件的根
    tensors: tuple[TensorDecl, ...]
    rule_tables: tuple[str, ...]  # 用到的规则表：限定名加版本
    hsdp_managed: bool
    shapes_final: bool                   # 写入之前就能确定每个张量的最终形状和类型（design 6.3、6.4）
    finalize_replaced: tuple[str, ...]   # 收尾第 2 步会整个替换的张量（第 9 节）
    fully_resident: bool | None          # commit() 之前为 None

@dataclass(frozen=True)
class TransientNeed:
    adapter_delta_bytes: int      # 最大的单个适配器 Δ（8.2）
    quant_layer_bytes: int        # 最大的一层量化前的数据（7.6）

class ComponentCatalog:
    entries: Mapping[str, CatalogEntry]        # 按组件名排序
    overlay_slots: tuple[TensorDecl, ...]      # 动态 LoRA 槽位（11.4），不属于任何组件的张量表
    write_transient: TransientNeed             # 写入期间加速卡上的瞬时需求，04 计入加速卡准入
    def owner(self, tensor_name: str) -> str: ...
    def module_owner(self, module_path: str) -> str: ...
    def set_fully_resident(self, component: str, value: bool) -> None: ...
```

- `tensors` 是最终张量表：参数、持久 buffer、登记过来源的数据张量、量化方法声明的输出张量（7.6）、`overlay` 固定适配器的 A/B（8.5）。量化层量化前的参数不在表里，它们只在写入期间存在于加速卡的暂存上。
- `kind="buffer"` 只指持久 buffer。没有登记来源的非持久 buffer（例如按配置算出的 RoPE 表）不在表里，它们跟随所在模块由 Offloader 放置。
- `set_fully_resident` 只给 04 的 `commit()` 用，每个组件只能调用一次，之后属性只读。
- `shapes_final`、`finalize_replaced` 和 `write_transient` 都在只建结构结束时算出，规划据此选择共享条件和冷启动路径（03、04），不需要等到预检。`write_transient` 是上限：`adapter_delta_bytes` 取所有要融合适配器的组件里最大的单个张量按计算类型的字节数，`quant_layer_bytes` 取量化层里量化前参数之和的最大值。

#### 2.8 查询接口

`weight_stack/components.py` 提供以下只读函数，供 `torch.compile` 的目标选择、缓存加速、动态 LoRA、模型注册表、序列并行钩子和 VAE 快路径使用（第 15 节）。每个函数带一个 `legacy` 参数：`legacy` 下函数只调用它并返回结果，使用方现有的判断原样移进这个局部函数，`legacy` 下行为不变。

```python
def component_paths(pipeline, role: Role, *, legacy: Callable[[], Sequence[str]]) -> tuple[str, ...]: ...
def lora_target_paths(pipeline, *, legacy: Callable[[], Sequence[str]]) -> tuple[str, ...]: ...
def repeated_block_paths(pipeline, component: str, *, legacy: Callable[[], Sequence[str]]) -> tuple[str, ...]: ...
def fully_resident(pipeline, component: str, *, legacy: Callable[[], bool]) -> bool: ...
def execution_device() -> torch.device: ...
def is_unified_stack() -> bool: ...
def weight_generation(pipeline) -> int: ...
def bump_weight_generation(pipeline) -> None: ...
```

- `unified` 下前三个函数读目录：`component_paths` 返回这个角色的组件的规范根路径；`lora_target_paths` 返回 `dit` 组件和 `_lora_components` 的根路径；`repeated_block_paths` 返回组件声明的 `repeated_blocks`，没有声明时返回 04 的自动发现结果。
- `fully_resident` 是读这个属性的通用函数：`unified` 下读目录里 `commit()` 填好的值，`commit()` 之前调用报错；`legacy` 下执行模型现有的判断。
- `execution_device()` 返回本 worker 的执行设备，与 `vllm_omni.diffusion.distributed.utils.get_local_device()` 相同。
- `is_unified_stack()` 返回 01 的 `process_weight_stack() is WeightStack.UNIFIED`。模型代码只在共存期必改清单登记的位置调用它（01 的静态测试检查调用位置）。
- `weight_generation` 和 `bump_weight_generation` 是以权重为键的缓存签名用的写入代数（design 7.8）。基础权重提交后不变；动态 LoRA 管理器每次写叠加层槽位后调用一次 `bump_weight_generation`。签名的计算归 04。

## 与其他子 RFC 的接口

（撰写中）

## 代码组织

（撰写中）

## 实现拆分

（撰写中）

## 测试计划

（撰写中）

## 验收标准

（撰写中）

## 删除 legacy 时

（撰写中）

## 风险

（撰写中）

## 反馈期

发布后一周。

## CC

Loader、模型接入、LoRA、量化、torch.compile 与缓存加速的负责人。
