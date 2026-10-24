# [RFC]: 配置与可观测性

> 上级 RFC：[umbrella RFC](./00-umbrella.md)。设计依据：[总体设计](../LOADER_OFFLOADER_DESIGN.md) 第 8.2 到 8.7 节、第 10 节、第 13 节，以及 umbrella RFC 的“配置”“可观测性”两节。本 RFC 取代 #7204。
>
> 范围：`vllm_omni/diffusion/weight_stack/policy.py`、`vllm_omni/diffusion/weight_stack/observability.py`；配置传递链上的 `vllm_omni/entrypoints/cli/serve.py`、`vllm_omni/engine/arg_utils.py`、`vllm_omni/config/stage_config.py`、`vllm_omni/config/omni_config.py`、`vllm_omni/diffusion/data.py`；指标链上的 `vllm_omni/diffusion/worker/diffusion_worker.py`、`vllm_omni/diffusion/diffusion_engine.py`、`vllm_omni/engine/orchestrator.py`、`vllm_omni/metrics/`；`recipes/`、`docs/`。
>
> 依赖：[子 RFC 01](./01-startup-transaction.md)（总开关 `is_unified()`、`ConfigError` 等错误类型和报错格式、`ExecutionOffloadFacts` 和 `execution_offload_facts`、`StartupContext.policy`、登记表和静态测试 T1、T5、T6）；[子 RFC 02](./02-unified-loader.md)（组件目录的组件名和角色、`DynamicLoraSpec`、支持矩阵测试 `test_structure_matrix.py` 的结果）；[子 RFC 03](./03-host-storage.md)（`Mapping.kind`、HWM 的启动日志条目）；[子 RFC 04](./04-unified-offloader.md)（`plan()` 按本 RFC 的选择规则读取策略，内存登记、暂存区和加速卡槽位的字节数）。

## 摘要

本 RFC 给出 `unified` 下的全部配置面和可观测性：

- `OffloadPolicy` 数据类型、它的 JSON schema，以及命令行、Python 参数和 YAML 共用的一个解析器和来源之间的优先级；
- 四个预设的展开；
- 旧开关和旧 `--diffusion-offload-config` 的兼容层、翻译测试和它的豁免，旧设置的迁移报错；
- host 策略参数 `--pin-cpu-memory` / `--no-pin-cpu-memory`、`--diffusion-host-registration-limit-gib`，动态 LoRA 参数 `--diffusion-max-lora-rank`；
- 不支持的组合、按总体设计 8.6 分层的全部校验规则和报错文本，`OMNI_DIFFUSION_PROMPT_EMBED_CACHE` 折进配置；
- 五项内存指标、启动日志的格式；
- recipe、文档和支持矩阵。

`legacy` 下的解析、校验和行为不变；本 RFC 新增的参数在 `legacy` 下报错，不静默生效。

## 动机

umbrella RFC 的“配置”一节给出了目标形态。现有代码在本 RFC 范围内的问题，逐项核对如下：

- **offload 配置分散在三套写法里**。`OmniDiffusionConfig` 有三个布尔开关（`enable_cpu_offload`、`enable_layerwise_offload`、`enable_distributed_layerwise_offload`）、`dlo_use_allgather`、`dlo_resident_layers` 和 `diffusion_offload_config`。`offloader/config.py` 的 `resolve_offload` 把它们合成 `ResolvedOffload`，`materialize_legacy_offload_flags` 再把结果写回三个布尔字段，供模型和后端读取。
- **旧配置只能表达两个组件**。`parse_diffusion_offload_config` 只接受 `mode`、`components`、`layer_options`、`pin_memory` 四个键，`components` 只能取 `OFFLOAD_COMPONENTS` 里的 `dit` 和 `text_encoder`，`_parse_layer_options` 只允许 `dit` 设 `resident_layers`。`vae`、`image_encoder`、`audio_encoder` 和按组件名的选择都无法表达。
- **校验分散在五处，时机不同**。`_DiffusionConfigProjection.from_kwargs` 调用 `parse_diffusion_offload_config`；`OmniDiffusionConfig.__post_init__` 里有 `resolve_offload` 的缓存加速和 prompt embedding 缓存组合检查、`validate_host_weight_runtime_options`、`validate_dlo_host_registration_options` 和 `diffusion_compile_granularity="full"` 的冲突检查；`DiffusionModelRunner.load_model` 在 worker 里再查一次 prompt embedding 缓存与文本编码器 AllGather；`DiffusersPipelineLoader` 在加载时才拒绝 `hsdp_weight_load_strategy="pre_sharded"` 配 LoRA 或量化；`resolve_offload_plan` 在构造之后才发现模型没有声明常驻路径。
- **worker 读环境变量**。`DiffusionModelRunner.load_model` 调用 `resolve_prompt_embed_cache_config`，后者在 worker 进程里读 `OMNI_DIFFUSION_PROMPT_EMBED_CACHE` 和 `OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE`，参数解析阶段看不到最终取值。
- **模型自带的 offload 开关藏在缓存配置里**。`MiniMaxH3Transformer3DModel` 从 `cache_config` 读 `minimax_h3_adaln_offload`，在 SM120、CUDA、未量化时自动开启。
- **字典配置在来源合并时被逐键拼接**。`StageConfig.to_omegaconf` 把命令行覆盖里字典类型的值用 `_get_recursively_merged_dict` 递归并进 YAML 的 `engine_args`，只有 `omni_kv_config` 整体替换；`_merge_config_fields` 对 `engine_args` 同样递归合并。一份字典形式的策略经过合并后可能是两个来源的拼接。
- **`pin_cpu_memory` 没有命令行参数**，旧配置的 `pin_memory` 在 `resolve_offload` 里优先于它；`--dlo-host-registration-limit-gib` 只对 Host Weight Runtime 的映射生效（`validate_dlo_host_registration_options` 要求开启不用 AllGather 的分布式逐层 offload 和 Host Weight Runtime）。
- **没有 host 存储的指标**。`docs/design/metrics.md` 列出的指标族和 `vllm_omni/metrics/definitions.py` 里没有 host 存储、内存登记、暂存区和加速卡槽位的字节数。

## 设计

### 1. 配置项总表

`unified` 下用户可见的配置项如下。“新增”指本 RFC 新增的字段或参数，“已有”指现有字段。

| 命令行参数 | 字段（Python 参数和 YAML） | 类型和默认值 | 来源 |
|---|---|---|---|
| `--diffusion-weight-stack {legacy,unified}` | `diffusion_weight_stack` | 由 01 定义，默认 `legacy`。 | 01 |
| `--diffusion-offload {none,module,layerwise,distributed-layerwise}` | `diffusion_offload` | `str \| None`，默认 `None`。 | 新增 |
| `--diffusion-offload-policy '<JSON>'` | `diffusion_offload_policy` | `dict[str, Any] \| str \| None`，默认 `None`。字符串按 JSON 解析。 | 新增 |
| `--pin-cpu-memory` / `--no-pin-cpu-memory` | `pin_cpu_memory` | `bool`，字段默认 `True`；命令行参数默认 `None`，表示没有给出。 | 字段已有，参数新增 |
| `--diffusion-host-registration-limit-gib` | `host_registration_limit_gib` | `float`，默认 `0.0`，0 表示不设上限。 | 新增 |
| `--diffusion-max-lora-rank` | `max_lora_rank` | `int \| None`，默认 `None`。 | 新增 |
| 没有命令行参数 | `max_cpu_loras` | 已有，含义不变。 | 已有 |
| `--lora-path`、`--lora-backend`、`--lora-scale` | `lora_path`、`lora_backend`、`lora_scale` | 已有，含义不变。 | 已有 |

新增字段沿用 01 第 1.1 节总开关的传递链，五个文件各加一项：

```text
vllm_omni/entrypoints/cli/serve.py   subparser_init 的 omni_config_group   新增五个参数，default 都是 None
vllm_omni/engine/arg_utils.py        OrchestratorArgs                      diffusion_offload、diffusion_offload_policy、
                                                                           host_registration_limit_gib、max_lora_rank，默认 None；
                                                                           pin_cpu_memory: bool | None = None
vllm_omni/config/stage_config.py     StageDeployConfig                     同上，默认 None
vllm_omni/config/omni_config.py      _DiffusionConfigProjection            diffusion_offload: str | None = None
                                                                           diffusion_offload_policy: dict[str, Any] | str | None = None
                                                                           host_registration_limit_gib: float = Field(default=0.0, ge=0)
                                                                           max_lora_rank: int | None = None
vllm_omni/diffusion/data.py          OmniDiffusionConfig                   同上四个字段，在 __post_init__ 里由 policy.py 校验
```

- 命令行参数默认是 `None`，`build_stage_runtime_overrides` 跳过值为 `None` 的键，所以没给出的参数不覆盖 YAML。`--pin-cpu-memory` 和 `--no-pin-cpu-memory` 的 `dest` 都是 `pin_cpu_memory`，前者 `store_const` 为 `True`，后者为 `False`。
- 四个新字段属于 `_DIFFUSION_CONFIG_FIELDS`，随 `od_config` 到达每个 worker，不进内容身份。
- `diffusion_offload_policy` 在配置里保存原始的映射或字符串，与现有 `diffusion_offload_config` 保存原始映射的做法相同，跨进程序列化不经过解析后的对象。解析后的 `OffloadPolicy` 不挂在配置对象上；worker 用 `startup_policy(od_config)`（第 13 节）从字段重新得到同一个结果。
- `_DiffusionConfigProjection.from_kwargs` 在调用 `parse_diffusion_offload_config` 之后，对 `diffusion_offload_policy` 调用 `policy.parse_policy_document()` 只做结构校验（第 4.1 节），让 YAML 和 Python 参数里的结构错误在构造 stage 之前报出。

### 2. `OffloadPolicy`

#### 2.1 数据类型

```python
# vllm_omni/diffusion/weight_stack/policy.py
SELECTABLE_ROLES: tuple[str, ...] = ("dit", "text_encoder", "image_encoder", "audio_encoder", "vae")
ALL_ROLES: tuple[str, ...] = SELECTABLE_ROLES + ("other",)

class Mode(str, enum.Enum):
    MODULE = "module"
    LAYER = "layer"

class WeightTransfer(str, enum.Enum):
    RANK_LOCAL = "rank-local"
    ALLGATHER = "allgather"

class Preset(str, enum.Enum):
    NONE = "none"
    MODULE = "module"
    LAYERWISE = "layerwise"
    DISTRIBUTED_LAYERWISE = "distributed-layerwise"

@dataclass(frozen=True)
class ComponentSetting:
    mode: Mode
    weight_transfer: WeightTransfer = WeightTransfer.RANK_LOCAL
    resident_layers: int = 0
    units: tuple[str, ...] = ()              # unit_groups 的名字，按给出的顺序去重

@dataclass(frozen=True)
class Selector:
    key: str                                 # 角色名或组件名
    by: Literal["role", "name"]              # key 属于 ALL_ROLES 时为 role
    setting: ComponentSetting | None         # None：不选这个键匹配的组件
    if_absent: Literal["error", "ignore"]    # 在本副本里没有匹配的组件时
    if_no_blocks: Literal["error", "skip"]   # 按 layer 处理但找不到重复块时

@dataclass(frozen=True)
class HostPolicy:
    pin_cpu_memory: bool
    registration_limit_bytes: int            # host_registration_limit_gib 换算成字节，向下取整；0 表示不设上限

@dataclass(frozen=True)
class PolicyOrigin:
    kind: Literal["preset", "custom", "legacy"]
    preset: Preset | None                    # kind 为 preset 时
    legacy_fields: tuple[str, ...]           # kind 为 legacy 时，参与翻译的旧字段名，按字段名排序
    canonical: str                           # 规范 JSON（第 3.3 节），用于日志和比较

@dataclass(frozen=True)
class OffloadPolicy:
    selectors: tuple[Selector, ...]          # 角色键在前、组件名键在后，各自按键排序
    host: HostPolicy
    origin: PolicyOrigin
    pending_resident_layers: int = 0         # 第 6.3 节；resolve_for_model() 之后为 0
    def any_allgather(self) -> bool: ...
    def execution_facts(self) -> ExecutionOffloadFacts: ...      # 第 12 节
    def resolve_for_model(self, model_class_name: str | None) -> OffloadPolicy: ...   # 第 6.3 节
```

- 所有类型都是冻结的数据类，可以 pickle，可以比较。两份策略相等当且仅当 `selectors` 和 `host` 相等；`origin` 不参与比较（`field(compare=False)`）。
- 没有任何 `setting` 不为 `None` 的选择器时，解析结果是 `None`，表示不做 offload。`StartupContext.policy` 为 `None` 时 04 按不做 offload 规划（01 第 2 节）。
- `OffloadPolicy` 只描述用户的选择，不含模型信息。哪些组件被选中由 04 在 `plan()` 里调用第 11 节的 `resolve_selection()` 决定。

#### 2.2 规范化规则

- `units` 写成字符串时视为只有一个名字的列表；列表里重复的名字只保留第一次出现的位置。
- `resident_layers` 和 `units` 只对解析出的 `mode` 为 `layer` 的设置有意义，校验见第 8 节。
- 组件名键的设置**整体覆盖**它所属角色键的设置：组件名键里没写的字段取默认值（顶层 `mode`、`rank-local`、0、空 `units`），不从角色键继承。
- `if_absent`、`if_no_blocks` 不出现在 JSON 里，由策略的来源决定：

| 来源 | `if_absent` | `if_no_blocks` |
|---|---|---|
| 自定义配置 `diffusion_offload_policy` | `error` | `error` |
| 预设 `diffusion_offload` | `ignore` | `error` |
| 旧开关（`enable_*_offload`） | `ignore` | `dit` 为 `error`，`text_encoder` 为 `skip` |
| 旧配置 `diffusion_offload_config` | `error` | `error` |

旧开关的 `skip` 对应总体设计 8.4“没有重复块的编码器不做 offload”；旧配置的 `error` 对应 `resolve_offload_plan` 对显式组件列表的严格检查（`explicit` 为真时报 `No DiT/transformer modules found for selected DiT offload` 等错误）。

### 3. JSON schema

#### 3.1 schema

`policy.json_schema()` 返回下面的 schema，文档页面直接引用它。运行期解析不依赖 `jsonschema` 包，由第 4 节的解析器实现同样的规则。

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://docs.vllm.ai/projects/vllm-omni/schemas/diffusion-offload-policy-v1.json",
  "title": "diffusion_offload_policy",
  "type": "object",
  "additionalProperties": false,
  "required": ["components"],
  "properties": {
    "mode": {"enum": ["module", "layer"]},
    "components": {
      "type": "object",
      "minProperties": 1,
      "propertyNames": {"type": "string", "minLength": 1, "not": {"const": "other"}},
      "additionalProperties": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/setting"}]}
    }
  },
  "$defs": {
    "setting": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "mode": {"enum": ["module", "layer"]},
        "weight_transfer": {"enum": ["rank-local", "allgather"], "default": "rank-local"},
        "resident_layers": {"type": "integer", "minimum": 0, "default": 0},
        "units": {
          "oneOf": [
            {"type": "string", "minLength": 1},
            {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1}
          ]
        }
      }
    }
  }
}
```

schema 之外由解析器检查的规则：每个被选中的设置必须能解析出 `mode`（自己的 `mode` 或顶层 `mode`）；`resident_layers` 不接受布尔值；第 8 节列出的组合规则。

#### 3.2 示例

```json
{"components": {"dit": {"mode": "layer", "units": "adaln"}}}
{"mode": "layer", "components": {"dit": {"weight_transfer": "allgather"}, "text_encoder": {"mode": "module"}, "vae": {"mode": "module"}, "latent_upscaler": null}}
{"mode": "module", "components": {"dit": {}, "vae": {}}}
```

第一条是 MiniMax H3 在 SM120 上的 AdaLN 投影；第三条是“DiT 和 VAE 都按整组件”，`{}` 表示按默认值选中。

#### 3.3 规范 JSON

`PolicyOrigin.canonical` 是策略的规范 JSON：只含 `components`，键排序，每个设置写全四个字段，`units` 写成列表，`null` 选择器保留。预设和旧开关的翻译结果也写成这种形式，所以启动日志里任何来源的策略都能直接复制成 `--diffusion-offload-policy` 的值。`host` 不写进规范 JSON。

### 4. 解析器、来源与优先级

#### 4.1 一个解析器

```python
def parse_policy_document(value: object) -> PolicyDocument: ...       # 只做结构校验，返回规范化的文档
def parse_unified_config(config: OmniDiffusionConfig) -> OffloadPolicy | None: ...   # 参数解析阶段的入口（第 13 节）
def startup_policy(config: OmniDiffusionConfig) -> OffloadPolicy | None: ...         # worker 里由 runner 调用（第 13 节）
```

- 命令行的 `--diffusion-offload-policy` 的 `type` 是 `str`，原样存进字段；Python 参数可以传映射或 JSON 字符串；YAML 写映射。三者都由 `parse_policy_document` 解析，字符串先用 `json.loads` 解析，解析失败报 `P04`（第 8 节）。
- `parse_policy_document` 是纯函数，不读配置的其他字段，不读环境变量，不 import `loader/`、`host/`、`offload/` 和 `vllm_omni.diffusion.offloader`（01 的 T1）。
- 同一份输入在任何进程里得到相同的结果；解析结果不依赖字典的插入顺序。

#### 4.2 同一来源里的互斥

`diffusion_offload` 和 `diffusion_offload_policy` 在同一个来源里只能给一个，两个都给时报 `P03`。旧开关和新写法的关系见第 6.5 节。

#### 4.3 来源之间的优先级

来源从高到低：

```text
1. 命令行参数和 Python 参数（Omni、AsyncOmni 的关键字参数）
   同一层里 stage_<id>_<字段> 的写法只作用于对应的 stage
2. 部署 YAML（--deploy-config）里 stage 的 engine_args，包括平台覆盖和管线部署覆盖合并之后的结果
3. 字段默认值
```

- 高优先级的来源给出了 `diffusion_offload` 或 `diffusion_offload_policy` 中的任何一个时，低优先级来源里的这两个键一起丢弃，“offload 策略”作为一个整体被覆盖。`stage_config.py` 新增 `reconcile_diffusion_offload_overrides(engine_args, runtime_overrides)`，写法与已有的 `reconcile_diffusion_attention_overrides` 相同，在 `StageConfig.to_omegaconf` 里紧跟它调用。
- `diffusion_offload_policy` 的值不做逐键合并：`StageConfig.to_omegaconf` 的覆盖循环把它和 `omni_kv_config` 一样整体替换；`_get_recursively_merged_dict` 遇到键 `diffusion_offload_policy` 时整体替换，不递归。YAML 覆盖之间（`_merge_config_fields`、平台覆盖）同样按“一个整体”处理：覆盖层给出两个键中的任何一个时，基础层的两个键一起丢弃。这个规则由 `policy.merge_offload_pair(base, overlay)` 实现，三个合并点都调用它。
- `diffusion_offload_policy` 是新字段，上面的改动不影响任何已有字段的合并结果，`legacy` 下的合并行为不变。
- 旧开关、旧配置和 `pin_cpu_memory` 的合并规则不变。

### 5. 预设

| 预设 | 展开成的选择器 |
|---|---|
| `none` | 没有选择器，解析结果为 `None`。 |
| `module` | `dit`、`text_encoder`、`image_encoder` 三个角色键，设置都是 `{"mode": "module"}`。 |
| `layerwise` | `dit` 角色键，设置是 `{"mode": "layer", "weight_transfer": "rank-local"}`。 |
| `distributed-layerwise` | `dit` 角色键，设置是 `{"mode": "layer", "weight_transfer": "allgather"}`。 |

- 预设的选择器 `if_absent` 为 `ignore`、`if_no_blocks` 为 `error`（第 2.2 节）。
- 预设不选 `vae`，也不设 `resident_layers` 和 `units`。
- `unified` 下不做与硬件相关的隐式 offload：没有给出预设和自定义配置时等同 `none`。显存放不下时由 04 的加速卡容量准入报 `DeviceCapacityError`，`fix` 写 `see the recipe for this model under recipes/ for an offload policy`。

### 6. 旧开关的兼容层

#### 6.1 旧开关的翻译

`unified` 下，`parse_unified_config` 把旧开关翻译成 `origin.kind` 为 `legacy` 的 `OffloadPolicy`。三个布尔开关同时出现时，与 `legacy` 的 `_LEGACY_STRATEGY_PRIORITY` 相同，只翻译优先级最高的一个：`enable_distributed_layerwise_offload` 高于 `enable_layerwise_offload` 高于 `enable_cpu_offload`。

| 旧开关 | 翻译成的选择器 |
|---|---|
| `--enable-cpu-offload` | 与预设 `module` 相同：`dit`、`text_encoder`、`image_encoder` 三个角色键，`{"mode": "module"}`。 |
| `--enable-layerwise-offload` | `dit` 角色键 `{"mode": "layer", "weight_transfer": "rank-local"}`；`text_encoder` 角色键 `{"mode": "layer", "weight_transfer": "rank-local"}`，`if_no_blocks` 为 `skip`。 |
| `--enable-distributed-layerwise-offload` | `dit` 角色键 `{"mode": "layer", "weight_transfer": "allgather"}`；`text_encoder` 角色键同上一行。 |
| `--dlo-no-use-allgather` | 只在 `--enable-distributed-layerwise-offload` 被翻译时生效，把 `dit` 的 `weight_transfer` 改成 `rank-local`；其他情况下忽略，与 `legacy` 相同。 |
| `--dlo-resident-layers N` | 见第 6.3 节。 |

- `--enable-cpu-offload` 的结果等于“现有实现发现的全部 DiT 和编码器，编码器名单里属于 `vae` 角色的除外”，前提是每个编码器在组件声明里的角色符合总体设计 8.4 的对应：`mllm`、`connectors` 是 `text_encoder`；`vision_model`、`gen_image_condition_refiner`、`conditioning` 是 `image_encoder`；JoyImage 名单里的 `vae` 是 `vae`。这个前提由第 6.6 节的翻译测试逐个 pipeline 断言。
- 旧开关从不选 `vae`。
- `dlo_use_allgather` 为真且 `dlo_resident_layers` 大于 0 时，`legacy` 在 `_validate_legacy_layer_options` 里报错；`unified` 下翻译出的 `allgather` 配常驻层数由第 8 节的 `P16` 报错，两边都拒绝这个组合。

#### 6.2 旧配置 `diffusion_offload_config`

旧配置的格式由 `parse_diffusion_offload_config` 定义：

```json
{
  "mode": "module | layer",
  "components": ["dit", "text_encoder"],
  "layer_options": {
    "dit": {"weight_transfer": "rank-local | allgather", "resident_layers": 0},
    "text_encoder": {"weight_transfer": "rank-local | allgather"}
  },
  "pin_memory": true
}
```

- `mode` 和 `components` 必填；`components` 只能取 `dit`、`text_encoder`；`layer_options` 的键必须是选中的组件，只在 `mode` 为 `layer` 时允许；`resident_layers` 只允许 `dit`，并要求 `dit` 的 `weight_transfer` 是 `rank-local`；`pin_memory` 是可选的布尔值。
- 翻译规则：`mode` 照搬成每个选择器的 `mode`；`components` 列表里的每一项成为一个角色键；`layer_options` 里每个组件的 `weight_transfer`、`resident_layers` 照搬，没写 `weight_transfer` 时取 `rank-local`（与 `resolve_offload` 的 `options.weight_transfer or DLOTransfer.RANK_LOCAL` 相同）；`pin_memory` 翻成 `HostPolicy.pin_cpu_memory`（第 6.5 节）。选择器的 `if_absent`、`if_no_blocks` 都是 `error`。
- `weight_stack` 不能 import `offloader/`（01 的 T1），所以 `policy.py` 有自己的 `parse_legacy_offload_config()`，规则和报错文本与 `parse_diffusion_offload_config` 逐条相同，旧配置与旧开关组合时的规则与 `_validate_legacy_layer_options`、`resolve_offload` 相同（第 8 节的 `P32`、`P33`）。第 6.6 节的等价测试保证两边对同一份输入的接受、拒绝和报错文本一致。
- `_DiffusionConfigProjection.from_kwargs` 对旧配置调用 `parse_diffusion_offload_config` 的语句不变，两个栈都执行。

#### 6.3 `--dlo-resident-layers`

`legacy` 下，`resolve_offload_plan` 只在模型的 `OffloadPlan.resident_dit_paths` 与 DiT 路径相交时使用常驻层数，否则记警告并全部流式搬运。主干里声明了 `resident_dit_paths` 的是 `MiniMaxH3Pipeline` 和 `Kandinsky6TI2VAPipeline`。参数解析阶段还不知道模型类，翻译分两步：

1. `parse_unified_config` 在 `--enable-distributed-layerwise-offload` 和 `--dlo-no-use-allgather` 同时给出时，把 `N` 记进 `OffloadPolicy.pending_resident_layers`；只给了 `--dlo-resident-layers`、没有分布式逐层开关时忽略它并记一次警告。
2. `OffloadPolicy.resolve_for_model(model_class_name)` 在 worker 里由 `startup_policy` 调用：`model_class_name` 在 `LEGACY_RESIDENT_DIT_MODELS` 里时把 `N` 写进 `dit` 选择器的 `resident_layers`；否则丢弃并记一次警告，警告文本与 `legacy` 相同：`resident_layers=N was requested, but this model declares no matching resident_dit_paths; all blocks will be streamed.`

```python
LEGACY_RESIDENT_DIT_MODELS: frozenset[str] = frozenset({
    "MiniMaxH3Pipeline", "MiniMaxH3ModularPipeline", "Kandinsky6TI2VAPipeline",
})   # DiffusionModelRegistry 的注册名
```

这张表是冻结的兼容数据。测试从 `_DIFFUSION_MODELS` 的每个注册名解析出 pipeline 类所在的源文件，用语法树找出构造 `OffloadPlan(..., resident_dit_paths=...)` 的类，断言表等于这些类的注册名集合。

#### 6.4 五个自己挪动 VAE 前后权重的模型

MiniMax H3、Wan2.2-S2V、MAGI-2、JoyImage 自己在 VAE 使用前后挪走 DiT 或常驻层，LingBot-Video 按请求参数 `offload_vae_during_denoise` 在去噪期间把 VAE 挪到 CPU（总体设计 8.4）。`unified` 下这些代码不起作用，旧开关又不选 `vae`，显存峰值高于 `legacy`。

```python
LEGACY_VAE_LIFECYCLE_MODELS: Mapping[str, str] = {    # 注册名到 recipe 路径
    "MiniMaxH3Pipeline": "recipes/MiniMaxAI/MiniMax-H3.md",
    "MiniMaxH3ModularPipeline": "recipes/MiniMaxAI/MiniMax-H3.md",
    "WanS2VPipeline": "recipes/Wan-AI/Wan2.2-S2V.md",
    "Magi2Pipeline": "recipes/SandAI/MAGI-2-preview-L20X.md",
    "JoyImageEditPipeline": "recipes/JD/JoyAI-Image-Edit.md",
    "LingBotVideoPipeline": "recipes/Robbyant/LingBot-Video.md",
}
```

- `startup_policy` 发现策略来自旧开关或旧配置、`model_class_name` 在这张表里时，在 0 号 rank 记一条 `WARNING` 启动日志（第 14 节的 `legacy_vae_hint` 事件），写明这个模型在 `unified` 下不再在 VAE 前后挪动权重，并给出 recipe 里选中 `vae` 的自定义配置。
- `offload_vae_during_denoise` 在 `unified` 下忽略并记一次警告，由 04 承担的必改行 C08 实现；本 RFC 只在文档里写明。

#### 6.5 新旧写法同时出现

- 旧开关或旧配置与 `diffusion_offload`、`diffusion_offload_policy` 同时出现时（不论来自哪个来源），比较两份策略：相等时采用新写法并记一次警告，提示删除旧设置；不相等时报 `P19`。
- 旧配置与旧开关同时出现时沿用 `legacy` 的规则（`P33`）：策略种类不同时报错，相同时采用旧配置。
- `pin_memory` 与 `pin_cpu_memory`：字段 `pin_cpu_memory` 的默认值是 `True`，取 `False` 必定是用户给出的。规则如下，`legacy` 下旧配置的 `pin_memory` 照旧优先：

| `diffusion_offload_config.pin_memory` | `pin_cpu_memory` | `unified` 下的结果 |
|---|---|---|
| 没有给出 | 任意 | 取 `pin_cpu_memory`。 |
| `false` | `True` | 取 `false`，即旧配置的翻译。 |
| `false` | `False` | 取 `false`。 |
| `true` | `True` | 取 `true`。 |
| `true` | `False` | 报 `P21`。 |

#### 6.6 翻译测试与豁免

`tests/diffusion/weight_stack/policy/test_legacy_translation.py` 对 `DiffusionModelRegistry` 里每个能在只建结构的上下文里构造的 pipeline（02 的支持矩阵里结果为成功的），对每种旧开关组合（三个布尔开关各自单独出现，分布式逐层配 `--dlo-no-use-allgather`，旧配置的 `module` 和 `layer` 两种 `mode`）做：

```text
legacy 一支：用同一份配置在 legacy 下构造 pipeline，取 ModuleDiscovery.discover(pipeline) 和 get_offload_plan(pipeline)
             算出现有实现会 offload 的 DiT 和编码器路径集合 L（编码器名单里属于 vae 角色的除外）
unified 一支：parse_unified_config 得到策略，用组件目录的 (名字, 角色) 调用 resolve_selection()，
             去掉 if_no_blocks 为 skip 且 04 的重复块发现结果为空的组件，得到选中组件的根路径集合 U
断言 U == L
```

- 豁免：`LEGACY_VAE_LIFECYCLE_MODELS` 里的注册名不参与 `U == L` 的断言，测试记录豁免原因“the legacy pipeline moves DiT or resident layers around VAE use; the recipe selects vae with a custom policy”，并断言 `startup_policy` 对它们输出 `legacy_vae_hint` 事件。豁免表就是这张映射，不另设。
- 等价测试 `test_legacy_schema_equivalence.py`：用一份覆盖每条规则的输入语料，断言 `policy.parse_legacy_offload_config` 和 `offloader.config.parse_diffusion_offload_config` 对每一项同时接受或同时拒绝，拒绝时异常类型和文本相同。
- 还不能在只建结构的上下文里构造的 pipeline 记为待定，测试日志列出它们。默认值翻转之前（umbrella Job 6），待定列表必须为空。

### 7. host 策略

| 设置 | 规则 |
|---|---|
| `pin_cpu_memory` | `HostPolicy.pin_cpu_memory`，按第 6.5 节的表确定。为假时 04 不查询平台能力、不做内存登记，暂存区不锁页（总体设计 8.3）。 |
| `host_registration_limit_gib` | 必须是有限的非负数，接受 `int`、`float` 和能转成数的字符串，不接受布尔值；大于 0 时要求策略不为 `None` 并且 `pin_cpu_memory` 为真，否则报 `P24`。换算成 `HostPolicy.registration_limit_bytes = floor(gib × 2**30)`。上限按每个 worker 计算，覆盖所有 Unit 和分片，按 04 对齐、合并之后的字节数统计。 |

- host 策略只经 `StartupContext.policy.host` 到达 Offloader。HWM 不读任何 host 策略：同机共享没有配置项（总体设计 8.3），后端由规划决定。
- 策略为 `None` 时没有 Host 驻留的 Unit，`pin_cpu_memory` 不起作用。
- `--pin-cpu-memory` / `--no-pin-cpu-memory` 写的是已有字段，在 `legacy` 下同样生效，含义与现有字段相同。

### 8. 校验规则

#### 8.1 规则表

参数解析阶段的规则都报 `ConfigError`（01 第 3 节），`stage_id` 取 `od_config.stage_id`，`rank` 和 `world_size` 为 `None`。规划阶段的规则报 `PlanningError`，由 04 的 `plan()` 经第 11 节的 `resolve_selection()` 抛出。

| 编号 | 规则 | 阶段 | 只在 `unified` 下检查 |
|---|---|---|---|
| P01 | `legacy` 下给出 `diffusion_offload`、`diffusion_offload_policy`、`max_lora_rank`，或 `host_registration_limit_gib` 大于 0。 | 参数解析 | 否，只在 `legacy` 下检查 |
| P02 | `diffusion_offload` 不是四个预设之一。 | 参数解析 | 是 |
| P03 | 同一来源里同时给出 `diffusion_offload` 和 `diffusion_offload_policy`。 | 参数解析 | 是 |
| P04 | `diffusion_offload_policy` 不是映射，或字符串不是合法的 JSON。 | 参数解析 | 是 |
| P05 | 顶层有 `mode`、`components` 以外的字段。 | 参数解析 | 是 |
| P06 | 没有 `components`，或它不是非空映射。 | 参数解析 | 是 |
| P07 | `components` 的键不是非空字符串。 | 参数解析 | 是 |
| P08 | `components` 的键是 `other`。 | 参数解析 | 是 |
| P09 | 设置里有 `mode`、`weight_transfer`、`resident_layers`、`units` 以外的字段，或设置既不是映射也不是 `null`。 | 参数解析 | 是 |
| P10 | `mode` 不是 `module` 或 `layer`。 | 参数解析 | 是 |
| P11 | `weight_transfer` 不是 `rank-local` 或 `allgather`。 | 参数解析 | 是 |
| P12 | `resident_layers` 不是非负整数（布尔值也报错）。 | 参数解析 | 是 |
| P13 | `units` 不是非空字符串，也不是非空字符串组成的非空列表。 | 参数解析 | 是 |
| P14 | 被选中的设置没有 `mode`，顶层也没有 `mode`。 | 参数解析 | 是 |
| P15 | `module` 配 `allgather`。 | 参数解析 | 是 |
| P16 | `resident_layers` 大于 0 配 `allgather`。 | 参数解析 | 是 |
| P17 | `units` 配 `module`。 | 参数解析 | 是 |
| P18 | `units` 配 `resident_layers` 大于 0。 | 参数解析 | 是 |
| P19 | 新旧写法同时出现且翻译结果不同。 | 参数解析 | 是 |
| P20 | `dlo_host_registration_limit_gib` 大于 0。 | 参数解析 | 是 |
| P21 | 旧配置 `pin_memory` 为真而 `pin_cpu_memory` 为假。 | 参数解析 | 是 |
| P22 | `host_weight_runtime_mode` 不是 `disabled`，或给出了 `host_weight_runtime_root`。 | 参数解析 | 是 |
| P23 | `cache_config` 里 `minimax_h3_adaln_offload` 为真。 | 参数解析 | 是 |
| P24 | `host_registration_limit_gib` 类型或取值非法，或大于 0 而策略为 `None` 或 `pin_cpu_memory` 为假。 | 参数解析 | 是 |
| P25 | `data_parallel_size` 或 `cfg_parallel_size` 大于 1、有选择器用 `allgather`、`cache_backend` 不是 `None` 或 `"none"`。 | 参数解析 | 是 |
| P26 | `sequence_parallel_size` 大于 1、有选择器用 `allgather`、`cache_backend` 已启用且不在 `SP_SYNCED_CACHE_BACKENDS` 里。 | 参数解析 | 是 |
| P27 | `data_parallel_size` 大于 1、`text_encoder` 角色键用 `allgather`、`enable_prompt_embed_cache` 为真（环境变量折叠之后）。 | 参数解析 | 是 |
| P28 | `hsdp_weight_load_strategy` 为 `pre_sharded`，并且给出了 `lora_path`、`max_lora_rank`，或 `quantization_config` 不为 `None`。 | 参数解析 | 是 |
| P29 | `diffusion_compile_granularity` 为 `full` 并且策略不为 `None`。 | 参数解析 | 是 |
| P30 | `max_lora_rank` 不是整数，或不在合法值里。 | 参数解析 | 是 |
| P31 | `OMNI_DIFFUSION_PROMPT_EMBED_CACHE` 或 `OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE` 的取值非法。 | 参数解析 | 是 |
| P32 | 旧配置的格式错误，规则与 `parse_diffusion_offload_config` 相同。 | 参数解析 | 是（`legacy` 下由现有代码检查） |
| P33 | 旧配置与旧开关的组合错误，规则与 `_validate_legacy_layer_options`、`resolve_offload` 相同。 | 参数解析 | 是（`legacy` 下由现有代码检查） |
| L01 | `if_absent` 为 `error` 的键在本副本的任何 rank 上都没有匹配的组件。 | 规划 | 是 |
| L02 | 同一角色下被选中的 `layer` 组件给出了不同的非零 `resident_layers`。 | 规划 | 是 |
| L03 | 被 HSDP 管理的组件被选中。 | 规划 | 是 |
| L04 | 组件名键匹配到 `other` 角色的组件。 | 规划 | 是 |
| L05 | `data_parallel_size` 大于 1、`enable_prompt_embed_cache` 为真、组件名键匹配到 `text_encoder` 角色的组件并用 `allgather`。 | 规划 | 是 |

- `SP_SYNCED_CACHE_BACKENDS = frozenset({"tea_cache", "sea_cache"})`：TeaCache 在 `teacache/extractors.py` 和 `teacache/hook.py` 里对 SP 组做 `all_reduce`，SeaCache 在 `seacache/hook.py` 里对 SP 组做 `all_reduce`。`cache_dit`、`mag_cache`、`step_cache` 不在 SP 组上同步跳块的决定，按不同步处理。把一个后端加进这张表的 PR 必须同时加一个多进程测试，断言 SP 组内各 rank 的跳块决定逐步相同。
- P27 在参数解析阶段只看 `text_encoder` 角色键，因为那时不知道组件名属于哪个角色；组件名键的情况由 L05 在规划阶段检查，报错文本相同。
- P29 不另起一条报错：`unified` 下 `OmniDiffusionConfig.__post_init__` 的全量编译检查把 `"diffusion offload"` 追加到已有的 `incompatible_features` 列表里，报错文本沿用现有格式。`legacy` 下这段检查不变。
- P01 让新参数在 `legacy` 下不被静默忽略；`pin_cpu_memory` 是已有字段，不在 P01 里。

#### 8.2 报错文本

报错按 01 第 3 节的格式输出，下面是 `what` 和 `fix` 两段。花括号是代入的值，`{key}` 是 `components` 里的键。

```text
P01 what: {flag} requires --diffusion-weight-stack unified
    fix:  pass --diffusion-weight-stack unified, or remove {flag}
P02 what: unknown --diffusion-offload preset {value!r}; choose from: none, module, layerwise, distributed-layerwise
P03 what: --diffusion-offload and --diffusion-offload-policy are mutually exclusive within one config source
    fix:  keep only one of them
P04 what: diffusion_offload_policy must be a JSON object or a mapping, got {type_name}
    what: diffusion_offload_policy is not valid JSON: {json_error}
P05 what: unknown diffusion_offload_policy field(s): {names}; choose from: components, mode
P06 what: diffusion_offload_policy requires a non-empty 'components' mapping
    fix:  use --diffusion-offload none to disable offload
P07 what: diffusion_offload_policy.components keys must be non-empty strings
P08 what: role 'other' cannot be selected for offload; components with role other stay on the accelerator
P09 what: unknown setting(s) {names} for components.{key}; choose from: mode, resident_layers, units, weight_transfer
    what: components.{key} must be a mapping or null, got {type_name}
P10 what: components.{key}.mode must be one of: layer, module; got {value!r}
P11 what: components.{key}.weight_transfer must be one of: allgather, rank-local; got {value!r}
P12 what: components.{key}.resident_layers must be a non-negative integer; got {value!r}
P13 what: components.{key}.units must be a non-empty string or a non-empty list of non-empty strings
P14 what: components.{key} has no mode
    fix:  set components.{key}.mode or the top-level mode
P15 what: components.{key}: mode='module' cannot use weight_transfer='allgather'; AllGather is used only by streamed blocks
    fix:  use mode='layer', or weight_transfer='rank-local'
P16 what: components.{key}: resident_layers={n} requires weight_transfer='rank-local'
P17 what: components.{key}: units requires mode='layer'
P18 what: components.{key}: units cannot be combined with resident_layers
P19 what: legacy offload option(s) {fields} translate to {legacy_canonical}, which conflicts with {new_flag}={new_canonical}
    fix:  remove the legacy option(s)
P20 what: --dlo-host-registration-limit-gib is not used by --diffusion-weight-stack unified
    fix:  use --diffusion-host-registration-limit-gib {value}
P21 what: diffusion_offload_config.pin_memory=true conflicts with pin_cpu_memory=false
    fix:  remove pin_memory from --diffusion-offload-config and use --pin-cpu-memory or --no-pin-cpu-memory
P22 what: Host Weight Runtime options (host_weight_runtime_mode={mode!r}, host_weight_runtime_root={root!r}) are not used
          by --diffusion-weight-stack unified; host weights are built from the checkpoint at every start and shared
          within one deployment automatically
    fix:  remove --host-weight-runtime-mode and --host-weight-runtime-root
P23 what: cache_config.minimax_h3_adaln_offload=true is replaced by the offload policy under --diffusion-weight-stack unified
    fix:  use --diffusion-offload-policy '{"components": {"dit": {"mode": "layer", "units": "adaln"}}}'
P24 what: host_registration_limit_gib must be a finite number >= 0; got {value!r}
    what: host_registration_limit_gib={value} requires an offload policy and pin_cpu_memory=true
P25 what: cache acceleration (cache_backend={backend!r}) cannot be combined with weight_transfer='allgather' when
          data_parallel_size={dp} or cfg_parallel_size={cfg} is greater than 1: ranks would skip different blocks and
          the AllGather order would diverge
    fix:  set --cache-backend none, or use weight_transfer='rank-local'
P26 what: cache_backend={backend!r} does not synchronize block-skip decisions across the sequence-parallel group
          (sequence_parallel_size={sp}) and cannot be combined with weight_transfer='allgather'
    fix:  use weight_transfer='rank-local', or a cache backend that synchronizes skips across the SP group: sea_cache, tea_cache
P27 what: prompt embedding cache cannot be combined with text_encoder weight_transfer='allgather' when
          data_parallel_size={dp} > 1: rank-local cache hits would skip different encoder collectives
    fix:  use weight_transfer='rank-local' for the text encoder, or disable enable_prompt_embed_cache
P28 what: hsdp_weight_load_strategy='pre_sharded' does not support {feature}
          ({feature} is one of: LoRA (lora_path), dynamic LoRA (max_lora_rank), quantization ({method}))
    fix:  use --hsdp-weight-load-strategy full
P29 what: diffusion_compile_granularity='full' is incompatible with {features}; use 'regional' compilation instead
P30 what: --diffusion-max-lora-rank must be one of {values}; got {value!r}
P31 what: OMNI_DIFFUSION_PROMPT_EMBED_CACHE={value!r} is not a boolean; use one of 1, 0, true, false, yes, no, on, off
    what: OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE={value!r} must be a positive integer
L01 what: offload policy key {key!r} matches no component of {pipeline} on any rank; declared: {name (role), ...}
    fix:  remove the key, or use one of the declared component names or roles
L02 what: role {role!r} has different resident_layers {values} on keys {keys}; resident layers are counted per role
    fix:  give resident_layers once, on the role key
L03 what: component {name!r} is managed by HSDP and cannot be offloaded under --diffusion-weight-stack unified
    fix:  either keep HSDP and remove {key!r} from the offload policy, or disable HSDP and use
          '{"components": {"{name}": {"mode": "layer", "weight_transfer": "allgather"}}}'
L04 what: component {name!r} has role 'other' and cannot be selected by key {key!r}
L05 what: (same as P27, with "text encoder component {name!r}")
```

P32、P33 的文本与 `offloader/config.py` 里对应的报错逐字相同，由第 6.6 节的等价测试保证。

### 9. 动态 LoRA 的配置

| 设置 | 规则 |
|---|---|
| `max_lora_rank` | `int`，不接受布尔值；合法值是 `typing.get_args(vllm.config.lora.MaxLoRARanks)`，与 `DiffusionLoRAManager._VALID_MAX_RANKS` 同源。非法时报 `P30`。`unified` 下给出时启用动态 LoRA，02 按它分配叠加层槽位（02 第 11.1 节）。`legacy` 下给出时报 `P01`。 |
| `max_cpu_loras` | 已有字段，校验（`None` 取 1，小于 1 报错）和含义不变。 |
| `lora_backend=peft` 加 `--lora-path`，没给 `max_lora_rank` | 参数解析阶段不读适配器文件；槽位的 rank 由 02 在只建结构时从适配器的头部取得并向上取到合法值（02 第 11.1 节）。 |
| 都没给 | 动态 LoRA 不启用。运行期 `add_lora` 由 02 的管理器拒绝，`fix` 写 `restart with --diffusion-max-lora-rank <rank>`。 |

- `lora_backend=distill` 可以与 `max_lora_rank` 同时给出：动态 LoRA 叠在融合了适配器的基础权重上（总体设计 5.6）。
- 现有 `serve.py` 对 `--lora-backend` 和 `--lora-path` 的检查（`distill` 要求 `--lora-path`，`peft` 只接受一个启动路径）两个栈都执行，不变。
- `max_lora_rank`、`lora_path` 与 HSDP 预分片的组合由 `P28` 拒绝。

### 10. 环境变量的折叠

`unified` 下，`parse_unified_config` 第一步折叠两个环境变量，之后的组合检查都读折叠后的字段：

```text
v = os.environ.get("OMNI_DIFFUSION_PROMPT_EMBED_CACHE")
  v 不为 None：小写去空格后属于 {1, true, yes, on} 时 enable_prompt_embed_cache = True；
              属于 {0, false, no, off} 时为 False；其他取值报 P31
s = os.environ.get("OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE")
  s 不为 None：能转成正整数时 prompt_embed_cache_size = int(s)；否则报 P31
```

- 环境变量优先于配置里的字段，与 `resolve_prompt_embed_cache_config` 的优先级相同。`legacy` 不经过这一步，worker 照旧调用 `resolve_prompt_embed_cache_config`，非法取值照旧被忽略。
- 折叠在构建配置的进程里做：编排进程或 headless 启动器。之后两个值作为普通字段随 `od_config` 下发并冻结；`unified` 一支的 `_load_model_unified` 只读字段（01 第 8.1 节），worker 不读这两个环境变量。
- 同一个进程里由投影重建配置时折叠再做一次，结果相同。
- 这一步取代 01 第 1.3 节里“设置了这两个环境变量就报错”的临时检查。

### 11. 选择的解析

04 的 `plan()` 用组件目录调用 `resolve_selection()`，得到本 rank 上每个被选中的组件和它的设置。这个函数是“角色键、组件名键、`null`、`if_absent`、`if_no_blocks`”这些规则的唯一实现。

```python
@dataclass(frozen=True)
class ComponentRef:
    name: str
    role: str                    # ALL_ROLES 之一
    hsdp_managed: bool

@dataclass(frozen=True)
class SelectedComponent:
    name: str
    role: str
    key: str                     # 选中它的键
    setting: ComponentSetting
    if_no_blocks: Literal["error", "skip"]

@dataclass(frozen=True)
class Selection:
    selected: tuple[SelectedComponent, ...]     # 按 refs 的顺序，即组件声明的顺序
    resident_layers: Mapping[str, int]          # 角色到常驻层数，只含大于 0 的角色
    unmatched: tuple[str, ...]                  # if_absent 为 error、在本 rank 上没有匹配的键，排序

def resolve_selection(policy: OffloadPolicy, refs: Sequence[ComponentRef], *, pipeline: str) -> Selection: ...
```

```text
对 refs 里的每个组件 c：
  有组件名键 k，k.key == c.name：
    c.role == "other" 时报 L04
    k.setting 为 None：不选 c；否则选 c，设置取 k.setting
  否则有角色键 r，r.key == c.role：
    r.setting 为 None：不选 c；否则选 c，设置取 r.setting
  否则：不选 c
  选中 c 且 c.hsdp_managed：报 L03
常驻层数：对每个角色，取被选中、mode 为 layer、weight_transfer 为 rank-local 的组件的非零 resident_layers；
  取值不止一个时报 L02；只有一个时就是这个角色的常驻层数 N
unmatched：if_absent 为 error 的键，组件名键没有同名组件、角色键没有这个角色的组件时列入
L05：选中的组件角色为 text_encoder、用 allgather、由组件名键选中，并且 DP 大于 1、启用了 prompt embedding 缓存时报错
```

- `resident_layers` 的计数方式（同一角色下按声明顺序和 `repeated_blocks` 的顺序排成一列、取前 N 个）和“N 不小于总块数时全部常驻并记日志”由 04 按总体设计 7.2 实现；本函数只给出每个角色的 N。
- `if_no_blocks` 为 `skip` 的组件找不到重复块时，04 不选它并记一条 `component_skipped` 启动日志（第 14 节）；为 `error` 时 04 报 `PlanningError`。
- `unmatched` 的判断按本 rank 的目录进行。组件可以按 rank 声明（02 第 2.5 节），L01 要求“在本副本的任何 rank 上都没有匹配”才报错，跨 rank 的汇总方式见“与其他子 RFC 的接口”里对 01 和 04 的约定。

### 12. 引擎和执行器读到的 offload 事实

`OffloadPolicy.execution_facts()` 给出 01 第 9.2 节 `ExecutionOffloadFacts` 的 `unified` 取值，只读策略：

| 字段 | 取值 |
|---|---|
| `weight_collectives` | 有任何选择器的设置用 `allgather` 时为真。 |
| `text_encoder_allgather` | `text_encoder` 角色键用 `allgather`，或有任何组件名键用 `allgather` 时为真。组件名键的角色在这时还不知道，按可能属于文本编码器处理；多做的同步执行不影响正确性（01 第 9.2 节）。 |
| `whole_component_offload` | 有任何选择器的设置是 `module` 时为真。 |

策略为 `None` 时三个字段都为假。01 的 `execution_offload_facts(config)` 在 `unified` 一支调用 `policy.cached_policy(config)`，后者第一次调用时用 `parse_policy_document` 和翻译规则得到策略，结果缓存在配置对象的私有属性 `_weight_stack_offload_policy` 上（与 `legacy` 的 `_resolved_diffusion_offload` 缓存方式相同，属性写不进去时不缓存），之后的调用直接返回。这个函数不读环境变量，不记日志。

### 13. 入口的位置

```text
_DiffusionConfigProjection.__post_init__
  unified：在 validate_host_weight_runtime_options、validate_dlo_host_registration_options 之前调用
           policy.check_migration(self)（P20、P22），让迁移提示先于旧校验的报错            D22
OmniDiffusionConfig.__post_init__
  parse_weight_stack()（01）
  parallel_config.resolve_data_parallel_size(...)（不变）
  legacy：materialize_legacy_offload_flags(self)（不变）；policy.reject_unified_only_fields(self)（P01）
  unified：parse_unified_config(self)：环境变量折叠（第 10 节），P02 到 P27、P30 到 P33，结果写进缓存（第 12 节）
  全量编译检查：unified 下策略不为 None 时向 incompatible_features 追加 "diffusion offload"（P29）
  量化解析、max_cpu_loras、validate_host_weight_runtime_options、validate_dlo_host_registration_options（不变）
  unified：policy.check_late(self)（P28，量化解析之后）
OmniDiffusionConfig.enrich_config 的末尾
  unified：policy.check_late(self)（P28；enrich_config 经 set_tf_model_config 可能自动探测出量化）       D20
DiffusionModelRunner._load_model_unified（01 第 8.1 节）
  ctx = StartupContext(..., policy=startup_policy(self.od_config, log=(rank == 0)), ...)
```

- `parse_unified_config` 放在 `legacy` 调用 `materialize_legacy_offload_flags` 的位置，因为 P25 到 P27 要读 DP 归一化之后的 `data_parallel_size`，与 `resolve_offload` 的时机相同。这一步取代 01 第 1.3 节的临时拒绝。
- `startup_policy(config, *, log)` 用缓存或重新解析得到策略（不读环境变量），调用 `resolve_for_model(config.model_class_name)`；`log` 为真时记 `policy`、`legacy_vae_hint`、`legacy_resident_ignored` 三类启动日志（第 14 节）。`model_class_name` 在 `enrich_config` 之后已经确定，`DiffusionModelRegistry` 按它构造 pipeline。
- D20、D21、D22 是本 RFC 的分发点，在登记表里用 `NEW_ROW` 声明（01 第 9.4 节）；`__post_init__` 里的改动在 01 的 D01 已登记的位置之内。

### 14. 校验分层

总体设计 8.6 的每一层在本栈里的落实如下。每一层都在自己那一步的第一次不可逆修改之前完成。

| 阶段 | 内容 | 规则和错误类型 |
|---|---|---|
| 参数解析 | 类型、未知字段、互斥项、新旧冲突、迁移报错、不支持的组合、环境变量折叠 | 本 RFC 的 P01 到 P33，`ConfigError`；01 的总开关取值检查。 |
| 构造 | 只建结构、读取陷阱、残留 meta 张量、保持 fp32 的层不量化、未迁移的 pipeline | 02 第 3、4 节，`ConstructionError`。 |
| Offloader 规划 | 组件完整、选择器匹配、`unit_groups` 存在、重复块可发现、常驻层数合法、HSDP 组件不被选中、共用存储的生命周期一致、分片后端下重复块以外的组件单机放得下 | 本 RFC 的 L01 到 L05 经 `resolve_selection()`；其余由 04 检查；`PlanningError`。 |
| Loader 预检 | 来源可读、内容身份、冷启动路径、适配器的键、`lora_scale`、已量化的 checkpoint 不融合 | 02 第 8.6 节，`PreflightError`。 |
| 全组确认第 1 点 | 参与集合通信的部分各 rank 一致、AllGather 组的成员、单机 Host 容量 | 01 第 7.2 节和 03 第 10 节，`GroupConfirmationError`、`HostCapacityError`。 |
| `prepare_materialization()` | 加速卡和 Host 的运行期资源准入、内存登记或暂存区的计划、多 rank 的全组确认 | 04，`PrepareError`、`DeviceCapacityError`。 |

### 15. 启动日志

#### 15.1 格式

```text
[weight-stack][stage <stage_id>][rank <rank>/<world_size>][startup] <event> key=value key=value ...
```

- 每个事件一行。值里有空格、等号或引号时写成 JSON 字符串；列表写成逗号分隔、不带空格的形式。
- `observability.py` 提供 `bind_context(stage_id, rank, world_size)`（runner 在 `_load_model_unified` 开头调用一次）、`format_event(event, **fields) -> str` 和 `log_event(logger, event, *, level=logging.INFO, **fields)`。各模块用自己的 logger：`vllm_omni.diffusion.weight_stack.policy`、`.loader`、`.host`、`.offload`、`.startup`。
- `observability.py` 只 import 标准库和 `weight_stack.errors`，`loader/`、`host/`、`offload/`、`policy.py` 都可以 import 它。

#### 15.2 事件

| 事件 | 发出方 | 字段 |
|---|---|---|
| `policy` | 05 | `kind`、`preset`、`legacy_fields`、`canonical`、`pin_cpu_memory`、`registration_limit_bytes`。 |
| `legacy_vae_hint` | 05，`WARNING` | `model`、`recipe`、`suggested`（选中 `vae` 的规范 JSON）。 |
| `legacy_resident_ignored` | 05，`WARNING` | `model`、`resident_layers`。 |
| `component_plan` | 04 | `component`、`role`、`key`、`mode`、`transfer`、`units`（Unit 数）、`unit_kinds`、`trigger`、`backend`（`local`、`shard`、`none`）、`cold_path`（`thin`、`full_then_trim`、`none`）。 |
| `component_skipped` | 04 | `component`、`reason`（例如 `no_repeated_blocks`）。 |
| `unit_resident` | 04 | `component`、`unit`、`reason`（`no_yielding_component` 或 `resident_layers_cover_all_blocks`）。 |
| `host_share` | 03 | `component`、`shared`、`reason`、`role`（`writer`、`waiter`、`private`）、`writer_pid`、`socket`。 |
| `host_admission` | 03 | `machine`、新增占用的每一项、`available_bytes`、`memory_source`。 |
| `unit_transfer` | 04 | `unit`、`path`（`registered` 或 `staging`）、`reason`（`limit`、`unsupported`、`failed`、`pin_disabled`）、`bytes`。 |
| `fully_resident` | 04 | `component`、`value`、`compile_mode`、`downgraded_from`（编译模式被降成 `default` 时给出原模式）。 |
| `startup_summary` | 05，`IN_SERVICE` 之前 | `host_private_bytes`、`host_shared_bytes`、`host_shard_bytes`、`registered_bytes`、`staging_bytes`、`device_slot_bytes`、`startup_s`。 |

- 03 第 14 节和 04 的启动日志按这张表的事件名和字段输出。03、04 的 PR 先于 05-6 合入时用各自的格式，05-6 把这些调用点改成 `log_event`，事件名和字段不变。
- `startup_summary` 由 runner 在 `run_unified_startup` 返回之后、`mark_in_service()` 之前调用 `observability.log_startup_summary(handle)` 输出。

### 16. 指标

#### 16.1 指标族

umbrella RFC“可观测性”一节的五项分别报告，不相加。名字放在 `vllm_omni/metrics/definitions.py`，指标族在 `vllm_omni/metrics/prometheus.py` 里随 `OmniPrometheusMetrics` 在 import 时注册，`--log-stats` 关闭时设置函数直接返回，与已有指标族相同。

| 指标 | 类型 | 标签 | 含义 |
|---|---|---|---|
| `vllm_omni:diffusion_worker_pss_bytes` | Gauge | `model_name`、`stage`、`replica`、`rank` | worker 进程按共享比例平摊后的内存占用，取 `/proc/self/smaps_rollup` 的 `Pss`。共享内存文件的页按映射它的进程数平摊。 |
| `vllm_omni:diffusion_machine_pss_bytes` | Gauge | `model_name`、`machine` | 本部署在一台机器上全部扩散 worker 的 `Pss` 之和。`machine` 是主机名，分组按 `boot_id`。 |
| `vllm_omni:diffusion_host_storage_bytes` | Gauge | `model_name`、`stage`、`replica`、`rank`、`kind` | 本 worker 映射的 host 存储字节数，`kind` 取 03 的 `Mapping.kind`：`private`、`shared`、`shard`。`shared` 是映射长度，不平摊。 |
| `vllm_omni:diffusion_host_registered_bytes` | Gauge | `model_name`、`stage`、`replica`、`rank` | 已登记内存的字节数，按 04 对齐、合并之后的区间统计。 |
| `vllm_omni:diffusion_staging_bytes` | Gauge | `model_name`、`stage`、`replica`、`rank` | 暂存区的字节数。 |
| `vllm_omni:diffusion_device_slot_bytes` | Gauge | `model_name`、`stage`、`replica`、`rank`、`slot` | 加速卡固定槽位的字节数，`slot` 取 `ring_rank_local`、`ring_allgather`、`shard`、`overlay`。 |

- 按进程 RSS 求和会把共享页算进每个映射它的进程（总体设计第 13 节），所以机器级只报 `Pss` 之和。`diffusion_host_storage_bytes{kind="shared"}` 跨 worker 相加会重复计数，文档写明。
- 标签的取值都有上限：`rank` 不超过副本的 world size，`kind` 和 `slot` 是固定集合。

#### 16.2 采集路径

```text
worker（每个 rank）
  03 在 allocate()、seal() 之后调用 observability.record_host_mapping(mapping)，HostStorage.close() 时 forget_host_mapping(mapping)
  04 在 commit() 之后调用 observability.record_device_resources(registered=..., staging=..., slots={...})，
     OffloadRuntime.close() 时清零
  DiffusionWorker.weight_stack_memory_report() -> WeightStackMemoryReport | None
    legacy 或启动未完成时返回 None；否则汇总上面的登记，并读一次 /proc/self/smaps_rollup
DiffusionEngine（只在 unified 下，D21）
  启动完成后一次，之后由 _process_rpc_queue 所在的 busy loop 在两批之间每 30 秒最多一次：
    reports = executor.collective_rpc("weight_stack_memory_report")
    与上一次不同时，发出一个 request_id 为 DIFFUSION_METRICS_ONLY_REQUEST_ID 的输出，
    metrics 里带 DIFFUSION_WEIGHT_STACK_MEMORY_KEY = "weight_stack_memory"：[report.to_dict(), ...]
Orchestrator._absorb_diffusion_metrics
  弹出这个键，调用 _update_stage_replica_weight_memory(stage_id, replica_id, reports)：
    按 (stage, replica, rank) 设置五个 Gauge；按 boot_id 对本部署所有副本的最新报告求和，设置机器级 Gauge
  _handle_dead_replica：删除这个副本的序列并重算机器级 Gauge，与 _remove_stage_replica_waiting 的处理方式相同
```

```python
# vllm_omni/diffusion/weight_stack/observability.py
@dataclass(frozen=True)
class WeightStackMemoryReport:
    rank: int
    boot_id: str
    machine: str
    pss_bytes: int | None                 # smaps_rollup 读不到时为 None，不导出这一项
    host_bytes: Mapping[str, int]         # "private"、"shared"、"shard"
    registered_bytes: int
    staging_bytes: int
    device_slot_bytes: Mapping[str, int]  # "ring_rank_local"、"ring_allgather"、"shard"、"overlay"
    def to_dict(self) -> dict[str, object]: ...
```

`docs/design/metrics.md` 增加一节“Diffusion weight stack memory”和第六条采集路径，内容是上面两张表。

### 17. recipe、文档与支持矩阵

#### 17.1 recipe

recipe 里的自定义配置只列出这个模型组件声明里存在的角色或组件名，否则规划阶段报 L01；05-7 的 recipe 测试在只建结构的上下文里解析每条命令并调用 `resolve_selection()`，断言 `unmatched` 为空。

| recipe | 改动 |
|---|---|
| `recipes/MiniMaxAI/MiniMax-H3.md`（“Default AdaLN weight offload”一节和开头部署说明里 SM120 的 AdaLN offload） | `unified` 下 AdaLN offload 不再自动开启；SM120 的命令写明 `--diffusion-weight-stack unified --diffusion-offload-policy '{"components": {"dit": {"mode": "layer", "units": "adaln"}}}'`；`minimax_h3_adaln_offload` 标为只用于 `legacy`；“放不下时用整组件 offload”的建议改成旧开关翻译结果加上 `vae` 的自定义配置。 |
| `recipes/MiniMaxAI/MiniMax-H3-RTX-PRO-6000.md` 和其他 SM120 硬件上的 MiniMax H3 recipe | 同上，写明 `units` 为 `adaln`。 |
| `recipes/Wan-AI/Wan2.2-S2V.md` | 每条 `--enable-cpu-offload` 命令增加 `unified` 写法：旧开关的翻译结果加上 `"vae": {"mode": "module"}`。 |
| `recipes/SandAI/MAGI-2-preview-L20X.md` | 内存窗口由配置给出：`{"components": {"dit": {"mode": "module", "weight_transfer": "rank-local"}, "vae": {"mode": "module"}}}`；说明 `unified` 下没有“单独 `--enable-cpu-offload` 被拒绝”的限制。 |
| `recipes/JD/JoyAI-Image-Edit.md` | `--enable-cpu-offload` 和 `--enable-layerwise-offload` 两条命令各自增加选中 `vae` 的 `unified` 写法。 |
| `recipes/Robbyant/LingBot-Video.md` | 增加 `unified` 下选中 `vae` 的写法，说明请求参数 `offload_vae_during_denoise` 在 `unified` 下忽略。 |

#### 17.2 文档

| 页面 | 改动 |
|---|---|
| `docs/user_guide/diffusion/cpu_offload.md`、`docs/user_guide/diffusion/offloader/module_offload.md`、`layerwise_offload.md`、`distributed_layerwise_offload.md` | 增加 `unified` 一节：预设、自定义配置、JSON schema、第 3.2 节的示例、host 策略参数、旧开关的翻译表。 |
| `docs/user_guide/diffusion/startup_and_loading.md` | 总开关、启动日志的事件表、同机共享的条件和日志里的角色。 |
| `docs/user_guide/diffusion/lora.md` | `--diffusion-max-lora-rank`、合法值和第 9 节的规则。 |
| `docs/user_guide/diffusion_features.md` | 链接支持矩阵，写明非 CUDA 平台的 Unit 走暂存区（总体设计第 13 节）。 |
| `docs/design/metrics.md` | 第 16 节的指标。 |
| `docs/design/feature/offloader/README.md`、`docs/design/module/diffusion/offloader.md` | 指向总体设计和本 RFC，写明现有三个后端只用于 `legacy`。 |
| 新增 `docs/user_guide/diffusion/weight_stack_migration.md` | 迁移说明：第 6 节的翻译和迁移报错、总体设计第 9 节第 3 步列出的门槛例外、写入进程退出后新副本重建一份的影响（总体设计第 13 节）。 |
| 新增 `docs/user_guide/diffusion/weight_stack_support_matrix.md` | 支持矩阵，见第 17.3 节。 |

#### 17.3 支持矩阵

- 02 的 `test_structure_matrix.py` 为每个注册的 pipeline 用测试资源里的配置文件在只建结构的上下文里构造一次，把结果写成 `structure_matrix.json`（02 第 5.7 节）。
- `tools/weight_stack/render_support_matrix.py` 把它渲染成 `docs/user_guide/diffusion/weight_stack_support_matrix.md`：每个注册名一行，列出构造结果（成功、读取陷阱的位置、残留 meta 张量的位置、完整性违例、缺少的规则表）、组件名和角色，以及还没有验证、但没有正确性问题的组合（总体设计 7.7）。
- `tests/diffusion/weight_stack/policy/test_support_matrix_doc.py` 断言仓库里的文档等于用当前结果渲染的内容，改变支持情况的 PR 必须同时更新文档。翻转默认值之前，矩阵里不能有 `unified` 不支持的 pipeline（总体设计第 9 节第 4 步）。

## 与其他子 RFC 的接口

### 本 RFC 提供

| 提供的内容 | 使用方 | 约定 |
|---|---|---|
| `weight_stack.policy` 的 `OffloadPolicy`、`Selector`、`ComponentSetting`、`HostPolicy`、`Mode`、`WeightTransfer` | 01、04 | 第 2 节。冻结、可 pickle；`StartupContext.policy` 是 `startup_policy()` 的结果或 `None`。 |
| `resolve_selection(policy, refs, *, pipeline) -> Selection` | 04 | 第 11 节。04 的 `plan()` 用目录的组件名、角色和 `hsdp_managed` 调用，报 L02 到 L05；`unmatched` 交给 L01 的跨 rank 判断。 |
| `OffloadPolicy.execution_facts()`、`cached_policy(config)` | 01 的 `execution_offload_facts` | 第 12 节；只读策略，不读环境变量。 |
| `parse_unified_config`、`check_late`、`check_migration`、`reject_unified_only_fields` | `OmniDiffusionConfig`、`_DiffusionConfigProjection` | 第 13 节，取代 01 第 1.3 节的临时拒绝。 |
| `startup_policy(config, *, log)` | 01 的 `_load_model_unified` | 第 13 节。 |
| `HostPolicy` | 04 | `pin_cpu_memory` 为假时不查询平台能力、不登记，暂存区不锁页；`registration_limit_bytes` 为 0 表示不设上限。 |
| `observability` 的 `log_event`、`record_host_mapping`、`record_device_resources`、`WeightStackMemoryReport` | 02、03、04 | 第 15、16 节。 |
| 新字段列表 `diffusion_offload`、`diffusion_offload_policy`、`host_registration_limit_gib`、`max_lora_rank` | 01 的 T5 | 05-2 把它们加进 `test_static_dependencies.py` 的 T5 字段表。 |
| `max_lora_rank` 的解析和合法值、HSDP 预分片配 LoRA 或量化的拒绝、支持矩阵文档 | 02 | 第 9 节、P28、第 17.3 节。 |

### 本 RFC 依赖的约定

| 子 RFC | 约定 |
|---|---|
| [01](./01-startup-transaction.md) | `is_unified(config)`；`ConfigError`、`PlanningError` 和报错格式；`execution_offload_facts` 的 `unified` 一支调用 `cached_policy(config).execution_facts()`，策略为 `None` 时三个字段为假；`_load_model_unified` 把 `startup_policy()` 的结果填进 `StartupContext.policy`；登记表接受 D20、D21、D22 的 `NEW_ROW`；T1 允许 `loader/`、`host/`、`offload/`、`policy.py` import `observability.py`，`observability.py` 只 import 标准库和 `errors.py`。L01 的跨 rank 判断需要在全组确认第 1 点收集各 rank 的 `unmatched`，见“开放问题”。 |
| [02](./02-unified-loader.md) | 组件目录给出每个组件的名字、角色和 `hsdp_managed`；编码器的角色符合总体设计 8.4 的对应；`DynamicLoraSpec` 读 `od_config.max_lora_rank`；`test_structure_matrix.py` 输出 `structure_matrix.json`；动态 LoRA 未启用时 `add_lora` 的报错提示 `--diffusion-max-lora-rank`。 |
| [03](./03-host-storage.md) | `Mapping.kind` 取 `shared`、`private`、`shard`；`allocate()`、`seal()` 之后经 `record_host_mapping` 登记映射，`close()` 时注销；`host_share`、`host_admission` 事件按第 15.2 节输出。HWM 不读 host 策略。 |
| [04](./04-unified-offloader.md) | `plan()` 只经 `resolve_selection()` 解释策略，按 `if_no_blocks` 处理找不到重复块的组件，按 `Selection.resident_layers` 计数常驻层；`policy` 为 `None` 时不做 offload；从 `policy.host` 读 `pin_cpu_memory` 和登记上限；`commit()` 之后经 `record_device_resources` 登记字节数；第 15.2 节里 04 的事件；HSDP 组件的报错用 L03 的文本；`DeviceCapacityError` 的 `fix` 指向 recipe；承担 C08（`offload_vae_during_denoise` 忽略并记警告）。 |

## 代码组织

新增文件：

```text
vllm_omni/diffusion/weight_stack/policy.py            数据类型、schema、解析器、预设、兼容层、校验、resolve_selection、执行事实
vllm_omni/diffusion/weight_stack/observability.py     启动日志、资源字节登记、WeightStackMemoryReport
tools/weight_stack/render_support_matrix.py           支持矩阵渲染
docs/user_guide/diffusion/weight_stack_migration.md
docs/user_guide/diffusion/weight_stack_support_matrix.md
tests/diffusion/weight_stack/policy/                  第“测试计划”节的测试
tests/diffusion/weight_stack/observability/
tests/diffusion/weight_stack/coexistence/sites/D20_05_3.py、D22_05_3.py、D21_05_6.py
```

修改的文件：

| 文件 | 改动 |
|---|---|
| `vllm_omni/entrypoints/cli/serve.py` | 五个新参数。 |
| `vllm_omni/engine/arg_utils.py`、`vllm_omni/config/stage_config.py`、`vllm_omni/config/omni_config.py`、`vllm_omni/diffusion/data.py` | 新字段；`reconcile_diffusion_offload_overrides`、`diffusion_offload_policy` 的整体替换；第 13 节的入口。 |
| `vllm_omni/diffusion/worker/diffusion_model_runner.py` | `_load_model_unified` 里的 `startup_policy`、`bind_context`、`log_startup_summary`（都在 01 的 D03 一支之内）。 |
| `vllm_omni/diffusion/worker/diffusion_worker.py` | `weight_stack_memory_report()`。 |
| `vllm_omni/diffusion/diffusion_engine.py` | D21 的报告采集。 |
| `vllm_omni/engine/orchestrator.py`、`vllm_omni/metrics/definitions.py`、`vllm_omni/metrics/prometheus.py` | 第 16 节。 |
| `vllm_omni/diffusion/weight_stack/host/`、`offload/` 的日志调用点 | 05-6 改用 `log_event`，事件名和字段不变。 |
| `tests/diffusion/weight_stack/test_static_dependencies.py` | T5 字段表加入四个新字段（01 允许的改动）。 |
| `recipes/`、`docs/` | 第 17 节。 |

## 实现拆分

本 RFC 的 PR 都依赖 01-1 到 01-7 全部合入。每个 PR 只改变 `unified` 下的行为，`legacy` 下的行为和已有测试不变，不修改已有测试文件（T5 字段表除外，01 允许）。本 RFC 不承担共存期必改清单的 C 行，只承担自己的分发点 D20、D21、D22。

| PR | 内容 | 主要文件 | 测试 | 依赖 | 大小 | 必改行 |
|---|---|---|---|---|---|---|
| 05-1 | 数据类型、JSON schema、`parse_policy_document`、预设、规范 JSON、`resolve_selection`、旧配置的解析和旧开关的翻译（纯函数）、P02 到 P18、L01 到 L05 的文本。 | `policy.py` | `test_policy_parse.py`、`test_resolve_selection.py`、`test_legacy_schema_equivalence.py` | 01-1 到 01-7 | M | 无 |
| 05-2 | 新字段和五个参数的传递链、来源优先级和整体替换、P01、T5 字段表。 | `serve.py`、`arg_utils.py`、`stage_config.py`、`omni_config.py`、`data.py` | `test_policy_sources.py`、`test_policy_legacy_stack.py` | 05-1 | M | 无 |
| 05-3 | `parse_unified_config`、`check_late`、`check_migration`、环境变量折叠、P19 到 P33、全量编译检查、`startup_policy`、`resolve_for_model`、`cached_policy` 和执行事实；取代 01 第 1.3 节的临时拒绝；D20、D22。 | `policy.py`、`data.py`、`omni_config.py`、`diffusion_model_runner.py` | `test_policy_checks.py`、`test_env_folding.py`、`test_execution_facts_unified.py` | 05-2，以及 04 里实现规划、两种 Trigger、两种 Transport 和 `commit()` 的 PR | L | D20、D22 |
| 05-4 | `--diffusion-max-lora-rank` 的检查（P30）和 HSDP 预分片的 LoRA 部分。 | `policy.py` | `test_lora_rank_flag.py` | 05-3、02-8 | S | 无 |
| 05-5 | 翻译测试和豁免、`LEGACY_RESIDENT_DIT_MODELS` 的派生测试。 | 只有测试 | `test_legacy_translation.py`、`test_resident_gate_table.py` | 05-3、02-2，以及 04 实现重复块发现的 PR | M | 无 |
| 05-6 | `observability.py`、启动日志事件、03 和 04 日志调用点的改写、`weight_stack_memory_report`、D21、指标族和编排器的处理、`docs/design/metrics.md`。 | `observability.py`、`diffusion_worker.py`、`diffusion_engine.py`、`orchestrator.py`、`metrics/` | `test_events.py`、`test_memory_report.py`、`test_metrics_absorb.py`、`test_memory_report_multiproc.py`、`test_metrics_e2e.py` | 05-3、03-8，以及 04 实现 `commit()` 和内存登记的 PR | L | D21 |
| 05-7 | 第 17.1 节的 recipe、第 17.2 节的文档和迁移说明。 | `recipes/`、`docs/` | `test_recipe_policies.py` | 05-4、05-5、05-6 | M | 无 |
| 05-8 | 支持矩阵的渲染工具、文档和一致性测试。 | `tools/weight_stack/`、`docs/` | `test_support_matrix_doc.py` | 02-2、05-7 | S | 无 |

```text
01-1..01-7
  05-1
    05-2
      05-3（另依赖 04 的规划、Trigger、Transport、commit 的 PR）
        05-4（另依赖 02-8）
        05-5（另依赖 02-2 和 04 的重复块发现）
        05-6（另依赖 03-8 和 04 的 commit、内存登记）
          05-7（依赖 05-4、05-5、05-6）
            05-8（另依赖 02-2）
```

- 05-1、05-2 不改变任何行为：`unified` 下 01 第 1.3 节的临时拒绝仍然生效，新参数在 `legacy` 下按 P01 报错。05-3 合入时 `unified` 开始接受 offload 配置。
- 本 RFC 的 PR 不被 02、03、04 的 PR 依赖。04 的规划 PR 早于 05-1 合入时，按第 2 节的字段读取策略，测试里用同名字段的冻结数据类构造策略；05-1 合入后改用 `OffloadPolicy`，字段不变。

## 测试计划

CI 等级和标记按 `docs/contributing/ci/test_system_overview.md`：L1 是 `core_model` 加 `cpu`，L2 是 `core_model` 加 `hardware_test`，L3 是 `advanced_model`，L4 是 `full_model`。本 RFC 的测试都带 `diffusion` 标记。

| 测试 | 内容 | 等级 |
|---|---|---|
| `test_policy_parse.py` | P02 到 P18 每条规则一个用例，断言类型、编号和文本；第 3.2 节的示例；规范 JSON 往返；字典顺序不影响结果。 | L1 |
| `test_policy_sources.py` | 命令行覆盖 YAML 时两个键一起丢弃；`stage_<id>_` 写法；`diffusion_offload_policy` 不被递归合并（`StageConfig.to_omegaconf`、`_merge_config_fields`、平台覆盖）；同一来源两个键都给时报 P03；`legacy` 下已有字段的合并结果不变。 | L1 |
| `test_policy_legacy_stack.py` | `legacy` 下 P01；`legacy` 下旧开关、旧配置和 `pin_memory` 的现有行为不变。 | L1 |
| `test_policy_checks.py` | P19 到 P33；`__post_init__` 和 `enrich_config` 末尾的 P28；P29 的文本与现有格式一致。 | L1 |
| `test_env_folding.py` | 用 `monkeypatch` 设置两个环境变量；折叠后 worker 一侧（runner 的 `unified` 一支）不读环境变量：在 worker 里改变环境变量，结果不变。 | L1 |
| `test_resolve_selection.py` | 角色键、组件名键整体覆盖、`null`、三种来源的 `if_absent` 和 `if_no_blocks`、L01 到 L05、按角色的常驻层数。 | L1 |
| `test_execution_facts_unified.py` | 第 12 节三个字段的每种组合；策略为 `None` 时三个字段为假。 | L1 |
| `test_legacy_schema_equivalence.py`、`test_legacy_translation.py`、`test_resident_gate_table.py` | 第 6.3、6.6 节。 | L1 |
| `test_lora_rank_flag.py` | 合法值与 `DiffusionLoRAManager._VALID_MAX_RANKS` 相同；P30；`legacy` 下 P01。 | L1 |
| `test_events.py`、`test_memory_report.py` | 行格式和转义；`smaps_rollup` 的解析用固定的样例文件；登记和注销。 | L1 |
| `test_metrics_absorb.py` | `_absorb_diffusion_metrics` 弹出键并设置 Gauge；机器级求和按 `boot_id`；死掉的副本被删除；`--log-stats` 关闭时不写数据。 | L1 |
| `test_memory_report_multiproc.py` | 用 03 的本机后端在两个进程里映射同一个内存文件，两个进程的 `Pss` 之和约等于一份文件大小加各自的私有占用。 | L1，`cpu` |
| `test_metrics_e2e.py` | 一个小模型在 `unified` 下用 `module` 预设和 `--log-stats` 启动，抓取 `/metrics`，断言六个指标族存在、数值与 `startup_summary` 一致。 | L2，`hardware_test(res={"cuda": "L4"})` |
| `test_recipe_policies.py` | 解析第 17.1 节每条 recipe 命令的策略，在只建结构的上下文里调用 `resolve_selection()`，`unmatched` 为空。 | L1 |
| `test_support_matrix_doc.py` | 第 17.3 节。 | L1 |
| recipe 的端到端 | 第 17.1 节改动的 recipe 在已有每晚例行测试的硬件上以 `unified` 写法运行，输出与 `legacy` 一致，显存峰值不高于 `legacy`。 | L4 |

## 验收标准

- 命令行、Python 参数和 YAML 给出同一份策略时，`OffloadPolicy` 相等；高优先级来源整体覆盖低优先级来源，字典策略不被逐键拼接。
- P01 到 P33 每条规则在参数解析阶段报 `ConfigError`，文本与第 8.2 节一致；L01 到 L05 在规划阶段报 `PlanningError`。
- `unified` 下 worker 不读 `OMNI_DIFFUSION_PROMPT_EMBED_CACHE` 和 `OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE`；组合检查只在参数解析时做一次。
- 翻译测试对每个能在只建结构的上下文里构造的 pipeline 通过，豁免只有 `LEGACY_VAE_LIFECYCLE_MODELS`；等价测试通过。
- `--dlo-resident-layers` 只对 `LEGACY_RESIDENT_DIT_MODELS` 生效，其他模型记一次警告，与 `legacy` 相同。
- `legacy` 下现有测试全部不变；本 RFC 的新参数在 `legacy` 下报错。
- 启动日志对每个组件写明 Unit 划分、后端、是否共享及原因、冷启动路径；共享时写明角色、写入进程的 pid 和套接字名；每个 Unit 写明登记还是走暂存区；每个组件写明 `fully_resident` 和编译模式的降级。
- 第 16.1 节的指标在 `--log-stats` 打开时导出；同一部署在一台机器上的 worker 数从 1 增加到 k 时，`diffusion_machine_pss_bytes` 里共享部分不随 k 增长。
- 第 17.1 节的 recipe 和第 17.2 节的文档已更新；支持矩阵文档与当前结果一致。

## 删除 legacy 时

- `policy.py` 里的兼容层：`parse_legacy_offload_config`、旧开关的翻译、`LEGACY_RESIDENT_DIT_MODELS`、`LEGACY_VAE_LIFECYCLE_MODELS`、`pending_resident_layers` 和 `resolve_for_model`、规则 P01、P19 到 P23、P32、P33，以及翻译测试和等价测试。
- 旧字段和它们的命令行参数：`enable_cpu_offload`、`enable_layerwise_offload`、`enable_distributed_layerwise_offload`、`dlo_use_allgather`（`--dlo-use-allgather`、`--dlo-no-use-allgather`）、`dlo_resident_layers`、`dlo_host_registration_limit_gib`、`diffusion_offload_config`、`host_weight_runtime_mode`、`host_weight_runtime_root`，涉及第 1 节的五个文件；`validate_host_weight_runtime_options`、`validate_dlo_host_registration_options`、`_DiffusionConfigProjection.from_kwargs` 里对 `parse_diffusion_offload_config` 的调用、`__post_init__` 里对 `materialize_legacy_offload_flags` 的调用。删除之后传入这些参数，由 argparse 和 `validate_omni_diffusion_kwargs` 报未知参数。
- D20、D21、D22 只保留 `unified` 一支，删除对应的登记行。
- 文档里旧开关的页面并入迁移说明。`pin_cpu_memory` 和本 RFC 新增的参数保留。

## 风险

| 风险 | 影响 | 处理 |
|---|---|---|
| `text_encoder_allgather` 对组件名键按保守取值 | 组件名键用 `allgather` 的 DP 部署多做同步执行，吞吐可能下降 | 文档写明用 `text_encoder` 角色键可以得到精确的取值；`policy` 启动日志给出规范 JSON。 |
| `SP_SYNCED_CACHE_BACKENDS` 与缓存后端的实现脱节 | 后端改掉同步后没有移出表时，AllGather 错位 | 加入表的 PR 必须带多进程测试；表放在 `policy.py`，由 weight-stack 和缓存的负责人共同评审。 |
| 五个模型用旧开关启动时显存峰值高于 `legacy` | 旧命令在 `unified` 下可能显存不足 | `legacy_vae_hint` 启动日志、recipe 改写，迁移说明列入门槛例外。 |
| `pin_memory=false` 与显式的 `--pin-cpu-memory` 无法识别为冲突 | 实际不锁页，H2D 走暂存区 | `policy` 启动日志写明最终取值；文档写明旧配置的 `pin_memory` 在 `unified` 下的规则。 |

## 开放问题

- L01 要求“在本副本的任何 rank 上都没有匹配”才报错，但 `plan()` 只看本 rank 的目录，组件又可以按 rank 声明（02 第 2.5 节）。需要 01 在全组确认第 1 点第一轮的 `Point1Payload` 里收集各 rank 的 `Selection.unmatched`（由 04 放进 `PlacementPlan`），取交集，非空时报 `PlanningError`；这要改 01 第 7.2 节和 04 的 `PlacementPlan`。

## 反馈期

发布后一周。

## CC

配置、Offloader、可观测性和指标、recipe 与文档的负责人，以及 MiniMax H3、Wan2.2-S2V、MAGI-2、JoyImage、LingBot-Video 的模型负责人。
