# [RFC]: 启动事务、总开关与共存期

> 上级 RFC：[umbrella RFC](./00-umbrella.md)。设计依据：[总体设计](../LOADER_OFFLOADER_DESIGN.md) 第 3、4、8.1、9（第 1、2 步）、11、12 节。
>
> 范围：`vllm_omni/diffusion/weight_stack/` 里的 `interfaces.py`、`errors.py`、`group.py`、`startup.py`；总开关经过的配置与参数文件；扩散 worker 与模型运行器的分发；`vllm_omni/diffusion/offloader/config.py` 的查询函数；引擎和执行器读 offload 事实的调用点；`tests/diffusion/weight_stack/` 里的静态依赖测试、冻结白名单和共存期必改清单的登记；`weight_stack/` 的模块文档和 `.github/CODEOWNERS` 规则。
>
> 依赖：无前置子 RFC。[子 RFC 02](./02-unified-loader.md)、[子 RFC 03](./03-host-storage.md)、[子 RFC 04](./04-unified-offloader.md)、[子 RFC 05](./05-configuration.md) 依赖本 RFC 定义的接口、错误类型和启动驱动。

## 摘要

本 RFC 定义新栈的骨架：

- 总开关 `--diffusion-weight-stack {legacy,unified}` 的定义、传递和进程级状态；
- `vllm_omni/diffusion/weight_stack/` 包的公共接口（`interfaces.py`）和错误类型（`errors.py`）；
- 把 Loader、HWM、Offloader 串起来的启动驱动（`startup.py`），包括状态机、资源账本、所有权的转移和失败时的释放、故障注入点；
- 副本内多个 rank 的全组确认（`group.py`），即总体设计 4.5 的第 0 点到第 3 点；
- 扩散 worker 和模型运行器在两个栈之间的分发：加载、关闭、睡眠和唤醒、`custom_pipeline` 的重新初始化；
- 共存期规则：`unified` 下 `offloader/config.py` 查询函数的行为、共存期必改清单的登记和评审、静态依赖测试和冻结白名单。

本 RFC 不实现 Loader、HWM、Offloader 的内部逻辑。它规定三者之间的接口、调用顺序、所有权的转移和失败时的释放，由 [子 RFC 02](./02-unified-loader.md)、[子 RFC 03](./03-host-storage.md)、[子 RFC 04](./04-unified-offloader.md)、[子 RFC 05](./05-configuration.md) 实现。

Job 1 单独合入后，`legacy` 的行为不变。`unified` 可以在参数解析阶段选中：不做 offload 的配置能进入启动驱动，真实 pipeline 在第 1 步报 `StackUnavailableError`，提示改用 `legacy`。启动驱动、全组确认、释放顺序和分发由测试替身覆盖。

## 动机

总体设计第 4 节规定了启动的步骤、状态、所有权和全组确认，第 9 节规定了两个栈并存的规则。这些规则横跨三个模块和扩散 worker，必须由一处代码统一执行：

- 三个模块分别由 02、03、04 实现，彼此只通过接口交互（总体设计 3.2）。接口、数据类型的字段和失败时归谁释放，必须在任何一个模块动工之前定下来，否则三个子 RFC 对调用顺序的假设会不一致。
- 全组确认和“任何一个 rank 出错，所有 rank 一起报错”是副本级的协议，不属于任何一个模块。
- 现有的启动状态挂在 pipeline 对象上传递：`vllm_omni/diffusion/model_loader/host_weight_loader.py` 的 `_attach_offload_startup_state` 调用 `vllm_omni/diffusion/offloader/startup.py` 的 `attach_offload_startup_state`，`vllm_omni/diffusion/offloader/__init__.py` 的 `enable_offload_backend` 再用 `take_offload_startup_state` 从模型上取回。总体设计 3.2 第 5 条不允许这种做法，新栈的启动状态由启动驱动持有。
- 共存期的规则（总开关、查询函数、必改清单、白名单）必须在第一个 `unified` 分支合入之前由测试保证，`legacy` 的行为才不会在没有察觉的情况下改变。

## 设计

### 1. 总开关

#### 1.1 取值和传递

```text
--diffusion-weight-stack {legacy,unified}
```

- 默认值是 `legacy`。`unified` 同时启用新加载流程、HWM 和新 Offloader，没有其他组合（总体设计 8.1）。
- 命令行、Python 参数和 YAML 使用同一个字段名 `diffusion_weight_stack`。传递链与 `hsdp_weight_load_strategy` 相同，涉及五个文件：

```text
vllm_omni/entrypoints/cli/serve.py   subparser_init 的 omni_config_group   --diffusion-weight-stack，choices 为 legacy、unified，default 为 None
vllm_omni/engine/arg_utils.py        OrchestratorArgs                      diffusion_weight_stack: str | None = None
vllm_omni/config/stage_config.py     StageDeployConfig                     diffusion_weight_stack: str | None = None
vllm_omni/config/omni_config.py      _DiffusionConfigProjection            diffusion_weight_stack: Literal["legacy", "unified"] = "legacy"
vllm_omni/diffusion/data.py          OmniDiffusionConfig                   diffusion_weight_stack: str = "legacy"，在 __post_init__ 里校验
```

- Python 入口构造单 stage 默认配置时，`OmniEngineBase._create_default_diffusion_stage_cfg` 交给 `StageConfigFactory.create_default_diffusion`（`vllm_omni/config/config_factory.py`），后者的 `_normalize_default_diffusion` 按 `fields(OmniDiffusionConfig)` 筛选参数，新字段自动经过，这两个文件不改。
- 这个字段因此属于 `_DIFFUSION_CONFIG_FIELDS`，随 `od_config` 到达每个 worker，不进内容身份。
- `OmniDiffusionConfig.__post_init__` 在调用 `materialize_legacy_offload_flags` 之前调用 `parse_weight_stack()` 校验取值；取值不是 `legacy` 或 `unified` 时报 `ConfigError`。

#### 1.2 `startup.py` 里的总开关函数

```python
# vllm_omni/diffusion/weight_stack/startup.py
class WeightStack(str, enum.Enum):
    LEGACY = "legacy"
    UNIFIED = "unified"

def parse_weight_stack(value: object) -> WeightStack: ...      # None 视为 legacy；其他非法值报 ConfigError
def is_unified(config: object) -> bool: ...                   # 读 config.diffusion_weight_stack；没有这个属性时为 legacy

def set_process_weight_stack(value: WeightStack, *, owner: object) -> None: ...
def clear_process_weight_stack(*, owner: object) -> None: ...
def process_weight_stack() -> WeightStack: ...               # 没有设置时为 legacy

def execution_offload_facts(config: object) -> ExecutionOffloadFacts: ...   # 见 9.2
```

- `startup.py` 在模块级只 import 标准库、`torch`、`interfaces.py` 和 `errors.py`；`group.py`、平台和分布式模块在函数内 import。`data.py`、`offloader/config.py`、引擎和执行器在函数内 import 它，不引入循环 import。
- 读总开关有两个入口：参数解析、引擎、执行器和 `offloader/config.py` 拿得到配置，用 `is_unified(config)`；构造上下文、读取陷阱和模型代码看进程级状态（总体设计 5.2）。
- 进程级状态由 `DiffusionWorker.__init__` 设置，位置在 `init_device()` 之后、构造 model runner 之前，放在 `__init__` 已有的 `try` 块里；`try` 块的 `except` 一支调用的 `shutdown()` 和正常关闭时的 `DiffusionWorker.shutdown` 都清除它。状态记录取值和持有它的 worker 集合，最后一个持有者清除之后回到未设置。同一个进程里已经有一个存活的 worker 持有另一个取值时，`set_process_weight_stack` 报 `ConfigError`，报错写明两个 stage 的编号和取值。单进程执行器把几个 stage 的 worker 放在同一个进程里时，这些 stage 必须使用同一个取值。
- 模型代码只经 `weight_stack.components.is_unified_stack()` 读总开关（02 提供，返回 `process_weight_stack() is WeightStack.UNIFIED`），只在共存期必改清单登记的位置读（9.3、9.4）。

#### 1.3 参数解析阶段的检查

`unified` 下，`OmniDiffusionConfig.__post_init__` 不调用 `materialize_legacy_offload_flags` 改写旧开关字段，后面全量编译的冲突检查读到的 `offload_strategy` 是 `OffloadStrategy.NONE`；旧开关由 05 的兼容层翻译。

在 05 的策略解析合入主干之前，`unified` 只接受不做 offload 的配置。出现以下任何一项时，参数解析阶段报 `ConfigError`，报错列出这一项并提示改用 `--diffusion-weight-stack legacy`：

- `enable_cpu_offload`、`enable_layerwise_offload`、`enable_distributed_layerwise_offload` 中任何一个为真；
- `diffusion_offload_config` 不是 `None`；
- `dlo_host_registration_limit_gib` 大于 0；
- `host_weight_runtime_mode` 不是 `disabled`，或给出了 `host_weight_runtime_root`；
- 环境变量 `OMNI_DIFFUSION_PROMPT_EMBED_CACHE` 或 `OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE` 已设置。`unified` 下 worker 不读这两个变量（总体设计 8.5），它们由 05 在构建配置时折进 `enable_prompt_embed_cache` 和 `prompt_embed_cache_size`。

05 的策略解析、环境变量的折叠和不支持组合的检查（总体设计 8.2 到 8.6）合入时取代这一项检查，见 PR 01-2 和 05 的实现拆分。

### 2. 接口

`interfaces.py` 只依赖标准库和 `torch`。其他子 RFC 拥有的数据类型只在 `if TYPE_CHECKING:` 里引用，运行期不 import。

```python
# vllm_omni/diffusion/weight_stack/interfaces.py
from __future__ import annotations

import enum
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from vllm.config import VllmConfig
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.weight_stack.components import ComponentCatalog                     # 02
    from vllm_omni.diffusion.weight_stack.loader import ExpectedManifest, FinalManifest, Preflight  # 02
    from vllm_omni.diffusion.weight_stack.host import (                                         # 03
        Election, ElectionRecord, HostLayout, LocalRecord, MemoryRange, SharePlan,
    )
    from vllm_omni.diffusion.weight_stack.offload import PlacementPlan, Unit                     # 04
    from vllm_omni.diffusion.weight_stack.policy import OffloadPolicy                            # 05
    from vllm_omni.diffusion.weight_stack.startup import FaultPlan                               # 01


class LoadState(enum.Enum):
    STRUCTURE_READY = "structure_ready"
    PLANNED = "planned"
    SOURCES_PREPARED = "sources_prepared"
    ADMITTED = "admitted"
    TRANSACTION_READY = "transaction_ready"
    WEIGHTS_READY = "weights_ready"
    FINALIZED = "finalized"
    COMMITTED = "committed"
    IN_SERVICE = "in_service"
    FAILED = "failed"            # 终止状态：启动失败，本 rank 的资源已释放
    CLOSED = "closed"            # 终止状态：关闭完成


@dataclass(frozen=True)
class ParallelCoordinate:
    size: int
    rank: int


@dataclass(frozen=True)
class RankCoordinates:
    tp: ParallelCoordinate
    sp: ParallelCoordinate
    ulysses: ParallelCoordinate
    ring: ParallelCoordinate
    cfg: ParallelCoordinate
    dp: ParallelCoordinate
    pp: ParallelCoordinate
    ep: ParallelCoordinate | None        # 没有启用专家并行时为 None
    fs: ParallelCoordinate | None        # 没有启用 HSDP 时为 None


@dataclass(frozen=True)
class Topology:
    deployment_id: str                           # 32 个十六进制字符
    deployment_scoped: bool                      # False：配置里没有部署标识，由 0 号 rank 在第 0 点生成，只在本副本内有效
    stage_id: int
    world_size: int
    records: tuple[LocalRecord, ...]             # 按 rank 排列，长度等于 world_size
    coordinates: tuple[RankCoordinates, ...]     # 按 rank 排列，长度等于 world_size


@dataclass(frozen=True)
class RankFingerprints:
    rank: int
    content_identities: Mapping[str, str]        # 组件名到内容身份摘要；只含本 rank 声明的组件
    plan_fingerprint: str                        # HostLayout 里本机后端部分的摘要
    collective: Mapping[str, str]                # 组件名到这个组件的 AllGather Unit、它们的内容身份和布局、Transport 的摘要；
                                                 # 只含有 allgather Unit 的组件


Fingerprints = tuple[RankFingerprints, ...]      # 按 rank 排列


@dataclass(frozen=True)
class AllGatherGroup:
    component: str
    ranks: tuple[int, ...]                       # 升序，至少 2 个
    group: ProcessGroup | None                   # 本 rank 不在这个组里时为 None


AllGatherGroups = Mapping[str, tuple[AllGatherGroup, ...]]   # 组件名到这个组件的各组，按最小 rank 排序


@dataclass(frozen=True)
class Elections:
    local: Election                              # 本 rank 的参选结果；写入进程的 Election 持有监听套接字和监听线程
    records: tuple[ElectionRecord, ...]          # 第 1 点第二轮收集到的各 rank 参选结果，按 rank 排列


class MemoryPools(Protocol):
    def weights(self) -> AbstractContextManager[None]: ...    # 睡眠时备份的池
    def slots(self) -> AbstractContextManager[None]: ...      # 睡眠时只释放、不备份的池


@dataclass(frozen=True)
class StartupContext:
    od_config: OmniDiffusionConfig
    vllm_config: VllmConfig
    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    load_format: str                             # "default"、"diffusers" 或 "custom_pipeline"
    custom_pipeline_name: str | None
    memory_pools: MemoryPools
    policy: OffloadPolicy | None                 # None 表示不做 offload
    compile_enabled: bool                        # 不是 enforce_eager，并且平台支持 torch inductor
    faults: FaultPlan | None                     # 只用于故障注入测试


class Loader(Protocol):
    def build_structure(self) -> tuple[nn.Module, ComponentCatalog]: ...                     # 第 2 步
    def preflight(self, catalog: ComponentCatalog, plan: PlacementPlan) -> Preflight: ...     # 第 5 步
    def write(self, model: nn.Module, storage: HostStorage, transaction: LoadTransaction) -> None: ...  # 第 9 步
    def finalize(self, model: nn.Module, storage: HostStorage) -> FinalManifest: ...          # 第 10 步
    def close(self) -> None: ...                                                              # 幂等


class Offloader(Protocol):
    def plan(self, catalog: ComponentCatalog, topology: Topology, policy: OffloadPolicy | None) -> PlacementPlan: ...
    def prepare_materialization(self, model: nn.Module, plan: PlacementPlan, storage: HostStorage) -> LoadTransaction: ...
    def setup_compile(self, model: nn.Module, catalog: ComponentCatalog, runtime: OffloadRuntime) -> None: ...  # 第 12 步


class LoadTransaction(Protocol):
    def target(self, tensor_key: str) -> torch.Tensor | None: ...
    def commit(self, final_manifest: FinalManifest) -> OffloadRuntime: ...
    def abort(self) -> None: ...                                                              # 幂等；提交后什么都不做


class OffloadRuntime(Protocol):
    def close(self) -> None: ...                                                              # 幂等、本地、终止性
    def grad_mode(self) -> AbstractContextManager[None]: ...
    def execution(self) -> AbstractContextManager[None]: ...
    def before_sleep(self) -> None: ...
    def after_wake(self) -> None: ...


class HostWeightManager(Protocol):
    def local_record(self) -> LocalRecord: ...
    def deployment_id(self) -> tuple[str, bool]: ...                                          # （部署标识，是否由 StageRuntime 下发）
    def layout(self, plan: PlacementPlan, expected_manifest: ExpectedManifest) -> HostLayout: ...
    def share_plan(self, topology: Topology, fingerprints: Fingerprints) -> SharePlan: ...
    def elect(self, share_plan: SharePlan) -> Election: ...
    def admit(self, topology: Topology, share_plan: SharePlan, elections: Elections, layout: HostLayout) -> None: ...
    def allocate(self, layout: HostLayout, share_plan: SharePlan, elections: Elections) -> HostStorage: ...


class HostStorage(Protocol):
    def view(self, tensor_key: str) -> torch.Tensor: ...
    def ranges(self, unit: Unit) -> list[MemoryRange]: ...
    def seal(self, final_manifest: FinalManifest) -> None: ...
    def connect_writer(self) -> None: ...
    def close(self) -> None: ...                                                              # 幂等
```

`Offloader`、`LoadTransaction`、`OffloadRuntime`、`HostWeightManager`、`HostStorage` 里总体设计 4.1 列出的方法，参数和返回值与 4.1 相同。各方法的规范：

| 方法 | 规范 |
|---|---|
| `Loader.build_structure()` | 在只建结构的上下文里构造 pipeline，返回模型和组件声明。组件声明在 `Loader.close()` 之后仍然可以查询。 |
| `Loader.preflight()` | 返回的 `Preflight` 含按组件的内容身份和预期清单，不写模型。 |
| `Loader.write()` | 每个张量的写入目标取 `transaction.target(tensor_key)`；返回 `None` 的张量常驻加速卡，由 Loader 在执行设备上、在 `ctx.memory_pools.weights()` 里分配，因此进入睡眠时备份的池。写入期间的临时张量不在这个池里分配（总体设计 5.2：只有 Loader 分配的常驻存储进池）。 |
| `Loader.finalize()` | 执行总体设计 5.4 的第 1 到第 7 步，其中第 6 步调用 `storage.seal()`，返回最终清单。收尾里新建、替换常驻加速卡参数的张量同样由 Loader 在 `ctx.memory_pools.weights()` 里分配。 |
| `Loader.close()` | 关闭构造上下文、读取陷阱、checkpoint 读取器和临时张量。驱动在第 10 步成功之后立即调用它（总体设计 5.2：读取陷阱开到收尾结束）；失败时由释放流程调用。可以多次调用。 |
| `Offloader.plan()` | 纯计算，不分配大块资源，不改模型，不装钩子。 |
| `Offloader.prepare_materialization()` | 成功时返回的启动事务持有 `storage` 和全部运行期资源。失败时它自己分配的资源由它释放，`storage` 仍归调用方。 |
| `Offloader.setup_compile()` | 在提交之后运行，编译目标从组件声明读取（总体设计 7.8）。 |
| `LoadTransaction.target()` | Host 驻留的张量返回 `storage.view(tensor_key)`；常驻加速卡的张量返回 `None`。 |
| `LoadTransaction.commit()` | 成功时全部资源一次性移交返回的 `OffloadRuntime`，启动事务变为空。失败时全部资源仍在启动事务里，由 `abort()` 释放。 |
| `LoadTransaction.abort()` | 按资源账本的顺序释放启动事务持有的全部资源，其中包括关闭 host 存储。 |
| `OffloadRuntime.close()` | 按总体设计 7.5 的顺序关闭，不把权重恢复回 CPU，不发起集合通信。 |
| `OffloadRuntime.grad_mode()` | 返回前向使用的梯度模式（总体设计 7.5）。 |
| `OffloadRuntime.execution()` | 包住一次前向；同一时刻只允许一次，重入或并发时报错。 |
| `OffloadRuntime.before_sleep()`、`after_wake()` | 在睡眠前、唤醒后调用（总体设计 7.6）。 |
| `HostWeightManager.deployment_id()` | `od_config.deployment_id` 不为 `None` 时返回（这个值，`True`）；否则返回（`secrets.token_hex(16)` 新生成的值，`False`）。只有 0 号 rank 的结果被采用（7.2），生成和下发由 03 负责。 |
| `HostWeightManager.layout()` | 由规划和预期清单算出 `HostLayout`；`HostLayout.plan_fingerprint` 是本机后端部分的摘要。 |
| `HostWeightManager.elect()` | 每个 rank 都调用。本 rank 不是任何共享集合的代表时不参选，返回空的 `Election`。 |
| `HostWeightManager.admit()` | 纯函数，只读 `elections.records`。 |
| `HostWeightManager.allocate()` | 成功时接管 `elections.local` 的监听套接字和监听线程。失败时 `Election` 仍归调用方。 |
| `HostStorage.connect_writer()` | 第 2 点之后调用：等待者连接写入进程，代表沿用选举时的连接；本 rank 不是等待者时什么都不做。 |
| `HostStorage.close()` | 只释放本进程的映射；写入进程同时关闭监听套接字和文件描述符。 |

`Election` 必须提供 `record() -> ElectionRecord`（可以 pickle 的参选结果）和可以多次调用的 `close()`。`PlacementPlan` 必须提供 `collective_signature() -> Mapping[str, str]`（即 `RankFingerprints.collective`）和 `bind_groups(groups: AllGatherGroups) -> None`。`HostLayout` 必须提供 `plan_fingerprint` 和 `bind_groups(groups: AllGatherGroups) -> None`。`bind_groups` 发现组的大小与规划的假设不同时报 `GroupConfirmationError`。

HWM 和 Offloader 对象本身不持有资源：它们分配的资源都在返回值里交给调用方。

### 3. 错误类型

`errors.py` 只依赖标准库。

```python
class WeightStackError(RuntimeError):
    phase: ClassVar[str]
    def __init__(self, what: str, *, where: str | None = None, fix: str | None = None,
                 component: str | None = None) -> None: ...
    stage_id: int | None          # 由启动驱动在抛出前填写
    rank: int | None
    world_size: int | None
    def summary(self) -> ErrorSummary: ...

@dataclass(frozen=True)
class ErrorSummary:               # 全组确认时随消息收集，可以 pickle
    error_type: str
    phase: str
    rank: int
    message: str                  # 截断到 4096 个字符
```

错误类型按总体设计 8.6 的校验阶段各有一个子类，另有启动事务各步骤的失败类型：

| 类型 | 说明 |
|---|---|
| `ConfigError`（同时是 `ValueError`） | 对应参数解析阶段。`phase` 是 `config`，由 01 和 05 抛出。总开关取值非法、同一个进程里的 worker 取值冲突、参数解析阶段发现不支持的组合时报这个错。 |
| `ConstructionError` | 对应构造阶段。`phase` 是 `build_structure`，由 02 抛出。读取陷阱触发、构造后残留 meta 张量、pipeline 还没有迁移到只建结构时报这个错。 |
| `PlanningError` | 对应 Offloader 规划阶段。`phase` 是 `plan`，由 04 抛出。组件不完整、找不到重复块、重复块以外的组件单机放不下时报这个错。 |
| `PreflightError` | 对应 Loader 预检阶段。`phase` 是 `preflight`，由 02 抛出。来源不可读、适配器的键对不上时报这个错。 |
| `GroupConfirmationError` | 对应全组确认第 1 点。`phase` 是所在的确认点或第 1 点的本地步骤（`confirm`、`admit`），由 01、03、04 抛出。参与集合通信的部分各 rank 不一致、AllGather 组只剩一个 rank、部署标识不一致、各 rank 不在同一个确认点上时报这个错。子类 `HostCapacityError` 的 `phase` 是 `admit`，由 03 的 `admit()` 在单机 Host 容量不足时抛出。 |
| `PrepareError` | 对应 `prepare_materialization()` 阶段。`phase` 是 `prepare`，由 04 抛出。加速卡槽位、流、事件或暂存区分配失败，内存登记或暂存区的计划不成立时报这个错。子类 `DeviceCapacityError` 在加速卡容量准入不通过时抛出。 |
| `StackUnavailableError` | 启动事务的失败。`phase` 是 `construct_modules`，由 01 抛出。新栈的某个实现模块或工厂函数不存在时报这个错。 |
| `AllocationError` | 启动事务的失败。`phase` 是 `allocate` 或 `connect_writer`，由 03 抛出。host 存储分配失败、等待者连接写入进程失败时报这个错。 |
| `WriteError` | 启动事务的失败。`phase` 是 `write`，由 02 抛出。写入失败时报这个错。 |
| `FinalizeError` | 启动事务的失败。`phase` 是 `finalize`，由 02 和 03 抛出。收尾核对失败，或封存、交接、只读绑定失败时报这个错。 |
| `CommitError` | 启动事务的失败。`phase` 是 `commit`，由 04 抛出。校验最终清单、内存登记、装钩子或移交失败时报这个错。 |
| `PeerFailureError` | 启动事务的失败。`phase` 是所在的确认点，由 01 抛出。别的 rank 在这个确认点报告了错误时报这个错。 |
| `ConfirmationTimeoutError` | 启动事务的失败。`phase` 是所在的确认点，由 01 抛出。有 rank 在超时之内没有到达确认点时报这个错。 |
| `LifecycleError` | `phase` 是发生错误的那一步，01、02、03、04 都会抛出。非法的状态转换、资源重复登记、在关闭以后调用时报这个错。 |
| `RuntimeFault` | 服务期间的失败。`phase` 是 `service`，由 04 抛出。进入 `IN_SERVICE` 之后的运行期故障报这个错，例如集合通信的顺序不再可信。 |

启动驱动在某一步里捕获到不是 `WeightStackError` 的异常时，把它包成这一步对应的类型再抛出，原异常作为 `__cause__` 保留。步骤和类型的对应：

| 步骤 | 包成的类型 |
|---|---|
| `construct_modules` | 包成 `StackUnavailableError`。 |
| `build_structure` | 包成 `ConstructionError`。 |
| `plan` | 包成 `PlanningError`。 |
| `preflight` | 包成 `PreflightError`。 |
| `point0_topology`、`point1_fingerprints`、`confirm`、`point1_elections`、`admit`、`point2_ready`、`point3_commit` | 包成 `GroupConfirmationError`。 |
| `allocate`、`connect_writer` | 包成 `AllocationError`。 |
| `prepare` | 包成 `PrepareError`。 |
| `write` | 包成 `WriteError`。 |
| `finalize` | 包成 `FinalizeError`。 |
| `commit` | 包成 `CommitError`。 |

报错信息的格式：

```text
[weight-stack][stage <stage_id>][rank <rank>/<world_size>][<phase>] <what>; at <where>; fix: <fix>
```

- `what` 写明事实，包含组件名和不一致的值。
- `where` 写明位置：文件路径、模块路径、张量名或套接字名。没有位置时省略这一段。
- `fix` 给出一个具体的动作：一个参数、一个 recipe 字段或一个命令。没有可行的动作时省略这一段。
- 未迁移的 pipeline、新栈不可用时的 `fix` 是 `use --diffusion-weight-stack legacy`。
- `PeerFailureError` 的 `what` 列出每个出错的 rank 和它的 `ErrorSummary.message`。

### 4. 状态机

启动驱动按总体设计 4.3 的顺序推进状态，只能前进到下一个状态，或者进入 `FAILED`；其他转换报 `LifecycleError`。

| 状态 | 由哪个调用进入 | 进入时新增的持有 | 在这个状态里失败时释放什么 |
|---|---|---|---|
| （开始） | 调用 `run_unified_startup()`，第 1 步由工厂构造 Loader、HWM、Offloader。 | 三个对象；HWM 和 Offloader 对象本身不持有资源，Loader 还没有进入构造上下文。 | 已经构造出的 Loader 调用 `close()`；工厂失败时没有需要释放的资源。 |
| `STRUCTURE_READY` | `Loader.build_structure()` 返回。 | 新增 meta 模型、组件声明、构造上下文和读取陷阱。 | 调用 `loader.close()`，丢弃模型。 |
| `PLANNED` | 第 0 点成功，`Offloader.plan()` 返回。 | 新增 `Topology` 和 `PlacementPlan`，两者都不持有资源。 | 释放的内容与 `STRUCTURE_READY` 相同。 |
| `SOURCES_PREPARED` | `Loader.preflight()` 和 `HostWeightManager.layout()` 都返回。 | 新增来源、内容身份、预期清单和 `HostLayout`；第 1 点里 `elect()` 返回之后还有 `Election`。 | 释放的内容与 `STRUCTURE_READY` 相同；已经有 `Election` 时，驱动另外调用 `election.close()`。 |
| `ADMITTED` | 第 1 点两轮都成功，`admit()` 返回。 | 新增的持有是 `Election` 和 AllGather 进程组，写入进程的 `Election` 持有监听套接字和监听线程。 | `allocate()` 返回之前，驱动关闭 `Election`；`allocate()` 返回之后、`prepare_materialization()` 返回之前，驱动关闭 host 存储。然后调用 `loader.close()`，丢弃模型。进程组不在这里释放（第 5 节）。 |
| `TRANSACTION_READY` | `prepare_materialization()` 返回，第 2 点成功。 | 新增启动事务，它持有 host 存储和全部运行期资源。 | 调用 `transaction.abort()`，然后调用 `loader.close()`，丢弃模型，执行 `gc.collect()` 和 `empty_cache`。 |
| `WEIGHTS_READY` | `Loader.write()` 返回。 | 新增已写入的字节。 | 释放的内容与 `TRANSACTION_READY` 相同。 |
| `FINALIZED` | `Loader.finalize()` 返回，驱动随即调用 `loader.close()`。 | 新增最终清单；共享存储已封存，本进程持有只读映射；构造上下文和读取陷阱已关闭。 | 释放的内容与 `TRANSACTION_READY` 相同。 |
| `COMMITTED` | `transaction.commit()` 返回。 | `OffloadRuntime` 持有全部资源，启动事务为空；叠加层槽位已移出账本，由模型的包装层引用，等待动态 LoRA 管理器接收。 | 调用 `runtime.close()`，丢弃模型，执行 `gc.collect()` 和 `empty_cache`；叠加层槽位随模型一起回收。第 3 点上别的 rank 报错时同样处理。 |
| `IN_SERVICE` | 第 3 点成功、编译设置结束、runner 完成后续步骤之后，runner 调用 `handle.mark_in_service()`。 | 持有的内容与 `COMMITTED` 相同。 | 运行期故障报 `RuntimeFault`，这个副本的全部 worker 终止。关闭时调用 `runtime.close()`。 |
| `FAILED` | 任何一步失败，并且释放完成。 | 不再持有任何资源。 | 没有需要释放的资源。 |
| `CLOSED` | `handle.close()` 返回。 | 不再持有任何资源。 | 没有需要释放的资源。 |

释放总是按所有权链上当时的持有者进行（5.1），上表写的是每个状态下的持有者。从开始写模型到提交成功，模型实例是可丢弃的（总体设计 4.4）：释放时启动事务、Loader 和驱动依次放弃对模型的引用，模型由 `gc.collect()` 回收。

### 5. 资源图与释放顺序

#### 5.1 所有权链

host 存储和运行期资源沿一条链移交，每个时刻只有一个持有者：

```text
Election ──allocate()──▶ HostStorage ──prepare_materialization()──▶ LoadTransaction ──commit()──▶ OffloadRuntime
驱动持有                  驱动持有                                    启动事务是唯一持有者              服务期间持有
```

| 资源 | 持有者 | 失败或关闭时由谁释放 |
|---|---|---|
| 构造上下文、读取陷阱、checkpoint 读取器、变换中的临时张量、适配器的 Δ | 这些资源由 Loader 持有。 | 驱动调用 `loader.close()` 释放它们。 |
| 组件声明、内容身份、最终清单 | 由 Loader 建立（总体设计 11）。`Loader.close()` 之后组件声明仍然可以查询，句柄 `WeightStackHandle` 保留对它的引用。 | 句柄关闭时丢弃。它们不持有系统资源。 |
| `Topology`、`SharePlan`、指纹 | 它们只在启动期间由驱动持有（总体设计 11 的 Runner，仅在启动期间）。 | 驱动返回时丢弃，句柄不保留它们。 |
| 监听套接字和监听线程（写入进程） | `allocate()` 之前由 `Election` 持有，之后由 `HostStorage` 持有。 | `allocate()` 之前失败时驱动调用 `election.close()`；之后随 `HostStorage.close()` 关闭。 |
| host 存储 | `prepare_materialization()` 之前归驱动，之后归启动事务，提交后归 `OffloadRuntime`。 | 三个阶段分别由 `storage.close()`、`transaction.abort()`、`runtime.close()` 释放。 |
| 加速卡槽位、流、事件、集合通信工作区、暂存区、内存登记、钩子 | 提交前由启动事务持有，提交后由 `OffloadRuntime` 持有。 | 由 `transaction.abort()` 或 `runtime.close()` 释放。 |
| 叠加层槽位 | 提交前由启动事务持有，提交后由动态 LoRA 管理器持有。 | 提交前由 `transaction.abort()` 释放。提交后、动态 LoRA 管理器接收之前，它们只被模型的包装层引用，丢弃模型时回收；接收之后随动态 LoRA 管理器释放。 |
| AllGather 进程组 | 进程组记在 `group.py` 的进程级表里。 | `DiffusionWorker.shutdown` 在 `destroy_distributed_env()` 之前调用 `destroy_allgather_groups()` 销毁它们。 |

AllGather 进程组按成员和后端记在进程级的表里。同一个进程里后来的启动流程（`custom_pipeline` 的重新初始化）需要成员相同的组时直接复用，不再创建。关闭事务或运行时不销毁进程组，所以关闭不涉及进程组上的任何操作。

#### 5.2 资源账本

启动事务和 `OffloadRuntime` 把持有的每个资源登记在 `startup.py` 的 `ResourceLedger` 里。账本的关闭顺序就是总体设计 7.5 的关闭顺序：

```python
# vllm_omni/diffusion/weight_stack/startup.py
class ResourceKind(enum.IntEnum):    # 关闭时按数值从小到大
    EXECUTION = 1                    # 停止接受新的执行
    INFLIGHT = 2                     # 等加速卡上的拷贝和集合通信做完
    HOOKS = 3                        # 撤掉钩子和对模型的引用
    DEVICE = 4                       # 加速卡槽位、事件、流、集合通信工作区
    REGISTRATION = 5                 # 撤销内存登记
    STAGING = 6                      # 释放暂存区
    PLACEHOLDER = 7                  # 参数改指向占位
    HOST_STORAGE = 8                 # 关闭 host 存储的本进程映射

class ResourceLedger:
    def register(self, handle: object, kind: ResourceKind, close: Callable[[], None]) -> None: ...
    def hand_over(self, handle: object) -> None: ...    # 移出账本，交给账本以外的持有者
    def transfer(self) -> ResourceLedger: ...           # 全部条目一次性转入新账本
    def close(self) -> None: ...
```

- 同一个句柄（按对象身份）登记第二次时报 `LifecycleError`。
- `transfer()` 之后，原账本为空，再登记时报 `LifecycleError`。`commit()` 用它把启动事务的资源一次性移交给 `OffloadRuntime`。
- `hand_over()` 用于提交时把叠加层槽位交给动态 LoRA 管理器。
- `close()` 按种类的数值从小到大关闭；同一种类里按登记的逆序关闭。可以多次调用，第二次起什么都不做。
- 一个关闭函数抛出异常时，其余的关闭函数照常执行；全部执行完之后报 `LifecycleError`，第一个异常作为 `__cause__`，其余的写进日志。
- 账本不调用 `torch.distributed` 的任何函数。

### 6. 启动驱动

#### 6.1 入口

```python
# vllm_omni/diffusion/weight_stack/startup.py
class StackFactory(Protocol):
    def create(self, ctx: StartupContext) -> tuple[Loader, HostWeightManager, Offloader]: ...

class EntryPointFactory:          # 默认工厂
    LOADER = "vllm_omni.diffusion.weight_stack.loader:create_loader"
    HOST = "vllm_omni.diffusion.weight_stack.host:create_host_weight_manager"
    OFFLOADER = "vllm_omni.diffusion.weight_stack.offload:create_offloader"

class WeightStackHandle:
    state: LoadState
    model: nn.Module
    catalog: ComponentCatalog
    runtime: OffloadRuntime
    def mark_in_service(self) -> None: ...     # COMMITTED 到 IN_SERVICE
    def close(self) -> None: ...               # 幂等、本地；调用 runtime.close()，丢弃模型，进入 CLOSED

def run_unified_startup(ctx: StartupContext, factory: StackFactory | None = None) -> WeightStackHandle: ...
```

- `EntryPointFactory` 用 `importlib` 按名字取三个工厂函数，`startup.py` 不在模块级 import 任何实现。任何一个模块或函数不存在时报 `StackUnavailableError`，列出缺少的名字。三个工厂函数各自接收 `StartupContext`，返回对应的实现。
- 驱动在 runner 里运行，总体设计里 Runner 的启动职责（构造三个模块、推进流程、失败时关闭 `Election`）都由驱动承担。
- `StartupContext.policy` 由驱动之外的 runner 填写：05 合入之后是 05 解析出的策略，之前是 `None`。

#### 6.2 步骤

```python
def run_unified_startup(ctx, factory=None):
    s = _Startup(ctx, factory or EntryPointFactory())
    try:
        s.local("construct_modules", s.construct_modules)  # 第 1 步：构造 Loader、HWM、Offloader
        s.local("build_structure", s.build_structure)      # 第 2 步：进入 STRUCTURE_READY
        s.confirm_topology()                               # 第 3 步：第 0 点，得到 Topology
        s.local("plan", s.plan)                            # 第 4 步：进入 PLANNED
        s.local("preflight", s.preflight)                  # 第 5 步：预检和 HostLayout，进入 SOURCES_PREPARED
        s.confirm_admission()                              # 第 6 步：第 1 点，进入 ADMITTED
        s.local("allocate", s.allocate)                    # 第 7 步：Election 移交 HostStorage
        s.local("prepare", s.prepare)                      # 第 8 步：HostStorage 移交启动事务
        s.confirm_ready()                                  # 第 2 点，进入 TRANSACTION_READY
        s.local("connect_writer", s.connect_writer)        # 等待者连接写入进程
        s.local("write", s.write)                          # 第 9 步：进入 WEIGHTS_READY
        s.local("finalize", s.finalize)                    # 第 10 步：进入 FINALIZED，关闭 Loader
        s.local("commit", s.commit)                        # 第 11 步：启动事务移交 OffloadRuntime，进入 COMMITTED
        s.confirm_commit()                                 # 第 3 点
        s.compile_setup()                                  # 第 12 步
        return s.handle()                                  # 第 13 步由 runner 完成
    except BaseException:
        s.release()                                        # 按所有权链释放，进入 FAILED
        raise
```

每一步的内容：

```python
class _Startup:
    owner: Election | HostStorage | LoadTransaction | OffloadRuntime | None   # 所有权链上此时的持有者
    pending: WeightStackError | None                                          # 本 rank 已经发生、还没在确认点报告的错误

    def construct_modules(self):
        self.loader, self.hwm, self.offloader = self.factory.create(self.ctx)

    def build_structure(self):
        self.model, self.catalog = self.loader.build_structure()
        self.enter(LoadState.STRUCTURE_READY)

    def confirm_topology(self):                                    # 第 0 点
        gathered = self.round("point0_topology", lambda: Point0Payload(
            record=self.hwm.local_record(),
            coordinates=read_rank_coordinates(),
            deployment=self.hwm.deployment_id(),
        ))
        self.topology = assemble_topology(gathered, self.ctx)      # 只依赖收集结果，各 rank 结论相同

    def plan(self):
        self.plan_ = self.offloader.plan(self.catalog, self.topology, self.ctx.policy)
        self.enter(LoadState.PLANNED)

    def preflight(self):
        self.preflight_ = self.loader.preflight(self.catalog, self.plan_)
        self.layout = self.hwm.layout(self.plan_, self.preflight_.expected_manifest)
        self.fingerprints = RankFingerprints(
            rank=self.ctx.rank,
            content_identities=self.preflight_.content_identities,
            plan_fingerprint=self.layout.plan_fingerprint,
            collective=self.plan_.collective_signature(),
        )
        self.enter(LoadState.SOURCES_PREPARED)

    def confirm_admission(self):                                   # 第 1 点
        gathered = self.round("point1_fingerprints", lambda: Point1Payload(
            fingerprints=self.fingerprints, group_keys=group_table_keys(),
            unmatched_policy_keys=self.plan_.unmatched_policy_keys()))
        all_fps = tuple(p.fingerprints for p in gathered)
        check_unmatched_policy_keys(gathered)                      # 各 rank 的集合取交集，非空时所有 rank 同时报 PlanningError
        check_collective_parts(all_fps)                            # 只依赖收集结果；不一致时所有 rank 同时报错
        groups = create_allgather_groups(all_fps, gathered)        # 集合通信，本副本每个 rank 以相同顺序执行
        self.local("confirm", lambda: self.bind_and_elect(groups, all_fps))
        records = self.round("point1_elections", lambda: self.election.record())
        self.local("admit", lambda: self.admit(records))

    def bind_and_elect(self, groups, all_fps):
        self.layout.bind_groups(groups)
        self.plan_.bind_groups(groups)
        self.share_plan = self.hwm.share_plan(self.topology, all_fps)
        self.election = self.hwm.elect(self.share_plan)
        self.owner = self.election                     # 写入进程的 Election 持有监听套接字和监听线程

    def admit(self, records):
        self.elections = Elections(local=self.election, records=records)
        self.hwm.admit(self.topology, self.share_plan, self.elections, self.layout)
        self.enter(LoadState.ADMITTED)

    def allocate(self):
        self.storage = self.hwm.allocate(self.layout, self.share_plan, self.elections)
        self.owner = self.storage                      # Election 已移交 HostStorage

    def prepare(self):
        self.transaction = self.offloader.prepare_materialization(self.model, self.plan_, self.storage)
        self.owner = self.transaction                  # HostStorage 已移交启动事务

    def confirm_ready(self):                           # 第 2 点
        self.round("point2_ready")
        self.enter(LoadState.TRANSACTION_READY)

    def connect_writer(self):
        self.storage.connect_writer()

    def write(self):
        self.loader.write(self.model, self.storage, self.transaction)
        self.enter(LoadState.WEIGHTS_READY)

    def finalize(self):
        self.final_manifest = self.loader.finalize(self.model, self.storage)
        self.loader.close()                            # 读取陷阱开到收尾结束；组件声明仍然可以查询
        self.enter(LoadState.FINALIZED)

    def commit(self):
        self.runtime = self.transaction.commit(self.final_manifest)
        self.owner = self.runtime                      # 启动事务为空
        self.enter(LoadState.COMMITTED)

    def confirm_commit(self):                          # 第 3 点
        self.round("point3_commit")

    def compile_setup(self):
        if self.ctx.compile_enabled:
            try:
                if self.ctx.faults is not None:
                    self.ctx.faults.hit("compile_setup", self.ctx.rank)
                self.offloader.setup_compile(self.model, self.catalog, self.runtime)
            except Exception as exc:                   # 与 runner 现有的 setup_compile() 规则相同
                logger.warning("setup_compile failed (%s); running without compile", exc)
```

失败的捕获和报告：

```python
    def local(self, phase, fn):
        if self.pending is not None:                   # 本 rank 已经失败：跳过，等下一个确认点报告
            return
        try:
            if self.ctx.faults is not None:
                self.ctx.faults.hit(phase, self.ctx.rank)
            fn()
        except Exception as exc:
            self.fail(exc, phase)

    def round(self, point, compute=None):              # 一个确认点上的一轮收集
        payload = None
        if self.pending is None:
            try:
                if self.ctx.faults is not None:
                    self.ctx.faults.hit(point, self.ctx.rank)
                payload = compute() if compute is not None else None
            except Exception as exc:
                self.fail(exc, point)
        message = RoundMessage.of(point, self.ctx.rank, self.pending, payload)   # pending 不为空时 ok 为假
        gathered = self.group.exchange(point, message)
        raise_if_any_failed(gathered, self.pending)    # 本 rank 出错时抛出自己的错误，否则抛出 PeerFailureError
        return tuple(m.payload for m in gathered)

    def fail(self, exc, phase):
        self.pending = wrap(exc, phase, self.ctx)      # 不是 WeightStackError 的包成这一步的类型
        self.release()                                 # 立即释放本 rank 的资源

    def release(self):                                 # 幂等；只做本地操作
        close_owner(self.owner)                        # runtime.close()、transaction.abort()、storage.close() 或 election.close()
        self.owner = None
        if self.loader is not None:
            self.loader.close()
        self.model = self.catalog = None
        gc.collect()
        current_omni_platform.empty_cache()
        self.state = LoadState.FAILED
```

- 一个 rank 在两个确认点之间失败时，立即释放本 rank 的全部资源，跳过后面的本地步骤，带着错误进入下一个确认点。确认点收集到任何错误时，所有 rank 一起报错：出错的 rank 抛出自己的错误，其余 rank 抛出 `PeerFailureError`。收集到错误之后，没有一个 rank 继续执行下一步。
- 第 1 点第一轮之后，核对和建组只依赖收集到的结果，各 rank 的输入相同、结论相同，所以它们不是本地步骤：核对不通过时所有 rank 同时报错，不需要再收集一轮；建组是集合通信，本副本的每个 rank 都执行。本地步骤 `confirm` 在建组之后才开始，它的失败不会让别的 rank 停在建组的集合通信上。
- 每个本地步骤都在第 3 点之前，所以每个错误都会在某个确认点上被报告。第 3 点之后只剩编译设置和 runner 的后续步骤。编译设置的异常只记警告；runner 后续步骤的失败是本 rank 的启动失败：worker 初始化失败，执行器的启动随之失败，本副本的其余 worker 一起终止（总体设计 4.4、6.5）。
- `KeyboardInterrupt`、`SystemExit` 这类不是 `Exception` 的异常不进入确认点，由最外层的 `except BaseException` 释放资源后立即抛出。
- 驱动自己不进入任何睡眠池。Loader 在 `memory_pools.weights()` 里分配常驻加速卡的权重（02）；Offloader 在 `prepare_materialization()` 里于 `memory_pools.slots()` 分配块环缓冲区和分片槽位，于 `memory_pools.weights()` 分配叠加层槽位（04，总体设计 7.6）。只建结构期间的分配和写入期间的临时张量不进任何睡眠池（总体设计 5.2）。

### 7. 全组确认

#### 7.1 通道和每一轮的协议

全组确认走 world 组的 gloo 组 `get_world_group().cpu_group`（`vllm_omni/diffusion/distributed/parallel_state.py`），只覆盖本副本的 rank，与 `vllm_omni/diffusion/worker/diffusion_worker.py` 的 `_all_gather_rank_values` 用的是同一个组。副本之间不经过 torch.distributed。确认点的实现放在 `group.py`。

每一轮交换一条消息：

```python
@dataclass(frozen=True)
class RoundMessage:
    point: str                    # 轮次名
    rank: int
    ok: bool
    error: ErrorSummary | None    # ok 为假时给出
    payload: object | None        # ok 为真时给出，可以 pickle

    @classmethod
    def of(cls, point, rank, pending, payload) -> RoundMessage: ...   # pending 不为空时 ok 为假
```

```python
def exchange(self, point, message):
    if self.world_size == 1:
        return (message,)
    torch.distributed.monitored_barrier(group=self.cpu_group, timeout=self.timeout, wait_all_ranks=True)
    gathered = [None] * self.world_size
    torch.distributed.all_gather_object(gathered, message, group=self.cpu_group)
    return tuple(gathered)
```

- 本 rank 有未报告的错误，或者计算这一轮要交出的内容时出错，这一轮发出 `ok=False` 和 `ErrorSummary`。
- 收集到的消息里 `point` 不一致时报 `GroupConfirmationError`，说明各 rank 不在同一个确认点上。
- 任何一条消息 `ok` 为假时，所有 rank 一起报错：本 rank 有错误的抛出自己的错误，其余 rank 抛出 `PeerFailureError`，列出每个出错的 rank 和它的错误摘要。
- 超时：`od_config.dist_timeout` 给出时取它（秒），否则是 1800 秒。这个字段已经存在，现有代码不读它。`monitored_barrier` 超时的 rank 报 `ConfirmationTimeoutError`；0 号 rank 的报错列出没有到达的 rank。`monitored_barrier` 通过之后所有 rank 都已到达，`all_gather_object` 不再等待缺席的 rank。
- gloo 的收发本身报错时（例如某个 rank 的进程已经退出），这个异常不经过 `ErrorSummary`，由驱动最外层的 `except BaseException` 释放资源后抛出。
- world size 为 1 时不调用 `torch.distributed`，各确认点在本进程内完成，核对和建组的规则不变。

#### 7.2 各个确认点

```python
# vllm_omni/diffusion/weight_stack/group.py
@dataclass(frozen=True)
class Point0Payload:
    record: LocalRecord                      # HostWeightManager.local_record()
    coordinates: RankCoordinates             # 本 rank 在各并行维度上的大小和编号
    deployment: tuple[str, bool]             # HostWeightManager.deployment_id()

@dataclass(frozen=True)
class Point1Payload:
    fingerprints: RankFingerprints
    group_keys: frozenset[tuple[tuple[int, ...], str]]   # 进程级 AllGather 组表里已有的（成员，后端）
    unmatched_policy_keys: frozenset[str]                 # 本 rank 的目录里没有匹配的、if_absent 为 error 的策略键（05 的 L01）
```

| 确认点 | 轮次名 | 每个 rank 交出的内容 | 收集之后每个 rank 做的事 |
|---|---|---|---|
| 第 0 点 | `point0_topology` | 交出 `Point0Payload`。 | 核对部署标识，组装同一份 `Topology`。 |
| 第 1 点第一轮 | `point1_fingerprints` | 交出 `Point1Payload`。 | 核对参与集合通信的部分；计算 AllGather 组并按固定顺序创建或复用；然后进入本地步骤 `confirm`：调用 `bind_groups`、`share_plan()`、`elect()`。 |
| 第 1 点第二轮 | `point1_elections` | 交出本 rank 的 `Election.record()`。 | 进入本地步骤 `admit`：组装 `Elections`，调用 `admit()`。 |
| 第 2 点 | `point2_ready` | 只交出成功或失败，没有其他内容。 | 进入 `TRANSACTION_READY`，然后调用 `connect_writer()`。 |
| 第 3 点 | `point3_commit` | 只交出成功或失败，没有其他内容。 | 第 3 点成功之后才能进入 `IN_SERVICE`；失败时每个 rank 关闭自己的运行时。 |

本地步骤 `confirm` 的错误带到第二轮报告。`admit()` 是纯函数，各 rank 的输入相同，超出容量时所有 rank 同时报 `HostCapacityError`；它是第二轮之后的本地步骤，错误在第 2 点报告，这时还没有分配 host 存储，也没有锁页（总体设计 4.5 第 1 点）。

**部署标识**。每个 rank 交出 `HostWeightManager.deployment_id()` 的结果，所有 rank 采用 0 号 rank 的那一对值作为 `Topology.deployment_id` 和 `Topology.deployment_scoped`。0 号 rank 的 `od_config.deployment_id`（03 增加的内部字段）为 `None` 时，它交出的是新生成、只在本副本内有效的标识，`deployment_scoped` 为假（总体设计 6.3）。任何一个 rank 交出的第二项为真、而它的标识与 0 号 rank 采用的不同时（包括 0 号 rank 的第二项为假），报 `GroupConfirmationError`。03 合入之前 `od_config` 没有这个字段，`HostWeightManager` 的实现也不存在，第 1 步已经报 `StackUnavailableError`。

**坐标**。每个 rank 在第 0 点从 `vllm_omni/diffusion/distributed/parallel_state.py` 和 vLLM 的张量并行状态读出本 rank 在各并行维度上的大小和编号，填进 `RankCoordinates`。没有启用专家并行时 `ep` 为 `None`，没有启用 HSDP 时 `fs` 为 `None`。`Topology.coordinates` 用于 `plan()` 推算 AllGather 组的大小，以及报错和启动日志里写明 rank 的位置。`plan()` 推算的组大小在第 1 点由 `bind_groups` 与实际的组核对。

**核对**。只核对参与集合通信的部分（总体设计 4.5 第 1 点）：

- 某个组件在任何一个 rank 的 `collective` 里出现时，每个在 `content_identities` 里声明了这个组件的 rank，`collective` 里也必须有它；
- 同一个 AllGather 组里，各成员的 `collective[component]` 必须相同。
- 不一致时报 `GroupConfirmationError`，写明组件名和取值不同的 rank。`rank-local` 的 Unit 和只在部分 rank 上声明、不用 `allgather` 的组件不参与核对。

**策略键的匹配**。组件可以按 rank 声明（02 第 2.5 节），所以 `if_absent` 为 `error` 的策略键只有在本副本每个 rank 的目录里都没有匹配时才报错（05 的 L01）。每个 rank 交出 `PlacementPlan.unmatched_policy_keys()`（04），收集之后取所有 rank 的交集；交集不为空时，每个 rank 以同样的内容报 `PlanningError`，写明这些键和本 rank 声明的组件。这项判断只依赖收集到的结果，所以各 rank 同时失败。world size 为 1 时，交集就是本 rank 的集合。

**AllGather 组**。对每个出现在 `collective` 里的组件，声明了它的 rank 按 `content_identities[component]` 分组，内容身份相同的 rank 组成一个 AllGather 组（总体设计 6.4）。

- 组里只有一个 rank 时报 `GroupConfirmationError`（总体设计 8.5）。world size 为 1 时，任何使用 `allgather` 的组件都在这里报错。
- 全部组按（组件名，组内最小 rank）排序。一个组的（成员，后端）出现在每个 rank 交出的 `group_keys` 里时，所有 rank 复用进程级表里的组，不调用 `new_group`；否则每个 rank，包括不在这个组里、没有声明这个组件的 rank，按这个顺序调用 `torch.distributed.new_group(ranks, backend=current_omni_platform.dist_backend)`，新组替换表里成员相同的旧组，旧组留到 `destroy_allgather_groups()` 一起销毁。复用的判断只依赖收集到的结果，所以各 rank 调用 `new_group` 的次数和顺序相同。
- 结果作为 `AllGatherGroups` 交给 `HostLayout.bind_groups()`（03）和 `PlacementPlan.bind_groups()`（04）。

### 8. worker 与模型运行器的分发

#### 8.1 加载

```text
执行器（多进程、Ray、单进程）
  DiffusionWorker.__init__
    init_device()                                 init_distributed_environment、initialize_model_parallel（不变）
    try:
      set_process_weight_stack(od_config.diffusion_weight_stack, owner=self)   新增
      构造 model runner                           DiffusionModelRunner 或它的子类（不变）
      DiffusionWorker.load_model(load_format)     set_forward_context、set_current_vllm_config、cutlass 上下文（不变）
        unified：构造 MemoryPools（8.2），作为 memory_pools 传给 runner
        DiffusionModelRunner.load_model(...)
          load_format 为 "dummy"：直接返回（不变）
          legacy：现有语句（不变）
          unified：_load_model_unified(...)
      DiffusionWorker.init_lora_manager()         unified 下的改动是必改清单 C16（02）
    except：调用 shutdown()（不变），shutdown 里新增的 clear_process_weight_stack 同样执行
```

`DiffusionModelRunner.load_model` 在 `load_format == "dummy"` 的判断之后按 `is_unified(self.od_config)` 分支。`legacy` 一支是现有的语句，不改动。`unified` 一支：

```text
_load_model_unified(memory_pools, load_format, custom_pipeline_name)
  self.weight_stack 不是 None 时报 LifecycleError                            旧的运行时必须先关闭（8.5）
  enable_pec = od_config.enable_prompt_embed_cache                           不读环境变量（总体设计 8.5）
  pec_size = od_config.prompt_embed_cache_size
  current_omni_platform.init_diffusion_model_runner_runtime(...)             与 legacy 相同
  ctx = StartupContext(..., policy=05 解析出的策略或 None, faults=FaultPlan.from_environ())
  with DeviceMemoryProfiler() as m:
      handle = run_unified_startup(ctx)                                      第 1 到第 12 步
  self.weight_stack = handle；self.pipeline = handle.model；self.offload_backend = None
  self.model_memory_usage = m.consumed_memory
  step_execution、streaming_output 的检查                                   与 legacy 相同
  self._install_runtime_services(enable_pec, pec_size)                      缓存后端、prompt embedding 缓存、InteractionCoordinator
  handle.mark_in_service()
```

- `_install_runtime_services` 是 `load_model` 里设置缓存后端、安装 prompt embedding 缓存、构造 `InteractionCoordinator` 的那几条语句，原样移进这个方法，两个栈共用。
- `unified` 一支不用 `memory_pool_context_fn` 包住整个加载，睡眠池经 `StartupContext.memory_pools` 交给驱动。`DiffusionModelRunner.load_model` 增加一个关键字参数 `memory_pools: MemoryPools | None = None`，只在 `unified` 下由 worker 传入。
- `ctx.compile_enabled` 为 `not od_config.enforce_eager and current_omni_platform.supports_torch_inductor()`，与 legacy 选择是否编译的条件相同。
- `unified` 下 `load_format` 为 `default`、`diffusers` 或 `custom_pipeline` 时都走驱动，只建结构的方式由 Loader 按 `load_format` 选择（02）。
- prompt embedding 缓存与文本编码器 `allgather` 的组合检查，`legacy` 下照旧在 runner 里做；`unified` 下由 05 在参数解析阶段做（总体设计 8.5）。`legacy` 一支照旧调用 `resolve_prompt_embed_cache_config`（`vllm_omni/diffusion/cache/prompt_embed_cache.py`），它在 worker 里读 `OMNI_DIFFUSION_PROMPT_EMBED_CACHE` 和 `OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE`；`unified` 一支不调用它。
- `ARDiffusionModelRunner.load_model` 调用父类的 `load_model`，自动走同一个分支。
- 驱动返回之后、`mark_in_service()` 之前的任何一步失败时，runner 调用 `close_weight_stack()` 后抛出，worker 初始化失败。
- 一个进程里同一时刻只运行一次 `unified` 启动。单进程执行器把几个 worker 放在同一个进程里时，各 worker 的启动依次进行：02 的只建结构上下文在一个进程里同一时刻只能打开一个，读取陷阱对整个进程生效。第二次启动在上下文打开期间进入时报 `LifecycleError`，不排队：排队的启动可能是持有上下文的 worker 在 `seal()` 里等待的写入进程。

#### 8.2 睡眠池

`unified` 下，`DiffusionWorker` 构造一个 `MemoryPools` 传给 runner：

- 没有启用睡眠模式时，`weights()` 和 `slots()` 都是空的上下文。
- 启用睡眠模式时，构造时检查一次 CuMem 分配器的占用为 0，与 `_maybe_get_memory_pool_context` 对 `weights` 池的检查相同，并设置 `_owns_sleep_pool`。`weights()` 返回 `allocator.use_memory_pool(tag="weights")`，`slots()` 返回 `allocator.use_memory_pool(tag="weight_slots")`。
- `vllm_omni/diffusion/data.py` 的 `CuMemTag` 增加 `WEIGHT_SLOTS = "weight_slots"`。

检查只做一次，因为 Offloader 在第 8 步、Loader 在第 9 步分多次进入睡眠池，`_maybe_get_memory_pool_context` 每次进入 `weights` 池都检查占用为 0，不适用于这种用法。`legacy` 继续使用 `_maybe_get_memory_pool_context`，不改动；`kv_cache` 池在两个栈下都照旧经它进入。

#### 8.3 关闭

```text
DiffusionWorker.shutdown
  model_runner.offload_backend.shutdown()     legacy（不变）
  model_runner.close_weight_stack()           unified：本地，不发起集合通信
  KV 传输、KV connector、a2a 工作区的释放（不变）
  destroy_allgather_groups()                  unified：第 1 点创建的 AllGather 组
  destroy_distributed_env()（不变）
  clear_process_weight_stack(owner=self)
  CuMem 池的释放（不变）
```

`DiffusionModelRunner.close_weight_stack()` 取出 `self.weight_stack` 并置为 `None`，调用 `handle.close()`，把 `self.pipeline` 置为 `None`。它可以多次调用。`shutdown` 里新增的三步与现有各步一样各自放在 `try` 块里，异常记日志后继续执行后面的步骤。单进程执行器把 worker 放在编排进程里时，私有内存、共享映射、监听套接字、内存登记和加速卡资源都由这次关闭释放，不靠进程退出（总体设计 6.5）。

#### 8.4 睡眠和唤醒

```text
DiffusionWorker.sleep(level)
  现有的前置检查、level 2 释放 CUDA graph 和保存 buffer 的语句（不变）
  unified：self.model_runner.weight_stack.runtime.before_sleep()
  allocator.sleep(...) 及其后的语句（不变）：level 1 只把 "weights" 作为 offload_tags，其余池释放、不备份

DiffusionWorker.wake_up(tags)
  unified：tags 为 None 或包含 "weights" 时，同时唤醒 "weight_slots"
  现有语句（不变）
  unified：self.model_runner.weight_stack.runtime.after_wake()
```

- 没有运行时（`load_format` 为 `dummy`）时，`unified` 下的睡眠和唤醒与 `legacy` 相同。
- level 2 保存 buffer、释放 CUDA graph 的语句不变；level 2 之后的唤醒不在范围之内（总体设计 1.2）。

#### 8.5 `custom_pipeline` 的重新初始化

```text
WorkerWrapperBase.__init__（uses_custom_pipeline 为真），或者经 collective_rpc 调用的 "re_init_pipeline"
  CustomPipelineWorkerExtension.re_init_pipeline(custom_pipeline_args)
    unified：self.model_runner.close_weight_stack()；self.lora_manager = None；gc.collect()；empty_cache
    legacy：现有语句（不变）
    self.load_model(load_format="custom_pipeline", custom_pipeline_name=...)
    self.init_lora_manager()
```

`unified` 下，关闭运行时和本进程的 host 存储映射之后，完整走一遍启动流程，包括第 0 到第 3 点和 HWM 的选举（总体设计 5.2）。运行期的 `re_init_pipeline` 经 `StageDiffusionProc._handle_collective_rpc`（`vllm_omni/diffusion/stage_diffusion_proc.py`）交给 `DiffusionEngine.collective_rpc`，执行器在没有指定回复 rank 时在每个 rank 上执行，所以本副本所有 rank 一起调用，各个确认点因此对齐。首次加载（`skip_load_model` 为真、在 `WorkerWrapperBase.__init__` 里调用 `re_init_pipeline`）时没有旧的运行时，`close_weight_stack()` 什么都不做。

### 9. 共存期

#### 9.1 `offloader/config.py` 的查询函数

`unified` 下，`vllm_omni/diffusion/offloader/config.py` 里每个查询函数的结果，等于 `legacy` 下同一个函数对不含任何 offload 设置的配置的结果。模型里读这些函数的分支因此按“不做 offload”执行，这是这些分支在 `legacy` 下已经覆盖的状态。

| 函数 | `unified` 下的结果 |
|---|---|
| `resolve_offload` | 返回 `strategy` 为 `OffloadStrategy.NONE`、`public` 为 `None`、`components` 为 `DEFAULT_OFFLOAD_COMPONENTS`、`transfers` 里 `dit` 为 `ALLGATHER` 和 `text_encoder` 为 `RANK_LOCAL`、`pin_memory` 取 `pin_cpu_memory`、`resident_layers` 为 0 的 `ResolvedOffload`。它不执行 legacy 的校验，也不写 `_resolved_diffusion_offload` 缓存。 |
| `resolve_offload_strategy` | 返回 `OffloadStrategy.NONE`。 |
| `offload_enabled`、`offload_streams_blocks`、`should_offload_component`、`any_selected_component_uses_allgather` | 返回 `False`。 |
| `selected_offload_components`、`component_uses_allgather` | 结果与 legacy 不做 offload 时相同，由上面的 `ResolvedOffload` 算出。 |
| `materialize_legacy_offload_flags` | 不写任何字段，返回 `OffloadStrategy.NONE`。 |
| `parse_diffusion_offload_config`、`parse_offload_components`、`parse_dlo_transfer` | 纯解析函数，行为不变。 |

`legacy` 下这些函数的行为不变。

#### 9.2 引擎和执行器读到的 offload 事实

引擎和执行器不是模型代码。它们读查询函数，用来判断 DP 下各 rank 是否同步执行、文本编码器是否参与集合通信、输出是否在后处理之前移到 CPU：

- `diffusion_engine.py` 的 `_uses_dlo_dp_concurrency`，以及 `DiffusionEngine.postprocess_output` 里两处 `OffloadStrategy.MODEL_LEVEL` 的判断；
- `multiproc_executor.py` 的 `_uses_text_encoder_allgather`，以及 `MultiprocDiffusionExecutor.execute_request` 和 `execute_batch` 里的 `any_selected_component_uses_allgather`；
- `ray_executor.py` 的 `_uses_dlo_dp_concurrency`、`_uses_text_encoder_allgather`，以及 `RayDiffusionExecutor.execute_request` 里的 `any_selected_component_uses_allgather`。

`unified` 下查询函数报告“不做 offload”，这些调用点如果继续读查询函数，DP 下 `allgather` 需要的同步执行就会关掉（总体设计 6.4 的约束）。它们改读 `startup.execution_offload_facts(config)`。`ExecutionOffloadFacts` 是跨模块的数据类型，放在 `interfaces.py`：

```python
# vllm_omni/diffusion/weight_stack/interfaces.py
@dataclass(frozen=True)
class ExecutionOffloadFacts:
    weight_collectives: bool        # 有 allgather 的 Unit，各 DP rank 必须以相同顺序执行相同的块
    text_encoder_allgather: bool    # 文本编码器的 Unit 用 allgather
    whole_component_offload: bool   # 有整组件 Unit，输出在后处理之前移到 CPU
```

| 字段 | `legacy` 下的取值 | `unified` 下的取值 |
|---|---|---|
| `weight_collectives` | `any_selected_component_uses_allgather(config)` 的值。 | 05 解析出的策略里有被选中的组件用 `allgather` 时为真。 |
| `text_encoder_allgather` | `_uses_text_encoder_allgather` 现有表达式的值。 | 05 解析出的策略里 `text_encoder` 角色或属于这个角色的组件名用 `allgather` 时为真。 |
| `whole_component_offload` | `resolve_offload_strategy(config) is OffloadStrategy.MODEL_LEVEL` 的值。 | 05 解析出的策略里有组件按 `module` 选中时为真。 |

- 调用点上 DP 大小、请求数等其余条件不变，只把读查询函数的那一项换成对应的字段。`legacy` 下每个字段与原表达式逐项相同，引擎和执行器的行为不变。
- `unified` 下的取值只读策略，不读规划：引擎和执行器所在的进程没有规划结果。策略选了某个组件、而模型里没有这个组件时，字段可能为真而实际没有 `allgather` 的 Unit，这时 DP 下多做的同步执行不影响正确性。
- 在 05 的策略解析合入主干之前，`unified` 不接受 offload 配置（1.3），三个字段都为假。
- `execution_offload_facts` 放在 `weight_stack/startup.py`，`legacy` 一支在函数内 import `offloader/config.py`，`unified` 一支在函数内 import `weight_stack.policy`。删除 `legacy` 时只删 `legacy` 一支，函数保留。

#### 9.3 共存期必改清单的规则

每个必改 PR 都必须满足以下规则：

1. 清单里的每一项只改变 `unified` 下的行为。`legacy` 一支是原来的语句，原样移进分支，执行路径、参数和调用顺序不变。
2. 每一项和启用对应策略的 Job 一起改，由表里写明的子 RFC 承担。
3. 改动不修改已有测试文件。新增的测试覆盖 `unified` 一支，已有测试在 `legacy` 下全部通过。
4. 模型代码只经 `weight_stack.components.is_unified_stack()` 读总开关，或者经 02 的构造上下文判断是否在只建结构的上下文里。PR 新增一个位置文件，登记这些位置（9.4）。
5. 按总开关或构造上下文分出的 legacy 一支随 `legacy` 在 Job 6 删除。

承担方：

| 编号 | 位置 | 承担方 |
|---|---|---|
| C01 | MAGI-2 pipeline 的两处 offload 校验、分阶段组件、内存窗口。 | 由 [04](./04-unified-offloader.md) 承担。 |
| C02 | MAGI-2 MoE 加载后把 `W_gate`、`W_up` 改绑成 `_owned_w13` 的视图；只建结构时注册合并参数 `W13`。 | 由 [02](./02-unified-loader.md) 承担。 |
| C03 | MiniMax H3 transformer 的 SM120 AdaLN offload 自动默认，以及放大器的搬运。 | 由 [04](./04-unified-offloader.md) 承担。 |
| C04 | MiniMax H3 的 AdaLN sidecar 在 `unified` 下由 Loader 写入非持久 buffer。 | 由 [02](./02-unified-loader.md) 承担。 |
| C05 | MiniMax H3 视频 VAE 按 `next(params)` 判断设备、类型不是 fp32 时调用 `self.to(float32)`。 | 由 [02](./02-unified-loader.md) 承担。 |
| C06 | MiniMax H3 多输出时的 `_offload_model_cpu_stage_output`。 | 由 [04](./04-unified-offloader.md) 承担。 |
| C07 | 精确投影缓存、AdaLN 缓存的签名改用参数身份加写入代数。 | 由 [04](./04-unified-offloader.md) 承担。 |
| C08 | LingBot-Video 的 `offload_vae_during_denoise`。 | 由 [04](./04-unified-offloader.md) 承担。 |
| C09 | SANA-WM 每个请求的 `_ensure_*`。 | 由 [02](./02-unified-loader.md) 承担。 |
| C10 | HunyuanImage3 `load_weights` 开头对零散模块的 `.to(device)`。 | 由 [02](./02-unified-loader.md) 承担。 |
| C11 | Kandinsky6 的 `self.device`。 | 由 [02](./02-unified-loader.md) 承担。 |
| C12 | Qwen-Image-2.1、SenseNova-U1、AuK、OmniVoice 的 CUDA graph，Ming-Image、InternVLA-A1、DreamZero 的 cudagraph 编译模式，经通用函数按 `fully_resident` 门控。 | 由 [04](./04-unified-offloader.md) 承担。 |
| C13 | Wan VAE 快路径的 `tensor.data=` 恢复。 | 由 [02](./02-unified-loader.md) 承担。 |
| C14 | Cosmos3 混合精度的 NVFP4 查找表。 | 由 [04](./04-unified-offloader.md) 承担。 |
| C15 | LTX2 vocoder 在 mps、xpu、cpu 上的 `.float()` 往返。 | 由 [02](./02-unified-loader.md) 承担。 |
| C16 | LoRA 管理器提交后才包装 PEFT 层，以及 worker 提交后融合蒸馏 LoRA。 | 由 [02](./02-unified-loader.md) 承担。 |
| C17 | runner 在分布式逐层 offload 下把 `inference_mode` 换成 `no_grad`；`unified` 下改读 `OffloadRuntime.grad_mode()`。 | 由 [04](./04-unified-offloader.md) 承担。 |
| C18 | diffusers 适配器的 `_raise_unsupported_features` 在 `__init__` 里拒绝 `diffusion_offload_config`；`unified` 下这项拒绝不执行，旧配置由 05 翻译。 | 由 [02](./02-unified-loader.md) 承担。 |

C01 到 C17 是 umbrella RFC Job 1 表里的各行，按原顺序编号。C18 是本 RFC 新增的一行：`DiffusersAdapterPipeline` 的构造函数调用 `_raise_unsupported_features`，后者在 `od_config.diffusion_offload_config` 不是 `None` 时抛出 `NotImplementedError`；这项拒绝只为现有后端存在（总体设计 7.7），而 `unified` 下旧的 `--diffusion-offload-config` 由兼容层翻译（总体设计 8.4）。

本 RFC 自己的分发点也读总开关，同样登记，随 `legacy` 删除。01-2、01-3、01-6 合入时登记表还不存在，这五行的位置文件由 01-7 建立登记表时一并新增：

| 编号 | 位置 | 承担方 |
|---|---|---|
| D01 | `OmniDiffusionConfig.__post_init__` 的参数解析检查和对 `materialize_legacy_offload_flags` 的跳过。 | 由 PR 01-2 承担。 |
| D02 | `offloader/config.py` 的 `resolve_offload` 和 `materialize_legacy_offload_flags`。 | 由 PR 01-3 承担。 |
| D03 | `DiffusionModelRunner.load_model` 的分支。 | 由 PR 01-6 承担。 |
| D04 | `DiffusionWorker` 的 `__init__`、`load_model`、`shutdown`、`sleep`、`wake_up`。 | 由 PR 01-6 承担。 |
| D05 | `CustomPipelineWorkerExtension.re_init_pipeline`。 | 由 PR 01-6 承担。 |

#### 9.4 登记表与跟踪

登记表放在测试目录里，分成冻结的行表和每个 PR 新增的位置文件：

```text
tests/diffusion/weight_stack/coexistence/
  __init__.py
  registry.py                   行表 ROWS 和四个冻结白名单（第 11 节），由 01-7 建立
  sites/<行编号>_<PR 编号>.py    每个必改 PR 新增一个，PR 编号里的连字符写成下划线
```

```python
# tests/diffusion/weight_stack/coexistence/registry.py
@dataclass(frozen=True)
class CoexistenceRow:
    row_id: str                  # "C01" 到 "C18"、"D01" 到 "D05"，或者其他子 RFC 用 NEW_ROW 声明的 "D06" 起的编号
    title: str
    carrier: str                 # 承担的子 RFC，"01" 到 "05" 之一
    legacy_removal: str          # 删除 legacy 时要删的语句

ROWS: tuple[CoexistenceRow, ...]

# tests/diffusion/weight_stack/coexistence/sites/<行编号>_<PR 编号>.py
ROW: str                                  # ROWS 里的一个 row_id，或者本文件 NEW_ROW 的 row_id
PR: str                                   # 合入的 PR 编号
SITES: tuple[str, ...]                    # 这个 PR 新增的读总开关或构造上下文的位置，写成 "<相对路径>::<限定名>"
NEW_ROW: CoexistenceRow | None = None     # 其他子 RFC 自己的分发点在这里声明新的一行，编号从 D06 起
```

- `registry.py` 由 01-7 建立，`ROWS` 含 C01 到 C18 和 D01 到 D05，之后只在 Job 6 删除 legacy 时修改。必改 PR 不修改 `registry.py` 和其他已有测试文件，只在 `sites/` 下新增自己的文件，因此满足 9.3 第 3 条。
- 02、03、04、05 自己的分发点（例如 02 在 `models/utils.py` 的分发点）不在 `ROWS` 里，由承担它的 PR 在位置文件里用 `NEW_ROW` 声明，`ROW` 等于 `NEW_ROW.row_id`。`NEW_ROW` 的编号不能与 `ROWS` 和其他位置文件的 `NEW_ROW` 重复。
- 位置文件不以 `test_` 开头，pytest 不收集它们；静态测试（第 11 节 T6）按路径加载 `sites/` 下的全部文件。
- 每个位置文件的 `ROW` 必须在 `ROWS` 里或者等于本文件 `NEW_ROW` 的编号，`PR` 不为空；`SITES` 里的每个位置必须存在，即文件存在并且限定名能在这个文件的语法树里找到。一个 PR 没有新增读总开关的位置时，`SITES` 为空，文件照样新增，用来记录这一行已由这个 PR 完成。
- T6 在 `vllm_omni/` 里找出 `weight_stack/`、`model_loader/construct.py`、`model_loader/hub_prefetch.py` 以外每个读总开关或构造上下文的位置，每个位置必须出现在某个位置文件的 `SITES` 里。一行拆成多个 PR 时有多个位置文件。
- 测试在日志里输出每一行的状态：有哪些位置文件、对应哪些 PR。默认值翻转为 `unified` 之前，`ROWS` 里的每一行都至少有一个位置文件。
- `ROWS`、各个 `NEW_ROW` 和位置文件就是 Job 6 删除 legacy 一支的清单：每个 `SITES` 位置只保留 `unified` 一支，按 `legacy_removal` 删除其余语句，然后删除这一行和它的位置文件。

#### 9.5 必改 PR 的评审清单

- PR 描述写明登记表的行编号和这一行的 `legacy_removal`。
- `legacy` 一支是原来的语句，diff 只显示把它们移进分支。
- 没有修改已有测试文件，包括 `registry.py`；新增的测试覆盖 `unified` 一支。
- 模型代码读总开关只用 `weight_stack.components.is_unified_stack()`，不读 `od_config.diffusion_weight_stack`，不 import `weight_stack` 里 `components` 以外的模块。
- PR 新增了 `sites/<行编号>_<PR 编号>.py`，新增的每个读总开关或构造上下文的位置都写进了它的 `SITES`。
- 模型代码（`models/`、`lora/`、`cache/`）里的 `unified` 一支不 import Offloader，不读 offload 或 host 的配置字段，不按名字读钩子，不调用 `OffloadRuntime` 的方法。
- 没有往冻结的白名单里加条目。
- 一行要改多个家族时可以拆成多个 PR，每个 PR 各自满足以上各条，各自新增位置文件。

### 10. 故障注入点

命名的故障注入点：

```text
construct_modules   build_structure   point0_topology   plan              preflight
point1_fingerprints confirm           point1_elections  admit             allocate
prepare             point2_ready      connect_writer    write             finalize
commit              point3_commit     compile_setup     close
```

- 本地步骤的注入点在 `local()` 里、调用这一步之前触发；确认点的注入点在 `round()` 里、计算要交出的内容之前触发（6.2）。确认点上的 `raise` 让本 rank 带着错误进入这一轮，`hang` 让本 rank 不到达这一轮。
- `compile_setup` 注入点在编译设置的 `try` 块里触发，只在 `ctx.compile_enabled` 为真时生效；它的 `raise` 只产生一次警告，启动继续。
- `close` 注入点在 `handle.close()` 里、调用 `runtime.close()` 之前触发。`handle.close()` 照常调用 `runtime.close()`、丢弃模型、进入 `CLOSED`，然后报 `LifecycleError`，注入的异常作为 `__cause__`。
- `FaultPlan` 把注入点映射到 rank 集合和动作。动作是 `raise`（抛出 `InjectedFault`，驱动按第 3 节的表包装）或 `hang`（阻塞，用于超时测试）。不给 rank 时对所有 rank 生效。
- runner 用 `FaultPlan.from_environ()` 构造它，读环境变量 `VLLM_OMNI_WEIGHT_STACK_FAULTS`，格式是逗号分隔的 `<注入点>[@<rank>][:raise|:hang]`。变量不存在时为 `None`，驱动不做任何检查。这个变量只用于测试，不写进用户文档。
- 02、03、04 在各自模块内部增加更细的注入点（例如 03 的封存），命名规则相同，用同一个 `FaultPlan`。

### 11. 静态依赖测试与冻结白名单

| 编号 | 检查 | 范围和规则 |
|---|---|---|
| T1 | `weight_stack` 内部的 import 方向 | 用 AST 扫描运行期 import，包括函数内的 import，跳过 `if TYPE_CHECKING:` 的块。`weight_stack/` 里的任何模块都不 import `vllm_omni.host_weight_runtime`（总体设计 1.2）。`weight_stack/` 里只有 `startup.py` 的 `execution_offload_facts` 可以 import `vllm_omni.diffusion.offloader`，而且只在函数内 import `vllm_omni.diffusion.offloader.config`。`loader/` 不 import `host/`、`offload/`。`offload/` 不 import `loader/`、`vllm_omni.diffusion.model_loader`、`safetensors` 和 `vllm.model_executor.model_loader`。`host/` 不 import `loader/`、`offload/` 和 `vllm_omni.diffusion.model_loader`。`policy.py` 不 import `loader/`、`host/`、`offload/`。`interfaces.py`、`errors.py` 只 import 标准库、`torch` 和 `weight_stack.errors`。`group.py`、`startup.py` 不 import `loader/`、`host/`、`offload/`：三个工厂函数由 `EntryPointFactory` 经 `importlib` 按名字取得；`execution_offload_facts` 的 `unified` 一支在函数内 import `weight_stack.policy`。 |
| T2 | 运行期的模块图 | 在子进程里以 `import vllm_omni` 之后已加载的模块为基线，分别 import `weight_stack.loader`、`weight_stack.host`、`weight_stack.offload`，只看新增的模块：不出现 T1 禁止的模块。 |
| T3 | 模型代码对新栈和 Offloader 的 import | `models/`、`lora/`、`cache/` 只能 import 新栈里的 `model_loader.construct`、`model_loader.hub_prefetch` 和 `weight_stack.components`，`if TYPE_CHECKING:` 里的 import 同样计入。import `vllm_omni.diffusion.offloader`（任何子模块，绝对或相对写法）的文件必须在冻结白名单 `OFFLOADER_IMPORT_ALLOWLIST` 里：现有的 28 个模型文件和 `lora/manager.py`。 |
| T4 | 按字符串名读 offload 钩子 | `models/`、`lora/`、`cache/` 里等于某个 offload 钩子名的字符串常量，只允许出现在冻结白名单 `HOOK_NAME_ALLOWLIST` 里：`cache/seacache/hook.py`。钩子名从 `SequentialOffloadHook`、`LayerwiseOffloadHook`、`DistributedLayerwiseOffloadHook` 的 `_HOOK_NAME` 和 04 导出的 `HOOK_NAMES` 读取。 |
| T5 | 直接读 offload 和 host 配置字段 | `models/`、`lora/`、`cache/` 里对 `enable_cpu_offload`、`enable_layerwise_offload`、`enable_distributed_layerwise_offload`、`dlo_use_allgather`、`dlo_resident_layers`、`dlo_host_registration_limit_gib`、`diffusion_offload_config`、`pin_cpu_memory`、`host_weight_runtime_mode`、`host_weight_runtime_root`、`diffusion_weight_stack` 以及 05 新增字段的属性访问和 `getattr` 字符串读取，只允许出现在冻结白名单 `CONFIG_FIELD_ALLOWLIST` 里：`models/diffusers_adapter/pipeline_diffusers_adapter.py`、`models/joy_image/pipeline_joy_image_edit.py`、`models/magi2/pipeline_magi2.py`、`models/minimax_h3/pipeline_minimax_h3.py`。`diffusion_weight_stack` 不在任何白名单里。 |
| T6 | 读总开关的位置 | `vllm_omni/` 里 `weight_stack/`、`model_loader/construct.py`、`model_loader/hub_prefetch.py` 以外，每个调用 `is_unified_stack`、`is_unified`、`process_weight_stack` 和 02 的构造上下文判断函数的位置，都必须出现在登记表某个位置文件的 `SITES` 里（9.4）。模型代码里只能调用 `is_unified_stack` 和构造上下文判断函数。直接读 `diffusion_weight_stack` 字段只允许出现在 1.1 列出的五个配置传递文件、`weight_stack/startup.py` 和 `DiffusionWorker.__init__` 里：前者只声明和传递这个字段，后两者只用它判断和设置总开关。 |
| T7 | 模型以外对查询函数的调用 | `offloader/`、`models/`、`lora/`、`cache/` 以外，调用 9.1 表里的查询函数的位置必须在冻结白名单 `QUERY_CALLER_ALLOWLIST` 里：`OmniDiffusionConfig.__post_init__`、`DiffusionModelRunner.load_model`、`DiffusionModelRunner._execute_request_list`、`DiffusersPipelineLoader.load_model`、`weight_stack/startup.py` 的 `execution_offload_facts`。白名单按 01-3 合入之后的状态冻结，那时引擎和执行器已经改读 `execution_offload_facts`。新的非模型调用方必须读 `execution_offload_facts`。 |
| T8 | 白名单冻结 | 测试文件里记录四个白名单排序后内容的 sha256。白名单的内容变化时测试失败。 |

- 四个白名单放在 `tests/diffusion/weight_stack/coexistence/registry.py` 里，T8 的摘要写在 `test_static_dependencies.py` 里。检查的规则是“实际出现的位置是白名单的子集”：白名单里的文件不再 import 时测试照样通过，必改 PR 不需要修改测试文件。
- T3 到 T5 的白名单写成相对 `vllm_omni/diffusion/` 的文件路径，T7 的白名单写成“<相对路径>::<限定名>”。
- 冻结白名单只在 Job 6 删除 `legacy` 时清空，同时删除 T8 的摘要。
- `.github/CODEOWNERS` 的源码规则由 `docs/design/module/` 下模块文档的前置元数据生成。01-7 新增 `docs/design/module/diffusion/weight_stack.md`，`primary_code_paths` 是 `vllm_omni/diffusion/weight_stack/**`，`validation_paths` 是 `tests/diffusion/weight_stack/**`，再生成 `CODEOWNERS` 里对应的规则；`tests/diffusion/weight_stack/coexistence/` 另加一条规则，登记表和白名单的改动由 weight-stack 的负责人评审。

### 12. 不变量的落实

| 不变量（总体设计 12） | 落实的子 RFC | 机制 |
|---|---|---|
| 1. 可预知的错误在第一次写模型之前报出 | 由 01、02、03、04、05 落实。 | 05 在参数解析阶段检查组合；04 在规划和 `prepare_materialization()` 里做加速卡准入；03 在第 1 点做 Host 准入；02 在预检里检查来源。驱动保证第 9 步只在第 2 点成功之后执行，故障注入测试断言第 9 步之前的任何失败都没有写模型。 |
| 2. 每个最终张量只写一次 | 由 02 落实。 | Loader 按参数跟踪写入，在收尾第 4 步核对。 |
| 3. 收尾后字节不变，提交后基础权重不可变 | 由 02、03、04 落实。 | 02 按固定顺序收尾；03 封存共享存储；04 在提交后装上守卫。 |
| 4. 内容身份不含 offload 策略等 | 由 02 落实。 | 02 按总体设计 5.5 计算内容身份，并用测试断言它不含这些项。 |
| 5. 一个 rank 稳定后只持有规划指定的一份 | 由 03 落实。 | `allocate()` 按共享计划分配；分片后端的瘦路径归还页。 |
| 6. 不同部署之间不共享、不连接 | 由 03、01 落实。 | 03 的套接字名含部署标识，交接时再核对；没有下发部署标识时，03 的 `deployment_id()` 生成只在本副本内有效的标识。01 在第 0 点让副本内所有 rank 采用 0 号 rank 交出的部署标识，并核对各 rank 下发的标识一致。 |
| 7. 一个共享集合在一台机器上最多一个写入进程 | 由 03 落实。 | 03 用抽象套接字的原子绑定选出写入进程。 |
| 8. `seal()` 里不做集合通信 | 由 03、02、01 落实。 | 03 的 `seal()` 只做套接字和映射；02 在第 6 步之前完成收尾里的集合通信；01 在第 2 点之后才让等待者连接写入进程，第 3 点在收尾之后。 |
| 9. 让出不回写 | 由 04 落实。 | 让出只释放加速卡存储。 |
| 10. 进程退出后内存里不留权重 | 由 03、01 落实。 | 03 使用无名内存文件并处理 fork；01 保证关闭路径调用 `runtime.close()`，资源账本里每个句柄只登记一次。 |
| 11. 关闭不恢复 CPU 权重、不发起集合通信 | 由 04、01 落实。 | 04 的 `OffloadRuntime.close()` 不恢复权重；01 的 `close_weight_stack()`、`release()` 和资源账本不调用 `torch.distributed`，测试把集合通信函数替换成报错的函数来验证。 |
| 12. 集合通信失败后整组终止 | 由 01、04 落实。 | 01 的确认点上任何 rank 出错时所有 rank 一起报错，超时有上限；04 在运行期把运行时标记为不可用并报 `RuntimeFault`，worker 退出后执行器终止整个副本。 |
| 13. 提交后模块结构不变 | 由 02、04 落实。 | 02 在只建结构时完成叠加层包装；04 在提交后装上守卫。 |
| 14. CUDA graph 只读 `fully_resident` 组件 | 由 04 落实。 | `fully_resident` 由 `commit()` 填写，钩子在捕获期间被调用时报错。 |
| 15. 睡眠时只有块环缓冲区和分片槽位不备份 | 由 04、01 落实。 | 04 把这两类分配在 `slots()` 池里；01 的 worker 在睡眠前调用 `before_sleep()`，唤醒时连同 `weight_slots` 一起唤醒，并调用 `after_wake()`。 |

### 13. Job 1 单独交付的内容

- `legacy` 下的行为和已有测试不变。
- `--diffusion-weight-stack unified` 可以在参数解析阶段选中。配置里有 offload 设置时，参数解析阶段报 `ConfigError`（1.3）。
- 不做 offload 的配置在 `unified` 下进入启动驱动。02、03、04 的工厂函数还不存在时，每个 rank 在第 1 步报 `StackUnavailableError`，在第 0 点一起失败，报错写明缺少的模块并提示 `use --diffusion-weight-stack legacy`。驱动不退回 `legacy`。
- `load_format` 为 `dummy` 时 `unified` 与 `legacy` 相同，直接返回；之后的 `re_init_pipeline` 在 `unified` 下走驱动，同样在第 1 步报错。
- 启动驱动、确认点、资源账本、释放顺序、worker 的关闭、睡眠和唤醒、重新初始化，都由 `tests/diffusion/weight_stack/fakes.py` 里的测试替身沿“不做 offload”这条路径覆盖：替身的规划让所有 Unit 常驻加速卡，host 存储为空，提交返回的运行时只登记一个加速卡槽位。
- 查询函数在 `unified` 下报告“不做 offload”，引擎和执行器改读 `execution_offload_facts`。
- 静态依赖测试、冻结白名单和登记表生效，之后每个 `unified` 分支都受它们约束。

## 与其他子 RFC 的接口

### 本 RFC 提供

| 提供的内容 | 使用方 | 约定 |
|---|---|---|
| `interfaces.py` 的数据类型和 Protocol | 02、03、04、05 使用。 | 各实现按第 2 节的签名和规范实现，不在 Protocol 之外互相调用。总体设计 4.1 列出的方法签名不变，其余方法是本 RFC 为驱动增加的。 |
| `errors.py` 的错误类型和报错格式 | 02、03、04、05 使用。 | 各模块按第 3 节的表选择类型；不是 `WeightStackError` 的异常由驱动按步骤包装。 |
| `startup.py` 的 `ResourceKind`、`ResourceLedger` | 04 使用。 | 启动事务和 `OffloadRuntime` 把每个资源登记在账本里，提交时用 `transfer()` 移交，叠加层槽位用 `hand_over()` 移出。 |
| `startup.py` 的 `WeightStack`、`is_unified()`、`process_weight_stack()` | 02 和 05 使用。 | `weight_stack.components.is_unified_stack()` 返回 `process_weight_stack() is WeightStack.UNIFIED`；构造上下文和读取陷阱的分发读进程级状态；参数解析阶段用 `is_unified(config)`。 |
| `startup.execution_offload_facts` | 引擎和执行器使用，05 提供 `unified` 下的取值。 | `unified` 一支读 05 的策略。 |
| `startup.py` 的驱动、`StartupContext`、`EntryPointFactory`、`FaultPlan` | 02、03、04 使用。 | 三个工厂函数各自接收 `StartupContext`；模块内部的故障注入点使用同一个 `FaultPlan`。 |
| `group.py` 的 `AllGatherGroups` | 03、04 使用。 | 第 1 点之后经 `bind_groups()` 交给 `HostLayout` 和 `PlacementPlan`。 |
| `MemoryPools` | 02、04 使用。 | 驱动不进入任何睡眠池。02 在 `weights()` 里分配常驻加速卡的张量，包括收尾里新建的；04 在 `prepare_materialization()` 里于 `slots()` 分配块环缓冲区和分片槽位，于 `weights()` 分配叠加层槽位。 |
| 登记表 `tests/diffusion/weight_stack/coexistence/` | 02、03、04、05 使用。 | 承担必改行或自己分发点的 PR 在 `sites/` 下新增位置文件，自己的分发点用 `NEW_ROW` 声明；不修改 `registry.py` 和白名单。 |
| 测试替身 `tests/diffusion/weight_stack/fakes.py` | 02、03、04 使用。 | 各自的测试用替身替换另外两个模块，经 `run_unified_startup(ctx, factory)` 走完整的启动流程。 |

### 本 RFC 依赖的约定

| 子 RFC | 约定 |
|---|---|
| [02](./02-unified-loader.md) | `vllm_omni.diffusion.weight_stack.loader` 导出 `create_loader(ctx: StartupContext) -> Loader`，以及 `Preflight`（含 `content_identities: Mapping[str, str]` 和 `expected_manifest`）、`ExpectedManifest`、`FinalManifest`。`vllm_omni.diffusion.weight_stack.components` 导出 `ComponentCatalog` 和 `is_unified_stack()`。组件声明在 `Loader.close()` 之后仍然可以查询。Loader 自己在 `ctx.memory_pools.weights()` 里分配常驻加速卡的张量。`Loader.finalize()` 在第 6 步调用 `storage.seal()`，在此之前完成收尾里的全部集合通信。`model_loader/construct.py` 的 `is_building_structure()` 是构造上下文的判断函数，模型代码对它的调用按 T6 登记。02 承担 C02、C04、C05、C09、C10、C11、C13、C15、C16、C18。 |
| [03](./03-host-storage.md) | `vllm_omni.diffusion.weight_stack.host` 导出 `create_host_weight_manager(ctx: StartupContext) -> HostWeightManager`，以及 `LocalRecord`、`SharePlan`、`Election`、`ElectionRecord`、`HostLayout`、`MemoryRange`。`LocalRecord` 和 `ElectionRecord` 可以 pickle。`HostWeightManager.deployment_id()` 按第 2 节的规范实现。`elect()` 在每个 rank 上都可以调用，非代表返回空的 `Election`。`Election.close()`、`HostStorage.close()` 可以多次调用。`allocate()` 成功时接管 `Election`，失败时不接管。`HostStorage.connect_writer()` 在第 2 点之后由驱动调用。`HostLayout.plan_fingerprint` 只依赖配置和并行布局。03 给 `OmniDiffusionConfig` 增加 `init=False` 的 `deployment_id` 字段，没有下发时为 `None`。 |
| [04](./04-unified-offloader.md) | `vllm_omni.diffusion.weight_stack.offload` 导出 `create_offloader(ctx: StartupContext) -> Offloader`、`PlacementPlan`、`Unit` 和 `HOOK_NAMES`。`PlacementPlan.collective_signature()` 只含有 `allgather` Unit 的组件；`PlacementPlan.unmatched_policy_keys()` 返回 05 的 `resolve_selection()` 在本 rank 上没有匹配的、`if_absent` 为 `error` 的键。`prepare_materialization()` 失败时不接管 `storage`；`commit()` 失败时资源全部留在启动事务里。`plan()` 收到的 `policy` 为 `None` 时按不做 offload 规划。启动事务和运行时用 `ResourceLedger`。04 实现 `setup_compile()`、`grad_mode()`、`execution()`、`before_sleep()`、`after_wake()`，并承担 C01、C03、C06、C07、C08、C12、C14、C17，其中 C17 把 runner 前向的梯度模式改读 `grad_mode()`，同一个 PR 让前向经过 `execution()`。 |
| [05](./05-configuration.md) | `vllm_omni.diffusion.weight_stack.policy` 提供 `unified` 下在 `OmniDiffusionConfig.__post_init__` 里调用的解析入口，取代 1.3 的检查，并在解析时完成总体设计 8.5 的组合检查。`unified` 下 `diffusion_compile_granularity="full"` 与 offload 的冲突由 05 按策略检查，与 legacy 现有的检查等价。05 的策略对象提供 9.2 表里三个字段的取值，runner 把它填进 `StartupContext.policy`。05 的 PR 把新增的配置字段加进 T5 的字段表（`test_static_dependencies.py`）；字段表是检查规则，不是冻结白名单，不受 T8 约束。 |

## 代码组织

```text
vllm_omni/diffusion/weight_stack/
  __init__.py            01  只导出 LoadState、WeightStack 和错误类型
  interfaces.py          01  数据类型和 Protocol（第 2 节），ExecutionOffloadFacts
  errors.py              01  错误类型（第 3 节）
  group.py               01  确认点、Topology 的组装、AllGather 组的创建和进程级表
  startup.py             01  总开关和进程级状态、execution_offload_facts、ResourceKind、ResourceLedger、
                             run_unified_startup、EntryPointFactory、WeightStackHandle、FaultPlan
  components.py          02
  loader/                02
  host/                  03
  offload/               04
  policy.py              05

tests/diffusion/weight_stack/
  __init__.py
  fakes.py                         测试替身：Loader、HWM、Offloader、启动事务、运行时
  test_switch.py
  test_interfaces.py
  test_errors.py
  test_resources.py
  test_group.py                    world size 为 1
  test_group_multiproc.py          多进程 gloo
  test_startup.py
  test_startup_faults.py
  test_offload_query_coexistence.py
  test_execution_facts.py
  test_worker_dispatch.py
  test_worker_sleep_pools.py       L2，CUDA 单卡
  test_static_dependencies.py      T1 到 T8
  coexistence/
    __init__.py
    registry.py                    行表和四个冻结白名单（9.4、第 11 节）
    sites/                         每个必改 PR 新增的位置文件；01-7 新增 D01 到 D05 的位置文件

docs/design/module/diffusion/weight_stack.md     模块文档，前置元数据生成 CODEOWNERS 的规则
docs/contributing/                                必改 PR 的评审清单（9.5）
```

本 RFC 修改的已有文件：

| 文件 | 改动 |
|---|---|
| `vllm_omni/entrypoints/cli/serve.py` | `subparser_init` 增加 `--diffusion-weight-stack`。 |
| `vllm_omni/engine/arg_utils.py` | `OrchestratorArgs` 增加字段。 |
| `vllm_omni/config/stage_config.py` | `StageDeployConfig` 增加字段。 |
| `vllm_omni/config/omni_config.py` | `_DiffusionConfigProjection` 增加字段。 |
| `vllm_omni/diffusion/data.py` | `OmniDiffusionConfig` 增加字段和校验，`unified` 下跳过 `materialize_legacy_offload_flags` 并做 1.3 的检查；`CuMemTag` 增加 `WEIGHT_SLOTS`。 |
| `vllm_omni/diffusion/offloader/config.py` | 查询函数的 `unified` 分支（9.1）。 |
| `vllm_omni/diffusion/diffusion_engine.py`、`vllm_omni/diffusion/executor/multiproc_executor.py`、`vllm_omni/diffusion/executor/ray_executor.py` | 9.2 列出的调用点改读 `execution_offload_facts`。 |
| `vllm_omni/diffusion/worker/diffusion_model_runner.py` | `load_model` 的分支和 `memory_pools` 参数、`_load_model_unified`、`_install_runtime_services`、`close_weight_stack`、`weight_stack` 属性。 |
| `vllm_omni/diffusion/worker/diffusion_worker.py` | 进程级状态的设置和清除、睡眠池、`shutdown`、`sleep`、`wake_up`、`re_init_pipeline` 的 `unified` 分支。 |
| `.github/CODEOWNERS` | 由 `weight_stack.md` 的前置元数据生成 `weight_stack/` 的规则，另加 `tests/diffusion/weight_stack/coexistence/` 的规则。 |

`OmniDiffusionConfig.deployment_id` 由 03 增加，本 RFC 不改它；`HostWeightManager.deployment_id()` 读它。

删除 `legacy` 时删除的内容见“删除 legacy 时”一节。

## 实现拆分

| PR | 内容 | 文件 | 测试 | 依赖 | 大小 | 必改行 |
|---|---|---|---|---|---|---|
| 01-1 | `weight_stack` 包骨架；数据类型、Protocol、错误类型、`ResourceKind` 和 `ResourceLedger`。这个 PR 不改变任何已有行为。 | `weight_stack/__init__.py`、`interfaces.py`、`errors.py`、`startup.py`（只含账本） | `test_interfaces.py`、`test_errors.py`、`test_resources.py` | 不依赖其他 PR。 | 大小为 M。 | 不承担必改行。 |
| 01-2 | 总开关：`WeightStack` 和进程级状态，1.1 的传递链，1.3 的参数解析检查，`DiffusionWorker.__init__` 和 `shutdown` 里进程级状态的设置和清除。 | `startup.py`、`serve.py`、`arg_utils.py`、`stage_config.py`、`omni_config.py`、`data.py`、`diffusion_worker.py` | `test_switch.py` | 依赖 01-1。 | 大小为 S。 | 承担 D01。 |
| 01-3 | 查询函数的共存：`offloader/config.py` 的 `unified` 分支，`ExecutionOffloadFacts` 和 `execution_offload_facts`，引擎和执行器的调用点。 | `offloader/config.py`、`interfaces.py`、`startup.py`、`diffusion_engine.py`、`multiproc_executor.py`、`ray_executor.py` | `test_offload_query_coexistence.py`、`test_execution_facts.py` | 依赖 01-2。 | 大小为 S。 | 承担 D02。 |
| 01-4 | 全组确认：确认点的协议和超时、`Topology` 的组装、部署标识的核对、坐标的读取、AllGather 组的计算、创建、复用和进程级表。 | `group.py` | `test_group.py`、`test_group_multiproc.py` | 依赖 01-1。 | 大小为 M。 | 不承担必改行。 |
| 01-5 | 启动驱动：`run_unified_startup`、`EntryPointFactory`、`WeightStackHandle`、状态机、所有权链和释放、`FaultPlan`；测试替身和全部故障注入测试。 | `startup.py`、`tests/diffusion/weight_stack/fakes.py` | `test_startup.py`、`test_startup_faults.py`，以及 `test_group_multiproc.py` 里经驱动的用例 | 依赖 01-1 和 01-4。 | 大小为 L。 | 不承担必改行。 |
| 01-6 | worker 和 runner 的分发：`load_model` 的分支和 `memory_pools` 参数、`_load_model_unified`、`_install_runtime_services`、`close_weight_stack`、睡眠池和 `CuMemTag.WEIGHT_SLOTS`、`shutdown`、`sleep`、`wake_up`、`re_init_pipeline`。 | `diffusion_model_runner.py`、`diffusion_worker.py`、`data.py` | `test_worker_dispatch.py`、`test_worker_sleep_pools.py` | 依赖 01-2 和 01-5。 | 大小为 M。 | 承担 D03、D04、D05。 |
| 01-7 | 静态依赖测试 T1 到 T8，登记表 `registry.py` 和 D01 到 D05 的位置文件，四个冻结白名单，模块文档和 `CODEOWNERS`，写进 `docs/contributing/` 的评审清单。 | `tests/diffusion/weight_stack/coexistence/`、`docs/design/module/diffusion/weight_stack.md`、`.github/CODEOWNERS`、`docs/contributing/` | `test_static_dependencies.py` | 依赖 01-3 和 01-6。 | 大小为 M。 | 登记 D01 到 D05 的位置。 |

- 本 RFC 的 PR 不承担共存期必改清单的 C 行，只承担自己的分发点 D01 到 D05。每个 PR 只改变 `unified` 下的行为，`legacy` 下的行为和已有测试不变。
- 01-2、01-3、01-6 的分发点在 01-7 建立登记表之前合入，01-7 合入时为它们新增位置文件，T6 从这时起生效。
- 02、03、04 的 PR 依赖 01-1 到 01-7 全部合入。

## 测试计划

测试放在 `tests/diffusion/weight_stack/`。除特别写明的以外，都是 L1 测试，标记 `core_model`、`diffusion` 和 `cpu`，不需要加速卡。

**总开关**（`test_switch.py`）

- `parse_weight_stack` 接受 `legacy`、`unified` 和 `None`，其余取值报 `ConfigError`。
- 命令行、Python 参数和 YAML 三种来源都能把取值传到 `OmniDiffusionConfig`，默认值是 `legacy`。
- 同一个进程里两个 worker 持有不同取值时报 `ConfigError`；持有相同取值时可以共存；最后一个 worker 关闭后状态回到未设置。
- `unified` 下 1.3 列出的每一项设置各自触发 `ConfigError`，报错写明这一项。

**接口、错误和资源账本**（`test_interfaces.py`、`test_errors.py`、`test_resources.py`）

- 在子进程里以 `import vllm_omni` 之后已加载的模块为基线，import `interfaces.py` 和 `errors.py` 不加载 `TYPE_CHECKING` 里列出的任何模块。
- `fakes.py` 里的每个替身都实现了对应 Protocol 的全部方法，方法名和参数与 Protocol 一致。
- 错误信息符合第 3 节的格式；`ErrorSummary` 可以 pickle，超长的信息被截断；非 `WeightStackError` 的异常按步骤包装，`__cause__` 保留。
- `ResourceLedger`：关闭顺序按种类、同种类内逆序；重复登记、`transfer()` 之后登记、关闭之后登记都报 `LifecycleError`；关闭可以多次调用；一个关闭函数报错时其余照常执行，最后报 `LifecycleError`。

**启动驱动**（`test_startup.py`、`test_startup_faults.py`）

- 沿“不做 offload”这条路径，用替身走完第 1 到第 12 步，状态依次经过 `STRUCTURE_READY` 到 `COMMITTED`，`mark_in_service()` 之后是 `IN_SERVICE`，`close()` 之后是 `CLOSED`。
- 工厂函数不存在时报 `StackUnavailableError`，信息列出缺少的模块并提示 `legacy`。
- `FaultPlan.from_environ()`：变量不存在时返回 `None`；`<注入点>[@<rank>][:raise|:hang]` 的各种写法都能解析；格式不对时报 `ConfigError`。
- 第 10 节里 `compile_setup` 和 `close` 以外的每个故障注入点各一个用例，断言：
  - 抛出的类型是第 3 节的表里这一步对应的类型；
  - 所有权链上的持有者被关闭恰好一次，`loader.close()` 被调用，模型被丢弃；
  - 失败发生在第 9 步之前时，替身记录到的写入次数为 0；
  - `Election` 在 `allocate()` 之前的失败里由驱动关闭，在之后的失败里随 host 存储关闭，不会被关闭两次；
  - 状态最终是 `FAILED`。
- 驱动在任何一步都不进入 `memory_pools` 的两个池；替身 Loader 只在分配常驻张量时进入 `weights()`，替身 Offloader 只在 `prepare_materialization()` 里进入 `slots()` 和 `weights()`。
- `compile_setup` 注入点上的 `raise`：启动继续，记一次警告，状态是 `COMMITTED`。
- `close` 注入点上的 `raise`：`handle.close()` 照常调用 `runtime.close()`，状态是 `CLOSED`，然后报 `LifecycleError`。
- `handle.close()` 和 `release()` 期间把 `torch.distributed` 的集合通信函数替换成报错的函数，关闭照常完成。
- 驱动返回之后，模型和它的每个子模块的 `__dict__` 里都没有驱动、启动事务、`Topology`、`SharePlan` 或指纹的引用（总体设计 3.2 第 5 条）。
- 驱动返回的句柄不引用 `Topology`、`SharePlan` 和指纹。

**全组确认**（`test_group.py`、`test_group_multiproc.py`）

- world size 为 1：各确认点不调用 `torch.distributed`；使用 `allgather` 的组件在第 1 点报 `GroupConfirmationError`。
- 部署标识经替身 HWM 核对：所有 rank 采用 0 号 rank 交出的那一对值；某个 rank 交出的第二项为真、而标识与 0 号 rank 的不同时，所有 rank 报 `GroupConfirmationError`。
- 多进程测试用 `torch.multiprocessing` 启动 2 个和 4 个进程，初始化 gloo，不需要加速卡，标记 `parallel`：
  - 全部成功时各 rank 得到相同的 `Topology`、`AllGatherGroups` 和 `Elections`；
  - 每个确认点之前的每个本地步骤上让一个 rank 失败：出错的 rank 抛出自己的错误，其余 rank 抛出 `PeerFailureError`，都在超时之内返回；
  - 在本地步骤 `confirm` 上让一个 rank 失败：每个 rank 都完成了建组的 `new_group` 调用，然后在第 1 点第二轮一起报错，没有 rank 停在 `new_group` 上；
  - 让一个 rank 在确认点上 `hang`，用 5 秒的超时：其余 rank 报 `ConfirmationTimeoutError`，0 号 rank 的信息写明没有到达的 rank；
  - TP 为 2 时同一个组件得到两个 AllGather 组，各 rank 的创建顺序相同；声明了这个组件和没有声明的 rank 都调用了 `new_group`；
  - 同一个进程里第二次启动复用成员相同的组，不再调用 `new_group`；
  - 一个 rank 的进程级表里没有某个组、其余 rank 有时，所有 rank 都为这个组调用 `new_group`，各 rank 的调用次数和顺序相同；
  - `collective` 摘要在组内不一致时所有 rank 报 `GroupConfirmationError`。

**共存**（`test_offload_query_coexistence.py`、`test_execution_facts.py`）

- 对 legacy 下各种 offload 设置的配置（每个旧开关、`diffusion_offload_config` 的各种取值）各构造一份 `unified` 配置，绕过 1.3 的检查后断言：9.1 表里每个查询函数的结果等于 legacy 下不含 offload 设置的配置的结果；`materialize_legacy_offload_flags` 不改任何字段。
- `legacy` 下，`execution_offload_facts` 的三个字段与 9.2 表里的原表达式在各种配置下逐项相等。
- `unified` 下三个字段都为假。

**worker 和 runner 的分发**（`test_worker_dispatch.py`）

- 用替身工厂替换 `EntryPointFactory`，构造 `DiffusionModelRunner`，在 CPU 设备上走 `unified` 一支：`self.pipeline` 是替身的模型，`offload_backend` 为 `None`，`_install_runtime_services` 被调用。
- `legacy` 一支的调用序列与现有实现相同：`DiffusersPipelineLoader.load_model`、`enable_offload_backend`、编译设置依次被调用。
- `DiffusionWorker.shutdown` 在 `unified` 下调用 `close_weight_stack()` 恰好一次，在 `destroy_distributed_env()` 之前调用 `destroy_allgather_groups()`，清除进程级状态。
- `re_init_pipeline` 在 `unified` 下关闭旧的运行时之后才启动新的；首次调用时 `close_weight_stack()` 什么都不做。
- 睡眠和唤醒用替身分配器：`sleep` 在 `allocator.sleep` 之前调用 `before_sleep()`；`wake_up(["weights"])` 和 `wake_up(None)` 都唤醒 `weight_slots`；`wake_up(["kv_cache"])` 不唤醒它；`after_wake()` 在最后调用。
**睡眠池**（`test_worker_sleep_pools.py`，L2，标记 `core_model`、`diffusion` 和 `hardware_test(res={"cuda": "L4"}, num_cards=1)`）

- 启用睡眠模式，替身 Offloader 在 `slots()` 和 `weights()` 里各分配一块显存，替身 Loader 在 `weights()` 里分配常驻张量；level 1 睡眠后 `weights` 池的内容在唤醒后不变，地址不变。
- 单进程执行器关闭后 CuMem 分配器的占用为 0。

**静态依赖**（`test_static_dependencies.py`）

- T1 到 T8 各一个测试。T2 在子进程里运行。
- 每个检查另有一个反例测试：在临时目录里放一个违反规则的文件，断言检查函数报出它。
- 登记表的反例：位置文件的 `ROW` 不在 `ROWS` 里、`NEW_ROW` 的编号重复、`SITES` 里的限定名不存在、读总开关的位置不在任何 `SITES` 里，各自报错。

本 RFC 的 PR 不修改已有测试文件；已有测试在 `legacy` 下全部通过。

## 验收标准

- 总开关为 `legacy` 时，现有行为和已有测试完全不变；本 RFC 的 PR 没有修改已有测试文件。
- `--diffusion-weight-stack` 在命令行、Python 参数和 YAML 里都可用，默认值是 `legacy`，非法取值在参数解析阶段报错。
- `unified` 下有 offload 设置的配置在参数解析阶段报 `ConfigError`，直到 05 取代这项检查。
- `unified` 下真实 pipeline 在每个 rank 上报 `StackUnavailableError`，信息写明缺少的模块并提示改用 `legacy`，不退回 `legacy`。
- 第 10 节的每个故障注入点都有测试；每个失败点上，所有权链的持有者恰好关闭一次，第 9 步之前的失败没有写模型。
- 用测试替身时，多 rank 下任何一个 rank 在任何一步失败，本副本所有 rank 一起报错，在确认点的超时之内返回，不死锁；world size 为 1 时不调用 `torch.distributed`。
- 关闭和失败时的释放不调用 `torch.distributed`；资源账本的每个句柄只关闭一次。
- `unified` 下 `offloader/config.py` 的每个查询函数的结果等于 legacy 下不含 offload 设置时的结果；`legacy` 下 `execution_offload_facts` 与原表达式逐项相同。
- `unified` 下 `re_init_pipeline` 关闭旧的运行时和 host 存储映射之后才开始新的启动流程，所有 rank 都经过第 0 到第 3 点。
- `unified` 下睡眠前调用 `before_sleep()`，唤醒时连同 `weight_slots` 一起唤醒并调用 `after_wake()`。
- 静态依赖测试 T1 到 T8 通过；四个白名单的摘要与冻结时相同；`registry.py` 列出 C01 到 C18 和 D01 到 D05，每一行写明承担的子 RFC；D01 到 D05 各有位置文件，`vllm_omni/` 里每个读总开关的位置都已登记。

## 删除 legacy 时

- 模型里按 offload 策略改变行为的分支按总体设计 9.3 删除，读查询函数的调用随之消失；`offloader/config.py` 随现有的三个后端一起删除。
- `execution_offload_facts` 只保留读 05 策略的一支。
- `OmniDiffusionConfig.__post_init__` 不再调用 `materialize_legacy_offload_flags`。
- `DiffusionModelRunner.load_model` 只保留 `unified` 一支，删除 `offload_backend` 属性和对 `enable_offload_backend`、`DiffusersPipelineLoader` 的调用。`DiffusionWorker.shutdown` 删除对 `offload_backend.shutdown()` 的调用；`re_init_pipeline`、`sleep`、`wake_up` 只保留 `unified` 一支。加载不再经 `_maybe_get_memory_pool_context` 进入 `weights` 池，由 8.2 的睡眠池取代；`kv_cache` 池照旧经它进入。
- 登记表的每一行按 `legacy_removal` 删除 legacy 一支之后，从 `ROWS` 里删除这一行和它的位置文件；`NEW_ROW` 声明的行随它的位置文件一起删除。全部行删除之后，删除 `tests/diffusion/weight_stack/coexistence/`、`startup.py` 的进程级状态和 `is_unified()`，以及 `weight_stack.components.is_unified_stack()`。
- 四个冻结白名单清空，T8 的摘要删除。T6 随总开关的读取一起删除，T7 随 `offloader/config.py` 一起删除。T1 到 T5 的规则保留：模型代码不 import Offloader，不按名字读钩子，不读 offload 和 host 的配置字段；T4 只从 04 的 `HOOK_NAMES` 读钩子名。
- `--diffusion-weight-stack legacy` 在参数解析阶段报 `ConfigError`，说明 `legacy` 已删除；`unified` 仍被接受，不起作用。

## 风险

| 风险 | 影响 | 处理 |
|---|---|---|
| 第 2 点和第 3 点之间的写入和收尾里有集合通信，某个 rank 在这些集合通信之前失败 | 失败的 rank 跳过后面的本地步骤，不参加这些集合通信；其余 rank 停在集合通信上，直到集合通信自身的超时报错。失败的 rank 在第 3 点最多等 `dist_timeout`（没有给出时 1800 秒）后报 `ConfirmationTimeoutError`。失败要等一个超时才报出，等待有上限，不会无限期挂起。 | 第 1 点的建组在任何本地步骤之前完成，不受这项影响（6.2）。失败的 rank 立即释放资源；它在第 3 点抛出之后 worker 初始化失败，执行器随之终止本副本的其余 worker，停在集合通信上的 rank 也随之结束（总体设计 4.4、6.5）；报错写明失败的 rank 和步骤；多进程测试用带集合通信的替身 Loader 记录这种情况下的返回时间。 |
| 单进程执行器把几个 stage 的 worker 放进同一个进程，而这些 stage 的总开关取值不同 | 第二个 worker 初始化时报 `ConfigError`，部署启动失败。 | 报错写明两个 stage 的编号和取值；一个部署的各扩散 stage 默认取值相同。 |
| 新的非模型代码在共存期调用查询函数来确定集合通信的顺序 | `unified` 下读到“不做 offload”，DP 下 `allgather` 需要的同步执行被关掉，各 rank 的块执行顺序可能不一致。 | T7 把非模型调用方冻结在白名单里，新的调用方必须读 `execution_offload_facts`。 |
| 外部代码按 tag 唤醒 CuMem 分配器时只传 `weights` | `weight_slots` 池不被唤醒，块环缓冲区和分片槽位没有重新映射。 | `DiffusionWorker.wake_up` 在 `tags` 包含 `weights` 时一并唤醒 `weight_slots`；不经过 `wake_up` 直接操作分配器的代码不受支持。 |

## 反馈期

发布后一周。

## CC

启动流程、分布式运行时、扩散 worker 与模型运行器、Loader、Offloader、HWM、配置的负责人。
