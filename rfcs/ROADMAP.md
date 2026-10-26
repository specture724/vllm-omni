# 路线图：统一 Loader / Offloader / Host Weight Manager

> 上级 RFC：[umbrella RFC](./00-umbrella.md)。设计依据：[总体设计](../LOADER_OFFLOADER_DESIGN.md)。
>
> 本文按五份子 RFC“实现拆分”里的 PR 编号排出合入顺序、里程碑、并行的工作线和每个里程碑的完成条件。PR 的内容、依赖、测试和大小以子 RFC 为准，本文只做汇总和排序。

## 子 RFC

| 子 RFC | umbrella 的 Job | 范围 | PR | 承担的必改行和分发点 |
|---|---|---|---|---|
| [01 启动事务、总开关与共存期](./01-startup-transaction.md) | Job 1 | `weight_stack/` 的接口、错误类型、启动驱动和全组确认；总开关；查询函数的共存；worker 和 runner 的分发；静态依赖测试、冻结白名单和必改清单的登记表 | 01-1 到 01-7 | D01 到 D05 |
| [02 统一加载](./02-unified-loader.md) | Job 2（2a 到 2f） | 组件声明、只建结构的构造上下文、规则表、读取器和延迟张量、先分配后写入、收尾、内容身份、适配器融合、动态 LoRA 包装、`load_format="diffusers"`、19 个构造期读张量的家族迁移、查询接口 | 02-1 到 02-18 | C02、C04、C05、C09、C10、C11、C13、C15、C16、C18；D06 |
| [03 Host 存储](./03-host-storage.md) | Job 3 | HWM 的本机后端和分片后端、部署标识、选举和封存、容量准入、fork 处理 | 03-1 到 03-9 | 无 |
| [04 统一 Offloader](./04-unified-offloader.md) | Job 4 | 重复块发现、Unit 规划、两种 Trigger、两种 Transport、块环、内存登记和暂存区、守卫、编译设置、缓存签名、睡眠模式 | 04-1 到 04-13 | C01、C03、C06、C07、C08、C12、C14、C17、C19 |
| [05 配置与可观测性](./05-configuration.md) | Job 5 | 策略解析和校验、旧开关的兼容层、host 策略、`--diffusion-max-lora-rank`、启动日志、指标、recipe、文档、支持矩阵、门槛矩阵 | 05-1 到 05-9 | D20、D21、D22 |

umbrella 的 Job 6（达标、翻转、删除）没有单独的子 RFC：门槛矩阵由 05-9 提供，删除的内容写在每份子 RFC 的“删除 legacy 时”一节，顺序见本文的 M7。

## PR 的编号和依赖

- PR 编号写成 `<子 RFC>-<序号>`，例如 02-5。大小 S、M、L 取自子 RFC 的表，只用来比较工作量和求关键路径。
- 02、03、04、05 的 PR 都依赖 01-1 到 01-7 全部合入，下文不再重复。
- 一个 PR 在它依赖的 PR 全部合入之后就可以合入，不等所在里程碑的其他 PR。里程碑是验收的节点，不是开工的门槛。
- 共 56 个 PR。把五份子 RFC“实现拆分”表里的依赖列合在一起，依赖图没有环，每个被引用的 PR 编号都存在。

跨子 RFC 的依赖边（左边的 PR 依赖右边的 PR）：

| 方向 | 边 |
|---|---|
| 03 依赖 02、04 | 03-9 ← 02-5、04-5 |
| 04 依赖 02 | 04-1 ← 02-1；04-2 ← 02-2；04-8 ← 02-10、02-14、02-15、02-16；04-9 ← 02-10；04-10 ← 02-12；04-11 ← 02-11；04-12 ← 02-18；04-13 ← 02-5 |
| 04 依赖 03 | 04-3 ← 03-3；04-13 ← 03-5、03-6 |
| 05 依赖 02 | 05-4 ← 02-8；05-5 ← 02-2；05-6 ← 02-5；05-7 ← 02-11、02-12、02-14、02-18；05-8 ← 02-2 |
| 05 依赖 03 | 05-3 ← 03-5、03-6、03-7；05-6 ← 03-8；05-9 ← 03-9 |
| 05 依赖 04 | 05-3 ← 04-3、04-5、04-6、04-7；05-5 ← 04-2；05-6 ← 04-5；05-9 ← 04-13 |

02 不依赖 03、04、05 的 PR；03 除 03-9 外只依赖 01；02、03、04 都不依赖 05。

## 里程碑

```text
M1 启动事务与共存期骨架
 └─▶ M2 不做 offload 的 unified 启动
      ├─▶ M3 offload 机制
      │    └─▶ M4 配置与可观测性
      └─────────▶ M5 迁移家族与使用方（另依赖 M3、M4 的部分 PR）
                   └─▶ M6 门槛与翻转
                        └─▶ M7 删除 legacy
```

可以提前开始的 PR：

- 04-4、05-1、05-2 只依赖 M1，M1 完成后就可以开始。
- 03-6 只依赖 03-3，与 M2 的其余部分并行。
- M5 里 Loader 一侧的 PR（02-7 到 02-18）只依赖 M2 的 02 PR，与 M3、M4 并行。

### M1：启动事务与共存期骨架

| PR | 内容 | 依赖 | 大小 | 必改行 |
|---|---|---|---|---|
| 01-1 | `weight_stack` 包骨架、数据类型、Protocol、错误类型、`ResourceLedger` | 无 | M | 无 |
| 01-2 | 总开关和进程级状态、传递链、参数解析检查 | 01-1 | S | D01 |
| 01-3 | 查询函数的共存、`execution_offload_facts` | 01-2 | S | D02 |
| 01-4 | 全组确认、`Topology`、AllGather 组 | 01-1 | M | 无 |
| 01-5 | 启动驱动、状态机、所有权链、`FaultPlan`、测试替身 | 01-1、01-4 | L | 无 |
| 01-6 | worker 和 runner 的分发、睡眠池、关闭和重新初始化 | 01-2、01-5 | M | D03、D04、D05 |
| 01-7 | 静态依赖测试 T1 到 T8、登记表、冻结白名单、评审清单 | 01-3、01-6 | M | D01 到 D05 的位置文件 |

完成条件：

- `legacy` 下现有行为和已有测试不变。
- `--diffusion-weight-stack unified` 在命令行、Python 参数和 YAML 里都可以选中；`unified` 下有 offload 设置时在参数解析阶段报 `ConfigError`。
- `unified` 下真实 pipeline 在每个 rank 的第 1 步报 `StackUnavailableError`，在第 0 点一起失败，报错写明缺少的模块，不退回 `legacy`。
- 测试替身沿“不做 offload”的路径覆盖启动驱动、确认点、资源账本、释放顺序、关闭、睡眠和唤醒、重新初始化；01 第 10 节的每个故障注入点都有测试。
- `unified` 下查询函数报告“不做 offload”，引擎和执行器改读 `execution_offload_facts`。
- T1 到 T8 通过；四个白名单冻结；`registry.py` 列出 C01 到 C19 和 D01 到 D05；评审清单写进 `docs/contributing/`。从这里开始，每个 `unified` 分支都受静态测试和登记表约束。

### M2：不做 offload 的 unified 启动

| PR | 内容 | 依赖 | 大小 | 必改行 |
|---|---|---|---|---|
| 02-1 | 组件声明、组件目录、`pipeline_of` | 无 | M | 无 |
| 02-2 | 构造上下文、读取陷阱、最终类型、`build_component`、`declare_weights`、`hf_component`、支持矩阵测试 | 02-1 | L | D06 |
| 02-3 | 规则表的格式和执行、`generate` 工具、主干用到的第一批表 | 02-2 | L | 无 |
| 02-4 | 来源解析、读取流水线、延迟张量、内容身份、预期清单 | 02-2 | L | 无 |
| 02-5 | 存储绑定、写入跟踪、逐层在线量化、收尾、最终清单、HSDP 组件 | 02-3、02-4 | L | 无 |
| 02-6 | 以 legacy 为基准的逐张量对比框架、`export_legacy_hashes`、指纹清单 | 02-5 | M | 无 |
| 03-1 | HWM 包骨架、本机键、`create_host_weight_manager`、`local_record()` | 无 | M | 无 |
| 03-2 | 部署标识的生成和下发 | 03-1 | S | 无 |
| 03-3 | `HostRequest`、两种后端的布局、`flatten_shard_unit`、规划指纹 | 03-1 | M | 无 |
| 03-4 | 共享指纹、选举、协议消息、监听线程 | 03-3 | L | 无 |
| 03-5 | 本机后端和私有内存、`view()`、`seal()`、只读重绑定 | 03-4 | L | 无 |
| 03-7 | 容量准入 `admit()` | 03-4 | S | 无 |
| 04-1 | Offloader 包骨架、`policy` 为 `None` 的规划、`prepare_materialization()` 的准入、`commit()` 的校验和守卫、没有 Unit 的运行时 | 02-1 | M | 无 |

完成条件：

- 启动驱动用真实的 Loader、HWM、Offloader 走完全部步骤和确认点；`policy` 为 `None`，所有组件常驻加速卡。
- 支持矩阵测试列出每个注册的 pipeline 在只建结构的上下文里的构造结果。可以构造、并且不涉及未完成的必改行的 pipeline（例如 Z-Image）在 `unified` 下完成启动，逐张量对比与 `legacy` 一致；构造期读张量的 pipeline 报 `ConstructionError`，写明读取位置并提示 `--diffusion-weight-stack legacy`。
- 每个最终张量只写一次；收尾按固定顺序，不调用模型的加载后钩子；在线量化时 Host 和加速卡的峰值不高于 `legacy` 的逐层量化。
- 常驻加速卡的张量都在 `memory_pools.weights()` 里分配。
- 逐张量对比框架和指纹测试就位，用已按新写法构造的 pipeline 自测。
- 本机后端的选举、封存、只读重绑定和容量准入有多进程测试。

### M3：offload 机制

| PR | 内容 | 依赖 | 大小 | 必改行 |
|---|---|---|---|---|
| 03-6 | 分片后端、瘦路径的视图、`note_written()` 和同步页归还、先全量后裁剪 | 03-3 | M | 无 |
| 03-8 | fork 处理、故障注入点、经 01 驱动的多进程生命周期测试 | 03-5、03-6、03-7 | M | 无 |
| 04-2 | 重复块发现、`repeated_blocks` 覆盖、每个注册 pipeline 的发现期望表 | 02-2、04-1 | M | 无 |
| 04-3 | 选择、Unit 划分、常驻层、让出关系、Transport、AllGather 组推算、冷启动路径、`host_request()` | 03-3、04-2 | L | 无 |
| 04-4 | 平台的内存登记接口和 CUDA 实现 | 无 | S | 无 |
| 04-5 | 启动事务的完整实现：登记计划、准入、固定槽位、暂存区、`commit()` 的内存登记 | 04-3、04-4 | L | 无 |
| 04-6 | 钩子、使用触发与让出、`rank-local` 块环、单次执行、梯度模式、关闭、睡眠；runner 和 worker 扩展方法经过 `execution()` | 04-5 | L | C17、C19 |
| 04-7 | `allgather`：分片槽位、拼接流、跨组件的预取限制 | 04-6 | M | 无 |
| 03-9 | Host 存储与真实 Loader、Offloader 的 GPU 端到端测试 | 02-5、03-8、04-5 | M | 无 |

完成条件（策略经 `UnifiedOffloader` 的 `select` 参数注入；M4 之前命令行仍拒绝 `unified` 下的 offload 配置）：

- 每个注册的 pipeline 在测试覆盖的拓扑和量化配置下，重复块发现的结果与期望表一致。
- 各种策略产生互不重叠的 Unit；`rank-local` 和 `allgather` 的输出与完整加载一致；块环的内存有固定上限，正确处理跳块和异常。
- 提交后对被管理的模块改设备或改类型报错；钩子在 graph 捕获期间需要动作时报错。
- 关闭不恢复 CPU 权重、不发起集合通信；睡眠再唤醒后输出一致，常驻部分和固定槽位的地址不变。
- 同一个部署在一台机器上的 worker 数从 1 增加到 k，共享内存的增量不随 k 增长；分片后端稳定后每个 rank 占 `W / N`；瘦路径冷启动的额外占用严格小于“释放阈值 + 最大的两个张量”。
- 对一个 worker 执行 `kill -9` 后，它所在副本的全部 worker 退出，其他副本不受影响；fork 出的子进程不持有内存文件和监听套接字。
- C17、C19 有位置文件和 `unified` 用例。

### M4：配置与可观测性

| PR | 内容 | 依赖 | 大小 | 必改行 |
|---|---|---|---|---|
| 05-1 | 策略的数据类型、JSON schema、预设、`resolve_selection`、旧配置的解析和旧开关的翻译 | 无 | M | 无 |
| 05-2 | 新字段和参数的传递链、来源优先级；`unified` 下新字段在 05-3 合入之前的拒绝 | 05-1 | M | 无 |
| 05-3 | `parse_unified_config`、分层校验、环境变量折叠、`startup_policy`、执行事实；`unified` 开始接受 offload 配置 | 03-5、03-6、03-7、04-3、04-5、04-6、04-7、05-2 | L | D20、D22 |
| 05-5 | 翻译测试和豁免、`LEGACY_RESIDENT_DIT_MODELS` 的派生测试 | 02-2、04-2、05-3 | M | 无 |
| 05-6 | `observability.py`、启动日志事件、`weight_stack_memory_report`、指标 | 02-5、03-8、04-5、05-3 | L | D21 |

完成条件：

- `unified` 下 offload 策略可以经命令行、Python 参数和 YAML 给出，三者给出同一份策略时结果相等；高优先级来源整体覆盖低优先级来源。
- 旧开关和旧的 `--diffusion-offload-config` 由兼容层翻译；翻译测试对每个能在只建结构的上下文里构造的 pipeline 通过。
- P01 到 P33（P30 除外）在参数解析阶段报 `ConfigError`；L01 到 L05 在规划阶段报 `PlanningError`，L01 按全部 rank 判断。
- 启动日志对每个组件写明 Unit 划分、后端、共享与否和原因、冷启动路径、`fully_resident`；05 第 16.1 节的指标在 `--log-stats` 打开时导出。

### M5：迁移家族与使用方

| PR | 内容 | 依赖 | 大小 | 必改行 |
|---|---|---|---|---|
| 02-7 | 适配器格式转换器、读取器里的融合、`fuse` 固定适配器、worker 的 distill 分支 | 02-5 | L | C16（融合） |
| 02-8 | 动态 LoRA 包装、槽位声明、`overlay` 固定适配器、LoRA 管理器的 `unified` 一支 | 02-5 | M | C16（包装） |
| 02-9 | `load_format="diffusers"` 的 `unified` 一支 | 02-3、02-5 | M | C18 |
| 02-10 | 查询接口和使用方迁移、Wan VAE 的内存格式规则表 | 02-2、02-3 | M | C13 |
| 02-11 | MiniMax H3 | 02-6、02-7 | L | C04、C05 |
| 02-12 | MAGI-2、Lance | 02-6 | L | C02 |
| 02-13 | LTX2 | 02-6、02-8 | M | C15 |
| 02-14 | Wan2.2-S2V、DreamZero、LongCat-Video-Avatar | 02-6、02-8 | M | 无 |
| 02-15 | Ming-Flash-Omni、Ming-Image、Anima、NextStep-1.1、Cosmos3 声音 tokenizer、SeedVR2、AuK、OmniVoice | 02-6 | L | 无 |
| 02-16 | π0、π0.5、InternVLA-A1、Gr00t | 02-6 | M | 无 |
| 02-17 | SANA-WM、HunyuanImage3、Kandinsky6 的必改行 | 02-3 | S | C09、C10、C11 |
| 02-18 | 直接调用 `X.from_pretrained` 的 pipeline 改用 `hf_component`，以及所需的规则表 | 02-3、02-6 | M | 无 |
| 04-8 | 编译设置、驻留表、`regionally_compile` 的 `unified` 一支 | 02-10、02-14、02-15、02-16、04-6 | M | C12 |
| 04-9 | 以权重为键的缓存签名、Cosmos3 NVFP4 查找表 | 02-10、04-5 | S | C07、C14 |
| 04-10 | MAGI-2 的模型侧改动 | 02-12、04-6 | S | C01 |
| 04-11 | MiniMax H3 的模型侧改动 | 02-11、04-6 | S | C03、C06 |
| 04-12 | LingBot-Video 的模型侧改动 | 02-18、04-6 | S | C08 |
| 04-13 | 端到端和 GPU 测试：与完整加载的输出对比、全部组件 offload 的冒烟、睡眠唤醒、故障注入 | 02-5、03-5、03-6、04-7、04-8、04-9 | M | 无 |
| 05-4 | `--diffusion-max-lora-rank` 的检查（P30） | 02-8、05-3 | S | 无 |
| 05-7 | recipe、文档和迁移说明 | 02-11、02-12、02-14、02-18、05-4、05-5、05-6 | M | 无 |
| 05-8 | 支持矩阵的渲染工具、文档和一致性测试 | 02-2、05-7 | S | 无 |

完成条件：

- 19 个构造期读张量的家族都已迁移，每个家族的逐张量对比与 `legacy` 一致，对比例外表里的每一条写明原因；构造函数里直接调用 `X.from_pretrained` 的 pipeline 都改用 `hf_component`。
- 支持矩阵的待定列表为空，矩阵里没有 `unified` 不支持的 pipeline。
- 融合适配器后再在线量化的结果，在 TP 为 1 和大于 1 时都与“先融合、后量化”的参考一致；加载、切换、卸载动态 LoRA 前后模块结构和基础权重的地址不变。
- 02 第 15 节的使用方在 `unified` 下都从组件目录读取；不是 `fully_resident` 的组件不进 CUDA graph，cudagraph 编译模式降为 `default`。
- 每个支持的 pipeline 都有“全部组件做 offload”的冒烟测试。
- 登记表的 `ROWS` 里每一行都至少有一个位置文件。
- recipe、迁移说明和支持矩阵文档与当前结果一致。

### M6：门槛与翻转

| PR | 内容 | 依赖 | 大小 | 必改行 |
|---|---|---|---|---|
| 05-9 | 门槛矩阵：组合清单、对比基准、Host 和加速卡内存峰值的采集、原始数据的保存 | 03-9、04-13、05-7 | M | 无 |

步骤：

1. 05-9 合入，门槛矩阵每晚运行，每个组合保存输出对比和两项峰值的原始数据。
2. 翻转的条件（umbrella“上线方式”第 4 步）全部满足：
   - 门槛矩阵里门槛例外以外的组合在 `unified` 下输出与 `legacy` 一致，Host 和加速卡的峰值不高于 `legacy`；
   - 主干里注册的每个 pipeline 都能在只建结构的上下文里构造；
   - 每个迁移家族的逐张量对比通过，对比例外表里的每一条都已列入门槛例外；
   - 指纹清单没有未处理的变化；
   - 登记表的每一行都至少有一个位置文件。
3. 翻转默认值：`--diffusion-weight-stack` 的默认值改为 `unified`。
4. 翻转后的一个版本保留 `legacy` 作为回退，共存期的规则照旧执行。主干外没有迁移的 pipeline（`register_diffusion_model` 注册的、`load_format="custom_pipeline"` 的）在这个版本里显式传 `--diffusion-weight-stack legacy`，迁移说明写明这一点。

### M7：删除 legacy

在翻转后的下一个版本执行。

删除之前：

- 用 02-6 的 `export_legacy_hashes` 把逐张量对比的 legacy 结果存成逐张量的哈希，对比测试改为与哈希对比。
- 每个注册的 pipeline 都显式调用 `declare_components`。
- 只靠 `_layerwise_offload_blocks_attrs` 回退被编译的九个块类补进 `_repeated_blocks`；`test_compile_targets.py` 断言删除前后被编译的模块集合相同；每个注册的 pipeline 的 `layer` 冒烟测试通过。
- MAGI-2 的内存窗口由 recipe 里的配置给出：`dit` 选 `module`、`rank-local`。

删除按下面的顺序，每一步按对应子 RFC 的“删除 legacy 时”一节执行：

1. 分发点（01）：`DiffusionModelRunner.load_model`、`DiffusionWorker` 的 `re_init_pipeline`、`sleep`、`wake_up` 只保留 `unified` 一支，`__post_init__` 不再调用 `materialize_legacy_offload_flags`，`execution_offload_facts` 只保留读 05 策略的一支；`--diffusion-weight-stack legacy` 在参数解析阶段报 `ConfigError`。这一步之后，`legacy` 的代码都不可达。
2. 模型侧和加载路径（02、04）：04 删除三个现有后端、`offloader/` 包（包括 `config.py`）、模型里的 offload 专用声明和生命周期代码、按 offload 策略改变行为的分支、SeaCache 按名字读钩子的代码；02 删除每个 `build_component` 的 legacy 一支和 `build_component` 本身、HF 组件入口的 legacy 一支、现有的非 HSDP 加载路径和其中对 Host Weight Runtime 的接入、只服务于 legacy 的 `load_weights` 主体、指纹清单和指纹测试。
3. 配置（05）：`policy.py` 里的兼容层和它的规则、翻译测试、等价测试；旧字段、它们的命令行参数和校验函数；D20、D21、D22 只保留 `unified` 一支。
4. 共存期设施（01）：登记表的每一行按 `legacy_removal` 删除 legacy 一支之后，删除这一行和它的位置文件；全部行删除之后，删除 `tests/diffusion/weight_stack/coexistence/`、进程级状态、`is_unified()` 和 `weight_stack.components.is_unified_stack()`；四个冻结白名单清空，T6、T7 删除。

每一步的 PR 里，删除一个登记行的 legacy 一支和删除这一行在同一个 PR 里完成，静态依赖测试在每一步之后都通过。

## 工作线

M1 之后可以同时推进的工作线：

| 工作线 | PR 顺序 | 汇合点 |
|---|---|---|
| Loader 主干（02） | 02-1 → 02-2 → 02-3、02-4 → 02-5 → 02-6 | 04-1 等 02-1，04-2 等 02-2；03-9、04-13、05-6 等 02-5 |
| 家族迁移（02） | 02-6 之后 02-11 到 02-16、02-18 并行；02-7、02-8、02-9 在 02-5 之后；02-10、02-17 在 02-3 之后 | 04-8 等 02-10、02-14、02-15、02-16；04-10、04-11、04-12 分别等 02-12、02-11、02-18；05-7 等 02-11、02-12、02-14、02-18 |
| Host 存储（03） | 03-1 → 03-2、03-3 → 03-4 → 03-5、03-7；03-3 → 03-6；03-5、03-6、03-7 → 03-8 → 03-9 | 04-3 等 03-3；05-3 等 03-5、03-6、03-7；04-13 等 03-5、03-6 |
| Offloader（04） | 04-4 随时；04-1 → 04-2 → 04-3 → 04-5 → 04-6 → 04-7；04-6 之后 04-8 到 04-12；04-13 收尾 | 05-3 等 04-3、04-5、04-6、04-7 |
| 配置（05） | 05-1 → 05-2；05-3 → 05-4、05-5、05-6 → 05-7 → 05-8、05-9 | 05-9 等 03-9、04-13 |

## 关键路径

按大小记 S = 1、M = 2、L = 3，最长的依赖链是：

```text
01-1 → 01-4 → 01-5 → 01-6 → 01-7 → 02-1 → 02-2 → 04-2 → 04-3 → 04-5 → 04-6 → 04-7 → 05-3 → 05-6 → 05-7 → 05-9
```

- 01-1、01-4、01-5 和 02-1、02-2 是最早的瓶颈：02、04 的所有 PR 都在它们之后。
- M3 的 04-2 到 04-7 决定 05-3 能多早合入，05-3 之后的配置、日志和 recipe 都等它。
- 与关键路径只差 1 的 PR：04-1、05-5、05-8；差 2 到 3 的：05-4、03-1、03-3。
- 家族迁移（02-11 到 02-18）和 Host 存储（03）不在关键路径上，可以按评审能力并行推进。

评审先排关键路径上的 PR。

## 必改行与分发点

| 行 | 位置 | PR | 里程碑 |
|---|---|---|---|
| C01 | MAGI-2 pipeline 的两处 offload 校验、分阶段组件、内存窗口 | 04-10 | M5 |
| C02 | MAGI-2 MoE 的 `_owned_w13` 视图；只建结构时注册合并参数 `W13` | 02-12 | M5 |
| C03 | MiniMax H3 的 SM120 AdaLN offload 自动默认、放大器的搬运 | 04-11 | M5 |
| C04 | MiniMax H3 的 AdaLN sidecar | 02-11 | M5 |
| C05 | MiniMax H3 视频 VAE 的设备判断和 `self.to(float32)` | 02-11 | M5 |
| C06 | MiniMax H3 多输出时的 `_offload_model_cpu_stage_output` | 04-11 | M5 |
| C07 | 精确投影缓存、AdaLN 缓存的签名 | 04-9 | M5 |
| C08 | LingBot-Video 的 `offload_vae_during_denoise` | 04-12 | M5 |
| C09 | SANA-WM 每个请求的 `_ensure_*` | 02-17 | M5 |
| C10 | HunyuanImage3 `load_weights` 开头的 `.to(device)` | 02-17 | M5 |
| C11 | Kandinsky6 的 `self.device` | 02-17 | M5 |
| C12 | CUDA graph 和 cudagraph 编译模式按 `fully_resident` 门控 | 04-8 | M5 |
| C13 | Wan VAE 快路径的 `tensor.data=` 恢复 | 02-10 | M5 |
| C14 | Cosmos3 混合精度的 NVFP4 查找表 | 04-9 | M5 |
| C15 | LTX2 vocoder 的 `.float()` 往返 | 02-13 | M5 |
| C16 | 提交后包装 PEFT 层、提交后融合蒸馏 LoRA | 02-7（融合）、02-8（包装） | M5 |
| C17 | runner 的梯度模式和执行作用域 | 04-6 | M3 |
| C18 | diffusers 适配器的 `_raise_unsupported_features` | 02-9 | M5 |
| C19 | worker 扩展类的方法在 `execution()` 之外调用 | 04-6 | M3 |
| D01 | `OmniDiffusionConfig.__post_init__` 的参数解析检查 | 01-2 | M1 |
| D02 | `offloader/config.py` 的 `resolve_offload` 和 `materialize_legacy_offload_flags` | 01-3 | M1 |
| D03 | `DiffusionModelRunner.load_model` 的分支 | 01-6 | M1 |
| D04 | `DiffusionWorker` 的 `__init__`、`load_model`、`shutdown`、`sleep`、`wake_up` | 01-6 | M1 |
| D05 | `CustomPipelineWorkerExtension.re_init_pipeline` | 01-6 | M1 |
| D06 | `models/utils.py` 的 `create_transformers_model`、`create_transformers_model_with_vllm_linears` | 02-2 | M2 |
| D20 | `enrich_config` 末尾的 `check_late` | 05-3 | M4 |
| D21 | stage 进程的内存报告采集 | 05-6 | M4 |
| D22 | `__post_init__` 里的 `check_migration` | 05-3 | M4 |

C01 到 C19、D01 到 D05 在 01-7 的 `registry.py` 里；D06、D20、D21、D22 由承担的 PR 在位置文件里用 `NEW_ROW` 声明。D01 到 D05 的位置文件由 01-7 新增。

## 合入规则

M1 到 M6 期间，每个 PR 都满足：

- 只改变 `unified` 下的行为；`legacy` 下的行为和已有测试不变；不修改已有测试文件（05-2 给 T5 字段表加入新字段除外，01 允许）。
- 承担必改行或分发点的 PR 在描述里写明行编号和这一行的 `legacy_removal`，新增 `tests/diffusion/weight_stack/coexistence/sites/<行编号>_<PR 编号>.py`，把新增的每个读总开关或构造上下文的位置写进 `SITES`；`legacy` 一支是原来的语句，diff 只显示把它们移进分支；不往冻结的白名单里加条目。
- 模型代码读总开关只用 `weight_stack.components.is_unified_stack()`；`models/`、`lora/`、`cache/` 里的 `unified` 一支不 import Offloader，不读 offload 和 host 的配置字段，不按名字读钩子，不调用 `OffloadRuntime` 的方法。
- 共存期间不合入构造期读张量的新 pipeline；新增的 pipeline 按 Z-Image、Kandinsky6 的写法，两个栈共用一份构造代码。
- legacy 一支的修复按 umbrella“共存期 legacy 修复的同步”一节同步到 unified 一支：改变最终张量的修复由逐张量对比测试发现，不改变最终张量的修复由指纹测试发现，两者都在同一个 PR 里改 unified 一支。
- 评审按 `docs/contributing/` 里由 01-7 写入的评审清单逐项核对。
