# Matcher 包目录职责整理清单

本文依据 `文档/记录/目标草案.md` 和 `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`，记录在不改变科学行为的前提下整理 `matcher/` Python 包的边界。它只管理模块归属、导入路径和正式命令入口，不改变候选集合、监督、模型结构、损失、指标、随机化、global batch、配置值、产物字段或 calibration 边界。

当前冻结基点为 `Learn/CUMULATIVE` 提交 `a74deca9b7a3f6a58909410623ba1c8b99eb9625`。该基点在本地重新通过 `118 passed`、mypy 检查 `matcher/` 下 40 个 Python 文件零问题、19 份正式 YAML 解析、8 个 Shell 入口语法检查和运行相关文件等价核验。服务器六个受管任务均运行各自的内容寻址 release；本次移动不会修改这些 release、动态命令或锁。

## 目标目录与依赖方向

```text
matcher/
  data/          # Stage1/A–G 来源适配、标签、几何、清单和候选级缓存
  datasets/      # CandidateDataset、样本/批次契约、采样器、collate 和 DataLoader
  model/         # 候选编码器、融合、RAUNet 和闭集身份模型
  training/      # 配置、损失、初始化、checkpoint、数值后端、追踪和训练编排
  evaluation/    # validation/calibration 指标累计、选模分数和一次性评估入口
  utils/         # 不决定科学语义的字节缓存、严格 JSON 和文件锁
  tests/         # 按行为域保留现有测试文件，只更新正式导入路径
```

允许的依赖方向为 `utils → data → datasets → model/evaluation → training`，其中箭头表示右侧可以读取左侧。`evaluation.calibration` 可以调用 `training.engine.evaluate_model`；`training.engine` 只读取 `evaluation.metrics`，不得从 `evaluation.__init__` 反向导出 calibration，从而避免循环导入。`matcher.model` 保留现有公开导出。新增子包的 `__init__.py` 只说明职责，不建立大规模转发层。

整理前基点的四个正式命令入口全部由仓库内 Shell 调用，未发现仓库外稳定调用证据：

| 整理前命令 | 整理前调用者 | 整理后命令 |
|---|---|---|
| `python -m matcher.manifest` | `训练与运行/sh/build_manifest.sh` | `python -m matcher.data.manifest` |
| `python -m matcher.precompute` | `训练与运行/sh/precompute_data.sh` | `python -m matcher.data.precompute` |
| `python -m matcher.experiment` | `训练与运行/sh/train_experiment.sh` | `python -m matcher.training.experiment` |
| `python -m matcher.calibration` | `训练与运行/sh/evaluate_calibration.sh` | `python -m matcher.evaluation.calibration` |

运行中的任务使用冻结 release，不读取本地可变项目根。因此不保留旧顶层模块兼容包装；包装会继续暴露本次要消除的平铺布局，并增加双重入口和循环导入风险。

## 正式模块清单

| 整理前文件 | 基点职责 | 具体调用者 | 生命周期与既有测试 | 整理后位置与动作 | 理由 |
|---|---|---|---|---|---|
| `matcher/__init__.py` | 暴露类别顺序与 manifest schema 版本 | 包使用者 | 正式包入口；导入与配置测试间接覆盖 | 原位保留，改从 `data.schema` 导入 | 包级稳定导出仍有价值 |
| `matcher/schema.py` | 定义类别、语言、裁剪策略、版本号和 `SourceRoots` | 数据、模型、训练、评估与测试 | 正式数据契约；多组测试覆盖 | 移到 `matcher/data/schema.py` | 字段和路径均属于数据来源契约 |
| `matcher/labels.py` | 由 occurrence 覆盖率生成六分类与身份监督 | `manifest.py`、数据测试 | 正式科学适配；`test_data_pipeline.py` | 移到 `matcher/data/labels.py` | 与清单生成共享监督定义 |
| `matcher/geometry.py` | 解析候选 48³ 起点并统一坐标与采样统计 | Dataset、体素/PP 缓存读写模块、预计算、数据测试 | 正式科学变换；`test_data_pipeline.py` | 移到 `matcher/data/geometry.py` | 多个来源适配器必须共享同一几何公式 |
| `matcher/density.py` | 由实验/模拟密度构造 56 通道 | `voxel_input.py`、数据测试 | 正式来源适配；`test_data_pipeline.py` | 移到 `matcher/data/density.py` | 决定 Stage1 密度输入语义，不是通用工具 |
| `matcher/ligand_graph.py` | 读取 LigandObject 并构造配体图 | Dataset 契约、Dataset、数据测试 | 正式来源适配；数据与模型测试间接覆盖 | 移到 `matcher/data/ligand_graph.py` | 把 A–G 文件适配成模型可用图 |
| `matcher/manifest.py` | 构建、要求和读取版本化监督清单 | Dataset、预计算、训练、calibration、Shell、测试 | 正式数据入口；`test_data_pipeline.py` | 移到 `matcher/data/manifest.py` | 清单属于数据准备，不属于训练编排 |
| `matcher/voxel_input.py` | 构造并读取候选级 105×48³ 体素缓存 | Dataset、预计算、数据测试 | 正式候选输入缓存读写模块；`test_data_pipeline.py` | 移到 `matcher/data/voxel_input.py` | 拥有数据适配和缓存格式契约 |
| `matcher/pp_input.py` | 构造并读取候选级 PP token 缓存 | Dataset、预计算、数据测试 | 正式候选输入缓存读写模块；`test_data_pipeline.py` | 移到 `matcher/data/pp_input.py` | 与 Stage1 概率图适配和缓存版本绑定 |
| `matcher/precompute.py` | 以 PDB 为单位预计算 voxel/PP | 预计算 Shell、数据测试 | 正式数据准备入口；`test_data_pipeline.py` | 移到 `matcher/data/precompute.py` | 只编排数据产物，不训练模型 |
| `matcher/contracts.py` | 声明 `CandidateItem`、`CandidateBatch` 和 metadata | Dataset、collate、模型、指标、训练、测试 | 正式 Dataset/loader 边界；多组测试覆盖 | 移到 `matcher/datasets/contracts.py` | 结构化对象由 Dataset 产生并由 loader 传递 |
| `matcher/candidate_data.py` | 把清单记录装配成单候选 `CandidateItem` | batching、训练、calibration、测试 | 正式 Dataset；数据与模型测试覆盖 | 移到 `matcher/datasets/candidate_data.py` | 与来源适配分开，专注样本装配和缓存读取 |
| `matcher/batching.py` | 实现全局随机 sampler、collate 和 DataLoader 构造 | 训练、calibration、测试 | 正式 loader；数据、模型与运行测试覆盖 | 移到 `matcher/datasets/batching.py` | 与 Dataset 共享进程和批次生命周期 |
| `matcher/cache.py` | 提供按字节预算的进程内 LRU | 图、体素/PP 缓存读写模块、数据测试 | 正式通用支撑；`test_data_pipeline.py` | 移到 `matcher/utils/cache.py` | 不决定任何候选或模型科学语义 |
| `matcher/json_artifacts.py` | 原子写严格 JSON、追加 JSONL | 训练、实验、追踪、calibration、测试 | 正式通用支撑；运行与 calibration 测试覆盖 | 移到 `matcher/utils/json_artifacts.py` | 只管理序列化事务 |
| `matcher/locking.py` | 提供跨进程文件锁 | manifest、voxel/PP、calibration、测试 | 正式通用支撑；数据测试覆盖 | 移到 `matcher/utils/locking.py` | 只管理并发，不拥有产物科学字段 |
| `matcher/losses.py` | 计算六分类、区域和三种身份损失 | 训练配置、训练引擎、模型/运行测试 | 正式训练科学逻辑；`test_model_and_loss.py`、`test_metrics_and_config.py` | 移到 `matcher/training/losses.py` | 损失属于训练职责，并由配置与引擎共同读取 |
| `matcher/experiment_config.py` | 严格解析正式 YAML 与路径 | 实验、训练、追踪、初始化、calibration、测试 | 正式训练配置入口；多组配置测试覆盖 | 移到 `matcher/training/config.py` | 冻结训练阶段、数据和追踪配置 |
| `matcher/checkpointing.py` | 保存、验证和加载 BEST/LAST | 训练、初始化、calibration、测试 | 正式 checkpoint 读写与校验模块；初始化与运行测试覆盖 | 移到 `matcher/training/checkpointing.py` | checkpoint 生命周期由训练阶段拥有 |
| `matcher/initialization.py` | 验证候选预训练 BEST 并初始化候选编码器 | 训练引擎、初始化测试 | 正式训练初始化入口；`test_initialization.py` | 移到 `matcher/training/initialization.py` | 只在构造目标训练运行时使用 |
| `matcher/numeric.py` | 配置随机种子和 PyTorch 数值后端 | 训练、calibration、测试 | 正式数值运行契约；calibration/运行测试覆盖 | 移到 `matcher/training/numeric.py` | 与训练和复用评估的数值环境绑定 |
| `matcher/tracking.py` | 建立并写入 W&B online run | 训练引擎、追踪测试 | 正式训练追踪；`test_tracking.py` | 移到 `matcher/training/tracking.py` | 由训练阶段拥有 run 生命周期 |
| `matcher/training.py` | 训练 epoch、完整划分评估、早停与阶段产物 | 实验、calibration、运行测试 | 正式训练引擎；多组运行测试覆盖 | 移到 `matcher/training/engine.py` | 文件名与包名分开，保留完整阶段主流程而不额外拆函数 |
| `matcher/experiment.py` | 顺序执行 YAML 计划并发布运行摘要 | 训练 Shell、运行测试 | 正式训练命令入口；`test_metrics_and_config.py` | 移到 `matcher/training/experiment.py` | 属于训练计划编排 |
| `matcher/metrics.py` | 累积评估指标并计算 validation 选模分数 | 训练引擎、calibration、指标测试 | 正式评估逻辑；`test_metrics_and_config.py`、`test_calibration.py` | 移到 `matcher/evaluation/metrics.py` | 与优化器和 checkpoint 生命周期无关 |
| `matcher/calibration.py` | 对最终 BEST 执行一次 calibration 评估和 claim 事务 | calibration Shell 与测试 | 正式评估入口；`test_calibration.py` | 移到 `matcher/evaluation/calibration.py` | 与 validation 指标同属评估，但继续受禁读边界约束 |
| `matcher/model/__init__.py` | 导出模型配置、候选编码器和完整 Matcher | 配置、训练、calibration、测试 | 正式稳定模型入口；模型测试覆盖 | 原位保留，更新跨包导入 | 模型子包边界已经清楚 |
| `matcher/model/config.py` | 定义模型结构配置及历史迁移 | 训练配置、模型、初始化、测试 | 正式模型契约；配置与模型测试覆盖 | 原位保留 | 不与运行时配置混合 |
| `matcher/model/raunet.py` | 编码 105×48³ 候选体素 | 候选编码器、模型测试 | 正式模型；`test_model_and_loss.py` | 原位保留 | 单一主干职责明确 |
| `matcher/model/fusion.py` | 编码 A/P/V/PP 并聚合候选表示 | 候选编码器、身份模型、追踪测试 | 正式模型；模型与追踪测试覆盖 | 原位保留，更新数据 schema 导入 | 单一多模态融合职责明确 |
| `matcher/model/candidate_encoder.py` | 组合 RAUNet、融合和辅助头 | 完整 Matcher、训练、calibration、测试 | 正式模型；模型测试覆盖 | 原位保留，改读 `datasets.contracts` 与 `data.schema` | 候选模型边界不变 |
| `matcher/model/ligand_matcher.py` | 组合候选编码器与三条身份分支 | 训练、calibration、模型测试 | 正式模型；`test_model_and_loss.py` | 原位保留，更新跨包导入 | 完整身份模型职责明确 |

## 测试文件清单

| 整理前文件 | 基点职责与被测模块 | 生命周期 | 整理后位置与动作 | 理由 |
|---|---|---|---|---|
| `matcher/tests/__init__.py` | 标记测试包 | 测试代码 | 原位保留 | 测试目录无需随正式模块分层 |
| `matcher/tests/factories.py` | 构造来源根和合成 Stage1/A–G 数据 | 测试代码；被数据、模型和初始化测试复用 | 原位保留，更新 schema 导入 | 单一共享夹具职责明确 |
| `matcher/tests/test_data_pipeline.py` | 覆盖清单、标签、几何、缓存、Dataset、loader 与预计算 | 测试代码 | 原位保留，更新到 `data/datasets/utils` 导入 | 同一端到端数据链便于验证移动等价 |
| `matcher/tests/test_model_and_loss.py` | 覆盖候选编码器、完整 Matcher、聚合与损失 | 测试代码 | 原位保留，更新 `datasets/training` 导入 | 科学断言不变 |
| `matcher/tests/test_metrics_and_config.py` | 覆盖配置、指标、checkpoint、阶段训练与实验编排 | 测试代码 | 原位保留，更新 `training/evaluation/utils` 导入 | 运行状态机断言不变 |
| `matcher/tests/test_tracking.py` | 覆盖 W&B 字段与注意力偏置诊断 | 测试代码 | 原位保留，更新 `training` 导入 | 追踪字段断言不变 |
| `matcher/tests/test_initialization.py` | 覆盖候选预训练来源核对与严格加载 | 测试代码 | 原位保留，更新 `training` 导入 | 失败关闭与参数不变断言不变 |
| `matcher/tests/test_calibration.py` | 覆盖一次性评估编排、claim 与恢复 | 测试代码 | 原位保留，更新 `evaluation/training/data/utils` 导入 | 不运行真实 calibration 数据 |

## 移动顺序与验收

1. 新建空的 `utils`、`data`、`datasets`、`training`、`evaluation` 子包；先移动通用支撑和数据适配，再移动 Dataset/loader。
2. 保持 `model/` 文件原位，只更新它们对数据契约的导入。
3. 移动训练与评估模块；`training.engine` 继续拥有 `evaluate_model`，避免为纯目录整理拆开已审查的训练主流程。
4. 更新四个 Shell 的 `python -m` 入口、测试导入和当前 README。历史执行记录中的旧命令作为当时事实保留，不机械改写。
5. 删除全部旧顶层模块；不建立兼容包装。用 `rg` 证明正式代码、Shell 和测试没有旧导入。
6. 重新运行完整 pytest、mypy、pyflakes、compileall、19 份 YAML 解析、8 个 Shell 的 `bash -n` 和 `git diff --check`。
7. 对移动前后 Python 文件做 AST 归一比较；除导入路径和模块说明外，函数、类、常量与可执行语句必须等价。独立科学审查确认科学契约未漂移，严格技术表达审查确认目录和说明可以冷读。

若移动需要改变函数签名、默认值、配置字段、数据路径、缓存 schema、checkpoint schema、指标口径或命令参数，本轮立即停止，并把该变化作为独立科学实现另行审批。

## 执行状态

2026-08-14 已按上述映射完成本地整理。`matcher/` 顶层只保留包入口，正式模块分布在 `data/datasets/model/training/evaluation/utils` 六个职责子包；五个新增子包入口只含模块职责说明，不转发实现。四个 Shell 命令和十九份正式 YAML 的读取入口已同步到新模块路径，旧顶层 Python 模块不保留兼容包装。

对基点 `a74deca9b7a3f6a58909410623ba1c8b99eb9625` 的 36 个既有 Python 文件执行语法树归一比较：移除导入语句与模块说明后，函数、类、常量和可执行语句全部相同。六个测试文件按四组得到 `35 + 28 + 24 + 31 = 118 passed`；其中 31 项由 calibration、初始化和追踪三个测试文件合并执行。mypy 检查 `matcher/` 下 45 个 Python 文件零问题，pyflakes、compileall、19 份 YAML 正式解析、8 个 Shell 的 `bash -n`、四个新模块入口的 `--help`、真实 `spawn` 工作进程创建与退出，以及 `git diff HEAD --check` 均通过。

本次整理没有修改配置值、数据和缓存路径、模型或损失计算、指标、随机状态、批次顺序、checkpoint/JSON schema 或 calibration 禁读边界。运行中的六个受管 Job 继续读取各自既有内容寻址 release；本次本地整理没有访问或修改这些 release、动态命令、Slurm 作业和任何锁。独立科学正确性与严格技术表达复核均为 PASS，阻断项为零。

实现端点为 `codex/matcher-package-layout@6176122`；学习历史按“职责清单→运行模块与入口→测试导入→文档证据”重建到 `Learn/matcher-package-layout@8d6ba38`。两个端点的树摘要均为 `dd86b658cab8d0401248726563ff504b662a2c56`，逐文件内容完全一致；`Learn/CUMULATIVE` 已快进到学习端点。
