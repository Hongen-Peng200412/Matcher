# Matcher 全库细粒度注释清单

本文列出 Matcher 正式源码、配置和学习文档需要补充细粒度注释的完整范围，并保留测试文件作为只读行为门禁清单。`369152756818f04730efd5865ad8fdb18ed9b620` 是 A–F 批 Python 可执行内容的历史比较基点；`2f80bc9880591c69c1e51f7125bb4f989ab22b9f` 是本次最终整合的累计学习基点。细粒度注释工作树的 85 项资产已经进入最终双线历史：77 个路径逐字节映射，6 个实现或配置交叉路径保留累计线最新可执行逻辑并移植已审注释，`matcher/model/README.md` 与本清单 2 个最终整合文档按当前 concat 和 Git 事实更新。目标是让读者沿“来源文件 → 候选样本 → 批次张量 → 模型 → 损失与评估 → 训练产物”的真实数据链冷读正式代码；本轮只允许修改中文注释、Docstring、YAML 注释、包级 README 和 `talk/` 学习文档，不允许修改 Python 可执行语句、函数签名、常量、YAML 值或测试行为。

源码 Docstring 使用逐实体的 `    - 字段名:` 物理行说明结构化输入和输出，不使用双反引号包裹变量名。每个字段按事实补充类型、形状、轴顺序、单位、来源、缺失值、掩码、索引或 offsets 关系；没有事实依据的单位和领域含义明确保留为尚未核实，不从变量名猜测。

## 注释批次与验收

| 批次 | 文件范围 | 主要阅读链 | 每批机械验收 | 当前状态 |
| --- | --- | --- | --- | --- |
| A | `matcher/data/` 全部 10 个 Python 文件 | 正式清单入口、来源产物字段、几何与标签、体素和 PP 缓存、LigandObject 图、预计算入口 | 去除注释与 Docstring 后的 AST 等价；compileall、pyflakes、mypy；`test_data_pipeline.py`；变更路径清单和 diff 核对 | 独立终审 PASS；第五次冻结 |
| B | `matcher/datasets/` 全部 4 个 Python 文件 | `CandidateItem`、`CandidateBatch`、Dataset 读取链、collate 和 DataLoader | 同类 AST 等价；数据、模型相关 pytest；类型检查 | 第五次冻结独立终审 PASS；保留前四次 FAIL 历史 |
| C | `matcher/model/` 全部 6 个 Python 文件 | 候选编码器、RAUNet、四模态聚合、配体身份三分支 | 同类 AST 等价；`test_model_and_loss.py`；BF16/CUDA 测试按可用环境分开记录 | 三次冻结独立终审 PASS；保留前两次 FAIL 历史；累计线后增 concat 聚合器已补细粒度注释，待最终整合终审 |
| D | `matcher/training/` 全部 9 个 Python 文件 | YAML 解析、损失、初始化、checkpoint、数值后端、W&B、训练与 validation | 同类 AST 等价；训练、配置、初始化、追踪测试；正式 YAML 解析 | 二次冻结定点终审 PASS；保留首审 FAIL 历史 |
| E | `matcher/evaluation/` 全部 3 个 Python 文件 | 完整划分指标、validation 选模、一次性 calibration | 同类 AST 等价；指标与 calibration 测试；不读取真实 calibration | 二次冻结最终窄复核 PASS |
| F | `matcher/utils/` 全部 4 个 Python 文件与根包入口 | 字节预算缓存、严格 JSON、文件锁、稳定包导出 | 同类 AST 等价；相关 pytest；类型检查 | 二次冻结最终窄复核 PASS；保留首审 FAIL 历史 |
| G | `matcher/tests/` 全部 8 个 Python 文件 | 只作为合成来源产物、科学不变量与失败关闭的行为门禁 | 不修改测试注释、Docstring 或可执行内容；按各批范围运行 pytest | 用户取消；不纳入注释范围 |
| H | `configs/experiments/` 全部 34 份正式 YAML 与现有 7 份批次 README | `load_experiment_plan` 的真实读取位置、特殊值、批次对照变量与目录内配置依赖 | PyYAML 与正式加载器双重解析；全部读取位置核实；解析后的键值完全相同，剥除全行和行尾注释后的值行保持相同；README 历史结论不变 | 二次冻结最终窄复核 PASS；保留首审 FAIL 历史；累计线后增 12 份 concat/mean-max 配置不冒充 H 批审查对象 |
| I | 包级导航与两份既有 `talk/` 概览 | 子包入口和模块依赖 | 六个正式子包 README 已分别随 A–F 批验收；两份既有概览保持基点内容 | 不另立改写批次；不把未改概览计入 85 项注释资产 |

## Python 文件清单

### 包入口与数据适配

| 文件 | 职责、主要入口与具体调用者 | 结构化对象、字段与关键非标量变量 | 既有验证 | 批次与状态 |
| --- | --- | --- | --- | --- |
| `matcher/__init__.py` | 包级稳定导出；包使用者从这里获得类别顺序和清单 schema 版本 | `CLASS_NAMES`、`MANIFEST_SCHEMA_VERSION` | 导入测试与配置测试间接覆盖 | F；二次冻结最终窄复核 PASS |
| `matcher/data/__init__.py` | 标记 Stage1 与 A–G 来源适配子包；正式入口使用完整模块路径 | 无运行字段 | 包导入 | A；独立终审 PASS |
| `matcher/data/schema.py` | 定义数据版本、类别、联合身份、语言家族、裁剪策略、维度和来源根目录；数据、Dataset、模型、训练和评估共同读取 | `CLASS_NAMES`、`IDENTITY_KEY_FIELDS`、`PreparedSmilesStatus`、`SourceRoots`；A/P/V/PP token 宽度；手工特征和采样特征名称；80³ 与 48³ 空间常量 | 数据、模型、配置测试 | A；独立终审 PASS |
| `matcher/data/labels.py` | `assign_candidate_target` 用候选体素与 occurrence 体素的双向覆盖率产生六分类和身份匹配监督；清单构建器调用 | `OccurrenceMask` 的候选编号、模板键、类型和 ZYX 体素；`CandidateTarget` 的类别、命中键、双向覆盖率和乘积得分 | 覆盖率排序与空间边界测试 | A；独立终审 PASS |
| `matcher/data/geometry.py` | `resolve_candidate_crop` 统一 80³ 来源坐标与 48³ 局部裁剪；Dataset、体素缓存、PP 缓存和预计算共同调用 | `CandidateCropGeometry` 的请求/实际 ZYX 起点、V 质心、中心偏移、边界裁剪掩码和保留计数；世界坐标、体素索引和受体散射数组 | 六个边界面、ties-to-even、各来源协同裁剪测试 | A；独立终审 PASS |
| `matcher/data/density.py` | `build_density_channels` 把实验密度、模拟密度和受体硬掩码变成 56 个 Find_0 密度通道；`VoxelInputStore` 调用 | 三个同形非空 `(Z,Y,X)` 网格；当前正式调用为 80³；缩放拟合、截断归一、Gaussian 平滑结果；`DENSITY_CHANNEL_NAMES` 与输出第一维逐通道对齐 | 体素输入与几何一致性测试 | A；独立终审 PASS |
| `matcher/data/ligand_graph.py` | `LigandGraphStore.load` 按 `object_key` 读取 LigandObject NPZ 并建立缓存；Dataset 读取，批次契约和身份编码器消费 | `LigandGraph` 的元素、形式电荷、残基编号、连续节点特征、边索引和边特征；节点维与边维对齐关系 | LigandObject、跨残基图与模型图分支测试 | A；独立终审 PASS |
| `matcher/data/manifest.py` | `ensure_manifest`、`require_manifest`、`build_manifest` 扫描冻结 Stage1 F1 候选并发布 schema v4 清单；训练、预计算、Dataset 和 calibration 使用 | manifest JSON；每个划分的 candidates/identities JSONL；exclusions 与 summary；联合身份 `(object_key, prepared_smiles)`；语言表示选择、seen 标记和跨 PDB 诊断 | `test_data_pipeline.py` 中 v4 联合身份、缺失、划分隔离、扩展和失败关闭测试 | A；独立终审 PASS |
| `matcher/data/voxel_input.py` | `VoxelInputStore.load` 构造或读取候选级 `(105,48,48,48)` 体素缓存；Dataset 与预计算调用 | `_CenteredGeometry`、`_VoxelSources`；实验/模拟密度、XYZ 体素大小和原点、受体坐标与 49 维特征；缓存路径绑定裁剪起点 | 各向异性几何、位移拒绝、裁剪策略隔离和缓存键测试 | A；独立终审 PASS |
| `matcher/data/pp_input.py` | `PpInputStore.load` 构造或读取候选级 PP token 缓存；Dataset 与预计算调用 | `PpCandidateInput` 与 `_PpPdbCache` 的 token、ZYX 索引、裁剪起点、概率图原点和体素大小；PDB 缓存采用固定 `(C,K,...)` 候选轴布局，不使用 offsets | top-k 选择、缓存脱离概率图和裁剪策略隔离测试 | A；独立终审 PASS |
| `matcher/data/precompute.py` | `precompute_inputs` 只读既有 manifest 并按 PDB 并行预计算 voxel/PP；Shell 入口 `main` 才在调用它之前执行 `ensure_manifest` | PDB 工作项列表、`PrecomputeProduct`、每 PDB 候选计数和失败传播 | 显式 products、PP 产物与进程边界测试 | A；独立终审 PASS |

### Dataset 与批次

| 文件 | 职责、主要入口与具体调用者 | 结构化对象、字段与关键非标量变量 | 既有验证 | 批次与状态 |
| --- | --- | --- | --- | --- |
| `matcher/datasets/__init__.py` | 标记 Dataset、批次和 DataLoader 子包 | 无运行字段 | 包导入 | B；第五次冻结独立终审 PASS |
| `matcher/datasets/contracts.py` | 声明跨模块传递的 `CandidateMetadata`、`CandidateItem` 和 `CandidateBatch`；Dataset 生产，collate、模型、损失与指标消费 | 单候选与批次的体素、A/P/V/PP token 和 mask、区域监督、联合身份字符串、语言向量、配体图拼接数组、图到样本/身份索引 | 数据、模型、损失和指标测试 | B；第五次冻结独立终审 PASS |
| `matcher/datasets/candidate_data.py` | `CandidateDataset.__getitem__` 把一条冻结候选清单记录装配为 `CandidateItem`；训练和 calibration 构造 | 清单记录、PDB 级缓存、联合身份槽、语言向量、图列表、manual/sampling 特征、四模态 token、区域标签与身份目标 | 数据链与模型输入测试 | B；第五次冻结独立终审 PASS |
| `matcher/datasets/batching.py` | `build_candidate_loader`、`collate_candidates` 与 `ShuffledBatchSampler` 形成固定物理批次 | padding token/mask、identity mask、图节点/边拼接、graph-to-sample 和 graph-to-identity 索引、epoch 随机排列 | 采样确定性、跨硬件候选顺序、模型批次测试 | B；第五次冻结独立终审 PASS |

### 模型

| 文件 | 职责、主要入口与具体调用者 | 结构化对象、字段与关键非标量变量 | 既有验证 | 批次与状态 |
| --- | --- | --- | --- | --- |
| `matcher/model/__init__.py` | 导出模型配置、候选编码器与完整闭集 Matcher | `__all__` 中六个稳定符号 | 导入与模型测试 | C；三次冻结独立终审 PASS |
| `matcher/model/config.py` | `MatcherModelConfig` 固定模型科学字段并与 checkpoint/JSON 字典转换 | mean/max、条件注意力与摘要拼接残差三种聚合方式；三类特征开关、两门初值、维度、RAUNet 平面数、回收次数、checkpoint 和图层数 | 配置迁移与非法契约测试 | C；三次冻结独立终审 PASS；第三种聚合随最终整合核验 |
| `matcher/model/raunet.py` | `RaUnetBackbone.forward` 把 105 通道 48³ 候选体素编码为同分辨率特征 | 多尺度 3D 特征、位置网格、三轴旋转位置编码、skip、attention gate 和 recycle 状态 | 形状、BF16、梯度 checkpoint 和无死参数测试 | C；三次冻结独立终审 PASS |
| `matcher/model/fusion.py` | A/P/V/PP token 投影、候选条件注意力、mean/max 聚合、两个完整摘要的零输出残差拼接和手工/采样特征归一 | token/mask、voxel gather 索引、注意力偏置特征、每模态摘要、两个 `(B,D_m)` 完整摘要、`(B,2D_m)` 拼接向量和 `(B,D_m)` 候选表示 | 空模态、归一、注意力零初始化、随机流隔离与梯度唤醒测试 | C；既有结构三次冻结独立终审 PASS；concat 新增类已补实例字段和非标量局部变量注释，待最终整合终审 |
| `matcher/model/candidate_encoder.py` | `CandidateEncoder.forward` 组合 RAUNet、三种四模态聚合、六分类与两个体素头 | `CandidateEncoderOutput` 的候选向量、类别 logits、配体/结合区域 logits 和体素特征 | 候选形状、BF16、辅助头和 concat 退化等价测试 | C；三次冻结独立终审 PASS；concat 分派随最终整合核验 |
| `matcher/model/ligand_matcher.py` | `ClosedSetMatcher.forward` 组合候选编码器与 native/graph/inventory 三条身份分支 | `IdentityRepresentations`、`IdentityScoreComponents`、`FullMatcherOutput`；图消息、候选—身份配对、排除自身的 inventory 注意力和门控增量 | 三类身份变体、门控零增量、单身份、图重复与分量求和测试 | C；三次冻结独立终审 PASS |

### 训练

| 文件 | 职责、主要入口与具体调用者 | 结构化对象、字段与关键非标量变量 | 既有验证 | 批次与状态 |
| --- | --- | --- | --- | --- |
| `matcher/training/__init__.py` | 标记配置、损失、训练状态和实验编排子包 | 无运行字段 | 包导入 | D；二次冻结定点终审 PASS |
| `matcher/training/config.py` | `load_experiment_plan` 严格解析正式 YAML | `DataConfig`、`TrackingConfig`、`TrainingConfig`、`StageConfig`、`ExperimentPlan` 的全部字段及路径和特殊值 | 34 份正式 YAML、非法键、输出逃逸、种子和初始化规则测试 | D；二次冻结定点终审 PASS |
| `matcher/training/losses.py` | `compute_matcher_loss` 计算六分类、两个区域和身份损失 | `LossWeights`、`MatcherLossBatch`、`LossOutput`；identity mask、class index 与可监督计数 | 三种身份变体、空身份、全局批次累积测试 | D；二次冻结定点终审 PASS |
| `matcher/training/checkpointing.py` | 原子保存、验证和加载 BEST/LAST checkpoint | `CheckpointEnvelope` 的 schema 版本与模型状态；候选编码器前缀映射 | checkpoint schema 与候选前缀加载测试 | D；二次冻结定点终审 PASS |
| `matcher/training/initialization.py` | 校验候选预训练 BEST 及关联记录，只加载完整模型的候选编码器 | 三个 TypedDict 中 checkpoint、来源计划、模型和科学数据嵌套字段；missing/unexpected keys | Batch4 初始化、历史字段迁移与失败前参数不变测试 | D；二次冻结定点终审 PASS |
| `matcher/training/numeric.py` | 配置 Python、NumPy、Torch 随机种子与数值后端 | `NumericBackendRecord` 的种子、cuDNN、确定性算法和矩阵乘精度 | calibration 与训练运行测试 | D；二次冻结定点终审 PASS |
| `matcher/training/tracking.py` | `start_wandb_tracking` 建立 W&B online run，`WandbTracker` 写 global step、epoch 和 summary | `TrackingMetadata`、W&B Protocol、规范化指标映射和空值规则 | online 初始化、间隔、非有限值、摘要与失败收口测试 | D；二次冻结定点终审 PASS |
| `matcher/training/engine.py` | `run_training_stage` 构造单卡运行时并完成梯度累积、逐 epoch validation、BEST/LAST 和早停 | 损失统计、梯度与门控诊断、训练/评估结果、epoch 记录、阶段摘要和 `_StageRuntime`；批次迁移、AMP、scheduler、checkpoint 与 W&B | 配置、指标、累积损失、DataLoader spawn、初始化和追踪测试 | D；二次冻结定点终审 PASS |
| `matcher/training/experiment.py` | `run_plan` 在唯一运行目录内顺序执行 YAML 各阶段 | resolved plan、运行记录、阶段摘要、运行 summary 和退出状态 | 阶段顺序、严格 JSON 和失败收口测试 | D；二次冻结定点终审 PASS |

### 评估与通用支撑

| 文件 | 职责、主要入口与具体调用者 | 结构化对象、字段与关键非标量变量 | 既有验证 | 批次与状态 |
| --- | --- | --- | --- | --- |
| `matcher/evaluation/__init__.py` | 标记指标与一次性 calibration 子包，不反向导出 calibration | 无运行字段 | 包导入 | E；二次冻结最终窄复核 PASS |
| `matcher/evaluation/metrics.py` | `EvaluationAccumulator` 累积完整划分指标，`selection_score` 只用 validation 报告选模 | 六分类混淆矩阵、逐类指标、intrinsic/conditional/end-to-end 身份计数、seen/unseen、区域 Dice 和 `EvaluationReport` | 指标口径、非有限值、轴对齐与 selection 测试 | E；二次冻结最终窄复核 PASS |
| `matcher/evaluation/calibration.py` | `run_calibration` 对一个 BEST checkpoint 执行一次 calibration claim 与评估 | checkpoint、数据、运行时、artifact 和 claim 五类 TypedDict；路径与字节摘要只属于此一次性评估边界 | claim、绑定、恢复、重入与安全输出测试；本轮不读真实 calibration | E；二次冻结最终窄复核 PASS，未读取真实 calibration |
| `matcher/utils/__init__.py` | 标记运行支撑子包且不建立包级转发接口 | 无运行字段；导入时不预先导入三个工具模块 | 包导入 | F；二次冻结最终窄复核 PASS |
| `matcher/utils/cache.py` | `ByteBudgetLru` 按 `array_nbytes` 选定载荷管理单实例缓存 | 有序键值、插入时当前/最大字节数、递归数组计量与不统计边界 | LRU 提升与逐字节淘汰测试 | F；二次冻结最终窄复核 PASS |
| `matcher/utils/json_artifacts.py` | 同步同目录临时 JSON 后替换目标，或同步追加 JSONL | JSON 映射、Python 非有限 float 转换、PID 临时文件、文件同步、并发与失败边界 | 严格 JSON 与运行记录测试 | F；二次冻结最终窄复核 PASS |
| `matcher/utils/locking.py` | `exclusive_file_lock` 提供同路径合作进程互斥 | 锁文件句柄、Windows 首字节/非 Windows flock 分支和加锁、正文、解锁、关闭异常优先级 | 加锁失败和解锁失败测试 | F；二次冻结最终窄复核 PASS |

### 测试

| 文件 | 职责、主要入口与具体调用者 | 结构化对象、字段与关键非标量变量 | 既有验证 | 批次与状态 |
| --- | --- | --- | --- | --- |
| `matcher/tests/__init__.py` | 标记测试包 | 无运行字段 | pytest 收集 | G；用户取消注释，只读门禁 |
| `matcher/tests/factories.py` | `build_synthetic_roots` 构造最小可端到端读取的 Stage1、A–G、语言与 LigandObject 产物 | 合成清单根目录、centered NPZ、语言 NPZ 和配体图 NPZ 字段 | 被数据、模型和初始化测试复用 | G；用户取消注释，只读门禁 |
| `matcher/tests/test_data_pipeline.py` | 验证清单、联合身份、标签、几何、缓存、Dataset、loader、预计算和锁 | 合成 occurrence、SMILES 替换、候选/身份记录和缓存字段 | 本文件测试 | G；用户取消注释，只读门禁 |
| `matcher/tests/test_model_and_loss.py` | 验证候选模型、融合、完整身份模型、门控和损失 | 小模型配置、合成 `CandidateBatch`、前向分量和梯度 | 本文件测试 | G；用户取消注释，只读门禁 |
| `matcher/tests/test_metrics_and_config.py` | 验证指标、配置、checkpoint、阶段运行、严格 JSON 和正式 YAML | `EvaluationBatch`、实验计划、运行产物和 monkeypatch 运行时 | 本文件测试，覆盖 34 份 YAML | G；用户取消注释，只读门禁 |
| `matcher/tests/test_initialization.py` | 验证 Batch4 候选 BEST 初始化和失败关闭 | 合成 checkpoint、运行记录、计划 YAML 与目标 stage | 本文件测试 | G；用户取消注释，只读门禁 |
| `matcher/tests/test_tracking.py` | 验证 W&B run 生命周期和注意力偏置诊断 | 三个 fake W&B 类型、记录列表和 summary 映射 | 本文件测试 | G；用户取消注释，只读门禁 |
| `matcher/tests/test_calibration.py` | 验证一次性 calibration artifact、claim 和第二次调用拒绝 | Dataset stub、固定 evaluation 和证据文件 | 本文件测试 | G；用户取消注释，只读门禁 |

## A 批独立审查与修订历史

A 批只修改 `matcher/data/` 的 10 个 Python 文件、[`matcher/data/README.md`](../matcher/data/README.md) 和本清单。首轮机械门禁通过后，独立首审于 2026-08-15 判定 FAIL；按首审清单修订并重新通过门禁后形成二次冻结。独立二审再次判定 FAIL，原因是函数边界仍有结构化对象字段省略，README 也未能冷读回答全部掩码、105 通道和 LigandGraph 契约。第三轮定点修订形成第三次冻结后，独立三审仍判定 FAIL，剩余问题是少量 Docstring 条目格式、README 首次术语定义与轴名歧义，以及 LigandObject 来源 schema 和 Matcher 读取器实际门禁没有明确分层。第四轮定点修订形成第四次冻结后，独立四审仍判定 FAIL，剩余问题是 atom_labels 内容核验、LigandGraph 读取门禁和允许文档范围各有一处过称或遗漏。第五轮按四审及补充清单完成四处文字修订并冻结，随后独立终审判定 PASS。源码范围始终只包含模块 Docstring、类/函数 Docstring 与局部变量注释，不修改可执行语句、函数签名、常量或测试。

第五次冻结重新执行并通过的机械门禁如下：

- 逐个比较 10 个 Python 文件相对 `369152756818f04730efd5865ad8fdb18ed9b620` 的语法树；移除模块、类和函数 Docstring 后，可执行 AST 全部相同。
- `python -m compileall -q matcher/data` 通过；生成的 Python 字节码目录外置到系统临时目录。
- `python -m pyflakes matcher/data` 通过。
- `python -m mypy matcher` 检查 45 个 Python 文件，结果为零问题；mypy 缓存外置到系统临时目录。
- `python -m pytest matcher/tests/test_data_pipeline.py -q -p no:cacheprovider` 得到 `62 passed`，且禁用 pytest 工作树缓存。
- `python -m matcher.data.manifest --help` 与 `python -m matcher.data.precompute --help` 都能正常生成帮助并以 0 退出。两者的说明正文随模块 Docstring 变得更完整；参数名、类型、必填性、choices、默认值、解析和退出逻辑所在的可执行 AST 未变。
- `matcher/training/experiment.py` 与 `matcher/evaluation/calibration.py` 尚未进入本批，因此另外两个读取模块 Docstring 的 CLI 帮助正文没有变化。
- `git diff --check` 通过，工作树内没有 `.mypy_cache`、`.pytest_cache` 或 `__pycache__`。

## B 批首次冻结

B 批只修改 `matcher/datasets/__init__.py`、`matcher/datasets/contracts.py`、`matcher/datasets/candidate_data.py`、`matcher/datasets/batching.py`、[`matcher/datasets/README.md`](../matcher/datasets/README.md) 和本清单。四个 Python 文件已经逐参数、逐结构化字段和逐关键非标量变量说明候选物化、缓存分账、四模态、区域目标、联合身份、补零掩码、图连接、采样器与 DataLoader；源码只改变 Docstring 和注释。

首次冻结前执行并通过的机械门禁如下：

- 四个 Python 文件相对 `369152756818f04730efd5865ad8fdb18ed9b620` 去除模块、类和函数 Docstring 后，可执行 AST 全部相同。
- 四个 Python 文件统一为 LF；源码注释与 Docstring 不含反引号；全部模块、类和函数都有 Docstring，函数参数与 `TypedDict` 字段都有独立的 `- 实体名:` 物理行。
- `python -m compileall -q matcher/datasets` 通过，Python 字节码目录外置到系统临时目录。
- `python -m pyflakes matcher/datasets` 通过。
- `python -m mypy matcher` 检查 45 个 Python 文件，结果为零问题；mypy 缓存外置到系统临时目录。
- `python -m pytest matcher/tests/test_data_pipeline.py matcher/tests/test_model_and_loss.py matcher/tests/test_metrics_and_config.py -q -p no:cacheprovider` 得到 `130 passed`。
- `matcher/datasets/README.md` 与本清单的相对 Markdown 链接通过存在性检查；`git diff --check` 通过，工作树缓存扫描未发现 `.mypy_cache`、`.pytest_cache` 或 `__pycache__`。
- B 批模块没有把模块 Docstring 传给 argparse；本批不存在 A 批两个 CLI 说明正文的允许差异。

首次冻结后的独立首审判定 FAIL；四个 Python 文件的可执行 AST、LF、源码无反引号和字段条目格式均通过，失败只涉及技术表达事实。首审要求修正 `type_tag=other` 背景仍可保留 `matched_object_key`、候选 seen 的 `None` 边界、畸形 occurrence mask 与统一错误批次形状可能被 NumPy 接受、跨物理批量采样序列的完整条件、`identity_scope` 非空枚举检查的实际时点、`__getitem__` 消费的候选 JSON 嵌套字段、缓存余量和来源根目录边界。本节保留首次冻结门禁事实。

首审清单已经逐项定点修订并形成二次冻结。二次冻结再次取得四个 Python 文件去 Docstring AST 等价、LF、无源码反引号、参数与 `TypedDict` 字段条目格式 PASS；`compileall`、`pyflakes` 和 45 文件 mypy 均通过；同一二次冻结态的三份 pytest 文件得到 `130 passed`。Markdown 链接、diff 和工作树缓存扫描在二次冻结前再次核对。

二次冻结后的独立复审仍判定 CONTENT FAIL；首审问题均已闭合，剩余三项是身份 JSONL 的 `seen_in_train_by_language` 被误写成候选 metadata 来源、根目录表遗漏 VoxelInputStore 的密度来源与缓存写入副作用，以及 `__getitem__` 的局部注释把未在本地核验的 `pdb_id` 无条件称为小写。第三轮只修订这些技术表达和本段历史，形成第三次冻结。

第三次冻结重新取得四个 Python 文件去 Docstring AST 等价、LF、源码无反引号和结构化条目格式 PASS；`compileall`、`pyflakes` 与 45 文件 mypy 通过。针对 Dataset 构造门禁、联合身份记录与跨硬件采样序列选择的快速测试得到 `8 passed`。本次只改 Docstring、局部注释和 Markdown，未提前重复运行二次冻结已通过的 130 项完整测试；待独立内容复审不再发现表达问题后，再把完整测试绑定到最终冻结态。

第三次冻结后的独立复审仍判定 CONTENT FAIL；此前三项已经闭合，剩余两处均是新补缓存说明的过称：`__getstate__` 只修改返回的序列化状态而不清空父进程对象，且 LRU 在预算为零、载荷超预算或体素 NPY 直接命中时不一定填充。第四轮统一修订这些缓存边界、预算为零的特殊值和 occurrence 首次定义，形成第四次冻结。

第四次冻结后的独立复审确认缓存修订正确，但 occurrence 的首次定义仍误写成必然对应 PreparedLigand 记录；PreparedLigand 文件允许整体缺失且不覆盖 `type_tag=other`。第五轮只把 occurrence 收紧为“一个 PDB 中的一次具体配体实例”，形成第五次冻结。

第五次冻结的独立终审判定 PASS。独立审查确认四个 Python 文件去 Docstring 后的可执行 AST 等价、字段级说明和 README 冷读契约闭合，并在同一冻结状态运行完整测试得到 `130 passed`；B 批至此收口。

## C 批初次冻结

C 批只修改 `matcher/model/__init__.py`、`matcher/model/config.py`、`matcher/model/candidate_encoder.py`、`matcher/model/raunet.py`、`matcher/model/fusion.py`、`matcher/model/ligand_matcher.py`，新增 [`matcher/model/README.md`](../matcher/model/README.md)，并同步本清单。六个源码已经逐参数、逐结构化字段和逐关键非标量变量说明候选体素、四模态 token、C 批冻结时已有的两种聚合、RAUNet 多尺度与回收、语言/图/清单身份表示、三个配对分数组件和最终补零屏蔽；源码只改变 Docstring 与注释。后来进入累计线的第三种 concat 聚合在本文件“最终整合映射证据”中单独记录，不追写为 C 批历史审查对象。

初次冻结前执行并通过的机械门禁如下：

- 六个 Python 文件相对 `369152756818f04730efd5865ad8fdb18ed9b620` 去除模块、类和函数 Docstring 后，可执行 AST 全部相同。
- 初次冻结自检曾把六个 Python 文件报告为 LF；独立首审随后按原始字节计数证明六文件均混有 CRLF/LF，因此该项自检结论无效。源码无反引号、Docstring 与参数/字段条目的其它机械结论仍保留。
- `python -m compileall -q matcher/model` 通过，Python 字节码目录外置到系统临时目录。
- `python -m pyflakes matcher/model` 通过。
- `python -m mypy matcher` 检查 45 个 Python 文件，结果为零问题；mypy 缓存外置到系统临时目录。
- `python -m pytest matcher/tests/test_model_and_loss.py -q -p no:cacheprovider` 得到 `31 passed in 147.88s`。本机 `torch.cuda.is_available()` 为真且设备数为 1，因此文件内 CUDA 条件分支也在本次运行环境中具备执行条件。
- C 批不涉及四个把模块 Docstring 传给 argparse 的 CLI 文件，不新增允许的帮助文本差异。
- `matcher/model/README.md` 与本清单的相对 Markdown 链接、`git diff --check`、精确变更范围和工作树缓存将在初次冻结哈希固定前再次核对。

本节只记录编写代理的初次冻结结果；内容正确性和严格技术表达仍待独立审查，不把自检通过写成独立 PASS。

C 批初次冻结的独立审查判定 FAIL，但再次确认六个 Python 文件去 Docstring 后的可执行 AST 仍为 6/6 等价。阻断项全部属于技术表达与文件格式：六文件混合换行；回收最后一轮被无条件写成建图或反向而遗漏外层梯度模式；有限精度 Sigmoid 被过称为开区间；多个类没有逐项列齐真实 `self` 字段；通用 GraphMessageLayer 的 `F_e` 被写死为正式调用的 10；配置 dropout 消费者和 `from_dict` Python 值边界不完整；README 把没有具名正式消费者的 `voxel_features` 写成用于诊断。编写代理已按该清单恢复 C 批定点修订，未进入 D 批。

定点修订完成后形成 C 批二次冻结：六个 Python 文件原始字节均为 `CRLF=0` 的 LF-only；相对同一基点去 Docstring 可执行 AST 仍为 6/6 等价；源码无反引号；机械扫描确认全部真实 `self` 赋值字段和全部函数参数各有独立条目。回收说明现已区分内部前置轮 `no_grad` 与最后一轮服从外层梯度模式，两个 Sigmoid 说明收紧为有限精度 `[0,1]`，其余首审内容项也逐项闭合。`compileall` 与 `pyflakes` 快速门禁通过；首冻已通过的全库 mypy 和 31 项模型测试不在这个中间内容复审前重复运行。二次冻结仍待独立复审，不提前记为 PASS。

C 批二次冻结的独立复审仍判定 CONTENT FAIL；首审八项均已闭合，唯一剩余问题是回收说明把训练模式误写成最后一轮建立自动求导计算图的必要条件。`RaUnetBackbone.forward` 的最后一轮不检查 `self.training`；是否建图只服从外层梯度模式及参与运算张量的 `requires_grad`，而训练模式只参与激活重计算开关。第三轮据此定点修订源码 Docstring、模型 README 和本段历史，形成 C 批三次冻结；本次只需重新核对可执行 AST、换行和格式，不提前重复长门禁。

C 批三次冻结的独立终审判定 PASS。独立审查确认六个 Python 文件相对比较基点去模块、类与函数 Docstring 后的可执行 AST 均等价，源码为 LF-only 且没有反引号；模块、类、函数、参数、结构化字段和真实 `self` 赋值字段的逐实体条目全部通过。独立门禁还包括 `compileall`、`pyflakes`、全库 45 个源码文件 mypy 零问题、`test_model_and_loss.py` 的 `31 passed in 143.23s`、本机 CUDA 条件分支、Markdown 相对链接、`git diff --check`、精确路径范围和缓存扫描；三次冻结的八个文件哈希在测试前后保持不变。C 批至此收口，前两次 FAIL 作为历史时点保留。

## D 批初次冻结

D 批只修改 `matcher/training/__init__.py`、`matcher/training/config.py`、`matcher/training/losses.py`、`matcher/training/checkpointing.py`、`matcher/training/initialization.py`、`matcher/training/numeric.py`、`matcher/training/tracking.py`、`matcher/training/engine.py`、`matcher/training/experiment.py`，新增 [`matcher/training/README.md`](../matcher/training/README.md)，并同步本清单。九个源码已经逐字段说明正式 YAML、损失分量与分母、初始化来源、checkpoint 信封、随机与数值后端、W&B 生命周期、阶段运行时、epoch 记录和运行产物；源码只改变 Docstring 与注释。

初次冻结前执行并通过的机械门禁如下：

- 九个 Python 文件相对 `369152756818f04730efd5865ad8fdb18ed9b620` 去除模块、类和函数 Docstring 后，可执行 AST 全部相同。
- 九个 Python 文件均为 LF-only；源码注释与 Docstring 没有反引号；模块、类、函数、函数参数、结构化字段和真实 `self` 字段均已执行逐实体条目扫描。
- `python -m compileall -q matcher/training` 和 `python -m pyflakes matcher/training` 通过，Python 字节码目录外置到系统临时目录。
- `python -m mypy matcher` 检查 45 个 Python 文件，结果为零问题；mypy 缓存外置到系统临时目录。
- 全部 34 份正式 YAML 同时通过 PyYAML 安全解析与 `load_experiment_plan` 正式解析；本批未修改 YAML 文件或解析后的键值。
- `python -m matcher.training.experiment --help` 正常生成帮助并以状态码 0 退出；私有参数解析器的 `--plan` 类型、缺参和非法参数状态码均保持原行为。该模块把模块 Docstring 作为 argparse 说明正文，因此仅允许说明文字变化，不宣称帮助文本逐字相同。
- `python -m pytest matcher/tests/test_initialization.py matcher/tests/test_tracking.py -q -p no:cacheprovider` 得到 `30 passed in 18.68s`；完整训练、配置和损失测试留待内容审查通过后的最终冻结态一次执行，避免给中间文字版本重复绑定长门禁。
- `matcher/training/README.md` 与本清单的相对 Markdown 链接、`git diff --check`、精确变更范围和工作树缓存将在初次冻结哈希固定前再次核对。

本节只记录编写代理的初次冻结结果；内容正确性和严格技术表达仍待独立审查，不把机械自检通过写成独立 PASS，也不在审查前进入 E 批。

D 批初次冻结的独立内容审查判定 CONTENT FAIL，九项阻断均为可执行行为之外的技术表达：身份损失两类零值没有区分计算图连接；`class_weights` 缺 shape、dtype、设备、来源与本地未核验边界；跨批 Python float 聚合被过称为始终有限；候选预训练与完整 Matcher 的选模公式被混写；checkpoint 的 PID 临时路径、同进程并发和失败清理被过称；W&B `force` 与凭据来源被扩写成 SDK 未承诺语义；`log_global_step` 未披露缺少严格整数类型门禁；阶段 summary 被误当作远端 W&B 收口证明；梯度累积的代数等价没有限定相同局部损失、分母与精确算术。独立审查随后补充 `_finite_float` 的 `isinstance` 子类边界。

二次修订逐项收紧上述事实，并同步源码 Docstring、局部关键变量注释、训练 README 与本清单；未修改可执行语句、异常字符串、函数签名、常量或 YAML。二次冻结只重新执行 AST、换行、结构化条目、compile、pyflakes、CLI、链接、diff、范围和缓存快速门禁；初次冻结的 mypy、YAML 与 30 项测试保留为历史证据，不在中间内容版本重复运行。二次冻结仍待独立内容复审，不提前记录 PASS。

D 批二次冻结的最终窄复核判定 PASS，复核范围严格限于首审九项、同轮补充的 `_finite_float` 边界和机械门禁，没有再次扩展全量内容审查。独立复核确认九个源码去 Docstring AST 等价、LF-only、字段条目和无反引号；`pyflakes`、experiment 帮助、`git diff --check`、全库 45 个源码文件 mypy、34 份正式 YAML 加载、Markdown 链接、精确范围和缓存扫描均通过，训练相关四个测试文件得到 `98 passed in 221.27s`，冻结文件哈希在测试前后不变。D 批至此收口；后续批次继续采用一次全审、首审失败时至多一次点名位置定点复核。

## E 批首次冻结

E 批只修改 `matcher/evaluation/__init__.py`、`matcher/evaluation/metrics.py`、`matcher/evaluation/calibration.py`、[`matcher/evaluation/README.md`](../matcher/evaluation/README.md) 和本清单。源码改动仅包含模块、类与函数 Docstring 和局部学习注释；没有修改可执行语句、函数签名、常量、测试或正式 calibration 数据。`calibration.py` 的模块 Docstring 会进入 argparse 帮助说明，因此只允许命令说明正文变化，不宣称帮助文本逐字相同。

- 三个 Python 文件相对 `369152756818f04730efd5865ad8fdb18ed9b620` 去除模块、类和函数 Docstring 后，可执行 AST 全部相同。
- 三个 Python 文件和两个 Markdown 文件均为 LF-only 并以 LF 结尾；源码注释与 Docstring 没有反引号，模块、类、函数、函数参数和结构化字段已完成逐实体条目扫描。
- `python -m compileall -q matcher/evaluation`、`python -m pyflakes matcher/evaluation` 和 `python -m mypy matcher` 通过；mypy 检查 45 个源码文件，结果为零问题，所有缓存均写入系统临时目录。
- `python -m matcher.evaluation.calibration --help` 正常生成帮助并以状态码 0 退出，四个命令行选项仍然存在；参数解析的可执行 AST 未变。
- `python -m pytest matcher/tests/test_calibration.py matcher/tests/test_metrics_and_config.py -q -p no:cacheprovider` 得到 `38 passed in 9.51s`；两个测试文件保持只读。
- `matcher/evaluation/README.md` 与本清单的相对 Markdown 链接、`git diff --check`、精确范围和工作树缓存检查均通过；E 批没有修改任何测试文件，工作树内没有 Python、mypy 或 pytest 缓存。

本节只记录编写代理的首次冻结事实；内容正确性和严格技术表达待一次完整独立审查，不把机械门禁通过写成独立 PASS。若首审 FAIL，只修首审点名位置并执行一次窄复核，不再扩展第三轮审查。

E 批唯一一次完整独立审查判定 CONTENT FAIL；三份源码的可执行 AST、换行和冻结哈希已经通过，九类阻断均为技术表达：`run_calibration`、恢复 claim 与数据语义摘要没有在函数边界展开完整嵌套字段；“冻结数据”混淆科学方案固定与文件快照；路径门禁没有披露符号链接边界且 SHA-256 字符数被误写成位数；模型加载、`best_score` 与 Dataset 缓存预算被过称；阶段 summary 写入时序和本地收口证据不完整；`identity_eligible_count` 遗漏 `identity_scope`；严格 JSON 漏写损失正负无穷值；completed、防覆盖和一次性只在 claim 留存及合作锁边界成立；指标除法、混淆矩阵 dtype 和窄身份范围假阳性各有一处过称。

一次定点修订只处理上述九类点名位置：完整展开 15 个模型字段、训练与损失字段、六分类、三套当前范围身份对象、small/all、五个逐类身份对象和区域对象；逐字段补齐恢复 claim 与 manifest calibration 语义值；同步非快照、词法路径、阶段 summary、身份分母、非有限 JSON 与合作并发边界，并修正三处指标事实。二次冻结自检继续确认三个源码去 Docstring AST 等价、LF-only、结构化条目和无源码反引号；compile、pyflakes、calibration 帮助、全库 45 源码 mypy 均通过，两个相称测试文件得到 `38 passed in 12.77s`，测试文件保持只读。按用户上限，后续独立复核只检查首审九类点名项和机械门禁，不再扩展完整内容审查或第三轮修订。

二次冻结的最终独立窄复核判定 PASS，范围严格限于首审九类点名项和机械门禁，没有扩展第二次完整内容审查。独立复核确认完整 artifact、claim 与数据嵌套树，非快照、路径、缓存、阶段 summary、身份分母、非有限 JSON、合作锁和指标边界均已闭合；三个源码去 Docstring AST、换行、条目、链接、diff、范围、缓存、mypy、命令帮助和 38 项测试再次通过，五个冻结文件哈希在门禁前后保持不变。E 批至此收口。

## F 批首次冻结

F 批只修改 `matcher/__init__.py`、`matcher/utils/__init__.py`、`matcher/utils/cache.py`、`matcher/utils/json_artifacts.py`、`matcher/utils/locking.py`，新增 [`matcher/utils/README.md`](../matcher/utils/README.md)，并同步本清单。五个源码只改变模块、类与函数 Docstring 和局部学习注释；没有修改可执行语句、函数签名、常量、测试或服务器资源锁。

源码和 README 已逐项区分以下边界：`ByteBudgetLru` 只统计 `array_nbytes` 选定的 NumPy 载荷而非进程 RSS，插入后原地修改不会重算；`write_json` 使用 PID 命名同目录临时文件、文件 `fsync` 与 `os.replace`，`append_jsonl` 则没有临时替换或回滚，两者都没有内部并发锁；`exclusive_file_lock` 只串行使用同一路径协议的合作调用者，Windows 首字节锁和非 Windows `flock` 的等待与错误语义服从平台，且不等同于 Slurm 四锁状态机。

首次冻结前执行并通过的机械门禁如下：

- 五个 Python 文件相对 `369152756818f04730efd5865ad8fdb18ed9b620` 去除模块、类和函数 Docstring 后，可执行 AST 全部相同。
- 五个 Python 文件均为 LF-only 并以 LF 结尾；源码注释和 Docstring 没有反引号，模块、类、函数、函数参数和 `ByteBudgetLru` 三个真实实例字段已完成逐实体条目扫描。
- `python -m compileall -q matcher/__init__.py matcher/utils` 和 `python -m pyflakes matcher/__init__.py matcher/utils` 通过，Python 字节码写入系统临时目录。
- `python -m mypy matcher` 检查 45 个源码文件，结果为零问题，mypy 缓存写入系统临时目录。
- 直接覆盖 LRU、加锁失败、正文与解锁双重失败、严格 JSON/JSONL 的四个既有测试得到 `4 passed in 2.35s`；测试文件保持只读。
- 本批不涉及把模块 Docstring 传给 argparse 的四个 CLI 模块，不新增允许的帮助文本差异。
- [`matcher/utils/README.md`](../matcher/utils/README.md) 与本清单的相对 Markdown 链接、`git diff --check`、精确七路径范围和工作树缓存已在首次冻结哈希固定前核对并通过。

本节只记录编写代理的首次冻结事实；内容正确性和严格技术表达待一次完整独立审查，不把机械门禁通过写成独立 PASS。若首审 FAIL，只修首审点名位置并执行一次窄复核，不再扩展第三轮审查。

F 批唯一一次完整独立审查判定 CONTENT FAIL，但五个源码和 `matcher/utils/README.md` 的技术内容以及全部机械门禁均已通过。阻断只来自本清单的两个冻结状态错误：开头遗漏 F 批也相对同一基点完成 AST 比较，首次冻结段又把已经完成的链接、diff、范围和缓存核对写成未来动作。本次唯一一次定点修订只更正这两项并同步批次状态，形成 F 批二次冻结；最终独立复核只检查上述点名位置与相称机械门禁，不再扩展内容审查或启动第三轮修订。

F 批二次冻结的最终独立窄复核判定 PASS。独立复核确认两处状态文字和首审 FAIL 历史已经闭合，清单换行、相对链接与 `git diff --check` 通过；其余五个源码和 `matcher/utils/README.md` 没有再次变化，沿用唯一一次完整审查已经通过的内容、AST、静态检查、类型检查和四个定向测试证据。F 批至此收口。

## G 批取消

用户已取消 `matcher/tests/` 全部八个 Python 文件的学习注释工作。测试文件不纳入源码注释范围，不修改其注释、Docstring 或可执行内容；各批仍可把相称 pytest 作为行为门禁运行。

## 正式 YAML 清单

34 份文件都由 `matcher.training.config.load_experiment_plan` 读取，并共享以下顶层契约：`plan_name` 是运行目录名组成部分；`output_root` 是实验根目录；`tracking` 定义 W&B 生命周期；`data` 定义三类根目录、清单选择、裁剪与 DataLoader；`stages` 按列表顺序训练。每个 stage 都包含任务类型、语言家族、身份变体、身份范围、完整模型结构、完整训练字段和候选预训练初始化来源。H 批已经逐字段核对真实读取位置、`null` 语义、列表顺序和目录内对照依赖；34 份 YAML 的解析值以及剥除全行和行尾注释后的值行顺序均保持不变。

| 文件 | 当前实验职责与唯一主要变量 | 读取与验证 | 批次与状态 |
| --- | --- | --- | --- |
| `configs/experiments/initial/candidate_pretrain_pair.yaml` | 同一计划顺序训练 mean/max 与候选条件注意力两个候选预训练 stage；不使用语言身份监督 | 配置解析；initial 资源与早停契约测试 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/initial/full_scratch_molformer_detached.yaml` | MolFormer、detached、小分子身份范围、从零训练 | 配置解析；initial 矩阵测试 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/initial/full_scratch_molformer_joint.yaml` | MolFormer、joint、小分子身份范围、从零训练 | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/initial/full_scratch_molformer_oracle.yaml` | MolFormer、oracle、小分子身份范围、从零训练 | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/initial/full_scratch_smi_ted_light_289m_detached.yaml` | SMI-TED 289M、detached、小分子身份范围、从零训练 | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/initial/full_scratch_smi_ted_light_289m_joint.yaml` | SMI-TED 289M、joint、小分子身份范围、从零训练 | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/initial/full_scratch_smi_ted_light_289m_oracle.yaml` | SMI-TED 289M、oracle、小分子身份范围、从零训练 | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch1/full_smi_ted_all_foreground_detached_a800.yaml` | SMI-TED、all_non_background、detached；manual=true，sampling=false | Batch1 五类手工特征对照测试 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch1/full_smi_ted_all_foreground_joint_a100.yaml` | 同 Batch1 科学字段，identity_variant=joint | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch1/full_smi_ted_all_foreground_oracle_a800.yaml` | 同 Batch1 科学字段，identity_variant=oracle | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch2/candidate_v_centroid_cross_attention_a800.yaml` | 候选预训练；v_centroid；candidate_cross_attention；manual/sampling=true | Batch2 v_centroid 对照测试 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch2/candidate_v_centroid_mean_max_a800.yaml` | 候选预训练；v_centroid；mean_max；manual/sampling=true | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch2/full_smi_ted_v_centroid_detached_a100.yaml` | 完整 Matcher；v_centroid；detached | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch2/full_smi_ted_v_centroid_joint_a100.yaml` | 完整 Matcher；v_centroid；joint | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch2/full_smi_ted_v_centroid_oracle_a100.yaml` | 完整 Matcher；v_centroid；oracle | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch3/candidate_enhanced_cross_attention_a800.yaml` | 候选预训练；条件注意力并启用候选—token 注意力偏置 | Batch3 enhanced attention 测试 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch3/full_smi_ted_enhanced_cross_attention_detached_a100.yaml` | 完整 Matcher；enhanced attention；detached | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch3/full_smi_ted_enhanced_cross_attention_joint_a100.yaml` | 完整 Matcher；enhanced attention；joint | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch3/full_smi_ted_enhanced_cross_attention_oracle_a100.yaml` | 完整 Matcher；enhanced attention；oracle | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch4/full_smi_ted_v_centroid_pretrained_mean_max_detached_a800.yaml` | 从 Batch2 mean/max 候选 BEST 初始化；detached | Batch4 初始化来源与兼容测试 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch4/full_smi_ted_v_centroid_pretrained_mean_max_joint_a800.yaml` | 同一候选 BEST 初始化；joint | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch4/full_smi_ted_v_centroid_pretrained_mean_max_oracle_a100.yaml` | 同一候选 BEST 初始化；oracle | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch5/batch5_smi_ted_v_centroid_gate_002_oracle_a100.yaml` | oracle；graph/inventory 门初值均为 0.02；clip=1 | Batch5 只改变声明变量测试 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch5/batch5_smi_ted_v_centroid_graph_gate_002_oracle_a100.yaml` | oracle；仅 graph 门初值 0.02 | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch5/batch5_smi_ted_v_centroid_inventory_gate_002_oracle_a100.yaml` | oracle；仅 inventory 门初值 0.02 | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch5/batch5_smi_ted_v_centroid_gate_010_oracle_a100.yaml` | oracle；graph/inventory 门初值均为 0.10 | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch5/batch5_smi_ted_v_centroid_clip_5_oracle_a800.yaml` | oracle；零门；梯度裁剪阈值 5 | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch5/batch5_smi_ted_v_centroid_clip_200_oracle_a800.yaml` | oracle；零门；梯度裁剪阈值 200，作为近似不裁剪对照 | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/joint_identity_v4/joint_identity_v4_zero_gate_oracle_a100.yaml` | v4 联合身份；seed 3407；LR=1e-4；零门 | v3/v4 等价与 v4 LR 三对测试 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/joint_identity_v4/joint_identity_v4_lr_2e4_oracle_a100.yaml` | 与 seed 3407 基线配对；唯一科学变量为 LR=2e-4 | v4 LR 三对测试 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/joint_identity_v4/joint_identity_v4_seed_3408_lr_1e4_oracle_a100.yaml` | seed 3408、LR=1e-4 配对基线 | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/joint_identity_v4/joint_identity_v4_seed_3408_lr_2e4_oracle_a100.yaml` | seed 3408、LR=2e-4 | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/joint_identity_v4/joint_identity_v4_seed_3409_lr_1e4_oracle_a100.yaml` | seed 3409、LR=1e-4 配对基线 | 同上 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/joint_identity_v4/joint_identity_v4_seed_3409_lr_2e4_oracle_a100.yaml` | seed 3409、LR=2e-4 | 同上 | H；二次冻结最终窄复核 PASS |

## H 批首次冻结

H 批只修改前表逐路径列出的 34 份正式 YAML、现有
`configs/experiments/initial`、`batch1` 至 `batch5`、`joint_identity_v4`
七份 README，并同步本清单，共 42 个路径。本批没有新建
`configs/experiments/README.md`，没有修改 Python、测试、YAML 值、服务器资源或 calibration。

每份 YAML 都新增同一套逐字段契约，覆盖顶层五字段、tracking 七字段、data 十一字段、
stage 八字段、model 十五字段、training 十二字段及 loss_weights 四个嵌套字段；
各文件原有开头注释继续说明本次实验的具体变量。七份 README 在保留批次矩阵、历史结果和
validation 结论的前提下，补充正式解析入口、完整字段表、特殊值、代码读取位置和目录内依赖。
同时依据已经通过 A–F 批审查的源码事实，收紧 W&B 凭据解析、Dataset 实例缓存预算、
PreparedLigand 顶层 smiles 原文回退和跨 GPU 对照边界；一次性服务器审计路径不再留在正式
YAML 或配置 README，相关证据仍应由实验执行记录保存。

首次冻结前的自检结果如下：

- 逐文件比较基点 YAML 与当前文件，剥除全行和行尾注释后的 34 组非空值行顺序和值文字全部相同；27 份 YAML 的 `cache_gib_per_worker` 原值行只改写了既有行尾注释，字段值和解析对象没有改变；
- PyYAML `safe_load` 的 34 组对象全部相等，当前
  `matcher.training.config.load_experiment_plan` 也成功解析全部 34 份计划；
- 从 YAML 对象展开得到的 63 个字段路径都能在每份 YAML 注释和七份 README 中逐项定位；
- 34 份 YAML、七份 README 与本清单均为 LF 且保留末尾换行，Markdown 相对链接和
  `git diff --check` 通过；
- 未修改 `matcher/tests/`，只把
  `matcher/tests/test_metrics_and_config.py` 作为行为门禁运行，结果为
  `37 passed in 7.87s`；
- 工作树未暂存、未提交；最终扫描发现六个导入缓存目录后，已把这些精确
  `matcher/**/__pycache__` 目录移到系统临时目录，冻结态工作树不含缓存；没有访问服务器或
  calibration。

按用户为后续批次设定的审查上限，本冻结态只接受一次完整独立审查；若结论为 FAIL，只修订
首审集中清单点名的位置，并接受一次窄复核，不再启动第三轮内容审查。

H 批唯一一次完整独立审查判定 CONTENT/STYLE FAIL；机械行为门禁已经通过，七类阻断都属于配置注释与 README 的技术表达：约 63 行通用字段说明集中在文件头，没有紧邻对应字段组与字段；候选预训练被误写成只训练三个输出头，而实际由六分类和两个区域损失优化整套 `CandidateEncoder`；通用身份措辞把历史 schema v3 配置也无条件解释成联合身份；`physical_batch_size` 没有区分训练满批与 validation 末批；`voxel_feature_channels` 被误读成体素头输出宽度；七份 README 把不存在的 `plan` 写成顶层键；冻结记录没有说明 27 份 YAML 的缓存预算值行改写过行尾注释。

唯一一次定点修订只处理上述七类位置。34 份 YAML 保留简短文件职责头，把 `tracking`、`data`、`stages[*]`、`model`、`training` 和 `loss_weights` 的类型、特殊值与消费者说明移到实际组或字段旁；行内映射的四个损失权重也在同一值行前逐字段说明。候选预训练现明确由三项损失优化整套候选编码器且不构造身份分支或身份损失；通用身份文字改为由执行代码 schema 决定的闭集身份，只有 `joint_identity_v4` 目录的任务专属说明继续明确 schema v4。训练物理批量、validation 末批、体素头输入与隐藏宽度、单 logit 输出和真实 YAML 顶层映射已同步到 34 份 YAML 与七份 README。

二次冻结机械门禁确认：34 份 YAML 的 PyYAML `safe_load` 对象和正式 `load_experiment_plan` 对象均与基点相等；剥除全行及行尾注释后的值行顺序与文字 34/34 相等；每份计划展开得到的 63 个字段路径均能在本文件注释中定位。27 份 `cache_gib_per_worker` 原值行只改写既有行尾注释，字段值未变。`matcher/tests/test_metrics_and_config.py` 在同一修订态得到 `37 passed in 13.68s`；测试代码保持只读。二次冻结仍待独立最终窄复核，复核范围只包含首审七类点名项和相称机械门禁，不再扩展完整内容审查。

H 批二次冻结的最终独立窄复核判定 PASS。独立复核只检查首审七类点名项和相称机械门禁，没有再次扩展完整内容审查；字段邻接、候选预训练、schema 中性身份、训练与 validation 批量、体素头宽度、YAML 顶层映射和 27 处行尾缓存注释记录均已闭合。复核态继续满足 34/34 PyYAML 与正式加载器基点等价、34/34 去注释值行等价、34/34 每份 63 字段覆盖、42 路径字节与链接检查、`git diff --check`、精确范围、无暂存、测试只读和工作树无缓存；同一测试文件独立得到 `37 passed in 13.75s`。H 批至此收口。

## README 与学习概览

| 文档 | 当前职责或待建职责 | 覆盖批次与状态 |
| --- | --- | --- |
| `matcher/README.md` | 本轮不创建；用户要求的冷读入口由下列六个正式子包 README 分别承担 | I 不另立改写批次 |
| `matcher/data/README.md` | 已建立来源文件、manifest、联合身份、候选体素、PP token 与 LigandObject 图的冷读契约 | A；独立终审 PASS |
| `matcher/datasets/README.md` | 已建立 `CandidateItem`、`CandidateBatch`、缓存分账、区域目标、联合身份、图连接和 DataLoader 的字段与对齐契约 | B；第五次冻结独立终审 PASS |
| `matcher/model/README.md` | 已建立模型输入、三种聚合路径、RAUNet 回收、联合身份三分支、门控与输出的冷读契约 | C；既有内容三次冻结独立终审 PASS；concat 增量随最终整合核验 |
| `matcher/training/README.md` | 已建立正式 YAML、训练阶段、损失、初始化、checkpoint、W&B 和运行产物的字段级契约 | D；二次冻结定点终审 PASS |
| `matcher/evaluation/README.md` | 已建立 validation 指标、selection score、一次性 calibration 输入证据、claim、恢复与失败边界 | E；二次冻结最终窄复核 PASS |
| `matcher/utils/README.md` | 已建立缓存计量、严格 JSON/JSONL、合作文件锁和根包稳定导出的冷读契约 | F；二次冻结最终窄复核 PASS |
| `configs/experiments/initial/README.md` | 首轮七项计划、YAML 锚点复用、完整字段与运行产物边界 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch1/README.md` | 五类身份变体矩阵、手动特征、身份指标与完整 YAML 字段 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch2/README.md` | V 质心视野、采样统计、候选/完整任务矩阵与完整 YAML 字段 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch3/README.md` | 候选—token 注意力偏置、梯度路径、对照依赖与完整 YAML 字段 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch4/README.md` | 候选 checkpoint 初始化来源、硬件对照边界与完整 YAML 字段 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/batch5/README.md` | 门初值/梯度裁剪矩阵、历史 v3 边界与完整 YAML 字段 | H；二次冻结最终窄复核 PASS |
| `configs/experiments/joint_identity_v4/README.md` | 联合身份、三组 LR 配对、摘要拼接对照、顶层 smiles 回退与完整 YAML 字段 | H 的既有字段契约二次冻结最终窄复核 PASS；后增实验事实随最终整合核验 |
| `talk/Matcher_输入与产物概览.md` | 保持累计基点内容；正式 v4 字段以 `matcher/data/README.md` 与 `matcher/datasets/README.md` 为当前冷读入口 | 未纳入本轮 85 项注释资产，不另立 I 批改写 |
| `talk/Matcher_模块结构概览.md` | 保持累计基点内容；模块职责以六个正式子包 README 为当前冷读入口 | 未纳入本轮 85 项注释资产，不另立 I 批改写 |

## 明确不在本轮注释范围内的内容

- `matcher/data/manifest.py`、`matcher/data/precompute.py`、`matcher/training/experiment.py` 和 `matcher/evaluation/calibration.py` 的模块 Docstring 会被 argparse 用作命令帮助说明；本轮允许这四个命令的 `--help` 说明文字随 Docstring 改写，但参数名称、默认值、解析行为和退出语义必须保持相同并重新生成帮助文本验证。
- `文档/记录/`、`文档/exec_plan/` 和服务器运行证据由训练与分析主线维护，不复制到源码 Docstring。
- 细粒度注释工作树没有修改 `训练与运行/`；历史批次的 21 个薄启动脚本和批次 README 已由 `f6a3ecb` 进入当前累计基点，并在最终整合树保留。
- calibration 的真实数据、输出和指标不读取；这里只解释代码可证明的接口、claim 状态和测试夹具。
- 发现源码行为、旧注释或文档互相冲突时，先记录事实差异并通知主任务；不在学习注释分支顺带修复可执行逻辑。

## 基点现有 README 的具名覆盖边界

下列基点现有 README 需要逐路径说明本工作树是覆盖还是排除，不能用“README 文档”概括：

- `README.md`：仓库总览由训练、实验状态和最终整理共同维护，待所有科学实验收口后再统一核对。
- `configs/experiments/initial/README.md`、`configs/experiments/batch1/README.md`、`configs/experiments/batch2/README.md`、`configs/experiments/batch3/README.md`、`configs/experiments/batch4/README.md`、`configs/experiments/batch5/README.md`、`configs/experiments/joint_identity_v4/README.md`：H 批补充目录内 YAML 的字段、依赖与冷读契约，保留既有批次历史结论、实验结果和运行事实，不改写为新的运行记录。
- `与服务器交互/README.md`：属于服务器操作纪律；本任务不访问服务器。
- `训练与运行/README.md`、`训练与运行/sh/README.md`：属于 release、launch、锁和正式脚本说明；由 `Matcher_batch_launch_layout` 隔离工作树维护。
- `文档/记录/单任务记录/第四批次/README.md`、`文档/记录/单任务记录/第五批次/README.md`、`文档/记录/批次记录/联合身份v4/README.md`：属于实验执行记录，由训练与分析主线维护。

细粒度注释资产中的文档严格为六个正式子包 README、七份
`configs/experiments/*/README.md` 与本清单 `talk/Matcher_全库细粒度注释清单.md`；没有创建
`matcher/README.md`，也没有改写两份既有 `talk/` 概览。测试包不生成重复字段文档。

## 最终整合映射证据

- 细粒度工作树共有 85 项资产：37 个 Python 文件、34 份 YAML、7 份配置 README、6 份子包 README 和本清单；测试文件保持只读。
- 其中 77 个路径在最终整合工作树逐字节相等；6 个实现或配置交叉路径为 `configs/experiments/initial/README.md`、`configs/experiments/joint_identity_v4/README.md`、`matcher/model/candidate_encoder.py`、`matcher/model/config.py`、`matcher/model/fusion.py` 和 `matcher/training/initialization.py`。其余 2 个路径是 `matcher/model/README.md` 与本清单，它们分别按最新 concat 冷读契约和最终整合 Git 事实更新，不再与注释工作树冻结字节相等。
- 四个交叉 Python 文件去 Docstring 后的 AST 与累计基点 `2f80bc9880591c69c1e51f7125bb4f989ab22b9f` 相等；这证明最新 concat 可执行逻辑没有被旧注释工作树覆盖。累计基点后增的 concat 聚合类已在整合树补充实例字段、拼接向量、残差变量与构造函数契约，并通过唯一完整整合审查的点名修订。
- H 批 PASS 只覆盖当时的 34 份 YAML；当前累计树共有 46 份正式 YAML，后增的 12 份 concat/mean-max 配置保留各自科学配置测试，但不冒充 H 批逐字段注释审查对象。
- 85 项资产已经提交到实现历史，并按六个阅读层次重建学习历史。最终治理记录提交前，两端树对象同为 `7181cc3a181390917a97bdf097deea7ba53747d0`，测试目录无差异；`Learn/CUMULATIVE` 已推进，17 个 worktree 已清理为唯一主工作树。最后一次窄复核只剩映射计数这一机械问题，主代理按用户限制完成 `77 + 6 + 2 = 85` 复算，没有开启第三轮独立审查。

## 最终集成与工作树清理门禁

注释批次、concat_summary 全部科学实验与双线 Git 整理完成后，必须依次满足下列门禁，不能为减少工作树数量提前删除隔离资产：

- 先把每个隔离工作树的预期成果安全整合到最终学习历史，并重新证明最终实现端点与学习端点的可执行内容等价；
- 再逐个核对主工作树与全部隔离工作树的分支归属、追踪文件、未追踪文件和未提交修改，确认没有只存在于待移除工作树中的资产；
- 只有上述核对全部通过后，才移除注释、批次脚本、实验配置等隔离 worktree；最终只保留一个主工作树；
- 按提交者时间检查全仓库提交拓扑，只允许一个唯一最新提交，且最终 `Learn/CUMULATIVE` 必须指向该提交；
- concat_summary 科学实验从当前裁定起最多再增加五项；这些任务完成后不再扩张实验矩阵，主线转入结果汇总、报告、注释与 Git 收口；
- 最终在 `训练与运行/sh/最优变体推荐/` 交付 2–5 个从头训练脚本，并逐脚本交叉核对文件头提交命令、同目录 README 和引用 YAML；模型、数据、损失、随机种子、门初值、梯度裁剪、批量、checkpoint 初始化与评估口径必须在三处表达一致，不把 SHA-256、release 或 launch 取证写入正式脚本和 YAML；
- 移除工作树与推进 `Learn/CUMULATIVE` 后，再执行工作树列表、分支指向、仓库状态、完整门禁和端点等价复核，把结果写入最终收口记录。

上述门禁已经完成：外置恢复包保留源资产，16 个辅助 worktree 已按精确路径移除，主工作树 ignored `tmp/` 保持原位；最终只剩 `C:/Users/15919/Desktop/Matcher`，并由 `Learn/CUMULATIVE` 管理。
