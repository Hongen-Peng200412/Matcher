# Matcher 精简重整、四模型训练与正式评估计划

## 1. 文档状态与实施边界

本文给出 Matcher 从当前并行实现回归单一成熟主干的代码改造、训练和评估方案。方案依据 2026-09-16 的本地仓库与服务器只读核查结果编写。

当前阶段只允许形成计划，不修改 Python、YAML、Shell 脚本、Git 历史或服务器文件。开始实施仍需用户另行明确批准。实施后也不保留“旧管线”和“新管线”两套可运行入口；经门禁证明新主干完整承接有效能力后，重复实现和过时功能直接删除，由 Git 历史保存旧实现。

本轮最终目标不是重写第三套 Matcher，而是完成以下闭环：

1. 将已经验证的精确 SMILES 身份、SMILES 化学图和 Find_1 数据适配迁入现有的 `matcher/data`、`matcher/datasets`、`matcher/model`、`matcher/training`、`matcher/evaluation` 分层结构。
2. 删除 `matcher/smiles_identity` 形成的并行数据、模型、损失和训练实现，并删除已迁移测试所在的并行测试目录。
3. 把 Matcher 的科学任务收敛为相互分开的两个问题：候选是否为真实前景，以及真实前景候选对应哪个精确 SMILES。
4. 使用同一冻结训练清单训练 full、blob-only、pocket-only、trunk-only 四个模型。
5. 以完整 `held_out_test_0` 为主要测试范围，分别在真实受体与 CryoAtom2 受体产物上完成统一评估和规定图表。

## 2. 已核实事实与信息来源

### 2.1 本地代码与 Git

- 历史参考点是提交 `9e6e974dd98cba236290fcf59966ff2fb85cd49e`。它用于判断哪些成熟实现值得保留，不是实施时回退 Git 的目标。
- 2026-09-16 核查时，`Learn/CUMULATIVE` 指向 `02572559e9e530024335c8adc151d0ce050dcfab`，是仓库按提交者时间形成的唯一最新提交；仓库只有一个工作树。
- 从历史参考点到当前端点增加约 7019 行、删除约 22 行。新增主体集中在 `matcher/smiles_identity`，而原有 `matcher/data`、`matcher/datasets`、`matcher/model`、`matcher/training`、`matcher/evaluation` 基本仍在。因此当前问题确实是两套实现并存，而不是单一主干已经完成迁移。
- 当前同时存在 `matcher/data/manifest.py` 与 `matcher/smiles_identity/manifest.py`；Dataset、批处理、模型、损失和训练引擎也有对应重复实现。
- `matcher/data/labels.py` 已有用户 NOTE，明确指出真实配体编号不应继续命名为 `candidate_id`，并要求标签生成只有一套规则。新实现中应统一使用 `occurrence_id`，Stage1 候选只使用 `source_blob_index` 和 `centered_box_index`。
- 历史最佳设置来自 `configs/experiments/joint_identity_v4/joint_identity_v4_concat_zero_gate_clip_200_seed_3409_lr_1e4_oracle_a100.yaml` 及 `文档/记录` 中相应结果。可继承的主要经验包括摘要拼接、模型宽度、RAUNet、优化器、学习率、批量、随机种子、零门初始化和宽松梯度裁剪；旧身份字段、旧六分类评估和旧输出路径不继承。

### 2.2 Find_1 产物与清单范围

正式数据来源是以下两个 Find_1 根目录：

- 真实受体：`/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1`
- CryoAtom2 受体：`/storage/penghongen/AdaLigand_stage1_inference/Find_1/CryoAtom2受体/artifacts/Find_1`

Find_1 正式训练划分规划为 13,717 个 PDB、50 个确定性分片。2026-09-16 核查时，服务器训练目录已有 1180 个 PDB 目录，其中前 4 片共 1100 个 PDB 具有完整 centered 产物，第 5 片仍在生成。这个数量只是核查时点，不构成 Matcher 等待全部 50 片的条件。

用户已经确定 Matcher 只需要两个清单函数：

1. `build_manifest(...)`：扫描指定 Find_1 文件夹中当时已经完整的产物并冻结清单。
2. `load_manifest(...)`：读取并核验冻结清单，供 Dataset、训练和评估使用。

因此正式四模型训练不等待 50 片全部完成。实施时选定一个冻结时点，四个模型共同绑定同一清单内容和 SHA-256。冻结后新到达的 Stage1 产物不会自动进入既有实验；若以后确需扩充训练数据，应显式构建另一份清单并视为另一批实验。

当前 `/storage/penghongen/tmp/Matcher/manifests/smiles_identity/manifest.json` 只作为已经完成的 validation 接入证据：它覆盖 200 个 PDB、7403 个保留候选、3619 个前景候选、603 个 PDB 内身份槽和 208 个跨 PDB 唯一 SMILES，并记录 10 个因图或语言资产不可用而跳过的前景候选。该文件位于 tmp，不能直接升格为正式四模型清单；新主干必须在正式根目录重新构建并绑定来源。

### 2.3 测试范围与现有评估资产

- 真实受体和 CryoAtom2 受体的 `held_out_test_0` 都包含相同的 179 个 PDB，是正式主要测试范围。
- `held_out_test_1` 含 149 个 PDB，是 `held_out_test_0` 按 occurrence 数过滤得到的保序子集，不是独立测试集。它只产生补充结果，并从同一次 `held_out_test_0` 推理结果中过滤，不重复前向、不冒充第二个测试集。
- 两种受体条件的 `held_out_test_0` 已有 `evaluation/f2_centered_gaussian_macro_selected.npz`。该文件提供 Stage1 分数、`candidate_selected`、`source_blob_index`、双向覆盖统计和匹配字段；测试 centered 文件本身不一定包含分数与 selected 字段。新代码应在读取时连接两个只读产物，不得为了补字段改写 Stage1 文件。
- 主科学口径使用 0.30 双向覆盖阈值。0.50、0.60 可作为补充表格，但不增加正式训练任务，也不替换 0.30 主图。

在 0.30 阈值并只保留 Stage1 selected 候选时，核查结果为：

| 受体条件 | selected 候选 | 真实前景 | 背景 | 金属/离子 | 肽 | 小分子 | 糖 | 核酸状 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 真实受体 | 1653 | 1240 | 413 | 564 | 1 | 391 | 284 | 0 |
| CryoAtom2 | 1651 | 1162 | 489 | 534 | 1 | 350 | 277 | 0 |

因此正式条形图必须把核酸状配体显示为 `N/A (support=0)`；肽类必须标注 `support=1`，不得把单一样本比例描述为稳定估计。

### 2.4 两个不具备受支持身份的测试 PDB

测试清单中确有两个 PDB 包含无法建立合法 SMILES 图且缺少 SMI-TED 语言资产的 small-molecule occurrence，并非统计误入：

- `9ian`：occurrence 0、5；occurrence 0 在真实受体中为 selected rank 3，在 CryoAtom2 中为 selected rank 5。
- `9oq7`：occurrence 4、10、13；occurrence 10 在真实受体中为 selected rank 2，在 CryoAtom2 中为 selected rank 1。

这两个 PDB 同时属于 `held_out_test_0` 和 `held_out_test_1`。最终处理规则已经确定：

- 保留 PDB、候选、Stage1 分数、selected 状态和原始名次。
- 假阳性二分类照常评价，因为几何真值仍能说明候选是不是前景。
- 将相关候选标记为“不具备受支持身份”，不放入 conditional SMILES 准确率分母。
- Stage1 top-k 曲线可以因该候选命中真实前景；Stage1+Matcher 曲线不能由该候选形成命中。
- 不删除候选，不让后一名补位，不重新排列名次。

### 2.5 SMILES 图资产

现有正式资产为：

`/storage/penghongen/AdaLigand/Ori_Data/smiles_assets/smiles_graphs_v1.npz`

它以精确的顶层 `PreparedLigand.smiles` 为唯一逻辑键，集中保存重原子和化学键字段。已核实包含 3497 个合法 SMILES；另有 91 个输入被严格构图排除。当前 `element` 字段是原子序数，并由 `nn.Embedding(128, model_dim)` 当作离散原子类别使用，因此“配体原子类别”已经有明确实现。此次四模型实验不再增加一个重复的 `atom_type` 字段，也不擅自加入杂化、芳香性等新的科学变量。

Matcher 图仍只保留由 SMILES 唯一决定的字段：原子序数、形式电荷、原子环标志、化学键端点、键型和化学键环标志。坐标、参考构象键长、residue 编号、模板 present 掩码和模板手性不回到 Matcher 图契约。

## 3. 最终科学契约

### 3.1 身份定义

Matcher 的唯一监督身份是精确的顶层 `PreparedLigand.smiles`，即交给 SMI-TED 公开接口的 canonical non-isomeric SMILES。不得使用 `object_key`、occurrence 坐标、`present` 掩码、沉积原子选择或语言模型内部的 `model_smiles` 定义身份。

同一个 PDB 内，精确 SMILES 相同的 occurrence 共享一个身份槽；SMILES 不同则必须属于不同身份槽。`object_key` 不进入清单、Dataset、Batch、模型输入、checkpoint 或评估输出。每个 PDB 的身份候选集合仍由该 PDB 中具备图和语言资产的真实 occurrence 产生，Matcher 不在不同 PDB 之间做全库分类。

因此本轮仍是 per-PDB oracle closed-set：给定该 PDB 的受支持精确 SMILES 集合后，Matcher 判断哪个 SMILES 与当前候选相匹配。正式报告必须明确披露候选 SMILES 集合来自真实 occurrence，不能将结果描述为从任意化学空间发现未知配体。把真实身份集合替换为外部候选库属于另一项科学任务，不混入本轮四模型实验。

### 3.2 两个独立任务

模型同时回答两个问题，但评估时不把它们重新组合成旧的“端到端身份 F1”：

1. 前景判断：当前 Stage1 候选是不是符合现有科学定义的真实前景。输出一个前景 logit，使用二元交叉熵训练，以 PRAUC 和 F1 评价。
2. 条件身份匹配：在真实前景且身份资产受支持的候选上，从当前 PDB 的精确 SMILES 槽中选择正确 SMILES。使用交叉熵训练，以 Top-1 准确率评价。

两项任务可以共享候选编码器，但前景概率不得乘入或裁剪身份 logits，不得门控身份损失，也不得决定是否返回 SMILES。旧 PairScorer 中依赖身份类别概率的耦合输入应删除；真实类别只留在标签侧，用于按五类汇总 conditional 结果。

五种配体类别只作为 conditional Top-1 的真值分层：small_molecule、sugar、ion、peptide_like、nucleotide_like。模型不再预测旧六分类，`other` 仍按现有定义属于背景。旧六分类混淆矩阵、宏 F1、seen/unseen、intrinsic/end-to-end 身份族、identity scope 和 calibration claim 全部退出正式契约。

### 3.3 候选范围开关

配置只提供一个统一开关：

`data.selected_candidates_only: bool`

- `true`：只消费 `candidate_selected=true` 的候选。
- `false`：消费相应 Find_1 目录下的全部 centered 候选。

同一次实验的 train、validation 和 test 使用同一值，不设计三个划分各自独立的开关。正式四模型训练与正式测试固定为 `true`，以免在四个变体之间引入候选范围差异。`false` 只用于机械核查或未来另立实验。

### 3.4 缺失身份资产规则

- train/validation：若候选在 0.30 双向覆盖下匹配到真实前景 occurrence，但其精确 SMILES 没有合法图或 SMI-TED 向量，则整条候选跳过；不回退到 `object_key`，不改成背景，不重新分配到另一个 occurrence。背景候选不需要身份资产，仍保留。
- test：候选始终保留以维持 Stage1 排名和前景二分类口径；缺图或缺语言时设置 `identity_supported=false`，只退出 conditional 身份分母和 Stage1+Matcher 命中判断。

候选与 occurrence 的匹配规则只实现一次：两向覆盖率都不低于 0.30 后，最大化两者乘积的平方根；并列时选择较小的 `occurrence_id`。标签生成、训练清单和测试 top-k 必须调用同一实现，不能各自复制一套近似规则。

## 4. 清单与 I/O 方案

### 4.1 清单的最小公开接口

`matcher/data/manifest.py` 最终只公开以下两个业务函数：

- `build_manifest(...)`：扫描指定 Find_1 划分目录，只接纳完成标记和必需文件齐全的 PDB，按 PDB 编号与候选编号确定性排序，原子发布冻结清单及摘要。
- `load_manifest(...)`：一次性读取清单、核验来源和字段，返回 Dataset 可直接消费的内存结构。

不再保留 `ensure_manifest`、`require_manifest`、增量 extend、v3/v4 分支、seen/unseen 生成器或多套候选/身份清单入口。标签旁路文件可由一个职责单一的命令行脚本生成，但它不扩张清单的公开业务 API。

标签脚本沿用用户指定的简单形式：输入一个 Find_1 centered 目录，在同目录生成 `F2_centered_label.npz`；只提供“已存在则跳过”和“明确覆盖”两种模式。manifest 构造器只读并核验该文件，不在扫描过程中隐式重算或改写标签。

### 4.2 完整产物判定与冻结

一个 PDB 只有同时满足以下条件才进入对应清单：

1. Find_1 已写出完整完成证据，而不是仅存在目录。
2. 必需的 `F2_centered.npz` 数组完整且候选轴一致。
3. Stage1 分数和 selected 状态可读取；测试划分允许从现有 evaluation NPZ 连接。
4. 标签旁路文件与 `centered_box_index`、`source_blob_index` 逐项对齐。
5. 原始 occurrence 与 `PreparedLigand.smiles` 可连接。

清单保存全部候选记录及 `candidate_selected`，由 Dataset 的单一开关决定是否过滤。这样无需为 selected 与 all 复制两份清单。每次构建都记录输入根目录、PDB 数、候选数、前景数、受支持身份数、跳过原因计数和内容 SHA-256。

### 4.3 图与体素读取

- `smiles_graphs_v1.npz` 由每个持久化 DataLoader worker 在初始化时完整载入一次；单个训练样本不得反复打开 NPZ，也不得现场调用 RDKit 解析 SMILES。
- 继续按精确 SMILES 读取 SMI-TED 向量，不创建 `graph_id`，不恢复 `object_key`。
- Find_1 提供的 A、P、V、PP 及 centered 几何是候选输入来源。106 通道局部体素输入在 worker 内存中按需构造，不预先落盘 100 维以上的输入体素。
- 使用 PDB 级字节预算 LRU，缓存确实重复读取的只读数组；不为每个细小中间量再建立独立缓存层。
- 正式 I/O 门禁比较首批耗时、预热后吞吐、文件打开次数和单 worker 内存。若预热吞吐明显恶化，先修正 worker 生命周期或重复解压，不以恢复 LigandObject 身份解决性能问题。

I/O 门禁可以在相同候选顺序、相同 batch 和相同 GPU 前向下短测三种生命周期，而不把它们变成三项科学训练：压缩图包按 worker 一次性载入、未压缩图包按 worker 一次性载入、以及关闭 persistent worker 的反例对照。若 PDB 数组仍是主要瓶颈，再比较两个明确的 PDB-LRU 字节预算。最终只保留预热吞吐最好且内存可接受的一套正式实现，删除试验性分支。

既有 validation I/O 验收已经证明图包每 worker 首次只执行一次 `numpy.load`，预热后不再打开图包；既有乱序训练前缀测得数据加载约 5.95 samples/s、训练计算约 10.19 samples/s。它们只作为重构后的回归参考，不能与不同 GPU、worker 数或缓存状态的结果直接作科学比较。

### 4.4 正式落盘位置

建议统一使用：

`/storage/penghongen/AdaLigand_stage2_matcher/`

其下只分为 `manifests/`、`experiments/`、`evaluations/` 等稳定职责目录。正式清单、训练、checkpoint、预测和评估产物不得写入 `/storage/penghongen/tmp`。release 与 launch 留证仍遵循服务器交互规则，放在 `/home/penghongen/Feedback/Matcher/` 的对应目录；一次性 SHA 或核验脚本只进入 ignored ops/tmp 与执行记录，不进入正式 YAML 和 resolved 配置。

## 5. 代码布局与删除清单

### 5.1 迁入成熟分层目录

| 目录 | 最终职责 |
| --- | --- |
| `matcher/data` | Find_1 标签连接、唯一 manifest 实现、SMILES 图读取、密度与体素构造 |
| `matcher/datasets` | Dataset 返回契约、Batch 契约、collate、sampler、worker 初始化 |
| `matcher/model` | 候选编码、RAUNet、摘要融合、精确 SMILES 身份编码与打分 |
| `matcher/training` | 精简配置、损失、训练循环、BEST/LAST、调度器、W&B、checkpoint |
| `matcher/evaluation` | 二分类与 conditional 指标、测试 top-k、结果导出与两类图表 |
| `matcher/tests` | 迁入仍有价值的单元测试和契约测试；测试代码无需按业务代码密度补学习型注释 |

实施时逐项比较成熟实现和 `matcher/smiles_identity` 实现，只迁移后者已经验证且本计划仍需要的逻辑。不得把整个文件换一个目录名后继续保留两套抽象。

文件级目标布局如下；实施中只有在职责确实完整时才允许合并文件，不能继续细分大量一行包装函数：

| 目标文件 | 主要职责与来源 |
| --- | --- |
| `matcher/data/schema.py` | 只保留类别名、输入宽度等固定常量；删除 v3/v4、语言家族和身份范围枚举 |
| `matcher/data/labels.py` | 唯一候选—occurrence 几何匹配纯函数，统一 `occurrence_id` 命名 |
| `matcher/data/smiles_graph.py` | 迁入全局图包读取与 worker 初始化；替代 LigandObject 图读取 |
| `matcher/data/manifest.py` | 唯一的 `build_manifest` 与 `load_manifest`；连接 centered、标签、PreparedLigand、SMI-TED 和测试 evaluation NPZ |
| `matcher/datasets/contracts.py` | `CandidateItem`、`CandidateBatch` 的形状、掩码和特殊值契约 |
| `matcher/datasets/candidate_data.py` | Find_1 候选物化、现场 106 通道体素、A/P/V/PP token、PDB-LRU 和 selected 过滤 |
| `matcher/datasets/batching.py` | 物理批处理、padding、图连接、持久 worker 与确定性 sampler |
| `matcher/model/config.py` | 两个结构枚举及共同模型参数，不保留任意布尔组合实验面板 |
| `matcher/model/candidate_encoder.py` | 三种候选输入结构、单一前景 logit、可选 RAUNet 与区域头 |
| `matcher/model/fusion.py` | 历史 top1 的 mean/max 与候选条件注意力摘要拼接 |
| `matcher/model/ligand_matcher.py` | SMI-TED 主干配对、可选 SMILES 图与其他 SMILES 上下文；不消费类别概率 |
| `matcher/model/raunet.py` | 保留成熟三维主干，除适配构造接口外不重写网络 |
| `matcher/training/config.py` | 数据、模型、优化、追踪的最小配置与交叉字段核验 |
| `matcher/training/losses.py` | 前景 BCE、身份 CE、实际存在时的两个区域损失 |
| `matcher/training/engine.py` | epoch、validation、selection、阈值、早停和 BEST/LAST |
| `matcher/training/checkpointing.py`、`tracking.py`、`numeric.py` | 保留成熟原子发布、W&B 与有限数失败关闭，删除旧指标展开 |
| `matcher/training/experiment.py` | 单一薄训练入口 |
| `matcher/evaluation/metrics.py` | PRAUC、F1、conditional Top-1 与 support |
| `matcher/evaluation/test.py` | 连接两种受体测试产物、一次前向、逐 PDB top-k 和机器可读输出 |
| `matcher/evaluation/plots.py` | 仅生成规定的七组条形图和四曲线图 |

### 5.2 明确删除或退场

通过迁移门禁后删除：

- 整个 `matcher/smiles_identity/`。
- 将有效测试迁入 `matcher/tests/` 后删除 `matcher/tests_smiles/`。
- `matcher/data/ligand_graph.py` 中 LigandObject 运行时读取逻辑，由单一 SMILES 图读取模块替代。
- `matcher/evaluation/calibration.py` 及其 claim、租约和恢复逻辑；本轮没有 calibration split。
- seen/unseen、v3/v4 schema 分支、`identity_variant`、`identity_scope`、旧六分类预测与指标、intrinsic/end-to-end 身份指标。
- 已无调用者的 `voxel_input.py`、`pp_input.py`、`precompute.py` 和重复几何函数；是否删除必须先由导入图和测试证明无调用，不按文件名盲删。
- 活跃配置和启动目录中已失效的历史实验配置。历史事实由 Git 与结果文档保存，不再建立一个代码内 archive 目录。

最终应满足：仓库中只有一个 manifest 实现、一个 CandidateDataset、一个 Batch 契约、一个 Matcher 模型入口、一个训练引擎和一个正式评估入口；运行时代码中没有 `matcher.smiles_identity` 导入，也没有 `object_key` 身份依赖。

## 6. 模型与四个正式变体

历史 top1 的共同设置作为四个变体的默认基线：摘要采用 mean/max 与候选条件注意力拼接，`model_dim=256`、8 个注意力头、dropout 0.10、RAUNet planes 为 `[64,256,256,256,256,256,128,64,64]`、3 次 recycle、4 层 bottleneck attention、4 层图消息传递；图分支和 PDB 内其他 SMILES 摘要分支的门初值都为 0。

为了避免布尔开关组合爆炸，配置只使用两个有明确枚举值的结构字段：

- `model.candidate_input: full | blob_only | pocket_only`
- `model.identity_context: full | trunk_only`

四个正式变体定义如下：

| 变体 | 候选输入 | RAUNet 与区域头 | 身份侧 |
| --- | --- | --- | --- |
| full | A、P、V、PP；使用历史 top1 摘要拼接 | 执行 RAUNet，训练 ligand-area 与 binding-area 辅助头 | SMI-TED 主干 + 自身 SMILES 图注意力 + PDB 内其他 SMILES 摘要注意力，两个附加分支零门初始化 |
| blob-only | 只使用 V/blob 及其 RAUNet 调制；A、P、PP 不读取、不进入计算 | 执行 RAUNet，保留两个区域辅助头 | 与 full 相同 |
| pocket-only | 只使用 `A_feat_L0` 原始 50 维 A token；不使用 A 多尺度特征、V/blob、P、PP 或手工统计 | 完全不实例化、不执行 RAUNet 和两个区域头；两个区域损失权重为 0 | 与 full 相同 |
| trunk-only | 候选侧与 full 完全相同 | 与 full 相同 | 只保留候选表示与当前 SMI-TED 向量的主干配对打分；不实例化、不执行 SMILES 图分支与 PDB 内其他 SMILES 注意力分支 |

这里的“禁用”必须由模块构造和前向调用路径证明，不接受仅把输出门乘以 0 但仍加载数据、执行模块的伪消融。trunk-only 仍以合法精确 SMILES 槽训练，只是不消费图张量和 PDB 内其他 SMILES 上下文。

## 7. 损失、验证与 BEST 选择

### 7.1 训练损失

总损失只由当前变体实际存在的项组成：

`foreground BCE + identity cross-entropy + optional 0.1 × ligand-area loss + optional 0.1 × binding-area loss`

- foreground BCE 在所有保留候选上计算。
- identity cross-entropy 只在真值前景且 `identity_supported=true` 的候选上计算。
- pocket-only 不创建区域输出，区域损失不是返回常数 0 的假接口，而是从模型输出和总损失结构中消失。
- 不保留旧六分类交叉熵、oracle/detached/joint 身份变体或 identity scope。

### 7.2 验证指标

验证只保留两族科学指标及必要 support：

1. `foreground/prauc` 与 `foreground/f1`。
2. `identity/conditional_top1`：总体及 small_molecule、sugar、ion、peptide_like、nucleotide_like 五类，并记录每项 support。

conditional 的条件是“真值为前景且身份受支持”，不是“旧六分类预测正确”。它直接取当前 PDB 身份 logits 的 argmax，不使用二分类阈值。

PRAUC 固定实现为以前景为正类的 average precision；同分预测共同处理，不另存一套梯形插值 PR 面积。F1、PRAUC 和 conditional Top-1 都必须同时保存实际分子、分母或 support，零分母写空值与 `N/A`，不能伪造成 0。

W&B 只额外记录理解训练是否正常所需的总损失、各实际损失项、学习率、更新次数、裁剪前梯度范数、吞吐以及 full/blob-only/pocket-only 中实际存在的门值。旧的数百个六分类、seen/unseen、end-to-end、贡献诊断和 validation 区域指标不再上传。

### 7.3 BEST 与二分类阈值

每轮 validation 的 BEST 分数固定为：

`selection_score = 0.5 × foreground_PRAUC + 0.5 × overall_conditional_top1`

F1 不参与逐轮选模。只有 selection score 严格改善时才更新 BEST；相等时保留更早 checkpoint，确保行为确定。

当某一轮成为新的 BEST 时，在该轮完整 validation 前景概率上选择最大化 F1 的阈值；F1 并列时先选择 precision 更高者，再选择阈值更高者。阈值、对应 F1、PRAUC、conditional Top-1、support、manifest SHA、模型配置和 checkpoint 参数一起写入 BEST。最终真实受体和 CryoAtom2 测试必须使用同一个 BEST 阈值，禁止分别调参。

## 8. 四模型训练方案

四个实验除模型变体的必要差异外，严格共享：

- 同一冻结 train/validation manifest 及 SHA-256。
- `data.selected_candidates_only=true`。
- `coverage_threshold=0.30`。
- SMI-TED 模型与精确 SMILES 契约。
- 随机种子 3409。
- AdamW，初始学习率 `1e-4`，weight decay `1e-4`。
- 物理 batch 8、梯度累积 4、全局 batch 32。
- BF16、`gradient_clip_norm=200`。
- `ReduceLROnPlateau` 与连续 2 个完整 validation 无严格改善后早停。
- 从零初始化，不加载旧 checkpoint。
- 相同 GPU 型号。若资源只能分批取得，也必须使四项使用同一硬件族；不可把 A100 与 A800 的时间或数值差异解释成模型差异。

正式执行顺序为：先构建并冻结一次 manifest，再分别启动四个短而自包含的 YAML。配置之间只允许 `plan_name`、输出目录、W&B group、`candidate_input`、`identity_context` 及 pocket-only 必需的区域损失删除不同。执行记录应保存 release、launch、YAML、动态命令、W&B、运行根、BEST/LAST、早停轮次和最终指标。

## 9. 正式测试与图表

### 9.1 推理单位与排序

每个模型的 BEST checkpoint 分别在真实受体和 CryoAtom2 的完整 `held_out_test_0` 上运行。正式候选范围固定为 selected 候选，按 Stage1 分数降序排序；分数并列时按 `source_blob_index` 升序。一个 PDB 的 selected 候选少于 k 时使用全部候选；没有 selected 候选时，该 PDB 在对应 k 上记为未命中。

Matcher 对每个具有受支持身份集合的候选都返回一个 argmax SMILES，不因其二分类结果是假阳性而省略身份预测。这保证二分类能力与匹配能力可分别分析。

### 9.2 测试指标

每个模型、每种受体条件都输出：

- foreground PRAUC。
- 使用 BEST 中固定阈值计算的 foreground F1。
- conditional SMILES Top-1 总体、五类分项及 support。
- 每个 PDB、每个 k=1…20 的 Stage1 hit 与 Stage1+Matcher hit。

Stage1 hit 的定义是前 k 个原始候选中至少有一个候选在 0.30 双向覆盖下命中真实前景，不要求该 occurrence 有可用身份资产。Stage1+Matcher hit 还要求该候选预测的精确 SMILES 等于匹配 occurrence 的顶层 `PreparedLigand.smiles`。二分类阈值不参与这条 Stage1+Matcher 曲线，因为用户要求模型即使把候选判断为假阳性也返回匹配结果。

`9ian` 与 `9oq7` 的不受支持候选遵循第 2.4 节规则：保留 Stage1 hit 资格，永远不能形成 Stage1+Matcher hit，且不允许删掉后由下一名补位。

### 9.3 必需图表

1. 七组条形图：总体 conditional Top-1、small_molecule、sugar、ion、peptide_like、nucleotide_like conditional Top-1，以及 foreground F1。每组左右两栏分别是真实受体和 CryoAtom2。每栏标注分子/分母或 support；核酸状显示 `N/A (support=0)`，肽显示 `support=1`。PRAUC 放入配套结果表，不把不同含义的 PRAUC 强塞入第七组。
2. 四条折线：真实受体 Stage1、真实受体 Stage1+Matcher、CryoAtom2 Stage1、CryoAtom2 Stage1+Matcher；横轴 k=1…20，纵轴为 179 个 PDB 中的 hit 比例。

四个模型都生成机器可读结果。主报告重点展示哪个模型，必须只依据 validation selection score 预先选择，不能看 test 后再挑模型。其余三个模型进入消融表和补充图。

测试输出至少包含逐候选预测 JSONL、逐 PDB top-k JSONL、汇总 JSON、SVG/PNG 图、checkpoint/manifest/config SHA 与软件环境。`held_out_test_1` 结果由 179-PDB 逐 PDB预测按固定 PDB 列表过滤后生成，并明确标注“149-PDB 保序子集”。

## 10. 实施步骤与停止条件

### 10.1 服务器资源边界

2026-09-16 只读核查时，Job 378693 是一张 A800 上的旧 validation-on-validation 驻留任务；它不是本计划的科学结果。Job 378587 正在生成 Find_1 第 5 个训练分片，Matcher 不修改、终止或等待该任务。

用户批准实施后，任何 GPU 操作前都重新核对 Job、GPU、try/after/pre/kill lock 和实际进程，不能把本文的时点信息当成持续授权。若 378693 仍可安全接管，优先用于真实 GPU 的小规模前向、反向、validation 与 I/O 门禁；正式四模型是否使用该 allocation，取决于冻结清单、release 和同硬件比较条件是否全部满足。门禁命令和正式训练命令分别留证。

早先约定的 validation-on-validation 驻留运行不计入四项科学实验。只有代码、门禁、四模型训练与评估工作全部完成且用户仍要求保留占卡运行时，才用最终代码和在线 W&B 启动；确认能够持续正常前向后即视为资源侧收口，不把该运行结果写入科学结论。

### 10.2 阶段 A：建立单一数据契约

1. 从当前 `Learn/CUMULATIVE` 唯一最新端点开始，不回退到历史参考提交。
2. 统一 `occurrence_id`、`source_blob_index`、`centered_box_index` 命名；只保留一份候选—occurrence 匹配函数。
3. 在 `matcher/data` 中转正 SMILES 图读取、Find_1 标签连接以及仅含 `build_manifest`/`load_manifest` 的清单实现。
4. 对已完成 Find_1 小范围产物建立机械清单，验证重载逐值一致、selected 开关计数正确、缺失资产规则正确。

停止条件：单一 manifest 能确定性重建；同一输入重复构建内容哈希一致；运行时不再需要 `object_key`。

### 10.3 阶段 B：合并 Dataset、模型、损失与训练

1. 将确有价值的新逻辑逐项并入成熟目录，复用成熟的 RAUNet、摘要拼接、checkpoint、早停和追踪实现。
2. 将六分类头替换为单一前景头，将身份监督简化为精确 SMILES 交叉熵。
3. 实现两个枚举开关和四个明确变体；用前向 hook 或等价机械证据证明被禁用模块没有执行。
4. 将有效测试迁入 `matcher/tests`，随后删除并行包和并行测试目录。

停止条件：完整 pytest 通过；`rg` 不再发现运行时 `matcher.smiles_identity` 导入、LigandObject 身份读取或两套 manifest；四个变体都能完成一次真实 GPU 前向、反向和 validation。

### 10.4 阶段 C：冻结正式数据并训练

1. 在用户批准的冻结时点扫描已经完整的 Find_1 train/validation 产物；不等待固定分片数。
2. 原子发布正式 manifest，记录计数和 SHA；四个 YAML 只引用该清单。
3. 在同一硬件族上启动四项正式训练，按 BEST 规则自然早停。

停止条件：四项训练均产生可读取的 BEST/LAST；BEST 内含固定二分类阈值、manifest/config/checkpoint 证据；无任务仍错误写入 tmp。

### 10.5 阶段 D：正式评估和报告

1. 对四个 BEST 在两种受体条件的 179-PDB test0 上推理一次。
2. 生成所有测试指标、两类规定图和 149-PDB 补充结果。
3. 依据 validation 预选主模型，撰写结果解释、消融结论与限制，不把 support 0 或 1 的比例过度解释。

停止条件：机器可读结果可重新生成图表；所有百分比可追溯到逐候选/逐 PDB记录；真实受体与 CryoAtom2 使用同一 validation 阈值；两处不受支持身份保持原始名次。

### 10.6 阶段 E：Git 与文档收口

1. 真实实现历史保存在实现分支；稳定后按人类理解顺序重建学习分支。
2. 主代理先完成两遍自查：第一遍逐文件核对职责、位置、调用关系、嵌套与 Docstring；第二遍按代码注释规范核对函数职责、数据形状、变量语义和关键科学步骤。
3. 在关键收口点节制派发三个独立审查：Python 结构与双线 Git、代码注释、科学逻辑与契约。进行两轮全面审查；修正后只对指出位置做一次窄口径复核，不无限迭代。
4. 实现端点与学习端点通过 tree 等价核验后，将 `Learn/CUMULATIVE` 快进到学习端点；最终保持一个工作树且该分支为唯一最新提交。

停止条件：代码、测试、正式配置、执行记录和结果报告一致；无临时审查提交或并行工作树遗留；用户原有未提交内容没有被混入、丢弃或覆盖。

## 11. 门禁与验收清单

### 11.1 机械门禁

- 清单重复构建哈希一致，PDB/候选顺序确定。
- 标签旁路与 centered 的两个连接轴逐项一致。
- `selected_candidates_only` 的 true/false 计数符合来源文件。
- train/validation 缺失正样本整条跳过；测试缺失正样本只退出身份指标。
- SMILES 图包每个 persistent worker 只初始化一次。
- full、blob-only、pocket-only、trunk-only 的输入和模块执行集合与表格完全一致。
- pocket-only 不构造 RAUNet 或区域头；trunk-only 不构造图和其他 SMILES 注意力模块。
- 人工构造小样本能验证 PRAUC、F1 阈值、conditional Top-1、support 和 top-k。
- `9ian`、`9oq7` 的名次、Stage1 hit 与 Stage1+Matcher 非命中规则有固定回归测试。
- 完整 CPU pytest、一次真实 GPU 训练步、一次完整 validation 和 I/O 基准通过。

### 11.2 科学门禁

- 身份键逐字等于顶层 `PreparedLigand.smiles`。
- 模型、清单、checkpoint 和评估中没有以 `object_key` 回退。
- 四模型除声明的消融结构外使用同一数据和训练参数。
- BEST 只由 validation PRAUC 与 overall conditional Top-1 等权决定。
- 测试不参与阈值、模型或超参数选择。
- test0 是唯一主要测试范围；test1 明确是子集补充。
- 支持数随每项比例共同报告，零支持输出 N/A。

### 11.3 技术表达与可维护性门禁

- 每个业务概念只有一处实现和一套名称。
- 函数围绕完整业务动作组织，不把简单下标或一行转换拆成大量无意义小函数。
- 类、函数和关键数组按项目规范说明职责、形状、单位、轴顺序和特殊值；测试文件不做无收益的逐行教学注释。
- YAML 逐字段说明类型、作用、特殊值与实际读取位置，但不复制运行取证和一次性 SHA。
- 正式命令保持短、稳定、自包含；pytest、smoke、审计脚本等门禁命令与正式运行命令在执行记录中分栏保存。

## 12. 明确不做的事项

- 不等待 Find_1 全部 50 个训练分片才开始 Matcher 四模型训练。
- 不恢复 LigandObject、`object_key`、model_smiles 或 graph_id。
- 不增加第二套 SMILES 图格式或新的原子化学特征实验。
- 不恢复 seen/unseen、六分类预测、end-to-end F1 或 calibration split。
- 不为 selected/all 或 train/validation/test 建立多套重复清单实现。
- 不把 `held_out_test_1` 当独立测试，不因缺少身份资产而删候选补位。
- 不在本次四模型任务中继续扩充模型变体或超参数搜索。
- 不在得到用户下一次明确实施许可前修改代码、YAML、Git 或服务器文件。

## 13. 预期最终交付物

1. 单一、分层、无 `matcher/smiles_identity` 并行包的 Matcher 代码。
2. 一份冻结 train/validation manifest 及其来源、摘要和 SHA。
3. full、blob-only、pocket-only、trunk-only 四份简洁正式配置与启动脚本。
4. 四项训练的 BEST/LAST、W&B 记录和执行证据。
5. 两种受体条件、179-PDB test0 的逐候选与逐 PDB 结果，以及 149-PDB 补充结果。
6. 七组条形图、四曲线 top-k 图、汇总表和科学结论报告。
7. 更新后的规划—执行映射、执行记录与 handoff。
8. 双线 Git 等价、一个工作树、`Learn/CUMULATIVE` 为唯一最新提交的收口状态。

这套方案把 LigandObject→SMILES 的契约改造视为已验证基础，把本轮工作重点放在“单一实现、最少配置、两个可解释任务、四个必要模型、一次正式测试”上。它不依赖尚未完成的全部 Stage1 训练分片，也不会通过保留兼容分支继续积累代码复杂度。
