# Stage3 与端到端管线从 LigandObject 切换到 SMILES 的建议

## 交接对象与范围

本文供未来专门改造 Stage3 的 AI agent 或开发者使用。当前 Stage3 仓库是：

```text
C:\Users\15919\Desktop\PocketXMol
```

不是历史讨论文档曾写的 Builder 仓库。本次 Matcher 决策只产出建议，不修改 PocketXMol 代码、配置、模型权重或服务器产物。

目标不是删除历史 LigandObject 文件，而是让新的正式端到端管线不再读取它：

```text
Stage1 产生配体候选空间区域
→ Matcher 接收密度图和精确 PreparedLigand 顶层 smiles 集合
→ Matcher 选择一个精确 smiles
→ 只对满足 Stage3 累计条件的 occurrence 执行 docking
→ PocketXMol 从同一精确 smiles 图和候选空间区域构造输入
```

Stage3 累计条件至少包括：单残基、small molecule、含碳、沉积重原子相对目标图完整。`candidate_id` 继续定位具体 occurrence 和空间位置，但不定义化学身份。

## 已证实的可行性

对 PocketXMol 当前冻结的 train、validation、calibration、test 四个划分进行了完整只读图审计：

| 项目 | 结果 |
|---|---:|
| 正式 occurrence | 66,878 |
| 精确 PreparedLigand 顶层 smiles | 1,979 |
| `(smiles, object_key)` 对 | 2,039 |
| SMILES 重建图与当前 LigandObject 图双向同构 | 2,039 / 2,039 |
| 严格解析失败 | 0 |
| 多片段图 | 0 |
| 原子、键或图不匹配 | 0 |

审计的重建流程是：精确字符串解析、保留显式氢完成初始建图、`RemoveHs`、严格 sanitize，再比较原子序数、形式电荷、化学键端点和键型。比较允许原子重编号，不允许删改原子或化学键。

因此，在当前 Stage3 的完整正式四划分范围内，精确 PreparedLigand 顶层 smiles 可以无歧义地重建 PocketXMol 目前实际使用的重原子化学图。这一结论直接比较了 SMILES 重建图与 Stage3 当前图，不依赖 object_key 别名之间的间接等价推断。

顶层 smiles 是 non-isomeric，因而有意不区分立体异构。当前 Stage3 模板手性均为 `CHI_UNSPECIFIED`，这次身份改造不会删除 PocketXMol 现行模型实际消费的手性信息。

## 为什么自同构不是身份歧义

现有 [preparation.py](../../../PocketXMol/docking/preparation.py) 已调用 `GetSubstructMatches` 为配体生成 `matches_iso`。这一步服务于逐原子坐标训练和对称性 RMSD，并不是在多个化学图中选择一个。

同一重原子图可能有多个保持元素、电荷和全部化学键不变的等价原子置换：

```text
一个精确 smiles
└── 一个确定的重原子化学图
    └── 一个或多个图自同构原子置换
```

因此：

- Matcher 不预测原子坐标，只对完整身份打分，不需要保存或选择自同构；
- PocketXMol 需要继续生成和使用 `matches_iso`，因为坐标损失与 RMSD 必须识别等价原子；
- `matches_iso` 是 Stage3 派生产物，不应放进 Matcher 与 Stage3 共用的最小 SMILES 图归档；
- 一个基础原子顺序和多种等价原子置换可以同时成立，不会导致 Matcher 输出对应两种 Stage3 化学构图。

## 建议的最小数据契约

### 公共 SMILES 图

Matcher 计划离线生成每个 PDB 一份：

```text
smiles_graphs/{pdb_id}.npz
```

查询键是 `(pdb_id, exact prepared_smiles)`，其中 PDB 只定位文件，精确 smiles 是化学身份。归档只保存：

- `schema_version`；
- `smiles`；
- `atom_offsets`、`element`、`charge`、`atom_in_ring`；
- `bond_offsets`、`bond_index`、`bond_type`、`bond_in_ring`。

不保存 object_key、graph_id、参考坐标、沉积坐标、`present`、模板残基、模板手性或自同构。PocketXMol 应直接读取这份公共图，而不是在每次 Dataset 取样时重新解析 SMILES，也不要复制一套语义不同的 Stage3 图缓存。

如果未来工程边界要求两个仓库各自持有只读副本，必须用内容散列和逐字段比较证明副本完全相同；不能让 Matcher 与 PocketXMol 各自定义一套构图规则。

### SMILES 顺序坐标

Stage3 还需要一种仅供坐标任务使用的派生产物：

```text
smiles_ordered_coords/{pdb_id}.npz
```

最小字段为：

| 字段 | dtype 与形状 | 含义 |
|---|---|---|
| `schema_version` | 整数标量 | 坐标归档格式版本。 |
| `candidate_ids` | `int64 (M,)` | 当前 PDB 内通过 Stage3 累计条件的 occurrence 标识，升序保存。 |
| `prepared_smiles` | Unicode `(M,)` | 每个 occurrence 的精确顶层 smiles。 |
| `coord_offsets` | `int64 (M+1,)` | 第 m 个 occurrence 的坐标切片。 |
| `coords` | `float32 (Q,3)` | 所有 occurrence 的真值坐标，按各自 SMILES 基础重原子顺序连接，单位 Å。 |

由于正式 Stage3 条件已经要求沉积重原子相对目标图完整，新的坐标归档不需要 `present` 掩码。若未来纳入不完整样本，应新立科学任务，不能悄悄恢复掩码并改变本契约。

坐标迁移可以一次性读取旧 LigandObject：

1. 从 exact prepared_smiles 产生基础重原子图及 RDKit 原子顺序。
2. 将旧模板图与 SMILES 图做完整同构匹配。
3. 若有多个完整映射，按映射下标元组的字典序选择最小项。
4. 用所选映射把 occurrence 真值坐标排列到 SMILES 原子顺序。
5. 校验元素、电荷、全部键、坐标长度和有限性后写入新归档。
6. 迁移完成后，正式 Dataset 不再读取 LigandObject。

多个可选映射只相差图自同构。选一个确定映射用于基础坐标顺序，Stage3 的 `matches_iso` 继续覆盖其它等价排列。

### Stage3 自同构资产

Stage3 可继续保存自己的对称性归档，但身份字段应改为 exact prepared_smiles。建议从公共图恢复 RDKit Mol，再执行与当前一致的自匹配算法。必须记录并固定：

- 是否使用手性匹配；
- 最大匹配数 10,000；
- 恒等排列是否补入；
- 只保存发生置换的原子列还是完整排列；
- 基础原子顺序来自公共图。

当前算法的自同构数量可能达到 10,000；迁移后应逐图比较新旧排列所代表的等价集合，而不能只比较数组行顺序。

## PocketXMol 建议修改点

### 清单与准备阶段

1. Stage3 manifest 用 `prepared_smiles` 替换 `object_key`，不要长期同时保留两者。
2. 筛选身份时使用精确字符串，不二次 canonicalize，不使用 `model_smiles`。
3. 图合法性、单片段、small molecule、单残基、含碳和沉积重原子完整性继续作为 Stage3 累计门禁。
4. `candidate_id` 只定位 occurrence、真值坐标和 Stage1 空间区域。
5. 从公共图生成 Stage3 自同构资产；不要从 object_key 模板生成。

### 代码入口

| 现有位置 | 建议变化 |
|---|---|
| [assets.py](../../../PocketXMol/docking/assets.py) | 增加从公共 SMILES 图字段恢复 RDKit Mol 的单一入口；正式路径停止读取 LigandObject。 |
| [preparation.py](../../../PocketXMol/docking/preparation.py) | 按 prepared_smiles 生成对称排列；完成一次性坐标迁移；清单不再按 object_key 分片。 |
| [dataset.py](../../../PocketXMol/docking/dataset.py) | 从公共图和 SMILES 顺序坐标读取元素、键与坐标；去掉模板文件名转换。 |
| [sampling.py](../../../PocketXMol/docking/sampling.py) | 从清单中的 prepared_smiles 取图；采样结果保存精确 smiles。 |
| [evaluation.py](../../../PocketXMol/docking/evaluation.py) | 评估身份改为 `(pdb_id, candidate_id, prepared_smiles, view)`；RMSD 继续考虑自同构。 |

PocketXMol 神经网络主体无需因身份字符串变化而调整：模型实际消费的是元素、键、坐标、对称排列和口袋特征。当前模型权重是否能逐位复用，应由未来实现者用同一 occurrence 的新旧 Batch 与固定前向证明；不能只根据架构未改就宣称数值完全相同。

## Stage1 与端到端定位

Stage1 不需要知道配体化学身份。端到端推理时可以从 Stage1 预测的 blob 或候选体素区域计算确定性中心，供 PocketXMol 选择口袋。该中心的定义、阈值和坐标系应由 Stage1→Stage3 的独立接口契约规定，不应重新塞进 object_key 或 SMILES 身份。

Stage3 收到的最小信息可以是：

```text
pdb_id
candidate_id 或 Stage1 候选区域标识
exact prepared_smiles
候选区域中心与必要的空间范围
```

其中只有 exact prepared_smiles 定义化学图，空间字段只定义“在哪里 docking”。因此，即使同一 SMILES 在一个 PDB 内出现多次，各 occurrence 共享化学图但使用各自空间区域，仍然无歧义。

## 端到端无歧义条件

新管线满足以下条件时，“Matcher 输出一个 SMILES 导致 Stage3 两种构图”的情况不可能发生：

1. 公共归档中，一个 `(pdb_id, exact prepared_smiles)` 只能出现一次；
2. 同一 exact prepared_smiles 跨 PDB 的图字段必须逐位一致；
3. Stage3 图只能来自公共归档，不允许运行时重新解析或回退到 LigandObject；
4. 坐标归档必须保存 exact prepared_smiles，并与公共图的原子数和元素逐位置对齐；
5. 自同构只改变等价原子编号，不改变公共图；
6. 缺失图、缺失坐标、字符串不一致或数组门禁失败时快速失败，不能选择另一模板继续。

## Stage3 验收建议

未来改造至少应通过：

- 四划分所有 66,878 个当前正式 occurrence 的新旧重原子图双向同构；
- 每个 occurrence 的迁移坐标长度、元素顺序和有限性门禁；
- 新旧 `matches_iso` 表示相同的图自同构集合；
- 固定随机数、固定输入、同 checkpoint 的 Dataset 张量与模型前向等价检查；
- 采样输出和评估记录完全不需要 object_key；
- 全仓正式运行路径扫描不再读取 `ligand_objects/`；
- train、validation、calibration、test 的样本数量保持与冻结输入相符，任何变化都有逐 occurrence 排除清单；
- 一个小规模完整 docking 和 RMSD 评估闭环；
- 独立科学正确性审查与技术表达审查。

## 不建议的实现

- 不新增换名字后的 object_key 或 graph_id。
- 不把 raw SMILES 直接用于文件名。
- 不在 Dataset 中每次解析 SMILES。
- 不同时长期维护 object_key 图与 SMILES 图两条正式路径。
- 不把 `model_smiles` 当身份。
- 不把自同构排列误认为多个化学身份。
- 不为了兼容历史 checkpoint 把多个旧模板 logit 用 `logsumexp` 临时合并；本项目允许重训并重新定义监督任务。
- 不删除历史 LigandObject 资产；它们可作为迁移取证和历史复现材料保留，但不能进入新正式管线。

## 给未来 Stage3 实现者的最短结论

精确 PreparedLigand 顶层 smiles 已在 PocketXMol 当前四个正式划分的 2,039 个 `(smiles, object_key)` 对上直接证明能够重建当前重原子图，零失败、零图不匹配。Stage3 无需保留 object_key；只需读取 Matcher 生成的公共 SMILES 图、把真值坐标一次性迁移到其基础原子顺序，并继续从同一图生成自同构。模型主体可以保持不变，但必须用固定前向和完整四划分门禁证明数据入口迁移正确。
