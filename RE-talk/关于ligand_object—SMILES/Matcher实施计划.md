# Matcher 单一 SMILES 身份实施计划

> **现行结论（2026-09-14）**：本文以下正文保留 2026-09-13 讨论收口前的设计轨迹，
> 其中“等待许可”、schema 5、每 PDB 图包、拒绝多片段和不单独保存自同构等说法已经
> 被后续授权取代，不能继续作为实现契约。现行实现采用
> `matcher/smiles_identity/`、`smiles_identity_v1`、版本化全局
> `smiles_graphs_v1.npz` 与同序 `smiles_symmetries_v1.npz`；合法断开组分保留，严格
> RDKit 解析或 sanitize 失败的字符串排除。当前字段、目录和正式命令以
> [精确 SMILES 身份管线 README](../../matcher/smiles_identity/README.md) 和
> [执行记录](执行记录.md) 为准。

## 状态与授权边界

- 历史状态：本文初稿写成时设计完成、尚待用户许可；用户随后已明确授权实施。
- 本文允许落盘的内容仅是计划，不代表任何步骤已经执行。
- 后续实现允许重新定义监督任务、修改模型输入并从头训练；不要求加载联合身份 schema 4 的 checkpoint。
- 开始实现前还需确认一个数据口径：是否接受统一严格 SMILES 门禁排除 held-out PDB `9oq7` 的 3 个 `model_failed` occurrence。详细证据见 [信息.md](信息.md)。

## 目标

把 Matcher 的正式身份和配体图入口统一为 PreparedLigand 顶层 smiles：

```text
AdaLigand 产生密度图与 PreparedLigand 顶层 smiles
→ Matcher 读取预先构造的 SMILES 重原子图
→ Matcher 在当前 PDB 的身份集合中选择一个精确 smiles
→ 满足 Stage3 条件的 occurrence 由 PocketXMol 使用同一 smiles 图做 docking
```

完成后，Matcher 的正式清单、Dataset、Batch、模型、损失、评估和结果均不需要 `object_key`。历史 LigandObject 代码与资产保留在原位置，但不进入新运行路径。

## 不在本轮实现范围内的事项

- 不修改 PocketXMol；只提供交接建议。
- 不删除 AdaLigand、Matcher 或 PocketXMol 中既有 LigandObject 代码和产物。
- 不维护旧 checkpoint 兼容层。
- 不保留 schema 4 与新 schema 的双路径运行开关。
- 不添加异常 SMILES 的宽容解析、LigandObject 回退或手工修补表。
- 不把自同构排列写入 Matcher 公共图资产。
- 不保留 seen/unseen 指标或同义统计。

## 新科学契约

### 身份

暂定把新清单版本称为 schema 5。正式身份规则为：

1. 身份源必须是 AdaLigand `prepared_smiles.jsonl` 记录的顶层 `smiles`；语言模型结果、向量 NPZ 和 Matcher 清单中的 `prepared_smiles` 只能保存它的逐字副本，并在清单构建时核验相等。
2. 不使用 `model_smiles`、occurrence 坐标、`present` 掩码、沉积原子选择或旧模板 `object_key` 定义身份。
3. 同一 PDB 内，精确 prepared_smiles 相同的 occurrence 共享一个身份槽。
4. 不跨 PDB 建立监督身份槽。相同字符串在不同 PDB 中仍分别参与各自候选集合。
5. 类别标签仍属于当前 PDB 中 occurrence 的监督事实，不成为身份键的一部分。
6. SMILES 严格图门禁失败的 occurrence 在进入候选和身份分组前整体排除。

### 配体图

每个身份槽的图只含能从精确 prepared_smiles 确定的字段：

| 字段 | 建议 dtype 与形状 | 含义 |
|---|---|---|
| `element` | `int16 (N,)` | 重原子原子序数。 |
| `charge` | `int8 (N,)` | 形式电荷。 |
| `atom_in_ring` | `bool (N,4)` | 依次表示原子是否位于 3、4、5、6 元环。 |
| `bond_index` | `int32 (2,E)` | 每条无向化学键的两个端点；端点是当前图内的局部原子下标。 |
| `bond_type` | `uint8 (E,)` | 化学键类别枚举；枚举顺序必须在 schema 中固定。 |
| `bond_in_ring` | `bool (E,4)` | 依次表示化学键是否位于 3、4、5、6 元环。 |

不再产生或消费：

- `ref_pos` 和由它计算的参考键长；
- `residue_id`；
- LigandObject 模板 `is_present`；
- 七维模板 chirality；
- object_key、模板文件名或新的 `graph_id`。

## 离线图归档

### 为什么建议落盘

可以并且建议按 `(pdb_id, exact prepared_smiles)` 离线落盘图数据。这样可让 Matcher 和未来 PocketXMol 读取同一份确定数组，避免每个 DataLoader worker 在训练时初始化 RDKit、解析字符串和重复构图。

预期收益主要是缩短冷启动、降低多 worker 的 CPU 峰值和消除 RDKit 版本或异常处理进入训练热路径的机会。现有 worker 内 LRU 缓存已经避免同一 worker 无限重复解析，因此不能在实际基准前声称训练每步会显著加速；实现后应分别测量首个 batch 延迟、一个 epoch 的 wall time 和 GPU 等待数据比例。

### 查询键与文件布局

推荐每个 PDB 一个压缩 NPZ：

```text
<prepared-data-root>/
└── smiles_graphs/
    ├── 6et5.npz
    ├── 6zxa.npz
    └── ...
```

运行时查询过程固定为：

```text
pdb_id
→ smiles_graphs/{pdb_id}.npz
→ 在 smiles 数组中逐字查找 exact prepared_smiles
→ 用 offsets 取出对应图
```

`(pdb_id, smiles)` 是显式可读的查询条件，不是新的科学身份。PDB 只定位归档文件；真正的化学身份仍是精确 smiles。不要把原始 SMILES 写进文件名，也不要创建散列 `graph_id`。

### NPZ 最小字段

一份 PDB 归档包含该 PDB 中通过严格门禁的所有唯一 exact prepared_smiles。建议字段为：

| 字段 | dtype 与形状 | 约束 |
|---|---|---|
| `schema_version` | 整数标量 | 初始值为 1；只描述图归档格式，不等于 Matcher 清单 schema 5。 |
| `smiles` | Unicode `(K,)` | 当前 PDB 的 K 个精确字符串；去重后按字典序排列。 |
| `atom_offsets` | `int64 (K+1,)` | 第 k 个图的原子切片为 `[atom_offsets[k], atom_offsets[k+1])`。首项为 0，末项等于全部原子数 A。 |
| `element` | `int16 (A,)` | K 个图的原子序数顺序连接。 |
| `charge` | `int8 (A,)` | 与 `element` 逐位置对齐。 |
| `atom_in_ring` | `bool (A,4)` | 与 `element` 逐位置对齐。 |
| `bond_offsets` | `int64 (K+1,)` | 第 k 个图的键切片为 `[bond_offsets[k], bond_offsets[k+1])`。首项为 0，末项等于全部无向键数 E。 |
| `bond_index` | `int32 (2,E)` | 每个键切片内的端点均使用该图的局部下标 `0..N_k-1`。 |
| `bond_type` | `uint8 (E,)` | 与键切片逐位置对齐。 |
| `bond_in_ring` | `bool (E,4)` | 与键切片逐位置对齐。 |

具体例子：若一个 PDB 只有 `[Ca+2]` 和 `CCO` 两张图，则：

```text
smiles       = ["CCO", "[Ca+2]"]       # 实际顺序按 Unicode 字典序
atom_offsets = [0, 3, 4]
element      = [6, 6, 8, 20]
charge       = [0, 0, 0, 2]
bond_offsets = [0, 2, 2]
bond_index   = [[0, 1], [1, 2]]          # CCO 图内局部下标；钙离子无键
```

该归档不保存 `valid`、`error`、object_key、坐标、自同构或语言向量。无效字符串根本不写入正式归档；一次性构建诊断另存于清单构建目录，Dataset 不读取它。

### 离线构图算法

对每个精确 prepared_smiles 只执行一次：

1. 以精确字符串调用 RDKit 解析，不先改写字符串。
2. 保留字符串中的显式氢完成初始分子建立，再调用 `Chem.RemoveHs`。
3. 执行完整 `Chem.SanitizeMol`；任何异常均判定该 occurrence 不合格。
4. 拒绝空图和多片段图。
5. 使用 RDKit 返回的重原子顺序作为唯一基础顺序。
6. 提取原子、键和环字段；键只保存一次，Matcher 装批时再按现行图网络需要补反向边。
7. 对同一 exact prepared_smiles 跨 PDB 生成的字段做逐位一致性核验。任何不一致都使构建失败，不能选择其中一份继续。
8. 对临时文件完成字段、offset、范围和内容校验后再原子替换正式 NPZ。

正式运行不提供“归档缺失时现场解析”的回退。缺文件、缺字符串、schema 不符或数组门禁失败都应快速失败，从而保证训练实际消费的图就是被审计过的图。

## 清单 schema 5

新清单保持现有“一个 PDB 的候选集合与身份集合”结构，但删除所有模板和可见性字段。

### 应保留的核心字段

- `pdb_id`、`candidate_id`、类别与几何匹配所需的 occurrence 字段；
- 候选命中时的精确 `prepared_smiles`；
- 身份记录中的精确 `prepared_smiles`、类别、成员 occurrence 列表；
- 按现行规则确定性选择的语言向量来源；
- 清单输入快照、构建配置和 schema 版本。

### 应删除的字段

- `object_key`、`identity_object_keys`、`matched_object_key`；
- `graph_relative_path` 或其它 LigandObject 模板定位字段；
- `seen_in_train_by_language`；
- `identity_seen_in_train_by_language`；
- `identity_seen_in_train`；
- `seen_positive_identity_count`；
- 所有 `seen_*`、`unseen_*` 指标、计数器和输出列。

### 无效字符串诊断

为了让排除行为可复核，而不把异常逻辑带入训练，清单构建可以单独输出一个只供人检查的 `excluded_invalid_smiles.jsonl`。每条只需：

```json
{"split":"validation","pdb_id":"...","candidate_id":0,"prepared_smiles":"...","reason":"strict_smiles_sanitize_failed"}
```

该文件不是正式 Dataset 输入，不影响运行时分支，也不做异常种类的复杂分类。构建摘要只报告总排除数和各划分排除数。

## Matcher 模型调整

### 图张量

当前 `LigandGraph` 和 Batch 应收缩为：

- `graph_element`；
- `graph_charge`；
- `graph_atom_in_ring`；
- `graph_edge_index`；
- `graph_bond_type`；
- `graph_bond_in_ring`；
- `graph_batch` 和图偏移。

删除 `graph_residue_id`，把 `graph_node_continuous` 从混合模板特征改为明确命名的 4 维原子环标志；把 `edge_features` 从 10 维混合特征拆成键型和 4 维键环标志，避免字段切片继续隐含旧契约。

### 编码器

`LigandGraphEncoder` 需要：

1. 删除 residue embedding；
2. 删除参考坐标、参考距离、模板手性和 `is_present` 对应的投影；
3. 仅嵌入元素、形式电荷、原子环标志、键型和键环标志；
4. 保持现有消息传递、整图汇聚和与语言向量融合的主体结构，除非机械门禁证明接口不再成立；
5. 重新初始化全部模型权重并从头训练。

### 自同构

Matcher 不预测原子坐标，也没有原子编号真值，因此无需生成或选择 `matches_iso`。RDKit 基础顺序只负责稳定序列化。必须增加一个模型级门禁：对同一图同时置换节点顺序和边端点后，整图表示与身份 logit 应在既定浮点容差内相同。若不相同，应修复图网络或汇聚逻辑，不能重新引入 object_key 来掩盖问题。

## 预计修改位置

| 位置 | 后续实现职责 |
|---|---|
| `matcher/data/ligand_graph.py` | 用只读 `SmilesGraphStore` 替换 `LigandGraphStore`；按 `(pdb_id, prepared_smiles)` 查归档。 |
| `matcher/data/manifest.py` | 构建 schema 5；在分组前应用严格图门禁；按单一 smiles 建身份；删除 seen/unseen。 |
| `matcher/data/schema.py` 与 `matcher/__init__.py` | 固定新 schema 和精简后的字段约束。 |
| `matcher/datasets/contracts.py` | 删除 object_key、旧图张量和 seen 元数据；定义新图字段。 |
| `matcher/datasets/candidate_data.py` | 加载当前 PDB 的 SMILES 图；身份目标按精确 smiles 定位。 |
| `matcher/datasets/batching.py` | 更新空图形状、连接规则和设备张量，不再构造 residue 张量。 |
| `matcher/model/ligand_matcher.py` | 收缩图编码器输入并删除 residue embedding。 |
| `matcher/evaluation/metrics.py` | 删除 seen/unseen 分桶；身份输出只保存精确 smiles。 |
| `matcher/evaluation/calibration.py` | 删除可见性相关字段与逻辑；保持其它校准口径不变。 |
| `matcher/training/engine.py` | 删除旧元数据搬运和 seen 统计；适配新 Batch。 |
| YAML、README 和字段参考 | 删除 object_key 与旧图特征说明，明确图归档路径和单一身份。 |

测试代码只按行为变化补充或更新，不进行学习型注释扩写。

## 实施顺序

得到代码修改授权后，建议按以下顺序执行：

1. 冻结并记录当前输入数据、清单、代码提交和依赖版本；不覆盖旧产物。
2. 在本地实现独立的 SMILES 图归档构建器和读取器，先用少量 PDB 验证字段。
3. 对 train、validation 和未来评估清单执行一次性严格门禁；保存最小排除诊断。
4. 生成全部 `smiles_graphs/{pdb_id}.npz`，完成跨 PDB 同字符串逐位一致性审计。
5. 改造 schema 5 清单构建和单一 SMILES 身份分组。
6. 改造 Dataset、Batch、图编码器、训练和评估；一次性删除 object_key 与 seen/unseen 路径，不维护双实现。
7. 运行机械门禁、CPU Dataset 门禁、GPU 固定前向和小规模过拟合检查。
8. 对正式 train/validation 重新生成清单并从头训练；旧 schema 4 结果只作为历史参考，不作为严格同轨对照。
9. 实验稳定后，把精确图契约与坐标迁移建议交给 PocketXMol 维护者；Stage3 改造作为独立任务验收。
10. 按项目双线 Git 规则完成实现历史、学习历史和端点等价核验。

## 必须通过的验收门禁

### 数据契约

- 身份源只允许 PreparedLigand 顶层 smiles；扫描正式代码和配置时不得发现以 `model_smiles`、object_key 或坐标字段定义身份。
- 正式图归档不含 object_key、graph_id、坐标、模板残基、模板手性或模板 `is_present`。
- 每个 PDB 的 `smiles` 唯一且按确定顺序保存；offset 单调、边端点合法、数组 dtype 和形状精确匹配 schema。
- 同一精确 smiles 跨 PDB 的全部图数组逐位相同。
- 无效字符串不会进入清单、Dataset 或评估分母；排除数量与冻结审计结果一致。

### 身份与语言向量

- 当前 train 身份槽应由 1,477 变为 1,475，validation 由 536 变为 535；若输入快照改变，必须先解释差异，不能硬编码数量。
- 三个已知合并组的可用 MolFormer、SMI-TED 向量仍逐位相等。
- 一个 PDB 内不得出现同一 exact smiles 对应多个类别；出现时构建失败并要求科学决策。
- 不再输出任何 seen/unseen 统计。

### 数值与训练

- 随机节点置换同时更新边端点后，图表示和身份 logit 在预先声明的浮点容差内相同。
- 单样本、空背景槽、无键图、单原子图和普通多键图均能完成前向与反向。
- 小数据过拟合试验能明显降低身份损失，且梯度和参数保持有限。
- 一个正式 validation 前向能完整产生候选类别、区域和新身份指标，不依赖旧 checkpoint。

### 复杂度收口

- 正式运行路径没有 LigandObject 读取和宽容解析回退。
- 正式 Matcher 包、YAML、resolved 配置和新结果中没有 object_key 或 seen/unseen 字段；历史文档、旧产物和一次性迁移工具不受此扫描约束。
- 图构建发生在离线准备阶段；训练 worker 只读已验证 NPZ。
- 删除旧路径后，总代码分支应减少；不能用兼容层同时长期维护 schema 4 和 schema 5。

## 风险与处置

| 风险 | 处置 |
|---|---|
| 69 个有效 SMILES 图与旧模板不同 | 这是已接受的任务重定义；从头训练并用新 validation 评价，不伪装成无损迁移。 |
| 严格解析失败 | 一致排除，不增加例外；实施前确认 `9oq7` 对未来 held-out 口径的影响。 |
| 原子顺序或自同构 | Matcher 用置换不变性门禁；Stage3 用确定性坐标迁移和现有对称性逻辑。 |
| 跨 PDB 重复存图 | 为简化查询而接受少量存储重复；逐位一致性门禁防止同一身份出现两种图。 |
| 离线 NPZ 未明显提升吞吐 | 仍可获得确定数据契约和更简单运行环境；用实测决定压缩级别和 worker 预取参数，不改变科学字段。 |
| 旧报告依赖 seen/unseen | 新报告停止延续这些指标；历史报告保留原口径，不重写历史。 |

## 停止条件

本改造在同时满足下列条件后才可视为完成：

1. 新正式管线从清单到训练、推理和评估均只使用 exact prepared_smiles 身份；
2. Matcher 的数值前向不再读取 LigandObject 派生字段；
3. seen/unseen 全部删除；
4. 严格解析失败项按用户确认的统一口径排除；
5. 数据、置换不变性、前向、反向、validation 和独立科学审查均通过；
6. 新模型完成从头训练并形成可与历史结果区分口径的报告；
7. 双线 Git 收口后，Learn/CUMULATIVE 是仓库按提交者时间形成的唯一最新提交。
