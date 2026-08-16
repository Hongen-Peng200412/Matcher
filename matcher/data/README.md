# Matcher 数据适配与缓存

`matcher/data/` 把冻结的 A–G 数据产物与 Stage1 F1 候选转换成 Matcher 使用的监督清单、候选级体素输入、PP 模态 token 和 LigandObject 模板图。本文所称 occurrence，是单个 PDB 中的一次真实配体实例。这个目录不负责 Dataset 批处理、模型前向、训练循环或指标统计；它负责来源字段核验、候选与真实配体 occurrence 的匹配、PDB 内联合身份构造，以及与科学参数绑定的缓存读写。

## 稳定入口与产物

| 入口 | 作用 | 主要产物 |
| --- | --- | --- |
| `ensure_manifest` | 在构建锁内建立、复用或扩展调用者显式请求的清单划分 | `manifest.json`、候选 JSONL、身份 JSONL、排除 JSONL |
| `require_manifest` | 只读核验既有清单；参数或划分缺失时失败，不补建文件 | 已核验的 `manifest.json` 路径 |
| `VoxelInputStore.load` | 读取或构建一个候选的 105 通道局部体素输入 | 候选级 `float16` NPY |
| `PpInputStore.load` | 读取或构建一个 PDB 的 PP 缓存，再返回单候选切片 | PDB 级 PP NPZ、单候选内存对象 |
| `LigandGraphStore.load` | 由 `object_key` 读取并复用 LigandObject 模板图 | 内存中的 `LigandGraph` |
| `precompute_inputs` | 只读既有清单，按 PDB 并行物化显式选择的 voxel、PP 缓存 | 上述 NPY 或 PP NPZ |
| `precompute.main` | 先调用 `ensure_manifest`，再调用 `precompute_inputs` | 清单与显式选择的缓存 |

三个版本号分别进入清单或缓存目录：清单 schema 为 v4，体素缓存 schema 为 v2，PP 缓存 schema 为 v2。版本不同的字段契约不会共用同一路径。

## 联合身份

Matcher 的前景身份只在单个 PDB 内构造，键按顺序固定为：

```text
(object_key, PreparedLigand.smiles)
```

- `object_key` 定位 `data_root/ligand_objects/` 下的 LigandObject 模板图。
- `PreparedLigand.smiles` 是 `prepared/{pdb_id}/prepared_smiles.jsonl` 的顶层 `smiles`。Matcher 逐字采用交给语言模型公开接口的顶层字符串；该值通常是 canonical non-isomeric SMILES，`diagnostics.used_raw_smiles=true` 时则是规范化失败后的非空原文回退。
- 身份只认顶层 `smiles` 的实际值。`diagnostics`、语言归档的 `model_smiles`、occurrence 坐标、`present` 掩码和 LigandObject 的 `atoms.is_present` 都不定义身份。
- 相同 `object_key`、不同顶层 `smiles` 建立不同身份；相同顶层 `smiles`、不同 `object_key` 也不合并。
- 顶层 `smiles=null` 或整个 PreparedLigand 文件缺失时不建立身份，也不创建 `(object_key, null)`。候选记录分别写 `explicit_null` 或 `missing_file`。
- 不同 PDB 不共享同一次身份轴。跨 PDB 的同联合键类别差异只进入 `cross_pdb_joint_identity_class_variants` 诊断，不改变训练候选或分数归一化。
- `seen_in_train_by_language` 使用不含 `pdb_id` 的 `(object_key, prepared_smiles)` 查询训练可见性，因此不同 PDB 的相同联合键共享逐语言家族的 seen 事实；这项元数据共享不合并 PDB 内身份轴，也不引入跨 PDB 分数归一化。

## 来源文件字段与存在性边界

### Stage1 F1 候选

`{stage1_root}/{split_name}/{pdb_id}/centered/F1_centered.npz` 的候选轴长度记为 C，拼接稀疏体素数记为 V_total。

| 字段 | dtype 与形状 | 含义与对齐 |
| --- | --- | --- |
| `centered_box_index` | `int32 (C,)` | 候选顺序索引，严格为 `0..C-1` |
| `box_start_zyx` | `int32 (C, 3)` | 每个 80³ 候选在完整网格中的 ZYX 起点 |
| `box_shape_zyx` | `uint8 (C, 3)` | 每项固定为 `(80, 80, 80)` |
| `box_origin_world` | `float32 (C, 3)` | 每个候选低角点的世界 XYZ 坐标，单位 Å |
| `voxel_size_world` | `float32 (C, 3)` | 每个候选的 XYZ 体素尺寸，单位 Å/voxel |
| `source_tree_id` | `int32 (C,)` | 候选来源连通树编号，与候选轴逐项对齐 |
| `source_node_id` | `int32 (C,)` | 候选来源树节点编号，与候选轴逐项对齐 |
| `source_threshold_grid_index` | `int32 (C,)` | 候选来源阈值网格索引 |
| `source_threshold_value` | `float32 (C,)` | 对应阈值原值；Matcher 清单当前只保存网格索引 |
| `voxel_offsets` | `int64 (C + 1,)` | 首值为 0、单调不减；区间 `[offsets[c], offsets[c+1])` 同时切分下面三个拼接数组 |
| `voxel_index_local_zyx` | `int16 (V_total, 3)` | 候选 80³ 内的局部 ZYX 稀疏索引，逐轴位于 `[0, 80)` |
| `centered_probability` | `float32 (V_total,)` | 与稀疏索引逐体素对齐的 Stage1 概率 |
| `voxel_final` | `float16 (V_total, 48)` | 与稀疏索引逐体素对齐的 Stage1 V token 主体；第二维固定为 48 |

`voxel_offsets[-1]` 必须同时等于 `voxel_index_local_zyx`、`centered_probability` 和 `voxel_final` 的第一维长度。各适配器只读取自己需要的并列数组，但不能改变共同切分关系。

### Stage1 完整概率图

`{stage1_root}/{split_name}/{pdb_id}/probability/probability_map.npz`：

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `probability_map` | `float32 (Z_full, Y_full, X_full)` | 完整 Stage1 融合概率图，轴顺序为 ZYX，元素必须有限 |
| `origin_xyz` | `float32 (3,)` | 完整概率图低角点的世界 XYZ 坐标，单位 Å |
| `voxel_size_xyz` | `float32 (3,)` | XYZ 三轴体素尺寸，单位 Å/voxel，逐轴为有限正数 |

### occurrence、PreparedLigand 与语言向量

`data_root/parse/{pdb_id}/occurrences.jsonl` 的每个非空物理行是一个 JSON 对象。Matcher 只解释下列四个字段，未列出的附加字段原样保留在内存记录中。

| 字段 | JSON 值与转换 | 门禁和用途 |
| --- | --- | --- |
| `pdb_id` | 读取后执行 `str(value).lower()` | 必须等于当前小写目录标识；用于拒绝把其它 PDB 的记录静默混入当前目录 |
| `candidate_id` | 任意可由 `int(value)` 转换的 JSON 值 | 转换后的整数在当前 PDB 内必须唯一；用于记录排序、`mask_{candidate_id}` 字段名和语言归档文件名；本读取器不额外要求非负 |
| `object_key` | 非空 JSON 字符串 | 定位 LigandObject 模板文件，并作为联合身份第一分量 |
| `type_tag` | 读取后执行 `str(value)` | 必须是 `ion`、`nucleotide_like`、`peptide_like`、`small_molecule`、`sugar`、`other` 之一；前五项映射到同名类别，`other` 映射为 `background` |

`data_root/stage1_preparation_box_pool_2/ligand_language_models/prepared/{pdb_id}/prepared_smiles.jsonl` 每条 `PreparedLigand` 的正式顶层字段如下。文件可以整体缺失；文件一旦存在，九个字段必须恰好齐全，并覆盖 occurrence 主表的全部五类前景编号。

| 字段 | JSON 类型 | 含义与门禁 |
| --- | --- | --- |
| `pdb_id` | 非空字符串 | 必须逐字等于当前小写 PDB 标识 |
| `candidate_id` | 非负整数 | 必须对应一个五类前景 occurrence，文件内不得重复 |
| `object_key` | 非空字符串 | 必须逐字等于同编号 occurrence 的模板键 |
| `kind` | 非空字符串 | 通常为 CCD 或 BRANCHED；Matcher 不用它定义身份 |
| `type_tag` | 字符串 | 必须逐字等于同编号 occurrence 的类别来源标签 |
| `is_covalent` | 布尔值 | 受体共价连接事实；Matcher 不用它定义身份 |
| `smiles` | 非空字符串或 `null` | 交给语言模型公开接口的顶层值；非空时逐字成为联合身份第二分量 |
| `smiles_source` | 枚举字符串 | `ccd`、`clc`、`present_graph`、`unsupported` 或 `occurrence_error` |
| `diagnostics` | JSON 对象 | 转换与回退诊断；`used_raw_smiles=true` 表示顶层 `smiles` 是非空原文回退，任何诊断字段都不参与身份 |

`data_root/stage1_preparation_box_pool_2/ligand_language_models/{family}/{pdb_id}/candidate_{candidate_id}.npz` 中，`family` 为 `molformer` 或 `smi_ted_289m`：

| 字段 | dtype 与形状 | 含义与门禁 |
| --- | --- | --- |
| `pdb_id` | Unicode 标量 | 与当前 PDB 核对 |
| `candidate_id` | 可由 `int` 转换的标量；读取器不额外限制来源整数 dtype | 转换后的值与文件名和 occurrence 编号核对 |
| `object_key` | Unicode 标量 | 与联合身份第一分量核对 |
| `model_name` | Unicode 标量 | 与语言家族的固定公开模型名核对 |
| `prepared_smiles` | Unicode 标量 | 与联合身份顶层 `smiles` 逐字核对 |
| `model_smiles` | 非空 Unicode 标量 | 实际 tokenizer 输入诊断；可用于发现同身份的多字符串情况，但不定义身份 |
| `embedding` | 数值数组 `(768,)` | 读取为 `float32`，所有元素必须有限 |

### 密度、受体与真实配体体素

| 文件与字段 | dtype 与形状 | 用途 |
| --- | --- | --- |
| `density/{pdb_id}/exp.npz:grid` | `float32 (1, Z_full, Y_full, X_full)` | 完整实验密度 |
| `density/{pdb_id}/exp.npz:voxel_size` | `float32 (3,)` | XYZ 体素尺寸，单位 Å/voxel |
| `density/{pdb_id}/exp.npz:origin` | `float32 (3,)` | 完整网格低角点的世界 XYZ 坐标，单位 Å |
| `density/{pdb_id}/sim.npz:grid` | `float32 (1, Z_full, Y_full, X_full)` | 完整受体模拟密度；形状与几何必须和实验密度完全一致 |
| `density/{pdb_id}/sim.npz:voxel_size` | `float32 (3,)` | 与实验密度逐值相同的 XYZ 体素尺寸 |
| `density/{pdb_id}/sim.npz:origin` | `float32 (3,)` | 与实验密度逐值相同的世界 XYZ 原点 |
| `parse/{pdb_id}/receptor_tokens.npz:coords` | `float32 (N_atom, 3)` | 受体重原子的世界 XYZ 坐标，单位 Å |
| `parse/{pdb_id}/receptor_tokens.npz:feat` | `float32 (N_atom, 49)` | 与 `coords` 逐原子对齐的 49 维特征 |
| `density/{pdb_id}/ligand_area.npz:mask_{candidate_id}` | `int32 (V_occurrence, 3)` | 当前 occurrence 在完整密度图中的稀疏 ZYX 真值体素；允许 `V_occurrence=0`；清单读取时规范化为 `int64` |
| `density/{pdb_id}/ligand_area.npz:union_mask` | `bool (1, Z_full, Y_full, X_full)` | `True` 或数值语义上的 1 表示至少属于一个 `mask_{candidate_id}`，`False` 或 0 表示不属于任何 occurrence；它是全部稀疏 occurrence 掩码的并集，本数据适配层故意不解压该字段 |

`receptor_tokens.npz:feat` 的 49 列顺序固定如下，改变顺序会改变体素输入后 49 通道的含义。

| 切片 | dtype 与形状 | 固定顺序与数值语义 |
| --- | --- | --- |
| `feat[:, 0:6]` | `float32 (N_atom, 6)` | 元素独热，依次为 `C`、`N`、`O`、`S`、`P`、`X`；其它元素归入 `X` |
| `feat[:, 6:31]` | `float32 (N_atom, 25)` | 残基独热，依次为 `ALA`、`ARG`、`ASN`、`ASP`、`CYS`、`GLN`、`GLU`、`GLY`、`HIS`、`ILE`、`LEU`、`LYS`、`MET`、`PHE`、`PRO`、`SER`、`THR`、`TRP`、`TYR`、`VAL`、`A`、`U`、`C`、`G`、`X` |
| `feat[:, 31:39]` | `float32 (N_atom, 8)` | 残基理化标志，依次为极性、非极性、酸性、碱性、中性、正电、负电、无电 |
| `feat[:, 39]` | `float32 (N_atom,)` | 元素质量除以 32；未知元素 `X` 使用 `14.0 / 32.0` |
| `feat[:, 40]` | `float32 (N_atom,)` | 距离不大于 2 Å 的其它原子数执行 `log1p`；查询原子自身从计数中减去 |
| `feat[:, 41:49]` | `float32 (N_atom, 8)` | 依次为 `(2,4]`、`(4,6]`、`(6,8]`、`(8,10]`、`(10,12]`、`(12,14]`、`(14,16]`、`(16,18]` Å 壳层内其它原子数的 `log1p` |

`labels/{pdb_id}/atom_labels.npz` 是清单纳入 PDB 的必要存在性产物，但 `matcher/data/` 当前不读取其中字段。

## 清单目录与主清单字段

正式目录是 `output_root/manifests/closed_set_v4/`。其中 `manifest.json` 的所有顶层字段如下。

| 字段 | JSON 类型 | 含义 |
| --- | --- | --- |
| `schema_version` | 整数 | 固定为 4 |
| `created_at_utc` | 字符串 | 首次建立该 schema 清单的 UTC ISO 8601 时间；扩展划分时保留 |
| `updated_at_utc` | 字符串 | 本次原子发布的 UTC ISO 8601 时间 |
| `data_root` | 字符串 | 冻结 A–G 数据根路径文本 |
| `stage1_root` | 字符串 | Stage1 F1 候选根路径文本 |
| `output_root` | 字符串 | Matcher 输出根路径文本 |
| `coverage_threshold` | 浮点数 | 候选与 occurrence 两个方向覆盖率共同采用的下限 |
| `selection_seed` | 整数 | 联合身份内选择 occurrence 语言归档的确定性种子 |
| `identity_key_fields` | 字符串列表 | 顺序固定为 `object_key`、`prepared_smiles` |
| `prepared_smiles_rule` | 字符串 | 顶层 `smiles`、显式 `null` 和缺失 PreparedLigand 文件的身份规则说明 |
| `identity_class_rule` | 字符串 | 同一 PDB 内联合身份只能映射到一个类别；跨 PDB 只诊断 |
| `split_pdb_rule` | 字符串 | 同一小写 PDB 的候选不得同时发布到多个划分 |
| `cross_pdb_joint_identity_class_variants` | 对象列表 | 跨 PDB 同联合键出现多个局部类别时的条件诊断；没有差异时为空列表 |
| `language_selection_rule` | 字符串 | 有效 occurrence 排序、稳定盐值和循环选择方式 |
| `seen_identity_rule` | 字符串 | 各语言家族依据 train 可监督前景正目标判断 seen 的规则 |
| `class_names` | 字符串列表 | 固定顺序为 `background`、`ion`、`nucleotide_like`、`peptide_like`、`small_molecule`、`sugar` |
| `available_splits` | 字符串列表 | 已物化划分，按 train、validation、calibration 稳定排序 |
| `deferred_splits` | 字符串列表 | 尚未物化划分，顺序同上 |
| `splits` | 对象 | 已物化划分到两个 JSONL 文件名的映射 |
| `excluded_pdb` | 字符串 | 固定为 `excluded_pdb.jsonl` |
| `split_pdb_counts` | 对象 | 每个已物化划分的 PDB 目录发现、纳入和排除计数 |
| `summary` | 对象 | 候选、类别、身份、语言可用性、seen 和排除原因汇总 |

`splits` 和 `split_pdb_counts` 的嵌套字段：

| 字段 | JSON 类型 | 含义 |
| --- | --- | --- |
| `splits.{split_name}.candidates` | 字符串 | `{split_name}_candidates.jsonl` 的相对文件名 |
| `splits.{split_name}.identities` | 字符串 | `{split_name}_identities.jsonl` 的相对文件名 |
| `split_pdb_counts.{split_name}.discovered` | 整数 | Stage1 划分目录下发现的全部 PDB 目录数 |
| `split_pdb_counts.{split_name}.included` | 整数 | 成功扫描并进入候选/身份文件的 PDB 数 |
| `split_pdb_counts.{split_name}.excluded` | 整数 | 因允许原因整 PDB 排除的目录数 |

`cross_pdb_joint_identity_class_variants[*]` 只在一个联合键跨至少两个 PDB 且观察到多个类别时出现：

| 字段 | JSON 类型 | 含义 |
| --- | --- | --- |
| `object_key` | 字符串 | 跨 PDB 比较的模板图键 |
| `prepared_smiles` | 字符串 | 跨 PDB 比较的顶层公开接口字符串 |
| `classes` | 对象列表 | 不同类别，按 `class_name`、`class_index` 排序 |
| `classes[*].class_name` | 字符串 | 五类非背景类别名 |
| `classes[*].class_index` | 整数 | 对应六分类索引 |
| `locations` | 对象列表 | 该联合键的划分/PDB 位置，按 `split`、`pdb_id` 排序 |
| `locations[*].split` | 字符串 | 身份记录所在划分 |
| `locations[*].pdb_id` | 字符串 | 身份记录所在 PDB |
| `locations[*].class_name` | 字符串 | 该 PDB 内的局部类别名 |
| `locations[*].class_index` | 整数 | 该 PDB 内的类别索引 |
| `locations[*].occurrence_count` | 整数 | 共享该 PDB 内联合身份的 occurrence 数 |

## 候选 JSONL

`{split_name}_candidates.jsonl` 每个非空物理行表示一个 Stage1 F1 候选，按 PDB 扫描顺序和 `candidate_index` 递增顺序发布。

| 字段 | JSON 类型 | 含义 |
| --- | --- | --- |
| `split` | 字符串 | Stage1 划分名 |
| `pdb_id` | 字符串 | 小写 PDB 标识 |
| `candidate_index` | 整数 | F1 候选轴中的零基索引 |
| `source_tree_id` | 整数 | F1 来源连通树编号 |
| `source_node_id` | 整数 | F1 来源树节点编号 |
| `source_threshold_grid_index` | 整数 | F1 来源阈值网格索引 |
| `full_candidate_voxel_count` | 整数 | 当前候选 80³ 中的稀疏体素数 |
| `class_name` | 字符串 | 六分类名称；未通过阈值或最佳 `type_tag=other` 时为 `background` |
| `class_index` | 整数 | `class_name` 在 `class_names` 中的索引 |
| `matched_candidate_id` | 整数或 `null` | 最佳 occurrence 编号；未命中为 `null`，命中 `other` 时仍保留 |
| `matched_object_key` | 字符串或 `null` | 最佳 occurrence 的模板键；未命中为 `null` |
| `matched_prepared_smiles` | 字符串或 `null` | 仅五类前景命中可关联顶层 `smiles`；背景、缺文件或显式 `null` 都写 `null` |
| `matched_prepared_smiles_status` | 枚举字符串 | `available`、`explicit_null`、`missing_file` 或 `not_applicable` |
| `candidate_coverage` | 浮点数 | 交集唯一体素数除以候选唯一体素数，范围 `[0, 1]` |
| `occurrence_coverage` | 浮点数 | 交集唯一体素数除以 occurrence 唯一体素数，范围 `[0, 1]` |
| `overlap_score` | 浮点数 | 两个覆盖率乘积的平方根，范围 `[0, 1]` |
| `language_target_available` | 对象 | 每个语言家族是否同时建立前景联合身份、存在对应 LigandObject 文件并具有有效语言归档；这里只检查图文件存在性，图内容由 `LigandGraphStore` 后验核验 |
| `language_target_available.molformer` | 布尔值 | 上述三个条件在 MoLFormer 语言家族下是否同时成立 |
| `language_target_available.smi_ted_289m` | 布尔值 | 上述三个条件在 SMI-TED 289M 语言家族下是否同时成立 |
| `identity_seen_in_train_by_language` | 对象 | 各语言家族的训练可见性；不具备前景联合身份时两个值均为 `null` |
| `identity_seen_in_train_by_language.molformer` | 布尔值或 `null` | MoLFormer 监督口径下的 seen |
| `identity_seen_in_train_by_language.smi_ted_289m` | 布尔值或 `null` | SMI-TED 289M 监督口径下的 seen |

候选与 occurrence 的稀疏 ZYX 体素先去重。只有 `candidate_coverage` 和 `occurrence_coverage` 都达到 `coverage_threshold` 才能命中；多个 occurrence 通过时先最大化 `overlap_score`，再选择较小 `candidate_id`。

## 身份 JSONL 与语言子对象

`{split_name}_identities.jsonl` 每个非空物理行表示一个 PDB 内的五类前景联合身份。

| 字段 | JSON 类型 | 含义 |
| --- | --- | --- |
| `split` | 字符串 | 当前 Stage1 划分名 |
| `pdb_id` | 字符串 | 限定本地身份轴的小写 PDB 标识 |
| `class_name` | 字符串 | 共享该联合身份的唯一五类前景类别 |
| `class_index` | 整数 | 与 `class_name` 对齐的六分类索引 |
| `object_key` | 字符串 | 联合身份第一分量和模板图定位键 |
| `prepared_smiles` | 非空字符串 | 联合身份第二分量，逐字来自 PreparedLigand 顶层 `smiles` |
| `occurrence_candidate_ids` | 整数列表 | 共享该联合身份的 occurrence 编号，严格递增 |
| `graph_available` | 布尔值 | `graph_relative_path` 指向的文件是否存在；图内容由读取器另行核验 |
| `graph_relative_path` | 字符串 | 相对 `data_root` 的 LigandObject NPZ 路径 |
| `language` | 对象 | `molformer` 和 `smi_ted_289m` 两个家族的选择与失败证据 |
| `seen_in_train_by_language` | 对象 | 每个语言家族下，该联合键是否在 train 候选中作为可监督前景正目标出现；查询键不含 `pdb_id`，因此相同联合键跨 PDB 共享 seen 事实，但不共享身份轴 |
| `seen_in_train_by_language.molformer` | 布尔值 | MoLFormer 监督口径下的 seen |
| `seen_in_train_by_language.smi_ted_289m` | 布尔值 | SMI-TED 289M 监督口径下的 seen |

`language.{family}` 对两个语言家族使用相同字段：

| 字段 | JSON 类型 | 含义与条件 |
| --- | --- | --- |
| `available` | 布尔值 | 至少一个 occurrence 语言归档通过全部门禁时为 true |
| `selected_candidate_id` | 整数或 `null` | `available=true` 时确定性选中的 occurrence 编号，否则为 `null` |
| `selected_relative_path` | 字符串或 `null` | 选中归档相对 `language_root` 的路径；不可用时为 `null` |
| `valid_candidate_ids` | 整数列表 | 按身份内 occurrence 顺序保留的全部有效归档编号 |
| `failed_occurrences` | 对象列表 | 缺失或无效归档的诊断，保持 occurrence 顺序 |
| `failed_occurrences[*].candidate_id` | 整数 | 失败归档对应的 occurrence 编号 |
| `failed_occurrences[*].reason` | 枚举字符串 | `missing_file` 或 `invalid_archive` |
| `failed_occurrences[*].detail` | 字符串 | 仅 `reason=invalid_archive` 时出现，保存捕获的核验错误文本 |
| `distinct_model_smiles` | 字符串列表 | 有效归档中 `model_smiles` 精确去重后的排序列表；只作 tokenizer 输入诊断 |

选择位置由 `selection_seed` 和稳定文本 `(pdb_id, object_key, language_family)` 的 UTF-8 加权和共同决定；稳定文本故意不加入 `prepared_smiles`，使未发生拆分的旧身份保持原 occurrence 选择位置。

## 排除 JSONL

`excluded_pdb.jsonl` 每个非空物理行表示一个整 PDB 排除事实。已经存在但内容损坏的必要文件不会降级成 `missing_required_product` 排除；被实际读取的字段在 manifest 扫描或对应缓存消费者打开时失败。`atom_labels.npz` 在 `matcher/data` 本层只执行 `is_file()` 存在性检查，不打开内容，因此其内容损坏不会由本层发现。

| 字段 | JSON 类型 | 含义与条件 |
| --- | --- | --- |
| `split` | 字符串 | 被排除目录所在 Stage1 划分 |
| `pdb_id` | 字符串 | 被排除的小写 PDB 标识 |
| `reason` | 枚举字符串 | `_BLOB_EXCEED` 标记存在时为 `blob_exceed`，必要产物缺失时为 `missing_required_product` |
| `missing_paths` | 字符串列表 | 仅 `reason=missing_required_product` 时出现；按必要文件检查顺序保存由调用者提供的 `data_root`、`stage1_root` 拼出的不存在路径文本，不保证是绝对路径 |

必要文件依次是 F1 候选、完整概率图、occurrence 主表、受体 token、实验密度、模拟密度、真实配体区域和原子标签。PreparedLigand 文件允许整体缺失；LigandObject 图和语言向量按身份级可用性记录，所以三者都不触发整 PDB 排除。

## summary 字段

`manifest.json:summary` 的顶层字段：

| 字段 | JSON 类型 | 含义 |
| --- | --- | --- |
| `splits` | 对象 | 每个已物化划分的候选、类别、身份和语言计数 |
| `excluded_pdb_reason_counts` | 对象 | 实际出现的排除原因到 PDB 数的映射，键按原因排序 |
| `excluded_pdb_reason_counts.blob_exceed` | 整数 | 条件字段；存在该原因时的 PDB 数 |
| `excluded_pdb_reason_counts.missing_required_product` | 整数 | 条件字段；存在该原因时的 PDB 数 |

`summary.splits.{split_name}` 的全部字段：

| 字段 | JSON 类型 | 含义 |
| --- | --- | --- |
| `candidate_count` | 整数 | 当前划分候选记录总数 |
| `class_counts` | 对象 | 实际出现的候选类别计数；未出现的类别键省略 |
| `class_counts.{class_name}` | 整数 | 条件字段；`background`、`ion`、`nucleotide_like`、`peptide_like`、`small_molecule`、`sugar` 中实际出现类别的数量 |
| `identity_count` | 整数 | 顶层 `prepared_smiles` 非空并建立的 PDB 内身份记录数 |
| `unique_object_key_count` | 整数 | 身份记录中不同模板键的数量 |
| `same_pdb_object_key_multi_prepared_smiles_group_count` | 整数 | 同一 PDB、同一 `object_key` 拆成两个或更多顶层字符串身份的分组数 |
| `matched_foreground_without_prepared_smiles_candidate_count` | 整数 | 五类前景命中但 `matched_prepared_smiles=null` 的候选数，合并缺文件和显式 `null` |
| `matched_foreground_explicit_null_candidate_count` | 整数 | 五类前景命中且顶层 `smiles` 显式为 `null` 的候选数 |
| `matched_foreground_missing_prepared_file_candidate_count` | 整数 | 五类前景命中但整个 PreparedLigand 文件缺失的候选数 |
| `language_available_identity_count` | 对象 | 每个语言家族至少有一个有效归档的身份数 |
| `language_available_identity_count.molformer` | 整数 | MoLFormer 可用身份数 |
| `language_available_identity_count.smi_ted_289m` | 整数 | SMI-TED 289M 可用身份数 |
| `language_multi_model_smiles_identity_count` | 对象 | 每个语言家族中，有效归档出现多个精确 `model_smiles` 的身份数 |
| `language_multi_model_smiles_identity_count.molformer` | 整数 | MoLFormer 多 tokenizer 字符串诊断数 |
| `language_multi_model_smiles_identity_count.smi_ted_289m` | 整数 | SMI-TED 289M 多 tokenizer 字符串诊断数 |
| `seen_positive_identity_count` | 对象 | 每个语言家族中 `seen_in_train_by_language=true` 的身份数 |
| `seen_positive_identity_count.molformer` | 整数 | MoLFormer 监督口径 seen 身份数 |
| `seen_positive_identity_count.smi_ted_289m` | 整数 | SMI-TED 289M 监督口径 seen 身份数 |

## 坐标、裁剪与体素缓存

密度网格、稀疏体素索引和裁剪起点使用 ZYX；世界坐标、世界原点和体素尺寸使用 XYZ。世界坐标单位为 Å，体素尺寸单位为 Å/voxel。

Stage1 候选框固定为 80³，Matcher 局部输入固定为 48³。`fixed_center` 请求起点 `(16, 16, 16)`；`v_centroid` 对完整候选 V 稀疏体素求均值，以 NumPy ties-to-even 的 `rint(mean + 0.5 - 24)` 解析请求起点，再逐轴限制到 `[0, 32]`。缓存路径保存实际起点，防止不同裁剪结果静默复用。

候选级体素缓存路径为：

```text
output_root/cache/voxel_input_v2/{crop_strategy}/{split_name}/{pdb_id}/candidate_{candidate_index}_start_{z}_{y}_{x}.npy
```

这个 NPY 没有命名字段，只保存一个 `float16 (105, 48, 48, 48)` 数组；轴顺序为通道、Z、Y、X。`float16` 是正式落盘量化，不是读取器临时转换。前 56 个通道与 `DENSITY_CHANNEL_NAMES` 逐位置对齐；后 49 个通道把受体原子特征按 `floor` 得到的 80³ 体素归属逐通道求和后裁成 48³。`build_density_channels` 的低层接口只要求实验密度、模拟密度和受体硬掩码是三个同形、非空的 `(Z, Y, X)` 网格；`VoxelInputStore` 的正式调用传入 80³。

前 56 个密度通道按三层嵌套顺序生成。基础运算顺序是 `exp`、`sim`、`diff`、`posdiff`；每种运算内部的归一化顺序是 `nonorm`、`clipnorm`；每种归一化内部的后处理顺序是 `nopost`、`gauss1`、`gauss2`、`DoG1`、`DoG2`、`smooth1`、`smooth2`。若三层零基序号分别为 `o`、`n`、`p`，则缓存通道号为 `14 * o + 7 * n + p`。

| 层级值 | 公式与固定参数 |
| --- | --- |
| `exp` | 原始实验密度 `experimental` |
| `sim` | 原始受体模拟密度 `simulated`；本通道不应用后述线性拟合 |
| `diff` | `experimental - (scale * simulated + offset)`；`scale`、`offset` 由受体优先的高密度体素最小二乘拟合 |
| `posdiff` | `max(diff, 0)`，逐体素截去负值 |
| `nonorm` | 保持对应基础运算值不变 |
| `clipnorm` | 先按当前三维网格的 0.1% 和 99.9% 分位逐元素截断，再计算 `(clipped - mean) / (std + 1e-8)` |
| `nopost` | 保持归一化后的网格不变 |
| `gauss1` | SciPy 反射边界 Gaussian 滤波，`sigma=1` voxel |
| `gauss2` | SciPy 反射边界 Gaussian 滤波，`sigma=2` voxels |
| `DoG1` | `Gaussian(sigma=1.6) - Gaussian(sigma=1.0)` |
| `DoG2` | `Gaussian(sigma=3.2) - Gaussian(sigma=2.0)` |
| `smooth1` | `base * (1 - strength)`；`strength=exp(-distance²/(2*1²))`，受体硬掩码含 True 时 `distance` 是到最近受体占据体素的欧氏距离，超过 2 voxels 后置 0 |
| `smooth2` | `base * (1 - strength)`；`strength=exp(-distance²/(2*2²))`，受体硬掩码含 True 时沿用相同距离定义，超过 2 voxels 后置 0 |

受体硬掩码允许全为 False，例如当前 PDB 没有受体原子时。此时不存在“最近受体占据体素”，实现保留 SciPy `distance_transform_edt` 对无内部零点输入的既有距离结果，再应用相同 Gaussian 公式和 2 voxels 截断。

`diff` 的线性拟合先取实验密度和模拟密度都达到各自 98.5% 分位的体素，再尝试只保留其中的受体占据体素。受体优先集合至少包含 `max(floor(0.003 * 网格总体素数), 10)` 个体素时使用该集合，否则退回全部高密度交集。最终集合少于 10 个体素、模拟密度方差小于 `1e-12` 或去均值回归分母小于 `1e-12` 时使用 `scale=1`、`offset=0`；其它情况用带截距的一元最小二乘斜率，并令 `offset=experimental_mean-scale*simulated_mean`。

后 49 个通道与上文 `receptor_tokens.npz:feat[:, 0:49]` 逐列对齐。同一 80³ 体素内若落入多枚受体原子，49 维特征逐列求和；随后才按实际裁剪起点取得 48³。

## PP 缓存字段

PDB 级 PP 缓存路径为：

```text
output_root/cache/pp_input_v2/{crop_strategy}/top_{top_k}/{split_name}/{pdb_id}.npz
```

C 是当前 PDB 的 F1 候选数，K 是 `min(top_k, 48³)`。缓存采用固定 `(C, K, ...)` 布局，不使用候选 offsets。

| 字段 | dtype 与形状 | 对齐、范围与含义 |
| --- | --- | --- |
| `schema_version` | `int64` 标量 | 固定等于 PP 缓存 schema 版本 2 |
| `top_k` | `int64` 标量 | 构建时的正整数 token 上限，必须与路径和调用参数相同 |
| `candidate_indices` | `int64 (C,)` | 严格等于 `0..C-1`，与 F1 候选轴逐位置一致，C 必须大于 0 |
| `tokens` | `float32 (C, K, 4)` | 最后一维的四个分量依次为概率原值、relative_x、relative_y、relative_z；全部有限 |
| `index_zyx` | `int16 (C, K, 3)` | 前两维与 `tokens` 对齐；表示各 token 在当前 48³ 内的 ZYX 索引，逐轴位于 `[0, 48)` |
| `crop_start_zyx` | `int16 (C, 3)` | 与 `candidate_indices` 逐候选对齐；48³ 在候选 80³ 内的实际起点，逐轴位于 `[0, 32]` |
| `probability_origin` | `float32 (3,)` | 当前 PDB 完整概率图低角点的世界 XYZ 坐标，单位 Å，全部有限 |
| `probability_voxel_size` | `float32 (3,)` | 完整概率图 XYZ 体素尺寸，单位 Å/voxel，全部有限且逐轴大于 0 |

每个候选先用 `argpartition` 选出 48³ 内的 K 个最高概率体素；入选 token 再按概率降序、NumPy C-order（行主序）展平索引升序排列。第 K 名概率并列时，入选子集沿用当前 NumPy `argpartition` 语义。`relative_xyz` 是 token 体素中心相对实际 48³ 中心的世界坐标差，再逐轴除以 24 个体素对应的物理半边长，因此写入的三个值无量纲。

模型注意力偏置使用的四个特征分量顺序是 `relative_x`、`relative_y`、`relative_z`、`probability`，与 PP token 的四列顺序不同。PP 缓存不保存 mask；其中每个候选恰有 K 个已选 token。`PpInputStore.load` 返回单候选时，把落盘的 `index_zyx` 和 `crop_start_zyx` 从 `int16` 转为 `int64`，其余数组保持正式 dtype。

## LigandObject 模板图

`data_root/ligand_objects/{safe_object_filename(object_key)}.npz` 由 `object_key` 定位。Matcher 使用的来源字段如下。

`N_atom` 是 `atoms` 的第一维长度，必须大于 0；`N_bond` 是 `bonds` 的第一维长度，必须大于或等于 0。来源 schema 规定 `atoms.is_present` 固定为 False、`atoms.chirality` 为手性类别独热、`bonds.type` 为键类型独热；Matcher 读取器只将这些字段转换为正式 dtype，不额外核验固定值或每项恰有一列为 True。

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `name` | Unicode 标量 | 必须逐字等于请求的 `object_key`，用于阻止安全文件名映射到错误模板 |
| `atoms.element` | `int8 (N_atom,)` | 节点元素编号 |
| `atoms.charge` | `int8 (N_atom,)` | 节点形式电荷 |
| `atoms.ref_pos` | `float32 (N_atom, 3)` | 模板参考 XYZ 坐标，单位 Å |
| `atoms.is_present` | `bool (N_atom,)` | 来源 schema 定义为固定 False；读取器通过 `np.asarray(..., dtype=float32)` 转成节点连续特征但不核验全为 False；它不是 occurrence 掩码，也不定义身份 |
| `atoms.chirality` | `bool (N_atom, 7)` | 来源 schema 定义为七种手性类别独热；读取器转换为 `float32` 但不核验每个原子恰有一列为 True |
| `atoms.in_ring` | `bool (N_atom, 4)` | 节点环大小标志 |
| `atoms.residue_id` | `int32 (N_atom,)` | 模板内部残基编号 |
| `bonds.atom_1` | `int32 (N_bond,)` | 原始键第一端点索引 |
| `bonds.atom_2` | `int32 (N_bond,)` | 原始键第二端点索引 |
| `bonds.type` | `bool (N_bond, 5)` | 来源 schema 定义为单/双/三/配位/芳香键独热；读取器转换为 `float32` 但不核验每条键恰有一列为 True |
| `bonds.in_ring` | `bool (N_bond, 4)` | 键环大小标志 |

读取器把来源字段转换成下列内存 `LigandGraph`。`N_directed_edge=2*N_bond`；`N_bond=0` 时，`edge_index` 的形状为 `(2, 0)`，`edge_features` 的形状为 `(0, 10)`。

| 字段 | dtype 与形状 | 对齐与固定顺序 |
| --- | --- | --- |
| `element` | `int64 (N_atom,)` | 逐模板原子序数，保持 `atoms` 顺序 |
| `charge` | `int64 (N_atom,)` | 与 `element` 逐原子对齐的形式电荷 |
| `residue_id` | `int64 (N_atom,)` | 模板内部残基编号，数值位于 `[0, 512)` |
| `node_continuous` | `float32 (N_atom, 15)` | `0:7` 逐值转换来源手性字段，来源 schema 定义其为独热但读取器不核验；`7:11` 为 3/4/5/6 元环标志；`11` 把来源 schema 定义为固定 False 的模板 `atoms.is_present` 转成 `float32`，读取器不核验全为 False；`12:15` 为每个 `residue_id` 组内中心化后再除以 10 Å 的参考 X/Y/Z |
| `edge_index` | `int64 (2, N_directed_edge)` | 第 0 行为源端点、第 1 行为目标端点；先保存全部 `atom_1→atom_2`，再保存全部反向边，第 `b` 条原始键的反向边位于 `b+N_bond` |
| `edge_features` | `float32 (N_directed_edge, 10)` | `0:5` 逐值转换来源键类型字段，来源 schema 定义其为单/双/三/配位/芳香键独热但读取器不核验；`5:9` 为 3/4/5/6 元环标志；`9` 为参考距离除以 10 Å；正反边共享特征 |

同残基边的距离来自模板参考坐标。跨 `residue_id` 边仍保留拓扑、键型和环标志，但距离列固定为 0，因为不同组件参考构象没有共同坐标系。

## 失败关闭与并发

- 同一小写 PDB 的候选不得进入多个已物化划分；排除记录不属于候选集合。
- 同一 PDB 内的联合身份只能映射到一个五类前景类别；跨 PDB 类别差异只诊断。
- 语言归档必须匹配 PDB、occurrence 编号、`object_key`、顶层 `prepared_smiles`、模型名和有限 `(768,)` 向量；无效归档写入身份的失败证据。
- F1、密度、概率图、受体数组和缓存的形状、dtype、范围、有限值及空间几何不满足契约时直接失败。
- 同一路径缓存缺失时先取得进程文件锁，再二次检查；写入使用同目录临时文件、刷盘和原子替换。
- `precompute_inputs` 只读既有 manifest，并以 PDB 为进程工作项；`future.result()` 在父进程重新抛出失败，函数不吞掉子进程失败，也不会在失败后成功返回。只有命令行 `main` 会在预计算前调用 `ensure_manifest`。
