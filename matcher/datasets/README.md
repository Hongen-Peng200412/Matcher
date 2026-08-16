# `matcher/datasets` 数据集与物理批次契约

## 中心职责

本目录把 [`matcher/data`](../data/README.md) 已经定义的冻结文件和缓存转换成训练、validation 与 calibration 共用的内存对象：

- `candidate_data.py` 按候选 JSONL 的顺序建立 `CandidateDataset`，每次返回一个 `CandidateItem`。
- `contracts.py` 集中声明 `CandidateMetadata`、`CandidateItem` 和 `CandidateBatch` 的字段。
- `batching.py` 决定候选下标顺序，把四种变长 token 和身份轴补零，并把每个身份对应的 LigandObject 图连接成批次图。

本页所称 occurrence，是一个 PDB 中的一次具体配体实例。

本目录不重新生成 manifest，不改变六分类标签，不重建 LigandObject，也不重新选择 occurrence 语言向量。联合身份始终是当前 PDB 内的 `(object_key, PreparedLigand 顶层 smiles)`；不同 PDB 不共享身份分数轴或同一次归一化。

## 符号与固定顺序

| 符号 | 含义 |
| --- | --- |
| `C` | 一个 PDB 的 Stage1 F1 候选数 |
| `B` | 一个物理批次的候选数 |
| `N_A`、`N_P` | 一个候选完整 `80³` 框中的 A、P token 数 |
| `N_V` | 一个候选实际 `48³` 裁剪内的 V token 数 |
| `K` | 每候选固定 PP token 数，等于 `min(pp_top_k, 48³)` |
| `I_b` | 批次中第 `b` 个候选的有效联合身份数 |
| `I_max` | 一个批次中最大的 `I_b`，可为零 |
| `G` | 一个批次的身份图总数，等于 `sum(I_b)` |
| `N_node` | `G` 张图连接后的节点总数 |
| `N_directed_edge` | `G` 张图连接后的有向边总数 |

六分类下标顺序固定如下；所有 `class_target` 和 `identity_class_index` 都按此解释。

| 下标 | 类别 |
| ---: | --- |
| 0 | `background` |
| 1 | `ion` |
| 2 | `nucleotide_like` |
| 3 | `peptide_like` |
| 4 | `small_molecule` |
| 5 | `sugar` |

## `CandidateDataset` 构造契约

### 输入参数

| 参数 | 类型与允许值 | 作用与边界 |
| --- | --- | --- |
| `roots.data_root` | `Path` | 本 Dataset 直接读取 `atom_labels.npz:binding_atom`、`receptor_tokens.npz:coords` 与 `ligand_area.npz`；委托 `VoxelInputStore` 读取 `density/{pdb_id}/exp.npz`、`density/{pdb_id}/sim.npz` 和 `parse/{pdb_id}/receptor_tokens.npz:feat`，委托 `LigandGraphStore` 读取 LigandObject，并通过派生 `language_root` 读取语言 NPZ；不直接读取 `occurrences.jsonl` |
| `roots.stage1_root` | `Path` | 定位按划分和 PDB 组织的 `F1_centered.npz` 与 `probability_map.npz` |
| `roots.output_root` | `Path` | 供 `VoxelInputStore` 与 `PpInputStore` 定位版本化 105 通道体素缓存和 PP token 缓存；缓存缺失时对应 Store 会从上述来源构建并写入，不用于推导本次 manifest |
| `manifest_path` | `Path` | 显式传入并独立定位 schema v4 `manifest.json`；候选与身份 JSONL 相对其父目录解释，不要求由 `roots.output_root` 推导 |
| `split_name` | 字符串 | 必须是 `manifest.splits` 已声明的划分 |
| `task_name` | `candidate_pretrain` 或 `full_matcher` | 决定返回空身份轴还是完整身份输入 |
| `language_family` | `molformer`、`smi_ted_289m` 或 `None` | `full_matcher` 必须指定；`candidate_pretrain` 未强制为 `None`，非空时仍可填充 metadata 的 train 可见性 |
| `identity_scope` | API 类型契约为 `small_molecule`、`all_non_background` 或 `None` | `candidate_pretrain` 构造时必须为 `None`；`full_matcher` 构造时只检查非 `None`，非法非空字符串会在首次物化身份轴时失败 |
| `pp_top_k` | 正整数 | 每候选实际 PP 数固定为 `K=min(pp_top_k, 48³)`，并进入 PP 缓存路径 |
| `local_crop_strategy` | `fixed_center` 或 `v_centroid` | 统一决定体素输入、V、PP 和两个区域目标共享的 `48³` 起点，同时移动 A/P 相对坐标原点 |
| `cache_bytes_per_worker` | 非负整数，默认 16 GiB | 单个 Dataset 实例五类 `ByteBudgetLru` 的 `array_nbytes` 记账总上限；不是进程 RSS 或全部对象内存上限，也不是预分配量，多进程时每个 worker 分别持有 |

`full_matcher` 的身份范围如下：

| `identity_scope` | 允许进入身份轴的类别下标 |
| --- | --- |
| `small_molecule` | 仅 4 |
| `all_non_background` | 1、2、3、4、5 |

### 构造时读取和检查的 manifest 字段

| 字段 | 构造期要求 |
| --- | --- |
| `schema_version` | 必须等于当前 `MANIFEST_SCHEMA_VERSION`，即联合身份 v4 |
| `identity_key_fields` | 必须严格为 `['object_key', 'prepared_smiles']`，顺序也必须相同 |
| `class_names` | 必须与本页六分类顺序完全相同 |
| `splits[split_name]` | 必须存在并提供 `candidates`、`identities` 相对路径；缺键由 JSON 访问直接失败 |

`CandidateDataset` 不在构造时完整复验 manifest 的所有摘要和诊断字段。候选 JSONL 由 `load_jsonl` 读取后保持文件顺序，该顺序就是 Dataset 全局样本轴。

### 按样本读取的候选 JSON 字段

`CandidateDataset.__getitem__` 从 `records[index]` 消费下列字段。occurrence 匹配已经冻结进候选 JSONL；本函数不重新打开 `occurrences.jsonl`，也不执行候选 JSON 的完整 schema 验证。

| 字段 | 本函数的转换或门禁 | 用途与空值边界 |
| --- | --- | --- |
| `pdb_id` | 执行 `str(value)`；不在此重新核验非空或小写 | 定位当前 PDB 的来源文件和缓存键 |
| `candidate_index` | 执行 `int(value)` | 索引 centered 候选轴、PP 缓存和体素缓存 |
| `full_candidate_voxel_count` | 执行 `int(value)`，并与当前完整 V 数量核对 | 写入输出并作为 V 保留比例的分母来源 |
| `class_index` | 执行 `int(value)`；不在此重新核验 `[0,5]` | 写入六分类目标并参与身份真值定位 |
| `matched_object_key` | 不做本地类型检查 | 有 occurrence 通过覆盖阈值时保留模板键；`type_tag=other` 映射成 `background` 时仍可非空，未命中时为 `None` |
| `matched_prepared_smiles` | `available` 时必须为非空精确字符串；其它状态必须为 `None` | 仅五类前景命中可提供联合身份第二分量；`other` 背景和未命中为 `None` |
| `matched_prepared_smiles_status` | 必须属于 `available`、`explicit_null`、`missing_file`、`not_applicable` | `other` 背景和未命中均为 `not_applicable` |
| `identity_seen_in_train_by_language` | 选择 `language_family` 时按键读取；不在此重新核验对象键或值类型 | 完整前景联合身份对应布尔值；背景、未命中或缺少 object_key/prepared_smiles 时对应值为 `None` |

`type_tag=other` 的最佳命中仍保留 `matched_object_key`，但 `class_index=0`、`matched_prepared_smiles=None`、状态为 `not_applicable`、`identity_seen_in_train=None`，因此 `identity_target` 固定为 `-1`。

### 身份 JSONL 的构造期门禁

身份记录在按 PDB 分组前检查下列字段。通过后，每个 PDB 的列表按 `(class_index, object_key, prepared_smiles)` 排序。

| 字段 | 类型与约束 | 用途 |
| --- | --- | --- |
| `split` | 必须逐字等于 `split_name` | 防止划分文件混用 |
| `pdb_id` | 非空小写精确字符串 | PDB 局部身份轴第一层分组 |
| `class_name` | 五类前景名称之一 | 背景不建立身份槽 |
| `class_index` | 精确 `int` 且与 `class_name` 固定映射一致 | 身份类别与范围筛选 |
| `object_key` | 非空精确字符串 | 联合身份第一分量和 LigandObject 图定位键 |
| `prepared_smiles` | 非空精确字符串 | 联合身份第二分量，逐字采用 manifest 冻结的顶层值 |
| `occurrence_candidate_ids` | 非空、升序、去重的非负精确整数列表 | 记录形成该身份的 occurrence 集合 |
| `graph_available` | 精确布尔值 | 为真才会在样本物化时尝试读取图 |
| `graph_relative_path` | 字符串 | 构造时只核验类型；Dataset 实际按 `object_key` 交给 `LigandGraphStore` 定位图 |
| `language` | 键集合必须恰好为两个语言家族 | 每个家族的证据对象至少必须含精确布尔 `available` |
| `seen_in_train_by_language` | 两个语言家族到精确布尔值的映射 | 构造时仅验证并保留身份记录自身的 train 可见性事实；不用于填充候选 metadata，也不与候选 JSONL 的同名字段交叉核对 |

同一 `(pdb_id, object_key, prepared_smiles)` 不得重复。构造期不会完整核验语言证据对象中的 `selected_relative_path`、计数和失败 occurrence；身份实际进入当前语言轴时才检查所需路径和 NPZ 内容。

`seen_in_train_by_language` 的计算键不含 `pdb_id`，所以不同 PDB 的相同 `(object_key, prepared_smiles)` 可以共享 train 可见性布尔值。这不合并两个 PDB 的身份槽，也不让它们进入同一次身份分数归一化。

候选 metadata 的 `identity_seen_in_train` 来自候选 JSONL 的 `identity_seen_in_train_by_language[language_family]`，不来自上表的身份记录字段。两份 JSONL 保存的可见性事实在本 Dataset 中没有运行时交叉核对。

### 每个 worker 的缓存预算

| 缓存 | 上限 | 保存内容 |
| --- | ---: | --- |
| `artifact` | 总预算减去其余四项向下取整份额后的全部余量；总数可被 32 整除时恰为 13/16 | PDB 级 centered、完整受体坐标、`binding_atom` 和 occurrence 稀疏掩码 |
| `source` | `total//16` | `VoxelInputStore` 构建 105 通道时复用的来源数组 |
| `pp` | `total//16` | `PpInputStore` 的 PDB 级固定 `K` PP 缓存 |
| `identity` | `total//32` | 已按 PDB、语言家族和范围筛选的身份字符串、类别、语言向量与图列表 |
| `graph` | `total//32` | `LigandGraphStore` 按 `object_key` 复用的模板图 |

五项整数之和严格等于总预算。`ByteBudgetLru` 的 `array_nbytes` 只递归统计 NumPy 数组、映射、序列和 dataclass 载荷；候选与身份清单、字符串与容器开销、批次张量以及其它进程内存不受该值约束。

表中的“保存内容”是允许尝试缓存的载荷，不保证每次读取都被保留；预算为零或单个载荷超过对应预算时，`put` 不会留下该对象。候选级 105 通道体素以 `output_root` 下的 NPY 为主要缓存，命中已有 NPY 时不填充来源 LRU；只有构建缺失 NPY 时才读取并尝试缓存密度和受体来源数组。

`spawn` 序列化时，`CandidateDataset.__getstate__` 只在返回状态中重建空 `artifact` 和 `identity` 缓存；`voxel_store`、`pp_store`、`graph_store` 分别让自身的序列化状态不携带内部缓存。`records` 与 `identities_by_pdb` 仍参与 worker 序列化，父进程现有对象及其缓存不被就地修改。

## 单候选物化

### PDB 级来源

`_load_pdb_artifacts` 为同一 `(split_name, pdb_id)` 读取下列对象，并在 artifact 预算允许且 `put` 保留时供后续访问复用。

| 对象 | dtype 与形状 | 读取和使用边界 |
| --- | --- | --- |
| `centered` | 字段到 NumPy 数组的映射 | 复制 `F1_centered.npz` 的二十个指定字段；复制时保留归档 dtype |
| `binding_atom` | `bool (N_receptor,)` | 来自 `atom_labels.npz`，必须与完整受体坐标逐原子对齐 |
| `receptor_coords` | `float32 (N_receptor, 3)` | 来自 `receptor_tokens.npz:coords`，世界 X/Y/Z，单位 Å，必须有限 |
| `occurrence_masks` | `list[int64 (V_occurrence, 3)]` | 收集 `ligand_area.npz` 中字段名为 `mask_` 加十进制数字的数组；列表顺序不参与并集语义 |

`centered` 中各拼接数组的候选切分关系如下：

| offsets | 同一半开区间切分的数组 |
| --- | --- |
| `voxel_offsets (C+1,)` | `voxel_index_local_zyx (V_total,3)`、`centered_probability (V_total,)`、`voxel_final (V_total,48)` |
| `P_offsets (C+1,)` | `P_coord_local_xyz (P_total,3)`、`P_probability (P_total,)`、`P_feat_L2 (P_total,64)`、`P_feat_L3 (P_total,48)` |
| `A_offsets (C+1,)` | `A_global_index (A_total,)`、`A_coord_local_xyz (A_total,3)`、`A_probability (A_total,)`、`A_feat_L0 (A_total,49)`、`A_feat_L1 (A_total,64)`、`A_feat_L2 (A_total,64)`、`A_feat_L3 (A_total,48)` |

正式来源 dtype 为：三个 offsets 均为 `int64`；`voxel_index_local_zyx` 为 `int16`；三类概率和 A/P 连续坐标为 `float32`；`voxel_final`、`P_feat_L2/L3`、`A_feat_L1/L2/L3` 为 `float16`；`A_feat_L0` 为 `float32`；`A_global_index` 为 `int64`。最终 A/P/V token 都转换为 `float32`。`A_global_index` 会随 centered 一起缓存，但当前 `CandidateItem` 不返回该字段。

### 裁剪和跨缓存几何核对

完整 V 下标 `full_v_index` 是 `int64 (N_V_full,3)` 的候选 `80³` 内 Z/Y/X。`resolve_candidate_crop` 产生实际 `48³` 起点：

- `fixed_center` 使用 `(16,16,16)`。
- `v_centroid` 以完整 V 质心请求中心，再把起点逐轴限制到 `[0,32]`。

解析后的 `full_v_count` 必须等于候选 JSONL 的 `full_candidate_voxel_count`。随后 `_validate_candidate_geometry` 核对：

- `F1_centered.voxel_size_world[candidate_index]` 与 PP 缓存保存的 `probability_voxel_size` 逐元素相等。
- PP 缓存的 `crop_start_zyx` 与 Dataset 重新解析的起点逐元素相等。
- `box_origin_world` 必须等于 `probability_origin + box_start_zyx` 按 X/Y/Z 重排后乘体素尺寸；float32 重算允许的绝对误差上限为 `BOX_ORIGIN_ABSOLUTE_TOLERANCE_ANGSTROM`。

### 四种 token

四种 token 的概率都是 Stage1 固化原值，手动特征使用前必须有限且位于 `[0,1]`。A/P 不按 `48³` 裁剪筛除；它们只把坐标原点移动到实际裁剪中心。V/PP 只保留实际裁剪内位置。

| 模态 | dtype 与形状 | 固定特征顺序 | 坐标归一化 |
| --- | --- | --- | --- |
| A | `float32 (N_A,229)` | `0:49=L0`；`49:113=L1`；`113:177=L2`；`177:225=L3`；`225=probability`；`226:229=relative X/Y/Z` | 相对实际裁剪中心的世界坐标除以完整 `80³` 的 40 体素物理半边长 |
| P | `float32 (N_P,116)` | `0:64=L2`；`64:112=L3`；`112=probability`；`113:116=relative X/Y/Z` | 相对实际裁剪中心的世界坐标除以完整 `80³` 的 40 体素物理半边长 |
| V | `float32 (N_V,52)` | `0:48=voxel_final`；`48=probability`；`49:52=relative X/Y/Z` | 相对实际裁剪中心的世界坐标除以实际 `48³` 的 24 体素物理半边长 |
| PP | `float32 (K,4)` | `0=probability`；`1:4=relative X/Y/Z` | 相对实际裁剪中心的世界坐标除以实际 `48³` 的 24 体素物理半边长 |

`v_index_zyx` 为 `int64 (N_V,3)`，`pp_index_zyx` 为 `int64 (K,3)`；两者最后一轴都是裁剪内 Z/Y/X，逐轴位于 `[0,48)`，并与对应 token 第一轴逐项对齐。

### 两个区域目标

| 字段 | dtype 与形状 | True 的含义 |
| --- | --- | --- |
| `ligand_area_target` | `bool (48,48,48)`，轴为 Z/Y/X | 当前 PDB 任一 `mask_数字` occurrence 稀疏掩码覆盖该裁剪内体素 |
| `binding_area_target` | `bool (48,48,48)`，轴为 Z/Y/X | 完整受体表中至少一个 `binding_atom=True` 的原子经世界坐标换算后落入该体素 |

配体目标使用当前 PDB 全部 occurrence 掩码的并集，不只使用候选命中的 occurrence。空掩码列表产生全 False。`_build_ligand_target` 没有单独验证每个 mask 的二维三列形状；部分畸形输入可能经 NumPy 广播后被接受，正式 `int64 (V_occurrence,3)` 契约必须由生产侧保证，不能依赖本函数失败关闭。

结合目标不能由 A token 构造：A 只是来源组件周围的筛选表，完整受体坐标与 `binding_atom` 才是监督对齐轴。世界 X/Y/Z 先换算到候选 `80³` 连续体素坐标，向下取整后减去裁剪起点；目标写入时重排为 Z/Y/X。

### 候选辅助特征

`candidate_manual_features` 是 `float32 (12,)`。A、P、V、PP 各占连续三个位置：

| 切片 | 模态 | 三个值 |
| --- | --- | --- |
| `0:3` | A | token 数、概率均值、概率最大值 |
| `3:6` | P | token 数、概率均值、概率最大值 |
| `6:9` | V | token 数、概率均值、概率最大值 |
| `9:12` | PP | token 数、概率均值、概率最大值 |

计数以 `float32` 保存，单位为 token 个数；概率无量纲。某模态为空时对应三个值均为 0。

`candidate_sampling_features` 是 `float32 (4,)`：

| 位置 | 含义与单位 |
| ---: | --- |
| 0 | 请求裁剪起点是否有任一轴超出合法范围并被限制，保存 0 或 1；实际起点恰好等于边界但未发生限制时仍为 0 |
| 1 | 实际裁剪中心到完整 V 质心的 L2 距离，单位 voxel |
| 2 | 同一位移的 L∞ 距离，单位 voxel |
| 3 | `N_V/N_V_full`，位于 `[0,1]`；裁剪几何要求 `N_V_full>0`，完整 V 为空会在此之前失败 |

### 联合身份轴

`full_matcher` 只保留类别属于当前 `identity_scope`、`graph_available=True` 且当前语言家族 `available=True` 的身份。身份列表已经按 `(class_index, object_key, prepared_smiles)` 排序，因此筛选保持稳定顺序。

每个保留身份读取 `selected_relative_path` 指向的语言 NPZ：

| 字段 | 运行时要求 |
| --- | --- |
| `embedding` | 转为 `float32 (768,)` 后必须全部有限 |
| `prepared_smiles` | 必须是可由读取器取得的 Unicode 标量，并与身份记录的顶层字符串逐字相同 |

语言读取器此处不重复核验 manifest 扫描时使用的所有归档元数据。图由 `LigandGraphStore.load(object_key)` 读取；同一 `object_key`、不同 `prepared_smiles` 是两个身份槽，各自有一项图引用，图预算允许且缓存写入成功时，两项可以引用同一个 dataclass 实例。

身份缓存键为 `(pdb_id, language_family:identity_scope)`；预算允许且 `put` 保留时，下列候选无关字段可供同一进程后续访问复用：

| 字段 | dtype 与形状 |
| --- | --- |
| `identity_object_keys` | 长度 `I_b` 的字符串列表 |
| `identity_prepared_smiles` | 长度 `I_b` 的非空字符串列表 |
| `identity_class_index` | `int64 (I_b,)` |
| `language_embeddings` | `float32 (I_b,768)` |
| `ligand_graphs` | 长度 `I_b` 的 `LigandGraph` 列表 |

`identity_target` 不进入缓存，而是为每个候选重新计算。只有 `class_index`、`object_key` 和顶层 `prepared_smiles` 三者都相同的唯一身份槽才成为目标。任一联合键分量缺失、当前可用身份轴没有真值时返回 `-1`；重复真值槽失败关闭。

`candidate_pretrain` 固定返回：空 `identity_object_keys`、空 `identity_prepared_smiles`、`int64 (0,)` 的类别、`float32 (0,768)` 的语言数组、空图列表和 `identity_target=-1`。

## `CandidateItem` 字段

### 元数据

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `metadata.split` | 字符串 | 当前物化划分 |
| `metadata.pdb_id` | 非空小写字符串 | 候选所属 PDB |
| `metadata.candidate_index` | 整数 | 当前 PDB 内从零开始的 Stage1 F1 候选编号 |
| `metadata.matched_object_key` | 字符串或 `None` | 通过覆盖阈值的 occurrence 模板键；`type_tag=other` 背景也可非空，未命中为 `None` |
| `metadata.matched_prepared_smiles` | 字符串或 `None` | 仅五类前景命中的顶层 PreparedLigand.smiles 可非空；`other` 背景、未命中或字符串不可用时为 `None` |
| `metadata.matched_prepared_smiles_status` | 枚举字符串 | `available`、`explicit_null`、`missing_file`、`not_applicable` 依次表示顶层字符串可用、明确为 null、PreparedLigand 文件缺失、背景不适用 |
| `metadata.identity_seen_in_train` | 布尔值或 `None` | 已选择语言家族且具备完整前景联合身份时为 train 可见性布尔值；未选语言、背景、未命中或联合键不完整时为 `None` |

### 数值与身份字段总表

| 字段 | dtype 与形状 | 对齐关系 |
| --- | --- | --- |
| `voxel_input` | `float16 (105,48,48,48)` | 第一轴为 56 个候选密度通道和 49 个受体通道，后三轴为 Z/Y/X |
| `a_tokens` | `float32 (N_A,229)` | 完整 `80³` A 轴 |
| `p_tokens` | `float32 (N_P,116)` | 完整 `80³` P 轴 |
| `v_tokens` | `float32 (N_V,52)` | 实际 `48³` 内 V 轴 |
| `v_index_zyx` | `int64 (N_V,3)` | 与 `v_tokens` 逐项对齐 |
| `pp_tokens` | `float32 (K,4)` | 固定 `K` PP 轴 |
| `pp_index_zyx` | `int64 (K,3)` | 与 `pp_tokens` 逐项对齐 |
| `class_target` | 整数标量 | 六分类下标 |
| `ligand_area_target` | `bool (48,48,48)` | 配体 occurrence 区域并集 |
| `binding_area_target` | `bool (48,48,48)` | 完整受体结合原子区域 |
| `full_candidate_voxel_count` | 整数标量 | 裁剪前完整 V 数量 |
| `candidate_manual_features` | `float32 (12,)` | 四模态统计 |
| `candidate_sampling_features` | `float32 (4,)` | 裁剪诊断 |
| `identity_object_keys` | 长度 `I_b` 的字符串列表 | 与其余身份字段逐槽对齐 |
| `identity_prepared_smiles` | 长度 `I_b` 的非空字符串列表 | 与 `object_key` 共同定义联合身份 |
| `identity_class_index` | `int64 (I_b,)` | 前景类别下标 |
| `language_embeddings` | `float32 (I_b,768)` | 当前语言家族表示 |
| `ligand_graphs` | 长度 `I_b` 的图列表 | 每个身份槽恰好一张图 |
| `identity_target` | 整数标量 | 有效槽下标或 `-1` |

## 物理批次

### 变长 token 与掩码

`collate_candidates` 拒绝空列表。所有浮点 token 转为 `float32`，在 token 轴右侧补 0；对应 mask 中真实位置为 `True`，补零位置为 `False`。

| token 字段 | 批次形状 | mask | 索引字段 |
| --- | --- | --- | --- |
| `a_tokens` | `float32 (B,N_A_max,229)` | `a_mask: bool (B,N_A_max)` | 无 |
| `p_tokens` | `float32 (B,N_P_max,116)` | `p_mask: bool (B,N_P_max)` | 无 |
| `v_tokens` | `float32 (B,N_V_max,52)` | `v_mask: bool (B,N_V_max)` | `v_index_zyx: int64 (B,N_V_max,3)` |
| `pp_tokens` | `float32 (B,K,4)` | `pp_mask: bool (B,K)`；正式 Dataset 的全部位置为 `True` | `pp_index_zyx: int64 (B,K,3)` |

V/PP 索引的补零位置填 0，但数值 0 也可以是真实 Z、Y 或 X 下标；必须用对应 mask 判断位置是否有效，不能根据索引值判断。

### 固定形状字段

| 字段 | 批次 dtype 与形状 |
| --- | --- |
| `metadata` | 长度 `B` 的 Python 对象列表，不补齐 |
| `voxel_input` | 正式 Dataset 输出时为 `float16 (B,105,48,48,48)` |
| `class_target` | `int64 (B,)` |
| `ligand_area_target` | `bool (B,48,48,48)` |
| `binding_area_target` | `bool (B,48,48,48)` |
| `full_candidate_voxel_count` | `int64 (B,)` |
| `candidate_manual_features` | `float32 (B,12)` |
| `candidate_sampling_features` | `float32 (B,4)` |

批处理代码明确核验 V/PP token 宽度和索引数量、两个候选特征的宽度、身份字段数量和目标范围。它不核验 A/P 固定宽度、voxel/area 固定形状及若干 dtype/range；`np.stack`、token 补齐或身份连接只会暴露部分不相容输入，批内统一但科学宽度或形状错误的值可能被接受，因此这些字段依赖正式 `CandidateDataset` 契约。

### 身份补齐

| 字段 | 批次 dtype 与形状 | 补齐语义 |
| --- | --- | --- |
| `language_embeddings` | `float32 (B,I_max,768)` | 无效槽为 0 |
| `identity_mask` | `bool (B,I_max)` | 每个样本前 `I_b` 个位置为 `True` |
| `identity_class_index` | `int64 (B,I_max)` | 无效槽为 `-1` |
| `identity_target` | `int64 (B,)` | 每项必须位于 `[-1,I_b)` |
| `identity_object_keys` | `list[list[str]]` | 第 `b` 项保持原长 `I_b`，不补到 `I_max` |
| `identity_prepared_smiles` | `list[list[str]]` | 第 `b` 项保持原长 `I_b`，不补到 `I_max` |

身份连接核验 object key、smiles、类别、语言向量和图的数量一致，smiles 非空，目标位于允许范围。它不再次核验目标槽的类别和联合键是否与 metadata 相同；Dataset 负责构造该关系，损失与评估继续使用同一轴。

### 配体图连接

每个有效身份恰好产生一张图，因此 `G=sum(I_b)`。同一 `object_key`、不同 smiles 的两个身份槽仍有两个 `graph_to_*` 映射项。

| 字段 | dtype 与形状 | 含义 |
| --- | --- | --- |
| `graph_to_sample` | `int64 (G,)` | 第 `g` 张图所属样本下标 `[0,B)` |
| `graph_to_identity` | `int64 (G,)` | 第 `g` 张图在所属样本中的身份下标 `[0,I_b)` |
| `graph_element` | `int64 (N_node,)` | 模板原子序数 |
| `graph_charge` | `int64 (N_node,)` | 逐节点形式电荷 |
| `graph_residue_id` | `int64 (N_node,)` | 模板内部残基编号 |
| `graph_node_continuous` | `float32 (N_node,15)` | `0:7` 手性、`7:11` 3/4/5/6 元环、`11` 模板 `is_present`、`12:15` 组内中心化后除以 10 Å 的参考 X/Y/Z |
| `graph_node_graph_index` | `int64 (N_node,)` | 每个连接后节点所属的全局图下标 `[0,G)` |
| `graph_edge_index` | `int64 (2,N_directed_edge)` | 第一轴为源端点和目标端点；每张图的局部端点已加此前图节点总数 |
| `graph_edge_features` | `float32 (N_directed_edge,10)` | `0:5` 键型、`5:9` 3/4/5/6 元环、`9` 参考距离除以 10 Å；跨残基边距离分量为 0 |

没有有效身份时，图字段固定为：

| 字段 | 空张量 |
| --- | --- |
| `graph_element`、`graph_charge`、`graph_residue_id`、`graph_node_graph_index` | `int64 (0,)` |
| `graph_node_continuous` | `float32 (0,15)` |
| `graph_edge_index` | `int64 (2,0)` |
| `graph_edge_features` | `float32 (0,10)` |
| `graph_to_sample`、`graph_to_identity` | `int64 (0,)` |

## 采样器与 DataLoader

### `ShuffledBatchSampler`

采样器只调用 `len(dataset)`，初始下标为 `0..N-1`。它不读取 PDB 标识，也不按 PDB 分组。

1. `shuffle=True` 时，用独立 `random.Random(seed+epoch)` 原地洗牌；不读取或推进 Python 全局随机状态。
2. `sample_count_multiple` 非空时，必须是 `batch_size` 的正整数倍；洗牌后只保留最大完整倍数的前缀。
3. 按 `batch_size` 切分；`drop_last=True` 丢弃不足一个物理批量的末批。

相同 Dataset 长度、`shuffle`、`seed` 和 `epoch` 会得到相同的初始排列。跨不同物理 `batch_size` 比较展平后的保留序列时，还必须让共同 `sample_count_multiple` 同时是两种 `batch_size` 的整数倍，或用其它明确条件保证两边保留相同前缀；没有共同截短倍数时，`drop_last` 与不同物理批量可能丢弃不同长度的末尾。不同 GPU 类型之间只能把满足这些条件的序列一致性作为重复实验的样本控制证据，不能据此宣称数值后端逐位同轨。

### `build_candidate_loader`

| DataLoader 字段 | 固定值或来源 | 作用 |
| --- | --- | --- |
| `batch_sampler` | 输入 `ShuffledBatchSampler` | 唯一决定样本顺序和物理批量 |
| `collate_fn` | `collate_candidates` | 生成本页定义的 `CandidateBatch` |
| `num_workers` | 配置输入 | 合法范围由 PyTorch 检查 |
| `prefetch_factor` | 配置输入 | 与 `num_workers` 的组合约束由 PyTorch 检查 |
| `pin_memory` | `True` | 尝试为返回张量使用页锁定内存 |
| `persistent_workers` | `False` | 一次完整迭代结束后 worker 退出，下次迭代重新创建 |
| `multiprocessing_context` | `spawn` | worker 通过序列化取得 Dataset，不继承父进程已填充的大数组缓存 |
| `generator` | 专用 `torch.Generator` | 只用于 DataLoader 的 worker 基准种子，不推进模型使用的全局 Torch RNG |

候选洗牌由 `ShuffledBatchSampler` 的 Python 随机生成器决定，不由 DataLoader 的 Torch 生成器决定。worker 内异常会在父进程迭代时重新抛出；本目录不吞掉失败，也不承诺跨进程后的异常对象逐字保持原样。

## 检查职责分层

| 时点 | 明确检查 | 不在该时点完整检查 |
| --- | --- | --- |
| Dataset 构造 | task_name、language_family、裁剪、缓存、candidate_pretrain 的 identity_scope=None、full_matcher 的 identity_scope 非 None；manifest 版本、联合键和类别顺序；身份记录的关键类型与 PDB 局部联合键唯一性 | full_matcher 非空 identity_scope 是否属于枚举、候选 JSONL 每个字段、语言证据全部嵌套字段、所有来源文件存在性和内容 |
| PDB artifact 首次读取 | `receptor_coords` 为有限 `(N,3)`；`binding_atom` 与完整受体逐项对齐 | centered 二十字段的全部 dtype/形状；occurrence mask 的二维三列形状 |
| 单候选物化 | 完整 V 数与清单一致；PP/centered 几何一致；smiles 状态和值相容；最终 A/P/V 宽度和有限值；语言向量与字符串；身份真值唯一 | 语言 NPZ 的全部 manifest 扫描字段；每个 centered 来源字段的全局 offsets 终点关系 |
| 批处理 | 非空 items；V/PP token-index 数量与固定宽度；候选辅助特征宽度；身份字段数量、smiles 非空和目标范围 | A/P 固定宽度、voxel/area 固定形状、若干 dtype/range；identity_target 指向槽与 metadata 的重新比对；统一错误形状可能被补齐或堆叠接受 |

## 调用关系与验证

训练入口在 `matcher/training/engine.py` 中分别构造 train 和 validation Dataset、采样器与 DataLoader：训练使用洗牌和末批规则，validation 保持固定下标序列。模型、损失和评估只读取 `CandidateBatch`，不直接打开本目录所列来源文件。

与本目录契约直接相关的测试集中在：

- `matcher/tests/test_data_pipeline.py`：manifest v4 拒绝、联合身份拆槽、缺失或 null smiles、裁剪几何、四模态与区域目标、身份范围、采样确定性和跨硬件配对序列。
- `matcher/tests/test_model_and_loss.py`：`CandidateBatch` 各字段进入模型和损失后的形状、掩码与身份图消费关系。
- `matcher/tests/test_metrics_and_config.py`：训练配置解析出的 Dataset、批量和采样字段，以及评估对 metadata 和身份轴的解释。
