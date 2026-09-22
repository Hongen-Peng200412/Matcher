# Stage3 小分子测试清单

本目录连接 PocketXMol Stage3 测试 occurrence、Matcher 受支持 SMILES 身份和三套 Stage1 候选。`manifest.json` 是读取入口；所有编号保留来源产物中的原始编号，不按 JSONL 文件位置重新编号。

当前 `test_smallmolecule_v1` 覆盖 77 个 PDB 和 446 个 PocketXMol Stage3 测试 occurrence。真实受体 Find_1、CryoAtom2 Find_1 与 Emap2lig 保留各自独立的候选集合；三套候选不能按文件位置或相同 `source_blob_index` 互相配对。

## 目录结构

```text
## <科学产物>
<manifest_root>/
├── manifest.json                 # 77-PDB 轴、双向覆盖门槛和其余文件的相对文件名
├── stage3_occurrences.jsonl      # PocketXMol 测试 occurrence；每行对应一个沉积小分子实例
├── identities.jsonl              # PDB 内受支持精确 SMILES 身份；每行对应一个 (pdb_id, smiles)
├── real_candidates.jsonl         # 真实受体 Find_1 的全部 Stage3 测试 PDB centered 候选
├── cryoatom2_candidates.jsonl    # CryoAtom2 Find_1 的全部 Stage3 测试 PDB centered 候选
└── emap2lig_candidates.jsonl     # Emap2lig 原生候选；不具有 Find_1 centered 框或 Gaussian 分数

## <其他文件>
<manifest_root>/
└── README.md                     # 本文件的发布副本，说明字段与跨文件编号关系（可再生说明）
```

两棵树指向同一个物理目录，只按生命周期分开列示。前五份 JSON/JSONL 是科学接口；`README.md` 可以从仓库内的 `MANIFEST_README.md` 重新复制。checkpoint、release、launch、命令、文件摘要、运行时间和作业状态不写入该目录；这些追溯事实由实验日志记录。

## `manifest.json`

| 字段 | 类型 | 含义 | 示例值 |
| --- | --- | --- | --- |
| `coverage_threshold` | 浮点数 | 候选覆盖率与 occurrence 覆盖率共同使用的门槛 | `0.3` |
| `pdb_ids` | 长度 77 的字符串列表 | Stage3 测试集的有序小写 PDB 标识；候选曲线以该列表为固定分母和输出顺序 | 格式片段如 `["9abc", "9def"]` |
| `files` | 具有 6 个键的字符串映射 | 逻辑名称到本目录相对文件名的映射；读取者以 `manifest.json` 所在目录为根解析 | `files["identities"] == "identities.jsonl"` |

`files` 的键与对应文件如下：

| `files` 键 | 相对文件名 | 文件中一个 JSONL 记录对应的实体 |
| --- | --- | --- |
| `stage3_occurrences` | `stage3_occurrences.jsonl` | 一个 PocketXMol Stage3 测试 occurrence |
| `identities` | `identities.jsonl` | 一个 PDB 内的受支持精确 SMILES 身份 |
| `real_candidates` | `real_candidates.jsonl` | 一个真实受体 Find_1 centered 候选 |
| `cryoatom2_candidates` | `cryoatom2_candidates.jsonl` | 一个 CryoAtom2 Find_1 centered 候选 |
| `emap2lig_candidates` | `emap2lig_candidates.jsonl` | 一个 Emap2lig 原生候选 |
| `readme` | `README.md` | 当前字段契约文档 |

下面是缩短 `pdb_ids` 以展示结构的构造示例；正式文件必须包含完整 77 个 PDB 标识。

```json
{
  "coverage_threshold": 0.3,
  "files": {
    "cryoatom2_candidates": "cryoatom2_candidates.jsonl",
    "emap2lig_candidates": "emap2lig_candidates.jsonl",
    "identities": "identities.jsonl",
    "readme": "README.md",
    "real_candidates": "real_candidates.jsonl",
    "stage3_occurrences": "stage3_occurrences.jsonl"
  },
  "pdb_ids": ["9abc", "9def"]
}
```

## 身份与 occurrence

### `stage3_occurrences.jsonl`

每行原样保留 PocketXMol 冻结测试清单中的一个 occurrence。`pdb_id` 与 `candidate_id` 共同定位沉积 occurrence；`candidate_id` 不等于 Find_1 的 `source_blob_index` 或 `centered_box_index`。

| 字段 | 数据类型与形状 | 含义 | 示例值 |
| --- | --- | --- | --- |
| `split` | 字符串 | PocketXMol 冻结划分；本文件固定为测试划分 | `"test"` |
| `pdb_id` | 字符串 | occurrence 所属的小写 PDB 标识 | `"9abc"` |
| `candidate_id` | 非负整数 | occurrence 在当前 PDB 的原始编号；与 `pdb_id` 共同构成唯一键 | `6` |
| `prepared_smiles` | 字符串 | 精确顶层 `PreparedLigand.smiles`；也是 Matcher 对该 occurrence 的正确身份答案 | `"CCO"` |
| `n_heavy_atoms` | 正整数 | 公共 SMILES 图与迁移坐标共同包含的重原子数 | `3` |
| `center_offset_xyz_A` | 长度 3 的浮点列表 | PocketXMol 冻结 C5 测试使用的世界坐标偏移；分量依次为 X、Y、Z，单位 Å | `[1.0, 0.0, 0.0]` |
| `sampling_seed` | 整数 | PocketXMol 为该 occurrence 冻结的采样种子 | `10831` |
| `views` | 字符串列表 | 该 occurrence 的冻结测试视图；可包含 `ALL`、`CAP10` 和 `HF10_TO5` | `["ALL", "CAP10", "HF10_TO5"]` |

下面是字段齐全的构造示例，不代表正式测试 occurrence 或种子。

```json
{
  "candidate_id": 6,
  "center_offset_xyz_A": [1.0, 0.0, 0.0],
  "n_heavy_atoms": 3,
  "pdb_id": "9abc",
  "prepared_smiles": "CCO",
  "sampling_seed": 10831,
  "split": "test",
  "views": ["ALL", "CAP10", "HF10_TO5"]
}
```

`prepared_smiles` 与 AdaLigand 对应 `(pdb_id, candidate_id)` 的 `prepared_smiles.jsonl` 顶层 `smiles` 完全相同。Stage3 冻结清单中的 occurrence 均属于原始 `small_molecule` 类别。

### `identities.jsonl`

每行对应一个 `(pdb_id, smiles)` 身份。相同 PDB 内共享精确 SMILES 的多个 occurrence 合并为一个身份；不同 PDB 即使具有相同 SMILES，仍保存为不同记录。两个 occurrence 列表都保存原始 `candidate_id`，不指向 JSONL 文件位置。

| 字段 | 数据类型与形状 | 含义 | 示例值或空值规则 |
| --- | --- | --- | --- |
| `pdb_id` | 字符串 | 身份所属的小写 PDB 标识 | `"9abc"` |
| `smiles` | 字符串 | 精确顶层 `PreparedLigand.smiles` | `"CCO"` |
| `language_relative_path` | 字符串 | SMI-TED 语言表示相对 Matcher manifest `data_root` 的 NPZ 路径 | `"stage1_preparation_box_pool_2/ligand_language_models/smi_ted_289m/9abc/candidate_0.npz"` |
| `type_tag` | 枚举字符串 | 共享该 SMILES 的 occurrence 原始类别；取 `ion`、`nucleotide_like`、`peptide_like`、`small_molecule` 或 `sugar` | `"small_molecule"` |
| `occurrence_ids` | 升序整数列表，长度至少 1 | 当前 PDB 中共享该 SMILES 的全部 occurrence 编号 | `[0, 3]` |
| `stage3_occurrence_ids` | 升序整数列表，长度可为 0 | `occurrence_ids` 中进入 Stage3 冻结测试集的子集 | `[3]`；没有 Stage3 成员时为 `[]` |
| `allowed_as_small_molecule_answer` | 布尔值 | `type_tag == "small_molecule"` 时为 `true`；控制同一次完整身份前向后的 small-molecule argmax 掩码 | `true` |

下面是说明身份合并关系的构造示例。occurrence 0 和 3 共享同一 SMILES，但只有 occurrence 3 属于 Stage3 测试集。

```json
{
  "allowed_as_small_molecule_answer": true,
  "language_relative_path": "stage1_preparation_box_pool_2/ligand_language_models/smi_ted_289m/9abc/candidate_0.npz",
  "occurrence_ids": [0, 3],
  "pdb_id": "9abc",
  "smiles": "CCO",
  "stage3_occurrence_ids": [3],
  "type_tag": "small_molecule"
}
```

`identities.jsonl` 是完整受支持身份轴，不只保存 Stage3 正确答案。其它受支持 small-molecule 身份仍可进入 `small_molecule_answer_mask`，因此该掩码不会直接泄露唯一正确身份。

## Stage1 候选

三份文件使用同一目标与排序字段，因此可以交给同一个 top-k 状态机。只有两类 Find_1 候选能够送入 Matcher 身份前向；Emap2lig 只提供 Stage1-only 基线。

### `real_candidates.jsonl`

每行对应真实受体 Find_1 `held_out_test_0` 中的一个 centered 候选。记录保留完整 test_0 manifest 已经确定的唯一 occurrence 目标，并补充 Stage3 三态范围、完整来源 blob 的世界中心和该 blob 是否完全位于 centered 80³ 框内。

### `cryoatom2_candidates.jsonl`

字段与 `real_candidates.jsonl` 相同，但候选来自 CryoAtom2 受体条件。两个 Find_1 文件的候选数量、中心、排序分数和唯一目标均可不同，不能按 JSONL 位置逐项配对。

### `emap2lig_candidates.jsonl`

每行对应一个 Emap2lig 原生候选。它没有 Find_1 centered 框，因此 `centered_box_index`、`gaussian_score` 和 `source_blob_fits_centered_box` 均为 `null`。正式排序只使用实例内原生 ligand probability 均值 `source_probability_mean`。

### 三类候选的共同字段

| 字段 | 数据类型与形状 | 含义与特殊值 | 示例值 |
| --- | --- | --- | --- |
| `pdb_id` | 字符串 | 候选所属 PDB | `"9abc"` |
| `centered_box_index` | 非负整数或 `null` | Find_1 候选在当前 PDB `F2_centered.npz` 候选轴上的编号；Emap2lig 为 `null` | `4` |
| `source_blob_index` | 非负整数 | 指向当前 PDB、当前 Stage1 来源的完整 blob；与 `pdb_id` 共同构成该来源内候选唯一键 | `17` |
| `candidate_selected` | 布尔值 | `true` 候选整体进入排序前段，`false` 候选整体进入追加段 | `true` |
| `gaussian_score` | 有限浮点数或 `null` | Find_1 Gaussian 段内排序分数；Emap2lig 没有该分数，因此为 `null` | `0.82` |
| `source_probability_mean` | 有限浮点数 | 完整来源 blob 内 Stage1 原生概率的算术平均值 | `0.74` |
| `foreground_target` | 布尔值 | 是否存在满足双向 0.30 覆盖门槛的唯一目标 | `true` |
| `target_category` | 枚举字符串或 `null` | 唯一目标的原始五类标签；背景为 `null` | `"small_molecule"` |
| `matched_occurrence_id` | 非负整数或 `null` | 最大化覆盖率几何平均并按较小编号决胜后的唯一 occurrence；背景为 `null` | `3` |
| `target_smiles` | 字符串或 `null` | 唯一目标的精确顶层 SMILES；背景为 `null` | `"CCO"` |
| `candidate_coverage` | 浮点数 | 交集体素数除以完整候选体素数；背景固定为 `0.0` | `0.56` |
| `occurrence_coverage` | 浮点数 | 交集体素数除以唯一目标 occurrence 体素数；背景固定为 `0.0` | `0.68` |
| `overlap_score` | 浮点数 | `sqrt(candidate_coverage * occurrence_coverage)`；背景固定为 `0.0` | `0.617089...` |
| `target_scope` | 枚举字符串 | 取 `background`、`non_stage3_foreground` 或 `stage3_foreground`，决定是否消耗 attempt 及是否允许命中 | `"stage3_foreground"` |
| `blob_center_world_xyz_A` | 长度 3 的浮点列表 | 完整来源 blob 的全局体素中心算术平均转换到世界坐标；轴顺序为 X、Y、Z，单位 Å | `[12.5, -3.0, 8.25]` |
| `source_blob_fits_centered_box` | 布尔值或 `null` | Find_1 完整 blob 是否位于 centered 框；Emap2lig 为 `null` | `true` |

`foreground_target=false` 时，`target_category`、`matched_occurrence_id` 和 `target_smiles` 都为 `null`，三项覆盖数值都为 `0.0`，且 `target_scope="background"`。存在唯一前景目标时三个目标字段均非空；`target_scope` 只进一步说明该 occurrence 是否属于 Stage3 冻结测试范围。

下面第一条是字段齐全的构造 Find_1 前景候选，第二条展示背景候选的空值联动；二者都不代表正式运行观测。

```json
{"blob_center_world_xyz_A":[12.5,-3.0,8.25],"candidate_coverage":0.56,"candidate_selected":true,"centered_box_index":4,"foreground_target":true,"gaussian_score":0.82,"matched_occurrence_id":3,"occurrence_coverage":0.68,"overlap_score":0.617089,"pdb_id":"9abc","source_blob_fits_centered_box":true,"source_blob_index":17,"source_probability_mean":0.74,"target_category":"small_molecule","target_scope":"stage3_foreground","target_smiles":"CCO"}
{"blob_center_world_xyz_A":[4.0,5.0,6.0],"candidate_coverage":0.0,"candidate_selected":false,"centered_box_index":9,"foreground_target":false,"gaussian_score":0.18,"matched_occurrence_id":null,"occurrence_coverage":0.0,"overlap_score":0.0,"pdb_id":"9abc","source_blob_fits_centered_box":true,"source_blob_index":22,"source_probability_mean":0.12,"target_category":null,"target_scope":"background","target_smiles":null}
```

## 跨文件编号与对齐关系

- `stage3_occurrences.jsonl.candidate_id`、`identities.jsonl.occurrence_ids`、`identities.jsonl.stage3_occurrence_ids` 和三个候选文件的 `matched_occurrence_id` 都指向 AdaLigand 原始 occurrence 编号。
- `source_blob_index` 只指向当前 Stage1 来源的完整 blob 数组；同一数值在真实受体、CryoAtom2 和 Emap2lig 之间没有共享身份。
- `centered_box_index` 只索引 Find_1 `F2_centered.npz` 的候选轴；它不索引 occurrence、SMILES 身份或完整 blob。
- `target_smiles` 与同一 `(pdb_id, matched_occurrence_id)` 的顶层 `PreparedLigand.smiles` 完全相同。对应身份受支持时，它还与 `identities.jsonl` 中同一 `(pdb_id, smiles)` 记录连接。
- `identities.jsonl` 为模型完整身份轴提供类别事实。模型实际 `identity_smiles` 顺序来自对应 test_0 manifest；评估按精确 `(pdb_id, smiles)` 联合键逐槽生成 `small_molecule_answer_mask`。
- `blob_center_world_xyz_A` 始终由完整来源 blob 计算；`source_blob_fits_centered_box=false` 时也不会改用 centered 框中心。

## 唯一目标与排序规则

候选与 occurrence 的候选覆盖率及 occurrence 覆盖率都不低于 `coverage_threshold=0.3` 时，该 occurrence 才能成为目标。多个 occurrence 同时满足门槛时，选择两项覆盖率几何平均最大的 occurrence；几何平均并列时选择较小的 `occurrence_id`。

每个 PDB 先排列全部 `candidate_selected=true` 候选，再追加全部 `candidate_selected=false` 候选。两段内部使用当前指定分数降序排列，并以较小 `source_blob_index` 作为同分次级键。Gaussian 曲线使用 `gaussian_score`；来源概率均值曲线使用 `source_probability_mean`。Matcher 只判断身份是否正确，不改变这套 Stage1 候选顺序。
