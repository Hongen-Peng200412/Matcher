# `test_smallmolecule` manifest 字段契约

本目录冻结 PocketXMol Stage3 小分子测试范围、三套 Stage1 候选、Matcher 完整身份轴和完整 blob 世界中心。`manifest.json` 是唯一入口；训练与 validation 不读取本目录。

## 产物目录

```text
## <科学产物>
/storage/penghongen/Matcher/manifests/test_smallmolecule_v1/
├── manifest.json                    # 文件身份、科学规则、77-PDB 顺序和摘要计数
├── stage3_occurrences.jsonl         # PocketXMol 的 446 个 Stage3 测试 occurrence
├── identities.jsonl                 # 77 个 PDB 内 180 个受支持精确 SMILES 身份
├── real_candidates.jsonl            # 真实受体 Find_1 的 598 个 centered 候选
├── cryoatom2_candidates.jsonl       # CryoAtom2 Find_1 的 632 个 centered 候选
└── emap2lig_candidates.jsonl        # Emap2lig 的 30,561 个原生候选

## <其他文件>
/storage/penghongen/Matcher/manifests/test_smallmolecule_v1/
├── source_snapshot.json             # 每个实际读取来源文件的路径、字节数和 SHA-256（来源留证）
└── README.md                        # 当前文件字段与对齐关系（可再生说明）
```

## 核心范围与身份

### `stage3_occurrences.jsonl`

每条记录逐字段保存 PocketXMol 正式 `test.jsonl` 的一个 occurrence。`pdb_id` 与 `candidate_id` 共同定义空间实例；`prepared_smiles` 是 Matcher 的唯一化学答案。例如 `{"pdb_id":"11jb","candidate_id":0,"prepared_smiles":"[H]OC1..."}` 表示 `11jb` 的 occurrence 0。

### `identities.jsonl`

每条记录对应一个 `(pdb_id, smiles)` 身份。下列列表字段都使用原始 occurrence 编号，不是 JSONL 位置。

| 字段 | 类型与含义 | 示例值 |
| --- | --- | --- |
| `pdb_id` | 字符串；小写 PDB 标识 | `"11jb"` |
| `smiles` | 字符串；精确顶层 `PreparedLigand.smiles` | `"[H]OC1..."` |
| `language_relative_path` | 字符串；相对 AdaLigand `data_root` 的 SMI-TED NPZ | `"stage1_preparation_box_pool_2/ligand_language_models/smi_ted_289m/11jb/candidate_0.npz"` |
| `type_tag` | 字符串；共享该 SMILES 的 occurrence 唯一原始类别 | `"small_molecule"` |
| `occurrence_ids` | 整数列表；当前 PDB 中共享该 SMILES 的全部 occurrence 编号 | `[0, 1]` |
| `stage3_occurrence_ids` | 整数列表；`occurrence_ids` 中属于 Stage3 测试集的子集，可为空 | `[0, 1]` |
| `allowed_as_small_molecule_answer` | 布尔值；True 表示最终 small-molecule argmax 允许该身份 | `true` |

## Stage1 候选

### `real_candidates.jsonl` 与 `cryoatom2_candidates.jsonl`

两个文件字段严格同构。每条记录对应一个 Find_1 centered 候选，并继承既有 `test_0` 的唯一 occurrence 目标。

### `emap2lig_candidates.jsonl`

每条记录对应一个 Emap2lig 原生候选。它没有 centered 框，因此 `centered_box_index` 和 `source_blob_fits_centered_box` 为 `null`；`gaussian_score` 也为 `null`，正式排序只使用 `source_probability_mean`。

### 三类候选的共同字段

| 字段 | 类型与含义 | 示例值 |
| --- | --- | --- |
| `pdb_id` | 字符串；候选所属 PDB | `"9rd8"` |
| `centered_box_index` | 整数或 null；Find_1 候选在 `F2_centered.npz` 的稳定编号 | `0` |
| `source_blob_index` | 整数；指向当前来源完整 blob 数组 | `53` |
| `candidate_selected` | 布尔值；True 候选整体排在 False 候选之前 | `true` |
| `gaussian_score` | 有限浮点或 null；Find_1 Gaussian 段内排序分数 | `1.7736837` |
| `source_probability_mean` | 有限浮点；完整来源 blob 内原生概率均值 | `0.98322934` |
| `foreground_target` | 布尔值；是否存在满足双向 0.30 覆盖门槛的唯一目标 | `true` |
| `matched_occurrence_id` | 整数或 null；最大覆盖规则选出的唯一 occurrence 编号 | `0` |
| `target_smiles` | 字符串或 null；唯一目标的精确顶层 SMILES | `"CCO"` |
| `candidate_coverage` | 浮点；交集体素数除以完整候选体素数 | `0.73` |
| `occurrence_coverage` | 浮点；交集体素数除以目标 occurrence 体素数 | `0.89` |
| `overlap_score` | 浮点；两项覆盖率的几何平均 | `0.81` |
| `target_scope` | 字符串；取 `background`、`non_stage3_foreground` 或 `stage3_foreground` | `"stage3_foreground"` |
| `blob_center_world_xyz_A` | 三个有限浮点；完整 blob 的世界 XYZ 体素中心均值，单位 Å | `[452.8, 505.8, 559.7]` |
| `source_blob_fits_centered_box` | 布尔值或 null；Find_1 完整 blob 是否全部位于 centered 80³ 框 | `true` |

`9rd8` 真实受体的 `source_blob_index=0` 继续使用既有唯一目标 occurrence 0。即使它也覆盖 Stage3 occurrence 1，`target_scope` 仍是 `non_stage3_foreground`，因此免费跳过。

## 来源与完整性

`source_snapshot.json.files[*]` 的 `path` 是绝对路径，`size_bytes` 是文件字节数，`sha256` 是 64 位十六进制内容摘要。例如 `{"path":"/storage/.../test.jsonl","size_bytes":1234,"sha256":"abc..."}`。`manifest.json.file_sha256` 绑定本目录各派生文件；任何来源或派生文件变化都需要建立新版本目录，不能覆盖本目录。
