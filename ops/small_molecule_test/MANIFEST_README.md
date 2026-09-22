# `test_smallmolecule_v1` manifest 字段契约

本目录冻结 PocketXMol Stage3 小分子测试范围、真实受体与 CryoAtom2 的 Find_1 候选、Emap2lig 原生候选、Matcher 完整 PDB 内 SMILES 身份轴和完整 blob 世界中心。`manifest.json` 是唯一入口；Matcher 训练与 validation 不读取本目录。

## 目录结构

```text
## <科学产物>
/storage/penghongen/Matcher/manifests/test_smallmolecule_v1/
├── manifest.json                    # 派生文件、科学规则、77-PDB 顺序、冻结计数和 SHA-256 的唯一入口
├── stage3_occurrences.jsonl         # PocketXMol 的 446 个 Stage3 测试 occurrence
├── identities.jsonl                 # 77 个 PDB 内 180 个受支持精确 SMILES 身份
├── real_candidates.jsonl            # 真实受体 Find_1 的 598 个 centered 候选
├── cryoatom2_candidates.jsonl       # CryoAtom2 Find_1 的 632 个 centered 候选
└── emap2lig_candidates.jsonl        # Emap2lig 的 30,561 个原生候选

## <其他文件>
/storage/penghongen/Matcher/manifests/test_smallmolecule_v1/
├── source_snapshot.json             # 每个实际读取来源文件的路径、字节数和 SHA-256（来源留证）
└── README.md                        # 当前文件字段与跨文件关系（可再生说明）
```

以上两棵树指向同一个物理目录，只按生命周期分开列示。目录先在同级临时目录完整构建并通过计数、身份和有限数检查，再由原子重命名一次发布；目标目录已经存在时构建失败，不覆盖旧版本。

## 唯一入口与冻结规则

### `manifest.json`

该 JSON 顶层为对象。`files` 给出其它产物相对本目录的文件名，读取者必须经 `files` 定位文件，并用 `file_sha256` 核对内容。

| 字段 | 数据类型或结构 | 具体含义 | 示例值 |
| --- | --- | --- | --- |
| `schema` | 字符串 | manifest 契约版本 | `"matcher_test_smallmolecule_v1"` |
| `coverage_threshold` | 浮点数 | 候选覆盖率与 occurrence 覆盖率共同使用的门槛 | `0.3` |
| `pdb_ids` | 字符串列表，长度 77 | 固定分母中的小写 PDB 标识，按字典序排列 | 列表包含 `"11jb"` |
| `generation_runtime` | 对象 | 发布 manifest 时的 release、launch 和 Slurm Job 来源 | `generation_runtime.slurm_job_id="123456"` 是格式示例 |
| `files` | 字符串映射 | 逻辑名称到本目录相对文件名的映射 | `files.identities="identities.jsonl"` |
| `file_sha256` | 字符串映射 | 与 `files` 使用相同逻辑键；值是对应文件的 64 位十六进制摘要 | `file_sha256.identities="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"` 是构造格式示例 |
| `scientific_contract` | 对象 | 身份、唯一目标、候选顺序、尝试次数、Matcher 判定、答案掩码和 blob 中心规则 | 见下表 |
| `summary` | 对象 | PDB、occurrence、身份、候选和三态范围的冻结计数 | `summary.pdb_count=77` |

`scientific_contract` 的字段均为自包含字符串：

| 字段 | 具体规则 |
| --- | --- |
| `identity_key` | 身份只由精确顶层 `PreparedLigand.smiles` 定义 |
| `candidate_target` | 候选覆盖率和 occurrence 覆盖率均不低于 0.30 后，最大化两者几何平均；并列时取较小 occurrence 编号 |
| `candidate_order` | selected 段后接 unselected 段；每段内按指定分数降序、`source_blob_index` 升序 |
| `attempt_rule` | 背景与 Stage3 目标消耗尝试次数；唯一目标不属于 Stage3 的前景免费跳过 |
| `matcher_rule` | 只比较同一次完整身份 forward 的精确 SMILES argmax，不使用前景阈值 |
| `small_molecule_answer_rule` | 完整身份 forward 后，只在最终 argmax 前屏蔽 `type_tag` 不是 `small_molecule` 的 logits |
| `blob_center` | 完整来源 blob 的全局体素中心算术平均，转换为世界 XYZ，单位 Å |

`summary` 记录以下字段：

| 字段 | 数据类型 | 具体含义 | 冻结值 |
| --- | --- | --- | ---: |
| `pdb_count` | 整数 | 固定 PDB 分母 | 77 |
| `stage3_occurrence_count` | 整数 | PocketXMol 测试 occurrence 数 | 446 |
| `stage3_identity_count` | 整数 | 按 `(pdb_id, prepared_smiles)` 去重的 Stage3 身份数 | 104 |
| `identity_count` | 整数 | Matcher 完整受支持身份槽总数 | 180 |
| `small_molecule_answer_identity_count` | 整数 | `allowed_as_small_molecule_answer=true` 的身份数 | 127 |
| `real_candidate_count` | 整数 | 真实受体 Find_1 全部 centered 候选数 | 598 |
| `cryoatom2_candidate_count` | 整数 | CryoAtom2 Find_1 全部 centered 候选数 | 632 |
| `emap2lig_candidate_count` | 整数 | Emap2lig 全部原生候选数 | 30,561 |
| `real_outside_centered_candidate_count` | 整数 | 真实受体完整 blob 不能完全装入 centered 80³ 框的候选数 | 1 |
| `cryoatom2_outside_centered_candidate_count` | 整数 | CryoAtom2 完整 blob 不能完全装入 centered 80³ 框的候选数 | 1 |
| `real_target_scope_counts` | 字符串到整数的映射 | 真实受体候选的三态计数 | `background=209`、`non_stage3_foreground=85`、`stage3_foreground=304` |
| `cryoatom2_target_scope_counts` | 字符串到整数的映射 | CryoAtom2 候选的三态计数 | `background=263`、`non_stage3_foreground=76`、`stage3_foreground=293` |
| `emap2lig_target_scope_counts` | 字符串到整数的映射 | Emap2lig 候选的三态计数 | `background=30288`、`non_stage3_foreground=30`、`stage3_foreground=243` |
| `pdb_ids` | 字符串列表，长度 77 | 与顶层 `pdb_ids` 相同的冻结顺序 | 列表包含 `"11jb"` |

## 核心范围与身份

这一组定义 Stage3 occurrence 和 Matcher 可比较的完整身份集合。`candidate_id` 是 occurrence 编号；它不等于 Find_1 的 `source_blob_index` 或 `centered_box_index`。

### `stage3_occurrences.jsonl`

每条 JSONL 记录对应 PocketXMol 正式 `test.jsonl` 的一个 Stage3 测试 occurrence。文件保留来源记录的全部字段，不重编号；`pdb_id` 与 `candidate_id` 共同构成唯一键。

| 字段 | 数据类型与形状 | 具体含义 | 示例值 |
| --- | --- | --- | --- |
| `split` | 字符串 | PocketXMol 冻结划分 | `"test"` |
| `pdb_id` | 字符串 | 小写 PDB 标识 | `"11jb"` |
| `candidate_id` | 整数 | 当前 PDB 内 occurrence 编号 | `0` |
| `n_heavy_atoms` | 整数 | 当前小分子的沉积重原子数 | `24` 是格式示例 |
| `center_offset_xyz_A` | 浮点列表，长度 3 | PocketXMol 使用的中心偏移，轴顺序 XYZ，单位 Å | PocketXMol 的 C5 评测中，“提供给模型的口袋中心”相对于“真实配体重原子几何中心”的冻结位移向量。 |
| `sampling_seed` | 整数 | PocketXMol 冻结采样种子 | `3407` 是格式示例 |
| `views` | 字符串列表 | PocketXMol 冻结视图名称 | `["ALL", "CAP10", "HF10_TO5"]` 的子列表 |
| `prepared_smiles` | 字符串 | 精确顶层 `PreparedLigand.smiles`，也是 Matcher 正确答案 | `"CCO"` 是格式示例 |

`prepared_smiles` 必须与 AdaLigand 对应 `prepared_smiles.jsonl` 的同一 `(pdb_id, candidate_id)` 完全相同，且该 occurrence 的原始 `type_tag` 必须是 `small_molecule`。

下面是说明字段关系的构造记录，不代表 PocketXMol 正式产物中的真实数值：

```json
{"candidate_id":0,"center_offset_xyz_A":[0.0,0.0,0.0],"n_heavy_atoms":3,"pdb_id":"11jb","prepared_smiles":"CCO","sampling_seed":3407,"split":"test","views":["centered"]}
```

### `identities.jsonl`

每条记录对应一个 `(pdb_id, smiles)` 身份。相同 PDB 内共享精确 SMILES 的 occurrence 合并为一个身份槽；不同 PDB 的相同 SMILES 仍是不同记录。下列 occurrence 列表保存原始编号，不是 JSONL 位置。

| 字段 | 数据类型与形状 | 具体含义 | 示例值 |
| --- | --- | --- | --- |
| `pdb_id` | 字符串 | 小写 PDB 标识 | `"11jb"` |
| `smiles` | 字符串 | 精确顶层 `PreparedLigand.smiles` | `"CCO"` 是格式示例 |
| `language_relative_path` | 字符串 | 相对 AdaLigand `data_root` 的 SMI-TED NPZ 路径 | `"stage1_preparation_box_pool_2/ligand_language_models/smi_ted_289m/11jb/candidate_0.npz"` |
| `type_tag` | 字符串 | 共享该 SMILES 的 occurrence 唯一原始类别 | `"small_molecule"` |
| `occurrence_ids` | 整数列表，长度可变 | 当前 PDB 中共享该 SMILES 的全部 occurrence 编号，升序排列 | `[0,1]` |
| `stage3_occurrence_ids` | 整数列表，长度可为 0 | `occurrence_ids` 中属于 Stage3 测试集的子集 | `[0]`；没有 Stage3 成员时为 `[]` |
| `allowed_as_small_molecule_answer` | 布尔值 | `type_tag=="small_molecule"` 时为 True；控制最终第二档 argmax 掩码 | `true` |

104 个 Stage3 `(pdb_id, prepared_smiles)` 必须全部存在于这 180 个身份中，并且对应 `allowed_as_small_molecule_answer=true`。127 个允许答案的身份包含 Stage3 真值和其它受支持 small-molecule 干扰身份，因此该掩码不会直接泄露 Stage3 正确答案。

下面是说明字段关系的构造记录，不代表正式 manifest 中的真实身份：

```json
{"allowed_as_small_molecule_answer":true,"language_relative_path":"stage1_preparation_box_pool_2/ligand_language_models/smi_ted_289m/11jb/candidate_0.npz","occurrence_ids":[0,1],"pdb_id":"11jb","smiles":"CCO","stage3_occurrence_ids":[0],"type_tag":"small_molecule"}
```

## Stage1 候选

这一组保存两类 Find_1 候选和一类 Emap2lig 候选。三份文件使用相同的候选目标字段，因此可交给同一个 top-k 状态机；只有 Find_1 候选能送入 Matcher 身份前向。

### `real_candidates.jsonl`

每条记录对应真实受体 Find_1 `held_out_test_0` 中一个 centered 候选。记录继承完整 `test_0` manifest 的唯一 occurrence 目标，并增加 Stage3 三态范围、完整 blob 世界中心和框内状态。

### `cryoatom2_candidates.jsonl`

字段与 `real_candidates.jsonl` 严格同构，但每条记录来自 CryoAtom2 受体条件。两个文件的候选集合、中心、排序分数和目标可能不同，不能按 JSONL 位置逐项配对。

### `emap2lig_candidates.jsonl`

每条记录对应一个 Emap2lig 原生候选。它没有 Find_1 centered 框，因此 `centered_box_index`、`gaussian_score` 和 `source_blob_fits_centered_box` 均为 `null`。正式排序只使用实例内原生 ligand probability 均值 `source_probability_mean`。

### 三类候选的共同字段

| 字段 | 数据类型 | 具体含义与特殊值 | 示例值 |
| --- | --- | --- | --- |
| `pdb_id` | 字符串 | 候选所属小写 PDB | `"9rd8"` |
| `centered_box_index` | 整数或 null | Find_1 候选在 `F2_centered.npz` 的稳定编号；Emap2lig 为 `null` | `0` |
| `source_blob_index` | 整数 | 指向当前来源完整 blob 数组；与 `pdb_id` 共同构成当前来源内候选唯一键 | `53` |
| `candidate_selected` | 布尔值 | True 候选整体排在 False 候选之前 | `true` |
| `gaussian_score` | 有限浮点或 null | Find_1 Gaussian 段内排序分数；Emap2lig 为 `null` | `1.7736837` |
| `source_probability_mean` | 有限浮点 | 完整来源 blob 内原生概率均值 | `0.98322934` |
| `foreground_target` | 布尔值 | 是否有 occurrence 同时达到两项 0.30 覆盖门槛 | `true` |
| `target_category` | 字符串或 null | 唯一目标的原始 `type_tag`；背景为 `null` | `"small_molecule"` |
| `matched_occurrence_id` | 整数或 null | 最大几何平均并按较小编号决胜后的唯一 occurrence；背景为 `null` | `0` |
| `target_smiles` | 字符串或 null | 唯一目标的精确顶层 SMILES；背景为 `null` | `"CCO"` |
| `candidate_coverage` | 浮点 | 候选与目标交集体素数除以完整候选体素数；背景为 `0.0` | `0.73` |
| `occurrence_coverage` | 浮点 | 候选与目标交集体素数除以目标 occurrence 体素数；背景为 `0.0` | `0.89` |
| `overlap_score` | 浮点 | 两项覆盖率的几何平均；背景为 `0.0` | `0.8060` 是格式示例 |
| `target_scope` | 字符串 | `background`、`non_stage3_foreground` 或 `stage3_foreground` | `"stage3_foreground"` |
| `blob_center_world_xyz_A` | 浮点列表，长度 3 | 完整 blob 的世界 XYZ 体素中心均值，单位 Å | `[452.8,505.8,559.7]` |
| `source_blob_fits_centered_box` | 布尔值或 null | Find_1 完整 blob 是否全部位于 centered 80³ 框；Emap2lig 为 `null` | `true` |

`foreground_target=false` 时，`target_category`、`matched_occurrence_id` 和 `target_smiles` 必须都是 `null`，三项覆盖数值都是 `0.0`，`target_scope="background"`。有唯一前景目标时，`foreground_target=true`，目标字段均非空；`target_scope` 只说明该 occurrence 是否属于 Stage3 冻结测试范围。

下面是说明字段和数值关系的构造 Find_1 记录，不代表正式 manifest 中的真实候选：

```json
{"blob_center_world_xyz_A":[452.8,505.8,559.7],"candidate_coverage":0.73,"candidate_selected":true,"centered_box_index":0,"foreground_target":true,"gaussian_score":1.7736837,"matched_occurrence_id":0,"occurrence_coverage":0.89,"overlap_score":0.8060397013,"pdb_id":"9rd8","source_blob_fits_centered_box":true,"source_blob_index":53,"source_probability_mean":0.98322934,"target_category":"small_molecule","target_scope":"stage3_foreground","target_smiles":"CCO"}
```

`9rd8` 真实受体的 `source_blob_index=0` 使用既有最大覆盖目标 occurrence 0。即使该 blob 也覆盖 Stage3 occurrence 1，唯一目标仍不变，因此 `target_scope="non_stage3_foreground"`，top-k 状态机免费跳过它。

## 跨文件编号、对齐与排序

- `stage3_occurrences.jsonl.candidate_id`、`identities.jsonl.occurrence_ids` 和候选文件的 `matched_occurrence_id` 都指向 AdaLigand 原始 occurrence 编号。
- `source_blob_index` 只指向各自 Stage1 来源的完整 blob 数组；同一数值在真实受体、CryoAtom2 和 Emap2lig 之间没有共享身份。
- `centered_box_index` 只索引 Find_1 `F2_centered.npz` 的候选轴；它不索引 occurrence 或完整 blob。
- `identities.jsonl` 按 `(pdb_id, smiles)` 为最终答案掩码提供类别事实；模型前向的实际 `identity_smiles` 顺序来自对应完整 `test_0` manifest，评测按精确 `(pdb_id, smiles)` 键逐槽生成 `small_molecule_answer_mask`。
- 每个 PDB 先排列全部 `candidate_selected=true` 候选，再追加 `false` 候选；每段按当前分数降序、`source_blob_index` 升序。
- `blob_center_world_xyz_A` 始终从完整来源 blob 计算，不因 `source_blob_fits_centered_box=false` 改用 centered 框中心。

## 来源留证

### `source_snapshot.json`

该 JSON 顶层为对象，用于绑定构建时实际读取的每一个普通文件，而不是只记录来源目录。

| 字段 | 数据类型或结构 | 具体含义 | 示例值 |
| --- | --- | --- | --- |
| `schema` | 字符串 | 来源快照契约版本 | `"matcher_test_smallmolecule_source_snapshot_v1"` |
| `files` | 对象列表 | 按绝对路径排序、去重后的实际来源文件 | 见下方记录示例 |
| `files[].path` | 字符串 | 来源文件绝对路径 | `"/storage/penghongen/PocketXMol/data/smiles-v1/frozen/test.jsonl"` |
| `files[].size_bytes` | 整数 | 来源文件字节数 | `1234` 是格式示例 |
| `files[].sha256` | 字符串 | 来源文件内容的 64 位十六进制摘要 | `"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"` 是构造格式示例 |

下面是说明结构的构造示例，不代表一次正式运行的实际摘要：

```json
{
  "files": [
    {
      "path": "/storage/project/frozen/test.jsonl",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "size_bytes": 1234
    }
  ],
  "schema": "matcher_test_smallmolecule_source_snapshot_v1"
}
```

任何来源内容或派生文件内容变化都必须建立新的版本目录，不能覆盖 `test_smallmolecule_v1`。`manifest.json.file_sha256` 绑定派生文件，`source_snapshot.json` 绑定外部输入，两者职责不同。
