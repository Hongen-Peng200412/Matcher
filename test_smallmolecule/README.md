# Stage3 小分子测试集评估

本目录保存 Matcher 在 PocketXMol Stage3 小分子测试集上的正式清单、Stage1-only 统计、身份前向和绘图代码。代码只生成科学评估产物；checkpoint、release、launch、命令、服务器作业和文件摘要由实验追溯文档记录，不写入 JSON 或 JSONL。

该测试固定使用 PocketXMol Stage3 的 77 个测试 PDB 和 446 个 small-molecule occurrence，独立于 Matcher 的完整 179-PDB held-out test_0 指标。真实受体 Find_1、CryoAtom2 Find_1 和 Emap2lig 各自保留原候选集合；Matcher 只评估 Find_1 中唯一目标属于 Stage3 测试范围的候选。

## 输入、处理与输出概览

`test_smallmolecule.yaml` 集中指定以下只读输入：

- PocketXMol Stage3 冻结 `test.jsonl`，提供 446 个测试 occurrence 及精确 `prepared_smiles`。
- 真实受体与 CryoAtom2 两套 Find_1 held-out test_0 候选、完整 blob 体素和 centered 几何。
- Emap2lig 原生候选、概率均值及候选与真实 occurrence 的体素交集。
- 两套 Matcher test_0 manifest 中的完整 PDB 内受支持 SMILES 身份轴。
- 每个实验的冻结 BEST checkpoint；模型类、模型配置和 checkpoint 解释从对应训练代码快照恢复。

```text
PocketXMol Stage3 occurrence + Matcher 身份轴 + 三套 Stage1 候选
    ↓ build_manifest.py
模型无关清单：occurrence、身份和候选三态
    ↓ evaluate_stage1.py
Stage1-only top-k 曲线与逐候选 attempt 轨迹
    ↓ evaluate_model.py + 冻结 BEST
两档 SMILES 答案范围的预测、Stage1+Matcher 曲线与 docking handoff
    ↓ plot_results.py
只由冻结 JSON 重建的 PNG 图片
```

`build_manifest.py`、`evaluate_stage1.py` 和 `evaluate_model.py` 不修改任何来源文件。`plot_results.py` 不加载模型，也不重新计算候选顺序或命中。

## 代码组织与阅读顺序

```text
test_smallmolecule/
├── build_manifest.py          # 连接 Stage3 occurrence、Matcher test_0 和三套 Stage1 候选
├── ranking.py                 # 生成 selected 优先顺序、attempt 轨迹和 top-k 曲线
├── evaluate_stage1.py         # 生成 Find_1 与 Emap2lig 的 Stage1-only 结果
├── evaluate_model.py          # 加载训练代码快照并生成 Matcher 预测与 PocketXMol handoff
├── plot_results.py            # 只读取冻结 JSON 绘制两种排序的 top-k 图片
├── artifacts.py              # 读取 YAML、JSON、JSONL 并写出 JSON、JSONL
├── test_smallmolecule.yaml    # 只读来源、产物根、前向参数和冻结 checkpoint 输入
└── MANIFEST_README.md         # manifest 目录的字段契约，构建时复制到产物目录
```

推荐先读 `build_manifest.py` 中的 `_target_from_intersections()` 和 `build_manifest()`，再读 `ranking.py::build_topk_curve()`。清单的逐文件、逐字段及跨文件编号契约见 [`MANIFEST_README.md`](MANIFEST_README.md)。`evaluate_stage1.py` 与 `evaluate_model.py` 使用相同候选顺序。

## 科学规则

### 唯一目标

候选与 occurrence 的候选覆盖率和 occurrence 覆盖率都不低于 0.30 时，该 occurrence 可以成为目标。若多个 occurrence 同时满足门槛，先选择两项覆盖率几何平均最大的 occurrence；并列时选择较小的 `occurrence_id`。固定 top-k 分母为 77 个 PDB，包括没有候选或没有命中的 PDB。

### 候选顺序与 attempt

每个 PDB 先排列全部 `candidate_selected=true` 候选，再追加全部 `candidate_selected=false` 候选。两段内部均按指定分数降序排列；分数并列时按较小的 `source_blob_index` 排列。

- `background`：消耗一次 attempt，不能命中。
- `non_stage3_foreground`：不属于 Stage3 小分子测试范围，不消耗 attempt。
- `stage3_foreground`：消耗一次 attempt；Stage1-only 直接命中，Stage1+Matcher 还要求预测 SMILES 与唯一目标 SMILES 完全相同。

Gaussian 曲线使用 `gaussian_score`；概率均值曲线使用 `source_probability_mean`。Matcher 不改变 Stage1 候选顺序。

### 身份答案范围

Matcher 对每个 Stage3 前景候选只执行一次完整身份前向。同一组 `identity_logits` 生成两种答案：

- `all_ligand`：在当前 PDB 的全部受支持 SMILES 身份中取最大分数。
- `small_molecule`：只保留原始类别为 `small_molecule` 的身份后取最大分数。

前景头和 validation 前景阈值不参与本测试。

当前完整受支持身份轴包含 Stage3 真值身份和同 PDB 的干扰身份。`small_molecule_answer_mask` 只依据原始 `type_tag` 限制最终 argmax；非 small-molecule 身份仍参与同一次身份编码和 inventory 注意力。该掩码不把答案集合缩减成 Stage3 真值列表。

### blob 世界中心

Find_1 与 Emap2lig 都从完整来源 blob 的全局 ZYX 体素索引计算体素中心算术平均，再按来源网格的原点和 XYZ spacing 转换为 `blob_center_world_xyz_A`。输出三个分量依次为世界 X、Y、Z，单位 Å。该中心不是 Find_1 centered 80³ 框的中心；即使完整 blob 不能完全装入 centered 框，仍使用完整 blob 计算。

## 科学产物

输出根由 `test_smallmolecule.yaml` 的 `outputs` 决定。

```text
## <科学产物>
<manifest_root>/
├── manifest.json                                      # 77-PDB 轴、覆盖门槛和派生文件索引
├── stage3_occurrences.jsonl                           # 446 个 PocketXMol Stage3 测试 occurrence
├── identities.jsonl                                  # PDB 内完整受支持 SMILES 身份轴
├── real_candidates.jsonl                             # 真实受体 Find_1 候选及 Stage3 三态目标
├── cryoatom2_candidates.jsonl                        # CryoAtom2 Find_1 候选及 Stage3 三态目标
└── emap2lig_candidates.jsonl                         # Emap2lig 原生候选及 Stage3 三态目标

<evaluation_root>/
├── stage1_only/
│   └── stage1_only.json                              # 三套 Stage1 来源的 top-k 曲线与候选轨迹
└── <experiment_id>/<training_run_name>/
    ├── evaluation.json                               # 两档身份答案范围的预测、曲线和轨迹
    └── docking_handoff_candidates.jsonl              # 身份预测正确的 Stage3 候选评测交接记录

## <其他文件>
<manifest_root>/
└── README.md                                         # MANIFEST_README.md 的发布副本（可再生说明）

<evaluation_root>/
├── stage1_only/figures/
│   ├── topk_gaussian_stage1_only.png                 # Gaussian 排序的 Stage1-only 曲线（可再生辅助导出）
│   └── topk_probability_mean_stage1_only.png         # 概率均值排序的 Stage1-only 曲线（可再生辅助导出）
└── <experiment_id>/<training_run_name>/figures/
    ├── topk_gaussian_stage1_stage2.png               # Gaussian 排序的 Stage1 与两档 Matcher 曲线（可再生辅助导出）
    └── topk_probability_mean_stage1_stage2.png       # 概率均值排序的 Stage1 与两档 Matcher 曲线（可再生辅助导出）
```

两组 `<evaluation_root>` 指向同一物理目录，只分别列出稳定 JSON/JSONL 与可由它们重建的图片。PNG 不能替代冻结 JSON 科学产物。

### `stage1_only.json`

| 字段 | 类型与结构 | 含义 | 示例值 |
| --- | --- | --- | --- |
| `coverage_threshold` | 浮点数 | 候选和 occurrence 共用的双向覆盖门槛 | `0.3` |
| `curves` | 两层映射 | 第一层是 `gaussian` 或 `source_probability_mean`，第二层是 `real_receptor` 或 `cryoatom2_receptor` | `curves["gaussian"]["real_receptor"]` |
| `emap2lig` | 曲线映射 | Emap2lig 原生概率均值的单条 receptor-independent 曲线 | `emap2lig["top_k"][0] == 1` |
| `traces` | 嵌套映射 | 每种来源和排序下逐 PDB 的完整候选顺序、计费位置与命中状态 | `traces["gaussian"]["real_receptor"]["9abc"]` |

`curves` 的两种排序都包含 `real_receptor` 和 `cryoatom2_receptor`。`emap2lig` 只使用原生概率均值排序，因此顶层只保存一条曲线，不复制成两种受体条件。`traces.gaussian` 和 `traces.source_probability_mean` 使用与 `curves` 相同的两种 Find_1 受体条件；`traces.emap2lig` 直接按 PDB 保存 Emap2lig 候选轨迹。

#### 曲线字段

一条曲线对应一种候选来源、受体条件、排序方式和答案范围。小分子测试集的固定分母为 77 个 PDB；没有候选或没有命中的 PDB 仍保留在 `pdb_ids` 和 `pdb_hit` 中。

| 字段 | 数据类型与形状 | 含义与对齐关系 | 示例值 |
| --- | --- | --- | --- |
| `pdb_ids` | 长度 77 的字符串列表 | 曲线分母中的有序小写 PDB 标识；`pdb_hit` 必须恰好包含这些键 | 格式片段如 `["9abc", "9def"]` |
| `top_k` | 长度 20 的整数列表 | 计费尝试次数，固定为 1 至 20；第 `i` 项与其余三个列表的第 `i` 项对齐 | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]` |
| `hit_count` | 长度 20 的整数列表 | 每个 `top_k` 下已经命中的 PDB 数量，数值范围为 0 至 77，并随 `k` 单调不减 | 格式片段如 `[21, 30, 35]`，分别对应 `k=1,2,3` |
| `hit_ratio` | 长度 20 的浮点列表 | `hit_count[i] / 77`；与 `top_k` 逐项对齐，无量纲 | `hit_count[0]=21` 时首项为 `21 / 77 = 0.272727...` |
| `pdb_hit` | 具有 77 个键的映射；每个值是长度 20 的布尔列表 | `pdb_hit[pdb_id][i]` 表示该 PDB 是否在不超过 `top_k[i]` 次计费尝试内首次命中；一旦变为 `true`，后续项保持 `true` | `{"9abc": [false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]}` |

#### 候选轨迹字段

一条候选轨迹记录对应排序后的一项 Stage1 候选。轨迹保存完整候选顺序，因此 `non_stage3_foreground` 即使不消耗尝试次数也不会从列表中删除。

| 字段 | 数据类型 | 含义 | 示例值或空值规则 |
| --- | --- | --- | --- |
| `source_blob_index` | 整数 | 候选在当前 PDB、当前 Stage1 来源中的完整 blob 编号；与清单候选的同名字段保持一致 | `17` |
| `centered_box_index` | 整数或 `null` | Find_1 centered 候选编号；Emap2lig 没有 centered 框，因此为 `null` | Find_1 如 `4`；Emap2lig 为 `null` |
| `raw_rank` | 正整数 | 当前候选在 selected 优先的完整排序中的一基位置；免费跳过的候选仍占据该排名 | `3` 表示完整排序中的第 3 项 |
| `attempt_index` | 正整数或 `null` | 当前候选消耗的累计尝试编号；`background` 和 `stage3_foreground` 递增计数，`non_stage3_foreground` 为 `null` | `2`；免费跳过时为 `null` |
| `action` | 枚举字符串 | 当前候选的计费动作：`spend_background`、`skip_non_stage3_foreground` 或 `spend_stage3_target` | `"spend_stage3_target"` |
| `target_scope` | 枚举字符串 | 唯一目标与 Stage3 测试范围的关系：背景、范围外真实前景或 Stage3 真实前景 | `"stage3_foreground"` |
| `matched_occurrence_id` | 整数或 `null` | 满足双向覆盖门槛并按几何平均选出的唯一 occurrence 编号；背景为 `null` | `6` |
| `target_smiles` | 字符串或 `null` | 唯一 occurrence 的精确顶层 `PreparedLigand.smiles`；背景为 `null` | `"CCO"` |
| `success` | 布尔值 | 当前候选是否形成命中；Stage1-only 的 Stage3 真实前景为 `true`，Stage1+Matcher 还要求预测 SMILES 与 `target_smiles` 相同 | `true` |

下面是说明字段关系的构造示例，不是正式运行观测。该候选位于完整排序第 3 位；前面有一个免费跳过项，因此它只消耗第 2 次尝试。

```json
{
  "action": "spend_stage3_target",
  "attempt_index": 2,
  "centered_box_index": 4,
  "matched_occurrence_id": 6,
  "raw_rank": 3,
  "source_blob_index": 17,
  "success": true,
  "target_scope": "stage3_foreground",
  "target_smiles": "CCO"
}
```

### `evaluation.json`

顶层只包含以下三个科学结果：

| 字段 | 类型与结构 | 含义 | 示例值 |
| --- | --- | --- | --- |
| `predictions` | 受体条件到候选列表的映射 | 保存完整身份分数、两种答案范围的预测 SMILES、唯一目标和 blob 世界中心 | `predictions["real_receptor"][0]` |
| `curves` | 排序、受体条件和答案范围的三层映射 | 保存 Stage1+Matcher 的 top-k 曲线 | `curves["gaussian"]["real_receptor"]["small_molecule"]` |
| `traces` | 与 `curves` 同键的三层映射 | 保存每个 PDB 的完整候选顺序和 attempt 状态 | `traces["source_probability_mean"]["cryoatom2_receptor"]["all_ligand"]` |

`curves` 和 `traces` 的第一层是 `gaussian` 或 `source_probability_mean`，第二层是 `real_receptor` 或 `cryoatom2_receptor`，第三层是 `all_ligand` 或 `small_molecule`。每条曲线和每条候选轨迹使用前述字段契约。

#### `predictions.<condition>[]` 候选预测字段

`predictions.real_receptor` 和 `predictions.cryoatom2_receptor` 都是候选列表，只包含唯一目标属于 Stage3 测试范围的候选。形状符号 $I$ 表示当前候选所属 PDB 的受支持 SMILES 身份数量。

| 字段 | 数据类型与形状 | 含义与对齐关系 | 示例值 |
| --- | --- | --- | --- |
| `pdb_id` | 字符串 | 候选所属的小写 PDB 标识 | `"9abc"` |
| `centered_box_index` | 整数 | 候选在当前 PDB 的 Find_1 centered 候选轴上的编号 | `4` |
| `source_blob_index` | 整数 | centered 候选对应的完整 Stage1 blob 编号 | `17` |
| `candidate_selected` | 布尔值 | 是否属于 Stage1 原 selected 集合；该字段决定候选进入排序前段还是追加段，不决定是否执行身份前向 | `true` |
| `matched_occurrence_id` | 整数 | 当前候选按双向覆盖和几何平均规则选出的唯一 Stage3 occurrence 编号 | `6` |
| `target_smiles` | 字符串 | 唯一目标 occurrence 的精确顶层 `PreparedLigand.smiles` | `"CCO"` |
| `blob_center_world_xyz_A` | 长度 3 的浮点列表 | 完整来源 blob 的世界坐标中心；三个分量依次为 X、Y、Z，单位 Å | `[12.5, -3.0, 8.25]` |
| `identity_smiles` | 长度 $I$ 的字符串列表 | 当前 PDB 的完整受支持身份轴；第 `j` 项与 `identity_logits[j]` 和 `small_molecule_answer_mask[j]` 对齐 | `["CC", "CCO", "O"]` |
| `identity_logits` | 长度 $I$ 的浮点列表 | Matcher 对每个身份槽给出的未归一化匹配分数；数值越大表示匹配偏好越强 | `[-0.4, 2.1, 0.3]` |
| `small_molecule_answer_mask` | 长度 $I$ 的布尔列表 | `true` 表示同位置身份的原始类别为 `small_molecule`，可以参与 small-molecule 答案范围的最大值选择 | `[true, true, false]` |
| `predicted_smiles_all_ligand` | 字符串 | 在全部 $I$ 个身份槽中取得最大 `identity_logits` 的精确 SMILES | `"CCO"` |
| `predicted_smiles_small_molecule` | 字符串 | 只在 `small_molecule_answer_mask=true` 的身份槽中取得最大分数的精确 SMILES | `"CCO"` |

下面是字段齐全的构造示例。三个身份相关列表长度均为 3，且最大分数 `2.1` 对应 `identity_smiles[1]="CCO"`。

```json
{
  "blob_center_world_xyz_A": [12.5, -3.0, 8.25],
  "candidate_selected": true,
  "centered_box_index": 4,
  "identity_logits": [-0.4, 2.1, 0.3],
  "identity_smiles": ["CC", "CCO", "O"],
  "matched_occurrence_id": 6,
  "pdb_id": "9abc",
  "predicted_smiles_all_ligand": "CCO",
  "predicted_smiles_small_molecule": "CCO",
  "small_molecule_answer_mask": [true, true, false],
  "source_blob_index": 17,
  "target_smiles": "CCO"
}
```

### `docking_handoff_candidates.jsonl`

每行对应一个预测正确的 Stage3 前景候选。同一物理候选可以因两种排序或两种答案范围分别出现；`sorting`、`answer_scope` 和 `receptor_condition` 共同说明该记录属于哪一种评估条件。该文件是带测试真值筛选的评估交接文件，不是无真值生产推理接口。

| 字段 | 数据类型 | 含义 | 示例值 |
| --- | --- | --- | --- |
| `pdb_id` | 字符串 | 候选所属的小写 PDB 标识 | `"9abc"` |
| `receptor_condition` | 枚举字符串 | Stage1 候选使用的受体条件，取 `real_receptor` 或 `cryoatom2_receptor` | `"real_receptor"` |
| `sorting` | 枚举字符串 | 候选段内排序字段，取 `gaussian` 或 `source_probability_mean` | `"source_probability_mean"` |
| `answer_scope` | 枚举字符串 | Matcher 身份最大值的允许范围，取 `all_ligand` 或 `small_molecule` | `"small_molecule"` |
| `source_blob_index` | 整数 | 完整 Stage1 blob 编号；与 `evaluation.json` 的候选预测及轨迹同名字段联合定位 | `17` |
| `centered_box_index` | 整数 | Find_1 centered 候选编号 | `4` |
| `raw_rank` | 正整数 | 候选在当前排序的完整 Stage1 候选列表中的一基位置 | `3` |
| `attempt_index` | 正整数 | 去除免费跳过项后的累计尝试编号；handoff 只保存成功的 Stage3 候选，因此该字段不会为 `null` | `2` |
| `blob_center_world_xyz_A` | 长度 3 的浮点列表 | 交给 PocketXMol 的世界坐标中心；三个分量依次为 X、Y、Z，单位 Å | `[12.5, -3.0, 8.25]` |
| `predicted_smiles` | 字符串 | 当前答案范围内预测正确的精确 SMILES；与对应候选的 `target_smiles` 相同 | `"CCO"` |

下面是字段齐全的构造示例，不包含 checkpoint、manifest 或文件摘要等追溯信息。

```json
{
  "answer_scope": "small_molecule",
  "attempt_index": 2,
  "blob_center_world_xyz_A": [12.5, -3.0, 8.25],
  "centered_box_index": 4,
  "pdb_id": "9abc",
  "predicted_smiles": "CCO",
  "raw_rank": 3,
  "receptor_condition": "real_receptor",
  "sorting": "source_probability_mean",
  "source_blob_index": 17
}
```

## 正式入口

正式 shell 位于 `训练与运行/sh/small_molecule_test/`。清单、Stage1-only、模型前向和绘图相互分离；绘图失败不需要重新执行模型前向。本目录不读取 calibration，也不在测试集拟合阈值。
