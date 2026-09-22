# Matcher 评估

本目录计算 validation 指标、冻结全测试集预测和正式图片。测试只使用 BEST checkpoint 在 validation 上拟合并保存的前景阈值，不读取 calibration，也不在测试集重新拟合阈值。

Stage3 小分子专用测试位于 [`../../test_smallmolecule/`](../../test_smallmolecule/)。两套评估共享 selected 优先的候选顺序，但使用不同的 PDB 集合和命中定义。

## 代码组织与阅读顺序

```text
matcher/evaluation/
├── metrics.py       # 前景 PRAUC、固定阈值 F1、SMILES 准确率和 selection
├── checkpoint.py    # 从训练 run 指向的代码快照恢复模型与 checkpoint
├── test_set.py      # Stage1-only 曲线、冻结 Matcher 前向、test_1 筛选和 top-k
├── plotting.py      # 只读取冻结 JSON 绘制指标图与 top-k 图
└── __init__.py      # 包职责说明
```

推荐先读 `metrics.py`，再读 `test_set.py` 中的 `selected_priority_order()`、`build_stage1_only_preview_metadata()` 和 `build_final_test_artifact()`。`plotting.py` 不加载模型或来源 NPZ，也不重新计算科学结果。

## 正式产物目录

通用评估的正式服务器根是 `/storage/penghongen/Matcher/evaluation/`。下列两棵树指向同一物理根，只分别列出稳定科学接口和可由冻结 JSON 重建的图片。`<experiment_dir>` 是一个稳定实验目录，如 `strongest-1` 或 `density_only-106`。

```text
## <科学产物>
/storage/penghongen/Matcher/evaluation/
├── stage1_only_preview/
│   └── stage1_only_preview.json                     # 179-PDB Stage1-only 曲线，不含 Matcher 预测
└── final_smiles/
    └── <experiment_dir>/
        └── frozen_test.json                         # test_0 指标、test_1 子集指标、逐候选预测和正式 top-k 曲线

## <其他文件>
/storage/penghongen/Matcher/evaluation/
├── stage1_only_preview/
│   ├── topk_gaussian_stage1_only_preview.png        # Gaussian 排序 Stage1-only 预览（可再生辅助导出）
│   └── topk_source_probability_mean_stage1_only_preview.png # 概率均值排序 Stage1-only 预览（可再生辅助导出）
└── final_smiles/
    └── <experiment_dir>/
        ├── foreground_and_smiles_metrics.png        # 前景与 SMILES 指标图（可再生辅助导出）
        ├── topk_gaussian_final.png                  # Gaussian 排序正式 top-k 图（可再生辅助导出）
        └── topk_source_probability_mean_final.png   # 概率均值排序正式 top-k 图（可再生辅助导出）
```

Stage3 小分子专用产物位于独立的 `/storage/penghongen/Matcher/evaluation/test_smallmolecule_v1/`，其 77-PDB 分母、attempt 计费和 handoff 契约见 [`../../test_smallmolecule/README.md`](../../test_smallmolecule/README.md)。两套评估不能仅因字段同名而混用分母或命中定义。

## 指标定义

- 前景 PRAUC：全部 selected 候选的二分类 average precision。
- 前景 F1：使用 BEST checkpoint 中由 validation 拟合的固定阈值。
- SMILES 匹配准确率：只统计 `foreground_target=true` 且 `identity_supported=true` 的 selected 候选。
- 五类分层：`ion`、`nucleotide_like`、`peptide_like`、`small_molecule` 和 `sugar`；零支持时 `accuracy=null`。
- validation 选模分数：`(0.3 × 前景 PRAUC + SMILES 匹配准确率) / 1.3`。前景 F1 不参与选模。

`9ian` 与 `9oq7` 的部分真前景缺少严格图和语言资产。这些候选保留前景真值与 Stage1 排名，不进入 SMILES 准确率分母，也不能形成 Stage1+Matcher 命中。

## 候选顺序

每个 PDB 先排列全部 `candidate_selected=true` 候选，再追加全部 `candidate_selected=false` 候选。两段内部均按 Gaussian `candidate_score` 或 `source_probability_mean` 降序排列；分数并列时按较小的 `source_blob_index` 排列。Stage1-only 与 Stage1+Matcher 使用同一候选顺序，Matcher 不重新排名。

## Stage1-only 预览

`build_stage1_only_preview_metadata()` 写出一个 JSON 对象，顶层字段如下：

| 字段 | 类型与结构 | 含义 | 示例值 |
| --- | --- | --- | --- |
| `coverage_threshold` | 浮点数 | 候选和 occurrence 共用的双向覆盖门槛 | `0.3` |
| `denominator_pdb_count` | 整数 | top-k 固定 PDB 分母 | `179` |
| `curves.gaussian` | 具有 3 个条件键的曲线映射 | 真实受体 Find_1、CryoAtom2 Find_1 和 Emap2lig 的 Gaussian 图数据；Emap2lig 仍复用自身唯一原生概率均值曲线 | `curves["gaussian"]["real_receptor"]` |
| `curves.source_probability_mean` | 具有 3 个条件键的曲线映射 | 同三套来源的概率均值图数据 | `curves["source_probability_mean"]["emap2lig"]` |

两种排序映射都含 `real_receptor`、`cryoatom2_receptor` 和 `emap2lig`。每张图只绘制一条不依赖受体模型的 Emap2lig 曲线；两种排序位置中的 Emap2lig 记录逐字段相同。

### Stage1-only 曲线字段

一条曲线对应一种候选来源、受体条件和排序方式。固定分母是 179 个 held-out test_0 PDB；没有 centered 候选的 PDB 仍保留在 `pdb_ids` 和 `pdb_hit` 中。

| 字段 | 数据类型与形状 | 含义与对齐关系 | 示例值 |
| --- | --- | --- | --- |
| `name` | 字符串 | 图例使用的完整名称；Emap2lig 名称明确标记 density-only 和 receptor-independent | `"Find_1 real-receptor Stage1-only"` |
| `pdb_count` | 整数 | 曲线固定 PDB 分母，当前为 179 | `179` |
| `pdb_ids` | 长度 179 的字符串列表 | 分母中的有序小写 PDB 标识；`pdb_hit` 必须恰好包含这些键 | 格式片段如 `["9abc", "9def"]` |
| `top_k` | 长度 20 的整数列表 | 候选数量，固定为 1 至 20；第 `i` 项与 `hit_count[i]` 和 `hit_ratio[i]` 对齐 | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]` |
| `hit_count` | 长度 20 的整数列表 | 在扩展顺序前 `top_k[i]` 个候选中至少命中一次的 PDB 数量，范围为 0 至 179，并随 `k` 单调不减 | 格式片段如 `[80, 92, 101]` |
| `hit_ratio` | 长度 20 的浮点列表 | `hit_count[i] / 179`；与 `top_k` 逐项对齐，无量纲 | `hit_count[0]=80` 时首项为 `80 / 179 = 0.446927...` |
| `pdb_hit` | 具有 179 个键的映射；每个值是长度 20 的布尔列表 | `pdb_hit[pdb_id][i]` 表示该 PDB 在 top-`top_k[i]` 内是否至少出现一个命中；首次变为 `true` 后，后续项保持 `true` | `{"9abc": [false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]}` |

预览产物不包含 Matcher 概率、预测 SMILES 或 Stage1+Matcher 数值。

## 冻结全测试集结果

`build_final_test_artifact()` 对真实受体和 CryoAtom2 各执行一次 Matcher 前向。顶层只保存科学结果：

| 字段 | 类型与结构 | 含义 | 示例值 |
| --- | --- | --- | --- |
| `metrics` | 受体条件映射 | test_0 的前景和 SMILES 指标 | `metrics['real_receptor']['foreground_prauc']` |
| `predictions` | 受体条件与 PDB 映射 | 每个 centered 候选的真值、预测 SMILES 和前景概率 | `predictions['cryoatom2_receptor']['9abc'][0]` |
| `curves` | 排序映射 | 两种受体条件的 Stage1-only、Stage1+Matcher 和 Emap2lig 曲线 | `curves['gaussian']['real_stage1_matcher']` |
| `test_1` | 可选指标映射 | 复用同一次 test_0 前向筛选官方 149-PDB 子集 | `test_1['metrics']['real_receptor']` |

### `metrics.<condition>`

`condition` 取 `real_receptor` 或 `cryoatom2_receptor`。所有候选指标只统计 `candidate_selected=true` 的候选；未选择候选仍保留在 `predictions` 中并参与扩展 top-k。

| 字段 | 数据类型与结构 | 含义 | 示例值或空值规则 |
| --- | --- | --- | --- |
| `candidate_count` | 非负整数 | 当前 PDB 集合中进入指标计算的 selected 候选数量 | `1029` |
| `foreground_prauc` | 浮点数 | 把五类真实配体合并为前景后的 average precision，无量纲，通常位于 0 至 1 | `0.71` |
| `foreground_threshold` | 浮点数 | BEST checkpoint 在 validation 上拟合并保存的前景概率阈值；测试仅应用该值 | `0.43` |
| `foreground` | 对象映射 | 固定 `foreground_threshold` 下的四项混淆计数与 F1 | `{"false_negative": 18, "false_positive": 27, "f1": 0.79, "true_negative": 900, "true_positive": 84}` |
| `smiles_match` | 具有 6 个分组键的映射 | `overall` 与五类配体各自的正确数、支持数和准确率 | `smiles_match["small_molecule"]["eligible_count"] == 73` |
| `pdb_count` | 正整数 | 指标覆盖的 PDB 数；test_0 为 179，test_1 为 149，并包含零候选 PDB | `179` |
| `unsupported_identity_target_candidate_count` | 非负整数 | 真实前景但目标身份缺少合法 SMILES 图或 SMI-TED 表征的 selected 候选数量 | `3` |
| `unsupported_identity_targets` | 候选定位记录列表 | 与上一计数逐项对应；每项定位一个不进入 SMILES 准确率分母的真前景候选 | 见下方字段表 |

#### `foreground` 子字段

| 字段 | 数据类型 | 含义 | 示例值 |
| --- | --- | --- | --- |
| `true_positive` | 非负整数 | 真实前景且 `foreground_probability >= foreground_threshold` 的候选数 | `84` |
| `false_positive` | 非负整数 | 真实背景但前景概率达到阈值的候选数 | `27` |
| `false_negative` | 非负整数 | 真实前景但前景概率低于阈值的候选数 | `18` |
| `true_negative` | 非负整数 | 真实背景且前景概率低于阈值的候选数 | `900` |
| `f1` | 浮点数 | `2 * true_positive / (2 * true_positive + false_positive + false_negative)` | 上述构造计数对应 `168 / 213 = 0.788732...` |

#### `smiles_match.<group>` 子字段

`group` 取 `overall`、`ion`、`nucleotide_like`、`peptide_like`、`small_molecule` 或 `sugar`。每组只统计 `foreground_target=true` 且 `identity_supported=true` 的 selected 候选。

| 字段 | 数据类型 | 含义 | 示例值或空值规则 |
| --- | --- | --- | --- |
| `correct_count` | 非负整数 | 当前分组中 `predicted_smiles == target_smiles` 的候选数 | `61` |
| `eligible_count` | 非负整数 | 当前分组进入 SMILES 准确率分母的受支持真前景候选数 | `73` |
| `accuracy` | 浮点数或 `null` | `correct_count / eligible_count`；`eligible_count=0` 时为 `null`，表示 N/A 而不是准确率 0 | `61 / 73 = 0.835616...`；零支持时为 `null` |

#### `unsupported_identity_targets[]` 子字段

| 字段 | 数据类型 | 含义 | 示例值或空值规则 |
| --- | --- | --- | --- |
| `pdb_id` | 字符串 | 不受支持候选所属的小写 PDB 标识 | `"9ian"` |
| `centered_box_index` | 非负整数 | 当前 PDB centered 候选轴上的编号 | `2` |
| `source_blob_index` | 非负整数 | 对应的完整 Stage1 blob 编号 | `7` |
| `matched_occurrence_id` | 非负整数 | 双向覆盖与几何平均规则确定的唯一 occurrence 编号 | `0` |
| `target_smiles` | 字符串或 `null` | occurrence 的原始顶层 SMILES；来源明确为空时可为 `null` | 构造示例如 `"unsupported_smiles"` |

下面是说明嵌套关系的构造指标片段；省略了其它五个 `smiles_match` 分组和完整不受支持候选列表。

```json
{
  "candidate_count": 1029,
  "foreground": {"false_negative": 18, "false_positive": 27, "f1": 0.7887323943661971, "true_negative": 900, "true_positive": 84},
  "foreground_prauc": 0.71,
  "foreground_threshold": 0.43,
  "pdb_count": 179,
  "smiles_match": {"overall": {"accuracy": 0.8356164383561644, "correct_count": 61, "eligible_count": 73}},
  "unsupported_identity_target_candidate_count": 1,
  "unsupported_identity_targets": [{"centered_box_index": 2, "matched_occurrence_id": 0, "pdb_id": "9ian", "source_blob_index": 7, "target_smiles": "unsupported_smiles"}]
}
```

### `predictions.<condition>.<pdb_id>[]`

`condition` 取 `real_receptor` 或 `cryoatom2_receptor`，`pdb_id` 是小写 PDB 标识。每个列表元素对应一个 centered 候选；同一 PDB 中没有 centered 候选时可以不建立该 `pdb_id` 键，但该 PDB 仍计入 `metrics.pdb_count` 和 top-k 固定分母。

| 字段 | 数据类型 | 含义与对齐关系 | 示例值或空值规则 |
| --- | --- | --- | --- |
| `centered_box_index` | 非负整数 | 候选在当前 PDB `F2_centered.npz` 候选轴上的稳定编号 | `4` |
| `source_blob_index` | 非负整数 | centered 候选对应的完整 F2 blob 编号；也是同分排序时的升序次级键 | `17` |
| `gaussian_score` | 有限浮点数 | Stage1 Gaussian `candidate_score`；只决定 Gaussian 曲线中 selected 段或追加段的内部顺序 | `0.82` |
| `source_probability_mean` | 有限浮点数 | 完整来源 blob 内 Stage1 概率的算术平均值；只决定概率均值曲线的段内顺序 | `0.74` |
| `candidate_selected` | 布尔值 | `true` 进入扩展顺序前段并进入 `metrics`；`false` 进入追加段且不进入 `metrics` | `true` |
| `foreground_target` | 布尔值 | 是否存在候选覆盖率和 occurrence 覆盖率都不低于 0.30 的唯一目标 | `true` |
| `target_category` | 枚举字符串或 `null` | 真前景的原始类别，取 `ion`、`nucleotide_like`、`peptide_like`、`small_molecule` 或 `sugar`；背景为 `null` | `"small_molecule"` |
| `matched_occurrence_id` | 非负整数或 `null` | 最大化两项覆盖率几何平均并按较小编号决胜后的 occurrence 编号；背景为 `null` | `3` |
| `target_smiles` | 字符串或 `null` | 唯一目标 occurrence 的精确顶层 `PreparedLigand.smiles`；背景为 `null`，不受支持真前景仍保留原字符串 | `"CCO"` |
| `identity_supported` | 布尔值 | `target_smiles` 是否同时具有合法 SMILES 图和 SMI-TED 表征；只有受支持真前景进入 SMILES 准确率分母 | `true` |
| `predicted_smiles` | 字符串或 `null` | 当前 PDB 闭集身份 logits 最大槽对应的精确 SMILES；整个批次身份轴为空时为 `null` | `"CCO"` |
| `foreground_probability` | 0 至 1 的浮点数 | Matcher 前景 logit 经过 sigmoid 后的概率；只用于 PRAUC 和固定阈值 F1，不改变 top-k 候选顺序 | `0.91` |

下面两条是字段齐全的构造示例。第一条是受支持真前景；第二条是背景候选。背景仍可以得到一个闭集 `predicted_smiles`，但该身份预测不进入 SMILES 准确率，也不能形成 Stage1+Matcher 命中。

```json
{"candidate_selected":true,"centered_box_index":4,"foreground_probability":0.91,"foreground_target":true,"gaussian_score":0.82,"identity_supported":true,"matched_occurrence_id":3,"predicted_smiles":"CCO","source_blob_index":17,"source_probability_mean":0.74,"target_category":"small_molecule","target_smiles":"CCO"}
{"candidate_selected":false,"centered_box_index":9,"foreground_probability":0.12,"foreground_target":false,"gaussian_score":0.18,"identity_supported":false,"matched_occurrence_id":null,"predicted_smiles":"CCO","source_blob_index":22,"source_probability_mean":0.12,"target_category":null,"target_smiles":null}
```

不受支持身份的真前景候选保留原始排名和目标字段，但 `identity_supported=false`，不能形成 Stage1+Matcher 命中。

### `curves.<sorting>`

`sorting` 取 `gaussian` 或 `source_probability_mean`。每种排序包含下列五条曲线：

| 曲线键 | 候选来源与命中定义 |
| --- | --- |
| `real_stage1_only` | 真实受体 Find_1；前 `k` 个候选中存在双向覆盖命中即成功 |
| `real_stage1_matcher` | 与 `real_stage1_only` 使用相同顺序；命中还要求真前景、身份受支持且 `predicted_smiles == target_smiles` |
| `cryo_stage1_only` | CryoAtom2 Find_1；Stage1-only 命中定义 |
| `cryo_stage1_matcher` | 与 `cryo_stage1_only` 使用相同顺序；增加 Matcher 精确 SMILES 条件 |
| `emap2lig` | 唯一的 receptor-independent Emap2lig Stage1-only 基线；两种排序下内容相同 |

每条记录使用前述 Stage1-only 曲线字段：`name`、`pdb_count`、`pdb_ids`、`top_k`、`hit_count`、`hit_ratio` 和 `pdb_hit`。Stage1+Matcher 不使用 `foreground_probability` 或 `foreground_threshold`；失败候选保留原排名，不能删除后让后一名补位。

### 可选 `test_1`

提供官方 149-PDB 列表时，`test_1` 顶层字段出现。它只包含 `metrics.real_receptor` 和 `metrics.cryoatom2_receptor`，字段与 test_0 指标完全相同，但 `pdb_count=149`。这些指标复用同一次 test_0 冻结前向并按 PDB 标识筛选，不保存第二份 `predictions`，也不启动第二次模型推理。

## 训练代码快照

`checkpoint.py::load_matcher_from_training_release()` 从 checkpoint 所属运行目录的 `run.json` 读取 `source_project_root`，并从该代码快照导入模型配置、模型类和 checkpoint 读取器。函数返回模型、checkpoint 和配置；代码快照路径不复制进评估产物。

## 正式图片

`plot_stage1_only_preview()` 绘制两张 Stage1-only 预览图。`plot_topk_final()` 绘制 Gaussian 与来源概率均值两张正式 top-k 图；`plot_metric_bars()` 绘制前景与 SMILES 指标图。三个入口都只读取冻结 JSON。
