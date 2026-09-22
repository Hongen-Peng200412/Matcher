# evaluation：冻结指标、test 连接与纯绘图

本目录把训练期 validation 与冻结 test 分开。test 只读取 BEST 中保存的 validation 前景阈值，不在真实受体或 CryoAtom2 测试数据上重新拟合。

## 服务器正式产物位置

服务器通用评测产物根为 `/storage/penghongen/Matcher/evaluation/`。以下目录树已与服务器现存文件做只读核对；`final_smiles/` 下列出的 7 个实验目录具有相同文件契约。

```text
## <科学产物>
/storage/penghongen/Matcher/evaluation/
├── stage1_only_preview/
│   └── stage1_only_preview.json                 # 179-PDB Stage1-only 预览元数据，不含 Matcher 预测
└── final_smiles/
    └── {strongest,density_blob_only,blob_only-0,strongest-1,density_only-0,density_only-106,pocket_only-0}/
        └── frozen_test.json                     # 当前实验的 test_0 前向、test_1 子集指标、逐候选预测和两种 top-k 排序

## <其他文件>
/storage/penghongen/Matcher/evaluation/
├── stage1_only_preview/
│   ├── topk_gaussian_stage1_only_preview.png    # Gaussian 排序的 Stage1-only 预览图（可再生辅助导出）
│   └── topk_source_probability_mean_stage1_only_preview.png # 来源概率均值排序的 Stage1-only 预览图（可再生辅助导出）
└── final_smiles/
    └── {strongest,density_blob_only,blob_only-0,strongest-1,density_only-0,density_only-106,pocket_only-0}/
        ├── foreground_and_smiles_metrics.png    # 前景与 SMILES 指标图（可再生辅助导出）
        ├── topk_gaussian_final.png              # Gaussian 排序的正式 top-k 图（可再生辅助导出）
        └── topk_source_probability_mean_final.png # 来源概率均值排序的正式 top-k 图（可再生辅助导出）
```

目录名 `strongest` 对应稳定实验标识 `strongest-0`，`density_blob_only` 对应稳定实验标识 `dense_blob_only-0`；其余 5 个目录名与稳定实验标识相同。图片只从同目录或已冻结的 Stage1-only JSON 生成，不能替代 JSON 科学产物。

同一服务器父目录中的 `/storage/penghongen/Matcher/evaluation/test_smallmolecule_v1/` 属于独立的 Stage3 小分子评测，不由本目录的 `build_final_test_artifact()` 写出；其 `evaluation.json`、`docking_handoff_candidates.jsonl` 和图片契约见 [`../../ops/small_molecule_test/README.md`](../../ops/small_molecule_test/README.md)。

## 推荐阅读顺序

1. `metrics.py`：先理解前景 PRAUC、固定阈值 F1、SMILES 匹配准确率和 selection。
2. `test_set.py`：理解 Stage1-only 预览、BEST 冻结前向、Stage1+Matcher 曲线和 test_1 筛选。
3. `plotting.py`：理解 `plot_stage1_only_preview()`、`plot_topk_final()` 和 `plot_metric_bars()` 三个 JSON 驱动入口；绘图不加载模型，也不重新计算科学事实。
4. `__init__.py`：查看目录的公开入口。

## 指标定义

- 前景 PRAUC：全部候选的二分类 average precision，同分预测作为一组共同处理。
- 前景 F1：只使用 BEST 在 validation 上保存的阈值。
- SMILES 匹配准确率：只统计 `foreground_target=true` 且 `identity_supported=true` 的候选。
- 五类分层：`ion`、`nucleotide_like`、`peptide_like`、`small_molecule`、`sugar`；零支持时 `accuracy=null`，绘图显示 `N/A (n=0)`。
- validation 选模分数：`(0.3 × 前景 PRAUC + SMILES 匹配准确率) / 1.3`。前景 F1 与阈值不参与选模。

`9ian` 与 `9oq7` 的部分真前景缺少严格图和语言资产。这些候选保留前景真值与 Stage1 原排名，不进入 SMILES 准确率分母，也不能产生 Stage1+Matcher 命中。

## Stage1-only 预览产物

`build_stage1_only_preview_metadata()` 把 `schema="matcher_stage1_only_preview_v1"` 的对象原子写入调用方指定的 `output_path`。当前正式 `output_path` 是 `/storage/penghongen/Matcher/evaluation/stage1_only_preview/stage1_only_preview.json`；`matcher_stage1_only_preview_v1` 是 JSON 顶层 `schema` 字段的值，不是文件名。该 JSON 保存 179-PDB Stage1-only 曲线元数据，绘图脚本用它生成同目录的两张 `stage1_only_preview` 图片；它不含 Matcher 概率、预测 SMILES 或 Stage1+Matcher 数值，不能称为最终图的数据源。

| 顶层字段 | 类型或结构 | 含义与示例 |
| --- | --- | --- |
| `schema` | `str` | 固定为 `"matcher_stage1_only_preview_v1"`。 |
| `status` | `str` | 固定为 `"stage1_only_preview"`。 |
| `coverage_threshold` | `float` | 双向覆盖门槛，当前为 `0.3`。 |
| `denominator_pdb_count` | `int` | 每个 top-k 点固定分母，当前为 `179`。 |
| `candidate_order` | `str` | 先排列全部 `candidate_selected=true` 候选，再追加全部 `false` 候选；两段使用同一指定分数。 |
| `stage1_only_hit_definition` | `str` | top-k 内任一候选对任一真实 occurrence 的两项覆盖率均不低于 0.30 即命中。 |
| `candidate_set_note` | `str` | 声明 Find_1 与 Emap2lig 保留各自候选集合，不能逐候选配对。 |
| `curves.gaussian` | 曲线映射 | 含 `real_receptor`、`cryoatom2_receptor`、`emap2lig`。 |
| `curves.source_probability_mean` | 曲线映射 | 含相同三个键；两种排序中的 `emap2lig` 曲线内容必须完全相同。 |

每条"曲线映射"记录的字段如下：

| 字段 | 类型与形状 | 含义与对齐 | 示例值 |
| --- | --- | --- | --- |
| `name` | `str` | 图例文字；Emap2lig 明确标记 density-only、receptor-independent。 | `"Find_1 real-receptor Find_1 Stage1-only"` |
| `source` | `str` | 正式只读来源根或文件路径。 | `"/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1/held_out_test_0"` |
| `pdb_count` | `int` | 固定为 179。 | `179` |
| `pdb_ids` | `list[str]`，长度 179 | 曲线固定 PDB 轴，按小写标识排序。 | `["9abc", "9def"]` 是缩短格式示例 |
| `top_k` | `list[int]`，长度 20 | 固定为 1 到 20。 | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]` |
| `hit_count` | `list[int]`，长度 20 | 第 `j` 项是扩展 top-`top_k[j]` 命中的 PDB 数。 | `[80, 92, 101, ...]` 是缩短格式示例 |
| `hit_ratio` | `list[float]`，长度 20 | `hit_count[j] / 179`，与 `top_k` 逐项对齐。 | `87 / 179` |
| `pdb_hit` | `dict[str,list[bool]]` | 每个 PDB 对应 20 个累积命中值；一旦命中，后续更大 k 保持 `true`。 | `{"9abc": [false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]}` |

三条 Stage1-only 来源曲线分别追溯到以下只读目录：

- `real_receptor`：`/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1/held_out_test_0`；
- `cryoatom2_receptor`：`/storage/penghongen/AdaLigand_stage1_inference/Find_1/CryoAtom2受体/artifacts/Find_1/held_out_test_0`；
- `emap2lig`：`/storage/penghongen/AdaLigand_stage1_inference/Emap2lig/official_find_li/held_out_test_0/evaluation`。

Find_1 对 `candidate_selected=True` 和 `False` 分别按 Gaussian `candidate_score` 或 `source_probability_mean` 降序，再把 `False` 段追加到 `True` 段之后。分数并列时使用较小的 `source_blob_index`；两项键都相同时保留来源顺序。因此，unselected 候选不能挤占原 selected 前缀。Emap2lig 对自身候选使用相同的分段规则和实例内原生 ligand probability 均值；每张图只有一条 Emap2lig 曲线，不复制为两种受体条件。

## 正式冻结 test 产物

`build_final_test_artifact()` 用同一个 BEST 对真实受体与 CryoAtom2 test_0 各前向一次，并把 `schema="matcher_frozen_test_v1"` 的对象原子写入调用方指定的 `output_path`。当前正式路径为 `/storage/penghongen/Matcher/evaluation/final_smiles/<experiment_dir>/frozen_test.json`，其中 `<experiment_dir>` 是本页“服务器正式产物位置”列出的 7 个实验目录之一；`matcher_frozen_test_v1` 是 JSON 顶层 `schema` 字段的值，不是文件名。同一次候选预测同时服务两种 Find_1 排序，不为第二种排序重复前向。模型配置、模型类和 checkpoint 加载规则固定从训练 run 的 `run.json` 所指 release 读取；当前评估 release 只提供冻结清单读取、扩展 top-k 汇总和绘图逻辑。

| 顶层字段 | 类型或结构 | 含义 |
| --- | --- | --- |
| `schema` | `str` | 固定为 `"matcher_frozen_test_v1"`。 |
| `status` | `str` | 固定为 `"final_with_matcher"`。 |
| `model_code_source` | `dict` | 训练 run、训练项目根和 release manifest 路径；两种受体条件必须逐字段相同。 |
| `checkpoint_path` | `str` | 被评估的真实 BEST 路径。 |
| `real_manifest_path` | `str` | 真实受体 test_0 manifest 路径。 |
| `cryoatom2_manifest_path` | `str` | CryoAtom2 test_0 manifest 路径。 |
| `stage1_preview_path` | `str` | 已验收 Stage1-only 预览 JSON 路径；当前正式值为 `/storage/penghongen/Matcher/evaluation/stage1_only_preview/stage1_only_preview.json`。 |
| `candidate_order` | `str` | Stage1-only 与 Stage1+Matcher 共用的 selected 优先扩展顺序。 |
| `foreground_threshold_source` | `str` | 固定声明阈值来自 BEST validation 拟合。 |
| `metrics` | 条件映射 | `real_receptor` 和 `cryoatom2_receptor` 各一份指标。 |
| `predictions` | 条件映射 | 两种受体条件下按 PDB 分组的候选预测。 |
| `curves` | 排序映射 | `gaussian` 和 `source_probability_mean` 下各五条冻结曲线。 |
| `test_1` | 可选映射 | 提供官方 149-PDB 列表时出现；只筛选 test_0 预测，不重新前向。 |

### `metrics.<condition>`

| 字段 | 类型或结构 | 含义与特殊值 |
| --- | --- | --- |
| `candidate_count` | `int` | 当前 PDB 集合内 `candidate_selected=true` 的候选数；保持原冻结指标口径。 |
| `foreground_prauc` | `float|null` | 原 selected 候选口径的前景 PRAUC；没有正类时严格 JSON 写为 `null`。 |
| `foreground_threshold` | `float` | BEST validation 保存的固定阈值。 |
| `foreground` | 映射 | 含 `true_positive`、`false_positive`、`false_negative`、`true_negative` 和 `f1`。 |
| `smiles_match.overall` | 映射 | 含 `correct_count`、`eligible_count`、`accuracy`；零支持时准确率为 `null`。 |
| `smiles_match.<category>` | 五个类别键 | `ion`、`nucleotide_like`、`peptide_like`、`small_molecule`、`sugar` 各含 `correct_count`、`eligible_count`、`accuracy`。 |
| `pdb_count` | `int` | test_0 为 179，test_1 为 149；包括没有候选的 PDB。 |
| `unsupported_identity_target_candidate_count` | `int` | 真前景但没有受支持身份的候选数。 |
| `unsupported_identity_targets` | `list[object]` | 每项含 `pdb_id`、`centered_box_index`、`source_blob_index`、`matched_occurrence_id` 和 `target_smiles`。 |

### `predictions.<condition>.<pdb_id>[]`

| 字段 | 类型 | 含义与特殊值 |
| --- | --- | --- |
| `centered_box_index` | `int >= 0` | 当前 PDB `F2_centered.npz` 中的稳定候选编号。 |
| `source_blob_index` | `int >= 0` | 源 F2 blob 编号，也是排序同分时的升序次级键。 |
| `gaussian_score` | 有限 `float` | Find_1 Gaussian 排序分数。 |
| `source_probability_mean` | 有限 `float` | 源 blob 内 Stage1 概率均值。 |
| `candidate_selected` | `bool` | `true` 进入扩展顺序前段，`false` 进入追加段。 |
| `foreground_target` | `bool` | 候选是否通过双向 0.30 覆盖门槛。 |
| `target_category` | `str|null` | 真前景的五类分层标签；背景为 `null`。 |
| `matched_occurrence_id` | `int >= 0|null` | 最大覆盖 occurrence 编号；背景为 `null`。 |
| `target_smiles` | `str|null` | 唯一正确的精确顶层 SMILES；背景为 `null`，不受支持真前景可以保留原字符串。 |
| `identity_supported` | `bool` | 目标 SMILES 是否同时具有合法图和 SMI-TED 资产。 |
| `predicted_smiles` | `str|null` | PDB 内闭集 argmax 身份；身份轴为空时为 `null`。 |
| `foreground_probability` | 有限 `float` | `foreground_logit` 经过 sigmoid 后的概率。 |

不受支持候选保留原始排名与上述目标字段，但 `identity_supported=false`，不能成为 Stage1+Matcher top-k 命中。

### `curves.<sorting>`

每种排序含 `real_stage1_only`、`real_stage1_matcher`、`cryo_stage1_only`、`cryo_stage1_matcher` 和 `emap2lig`。Stage1-only 与 Stage1+Matcher 使用逐候选相同的 selected 优先扩展顺序。每条记录沿用预览曲线的 `name`、`source`、`pdb_count`、`pdb_ids`、`top_k`、`hit_count`、`hit_ratio`、`pdb_hit` 字段。Stage1+Matcher 命中不使用前景阈值：候选必须是真前景、身份受支持且 `predicted_smiles` 与唯一 `target_smiles` 逐字相同；失败候选留在原排名，不能删除后让后一名补位。

### 可选 `test_1`

`test_1` 是同一份 `frozen_test.json` 内的可选顶层字段，不是单独的 JSON 文件。`test_1.source` 是官方 149-PDB 文本路径，`definition` 固定说明复用 test_0 前向，`metrics` 含两种受体条件的 149-PDB 指标。它不保存第二份预测，也不创建新的模型前向。

## 绘图边界

`ops/matcher/plot_stage1_preview.py` 读取 `/storage/penghongen/Matcher/evaluation/stage1_only_preview/stage1_only_preview.json`，以 `gaussian` 和 `source_probability_mean` 分别调用一次 `plot_stage1_only_preview()`，写出 `topk_gaussian_stage1_only_preview.png` 和 `topk_source_probability_mean_stage1_only_preview.png`。该函数只接受 `status=stage1_only_preview`，输出文件名必须含 `stage1_only_preview`。

`ops/matcher/plot_frozen_test.py` 读取某个实验目录的 `frozen_test.json`，以两种排序分别调用 `plot_topk_final()`，写出 `topk_gaussian_final.png` 和 `topk_source_probability_mean_final.png`；同一脚本调用 `plot_metric_bars()` 写出 `foreground_and_smiles_metrics.png`。这两个正式绘图函数只接受 `status=final_with_matcher`。三类绘图函数都只读取冻结 JSON，不加载模型、checkpoint 或来源 NPZ。
