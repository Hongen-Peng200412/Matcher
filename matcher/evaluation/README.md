# evaluation：冻结指标、test 连接与纯绘图

本目录把训练期 validation 与冻结 test 分开。test 只读取 BEST 中保存的 validation 前景阈值，不在真实受体或 CryoAtom2 测试数据上重新拟合。

## 推荐阅读顺序

1. `metrics.py`：先理解前景 PRAUC、固定阈值 F1、SMILES 匹配准确率和 selection。
2. `test_set.py`：理解 Stage1-only 预览、BEST 冻结前向、Stage1+Matcher 曲线和 test_1 筛选。
3. `plotting.py`：理解三个 JSON 驱动图入口；绘图不加载模型，也不重新计算科学事实。
4. `__init__.py`：查看目录的公开入口。

## 指标定义

- 前景 PRAUC：全部候选的二分类 average precision，同分预测作为一组共同处理。
- 前景 F1：只使用 BEST 在 validation 上保存的阈值。
- SMILES 匹配准确率：只统计 `foreground_target=true` 且 `identity_supported=true` 的候选。
- 五类分层：`ion`、`nucleotide_like`、`peptide_like`、`small_molecule`、`sugar`；零支持时 `accuracy=null`，绘图显示 `N/A (n=0)`。

`9ian` 与 `9oq7` 的部分真前景缺少严格图和语言资产。这些候选保留前景真值与 Stage1 原排名，不进入 SMILES 准确率分母，也不能产生 Stage1+Matcher 命中。

## Stage1-only 预览产物

`build_stage1_only_preview_metadata()` 写出一个 `matcher_stage1_only_preview_v1` JSON。它不含 Matcher 概率、预测 SMILES 或 Stage1+Matcher 数值，不能称为最终图的数据源。

| 顶层字段 | 类型或结构 | 含义与示例 |
| --- | --- | --- |
| `schema` | `str` | 固定为 `"matcher_stage1_only_preview_v1"`。 |
| `status` | `str` | 固定为 `"stage1_only_preview"`。 |
| `coverage_threshold` | `float` | 双向覆盖门槛，当前为 `0.3`。 |
| `denominator_pdb_count` | `int` | 每个 top-k 点固定分母，当前为 `179`。 |
| `stage1_only_hit_definition` | `str` | top-k 内任一候选对任一真实 occurrence 的两项覆盖率均不低于 0.30 即命中。 |
| `candidate_set_note` | `str` | 声明 Find_1 与 Emap2lig 保留各自候选集合，不能逐候选配对。 |
| `curves.gaussian` | 曲线映射 | 含 `real_receptor`、`cryoatom2_receptor`、`emap2lig`。 |
| `curves.source_probability_mean` | 曲线映射 | 含相同三个键；两种排序中的 `emap2lig` 曲线内容必须完全相同。 |

每条曲线记录的字段如下：

| 字段 | 类型与形状 | 含义与对齐 | 示例值 |
| --- | --- | --- | --- |
| `name` | `str` | 图例文字；Emap2lig 明确标记 density-only、receptor-independent。 | `"Find_1 real-receptor Find_1 Stage1-only"` |
| `source` | `str` | 正式只读来源根或文件路径。 | `"/storage/.../held_out_test_0"` |
| `pdb_count` | `int` | 固定为 179。 | `179` |
| `pdb_ids` | `list[str]`，长度 179 | 曲线固定 PDB 轴，按小写标识排序。 | `["9abc", "9def"]` 是缩短格式示例 |
| `top_k` | `list[int]`，长度 20 | 固定为 1 到 20。 | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]` |
| `hit_count` | `list[int]`，长度 20 | 第 `j` 项是 top-`top_k[j]` 命中的 PDB 数。 | Emap2lig 的第 3、4、5 项为 `87`、`89`、`90` |
| `hit_ratio` | `list[float]`，长度 20 | `hit_count[j] / 179`，与 `top_k` 逐项对齐。 | `87 / 179` |
| `pdb_hit` | `dict[str,list[bool]]` | 每个 PDB 对应 20 个累积命中值；一旦命中，后续更大 k 保持 `true`。 | `{"9abc": [false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]}` |

Find_1 只在冻结的 `candidate_selected=True` 集合内分别按 Gaussian `candidate_score` 和 `source_probability_mean` 降序。Emap2lig 按实例内原生 ligand probability 均值排序；每张图只有一条 Emap2lig 曲线，不复制为两种受体条件。

## 正式冻结 test 产物

`build_final_test_artifact()` 用同一个 BEST 对真实受体与 CryoAtom2 test_0 各前向一次，写出 `matcher_frozen_test_v1` JSON。同一次候选预测同时服务两种 Find_1 排序，不为第二种排序重复前向。

| 顶层字段 | 类型或结构 | 含义 |
| --- | --- | --- |
| `schema` | `str` | 固定为 `"matcher_frozen_test_v1"`。 |
| `status` | `str` | 固定为 `"final_with_matcher"`。 |
| `checkpoint_path` | `str` | 被评估的真实 BEST 路径。 |
| `real_manifest_path` | `str` | 真实受体 test_0 manifest 路径。 |
| `cryoatom2_manifest_path` | `str` | CryoAtom2 test_0 manifest 路径。 |
| `stage1_preview_path` | `str` | 已验收 Stage1-only 预览 JSON 路径。 |
| `foreground_threshold_source` | `str` | 固定声明阈值来自 BEST validation 拟合。 |
| `metrics` | 条件映射 | `real_receptor` 和 `cryoatom2_receptor` 各一份指标。 |
| `predictions` | 条件映射 | 两种受体条件下按 PDB 分组的候选预测。 |
| `curves` | 排序映射 | `gaussian` 和 `source_probability_mean` 下各五条冻结曲线。 |
| `test_1` | 可选映射 | 提供官方 149-PDB 列表时出现；只筛选 test_0 预测，不重新前向。 |

### `metrics.<condition>`

| 字段 | 类型或结构 | 含义与特殊值 |
| --- | --- | --- |
| `candidate_count` | `int` | 当前 PDB 集合内的候选数。 |
| `foreground_prauc` | `float|null` | 全候选前景 PRAUC；没有正类时严格 JSON 写为 `null`。 |
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
| `foreground_target` | `bool` | 候选是否通过双向 0.30 覆盖门槛。 |
| `target_category` | `str|null` | 真前景的五类分层标签；背景为 `null`。 |
| `matched_occurrence_id` | `int >= 0|null` | 最大覆盖 occurrence 编号；背景为 `null`。 |
| `target_smiles` | `str|null` | 唯一正确的精确顶层 SMILES；背景为 `null`，不受支持真前景可以保留原字符串。 |
| `identity_supported` | `bool` | 目标 SMILES 是否同时具有合法图和 SMI-TED 资产。 |
| `predicted_smiles` | `str|null` | PDB 内闭集 argmax 身份；身份轴为空时为 `null`。 |
| `foreground_probability` | 有限 `float` | `foreground_logit` 经过 sigmoid 后的概率。 |

不受支持候选保留原始排名与上述目标字段，但 `identity_supported=false`，不能成为 Stage1+Matcher top-k 命中。

### `curves.<sorting>`

每种排序含 `real_stage1_only`、`real_stage1_matcher`、`cryo_stage1_only`、`cryo_stage1_matcher` 和 `emap2lig`。每条记录沿用预览曲线的 `name`、`source`、`pdb_count`、`pdb_ids`、`top_k`、`hit_count`、`hit_ratio`、`pdb_hit` 字段。Stage1+Matcher 命中不使用前景阈值：候选必须是真前景、身份受支持且 `predicted_smiles` 与唯一 `target_smiles` 逐字相同；失败候选留在原排名，不能删除后让后一名补位。

### 可选 `test_1`

`test_1.source` 是官方 149-PDB 文本路径，`definition` 固定说明复用 test_0 前向，`metrics` 含两种受体条件的 149-PDB 指标。它不保存第二份预测，也不创建新的模型前向。

## 绘图边界

`plot_stage1_only_preview()` 只接受 `status=stage1_only_preview`，输出文件名必须含 `stage1_only_preview`。正式 top-k 图和七栏指标图只接受 `status=final_with_matcher`。三类绘图函数都只读取冻结 JSON，不加载模型、checkpoint 或来源 NPZ。
