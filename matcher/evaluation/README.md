# `matcher.evaluation` 指标与一次性 calibration 契约

## 中心职责

`matcher.evaluation` 只负责两类边界：

- validation 是训练期间反复运行的验证划分。`EvaluationAccumulator` 把完整划分的六分类、身份和区域计数合并为 `EvaluationReport`，`selection_score` 再从 validation 报告生成 BEST checkpoint 的选择分数。
- calibration 是 validation 已经选定方案后只评估一次的保留划分。`run_calibration` 不创建 calibration 数据，不训练模型，不修改 checkpoint，也不产生新的选模分数；它记录所选 BEST checkpoint、当前读取的数据证据和一次性 claim 后写出严格 JSON，输入文件不会被快照或锁定。

这里的 occurrence 是一个 PDB 中的一次具体配体实例。联合身份是同一 PDB 内由 `object_key` 与顶层 `PreparedLigand.smiles` 共同确定的身份槽；评估器只消费 Dataset 已经建立的身份轴，不重新读取 occurrence 或重建联合身份。

## 文件边界

| 文件 | 入口 | 负责内容 | 不负责内容 |
| --- | --- | --- | --- |
| `__init__.py` | 包导入 | 定义包级职责，不重导出子模块实体 | 不触发 calibration，不读取文件 |
| `metrics.py` | `EvaluationAccumulator`、`selection_score` | 跨批次整数计数、最终比率、validation 选模公式 | 不读取 manifest，不写 JSON，不判断输入来自哪个划分 |
| `calibration.py` | `run_calibration`、`main` | 一次性 claim、checkpoint 与数据留证、CUDA 完整评估、严格 JSON | 不物化 calibration，不访问 W&B，不训练或优化模型，不写回 checkpoint；新构造模型仍会加载状态并迁移到 CUDA |

训练引擎在 [`matcher/training/engine.py`](../training/engine.py) 中调用指标入口；候选与身份批次字段由 [`matcher/datasets/README.md`](../datasets/README.md) 定义。

## 形状与类别符号

| 符号 | 含义 |
| --- | --- |
| `B` | 一个物理批次中的候选数 |
| `I` | 当前物理批次补齐后的最大身份槽数，可为 0 |
| `Z,Y,X` | 区域目标和区域 logit 的网格轴 |
| `K` | 配置的最大身份排名名次，正式评估固定为 5 |

`CLASS_NAMES` 固定含 6 类，第 0 类是 `background`，其余 5 类是前景。`FOREGROUND_CLASS_INDICES` 与 `FOREGROUND_CLASS_NAMES` 使用同一顺序；`SMALL_MOLECULE_INDEX` 是其中 `small_molecule` 的类别下标。

## `EvaluationBatch` 输入

`EvaluationAccumulator.update` 读取一个 `CandidateBatch` 子集。区域和身份轴必须与同批 `class_logits` 对齐。

| 字段 | dtype 与形状 | 语义与门禁 |
| --- | --- | --- |
| `class_target` | Torch `int64 (B,)` | 每项位于 `[0,6)` |
| `ligand_area_target` | Torch `bool (B,Z,Y,X)` | `True` 是候选配体区域真值 |
| `binding_area_target` | Torch `bool (B,Z,Y,X)` | `True` 是受体结合区域真值 |
| `identity_target` | Torch `int64 (B,)` | `-1` 表示当前候选没有完整可用的联合身份监督；否则指向 `[0,I)` 的有效槽 |
| `identity_mask` | Torch `bool (B,I)` | `True` 是该 PDB 的真实身份槽，`False` 是批次补零槽 |
| `identity_class_index` | Torch `int64 (B,I)` | 有效槽为前景类别下标 1–5，补零槽为 `-1`；真实目标槽的类别必须等于 `class_target` |
| `metadata` | 长度 `B` 的列表 | 身份路径只读取 `identity_seen_in_train`，其它字段不参与数值评估 |

每个 `metadata[*]` 保留以下字段：

| 字段 | 类型 | 本模块用途 |
| --- | --- | --- |
| `split` | 字符串 | 不读取；调用方负责划分来源 |
| `pdb_id` | 字符串 | 不读取；身份槽已按 PDB 建立 |
| `candidate_index` | 整数 | 不读取 |
| `matched_object_key` | 字符串或 `null` | 不读取，不据此重新判定身份资格 |
| `matched_prepared_smiles` | 字符串或 `null` | 不读取 |
| `matched_prepared_smiles_status` | `available`、`explicit_null`、`missing_file` 或 `not_applicable` | 不读取 |
| `identity_seen_in_train` | 布尔值或 `null` | 进入已配置条件身份分母时必须是严格布尔值；其它候选允许 `null` |

模型输出必须提供有限的 `class_logits (B,6)`、`ligand_area_logits (B,Z,Y,X)` 和 `binding_area_logits (B,Z,Y,X)`。完整 Matcher 另提供有限 `identity_logits (B,I)`；候选预训练没有该键时，身份字段在本批次不会被验证或累计。评估器显式检查关键形状、dtype、目标范围和有限性，但不重新核验完整 CandidateBatch schema。

`update` 不是事务：混淆矩阵先增加，区域和身份门禁随后执行。如果后段输入错误，前面已经增加的计数不会回滚，因此正式调用必须来自通过 Dataset 与 collate 门禁的批次。

## 六分类指标

每个批次先按 `class_logits.argmax(-1)` 得到预测下标，再增加 `int64 (6,6)` 混淆矩阵。第一轴是真实类别，第二轴是预测类别。

| 字段 | 公式 | 零分母 |
| --- | --- | --- |
| `six_class.accuracy` | 对角线总和 / 全候选数 | `NaN` |
| `per_class[*].precision` | 当前类真阳性 / 当前类预测数 | `NaN` |
| `per_class[*].recall` | 当前类真阳性 / 当前类真实支持数 | `NaN` |
| `per_class[*].f1` | 有限 precision 与 recall 的调和平均 | `0` |
| `per_class[*].support` | 当前类真实候选数 | 整数 0 |
| `per_class[*].predicted_count` | 当前类预测候选数 | 整数 0 |
| `macro_f1_supported_classes` | 只平均 `support>0` 类别的 F1 | 没有支持类别时 `NaN` |
| `macro_f1_all_classes` | 固定平均 6 类，零支持类按 F1=0 | 始终有 6 项 |
| `macro_f1` | `macro_f1_supported_classes` 的兼容别名 | 同上 |

固定六类平均和支持类平均不是同一数值。训练的历史选模口径使用 `macro_f1`，calibration artifact 同时保留两者。

## 区域 Dice

两个区域头都以 `logit >= 0` 预测 `True`；恰为 0 的 logit 也属于预测区域。每个物理批次分别累计交集体素数、预测 `True` 体素数和真实 `True` 体素数，不先计算逐候选 Dice。

| 字段 | 公式 |
| --- | --- |
| `area.ligand_micro_dice` | `2*ligand_intersection/(ligand_predicted+ligand_actual)` |
| `area.binding_micro_dice` | `2*binding_intersection/(binding_predicted+binding_actual)` |
| `area.ligand_dice_denominator` | `ligand_predicted+ligand_actual` |
| `area.binding_dice_denominator` | `binding_predicted+binding_actual` |

预测和真值都为空时分母为 0，内存报告保留 `NaN`，不把空真负候选记为完美 Dice。

## 身份排名与端到端门

### 槽内排名

无效身份槽先被替换为当前浮点 dtype 的最小有限值。`Top-1` 返回原身份轴下标；实际 `Top-K` 查询数是 `min(K,I)`。`I=0` 或某个候选没有任何允许槽时，Top-1 哨兵为 `-1`，Top-K 命中为 `False`。

`identity_target=-1` 的候选没有完整可用的联合身份监督，不进入固有或条件身份分母。真实类别位于当前统计范围且 `identity_target=-1` 时，该候选不会被当成该范围的端到端假阳性；窄范围中的其它前景类别属于范围外负例，仍可成为该范围假阳性。真实背景候选若所在 PDB 有身份槽且被错误预测为前景，也会进入预测阳性并降低 precision。

### 三种统计口径

| 口径 | 分母或预测门 |
| --- | --- |
| 固有 `intrinsic` | 真实类别在当前范围且 `identity_target>=0`；忽略六分类预测 |
| 条件 `conditional` | 固有分母且六分类预测精确等于真实类别 |
| 端到端 `end_to_end` | 真实阳性沿用固有分母；预测阳性另由当前范围前景门和可用身份槽决定 |

`conditional` 在五类总体口径仍要求前景子类精确判对。`all_non_background` 的端到端口径只把六分类当作背景/前景门：预测为任一前景类即可，最终类别由 Top-1 身份槽所属类别决定。

### 并行身份范围

| 报告字段 | 排名槽 | 端到端六分类门 |
| --- | --- | --- |
| `small_molecule_identity` | 只含 `small_molecule` 槽 | 必须预测为 `small_molecule` |
| `all_non_background_identity` | 五类全部有效身份槽 | 必须预测为任一前景类 |
| `identity_per_class[class_name]` | 五类共享排名后按真实类别拆固有/条件分母 | 六分类只需通过任一前景门；预测阳性按最终身份槽类别归档 |

`identity_scope=None` 与 `small_molecule` 都选择小分子口径填充三个兼容顶层字段；`all_non_background` 选择五类总体口径。两个详细范围报告和五个逐类报告无论配置为何都会返回。

### seen 与 unseen

`conditional_identity` 的 `seen` 与 `unseen` 只拆分已配置身份范围的条件分母：

- `identity_seen_in_train=True` 表示同一 `(object_key, PreparedLigand.smiles)` 联合键曾作为训练正目标出现；
- `False` 表示未作为训练正目标出现；
- 同一联合键的 seen 状态可跨 PDB 共享，但各 PDB 身份轴和同次分数归一化仍彼此独立；
- 每个条件候选必须是严格布尔值，累计条件分母必须恰好等于 seen 与 unseen 数量之和。

## `EvaluationReport` 字段

### 身份基础对象

| 对象 | 字段 | 含义 |
| --- | --- | --- |
| `IntrinsicIdentityMetrics` | `top1` | Top-1 正确数 / `eligible_count` |
|  | `topk` | Top-K 命中数 / `eligible_count` |
|  | `top_k` | 配置最大名次数 |
|  | `eligible_count` | 当前身份范围可监督候选数 |
| `ConditionalIdentityMetrics` | `top1`、`topk`、`top_k`、`eligible_count` | 条件身份基本排名字段 |
|  | `seen_top1`、`seen_topk`、`seen_count` | 训练可见联合身份子集 |
|  | `unseen_top1`、`unseen_topk`、`unseen_count` | 训练未见联合身份子集 |
| `EndToEndIdentityMetrics` | `precision` | `true_positive/predicted_positive` |
|  | `recall` | `true_positive/actual_positive` |
|  | `f1` | 有限 precision 与 recall 的调和平均，未定义时为 0 |
|  | `true_positive` | 通过门且 Top-1 身份正确数 |
|  | `predicted_positive` | 通过门并产生范围内最终身份的候选数 |
|  | `actual_positive` | 当前范围固有身份分母 |

`IdentityScopeMetrics` 由 `intrinsic`、`conditional` 和 `end_to_end` 三个对象组成。范围对象内部的 `conditional` 只含四个基本排名字段；只有顶层 `conditional_identity` 补充 seen/unseen 六项。

### 顶层字段

| 字段 | 类型与内容 |
| --- | --- |
| `six_class` | `accuracy`、三个 macro-F1 字段、`per_class` 和 `confusion_matrix` |
| `intrinsic_identity` | 当前 `identity_scope` 的固有排名 |
| `conditional_identity` | 当前 `identity_scope` 的条件排名与 seen/unseen |
| `end_to_end_identity` | 当前 `identity_scope` 的端到端报告 |
| `small_molecule_identity` | 小分子范围的完整 `IdentityScopeMetrics` |
| `all_non_background_identity` | 五类总体范围的完整 `IdentityScopeMetrics` |
| `identity_per_class` | 五个固定前景类别名到 `IdentityScopeMetrics` 的映射 |
| `area` | 两项微平均 Dice 与两个整数分母 |

所有比率在 `compute` 时由完整划分整数计数统一计算，因而不依赖物理批次大小或切分。`compute` 不清空累计器，可在没有新增 `update` 时重复调用。

## 非有限值与 JSON 边界

模型的三类 logits 在 argmax 或阈值前必须全部有限，发现 `NaN` 或正负 `Inf` 会抛出 `FloatingPointError`。零支持产生的指标 `NaN` 是“没有分母”，不是模型非有限输出。

`EvaluationReport` 是内存对象，保留 `NaN`。训练跟踪适配层会省略或规范化未定义指标；`write_json` 在严格 JSON 边界把任意非有限浮点转换为 `null`。这既包括零分母指标的 `NaN`，也包括 `_LossStatistics` 用 Python float 跨批聚合时理论上可能溢出的损失正负 `Inf`。内存 TypedDict 仍把这些字段声明为 float；不要把内存非有限值、落盘 `null` 和数值 0 混为同一含义。

## validation 选模分数

`selection_score` 不读取报告或文件，只组合调用方传入的标量：

| `task_name` | 公式 |
| --- | --- |
| `candidate_pretrain` | `six_class_macro_f1` |
| `full_matcher` | `(six_class_macro_f1 + finite(conditional_identity_top1) + finite(end_to_end_identity_f1))/3` |

`finite(x)` 对有限值返回自身，对 `NaN` 或正负 `Inf` 返回 0。六分类 macro-F1 不做该回退；训练引擎会在保存 checkpoint 前拒绝最终非有限 selection score。

函数签名没有划分名，无法识别标量来自 validation 还是 calibration。正式调用约定只允许训练引擎用 validation 报告选模；calibration 入口不调用该函数。

## 一次性 calibration 输入

`run_calibration(plan_path, stage_name, checkpoint_path, output_path)` 的四项路径或名称都由调用方显式提供。

| 输入 | 门禁 |
| --- | --- |
| `plan_path` | 现存 UTF-8 正式 YAML；完整计划通过 `load_experiment_plan` |
| `stage_name` | 在计划中恰好出现一次 |
| `checkpoint_path` | POSIX 词法规范化后的文本位于 `/storage/penghongen/tmp/Matcher`，文件名恰为 `BEST.pt` 且存在；门禁不解析符号链接目标 |
| `output_path` | 同一服务器产物根内的 `.json` 路径，不得与 claim 或锁路径相同 |

`require_manifest` 只接受已经存在、关键字段相容并声明 `calibration` 的 manifest。入口不会调用 `ensure_manifest`，不会从 train/validation 选择 PDB，也不会新增候选或身份记录。两个 calibration JSONL 的字段 schema 在 `CandidateDataset` 消费时继续核验。

首次 Dataset 访问不是绝对只读：体素或 PP 版本化缓存缺失时，Store 可按 [`matcher/data/README.md`](../data/README.md) 的锁与原子发布契约在 `output_root` 下构建缓存。它不改变 manifest、候选 JSONL、身份 JSONL 或 checkpoint。

## checkpoint 自洽证据

### BEST.pt 顶层

| 字段 | 关系 |
| --- | --- |
| `stage_name`、`task_name` | 等于所选 YAML 阶段 |
| `epoch`、`best_epoch` | 都是非负 Python 整数且彼此相等 |
| `global_step` | 非负 Python 整数 |
| `best_score` | 非 bool 的 Python int 或 float 实例，转换为 float 后有限；其它可转换对象不被接受 |
| `epoch_record.epoch` | 等于 `epoch` |
| `epoch_record.global_step` | 等于顶层 `global_step` |
| `epoch_record.selection_score` | 等于 `best_score` |
| `epoch_record.improved` | 必须为 `True` |
| `model_config` | 经 `MatcherModelConfig.from_dict` 规范化后等于所选阶段 |
| `training_config` | 等于 `asdict(stage.training)` |

`load_checkpoint` 负责基础 checkpoint schema；`_validate_checkpoint_for_stage` 不再次核验 `model_state`，模型构造后用 `strict=True` 加载状态字典。

### 相邻训练产物

| 文件 | 核对字段 | 不证明的内容 |
| --- | --- | --- |
| 阶段 `resolved.json` | 完整 `stage`；`data_root`、`stage1_root`、`output_root`、`coverage_threshold`、`selection_seed`、`pp_top_k`、`local_crop_strategy` | 不核对 worker、缓存等其它 data 字段 |
| 阶段 `summary.json` | 阶段名、最佳轮次、最佳分数、`last_epoch>=best_epoch`、BEST 路径 | 训练引擎在 `tracker.finish(0)` 前写入；这里只核对所列本地字段，不证明 finish 成功、完整本地阶段收口或远端 W&B 状态 |
| 根 `run.json` | `plan_name` 与当前计划相同；`plan_path` 是仍存在的字符串路径 | 不要求训练 YAML 路径或字节与当前 calibration YAML 相同 |
| 训练 YAML | 对当前字节计算 SHA-256 | 不重新把它解析成第二份计划 |

calibration 不读取训练根 summary。阶段 `summary.json` 在 `tracker.finish(0)` 之前写入，即使该文件存在且上述字段一致，也只构成本函数列出的本地交叉核对，不证明 tracker 正常结束、完整本地阶段已经收口或远端 W&B run 已完成。

历史 `resolved.json` 缺少模型门初值时由 `MatcherModelConfig.from_dict` 补默认值；缺少 `identity_scope` 时按任务补入兼容值；缺少 `local_crop_strategy` 时补 `fixed_center`。补值只用于比较，不改写原文件。

`CalibrationCheckpointEvidence` 保存以下字段：

| 字段 | 内容 |
| --- | --- |
| `path`、`sha256` | BEST.pt 的 POSIX 词法规范化路径文本与字节摘要；路径门禁不解析符号链接目标 |
| `epoch`、`global_step`、`best_score` | 最佳轮次、更新数与 validation 分数 |
| `resolved_path`、`resolved_sha256` | 解析配置路径与摘要 |
| `summary_path`、`summary_sha256` | 阶段摘要路径与摘要 |
| `run_path`、`run_sha256` | 根运行记录路径与摘要 |
| `training_plan_path`、`training_plan_sha256` | 训练 YAML 当前路径与摘要 |
| `model_config` | 15 个规范化模型字段 |
| `training_config` | 训练批量、优化、精度、裁剪、类别权重和四项损失系数 |

`model_config` 的精确键是 `aggregation`、`use_manual_features`、`use_sampling_features`、`use_candidate_token_attention_bias`、`initial_graph_gate`、`initial_inventory_gate`、`model_dim`、`attention_heads`、`dropout`、`voxel_feature_channels`、`raunet_planes`、`recycle_iterations`、`gradient_checkpointing`、`bottleneck_attention_layers`、`graph_layers`。

`training_config` 的精确键是 `seed`、`physical_batch_size`、`gradient_accumulation`、`global_batch_size`、`learning_rate`、`weight_decay`、`maximum_epochs`、`patience_bad_epochs`、`precision`、`gradient_clip_norm`、`class_weights` 和 `loss_weights`；`loss_weights` 继续包含 `classification`、`ligand_area`、`binding_area`、`identity`。

## calibration 数据指纹

`CalibrationDataEvidence` 保存：

| 字段 | 内容 |
| --- | --- |
| `manifest_path` | 现存 manifest 路径 |
| `manifest_sha256` | manifest 全部字节摘要，创建时间或其它划分变化也会改变 |
| `training_manifest_sha256` | 当前实现固定为 `null` 的预留字段 |
| `manifest_semantic_sha256` | calibration 相关稳定 manifest 子对象的摘要 |
| `candidates_path`、`candidates_sha256` | calibration 候选 JSONL 路径与全部字节摘要 |
| `identities_path`、`identities_sha256` | calibration 身份 JSONL 路径与全部字节摘要 |
| `fingerprint` | 三个稳定内容摘要的组合摘要 |

`manifest_semantic_sha256` 使用 UTF-8、按键排序、紧凑分隔符且禁止 NaN/Inf 的 JSON 编码，覆盖以下字段：

- `schema_version`、`data_root`、`stage1_root`、`output_root`；
- `coverage_threshold`、`selection_seed`；
- `identity_key_fields`、`prepared_smiles_rule`、`identity_class_rule`；
- `split_pdb_rule`、`language_selection_rule`、`seen_identity_rule`；
- `class_names`；
- 完整 `splits.calibration` 对象；
- `split_pdb_counts.calibration`；
- `summary.splits.calibration`。

三个嵌套对象都按完整 JSON 值进入摘要，未识别附加键也会保留，但本函数不解析或验证附加键含义。正式 manifest 的已知字段是：

- `splits.calibration.candidates` 和 `splits.calibration.identities` 两个单级文件名；
- `split_pdb_counts.calibration.discovered`、`included`、`excluded` 三项 PDB 目录计数；
- `summary.splits.calibration.candidate_count`、`class_counts`、`identity_count`、`unique_object_key_count`、`same_pdb_object_key_multi_prepared_smiles_group_count`；`class_counts` 可按实际出现类别包含 `background`、`ion`、`nucleotide_like`、`peptide_like`、`small_molecule`、`sugar` 条件键；
- `summary.splits.calibration.matched_foreground_without_prepared_smiles_candidate_count`、`matched_foreground_explicit_null_candidate_count`、`matched_foreground_missing_prepared_file_candidate_count`；
- `summary.splits.calibration.language_available_identity_count`、`language_multi_model_smiles_identity_count`、`seen_positive_identity_count`，三个对象都继续包含 `molformer` 和 `smi_ted_289m` 两个已知语言家族键。

`created_at_utc` 和其它划分 summary 不进入语义摘要，但仍进入完整 `manifest_sha256`。候选与身份 JSONL 不在此阶段解析内容，只绑定完整字节。

manifest 的 JSON 解析和完整字节摘要是两次独立文件读取；BEST.pt 的反序列化和摘要也不是同一份内存字节。claim 锁只保护状态转换，不锁定这些输入文件。正式契约要求 release YAML、checkpoint、相邻训练产物、manifest 和 calibration JSONL 在评估期间保持不变；当前代码不提供跨文件系统快照，也不能阻止外部并发改写造成混合证据。

组合指纹使用固定顺序：

```text
manifest_semantic_sha256 + "\n"
candidates_sha256       + "\n"
identities_sha256       + "\n"
```

以上 ASCII 字节再计算 SHA-256。顺序或换行不同都会得到不同 fingerprint。

## claim 状态机与恢复

claim 路径是：

```text
<output_root>/calibration/claim_<data.fingerprint>.json
```

锁路径在 claim 文件名后追加 `.lock`。锁只串行化争用该文件锁的同一数据指纹合作调用者，不是 `try_lock`、`after_lock` 等 Slurm 资源锁，也不表示评估成功。跨节点互斥是否成立还依赖共享文件系统的锁语义；`output_path` 没有独立全局锁。

| claim 字段 | `claimed` | `completed` |
| --- | --- | --- |
| `schema_version` | 固定格式版本 | 不变 |
| `status` | `claimed` | `completed` |
| `data_fingerprint` | 首次绑定 | 不变 |
| `plan_sha256` | 首次绑定 | 不变 |
| `stage_name` | 首次绑定 | 不变 |
| `checkpoint_sha256`、`checkpoint_path` | 首次绑定的 BEST.pt | 不变 |
| `output_path` | 首次绑定 | 不变 |
| `output_sha256` | `null` | 已发布结果摘要 |
| `claimed_at_utc` | 首次 claim 时间 | 不变 |
| `completed_at_utc` | `null` | artifact 的 `finished_at_utc` |

既有 `claimed` 只允许相同数据、计划、阶段、checkpoint 和输出路径重试；任一不可变字段变化都会失败。对应 claim 文件保留且状态为 `completed` 时，合作调用者会拒绝再次评估；删除 claim 后入口看不到既有完成状态，仍可按新输出路径建立 claim。

成功写入顺序是：

1. 在锁内创建或复用 `claimed`；
2. 完成 CUDA 评估和样本数核验；
3. 原子发布 `output_path`；
4. 计算输出摘要并原子把 claim 改写为 `completed`。

若进程在第 3、4 步之间退出，下一次相同调用会核对已有 artifact 的 `schema_version`、`split`、`plan_sha256`、`stage_name`、`claim_path`、`checkpoint.sha256`、`data.fingerprint` 和 `finished_at_utc`，然后只完成 claim，不再次评估。恢复不重新核验 artifact 的全部指标或运行环境。本地检查观察到输出存在而 claim 不存在时，会把它视为孤立文件并拒绝恢复或覆盖。

两次 `output_path.exists()` 检查只能处理检查当时可见的文件。外部写者或不同数据 fingerprint 的调用者不争用同一 claim 锁，仍可能在检查与 `write_json` 的原子替换之间竞争；因此这里的“禁止覆盖”不是面向任意并发写者的绝对保证。

claim 创建后的其它失败保留 `claimed`，以便同一不可变选择重试。本入口不删除 claim、锁文件、输出或缓存。

## calibration 采样与运行环境

`CandidateDataset` 固定使用 `split_name=calibration`，其余身份、语言、PP 与裁剪配置沿用所选阶段和计划。`ShuffledBatchSampler` 固定：

| 字段 | 值 |
| --- | --- |
| `batch_size` | `stage.training.physical_batch_size` |
| `shuffle` | `False` |
| `drop_last` | `False` |
| `seed` | `stage.training.seed` |
| `sample_count_multiple` | `None` |

因此展平索引完整覆盖 Dataset，不做训练式洗牌或跨硬件截断。worker 生成器种子固定为 `2*seed+2`；DataLoader 使用 `validation_num_workers`、`prefetch_factor`、`spawn` 和 `persistent_workers=False`。

`CalibrationRuntimeEvidence` 字段如下：

| 字段 | 内容 |
| --- | --- |
| `candidate_count` | 构造 `CandidateDataset` 后调用 `len` 得到的当时候选数；Dataset 与输入文件不被快照或锁定 |
| `physical_batch_size` | 单次前向候选数 |
| `num_workers` | validation worker 数 |
| `prefetch_factor` | 每个 worker 预取批次数 |
| `cache_gib_per_worker` | 每个 `CandidateDataset` 实例分给五类 LRU 的总预算；DataLoader 场景通常每个 worker 副本各有实例，预算为 0 或单项超预算时不保留，且不是 RSS 硬上限 |
| `local_crop_strategy` | `fixed_center` 或 `v_centroid` |
| `multiprocessing_context` | `spawn` |
| `persistent_workers` | `False` |
| `worker_generator_seed` | `2*seed+2` |
| `numeric_backend` | 随机种子和四个 PyTorch 后端字段 |
| `device` | 实际 CUDA 设备名称 |
| `torch_version` | 实际 PyTorch 版本 |

`numeric_backend` 继续包含 `python_numpy_torch_seed`、`cudnn_benchmark`、`cudnn_deterministic`、`deterministic_algorithms` 和 `float32_matmul_precision`。这些字段记录实际设置，但不承诺跨 GPU 型号或跨软件版本逐位复现。

正式入口在无 CUDA 时失败，不回退到 CPU。模型结构由 `task_name` 选择，checkpoint 状态用 `strict=True` 加载。类别权重为 `None` 或 CUDA `float32 (6,)`，顺序与 `CLASS_NAMES` 一致。

`evaluate_model` 返回后，`evaluation.sample_count` 必须等于 `len(calibration_dataset)`；不等时禁止发布 artifact。

## calibration artifact

顶层字段如下：

| 字段 | 类型与来源 |
| --- | --- |
| `schema_version` | `CALIBRATION_SCHEMA_VERSION` |
| `split` | 固定 `calibration` |
| `plan_path`、`plan_sha256`、`plan_name` | 当前 calibration YAML 路径、摘要与计划名 |
| `stage_name`、`task_name` | 所选阶段名称与任务 |
| `language_family`、`identity_variant` | 完整 Matcher 科学字段；候选预训练为 `null` |
| `checkpoint` | `CalibrationCheckpointEvidence` |
| `data` | `CalibrationDataEvidence` |
| `claim_path` | 本 artifact 对应 claim 路径 |
| `runtime` | `CalibrationRuntimeEvidence` |
| `evaluation` | 完整 `EvaluationResult` |
| `hostname` | 当前节点名称 |
| `slurm_job_id` | `SLURM_JOB_ID` 文本或 `null` |
| `task_launch_dir` | `TASK_LAUNCH_DIR` 文本或 `null` |
| `started_at_utc`、`finished_at_utc` | 带时区 ISO 8601 时间文本 |

`evaluation` 含：

| 字段 | 内容 |
| --- | --- |
| `loss.total` | 四项加权损失总和 |
| `loss.classification` | 六分类全局归一损失 |
| `loss.ligand_area` | 配体区域逐候选均值 |
| `loss.binding_area` | 结合区域逐候选均值 |
| `loss.identity` | 有效身份候选均值，无有效候选时为 0 |
| `metrics` | 本文定义的完整 `EvaluationReport` |
| `sample_count` | 实际遍历候选数 |
| `identity_eligible_count` | `compute_matcher_loss` 的 `identity_eligible_mask` 计数；除 `identity_target>=0` 外还受当前 `stage.identity_scope` 限定，候选预训练或 `identity_scope=null` 时为 0 |
| `elapsed_seconds` | 完整评估墙钟秒数 |
| `samples_per_second` | 候选吞吐 |

`slurm_job_id` 和 `task_launch_dir` 只是环境文本，本入口不验证 Job、release、launch 文件或远端状态；非空值不能单独证明 provenance 完整。

artifact 顶层没有独立 `identity_scope` 字段。`identity_eligible_count` 的统计范围只能结合本次绑定的计划、`resolved.json` 和 checkpoint 配置解释，不能只从该顶层计数字段反推出身份范围。

## 命令入口

`python -m matcher.evaluation.calibration` 接受四个必填参数：

| 参数 | argparse 类型 | 后续门禁 |
| --- | --- | --- |
| `--plan` | `Path` | 正式 YAML 且存在 |
| `--stage` | 字符串 | 计划中唯一阶段名 |
| `--checkpoint` | `Path` | Matcher 根内现存 `BEST.pt` |
| `--output` | `Path` | Matcher 根内 `.json`；本地两次存在性检查拒绝当时观察到的既有或新出现文件，但不提供独立全局输出锁 |

argparse 的 description 来自模块 Docstring；说明文字变化不改变参数名、required 状态、Path 转换或退出语义。成功时先打印一个严格 JSON 事件，再打印结果 Path；异常不在 `main` 中吞掉。

## 失败关闭与验证边界

- logits 含非有限值、批次轴错位、身份目标或槽类别错误会在指标累计中失败；
- 缺失或不相容 checkpoint、resolved、summary、run、训练 YAML、manifest 或 calibration JSONL 会在评估前失败；
- claim 与既有选择不一致、claim 文件仍保留时的 completed 重入、本地检查观察到的孤立输出或评估期间新出现输出都会失败；
- 无 CUDA、模型状态不相容、Dataset/loader 读取失败、样本数不一致或严格 JSON 发布失败都不会把 claim 标为 completed；
- 评估器和 calibration 状态转换都不是全流程事务，已增加的内存计数、已创建的 claimed 文件或已构建的 Dataset 缓存不会自动回滚；
- 本 README 和注释批次只读取源码与合成测试，不读取任何真实 calibration 划分、结果或服务器文件。

相称回归由 [`matcher/tests/test_metrics_and_config.py`](../tests/test_metrics_and_config.py) 覆盖指标公式、非有限值、轴对齐、selection、checkpoint 和数据指纹，由 [`matcher/tests/test_calibration.py`](../tests/test_calibration.py) 用合成 stub 覆盖 artifact、claim、崩溃恢复与第二次调用拒绝。
