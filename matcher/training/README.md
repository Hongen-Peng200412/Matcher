# training：四变体共用训练、选模与 checkpoint

本目录把一份严格 YAML 配置执行为单卡从零训练，使用完整 validation 选择 BEST，并以原子替换发布 checkpoint。正式入口是 `python -m matcher.training.experiment --config <YAML>`。

## 推荐阅读顺序

1. `config.py`：理解 YAML 的精确字段、类型和跨字段约束。
2. `losses.py`：理解前景、区域和直接 SMILES 身份交叉熵。
3. `engine.py`：理解训练、完整 validation、早停、阈值拟合和运行产物。
4. `checkpointing.py`：理解 BEST/LAST 的原子保存和最小加载核验。
5. `experiment.py`：查看命令行入口。

## 损失与选模

```text
total = foreground_weight × foreground_bce
      + ligand_area_weight × ligand_area_loss
      + binding_area_weight × binding_area_loss
      + identity_weight × smiles_identity_ce
```

`smiles_identity_ce` 只在 `foreground_target & (identity_target >= 0)` 上计算，不乘前景预测概率，也不依赖前景是否预测正确。每个完整 epoch 后计算：

```text
selection = (foreground PRAUC + 真实前景候选上的 SMILES 匹配准确率) / 2
```

selection 严格改善时覆盖 `BEST.pt`，每轮覆盖 `LAST.pt`。训练结束后重新载入 BEST 权重，只在 validation 上选择 F1 最大、并列时较高的前景阈值，再写回 BEST；F1 不参与选模。

## 运行目录

```text
<output_root>/<name>/<run_stamp>/
├── run.json                 # 启动环境、正式配置和 manifest 路径绑定
├── resolved_config.json     # YAML 经过严格解析后的完整配置值
├── manifest.snapshot.json   # 启动时读取的 manifest.json 内容副本
├── history.jsonl            # 每轮事件和最终阈值事件
├── BEST.pt                  # selection 最优权重及 validation 阈值
├── LAST.pt                  # 最后一轮权重和优化器状态
└── summary.json             # 仅在自然完成或早停后发布的收口摘要
```

异常退出不伪造 `summary.json`。`resolved_config.json` 保存 dataclass 解析后的完整配置值，供读者区别原始 YAML 文字与实际输入。`manifest.snapshot.json` 冻结入口内容，并把候选与身份 JSONL 解析为绝对路径；Dataset 直接读取该快照。训练不会把外部 JSONL 或图包复制进运行目录。

## `run.json`

| 字段 | 类型 | 含义与示例 |
| --- | --- | --- |
| `schema` | `str` | 固定为 `"matcher_smiles_training_v1"`。 |
| `run_stamp` | `str` | 运行目录名；例如 `"20260917T010203Z"`。 |
| `name` | `str` | YAML 的实验名；例如 `"strongest"`。 |
| `started_at_utc` | ISO 8601 `str` | UTC 启动时间。 |
| `hostname` | `str` | 训练工作进程所在主机。 |
| `slurm_job_id` | `str|null` | `SLURM_JOB_ID`；非 Slurm 环境为 `null`。 |
| `task_launch_dir` | `str|null` | 调度包装器提供的正式 launch 目录；未设置时为 `null`。 |
| `config_path` | `str` | 本次解析的正式 YAML 绝对路径。 |
| `resolved_config_path` | `str` | 同一运行目录中的 `resolved_config.json` 路径。 |
| `manifest_path` | `str` | 本次解析的 `manifest.json` 绝对路径。 |
| `manifest_snapshot_path` | `str` | 同一运行目录中的 `manifest.snapshot.json` 路径。 |
| `worker_provenance.python` | `str` | Python 版本；例如 `"3.11.9"`。 |
| `worker_provenance.torch` | `str` | PyTorch 版本。 |
| `worker_provenance.cuda_runtime` | `str|null` | PyTorch 声明的 CUDA runtime；CPU 构建可为 `null`，但正式训练会在更早处拒绝无 GPU 环境。 |
| `worker_provenance.gpu_name` | `str` | 当前 CUDA 设备名称。 |
| `worker_provenance.num_workers` | `int >= 0` | 配置中的 DataLoader 工作进程数。 |

## `resolved_config.json` 与 `manifest.snapshot.json`

`resolved_config.json` 的顶层字段是 `name`、`output_root`、`seed`、`data`、`model`、`loss_weights`、`optimization` 和 `tracking`。嵌套字段与 `config.py` 的 dataclass 一一对应；Path 值写为字符串，`raunet_planes` 元组写为 JSON 列表。`manifest.snapshot.json` 的字段契约与 `matcher/data/README.md` 中的 `manifest.json` 完全相同，但 `splits.*.candidates` 和 `splits.*.identities` 已解析为不可变代际 JSONL 的绝对路径。train 与 validation Dataset 均从该快照加载，因此原始入口后续发布不会改变本次运行的数据来源。

## `history.jsonl`

普通 epoch 行包含 `epoch`、`global_step`、`training`、`validation`、`selection`、`improved` 和 `bad_validation_epochs`。`training` 保存每个损失的物理批次均值及 `identity_eligible_count`；`validation` 保存 validation 损失和下述指标。最后一行阈值事件另含 `event="best_threshold"`、`best_epoch`、`foreground_threshold`、`foreground` 和重新计算的 `validation`。

## `BEST.pt` 与 `LAST.pt`

两者都是 `torch.save` 映射，以下字段在每轮 checkpoint 中同时存在：

| 字段 | 类型或结构 | 含义与特殊值 |
| --- | --- | --- |
| `schema_version` | `int` | 当前固定为 1。 |
| `epoch` | `int >= 0` | 该 checkpoint 对应的完整 epoch。 |
| `global_step` | `int >= 0` | 已完成的优化器更新次数。 |
| `best_epoch` | `int >= 0` | 截至该轮 selection 最优的 epoch。 |
| `best_score` | 有限 `float` | 截至该轮最高 selection。 |
| `bad_validation_epochs` | `int >= 0` | 连续未严格改善次数。 |
| `model_config` | `dict` | `MatcherModelConfig` 的十个结构字段，用于按原结构恢复模型。 |
| `experiment_config` | `dict` | 完整训练配置的 JSON 兼容映射；路径序列化为字符串。 |
| `manifest_path` | `str` | 本轮读取的正式 manifest 路径。 |
| `manifest_snapshot_path` | `str` | 运行目录内入口快照路径。 |
| `model_state` | `dict[str, Tensor]` | 参数和缓冲区名称到张量的映射；加载设备由 `map_location` 决定。 |
| `optimizer_state` | `dict` | AdamW 的参数组和动量状态；正式 test 不读取。 |
| `validation` | `dict` | 该轮 validation 损失和指标。 |

训练收口后，`BEST.pt` 额外包含：

| 字段 | 类型或结构 | 含义 |
| --- | --- | --- |
| `foreground_threshold` | 有限 `float` | 只由 BEST 权重在 validation 上拟合的 F1 阈值。 |
| `positive_class_definition` | `str` | 固定为 `"five_true_ligand_categories"`。 |
| `best_validation.foreground_f1` | `float` | 该阈值在 validation 上的前景 F1。 |
| `best_validation.foreground_prauc` | `float` | BEST validation 前景 PRAUC。 |
| `best_validation.smiles_match_overall_accuracy` | `float|null` | 受支持真前景的 SMILES 匹配准确率；零支持时为 `null`。 |
| `best_validation.selection` | `float` | BEST 的正式选模分数。 |

`LAST.pt` 不保证含上述三个收口字段。checkpoint 通过同目录进程专属临时文件写入、刷新、同步后原子替换目标；加载器只最低限度验证 `schema_version` 和 `model_state`，其余字段由具体调用者验证。

## validation 指标结构

`validation.metrics` 的 `candidate_count` 是候选数，`foreground_prauc` 是全候选前景 PRAUC，`selection` 是正式选模值。`smiles_match` 映射含 `overall` 和五个类别键；每项均含 `correct_count`、`eligible_count` 和 `accuracy`，零支持时 `accuracy=null`。传入固定阈值时还含 `foreground_threshold`，以及 `foreground.true_positive`、`false_positive`、`false_negative`、`true_negative` 和 `f1`。

## `summary.json`

`summary.json` 复制 `run.json` 全部来源字段，并增加 `finished_at_utc`、`best_epoch`、`best_score`、`last_epoch`、`global_step`、`stopped_by_patience`、`foreground_threshold` 和 `tracking_url`。`tracking_url` 在 W&B 未返回网址时为 `null`。
