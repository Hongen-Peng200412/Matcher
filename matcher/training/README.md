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

`smiles_identity_ce` 只在 `foreground_target & (identity_target >= 0)` 上计算，不乘前景预测概率，也不依赖前景是否预测正确。每个 epoch 的优化器更新按顺序划为三个连续区段；同一训练 DataLoader 只遍历一次，每个区段后遍历一次完整 validation 并计算：

```text
selection = (foreground PRAUC + 真实前景候选上的 SMILES 匹配准确率) / 2
```

`selection` 相对当前 BEST 至少提高 `0.001` 才更新 BEST 并清零坏检查数。连续两次未改善只累计状态，第三次未改善触发平台动作：前两次平台分别把当前学习率乘以 `0.8`，第三次平台不再降低学习率并直接早停。累计降低次数不因后续改善而清零。训练最多执行 15 个 epoch、45 次完整 validation。

每次完整 validation 都保存独立 checkpoint 并覆盖 `LAST.pt`；有效改善时另行覆盖 `BEST.pt`。训练结束后直接复用当时保留的 BEST validation 预测选择 F1 最大、并列时较高的前景阈值，再写回 BEST；不会额外遍历 validation，F1 也不参与选模。

## 运行目录

```text
<output_root>/<name>/<run_stamp>/
├── run.json                 # 启动环境、正式配置和 manifest 路径绑定
├── resolved_config.json     # YAML 经过严格解析后的完整配置值
├── manifest.snapshot.json   # 启动时读取的 manifest.json 内容副本
├── history.jsonl            # 每轮事件和最终阈值事件
├── checkpoints/             # 每次完整 validation 后保留的独立 checkpoint
├── BEST.pt                  # selection 最优权重及 validation 阈值
├── LAST.pt                  # 最后一次 validation 的权重和优化器状态
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
| `source_project_root` | `str` | 当前 Python 实际导入的 Matcher release 项目根；不依赖调度器原任务根。 |
| `source_release_manifest_path` | `str|null` | release 内容身份文件；直接从开发目录运行时为 `null`。 |
| `config_path` | `str` | 本次解析的正式 YAML 绝对路径。 |
| `resolved_config_path` | `str` | 同一运行目录中的 `resolved_config.json` 路径。 |
| `manifest_path` | `str` | 本次解析的 `manifest.json` 绝对路径。 |
| `manifest_snapshot_path` | `str` | 同一运行目录中的 `manifest.snapshot.json` 路径。 |
| `worker_provenance.python` | `str` | Python 版本；例如 `"3.11.9"`。 |
| `worker_provenance.torch` | `str` | PyTorch 版本。 |
| `worker_provenance.cuda_runtime` | `str|null` | PyTorch 声明的 CUDA runtime；CPU 构建可为 `null`，但正式训练会在更早处拒绝无 GPU 环境。 |
| `worker_provenance.gpu_name` | `str` | 当前 CUDA 设备名称。 |
| `worker_provenance.gpu_capability` | `list[int]`，长度 2 | 当前 GPU 的 CUDA 计算能力主次版本。 |
| `worker_provenance.gpu_total_memory_bytes` | `int` | 当前 GPU 的总显存字节数。 |
| `worker_provenance.cuda_visible_devices` | `str|null` | Slurm 注入的可见 GPU 编号；未设置时为 `null`。 |
| `worker_provenance.num_workers` | `int >= 0` | 配置中的 DataLoader 工作进程数。 |

## `resolved_config.json` 与 `manifest.snapshot.json`

`resolved_config.json` 的顶层字段是 `name`、`output_root`、`seed`、`data`、`model`、`loss_weights`、`optimization` 和 `tracking`。嵌套字段与 `config.py` 的 dataclass 一一对应；Path 值写为字符串，`raunet_planes` 元组写为 JSON 列表。`manifest.snapshot.json` 的字段契约与 `matcher/data/README.md` 中的 `manifest.json` 完全相同，但 `splits.*.candidates` 和 `splits.*.identities` 已解析为不可变代际 JSONL 的绝对路径。train 与 validation Dataset 均从该快照加载，因此原始入口后续发布不会改变本次运行的数据来源。

## `history.jsonl`

普通检查记录的 `event` 是 `validation_check`，并保存 `epoch`、`epoch_segment`、`validation_check`、`global_step`、`training`、`validation`、`selection`、四项控制决定、BEST 状态、坏检查数、累计学习率降低次数和当前学习率。`training` 保存前一个连续区段的损失均值、身份支持数和优化器更新数。最后一条阈值记录的 `event` 是 `best_threshold`，并保存 BEST 检查编号、前景阈值、F1 结果和当时已经计算的完整 validation；该记录不是新增 validation。

## `BEST.pt` 与 `LAST.pt`

两者都是 `torch.save` 映射，以下字段在每轮 checkpoint 中同时存在：

| 字段 | 类型或结构 | 含义与特殊值 |
| --- | --- | --- |
| `schema_version` | `int` | 当前固定为 1。 |
| `epoch` | `int >= 0` | 该 checkpoint 对应的完整 epoch。 |
| `epoch_segment` | `int`，范围 1–3 | 该 checkpoint 对应的 epoch 内连续训练区段。 |
| `validation_check` | `int >= 1` | 整次训练中的完整 validation 顺序编号。 |
| `global_step` | `int >= 0` | 已完成的优化器更新次数。 |
| `best_epoch` | `int >= 0` | 截至该轮 selection 最优的 epoch。 |
| `best_validation_check` | `int >= 1` | 截至当前的 BEST validation 顺序编号。 |
| `best_score` | 有限 `float` | 截至该轮最高 selection。 |
| `bad_validation_checks` | `int >= 0` | 自上次有效改善或平台动作后的连续未改善次数。 |
| `learning_rate_reductions` | `int`，范围 0–2 | 整次训练已实际降低学习率的累计次数。 |
| `learning_rate` | 有限正 `float` | 本次平台动作完成后的当前学习率。 |
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

`summary.json` 复制 `run.json` 全部来源字段，并增加 `finished_at_utc`、BEST epoch 与检查编号、`best_score`、`last_epoch`、`global_step`、validation 总次数、累计学习率降低次数、早停或 epoch 上限停止标记、前景阈值和 `tracking_url`。`tracking_url` 在 W&B 未返回网址时为 `null`。
