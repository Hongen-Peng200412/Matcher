# strongest-1 实验日志

本记录保存 `strongest-1` 的单实验追溯。正式模型使用 106 通道体素输入、A/P/V/PP token、末端无多尺度卷积的 RAUNet, 以及初值为 `0.01` 的 A/P/PP 独立门。

## 当前状态

- 阶段: 旧 `0.30/0.05/0.05/1.00` 训练已按用户授权中断并由新契约取代；当前等待新 release 的实际 H100 门禁与从头训练。
- 正式资源: H100 Job `378587`，30 个阶段互斥 worker；`after_lock_378587` 保留。
- 正式实验目录: `测试结果与追溯/experiment-centric/strongest-1/`。
- 新 release、门禁 launch、训练 launch、运行根与 W&B：待新契约正式触发后填写。
- 测试 release：与本轮训练共用的新 release；待实际 H100 门禁通过并触发正式训练后填写。全测试集与 Stage3 小分子测试必须从该训练 release 快照恢复模型类、配置和 checkpoint。
- BEST checkpoint 与两套评测产物: 待训练自然停止后填写。
- 正式 YAML: `configs/experiments/final_smiles/strongest.yaml`；实验标识 `strongest-1`，30 个阶段互斥 worker，修订后的损失权重 `0.30/0.00/0.00/1.00`。

## 正式命令

`bash 训练与运行/sh/final_smiles/train_strongest.sh`

训练自然停止后使用：

`bash 训练与运行/sh/final_smiles/evaluate_strongest_1.sh`

`bash 训练与运行/sh/small_molecule_test/evaluate_strongest_1.sh`

`bash 训练与运行/sh/small_molecule_test/plot_strongest_1.sh`

## 失败或中断尝试

- 首轮 H100 门禁 launch `gate_strongest1_job378587_20260920T141757` 在加载 Dataset 前退出：绝对路径执行的临时门禁脚本没有把 Matcher release 根加入模块搜索路径，报 `ModuleNotFoundError: matcher`。Job 自然恢复 `try_lock`，未启动正式训练。第二次门禁只补充 `PYTHONPATH`，科学配置、release、manifest、GPU 和 worker 数均未改变。
- 被新契约取代的旧正式训练使用 release `/home/penghongen/Feedback/Matcher/releases/Matcher_6701e7a7ca8d/Matcher`、门禁 launch `gate_strongest1_job378587_20260920T142200`、训练 launch `train_strongest1_job378587_20260920T142850`、运行根 `/storage/penghongen/Matcher/runs/strongest-1/train_strongest1_job378587_20260920T142850` 和 W&B `https://wandb.ai/pencounkdual-111/Matcher/runs/qm0n26ai`。该运行使用旧损失权重 `0.30/0.05/0.05/1.00`，在第 13 次完整 validation 后由精确 `kill_lock_378587` 中断；没有伪造 `summary.json`。13 份逐检查 checkpoint、BEST、LAST、历史和 W&B 均原样保留。旧 BEST 来自 epoch 2、第 8 次 validation，PRAUC/SMILES 准确率/selection 为 `0.7958746329/0.9024238658/0.8778355813`；BEST SHA-256 `cedec8d1a58bcc007060df94e28b9c70d55d4a1f3a0dbf01b148b11f7d28b4b3`，LAST SHA-256 `1068a147952f0a215c68c1fb216287f3a41871f5f1d100f6796a6880dd524440`。该尝试不进入最终 strongest-1 结果。
