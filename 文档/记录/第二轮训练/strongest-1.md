# strongest-1 实验日志

本记录保存 `strongest-1` 的单实验追溯。正式模型使用 106 通道体素输入、A/P/V/PP token、末端无多尺度卷积的 RAUNet, 以及初值为 `0.01` 的 A/P/PP 独立门。

## 当前状态

- 阶段：修订后的 `0.30/0.00/0.00/1.00` 训练、全测试集评估和 Stage3 小分子评估均已自然完成。
- 正式资源：H100 Job `378587`，30 个阶段互斥 worker；全部命令结束后已恢复 `try_lock_378587`，`after_lock_378587` 保留，未发现残留训练或评估进程。
- 正式实验目录: `测试结果与追溯/experiment-centric/strongest-1/`。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_a67eb7f830e0/Matcher`；内容 SHA-256 `a67eb7f830e0a3625da349daf9481fe653f2231ba4d860bc9aa8a1101a760f87`；对应实现提交 `ab87ead233e08c70fb387bae081403ebcbb4eba7`。
- 门禁 launch：`/home/penghongen/Feedback/Matcher/launches/378587/gate_strongest1_revised_job378587_20260920T230032`；报告：`/home/penghongen/Feedback/Matcher/gates/second_round/strongest-1/gate_strongest1_revised_job378587_20260920T230032.json`，`passed=true`。
- 训练 launch：`/home/penghongen/Feedback/Matcher/launches/378587/train_strongest1_revised_job378587_20260920T230727`。
- 运行根：`/storage/penghongen/Matcher/runs/strongest-1/train_strongest1_revised_job378587_20260920T230727`。
- W&B：`https://wandb.ai/pencounkdual-111/Matcher/runs/smn9trvg`。
- 测试 release：`/home/penghongen/Feedback/Matcher/releases/Matcher_cab21d9ceea5/Matcher`；内容 SHA-256 `cab21d9ceea55c5c53b0f5d4589d5c34b1148cff8b77ed42691bda770fc91cb6`。模型类、配置和 checkpoint 加载器均来自训练 release 快照。
- BEST：epoch 6、第 20 次 validation，`global_step=9300`；PRAUC `0.7986851409542616`、SMILES 准确率 `0.9101926662523306`、selection `0.8844601604143147`。BEST SHA-256 为 `4d1109c403b2b5f64c0f220d3ee8643298bf66f74590bee799871bcffa404058`，冻结阈值为 `0.359375`。
- 自然停止：epoch 7、第 24 次 validation 在两次实际降学习率后达到平台早停；24 份逐检查 checkpoint 均保留。
- 全测试集：launch `evaluate_full_strongest1_revised_job378587_20260921T151639`；`frozen_test.json` SHA-256 `2ddfb14c063c8cdbe1ffebeb7be9493ae6014d73bf6e6be9eaca1746cb85495d`。
- Stage3 小分子：前向 launch `evaluate_small_strongest1_revised_job378587_20260921T153226`，绘图 launch `plot_small_strongest1_revised_job378587_20260921T154815`；`evaluation.json` SHA-256 `e3be9998b833ecc0f39f604c2500e17173987ed5f411043a27d9007c3a423880`。
- 正式 YAML: `configs/experiments/final_smiles/strongest.yaml`；实验标识 `strongest-1`，30 个阶段互斥 worker，修订后的损失权重 `0.30/0.00/0.00/1.00`。
- 首次完整 validation：epoch 0、第 1 区段、`global_step=465`；前景 PRAUC `0.766326075409999`、SMILES 匹配准确率 `0.8502175264139217`、selection `0.8308579607976317`。该记录只用于确认正式运行稳定，不替代最终 BEST。

## 正式命令

`bash 训练与运行/sh/final_smiles/train_strongest.sh`

`bash 训练与运行/sh/final_smiles/evaluate_strongest_1.sh`

`bash 训练与运行/sh/small_molecule_test/evaluate_strongest_1.sh`

`bash 训练与运行/sh/small_molecule_test/plot_strongest_1.sh`

## 失败或中断尝试

- 首轮 H100 门禁 launch `gate_strongest1_job378587_20260920T141757` 在加载 Dataset 前退出：绝对路径执行的临时门禁脚本没有把 Matcher release 根加入模块搜索路径，报 `ModuleNotFoundError: matcher`。Job 自然恢复 `try_lock`，未启动正式训练。第二次门禁只补充 `PYTHONPATH`，科学配置、release、manifest、GPU 和 worker 数均未改变。
- 被新契约取代的旧正式训练使用 release `/home/penghongen/Feedback/Matcher/releases/Matcher_6701e7a7ca8d/Matcher`、门禁 launch `gate_strongest1_job378587_20260920T142200`、训练 launch `train_strongest1_job378587_20260920T142850`、运行根 `/storage/penghongen/Matcher/runs/strongest-1/train_strongest1_job378587_20260920T142850` 和 W&B `https://wandb.ai/pencounkdual-111/Matcher/runs/qm0n26ai`。该运行使用旧损失权重 `0.30/0.05/0.05/1.00`，在第 13 次完整 validation 后由精确 `kill_lock_378587` 中断；没有伪造 `summary.json`。13 份逐检查 checkpoint、BEST、LAST、历史和 W&B 均原样保留。旧 BEST 来自 epoch 2、第 8 次 validation，PRAUC/SMILES 准确率/selection 为 `0.7958746329/0.9024238658/0.8778355813`；BEST SHA-256 `cedec8d1a58bcc007060df94e28b9c70d55d4a1f3a0dbf01b148b11f7d28b4b3`，LAST SHA-256 `1068a147952f0a215c68c1fb216287f3a41871f5f1d100f6796a6880dd524440`。该尝试不进入最终 strongest-1 结果。
