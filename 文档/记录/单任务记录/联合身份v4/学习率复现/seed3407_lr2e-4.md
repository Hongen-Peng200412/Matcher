# 联合身份 v4：seed 3407、LR 2e-4

- 任务：A800 学习率对照；除学习率外与 Job 340894 的科学配置相同。
- Job：`340895`，A800，16 CPU。
- 配置：`configs/experiments/joint_identity_v4/joint_identity_v4_lr_2e4_oracle_a100.yaml`。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_28e1833e0179/Matcher`；正式 Matcher 源码与 Job 340894 release 一致，新 release 只增加学习率矩阵配置、配置测试和说明。
- 动态命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" --plan 'configs/experiments/joint_identity_v4/joint_identity_v4_lr_2e4_oracle_a100.yaml'`。
- launch：`train_experiment_job340895_20260815T122350_a7`，创建于 2026-08-15 12:23:50（Asia/Shanghai）。
- W&B：entity `pencounkdual-111`，project `AdaLigand-Matcher-JointIdentityV4`，group `joint-identity-v4-learning-rate`，run `s80l6dbp`。
- 运行根：`/storage/penghongen/tmp/Matcher/experiments/joint_identity_v4_lr_2e4_oracle_a100/train_experiment_job340895_20260815T122350_a7/`。
- 状态：科学正确性与严格技术表达两项独立终审均 PASS 后，由摘要为 `f7e4694d5c056d87ef12ea2212f6999e3b7716a57026d3e53aff05f7c6f6ccfa` 的一次性控制器触发。任务自然早停时恢复 `try_lock_340895`；该 allocation 随后已用于摘要拼接任务，当前锁状态见相应摘要拼接单任务记录。
- 13:31 进度：epoch 0/1 selection 为 `0.6362130808272672/0.6253657602455567`，当前 BEST 仍为 epoch 0。epoch 1 后调度器把下一轮学习率从 `2e-4` 降为 `1e-4`；任务仍在运行，不能用异步早期 epoch 与配对项作最终比较。
- 15:02 进度：epoch 2/3 selection 为 `0.6526770812716048/0.6787082021150166`，epoch 3 刷新当前 BEST；实际学习率保持 `1e-4`。Job 仍为 `RUNNING`，只有 `after_lock_340895`，没有 `try/pre/kill` 锁，尚未自然早停。
- 15:35 进度：epoch 4 selection 为 `0.6885123509717196`，刷新当前 BEST；实际学习率保持 `1e-4`。Job 继续 `RUNNING`，仅保留 `after_lock_340895`，尚未自然早停。
- 最终结果：BEST/last epoch 为 `8/10`，global step 为 `4917`，BEST selection score 为 `0.6914841865879034`。阶段与运行根 summary、BEST、LAST 均已写出；相对同随机种子 `1e-4` 的 `0.6920686526492751` 低 `0.084452%`，属于近似持平的单配对结果。
