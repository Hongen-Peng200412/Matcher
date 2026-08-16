# 联合身份 v4：seed 3408、LR 2e-4

- 任务：第一对 A100 学习率复现的高学习率项；除学习率外与 Job 340893 相同。
- Job：`340898`，A100，16 CPU。
- 配置：`configs/experiments/joint_identity_v4/joint_identity_v4_seed_3408_lr_2e4_oracle_a100.yaml`。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_28e1833e0179/Matcher`。
- 动态命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" --plan 'configs/experiments/joint_identity_v4/joint_identity_v4_seed_3408_lr_2e4_oracle_a100.yaml'`。
- launch：`train_experiment_job340898_20260815T122554_a5`，创建于 2026-08-15 12:25:54（Asia/Shanghai）。
- W&B：entity `pencounkdual-111`，project `AdaLigand-Matcher-JointIdentityV4`，group `joint-identity-v4-learning-rate-seed-3408`，run `7csoon5v`。
- 运行根：`/storage/penghongen/tmp/Matcher/experiments/joint_identity_v4_seed_3408_lr_2e4_oracle_a100/train_experiment_job340898_20260815T122554_a5/`。
- 状态：双独立终审 PASS 后受保护触发；任务自然早停时恢复 `try_lock_340898`。该 allocation 随后已用于摘要拼接任务，当前锁状态见相应摘要拼接单任务记录。
- 13:31 进度：epoch 0 selection 为 `0.6016802541033975`，validation `sample_count=6495`、`identity_eligible_count=1466`；学习率仍为 `2e-4`，任务继续运行。
- 15:02 进度：epoch 1–3 selection 依次为 `0.6426669471259302/0.6655444622455226/0.6601393454023504`，当前 BEST 为 epoch 2；epoch 3 为第一个坏验证轮，调度器把下一轮学习率从 `2e-4` 降为 `1e-4`。Job 仍为 `RUNNING`，只有 `after_lock_340898`，没有 `try/pre/kill` 锁，尚未自然早停。
- 15:35 进度：epoch 4 selection 为 `0.6887014183631415`，在实际学习率 `1e-4` 下刷新当前 BEST，并把坏验证轮数清零。Job 继续 `RUNNING`，仅保留 `after_lock_340898`，尚未自然早停。
- 最终结果：BEST/last epoch 为 `8/10`，global step 为 `4917`，BEST selection score 为 `0.6985805289999240`。阶段与运行根 summary、BEST、LAST 均已写出；相对同随机种子 `1e-4` 提高 `0.970219%`。
