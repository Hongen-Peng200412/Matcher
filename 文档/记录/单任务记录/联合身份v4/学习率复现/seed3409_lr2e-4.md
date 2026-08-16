# 联合身份 v4：seed 3409、LR 2e-4

- 任务：第二对 A100 学习率复现的高学习率项；除学习率外与 Job 340897 相同。
- Job：`340899`，A100，16 CPU。
- 配置：`configs/experiments/joint_identity_v4/joint_identity_v4_seed_3409_lr_2e4_oracle_a100.yaml`。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_28e1833e0179/Matcher`。
- 动态命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" --plan 'configs/experiments/joint_identity_v4/joint_identity_v4_seed_3409_lr_2e4_oracle_a100.yaml'`。
- launch：`train_experiment_job340899_20260815T122600_a5`，创建于 2026-08-15 12:26:00（Asia/Shanghai）。
- W&B：entity `pencounkdual-111`，project `AdaLigand-Matcher-JointIdentityV4`，group `joint-identity-v4-learning-rate-seed-3409`，run `rgyq1i8k`。
- 运行根：`/storage/penghongen/tmp/Matcher/experiments/joint_identity_v4_seed_3409_lr_2e4_oracle_a100/train_experiment_job340899_20260815T122600_a5/`。
- 状态：双独立终审 PASS 后受保护触发；任务自然早停时恢复 `try_lock_340899`。该 allocation 随后已用于摘要拼接任务，当前锁状态见相应摘要拼接单任务记录。
- 13:31 进度：epoch 0/1 selection 为 `0.6150331204859985/0.6410453244131187`，epoch 1 为当前 BEST；学习率仍为 `2e-4`，任务继续运行。
- 15:02 进度：epoch 2/3 selection 为 `0.6490450794046766/0.6562047225560245`，epoch 3 刷新当前 BEST；实际学习率仍为 `2e-4`。Job 仍为 `RUNNING`，只有 `after_lock_340899`，没有 `try/pre/kill` 锁，尚未自然早停。
- 15:35 进度：epoch 4 selection 为 `0.6778490020766442`，刷新当前 BEST；实际学习率仍为 `2e-4`。Job 继续 `RUNNING`，仅保留 `after_lock_340899`，尚未自然早停。
- 最终结果：BEST/last epoch 为 `7/9`，global step 为 `4470`，BEST selection score 为 `0.6967620030292677`。阶段与运行根 summary、BEST、LAST 均已写出；相对同随机种子 `1e-4` 提高 `0.585417%`。
