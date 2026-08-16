# 联合身份 v4：seed 3409、LR 1e-4

- 任务：第二对 A100 学习率复现的基线；与 Job 340899 构成直接配对。
- Job：`340897`，A100，16 CPU。
- 配置：`configs/experiments/joint_identity_v4/joint_identity_v4_seed_3409_lr_1e4_oracle_a100.yaml`。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_28e1833e0179/Matcher`。
- 动态命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" --plan 'configs/experiments/joint_identity_v4/joint_identity_v4_seed_3409_lr_1e4_oracle_a100.yaml'`。
- launch：`train_experiment_job340897_20260815T122606_a5`，创建于 2026-08-15 12:26:06（Asia/Shanghai）。
- W&B：entity `pencounkdual-111`，project `AdaLigand-Matcher-JointIdentityV4`，group `joint-identity-v4-learning-rate-seed-3409`，run `7n2y1aek`。
- 运行根：`/storage/penghongen/tmp/Matcher/experiments/joint_identity_v4_seed_3409_lr_1e4_oracle_a100/train_experiment_job340897_20260815T122606_a5/`。
- 状态：双独立终审 PASS 后受保护触发；任务自然早停时恢复 `try_lock_340897`。该 allocation 随后已用于摘要拼接任务，当前锁状态见相应摘要拼接单任务记录。
- 13:31 进度：epoch 0 selection 为 `0.6086108130896042`，validation `sample_count=6495`、`identity_eligible_count=1466`；学习率仍为 `1e-4`，任务继续运行。
- 15:02 进度：epoch 1–3 selection 依次为 `0.6496927126605520/0.6495555672571953/0.6810926252573344`。epoch 2 的坏验证轮把下一轮学习率降为 `5e-5`，epoch 3 在该学习率下刷新当前 BEST。Job 仍为 `RUNNING`，只有 `after_lock_340897`，没有 `try/pre/kill` 锁，尚未自然早停。
- 15:35 进度：epoch 4 selection 为 `0.6725549412187860`，坏验证轮数为 1，下一轮学习率降为 `2.5e-5`；当前 BEST 仍是 epoch 3 的 `0.6810926252573344`。Job 继续 `RUNNING`，仅保留 `after_lock_340897`，尚未自然早停。
- 最终结果：BEST/last epoch 为 `5/7`，global step 为 `3576`，BEST selection score 为 `0.6927067815952292`。阶段与运行根 summary、BEST、LAST 均已写出；本项是 seed 3409 的 `1e-4` 配对基线。
