# 联合身份 v4：seed 3407、LR 1e-4

- 任务：三对学习率复现中的 A800 基线；与 Job 340895 构成同硬件类型、同训练随机种子的直接配对。
- Job：`340894`，A800，16 CPU。
- 配置：`configs/experiments/joint_identity_v4/joint_identity_v4_zero_gate_oracle_a100.yaml`。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_28e4bacd0dbc/Matcher`。
- 动态命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" --plan 'configs/experiments/joint_identity_v4/joint_identity_v4_zero_gate_oracle_a100.yaml'`。
- launch：`train_experiment_job340894_20260815T115031_a12`，创建于 2026-08-15 11:50:31（Asia/Shanghai）。
- W&B：entity `pencounkdual-111`，project `AdaLigand-Matcher-JointIdentityV4`，group `joint-identity-v4-zero-gate-baseline`，run `hvtzju50`。
- 运行根：`/storage/penghongen/tmp/Matcher/experiments/joint_identity_v4_zero_gate_oracle_a100/train_experiment_job340894_20260815T115031_a12/`。
- 状态：训练于 2026-08-15 15:42:59（Asia/Shanghai）自然早停；结束时 `try_lock_340894` 由 allocation runner 恢复，`after_lock_340894` 保留，pre/kill 锁不存在。该 allocation 随后已用于摘要拼接任务；当前锁状态见相应摘要拼接单任务记录。
- epoch 0：global step 447，selection score `0.6418496024938629`，validation `sample_count=6495`、`identity_eligible_count=1466`；训练继续，尚不据此判断学习率优劣。
- 证据边界：触发前 v3/v4 同卡 GPU 等价门已通过；validation 长度与 `sample_count` 均为 6495，`identity_eligible_count` 均为 1466，selection score 均为 `0.69798148434599`，固定前向与完整科学 validation 严格一致。
- 13:31 进度：epoch 0–2 selection 依次为 `0.6418496024938629/0.6370543404363079/0.6469126497476826`，当前 BEST 为 epoch 2。epoch 1 后调度器把下一轮学习率降为 `5e-5`，epoch 2 在该学习率下刷新 BEST；任务仍在运行。
- 15:02 进度：epoch 3–5 selection 依次为 `0.6758616800263981/0.6809085685414017/0.6920686526492751`，epoch 5 刷新当前 BEST；实际学习率保持 `5e-5`。Job 仍为 `RUNNING`，只有 `after_lock_340894`，没有 `try/pre/kill` 锁，尚未自然早停。
- 15:35 进度：epoch 6 selection 为 `0.6748063746596293`，坏验证轮数为 1，下一轮学习率降为 `2.5e-5`；当前 BEST 仍是 epoch 5 的 `0.6920686526492751`。Job 继续 `RUNNING`，仅保留 `after_lock_340894`，尚未自然早停。
- 最终结果：BEST/last epoch 为 `5/7`，BEST selection score 为 `0.6920686526492751`，`stopped_by_patience=true`。BEST epoch 的六分类 macro-F1 为 `0.5284033454997990`，固有身份 Top-1 为 `0.8656207366984994`，条件身份 Top-1 为 `0.8781275221953188`，宽松端到端 F1 为 `0.6696750902527075`；small-molecule 兼容口径的固有 Top-1、条件 Top-1、端到端 F1 分别为 `0.7669724770642202/0.7863436123348018/0.6629526462395543`，ligand/binding micro Dice 为 `0.2396188019041835/0.021941170508661578`。
- 完整性：BEST epoch validation 总损失为 `0.7647487736846721`，`sample_count=6495`、`identity_eligible_count=1466`；BEST、LAST、阶段 summary 和运行根 summary 均已写出。该项单独收口不形成学习率结论，仍须等待 Job 340895 的同硬件、同随机种子对照自然早停。
