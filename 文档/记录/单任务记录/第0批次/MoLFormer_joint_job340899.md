# 第 0 批单实验：MoLFormer joint

- 提交命令：`bash 训练与运行/submit_task.sh --sh 训练与运行/sh/train_experiment.sh --resource a100 --gpus 1 --cpus 16 --partition a100 --qos a100g4 --pre_hold --after_hold --job-name matcher-molf-joint -- --plan configs/experiments/initial/full_scratch_molformer_joint.yaml`
- Slurm Job：`340899`；A100、16 CPU、无时间上限，`after_lock_340899` 保留。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_95a9659383c2/Matcher`
- launch：`/home/penghongen/Feedback/Matcher/launches/340899/train_experiment_job340899_20260813T180559_a1`
- 配置差异：MoLFormer、small_molecule-only、mean/max、joint；物理批量 8、累积 4。
- W&B：[1z2vvfxj](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/1z2vvfxj)
- 首步：计算 `22.2045 s`，有效身份目标 5，裁剪前梯度范数 `29.3475`，峰值已分配/保留显存 `14192313856/19457376256 bytes`。
- 结果：BEST/last epoch `7/9`；selection score `0.6618542906`；六分类 macro-F1 `0.5129085101`；条件身份 Top-1 `0.8405405405`；端到端身份 F1 `0.6321138211`。
- 分析：第 0 批第三名且是已完成任务中唯一 joint 基线；SMI-TED joint 缺失，不能把本结果当作跨语言 joint 对照。

