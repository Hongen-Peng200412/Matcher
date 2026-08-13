# 第 0 批单实验：MoLFormer detached

- 提交命令：`bash 训练与运行/submit_task.sh --sh 训练与运行/sh/train_experiment.sh --resource a100 --gpus 1 --cpus 16 --partition a100 --qos a100g4 --pre_hold --after_hold --job-name matcher-molf-detached -- --plan configs/experiments/initial/full_scratch_molformer_detached.yaml`
- Slurm Job：`340898`；A100、16 CPU、无时间上限，`after_lock_340898` 保留。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_95a9659383c2/Matcher`
- launch：`/home/penghongen/Feedback/Matcher/launches/340898/train_experiment_job340898_20260813T180543_a1`
- 配置差异：MoLFormer、small_molecule-only、mean/max、detached；物理批量 8、累积 4。
- W&B：[0lam2vpk](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/0lam2vpk)
- 首步：计算 `24.8004 s`，有效身份目标 5，裁剪前梯度范数 `29.4950`，峰值已分配/保留显存 `14192313856/19457376256 bytes`。
- 结果：BEST/last epoch `7/9`；selection score `0.6398127499`；六分类 macro-F1 `0.5243064395`；条件身份 Top-1 `0.7648401826`；端到端身份 F1 `0.6302916275`。
- 分析：第 0 批五个完整 Matcher 中最低，但这是单随机种子 validation 结果，不作为 MoLFormer 的一般性结论。

