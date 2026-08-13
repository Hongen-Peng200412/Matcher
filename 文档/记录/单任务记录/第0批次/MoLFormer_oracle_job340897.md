# 第 0 批单实验：MoLFormer oracle

- 提交命令：`bash 训练与运行/submit_task.sh --sh 训练与运行/sh/train_experiment.sh --resource a100 --gpus 1 --cpus 16 --partition a100 --qos a100g4 --pre_hold --after_hold --job-name matcher-molf-oracle -- --plan configs/experiments/initial/full_scratch_molformer_oracle.yaml`
- Slurm Job：`340897`；A100、16 CPU、无时间上限，`after_lock_340897` 保留。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_95a9659383c2/Matcher`
- launch：`/home/penghongen/Feedback/Matcher/launches/340897/train_experiment_job340897_20260813T175915_a1`
- 配置差异：MoLFormer、small_molecule-only、mean/max、oracle；物理批量 8、累积 4。
- W&B：[mkaiyufz](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/mkaiyufz)
- 首步：计算 `25.0321 s`，有效身份目标 5，裁剪前梯度范数 `53.1809`，峰值已分配/保留显存 `14190904832/19526582272 bytes`。
- 结果：BEST/last epoch `7/9`；selection score `0.6539754941`；六分类 macro-F1 `0.5239893710`；条件身份 Top-1 `0.7850678733`；端到端身份 F1 `0.6528692380`。
- 分析：低于 SMI-TED oracle，结合五类覆盖审计后不作为第 1 批首选语言模型。

