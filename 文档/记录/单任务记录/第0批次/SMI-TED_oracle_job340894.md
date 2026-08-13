# 第 0 批单实验：SMI-TED oracle

- 提交命令：`bash 训练与运行/submit_task.sh --sh 训练与运行/sh/train_experiment.sh --resource a800 --gpus 1 --cpus 16 --partition nvlink --qos cpu96 --pre_hold --after_hold --job-name matcher-smi-oracle -- --plan configs/experiments/initial/full_scratch_smi_ted_light_289m_oracle.yaml`
- Slurm Job：`340894`；A800、16 CPU、无时间上限，`after_lock_340894` 保留。
- 最终 release：`/home/penghongen/Feedback/Matcher/releases/Matcher_67c4e42bd46a/Matcher`
- 最终 launch：`/home/penghongen/Feedback/Matcher/launches/340894/train_experiment_job340894_20260813T165002_a5`
- 配置差异：SMI-TED Light 289M、small_molecule-only、mean/max、oracle；物理批量 16、累积 2。
- W&B：[aiqbj6b5](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/aiqbj6b5)
- 首步：计算 `62.2150 s`，有效身份目标 5，裁剪前梯度范数 `39.9681`，峰值已分配/保留显存 `28140193792/39564869632 bytes`。
- 结果：BEST/last epoch `7/9`；selection score `0.6654351100`；六分类 macro-F1 `0.5254060463`；条件身份 Top-1 `0.8093023256`；端到端身份 F1 `0.6615969582`。
- 分析：第 0 批完整 Matcher 第二名，是第 1 批 oracle 的直接 small_molecule 口径基线。

