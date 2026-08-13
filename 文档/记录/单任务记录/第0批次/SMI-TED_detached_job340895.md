# 第 0 批单实验：SMI-TED detached

- 提交命令：`bash 训练与运行/submit_task.sh --sh 训练与运行/sh/train_experiment.sh --resource a800 --gpus 1 --cpus 16 --partition nvlink --qos cpu96 --pre_hold --after_hold --job-name matcher-smi-detached -- --plan configs/experiments/initial/full_scratch_smi_ted_light_289m_detached.yaml`
- Slurm Job：`340895`；A800、16 CPU、无时间上限，`after_lock_340895` 保留。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_16b0873c6884/Matcher`
- launch：`/home/penghongen/Feedback/Matcher/launches/340895/train_experiment_job340895_20260813T121932_a1`
- 配置差异：SMI-TED Light 289M、small_molecule-only、mean/max、detached；旧 release 使用 16 个持久工作进程。
- W&B：[pki0o1zr](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/pki0o1zr)
- 结果：BEST/last epoch `9/11`；selection score `0.6696320918`；六分类 macro-F1 `0.5283099092`；条件身份 Top-1 `0.8034934498`；端到端身份 F1 `0.6770929163`。
- 分析：第 0 批第一名，但只比 oracle 高 `0.0041969818`，且 SMI-TED joint 缺失；不能解释为稳定的损失排序。首次 validation 的 I/O 异常慢属于旧 loader 实现，不计为科学性能。

