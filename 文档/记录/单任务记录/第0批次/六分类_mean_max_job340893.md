# 第 0 批单实验：六分类 mean/max

- 提交命令：`bash 训练与运行/submit_task.sh --sh 训练与运行/sh/train_experiment.sh --resource a100 --gpus 1 --cpus 16 --partition a100 --qos a100g4 --pre_hold --after_hold --job-name matcher-pretrain-pair -- --plan configs/experiments/initial/candidate_pretrain_pair.yaml`
- Slurm Job：`340893`；A100、16 CPU、无时间上限，`after_lock_340893` 保留。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_95a9659383c2/Matcher`
- launch：`/home/penghongen/Feedback/Matcher/launches/340893/train_experiment_job340893_20260813T175925_a1`
- 配置差异：该 Job 的第一个 stage；mean/max、候选六分类、物理批量 8、累积 4、global batch 32。
- W&B：[mxornctd](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/mxornctd)
- 首步：计算 `23.8973 s`，裁剪前梯度范数 `28.1286`，峰值已分配/保留显存 `14154330112/19363004416 bytes`。
- 结果：BEST epoch `1`，last epoch `3`，validation macro-F1 `0.5304244533`。
- 分析：第 0 批两个六分类聚合中分数较高；作为后续弱聚合和稳定性基线，不代表差值已经过多随机种子验证。

