# 第 1 批单实验：SMI-TED 五类 detached

- 预定配置：`configs/experiments/batch1/full_smi_ted_all_foreground_detached_a800.yaml`
- 预定资源：受管 Job `340895` 的 1×A800；保留 `after_lock_340895`。
- 个性化变量：五类非背景身份、12 维固化手动特征、mean/max、detached、物理批量 16 × 累积 2。
- 直接对照：第 0 批 `SMI-TED detached / Job 340895` 的共同 small_molecule validation 指标。
- Slurm allocation：复用 Job `340895` 的 A800；本批不新建 `sbatch`。放行前原子覆盖曾用于已终止 calibration 准备的动态命令，未恢复或执行 calibration。
- 动态命令 SHA-256：`87fd62a9318b765f8fd291ad5e6e297dbe858d303791c9dc7bbe972cd039edc6`。

```bash
#!/usr/bin/env bash
set -euo pipefail

exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" \
    --plan configs/experiments/batch1/full_smi_ted_all_foreground_detached_a800.yaml
```

- 触发：2026-08-14 04:18:48（Asia/Shanghai）只删除 `/home/penghongen/Feedback/Matcher/allocations/try_lock_340895`；`after_lock_340895` 保留。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_4cf3ef45e8a8/Matcher`。
- launch：`/home/penghongen/Feedback/Matcher/launches/340895/train_experiment_job340895_20260814T041633_a3`。launch 名中的节点时间早于控制端触发时间，属于节点时钟偏差；attempt 3 与 release、动态命令和业务目录的身份一致。
- W&B：run ID `vry3gxt2`；[在线曲线](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch1/runs/vry3gxt2)。project 为 `AdaLigand-Matcher-Batch1`，group 为 `matcher-batch1-five-class-manual-features`。
- 首步验收：global batch 32；有效身份目标 7；计算 `62.6836 s`；裁剪前梯度范数 `32.3384`；峰值已分配/保留显存 `28,143,814,144 / 39,522,926,592 bytes`。首步已完成有限性检查、参数更新和 CUDA 同步；未见 OOM、NaN 或 Traceback。
- 2026-08-14 06:54 读取完整 validation 记录：epoch 4 的 selection score 改善为 `0.6756995547`，坏验证轮数为 0，学习率继续为 `1e-4`。六分类 macro-F1、五类条件 Top-1、五类端到端 F1 和小分子兼容端到端 F1 分别为 `0.5327544/0.8462852/0.6480590/0.6321627`；任务继续运行。
- 最终结果：训练中；只在新 epoch、validation、失败或自然收口时补充。
