# 第 1 批单实验：SMI-TED 五类 oracle

- 预定配置：`configs/experiments/batch1/full_smi_ted_all_foreground_oracle_a800.yaml`
- 预定资源：受管 Job `340894` 的 1×A800；保留 `after_lock_340894`。
- 个性化变量：五类非背景身份、12 维固化手动特征、mean/max、oracle、物理批量 16 × 累积 2。
- 直接对照：第 0 批 `SMI-TED oracle / Job 340894` 的共同 small_molecule validation 指标。
- Slurm allocation：复用 Job `340894` 的 A800；本批不新建 `sbatch`，只在 `try_lock_340894` 存在且 `after_lock_340894` 保留时原子替换动态命令。
- 动态命令 SHA-256：`91b8825f927a177dcc4416782a2ec28f58384e01f856d137bfd8f06ca1096588`。

```bash
#!/usr/bin/env bash
set -euo pipefail

exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" \
    --plan configs/experiments/batch1/full_smi_ted_all_foreground_oracle_a800.yaml
```

- 触发：2026-08-14 04:18:48（Asia/Shanghai）只删除 `/home/penghongen/Feedback/Matcher/allocations/try_lock_340894`；`after_lock_340894` 保留。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_4cf3ef45e8a8/Matcher`。
- launch：`/home/penghongen/Feedback/Matcher/launches/340894/train_experiment_job340894_20260814T041802_a6`。
- W&B：run ID `h41na5ea`；[在线曲线](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch1/runs/h41na5ea)。project 为 `AdaLigand-Matcher-Batch1`，group 为 `matcher-batch1-five-class-manual-features`。
- 首步验收：global batch 32；有效身份目标 7；计算 `61.7351 s`；裁剪前梯度范数 `49.9738`；峰值已分配/保留显存 `28,143,265,792 / 39,747,321,856 bytes`。首步已完成有限性检查、参数更新和 CUDA 同步；未见 OOM、NaN 或 Traceback。
- 最终结果：训练中；只在新 epoch、validation、失败或自然收口时补充。
