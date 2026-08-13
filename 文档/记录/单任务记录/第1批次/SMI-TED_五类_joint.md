# 第 1 批单实验：SMI-TED 五类 joint

- 预定配置：`configs/experiments/batch1/full_smi_ted_all_foreground_joint_a100.yaml`
- 预定资源：受管 Job `340893` 的 1×A100；保留 `after_lock_340893`。
- 个性化变量：五类非背景身份、12 维固化手动特征、mean/max、joint、物理批量 8 × 累积 4。
- 直接对照：第 0 批没有 SMI-TED joint；只与本批 oracle/detached 比较，不虚构缺失基线。
- Slurm allocation：复用 Job `340893` 的 A100；本批不新建 `sbatch`，只在 `try_lock_340893` 存在且 `after_lock_340893` 保留时原子替换动态命令。
- 动态命令 SHA-256：`483412ff7af602feb3f8d47e480a36afaeea978d4c2f7ecf68ef1fd33a7b32db`。

```bash
#!/usr/bin/env bash
set -euo pipefail

exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" \
    --plan configs/experiments/batch1/full_smi_ted_all_foreground_joint_a100.yaml
```

- 触发：2026-08-14 04:18:48（Asia/Shanghai）只删除 `/home/penghongen/Feedback/Matcher/allocations/try_lock_340893`；`after_lock_340893` 保留。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_4cf3ef45e8a8/Matcher`。
- launch：`/home/penghongen/Feedback/Matcher/launches/340893/train_experiment_job340893_20260814T041902_a2`。
- W&B：run ID `amdxqwjc`；[在线曲线](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch1/runs/amdxqwjc)。project 为 `AdaLigand-Matcher-Batch1`，group 为 `matcher-batch1-five-class-manual-features`。
- 首步验收：global batch 32；有效身份目标 7；计算 `24.2634 s`；裁剪前梯度范数 `30.3424`；峰值已分配/保留显存 `14,195,499,008 / 19,511,902,208 bytes`。首步已完成有限性检查、参数更新和 CUDA 同步；未见 OOM、NaN 或 Traceback。
- 2026-08-14 06:54 读取完整 validation 记录：epoch 3 的 selection score 改善为 `0.6452062205`，坏验证轮数为 0，学习率为 `5e-5`。六分类 macro-F1、五类条件 Top-1、五类端到端 F1 和小分子兼容端到端 F1 分别为 `0.4506733/0.8943270/0.5906183/0.5346985`；当前仍在改善，不因与两项 A800 任务的单点差距提前判负。
- 最终结果：训练中。第 0 批没有对应 SMI-TED joint，最终只做批内 oracle/detached/joint 比较，不虚构基线。
