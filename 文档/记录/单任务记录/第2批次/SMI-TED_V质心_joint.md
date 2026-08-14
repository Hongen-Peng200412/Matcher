# 第 2 批 SMI-TED V 质心 joint

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；mean/max；joint。
- 配置：`configs/experiments/batch2/full_smi_ted_v_centroid_joint_a100.yaml`。
- 直接对照：第 1 批 SMI-TED 五类 joint；只改变局部视野与四项采样统计。
- 预定资源：受管 Job `340899`，单张 A100，16 CPU；物理批量 8、梯度累积 4、global batch 32。
- W&B：project `AdaLigand-Matcher-Batch2`，group `matcher-batch2-full-v-centroid-sampling`。
- 控制端触发：2026-08-14 07:33:00（Asia/Shanghai）。先原子安装下列动态命令，再只删除 `try_lock_340899`；`after_lock_340899` 保留。

```bash
rm -- /home/penghongen/Feedback/Matcher/allocations/try_lock_340899
```

```bash
#!/usr/bin/env bash
set -euo pipefail

exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" \
    --plan configs/experiments/batch2/full_smi_ted_v_centroid_joint_a100.yaml
```

- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_3cc75993d69e/Matcher`。
- launch：`/home/penghongen/Feedback/Matcher/launches/340899/train_experiment_job340899_20260814T073303_a2`。
- W&B：run ID `s3lj6298`；[在线曲线](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch2/runs/s3lj6298)。
- 首步验收：global batch 32；有效身份目标 7；计算 `24.4089 s`；裁剪前梯度范数 `25.6375`；峰值已分配/保留显存 `14,195,192,320 / 19,442,696,192 bytes`。参数更新与 CUDA 同步均已完成，未见 OOM、NaN 或 Traceback。
- 2026-08-14 09:52:15 的 epoch 3 validation 刷新当前最好值：selection score `0.6543584471`，六分类 macro-F1 `0.4532580851`，五类固有 Top-1 `0.8437926330`，精确子类条件 Top-1 `0.9113082040`，宽松五类端到端 F1 `0.5985090522`，小分子兼容端到端 F1 `0.5632183908`，有效身份数 `1,466`。该轮仍为改善轮，不据此提前判定相对第 1 批的方向。
- 状态：训练中；最终 validation 与第 1 批 joint 头对头结果待自然收口后追加。
