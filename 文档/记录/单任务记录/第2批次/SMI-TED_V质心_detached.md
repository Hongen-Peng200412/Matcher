# 第 2 批 SMI-TED V 质心 detached

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；mean/max；detached。
- 配置：`configs/experiments/batch2/full_smi_ted_v_centroid_detached_a100.yaml`。
- 直接对照：第 1 批 SMI-TED 五类 detached；只改变局部视野与四项采样统计。
- 预定资源：受管 Job `340898`，单张 A100，16 CPU；物理批量 8、梯度累积 4、global batch 32。
- W&B：project `AdaLigand-Matcher-Batch2`，group `matcher-batch2-full-v-centroid-sampling`。
- 控制端触发：2026-08-14 07:33:00（Asia/Shanghai）。先原子安装下列动态命令，再只删除 `try_lock_340898`；`after_lock_340898` 保留。

```bash
rm -- /home/penghongen/Feedback/Matcher/allocations/try_lock_340898
```

```bash
#!/usr/bin/env bash
set -euo pipefail

exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" \
    --plan configs/experiments/batch2/full_smi_ted_v_centroid_detached_a100.yaml
```

- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_3cc75993d69e/Matcher`。
- launch：`/home/penghongen/Feedback/Matcher/launches/340898/train_experiment_job340898_20260814T073316_a2`。
- W&B：run ID `i0j1i1hc`；[在线曲线](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch2/runs/i0j1i1hc)。
- 首步验收：global batch 32；有效身份目标 7；计算 `25.0679 s`；裁剪前梯度范数 `26.4270`；峰值已分配/保留显存 `14,195,192,320 / 19,440,599,040 bytes`。参数更新与 CUDA 同步均已完成，未见 OOM、NaN 或 Traceback。
- 2026-08-14 09:52:13 的 epoch 3 validation 刷新当前最好值：selection score `0.6739567888`，六分类 macro-F1 `0.5284146987`，五类固有 Top-1 `0.8431105048`，精确子类条件 Top-1 `0.8489984592`，宽松五类端到端 F1 `0.6444572085`，小分子兼容端到端 F1 `0.6089965398`，有效身份数 `1,466`。该轮仍为改善轮，不据此提前判定相对第 1 批的方向。
- 2026-08-14 10:26:47 的 epoch 4 validation 为 `0.6594240440`，未刷新 epoch 3 BEST；六分类 macro-F1 `0.5264336859`、五类固有 Top-1 `0.8294679400`、精确子类条件 Top-1 `0.8269230769`、宽松五类端到端 F1 `0.6249153690`、小分子兼容端到端 F1 `0.5901639344`。坏验证轮数为 1，下一轮学习率 `5e-5`；epoch 3 BEST 相对第 1 批 detached 最终 BEST 低约 `2.82%`，继续运行。
- 状态：训练中；最终 validation 与第 1 批 detached 头对头结果待自然收口后追加。
