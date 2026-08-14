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
- 2026-08-14 10:26:38 的 epoch 4 validation 明显收复六分类并刷新 BEST：selection score `0.6840046283`，六分类 macro-F1 `0.5056822360`，五类固有 Top-1 `0.8362892224`，精确子类条件 Top-1 `0.9014598540`，宽松五类端到端 F1 `0.6448717949`，小分子兼容端到端 F1 `0.6302521008`。该 BEST 相对第 1 批 joint 当前 BEST 仅低约 `1.28%`；任务继续运行。
## 最终 validation 与自然收口

2026-08-14 12:44:44（Asia/Shanghai），训练命令正常发布运行根 `summary.json`，在 epoch 8 后因连续两个坏验证轮自然早停；同一 allocation 随后按既定优先级接续第 3 批任务，`after_lock_340899` 始终保留。BEST 为 epoch 6：selection score `0.6887556119`，六分类 macro-F1 `0.5097478210`，五类固有 Top-1 `0.8560709413`，精确子类条件 Top-1 `0.9019980971`，宽松五类端到端 F1 `0.6545209177`，小分子兼容端到端 F1 `0.6367346939`，ligand/binding micro-Dice 为 `0.3019974340/0.0669487746`；有效身份监督数为 `1,466`。

| 真实配体类别 | 固有 Top-1（监督数） | 精确子类条件 Top-1（监督数） | 宽松端到端 F1（实际阳性数） |
|---|---|---|---|
| ion | `0.9878472222 (576)` | `0.9845132743 (452)` | `0.7205764612 (576)` |
| nucleotide_like | `0.3333333333 (3)` | `null (0)` | `0.0000000000 (3)` |
| peptide_like | `0.8750000000 (8)` | `null (0)` | `0.1590909091 (8)` |
| small_molecule | `0.7082568807 (545)` | `0.8201058201 (378)` | `0.6358974359 (545)` |
| sugar | `0.8742514970 (334)` | `0.8733031674 (221)` | `0.6255778120 (334)` |

相对第 1 批 joint，selection score、macro-F1、精确子类条件 Top-1 和宽松端到端 F1均只下降约 `0.6%–1.1%`，五类固有 Top-1反而上升约 `0.8%`；小分子兼容端到端 F1下降约 `2.9%`。binding Dice 单项下降较大，但没有多个核心指标共同下降约 10%，因此不判定新采样明显有害。

运行根：`/storage/penghongen/tmp/Matcher/experiments/batch2_smi_ted_v_centroid_joint_a100/train_experiment_job340899_20260814T073303_a2/summary.json`。
