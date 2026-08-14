# 第 2 批 SMI-TED V 质心 oracle

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；mean/max；oracle。
- 配置：`configs/experiments/batch2/full_smi_ted_v_centroid_oracle_a100.yaml`。
- 直接对照：第 1 批 SMI-TED 五类 oracle；只改变局部视野与四项采样统计。
- 预定资源：受管 Job `340897`，单张 A100，16 CPU；物理批量 8、梯度累积 4、global batch 32。
- W&B：project `AdaLigand-Matcher-Batch2`，group `matcher-batch2-full-v-centroid-sampling`。
- 控制端触发：2026-08-14 07:33:00（Asia/Shanghai）。先原子安装下列动态命令，再只删除 `try_lock_340897`；`after_lock_340897` 保留。

```bash
rm -- /home/penghongen/Feedback/Matcher/allocations/try_lock_340897
```

```bash
#!/usr/bin/env bash
set -euo pipefail

exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" \
    --plan configs/experiments/batch2/full_smi_ted_v_centroid_oracle_a100.yaml
```

- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_3cc75993d69e/Matcher`。
- launch：`/home/penghongen/Feedback/Matcher/launches/340897/train_experiment_job340897_20260814T073305_a2`。
- W&B：run ID `6soiol5e`；[在线曲线](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch2/runs/6soiol5e)。
- 首步验收：global batch 32；有效身份目标 7；计算 `25.8580 s`；裁剪前梯度范数 `42.1343`；峰值已分配/保留显存 `14,194,926,080 / 19,610,468,352 bytes`。参数更新与 CUDA 同步均已完成，未见 OOM、NaN 或 Traceback。
- 2026-08-14 09:52:12 的 epoch 3 validation 刷新当前最好值：selection score `0.6624755578`，六分类 macro-F1 `0.5232567136`，五类固有 Top-1 `0.8253751705`，精确子类条件 Top-1 `0.8356807512`，宽松五类端到端 F1 `0.6284892086`，小分子兼容端到端 F1 `0.5639639640`，有效身份数 `1,466`。该轮仍为改善轮，不据此提前判定相对第 1 批的方向。
- 2026-08-14 10:26:37 的 epoch 4 validation 再次改善：selection score `0.6699438211`，六分类 macro-F1 `0.5248891045`，五类固有 Top-1 `0.8349249659`，精确子类条件 Top-1 `0.8470279720`，宽松五类端到端 F1 `0.6379143869`，小分子兼容端到端 F1 `0.6452241715`。该 BEST 相对第 1 批 oracle 最终 BEST 低约 `3.29%`，未达到多指标共同约 10% 的负优化判断尺度；任务继续运行。
## 最终 validation 与自然收口

2026-08-14 15:45:19（Asia/Shanghai），训练命令正常发布运行根 `summary.json`，在 epoch 13 后因连续两个坏验证轮自然早停；受管 Slurm allocation 继续由 `after_lock_340897` 保留。BEST 为 epoch 11：selection score `0.6981048482`，六分类 macro-F1 `0.5285113541`，五类固有 Top-1 `0.8758526603`，精确子类条件 Top-1 `0.8899237934`，宽松五类端到端 F1 `0.6758793970`，小分子兼容端到端 F1 `0.6834733894`，ligand/binding micro-Dice 为 `0.3171820106/0.1013630926`；有效身份监督数为 `1,466`。

| 真实配体类别 | 固有 Top-1（监督数） | 精确子类条件 Top-1（监督数） | 宽松端到端 F1（实际阳性数） |
|---|---|---|---|
| ion | `0.9913194444 (576)` | `0.9894291755 (473)` | `0.7059701493 (576)` |
| nucleotide_like | `1.0000000000 (3)` | `null (0)` | `0.8571428571 (3)` |
| peptide_like | `0.8750000000 (8)` | `null (0)` | `0.2372881356 (8)` |
| small_molecule | `0.7651376147 (545)` | `0.8098434004 (447)` | `0.6848541863 (545)` |
| sugar | `0.8562874251 (334)` | `0.8467432950 (261)` | `0.6405594406 (334)` |

与第 1 批 oracle 相比，selection score 从 `0.6927506901` 升到 `0.6981048482`；六分类 macro-F1、五类固有 Top-1、宽松端到端 F1 和小分子兼容端到端 F1 均上升。V 质心视野在本对照中没有出现多个核心指标约 10% 的共同下降，且区域指标明显改善，因此支持后续批次继续采用新采样。

运行根：`/storage/penghongen/tmp/Matcher/experiments/batch2_smi_ted_v_centroid_oracle_a100/train_experiment_job340897_20260814T073305_a2/summary.json`。
