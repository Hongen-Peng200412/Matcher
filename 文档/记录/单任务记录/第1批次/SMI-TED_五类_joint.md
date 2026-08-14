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
- 2026-08-14 09:53:47 完成 epoch 9 validation 并再次改善：selection score `0.6928435444`，六分类 macro-F1 `0.5193692059`，五类固有 Top-1 `0.8526603001`，精确子类条件 Top-1 `0.8971229294`，宽松五类端到端 F1 `0.6620384979`，小分子兼容端到端 F1 `0.6646646647`，有效身份数 `1,466`。`summary.json` 尚未出现，任务继续运行。
- 2026-08-14 10:28:40 完成 epoch 10 validation：selection score `0.6841031473`，比 epoch 9 BEST 低约 `1.26%`；六分类 macro-F1 `0.5034409102`、五类固有 Top-1 `0.8512960437`、精确子类条件 Top-1 `0.9029702970`、宽松五类端到端 F1 `0.6458982347`、小分子兼容端到端 F1 `0.6645962733`。坏验证轮数为 1，下一轮学习率 `1.25e-5`；`summary.json` 仍不存在，继续运行，不因单轮小幅回落提前判停。
## 最终 validation 与自然收口

2026-08-14 12:14:17（Asia/Shanghai），训练命令正常发布运行根 `summary.json`，在 epoch 13 后因连续两个坏验证轮自然早停；同一 allocation 随后按既定顺序接续第 3 批 oracle，`after_lock_340893` 始终保留。BEST 为 epoch 11：selection score `0.6945667425`，六分类 macro-F1 `0.5145892054`，五类固有 Top-1 `0.8492496589`，精确子类条件 Top-1 `0.9076782450`，宽松五类端到端 F1 `0.6614327772`，小分子兼容端到端 F1 `0.6556016598`，ligand/binding micro-Dice 为 `0.3179151126/0.1132438254`；有效身份监督数为 `1,466`。

| 真实配体类别 | 固有 Top-1（监督数） | 精确子类条件 Top-1（监督数） | 宽松端到端 F1（实际阳性数） |
|---|---|---|---|
| ion | `0.9878472222 (576)` | `0.9876288660 (485)` | `0.7628053586 (576)` |
| nucleotide_like | `0.3333333333 (3)` | `null (0)` | `0.5000000000 (3)` |
| peptide_like | `0.2500000000 (8)` | `null (0)` | `0.0287769784 (8)` |
| small_molecule | `0.7266055046 (545)` | `0.8501362398 (367)` | `0.6486486486 (545)` |
| sugar | `0.8293413174 (334)` | `0.8347107438 (242)` | `0.6207906296 (334)` |

第 0 批没有对应 SMI-TED joint，因此不虚构基线。批内比较中，joint 的 selection score `0.6945667425` 略高于 detached/oracle 的 `0.6935109378/0.6927506901`，但三者差距均小于 `0.3%`，且六分类、身份和区域指标互有胜负；单随机种子不足以宣称损失排序。

运行根：`/storage/penghongen/tmp/Matcher/experiments/batch1_smi_ted_all_foreground_joint_a100/train_experiment_job340893_20260814T041902_a2/summary.json`。
