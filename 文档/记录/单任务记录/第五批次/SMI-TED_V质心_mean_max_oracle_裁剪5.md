# 第 5 批 SMI-TED V 质心 mean/max oracle 裁剪阈值 5

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；V 质心 48³；mean/max；oracle。
- 配置：`configs/experiments/batch5/batch5_smi_ted_v_centroid_clip_5_oracle_a800.yaml`。
- 唯一科学变量：`gradient_clip_norm=5.0`；双门为零，学习率、优化器、损失和 global batch 与第 2 批 oracle 相同。
- 直接对照：同一 A800 资源适配的 `clip200` 是主对照；第 2 批 A100 `clip1` 只作次级参考，因为物理批量由 `8×4` 改为 `16×2`，不宣称逐位同轨。
- 动机：第 2 批 oracle 的裁剪前梯度范数 P50 为 `6.5609`，阈值 `1.0` 在 `99.93%` 更新中生效；阈值 `5.0` 仍保留尾部保护，同时把中位更新缩放系数从约 `0.152` 放宽到约 `0.762`。
- 判断方式：比较自然早停 BEST 的全部核心 validation 指标，并报告总更新数、裁剪触发数/比例、最大裁剪前范数和最小缩放系数；失败或非有限值本身也是结果，不无记录重启。
- 实际资源：受管 A800 Job `340894`；16 CPU；物理批量 16、梯度累积 2、global batch 32；既有 allocation 的调度器身份时限只作核对，本任务不设置或缩短训练时限。
- W&B：online project `AdaLigand-Matcher-Batch5`；group `matcher-batch5-gradient-clipping`；run [`bh2s3lm7`](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch5/runs/bh2s3lm7)。
- 状态：已于 2026-08-15 08:14:39（Asia/Shanghai）自然早停；未读取 calibration。

## 提交与结果留证

- 实现提交为 `5548a1021729ac514896cb41a54bfd599cf26517`；等价学习端点为 `62c59f299612f4752f84aafec45916a2ceef602d`；内容寻址 release 为 `Matcher_d14dd1c3cee4`。
- 控制端命令：`& "$env:USERPROFILE\.codex\tools\Invoke-ProjectSsh.ps1" -Command 'bash -s' -InputFile 'C:\Users\15919\Desktop\Matcher\tmp\batch5_launch\activate_batch5_matrix.sh'`。双审通过的一次性控制器摘要为 `ce909d15990207f46387e1c299dffbcdaff25a9958549148d218896d015d8a3e`。
- 动态训练命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" --plan 'configs/experiments/batch5/batch5_smi_ted_v_centroid_clip_5_oracle_a800.yaml'`；命令先要求 `TASK_PROJECT_ROOT=/home/penghongen/Feedback/Matcher/releases/Matcher_d14dd1c3cee4/Matcher`。
- launch：`train_experiment_job340894_20260815T032512_a10`；运行根：`/storage/penghongen/tmp/Matcher/experiments/batch5_smi_ted_v_centroid_clip_5_oracle_a800/train_experiment_job340894_20260815T032512_a10/`。
- 锁边界：控制器只删除 `try_lock_340894`；`after_lock_340894` 保留，`pre_lock_340894/kill_lock_340894` 均不存在。没有取消、重启或释放 allocation。
- 首个 global batch 已完成参数更新：global batch 32、身份有效候选数 7（`identity_eligible_count=7`）；classification/ligand/binding/identity/total loss 为 `1.8027343750/1.7515190840/1.6761010885/1.5334821258/3.6789785181`，裁剪前梯度范数 `46.1340217590`。数据等待/计算为 `9.4375751466/62.1503695659 s`，吞吐 `0.4470026361 candidate/s`，显存峰值 allocated/reserved 为 `28,143,181,312/39,608,909,824` bytes；首次更新后 graph/inventory 门值均约 `0.0001`。
- BEST/last epoch 为 `8/10`，global step 为 `4917`，selection score 为 `0.6908370762`；运行根与阶段 `summary.json` 均已正常写出，runner 自然恢复 `try_lock_340894`，`after_lock_340894` 保留。
- BEST epoch 8 的六分类 macro-F1、五类固有 Top-1、精确子类条件 Top-1、宽松五类端到端 F1、small-molecule 兼容端到端 F1、ligand Dice 与 binding Dice 依次为 `0.5250978501/0.8683492497/0.8813852814/0.6660280971/0.6627680312/0.2646142935/0.0588616072`。
- 相对第 2 批 A100 `clip1` 次级基线，上述 selection 与七项指标分别变化 `-1.041%/-0.646%/-0.857%/-0.959%/-1.458%/-3.029%/-16.573%/-41.929%`。该比较同时包含 A100 `8×4` 与 A800 `16×2` 的已声明资源差异，不能单独归因为阈值 5。
- 同一 A800 的 `clip200` 已自然早停，BEST selection score 为 `0.6960835398398743`，比本任务高 `0.0052464636594069`，即 `0.5246463659` 个百分点、相对高 `0.7594357397%`。该单 seed 主对照对放宽裁剪有利，但差异不足 1%，不能写成显著或普遍优势；`clip200` 的零触发证据见对应单任务记录。
