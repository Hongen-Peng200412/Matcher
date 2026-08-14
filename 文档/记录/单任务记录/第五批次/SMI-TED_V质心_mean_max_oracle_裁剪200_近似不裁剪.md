# 第 5 批 SMI-TED V 质心 mean/max oracle 裁剪阈值 200

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；V 质心 48³；mean/max；oracle。
- 配置：`configs/experiments/batch5/batch5_smi_ted_v_centroid_clip_200_oracle_a800.yaml`。
- 唯一科学变量：`gradient_clip_norm=200.0`；双门为零，学习率、优化器、损失和 global batch 与第 2 批 oracle 相同。
- 直接对照：同一 A800 资源适配的 `clip5` 是主对照；第 2 批 A100 `clip1` 只作次级参考，因为物理批量由 `8×4` 改为 `16×2`，不宣称逐位同轨。
- 动机：历史 Batch2 oracle 的最高裁剪前梯度范数为 `182.85`；阈值 `200` 在既有逐步值上会零触发，同时仍为更大异常保留有限保护。只有实际触发率不超过 `0.1%` 才称“近似不裁剪”。
- 判断方式：比较自然早停 BEST 的全部核心 validation 指标，并报告总更新数、裁剪触发数/比例、最大裁剪前范数和最小缩放系数；若触发率超过 `0.1%`，记录中改称“宽松裁剪”。失败或非有限值本身也是结果，不无记录重启。
- 实际资源：受管 A800 Job `340895`；16 CPU；物理批量 16、梯度累积 2、global batch 32；既有 allocation 的调度器身份时限只作核对，本任务不设置或缩短训练时限。
- W&B：online project `AdaLigand-Matcher-Batch5`；group `matcher-batch5-gradient-clipping`；run [`ufny2evy`](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch5/runs/ufny2evy)。
- 状态：已提交并完成首个 global batch，继续训练；是否可称“近似不裁剪”仍待完整运行的实际触发率；未读取 calibration。

## 提交与结果留证

- 实现提交为 `5548a1021729ac514896cb41a54bfd599cf26517`；等价学习端点为 `62c59f299612f4752f84aafec45916a2ceef602d`；内容寻址 release 为 `Matcher_d14dd1c3cee4`。
- 控制端命令：`& "$env:USERPROFILE\.codex\tools\Invoke-ProjectSsh.ps1" -Command 'bash -s' -InputFile 'C:\Users\15919\Desktop\Matcher\tmp\batch5_launch\activate_batch5_matrix.sh'`。双审通过的一次性控制器摘要为 `ce909d15990207f46387e1c299dffbcdaff25a9958549148d218896d015d8a3e`。
- 动态训练命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" --plan 'configs/experiments/batch5/batch5_smi_ted_v_centroid_clip_200_oracle_a800.yaml'`；命令先要求 `TASK_PROJECT_ROOT=/home/penghongen/Feedback/Matcher/releases/Matcher_d14dd1c3cee4/Matcher`。
- launch：`train_experiment_job340895_20260815T032343_a6`；运行根：`/storage/penghongen/tmp/Matcher/experiments/batch5_smi_ted_v_centroid_clip_200_oracle_a800/train_experiment_job340895_20260815T032343_a6/`。
- 锁边界：控制器只删除 `try_lock_340895`；`after_lock_340895` 保留，`pre_lock_340895/kill_lock_340895` 均不存在。没有取消、重启或释放 allocation。
- 首个 global batch 已完成参数更新：global batch 32、身份有效候选数 7（`identity_eligible_count=7`）；classification/ligand/binding/identity/total loss 为 `1.8027343750/1.7515190840/1.6761010885/1.5334821258/3.6789785181`，裁剪前梯度范数 `46.0992240906`，低于阈值 200。数据等待/计算为 `10.4452389674/63.1941584488 s`，吞吐 `0.4345499980 candidate/s`，显存峰值 allocated/reserved 为 `28,143,181,312/39,495,663,616` bytes；首次更新后 graph/inventory 门值均约 `0.0001`。
- 首次完整 validation、BEST/last、完整裁剪触发率与最终头对头结论：待自然事件后回填。
