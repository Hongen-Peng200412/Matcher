# 第 5 批 SMI-TED V 质心 mean/max oracle 仅图门 0.02

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；V 质心 48³；mean/max；oracle。
- 配置：`configs/experiments/batch5/batch5_smi_ted_v_centroid_graph_gate_002_oracle_a100.yaml`。
- 唯一科学变量：`initial_graph_gate=0.02`；`initial_inventory_gate=0.0`，其它科学字段与第 2 批 oracle 相同。
- 直接对照：第 2 批零门 `00` 与第 5 批双门 `11`；它与仅清单门 `01` 共同补齐 2×2 因子矩阵。
- 动机：当前双门任务 epoch 3 的图增量/语言基线中心化 L2 比例约为 `0.2174`，清单增量约为 `0.0458`；只启用图门可分辨图分支主效应与两门交互，不能把该比例直接解释成有符号收益。
- 判断方式：使用自然早停 BEST 的 validation selection score、六分类、五类身份、small_molecule 兼容身份、区域指标和 `identity_contribution/validation/*`；单个指标或单轮波动不下结论。
- 实际资源：受管 A100 Job `340893`；16 CPU；物理批量 8、梯度累积 4、global batch 32；本任务不设置训练或 Slurm 时间上限。
- W&B：online project `AdaLigand-Matcher-Batch5`；group `matcher-batch5-gate-initialization`；run [`c78pugvk`](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch5/runs/c78pugvk)。
- 状态：已自然早停；BEST/last epoch 为 `8/10`，未读取 calibration。

## 提交与结果留证

- 实现提交为 `5548a1021729ac514896cb41a54bfd599cf26517`；等价学习端点为 `62c59f299612f4752f84aafec45916a2ceef602d`；内容寻址 release 为 `Matcher_d14dd1c3cee4`。
- 控制端命令：`& "$env:USERPROFILE\.codex\tools\Invoke-ProjectSsh.ps1" -Command 'bash -s' -InputFile 'C:\Users\15919\Desktop\Matcher\tmp\batch5_launch\activate_batch5_matrix.sh'`。双审通过的一次性控制器摘要为 `ce909d15990207f46387e1c299dffbcdaff25a9958549148d218896d015d8a3e`。
- 动态训练命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" --plan 'configs/experiments/batch5/batch5_smi_ted_v_centroid_graph_gate_002_oracle_a100.yaml'`；命令先要求 `TASK_PROJECT_ROOT=/home/penghongen/Feedback/Matcher/releases/Matcher_d14dd1c3cee4/Matcher`。
- launch：`train_experiment_job340893_20260815T032614_a4`；运行根：`/storage/penghongen/tmp/Matcher/experiments/batch5_smi_ted_v_centroid_graph_gate_002_oracle_a100/train_experiment_job340893_20260815T032614_a4/`。
- 锁边界：控制器只删除 `try_lock_340893`；`after_lock_340893` 保留，`pre_lock_340893/kill_lock_340893` 均不存在。没有取消、重启或释放 allocation。
- 首个 global batch 已完成参数更新：global batch 32、身份有效候选数 7（`identity_eligible_count=7`）；classification/ligand/binding/identity/total loss 为 `1.7836914062/1.7509000003/1.6756209433/1.4832589115/3.6096024122`，裁剪前梯度范数 `42.2546005249`。数据等待/计算为 `3.5967940530/24.6252047321 s`，吞吐 `1.1338672446 candidate/s`，显存峰值 allocated/reserved 为 `14,195,046,400/19,440,599,040` bytes；首次更新后 graph/inventory 门值为 `0.0200999994/0.0000999983`。
- BEST/last epoch 为 `8/10`，global step 为 `4917`，BEST selection score 为 `0.6914142117510179`；运行根与阶段摘要已正常写出并由 runner 自然恢复 `try_lock_340893`。
- 该分数低于第 2 批零门 oracle，也低于双门 `0.02`；单随机种子下没有支持把仅图门 `0.02` 作为默认，但不外推为图分支本身无用。
