# 第 5 批 SMI-TED V 质心 mean/max oracle 双门 0.1

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；V 质心 48³；mean/max；oracle。
- 配置：`configs/experiments/batch5/batch5_smi_ted_v_centroid_gate_010_oracle_a100.yaml`。
- 唯一科学变量：`initial_graph_gate=0.1`、`initial_inventory_gate=0.1`；其它科学字段与第 2 批 oracle 相同。
- 直接对照：同硬件双门 `0.02/0.02` 和第 2 批零门 `0.0/0.0`；graph-only/inventory-only 仍只解释 `0.02` 因子矩阵。
- 动机：用户明确指出门初值没有理由只认可单一值，并早期提出 `0.1`。当前双门任务训练到 epoch 3 后门值约为 `0.043/0.036`，因此 `0.1` 是有意更高的尺度探针，不作为默认建议。
- 判断方式：使用自然早停 BEST 的 validation selection score、六分类、五类身份、small_molecule 兼容身份、区域指标和 `identity_contribution/validation/*`；同时观察首轮稳定性、门值轨迹和非有限事件，不能只按收敛速度判优。
- 实际资源：受管 A100 Job `340897`；16 CPU；物理批量 8、梯度累积 4、global batch 32；本任务不设置训练或 Slurm 时间上限。
- W&B：online project `AdaLigand-Matcher-Batch5`；group `matcher-batch5-gate-initialization`；run [`57uwuxb6`](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch5/runs/57uwuxb6)。
- 状态：已提交并完成首个 global batch，继续训练；未读取 calibration。

## 提交与结果留证

- 实现提交为 `5548a1021729ac514896cb41a54bfd599cf26517`；等价学习端点为 `62c59f299612f4752f84aafec45916a2ceef602d`；内容寻址 release 为 `Matcher_d14dd1c3cee4`。
- 控制端命令：`& "$env:USERPROFILE\.codex\tools\Invoke-ProjectSsh.ps1" -Command 'bash -s' -InputFile 'C:\Users\15919\Desktop\Matcher\tmp\batch5_launch\activate_batch5_matrix.sh'`。双审通过的一次性控制器摘要为 `ce909d15990207f46387e1c299dffbcdaff25a9958549148d218896d015d8a3e`。
- 动态训练命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" --plan 'configs/experiments/batch5/batch5_smi_ted_v_centroid_gate_010_oracle_a100.yaml'`；命令先要求 `TASK_PROJECT_ROOT=/home/penghongen/Feedback/Matcher/releases/Matcher_d14dd1c3cee4/Matcher`。
- launch：`train_experiment_job340897_20260815T032605_a4`；运行根：`/storage/penghongen/tmp/Matcher/experiments/batch5_smi_ted_v_centroid_gate_010_oracle_a100/train_experiment_job340897_20260815T032605_a4/`。
- 锁边界：控制器只删除 `try_lock_340897`；`after_lock_340897` 保留，`pre_lock_340897/kill_lock_340897` 均不存在。没有取消、重启或释放 allocation。
- 首个 global batch 已完成参数更新：global batch 32、身份有效候选数 7（`identity_eligible_count=7`）；classification/ligand/binding/identity/total loss 为 `1.7817382812/1.7512671351/1.6758121848/1.4810268027/3.6054730160`，裁剪前梯度范数 `48.4062690735`。数据等待/计算为 `9.0371542359/25.1023310081 s`，吞吐 `0.9373310632 candidate/s`，显存峰值 allocated/reserved 为 `14,195,046,400/19,440,599,040` bytes；首次更新后 graph/inventory 门值为 `0.1001000032/0.1000999957`。
- 首次完整 validation、BEST/last 与最终头对头结论：待自然事件后回填。
