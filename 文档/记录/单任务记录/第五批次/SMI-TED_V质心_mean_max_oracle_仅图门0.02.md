# 第 5 批 SMI-TED V 质心 mean/max oracle 仅图门 0.02

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；V 质心 48³；mean/max；oracle。
- 配置：`configs/experiments/batch5/batch5_smi_ted_v_centroid_graph_gate_002_oracle_a100.yaml`。
- 唯一科学变量：`initial_graph_gate=0.02`；`initial_inventory_gate=0.0`，其它科学字段与第 2 批 oracle 相同。
- 直接对照：第 2 批零门 `00` 与第 5 批双门 `11`；它与仅清单门 `01` 共同补齐 2×2 因子矩阵。
- 动机：当前双门任务 epoch 3 的图增量/语言基线中心化 L2 比例约为 `0.2174`，清单增量约为 `0.0458`；只启用图门可分辨图分支主效应与两门交互，不能把该比例直接解释成有符号收益。
- 判断方式：使用自然早停 BEST 的 validation selection score、六分类、五类身份、small_molecule 兼容身份、区域指标和 `identity_contribution/validation/*`；单个指标或单轮波动不下结论。
- 预定资源：受管 A100 Job `340893`；最多 16 CPU；物理批量 8、梯度累积 4、global batch 32；不设置训练或 Slurm 时间上限。
- W&B：online project `AdaLigand-Matcher-Batch5`；group `matcher-batch5-gate-initialization`；run ID 与 URL 待启动后回填。
- 状态：已决定，尚未提交；配置、测试、双线 Git、release、独立双审和受保护控制器全部通过前不删除 `try_lock_340893`。

## 提交与结果留证

- 实现提交、等价学习端点与 release：待回填。
- 控制端完整命令、动态训练命令、launch、运行根和锁边界：待受保护触发后回填。
- 首个 global batch、首次完整 validation、BEST/last 与最终头对头结论：待实际事件后回填。
