# 第 5 批 SMI-TED V 质心 mean/max oracle 双门 0.1

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；V 质心 48³；mean/max；oracle。
- 配置：`configs/experiments/batch5/batch5_smi_ted_v_centroid_gate_010_oracle_a100.yaml`。
- 唯一科学变量：`initial_graph_gate=0.1`、`initial_inventory_gate=0.1`；其它科学字段与第 2 批 oracle 相同。
- 直接对照：同硬件双门 `0.02/0.02` 和第 2 批零门 `0.0/0.0`；graph-only/inventory-only 仍只解释 `0.02` 因子矩阵。
- 动机：用户明确指出门初值没有理由只认可单一值，并早期提出 `0.1`。当前双门任务训练到 epoch 3 后门值约为 `0.043/0.036`，因此 `0.1` 是有意更高的尺度探针，不作为默认建议。
- 判断方式：使用自然早停 BEST 的 validation selection score、六分类、五类身份、small_molecule 兼容身份、区域指标和 `identity_contribution/validation/*`；同时观察首轮稳定性、门值轨迹和非有限事件，不能只按收敛速度判优。
- 预定资源：受管 A100 Job `340897`；最多 16 CPU；物理批量 8、梯度累积 4、global batch 32；不设置训练或 Slurm 时间上限。
- W&B：online project `AdaLigand-Matcher-Batch5`；group `matcher-batch5-gate-initialization`；run ID 与 URL 待启动后回填。
- 状态：已决定，尚未提交；配置、测试、双线 Git、release、独立双审和受保护控制器全部通过前不删除 `try_lock_340897`。

## 提交与结果留证

- 实现提交、等价学习端点与 release：待回填。
- 控制端完整命令、动态训练命令、launch、运行根和锁边界：待受保护触发后回填。
- 首个 global batch、首次完整 validation、BEST/last 与最终头对头结论：待实际事件后回填。
