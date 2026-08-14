# 第 5 批 SMI-TED V 质心 mean/max oracle 双门 0.02

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；V 质心 48³；mean/max；oracle。
- 配置：`configs/experiments/batch5/batch5_smi_ted_v_centroid_gate_002_oracle_a100.yaml`。
- 唯一科学变量：`initial_graph_gate=0.02`、`initial_inventory_gate=0.02`；第 2 批直接基线均为 `0.0`。
- 直接对照：第 2 批 `batch2_smi_ted_v_centroid_oracle_a100`，BEST selection score `0.6981048482`。
- 动机：第 1–3 批门值自然学习到约 `0.01–0.03`，首项选择同量级 `0.02`；不直接测试缺少证据的 `0.1`，也不同时改变裁剪阈值或学习率。
- 判断方式：优先比较 validation selection score、六分类、五类身份、small_molecule 兼容身份和区域指标；结合 `identity_contribution/validation/*` 判断门控增量相对语言基线的尺度。单个指标波动不构成正负结论。
- 预定资源：受管 A100 Job `340899`；最多 16 CPU；物理批量 8、梯度累积 4、global batch 32；不设置训练或 Slurm 时间上限。
- W&B：online project `AdaLigand-Matcher-Batch5`；group `matcher-batch5-gate-initialization`；run ID 和在线地址在实际启动后回填。
- 状态：代码、配置与本记录准备中，尚未提交；Job `340899` 的第 3 批 joint 已自然早停并由 `try_lock_340899` 保留，当前未删除该锁。

## 提交留证

- 实现提交：待双线 Git 收口后回填。
- release：待安全同步、内容寻址发布和远端门禁后回填。
- 控制端完整命令：待受保护激活脚本双审并实际执行后回填。
- 动态训练命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" --plan configs/experiments/batch5/batch5_smi_ted_v_centroid_gate_002_oracle_a100.yaml`。
- Slurm/launch/W&B：待实际触发后回填；不得预填虚构标识。
- 锁边界：实际触发只允许删除 `try_lock_340899`；必须保留 `after_lock_340899`，不得创建或删除 `kill_lock/pre_lock`。

## 验收与结果

- 首个 global batch：待实际启动后回填有限损失、裁剪前梯度范数、数据等待、吞吐和显存。
- 首次完整 validation：待回填正式指标与门控贡献诊断。
- 最终结果：待自然早停后回填 BEST/last、W&B、运行产物和与第 2 批 oracle 的头对头判断。
