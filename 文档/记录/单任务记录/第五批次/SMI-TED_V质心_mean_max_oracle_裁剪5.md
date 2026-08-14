# 第 5 批 SMI-TED V 质心 mean/max oracle 裁剪阈值 5

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；V 质心 48³；mean/max；oracle。
- 配置：`configs/experiments/batch5/batch5_smi_ted_v_centroid_clip_5_oracle_a800.yaml`。
- 唯一科学变量：`gradient_clip_norm=5.0`；双门为零，学习率、优化器、损失和 global batch 与第 2 批 oracle 相同。
- 直接对照：同一 A800 资源适配的 `clip200` 是主对照；第 2 批 A100 `clip1` 只作次级参考，因为物理批量由 `8×4` 改为 `16×2`，不宣称逐位同轨。
- 动机：第 2 批 oracle 的裁剪前梯度范数 P50 为 `6.5609`，阈值 `1.0` 在 `99.93%` 更新中生效；阈值 `5.0` 仍保留尾部保护，同时把中位更新缩放系数从约 `0.152` 放宽到约 `0.762`。
- 判断方式：比较自然早停 BEST 的全部核心 validation 指标，并报告总更新数、裁剪触发数/比例、最大裁剪前范数和最小缩放系数；失败或非有限值本身也是结果，不无记录重启。
- 预定资源：受管 A800 Job `340894`；最多 16 CPU；物理批量 16、梯度累积 2、global batch 32；既有 allocation 的调度器身份时限只作核对，本脚本不设置或缩短训练时限。
- W&B：online project `AdaLigand-Matcher-Batch5`；group `matcher-batch5-gradient-clipping`；run ID 与 URL 待启动后回填。
- 状态：已决定，尚未提交；配置、测试、双线 Git、release、独立双审和受保护控制器全部通过前不删除 `try_lock_340894`。

## 提交与结果留证

- 实现提交、等价学习端点与 release：待回填。
- 控制端完整命令、动态训练命令、launch、运行根和锁边界：待受保护触发后回填。
- 首个 global batch、首次完整 validation、BEST/last 与最终头对头结论：待实际事件后回填。
