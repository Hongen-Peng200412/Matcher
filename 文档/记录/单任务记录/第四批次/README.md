# 第四批次单任务记录入口

本目录保存第四批“候选预训练 BEST 初始化完整 Matcher”的逐任务提交与结果记录。2026-08-14，第 3 批增强六分类运行产物于 18:27 自然收口，18:51 只读核验后只依据 validation 选择第 2 批 V 质心 mean/max 候选预训练 BEST；本批 oracle、detached、joint 三项同架构直接对照已于 20:01–20:03 全部启动。detached 已自然早停，oracle 与 joint 继续训练。

每个任务在触发 GPU 前建立一个独立 Markdown 文件，并至少记录：

- 科学变量：来源候选预训练、目标完整 Matcher、损失、聚合方式、特征开关和对照任务。
- 配置与版本：YAML 路径、实现提交、内容寻址 release 和一次性来源审计报告路径。
- 资源身份：受管 Slurm Job、GPU 型号、CPU 数、物理批量、梯度累积与 global batch。
- 完整提交：控制端命令、动态训练命令、触发时间、launch 和保留的 `after_lock`。
- 状态时间线：提交、开始、首步、每次刷新 BEST、自然早停或失败，以及训练命令退出状态；Slurm allocation 的保留状态单独记录。
- 产物位置：运行根、阶段目录、`resolved.json`、`history.jsonl`、BEST/LAST、阶段 `summary.json` 和运行根 `summary.json`。
- W&B：project、group、run ID、在线地址和分层指标前缀。
- 个性化信息：该任务相对来源候选预训练及最近头对头基线的唯一变量。
- 运行验收：首个 global batch 32 更新、有限损失与梯度、显存和数据等待。
- validation 结果：selection score、六分类、身份层次、区域指标、有效监督数量和混淆情况。
- 结论：与明确对照的差异、是否出现多个核心指标约 10% 同向下降，以及下一批采用或放弃的理由。

当前任务记录为：

- [SMI-TED 候选预训练初始化 oracle](SMI-TED_候选预训练初始化_oracle.md)
- [SMI-TED 候选预训练初始化 detached](SMI-TED_候选预训练初始化_detached.md)
- [SMI-TED 候选预训练初始化 joint](SMI-TED_候选预训练初始化_joint.md)

三项的本地配置、测试、双线 Git、服务器 release、CPU-only 初始化 smoke、受保护触发和两项独立终审均已通过；每份任务记录已经保存 Job、launch、W&B、动态命令、锁状态和首个已核实训练快照。批次科学契约与当前实现状态见 [第四批候选预训练初始化](../../批次记录/第4批次/第4批候选预训练初始化.md)。calibration 在用户最终确认前不得用于本目录任何任务的模型推理、指标或选择。
