# 第 3 批 SMI-TED 增强注意力 oracle

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；增强候选条件注意力；oracle。
- 配置：`configs/experiments/batch3/full_smi_ted_enhanced_cross_attention_oracle_a100.yaml`。
- 预算内对照：第 2 批 SMI-TED V 质心 mean/max oracle；结合第 2/3 批六分类注意力对照判断聚合差异。
- 预定资源：一张自然空闲的受管 A100，最多 16 CPU；物理批量 8、梯度累积 4、global batch 32。
- W&B：project `AdaLigand-Matcher-Batch3`，group `matcher-batch3-full-enhanced-cross-attention`。
- 个性化变量：身份系数恒为 1；A/P/V/PP 各自使用零初始化实体偏置 MLP。
- 实现端点：`495e7de54dca88ab3464fe0a667dcce5a540ef72`；预冻结 release：`/home/penghongen/Feedback/Matcher/releases/Matcher_d15132948a18/Matcher`，内容摘要 `d15132948a1885bae9a2019ac55175503c5c15459f043145147fe4a6d3598f89`。
- 状态：代码、配置、机械检查、两项独立复审、提交、安全同步、远端哈希与配置门禁均已通过；第 2 批五项也已全部提交。当前等待一张受管 A100 自然空闲，没有本任务的 Slurm Job、launch 或 W&B run，不虚构提交命令与结果。

实际提交时必须在本文件追加完整 `submit_task.sh` 命令、动态运行命令、release、Slurm Job、W&B run ID/URL、首步验收、validation 结果和分析。
