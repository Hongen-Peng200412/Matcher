# 第 3 批 SMI-TED 增强注意力 detached

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；增强候选条件注意力；detached。
- 配置：`configs/experiments/batch3/full_smi_ted_enhanced_cross_attention_detached_a100.yaml`。
- 预算内对照：第 2 批 SMI-TED V 质心 mean/max detached；结合第 2/3 批六分类注意力对照判断聚合差异。
- 预定资源：一张自然空闲的受管 A100，最多 16 CPU；物理批量 8、梯度累积 4、global batch 32。
- W&B：project `AdaLigand-Matcher-Batch3`，group `matcher-batch3-full-enhanced-cross-attention`。
- 个性化变量：身份损失乘以切断梯度的真实身份类别概率；A/P/V/PP 各自使用零初始化实体偏置 MLP。
- 状态：代码、配置、pytest、mypy、compileall、pyflakes、YAML 解析、shell 语法、diff-check 机械检查以及科学正确性、严格技术表达两项独立审查均已通过；第 2 批五项也已全部提交。当前等待实现提交与安全发布，没有 Slurm Job、release、launch 或 W&B run，不虚构提交命令与结果。

实际提交时必须在本文件追加完整 `submit_task.sh` 命令、动态运行命令、release、Slurm Job、W&B run ID/URL、首步验收、validation 结果和分析。
