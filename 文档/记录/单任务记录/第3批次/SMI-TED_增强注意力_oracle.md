# 第 3 批 SMI-TED 增强注意力 oracle

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；增强候选条件注意力；oracle。
- 配置：`configs/experiments/batch3/full_smi_ted_enhanced_cross_attention_oracle_a100.yaml`。
- 预算内对照：第 2 批 SMI-TED V 质心 mean/max oracle；结合第 2/3 批六分类注意力对照判断聚合差异。
- 预定资源：一张自然空闲的受管 A100，最多 16 CPU；物理批量 8、梯度累积 4、global batch 32。
- W&B：project `AdaLigand-Matcher-Batch3`，group `matcher-batch3-full-enhanced-cross-attention`。
- 个性化变量：身份系数恒为 1；A/P/V/PP 各自使用零初始化实体偏置 MLP。
- 实现端点：`495e7de54dca88ab3464fe0a667dcce5a540ef72`；预冻结 release：`/home/penghongen/Feedback/Matcher/releases/Matcher_d15132948a18/Matcher`，内容摘要 `d15132948a1885bae9a2019ac55175503c5c15459f043145147fe4a6d3598f89`。
- 状态：已于 2026-08-14 22:59:04（Asia/Shanghai）自然早停；BEST/last epoch 为 `11/13`。

## 实际提交

- 控制端命令：`& "$env:USERPROFILE\.codex\tools\Invoke-ProjectSsh.ps1" -Command "bash -s -- '340893'" -InputFile 'C:\Users\15919\Desktop\Matcher\tmp\batch3_launch\activate_next_a100.sh'`。
- 动态训练命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" --plan configs/experiments/batch3/full_smi_ted_enhanced_cross_attention_oracle_a100.yaml`。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_d15132948a18/Matcher`。
- Slurm 与 launch：Job `340893`；`train_experiment_job340893_20260814T151725_a3`。
- W&B：run ID [`4iqynr6i`](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch3/runs/4iqynr6i)；project/group 采用本文件上方冻结值。
- 锁：只删除 `try_lock_340893`；`after_lock_340893` 保留，未创建或修改 `kill_lock`。

## 最终 validation 与结论

BEST epoch 11 的 selection score 为 `0.6921287290`；六分类 macro-F1、五类固有 Top-1、精确子类条件 Top-1、宽松五类端到端 F1 和 small_molecule 兼容端到端 F1 依次为 `0.5296075890/0.8649386085/0.8764320786/0.6703465195/0.6895927602`。ligand/binding micro-Dice 为 `0.2974951006/0.0706982525`，身份有效监督数为 `1,466`。

相对第 2 批 V 质心 mean/max oracle，selection score 低 `0.856%`，六分类 macro-F1 高 `0.207%`，五类固有/精确条件/宽松端到端约低 `1.246%/1.516%/0.819%`，small_molecule 兼容端到端高 `0.895%`，ligand/binding Dice 低 `6.207%/30.252%`。变化方向不一致，不能判为多个核心指标共同约 10% 的明显负优化，但不支持用增强注意力替代更简单的 mean/max oracle。

训练命令正常写出运行根与阶段 `summary.json`，`stopped_by_patience=true`；Job `340893` 的 allocation 和 `after_lock_340893` 继续保留，runner 已恢复 `try_lock_340893`。运行根为 `/storage/penghongen/tmp/Matcher/experiments/batch3_smi_ted_enhanced_cross_attention_oracle_a100/train_experiment_job340893_20260814T151725_a3/`。
