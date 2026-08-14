# 第 3 批 SMI-TED 增强注意力 joint

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；增强候选条件注意力；joint。
- 配置：`configs/experiments/batch3/full_smi_ted_enhanced_cross_attention_joint_a100.yaml`。
- 预算内对照：第 2 批 SMI-TED V 质心 mean/max joint；结合第 2/3 批六分类注意力对照判断聚合差异。
- 预定资源：一张自然空闲的受管 A100，最多 16 CPU；物理批量 8、梯度累积 4、global batch 32。
- W&B：project `AdaLigand-Matcher-Batch3`，group `matcher-batch3-full-enhanced-cross-attention`。
- 个性化变量：身份损失乘以保留分类梯度的真实身份类别概率；A/P/V/PP 各自使用零初始化实体偏置 MLP。
- 实现端点：`495e7de54dca88ab3464fe0a667dcce5a540ef72`；预冻结 release：`/home/penghongen/Feedback/Matcher/releases/Matcher_d15132948a18/Matcher`，内容摘要 `d15132948a1885bae9a2019ac55175503c5c15459f043145147fe4a6d3598f89`。
- 状态：训练命令已于 2026-08-14 19:08:52 自然早停并正常收口；Slurm allocation 继续保留，`try_lock_340899` 已由 runner 自然恢复。

## 实际提交

- 控制端命令：`& "$env:USERPROFILE\.codex\tools\Invoke-ProjectSsh.ps1" -Command "bash -s -- '340899'" -InputFile 'C:\Users\15919\Desktop\Matcher\tmp\batch3_launch\activate_next_a100.sh'`。
- 动态训练命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" --plan configs/experiments/batch3/full_smi_ted_enhanced_cross_attention_joint_a100.yaml`。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_d15132948a18/Matcher`。
- Slurm 与 launch：Job `340899`；`train_experiment_job340899_20260814T151852_a3`。
- W&B：run ID `z41ltqiv`；project/group 采用本文件上方冻结值。
- 锁：只删除 `try_lock_340899`；`after_lock_340899` 保留，未创建或修改 `kill_lock`。
- W&B 在线地址：[z41ltqiv](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch3/runs/z41ltqiv)。

## 最终结果

- BEST/last epoch：`4/6`；累计参数更新 `3129`；连续两个坏验证轮触发自然早停。
- BEST selection score：`0.6763544866`。
- BEST 六分类 macro-F1：`0.4939851657`。
- 五类身份固有 Top-1 / 精确子类条件 Top-1：`0.8328785812 / 0.8985645933`。
- 宽松五类端到端 F1 / small_molecule 兼容端到端 F1：`0.6365137009 / 0.6194503171`。
- ligand / binding micro Dice：`0.2493513245 / 0.0801483948`。
- 有效身份监督候选数：`1466`。
- 运行根：`/storage/penghongen/tmp/Matcher/experiments/batch3_smi_ted_enhanced_cross_attention_joint_a100/train_experiment_job340899_20260814T151852_a3/`。

## 头对头判断

直接基线是第 2 批 V 质心 mean/max joint：selection score `0.6887556119`，六分类 macro-F1 `0.5097478210`，五类固有 Top-1 `0.8560709413`，精确子类条件 Top-1 `0.9019980971`，宽松五类端到端 F1 `0.6545209177`，small_molecule 兼容端到端 F1 `0.6367346939`，ligand/binding micro Dice `0.3019974340/0.0669487746`。

增强注意力 joint 的 selection score、六分类、两项身份 Top-1 和两项端到端 F1 约低 `0.38%–3.09%`，ligand Dice 低约 `17.43%`，binding Dice 高约 `19.72%`。它没有出现多个核心指标同时下降约 10%，因此不判定为明显有害；但结合增强六分类也未形成优势，目前没有证据优先采用增强注意力。第 5 批最小对照继续使用更简单、当前更强的 V 质心 mean/max 架构。
