# 第 3 批 SMI-TED 增强注意力 joint

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；增强候选条件注意力；joint。
- 配置：`configs/experiments/batch3/full_smi_ted_enhanced_cross_attention_joint_a100.yaml`。
- 预算内对照：第 2 批 SMI-TED V 质心 mean/max joint；结合第 2/3 批六分类注意力对照判断聚合差异。
- 预定资源：一张自然空闲的受管 A100，最多 16 CPU；物理批量 8、梯度累积 4、global batch 32。
- W&B：project `AdaLigand-Matcher-Batch3`，group `matcher-batch3-full-enhanced-cross-attention`。
- 个性化变量：身份损失乘以保留分类梯度的真实身份类别概率；A/P/V/PP 各自使用零初始化实体偏置 MLP。
- 实现端点：`495e7de54dca88ab3464fe0a667dcce5a540ef72`；预冻结 release：`/home/penghongen/Feedback/Matcher/releases/Matcher_d15132948a18/Matcher`，内容摘要 `d15132948a1885bae9a2019ac55175503c5c15459f043145147fe4a6d3598f89`。
- 状态：已于 2026-08-14 15:18:48 启动，W&B online 运行已建立；首个 global batch 与 validation 尚待运行产物确认。

## 实际提交

- 控制端命令：`& "$env:USERPROFILE\.codex\tools\Invoke-ProjectSsh.ps1" -Command "bash -s -- '340899'" -InputFile 'C:\Users\15919\Desktop\Matcher\tmp\batch3_launch\activate_next_a100.sh'`。
- 动态训练命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" --plan configs/experiments/batch3/full_smi_ted_enhanced_cross_attention_joint_a100.yaml`。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_d15132948a18/Matcher`。
- Slurm 与 launch：Job `340899`；`train_experiment_job340899_20260814T151852_a3`。
- W&B：run ID `z41ltqiv`；project/group 采用本文件上方冻结值。
- 锁：只删除 `try_lock_340899`；`after_lock_340899` 保留，未创建或修改 `kill_lock`。
- 待回填：首个 global batch 32 更新、validation 指标、与第 2 批 joint 的头对头分析和最终运行状态。
