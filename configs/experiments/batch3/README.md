# 第 3 批：实体偏置增强的候选条件注意力

本目录定义四个第 3 批实验。三种完整 Matcher 分别采用 oracle、detached、joint 身份损失；候选预训练只训练六分类与两个区域头。四项实验均沿用第 2 批的候选级 V 质心 `48³` 视野、12 项四模态统计与 4 项采样统计，并把聚合器统一改为增强后的候选条件注意力。

`use_candidate_token_attention_bias=true` 为 A、P、V、PP 四个模态各建立一套独立 MLP。一个实体是某一模态中掩码为真的 token；补零 token 不参与注意力。每个有效实体的输入顺序为 `(relative_x, relative_y, relative_z, stage1_probability)`：相对 XYZ 已按对应空间的物理半边长归一化；概率是数据层冻结的 Stage1 原值。MLP 为 `Linear(4,D_m) → SiLU → Linear(D_m,K)`，其中 `K` 是注意力头数；末层权重和偏置采用零初始化，所以初始化时增强模型与未增强候选条件注意力逐元素等价。偏置在有效 token 掩码生效前加到 softmax 分数，补零实体不产生影响。

12 项手动统计依次为 A/P/V/PP 各自的 `(token_count, probability_mean, probability_max)`；仅计数进入模型时执行 `log1p`，概率保持 Stage1 无量纲原值，空模态三项均为零。A/P 统计完整 `80³`，V/PP 统计候选级 `48³`。4 项采样统计依次为边界限制标志、裁剪中心到完整 V 质心的 L2/L∞ voxel 距离和 V 保留比例；两项距离分别除以 `24√3` 和 `24`。

PP 原有注意力输出投影仍严格零初始化。当 PP 含有效 token 且损失沿该分支回传时，第一次反向传播先让 PP 输出投影权重获得非零梯度；该权重更新后，下一次反向才向 PP 偏置 MLP 末层权重传递非零梯度。偏置 MLP 末层的 `bias` 对每个注意力头沿 token 轴只产生常数平移，数学上不改变 softmax，因此不作为可学习路径或监控对象。没有增加额外门控，也没有改变候选集合、监督、global batch 或优化器语义。

W&B 在 `gradient/train_step/{a|p|v|pp}_attention_bias_final_weight_postclip_norm` 记录全局梯度裁剪后的末层权重梯度范数，并在 `attention_bias/train_step/{a|p|v|pp}_final_weight_poststep_norm` 记录同次优化器更新后的末层权重范数。某模态在整个 global batch 中没有有效 token、末层权重没有梯度时，两项都不发送；末层 `bias` 不记录。

四项实验使用 W&B project `AdaLigand-Matcher-Batch3`。完整 Matcher group 为 `matcher-batch3-full-enhanced-cross-attention`，候选预训练 group 为 `matcher-batch3-candidate-enhanced-cross-attention`。完整 Matcher 以第 2 批相同损失的 mean/max 任务为预算内基线；候选预训练与第 2 批未增强候选条件注意力直接比较。开发和选择只读取 train/validation；本目录配置不读取 calibration。
