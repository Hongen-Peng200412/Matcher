# 联合身份 v4 实验配置

本目录保存把身份键从 `object_key` 提升为 `(object_key, PreparedLigand.smiles)` 后的独立基线。`PreparedLigand.smiles` 是语言模型公开接口实际接收的顶层规范非同分异构 SMILES（canonical non-isomeric SMILES）；配置不会使用 occurrence 坐标、`present` 掩码、沉积原子选择或语言模型归档中的 `model_smiles` 定义身份。

实际身份清单版本由执行配置的 Matcher 包中 `MANIFEST_SCHEMA_VERSION` 决定，不由 YAML 重复声明；本目录配置必须由采用 schema v4 联合身份的 Matcher release 执行。

六份学习率配置组成三个训练随机种子配对实验。六项 `data.selection_seed` 全部固定为 `3407`，保证联合身份的语言 occurrence 选择和清单契约不变；三个配对的 `training.seed` 分别为 `3407`、`3408` 和 `3409`。每个配对只比较 `training.learning_rate=1.0e-4` 与 `2.0e-4`；计划名、阶段名和 W&B 分组只用于隔离任务，不改变模型输入、监督或优化规则。所有配置都使用物理批量 8、梯度累积 4、global batch 32、梯度裁剪阈值 1、两个零增量身份门、oracle 身份损失和 `initialize_candidate_from: null` 从零训练。global batch 32 表示每次参数更新累计使用 32 个候选。

| 随机种子 | LR=`1.0e-4` 配置 | LR=`2.0e-4` 配置 | W&B group |
|---:|---|---|---|
| `3407` | `joint_identity_v4_zero_gate_oracle_a100.yaml` | `joint_identity_v4_lr_2e4_oracle_a100.yaml` | 既有基线使用 `joint-identity-v4-zero-gate-baseline`；学习率探针使用 `joint-identity-v4-learning-rate` |
| `3408` | `joint_identity_v4_seed_3408_lr_1e4_oracle_a100.yaml` | `joint_identity_v4_seed_3408_lr_2e4_oracle_a100.yaml` | `joint-identity-v4-learning-rate-seed-3408` |
| `3409` | `joint_identity_v4_seed_3409_lr_1e4_oracle_a100.yaml` | `joint_identity_v4_seed_3409_lr_2e4_oracle_a100.yaml` | `joint-identity-v4-learning-rate-seed-3409` |

seed 3407 的 LR=`1.0e-4` 配置保持与[第 2 批零门 oracle 配置](../batch2/full_smi_ted_v_centroid_oracle_a100.yaml)相同的数据参数、表示、模型、损失、随机种子、训练超参数和 A100 批量；计划名、阶段名及 W&B 项目和组用于隔离联合身份 v4 实验。历史 v3 零门结果只作等价迁移的辅助参照，不替代三个 v4 配对内部的正式对照。

清单来源审计、Dataset 与固定权重数值等价、输出目录查重和受管资源触发属于任务执行证据，应记录在联合身份 v4 的单任务、批次和总记录中，不写入本配置目录的稳定科学契约。

六项学习率任务使用 `AdaLigand-Matcher-JointIdentityV4` 项目；下述八项摘要拼接实验使用独立的 `AdaLigand-Matcher-JointIdentityV4-SummaryConcat` 项目。上表列出六项学习率任务的分组。全部任务都只用 validation 选模；calibration 仍等待全部迭代结束及用户单独确认。

## 摘要拼接结构锚点

`joint_identity_v4_concat_zero_gate_seed_3407_lr_1e4_oracle_a800.yaml` 是最后一组摘要结构实验的首个直接对照。它在 seed 3407、A800、LR=`1.0e-4`、零身份门和 clip 1 下，把候选聚合从 `mean_max` 改为 `mean_max_candidate_cross_attention_concat`；数据、模型宽度、损失、批量、初始化和评估口径保持不变。计划名、阶段名、输出目录名及 W&B project/group 只隔离产物。

该配置先完整计算 mean/max 与候选条件注意力的两个 `D_m=model.model_dim` 维表示，再把二者拼接为 `2D_m` 维残差输入。残差末层为零初始化，初始候选表示逐值等于 mean/max 主路径；训练后允许条件注意力表示通过残差参与任务。两种摘要各自按 A/P/V/PP 顺序读取每模态的 token 数、Stage1 固化概率均值和 Stage1 固化概率最大值，共 12 项手动统计；另读取请求裁剪起点是否在任一轴被边界限制（`requested_start_zyx != start_zyx`）、裁剪中心相对完整 V 质心的 L2 与 L∞ 体素距离、裁剪保留的 V 体素比例，共 4 项采样统计。`data.local_crop_strategy=v_centroid` 先以 `np.rint(mean(V_zyx)+0.5-24)` 的最近偶数规则求裁剪起点，再逐轴限制到 `[0,32]`。残差入口不再第三次追加这些原始统计。该任务只用 validation 选择，不读取 calibration。

## 摘要拼接参数矩阵

最后一组实验包含八个摘要拼接任务。四项 A800 任务固定训练随机种子 3407、LR=`1.0e-4` 和相同批量，以双身份门 `{0, 0.02}` 与梯度裁剪阈值 `{1, 200}` 形成完整 2×2 矩阵。四项 A100 任务固定训练随机种子 3408、LR=`1.0e-4` 和相同批量，使用相同的 2×2 设计。A100 与 A800 之间同时改变硬件和训练随机种子，只提供跨硬件、跨种子的方向性重复证据，不能把差异归因于硬件或种子任一单轴，也不冒充逐位同轨对照。

| GPU | 门值 | 裁剪阈值 | 配置 | 直接用途 |
|---|---:|---:|---|---|
| A800 | 0 | 1 | `joint_identity_v4_concat_zero_gate_seed_3407_lr_1e4_oracle_a800.yaml` | 与同卡 v4 mean/max 基线比较摘要结构；也是 A800 裁剪对照基线 |
| A800 | 0.02 | 1 | `joint_identity_v4_concat_gate_002_seed_3407_lr_1e4_oracle_a800.yaml` | 与 A800 零门、clip 1 任务只比较双门初值；复核 A100 门值方向 |
| A800 | 0 | 200 | `joint_identity_v4_concat_zero_gate_clip_200_seed_3407_lr_1e4_oracle_a800.yaml` | 与 A800 零门、clip 1 任务只比较裁剪阈值 |
| A800 | 0.02 | 200 | `joint_identity_v4_concat_gate_002_clip_200_seed_3407_lr_1e4_oracle_a800.yaml` | 补齐 A800 2×2；相对双门、clip 1 只改变裁剪，相对零门、clip 200 只改变双门 |
| A100 | 0 | 1 | `joint_identity_v4_concat_zero_gate_seed_3408_lr_1e4_oracle_a100.yaml` | A100 2×2 矩阵共同基线 |
| A100 | 0.02 | 1 | `joint_identity_v4_concat_gate_002_seed_3408_lr_1e4_oracle_a100.yaml` | clip1 下的双门效应 |
| A100 | 0 | 200 | `joint_identity_v4_concat_zero_gate_clip_200_seed_3408_lr_1e4_oracle_a100.yaml` | 零门下的宽松裁剪效应 |
| A100 | 0.02 | 200 | `joint_identity_v4_concat_gate_002_clip_200_seed_3408_lr_1e4_oracle_a100.yaml` | 双门与宽松裁剪的联合单元，用于估计交互 |

八项都固定 `data.selection_seed=3407`、oracle 身份损失、物理批量 8、梯度累积 4、global batch 32、BF16、从零训练和完整 validation 选模。裁剪阈值 200 只表示宽松裁剪；以 W&B 逐更新字段 `gradient/train_step/preclip_norm` 为裁剪前全局梯度范数，分母为全部 global update。只有范数大于 200 的更新比例不超过 0.1% 时，运行记录才能称其为“近似不裁剪”；同时必须保存总更新数、触发数、触发比例、最大范数，以及按每步 `min(1, 200 / (preclip_norm + 1e-6))` 派生的最小缩放系数。
