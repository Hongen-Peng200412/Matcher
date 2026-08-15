# 联合身份 v4 实验配置

本目录保存把身份键从 `object_key` 提升为 `(object_key, PreparedLigand.smiles)` 后的独立基线。`PreparedLigand.smiles` 是语言模型公开接口实际接收的顶层规范非同分异构 SMILES（canonical non-isomeric SMILES）；配置不会使用 occurrence 坐标、`present` 掩码、沉积原子选择或语言模型归档中的 `model_smiles` 定义身份。

实际身份清单版本由执行配置的 Matcher 包中 `MANIFEST_SCHEMA_VERSION` 决定，不由 YAML 重复声明；本目录配置必须由采用 schema v4 联合身份的 Matcher release 执行。

六份配置组成三个训练随机种子配对实验。六项 `data.selection_seed` 全部固定为 `3407`，保证联合身份的语言 occurrence 选择和清单契约不变；三个配对的 `training.seed` 分别为 `3407`、`3408` 和 `3409`。每个配对只比较 `training.learning_rate=1.0e-4` 与 `2.0e-4`；计划名、阶段名和 W&B 分组只用于隔离任务，不改变模型输入、监督或优化规则。所有配置都使用物理批量 8、梯度累积 4、global batch 32、梯度裁剪阈值 1、两个零增量身份门、oracle 身份损失和 `initialize_candidate_from: null` 从零训练。global batch 32 表示每次参数更新累计使用 32 个候选。

| 随机种子 | LR=`1.0e-4` 配置 | LR=`2.0e-4` 配置 | W&B group |
|---:|---|---|---|
| `3407` | `joint_identity_v4_zero_gate_oracle_a100.yaml` | `joint_identity_v4_lr_2e4_oracle_a100.yaml` | 既有基线使用 `joint-identity-v4-zero-gate-baseline`；学习率探针使用 `joint-identity-v4-learning-rate` |
| `3408` | `joint_identity_v4_seed_3408_lr_1e4_oracle_a100.yaml` | `joint_identity_v4_seed_3408_lr_2e4_oracle_a100.yaml` | `joint-identity-v4-learning-rate-seed-3408` |
| `3409` | `joint_identity_v4_seed_3409_lr_1e4_oracle_a100.yaml` | `joint_identity_v4_seed_3409_lr_2e4_oracle_a100.yaml` | `joint-identity-v4-learning-rate-seed-3409` |

seed 3407 的 LR=`1.0e-4` 配置保持与[第 2 批零门 oracle 配置](../batch2/full_smi_ted_v_centroid_oracle_a100.yaml)相同的数据参数、表示、模型、损失、随机种子、训练超参数和 A100 批量；计划名、阶段名及 W&B 项目和组用于隔离联合身份 v4 实验。历史 v3 零门结果只作等价迁移的辅助参照，不替代三个 v4 配对内部的正式对照。

清单来源审计、Dataset 与固定权重数值等价、输出目录查重和受管资源触发属于任务执行证据，应记录在联合身份 v4 的单任务、批次和总记录中，不写入本配置目录的稳定科学契约。

W&B 统一使用 `AdaLigand-Matcher-JointIdentityV4` 项目；上表列出六项任务的分组。六项任务都只用 validation 选模；calibration 仍等待全部迭代结束及用户单独确认。
