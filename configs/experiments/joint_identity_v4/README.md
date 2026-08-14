# 联合身份 v4 实验配置

本目录保存把身份键从 `object_key` 提升为 `(object_key, PreparedLigand.smiles)` 后的独立基线。这里的 `PreparedLigand.smiles` 是语言模型公开接口实际接收的顶层 canonical non-isomeric SMILES；配置不会使用 occurrence 坐标、`present` 掩码、沉积原子选择或语言模型归档中的 `model_smiles` 定义身份。

`joint_identity_v4_zero_gate_oracle_a100.yaml` 是第一项零门 oracle 基线。解析后的数据、表示、模型、损失、随机种子、训练超参数和 A100 批量与[第 2 批零门 oracle 配置](../batch2/full_smi_ted_v_centroid_oracle_a100.yaml)相同；计划名、阶段名和 W&B 项目/组用于隔离新实验。实际身份清单版本由执行该配置的 Matcher 包中 `MANIFEST_SCHEMA_VERSION` 决定，不由 YAML 重复声明。

清单来源审计、Dataset 与固定权重数值等价、输出目录查重和受管资源触发属于任务执行证据，应记录在联合身份 v4 的单任务、批次和总记录中，不写入本配置目录的稳定科学契约。

W&B 单独使用 `AdaLigand-Matcher-JointIdentityV4` 项目与 `joint-identity-v4-zero-gate-baseline` 组，不与既有实验批次曲线混排。该基线只用 validation 选模；calibration 仍等待全部迭代结束及用户单独确认。
