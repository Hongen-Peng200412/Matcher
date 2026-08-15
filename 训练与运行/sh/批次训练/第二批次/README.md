# 第二批次固定入口

本目录保存第二批次五份训练入口。三份完整 Matcher 计划使用候选 V 质心裁剪、mean/max 聚合，并分别采用 `oracle`、`detached`、`joint` 身份损失；两份候选六分类计划分别使用 mean/max 和未增强的候选条件交叉注意力。

脚本均不接受参数，并按当前代码和依赖解释对应 YAML。它们用于提供可读、固定的提交入口，不保证与最初实验环境兼容。

## 脚本与计划

| 脚本 | 唯一 YAML | 提交资源 | 物理批量 × 累积次数 = 全局批量 |
| --- | --- | --- | ---: |
| [`full_smi_ted_v_centroid_oracle_a100.sh`](full_smi_ted_v_centroid_oracle_a100.sh) | [`full_smi_ted_v_centroid_oracle_a100.yaml`](../../../../configs/experiments/batch2/full_smi_ted_v_centroid_oracle_a100.yaml) | 1 张 A100、16 核 CPU | 8 × 4 = 32 |
| [`full_smi_ted_v_centroid_detached_a100.sh`](full_smi_ted_v_centroid_detached_a100.sh) | [`full_smi_ted_v_centroid_detached_a100.yaml`](../../../../configs/experiments/batch2/full_smi_ted_v_centroid_detached_a100.yaml) | 1 张 A100、16 核 CPU | 8 × 4 = 32 |
| [`full_smi_ted_v_centroid_joint_a100.sh`](full_smi_ted_v_centroid_joint_a100.sh) | [`full_smi_ted_v_centroid_joint_a100.yaml`](../../../../configs/experiments/batch2/full_smi_ted_v_centroid_joint_a100.yaml) | 1 张 A100、16 核 CPU | 8 × 4 = 32 |
| [`candidate_v_centroid_mean_max_a800.sh`](candidate_v_centroid_mean_max_a800.sh) | [`candidate_v_centroid_mean_max_a800.yaml`](../../../../configs/experiments/batch2/candidate_v_centroid_mean_max_a800.yaml) | 1 张 A800、24 核 CPU | 16 × 2 = 32 |
| [`candidate_v_centroid_cross_attention_a800.sh`](candidate_v_centroid_cross_attention_a800.sh) | [`candidate_v_centroid_cross_attention_a800.yaml`](../../../../configs/experiments/batch2/candidate_v_centroid_cross_attention_a800.yaml) | 1 张 A800、24 核 CPU | 16 × 2 = 32 |

## 依赖与阅读顺序

先区分完整 Matcher 与候选六分类计划，再核对聚合方式和资源；随后阅读唯一 YAML 与 [`train_experiment.sh`](../../train_experiment.sh)。提交时直接复制目标脚本头部命令，不要向脚本追加参数。
