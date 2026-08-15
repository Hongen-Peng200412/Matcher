# 第一批次固定入口

本目录保存第一批次三份完整 Matcher 训练入口。三项实验都使用 SMI-TED 289M、mean/max 聚合、固定中心裁剪和五类非背景身份监督。模型与监督的科学设置仅在身份损失的 `oracle`、`detached` 和 `joint` 变体之间不同；提交资源与物理批量另见下表。

脚本均不接受参数，并按当前代码和依赖解释对应 YAML。它们用于提供可读、固定的提交入口，不保证与最初实验环境兼容。

## 脚本与计划

| 脚本 | 唯一 YAML | 提交资源 | 物理批量 × 累积次数 = 全局批量 |
| --- | --- | --- | ---: |
| [`full_smi_ted_all_foreground_oracle_a800.sh`](full_smi_ted_all_foreground_oracle_a800.sh) | [`full_smi_ted_all_foreground_oracle_a800.yaml`](../../../../configs/experiments/batch1/full_smi_ted_all_foreground_oracle_a800.yaml) | 1 张 A800、24 核 CPU | 16 × 2 = 32 |
| [`full_smi_ted_all_foreground_detached_a800.sh`](full_smi_ted_all_foreground_detached_a800.sh) | [`full_smi_ted_all_foreground_detached_a800.yaml`](../../../../configs/experiments/batch1/full_smi_ted_all_foreground_detached_a800.yaml) | 1 张 A800、24 核 CPU | 16 × 2 = 32 |
| [`full_smi_ted_all_foreground_joint_a100.sh`](full_smi_ted_all_foreground_joint_a100.sh) | [`full_smi_ted_all_foreground_joint_a100.yaml`](../../../../configs/experiments/batch1/full_smi_ted_all_foreground_joint_a100.yaml) | 1 张 A100、16 核 CPU | 8 × 4 = 32 |

## 依赖与阅读顺序

先核对上表中的身份损失与资源，再阅读唯一 YAML；随后阅读 [`train_experiment.sh`](../../train_experiment.sh) 了解统一训练入口。提交时直接复制目标脚本头部命令，不要向脚本追加参数。
