# 第四批次固定入口

本目录保存第四批次三份完整 Matcher 训练入口。三项计划都从候选六分类模型初始化候选分支，并使用 SMI-TED 289M、候选 V 质心裁剪和 mean/max 聚合。模型与监督的科学设置仅在身份损失的 `oracle`、`detached` 和 `joint` 变体之间不同；提交资源与物理批量另见下表。

脚本均不接受参数，并按当前代码和依赖解释对应 YAML。YAML 中的候选模型初始化路径属于计划输入；当前环境是否具备该文件，由统一训练入口在执行时核验。因此这些入口不保证与最初实验环境兼容。

## 脚本与计划

| 脚本 | 唯一 YAML | 提交资源 | 物理批量 × 累积次数 = 全局批量 |
| --- | --- | --- | ---: |
| [`full_smi_ted_v_centroid_pretrained_mean_max_oracle_a100.sh`](full_smi_ted_v_centroid_pretrained_mean_max_oracle_a100.sh) | [`full_smi_ted_v_centroid_pretrained_mean_max_oracle_a100.yaml`](../../../../configs/experiments/batch4/full_smi_ted_v_centroid_pretrained_mean_max_oracle_a100.yaml) | 1 张 A100、16 核 CPU | 8 × 4 = 32 |
| [`full_smi_ted_v_centroid_pretrained_mean_max_detached_a800.sh`](full_smi_ted_v_centroid_pretrained_mean_max_detached_a800.sh) | [`full_smi_ted_v_centroid_pretrained_mean_max_detached_a800.yaml`](../../../../configs/experiments/batch4/full_smi_ted_v_centroid_pretrained_mean_max_detached_a800.yaml) | 1 张 A800、24 核 CPU | 16 × 2 = 32 |
| [`full_smi_ted_v_centroid_pretrained_mean_max_joint_a800.sh`](full_smi_ted_v_centroid_pretrained_mean_max_joint_a800.sh) | [`full_smi_ted_v_centroid_pretrained_mean_max_joint_a800.yaml`](../../../../configs/experiments/batch4/full_smi_ted_v_centroid_pretrained_mean_max_joint_a800.yaml) | 1 张 A800、24 核 CPU | 16 × 2 = 32 |

## 依赖与阅读顺序

先核对身份损失、资源及 YAML 中的候选模型初始化路径，再阅读 [`train_experiment.sh`](../../train_experiment.sh) 的初始化检查。提交时直接复制目标脚本头部命令，不要向脚本追加参数。
