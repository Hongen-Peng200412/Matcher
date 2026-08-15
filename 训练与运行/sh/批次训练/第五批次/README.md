# 第五批次固定入口

本目录保存第五批次六份 `oracle` 完整 Matcher 训练入口。四份 A100 计划比较身份图分支与清单分支的初始门控；两份 A800 计划比较梯度裁剪阈值。其余训练差异应以对应 YAML 为准。

脚本均不接受参数，并按当前代码和依赖解释对应 YAML。它们用于提供可读、固定的提交入口，不保证与最初实验环境兼容。

## 脚本与计划

| 脚本 | 唯一 YAML | 提交资源 | 物理批量 × 累积次数 = 全局批量 |
| --- | --- | --- | ---: |
| [`batch5_smi_ted_v_centroid_gate_002_oracle_a100.sh`](batch5_smi_ted_v_centroid_gate_002_oracle_a100.sh) | [`batch5_smi_ted_v_centroid_gate_002_oracle_a100.yaml`](../../../../configs/experiments/batch5/batch5_smi_ted_v_centroid_gate_002_oracle_a100.yaml) | 1 张 A100、16 核 CPU | 8 × 4 = 32 |
| [`batch5_smi_ted_v_centroid_gate_010_oracle_a100.sh`](batch5_smi_ted_v_centroid_gate_010_oracle_a100.sh) | [`batch5_smi_ted_v_centroid_gate_010_oracle_a100.yaml`](../../../../configs/experiments/batch5/batch5_smi_ted_v_centroid_gate_010_oracle_a100.yaml) | 1 张 A100、16 核 CPU | 8 × 4 = 32 |
| [`batch5_smi_ted_v_centroid_graph_gate_002_oracle_a100.sh`](batch5_smi_ted_v_centroid_graph_gate_002_oracle_a100.sh) | [`batch5_smi_ted_v_centroid_graph_gate_002_oracle_a100.yaml`](../../../../configs/experiments/batch5/batch5_smi_ted_v_centroid_graph_gate_002_oracle_a100.yaml) | 1 张 A100、16 核 CPU | 8 × 4 = 32 |
| [`batch5_smi_ted_v_centroid_inventory_gate_002_oracle_a100.sh`](batch5_smi_ted_v_centroid_inventory_gate_002_oracle_a100.sh) | [`batch5_smi_ted_v_centroid_inventory_gate_002_oracle_a100.yaml`](../../../../configs/experiments/batch5/batch5_smi_ted_v_centroid_inventory_gate_002_oracle_a100.yaml) | 1 张 A100、16 核 CPU | 8 × 4 = 32 |
| [`batch5_smi_ted_v_centroid_clip_5_oracle_a800.sh`](batch5_smi_ted_v_centroid_clip_5_oracle_a800.sh) | [`batch5_smi_ted_v_centroid_clip_5_oracle_a800.yaml`](../../../../configs/experiments/batch5/batch5_smi_ted_v_centroid_clip_5_oracle_a800.yaml) | 1 张 A800、24 核 CPU | 16 × 2 = 32 |
| [`batch5_smi_ted_v_centroid_clip_200_oracle_a800.sh`](batch5_smi_ted_v_centroid_clip_200_oracle_a800.sh) | [`batch5_smi_ted_v_centroid_clip_200_oracle_a800.yaml`](../../../../configs/experiments/batch5/batch5_smi_ted_v_centroid_clip_200_oracle_a800.yaml) | 1 张 A800、24 核 CPU | 16 × 2 = 32 |

## 依赖与阅读顺序

先根据要比较的门控或梯度裁剪阈值选择脚本，再核对唯一 YAML 与资源；随后阅读 [`train_experiment.sh`](../../train_experiment.sh)。提交时直接复制目标脚本头部命令，不要向脚本追加参数。
