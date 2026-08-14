# 第 2 批：V 质心视野与采样统计

本目录定义五个只使用 train/validation 的正式实验。三个完整 Matcher 固定使用 SMI-TED Light 289M、mean/max 聚合和五类非背景身份监督，只比较 `oracle/detached/joint`；两个候选预训练分别使用 mean/max 和未增强的候选条件交叉注意力。五项均从零开始，不读取第 1 批或第 0 批 checkpoint。

五份配置均显式设置 `use_candidate_token_attention_bias=false`。未增强候选条件交叉注意力只使用 query/key 内容分数；第 3 批才把实体相对坐标与 Stage1 概率转换成额外注意力分数。两个候选预训练和三个完整 Matcher 也都显式设置 `initial_graph_gate=0.0`、`initial_inventory_gate=0.0`；候选任务不构造身份分支，完整任务保持原有零增量初始化。

## 候选级 48³ 视野

`local_crop_strategy=v_centroid` 先对候选的完整权威 V 索引计算 `np.rint(mean(V_zyx)+0.5-24)`，其中半整数采用最近偶数规则；再逐轴限制到 `[0,32]`。同一个实际起点同时用于 105 通道体素、V 筛选与重编号、PP、配体区域目标和受体结合区域目标。A/P token 仍保留完整 `80³`，但相对 XYZ 原点改为新 `48³` 几何中心。候选—occurrence 双向覆盖率继续使用完整全局组件，不受局部视野影响。

体素缓存写入 `cache/voxel_input_v2/v_centroid/`；PP 缓存写入 `cache/pp_input_v2/v_centroid/top_1024/`。两类 v2 路径不会覆盖第 0/1 批的固定中心缓存。

## 手动特征

`use_manual_features=true` 保留第 1 批的 12 项统计：A/P/V/PP 各自的有效 token 数、Stage1 固化概率均值和最大值。A/P 基于完整 `80³`，V/PP 基于当前候选级 `48³`；空模态三项均为 0。仅计数在进入模型时执行 `log1p`，概率保持 Stage1 无量纲原值。

`use_sampling_features=true` 追加以下固定顺序的四项 float32 候选统计：

1. `crop_boundary_limited`：请求起点被 `80³` 边界限制时为 1，否则为 0。
2. `crop_center_to_v_centroid_l2_voxels`：实际裁剪中心到完整 V 体素中心质心的 L2 距离，单位 voxel。
3. `crop_center_to_v_centroid_linf_voxels`：同一偏移的 L∞ 距离，单位 voxel。
4. `v_retained_fraction`：新 `48³` 内 V 体素数除以完整 V 体素数，无量纲且位于 `[0,1]`。

模型分别用 `24√3` 和 `24` 归一化两项距离，标志与比例保持原值。四项统计紧邻 U-Net 候选摘要进入候选融合，并进入完整 Matcher 的 native、graph、inventory 三个最终身份打分 MLP。

## 实验矩阵与在线分组

| 配置 | 任务 | 硬件批量 | W&B group | 直接对照 |
| --- | --- | --- | --- | --- |
| `full_smi_ted_v_centroid_oracle_a100.yaml` | 完整 Matcher / oracle | A100 `8×4=32` | `matcher-batch2-full-v-centroid-sampling` | 第 1 批 oracle |
| `full_smi_ted_v_centroid_detached_a100.yaml` | 完整 Matcher / detached | A100 `8×4=32` | 同上 | 第 1 批 detached |
| `full_smi_ted_v_centroid_joint_a100.yaml` | 完整 Matcher / joint | A100 `8×4=32` | 同上 | 第 1 批 joint |
| `candidate_v_centroid_mean_max_a800.yaml` | 候选预训练 / mean/max | A800 `16×2=32` | `matcher-batch2-candidate-v-centroid-sampling` | 第 0 批 mean/max |
| `candidate_v_centroid_cross_attention_a800.yaml` | 候选预训练 / 未增强交叉注意力 | A800 `16×2=32` | 同上 | 第 0 批交叉注意力 |

五项使用 W&B project `AdaLigand-Matcher-Batch2`。完整 Matcher 还报告共同小分子子集、五类总体和逐类身份指标；候选预训练重点比较 validation 六分类 macro-F1、逐类指标与混淆矩阵。calibration 不在本批配置的开发或选择流程中。
