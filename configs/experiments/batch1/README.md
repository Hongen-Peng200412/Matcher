# Matcher 第 1 批实验配置

本目录定义三个五类非背景身份 Matcher。三者均从零开始、使用 SMI-TED Light 289M、固定 `[16,64)³`、mean/max 聚合及相同 12 维候选手动特征，只改变 `identity_variant`。`local_crop_strategy=fixed_center` 与 `use_sampling_features=false` 使其保持第 2 批以前的几何语义；开发和选模只读取 train/validation，配置不会构造或读取 calibration。

| 配置 | GPU | 物理批量 × 累积 | 身份变体 |
|---|---|---:|---|
| `full_smi_ted_all_foreground_oracle_a800.yaml` | A800 | 16 × 2 | oracle |
| `full_smi_ted_all_foreground_detached_a800.yaml` | A800 | 16 × 2 | detached |
| `full_smi_ted_all_foreground_joint_a100.yaml` | A100 | 8 × 4 | joint |

三者的 global batch 都是 32。A100 与 A800 的候选序列都先按同一 `training.seed + epoch` 全局洗牌，再截到 32 的整倍数，因此硬件批量差异不改变每轮候选集合或顺序。

## 科学字段

`identity_scope=all_non_background` 表示身份清单同时包含 `ion`、`nucleotide_like`、`peptide_like`、`small_molecule` 和 `sugar`；背景不进入身份损失。真实候选缺少当前语言表示或配体图时保留该候选，只令 `identity_target=-1`。

`use_manual_features=true` 使候选聚合读取 12 项固定顺序的 float32 统计：`A_count/A_probability_mean/A_probability_max/P_count/P_probability_mean/P_probability_max/V_count/V_probability_mean/V_probability_max/PP_count/PP_probability_mean/PP_probability_max`。`count` 的单位是有效 token 个数，进入模型时仅该位置执行 `log1p`；概率是无量纲 Stage1 固化原值，均值和最大值不再归一化。A/P 统计完整 `80³` token，V/PP 统计本批固定 `[16,64)³` 视野；空模态的三项值均为 0。每组三维统计在融合输入中紧邻其对应模态的 mean/max 摘要；未增强的候选条件注意力则紧邻对应模态的注意力增量。身份兼容性 MLP 还读取同一 12 维统计和 `stopgrad(softmax(class_logits)[identity_class])`。

`oracle`、`detached`、`joint` 的身份交叉熵系数分别是 1、切断梯度的目标类别概率、保留梯度的目标类别概率。三者都除以 global batch 内具有有效身份目标的真实前景候选数。

## W&B 曲线树

三项任务进入 project `AdaLigand-Matcher-Batch1`、group `matcher-batch1-five-class-manual-features`。标量按下列路径拆分：

- `loss/*`：train step、train epoch 与 validation 的总损失和四个分项。
- `gradient/*`、`gate/*` 与 `learning_rate/*`：裁剪前梯度、身份分支梯度、两个门值和学习率。
- `performance/*`、`memory/*` 与 `statistics/*`：DataLoader 等待、GPU 计算、吞吐、墙钟、显存和分母。
- `classification_summary/*`、`classification_per_class/*` 与 `classification_confusion/*`：六分类汇总、逐类 precision/recall/F1/分母及 36 项混淆计数。
- `identity_configured/*`：参与选模的五类总体身份指标及 seen/unseen。
- `identity_small_molecule/*`、`identity_all_non_background/*` 与五个 `identity_<class>/*`：第 0 批兼容子集、五类总体和逐类身份指标。
- `area/*` 与 `selection/*`：区域指标和选模状态。

W&B 自动工作区默认按指标名第一个 `/` 前缀建立区块，因此职责名称放在路径第一层；候选预训练不写身份分支、身份指标或门值的空面板。完整嵌套报告仍逐 epoch 写入 `history.jsonl`，在线曲线不替代 JSON 证据。

## 身份指标口径

`intrinsic` 忽略六分类预测，在对应身份清单内评价 Top-K。`conditional` 只统计六分类精确判对真实子类的候选，用于诊断 ion、sugar 等中间辅助类别是否正确。五类总体和五个逐类 `end_to_end` 只把六分类的背景/任一前景判断作为门：预测为任一前景类且 Matcher 选中真实身份即为成功，不要求六分类子类精确。逐类预测阳性由 Matcher 最终 Top-1 身份所属类别归桶，因此五个逐类预测阳性数与真阳性数都能分别加和到五类总体。真实前景但 `identity_target=-1` 的候选不进入这些身份分母；背景误报为前景且存在身份清单时计入端到端假阳性。

`identity_small_molecule/*` 为了与第 0 批直接比较，仍使用只含小分子身份的清单，并要求六分类预测为 `small_molecule` 才通过端到端门。它是在五类模型的一次前向结果上屏蔽非小分子身份槽得到的同清单、同分母子报告；inventory 分支仍看过五类共享清单，因此不能把它解释为另做一次旧 `identity_scope=small_molecule` 模型前向。该兼容报告不替代新的五类总体口径。
