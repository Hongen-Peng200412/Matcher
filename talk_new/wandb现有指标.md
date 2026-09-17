# Matcher 现有 W&B 指标：定义、分母与阅读方法

本文说明当前 Matcher 训练任务显式写入 Weights & Biases（W&B）的指标。重点不是罗列曲线名称，而是回答以下问题：

- 指标在参数更新后记录，还是在完整训练轮次和验证结束后记录；
- 指标的计算对象、分子、分母和特殊值分别是什么；
- 指标描述训练目标、最终任务表现，还是只用于排查运行状态；
- 哪些指标真正决定最优检查点，哪些指标完全不参与选模；
- 第一名配置中的具体数值应该怎样解释。

本文以当前总结果中的第一名配置为贯穿示例：联合身份 v4、摘要拼接、零门初始化、梯度裁剪阈值 200、随机种子 3409、学习率 1e-4、A100。对应配置见[第一名 YAML](../configs/experiments/joint_identity_v4/joint_identity_v4_concat_zero_gate_clip_200_seed_3409_lr_1e4_oracle_a100.yaml)，对应结果见[第一名单任务记录](../文档/记录/单任务记录/联合身份v4/摘要拼接/seed3409_lr1e-4_a100_摘要拼接_零门clip200.md)。

## 1. 先记住：真正决定 BEST 的只有三个验证指标

完整 Matcher 的选择分数为：

$$
S_{\text{selection}}
=
\frac{
F1_{\text{six-class macro}}
+A_{\text{identity conditional Top-1}}
+F1_{\text{identity end-to-end}}
}{3}
$$

对应 W&B 路径是：

1. `classification_summary/macro_f1`；
2. `identity_configured/conditional/top1`；
3. `identity_configured/end_to_end/f1`。

三项等权平均得到 `selection/score`。只有当前分数严格大于历史 `selection/best_score` 时，`selection/improved` 才为真，当前检查点才会写成 `BEST.pt`。损失、Dice、梯度、门值、吞吐和显存都不直接参与 BEST 选择。

第一名 BEST validation 的三个组成项是：

| 组成项 | 数值 |
|---|---:|
| 六分类 macro-F1 | 0.5336984208274029 |
| 配置身份范围 conditional Top-1 | 0.8935264054514480 |
| 配置身份范围 end-to-end F1 | 0.6899871630295250 |

因此：

$$
S_{\text{selection}}
=
\frac{0.5336984208274029+0.8935264054514480+0.6899871630295250}{3}
=0.7057373297694586
$$

这正是第一名在 epoch 11 保存 BEST 时的选择分数。最后完成的 epoch 13 分数是 0.6961865478267635，并没有覆盖 epoch 11 的 BEST。

## 2. W&B 中的两种时间轴

### 2.1 `train_step`：每次参数更新记录一次

第一名配置为：

- 单次物理 batch：8 个候选；
- 梯度累积：4 个物理 batch；
- 名义 global batch：32 个候选；
- `log_every_global_steps=1`：每次参数更新都记录。

因此，一个 `global_step` 不是“读取一个物理 batch”，而是累计 4 个物理 batch 的梯度，随后执行一次 AdamW 参数更新。第一名的一般更新含 32 个候选；数据尾部是否形成不足 32 的最后一组，取决于数据加载器的实际批次边界。

所有 `*/train_step/*` 指标都以 `global_step` 为横轴。当前写入入口见 [engine.py](../matcher/training/engine.py) 第 1759 行附近。

### 2.2 `train_epoch` 和 `validation`：每个完整 epoch 记录一次

每个 epoch 完成后，程序按以下顺序工作：

1. 汇总完整训练 epoch；
2. 对完整 validation 划分执行推理；
3. 计算 selection score；
4. 用 selection score 更新学习率调度器；
5. 判断是否保存新的 BEST；
6. 把训练汇总、validation 指标和选模状态写入同一条 W&B epoch 记录。

epoch 记录不受 `log_every_global_steps` 限制。每个完整 epoch 都会写一次，见 [tracking.py](../matcher/training/tracking.py) 中 `log_epoch`，以及 [engine.py](../matcher/training/engine.py) 第 3154 行附近的指标组装代码。

### 2.3 三种同名损失不能直接当成同一采样粒度

- `loss/train_step/*`：本次参数更新所累计候选的损失；
- `loss/train_epoch/*`：完整训练划分在本 epoch 的汇总损失；
- `loss/validation/*`：完整 validation 划分的汇总损失。

比较训练和 validation 时，应优先比较 `train_epoch` 与 `validation`，不要用某一个噪声较大的 `train_step` 去代表完整训练集。

## 3. 损失指标：`loss/*`

训练步、训练 epoch 和 validation 都包含五项：

| 名称后缀 | 定义 |
|---|---|
| `total` | 四项任务损失按配置权重相加后的总损失 |
| `classification` | 六分类交叉熵 |
| `ligand_area` | 配体区域的平衡二元交叉熵加软 Dice 损失 |
| `binding_area` | 结合区域的平衡二元交叉熵加软 Dice 损失 |
| `identity` | PDB 内联合身份分类交叉熵 |

五项后缀在三个层级展开成以下 15 个实际路径：

```text
loss/train_step/total
loss/train_step/classification
loss/train_step/ligand_area
loss/train_step/binding_area
loss/train_step/identity

loss/train_epoch/total
loss/train_epoch/classification
loss/train_epoch/ligand_area
loss/train_epoch/binding_area
loss/train_epoch/identity

loss/validation/total
loss/validation/classification
loss/validation/ligand_area
loss/validation/binding_area
loss/validation/identity
```

第一名的权重为：

$$
L_{\text{total}}
=L_{\text{classification}}
+0.1L_{\text{ligand area}}
+0.1L_{\text{binding area}}
+L_{\text{identity}}
$$

所以不能仅看四项原始损失谁的数值大，就断言谁对参数更新“更重要”。必须同时考虑配置权重、有效样本分母以及各项梯度的尺度。

### 3.1 `classification`：六分类交叉熵

六个类别的固定顺序是：

1. `background`：背景；
2. `ion`：离子；
3. `nucleotide_like`：核苷酸样配体；
4. `peptide_like`：肽样配体；
5. `small_molecule`：小分子；
6. `sugar`：糖类。

第一名没有设置 `class_weights`，因此每个候选在分类交叉熵中的基础权重相同。这不意味着每个类别贡献相同：候选数多的类别仍会贡献更多样本项。

### 3.2 `ligand_area` 和 `binding_area`：平衡 BCE 加软 Dice

两项区域损失使用相同结构，只是监督掩码不同。对每个候选分别计算：

1. 正体素的二元交叉熵均值；
2. 负体素的二元交叉熵均值；
3. 软 Dice 损失。

若候选存在正体素，正、负体素 BCE 各占 BCE 部分的一半，避免海量负体素完全压过稀少正体素。若没有正体素，则只使用负体素 BCE。软 Dice 损失为：

$$
L_{\text{soft Dice}}
=1-
\frac{2\sum_i p_i y_i+1}
{\sum_i p_i+\sum_i y_i+1}
$$

其中，$p_i$ 是 sigmoid 后的预测概率，$y_i$ 是 0 或 1 的目标。分子和分母中的 1 是平滑项，用于避免空集合或极小区域带来的数值不稳定。实现见 [losses.py](../matcher/training/losses.py)。

因此，`loss/*/ligand_area` 并不是 `area/ligand/micro_dice` 的负数，二者不能直接相减或换算：前者含平衡 BCE、软概率和逐候选处理；后者使用阈值化后的验证预测，并在整个 validation 上计算 micro-Dice。

### 3.3 `identity`：PDB 内联合身份交叉熵

联合身份由以下二元组定义：

$$
(\text{object_key},\ \text{PreparedLigand.smiles})
$$

`object_key` 指定 LigandObject 模板图，顶层 `PreparedLigand.smiles` 指定语言模型公开接口所接收的 canonical non-isomeric SMILES。只有同时具有有效身份目标的候选才进入身份损失，分母可通过 `statistics/*/identity_eligible_count` 判断。

第一名使用 `oracle` 身份损失门控：

- 有身份监督的候选直接计算身份交叉熵；
- 不乘当前六分类模型对真实类别的置信度；
- 每个有效身份候选的乘数恒为 1。

这里的 `oracle` 不表示把正确身份提前输入模型，也不表示评估时使用真值。它只表示训练身份损失时，不让六分类置信度缩小该候选的身份梯度。

其他可用门控口径的区别是：

- `detached`：乘真实类别预测概率，但该概率从计算图分离，身份损失不会反向改变六分类概率；
- `joint`：乘真实类别预测概率，并允许身份损失通过该乘数向六分类分支反向传播。

### 3.4 怎样联合阅读损失曲线

建议至少同时显示：

- `loss/train_epoch/total` 与 `loss/validation/total`；
- `loss/train_epoch/classification` 与 `loss/validation/classification`；
- `loss/train_epoch/identity` 与 `loss/validation/identity`；
- `selection/score`。

常见模式如下：

- 训练与 validation 同时下降，selection 同时上升：学习方向与评价目标一致；
- 训练损失继续下降而 validation 损失上升：可能开始过拟合；
- identity loss 下降但端到端 F1 不升：身份排序可能改善，但完整系统仍被背景/前景门或六分类错误限制；
- total loss 下降但 selection 不升：可微优化目标与离散评价指标发生偏离，不能只凭 loss 决定训练更久；
- validation loss 略升但 selection 上升：不一定是坏事，因为 BEST 的正式目标是三项离散指标的均值，而不是 total loss。

## 4. 六分类指标：`classification_*`

### 4.1 汇总指标

| W&B 路径 | 定义 | 是否进入 selection |
|---|---|---|
| `classification_summary/accuracy` | 全部候选中六分类 argmax 正确的比例 | 否 |
| `classification_summary/macro_f1` | 对真实支持数大于零的类别等权平均 F1 | 是 |
| `classification_summary/macro_f1_supported_classes` | 与当前兼容口径的 `macro_f1` 相同 | 否，内容重复 |
| `classification_summary/macro_f1_all_classes` | 固定对全部六类等权平均；无支持类别按 F1=0 计 | 否 |

准确率为：

$$
\text{accuracy}
=\frac{\text{六分类预测正确候选数}}{\text{validation 全部候选数}}
$$

accuracy 会被样本数多的类别主导。如果背景候选很多，模型即使在稀少配体类别上很差，accuracy 仍可能看起来不错。

对类别 $c$：

$$
P_c=\frac{TP_c}{TP_c+FP_c},\qquad
R_c=\frac{TP_c}{TP_c+FN_c}
$$

$$
F1_c=\frac{2P_cR_c}{P_c+R_c}
$$

`macro_f1` 先为每个在当前 validation 中确实有真实候选的类别计算 $F1_c$，再对这些类别等权平均，不按候选数量加权。它比 accuracy 更重视少数类别，并且是 selection 的第一项。

`macro_f1_all_classes` 固定平均六个类别。如果某类在小型或特殊 validation 中没有真实候选，该类以 F1=0 参与平均，因此它可能低于 `macro_f1`。在完整 validation 含六类时，两者通常相同或非常接近。

### 4.2 每类指标

每个类别都写入以下五项：

```text
classification_per_class/<class_name>/precision
classification_per_class/<class_name>/recall
classification_per_class/<class_name>/f1
classification_per_class/<class_name>/support
classification_per_class/<class_name>/predicted_count
```

其中：

- `support`：真实属于该类的候选数，即 $TP+FN$；
- `predicted_count`：模型预测成该类的候选数，即 $TP+FP$；
- precision 低而 recall 高：模型倾向把其他类误报为该类；
- precision 高而 recall 低：模型对该类较保守，漏掉很多真实候选；
- F1 同时惩罚误报与漏报，但不能代替查看支持数。

例如，小分子类别路径为：

```text
classification_per_class/small_molecule/precision
classification_per_class/small_molecule/recall
classification_per_class/small_molecule/f1
classification_per_class/small_molecule/support
classification_per_class/small_molecule/predicted_count
```

### 4.3 6×6 混淆矩阵

W&B 把混淆矩阵拆成 36 个标量：

```text
classification_confusion/<真实类别>/<预测类别>
```

第一轴是真实类别，第二轴是预测类别。例如：

```text
classification_confusion/background/small_molecule
```

表示真实为背景、却被预测成小分子的候选数量。对角线是正确分类数，非对角线显示具体错误方向。这 36 个标量是 W&B epoch 指标数量看起来很多的主要原因之一。展开逻辑见 [engine.py](../matcher/training/engine.py) 第 2414 行附近。

## 5. 身份指标：`intrinsic`、`conditional` 与 `end_to_end`

身份指标最容易被误读。三个层次不是三个近义名字，而是逐层加入现实约束。

### 5.1 `intrinsic`：忽略六分类门的固有身份排序能力

典型路径：

```text
identity_configured/intrinsic/top1
identity_configured/intrinsic/topk
identity_configured/intrinsic/eligible_count
```

eligible 候选必须同时满足：

- 真实类别位于当前身份评价范围；
- 有有效联合身份目标；
- 正确联合身份存在于该 PDB 的身份槽中。

它不要求六分类预测正确，只问正确身份在 PDB 内候选身份轴上排第几。

$$
\text{intrinsic Top-1}
=\frac{\text{正确联合身份得分最高的候选数}}
{\text{intrinsic eligible count}}
$$

当前默认 `top_k=5`。若某个 PDB 只有 3 个有效身份槽，实际检查全部 3 个槽，但 W&B 名称仍是 `topk`。

第一名 BEST validation：

```text
identity_configured/intrinsic/top1 = 0.8847203274215553
```

含义是：忽略六分类是否正确时，约 88.47% 的有效候选把正确联合身份排在第一。

### 5.2 `conditional`：六分类已经正确时的身份排序能力

典型路径：

```text
identity_configured/conditional/top1
identity_configured/conditional/topk
identity_configured/conditional/eligible_count
```

在 intrinsic 条件之外，它还要求六分类 argmax 恰好等于真实类别。它回答：如果上一级六分类已经判对，身份模块能否找对具体联合身份？

$$
\text{conditional Top-1}
=\frac{\text{六分类正确且身份 Top-1 正确的候选数}}
{\text{六分类正确且身份有效的候选数}}
$$

第一名 BEST validation：

```text
identity_configured/conditional/top1 = 0.8935264054514480
```

它略高于 intrinsic Top-1，因为六分类错误的候选通常更困难，而这些候选已经被 conditional 分母排除。因此，conditional Top-1 高并不等于完整系统表现同样高，必须与 `conditional/eligible_count` 和端到端 F1 一起看。

### 5.3 `end_to_end`：完整系统的最终身份结果

路径：

```text
identity_configured/end_to_end/precision
identity_configured/end_to_end/recall
identity_configured/end_to_end/f1
identity_configured/end_to_end/true_positive_count
identity_configured/end_to_end/predicted_positive_count
identity_configured/end_to_end/actual_positive_count
```

定义为：

$$
\text{precision}
=\frac{\text{最终身份正确数}}
{\text{预测为身份阳性的候选数}}
$$

$$
\text{recall}
=\frac{\text{最终身份正确数}}
{\text{真实存在身份目标的候选数}}
$$

$$
F1=\frac{2PR}{P+R}
$$

三个计数正好是上述分子和分母：

- `true_positive_count`：最终身份预测正确数；
- `predicted_positive_count`：系统最终给出身份阳性预测的候选数；
- `actual_positive_count`：真实身份阳性候选数。

第一名 `identity_scope=all_non_background`。在这个总体口径下，六分类承担背景/前景门控：预测为五个前景类别中的任意一类即可通过门，随后由完整身份轴的 Top-1 决定具体身份，不要求六分类预测的前景子类等于真实前景子类。

第一名 BEST validation：

```text
identity_configured/end_to_end/f1 = 0.6899871630295250
```

它明显低于 intrinsic 和 conditional Top-1，差距来自完整系统额外承担的背景/前景误判、未产生有效身份阳性，以及身份 Top-1 排错。

### 5.4 `seen` 与 `unseen`：联合身份是否在训练正样本中出现

只在 `identity_configured/conditional/*` 下提供：

```text
identity_configured/conditional/seen/top1
identity_configured/conditional/seen/topk
identity_configured/conditional/seen/count
identity_configured/conditional/unseen/top1
identity_configured/conditional/unseen/topk
identity_configured/conditional/unseen/count
```

这里的 seen/unseen 判断对象是完整联合身份：

$$
(\text{object_key},\ \text{PreparedLigand.smiles})
$$

它不是判断 SMILES 字符串单独是否出现，也不是判断 `object_key` 单独是否出现。只有该联合键曾在训练集作为正身份目标出现，才属于 seen。

判断泛化时必须同时看 Top-1 和 count。若 unseen/count 很小，极高或极低的 unseen Top-1 都可能只是小分母波动。

## 6. 为什么身份指标会出现很多组

W&B 同时报告以下前缀：

```text
identity_configured/*
identity_small_molecule/*
identity_all_non_background/*
identity_ion/*
identity_nucleotide_like/*
identity_peptide_like/*
identity_sugar/*
```

除 `identity_configured` 的 seen/unseen 扩展外，每组一般都有 12 项：intrinsic 的 Top-1、Top-K 和支持数，conditional 的 Top-1、Top-K 和支持数，以及 end-to-end 的 precision、recall、F1 和三个计数。

### 6.1 `identity_configured`

它服从当前 YAML 的 `identity_scope`。第一名是 `all_non_background`，所以报告五类非背景候选的总体身份表现，并额外包含 seen/unseen 条件指标。

### 6.2 `identity_all_non_background`

它始终固定报告五类非背景总体，不随 YAML 的 `identity_scope` 改变。在第一名配置中，它与 `identity_configured` 的主要值应相同；`identity_configured` 还多出 seen/unseen。

### 6.3 五个逐类身份前缀

`identity_ion`、`identity_nucleotide_like`、`identity_peptide_like`、`identity_small_molecule` 和 `identity_sugar` 用于定位总体身份表现究竟被哪个类别拖累。

对逐类 end-to-end 指标，前景类别门要求六分类预测为该具体类别。例如小分子逐类报告要求预测类别确实为 `small_molecule`；这与 `all_non_background` 总体报告只要求预测为任意前景类不同。

### 6.4 当前 `identity_small_molecule/*` 存在命名覆盖

当前代码先把专用“小分子子清单报告”写到 `identity_small_molecule/*`，随后又把逐类别 `small_molecule` 报告写到相同前缀。后写入的值覆盖前值，写入顺序见 [engine.py](../matcher/training/engine.py) 第 2479 行附近。

两种口径并不完全相同：

- 专用小分子报告只在小分子身份槽中排名；
- 逐类别 small-molecule 报告仍在完整身份轴上排名，再筛选真实小分子候选。

因此，当前 W&B 最终显示的 `identity_small_molecule/*` 是后写入的逐类别口径。完整 validation JSON 仍分别保留 `small_molecule_identity` 和 `identity_per_class["small_molecule"]`。阅读 W&B 时不能把该前缀自动解释成“小分子子清单排名”。

第一名记录中的小分子逐类结果为：

| 指标 | 数值 |
|---|---:|
| intrinsic Top-1 | 0.7944954128440367 |
| conditional Top-1 | 0.8271334792122538 |
| end-to-end F1 | 0.7078651685393259 |

## 7. 区域指标：`area/*`

路径为：

```text
area/ligand/micro_dice
area/ligand/dice_denominator
area/binding/micro_dice
area/binding/dice_denominator
```

validation 时先用 `logit >= 0` 把区域预测转成布尔值，这等价于 sigmoid 概率至少 0.5。随后在所有 validation 候选、所有体素上统一累计交集、预测正体素数和真实正体素数：

$$
\text{micro-Dice}
=\frac{2|\text{prediction}\cap\text{target}|}
{|\text{prediction}|+|\text{target}|}
$$

它不是“每个候选先算 Dice，再对候选平均”。大区域候选含更多体素，因此对 micro-Dice 的影响更大。

`dice_denominator` 是：

$$
|\text{prediction}|+|\text{target}|
$$

它用来判断 Dice 是否建立在足够多的正体素上。如果分母为零，Dice 没有定义，W&B 应显示空值而不是伪造为 0 或 1。

第一名 BEST validation：

- ligand micro-Dice：0.29623600147634194；
- binding micro-Dice：0.06294697912110488。

binding Dice 的绝对基线很低，所以很小的绝对变化可能产生很大的相对百分比。它没有进入 selection，不能用相对涨幅替代完整任务结论。

## 8. 身份三分支贡献诊断：`identity_contribution/*`

路径为：

```text
identity_contribution/validation/multi_identity_eligible_count
identity_contribution/validation/nonzero_native_count
identity_contribution/validation/native_centered_l2_mean
identity_contribution/validation/graph_increment_centered_l2_mean
identity_contribution/validation/inventory_increment_centered_l2_mean
identity_contribution/validation/graph_to_native_ratio_mean
identity_contribution/validation/inventory_to_native_ratio_mean
identity_contribution/validation/combined_to_native_ratio_mean
```

这组指标衡量语言基线、LigandObject 图增量和同 PDB 身份清单增量对身份槽分数差异的尺度。

对一个至少有两个有效身份槽的候选，设一组槽分数为 $s_1,\ldots,s_I$。程序先减去槽内均值：

$$
\tilde{s}_i=s_i-\frac{1}{I}\sum_{j=1}^{I}s_j
$$

再计算中心化向量的 L2 范数。这样做是因为给全部身份槽同时加同一个常数不会改变排序，不应被当作有效身份贡献。

三个绝对范数的含义是：

- `native_centered_l2_mean`：语言基线造成的身份槽差异尺度；
- `graph_increment_centered_l2_mean`：已经乘 graph gate 后，图分支增量的差异尺度；
- `inventory_increment_centered_l2_mean`：已经乘 inventory gate 后，清单分支增量的差异尺度。

三个比值的分母都是语言基线中心化范数，只在该分母大于 1e-12 时计算：

- `graph_to_native_ratio_mean`：图增量相对语言基线的尺度；
- `inventory_to_native_ratio_mean`：清单增量相对语言基线的尺度；
- `combined_to_native_ratio_mean`：图增量与清单增量之和相对语言基线的尺度。

例如 `graph_to_native_ratio_mean=0.20` 表示图增量中心化范数平均约为语言基线的 20%，不表示图分支让准确率提高 20%。范数只描述大小，不描述方向：增量可能改善排序，也可能破坏排序。若 combined 比两个单独比值之和小，通常说明两条增量在身份槽方向上存在部分抵消。

支持数含义：

- `multi_identity_eligible_count`：有身份监督且至少有两个有效身份槽的候选数；
- `nonzero_native_count`：其中语言基线中心化范数足以作为比值分母的候选数。

这八项只用于 validation 诊断，不参与损失、selection 或 BEST 保存。

## 9. 门值指标：`gate/*`

路径为：

```text
gate/train_step/graph
gate/train_step/inventory
gate/train_epoch/graph
gate/train_epoch/inventory
```

身份得分可以概括为：

$$
s_{\text{identity}}
=s_{\text{language}}
+g_{\text{graph}}\Delta s_{\text{graph}}
+g_{\text{inventory}}\Delta s_{\text{inventory}}
$$

- graph gate：候选自身 LigandObject 图分数增量的可训练标量乘数；
- inventory gate：同一 PDB 其他身份组成的清单分数增量的可训练标量乘数。

“零门”是指两个门的初值都为 0.0。训练刚开始时，最终身份分数等于语言基线，图分支和清单分支尚不改变前向身份排序；但两个门本身仍可训练。只要对应门能获得非零梯度，训练就能把门从 0 推开，逐渐引入该分支。零门不是永久关闭分支，也不是删除分支。

门值没有 sigmoid 或区间裁剪，因此训练后可以小于 0，也可以大于 1。门值不能单独代表实际贡献：很小的门乘上很大的门前增量，仍可能产生明显影响；较大的门乘上接近零的增量，实际影响仍可能很小。应把 `gate/*` 与 `identity_contribution/*` 联合阅读。

`gate/train_step/*` 是每次参数更新后的门值；`gate/train_epoch/*` 是完整 epoch 结束时的门值快照，不是该 epoch 所有步门值的平均。

## 10. 梯度与裁剪指标：`gradient/*`

### 10.1 全模型裁剪前梯度范数

```text
gradient/train_step/preclip_norm
gradient/train_epoch/preclip_mean_norm
gradient/train_epoch/preclip_maximum_norm
```

`preclip_norm` 是一次参数更新中，所有已有梯度合并后的全模型 L2 范数，并且在梯度裁剪之前测量。若裁剪阈值为 $C$，裁剪缩放系数可推导为：

$$
\alpha=\min\left(1,\frac{C}{\|g\|_2}\right)
$$

- `preclip_norm <= C`：不发生实质裁剪，$\alpha=1$；
- `preclip_norm > C`：全部梯度按同一比例缩小。

第一名阈值是 200。在 6258 次参数更新中：

- 超过 200 的更新数：0；
- 实际裁剪比例：0%；
- 最大裁剪前范数：73.18346405029297；
- 最小缩放系数：1.0。

所以 clip200 在这条实际训练轨迹中没有改变任何一次参数更新。W&B 当前没有直接记录 `clip_trigger` 或缩放系数；上述次数和比例来自完整 `preclip_norm` 历史与阈值 200 的离线比较。

### 10.2 图分支和清单分支梯度

```text
gradient/train_step/graph_branch_norm
gradient/train_step/inventory_branch_norm
gradient/train_epoch/graph_branch_mean_norm
gradient/train_epoch/inventory_branch_mean_norm
```

这两项在全局裁剪之后测量，统计的是身份 PairScorer 中图分支参数和清单分支参数的梯度 L2 范数。它们不覆盖整个图编码器或所有注意力模块。

非零范数说明对应参数得到了训练信号；接近零可能表示门控、饱和、数据分母或计算路径使梯度很弱。但“获得梯度”不等于“改善 validation”，最终仍要看身份指标与 selection。

## 11. 可选注意力偏置诊断

启用候选 token 注意力偏置的旧配置可能出现 A、P、V、PP 四种模态各两项：

```text
gradient/train_step/<a|p|v|pp>_attention_bias_final_weight_postclip_norm
attention_bias/train_step/<a|p|v|pp>_final_weight_poststep_norm
```

- `...postclip_norm`：全局裁剪之后，最终注意力偏置权重的梯度范数；
- `...poststep_norm`：优化器完成本次更新之后，该权重参数本身的范数。

第一名配置的 `use_candidate_token_attention_bias=false`，因此不会出现这些指标。它们与第一名使用的摘要拼接 conditional-attention 不是同一个功能，不能据此评价摘要拼接模块。

## 12. 学习率、选模与早停

### 12.1 学习率指标

```text
learning_rate/train_step/current
learning_rate/train_epoch/used
learning_rate/train_epoch/next
```

- `current`：当前参数更新实际使用的第一个优化器参数组学习率；
- `used`：刚完成 epoch 所使用的学习率；
- `next`：本轮 validation 后，调度器为下一 epoch 决定的学习率。

当前调度器为 `ReduceLROnPlateau(mode="max")`，监控 `selection/score`。一个未改善 epoch 后，学习率乘 0.5，最低为 1e-7。因此可能看到：

```text
learning_rate/train_epoch/used = 1e-4
learning_rate/train_epoch/next = 5e-5
```

这表示刚结束的 epoch 用 1e-4 训练，下一 epoch 将用 5e-5，不表示本 epoch 中途发生了变化。

### 12.2 选模与早停指标

```text
selection/score
selection/best_score
selection/improved
selection/bad_validation_epochs
```

- `score`：当前 validation 的三项等权平均分；
- `best_score`：截至当前 epoch 的历史最高分；
- `improved`：当前分数是否严格高于此前历史最高分；
- `bad_validation_epochs`：连续多少个 epoch 没有创造严格更高的新分数。

第一名 `patience_bad_epochs=2`。epoch 11 创造 BEST，epoch 12 和 13 连续两次没有超过它，因此训练在 epoch 13 后自然早停。相等分数不算改善。

训练结束后的本地摘要还保存 `best_epoch`、`best_score`、`last_epoch`、`global_step` 和 `stopped_by_patience`。它们属于最终任务摘要，不是每一步都变化的科学曲线。

## 13. 候选裁剪与数据分布统计：`statistics/*`

### 13.1 身份有效候选数

```text
statistics/train_step/identity_eligible_count
statistics/train_epoch/identity_eligible_count
statistics/validation/identity_eligible_count
```

它表示真正进入身份损失或相应身份汇总的候选数。第一名名义 global batch 是 32，但其中背景候选、没有有效身份目标的候选或不在身份监督范围的候选不会进入身份损失，所以单步有效数可能明显小于 32。

第一名完整 validation：

```text
statistics/validation/sample_count = 6495
statistics/validation/identity_eligible_count = 1466
```

也就是说，validation 有 6495 个候选，其中 1466 个具备当前身份损失定义下的有效监督。身份指标内部还可能使用更严格分母，例如 conditional 继续要求六分类正确，所以应以各自的 `eligible_count` 为准。

### 13.2 裁剪中心与边界统计

```text
statistics/train_step/crop_boundary_fraction
statistics/train_step/crop_offset_l2_mean_voxels
statistics/train_step/crop_offset_linf_mean_voxels
```

- `crop_boundary_fraction`：本 global batch 中，裁剪区域触及允许边界的候选比例；
- `crop_offset_l2_mean_voxels`：裁剪中心相对参考中心偏移的欧氏距离均值，单位是体素；
- `crop_offset_linf_mean_voxels`：偏移三个坐标轴绝对值最大者的均值，单位是体素。

L2 同时反映多轴综合偏移，L∞ 反映最严重单轴偏移。边界比例持续异常升高可能意味着裁剪中心分布、空间范围或数据几何出现变化；它不是模型效果分数。

### 13.3 V 模态保留比例

```text
statistics/train_step/v_retained_fraction_mean
statistics/train_step/v_retained_fraction_minimum
```

它们表示 V 模态在空间裁剪后仍被保留的元素比例：

- `mean`：当前 global batch 的候选平均值；
- `minimum`：当前 global batch 中保留比例最低的单个候选。

均值正常而 minimum 长期接近零，说明只有少量极端候选几乎丢失 V 模态；二者同时降低则说明更广泛的数据裁剪问题。

### 13.4 样本总数

```text
statistics/train_epoch/sample_count
statistics/validation/sample_count
```

它们是本轮训练划分和完整 validation 实际处理的候选总数。比较两个实验前应先确认这些计数一致；否则即使指标路径相同，也可能不是同一评价集合。

## 14. 性能与显存指标

### 14.1 单次参数更新性能

```text
performance/train_step/data_wait_seconds
performance/train_step/compute_seconds
performance/train_step/samples_per_second
```

- `data_wait_seconds`：本次 global step 等待并取得梯度累积所需数据的墙钟时间；
- `compute_seconds`：前向、反向、梯度处理和参数更新的计算时间；
- `samples_per_second`：本次处理候选数除以数据等待时间与计算时间之和。

数据等待远高于计算时间，通常说明 CPU 数据准备、磁盘读取或 worker 供给是瓶颈；计算时间占主导则更可能是 GPU 模型计算瓶颈。

### 14.2 epoch 与 validation 性能

```text
performance/train_epoch/elapsed_seconds
performance/train_epoch/data_wait_seconds
performance/train_epoch/compute_seconds
performance/train_epoch/samples_per_second
performance/validation/elapsed_seconds
performance/validation/samples_per_second
```

epoch 吞吐用于比较训练配置开销，validation 吞吐用于比较完整推理开销。不同 GPU 类型、缓存状态、worker 数或数据位置会影响这些值，因此它们不是科学效果指标。

### 14.3 CUDA 显存

```text
memory/train_step/peak_cuda_allocated_bytes
memory/train_step/peak_cuda_reserved_bytes
memory/train_epoch/peak_cuda_allocated_bytes
memory/train_epoch/peak_cuda_reserved_bytes
```

- allocated：张量当前或历史峰值实际占用的 CUDA 内存；
- reserved：PyTorch CUDA 缓存分配器向驱动保留的内存，通常大于或等于 allocated。

峰值统计在 epoch 开始时重置一次。因此 `train_step/peak_*` 是“从本 epoch 开始到当前步为止的运行峰值”，不是隔离测得的单步峰值。`train_epoch/peak_*` 是整个 epoch 的最终峰值。字节可除以 $1024^3$ 转成 GiB。

## 15. 空值、NaN 与失败关闭

有些指标在数学上可能没有定义，例如：

- 某类 `support=0`，因此该类 recall 没有分母；
- 某身份范围 `eligible_count=0`，因此 Top-1 没有分母；
- 区域 `dice_denominator=0`；
- `nonzero_native_count=0`，无法计算相对语言基线的比值。

对已经明确由零支持数解释的无定义值，W&B 跟踪层写入空值，而不是把它伪造成 0。空值表示“没有可计算对象”，0 表示“存在计算对象但结果为零”，两者科学含义不同。

未被合法零分母解释的 NaN、任何无穷值或其他非有限数会触发失败关闭，而不是静默上传。这有助于防止数值异常被一条看似正常的曲线掩盖。

## 16. W&B 自动系统指标不是 Matcher 科学契约

除本文列出的显式 Matcher 指标外，W&B SDK 还可能自动记录：

- GPU 利用率、显存利用率、温度和功耗；
- CPU 利用率和系统内存；
- 磁盘、网络和进程状态。

这类 `system.*` 指标由 W&B SDK 和运行环境产生，实际字段会随 SDK 版本、机器和权限变化。它们适合判断 GPU 是否空闲、数据供给是否跟不上或是否接近显存上限，但不属于 Matcher 的稳定科学评价口径，也不参与 selection。

## 17. 第一名配置的完整阅读示例

第一名的关键事实如下：

| 项目 | 值 | 解释 |
|---|---:|---|
| 物理 batch | 8 | 单次前向与反向处理的候选数 |
| 梯度累积 | 4 | 累计四个物理 batch 后更新参数 |
| 名义 global batch | 32 | 一般每次 AdamW 更新对应的候选数 |
| 初始 graph/inventory gate | 0 / 0 | 开始时两个增量不改变身份分数，但门可学习 |
| 梯度裁剪阈值 | 200 | 实际最大范数 73.18，故未触发 |
| 学习率 | 1e-4 | 初始 AdamW 学习率 |
| BEST / last epoch | 11 / 13 | epoch 11 最优，连续两次未改善后停止 |
| 总参数更新数 | 6258 | W&B `train_step` 时间轴的最终步数 |
| BEST selection | 0.7057373297694586 | 三个正式验证指标等权平均 |
| last selection | 0.6961865478267635 | 最后一轮不等于最优轮 |
| validation 候选数 | 6495 | 完整验证集合大小 |
| validation 身份有效数 | 1466 | 具备身份损失有效监督的候选数 |
| 六分类 macro-F1 | 0.5336984208274029 | selection 第一项 |
| intrinsic identity Top-1 | 0.8847203274215553 | 忽略六分类门的身份排序能力 |
| conditional identity Top-1 | 0.8935264054514480 | 六分类已正确子集中的身份能力；selection 第二项 |
| end-to-end identity F1 | 0.6899871630295250 | 完整身份系统表现；selection 第三项 |
| ligand micro-Dice | 0.29623600147634194 | 不参与选模 |
| binding micro-Dice | 0.06294697912110488 | 不参与选模 |

正确结论不是“conditional Top-1 接近 0.9，所以整个模型有 90% 的最终身份准确率”。更准确的表述是：模型在六分类已经正确的身份有效子集中，联合身份 Top-1 为 89.35%；加入实际背景/前景门和最终阳性决策后，端到端身份 F1 为 69.00%。六分类 macro-F1 只有 53.37%，仍是完整系统的重要限制因素。

## 18. 建议在 W&B 建立的五组面板

### 面板 A：正式选模

```text
selection/score
selection/best_score
classification_summary/macro_f1
identity_configured/conditional/top1
identity_configured/end_to_end/f1
```

这组回答“哪个 epoch 和哪个实验真正更好”。

### 面板 B：训练与过拟合

```text
loss/train_epoch/total
loss/validation/total
loss/train_epoch/classification
loss/validation/classification
loss/train_epoch/identity
loss/validation/identity
learning_rate/train_epoch/used
learning_rate/train_epoch/next
```

这组回答“训练是否收敛、何时过拟合、学习率何时下降”。

### 面板 C：身份分支是否被使用

```text
gate/train_epoch/graph
gate/train_epoch/inventory
identity_contribution/validation/graph_to_native_ratio_mean
identity_contribution/validation/inventory_to_native_ratio_mean
identity_contribution/validation/combined_to_native_ratio_mean
gradient/train_epoch/graph_branch_mean_norm
gradient/train_epoch/inventory_branch_mean_norm
```

这组回答“门有没有离开初值、分支有没有梯度、实际增量相对语言基线有多大”。它仍不能单独证明分支有益，必须与正式选模面板对照。

### 面板 D：数据与数值稳定性

```text
statistics/train_step/identity_eligible_count
statistics/train_step/crop_boundary_fraction
statistics/train_step/v_retained_fraction_minimum
gradient/train_step/preclip_norm
memory/train_step/peak_cuda_allocated_bytes
```

这组用于发现身份监督稀疏、异常裁剪、V 模态严重丢失、梯度尖峰或显存增长。

### 面板 E：运行效率

```text
performance/train_step/data_wait_seconds
performance/train_step/compute_seconds
performance/train_step/samples_per_second
performance/train_epoch/samples_per_second
performance/validation/samples_per_second
```

这组回答“时间花在数据还是计算上”，不能用于判断科学效果。

## 19. 指标路径总表

下面的通配符表示一组稳定展开路径，不代表 W&B 中真的保存星号字符。

| 路径组 | 记录频率 | 用途 | 参与 selection |
|---|---|---|---|
| `loss/train_step/*` | 每次参数更新 | 当前 global batch 的五项损失 | 否 |
| `loss/train_epoch/*` | 每个 epoch | 完整训练轮次的五项损失 | 否 |
| `loss/validation/*` | 每个 epoch | 完整 validation 的五项损失 | 否 |
| `classification_summary/*` | 每个 epoch | 六分类整体表现 | 仅 `macro_f1` |
| `classification_per_class/<class>/*` | 每个 epoch | 六个类别各自的 precision、recall、F1 和计数 | 否 |
| `classification_confusion/<true>/<predicted>` | 每个 epoch | 36 个分类错误方向计数 | 否 |
| `identity_configured/intrinsic/*` | 每个 epoch | 忽略六分类门的身份排序 | 否 |
| `identity_configured/conditional/*` | 每个 epoch | 六分类已正确子集的身份排序及 seen/unseen | 仅总体 `top1` |
| `identity_configured/end_to_end/*` | 每个 epoch | 完整身份系统 precision、recall、F1 与计数 | 仅 `f1` |
| `identity_<scope>/*` | 每个 epoch | 固定总体或逐类身份诊断 | 否 |
| `area/*` | 每个 epoch | 配体区域与结合区域 micro-Dice | 否 |
| `identity_contribution/validation/*` | 每个 epoch | 三种身份得分来源的相对尺度 | 否 |
| `gate/train_step/*` | 每次参数更新 | 两个可训练门的即时值 | 否 |
| `gate/train_epoch/*` | 每个 epoch | 两个门的 epoch 末快照 | 否 |
| `gradient/train_step/*` | 每次参数更新 | 裁剪前全模型范数及裁剪后分支范数 | 否 |
| `gradient/train_epoch/*` | 每个 epoch | 梯度范数汇总 | 否 |
| `attention_bias/train_step/*` | 可选、每次参数更新 | 旧候选注意力偏置参数范数 | 否 |
| `learning_rate/train_step/current` | 每次参数更新 | 当前实际学习率 | 否 |
| `learning_rate/train_epoch/{used,next}` | 每个 epoch | 本轮与下一轮学习率 | 否 |
| `selection/*` | 每个 epoch | 当前分数、历史最优和早停状态 | `score` 是最终选模值 |
| `statistics/train_step/*` | 每次参数更新 | 身份有效数与裁剪数据分布 | 否 |
| `statistics/train_epoch/*` | 每个 epoch | 训练样本数与身份有效数 | 否 |
| `statistics/validation/*` | 每个 epoch | validation 样本数与身份有效数 | 否 |
| `performance/*` | 每步或每 epoch | 数据等待、计算时间和吞吐 | 否 |
| `memory/*` | 每步或每 epoch | CUDA allocated/reserved 峰值 | 否 |
| W&B 自动 `system.*` | SDK 自定频率 | 主机与 GPU 运行状态 | 否 |

按第一名配置计算：候选注意力偏置关闭时，每次参数更新通常显式写入约 22 个 Matcher 指标；每个完整 epoch 完成 validation 后，展开分类、混淆矩阵、多个身份范围和诊断项，会写入约 200 个 Matcher 指标。数量多不代表有 200 个选模目标：正式 BEST 仍只由三个 validation 指标的等权平均决定。

## 20. 最简阅读原则

如果只想判断实验好坏，先看 `selection/score` 及其三个组成项；如果想理解为什么好或坏，再按顺序查看：

1. 六分类逐类 F1 与混淆矩阵，定位类别瓶颈；
2. intrinsic、conditional 和 end-to-end 的差距，区分身份排序问题与门控问题；
3. seen/unseen 及其 count，判断联合身份泛化；
4. gate、贡献范数和分支梯度，判断图与清单分支是否真正介入；
5. 区域 Dice、采样统计、梯度、吞吐和显存，排查辅助任务、数据与运行异常。

任何比例都要连同分母阅读；任何门值都要连同门后增量阅读；任何最后一轮结果都不要自动当成 BEST 结果。
