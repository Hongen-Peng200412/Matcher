# model：前景二分类与 PDB 内闭集 SMILES 匹配

模型接收一个 Stage1 候选和该 PDB 的受支持精确 SMILES 集合，输出候选前景分数与闭集身份分数。五种配体类别只保留在真值和报告中；模型没有六分类头，也不把类别概率拼入身份打分器。

## 推荐阅读顺序

1. `config.py`：先理解具名变体与共同结构参数。
2. `fusion.py`：理解变长 token 的 mean/max 和候选条件注意力摘要。
3. `raunet.py`：理解三维体素主干与区域辅助输出。
4. `candidate_encoder.py`：理解各变体如何选择输入并产生候选表示和前景 logit。
5. `ligand_matcher.py`：理解 SMI-TED native 分支、SMILES 图消息传递、同 PDB 身份清单增量和最终闭集分数。图消息传递实现位于本文件，不存在独立的 `graph_blocks.py`。
6. `__init__.py`：查看目录公开入口。

## 模型输出

| 字段 | 形状 | 含义与特殊值 |
| --- | --- | --- |
| `candidate_embedding` | `(B,D_m)` | 候选体素与当前变体启用 token 融合后的候选级表示。 |
| `foreground_logit` | `(B,)` | 候选是否属于五类真实配体的单一 logit。 |
| `identity_logits` | `(B,I_max)` | 当前 PDB 闭集 SMILES 身份分数；`identity_mask=false` 的补齐槽写为当前 dtype 的最小有限值。 |
| `ligand_area_logits` | `(B,48,48,48)` 或 `None` | 启用区域监督的 U-Net 变体返回配体区域辅助 logit；`blob_only`、`density_only`、`density_only_106` 与 `pocket_only` 返回 `None`。 |
| `binding_area_logits` | `(B,48,48,48)` 或 `None` | 启用区域监督的 U-Net 变体返回结合区域辅助 logit；`blob_only`、`density_only`、`density_only_106` 与 `pocket_only` 返回 `None`。 |
| `voxel_features` | `(B,C_v,48,48,48)` 或 `None` | 有 U-Net 的变体返回最高分辨率体素特征；`blob_only` 与 `pocket_only` 返回 `None`。 |
| `identity_score_native` | `(B,I_max)` | 未屏蔽的语言基线分数；补零槽不得读取。 |
| `identity_score_graph_increment` | `(B,I_max)` | 已乘图门的未屏蔽图增量；`trunk_only` 为零。 |
| `identity_score_inventory_increment` | `(B,I_max)` | 已应用多身份条件并乘清单门的未屏蔽增量；`trunk_only` 为零。 |

身份字符串不进入模型；Dataset 用它建立轴，模型只消费语言向量、图张量、掩码和索引。

## 具名变体

| 变体 | 候选侧 | 身份侧 |
| --- | --- | --- |
| `strongest` | 106 通道 U-Net；A/P/V/PP；mean/max 与候选条件注意力摘要拼接。A、P、PP 各有一个从 0.01 开始的独立直接标量门，门同时作用于投影 token 和该模态三项统计；V 不加门。 | SMI-TED native、SMILES 图增量、同 PDB 身份清单增量。 |
| `blob_only` | 只使用 V/blob token、V 三项统计和四项裁剪统计；不建立 U-Net、体素调制器或区域头。 | 与 `strongest` 相同。 |
| `density_only` | 单实验密度 U-Net；最高分辨率特征执行空间 mean/max 并投影为候选表示；不建立 token 投影器、体素调制器、候选模态聚合器或区域头。 | 与 `strongest` 相同。 |
| `density_only_106` | 56 通道完整密度 bank 与 50 通道受体特征体素网格；RAUNet 摘要结构与 density_only 相同；不建立 token、人工统计或区域头。 | 与 `strongest` 相同。 |
| `density_blob_only` | 单实验密度 U-Net；V/blob token、V 三项统计和四项裁剪统计。 | 与 `strongest` 相同。 |
| `pocket_only` | 50 维 A-L0 token 的掩码 mean/max；无 U-Net 和区域头。 | 与 `strongest` 相同。 |
| `trunk_only` | 候选侧与 `strongest` 相同。 | 只有 SMI-TED native 打分器；不实例化图和清单增量模块。 |

## 配置字段

| 字段 | 类型 | 作用与约束 | 示例值 |
| --- | --- | --- | --- |
| `variant` | `str` | 具名变体之一，同时决定候选输入和身份增量模块。 | `"strongest"` |
| `model_dim` | 正整数 | 候选和身份共同表示宽度，必须能被 `attention_heads` 整除。 | `256` |
| `attention_heads` | 正整数 | 候选条件、图和身份清单注意力头数。 | `8` |
| `dropout` | `float`，`[0,1)` | 模型内部丢弃概率。 | `0.1` |
| `voxel_feature_channels` | 正整数 | U-Net 最高分辨率输出宽度。 | `64` |
| `raunet_planes` | 九个正整数 | U-Net 编解码阶段的固定顺序通道宽度。 | `[64,256,256,256,256,256,128,64,64]` |
| `recycle_iterations` | 正整数 | U-Net 回收次数。 | `3` |
| `gradient_checkpointing` | `bool` | 训练时是否用激活重计算降低显存。 | `true` |
| `bottleneck_attention_layers` | 正整数 | U-Net 瓶颈注意力层数。 | `4` |
| `graph_layers` | 正整数 | SMILES 图消息传递层数；`trunk_only` 不实例化该模块，但 YAML 仍需提供公共结构字段。 | `4` |

## 身份分数

`strongest`、`blob_only`、`density_only`、`density_only_106`、`density_blob_only` 和 `pocket_only` 的总身份分数为：

```text
native 语言分数 + graph_gate × 图增量 + inventory_gate × 同 PDB 身份清单增量
```

两个门参数固定从 0 初始化并在训练中学习。`trunk_only` 只计算 native 语言分数，不通过将增量输出乘零来模拟消融，因此参数树中不存在图编码器、图注意力、清单注意力、增量打分器和两个门参数。
