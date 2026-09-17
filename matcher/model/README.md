# model：前景二分类与 PDB 内闭集 SMILES 匹配

模型接收一个 Stage1 候选和该 PDB 的受支持 SMILES 集合，返回：

- `foreground_logit (B,)`：候选是否为五类真实配体前景；
- `identity_logits (B,I)`：PDB 内给定候选 SMILES 集合的闭集匹配分数；
- 有 U-Net 的变体另返回两个 `(B,48,48,48)` 区域辅助 logit。

五种配体类别仅保留在真值和报告中。模型没有六分类头，也不会把类别概率拼入身份 PairScorer。

## 文件职责

| 文件 | 职责 |
|---|---|
| `config.py` | 限定四个且仅四个具名变体 |
| `candidate_encoder.py` | 编码候选并产生单一前景 logit 与可选区域 logit |
| `ligand_matcher.py` | 编码 SMI-TED、SMILES 图和同 PDB 身份清单，并计算闭集分数 |
| `fusion.py` | token 投影、mean/max 与候选条件注意力摘要 |
| `graph_blocks.py` | SMILES 图消息传递 |
| `raunet.py` | 三维体素主干 |

## 四个正式变体

| 变体 | 候选侧 | 身份侧 |
|---|---|---|
| `strongest` | 106 通道 U-Net；A/P/V/PP；mean/max 与候选条件注意力摘要拼接 | SMI-TED native、SMILES 图增量、同 PDB 身份清单增量 |
| `density_blob_only` | 单实验密度 U-Net；V/blob token 和 V/裁剪统计 | 与 `strongest` 相同 |
| `pocket_only` | 50 维 A-L0 token 的掩码 mean/max | 与 `strongest` 相同 |
| `trunk_only` | 与 `strongest` 相同 | 只有 SMI-TED native PairScorer |

图和清单增量门固定从 0 初始化，但训练中可学习。`trunk_only` 不构造图编码器、图注意力、清单注意力、增量打分器或门参数，不是将这些模块的输出乘零。

## 身份分数

启用全部身份分支时，总分为：

```text
native语言分数 + graph_gate × 图增量 + inventory_gate × 同PDB身份清单增量
```

补零身份槽写为当前 dtype 的最小有限值。模型只消费数值张量；身份字符串只用于 Dataset 建轴、报告和可追溯输出。
