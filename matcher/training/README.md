# training：四变体共用训练、选模与 checkpoint

正式入口是：

```bash
python -m matcher.training.experiment --config <final_smiles YAML>
```

四个 YAML 位于 `configs/experiments/final_smiles/`，除 `model.variant` 和 `pocket_only` 必须关闭区域损失外保持同一科学设置。

## 损失

```text
total = 1.0 × foreground_bce
      + ligand_area_weight × ligand_area_loss
      + binding_area_weight × binding_area_loss
      + 1.0 × smiles_identity_ce
```

`smiles_identity_ce` 只在 `foreground_target & (identity_target >= 0)` 上计算。它不乘前景预测概率，不依赖前景是否预测正确，也没有其它身份损失变体或历史兼容开关。

梯度累积按名义 global batch 的真实分母计算：前景和区域物理批次等大；身份 CE 按每个物理批次合格身份数占完整累积窗口合格数的比例缩放。因此资源回退到 `16×2` 或 `8×4` 不改变身份损失的定义。

## validation 与 BEST

每个完整 epoch 后运行完整 validation：

```text
selection = (foreground PRAUC + 真实前景候选上的 SMILES 匹配准确率) / 2
```

selection 严格改善时覆盖 `BEST.pt`；每轮覆盖 `LAST.pt`。连续 `patience_bad_epochs=2` 次未严格改善后早停。训练结束后重新载入 BEST 权重，只在 validation 上选择 F1 最大、并列时较高的前景阈值，并把阈值写回 BEST。F1 不参与选模。

## 运行产物

| 文件 | 内容 |
|---|---|
| `run.json` | YAML、manifest、Job、主机、GPU、Python、PyTorch 与 CUDA 来源绑定 |
| `history.jsonl` | 每轮训练、validation、selection 及最终阈值事件 |
| `BEST.pt` | 最优权重、模型配置、manifest SHA-256、validation 和前景阈值 |
| `LAST.pt` | 最后一轮权重、优化器和停止状态 |
| `summary.json` | 自然收口时的 BEST/LAST、更新数、停止原因和 W&B 链接 |

异常退出不伪造 `summary.json`。checkpoint 使用同目录临时文件同步后原子替换。
