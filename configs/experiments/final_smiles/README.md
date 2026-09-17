# final_smiles：四项正式训练配置

本目录只允许四个 YAML：`strongest`、`density_blob_only`、`pocket_only`、`trunk_only`。它们共享 manifest、seed 3409、AdamW、学习率 `1e-4`、名义 global batch 32、BF16、裁剪 200、validation 选模和 W&B 分组。

唯一结构变量是 `model.variant`。`pocket_only` 不存在区域头，因此两个区域损失显式为 0；其余三项均为 0.10。禁止通过新增布尔开关从这些 YAML 组合第五种模型。

阶段 E 启动前必须把四份 YAML 的 `data.manifest_path` 冻结到同一份已经闭合的 train/validation manifest。物理 batch 32 若 OOM，可以从零依次改成 `16×2`、`8×4`，并在单任务记录中保存实际组合；不能从部分更新状态续训。

