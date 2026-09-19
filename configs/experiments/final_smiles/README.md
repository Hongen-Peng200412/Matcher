# final_smiles：四项正式训练配置

本目录只允许四个 YAML：`strongest`、`density_blob_only`、`pocket_only`、`trunk_only`。它们共享 manifest、seed 3409、AdamW、学习率 `1e-4`、物理 batch 32、梯度累积 1、BF16、裁剪 200、validation 选模和 W&B 分组。

唯一结构变量是 `model.variant`。`pocket_only` 不存在区域头，因此两个区域损失显式为 0；其余三项均为 0.10。禁止通过新增布尔开关从这些 YAML 组合第五种模型。

阶段 E 启动前必须把四份 YAML 的 `data.manifest_path` 冻结到同一份已经闭合的 train/validation manifest。本轮指定的 strongest 与 density_blob_only 都固定使用 `32×1`，不得在门禁失败后自动改变批量。

每个 epoch 的更新按顺序划为三个连续区段，每段后执行一次完整 validation。统一改善门槛是 `0.001`；连续第三次未改善触发平台动作，前两次平台把学习率乘以 `0.8`，第三次平台直接早停。训练最多 15 个 epoch、45 次完整 validation，并保留每次检查的 checkpoint。

