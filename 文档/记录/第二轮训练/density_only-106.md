# density_only-106 实验日志

本记录保存 `density_only-106` 的单实验追溯。代码模型变体为 `density_only_106`；候选侧只使用 56 通道完整密度 bank 与 50 通道受体原子特征体素网格，准确含义是“106 通道体素-only”。

## 当前状态

- 阶段：实际 H100 门禁已通过，正式从头训练稳定运行；首个训练区段和完整 validation 已完成。
- 正式资源：H100 Job `386151`，32 CPU，剩余 walltime 为无限；活动阶段计划使用 30 个非持久 DataLoader worker。
- 稳定实验标识：`density_only-106`；代码模型变体：`density_only_106`。
- 正式 YAML：`configs/experiments/final_smiles/density_only_106.yaml`。
- 冻结 train/validation manifest：`/storage/penghongen/Matcher/manifests/final_train_validation_shards01_06_v1/manifest.json`；SHA-256 `3551e4a7b111c9d51a718fe11586769f31a7a50f66b8d25839bb9742418ad247`。
- 候选输入：56 通道完整密度 bank 连接 50 通道受体原子特征体素网格，形状为 `(106,48,48,48)`。56 通道包含实验密度、模拟密度与受体掩码参与的派生通道；50 通道由 49 维 L0 与 `is_backbone` 构成。
- 模型边界：使用无末端多尺度卷积的 RAUNet；最高分辨率特征空间 mean/max 经投影直接形成候选表示；不读取 A/P/V/PP token、人工统计或区域标签，不建立体素调制器、候选模态聚合器或区域头。
- 损失权重：前景 `0.30`、ligand-area `0.00`、binding-area `0.00`、SMILES 身份 `1.00`。
- 选模公式：`(0.3 * foreground_PRAUC + supported_foreground_SMILES_accuracy) / 1.3`。
- 解释边界：本实验使用 H100，单通道 `density_only-0` 使用 A800；两者只作描述性性能比较，不能把差异严格归因于 1→106 通道。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_a67eb7f830e0/Matcher`；内容 SHA-256 `a67eb7f830e0a3625da349daf9481fe653f2231ba4d860bc9aa8a1101a760f87`；对应实现提交 `ab87ead233e08c70fb387bae081403ebcbb4eba7`。
- 门禁 launch：`/home/penghongen/Feedback/Matcher/launches/386151/gate_density_only106_job386151_20260920T230033`；报告：`/home/penghongen/Feedback/Matcher/gates/second_round/density_only-106/gate_density_only106_job386151_20260920T230033.json`，`passed=true`。
- 训练 launch：`/home/penghongen/Feedback/Matcher/launches/386151/train_density_only106_job386151_20260920T230728`。
- 运行根：`/storage/penghongen/Matcher/runs/density_only-106/train_density_only106_job386151_20260920T230728`。
- W&B：`https://wandb.ai/pencounkdual-111/Matcher/runs/jfdp3ifl`。
- 首次完整 validation：epoch 0、第 1 区段、`global_step=465`；前景 PRAUC `0.712084939914394`、SMILES 匹配准确率 `0.7569919204474829`、selection `0.7466287710936931`。该记录只用于确认正式运行稳定，不替代最终 BEST。

## 正式命令

`bash 训练与运行/sh/final_smiles/train_density_only_106.sh`

全测试集与 Stage3 小分子正式入口将在训练运行根冻结后绑定实际 BEST，并与门禁和只读核查命令分开记录。

## 测试、门禁和只读核查

本节只记录机械测试、实际 GPU 门禁和只读状态核查；不得把其中命令当作正式训练命令。本地完整测试为 `45 passed in 37.60s`。实际门禁绑定上述 release、正式 YAML、冻结 manifest、H100 GPU 和 30 个阶段互斥 worker；train/validation 数量为 `44643/5544`，epoch 批次计划 SHA-256 为 `f8bba32ecb6120ad61c83f3abc9f331729e69e7f7d337eec68fde5aa9b646053`，训练前后向、validation 前向与两个阶段的 worker 关闭检查均通过。

## 失败或中断尝试

- Job `386151` 原本执行 Pocket_Plus 的 Find_1_p 命令。用户明确授权复用后，先核对 `after_lock_386151` 存在且 pre/kill 不存在，再创建精确 `kill_lock_386151`。原命令退出后 `try_lock_386151` 恢复，kill 锁由 runner 清除，after 锁保留；未使用 `scancel`，未读取或修改其科学产物。
