# density_only-0 实验日志

本记录保存 `density_only-0` 的单实验追溯。候选侧只读取 candidate-centered 单通道实验密度；身份侧使用 SMI-TED、SMILES 图和同 PDB 身份清单。

## 当前状态

- 阶段：正式训练已自然早停；全测试集与 Stage3 小分子冻结评估及绘图均已完成，等待本地结果归档。
- 正式资源：计划顺序复用 A800 数组任务 `379403_1`；锁与反馈目录使用基础编号 `379403`，活动阶段使用 15 个非持久 DataLoader worker。
- 稳定实验标识：`density_only-0`；代码模型变体：`density_only`。
- 正式 YAML：`configs/experiments/final_smiles/density_only.yaml`。
- 训练 release：`/home/penghongen/Feedback/Matcher/releases/Matcher_8c8f126c6d8b/Matcher`；内容 SHA-256 `8c8f126c6d8b69f9ea8471f15f6b3772969b59a3ded39ca14fa31140e520636a`。
- 门禁 launch/report：`gate_density_only0_job379403_20260920T204613`；`/home/penghongen/Feedback/Matcher/gates/second_round/density_only-0/gate_density_only0_job379403_20260920T204613.json`。
- 训练 launch：`train_density_only0_job379403_20260920T205216`。
- W&B：`https://wandb.ai/pencounkdual-111/Matcher/runs/3wfaiq6v`；运行根：`/storage/penghongen/Matcher/runs/density_only-0/train_density_only0_job379403_20260920T205216`。
- 冻结 train/validation manifest：`/storage/penghongen/Matcher/manifests/final_train_validation_shards01_06_v1/manifest.json`；SHA-256 `3551e4a7b111c9d51a718fe11586769f31a7a50f66b8d25839bb9742418ad247`。
- 候选输入：`experimental_density_48` 经 0.1%/99.9% 分位截断与全网格 z-score 后形成 `(1,48,48,48)`；不读取或构造 V、A、P、PP、模拟密度、受体栅格和候选人工统计。
- 模型边界：使用无末端多尺度卷积的 RAUNet；最高分辨率特征的空间 mean/max 经投影直接形成候选表示；不建立 token 投影器、体素调制器、候选模态聚合器或区域头。
- 损失权重：前景 `0.30`、ligand-area `0.00`、binding-area `0.00`、SMILES 身份 `1.00`。
- 选模公式：`(0.3 * foreground_PRAUC + supported_foreground_SMILES_accuracy) / 1.3`。
- 自然停止：epoch 12、第 38 次完整 validation 触发早停；BEST 位于 epoch 9、第 30 次 validation，selection `0.850683342024217`，冻结 validation 前景阈值 `0.52734375`。38 份逐检查 checkpoint 全部保留；BEST SHA-256 为 `7667d645dd871651af85283fcbae147344b3834c547ee4048c52ec3684015376`。
- 通用测试：launch `evaluate_full_density_only0_job379403_20260921T114259`；`frozen_test.json` 位于 `/storage/penghongen/Matcher/evaluation/final_smiles/density_only-0/frozen_test.json`，SHA-256 `cbfe0d18f03433f640806ed77ee6f1bdce6d4e472258a0101e0ad822dfa779a5`。
- 小分子测试：launch `evaluate_small_density_only0_job379403_20260921T115924`；绘图 launch `plot_small_density_only0_job379403_20260921T121457`；`evaluation.json` SHA-256 `4fd526617489104ef307254a3f8c1f96205ad3a20e5148aacdd4f238dfc25208`。

## 正式命令

`bash 训练与运行/sh/final_smiles/train_density_only.sh`

`bash 训练与运行/sh/final_smiles/evaluate_density_only.sh`

`bash 训练与运行/sh/small_molecule_test/evaluate_density_only.sh`

`bash 训练与运行/sh/small_molecule_test/plot_density_only.sh`

后三项已在 BEST 冻结后执行；它们绑定固定运行目录中的 `BEST.pt`，并与测试和门禁命令分开记录。

## 测试、门禁和只读核查

- `D:\Anaconda\python.exe -m pytest -q`：`42 passed in 32.88s`。
- `D:\Anaconda\python.exe -m compileall -q matcher ops`：通过。
- 两个活动训练脚本经 Git Bash `bash -n`：通过。
- `density_only.yaml` 与 `pocket_only.yaml` 经 `load_smiles_experiment()` 严格解析：实验标识、模型变体、15 个 worker、损失权重和 patience 均符合第二轮协议。
- 面向本轮拥有文件的 `git diff --check`：通过；用户已有的三个未提交 Python 文件不属于本轮改动或提交范围。

- 实际 A800 门禁：`matcher_second_round_gpu_gate_v1`，`passed=true`；GPU `NVIDIA A800-SXM4-80GB`，15 个 worker，train/validation 数为 `44643/5544`，epoch 批次计划 SHA-256 为 `f8bba32ecb6120ad61c83f3abc9f331729e69e7f7d337eec68fde5aa9b646053`。
- 门禁单个训练批次完成 BF16 前向、反向、梯度裁剪与优化器更新；裁剪前梯度范数 `58.76240921020508`，损失有限。单个 validation 批次前向损失有限。训练与 validation loader 关闭后均无残留 worker。

本节门禁命令不得当作正式训练命令。

## 失败或中断尝试

当前没有失败或中断尝试。后续若出现重试，继续追加在本节，不新建替代日志。
