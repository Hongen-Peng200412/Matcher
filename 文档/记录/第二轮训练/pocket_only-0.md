# pocket_only-0 实验日志

本记录保存 `pocket_only-0` 的单实验追溯。候选侧只读取原始 50 维 A_feat_L0；身份侧使用 SMI-TED、SMILES 图和同 PDB 身份清单。

## 当前状态

- 阶段：实际 A800 门禁、正式训练、全测试集评估和 Stage3 小分子评估均已完成。
- 正式资源：A800 数组任务 `379403_1`；锁与反馈目录使用基础编号 `379403`，活动阶段使用 15 个非持久 DataLoader worker。全部命令结束后已恢复 `try_lock_379403`，`after_lock_379403` 保留，未发现残留训练或评估进程。
- 稳定实验标识：`pocket_only-0`；代码模型变体：`pocket_only`。
- 正式 YAML：`configs/experiments/final_smiles/pocket_only.yaml`。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_8c8f126c6d8b/Matcher`；内容 SHA-256 `8c8f126c6d8b69f9ea8471f15f6b3772969b59a3ded39ca14fa31140e520636a`。
- 门禁 launch/report：`gate_pocket_only0_job379403_20260921T121751`；`/home/penghongen/Feedback/Matcher/gates/second_round/pocket_only-0/gate_pocket_only0_job379403_20260921T121751.json`，`passed=true`。
- 训练 launch：`train_pocket_only0_job379403_20260921T122846`；运行根：`/storage/penghongen/Matcher/runs/pocket_only-0/train_pocket_only0_job379403_20260921T122846`。
- W&B：`https://wandb.ai/pencounkdual-111/Matcher/runs/7xybsvhj`。
- 冻结 train/validation manifest：`/storage/penghongen/Matcher/manifests/final_train_validation_shards01_06_v1/manifest.json`；SHA-256 `3551e4a7b111c9d51a718fe11586769f31a7a50f66b8d25839bb9742418ad247`。
- 候选输入：原始 `(N_A,50)` A_feat_L0 经 token 投影、掩码 mean/max 形成候选表示；不读取 A probability、坐标、L1/L2/L3、密度、V、P、PP 或候选人工统计。
- 模型边界：不实例化 RAUNet、体素调制器、候选模态聚合器或区域头。
- 损失权重：前景 `0.30`、ligand-area `0.00`、binding-area `0.00`、SMILES 身份 `1.00`。
- 选模公式：`(0.3 * foreground_PRAUC + supported_foreground_SMILES_accuracy) / 1.3`。
- BEST：epoch 3、第 11 次 validation，`global_step=5115`；PRAUC `0.7401372556157125`、SMILES 准确率 `0.8228713486637663`、selection `0.803778865652677`。BEST SHA-256 为 `8c8b0385f7fa1ba1af9379f38f45bbc9cb7f1a9dd7eb7b93158fb6e701189ce1`，冻结阈值为 `0.53515625`。
- 自然停止：epoch 7、第 23 次 validation 在两次实际降学习率后达到平台早停；23 份逐检查 checkpoint 均保留。
- 测试 release：`/home/penghongen/Feedback/Matcher/releases/Matcher_cab21d9ceea5/Matcher`；内容 SHA-256 `cab21d9ceea55c5c53b0f5d4589d5c34b1148cff8b77ed42691bda770fc91cb6`。模型类、配置和 checkpoint 加载器均来自训练 release 快照。
- 全测试集：launch `evaluate_full_pocket_only0_job379403_20260921T151639`；`frozen_test.json` SHA-256 `f699dd0c4334fb4816ef4c0a220ec75e33370d13da696ada069bddf597137327`。
- Stage3 小分子：前向 launch `evaluate_small_pocket_only0_job379403_20260921T153229`，绘图 launch `plot_small_pocket_only0_job379403_20260921T154817`；`evaluation.json` SHA-256 `69a38231ab8d4ff39f6c21ffb9fdee7ad05f2dfc0b88855112fce88654fa1259`。

## 正式命令

`bash 训练与运行/sh/final_smiles/train_pocket_only.sh`

`bash 训练与运行/sh/final_smiles/evaluate_pocket_only.sh`

`bash 训练与运行/sh/small_molecule_test/evaluate_pocket_only.sh`

`bash 训练与运行/sh/small_molecule_test/plot_pocket_only.sh`

## 测试、门禁和只读核查

- `D:\Anaconda\python.exe -m pytest -q`：`42 passed in 32.88s`。
- `D:\Anaconda\python.exe -m compileall -q matcher ops`：通过。
- 两个活动训练脚本经 Git Bash `bash -n`：通过。
- `density_only.yaml` 与 `pocket_only.yaml` 经 `load_smiles_experiment()` 严格解析：实验标识、模型变体、15 个 worker、损失权重和 patience 均符合第二轮协议。
- 面向本轮拥有文件的 `git diff --check`：通过；用户已有的三个未提交 Python 文件不属于本轮改动或提交范围。

实际 A800 门禁已通过：15 个阶段互斥 worker，train/validation 数为 `44643/5544`，epoch 批次计划 SHA-256 为 `f8bba32ecb6120ad61c83f3abc9f331729e69e7f7d337eec68fde5aa9b646053`；训练前后向、validation 前向与 worker 关闭检查通过。本节门禁命令不得当作正式训练命令。

## 失败或中断尝试

没有失败或中断训练尝试。
