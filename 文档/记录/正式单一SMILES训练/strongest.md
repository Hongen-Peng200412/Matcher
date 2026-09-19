# strongest 正式训练日志

## 人工检查摘要

| 字段 | 当前值 |
| --- | --- |
| 状态 | 正式训练已自然早停；冻结 BEST 测试运行中 |
| Slurm Job | `378587` |
| 硬件 | hnode02，H100 GPU 0，32 CPU |
| 正式 YAML | `configs/experiments/final_smiles/strongest.yaml` |
| 正式 manifest | `/storage/penghongen/Matcher/manifests/final_train_validation_shards01_06_v1/manifest.json` |
| 批量 | `batch_size=32`，`accumulation_steps=1` |
| Matcher release | `/home/penghongen/Feedback/Matcher/releases/Matcher_6fe5cfa8f1f1/Matcher` |
| allocation launch | `/home/penghongen/Feedback/PocketXMol/launches/378587/train_docking_job378587_20260918T185357_a8` |
| W&B | `https://wandb.ai/pencounkdual-111/Matcher/runs/sb11f8nk` |
| 运行根 | `/storage/penghongen/Matcher/runs/strongest/train_docking_job378587_20260918T185357_a8` |
| 首个训练区段与首次 validation | `global_step=465`；foreground PRAUC=`0.7775615340`，SMILES 匹配准确率=`0.8520820385`，selection=`0.8148217863` |
| BEST validation | `best_epoch=3`，`best_validation_check=12`，selection=`0.8577917868` |
| BEST 前景与身份指标 | PRAUC=`0.8147134678`；SMILES 匹配准确率=`0.9008701057` |
| 冻结前景阈值 | `0.333984375`；validation F1=`0.7553133515` |
| BEST SHA-256 | `ebf53b1d65cdcd31c1857df454db85aee183551f4fb015ecccb5f7e8ff053941` |
| 测试 release | `/home/penghongen/Feedback/Matcher/releases/Matcher_c2b763cf90e2/Matcher` |
| 测试 launch | `/home/penghongen/Feedback/PocketXMol/launches/378587/evaluate_strongest_job378587_20260919T235054_a9` |
| 正式测试 | 运行中；输出根为 `/storage/penghongen/Matcher/evaluation/final_smiles/strongest` |

## 正式运行命令

```bash
bash 训练与运行/sh/final_smiles/train_strongest.sh
```

训练完成后的冻结测试使用：

```bash
bash 训练与运行/sh/final_smiles/evaluate_strongest.sh
```

## 门禁与只读核查

- 精确 try_lock：`/home/penghongen/Feedback/PocketXMol/allocations/try_lock_378587`。
- 精确 after_lock：`/home/penghongen/Feedback/PocketXMol/allocations/378587/after_lock_378587`；全程不得删除。
- 动态命令：`/home/penghongen/Feedback/PocketXMol/allocations/378587/run_cmd_378587.sh`。
- 门禁原始测量：`/storage/penghongen/Matcher/gates/strongest/job378587/train_docking_job378587_20260918T162959_a6/failed_gate.json`。
- 门禁最终裁决：同目录 `gate.json`；状态为 `pass`，GPU 为 `NVIDIA H100 PCIe`，manifest SHA-256 为 `3551e4a7b111c9d51a718fe11586769f31a7a50f66b8d25839bb9742418ad247`。
- `8+8` 并存与 30 个阶段独占 worker 的端到端时间分别为 5,718.374 秒和 2,241.926 秒；阶段独占 worker 峰值内存增长约 101%，因此唯一正式方案保留 `8+8` 并存生命周期。

## 先前尝试

- 第一次门禁启动因一次性脚本未把 Matcher release 加入 `PYTHONPATH` 而失败；未进入训练。
- 第二次门禁完整运行后证明候选顺序和 worker 关闭正确，但一次性脚本错误使用绝对损失差 `0.05` 判定相称一致。最终裁决改用固定计数严格一致、核心指标绝对差不超过 `0.01`、损失相对差不超过 `0.10`，并绑定原始报告 SHA-256 `5e033b75cec7d12d6702f8071ae1b4dc14922ab7784a47c64f813ba85aab1430`；没有重跑完整门禁。
- 当前正式训练是本实验唯一训练尝试；上述事件均为同一实验的训练前门禁尝试。

## 训练结果

- 训练于 2026-09-19 21:52:11（Asia/Shanghai）自然早停；共执行 18 次完整 validation、8,370 次优化器更新和两次实际学习率降低，18 份逐次 checkpoint 全部保留。
- BEST 来自 `best_epoch=3`、第 12 次 validation。selection 为 `0.8577917867509867`，由 foreground PRAUC `0.8147134678462865` 与真实前景且身份受支持候选上的 SMILES 匹配准确率 `0.9008701056556868` 取平均得到。
- 第 18 次 validation 构成第三次平台条件；此前已经实际降低两次学习率，因此不再执行第三次无意义降低并直接早停。`summary.json` 声明 `stopped_by_early_stopping=true`、`stopped_by_max_epochs=false`。
- 训练结束后只在 BEST validation 预测上拟合一次前景阈值，得到 `0.333984375` 和 F1 `0.7553133514986377`；阈值已写回最终 `BEST.pt`。
- `BEST.pt`、`LAST.pt` 与 `summary.json` 的 SHA-256 依次为 `ebf53b1d65cdcd31c1857df454db85aee183551f4fb015ecccb5f7e8ff053941`、`7b496a474193c849d71498069839c5f7bc8a051f000024f7dc7d2b07d1d38e89` 和 `5c4513ca67e83053c75caa8227525867e01b1bfe8605373c4b85db046873db30`。
- allocation 已恢复精确 `try_lock_378587`，`after_lock_378587` 继续保留；没有读取 calibration 科学产物。

## 冻结测试状态

- 2026-09-19 23:50:54（Asia/Shanghai）使用测试 release `Matcher_c2b763cf90e2` 启动正式 test_0/test_1；动态命令 SHA-256 为 `7a90b00685910be62a031d75c4957c9ade2bf03ec868626b74ddda305572aca4`。
- launch 明确绑定 H100、32 CPU、冻结 BEST、正式评估脚本和测试 release；模型配置、模型类和 checkpoint 规则仍由训练 run 指向的 `Matcher_6fe5cfa8f1f1` 加载。
- 测试只删除精确 `try_lock_378587`，`after_lock_378587` 保留；两种受体条件共用冻结阈值 `0.333984375`，不读取 calibration、不在测试集拟合阈值、不改变 Stage1 候选排序。
