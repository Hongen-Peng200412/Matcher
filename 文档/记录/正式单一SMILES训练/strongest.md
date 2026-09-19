# strongest 正式训练日志

## 人工检查摘要

| 字段 | 当前值 |
| --- | --- |
| 状态 | 正式训练已自然早停；`test_smallmolecule` 实现、机械测试与独立审查已完成 |
| Slurm Job | `378587` |
| 硬件 | hnode02，H100 GPU 0，32 CPU |
| 正式 YAML | `configs/experiments/final_smiles/strongest.yaml` |
| 正式 manifest | `/storage/penghongen/Matcher/manifests/final_train_validation_shards01_06_v1/manifest.json` |
| 批量 | `batch_size=32`，`accumulation_steps=1` |
| Matcher release | `/home/penghongen/Feedback/Matcher/releases/Matcher_6fe5cfa8f1f1/Matcher` |
| allocation launch | `/home/penghongen/Feedback/PocketXMol/launches/378587/train_docking_job378587_20260918T185357_a8` |
| W&B | `https://wandb.ai/pencounkdual-111/Matcher/runs/sb11f8nk` |
| 运行根 | `/storage/penghongen/Matcher/runs/strongest/train_docking_job378587_20260918T185357_a8` |
| 训练完成 | 2026-09-19 21:52:11（Asia/Shanghai）；18 次完整 validation，8,370 次优化器更新，两次学习率降低 |
| BEST | `best_epoch=3`，`best_validation_check=12`，selection=`0.8577917867509867` |
| BEST SHA-256 | `ebf53b1d65cdcd31c1857df454db85aee183551f4fb015ecccb5f7e8ff053941` |
| summary SHA-256 | `5c4513ca67e83053c75caa8227525867e01b1bfe8605373c4b85db046873db30` |
| `test_smallmolecule` | 尚未触发 A800 正式运行 |

## 正式运行命令

```bash
bash 训练与运行/sh/final_smiles/train_strongest.sh
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

- strongest 于 2026-09-19 21:52:11（Asia/Shanghai）自然早停。训练执行 18 次完整 validation、8,370 次优化器更新和两次实际学习率降低；`summary.json` 声明 `stopped_by_early_stopping=true`、`stopped_by_max_epochs=false`。
- BEST 来自 `best_epoch=3`、第 12 次 validation，selection 为 `0.8577917867509867`。`BEST.pt` SHA-256 为 `ebf53b1d65cdcd31c1857df454db85aee183551f4fb015ecccb5f7e8ff053941`；`summary.json` SHA-256 为 `5c4513ca67e83053c75caa8227525867e01b1bfe8605373c4b85db046873db30`。
- 训练 source release 仍为 `/home/penghongen/Feedback/Matcher/releases/Matcher_6fe5cfa8f1f1/Matcher`，W&B 仍为 `https://wandb.ai/pencounkdual-111/Matcher/runs/sb11f8nk`。

## `test_smallmolecule` 当前状态

- 2026-09-19 已从 `6d4c886` 建立独立工作树 `C:/Users/15919/Desktop/Matcher_small_molecule_test`。本轮在 density_blob_only 结果完成后，使用上述 BEST 和训练 source release 在 A800 上执行小分子子集评测。
- 新评测不读取、不应用且不报告 BEST 中的前景阈值；Matcher 前景头不参与尝试次数、SMILES 命中或 docking handoff。
- 当前已完成独立实现、4 项本地契约测试和三项独立审查；科学逻辑与注释窄复核均为 PASS。正式运行命令、评测 release、launch、结果路径和指标将在实际生成后回填到本节前部；失败尝试只追加到本日志的“先前尝试”，不另建日志。
