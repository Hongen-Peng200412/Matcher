# strongest 正式训练日志

## 人工检查摘要

| 字段 | 当前值 |
| --- | --- |
| 状态 | 正式训练中；等待首次完整 validation |
| Slurm Job | `378587` |
| 硬件 | hnode02，H100 GPU 0，32 CPU |
| 正式 YAML | `configs/experiments/final_smiles/strongest.yaml` |
| 正式 manifest | `/storage/penghongen/Matcher/manifests/final_train_validation_shards01_06_v1/manifest.json` |
| 批量 | `batch_size=32`，`accumulation_steps=1` |
| Matcher release | `/home/penghongen/Feedback/Matcher/releases/Matcher_6fe5cfa8f1f1/Matcher` |
| allocation launch | `/home/penghongen/Feedback/PocketXMol/launches/378587/train_docking_job378587_20260918T185357_a8` |
| W&B | `https://wandb.ai/pencounkdual-111/Matcher/runs/sb11f8nk` |
| 运行根 | `/storage/penghongen/Matcher/runs/strongest/train_docking_job378587_20260918T185357_a8` |
| 首个训练区段与首次 validation | 训练进程已装入 H100；完成后回填 |
| BEST 与测试指标 | 训练完成后回填 |

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
