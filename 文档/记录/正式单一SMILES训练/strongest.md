# strongest 正式训练日志

## 人工检查摘要

| 字段 | 当前值 |
| --- | --- |
| 状态 | GPU 门禁前 |
| Slurm Job | `378587` |
| 硬件 | hnode02，H100 GPU 0，32 CPU |
| 正式 YAML | `configs/experiments/final_smiles/strongest.yaml` |
| 正式 manifest | `/storage/penghongen/Matcher/manifests/final_train_validation_shards01_06_v1/manifest.json` |
| 批量 | `batch_size=32`，`accumulation_steps=1` |
| W&B | 正式训练启动后回填 |
| 运行根 | `/storage/penghongen/Matcher/runs/strongest/` |
| BEST 与测试指标 | 训练完成后回填 |

## 正式运行命令

```bash
bash 训练与运行/sh/final_smiles/train_strongest.sh
```

## 门禁与只读核查

- 精确 try_lock：`/home/penghongen/Feedback/PocketXMol/allocations/try_lock_378587`。
- 精确 after_lock：`/home/penghongen/Feedback/PocketXMol/allocations/378587/after_lock_378587`；全程不得删除。
- 动态命令：`/home/penghongen/Feedback/PocketXMol/allocations/378587/run_cmd_378587.sh`。
- 门禁命令、Matcher release、allocation launch、GPU 报告与首段状态在实际执行后回填。

## 先前尝试

当前没有失败或中断的 strongest 正式训练尝试。

