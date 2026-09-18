# density_blob_only 正式训练日志

## 人工检查摘要

| 字段 | 当前值 |
| --- | --- |
| 状态 | GPU 门禁前 |
| Slurm Job | 显示名 `379403_1`；runner 实际 `SLURM_JOB_ID=379403` |
| 硬件 | gnode10，A800 GPU 1，16 CPU |
| 正式 YAML | `configs/experiments/final_smiles/density_blob_only.yaml` |
| 正式 manifest | `/storage/penghongen/Matcher/manifests/final_train_validation_shards01_06_v1/manifest.json` |
| 批量 | `batch_size=32`，`accumulation_steps=1` |
| W&B | 正式训练启动后回填 |
| 运行根 | `/storage/penghongen/Matcher/runs/density_blob_only/` |
| BEST 与测试指标 | 训练完成后回填 |

## 正式运行命令

```bash
bash 训练与运行/sh/final_smiles/train_density_blob_only.sh
```

## 门禁与只读核查

- 精确 try_lock：`/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/运行日志与统计/slurm/allocations/try_lock_379403`。
- 精确 after_lock：`/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/运行日志与统计/slurm/allocations/379403/after_lock_379403`；全程不得删除。
- 动态命令：`/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/运行日志与统计/slurm/allocations/379403/run_cmd_379403.sh`。
- 只读取上述 allocation 控制文件，不读取同一反馈根中的 calibration 科学产物。
- 门禁命令、Matcher release、allocation launch、GPU 报告与首段状态在实际执行后回填。

## 先前尝试

当前没有失败或中断的 density_blob_only 正式训练尝试。

