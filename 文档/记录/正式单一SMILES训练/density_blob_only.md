# density_blob_only 正式训练日志

## 人工检查摘要

| 字段 | 当前值 |
| --- | --- |
| 状态 | 正式训练中；首次完整 validation 已通过 |
| Slurm Job | 显示名 `379403_1`；runner 实际 `SLURM_JOB_ID=379403` |
| 硬件 | gnode10，A800 GPU 1，16 CPU |
| 正式 YAML | `configs/experiments/final_smiles/density_blob_only.yaml` |
| 正式 manifest | `/storage/penghongen/Matcher/manifests/final_train_validation_shards01_06_v1/manifest.json` |
| 批量 | `batch_size=32`，`accumulation_steps=1` |
| Matcher release | `/home/penghongen/Feedback/Matcher/releases/Matcher_6fe5cfa8f1f1/Matcher` |
| allocation launch | `/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/运行日志与统计/slurm/launches/379403/predict_calibration_job379403_20260918T184914_a8` |
| W&B | `https://wandb.ai/pencounkdual-111/Matcher/runs/wl1zm8mr` |
| 运行根 | `/storage/penghongen/Matcher/runs/density_blob_only/predict_calibration_job379403_20260918T184914_a8` |
| 首个训练区段与首次 validation | `global_step=465`；foreground PRAUC=`0.7759281826`，SMILES 匹配准确率=`0.8499067744`，selection=`0.8129174785` |
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
- 门禁报告：`/storage/penghongen/Matcher/gates/density_blob_only/job379403/predict_calibration_job379403_20260918T152102_a7/gate.json`；状态为 `pass`，GPU 为 `NVIDIA A800-SXM4-80GB`，manifest SHA-256 为 `3551e4a7b111c9d51a718fe11586769f31a7a50f66b8d25839bb9742418ad247`。
- `8+8` 并存与 15 个阶段独占 worker 的端到端时间分别为 1,533.079 秒和 1,539.476 秒；阶段独占方案没有更快，且 worker 峰值内存从 96,959,393,792 字节增至 142,548,070,400 字节，因此唯一正式方案保留 `8+8` 并存生命周期。

## 先前尝试

- 第一次门禁启动因一次性脚本未把 Matcher release 加入 `PYTHONPATH` 而失败；未进入训练。
- 第二次门禁使用绝对数值 `1e-6` 比较经过完整训练区段后的两条非确定性 CUDA 轨迹，并在显式回收前检查训练 worker，因门禁表达错误而失败；未进入训练。
- 第三次门禁尝试启用 PyTorch 全确定性算法，但 `upsample_trilinear3d_backward_out_cuda` 没有确定性实现，因此失败关闭；未进入训练。
- 最终门禁以候选顺序与固定计数严格一致、核心指标与损失相称一致并显式关闭阶段 worker 的口径通过。当前正式训练是本实验唯一训练尝试。

## 训练状态快照

- 2026-09-18 19:16（Asia/Shanghai）完成第 1 个连续训练区段和第 1 次完整 validation；训练区段执行 465 次参数更新，身份损失实际覆盖 10,074 个受支持真前景候选。
- validation 固定遍历 5,544 个 selected 候选，其中 3,218 个真前景候选具备受支持身份；SMILES 匹配正确 2,735 个。
- 本次 selection 相对初始状态构成有效改善，`bad_validation_checks=0`，学习率保持 `1.0e-4`。
- 已写出 `checkpoints/epoch_01_validation_01.pt`、`LAST.pt` 和 `BEST.pt`；训练继续进入同一 epoch 的第 2 个连续区段。
