# density_blob_only 正式训练日志

## 人工检查摘要

| 字段 | 当前值 |
| --- | --- |
| 状态 | 正式训练已自然早停；冻结 BEST 测试运行中 |
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
| BEST validation | `best_epoch=3`，`best_validation_check=12`，selection=`0.8490316713` |
| BEST 前景与身份指标 | PRAUC=`0.8015437651`；SMILES 匹配准确率=`0.8965195774` |
| 冻结前景阈值 | `0.248046875`；validation F1=`0.7458120903` |
| BEST SHA-256 | `932d1d52c35104c8af55235f080850652d6a7d5ab7924cb9d4a9fb9a8ab17748` |
| 测试 release | `/home/penghongen/Feedback/Matcher/releases/Matcher_52db89c9dabe/Matcher` |
| 测试 launch | `/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/运行日志与统计/slurm/launches/379403/evaluate_density_job379403_20260919T042516_a9` |
| 正式测试 | 运行中；输出根为 `/storage/penghongen/Matcher/evaluation/final_smiles/density_blob_only` |

## 正式运行命令

```bash
bash 训练与运行/sh/final_smiles/train_density_blob_only.sh
```

训练完成后的冻结测试使用：

```bash
bash 训练与运行/sh/final_smiles/evaluate_density_blob_only.sh
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

## 训练结果

- 训练于 2026-09-19 03:06:59（Asia/Shanghai）自然早停；共执行 21 次完整 validation、9,765 次优化器更新和两次实际学习率降低，21 份逐次 checkpoint 全部保留。
- BEST 来自 `best_epoch=3`、第 12 次 validation。selection 为 `0.8490316712550197`，由 foreground PRAUC `0.8015437651327864` 与真实前景且身份受支持候选上的 SMILES 匹配准确率 `0.896519577377253` 取平均得到。
- 第 21 次 validation 构成第三次平台条件；此前已经实际降低两次学习率，因此不再执行第三次无意义降低并直接早停。`summary.json` 声明 `stopped_by_early_stopping=true`、`stopped_by_max_epochs=false`。
- 训练结束后只在 BEST validation 预测上拟合一次前景阈值，得到 `0.248046875` 和 F1 `0.7458120903131829`；阈值已写回最终 `BEST.pt`。
- allocation 已恢复精确 `try_lock_379403`，`after_lock_379403` 继续保留；没有读取 calibration 科学产物。

## 冻结测试状态

- 2026-09-19 04:25:16（Asia/Shanghai）使用测试 release `Matcher_52db89c9dabe` 启动正式 test_0 前向；动态命令 SHA-256 为 `5fc7588a3dde4844b42e26799b4be78635db17bfdf9a20dced99ec966052f13b`。
- 模型配置、模型类和 checkpoint 规则由训练 run 指向的 `Matcher_6fe5cfa8f1f1` 加载；测试 release 只负责全候选 Dataset、selected 优先扩展 top-k、结果汇总和绘图。
- 两种受体条件共用冻结 BEST 前景阈值 `0.248046875`，不读取 calibration、不在测试集拟合阈值、不改变 Stage1 候选排序。测试运行时精确 try_lock 已删除，after_lock 保留。

## 首次 validation 历史快照

- 2026-09-18 19:16（Asia/Shanghai）完成第 1 个连续训练区段和第 1 次完整 validation；训练区段执行 465 次参数更新，身份损失实际覆盖 10,074 个受支持真前景候选。
- validation 固定遍历 5,544 个 selected 候选，其中 3,218 个真前景候选具备受支持身份；SMILES 匹配正确 2,735 个。
- 本次 selection 相对初始状态构成有效改善，`bad_validation_checks=0`，学习率保持 `1.0e-4`。
- 当时已写出 `checkpoints/epoch_01_validation_01.pt`、`LAST.pt` 和 `BEST.pt`；该段记录只保留首步历史，不代表当前状态。
