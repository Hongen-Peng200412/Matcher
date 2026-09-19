# density_blob_only 正式训练日志

## 人工检查摘要

| 字段 | 当前值 |
| --- | --- |
| 状态 | 正式训练与冻结 BEST 测试均已完成 |
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
| 测试 release | `/home/penghongen/Feedback/Matcher/releases/Matcher_c2b763cf90e2/Matcher` |
| 测试 launch | `/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/calibration/运行日志与统计/slurm/launches/379403/evaluate_density_job379403_20260919T050117_a10` |
| 正式测试 | 已完成；`frozen_test.json` SHA-256=`82c9864268570bd01b3d226e99e57e0eb3b7d7b072ac16a086c0fa7c0521ef37` |
| `test_smallmolecule` | 实现、机械测试与独立审查已完成；尚未触发 A800 正式运行 |

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
- 第一次冻结测试已经完成真实受体与 CryoAtom2 的 test_0 前向，但旧 `_read_pdb_subset()` 按逐行文本读取官方 JSON，把 160 行 JSON 文本误当成 160 个 PDB 标识，因而在 test_1 筛选前失败关闭。未发布 `frozen_test.json` 或图表；try_lock 自动恢复，after_lock 保留。修复只把该入口改为读取官方 JSON 顶层 `pdb_ids`，没有修改模型前向、阈值、候选排序或指标。

## 训练结果

- 训练于 2026-09-19 03:06:59（Asia/Shanghai）自然早停；共执行 21 次完整 validation、9,765 次优化器更新和两次实际学习率降低，21 份逐次 checkpoint 全部保留。
- BEST 来自 `best_epoch=3`、第 12 次 validation。selection 为 `0.8490316712550197`，由 foreground PRAUC `0.8015437651327864` 与真实前景且身份受支持候选上的 SMILES 匹配准确率 `0.896519577377253` 取平均得到。
- 第 21 次 validation 构成第三次平台条件；此前已经实际降低两次学习率，因此不再执行第三次无意义降低并直接早停。`summary.json` 声明 `stopped_by_early_stopping=true`、`stopped_by_max_epochs=false`。
- 训练结束后只在 BEST validation 预测上拟合一次前景阈值，得到 `0.248046875` 和 F1 `0.7458120903131829`；阈值已写回最终 `BEST.pt`。
- allocation 已恢复精确 `try_lock_379403`，`after_lock_379403` 继续保留；没有读取 calibration 科学产物。

## 冻结测试结果

- 2026-09-19 05:01:17（Asia/Shanghai）使用测试 release `Matcher_c2b763cf90e2` 重新启动冻结测试；动态命令 SHA-256 为 `5432d9736a85455a46bdd04fd07ca24f2a68c456d8a888120c114ba53d44e1c1`。第 10 次 allocation 执行成功，随后恢复精确 `try_lock_379403`；`after_lock_379403` 始终保留。
- 模型配置、模型类和 checkpoint 规则由训练 run 指向的 `Matcher_6fe5cfa8f1f1` 加载；测试 release 只负责全候选 Dataset、selected 优先扩展 top-k、结果汇总和绘图。真实受体和 CryoAtom2 分别完成 2,251 和 2,407 个全部 centered 候选的前向，其中 selected 候选分别为 1,653 和 1,651 个。
- 两种受体条件共用冻结 BEST 前景阈值 `0.248046875`，不读取 calibration、不在测试集拟合阈值、不改变 Stage1 候选排序。真实受体 test_0 的前景 PRAUC/F1 为 `0.9028440725/0.8599508600`，SMILES 匹配准确率为 `0.8626817447`（1,068/1,238）；CryoAtom2 的对应结果为 `0.8365511548/0.8254665203` 和 `0.8612068966`（999/1,160）。
- test_1 只复用 test_0 前向筛选官方 149-PDB 集合。真实受体的前景 PRAUC/F1 为 `0.9266899798/0.8574126534`，SMILES 匹配准确率为 `0.8612021858`（788/915）；CryoAtom2 的对应结果为 `0.8622134403/0.8231155779` 和 `0.8518072289`（707/830）。两种受体条件各有两个不受支持身份目标，按契约保留 Stage1 排名、纳入前景评价并排除出身份准确率分母。
- Gaussian 排序下，真实受体 Stage1-only/Stage1+Matcher 的 top-1、top-5、top-20 命中数分别为 `138/127`、`167/164`、`167/166`；CryoAtom2 为 `122/113`、`155/149`、`159/155`。来源概率均值排序下，真实受体为 `148/138`、`166/162`、`167/166`；CryoAtom2 为 `134/122`、`157/151`、`159/155`。每项分母均固定为 179 个 PDB。
- `foreground_and_smiles_metrics.png`、`topk_gaussian_final.png` 和 `topk_source_probability_mean_final.png` 的 SHA-256 依次为 `ab78e594ad6ad9a7c5e9c9f95d2c6914d331a220c765df4b3a66d881763fa91f`、`167c74714952134d3bbac05d964f89ddb6d90af49000bdcbc03cb643339ddf64` 和 `d88e68a288d4f1b0f570ce397c1a91579927179b94a5e633ac8febc2e3629204`。人工检查确认图中没有预览标记残留，Emap2lig 在两张 top-k 图中均为同一条 receptor-independent 曲线。

## 首次 validation 历史快照

- 2026-09-18 19:16（Asia/Shanghai）完成第 1 个连续训练区段和第 1 次完整 validation；训练区段执行 465 次参数更新，身份损失实际覆盖 10,074 个受支持真前景候选。
- validation 固定遍历 5,544 个 selected 候选，其中 3,218 个真前景候选具备受支持身份；SMILES 匹配正确 2,735 个。
- 本次 selection 相对初始状态构成有效改善，`bad_validation_checks=0`，学习率保持 `1.0e-4`。
- 当时已写出 `checkpoints/epoch_01_validation_01.pt`、`LAST.pt` 和 `BEST.pt`；该段记录只保留首步历史，不代表当前状态。

## `test_smallmolecule` 当前状态

- 2026-09-19 已从 `6d4c886` 建立独立工作树 `C:/Users/15919/Desktop/Matcher_small_molecule_test`。评测将继续使用本日志记录的冻结 BEST 和训练 source release，不重训模型，也不依据小分子测试结果重新选择 checkpoint。
- 新评测不读取、不应用且不报告 BEST 中的前景阈值；Matcher 前景头不参与尝试次数、SMILES 命中或 docking handoff。
- 当前已完成独立实现、4 项本地契约测试和三项独立审查；科学逻辑与注释窄复核均为 PASS。正式运行命令、评测 release、launch、结果路径和指标将在实际生成后回填到本节前部；失败尝试只追加到本日志的“先前尝试”，不另建日志。
