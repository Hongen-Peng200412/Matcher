# strongest 正式训练日志

## 人工检查摘要

| 字段 | 当前值 |
| --- | --- |
| 状态 | 正式训练、冻结 BEST 测试与 `test_smallmolecule` 均已完成 |
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
| 正式测试 | 已完成；`frozen_test.json` SHA-256=`89946a64a27959b79f7ac26a8f8efff2237cb8a3fcedbe47e555c846c61078f4` |
| `test_smallmolecule` | 已完成；正式结果 SHA-256=`7403b549e43b666477ff9e8e29311d285c0c82d8dfc8077e5ffd79dd11dee400` |

## 正式运行命令

```bash
bash 训练与运行/sh/final_smiles/train_strongest.sh
```

训练完成后的冻结测试使用：

```bash
bash 训练与运行/sh/final_smiles/evaluate_strongest.sh
```

Stage3 小分子子集正式评测和独立绘图使用：

```bash
bash 训练与运行/sh/small_molecule_test/evaluate_strongest.sh
bash 训练与运行/sh/small_molecule_test/plot_strongest.sh
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

## 冻结测试结果

- 2026-09-19 23:50:54（Asia/Shanghai）使用测试 release `Matcher_c2b763cf90e2` 启动正式 test_0/test_1；动态命令 SHA-256 为 `7a90b00685910be62a031d75c4957c9ade2bf03ec868626b74ddda305572aca4`。
- launch 明确绑定 H100、32 CPU、冻结 BEST、正式评估脚本和测试 release；模型配置、模型类和 checkpoint 规则仍由训练 run 指向的 `Matcher_6fe5cfa8f1f1` 加载。
- 第 9 次 allocation 执行成功，随后恢复精确 `try_lock_378587`；`after_lock_378587` 始终保留。真实受体和 CryoAtom2 分别完成 2,251 和 2,407 个全部 centered 候选的前向，其中 selected 候选分别为 1,653 和 1,651 个。
- 两种受体条件共用冻结阈值 `0.333984375`，不读取 calibration、不在测试集拟合阈值、不改变 Stage1 候选排序。真实受体 test_0 的前景 PRAUC/F1 为 `0.9066759340/0.8528660095`，SMILES 匹配准确率为 `0.8586429725`（1,063/1,238）；CryoAtom2 的对应结果为 `0.8657636064/0.8157589803` 和 `0.8681034483`（1,007/1,160）。
- test_1 只复用 test_0 前向筛选官方 149-PDB 集合。真实受体的前景 PRAUC/F1 为 `0.9277244216/0.8579823702`，SMILES 匹配准确率为 `0.8546448087`（782/915）；CryoAtom2 的对应结果为 `0.8894804601/0.8243314106` 和 `0.8566265060`（711/830）。两种受体条件各有两个不受支持身份目标，按契约保留 Stage1 排名、纳入前景评价并排除出身份准确率分母。
- Gaussian 排序下，真实受体 Stage1-only/Stage1+Matcher 的 top-1、top-5、top-20 命中数分别为 `138/128`、`167/163`、`167/165`；CryoAtom2 为 `122/114`、`155/148`、`159/154`。来源概率均值排序下，真实受体为 `148/140`、`166/161`、`167/165`；CryoAtom2 为 `134/122`、`157/151`、`159/154`。每项分母均固定为 179 个 PDB。
- `foreground_and_smiles_metrics.png`、`topk_gaussian_final.png` 和 `topk_source_probability_mean_final.png` 的 SHA-256 依次为 `183a302d017b73ab2021571270566ff351ad3d7e1d4e0e2c6238f151bd40b692`、`f0e3e11dd222be409133eba9072cc8daf52556b8468dd10958c714a74ee5c857` 和 `a892f7453eb11b6474699db57eaa194038649ed5900ffb9df52e5bb1fff899d5`。人工检查确认图中没有预览标记残留，Emap2lig 在两张 top-k 图中均为同一条 receptor-independent 曲线。

## `test_smallmolecule` 结果

- 2026-09-20 使用评测 release `/home/penghongen/Feedback/Matcher/releases/Matcher_4f819012083c/Matcher` 在 Job 379403 的 A800 上完成门禁与正式前向。门禁 launch 为 `/home/penghongen/Feedback/Matcher/launches/379403/test_smallmolecule_gate_strongest_job379403_20260920T011249`；正式评测与绘图 launch 分别为同一父目录下的 `test_smallmolecule_evaluate_strongest_job379403_20260920T014327` 和 `test_smallmolecule_plot_strongest_job379403_20260920T021405`。
- 正式结果位于 `/storage/penghongen/Matcher/evaluation/test_smallmolecule_v1/strongest/train_docking_job378587_20260918T185357_a8/`。`evaluation.json` SHA-256 为 `7403b549e43b666477ff9e8e29311d285c0c82d8dfc8077e5ffd79dd11dee400`；`docking_handoff_candidates.jsonl` 含 2,130 条记录，SHA-256 为 `54ac6c16635dc7ed06e04cae85dd9330157f5d22c4d14f505b8b173e7ae3db8f`。
- 评测直接加载本日志记录的冻结 BEST 和训练 source release，不重训模型，也不依据小分子测试结果重新选择 checkpoint。前景头和冻结阈值均不参与尝试次数、SMILES 命中或 docking 交接候选。
- 按 `k=1/3/5/10/20`，真实受体两档答案轴结果相同：Gaussian 为 `58/61/63/64/64`，概率均值为 `59/61/62/64/64`。CryoAtom2 完整身份轴的 Gaussian 与概率均值分别为 `43/57/59/62/62` 和 `50/61/61/62/62`；small-molecule 答案轴分别为 `44/58/60/63/63` 和 `51/62/62/63/63`。答案范围掩码只在 CryoAtom2 上带来 1 个额外命中，没有改变模型前向使用的完整身份集合。
- Gaussian 与概率均值图 SHA-256 分别为 `bcfa657f5477c00dc7f6d970aeb040459146f1334196199c8f432fbe86a67c95` 和 `9fc9f93185264645e50912053b85077be145f9eb02f0a18cd8848c2276f66ecd`。正式运行完成后，精确 `try_lock_379403` 已恢复，`after_lock_379403` 保留，`pre_lock_379403` 与 `kill_lock_379403` 不存在。
