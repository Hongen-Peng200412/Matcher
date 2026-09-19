# Handoff: strongest 与 density_blob_only 训练和冻结测试完成

Date: 2026-09-20

## Current State

strongest 与 density_blob_only 已使用包含 Find_1 第 6 分片的冻结 train/validation manifest 完成正式训练，并分别使用自身冻结 BEST 完成真实受体和 CryoAtom2 的 test_0/test_1 测试。测试的模型配置、模型类和 checkpoint 加载规则均来自训练 run 指向的 `Matcher_6fe5cfa8f1f1` release；当前评估 release 只负责读取全部 centered 候选、汇总 selected 优先扩展 top-k 和绘图。

Job 378587 与 runner 实际 Job 379403 均已恢复精确 try_lock，两个 after_lock 均保留。全过程没有读取 calibration 科学产物，也没有在测试集重新拟合前景阈值。

## Completed

- 冻结 manifest：`/storage/penghongen/Matcher/manifests/final_train_validation_shards01_06_v1/manifest.json`，SHA-256 为 `3551e4a7b111c9d51a718fe11586769f31a7a50f66b8d25839bb9742418ad247`。
- strongest：BEST selection=`0.8577917867509867`，validation 前景阈值=`0.333984375`；test_0 真实受体/CryoAtom2 的前景 PRAUC 为 `0.9066759340/0.8657636064`，SMILES 匹配准确率为 `0.8586429725/0.8681034483`。
- density_blob_only：BEST selection=`0.8490316712550197`，validation 前景阈值=`0.248046875`；test_0 真实受体/CryoAtom2 的前景 PRAUC 为 `0.9028440725/0.8365511548`，SMILES 匹配准确率为 `0.8626817447/0.8612068966`。
- 两项测试都对真实受体 2,251 个、CryoAtom2 2,407 个全部 centered 候选执行前向；冻结指标仍只统计 selected 候选，扩展 top-k 先排列 selected，再追加 unselected。
- 两项均生成 `frozen_test.json`、前景与身份指标图、Gaussian top-k 图和来源概率均值 top-k 图。人工检查确认 Emap2lig 在两张 top-k 图中保持同一条 receptor-independent 曲线。

## Decisions

- I/O 生命周期保留训练与 validation 各 8 个持久 worker 的并存实现；H100 的 30-worker 阶段独占方案内存约翻倍，A800 的 15-worker 阶段独占方案没有加速。
- Stage1-only 与 Stage1+Matcher 复用同一 selected 优先扩展顺序；Matcher 不重新排列候选，只判断前景并匹配 PDB 内闭集 SMILES。
- test_1 只筛选同一次 test_0 前向的官方 149-PDB 集合。
- `9ian` 和 `9oq7` 的不受支持身份候选保留 Stage1 排名和前景评价，但排除出 SMILES 身份准确率分母。

## Next Actions

1. 运行最终本地机械门禁并提交最后的日志与 handoff 变更。
2. 按双线 Git 规则重建学习历史，核实实现端点与学习端点树对象一致，再把 `Learn/CUMULATIVE` 快进到唯一最新提交。
3. 不再触发 GPU 任务；保持两个 allocation 的 try_lock 与 after_lock 状态，等待用户后续资源安排。

## Files To Reopen

- `文档/记录/正式单一SMILES训练/总日志.md`
- `文档/记录/正式单一SMILES训练/strongest.md`
- `文档/记录/正式单一SMILES训练/density_blob_only.md`
- `matcher/evaluation/checkpoint.py`
- `matcher/evaluation/test_set.py`
- `训练与运行/sh/final_smiles/evaluate_strongest.sh`
- `训练与运行/sh/final_smiles/evaluate_density_blob_only.sh`
