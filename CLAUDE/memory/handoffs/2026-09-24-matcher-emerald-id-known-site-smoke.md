# Handoff：Matcher 与 EMERALD-ID 已知位点比较的 CPU 试运行

Date: 2026-09-24

## Current State

本轮只完成比较计划与三个案例的 CPU smoke/门禁，没有运行完整 NxU/nxu 测试，也没有得到任何方法的正式准确率。测试仍以固定原始 occurrence 的逐配体 micro Top-1 为主指标：NxU 为 179 PDB、2,502 点，nxu 小分子为 94 PDB、821 点；多身份分层与 PDB 平均仅作补充。完整组合数为 9,172+3,114 次位置—身份评分。所有本轮 Slurm Job 已结束，不占用 allocation；没有读取 calibration、修改旧科学产物或提交 Git。

## Completed

- 计划写在 `test_真实场景(Structure)/参考材料 & 想法/Matcher_EMERALD-ID 比较计划书.md`；总日志与三个实验各自日志在 `test_真实场景(Structure)/执行记录/`。试运行代码在 2026-09-24 位于本地 `代码/`，2026-09-25 已整体移至同级 `smoke/`；服务器当次运行副本仍在 `/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/code/`，未随本地整理移动。
- 官方 7bxu 教程 Job `387990` 自然完成，用时 4:14:12；本次新 `.sc` 含 ADP、ATP、CLR、Y01 各一条有效评分，官方排名 ATP、ADP、CLR、Y01。作者自带旧参考文件排 ADP、ATP、CLR、Y01；两者分目录保存，不混成同一次运行。
- `9nal/candidate_id=0/nxu` 两身份小分子 Job `387998_0` 自然完成，用时 1:41:41；L00、L01 各有一条有效评分，官方排名首位 L01 与原始精确 SMILES 真值一致。这只是一个查询点的 smoke，不是 nxu 准确率。第一次 Job `387992_0` 因 RDKit MOL2 芳香环回读限制失败；重试改用 Open Babel 回读同一 MOL2，原始与回读化学图一致，旧失败产物保留。
- `9nnz/candidate_id=0/NxU` 两糖身份首次 Job `387992_1` 和独立复核 `388007_1` 都在官方 Rosetta `RingConformerSet.cc:325` 报环构象错误，没有任何新 `.sc`。复核在脚本未更新的条件下复现；虽然 Slurm 将复核标为 `COMPLETED`/退出码 0，科学评分门禁仍失败。首次运行还因脚本热更新出现另一条 shell EOF，不能把它与 Rosetta 错误混淆。
- 评分门禁只把一个查询点的 `.sc` 交给官方 `rank_ligand_identity.py`，用 `zscore` 和 `lig_dens` 生成概率；没有按裸 zscore 排名。教程参考 `.sc` 只用于预先验证读取格式，不进入新运行结果。

## Decisions

- 用户接受并要求披露官方 `mol2genparams.py` 的内部共轭键编码：`9nal` 的 L01 从原始 SMILES 到 MOL2 保留化学图、键级和电荷；Rosetta `.params` 保留相同原子、显式氢与连接，但将一条共轭 C–N 单键写为 `BOND_TYPE C5 N3 2`。服务器源码 `BasicClasses.py` 的 `BondClass.order_in_params` 支持这是官方编码。不得宣称 `.params` 与 MOL2 字面逐键级一致，也不得据此自动放行其他变化；不修改官方参数生成或评分。
- 保持原始 EMDB 图给 Rosetta，Matcher 使用既有约 1 Å 图；查询点来自真实 occurrence 质心，真实身份不进入方法输入。原始顶层 `PreparedLigand.smiles` 是唯一身份键，失败身份和失败点仍留在正式 micro 分母。
- 只批准并完成 CPU smoke。未建立或提交 12,286 次完整评分的生产命令，也未运行 Matcher 的真实点输入接入或正式推理。

## Open Questions

- 糖样本的环构象失败是否需要使用仍属官方方法的其他准备方式，或作为 EMERALD-ID 覆盖失败按固定分母计 0；不得静默删除该类别。
- 正式大规模运行的随机种子、重复次数、每查询目录组织及资源预算尚未决定。教程本次 ATP/ADP 接近，且与作者参考的前两名顺序不同，不能声称逐位确定。
- 两个成功案例约需每个身份 51–64 分钟；机械外推 12,286 次约为 10,400–13,000 单核 CPU 小时、96 核理想满载约 4.5–5.6 天。分子和失败路径差异很大，这不是正式时限承诺。

## Next Actions

先与用户讨论这三项门禁的可行性结论和资源/失败口径，再决定是否授权完整 NxU/nxu 比较。若获准，先冻结随机性和正式短命令，完成 Matcher 给定真实点接入、模型训练 release 快照加载及输入等价门禁；EMERALD-ID 继续使用官方流程，按查询点隔离评分并保存精确 SMILES 名称映射。完整运行不得减少固定分母，最终必须保留与现有 Matcher 分类图科学含义相同的逐点身份准确率及类别支持数图。

## Files To Reopen

- `test_真实场景(Structure)/参考材料 & 想法/Matcher_EMERALD-ID 比较计划书.md`
- `test_真实场景(Structure)/执行记录/总日志.md` 与 `EMERALD-ID.md`
- `test_真实场景(Structure)/smoke/README.md`、`prepare_project_smoke.py`、`check_smoke_scores.py`、`check_params_graph.py`
- `文档/mapping/计划执行映射.md`

本轮新增文件和映射修改均留作 unstaged。开始前已有的两项用户暂存文件保持 staged，未修改、未重置；Learn/CUMULATIVE 仍在 `4f726c0e030e445c80f31c3ea62a8b99d1785804`。
