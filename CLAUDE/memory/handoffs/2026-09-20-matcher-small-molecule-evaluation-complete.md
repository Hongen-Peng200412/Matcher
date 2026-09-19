# Handoff: Matcher Stage3 小分子子集评测完成

Date: 2026-09-20

## Current State

Stage3 专用小分子子集的 Stage1-only、density_blob_only 与 strongest 正式评测均已完成。服务器结果根为 `/storage/penghongen/Matcher/evaluation/test_smallmolecule_v1/`；Job 379403 已恢复精确 try_lock，after_lock 保留，pre_lock 与 kill_lock 不存在。

实现线为 `codex/test-smallmolecule@33da59eade04700577fd3f7bf85178b3624878a2`。学习线为 `Learn/test-smallmolecule`，从已完成的 `Learn/final-train-control@4d57f4dc531cc0285253c8148b5da5d027c5ebd2` 继续组织文档、实现与测试。两条线中的 `ops/small_molecule_test/`、`训练与运行/sh/small_molecule_test/` 和 `tmp/test_smallmolecule/` 已逐路径确认完全一致。

## Completed

- 冻结 77-PDB manifest：`/storage/penghongen/Matcher/manifests/test_smallmolecule_v1/manifest.json`，SHA-256 为 `f69096612c89e360218f93481c11f621cafa7176a46b6d2883b4f6dd729caa8d`。
- 冻结 Stage1-only 结果：`/storage/penghongen/Matcher/evaluation/test_smallmolecule_v1/stage1_only/stage1_only.json`，SHA-256 为 `6466663de584accdea28c16a35f8eac1d35597314c2265f8281c21696d8eb86f`。
- 完成 density_blob_only 正式评测；`evaluation.json` SHA-256 为 `ef6935cdd0ac7a1f0cf859406da0a8b9c79aaa6286c0ea458e992b7436448eeb`，docking 交接文件含 2,148 条记录。
- 完成 strongest 正式评测；`evaluation.json` SHA-256 为 `7403b549e43b666477ff9e8e29311d285c0c82d8dfc8077e5ffd79dd11dee400`，docking 交接文件含 2,130 条记录。
- 生成两张 Stage1-only 图及两个模型各两张 Stage1+Stage2 图。
- manifest、density_blob_only 与 strongest 三项实际 A800 门禁均 PASS；模型代码均从各自训练 release 快照加载。
- 本地 `compileall`、4 项契约测试和差异空白检查通过；科学逻辑、注释表达和代码布局三项独立审查已经完成。

## Decisions

- 背景候选消耗一次尝试；命中非 Stage3 occurrence 的前景候选免费跳过；命中 Stage3 occurrence 的前景候选消耗一次尝试并进入身份判断。
- Matcher 正式小分子评测完全忽略前景头和冻结阈值，只检查目标 SMILES 是否被身份 logits 选中。
- 两档答案范围复用同一次完整身份前向：第一档在全部 PDB 内身份上取 argmax，第二档仅在原始类别为 small_molecule 的身份 logits 上施加答案掩码。
- `9rd8/source_blob_index=0` 继续服从唯一最大覆盖 occurrence，判为非 Stage3 前景并免费跳过。
- 正式评测逻辑只位于 `ops/small_molecule_test/`，没有进入 `matcher/` 训练主线。

## Open Questions

- 主工作树 `C:/Users/15919/Desktop/Matcher` 的 `Learn/CUMULATIVE@d275bc6` 仍保存用户已暂存的注释、Docstring 和 557 行计划书。学习分支已经逐字采用该计划书版本，但不得在未处理其余 staged 用户资产时直接移动 `Learn/CUMULATIVE`。需要由后续 Git 收口明确决定这些 staged 注释如何进入累计学习历史。

## Next Actions

1. 在不覆盖主工作树 staged 用户资产的前提下，把 `Learn/final-train-control` 与 `Learn/test-smallmolecule` 线性纳入 `Learn/CUMULATIVE`。
2. 推进前再次比较实现线和学习线的本轮运行路径，并重跑 `python -m unittest tmp.test_smallmolecule.test_contract -v`。
3. 保留 `codex/test-smallmolecule` 作为真实实现历史；累计学习线稳定后再安全移除两个临时工作树。

## Files To Reopen

- `文档/记录/正式单一SMILES训练/总日志.md`
- `文档/记录/正式单一SMILES训练/density_blob_only.md`
- `文档/记录/正式单一SMILES训练/strongest.md`
- `文档/mapping/计划执行映射.md`
- `talk_new/反馈/9-19小分子测试计划.md`
- `ops/small_molecule_test/README.md`
- `ops/small_molecule_test/test_smallmolecule.yaml`
