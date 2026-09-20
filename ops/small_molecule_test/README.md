# Stage3 小分子独立评测工具

本目录只服务 `test_smallmolecule`：在 PocketXMol Stage3 的 77 个测试 PDB 上，重新汇总 Find_1、Emap2lig 和冻结 Matcher BEST 的表现。代码不进入 `matcher/`，不改变训练 Dataset、模型、损失或既有 `test_0` 评测。

## 推荐阅读顺序

1. `test_smallmolecule.yaml`：核对全部只读来源、冻结 checkpoint、输出根和机械计数。
2. `build_manifest.py`：连接 Stage3 occurrence、既有候选和完整 blob 世界中心。
3. `ranking.py`：定义候选顺序与“非 Stage3 前景免费跳过”的尝试次数规则。
4. `evaluate_stage1.py`：生成不依赖 Matcher 的三套 Stage1-only 曲线。
5. `evaluate_model.py`：从训练 source release 恢复模型，并从同一次身份 logits 生成两档答案。
6. `plot_results.py`：只读取冻结 JSON 绘图，不参与正式统计。
7. `gates/`：保存必须进入服务器 release、但不属于正式运行入口的一次性门禁。

`artifacts.py` 只提供 YAML、JSON、JSONL 和 SHA-256 文件操作。`MANIFEST_README.md` 是正式 manifest 目录随产物发布的字段说明。

`test_smallmolecule.yaml` 的 `models` 键是稳定实验标识；每条记录的 `variant` 是 checkpoint 内必须精确匹配的模型结构变体。两者显式分离后，`strongest-1` 可以绑定 `strongest`，`blob_only-0` 可以绑定 `blob_only`，而历史实验目录不需要改名。

## 科学契约

- PDB 分母固定为 PocketXMol Stage3 测试清单中的 77 个 PDB。
- 候选顺序复用 `matcher.evaluation.test_set._selected_priority_order()`：selected 段在前，unselected 段在后；每段内按当前分数降序、`source_blob_index` 升序。
- 背景候选消耗一次尝试；唯一目标不属于 Stage3 的前景候选免费跳过；唯一目标属于 Stage3 的候选消耗一次尝试。
- Stage1-only 在 Stage3 目标候选处直接命中。Stage1+Stage2 只比较 Matcher argmax SMILES 与唯一目标的精确顶层 `PreparedLigand.smiles`。
- Matcher 前景头是训练辅助输出；本评测不读取 validation 前景阈值，不按前景概率筛选，不把前景概率写入正式结果或 docking handoff。
- 全配体答案与 small-molecule 答案来自同一次完整身份 forward。后者只在最终 argmax 前屏蔽原始 `type_tag` 不是 `small_molecule` 的 logits。
- Find_1 和 Emap2lig 的 blob 中心都从完整来源 blob 的全局 ZYX 体素计算，再转换成世界 XYZ，单位 Å。

## 稳定入口

正式任务只调用 `训练与运行/sh/small_molecule_test/` 中的薄脚本。服务器门禁调用
`ops/small_molecule_test/gates/`，本地测试调用 `tmp/test_smallmolecule/`；两者都不得被抄成
正式运行入口。
