# ops/matcher：一次性数据构建与冻结评估入口

本目录放置需要扫描正式产物或发布派生文件的短入口。运行时代码位于 `matcher/`；这里不保存训练中的隐藏状态。

| 入口 | 正式作用 |
|---|---|
| `build_smiles_assets.py` | 从 PreparedLigand 顶层 SMILES 发布全局图包和 Stage3 自同构包 |
| `build_stage1_manifest.py` | 扫描指定 Stage1 划分，发布候选、身份 JSONL 和 manifest |
| `plot_stage1_preview.py` | 从冻结元数据画一张明确标记的 Stage1-only preview |
| `evaluate_frozen_test.py` | 阶段 E 用同一 BEST 运行两种受体条件和可选 test_1 筛选 |

正式 shell 命令由 `训练与运行/sh/` 提供。测试命令、临时 SHA-256 取证和服务器一次性动态命令不写入这些 Python 文件。

## 写入边界

构建器只写显式 `--output-path` 及其同目录派生文件。Emap2lig、Find_1、AdaLigand 和 Pocket_Plus 来源均只读。train manifest 只能在用户确认本轮全部 Stage1 分片闭合后发布；当前不纳入未完成的第 6 片。

