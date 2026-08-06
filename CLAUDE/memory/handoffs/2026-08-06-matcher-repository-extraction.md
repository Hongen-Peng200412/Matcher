# Handoff: Matcher 独立仓库迁移

Date: 2026-08-06

## Current State

Matcher 已从 AdaLigand 快照导入独立仓库，正在 `codex/matcher-repository-extraction` 实现分支上收口。正式代码已整合为一个 `matcher/` 包；审查修正后单包回归 26 项全部通过。尚未完成学习线重建、AdaLigand 活动代码删除与服务器安全同步。

## Completed

- AdaLigand 已先完成双线协调，`Learn/CUMULATIVE@28eb6c3` 是迁移所用的干净事实基点。
- Matcher 仓库基点为 `b9c3303`，导入快照提交为 `a191589`。
- 删除旧 Anchor 活动入口、旧画像脚本、重复 `matcher_v2/` 包和被误提交的 Python 缓存。
- 正式代码、配置、脚本、服务器交互与结果目录均改为 Matcher 独立边界。
- 完成契约与可读性双审查：冻结模式二指针的 occurrence ID，阻止模式二误用模式一验证，去除旧两头整模语义和失效文档入口。
- 修正同身份 slot 固定下标判分错误；新指标允许相同配体身份的副本交换，不允许不同身份交换。

## Decisions

- `model_core.py` 保存双模式六头模型实际复用的旧主干算子，`model.py` 是正式模型入口。
- A–G 和 Stage1 数据仍归 AdaLigand；Matcher 只读取，不复制生产代码。
- 旧 Phase2 最佳 F1 为 `0.15238526`，下一次 GPU 提交前必须先诊断和修改，并再次请求用户授权。

## Open Questions

- 旧固定 slot 指标已确认会低估同身份多 occurrence，但尚未在旧 validation 全量离线重算；仍需判断它能解释多少 F1 差距，以及模型本身是否还存在主要问题。

## Next Actions

1. 完成文档路径核对、脚本静态验证和两类审查。
2. 重建 Matcher 学习线并验证与实现端点等价。
3. 从 AdaLigand 活动树删除 Matcher 运行代码，写入短跨仓库索引并双线收口。
4. 恢复被暂停的 Stage1 automation。
5. 安全同步到 `/home/penghongen/My_Project/Matcher`，完成无卡服务器 smoke。

## Files To Reopen

- `文档/exec_plan/Matcher独立仓库迁移与单包整合.md`
- `文档/mapping/计划执行映射.md`
- `文档/规划文档/Matcher_双模式ABO实施计划.md`
- `文档/规划文档/Matcher_双模式数据契约.md`
- `matcher/README.md`
- `talk/Matcher仓库迁移文件清单.md`
