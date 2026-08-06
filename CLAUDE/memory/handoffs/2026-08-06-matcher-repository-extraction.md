# Handoff: Matcher 独立仓库迁移

Date: 2026-08-06

## Current State

Matcher 已完成独立仓库迁移、单包整合、AdaLigand 活动代码退出和服务器隔离 smoke。正式代码只有一个 `matcher/` 包；本地与服务器回归均为 26 项通过。当前尚未完成旧 checkpoint 的新指标离线重算，也未获得下一次 GPU 正式重训授权。

## Completed

- AdaLigand 已先完成双线协调，`Learn/CUMULATIVE@28eb6c3` 是迁移所用的干净事实基点。
- Matcher 仓库基点为 `b9c3303`，导入快照提交为 `a191589`。
- 删除旧 Anchor 活动入口、旧画像脚本、重复 `matcher_v2/` 包和被误提交的 Python 缓存。
- 正式代码、配置、脚本、服务器交互与结果目录均改为 Matcher 独立边界。
- 完成契约与可读性双审查：冻结模式二指针的 occurrence ID，阻止模式二误用模式一验证，去除旧两头整模语义和失效文档入口。
- 修正同身份 slot 固定下标判分错误；新指标允许相同配体身份的副本交换，不允许不同身份交换。
- Matcher 实现端点 `75ad965` 与学习端点 `1cb04f9` tree 完全相同；`Learn/CUMULATIVE` 已快进。
- AdaLigand 已删除活动 Matcher 包、配置、训练入口和旧画像脚本，并完成自身实现线与学习线等价收口；Stage1 automation 已恢复。
- 安全同步已写入 `/home/penghongen/My_Project/Matcher`。远端不存在 `matcher_v2` 目录；YAML、shell/sbatch、Stage3 轻量导入和 26 项回归全部通过。

## Decisions

- `model_core.py` 保存双模式六头模型实际复用的旧主干算子，`model.py` 是正式模型入口。
- A–G 和 Stage1 数据仍归 AdaLigand；Matcher 只读取，不复制生产代码。
- 旧 Phase2 最佳 F1 为 `0.15238526`，下一次 GPU 提交前必须先诊断和修改，并再次请求用户授权。

## Open Questions

- 旧固定 slot 指标已确认会低估同身份多 occurrence，但尚未在旧 validation 全量离线重算；仍需判断它能解释多少 F1 差距，以及模型本身是否还存在主要问题。

## Next Actions

1. 用旧 validation 与 checkpoint 量化同身份固定 slot 指标的低估幅度，并区分指标错误与模型学习不足。
2. 根据诊断结果形成最小修改与重训建议；科学定义变化须重新确认。
3. 展示最终 Phase1/Phase2 YAML、正式训练 shell 与无 `--time` 的提交命令，获得用户明确授权后才提交 GPU 训练。

## Files To Reopen

- `文档/exec_plan/Matcher独立仓库迁移与单包整合.md`
- `文档/mapping/计划执行映射.md`
- `文档/规划文档/Matcher_双模式ABO实施计划.md`
- `文档/规划文档/Matcher_双模式数据契约.md`
- `matcher/README.md`
- `talk/Matcher仓库迁移文件清单.md`
