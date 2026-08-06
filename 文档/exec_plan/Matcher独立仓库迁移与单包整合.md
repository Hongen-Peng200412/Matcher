# Matcher 独立仓库迁移与单包整合执行记录

## 目标

把 AdaLigand 中已经验证的 Matcher 实现迁入独立仓库，只保留一个正式 `matcher/` 包，并建立不会与 AdaLigand 或其他服务器项目混淆的同步、训练和结果边界。迁移不重写 AdaLigand 历史；AdaLigand 继续管理 A–G、Stage1 产物、生产方契约和旧运行证据。

## 当前进度

- [x] 2026-08-06：暂停两个会读取 AdaLigand 工作树的 Stage1 automation；未修改、取消或重新提交服务器任务。
- [x] 2026-08-06：把 AdaLigand 当前混合改动按职责整理为实现线与学习线；两个端点树完全一致，`Learn/CUMULATIVE` 指向唯一最新提交且工作区干净。
- [x] 2026-08-06：初始化 Matcher 仓库，保留用户提供的 `AGENTS.md`，建立迁移前文件清单，并导入 AdaLigand `Learn/CUMULATIVE@28eb6c3` 的 Matcher 快照。
- [x] 2026-08-06：导入快照的合并基线为 55 项通过、1 项跳过。
- [x] 2026-08-06：把双模式实现整合为单个 `matcher/` 包。共享科学算子位于 `model_core.py`，六头正式入口位于 `model.py`；旧 Anchor Dataset、训练入口、配置、画像脚本和缓存文件不再保留在活动代码树。
- [x] 2026-08-06：单包整合后先有 22 项测试通过；审查修正并补充回归后为 26 项全部通过。
- [x] 2026-08-06：正式配置迁至 `configs/mode1/`，正式脚本迁至 `训练与运行/sh/mode1/`；代码、控制证据和结果分别使用 `/home/penghongen/My_Project/Matcher`、`/home/penghongen/Feedback/Matcher` 与 `/storage/penghongen/Matcher/Results`。
- [x] 2026-08-06：完成契约与可读性双审查。模式二正式训练在独立验证实现前明确拒绝模式一后处理；Stage1 指针升级为 schema 2，冻结 occurrence ID 与两项数量；旧两头整模继承收敛为共享编排核心与唯一六头实现。
- [x] 2026-08-06：修正同身份多 occurrence 的评估错误。训练允许同身份 slot 交换，因此验证现在也在同身份内最大匹配；不同配体身份仍不能交换。旧训练的固定 slot 指标会低估这类样本，不能直接作为新评估的基线。
- [ ] 完成脚本静态验证、实现线提交与学习线等价收口。
- [ ] 从 AdaLigand 活动树删除 Matcher 运行代码，只保留最新生产方事实、契约、历史证据和跨仓库索引。
- [ ] 只执行安全同步并完成服务器无卡 smoke；不运行删除式同步。
- [ ] 诊断旧模式一 Phase2 最佳 F1 `0.15238526` 的主要原因，形成修改与重训建议。
- [ ] 向用户展示最终 YAML、shell 和无 `--time` 的提交命令；得到新授权后才提交 GPU 重训。

## 关键决定

1. 最终仓库只有 `matcher/`，不并存 `matcher_v2/`。旧实现仍作为 `model_core.py` 内的共享算子被六头模型复用，不复制同构主干。
2. 两个 Dataset、模式一后处理和未来模式二后处理保持显式隔离；不建立通用插件或动态注册系统。
3. Matcher 不拥有 A–G 或 Stage1 盘上生产契约。输入继续来自 AdaLigand 正式存储，输出不写回 AdaLigand 项目目录。
4. 安全同步固定指向远端 Matcher 项目；删除式同步脚本保留给人类手动使用，AI 不运行。
5. 旧训练已经证明端到端工程链可运行，但 F1 低于 0.2，不能把“无异常退出”误记为科学验收通过。

## 收口门禁

- 本地完整测试、Python 编译、YAML 解析、Bash 语法和 PowerShell 解析全部通过。
- 契约审查确认两个模式与 Stage3 叶子未因迁移失真；可读性审查确认无重复包、失效入口和不必要嵌套。
- 实现分支与学习分支端点树完全一致，`Learn/CUMULATIVE` 为唯一最新提交。
- AdaLigand 与 Matcher 两个工作区均干净；暂停的 Stage1 automation 恢复原状态。
