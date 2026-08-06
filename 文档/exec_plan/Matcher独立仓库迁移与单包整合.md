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
- [x] 2026-08-06：实现端点 `75ad965` 与学习端点 `1cb04f9` 的 tree 均为 `8e43f2ff82b4d9edbed20ca674138d30bd8461a5`；`Learn/CUMULATIVE` 已快进，主工作区干净。Python 编译、YAML 解析、Bash、PowerShell 与本地 26 项测试均通过。
- [x] 2026-08-06：AdaLigand 已从活动树删除 Matcher 运行代码，并在独立的实现线与学习线完成等价收口；A–G、Stage1、BOX 契约、总体讨论和旧执行证据保留。两个临时暂停的 Stage1 automation 已恢复原来的 3 小时间隔与原线程。
- [x] 2026-08-06：只执行安全同步，把仓库写入空的 `/home/penghongen/My_Project/Matcher`。同步后删除两个未跟踪的空 `matcher_v2` 目录；远端确认不存在任何名为 `matcher_v2` 的目录。服务器使用 `Pocket_Plus_centos7_cu121_allgpu` 环境完成 YAML、全部 shell/sbatch 和 Stage3 轻量导入检查，26 项测试全部通过；未申请 GPU、未运行删除式同步。
- [x] 2026-08-06：把 PyTorch `file_system` 张量共享策略从训练入口下沉到共同 `build_loader`。训练、验证和独立推理现在遵守同一文件句柄策略，避免训练后新推理进程再次触发旧 `Too many open files` 问题；本地和服务器均为 26 项测试通过。
- [x] 2026-08-06：CPU Job `337745` 使用 release `Matcher_145f92c138a3` 生成正式 mode1 清单。train/validation/calibration 分别为 12,881/188/89 个 PDB 和 180,382/2,914/1,386 个 occurrence；新清单与迁移前清单字节级相同，SHA-256 均为 `7845463785b27927ebd509f50167075f08241f6beda3c5117a713d9d3c5e8131`。
- [ ] H200 simple Job `337744` 等待资源，只读重算旧 Phase2 BEST 在身份感知指标下的 validation 结果；任务不设时限、不保留 allocation、不写正式结果根，也不训练参数。
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

## 服务器隔离 smoke 证据

- 本机安全同步入口：`C:\Users\15919\Desktop\Matcher\与服务器交互\sync_code.ps1`。
- 远端代码根：`/home/penghongen/My_Project/Matcher`；同步前目录存在但为空。
- Python：`/home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python`。
- 静态检查：两个正式 YAML 可解析；仓库内所有 `.sh` 与 `.sbatch` 均通过 `bash -n`；`import matcher.stage3` 不会连带导入 `matcher.train` 或 Dataset。
- 回归结果：`26 passed, 8 warnings in 17.79s`。警告来自 PyTorch Transformer nested tensor 与 checkpoint CPU autocast 的弃用提示，不是测试失败。

## 旧 F1 诊断

- 旧 Phase2 BEST 位于 `global_step=16729`。旧固定 slot 指标为 precision `0.22512755`、recall `0.12113933`、F1 `0.15751896`、阈值 `0.12`，共预测 1,568 对、判对 353 对、validation occurrence 总数 2,914。
- 旧 checkpoint 已由当前 85,801,217 参数模型严格加载成功，说明可在不转换权重的情况下重算指标。
- validation 中 146/188 个 PDB 含同身份多 occurrence；2,732/2,914 个 occurrence（93.75%）属于重复身份组。共 354 个重复身份组，最大单组 90 个 occurrence。calibration 的对应比例为 1,307/1,386（94.30%）。
- 训练使用身份内 Hungarian，且同身份 slot 不代表固定真实 occurrence；旧验证却按 slot 下标判分。这个错误覆盖绝大多数 validation occurrence，并直接控制 BEST、学习率下降和提前停止，因此不能在身份感知重评完成前把 `0.1575` 解释为模型能力。
- 当前证据支持先保持模型与训练分布不变，只用修正后的指标重评并从头训练一个基线；若身份感知结果仍低，再讨论六头打分权重、损失或架构。
