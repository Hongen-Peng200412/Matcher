# Handoff: Matcher 包目录整理与第 4 批接续

Date: 2026-08-14

## Current State

活动 Goal 仍是 Matcher 第 1–5 批开发、validation 头对头比较、六张受管 GPU 调度、分批 W&B 和三层实验记录。恢复工作时先读取 `文档/记录/目标草案.md`，再核对本文件、Git 状态和服务器事件；不得运行 calibration 模型推理或把 calibration 用于选择。

当前工作分支为 `Learn/CUMULATIVE`；最近完成的实验记录基点 `c4ac5c8` 已回填第 1、2 批最终 validation 结果，本文件继续记录随后完成的服务器目录收口。目录整理实现端点是 `codex/matcher-package-layout@6176122`；学习历史按职责清单、运行模块与入口、测试导入、文档证据重建到 `8d6ba38`。目录整理的实现端点与学习端点树摘要均为 `dd86b658cab8d0401248726563ff504b662a2c56`，逐文件内容完全一致；科学正确性与严格技术表达两项独立复核均为 PASS。

六个受管 allocation 仅为 Job `340893/340894/340895/340897/340898/340899`。2026-08-14 17:26 后只读核验时六项均为 `RUNNING` 且全部保留 `after_lock`；Job `340895` 与 `340897` 已自然结束当前命令并生成 `try_lock`，其余四项继续第 3 批。不得删除任何 `after_lock`、释放 allocation、触碰用户任务或未经计划重复实验。

## Completed

- 第 4 批候选预训练 BEST 初始化基础已经在整理前端点完成：正式代码只核对来源语义与候选结构并严格加载 `candidate_encoder`；launch/release 与文件字节取证留在一次性 `tmp` 审计，不进入训练主线。
- `matcher/` 已整理为 `data/datasets/model/training/evaluation/utils` 六个职责子包；顶层只保留包入口，新增子包入口只写职责，不批量转发实现。
- 四个正式 Shell 已改用 `matcher.data.manifest`、`matcher.data.precompute`、`matcher.training.experiment` 和 `matcher.evaluation.calibration`；十九份正式 YAML 只更新读取入口说明。
- 对整理前 36 个既有 Python 文件做语法树归一比较，移除导入与模块说明后全部等价。六个测试文件按四组精确得到 `118 passed`；mypy 检查 `matcher/` 下 45 个 Python 文件零问题，pyflakes、compileall、19 份 YAML、8 个 Shell、四个帮助入口、公共导入和真实 `spawn` 工作进程生命周期均通过。
- 独立科学正确性审查确认候选集合、监督、模型状态、随机状态、损失、指标、global batch、持久产物和 calibration 禁读边界均未漂移；严格技术表达审查确认目录职责、依赖方向、入口与记录可以脱离本次对话冷读。两项结论均为 PASS，阻断项为零。
- 第 1 批 joint 已自然早停：BEST/last epoch `11/13`，selection score `0.6945667425`；第 1 批三种损失最大分差约 `0.26%`，不能确定稳定排序。
- 第 2 批五项均已自然早停并完成单任务/批次头对头回填。三个完整 Matcher oracle/detached/joint 的 selection score 为 `0.6981048482/0.6864829756/0.6887556119`；两项候选预训练 mean/max/未增强交叉注意力的 macro-F1 为 `0.5303632887/0.5272674639`。没有多个核心指标约 10% 的共同下降，后续继续使用 V 质心单视野。
- 2026-08-14 18:22 已把当前累计端点安全同步到可变服务器项目根。清理前后分别通过 36 个正式模块导入、19 份 YAML 解析、compileall 和四个命令入口检查；随后只从 `/home/penghongen/My_Project/Matcher/matcher/` 删除 23 个已经移动的旧顶层 Python 副本。既有 release、实验产物、动态命令、Slurm Job 和锁均未修改。

## Decisions

- 目录整理是独立的行为等价改动，不混入模型、损失、指标、配置值、缓存 schema、checkpoint schema 或 calibration 治理变化。
- 历史执行记录保留当时的旧模块命令；活动 Shell、YAML、结构概览、mapping 和新 handoff 使用整理后的路径。
- 不建立旧顶层模块兼容包装。可变服务器项目根已经通过安全同步与前后门禁精确清除 23 个已移动的旧顶层 Python 文件；不得使用删除整个项目的同步方式，不得改动任何既有 release。
- 来源 launch/release 与 SHA-256 证据只属于一次性 `tmp/<task-id>/` 或具名 `ops/`，不得重新放回 Batch4 正式初始化代码或 `resolved.json`。calibration 自身尚未执行，其一次性 claim 契约不在本轮目录整理中改动。
- Job `340895/340897` 虽已空闲保留，但 Batch4 来源模型必须等待第 3 批增强六分类完成后再由 validation 选择；不得为填卡跳过尚未决定的第 4 批或重复更早任务。

## Open Questions

- 第 3 批增强六分类和三个增强完整 Matcher 尚未全部自然收口；候选预训练 BEST 来源与第 4 批最小任务矩阵仍未确定。
- “去 SHA-256”只在 Batch4 来源运维取证边界内已经闭合；若要调整尚未执行的 calibration claim 格式，必须作为目录整理之外的独立治理改动评审。

## Next Actions

1. 继续只读观察第 3 批首个完整 validation、自然收口或失败事件；不按固定心跳重复记录无变化状态。
2. 第 3 批增强六分类自然收口后，只用 validation 在候选预训练模型中选源，运行一次性来源审计，建立第 4 批最小 YAML 与逐任务记录。
3. 选源、配置、逐任务记录和双审完成后，从当前已整理的可变项目根生成内容寻址 release；W&B 使用独立 Batch4 project/group。
4. 不读取 calibration 模型结果，不删除 `after_lock`，不释放 GPU，不触碰用户任务。

## Files To Reopen

- `文档/记录/目标草案.md`
- `CLAUDE/memory/handoffs/2026-08-14-matcher-package-layout.md`
- `talk/refactor/Matcher_包目录职责整理.md`
- `talk/Matcher_模块结构概览.md`
- `matcher/data/schema.py`
- `matcher/datasets/contracts.py`
- `matcher/datasets/candidate_data.py`
- `matcher/datasets/batching.py`
- `matcher/training/config.py`
- `matcher/training/engine.py`
- `matcher/training/initialization.py`
- `matcher/evaluation/metrics.py`
- `文档/记录/单任务记录/第四批次/README.md`
- `文档/记录/批次记录/第4批次/第4批候选预训练初始化.md`
- `文档/记录/总记录.md`
- `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`
- `文档/mapping/计划执行映射.md`
