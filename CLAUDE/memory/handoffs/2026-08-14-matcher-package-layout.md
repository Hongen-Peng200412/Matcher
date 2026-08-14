# Handoff: Matcher 包目录整理与第 4 批接续

Date: 2026-08-14

## Current State

活动 Goal 仍是 Matcher 第 1–5 批开发、validation 头对头比较、六张受管 GPU 调度、分批 W&B 和三层实验记录。恢复工作时先读取 `文档/记录/目标草案.md`，再核对本文件、Git 状态和服务器事件；不得运行 calibration 模型推理或把 calibration 用于选择。

当前分支为 `codex/matcher-package-layout`。整理前累计基点与 `Learn/CUMULATIVE` 均为 `a74deca9b7a3f6a58909410623ba1c8b99eb9625`；分支已提交逐文件清单 `cc1f70e`，目录移动、导入、Shell/YAML 入口和当前文档已经暂存。科学正确性与严格技术表达两项独立复核均为 PASS，正在完成实现提交与双线 Git 收口。

六个受管 allocation 仅为 Job `340893/340894/340895/340897/340898/340899`。2026-08-14 17:26 后只读核验时六项均为 `RUNNING` 且全部保留 `after_lock`；Job `340895` 与 `340897` 已自然结束当前命令并生成 `try_lock`，其余四项继续第 3 批。不得删除任何 `after_lock`、释放 allocation、触碰用户任务或未经计划重复实验。

## Completed

- 第 4 批候选预训练 BEST 初始化基础已经在整理前端点完成：正式代码只核对来源语义与候选结构并严格加载 `candidate_encoder`；launch/release 与文件字节取证留在一次性 `tmp` 审计，不进入训练主线。
- `matcher/` 已整理为 `data/datasets/model/training/evaluation/utils` 六个职责子包；顶层只保留包入口，新增子包入口只写职责，不批量转发实现。
- 四个正式 Shell 已改用 `matcher.data.manifest`、`matcher.data.precompute`、`matcher.training.experiment` 和 `matcher.evaluation.calibration`；十九份正式 YAML 只更新读取入口说明。
- 对整理前 36 个既有 Python 文件做语法树归一比较，移除导入与模块说明后全部等价。六个测试文件按四组精确得到 `118 passed`；mypy 检查 `matcher/` 下 45 个 Python 文件零问题，pyflakes、compileall、19 份 YAML、8 个 Shell、四个帮助入口、公共导入和真实 `spawn` 工作进程生命周期均通过。
- 独立科学正确性审查确认候选集合、监督、模型状态、随机状态、损失、指标、global batch、持久产物和 calibration 禁读边界均未漂移；严格技术表达审查确认目录职责、依赖方向、入口与记录可以脱离本次对话冷读。两项结论均为 PASS，阻断项为零。
- Batch2 未增强候选条件注意力在 Job `340895` 自然早停：BEST epoch 4，macro-F1 `0.5272674639`，W&B run `12oeeshn`。
- Batch2 V 质心 oracle 完整 Matcher 在 Job `340897` 自然早停：BEST epoch 11，selection score `0.6981048482`，六分类 macro-F1 `0.5285113541`，五类固有 Top-1 `0.8758526603`，宽松端到端 F1 `0.6758793970`，W&B run `6soiol5e`。

## Decisions

- 目录整理是独立的行为等价改动，不混入模型、损失、指标、配置值、缓存 schema、checkpoint schema 或 calibration 治理变化。
- 历史执行记录保留当时的旧模块命令；活动 Shell、YAML、结构概览、mapping 和新 handoff 使用整理后的路径。
- 不建立旧顶层模块兼容包装。新 release 前必须在可变服务器项目根精确清除已移动的旧顶层 Python 文件；不得使用删除整个项目的同步方式，不得改动任何既有 release。
- 来源 launch/release 与 SHA-256 证据只属于一次性 `tmp/<task-id>/` 或具名 `ops/`，不得重新放回 Batch4 正式初始化代码或 `resolved.json`。calibration 自身尚未执行，其一次性 claim 契约不在本轮目录整理中改动。
- Job `340895/340897` 虽已空闲保留，但 Batch4 来源模型必须等待第 3 批增强六分类完成后再由 validation 选择；不得为填卡跳过尚未决定的第 4 批或重复更早任务。

## Open Questions

- 第 3 批增强六分类和三个增强完整 Matcher 尚未全部自然收口；候选预训练 BEST 来源与第 4 批最小任务矩阵仍未确定。
- 当前目录整理尚待实现提交和学习端点等价证明；两项独立复核已经通过。
- 包整理发布前需要对可变服务器项目根执行一次精确旧路径清理；现有内容寻址 release 与运行任务必须保持不变。
- “去 SHA-256”只在 Batch4 来源运维取证边界内已经闭合；若要调整尚未执行的 calibration claim 格式，必须作为目录整理之外的独立治理改动评审。

## Next Actions

1. 提交 `codex/matcher-package-layout` 实现端点，在隔离工作树按人类理解顺序建立学习端点，验证端点等价后快进 `Learn/CUMULATIVE`。
2. 把 Batch2 两项新自然收口结果回填对应单任务、Batch2 批次、总记录与 ExecPlan；继续只读观察第 3 批明确事件。
3. 第 3 批增强六分类自然收口后，只用 validation 在候选预训练模型中选源，运行一次性来源审计，建立第 4 批最小 YAML 与逐任务记录。
4. 新任务发布前安全同步新包，并仅在可变服务器项目根精确删除旧顶层模块；从内容寻址 release 启动，W&B 使用独立 Batch4 project/group。
5. 不读取 calibration 模型结果，不删除 `after_lock`，不释放 GPU，不触碰用户任务。

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
