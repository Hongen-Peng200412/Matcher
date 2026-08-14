# Handoff: Matcher 第 4 批候选预训练初始化

Date: 2026-08-14

## Current State

活动 Goal 是 Matcher 第 1–5 批开发、validation 头对头比较和六张受管 GPU 调度。当前分支为 `codex/matcher-closed-set-rewrite`；恢复工作时先读取 `文档/记录/目标草案.md`，再用 `git log -1` 和 `git status` 核对实际端点。工作树包含尚未提交的第 4 批候选预训练 BEST 初始化基础；用户要求在该提交收口后，把 `matcher/` 的正式模块按数据、加载、训练和评估职责做一次不改变科学行为的独立目录重构。

第 3 批仍由冻结 release `/home/penghongen/Feedback/Matcher/releases/Matcher_d15132948a18/Matcher` 承担，内容摘要为 `d15132948a1885bae9a2019ac55175503c5c15459f043145147fe4a6d3598f89`。四个非重复任务已经全部启动：Job `340893/340898/340899/340894` 分别对应 oracle/detached/joint/增强六分类，W&B run ID 为 `4iqynr6i/b3vua4zk/z41ltqiv/7k0bcaek`。Job `340897` 继续此前任务，Job `340895` 保留给后续最低编号且已经准备好的任务；不得用重复实验填充。

六个受管 allocation 只包括 Job `340893/340894/340895/340897/340898/340899`。不得取消、重启、释放或删除任何 `after_lock`，不得触碰用户另行提交的任务。自动化 `matcher-1-5` 的持久状态为 `PAUSED`，当前没有活动 heartbeat；只在明确运行事件发生时做人工检查。

calibration 清单与缓存可以准备，但在全部批次结束且用户再次明确确认前，不得运行 calibration 模型推理、读取正式指标或用于方案选择。

## Completed

- `matcher/initialization.py` 已收缩为正式科学契约：语义核对、来源从零条件、候选结构兼容、参数名/形状/dtype 预检，以及只加载 `candidate_encoder`。
- `matcher/training.py` 在模型迁移设备和 optimizer 构造前执行初始化，并只把语义来源记录写入 `resolved.json.initialization`。
- `matcher/experiment_config.py` 只允许完整 Matcher 使用绝对 `BEST.pt`；相对路径、`LAST.pt`、阶段缩写和候选预训练继承参数均拒绝。
- `matcher/tests/test_initialization.py` 已加入非候选哨兵、随机状态、历史字段迁移、从零来源和失败前参数不变回归。
- 初始化入口现在还拒绝正确参数名下的非张量 checkpoint 值，并在任何目标参数变化前失败；`source_best_score` 的契约明确绑定 `evaluation.metrics.six_class.macro_f1`，即只平均真实支持数大于 0 的类别。
- `文档/记录/单任务记录/第四批次/README.md` 已建立。当前第四批 0 项任务已决定、0 项已提交；README 冻结未来每个任务必须保存的科学变量、配置、release、审计报告、Slurm、W&B、首步、validation 和结论字段。
- 当前冻结候选通过 pytest `118 passed`、mypy 40 个正式源码文件零问题、pyflakes、compileall、19 份 YAML 解析、8 个 shell 的 `bash -n`、临时审计脚本编译和 diff-check。Batch4 记录、总记录、ExecPlan、映射和训练 README 已回填到同一快照。

## Decisions

- 用户明确援引 `python-writing-style-cn`：正式代码禁用哈希。来源 `launch.json`、`run_cmd.sh`、release manifest、文件 SHA-256 和同字节解析/摘要全部属于一次性运维取证，不得进入 `matcher` 包、训练入口或 `resolved.json`。
- 当前临时审计脚本为 `tmp/batch4_source_audit/audit_candidate_pretrain_source.py`。只在选定实际来源 BEST 后于服务器运行，报告只能写入 Matcher diagnostics，并回填批次/单任务记录；脚本不进入正式包或默认命令。
- 主线初始化仍要求 checkpoint、resolved、阶段摘要、运行摘要、训练 YAML、W&B online 收口、候选结构和七项科学数据配置在语义上自洽。
- 来源候选阶段的 YAML 与 resolved 都必须明确 `initialize_candidate_from=null`。历史恢复只允许三个模型开关缺失→false、局部裁剪策略缺失→fixed_center、候选 identity_scope 缺失→null。
- 初始化只改变 `candidate_encoder`，且发生在设备迁移和 optimizer 构造前；身份相关分支与随机状态保持目标自身初值。
- 第 4 批任务组合等待第 1–3 批完整 validation；不把缺失实验当成低分。第 3 批四个非重复任务已经全部启动，后续不得重复提交。

## Open Questions

- 科学正确性与严格技术表达两项独立复审均为 PASS，阻断项为零。
- 第 1–3 批尚未全部自然收口，因此具体采用哪个候选预训练 BEST、初始化哪些完整 Matcher 组合仍未决定。
- 一次性运维报告的最终文件名、来源 checkpoint 和 release 路径只能在来源方案确定后填写。
- `matcher/` 目录重构尚未建立 `talk/refactor/` 全文件清单；不得在 Batch4 提交和双线 Git 状态核对前直接移动正式模块。

## Next Actions

1. 检查完整 diff、暂存范围和 `git diff --cached --check`，再提交第 4 批初始化基础。
2. 核对双线 Git 端点后，在 `talk/refactor/` 建立 `matcher/` 每个正式文件的职责、调用者、生命周期、测试和目标位置清单；把目录重构作为独立行为等价提交。
3. 只在首步、完整 validation、自然收口或失败等明确事件发生时检查受管 Job；不启用固定 heartbeat，不重复提交第 3 批。
4. 第 1–3 批自然结束后，只用 validation 选源；随后运行一次性来源审计并建立最小第 4 批 YAML、单任务记录和独立 W&B project/group。
5. 不运行 calibration 模型推理，不删除 `after_lock`，不释放 GPU，不触碰用户任务。

## Files To Reopen

- `文档/记录/目标草案.md`
- `CLAUDE/memory/handoffs/2026-08-14-matcher-batch4-initialization.md`
- `matcher/initialization.py`
- `matcher/training.py`
- `matcher/experiment_config.py`
- `matcher/tests/test_initialization.py`
- `matcher/tests/test_metrics_and_config.py`
- `训练与运行/README.md`
- `文档/记录/批次记录/第4批次/第4批候选预训练初始化.md`
- `文档/记录/单任务记录/第四批次/README.md`
- `文档/记录/总记录.md`
- `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`
- `文档/mapping/计划执行映射.md`
- `tmp/batch4_source_audit/audit_candidate_pretrain_source.py`
