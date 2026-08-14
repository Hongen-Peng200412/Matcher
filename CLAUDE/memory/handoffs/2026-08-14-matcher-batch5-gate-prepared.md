# Handoff: Matcher 第 5 批双门 0.02 首项准备

Date: 2026-08-14

## Current State

活动 Goal 仍覆盖 Matcher 第 1–5 批开发、validation 头对头比较和六张受管 GPU 调度。当前实现分支是 `codex/matcher-batch5-gate-diagnostics`，基点为 `Learn/CUMULATIVE@ecddacf`。恢复时先读取 `文档/记录/目标草案.md`、本 handoff、总记录和第 3/5 批批次记录，再核对 Git 与六个受管 Job 的只读状态。

第 3 批增强注意力 joint 已于 2026-08-14 19:08:52 自然早停；Job `340899` 的 Slurm allocation 继续保留，`after_lock_340899` 存在，runner 已自然恢复 `try_lock_340899`。第 4 批 oracle/detached/joint 继续使用 Job `340897/340895/340894` 训练；第 3 批 oracle/detached 继续使用 Job `340893/340898`。不得取消、重启或释放这些 allocation，不得删除任何 `after_lock`，不得触碰用户另行提交的任务。

第 5 批当前为 1 项已决定、0 项已提交。首项预定使用空闲 A100 Job `340899`，但在双线 Git、内容寻址 release、远端门禁和受保护激活控制器全部完成前，不删除 `try_lock_340899`。

calibration 清单与缓存可以作为准备产物，但在全部批次结束且用户再次明确确认前，不得执行 calibration 模型推理、读取正式 calibration 指标或用其选择方案。

## Completed in the Current Candidate

- 正式模型配置新增 `initial_graph_gate/initial_inventory_gate`；第 0–4 批 22 份 YAML 的 23 个阶段显式保持 `0.0/0.0`，历史产物缺字段时只在恢复入口迁移为零。
- validation 正式前向保存语言基线、图门增量和清单门增量；W&B `identity_contribution/validation/*` 只读取同批组件并聚合，不重跑打分器，不进入损失、选模、历史、checkpoint 或 calibration。
- 首项配置为 `configs/experiments/batch5/batch5_smi_ted_v_centroid_gate_002_oracle_a100.yaml`。相对第 2 批 oracle 的唯一科学变化是双门 `0.0/0.0 → 0.02/0.02`；另只改变任务名称和 W&B project/group，global batch 保持 `8×4=32`。
- 第 3 批 joint BEST selection score 为 `0.6763544866`。它与增强六分类都没有支持优先采用增强注意力，因此首项使用第 2 批更简单、更强的 V 质心 mean/max oracle 作为直接基线，其 BEST selection score 为 `0.6981048482`。
- 全套 pytest `126 passed`；mypy 检查 45 个 Python 文件零问题；pyflakes、compileall、23 份 YAML/24 个阶段正式解析、4 个 Shell 语法和 diff-check 通过。独立科学终审为 PASS；严格技术表达追加终审仍在等待最终回填。

## Decisions

- 双门同时改为 `0.02`，本项只能判断组合初值效果，不能单独归因 graph 或 inventory；单任务记录已经明确这一控制变量边界。
- `0.02` 来自既有任务自然学习到的约 `0.01–0.03`，不直接采用缺少证据的 `0.1`。
- 不在同一项中改变梯度裁剪、学习率、注意力结构、候选初始化、损失或随机种子。
- W&B online project/group 固定为 `AdaLigand-Matcher-Batch5/matcher-batch5-gate-initialization`；曲线与第 2 批 oracle 做 validation 头对头。
- 正式代码不加入 launch/release SHA-256 取证；这类证据只属于任务专用 ignored tmp/ops。
- 当前没有活动 heartbeat；只在首步、完整 validation、自然收口或失败等明确事件发生时检查和回填。

## Next Actions

1. 取得严格技术表达追加终审 PASS；如引起修改，重跑相称门禁并更新本 handoff 数字。
2. 在实现分支提交当前候选；按人类理解顺序重建学习分支，验证实现端点与学习端点树等价，再快进 `Learn/CUMULATIVE`。
3. 安全同步累计学习端点，建立内容寻址 release；在服务器正式环境重跑配置解析、导入、compileall 与 Dataset smoke，不读取 calibration。
4. 建立只服务 Job `340899` 的一次性受保护激活控制器；核对前序自然收口、资源、锁、release、目标目录、重复任务和 W&B 配置。只在所有门禁通过后原子安装动态命令并删除 `try_lock_340899`，永不删除 `after_lock_340899`。
5. 实际启动后立即回填单任务、批次、总记录、ExecPlan、mapping 和本 handoff 的提交、release、完整命令、launch、W&B 与首步验收。

## Files To Reopen

- `文档/记录/目标草案.md`
- `文档/记录/总记录.md`
- `文档/记录/批次记录/第3批次/第3批增强候选条件注意力.md`
- `文档/记录/批次记录/第5批次/第5批证据驱动优化.md`
- `文档/记录/单任务记录/第五批次/SMI-TED_V质心_mean_max_oracle_双门0.02.md`
- `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`
- `文档/mapping/计划执行映射.md`
- `configs/experiments/batch5/`
- `matcher/model/ligand_matcher.py`
- `matcher/training/engine.py`
