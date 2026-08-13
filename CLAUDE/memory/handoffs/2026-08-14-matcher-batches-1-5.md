# Handoff: Matcher 第 1–5 批开发与六卡调度

Date: 2026-08-14

## Current State

Matcher 第 0 批重建与基线 Goal 已完成，新的第 1–5 批 Goal 已激活。当前分支是 `codex/matcher-closed-set-rewrite`；第 1 批实现端点为 `8f7f56c`，其中 `2467b39` 保存实现与测试，`8f7f56c` 保存冻结门禁和三层账本。源码已安全同步到服务器，尚未启动第 1 批 GPU 训练。

6 个受管 Slurm allocation 为 `340893/340894/340895/340897/340898/340899`。2026-08-14 03:59 只读核验均为 RUNNING 且已空闲，每个都有共享锁目录中的 `try_lock_<job>` 和本 Job 目录中的 `after_lock_<job>`；不得释放资源或触碰用户其它任务。

## Completed

- 第 0 批完成两个六分类阶段和五个完整 Matcher；结果已写入 `文档/记录/批次记录/第0批次/第0批基线.md`。
- 当前第 0 批最高 selection score 是 SMI-TED detached `0.6696320918`；SMI-TED joint 未运行，不能记为低分。
- calibration 入口已实现，但正式推理冻结；曾启动的准备进程已停止，未创建 claim 或结果 JSON。
- I/O 方案已验证：PP 缓存命中后不读完整 probability map；train/validation 默认各 4 个 spawn、非持久工作进程。
- 第 1 批已实现 manifest v3、显式划分物化、五类身份清单、`identity_scope`、12 维候选手动特征、目标身份类别概率特征、多层身份指标及 W&B 职责优先命名；三个 Batch1 YAML 与十份单实验账本已建立。四组手动统计分别紧邻 A/P/V/PP 摘要进入融合层。冻结门禁为 `59 passed`、mypy 38 个源文件、pyflakes、compileall、10 份 YAML、全部 shell 和 diff-check；科学与严格表达双审均 PASS。实现已经提交并安全同步，远端关键源码与三份 YAML 的 SHA-256 和本地一致。
- CPU-only Job `341600` 正在用 release `Matcher_bb899c6e37ed` 物化 `closed_set_v3` 的 train/validation 清单；提交资源为 `cpu/cpu96`、2 CPU、16 GiB，不含 GPU 和时间上限，命令显式使用 `--splits train validation`。清单原子发布前只存在 `.build.lock`，calibration 仍 deferred。

## Decisions

- 第 1 批确定使用 SMI-TED Light 289M。CPU Job `341588` 证实 train 前景候选 `4,192/4,192`、validation `1,466/1,472` 可监督；唯一缺口是 `6rkd` 的六个小分子候选，它们保留但 identity_target=-1。
- small_molecule 兼容指标把输出身份槽限制回小分子清单，提供与第 0 批同清单、同分母的直接比较；它复用五类模型的一次前向，inventory 分支仍看过五类共享清单，不能解释为旧模型的反事实重跑。
- 候选手动特征固定顺序为 A/P/V/PP 各自的 token 数、Stage1 固化概率均值、最大值；A/P 基于完整 80³，V/PP 第 1 批基于固定 48³。
- 五类总体和逐类端到端指标只要求六分类预测为任一非背景类，再由 Matcher Top-1 身份决定是否成功；逐类预测阳性按最终身份类别归桶。`conditional` 仍要求六分类子类精确，小分子兼容端到端仍使用严格 `small_molecule` 门。
- 旧 checkpoint/resolved 缺少 `identity_scope` 和 `use_manual_features` 时，calibration 兼容层按旧任务语义补为 `small_molecule/false`；历史原始 YAML 本身仍必须通过显式新配置迁移，不在正式 YAML 解析器中增加静默默认。
- calibration 清单/缓存可以顺路准备，但不得运行模型、产生正式指标或进入选择结论。

## Open Questions

- 第 2 批真实 `calibration/7z2c/candidate_1` 回归与“不得正式读取 calibration”存在时序边界；可在获准的准备阶段只做几何缓存回归，但不能运行模型或把结果当评估。
- 第 4、5 批的任务矩阵等待前三批 validation 结果。

## Next Actions

1. 等待 Job `341600` 自然完成，核验 manifest v3 的候选数、身份数、六类计数、SMI-TED 有效身份目标数及 calibration deferred 状态。
2. 核验通过后原子替换 Job `340894/340895/340893` 的动态命令；按 oracle/detached/joint 顺序只删除相应 `try_lock`，不删除任何 `after_lock`。
3. 首步成功后记录 Job、release、launch、W&B run ID/URL、配置差异和显存/梯度/I/O。
4. 第 1 批稳定运行后开始第 2 批 V 质心视野实现；不等待第 1 批全部结束才开发，但调度仍服从批次优先级。

## Files To Reopen

- `文档/记录/目标草案.md`
- `configs/experiments/batch1/README.md`
- `configs/experiments/batch1/*.yaml`
- `matcher/manifest.py`
- `matcher/candidate_data.py`
- `matcher/losses.py`
- `matcher/metrics.py`
- `matcher/model/candidate_encoder.py`
- `matcher/model/fusion.py`
- `matcher/model/ligand_matcher.py`
- `matcher/training.py`
- `matcher/tracking.py`
- `文档/记录/总记录.md`
- `文档/记录/批次记录/第1批次/第1批五类身份扩展.md`
- `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`
