# Handoff: Matcher 第 5 批六项并行运行与联合身份 v4 接续

Date: 2026-08-15

## Current State

活动 Goal 继续覆盖 Matcher 第 1–5 批开发、validation 头对头比较、六张受管 GPU 调度、W&B 分批管理和三层实验记录。当前分支为 `codex/matcher-batch5-matrix-launch`；第 5 批六项配置的实现端点为 `5548a1021729ac514896cb41a54bfd599cf26517`，等价学习端点为 `62c59f299612f4752f84aafec45916a2ceef602d`，实际触发与首步记录提交为 `c872f93`。五项新增任务使用冻结 release `/home/penghongen/Feedback/Matcher/releases/Matcher_d14dd1c3cee4/Matcher`，既有双门 `0.02` 使用冻结 release `/home/penghongen/Feedback/Matcher/releases/Matcher_1e75c58542e5/Matcher`。

第 5 批当前为 6 项已决定、6 项已提交，六张受管 GPU 均有具名任务。2026-08-15 03:28 的只读验收确认六个 Slurm allocation 均为 `RUNNING`、六个 W&B run 均为 `running`；六个 `after_lock` 全部保留，没有 `pre_lock` 或 `kill_lock`。五项新增任务均已完成首个 global batch 32 参数更新，损失与裁剪前梯度范数有限；继续运行的双门 `0.02` 已完成 epoch 0–6，当前 BEST selection score 是 epoch 6 的 `0.6888498731`。

六项 W&B run 均位于 entity/project `pencounkdual-111/AdaLigand-Matcher-Batch5`。四项门值任务使用 group `matcher-batch5-gate-initialization`；两项裁剪任务使用 group `matcher-batch5-gradient-clipping`。

calibration 清单与缓存可以作为非评估准备，但在全部迭代结束且用户再次明确确认前，不得执行 calibration 模型推理、读取正式 calibration 指标或用其选择方案。当前没有 heartbeat，也不得创建与 Goal 模式冲突的 heartbeat。

## Completed

- Job `340899` 继续双门 `0.02/0.02`：launch `train_experiment_job340899_20260814T232001_a4`，W&B run `yd20vpcv`，当前 BEST 为 epoch 6 的 `0.6888498731`。
- Job `340893` 已触发 graph-only `0.02/0`：launch `train_experiment_job340893_20260815T032614_a4`，W&B run `c78pugvk`，首步总损失 `3.6096024122`。
- Job `340898` 已触发 inventory-only `0/0.02`：launch `train_experiment_job340898_20260815T032604_a4`，W&B run `ri9hh667`，首步总损失 `3.6129506605`。
- Job `340897` 已触发双门 `0.1/0.1`：launch `train_experiment_job340897_20260815T032605_a4`，W&B run `57uwuxb6`，首步总损失 `3.6054730160`。
- Job `340894` 已触发 `gradient_clip_norm=5`：launch `train_experiment_job340894_20260815T032512_a10`，W&B run `bh2s3lm7`，首步总损失 `3.6789785181`。
- Job `340895` 已触发 `gradient_clip_norm=200`：launch `train_experiment_job340895_20260815T032343_a6`，W&B run `ufny2evy`，首步总损失 `3.6789785181`。
- 一次性控制器只管理上述五个自然空闲 allocation，只删除对应 `try_lock`；它没有删除 `after_lock/pre_lock/kill_lock`、创建 heartbeat、取消 Slurm Job 或读取 calibration。
- 第 5 批科学配置与触发状态机均经独立科学正确性和严格技术表达审查判定 PASS；首步与 epoch 0–6 运行事实已经回填单任务、批次、总记录、ExecPlan 和目标草案。

## Decisions

- 当前六项继续冻结 `closed_set_v3`，以维持与第 2 批零门 oracle 的直接可比性；运行中的任务不得热替换成联合身份 v4。
- `00/10/01/11` 构成门值 `0.02` 的 2×2 因子比较；双门 `0.1` 只作共同初始尺度探针，不冒充第五个因子格。
- `clip5` 与 `clip200` 是同一 A800 资源条件下的主要裁剪对照。只有新轨迹中 `preclip_norm > 200` 的比例不超过 `0.1%`，才可把后者称为近似不裁剪；必须同时报告总更新数、触发数、最大范数和最小缩放系数。
- 联合身份 v4 只使用 train/validation 开发与审计。若不存在身份拆分，除 schema、路径、摘要和新增 SMILES 元数据外，Dataset 科学张量、固定权重前向及评估的 `loss`、`metrics`、`sample_count`、`identity_eligible_count` 必须精确一致；墙钟与吞吐不作逐值等价要求。
- 若 v4 确有拆分，未受影响 PDB 必须以单 PDB 或 singleton batch 精确一致；受影响 PDB 逐项报告旧新身份槽、顶层 `PreparedLigand.smiles`、occurrence、固定语言表示、target 和有效监督变化。v4 需要另建零门 oracle 基线，不得把 v3 的门或裁剪收益直接迁移归因。
- mean/max 与 candidate-conditioned attention 摘要拼接、正式批次脚本转正、全库注释循环和 `matcher/data` 进一步分层属于后续工作；不得阻断当前已经准备并获准触发的实验或让 GPU 空闲。

## Open Questions

- `00/10/01/11` 的主效应与交互在各自自然早停 BEST 上是否稳定，双门 `0.1` 是否显示明显尺度敏感性？
- `clip5/clip200` 的实际裁剪率、最小缩放系数、稳定性和完整 validation 是否支持放宽默认裁剪阈值？
- 联合身份 v4 在 train/validation 中实际拆分多少 PDB、身份槽、occurrence 和身份有效候选？用户给出的全量 17 个同 PDB 多 SMILES 组不能直接当作 train/validation 结果。

## Next Actions

1. 完成当前记录提交的双线 Git 学习端点与 `Learn/CUMULATIVE` 快进；不得修改运行任务、锁或 release。
2. 在六项运行期间实现联合身份 v4，只构建和读取 train/validation，并先建立 v3→v4 清单、Dataset、前向与评估等价门禁；不得访问 calibration。
3. 只在完整 validation、BEST 刷新、自然早停或失败等明确事件回填六项任务；不建立 heartbeat，不进行无事件高频轮询。
4. v4 科学实现冻结后，分别执行科学正确性与严格技术表达独立审查，再决定是否创建新契约零门 oracle 基线；不得把它混入当前 v3 第 5 批。
5. 在实验和 v4 主线稳定后，再整理正式批次脚本、目录 README、注释循环以及可选的数据子包进一步分层。

## Files To Reopen

- `文档/记录/目标草案.md`
- `文档/记录/总记录.md`
- `文档/记录/批次记录/第5批次/第5批证据驱动优化.md`
- `文档/记录/单任务记录/第五批次/`
- `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`
- `文档/mapping/计划执行映射.md`
- `configs/experiments/batch5/`
- `matcher/data/manifest.py`
- `matcher/datasets/candidate_data.py`
- `matcher/evaluation/metrics.py`
- `训练与运行/README.md`
