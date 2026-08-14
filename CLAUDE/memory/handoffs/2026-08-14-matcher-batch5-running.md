# Handoff: Matcher 第 5 批双门 0.02 首项运行

Date: 2026-08-14

## Current State

活动 Goal 继续覆盖 Matcher 第 1–5 批开发、validation 头对头比较和六张受管 GPU 调度。第 5 批科学实现端点为 `Learn/CUMULATIVE@ec012345153ad3d758a47dc88b4d0e81e5e5b0a8`；本 handoff 同时保存该端点发布后的启动与运行事实。

第 5 批当前为 1 项已决定、1 项已提交。`batch5_smi_ted_v_centroid_gate_002_oracle_a100` 正在 A100 Job `340899` 运行；launch 为 `train_experiment_job340899_20260814T232001_a4`，W&B run 为 [`yd20vpcv`](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch5/runs/yd20vpcv)，release 为 `Matcher_1e75c58542e5`。首个 global batch 已完成，首次完整 validation 尚未产生。

六个受管 allocation 不得取消、重启或释放，任何 `after_lock` 均不得删除。2026-08-14 23:20 前后的事件式复核中，第 3 批 oracle/detached 和第 4 批 detached 已自然早停，Job `340893/340898/340895` 恢复 `try_lock`；Job `340894/340897` 继续第 4 批 joint/oracle，Job `340899` 的 `try_lock` 已为本次 Batch5 触发精确删除。不得为了填卡重复首项；其它自然空闲卡只能用于科学动机、配置、测试和独立审查均已准备好的下一项。

calibration 清单与缓存可以作为非评估准备，但在全部迭代结束且用户再次明确确认前，不得执行 calibration 模型推理、读取正式 calibration 指标或用其选择方案。当前没有活动 heartbeat。

## Completed

- 实现提交 `5098f382076795033e3461a1685bd820a928c3f9` 与学习端点 `ec012345153ad3d758a47dc88b4d0e81e5e5b0a8` 逐文件等价；`Learn/CUMULATIVE` 已快进到学习端点。
- 本地最终候选通过 pytest `126 passed`、mypy 检查 45 个 Python 文件、pyflakes、compileall、23 份 YAML 的 24 个阶段正式解析、Shell 语法与 diff-check；科学正确性和严格技术表达两项独立审查均为 PASS。
- 远端 release `Matcher_1e75c58542e5` 通过 23 份 YAML/24 个阶段解析、Matcher Python 源码内存编译和 train/validation 各 3 个真实候选读取；没有读取 calibration。
- Git 忽略目录中的一次性控制器 `tmp/batch5_launch/activate_batch5_gate.sh` 经双审 PASS 后执行，只管理 Job `340899`，只删除 `try_lock_340899`，并保留 `after_lock_340899`。
- 首步总损失为 `3.6084863407`，裁剪前梯度范数 `42.3180923462`，数据等待/计算 `6.7542922919/24.3319061401 s`，吞吐 `1.0293957323 candidate/s`；第一次更新后的 graph/inventory 门值为 `0.0200999994/0.0200999975`。
- 第 3 批 oracle/detached 的 BEST/last epoch 为 `11/13` 与 `7/9`，selection score 为 `0.6921287290/0.6900402677`；四项第 3 批任务至此全部自然早停，综合结果不支持优先采用增强注意力。
- 第 4 批 detached 的 BEST/last epoch 为 `3/5`，selection score `0.6871224546`；相对第 2 批从零 detached 的核心指标方向混合，等待 oracle/joint 后再作批次结论。

## Decisions

- 首项相对第 2 批 V 质心 mean/max oracle 只把 `initial_graph_gate/initial_inventory_gate` 从 `0.0/0.0` 改为 `0.02/0.02`；另只改变任务名称和 W&B project/group。
- W&B 使用 `AdaLigand-Matcher-Batch5/matcher-batch5-gate-initialization`；validation 的 `identity_contribution/validation/*` 只作贡献尺度诊断，不进入损失、selection score、checkpoint 或 calibration。
- 首次完整 validation 前不同时铺开第二个门值、梯度裁剪阈值或学习率任务。下一项必须根据 validation 核心指标和门控贡献尺度建立独立动机、直接对照、配置、测试与单任务记录。
- release/launch/文件摘要取证只属于 Git 忽略的临时运维脚本，不进入正式 `matcher` 包、YAML 解析字段或科学产物。

## Open Questions

- 首次完整 validation 的 selection score、六分类、五类身份、small_molecule 兼容身份、区域指标和 `identity_contribution/validation/*` 是否支持继续采用双门 `0.02/0.02`？
- 若首项没有明显负优化，下一项应优先测试一个梯度裁剪替代阈值、一个学习率对照，还是基于贡献诊断拆分 graph/inventory 门值？不得在证据出现前预先铺开矩阵。
- Job `340893/340895/340898` 的收口摘要和关键 validation 指标已回填对应单任务与批次记录；如要在这些 allocation 上接续新任务，仍须重新核对最新 launch、运行摘要、动态命令和当前四类锁。

## Next Actions

1. 在首次完整 validation、BEST 刷新、自然收口或失败等明确事件出现时，事件式读取 Job `340899`；不建立 heartbeat。
2. 把首次 validation 的完整指标、混淆情况、身份分层和贡献诊断回填到单任务、批次、总记录与 ExecPlan，再决定第 5 批下一项。
3. 对任何新任务先建立 `文档/记录/单任务记录/第五批次/` 下的独立记录，写明动机、直接对照、唯一变量、Job、W&B 和验收方式；配置与测试通过后才允许受保护触发。
4. 继续保留六个 `after_lock`，不修改用户另行提交的任务，不读取 calibration 做正式评估。

## Files To Reopen

- `文档/记录/目标草案.md`
- `文档/记录/总记录.md`
- `文档/记录/批次记录/第5批次/第5批证据驱动优化.md`
- `文档/记录/单任务记录/第五批次/SMI-TED_V质心_mean_max_oracle_双门0.02.md`
- `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`
- `文档/mapping/计划执行映射.md`
- `configs/experiments/batch5/`
- `matcher/model/ligand_matcher.py`
- `matcher/training/engine.py`
