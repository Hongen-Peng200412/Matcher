# Handoff: Matcher 第 4 批运行中与第 5 批准备

Date: 2026-08-14

## Current State

活动 Goal 是 Matcher 第 1–5 批开发、validation 头对头比较和六张受管 GPU 调度。当前分支为 `Learn/CUMULATIVE`；第 4 批科学实现端点是 `704ce849fd8be0dbf27148d7dd23d500e9052070`，等价学习端点是 `e3f9dfad45cde4ecbc09d85ef400c76eccf36406`。恢复时先读取 `文档/记录/目标草案.md`、本 handoff、总记录和第 4/5 批批次记录，再用 `git status` 与服务器只读状态核对是否出现新事件。

六个受管 allocation 仅为 Job `340893/340894/340895/340897/340898/340899`。不得取消、重启或释放这些 allocation，不得删除任何 `after_lock`，不得触碰用户另行提交的任务。当前没有活动 heartbeat；只在完整 validation、BEST 刷新、自然收口、失败或自然出现空闲锁等明确事件发生时检查和回填。

2026-08-14 20:18 的低频只读复核显示六个 Job 均为 `RUNNING`，每项都只有 `after_lock`，没有 `try_lock/pre_lock/kill_lock`。Job `340893/340898/340899` 继续第 3 批三个完整 Matcher，Job `340897/340895/340894` 运行第 4 批三项；当前没有空闲受管 GPU。

calibration 清单与缓存可以作为准备产物，但在全部批次结束且用户再次明确确认前，不得执行 calibration 模型推理、读取正式 calibration 指标或用其选择方案。

## Completed

- 正式 Python 包已整理为 `matcher/data`、`matcher/datasets`、`matcher/model`、`matcher/training`、`matcher/evaluation` 和 `matcher/utils`，科学行为不变；运行中的旧 release 未被热替换。
- 第 4 批只把第 2 批 V 质心 mean/max 候选预训练 BEST 严格加载到完整 Matcher 的 `candidate_encoder`。来源 Job 为 `340894`，来源运行是 `train_experiment_job340894_20260814T093012_a7`，BEST epoch 1，macro-F1 `0.5303632887`。
- 一次性来源审计报告为 `/storage/penghongen/tmp/Matcher/diagnostics/batch4_source_audit_20260814/batch2_meanmax_job340894_a7_final.json`。发布路径与文件摘要只属于该次 Git 忽略目录运维取证，不进入正式包、YAML 解析值或 `resolved.json`。
- 第 4 批实现与学习历史已经等价收口，release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_ce65027869f4/Matcher`。远端 22 份 YAML 正式解析、compileall 和来源检查通过。
- CPU-only Job `341902` 只因临时 smoke 输出使用旧字段名而退出；其候选初始化、随机状态和非候选分支检查已经通过。修正临时输出后的 Job `341903` 为 `COMPLETED/0:0`，没有占用 GPU。
- 三份动态命令经受保护控制器一次安装，并按 oracle→detached→joint 删除精确 `try_lock`。控制器的科学正确性和严格技术表达两项独立只读审查均为 PASS；全部 `after_lock` 保留。

## Running Batch 4 Tasks

| 任务 | Job / GPU | launch | W&B | 首个已核实快照 |
| --- | --- | --- | --- | --- |
| oracle | `340897` / A100 | `train_experiment_job340897_20260814T200348_a3` | [`mtsmn19a`](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch4/runs/mtsmn19a) | step 29；loss `0.8443033`；preclip `8.8875952` |
| detached | `340895` / A800 | `train_experiment_job340895_20260814T200131_a5` | [`yep6nt4l`](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch4/runs/yep6nt4l) | step 30；loss `1.0011232`；preclip `2.2862473` |
| joint | `340894` / A800 | `train_experiment_job340894_20260814T200250_a9` | [`c527adm0`](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch4/runs/c527adm0) | step 30；loss `0.9081250`；preclip `0.9958403` |

三项 W&B project 均为 `AdaLigand-Matcher-Batch4`，group 均为 `matcher-batch4-pretrained-mean-max`。三个 `resolved.json.initialization` 均记录来源 epoch 1、global step 894、macro-F1 `0.5303632887`，严格加载的 `missing_keys/unexpected_keys` 为空。首批损失、梯度、数据等待、吞吐和显存均为有限值。

## Decisions

- 三项与第 2 批对应从零任务做直接头对头；唯一科学变化是候选编码器初始化。oracle 使用 A100 `8×4=global 32`，detached/joint 使用 A800 `16×2=global 32`，硬件适配不改变科学配置。
- W&B online 分项目展示。不要把第 4 批与前几批混入同一 project，也不要因单个首步或单轮指标波动判定改动有害。
- 第 5 批已经有梯度裁剪和 graph/inventory 门值证据，但尚未锁定任务。更早批次非重复任务已经全部提交后，可以使用自然空闲的受管卡准备第 5 批；配置、测试、直接对照和独立审查未完成前不得只为填卡触发。
- 正式代码不保存来源 launch/release SHA-256。临时运维脚本和摘要留在 Git 忽略目录或 diagnostics，不迁回 `matcher` 主线。

## Open Questions

- 第 4 批三项尚未完成首次完整 validation，当前不能比较 selection score、六分类、身份层次和区域指标。
- 第 3 批三个增强完整 Matcher 的最终结果仍需自然事件回填，并与第 2 批 mean/max、未增强注意力对照综合判断增强注意力是否保留。
- 第 5 批首个任务应选梯度裁剪阈值、非零 graph/inventory 门值或其它日志证据方向；必须先补齐贡献诊断和最小对照设计。

## Next Actions

1. 事件式只读检查六个受管 Job；只有完整 validation、BEST 刷新、自然收口、失败或自然空闲时才更新记录，不建立 heartbeat。
2. 第 4 批每次完整 validation 后回填三份单任务记录、批次记录、总记录和本 handoff；最终只使用 validation 与第 2 批对应从零任务比较。
3. 读取 `文档/记录/批次记录/第5批次/第5批证据驱动优化.md`，先完善一个最小、可直接比较的第 5 批候选，再建立配置、测试、逐任务记录和独立双审。
4. 任何 GPU 触发前重新核对批次优先级、Job 身份、动态命令、`try_lock/after_lock`、目标目录、release 和 W&B 分组；只删除目标 `try_lock`，永不删除 `after_lock`。

## Files To Reopen

- `文档/记录/目标草案.md`
- `文档/记录/总记录.md`
- `文档/记录/批次记录/第4批次/第4批候选预训练初始化.md`
- `文档/记录/单任务记录/第四批次/`
- `文档/记录/批次记录/第5批次/第5批证据驱动优化.md`
- `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`
- `文档/mapping/计划执行映射.md`
- `configs/experiments/batch4/`
- `matcher/training/initialization.py`
- `matcher/training/engine.py`
