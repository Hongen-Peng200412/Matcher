# Handoff: Matcher 最终摘要拼接与收口 Goal

Date: 2026-08-16

## Current State

当前活动 Goal 只剩本地 Git/worktree 收口。五个新增 concat_summary 任务、全部实验汇总、两份指定结果文档、三个推荐变体从头训练脚本和全库细粒度注释均已完成。运行记录与注释资产已经汇入 `codex/matcher-final-integration` 的 135 个未提交差异路径；当前只剩唯一完整终审点名修订的窄复核、双线提交与树等价、推进 `Learn/CUMULATIVE`，以及逐工作树资产核对后的单主工作树清理。本段不提前声称这些 Git 后置条件已经完成。

五个固定收尾任务已经全部自然早停，后续不再新增实验。精度优先推荐是 A100、seed3409、LR `1e-4`、摘要拼接、零门、clip200；另保留 A100 seed3408 摘要拼接 clip1 高分复现和 mean/max LR `2e-4` 简单结构入口。

五个固定收尾任务均已自然早停并完成结果回填。Job `340893/340894/340895/340897/340898/340899` 六个受管 allocation 已全部按锁协议安全释放：最终只读验收的 `squeue` 计数为 `0`，四类锁均不存在，六份 summary 均保留，`/storage` try-lock 替身文件计数为 `0`。没有调用 `scancel`，没有读取 calibration。

## Completed

- A800 零门 clip200 自然收口：BEST/last=`7/9`、selection=`0.6915842525509563`；4470 次更新中裁剪触发 0 次，最大裁剪前梯度范数 `71.21785736083984`。
- 第 1/5 项已自然早停：BEST/last=`7/9`、selection=`0.6956405636500063`；4470 次更新中裁剪触发 0 次。Job `340895` 已按锁协议安全释放，六个目标 allocation 已释放 3 个、剩余 3 个。
- 第 2–5 项配置已在实现端点 `ffc020354ff2bc1169eacd01a5fc6d3ffd41fd57` 与学习端点 `2f80bc9880591c69c1e51f7125bb4f989ab22b9f` 完成双线 Git，树对象均为 `7cd1241a6d184134a5c9638cef8a659bccb64f9a`；内容寻址 release 为 `Matcher_d3d65ad54fce`。
- 第 3–5 项受保护控制器两项独立终审均为 PASS，已经一次执行并完成 launch、W&B 与首步验收。第 2 项任务单在触发前建立；其控制器也已完成两项独立 PASS，并在 ENOSPC 处置后正式完成 launch、W&B 与首步验收。
- 第 2 项控制器科学正确性与严格技术表达均为 PASS；处理 `/home` ENOSPC 后，a241、W&B `d9thnzeq`、运行根和首步均已建立。失败尝试未进入训练，空 launch、缓存、旧日志和原 launch 树均按可恢复方式清单化或迁移保留。
- Job `340893` 已完成完整结果回填与锁协议释放；这是六个目标 allocation 中的第一个安全释放事件。
- A100、seed3408 摘要拼接 2×2 已完整收口；四个 selection 为 `0.7009100312/0.6979373129/0.6968598986/0.6903778566`。双门0.02与clip200在该 seed 的主效应均为负，两个 clip200 任务实际 0 次裁剪；零门clip1相对同 seed mean/max 提高 `1.306916%`，但与 A800 seed3407 的负结构效应方向不一致。
- 最终第 2 项 A800 mean/max 零门 clip200 已自然早停：BEST/last=`3/5`、selection=`0.6889974930159317`、2682 次更新中裁剪触发 0 次。Job `340894` 已按锁协议安全释放；这是第二个安全释放事件。
- 细粒度注释 A–F/H 批均已独立终审 PASS；H 批精确覆盖 34 份实验 YAML 与现有 7 份配置 README。G 批测试代码按用户裁定不注释，仅继续作为门禁。85 项注释资产已经映射到最终整合树：77 个路径逐字节相等，6 个实现或配置交叉路径保留最新 concat 逻辑并移植已审注释，`matcher/model/README.md` 与注释清单 2 个最终整合文档按当前事实更新。
- A800、seed3407 摘要拼接 2×2 已完整收口：四格 selection 为 `0.6475402763/0.6939342114/0.6915842526/0.6956405637`。零门 clip1 是异常弱轨迹；门值与 clip200 的修复效应强烈不加和，组合单元相对配对 mean/max clip200 只高 `0.964165%`。
- 最终第 5 项 A100、seed3409 mean/max 零门 clip200 已自然早停：BEST/last=`7/9`、selection=`0.6925343658028632`；4470 次更新中裁剪触发 0 次。它相对同 seed 的 mean/max clip1 仅低 `0.024890%`。Job `340899` 已安全释放，六个目标 allocation 当前已释放 4 个、剩余 2 个。
- 最终第 3 项 A100、seed3408 mean/max 零门 clip200 已自然早停：BEST/last=`7/9`、selection=`0.6944095969666253`；4470 次更新中裁剪触发 0 次。配对摘要拼接高 `0.352861%`，本 mean/max clip200 相对 clip1 高 `0.367368%`。Job `340897` 已安全释放，六个目标 allocation 当前已释放 5 个、只剩 1 个。
- 最终第 4 项 A100、seed3409 摘要拼接零门 clip200 已自然早停：BEST/last=`11/13`、selection=`0.7057373297694586`；6258 次更新中裁剪触发 0 次。摘要拼接比配对 mean/max 高 `1.906471%`。六个目标 allocation 已全部安全释放。
- 三组零门 clip200 结构配对的 selection 变化为 `+0.375438%/+0.352861%/+1.906471%`，平均 `+0.878257%`。方向一致，但收益主要由 seed3409 拉高；两份最终报告已经把它表述为小幅正收益且方差明显。
- `文档/记录/结果总记录.md` 与 `文档/记录/结果分析与最优结果.md` 已完成；`训练与运行/sh/最优变体推荐/` 已建立三个零参数从头训练脚本和 README。Shell 语法、正式 YAML 解析、科学字段、Markdown 链接、格式、禁用运维内容、关键百分比和 diff-check 均通过。

## Decisions

- 五项任务已固定：第 1 项是 A800 seed3407 摘要拼接双门0.02、clip200；其余四项是 A800 seed3407 mean/max 零门、clip200，A100 seed3408 mean/max 零门、clip200，以及 A100 seed3409 的摘要拼接与 mean/max 零门、clip200 配对。完成这五项后不再新增实验。
- 已恢复 `try_lock` 的 allocation 在科学决定尚差当前运行结果时可以短期保留；判定不再继续后果断删除精确 `after_lock` 正常释放，不用 `scancel`。
- 每个后续注释批次最多一次完整审查；首审 FAIL 时，只允许一次清单内定点修订和一次窄复核。
- calibration 继续禁止读取；不建立 heartbeat。

## Git Integration Preflight（历史整合前快照）

2026-08-16 执行整合前的只读复核确认 `Learn/CUMULATIVE=2f80bc9880591c69c1e51f7125bb4f989ab22b9f`，它与 `Learn/matcher-summary-concat-controls-final` 指向同一提交对象；该对象当时是按提交者时间计算的唯一最新提交。仓库当时有 17 个 worktree，其中 `C:/Users/15919/Desktop/Matcher_summary_concat_controls_final_learn` 是位于该累计端点且状态干净的可复用整合树。

主工作树当前有 50 个未提交资产路径，包含新增结果文档、三个推荐脚本和既有运行记录；它们与累计端点变更只有 `训练与运行/sh/README.md` 一个重叠路径。细粒度注释树有 85 个未提交资产路径，与累计端点重叠 6 个：`configs/experiments/initial/README.md`、`configs/experiments/joint_identity_v4/README.md`、`matcher/model/candidate_encoder.py`、`matcher/model/config.py`、`matcher/model/fusion.py`、`matcher/training/initialization.py`。四个 Python 必须保留累计端点的摘要拼接可执行代码，再移植学习注释；两个 README 必须合并当前配置与字段说明。

旧 `Matcher_comments_round1` 有 8 个失败样稿路径，其中三份 talk 文件不在细粒度注释树中；用户已经否决该样稿，最终不整合。`Matcher_batch5_lr_2e4` 只有工作区状态/换行脏标记，`git diff --name-only` 为空，没有独有语义资产。主工作树的 ignored `tmp/` 包含运行证据，任何恢复、切换或清理都不得触碰。

上述信息是用户确认前的历史预检边界，不代表当前状态。用户随后已经明确确认 17-worktree 可恢复整理方案；恢复包已经建立，运行记录与注释资产已经进入未提交整合树。工作树仍未物理清理，必须等本轮窄复核、双线提交和端点等价全部通过后再执行。

同一干净累计端点已使用外置 Python/pytest 缓存完成全套基线，结果为 `184 passed in 405.24s`；测试后累计 worktree 的 tracked/untracked 状态计数仍为 0。系统策略拒绝递归删除本轮 `%TEMP%/matcher_final_preflight_20260816104739887`，该目录位于工作树之外且不含项目资产；不要为了清理它绕过工具策略。

## Open Questions

- 无待用户裁定的整理问题。用户已确认 17-worktree 可恢复整理方案；仍须按恢复包、端点等价、逐树资产核对和最后单工作树的顺序执行。

## Next Actions

1. 对唯一完整终审点名的 concat 注释、映射计数、两份结果文档和当前状态记录执行一次窄复核；PASS 后提交实现历史并按人类阅读顺序重建学习历史。
2. 证明实现端点与学习端点树等价，推进 `Learn/CUMULATIVE`；逐个核对恢复包和分支后清理 16 个隔离 worktree，只保留 `C:/Users/15919/Desktop/Matcher`。
3. 复核主工作树干净、`Learn/CUMULATIVE` 是按提交者时间唯一最新提交、六个服务器 Job 仍已释放，再把 Goal 标为完成。

## Files To Reopen

- `文档/记录/目标草案.md`
- `文档/记录/总记录.md`
- `文档/记录/结果总记录.md`
- `文档/记录/结果分析与最优结果.md`
- `训练与运行/sh/最优变体推荐/README.md`
- `文档/记录/批次记录/联合身份v4/摘要拼接六项实验.md`
- `文档/记录/单任务记录/联合身份v4/摘要拼接/README.md`
- `文档/记录/单任务记录/联合身份v4/摘要拼接/seed3407_lr1e-4_a800_双门0.02_clip200.md`
- `文档/记录/单任务记录/联合身份v4/摘要拼接/seed3407_lr1e-4_a800_meanmax_零门clip200.md`
- `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`
- `文档/mapping/计划执行映射.md`
- `configs/experiments/joint_identity_v4/README.md`
- `C:/Users/15919/Desktop/Matcher_comments_fine_grained/talk/Matcher_全库细粒度注释清单.md`
