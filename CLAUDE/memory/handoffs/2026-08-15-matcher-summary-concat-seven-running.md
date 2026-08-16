# Handoff: Matcher 摘要拼接七项与细粒度注释 D 批

Date: 2026-08-15

## Current State

联合身份 v4 的清单、Dataset/Batch、固定前向和完整 validation 同卡等价门禁均已完成。三对 v4 学习率也已自然早停。当前训练主线只剩最后一组摘要拼接实验：原六项中的 A800 零门 clip1 已自然早停，其余五项继续运行；为复用该空闲 A800，新增同硬件、同训练随机种子、同学习率、同裁剪和同批量的双门 `0.02/0.02` 第七项。

第七项实现端点为 `codex/matcher-summary-concat-a800-gate@11f477ef87dc0253e372894d2d8218201dda9eaf`，等价学习端点和当前 `Learn/CUMULATIVE` 为 `087accb2bc26ca1ba1caa875576890ae1465a3e7`；两端树均为 `8b35837892751757f7790336c477ae826064f3f1`。内容寻址 release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_11b0176436c1/Matcher`。

2026-08-15 22:06（Asia/Shanghai），第七项在科学正确性与严格技术表达两项独立控制器终审均为 PASS 后安全触发。launch 为 `train_experiment_job340894_20260815T220601_a14`，W&B run 为 `8e9ufq3b`，运行根为 `/storage/penghongen/tmp/Matcher/experiments/joint_identity_v4_concat_gate_002_seed_3407_lr_1e4_oracle_a800/train_experiment_job340894_20260815T220601_a14/`。首个 global batch 32 已完成：`identity_eligible_count=7`、计算耗时 `22.0611021398 s`、裁剪前梯度范数 `42.3404655457`，CUDA allocated/reserved 为 `14,212,122,624/19,583,205,376` bytes。

六个受管 allocation 均为 `RUNNING`，六张 GPU 当前都有具名任务。六个 `after_lock` 全部保留，`try/pre/kill` 均不存在。未创建 heartbeat，未读取 calibration，未释放 allocation。

## Running Tasks

| Job | GPU | seed | 双身份门 | clip | launch | W&B run |
|---:|---|---:|---:|---:|---|---|
| `340894` | A800 | `3407` | `0.02/0.02` | `1` | `train_experiment_job340894_20260815T220601_a14` | `8e9ufq3b` |
| `340895` | A800 | `3407` | `0/0` | `200` | `train_experiment_job340895_20260815T192042_a8` | `80cfblti` |
| `340893` | A100 | `3408` | `0/0` | `1` | `train_experiment_job340893_20260815T192321_a6` | `39c2ca9k` |
| `340897` | A100 | `3408` | `0.02/0.02` | `1` | `train_experiment_job340897_20260815T192308_a6` | `seacrx6z` |
| `340899` | A100 | `3408` | `0/0` | `200` | `train_experiment_job340899_20260815T192320_a6` | `ksa6wb86` |
| `340898` | A100 | `3408` | `0.02/0.02` | `200` | `train_experiment_job340898_20260815T192303_a6` | `4191tn8b` |

已完成的 A800、seed3407、零门 clip1 摘要拼接锚点使用 launch `train_experiment_job340894_20260815T183625_a13`、W&B `a9nluhn9`，BEST/last=`0/2`、selection=`0.6475402763159072`。它相对同硬件 mean/max 基线下降 `6.434098%`，不支持把摘要拼接设为默认结构。当前六项仍须自然收口，才能分析非零门、clip200 与交互。

22:26 的最近一次低频 validation 事件把 Job `340895/340893/340897/340899/340898` 的当前 BEST 更新为 `0.6896432316683323/0.6803945989195906/0.6614407794021132/0.6821176224272261/0.6875310488307648`，BEST epoch 为 `5/4/2/4/4`；Job `340897` 最新完成 epoch 3，其余四项最新完成 BEST 所在轮次。新增 Job `340894` 尚未完成首次 validation。六项继续运行，不能按异步中途值形成最终因子结论。

## Annotation State

细粒度注释仍由编写代理和独立审查代理在 `C:/Users/15919/Desktop/Matcher_comments_fine_grained` 并行推进。A 批 `matcher/data`、B 批 `matcher/datasets`、C 批 `matcher/model` 已独立终审 PASS。D 批精确覆盖 `matcher/training/` 全部 9 个 Python 文件、新建 `matcher/training/README.md` 和 `talk/Matcher_全库细粒度注释清单.md`；当前正在编写，尚未冻结、暂存或提交。

注释工作树仍以旧基点做逐目录可执行 AST 等价核验。全库注释完成后，必须把全部已审注释重放到当前 `Learn/CUMULATIVE@087accb2` 及其后续累计学习端点，覆盖摘要拼接新增源码、配置和 README，重新运行树等价与全库门禁；只有新的学习端点成为按提交者时间唯一最新提交后才算收口。

## Next Actions

1. 只在完整 validation、BEST 刷新、自然早停或失败等明确事件低频检查当前六项；同步单任务、批次、总记录、ExecPlan、mapping 与本 handoff，不建立 heartbeat。
2. 六项自然收口后，先做 A100/A800 同硬件直接对照，再分析联合双门、clip200 和交互；不读取 calibration。
3. 让注释编写与审查代理继续按目录冻结—复审循环；每批内容 PASS 后才进入下一目录。
4. 注释全库完成后，从唯一最新累计学习端点建立最终集成工作树，纳入全部已审注释与摘要拼接后续内容，重跑双线树等价、全库测试、mypy、pyflakes、配置解析、README 链接和无缓存门禁。
5. 最终只保留一个主工作树：逐个确认隔离 worktree 的成果已整合、状态干净且无未提交资产后，再按精确路径安全移除；完成后要求 `Learn/CUMULATIVE` 指向按提交者时间唯一最新提交。
6. 摘要拼接与汇总分析完成后，按既有四锁协议自然释放六个 allocation；此前不删除 `after_lock`。

## Files To Reopen

- `文档/记录/目标草案.md`
- `文档/记录/总记录.md`
- `文档/记录/批次记录/联合身份v4/摘要拼接六项实验.md`
- `文档/记录/单任务记录/联合身份v4/摘要拼接/README.md`
- `文档/记录/单任务记录/联合身份v4/摘要拼接/seed3407_lr1e-4_a800_双门0.02_clip1.md`
- `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`
- `文档/mapping/计划执行映射.md`
- `configs/experiments/joint_identity_v4/README.md`
- `C:/Users/15919/Desktop/Matcher_comments_fine_grained/talk/Matcher_全库细粒度注释清单.md`
