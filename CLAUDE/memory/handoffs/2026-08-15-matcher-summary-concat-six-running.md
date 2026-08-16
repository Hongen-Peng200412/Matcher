# Handoff: Matcher 摘要拼接六项运行

Date: 2026-08-15

## Current State

联合身份 v4 的清单、Dataset/Batch 和同卡 GPU 等价门禁均已完成。v3/v4 validation 的 `sample_count=6495`、`identity_eligible_count=1466`、selection score `0.69798148434599` 完全相同；固定权重前向、完整科学 validation、checkpoint、GPU、数值后端和只读缓存清单严格一致。所有开发、训练与选择只使用 train/validation，calibration 未读取。

三个训练随机种子的 v4 学习率矩阵已经全部自然早停。`2e-4` 相对 `1e-4` 的 selection 变化为 `-0.084452%/+0.970219%/+0.585417%`，平均提高 `0.490377%`；两对支持 `2e-4`，一对近似持平地支持 `1e-4`，只能写成可能小幅有利、方向不完全一致。

最后一组 mean/max 与候选条件注意力摘要拼接实验已经全部触发。A800、seed3407、零门 clip1 锚点于 20:03:15 自然早停，BEST/last=`0/2`、selection=`0.6475402763159072`；相对同硬件 mean/max 结构基线降低 `6.434098%`，且六分类、总体身份、small-molecule 身份和两个区域 Dice 都下降，因此该锚点不支持默认采用摘要拼接。其余五项继续训练；截至 20:44，当前 BEST selection 按 Job `340895/340893/340897/340899/340898` 为 `0.6517883443/0.6403636783/0.6351221058/0.6179709176/0.6456118426`，但只完成 epoch 0–1，不作最终判断。六个 Slurm allocation 均为 `RUNNING`，全部保留 `after_lock`；Job `340894` 已于 20:03:19 正常恢复 allocation 根下的 `try_lock`，其余任务没有 `try_lock`，六项均无 `pre/kill`。没有 heartbeat、calibration 读取或 allocation 释放。

| Job | GPU | seed | 双身份门 | clip | launch | W&B run |
|---:|---|---:|---:|---:|---|---|
| `340894` | A800 | `3407` | `0/0` | `1` | `train_experiment_job340894_20260815T183625_a13` | `a9nluhn9` |
| `340895` | A800 | `3407` | `0/0` | `200` | `train_experiment_job340895_20260815T192042_a8` | `80cfblti` |
| `340893` | A100 | `3408` | `0/0` | `1` | `train_experiment_job340893_20260815T192321_a6` | `39c2ca9k` |
| `340897` | A100 | `3408` | `0.02/0.02` | `1` | `train_experiment_job340897_20260815T192308_a6` | `seacrx6z` |
| `340899` | A100 | `3408` | `0/0` | `200` | `train_experiment_job340899_20260815T192320_a6` | `ksa6wb86` |
| `340898` | A100 | `3408` | `0.02/0.02` | `200` | `train_experiment_job340898_20260815T192303_a6` | `4191tn8b` |

W&B entity/project 为 `pencounkdual-111/AdaLigand-Matcher-JointIdentityV4-SummaryConcat`。A800 两项 group 为 `joint-identity-v4-summary-concat-seed-3407`；A100 四项 group 为 `joint-identity-v4-summary-concat-a100-seed-3408`。

## Completed

- 摘要拼接正式结构、零残差精确退化、随机状态隔离、三步任务梯度和 checkpoint 双向失败关闭均已通过版本化回归与独立科学/表达终审。
- 矩阵实现端点为 `codex/matcher-summary-concat-matrix@965f1dbaa1ccbedc5bfbf2ed3ca90bbc40e4ab43`；等价学习端点和当前 `Learn/CUMULATIVE` 为 `36877d1b6e2f301e44f1e4611169f3451b47e4fb`。
- A800 clip1 锚点 release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_81a80171fa02/Matcher`；其余五项 release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_c225fbd2feee/Matcher`。
- 正式解析器通过 40 个计划、41 个阶段；摘要拼接配置专项 `7 passed`。两份一次性激活控制器均取得科学正确性与严格技术表达独立 PASS。
- A800、seed3407、零门 clip1 摘要拼接锚点已经自然早停；BEST validation 的 `sample_count/identity_eligible_count=6495/1466`。其相对同硬件、同 seed、同学习率和同批量 mean/max 基线的 selection 变化为 `-6.434098%`，首个严格结构对照不支持摘要拼接。
- 历史第 1–5 批 21 个固定 YAML 薄启动脚本及各目录 README 已完成三轮复审、双线 Git 和累计学习历史收口。
- 细粒度注释 A 批 `matcher/data` 与 B 批 `matcher/datasets` 均已独立终审 PASS。B 批最终为 `130 passed in 958.08s`，并通过 4/4 Python 去 Docstring AST 等价、mypy、pyflakes、格式、链接和缓存检查。C 批已开始，精确覆盖 `matcher/model/` 的 6 个 Python 文件、新建 `matcher/model/README.md` 和全库注释清单；工作树仍未暂存或提交。

## Decisions

- 摘要拼接六项固定 `LR=1e-4`，目的是与既有 v4 mean/max `1e-4` 基线形成同硬件、同训练种子的结构对照；这不表示学习率矩阵证明 `1e-4` 更优。
- A100 的四格只能解释联合双身份门、裁剪阈值及其交互，不能拆分 graph 与 inventory 的独立效应。A800 两项只复核裁剪方向。
- 每项必须等自然早停后比较 BEST validation 和完整核心指标。clip200 还要回填总更新数、实际裁剪触发数/比例、最大裁剪前梯度范数和最小缩放系数。
- 只在完整 validation、BEST 刷新、自然早停或失败等明确事件检查；不创建 heartbeat。
- calibration 推理和正式指标必须等待用户最终确认。任何 `after_lock` 或 allocation 都不得提前释放。
- 主工作树 `C:/Users/15919/Desktop/Matcher` 位于旧实现分支并保存未提交运行记录；不得丢弃、暂存或混入代码实现提交。

## Open Questions

- A800、seed3407 的首个同硬件结构锚点已经明确下降；A100、seed3408 的同硬件结构锚点是否复现该方向，仍需自然早停结果。
- 联合双门 `0.02/0.02` 与 clip200 在新结构内的主效应和交互尚未知。
- 注释全库收口时，当前旧基点上已 PASS 的目录注释必须重放到 `Learn/CUMULATIVE@36877d1`，并覆盖摘要拼接新增源码与配置；最终等价门禁尚未执行。

## Next Actions

1. 在其余五项出现完整 validation、BEST 刷新、自然早停或失败时，低频核验 summary、W&B 与锁，并同步单任务、批次、总记录、ExecPlan、mapping 和本 handoff 的接续文件。
2. 其余五项自然收口后，先完成 A100 同硬件结构对照，再分析 A100 2×2 的门值、裁剪和交互，并用 A800 clip1/clip200 检查裁剪方向；不读取 calibration。
3. 继续让细粒度注释编写/审查代理逐目录完成 2–3 轮循环。每一批内容 PASS 后再进入下一目录；不得把旧基点局部 PASS 写成最终集成 PASS。
4. 注释全部完成后，从当前唯一最新 `Learn/CUMULATIVE` 整合注释学习历史，补覆盖摘要拼接新增文件，重跑全库等价与门禁，并使新学习端点成为按提交者时间唯一最新提交。
5. 摘要拼接和汇总分析完成后，按既有四锁协议自然释放六个 allocation；此前不删除 `after_lock`。

## Files To Reopen

- `文档/记录/目标草案.md`
- `文档/记录/总记录.md`
- `文档/记录/批次记录/联合身份v4/联合身份v4开发与等价审计.md`
- `文档/记录/批次记录/联合身份v4/摘要拼接六项实验.md`
- `文档/记录/单任务记录/联合身份v4/摘要拼接/README.md`
- `文档/记录/单任务记录/联合身份v4/学习率复现/README.md`
- `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`
- `文档/mapping/计划执行映射.md`
- `configs/experiments/joint_identity_v4/README.md`
- `matcher/model/fusion.py`
- `matcher/model/config.py`
- `matcher/tests/test_model_and_loss.py`
- `matcher/tests/test_metrics_and_config.py`
- `C:/Users/15919/Desktop/Matcher_comments_fine_grained/talk/Matcher_全库细粒度注释清单.md`
