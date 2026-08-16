# Handoff: Matcher 联合身份 v4 学习率矩阵运行

Date: 2026-08-15

## Current State

联合身份 v4 的 CPU 清单、真实清单、Dataset/Batch 和同卡 GPU 四层等价门禁均已完成。GPU 门禁中，v3/v4 validation 长度与 `sample_count` 均为 `6495`，`identity_eligible_count` 均为 `1466`，selection score 均为 `0.69798148434599`；固定权重前向、完整科学 validation、checkpoint 身份、GPU、数值后端和缓存清单严格一致。该过程只读取 train/validation，没有读取 calibration。

正式学习率问题已经按用户裁定改为联合身份 v4 内部的 `LR=1e-4` 对 `LR=2e-4`。三个训练随机种子形成三对同 GPU 类型、同随机种子直接对照；六项 `data.selection_seed` 均为 `3407`。截至 2026-08-15 16:15，六个 Slurm allocation 仍为 `RUNNING`；Job `340894` 的训练已于 15:42:59 自然早停并恢复 `try_lock_340894`，其 BEST/last epoch 为 `5/7`、selection 为 `0.6920686526492751`。其余五个在线 W&B run 继续训练且仅保留各自 `after_lock`；所有 pre/kill 锁均不存在。seed3407 配对尚未闭合，seed3408/3409 两对也仍在运行，不提前判断学习率优劣。

| Job | GPU | 训练随机种子 | 学习率 | launch | W&B run |
|---:|---|---:|---:|---|---|
| `340894` | A800 | `3407` | `1e-4` | `train_experiment_job340894_20260815T115031_a12` | `hvtzju50` |
| `340895` | A800 | `3407` | `2e-4` | `train_experiment_job340895_20260815T122350_a7` | `s80l6dbp` |
| `340893` | A100 | `3408` | `1e-4` | `train_experiment_job340893_20260815T122610_a5` | `l0m0ao2i` |
| `340898` | A100 | `3408` | `2e-4` | `train_experiment_job340898_20260815T122554_a5` | `7csoon5v` |
| `340897` | A100 | `3409` | `1e-4` | `train_experiment_job340897_20260815T122606_a5` | `7n2y1aek` |
| `340899` | A100 | `3409` | `2e-4` | `train_experiment_job340899_20260815T122600_a5` | `rgyq1i8k` |

W&B entity/project 为 `pencounkdual-111/AdaLigand-Matcher-JointIdentityV4`。seed 3407 的 `1e-4` 基线 group 为 `joint-identity-v4-zero-gate-baseline`，`2e-4` group 为 `joint-identity-v4-learning-rate`；seed 3408/3409 各自使用 `joint-identity-v4-learning-rate-seed-3408/3409`。除计划名、阶段名、输出目录和 W&B 命名外，每一对的唯一科学变量是学习率；六项共享 v4 清单、模型、损失、零 graph/inventory 门、裁剪阈值 1、BF16、物理批量 8、梯度累积 4、global batch 32 和从零初始化。A100 与 A800 之间只用于检查学习率效应方向能否复现，不比较硬件效应，也不冒充逐位同轨对照。

主工作树 `C:/Users/15919/Desktop/Matcher` 当前位于 `codex/matcher-v4-zero-gate-baseline@18fe8cb7f2e24cb1552f307517dfa6fb93a88e27`，含尚未提交的运行记录和新单任务记录，不能丢弃或混入别的实现。学习率矩阵配置的实现提交为 `197cb750ba3cb4f06416fcf7a52cd4000595b0df`，其等价学习端点为 `369152756818f04730efd5865ad8fdb18ed9b620`。历史批次入口收口后，当前 `Learn/CUMULATIVE` 已继续快进到 `f6a3ecb9098e9c145b6c004017d7c700470222f8`。seed 3407 的 `1e-4` 使用 release `/home/penghongen/Feedback/Matcher/releases/Matcher_28e4bacd0dbc/Matcher`；其余五项使用 `/home/penghongen/Feedback/Matcher/releases/Matcher_28e1833e0179/Matcher`，两者正式 Matcher 源码相同。

## Completed

- 第 5 批六项均已自然早停。双门、graph-only、inventory-only、双门 0.1、clip5、clip200 的 BEST/last epoch 分别为 `7/9`、`8/10`、`6/8`、`9/11`、`8/10`、`11/13`；selection score 分别为 `0.6957138818452475/0.6914142117510179/0.6932194548785874/0.6937232999295260/0.6908370761804674/0.6960835398398743`。
- A800 `clip200` 共 6258 次更新，裁剪触发 0 次，最大裁剪前梯度范数 `73.61093139648438`，最小缩放系数 `1.0`；本次可称近似不裁剪。它相对同 A800 的 `clip5` 提高 `0.7594357397%`，但只有一个随机种子，不作显著或普遍优势结论。
- v4 清单没有同 PDB 身份拆槽、候选或监督数变化；只有 `train/7pnx` 与 `validation/7pnz` 的 SAH seen 身份数各增加 1。
- v3/v4 Dataset/Batch 与同卡 GPU 门禁均严格通过；历史 v3 只作等价迁移辅助证据。
- 六份 v4 学习率 YAML、矩阵回归、双线 Git 和内容寻址 release 已完成。科学正确性与严格技术表达两项独立审查均为 PASS。
- 六项已完成的每个 validation epoch 都保持 `sample_count=6495`、`identity_eligible_count=1466`。截至 15:35，seed3407 的两项实际学习率分别为下一轮 `2.5e-5/1e-4`；seed3408 的 `1e-4/2e-4` 下一轮均为 `5e-5/1e-4`；seed3409 的 `1e-4/2e-4` 下一轮分别为 `2.5e-5/2e-4`。该异步轨迹不能用于最终学习率结论。
- Job `340894` 已自然早停：BEST/last=`5/7`，selection=`0.6920686526492751`；BEST epoch 的六分类 macro-F1、固有身份 Top-1、条件身份 Top-1、宽松端到端 F1 为 `0.5284033454997990/0.8656207366984994/0.8781275221953188/0.6696750902527075`。该单边结果不构成学习率结论。
- 历史第 1–5 批固定 YAML 薄入口已完成三轮独立审查与双线 Git 收口。实现端点为 `0b0b93a96eb96c7df1d2df11eaa57906005d5bbd`，学习端点为 `f6a3ecb9098e9c145b6c004017d7c700470222f8`，两端树对象同为 `e4d00fd6202858301ba3a9ad8a39db0c04766c15`；`Learn/CUMULATIVE` 已快进到学习端点。
- 一次性五任务控制器只删除 Job `340895/340893/340898/340897/340899` 的精确 `try_lock`；没有删除任何 `after_lock`，没有触碰 pre/kill 锁、创建 heartbeat、释放 allocation 或读取 calibration。

## Decisions

- 正式对照是 v4 `1e-4` 对 v4 `2e-4`；历史 v3 只作辅助比较，不建立专用 v3 运行分支。
- 每一对必须等到自然早停后按 BEST validation 比较；不得用当前 epoch 或跨硬件差异提前归因。
- 只在完整 validation、BEST 刷新、自然早停或失败等明确事件回填，不创建 heartbeat。
- calibration 模型推理和正式指标仍需用户在全部迭代结束后单独确认。
- 一次性 SHA-256、release/launch 取证和控制器只存在于 ignored `tmp`/运维记录，不进入正式 `matcher`、YAML 或 `resolved.json`。
- mean/max 与 conditional-attention 摘要拼接另立科学任务，不混入当前学习率矩阵。

## Infrastructure Boundary

用户已确认两个隔离方案。历史批次薄启动脚本位于 `codex/matcher-batch-launch-layout` 与 `C:/Users/15919/Desktop/Matcher_batch_launch_layout`，原基点为 `Learn/CUMULATIVE@369152756818f04730efd5865ad8fdb18ed9b620`；21 份脚本与 6 份批次 README 已完成三轮编写和独立复审、实现提交与学习历史重建。实现端点与学习端点树一致，当前 `Learn/CUMULATIVE` 为 `f6a3ecb9098e9c145b6c004017d7c700470222f8`，且没有触碰两个既有脏工作树。

旧的 `codex/matcher-learning-comments-round1` 与 `C:/Users/15919/Desktop/Matcher_comments_round1` 只覆盖 5 个 Python 文件，用户已明确判定不符合全库细粒度注释目标；该工作树保留为失败样例，不暂存、不提交、不合并。新的编写工作树为 `codex/matcher-learning-comments-fine-grained` 与 `C:/Users/15919/Desktop/Matcher_comments_fine_grained`，同样基于 `369152756818f04730efd5865ad8fdb18ed9b620`。编写代理与独立审查代理都已重新完整阅读技术表达、代码注释、YAML、Python、文档和双线 Git skills；正式范围为 45 个 `matcher/**/*.py` 文件、34 份正式实验 YAML 及相关 README/talk，并按目录分批执行 2–3 轮编写—审查—修订。

## Next Actions

1. 只在六项 v4 任务出现完整 validation、BEST 刷新、自然早停或失败时做低频状态核验，并同步单任务、批次、总记录、ExecPlan、mapping 和本 handoff。
2. 三对全部自然早停后，在每个同硬件、同随机种子配对内比较 BEST selection 及完整核心指标；跨 GPU 类型只作重复证据。
3. 历史批次薄启动脚本已经完成；继续让新注释工作树按目录完成 45 个 Python、34 份 YAML 与相关 README/talk 的 2–3 轮编写—审查循环，旧窄注释工作树不得进入任何学习或实现分支。最终注释学习端点必须纳入当前 `Learn/CUMULATIVE@f6a3ecb` 的新增批次入口，不得退回旧累计基点。
4. GPU 学习率任务稳定后再另立 mean/max 与 conditional-attention 摘要拼接实验；不得混入当前矩阵。
5. 不读取 calibration，不删除 `after_lock`，不释放六个 allocation，不建立 heartbeat。

## Files To Reopen

- `文档/记录/目标草案.md`
- `文档/记录/总记录.md`
- `文档/记录/批次记录/第5批次/第5批证据驱动优化.md`
- `文档/记录/批次记录/联合身份v4/联合身份v4开发与等价审计.md`
- `文档/记录/单任务记录/联合身份v4/学习率复现/README.md`
- `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`
- `文档/mapping/计划执行映射.md`
- `configs/experiments/joint_identity_v4/README.md`
- `matcher/tests/test_metrics_and_config.py`
- `talk/refactor/Matcher_包目录职责整理.md`
- `C:/Users/15919/Desktop/Matcher_batch_launch_layout/训练与运行/sh/批次训练/README.md`
- `C:/Users/15919/Desktop/Matcher_comments_fine_grained/talk/Matcher_全库细粒度注释清单.md`
