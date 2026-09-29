# Handoff：CryoAtom2 小分子身份比较收口，全配体测试暂停

日期：2026-09-29

## 当前状态

Matcher 的已知真实位点比较使用冻结的 K=10 清单：nxu 小分子 84 PDB／357 个 occurrence 查询点，其中多身份视图为 19 PDB／143 点。真实受体与 CryoAtom2 预测受体条件下的 nxu 三方法结果均已完成；全配体 NxU 两种受体条件均暂停，未经用户通知不得恢复。全程未读取 calibration，未训练，也未在测试点拟合阈值。

## 已完成

- CryoAtom2 条件的全部 357 个查询点都有三方法正式逐点结果或明确评分状态。逐点 micro Top-1：density_only-0 为 286/357，density_only-106 为 303/357，官方 EMERALD-ID 为 263/357。多身份视图分别为 72/143、89/143、69/143。PDB 平均、类别支持数与评分覆盖见[结果页](../../../测试结果与追溯/测试结果总览/nxu小分子CryoAtom2测试.md)。
- 单通道 density_only-0 的原始实验密度、查询点与 PDB 内身份集合在两种受体条件下相同，逐点核对后复用已冻结的真实受体预测，不伪称第二次前向。density_only-106 在 H200 Job `390383` 通过门禁并从训练 release 快照完成正式 357 点前向。EMERALD-ID 的 357 点最终状态为 313 scored、42 no_prepared_identity、2 no_valid_score；没有 query_failed。
- CryoAtom2 受体中 `9nc9`、`9r7l`、`9rmi` 的超长链名曾使 22 个查询点在 Rosetta 前失败。旧目录完整保存在服务器 `CryoAtom2/EMERALD-ID/attempts/chain_name_pre_rosetta/nxu/`。输入适配仅对链名超长的预测受体生成唯一两字符链名并保留虚拟查询链 X；实际官方评分门禁通过后，从原冻结查询及完整候选身份补跑这 22 点。原始失败不是正式方法评分失败。
- 正式两视图汇总使用 Job `390207`，release `/home/penghongen/Feedback/Matcher/releases/Matcher_2351adf02d9c/Matcher`，launch `/home/penghongen/Feedback/Matcher/launches/390207/run_emerald_all_job390207_20260929T103801_a5`，正式命令 `bash 训练与运行/sh/known_site/summarize_cryo_small.sh`。服务器原件在 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/CryoAtom2/summary/nxu/{all_retained,multi_identity}/`，本地 JSON 与图片按原哈希复制到 `测试结果与追溯/测试结果总览/nxu小分子CryoAtom2测试图片/`。完整来源、正式命令、门禁命令和 SHA-256 见[CryoAtom2 独立日志](../../../test_真实场景(Structure)/执行记录/CryoAtom2-nxu.md)。
- 真实受体 NxU 数组 `390206` 先前经精确 kill_lock 安全中断：995 点中 317 点有完整逐点结果，678 点尚未完成，不能记为评分失败或汇总 NxU 指标。其旧产物、release、launch 和日志保留。用于 CryoAtom2 的 CPU Job `390207` 与 H200 Job `390383` 最终处于 try_lock、after_lock 存在、无 pre_lock／kill_lock 和残留进程；仅删除它们各自的精确 after_lock 后均退出 Slurm 队列。其余旧 CPU Job 已正常释放，空闲数组元素 `390217` 先前按用户授权取消。没有继续启动 NxU。
- 本地实现、正式薄脚本、门禁、计划、共享／独立日志、结果文档与图片当前均为未暂存或未跟踪状态，没有提交、重置或覆盖用户修改。代码根为 `test_真实场景(Structure)/known_site/`，正式脚本在 `训练与运行/sh/known_site/`，门禁在 `test_真实场景(Structure)/smoke/`。后续 Git 收口必须重新核对双线历史，不把本次暂停误报为已提交。

## 决定

- 只比较同一冻结 nxu 查询与每个 PDB 的完整精确 SMILES 候选集合；多身份视图从同批预测筛选，不重新运行或截尾。
- EMERALD-ID 的身份准备或评分未成功时保留原查询点与候选身份，按固定分母贡献零正确，并单列有效评分覆盖。
- 用户要求先报告本次 nxu 收口并将 Goal 暂停。真实受体 NxU 与 CryoAtom2 NxU 均等待后续明确通知；本次不自动续跑。

## 下次行动

等待用户通知后再评估全配体 NxU 的续跑安排。先只读核对服务器保留的 317 份完整结果、678 个未完成点、清单和资源状态，再决定真实受体 NxU；它完成后才能按用户另行指示考虑 CryoAtom2 NxU。双线 Git 整理也应在恢复目标后按项目规则执行，不把未暂存改动遗忘。

## 需要重开的文件

- `test_真实场景(Structure)/执行记录/CryoAtom2-nxu.md`：本轮正式 release、launch、命令、失败尝试、SHA-256、资源退出。
- `test_真实场景(Structure)/执行记录/总日志.md` 及三个方法日志：两种受体条件的实验状态。
- `测试结果与追溯/测试结果总览/nxu小分子CryoAtom2测试.md`：本轮已冻结结果和图。
- `test_真实场景(Structure)/参考材料 & 想法/Matcher_EMERALD-ID 比较计划书.md`：K=10 科学口径与仍未完成的 NxU。
