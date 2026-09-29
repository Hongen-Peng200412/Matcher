# Handoff：已知位点小分子结果收口，全配体对接已启动

日期：2026-09-28

## 当前状态

项目是 Matcher 与官方 EMERALD-ID 的已知真实位点身份比较。冻结 K=10 清单位于 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest/manifest.json`，SHA-256 为 `095f1bdc6b76a98374eca197e7a179bccc1dfe244d66ffcdefa941d112cde211`。小分子 nxu 的 84 PDB／357 点、其中多身份 19 PDB／143 点已完成三方法正式结果和图；全配体 NxU 的 161 PDB／995 点、其中多身份 72 PDB／540 点正在运行官方 EMERALD-ID 对接。两个 Matcher 模型的 NxU 与 nxu 逐点前向早已冻结，不要重跑。全程不读取 calibration。

## 已完成

- nxu 的 357 个官方逐点结果全部落盘：282 个 `scored`、42 个 `no_prepared_identity`、33 个 `no_valid_score`。三方法逐点 micro Top-1 为 density_only-0 的 286/357、density_only-106 的 301/357、EMERALD-ID 的 233/357；多身份视图分别为 72/143、87/143、70/143。固定分母包含官方方法的准备与评分失败。
- 正式汇总 Job `388458` 使用 release `/home/penghongen/Feedback/Matcher/releases/Matcher_09b9eb624d78/Matcher`、launch `/home/penghongen/Feedback/Matcher/launches/388458/run_emerald_job388458_20260928T180709_a4`、命令 `bash 训练与运行/sh/known_site/summarize_small.sh`。服务器两份 JSON 与两张 PNG 在 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/summary/nxu/`；本地按原样复制到 `测试结果与追溯/测试结果总览/nxu小分子真实测试图片/`，逐文件 SHA-256 与服务器相同。
- 小分子结果文档位于 `测试结果与追溯/测试结果总览/nxu小分子真实测试.md`。共享总日志、三个方法单实验日志及映射索引已回填；旧状态段落保留作为历史，不应再引用为当前进度。
- 小分子 CPU 数组 `388458` 的 12 个实际 Job 在 `try_lock` 时无残留任务，删除各自精确 `after_lock` 正常释放，未使用 `scancel`。之后以标准入口提交新的全配体数组 `390206`，直接进入 `debug/Cpu96`，12 元素各 8 CPU、最长 72 小时、`after_hold`。实际 Job `390207`–`390217` 与 `390206` 已 RUNNING；共同 release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_09b9eb624d78/Matcher`，launch 逐项列于 EMERALD-ID 单实验日志。首次 shard 输出合计 995 点、1803 个理论点—身份评分。

## 决定

用户要求先运行并收口小分子，再运行全配体。NxU 不能混入先前 nxu 数组，也不能用其临近结束的 walltime；新数组保留完整时限。每个 occurrence 查询点使用该任务 PDB 内完整候选 SMILES 集合，官方评分失败也保留在清单和固定分母。全部保留视图与多身份视图从同一批预测派生，不重新运行。

## 接下来

低频核查新数组 `390206` 的逐点结果、锁与剩余 walltime；不要在稳定运行时反复写日志，也不要设置 heartbeat。若接近 72 小时仍缺点，只对真正缺结果且进程已结束的查询点建立明确失败记录或用新 allocation 续跑，不能把仍在计算的点提前判错。995 点都有结果或明确失败记录后，在正式 CPU Job 上执行 `bash 训练与运行/sh/known_site/summarize_all.sh`，冻结 NxU 两视图的三方法 JSON 与图。随后填写 `测试结果与追溯/测试结果总览/NxU全量真实测试.md`，更新总日志、三方法日志、mapping、必要 handoff；安全释放本次 CPU 资源。最后按 Matcher 双线 Git 工作流整理当前仍为 unstaged/untracked 的改动，保留用户原有内容。

## 需要重开的文件

- `test_真实场景(Structure)/参考材料 & 想法/Matcher_EMERALD-ID 比较计划书.md`：当前科学契约与停止条件。
- `test_真实场景(Structure)/执行记录/总日志.md` 和同目录三个方法日志：正式 Job、release、launch、失败尝试、资源和结果。
- `测试结果与追溯/测试结果总览/nxu小分子真实测试.md`：已收口的小分子结果与两图。
- `训练与运行/sh/known_site/run_emerald_all.sh`、`summarize_all.sh`：短正式命令入口；门禁命令与它们分开。
