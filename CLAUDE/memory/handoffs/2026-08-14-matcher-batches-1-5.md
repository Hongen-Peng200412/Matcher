# Handoff: Matcher 第 1–5 批开发与六卡调度

Date: 2026-08-14

## Current State

Matcher 第 0 批重建与基线 Goal 已完成，新的第 1–5 批 Goal 已激活。当前分支是 `codex/matcher-closed-set-rewrite`；HEAD `852c569` 保存第 2 批 V 质心 `48³`、采样特征和 v2 缓存实现。第 1 批三个 GPU 训练与第 2 批三个完整 Matcher 均在运行。

6 个受管 Slurm allocation 为 `340893/340894/340895/340897/340898/340899`。Job `340894/340895/340893` 正在运行第 1 批 oracle/detached/joint；Job `340897/340898/340899` 正在运行第 2 批 oracle/detached/joint。六个 `after_lock` 都不得删除，不得释放资源或触碰用户其它任务。

## Completed

- 第 0 批完成两个六分类阶段和五个完整 Matcher；结果已写入 `文档/记录/批次记录/第0批次/第0批基线.md`。
- 当前第 0 批最高 selection score 是 SMI-TED detached `0.6696320918`；SMI-TED joint 未运行，不能记为低分。
- calibration 入口已实现，但正式推理冻结；曾启动的准备进程已停止，未创建 claim 或结果 JSON。
- I/O 方案已验证：PP 缓存命中后不读完整 probability map；train/validation 默认各 4 个 spawn、非持久工作进程。
- 第 1 批已实现 manifest v3、显式划分物化、五类身份清单、`identity_scope`、12 维候选手动特征、目标身份类别概率特征、多层身份指标及 W&B 职责优先命名；三个 Batch1 YAML 与十份单实验账本已建立。四组手动统计分别紧邻 A/P/V/PP 摘要进入融合层。冻结门禁为 `59 passed`、mypy 38 个源文件、pyflakes、compileall、10 份 YAML、全部 shell 和 diff-check；科学与严格表达双审均 PASS。实现已经提交并安全同步，远端关键源码与三份 YAML 的 SHA-256 和本地一致。
- 06:54 的第 1 批 validation 快照：oracle 最佳 `0.6812668732`、最新 `0.6786411725`；detached 最新并改善到 `0.6756995547`；joint 最新并改善到 `0.6452062205`。三项仍在运行，不据 oracle 一轮约 `0.39%` 的 selection score 回落提前判停。
- CPU-only Job `341600` 使用 release `Matcher_bb899c6e37ed` 和 `cpu/cpu96`、2 CPU、16 GiB 物化 `closed_set_v3`，未申请 GPU 或设置时间上限；现已 `COMPLETED/0:0`。manifest v3 验证为 train `14,312/1,477/4,192`、validation `6,495/536/1,466`，三项依次表示候选数、身份数、SMI-TED 可监督前景候选数；命令只请求 train/validation，calibration 仍 deferred。
- 第 1 批三项复用 release `Matcher_4cf3ef45e8a8`。oracle 为 Job 340894 / launch `...041802_a6` / W&B `h41na5ea`；detached 为 Job 340895 / launch `...041633_a3` / W&B `vry3gxt2`；joint 为 Job 340893 / launch `...041902_a2` / W&B `amdxqwjc`。三者均完成首个 global batch 32 更新，无 OOM、NaN 或 Traceback。
- 第 2 批核心代码已提交：共享候选几何、V 质心裁剪、voxel/PP schema v2、A/P 新相对原点、四项采样特征、候选融合与三条身份打分 MLP。voxel 文件名绑定候选编号与实际 ZYX 起点；首次构造还核对 exp/sim↔F1 的原点、轴序和体素尺寸，与已有 F1↔probability 校验共同闭合几何链。合成回归覆盖内部、六面、半整数取整、部分保留、高 X 端到端几何、各向异性 spacing 和跨产物漂移反例。
- 第 2 批五份 YAML 和三层账本已建立。三个完整 Matcher 预定 Job `340897/340898/340899`；两个候选预训练等待受管 A800 自然空闲。W&B project 固定为 `AdaLigand-Matcher-Batch2`，完整/候选各用独立 group。
- 06:28 的冻结快照通过 pytest `77 passed`、mypy 38 个源码文件零问题、pyflakes、compileall、15 份 YAML 正式解析、8 个 shell 的 `bash -n` 和 diff-check；独立科学终审与严格表达终审均为 PASS，阻断项为零。
- CPU-only Job `341618` 已验证 `calibration/7z2c/candidate_1`：完整 V `170`，旧固定视野 `0`，新起点 `[16,16,32]` 保留 `170`。它只读一个 F1-centered NPZ，没有模型、指标、checkpoint 或 claim。
- 第 2 批提交为 `852c569`，已安全同步；服务器全套门禁为 `76 passed, 1 skipped`。CPU Job `341634` 使用 release `Matcher_3cc75993d69e` 完成 train/validation v2 voxel 与 PP，`COMPLETED/0:0`、耗时 `00:50:45`；计数为 voxel `14,312/6,495`、PP PDB `480/181`。
- 第 2 批三个完整 Matcher 于 07:33:00 按 oracle→detached→joint 触发，复用同一 release `Matcher_3cc75993d69e`。Job/launch/W&B 分别为 `340897/...073305_a2/6soiol5e`、`340898/...073316_a2/i0j1i1hc`、`340899/...073303_a2/s3lj6298`；均完成首个 global batch 32 更新，无 OOM、NaN 或 Traceback。
- 第 3 批一次性只读结构审计已经完成：增强注意力可从原始 A/P/V/PP token 的 Stage1 概率与相对 XYZ 构造每模态独立 bias MLP；末层零初始化严格退化，PP 严格零出口保留。实现时必须覆盖 PP 偏置分支的预期两步梯度唤醒，并防止新增随机初始化扰动未增强对照的共同参数。

## Decisions

- 第 1 批确定使用 SMI-TED Light 289M。CPU Job `341588` 证实 train 前景候选 `4,192/4,192`、validation `1,466/1,472` 可监督；唯一缺口是 `6rkd` 的六个小分子候选，它们保留但 identity_target=-1。
- small_molecule 兼容指标把输出身份槽限制回小分子清单，提供与第 0 批同清单、同分母的直接比较；它复用五类模型的一次前向，inventory 分支仍看过五类共享清单，不能解释为旧模型的反事实重跑。
- 候选手动特征固定顺序为 A/P/V/PP 各自的 token 数、Stage1 固化概率均值、最大值；A/P 基于完整 80³，V/PP 第 1 批基于固定 48³。
- 五类总体和逐类端到端指标只要求六分类预测为任一非背景类，再由 Matcher Top-1 身份决定是否成功；逐类预测阳性按最终身份类别归桶。`conditional` 仍要求六分类子类精确，小分子兼容端到端仍使用严格 `small_molecule` 门。
- 第 0 批 checkpoint/resolved 缺 `identity_scope/use_manual_features/use_sampling_features` 时，calibration 兼容层按旧任务语义补为 `small_molecule/false/false`；第 1 批只缺采样开关时保留 `use_manual_features=true` 并补 `use_sampling_features=false`。历史原始 YAML 本身仍必须通过显式新配置迁移，不在正式 YAML 解析器中增加静默默认。
- calibration 清单/缓存可以顺路准备，但不得运行模型、产生正式指标或进入选择结论。
- 第 2 批采样特征固定顺序为边界限制标志、裁剪中心到完整 V 质心的 L2/L∞ voxel 距离和 V 保留比例；模型分别用 `24√3`、`24` 归一化距离。
- 105 通道仍先在完整 `80³` 构造再按候选起点裁剪，不能改为先裁剪再做密度通道。
- v2 voxel 缓存路径必须同时包含 `candidate_index` 和 `crop_start_zyx`；构建时 F1 体素尺寸必须与 exp/sim 精确一致，F1 原点重算只允许 `5e-5 Å` 的 float32 误差。

## Open Questions

- 第 4、5 批的任务矩阵等待前三批 validation 结果。

## Next Actions

1. 低频监控第 1/2 批，只在完整 validation、失败、自然收口或资源接续时更新三层账本；不删除任何 `after_lock`。
2. 首个自然空闲的受管 A800 运行第 2 批 mean/max 候选任务；下一张运行未增强交叉注意力候选任务，不中断第 1 批且不重复任务。
3. 按已完成的结构审计实现第 3 批增强注意力，完成本地门禁与阶段性双审后，只有在第 2 批五项全部提交后才能按优先级复用空闲 GPU。

## Files To Reopen

- `文档/记录/目标草案.md`
- `configs/experiments/batch1/README.md`
- `configs/experiments/batch1/*.yaml`
- `configs/experiments/batch2/README.md`
- `configs/experiments/batch2/*.yaml`
- `matcher/geometry.py`
- `matcher/voxel_input.py`
- `matcher/pp_input.py`
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
- `文档/记录/批次记录/第2批次/第2批V质心视野与采样特征.md`
- `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`
