# train/validation 清单 v4 构建与等价审计

## 任务边界

本任务把 Matcher 身份从 `object_key` 扩展为 `(object_key, PreparedLigand.smiles)`，只构建和比较 train/validation。它不读取 calibration，不申请 GPU，不修改第 5 批六项任务的动态命令或锁，也不把一次性摘要与服务器路径取证写入正式 `matcher` 包、YAML 或 `resolved.json`。

## 本地冻结与发布

- 实现分支：`codex/matcher-joint-identity-v4@f4f0b63265fe439de013836a9d51c3d5e26e48a7`。
- 学习分支与累计学习端点：`Learn/matcher-joint-identity-v4@f3aa704453dca61a2d96d6cf3cf061618a59e28b`、`Learn/CUMULATIVE@f3aa704453dca61a2d96d6cf3cf061618a59e28b`。
- 两端树摘要均为 `8bde60448fa365cbed13385dd176704788df36d4`，逐文件比较无差异。
- 最终本地门禁：pytest `162 passed in 147.18s`；mypy 检查 45 个 Python 文件零问题；compileall、pyflakes 和 `git diff --check` 通过。
- 安全同步只更新 `/home/penghongen/My_Project/Matcher`，没有删除远端项目、release、实验产物或锁；`tmp/` 按同步规则排除。
- 内容寻址 release：`/home/penghongen/Feedback/Matcher/releases/Matcher_936323d942c5/Matcher`。

Job `342285` 暴露跨 PDB 类别边界后，本地规则修订与零门配置从 `Learn/CUMULATIVE@f3aa704453dca61a2d96d6cf3cf061618a59e28b` 再次完成双线 Git：实现端点为 `codex/matcher-v4-zero-gate-baseline@18fe8cb7f2e24cb1552f307517dfa6fb93a88e27`，学习端点与当前累计学习端点均为 `011b289dd26422a98573508bafe9fc20520a9bc5`，两端树摘要均为 `ba7d910b617a3cc87b8ed7ae1531e886a1d29403`。后续重建必须使用由该学习端点创建的新 release，不能复用上列旧 release。

## 清单构建提交

2026-08-15 运行以下服务器命令：

```bash
release_root=/home/penghongen/Feedback/Matcher/releases/Matcher_936323d942c5/Matcher
cd "${release_root}"
bash 训练与运行/submit_task.sh \
  --task-root "${release_root}" \
  --sh 训练与运行/sh/build_manifest.sh \
  --resource cpu --partition cpu --qos cpu96 --cpus 2 --mem 16G \
  --job-name matcher-joint-identity-v4-manifest -- \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --stage1-root /storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477/artifacts/Find_0 \
  --output-root /storage/penghongen/tmp/Matcher \
  --coverage-threshold 0.30 --selection-seed 3407 \
  --splits train validation
```

- Slurm Job：`342285`。
- 资源：`cpu/cpu96`、2 CPU、16 GiB、无 GPU、未设置时间上限。
- launch：`/home/penghongen/Feedback/Matcher/launches/342285/build_manifest_job342285_20260815T060240_a1`。
- 任务开始时间：`2026-08-15 06:02:38`（服务器记录）。
- 最终状态：`FAILED/1`；运行时间 `00:37:29`，节点 `cnode01`。
- 目标清单：`/storage/penghongen/tmp/Matcher/manifests/closed_set_v4/manifest.json`。提交前该文件与 `.build.lock` 均不存在；构建器只接收 `train validation`。

## 首次构建结果与规则修订

Job `342285` 在正式发布前的类别一致性门禁失败，没有生成 `manifest.json` 或任何正式 JSONL；目标目录只留下构建过程使用的零字节 `.build.lock`。该文件不是成功标记，也不据此删除或覆盖任何已发布产物。CPU allocation 已结束，没有申请 GPU、修改六个受管 GPU 任务或读取 calibration。

失败键为 `CCD:SAH` 与同一顶层 `PreparedLigand.smiles`：

- `train/7fgg`：`peptide_like`，12 个 occurrence；
- `train/7pnx`：`small_molecule`，1 个 occurrence；
- `validation/7pnz`：`small_molecule`，1 个 occurrence。

三个 PDB 内部各自只有一个类别。Matcher 身份轴按 PDB 构造，六分类子类是当前 PDB 的辅助标签，因此不同 PDB 的类别差异不应拆分联合身份或阻断清单。正式规则改为：同一 PDB 内的联合键类别必须唯一；同一小写 `pdb_id` 的候选记录不得同时发布到多个划分，仅出现在 `excluded_pdb.jsonl` 的整 PDB 排除记录不构成候选重叠。不同 PDB 的差异进入 manifest 与一次性审计报告的结构化诊断，seen 仍只按联合键和语言家族计算。修订必须形成新的内容寻址 release；旧 release `Matcher_936323d942c5` 不再用于重试。

修订后的学习端点版本化正式测试为 `160 passed in 155.57s`；主工作区根目录合并运行时自动发现 ignored 审计器的 12 项测试，因此为 `172 passed in 150.01s`。mypy 检查 45 个 Python 文件零问题，compileall、pyflakes、29 份正式 YAML 解析和 diff-check 通过；一次性审计器单独运行仍为 `12 passed`。科学正确性与严格技术表达两项独立终审均为 PASS，阻断项为零。双线 Git 已收口；该证据仍不等于真实清单构建成功。

## 第二次构建提交

2026-08-15 先从干净的 `Learn/CUMULATIVE@011b289dd26422a98573508bafe9fc20520a9bc5` worktree 安全同步正式树，再创建内容寻址 release `/home/penghongen/Feedback/Matcher/releases/Matcher_28e4bacd0dbc/Matcher`。远端正式 Conda 环境已确认 schema 为 4、候选级划分互斥规则生效、v4 零门配置可解析。登录 shell 的系统 Python 因缺少 PyYAML 首次只读检查失败；该检查没有创建 release 或提交任务，随后改用正式 Conda 环境完成同一门禁。

第二次提交命令与第一次相同，只替换 release 并使用新的 Job 名：

```bash
release_root=/home/penghongen/Feedback/Matcher/releases/Matcher_28e4bacd0dbc/Matcher
cd "${release_root}"
bash 训练与运行/submit_task.sh \
  --task-root "${release_root}" \
  --sh 训练与运行/sh/build_manifest.sh \
  --resource cpu --partition cpu --qos cpu96 --cpus 2 --mem 16G \
  --job-name matcher-joint-identity-v4-manifest-r2 -- \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --stage1-root /storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477/artifacts/Find_0 \
  --output-root /storage/penghongen/tmp/Matcher \
  --coverage-threshold 0.30 --selection-seed 3407 \
  --splits train validation
```

- Slurm Job：`342303`。
- 资源：`cpu/cpu96`、2 CPU、16 GiB、无 GPU、未设置时间上限。
- launch：`/home/penghongen/Feedback/Matcher/launches/342303/build_manifest_job342303_20260815T075525_a1`。
- launch 绑定 release：`/home/penghongen/Feedback/Matcher/releases/Matcher_28e4bacd0dbc/Matcher`。
- 2026-08-15 07:55:46 的事件式查询：`RUNNING`，节点 `cnode01`，已运行约 21 秒。
- 最终状态：`COMPLETED/0:0`；运行时间 `00:35:03`，结束于 2026-08-15 08:30:27。正式发布 `manifest.json`、train/validation 各自的 candidates/identities JSONL 与 `excluded_pdb.jsonl`，没有读取 calibration。

## 一次性清单审计结果

本地最终审计器、输入快照器和提交脚本逐字节更新到 `/storage/penghongen/tmp/Matcher/diagnostics/joint_identity_v4_audit_20260815T0615/`。首次提交 Job `342316` 在 Conda 激活阶段因脚本过早启用 Bash 未定义变量检查而以 `FAILED/1` 结束，审计器尚未运行且没有发布报告。修订只把 `set -u` 移到正式 Conda 环境激活之后；Git Bash 语法与 LF 行尾复核通过。第二次提交 Job `342318` 完整发布输入前后快照、`summary.json` 与 `manifest_equivalence.json`，随后因 `--require-zero-affected` 发现真实 seen 差异而按设计以 `FAILED/2` 结束。输入前后快照逐字节相同，故该退出码是科学门禁结果，不是竞态或损坏。

- train：候选 `14,312→14,312`，身份 `1,477→1,477`；唯一受影响 PDB 为 `7pnx`。
- validation：候选 `6,495→6,495`，身份 `536→536`；唯一受影响 PDB 为 `7pnz`。
- 两个划分的候选键集合与 JSONL 顺序均相同，候选语义变化列表为空，整 PDB 排除记录相同，可监督候选数在两种语言家族中均无变化。
- 没有同一 PDB 内的联合身份拆槽。`7pnx/candidate_53` 与 `7pnz/candidate_45` 的 `CCD:SAH` 仍各占一个身份槽，并继续选择同一个 occurrence、同一语言归档和同一 LigandObject 图。
- 唯一变化是 `CCD:SAH` 的 `seen_in_train_by_language` 在 `7pnx/7pnz` 中由两种语言均为 false 改为 true；原因是相同联合键已作为 `train/7fgg` 的正身份出现。`7pnx` 的每种语言 seen 身份数 `7→8`、可监督候选数保持 41；`7pnz` 为 `7→8`、可监督候选数保持 42。
- `CCD:SAH` 在不同 PDB 的局部六分类仍存在 peptide_like/small_molecule 差异；它按已批准契约只作结构化诊断，不改变 PDB 内身份轴类别。

因此 v3→v4 不是“全字段零差异”，但模型输入与监督预计不变；只有评估元数据中的 seen/unseen 分层归属允许发生上述两处变化。下一门禁必须证明全部 Dataset 科学数组、身份 target、固定权重前向、损失、总体六分类/身份/区域指标和 selection score 精确一致，并把 validation seen/unseen 子统计的变化限制到 `7pnz` 对应候选；墙钟和吞吐仍不比较。

## Dataset 与 Batch 等价审计

一次性 Dataset 审计只读使用既有 v3/v4 清单、既有 voxel/PP 缓存和上述清单审计报告；缓存缺失时失败，不允许现场创建缓存。两个 release 分别在隔离 Python 进程中构造 `CandidateDataset`，再比较候选级科学输入、监督、联合身份轴和正式 collate 结果。抽样范围包含清单审计涉及 PDB 的全部候选，并为每个划分、每个六分类类别补充最多四个不同的未受影响 PDB；可用 PDB 不足四个时取该类全部，可为零。

首次目录 `/storage/penghongen/tmp/Matcher/diagnostics/joint_identity_v4_dataset_audit_20260815T0900/` 对应 Job `342324`。该任务在 Dataset 实例化前发现 `validation/nucleotide_like` 没有四个未受影响 PDB，因旧抽样规则要求每类恰好四个而以 `FAILED/1:0` 结束；运行时间 `00:00:05`，节点 `cnode01`。失败没有发布正式 `result/`，旧目录完整保留。修订只把抽样规则收窄为“每类最多四个，不足时取全部”，并增加低基数与零未受影响池回归；受影响 PDB 全候选、清单真值绑定和精确比较规则不变。

修订后目录为 `/storage/penghongen/tmp/Matcher/diagnostics/joint_identity_v4_dataset_audit_20260815T0945/`。服务器先运行一次性工具的 11 项回归，随后提交 CPU-only Job `342325`：

- 最终状态：`COMPLETED/0:0`；运行时间 `00:03:08`，节点 `cnode01`，4 CPU，无 GPU。
- 正式报告：`result/report.json`；选择清单：`result/selection.json`；stderr 为空。
- train 的完整 Dataset 长度在 v3/v4 中均为 `14,312`，实际比较 92 个候选、12 个 collated batch。
- validation 的完整 Dataset 长度在 v3/v4 中均为 `6,495`，实际比较 72 个候选、9 个 collated batch。
- 未受影响分层样本中，train 六类各覆盖 4 个不同 PDB；validation 的 background/ion/small_molecule/sugar 各 4 个，nucleotide_like 为全部 2 个，peptide_like 为全部 3 个。
- `7fgg`、`7pnx`、`7pnz` 等清单审计相关 PDB 的全部候选均强制纳入，不被分层抽样替代。
- 报告给出 `model_input_and_supervision_exact=true` 与 `collated_tensor_fields_exact=true`；允许的评估元数据变化列表为空。这证明本门禁范围内的候选级科学输入、监督、身份目标和 batch 张量逐值一致。

该 CPU 审计没有读取 calibration，没有写入 voxel/PP 缓存，也没有修改六个受管 GPU Job、动态命令或任何锁。它还不能替代固定权重模型前向或完整 validation；后者必须另行冻结并独立审查。

## 待回填

1. 在同一空闲 A800、同一 Batch2 oracle `BEST.pt` 和确定性数值设置下比较固定权重前向与完整 validation 科学字段。
2. `loss`、总体 `metrics`、`sample_count`、`identity_eligible_count` 与 selection score 必须精确一致；只允许 `validation/7pnz` 引起的 seen/unseen 子统计重新分层，墙钟和吞吐不比较。
3. v4 零门 oracle 配置已经准备，但只有同卡门禁通过后才决定是否复用 Job `340894`；任何触发仍须保留 `after_lock` 并继续禁止 calibration。
