# 第 4 批 SMI-TED 候选预训练初始化 oracle

## 任务身份与科学对照

- 状态：已于 2026-08-15 01:56（Asia/Shanghai）自然早停；W&B online、BEST/LAST、阶段与运行摘要均正常收口，Slurm allocation 保留。
- 目标任务：用第 2 批 mean/max 六分类预训练 BEST 初始化完整 Matcher 的候选编码器，再训练 oracle 身份损失。
- 直接基线：第 2 批从零 `batch2_smi_ted_v_centroid_oracle_a100`，BEST selection score `0.6981048482`。
- 来源：Job `340894` 的 `batch2_candidate_v_centroid_mean_max_a800`，BEST epoch 1，validation macro-F1 `0.5303632887`。
- 来源 checkpoint：`/storage/penghongen/tmp/Matcher/experiments/batch2_candidate_v_centroid_mean_max_a800/train_experiment_job340894_20260814T093012_a7/batch2_candidate_v_centroid_mean_max_a800/BEST.pt`。
- 来源审计：`/storage/penghongen/tmp/Matcher/diagnostics/batch4_source_audit_20260814/batch2_meanmax_job340894_a7_final.json`，已通过；完整命令与边界见[批次来源审计证据](../../批次记录/第4批次/第4批候选预训练初始化.md#一次性来源审计证据)，不进入训练主线。
- 唯一科学变化：相对直接基线只把 `candidate_encoder` 从随机初始化改为上述 BEST；语言、配体图与身份打分分支仍随机初始化。

## 配置、资源与追踪

- YAML：`configs/experiments/batch4/full_smi_ted_v_centroid_pretrained_mean_max_oracle_a100.yaml`。
- 计划资源：受管 Job `340897`，单张 A100、16 CPU；物理批量 8 × 梯度累积 4 = global batch 32。
- DataLoader：train/validation 各 4 个 spawn 非持久工作进程；每工作进程缓存上限 16 GiB。
- 实现提交：`704ce849fd8be0dbf27148d7dd23d500e9052070`；等价学习端点：`e3f9dfad45cde4ecbc09d85ef400c76eccf36406`。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_ce65027869f4/Matcher`。
- launch：`/home/penghongen/Feedback/Matcher/launches/340897/train_experiment_job340897_20260814T200348_a3`。
- 运行根：`/storage/penghongen/tmp/Matcher/experiments/batch4_smi_ted_v_centroid_pretrained_mean_max_oracle_a100/train_experiment_job340897_20260814T200348_a3`。
- W&B：online project `AdaLigand-Matcher-Batch4`，group `matcher-batch4-pretrained-mean-max`，run ID `mtsmn19a`，[在线曲线](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch4/runs/mtsmn19a)。

## 提交与启动证据

控制端在全部三项共同门禁通过后执行以下一次性激活命令；受保护脚本先原子安装三份动态命令，再按 oracle、detached、joint 顺序删除三个精确 `try_lock`：

```powershell
& "$env:USERPROFILE\.codex\tools\Invoke-ProjectSsh.ps1" `
  -Command 'bash -s' `
  -InputFile 'C:\Users\15919\Desktop\Matcher\tmp\batch4_launch\activate_batch4.sh'
```

本任务实际动态训练命令为：

```bash
exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" \
  --plan configs/experiments/batch4/full_smi_ted_v_centroid_pretrained_mean_max_oracle_a100.yaml
```

- 激活控制器只读审查版本：`E17BE1021E76223B3435A4FCE892ED5AB243E0A5FFCEB293D8157DFB05BE1203`；该摘要只属于 Git 忽略目录中的一次性运维脚本，不进入 YAML、`matcher` 或 `resolved.json`。
- allocation runner 轮询后于 20:03:48 建立 launch；因此实际 launch 墙钟时间晚于同批另外两项，不改变控制端固定触发顺序。
- 启动后 `try_lock_340897` 已删除，`after_lock_340897` 仍保留，`pre_lock_340897` 与 `kill_lock_340897` 均不存在。
- `resolved.json.initialization` 记录来源 epoch 1、macro-F1 `0.5303632887`，严格加载的 `missing_keys/unexpected_keys` 均为空。

## 验收与结果

- 启动前已通过正式 YAML 解析、来源严格初始化、本地机械检查、独立科学审查和严格技术表达审查。
- 首个已核实快照为 global step 29：总损失 `0.8443033`、六分类损失 `0.1953649`、ligand/binding 区域损失 `0.6243452/0.5895063`、身份损失 `0.5275532`，身份有效数 11。
- 同一步裁剪前梯度范数 `8.8875952`，data wait `0.0017231 s`，compute `3.4315415 s`，吞吐 `9.3206 candidate/s`，CUDA 峰值 allocated/reserved 为 `14541019136/19891486720 bytes`；全部为有限值。
- BEST/last epoch 为 `7/9`，global step `4470`，`stopped_by_patience=true`，selection score `0.6976087078`。
- BEST validation 的六分类 macro-F1、五类固有 Top-1、精确子类条件 Top-1、宽松五类端到端 F1 与 small_molecule 兼容端到端 F1 依次为 `0.5267788260/0.8703956344/0.8910472973/0.6750000000/0.6792097836`；ligand/binding micro Dice 为 `0.2855827681/0.0824023964`，有效身份候选数为 `1466`。
- 相对第 2 批从零 oracle，selection score、macro-F1、固有 Top-1、条件 Top-1、宽松端到端和 small_molecule 兼容端到端分别变化 `-0.071%/-0.328%/-0.623%/+0.126%/-0.130%/-0.624%`；ligand Dice `-9.962%`，binding Dice `-18.706%`。分类与身份近似持平，但两个区域指标均下降，其中 binding 超过约 10%；不能据此把预训练初始化称为跨目标稳定改进。
- 训练命令正常退出后 Job `340897` 自然恢复 `try_lock_340897`；`after_lock_340897` 保留，allocation 未取消或释放。最终判断只使用 validation；calibration 未读取。
