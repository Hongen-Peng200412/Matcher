# 第 4 批 SMI-TED 候选预训练初始化 detached

## 任务身份与科学对照

- 状态：已于 2026-08-14 22:37:16（Asia/Shanghai）自然早停；BEST/last epoch 为 `3/5`。
- 目标任务：用第 2 批 mean/max 六分类预训练 BEST 初始化完整 Matcher 的候选编码器，再训练 detached 身份损失。
- 直接基线：第 2 批从零 `batch2_smi_ted_v_centroid_detached_a100`，BEST selection score `0.6864829756`。
- 来源：Job `340894` 的 `batch2_candidate_v_centroid_mean_max_a800`，BEST epoch 1，validation macro-F1 `0.5303632887`。
- 来源 checkpoint：`/storage/penghongen/tmp/Matcher/experiments/batch2_candidate_v_centroid_mean_max_a800/train_experiment_job340894_20260814T093012_a7/batch2_candidate_v_centroid_mean_max_a800/BEST.pt`。
- 来源审计：`/storage/penghongen/tmp/Matcher/diagnostics/batch4_source_audit_20260814/batch2_meanmax_job340894_a7_final.json`，已通过；完整命令与边界见[批次来源审计证据](../../批次记录/第4批次/第4批候选预训练初始化.md#一次性来源审计证据)，不进入训练主线。
- 唯一科学变化：相对直接基线只把 `candidate_encoder` 从随机初始化改为上述 BEST；A100→A800、物理微批与缓存上限仅是资源适配，global batch 仍为 32。

## 配置、资源与追踪

- YAML：`configs/experiments/batch4/full_smi_ted_v_centroid_pretrained_mean_max_detached_a800.yaml`。
- 计划资源：受管 Job `340895`，单张 A800、16 CPU；物理批量 16 × 梯度累积 2 = global batch 32。
- DataLoader：train/validation 各 4 个 spawn 非持久工作进程；每工作进程缓存上限 32 GiB。
- 实现提交：`704ce849fd8be0dbf27148d7dd23d500e9052070`；等价学习端点：`e3f9dfad45cde4ecbc09d85ef400c76eccf36406`。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_ce65027869f4/Matcher`。
- launch：`/home/penghongen/Feedback/Matcher/launches/340895/train_experiment_job340895_20260814T200131_a5`。
- 运行根：`/storage/penghongen/tmp/Matcher/experiments/batch4_smi_ted_v_centroid_pretrained_mean_max_detached_a800/train_experiment_job340895_20260814T200131_a5`。
- W&B：online project `AdaLigand-Matcher-Batch4`，group `matcher-batch4-pretrained-mean-max`，run ID `yep6nt4l`，[在线曲线](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch4/runs/yep6nt4l)。

## 提交与启动证据

控制端执行的完整一次性激活命令见 [oracle 任务记录](SMI-TED_候选预训练初始化_oracle.md#提交与启动证据)；三项共用同一受保护控制器，本任务没有独立重复触发命令。

本任务实际动态训练命令为：

```bash
exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" \
  --plan configs/experiments/batch4/full_smi_ted_v_centroid_pretrained_mean_max_detached_a800.yaml
```

- 激活控制器只读审查版本：`E17BE1021E76223B3435A4FCE892ED5AB243E0A5FFCEB293D8157DFB05BE1203`；只用于一次性运维留证。
- allocation runner 轮询后于 20:01:31 建立 launch；墙钟时间不代表控制端删除锁的先后顺序。
- 启动后 `try_lock_340895` 已删除，`after_lock_340895` 仍保留，`pre_lock_340895` 与 `kill_lock_340895` 均不存在。
- `resolved.json.initialization` 记录来源 epoch 1、macro-F1 `0.5303632887`，严格加载的 `missing_keys/unexpected_keys` 均为空。

## 验收与结果

- 启动前已通过正式 YAML 解析、来源严格初始化、本地机械检查、独立科学审查和严格技术表达审查。
- 首个已核实快照为 global step 30：总损失 `1.0011232`、六分类损失 `0.1895230`、ligand/binding 区域损失 `0.6826798/0.5773869`、身份损失 `0.6855935`，身份有效数 13。
- 同一步裁剪前梯度范数 `2.2862473`，data wait `0.0005100 s`，compute `2.5781285 s`，吞吐 `12.4096 candidate/s`，CUDA 峰值 allocated/reserved 为 `28267852800/39996882944 bytes`；全部为有限值。
- BEST epoch 3 的 selection score 为 `0.6871224546`；六分类 macro-F1、五类固有 Top-1、精确子类条件 Top-1、宽松五类端到端 F1 和 small_molecule 兼容端到端 F1 依次为 `0.5317979900/0.8547066849/0.8656093489/0.6639600250/0.6396989652`。ligand/binding micro-Dice 为 `0.2603784612/0.0702168392`，身份有效监督数为 `1,466`。
- 相对第 2 批从零 detached，selection score、macro-F1、五类固有 Top-1、宽松端到端 F1 和 binding Dice 依次变化 `+0.093%/+0.966%/+0.481%/+0.726%/+35.134%`；精确子类条件 Top-1、small_molecule 兼容端到端 F1 和 ligand Dice 变化 `-0.911%/-2.186%/-3.264%`。方向混合，没有多个核心指标约 10% 的共同下降或一致改善，单项结果不足以判定候选预训练初始化明显有益或有害。
- 训练命令正常写出运行根与阶段 `summary.json`，`stopped_by_patience=true`；Job `340895` 的 allocation 和 `after_lock_340895` 继续保留，runner 已恢复 `try_lock_340895`。calibration 未读取。
