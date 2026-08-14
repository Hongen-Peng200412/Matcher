# 第 4 批 SMI-TED 候选预训练初始化 detached

## 任务身份与科学对照

- 状态：本地配置、测试和两项独立终审已通过；尚未发布 release 或触发 GPU。
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
- W&B：online project `AdaLigand-Matcher-Batch4`，group `matcher-batch4-pretrained-mean-max`；run ID 与 URL 在触发后回填。
- 实现提交、release、launch、完整控制命令和动态训练命令：冻结并触发后回填。

## 验收与结果

- 启动前必须通过正式 YAML 解析、来源严格初始化、本地机械检查、独立科学审查和严格技术表达审查。
- 触发后记录首个 global batch 32 更新、有限损失与梯度、显存、数据等待、BEST 时间线和训练命令退出状态。
- 最终只用 validation 与直接基线比较 selection score、六分类、身份层次、区域指标、有效监督数和混淆情况；calibration 禁止读取。
