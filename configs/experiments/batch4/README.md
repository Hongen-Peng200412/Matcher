# 第 4 批候选预训练初始化配置

本目录的三份计划都使用第 2 批 `batch2_candidate_v_centroid_mean_max_a800` 的 `BEST.pt` 初始化完整 Matcher 的 `candidate_encoder`。该来源在 validation 的 BEST epoch 1 得到六分类 macro-F1 `0.5303632887`；第 3 批增强候选条件注意力最终为 `0.5276366341`，相对未增强注意力只提高约 `0.07%`，没有超过 mean/max。来源选择只使用 validation，没有读取 calibration。

三份计划保持第 2 批 mean/max 完整 Matcher 的候选集合、V 质心视野、12 项 Stage1 固化统计、4 项采样统计、SMI-TED 289M、五类非背景身份范围、模型结构、损失权重、学习率与 global batch 32。唯一科学变化是候选编码器由同一个预训练 BEST 初始化；完整 Matcher 的语言、配体图、清单交互与身份打分分支仍按目标随机种子初始化。

三份计划的 `initial_graph_gate=0.0` 与 `initial_inventory_gate=0.0` 保持两个身份分数增量分支从严格零开始，因此候选预训练初始化仍是本批唯一科学变化。

- oracle 使用 A100，物理批量 8、梯度累积 4、每工作进程缓存上限 16 GiB。
- detached 与 joint 使用 A800，物理批量 16、梯度累积 2、每工作进程缓存上限 32 GiB。
- 三项均使用 4 个 train worker 与 4 个 validation worker、BF16、patience 2、W&B online project `AdaLigand-Matcher-Batch4` 和 group `matcher-batch4-pretrained-mean-max`。
- `maximum_epochs=null` 表示不设轮数上限；`class_weights=null` 表示六分类交叉熵不做类别重加权；`initialize_candidate_from` 必须是已审计的绝对 `BEST.pt`。

一次性来源审计报告位于 `/storage/penghongen/tmp/Matcher/diagnostics/batch4_source_audit_20260814/batch2_meanmax_job340894_a7_final.json`。报告只用于提交前运维留证，不被正式训练包读取，也不写入模型、损失或选模逻辑。
