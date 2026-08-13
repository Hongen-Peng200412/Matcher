# 第 2 批 V 质心 mean/max 六分类

- 科学任务：候选六分类预训练；mean/max；不读取语言、配体图或身份监督。
- 配置：`configs/experiments/batch2/candidate_v_centroid_mean_max_a800.yaml`。
- 直接对照：第 0 批 mean/max 六分类；改变局部视野并加入 12 项模态统计和 4 项采样统计。
- 预定资源：前三个完整 Matcher 全部提交后，使用首个自然空闲的受管 A800；物理批量 16、梯度累积 2、global batch 32。
- W&B：project `AdaLigand-Matcher-Batch2`，group `matcher-batch2-candidate-v-centroid-sampling`。
- 状态：未提交；不得越过尚未提交的第 2 批完整 Matcher。
- 提交命令、Job、release、launch、W&B run、首步验收和最终 validation：提交后追加，当前不得虚构。
