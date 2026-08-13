# 第 2 批 V 质心候选条件注意力六分类

- 科学任务：候选六分类预训练；未增强 candidate-conditioned cross-attention；不读取语言、配体图或身份监督。
- 配置：`configs/experiments/batch2/candidate_v_centroid_cross_attention_a800.yaml`。
- 直接对照：第 0 批未增强候选条件交叉注意力；改变局部视野并加入 12 项模态统计和 4 项采样统计。
- 预定资源：第 2 批前四项都已提交后，使用下一张自然空闲的受管 A800；物理批量 16、梯度累积 2、global batch 32。
- W&B：project `AdaLigand-Matcher-Batch2`，group `matcher-batch2-candidate-v-centroid-sampling`。
- 状态：未提交；这是第 2 批最后一个任务，不得提前越序。
- 提交命令、Job、release、launch、W&B run、首步验收和最终 validation：提交后追加，当前不得虚构。
