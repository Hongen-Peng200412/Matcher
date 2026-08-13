# 第 2 批 SMI-TED V 质心 oracle

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；mean/max；oracle。
- 配置：`configs/experiments/batch2/full_smi_ted_v_centroid_oracle_a100.yaml`。
- 直接对照：第 1 批 SMI-TED 五类 oracle；只改变局部视野与四项采样统计。
- 预定资源：受管 Job `340897`，单张 A100，16 CPU；物理批量 8、梯度累积 4、global batch 32。
- W&B：project `AdaLigand-Matcher-Batch2`，group `matcher-batch2-full-v-centroid-sampling`。
- 状态：未提交；等待最终门禁、双审、release 和 v2 缓存验收。
- 提交命令、release、launch、W&B run、首步验收和最终 validation：提交后追加，当前不得虚构。
