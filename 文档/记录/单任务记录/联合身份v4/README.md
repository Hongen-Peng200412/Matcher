# 联合身份 v4 单任务记录

本目录保存联合身份 v4 的真实数据准备、等价审计、零门基线、学习率复现和摘要拼接实验。它不追改第 5 批 `closed_set_v3` 历史结果，也不读取 calibration。

- [train/validation 清单 v4 构建与等价审计](train_validation清单v4构建与等价审计.md)：记录内容寻址 release、CPU 清单任务、一次性审计命令与结果，以及后续 Dataset、固定权重前向和科学评估等价门禁。
- [学习率复现](学习率复现/README.md)：记录三个随机种子的 `LR=1e-4/2e-4` 六项配对、受管 Job、launch、W&B 和运行根。
- [摘要拼接](摘要拼接/README.md)：记录 mean/max 与候选条件注意力摘要拼接的六项门值/裁剪矩阵、受管 Job、launch、W&B、运行根、首步和最终结果。
