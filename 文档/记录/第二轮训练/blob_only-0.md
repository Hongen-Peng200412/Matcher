# blob_only-0 实验日志

本记录保存 `blob_only-0` 的单实验追溯。正式模型的候选侧只使用 V/blob token、V 的 count/mean/max 和四项裁剪统计; 身份侧仍使用 SMI-TED、SMILES 图和同 PDB 身份清单。

## 当前状态

- 阶段: 本地实现已完成，等待冻结审查与 A800 实际门禁。
- 正式资源: A800 数组任务 `379403_1`, 锁与反馈目录使用基础编号 `379403`; 门禁通过前不启动训练。
- 正式实验目录: `测试结果与追溯/experiment-centric/blob_only-0/`。
- W&B、release、launch、运行根、BEST checkpoint 与两套评测产物: 待门禁通过后填写。
- 正式 YAML: `configs/experiments/final_smiles/blob_only.yaml`；实验标识 `blob_only-0`，15 个阶段互斥 worker，损失权重 `0.30/0/0/1.00`。

## 正式命令

尚未发布。本节不记录本地测试、GPU 门禁或只读核查命令。

## 失败或中断尝试

暂无。后续若有尝试中断, 将追加在本文档后半部, 不新建并行实验日志。
