# strongest-1 实验日志

本记录保存 `strongest-1` 的单实验追溯。正式模型使用 106 通道体素输入、A/P/V/PP token、末端无多尺度卷积的 RAUNet, 以及初值为 `0.01` 的 A/P/PP 独立门。

## 当前状态

- 阶段: 本地实现已完成，等待冻结审查与 H100 实际门禁。
- 正式资源: H100 Job `378587`, 门禁通过前不启动训练。
- 正式实验目录: `测试结果与追溯/experiment-centric/strongest-1/`。
- W&B、release、launch、运行根、BEST checkpoint 与两套评测产物: 待门禁通过后填写。
- 正式 YAML: `configs/experiments/final_smiles/strongest.yaml`；实验标识 `strongest-1`，30 个阶段互斥 worker，损失权重 `0.30/0.05/0.05/1.00`。

## 正式命令

尚未发布。本节不记录本地测试、GPU 门禁或只读核查命令。

## 失败或中断尝试

暂无。后续若有尝试中断, 将追加在本文档后半部, 不新建并行实验日志。
