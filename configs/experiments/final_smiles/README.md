# final_smiles：单一 SMILES 身份训练配置

第二轮活动配置是 `strongest.yaml`、`blob_only.yaml`、`density_only.yaml` 与 `pocket_only.yaml`，稳定实验标识依次为 `strongest-1`、`blob_only-0`、`density_only-0` 与 `pocket_only-0`。`density_blob_only.yaml` 与 `trunk_only.yaml` 只保留历史模型的可恢复训练入口。

两项活动实验共享完整第 1–6 片 train/validation manifest、seed 3409、AdamW、学习率 `1e-4`、物理 batch 32、梯度累积 1、BF16、裁剪 200 和同一停止规则。`strongest-1` 的候选侧使用完整输入，前景、配体区域、结合区域和身份损失权重依次为 `0.30/0.05/0.05/1.00`；`blob_only-0` 的候选侧只使用 V/blob token、V 三项统计和四项裁剪统计，没有 U-Net 或区域头，对应权重为 `0.30/0/0/1.00`。

H100 上的 `strongest-1` 使用 30 个活动 DataLoader worker，A800 上的 `blob_only-0` 使用 15 个。训练与 validation 的 worker 不并存：一个 epoch 的唯一批次计划先冻结，再连续切成三个区段；每个训练区段完全结束并关闭工作进程后才启动完整 validation。切换阶段不会重新采样、遗漏、重复或改变训练候选顺序。

validation 选模分数固定为 `(0.3 × 前景 PRAUC + 真实前景且身份受支持候选上的 SMILES 匹配准确率) / 1.3`。统一有效改善门槛是 `0.001`；连续前三次未改善只累计计数，第四次触发平台动作。前两次平台把学习率乘以 `0.8`，第三次平台不再降低学习率而直接早停。训练最多 15 个 epoch、45 次完整 validation，并保留每次检查的 checkpoint。

