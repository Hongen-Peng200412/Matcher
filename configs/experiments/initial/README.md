# Matcher 首轮实验配置

本目录的七个 YAML 与已排队的七个单卡 Slurm 任务一一对应。所有业务产物只能写入 `/storage/penghongen/tmp/Matcher/`；训练和模型选择只读取 train 与 validation，不读取 calibration。

| 配置 | GPU | 物理批量 | 梯度累积 | 任务 |
|---|---:|---:|---:|---|
| `candidate_pretrain_pair.yaml` | A100 | 8 | 4 | 先训练 mean/max 六分类，再从头训练 candidate-conditioned cross-attention 六分类 |
| `full_scratch_smi_ted_light_289m_oracle.yaml` | A800 | 16 | 2 | SMI-TED Light 289M，oracle |
| `full_scratch_smi_ted_light_289m_detached.yaml` | A800 | 16 | 2 | SMI-TED Light 289M，detached |
| `full_scratch_smi_ted_light_289m_joint.yaml` | A800 | 16 | 2 | SMI-TED Light 289M，joint |
| `full_scratch_molformer_oracle.yaml` | A100 | 8 | 4 | MoLFormer，oracle |
| `full_scratch_molformer_detached.yaml` | A100 | 8 | 4 | MoLFormer，detached |
| `full_scratch_molformer_joint.yaml` | A100 | 8 | 4 | MoLFormer，joint |

## 字段契约

`data.data_root` 指向只读 AdaLigand 冻结产物，`data.stage1_root` 指向只读 Find_0 候选产物，`data.output_root` 仅保存 Matcher 清单与体素缓存。

`coverage_threshold=0.30` 同时应用于候选覆盖率和真实配体 occurrence 覆盖率。`selection_seed=3407` 只冻结清单内同一 `object_key` 的语言表示选择；各阶段的 `training.seed` 独立冻结训练随机源。`pp_top_k=1024` 是首轮局部 `48³` PP 采样上限，必须在执行记录中作为待实验复核参数。

`num_workers=16` 表示每个训练进程的 DataLoader 工作进程数。`cache_gib_per_worker` 是每个 worker 的独立 LRU 字节上限，A100 任务为 16 GiB，A800 任务为 32 GiB；该上限不预分配内存。

`model.aggregation` 只允许 `mean_max` 或 `candidate_cross_attention`。`raunet_planes` 依次是 `(enc0, enc1, enc2, enc3, bottleneck, dec3, dec2, dec1, dec0)`。`gradient_checkpointing=true` 仅表示训练时激活重计算，不改变前向科学语义。

`physical_batch_size × gradient_accumulation` 必须严格等于 `global_batch_size=32`。`maximum_epochs=null` 表示不设 epoch 上限；任务也不设 Slurm 或程序墙钟上限。`patience_bad_epochs=2` 表示连续两个完整 validation epoch 没有改善后结束训练。

`identity_variant` 仅完整 Matcher 使用：`oracle` 以真实 small-molecule 掩码加权身份损失，`detached` 再乘不回传的 `p_small`，`joint` 乘会回传的 `p_small`。三者都以同一 global batch 的真实有效小分子数作分母。
