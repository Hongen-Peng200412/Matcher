# Matcher 首轮实验配置

本目录保存第 0 批七项单卡实验的稳定配置。所有业务产物只能写入 `/storage/penghongen/tmp/Matcher/`；训练和模型选择只读取 train 与 validation，不读取 calibration。实际 Job、release、W&B 和结果以 `文档/记录/` 的实验账本为准。

每个训练阶段都在 `pencounkdual-111/AdaLigand-Matcher` 建立独立 W&B online run，并归入 `matcher-initial-round1` 比较组。认证读取服务器进程环境或现有 `~/.netrc`，配置和项目产物都不保存 API key。`log_every_global_steps=1` 表示每个含 32 个候选的 global batch 完成一次参数更新后记录损失、梯度、学习率、吞吐和显存；完整 validation、选模分数、统计分母及身份/区域指标按 epoch 记录。在线初始化失败会终止对应 launch，不会静默产生离线 run。

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

`data.data_root` 指向只读 AdaLigand 冻结产物，`data.stage1_root` 指向只读 Find_0 候选产物，`data.output_root` 仅保存 Matcher 清单、体素缓存与 PP token 缓存。

`coverage_threshold=0.30` 同时应用于候选覆盖率和真实配体 occurrence 覆盖率。`selection_seed=3407` 只冻结清单内同一 `object_key` 的语言表示选择；各阶段的 `training.seed` 独立冻结训练随机源。`local_crop_strategy=fixed_center` 明确复现第 0 批 `[16,64)³` 视野；当前兼容入口把它写入 `cache/voxel_input_v2/fixed_center/` 和 `cache/pp_input_v2/fixed_center/top_1024/`，不会误读第 2 批 V 质心缓存。`pp_top_k=1024` 是每个局部 `48³` 的 PP 采样上限。PP 缓存按 PDB 一次读取完整 probability map，再为该 PDB 的全部候选保存 token；缓存命中后，训练工作进程只读取该小型产物，缺失时由第一个持锁进程按 PDB 构建一次。

`train_num_workers=4` 与 `validation_num_workers=4` 分别冻结 train 和 validation 单次完整迭代使用的 DataLoader 工作进程数。两套工作进程不会同时常驻：每次迭代开始时建立对应进程池，完整耗尽该 DataLoader 后销毁，再进入下一次 train 或 validation 迭代。train 和 validation 分别持有独立的 Torch 随机生成器，初始种子依次为 `2 × training.seed` 和 `2 × training.seed + 1`；配置解析器把 `training.seed` 限制在 NumPy 接受的 `[0,2³²-1]`，所以这两个公式无需折返。重建 worker 池不会推进模型与 dropout 使用的全局 Torch 随机流。4 是当前首轮配置的低内存启动值，后续 YAML 可以显式调整。`cache_gib_per_worker` 是每个当前存活 worker 的独立 LRU 字节上限，A100 任务为 16 GiB，A800 任务为 32 GiB；该上限按需增长而不预分配内存。

`tracking.provider=wandb` 指定在线曲线服务实现，`tracking.mode=online` 要求真实联网且禁止静默降级。`tracking.entity` 是拥有项目的 W&B 账户或团队，`tracking.project` 是曲线所在项目，`tracking.group` 是七个首轮任务共享的比较组。`tracking.log_every_global_steps` 是完成多少次 global-batch 参数更新后写一次逐步曲线；`tracking.init_timeout_seconds=120` 只限制 W&B 在线握手等待，不是训练或 Slurm 时限。

每个 stage 的 `tracking.json` 含下列十个字段，且不保存认证密钥：

- `provider`：在线追踪实现，本轮为 `wandb`。
- `mode`：W&B 运行模式，本轮为 `online`。
- `entity`：拥有在线项目的 W&B 账户或团队。
- `project`：收纳训练曲线的 W&B 项目。
- `group`：用于比较七个首轮任务的 W&B 分组。
- `run_id`：W&B 分配的不可变在线运行标识。
- `run_name`：由训练阶段名与计划运行目录名组成的显示名称。
- `run_url`：可直接打开的 HTTPS 曲线地址。
- `wandb_version`：服务器实际导入的 W&B SDK 版本。
- `local_directory`：阶段目录内由 W&B 管理的本地同步目录。

阶段 `summary.json` 同时保存 `tracking_run_id` 与 `tracking_url`，便于不读取 W&B 本地目录也能定位在线曲线。

`model.aggregation` 只允许 `mean_max` 或 `candidate_cross_attention`。`raunet_planes` 依次是 `(enc0, enc1, enc2, enc3, bottleneck, dec3, dec2, dec1, dec0)`。`gradient_checkpointing=true` 仅表示训练时激活重计算，不改变前向科学语义。

`identity_scope=null` 表示候选预训练没有身份损失；`identity_scope=small_molecule` 表示完整 Matcher 只在真实小分子且身份目标有效的候选上计算身份损失。`model.use_manual_features=false`、`model.use_sampling_features=false` 和 `model.use_candidate_token_attention_bias=false` 固定第 0 批结构，不加入第 1 批手动统计、第 2 批采样统计或第 3 批候选—实体注意力偏置。第 0 批 checkpoint 缺少三个开关时均迁移为 `false`；第 1/2 批 checkpoint 分别保留已经存在的手动/采样开关，仅把缺失的注意力偏置开关迁移为 `false`。

`physical_batch_size × gradient_accumulation` 必须严格等于 `global_batch_size=32`。`maximum_epochs=null` 表示不设 epoch 上限；任务也不设 Slurm 或程序墙钟上限。`patience_bad_epochs=2` 表示连续两个完整 validation epoch 没有改善后结束训练。

`identity_variant` 仅完整 Matcher 使用。令 `c(L)` 为真实身份 `L` 的六分类类别，`p_G=softmax(class_logits)[c(L)]`：`oracle` 的身份系数为 1，`detached` 使用不回传的 `p_G`，`joint` 使用会向六分类头回传的 `p_G`。本目录配置均为 `identity_scope=small_molecule`，所以这里的 `c(L)` 恒为 `small_molecule`，与第 0 批的 `p_small` 完全相同；三者都以同一 global batch 的真实有效小分子数作分母。
