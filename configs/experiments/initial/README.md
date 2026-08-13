# Matcher 首轮实验配置

本目录的七个 YAML 与已排队的七个单卡 Slurm 任务一一对应。所有业务产物只能写入 `/storage/penghongen/tmp/Matcher/`；训练和模型选择只读取 train 与 validation，不读取 calibration。

每个尚未启动的训练阶段都在 `pencounkdual-111/AdaLigand-Matcher` 建立独立 W&B online run，并归入 `matcher-initial-round1` 比较组。认证读取服务器进程环境或现有 `~/.netrc`，配置和项目产物都不保存 API key。`log_every_global_steps=1` 表示每个含 32 个候选的 global batch 完成一次参数更新后记录损失、梯度、学习率、吞吐和显存；完整 validation、选模分数、统计分母及身份/区域指标按 epoch 记录。在线初始化失败会终止对应 launch，不会静默产生离线 run。

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

`coverage_threshold=0.30` 同时应用于候选覆盖率和真实配体 occurrence 覆盖率。`selection_seed=3407` 只冻结清单内同一 `object_key` 的语言表示选择；各阶段的 `training.seed` 独立冻结训练随机源。`pp_top_k=1024` 是首轮局部 `48³` PP 采样上限，也进入 `cache/pp_input_v1/top_1024/` 路径，防止不同数量的 PP token 复用同一缓存。PP 缓存按 PDB 一次读取完整 probability map，再为该 PDB 的全部候选保存 token；缓存命中后，训练 worker 只读取该小型产物，缺失时由第一个持锁进程按 PDB 构建一次。该上限仍须在执行记录中作为待实验复核参数。

`num_workers=16` 表示每个训练进程的 DataLoader 工作进程数。`cache_gib_per_worker` 是每个 worker 的独立 LRU 字节上限，A100 任务为 16 GiB，A800 任务为 32 GiB；该上限不预分配内存。

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

`physical_batch_size × gradient_accumulation` 必须严格等于 `global_batch_size=32`。`maximum_epochs=null` 表示不设 epoch 上限；任务也不设 Slurm 或程序墙钟上限。`patience_bad_epochs=2` 表示连续两个完整 validation epoch 没有改善后结束训练。

`identity_variant` 仅完整 Matcher 使用：`oracle` 以真实 small-molecule 掩码加权身份损失，`detached` 再乘不回传的 `p_small`，`joint` 乘会回传的 `p_small`。三者都以同一 global batch 的真实有效小分子数作分母。
