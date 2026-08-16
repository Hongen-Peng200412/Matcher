# Matcher 首轮实验配置

本目录保存第 0 批七项单卡实验的稳定配置。所有业务产物只能写入 `/storage/penghongen/tmp/Matcher/`；训练和模型选择只读取 train 与 validation，不读取 calibration。实际 Job、release、W&B 和结果以 `文档/记录/` 的实验账本为准。

每个训练阶段都在 `pencounkdual-111/AdaLigand-Matcher` 建立独立 W&B online run，并归入 `matcher-initial-round1` 比较组。Matcher 入口不显式接收、传入或落盘 API key；凭据解析由已安装的 W&B SDK 负责，不能仅据本目录断言具体凭据来源。`log_every_global_steps=1` 表示每个含 32 个候选的 global batch 完成一次参数更新后记录损失、梯度、学习率、吞吐和显存；完整 validation、选模分数、统计分母及身份/区域指标按 epoch 记录。在线初始化失败会终止对应 launch，不会静默产生离线 run。

| 配置 | GPU | 物理批量 | 梯度累积 | 任务 |
|---|---:|---:|---:|---|
| `candidate_pretrain_pair.yaml` | A100 | 8 | 4 | 依次用 mean/max 与 candidate-conditioned cross-attention 聚合，以六分类和两个区域损失分别预训练整套候选编码器 |
| `full_scratch_smi_ted_light_289m_oracle.yaml` | A800 | 16 | 2 | SMI-TED Light 289M，oracle |
| `full_scratch_smi_ted_light_289m_detached.yaml` | A800 | 16 | 2 | SMI-TED Light 289M，detached |
| `full_scratch_smi_ted_light_289m_joint.yaml` | A800 | 16 | 2 | SMI-TED Light 289M，joint |
| `full_scratch_molformer_oracle.yaml` | A100 | 8 | 4 | MoLFormer，oracle |
| `full_scratch_molformer_detached.yaml` | A100 | 8 | 4 | MoLFormer，detached |
| `full_scratch_molformer_joint.yaml` | A100 | 8 | 4 | MoLFormer，joint |

## 字段契约

`candidate_pretrain_pair.yaml` 用 YAML 锚点 `&a100_training` 定义第一阶段的完整训练映射，再由 `*a100_training` 给第二阶段复用同一组解析值。别名只复用配置数据，不传递参数、优化器或 checkpoint；两个阶段的 `initialize_candidate_from` 都是 `null`，因此第二阶段仍从独立随机初始化开始。

`data.data_root` 指向只读 AdaLigand 既有产物，`data.stage1_root` 指向只读 Find_0 候选产物，`data.output_root` 仅保存 Matcher 清单、体素缓存与 PP token 缓存。

`coverage_threshold=0.30` 同时应用于候选覆盖率和真实配体 occurrence 覆盖率。`selection_seed=3407` 固定清单内同一身份的 occurrence 语言表示选择；身份字段由执行该 YAML 的 Matcher schema 决定，当前 v4 使用 `(object_key, PreparedLigand.smiles)`，原始第 0 批 release 则必须从对应运行记录核实。各阶段的 `training.seed` 分别设定训练随机源。`local_crop_strategy=fixed_center` 明确复现第 0 批 `[16,64)³` 视野；当前兼容入口把它写入 `cache/voxel_input_v2/fixed_center/` 和 `cache/pp_input_v2/fixed_center/top_1024/`，不会误读第 2 批 V 质心缓存。`pp_top_k=1024` 是每个局部 `48³` 的 PP 采样上限。PP 缓存按 PDB 一次读取完整 probability map，再为该 PDB 的全部候选保存 token；缓存命中后，训练工作进程只读取该小型产物，缺失时由第一个持锁进程按 PDB 构建一次。

`train_num_workers=4` 与 `validation_num_workers=4` 分别指定 train 和 validation 单次完整迭代使用的 DataLoader 工作进程数。两套工作进程不会同时常驻：每次迭代开始时建立对应进程池，完整耗尽该 DataLoader 后销毁，再进入下一次 train 或 validation 迭代。train 和 validation 分别持有独立的 Torch 随机生成器，初始种子依次为 `2 × training.seed` 和 `2 × training.seed + 1`；配置解析器把 `training.seed` 限制在 NumPy 接受的 `[0,2³²-1]`，所以这两个公式无需折返。重建 worker 池不会推进模型与 dropout 使用的全局 Torch 随机流。4 是当前首轮配置的低内存启动值，后续 YAML 可以显式调整。`cache_gib_per_worker` 是每个 `CandidateDataset` 实例五类 LRU 的总 GiB 预算；本目录的正 worker 数配置使每个 DataLoader worker 副本各自持有一个 Dataset 实例。A100 配置为 16 GiB，A800 配置为 32 GiB；缓存按需尝试保留 NumPy 载荷，不预分配内存，单项超过对应分账预算时不保留，也不代表进程 RSS 上限。

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

`model.aggregation` 允许 `mean_max`、`candidate_cross_attention` 或 `mean_max_candidate_cross_attention_concat`。第三项分别计算前两种完整摘要，把两个候选表示拼接后通过零输出残差加回 mean/max 主表示。`raunet_planes` 依次是 `(enc0, enc1, enc2, enc3, bottleneck, dec3, dec2, dec1, dec0)`。`gradient_checkpointing=true` 仅表示训练时激活重计算，不改变前向科学语义。

`identity_scope=null` 表示候选预训练没有身份损失；`identity_scope=small_molecule` 表示完整 Matcher 只在真实小分子且身份目标有效的候选上计算身份损失。`model.use_manual_features=false`、`model.use_sampling_features=false` 和 `model.use_candidate_token_attention_bias=false` 固定第 0 批结构，不加入第 1 批手动统计、第 2 批采样统计或第 3 批候选—实体注意力偏置。`initial_graph_gate=0.0` 与 `initial_inventory_gate=0.0` 保持图和清单身份分数增量的原有严格零初始化；候选预训练阶段虽不构造身份分支，也必须显式写零。第 0 批 checkpoint 缺少三个开关时均迁移为 `false`，缺少两个门初值时迁移为 `0.0`；第 1/2 批 checkpoint 保留已经存在的开关，只迁移实际缺失字段。

`physical_batch_size × gradient_accumulation` 必须严格等于 `global_batch_size=32`。`maximum_epochs=null` 表示不设 epoch 上限；任务也不设 Slurm 或程序墙钟上限。`patience_bad_epochs=2` 表示连续两个完整 validation epoch 没有改善后结束训练。

`identity_variant` 仅完整 Matcher 使用。令 `c(L)` 为真实身份 `L` 的六分类类别，`p_G=softmax(class_logits)[c(L)]`：`oracle` 的身份系数为 1，`detached` 使用不回传的 `p_G`，`joint` 使用会向六分类头回传的 `p_G`。本目录配置均为 `identity_scope=small_molecule`，所以这里的 `c(L)` 恒为 `small_molecule`，与第 0 批的 `p_small` 完全相同；三者都以同一 global batch 的真实有效小分子数作分母。

## 正式 YAML 解析与完整字段依赖

本目录每份 YAML 都由 `matcher.training.config.load_experiment_plan` 读取，再由
`python -m matcher.training.experiment --plan <YAML路径>` 按 `stages` 列表顺序执行。解析器要求
YAML 顶层映射、`tracking`、`data`、每个 `stage`、`model`、`training` 和 `loss_weights` 的键集合
精确匹配：没有缺省字段，未知键和缺失键都会在创建模型或访问数据前失败。YAML 注释不进入
`ExperimentPlan`、`resolved.json`、checkpoint 或 W&B config。

成功启动时，运行根为
`output_root/plan_name/TASK_RUN_STAMP`；`TASK_RUN_STAMP` 缺失时使用当前 UTC 秒级时间。
每个阶段写入该运行根下的 `stage.name/`。同名运行根采用 `exist_ok=false`，不会被复用覆盖。

### 顶层与在线追踪字段

| 字段路径 | 类型与特殊值 | 具体读取与影响 |
| --- | --- | --- |
| `plan_name` | 安全单级 `str` | 组成运行根第二层；以 ASCII 字母或数字开头，后续只允许字母、数字、下划线、点和连字符 |
| `output_root` | POSIX 绝对路径字符串 | 运行与阶段产物根；词法规范化后必须位于 `/storage/penghongen/tmp/Matcher`，不解析符号链接或检查存在性 |
| `tracking` | 精确七字段映射 | 全部阶段共用同一组在线追踪设置；每个阶段仍分别建立 W&B run |
| `tracking.provider` | `str`，固定 `wandb` | `start_wandb_tracking` 选择在线追踪实现 |
| `tracking.mode` | `str`，固定 `online` | 禁止由正式入口静默降级成离线 run |
| `tracking.entity` | 非空 `str` | W&B 账户或团队；本地解析不验证远端权限 |
| `tracking.project` | 非空 `str` | W&B 项目；每个阶段分别建立 run |
| `tracking.group` | 非空 `str` | W&B 曲线比较分组；不进入模型输入、损失或选模公式 |
| `tracking.log_every_global_steps` | 正 `int` | 参数更新曲线间隔；`global_step=1` 仍固定记录 |
| `tracking.init_timeout_seconds` | 正 `int`，单位 s | 传给 W&B `Settings.init_timeout`；不是训练、数据加载或 Slurm 时限 |

### 共享数据字段

| 字段路径 | 类型与特殊值 | 具体读取与影响 |
| --- | --- | --- |
| `data` | 精确十一字段映射 | 全部阶段共用来源根、清单选择、局部裁剪、DataLoader 与 Dataset 缓存设置 |
| `data.data_root` | 非空路径字符串 | AdaLigand 来源根；解析器不检查绝对性、存在性或内容，Dataset 与各 Store 在使用时读取 |
| `data.stage1_root` | 非空路径字符串 | Stage1 Find_0 来源根；解析器同样不访问文件系统 |
| `data.output_root` | 路径字符串 | manifest、105 通道候选体素与 PP token 缓存根；词法规范化后必须位于 Matcher 产物根 |
| `data.coverage_threshold` | 有限 `float`，范围 `(0,1]` | 候选和 occurrence 双向覆盖率共同使用的 manifest 命中阈值 |
| `data.selection_seed` | 精确 `int` | 固定同一身份采用哪个有效 occurrence 语言表示；不定义身份字段，解析器也不另限其范围 |
| `data.pp_top_k` | 正 `int` | 每个候选最多保留的 PP token 数 `K` |
| `data.local_crop_strategy` | `fixed_center` 或 `v_centroid` | 决定 80³ 来源到 48³ 模型输入的裁剪起点，并进入体素/PP 缓存目录 |
| `data.train_num_workers` | 正 `int` | 每次完整 train DataLoader 迭代建立的 worker 数 |
| `data.validation_num_workers` | 正 `int` | 每次完整 validation DataLoader 迭代建立的 worker 数 |
| `data.prefetch_factor` | 正 `int` | 每个 worker 预取的物理批次数 |
| `data.cache_gib_per_worker` | 正有限 `float`，单位 GiB | 转成每个 `CandidateDataset` 实例五类 LRU 的总预算；DataLoader 场景通常每个 worker 副本各有实例，不是 RSS 上限且不预分配 |

### 阶段与模型字段

| 字段路径 | 类型与特殊值 | 具体读取与影响 |
| --- | --- | --- |
| `stages` | 非空列表 | 顺序就是 `run_plan` 的执行顺序；`stage.name` 必须互不重复 |
| `stages[*]` | 精确八字段映射 | 一个阶段同时绑定任务、语言/身份口径、模型、训练参数和可选候选初始化来源 |
| `stages[*].name` | 安全单级 `str` | 阶段目录名、W&B run 名和阶段记录标识 |
| `stages[*].task_name` | `candidate_pretrain` 或 `full_matcher` | 前者以六分类与两个区域损失优化整套 `CandidateEncoder`，不构造身份分支或身份损失；后者还构造闭集身份路径，身份键由执行代码的 schema 决定 |
| `stages[*].language_family` | `molformer`、`smi_ted_289m` 或 `null` | `full_matcher` 必须选择语言家族；`candidate_pretrain` 必须为 `null` |
| `stages[*].identity_variant` | `oracle`、`detached`、`joint` 或 `null` | 三种值分别不用类别概率、使用停止梯度的真实类别概率、使用保留梯度的真实类别概率缩放身份损失；候选预训练必须为 `null` |
| `stages[*].identity_scope` | `small_molecule`、`all_non_background` 或 `null` | 限制身份轴与身份损失有效候选；候选预训练必须为 `null` |
| `stages[*].initialize_candidate_from` | POSIX 绝对 `BEST.pt` 路径或 `null` | 非空时只允许 `full_matcher`；配置解析只核对文字形式，初始化模块随后核验 checkpoint 与来源记录 |
| `stages[*].model` | 精确十五字段映射 | 完整定义候选编码器与身份匹配器结构；正式 YAML 不依赖历史 checkpoint 的缺省迁移字段 |
| `stages[*].model.aggregation` | `mean_max` 或 `candidate_cross_attention` | 选择四模态掩码均值/最大值，或候选体素摘要查询四模态 token 的注意力聚合 |
| `stages[*].model.use_manual_features` | `bool` | 控制候选表示拼接 A/P/V/PP 的 12 项固化统计，并控制三个身份打分器拼接同组统计与停止梯度的身份类别概率 |
| `stages[*].model.use_sampling_features` | `bool` | 控制候选表示和三个身份打分器是否拼接边界限制、V 质心 L2/L∞ 偏移与 V 保留比例，共 4 项裁剪统计 |
| `stages[*].model.use_candidate_token_attention_bias` | `bool` | 为真时聚合必须为 `candidate_cross_attention`；按 `relative_x/y/z/probability` 生成逐头注意力偏置 |
| `stages[*].model.initial_graph_gate` | 非 `bool` 的有限数值，范围 `[0,1]` | 图身份分数增量的可学习标量初值；候选预训练必须为 0，训练后参数不夹紧 |
| `stages[*].model.initial_inventory_gate` | 非 `bool` 的有限数值，范围 `[0,1]` | 同 PDB 自排除身份清单增量的可学习标量初值；候选预训练必须为 0，训练后参数不夹紧 |
| `stages[*].model.model_dim` | 正 `int` | 候选、语言与图身份的共同表示宽度，必须被 `attention_heads` 整除 |
| `stages[*].model.attention_heads` | 正 `int` | 候选条件、配体图和身份清单注意力头数；不控制 RAUNet 瓶颈自选头数 |
| `stages[*].model.dropout` | 非 `bool` 的有限数值，范围 `[0,1)` | token 投影、候选条件注意力、两种候选融合 MLP、候选分类头、图消息、配体图/身份清单注意力和三个身份配对残差网络共用的丢弃概率 |
| `stages[*].model.voxel_feature_channels` | 正 `int` | RAUNet 输出宽度，也是两个 `VoxelBinaryHead` 的输入与隐藏宽度；每个头最终为每个体素输出一个 logit |
| `stages[*].model.raunet_planes` | 九项正 `int` 列表 | 依次为 `enc0/enc1/enc2/enc3/bottleneck/dec3/dec2/dec1/dec0`；骨干还要求 `enc3=bottleneck=dec3`、`dec0=enc0` |
| `stages[*].model.recycle_iterations` | 正 `int` | RAUNet 回收轮数；前置轮使用内部 `no_grad`，最后一轮是否建图服从外层梯度模式与 `requires_grad` |
| `stages[*].model.gradient_checkpointing` | `bool` | 训练且外层启用梯度时，是否对 RAUNet 高开销模块采用非重入激活重计算 |
| `stages[*].model.bottleneck_attention_layers` | 正 `int` | RAUNet 最深网格的三轴旋转位置编码自注意力层数 |
| `stages[*].model.graph_layers` | 正 `int` | `LigandGraphEncoder` 沿双向化学键执行的消息传递层数 |

### 训练与损失字段

| 字段路径 | 类型与特殊值 | 具体读取与影响 |
| --- | --- | --- |
| `stages[*].training` | 精确十二字段映射 | 定义随机源、物理/全局批量、AdamW、轮数/早停、数值精度、梯度裁剪和损失 |
| `stages[*].training.seed` | 精确 `int`，范围 `[0,2**32-1]` | 固定模型、进程、候选洗牌和 DataLoader 派生随机源；不同于 `data.selection_seed` |
| `stages[*].training.physical_batch_size` | 正 `int` | 训练前向的物理批量；validation 前向以此为上限，末批可不足 |
| `stages[*].training.gradient_accumulation` | 正 `int` | 一次 AdamW 更新累计的物理批次数 |
| `stages[*].training.global_batch_size` | 固定 `int 32` | 必须等于 `physical_batch_size × gradient_accumulation` |
| `stages[*].training.learning_rate` | 正有限 `float` | AdamW 初始学习率；后续 `ReduceLROnPlateau` 监控 validation 选择分数并可下调 |
| `stages[*].training.weight_decay` | 非负有限 `float` | AdamW 解耦权重衰减系数 |
| `stages[*].training.maximum_epochs` | 正 `int` 或 `null` | `null` 不设 epoch 上限；patience 早停仍生效 |
| `stages[*].training.patience_bad_epochs` | 固定 `int 2` | 连续两个完整 validation 没有严格改善后停止 |
| `stages[*].training.precision` | `bf16`、`fp16` 或 `fp32` | 选择 CUDA autocast；只有 `fp16` 启用 `GradScaler` |
| `stages[*].training.gradient_clip_norm` | 正有限 `float` | 每个 global batch 在 unscale 后应用的全模型 L2 梯度裁剪阈值 |
| `stages[*].training.class_weights` | 六项非负有限且总和为正的列表或 `null` | 顺序为 `background/ion/nucleotide_like/peptide_like/small_molecule/sugar`；`null` 表示六分类不重加权 |
| `stages[*].training.loss_weights` | 精确四字段映射 | 四个无量纲非负有限系数按字段名组成总损失 |
| `stages[*].training.loss_weights.classification` | 非负有限 `float` | 六分类交叉熵的无量纲系数 |
| `stages[*].training.loss_weights.ligand_area` | 非负有限 `float` | 配体区域平衡 BCE 与软 Dice 之和的系数 |
| `stages[*].training.loss_weights.binding_area` | 非负有限 `float` | 受体结合区域平衡 BCE 与软 Dice 之和的系数 |
| `stages[*].training.loss_weights.identity` | 非负有限 `float` | 闭集身份交叉熵系数；身份键由执行代码的 schema 决定，`candidate_pretrain` 必须为 0 |

### YAML 没有固定的外部事实

文件名中的 `a100` 或 `a800` 不是配置字段，`load_experiment_plan` 不选择 GPU；实际设备由 launch、
Slurm allocation 和 CUDA 可见设备决定。YAML 也不保存 Job、release、W&B run ID、manifest
schema 版本或 calibration 许可。`MANIFEST_SCHEMA_VERSION` 来自执行该 YAML 的 Matcher 代码，
所以只复制 YAML 不能逐字复现另一 release 的身份契约；历史结果必须同时查看对应 release、
`resolved.json`、checkpoint 和实验记录。
