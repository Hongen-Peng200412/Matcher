# 联合身份 v4 实验配置

本目录保存把身份键从 `object_key` 提升为 `(object_key, PreparedLigand.smiles)` 后的独立基线。身份逐字采用交给语言模型公开接口的 PreparedLigand 顶层 `smiles`：通常是 canonical non-isomeric SMILES，`diagnostics.used_raw_smiles=true` 时则是规范化失败后的非空原文回退。身份只认该顶层值；不会使用 occurrence 坐标、`present` 掩码、沉积原子选择、`diagnostics` 或语言模型归档中的 `model_smiles`。顶层 `smiles=null` 的 occurrence 没有完整联合身份，不建立 `(object_key, null)` 身份槽。

实际身份清单版本由执行配置的 Matcher 包中 `MANIFEST_SCHEMA_VERSION` 决定，不由 YAML 重复声明；本目录配置必须由采用 schema v4 联合身份的 Matcher release 执行。

六份学习率配置组成三个训练随机种子配对实验。六项 `data.selection_seed` 全部固定为 `3407`，保证联合身份的语言 occurrence 选择和清单契约不变；三个配对的 `training.seed` 分别为 `3407`、`3408` 和 `3409`。每个配对只比较 `training.learning_rate=1.0e-4` 与 `2.0e-4`；计划名、阶段名和 W&B 分组只用于隔离任务，不改变模型输入、监督或优化规则。所有配置都使用物理批量 8、梯度累积 4、global batch 32、梯度裁剪阈值 1、两个零增量身份门、oracle 身份损失和 `initialize_candidate_from: null` 从零训练。global batch 32 表示每次参数更新累计使用 32 个候选。

| 随机种子 | LR=`1.0e-4` 配置 | LR=`2.0e-4` 配置 | W&B group |
|---:|---|---|---|
| `3407` | `joint_identity_v4_zero_gate_oracle_a100.yaml` | `joint_identity_v4_lr_2e4_oracle_a100.yaml` | 既有基线使用 `joint-identity-v4-zero-gate-baseline`；学习率探针使用 `joint-identity-v4-learning-rate` |
| `3408` | `joint_identity_v4_seed_3408_lr_1e4_oracle_a100.yaml` | `joint_identity_v4_seed_3408_lr_2e4_oracle_a100.yaml` | `joint-identity-v4-learning-rate-seed-3408` |
| `3409` | `joint_identity_v4_seed_3409_lr_1e4_oracle_a100.yaml` | `joint_identity_v4_seed_3409_lr_2e4_oracle_a100.yaml` | `joint-identity-v4-learning-rate-seed-3409` |

seed 3407 的 LR=`1.0e-4` 配置保持与[第 2 批零门 oracle 配置](../batch2/full_smi_ted_v_centroid_oracle_a100.yaml)相同的数据参数、表示、模型、损失、随机种子、训练超参数和 A100 批量；计划名、阶段名及 W&B 项目和组用于隔离联合身份 v4 实验。历史 v3 零门结果只作等价迁移的辅助参照，不替代三个 v4 配对内部的正式对照。

清单来源审计、Dataset 与固定权重数值等价、输出目录查重和受管资源触发属于任务执行证据，应记录在联合身份 v4 的单任务、批次和总记录中，不写入本配置目录的稳定科学契约。

六项学习率任务使用 `AdaLigand-Matcher-JointIdentityV4` 项目；下述九项摘要拼接配置和三项配对 mean/max 对照使用独立的 `AdaLigand-Matcher-JointIdentityV4-SummaryConcat` 项目。上表列出六项学习率任务的分组。全部任务都只用 validation 选模；calibration 仍等待全部迭代结束及用户单独确认。

## 摘要拼接结构锚点

`joint_identity_v4_concat_zero_gate_seed_3407_lr_1e4_oracle_a800.yaml` 是最后一组摘要结构实验的首个直接对照。它在 seed 3407、A800、LR=`1.0e-4`、零身份门和 clip 1 下，把候选聚合从 `mean_max` 改为 `mean_max_candidate_cross_attention_concat`；数据、模型宽度、损失、批量、初始化和评估口径保持不变。计划名、阶段名、输出目录名及 W&B project/group 只隔离产物。

该配置先完整计算 mean/max 与候选条件注意力的两个 `D_m=model.model_dim` 维表示，再把二者拼接为 `2D_m` 维残差输入。残差末层为零初始化，初始候选表示逐值等于 mean/max 主路径；训练后允许条件注意力表示通过残差参与任务。两种摘要各自按 A/P/V/PP 顺序读取每模态的 token 数、Stage1 固化概率均值和 Stage1 固化概率最大值，共 12 项手动统计；另读取请求裁剪起点是否在任一轴被边界限制（`requested_start_zyx != start_zyx`）、裁剪中心相对完整 V 质心的 L2 与 L∞ 体素距离、裁剪保留的 V 体素比例，共 4 项采样统计。`data.local_crop_strategy=v_centroid` 先以 `np.rint(mean(V_zyx)+0.5-24)` 的最近偶数规则求裁剪起点，再逐轴限制到 `[0,32]`。残差入口不再第三次追加这些原始统计。该任务只用 validation 选择，不读取 calibration。

## 摘要拼接参数矩阵

此前参数矩阵包含八个摘要拼接任务。四项 A800 任务固定训练随机种子 3407、LR=`1.0e-4` 和相同批量，以双身份门 `{0, 0.02}` 与梯度裁剪阈值 `{1, 200}` 形成完整 2×2 矩阵。四项 A100 任务固定训练随机种子 3408、LR=`1.0e-4` 和相同批量，使用相同的 2×2 设计。A100 与 A800 之间同时改变硬件和训练随机种子，只提供跨硬件、跨种子的方向性重复证据，不能把差异归因于硬件或种子任一单轴，也不冒充逐位同轨对照。

| GPU | 门值 | 裁剪阈值 | 配置 | 直接用途 |
|---|---:|---:|---|---|
| A800 | 0 | 1 | `joint_identity_v4_concat_zero_gate_seed_3407_lr_1e4_oracle_a800.yaml` | 与同卡 v4 mean/max 基线比较摘要结构；也是 A800 裁剪对照基线 |
| A800 | 0.02 | 1 | `joint_identity_v4_concat_gate_002_seed_3407_lr_1e4_oracle_a800.yaml` | 与 A800 零门、clip 1 任务只比较双门初值；复核 A100 门值方向 |
| A800 | 0 | 200 | `joint_identity_v4_concat_zero_gate_clip_200_seed_3407_lr_1e4_oracle_a800.yaml` | 与 A800 零门、clip 1 任务只比较裁剪阈值 |
| A800 | 0.02 | 200 | `joint_identity_v4_concat_gate_002_clip_200_seed_3407_lr_1e4_oracle_a800.yaml` | 补齐 A800 2×2；相对双门、clip 1 只改变裁剪，相对零门、clip 200 只改变双门 |
| A100 | 0 | 1 | `joint_identity_v4_concat_zero_gate_seed_3408_lr_1e4_oracle_a100.yaml` | A100 2×2 矩阵共同基线 |
| A100 | 0.02 | 1 | `joint_identity_v4_concat_gate_002_seed_3408_lr_1e4_oracle_a100.yaml` | clip1 下的双门效应 |
| A100 | 0 | 200 | `joint_identity_v4_concat_zero_gate_clip_200_seed_3408_lr_1e4_oracle_a100.yaml` | 零门下的宽松裁剪效应 |
| A100 | 0.02 | 200 | `joint_identity_v4_concat_gate_002_clip_200_seed_3408_lr_1e4_oracle_a100.yaml` | 双门与宽松裁剪的联合单元，用于估计交互 |

八项都固定 `data.selection_seed=3407`、oracle 身份损失、物理批量 8、梯度累积 4、global batch 32、BF16、从零训练和完整 validation 选模。裁剪阈值 200 只表示宽松裁剪；以 W&B 逐更新字段 `gradient/train_step/preclip_norm` 为裁剪前全局梯度范数，分母为全部 global update。只有范数大于 200 的更新比例不超过 0.1% 时，运行记录才能称其为“近似不裁剪”；同时必须保存总更新数、触发数、触发比例、最大范数，以及按每步 `min(1, 200 / (preclip_norm + 1e-6))` 派生的最小缩放系数。

## 最终五项结构对照

最终只新增五个实验，完成后不再扩展科学矩阵。第一项使用 A800、seed3407、双门 0.02 和 clip200 的摘要拼接配置，补齐上表 A800 2×2。其余四项统一使用零门、clip200、LR=`1.0e-4`、物理批量 8、梯度累积 4、global batch 32 和从零训练；这四项不等待第一项结果再做条件选择。

| 序号 | GPU | 训练种子 | 聚合结构 | 配置 | 对照职责 |
|---:|---|---:|---|---|---|
| 1 | A800 | 3407 | 摘要拼接 | `joint_identity_v4_concat_gate_002_clip_200_seed_3407_lr_1e4_oracle_a800.yaml` | 补齐 A800 门值 × 裁剪矩阵 |
| 2 | A800 | 3407 | mean/max | `joint_identity_v4_mean_max_zero_gate_clip_200_seed_3407_lr_1e4_oracle_a800.yaml` | 与同卡、同 seed、同零门和 clip200 的摘要拼接任务只比较聚合结构 |
| 3 | A100 | 3408 | mean/max | `joint_identity_v4_mean_max_zero_gate_clip_200_seed_3408_lr_1e4_oracle_a100.yaml` | 与同卡、同 seed、同零门和 clip200 的摘要拼接任务只比较聚合结构 |
| 4 | A100 | 3409 | 摘要拼接 | `joint_identity_v4_concat_zero_gate_clip_200_seed_3409_lr_1e4_oracle_a100.yaml` | 新种子结构复现的实验侧 |
| 5 | A100 | 3409 | mean/max | `joint_identity_v4_mean_max_zero_gate_clip_200_seed_3409_lr_1e4_oracle_a100.yaml` | 与序号 4 只比较聚合结构 |

三组 mean/max—摘要拼接直接对照分别固定硬件、训练种子、联合身份 v4 清单、数据选择、模型宽度、两项身份门、损失、优化器、裁剪阈值、批量和评估口径。两种聚合结构只通过 `model.aggregation` 区分；计划名、阶段名和 W&B 分组只隔离运行产物。A100 与 A800 结果仍只作跨硬件、跨种子的方向性证据，不能冒充同轨单变量对照。

## 正式 YAML 解析与完整字段依赖

本目录每份 YAML 都由 `matcher.training.config.load_experiment_plan` 读取，再由 `python -m matcher.training.experiment --plan <YAML路径>` 按 `stages` 列表顺序执行。解析器要求 YAML 顶层映射、`tracking`、`data`、每个 `stage`、`model`、`training` 和 `loss_weights` 的键集合精确匹配：没有缺省字段，未知键和缺失键都会在创建模型或访问数据前失败。YAML 注释不进入 `ExperimentPlan`、`resolved.json`、checkpoint 或 W&B config。

成功启动时，运行根为 `output_root/plan_name/TASK_RUN_STAMP`；`TASK_RUN_STAMP` 缺失时使用当前 UTC 秒级时间。每个阶段写入该运行根下的 `stage.name/`。同名运行根采用 `exist_ok=false`，不会被复用覆盖。

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
| `stages[*].model.aggregation` | `mean_max`、`candidate_cross_attention` 或 `mean_max_candidate_cross_attention_concat` | 依次选择四模态掩码均值/最大值、候选体素摘要查询四模态 token，或拼接前两种完整摘要后用零输出残差加回 mean/max 主表示 |
| `stages[*].model.use_manual_features` | `bool` | 控制候选表示拼接 A/P/V/PP 的 12 项固化统计，并控制三个身份打分器拼接同组统计与停止梯度的身份类别概率 |
| `stages[*].model.use_sampling_features` | `bool` | 控制候选表示和三个身份打分器是否拼接边界限制、V 质心 L2/L∞ 偏移与 V 保留比例，共 4 项裁剪统计 |
| `stages[*].model.use_candidate_token_attention_bias` | `bool` | 为真时聚合必须为 `candidate_cross_attention`；按 `relative_x/y/z/probability` 生成逐头注意力偏置 |
| `stages[*].model.initial_graph_gate` | 非 `bool` 的有限数值，范围 `[0,1]` | 图身份分数增量的可学习标量初值；候选预训练必须为 0，训练后参数不夹紧 |
| `stages[*].model.initial_inventory_gate` | 非 `bool` 的有限数值，范围 `[0,1]` | 同 PDB 自排除身份清单增量的可学习标量初值；候选预训练必须为 0，训练后参数不夹紧 |
| `stages[*].model.model_dim` | 正 `int` | 候选、语言与图身份的共同表示宽度，必须被 `attention_heads` 整除 |
| `stages[*].model.attention_heads` | 正 `int` | 候选条件、配体图和身份清单注意力头数；不控制 RAUNet 瓶颈自选头数 |
| `stages[*].model.dropout` | 非 `bool` 的有限数值，范围 `[0,1)` | token 投影、三种候选聚合路径、候选分类头、图消息、配体图/身份清单注意力和三个身份配对残差网络共用的丢弃概率 |
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

文件名中的 `a100` 或 `a800` 不是配置字段，`load_experiment_plan` 不选择 GPU；实际设备由 launch、Slurm allocation 和 CUDA 可见设备决定。YAML 也不保存 Job、release、W&B run ID、manifest schema 版本或 calibration 许可。`MANIFEST_SCHEMA_VERSION` 来自执行该 YAML 的 Matcher 代码，所以只复制 YAML 不能逐字复现另一 release 的身份契约；历史结果必须同时查看对应 release、`resolved.json`、checkpoint 和实验记录。
