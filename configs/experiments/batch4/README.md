# 第 4 批候选预训练初始化配置

本目录的三份计划都使用第 2 批 `batch2_candidate_v_centroid_mean_max_a800` 的 `BEST.pt` 初始化完整 Matcher 的 `candidate_encoder`。该来源在 validation 的 BEST epoch 1 得到六分类 macro-F1 `0.5303632887`；第 3 批增强候选条件注意力最终为 `0.5276366341`，相对未增强注意力只提高约 `0.07%`，没有超过 mean/max。来源选择只使用 validation，没有读取 calibration。

三份计划保持第 2 批 mean/max 完整 Matcher 的候选集合、V 质心视野、12 项 Stage1 固化统计、4 项采样统计、SMI-TED 289M、五类非背景身份范围、模型结构、损失权重、学习率与 global batch 32。配置层唯一科学变化是候选编码器由同一个预训练 BEST 初始化；完整 Matcher 的语言、配体图、清单交互与身份打分分支仍按目标随机种子初始化。oracle 与第 2 批对照同为 A100 物理批量 8；detached/joint 改在 A800 以物理批量 16 执行，因此后二者不是同 GPU 类型的逐位同轨对照，硬件和有限精度差异只能作为解释边界。

三份计划的 `initial_graph_gate=0.0` 与 `initial_inventory_gate=0.0` 保持两个身份分数增量分支从严格零开始，因此候选预训练初始化仍是本批唯一科学变化。

- oracle 使用 A100，物理批量 8、梯度累积 4；每个 `CandidateDataset` 实例的五类 LRU 总预算为 16 GiB。
- detached 与 joint 使用 A800，物理批量 16、梯度累积 2；每个 `CandidateDataset` 实例的五类 LRU 总预算为 32 GiB。
- 三项均使用 4 个 train worker 与 4 个 validation worker、BF16、patience 2、W&B online project `AdaLigand-Matcher-Batch4` 和 group `matcher-batch4-pretrained-mean-max`。
- `maximum_epochs=null` 表示不设轮数上限；`class_weights=null` 表示六分类交叉熵不做类别重加权；`initialize_candidate_from` 必须是已审计的绝对 `BEST.pt`。

候选初始化来源的 release、checkpoint 与服务器取证应保存在对应实验执行记录中；正式 YAML 和本目录 README 只保存稳定配置契约，不嵌入一次性运维证据路径。

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
