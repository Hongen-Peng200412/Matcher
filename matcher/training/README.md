# `matcher.training` 正式训练契约

本目录负责把一份正式实验 YAML 解析成不可变配置，在单张 CUDA GPU 上执行候选预训练或完整闭集 Matcher 训练，并发布能够恢复训练状态和追溯科学配置的阶段产物。这里的“阶段”是 YAML `stages` 列表中的一个训练单元；同一计划中的阶段按列表顺序执行，各自建立独立的 Weights & Biases（W&B）在线运行记录。

本目录只构造 `train` 与 `validation`。`calibration` 的清单、特征、指标和输出不由训练入口读取。release、launch、提交命令、文件摘要和服务器锁属于一次性运维或运行治理，不进入 Python 训练主链、正式 YAML 值或 `resolved.json`。

## 模块边界

| 文件 | 主要职责 | 正式调用关系 |
| --- | --- | --- |
| `config.py` | 读取 YAML，拒绝缺失键、未知键、非法枚举、批量关系错误和越界输出路径 | `experiment.run_plan`、训练初始化核验与 calibration 共同调用 `load_experiment_plan` |
| `losses.py` | 计算六分类、配体区域、结合区域和联合身份损失 | 训练与 validation 都调用 `compute_matcher_loss` |
| `numeric.py` | 重置随机源并设置 cuDNN、确定性算法和 float32 矩阵乘精度 | 阶段 runtime 和 calibration 使用同一入口 |
| `checkpointing.py` | 原子保存 checkpoint，核验最小 schema，兼容提取候选编码器状态 | 训练阶段写 `BEST.pt`/`LAST.pt`；初始化与 calibration 读取 |
| `initialization.py` | 交叉核对已完成候选预训练来源，并只加载目标候选编码器 | `_build_stage_runtime` 在模型上 GPU 和构造优化器前调用 |
| `tracking.py` | 建立 W&B online run，规范化指标，写 `tracking.json`，结束后台同步 | `run_training_stage` 在首次 DataLoader 迭代前调用 |
| `engine.py` | 构造 Dataset、模型与优化器，执行梯度累积、完整 validation、选模、早停和阶段产物发布 | `experiment.run_plan` 对每个 stage 顺序调用 |
| `experiment.py` | 创建唯一计划运行目录，写根运行记录并顺序编排阶段 | `python -m matcher.training.experiment --plan <YAML>` |

## 维度与计数

- `B`：一个物理批次的候选数，即一次模型前向的样本数。
- `I`：一个批次补零后的最大联合身份槽数。
- `Z/Y/X`：模型局部体素网格的三个空间轴，正式区域目标为 `48×48×48`。
- `global batch`：一次参数更新使用的全部候选；正式配置固定为 32。
- `global_step`：阶段内从 1 开始累计的参数更新数，不是物理批次数。
- `epoch`：从 0 开始的完整 train 迭代加完整 validation。
- `identity eligible candidate`：真实类别属于 `identity_scope` 且 `identity_target >= 0` 的候选。

## 正式 YAML

`load_experiment_plan` 使用 `yaml.safe_load` 读取 UTF-8 文件。下列每个映射的键集合必须精确相等：缺字段不会使用默认值，额外字段也会失败。

### 计划顶层

| 字段 | 类型与门禁 | 作用 |
| --- | --- | --- |
| `plan_name` | 安全单级字符串；以 ASCII 字母或数字开头，后续只允许字母数字、下划线、点、连字符 | 结果路径 `output_root/plan_name/run_stamp` 的第二层 |
| `output_root` | POSIX 绝对路径；词法规范化后必须位于 `/storage/penghongen/tmp/Matcher` 或其子目录 | 计划、运行和阶段结果根 |
| `data` | 字符串键映射 | 全部阶段共享的数据和 DataLoader 配置 |
| `tracking` | 字符串键映射 | 全部阶段共享、每阶段分别使用的 W&B online 配置 |
| `stages` | 非空列表 | 按列表顺序执行的阶段；`stage.name` 在计划内唯一 |

输出路径门禁只做 POSIX 词法规范化，不访问文件系统，也不解析既有符号链接。`data_root` 与 `stage1_root` 在配置层只转换为 `Path`；它们的存在性和具体文件字段由 Dataset 与 Store 消费时核验。

### `data`

| 字段 | 类型与范围 | 消费位置与语义 |
| --- | --- | --- |
| `data_root` | 非空路径字符串 | AdaLigand 解析、LigandObject、区域与语言来源根 |
| `stage1_root` | 非空路径字符串 | Stage1 Find_0 候选、密度和概率来源根 |
| `output_root` | 允许业务根下的 POSIX 绝对路径 | schema v4 manifest、候选体素与 PP token 缓存根 |
| `coverage_threshold` | 有限浮点数，`(0,1]` | 候选与 occurrence 双向覆盖率共同阈值 |
| `selection_seed` | 精确 Python 整数 | 同一联合身份存在多个有效 occurrence 时固定语言表示选择；本配置层未额外限制范围 |
| `pp_top_k` | 正整数 | 每个候选保留的 PP token 上限 |
| `local_crop_strategy` | `fixed_center` 或 `v_centroid` | 80³ 来源到 48³ 局部输入的裁剪策略 |
| `train_num_workers` | 正整数 | 每次完整 train loader 迭代创建的 worker 数 |
| `validation_num_workers` | 正整数 | 每次完整 validation loader 迭代创建的 worker 数 |
| `prefetch_factor` | 正整数 | 每个 worker 预取的物理批次数 |
| `cache_gib_per_worker` | 正有限浮点数 | 每个 Dataset 实例五类 `ByteBudgetLru` 的总预算；不是进程 RSS 或全部对象内存硬上限 |

GiB 预算在 runtime 构造时用 `int(cache_gib_per_worker * 1024**3)` 向零截断为 byte；极小的正配置因此仍可能得到 0 byte，此时各 LRU 不保留对象。Dataset 再把总预算分给 artifact、identity、PP、source 与 graph 缓存；字符串、Python 容器、清单记录和批次张量不都计入该预算。

### `tracking`

| 字段 | 类型与范围 | 作用 |
| --- | --- | --- |
| `provider` | 固定 `wandb` | 唯一正式追踪实现 |
| `mode` | 固定 `online` | 禁止初始化失败时静默降级 offline |
| `entity` | 非空字符串 | W&B 账户或团队；本地解析不核验远端权限 |
| `project` | 非空字符串 | W&B 项目 |
| `group` | 非空字符串 | 同一科学批次的比较分组 |
| `log_every_global_steps` | 正整数 | 逐参数更新曲线间隔；`global_step=1` 始终写入 |
| `init_timeout_seconds` | 正整数 | `wandb.Settings(init_timeout=...)` 的墙钟秒数 |

### `stages[*]`

| 字段 | `candidate_pretrain` | `full_matcher` |
| --- | --- | --- |
| `name` | 安全单级名称 | 同左 |
| `task_name` | 固定 `candidate_pretrain` | 固定 `full_matcher` |
| `language_family` | 必须 `null` | `molformer` 或 `smi_ted_289m` |
| `identity_variant` | 必须 `null` | `oracle`、`detached` 或 `joint` |
| `identity_scope` | 必须 `null` | `small_molecule` 或 `all_non_background` |
| `model` | 完整 `MatcherModelConfig`；两个身份门初值必须为 0 | 完整 `MatcherModelConfig` |
| `training` | 完整训练配置；`loss_weights.identity` 必须为 0 | 完整训练配置 |
| `initialize_candidate_from` | 必须 `null` | `null`，或另一项已完成候选预训练 `BEST.pt` 的 POSIX 绝对路径 |

初始化路径在 YAML 解析时只检查绝对路径文字和文件名 `BEST.pt`。文件存在性、来源阶段完成状态、科学数据与候选编码器兼容性由 `initialization.py` 在训练启动时核验。

### `stages[*].model`

| 字段 | 类型与关键关系 |
| --- | --- |
| `aggregation` | `mean_max` 或 `candidate_cross_attention` |
| `use_manual_features` | 精确 `bool` |
| `use_sampling_features` | 精确 `bool` |
| `use_candidate_token_attention_bias` | 精确 `bool`；只允许与候选条件注意力聚合共同启用 |
| `initial_graph_gate` | `[0,1]` 内有限数值；只限制初值，训练参数不经过 sigmoid 或夹紧 |
| `initial_inventory_gate` | `[0,1]` 内有限数值；只限制初值 |
| `model_dim` | 正整数；候选、语言、图和清单表示宽度 |
| `attention_heads` | 正整数且整除 `model_dim`；不控制 RAUNet 自选注意力头数 |
| `dropout` | `[0,1)` 内有限数值 |
| `voxel_feature_channels` | 正整数 RAUNet 输出通道数 |
| `raunet_planes` | 九项正整数，顺序为 `enc0,enc1,enc2,enc3,bottleneck,dec3,dec2,dec1,dec0` |
| `recycle_iterations` | 正整数回收轮数 |
| `gradient_checkpointing` | 精确 `bool`；开启后仍只在训练模式、外层梯度启用且至少一个参与张量需要梯度时重计算高开销模块激活 |
| `bottleneck_attention_layers` | 正整数 RAUNet 瓶颈注意力层数 |
| `graph_layers` | 正整数 LigandObject 图消息层数 |

模型各字段的前向形状、注意力和身份门细节由 [`matcher/model/README.md`](../model/README.md) 定义；本页仍列出全部 YAML 字段，以便单独核对训练计划。

### `stages[*].training`

| 字段 | 类型与范围 | 运行语义 |
| --- | --- | --- |
| `seed` | 精确整数，`[0,2**32-1]` | Python、NumPy、Torch、sampler 和 worker 派生种子来源 |
| `physical_batch_size` | 正整数 | 一次模型前向候选数 |
| `gradient_accumulation` | 正整数 | 一次参数更新包含的物理批次数 |
| `global_batch_size` | 固定 32 | 必须等于前两项乘积 |
| `learning_rate` | 正有限数值 | AdamW 初始学习率 |
| `weight_decay` | 非负有限数值 | AdamW 解耦权重衰减 |
| `maximum_epochs` | 正整数或 `null` | 轮数上限；`null` 不取消早停 |
| `patience_bad_epochs` | 固定 2 | 连续完整 validation 未严格改善的停止轮数 |
| `precision` | `bf16`、`fp16` 或 `fp32` | CUDA autocast；只有 `fp16` 启用 GradScaler |
| `gradient_clip_norm` | 正有限数值 | 每个 global batch 在 unscale 后的全模型 L2 梯度裁剪阈值 |
| `class_weights` | `null`，或六项非负有限数值且总和为正 | 顺序与 `background,ion,nucleotide_like,peptide_like,small_molecule,sugar` 对齐 |
| `loss_weights` | 精确四字段映射 | 四项损失系数 |

`loss_weights` 的字段是 `classification`、`ligand_area`、`binding_area`、`identity`；每项都必须为有限非负数。

## 随机源、数值后端与 worker

每个阶段在构造模型前调用 `configure_random_seeds_and_numeric_backend(seed)`，产生并执行以下五字段记录：

| `numeric_backend` 字段 | 固定语义 |
| --- | --- |
| `python_numpy_torch_seed` | 同值重置 Python `random`、NumPy 旧式全局随机状态、Torch CPU 与全部当前可见 CUDA 生成器 |
| `cudnn_benchmark` | `true` |
| `cudnn_deterministic` | `false` |
| `deterministic_algorithms` | `false` |
| `float32_matmul_precision` | `high` |

因此，相同 seed 不等于跨 GPU 型号、Torch/cuDNN 版本的逐位确定性承诺。`cudnn.benchmark=true` 允许按输入形状选择高吞吐算法。

train sampler 在完整候选下标轴上用独立 `random.Random(seed + epoch)` 洗牌，不按 PDB 分组。它先把候选数截短到 `global_batch_size=32` 的整数倍，再切成物理批次，所以不同物理批量但相同 seed、epoch 和共同截短前缀可得到相同展平候选顺序。validation 保持清单顺序、不开洗牌、不丢末批。

两个 DataLoader 使用 `spawn`、`pin_memory=true`、`persistent_workers=false`。train worker 生成器种子是 `2*seed`，validation 是 `2*seed+1`；两个 loader 各持一个独立 `torch.Generator`，创建迭代器只推进自身生成器。每次完整 loader 迭代后 worker 退出，下次 epoch 或 validation 重新创建。

## 四项损失

### 六分类

`class_logits` 形状为 `(B,6)`，`class_target` 为 `int64 (B,)`。正式 engine 把 YAML 的六项 `class_weights` 构造成 CUDA `float32 (6,)`，唯一轴与 `CLASS_NAMES` 对齐；`compute_matcher_loss` 本身不重新验证该张量的长度、dtype、设备、有限性、非负范围或总和。未配置类别权重时，物理批次分类分母是 `B`；配置后，分母是每个真实类别对应权重之和。跨物理批次和跨完整划分的统计先把局部均值乘回真实分母，再用全局分母相除，不能简单平均各物理批次均值。

### 配体区域与结合区域

两个区域分别读取 `(B,48,48,48)` logits 和 bool target。每个候选的 BCE 先分别平均正体素与负体素：存在正体素时两项各乘 0.5；没有正体素时只使用负项。随后加上分子与分母各平滑 1 的软 Dice 损失。最后沿候选轴平均。

### 联合身份

身份候选只在真实 `class_target` 属于 `identity_scope` 且 `identity_target >= 0` 时进入分母。目标槽必须由 `identity_mask` 标为有效，其 `identity_class_index` 必须等于真实六分类类别。

| `identity_variant` | 每候选身份交叉熵乘数 | 六分类概率路径梯度 |
| --- | --- | --- |
| `oracle` | 1 | 不使用 |
| `detached` | 模型对真实类别的 softmax 概率 | `detach`，不向六分类 logits 回传 |
| `joint` | 模型对真实类别的 softmax 概率 | 保留梯度 |

三种变体都除以真实有效身份候选数，不除以乘数之和。身份路径启用但一个物理批次没有有效身份候选时，返回 `identity_logits[eligible].sum() * 0.0`，因而仍连接身份 logits 计算图；`identity_variant=null` 时则用基础损失张量的 `new_zeros(())` 建立同设备、同 dtype 的零值，不读取也不连接 `identity_logits`。整个 epoch 或 validation 没有有效身份候选时，聚合后的身份损失为 0。

### 梯度累积

一个 global batch 的物理批次组必须恰含 32 个候选。每个物理批次反向前分别按以下真实比例缩放：

- 六分类：局部分类分母 / global batch 分类分母；
- 两个区域：局部候选数 / 32；
- 身份：局部有效身份候选数 / global batch 有效身份候选数，后者为零时身份贡献仍为零。

因此，同一候选顺序和同一 global batch 边界下，物理批量与梯度累积的不同配对不会把不等长身份分母错误地平均。

## 优化、裁剪与调度

AdamW 的 YAML 变量只有学习率与权重衰减；其余语义固定并写入 `resolved.json.frozen_optimizer_semantics`：

| 字段 | 值 |
| --- | --- |
| `adam_betas` | `[0.9,0.999]` |
| `adam_epsilon` | `1e-8` |
| `adam_amsgrad` | `false` |
| `adam_maximize` | `false` |
| `adam_foreach` | `true` |
| `adam_capturable` | `false` |
| `adam_differentiable` | `false` |
| `adam_fused` | `false` |
| `scheduler` | `ReduceLROnPlateau` |
| `scheduler_mode` | `max` |
| `scheduler_factor` | `0.5` |
| `scheduler_patience` | `0` |
| `scheduler_threshold` | `0.0` |
| `scheduler_threshold_mode` | `abs` |
| `scheduler_cooldown` | `0` |
| `scheduler_minimum_learning_rate` | `1e-7` |
| `scheduler_epsilon` | `1e-8` |
| `grad_scaler_initial_scale` | `65536.0` |
| `grad_scaler_growth_factor` | `2.0` |
| `grad_scaler_backoff_factor` | `0.5` |
| `grad_scaler_growth_interval` | `2000` |

每次累积组反向后先 `scaler.unscale_(optimizer)`，再调用 `clip_grad_norm_(..., error_if_nonfinite=true)`。该函数返回裁剪前全模型 L2 范数；`TrainingEpochResult.mean_preclip_gradient_norm` 和 `maximum_preclip_gradient_norm` 都基于这个返回值。图与清单 PairScorer 的梯度范数在全局裁剪后、参数更新前读取。当前引擎没有把“是否超过阈值”、缩放系数或最小缩放系数直接保存为结构化字段；只有逐步范数确实按每个 global step 留存时，才能结合明确阈值完整推导这些统计，稀疏 W&B 记录不足以恢复未写出的更新。

`ReduceLROnPlateau.step(selection_score)` 在每轮完整 validation 后调用。`learning_rate_used` 是本轮训练开始前的值，`next_learning_rate` 是调度器处理本轮分数后的值。

## Validation、选模与早停

`evaluate_model` 把模型切到 `eval`，在 `torch.inference_mode()` 和配置 autocast 下遍历完整 loader。损失使用全划分真实分母，`EvaluationAccumulator` 同时累积：

- 六分类 accuracy、支持类兼容 macro-F1、固定六类 macro-F1、逐类指标和 `6×6` 混淆矩阵；
- 配置身份范围的固有、条件、seen/unseen 条件与端到端指标；
- 固定 `small_molecule`、固定 `all_non_background` 和五个前景类别各自的身份报告；
- 配体区域与结合区域在 logit 0 阈值下的微平均 Dice。

零预测、零支持或零区域分母产生的指标在内存中可为 NaN。严格 JSON 写入层转为 `null`；W&B 只在同一指标映射含精确整数 0 支持数时把登记过的 NaN 转为 `None`。

选择分数只使用当前轮完整 validation：

- `candidate_pretrain`：`six_class.macro_f1`；
- `full_matcher`：`(six_class.macro_f1 + finite(conditional_identity.top1) + finite(end_to_end_identity.f1)) / 3`，两项身份值非有限时各按 0 计。

新分数必须严格大于历史最佳值才算改善。改善时更新 `BEST.pt` 并把 `bad_validation_epochs` 置 0；否则加 1。连续两轮不改善后自然早停。达到 `maximum_epochs` 与达到 patience 是两种不同退出原因，`stopped_by_patience` 只表示后者。

完整 Matcher 的 validation 还可累积不参与损失、正式指标或选模的身份贡献诊断。它只纳入身份目标有效且至少有两个真实身份槽的候选，把 `native`、已乘图门增量和已乘清单门增量分别在有效身份轴去均值后计算 L2 范数；相对比值只在语言基线范数大于 `1e-12` 时计算。

## 候选预训练初始化

`initialize_candidate_encoder_from_pretraining` 只接受目标 YAML 明确声明且实际存在的 `BEST.pt`。所有显式核验完成后才调用目标候选编码器的 `load_state_dict(strict=true)`；目标身份图、身份注意力、三个 PairScorer 与两个身份门保持自身初始化。

核验链如下：

1. `BEST.pt` 的最小 checkpoint schema、`task_name=candidate_pretrain`、阶段名、`epoch=best_epoch`、有限 best score、global step、model/training 配置和同轮 `epoch_record`；
2. 同目录 `resolved.json` 的同名、从零候选预训练阶段，model/training 配置与 checkpoint 相等；
3. 来源与目标十二项候选编码器字段逐值相等；身份门初值和 `graph_layers` 不属于候选编码器比较集合；
4. 来源与目标 `data_root`、`stage1_root`、`output_root`、`coverage_threshold`、`selection_seed`、`pp_top_k`、`local_crop_strategy` 七项科学数据字段逐值相等；worker、prefetch 和缓存预算不参与科学相等比较；
5. 阶段 `summary.json` 指向实际 `BEST.pt` 与同目录 `LAST.pt`，并保存相符的最佳轮次、分数、累计步数以及非空 W&B ID/HTTPS URL；该文件先于 `finish(0)` 写入，单独存在不证明成功收口；
6. 根 `run.json` 与根 `summary.json` 的公共运行字段一致，根 summary 含完全相等的阶段 summary、完成时刻、可空 Slurm Job ID 和原训练 YAML 绝对路径；根 summary 只会在各阶段 `run_training_stage` 已于 `finish(0)` 正常返回后发布，因此证明这些本地调用返回，但不核验 W&B 远端 run 状态；
7. 原训练 YAML 的 `plan_name`、tracking、data、同名 stage、模型与训练语义与上述记录一致；顶层计划 `output_root` 不参与该初始化门禁，科学数据使用的是 `data.output_root`；
8. 来源和目标候选编码器 state_dict 的键集合、每个张量 shape 与 dtype 相等。

历史兼容只包括模型配置解析器已定义的缺失字段迁移，以及来源 data 缺失 `local_crop_strategy` 时迁移为 `fixed_center`。初始化记录不保存文件摘要；release/launch 与字节绑定由训练主线之外的一次性运维证据负责。

初始化函数不访问 W&B 服务。它只核验本地产物之间的字段和路径关系；最终接受来源时，根 `summary.json` 的存在与阶段包含关系说明 `finish(0)` 调用曾正常返回，但不证明远端数据已经同步或仍可访问。

`resolved.json.initialization` 非空时包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `checkpoint_path` | 来源 `BEST.pt` 解析后的绝对 POSIX 路径 |
| `source_plan_name` | 来源计划名 |
| `source_stage_name` | 来源候选预训练阶段名 |
| `source_run_stamp` | 与来源运行目录名相等的标识 |
| `source_slurm_job_id` | 来源根 summary 的非空 Slurm Job ID |
| `source_epoch` | 来源最佳 epoch |
| `source_global_step` | 来源 BEST 保存时累计更新数 |
| `source_best_score` | 来源候选预训练 validation macro-F1 |
| `source_tracking_run_id` | 来源阶段 W&B run ID |
| `source_tracking_url` | 来源阶段 W&B HTTPS URL |
| `source_candidate_model_config` | 十二项来源候选编码器配置 |
| `source_scientific_data` | 七项来源科学数据配置 |
| `resolved_path` | 来源阶段 `resolved.json` 绝对路径 |
| `stage_summary_path` | 来源阶段 `summary.json` 绝对路径 |
| `run_record_path` | 来源根 `run.json` 绝对路径 |
| `run_summary_path` | 来源根 `summary.json` 绝对路径 |
| `training_plan_path` | 来源训练 YAML 绝对路径 |
| `missing_keys` | 严格加载成功后的缺失键列表，成功时为空 |
| `unexpected_keys` | 严格加载成功后的额外键列表，成功时为空 |

`source_candidate_model_config` 只保存候选编码器共享结构，不保存两个身份门初值和 `graph_layers`：

| 字段 | 含义 |
| --- | --- |
| `aggregation` | 来源候选摘要聚合方式 |
| `use_manual_features` | 来源 12 维手工特征开关 |
| `use_sampling_features` | 来源 4 维裁剪统计开关 |
| `use_candidate_token_attention_bias` | 来源候选—token 注意力偏置开关 |
| `model_dim` | 来源候选表示宽度 |
| `attention_heads` | 来源候选条件注意力头数 |
| `dropout` | 来源模型丢弃概率 |
| `voxel_feature_channels` | 来源 RAUNet 输出通道数 |
| `raunet_planes` | 来源九项 RAUNet 通道序列 |
| `recycle_iterations` | 来源 RAUNet 回收轮数 |
| `gradient_checkpointing` | 来源激活重计算开关 |
| `bottleneck_attention_layers` | 来源 RAUNet 瓶颈注意力层数 |

`source_scientific_data` 保存来源与目标必须逐值相等的七项科学数据字段：

| 字段 | 含义 |
| --- | --- |
| `data_root` | 来源 AdaLigand 数据根的 POSIX 绝对路径 |
| `stage1_root` | 来源 Stage1 Find_0 根的 POSIX 绝对路径 |
| `output_root` | 来源 Matcher 清单与缓存根的 POSIX 绝对路径 |
| `coverage_threshold` | 来源双向覆盖率阈值 |
| `selection_seed` | 来源 occurrence 语言表示选择种子 |
| `pp_top_k` | 来源每候选 PP token 上限 |
| `local_crop_strategy` | 来源局部裁剪策略 |

## W&B online

W&B 延迟导入，初始化参数固定为 `resume="never"`、`reinit=true`、`anonymous="never"`、`save_code=false`、`force=true`。在当前 W&B 0.17.3 中，`force=true` 交给登录流程，要求登录成功而不是强制保证 run 创建成功。该入口不显式接收、传入或落盘 API key，凭据解析由已安装 W&B SDK 处理。run 名为 `stage.name__plan_directory.name`，标签按固定顺序保存任务名、聚合方式、语言家族、身份变体和身份范围；空字段使用具名占位字符串。

`tracking.json` 有且只有以下十个字段：

| 字段 | 来源 |
| --- | --- |
| `provider` | 配置，固定 `wandb` |
| `mode` | 配置，固定 `online` |
| `entity` | 配置 |
| `project` | 配置 |
| `group` | 配置 |
| `run_id` | W&B 返回的非空字符串 |
| `run_name` | `stage.name__plan_directory.name` |
| `run_url` | W&B 返回、以 `https://` 开头 |
| `wandb_version` | 导入 SDK 的非空版本字符串 |
| `local_directory` | `stage_directory/wandb` 的路径字符串；记录时不另行检查目录存在 |

逐步指标使用 `global_step` 横轴；逐轮训练、完整 validation、身份报告和选模指标使用 `epoch` 横轴。正式 engine 传入整数；`WandbTracker.log_global_step` 本地只执行正值、首步和取模检查，不做 `type(value) is int` 门禁。每次被记录的参数更新使用下列键；图、清单和注意力偏置项只在对应参数存在现有梯度或模型分支适用时出现：

| 参数更新指标键 | 值 |
| --- | --- |
| `loss/train_step/total` | 当前 global batch 四项加权损失 |
| `loss/train_step/classification` | 当前 global batch 六分类损失 |
| `loss/train_step/ligand_area` | 当前 global batch 配体区域损失 |
| `loss/train_step/binding_area` | 当前 global batch 结合区域损失 |
| `loss/train_step/identity` | 当前 global batch 身份损失 |
| `statistics/train_step/identity_eligible_count` | 当前 global batch 有效身份候选数 |
| `statistics/train_step/crop_boundary_fraction` | 裁剪请求受边界限制的候选比例 |
| `statistics/train_step/crop_offset_l2_mean_voxels` | 裁剪中心到 V 质心的 L2 体素距离均值 |
| `statistics/train_step/crop_offset_linf_mean_voxels` | 裁剪中心到 V 质心的 L∞ 体素距离均值 |
| `statistics/train_step/v_retained_fraction_mean` | V token 裁剪保留比例均值 |
| `statistics/train_step/v_retained_fraction_minimum` | V token 裁剪保留比例最小值 |
| `gradient/train_step/preclip_norm` | 裁剪前全模型 L2 梯度范数 |
| `gradient/train_step/graph_branch_norm` | 裁剪后图 PairScorer 梯度范数 |
| `gradient/train_step/inventory_branch_norm` | 裁剪后清单 PairScorer 梯度范数 |
| `gradient/train_step/a_attention_bias_final_weight_postclip_norm` | A 模态注意力偏置末层 weight 的裁剪后梯度范数 |
| `gradient/train_step/p_attention_bias_final_weight_postclip_norm` | P 模态注意力偏置末层 weight 的裁剪后梯度范数 |
| `gradient/train_step/v_attention_bias_final_weight_postclip_norm` | V 模态注意力偏置末层 weight 的裁剪后梯度范数 |
| `gradient/train_step/pp_attention_bias_final_weight_postclip_norm` | PP 模态注意力偏置末层 weight 的裁剪后梯度范数 |
| `attention_bias/train_step/a_final_weight_poststep_norm` | A 模态同一 weight 在 optimizer.step 后的参数范数 |
| `attention_bias/train_step/p_final_weight_poststep_norm` | P 模态同一 weight 在 optimizer.step 后的参数范数 |
| `attention_bias/train_step/v_final_weight_poststep_norm` | V 模态同一 weight 在 optimizer.step 后的参数范数 |
| `attention_bias/train_step/pp_final_weight_poststep_norm` | PP 模态同一 weight 在 optimizer.step 后的参数范数 |
| `gate/train_step/graph` | 更新后的图分数增量门值 |
| `gate/train_step/inventory` | 更新后的清单分数增量门值 |
| `learning_rate/train_step/current` | 本次更新使用的当前 AdamW 学习率 |
| `performance/train_step/data_wait_seconds` | 当前 global batch 等待 DataLoader 的墙钟秒数 |
| `performance/train_step/compute_seconds` | 当前 global batch 搬运、前向、反向、更新与同步秒数 |
| `performance/train_step/samples_per_second` | 当前 global batch 候选数除以等待与计算秒数和 |
| `memory/train_step/peak_cuda_allocated_bytes` | 当前 epoch 截至本步的 PyTorch 最大已分配显存 byte 数 |
| `memory/train_step/peak_cuda_reserved_bytes` | 当前 epoch 截至本步的 PyTorch 最大保留显存 byte 数 |

每个完成 epoch 始终写入下列训练、学习率和选模键；完整 Matcher 另外写门值、分支梯度和身份贡献诊断：

| 逐轮指标键 | 值 |
| --- | --- |
| `loss/train_epoch/total` | 完整训练 epoch 四项损失加权和 |
| `loss/train_epoch/classification` | 完整训练 epoch 六分类损失 |
| `loss/train_epoch/ligand_area` | 完整训练 epoch 配体区域损失 |
| `loss/train_epoch/binding_area` | 完整训练 epoch 结合区域损失 |
| `loss/train_epoch/identity` | 完整训练 epoch 身份损失 |
| `statistics/train_epoch/sample_count` | 完整训练 epoch 候选数 |
| `statistics/train_epoch/identity_eligible_count` | 完整训练 epoch 有效身份候选数 |
| `performance/train_epoch/elapsed_seconds` | 完整训练 epoch 墙钟秒数 |
| `performance/train_epoch/samples_per_second` | 完整训练 epoch 候选吞吐 |
| `performance/train_epoch/data_wait_seconds` | 完整训练 epoch DataLoader 等待秒数和 |
| `performance/train_epoch/compute_seconds` | 完整训练 epoch CUDA 同步计算秒数和 |
| `memory/train_epoch/peak_cuda_allocated_bytes` | 完整训练 epoch 最大已分配显存 byte 数 |
| `memory/train_epoch/peak_cuda_reserved_bytes` | 完整训练 epoch 最大保留显存 byte 数 |
| `gradient/train_epoch/preclip_mean_norm` | 各参数更新裁剪前梯度范数均值 |
| `gradient/train_epoch/preclip_maximum_norm` | 各参数更新裁剪前梯度范数最大值 |
| `gradient/train_epoch/graph_branch_mean_norm` | 各参数更新裁剪后图分支梯度范数均值 |
| `gradient/train_epoch/inventory_branch_mean_norm` | 各参数更新裁剪后清单分支梯度范数均值 |
| `gate/train_epoch/graph` | epoch 末图门值 |
| `gate/train_epoch/inventory` | epoch 末清单门值 |
| `learning_rate/train_epoch/used` | 当前 epoch 训练开始前学习率 |
| `learning_rate/train_epoch/next` | 当前 validation 调度后的下一轮学习率 |
| `selection/score` | 当前 epoch 的 validation 选择分数 |
| `selection/best_score` | 截至当前 epoch 的最佳选择分数 |
| `selection/improved` | 当前分数是否严格改善 |
| `selection/bad_validation_epochs` | 截至当前 epoch 的连续未改善轮数 |

validation 扁平键与下文 `validation.metrics` 字段一一对应：损失写入 `loss/validation/*`，吞吐写入 `performance/validation/*`，支持数写入 `statistics/validation/*`，六分类汇总写入 `classification_summary/*`，逐类指标写入 `classification_per_class/{class_name}/*`，混淆矩阵写入 `classification_confusion/{true_name}/{predicted_name}`，区域写入 `area/{ligand,binding}/*`。启用身份范围时，配置范围写入 `identity_configured/*`，两个固定范围写入 `identity_small_molecule/*` 与 `identity_all_non_background/*`，五个前景类分别写入 `identity_{class_name}/*`；身份贡献诊断使用 `identity_contribution/validation/*`。各身份范围只写 Top-1、Top-K、支持数和端到端 precision、recall、F1 与三个计数，不写报告中的 `top_k` 配置值。

一旦 W&B run 已创建，后续 ID、URL、版本、横轴声明或 `tracking.json` 发布失败都会尝试 `finish(exit_code=1)` 并重新抛出原始错误。阶段训练失败同样尝试失败收口；成功路径在阶段 `summary.json` 发布后调用 `finish(exit_code=0)`。成功收口本身失败也会向根计划传播，不会伪装成完成。

## 产物字段

### 根 `run.json` 与根 `summary.json`

`experiment.run_plan` 在任何阶段开始前发布根 `run.json`：

| 字段 | 含义 |
| --- | --- |
| `plan_path` | 输入 YAML `resolve()` 后的本机绝对路径字符串 |
| `plan_name` | 解析后的计划名 |
| `run_stamp` | `TASK_RUN_STAMP` 的安全单级值；缺失或空时使用当前 UTC 秒级时间 |
| `hostname` | `socket.gethostname()` 返回值 |
| `slurm_job_id` | `SLURM_JOB_ID` 字符串或 `null` |
| `task_launch_dir` | `TASK_LAUNCH_DIR` 字符串或 `null` |
| `tracking` | 七字段 TrackingConfig 映射 |
| `started_at_utc` | UTC ISO 8601 时间字符串 |

全部阶段成功后，根 `summary.json` 复制上述八个字段，新增 `finished_at_utc` 和与 YAML 顺序一致的 `stages` 列表。任一阶段失败时根 summary 不发布；已存在根 run 与阶段文件不回滚。

### 阶段 `resolved.json`

顶层字段如下：

| 字段 | 结构 |
| --- | --- |
| `stage` | 完整 StageConfig；`model` 使用 `MatcherModelConfig.to_dict()`，`training.loss_weights` 为四字段映射 |
| `data` | 十一项 YAML data 字段加六项 loader 派生事实 |
| `tracking` | 七字段 TrackingConfig |
| `device` | `torch.cuda.get_device_name(device)` |
| `torch_version` | 当前 `torch.__version__` |
| `frozen_optimizer_semantics` | 上文列出的 AdamW、调度器和 GradScaler 固定字段 |
| `numeric_backend` | 五字段随机与数值后端记录 |
| `initialization` | 完整 CandidateInitializationRecord 或 `null` |
| `train_candidate_count` | train Dataset 清单候选总数 |
| `train_candidates_used_per_epoch` | 截短后每轮实际训练候选数 |
| `validation_candidate_count` | validation Dataset 清单候选总数 |

`resolved.json.data` 的六项派生事实是：

| 字段 | 值 |
| --- | --- |
| `multiprocessing_context` | `spawn` |
| `persistent_workers` | `false` |
| `worker_pool_lifecycle` | `per_complete_loader_iteration` |
| `train_worker_generator_seed` | `2*seed` |
| `validation_worker_generator_seed` | `2*seed+1` |
| `worker_generator_lifecycle` | 每个 loader 各一个生成器，创建迭代器只推进自身生成器的说明字符串 |

### `history.jsonl`

每个完成 epoch 追加一条记录：

| 字段 | 含义 |
| --- | --- |
| `epoch` | 零起始轮次 |
| `global_step` | 本轮结束累计更新数 |
| `learning_rate_used` | 本轮训练开始前学习率 |
| `next_learning_rate` | 本轮 validation 调度后的学习率 |
| `selection_score` | 本轮完整 validation 选择分数 |
| `improved` | 是否严格超过此前最佳值 |
| `bad_validation_epochs` | 当前连续未改善轮数 |
| `training` | 完整 `TrainingEpochResult` |
| `validation` | 完整 `EvaluationResult` |
| `timestamp_unix` | 记录构造时 `time.time()` 秒数 |

`training` 的字段如下：

| 字段 | 类型与含义 |
| --- | --- |
| `loss` | 五字段 `LossValues` 映射 |
| `loss.total` | 四项损失按配置系数相加的全局损失 |
| `loss.classification` | 按全轮真实分类分母归一的六分类损失 |
| `loss.ligand_area` | 按全轮候选数平均的配体区域损失 |
| `loss.binding_area` | 按全轮候选数平均的结合区域损失 |
| `loss.identity` | 按全轮有效身份候选数平均的身份损失；分母为零时为 0 |
| `sample_count` | 本轮实际训练候选数 |
| `identity_eligible_count` | 本轮有效身份候选数 |
| `optimizer_steps` | 本轮完成的参数更新数 |
| `skipped_optimizer_steps` | 当前失败关闭策略下固定为 0 |
| `mean_preclip_gradient_norm` | 每次更新裁剪前全模型 L2 梯度范数均值 |
| `maximum_preclip_gradient_norm` | 每次更新裁剪前全模型 L2 梯度范数最大值 |
| `mean_graph_branch_gradient_norm` | 裁剪后图 PairScorer 梯度范数均值；不适用时为 `null` |
| `mean_inventory_branch_gradient_norm` | 裁剪后清单 PairScorer 梯度范数均值；不适用时为 `null` |
| `identity_diagnostics` | 完整 Matcher 的两个身份门值映射；候选预训练为 `null` |
| `identity_diagnostics.graph_gate` | 轮末图门 Python 浮点值 |
| `identity_diagnostics.inventory_gate` | 轮末清单门 Python 浮点值 |
| `elapsed_seconds` | 完整训练轮墙钟秒数 |
| `data_wait_seconds` | 各 global batch 等待 DataLoader 的秒数和 |
| `compute_seconds` | 各 global batch 搬运、前向、反向、更新及 CUDA 同步秒数和 |
| `samples_per_second` | `sample_count / max(elapsed_seconds,1e-12)` |
| `peak_cuda_allocated_bytes` | 本轮最大已分配 CUDA 显存 byte 数 |
| `peak_cuda_reserved_bytes` | 本轮最大保留 CUDA 显存 byte 数 |
| `global_step` | 本轮结束后的累计参数更新数 |

`validation` 的顶层字段如下：

| 字段 | 类型与含义 |
| --- | --- |
| `loss` | 与 `training.loss` 同字段、按完整 validation 真实分母汇总的 `LossValues` |
| `metrics` | 完整 `EvaluationReport` |
| `sample_count` | validation 实际候选数 |
| `identity_eligible_count` | validation 有效身份候选数 |
| `elapsed_seconds` | 完整 validation 墙钟秒数 |
| `samples_per_second` | validation 候选吞吐 |

`validation.metrics` 的固定树如下；`identity_per_class` 以五个前景类别名为动态键，每个值与固定身份范围采用相同三子树：

| 字段 | 含义 |
| --- | --- |
| `six_class` | 全候选六分类报告 |
| `six_class.accuracy` | 全候选六分类准确率 |
| `six_class.macro_f1` | 与正式选模兼容的支持类 macro-F1 |
| `six_class.macro_f1_supported_classes` | 只平均真实支持数大于零类别的 macro-F1 |
| `six_class.macro_f1_all_classes` | 固定平均六类，零支持类 F1 按 0 计 |
| `six_class.per_class` | 以六个类别名为键的逐类指标映射 |
| `six_class.per_class[class_name].precision` | 当前类别 precision；零预测时为 NaN |
| `six_class.per_class[class_name].recall` | 当前类别 recall；零支持时为 NaN |
| `six_class.per_class[class_name].f1` | 当前类别 F1；无法计算时为 0 |
| `six_class.per_class[class_name].support` | 当前类别真实候选数 |
| `six_class.per_class[class_name].predicted_count` | 当前类别预测候选数 |
| `six_class.confusion_matrix` | `6×6` 整数嵌套列表；第一轴真实类、第二轴预测类 |
| `intrinsic_identity` | 配置身份范围内忽略六分类预测的身份报告 |
| `intrinsic_identity.top1` | 固有身份 Top-1 正确率 |
| `intrinsic_identity.topk` | 固有身份 Top-K 正确率 |
| `intrinsic_identity.top_k` | 固有身份报告采用的最大名次 K |
| `intrinsic_identity.eligible_count` | 固有身份可监督候选数 |
| `conditional_identity` | 只在六分类预测正确时统计的身份报告 |
| `conditional_identity.top1` | 条件身份 Top-1 正确率 |
| `conditional_identity.topk` | 条件身份 Top-K 正确率 |
| `conditional_identity.top_k` | 条件身份报告采用的最大名次 K |
| `conditional_identity.eligible_count` | 条件身份候选数 |
| `conditional_identity.seen_top1` | 当前语言家族 train seen 子集 Top-1 |
| `conditional_identity.seen_topk` | 当前语言家族 train seen 子集 Top-K |
| `conditional_identity.seen_count` | 当前语言家族 train seen 子集候选数 |
| `conditional_identity.unseen_top1` | 当前语言家族 train unseen 子集 Top-1 |
| `conditional_identity.unseen_topk` | 当前语言家族 train unseen 子集 Top-K |
| `conditional_identity.unseen_count` | 当前语言家族 train unseen 子集候选数 |
| `end_to_end_identity` | 配置身份范围经前景分类门后的最终身份报告 |
| `end_to_end_identity.precision` | 端到端身份 precision |
| `end_to_end_identity.recall` | 端到端身份 recall |
| `end_to_end_identity.f1` | 端到端身份 F1 |
| `end_to_end_identity.true_positive` | 通过前景门且最终身份正确的候选数 |
| `end_to_end_identity.predicted_positive` | 通过前景门且最终身份属于配置范围的候选数 |
| `end_to_end_identity.actual_positive` | 配置范围内具备身份目标的真实候选数 |
| `small_molecule_identity` | 只含小分子身份且使用严格小分子分类门的范围报告 |
| `small_molecule_identity.intrinsic.top1` | 小分子固有 Top-1 |
| `small_molecule_identity.intrinsic.topk` | 小分子固有 Top-K |
| `small_molecule_identity.intrinsic.top_k` | 小分子报告采用的最大名次 K |
| `small_molecule_identity.intrinsic.eligible_count` | 小分子固有身份候选数 |
| `small_molecule_identity.conditional.top1` | 小分子条件 Top-1 |
| `small_molecule_identity.conditional.topk` | 小分子条件 Top-K |
| `small_molecule_identity.conditional.top_k` | 小分子条件报告采用的最大名次 K |
| `small_molecule_identity.conditional.eligible_count` | 小分子条件身份候选数 |
| `small_molecule_identity.end_to_end.precision` | 小分子端到端 precision |
| `small_molecule_identity.end_to_end.recall` | 小分子端到端 recall |
| `small_molecule_identity.end_to_end.f1` | 小分子端到端 F1 |
| `small_molecule_identity.end_to_end.true_positive` | 小分子端到端最终身份正确数 |
| `small_molecule_identity.end_to_end.predicted_positive` | 小分子端到端预测阳性数 |
| `small_molecule_identity.end_to_end.actual_positive` | 小分子端到端真实阳性数 |
| `all_non_background_identity` | 五类前景共享身份清单的范围报告 |
| `all_non_background_identity.intrinsic.top1` | 五类前景固有 Top-1 |
| `all_non_background_identity.intrinsic.topk` | 五类前景固有 Top-K |
| `all_non_background_identity.intrinsic.top_k` | 五类前景报告采用的最大名次 K |
| `all_non_background_identity.intrinsic.eligible_count` | 五类前景固有身份候选数 |
| `all_non_background_identity.conditional.top1` | 五类前景条件 Top-1 |
| `all_non_background_identity.conditional.topk` | 五类前景条件 Top-K |
| `all_non_background_identity.conditional.top_k` | 五类前景条件报告采用的最大名次 K |
| `all_non_background_identity.conditional.eligible_count` | 五类前景条件身份候选数 |
| `all_non_background_identity.end_to_end.precision` | 五类前景端到端 precision |
| `all_non_background_identity.end_to_end.recall` | 五类前景端到端 recall |
| `all_non_background_identity.end_to_end.f1` | 五类前景端到端 F1 |
| `all_non_background_identity.end_to_end.true_positive` | 五类前景端到端最终身份正确数 |
| `all_non_background_identity.end_to_end.predicted_positive` | 五类前景端到端预测阳性数 |
| `all_non_background_identity.end_to_end.actual_positive` | 五类前景端到端真实阳性数 |
| `identity_per_class` | 以五个前景类别名为键的逐类身份范围报告 |
| `identity_per_class[class_name].intrinsic.top1` | 当前前景类固有 Top-1 |
| `identity_per_class[class_name].intrinsic.topk` | 当前前景类固有 Top-K |
| `identity_per_class[class_name].intrinsic.top_k` | 当前前景类报告采用的最大名次 K |
| `identity_per_class[class_name].intrinsic.eligible_count` | 当前前景类固有身份候选数 |
| `identity_per_class[class_name].conditional.top1` | 当前前景类条件 Top-1 |
| `identity_per_class[class_name].conditional.topk` | 当前前景类条件 Top-K |
| `identity_per_class[class_name].conditional.top_k` | 当前前景类条件报告采用的最大名次 K |
| `identity_per_class[class_name].conditional.eligible_count` | 当前前景类条件身份候选数 |
| `identity_per_class[class_name].end_to_end.precision` | 当前前景类端到端 precision |
| `identity_per_class[class_name].end_to_end.recall` | 当前前景类端到端 recall |
| `identity_per_class[class_name].end_to_end.f1` | 当前前景类端到端 F1 |
| `identity_per_class[class_name].end_to_end.true_positive` | 当前前景类端到端最终身份正确数 |
| `identity_per_class[class_name].end_to_end.predicted_positive` | 当前前景类端到端预测阳性数 |
| `identity_per_class[class_name].end_to_end.actual_positive` | 当前前景类端到端真实阳性数 |
| `area` | 两个体素区域的微平均 Dice 报告 |
| `area.ligand_micro_dice` | 配体区域微平均 Dice |
| `area.binding_micro_dice` | 结合区域微平均 Dice |
| `area.ligand_dice_denominator` | 配体区域预测体素数与真实体素数之和 |
| `area.binding_dice_denominator` | 结合区域预测体素数与真实体素数之和 |

### `LAST.pt` 与 `BEST.pt`

checkpoint 使用 `torch.save`，只应从受信任项目路径加载；`load_checkpoint(weights_only=false)` 会执行 Python 反序列化。每轮先覆盖 `LAST.pt`，严格改善时再用同一状态覆盖 `BEST.pt`。写入使用同目录 `.{目标名}.tmp.{PID}`、文件 `fsync` 与 `os.replace`；目录本身不执行 `fsync`。PID 命名不为同一进程内并发调用提供不同临时路径或互斥；失败后的 `finally` 会尝试删除临时文件，但删除本身也可失败、遮蔽原异常并留下临时文件。

checkpoint 顶层字段是：

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 当前为 1 |
| `stage_name` | 阶段名 |
| `task_name` | 任务名 |
| `epoch` | 当前轮次 |
| `global_step` | 当前累计更新数 |
| `best_epoch` | 截至当前最佳轮次 |
| `best_score` | 截至当前最佳分数 |
| `bad_validation_epochs` | 当前连续未改善轮数 |
| `model_config` | 完整规范化模型配置 |
| `training_config` | 完整训练配置普通映射 |
| `model_state` | 模型参数与缓冲区 state dict |
| `optimizer_state` | AdamW state dict |
| `scheduler_state` | ReduceLROnPlateau state dict |
| `scaler_state` | GradScaler state dict；禁用时可为空映射 |
| `epoch_record` | 与同轮 history 条目同源的完整 EpochRecord |

`load_checkpoint` 只验证 `schema_version` 与 `model_state` 两个最小字段，保留其它顶层键但不核验其类型或关系。候选预训练初始化由 `initialization.py` 继续执行严格交叉核验。

### 阶段 `summary.json`

训练循环完成后先写下列文件，再调用 W&B `finish(exit_code=0)`。只有该调用正常返回，`run_training_stage` 才返回同源映射并允许根计划继续；若 finish 失败，阶段 `summary.json` 可以保留，但不能单凭该文件断言阶段或远端 W&B run 已成功收口：

| 字段 | 含义 |
| --- | --- |
| `stage` | 阶段名 |
| `best_epoch` | 最佳 epoch |
| `best_score` | 最佳 validation 选择分数 |
| `last_epoch` | 实际完成的最后 epoch |
| `global_step` | 阶段累计更新数 |
| `stopped_by_patience` | 是否因 patience 停止 |
| `best_checkpoint` | `BEST.pt` 路径字符串 |
| `last_checkpoint` | `LAST.pt` 路径字符串 |
| `tracking_run_id` | W&B run ID |
| `tracking_url` | W&B HTTPS URL |

## 失败关闭边界

- 阶段目录和计划运行目录都用 `exist_ok=false` 创建；同名目录不会复用或覆盖。
- 配置、CUDA、manifest、Dataset、初始化、W&B、逐物理批次非有限损失张量、非有限梯度、写盘和同步错误直接传播。跨批次 Python float 乘加仍可能溢出，聚合器没有最终有限性门禁；直接 `evaluate_model` 可返回含 Inf 的损失，阶段路径会先把该值以 `null` 写入 history、以 Inf 保存在 checkpoint，再由后续 W&B 指标规范化拒绝未登记的 Inf。
- `resolved.json` 在 W&B 初始化前发布；`tracking.json` 在首次 DataLoader worker 启动前发布。
- `history.jsonl` 的本轮记录先于本轮 checkpoint 写入；中途写盘错误可能留下已经完成的前序产物，不执行事务回滚。
- checkpoint 临时文件仅按 PID 命名，不提供同进程并发互斥；失败清理也可能失败并遮蔽原始异常。
- 训练或阶段收口失败不会生成成功的根 `summary.json`。
- `BEST.pt` 只证明当前训练 selection 规则下的最佳 validation 轮次；它不证明 calibration 结果，也不包含 release、launch 或文件摘要证据。

## 回归验证边界

本目录的主要机械与行为回归由以下测试覆盖：

- `matcher/tests/test_model_and_loss.py`：四项损失、身份变体、空身份与模型梯度；
- `matcher/tests/test_metrics_and_config.py`：正式 YAML、跨物理批量累积、runtime、checkpoint、实验顺序和选模；
- `matcher/tests/test_initialization.py`：候选预训练来源交叉核验、历史迁移和失败前参数不变；
- `matcher/tests/test_tracking.py`：W&B 元数据、间隔、空值、注意力偏置遥测与失败收口；
- `matcher/tests/test_calibration.py`：训练配置、checkpoint 与数值后端被保留评估复用的边界，不读取真实 calibration 数据。
