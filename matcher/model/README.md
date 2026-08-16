# `matcher/model` 模型结构与数值契约

## 中心职责

本目录把 [`matcher/datasets`](../datasets/README.md) 提供的 `CandidateBatch` 转换成候选六分类、两个体素区域预测和 PDB 内联合身份分数：

- `candidate_encoder.py` 编码 `105×48³` 候选体素与 A、P、V、PP 四种 token。
- `raunet.py` 实现带回收状态、注意力门和三轴旋转位置编码的三维 RAUNet。
- `fusion.py` 实现 token 投影、固定统计归一化，以及 `mean_max`、`candidate_cross_attention`、`mean_max_candidate_cross_attention_concat` 三种候选聚合。
- `ligand_matcher.py` 编码语言向量、LigandObject 模板图和同 PDB 身份清单，再计算三条身份分数分支。
- `config.py` 冻结上述结构参数，并为历史 checkpoint 补齐明确的缺省字段。

本目录不建立联合身份，不选择 occurrence 语言向量，不跨 PDB 合并身份，也不决定 `oracle`、`detached` 或 `joint` 身份损失模式。联合身份是数据侧已经建立的 `(object_key, PreparedLigand 顶层 smiles)`；模型数值前向不读取身份字符串。

## 形状符号与固定顺序

| 符号 | 含义 |
| --- | --- |
| `B` | 当前物理批次的候选数 |
| `I` | 当前批次补零后的最大 PDB 内身份数，可为 0 |
| `D_m` | 候选和身份共用的模型表示宽度，即 `model_dim` |
| `C_v` | RAUNet 最高分辨率输出通道数，即 `voxel_feature_channels` |
| `N_A`、`N_P`、`N_V`、`N_PP` | 当前批次各模态补零后的 token 数 |
| `N` | 全部 LigandObject 图连接后的原子数 |
| `E` | 全部 LigandObject 图连接后的有向边数 |
| `G` | 全部 LigandObject 图数；正式完整 Matcher 批次中等于有效身份总数 |
| `K_m` | 候选条件注意力、配体图注意力和身份清单注意力共用的头数，即 `attention_heads` |
| `K_r` | RAUNet 瓶颈注意力独立选择的头数；不读取 `attention_heads` |
| `K_g` | 单个 RAUNet 跳跃门按当前粗特征通道选择的不超过 8 的最大整除头数；不读取 `attention_heads` |

六分类分数 `class_logits[:, class_index]` 的顺序固定如下。

| `class_index` | 类别 |
| ---: | --- |
| 0 | `background` |
| 1 | `ion` |
| 2 | `nucleotide_like` |
| 3 | `peptide_like` |
| 4 | `small_molecule` |
| 5 | `sugar` |

## 文件与公共入口

| 文件 | 主要入口 | 输入 | 输出 |
| --- | --- | --- | --- |
| `config.py` | `MatcherModelConfig` | Python 配置值或历史配置字典 | 通过结构门禁的 frozen dataclass |
| `raunet.py` | `RaUnetBackbone` | `(B,105,48,48,48)` 候选体素 | `(B,C_v,48,48,48)` 体素特征 |
| `fusion.py` | `MeanMaxAggregator`、`CandidateConditionedAggregator`、`MeanMaxCandidateConditionedResidualAggregator` | 候选体素摘要、四模态 token、可选统计 | `(B,D_m)` 候选表示 |
| `candidate_encoder.py` | `CandidateEncoder` | `CandidateBatch` 的候选数值字段 | `CandidateEncoderOutput` |
| `ligand_matcher.py` | `ClosedSetMatcher` | 完整 `CandidateBatch` | `FullMatcherOutput` |
| `__init__.py` | 包级稳定导出 | 模型构造与类型名称 | `__all__` 声明的六个名称 |

包级稳定导出为 `MatcherModelConfig`、`CandidateEncoder`、`CandidateEncoderOutput`、`ClosedSetMatcher`、`FullMatcherOutput` 和 `IdentityScoreComponents`。图消息层、配对打分器等辅助类保留在实现模块内，不由 `__all__` 承诺为稳定公共入口。

## `MatcherModelConfig`

### 字段、取值和直接门禁

| 字段 | 类型与允许值 | 结构作用 |
| --- | --- | --- |
| `aggregation` | `mean_max`、`candidate_cross_attention` 或 `mean_max_candidate_cross_attention_concat` | 选择四模态掩码均值/最大值、候选条件交叉注意力，或两个完整摘要的零输出残差拼接 |
| `use_manual_features` | 精确 `bool` | 控制候选聚合和三个身份 PairScorer 是否使用 12 维 A/P/V/PP 统计；身份分支另加候选的身份类别概率 |
| `use_sampling_features` | 精确 `bool` | 控制候选聚合和三个身份 PairScorer 是否使用 4 维裁剪统计 |
| `use_candidate_token_attention_bias` | 精确 `bool` | 只允许与 `candidate_cross_attention` 同时启用；为四模态建立独立坐标/概率偏置网络 |
| `initial_graph_gate` | 有限数值，`[0,1]` | 图分数增量的可训练标量门初值 |
| `initial_inventory_gate` | 有限数值，`[0,1]` | 同 PDB 身份清单增量的可训练标量门初值 |
| `model_dim` | 正的精确 `int` | 候选、token、语言和图表示宽度 `D_m`；必须能被 `attention_heads` 整除 |
| `attention_heads` | 正的精确 `int` | 候选、配体图和身份清单注意力共用头数 `K_m`；不控制 RAUNet 瓶颈注意力 |
| `dropout` | 有限数值，`[0,1)` | token、注意力、图消息和配对残差网络的丢弃率 |
| `voxel_feature_channels` | 正的精确 `int` | RAUNet 输出与两个体素预测头的通道数 `C_v` |
| `raunet_planes` | 九项正整数元组 | `enc0, enc1, enc2, enc3, bottleneck, dec3, dec2, dec1, dec0` |
| `recycle_iterations` | 正的精确 `int` | 同一候选体素执行的 RAUNet 回收轮数 |
| `gradient_checkpointing` | 精确 `bool` | 是否在训练且梯度启用时对高开销 RAUNet 模块做非重入激活重计算 |
| `bottleneck_attention_layers` | 正的精确 `int` | 最深网格串联的三轴旋转位置编码自注意力层数 |
| `graph_layers` | 正的精确 `int` | LigandObject 原子图消息层数 |

`MatcherModelConfig` 只检查配置层能够确定的关系。`RaUnetBackbone` 构造时另要求 `enc3=bottleneck=dec3` 和 `dec0=enc0`。`candidate_pretrain` 必须把两个身份门初值设为 0 的跨任务规则由 `matcher/training/config.py` 检查，不属于模型配置本身。

两个身份门只在初始化时受 `[0,1]` 门禁。`IdentityScorer` 把它们保存成不经过 sigmoid、也不夹紧的 `Parameter`，所以训练后的值可以小于 0 或大于 1。

### 序列化与历史迁移

`to_dict()` 返回全部 15 个字段组成的顶层普通字典；`raunet_planes` 在 Python 字典中仍为元组，JSON 编码时才变成数组。`from_dict()` 先复制调用者字典，再执行以下唯一历史迁移：

| 缺失字段 | 补充值 |
| --- | ---: |
| `use_manual_features` | `False` |
| `use_sampling_features` | `False` |
| `use_candidate_token_attention_bias` | `False` |
| `initial_graph_gate` | `0.0` |
| `initial_inventory_gate` | `0.0` |

`raunet_planes` 的列表或元组会统一转成元组。其它缺失键、未知键和不支持的结构不会被静默忽略，而会在 dataclass 构造或字段门禁处失败。

## 候选编码契约

### 输入字段

`CandidateEncoder.forward` 消费以下 `CandidateBatch` 字段。A、P token 来自完整 `80³` 框；V、PP token 与位置下标来自实际 `48³` 裁剪。

| 字段 | dtype 与形状 | 固定内容 |
| --- | --- | --- |
| `voxel_input` | 浮点 `(B,105,48,48,48)` | 通道优先，空间轴为 Z/Y/X；磁盘 `float16` 在主干前转换为输入卷积参数 dtype |
| `a_tokens` | 浮点 `(B,N_A,229)` | 末四项依次为 Stage1 概率、相对 X/Y/Z |
| `a_mask` | `bool (B,N_A)` | True 为有效 A token |
| `p_tokens` | 浮点 `(B,N_P,116)` | 末四项依次为 Stage1 概率、相对 X/Y/Z |
| `p_mask` | `bool (B,N_P)` | True 为有效 P token |
| `v_tokens` | 浮点 `(B,N_V,52)` | 末四项依次为 Stage1 概率、相对 X/Y/Z |
| `v_mask` | `bool (B,N_V)` | True 表示 V token 与 `v_index_zyx` 同时有效 |
| `v_index_zyx` | `int64 (B,N_V,3)` | V token 在 RAUNet 最高分辨率特征中的 Z/Y/X 下标 |
| `pp_tokens` | 浮点 `(B,N_PP,4)` | 四项依次为 Stage1 概率、相对 X/Y/Z |
| `pp_mask` | `bool (B,N_PP)` | True 表示 PP token 与 `pp_index_zyx` 同时有效 |
| `pp_index_zyx` | `int64 (B,N_PP,3)` | PP token 在 RAUNet 最高分辨率特征中的 Z/Y/X 下标 |
| `candidate_manual_features` | `float32 (B,12)` | A/P/V/PP 各三项：token 数、概率均值、概率最大值 |
| `candidate_sampling_features` | `float32 (B,4)` | 边界受限标志、中心偏移 L2、中心偏移 L∞、V 保留比例 |

`CandidateEncoder` 仅在相应开关开启时索引并归一化两个候选级统计字段；但是完整 `ClosedSetMatcher.forward` 无条件索引这两个键并传入身份打分器。关闭开关只表示数值不参与候选或身份计算，不能省略完整 Matcher 批次中的键。

### 体素主干与候选摘要

`RaUnetBackbone` 产生 `voxel_features (B,C_v,48,48,48)`。候选摘要先分别沿三个空间轴计算逐通道均值与最大值，按“全部均值、全部最大值”连接成 `(B,2C_v)`，再投影为 `(B,D_m)`。

V 与 PP token 使用 `gather_voxel_features` 从 `voxel_features` 的同一 Z/Y/X 位置读取 `(B,N_m,C_v)` 特征，再经带偏置 MLP 投影到 `D_m` 并加到 token 投影。即使补零位置的采样值为零，带偏置 MLP 仍可能输出非零，所以相加后必须再次乘 `v_mask` 或 `pp_mask`；这一步不是冗余清零。

`VoxelBinaryHead` 从同一 `voxel_features` 产生 `ligand_area_logits` 和 `binding_area_logits`。两者都是未经过 sigmoid 的 `(B,48,48,48)` 二分类 logits。

### 候选输出

| 字段 | dtype 与形状 | 消费者 |
| --- | --- | --- |
| `candidate_embedding` | 浮点 `(B,D_m)` | 六分类头与三个身份 PairScorer |
| `class_logits` | 浮点 `(B,6)` | 六分类损失；启用手动特征时还生成无梯度身份类别概率 |
| `ligand_area_logits` | 浮点 `(B,48,48,48)` | 配体区域二分类损失与评估 |
| `binding_area_logits` | 浮点 `(B,48,48,48)` | 受体结合区域二分类损失与评估 |
| `voxel_features` | 浮点 `(B,C_v,48,48,48)` | `CandidateEncoder` 内部用于空间均值/最大值摘要、V/PP 位置采样和两个体素预测头，随后作为输出字段返回；当前正式损失与评估不按该键读取 |

## 四模态融合

### 投影与统计归一化

四种原始 token 分别使用独立的 `LayerNorm → Linear → GELU → Dropout → LayerNorm` 投影器；投影后再次乘对应 mask，使带偏置线性层不能污染补零 token。

`candidate_manual_features` 按 A、P、V、PP 分成四组三元组。每组三元组只有 token 数替换为 `log1p(count)`，Stage1 概率均值和最大值保持原值。模型函数只检查全部值有限且非负，不再次检查概率上限 1；该范围由 Dataset 契约保证。

`candidate_sampling_features` 的归一化如下：

| 位置 | 输入 | 模型输入 |
| ---: | --- | --- |
| 0 | 边界受限标志，0 或 1 | 保持原值 |
| 1 | 实际裁剪中心到完整 V token 质心的 L2 距离，单位 voxel | 除以 `24×sqrt(3)` |
| 2 | 同一位移的 L∞ 距离，单位 voxel | 除以 `24` |
| 3 | V 保留比例，`[0,1]` | 保持原值 |

### `mean_max`

每种模态分别沿有效 token 轴计算 `(B,D_m)` 均值和逐通道最大值。`N_m=0` 或某个候选的 mask 全 False 时，两项都严格为零，不产生 NaN 或 dtype 最小值。

最终连接顺序固定为：候选体素摘要、可选 4 维采样统计、A 均值、A 最大值、可选 A 三项手动统计，然后按同一结构依次加入 P、V、PP。基础宽度为 `9D_m`，再按开关追加 12 和 4。

### `candidate_cross_attention`

候选体素摘要分别查询 A、P、V、PP 四种模态。每个分支有独立 query、key、value 和输出投影；softmax 只沿当前模态 token 轴归一化。补零 token 在加入可选偏置后仍被 mask 排除。某模态 `N_m=0` 或某个候选没有有效 token 时，该模态增量严格为零。训练时 dropout 作用于已归一化权重，丢弃后不再次归一化。

最终连接顺序固定为：候选体素摘要、可选 4 维采样统计、A 注意力增量、可选 A 三项手动统计，然后按同一结构依次加入 P、V、PP。基础宽度为 `5D_m`，再按开关追加 12 和 4。

PP 分支的输出投影权重与偏置初始化为零，因此初始 PP 增量严格为零；A、P、V 输出投影使用 PyTorch 常规初始化。零初始化只描述初始参数，优化器更新后 PP 分支可以产生非零增量。

### `mean_max_candidate_cross_attention_concat`

该结构分别运行完整 `MeanMaxAggregator` 和完整 `CandidateConditionedAggregator`，得到两个浮点 `(B,D_m)` 摘要。两个摘要按“mean/max、候选条件注意力”的固定顺序连接为 `(B,2D_m)`，再由 `LayerNorm → Linear → GELU → Dropout → Linear` 映射成 `(B,D_m)` 学习残差，并加回 mean/max 主摘要。残差末层的权重和偏置均初始化为零，因此初始输出与同参数 mean/max 主路径逐值相等；优化器更新后残差可以非零。

12 项手动统计和 4 项采样统计分别在两个完整聚合器内部复用。拼接层只接收两个 D_m 维摘要，不会第三次追加原始统计。该结构固定关闭候选 token 注意力偏置；调用时 `attention_bias_features` 必须为 `None`。

条件注意力与残差分支在域分离的 CPU 初始化随机流中构造，退出后恢复外层 CPU 随机状态。前向先执行共同 mean/max 主路径，再用 `torch.random.fork_rng` 隔离条件分支和残差分支的训练态 dropout；CPU 随机状态始终保存和恢复，CUDA 候选只保存并恢复候选所在设备的 CUDA 随机状态。该隔离使新增分支的随机消耗不改变后续共同分类头和区域头的随机序列。

### 相对坐标与概率注意力偏置

四种原始 token 的末四个特征分量顺序是“Stage1 概率、相对 X、相对 Y、相对 Z”。偏置网络输入必须先重排为“相对 X、相对 Y、相对 Z、Stage1 概率”，形状为 `(B,N_m,4)`。每种模态有一套互不共享参数的 `4 → D_m → K_m` 网络，输出 `(B,K_m,N_m)` 并在 softmax 前加到内容分数。

每套偏置网络的末层权重与偏置均零初始化，所以启用偏置后的初始前向与同参数无偏置聚合器一致。末层 bias 对同一候选、同一头的全部 token 只增加同一常数，数学上不改变 softmax；末层 weight 才能制造 token 间差异。

PP 注意力输出投影同样从零开始，因此第一次反向中没有从 PP 分支传到偏置末层 weight 的梯度；输出投影经过一次更新后，该路径可以唤醒。A、P、V 的非零初始化输出投影允许偏置末层 weight 在第一次反向获得梯度。

偏置模块在 `torch.random.fork_rng(devices=[])` 内构造。该上下文显式保存并恢复 CPU RNG；它不承诺保存所有设备的随机状态。现有 CPU 构造测试证明启用偏置不会改变共同 CPU 参数初值，也不会改变随后 CPU 随机数序列。

## RAUNet 体素主干

### 编解码尺度与通道表

正式输入固定为 `(B,105,48,48,48)`，空间轴依次为 Z/Y/X。`raunet_planes` 的九项依次为：

| 阶段 | 空间边长 | 通道 |
| --- | ---: | --- |
| `level0` | 48 | `4×enc0` |
| `level1` | 24 | `enc1` |
| `level2` | 12 | `enc2` |
| `level3` | 6 | `enc3` |
| `level4` | 3 | `bottleneck` |
| `decoded3` | 6 | `dec3` |
| `decoded2` | 12 | `dec2` |
| `decoded1` | 24 | `dec1` |
| `decoded0` | 48 | `dec0` |

构造器要求 `enc3=bottleneck=dec3`，使最深特征可上采样后与 `level3` 相加；还要求 `dec0=enc0`，使上一轮 `decoded0` 可直接进入下一轮 `recycle_norm`。最终使用 3、5、7 三种卷积核的最高分辨率分支，每支宽度为 `max(32,dec0//2)`，连接后投影为 `C_v`。

编码下采样使用残差瓶颈；解码每级使用多个 Res2Net 分支块。`AttentionGate3d` 先用粗尺度特征与跳跃特征生成逐头、逐空间位置 sigmoid 门，同一头内的 `C_c/K_g` 通道共享该头门值，再门控跳跃特征并与上采样粗特征相加；它不是沿 token 轴归一化的 dot-product attention。这里的 `K_g` 由当前粗特征通道独立选择，不读取 `attention_heads`。

### 最深网格注意力

最深 `3³` 网格展平为 `L=Z×Y×X` 个位置。瓶颈头数 `K_r` 选择为不超过 8 且能整除 bottleneck 通道数的最大整数；它不读取配置字段 `attention_heads`。每头 query、key、value 宽度为 `6C/K_r`，再平均分给 Z、Y、X 三轴，每轴相邻通道对执行旋转。

位置张量按 Z/Y/X 保存，X 最先变化；相邻网格位置的内部步长固定为 1.5。这个数没有 Å 等物理单位，也不读取候选的世界原点或体素大小。自注意力没有 padding mask，因为规则瓶颈网格的全部位置都有效。

### 回收、梯度与激活重计算

首轮回收状态是 `(B,enc0,48,48,48)` 全零张量。第 0 到 `R-2` 轮在 `torch.no_grad()` 中执行，并把与计算图断开的 `decoded0` 送到下一轮；第 `R-1` 轮不受该内部 `no_grad` 包裹并产生正式 `voxel_features`。最后一轮是否建立计算图服从调用位置的外层梯度模式及参与运算张量的 `requires_grad`；外层 `no_grad` 或 `inference_mode` 下不建图。前置轮仍会改变最后一轮输入状态，但不保留自身的反向图。

每个前置轮结束后调用 `torch.clear_autocast_cache()`，清理当前进程的自动混合精度权重转换缓存；该操作不写文件，也不修改输入张量。

只有 `gradient_checkpointing=True`、模块处于训练模式且全局梯度启用三项同时成立时，高开销子模块才使用 `checkpoint(..., use_reentrant=False, preserve_rng_state=False)`。当前被包装的 RAUNet 高开销模块不含 dropout；若以后加入随机算子，`preserve_rng_state=False` 不会为重计算恢复随机状态。

## 联合身份编码

### 输入身份轴与图字段

| 字段 | dtype 与形状 | 模型用途 |
| --- | --- | --- |
| `language_embeddings` | 浮点 `(B,I,768)` | 投影为 native 身份表示 |
| `identity_mask` | `bool (B,I)` | 定义真实身份槽、语言补零、图 occupancy 和最终 logits 屏蔽 |
| `identity_class_index` | `int64 (B,I)` | 启用手动特征时选择候选六分类中的身份类别概率；真实槽为 1–5，补零槽为 -1 |
| `graph_element` | `int64 (N,)` | 原子序数，模型要求 `[0,128)` |
| `graph_charge` | `int64 (N,)` | 形式电荷，模型要求 `[-16,16]` |
| `graph_residue_id` | `int64 (N,)` | 连接后的残基编号，模型要求 `[0,512)` |
| `graph_node_continuous` | 浮点 `(N,15)` | `0:7` 手性、`7:11` 环大小标志、`11` 模板 `is_present`、`12:15` 组内中心化并除以 10 Å 的参考 X/Y/Z |
| `graph_edge_index` | `int64 (2,E)` | 第 0 行为源原子，第 1 行为目标原子 |
| `graph_edge_features` | 浮点 `(E,10)` | `0:5` 键型、`5:9` 环大小标志、`9` 参考距离除以 10 Å；跨残基边的距离分量为 0 |
| `graph_node_graph_index` | `int64 (N,)` | 每个原子所属的图下标 `[0,G)` |
| `graph_to_sample` | `int64 (G,)` | 每张图所属的样本下标 `[0,B)` |
| `graph_to_identity` | `int64 (G,)` | 每张图在所属样本中的身份槽 `[0,I)` |

语言投影含偏置，所以投影后再次乘 `identity_mask`，保证补零 native 槽严格为零。`IdentityEncoder` 累计 `(graph_to_sample, graph_to_identity)` 的图数，并要求 occupancy 逐槽恰好等于 `identity_mask`：每个真实身份必须一张图，补零身份不得有图，重复图也会失败。

`GraphIdentityAttention` 的 `G=0` 分支在其它形状和范围检查之前直接返回全零 `(B,I,D_m)`。正式 `IdentityEncoder` 的 occupancy 门禁与 Dataset 图契约提供完整条件；直接调用该辅助类时，不能把正式调用保证误认为它的本地门禁。

### 原子图消息与图查询

`LigandGraphEncoder` 把元素、电荷、残基编号三项嵌入与 15 维连续特征投影相加，归一化后执行 `graph_layers` 个消息层。每条有向边的消息是源原子投影与 10 维边特征投影之和；同一目标原子的消息按入度取平均。

消息层通过逐通道 sigmoid 门、消息残差和前馈残差更新原子 token。`E=0` 时聚合消息确实为零，但 `message_output` 含偏置，后续前馈网络也含偏置，所以整层并非恒等映射，也不保证更新为零。

图注意力为每张图从对应 native 身份取一个 query，为每个原子产生 key/value，并只在同一图的原子轴内分头归一化。输出写入 `(B,I,D_m)` 的对应槽。训练时 dropout 作用于已归一化原子权重，丢弃后不再次归一化。

### 自排除身份清单

`SelfExcludingInventoryAttention` 的 query 和 key 都沿同一 PDB 的身份轴 I。有效掩码同时排除补零身份和对角线，因此每个身份只能读取其它真实身份。零身份轴原样返回空张量；单身份样本没有可用 key，输出严格为零；多身份样本得到 `(B,I,D_m)` 清单上下文。dropout 后同样不再次归一化。

三种身份表示如下：

| 字段 | 形状 | 语义 |
| --- | --- | --- |
| `native` | `(B,I,D_m)` | 联合身份语言向量的投影基线 |
| `graph` | `(B,I,D_m)` | native 查询同一身份 LigandObject 原子 token 的结果 |
| `inventory` | `(B,I,D_m)` | native 排除自身后查询同 PDB 其它 native 的结果 |

## 身份配对分数

### 单分支 `PairScorer`

候选与一种身份表示分别经过无偏置线性投影和 L2 归一化，沿 `D_m` 得到 `(B,I)` 余弦分数。余弦尺度保存为可训练 `log_scale`，前向使用 `exp(log_scale)` 并只截断上限 100，不设下限。

兼容性残差固定拼接四个 `(B,I,D_m)` 张量：候选广播、身份、逐元素乘积、逐元素绝对差。开关启用时再追加：

- `(B,I,12)` 的候选模态统计广播；
- `(B,I,1)` 的候选六分类身份类别概率；
- `(B,I,4)` 的候选裁剪统计广播。

身份类别概率从 `softmax(class_logits)` 逐槽 gather 后立即 `detach()`，因此身份损失不会通过这条辅助概率路径回传到 `class_logits`。候选表示仍直接进入三个 PairScorer，身份损失可以沿正常候选配对路径训练候选编码器。

`PairScorer` 不接收 `identity_mask`。补零身份表示即使为零，也可能经兼容性网络偏置产生非零分数；不能依赖零表示自然清除补零槽。

### 三组件、门与最终屏蔽

| 组件 | 计算方式 | 补零槽状态 |
| --- | --- | --- |
| `native` | native PairScorer 原始分数 | 未屏蔽，可能非零 |
| `graph_increment` | graph PairScorer 原始分数乘 `graph_gate` | 未屏蔽，可能非零 |
| `inventory_increment` | inventory PairScorer 原始分数先按“样本有效身份数大于 1”归零，再乘 `inventory_gate` | 未屏蔽；多身份样本的补零槽可能非零 |

清单 PairScorer 在单身份样本上仍可能产生非零原始分数；`IdentityScorer.score_components` 随后用 `has_other_identity` 把该样本整条清单增量严格归零。三个组件只允许在 `identity_mask=True` 的槽位做诊断。

最终 `identity_logits` 是三个组件之和，再把 `identity_mask=False` 槽替换为当前 dtype 的最小有限值，而不是负无穷。若 `I=0`，三个组件与 logits 都是 `(B,0)` 空张量。

## 完整前向输出与诊断

`ClosedSetMatcher.forward` 返回以下九个字段：

| 字段 | 形状 | 屏蔽状态 |
| --- | --- | --- |
| `candidate_embedding` | `(B,D_m)` | 不适用 |
| `class_logits` | `(B,6)` | 未归一化六分类分数 |
| `ligand_area_logits` | `(B,48,48,48)` | 未经过 sigmoid |
| `binding_area_logits` | `(B,48,48,48)` | 未经过 sigmoid |
| `voxel_features` | `(B,C_v,48,48,48)` | 不适用 |
| `identity_logits` | `(B,I)` | 补零槽已写为 dtype 最小有限值 |
| `identity_score_native` | `(B,I)` | 补零槽未屏蔽 |
| `identity_score_graph_increment` | `(B,I)` | 补零槽未屏蔽 |
| `identity_score_inventory_increment` | `(B,I)` | 补零槽未屏蔽 |

`identity_score_components(batch, output)` 只允许在 `eval` 模式调用。它不重新前向，而是建立一个新字典，直接引用 `output` 中三个已保存张量，并只检查形状等于 `batch.identity_mask.shape`。该方法不证明 `batch` 与 `output` 来自同一次前向，也不重新核对组件之和与 `identity_logits`。

`identity_object_keys`、`identity_prepared_smiles`、候选 `metadata` 和其它身份字符串不参与数值前向。修改这些字符串但保持全部数值张量不变，不会改变模型输出；字符串仍是数据分组、监督真值定位和评估可追溯性的必要契约。

## 门禁分层

模型配置、Dataset/collate 与辅助模块各自只负责一层门禁：

- `MatcherModelConfig` 检查正式配置字段类型、范围和少量跨字段关系。
- `CandidateDataset` 与 collate 建立完整字段、固定宽度、补零轴和图连接关系。
- 各模型辅助模块只检查自身代码中显式出现的部分形状、dtype 或范围。

因此，不能把配置层保证写成直接构造辅助类时的本地保证。例如 `RaUnetBackbone` 直接构造允许非正整数 `bottleneck_attention_layers` 建立空注意力序列；`LigandGraphEncoder` 直接构造时负 `layer_count` 会形成空层列表；`GraphIdentityAttention` 的空图分支不会执行后续形状检查。这些行为不是正式训练配置的允许值，但属于实现边界的一部分。

本目录对应的机械回归集中在 [`matcher/tests/test_model_and_loss.py`](../tests/test_model_and_loss.py)：覆盖候选与完整前向、BF16 自动混合精度、空模态、偏置零初始化与唤醒、回收梯度、单身份清单零增量、图 occupancy、身份门和配置迁移。科学训练配置仍由 `configs/experiments` 与训练配置层共同定义。
