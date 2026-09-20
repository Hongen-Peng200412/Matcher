# datasets：从 manifest 到模型批次

本目录把冻结 manifest 中的一个 Stage1 候选物化为 `CandidateItem`，再把一组样本补齐和连接为 `CandidateBatch`。`load_manifest_dataset()` 是唯一公开 Dataset 入口；它不会扫描来源目录或改变 manifest 的候选顺序。

## 推荐阅读顺序

1. `contracts.py`：先掌握 `CandidateMetadata`、`CandidateItem` 和 `CandidateBatch` 的完整字段。
2. `candidate_data.py`：理解具名模型变体各自读取哪些 centered 字段，以及坐标、token 和身份轴如何构造。
3. `batching.py`：理解可复现采样、变长轴补零、掩码和 SMILES 图连接。
4. `__init__.py`：查看目录对外暴露的稳定入口。

## Dataset 过滤规则

`selected_only=True` 只保留 `candidate_selected=True`。`mode=train` 和 `mode=validation` 会跳过身份图或 SMI-TED 向量缺失的真前景；`mode=test` 保留这些候选和原始排名。背景不需要身份答案，因此不会因身份资产缺失而被跳过。

## 记号

- `B`：物理批次候选数。
- `N_A`、`N_P`、`N_V`、`N_PP`：单个候选的四类 token 数；批处理中的 `*_max` 表示批内最大值。
- `I`：候选所属 PDB 的受支持精确 SMILES 数；`I_max` 是批内最大值。
- `G`：批次内连接的 SMILES 图数；图分支启用时等于所有有效身份槽数，`trunk_only` 为 0。
- `N`、`E`：批次内全部图连接后的节点数和有向边数。

## `CandidateMetadata`

`metadata` 不进入模型数值计算，只保存来源、排序和真值解释。

| 字段 | 类型 | 含义与特殊值 | 示例值 |
| --- | --- | --- | --- |
| split | str | manifest 划分名。 | "validation" |
| pdb_id | str | 小写 PDB 标识。 | "9abc" |
| centered_box_index | int | F2_centered.npz 的稳定候选编号。 | 7 |
| source_blob_index | int | Stage1 源 blob 编号。 | 3 |
| candidate_selected | bool | Stage1 是否选择该候选。 | true |
| gaussian_score | float | Gaussian 排序分数。 | 0.731 |
| source_probability_mean | float | 源 blob 概率均值排序分数。 | 0.684 |
| foreground_target | bool | 双向 0.30 覆盖门是否产生真前景。 | true |
| target_category | `str | null` | 五类真值分层标签；背景为 null。 |
| matched_occurrence_id | `int | null` | 唯一最大覆盖 occurrence；背景为 null。 |
| target_smiles | `str | null` | 唯一正确精确 SMILES；背景或没有可用字符串时为 null。 |
| identity_supported | bool | target_smiles 是否同时出现在本 PDB 的合法图和语言身份轴中。 | true |

## `CandidateItem`

每个字段对应一个候选，变长数组尚未补齐。token 坐标都以当前 48 立方体中心为参考；索引数组用 Z/Y/X 轴顺序，token 中的三维相对坐标用 X/Y/Z 顺序。

| 字段 | 数据类型与形状 | 含义、对齐和特殊值 |
| --- | --- | --- |
| metadata | CandidateMetadata | 上述非张量来源信息。 |
| voxel_input | float16 (C,48,48,48) | strongest、trunk_only 的 C=106；density_only、density_blob_only 的 C=1；blob_only、pocket_only 的 C=0。106 通道由 56 个密度通道和 Dataset 散射的 50 个受体通道连接。 |
| a_tokens | float32 (N_A,D_A) | strongest、trunk_only 的 D_A=230，切片依次为 L0 [0:50]、L1 [50:114]、L2 [114:178]、L3 [178:226]、概率 [226]、相对 XYZ [227:230]；pocket_only 为原始 L0 D_A=50；禁用时 (0,230)。 |
| p_tokens | float32 (N_P,116) | 切片依次为 L2 [0:64]、L3 [64:112]、概率 [112]、相对 XYZ [113:116]；禁用时 (0,116)。 |
| v_tokens | float32 (N_V,68) | supplied crop 内 V token；切片为 64 维特征 [0:64]、概率 [64]、相对 XYZ [65:68]；禁用时 (0,68)。 |
| v_index_zyx | int64 (N_V,3) | 与 v_tokens 逐项对齐的 48 立方体 Z/Y/X 整数索引；禁用时 (0,3)。 |
| pp_tokens | float32 (N_PP,4) | 按 Stage1 概率降序、C 顺序下标稳定选择；列依次为概率和相对 X/Y/Z；禁用时 (0,4)。 |
| pp_index_zyx | int64 (N_PP,3) | 与 pp_tokens 对齐的 Z/Y/X 索引；禁用时 (0,3)。 |
| foreground_target | bool | 单一前景二分类目标。 |
| ligand_area_target | bool (48,48,48) | 全部真实 occurrence 稀疏体素并集投影到 supplied crop；blob_only、density_only、pocket_only 返回空全假数组且区域损失权重必须为 0。 |
| binding_area_target | bool (48,48,48) | 受体 binding_atom=True 原子投影到 supplied crop；blob_only、density_only、pocket_only 同样为全假数组。 |
| full_candidate_voxel_count | int | crop 前完整 centered 候选的 V 体素数；例如 321，用于解释裁剪保留比例。 |
| candidate_manual_features | float32 (12,) | 按 A、P、V、PP 分四组，每组依次为 token 数、概率均值、概率最大值；禁用模态的三项为 0，density_only、pocket_only 全 0。 |
| candidate_sampling_features | float32 (4,) | 依次为是否触及 crop 边界、crop 中心到 V 质心的 L2 距离、L∞ 距离和 V 保留比例；距离单位为体素，density_only、pocket_only 全 0。 |
| identity_smiles | list[str]，长度 I | 当前 PDB 的受支持精确 SMILES，按字符串递增；可为空。 |
| language_embeddings | float32 (I,768) | 与 identity_smiles 逐槽对齐的 SMI-TED 向量。 |
| smiles_graphs | list[SmilesGraph] | 图分支启用时长度为 I 并逐槽对齐；trunk_only 固定为空列表。每图含 element (N_i,)、charge (N_i,)、atom_in_ring (N_i,4)、edge_index (2,E_i) 和 edge_features (E_i,9)。 |
| identity_target | int | 正确身份槽下标 0..I-1；背景和不具备受支持身份的真前景均为 -1。 |

## `CandidateBatch`

`collate_candidates()` 在每个 `CandidateItem` 字段前增加 B 轴，并把变长 token 和身份轴补到批内最大长度。所有掩码均为 `bool`；`True` 表示相应补齐轴位置来自原样本，`False` 表示补零。

| 字段 | 数据类型与形状 | 含义、索引和空值 |
| --- | --- | --- |
| metadata | list[CandidateMetadata]，长度 B | 与全部批张量第一维逐候选对齐。 |
| voxel_input | 张量 (B,C,48,48,48) | 各样本 voxel_input 堆叠。 |
| foreground_target | bool (B,) | 前景二分类真值。 |
| ligand_area_target | bool (B,48,48,48) | 配体区域辅助真值。 |
| binding_area_target | bool (B,48,48,48) | 受体结合区域辅助真值。 |
| full_candidate_voxel_count | int64 (B,) | crop 前 V 体素数。 |
| candidate_manual_features | float32 (B,12) | 四模态手动统计。 |
| candidate_sampling_features | float32 (B,4) | 四项裁剪统计。 |
| a_tokens | float32 (B,N_A_max,D_A) | A token 前缀有效，其余补零。 |
| a_mask | bool (B,N_A_max) | 遮盖 a_tokens 第二维；True 为有效 A token。 |
| p_tokens | float32 (B,N_P_max,116) | P token 补齐结果。 |
| p_mask | bool (B,N_P_max) | 遮盖 p_tokens 第二维。 |
| v_tokens | float32 (B,N_V_max,68) | V token 补齐结果。 |
| v_mask | bool (B,N_V_max) | 同时遮盖 v_tokens 与 v_index_zyx 第二维。 |
| v_index_zyx | int64 (B,N_V_max,3) | V token 的 Z/Y/X 索引；补齐位置为 0，必须结合 v_mask 使用。 |
| pp_tokens | float32 (B,N_PP_max,4) | PP token 补齐结果。 |
| pp_mask | bool (B,N_PP_max) | 同时遮盖 pp_tokens 与 pp_index_zyx 第二维。 |
| pp_index_zyx | int64 (B,N_PP_max,3) | PP token 的 Z/Y/X 索引；补齐位置为 0。 |
| language_embeddings | float32 (B,I_max,768) | 每个样本有效身份写入前缀，其余补零。 |
| identity_mask | bool (B,I_max) | 遮盖语言向量和身份 logits 的身份轴；True 为真实身份槽。 |
| identity_target | int64 (B,) | 每个样本的正确身份槽，忽略值为 -1。 |
| identity_smiles | list[list[str]]，外层 B | 未补齐的字符串身份轴，内层顺序与语言向量有效前缀一致。 |
| graph_to_sample | int64 (G,) | 第 g 张图所属的批次样本下标，指向 B 轴。 |
| graph_to_identity | int64 (G,) | 第 g 张图在所属样本内的身份槽下标，指向 I 轴。 |
| graph_element | int64 (N,) | 连接节点轴上的原子序数；G=0 时长度 0。 |
| graph_charge | int64 (N,) | 与 graph_element 逐节点对齐的形式电荷。 |
| graph_atom_in_ring | float32 (N,4) | 与节点轴对齐的 3、4、5、6 元环标志。 |
| graph_node_graph_index | int64 (N,) | 每个节点所属图下标，数值指向 G 轴。 |
| graph_edge_index | int64 (2,E) | graph.edge_index + node_offsets, 全局地连接节点轴上的有向边端点；两行依次为源节点和目标节点，G=0 时 (2,0)。 |
| graph_edge_features | float32 (E,9) | 与 graph_edge_index 第二维逐边对齐；前五列是键型独热，后四列是环标志，G=0 时 (0,9)。 |

一个批次中若样本 0 有两个身份、样本 1 有一个身份，则图分支启用时可得到 `graph_to_sample=[0,0,1]` 和 `graph_to_identity=[0,1,0]`；这两个数组共同把三张图的表示写回 `(B,I_max)` 身份槽。

## 输入变体

| 变体 | 实际候选输入 | 身份图 |
| --- | --- | --- |
| strongest | 106 通道体素、A/P/V/PP、12 项模态统计和 4 项裁剪统计。 | 读取。 |
| blob_only | V/blob token、V 三项统计和 4 项裁剪统计；不读取密度、模拟密度、受体栅格或区域标签。 | 读取。 |
| density_blob_only | 1 通道实验密度、V、V 对应三项统计和 4 项裁剪统计。 | 读取。 |
| density_only | 1 通道实验密度；不读取 V/A/P/PP、模拟密度、受体栅格或人工统计。 | 读取。 |
| pocket_only | 原始 A_feat_L0 (N_A,50)；不读取 U-Net、V、P、PP。 | 读取。 |
| trunk_only | 候选侧与 strongest 相同。 | 不读取，连接图字段为空。 |

DataLoader 在 `num_workers>0` 时使用 `spawn`，但不跨遍历保留工作进程。训练引擎先冻结一个 epoch 的唯一物理批次计划，再切成三个连续区段；当前训练区段穷尽并关闭 worker 后才启动完整 validation，validation worker 退出后才进入下一训练区段。每个 worker 启动时完整载入一次图包；centered、受体、标签和语言向量按每个 worker 的字节预算由最近使用缓存复用。预算为 0 时不缓存，但不改变候选顺序或数组内容。
