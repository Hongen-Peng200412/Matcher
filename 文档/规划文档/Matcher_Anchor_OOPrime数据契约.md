# Matcher Anchor O/O′ 数据契约

## 文档职责

本文完整规定“不依赖 Stage1 推理产物”的 `AnchorPocketDataset` 输入、正式实验清单、内存对象、batch 边界和监督字段。读者不需要阅读运行日志或 Python 实现，就能核对每个字段的来源、形状、类型和不变量。

本文只适用于当前 `anchor_O_O_prime` 路线。未来使用 Stage1 推理产物时，口袋、候选、输入字段和后处理语义都可能改变，必须新增独立数据入口和独立契约，不能修改本文字段来兼容第二种路线。

## 正式输入

设一个 PDB 有 `N_occurrence` 个真实 occurrence、`N_identity` 个去重配体身份，本 epoch 或冻结验证清单给出 `N_candidate` 个 synthetic anchor 候选。

| 路径 | 必需字段 | 用途 |
|---|---|---|
| `stage1_preparation/box_pool/manifest.json` | `splits.train[]`、`splits.validation[]` | 生成 Matcher 正式实验清单的来源索引。Dataset 不直接读取它。 |
| `stage1_preparation/box_pool/config.json` | 完整 JSON | 只记录 SHA-256 来源身份，不参与运行时分支。 |
| `stage1_preparation/box_pool/{split}/{pdb_id}.npz` | `occurrence_id [N_occurrence]`、`bias_start_zyx [N_occurrence,N_bias,3]`、`context_start_zyx [N_context,3]` | 训练候选逐 epoch 抽样；验证只用其中的 occurrence 顺序，候选起点从正式清单读取。 |
| `parse/{pdb_id}/occurrences.jsonl` | `candidate_id`、`object_key` | 按 `occurrence_id` 排列真实 slot，并把 slot 映射到去重配体身份。 |
| `parse/{pdb_id}/receptor_tokens.npz` | `coords`、`feat`、`element`、`bond_index`、`bond_type` | 构造候选中心 18 Å 球形受体原子集合 A 及其图。 |
| `density/{pdb_id}/exp.npz` | `grid`、`origin`、`voxel_size` | 从原始 80³ 候选的中心裁出 48³ Map，并把体素起点转成世界坐标。 |
| `ligand_objects/{safe_object_key}.npz` | `atoms`、`bonds` | 构造每个去重配体身份的模板图。 |
| `parse/{pdb_id}/ligand_coords.npz` | `centroid_atom_{candidate_id}`；细监督开启时另需 `coords_*`、`present_*` | 构造 O/O′ 标签和可选 ligand↔A 原子级细监督。 |
| `labels/{pdb_id}/atom_labels.npz` | `binding_atom [N_receptor]` | `EntityAuxiliaryHead` 的 A 原子二值 binding 标签；关闭实体辅助监督时不读取。 |

坐标一律使用 XYZ 顺序、Å 单位。BOX 起点使用 ZYX 顺序。候选中心按下式计算：

```text
candidate_center_xyz
  = origin_xyz
  + (candidate_start_zyx[::-1] + 40) * voxel_size_xyz
```

## 正式实验清单

清单是正式 JSON，`schema_version=1` 且 `route="anchor_O_O_prime"`。生成入口为 `python -m matcher.anchor_manifest`。写入使用同目录临时文件和原子替换；正式文件不能放在临时目录。

顶层字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `schema_version` | int | 当前固定为 1。 |
| `route` | string | 当前固定为 `anchor_O_O_prime`。 |
| `source_box_manifest` / `source_box_manifest_sha256` | string | 来源 BOX manifest 的路径和内容身份。 |
| `source_box_config` / `source_box_config_sha256` | string | 来源 BOX config 的路径和内容身份。 |
| `source_box_metadata` | object | 来源 manifest 除 `splits` 外的原始元数据。 |
| `seed` | int | 小数据集选择和冻结验证候选的随机种子。 |
| `max_slots_per_pdb` | int | occurrence 数严格大于该值的 PDB 固化到排除清单；首版为 100。 |
| `pdb_sample_fraction` | float | PDB 级小数据集比例；1.0 表示不额外缩小。 |
| `sampling` | object | 候选抽样参数，见下表。Dataset 会逐项核对运行 YAML，任何不一致都立即失败。 |
| `splits.train` | array | 训练 PDB 条目。候选不冻结，每个 epoch 重新抽样。 |
| `splits.validation` | array | 验证 PDB 条目，必须含非空的冻结候选起点。 |
| `excluded` | object | 四个缺失 BOX PDB、超出 slot 上限的 PDB、冻结验证零候选 PDB。 |
| `counts` | object | train/validation 的 PDB 与 occurrence 总数。 |

`sampling` 必须包含：

```text
p_miss = 0.30
p_split = 0.20
p_hit = 0.50
max_context_ratio = 2.0
empty_A_context_skip_probability = 0.80
receptor_radius_angstrom = 18.0
```

train 条目字段为 `pdb_id: string`、`box_path: string`、`num_occurrences: int`。validation 条目在此基础上增加 `candidate_start_zyx: list[list[int,int,int]]`、`bias_count: int`、`context_requested: int`、`context_accepted: int`；`candidate_start_zyx` 不得为空。

## 候选抽样与零候选边界

每个真实 occurrence 独立执行一次互斥事件：miss 取 0 个 bias，split 取 2 个 bias，hit 取 1 个 bias。`p_miss+p_split+p_hit` 必须等于 1。center 不进入当前正式抽样。

context 请求数从闭区间 `0..floor(2.0*N_occurrence)` 离散均匀采样。候选中心 18 Å 内没有 A 原子时，以 80% 概率跳过，并继续遍历 context 池，直到满足请求数或池耗尽。候选来源只控制抽样，不进入模型字段，也不直接写死标签。

训练 Dataset 在每个 epoch 开始时一次性固定全部候选和一次 PDB 级 90° 立方体旋转。零候选 PDB 在这里从 `available_indices` 中删除；sampler 只收到保留的索引。此边界之后，`AnchorSample`、collate、模型、训练、验证和推理都要求 `N_candidate>=1`，不接受 `None` 或零候选特殊状态。

验证候选在清单生成时固定，验证和推理不旋转 Map。

## 内存字段

### 图字段

`RawGraph` 同时用于配体和 A：

| 字段 | 形状与类型 | 含义 |
|---|---|---|
| `node_input` | float32 `[N_node,F_node]` | 配体为 149 维 Emap2lig 字段：4 维 atom name、128 维元素 one-hot、1 维 charge、7 维 chirality、4 维 ring、1 维 residue_id、4 维元素属性。A 首版只读 49 维 `feat_l0`。 |
| `coordinates` | float32 `[N_node,3]` | XYZ Å。 |
| `graph.edge_index` | int64 `[2,N_edge]` | 有向边；第一行是消息来源，第二行是目标。 |
| `graph.edge_input` | float32 `[N_edge,31]` | 5 类键序、3 类键角色、4 类环、`is_chemical`、`is_radius`、`distance_valid`、16 维 0–4 Å RBF。 |
| `graph.edge_class` | int64 `[N_edge]` | 0 为纯 radius；1–5 为五类化学键，重合边按化学键类别编码。 |
| `element` | int64 `[N_node]` | 原子序数。 |
| `binding_atom` | bool `[N_node]` 或 `None` | A 的实体辅助标签；配体为 `None`。 |

配体与 A 都使用“化学键 ∪ 4 Å radius 边”，每个目标原子最多保留 48 条 radius 邻边，不含自环。两侧图字段和算法同构，但图编码器权重不共享。A 可以是合法的零节点图。

### 候选与 PDB 样本

`CandidateEntity`：

| 字段 | 形状与类型 |
|---|---|
| `center_xyz` | float32 `[3]` |
| `density` | float32 `[1,48,48,48]`，按 0.1%/99.9% 分位裁剪后逐候选标准化 |
| `A_graph` | `RawGraph`，只含中心 18 Å 内的受体原子 |
| `A_global_index` | int64 `[N_A]`，A 原子在原受体数组中的下标 |

`AnchorSample`：

| 字段 | 形状与类型 |
|---|---|
| `pdb_id` / `manifest_index` | string / int |
| `ligands` | 长度 `N_identity` 的去重 `LigandEntity` 元组 |
| `candidates` | 长度 `N_candidate>=1` 的 `CandidateEntity` 元组 |
| `occurrence_to_ligand` | int64 `[N_occurrence]`，slot 到 `ligands` 下标 |
| `occurrence_candidate_id` | int64 `[N_occurrence]`，上游真实 occurrence 标识 |
| `occurrence_centroid_xyz` | float32 `[N_occurrence,3]` |
| `ligand_gt_coordinates` | 细监督开启时为长度 `N_occurrence` 的坐标元组，否则 `None` |
| `ligand_gt_present` | 细监督开启时为长度 `N_occurrence` 的 bool 元组，否则 `None` |
| `O_target` | bool `[N_candidate,N_occurrence]` |
| `O_prime_target` | float32 `[N_candidate,N_occurrence]` |
| `O_prime_exact` | bool `[N_candidate,N_occurrence]` |

真实 occurrence 无论是否采到对应 bias 候选，都保留为输入 slot。真实质心和真实配体坐标只能生成标签，不能进入模型输入。

## O 与 O′ 标签

令 `d[i,o]` 为候选 `i` 中心到真实 occurrence `o` 质心的欧氏距离：

```text
O_target[i,o] = d[i,o] < 12 Å
O_prime_target[i,o] = 1 / (1 + d[i,o] / 6 Å)
O_prime_exact[i,o] = d[i,o] < 24 Å
```

`O_prime_exact=true` 时回归精确目标；24 Å 外不要求恢复不可观察的精确远距离，只施加 `O_prime<=0.2` 的单边截尾约束。A 的 18 Å 收集半径与 O 的 12 Å 正标签半径是两个独立概念。

## Batch 契约

`OccurrenceBudgetBatchSampler` 按清单中的 `num_occurrences` 贪心加入完整 PDB，默认预算为 64。加入下一个 PDB 会超过 64 时先结束当前 batch；单个 PDB 超过 64 时独占一个 batch，不拆分 PDB。

`MatcherBatch` 包含：

| 字段 | 形状与含义 |
|---|---|
| `samples` | 当前 batch 的完整 `AnchorSample` 元组 |
| `density` | float32 `[sum(N_candidate),1,48,48,48]` |
| `ligand_graphs` | 按 PDB、身份顺序展平的配体图 |
| `A_graphs` | 按 PDB、候选顺序展平的 A 图 |
| `candidate_ptr` | 长度 `N_PDB+1`，恢复每个 PDB 的候选区间 |
| `ligand_identity_ptr` | 长度 `N_PDB+1`，恢复每个 PDB 的配体身份区间 |

图编码器可以把多个互不连边图打包计算，但粗匹配、Hungarian、解码和指标始终按 PDB 切开。所有正式损失按每个 PDB 内的独立损失单元求和后，统一除以预期 occurrence 预算 64；不是除以当前 batch 的实际 occurrence 数。

## 当前路线输出边界

当前推理只输出 O/O′ 解码结果：每个非空 slot 的 `slot_index`、`candidate_index`、`O_probability`、`O_prime`、`decode_score`，以及被拒绝的 slot 下标。decoder 允许多个 slot 复用同一候选，不运行 Hungarian。checkpoint 保存并冻结验证选出的 O 概率阈值；独立验证评估使用该阈值，不重新扫描。

未来 A/B 逻辑必须使用新的 decoder/evaluator。未来 Stage1 推理数据必须使用新的 Dataset、清单与推理入口；两者都不能向本文当前路线追加模式分支。
