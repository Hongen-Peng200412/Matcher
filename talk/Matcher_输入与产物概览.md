# Matcher 输入与产物概览

## 当前实现读取什么

当前 `AnchorPocketDataset` 只读取已有真实标注和预定位候选池，不读取 Stage1 模型预测：

- `stage1_preparation/box_pool/manifest.json`：train/validation 的 PDB 与 BOX 文件入口。
- 每个 PDB 的 BOX NPZ：真实 occurrence 的 center、每个 occurrence 的 30 个 bias 起点和 PDB 级 context 起点。
- `parse/{pdb_id}/occurrences.jsonl`：occurrence、配体身份和配体对象关系。
- `parse/{pdb_id}/ligand_coords.npz`：真实沉积配体坐标、present mask 和 occurrence 质心；真实坐标只生成监督标签，不作为模型输入。
- `ligand_objects/*.npz`：配体模板原子、化学键和参考构象。
- `receptor_tokens.npz`：受体原子世界坐标、49 维输入特征、元素与化学键。
- 实验密度 NPZ：完整 ZYX 网格、XYZ voxel size 和世界坐标原点。
- Stage D `atom_labels.npz`：受体 `binding_atom` 辅助监督。

候选的 80³ 起点使用 ZYX 索引保存。世界坐标中心通过 `origin_xyz + start_xyz * voxel_size_xyz + 40 * voxel_size_xyz` 计算。模型读取中心 48³，因此密度裁剪起点为 80³ 起点的每轴加 16；裁剪后按自身 0.1%/99.9% 分位截断并 z-score。

## 一个训练样本包含什么

一个样本对应一个完整 PDB，而不是单个候选或单个 occurrence。样本保留全部真实 occurrence slot，并包含本轮随机生成的非空候选集合。模型输入包括：

- 每个 occurrence 的配体模板图和配体身份分组；
- 每个候选中心 18 Å 内的 A 原子图，允许为空；
- 每个候选的单通道中心 48³ Map；
- 候选中心、occurrence 质心和由二者距离生成的 O/O′ 标签；
- 仅在辅助监督开启时需要的元素、六类边、binding atom 和 4 Å 接触边缘标签来源字段。

训练候选逐 epoch 重采样；验证候选关系只生成一次并正式落盘。零候选 PDB 在样本清单边界剔除，模型、解码和评估不再维护零候选分支。

## 当前正式输出

训练产物按 Phase1/Phase2 独立运行目录保存：最终配置、训练摘要、按全验证集 F1 选择的 checkpoint、该 checkpoint 对应的 O 概率阈值和简洁损失/指标日志。当前推理结果只属于 O/O′ 路线，保存：

- `selected_pairs`：`slot_index/candidate_index/O_probability/O_prime/decode_score`；
- `unmatched_slot_indices`：低于阈值而拒绝匹配的 slot；
- occurrence 级 precision、recall、F1 和冻结阈值。

同一候选允许被多个 slot 复用。当前结果不声明已完成 A/B 的 merge/split 解释。

## 与未来 Stage1 路线的边界

未来 Stage1 推理产物可能改变候选、口袋、P/V/A/PP 字段和后处理语义。它必须新增独立的数据入口、清单生成和推理/评估入口；当前 Anchor 数据文件不导入未来入口，模型也不读取候选来源或数据模式标记。双方只在字段含义确实相同时复用底层张量和图算子。
