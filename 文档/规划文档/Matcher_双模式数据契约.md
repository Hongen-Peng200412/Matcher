# Matcher 双模式数据契约

本文规定 `matcher_v2` 两个 Dataset 返回的共同内存结构和各自的数据来源。盘上 A–G 与 Stage1 字段仍分别由 `Data_Preprocessing/Ori_Data/README.md` 和 `文档/讨论/BOX-level数据契约.md` 管理；本文不复制或改写生产方契约。

## 共同样本

一个样本对应一个 PDB，包含该 PDB 的全部真实 occurrence、按 `object_key` 去重的配体模板和一个或多个候选。候选数记为 $N_c$，occurrence 数记为 $N_o$，去重配体身份数记为 $N_l$。

共同字段包括：

| 字段 | 类型与形状 | 含义 |
| --- | --- | --- |
| `pdb_id` | `str` | 小写 PDB 身份 |
| `ligands` | 长度 $N_l$ 的配体图序列 | 去重配体模板；节点和边字段来自 `ligand_objects/{safe_object_key}.npz` |
| `occurrence_to_ligand` | `int64 (N_o,)` | 每个 occurrence 对应的 `ligands` 索引 |
| `occurrence_candidate_id` | `int64 (N_o,)` | A–G `occurrences.jsonl.candidate_id` |
| `occurrence_centroid_xyz` | `float32 (N_o,3)` | 配体实际存在重原子的世界 XYZ 几何中心，单位 Å |
| `candidates` | 长度 $N_c$ 的候选序列 | 每项包含中心、48³ 密度、A 图、候选 mask 及可选 Stage1 专属实体 |
| `A_target`、`B_target`、`O_target` | `float32 (N_c,N_o)` | 三个连续监督矩阵，定义见实施计划 |
| `A_prime_target`、`B_prime_target`、`O_prime_target` | `bool (N_c,N_o)` | 10%、10%、10 Å 的三个硬监督矩阵 |

## `ground_truth_context`

模式一的正式清单只冻结 PDB 划分和可读取 occurrence，不保存每个 epoch 的候选。PDB 成员沿用当前新版 Stage1-Find 的正式 train、validation 与 calibration 划分；不重新随机划分。

每个 occurrence 产生一个候选：

- 候选中心为 `ligand_coords.npz.centroid_atom_{cid}`。
- 48³ 实验密度从 `density/{pdb_id}/exp.npz.grid` 按候选中心裁剪；越出完整图的部分用零填充，坐标中心不因边界移动。
- 候选 mask 为 `density/{pdb_id}/ligand_area.npz.mask_{cid}`。
- A 包含到该 occurrence 任一 `present=True` 真实配体重原子不大于 10 Å 的受体重原子。GT ligand-area mask 只用于 A/B 覆盖监督，不参与扩大 A 的物理范围。A 图保留这些原子间化学边，并补充半径边。
- 训练时同一 PDB 的全部候选密度执行同一个 90° 立方体旋转；验证、推理与评估不旋转。当前图网络只消费距离等旋转不变量，世界坐标不随密度改写。以后启用空间采样点时，点的局部坐标必须与密度执行同一旋转。
- 不生成 miss、split、bias 或 context 候选；PDB 没有 occurrence 时在清单生成边界排除。

模式一为未来 U-Net 采样点保留内存边界，但当前正式版本不启用点生成头；本次 checkpoint 只使用密度摘要。以后启用的点仍由 `matcher_v2/context.py` 组装，不写入模式一清单。

## `stage1_context`

模式二使用单独的正式 JSON 指针。运维工具扫描指定 Stage1 推理根目录，把已存在且具有对应 `_COMPLETE` 的产物路径写入指针；Dataset 只读取该 JSON，不在训练时扫描目录，也不做文件哈希校验。

每个 Stage1 F1-centered 候选读取：

- 候选身份、80³ BOX 几何和 blob 成员体素；
- V：`voxel_index_local_zyx`、`centered_probability` 与同序 `voxel_final`；
- P：Find 来源归档中的 `P_coord_local_xyz`、`P_probability`、`P_feat_L2` 与 `P_feat_L3`；
- A：Find 来源归档中的受体原子身份、坐标、概率和 L0–L3 可用特征；
- 密度 BOX：根据 centered 几何从 A–G `exp.npz` 裁出中心 48³；
- PP：从 Stage1 完整图概率在当前 48³ 范围内确定 top-k 坐标，并从 U-Net 多尺度特征采样初始表示；PP 不作为独立盘上字段保存。

Stage1 不同模型来源可能具有不同的 `voxel_final`、P 和 A 特征宽度。宽度由配置声明并在文件读取边界与实际非空数组核对，不在 Dataset 或模型中写死。

模式二候选与真实 occurrence 的 A/B 由预测 blob mask 与 GT ligand-area mask 的真实交集计算；O/O′ 由候选中心与真实 occurrence 几何中心距离计算。没有候选的 PDB 在训练样本收集边界跳过，batch 和模型不增加零候选特殊逻辑；端到端推理清单另行累计这些 PDB 的全部 occurrence，并在评估中把它们计为漏检。

## 批次与隔离

批次按预期 occurrence 预算装箱。若单个 PDB 的 occurrence 数超过预算，该 PDB 单独构成一个 batch；否则按确定的 PDB 顺序追加，直到再加入一个 PDB 会超过预算。损失先在每个 PDB 内计算，batch 中各 PDB 的损失相加后除以配置中的预期 occurrence 预算。

所有跨候选和跨 slot 的注意力、匈牙利匹配、解码与评估严格限制在同一个 PDB 内。密度 U-Net 可以把多个候选沿 batch 维一起计算，不能让注意力跨 PDB 混合。

## 推理与评估隔离

`ground_truth_context` 和 `stage1_context` 分别拥有显式推理数据入口和候选结果构造。二者复用模型前向输出、阈值搜索、checkpoint 选择和指标累计的纯计算，不建立通用数据插件或动态注册系统。

当前 A/B/O 推理根据六个输出产生候选—slot 分数，再在一个 PDB 内解码。评估至少报告阈值扫描后的 precision、recall、F1 与 PR-AUC，并把验证选出的阈值冻结进 checkpoint；独立评估只使用冻结阈值，不重新扫描。
