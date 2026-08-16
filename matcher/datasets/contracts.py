"""集中声明单候选、物理批次和配体图连接结果的结构化字段。

CandidateDataset 产生 CandidateItem，collate_candidates 把多个 CandidateItem 转为
CandidateBatch，模型和评估代码只通过这些具名字段读取数据。字段声明固定数组形状、
轴顺序、补零掩码和联合身份对齐关系，不负责再次验证来源文件。
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import torch

from ..data.ligand_graph import LigandGraph
from ..data.schema import PreparedSmilesStatus


class CandidateMetadata(TypedDict):
    """标识冻结划分中的一个 Stage1 F1 候选及其匹配身份。

    字段：
        - split: 字符串；候选所属的物化划分名称，例如 train 或 validation。
        - pdb_id: 非空小写字符串；候选所属 PDB，candidate_index 只在该 PDB 内解释。
        - candidate_index: 整数；当前 PDB 的 Stage1 F1 候选编号，从零开始并用于切分 PDB 级数组。
        - matched_object_key: 字符串或 None；有 occurrence 通过覆盖阈值时保留其 LigandObject 模板键，包括 type_tag=other 映射成 background 的情况；没有 occurrence 通过阈值时为 None。
        - matched_prepared_smiles: 字符串或 None；仅五类前景命中的顶层 PreparedLigand.smiles 可为非空字符串；type_tag=other 背景、未命中或字符串不可用时为 None。
        - matched_prepared_smiles_status: PreparedSmilesStatus；available、explicit_null、missing_file、not_applicable 依次表示顶层字符串可用、源字段明确为 null、PreparedLigand 文件缺失、背景候选不适用。
        - identity_seen_in_train: 布尔值或 None；已选择语言家族且候选具备完整前景 object_key+prepared_smiles 时为 train 可见性布尔值；未选择语言家族、背景、未命中或缺少完整前景联合身份时为 None；布尔值可跨 PDB 共享联合键，但不合并 PDB 内身份轴。
    """

    split: str
    pdb_id: str
    candidate_index: int
    matched_object_key: str | None
    matched_prepared_smiles: str | None
    matched_prepared_smiles_status: PreparedSmilesStatus
    identity_seen_in_train: bool | None


class CandidateItem(TypedDict):
    """描述 CandidateDataset 返回的一个候选。

    维度符号：
        - N_A: 当前候选完整 80³ Stage1 框中的 A token 数，可为零。
        - N_P: 当前候选完整 80³ Stage1 框中的 P token 数，可为零。
        - N_V: 当前 48³ 裁剪内保留的 V token 数，可为零。
        - K: 当前 48³ 裁剪内的固定 PP token 数，等于 min(pp_top_k, 48³) 且大于零。
        - I: 当前 PDB 中满足身份范围、语言表示可用且 LigandObject 文件存在的联合身份数；候选预训练时为零。

    字段：
        - metadata: CandidateMetadata；候选标识、命中 occurrence 的联合身份字段和训练集可见性。
        - metadata.split: 字符串；当前物化划分名称。
        - metadata.pdb_id: 非空小写字符串；当前候选所属 PDB。
        - metadata.candidate_index: 整数；当前 PDB 内从零开始的 Stage1 F1 候选编号。
        - metadata.matched_object_key: 字符串或 None；有 occurrence 通过覆盖阈值时保留模板键，type_tag=other 映射成 background 时也可非空；未命中时为 None。
        - metadata.matched_prepared_smiles: 字符串或 None；仅五类前景命中的顶层 PreparedLigand.smiles 可为非空字符串；type_tag=other 背景和未命中均为 None。
        - metadata.matched_prepared_smiles_status: PreparedSmilesStatus；说明顶层字符串可用、明确为 null、来源文件缺失或背景不适用。
        - metadata.identity_seen_in_train: 布尔值或 None；已选择语言家族且候选具备完整前景联合身份时为可见性布尔值；未选择语言家族、背景、未命中或缺少 object_key/prepared_smiles 时为 None。
        - voxel_input: float16 数组，(105, 48, 48, 48)；第一轴是 56 个候选密度通道和 49 个受体通道，后三轴依次为 Z、Y、X。
        - a_tokens: float32 数组，(N_A, 229)；0:49、49:113、113:177、177:225 为 L0/L1/L2/L3，225 为 Stage1 概率，226:229 为相对 X/Y/Z。
        - p_tokens: float32 数组，(N_P, 116)；0:64、64:112 为 L2/L3，112 为 Stage1 概率，113:116 为相对 X/Y/Z。
        - v_tokens: float32 数组，(N_V, 52)；0:48 为 voxel_final，48 为 Stage1 概率，49:52 为相对 X/Y/Z。
        - v_index_zyx: int64 数组，(N_V, 3)；与 v_tokens 第一轴逐项对齐，最后一轴依次为 48³ 裁剪内 Z/Y/X 且数值位于 [0, 48)。
        - pp_tokens: float32 数组，(K, 4)；四个特征分量依次为 Stage1 概率和相对 X/Y/Z。
        - pp_index_zyx: int64 数组，(K, 3)；与 pp_tokens 第一轴逐项对齐，最后一轴依次为 48³ 裁剪内 Z/Y/X 且数值位于 [0, 48)。
        - class_target: 整数；六分类下标，0:5 依次为 background、ion、nucleotide_like、peptide_like、small_molecule、sugar。
        - ligand_area_target: bool 数组，(48, 48, 48)；后三轴依次为 Z/Y/X，True 表示当前 PDB 全部 occurrence 稀疏掩码在候选裁剪内的并集体素。
        - binding_area_target: bool 数组，(48, 48, 48)；后三轴依次为 Z/Y/X，True 表示至少一个完整受体表中的 binding_atom 落入该体素。
        - full_candidate_voxel_count: 整数；裁剪前完整 80³ 候选中权威 V 的体素数，用于解释裁剪保留比例。
        - candidate_manual_features: float32 数组，(12,)；按 A、P、V、PP 顺序分别保存 token 数、Stage1 概率均值和 Stage1 概率最大值，空模态对应三项均为零。
        - candidate_manual_features[0:3]: float32 数组，(3,)；A 的 token 数、无量纲概率均值、无量纲概率最大值。
        - candidate_manual_features[3:6]: float32 数组，(3,)；P 的 token 数、无量纲概率均值、无量纲概率最大值。
        - candidate_manual_features[6:9]: float32 数组，(3,)；V 的 token 数、无量纲概率均值、无量纲概率最大值。
        - candidate_manual_features[9:12]: float32 数组，(3,)；PP 的 token 数、无量纲概率均值、无量纲概率最大值。
        - candidate_sampling_features: float32 数组，(4,)；依次为请求裁剪起点是否有任一轴被边界限制、裁剪中心到 V 质心的 L2 体素距离、对应 L∞ 体素距离、位于 48³ 裁剪内的 V 保留比例。
        - identity_object_keys: 长度 I 的字符串列表；每个值定位一份 LigandObject 模板图。
        - identity_prepared_smiles: 长度 I 的非空字符串列表；逐槽保存顶层 PreparedLigand.smiles，并与 identity_object_keys 共同定义联合身份。
        - identity_class_index: int64 数组，(I,)；与两个身份字符串列表逐槽对齐，值属于前景六分类下标 [1, 5]。
        - language_embeddings: float32 数组，(I, 768)；当前语言家族为每个联合身份选定的 occurrence 语言表示。
        - ligand_graphs: 长度 I 的 LigandGraph 列表；每个身份槽对应一张由 object_key 定位的模板图，图缓存预算允许且写入成功时，相同 object_key 的不同 smiles 槽可引用同一个实例。
        - ligand_graphs[*].element: int64 数组，(N_atom,)，模板原子序数。
        - ligand_graphs[*].charge: int64 数组，(N_atom,)，与 element 逐原子对齐的形式电荷。
        - ligand_graphs[*].residue_id: int64 数组，(N_atom,)，模板内部残基编号。
        - ligand_graphs[*].node_continuous: float32 数组，(N_atom, 15)，依次为手性 7 维、3/4/5/6 元环 4 维、模板 is_present 1 维和组内中心化后除以 10 Å 的参考 X/Y/Z。
        - ligand_graphs[*].edge_index: int64 数组，(2, 2*N_bond)，第一轴依次为源端点和目标端点，先全部原始方向再全部反向。
        - ligand_graphs[*].edge_features: float32 数组，(2*N_bond, 10)，依次为键型 5 维、3/4/5/6 元环 4 维和参考距离除以 10 Å；跨 residue_id 边的距离分量为零。
        - identity_target: 整数；非负值指向类别、object_key 和顶层 smiles 都与当前候选真值一致的身份槽；-1 表示真值字段不完整或当前可用身份轴不含该联合身份，type_tag=other 背景即使保留 object_key 也固定为 -1。

    坐标：
        - a_tokens[:, 226:229]: 相对 X/Y/Z，按完整 80³ 框的 40 体素物理半边长逐轴归一化。
        - p_tokens[:, 113:116]: 相对 X/Y/Z，按完整 80³ 框的 40 体素物理半边长逐轴归一化。
        - v_tokens[:, 49:52]: 相对 X/Y/Z，按实际 48³ 裁剪的 24 体素物理半边长逐轴归一化。
        - pp_tokens[:, 1:4]: 相对 X/Y/Z，按实际 48³ 裁剪的 24 体素物理半边长逐轴归一化。
    """

    metadata: CandidateMetadata
    voxel_input: np.ndarray
    a_tokens: np.ndarray
    p_tokens: np.ndarray
    v_tokens: np.ndarray
    v_index_zyx: np.ndarray
    pp_tokens: np.ndarray
    pp_index_zyx: np.ndarray
    class_target: int
    ligand_area_target: np.ndarray
    binding_area_target: np.ndarray
    full_candidate_voxel_count: int
    candidate_manual_features: np.ndarray
    candidate_sampling_features: np.ndarray
    identity_object_keys: list[str]
    identity_prepared_smiles: list[str]
    identity_class_index: np.ndarray
    language_embeddings: np.ndarray
    ligand_graphs: list[LigandGraph]
    identity_target: int


class CandidateBatch(TypedDict):
    """描述 collate_candidates 生成的固定物理批次。

    维度符号：
        - B: 物理批次中的候选数，必须大于零。
        - N_A_max: 批内最大的 A token 数；每个样本的有效长度由 a_mask 给出。
        - N_P_max: 批内最大的 P token 数；每个样本的有效长度由 p_mask 给出。
        - N_V_max: 批内最大的 V token 数；每个样本的有效长度由 v_mask 给出。
        - K: 每个正式 CandidateItem 的固定 PP token 数，等于 min(pp_top_k, 48³)。
        - I_max: 批内最大的联合身份数，可为零；每个样本的有效长度由 identity_mask 给出。
        - G: 批内有效身份对应的配体图总数，等于各样本有效身份数之和。
        - N_node: G 张图连接后的节点总数。
        - N_directed_edge: G 张图连接后的有向边总数。

    字段：
        - metadata: 长度 B 的 CandidateMetadata 列表；保持候选物化时的顺序且不补齐。
        - metadata[*].split: 字符串；对应候选的物化划分名称。
        - metadata[*].pdb_id: 非空小写字符串；对应候选所属 PDB。
        - metadata[*].candidate_index: 整数；对应 PDB 内的 Stage1 F1 候选编号。
        - metadata[*].matched_object_key: 字符串或 None；对应通过覆盖阈值的 occurrence 模板键，type_tag=other 背景也可非空；未命中时为 None。
        - metadata[*].matched_prepared_smiles: 字符串或 None；仅五类前景命中的顶层 PreparedLigand.smiles 可非空；type_tag=other 背景和未命中为 None。
        - metadata[*].matched_prepared_smiles_status: PreparedSmilesStatus；说明顶层字符串可用性。
        - metadata[*].identity_seen_in_train: 布尔值或 None；具备完整前景联合身份时是当前语言家族的 train 可见性；背景、未命中、联合键不完整或未选语言家族时为 None。
        - voxel_input: float16 张量，(B, 105, 48, 48, 48)；第二轴为通道，后三轴依次为 Z、Y、X。
        - class_target: int64 张量，(B,)；六分类下标。
        - ligand_area_target: bool 张量，(B, 48, 48, 48)；各候选的 occurrence 配体区域并集。
        - binding_area_target: bool 张量，(B, 48, 48, 48)；各候选的受体结合原子区域。
        - full_candidate_voxel_count: int64 张量，(B,)；每个候选裁剪前的完整权威 V 体素数。
        - candidate_manual_features: float32 张量，(B, 12)；第二轴按 A、P、V、PP 分成四组三元组，每组依次为 token 数、概率均值、概率最大值。
        - candidate_sampling_features: float32 张量，(B, 4)；第二轴依次为边界限制标志、L2 体素距离、L∞ 体素距离和 V 保留比例。
        - a_tokens: float32 张量，(B, N_A_max, 229)；不足部分补零，特征顺序与 CandidateItem.a_tokens 相同。
        - a_mask: bool 张量，(B, N_A_max)；True 表示对应 a_tokens 位置来自样本，False 表示补零位置。
        - p_tokens: float32 张量，(B, N_P_max, 116)；不足部分补零，特征顺序与 CandidateItem.p_tokens 相同。
        - p_mask: bool 张量，(B, N_P_max)；True 表示对应 p_tokens 位置来自样本，False 表示补零位置。
        - v_tokens: float32 张量，(B, N_V_max, 52)；不足部分补零，特征顺序与 CandidateItem.v_tokens 相同。
        - v_mask: bool 张量，(B, N_V_max)；True 表示对应 v_tokens 和 v_index_zyx 位置来自样本，False 表示补零位置。
        - v_index_zyx: int64 张量，(B, N_V_max, 3)；最后一轴为裁剪内 Z/Y/X，补零位置为 0 且只有 v_mask 为 True 的位置可读。
        - pp_tokens: float32 张量，(B, K, 4)；正式 Dataset 的 K 在同一 loader 内固定，最后一轴依次为 Stage1 概率和相对 X/Y/Z。
        - pp_mask: bool 张量，(B, K)；正式 CandidateItem 的全部 K 个位置均为 True；通用补齐函数仍按实际输入长度构造掩码。
        - pp_index_zyx: int64 张量，(B, K, 3)；最后一轴为裁剪内 Z/Y/X，只有 pp_mask 为 True 的位置可读。
        - language_embeddings: float32 张量，(B, I_max, 768)；每个有效身份的语言表示，补零身份槽为零。
        - identity_mask: bool 张量，(B, I_max)；True 表示当前样本实际存在该身份槽，False 表示批次补零槽。
        - identity_class_index: int64 张量，(B, I_max)；有效槽为前景类别下标 [1, 5]，补零槽为 -1。
        - identity_target: int64 张量，(B,)；非负值指向同一样本的有效身份槽，-1 表示没有可监督身份真值。
        - identity_object_keys: 长度 B 的嵌套字符串列表；第 b 项长度 I_b，不补到 I_max，并与第 b 项有效身份槽逐项对齐。
        - identity_prepared_smiles: 长度 B 的嵌套非空字符串列表；第 b 项长度 I_b，不补到 I_max，并与 identity_object_keys 共同定义联合身份。
        - graph_to_sample: int64 张量，(G,)；第 g 项把连接前第 g 张图映射到样本下标 [0, B)。
        - graph_to_identity: int64 张量，(G,)；第 g 项把连接前第 g 张图映射到对应样本的身份下标 [0, I_b)。
        - graph_element: int64 张量，(N_node,)；所有图连接后的模板原子序数。
        - graph_charge: int64 张量，(N_node,)；与 graph_element 逐节点对齐的形式电荷。
        - graph_residue_id: int64 张量，(N_node,)；与 graph_element 逐节点对齐的模板内部残基编号。
        - graph_node_continuous: float32 张量，(N_node, 15)；手性 7 维、环标志 4 维、模板 is_present 1 维和组内中心化参考 X/Y/Z 3 维。
        - graph_node_graph_index: int64 张量，(N_node,)；把每个连接后节点映射到图下标 [0, G)。
        - graph_edge_index: int64 张量，(2, N_directed_edge)；端点已经加上各图节点偏移，第一轴依次为源节点和目标节点。
        - graph_edge_features: float32 张量，(N_directed_edge, 10)；与 graph_edge_index 第二轴逐边对齐的键型、环标志和参考距离特征。

    空身份边界：
        - graph_element: 没有有效身份时为 int64 (0,) 空张量。
        - graph_charge: 没有有效身份时为 int64 (0,) 空张量。
        - graph_residue_id: 没有有效身份时为 int64 (0,) 空张量。
        - graph_node_continuous: 没有有效身份时为 float32 (0, 15) 空张量。
        - graph_node_graph_index: 没有有效身份时为 int64 (0,) 空张量。
        - graph_edge_index: 没有有效身份时为 int64 (2, 0) 空张量。
        - graph_edge_features: 没有有效身份时为 float32 (0, 10) 空张量。
    """

    metadata: list[CandidateMetadata]
    voxel_input: torch.Tensor
    class_target: torch.Tensor
    ligand_area_target: torch.Tensor
    binding_area_target: torch.Tensor
    full_candidate_voxel_count: torch.Tensor
    candidate_manual_features: torch.Tensor
    candidate_sampling_features: torch.Tensor
    a_tokens: torch.Tensor
    a_mask: torch.Tensor
    p_tokens: torch.Tensor
    p_mask: torch.Tensor
    v_tokens: torch.Tensor
    v_mask: torch.Tensor
    v_index_zyx: torch.Tensor
    pp_tokens: torch.Tensor
    pp_mask: torch.Tensor
    pp_index_zyx: torch.Tensor
    language_embeddings: torch.Tensor
    identity_mask: torch.Tensor
    identity_class_index: torch.Tensor
    identity_target: torch.Tensor
    identity_object_keys: list[list[str]]
    identity_prepared_smiles: list[list[str]]
    graph_to_sample: torch.Tensor
    graph_to_identity: torch.Tensor
    graph_element: torch.Tensor
    graph_charge: torch.Tensor
    graph_residue_id: torch.Tensor
    graph_node_continuous: torch.Tensor
    graph_node_graph_index: torch.Tensor
    graph_edge_index: torch.Tensor
    graph_edge_features: torch.Tensor


# 无固定 schema 的 JSON 对象别名仅用于需要保留动态键的边界。
JsonObject = dict[str, Any]
