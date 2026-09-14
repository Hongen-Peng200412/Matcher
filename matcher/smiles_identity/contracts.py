"""声明单一 SMILES 身份管线在 Dataset、批处理和模型之间传递的字段. 

字符串只在 CPU 侧定位监督身份; 模型数值前向消费语言向量、SMILES 图和候选输入. 
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch

from .graph import SmilesGraph


class CandidateMetadata(TypedDict):
    """标识一个 Find_1 centered 候选及其真实匹配. 

    字段:
        - split: str, centered 产物划分名称, 如 validation. 
        - pdb_id: str, 小写 PDB 标识, 如 1abc. 
        - centered_box_index: int, centered NPZ 中候选的稳定编号; CandidateDataset 通过编号到第一轴位置的映射定位候选, 不要求编号等于列表位置. 
        - source_blob_index: int, 原 F2_blobs.npz 的 blob_index 编号; 用于连接 centered 候选与完整 blob 体素. 
        - stage1_selected: bool, Find_1 是否把该候选选为 Stage1 输出. 
        - matched_occurrence_id: int|None, 最佳真实 occurrence 编号; 没有 occurrence 通过双向覆盖门槛时为 None. 
        - matched_smiles: str|None, 正类候选的精确顶层 PreparedLigand.smiles; 背景候选为 None. 
    """

    split: str
    pdb_id: str
    centered_box_index: int
    source_blob_index: int
    stage1_selected: bool
    matched_occurrence_id: int | None
    matched_smiles: str | None


class CandidateItem(TypedDict):
    """保存一个候选的模型输入、监督和 PDB 内身份轴. 

    形状符号:
        - N_A: int, 当前候选完整 80 立方体中的 A token 数, 可为零. 
        - N_P: int, 当前候选完整 80 立方体中的 P token 数, 可为零. 
        - N_V: int, 当前 supplied 48 立方体内的 V token 数, 可为零. 
        - K: int, 当前候选的 PP token 数, 等于 min(pp_top_k, 48^3). 
        - I: int, 当前 PDB 的有效精确 SMILES 身份数, 可为零. 

    字段:
        - metadata: CandidateMetadata, 当前候选的划分、PDB、centered 编号、源 blob 编号和真实匹配字段. 
        - voxel_input: float16 数组, (106, 48, 48, 48); 第一轴依次为 56 个密度通道和 50 个受体通道, 空间轴依次为 Z、Y、X. 
        - a_tokens: float32 数组, (N_A, 230); 0:50、50:114、114:178、178:226 为 A_feat_L0/L1/L2/L3, 226 为 Stage1 概率, 227:230 为相对 X/Y/Z. 
        - p_tokens: float32 数组, (N_P, 116); 0:64、64:112 为 P_feat_L2/L3, 112 为 Stage1 概率, 113:116 为相对 X/Y/Z. 
        - v_tokens: float32 数组, (N_V, 68); 0:64 为 voxel_final, 64 为 Stage1 概率, 65:68 为相对 X/Y/Z. 
        - v_index_zyx: int64 数组, (N_V, 3); 与 v_tokens 第一轴逐项对齐, 最后一轴是 supplied 48 立方体内 Z/Y/X 下标. 
        - pp_tokens: float32 数组, (K, 4); 四个分量依次为 Stage1 概率和相对 X/Y/Z. 
        - pp_index_zyx: int64 数组, (K, 3); 与 pp_tokens 第一轴逐项对齐, 最后一轴是 supplied 48 立方体内 Z/Y/X 下标. 
        - class_target: int, 六分类下标; 0:6 依次为 background、ion、nucleotide_like、peptide_like、small_molecule、sugar. 
        - ligand_area_target: bool 数组, (48, 48, 48); 空间轴依次为 Z、Y、X, True 表示当前 PDB 任一 occurrence 占据该体素. 
        - binding_area_target: bool 数组, (48, 48, 48); 空间轴依次为 Z、Y、X, True 表示至少一个 binding_atom 占据该体素. 
        - full_candidate_voxel_count: int, 裁剪前完整 80 立方体候选中 source blob 的体素数. 
        - candidate_manual_features: float32 数组, (12,); 按 A、P、V、PP 分成四组, 每组依次为 token 数、Stage1 概率均值和最大值, 空 token 集合的三项为零. 
        - candidate_sampling_features: float32 数组, (4,); 依次为裁剪起点是否被边界限制、裁剪中心到 V 质心的 L2 体素距离、对应 L∞ 体素距离和 V 保留比例. 
        - identity_smiles: list[str], 长度 I; 每项是当前 PDB 一个身份槽的精确顶层 PreparedLigand.smiles. 
        - identity_class_index: int64 数组, (I,); 与 identity_smiles 逐槽对齐, 数值属于前景下标 [1, 6). 
        - language_embeddings: float32 数组, (I, 768); 与 identity_smiles 逐槽对齐的固定 SMI-TED 表示. 
        - smiles_graphs: list[SmilesGraph], 长度 I; 与 identity_smiles 逐槽对齐, 每张图含 element、charge、atom_in_ring、edge_index 和 edge_features. 
        - identity_target: int, -1 表示背景, 非负值指向 [0, I) 内与 matched_smiles 一致的唯一身份槽. 

    坐标:
        - a_tokens[:, 227:230]: 相对 X/Y/Z, 以 supplied crop 中心为原点并除以完整 80 立方体半边长 40 体素. 
        - p_tokens[:, 113:116]: 相对 X/Y/Z, 以 supplied crop 中心为原点并除以完整 80 立方体半边长 40 体素. 
        - v_tokens[:, 65:68]: 相对 X/Y/Z, 以 supplied crop 中心为原点并除以局部 48 立方体半边长 24 体素. 
        - pp_tokens[:, 1:4]: 相对 X/Y/Z, 以 supplied crop 中心为原点并除以局部 48 立方体半边长 24 体素. 
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
    identity_smiles: list[str]
    identity_class_index: np.ndarray
    language_embeddings: np.ndarray
    smiles_graphs: list[SmilesGraph]
    identity_target: int


class CandidateBatch(TypedDict):
    """保存补齐后的物理批次和连接后的 SMILES 图. 

    形状符号:
        - B: int, 当前物理批次的候选数. 
        - N_A_max: int, 批内最大 A token 数. 
        - N_P_max: int, 批内最大 P token 数. 
        - N_V_max: int, 批内最大 V token 数. 
        - N_PP_max: int, 批内最大 PP token 数. 
        - I_max: int, 批内单个 PDB 的最大有效身份数. 
        - G: int, 连接后的有效身份图数, 等于每个候选身份数之和. 
        - N_node: int, G 张图连接后的总原子数. 
        - N_edge: int, G 张图连接后的有向边总数. 

    字段:
        - metadata: list[CandidateMetadata], 长度 B; 按物理批次顺序保留未张量化的候选标识和真实匹配. 
        - voxel_input: 浮点张量, (B, 106, 48, 48, 48); 候选体素输入, 空间轴依次为 Z、Y、X. 
        - class_target: int64 张量, (B,); 六分类下标. 
        - ligand_area_target: bool 张量, (B, 48, 48, 48); 逐候选配体区域目标. 
        - binding_area_target: bool 张量, (B, 48, 48, 48); 逐候选受体结合区域目标. 
        - full_candidate_voxel_count: int64 张量, (B,); 每个候选裁剪前的 source blob 体素数. 
        - candidate_manual_features: float32 张量, (B, 12); 逐候选 A、P、V、PP 数量与概率统计. 
        - candidate_sampling_features: float32 张量, (B, 4); 逐候选裁剪边界、中心偏移和 V 保留率统计. 
        - a_tokens: float32 张量, (B, N_A_max, 230); 每个候选的 A token 沿第二轴补零. 
        - a_mask: bool 张量, (B, N_A_max); True 表示 a_tokens 第二轴的真实 token, False 表示补零. 
        - p_tokens: float32 张量, (B, N_P_max, 116); 每个候选的 P token 沿第二轴补零. 
        - p_mask: bool 张量, (B, N_P_max); True 表示 p_tokens 第二轴的真实 token, False 表示补零. 
        - v_tokens: float32 张量, (B, N_V_max, 68); 每个候选的 V token 沿第二轴补零. 
        - v_mask: bool 张量, (B, N_V_max); True 表示 v_tokens 和 v_index_zyx 第二轴的真实 token, False 表示补零. 
        - v_index_zyx: int64 张量, (B, N_V_max, 3); 与 v_tokens 前两轴对齐的 supplied crop 内 Z/Y/X 下标. 
        - pp_tokens: float32 张量, (B, N_PP_max, 4); 每个候选的 PP token 沿第二轴补零. 
        - pp_mask: bool 张量, (B, N_PP_max); True 表示 pp_tokens 和 pp_index_zyx 第二轴的真实 token, False 表示补零. 
        - pp_index_zyx: int64 张量, (B, N_PP_max, 3); 与 pp_tokens 前两轴对齐的 supplied crop 内 Z/Y/X 下标. 
        - language_embeddings: float32 张量, (B, I_max, 768); 每个候选所属 PDB 的身份语言表示, 无效槽补零. 
        - identity_mask: bool 张量, (B, I_max); True 表示真实精确 SMILES 身份槽, False 表示补零槽. 
        - identity_class_index: int64 张量, (B, I_max); 有效槽为 [1, 6) 前景类别下标, 补零槽为 -1. 
        - identity_target: int64 张量, (B,); 背景为 -1, 前景指向对应候选在第二轴上的有效身份槽. 
        - identity_smiles: list[list[str]], 外层长度 B; 每个内层列表只保存当前 PDB 的有效槽, 不为 I_max 补字符串. 
        - graph_to_sample: int64 张量, (G,); 每张连接图所属的候选下标, 取值属于 [0, B). 
        - graph_to_identity: int64 张量, (G,); 每张连接图在所属候选中的身份槽下标, 取值属于 [0, I_max). 
        - graph_element: int64 张量, (N_node,); 连接后的原子序数. 
        - graph_charge: int64 张量, (N_node,); 与 graph_element 逐原子对齐的形式电荷. 
        - graph_atom_in_ring: float32 张量, (N_node, 4); 与 graph_element 逐原子对齐, 四列依次标记 3、4、5、6 元环. 
        - graph_node_graph_index: int64 张量, (N_node,); 每个连接原子所属的图下标, 取值属于 [0, G). 
        - graph_edge_index: int64 张量, (2, N_edge); 第一轴依次为有向键的源原子下标和目标原子下标. 
        - graph_edge_features: float32 张量, (N_edge, 9); 与 graph_edge_index 第二轴逐边对齐, 前五列为键型独热编码, 后四列为 3、4、5、6 元环标志. 
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
    identity_smiles: list[list[str]]
    graph_to_sample: torch.Tensor
    graph_to_identity: torch.Tensor
    graph_element: torch.Tensor
    graph_charge: torch.Tensor
    graph_atom_in_ring: torch.Tensor
    graph_node_graph_index: torch.Tensor
    graph_edge_index: torch.Tensor
    graph_edge_features: torch.Tensor
