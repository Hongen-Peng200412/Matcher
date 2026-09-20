"""声明候选 Dataset、批处理和 Matcher 模型之间的单一 SMILES 数据契约. """

from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch

from ..data.smiles_graph import SmilesGraph


class CandidateMetadata(TypedDict):
    """
    保存一个 Stage1 候选的稳定来源、排序分数和真值说明.

    - split: 清单划分名称.
    - pdb_id: 小写 PDB 标识.
    - centered_box_index: F2_centered.npz 内的稳定候选编号.
    - source_blob_index: F2_blobs.npz 内的源 blob 编号.
    - candidate_selected: Stage1 是否选择该候选.
    - gaussian_score: Gaussian 候选排序分数.
    - source_probability_mean: 源 blob 内 Stage1 概率均值.
    - foreground_target: 候选是否通过双向 0.30 覆盖门槛.
    - target_category: 真前景的五类结果分层标签; 背景为 None.
    - matched_occurrence_id: 最大覆盖 occurrence 编号; 背景为 None.
    - target_smiles: 目标 occurrence 的精确 PreparedLigand.smiles; 背景或缺失时为 None.
    - identity_supported: 目标 SMILES 是否同时具有合法图和 SMI-TED 语言向量.
    """
    split: str
    pdb_id: str
    centered_box_index: int
    source_blob_index: int
    candidate_selected: bool
    gaussian_score: float
    source_probability_mean: float
    foreground_target: bool
    target_category: str | None
    matched_occurrence_id: int | None
    target_smiles: str | None
    identity_supported: bool


class CandidateItem(TypedDict):
    """
    保存一个候选的未补齐输入、监督和 PDB 内闭集 SMILES 身份轴.

    - metadata: CandidateMetadata. 保存当前候选来源、两种排序分数、几何真值和身份支持状态.

    - voxel_input: float16 数组, (C,48,48,48). strongest/trunk_only 的 C=106, density_blob_only 的 C=1, blob_only/pocket_only 的 C=0.
    - a_tokens: float32 数组. strongest/trunk_only 为 (N_A,230), 依次连接 L0[0:50]、L1[50:114]、L2[114:178]、L3[178:226]、概率[226]、归一化相对 XYZ[227:230]. pocket_only 为原始 L0 的 (N_A,50). blob_only/density_blob_only 为 (0,230).
    - p_tokens: float32 数组, (N_P,116). strongest/trunk_only 依次连接 L2[0:64]、L3[64:112]、概率[112]、归一化相对 XYZ[113:116]. 其它变体为 (0,116).
    - v_tokens: float32 数组, (N_V,68). strongest/blob_only/density_blob_only/trunk_only 依次连接 64 维体素特征[0:64]、概率[64]、归一化相对 XYZ[65:68]. pocket_only 为 (0,68).
    - v_index_zyx: int64 数组, (N_V,3). 每个 V token 在 supplied 48 立方体中的 Z、Y、X 整数索引, 与 v_tokens 第一轴对齐.
    - pp_tokens: float32 数组, (N_PP,4). strongest/trunk_only 依次保存概率[0]和归一化相对 XYZ[1:4]. 其它变体为 (0,4).
    - pp_index_zyx: int64 数组, (N_PP,3). 每个 PP token 在 supplied 48 立方体中的 Z、Y、X 整数索引, 与 pp_tokens 第一轴对齐.

    - foreground_target: bool. 候选是否通过候选覆盖率和 occurrence 覆盖率均不低于 0.30 的前景目标.
    - ligand_area_target: bool 数组, (48,48,48). True 表示 supplied crop 中至少一个真实 occurrence 占据该体素.
    - binding_area_target: bool 数组, (48,48,48). True 表示 supplied crop 中至少一个 binding_atom 受体原子占据该体素.

    - full_candidate_voxel_count: int. 当前 source blob 在完整 F2 网格中的体素数, 用于计算 supplied crop 保留率.
    - candidate_manual_features: float32 数组, (12,). 按 A、P、V、PP 顺序, 每组依次为 token 数、概率均值、概率最大值. 禁用模态的三项均为零.
    - candidate_sampling_features: float32 数组, (4,). 依次为 crop_boundary_limited、crop_center_to_v_centroid_l2_voxels、crop_center_to_v_centroid_linf_voxels、v_retained_fraction. pocket_only 全零.

    - identity_smiles: list[str], 长度 I. 当前 PDB 内同时具有合法图和 SMI-TED 向量的精确 PreparedLigand.smiles, 按字符串升序且不重复.
    - language_embeddings: float32 数组, (I,768). 与 identity_smiles 逐槽对齐的 SMI-TED 表示.
    - smiles_graphs: list[SmilesGraph]. strongest/blob_only/density_blob_only/pocket_only 的长度为 I 并逐槽对齐, trunk_only 固定为空列表.
    - smiles_graphs[].element: int64 数组, (N_i,). 第 i 个身份按 RDKit 重原子顺序保存的原子序数.
    - smiles_graphs[].charge: int64 数组, (N_i,). 与 element 逐原子对齐的形式电荷.
    - smiles_graphs[].atom_in_ring: float32 数组, (N_i,4). 四列依次为 3、4、5、6 元环标志.
    - smiles_graphs[].edge_index: int64 数组, (2,E_i). 两行依次为有向边源原子和目标原子的图内下标.
    - smiles_graphs[].edge_features: float32 数组, (E_i,9). 与 edge_index 第二轴对齐, 前五列为键型独热编码, 后四列为 3、4、5、6 元环标志.
    - identity_target: int. 受支持真前景在 identity_smiles 中的槽下标. 背景或不受支持真前景为 -1.
    """
    metadata: CandidateMetadata
    voxel_input: np.ndarray
    a_tokens: np.ndarray
    p_tokens: np.ndarray
    v_tokens: np.ndarray
    v_index_zyx: np.ndarray
    pp_tokens: np.ndarray
    pp_index_zyx: np.ndarray
    foreground_target: bool
    ligand_area_target: np.ndarray
    binding_area_target: np.ndarray
    full_candidate_voxel_count: int
    candidate_manual_features: np.ndarray
    candidate_sampling_features: np.ndarray
    identity_smiles: list[str]
    language_embeddings: np.ndarray
    smiles_graphs: list[SmilesGraph]
    identity_target: int


class CandidateBatch(TypedDict):
    """
    保存补齐后的物理批次和按需连接的 SMILES 图.

    - metadata: list[CandidateMetadata], 长度 B. 与全部批次张量第一轴逐候选对齐.
    - voxel_input: 浮点张量, (B, C, 48, 48, 48). C 的变体规则与 CandidateItem 相同.
    - foreground_target: bool 张量, (B,). True 表示几何真前景, 与身份资产是否可用无关.
    - ligand_area_target: bool 张量, (B, 48, 48, 48). True 表示真实配体占据体素.
    - binding_area_target: bool 张量, (B, 48, 48, 48). True 表示结合受体原子占据体素.

    - full_candidate_voxel_count: int64 张量, (B,). 每个完整 source blob 的体素数.
    - candidate_manual_features: float32 张量, (B, 12). 字段顺序与 CandidateItem 相同.
    - candidate_sampling_features: float32 张量, (B, 4). 字段顺序与 CandidateItem 相同.
    - a_tokens: float32 张量, (B, N_A_max, F_A). F_A 为 230 或 pocket_only 的 50, 尾部补零.
    - a_mask: bool 张量, (B, N_A_max). True 表示 a_tokens 的真实 token, False 表示补零槽.
    - p_tokens: float32 张量, (B, N_P_max, 116). 尾部补零, 禁用变体的 N_P_max=0.
    - p_mask: bool 张量, (B, N_P_max). True 表示 p_tokens 的真实 token, False 表示补零槽.
    - v_tokens: float32 张量, (B, N_V_max, 68). 尾部补零, pocket_only 的 N_V_max=0.
    - v_mask: bool 张量, (B, N_V_max). True 表示 v_tokens 的真实 token, False 表示补零槽.
    - v_index_zyx: int64 张量, (B, N_V_max, 3). 与 v_tokens 对齐的 supplied crop Z、Y、X 索引, 补零槽为零.
    - pp_tokens: float32 张量, (B, N_PP_max, 4). 尾部补零, 禁用变体的 N_PP_max=0.
    - pp_mask: bool 张量, (B, N_PP_max). True 表示 pp_tokens 的真实 token, False 表示补零槽.
    - pp_index_zyx: int64 张量, (B, N_PP_max, 3). 与 pp_tokens 对齐的 supplied crop Z、Y、X 索引, 补零槽为零.

    - language_embeddings: float32 张量, (B, I_max, 768). 每个样本的 I 个有效身份写入前缀, 尾部补零.
    - identity_mask: bool 张量, (B, I_max). True 表示 language_embeddings 和 identity_smiles 的真实身份槽, False 表示补零槽.
    - identity_target: int64 张量, (B,). 受支持真前景指向有效身份槽, 背景或不受支持真前景为 -1.
    - identity_smiles: list[list[str]], 外层长度 B. 正确身份槽下标 0..I-1. 每个内层列表只含 I 个有效精确 SMILES, 不保存补零字符串.


    - graph_to_sample: int64 张量, (G,). 第 g 项把连接图 g 映射到批次样本下标 [0,B).
    - graph_to_identity: int64 张量, (G,). 第 g 项把连接图 g 映射到所属样本的身份槽 [0,I). 与 graph_to_sample 共同定位写回位置.

    - graph_element: int64 张量, (N,). G 张图连接后的原子序数. trunk_only 的 N=0.
    - graph_charge: int64 张量, (N,). 与 graph_element 逐原子对齐的形式电荷.
    - graph_atom_in_ring: float32 张量, (N,4). 与连接原子轴对齐的四种环大小标志.
    - graph_node_graph_index: int64 张量, (N,). 第 n 项把连接原子 n 映射到图下标 [0,G).
    - graph_edge_index: int64 张量, (2,E). 两行是连接原子轴上的有向边源端点和目标端点. trunk_only 的 E=0.
    - graph_edge_features: float32 张量, (E,9). 与 graph_edge_index 第二轴逐边对齐的键型和环标志.

    B 是物理批次大小. N_A、N_P、N_V、N_PP 是单样本真实 token 数. I 是单样本有效身份数.
    G、N、E 分别是连接后的图数、原子数和有向边数. trunk_only 的全部图张量为空, 但
    identity_mask 和语言身份轴仍保留. identity_target=-1 同时表示背景和不受支持真前景.
    """
    metadata: list[CandidateMetadata]
    voxel_input: torch.Tensor
    foreground_target: torch.Tensor
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
