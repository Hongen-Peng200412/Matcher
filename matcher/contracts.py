"""集中声明 Dataset、批次和配体图在模块边界传递的结构化字段。"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import torch

from .ligand_graph import LigandGraph


class CandidateMetadata(TypedDict):
    """标识一个冻结划分中的 Stage1 F1 候选及其匹配身份。"""

    split: str
    pdb_id: str
    candidate_index: int
    matched_object_key: str | None
    identity_seen_in_train: bool | None


class CandidateItem(TypedDict):
    """描述 ``CandidateDataset`` 返回的一个候选。

    ``metadata`` 是候选标识、匹配身份、划分与训练集可见性元数据。
    ``voxel_input`` 是 float16 ``(105,48,48,48)`` 候选级体素输入。
    ``a_tokens`` 是 float32 ``(N_A,229)`` 完整 A token。
    ``p_tokens`` 是 float32 ``(N_P,116)`` 完整 P token。
    ``v_tokens`` 是 float32 ``(N_V,52)`` 实际裁剪内 V token。
    ``v_index_zyx`` 是 int64 ``(N_V,3)``，与 V token 对齐且索引属于 ``[0,48)``。
    ``pp_tokens`` 是 float32 ``(N_PP,4)`` 实际裁剪内 PP token。
    ``pp_index_zyx`` 是 int64 ``(N_PP,3)``，与 PP token 对齐且索引属于 ``[0,48)``。
    ``a_tokens/p_tokens/v_tokens/pp_tokens`` 的共同末四维依次为 ``stage1_probability/relative_x/relative_y/relative_z``。
    ``stage1_probability`` 是位于 ``[0,1]`` 的无量纲 Stage1 固化概率。
    A/P 相对 XYZ 按完整 ``80³`` 物理半边长归一化，V/PP 相对 XYZ 按实际 ``48³`` 物理半边长归一化。
    ``class_target`` 是六分类整数下标。
    ``ligand_area_target`` 是 bool ``(48,48,48)`` 配体区域目标。
    ``binding_area_target`` 是 bool ``(48,48,48)`` 结合区域目标。
    ``full_candidate_voxel_count`` 是裁剪前完整权威 V 的体素数。
    ``candidate_manual_features`` 是 float32 ``(12,)``，顺序为 A/P/V/PP 各自的 ``token_count/probability_mean/probability_max``。
    ``candidate_manual_features`` 的计数单位为 token 个数，概率是无量纲 Stage1 固化原值，空模态三项均为零。
    ``candidate_sampling_features`` 是 float32 ``(4,)``，顺序为边界限制标志、L2 voxel 距离、L∞ voxel 距离和 ``[0,1]`` V 保留比例。
    ``identity_object_keys`` 是长度 ``I`` 的身份键列表。
    ``identity_class_index`` 是 int64 ``(I,)``，与身份轴对齐且有效值属于六分类下标 ``[1,5]``。
    ``language_embeddings`` 是 float32 ``(I,768)`` 身份语言表示。
    ``ligand_graphs`` 是长度 ``I`` 的配体图列表。
    ``identity_target`` 是当前候选真实身份在身份轴上的整数下标；``-1`` 表示当前语言模型下没有可监督目标。
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
    identity_class_index: np.ndarray
    language_embeddings: np.ndarray
    ligand_graphs: list[LigandGraph]
    identity_target: int


class CandidateBatch(TypedDict):
    """描述 ``collate_candidates`` 生成的固定物理批次。

    ``B`` 是物理批量；``N_A/N_P/N_V/N_PP`` 是四模态各自的批内最大 token 数。
    ``I`` 是批内最大身份数；``G/N/E`` 是连接后的配体图、节点和有向边总数。
    ``metadata`` 是长度 ``B`` 的候选元数据列表。
    ``voxel_input`` 是 float16 ``(B,105,48,48,48)`` 候选级体素输入。
    ``class_target`` 是 int64 ``(B,)`` 六分类目标。
    ``ligand_area_target`` 是 bool ``(B,48,48,48)`` 配体区域目标。
    ``binding_area_target`` 是 bool ``(B,48,48,48)`` 结合区域目标。
    ``full_candidate_voxel_count`` 是 int64 ``(B,)`` 裁剪前完整权威 V 体素数。
    ``candidate_manual_features`` 是 float32 ``(B,12)``，顺序为 A/P/V/PP 各自的 ``token_count/probability_mean/probability_max``。
    ``candidate_manual_features`` 的计数单位为 token 个数，概率是无量纲 Stage1 固化原值，空模态三项均为零。
    ``candidate_sampling_features`` 是 float32 ``(B,4)``，第二轴顺序和单位与 ``CandidateItem`` 相同。
    ``a_tokens`` 是 float32 ``(B,N_A,229)`` 补零 A token。
    ``a_mask`` 是 bool ``(B,N_A)`` A 有效掩码。
    ``p_tokens`` 是 float32 ``(B,N_P,116)`` 补零 P token。
    ``p_mask`` 是 bool ``(B,N_P)`` P 有效掩码。
    ``v_tokens`` 是 float32 ``(B,N_V,52)`` 补零 V token。
    ``v_mask`` 是 bool ``(B,N_V)`` V 有效掩码。
    ``v_index_zyx`` 是 int64 ``(B,N_V,3)`` V 局部索引；补零索引为零且仅在 ``v_mask`` 为真时可读。
    ``pp_tokens`` 是 float32 ``(B,N_PP,4)`` 补零 PP token。
    ``pp_mask`` 是 bool ``(B,N_PP)`` PP 有效掩码。
    ``pp_index_zyx`` 是 int64 ``(B,N_PP,3)`` PP 局部索引；补零索引为零且仅在 ``pp_mask`` 为真时可读。
    ``a_tokens/p_tokens/v_tokens/pp_tokens`` 的共同末四维和数值语义与 ``CandidateItem`` 相同。
    ``language_embeddings`` 是 float32 ``(B,I,768)`` 身份语言表示。
    ``identity_mask`` 是 bool ``(B,I)`` 身份有效掩码。
    ``identity_class_index`` 是 int64 ``(B,I)``，有效槽属于六分类下标 ``[1,5]``，补零槽为 ``-1``。
    ``identity_target`` 是 int64 ``(B,)``；``-1`` 表示没有可监督目标，否则指向类别等于 ``class_target`` 的有效身份槽。
    ``identity_object_keys`` 是 ``B`` 个长度 ``I_b`` 的身份键列表。
    ``graph_to_sample`` 是 int64 ``(G,)``，把每张图映射到样本下标。
    ``graph_to_identity`` 是 int64 ``(G,)``，把每张图映射到对应样本的身份下标。
    ``graph_element`` 是 int64 ``(N,)`` 节点元素编号。
    ``graph_charge`` 是 int64 ``(N,)`` 节点电荷编号。
    ``graph_residue_id`` 是 int64 ``(N,)`` 节点残基编号。
    ``graph_node_continuous`` 是 float32 ``(N,15)`` 节点连续特征。
    ``graph_node_graph_index`` 是 int64 ``(N,)``，把节点映射到 ``[0,G)``。
    ``graph_edge_index`` 是 int64 ``(2,E)`` 有向边索引。
    ``graph_edge_features`` 是 float32 ``(E,10)`` 有向边特征。
    每个有效身份恰好对应一张图；候选预训练时 ``G/N/E`` 均可为零。
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
    graph_to_sample: torch.Tensor
    graph_to_identity: torch.Tensor
    graph_element: torch.Tensor
    graph_charge: torch.Tensor
    graph_residue_id: torch.Tensor
    graph_node_continuous: torch.Tensor
    graph_node_graph_index: torch.Tensor
    graph_edge_index: torch.Tensor
    graph_edge_features: torch.Tensor


JsonObject = dict[str, Any]
