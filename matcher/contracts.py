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

    A/P/V/PP 分别为 ``(N,229)/(N,116)/(N,52)/(N,4)`` float32；V/PP 的
    ``index_zyx`` 为对应 token 在中心 ``48³`` 中的整数索引。体素输入是
    ``(105,48,48,48)`` float16；两个区域目标是同空间形状的 bool 数组。
    身份语言表示为 ``(I,768)``，配体图和 object_key 均与身份轴一一对应；
    ``identity_target=-1`` 表示当前语言模型下没有可监督的真实身份。
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
    identity_object_keys: list[str]
    language_embeddings: np.ndarray
    ligand_graphs: list[LigandGraph]
    identity_target: int


class CandidateBatch(TypedDict):
    """描述 ``collate_candidates`` 生成的固定物理批次。

    令 B 为物理批量，N_A/N_P/N_V/N_PP 为四模态各自的批内最大 token 数，
    I 为批内最大身份数，G/N/E 为连接后的配体图、节点和有向边总数：

    ``voxel_input`` 是 float16 ``(B,105,48,48,48)``。``class_target`` 是
    int64 ``(B,)``。``full_candidate_voxel_count`` 是 int64 ``(B,)``。
    ``ligand_area_target`` 和 ``binding_area_target`` 都是 bool
    ``(B,48,48,48)``。

    ``a_tokens/p_tokens/v_tokens/pp_tokens`` 分别是 float32
    ``(B,N_A,229)/(B,N_P,116)/(B,N_V,52)/(B,N_PP,4)``。对应的
    ``a_mask/p_mask/v_mask/pp_mask`` 是 bool ``(B,N_*)``，补零 token 的掩码
    为假。``v_index_zyx`` 和 ``pp_index_zyx`` 分别是 int64
    ``(B,N_V,3)/(B,N_PP,3)``；补零索引为零，只有对应掩码为真时可读取。

    ``language_embeddings`` 是 float32 ``(B,I,768)``。``identity_mask`` 是
    bool ``(B,I)``。``identity_target`` 是 int64 ``(B,)``；``-1`` 表示当前
    语言模型没有可监督目标，否则是该样本身份轴上的有效下标。

    ``graph_element/graph_charge/graph_residue_id`` 是 int64 ``(N,)``。
    ``graph_node_continuous`` 是 float32 ``(N,15)``。
    ``graph_node_graph_index`` 是 int64 ``(N,)``，把节点映射到 ``[0,G)``。
    ``graph_edge_index`` 是 int64 ``(2,E)``。``graph_edge_features`` 是
    float32 ``(E,10)``。``graph_to_sample`` 和 ``graph_to_identity`` 都是
    int64 ``(G,)``，把每张图映射到样本下标和该样本的身份下标。每个有效
    身份恰好对应一张图；候选预训练时 G/N/E 均可为零。
    """

    metadata: list[CandidateMetadata]
    voxel_input: torch.Tensor
    class_target: torch.Tensor
    ligand_area_target: torch.Tensor
    binding_area_target: torch.Tensor
    full_candidate_voxel_count: torch.Tensor
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
