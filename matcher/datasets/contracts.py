"""声明候选 Dataset、批处理和 Matcher 模型之间的单一 SMILES 数据契约。"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch

from ..data.smiles_graph import SmilesGraph


class CandidateMetadata(TypedDict):
    """保存一个 Stage1 候选的稳定来源、排序分数和真值说明。

    - split: 清单划分名称。
    - pdb_id: 小写 PDB 标识。
    - centered_box_index: F2_centered.npz 内的稳定候选编号。
    - source_blob_index: F2_blobs.npz 内的源 blob 编号。
    - candidate_selected: Stage1 是否选择该候选。
    - gaussian_score: Gaussian 候选排序分数。
    - source_probability_mean: 源 blob 内 Stage1 概率均值。
    - foreground_target: 候选是否通过双向 0.30 覆盖门槛。
    - target_category: 真前景的五类结果分层标签；背景为 None。
    - matched_occurrence_id: 最大覆盖 occurrence 编号；背景为 None。
    - target_smiles: 目标 occurrence 的精确 PreparedLigand.smiles；背景或缺失时为 None。
    - identity_supported: 目标 SMILES 是否同时具有合法图和 SMI-TED 语言向量。
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
    """保存一个候选的未补齐输入、监督和 PDB 内闭集 SMILES 身份轴。

    - voxel_input: float16 数组，形状为 (C,48,48,48)；C 依变体为 106、1 或 0。
    - a_tokens: float32 数组；完整变体为 (N_A,230)，pocket_only 为 (N_A,50)。
    - p_tokens: float32 数组，形状为 (N_P,116) 或禁用时 (0,116)。
    - v_tokens: float32 数组，形状为 (N_V,68) 或禁用时 (0,68)。
    - v_index_zyx: int64 数组，形状为 (N_V,3)。
    - pp_tokens: float32 数组，形状为 (N_PP,4) 或禁用时 (0,4)。
    - pp_index_zyx: int64 数组，形状为 (N_PP,3)。
    - foreground_target: bool；候选前景二分类目标。
    - ligand_area_target: bool 数组，形状为 (48,48,48)。
    - binding_area_target: bool 数组，形状为 (48,48,48)。
    - candidate_manual_features: float32 数组，形状为 (12,)；禁用模态对应三项为零。
    - candidate_sampling_features: float32 数组，形状为 (4,)；pocket_only 全零。
    - identity_smiles: 当前 PDB 的受支持精确 SMILES，长度为 I。
    - language_embeddings: float32 数组，形状为 (I,768)。
    - smiles_graphs: 与身份轴逐项对齐的图；trunk_only 固定为空列表。
    - identity_target: 目标身份槽下标；背景或不具备受支持身份时为 -1。
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
    """保存补齐后的物理批次和按需连接的 SMILES 图。

    token 与体素字段的形状是在 CandidateItem 前增加批次轴 B 并补齐可变 token 轴。
    identity_mask 定义语言身份轴；trunk_only 的图连接张量全部为空。identity_target=-1
    同时表示背景和不具备受支持身份的真前景。
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
