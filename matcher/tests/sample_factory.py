"""Matcher 单元测试共用的小型内存样本。"""

from __future__ import annotations

import torch

from matcher.anchor_data import AnchorSample, CandidateEntity, LigandEntity, RawGraph
from matcher.graph import build_molecular_graph


def raw_graph(node_count: int, node_input_dim: int, *, binding: bool) -> RawGraph:
    coordinates = torch.arange(node_count, dtype=torch.float32)[:, None].repeat(1, 3)
    if node_count >= 2:
        chemical_edges = torch.tensor([[0], [1]])
        bond_order = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.bool)
    else:
        chemical_edges = torch.empty(2, 0, dtype=torch.long)
        bond_order = torch.empty(0, 5, dtype=torch.bool)
    graph = build_molecular_graph(
        coordinates,
        torch.zeros(node_count, dtype=torch.long),
        chemical_edges,
        bond_order,
        torch.zeros((chemical_edges.shape[1], 3), dtype=torch.bool),
        torch.zeros((chemical_edges.shape[1], 4), dtype=torch.bool),
    )
    return RawGraph(
        node_input=torch.randn(node_count, node_input_dim),
        coordinates=coordinates,
        graph=graph,
        element=torch.full((node_count,), 6, dtype=torch.long),
        binding_atom=torch.zeros(node_count, dtype=torch.bool) if binding else None,
    )


def sample() -> AnchorSample:
    ligand = LigandEntity("CCD:TST", raw_graph(3, 149, binding=False))
    candidates = tuple(
        CandidateEntity(
            center_xyz=torch.tensor([float(index) * 5, 0.0, 0.0]),
            density=torch.randn(1, 48, 48, 48),
            A_graph=raw_graph(index, 7, binding=True),
            A_global_index=torch.arange(index),
        )
        for index in (0, 2)
    )
    return AnchorSample(
        pdb_id="test",
        manifest_index=0,
        ligands=(ligand,),
        candidates=candidates,
        occurrence_to_ligand=torch.tensor([0, 0]),
        occurrence_candidate_id=torch.tensor([1, 2]),
        occurrence_centroid_xyz=torch.zeros(2, 3),
        ligand_gt_coordinates=None,
        ligand_gt_present=None,
        O_target=torch.tensor([[False, False], [True, True]]),
        O_prime_target=torch.full((2, 2), 0.5),
        O_prime_exact=torch.ones(2, 2, dtype=torch.bool),
    )
