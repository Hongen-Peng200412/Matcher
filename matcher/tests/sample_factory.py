"""Matcher 单元测试共用的小型内存样本。"""

from __future__ import annotations

import torch

from matcher.contracts import RawGraph
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
