from __future__ import annotations

import torch

from matcher.graph import EdgeBlock, GraphState, NodeBlock, NodeEdgeLayer, build_molecular_graph


def test_build_molecular_graph_merges_chemical_and_radius_edges() -> None:
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [6.0, 0.0, 0.0]])
    graph = build_molecular_graph(
        coordinates,
        torch.zeros(3, dtype=torch.long),
        torch.tensor([[0], [1]]),
        torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.bool),
        torch.zeros((1, 3), dtype=torch.bool),
        torch.zeros((1, 4), dtype=torch.bool),
    )

    pairs = {tuple(pair) for pair in graph.edge_index.T.tolist()}
    assert pairs == {(0, 1), (1, 0)}
    assert torch.all(graph.edge_input[:, 12] == 1)
    assert torch.all(graph.edge_input[:, 13] == 1)
    assert torch.all(graph.edge_class == 1)


def test_cross_group_chemical_edge_has_no_distance_feature() -> None:
    graph = build_molecular_graph(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        torch.tensor([0, 1]),
        torch.tensor([[0], [1]]),
        torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.bool),
        torch.zeros((1, 3), dtype=torch.bool),
        torch.zeros((1, 4), dtype=torch.bool),
    )

    assert torch.all(graph.edge_input[:, 12] == 1)
    assert torch.all(graph.edge_input[:, 13] == 0)
    assert torch.all(graph.edge_input[:, 14:] == 0)
    assert torch.all(graph.edge_class == 2)


def test_duplicate_chemical_pair_keeps_last_input_fields() -> None:
    bond_order = torch.zeros((2, 5), dtype=torch.bool)
    bond_order[0, 0] = True
    bond_order[1, 1] = True
    graph = build_molecular_graph(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        torch.zeros(2, dtype=torch.long),
        torch.tensor([[0, 1], [1, 0]]),
        bond_order,
        torch.zeros((2, 3), dtype=torch.bool),
        torch.zeros((2, 4), dtype=torch.bool),
    )

    assert torch.all(graph.edge_class == 2)


def test_radius_neighbor_limit_keeps_stable_source_order_for_ties() -> None:
    graph = build_molecular_graph(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        torch.zeros(4, dtype=torch.long),
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((0, 5), dtype=torch.bool),
        torch.empty((0, 3), dtype=torch.bool),
        torch.empty((0, 4), dtype=torch.bool),
        radius=1.1,
        max_radius_neighbors=2,
    )

    incoming = graph.edge_index[0, graph.edge_index[1] == 0]
    assert torch.equal(incoming, torch.tensor([1, 2]))


def test_node_edge_layer_supports_empty_edges() -> None:
    layer = NodeEdgeLayer(node_dim=8, edge_dim=4)
    state = GraphState(
        node=torch.randn(3, 8),
        edge=torch.empty(0, 4),
        edge_index=torch.empty(2, 0, dtype=torch.long),
    )
    result = layer(state)

    assert result.node.shape == (3, 8)
    assert result.edge.shape == (0, 4)


def test_node_block_matches_pocketxmol_gated_message_formula() -> None:
    torch.manual_seed(3)
    block = NodeBlock(node_dim=6, edge_dim=4)
    state = GraphState(
        node=torch.randn(3, 6),
        edge=torch.randn(3, 4),
        edge_index=torch.tensor([[0, 2, 1], [1, 1, 2]]),
    )

    source, target = state.edge_index
    source_node = state.node[source]
    message = block.message(block.edge(state.edge) * block.source(source_node))
    message = message * torch.sigmoid(block.gate(torch.cat((state.edge, source_node), -1)))
    aggregate = torch.zeros_like(state.node).index_add_(0, target, message)
    expected = block.output(torch.relu(block.norm(aggregate + block.self_projection(state.node))))

    assert torch.allclose(block(state), expected)


def test_edge_block_matches_pocketxmol_two_endpoint_formula() -> None:
    torch.manual_seed(5)
    block = EdgeBlock(node_dim=6, edge_dim=4)
    state = GraphState(
        node=torch.randn(3, 6),
        edge=torch.randn(4, 4),
        edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]]),
    )

    source, target = state.edge_index
    left_message = block.left_message(state.edge, state.node[source])
    right_message = block.right_message(state.edge, state.node[target])
    left_context = torch.zeros(3, 4).index_add_(0, target, left_message)[source]
    right_context = torch.zeros(3, 4).index_add_(0, source, right_message)[target]
    hidden = (
        left_context
        + right_context
        + block.left_node(state.node[source])
        + block.right_node(state.node[target])
        + block.self_projection(state.edge)
    )
    expected = block.output(torch.relu(block.norm(hidden)))

    assert torch.allclose(block(state), expected)
