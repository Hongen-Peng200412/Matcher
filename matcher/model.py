"""Matcher 的图状态、四类粗表示、两阶段 Block 和 O/O′ 预测。

本文件把训练主干保留在一条直接调用链中。数据读取、Hungarian/损失和解码分别位于
``anchor_data.py``、``objectives.py`` 与 ``oo_prime.py``，避免模型读取盘上身份或阈值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

from .anchor_data import AnchorSample, RawGraph
from .batching import MatcherBatch
from .graph import GraphState, NodeEdgeLayer
from .map import MapBackbone, MapSummaryHead


@dataclass(frozen=True)
class PackedEntities:
    """同类实体的打包原始图。

    节点相关张量的第一维均为打包后的原子轴 ``N``，边相关张量的第一维为有向边轴
    ``E``；``edge_index`` 是 ``int64 [2,E]``。``node_ptr`` 与 ``edge_ptr``
    长度均为实体数加一，用相邻指针切回每个配体身份或候选 A。
    """

    node_input: Tensor
    coordinates: Tensor
    edge_input: Tensor
    edge_index: Tensor
    edge_class: Tensor
    node_ptr: tuple[int, ...]
    edge_ptr: tuple[int, ...]
    element: Tensor
    binding_atom: Tensor | None


@dataclass(frozen=True)
class CoarseState:
    """一个 PDB 内四组持久粗表示，末维均为 ``repr_dim``。

    ``CCD_repr [S,D]`` 的 ``S`` 是完整真实 occurrence 槽位数；其余三项均为
    ``[C,D]``，``C`` 是当前 synthetic-anchor 候选数。
    """

    CCD_repr: Tensor
    BOX_repr: Tensor
    A_repr: Tensor
    Map_repr: Tensor


@dataclass(frozen=True)
class HeadOutput:
    """一个 Block 出口。

    ``coarse_pair_repr`` 为 ``[C,S,D]``；``O_logit``、``O_prime_logit`` 与
    ``O_prime`` 均为 ``[C,S]``。
    """

    coarse_pair_repr: Tensor
    O_logit: Tensor
    O_prime_logit: Tensor
    O_prime: Tensor


@dataclass(frozen=True)
class EntityAuxiliaryOutput:
    """实体状态重建输出。

    ``element_logit [N,128]``、``edge_class_logit [E,6]``；A 另有
    ``binding_logit [N]``，配体该字段为 ``None``。
    """

    element_logit: Tensor
    edge_class_logit: Tensor
    binding_logit: Tensor | None


@dataclass(frozen=True)
class PDBMatcherOutput:
    """一个 PDB 的全部当前阶段监督出口。

    ``block_outputs`` 长度为 ``num_blocks``；Phase1 全部有效，Phase2 的前
    ``phase1_blocks`` 项为 ``None``。实体状态与相应 ``PackedEntities`` 的节点、边顺序一致。
    FinePair 损失为 ``[C,S_pred,S_gt]``；最后一轴在 Hungarian 完成后按
    ``predicted_slot -> true_occurrence`` 选择，避免同身份槽位编号泄漏。
    """

    block_outputs: tuple[HeadOutput | None, ...]
    ligand_auxiliary: EntityAuxiliaryOutput | None
    A_auxiliary: EntityAuxiliaryOutput | None
    ligand_state: GraphState
    A_state: GraphState
    ligand_entities: PackedEntities
    A_entities: PackedEntities
    fine_recall_pair_loss: Tensor | None = None
    fine_precision_pair_loss: Tensor | None = None


def _pack_raw_graphs(graphs: tuple[RawGraph, ...], device: torch.device) -> PackedEntities:
    node_input = []
    coordinates = []
    edge_input = []
    edge_index = []
    edge_class = []
    element = []
    binding = []
    node_ptr = [0]
    edge_ptr = [0]
    for raw_graph in graphs:
        node_input.append(raw_graph.node_input)
        coordinates.append(raw_graph.coordinates)
        edge_input.append(raw_graph.graph.edge_input)
        edge_index.append(raw_graph.graph.edge_index + node_ptr[-1])
        edge_class.append(raw_graph.graph.edge_class)
        element.append(raw_graph.element)
        if raw_graph.binding_atom is not None:
            binding.append(raw_graph.binding_atom)
        node_ptr.append(node_ptr[-1] + raw_graph.node_input.shape[0])
        edge_ptr.append(edge_ptr[-1] + raw_graph.graph.edge_input.shape[0])
    return PackedEntities(
        node_input=torch.cat(node_input).to(device),
        coordinates=torch.cat(coordinates).to(device),
        edge_input=torch.cat(edge_input).to(device),
        edge_index=torch.cat(edge_index, dim=1).to(device),
        edge_class=torch.cat(edge_class).to(device),
        node_ptr=tuple(node_ptr),
        edge_ptr=tuple(edge_ptr),
        element=torch.cat(element).to(device),
        binding_atom=torch.cat(binding).to(device) if binding else None,
    )


def _slice_packed_entities(
    packed: PackedEntities,
    state: GraphState,
    entity_start: int,
    entity_end: int,
) -> tuple[PackedEntities, GraphState]:
    """按实体指针从全 batch 互不连边图中取回一个 PDB，并把下标归零。"""

    node_start, node_end = packed.node_ptr[entity_start], packed.node_ptr[entity_end]
    edge_start, edge_end = packed.edge_ptr[entity_start], packed.edge_ptr[entity_end]
    node_ptr = tuple(value - node_start for value in packed.node_ptr[entity_start : entity_end + 1])
    edge_ptr = tuple(value - edge_start for value in packed.edge_ptr[entity_start : entity_end + 1])
    local = PackedEntities(
        packed.node_input[node_start:node_end],
        packed.coordinates[node_start:node_end],
        packed.edge_input[edge_start:edge_end],
        packed.edge_index[:, edge_start:edge_end] - node_start,
        packed.edge_class[edge_start:edge_end],
        node_ptr,
        edge_ptr,
        packed.element[node_start:node_end],
        packed.binding_atom[node_start:node_end] if packed.binding_atom is not None else None,
    )
    local_state = GraphState(
        state.node[node_start:node_end],
        state.edge[edge_start:edge_end],
        local.edge_index,
    )
    return local, local_state


def _pad_packed_values(values: Tensor, ptr: tuple[int, ...]) -> tuple[Tensor, Tensor]:
    """把 ``[N,...]`` 的连续实体补成 ``[E,W,...]``，并返回 ``[E,W]`` 掩码。

    ``ptr`` 给出 ``E`` 个实体在首维中的起止位置；掩码中的 ``True`` 表示真实元素。
    空实体仍保留一个全零占位位置，供后续批量算子维持合法形状。
    """

    ptr_tensor = torch.tensor(ptr, device=values.device)
    lengths = ptr_tensor[1:] - ptr_tensor[:-1]
    padded_width = max(int(lengths.max().item()), 1)
    entity_index = torch.repeat_interleave(
        torch.arange(len(lengths), device=values.device), lengths
    )
    local_index = torch.arange(len(values), device=values.device) - torch.repeat_interleave(
        ptr_tensor[:-1], lengths
    )
    padded = values.new_zeros((len(lengths), padded_width, *values.shape[1:]))
    padded[entity_index, local_index] = values
    mask = torch.arange(padded_width, device=values.device)[None] < lengths[:, None]
    return padded, mask


class TypedAttention(nn.Module):
    """按语义类型使用独立 K/V 投影的 scaled-dot-product attention。

    该算子只计算 attention 增量，不隐藏 LayerNorm、残差、dropout 或 FFN。二维输入
    表示单个集合；三维输入表示每个 batch 成员各有一组独立的 query/context。
    ``query_mask/context_mask [B,N]`` 的 True 表示真实 token；CUDA BF16/FP16 的
    单 context、无 bias 调用会先移除 padding，再执行同一注意力的 varlen Flash 内核。
    """

    def __init__(
        self,
        query_dim: int,
        context_dims: tuple[int, ...],
        output_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.query = nn.Linear(query_dim, output_dim, bias=False)
        self.keys = nn.ModuleList(
            nn.Linear(context_dim, output_dim, bias=False) for context_dim in context_dims
        )
        self.values = nn.ModuleList(
            nn.Linear(context_dim, output_dim, bias=False) for context_dim in context_dims
        )
        self.output = nn.Linear(output_dim, output_dim)

    def forward(
        self,
        query: Tensor,
        contexts: tuple[Tensor, ...],
        attention_bias: Tensor | None = None,
        query_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        squeeze_batch = query.ndim == 2
        if squeeze_batch:
            query = query.unsqueeze(0)
            contexts = tuple(context.unsqueeze(0) for context in contexts)
        batch, query_count, _ = query.shape
        can_try_varlen_flash = (
            flash_attn_varlen_func is not None
            and query.is_cuda
            and query_mask is not None
            and context_mask is not None
            and attention_bias is None
            and len(contexts) == 1
            and self.head_dim % 8 == 0
            and (
                query.dtype in (torch.float16, torch.bfloat16)
                or torch.is_autocast_enabled()
            )
        )
        if can_try_varlen_flash:
            projected_query = self.query(query[query_mask]).reshape(
                -1, self.num_heads, self.head_dim
            )
            projected_key = self.keys[0](contexts[0][context_mask]).reshape(
                -1, self.num_heads, self.head_dim
            )
            projected_value = self.values[0](contexts[0][context_mask]).reshape(
                -1, self.num_heads, self.head_dim
            )
            if projected_query.dtype in (torch.float16, torch.bfloat16):
                query_lengths = query_mask.sum(dim=1, dtype=torch.int32)
                context_lengths = context_mask.sum(dim=1, dtype=torch.int32)
                attended = flash_attn_varlen_func(
                    projected_query.contiguous(),
                    projected_key.contiguous(),
                    projected_value.contiguous(),
                    F.pad(
                        query_lengths.cumsum(dim=0, dtype=torch.int32), (1, 0)
                    ).contiguous(),
                    F.pad(
                        context_lengths.cumsum(dim=0, dtype=torch.int32), (1, 0)
                    ).contiguous(),
                    query.shape[1],
                    contexts[0].shape[1],
                    dropout_p=0.0,
                    causal=False,
                )
                attended = self.output(attended.flatten(1))
                result = attended.new_zeros(
                    (batch, query_count, self.num_heads * self.head_dim)
                )
                result[query_mask] = attended
                return result.squeeze(0) if squeeze_batch else result

        projected_query = self.query(query).reshape(
            batch, query_count, self.num_heads, self.head_dim
        )
        keys = []
        values = []
        for context, key, value in zip(contexts, self.keys, self.values, strict=True):
            keys.append(
                key(context).reshape(batch, context.shape[1], self.num_heads, self.head_dim)
            )
            values.append(
                value(context).reshape(batch, context.shape[1], self.num_heads, self.head_dim)
            )
        key_tensor = torch.cat(keys, dim=1)
        value_tensor = torch.cat(values, dim=1)
        if attention_bias is not None and attention_bias.ndim == 3:
            attention_bias = attention_bias.unsqueeze(0)
        if context_mask is not None:
            padding_mask = context_mask[:, None, None, :]
            if attention_bias is None:
                attention_bias = padding_mask
            elif attention_bias.dtype == torch.bool:
                attention_bias = attention_bias & padding_mask
            else:
                attention_bias = attention_bias.masked_fill(~padding_mask, -torch.inf)
        attended = F.scaled_dot_product_attention(
            projected_query.transpose(1, 2),
            key_tensor.transpose(1, 2),
            value_tensor.transpose(1, 2),
            attn_mask=attention_bias,
        ).transpose(1, 2)
        attended = attended.reshape(batch, query_count, -1)
        result = self.output(attended)
        if query_mask is not None:
            result = result * query_mask[:, :, None]
        return result.squeeze(0) if squeeze_batch else result


class LearnedQueryReadout(nn.Module):
    """用单个 learned query 初始化一个实体的 256 维持久表示。"""

    def __init__(self, node_dim: int, repr_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, repr_dim) / math.sqrt(repr_dim))
        self.node_norm = nn.LayerNorm(node_dim)
        self.attention = TypedAttention(repr_dim, (node_dim,), repr_dim, num_heads)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(repr_dim)
        self.ffn = nn.Sequential(
            nn.Linear(repr_dim, 2 * repr_dim),
            nn.SiLU(),
            nn.Linear(2 * repr_dim, repr_dim),
            nn.Dropout(dropout),
        )

    def forward(self, nodes: Tensor) -> Tensor:
        value = self.query + self.attention_dropout(
            self.attention(self.query, (self.node_norm(nodes),))
        )
        return (value + self.ffn(self.ffn_norm(value)))[0]


class RepresentationReadback(nn.Module):
    """让持久 `_repr` 查询对应未塌缩实体，再执行一次残差 FFN。"""

    def __init__(self, node_dim: int, repr_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(repr_dim)
        self.node_norm = nn.LayerNorm(node_dim)
        self.attention = TypedAttention(repr_dim, (node_dim,), repr_dim, num_heads)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(repr_dim)
        self.ffn = nn.Sequential(
            nn.Linear(repr_dim, 2 * repr_dim),
            nn.SiLU(),
            nn.Linear(2 * repr_dim, repr_dim),
            nn.Dropout(dropout),
        )

    def forward(self, representation: Tensor, nodes: Tensor) -> Tensor:
        value = representation + self.attention_dropout(
            self.attention(self.query_norm(representation), (self.node_norm(nodes),))
        )
        return value + self.ffn(self.ffn_norm(value))


class CandidateDistanceBias(nn.Module):
    """把候选中心距离编码为逐 attention head 的可学习 logit bias。"""

    def __init__(
        self,
        num_heads: int,
        *,
        rbf_bins: int = 32,
        hidden_dim: int = 64,
        distance_scale: float = 12.0,
    ) -> None:
        super().__init__()
        self.rbf_bins = rbf_bins
        self.distance_scale = distance_scale
        self.mlp = nn.Sequential(nn.Linear(rbf_bins, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, num_heads))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, centers_xyz: Tensor) -> Tensor:
        distance = torch.cdist(centers_xyz, centers_xyz)
        bounded = distance / (distance + self.distance_scale)
        centers = torch.linspace(0, 1, self.rbf_bins, device=distance.device, dtype=distance.dtype)
        sigma = 1.0 / (self.rbf_bins - 1)
        rbf = torch.exp(-0.5 * ((bounded[..., None] - centers) / sigma) ** 2)
        return self.mlp(rbf).permute(2, 0, 1)


class CoarseLayer(nn.Module):
    """一次跨类交互、同类交互和每状态一套 FFN。"""

    _STATE_NAMES = ("CCD", "BOX", "A", "Map")

    def __init__(
        self,
        repr_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        mode: Literal["coupled_only", "coupled_plus_modal"] = "coupled_plus_modal",
        candidate_bias_rbf_bins: int = 32,
        candidate_bias_hidden_dim: int = 64,
        candidate_bias_distance_scale: float = 12.0,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.cross_norm = nn.ModuleDict(
            {name: nn.LayerNorm(repr_dim) for name in self._STATE_NAMES}
        )
        self.self_norm = nn.ModuleDict(
            {name: nn.LayerNorm(repr_dim) for name in self._STATE_NAMES}
        )
        self.ffn_norm = nn.ModuleDict(
            {name: nn.LayerNorm(repr_dim) for name in self._STATE_NAMES}
        )
        self.cross_CCD = TypedAttention(
            repr_dim,
            (repr_dim,) if mode == "coupled_only" else (repr_dim, repr_dim, repr_dim),
            repr_dim,
            num_heads,
        )
        self.cross_candidate = nn.ModuleDict(
            {
                name: TypedAttention(repr_dim, (repr_dim,), repr_dim, num_heads)
                for name in ("BOX", "A", "Map")
            }
        )
        self.self_attention = nn.ModuleDict(
            {
                name: TypedAttention(repr_dim, (repr_dim,), repr_dim, num_heads)
                for name in self._STATE_NAMES
            }
        )
        self.distance_bias = nn.ModuleDict(
            {
                name: CandidateDistanceBias(
                    num_heads,
                    rbf_bins=candidate_bias_rbf_bins,
                    hidden_dim=candidate_bias_hidden_dim,
                    distance_scale=candidate_bias_distance_scale,
                )
                for name in ("BOX", "A", "Map")
            }
        )
        self.ffn = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(repr_dim, ffn_dim),
                    nn.SiLU(),
                    nn.Linear(ffn_dim, repr_dim),
                )
                for name in self._STATE_NAMES
            }
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, state: CoarseState, candidate_centers: Tensor) -> CoarseState:
        names = ("CCD", "BOX") if self.mode == "coupled_only" else self._STATE_NAMES
        old = {
            "CCD": state.CCD_repr,
            "BOX": state.BOX_repr,
            "A": state.A_repr,
            "Map": state.Map_repr,
        }
        normalized_old = {name: self.cross_norm[name](value) for name, value in old.items()}
        candidate_contexts = (normalized_old["BOX"],)
        if self.mode == "coupled_plus_modal":
            candidate_contexts += (
                normalized_old["A"],
                normalized_old["Map"],
            )
        cross = dict(old)
        cross["CCD"] = old["CCD"] + self.dropout(
            self.cross_CCD(normalized_old["CCD"], candidate_contexts)
        )
        for name in names[1:]:
            cross[name] = old[name] + self.dropout(
                self.cross_candidate[name](
                    normalized_old[name],
                    (normalized_old["CCD"],),
                )
            )

        updated = dict(cross)
        normalized_cross = {name: self.self_norm[name](cross[name]) for name in names}
        for name in names:
            bias = None if name == "CCD" else self.distance_bias[name](candidate_centers)
            updated[name] = cross[name] + self.dropout(
                self.self_attention[name](
                    normalized_cross[name],
                    (normalized_cross[name],),
                    bias,
                )
            )
            updated[name] = updated[name] + self.dropout(
                self.ffn[name](self.ffn_norm[name](updated[name]))
            )
        return CoarseState(updated["CCD"], updated["BOX"], updated["A"], updated["Map"])


class MatcherBlock(nn.Module):
    """一层可选图更新、配置数量的 coarse 运算、实体回吸和单向 BOX 融合。"""

    def __init__(
        self,
        *,
        run_entities: bool,
        node_dim: int,
        edge_dim: int,
        repr_dim: int,
        num_heads: int,
        coarse_layers_per_block: int,
        dropout: float,
        mode: Literal["coupled_only", "coupled_plus_modal"],
        coarse_ffn_dim: int,
        candidate_bias_rbf_bins: int,
        candidate_bias_hidden_dim: int,
        candidate_bias_distance_scale: float,
        graph_update_edge: bool,
    ) -> None:
        super().__init__()
        self.run_entities = run_entities
        if run_entities:
            self.ligand_gnn_layer = NodeEdgeLayer(
                node_dim, edge_dim, update_edge=graph_update_edge
            )
            self.A_gnn_layer = NodeEdgeLayer(
                node_dim, edge_dim, update_edge=graph_update_edge
            )
            self.CCD_readback = RepresentationReadback(node_dim, repr_dim, num_heads, dropout)
            self.A_readback = RepresentationReadback(node_dim, repr_dim, num_heads, dropout)
        self.coarse_layers = nn.ModuleList(
            CoarseLayer(
                repr_dim,
                num_heads,
                coarse_ffn_dim,
                dropout,
                mode,
                candidate_bias_rbf_bins,
                candidate_bias_hidden_dim,
                candidate_bias_distance_scale,
            )
            for _ in range(coarse_layers_per_block)
        )
        self.mode = mode
        if mode == "coupled_plus_modal":
            self.BOX_norm = nn.LayerNorm(repr_dim)
            self.modal_norm = nn.ModuleList((nn.LayerNorm(repr_dim), nn.LayerNorm(repr_dim)))
            self.BOX_fusion = TypedAttention(
                repr_dim, (repr_dim, repr_dim), repr_dim, num_heads
            )
            self.BOX_fusion_dropout = nn.Dropout(dropout)

    def update_entities(
        self, ligand_state: GraphState, A_state: GraphState
    ) -> tuple[GraphState, GraphState]:
        """对全 batch 的互不连边配体图与 A 图各执行一次本 Block 更新。"""

        if not self.run_entities:
            return ligand_state, A_state
        return self.ligand_gnn_layer(ligand_state), self.A_gnn_layer(A_state)

    def update_entity_tensors(
        self,
        ligand_node: Tensor,
        ligand_edge: Tensor,
        ligand_edge_index: Tensor,
        A_node: Tensor,
        A_edge: Tensor,
        A_edge_index: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """执行实体更新并返回 activation checkpoint 可直接承载的四个浮点张量。"""

        ligand_state, A_state = self.update_entities(
            GraphState(ligand_node, ligand_edge, ligand_edge_index),
            GraphState(A_node, A_edge, A_edge_index),
        )
        return ligand_state.node, ligand_state.edge, A_state.node, A_state.edge

    def forward(
        self,
        coarse: CoarseState,
        ligand_state: GraphState,
        A_state: GraphState,
        ligand_ptr: tuple[int, ...],
        A_ptr: tuple[int, ...],
        occurrence_to_ligand: Tensor,
        candidate_centers: Tensor,
        *,
        detach_readback: bool,
    ) -> CoarseState:
        for layer in self.coarse_layers:
            coarse = layer(coarse, candidate_centers)
        if self.run_entities:
            ligand_nodes = ligand_state.node.detach() if detach_readback else ligand_state.node
            A_nodes = A_state.node.detach() if detach_readback else A_state.node
            CCD_rows = []
            for occurrence_index, ligand_index in enumerate(occurrence_to_ligand.tolist()):
                start, end = ligand_ptr[ligand_index : ligand_index + 2]
                CCD_rows.append(
                    self.CCD_readback(coarse.CCD_repr[occurrence_index : occurrence_index + 1], ligand_nodes[start:end])[0]
                )
            A_rows = []
            for candidate_index in range(len(A_ptr) - 1):
                start, end = A_ptr[candidate_index : candidate_index + 2]
                if start == end:
                    A_rows.append(coarse.A_repr[candidate_index])
                else:
                    A_rows.append(
                        self.A_readback(coarse.A_repr[candidate_index : candidate_index + 1], A_nodes[start:end])[0]
                    )
            coarse = CoarseState(
                torch.stack(CCD_rows), coarse.BOX_repr, torch.stack(A_rows), coarse.Map_repr
            )
        if self.mode == "coupled_plus_modal":
            query = self.BOX_norm(coarse.BOX_repr).unsqueeze(1)
            A_context = self.modal_norm[0](coarse.A_repr).unsqueeze(1)
            Map_context = self.modal_norm[1](coarse.Map_repr).unsqueeze(1)
            BOX_repr = coarse.BOX_repr + self.BOX_fusion_dropout(
                self.BOX_fusion(query, (A_context, Map_context))[:, 0]
            )
            coarse = CoarseState(coarse.CCD_repr, BOX_repr, coarse.A_repr, coarse.Map_repr)
        return coarse


class CoarsePredictionHead(nn.Module):
    """从四个双方向粗码构造共享 pair_repr，再输出独立 O/O′。"""

    def __init__(
        self,
        *,
        node_dim: int = 128,
        repr_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        mode: Literal["coupled_only", "coupled_plus_modal"] = "coupled_plus_modal",
        O_initial_prior: float = 0.01,
        O_prime_initial_value: float = 0.2,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.recall_A = TypedAttention(repr_dim, (node_dim,), repr_dim, num_heads)
        precision_names = ("BOX",) if mode == "coupled_only" else ("BOX", "A", "Map")
        self.precision = nn.ModuleDict(
            {
                name: TypedAttention(repr_dim, (node_dim,), repr_dim, num_heads)
                for name in precision_names
            }
        )
        code_count = 2 if mode == "coupled_only" else 4
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(code_count * repr_dim),
            nn.Linear(code_count * repr_dim, repr_dim),
        )
        self.pair_ffn_norm = nn.LayerNorm(repr_dim)
        self.pair_ffn = nn.Sequential(
            nn.Linear(repr_dim, 2 * repr_dim),
            nn.SiLU(),
            nn.Linear(2 * repr_dim, repr_dim),
            nn.Dropout(dropout),
        )
        self.OHead = nn.Linear(repr_dim, 1)
        self.OPrimeHead = nn.Linear(repr_dim, 1)
        nn.init.normal_(self.OHead.weight, std=0.01)
        nn.init.constant_(self.OHead.bias, math.log(O_initial_prior / (1 - O_initial_prior)))
        nn.init.normal_(self.OPrimeHead.weight, std=0.01)
        nn.init.constant_(
            self.OPrimeHead.bias,
            math.log(O_prime_initial_value / (1 - O_prime_initial_value)),
        )

    def build_pair_repr(
        self,
        coarse: CoarseState,
        ligand_nodes: Tensor,
        A_nodes: Tensor,
        ligand_ptr: tuple[int, ...],
        A_ptr: tuple[int, ...],
        occurrence_to_ligand: Tensor,
    ) -> Tensor:
        candidate_count = coarse.BOX_repr.shape[0]
        occurrence_count = coarse.CCD_repr.shape[0]
        recall = A_nodes.new_zeros((candidate_count, occurrence_count, coarse.CCD_repr.shape[-1]))
        for candidate_index in range(candidate_count):
            start, end = A_ptr[candidate_index : candidate_index + 2]
            if start != end:
                recall[candidate_index] = self.recall_A(
                    coarse.CCD_repr, (A_nodes[start:end],)
                )

        precision_codes = {
            name: ligand_nodes.new_empty((candidate_count, occurrence_count, coarse.CCD_repr.shape[-1]))
            for name in self.precision
        }
        representation_by_name = {
            "BOX": coarse.BOX_repr,
            "A": coarse.A_repr,
            "Map": coarse.Map_repr,
        }
        for ligand_index in range(len(ligand_ptr) - 1):
            occurrence_indices = torch.nonzero(
                occurrence_to_ligand == ligand_index, as_tuple=False
            ).flatten()
            start, end = ligand_ptr[ligand_index : ligand_index + 2]
            for name, attention in self.precision.items():
                queries = representation_by_name[name][:, None, :].expand(
                    -1, len(occurrence_indices), -1
                )
                values = attention(
                    queries.reshape(1, -1, queries.shape[-1]),
                    (ligand_nodes[start:end].unsqueeze(0),),
                )[0].reshape(candidate_count, len(occurrence_indices), -1)
                precision_codes[name][:, occurrence_indices] = values
        ordered_codes = [recall, precision_codes["BOX"]]
        if self.mode == "coupled_plus_modal":
            ordered_codes.extend((precision_codes["A"], precision_codes["Map"]))
        pair_repr = self.pair_projection(torch.cat(ordered_codes, dim=-1))
        return pair_repr + self.pair_ffn(self.pair_ffn_norm(pair_repr))

    def predict(self, pair_repr: Tensor) -> HeadOutput:
        O_logit = self.OHead(pair_repr).squeeze(-1)
        O_prime_logit = self.OPrimeHead(pair_repr).squeeze(-1)
        return HeadOutput(pair_repr, O_logit, O_prime_logit, torch.sigmoid(O_prime_logit))

    def forward(
        self,
        coarse: CoarseState,
        ligand_nodes: Tensor,
        A_nodes: Tensor,
        ligand_ptr: tuple[int, ...],
        A_ptr: tuple[int, ...],
        occurrence_to_ligand: Tensor,
    ) -> HeadOutput:
        return self.predict(
            self.build_pair_repr(
                coarse,
                ligand_nodes,
                A_nodes,
                ligand_ptr,
                A_ptr,
                occurrence_to_ligand,
            )
        )


class EntityStem(nn.Module):
    """把配体/A 原始节点和 31 维边字段投影到共同图状态维度。"""

    def __init__(
        self,
        ligand_input_dim: int,
        A_input_dim: int,
        edge_input_dim: int,
        node_dim: int,
        edge_dim: int,
    ) -> None:
        super().__init__()
        self.ligand_node = nn.Linear(ligand_input_dim, node_dim, bias=False)
        self.ligand_edge = nn.Linear(edge_input_dim, edge_dim, bias=False)
        self.A_node = nn.Linear(A_input_dim, node_dim, bias=False)
        self.A_edge = nn.Linear(edge_input_dim, edge_dim, bias=False)

    def forward(
        self, ligand: PackedEntities, A: PackedEntities
    ) -> tuple[GraphState, GraphState]:
        if ligand.node_input.shape[-1] != self.ligand_node.in_features:
            raise ValueError(
                f"配体节点特征末维应为 {self.ligand_node.in_features}，"
                f"收到 {ligand.node_input.shape[-1]}。"
            )
        if A.node_input.shape[-1] != self.A_node.in_features:
            raise ValueError(
                f"A 节点特征末维应为 {self.A_node.in_features}，"
                f"收到 {A.node_input.shape[-1]}。"
            )
        return (
            GraphState(
                self.ligand_node(ligand.node_input),
                self.ligand_edge(ligand.edge_input),
                ligand.edge_index,
            ),
            GraphState(self.A_node(A.node_input), self.A_edge(A.edge_input), A.edge_index),
        )


class FiLMPlusCombine(nn.Module):
    """以零初始化末层产生 ``main * (1 + gamma) + beta``。"""

    def __init__(self, dimension: int, ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dimension * ratio)
        self.generator = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * dimension),
        )
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)

    def forward(self, main: Tensor, condition: Tensor) -> Tensor:
        gamma, beta = self.generator(condition).chunk(2, dim=-1)
        return main * (1.0 + gamma) + beta


class GraphStateFiLMPlus(nn.Module):
    """用相同短算子分别调制节点和边状态。"""

    def __init__(self, node_dim: int, edge_dim: int, ratio: float, dropout: float) -> None:
        super().__init__()
        self.node = FiLMPlusCombine(node_dim, ratio, dropout)
        self.edge = FiLMPlusCombine(edge_dim, ratio, dropout)

    def forward(self, main: GraphState, condition: GraphState) -> GraphState:
        return GraphState(
            self.node(main.node, condition.node.detach()),
            self.edge(main.edge, condition.edge.detach()),
            main.edge_index,
        )


class EntityAuxiliaryHead(nn.Module):
    """节点元素、六类边和可选 A binding atom 预测。"""

    def __init__(self, node_dim: int, edge_dim: int, *, predict_binding: bool) -> None:
        super().__init__()
        self.element_head = nn.Linear(node_dim, 128)
        self.edge_class_head = nn.Linear(edge_dim, 6)
        self.binding_head = nn.Linear(node_dim, 1) if predict_binding else None

    def forward(self, state: GraphState) -> EntityAuxiliaryOutput:
        return EntityAuxiliaryOutput(
            element_logit=self.element_head(state.node),
            edge_class_logit=self.edge_class_head(state.edge),
            binding_logit=(self.binding_head(state.node).squeeze(-1) if self.binding_head else None),
        )


def build_contact_marginals(
    A_coordinates: Tensor,
    A_mask: Tensor,
    ligand_coordinates: Tensor,
    ligand_present: Tensor,
    pair_chunk_size: int,
    contact_radius: float = 4.0,
) -> tuple[Tensor, Tensor]:
    """按候选框—真实 occurrence 批量归约 4 Å 原子接触标签。

    ``A_coordinates`` 与 ``ligand_coordinates`` 分别为 ``[C,N_A,3]`` 和
    ``[S,N_L,3]``；返回 ``recall [C,S,N_L]`` 与 ``precision [C,S,N_A]``。
    每个候选框—真实 occurrence 只计算一次，预测槽位随后复用这些标签。
    """

    candidate_count, A_width = A_mask.shape
    occurrence_count, ligand_width = ligand_present.shape
    candidate_index = torch.arange(candidate_count, device=A_coordinates.device).repeat_interleave(
        occurrence_count
    )
    occurrence_index = torch.arange(
        occurrence_count, device=A_coordinates.device
    ).repeat(candidate_count)
    recall_chunks = []
    precision_chunks = []
    with torch.no_grad():
        for start in range(0, len(candidate_index), pair_chunk_size):
            candidate_chunk = candidate_index[start : start + pair_chunk_size]
            occurrence_chunk = occurrence_index[start : start + pair_chunk_size]
            chunk_A_mask = A_mask[candidate_chunk]
            chunk_A_width = max(int(chunk_A_mask.sum(dim=1).max().item()), 1)
            chunk_A_mask = chunk_A_mask[:, :chunk_A_width]
            valid = (
                chunk_A_mask[:, :, None] & ligand_present[occurrence_chunk, None, :]
            )
            contact = (
                torch.cdist(
                    A_coordinates[candidate_chunk, :chunk_A_width],
                    ligand_coordinates[occurrence_chunk],
                )
                <= contact_radius
            ) & valid
            recall_chunks.append(contact.any(dim=1))
            precision = torch.zeros(
                (len(candidate_chunk), A_width),
                dtype=torch.bool,
                device=A_coordinates.device,
            )
            precision[:, :chunk_A_width] = contact.any(dim=2)
            precision_chunks.append(precision)
    return (
        torch.cat(recall_chunks).reshape(candidate_count, occurrence_count, ligand_width),
        torch.cat(precision_chunks).reshape(candidate_count, occurrence_count, A_width),
    )


def _masked_sigmoid_focal_mean(logit: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """把同形 ``[T,N]`` 原子 logit 与标签比较，按原子轴分别求均值。"""

    binary_cross_entropy = F.binary_cross_entropy_with_logits(
        logit.float(), target.float(), reduction="none"
    )
    probability = torch.sigmoid(logit.float())
    target_float = target.float()
    target_probability = probability * target_float + (1.0 - probability) * (
        1.0 - target_float
    )
    focal = ((1.0 - target_probability) ** 2) * binary_cross_entropy
    denominator = mask.sum(dim=-1)
    mean = (focal * mask).sum(dim=-1) / denominator.clamp_min(1)
    return mean.masked_fill(denominator == 0, 0.0)


class FineReadout(nn.Module):
    """先在 128 维批量池化 pair-specific 原子，再投影到 512 维摘要。"""

    def __init__(self, node_dim: int, readout_dim: int, num_heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, node_dim) / math.sqrt(node_dim))
        self.node_norm = nn.LayerNorm(node_dim)
        self.attention = TypedAttention(node_dim, (node_dim,), node_dim, num_heads)
        self.output = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, readout_dim),
            nn.SiLU(),
            nn.LayerNorm(readout_dim),
        )

    def forward(self, atoms: Tensor, atom_mask: Tensor) -> Tensor:
        query = self.query.unsqueeze(0).expand(len(atoms), -1, -1)
        pooled = query + self.attention(
            query,
            (self.node_norm(atoms),),
            query_mask=torch.ones(
                (len(atoms), 1), dtype=torch.bool, device=atoms.device
            ),
            context_mask=atom_mask,
        )
        return self.output(pooled[:, 0])


class FinePairBranch(nn.Module):
    """批量执行 ligand↔A 双向只读原子 attention、readout 和接触辅助头。"""

    def __init__(
        self,
        *,
        node_dim: int,
        repr_dim: int,
        slot_repr_dim: int,
        num_heads: int = 8,
        readout_dim: int = 512,
        dropout: float = 0.1,
        use_ffn: bool = False,
        ffn_hidden_dim: int = 512,
        auxiliary_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.slot_projection = nn.Linear(slot_repr_dim, node_dim, bias=False)
        self.ligand_norm = nn.LayerNorm(node_dim)
        self.A_norm = nn.LayerNorm(node_dim)
        self.recall_attention = TypedAttention(node_dim, (node_dim,), node_dim, num_heads)
        self.precision_attention = TypedAttention(node_dim, (node_dim,), node_dim, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.use_ffn = use_ffn
        self.recall_ffn_norm: nn.LayerNorm | None = None
        self.precision_ffn_norm: nn.LayerNorm | None = None
        self.recall_ffn: nn.Sequential | None = None
        self.precision_ffn: nn.Sequential | None = None
        if use_ffn:
            self.recall_ffn_norm = nn.LayerNorm(node_dim)
            self.precision_ffn_norm = nn.LayerNorm(node_dim)
            self.recall_ffn = nn.Sequential(
                nn.Linear(node_dim, ffn_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_hidden_dim, node_dim),
            )
            self.precision_ffn = nn.Sequential(
                nn.Linear(node_dim, ffn_hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_hidden_dim, node_dim),
            )
        self.recall_readout = FineReadout(node_dim, readout_dim, num_heads)
        self.precision_readout = FineReadout(node_dim, readout_dim, num_heads)
        self.pair_mlp = nn.Sequential(
            nn.LayerNorm(2 * readout_dim),
            nn.Linear(2 * readout_dim, readout_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_dim, repr_dim),
            nn.LayerNorm(repr_dim),
        )
        self.recall_atom_head = nn.Linear(node_dim, 1) if auxiliary_enabled else None
        self.precision_atom_head = nn.Linear(node_dim, 1) if auxiliary_enabled else None

    def forward(
        self,
        ligand_atoms: Tensor,
        A_atoms: Tensor,
        ligand_mask: Tensor,
        A_mask: Tensor,
        slot_embedding: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """一次处理 ``P`` 个候选框—occurrence 配对。

        配体与 A 原子分别使用 ``[P,N_L,D]`` 和 ``[P,N_A,D]``；两个掩码中的
        ``True`` 表示真实原子。返回配对表示 ``[P,D_repr]``，以及配体侧和 A 侧的
        原子接触 logit ``[P,N_L]`` 和 ``[P,N_A]``。``P`` 是当前执行 chunk 中的
        配对数量，不改变完整配对网格。
        """

        ligand_query = ligand_atoms + self.slot_projection(slot_embedding)[:, None]
        normalized_ligand = self.ligand_norm(ligand_query)
        normalized_A = self.A_norm(A_atoms)
        nonempty_A = A_mask.any(dim=1)
        has_nonempty_A = bool(nonempty_A.any())
        recall_increment = torch.zeros_like(ligand_query)
        if has_nonempty_A:
            recall_increment[nonempty_A] = self.recall_attention(
                normalized_ligand[nonempty_A],
                (normalized_A[nonempty_A],),
                query_mask=ligand_mask[nonempty_A],
                context_mask=A_mask[nonempty_A],
            ).to(recall_increment.dtype)
        fine_recall_atoms = ligand_query + self.dropout(recall_increment)
        precision_increment = torch.zeros_like(A_atoms)
        if has_nonempty_A:
            precision_increment[nonempty_A] = self.precision_attention(
                normalized_A[nonempty_A],
                (normalized_ligand[nonempty_A],),
                query_mask=A_mask[nonempty_A],
                context_mask=ligand_mask[nonempty_A],
            ).to(precision_increment.dtype)
        fine_precision_atoms = A_atoms + self.dropout(precision_increment)
        fine_recall_atoms = fine_recall_atoms * ligand_mask[:, :, None]
        fine_precision_atoms = fine_precision_atoms * A_mask[:, :, None]
        if self.recall_ffn is not None and self.precision_ffn is not None:
            recall_ffn_mask = ligand_mask & nonempty_A[:, None]
            fine_recall_atoms = fine_recall_atoms + self.recall_ffn(
                self.recall_ffn_norm(fine_recall_atoms)
            ) * recall_ffn_mask[:, :, None]
            fine_precision_atoms = fine_precision_atoms + self.precision_ffn(
                self.precision_ffn_norm(fine_precision_atoms)
            ) * A_mask[:, :, None]
        recall_repr = self.recall_readout(fine_recall_atoms, ligand_mask)
        precision_repr = torch.zeros_like(recall_repr)
        if has_nonempty_A:
            precision_repr[nonempty_A] = self.precision_readout(
                fine_precision_atoms[nonempty_A], A_mask[nonempty_A]
            )
        fine_pair_repr = self.pair_mlp(torch.cat((recall_repr, precision_repr), dim=-1))

        recall_logit = (
            self.recall_atom_head(fine_recall_atoms).squeeze(-1)
            if self.recall_atom_head is not None
            else ligand_atoms.new_empty((len(ligand_atoms), 0))
        )
        precision_logit = (
            self.precision_atom_head(fine_precision_atoms).squeeze(-1)
            if self.precision_atom_head is not None
            else ligand_atoms.new_empty((len(ligand_atoms), 0))
        )
        return fine_pair_repr, recall_logit, precision_logit


# 当前路线的模型入口。上方均为直接参与该入口的科学算子，不建立注册框架。


class Matcher(nn.Module):
    """当前 A+Map、O/O′ Matcher 的 Phase1 或 Phase2 显式模型。"""

    def __init__(
        self,
        *,
        phase: Literal[1, 2],
        A_node_input_dim: int = 49,
        edge_input_dim: int = 31,
        node_dim: int = 128,
        edge_dim: int = 64,
        repr_dim: int = 256,
        num_heads: int = 8,
        coarse_layers_per_block: int = 2,
        coarse_ffn_dim: int = 512,
        dropout: float = 0.1,
        mode: Literal["coupled_only", "coupled_plus_modal"] = "coupled_plus_modal",
        phase1_blocks: int = 4,
        phase2_blocks: int = 4,
        map_channels: tuple[int, int, int, int] = (16, 32, 48, 64),
        map_candidate_chunk_size: int | None = 64,
        map_input_channels: int = 1,
        map_input_shape_zyx: tuple[int, int, int] = (48, 48, 48),
        map_norm_groups: int = 8,
        map_bottleneck_heads: int = 4,
        map_bottleneck_layers: int = 1,
        map_bottleneck_ffn_dim: int = 128,
        map_activation_checkpoint: bool = False,
        map_scale_summary_dim: int = 64,
        map_summary_hidden_dim: int = 512,
        candidate_bias_rbf_bins: int = 32,
        candidate_bias_hidden_dim: int = 64,
        candidate_bias_distance_scale: float = 12.0,
        O_initial_prior: float = 0.01,
        O_prime_initial_value: float = 0.2,
        graph_update_edge: bool = True,
        graph_activation_checkpoint: bool = False,
        phase2_use_FiLM_plus: bool = False,
        phase2_condition_mlp_ratio: float = 2.0,
        phase2_condition_dropout: float = 0.0,
        entity_aux_weight: float = 0.1,
        fine_pair_aux_weight: float = 0.0,
        fine_pair_chunk_size: int | None = 2048,
        fine_pair_activation_checkpoint: bool = True,
        fine_attention_use_ffn: bool = False,
        fine_attention_ffn_hidden_dim: int = 512,
        fine_readout_dim: int = 512,
    ) -> None:
        super().__init__()
        self.phase = phase
        self.mode = mode
        self.phase1_blocks = phase1_blocks
        self.phase2_blocks = phase2_blocks
        self.num_blocks = phase1_blocks + phase2_blocks
        self.map_candidate_chunk_size = map_candidate_chunk_size
        self.fine_pair_chunk_size = fine_pair_chunk_size
        self.fine_pair_activation_checkpoint = fine_pair_activation_checkpoint
        self.graph_activation_checkpoint = graph_activation_checkpoint
        if phase1_blocks <= 0 or phase2_blocks <= 0:
            raise ValueError("phase1_blocks 与 phase2_blocks 都必须为正整数。")
        self.MapBackbone = MapBackbone(
            map_channels,
            input_channels=map_input_channels,
            input_shape_zyx=map_input_shape_zyx,
            norm_groups=map_norm_groups,
            bottleneck_heads=map_bottleneck_heads,
            bottleneck_layers=map_bottleneck_layers,
            bottleneck_ffn_dim=map_bottleneck_ffn_dim,
            dropout=dropout,
            activation_checkpoint=map_activation_checkpoint,
        )
        self.MapSummaryHead = MapSummaryHead(
            map_channels,
            scale_summary_dim=map_scale_summary_dim,
            summary_hidden_dim=map_summary_hidden_dim,
            repr_dim=repr_dim,
            dropout=dropout,
        )
        self.phase1_entity_stem = EntityStem(
            149, A_node_input_dim, edge_input_dim, node_dim, edge_dim
        )
        self.CCD_initial_readout = LearnedQueryReadout(node_dim, repr_dim, num_heads, dropout)
        self.A_initial_readout = LearnedQueryReadout(node_dim, repr_dim, num_heads, dropout)
        self.empty_A_repr = nn.Parameter(torch.randn(repr_dim) / math.sqrt(repr_dim))
        self.slot_embedding = nn.Embedding(100, repr_dim)
        self.count_embedding = nn.Embedding(101, repr_dim)
        self.CCD_initial_norm = nn.LayerNorm(repr_dim)
        self.BOX_initial = nn.Sequential(
            nn.Linear(2 * repr_dim, 2 * repr_dim),
            nn.SiLU(),
            nn.Linear(2 * repr_dim, repr_dim),
            nn.LayerNorm(repr_dim),
        )
        self.blocks = nn.ModuleList(
            MatcherBlock(
                run_entities=(block_index < phase1_blocks or phase == 2),
                node_dim=node_dim,
                edge_dim=edge_dim,
                repr_dim=repr_dim,
                num_heads=num_heads,
                coarse_layers_per_block=coarse_layers_per_block,
                dropout=dropout,
                mode=mode,
                coarse_ffn_dim=coarse_ffn_dim,
                candidate_bias_rbf_bins=candidate_bias_rbf_bins,
                candidate_bias_hidden_dim=candidate_bias_hidden_dim,
                candidate_bias_distance_scale=candidate_bias_distance_scale,
                graph_update_edge=graph_update_edge,
            )
            for block_index in range(self.num_blocks)
        )
        self.prediction_heads = nn.ModuleList(
            CoarsePredictionHead(
                node_dim=node_dim,
                repr_dim=repr_dim,
                num_heads=num_heads,
                dropout=dropout,
                mode=mode,
                O_initial_prior=O_initial_prior,
                O_prime_initial_value=O_prime_initial_value,
            )
            for _ in range(self.num_blocks)
        )
        self.phase2_entity_stem: EntityStem | None = None
        self.ligand_FiLM_plus: GraphStateFiLMPlus | None = None
        self.A_FiLM_plus: GraphStateFiLMPlus | None = None
        self.fine_pair_branch: FinePairBranch | None = None
        self.fine_adapter: nn.Sequential | None = None
        if phase == 2:
            self.phase2_entity_stem = EntityStem(
                149, A_node_input_dim, edge_input_dim, node_dim, edge_dim
            )
            self.phase2_use_FiLM_plus = phase2_use_FiLM_plus
            if phase2_use_FiLM_plus:
                self.ligand_FiLM_plus = GraphStateFiLMPlus(
                    node_dim, edge_dim, phase2_condition_mlp_ratio, phase2_condition_dropout
                )
                self.A_FiLM_plus = GraphStateFiLMPlus(
                    node_dim, edge_dim, phase2_condition_mlp_ratio, phase2_condition_dropout
                )
            self.fine_pair_branch = FinePairBranch(
                node_dim=node_dim,
                repr_dim=repr_dim,
                slot_repr_dim=repr_dim,
                num_heads=num_heads,
                readout_dim=fine_readout_dim,
                dropout=dropout,
                use_ffn=fine_attention_use_ffn,
                ffn_hidden_dim=fine_attention_ffn_hidden_dim,
                auxiliary_enabled=fine_pair_aux_weight > 0,
            )
            self.fine_adapter = nn.Sequential(nn.LayerNorm(repr_dim), nn.Linear(repr_dim, repr_dim))
            nn.init.zeros_(self.fine_adapter[-1].weight)
            nn.init.zeros_(self.fine_adapter[-1].bias)
        self.ligand_entity_auxiliary_head: EntityAuxiliaryHead | None = None
        self.A_entity_auxiliary_head: EntityAuxiliaryHead | None = None
        if entity_aux_weight > 0:
            self.ligand_entity_auxiliary_head = EntityAuxiliaryHead(
                node_dim, edge_dim, predict_binding=False
            )
            self.A_entity_auxiliary_head = EntityAuxiliaryHead(
                node_dim, edge_dim, predict_binding=True
            )

    def freeze_phase1_for_phase2(self) -> None:
        """冻结 Phase1 stem、初始读出/embedding、Block1–4 和对应预测头。"""

        modules = (
            self.phase1_entity_stem,
            self.CCD_initial_readout,
            self.A_initial_readout,
            self.slot_embedding,
            self.count_embedding,
            self.CCD_initial_norm,
            self.BOX_initial,
            *self.blocks[: self.phase1_blocks],
            *self.prediction_heads[: self.phase1_blocks],
        )
        for module in modules:
            module.requires_grad_(False)
            module.eval()
        self.empty_A_repr.requires_grad_(False)

    def train(self, mode: bool = True) -> Matcher:
        super().train(mode)
        if mode and self.phase == 2:
            self.freeze_phase1_for_phase2()
        return self

    @staticmethod
    def _slot_indices(occurrence_to_ligand: Tensor, ligand_count: int) -> tuple[Tensor, Tensor]:
        slot_indices = torch.empty_like(occurrence_to_ligand)
        identity_counts = torch.empty_like(occurrence_to_ligand)
        for ligand_index in range(ligand_count):
            occurrences = torch.nonzero(
                occurrence_to_ligand == ligand_index, as_tuple=False
            ).flatten()
            slot_indices[occurrences] = torch.arange(len(occurrences), device=occurrences.device)
            identity_counts[occurrences] = len(occurrences)
        return slot_indices, identity_counts

    def _Map_repr(self, batch: MatcherBatch, device: torch.device) -> list[Tensor]:
        counts = [sample.num_candidates for sample in batch.samples]
        density = batch.density.to(device)
        chunk_size = self.map_candidate_chunk_size or len(density)
        summaries = []
        for start in range(0, len(density), chunk_size):
            features = self.MapBackbone(density[start : start + chunk_size])
            summaries.append(self.MapSummaryHead(features))
        return list(torch.cat(summaries).split(counts))

    def _initial_coarse(
        self,
        sample: AnchorSample,
        ligand_state: GraphState,
        A_state: GraphState,
        ligand_ptr: tuple[int, ...],
        A_ptr: tuple[int, ...],
        Map_repr: Tensor,
        device: torch.device,
    ) -> CoarseState:
        identity_repr = torch.stack(
            [
                self.CCD_initial_readout(ligand_state.node[ligand_ptr[i] : ligand_ptr[i + 1]])
                for i in range(len(ligand_ptr) - 1)
            ]
        )
        occurrence_to_ligand = sample.occurrence_to_ligand.to(device)
        slot_indices, identity_counts = self._slot_indices(
            occurrence_to_ligand, len(ligand_ptr) - 1
        )
        CCD_repr = self.CCD_initial_norm(
            identity_repr[occurrence_to_ligand]
            + self.slot_embedding(slot_indices)
            + self.count_embedding(identity_counts)
        )
        A_rows = []
        for candidate_index in range(len(A_ptr) - 1):
            start, end = A_ptr[candidate_index : candidate_index + 2]
            A_rows.append(
                self.empty_A_repr
                if start == end
                else self.A_initial_readout(A_state.node[start:end])
            )
        A_repr = torch.stack(A_rows)
        BOX_repr = self.BOX_initial(torch.cat((A_repr, Map_repr), dim=-1))
        return CoarseState(CCD_repr, BOX_repr, A_repr, Map_repr)

    def _run_fine_pairs(
        self,
        sample: AnchorSample,
        coarse_pair_repr: Tensor,
        prediction_head: CoarsePredictionHead,
        ligand_state: GraphState,
        A_state: GraphState,
        ligand: PackedEntities,
        A: PackedEntities,
        device: torch.device,
    ) -> tuple[HeadOutput, Tensor, Tensor]:
        """分执行 chunk 计算完整 ``C×S_pred`` 网格及 ``[C,S_pred,S_gt]`` 辅助损失。

        chunk 只限制一次 FinePair forward 的配对数，不筛除任何候选框或 occurrence。
        """

        if self.fine_pair_branch is None or self.fine_adapter is None:
            raise RuntimeError("FinePair 仅能在 Phase2 中运行。")
        candidate_count, occurrence_count = coarse_pair_repr.shape[:2]
        candidate_index = torch.arange(candidate_count, device=device).repeat_interleave(
            occurrence_count
        )
        occurrence_index = torch.arange(occurrence_count, device=device).repeat(candidate_count)
        occurrence_to_ligand = sample.occurrence_to_ligand.to(device)
        slot_index, _ = self._slot_indices(
            occurrence_to_ligand, len(ligand.node_ptr) - 1
        )
        ligand_atoms, ligand_mask = _pad_packed_values(ligand_state.node, ligand.node_ptr)
        A_atoms, A_mask = _pad_packed_values(A_state.node, A.node_ptr)
        A_coordinates, _ = _pad_packed_values(A.coordinates, A.node_ptr)
        pair_chunk_size = self.fine_pair_chunk_size or len(candidate_index)
        auxiliary_enabled = self.fine_pair_branch.recall_atom_head is not None
        gt_coordinates = (
            pad_sequence(
                tuple(value.to(device) for value in sample.ligand_gt_coordinates),
                batch_first=True,
            )
            if auxiliary_enabled and sample.ligand_gt_coordinates is not None
            else None
        )
        gt_present = (
            pad_sequence(
                tuple(value.to(device) for value in sample.ligand_gt_present),
                batch_first=True,
            )
            if auxiliary_enabled and sample.ligand_gt_present is not None
            else None
        )
        contact_recall = None
        contact_precision = None
        if gt_coordinates is not None and gt_present is not None:
            contact_recall, contact_precision = build_contact_marginals(
                A_coordinates,
                A_mask,
                gt_coordinates,
                gt_present,
                pair_chunk_size,
            )

        def compute_chunk(
            coarse_chunk: Tensor, candidate_chunk: Tensor, occurrence_chunk: Tensor
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            ligand_index = occurrence_to_ligand[occurrence_chunk]
            chunk_ligand_mask = ligand_mask[ligand_index]
            chunk_A_mask = A_mask[candidate_chunk]
            ligand_width = int(chunk_ligand_mask.sum(dim=1).max().item())
            A_width = max(int(chunk_A_mask.sum(dim=1).max().item()), 1)
            chunk_ligand_mask = chunk_ligand_mask[:, :ligand_width]
            chunk_A_mask = chunk_A_mask[:, :A_width]
            fine_repr, recall_logit, precision_logit = self.fine_pair_branch(
                ligand_atoms[ligand_index, :ligand_width],
                A_atoms[candidate_chunk, :A_width],
                chunk_ligand_mask,
                chunk_A_mask,
                self.slot_embedding(slot_index[occurrence_chunk]),
            )
            recall_by_gt = coarse_chunk.new_zeros(
                (len(coarse_chunk), occurrence_count), dtype=torch.float32
            )
            precision_by_gt = torch.zeros_like(recall_by_gt)
            if contact_recall is not None and contact_precision is not None:
                same_identity = (
                    occurrence_to_ligand[occurrence_chunk, None]
                    == occurrence_to_ligand[None, :]
                )
                pair_row, gt_index = torch.nonzero(same_identity, as_tuple=True)
                target_candidate = candidate_chunk[pair_row]
                recall_by_gt[pair_row, gt_index] = _masked_sigmoid_focal_mean(
                    recall_logit[pair_row],
                    contact_recall[target_candidate, gt_index, :ligand_width],
                    gt_present[gt_index, :ligand_width],
                )
                precision_by_gt[pair_row, gt_index] = _masked_sigmoid_focal_mean(
                    precision_logit[pair_row],
                    contact_precision[target_candidate, gt_index, :A_width],
                    A_mask[target_candidate, :A_width],
                )
            joint_pair_repr = coarse_chunk + self.fine_adapter(fine_repr)
            prediction = prediction_head.predict(joint_pair_repr)
            return (
                prediction.O_logit,
                prediction.O_prime_logit,
                recall_by_gt,
                precision_by_gt,
            )

        O_logits = []
        O_prime_logits = []
        recall_losses = []
        precision_losses = []
        flat_coarse = coarse_pair_repr.flatten(0, 1)
        for start in range(0, len(candidate_index), pair_chunk_size):
            arguments = (
                flat_coarse[start : start + pair_chunk_size],
                candidate_index[start : start + pair_chunk_size],
                occurrence_index[start : start + pair_chunk_size],
            )
            if self.training and self.fine_pair_activation_checkpoint:
                chunk_output = checkpoint(
                    compute_chunk,
                    *arguments,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                chunk_output = compute_chunk(*arguments)
            O_logits.append(chunk_output[0])
            O_prime_logits.append(chunk_output[1])
            recall_losses.append(chunk_output[2])
            precision_losses.append(chunk_output[3])
        O_logit = torch.cat(O_logits).reshape(candidate_count, occurrence_count)
        O_prime_logit = torch.cat(O_prime_logits).reshape(candidate_count, occurrence_count)
        return (
            HeadOutput(
                coarse_pair_repr,
                O_logit,
                O_prime_logit,
                torch.sigmoid(O_prime_logit),
            ),
            torch.cat(recall_losses).reshape(
                candidate_count, occurrence_count, occurrence_count
            ),
            torch.cat(precision_losses).reshape(
                candidate_count, occurrence_count, occurrence_count
            ),
        )

    def forward(self, batch: MatcherBatch) -> tuple[PDBMatcherOutput, ...]:
        """按 ``batch.samples`` 顺序返回逐 PDB 结果。

        配体图和 A 图可作为互不连边的全 batch 图共同更新；每次实体更新后立即按
        两个 ptr 切回 PDB。coarse 运算、预测头和返回对象始终严格按 PDB 隔离。
        """

        device = next(self.parameters()).device
        Map_repr_by_pdb = self._Map_repr(batch, device)
        packed_ligand = _pack_raw_graphs(batch.ligand_graphs, device)
        packed_A = _pack_raw_graphs(batch.A_graphs, device)
        ligand_state, A_state = self.phase1_entity_stem(
            packed_ligand, packed_A
        )
        packed_phase2 = (
            self.phase2_entity_stem(packed_ligand, packed_A)
            if self.phase2_entity_stem is not None
            else None
        )

        coarse_by_pdb = []
        candidate_centers_by_pdb = []
        occurrence_to_ligand_by_pdb = []
        for pdb_index, (sample, Map_repr) in enumerate(
            zip(batch.samples, Map_repr_by_pdb, strict=True)
        ):
            ligand, local_ligand_state = _slice_packed_entities(
                packed_ligand,
                ligand_state,
                batch.ligand_identity_ptr[pdb_index],
                batch.ligand_identity_ptr[pdb_index + 1],
            )
            A, local_A_state = _slice_packed_entities(
                packed_A,
                A_state,
                batch.candidate_ptr[pdb_index],
                batch.candidate_ptr[pdb_index + 1],
            )
            coarse_by_pdb.append(
                self._initial_coarse(
                    sample,
                    local_ligand_state,
                    local_A_state,
                    ligand.node_ptr,
                    A.node_ptr,
                    Map_repr,
                    device,
                )
            )
            candidate_centers_by_pdb.append(
                torch.stack([candidate.center_xyz for candidate in sample.candidates]).to(device)
            )
            occurrence_to_ligand_by_pdb.append(sample.occurrence_to_ligand.to(device))

        block_outputs_by_pdb: list[list[HeadOutput | None]] = [
            [] for _ in batch.samples
        ]
        fine_losses_by_pdb: list[tuple[Tensor | None, Tensor | None]] = [
            (None, None) for _ in batch.samples
        ]
        phase1_final: tuple[GraphState, GraphState] | None = None
        for block_index, block in enumerate(self.blocks):
            if self.phase == 2 and block_index == self.phase1_blocks:
                if packed_phase2 is None:
                    raise RuntimeError("Phase2 缺少独立 entity stem 状态。")
                phase1_final = (ligand_state, A_state)
                ligand_state, A_state = packed_phase2
                if self.ligand_FiLM_plus is not None and self.A_FiLM_plus is not None:
                    ligand_state = self.ligand_FiLM_plus(ligand_state, phase1_final[0])
                    A_state = self.A_FiLM_plus(A_state, phase1_final[1])

            if self.training and self.graph_activation_checkpoint and block.run_entities:
                ligand_node, ligand_edge, A_node, A_edge = checkpoint(
                    block.update_entity_tensors,
                    ligand_state.node,
                    ligand_state.edge,
                    ligand_state.edge_index,
                    A_state.node,
                    A_state.edge,
                    A_state.edge_index,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
                ligand_state = GraphState(ligand_node, ligand_edge, ligand_state.edge_index)
                A_state = GraphState(A_node, A_edge, A_state.edge_index)
            else:
                ligand_state, A_state = block.update_entities(ligand_state, A_state)
            for pdb_index, sample in enumerate(batch.samples):
                ligand, local_ligand_state = _slice_packed_entities(
                    packed_ligand,
                    ligand_state,
                    batch.ligand_identity_ptr[pdb_index],
                    batch.ligand_identity_ptr[pdb_index + 1],
                )
                A, local_A_state = _slice_packed_entities(
                    packed_A,
                    A_state,
                    batch.candidate_ptr[pdb_index],
                    batch.candidate_ptr[pdb_index + 1],
                )
                coarse = block(
                    coarse_by_pdb[pdb_index],
                    local_ligand_state,
                    local_A_state,
                    ligand.node_ptr,
                    A.node_ptr,
                    occurrence_to_ligand_by_pdb[pdb_index],
                    candidate_centers_by_pdb[pdb_index],
                    detach_readback=(
                        self.phase == 2 and block_index >= self.phase1_blocks
                    ),
                )
                coarse_by_pdb[pdb_index] = coarse
                active_head = self.phase == 1 or block_index >= self.phase1_blocks
                if not active_head:
                    block_outputs_by_pdb[pdb_index].append(None)
                    continue
                entity_nodes = self.phase != 2 or block_index < self.phase1_blocks
                head_output = self.prediction_heads[block_index](
                    coarse,
                    local_ligand_state.node if entity_nodes else local_ligand_state.node.detach(),
                    local_A_state.node if entity_nodes else local_A_state.node.detach(),
                    ligand.node_ptr,
                    A.node_ptr,
                    occurrence_to_ligand_by_pdb[pdb_index],
                )
                if self.phase == 2 and block_index == self.num_blocks - 1:
                    head_output, recall_loss, precision_loss = self._run_fine_pairs(
                        sample,
                        head_output.coarse_pair_repr,
                        self.prediction_heads[block_index],
                        local_ligand_state,
                        local_A_state,
                        ligand,
                        A,
                        device,
                    )
                    fine_losses_by_pdb[pdb_index] = (recall_loss, precision_loss)
                block_outputs_by_pdb[pdb_index].append(head_output)

        outputs = []
        for pdb_index in range(len(batch.samples)):
            ligand, local_ligand_state = _slice_packed_entities(
                packed_ligand,
                ligand_state,
                batch.ligand_identity_ptr[pdb_index],
                batch.ligand_identity_ptr[pdb_index + 1],
            )
            A, local_A_state = _slice_packed_entities(
                packed_A,
                A_state,
                batch.candidate_ptr[pdb_index],
                batch.candidate_ptr[pdb_index + 1],
            )
            ligand_auxiliary = (
                self.ligand_entity_auxiliary_head(local_ligand_state)
                if self.ligand_entity_auxiliary_head is not None
                else None
            )
            A_auxiliary = (
                self.A_entity_auxiliary_head(local_A_state)
                if self.A_entity_auxiliary_head is not None
                else None
            )
            outputs.append(
                PDBMatcherOutput(
                    tuple(block_outputs_by_pdb[pdb_index]),
                    ligand_auxiliary,
                    A_auxiliary,
                    local_ligand_state,
                    local_A_state,
                    ligand,
                    A,
                    *fine_losses_by_pdb[pdb_index],
                )
            )
        return tuple(outputs)
