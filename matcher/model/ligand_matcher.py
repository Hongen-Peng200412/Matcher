"""编码配体图和冻结语言表示，并在 PDB 身份清单内执行闭集匹配。"""

from __future__ import annotations

import math
from typing import TypedDict

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import CandidateBatch
from ..schema import LANGUAGE_EMBEDDING_DIM
from .candidate_encoder import CandidateEncoder, CandidateEncoderOutput
from .config import MatcherModelConfig


class FullMatcherOutput(CandidateEncoderOutput):
    """在候选编码输出上增加身份 logits 和身份级表示。"""

    identity_logits: torch.Tensor
    identity_embeddings: torch.Tensor


class GraphMessageLayer(nn.Module):
    """沿双向化学键聚合带键属性的邻居消息。"""

    def __init__(self, model_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.node_message = nn.Linear(model_dim, model_dim, bias=False)
        self.edge_message = nn.Linear(edge_dim, model_dim, bias=False)
        self.gate = nn.Linear(model_dim * 2, model_dim)
        self.message_output = nn.Linear(model_dim, model_dim)
        self.message_norm = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, model_dim * 3),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 3, model_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """保持节点数不变；无化学键图只执行节点前馈残差。"""

        aggregated = torch.zeros_like(nodes)
        if edge_index.shape[1] > 0:
            source, target = edge_index
            message = self.node_message(nodes[source]) + self.edge_message(edge_features)
            aggregated.index_add_(0, target, message)
            degree = torch.zeros(nodes.shape[0], device=nodes.device, dtype=nodes.dtype)
            degree.index_add_(0, target, torch.ones_like(target, dtype=nodes.dtype))
            aggregated = aggregated / degree.clamp_min(1.0)[:, None]
        gate = torch.sigmoid(self.gate(torch.cat((nodes, aggregated), dim=-1)))
        update = self.message_output(F.silu(aggregated)) * gate
        nodes = self.message_norm(nodes + self.dropout(update))
        return self.feed_forward_norm(nodes + self.dropout(self.feed_forward(nodes)))


class LigandGraphEncoder(nn.Module):
    """把连接后的 LigandObject 图批次还原为每个身份一个向量。"""

    def __init__(self, model_dim: int, layer_count: int, dropout: float) -> None:
        super().__init__()
        self.element = nn.Embedding(128, model_dim)
        self.charge = nn.Embedding(33, model_dim)
        self.residue = nn.Embedding(512, model_dim)
        self.continuous = nn.Linear(15, model_dim, bias=False)
        self.input_norm = nn.LayerNorm(model_dim)
        self.layers = nn.ModuleList(
            GraphMessageLayer(model_dim, edge_dim=10, dropout=dropout)
            for _ in range(layer_count)
        )
        self.pool = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )

    def forward(self, batch: CandidateBatch) -> torch.Tensor:
        """返回 ``(图数量,model_dim)``，顺序与 ``graph_to_*`` 一致。"""

        graph_count = int(batch["graph_to_sample"].shape[0])
        if graph_count == 0:
            return self.element.weight.new_empty((0, self.element.embedding_dim))
        element = batch["graph_element"]
        charge = batch["graph_charge"]
        residue = batch["graph_residue_id"]
        if torch.any(element < 0) or torch.any(element >= self.element.num_embeddings):
            raise ValueError("LigandObject element 超出 [0,128) 编码范围。")
        if torch.any(charge < -16) or torch.any(charge > 16):
            raise ValueError("LigandObject charge 超出 [-16,16] 编码范围。")
        if torch.any(residue < 0) or torch.any(residue >= self.residue.num_embeddings):
            raise ValueError("LigandObject residue_id 超出 [0,512) 编码范围。")
        nodes = self.input_norm(
            self.element(element)
            + self.charge(charge + 16)
            + self.residue(residue)
            + self.continuous(batch["graph_node_continuous"])
        )
        for layer in self.layers:
            nodes = layer(nodes, batch["graph_edge_index"], batch["graph_edge_features"])
        return self.pool(_pool_graph_nodes(nodes, batch["graph_node_graph_index"], graph_count))


class SelfExcludingInventoryAttention(nn.Module):
    """每个身份只读取同一 PDB 内其他身份，不允许读取自身作为清单上下文。"""

    def __init__(self, model_dim: int, head_count: int, dropout: float) -> None:
        super().__init__()
        if model_dim % head_count != 0:
            raise ValueError("model_dim 必须能被身份清单注意力头数整除。")
        self.model_dim = model_dim
        self.head_count = head_count
        self.head_dim = model_dim // head_count
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim, bias=False)
        self.value = nn.Linear(model_dim, model_dim, bias=False)
        self.output = nn.Linear(model_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, identities: torch.Tensor, identity_mask: torch.Tensor) -> torch.Tensor:
        """返回加入自排除清单增量的身份表示；单身份清单的增量严格为零。"""

        if identities.ndim != 3 or identities.shape[:2] != identity_mask.shape:
            raise ValueError("identities 必须为 (B,I,D)，identity_mask 必须为 (B,I)。")
        batch_size, identity_count, _ = identities.shape
        if identity_count == 0:
            return identities
        query = self._heads(self.query(identities))
        key = self._heads(self.key(identities))
        value = self._heads(self.value(identities))
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        off_diagonal = ~torch.eye(identity_count, device=identities.device, dtype=torch.bool)
        valid = (
            identity_mask[:, None, :, None]
            & identity_mask[:, None, None, :]
            & off_diagonal[None, None, :, :]
        )
        score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
        weight = torch.softmax(score, dim=-1) * valid.to(dtype=score.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        context = torch.matmul(self.dropout(weight), value)
        context = context.transpose(1, 2).reshape(batch_size, identity_count, self.model_dim)
        increment = self.output(context) * valid.any(dim=-1).transpose(1, 2).to(context.dtype)
        enriched = self.norm(identities + increment)
        return enriched * identity_mask[..., None].to(dtype=enriched.dtype)

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        """把身份表示改排为 ``(B,H,I,D_head)``。"""

        batch_size, identity_count, _ = values.shape
        return values.reshape(
            batch_size,
            identity_count,
            self.head_count,
            self.head_dim,
        ).transpose(1, 2)


class IdentityEncoder(nn.Module):
    """逐身份融合配体图和冻结语言向量，再加入自排除清单上下文。"""

    def __init__(self, config: MatcherModelConfig) -> None:
        super().__init__()
        self.graph_encoder = LigandGraphEncoder(
            config.model_dim,
            config.graph_layers,
            config.dropout,
        )
        self.language = nn.Sequential(
            nn.LayerNorm(LANGUAGE_EMBEDDING_DIM),
            nn.Linear(LANGUAGE_EMBEDDING_DIM, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )
        self.graph_projection = nn.Linear(config.model_dim, config.model_dim)
        self.modality_gate = nn.Linear(config.model_dim * 2, config.model_dim)
        self.fusion_norm = nn.LayerNorm(config.model_dim)
        self.inventory = SelfExcludingInventoryAttention(
            config.model_dim,
            config.attention_heads,
            config.dropout,
        )

    def forward(self, batch: CandidateBatch) -> torch.Tensor:
        """返回按 ``identity_mask`` 补零的 ``(B,I,D)`` 身份表示。"""

        language = self.language(batch["language_embeddings"])
        graph_vectors = self.graph_encoder(batch)
        graph_grid = torch.zeros_like(language)
        occupancy = torch.zeros_like(batch["identity_mask"], dtype=torch.long)
        if graph_vectors.shape[0] > 0:
            sample = batch["graph_to_sample"]
            identity = batch["graph_to_identity"]
            graph_grid[sample, identity] = graph_vectors
            occupancy.index_put_((sample, identity), torch.ones_like(sample), accumulate=True)
        if not torch.equal(occupancy.bool(), batch["identity_mask"]):
            raise ValueError("每个有效身份必须恰好对应一张配体图，补零身份不得含图。")
        graph_grid = self.graph_projection(graph_grid)
        gate = torch.sigmoid(self.modality_gate(torch.cat((language, graph_grid), dim=-1)))
        fused = self.fusion_norm(gate * language + (1.0 - gate) * graph_grid)
        fused = fused * batch["identity_mask"][..., None].to(dtype=fused.dtype)
        return self.inventory(fused, batch["identity_mask"])


class IdentityScorer(nn.Module):
    """用归一化双线性分数和逐身份兼容性残差产生闭集 logits。"""

    def __init__(self, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.candidate = nn.Linear(model_dim, model_dim, bias=False)
        self.identity = nn.Linear(model_dim, model_dim, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.compatibility = nn.Sequential(
            nn.LayerNorm(model_dim * 4),
            nn.Linear(model_dim * 4, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def forward(
        self,
        candidate: torch.Tensor,
        identities: torch.Tensor,
        identity_mask: torch.Tensor,
    ) -> torch.Tensor:
        """返回 ``(B,I)`` logits；补零身份为负无穷。"""

        if identities.shape[:2] != identity_mask.shape or candidate.shape[0] != identities.shape[0]:
            raise ValueError("候选、身份与 identity_mask 的批次或身份轴不一致。")
        if identities.shape[1] == 0:
            return identities.new_empty((identities.shape[0], 0))
        candidate_projected = F.normalize(self.candidate(candidate), dim=-1)
        identity_projected = F.normalize(self.identity(identities), dim=-1)
        cosine = torch.einsum("bd,bid->bi", candidate_projected, identity_projected)
        candidate_expanded = candidate[:, None, :].expand_as(identities)
        residual = self.compatibility(
            torch.cat(
                (
                    candidate_expanded,
                    identities,
                    candidate_expanded * identities,
                    torch.abs(candidate_expanded - identities),
                ),
                dim=-1,
            )
        )[..., 0]
        logits = cosine * self.log_scale.exp().clamp_max(100.0) + residual
        return logits.masked_fill(~identity_mask, torch.finfo(logits.dtype).min)


class ClosedSetMatcher(nn.Module):
    """在候选六分类主干后，对当前 PDB 的可用小分子身份执行闭集分类。"""

    def __init__(self, config: MatcherModelConfig) -> None:
        super().__init__()
        self.config = config
        self.candidate_encoder = CandidateEncoder(config)
        self.identity_encoder = IdentityEncoder(config)
        self.identity_scorer = IdentityScorer(config.model_dim, config.dropout)

    def forward(self, batch: CandidateBatch) -> FullMatcherOutput:
        """同时返回候选级六分类/区域输出和 PDB 内身份 logits。"""

        candidate_output = self.candidate_encoder(batch)
        identity_embeddings = self.identity_encoder(batch)
        identity_logits = self.identity_scorer(
            candidate_output["candidate_embedding"],
            identity_embeddings,
            batch["identity_mask"],
        )
        return {
            **candidate_output,
            "identity_logits": identity_logits,
            "identity_embeddings": identity_embeddings,
        }


def _pool_graph_nodes(
    nodes: torch.Tensor,
    graph_index: torch.Tensor,
    graph_count: int,
) -> torch.Tensor:
    """按图计算节点均值和最大值，并返回 ``(G,2D)``。"""

    if graph_index.shape != (nodes.shape[0],):
        raise ValueError("graph_node_graph_index 必须与节点轴一一对应。")
    if torch.any(graph_index < 0) or torch.any(graph_index >= graph_count):
        raise ValueError("graph_node_graph_index 超出图数量。")
    total = torch.zeros((graph_count, nodes.shape[1]), device=nodes.device, dtype=nodes.dtype)
    total.index_add_(0, graph_index, nodes)
    count = torch.zeros(graph_count, device=nodes.device, dtype=nodes.dtype)
    count.index_add_(0, graph_index, torch.ones_like(graph_index, dtype=nodes.dtype))
    if torch.any(count == 0):
        raise ValueError("每张 LigandObject 图必须至少含一个节点。")
    mean = total / count[:, None]
    maximum = torch.full_like(total, torch.finfo(nodes.dtype).min)
    expanded_index = graph_index[:, None].expand_as(nodes)
    maximum.scatter_reduce_(0, expanded_index, nodes, reduce="amax", include_self=True)
    return torch.cat((mean, maximum), dim=-1)
