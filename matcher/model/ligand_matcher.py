"""由 ``ClosedSetMatcher`` 组合候选编码器与配体身份分支，并以 ``FullMatcherOutput`` 返回六分类、区域和闭集身份结果。"""

from __future__ import annotations

import math
from typing import TypedDict, cast

import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import CandidateBatch
from ..schema import LANGUAGE_EMBEDDING_DIM
from .candidate_encoder import CandidateEncoder, CandidateEncoderOutput
from .config import MatcherModelConfig


class IdentityRepresentations(TypedDict):
    """三种身份表示：``native`` 是 ``(B,I,D)`` 语言基线；``graph`` 是 ``(B,I,D)`` 原生表示查询自身未池化原子 token 的结果；``inventory`` 是 ``(B,I,D)`` 排除自身后读取其他原生身份的结果。"""

    native: torch.Tensor
    graph: torch.Tensor
    inventory: torch.Tensor


class FullMatcherOutput(CandidateEncoderOutput):
    """完整输出：继承候选输出五个字段；``identity_logits`` 是 ``(B,I)`` 闭集分数；``identity_native/graph/inventory`` 是三个 ``(B,I,D)`` 身份表示。"""

    identity_logits: torch.Tensor
    identity_native: torch.Tensor
    identity_graph: torch.Tensor
    identity_inventory: torch.Tensor


class GraphMessageLayer(nn.Module):
    """沿双向化学键更新原子 token。构造参数是模型宽度 ``D``、键特征宽度和 dropout；前向输入 ``(N,D)`` 节点、``(2,E)`` 有向边与 ``(E,F_e)`` 边特征，输出 ``(N,D)``。"""

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
        """求和轴是指向同一目标节点的边轴 ``E``；节点轴 ``N`` 和通道轴 ``D`` 保留。"""

        if edge_index.shape != (2, edge_features.shape[0]):
            raise ValueError("graph_edge_index 必须是与边特征对齐的 (2,E)。")
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
    """保留每个配体图的原子 token。构造参数是模型宽度 ``D``、消息层数和 dropout；前向读取 ``CandidateBatch`` 图字段，输出 ``(N,D)`` 而不做 mean/max 池化。"""

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

    def forward(self, batch: CandidateBatch) -> torch.Tensor:
        """将 ``N`` 个原子的元素、电荷、残基编号和 15 维连续特征融合后，沿 ``E`` 条有向键更新并返回 ``(N,D)``。"""

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
        return nodes


class GraphIdentityAttention(nn.Module):
    """以原生语言表示查询自身原子 token。构造参数是模型宽度 ``D``、头数 ``H`` 和 dropout；前向输入 ``(B,I,D)`` native、``(N,D)`` 节点及图映射，输出 ``(B,I,D)``。"""

    def __init__(self, model_dim: int, head_count: int, dropout: float) -> None:
        super().__init__()
        if model_dim % head_count != 0:
            raise ValueError("model_dim 必须能被配体图注意力头数整除。")
        self.model_dim = model_dim
        self.head_count = head_count
        self.head_dim = model_dim // head_count
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim, bias=False)
        self.value = nn.Linear(model_dim, model_dim, bias=False)
        self.output = nn.Linear(model_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        native: torch.Tensor,
        nodes: torch.Tensor,
        node_graph_index: torch.Tensor,
        graph_to_sample: torch.Tensor,
        graph_to_identity: torch.Tensor,
    ) -> torch.Tensor:
        """点积求和轴是每头通道 ``D/H``，softmax 求和轴是同一图的节点轴，保留图轴 ``G`` 和头轴 ``H``。"""

        graph_count = int(graph_to_sample.shape[0])
        if graph_count == 0:
            return torch.zeros_like(native)
        if node_graph_index.shape != (nodes.shape[0],):
            raise ValueError("graph_node_graph_index 必须与原子 token 轴对齐。")
        if torch.any(node_graph_index < 0) or torch.any(node_graph_index >= graph_count):
            raise ValueError("graph_node_graph_index 超出图轴 [0,G)。")
        graph_query = native[graph_to_sample, graph_to_identity]
        query = self.query(graph_query).reshape(graph_count, self.head_count, self.head_dim)
        key = self.key(nodes).reshape(nodes.shape[0], self.head_count, self.head_dim)
        value = self.value(nodes).reshape(nodes.shape[0], self.head_count, self.head_dim)
        score = (query[node_graph_index] * key).sum(dim=-1) / math.sqrt(self.head_dim)
        scatter_index = node_graph_index[:, None].expand_as(score)
        group_max = score.new_full((graph_count, self.head_count), torch.finfo(score.dtype).min)
        group_max.scatter_reduce_(0, scatter_index, score, reduce="amax", include_self=True)
        unnormalized = torch.exp(score - group_max[node_graph_index])
        denominator = score.new_zeros((graph_count, self.head_count))
        denominator.index_add_(0, node_graph_index, unnormalized)
        if torch.any(denominator == 0):
            raise ValueError("每张 LigandObject 图必须至少含一个原子 token。")
        weight = self.dropout(unnormalized / denominator[node_graph_index])
        weighted_value = weight[..., None] * value
        context = value.new_zeros((graph_count, self.head_count, self.head_dim))
        context.index_add_(0, node_graph_index, weighted_value)
        graph_vector = self.norm(self.output(context.reshape(graph_count, self.model_dim)))
        result = torch.zeros_like(native)
        result[graph_to_sample, graph_to_identity] = graph_vector
        return result


class SelfExcludingInventoryAttention(nn.Module):
    """以每个原生身份查询同一 PDB 其他原生身份。构造参数是模型宽度 ``D``、头数 ``H`` 和 dropout；前向输入 ``(B,I,D)`` 与 ``(B,I)`` 有效掩码，输出 ``(B,I,D)`` 上下文。"""

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
        """注意力的 query/key 轴均为 ``I``；最后一轴 key 做 softmax，``(B,1,I,I)`` 掩码向 ``H`` 头广播并屏蔽无效身份与对角线。"""

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
        valid = identity_mask[:, None, :, None] & identity_mask[:, None, None, :]
        valid = valid & off_diagonal[None, None, :, :]
        score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
        weight = torch.softmax(score, dim=-1) * valid.to(dtype=score.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        context = torch.matmul(self.dropout(weight), value)
        context = context.transpose(1, 2).reshape(batch_size, identity_count, self.model_dim)
        has_other = valid.any(dim=-1).transpose(1, 2)
        result = self.norm(self.output(context)) * has_other.to(context.dtype)
        return result * identity_mask[..., None].to(dtype=result.dtype)

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        """把 ``(B,I,D)`` 改排为 ``(B,H,I,D/H)``。"""

        batch_size, identity_count, _ = values.shape
        return values.reshape(
            batch_size,
            identity_count,
            self.head_count,
            self.head_dim,
        ).transpose(1, 2)


class IdentityEncoder(nn.Module):
    """产生原生、配体图和自排除清单三种表示。构造参数 ``config`` 定义 ``D/H``、图层数和 dropout；前向读取语言、图及映射字段，输出 ``IdentityRepresentations``。"""

    def __init__(self, config: MatcherModelConfig) -> None:
        super().__init__()
        self.language = nn.Sequential(
            nn.LayerNorm(LANGUAGE_EMBEDDING_DIM),
            nn.Linear(LANGUAGE_EMBEDDING_DIM, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )
        self.graph_encoder = LigandGraphEncoder(
            config.model_dim,
            config.graph_layers,
            config.dropout,
        )
        self.graph_attention = GraphIdentityAttention(
            config.model_dim,
            config.attention_heads,
            config.dropout,
        )
        self.inventory_attention = SelfExcludingInventoryAttention(
            config.model_dim,
            config.attention_heads,
            config.dropout,
        )

    def forward(self, batch: CandidateBatch) -> IdentityRepresentations:
        """语言表示按 ``identity_mask`` 补零；每个真值身份必须恰好对应一张图，图原子轴不会在返回前做 mean/max 池化。"""

        identity_mask = batch["identity_mask"]
        native = self.language(batch["language_embeddings"])
        native = native * identity_mask[..., None].to(dtype=native.dtype)
        graph_count = int(batch["graph_to_sample"].shape[0])
        occupancy = torch.zeros_like(identity_mask, dtype=torch.long)
        if graph_count > 0:
            sample = batch["graph_to_sample"]
            identity = batch["graph_to_identity"]
            if torch.any(sample < 0) or torch.any(sample >= native.shape[0]):
                raise ValueError("graph_to_sample 超出批次轴 [0,B)。")
            if torch.any(identity < 0) or torch.any(identity >= native.shape[1]):
                raise ValueError("graph_to_identity 超出身份轴 [0,I)。")
            occupancy.index_put_((sample, identity), torch.ones_like(sample), accumulate=True)
        if not torch.equal(occupancy, identity_mask.to(dtype=occupancy.dtype)):
            raise ValueError("每个有效身份必须恰好对应一张配体图，补零身份不得含图。")
        graph_nodes = self.graph_encoder(batch)
        graph = self.graph_attention(
            native,
            graph_nodes,
            batch["graph_node_graph_index"],
            batch["graph_to_sample"],
            batch["graph_to_identity"],
        )
        inventory = self.inventory_attention(native, identity_mask)
        return {"native": native, "graph": graph, "inventory": inventory}


class PairScorer(nn.Module):
    """对候选和一类身份表示打分。构造参数是模型宽度 ``D`` 和 dropout；前向输入 ``(B,D)`` 与 ``(B,I,D)``，输出 ``(B,I)``。"""

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

    def forward(self, candidate: torch.Tensor, identities: torch.Tensor) -> torch.Tensor:
        """余弦点积对通道轴 ``D`` 求和，保留批次轴 ``B`` 和身份轴 ``I``，再加逐身份兼容性残差。"""

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
        return cosine * self.log_scale.exp().clamp_max(100.0) + residual


class IdentityScorer(nn.Module):
    """合并三条候选—身份分支。构造参数是模型宽度 ``D`` 和 dropout；native 始终参与，graph/inventory 出口标量门从严格零开始；前向输出 ``(B,I)``。"""

    def __init__(self, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.native = PairScorer(model_dim, dropout)
        self.graph = PairScorer(model_dim, dropout)
        self.inventory = PairScorer(model_dim, dropout)
        self.graph_gate = nn.Parameter(torch.zeros(()))
        self.inventory_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        candidate: torch.Tensor,
        identities: IdentityRepresentations,
        identity_mask: torch.Tensor,
    ) -> torch.Tensor:
        """将 native 分数与两个门控增量逐元素相加；``(B,I)`` 掩码为假的位置在末端设为当前 dtype 的有限最小值。"""

        if candidate.shape[0] != identity_mask.shape[0]:
            raise ValueError("候选和身份掩码的批次轴不一致。")
        if (
            identities["graph"].shape != identities["native"].shape
            or identities["inventory"].shape != identities["native"].shape
        ):
            raise ValueError("三种身份表示必须具有同一 (B,I,D) 形状。")
        if identities["native"].shape[:2] != identity_mask.shape:
            raise ValueError("三种身份表示必须与 (B,I) 掩码对齐。")
        if identity_mask.shape[1] == 0:
            return candidate.new_empty((candidate.shape[0], 0))
        logits = self.native(candidate, identities["native"])
        logits = logits + self.graph_gate * self.graph(candidate, identities["graph"])
        logits = logits + self.inventory_gate * self.inventory(candidate, identities["inventory"])
        return logits.masked_fill(~identity_mask, torch.finfo(logits.dtype).min)


# ================================================================================================


class ClosedSetMatcher(nn.Module):
    """在候选六分类主干后执行 PDB 内小分子闭集身份分类。构造参数 ``config`` 定义候选、身份与打分模块；前向输入 ``CandidateBatch``，输出 ``FullMatcherOutput``。"""

    def __init__(self, config: MatcherModelConfig) -> None:
        super().__init__()
        self.config = config
        self.candidate_encoder = CandidateEncoder(config)
        self.identity_encoder = IdentityEncoder(config)
        self.identity_scorer = IdentityScorer(config.model_dim, config.dropout)

    def forward(self, batch: CandidateBatch) -> FullMatcherOutput:
        """候选分支只读体素和 A/P/V/PP；身份分支另读语言、图与清单，最后仅通过 ``(B,I)`` 配对分数与候选表示交互。"""

        candidate_output = self.candidate_encoder(batch)
        identities = self.identity_encoder(batch)
        identity_logits = self.identity_scorer(
            candidate_output["candidate_embedding"],
            identities,
            batch["identity_mask"],
        )
        output = dict(candidate_output)
        output["identity_logits"] = identity_logits
        output["identity_native"] = identities["native"]
        output["identity_graph"] = identities["graph"]
        output["identity_inventory"] = identities["inventory"]
        return cast(FullMatcherOutput, output)
