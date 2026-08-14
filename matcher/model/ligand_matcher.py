"""由 ``ClosedSetMatcher`` 组合候选编码器与配体身份分支。

符号 ``B/I/D_m/N/E/G/K/F_e`` 分别表示批次、批内最大身份数、模型宽度、
连接后的原子数、有向边数、配体图数、注意力头数和边特征宽度。
"""

from __future__ import annotations

import math
from typing import TypedDict, cast

import torch
from torch import nn
from torch.nn import functional as F

from ..data.schema import (
    CANDIDATE_MANUAL_FEATURE_DIM,
    CANDIDATE_SAMPLING_FEATURE_DIM,
    CLASS_NAMES,
    LANGUAGE_EMBEDDING_DIM,
)
from ..datasets.contracts import CandidateBatch
from .candidate_encoder import CandidateEncoder, CandidateEncoderOutput
from .config import MatcherModelConfig
from .fusion import normalize_candidate_manual_features, normalize_candidate_sampling_features


class IdentityRepresentations(TypedDict):
    """三种身份表示：``native`` 是 ``(B,I,D_m)`` 语言基线；``graph`` 是 ``(B,I,D_m)`` 原生表示查询自身未池化原子 token 的结果；``inventory`` 是 ``(B,I,D_m)`` 排除自身后读取其他原生身份的结果。"""

    native: torch.Tensor
    graph: torch.Tensor
    inventory: torch.Tensor


class FullMatcherOutput(CandidateEncoderOutput):
    """完整输出：继承候选输出五个字段；``identity_logits`` 是 ``(B,I)`` 闭集分数；``identity_native/graph/inventory`` 是三个 ``(B,I,D_m)`` 身份表示。"""

    identity_logits: torch.Tensor
    identity_native: torch.Tensor
    identity_graph: torch.Tensor
    identity_inventory: torch.Tensor


class GraphMessageLayer(nn.Module):
    """沿双向化学键更新原子 token。构造参数是模型宽度 ``D_m``、键特征宽度和 dropout；前向输入 ``(N,D_m)`` 节点、``(2,E)`` 有向边与 ``(E,F_e)`` 边特征，输出 ``(N,D_m)``；``graph_edge_index[0]`` 是源原子，``graph_edge_index[1]`` 是目标原子。"""

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
        """求和轴是指向同一目标节点的边轴 ``E``；节点轴 ``N`` 和通道轴 ``D_m`` 保留。"""

        if edge_index.shape != (2, edge_features.shape[0]):
            raise ValueError("graph_edge_index 必须是与边特征对齐的 (2,E)。")
        aggregated = torch.zeros_like(nodes)
        if edge_index.shape[1] > 0:
            source, target = edge_index
            message = self.node_message(nodes[source]) + self.edge_message(edge_features)
            # 自动混合精度可使 (E,D_m) 化学键消息变为 BF16/FP16，而累加器继承节点 token 的 dtype。
            # index_add_ 不自动提升 dtype；先对齐累加器 dtype，再按目标原子编号沿边轴 E 累加消息。
            aggregated.index_add_(0, target, message.to(dtype=aggregated.dtype))
            degree = torch.zeros(nodes.shape[0], device=nodes.device, dtype=nodes.dtype)
            degree.index_add_(0, target, torch.ones_like(target, dtype=nodes.dtype))
            aggregated = aggregated / degree.clamp_min(1.0)[:, None]
        gate = torch.sigmoid(self.gate(torch.cat((nodes, aggregated), dim=-1)))
        update = self.message_output(F.silu(aggregated)) * gate
        nodes = self.message_norm(nodes + self.dropout(update))
        return self.feed_forward_norm(nodes + self.dropout(self.feed_forward(nodes)))


class LigandGraphEncoder(nn.Module):
    """保留每个配体图的原子 token。构造参数是模型宽度 ``D_m``、消息层数和 dropout；前向读取 ``CandidateBatch`` 图字段，输出 ``(N,D_m)`` 而不做 mean/max 池化。"""

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
        """将 ``N`` 个原子的元素、电荷、残基编号和 15 维连续特征融合后，沿 ``E`` 条有向键更新并返回 ``(N,D_m)``。"""

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
    """以原生语言表示查询自身原子 token。

    构造参数是模型宽度 ``D_m``、头数 ``K`` 和 dropout。前向输入
    ``(B,I,D_m)`` native、``(N,D_m)`` 节点；``node_graph_index`` 是
    ``(N,)`` 且把节点映射到 ``[0,G)``，``graph_to_sample`` 是 ``(G,)``
    且把图映射到 ``[0,B)``，``graph_to_identity`` 是 ``(G,)`` 且把图映射到
    相应样本的 ``[0,I)``。输出为 ``(B,I,D_m)``。
    """

    def __init__(self, model_dim: int, head_count: int, dropout: float) -> None:
        super().__init__()
        if head_count <= 0 or model_dim <= 0 or model_dim % head_count != 0:
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
        """点积求和轴是每头通道 ``D_m/K``，softmax 求和轴是同一图的节点轴，保留图轴 ``G`` 和头轴 ``K``。"""

        graph_count = int(graph_to_sample.shape[0])
        if graph_count == 0:
            return torch.zeros_like(native)
        if native.ndim != 3 or nodes.ndim != 2 or nodes.shape[1] != self.model_dim:
            raise ValueError("native 必须为 (B,I,D_m)，nodes 必须为 (N,D_m)。")
        if graph_to_sample.ndim != 1 or graph_to_identity.shape != graph_to_sample.shape:
            raise ValueError("graph_to_sample 和 graph_to_identity 必须是同形 (G,) 张量。")
        if torch.any(graph_to_sample < 0) or torch.any(graph_to_sample >= native.shape[0]):
            raise ValueError("graph_to_sample 超出批次轴 [0,B)。")
        if torch.any(graph_to_identity < 0) or torch.any(graph_to_identity >= native.shape[1]):
            raise ValueError("graph_to_identity 超出身份轴 [0,I)。")
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
        # (N,K,D_m/K) 加权原子值可能因求和等归约运算升为 FP32；(G,K,D_m/K) 图累加器必须继承其 dtype。
        context = weighted_value.new_zeros((graph_count, self.head_count, self.head_dim))
        context.index_add_(0, node_graph_index, weighted_value)
        graph_vector = self.norm(self.output(context.reshape(graph_count, self.model_dim)))
        result = torch.zeros_like(native)
        result[graph_to_sample, graph_to_identity] = graph_vector
        return result


class SelfExcludingInventoryAttention(nn.Module):
    """以每个原生身份查询同一 PDB 其他原生身份。构造参数是模型宽度 ``D_m``、头数 ``K`` 和 dropout；前向输入 ``(B,I,D_m)`` 与 ``(B,I)`` 有效掩码，输出 ``(B,I,D_m)`` 上下文。"""

    def __init__(self, model_dim: int, head_count: int, dropout: float) -> None:
        super().__init__()
        if head_count <= 0 or model_dim <= 0 or model_dim % head_count != 0:
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
        """注意力的 query/key 轴均为 ``I``；最后一轴 key 做 softmax，``(B,1,I,I)`` 掩码向 ``K`` 头广播并屏蔽无效身份与对角线。"""

        if identities.ndim != 3 or identities.shape[:2] != identity_mask.shape:
            raise ValueError("identities 必须为 (B,I,D_m)，identity_mask 必须为 (B,I)。")
        if identity_mask.dtype != torch.bool:
            raise TypeError("identity_mask 必须是 bool。")
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
        """把 ``(B,I,D_m)`` 改排为 ``(B,K,I,D_m/K)``。"""

        batch_size, identity_count, _ = values.shape
        return values.reshape(
            batch_size,
            identity_count,
            self.head_count,
            self.head_dim,
        ).transpose(1, 2)


class IdentityEncoder(nn.Module):
    """产生原生、配体图和自排除清单三种表示。构造参数 ``config`` 定义 ``D_m/K``、图层数和 dropout；前向读取语言、图及映射字段，输出 ``IdentityRepresentations``。"""

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
    """对候选和一类身份表示打分，并按配置读取模态统计与采样统计。

    构造参数冻结 ``D_m``、dropout 和两个特征开关。前向接收候选 ``(B,D_m)``、
    身份 ``(B,I,D_m)``、可选手动统计 ``(B,12)``、身份类别概率 ``(B,I)`` 和
    采样统计 ``(B,4)``，返回 ``(B,I)``。
    """

    def __init__(
        self,
        model_dim: int,
        dropout: float,
        use_manual_features: bool,
        use_sampling_features: bool,
    ) -> None:
        super().__init__()
        self.use_manual_features = use_manual_features
        self.use_sampling_features = use_sampling_features
        self.candidate = nn.Linear(model_dim, model_dim, bias=False)
        self.identity = nn.Linear(model_dim, model_dim, bias=False)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        compatibility_dim = (
            model_dim * 4
            + (CANDIDATE_MANUAL_FEATURE_DIM + 1 if use_manual_features else 0)
            + (CANDIDATE_SAMPLING_FEATURE_DIM if use_sampling_features else 0)
        )
        self.compatibility = nn.Sequential(
            nn.LayerNorm(compatibility_dim),
            nn.Linear(compatibility_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def forward(
        self,
        candidate: torch.Tensor,
        identities: torch.Tensor,
        normalized_manual_features: torch.Tensor | None,
        identity_class_probability: torch.Tensor | None,
        normalized_sampling_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """余弦点积沿 ``D_m`` 求和，再拼接可选 12 维手动统计、4 维采样统计和身份类别概率计算残差。"""

        candidate_projected = F.normalize(self.candidate(candidate), dim=-1)
        identity_projected = F.normalize(self.identity(identities), dim=-1)
        cosine = torch.einsum("bd,bid->bi", candidate_projected, identity_projected)
        candidate_expanded = candidate[:, None, :].expand_as(identities)
        compatibility_parts = [
            candidate_expanded,
            identities,
            candidate_expanded * identities,
            torch.abs(candidate_expanded - identities),
        ]
        if self.use_manual_features:
            if normalized_manual_features is None or identity_class_probability is None:
                raise ValueError("启用手动特征的身份打分必须提供候选统计和身份类别概率。")
            if normalized_manual_features.shape != (
                candidate.shape[0],
                CANDIDATE_MANUAL_FEATURE_DIM,
            ):
                raise ValueError("身份打分的候选手动特征形状错误。")
            if identity_class_probability.shape != identities.shape[:2]:
                raise ValueError("身份类别概率必须与身份轴对齐。")
            compatibility_parts.extend(
                (
                    normalized_manual_features[:, None, :].expand(
                        -1,
                        identities.shape[1],
                        -1,
                    ),
                    identity_class_probability[..., None],
                )
            )
        elif normalized_manual_features is not None or identity_class_probability is not None:
            raise ValueError("未启用手动特征的身份打分不得接收额外特征。")
        if self.use_sampling_features:
            if normalized_sampling_features is None or normalized_sampling_features.shape != (
                candidate.shape[0],
                CANDIDATE_SAMPLING_FEATURE_DIM,
            ):
                raise ValueError("启用采样特征的身份打分必须提供与候选轴对齐的 (B,4)。")
            compatibility_parts.append(
                normalized_sampling_features[:, None, :].expand(
                    -1,
                    identities.shape[1],
                    -1,
                )
            )
        elif normalized_sampling_features is not None:
            raise ValueError("未启用采样特征的身份打分不得接收采样统计。")
        residual = self.compatibility(torch.cat(compatibility_parts, dim=-1))[..., 0]
        return cosine * self.log_scale.exp().clamp_max(100.0) + residual


class IdentityScorer(nn.Module):
    """合并三条候选—身份分支。

    构造参数包括模型宽度 ``D_m``、dropout、12 维手动统计开关和 4 维采样统计开关。
    前向接收候选 ``(B,D_m)``、三组身份 ``(B,I,D_m)``、bool 掩码 ``(B,I)``、
    六分类 ``(B,6)``、身份类别 ``(B,I)``、候选手动统计 ``(B,12)`` 和采样统计
    ``(B,4)``，返回 ``(B,I)``。native 始终参与，graph/inventory 出口门从零开始。
    """

    def __init__(
        self,
        model_dim: int,
        dropout: float,
        use_manual_features: bool,
        use_sampling_features: bool,
    ) -> None:
        super().__init__()
        self.use_manual_features = use_manual_features
        self.use_sampling_features = use_sampling_features
        self.native = PairScorer(
            model_dim, dropout, use_manual_features, use_sampling_features
        )
        self.graph = PairScorer(
            model_dim, dropout, use_manual_features, use_sampling_features
        )
        self.inventory = PairScorer(
            model_dim, dropout, use_manual_features, use_sampling_features
        )
        self.graph_gate = nn.Parameter(torch.zeros(()))
        self.inventory_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        candidate: torch.Tensor,
        identities: IdentityRepresentations,
        identity_mask: torch.Tensor,
        class_logits: torch.Tensor,
        identity_class_index: torch.Tensor,
        candidate_manual_features: torch.Tensor,
        candidate_sampling_features: torch.Tensor,
    ) -> torch.Tensor:
        """合并三条分支；按开关拼接 12 维手动统计、4 维采样统计和切断梯度的身份类别概率。"""

        if candidate.shape[0] != identity_mask.shape[0]:
            raise ValueError("候选和身份掩码的批次轴不一致。")
        if (
            identities["graph"].shape != identities["native"].shape
            or identities["inventory"].shape != identities["native"].shape
        ):
            raise ValueError("三种身份表示必须具有同一 (B,I,D_m) 形状。")
        if identities["native"].shape[:2] != identity_mask.shape:
            raise ValueError("三种身份表示必须与 (B,I) 掩码对齐。")
        if identity_mask.shape[1] == 0:
            return candidate.new_empty((candidate.shape[0], 0))
        normalized_manual: torch.Tensor | None = None
        normalized_sampling: torch.Tensor | None = None
        identity_class_probability: torch.Tensor | None = None
        if self.use_manual_features:
            if class_logits.shape != (candidate.shape[0], len(CLASS_NAMES)):
                raise ValueError("class_logits 必须是与候选轴对齐的 (B,6)。")
            if identity_class_index.shape != identity_mask.shape:
                raise ValueError("identity_class_index 必须与 identity_mask 同形。")
            valid_class = identity_class_index[identity_mask]
            if torch.any(valid_class <= 0) or torch.any(valid_class >= len(CLASS_NAMES)):
                raise ValueError("有效身份的 class_index 必须属于五类非背景下标。")
            if torch.any(identity_class_index[~identity_mask] != -1):
                raise ValueError("补零身份的 class_index 必须为 -1。")
            normalized_manual = normalize_candidate_manual_features(
                candidate_manual_features
            )
            safe_class = identity_class_index.clamp_min(0)
            identity_class_probability = torch.softmax(class_logits, dim=-1).gather(
                1,
                safe_class,
            ).detach()
            identity_class_probability = identity_class_probability * identity_mask.to(
                dtype=identity_class_probability.dtype
            )
        if self.use_sampling_features:
            normalized_sampling = normalize_candidate_sampling_features(
                candidate_sampling_features
            )
        logits = self.native(
            candidate,
            identities["native"],
            normalized_manual,
            identity_class_probability,
            normalized_sampling,
        )
        logits = logits + self.graph_gate * self.graph(
            candidate,
            identities["graph"],
            normalized_manual,
            identity_class_probability,
            normalized_sampling,
        )
        inventory_increment = self.inventory(
            candidate,
            identities["inventory"],
            normalized_manual,
            identity_class_probability,
            normalized_sampling,
        )
        has_other_identity = identity_mask.sum(dim=1, keepdim=True) > 1
        inventory_increment = inventory_increment * has_other_identity.to(
            dtype=inventory_increment.dtype
        )
        logits = logits + self.inventory_gate * inventory_increment
        return logits.masked_fill(~identity_mask, torch.finfo(logits.dtype).min)


# ================================================================================================


class ClosedSetMatcher(nn.Module):
    """在候选六分类主干后执行 PDB 内可配置前景范围的闭集身份分类。构造参数 ``config`` 定义候选、身份与打分模块；前向输入 ``CandidateBatch``，输出 ``FullMatcherOutput``。"""

    def __init__(self, config: MatcherModelConfig) -> None:
        super().__init__()
        self.config = config
        self.candidate_encoder = CandidateEncoder(config)
        self.identity_encoder = IdentityEncoder(config)
        self.identity_scorer = IdentityScorer(
            config.model_dim,
            config.dropout,
            config.use_manual_features,
            config.use_sampling_features,
        )

    def forward(self, batch: CandidateBatch) -> FullMatcherOutput:
        """编码候选与 PDB 内身份，并返回 ``FullMatcherOutput``。

        候选分支读取体素、A/P/V/PP 及配置启用的 12 维模态统计和 4 维采样统计；
        身份分支另读语言、图与清单。身份打分按相同开关读取两组候选统计，最终仅
        通过 ``(B,I)`` 配对分数与候选表示交互。
        """

        candidate_output = self.candidate_encoder(batch)
        identities = self.identity_encoder(batch)
        identity_logits = self.identity_scorer(
            candidate_output["candidate_embedding"],
            identities,
            batch["identity_mask"],
            candidate_output["class_logits"],
            batch["identity_class_index"],
            batch["candidate_manual_features"],
            batch["candidate_sampling_features"],
        )
        output = dict(candidate_output)
        output["identity_logits"] = identity_logits
        output["identity_native"] = identities["native"]
        output["identity_graph"] = identities["graph"]
        output["identity_inventory"] = identities["inventory"]
        return cast(FullMatcherOutput, output)
