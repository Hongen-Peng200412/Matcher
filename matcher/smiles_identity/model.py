"""组合候选编码、配体身份编码与 PDB 内闭集身份打分.

候选分支先把一个 PDB 中的局部体素与 A、P、V、PP 四类 token 编码为单个候选表示.
身份分支把精确 SMILES 的语言向量、化学图和同 PDB 身份清单分别编码; 身份字符串只
负责数据侧分组与诊断, 不参与本模块的数值前向. 打分器随后分别计算语言
基线、配体图增量与身份清单增量, 并只在最终 identity_logits 中屏蔽补零身份槽.

形状符号如下.

- B: 当前候选批次大小.
- I: 当前批次补齐后的最大身份数.
- D_m: 统一的模型表示宽度.
- N: 当前批次全部 SMILES 图连接后的原子数.
- E: 当前批次全部 SMILES 图连接后的有向边数.
- G: 当前批次连接后的 SMILES 图数.
- K: 多头注意力的头数.
- F_e: SMILES 有向边特征宽度; 正式数据契约固定为 9.

本模块不决定 oracle、detached 或 joint 身份监督模式; 这些模式属于损失层契约.
"""

from __future__ import annotations

import math
from typing import TypedDict, cast

import torch
from torch import nn
from torch.nn import functional as F

from ..model.fusion import normalize_candidate_manual_features, normalize_candidate_sampling_features
from .candidate_encoder import CandidateEncoder, CandidateEncoderOutput
from .contracts import CandidateBatch
from .model_config import MatcherModelConfig
from .schema import (
    CANDIDATE_MANUAL_FEATURE_DIM,
    CANDIDATE_SAMPLING_FEATURE_DIM,
    CLASS_NAMES,
    LANGUAGE_EMBEDDING_DIM,
)


class IdentityRepresentations(TypedDict):
    """同一批精确 SMILES 身份在三条打分分支中的数值表示.

    - native: 浮点张量, (B, I, D_m); 由语言向量投影得到, 有效槽对应顶层 PreparedLigand.smiles, 补零槽为零.
    - graph: 浮点张量, (B, I, D_m); 由 native 查询同一精确 SMILES 的原子 token 得到, 补零槽为零.
    - inventory: 浮点张量, (B, I, D_m); 由 native 查询同一 PDB 内其它有效身份得到, 没有其它身份或属于补零槽时为零.
    """

    native: torch.Tensor
    graph: torch.Tensor
    inventory: torch.Tensor


class IdentityScoreComponents(TypedDict):
    """同一次身份前向产生的三个未屏蔽分数组件.

    - native: 浮点张量, (B, I); 语言表示分支的基线分数, 补零槽可能因配对网络偏置而非零.
    - graph_increment: 浮点张量, (B, I); 配体图分支原始分数乘 graph_gate 后的增量, 补零槽尚未屏蔽.
    - inventory_increment: 浮点张量, (B, I); 身份清单分支分数先按样本的多身份条件归零、再乘 inventory_gate 后的增量, 补零槽尚未屏蔽.

    三个字段的 dtype 由当前数值上下文决定, 设备与本次模型前向一致. 调用方只能在
    identity_mask 为 True 的槽位读取组件; 补零身份表示为零并不保证配对网络的输出为零.
    """

    native: torch.Tensor
    graph_increment: torch.Tensor
    inventory_increment: torch.Tensor


class IdentityScorerOutput(TypedDict):
    """身份打分器的一次正式前向结果.

    - logits: 浮点张量, (B, I); 三个分数组件之和, 有效槽保留数值, 补零槽替换为相应 dtype 的最小有限值.
    - components: IdentityScoreComponents; 同次前向产生的未屏蔽组件.
    - components.native: 浮点张量, (B, I); 语言基线分数.
    - components.graph_increment: 浮点张量, (B, I); 已乘图门的分数增量.
    - components.inventory_increment: 浮点张量, (B, I); 已应用多身份条件且已乘清单门的分数增量.
    """

    logits: torch.Tensor
    components: IdentityScoreComponents


class FullMatcherOutput(CandidateEncoderOutput):
    """候选编码输出与 PDB 内身份打分的合并结果.

    - candidate_embedding: 浮点张量, (B, D_m); 融合候选体素、四模态 token 与配置启用统计的候选表示.
    - class_logits: 浮点张量, (B, 6); 固定 CLASS_NAMES 顺序的六分类未归一化分数.
    - ligand_area_logits: 浮点张量, (B, 48, 48, 48); 配体区域逐体素未归一化分数, 空间轴为 Z/Y/X.
    - binding_area_logits: 浮点张量, (B, 48, 48, 48); 受体结合区域逐体素未归一化分数, 空间轴为 Z/Y/X.
    - voxel_features: 浮点张量, (B, C_v, 48, 48, 48); RAUNet 最高分辨率体素特征.
    - identity_logits: 浮点张量, (B, I); 三个身份分数组件之和, 补零槽为相应 dtype 的最小有限值.
    - identity_score_native: 浮点张量, (B, I); 语言基线分数, 补零槽尚未屏蔽且不得读取.
    - identity_score_graph_increment: 浮点张量, (B, I); 已乘 graph_gate 的配体图增量, 补零槽尚未屏蔽且不得读取.
    - identity_score_inventory_increment: 浮点张量, (B, I); 已应用多身份条件并乘 inventory_gate 的清单增量, 补零槽尚未屏蔽且不得读取.

    四个身份字段都与输入 identity_mask 逐槽对齐; dtype 由当前数值上下文决定, 设备与
    本次模型前向一致.
    """

    identity_logits: torch.Tensor
    identity_score_native: torch.Tensor
    identity_score_graph_increment: torch.Tensor
    identity_score_inventory_increment: torch.Tensor


class GraphMessageLayer(nn.Module):
    """沿有向化学键汇聚消息并更新每个 SMILES 原子 token.

    每条边的消息是源原子投影与边特征投影之和; 同一目标原子的消息按入度取平均,
    再经过逐通道门控、残差和前馈网络. 正式数据把每条化学键展开为正反两个方向.

    - model_dim: 正整数, 统一的原子 token 宽度 D_m; 本类不单独检查正值.
    - edge_dim: 正整数, 有向边特征宽度 F_e; 正式 SMILES 图契约为 9.
    - dropout: 浮点丢弃率; 有效范围由 torch.nn.Dropout 构造过程检查.
    """

    def __init__(self, model_dim: int, edge_dim: int, dropout: float) -> None:
        """建立消息投影、逐通道门、两个残差归一化块和共享丢弃层.

        - model_dim: 原子 token 的输入、消息和输出宽度 D_m.
        - edge_dim: 输入有向边特征的最后一轴宽度 F_e.
        - dropout: 消息残差和前馈残差共用的丢弃概率.

        - self.node_message: 无偏置 Linear; 把源原子 token 从 D_m 投影到 D_m.
        - self.edge_message: 无偏置 Linear; 把 F_e 维边特征投影到 D_m.
        - self.gate: Linear; 把原子 token 与平均入边消息的 2D_m 拼接映射成 D_m 个门值.
        - self.message_output: Linear; 把激活后的平均入边消息投影为 D_m 维更新, 包含偏置.
        - self.message_norm: LayerNorm; 归一化消息残差和.
        - self.feed_forward: Sequential; D_m 到 3D_m 再回到 D_m 的 SiLU、Dropout 前馈网络.
        - self.feed_forward_norm: LayerNorm; 归一化前馈残差和.
        - self.dropout: Dropout; 共用于消息更新和前馈更新进入各自残差前的丢弃.
        """

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
        """按目标原子平均有向边消息并返回更新后的原子 token.

        - nodes: 浮点张量, (N, D_m); 连接后全部图的原子 token. 本函数不单独检查二维形状或 D_m.
        - edge_index: 整数张量, (2, E); 第 0 行是源原子下标, 第 1 行是目标原子下标.
        - edge_features: 浮点张量, (E, F_e); 与 edge_index 的有向边轴逐项对齐.
        - return: 浮点张量, (N, D_m); 保留原子轴顺序, dtype 由输入与当前数值上下文决定.

        本函数只显式检查 edge_index 的两轴形状与 E 对齐; 节点下标范围、F_e 和 D_m
        由张量索引或线性层在消费时约束. E=0 时汇聚消息为零, 但 message_output 含偏置,
        后续门控与前馈网络仍可能产生非零更新, 因此空边图并不使整层成为恒等映射.
        """

        if edge_index.shape != (2, edge_features.shape[0]):
            raise ValueError("graph_edge_index 必须是与边特征对齐的 (2,E)。")
        # 浮点 (N, D_m); 按目标原子保存入边消息之和, 初始化为零并继承 nodes 的设备与 dtype.
        aggregated = torch.zeros_like(nodes)
        if edge_index.shape[1] > 0:
            # 整数 (E,) 与 (E,); 分别保存每条有向边的源原子下标和目标原子下标.
            source, target = edge_index
            # 浮点 (E, D_m); 融合源原子 token 与同一条边的 F_e 维特征; 正式图调用的 F_e=9.
            message = self.node_message(nodes[source]) + self.edge_message(edge_features)
            # 自动混合精度可能改变消息 dtype; index_add_ 不自动提升, 故先转为累加器 dtype.
            aggregated.index_add_(0, target, message.to(dtype=aggregated.dtype))
            # 浮点 (N,); 每个原子的入边数量, 与 nodes 共设备和 dtype, 零入度由后续下限 1 保护.
            degree = torch.zeros(nodes.shape[0], device=nodes.device, dtype=nodes.dtype)
            degree.index_add_(0, target, torch.ones_like(target, dtype=nodes.dtype))
            # 浮点 (N, D_m); 把入边消息之和变为按目标原子的入度平均, 零入度原子仍为零.
            aggregated = aggregated / degree.clamp_min(1.0)[:, None]
        # 浮点 (N, D_m), 有限精度 Sigmoid 输出位于 [0, 1]; 逐原子、逐通道调节消息增量.
        gate = torch.sigmoid(self.gate(torch.cat((nodes, aggregated), dim=-1)))
        # 浮点 (N, D_m); 消息投影含偏置, aggregated 为零时也不保证 update 为零.
        update = self.message_output(F.silu(aggregated)) * gate
        # 浮点 (N, D_m); 完成消息残差后继续作为前馈网络的输入.
        nodes = self.message_norm(nodes + self.dropout(update))
        return self.feed_forward_norm(nodes + self.dropout(self.feed_forward(nodes)))


class LigandGraphEncoder(nn.Module):
    """把连接后的 SMILES 图字段编码为保留原子轴的图 token.

    元素和电荷分别查表, 四种环大小标志经无偏置线性层投影; 三项相加并
    归一化后, 依次经过若干 GraphMessageLayer. 本类不在图内或图间做 mean/max 池化.

    - model_dim: 正整数, 输出原子 token 宽度 D_m; 本类不单独检查正值.
    - layer_count: 整数, 顺序执行的图消息层数; 正式配置要求为正, 直接传入零或负整数会产生空层列表.
    - dropout: 浮点丢弃率, 传给每个 GraphMessageLayer.
    """

    def __init__(self, model_dim: int, layer_count: int, dropout: float) -> None:
        """建立两张离散嵌入表、环标志投影和图消息层列表.

        - model_dim: 所有离散嵌入、连续投影和输出 token 的宽度 D_m.
        - layer_count: GraphMessageLayer 的数量.
        - dropout: 每个图消息层使用的丢弃概率.

        - self.element: Embedding; 128 个原子序数编码到 D_m.
        - self.charge: Embedding; 平移后的 33 个形式电荷编码到 D_m.
        - self.atom_ring: 无偏置 Linear; 四种环大小标志投影到 D_m.
        - self.input_norm: LayerNorm; 归一化两项嵌入与环投影的逐原子和.
        - self.layers: ModuleList; 按构造顺序保存 layer_count 个 GraphMessageLayer.
        """

        super().__init__()
        self.element = nn.Embedding(128, model_dim)
        self.charge = nn.Embedding(33, model_dim)
        self.atom_ring = nn.Linear(4, model_dim, bias=False)
        self.input_norm = nn.LayerNorm(model_dim)
        self.layers = nn.ModuleList(
            GraphMessageLayer(model_dim, edge_dim=9, dropout=dropout)
            for _ in range(layer_count)
        )

    def forward(self, batch: CandidateBatch) -> torch.Tensor:
        """融合原子离散字段和连续字段, 再沿全部有向边更新 token.

        - batch: CandidateBatch; 本函数消费下列连接图字段.
        - batch.graph_element: 整数张量, (N,); 原子序数, 显式要求属于 [0, 128).
        - batch.graph_charge: 整数张量, (N,); 形式电荷, 显式要求属于 [-16, 16].
        - batch.graph_atom_in_ring: 浮点张量, (N, 4); 依次标记 3、4、5、6 元环.
        - batch.graph_edge_index: 整数张量, (2, E); 连接后有向边端点, 由每层 GraphMessageLayer 消费.
        - batch.graph_edge_features: 浮点张量, (E, 9); 五类键型与四类环标志.
        - return: 浮点张量, (N, D_m); 保留连接后的原子顺序, 不进行图级池化.

        本函数显式检查三类离散编码的取值范围, 但不独立核验所有字段的 dtype、维数和
        轴间对齐; 正式 CandidateBatch 由 Dataset 与 collate 契约提供这些条件.
        """

        # 整数 (N,); 当前批次连接后的原子序数, 原子轴与另外三项节点字段一致.
        element = batch["graph_element"]
        # 整数 (N,); 当前批次连接后的形式电荷, 后续整体平移 16 以查 33 项嵌入表.
        charge = batch["graph_charge"]
        if torch.any(element < 0) or torch.any(element >= self.element.num_embeddings):
            raise ValueError("SMILES 图 element 超出 [0,128) 编码范围。")
        if torch.any(charge < -16) or torch.any(charge > 16):
            raise ValueError("SMILES 图 charge 超出 [-16,16] 编码范围。")
        atom_ring = batch["graph_atom_in_ring"]
        if atom_ring.shape != (len(element), 4):
            raise ValueError("graph_atom_in_ring 必须是与原子轴对齐的 (N,4)。")
        # 浮点 (N, D_m); 三类原子表示逐项相加后归一化, 作为第一层图消息的输入.
        nodes = self.input_norm(
            self.element(element)
            + self.charge(charge + 16)
            + self.atom_ring(atom_ring)
        )
        for layer in self.layers:
            # 浮点 (N, D_m); 每层保留原子轴并复用同一连接图的有向边及边特征.
            nodes = layer(nodes, batch["graph_edge_index"], batch["graph_edge_features"])
        return nodes


class GraphIdentityAttention(nn.Module):
    """以每个精确 SMILES 的语言表示查询同一化学图的原子 token.

    一个图产生一个 query, 每个原子产生 key 和 value. softmax 仅在同一图的原子轴内、
    分头归一化; 得到的图向量写回对应样本的身份槽. 正式 IdentityEncoder 先保证每个有效
    身份恰好对应一张图, 因此图到身份槽的映射没有重复.

    - model_dim: 正整数, query、key、value 与输出宽度 D_m.
    - head_count: 正整数, 注意力头数 K; 必须整除 model_dim.
    - dropout: 浮点丢弃率; 作用于已归一化的原子注意力权重, 丢弃后不再次归一化.
    """

    def __init__(self, model_dim: int, head_count: int, dropout: float) -> None:
        """建立图级 query、原子级 key/value、输出投影和归一化层.

        - model_dim: 正整数, 统一模型宽度 D_m.
        - head_count: 正整数, 注意力头数 K; 要求 model_dim 能被它整除.
        - dropout: 原子注意力权重的丢弃概率.

        - self.model_dim: 正整数 D_m; 前向核验原子 token 宽度并恢复连接头的图表示.
        - self.head_count: 正整数 K; 图内注意力头数.
        - self.head_dim: 正整数 D_m/K; 每头 query、key 和 value 宽度.
        - self.query: Linear; 把每张图对应的 native 身份表示投影成分头 query.
        - self.key: 无偏置 Linear; 把每个原子 token 投影成分头 key.
        - self.value: 无偏置 Linear; 把每个原子 token 投影成分头 value.
        - self.output: Linear; 把连接后的 K 头图上下文投影回 D_m.
        - self.norm: LayerNorm; 归一化每张图的 D_m 维输出.
        - self.dropout: Dropout; 作用于图内已经归一化的原子注意力权重.
        """

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
        """在每张图内部对原子注意力归一化并写回对应身份槽.

        - native: 浮点张量, (B, I, D_m); 语言分支的身份表示, 补零槽为零.
        - nodes: 浮点张量, (N, D_m); 全部 SMILES 图连接后的原子 token.
        - node_graph_index: 整数张量, (N,); 每个原子所属的连接图下标, 取值应属于 [0, G).
        - graph_to_sample: 整数张量, (G,); 每张图所属的样本下标, 取值应属于 [0, B).
        - graph_to_identity: 整数张量, (G,); 每张图在所属样本中的身份槽, 取值应属于 [0, I).
        - return: 浮点张量, (B, I, D_m); 有图槽保存图注意力表示, 其余槽为零.

        G=0 时函数立即返回与 native 同形的零张量, 且不会执行其后的形状、dtype 或范围
        检查. G>0 时显式检查主要维数与索引范围, 但不预先检查一图至少一原子, 直至归一化
        分母出现零才失败; 也不检查多个图是否写入同一身份槽. 正式 IdentityEncoder 的
        occupancy 门禁负责保证每个有效槽恰好一张图, 并由正式图数据保证每张图非空.
        """

        # Python 整数 G; 以图到样本映射的长度定义本次连接图数量.
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
        # 浮点 (G, D_m); 按图到样本、身份槽映射选出每张图对应的语言 query.
        graph_query = native[graph_to_sample, graph_to_identity]
        # 浮点 (G, K, D_m/K); 每张图的分头 query.
        query = self.query(graph_query).reshape(graph_count, self.head_count, self.head_dim)
        # 浮点 (N, K, D_m/K); 每个原子的分头 key.
        key = self.key(nodes).reshape(nodes.shape[0], self.head_count, self.head_dim)
        # 浮点 (N, K, D_m/K); 每个原子的分头 value.
        value = self.value(nodes).reshape(nodes.shape[0], self.head_count, self.head_dim)
        # 浮点 (N, K); 每个原子与所属图 query 的缩放点积分数.
        score = (query[node_graph_index] * key).sum(dim=-1) / math.sqrt(self.head_dim)
        # 整数 (N, K); 把每个原子的图下标广播到各注意力头, 供分组规约使用.
        scatter_index = node_graph_index[:, None].expand_as(score)
        # 浮点 (G, K); 每图每头的最大原子分数, 用于稳定 softmax.
        group_max = score.new_full((graph_count, self.head_count), torch.finfo(score.dtype).min)
        group_max.scatter_reduce_(0, scatter_index, score, reduce="amax", include_self=True)
        # 浮点 (N, K), 正值; 减去所属图最大值后的指数权重.
        unnormalized = torch.exp(score - group_max[node_graph_index])
        # 浮点 (G, K); 每图每头全部原子指数权重之和.
        denominator = score.new_zeros((graph_count, self.head_count))
        denominator.index_add_(0, node_graph_index, unnormalized)
        if torch.any(denominator == 0):
            raise ValueError("每张 SMILES 图必须至少含一个原子 token。")
        # 浮点 (N, K); 图内归一化后执行 dropout, 丢弃后不重新把每图权重和缩放回 1.
        weight = self.dropout(unnormalized / denominator[node_graph_index])
        # 浮点 (N, K, D_m/K); 每个原子的分头 value 乘同位置注意力权重.
        weighted_value = weight[..., None] * value
        # 浮点 (G, K, D_m/K); 按图累加加权原子值, 继承 weighted_value 的 dtype.
        context = weighted_value.new_zeros((graph_count, self.head_count, self.head_dim))
        context.index_add_(0, node_graph_index, weighted_value)
        # 浮点 (G, D_m); 拼接注意力头, 经输出投影与 LayerNorm 得到每张图的表示.
        graph_vector = self.norm(self.output(context.reshape(graph_count, self.model_dim)))
        # 浮点 (B, I, D_m); 先全零, 再把每张图表示写入对应身份槽.
        result = torch.zeros_like(native)
        result[graph_to_sample, graph_to_identity] = graph_vector
        return result


class SelfExcludingInventoryAttention(nn.Module):
    """让每个有效身份查询同一 PDB 内除自身以外的其它身份.

    query 与 key 轴都对应批内身份槽 I. 注意力同时屏蔽补零槽和对角线, 所以单身份
    样本没有可用 key, 并得到严格零清单表示; 多个身份的样本只读取其它有效身份.

    - model_dim: 正整数, 输入、query、key、value 和输出宽度 D_m.
    - head_count: 正整数, 注意力头数 K; 必须整除 model_dim.
    - dropout: 浮点丢弃率; 作用于归一化权重, 丢弃后不再次归一化.
    """

    def __init__(self, model_dim: int, head_count: int, dropout: float) -> None:
        """建立身份清单的 query、key、value、输出投影与归一化层.

        - model_dim: 正整数, 统一模型宽度 D_m.
        - head_count: 正整数, 注意力头数 K; 要求 model_dim 能被它整除.
        - dropout: 身份间注意力权重的丢弃概率.

        - self.model_dim: 正整数 D_m; 恢复连接头的身份清单表示.
        - self.head_count: 正整数 K; 身份清单注意力头数.
        - self.head_dim: 正整数 D_m/K; 每头 query、key 和 value 宽度.
        - self.query: Linear; 逐身份投影分头 query.
        - self.key: 无偏置 Linear; 逐身份投影分头 key.
        - self.value: 无偏置 Linear; 逐身份投影分头 value.
        - self.output: Linear; 把连接后的 K 头清单上下文投影回 D_m.
        - self.norm: LayerNorm; 归一化每个身份槽的 D_m 维输出.
        - self.dropout: Dropout; 作用于已经归一化的身份间注意力权重.
        """

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
        """沿其它身份轴归一化注意力并返回自排除清单上下文.

        - identities: 浮点张量, (B, I, D_m); 通常是语言分支的 native 表示.
        - identity_mask: bool 张量, (B, I); True 表示真实身份槽, False 表示补零槽.
        - return: 浮点张量, (B, I, D_m); 有效且存在其它身份的槽保存上下文, 其余槽严格为零.

        本函数显式检查 identities 的三维前缀与 identity_mask 对齐, 并要求掩码为 bool,
        但不单独检查最后一轴等于构造时 model_dim; 线性层会在消费时约束该宽度. I=0
        在这些检查后原样返回空身份张量. softmax 作用于最后一个 key 身份轴; 对无有效 key
        的 query, 先通过显式掩码与再归一化得到零权重, 避免全屏蔽 softmax 的伪概率.
        """

        if identities.ndim != 3 or identities.shape[:2] != identity_mask.shape:
            raise ValueError("identities 必须为 (B,I,D_m)，identity_mask 必须为 (B,I)。")
        if identity_mask.dtype != torch.bool:
            raise TypeError("identity_mask 必须是 bool。")
        # Python 整数 B、I; 分别来自身份表示的样本轴和补齐身份轴.
        batch_size, identity_count, _ = identities.shape
        if identity_count == 0:
            return identities
        # 浮点 (B, K, I, D_m/K); 每个身份的分头 query.
        query = self._heads(self.query(identities))
        # 浮点 (B, K, I, D_m/K); 每个身份的分头 key.
        key = self._heads(self.key(identities))
        # 浮点 (B, K, I, D_m/K); 每个身份的分头 value.
        value = self._heads(self.value(identities))
        # 浮点 (B, K, I, I); 倒数第二轴为 query 身份, 最后一轴为 key 身份.
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # bool (I, I); 对角线 False, 阻止身份读取自身, 其余位置 True.
        off_diagonal = ~torch.eye(identity_count, device=identities.device, dtype=torch.bool)
        # bool (B, 1, I, I); 同时要求 query、key 都有效, 头轴为 1 并在运算中广播到 K.
        valid = identity_mask[:, None, :, None] & identity_mask[:, None, None, :]
        valid = valid & off_diagonal[None, None, :, :]
        score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
        # 浮点 (B, K, I, I); 屏蔽后沿 key 轴重新归一化, 无可用 key 的 query 保持全零.
        weight = torch.softmax(score, dim=-1) * valid.to(dtype=score.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        # 浮点 (B, K, I, D_m/K); dropout 作用于权重后读取其它身份 value, 不做二次归一化.
        context = torch.matmul(self.dropout(weight), value)
        # 浮点 (B, I, D_m); 恢复身份轴在前、拼接 K 个注意力头.
        context = context.transpose(1, 2).reshape(batch_size, identity_count, self.model_dim)
        # bool (B, I, 1); 标记每个 query 是否至少存在一个其它有效身份.
        has_other = valid.any(dim=-1).transpose(1, 2)
        # 浮点 (B, I, D_m); 输出投影含偏置, 故在投影后显式按 has_other 归零.
        result = self.norm(self.output(context)) * has_other.to(context.dtype)
        return result * identity_mask[..., None].to(dtype=result.dtype)

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        """把模型宽度拆为注意力头并把头轴移到身份轴之前.

        - values: 浮点张量, (B, I, D_m); 最后一轴必须可重排为 K 个 D_m/K 通道.
        - return: 浮点张量, (B, K, I, D_m/K); 仅重排形状和轴, 不复制语义.
        """

        # Python 整数 B、I; 用于先重排为 (B, I, K, D_m/K), 再交换 I 与 K.
        batch_size, identity_count, _ = values.shape
        return values.reshape(
            batch_size,
            identity_count,
            self.head_count,
            self.head_dim,
        ).transpose(1, 2)


class IdentityEncoder(nn.Module):
    """为每个 PDB 内精确 SMILES 身份生成语言、配体图和自排除清单表示.

    语言向量先投影为 native. 图分支保留 SMILES 图原子轴, 用 native 查询自身图;
    清单分支让 native 查询同一 PDB 的其它 native. 图 occupancy 门禁要求每个有效身份槽
    恰好一张图且补零槽无图, 但本类不读取 identity_smiles 或其它身份字符串.

    - config: MatcherModelConfig; 读取 model_dim、graph_layers、attention_heads 和 dropout 构造三个分支.
    """

    def __init__(self, config: MatcherModelConfig) -> None:
        """建立语言投影、原子图编码、图查询与身份清单查询模块.

        - config: MatcherModelConfig; 本构造函数直接读取其中的模型宽度、图层数、注意力头数和丢弃率, 不重新执行 dataclass 的 __post_init__.

        - self.language: Sequential; 把每个 768 维语言向量归一化、投影并激活成 D_m 维 native 表示.
        - self.graph_encoder: LigandGraphEncoder; 把连接后的 SMILES 图字段编码为逐原子 token.
        - self.graph_attention: GraphIdentityAttention; 让 native 表示查询同一身份的原子 token.
        - self.inventory_attention: SelfExcludingInventoryAttention; 让 native 表示查询同 PDB 的其它身份.
        """

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
        """编码精确 SMILES 身份并核验有效身份槽与化学图的一一对应关系.

        - batch: CandidateBatch; 本函数直接或通过图子模块消费下列字段.
        - batch.language_embeddings: 浮点张量, (B, I, 768); 按身份槽补齐的语言模型向量.
        - batch.identity_mask: bool 张量, (B, I); True 是真实精确 SMILES 身份, False 是补零槽.
        - batch.graph_element: 整数张量, (N,); 连接后的原子序数.
        - batch.graph_charge: 整数张量, (N,); 连接后的形式电荷.
        - batch.graph_atom_in_ring: 浮点张量, (N, 4); 四种环大小标志.
        - batch.graph_edge_index: 整数张量, (2, E); 连接后的有向边端点.
        - batch.graph_edge_features: 浮点张量, (E, 9); 五类键型与四类环标志.
        - batch.graph_node_graph_index: 整数张量, (N,); 每个原子所属的图下标.
        - batch.graph_to_sample: 整数张量, (G,); 每张图所属的样本下标.
        - batch.graph_to_identity: 整数张量, (G,); 每张图在所属样本中的身份槽.
        - return: IdentityRepresentations; 三种字段均与 (B, I, D_m) 身份轴对齐.
        - return.native: 浮点张量, (B, I, D_m); 语言投影表示, 补零槽严格为零.
        - return.graph: 浮点张量, (B, I, D_m); 自身配体图注意力表示, 补零槽严格为零.
        - return.inventory: 浮点张量, (B, I, D_m); 自排除身份清单表示, 无其它身份或补零时严格为零.

        本函数不消费 batch.identity_smiles 或候选 metadata 中的字符串; 这些字段只服务于
        数据分组、监督定位和评估诊断.
        """

        # bool (B, I); 决定语言补零槽、图 occupancy 目标和清单注意力的有效身份范围.
        identity_mask = batch["identity_mask"]
        # 浮点 (B, I, D_m); 768 维语言向量经归一化、线性投影和 GELU 后得到的原始结果.
        native = self.language(batch["language_embeddings"])
        # 浮点 (B, I, D_m); 投影层偏置可能使补零输入非零, 故显式按身份掩码再次归零.
        native = native * identity_mask[..., None].to(dtype=native.dtype)
        # Python 整数 G; 以图到样本映射长度定义连接后的 SMILES 图数.
        graph_count = int(batch["graph_to_sample"].shape[0])
        # 整数 (B, I); 累计每个样本、身份槽实际关联的图数, 用于一图一身份门禁.
        occupancy = torch.zeros_like(identity_mask, dtype=torch.long)
        if graph_count > 0:
            # 整数 (G,); 每张图所属的候选样本下标.
            sample = batch["graph_to_sample"]
            # 整数 (G,); 每张图在所属候选样本中的身份槽下标.
            identity = batch["graph_to_identity"]
            if torch.any(sample < 0) or torch.any(sample >= native.shape[0]):
                raise ValueError("graph_to_sample 超出批次轴 [0,B)。")
            if torch.any(identity < 0) or torch.any(identity >= native.shape[1]):
                raise ValueError("graph_to_identity 超出身份轴 [0,I)。")
            occupancy.index_put_((sample, identity), torch.ones_like(sample), accumulate=True)
        if not torch.equal(occupancy, identity_mask.to(dtype=occupancy.dtype)):
            raise ValueError("每个有效身份必须恰好对应一张配体图，补零身份不得含图。")
        # 浮点 (N, D_m); 连接图的逐原子表示, 不做图级 mean/max 池化.
        graph_nodes = self.graph_encoder(batch)
        # 浮点 (B, I, D_m); 每个 native 身份查询自身图原子 token 后的图表示.
        graph = self.graph_attention(
            native,
            graph_nodes,
            batch["graph_node_graph_index"],
            batch["graph_to_sample"],
            batch["graph_to_identity"],
        )
        # 浮点 (B, I, D_m); 每个 native 身份排除自身后读取同 PDB 其它有效身份的表示.
        inventory = self.inventory_attention(native, identity_mask)
        return {"native": native, "graph": graph, "inventory": inventory}


class PairScorer(nn.Module):
    """计算一条候选与身份表示分支的余弦基线与可学习残差.

    候选和身份先分别投影并归一化, 得到缩放余弦分数; 残差网络拼接候选、身份、逐元素
    乘积和绝对差. 配置启用时, 还会拼接 12 维候选模态统计、对应身份类别的候选六分类
    概率, 以及 4 维候选裁剪统计. 该类本身不接收 identity_mask, 也不屏蔽补零身份槽.

    - model_dim: 正整数, 候选与身份表示宽度 D_m; 本类不单独检查.
    - dropout: 浮点丢弃率; 用于残差兼容性网络.
    - use_manual_features: bool; 是否要求并拼接 12 维模态统计与 1 维身份类别概率.
    - use_sampling_features: bool; 是否要求并拼接 4 维裁剪采样统计.
    """

    def __init__(
        self,
        model_dim: int,
        dropout: float,
        use_manual_features: bool,
        use_sampling_features: bool,
    ) -> None:
        """建立余弦投影、可学习尺度和候选与身份兼容性网络.

        - model_dim: 候选表示和单类身份表示的宽度 D_m.
        - dropout: 兼容性残差网络中间层的丢弃概率.
        - use_manual_features: 是否把 12 维已归一化模态统计和身份类别概率加入残差输入.
        - use_sampling_features: 是否把 4 维已归一化裁剪统计加入残差输入.

        - self.use_manual_features: 布尔语义值; 前向控制 12 维模态统计和身份类别概率的存在性.
        - self.use_sampling_features: 布尔语义值; 前向控制 4 维裁剪统计的存在性.
        - self.candidate: 无偏置 Linear; 把候选表示从 D_m 投影到余弦空间 D_m.
        - self.identity: 无偏置 Linear; 把每个身份表示从 D_m 投影到另一余弦空间 D_m.
        - self.log_scale: 标量 Parameter; 保存初值 log(10) 的余弦尺度自然对数.
        - self.compatibility: Sequential; 把四组配对表示与可选统计映射成单个残差分数.
        """

        super().__init__()
        self.use_manual_features = use_manual_features
        self.use_sampling_features = use_sampling_features
        self.candidate = nn.Linear(model_dim, model_dim, bias=False)
        self.identity = nn.Linear(model_dim, model_dim, bias=False)
        # 标量 Parameter; 保存余弦尺度的自然对数, 前向只限制 exp 后的上限 100, 不设下限.
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        # Python 整数; 残差网络输入宽度, 基础四组各 D_m, 并按开关追加 13 维和 4 维.
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
        """对批内每个候选与其 PDB 内每个身份槽计算配对分数.

        - candidate: 浮点张量, (B, D_m); 候选编码器输出的候选级表示.
        - identities: 浮点张量, (B, I, D_m); 当前分支的 native、graph 或 inventory 身份表示.
        - normalized_manual_features: None 或浮点张量, (B, 12); 按正式顺序归一化的 A、P、V、PP 计数、概率均值和概率最大值.
        - identity_class_probability: None 或浮点张量, (B, I); 候选六分类中每个身份类别对应的概率, 调用方已阻断其梯度.
        - normalized_sampling_features: None 或浮点张量, (B, 4); 已固定尺度归一化的裁剪采样统计.
        - return: 浮点张量, (B, I); 缩放余弦分数与兼容性残差之和, 补零槽尚未屏蔽.

        两组可选输入是否必须存在由构造开关决定; 关闭开关时传入非 None 会失败. 函数不
        显式检查 candidate、identities 的全部维数或 B、D_m 对齐, 相关不相容形状由投影、
        expand 或拼接操作暴露. 零身份表示也可能因兼容性网络偏置产生非零分数.
        """

        # 浮点 (B, D_m); 候选经无偏置投影后沿最后一轴做 L2 归一化.
        candidate_projected = F.normalize(self.candidate(candidate), dim=-1)
        # 浮点 (B, I, D_m); 每个身份经另一无偏置投影后沿最后一轴做 L2 归一化.
        identity_projected = F.normalize(self.identity(identities), dim=-1)
        # 浮点 (B, I); 沿模型通道轴计算候选与各身份的余弦点积.
        cosine = torch.einsum("bd,bid->bi", candidate_projected, identity_projected)
        # 浮点 (B, I, D_m); 候选表示仅广播到身份轴, 不物理改变身份语义.
        candidate_expanded = candidate[:, None, :].expand_as(identities)
        # 浮点张量列表; 前四项均为 (B, I, D_m), 依次是候选、身份、乘积和绝对差.
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
            # 追加浮点 (B, I, 12) 和 (B, I, 1); 同一候选统计广播到全部身份, 类别概率逐槽对齐.
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
            # 追加浮点 (B, I, 4); 同一候选的裁剪统计广播到该样本全部身份槽.
            compatibility_parts.append(
                normalized_sampling_features[:, None, :].expand(
                    -1,
                    identities.shape[1],
                    -1,
                )
            )
        elif normalized_sampling_features is not None:
            raise ValueError("未启用采样特征的身份打分不得接收采样统计。")
        # 浮点 (B, I); 拼接宽度由构造时 compatibility_dim 决定, 末层单通道在此移除.
        residual = self.compatibility(torch.cat(compatibility_parts, dim=-1))[..., 0]
        # 浮点 (B, I); 可学习余弦尺度取 exp 后仅截断上限 100, 再与兼容性残差逐槽相加.
        return cosine * self.log_scale.exp().clamp_max(100.0) + residual


class IdentityScorer(nn.Module):
    """合并语言基线、配体图增量和身份清单增量三条配对打分分支.

    三条分支各自拥有独立 PairScorer. graph_gate 与 inventory_gate 是直接乘在分数上的
    可训练标量; 它们没有 sigmoid 或区间夹紧, 训练后可以离开初始化区间. 清单分支另在
    每个样本只有零或一个有效身份时严格归零, 因为这时不存在可读取的其它身份.

    - model_dim: 正整数, 候选与三类身份表示的宽度 D_m.
    - dropout: 浮点丢弃率, 传给三个 PairScorer.
    - use_manual_features: bool; 是否给三个分支追加 12 维模态统计和无梯度身份类别概率.
    - use_sampling_features: bool; 是否给三个分支追加 4 维裁剪采样统计.
    - initial_graph_gate: 可由 float 转换的值, graph_gate 的初值; 本类直接构造时不检查有限性或 [0, 1].
    - initial_inventory_gate: 可由 float 转换的值, inventory_gate 的初值; 本类直接构造时不检查有限性或 [0, 1].

    正式 MatcherModelConfig 检查两个初值属于 [0, 1], 但该约束只针对初始化, 不约束后续
    参数更新. 本类不会用 identity_mask 提前跳过补零槽计算; 只有最终 logits 被屏蔽.
    """

    def __init__(
        self,
        model_dim: int,
        dropout: float,
        use_manual_features: bool,
        use_sampling_features: bool,
        initial_graph_gate: float,
        initial_inventory_gate: float,
    ) -> None:
        """建立三套独立 PairScorer 和两个未经激活函数约束的标量门.

        - model_dim: 候选与身份表示宽度 D_m.
        - dropout: 三个 PairScorer 兼容性网络的丢弃概率.
        - use_manual_features: 是否启用 12 维模态统计与身份类别概率.
        - use_sampling_features: 是否启用 4 维裁剪采样统计.
        - initial_graph_gate: graph_gate 的初始标量; 仅转为 float 后写入 Parameter.
        - initial_inventory_gate: inventory_gate 的初始标量; 仅转为 float 后写入 Parameter.

        - self.use_manual_features: 布尔语义值; 控制类别概率与 12 维模态统计路径.
        - self.use_sampling_features: 布尔语义值; 控制 4 维裁剪统计路径.
        - self.native: PairScorer; 为语言 native 表示计算未门控基线分数.
        - self.graph: PairScorer; 为 SMILES 图表示计算乘图门前的原始分数.
        - self.inventory: PairScorer; 为自排除清单表示计算多身份归零前的原始分数.
        - self.graph_gate: 标量 Parameter; 由 initial_graph_gate 初始化, 前向直接乘图分数且不经激活或夹紧.
        - self.inventory_gate: 标量 Parameter; 由 initial_inventory_gate 初始化, 在多身份条件归零后直接乘清单分数.
        """

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
        self.graph_gate = nn.Parameter(torch.tensor(float(initial_graph_gate)))
        self.inventory_gate = nn.Parameter(torch.tensor(float(initial_inventory_gate)))

    def score_components(
        self,
        candidate: torch.Tensor,
        identities: IdentityRepresentations,
        identity_mask: torch.Tensor,
        class_logits: torch.Tensor,
        identity_class_index: torch.Tensor,
        candidate_manual_features: torch.Tensor,
        candidate_sampling_features: torch.Tensor,
    ) -> IdentityScoreComponents:
        """计算三个未屏蔽分数组件, 并应用图门、清单有效条件和清单门.

        - candidate: 浮点张量, (B, D_m); 候选编码器产生的候选级表示.
        - identities: IdentityRepresentations; 三种表示应具有相同 (B, I, D_m) 形状.
        - identities.native: 浮点张量, (B, I, D_m); 语言投影表示.
        - identities.graph: 浮点张量, (B, I, D_m); 自身 SMILES 图表示.
        - identities.inventory: 浮点张量, (B, I, D_m); 自排除身份清单表示.
        - identity_mask: bool 张量, (B, I); True 为真实身份槽, False 为补零槽.
        - class_logits: 浮点张量, (B, 6); 候选六分类未归一化分数, 仅在启用手动特征时检查和使用.
        - identity_class_index: 整数张量, (B, I); 有效槽属于 [1, 6), 补零槽为 -1, 仅在启用手动特征时检查.
        - candidate_manual_features: 浮点张量, (B, 12); A、P、V、PP 模态统计, 仅在启用手动特征时归一化.
        - candidate_sampling_features: 浮点张量, (B, 4); 裁剪采样统计, 仅在启用采样特征时归一化.
        - return: IdentityScoreComponents; 三个字段均为与身份槽对齐的未屏蔽 (B, I) 分数.
        - return.native: 浮点张量, (B, I); 语言分支基线.
        - return.graph_increment: 浮点张量, (B, I); 图分支原始分数乘 graph_gate.
        - return.inventory_increment: 浮点张量, (B, I); 清单分支先按多身份条件归零、再乘 inventory_gate.

        I=0 时在基础形状检查后直接返回三个同一空张量, 不继续检查类别或可选特征.
        I>0 时只在相应开关开启时读取附加特征. 三个返回组件的补零槽均未屏蔽, 可能因
        PairScorer 偏置而非零; 调用方必须用 identity_mask 限定可读槽位.
        """

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
            # 浮点 (B, 0); 复用于三个键, 因为空张量没有可读元素且设备、dtype 与 candidate 一致.
            empty = candidate.new_empty((candidate.shape[0], 0))
            return {
                "native": empty,
                "graph_increment": empty,
                "inventory_increment": empty,
            }
        # None 或浮点 (B, 12); 仅在 use_manual_features=True 时填入已归一化模态统计.
        normalized_manual: torch.Tensor | None = None
        # None 或浮点 (B, 4); 仅在 use_sampling_features=True 时填入已归一化裁剪统计.
        normalized_sampling: torch.Tensor | None = None
        # None 或浮点 (B, I); 仅在手动特征分支中填入无梯度的候选身份类别概率.
        identity_class_probability: torch.Tensor | None = None
        if self.use_manual_features:
            if class_logits.shape != (candidate.shape[0], len(CLASS_NAMES)):
                raise ValueError("class_logits 必须是与候选轴对齐的 (B,6)。")
            if identity_class_index.shape != identity_mask.shape:
                raise ValueError("identity_class_index 必须与 identity_mask 同形。")
            # 整数一维张量; 依 identity_mask 压缩出的全部真实身份类别下标.
            valid_class = identity_class_index[identity_mask]
            if torch.any(valid_class <= 0) or torch.any(valid_class >= len(CLASS_NAMES)):
                raise ValueError("有效身份的 class_index 必须属于五类非背景下标。")
            if torch.any(identity_class_index[~identity_mask] != -1):
                raise ValueError("补零身份的 class_index 必须为 -1。")
            # 浮点 (B, 12); 仅四个 token 计数做 log1p, 其余概率统计保持原尺度.
            normalized_manual = normalize_candidate_manual_features(
                candidate_manual_features
            )
            # 整数 (B, I); 把补零槽的 -1 临时变成背景下标 0, 避免 gather 使用负下标.
            safe_class = identity_class_index.clamp_min(0)
            # 浮点 (B, I); 从候选六分类概率中逐槽选取身份类别, detach 阻断此辅助路径梯度.
            identity_class_probability = torch.softmax(class_logits, dim=-1).gather(
                1,
                safe_class,
            ).detach()
            identity_class_probability = identity_class_probability * identity_mask.to(
                dtype=identity_class_probability.dtype
            )
        if self.use_sampling_features:
            # 浮点 (B, 4); L2 除以 24*sqrt(3), Linf 除以 24, 边界标志和保留率保持原值.
            normalized_sampling = normalize_candidate_sampling_features(
                candidate_sampling_features
            )
        # 浮点 (B, I); 语言 native 分支的未屏蔽配对基线.
        native = self.native(
            candidate,
            identities["native"],
            normalized_manual,
            identity_class_probability,
            normalized_sampling,
        )
        # 浮点 (B, I); 图分支原始配对分数直接乘未夹紧的可训练 graph_gate.
        graph_increment = self.graph_gate * self.graph(
            candidate,
            identities["graph"],
            normalized_manual,
            identity_class_probability,
            normalized_sampling,
        )
        # 浮点 (B, I); 清单 PairScorer 的原始未屏蔽输出, 单身份样本此时仍可能非零.
        inventory_increment = self.inventory(
            candidate,
            identities["inventory"],
            normalized_manual,
            identity_class_probability,
            normalized_sampling,
        )
        # bool (B, 1); 每个样本是否至少有两个有效身份, 沿身份轴广播到所有槽.
        has_other_identity = identity_mask.sum(dim=1, keepdim=True) > 1
        inventory_increment = inventory_increment * has_other_identity.to(
            dtype=inventory_increment.dtype
        )
        # 浮点 (B, I); 先保证零/单身份样本严格为零, 再乘未夹紧的可训练 inventory_gate.
        inventory_increment = self.inventory_gate * inventory_increment
        return {
            "native": native,
            "graph_increment": graph_increment,
            "inventory_increment": inventory_increment,
        }

    def forward(
        self,
        candidate: torch.Tensor,
        identities: IdentityRepresentations,
        identity_mask: torch.Tensor,
        class_logits: torch.Tensor,
        identity_class_index: torch.Tensor,
        candidate_manual_features: torch.Tensor,
        candidate_sampling_features: torch.Tensor,
    ) -> IdentityScorerOutput:
        """计算三个分数组件, 并在它们的和上屏蔽补零身份槽.

        - candidate: 浮点张量, (B, D_m); 候选级表示.
        - identities: IdentityRepresentations; 三种表示均应为 (B, I, D_m).
        - identities.native: 浮点张量, (B, I, D_m); 语言投影表示.
        - identities.graph: 浮点张量, (B, I, D_m); 自身配体图表示.
        - identities.inventory: 浮点张量, (B, I, D_m); 自排除身份清单表示.
        - identity_mask: bool 张量, (B, I); 最终分数的有效身份掩码.
        - class_logits: 浮点张量, (B, 6); 候选六分类分数, 手动特征分支开启时用于身份类别概率.
        - identity_class_index: 整数张量, (B, I); 真实槽的五类前景下标与补零槽 -1.
        - candidate_manual_features: 浮点张量, (B, 12); 候选模态统计, 关闭手动特征时不参与计算.
        - candidate_sampling_features: 浮点张量, (B, 4); 候选裁剪统计, 关闭采样特征时不参与计算.
        - return: IdentityScorerOutput; 包含已屏蔽总分与同次未屏蔽组件.
        - return.logits: 浮点张量, (B, I); 三个组件之和, 补零槽替换为 dtype 最小有限值.
        - return.components: IdentityScoreComponents; 未屏蔽的三个组件.
        - return.components.native: 浮点张量, (B, I); 语言基线.
        - return.components.graph_increment: 浮点张量, (B, I); 已乘 graph_gate 的图增量.
        - return.components.inventory_increment: 浮点张量, (B, I); 已应用多身份条件并乘 inventory_gate 的清单增量.
        """

        # IdentityScoreComponents; 同一次调用得到的三个 (B, I) 未屏蔽组件.
        components = self.score_components(
            candidate,
            identities,
            identity_mask,
            class_logits,
            identity_class_index,
            candidate_manual_features,
            candidate_sampling_features,
        )
        return {
            "logits": self._masked_logits(components, identity_mask),
            "components": components,
        }

    @staticmethod
    def _masked_logits(
        components: IdentityScoreComponents,
        identity_mask: torch.Tensor,
    ) -> torch.Tensor:
        """合并三个分数组件并把补零身份槽替换为最小有限值.

        - components: IdentityScoreComponents; 三个字段应为同形 (B, I) 浮点张量.
        - components.native: 浮点张量, (B, I); 语言基线分数.
        - components.graph_increment: 浮点张量, (B, I); 已乘图门的增量.
        - components.inventory_increment: 浮点张量, (B, I); 已应用多身份条件并乘清单门的增量.
        - identity_mask: bool 张量, (B, I); True 保留组件之和, False 写入当前 logits dtype 的最小有限值.
        - return: 浮点张量, (B, I); 供闭集身份分类直接消费的已屏蔽分数.

        本函数不独立检查四个输入的形状或 identity_mask 的 dtype; 正式 forward 在前序边界
        提供同形掩码. 屏蔽值是有限数而非负无穷, 避免后续数值路径引入无穷值.
        """

        # 浮点 (B, I); 语言基线与两个已门控增量逐槽相加, 补零槽此时仍可能非零.
        logits = (
            components["native"]
            + components["graph_increment"]
            + components["inventory_increment"]
        )
        return logits.masked_fill(~identity_mask, torch.finfo(logits.dtype).min)

class ClosedSetMatcher(nn.Module):
    """组合候选编码器、精确 SMILES 身份编码器和 PDB 内闭集身份打分器.

    数据集已经为每个候选建立所属 PDB 的身份轴; 本模型不跨 PDB 合并身份, 也不根据
    identity_smiles 重新分组. 候选六分类和两个体素区域头
    与身份打分共享 CandidateEncoder 的候选表示与体素特征.

    - config: MatcherModelConfig; 冻结候选主干、融合方式、身份图编码和三分支打分结构.
    """

    def __init__(self, config: MatcherModelConfig) -> None:
        """按同一配置建立候选编码、身份编码和身份打分三个子模块.

        - config: MatcherModelConfig; 通常已在实例化时通过 dataclass 门禁, 本构造函数直接保存并传给子模块, 不重新执行 __post_init__.

        - self.config: MatcherModelConfig; 保留完整冻结配置供外部记录和子模块结构追溯.
        - self.candidate_encoder: CandidateEncoder; 生成候选表示、六分类、两个区域 logits 和体素特征.
        - self.identity_encoder: IdentityEncoder; 生成 native、graph 和 inventory 三种身份表示.
        - self.identity_scorer: IdentityScorer; 计算三个身份分数组件及已屏蔽 identity_logits.
        """

        super().__init__()
        self.config = config
        self.candidate_encoder = CandidateEncoder(config)
        self.identity_encoder = IdentityEncoder(config)
        self.identity_scorer = IdentityScorer(
            config.model_dim,
            config.dropout,
            config.use_manual_features,
            config.use_sampling_features,
            config.initial_graph_gate,
            config.initial_inventory_gate,
        )

    def identity_score_components(
        self,
        batch: CandidateBatch,
        output: FullMatcherOutput,
    ) -> IdentityScoreComponents:
        """在评估模式下读取正式前向已保存的三个身份分数组件.

        - batch: CandidateBatch; 本方法只读取 batch.identity_mask 的 (B, I) 形状作为期望形状.
        - output: FullMatcherOutput; 必须含下列三个由同次正式前向保存的字段.
        - output.identity_score_native: 浮点张量, (B, I); 未屏蔽语言基线.
        - output.identity_score_graph_increment: 浮点张量, (B, I); 未屏蔽且已乘图门的增量.
        - output.identity_score_inventory_increment: 浮点张量, (B, I); 未屏蔽且已应用多身份条件、已乘清单门的增量.
        - return: IdentityScoreComponents; 新字典引用 output 中原有三个张量, 不重新执行模型前向.
        - return.native: 浮点张量, (B, I); 与 output.identity_score_native 是同一张量对象.
        - return.graph_increment: 浮点张量, (B, I); 与 output.identity_score_graph_increment 是同一张量对象.
        - return.inventory_increment: 浮点张量, (B, I); 与 output.identity_score_inventory_increment 是同一张量对象.

        本方法要求 self.training=False, 并只核验三个张量的形状; 不检查 batch 与 output
        是否确由同一次前向产生, 也不复核三个组件之和与 identity_logits 的关系. 组件补零
        槽尚未屏蔽, 调用方必须用 batch.identity_mask 限定诊断范围.
        """

        if self.training:
            raise RuntimeError("身份分数贡献诊断只允许在 eval 模式下读取。")
        # torch.Size (B, I); 当前批次身份掩码定义的唯一可接受组件形状.
        expected_shape = batch["identity_mask"].shape
        # IdentityScoreComponents; 三个值直接引用 output 已保存张量, 不触发随机算子或重新打分.
        components: IdentityScoreComponents = {
            "native": output["identity_score_native"],
            "graph_increment": output["identity_score_graph_increment"],
            "inventory_increment": output["identity_score_inventory_increment"],
        }
        if any(
            component.shape != expected_shape
            for component in (
                components["native"],
                components["graph_increment"],
                components["inventory_increment"],
            )
        ):
            raise ValueError("正式前向身份分数组件必须与当前 batch 的 (B,I) 身份掩码对齐。")
        return components

    def forward(self, batch: CandidateBatch) -> FullMatcherOutput:
        """编码候选及其 PDB 内身份, 并返回候选预测与闭集身份分数.

        - batch: CandidateBatch; 候选、身份和打分子模块消费下列数值字段.
        - batch.voxel_input: 浮点张量, (B, 106, 48, 48, 48); 候选体素输入, 空间轴为 Z/Y/X.
        - batch.a_tokens: 浮点张量, (B, N_A, 230); A 模态 token.
        - batch.a_mask: bool 张量, (B, N_A); A token 有效掩码.
        - batch.p_tokens: 浮点张量, (B, N_P, 116); P 模态 token.
        - batch.p_mask: bool 张量, (B, N_P); P token 有效掩码.
        - batch.v_tokens: 浮点张量, (B, N_V, 68); V 模态 token.
        - batch.v_mask: bool 张量, (B, N_V); V token 与位置下标的有效掩码.
        - batch.v_index_zyx: 整数张量, (B, N_V, 3); V token 在候选体素中的 Z/Y/X 下标.
        - batch.pp_tokens: 浮点张量, (B, N_PP, 4); PP 模态 token.
        - batch.pp_mask: bool 张量, (B, N_PP); PP token 与位置下标的有效掩码.
        - batch.pp_index_zyx: 整数张量, (B, N_PP, 3); PP token 在候选体素中的 Z/Y/X 下标.
        - batch.candidate_manual_features: 浮点张量, (B, 12); A、P、V、PP 模态统计.
        - batch.candidate_sampling_features: 浮点张量, (B, 4); 裁剪边界、中心偏移和 V 保留率统计.
        - batch.language_embeddings: 浮点张量, (B, I, 768); 按精确 SMILES 身份槽补齐的语言向量.
        - batch.identity_mask: bool 张量, (B, I); 真实身份与补零身份槽掩码.
        - batch.identity_class_index: 整数张量, (B, I); 真实槽的五类前景下标和补零槽 -1.
        - batch.graph_element: 整数张量, (N,); 连接后的原子序数.
        - batch.graph_charge: 整数张量, (N,); 连接后的形式电荷.
        - batch.graph_atom_in_ring: 浮点张量, (N, 4); 3、4、5、6 元环标志.
        - batch.graph_edge_index: 整数张量, (2, E); 有向化学键端点.
        - batch.graph_edge_features: 浮点张量, (E, 9); 五类键型和四类环标志.
        - batch.graph_node_graph_index: 整数张量, (N,); 每个原子所属的连接图下标.
        - batch.graph_to_sample: 整数张量, (G,); 每张图所属的样本下标.
        - batch.graph_to_identity: 整数张量, (G,); 每张图在所属样本中的身份槽.
        - return: FullMatcherOutput; 合并以下五个候选字段和四个身份字段.
        - return.candidate_embedding: 浮点张量, (B, D_m); 候选级融合表示.
        - return.class_logits: 浮点张量, (B, 6); 固定 CLASS_NAMES 顺序的六分类未归一化分数.
        - return.ligand_area_logits: 浮点张量, (B, 48, 48, 48); 配体区域逐体素未归一化分数.
        - return.binding_area_logits: 浮点张量, (B, 48, 48, 48); 受体结合区域逐体素未归一化分数.
        - return.voxel_features: 浮点张量, (B, C_v, 48, 48, 48); 候选 RAUNet 最高分辨率特征.
        - return.identity_logits: 浮点张量, (B, I); 三个组件之和, 补零槽为 dtype 最小有限值.
        - return.identity_score_native: 浮点张量, (B, I); 未屏蔽语言基线.
        - return.identity_score_graph_increment: 浮点张量, (B, I); 未屏蔽且已乘图门的增量.
        - return.identity_score_inventory_increment: 浮点张量, (B, I); 未屏蔽且已应用多身份条件、已乘清单门的增量.

        candidate_manual_features 与 candidate_sampling_features 在此始终按键读取并传给
        IdentityScorer; 关闭相应开关只表示数值不参与打分, 不允许完整模型批次省略这两个键.
        identity_smiles 及 metadata 不参与数值前向.
        """

        # CandidateEncoderOutput; 包含候选表示、六分类、两个体素区域分数和共享体素特征.
        candidate_output = self.candidate_encoder(batch)
        # IdentityRepresentations; 包含与 (B, I) 身份轴对齐的 native、graph、inventory 表示.
        identities = self.identity_encoder(batch)
        # IdentityScorerOutput; 包含已屏蔽 identity_logits 与同次未屏蔽三个分数组件.
        identity_scores = self.identity_scorer(
            candidate_output["candidate_embedding"],
            identities,
            batch["identity_mask"],
            candidate_output["class_logits"],
            batch["identity_class_index"],
            batch["candidate_manual_features"],
            batch["candidate_sampling_features"],
        )
        # IdentityScoreComponents; 三个 (B, I) 组件直接从同次打分结果取出, 补零槽仍未屏蔽.
        identity_score_components = identity_scores["components"]
        # 普通字典; 先复制五个候选输出, 再追加四个身份字段.
        output = dict(candidate_output)
        output["identity_logits"] = identity_scores["logits"]
        output["identity_score_native"] = identity_score_components["native"]
        output["identity_score_graph_increment"] = identity_score_components[
            "graph_increment"
        ]
        output["identity_score_inventory_increment"] = identity_score_components[
            "inventory_increment"
        ]
        return cast(FullMatcherOutput, output)
