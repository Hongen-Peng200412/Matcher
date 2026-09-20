"""组合候选编码、配体身份编码与 PDB 内闭集身份打分.

候选分支先把一个 PDB 中的局部体素与 A、P、V、PP 四类 token 编码为单个候选表示.
身份分支把精确 SMILES 的语言向量、化学图和同 PDB 身份清单分别编码;
身份字符串只负责数据侧分组与诊断, 不参与本模块的数值前向.
打分器随后分别计算语言基线、配体图增量与身份清单增量, 并只在最终 identity_logits 中屏蔽补零身份槽.

形状符号如下.

- B: 当前候选批次大小.
- I: 当前批次补齐后的最大身份数.
- D_m: 统一的模型表示宽度.
- K: 多头注意力的头数.

- N: 当前批次全部 SMILES 图连接后的原子数.
- E: 当前批次全部 SMILES 图连接后的有向边数.
- G: 当前批次连接后的 SMILES 图数.
- F_e: SMILES 有向边特征宽度; 正式数据契约固定为 9.

本模块只输出候选前景 logit 和闭集 SMILES 分数; 身份监督由训练损失直接计算.
"""

from __future__ import annotations

import math
from typing import TypedDict, cast

import torch
from torch import nn
from torch.nn import functional as F

from ..data.schema import (
    CANDIDATE_SAMPLING_FEATURE_DIM,
    LANGUAGE_EMBEDDING_DIM,
)
from ..datasets.contracts import CandidateBatch
from .candidate_encoder import CandidateEncoder, CandidateEncoderOutput
from .config import MatcherModelConfig
from .fusion import (
    CandidateCrossAttention,
    MeanMaxCandidateConditionedResidualAggregator,
    normalize_candidate_sampling_features,
)


def _initialization_domain_seed(base_seed: int, module_name: str) -> int:
    """
    把实验种子和叶模块路径组合为稳定的初始化子流种子.

    - base_seed: 调用模型构造前由 torch.manual_seed 固定的实验种子.
    - module_name: named_modules 产生的完整叶模块路径.
    - return: 位于 PyTorch 手动种子有效范围内的非负整数.

    路径中的每个 UTF-8 字节按位置加入种子. 该值只用于隔离初始化随机流, 不作为内容
    摘要、文件标识或持久化取证字段.
    """
    maximum_seed = 2**63 - 1
    domain_seed = int(base_seed) % maximum_seed
    for position, value in enumerate(module_name.encode("utf-8"), start=1):
        domain_seed = (domain_seed + position * value * 1_000_003) % maximum_seed
    return domain_seed


def _reset_named_leaf_modules(model: nn.Module, base_seed: int) -> None:
    """
    按叶模块完整路径重新执行 PyTorch 默认参数初始化.

    - model: 已完成子模块注册的 ClosedSetMatcher.
    - base_seed: 当前实验种子，同一 seed 的各变体共享该值.
    - return: None. 仅重置具有 reset_parameters 的叶模块, 不推进调用方全局随机状态.

    可选分支是否存在不会改变其它叶模块的子流。因此同名同形参数在各消融变体中逐值
    相同, 模型构造也不会让后续 dropout 随机流因变体结构不同而错位.
    """
    for module_name, module in model.named_modules():
        if not module_name or any(module.children()):
            continue
        reset_parameters = getattr(module, "reset_parameters", None)
        if not callable(reset_parameters):
            continue
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_initialization_domain_seed(base_seed, module_name))
            reset_parameters()


def _restore_scientific_zero_initialization(model: "ClosedSetMatcher") -> None:
    """
    恢复条件注意力和摘要残差明确约定的零输出初值.

    - model: 已完成域分离默认初始化的 ClosedSetMatcher.
    - return: None. 只修改 strongest/trunk_only 聚合器中的 PP 输出投影和残差末层.

    域分离初始化会调用叶 Linear 的默认 reset_parameters. 本函数随后恢复 fusion 模块声明的两处科学零初值, 保证新增条件分支在第一个更新前不改变 mean/max 主路径.
    """
    aggregator = model.candidate_encoder.aggregator
    if not isinstance(aggregator, MeanMaxCandidateConditionedResidualAggregator):
        return
    pp_attention = aggregator.conditioned.branches["pp"]
    if not isinstance(pp_attention, CandidateCrossAttention):
        raise TypeError("PP 条件分支必须是 CandidateCrossAttention。")
    final = aggregator.residual[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError("摘要残差末层必须是 Linear。")
    nn.init.zeros_(pp_attention.output.weight)
    nn.init.zeros_(pp_attention.output.bias)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)





class IdentityRepresentations(TypedDict):
    """
    同一批精确 SMILES 身份在三条打分分支中的数值表示.

    - native: 浮点张量, (B, I, D_m); 由语言向量投影得到, 有效槽对应顶层 PreparedLigand.smiles, 补零槽为零.
    - graph: 浮点张量, (B, I, D_m); 由 native 查询同一精确 SMILES 的原子 token 得到, 补零槽为零.
    - inventory: 浮点张量, (B, I, D_m); 由 native 查询同一 PDB 内其它有效身份得到, 没有其它身份或属于补零槽时为零.
    """
    native: torch.Tensor
    graph: torch.Tensor
    inventory: torch.Tensor


class IdentityScoreComponents(TypedDict):
    """
    同一次身份前向产生的三个未屏蔽分数组件.

    - native: 浮点张量, (B, I); 语言表示分支的基线分数, 补零槽可能因配对网络偏置而非零.
    - graph_increment: 浮点张量, (B, I); 配体图分支原始分数乘 graph_gate 后的增量, 补零槽尚未屏蔽.
    - inventory_increment: 浮点张量, (B, I); 身份清单分支分数先按样本的多身份条件归零、再乘 inventory_gate 后的增量, 补零槽尚未屏蔽.

    三个字段的 dtype 由当前数值上下文决定, 设备与本次模型前向一致. 调用方只能在 identity_mask 为 True 的槽位读取组件; 补零身份表示为零并不保证配对网络的输出为零.
    """
    native: torch.Tensor
    graph_increment: torch.Tensor
    inventory_increment: torch.Tensor


class IdentityScorerOutput(TypedDict):
    """
    身份打分器的一次正式前向结果.

    - logits: 浮点张量, (B, I); 三个分数组件之和, 有效槽保留数值, 补零槽替换为相应 dtype 的最小有限值.
    - components: IdentityScoreComponents; 同次前向产生的未屏蔽组件.
    - components.native: 浮点张量, (B, I); 语言基线分数.
    - components.graph_increment: 浮点张量, (B, I); 已乘图门的分数增量.
    - components.inventory_increment: 浮点张量, (B, I); 已应用多身份条件且已乘清单门的分数增量.
    """
    logits: torch.Tensor
    components: IdentityScoreComponents


class FullMatcherOutput(CandidateEncoderOutput):
    """
    候选编码输出与 PDB 内身份打分的合并结果.

    - candidate_embedding: 浮点张量, (B, D_m); 融合候选体素、四模态 token 与配置启用统计的候选表示.
    - foreground_logit: 浮点张量, (B,); 五类真实前景相对背景的二分类 logit.
    - ligand_area_logits: 浮点张量或 None; 有 U-Net 时为 (B,48,48,48), pocket_only 为 None.
    - binding_area_logits: 浮点张量或 None; 有 U-Net 时为 (B,48,48,48), pocket_only 为 None.
    - voxel_features: 浮点张量或 None; 有 U-Net 时为 (B,C_v,48,48,48), pocket_only 为 None.

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






# ===========================================================================================================================================
# ===========================================================================================================================================
class GraphMessageLayer(nn.Module):
    def __init__(self, model_dim: int, edge_dim: int, dropout: float) -> None:
        """
        沿有向化学键汇聚消息并更新每个 SMILES 原子 token.
        每条边的消息是源原子投影与边特征投影之和; 同一目标原子的消息按入度取平均, 再经过逐通道门控、残差和前馈网络. 正式数据把每条化学键展开为正反两个方向.

        构造参数:
            - model_dim: 正整数, 统一的原子 token 宽度 D_m; 本类不单独检查正值.
            - edge_dim: 正整数, 有向边特征宽度 F_e; 正式 SMILES 图契约为 9.
            - dropout: 浮点丢弃率; 有效范围由 torch.nn.Dropout 构造过程检查.

        forward 输入:
            - nodes: 浮点张量, (N, D_m); 连接后全部图的原子 token. 本函数不单独检查二维形状或 D_m.
            - edge_index: 整数张量, (2, E); 第 0 行是源原子下标, 第 1 行是目标原子下标.
            - edge_features: 浮点张量, (E, F_e); 与 edge_index 的有向边轴逐项对齐.

        forward 输出:
            - 点特征 (N, D_m); 保留原子轴顺序, dtype 由输入与当前数值上下文决定.
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
        """
        按目标原子平均有向边消息并返回更新后的原子 token.

        - nodes: 浮点张量, (N, D_m); 连接后全部图的原子 token. 本函数不单独检查二维形状或 D_m.
        - edge_index: 整数张量, (2, E); 第 0 行是源原子下标, 第 1 行是目标原子下标.
        - edge_features: 浮点张量, (E, F_e); 与 edge_index 的有向边轴逐项对齐.
        - return: 浮点张量, (N, D_m); 保留原子轴顺序, dtype 由输入与当前数值上下文决定.
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
    def __init__(self, model_dim: int, layer_count: int, dropout: float) -> None:
        """
        把连接后的 SMILES 图字段编码为保留原子轴的图 token.

        元素和电荷分别查表, 四种环大小标志经无偏置线性层投影; 依次经过若干 GraphMessageLayer. 本类不在图内或图间做 mean/max 池化.

        构造参数:
            - model_dim: 正整数, 输出原子 token 宽度 D_m; 本类不单独检查正值.
            - layer_count: 整数, 顺序执行的图消息层数; 正式配置要求为正, 直接传入零或负整数会产生空层列表.
            - dropout: 浮点丢弃率, 传给每个 GraphMessageLayer.

        输入:
            - batch: CandidateBatch; 本函数消费下列连接图字段.
            - batch.graph_element: 整数张量, (N,); 原子序数, 显式要求属于 [0, 128).
            - batch.graph_charge: 整数张量, (N,); 形式电荷, 显式要求属于 [-16, 16].
            - batch.graph_atom_in_ring: 浮点张量, (N, 4); 依次标记 3、4、5、6 元环.
            - batch.graph_edge_index: 整数张量, (2, E); 连接后有向边端点, 由每层 GraphMessageLayer 消费.
            - batch.graph_edge_features: 浮点张量, (E, 9); 五类键型与四类环标志.

        输出:
            - return: 点特征, (N, D_m); 保留连接后的原子顺序, 不进行图级池化.
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
        """
        融合原子离散字段和连续字段, 再沿全部有向边更新 token.

        - batch: CandidateBatch; 本函数消费下列连接图字段.
        - batch.graph_element: 整数张量, (N,); 原子序数, 显式要求属于 [0, 128).
        - batch.graph_charge: 整数张量, (N,); 形式电荷, 显式要求属于 [-16, 16].
        - batch.graph_atom_in_ring: 浮点张量, (N, 4); 依次标记 3、4、5、6 元环.
        - batch.graph_edge_index: 整数张量, (2, E); 连接后有向边端点, 由每层 GraphMessageLayer 消费.
        - batch.graph_edge_features: 浮点张量, (E, 9); 五类键型与四类环标志.
        - return: 浮点张量, (N, D_m); 保留连接后的原子顺序, 不进行图级池化.
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









# ===========================================================================================================================================
# ===========================================================================================================================================
# see me：这没有加残差, 返回的就是 attention 真值(相当于增量)
class GraphIdentityAttention(nn.Module):
    def __init__(self, model_dim: int, head_count: int, dropout: float) -> None:
        """
        以每个精确 SMILES 的语言表示查询同一化学图的原子 token.
        一个图产生一个 query, 每个原子产生 key 和 value. softmax 仅在同一图的原子轴内、分头归一化; 得到的图向量写回对应样本的身份槽.

        构造参数:
            - model_dim: 正整数, query、key、value 与输出宽度 D_m.
            - head_count: 正整数, 注意力头数 K; 必须整除 model_dim.
            - dropout: 浮点丢弃率; 作用于已归一化的原子注意力权重, 丢弃后不再次归一化.

        forward 输入:
            - native: 浮点张量, (B, I, D_m); 语言分支的身份表示, 补零槽为零.
            - nodes: 浮点张量, (N, D_m); 全部 SMILES 图连接后的原子 token.
            - node_graph_index: 整数张量, (N,); 每个原子所属的连接图下标, 取值应属于 [0, G).
            - graph_to_sample: 整数张量, (G,); 每张图所属的样本下标, 取值应属于 [0, B).
            - graph_to_identity: 整数张量, (G,); 每张图在所属样本中的身份槽, 取值应属于 [0, I).

        forward 输出:
            - SMILES更新后的表征: (B, I, D_m); 补零槽为零.
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
        """
        在每张图内部对原子注意力归一化并写回对应身份槽.

        forward 输入:
            - native: 浮点张量, (B, I, D_m); 语言分支的身份表示, 补零槽为零.
            - nodes: 浮点张量, (N, D_m); 全部 SMILES 图连接后的原子 token.
            - node_graph_index: 整数张量, (N,); 每个原子所属的连接图下标, 取值应属于 [0, G).
            - graph_to_sample: 整数张量, (G,); 每张图所属的样本下标, 取值应属于 [0, B).
            - graph_to_identity: 整数张量, (G,); 每张图在所属样本中的身份槽, 取值应属于 [0, I).

        forward 输出:
            - SMILES更新后的表征: (B, I, D_m); 补零槽为零.
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


# see me：这没有加残差, 返回的就是 attention 真值(相当于增量)
class SelfExcludingInventoryAttention(nn.Module):
    def __init__(self, model_dim: int, head_count: int, dropout: float) -> None:
        """
        让每个有效身份查询同一 PDB 内除自身以外的其它身份.
        query 与 key 轴都对应批内身份槽 I. 注意力同时屏蔽补零槽和对角线, 所以单身份样本没有可用 key, 并得到严格零清单表示; 多个身份的样本只读取其它有效身份.

        构造参数:
            - model_dim: 正整数, 输入、query、key、value 和输出宽度 D_m.
            - head_count: 正整数, 注意力头数 K; 必须整除 model_dim.
            - dropout: 浮点丢弃率; 作用于归一化权重, 丢弃后不再次归一化.

        forward 输入:
            - identities: 浮点张量, (B, I, D_m); 通常是语言分支的 native 表示.
            - identity_mask: bool 张量, (B, I); True 表示真实身份槽, False 表示补零槽.

        forward 输出:
            - SMILES更新后的表征, (B, I, D_m); 有效且存在其它身份的槽保存上下文, 其余槽严格为零.
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
        """
        让每个有效身份查询同一 PDB 内除自身以外的其它身份.
        query 与 key 轴都对应批内身份槽 I. 注意力同时屏蔽补零槽和对角线, 所以单身份样本没有可用 key, 并得到严格零清单表示; 多个身份的样本只读取其它有效身份.

        forward 输入:
            - identities: 浮点张量, (B, I, D_m); 通常是语言分支的 native 表示.
            - identity_mask: bool 张量, (B, I); True 表示真实身份槽, False 表示补零槽.

        forward 输出:
            - SMILES更新后的表征, (B, I, D_m); 有效且存在其它身份的槽保存上下文, 其余槽严格为零.
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
        # bool (B, 1, I, I); 同时要求 query、key 都有效.
        valid = identity_mask[:, None, :, None] & identity_mask[:, None, None, :]
        valid = valid & off_diagonal[None, None, :, :]

        score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
        # 浮点 (B, K, I, I); 屏蔽后沿 key 轴重新归一化, 无可用 key 的 query 保持全零.
        weight = torch.softmax(score, dim=-1) * valid.to(dtype=score.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)   # see me: / weight.sum(dim=-1, keepdim=True) 实际上似乎无必要
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
        """
        把模型宽度拆为注意力头并把头轴移到身份轴之前.

        - values: 浮点张量, (B, I, D_m); 最后一轴必须可重排为 K 个 D_m/K 通道.
        - return: 浮点张量, (B, K, I, D_m/K); 仅重排形状和轴, 不复制语义.
        """
        # (B, I, D_m) ————> (B, I, K, D_m/K) ————> (B, K, I, D_m/K)
        batch_size, identity_count, _ = values.shape
        return values.reshape(
            batch_size,
            identity_count,
            self.head_count,
            self.head_dim,
        ).transpose(1, 2)



# -------- 综合 ---------
class IdentityEncoder(nn.Module):
    def __init__(self, config: MatcherModelConfig) -> None:
        """
        为每个 PDB 内精确 SMILES 身份生成语言、配体图和自排除清单表示.
        语言向量投影为 native, graph 分支用 native 查询自身图, inventory 分支让 native 查询同一 PDB 的其它 native. [两者这没有加残差, 返回的就是 attention 真值(相当于增量)]

        输入与构造:
            - config: MatcherModelConfig; 读取 model_dim、graph_layers、attention_heads 和 dropout 构造三个分支.
            - self.language: Sequential; 把每个 768 维语言向量归一化、投影并激活成 D_m 维 native 表示.
            - self.graph_encoder: LigandGraphEncoder; 把连接后的 SMILES 图字段编码为逐原子 token.
            - self.graph_attention: GraphIdentityAttention; 让 native 表示查询同一身份的原子 token.
            - self.inventory_attention: SelfExcludingInventoryAttention; 让 native 表示查询同 PDB 的其它身份.

        forward 输入:
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

        forward 输出:
            - return: IdentityRepresentations; 三种字段均与 (B, I, D_m) 身份轴对齐.
            - return.native: 浮点张量, (B, I, D_m); 语言投影表示, 补零槽严格为零.
            - return.graph: 浮点张量, (B, I, D_m); 自身配体图注意力表示, 补零槽严格为零.
            - return.inventory: 浮点张量, (B, I, D_m); 自排除身份清单表示, 无其它身份或补零时严格为零.
        """
        super().__init__()
        self.language = nn.Sequential(
            nn.LayerNorm(LANGUAGE_EMBEDDING_DIM),
            nn.Linear(LANGUAGE_EMBEDDING_DIM, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )
        self.graph_encoder: LigandGraphEncoder | None = None
        self.graph_attention: GraphIdentityAttention | None = None
        self.inventory_attention: SelfExcludingInventoryAttention | None = None
        if config.uses_identity_increments:
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
        """
        forward 输入:
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

        forward 输出:
            - return: IdentityRepresentations; 三种字段均与 (B, I, D_m) 身份轴对齐.
            - return.native: 浮点张量, (B, I, D_m); 语言投影表示, 补零槽严格为零.
            - return.graph: 浮点张量, (B, I, D_m); 自身配体图注意力表示, 补零槽严格为零.
            - return.inventory: 浮点张量, (B, I, D_m); 自排除身份清单表示, 无其它身份或补零时严格为零.
        """
        # bool (B, I); 决定语言补零槽、图 occupancy 目标和清单注意力的有效身份范围.
        identity_mask = batch["identity_mask"]
        # 浮点 (B, I, D_m); 768 维语言向量经归一化、线性投影和 GELU 后得到的原始结果.
        native = self.language(batch["language_embeddings"])
        # 浮点 (B, I, D_m); 投影层偏置可能使补零输入非零, 故显式按身份掩码再次归零.
        native = native * identity_mask[..., None].to(dtype=native.dtype)
        if self.graph_encoder is None:
            zero_increment = torch.zeros_like(native)
            return {
                "native": native,
                "graph": zero_increment,
                "inventory": zero_increment,
            }
        if self.graph_attention is None or self.inventory_attention is None:
            raise RuntimeError("身份增量模块未完整建立。")

        # 仅用于检验
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









# ===========================================================================================================================================
# ===========================================================================================================================================
class PairScorer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        dropout: float,
        manual_feature_dim: int,
        use_sampling_features: bool,
    ) -> None:
        """
        PairScorer 是一个“候选(candidate)—身份(identity)配对打分头”：给定一个候选表示和该候选所属 PDB 的全部身份表示，输出每个身份的匹配分数。
        candidate 和 identity 先分别投影并归一化, 得到缩放余弦分数;
        配置启用时还会拼接 12 维人工特征与 4 维裁剪统计特征.

        构造参数:
            - model_dim: 候选表示和单类身份表示的宽度 D_m.
            - dropout: 兼容性残差网络中间层的丢弃概率.
            - manual_feature_dim: 已归一化模态统计宽度; 0 表示不把手动统计加入残差输入.
            - use_sampling_features: 是否把 4 维已归一化裁剪统计加入残差输入.

        forward 输入:
            - candidate: 浮点张量, (B, D_m); 候选编码器输出的候选级表示.
            - identities: 浮点张量, (B, I, D_m); 当前分支的 native、graph 或 inventory 身份表示.
            - normalized_manual_features: None 或浮点张量, (B,D_h); D_h 由 manual_feature_dim 固定.
            - normalized_sampling_features: None 或浮点张量, (B, 4); 已固定尺度归一化的裁剪采样统计.

        forward 输出:
            - return: 浮点张量, (B, I); 缩放余弦分数与兼容性残差之和, 补零槽尚未屏蔽.
        """
        super().__init__()
        if manual_feature_dim < 0:
            raise ValueError("身份打分的手动特征宽度不得为负数。")
        self.manual_feature_dim = manual_feature_dim
        self.use_sampling_features = use_sampling_features
        self.candidate = nn.Linear(model_dim, model_dim, bias=False)
        self.identity = nn.Linear(model_dim, model_dim, bias=False)
        # 标量 Parameter; 保存余弦尺度的自然对数, 前向只限制 exp 后的上限 100, 不设下限.
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        # Python 整数; 残差网络输入宽度, 基础四组各 D_m, 并按开关追加 12 维和 4 维.
        compatibility_dim = (
            model_dim * 4
            + manual_feature_dim
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
        normalized_sampling_features: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        对批内每个候选与其 PDB 内每个身份槽计算配对分数.

        forward 输入:
            - candidate: 浮点张量, (B, D_m); 候选编码器输出的候选级表示.
            - identities: 浮点张量, (B, I, D_m); 当前分支的 native、graph 或 inventory 身份表示.
            - normalized_manual_features: None 或浮点张量, (B,D_h); D_h 由构造时 manual_feature_dim 固定.
            - normalized_sampling_features: None 或浮点张量, (B, 4); 已固定尺度归一化的裁剪采样统计.

        forward 输出:
            - return: 浮点张量, (B, I); 缩放余弦分数与兼容性残差之和, 补零槽尚未屏蔽.
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
        if self.manual_feature_dim > 0:
            if normalized_manual_features is None:
                raise ValueError("启用手动特征的身份打分必须提供候选统计。")
            if normalized_manual_features.shape != (
                candidate.shape[0],
                self.manual_feature_dim,
            ):
                raise ValueError("身份打分的候选手动特征形状错误。")
            # 追加浮点 (B,I,D_h); 同一候选统计广播到当前 PDB 的全部身份槽.
            compatibility_parts.append(
                normalized_manual_features[:, None, :].expand(
                    -1,
                    identities.shape[1],
                    -1,
                )
            )
        elif normalized_manual_features is not None:
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




# ——————————————————————————>>>>>>>>>>>>>
class IdentityScorer(nn.Module):
    def __init__(self, config: MatcherModelConfig) -> None:
        """
        给定候选(candidate), 以及对应的身份表示(identities), 通过三路结果的综合(naive, graph, inventory)计算身份打分.

        构造参数:
            - self.native: PairScorer; 精确 SMILES 语言表示的未门控基线.
            - self.graph: PairScorer 或 None; SMILES 化学图增量, trunk_only 不实例化.
            - self.inventory: PairScorer 或 None; 同 PDB 其它身份摘要增量, trunk_only 不实例化.
            - self.graph_gate: 标量 Parameter 或 None; 图增量的零初始化可训练门.
            - self.inventory_gate: 标量 Parameter 或 None; 身份清单增量的零初始化可训练门.

        forward 输入:
            - candidate: 浮点张量, (B, D_m); 候选编码器产生的候选级表示.
            - identities: IdentityRepresentations; 三种表示应具有相同 (B, I, D_m) 形状.
            - identities.native: 浮点张量, (B, I, D_m); 语言投影表示.
            - identities.graph: 浮点张量, (B, I, D_m); 自身 SMILES 图表示.
            - identities.inventory: 浮点张量, (B, I, D_m); 自排除身份清单表示.
            - identity_mask: bool 张量, (B, I); True 为真实身份槽, False 为补零槽.

            - normalized_candidate_manual_features: 浮点张量, (B, 12); 候选编码器已归一化并按当前变体门控的 A、P、V、PP 模态统计.
            - candidate_sampling_features: 浮点张量, (B, 4); 裁剪采样统计, 仅在启用采样特征时归一化.

        forward 输出:
            - return: IdentityScoreComponents; 三个字段均为与身份槽对齐的未屏蔽 (B, I) 分数.
            - return.native: 浮点张量, (B, I); 语言分支基线.
            - return.graph_increment: 浮点张量, (B, I); 图分支原始分数乘 graph_gate.
            - return.inventory_increment: 浮点张量, (B, I); 清单分支先按多身份条件归零、再乘 inventory_gate.

            - return.logits: 浮点张量, (B, I); 三个组件之和, 补零槽替换为 dtype 最小有限值.
        """
        super().__init__()
        self.manual_feature_dim = config.identity_manual_feature_dim
        self.use_sampling_features = config.use_sampling_features
        self.native = PairScorer(
            config.model_dim,
            config.dropout,
            self.manual_feature_dim,
            config.use_sampling_features,
        )
        if config.uses_identity_increments:
            self.graph: PairScorer | None = PairScorer(
                config.model_dim,
                config.dropout,
                self.manual_feature_dim,
                config.use_sampling_features,
            )
            self.inventory: PairScorer | None = PairScorer(
                config.model_dim,
                config.dropout,
                self.manual_feature_dim,
                config.use_sampling_features,
            )
            # 两个增量门正式固定为零初值, 不由 YAML 改写.
            self.graph_gate: nn.Parameter | None = nn.Parameter(torch.tensor(0.0))
            self.inventory_gate: nn.Parameter | None = nn.Parameter(torch.tensor(0.0))
        else:
            self.graph = None
            self.inventory = None
            self.graph_gate = None
            self.inventory_gate = None

    def score_components(
        self,
        candidate: torch.Tensor,
        identities: IdentityRepresentations,
        identity_mask: torch.Tensor,
        normalized_candidate_manual_features: torch.Tensor,
        candidate_sampling_features: torch.Tensor,
    ) -> IdentityScoreComponents:
        """
        forward 输入:
            - candidate: 浮点张量, (B, D_m); 候选编码器产生的候选级表示.
            - identities: IdentityRepresentations; 三种表示应具有相同 (B, I, D_m) 形状.
            - identities.native: 浮点张量, (B, I, D_m); 语言投影表示.
            - identities.graph: 浮点张量, (B, I, D_m); 自身 SMILES 图表示.
            - identities.inventory: 浮点张量, (B, I, D_m); 自排除身份清单表示.
            - identity_mask: bool 张量, (B, I); True 为真实身份槽, False 为补零槽.

            - normalized_candidate_manual_features: 浮点张量, (B, 12); CandidateEncoder 已归一化并应用当前变体模态门的 A、P、V、PP 统计.
            - candidate_sampling_features: 浮点张量, (B, 4); 裁剪采样统计, 仅在启用采样特征时归一化.

        forward 输出:
            - return: IdentityScoreComponents; 三个字段均为与身份槽对齐的未屏蔽 (B, I) 分数.
            - return.native: 浮点张量, (B, I); 语言分支基线.
            - return.graph_increment: 浮点张量, (B, I); 图分支原始分数乘 graph_gate.
            - return.inventory_increment: 浮点张量, (B, I); 清单分支先按多身份条件归零、再乘 inventory_gate.

            - return.logits: 浮点张量, (B, I); 三个组件之和, 补零槽替换为 dtype 最小有限值.
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

        # None 或浮点 (B,D_h); 仅在当前变体的 D_h 大于零时填入已归一化模态统计.
        normalized_manual: torch.Tensor | None = None
        # None 或浮点 (B, 4); 仅在 use_sampling_features=True 时填入已归一化裁剪统计.
        normalized_sampling: torch.Tensor | None = None
        if self.manual_feature_dim > 0:
            # 浮点 (B,D_h); blob_only 的 D_h=3，其余启用完整模态统计的变体为 12.
            normalized_manual = normalized_candidate_manual_features
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
            normalized_sampling,
        )
        if self.graph is None or self.inventory is None:
            zero_increment = torch.zeros_like(native)
            return {
                "native": native,
                "graph_increment": zero_increment,
                "inventory_increment": zero_increment,
            }
        if self.graph_gate is None or self.inventory_gate is None:
            raise RuntimeError("身份增量模块和门参数必须同时存在。")
        graph_increment = self.graph_gate * self.graph(candidate, identities["graph"], normalized_manual, normalized_sampling)
        inventory_increment = self.inventory(candidate, identities["inventory"], normalized_manual, normalized_sampling)
        # bool (B, 1); 每个样本是否至少有两个有效身份, 沿身份轴广播到所有槽.
        has_other_identity = identity_mask.sum(dim=1, keepdim=True) > 1
        inventory_increment = inventory_increment * has_other_identity.to(dtype=inventory_increment.dtype)
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
        normalized_candidate_manual_features: torch.Tensor,
        candidate_sampling_features: torch.Tensor,
    ) -> IdentityScorerOutput:
        # IdentityScoreComponents; 同一次调用得到的三个 (B, I) 未屏蔽组件.
        components = self.score_components(
            candidate,
            identities,
            identity_mask,
            normalized_candidate_manual_features,
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
        """
        合并三个分数组件并把补零身份槽替换为最小有限值.
        - return: 浮点张量, (B, I); 供闭集身份分类直接消费的已屏蔽分数.
        """
        # 浮点 (B, I); 语言基线与两个已门控增量逐槽相加, 补零槽此时仍可能非零.
        logits = (
            components["native"]
            + components["graph_increment"]
            + components["inventory_increment"]
        )
        return logits.masked_fill(~identity_mask, torch.finfo(logits.dtype).min)











# ===========================================================================================================================================
# ===========================================================================================================================================
# ----------- 总函数 -----------
class ClosedSetMatcher(nn.Module):
    def __init__(self, config: MatcherModelConfig) -> None:
        """
        总类:
            - 候选(candidate)编码器 CandidateEncoder
            - SMILES 身份(identity)编码器 IdentityEncoder
            - PDB 内身份打分器 IdentityScorer.

        构造参数:
            - self.candidate_encoder: CandidateEncoder; 生成候选表示、前景 logit、两个区域 logits 和体素特征.
            - self.identity_encoder: IdentityEncoder; 生成 native、graph 和 inventory 三种身份表示.
            - self.identity_scorer: IdentityScorer; 计算三个身份分数组件及已屏蔽 identity_logits.

        forward 输入:
            - batch: CandidateBatch; 候选、身份和打分子模块消费下列数值字段.
            - batch.voxel_input: 浮点张量, (B,C,48,48,48). strongest/trunk_only 的 C=106, density_blob_only 的 C=1, pocket_only 的 C=0.
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
            - batch.graph_element: 整数张量, (N,); 连接后的原子序数.
            - batch.graph_charge: 整数张量, (N,); 连接后的形式电荷.
            - batch.graph_atom_in_ring: 浮点张量, (N, 4); 3、4、5、6 元环标志.
            - batch.graph_edge_index: 整数张量, (2, E); 有向化学键端点.
            - batch.graph_edge_features: 浮点张量, (E, 9); 五类键型和四类环标志.
            - batch.graph_node_graph_index: 整数张量, (N,); 每个原子所属的连接图下标.
            - batch.graph_to_sample: 整数张量, (G,); 每张图所属的样本下标.
            - batch.graph_to_identity: 整数张量, (G,); 每张图在所属样本中的身份槽.

        forward 输出:
            - return: FullMatcherOutput; 合并以下五个候选字段和四个身份字段.
            - return.candidate_embedding: 浮点张量, (B, D_m); 候选级融合表示.
            - return.foreground_logit: 浮点张量, (B,); 候选是真实前景的二分类 logit.
            - return.ligand_area_logits: strongest/density_blob_only/trunk_only 为浮点张量 (B,48,48,48), blob_only/pocket_only 为 None.
            - return.binding_area_logits: strongest/density_blob_only/trunk_only 为浮点张量 (B,48,48,48), blob_only/pocket_only 为 None.
            - return.voxel_features: strongest/density_blob_only/trunk_only 为浮点张量 (B,C_v,48,48,48), blob_only/pocket_only 为 None.
            - return.identity_logits: 浮点张量, (B, I); 三个组件之和, 补零槽为 dtype 最小有限值.
            - return.identity_score_native: 浮点张量, (B, I); 未屏蔽语言基线.
            - return.identity_score_graph_increment: 浮点张量, (B, I); 未屏蔽且已乘图门的增量.
            - return.identity_score_inventory_increment: 浮点张量, (B, I); 未屏蔽且已应用多身份条件、已乘清单门的增量.
        """
        super().__init__()
        caller_rng_state = torch.random.get_rng_state()
        base_seed = torch.initial_seed()
        try:
            self.config = config
            self.candidate_encoder = CandidateEncoder(config)
            self.identity_encoder = IdentityEncoder(config)
            self.identity_scorer = IdentityScorer(config)
            _reset_named_leaf_modules(self, base_seed)
            _restore_scientific_zero_initialization(self)
        finally:
            torch.random.set_rng_state(caller_rng_state)

    def identity_score_components(
        self,
        batch: CandidateBatch,
        output: FullMatcherOutput,
    ) -> IdentityScoreComponents:
        """
        在评估模式下, 根据 FullMatcherOutput 和 batch.identity_mask 简单读取 IdentityScoreComponents.
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
        """
        candidate_manual_features 与 candidate_sampling_features 在此始终按键读取。候选编码器负责
        归一化 12 项模态统计并应用 strongest 的 A/P/PP 三门，身份打分器不得读取未门控统计。
        identity_smiles 及 metadata 不参与数值前向.
        """
        # CandidateEncoderOutput; 包含候选表示、前景分数、两个体素区域分数和共享体素特征.
        candidate_output = self.candidate_encoder(batch)
        # IdentityRepresentations; 包含与 (B, I) 身份轴对齐的 native、graph、inventory 表示.
        identities = self.identity_encoder(batch)
        # IdentityScorerOutput; 包含已屏蔽 identity_logits 与同次未屏蔽三个分数组件.
        identity_scores = self.identity_scorer(
            candidate_output["candidate_embedding"],
            identities,
            batch["identity_mask"],
            self.candidate_encoder.prepare_manual_features(
                batch["candidate_manual_features"]
            ),
            batch["candidate_sampling_features"],
        )
        # IdentityScoreComponents; 三个 (B, I) 组件直接从同次打分结果取出, 补零槽仍未屏蔽.
        identity_score_components = identity_scores["components"]
        # 普通字典; 先复制五个候选输出, 再追加四个身份字段.
        output = dict(candidate_output)
        output["identity_logits"] = identity_scores["logits"]
        output["identity_score_native"] = identity_score_components["native"]
        output["identity_score_graph_increment"] = identity_score_components["graph_increment"]
        output["identity_score_inventory_increment"] = identity_score_components["inventory_increment"]
        return cast(FullMatcherOutput, output)
