"""定义候选编码器与闭集身份匹配器共享的结构配置. 

主要入口 MatcherModelConfig 在实例化时检查模型宽度、注意力头数、特征开关、
身份增量门、RAUNet 通道表和层数. to_dict 返回运行记录使用的普通字典, from_dict
只接受当前版本的完整字段, 不迁移历史 checkpoint 配置. 
本模块不构造模型, 也不解释训练数据划分或训练阶段; 训练入口负责检查这些运行契约. 
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

AggregationName = Literal[
    "mean_max",
    "candidate_cross_attention",
    "mean_max_candidate_cross_attention_concat",
]


@dataclass(frozen=True)
class MatcherModelConfig:
    """保存候选编码器和身份匹配器的完整结构参数. 

    配置字段: 
        - aggregation: mean_max、candidate_cross_attention 或 mean_max_candidate_cross_attention_concat; 依次选择四模态掩码均值/最大值、以候选体素摘要查询四模态 token, 或拼接前两种完整摘要并把零输出残差加回 mean/max 主表示. 
        - use_manual_features: 布尔值; 控制候选聚合是否拼接 A/P/V/PP 各三项固化统计, 也控制三个身份打分器是否拼接同一 12 维统计和停止梯度的身份类别概率. 
        - use_sampling_features: 布尔值; 控制候选聚合和三个身份打分器是否拼接四项裁剪几何统计. 
        - use_candidate_token_attention_bias: 布尔值; 只允许与 candidate_cross_attention 同时启用, 为 A/P/V/PP 各建立一套由相对坐标和 Stage1 概率生成注意力偏置的网络. 
        - initial_graph_gate: 有限数值, 范围 [0, 1]; IdentityScorer 图分数增量的可学习标量门初值. 
        - initial_inventory_gate: 有限数值, 范围 [0, 1]; IdentityScorer 同 PDB 自排除身份清单增量的可学习标量门初值. 
        - model_dim: 正整数 D_m; 候选摘要、四模态投影、语言身份和图身份的共同通道宽度, 必须能被 attention_heads 整除. 
        - attention_heads: 正整数 K; 候选条件注意力、配体图注意力和身份清单注意力共用的头数. 
        - dropout: 有限数值, 范围 [0, 1); token 投影、三种候选聚合路径、候选分类头、图消息、配体图注意力、身份清单注意力和三个身份配对残差网络共用的丢弃概率. 
        - voxel_feature_channels: 正整数 C_v; RAUNet 最高分辨率输出及两个体素二分类头的输入通道数. 
        - raunet_planes: 含九个正整数的元组; 顺序为 enc0、enc1、enc2、enc3、bottleneck、dec3、dec2、dec1、dec0, 本类不检查 enc3=bottleneck=dec3 与 dec0=enc0, RaUnetBackbone 构造时才检查这两项关系. 
        - recycle_iterations: 正整数; RAUNet 对同一候选体素输入执行的回收轮数, 前置轮显式禁用梯度, 最后一轮不受该内部 no_grad 包裹; 是否建图仍服从调用位置的外层梯度模式. 
        - gradient_checkpointing: 布尔值; 训练且启用梯度时, 是否对 RAUNet 的高开销子模块使用非重入激活重计算. 
        - bottleneck_attention_layers: 正整数; RAUNet 最深网格串联的三轴旋转位置编码自注意力层数. 
        - graph_layers: 正整数; LigandGraphEncoder 沿双向化学键执行的消息传递层数. 

    该 dataclass 为 frozen, 只阻止字段重新绑定; raunet_planes 本身是不可变元组. 
    """

    aggregation: AggregationName
    use_manual_features: bool
    use_sampling_features: bool
    use_candidate_token_attention_bias: bool
    initial_graph_gate: float
    initial_inventory_gate: float
    model_dim: int
    attention_heads: int
    dropout: float
    voxel_feature_channels: int
    raunet_planes: tuple[int, ...]
    recycle_iterations: int
    gradient_checkpointing: bool
    bottleneck_attention_layers: int
    graph_layers: int

    def __post_init__(self) -> None:
        """拒绝本配置层能够确定的非法结构参数. 

        校验字段: 
            - self.aggregation: 必须属于 mean_max、candidate_cross_attention 或 mean_max_candidate_cross_attention_concat. 
            - self.use_manual_features: 必须是精确 bool, 不接受整数 0 或 1. 
            - self.use_sampling_features: 必须是精确 bool, 不接受整数 0 或 1. 
            - self.use_candidate_token_attention_bias: 必须是精确 bool, 且为真时 aggregation 必须是 candidate_cross_attention. 
            - self.initial_graph_gate: 不接受 bool, 必须是可转换为 float 的有限 int 或 float, 范围 [0, 1]. 
            - self.initial_inventory_gate: 不接受 bool, 必须是可转换为 float 的有限 int 或 float, 范围 [0, 1]. 
            - self.model_dim: 必须是正的精确 int, 且能被 attention_heads 整除. 
            - self.attention_heads: 必须是正的精确 int. 
            - self.dropout: 不接受 bool, 必须是有限 int 或 float, 范围 [0, 1). 
            - self.voxel_feature_channels: 必须是正的精确 int. 
            - self.raunet_planes: 必须是长度九的 tuple, 每项都是正的精确 int. 
            - self.recycle_iterations: 必须是正的精确 int. 
            - self.gradient_checkpointing: 必须是精确 bool. 
            - self.bottleneck_attention_layers: 必须是正的精确 int. 
            - self.graph_layers: 必须是正的精确 int. 

        返回值: 
            - return: None; 字段全部通过时保留实例, 失败时在模型构造前抛出 TypeError 或 ValueError. 

        本方法不检查 RAUNet 编解码通道间的相等关系, 也不检查任务类型与身份门的组合; 这些规则分别在 RaUnetBackbone 和训练配置层执行. 
        """

        if self.aggregation not in (
            "mean_max",
            "candidate_cross_attention",
            "mean_max_candidate_cross_attention_concat",
        ):
            raise ValueError(f"不支持 aggregation={self.aggregation!r}。")
        if type(self.use_manual_features) is not bool:
            raise TypeError("use_manual_features 必须是布尔值。")
        if type(self.use_sampling_features) is not bool:
            raise TypeError("use_sampling_features 必须是布尔值。")
        if type(self.use_candidate_token_attention_bias) is not bool:
            raise TypeError("use_candidate_token_attention_bias 必须是布尔值。")
        if self.use_candidate_token_attention_bias and self.aggregation != "candidate_cross_attention":
            raise ValueError("候选实体注意力偏置只适用于 candidate_cross_attention 聚合。")
        for name, value in (
            ("initial_graph_gate", self.initial_graph_gate),
            ("initial_inventory_gate", self.initial_inventory_gate),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} 必须是数值。")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} 必须位于 [0,1]。")
        # str 到配置值的映射; 集中保证六个宽度、头数与层数字段都是精确 int, 避免 bool 作为 int 子类通过. 
        integer_fields = {
            "model_dim": self.model_dim,
            "attention_heads": self.attention_heads,
            "voxel_feature_channels": self.voxel_feature_channels,
            "recycle_iterations": self.recycle_iterations,
            "bottleneck_attention_layers": self.bottleneck_attention_layers,
            "graph_layers": self.graph_layers,
        }
        if any(type(value) is not int for value in integer_fields.values()):
            raise TypeError("模型宽度、头数、通道数和层数必须是整数。")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads 必须为正整数。")
        if self.model_dim <= 0 or self.model_dim % self.attention_heads != 0:
            raise ValueError("model_dim 必须为正数且能被 attention_heads 整除。")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("dropout 必须是数值。")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0,1)。")
        if self.voxel_feature_channels <= 0:
            raise ValueError("voxel_feature_channels 必须为正整数。")
        if (
            not isinstance(self.raunet_planes, tuple)
            or len(self.raunet_planes) != 9
            or any(type(value) is not int or value <= 0 for value in self.raunet_planes)
        ):
            raise ValueError("raunet_planes 必须含九个正整数通道宽度。")
        if type(self.gradient_checkpointing) is not bool:
            raise TypeError("gradient_checkpointing 必须是布尔值。")
        if self.recycle_iterations <= 0 or self.bottleneck_attention_layers <= 0:
            raise ValueError("RAUNet recycle 次数和瓶颈注意力层数必须为正整数。")
        if self.graph_layers <= 0:
            raise ValueError("graph_layers 必须为正整数。")

    def to_dict(self) -> dict[str, Any]:
        """返回包含全部配置字段的顶层普通字典. 

        返回字段: 
            - return: dict[str, Any], MatcherModelConfig 全部字段到运行时 Python 值的映射. 
            - return.aggregation: 字符串; 与 self.aggregation 相同. 
            - return.use_manual_features: 布尔值; 与 self.use_manual_features 相同. 
            - return.use_sampling_features: 布尔值; 与 self.use_sampling_features 相同. 
            - return.use_candidate_token_attention_bias: 布尔值; 与 self.use_candidate_token_attention_bias 相同. 
            - return.initial_graph_gate: int 或 float; 与 self.initial_graph_gate 的运行时类型和值相同. 
            - return.initial_inventory_gate: int 或 float; 与 self.initial_inventory_gate 的运行时类型和值相同. 
            - return.model_dim: 整数; 与 self.model_dim 相同. 
            - return.attention_heads: 整数; 与 self.attention_heads 相同. 
            - return.dropout: int 或 float; 与 self.dropout 的运行时类型和值相同. 
            - return.voxel_feature_channels: 整数; 与 self.voxel_feature_channels 相同. 
            - return.raunet_planes: 九项整数元组; asdict 不把元组改成列表, JSON 序列化器写出时才会编码为数组. 
            - return.recycle_iterations: 整数; 与 self.recycle_iterations 相同. 
            - return.gradient_checkpointing: 布尔值; 与 self.gradient_checkpointing 相同. 
            - return.bottleneck_attention_layers: 整数; 与 self.bottleneck_attention_layers 相同. 
            - return.graph_layers: 整数; 与 self.graph_layers 相同. 
        """

        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MatcherModelConfig":
        """从当前版本的完整配置字典恢复模型参数. 

        输入参数: 
            - values: 字符串到 Python 配置值的映射; 预期键与 MatcherModelConfig 字段同名, JSON 解码字典是常见来源, 但本函数也接受 raunet_planes 元组; 先复制且绝不就地修改调用者字典. 
            - values.aggregation: 必需字符串; mean_max、candidate_cross_attention 或 mean_max_candidate_cross_attention_concat. 
            - values.use_manual_features: 必需布尔值; 是否使用十二维人工统计. 
            - values.use_sampling_features: 必需布尔值; 是否使用四维裁剪统计. 
            - values.use_candidate_token_attention_bias: 必需布尔值; 是否建立候选对 token 的注意力偏置. 
            - values.initial_graph_gate: 必需数值; 图分数增量的门初值. 
            - values.initial_inventory_gate: 必需数值; 同 PDB 身份清单分数增量的门初值. 
            - values.model_dim: 必需整数; 候选与身份共同宽度. 
            - values.attention_heads: 必需整数; 候选条件、配体图和身份清单注意力共用头数, 不控制 RAUNet 瓶颈头数. 
            - values.dropout: 必需数值; token 投影、候选聚合与分类、图/清单注意力、图消息和身份配对网络共用的丢弃概率. 
            - values.voxel_feature_channels: 必需整数; RAUNet 输出通道数. 
            - values.raunet_planes: 必需整数列表或元组; 在构造 dataclass 前统一转换成元组. 
            - values.recycle_iterations: 必需整数; RAUNet 回收轮数. 
            - values.gradient_checkpointing: 必需布尔值; RAUNet 激活重计算开关. 
            - values.bottleneck_attention_layers: 必需整数; RAUNet 瓶颈注意力层数. 
            - values.graph_layers: 必需整数; 配体图消息层数. 

        返回值: 
            - config: MatcherModelConfig; 保留全部显式字段并通过 __post_init__ 门禁的冻结配置. 

        未知键、缺失键和不能作为 dataclass 参数的结构由构造调用直接失败; 本函数不迁移旧配置, 也不静默删除未知键. 
        """

        # dict[str, Any], 调用者配置的浅拷贝; 只规范化 YAML 列表形式的通道元组. 
        normalized = dict(values)
        if "raunet_planes" in normalized:
            # Any, YAML 或 JSON 数组通常恢复为列表, 内存配置也可已经是元组; 其它容器在此失败. 
            raw_planes = normalized["raunet_planes"]
            if not isinstance(raw_planes, (list, tuple)):
                raise TypeError("raunet_planes 必须是整数列表。")
            normalized["raunet_planes"] = tuple(raw_planes)
        return cls(**normalized)
