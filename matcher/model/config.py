"""用 ``MatcherModelConfig`` 定义模型结构，并与 checkpoint/JSON 中的普通字典相互转换。"""

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
    """保存候选编码器和身份匹配器的完整结构参数。

    - aggregation: 必须是 mean_max、candidate_cross_attention 或 mean_max_candidate_cross_attention_concat。
    - aggregation=mean_max: 对 A/P/V/PP 各自计算掩码均值和最大值，再由融合层产生候选表示。
    - aggregation=candidate_cross_attention: 用候选摘要分别查询 A/P/V/PP token，再融合四个注意力摘要。
    - aggregation=mean_max_candidate_cross_attention_concat: 拼接前两种完整摘要，并把零输出残差加到 mean/max 主表示。
    ``use_manual_features`` 控制候选聚合是否拼接四模态固化统计。
    ``use_manual_features`` 还控制身份打分是否拼接同一统计与每个身份槽所属类别的六分类概率。
    ``use_sampling_features`` 控制候选聚合是否拼接四项裁剪几何统计。
    ``use_sampling_features`` 还控制三个身份打分 MLP 是否拼接同一四项统计。
    ``use_candidate_token_attention_bias`` 控制候选条件注意力是否加入每模态独立的实体偏置 MLP。
    ``initial_graph_gate`` 是完整 Matcher 图身份分数增量的有限初始标量门，候选预训练必须为 0。
    ``initial_inventory_gate`` 是完整 Matcher 清单身份分数增量的有限初始标量门，候选预训练必须为 0。
    ``model_dim`` 是候选、token 和身份表示的共同宽度。
    ``attention_heads`` 是所有多头注意力的头数。
    ``dropout`` 是候选、图和身份分支共同采用的丢弃率。
    ``voxel_feature_channels`` 是 RAUNet 输出通道数。
    ``raunet_planes`` 依次为 ``(enc0,enc1,enc2,enc3,bottleneck,dec3,dec2,dec1,dec0)``。
    ``recycle_iterations`` 是 RAUNet 回收轮数。
    ``gradient_checkpointing`` 控制训练时是否以重计算换显存。
    ``bottleneck_attention_layers`` 是 RAUNet 瓶颈注意力层数。
    ``graph_layers`` 是配体图消息传递层数。
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
        """拒绝不能组成完整前向的结构参数。"""

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
        """返回适合写入 checkpoint 和 JSON 运行记录的普通字典。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MatcherModelConfig":
        """从配置字典恢复模型参数，并显式迁移第 0–4 批历史模型配置。

        第 0 批缺少两个特征开关，二者都迁移为 ``False``；第 1 批已有
        ``use_manual_features=True`` 而只缺采样开关，因此保留手动统计并仅把
        ``use_sampling_features`` 迁移为 ``False``。
        第 0–2 批均没有候选—实体注意力偏置，因此缺少
        ``use_candidate_token_attention_bias`` 时迁移为 ``False``。
        第 0–4 批没有显式身份门初值，因此缺少 ``initial_graph_gate`` 或
        ``initial_inventory_gate`` 时迁移为 ``0.0``，逐值保持原有零增量初始化。
        """

        normalized = dict(values)
        if "use_manual_features" not in normalized:
            normalized["use_manual_features"] = False
        if "use_sampling_features" not in normalized:
            normalized["use_sampling_features"] = False
        if "use_candidate_token_attention_bias" not in normalized:
            normalized["use_candidate_token_attention_bias"] = False
        if "initial_graph_gate" not in normalized:
            normalized["initial_graph_gate"] = 0.0
        if "initial_inventory_gate" not in normalized:
            normalized["initial_inventory_gate"] = 0.0
        if "raunet_planes" in normalized:
            raw_planes = normalized["raunet_planes"]
            if not isinstance(raw_planes, (list, tuple)):
                raise TypeError("raunet_planes 必须是整数列表。")
            normalized["raunet_planes"] = tuple(raw_planes)
        return cls(**normalized)
