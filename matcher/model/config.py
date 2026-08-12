"""用 ``MatcherModelConfig`` 定义模型结构，并与 checkpoint/JSON 中的普通字典相互转换。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

AggregationName = Literal["mean_max", "candidate_cross_attention"]


@dataclass(frozen=True)
class MatcherModelConfig:
    """模型配置：``aggregation`` 是 ``mean_max/candidate_cross_attention``；``model_dim/attention_heads/dropout`` 是共同宽度、头数和丢弃率；``voxel_feature_channels`` 是 RAUNet 输出宽度；``raunet_planes`` 顺序为 ``(enc0,enc1,enc2,enc3,bottleneck,dec3,dec2,dec1,dec0)``；``recycle_iterations`` 是回收轮数；``gradient_checkpointing`` 决定训练时是否激活重计算；``bottleneck_attention_layers/graph_layers`` 是瓶颈注意力与配体图消息层数。"""

    aggregation: AggregationName
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

        if self.aggregation not in ("mean_max", "candidate_cross_attention"):
            raise ValueError(f"不支持 aggregation={self.aggregation!r}。")
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
        """从配置字典恢复模型参数，并把 RAUNet 通道列表冻结为元组。"""

        normalized = dict(values)
        if "raunet_planes" in normalized:
            raw_planes = normalized["raunet_planes"]
            if not isinstance(raw_planes, (list, tuple)):
                raise TypeError("raunet_planes 必须是整数列表。")
            normalized["raunet_planes"] = tuple(raw_planes)
        return cls(**normalized)
