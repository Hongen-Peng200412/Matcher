"""用 ``MatcherModelConfig`` 定义模型结构，并与 checkpoint/JSON 中的普通字典相互转换。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

AggregationName = Literal["mean_max", "candidate_cross_attention"]


@dataclass(frozen=True)
class MatcherModelConfig:
    """保存候选预训练与完整 Matcher 的聚合方式、模型宽度、头数、dropout、体素特征宽度、RAUNet 九级通道、recycle 次数、checkpoint 开关、瓶颈注意力层数和图消息层数。"""

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
        if self.attention_heads <= 0:
            raise ValueError("attention_heads 必须为正整数。")
        if self.model_dim <= 0 or self.model_dim % self.attention_heads != 0:
            raise ValueError("model_dim 必须为正数且能被 attention_heads 整除。")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0,1)。")
        if self.voxel_feature_channels <= 0:
            raise ValueError("voxel_feature_channels 必须为正整数。")
        if len(self.raunet_planes) != 9 or any(value <= 0 for value in self.raunet_planes):
            raise ValueError("raunet_planes 必须含九个正整数通道宽度。")
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
            normalized["raunet_planes"] = tuple(int(value) for value in normalized["raunet_planes"])
        return cls(**normalized)
