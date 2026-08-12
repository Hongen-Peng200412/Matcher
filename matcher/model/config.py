"""定义模型结构的可序列化配置，并在构造前集中验证跨模块约束。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .raunet import DEFAULT_PLANES


AggregationName = Literal["mean_max", "candidate_cross_attention"]


@dataclass(frozen=True)
class MatcherModelConfig:
    """保存候选预训练和完整 Matcher 共用的模型结构参数。"""

    aggregation: AggregationName
    model_dim: int = 256
    attention_heads: int = 8
    dropout: float = 0.10
    voxel_feature_channels: int = 48
    raunet_planes: tuple[int, ...] = DEFAULT_PLANES
    recycle_iterations: int = 3
    gradient_checkpointing: bool = True
    bottleneck_attention_layers: int = 4
    graph_layers: int = 4

    def __post_init__(self) -> None:
        """拒绝不能组成完整前向的结构参数。"""

        if self.aggregation not in ("mean_max", "candidate_cross_attention"):
            raise ValueError(f"不支持 aggregation={self.aggregation!r}。")
        if self.model_dim <= 0 or self.model_dim % self.attention_heads != 0:
            raise ValueError("model_dim 必须为正数且能被 attention_heads 整除。")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0,1)。")
        if self.voxel_feature_channels <= 0:
            raise ValueError("voxel_feature_channels 必须为正整数。")
        if len(self.raunet_planes) != 9:
            raise ValueError("raunet_planes 必须含九个通道宽度。")
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
