"""定义四个且仅四个 Matcher 模型变体及其共同结构参数.

主要入口是 MatcherModelConfig. 该 dataclass 校验 YAML 或 checkpoint 提供的结构字段,
并向候选编码器与身份编码器暴露少量派生开关. 本模块不实例化模型且不读写磁盘.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal


ModelVariant = Literal[
    "strongest",
    "density_blob_only",
    "pocket_only",
    "trunk_only",
]

MODEL_VARIANTS = (
    "strongest",
    "density_blob_only",
    "pocket_only",
    "trunk_only",
)


@dataclass(frozen=True)
class MatcherModelConfig:
    """
    保存具名模型变体和四项实验共同使用的结构超参数.

    配置字段:
    - variant: 四个正式变体之一; 该字段同时决定候选输入和身份增量模块.
    - model_dim: 候选与身份表示共同宽度 D_m, 必须能被 attention_heads 整除.
    - attention_heads: 候选条件、配体图和身份清单注意力的头数.
    - dropout: 模型丢弃概率, 范围为 [0,1).
    - voxel_feature_channels: U-Net 最高分辨率输出宽度 C_v.
    - raunet_planes: 九项 U-Net 编解码通道宽度.
    - recycle_iterations: U-Net 回收次数.
    - gradient_checkpointing: 是否在训练时启用激活重计算.
    - bottleneck_attention_layers: U-Net 瓶颈注意力层数.
    - graph_layers: SMILES 图消息传递层数; trunk_only 不实例化对应模块.

    图分支和身份清单分支的门初值固定为 0, 不作为 YAML 可变字段. strongest 与 trunk_only
    使用 mean/max 和候选条件注意力摘要拼接; density_blob_only 使用单密度 U-Net、V token、
    V 三项统计和四项采样统计; pocket_only 只使用 A token 的原始 50 维 L0.
    """
    variant: ModelVariant
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
        """
        在模型构造前核对有限的结构约束.

        - return: 全部字段合法时返回 None; 否则抛出 TypeError 或 ValueError.
        """
        if self.variant not in MODEL_VARIANTS:
            raise ValueError(f"不支持 model.variant={self.variant!r}。")
        integer_fields = (
            self.model_dim,
            self.attention_heads,
            self.voxel_feature_channels,
            self.recycle_iterations,
            self.bottleneck_attention_layers,
            self.graph_layers,
        )
        if any(type(value) is not int or value <= 0 for value in integer_fields):
            raise ValueError("模型宽度、头数、通道数、回收次数和层数必须是正整数。")
        if self.model_dim % self.attention_heads != 0:
            raise ValueError("model_dim 必须能被 attention_heads 整除。")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("dropout 必须是数值。")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout 必须位于 [0,1)。")
        if (
            not isinstance(self.raunet_planes, tuple)
            or len(self.raunet_planes) != 9
            or any(type(value) is not int or value <= 0 for value in self.raunet_planes)
        ):
            raise ValueError("raunet_planes 必须含九个正整数。")
        if type(self.gradient_checkpointing) is not bool:
            raise TypeError("gradient_checkpointing 必须是布尔值。")

    @property
    def use_manual_features(self) -> bool:
        """
        判断是否消费 12 项候选手动统计.

        - return: pocket_only 为 False, 其余三个变体为 True.
        """
        return self.variant != "pocket_only"

    @property
    def use_sampling_features(self) -> bool:
        """
        判断是否消费四项候选裁剪统计.

        - return: pocket_only 为 False, 其余三个变体为 True.
        """
        return self.variant != "pocket_only"

    @property
    def uses_identity_increments(self) -> bool:
        """
        判断是否实例化两个身份增量分支.

        - return: trunk_only 为 False, 其余三个变体为 True.
        """
        return self.variant != "trunk_only"

    def to_dict(self) -> dict[str, Any]:
        """
        序列化当前结构配置.

        - return: 只含正式 dataclass 字段的普通字典, 供 checkpoint 保存.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MatcherModelConfig":
        """
        从当前版本字典恢复模型配置.

        - values: checkpoint 或 YAML 提供的完整正式字段; raunet_planes 可以是列表或元组.
        - return: 经 dataclass 边界检查的 MatcherModelConfig, 不迁移历史开关.
        """
        normalized = dict(values)
        planes = normalized.get("raunet_planes")
        if isinstance(planes, list):
            normalized["raunet_planes"] = tuple(planes)
        return cls(**normalized)
