"""由 ``CandidateEncoder`` 组合 RAUNet、四模态聚合和辅助头，并以 ``CandidateEncoderOutput`` 返回候选级结果。"""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn

from ..contracts import CandidateBatch
from ..schema import CLASS_NAMES
from .config import MatcherModelConfig
from .fusion import (
    CandidateConditionedAggregator,
    MeanMaxAggregator,
    MODALITY_INPUT_DIMS,
    TokenProjector,
    gather_voxel_features,
    normalize_candidate_manual_features,
    normalize_candidate_sampling_features,
)
from .raunet import RaUnetBackbone


class CandidateEncoderOutput(TypedDict):
    """令 ``B/D_m/C_v`` 为批次、模型宽度和体素特征通道：``candidate_embedding`` 是 ``(B,D_m)`` 候选表示；``class_logits`` 是 ``(B,6)`` 六分类分数；``ligand_area_logits`` 是 ``(B,48,48,48)`` 配体区域分数；``binding_area_logits`` 是 ``(B,48,48,48)`` 结合区域分数；``voxel_features`` 是 ``(B,C_v,48,48,48)`` RAUNet 特征。"""

    candidate_embedding: torch.Tensor
    class_logits: torch.Tensor
    ligand_area_logits: torch.Tensor
    binding_area_logits: torch.Tensor
    voxel_features: torch.Tensor


class VoxelBinaryHead(nn.Module):
    """从 U-Net 特征产生区域分数。构造参数 ``channels`` 是输入通道数；前向输入 ``(B,C_v,48,48,48)``，输出 ``(B,48,48,48)``。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(channels, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """返回去掉单通道轴的 ``(B,48,48,48)`` logits。"""

        return self.layers(features)[:, 0]


class CandidateEncoder(nn.Module):
    """编码单个 Stage1 候选。构造参数 ``config`` 定义 RAUNet、token 投影与聚合器；前向输入 ``CandidateBatch``，输出 ``CandidateEncoderOutput``；不读取配体语言或配体图。"""

    def __init__(self, config: MatcherModelConfig) -> None:
        super().__init__()
        self.config = config
        self.voxel_backbone = RaUnetBackbone(
            input_channels=105,
            feature_channels=config.voxel_feature_channels,
            planes=config.raunet_planes,
            recycle_iterations=config.recycle_iterations,
            gradient_checkpointing=config.gradient_checkpointing,
            bottleneck_attention_layers=config.bottleneck_attention_layers,
        )
        self.candidate_summary = nn.Sequential(
            nn.LayerNorm(config.voxel_feature_channels * 2),
            nn.Linear(config.voxel_feature_channels * 2, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )
        self.token_projectors = nn.ModuleDict(
            {
                name: TokenProjector(input_dim, config.model_dim, config.dropout)
                for name, input_dim in MODALITY_INPUT_DIMS.items()
            }
        )
        self.voxel_modulators = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(config.voxel_feature_channels, config.model_dim),
                    nn.GELU(),
                    nn.Linear(config.model_dim, config.model_dim),
                )
                for name in ("v", "pp")
            }
        )
        if config.aggregation == "mean_max":
            self.aggregator: nn.Module = MeanMaxAggregator(
                config.model_dim,
                config.dropout,
                config.use_manual_features,
                config.use_sampling_features,
            )
        else:
            self.aggregator = CandidateConditionedAggregator(
                config.model_dim,
                config.attention_heads,
                config.dropout,
                config.use_manual_features,
                config.use_sampling_features,
            )
        self.class_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim, len(CLASS_NAMES)),
        )
        self.ligand_area_head = VoxelBinaryHead(config.voxel_feature_channels)
        self.binding_area_head = VoxelBinaryHead(config.voxel_feature_channels)

    def forward(self, batch: CandidateBatch) -> CandidateEncoderOutput:
        """把候选体素与四模态字段编码为 ``CandidateEncoderOutput``。

        输入读取 ``voxel_input``、A/P/V/PP token 与掩码，以及 V/PP ZYX 索引；
        ``use_manual_features`` 启用时读取 ``candidate_manual_features``，
        ``use_sampling_features`` 启用时读取 ``candidate_sampling_features``。
        返回候选表示、六分类分数、两个区域分数和 U-Net 体素特征。
        """

        # 磁盘缓存是 float16；显式对齐主干参数 dtype 以兼容 CPU 测试，GPU autocast 仍可在算子内选择 BF16/FP16。
        voxel_input = batch["voxel_input"].to(
            dtype=self.voxel_backbone.input_projection.weight.dtype
        )
        voxel_features = self.voxel_backbone(voxel_input)
        pooled = torch.cat(
            (voxel_features.mean(dim=(2, 3, 4)), voxel_features.amax(dim=(2, 3, 4))),
            dim=-1,
        )
        candidate_summary = self.candidate_summary(pooled)

        a_projected = self.token_projectors["a"](batch["a_tokens"], batch["a_mask"])
        p_projected = self.token_projectors["p"](batch["p_tokens"], batch["p_mask"])
        v_projected = self.token_projectors["v"](batch["v_tokens"], batch["v_mask"])
        v_sampled = gather_voxel_features(voxel_features, batch["v_index_zyx"], batch["v_mask"])
        v_projected = v_projected + self.voxel_modulators["v"](v_sampled)
        v_projected = v_projected * batch["v_mask"][..., None].to(dtype=v_projected.dtype)
        pp_projected = self.token_projectors["pp"](batch["pp_tokens"], batch["pp_mask"])
        pp_sampled = gather_voxel_features(voxel_features, batch["pp_index_zyx"], batch["pp_mask"])
        pp_projected = pp_projected + self.voxel_modulators["pp"](pp_sampled)
        pp_projected = pp_projected * batch["pp_mask"][..., None].to(dtype=pp_projected.dtype)
        modalities = {
            "a": (a_projected, batch["a_mask"]),
            "p": (p_projected, batch["p_mask"]),
            "v": (v_projected, batch["v_mask"]),
            "pp": (pp_projected, batch["pp_mask"]),
        }

        normalized_manual = None
        if self.config.use_manual_features:
            normalized_manual = normalize_candidate_manual_features(
                batch["candidate_manual_features"]
            )
        normalized_sampling = None
        if self.config.use_sampling_features:
            normalized_sampling = normalize_candidate_sampling_features(
                batch["candidate_sampling_features"]
            )
        candidate_embedding = self.aggregator(
            candidate_summary,
            modalities,
            normalized_manual,
            normalized_sampling,
        )
        return {
            "candidate_embedding": candidate_embedding,
            "class_logits": self.class_head(candidate_embedding),
            "ligand_area_logits": self.ligand_area_head(voxel_features),
            "binding_area_logits": self.binding_area_head(voxel_features),
            "voxel_features": voxel_features,
        }
