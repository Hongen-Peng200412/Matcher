"""按四个具名变体把 Stage1 候选编码成候选表示和前景分数.

主要入口是 CandidateEncoder.forward. strongest/trunk_only 使用完整 106 通道体素与四类
token, density_blob_only 使用单实验密度通道与 V token, pocket_only 只使用原始 A/L0.
本模块返回内存张量, 不读取清单、不计算损失且不写入磁盘.
"""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn

from ..data.schema import A_TOKEN_DIM, P_TOKEN_DIM, PP_TOKEN_DIM, V_TOKEN_DIM
from ..datasets.contracts import CandidateBatch
from .config import MatcherModelConfig
from .fusion import (
    MeanMaxAggregator,
    MeanMaxCandidateConditionedResidualAggregator,
    TokenProjector,
    gather_voxel_features,
    normalize_candidate_manual_features,
    normalize_candidate_sampling_features,
)
from .raunet import RaUnetBackbone


class CandidateEncoderOutput(TypedDict):
    """
    保存候选编码器的数值输出.

    字段:
    - candidate_embedding: 浮点张量, (B,D_m); 候选统一表示.
    - foreground_logit: 浮点张量, (B,); 真实五类前景相对背景的二分类 logit.
    - ligand_area_logits: 浮点张量或 None; 有区域主干时为 (B,48,48,48).
    - binding_area_logits: 浮点张量或 None; 有区域主干时为 (B,48,48,48).
    - voxel_features: 浮点张量或 None; 有 U-Net 时为 (B,C_v,48,48,48).
    """
    candidate_embedding: torch.Tensor
    foreground_logit: torch.Tensor
    ligand_area_logits: torch.Tensor | None
    binding_area_logits: torch.Tensor | None
    voxel_features: torch.Tensor | None


class VoxelBinaryHead(nn.Module):
    """
    从同分辨率三维特征生成区域 logit.

    - 输入: 浮点 (B,C_v,Z,Y,X) U-Net 特征.
    - 输出: 浮点 (B,Z,Y,X); 每个体素一个未归一化二分类分数.
    """
    def __init__(self, channels: int) -> None:
        """
        建立两个一乘一乘一卷积.

        - channels: 输入与隐藏通道数 C_v.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(channels, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        执行逐体素区域分类.

        - features: 浮点 (B,C_v,Z,Y,X); 空间分辨率保持不变.
        - return: 浮点 (B,Z,Y,X); 删除卷积输出的单通道轴.
        """
        return self.layers(features)[:, 0]


class PocketMeanMaxEncoder(nn.Module):
    """
    编码 pocket-only 的 A-L0 token.

    - 输入: 浮点 (B,N_A,50) 原始 L0 特征和 bool (B,N_A) 有效掩码.
    - 输出: 浮点 (B,D_m); 由有效 token 的逐通道均值与最大值融合得到.
    """
    def __init__(self, model_dim: int, dropout: float) -> None:
        """
        建立 pocket-only 投影和融合层.

        - model_dim: 正整数 D_m; 逐 token 投影和候选输出宽度.
        - dropout: 融合多层感知机的丢弃概率.
        - return: None; 建立 50→D_m TokenProjector 和 2D_m→D_m 融合层.
        """
        super().__init__()
        self.projector = TokenProjector(50, model_dim, dropout)
        self.fusion = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        把浮点 (B,N_A,50) A-L0 token 归约为浮点 (B,D_m) 表示.

        - tokens: A token 原始 L0 特征, 不含概率、坐标或 L1/L2/L3.
        - mask: bool (B,N_A) 有效位置; 全空候选得到零 mean/max 输入.
        - return: 浮点 (B,D_m) pocket-only 候选表示.
        """
        projected = self.projector(tokens, mask)
        if projected.shape[1] == 0:
            mean = torch.zeros(
                (projected.shape[0], projected.shape[2]),
                dtype=projected.dtype,
                device=projected.device,
            )
            maximum = mean
        else:
            weights = mask[..., None].to(dtype=projected.dtype)
            mean = (projected * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            maximum = projected.masked_fill(
                ~mask[..., None],
                torch.finfo(projected.dtype).min,
            ).amax(dim=1)
            maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        return self.fusion(torch.cat((mean, maximum), dim=-1))


class CandidateEncoder(nn.Module):
    """
    建立当前具名变体实际需要的候选模块.

    strongest 与 trunk_only 使用 106 通道 U-Net、A/P/V/PP 和摘要拼接; density_blob_only
    使用单实验密度通道、V token、V 三项统计与四项采样统计; pocket_only 只实例化
    PocketMeanMaxEncoder. 被禁用的 token 投影、U-Net 或区域头不会进入参数树.
    """
    def __init__(self, config: MatcherModelConfig) -> None:
        """
        按具名变体建立互斥的候选编码路径.

        - config: 已通过 MatcherModelConfig 校验的结构配置; variant 决定 U-Net 输入、token 投影、聚合器、区域辅助头和 pocket-only 编码器是否进入参数树.
        - self.foreground_head: 四个变体共用的二分类头; 把浮点 (B,D_m) 候选表示映射为 浮点 (B,1), 前向返回时去掉最后一个单例轴.
        - return: None; 所有实际使用的子模块写入当前实例, 禁用模块保持 None 或空 ModuleDict.
        """
        super().__init__()
        self.config = config
        self.voxel_backbone: RaUnetBackbone | None = None
        self.candidate_summary: nn.Module | None = None
        self.token_projectors = nn.ModuleDict()
        self.voxel_modulators = nn.ModuleDict()
        self.aggregator: nn.Module | None = None
        self.pocket_encoder: PocketMeanMaxEncoder | None = None
        self.ligand_area_head: VoxelBinaryHead | None = None
        self.binding_area_head: VoxelBinaryHead | None = None

        if config.variant == "pocket_only":
            self.pocket_encoder = PocketMeanMaxEncoder(config.model_dim, config.dropout)
        else:
            input_channels = 1 if config.variant == "density_blob_only" else 106
            self.voxel_backbone = RaUnetBackbone(
                input_channels=input_channels,
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
            input_dims = (
                {"v": V_TOKEN_DIM}
                if config.variant == "density_blob_only"
                else {"a": A_TOKEN_DIM, "p": P_TOKEN_DIM, "v": V_TOKEN_DIM, "pp": PP_TOKEN_DIM}
            )
            self.token_projectors.update(
                {
                    name: TokenProjector(width, config.model_dim, config.dropout)
                    for name, width in input_dims.items()
                }
            )
            modulated_names = ("v",) if config.variant == "density_blob_only" else ("v", "pp")
            self.voxel_modulators.update(
                {
                    name: nn.Sequential(
                        nn.Linear(config.voxel_feature_channels, config.model_dim),
                        nn.GELU(),
                        nn.Linear(config.model_dim, config.model_dim),
                    )
                    for name in modulated_names
                }
            )
            if config.variant == "density_blob_only":
                self.aggregator = MeanMaxAggregator(
                    config.model_dim,
                    config.dropout,
                    include_manual=True,
                    include_sampling=True,
                )
            else:
                self.aggregator = MeanMaxCandidateConditionedResidualAggregator(
                    config.model_dim,
                    config.attention_heads,
                    config.dropout,
                    include_manual=True,
                    include_sampling=True,
                )
            self.ligand_area_head = VoxelBinaryHead(config.voxel_feature_channels)
            self.binding_area_head = VoxelBinaryHead(config.voxel_feature_channels)

        self.foreground_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim, 1),
        )

    def _empty_modality(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        建立当前设备上的空投影 token 与掩码.

        - batch_size: 物理批次大小 B.
        - dtype: 候选表示浮点类型.
        - device: 候选表示所在设备.
        - return: 浮点 (B,0,D_m) 与 bool (B,0).
        """
        return (
            torch.empty((batch_size, 0, self.config.model_dim), dtype=dtype, device=device),
            torch.empty((batch_size, 0), dtype=torch.bool, device=device),
        )

    def forward(self, batch: CandidateBatch) -> CandidateEncoderOutput:
        """
        执行当前变体唯一允许的候选前向.

        - batch: 与 config.variant 同源的 CandidateBatch; 候选轴大小为 B. strongest 和 trunk_only 读取浮点 (B,106,48,48,48) 体素输入及 A/P/V/PP, density_blob_only 只读取浮点 (B,1,48,48,48) 实验密度和 V, pocket_only 只读取 A token 的前 50 个 L0 特征.
        - return.candidate_embedding: 浮点 (B,D_m); 后续身份打分和二分类共用的候选表示.
        - return.foreground_logit: 浮点 (B,); 五类真实配体相对背景的未归一化二分类分数.
        - return.ligand_area_logits / binding_area_logits: 有 U-Net 时为浮点 (B,48,48,48), pocket_only 时为 None.
        - return.voxel_features: 有 U-Net 时为浮点 (B,C_v,48,48,48), pocket_only 时为 None.
        """
        if self.config.variant == "pocket_only":
            if self.pocket_encoder is None:
                raise RuntimeError("pocket_only 编码器未建立。")
            candidate_embedding = self.pocket_encoder(batch["a_tokens"], batch["a_mask"])
            voxel_features = None
            ligand_area_logits = None
            binding_area_logits = None
        else:
            if self.voxel_backbone is None or self.candidate_summary is None or self.aggregator is None:
                raise RuntimeError("U-Net 候选路径未完整建立。")
            voxel_input = batch["voxel_input"].to(
                dtype=self.voxel_backbone.input_projection.weight.dtype
            )
            voxel_features = self.voxel_backbone(voxel_input)
            pooled = torch.cat(
                (voxel_features.mean(dim=(2, 3, 4)), voxel_features.amax(dim=(2, 3, 4))),
                dim=-1,
            )
            candidate_summary = self.candidate_summary(pooled)
            empty = self._empty_modality(
                candidate_summary.shape[0],
                candidate_summary.dtype,
                candidate_summary.device,
            )
            modalities: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
                "a": empty,
                "p": empty,
                "v": empty,
                "pp": empty,
            }
            names = ("v",) if self.config.variant == "density_blob_only" else ("a", "p", "v", "pp")
            for name in names:
                tokens = batch[f"{name}_tokens"]
                mask = batch[f"{name}_mask"]
                projected = self.token_projectors[name](tokens, mask)
                if name == "v":
                    if "v" not in self.voxel_modulators:
                        raise RuntimeError("V token 体素调制器未建立。")
                    sampled = gather_voxel_features(voxel_features, batch["v_index_zyx"], mask)
                    projected = projected + self.voxel_modulators["v"](sampled)
                    projected = projected * mask[..., None].to(dtype=projected.dtype)
                elif name == "pp":
                    if "pp" not in self.voxel_modulators:
                        raise RuntimeError("PP token 体素调制器未建立。")
                    sampled = gather_voxel_features(voxel_features, batch["pp_index_zyx"], mask)
                    projected = projected + self.voxel_modulators["pp"](sampled)
                    projected = projected * mask[..., None].to(dtype=projected.dtype)
                modalities[name] = (projected, mask)
            manual = normalize_candidate_manual_features(batch["candidate_manual_features"])
            sampling = normalize_candidate_sampling_features(batch["candidate_sampling_features"])
            candidate_embedding = self.aggregator(
                candidate_summary,
                modalities,
                manual,
                sampling,
            )
            if self.ligand_area_head is None or self.binding_area_head is None:
                raise RuntimeError("区域辅助头未建立。")
            ligand_area_logits = self.ligand_area_head(voxel_features)
            binding_area_logits = self.binding_area_head(voxel_features)
        return {
            "candidate_embedding": candidate_embedding,
            "foreground_logit": self.foreground_head(candidate_embedding)[:, 0],
            "ligand_area_logits": ligand_area_logits,
            "binding_area_logits": binding_area_logits,
            "voxel_features": voxel_features,
        }
