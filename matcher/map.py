"""中心 48³ 实验密度的完整小型 U-Net 与四尺度摘要。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class MapFeatures:
    """按输入空间的 1/8、1/4、1/2、原尺度保存四个 ``[B,C_k,D_k,H_k,W_k]`` 特征。"""

    bottleneck: Tensor
    decoder_quarter: Tensor
    decoder_half: Tensor
    decoder_full: Tensor


class ResidualConvBlock3d(nn.Module):
    """两层 pre-activation 3D 卷积残差块。"""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        input_groups: int = 8,
        output_groups: int = 8,
    ) -> None:
        super().__init__()
        if input_channels % input_groups or output_channels % output_groups:
            raise ValueError(
                "GroupNorm 组数必须分别整除对应通道数："
                f"{input_groups=}、{output_groups=}、{input_channels=}、{output_channels=}"
            )
        self.body = nn.Sequential(
            nn.GroupNorm(input_groups, input_channels),
            nn.SiLU(),
            nn.Conv3d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(output_groups, output_channels),
            nn.SiLU(),
            nn.Conv3d(output_channels, output_channels, 3, padding=1),
        )
        self.residual = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv3d(input_channels, output_channels, 1)
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.residual(value) + self.body(value)


class MapBackbone(nn.Module):
    """输入 ``[B,input_channels,D,H,W]``，返回完整 decoder 的四尺度特征。

    三个空间维必须能被 8 整除；当前 Anchor 配置使用单通道 48³ Map。
    """

    def __init__(
        self,
        channels: tuple[int, int, int, int] = (16, 32, 48, 64),
        *,
        input_channels: int = 1,
        input_shape_zyx: tuple[int, int, int] = (48, 48, 48),
        norm_groups: int = 8,
        bottleneck_heads: int = 4,
        bottleneck_layers: int = 1,
        bottleneck_ffn_dim: int = 128,
        dropout: float = 0.1,
        activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("MapBackbone.channels 必须恰好包含四个尺度。")
        if any(axis % 8 for axis in input_shape_zyx):
            raise ValueError(f"Map 三轴必须能被 8 整除，收到 {input_shape_zyx}。")
        if channels[-1] % bottleneck_heads:
            raise ValueError("Map bottleneck 通道数必须能被 Transformer head 数整除。")
        if norm_groups <= 0:
            raise ValueError("norm_groups 必须为正整数。")
        c0, c1, c2, c3 = channels
        self.channels = channels
        self.input_channels = input_channels
        self.input_shape_zyx = input_shape_zyx
        self.activation_checkpoint = activation_checkpoint
        self.encoder0 = ResidualConvBlock3d(
            input_channels, c0, 1, norm_groups
        )
        self.down0 = nn.Conv3d(c0, c1, 2, stride=2)
        self.encoder1 = ResidualConvBlock3d(c1, c1, norm_groups, norm_groups)
        self.down1 = nn.Conv3d(c1, c2, 2, stride=2)
        self.encoder2 = ResidualConvBlock3d(c2, c2, norm_groups, norm_groups)
        self.down2 = nn.Conv3d(c2, c3, 2, stride=2)
        self.encoder3 = ResidualConvBlock3d(c3, c3, norm_groups, norm_groups)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=c3,
            nhead=bottleneck_heads,
            dim_feedforward=bottleneck_ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.bottleneck_transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=bottleneck_layers
        )
        self.position_projection = nn.Linear(3, c3)

        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.decoder2 = ResidualConvBlock3d(c2 + c2, c2, norm_groups, norm_groups)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.decoder1 = ResidualConvBlock3d(c1 + c1, c1, norm_groups, norm_groups)
        self.up0 = nn.ConvTranspose3d(c1, c0, 2, stride=2)
        self.decoder0 = ResidualConvBlock3d(c0 + c0, c0, norm_groups, norm_groups)

    def _block(self, module: nn.Module, value: Tensor) -> Tensor:
        if self.training and self.activation_checkpoint:
            return checkpoint(module, value, use_reentrant=False, preserve_rng_state=True)
        return module(value)

    def forward(self, density: Tensor) -> MapFeatures:
        expected = (self.input_channels, *self.input_shape_zyx)
        if density.ndim != 5 or tuple(density.shape[1:]) != expected:
            raise ValueError(f"Map 输入应为 [B,{','.join(map(str, expected))}]，收到 {tuple(density.shape)}。")
        encoder0 = self._block(self.encoder0, density)
        encoder1 = self._block(self.encoder1, self.down0(encoder0))
        encoder2 = self._block(self.encoder2, self.down1(encoder1))
        bottleneck = self._block(self.encoder3, self.down2(encoder2))

        batch, channels, depth, height, width = bottleneck.shape
        z, y, x = torch.meshgrid(
            torch.linspace(-1, 1, depth, device=density.device, dtype=density.dtype),
            torch.linspace(-1, 1, height, device=density.device, dtype=density.dtype),
            torch.linspace(-1, 1, width, device=density.device, dtype=density.dtype),
            indexing="ij",
        )
        position_xyz = torch.stack((x, y, z), dim=-1).reshape(1, -1, 3)
        tokens = bottleneck.flatten(2).transpose(1, 2)
        tokens = self.bottleneck_transformer(tokens + self.position_projection(position_xyz))
        bottleneck = tokens.transpose(1, 2).reshape(batch, channels, depth, height, width)

        quarter = self._block(self.decoder2, torch.cat((self.up2(bottleneck), encoder2), dim=1))
        half = self._block(self.decoder1, torch.cat((self.up1(quarter), encoder1), dim=1))
        full = self._block(self.decoder0, torch.cat((self.up0(half), encoder0), dim=1))
        return MapFeatures(bottleneck, quarter, half, full)


class MapSummaryHead(nn.Module):
    """以 mean+max 汇聚四尺度 decoder 特征，得到每个候选的 Map_repr。"""

    def __init__(
        self,
        channels: tuple[int, int, int, int],
        *,
        scale_summary_dim: int = 64,
        summary_hidden_dim: int = 512,
        repr_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        scale_channels = (channels[3], channels[2], channels[1], channels[0])
        self.scale_projections = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, scale_summary_dim), nn.SiLU())
            for width in scale_channels
        )
        self.output = nn.Sequential(
            nn.LayerNorm(4 * scale_summary_dim),
            nn.Linear(4 * scale_summary_dim, summary_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(summary_hidden_dim, repr_dim),
            nn.LayerNorm(repr_dim),
        )

    def forward(self, features: MapFeatures) -> Tensor:
        summaries = []
        ordered = (
            features.bottleneck,
            features.decoder_quarter,
            features.decoder_half,
            features.decoder_full,
        )
        for feature, projection in zip(ordered, self.scale_projections, strict=True):
            mean = feature.mean(dim=(2, 3, 4))
            maximum = feature.amax(dim=(2, 3, 4))
            summaries.append(projection(torch.cat((mean, maximum), dim=-1)))
        return self.output(torch.cat(summaries, dim=-1))
