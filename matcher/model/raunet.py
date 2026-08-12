"""由 ``RaUnetBackbone`` 把候选中心 ``(B,105,48,48,48)`` 体素输入编码为 ``(B,C_v,48,48,48)`` 特征。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


DEFAULT_PLANES = (64, 256, 256, 256, 256, 256, 128, 64, 64)


class ResidualBottleneck3d(nn.Module):
    """用 ``1×1×1 → 3×3×3 → 1×1×1`` 残差块完成一次下采样。"""

    expansion = 4

    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        super().__init__()
        if output_channels % self.expansion != 0:
            raise ValueError("ResidualBottleneck3d 的输出通道必须能被 4 整除。")
        hidden_channels = output_channels // self.expansion
        self.main = nn.Sequential(
            nn.Conv3d(input_channels, hidden_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.ReLU(),
            nn.Conv3d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.ReLU(),
            nn.Conv3d(hidden_channels, output_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(output_channels, affine=True),
        )
        self.shortcut: nn.Module
        if stride == 1 and input_channels == output_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv3d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """返回下采样后的五维体素特征。"""

        return self.activation(self.main(inputs) + self.shortcut(inputs))


class Res2NetBlock3d(nn.Module):
    """在不改变空间形状的前提下执行层级残差卷积。"""

    def __init__(self, channels: int, scale: int) -> None:
        super().__init__()
        if channels <= 0 or scale < 2:
            raise ValueError("Res2NetBlock3d 要求 channels>0 且 scale>=2。")
        self.channels = channels
        self.scale = scale
        self.expand = nn.Sequential(
            nn.Conv3d(channels, channels * scale, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels * scale, affine=True),
        )
        self.hierarchical = nn.ModuleList(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
            for _ in range(scale - 1)
        )
        self.pre_mix_norm = nn.InstanceNorm3d(channels * scale, affine=True)
        self.mix = nn.Sequential(
            nn.Conv3d(channels * scale, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """返回与 ``inputs`` 形状相同的层级残差特征。"""

        chunks = self.activation(self.expand(inputs)).chunk(self.scale, dim=1)
        outputs = [chunks[0]]
        for index, chunk in enumerate(chunks[1:]):
            source = chunk if index == 0 else chunk + outputs[-1]
            outputs.append(self.hierarchical[index](source))
        concatenated = self.pre_mix_norm(torch.cat(outputs, dim=1))
        mixed = self.mix(self.activation(concatenated))
        return self.activation(mixed + inputs)


class ConvResidual3d(nn.Module):
    """融合注意力门输出，并用两层卷积恢复局部空间信息。"""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(output_channels, affine=True),
            nn.ReLU(),
            nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(output_channels, affine=True),
        )
        self.shortcut = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv3d(input_channels, output_channels, kernel_size=1)
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """返回卷积主路和投影捷径之和。"""

        return self.activation(self.main(inputs) + self.shortcut(inputs))


class AttentionGate3d(nn.Module):
    """用粗分辨率特征控制同尺度跳跃特征的多头门值。"""

    def __init__(
        self,
        skip_channels: int,
        coarse_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        attention_channels = 64
        head_count = _largest_divisor_not_above(coarse_channels, 8)
        self.head_count = head_count
        self.value_channels_per_head = coarse_channels // head_count
        self.query = nn.Sequential(
            nn.Conv3d(coarse_channels, attention_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(attention_channels, affine=True),
        )
        self.key = nn.Sequential(
            nn.Conv3d(skip_channels, attention_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(attention_channels, affine=True),
        )
        self.value = nn.Sequential(
            nn.Conv3d(skip_channels, coarse_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(coarse_channels, affine=True),
        )
        self.gate = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(attention_channels, head_count, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse = ConvResidual3d(coarse_channels, output_channels)

    def forward(self, coarse: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """上采样 ``coarse``，门控 ``skip``，并返回融合特征。"""

        upsampled = F.interpolate(
            coarse,
            size=skip.shape[-3:],
            mode="trilinear",
            align_corners=True,
        )
        gate = self.gate(self.query(upsampled) + self.key(skip))
        value = self.value(skip)
        batch_size, _, depth, height, width = value.shape
        value = value.reshape(
            batch_size,
            self.value_channels_per_head,
            self.head_count,
            depth,
            height,
            width,
        )
        gated = (value * gate[:, None]).reshape_as(upsampled)
        return self.fuse(F.relu(gated + upsampled))


class GatedTransition(nn.Module):
    """在瓶颈 token 上执行带乘法门的通道前馈变换。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        expansion = 3
        self.left = nn.Linear(channels, channels * expansion, bias=False)
        self.right = nn.Linear(channels, channels * expansion, bias=False)
        self.output = nn.Linear(channels * expansion, channels, bias=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """返回门控通道变换和输入的归一化残差和。"""

        update = self.output(F.silu(self.left(inputs)) * self.right(inputs))
        return self.norm(inputs + update)


class RotarySelfAttention3d(nn.Module):
    """在 RAUNet 最深的三维规则网格上执行旋转位置编码自注意力。"""

    def __init__(self, channels: int, head_count: int) -> None:
        super().__init__()
        if channels % head_count != 0:
            raise ValueError("瓶颈通道数必须能被注意力头数整除。")
        # 原 RAUNet 为每个头分配 ``channels/head_count×6`` 个 Q/K/V 通道。
        head_channels = channels // head_count * 6
        if head_channels % 6 != 0:
            raise ValueError("三维旋转位置编码要求每个头的通道数能被 6 整除。")
        self.channels = channels
        self.head_count = head_count
        self.head_channels = head_channels
        projected_channels = head_count * head_channels
        self.query = nn.Linear(channels, projected_channels)
        self.key = nn.Linear(channels, projected_channels, bias=False)
        self.value = nn.Linear(channels, projected_channels, bias=False)
        self.output = nn.Linear(projected_channels, channels, bias=False)
        self.attention_norm = nn.LayerNorm(channels)
        self.transition = GatedTransition(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """保持五维网格形状，并仅在最深层空间位置之间建立全连接注意力。"""

        batch_size, channels, depth, height, width = inputs.shape
        tokens = inputs.flatten(2).transpose(1, 2)
        query = self._split_heads(self.query(tokens))
        key = self._split_heads(self.key(tokens))
        value = self._split_heads(self.value(tokens))
        positions = _grid_positions(depth, height, width, inputs.device, inputs.dtype)
        query = _apply_three_axis_rope(query, positions)
        key = _apply_three_axis_rope(key, positions)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch_size, -1, self.head_count * self.head_channels)
        tokens = self.attention_norm(tokens + self.output(attended))
        tokens = self.transition(tokens)
        return tokens.transpose(1, 2).reshape(batch_size, channels, depth, height, width)

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        """把 ``(B,L,H×C)`` 投影结果改排为 ``(B,H,L,C)``。"""

        batch_size, token_count, _ = values.shape
        return values.reshape(
            batch_size,
            token_count,
            self.head_count,
            self.head_channels,
        ).transpose(1, 2)


class RaUnetBackbone(nn.Module):
    """复现 Find_0 所用 RAUNet 宽度、四级编解码和三轮 recycle。"""

    def __init__(
        self,
        input_channels: int,
        feature_channels: int,
        planes: Sequence[int],
        recycle_iterations: int,
        gradient_checkpointing: bool,
        bottleneck_attention_layers: int,
    ) -> None:
        super().__init__()
        plane_values = tuple(int(value) for value in planes)
        if len(plane_values) != 9 or any(value <= 0 for value in plane_values):
            raise ValueError("planes 必须由九个正整数构成。")
        if recycle_iterations <= 0:
            raise ValueError("recycle_iterations 必须为正整数。")
        enc0, enc1, enc2, enc3, bottleneck, dec3, dec2, dec1, dec0 = plane_values
        if not (enc3 == bottleneck == dec3):
            raise ValueError("当前 RAUNet 契约要求 enc3、bottleneck 和 dec3 通道相同。")
        if dec0 != enc0:
            raise ValueError("原 RAUNet 直接回收 decoded0，因此 dec0 必须等于 enc0。")
        self.input_channels = input_channels
        self.feature_channels = feature_channels
        self.planes = plane_values
        self.recycle_iterations = recycle_iterations
        self.gradient_checkpointing = gradient_checkpointing

        self.input_projection = nn.Conv3d(input_channels, enc0, kernel_size=3, padding=1, bias=False)
        self.input_norm = nn.InstanceNorm3d(enc0, affine=False)
        self.recycle_norm = nn.InstanceNorm3d(enc0, affine=True)
        self.input_activation = nn.ELU()
        self.stem = nn.Sequential(
            nn.Conv3d(enc0, enc0 * 4, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(enc0 * 4, affine=True),
            nn.ReLU(),
        )
        self.down1 = ResidualBottleneck3d(enc0 * 4, enc1, stride=2)
        self.down2 = ResidualBottleneck3d(enc1, enc2, stride=2)
        self.down3 = ResidualBottleneck3d(enc2, enc3, stride=2)
        self.down4 = ResidualBottleneck3d(enc3, bottleneck, stride=2)
        head_count = _largest_divisor_not_above(bottleneck, 8)
        self.bottleneck_attention = nn.Sequential(
            *(RotarySelfAttention3d(bottleneck, head_count) for _ in range(bottleneck_attention_layers))
        )
        self.decode3 = _res2_stack(dec3, scale=3)
        self.gate2 = AttentionGate3d(enc2, dec3, dec2)
        self.decode2 = _res2_stack(dec2, scale=4)
        self.gate1 = AttentionGate3d(enc1, dec2, dec1)
        self.decode1 = _res2_stack(dec1, scale=4)
        self.gate0 = AttentionGate3d(enc0 * 4, dec1, dec0)
        self.decode0 = _res2_stack(dec0, scale=4)

        branch_channels = max(32, dec0 // 2)
        self.multiscale3 = nn.Conv3d(dec0, branch_channels, kernel_size=3, padding=1)
        self.multiscale5 = nn.Conv3d(dec0, branch_channels, kernel_size=5, padding=2)
        self.multiscale7 = nn.Conv3d(dec0, branch_channels, kernel_size=7, padding=3)
        self.feature_projection = nn.Conv3d(branch_channels * 3, feature_channels, kernel_size=3, padding=1)

    def forward(self, voxel_input: torch.Tensor) -> torch.Tensor:
        """返回 ``(B,C_feature,48,48,48)`` 的最高分辨率体素特征。"""

        if voxel_input.ndim != 5 or voxel_input.shape[1] != self.input_channels:
            raise ValueError(
                f"voxel_input 必须为 (B,{self.input_channels},D,H,W)，"
                f"实际为 {tuple(voxel_input.shape)}。"
            )
        recycle = torch.zeros(
            (voxel_input.shape[0], self.planes[0], *voxel_input.shape[-3:]),
            device=voxel_input.device,
            dtype=voxel_input.dtype,
        )
        final_feature: torch.Tensor | None = None
        for iteration in range(self.recycle_iterations):
            compute_gradients = iteration == self.recycle_iterations - 1
            if compute_gradients:
                final_feature, _ = self._forward_once(voxel_input, recycle)
            else:
                with torch.no_grad():
                    _, decoded0 = self._forward_once(voxel_input, recycle)
                recycle = decoded0.detach()
        if final_feature is None:
            raise AssertionError("至少应执行一次 RAUNet recycle。")
        return final_feature

    def _forward_once(
        self,
        voxel_input: torch.Tensor,
        recycle: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(B,C_v,D,H,W)`` 最终特征与供下轮回收的 ``(B,C_0,D,H,W)`` decoded0。"""

        fused = self.input_activation(
            self.input_norm(self.input_projection(voxel_input)) + self.recycle_norm(recycle)
        )
        level0 = self.stem(fused)
        level1 = self._heavy(self.down1, level0)
        level2 = self._heavy(self.down2, level1)
        level3 = self._heavy(self.down3, level2)
        level4 = self._heavy(self.down4, level3)
        deepest = self._heavy(self.bottleneck_attention, level4)
        decoded3 = self._heavy(self.decode3, _upsample_add(deepest, level3))
        decoded2 = self._heavy(self.decode2, self._heavy(self.gate2, decoded3, level2))
        decoded1 = self._heavy(self.decode1, self._heavy(self.gate1, decoded2, level1))
        decoded0 = self._heavy(self.decode0, self._heavy(self.gate0, decoded1, level0))
        multiscale = F.relu(
            torch.cat(
                (
                    self.multiscale3(decoded0),
                    self.multiscale5(decoded0),
                    self.multiscale7(decoded0),
                ),
                dim=1,
            )
        )
        return self.feature_projection(multiscale), decoded0

    def _heavy(self, module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
        """训练时按配置对显存占用较大的模块启用激活重计算。"""

        if not (self.gradient_checkpointing and self.training and torch.is_grad_enabled()):
            return module(*inputs)
        return checkpoint(module, *inputs, use_reentrant=False, preserve_rng_state=False)


def _res2_stack(channels: int, scale: int) -> nn.Sequential:
    """构造固定四层的同宽 Res2Net 解码段。"""

    return nn.Sequential(*(Res2NetBlock3d(channels, scale) for _ in range(4)))


def _upsample_add(coarse: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """把粗特征三线性插值到跳跃特征分辨率后逐元素相加。"""

    return skip + F.interpolate(
        coarse,
        size=skip.shape[-3:],
        mode="trilinear",
        align_corners=True,
    )


def _largest_divisor_not_above(value: int, maximum: int) -> int:
    """返回不超过 ``maximum`` 且能整除 ``value`` 的最大正整数。"""

    for candidate in range(min(value, maximum), 0, -1):
        if value % candidate == 0:
            return candidate
    raise AssertionError("正整数至少能被 1 整除。")


def _grid_positions(
    depth: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """按展平顺序返回 ``(D×H×W,3)`` 的 ZYX 网格坐标。"""

    axes = [torch.arange(size, device=device, dtype=dtype) for size in (depth, height, width)]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3) * 1.5


def _apply_three_axis_rope(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """把每个注意力头等分给 Z、Y、X 三轴并应用成对旋转。"""

    axis_channels = values.shape[-1] // 3
    if axis_channels % 2 != 0:
        raise ValueError("每个空间轴的旋转位置编码通道必须为偶数。")
    rotated_parts = []
    frequency_index = torch.arange(
        axis_channels // 2,
        device=values.device,
        dtype=torch.float32,
    )
    inverse_frequency = torch.exp(
        -math.log(100.0) * frequency_index / max(axis_channels // 2, 1)
    ).to(dtype=values.dtype)
    for axis, part in enumerate(values.split(axis_channels, dim=-1)):
        angles = positions[:, axis, None] * inverse_frequency[None, :]
        cosine = angles.cos()[None, None]
        sine = angles.sin()[None, None]
        even = part[..., 0::2]
        odd = part[..., 1::2]
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine),
            dim=-1,
        ).flatten(-2)
        rotated_parts.append(rotated)
    return torch.cat(rotated_parts, dim=-1)
