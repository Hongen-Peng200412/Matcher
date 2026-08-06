"""Matcher 与 Stage3 可共同读取的密度特征和采样点叶子模块。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from matcher.map import MapBackbone, MapFeatures, MapSummaryHead


@dataclass(frozen=True)
class DensityPoints:
    """候选局部 XYZ 坐标、概率和 U-Net 多尺度采样特征。"""

    coordinate_local_xyz: Tensor
    probability: Tensor
    feature: Tensor


class DensityPointBuilder(nn.Module):
    """在给定坐标采样四尺度 U-Net 特征，不负责选择 top-k 身份。"""

    def __init__(self, channels: tuple[int, int, int, int], output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(sum(channels) + 1),
            nn.Linear(sum(channels) + 1, output_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        features: MapFeatures,
        coordinate_local_xyz: tuple[Tensor, ...],
        probability: tuple[Tensor, ...],
        box_size_xyz: tuple[Tensor, ...],
    ) -> tuple[DensityPoints, ...]:
        """坐标以 48³ BOX 角点为零点，`box_size_xyz` 为物理边长。"""

        result = []
        maps = (
            features.bottleneck,
            features.decoder_quarter,
            features.decoder_half,
            features.decoder_full,
        )
        for batch_index, (coordinate, score, physical_size) in enumerate(
            zip(coordinate_local_xyz, probability, box_size_xyz, strict=True)
        ):
            if coordinate.shape[0] == 0:
                result.append(DensityPoints(coordinate, score, coordinate.new_empty((0, self.projection[-2].out_features))))
                continue
            normalized = 2.0 * coordinate / physical_size - 1.0
            grid = normalized[:, [2, 1, 0]].reshape(1, -1, 1, 1, 3)
            sampled = [
                F.grid_sample(value[batch_index : batch_index + 1], grid, align_corners=False)
                .squeeze(0).squeeze(-1).squeeze(-1).transpose(0, 1)
                for value in maps
            ]
            raw = torch.cat((*sampled, score[:, None]), dim=-1)
            result.append(DensityPoints(coordinate, score, self.projection(raw)))
        return tuple(result)


def topk_probability_points(
    probability_map: np.ndarray,
    map_start_zyx: np.ndarray,
    *,
    map_size: int,
    voxel_size_xyz: np.ndarray,
    top_k: int,
) -> tuple[Tensor, Tensor]:
    """只在当前候选 48³ 可见区域选择完整图概率最高的点。"""

    full_shape = np.asarray(probability_map.shape, dtype=np.int64)
    start = np.asarray(map_start_zyx, dtype=np.int64)
    source_start = np.maximum(start, 0)
    source_end = np.minimum(start + map_size, full_shape)
    region = probability_map[
        source_start[0] : source_end[0],
        source_start[1] : source_end[1],
        source_start[2] : source_end[2],
    ]
    count = min(int(top_k), int(region.size))
    if count == 0:
        return torch.empty(0, 3), torch.empty(0)
    flat = np.argpartition(region.ravel(), -count)[-count:]
    global_zyx = np.stack(np.unravel_index(flat, region.shape), axis=-1) + source_start
    local_xyz = ((global_zyx - start)[:, ::-1] + 0.5) * voxel_size_xyz
    probability = probability_map[tuple(global_zyx.T)]
    return (
        torch.from_numpy(np.asarray(local_xyz, dtype=np.float32)),
        torch.from_numpy(np.asarray(probability, dtype=np.float32)),
    )


__all__ = [
    "DensityPointBuilder", "DensityPoints", "MapBackbone", "MapFeatures",
    "MapSummaryHead", "topk_probability_points",
]
