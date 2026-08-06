"""验证 Stage3 可复用的概率点选择与密度特征采样。"""

from __future__ import annotations

import numpy as np
import torch

from matcher_v2.context import DensityPointBuilder, MapBackbone, topk_probability_points


def test_topk_is_limited_to_current_box() -> None:
    probability = np.zeros((6, 6, 6), dtype=np.float32)
    probability[0, 0, 0] = 1.0
    probability[3, 3, 3] = 0.8
    coordinate, score = topk_probability_points(
        probability, np.asarray([2, 2, 2]), map_size=3,
        voxel_size_xyz=np.ones(3, dtype=np.float32), top_k=1,
    )
    assert torch.equal(coordinate, torch.tensor([[1.5, 1.5, 1.5]]))
    assert torch.equal(score, torch.tensor([0.8]))


def test_density_point_features_keep_map_gradient() -> None:
    density = torch.randn(1, 1, 16, 16, 16, requires_grad=True)
    backbone = MapBackbone(
        input_channels=1, channels=(2, 4, 8, 8), norm_groups=2,
        bottleneck_heads=2, bottleneck_layers=1, bottleneck_ffn_dim=16,
        input_shape_zyx=(16, 16, 16),
    )
    features = backbone(density)
    builder = DensityPointBuilder((2, 4, 8, 8), output_dim=12)
    points = builder(
        features, (torch.tensor([[4.0, 4.0, 4.0]]),),
        (torch.tensor([0.7]),), (torch.tensor([16.0, 16.0, 16.0]),),
    )[0]
    points.feature.sum().backward()
    assert points.feature.shape == (1, 12)
    assert density.grad is not None
