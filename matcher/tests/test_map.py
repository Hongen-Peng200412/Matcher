from __future__ import annotations

import pytest
import torch

from matcher.map import MapBackbone, MapSummaryHead


def test_map_backbone_returns_four_decoder_scales() -> None:
    backbone = MapBackbone(
        (4, 8, 12, 16), norm_groups=4, bottleneck_heads=4, bottleneck_ffn_dim=32
    )
    summary = MapSummaryHead((4, 8, 12, 16), scale_summary_dim=8, summary_hidden_dim=32)

    features = backbone(torch.randn(2, 1, 48, 48, 48))
    Map_repr = summary(features)

    assert features.bottleneck.shape == (2, 16, 6, 6, 6)
    assert features.decoder_quarter.shape == (2, 12, 12, 12, 12)
    assert features.decoder_half.shape == (2, 8, 24, 24, 24)
    assert features.decoder_full.shape == (2, 4, 48, 48, 48)
    assert Map_repr.shape == (2, 256)


def test_map_checkpoint_preserves_outputs_and_input_gradient() -> None:
    plain = MapBackbone(
        (2, 4, 6, 8),
        norm_groups=2,
        bottleneck_heads=2,
        bottleneck_ffn_dim=16,
        dropout=0.0,
    ).train()
    checked = MapBackbone(
        (2, 4, 6, 8),
        norm_groups=2,
        bottleneck_heads=2,
        bottleneck_ffn_dim=16,
        dropout=0.0,
        activation_checkpoint=True,
    ).train()
    checked.load_state_dict(plain.state_dict())
    plain_input = torch.randn(1, 1, 48, 48, 48, requires_grad=True)
    checked_input = plain_input.detach().clone().requires_grad_(True)

    plain_output = plain(plain_input)
    checked_output = checked(checked_input)
    sum(value.sum() for value in plain_output.__dict__.values()).backward()
    sum(value.sum() for value in checked_output.__dict__.values()).backward()

    for name in plain_output.__dict__:
        assert torch.allclose(getattr(plain_output, name), getattr(checked_output, name))
    assert torch.allclose(plain_input.grad, checked_input.grad, atol=1.0e-5, rtol=1.0e-5)


def test_map_configuration_fails_instead_of_changing_norm_groups() -> None:
    with pytest.raises(ValueError, match="GroupNorm"):
        MapBackbone((6, 12, 18, 24), norm_groups=8)
    with pytest.raises(ValueError, match="被 8 整除"):
        MapBackbone((8, 16, 24, 32), input_shape_zyx=(46, 48, 48))
    with pytest.raises(ValueError, match="head"):
        MapBackbone((8, 16, 24, 30), bottleneck_heads=8)
