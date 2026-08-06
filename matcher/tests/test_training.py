from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
import pytest
from omegaconf import OmegaConf

from matcher.model import Matcher
from matcher.train import WarmupPlateau, _load_phase1_for_phase2, _start_wandb


def test_warmup_plateau_counts_only_actual_lr_reductions() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = WarmupPlateau(
        optimizer,
        warmup_steps=2,
        warmup_start_factor=0.5,
        factor=0.3,
        patience=0,
        threshold=0.003,
        stop_after_lr_reductions=3,
    )

    assert optimizer.param_groups[0]["lr"] == 0.5
    scheduler.step_validation(0.1)
    assert scheduler.reduction_count == 0
    scheduler.step_optimizer()
    scheduler.step_optimizer()
    assert optimizer.param_groups[0]["lr"] == 1.0
    scheduler.step_validation(0.5)
    scheduler.step_validation(0.5)
    assert scheduler.reduction_count == 1
    assert not scheduler.should_stop


def test_warmup_plateau_restores_completed_stop_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = WarmupPlateau(
        optimizer,
        warmup_steps=2,
        warmup_start_factor=0.5,
        factor=0.3,
        patience=0,
        threshold=0.003,
        stop_after_lr_reductions=3,
    )
    state = scheduler.state_dict()
    state["reduction_count"] = 3

    resumed = WarmupPlateau(
        optimizer,
        warmup_steps=9,
        warmup_start_factor=0.1,
        factor=0.5,
        patience=2,
        threshold=0.1,
        stop_after_lr_reductions=8,
    )
    resumed.load_state_dict(state)

    assert resumed.warmup_steps == 2
    assert resumed.stop_after_lr_reductions == 3
    assert resumed.should_stop


def test_start_wandb_uses_explicit_online_identity(monkeypatch, tmp_path) -> None:
    captured = {}
    defined_metrics = []
    expected_run = SimpleNamespace(
        define_metric=lambda *args, **kwargs: defined_metrics.append((args, kwargs))
    )

    def fake_init(**kwargs):
        captured.update(kwargs)
        return expected_run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    config = OmegaConf.create(
        {
            "run": {"phase": 1},
            "wandb": {
                "enabled": True,
                "mode": "online",
                "project": "AdaLigand_Matcher",
                "group": "anchor_O_O_prime_v1_seed_3407_occ48",
                "name": "anchor_O_O_prime_v1_seed_3407_occ48_phase1",
                "log_every_n_steps": 10,
            },
        }
    )

    assert _start_wandb(config, tmp_path) is expected_run
    assert captured["mode"] == "online"
    assert captured["project"] == "AdaLigand_Matcher"
    assert captured["group"] == "anchor_O_O_prime_v1_seed_3407_occ48"
    assert captured["name"].endswith("phase1")
    assert captured["dir"] == tmp_path.as_posix()
    assert captured["resume"] == "never"
    assert defined_metrics == [(("global_step",), {}), (("*",), {"step_metric": "global_step"})]


def _tiny_matcher(phase: int) -> Matcher:
    return Matcher(
        phase=phase,
        A_node_input_dim=7,
        node_dim=16,
        edge_dim=8,
        repr_dim=32,
        num_heads=4,
        coarse_layers_per_block=1,
        map_channels=(2, 4, 8, 8),
        map_norm_groups=2,
    )


def test_phase2_load_allows_only_declared_new_modules(tmp_path) -> None:
    phase1 = _tiny_matcher(1)
    checkpoint = tmp_path / "phase1.pt"
    torch.save({"model": phase1.state_dict()}, checkpoint)
    phase2 = _tiny_matcher(2)

    _load_phase1_for_phase2(phase2, checkpoint)

    assert torch.equal(
        phase1.blocks[0].coarse_layers[0].cross_norm["CCD"].weight,
        phase2.blocks[0].coarse_layers[0].cross_norm["CCD"].weight,
    )
    assert not any(parameter.requires_grad for parameter in phase2.blocks[0].parameters())


def test_phase2_load_rejects_missing_common_parameter(tmp_path) -> None:
    state = _tiny_matcher(1).state_dict()
    state.pop("blocks.0.coarse_layers.0.cross_norm.CCD.weight")
    checkpoint = tmp_path / "broken.pt"
    torch.save({"model": state}, checkpoint)

    with pytest.raises(RuntimeError, match="Phase2 checkpoint mismatch"):
        _load_phase1_for_phase2(_tiny_matcher(2), checkpoint)
