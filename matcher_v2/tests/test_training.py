"""验证长训练入口中不应漂移的轻量调度与恢复契约。"""

import torch

from matcher_v2.tests.test_model import _model
from matcher_v2.train import _validation_batches
from matcher_v2.train import _load_training_weights


def test_validation_batches_are_exactly_five_when_epoch_is_long_enough() -> None:
    assert _validation_batches(12, 5) == {3, 5, 8, 10, 12}


def test_validation_batches_do_not_repeat_when_epoch_is_short() -> None:
    assert _validation_batches(3, 5) == {1, 2, 3}


def test_phase2_resume_restores_freeze_before_optimizer_is_built(tmp_path) -> None:
    trained = _model(2)
    trained.freeze_phase1_for_phase2()
    checkpoint = tmp_path / "phase2.pt"
    torch.save({"model": trained.state_dict()}, checkpoint)
    resumed = _model(2)

    _load_training_weights(
        resumed, phase=2, resume_checkpoint=checkpoint, phase1_checkpoint=None
    )

    assert sum(parameter.requires_grad for parameter in resumed.parameters()) == sum(
        parameter.requires_grad for parameter in trained.parameters()
    )
