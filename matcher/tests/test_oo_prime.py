from __future__ import annotations

import torch

from matcher.consistency import O_O_prime_consistency
from matcher.oo_prime import OOPrimeDecoder, OOPrimeEvaluation


def test_decoder_allows_candidate_reuse_and_unmatched_slots() -> None:
    decoded = OOPrimeDecoder(0.5).decode(
        torch.tensor([[4.0, 4.0, -4.0], [1.0, 1.0, -3.0]]),
        torch.tensor([[0.8, 0.9, 0.9], [0.9, 0.9, 0.9]]),
    )

    assert [pair.candidate_index for pair in decoded.selected_pairs] == [0, 0]
    assert decoded.unmatched_slot_indices == (2,)


def test_evaluation_tie_chooses_higher_threshold() -> None:
    evaluation = OOPrimeEvaluation(threshold_step=0.5)
    evaluation.update(
        torch.tensor([[8.0]]),
        torch.tensor([[0.8]]),
        torch.tensor([[True]]),
        torch.tensor([0]),
    )

    metrics = evaluation.best()
    assert metrics.F1 == 1.0
    assert metrics.threshold == 0.5


def test_frozen_threshold_evaluation_does_not_rescan() -> None:
    evaluation = OOPrimeEvaluation(None, frozen_threshold=0.75)
    evaluation.update(
        torch.tensor([[0.0]]),
        torch.tensor([[1.0]]),
        torch.tensor([[True]]),
        torch.tensor([0]),
    )

    metrics = evaluation.best()
    assert metrics.threshold == 0.75
    assert metrics.predicted_nonempty == 0


def test_consistency_is_no_grad_diagnostic() -> None:
    O_logit = torch.tensor([[3.0, -3.0]], requires_grad=True)
    result = O_O_prime_consistency(O_logit, torch.tensor([[0.8, 0.1]]))

    assert result.agreement == 1.0
    assert O_logit.grad is None
