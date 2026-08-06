"""验证六头打分、候选互斥解码和阈值指标。"""

from __future__ import annotations

import torch

from matcher.decode_ground_truth import EvaluationPDB, decode_pairs, evaluate
from matcher.model import MatcherHeadOutput


def _head(logit: torch.Tensor) -> MatcherHeadOutput:
    return MatcherHeadOutput(torch.empty(*logit.shape, 1), *(logit for _ in range(6)))


def test_decode_uses_one_to_one_pairs_and_threshold() -> None:
    score = torch.tensor([[0.9, 0.8], [0.7, 0.1]])
    pairs = decode_pairs(score, 0.5)
    assert {(pair.candidate_index, pair.slot_index) for pair in pairs} == {(0, 1), (1, 0)}
    assert decode_pairs(score, 0.85)[0].candidate_index == 0


def test_evaluation_allows_same_identity_slots_to_exchange() -> None:
    record = EvaluationPDB(
        score=torch.tensor([[0.1, 0.9], [0.9, 0.1]]),
        correct_pair=torch.eye(2, dtype=torch.bool),
        slot_to_ligand=torch.zeros(2, dtype=torch.long),
    )

    metrics = evaluate((record,), threshold=0.5, step=0.5)

    assert metrics.correct == 2
    assert metrics.F1 == 1.0


def test_evaluation_does_not_exchange_different_identities() -> None:
    record = EvaluationPDB(
        score=torch.tensor([[0.1, 0.9], [0.9, 0.1]]),
        correct_pair=torch.eye(2, dtype=torch.bool),
        slot_to_ligand=torch.tensor([0, 1]),
    )

    metrics = evaluate((record,), threshold=0.5, step=0.5)

    assert metrics.correct == 0
