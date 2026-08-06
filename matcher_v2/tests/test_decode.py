"""验证六头打分、候选互斥解码和阈值指标。"""

from __future__ import annotations

import torch

from matcher_v2.decode_ground_truth import decode_pairs, evaluate
from matcher_v2.model import MatcherHeadOutput


def _head(logit: torch.Tensor) -> MatcherHeadOutput:
    return MatcherHeadOutput(torch.empty(*logit.shape, 1), *(logit for _ in range(6)))


def test_decode_uses_one_to_one_pairs_and_threshold() -> None:
    score = torch.tensor([[0.9, 0.8], [0.7, 0.1]])
    pairs = decode_pairs(score, 0.5)
    assert {(pair.candidate_index, pair.occurrence_index) for pair in pairs} == {(0, 1), (1, 0)}
    assert decode_pairs(score, 0.85)[0].candidate_index == 0
