"""验证双模式标签和 Stage1 严格零增量。"""

from __future__ import annotations

import numpy as np
import torch

from matcher_v2.contracts import MatcherSample, PocketInput, Stage1Context
from matcher_v2.data_common import coverage_targets
from matcher_v2.model import Stage1ContextDelta


def test_coverage_targets_keep_fractional_overlap_and_distance_definition() -> None:
    candidate = (np.asarray([[0, 0, 0], [0, 0, 1]], dtype=np.int32),)
    occurrences = (
        np.asarray([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]], dtype=np.int32),
    )
    A, B, O, A_prime, B_prime, O_prime = coverage_targets(
        candidate, occurrences, np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([[10.0, 0.0, 0.0]]),
    )
    assert A.item() == 0.5
    assert B.item() == 1.0
    assert torch.isclose(O, torch.tensor([[1.0 / 11.0]])).all()
    assert A_prime.item() and B_prime.item()
    assert not O_prime.item()


def _sample(stage1: Stage1Context | None) -> MatcherSample:
    pocket = PocketInput(
        center_xyz=torch.zeros(3), density=torch.zeros(1, 48, 48, 48),
        map_start_zyx=torch.zeros(3, dtype=torch.int32), voxel_size_xyz=torch.ones(3),
        origin_xyz=torch.zeros(3), A_graph=None, A_global_index=torch.empty(0, dtype=torch.long),
        candidate_mask_zyx=torch.empty(0, 3, dtype=torch.int32),
        rotation_axes_zyx=None, rotation_k=0, stage1=stage1,
    )
    target = torch.zeros(1, 1)
    return MatcherSample(
        "test", 0, (), (pocket,), torch.tensor([0]), torch.tensor([0]), torch.zeros(1, 3),
        (torch.empty(0, 3),), (torch.empty(0, dtype=torch.bool),),
        target, target, target, target.bool(), target.bool(), target.bool(),
    )


def test_stage1_condition_is_exact_zero_when_absent() -> None:
    module = Stage1ContextDelta(V_feature_dim=2, P_feature_dim=3, repr_dim=8)
    with torch.no_grad():
        module.network[-1].weight.fill_(0.1)
    reference = torch.randn(1, 8)
    PP_summary = torch.zeros(1, 8)
    absent = module(_sample(None), reference, PP_summary)
    assert torch.equal(absent, torch.zeros_like(absent))

    context = Stage1Context(
        torch.ones(2, 3), torch.ones(2), torch.ones(2, 2),
        torch.ones(2, 3), torch.ones(2), torch.ones(2, 3),
        torch.ones(2, 3), torch.ones(2), torch.ones(2), torch.ones(2, 4),
    )
    assert not torch.equal(
        module(_sample(context), reference, PP_summary), torch.zeros_like(absent)
    )
