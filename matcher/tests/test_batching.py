from __future__ import annotations

from matcher.batching import OccurrenceBudgetBatchSampler, collate_anchor_batch
from matcher.tests.sample_factory import sample


def test_occurrence_budget_keeps_whole_pdb_and_allows_oversized_batch() -> None:
    sampler = OccurrenceBudgetBatchSampler([20, 44, 65, 10], budget=64, shuffle=False)

    assert list(sampler) == [[0, 1], [2], [3]]


def test_epoch_shuffle_is_reproducible() -> None:
    left = OccurrenceBudgetBatchSampler([10] * 20, budget=30, seed=7)
    right = OccurrenceBudgetBatchSampler([10] * 20, budget=30, seed=7)
    left.set_epoch(3)
    right.set_epoch(3)

    assert list(left) == list(right)
    assert sorted(index for batch in left for index in batch) == list(range(20))


def test_sampler_packs_only_indices_kept_after_candidate_sampling() -> None:
    sampler = OccurrenceBudgetBatchSampler(
        [20, 30, 44, 10], budget=64, shuffle=False, indices=[0, 2, 3]
    )

    assert list(sampler) == [[0, 2], [3]]


def test_collate_builds_only_required_pdb_boundaries() -> None:
    item = sample()
    batch = collate_anchor_batch([item])

    assert batch.samples == (item,)
    assert batch.candidate_ptr == (0, 2)
    assert batch.ligand_identity_ptr == (0, 1)
    assert batch.density.shape == (2, 1, 48, 48, 48)
