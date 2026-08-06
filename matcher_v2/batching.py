"""按 occurrence 预算装箱，并把双模式样本收集为共同 MatcherBatch。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

from .contracts import MatcherBatch, MatcherSample


class OccurrenceBudgetBatchSampler(Sampler[list[int]]):
    """不拆分 PDB 地按预期 occurrence 上限贪心装箱。"""

    def __init__(
        self,
        occurrence_counts: Sequence[int],
        *,
        budget: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        self.occurrence_counts = tuple(int(value) for value in occurrence_counts)
        self.budget = int(budget)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> list[list[int]]:
        indices = np.arange(len(self.occurrence_counts))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(indices)
        batches: list[list[int]] = []
        current: list[int] = []
        current_count = 0
        for index in indices.tolist():
            count = self.occurrence_counts[index]
            if current and current_count + count > self.budget:
                batches.append(current)
                current = []
                current_count = 0
            current.append(index)
            current_count += count
            if count > self.budget:
                batches.append(current)
                current = []
                current_count = 0
        if current:
            batches.append(current)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())


def collate_matcher_batch(samples: Sequence[MatcherSample]) -> MatcherBatch:
    """建立 PDB 边界并沿候选轴堆叠密度，不改变样本内实体顺序。"""

    kept = tuple(samples)
    candidate_ptr = [0]
    ligand_identity_ptr = [0]
    for sample in kept:
        candidate_ptr.append(candidate_ptr[-1] + sample.num_candidates)
        ligand_identity_ptr.append(ligand_identity_ptr[-1] + len(sample.ligands))
    return MatcherBatch(
        kept,
        torch.stack([pocket.density for sample in kept for pocket in sample.pockets]),
        tuple(ligand.graph for sample in kept for ligand in sample.ligands),
        tuple(pocket.A_graph for sample in kept for pocket in sample.pockets),
        tuple(candidate_ptr),
        tuple(ligand_identity_ptr),
    )

