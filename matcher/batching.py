"""按 occurrence 预算装箱已经通过零候选过滤的完整 PDB。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Sampler

from .anchor_data import AnchorSample, RawGraph


@dataclass(frozen=True)
class MatcherBatch:
    """一个 optimizer step 的当前 Anchor 路线输入及其 PDB 边界。

    ``density`` 沿候选轴打包；``ligand_graphs`` 与 ``A_graphs`` 分别保存待拼接的
    互不连边图。两个 ``*_ptr`` 都以 PDB 为段，供模型恢复候选和配体身份。
    """

    samples: tuple[AnchorSample, ...]
    density: Tensor
    ligand_graphs: tuple[RawGraph, ...]
    A_graphs: tuple[RawGraph, ...]
    candidate_ptr: tuple[int, ...]
    ligand_identity_ptr: tuple[int, ...]


class OccurrenceBudgetBatchSampler(Sampler[list[int]]):
    """按固定清单 occurrence 数贪心装箱，不拆分完整 PDB。"""

    def __init__(
        self,
        occurrence_counts: Sequence[int],
        *,
        budget: int = 64,
        seed: int = 3407,
        shuffle: bool = True,
        indices: Sequence[int] | None = None,
    ) -> None:
        self.occurrence_counts = tuple(int(value) for value in occurrence_counts)
        self.budget = int(budget)
        self.seed = int(seed)
        self.shuffle = shuffle
        self.indices = tuple(range(len(self.occurrence_counts))) if indices is None else tuple(indices)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_indices(self, indices: Sequence[int]) -> None:
        """设置已经完成零候选过滤的 Dataset 下标。"""

        self.indices = tuple(int(index) for index in indices)

    def _batches(self) -> list[list[int]]:
        indices = np.asarray(self.indices, dtype=np.int64)
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(indices)
        batches: list[list[int]] = []
        current: list[int] = []
        current_count = 0
        for index_value in indices.tolist():
            count = self.occurrence_counts[index_value]
            if current and current_count + count > self.budget:
                batches.append(current)
                current = []
                current_count = 0
            current.append(index_value)
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


def collate_anchor_batch(samples: Sequence[AnchorSample]) -> MatcherBatch:
    """收集非空候选样本，并建立直接可读的 PDB 打包边界。"""

    kept = tuple(samples)
    candidate_ptr = [0]
    ligand_identity_ptr = [0]
    for sample in kept:
        candidate_ptr.append(candidate_ptr[-1] + sample.num_candidates)
        ligand_identity_ptr.append(ligand_identity_ptr[-1] + len(sample.ligands))
    return MatcherBatch(
        samples=kept,
        density=torch.stack(
            [candidate.density for sample in kept for candidate in sample.candidates]
        ),
        ligand_graphs=tuple(
            ligand.raw_graph for sample in kept for ligand in sample.ligands
        ),
        A_graphs=tuple(
            candidate.A_graph for sample in kept for candidate in sample.candidates
        ),
        candidate_ptr=tuple(candidate_ptr),
        ligand_identity_ptr=tuple(ligand_identity_ptr),
    )
