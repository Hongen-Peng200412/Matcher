"""用完整候选组件与真实配体实例的双向覆盖率生成六分类和身份监督。"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np

from .schema import CLASS_TO_INDEX, OCCURRENCE_TYPE_TO_CLASS


@dataclass(frozen=True)
class OccurrenceMask:
    """保存一个真实配体出现实例的身份和全局稀疏体素。"""

    candidate_id: int
    object_key: str
    type_tag: str
    voxel_index_zyx: np.ndarray


@dataclass(frozen=True)
class CandidateTarget:
    """保存一个 Stage1 候选的最终命中、类别和覆盖率审计字段。"""

    class_name: str
    class_index: int
    matched_candidate_id: int | None
    matched_object_key: str | None
    candidate_coverage: float
    occurrence_coverage: float
    overlap_score: float


def assign_candidate_target(
    candidate_voxel_zyx: np.ndarray,
    occurrences: Iterable[OccurrenceMask],
    coverage_threshold: float,
) -> CandidateTarget:
    """按双向覆盖阈值和几何平均覆盖率选择唯一真实配体实例。"""

    if not 0.0 <= coverage_threshold <= 1.0:
        raise ValueError(f"coverage_threshold 必须在 [0,1] 内，实际为 {coverage_threshold}。")
    candidate_rows = _unique_voxel_rows(candidate_voxel_zyx)
    if candidate_rows.size == 0:
        raise ValueError("候选完整全局组件不能为空。")

    best: tuple[float, int, OccurrenceMask, float, float] | None = None
    for occurrence in occurrences:
        occurrence_rows = _unique_voxel_rows(occurrence.voxel_index_zyx)
        if occurrence_rows.size == 0:
            continue
        intersection = int(
            np.intersect1d(candidate_rows, occurrence_rows, assume_unique=True).size
        )
        candidate_coverage = intersection / candidate_rows.size
        occurrence_coverage = intersection / occurrence_rows.size
        if (
            candidate_coverage < coverage_threshold
            or occurrence_coverage < coverage_threshold
        ):
            continue
        score = sqrt(candidate_coverage * occurrence_coverage)
        ranking = (score, -occurrence.candidate_id)
        if best is None or ranking > (best[0], best[1]):
            best = (
                score,
                -occurrence.candidate_id,
                occurrence,
                candidate_coverage,
                occurrence_coverage,
            )

    if best is None:
        return CandidateTarget("background", CLASS_TO_INDEX["background"], None, None, 0.0, 0.0, 0.0)

    score, _, occurrence, candidate_coverage, occurrence_coverage = best
    class_name = OCCURRENCE_TYPE_TO_CLASS.get(occurrence.type_tag)
    if class_name is None:
        raise ValueError(f"未知 occurrence type_tag={occurrence.type_tag!r}。")
    return CandidateTarget(
        class_name=class_name,
        class_index=CLASS_TO_INDEX[class_name],
        matched_candidate_id=occurrence.candidate_id,
        matched_object_key=occurrence.object_key,
        candidate_coverage=float(candidate_coverage),
        occurrence_coverage=float(occurrence_coverage),
        overlap_score=float(score),
    )


def _unique_voxel_rows(voxel_index_zyx: np.ndarray) -> np.ndarray:
    """把非负 ZYX 稀疏索引转换成与空间边界无关的唯一记录。"""

    indices = np.asarray(voxel_index_zyx, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"稀疏体素索引必须为 (N,3)，实际为 {indices.shape}。")
    if np.any(indices < 0):
        raise ValueError("稀疏体素索引不能含负数。")
    if indices.size == 0:
        return np.empty(0, dtype=np.int64)
    contiguous = np.ascontiguousarray(indices)
    record_dtype = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    return np.unique(contiguous.view(record_dtype).reshape(-1))
