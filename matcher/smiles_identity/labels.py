"""用 Stage1 候选与真实 occurrence 的双向体素覆盖率生成监督标签.

本模块明确区分 source_blob_index 和 occurrence_id. 前者定位 Find_1 候选, 后者定位
真实配体 occurrence. 身份字符串不参与几何匹配, 也不保存在标签旁路文件中.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np

from .schema import CLASS_TO_INDEX


@dataclass(frozen=True)
class OccurrenceMask:
    """保存一个真实配体 occurrence 的类型和全局稀疏体素集合.

    字段:
        - occurrence_id: int, PDB 内 occurrence 编号; 对应 occurrences.jsonl 的 candidate_id 源字段.
        - type_tag: str, occurrence 类型; other 映射为 background, 其余值必须属于正式前景类别.
        - voxel_index_zyx: int64 数组, (N_occurrence_voxel, 3); 当前 occurrence 的全局密度体素下标, 最后一轴依次为 Z/Y/X.
    """

    occurrence_id: int
    type_tag: str
    voxel_index_zyx: np.ndarray


@dataclass(frozen=True)
class CandidateTarget:
    """保存一个 Stage1 候选的六分类目标、最佳 occurrence 和覆盖率.

    字段:
        - class_index: int, 六分类下标; 未命中或命中 other 时为 background 下标 0.
        - matched_occurrence_id: int|None, 最佳 occurrence 编号; 未命中时为 None.
        - candidate_coverage: float, 交集唯一体素数除以候选唯一体素数, 无量纲且属于 [0, 1].
        - occurrence_coverage: float, 交集唯一体素数除以 occurrence 唯一体素数, 无量纲且属于 [0, 1].
        - overlap_score: float, 两个覆盖率乘积的平方根, 无量纲且属于 [0, 1].
    """

    class_index: int
    matched_occurrence_id: int | None
    candidate_coverage: float
    occurrence_coverage: float
    overlap_score: float


def _unique_voxel_rows(voxel_index_zyx: np.ndarray) -> np.ndarray:
    """把非负 Z/Y/X 索引去重并转换为一维集合记录.

    - voxel_index_zyx: 整数数组, (N, 3); N 可为零, 三列依次为非负 Z/Y/X 体素下标.
    - return: np.ndarray (N_unique,); 非空时是每项编码三个 int64 的 np.void 唯一记录, 空输入时是空 int64 数组, 两者只作为 np.intersect1d 的不透明集合键.
    """

    indices = np.asarray(voxel_index_zyx, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"稀疏体素索引必须为 (N,3)，实际为 {indices.shape}。")
    if np.any(indices < 0):
        raise ValueError("稀疏体素索引不能含负数。")
    if indices.size == 0:
        return np.empty(0, dtype=np.int64)
    contiguous = np.ascontiguousarray(indices)
    record_dtype = np.dtype((np.void, contiguous.dtype.itemsize * 3))
    return np.unique(contiguous.view(record_dtype).reshape(-1))


# ================================================================================================


def assign_candidate_target(
    candidate_voxel_zyx: np.ndarray,
    occurrences: Iterable[OccurrenceMask],
    coverage_threshold: float,
) -> CandidateTarget:
    """选择同时通过两个覆盖率阈值且几何平均覆盖率最高的 occurrence.

    - candidate_voxel_zyx: 整数数组, (N_candidate_voxel, 3); 当前 Stage1 候选的全局 Z/Y/X 体素下标.
    - occurrences: 同一 PDB 的真实 occurrence; 空掩码会被忽略.
    - coverage_threshold: 闭区间 [0, 1] 内的共同阈值.
    - return: CandidateTarget; 分数并列时选择 occurrence_id 较小者.
    """

    if not 0.0 <= coverage_threshold <= 1.0:
        raise ValueError(f"coverage_threshold 必须位于 [0,1]，实际为 {coverage_threshold}。")
    candidate_rows = _unique_voxel_rows(candidate_voxel_zyx)
    if candidate_rows.size == 0:
        raise ValueError("Stage1 候选的完整全局体素集合不能为空。")
    # 五元组依次保存几何平均覆盖率、负 occurrence_id、OccurrenceMask、候选覆盖率和 occurrence 覆盖率; 前两项共同决定最大值排序.
    best: tuple[float, int, OccurrenceMask, float, float] | None = None
    for occurrence in occurrences:
        # 一维唯一记录数组, (N_unique_occurrence_voxel,); 当前 occurrence 的去重全局体素集合.
        occurrence_rows = _unique_voxel_rows(occurrence.voxel_index_zyx)
        if occurrence_rows.size == 0:
            continue
        # int, 当前候选与 occurrence 两个去重体素集合的交集大小.
        intersection = int(np.intersect1d(candidate_rows, occurrence_rows, assume_unique=True).size)
        # float, 当前交集占候选去重体素集合的比例.
        candidate_coverage = intersection / candidate_rows.size
        # float, 当前交集占 occurrence 去重体素集合的比例.
        occurrence_coverage = intersection / occurrence_rows.size
        if min(candidate_coverage, occurrence_coverage) < coverage_threshold:
            continue
        # float, 两个方向覆盖率的几何平均; 只对共同通过阈值的 occurrence 排序.
        score = sqrt(candidate_coverage * occurrence_coverage)
        ranking = (score, -occurrence.occurrence_id)
        if best is None or ranking > best[:2]:
            best = (score, -occurrence.occurrence_id, occurrence, candidate_coverage, occurrence_coverage)
    if best is None:
        return CandidateTarget(CLASS_TO_INDEX["background"], None, 0.0, 0.0, 0.0)
    score, _, occurrence, candidate_coverage, occurrence_coverage = best
    class_name = "background" if occurrence.type_tag == "other" else occurrence.type_tag
    if class_name not in CLASS_TO_INDEX:
        raise ValueError(f"未知 occurrence type_tag={occurrence.type_tag!r}。")
    return CandidateTarget(
        class_index=CLASS_TO_INDEX[class_name],
        matched_occurrence_id=occurrence.occurrence_id,
        candidate_coverage=float(candidate_coverage),
        occurrence_coverage=float(occurrence_coverage),
        overlap_score=float(score),
    )
