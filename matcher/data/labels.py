"""
用完整 Stage1 候选组件与真实配体 occurrence 的双向体素覆盖率生成六分类和身份命中监督。

主要入口:
    - assign_candidate_target: 在一个 PDB 的全部 occurrence 中选择同时满足两个覆盖率阈值的唯一最佳命中。

返回对象:
    - CandidateTarget: 保存六分类标签、命中的 occurrence candidate_id、object_key 和三个覆盖率审计值。

本模块只计算内存对象，不读取或写入文件。候选与 occurrence 都使用全局稀疏 ZYX 体素索引；体素物理尺寸和世界坐标不参与本模块的集合覆盖率计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np

from .schema import CLASS_TO_INDEX, OCCURRENCE_TYPE_TO_CLASS


@dataclass(frozen=True)
class OccurrenceMask:
    """
    保存一个真实配体 occurrence 的身份字段和全局稀疏体素集合。

    字段:
        - candidate_id: int；parse/{pdb_id}/occurrences.jsonl 中该 occurrence 的 candidate_id，也用于读取 density/{pdb_id}/ligand_area.npz 的 mask_{candidate_id} 字段。
        - object_key: str；定位 ligand_objects/{safe_object_filename(object_key)}.npz 的配体模板图，不单独代表 v4 联合身份。
        - type_tag: str；occurrence 类型，输入契约要求属于 OCCURRENCE_TYPE_TO_CLASS；assign_candidate_target 在该 occurrence 成为最佳命中时执行映射，未知值会失败。
        - voxel_index_zyx: 整数数组，(N_occ_voxel, 3)；该 occurrence 占据的非负全局稀疏体素索引，最后一维依次为 Z、Y、X，允许输入包含重复坐标。
    """

    candidate_id: int
    object_key: str
    type_tag: str
    voxel_index_zyx: np.ndarray


@dataclass(frozen=True)
class CandidateTarget:
    """
    保存一个 Stage1 候选的最终 occurrence 命中、六分类标签和覆盖率审计值。

    字段:
        - class_name: str；CLASS_NAMES 中的六分类名称；没有 occurrence 同时通过两个覆盖率阈值，或最佳命中的 occurrence.type_tag 为 other 时为 background。
        - class_index: int；class_name 在 CLASS_NAMES 中的下标，直接作为候选六分类监督。
        - matched_candidate_id: int 或 None；最佳命中 occurrence 的 candidate_id；没有 occurrence 通过阈值时为 None，命中 other 时即使 class_name 为 background 也保留编号。
        - matched_object_key: str 或 None；最佳命中 occurrence 的 LigandObject 模板键；没有 occurrence 通过阈值时为 None，v4 联合身份还需在 manifest 中关联 prepared_smiles。
        - candidate_coverage: float；交集唯一体素数除以候选唯一体素数，范围 [0, 1]；没有 occurrence 通过阈值时为 0。
        - occurrence_coverage: float；交集唯一体素数除以 occurrence 唯一体素数，范围 [0, 1]；没有 occurrence 通过阈值时为 0。
        - overlap_score: float；candidate_coverage 与 occurrence_coverage 乘积的平方根，用于通过阈值后的 occurrence 排序；没有 occurrence 通过阈值时为 0。
    """

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
    """
    按双向覆盖阈值和几何平均覆盖率选择唯一真实配体 occurrence。

    输入参数:
        - candidate_voxel_zyx: 整数数组，(N_candidate_voxel, 3)；一个完整 Stage1 候选组件的非负全局稀疏体素索引，最后一维依次为 Z、Y、X，允许包含重复坐标但不能为空；_unique_voxel_rows 转为 int64 后核验非负。
        - occurrences: Iterable[OccurrenceMask]；同一 PDB 中可参与监督匹配的真实配体 occurrence；空体素 occurrence 被忽略。
        - occurrences[*].candidate_id: int；当前 occurrence 的 PDB 内编号，同时用于分数并列时选择较小编号。
        - occurrences[*].object_key: str；当前 occurrence 的 LigandObject 模板键，命中后写入 CandidateTarget.matched_object_key。
        - occurrences[*].type_tag: str；当前 occurrence 类型，输入契约要求属于 OCCURRENCE_TYPE_TO_CLASS；最佳命中使用该映射，未知值会失败。
        - occurrences[*].voxel_index_zyx: 整数数组，(N_occ_voxel, 3)；完整密度图坐标系中的非负稀疏 ZYX 体素，可含重复坐标或为空；_unique_voxel_rows 转为 int64 后核验非负。
        - coverage_threshold: float；范围 [0, 1]，candidate_coverage 和 occurrence_coverage 都不得低于该阈值。

    选择规则:
        - primary_score: float；两个覆盖率乘积的平方根，数值更大者优先。
        - tie_breaker: int；primary_score 相同时选择 candidate_id 更小的 occurrence。

    返回值:
        - target: CandidateTarget；当前候选的六分类、最佳 occurrence 和覆盖率结果。
        - target.class_name: str；六分类名称；未命中或命中 type_tag=other 时为 background。
        - target.class_index: int；class_name 在 CLASS_NAMES 中的索引。
        - target.matched_candidate_id: int 或 None；最佳 occurrence 编号；未命中时为 None。
        - target.matched_object_key: str 或 None；最佳 occurrence 模板键；未命中时为 None。
        - target.candidate_coverage: float；交集唯一体素数除以候选唯一体素数，范围 [0, 1]；未命中时为 0。
        - target.occurrence_coverage: float；交集唯一体素数除以最佳 occurrence 唯一体素数，范围 [0, 1]；未命中时为 0。
        - target.overlap_score: float；两个覆盖率乘积的平方根，范围 [0, 1]；未命中时为 0。
    """

    if not 0.0 <= coverage_threshold <= 1.0:
        raise ValueError(f"coverage_threshold 必须在 [0,1] 内，实际为 {coverage_threshold}。")
    # np.void, (N_candidate_unique,), 每个元素封装一个唯一 ZYX 三元组，仅用于集合求交；不再作为可分轴坐标读取。
    candidate_rows = _unique_voxel_rows(candidate_voxel_zyx)
    if candidate_rows.size == 0:
        raise ValueError("候选完整全局组件不能为空。")

    # tuple[float, int, OccurrenceMask, float, float] | None，保存当前最佳匹配；第二项使用负 candidate_id，使元组排序在分数相同时选择较小编号。
    best: tuple[float, int, OccurrenceMask, float, float] | None = None
    for occurrence in occurrences:
        # np.void, (N_occurrence_unique,), 当前 occurrence 的唯一 ZYX 三元组集合表示，与 candidate_rows 使用相同 record dtype。
        occurrence_rows = _unique_voxel_rows(occurrence.voxel_index_zyx)
        if occurrence_rows.size == 0:
            continue
        # int, 候选与当前 occurrence 共有的唯一全局体素数。
        intersection = int(
            np.intersect1d(candidate_rows, occurrence_rows, assume_unique=True).size
        )
        # float，当前交集占候选唯一体素集合的比例，范围 [0, 1]。
        candidate_coverage = intersection / candidate_rows.size
        # float，当前交集占 occurrence 唯一体素集合的比例，范围 [0, 1]。
        occurrence_coverage = intersection / occurrence_rows.size
        if (
            candidate_coverage < coverage_threshold
            or occurrence_coverage < coverage_threshold
        ):
            continue
        # float，两个方向覆盖率的几何平均，作为通过双阈值后的主排序分数。
        score = sqrt(candidate_coverage * occurrence_coverage)
        # tuple[float, int], 先最大化双向覆盖几何平均，再最小化 occurrence candidate_id。
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

    # float，最终最佳匹配的双向覆盖率几何平均。
    # 第二项是仅用于并列决胜的负 candidate_id，返回对象不单独保存该临时值。
    # OccurrenceMask，最终最佳匹配的真实配体 occurrence。
    # float，最终最佳匹配的候选方向覆盖率。
    # float，最终最佳匹配的 occurrence 方向覆盖率。
    score, _, occurrence, candidate_coverage, occurrence_coverage = best
    # str | None，把最佳 occurrence.type_tag 映射到六分类名称；None 表示来源枚举漂移。
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
    """
    把非负 ZYX 稀疏索引转换成可由 NumPy 一维集合运算比较的唯一记录。

    输入参数:
        - voxel_index_zyx: 数组，(N_voxel, 3)；最后一维依次为 Z、Y、X，坐标必须为非负整数，N_voxel 可以为 0。

    返回值:
        - unique_records: 数组，(N_unique_voxel,)；非空输入返回 np.void 记录数组，每个元素按原始内存字节封装一个 ZYX 三元组并已去重排序；空输入返回当前实现的 int64 空数组，两种情况都只按 size 或一维集合运算消费。
    """

    # int64, (N_voxel, 3), 统一稀疏体素索引的数据类型，同时保留 ZYX 轴顺序。
    indices = np.asarray(voxel_index_zyx, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"稀疏体素索引必须为 (N,3)，实际为 {indices.shape}。")
    if np.any(indices < 0):
        raise ValueError("稀疏体素索引不能含负数。")
    if indices.size == 0:
        return np.empty(0, dtype=np.int64)
    # int64, (N_voxel, 3), 连续内存保证每三个相邻整数可安全视为一个不可拆分的 record。
    contiguous = np.ascontiguousarray(indices)
    # np.dtype[np.void], 每个元素占 3 个 int64 的字节数，用于把一个 ZYX 三元组视为一条集合记录。
    record_dtype = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    return np.unique(contiguous.view(record_dtype).reshape(-1))
