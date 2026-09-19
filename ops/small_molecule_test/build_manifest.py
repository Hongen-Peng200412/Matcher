"""构建独立的 test_smallmolecule 评测 manifest.

主要入口是 build_manifest.它连接 PocketXMol Stage3 测试 occurrence、两份 Matcher
test_0 manifest、Find_1 完整 blob 和 Emap2lig 正式产物, 但不修改任何来源文件.

输出目录由 YAML 的 outputs.manifest_root 决定.目录中的 manifest.json 是唯一入口;
候选、身份、Stage3 occurrence、来源摘要和字段 README 都在全部机械门禁通过后一次发布.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import load_config, read_json, read_jsonl, sha256_file, write_json, write_jsonl


COVERAGE_THRESHOLD = 0.30


def _assert_unique_candidate_keys(
    name: str,
    candidates: list[dict[str, Any]],
) -> None:
    """要求一种来源的 PDB 与完整 blob 编号逐候选唯一.

    输入参数:
        - name: str, 错误信息使用的来源名称.
        - candidates: list[dict[str, Any]], 每项必须含 pdb_id 和 source_blob_index.

    返回值:
        - None: 键集合大小等于候选数时返回; 重复键失败关闭.
    """
    keys = [
        (str(candidate["pdb_id"]), int(candidate["source_blob_index"]))
        for candidate in candidates
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name}: PDB 与 source_blob_index 不是逐候选唯一键.")


def _read_occurrence_facts(
    data_root: Path,
    prepared_root: Path,
    pdb_id: str,
) -> tuple[dict[int, str], dict[int, str | None], list[Path]]:
    """读取一个 PDB 的 occurrence 类别和精确顶层 SMILES.

    输入参数:
        - data_root: Path, AdaLigand Ori_Data 根目录.
        - prepared_root: Path, 按 PDB 保存 prepared_smiles.jsonl 的共同父目录.
        - pdb_id: str, 小写 PDB 标识, 如 11jb.

    返回值:
        - type_by_id: dict[int, str], occurrence 编号到 type_tag 的映射.
        - smiles_by_id: dict[int, str|None], occurrence 编号到精确 PreparedLigand.smiles 的映射.
        - source_paths: list[Path], 本次读取并需写入来源快照的两个 JSONL 路径.
    """
    occurrence_path = data_root / "parse" / pdb_id / "occurrences.jsonl"
    prepared_path = prepared_root / pdb_id / "prepared_smiles.jsonl"
    occurrence_records = read_jsonl(occurrence_path)
    prepared_records = read_jsonl(prepared_path)
    type_by_id = {
        int(record["candidate_id"]): str(record["type_tag"])
        for record in occurrence_records
    }
    prepared_by_id = {
        int(record["candidate_id"]): record.get("smiles")
        for record in prepared_records
    }
    unknown = sorted(set(prepared_by_id) - set(type_by_id))
    if unknown:
        raise ValueError(f"{pdb_id}: PreparedLigand 含未知 occurrence 编号 {unknown}.")
    smiles_by_id = {
        occurrence_id: (
            str(prepared_by_id[occurrence_id])
            if isinstance(prepared_by_id.get(occurrence_id), str)
            and prepared_by_id[occurrence_id]
            else None
        )
        for occurrence_id in type_by_id
    }
    return type_by_id, smiles_by_id, [occurrence_path, prepared_path]


def _target_from_intersections(
    occurrence_ids: np.ndarray,
    intersections: np.ndarray,
    candidate_size: int,
    occurrence_sizes: np.ndarray,
) -> tuple[int | None, float, float, float]:
    """按双向 0.30 覆盖和最大几何平均规则选择 Emap2lig 唯一 occurrence.

    输入参数:
        - occurrence_ids: int 数组, (N_occurrence,), 与交集向量和真实 occurrence 体素数逐项对齐.
        - intersections: int 数组, (N_occurrence,), 当前 Emap2lig blob 与每个 occurrence 的交集体素数.
        - candidate_size: int, 当前完整 Emap2lig blob 的体素数.
        - occurrence_sizes: int 数组, (N_occurrence,), 每个 occurrence 的真实体素数.

    返回值:
        - matched_occurrence_id: int|None, 最大几何平均对应的 occurrence 编号; 无有效覆盖时为 None.
        - candidate_coverage: float, 交集体素数除以当前 blob 体素数; 背景为 0.
        - occurrence_coverage: float, 交集体素数除以目标 occurrence 体素数; 背景为 0.
        - overlap_score: float, 两项覆盖率几何平均; 背景为 0.
    """
    candidate_coverage = intersections.astype(np.float64) / float(candidate_size)
    occurrence_coverage = intersections.astype(np.float64) / occurrence_sizes.astype(np.float64)
    eligible = (candidate_coverage >= COVERAGE_THRESHOLD) & (
        occurrence_coverage >= COVERAGE_THRESHOLD
    )
    if not bool(np.any(eligible)):
        return None, 0.0, 0.0, 0.0
    overlap = np.sqrt(candidate_coverage * occurrence_coverage)
    eligible_positions = np.flatnonzero(eligible)
    # float64 数组, (K,), 先按几何平均降序, 再按 occurrence 编号升序选择唯一目标.
    order = np.lexsort((occurrence_ids[eligible_positions], -overlap[eligible_positions]))
    position = int(eligible_positions[int(order[0])])
    return (
        int(occurrence_ids[position]),
        float(candidate_coverage[position]),
        float(occurrence_coverage[position]),
        float(overlap[position]),
    )


def _find_centers(
    stage1_root: Path,
    pdb_id: str,
) -> tuple[dict[int, list[float]], dict[int, bool], list[Path]]:
    """从 Find_1 完整 blob 和 centered 世界几何计算每个来源 blob 的世界中心.

    输入参数:
        - stage1_root: Path, 一种受体条件的 held_out_test_0 根目录.
        - pdb_id: str, 当前小写 PDB 标识.

    返回值:
        - center_by_blob: dict[int, list[float]], source_blob_index 到世界 XYZ 中心的映射, 单位 Å.
        - fits_by_blob: dict[int, bool], 完整来源 blob 是否全部位于对应 centered 80³ 框.
        - source_paths: list[Path], 本次读取的 F2_blobs.npz 与 F2_centered.npz.
    """
    pdb_root = stage1_root / pdb_id
    blob_path = pdb_root / "blobs/F2_blobs.npz"
    centered_path = pdb_root / "centered/F2_centered.npz"
    with np.load(blob_path, allow_pickle=False) as archive:
        # int64, (N_blob,); 完整来源 blob 在当前 PDB 内的稳定编号.
        blob_ids = np.asarray(archive["blob_index"], dtype=np.int64)
        # int64, (N_blob+1,); 相邻两项切分 voxels 中一个完整 blob 的体素区间.
        offsets = np.asarray(archive["voxel_offsets"], dtype=np.int64)
        # int64, (N_voxel, 3); 连接保存的全局 ZYX 体素编号.
        voxels = np.asarray(archive["voxel_index_global_zyx"], dtype=np.int64)
        # bool, (N_blob,); 完整 blob 是否全部位于对应 centered 80³ 框.
        fits = np.asarray(archive["fits_centered_box"], dtype=np.bool_)
    with np.load(centered_path, allow_pickle=False) as archive:
        # int64, (N_centered,); centered 候选到完整来源 blob 的编号映射.
        centered_blob_ids = np.asarray(archive["source_blob_index"], dtype=np.int64)
        # float64, (N_centered, 3); centered 框起点的全局 ZYX 体素编号.
        box_start_zyx = np.asarray(archive["box_start_zyx"], dtype=np.float64)
        # float64, (N_centered, 3); centered 框首体素的世界 XYZ 原点, 单位 Å.
        box_origin_xyz = np.asarray(archive["box_origin_world"], dtype=np.float64)
        # float64, (N_centered, 3); 世界 XYZ 三轴体素尺寸, 单位 Å.
        voxel_size_xyz = np.asarray(archive["voxel_size_world"], dtype=np.float64)
    if offsets.shape != (len(blob_ids) + 1,) or fits.shape != blob_ids.shape:
        raise ValueError(f"{blob_path}: blob、offsets 和 fits_centered_box 未对齐.")
    if not (
        centered_blob_ids.shape == box_start_zyx.shape[:1] == box_origin_xyz.shape[:1]
        and box_origin_xyz.shape == voxel_size_xyz.shape == (len(centered_blob_ids), 3)
    ):
        raise ValueError(f"{centered_path}: centered 世界几何未与候选轴对齐.")
    geometry_position = {
        int(blob_id): position for position, blob_id in enumerate(centered_blob_ids)
    }
    center_by_blob: dict[int, list[float]] = {}
    fits_by_blob: dict[int, bool] = {}
    for blob_position, blob_id_value in enumerate(blob_ids):
        blob_id = int(blob_id_value)
        if blob_id not in geometry_position:
            continue
        start = int(offsets[blob_position])
        stop = int(offsets[blob_position + 1])
        blob_voxels = voxels[start:stop]
        if not len(blob_voxels):
            raise ValueError(f"{blob_path}: blob {blob_id} 没有体素.")
        centered_position = geometry_position[blob_id]
        spacing = voxel_size_xyz[centered_position]
        grid_origin_xyz = (
            box_origin_xyz[centered_position]
            - box_start_zyx[centered_position][::-1] * spacing
        )
        center_xyz = grid_origin_xyz + (blob_voxels.mean(axis=0)[::-1] + 0.5) * spacing
        center_by_blob[blob_id] = center_xyz.tolist()
        fits_by_blob[blob_id] = bool(fits[blob_position])
    return center_by_blob, fits_by_blob, [blob_path, centered_path]


def _build_find_candidates(
    source_manifest_path: Path,
    stage1_root: Path,
    pdb_ids: list[str],
    stage3_members: set[tuple[str, int]],
) -> tuple[list[dict[str, Any]], list[Path]]:
    """筛选 77 个 PDB 的现存 Find_1 候选并补充 Stage3 范围与世界中心.

    输入参数:
        - source_manifest_path: Path, 一种受体条件的完整 test_0 manifest.
        - stage1_root: Path, 同一受体条件的 Find_1 held_out_test_0 产物根.
        - pdb_ids: list[str], PocketXMol 测试集冻结的 77 个小写 PDB 标识.
        - stage3_members: set[(str, int)], 允许计入小分子测试的 PDB 与 occurrence 编号.

    返回值:
        - candidates: list[dict[str, Any]], 保留目标 PDB 全部 centered 候选.
          - pdb_id: str, 小写 PDB 标识.
          - centered_box_index: int, centered 候选稳定编号.
          - source_blob_index: int, 完整来源 blob 编号.
          - candidate_selected: bool, 候选进入排序前段还是追加段.
          - gaussian_score: float, Find_1 Gaussian 分数.
          - source_probability_mean: float, 完整 blob 内来源概率均值.
          - foreground_target: bool, 是否存在 Matcher 规则下的唯一前景目标.
          - target_category: str|None, 唯一目标原始类别.
          - matched_occurrence_id: int|None, 唯一目标 occurrence 编号.
          - target_smiles: str|None, 唯一目标精确顶层 SMILES.
          - candidate_coverage: float, 交集占候选体素比例.
          - occurrence_coverage: float, 交集占 occurrence 体素比例.
          - overlap_score: float, 两项覆盖率几何平均.
          - target_scope: str, background、non_stage3_foreground 或 stage3_foreground.
          - blob_center_world_xyz_A: list[float], 长度 3 的完整 blob 世界 XYZ 中心, 单位 Å.
          - source_blob_fits_centered_box: bool, 完整 blob 是否全部位于 centered 80³ 框.
        - source_paths: list[Path], 实际读取且需要进入来源快照的文件.
    """
    source_manifest = read_json(source_manifest_path)
    split = source_manifest["splits"]["held_out_test_0"]
    manifest_pdb_ids = {
        str(pdb_id).lower()
        for pdb_id in source_manifest["summary"]["held_out_test_0"]["pdb_ids"]
    }
    if not set(pdb_ids) <= manifest_pdb_ids:
        raise ValueError(f"{source_manifest_path}: 缺少 PocketXMol 测试 PDB.")
    candidate_path = source_manifest_path.parent / str(split["candidates"])
    identity_path = source_manifest_path.parent / str(split["identities"])
    source_records = read_jsonl(candidate_path)
    pdb_set = set(pdb_ids)
    centers: dict[str, dict[int, list[float]]] = {}
    fits: dict[str, dict[int, bool]] = {}
    source_paths = [source_manifest_path, candidate_path, identity_path]
    for pdb_id in pdb_ids:
        centers[pdb_id], fits[pdb_id], paths = _find_centers(stage1_root, pdb_id)
        source_paths.extend(paths)
    candidates: list[dict[str, Any]] = []
    for record in source_records:
        pdb_id = str(record["pdb_id"])
        if pdb_id not in pdb_set:
            continue
        blob_id = int(record["source_blob_index"])
        occurrence_id = record["matched_occurrence_id"]
        foreground = bool(record["foreground_target"])
        if not foreground:
            target_scope = "background"
        elif (pdb_id, int(occurrence_id)) in stage3_members:
            target_scope = "stage3_foreground"
        else:
            target_scope = "non_stage3_foreground"
        if blob_id not in centers[pdb_id]:
            raise ValueError(f"{pdb_id}: 候选 blob {blob_id} 缺少完整世界中心.")
        candidates.append(
            {
                **record,
                "target_scope": target_scope,
                "blob_center_world_xyz_A": centers[pdb_id][blob_id],
                "source_blob_fits_centered_box": fits[pdb_id][blob_id],
            }
        )
    return candidates, source_paths


def _build_emap2lig_candidates(
    emap_root: Path,
    pdb_ids: list[str],
    stage3_members: set[tuple[str, int]],
    occurrence_facts: dict[str, tuple[dict[int, str], dict[int, str | None]]],
) -> tuple[list[dict[str, Any]], list[Path]]:
    """从 Emap2lig 原生概率与覆盖数组构造 77-PDB Stage1-only 候选.

    输入参数:
        - emap_root: Path, Emap2lig 正式 held_out_test_0 产物根.
        - pdb_ids: list[str], 固定分母中的 77 个小写 PDB 标识.
        - stage3_members: set[(str, int)], PocketXMol 测试 occurrence 成员.
        - occurrence_facts: dict[str, tuple[dict[int, str], dict[int, str|None]]].
          - 外层键: str, 小写 PDB 标识.
          - tuple[0]: occurrence 编号到原始 type_tag 的映射.
          - tuple[1]: occurrence 编号到精确 PreparedLigand.smiles 的映射.

    返回值:
        - candidates: list[dict[str, Any]], 使用原生概率均值的全部 Emap2lig 候选.
          - pdb_id: str, 小写 PDB 标识.
          - source_blob_index: int, Emap2lig official blob 编号.
          - candidate_selected: bool, 候选进入排序前段还是追加段.
          - source_probability_mean: float, 原生 ligand probability 均值.
          - foreground_target: bool, 是否存在 Matcher 规则下的唯一前景目标.
          - target_category: str|None, 唯一目标原始类别.
          - matched_occurrence_id: int|None, 唯一目标 occurrence 编号.
          - target_smiles: str|None, 唯一目标精确顶层 SMILES.
          - target_scope: str, 候选三态适用范围.
          - candidate_coverage: float, 交集占候选体素比例.
          - occurrence_coverage: float, 交集占 occurrence 体素比例.
          - overlap_score: float, 两项覆盖率几何平均.
          - blob_center_world_xyz_A: list[float], 长度 3 的完整 blob 世界 XYZ 中心, 单位 Å.
          - centered_box_index: None, Emap2lig 不使用 Find_1 centered 框.
          - gaussian_score: None, Emap2lig 不产生 Gaussian 分数.
          - source_blob_fits_centered_box: None, Emap2lig 不使用 centered 框.
        - source_paths: list[Path], 实际读取的评价 NPZ 与 official blob NPZ.
    """
    candidates: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    for pdb_id in pdb_ids:
        evaluation_path = emap_root / "evaluation/per_pdb" / f"{pdb_id}.npz"
        mapped_path = emap_root / "mapped" / pdb_id / "official_blobs.npz"
        source_paths.extend([evaluation_path, mapped_path])
        with np.load(evaluation_path, allow_pickle=False) as archive:
            # int64, (N_occurrence,); 与交集矩阵第二轴逐项对齐的真实 occurrence 编号.
            occurrence_ids = np.asarray(archive["occurrence_id"], dtype=np.int64)
            # int64, (N_candidate,); 评价数组第一轴的 Emap2lig blob 编号.
            ranked_blob_ids = np.asarray(archive["source_blob_index"], dtype=np.int64)
            # float64, (N_candidate,); Emap2lig 原生 ligand probability 均值.
            ranked_scores = np.asarray(archive["candidate_score"], dtype=np.float64)
            # bool, (N_candidate,); Emap2lig 冻结候选选择掩码.
            selected = np.asarray(archive["candidate_selected"], dtype=np.bool_)
            # int64, (N_candidate, N_occurrence); 候选与真值掩码的交集体素数.
            intersections = np.asarray(archive["intersections"], dtype=np.int64)
            # int64, (N_candidate,); 每个完整候选的唯一体素数.
            candidate_sizes = np.asarray(archive["pred_sizes"], dtype=np.int64)
            # int64, (N_occurrence,); 每个真实 occurrence 的唯一体素数.
            occurrence_sizes = np.asarray(archive["gt_sizes"], dtype=np.int64)
        with np.load(mapped_path, allow_pickle=False) as archive:
            # int64, (N_mapped,); official blob 连接数组使用的稳定编号.
            mapped_blob_ids = np.asarray(archive["source_blob_index"], dtype=np.int64)
            # float64, (N_mapped,); 与 mapped_blob_ids 对齐的原生概率均值.
            mapped_scores = np.asarray(archive["source_probability_mean"], dtype=np.float64)
            # int64, (N_mapped+1,); 相邻两项切分 voxels 中一个完整 blob.
            offsets = np.asarray(archive["voxel_offsets"], dtype=np.int64)
            # int64, (N_voxel, 3); 连接保存的全局 ZYX 体素编号.
            voxels = np.asarray(archive["voxel_index_global_zyx"], dtype=np.int64)
            # float64, (3,); 全局网格的世界 XYZ 原点, 单位 Å.
            origin_xyz = np.asarray(archive["origin_xyz"], dtype=np.float64)
            # float64, (3,); 世界 XYZ 三轴体素尺寸, 单位 Å.
            spacing_xyz = np.asarray(archive["voxel_size_xyz"], dtype=np.float64)
        if intersections.shape != (len(ranked_blob_ids), len(occurrence_ids)):
            raise ValueError(f"{evaluation_path}: 交集数组未与候选和 occurrence 轴对齐.")
        mapped_position = {
            int(blob_id): position for position, blob_id in enumerate(mapped_blob_ids)
        }
        type_by_id, smiles_by_id = occurrence_facts[pdb_id]
        for rank_position, blob_id_value in enumerate(ranked_blob_ids):
            blob_id = int(blob_id_value)
            position = mapped_position[blob_id]
            score = float(mapped_scores[position])
            if not math.isclose(score, float(ranked_scores[rank_position]), rel_tol=0.0, abs_tol=1e-7):
                raise ValueError(f"{pdb_id}: Emap2lig 原生均值与评价排序分数不一致.")
            matched_id, candidate_coverage, occurrence_coverage, overlap_score = (
                _target_from_intersections(
                    occurrence_ids,
                    intersections[rank_position],
                    int(candidate_sizes[rank_position]),
                    occurrence_sizes,
                )
            )
            if matched_id is None:
                target_scope = "background"
                target_category = None
                target_smiles = None
            else:
                target_category = type_by_id[matched_id]
                if target_category == "other":
                    raise ValueError(
                        f"{pdb_id}/{blob_id}: Emap2lig 最大覆盖目标为 other; "
                        "冻结数据预演要求该计数为 0."
                    )
                target_smiles = smiles_by_id[matched_id]
                target_scope = (
                    "stage3_foreground"
                    if (pdb_id, matched_id) in stage3_members
                    else "non_stage3_foreground"
                )
            start = int(offsets[position])
            stop = int(offsets[position + 1])
            blob_voxels = voxels[start:stop]
            center_xyz = origin_xyz + (blob_voxels.mean(axis=0)[::-1] + 0.5) * spacing_xyz
            candidates.append(
                {
                    "pdb_id": pdb_id,
                    "centered_box_index": None,
                    "source_blob_index": blob_id,
                    "candidate_selected": bool(selected[rank_position]),
                    "gaussian_score": None,
                    "source_probability_mean": score,
                    "foreground_target": matched_id is not None,
                    "target_category": target_category,
                    "matched_occurrence_id": matched_id,
                    "target_smiles": target_smiles,
                    "candidate_coverage": candidate_coverage,
                    "occurrence_coverage": occurrence_coverage,
                    "overlap_score": overlap_score,
                    "target_scope": target_scope,
                    "blob_center_world_xyz_A": center_xyz.tolist(),
                    "source_blob_fits_centered_box": None,
                }
            )
    return candidates, source_paths


# ================================================================================================


def build_manifest(config_path: Path) -> Path:
    """构建并原子发布 matcher_test_smallmolecule_v1 目录.

    输入参数:
        - config_path: Path, 显式给出全部来源与输出根的 YAML 配置.

    返回值:
        - manifest_path: Path, 已发布的 manifest_root/manifest.json.

    目录产物:
        - manifest.json: object, 唯一入口.
          - schema: str, manifest 契约版本.
          - coverage_threshold: float, 候选与 occurrence 共用的双向覆盖阈值.
          - pdb_ids: list[str], 固定 77-PDB 分母顺序.
          - generation_runtime.task_project_root: str, 当前评测 release 项目根.
          - generation_runtime.task_launch_dir: str, 当前正式 launch 目录.
          - generation_runtime.slurm_job_id: str, 实际 Slurm Job ID.
          - files: dict[str, str], 逻辑文件名到相对路径的映射.
          - file_sha256: dict[str, str], 每个派生文件的 SHA-256.
          - scientific_contract.identity_key: str, 固定为 PreparedLigand.smiles.
          - scientific_contract.candidate_target: str, 唯一 occurrence 选择规则.
          - scientific_contract.candidate_order: str, selected 优先扩展顺序.
          - scientific_contract.attempt_rule: str, 三态候选的计费规则.
          - scientific_contract.matcher_rule: str, 只比较精确 SMILES 的身份规则.
          - scientific_contract.small_molecule_answer_rule: str, 最终类别 logits 掩码规则.
          - scientific_contract.blob_center: str, 世界坐标中心定义.
          - summary.pdb_count: int, 固定 PDB 分母.
          - summary.stage3_occurrence_count: int, PocketXMol 测试 occurrence 数.
          - summary.stage3_identity_count: int, PDB 内精确 SMILES 去重数.
          - summary.identity_count: int, Matcher 完整身份槽数.
          - summary.small_molecule_answer_identity_count: int, 类别掩码允许身份数.
          - summary.real_candidate_count: int, 真实受体全部候选数.
          - summary.cryoatom2_candidate_count: int, CryoAtom2 全部候选数.
          - summary.emap2lig_candidate_count: int, Emap2lig 全部候选数.
          - summary.real_outside_centered_candidate_count: int, 真实受体框外候选数.
          - summary.cryoatom2_outside_centered_candidate_count: int, CryoAtom2 框外候选数.
          - summary.real_target_scope_counts: dict[str, int], 真实受体三态计数.
          - summary.cryoatom2_target_scope_counts: dict[str, int], CryoAtom2 三态计数.
          - summary.emap2lig_target_scope_counts: dict[str, int], Emap2lig 三态计数.
          - summary.pdb_ids: list[str], 与顶层 pdb_ids 相同的冻结顺序.
        - stage3_occurrences.jsonl: 446 条 PocketXMol 原始记录.
          - split: str, 固定为 test.
          - pdb_id: str, 小写 PDB 标识.
          - candidate_id: int, PDB 内 occurrence 编号.
          - n_heavy_atoms: int, 当前小分子的沉积重原子数.
          - center_offset_xyz_A: list[float], 长度 3 的坐标中心偏移, 单位 Å.
          - sampling_seed: int, PocketXMol 冻结采样种子.
          - views: list[str], PocketXMol 冻结视图名称.
          - prepared_smiles: str, 精确顶层 PreparedLigand.smiles.
        - identities.jsonl: 180 条 object.
          - pdb_id: str, 小写 PDB 标识.
          - smiles: str, 精确顶层 PreparedLigand.smiles.
          - language_relative_path: str, 相对数据根的 SMI-TED NPZ 路径.
          - type_tag: str, 同一精确 SMILES 唯一原始类别.
          - occurrence_ids: list[int], 共享该 SMILES 的全部 occurrence 编号.
          - stage3_occurrence_ids: list[int], occurrence_ids 中的 Stage3 子集.
          - allowed_as_small_molecule_answer: bool, 最终类别掩码.
        - real_candidates.jsonl: 598 条真实受体 Find_1 候选, 字段由 _build_find_candidates 定义.
        - cryoatom2_candidates.jsonl: 632 条 CryoAtom2 Find_1 候选, 字段由 _build_find_candidates 定义.
        - emap2lig_candidates.jsonl: 30,561 条原生候选, 字段见 _build_emap2lig_candidates.
        - source_snapshot.json: object.
          - schema: str, 来源快照版本.
          - files: list[object], 按绝对路径排序的实际读取来源.
          - files[].path: str, 来源文件绝对路径.
          - files[].size_bytes: int, 来源文件字节数.
          - files[].sha256: str, 来源文件内容摘要.
        - README.md: Markdown, 说明当前目录的字段与跨文件对齐关系.
    """
    config = load_config(config_path)
    sources = config["sources"]
    expected = config["expected"]
    output_root = Path(config["outputs"]["manifest_root"])
    if output_root.exists():
        raise FileExistsError(f"正式 manifest 根已经存在, 禁止覆盖: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building.", dir=output_root.parent)
    )
    source_paths: list[Path] = [Path(config_path)]
    try:
        stage3_path = Path(sources["stage3_occurrences"])
        stage3_records = read_jsonl(stage3_path)
        source_paths.append(stage3_path)
        pdb_ids = sorted({str(record["pdb_id"]).lower() for record in stage3_records})
        stage3_members = {
            (str(record["pdb_id"]).lower(), int(record["candidate_id"]))
            for record in stage3_records
        }
        if len(stage3_members) != len(stage3_records):
            raise ValueError("Stage3 测试清单含重复的 PDB 与 occurrence 编号.")
        if len(stage3_records) != int(expected["stage3_occurrence_count"]):
            raise ValueError("Stage3 occurrence 数量与冻结期望不一致.")
        if len(pdb_ids) != int(expected["pdb_count"]):
            raise ValueError("Stage3 PDB 数量与冻结期望不一致.")

        data_root = Path(sources["data_root"])
        prepared_root = Path(sources["prepared_root"])
        occurrence_facts: dict[str, tuple[dict[int, str], dict[int, str | None]]] = {}
        for pdb_id in pdb_ids:
            type_by_id, smiles_by_id, paths = _read_occurrence_facts(
                data_root, prepared_root, pdb_id
            )
            occurrence_facts[pdb_id] = (type_by_id, smiles_by_id)
            source_paths.extend(paths)
        for record in stage3_records:
            pdb_id = str(record["pdb_id"]).lower()
            occurrence_id = int(record["candidate_id"])
            type_by_id, smiles_by_id = occurrence_facts[pdb_id]
            if occurrence_id not in type_by_id or occurrence_id not in smiles_by_id:
                raise ValueError(f"{pdb_id}/{occurrence_id}: Stage3 occurrence 无法回连来源记录.")
            if type_by_id[occurrence_id] != "small_molecule":
                raise ValueError(f"{pdb_id}/{occurrence_id}: Stage3 occurrence 不是 small_molecule.")
            if record["prepared_smiles"] != smiles_by_id[occurrence_id]:
                raise ValueError(f"{pdb_id}/{occurrence_id}: Stage3 SMILES 与 PreparedLigand 不一致.")

        real_manifest_path = Path(sources["real_manifest"])
        cryo_manifest_path = Path(sources["cryoatom2_manifest"])
        real_candidates, paths = _build_find_candidates(
            real_manifest_path,
            Path(sources["find_real_root"]),
            pdb_ids,
            stage3_members,
        )
        source_paths.extend(paths)
        cryo_candidates, paths = _build_find_candidates(
            cryo_manifest_path,
            Path(sources["find_cryoatom2_root"]),
            pdb_ids,
            stage3_members,
        )
        source_paths.extend(paths)

        for records in (real_candidates, cryo_candidates):
            for record in records:
                occurrence_id = record["matched_occurrence_id"]
                if occurrence_id is None:
                    continue
                pdb_id = str(record["pdb_id"])
                expected_smiles = occurrence_facts[pdb_id][1][int(occurrence_id)]
                if record["target_smiles"] != expected_smiles:
                    raise ValueError(
                        f"{pdb_id}/{record['source_blob_index']}: 候选目标 SMILES 与 occurrence 不一致."
                    )

        real_manifest = read_json(real_manifest_path)
        real_identity_path = real_manifest_path.parent / str(
            real_manifest["splits"]["held_out_test_0"]["identities"]
        )
        supported_identities = [
            record
            for record in read_jsonl(real_identity_path)
            if str(record["pdb_id"]) in set(pdb_ids)
        ]
        cryo_manifest = read_json(cryo_manifest_path)
        cryo_identity_path = cryo_manifest_path.parent / str(
            cryo_manifest["splits"]["held_out_test_0"]["identities"]
        )
        source_paths.append(cryo_identity_path)
        if read_jsonl(real_identity_path) != read_jsonl(cryo_identity_path):
            raise ValueError("真实受体与 CryoAtom2 的完整身份轴内容不一致.")
        identities: list[dict[str, Any]] = []
        stage3_by_pdb: dict[str, set[int]] = defaultdict(set)
        for pdb_id, occurrence_id in stage3_members:
            stage3_by_pdb[pdb_id].add(occurrence_id)
        for identity in supported_identities:
            pdb_id = str(identity["pdb_id"])
            smiles = str(identity["smiles"])
            type_by_id, smiles_by_id = occurrence_facts[pdb_id]
            occurrence_ids = sorted(
                occurrence_id
                for occurrence_id, occurrence_smiles in smiles_by_id.items()
                if occurrence_smiles == smiles
            )
            type_tags = {type_by_id[occurrence_id] for occurrence_id in occurrence_ids}
            if len(type_tags) != 1:
                raise ValueError(f"{pdb_id}: SMILES 身份没有唯一原始类别.")
            type_tag = next(iter(type_tags))
            # 同一精确 SMILES 共享一个最终答案类别; Stage3 成员只用于留证, 不定义答案轴.
            identities.append(
                {
                    **identity,
                    "type_tag": type_tag,
                    "occurrence_ids": occurrence_ids,
                    "stage3_occurrence_ids": [
                        occurrence_id
                        for occurrence_id in occurrence_ids
                        if occurrence_id in stage3_by_pdb[pdb_id]
                    ],
                    "allowed_as_small_molecule_answer": type_tag == "small_molecule",
                }
            )

        emap_candidates, paths = _build_emap2lig_candidates(
            Path(sources["emap2lig_root"]),
            pdb_ids,
            stage3_members,
            occurrence_facts,
        )
        source_paths.extend(paths)
        _assert_unique_candidate_keys("真实受体 Find_1", real_candidates)
        _assert_unique_candidate_keys("CryoAtom2 Find_1", cryo_candidates)
        _assert_unique_candidate_keys("Emap2lig", emap_candidates)

        files = {
            "stage3_occurrences": "stage3_occurrences.jsonl",
            "identities": "identities.jsonl",
            "real_candidates": "real_candidates.jsonl",
            "cryoatom2_candidates": "cryoatom2_candidates.jsonl",
            "emap2lig_candidates": "emap2lig_candidates.jsonl",
            "source_snapshot": "source_snapshot.json",
            "readme": "README.md",
        }
        write_jsonl(temporary_root / files["stage3_occurrences"], stage3_records)
        write_jsonl(temporary_root / files["identities"], identities)
        write_jsonl(temporary_root / files["real_candidates"], real_candidates)
        write_jsonl(temporary_root / files["cryoatom2_candidates"], cryo_candidates)
        write_jsonl(temporary_root / files["emap2lig_candidates"], emap_candidates)
        readme_source = Path(__file__).with_name("MANIFEST_README.md")
        shutil.copyfile(readme_source, temporary_root / files["readme"])

        # Path 列表, 对全部实际读取来源去重后逐文件绑定内容, 不只记录目录名称.
        unique_source_paths = sorted({path.resolve() for path in source_paths}, key=str)
        source_snapshot = {
            "schema": "matcher_test_smallmolecule_source_snapshot_v1",
            "files": [
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in unique_source_paths
            ],
        }
        write_json(temporary_root / files["source_snapshot"], source_snapshot)

        summary = {
            "pdb_count": len(pdb_ids),
            "stage3_occurrence_count": len(stage3_records),
            "stage3_identity_count": len(
                {
                    (str(record["pdb_id"]), str(record["prepared_smiles"]))
                    for record in stage3_records
                }
            ),
            "identity_count": len(identities),
            "small_molecule_answer_identity_count": sum(
                bool(record["allowed_as_small_molecule_answer"]) for record in identities
            ),
            "real_candidate_count": len(real_candidates),
            "cryoatom2_candidate_count": len(cryo_candidates),
            "emap2lig_candidate_count": len(emap_candidates),
            "real_outside_centered_candidate_count": sum(
                not bool(record["source_blob_fits_centered_box"])
                for record in real_candidates
            ),
            "cryoatom2_outside_centered_candidate_count": sum(
                not bool(record["source_blob_fits_centered_box"])
                for record in cryo_candidates
            ),
            "real_target_scope_counts": dict(
                sorted(
                    (scope, sum(record["target_scope"] == scope for record in real_candidates))
                    for scope in {record["target_scope"] for record in real_candidates}
                )
            ),
            "cryoatom2_target_scope_counts": dict(
                sorted(
                    (scope, sum(record["target_scope"] == scope for record in cryo_candidates))
                    for scope in {record["target_scope"] for record in cryo_candidates}
                )
            ),
            "emap2lig_target_scope_counts": dict(
                sorted(
                    (scope, sum(record["target_scope"] == scope for record in emap_candidates))
                    for scope in {record["target_scope"] for record in emap_candidates}
                )
            ),
            "pdb_ids": pdb_ids,
        }
        expected_counts = {
            "stage3_identity_count": int(expected["stage3_identity_count"]),
            "identity_count": int(expected["identity_count"]),
            "small_molecule_answer_identity_count": int(
                expected["small_molecule_answer_identity_count"]
            ),
            "real_candidate_count": int(expected["real_candidate_count"]),
            "cryoatom2_candidate_count": int(expected["cryoatom2_candidate_count"]),
            "emap2lig_candidate_count": int(expected["emap2lig_candidate_count"]),
            "real_outside_centered_candidate_count": int(
                expected["outside_centered_candidate_count"]["real_receptor"]
            ),
            "cryoatom2_outside_centered_candidate_count": int(
                expected["outside_centered_candidate_count"]["cryoatom2_receptor"]
            ),
        }
        for field, expected_value in expected_counts.items():
            if summary[field] != expected_value:
                raise ValueError(
                    f"{field} 与冻结期望不一致: {summary[field]} != {expected_value}."
                )
        expected_scope_counts = {
            "real_target_scope_counts": dict(expected["target_scope_counts"]["real_receptor"]),
            "cryoatom2_target_scope_counts": dict(
                expected["target_scope_counts"]["cryoatom2_receptor"]
            ),
            "emap2lig_target_scope_counts": dict(expected["target_scope_counts"]["emap2lig"]),
        }
        for field, expected_value in expected_scope_counts.items():
            if summary[field] != expected_value:
                raise ValueError(
                    f"{field} 与冻结期望不一致: {summary[field]} != {expected_value}."
                )

        stage3_identity_keys = {
            (str(record["pdb_id"]), str(record["prepared_smiles"]))
            for record in stage3_records
        }
        supported_identity_keys = {
            (str(record["pdb_id"]), str(record["smiles"])) for record in identities
        }
        small_molecule_identity_keys = {
            (str(record["pdb_id"]), str(record["smiles"]))
            for record in identities
            if record["allowed_as_small_molecule_answer"]
        }
        if not stage3_identity_keys <= supported_identity_keys:
            raise ValueError("存在不受 Matcher 完整身份轴支持的 Stage3 精确 SMILES.")
        if not stage3_identity_keys <= small_molecule_identity_keys:
            raise ValueError("存在未被 small-molecule 答案掩码允许的 Stage3 精确 SMILES.")
        if any(
            not np.all(np.isfinite(record["blob_center_world_xyz_A"]))
            or len(record["blob_center_world_xyz_A"]) != 3
            for records in (real_candidates, cryo_candidates, emap_candidates)
            for record in records
        ):
            raise ValueError("存在非有限或非三维的 blob 世界坐标中心.")

        manifest = {
            "schema": "matcher_test_smallmolecule_v1",
            "coverage_threshold": COVERAGE_THRESHOLD,
            "pdb_ids": pdb_ids,
            "generation_runtime": {
                "task_project_root": os.environ.get("TASK_PROJECT_ROOT"),
                "task_launch_dir": os.environ.get("TASK_LAUNCH_DIR"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            "files": files,
            "file_sha256": {
                name: sha256_file(temporary_root / filename)
                for name, filename in files.items()
            },
            "scientific_contract": {
                "identity_key": "PreparedLigand.smiles",
                "candidate_target": "双向覆盖率均不低于 0.30 后最大化几何平均, 并列取较小 occurrence_id",
                "candidate_order": "selected 段后接 unselected 段; 段内分数降序、source_blob_index 升序",
                "attempt_rule": "背景和 Stage3 目标消耗次数; 非 Stage3 前景免费跳过",
                "matcher_rule": "只比较同一次完整身份 forward 的精确 SMILES argmax, 不使用前景阈值",
                "small_molecule_answer_rule": "完整身份 forward 后仅屏蔽 type_tag 不是 small_molecule 的最终 logits",
                "blob_center": "完整来源 blob 的全局体素中心算术平均, 世界 XYZ, 单位 Å",
            },
            "summary": summary,
        }
        write_json(temporary_root / "manifest.json", manifest)
        os.replace(temporary_root, output_root)
        return output_root / "manifest.json"
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    """解析命令行并构建正式小分子测试 manifest.

    输入参数:
        - argv: list[str]|None, 可测试注入的命令行参数; None 时读取当前进程参数.

    返回值:
        - exit_code: int, 正常发布 manifest 后固定为 0; 异常直接传播并保持失败状态.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(build_manifest(arguments.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
