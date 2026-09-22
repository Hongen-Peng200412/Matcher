"""构建 Stage3 小分子测试集使用的 Matcher 评估清单.

主要入口是 build_manifest. 它连接 PocketXMol 测试 occurrence、Matcher test_0
清单、Find_1 完整 blob 和 Emap2lig 正式候选, 并写出身份与候选 JSONL. 本模块
不修改任何来源文件, 也不执行 Matcher 前向.
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import load_config, read_json, read_jsonl, write_json, write_jsonl


COVERAGE_THRESHOLD = 0.30


def _read_occurrence_facts(
    data_root: Path,
    prepared_root: Path,
    pdb_id: str,
) -> tuple[dict[int, str], dict[int, str | None]]:
    """读取一个 PDB 的 occurrence 类别与精确顶层 SMILES.

    输入参数:
        - data_root: Path, AdaLigand Ori_Data 根目录.
        - prepared_root: Path, 按 PDB 保存 prepared_smiles.jsonl 的共同父目录.
        - pdb_id: str, 小写 PDB 标识, 如 11jb.

    返回值:
        - type_by_id: dict[int, str], occurrence 编号到 type_tag 的映射.
        - smiles_by_id: dict[int, str|None], occurrence 编号到 PreparedLigand.smiles 的映射.
    """
    occurrence_records = read_jsonl(data_root / "parse" / pdb_id / "occurrences.jsonl")
    prepared_records = read_jsonl(prepared_root / pdb_id / "prepared_smiles.jsonl")
    type_by_id = {
        int(record["candidate_id"]): str(record["type_tag"])
        for record in occurrence_records
    }
    prepared_by_id = {
        int(record["candidate_id"]): record.get("smiles") for record in prepared_records
    }
    smiles_by_id = {
        occurrence_id: (
            str(prepared_by_id[occurrence_id])
            if isinstance(prepared_by_id.get(occurrence_id), str)
            and prepared_by_id[occurrence_id]
            else None
        )
        for occurrence_id in type_by_id
    }
    return type_by_id, smiles_by_id


def _target_from_intersections(
    occurrence_ids: np.ndarray,
    intersections: np.ndarray,
    candidate_size: int,
    occurrence_sizes: np.ndarray,
) -> tuple[int | None, float, float, float]:
    """按双向覆盖和几何平均规则选择唯一 occurrence.

    输入参数:
        - occurrence_ids: int, (N_occurrence,), 真实 occurrence 编号.
        - intersections: int, (N_occurrence,), 当前候选与每个 occurrence 的交集体素数.
        - candidate_size: int, 当前候选的体素数.
        - occurrence_sizes: int, (N_occurrence,), 每个 occurrence 的体素数.

    返回值:
        - matched_occurrence_id: int|None, 最大几何平均对应的 occurrence 编号, 无有效覆盖时为 None.
        - candidate_coverage: float, 交集体素数除以候选体素数, 背景为 0.
        - occurrence_coverage: float, 交集体素数除以 occurrence 体素数, 背景为 0.
        - overlap_score: float, 两项覆盖率的几何平均, 背景为 0.
    """
    candidate_coverage = intersections.astype(np.float64) / float(candidate_size)
    occurrence_coverage = intersections.astype(np.float64) / occurrence_sizes.astype(np.float64)
    eligible = (candidate_coverage >= COVERAGE_THRESHOLD) & (occurrence_coverage >= COVERAGE_THRESHOLD)
    if not bool(np.any(eligible)):
        return None, 0.0, 0.0, 0.0
    overlap_score = np.sqrt(candidate_coverage * occurrence_coverage)
    eligible_positions = np.flatnonzero(eligible)
    # float64, (K,), 先按几何平均降序, 再按 occurrence 编号升序选择唯一目标.
    order = np.lexsort(
        (
            occurrence_ids[eligible_positions],
            -overlap_score[eligible_positions],
        )
    )
    position = int(eligible_positions[int(order[0])])
    return (
        int(occurrence_ids[position]),
        float(candidate_coverage[position]),
        float(occurrence_coverage[position]),
        float(overlap_score[position]),
    )


def _find_centers(
    stage1_root: Path,
    pdb_id: str,
) -> tuple[dict[int, list[float]], dict[int, bool]]:
    """计算一个 PDB 中每个 Find_1 来源 blob 的世界坐标中心.

    输入参数:
        - stage1_root: Path, 一种受体条件的 held_out_test_0 根目录.
        - pdb_id: str, 当前小写 PDB 标识.

    返回值:
        - center_by_blob: dict[int, list[float]], source_blob_index 到世界 XYZ 中心的映射, 单位 Å.
        - fits_by_blob: dict[int, bool], 完整 blob 是否全部位于对应 centered 80³ 框.
    """
    pdb_root = stage1_root / pdb_id
    with np.load(pdb_root / "blobs/F2_blobs.npz", allow_pickle=False) as archive:
        # int64, (N_blob,), 完整来源 blob 在当前 PDB 内的稳定编号.
        blob_ids = np.asarray(archive["blob_index"], dtype=np.int64)
        # int64, (N_blob + 1,), 相邻两项切分 voxel_index_global_zyx 中一个完整 blob 的体素区间.
        offsets = np.asarray(archive["voxel_offsets"], dtype=np.int64)
        # int64, (N_voxel, 3), 连接保存的全局 ZYX 体素编号.
        voxels = np.asarray(archive["voxel_index_global_zyx"], dtype=np.int64)
        # bool, (N_blob,), 完整 blob 是否全部位于对应 centered 80³ 框.
        fits = np.asarray(archive["fits_centered_box"], dtype=np.bool_)
    with np.load(pdb_root / "centered/F2_centered.npz", allow_pickle=False) as archive:
        # int64, (N_centered,), centered 候选到完整来源 blob 的编号映射.
        centered_blob_ids = np.asarray(archive["source_blob_index"], dtype=np.int64)
        # float64, (N_centered, 3), centered 框起点的全局 ZYX 体素编号.
        box_start_zyx = np.asarray(archive["box_start_zyx"], dtype=np.float64)
        # float64, (N_centered, 3), centered 框首体素的世界 XYZ 原点, 单位 Å.
        box_origin_xyz = np.asarray(archive["box_origin_world"], dtype=np.float64)
        # float64, (N_centered, 3), 世界 XYZ 三轴体素尺寸, 单位 Å.
        voxel_size_xyz = np.asarray(archive["voxel_size_world"], dtype=np.float64)

    # blob_id 对应 F2_centered.npz 中的第几个实体
    centered_position = {
        int(blob_id): position for position, blob_id in enumerate(centered_blob_ids)
    }
    center_by_blob: dict[int, list[float]] = {}
    fits_by_blob: dict[int, bool] = {}
    for blob_position, blob_id_value in enumerate(blob_ids):
        blob_id = int(blob_id_value)
        if blob_id not in centered_position:
            continue
        start = int(offsets[blob_position])
        stop = int(offsets[blob_position + 1])
        blob_voxels = voxels[start:stop]
        position = centered_position[blob_id]
        spacing_xyz = voxel_size_xyz[position]
        grid_origin_xyz = (
            box_origin_xyz[position] - box_start_zyx[position][::-1] * spacing_xyz
        )
        center_xyz = (
            grid_origin_xyz + (blob_voxels.mean(axis=0)[::-1] + 0.5) * spacing_xyz
        )
        center_by_blob[blob_id] = center_xyz.tolist()
        fits_by_blob[blob_id] = bool(fits[blob_position])
    return center_by_blob, fits_by_blob


def _build_find_candidates(
    source_manifest_path: Path,
    stage1_root: Path,
    pdb_ids: list[str],
    stage3_members: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    """补充一种 Find_1 受体条件下的 Stage3 范围和 blob 世界中心.

    输入参数:
        - source_manifest_path: Path, 一种受体条件的完整 test_0 manifest.
        - stage1_root: Path, 同一受体条件的 Find_1 held_out_test_0 产物根.
        - pdb_ids: list[str], Stage3 测试集的有序小写 PDB 标识.
        - stage3_members: set[tuple[str, int]], Stage3 测试集中的 PDB 与 occurrence 编号.

    返回值:
        - candidates: list[dict[str, Any]], 保留目标 PDB 的全部 centered 候选, 并增加 target_scope、blob_center_world_xyz_A 和 source_blob_fits_centered_box.
    """
    source_manifest = read_json(source_manifest_path)
    split = source_manifest["splits"]["held_out_test_0"]
    candidate_path = source_manifest_path.parent / str(split["candidates"])
    source_records = read_jsonl(candidate_path)
    pdb_set = set(pdb_ids)
    centers: dict[str, dict[int, list[float]]] = {}
    fits: dict[str, dict[int, bool]] = {}
    for pdb_id in pdb_ids:
        centers[pdb_id], fits[pdb_id] = _find_centers(stage1_root, pdb_id)

    candidates: list[dict[str, Any]] = []
    for record in source_records:
        pdb_id = str(record["pdb_id"])
        if pdb_id not in pdb_set:
            continue
        blob_id = int(record["source_blob_index"])
        occurrence_id = record["matched_occurrence_id"]
        if not bool(record["foreground_target"]):
            target_scope = "background"
        elif (pdb_id, int(occurrence_id)) in stage3_members:
            target_scope = "stage3_foreground"
        else:
            target_scope = "non_stage3_foreground"
        candidates.append(
            {
                **record,
                "target_scope": target_scope,
                "blob_center_world_xyz_A": centers[pdb_id][blob_id],
                "source_blob_fits_centered_box": fits[pdb_id][blob_id],
            }
        )
    return candidates


def _build_emap2lig_candidates(
    emap_root: Path,
    pdb_ids: list[str],
    stage3_members: set[tuple[str, int]],
    occurrence_facts: dict[str, tuple[dict[int, str], dict[int, str | None]]],
) -> list[dict[str, Any]]:
    """从 Emap2lig 原生概率和覆盖数组构造 Stage1-only 候选.

    输入参数:
        - emap_root: Path, Emap2lig 正式 held_out_test_0 产物根.
        - pdb_ids: list[str], Stage3 测试集的有序 PDB 标识.
        - stage3_members: set[tuple[str, int]], Stage3 测试 occurrence 成员.
        - occurrence_facts: dict[str, tuple[dict[int, str], dict[int, str|None]]], 每个 PDB 的类别与精确 SMILES 映射.

    返回值:
        - candidates: list[dict[str, Any]], 使用原生概率均值排序的全部 Emap2lig 候选, 字段与 Find_1 候选的评估接口一致.
    """
    candidates: list[dict[str, Any]] = []
    for pdb_id in pdb_ids:
        evaluation_path = emap_root / "evaluation/per_pdb" / f"{pdb_id}.npz"
        mapped_path = emap_root / "mapped" / pdb_id / "official_blobs.npz"
        with np.load(evaluation_path, allow_pickle=False) as archive:
            # int64, (N_occurrence,), 与 intersections 第二维逐项对齐的真实 occurrence 编号.
            occurrence_ids = np.asarray(archive["occurrence_id"], dtype=np.int64)
            # int64, (N_candidate,), 评价数组第一维的 Emap2lig blob 编号.
            ranked_blob_ids = np.asarray(archive["source_blob_index"], dtype=np.int64)
            # bool, (N_candidate,), Emap2lig 候选的 selected 标记.
            selected = np.asarray(archive["candidate_selected"], dtype=np.bool_)
            # int64, (N_candidate, N_occurrence), 候选与真实 occurrence 的交集体素数.
            intersections = np.asarray(archive["intersections"], dtype=np.int64)
            # int64, (N_candidate,), 每个完整候选的唯一体素数.
            candidate_sizes = np.asarray(archive["pred_sizes"], dtype=np.int64)
            # int64, (N_occurrence,), 每个真实 occurrence 的唯一体素数.
            occurrence_sizes = np.asarray(archive["gt_sizes"], dtype=np.int64)
        with np.load(mapped_path, allow_pickle=False) as archive:
            # int64, (N_mapped,), official blob 连接数组使用的稳定编号.
            mapped_blob_ids = np.asarray(archive["source_blob_index"], dtype=np.int64)
            # float64, (N_mapped,), 与 mapped_blob_ids 对齐的原生概率均值.
            mapped_scores = np.asarray(
                archive["source_probability_mean"], dtype=np.float64
            )
            # int64, (N_mapped + 1,), 相邻两项切分 voxel_index_global_zyx 中一个完整 blob.
            offsets = np.asarray(archive["voxel_offsets"], dtype=np.int64)
            # int64, (N_voxel, 3), 连接保存的全局 ZYX 体素编号.
            voxels = np.asarray(archive["voxel_index_global_zyx"], dtype=np.int64)
            # float64, (3,), 全局网格的世界 XYZ 原点, 单位 Å.
            origin_xyz = np.asarray(archive["origin_xyz"], dtype=np.float64)
            # float64, (3,), 世界 XYZ 三轴体素尺寸, 单位 Å.
            spacing_xyz = np.asarray(archive["voxel_size_xyz"], dtype=np.float64)

        mapped_position = {
            int(blob_id): position for position, blob_id in enumerate(mapped_blob_ids)
        }
        type_by_id, smiles_by_id = occurrence_facts[pdb_id]
        for rank_position, blob_id_value in enumerate(ranked_blob_ids):
            blob_id = int(blob_id_value)
            position = mapped_position[blob_id]
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
                target_smiles = smiles_by_id[matched_id]
                target_scope = (
                    "stage3_foreground"
                    if (pdb_id, matched_id) in stage3_members
                    else "non_stage3_foreground"
                )
            start = int(offsets[position])
            stop = int(offsets[position + 1])
            center_xyz = (
                origin_xyz + (voxels[start:stop].mean(axis=0)[::-1] + 0.5) * spacing_xyz
            )
            candidates.append(
                {
                    "pdb_id": pdb_id,
                    "centered_box_index": None,
                    "source_blob_index": blob_id,
                    "candidate_selected": bool(selected[rank_position]),
                    "gaussian_score": None,
                    "source_probability_mean": float(mapped_scores[position]),
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
    return candidates


def build_manifest(config_path: Path) -> Path:
    """构建并写出 Stage3 小分子测试集的模型无关清单.

    输入参数:
        - config_path: Path, 含 sources 与 outputs 的 YAML 配置.

    返回值:
        - manifest_path: Path, 写出的 manifest_root/manifest.json.

    目录产物:
        - manifest.json: object, 保存 coverage_threshold、pdb_ids 和 files.
        - stage3_occurrences.jsonl: 每条对应一个 PocketXMol 测试 occurrence.
        - identities.jsonl: 每条对应一个 PDB 内精确 SMILES 身份, 并标记是否允许作为 small-molecule 答案.
        - real_candidates.jsonl: 真实受体 Find_1 的全部 Stage3 测试 PDB 候选.
        - cryoatom2_candidates.jsonl: CryoAtom2 Find_1 的全部 Stage3 测试 PDB 候选.
        - emap2lig_candidates.jsonl: Emap2lig 的全部 Stage3 测试 PDB 候选.
        - README.md: 当前清单字段与跨文件编号关系.
    """
    config = load_config(config_path)
    sources = config["sources"]
    output_root = Path(config["outputs"]["manifest_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    stage3_records = read_jsonl(Path(sources["stage3_occurrences"]))
    pdb_ids = sorted({str(record["pdb_id"]).lower() for record in stage3_records})
    stage3_members = {
        (str(record["pdb_id"]).lower(), int(record["candidate_id"]))
        for record in stage3_records
    }

    data_root = Path(sources["data_root"])
    prepared_root = Path(sources["prepared_root"])
    occurrence_facts = {
        pdb_id: _read_occurrence_facts(data_root, prepared_root, pdb_id)
        for pdb_id in pdb_ids
    }
    real_manifest_path = Path(sources["real_manifest"])
    cryoatom2_manifest_path = Path(sources["cryoatom2_manifest"])
    real_candidates = _build_find_candidates(
        real_manifest_path,
        Path(sources["find_real_root"]),
        pdb_ids,
        stage3_members,
    )
    cryoatom2_candidates = _build_find_candidates(
        cryoatom2_manifest_path,
        Path(sources["find_cryoatom2_root"]),
        pdb_ids,
        stage3_members,
    )

    real_manifest = read_json(real_manifest_path)
    identity_path = real_manifest_path.parent / str(
        real_manifest["splits"]["held_out_test_0"]["identities"]
    )
    supported_identities = [
        record
        for record in read_jsonl(identity_path)
        if str(record["pdb_id"]) in set(pdb_ids)
    ]
    stage3_by_pdb: dict[str, set[int]] = defaultdict(set)
    for pdb_id, occurrence_id in stage3_members:
        stage3_by_pdb[pdb_id].add(occurrence_id)

    identities: list[dict[str, Any]] = []
    for identity in supported_identities:
        pdb_id = str(identity["pdb_id"])
        smiles = str(identity["smiles"])
        type_by_id, smiles_by_id = occurrence_facts[pdb_id]
        occurrence_ids = sorted(
            occurrence_id
            for occurrence_id, occurrence_smiles in smiles_by_id.items()
            if occurrence_smiles == smiles
        )
        type_tag = type_by_id[occurrence_ids[0]]
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

    emap2lig_candidates = _build_emap2lig_candidates(
        Path(sources["emap2lig_root"]),
        pdb_ids,
        stage3_members,
        occurrence_facts,
    )
    files = {
        "stage3_occurrences": "stage3_occurrences.jsonl",
        "identities": "identities.jsonl",
        "real_candidates": "real_candidates.jsonl",
        "cryoatom2_candidates": "cryoatom2_candidates.jsonl",
        "emap2lig_candidates": "emap2lig_candidates.jsonl",
        "readme": "README.md",
    }
    write_jsonl(output_root / files["stage3_occurrences"], stage3_records)
    write_jsonl(output_root / files["identities"], identities)
    write_jsonl(output_root / files["real_candidates"], real_candidates)
    write_jsonl(output_root / files["cryoatom2_candidates"], cryoatom2_candidates)
    write_jsonl(output_root / files["emap2lig_candidates"], emap2lig_candidates)
    shutil.copyfile(
        Path(__file__).with_name("MANIFEST_README.md"),
        output_root / files["readme"],
    )
    manifest_path = output_root / "manifest.json"
    write_json(
        manifest_path,
        {
            "coverage_threshold": COVERAGE_THRESHOLD,
            "pdb_ids": pdb_ids,
            "files": files,
        },
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    """解析配置路径并构建 Stage3 小分子测试清单."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(build_manifest(arguments.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
