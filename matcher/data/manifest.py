"""扫描冻结 Stage1 F1 候选，并发布新 Matcher 的版本化监督清单。

``manifest.json`` 固定源目录、覆盖阈值、语言表示选择种子、六分类顺序、三个
划分的文件名和摘要。每个 ``<split>_candidates.jsonl`` 对应一个 Stage1 候选，
记录完整组件覆盖率标签、匹配 occurrence、object_key、顶层 ``PreparedLigand.smiles``
和语言监督可用性；每个 ``<split>_identities.jsonl`` 对应一个 PDB 内
``(object_key, PreparedLigand.smiles)`` 前景身份，记录类别、配体图与两种冻结语言
表示的选择证据。``excluded_pdb.jsonl`` 单独记录整 PDB
排除原因，任何排除项都不进入训练或指标分母。所有文件位于
``<output_root>/manifests/closed_set_v<schema>/``。覆盖阈值决定候选是否命中 occurrence，
联合键定义身份槽，选择种子只决定同一身份采用哪个有效 occurrence 语言表示。
训练入口只请求 train/validation；calibration 准备必须由单独命令显式物化，评估入口不会补建。
``ensure_manifest`` 供训练构建或复用清单；``require_manifest`` 供只读保留评估要求既有清单。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .labels import OccurrenceMask, assign_candidate_target
from ..utils.locking import exclusive_file_lock
from .schema import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    FOREGROUND_CLASS_NAMES,
    IDENTITY_KEY_FIELDS,
    LANGUAGE_EMBEDDING_DIM,
    LANGUAGE_FAMILIES,
    LANGUAGE_MODEL_NAMES,
    MANIFEST_SCHEMA_VERSION,
    OCCURRENCE_TYPE_TO_CLASS,
    SourceRoots,
    safe_object_filename,
)


SPLIT_NAMES = ("train", "validation", "calibration")
IDENTITY_CLASS_RULE = (
    "同一 PDB 内的 (object_key, prepared_smiles) 只能映射到一个局部类别；"
    "不同 PDB 不共享身份轴，跨 PDB 类别差异只作诊断。"
)
SPLIT_PDB_RULE = (
    "同一小写 pdb_id 的候选记录不得同时发布到多个划分；"
    "仅出现在 excluded_pdb.jsonl 的整 PDB 排除记录不构成候选重叠。"
)


def ensure_manifest(
    roots: SourceRoots,
    coverage_threshold: float,
    selection_seed: int,
    required_splits: tuple[str, ...],
) -> Path:
    """在跨任务文件锁内构建或扩展显式请求的划分，并返回主清单路径。"""

    _validate_split_names(required_splits)
    manifest_path = roots.manifest_root / "manifest.json"
    with exclusive_file_lock(roots.manifest_root / ".build.lock"):
        if manifest_path.is_file():
            manifest = _validate_existing_manifest(
                manifest_path,
                roots,
                coverage_threshold,
                selection_seed,
            )
            missing_splits = tuple(
                split_name
                for split_name in required_splits
                if split_name not in manifest["splits"]
            )
            if missing_splits:
                _extend_manifest(
                    roots,
                    manifest,
                    coverage_threshold,
                    selection_seed,
                    missing_splits,
                )
        else:
            if "train" not in required_splits:
                raise FileNotFoundError(
                    "首次构建清单必须显式包含 train；不得从其它划分推导 seen 身份。"
                )
            build_manifest(roots, coverage_threshold, selection_seed, required_splits)
    return manifest_path


def require_manifest(
    roots: SourceRoots,
    coverage_threshold: float,
    selection_seed: int,
    required_splits: tuple[str, ...],
) -> Path:
    """只接受已存在、来源一致且含显式请求划分的清单，不产生任何写入。"""

    _validate_split_names(required_splits)
    manifest_path = roots.manifest_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"冻结清单不存在，禁止在只读评估入口重建：{manifest_path}")
    manifest = _validate_existing_manifest(
        manifest_path,
        roots,
        coverage_threshold,
        selection_seed,
    )
    missing_splits = sorted(set(required_splits).difference(manifest["splits"]))
    if missing_splits:
        raise FileNotFoundError(
            f"冻结清单尚未物化请求划分 {missing_splits}；只读入口不得补建。"
        )
    return manifest_path


def build_manifest(
    roots: SourceRoots,
    coverage_threshold: float,
    selection_seed: int,
    split_names: tuple[str, ...],
) -> Path:
    """首次构造显式划分的候选、身份表示选择、排除原因和 seen 标记。"""

    if not 0.0 <= coverage_threshold <= 1.0:
        raise ValueError("coverage_threshold 必须位于 [0,1]。")
    _validate_split_names(split_names)
    if "train" not in split_names:
        raise ValueError("首次构建清单必须包含 train。")
    roots.manifest_root.mkdir(parents=True, exist_ok=True)

    split_candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in split_names}
    split_identities: dict[str, list[dict[str, Any]]] = {name: [] for name in split_names}
    exclusions: list[dict[str, Any]] = []
    split_pdb_counts: dict[str, dict[str, int]] = {}

    for split_name in split_names:
        split_root = roots.stage1_root / split_name
        if not split_root.is_dir():
            raise FileNotFoundError(f"Stage1 划分目录不存在：{split_root}")
        pdb_directories = sorted(path for path in split_root.iterdir() if path.is_dir())
        included_pdb = 0
        for pdb_directory in pdb_directories:
            pdb_id = pdb_directory.name.lower()
            exclusion = _allowed_exclusion(roots, split_name, pdb_id, pdb_directory)
            if exclusion is not None:
                exclusions.append(exclusion)
                continue
            candidates, identities = _scan_pdb(
                roots=roots,
                split_name=split_name,
                pdb_id=pdb_id,
                coverage_threshold=coverage_threshold,
                selection_seed=selection_seed,
            )
            split_candidates[split_name].extend(candidates)
            split_identities[split_name].extend(identities)
            included_pdb += 1
        split_pdb_counts[split_name] = {
            "discovered": len(pdb_directories),
            "included": included_pdb,
            "excluded": len(pdb_directories) - included_pdb,
        }

    _validate_materialized_split_pdb_disjointness(split_candidates)
    _validate_joint_identity_classes(split_identities)
    _annotate_seen(split_candidates, split_identities)
    return _publish_manifest(
        roots=roots,
        coverage_threshold=coverage_threshold,
        selection_seed=selection_seed,
        split_candidates=split_candidates,
        split_identities=split_identities,
        exclusions=exclusions,
        split_pdb_counts=split_pdb_counts,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def _extend_manifest(
    roots: SourceRoots,
    manifest: dict[str, Any],
    coverage_threshold: float,
    selection_seed: int,
    missing_splits: tuple[str, ...],
) -> Path:
    """在锁内补建缺失划分，并保持既有划分候选与身份记录逐值不变。"""

    split_candidates = {
        split_name: load_jsonl(roots.manifest_root / files["candidates"])
        for split_name, files in manifest["splits"].items()
    }
    split_identities = {
        split_name: load_jsonl(roots.manifest_root / files["identities"])
        for split_name, files in manifest["splits"].items()
    }
    exclusions = load_jsonl(roots.manifest_root / manifest["excluded_pdb"])
    split_pdb_counts = dict(manifest["split_pdb_counts"])
    for split_name in missing_splits:
        split_candidates[split_name] = []
        split_identities[split_name] = []
        split_root = roots.stage1_root / split_name
        if not split_root.is_dir():
            raise FileNotFoundError(f"Stage1 划分目录不存在：{split_root}")
        pdb_directories = sorted(path for path in split_root.iterdir() if path.is_dir())
        included_pdb = 0
        for pdb_directory in pdb_directories:
            pdb_id = pdb_directory.name.lower()
            exclusion = _allowed_exclusion(roots, split_name, pdb_id, pdb_directory)
            if exclusion is not None:
                exclusions.append(exclusion)
                continue
            candidates, identities = _scan_pdb(
                roots=roots,
                split_name=split_name,
                pdb_id=pdb_id,
                coverage_threshold=coverage_threshold,
                selection_seed=selection_seed,
            )
            split_candidates[split_name].extend(candidates)
            split_identities[split_name].extend(identities)
            included_pdb += 1
        split_pdb_counts[split_name] = {
            "discovered": len(pdb_directories),
            "included": included_pdb,
            "excluded": len(pdb_directories) - included_pdb,
        }
    _validate_materialized_split_pdb_disjointness(split_candidates)
    _validate_joint_identity_classes(split_identities)
    _annotate_seen(split_candidates, split_identities)
    return _publish_manifest(
        roots=roots,
        coverage_threshold=coverage_threshold,
        selection_seed=selection_seed,
        split_candidates=split_candidates,
        split_identities=split_identities,
        exclusions=exclusions,
        split_pdb_counts=split_pdb_counts,
        created_at_utc=str(manifest["created_at_utc"]),
    )


def _annotate_seen(
    split_candidates: dict[str, list[dict[str, Any]]],
    split_identities: dict[str, list[dict[str, Any]]],
) -> None:
    """按语言模型与联合模板—语言身份标记 train 可监督正身份。"""

    train_positive_keys = {
        family: {
            (candidate["matched_object_key"], candidate["matched_prepared_smiles"])
            for candidate in split_candidates["train"]
            if candidate["class_name"] in FOREGROUND_CLASS_NAMES
            and candidate["matched_object_key"] is not None
            and candidate["matched_prepared_smiles"] is not None
            and candidate["language_target_available"][family]
        }
        for family in LANGUAGE_FAMILIES
    }
    for split_name in split_candidates:
        for identity in split_identities[split_name]:
            identity_key = (identity["object_key"], identity["prepared_smiles"])
            identity["seen_in_train_by_language"] = {
                family: identity_key in train_positive_keys[family]
                for family in LANGUAGE_FAMILIES
            }
        for candidate in split_candidates[split_name]:
            object_key = candidate["matched_object_key"]
            prepared_smiles = candidate["matched_prepared_smiles"]
            identity_key = (object_key, prepared_smiles)
            candidate["identity_seen_in_train_by_language"] = {
                family: (
                    identity_key in train_positive_keys[family]
                    if object_key is not None
                    and prepared_smiles is not None
                    and candidate["class_name"] in FOREGROUND_CLASS_NAMES
                    else None
                )
                for family in LANGUAGE_FAMILIES
            }


def _validate_materialized_split_pdb_disjointness(
    split_candidates: dict[str, list[dict[str, Any]]],
) -> None:
    """拒绝同一 PDB 的候选记录同时进入多个准备发布的划分。"""

    splits_by_pdb: dict[str, set[str]] = defaultdict(set)
    for split_name, candidates in split_candidates.items():
        for record_index, candidate in enumerate(candidates):
            pdb_id = candidate.get("pdb_id")
            if type(pdb_id) is not str or not pdb_id:
                raise ValueError(
                    f"{split_name}_candidates.jsonl[{record_index}].pdb_id 必须是非空字符串。"
                )
            splits_by_pdb[pdb_id.lower()].add(split_name)
    overlaps = {
        pdb_id: sorted(split_names_for_pdb)
        for pdb_id, split_names_for_pdb in sorted(splits_by_pdb.items())
        if len(split_names_for_pdb) > 1
    }
    if overlaps:
        raise ValueError(f"同一 PDB 的候选记录不得同时发布到多个划分：{overlaps}")


def _validate_joint_identity_classes(
    split_identities: dict[str, list[dict[str, Any]]],
) -> None:
    """要求每个 PDB 内的同一联合身份只映射到一个六分类类别。

    ``split_identities`` 包含本次准备发布的全部划分身份记录。函数按
    ``(pdb_id,object_key,prepared_smiles)`` 聚合；同一 PDB 的同一键出现多个
    ``(class_name,class_index)`` 时在写任何正式清单前失败。不同 PDB
    不共享身份轴，跨 PDB 的六分类差异由独立诊断记录而不阻断清单。
    """

    classes_by_identity: dict[tuple[str, str, str], set[tuple[str, int]]] = defaultdict(set)
    locations_by_identity: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for split_name, identities in split_identities.items():
        for record_index, identity in enumerate(identities):
            prefix = f"{split_name}_identities.jsonl[{record_index}]"
            pdb_id = identity.get("pdb_id")
            object_key = identity.get("object_key")
            prepared_smiles = identity.get("prepared_smiles")
            class_name = identity.get("class_name")
            class_index = identity.get("class_index")
            if type(pdb_id) is not str or not pdb_id:
                raise ValueError(f"{prefix}.pdb_id 必须是非空字符串。")
            if type(object_key) is not str or not object_key:
                raise ValueError(f"{prefix}.object_key 必须是非空字符串。")
            if type(prepared_smiles) is not str or not prepared_smiles.strip():
                raise ValueError(f"{prefix}.prepared_smiles 必须是非空字符串。")
            if class_name not in FOREGROUND_CLASS_NAMES:
                raise ValueError(f"{prefix}.class_name 必须属于五类非背景类别。")
            if type(class_index) is not int or class_index != CLASS_TO_INDEX[class_name]:
                raise ValueError(f"{prefix}.class_index 与 class_name 不一致。")
            key = (pdb_id, object_key, prepared_smiles)
            classes_by_identity[key].add((class_name, class_index))
            locations_by_identity[key].append(f"{split_name}/{pdb_id}")
    conflicts = {
        key: {
            "classes": sorted(classes),
            "locations": sorted(locations_by_identity[key]),
        }
        for key, classes in classes_by_identity.items()
        if len(classes) > 1
    }
    if conflicts:
        raise ValueError(f"同一 PDB 内的联合身份类别不一致：{conflicts}")


def _joint_identity_cross_pdb_class_variants(
    split_identities: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """列出同一联合键在不同 PDB 中采用多个局部六分类类别的诊断记录。"""

    classes_by_identity: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    pdb_ids_by_identity: dict[tuple[str, str], set[str]] = defaultdict(set)
    locations_by_identity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for split_name, identities in split_identities.items():
        for identity in identities:
            key = (identity["object_key"], identity["prepared_smiles"])
            classes_by_identity[key].add((identity["class_name"], identity["class_index"]))
            pdb_ids_by_identity[key].add(identity["pdb_id"])
            locations_by_identity[key].append(
                {
                    "split": split_name,
                    "pdb_id": identity["pdb_id"],
                    "class_name": identity["class_name"],
                    "class_index": identity["class_index"],
                    "occurrence_count": len(identity["occurrence_candidate_ids"]),
                }
            )
    return [
        {
            "object_key": key[0],
            "prepared_smiles": key[1],
            "classes": [
                {"class_name": class_name, "class_index": class_index}
                for class_name, class_index in sorted(classes_by_identity[key])
            ],
            "locations": sorted(
                locations_by_identity[key],
                key=lambda item: (item["split"], item["pdb_id"]),
            ),
        }
        for key in sorted(classes_by_identity)
        if len(classes_by_identity[key]) > 1 and len(pdb_ids_by_identity[key]) > 1
    ]


def _publish_manifest(
    roots: SourceRoots,
    coverage_threshold: float,
    selection_seed: int,
    split_candidates: dict[str, list[dict[str, Any]]],
    split_identities: dict[str, list[dict[str, Any]]],
    exclusions: list[dict[str, Any]],
    split_pdb_counts: dict[str, dict[str, int]],
    created_at_utc: str,
) -> Path:
    """原子发布当前已物化划分的 JSONL、汇总和主清单。"""

    files: dict[str, dict[str, str]] = {}
    for split_name in SPLIT_NAMES:
        if split_name not in split_candidates:
            continue
        candidate_name = f"{split_name}_candidates.jsonl"
        identity_name = f"{split_name}_identities.jsonl"
        _atomic_write_jsonl(roots.manifest_root / candidate_name, split_candidates[split_name])
        _atomic_write_jsonl(roots.manifest_root / identity_name, split_identities[split_name])
        files[split_name] = {"candidates": candidate_name, "identities": identity_name}
    _atomic_write_jsonl(roots.manifest_root / "excluded_pdb.jsonl", exclusions)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(roots.data_root),
        "stage1_root": str(roots.stage1_root),
        "output_root": str(roots.output_root),
        "coverage_threshold": coverage_threshold,
        "selection_seed": selection_seed,
        "identity_key_fields": list(IDENTITY_KEY_FIELDS),
        "prepared_smiles_rule": (
            "prepared_smiles 逐字采用 PreparedLigand 顶层 smiles；显式 null 不建立身份；"
            "PreparedLigand 文件缺失单独标记，已有文件不完整时失败。"
        ),
        "identity_class_rule": IDENTITY_CLASS_RULE,
        "split_pdb_rule": SPLIT_PDB_RULE,
        "cross_pdb_joint_identity_class_variants": (
            _joint_identity_cross_pdb_class_variants(split_identities)
        ),
        "language_selection_rule": (
            "对每个 (pdb_id, object_key, prepared_smiles, language_family) 的有效 occurrence "
            "编号排序；沿用 v3 的 (pdb_id, object_key, language_family) 固定字节加权和与 "
            "selection_seed 决定循环起点，选择该位置的 occurrence。"
        ),
        "seen_identity_rule": (
            "按语言模型和 (object_key, prepared_smiles) 分别判断；只有该身份在 train "
            "候选中作为可监督前景正目标出现过，才标记为 seen。"
        ),
        "class_names": list(CLASS_NAMES),
        "available_splits": [name for name in SPLIT_NAMES if name in files],
        "deferred_splits": [name for name in SPLIT_NAMES if name not in files],
        "splits": files,
        "excluded_pdb": "excluded_pdb.jsonl",
        "split_pdb_counts": split_pdb_counts,
        "summary": _summarize(split_candidates, split_identities, exclusions),
    }
    manifest_path = roots.manifest_root / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    return manifest_path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取一个由 JSON 对象组成的 UTF-8 JSONL 文件。"""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON 对象。")
            records.append(value)
    return records


def read_npz_string_scalar(value: Any, field_name: str) -> str:
    """读取 NPZ 中的非空 Unicode 标量，拒绝数值、bytes 和非标量数组。

    ``value`` 是 ``numpy.load`` 返回的单字段数组；``field_name`` 是错误上下文。
    返回原始 Python 字符串，不执行 strip、规范化或其它隐式类型转换。
    """

    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "U":
        raise ValueError(f"{field_name} 必须是非空 Unicode 标量字符串。")
    result = array.item()
    if type(result) is not str or not result:
        raise ValueError(f"{field_name} 必须是非空 Unicode 标量字符串。")
    return result


def _allowed_exclusion(
    roots: SourceRoots,
    split_name: str,
    pdb_id: str,
    pdb_directory: Path,
) -> dict[str, Any] | None:
    """识别用户允许整 PDB 跳过的 blob_exceed 或缺失 F1 产物。"""

    if (pdb_directory / "_BLOB_EXCEED").exists():
        return {"split": split_name, "pdb_id": pdb_id, "reason": "blob_exceed"}
    required_paths = _required_pdb_paths(roots, split_name, pdb_id)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        return {
            "split": split_name,
            "pdb_id": pdb_id,
            "reason": "missing_required_product",
            "missing_paths": missing,
        }
    return None


def _required_pdb_paths(roots: SourceRoots, split_name: str, pdb_id: str) -> tuple[Path, ...]:
    """列出训练一个候选所需的 Stage1 与 A–G 文件。"""

    stage1_pdb = roots.stage1_root / split_name / pdb_id
    return (
        stage1_pdb / "centered/F1_centered.npz",
        stage1_pdb / "probability/probability_map.npz",
        roots.data_root / "parse" / pdb_id / "occurrences.jsonl",
        roots.data_root / "parse" / pdb_id / "receptor_tokens.npz",
        roots.data_root / "density" / pdb_id / "exp.npz",
        roots.data_root / "density" / pdb_id / "sim.npz",
        roots.data_root / "density" / pdb_id / "ligand_area.npz",
        roots.data_root / "labels" / pdb_id / "atom_labels.npz",
    )


def _scan_pdb(
    roots: SourceRoots,
    split_name: str,
    pdb_id: str,
    coverage_threshold: float,
    selection_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """扫描一个完整 PDB 的候选标签和五类非背景身份表示。"""

    occurrence_records = _read_occurrences(roots.data_root, pdb_id)
    prepared_smiles_by_candidate, prepared_smiles_file_available = (
        _read_prepared_smiles_by_candidate(
            roots,
            pdb_id,
            occurrence_records,
        )
    )
    occurrence_masks = _read_occurrence_masks(roots.data_root, pdb_id, occurrence_records)
    identities = _build_identity_records(
        roots,
        split_name,
        pdb_id,
        occurrence_records,
        prepared_smiles_by_candidate,
        selection_seed,
    )
    identity_by_key = {
        (identity["object_key"], identity["prepared_smiles"]): identity
        for identity in identities
    }

    centered_path = roots.stage1_root / split_name / pdb_id / "centered/F1_centered.npz"
    candidates: list[dict[str, Any]] = []
    with np.load(centered_path, allow_pickle=False) as centered:
        candidate_count = _validate_centered_offsets(centered, centered_path)
        for candidate_index in range(candidate_count):
            voxel_start = int(centered["voxel_offsets"][candidate_index])
            voxel_stop = int(centered["voxel_offsets"][candidate_index + 1])
            local_voxels = np.asarray(
                centered["voxel_index_local_zyx"][voxel_start:voxel_stop],
                dtype=np.int64,
            )
            box_start = np.asarray(centered["box_start_zyx"][candidate_index], dtype=np.int64)
            global_voxels = local_voxels + box_start[None, :]
            target = assign_candidate_target(global_voxels, occurrence_masks, coverage_threshold)
            matched_prepared_smiles: str | None = None
            if (
                target.class_name not in FOREGROUND_CLASS_NAMES
                or target.matched_candidate_id is None
            ):
                matched_prepared_smiles_status = "not_applicable"
            elif not prepared_smiles_file_available:
                matched_prepared_smiles_status = "missing_file"
            else:
                matched_prepared_smiles = prepared_smiles_by_candidate[
                    target.matched_candidate_id
                ]
                matched_prepared_smiles_status = (
                    "available" if matched_prepared_smiles is not None else "explicit_null"
                )
            identity = identity_by_key.get(
                (target.matched_object_key, matched_prepared_smiles)
            )
            if identity is not None and identity["class_name"] != target.class_name:
                raise ValueError(
                    f"{split_name}/{pdb_id} 的联合身份类别 {identity['class_name']!r} "
                    f"与候选目标类别 {target.class_name!r} 不一致。"
                )
            language_available = {
                family: bool(
                    identity
                    and identity["graph_available"]
                    and identity["language"][family]["available"]
                )
                for family in LANGUAGE_FAMILIES
            }
            candidates.append(
                {
                    "split": split_name,
                    "pdb_id": pdb_id,
                    "candidate_index": candidate_index,
                    "source_tree_id": int(centered["source_tree_id"][candidate_index]),
                    "source_node_id": int(centered["source_node_id"][candidate_index]),
                    "source_threshold_grid_index": int(
                        centered["source_threshold_grid_index"][candidate_index]
                    ),
                    "full_candidate_voxel_count": int(global_voxels.shape[0]),
                    "class_name": target.class_name,
                    "class_index": target.class_index,
                    "matched_candidate_id": target.matched_candidate_id,
                    "matched_object_key": target.matched_object_key,
                    "matched_prepared_smiles": matched_prepared_smiles,
                    "matched_prepared_smiles_status": matched_prepared_smiles_status,
                    "candidate_coverage": target.candidate_coverage,
                    "occurrence_coverage": target.occurrence_coverage,
                    "overlap_score": target.overlap_score,
                    "language_target_available": language_available,
                }
            )
    return candidates, identities


def _read_occurrences(data_root: Path, pdb_id: str) -> list[dict[str, Any]]:
    """读取并验证一个 PDB 的 A–G occurrence 主表。"""

    path = data_root / "parse" / pdb_id / "occurrences.jsonl"
    records = load_jsonl(path)
    for record in records:
        for field in ("candidate_id", "object_key", "type_tag"):
            if field not in record:
                raise KeyError(f"{path} 缺少 occurrence 字段 {field}。")
        if str(record.get("pdb_id", "")).lower() != pdb_id:
            raise ValueError(f"{path} 的 pdb_id 与目录名不一致。")
        try:
            int(record["candidate_id"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path} 的 candidate_id 必须可转换为整数。") from error
        if not isinstance(record["object_key"], str) or not record["object_key"]:
            raise ValueError(f"{path} 的 object_key 必须为非空字符串。")
        if str(record["type_tag"]) not in {"ion", "nucleotide_like", "peptide_like", "small_molecule", "sugar", "other"}:
            raise ValueError(f"{path} 含不支持的 type_tag={record['type_tag']!r}。")
    candidate_ids = [int(record["candidate_id"]) for record in records]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"{path} 含重复 candidate_id。")
    return sorted(records, key=lambda record: int(record["candidate_id"]))


def _read_prepared_smiles_by_candidate(
    roots: SourceRoots,
    pdb_id: str,
    occurrence_records: Iterable[dict[str, Any]],
) -> tuple[dict[int, str | None], bool]:
    """读取一个 PDB 的顶层 PreparedLigand.smiles。

    ``occurrence_records`` 是已核验的完整 occurrence 表。返回值第一项把每个五类前景
    ``candidate_id`` 映射到逐字读取的非空字符串或显式 None；第二项表示
    ``prepared_smiles.jsonl`` 是否存在。整个文件缺失时返回 ``({},False)``，已有文件的
    字段、身份或记录集合不完整时失败关闭。
    """

    occurrences = {int(record["candidate_id"]): record for record in occurrence_records}
    formal_candidate_ids = {
        candidate_id
        for candidate_id, record in occurrences.items()
        if OCCURRENCE_TYPE_TO_CLASS[str(record["type_tag"])] in FOREGROUND_CLASS_NAMES
    }
    path = roots.language_root / "prepared" / pdb_id / "prepared_smiles.jsonl"
    if not path.is_file():
        return {}, False

    required_fields = {
        "pdb_id",
        "candidate_id",
        "object_key",
        "kind",
        "type_tag",
        "is_covalent",
        "smiles",
        "smiles_source",
        "diagnostics",
    }
    prepared_smiles: dict[int, str | None] = {}
    for record in load_jsonl(path):
        missing = sorted(required_fields.difference(record))
        extra = sorted(set(record).difference(required_fields))
        if missing or extra:
            raise ValueError(f"{path} 的 PreparedLigand 字段漂移：missing={missing}, extra={extra}。")
        if type(record["candidate_id"]) is not int or int(record["candidate_id"]) < 0:
            raise ValueError(f"{path}:candidate_id 必须是非负整数。")
        candidate_id = int(record["candidate_id"])
        if candidate_id in prepared_smiles:
            raise ValueError(f"{path} 含重复 candidate_id={candidate_id}。")
        if candidate_id not in formal_candidate_ids:
            raise ValueError(f"{path} 含非五类前景 occurrence candidate_id={candidate_id}。")
        occurrence = occurrences[candidate_id]
        expected = {
            "pdb_id": pdb_id,
            "object_key": str(occurrence["object_key"]),
            "type_tag": str(occurrence["type_tag"]),
        }
        for field, expected_value in expected.items():
            if record[field] != expected_value:
                raise ValueError(
                    f"{path}:candidate_id={candidate_id} 的 {field}={record[field]!r}，"
                    f"期望 {expected_value!r}。"
                )
        if type(record["kind"]) is not str or not record["kind"]:
            raise ValueError(f"{path}:kind 必须是非空字符串。")
        if type(record["is_covalent"]) is not bool:
            raise ValueError(f"{path}:is_covalent 必须是布尔值。")
        smiles = record["smiles"]
        if smiles is not None and (type(smiles) is not str or not smiles.strip()):
            raise ValueError(f"{path}:smiles 必须是非空字符串或 null。")
        if record["smiles_source"] not in {
            "ccd",
            "clc",
            "present_graph",
            "unsupported",
            "occurrence_error",
        }:
            raise ValueError(f"{path}:smiles_source 不属于 PreparedLigand 枚举。")
        if type(record["diagnostics"]) is not dict:
            raise ValueError(f"{path}:diagnostics 必须是 JSON 对象。")
        prepared_smiles[candidate_id] = smiles

    missing_candidate_ids = sorted(formal_candidate_ids.difference(prepared_smiles))
    if missing_candidate_ids:
        raise ValueError(f"{path} 缺少五类前景 candidate_id={missing_candidate_ids}。")
    return prepared_smiles, True


def _read_occurrence_masks(
    data_root: Path,
    pdb_id: str,
    occurrence_records: Iterable[dict[str, Any]],
) -> list[OccurrenceMask]:
    """只解压每个 occurrence 的稀疏真值掩码，不读取巨大 union_mask。"""

    path = data_root / "density" / pdb_id / "ligand_area.npz"
    result: list[OccurrenceMask] = []
    with np.load(path, allow_pickle=False) as archive:
        for record in occurrence_records:
            candidate_id = int(record["candidate_id"])
            key = f"mask_{candidate_id}"
            if key not in archive.files:
                raise KeyError(f"{path} 缺少 {key}。")
            mask = np.asarray(archive[key], dtype=np.int64)
            result.append(
                OccurrenceMask(
                    candidate_id=candidate_id,
                    object_key=str(record["object_key"]),
                    type_tag=str(record["type_tag"]),
                    voxel_index_zyx=mask,
                )
            )
    return result


def _build_identity_records(
    roots: SourceRoots,
    split_name: str,
    pdb_id: str,
    occurrence_records: Iterable[dict[str, Any]],
    prepared_smiles_by_candidate: dict[int, str | None],
    selection_seed: int,
) -> list[dict[str, Any]]:
    """按 object_key 与顶层 PreparedLigand.smiles 构造 PDB 内前景身份。

    显式 null 或来源文件缺失的 occurrence 不建立身份。返回记录按
    ``(class_index,object_key,prepared_smiles)`` 排序；相同 object_key 的不同字符串
    保持独立槽，相同字符串的不同 object_key 也不合并。
    """

    grouped: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for record in occurrence_records:
        class_name = OCCURRENCE_TYPE_TO_CLASS[str(record["type_tag"])]
        candidate_id = int(record["candidate_id"])
        prepared_smiles = prepared_smiles_by_candidate.get(candidate_id)
        if class_name in FOREGROUND_CLASS_NAMES and prepared_smiles is not None:
            grouped[(str(record["object_key"]), prepared_smiles)].append(
                (candidate_id, class_name)
            )

    identities: list[dict[str, Any]] = []
    for object_key, prepared_smiles in sorted(grouped):
        grouped_occurrences = grouped[(object_key, prepared_smiles)]
        class_names = {class_name for _, class_name in grouped_occurrences}
        if len(class_names) != 1:
            raise ValueError(
                f"{pdb_id} 的联合身份 ({object_key!r}, {prepared_smiles!r}) "
                f"对应多个六分类类别：{sorted(class_names)}。"
            )
        class_name = next(iter(class_names))
        occurrence_ids = sorted(candidate_id for candidate_id, _ in grouped_occurrences)
        graph_relative_path = f"ligand_objects/{safe_object_filename(object_key)}.npz"
        graph_available = (roots.data_root / graph_relative_path).is_file()
        language = {
            family: _select_language_representation(
                roots,
                pdb_id,
                object_key,
                prepared_smiles,
                occurrence_ids,
                family,
                selection_seed,
            )
            for family in LANGUAGE_FAMILIES
        }
        identities.append(
            {
                "split": split_name,
                "pdb_id": pdb_id,
                "class_name": class_name,
                "class_index": CLASS_TO_INDEX[class_name],
                "object_key": object_key,
                "prepared_smiles": prepared_smiles,
                "occurrence_candidate_ids": occurrence_ids,
                "graph_available": graph_available,
                "graph_relative_path": graph_relative_path,
                "language": language,
            }
        )
    return sorted(
        identities,
        key=lambda item: (
            int(item["class_index"]),
            str(item["object_key"]),
            str(item["prepared_smiles"]),
        ),
    )


def _select_language_representation(
    roots: SourceRoots,
    pdb_id: str,
    object_key: str,
    prepared_smiles: str,
    occurrence_ids: list[int],
    family: str,
    selection_seed: int,
) -> dict[str, Any]:
    """验证联合身份内的 occurrence 表示，并从有效集合确定性地选择一个。

    有效归档必须同时匹配 ``pdb_id``、``candidate_id``、``object_key``、
    ``prepared_smiles`` 与语言模型名称。选择继续使用 v3 的
    ``pdb_id|object_key|family`` 盐值，使没有拆分的身份保持原 occurrence。
    """

    valid: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for candidate_id in occurrence_ids:
        relative_path = f"{family}/{pdb_id}/candidate_{candidate_id}.npz"
        path = roots.language_root / relative_path
        if not path.is_file():
            failures.append({"candidate_id": candidate_id, "reason": "missing_file"})
            continue
        try:
            metadata = _validate_language_archive(
                path,
                pdb_id,
                candidate_id,
                object_key,
                prepared_smiles,
                family,
            )
        except (KeyError, OSError, ValueError) as error:
            failures.append(
                {
                    "candidate_id": candidate_id,
                    "reason": "invalid_archive",
                    "detail": str(error),
                }
            )
            continue
        metadata["relative_path"] = relative_path
        valid.append(metadata)

    if not valid:
        return {
            "available": False,
            "selected_candidate_id": None,
            "selected_relative_path": None,
            "valid_candidate_ids": [],
            "failed_occurrences": failures,
            "distinct_model_smiles": [],
        }

    selector_text = f"{pdb_id}|{object_key}|{family}"
    weighted_bytes = sum(
        (position + 1) * byte for position, byte in enumerate(selector_text.encode("utf-8"))
    )
    selected = valid[(selection_seed + weighted_bytes) % len(valid)]
    return {
        "available": True,
        "selected_candidate_id": selected["candidate_id"],
        "selected_relative_path": selected["relative_path"],
        "valid_candidate_ids": [item["candidate_id"] for item in valid],
        "failed_occurrences": failures,
        "distinct_model_smiles": sorted({item["model_smiles"] for item in valid}),
    }


def _validate_language_archive(
    path: Path,
    pdb_id: str,
    candidate_id: int,
    object_key: str,
    prepared_smiles: str,
    family: str,
) -> dict[str, Any]:
    """核验 occurrence 归档的联合身份、模型名称和有限 768 维向量。

    ``prepared_smiles`` 必须逐字等于联合身份字符串；``model_smiles`` 只返回给诊断汇总，
    不参与身份定义或选择。
    """

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "pdb_id",
            "candidate_id",
            "object_key",
            "model_name",
            "prepared_smiles",
            "model_smiles",
            "embedding",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"缺少字段 {missing}")
        actual = {
            "pdb_id": read_npz_string_scalar(archive["pdb_id"], f"{path}:pdb_id").lower(),
            "candidate_id": int(archive["candidate_id"].item()),
            "object_key": read_npz_string_scalar(
                archive["object_key"],
                f"{path}:object_key",
            ),
            "model_name": read_npz_string_scalar(
                archive["model_name"],
                f"{path}:model_name",
            ),
            "prepared_smiles": read_npz_string_scalar(
                archive["prepared_smiles"],
                f"{path}:prepared_smiles",
            ),
            "model_smiles": read_npz_string_scalar(
                archive["model_smiles"],
                f"{path}:model_smiles",
            ),
        }
        embedding = np.asarray(archive["embedding"], dtype=np.float32)
    expected = {
        "pdb_id": pdb_id,
        "candidate_id": candidate_id,
        "object_key": object_key,
        "model_name": LANGUAGE_MODEL_NAMES[family],
        "prepared_smiles": prepared_smiles,
    }
    for field, expected_value in expected.items():
        if actual[field] != expected_value:
            raise ValueError(f"{field}={actual[field]!r}，期望 {expected_value!r}")
    if embedding.shape != (LANGUAGE_EMBEDDING_DIM,) or not np.isfinite(embedding).all():
        raise ValueError(f"embedding 必须为有限 ({LANGUAGE_EMBEDDING_DIM},)，实际为 {embedding.shape}")
    return {
        "candidate_id": candidate_id,
        "prepared_smiles": actual["prepared_smiles"],
        "model_smiles": actual["model_smiles"],
    }


def _validate_centered_offsets(archive: Any, path: Path) -> int:
    """核验 F1-centered 的候选级和稀疏 token 偏移字段。"""

    required = {
        "centered_box_index",
        "box_start_zyx",
        "source_tree_id",
        "source_node_id",
        "source_threshold_grid_index",
        "voxel_offsets",
        "voxel_index_local_zyx",
    }
    missing = sorted(required.difference(archive.files))
    if missing:
        raise KeyError(f"{path} 缺少字段 {missing}。")
    candidate_count = int(archive["centered_box_index"].shape[0])
    for key in (
        "box_start_zyx",
        "source_tree_id",
        "source_node_id",
        "source_threshold_grid_index",
    ):
        if archive[key].shape[0] != candidate_count:
            raise ValueError(f"{path}:{key} 第一维与候选数量不一致。")
    offsets = np.asarray(archive["voxel_offsets"], dtype=np.int64)
    if (
        offsets.shape != (candidate_count + 1,)
        or offsets[0] != 0
        or np.any(offsets[1:] < offsets[:-1])
        or offsets[-1] != archive["voxel_index_local_zyx"].shape[0]
    ):
        raise ValueError(f"{path}:voxel_offsets 非法。")
    return candidate_count


def _summarize(
    split_candidates: dict[str, list[dict[str, Any]]],
    split_identities: dict[str, list[dict[str, Any]]],
    exclusions: list[dict[str, Any]],
) -> dict[str, Any]:
    """汇总类别、身份、语言缺失和排除原因计数。"""

    result: dict[str, Any] = {"splits": {}}
    for split_name in SPLIT_NAMES:
        if split_name not in split_candidates:
            continue
        candidates = split_candidates[split_name]
        identities = split_identities[split_name]
        language_available = {
            family: sum(identity["language"][family]["available"] for identity in identities)
            for family in LANGUAGE_FAMILIES
        }
        language_model_smiles_conflicts = {
            family: sum(
                len(identity["language"][family]["distinct_model_smiles"]) > 1
                for identity in identities
            )
            for family in LANGUAGE_FAMILIES
        }
        seen_positive_identities = {
            family: sum(
                identity["seen_in_train_by_language"][family]
                for identity in identities
            )
            for family in LANGUAGE_FAMILIES
        }
        result["splits"][split_name] = {
            "candidate_count": len(candidates),
            "class_counts": dict(sorted(Counter(item["class_name"] for item in candidates).items())),
            "identity_count": len(identities),
            "unique_object_key_count": len({item["object_key"] for item in identities}),
            "same_pdb_object_key_multi_prepared_smiles_group_count": sum(
                count > 1
                for count in Counter(
                    (item["pdb_id"], item["object_key"]) for item in identities
                ).values()
            ),
            "matched_foreground_without_prepared_smiles_candidate_count": sum(
                item["class_name"] in FOREGROUND_CLASS_NAMES
                and item["matched_prepared_smiles"] is None
                for item in candidates
            ),
            "matched_foreground_explicit_null_candidate_count": sum(
                item["matched_prepared_smiles_status"] == "explicit_null"
                for item in candidates
            ),
            "matched_foreground_missing_prepared_file_candidate_count": sum(
                item["matched_prepared_smiles_status"] == "missing_file"
                for item in candidates
            ),
            "language_available_identity_count": language_available,
            "language_multi_model_smiles_identity_count": language_model_smiles_conflicts,
            "seen_positive_identity_count": seen_positive_identities,
        }
    result["excluded_pdb_reason_counts"] = dict(
        sorted(Counter(item["reason"] for item in exclusions).items())
    )
    return result


def _validate_existing_manifest(
    path: Path,
    roots: SourceRoots,
    coverage_threshold: float,
    selection_seed: int,
) -> dict[str, Any]:
    """拒绝复用来源根或监督参数不一致的既有正式清单。"""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "data_root": str(roots.data_root),
        "stage1_root": str(roots.stage1_root),
        "output_root": str(roots.output_root),
        "coverage_threshold": coverage_threshold,
        "selection_seed": selection_seed,
        "identity_key_fields": list(IDENTITY_KEY_FIELDS),
        "identity_class_rule": IDENTITY_CLASS_RULE,
        "split_pdb_rule": SPLIT_PDB_RULE,
        "class_names": list(CLASS_NAMES),
    }
    mismatches = {
        key: {"actual": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"既有清单与当前请求不一致：{mismatches}")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not all(
        split_name in SPLIT_NAMES and isinstance(files, dict)
        for split_name, files in splits.items()
    ):
        raise ValueError(f"{path}:splits 必须只含受支持划分的文件映射。")
    return manifest


def _validate_split_names(split_names: tuple[str, ...]) -> None:
    """要求显式划分序列非空、不重复且只含 train/validation/calibration。"""

    if (
        not split_names
        or len(set(split_names)) != len(split_names)
        or not set(split_names).issubset(SPLIT_NAMES)
    ):
        raise ValueError(
            f"split_names 必须是非空、不重复的 {SPLIT_NAMES} 子序列。"
        )


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """在目标目录写临时文件并原子替换 JSON。"""

    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, text)


def _atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """在目标目录写临时文件并原子替换 JSONL。"""

    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    _atomic_write_text(path, text)


def _atomic_write_text(path: Path, text: str) -> None:
    """使用同目录唯一临时文件避免读者观察到部分内容。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    """解析正式清单命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--coverage-threshold", type=float, required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLIT_NAMES,
        required=True,
        help="显式列出本次允许读取并物化的划分；不得预先加入 calibration。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """构建或核验正式清单，并把最终路径写到标准输出。"""

    arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
    roots = SourceRoots(arguments.data_root, arguments.stage1_root, arguments.output_root)
    path = ensure_manifest(
        roots,
        arguments.coverage_threshold,
        arguments.selection_seed,
        tuple(arguments.splits),
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
