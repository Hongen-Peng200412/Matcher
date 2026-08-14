"""扫描冻结 Stage1 F1 候选，并发布新 Matcher 的版本化监督清单。

``manifest.json`` 固定源目录、覆盖阈值、语言表示选择种子、六分类顺序、三个
划分的文件名和摘要。每个 ``<split>_candidates.jsonl`` 对应一个 Stage1 候选，
记录完整组件覆盖率标签、匹配 occurrence、object_key 和语言监督可用性；每个
``<split>_identities.jsonl`` 对应一个 ``(pdb_id,class_name,object_key)`` 前景身份，记录
配体图与两种冻结语言表示的选择证据。``excluded_pdb.jsonl`` 单独记录整 PDB
排除原因，任何排除项都不进入训练或指标分母。所有文件位于
``<output_root>/manifests/closed_set_v<schema>/``，由覆盖阈值与选择种子共同决定身份。
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
    LANGUAGE_EMBEDDING_DIM,
    LANGUAGE_FAMILIES,
    LANGUAGE_MODEL_NAMES,
    MANIFEST_SCHEMA_VERSION,
    OCCURRENCE_TYPE_TO_CLASS,
    SourceRoots,
    safe_object_filename,
)


SPLIT_NAMES = ("train", "validation", "calibration")


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
    """按语言模型与 ``(class_name,object_key)`` 标记 train 可监督正身份。"""

    train_positive_keys = {
        family: {
            (candidate["class_name"], candidate["matched_object_key"])
            for candidate in split_candidates["train"]
            if candidate["class_name"] in FOREGROUND_CLASS_NAMES
            and candidate["matched_object_key"] is not None
            and candidate["language_target_available"][family]
        }
        for family in LANGUAGE_FAMILIES
    }
    for split_name in split_candidates:
        for identity in split_identities[split_name]:
            identity_key = (identity["class_name"], identity["object_key"])
            identity["seen_in_train_by_language"] = {
                family: identity_key in train_positive_keys[family]
                for family in LANGUAGE_FAMILIES
            }
        for candidate in split_candidates[split_name]:
            object_key = candidate["matched_object_key"]
            identity_key = (candidate["class_name"], object_key)
            candidate["identity_seen_in_train_by_language"] = {
                family: (
                    identity_key in train_positive_keys[family]
                    if object_key is not None
                    and candidate["class_name"] in FOREGROUND_CLASS_NAMES
                    else None
                )
                for family in LANGUAGE_FAMILIES
            }


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
        "language_selection_rule": (
            "对每个 (pdb_id, object_key, language_family) 的有效 occurrence 编号排序；"
            "用固定字节加权和与 selection_seed 决定循环起点，选择该位置的 occurrence。"
        ),
        "seen_identity_rule": (
            "按语言模型和 (class_name, object_key) 分别判断；只有该身份在 train 候选中"
            "作为可监督前景正目标出现过，才标记为 seen。"
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
    occurrence_masks = _read_occurrence_masks(roots.data_root, pdb_id, occurrence_records)
    identities = _build_identity_records(
        roots,
        split_name,
        pdb_id,
        occurrence_records,
        selection_seed,
    )
    identity_by_key = {
        (identity["class_name"], identity["object_key"]): identity
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
            identity = identity_by_key.get((target.class_name, target.matched_object_key))
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
    selection_seed: int,
) -> list[dict[str, Any]]:
    """按类别与 object_key 去重五类前景身份，并固定选择语言 occurrence。"""

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for record in occurrence_records:
        class_name = OCCURRENCE_TYPE_TO_CLASS[str(record["type_tag"])]
        if class_name in FOREGROUND_CLASS_NAMES:
            grouped[(class_name, str(record["object_key"]))].append(
                int(record["candidate_id"])
            )

    identities: list[dict[str, Any]] = []
    for class_name, object_key in sorted(grouped):
        occurrence_ids = sorted(grouped[(class_name, object_key)])
        graph_relative_path = f"ligand_objects/{safe_object_filename(object_key)}.npz"
        graph_available = (roots.data_root / graph_relative_path).is_file()
        language = {
            family: _select_language_representation(
                roots,
                pdb_id,
                object_key,
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
                "occurrence_candidate_ids": occurrence_ids,
                "graph_available": graph_available,
                "graph_relative_path": graph_relative_path,
                "language": language,
            }
        )
    return identities


def _select_language_representation(
    roots: SourceRoots,
    pdb_id: str,
    object_key: str,
    occurrence_ids: list[int],
    family: str,
    selection_seed: int,
) -> dict[str, Any]:
    """验证同身份的 occurrence 表示，并从有效集合确定性地选择一个。"""

    valid: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for candidate_id in occurrence_ids:
        relative_path = f"{family}/{pdb_id}/candidate_{candidate_id}.npz"
        path = roots.language_root / relative_path
        if not path.is_file():
            failures.append({"candidate_id": candidate_id, "reason": "missing_file"})
            continue
        try:
            metadata = _validate_language_archive(path, pdb_id, candidate_id, object_key, family)
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
    family: str,
) -> dict[str, Any]:
    """核验一个 occurrence 语言表示的身份、模型名称和数值形状。"""

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "pdb_id",
            "candidate_id",
            "object_key",
            "model_name",
            "model_smiles",
            "embedding",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"缺少字段 {missing}")
        actual = {
            "pdb_id": str(archive["pdb_id"].item()).lower(),
            "candidate_id": int(archive["candidate_id"].item()),
            "object_key": str(archive["object_key"].item()),
            "model_name": str(archive["model_name"].item()),
            "model_smiles": str(archive["model_smiles"].item()),
        }
        embedding = np.asarray(archive["embedding"], dtype=np.float32)
    expected = {
        "pdb_id": pdb_id,
        "candidate_id": candidate_id,
        "object_key": object_key,
        "model_name": LANGUAGE_MODEL_NAMES[family],
    }
    for field, expected_value in expected.items():
        if actual[field] != expected_value:
            raise ValueError(f"{field}={actual[field]!r}，期望 {expected_value!r}")
    if embedding.shape != (LANGUAGE_EMBEDDING_DIM,) or not np.isfinite(embedding).all():
        raise ValueError(f"embedding 必须为有限 ({LANGUAGE_EMBEDDING_DIM},)，实际为 {embedding.shape}")
    return {
        "candidate_id": candidate_id,
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
        language_smiles_conflicts = {
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
            "language_available_identity_count": language_available,
            "language_multi_smiles_identity_count": language_smiles_conflicts,
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
