"""扫描冻结 Stage1 F1 候选，并发布新 Matcher 的版本化监督清单。"""

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
from .locking import exclusive_file_lock
from .schema import (
    CLASS_NAMES,
    LANGUAGE_EMBEDDING_DIM,
    LANGUAGE_FAMILIES,
    LANGUAGE_MODEL_NAMES,
    MANIFEST_SCHEMA_VERSION,
    SourceRoots,
    safe_object_filename,
)


SPLIT_NAMES = ("train", "validation", "calibration")


def ensure_manifest(
    roots: SourceRoots,
    coverage_threshold: float,
    selection_seed: int,
) -> Path:
    """若正式清单不存在则在跨任务文件锁内构建，并返回主清单路径。"""

    manifest_path = roots.manifest_root / "manifest.json"
    with exclusive_file_lock(roots.manifest_root / ".build.lock"):
        if manifest_path.is_file():
            _validate_existing_manifest(manifest_path, roots, coverage_threshold, selection_seed)
        else:
            build_manifest(roots, coverage_threshold, selection_seed)
    return manifest_path


def build_manifest(
    roots: SourceRoots,
    coverage_threshold: float,
    selection_seed: int,
) -> Path:
    """构造所有候选、身份表示选择、排除原因和 seen/unseen 标记。"""

    if not 0.0 <= coverage_threshold <= 1.0:
        raise ValueError("coverage_threshold 必须位于 [0,1]。")
    roots.manifest_root.mkdir(parents=True, exist_ok=True)

    split_candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    split_identities: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    exclusions: list[dict[str, Any]] = []
    split_pdb_counts: dict[str, dict[str, int]] = {}

    for split_name in SPLIT_NAMES:
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

    train_object_keys = {
        identity["object_key"] for identity in split_identities["train"]
    }
    for split_name in SPLIT_NAMES:
        for identity in split_identities[split_name]:
            identity["seen_in_train"] = identity["object_key"] in train_object_keys
        for candidate in split_candidates[split_name]:
            object_key = candidate["matched_object_key"]
            candidate["identity_seen_in_train"] = (
                object_key in train_object_keys if object_key is not None else None
            )

    files: dict[str, dict[str, str]] = {}
    for split_name in SPLIT_NAMES:
        candidate_name = f"{split_name}_candidates.jsonl"
        identity_name = f"{split_name}_identities.jsonl"
        _atomic_write_jsonl(roots.manifest_root / candidate_name, split_candidates[split_name])
        _atomic_write_jsonl(roots.manifest_root / identity_name, split_identities[split_name])
        files[split_name] = {
            "candidates": candidate_name,
            "identities": identity_name,
        }
    _atomic_write_jsonl(roots.manifest_root / "excluded_pdb.jsonl", exclusions)

    summary = _summarize(split_candidates, split_identities, exclusions)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(roots.data_root),
        "stage1_root": str(roots.stage1_root),
        "output_root": str(roots.output_root),
        "coverage_threshold": coverage_threshold,
        "selection_seed": selection_seed,
        "language_selection_rule": (
            "对每个 (pdb_id, object_key, language_family) 的有效 occurrence 编号排序；"
            "用固定字节加权和与 selection_seed 决定循环起点，选择该位置的 occurrence。"
        ),
        "class_names": list(CLASS_NAMES),
        "splits": files,
        "excluded_pdb": "excluded_pdb.jsonl",
        "split_pdb_counts": split_pdb_counts,
        "summary": summary,
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
    """扫描一个完整 PDB 的候选标签和小分子身份表示。"""

    occurrence_records = _read_occurrences(roots.data_root, pdb_id)
    occurrence_masks = _read_occurrence_masks(roots.data_root, pdb_id, occurrence_records)
    identities = _build_identity_records(
        roots,
        split_name,
        pdb_id,
        occurrence_records,
        selection_seed,
    )
    identity_by_key = {identity["object_key"]: identity for identity in identities}

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
            identity = identity_by_key.get(target.matched_object_key)
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
    candidate_ids = [int(record["candidate_id"]) for record in records]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"{path} 含重复 candidate_id。")
    for record in records:
        if str(record.get("pdb_id", "")).lower() != pdb_id:
            raise ValueError(f"{path} 的 pdb_id 与目录名不一致。")
        for field in ("candidate_id", "object_key", "type_tag"):
            if field not in record:
                raise KeyError(f"{path} 缺少 occurrence 字段 {field}。")
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
    """按 object_key 去重小分子身份，并为每种语言模型固定选择一个有效 occurrence。"""

    grouped: dict[str, list[int]] = defaultdict(list)
    for record in occurrence_records:
        if record["type_tag"] == "small_molecule":
            grouped[str(record["object_key"])].append(int(record["candidate_id"]))

    identities: list[dict[str, Any]] = []
    for object_key in sorted(grouped):
        occurrence_ids = sorted(grouped[object_key])
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
        result["splits"][split_name] = {
            "candidate_count": len(candidates),
            "class_counts": dict(sorted(Counter(item["class_name"] for item in candidates).items())),
            "identity_count": len(identities),
            "unique_object_key_count": len({item["object_key"] for item in identities}),
            "language_available_identity_count": language_available,
            "language_multi_smiles_identity_count": language_smiles_conflicts,
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
) -> None:
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """构建或核验正式清单，并把最终路径写到标准输出。"""

    arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
    roots = SourceRoots(arguments.data_root, arguments.stage1_root, arguments.output_root)
    path = ensure_manifest(roots, arguments.coverage_threshold, arguments.selection_seed)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
