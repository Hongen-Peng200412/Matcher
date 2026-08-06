"""生成双模式 Matcher 的正式样本清单与 Stage1 产物指针。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .data_common import read_jsonl


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """在目标目录内原子替换 JSON，避免训练读取半写文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def build_ground_truth_manifest(
    *, data_root: Path, split_files: dict[str, Path], output: Path,
    max_slots: int = 100, write: bool = True
) -> dict[str, Any]:
    """沿用给定 PDB 划分，只冻结实际可读 occurrence 身份。"""

    splits: dict[str, list[dict[str, Any]]] = {}
    excluded: list[dict[str, Any]] = []
    for split, path in split_files.items():
        pdb_ids = json.loads(path.read_text(encoding="utf-8"))
        entries = []
        for pdb_id in pdb_ids:
            occurrence_path = data_root / "parse" / pdb_id / "occurrences.jsonl"
            try:
                occurrence_ids = [int(row["candidate_id"]) for row in read_jsonl(occurrence_path)]
            except (FileNotFoundError, KeyError, ValueError) as error:
                excluded.append({"pdb_id": pdb_id, "split": split, "reason": str(error)})
                continue
            if not occurrence_ids or len(occurrence_ids) > max_slots:
                reason = "zero_occurrence" if not occurrence_ids else "over_max_slots"
                excluded.append({"pdb_id": pdb_id, "split": split, "reason": reason})
                continue
            entries.append(
                {
                    "pdb_id": pdb_id,
                    "occurrence_ids": occurrence_ids,
                    "num_occurrences": len(occurrence_ids),
                }
            )
        splits[split] = entries
    result = {
        "schema_version": 1,
        "mode": "ground_truth_context",
        "source_split_files": {name: path.as_posix() for name, path in split_files.items()},
        "max_slots": max_slots,
        "splits": splits,
        "excluded": excluded,
        "counts": {
            name: {
                "pdb": len(entries),
                "occurrence": sum(item["num_occurrences"] for item in entries),
            }
            for name, entries in splits.items()
        },
    }
    if write:
        _write_json(output, result)
    return result


def split_files_from_box_manifest(
    box_manifest: Path, calibration: Path, temporary_root: Path
) -> dict[str, Path]:
    """把新版 Stage1 BOX 划分转换为本次清单构造器读取的简单 PDB 列表。"""

    source = json.loads(box_manifest.read_text(encoding="utf-8"))
    result = {}
    for split in ("train", "validation"):
        path = temporary_root / f"{split}_pdb_ids.json"
        path.write_text(
            json.dumps([row["pdb_id"] for row in source["splits"][split]]) + "\n",
            encoding="utf-8",
        )
        result[split] = path
    result["calibration"] = calibration
    return result


def build_stage1_pointer(
    *, artifact_root: Path, data_root: Path, producer: str, splits: tuple[str, ...],
    role: str, output: Path, max_slots: int = 100
) -> dict[str, Any]:
    """扫描已有完整产物并写显式路径；Dataset 本身不扫描目录。"""

    records = []
    missing = []
    for split in splits:
        split_root = artifact_root / producer / split
        for pdb_root in sorted(path for path in split_root.glob("*") if path.is_dir()):
            complete = pdb_root / "status" / role / "_COMPLETE"
            centered = pdb_root / "centered" / f"{role}.npz"
            probability = pdb_root / "probability" / "probability_map.npz"
            geometry = pdb_root / "probability" / "geometry.json"
            if all(path.is_file() for path in (complete, centered, probability, geometry)):
                with np.load(centered, allow_pickle=False) as arrays:
                    num_candidates = int(len(arrays["centered_box_index"]))
                occurrence_path = data_root / "parse" / pdb_root.name / "occurrences.jsonl"
                num_occurrences = len(read_jsonl(occurrence_path)) if occurrence_path.is_file() else 0
                if not num_candidates or not num_occurrences or num_occurrences > max_slots:
                    missing.append(
                        {
                            "pdb_id": pdb_root.name,
                            "split": split,
                            "reason": (
                                "zero_candidate" if not num_candidates else
                                "zero_occurrence" if not num_occurrences else "over_max_slots"
                            ),
                        }
                    )
                    continue
                records.append(
                    {
                        "pdb_id": pdb_root.name,
                        "producer": producer,
                        "split": split,
                        "role": role,
                        "pdb_root": pdb_root.as_posix(),
                        "centered_path": centered.as_posix(),
                        "probability_path": probability.as_posix(),
                        "geometry_path": geometry.as_posix(),
                        "num_candidates": num_candidates,
                        "num_occurrences": num_occurrences,
                    }
                )
            else:
                missing.append({"pdb_id": pdb_root.name, "split": split})
    result = {
        "schema_version": 1,
        "mode": "stage1_context",
        "artifact_root": artifact_root.as_posix(),
        "producer": producer,
        "role": role,
        "records": records,
        "incomplete": missing,
    }
    _write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Matcher v2 正式数据清单。")
    subparsers = parser.add_subparsers(dest="command", required=True)
    mode1 = subparsers.add_parser("ground-truth", help="冻结模式一 PDB 与 occurrence。")
    mode1.add_argument("--data-root", type=Path, required=True)
    mode1.add_argument("--train", type=Path)
    mode1.add_argument("--validation", type=Path)
    mode1.add_argument("--box-manifest", type=Path)
    mode1.add_argument("--calibration", type=Path, required=True)
    mode1.add_argument("--output", type=Path, required=True)
    mode1.add_argument("--max-slots", type=int, default=100)
    mode2 = subparsers.add_parser("stage1", help="冻结已有 Stage1 产物路径。")
    mode2.add_argument("--artifact-root", type=Path, required=True)
    mode2.add_argument("--data-root", type=Path, required=True)
    mode2.add_argument("--producer", default="Find_0")
    mode2.add_argument("--splits", nargs="+", default=["train", "validation", "calibration"])
    mode2.add_argument("--role", default="F1_centered")
    mode2.add_argument("--output", type=Path, required=True)
    mode2.add_argument("--max-slots", type=int, default=100)
    args = parser.parse_args()
    if args.command == "ground-truth":
        if args.box_manifest:
            with tempfile.TemporaryDirectory(prefix="matcher_v2_split_") as directory:
                split_files = split_files_from_box_manifest(
                    args.box_manifest, args.calibration, Path(directory)
                )
                result = build_ground_truth_manifest(
                    data_root=args.data_root, split_files=split_files,
                    output=args.output, max_slots=args.max_slots, write=False,
                )
            result["source_box_manifest"] = args.box_manifest.as_posix()
            result["source_split_files"] = {
                "train": args.box_manifest.as_posix(),
                "validation": args.box_manifest.as_posix(),
                "calibration": args.calibration.as_posix(),
            }
            _write_json(args.output, result)
        else:
            if args.train is None or args.validation is None:
                parser.error("ground-truth 必须提供 --box-manifest，或同时提供 --train/--validation。")
            result = build_ground_truth_manifest(
                data_root=args.data_root,
                split_files={"train": args.train, "validation": args.validation, "calibration": args.calibration},
                output=args.output, max_slots=args.max_slots,
            )
    else:
        result = build_stage1_pointer(
            artifact_root=args.artifact_root, data_root=args.data_root,
            producer=args.producer,
            splits=tuple(args.splits),
            role=args.role,
            output=args.output,
            max_slots=args.max_slots,
        )
    summary = result["counts"] if "counts" in result else {"records": len(result["records"])}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
