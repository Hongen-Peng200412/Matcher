"""一次性生成 Matcher Anchor 正式实验清单。

该入口把 max-slots 排除和固定 validation synthetic-anchor 关系固化到正式 JSON。
长期 Dataset 只读该清单，不在每次训练时重新携带历史过滤逻辑。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .anchor_data import AnchorDataConfig, sample_candidate_starts


MISSING_BOX_PDB = {"5ocu": 7, "6tql": 6, "6wcb": 7, "7sar": 40}


def _resolve_box_path(preparation_root: Path, manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    direct = preparation_root / path
    return direct if direct.exists() else manifest_path.parent / path


def _relative_box_path(preparation_root: Path, path: Path) -> str:
    return path.resolve().relative_to(preparation_root.resolve()).as_posix()


def _choose_fraction(
    entries: list[dict[str, Any]], fraction: float, rng: np.random.Generator
) -> list[dict[str, Any]]:
    if fraction >= 1.0:
        return entries
    count = math.floor(fraction * len(entries))
    chosen = set(rng.choice(len(entries), size=count, replace=False).tolist())
    return [entry for index, entry in enumerate(entries) if index in chosen]


def build_anchor_manifest(
    *,
    data_root: Path,
    preparation_root: Path,
    source_manifest: Path,
    output_path: Path,
    seed: int = 3407,
    max_slots_per_pdb: int = 100,
    pdb_sample_fraction: float = 1.0,
    p_miss: float = 0.30,
    p_split: float = 0.20,
    p_hit: float = 0.50,
    max_context_ratio: float = 2.0,
    empty_A_context_skip_probability: float = 0.80,
    receptor_radius: float = 18.0,
    source_box_config: Path | None = None,
) -> dict[str, Any]:
    """生成基础 train 与冻结 validation 清单，并原子写入 ``output_path``。"""

    with source_manifest.open("r", encoding="utf-8") as handle:
        source = json.load(handle)

    base: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    oversized: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for split in ("train", "validation"):
        for source_entry in source["splits"][split]:
            box_path = _resolve_box_path(
                preparation_root, source_manifest, source_entry["path"]
            )
            with np.load(box_path, allow_pickle=False) as arrays:
                occurrence_count = int(len(arrays["occurrence_id"]))
            entry = {
                "pdb_id": source_entry["pdb_id"],
                "box_path": _relative_box_path(preparation_root, box_path),
                "num_occurrences": occurrence_count,
            }
            target = oversized[split] if occurrence_count > max_slots_per_pdb else base[split]
            target.append(entry)

    subset_rng = np.random.default_rng(seed)
    train_entries = _choose_fraction(base["train"], pdb_sample_fraction, subset_rng)
    validation_base = _choose_fraction(
        base["validation"], pdb_sample_fraction, np.random.default_rng(seed + 1)
    )

    sampling_config = AnchorDataConfig(
        data_root=data_root,
        stage1_preparation_root=preparation_root,
        experiment_manifest=output_path,
        split="validation",
        training=False,
        seed=seed,
        p_miss=p_miss,
        p_split=p_split,
        p_hit=p_hit,
        max_context_ratio=max_context_ratio,
        empty_A_context_skip_probability=empty_A_context_skip_probability,
        receptor_radius=receptor_radius,
    )
    if not np.isclose(p_miss + p_split + p_hit, 1.0):
        raise ValueError("p_miss、p_split、p_hit 之和必须为 1。")
    validation_entries = []
    zero_candidate = []
    validation_rng = np.random.default_rng(seed)
    for entry in validation_base:
        pdb_id = entry["pdb_id"]
        with np.load(preparation_root / entry["box_path"], allow_pickle=False) as arrays:
            box = {key: arrays[key] for key in arrays.files}
        with np.load(
            data_root / "parse" / pdb_id / "receptor_tokens.npz", allow_pickle=False
        ) as arrays:
            receptor_coordinates = arrays["coords"]
        with np.load(data_root / "density" / pdb_id / "exp.npz", allow_pickle=False) as arrays:
            voxel_size_xyz = arrays["voxel_size"]
            origin_xyz = arrays["origin"]
        starts, audit = sample_candidate_starts(
            box,
            receptor_coordinates,
            origin_xyz,
            voxel_size_xyz,
            validation_rng,
            sampling_config,
        )
        if not starts:
            zero_candidate.append(entry)
            continue
        validation_entries.append({**entry, "candidate_start_zyx": starts, **audit})

    box_config_path = source_box_config or source_manifest.parent / "config.json"
    if not box_config_path.is_file():
        raise FileNotFoundError(f"找不到来源 BOX 配置：{box_config_path}")
    result = {
        "schema_version": 1,
        "route": "anchor_O_O_prime",
        "source_box_manifest": source_manifest.as_posix(),
        "source_box_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        "source_box_config": box_config_path.as_posix(),
        "source_box_config_sha256": hashlib.sha256(box_config_path.read_bytes()).hexdigest(),
        "source_box_metadata": {
            key: value for key, value in source.items() if key != "splits"
        },
        "seed": seed,
        "max_slots_per_pdb": max_slots_per_pdb,
        "pdb_sample_fraction": pdb_sample_fraction,
        "sampling": {
            "p_miss": sampling_config.p_miss,
            "p_split": sampling_config.p_split,
            "p_hit": sampling_config.p_hit,
            "max_context_ratio": sampling_config.max_context_ratio,
            "empty_A_context_skip_probability": sampling_config.empty_A_context_skip_probability,
            "receptor_radius_angstrom": sampling_config.receptor_radius,
        },
        "splits": {"train": train_entries, "validation": validation_entries},
        "excluded": {
            "missing_box_pdb": [
                {
                    "pdb_id": pdb_id,
                    "num_occurrences": count,
                    "reason": "原始完整图不足以产生 80^3 BOX，未进入来源 BOX manifest",
                }
                for pdb_id, count in MISSING_BOX_PDB.items()
            ],
            "oversized_pdb": oversized,
            "validation_zero_candidate": zero_candidate,
        },
        "counts": {
            "train_pdb": len(train_entries),
            "train_occurrence": sum(item["num_occurrences"] for item in train_entries),
            "validation_pdb": len(validation_entries),
            "validation_occurrence": sum(
                item["num_occurrences"] for item in validation_entries
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, output_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Matcher Anchor 正式实验清单。")
    parser.add_argument("--data-root", type=Path, required=True, help="A–G 正式数据根目录。")
    parser.add_argument(
        "--stage1-preparation-root", type=Path, required=True, help="新版 Stage1 preparation 根目录。"
    )
    parser.add_argument("--source-manifest", type=Path, required=True, help="BOX pool manifest.json。")
    parser.add_argument("--output", type=Path, required=True, help="正式 Matcher 实验清单输出路径。")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-slots-per-pdb", type=int, default=100)
    parser.add_argument("--pdb-sample-fraction", type=float, default=1.0)
    parser.add_argument("--p-miss", type=float, default=0.30)
    parser.add_argument("--p-split", type=float, default=0.20)
    parser.add_argument("--p-hit", type=float, default=0.50)
    parser.add_argument("--max-context-ratio", type=float, default=2.0)
    parser.add_argument("--empty-A-context-skip-probability", type=float, default=0.80)
    parser.add_argument("--receptor-radius", type=float, default=18.0, help="A 球半径，单位 Å。")
    parser.add_argument(
        "--source-box-config",
        type=Path,
        default=None,
        help="BOX pool config.json；默认取 source-manifest 同目录文件。",
    )
    args = parser.parse_args()
    result = build_anchor_manifest(
        data_root=args.data_root,
        preparation_root=args.stage1_preparation_root,
        source_manifest=args.source_manifest,
        output_path=args.output,
        seed=args.seed,
        max_slots_per_pdb=args.max_slots_per_pdb,
        pdb_sample_fraction=args.pdb_sample_fraction,
        p_miss=args.p_miss,
        p_split=args.p_split,
        p_hit=args.p_hit,
        max_context_ratio=args.max_context_ratio,
        empty_A_context_skip_probability=args.empty_A_context_skip_probability,
        receptor_radius=args.receptor_radius,
        source_box_config=args.source_box_config,
    )
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
