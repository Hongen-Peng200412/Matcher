"""以 PDB 为并行单位构建冻结清单和 105 通道候选体素缓存。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .manifest import ensure_manifest, load_jsonl
from .schema import SourceRoots
from .voxel_input import VoxelInputStore


def precompute_voxel_inputs(
    roots: SourceRoots,
    manifest_path: Path,
    split_names: tuple[str, ...],
    workers: int,
) -> None:
    """按 PDB 分配候选缓存任务，使每个进程复用完整密度图和受体数组。"""

    if workers <= 0:
        raise ValueError("workers 必须为正整数。")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    work_items: list[tuple[SourceRoots, str, str, tuple[int, ...]]] = []
    for split_name in split_names:
        split_files = manifest["splits"].get(split_name)
        if split_files is None:
            raise KeyError(f"清单不含划分 {split_name!r}。")
        grouped: dict[str, list[int]] = defaultdict(list)
        for record in load_jsonl(manifest_path.parent / split_files["candidates"]):
            grouped[str(record["pdb_id"])].append(int(record["candidate_index"]))
        for pdb_id in sorted(grouped):
            work_items.append((roots, split_name, pdb_id, tuple(sorted(grouped[pdb_id]))))

    completed_candidates = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_precompute_one_pdb, item): item for item in work_items}
        for future in as_completed(futures):
            split_name, pdb_id, count = future.result()
            completed_candidates += count
            print(
                f"[voxel-cache] {split_name}/{pdb_id}: {count} candidates; "
                f"completed={completed_candidates}",
                flush=True,
            )


def _precompute_one_pdb(
    item: tuple[SourceRoots, str, str, tuple[int, ...]],
) -> tuple[str, str, int]:
    """在一个子进程内顺序构建同一 PDB 的全部候选缓存。"""

    roots, split_name, pdb_id, candidate_indices = item
    store = VoxelInputStore(roots)
    for candidate_index in candidate_indices:
        store.load(split_name, pdb_id, candidate_index)
    return split_name, pdb_id, len(candidate_indices)


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    """解析正式预计算命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--coverage-threshold", type=float, required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--workers", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """建立或复用清单后并行物化指定划分的体素输入。"""

    arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
    roots = SourceRoots(arguments.data_root, arguments.stage1_root, arguments.output_root)
    manifest_path = ensure_manifest(
        roots,
        arguments.coverage_threshold,
        arguments.selection_seed,
    )
    precompute_voxel_inputs(
        roots,
        manifest_path,
        tuple(arguments.splits),
        arguments.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
