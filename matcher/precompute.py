"""以 PDB 为并行单位构建冻结清单、105 通道体素和 PP token 缓存。

``precompute_inputs`` 是编程入口；命令行用 ``--products voxel pp`` 显式选择产物。
体素写入 ``cache/voxel_input_v1``，PP 写入含 ``top_k`` 的 ``cache/pp_input_v1``。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, TypeAlias, cast

from .manifest import ensure_manifest, load_jsonl
from .pp_input import PpInputStore
from .schema import SourceRoots
from .voxel_input import VoxelInputStore


PrecomputeProduct: TypeAlias = Literal["voxel", "pp"]


def precompute_inputs(
    roots: SourceRoots,
    manifest_path: Path,
    split_names: tuple[str, ...],
    workers: int,
    products: tuple[PrecomputeProduct, ...],
    pp_top_k: int | None,
) -> None:
    """按 PDB 分配显式选择的缓存任务，使每个进程只解压一次完整源数组。"""

    if workers <= 0:
        raise ValueError("workers 必须为正整数。")
    if not split_names or len(set(split_names)) != len(split_names):
        raise ValueError("split_names 必须是非空且不重复的清单划分序列。")
    if (
        not products
        or len(set(products)) != len(products)
        or not set(products).issubset({"voxel", "pp"})
    ):
        raise ValueError("products 必须是非空且不重复的 voxel/pp 序列。")
    if "pp" in products and (pp_top_k is None or pp_top_k <= 0):
        raise ValueError("选择 pp 产物时必须提供正整数 pp_top_k。")
    if "pp" not in products and pp_top_k is not None:
        raise ValueError("未选择 pp 产物时不得提供 pp_top_k。")
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
        futures = {
            executor.submit(_precompute_one_pdb, item, products, pp_top_k): item
            for item in work_items
        }
        for future in as_completed(futures):
            split_name, pdb_id, count, completed_products = future.result()
            completed_candidates += count
            print(
                f"[precompute] products={','.join(completed_products)} "
                f"{split_name}/{pdb_id}: {count} candidates; "
                f"completed={completed_candidates}",
                flush=True,
            )


def _precompute_one_pdb(
    item: tuple[SourceRoots, str, str, tuple[int, ...]],
    products: tuple[PrecomputeProduct, ...],
    pp_top_k: int | None,
) -> tuple[str, str, int, tuple[PrecomputeProduct, ...]]:
    """在一个子进程内顺序构建同一 PDB 已选择的全部候选缓存。"""

    roots, split_name, pdb_id, candidate_indices = item
    if "voxel" in products:
        voxel_store = VoxelInputStore(roots)
        for candidate_index in candidate_indices:
            voxel_store.load(split_name, pdb_id, candidate_index)
    if "pp" in products:
        if pp_top_k is None:
            raise AssertionError("pp 产品已通过入口验证，pp_top_k 不应为空。")
        pp_store = PpInputStore(roots, pp_top_k, cache_bytes=0)
        pp_candidate_count = pp_store.precompute_pdb(split_name, pdb_id)
        expected_candidate_indices = tuple(range(pp_candidate_count))
        if candidate_indices != expected_candidate_indices:
            raise ValueError(
                f"{split_name}/{pdb_id} 的 PP/F1 候选为 {expected_candidate_indices}，"
                f"清单候选为 {candidate_indices}。"
            )
    return split_name, pdb_id, len(candidate_indices), products


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    """解析正式预计算命令；``pp_top_k=None`` 只允许未选择 PP 时出现。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--coverage-threshold", type=float, required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument(
        "--products",
        nargs="+",
        choices=("voxel", "pp"),
        required=True,
        help="显式选择 voxel、pp 或两者。",
    )
    parser.add_argument(
        "--pp-top-k",
        type=int,
        help="选择 pp 时必填；否则必须省略。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """建立或复用清单后，并行物化指定划分与显式选择的缓存。"""

    arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
    roots = SourceRoots(arguments.data_root, arguments.stage1_root, arguments.output_root)
    split_names = tuple(arguments.splits)
    manifest_path = ensure_manifest(
        roots,
        arguments.coverage_threshold,
        arguments.selection_seed,
        split_names,
    )
    products = cast(tuple[PrecomputeProduct, ...], tuple(arguments.products))
    precompute_inputs(
        roots,
        manifest_path,
        split_names,
        arguments.workers,
        products,
        arguments.pp_top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
