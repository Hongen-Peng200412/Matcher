"""
以 PDB 为并行单位物化候选级 105 通道体素缓存和 PP token 缓存。

主要入口:
    - precompute_inputs: 读取既有清单候选记录，按划分/PDB 分组后交给独立子进程构建显式选择的缓存。
    - main: 建立或复用显式划分清单，再调用 precompute_inputs。

产物边界:
    - voxel: 每个候选一份 float16 数组，形状为 (105, 48, 48, 48)，路径由 VoxelInputStore 绑定 schema、裁剪策略、划分、PDB、候选编号和实际起点。
    - pp: 每个 PDB 一份 NPZ，保存每个候选的 PP 模态 token、48x48x48 内 ZYX 索引、裁剪起点和概率图几何，路径由 PpInputStore 绑定 schema、裁剪策略与 top_k。

完整来源读取成本按 PDB 内候选复用。命令行 products 必须显式选择 voxel、pp 或二者；只在选择 pp 时允许并要求提供正 pp_top_k。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np

from .manifest import ensure_manifest, load_jsonl
from .geometry import resolve_candidate_crop
from .pp_input import PpInputStore
from .schema import LOCAL_CROP_STRATEGIES, LocalCropStrategy, SourceRoots
from .voxel_input import VoxelInputStore


# TypeAlias, 允许预计算的两个产品名；序列顺序同时用于完成日志显示。
PrecomputeProduct: TypeAlias = Literal["voxel", "pp"]


def precompute_inputs(
    roots: SourceRoots,
    manifest_path: Path,
    split_names: tuple[str, ...],
    workers: int,
    products: tuple[PrecomputeProduct, ...],
    pp_top_k: int | None,
    local_crop_strategy: LocalCropStrategy,
) -> None:
    """
    按 PDB 并行物化显式选择的候选缓存。

    输入参数:
        - roots: SourceRoots；提供冻结数据根、Stage1 根和可写 Matcher 输出根。
        - roots.data_root: Path；供子进程构造候选体素时读取完整密度与受体原子表。
        - roots.stage1_root: Path；读取 F1 候选、完整概率图和 PP 来源。
        - roots.output_root: Path；派生候选体素和 PP 的版本化可写缓存目录。
        - manifest_path: Path；已经物化所需划分的 manifest.json 路径。
        - split_names: tuple[str, ...]；本次读取的清单划分，必须非空且不重复。
        - workers: int；ProcessPoolExecutor 最大子进程数，必须大于 0。
        - products: tuple[PrecomputeProduct, ...]；显式选择 voxel、pp 或二者，必须非空且不重复。
        - pp_top_k: int 或 None；选择 pp 时必须为正整数，未选择 pp 时必须为 None。
        - local_crop_strategy: LocalCropStrategy；fixed_center 或 v_centroid，决定 48³ 起点以及两个缓存目录的策略分区。

    读取的 manifest 字段:
        - splits: dict[str, dict[str, str]]；划分名到候选/身份文件名的映射。
        - splits.{split_name}.candidates: str；当前划分候选 JSONL 相对 manifest_path.parent 的文件名。

    读取的候选字段:
        - pdb_id: str；用于把候选按 PDB 分组。
        - candidate_index: int；当前候选在 F1 候选轴中的零基索引。

    工作项字段:
        - roots: SourceRoots；传入子进程的来源与输出根。
        - split_name: str；当前候选组所属划分。
        - pdb_id: str；当前候选组所属 PDB。
        - candidate_indices: tuple[int, ...]；当前 PDB 清单中的候选编号，按整数递增。

    副作用:
        - cache: 子进程通过 VoxelInputStore 和 PpInputStore 原子发布显式选择的缓存。
        - stdout: 每个完成 PDB 输出产品名、划分/PDB、候选数和累计完成候选数。

    本函数不返回值；任一子进程异常在 future.result 时传播并中止成功返回。
    """

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
    if local_crop_strategy not in LOCAL_CROP_STRATEGIES:
        raise ValueError(f"不支持 local_crop_strategy={local_crop_strategy!r}。")
    # dict[str, object], 既有主清单 JSON 对象；本函数只读取 splits 文件映射。
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # list[tuple[SourceRoots,str,str,tuple[int,...]]], 每项对应一个划分/PDB 的全部清单候选。
    work_items: list[tuple[SourceRoots, str, str, tuple[int, ...]]] = []
    for split_name in split_names:
        # dict[str, str] | None, 当前划分候选与身份 JSONL 的相对文件名映射。
        split_files = manifest["splits"].get(split_name)
        if split_files is None:
            raise KeyError(f"清单不含划分 {split_name!r}。")
        # defaultdict[str, list[int]], 当前划分按 PDB 收集的候选索引。
        grouped: dict[str, list[int]] = defaultdict(list)
        for record in load_jsonl(manifest_path.parent / split_files["candidates"]):
            grouped[str(record["pdb_id"])].append(int(record["candidate_index"]))
        for pdb_id in sorted(grouped):
            work_items.append((roots, split_name, pdb_id, tuple(sorted(grouped[pdb_id]))))

    # int, 已完成子进程返回的候选数累计值，仅用于进度日志。
    completed_candidates = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # dict[Future[tuple[str, str, int, tuple[PrecomputeProduct, ...]]], tuple[SourceRoots, str, str, tuple[int, ...]]]，子进程结果句柄到其来源根、划分、PDB 和候选编号工作项的映射。
        futures = {
            executor.submit(
                _precompute_one_pdb,
                item,
                products,
                pp_top_k,
                local_crop_strategy,
            ): item
            for item in work_items
        }
        for future in as_completed(futures):
            # str，已完成子进程处理的清单划分名。
            # str，已完成子进程处理的小写 PDB 标识。
            # int，该 PDB 清单工作项中的候选编号数量。
            # tuple[PrecomputeProduct, ...]，该子进程实际按输入顺序完成的 voxel/pp 产品序列；future.result 同时传播子进程异常。
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
    local_crop_strategy: LocalCropStrategy,
) -> tuple[str, str, int, tuple[PrecomputeProduct, ...]]:
    """
    在一个子进程内顺序构建同一 PDB 显式选择的全部候选缓存。

    输入参数:
        - item: tuple[SourceRoots, str, str, tuple[int, ...]]；单个 PDB 工作项。
        - item[0]: SourceRoots；冻结来源与可写输出根。
        - item[0].data_root: Path；voxel 产品读取完整密度与受体原子表的冻结 A–G 根。
        - item[0].stage1_root: Path；voxel 与 pp 产品共同读取 F1 和概率图的 Stage1 根。
        - item[0].output_root: Path；两个产品发布版本化缓存的 Matcher 输出根。
        - item[1]: str；当前划分名 split_name。
        - item[2]: str；当前小写 PDB 标识 pdb_id。
        - item[3]: tuple[int, ...]；清单候选编号，预期按递增顺序排列。
        - products: tuple[PrecomputeProduct, ...]；当前子进程要构建的 voxel、pp 或二者。
        - pp_top_k: int 或 None；选择 pp 时由入口保证为正整数。
        - local_crop_strategy: LocalCropStrategy；两个产品共享的 48³ 起点解析策略。

    F1 稀疏字段:
        - voxel_offsets: int64 数组，(C + 1,)；同时切分 voxel_index_local_zyx、centered_probability 和 voxel_final；本函数只读取前者。
        - voxel_index_local_zyx: int64 数组，(V_total, 3)；拼接全部候选的 80³ 局部 ZYX 稀疏体素坐标。

    返回值:
        - split_name: str；完成的划分名。
        - pdb_id: str；完成的小写 PDB 标识。
        - candidate_count: int；item 中候选编号数量。
        - completed_products: tuple[PrecomputeProduct, ...]；与输入 products 相同，用于父进程日志。

    voxel 按每个清单候选解析实际 48³ 起点并逐候选加载或构建；pp 一次构建整个 PDB，并要求清单候选恰好是从 0 到 C-1 的完整连续序列。
    """

    # SourceRoots，子进程读取冻结来源和写入版本化缓存所需的三个根目录。
    # str，当前工作项所属的清单划分名。
    # str，当前工作项的小写 PDB 标识。
    # tuple[int, ...]，当前 PDB 清单候选编号的递增元组。
    roots, split_name, pdb_id, candidate_indices = item
    if "voxel" in products:
        # Path, 当前 PDB 的 F1 候选稀疏体素产物。
        centered_path = roots.stage1_root / split_name / pdb_id / "centered/F1_centered.npz"
        with np.load(centered_path, allow_pickle=False) as archive:
            # int64, (C+1,), 切分拼接稀疏候选字段的右开偏移。
            voxel_offsets = np.asarray(archive["voxel_offsets"], dtype=np.int64)
            # int64, (V_total,3), 全部候选拼接的 80³ 局部 ZYX 稀疏体素坐标。
            voxel_indices = np.asarray(archive["voxel_index_local_zyx"], dtype=np.int64)
        # dict[int, int64 ndarray], 每个清单候选编号到解析后 (3,) ZYX 48³ 起点的映射。
        crop_starts: dict[int, np.ndarray] = {}
        for candidate_index in candidate_indices:
            # int, 当前候选在 voxel_indices 中的左闭切片起点。
            start = int(voxel_offsets[candidate_index])
            # int, 当前候选在 voxel_indices 中的右开切片终点。
            stop = int(voxel_offsets[candidate_index + 1])
            crop_starts[candidate_index] = resolve_candidate_crop(
                voxel_indices[start:stop],
                local_crop_strategy,
            ).start_zyx
        # VoxelInputStore, 复用当前 PDB 完整密度与受体数组的候选体素缓存适配器。
        voxel_store = VoxelInputStore(roots, local_crop_strategy)
        for candidate_index in candidate_indices:
            voxel_store.load(
                split_name,
                pdb_id,
                candidate_index,
                crop_starts[candidate_index],
            )
    if "pp" in products:
        if pp_top_k is None:
            raise AssertionError("pp 产品已通过入口验证，pp_top_k 不应为空。")
        # PpInputStore, 当前 PDB 的 PP 模态 token 批量缓存构造器；关闭进程内 PDB LRU。
        pp_store = PpInputStore(
            roots,
            pp_top_k,
            local_crop_strategy,
            cache_bytes=0,
        )
        # int, PP 来源核验后得到并已物化的 F1 候选数 C。
        pp_candidate_count = pp_store.precompute_pdb(split_name, pdb_id)
        # tuple[int, ...], PP/F1 要求的完整连续候选编号序列 0..C-1。
        expected_candidate_indices = tuple(range(pp_candidate_count))
        if candidate_indices != expected_candidate_indices:
            raise ValueError(
                f"{split_name}/{pdb_id} 的 PP/F1 候选为 {expected_candidate_indices}，"
                f"清单候选为 {candidate_indices}。"
            )
    return split_name, pdb_id, len(candidate_indices), products


# ================================================================================================


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    """
    解析正式预计算命令行参数。

    输入参数:
        - argv: list[str]；不含程序名的命令行 token 序列。

    返回字段:
        - data_root: Path；冻结 A–G 数据根，必须显式提供。
        - stage1_root: Path；Stage1 F1 候选根，必须显式提供。
        - output_root: Path；Matcher 输出根，必须显式提供。
        - coverage_threshold: float；构建或复用清单的候选双向覆盖阈值，必须显式提供。
        - selection_seed: int；构建或复用清单的语言表示选择种子，必须显式提供。
        - splits: list[str]；本次物化缓存的一个或多个划分名，具体合法性由清单入口核验。
        - workers: int；PDB 子进程数，必须显式提供并由 precompute_inputs 要求为正。
        - products: list[str]；非空的 voxel、pp 选择序列，重复性由 precompute_inputs 拒绝。
        - pp_top_k: int 或 None；选择 pp 时必填正整数，否则必须省略，组合门禁由 precompute_inputs 执行。
        - local_crop_strategy: str；fixed_center 或 v_centroid，必须显式提供。

    ArgumentParser 的 description 取自本模块 Docstring，因此学习注释修订允许改变 --help 的说明文字；参数名、类型、必填性、choices、解析和退出语义保持不变。
    """

    # ArgumentParser, 正式缓存预计算 CLI；模块 Docstring 仅用于帮助说明。
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
    parser.add_argument(
        "--local-crop-strategy",
        choices=LOCAL_CROP_STRATEGIES,
        required=True,
        help="显式选择 fixed_center 或按完整 V 质心解析的 v_centroid。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    建立或复用清单后，并行物化指定划分与显式选择的缓存。

    输入参数:
        - argv: list[str] 或 None；显式列表用于测试或嵌入调用，None 时读取 sys.argv[1:]。

    返回值:
        - exit_code: int；全部 PDB 缓存成功完成后固定为 0；参数、清单或子进程错误通过异常或 argparse 退出传播。
    """

    # argparse.Namespace, 已完成 CLI 类型转换、必填项和 choices 检查的参数对象。
    arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
    # SourceRoots，三个命令行根路径组成的 frozen dataclass；字段只在构造时绑定。
    roots = SourceRoots(arguments.data_root, arguments.stage1_root, arguments.output_root)
    # tuple[str, ...], 保持命令行顺序的显式清单划分序列。
    split_names = tuple(arguments.splits)
    # Path, 已构建、扩展或复用且包含 split_names 的 manifest.json。
    manifest_path = ensure_manifest(
        roots,
        arguments.coverage_threshold,
        arguments.selection_seed,
        split_names,
    )
    # tuple[PrecomputeProduct, ...], 从 argparse choices 收窄的显式产品序列。
    products = cast(tuple[PrecomputeProduct, ...], tuple(arguments.products))
    precompute_inputs(
        roots,
        manifest_path,
        split_names,
        arguments.workers,
        products,
        arguments.pp_top_k,
        cast(LocalCropStrategy, arguments.local_crop_strategy),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
