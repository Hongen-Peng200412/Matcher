"""
从 Stage1 完整概率图预计算候选级 48³ PP token，并按 PDB 原子发布和读取版本化缓存。

主要入口:
    - PpInputStore.load: 返回一个候选的 PP token、48³ 内 ZYX 索引、实际裁剪起点和概率图几何。
    - PpInputStore.precompute_pdb: 显式构建或严格复用一个 PDB 的全部候选 PP 缓存。

落盘产物:
    - output_root/cache/pp_input_v{PP_CACHE_SCHEMA_VERSION}/{crop_strategy}/top_{top_k}/{split_name}/{pdb_id}.npz: NPZ；保存一个 PDB 的固定数量候选和每候选 K=min(top_k, 48³) 个 PP token。

缓存缺失时，第一个持有同路径 lock 文件的进程只解压一次完整 probability_map.npz 并构建该 PDB 的全部候选；其他进程等待原子发布后复用。缓存命中后，训练 worker 不再打开完整概率图。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

import numpy as np

from ..utils.cache import ByteBudgetLru
from .geometry import resolve_candidate_crop, voxel_zyx_to_crop_centered_world
from ..utils.locking import exclusive_file_lock
from .schema import (
    LOCAL_BOX_SIZE,
    LOCAL_CROP_STRATEGIES,
    LOCAL_CONTEXT_HALF_EXTENT_VOXELS,
    PP_CACHE_SCHEMA_VERSION,
    PP_TOKEN_DIM,
    LocalCropStrategy,
    SourceRoots,
)


class PpCandidateInput(TypedDict):
    """
    保存 PpInputStore.load 返回的一个候选 PP 输入及其来源几何。

    形状符号:
        - K: 实际 PP 体素数，数值为 min(top_k, 48³)。

    字段:
        - tokens: float32 数组，(K, 4)；全部有限且无缺失值，四列依次为 Stage1 固化概率原值、relative_x、relative_y、relative_z，四列均无量纲。
        - tokens[:, 0]: float32 数组，(K,)；probability_map.npz 中入选体素的概率原值，不执行 clip、z-score 或其他重新缩放。
        - tokens[:, 1:4]: float32 数组，(K, 3)；体素中心相对实际 48³ 裁剪中心的 XYZ 物理坐标，再分别除以各轴 24 个体素对应的物理半边长。
        - index_zyx: int64 数组，(K, 3)；第 k 项与 tokens[k] 对齐，表示该 token 在实际 48³ 裁剪内的 ZYX 下标，数值逐轴位于 [0, 48)。
        - crop_start_zyx: int64 数组，(3,)；该候选实际 48³ 裁剪在 F1 80³ 内的 ZYX 起点，数值逐轴位于 [0, 32]。
        - probability_origin: float32 数组，(3,)；完整概率图角点的世界 XYZ 坐标，单位 Å，全部有限且无缺失值。
        - probability_voxel_size: float32 数组，(3,)；完整概率图 XYZ 三轴体素尺寸，单位 Å/voxel，全部有限、无缺失且逐轴大于 0。

    index_zyx 和 crop_start_zyx 在 NPZ 中分别以 int16 保存；load 返回候选切片时显式转换为 int64，避免 Dataset 侧索引运算继承存储压缩 dtype。
    """

    tokens: np.ndarray
    index_zyx: np.ndarray
    crop_start_zyx: np.ndarray
    probability_origin: np.ndarray
    probability_voxel_size: np.ndarray


class _PpPdbCache(TypedDict):
    """
    保存一个 PDB 的全部候选 PP 输入，是内存缓存与 NPZ values 字段共享的结构。

    形状符号:
        - C: 当前 PDB 的 Stage1 F1 候选数量，必须大于 0。
        - K: 每个候选固定的 PP 体素数，数值为 min(top_k, 48³)。

    字段:
        - candidate_indices: int64 数组，(C,)；严格等于 0, 1, ..., C-1，并与 F1_centered.npz 的候选第一维顺序一致。
        - tokens: float32 数组，(C, K, 4)；第 c 个切片是 candidate_indices[c] 的 PP token，最后一维依次为概率原值、relative_x、relative_y、relative_z。
        - index_zyx: int16 数组，(C, K, 3)；index_zyx[c, k] 与 tokens[c, k] 对齐，表示第 c 个候选 48³ 内的 ZYX 下标，数值逐轴位于 [0, 48)。
        - crop_start_zyx: int16 数组，(C, 3)；第 c 项与 candidate_indices[c] 对齐，表示该候选 48³ 在 F1 80³ 内的实际 ZYX 起点，数值逐轴位于 [0, 32]。
        - probability_origin: float32 数组，(3,)；该 PDB 完整概率图角点的世界 XYZ 坐标，单位 Å。
        - probability_voxel_size: float32 数组，(3,)；该 PDB 完整概率图 XYZ 三轴体素尺寸，单位 Å/voxel，逐轴大于 0。
    """

    candidate_indices: np.ndarray
    tokens: np.ndarray
    index_zyx: np.ndarray
    crop_start_zyx: np.ndarray
    probability_origin: np.ndarray
    probability_voxel_size: np.ndarray


# set[str], PP cache schema v2 允许的精确 NPZ 顶层字段集合；读取时不允许缺失字段或额外字段。
_CACHE_FIELDS = {
    "schema_version",
    "top_k",
    "candidate_indices",
    "tokens",
    "index_zyx",
    "crop_start_zyx",
    "probability_origin",
    "probability_voxel_size",
}

# NOTE: 在未来，如前所述，会预先保存 48^3 的概率，所以这部分逻辑需要重构——————因为在这种情况下，我们可能有条件现场计算PP而不产生I/O瓶颈。有个好处: k 可以作为参数低成本地调
def _build_pp_tokens(
    probability_map: np.ndarray,
    box_start_zyx: np.ndarray,
    crop_start_zyx: np.ndarray,
    voxel_size_xyz: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    从完整概率图中裁出候选 48³，并按 PP cache schema v2 构造最高概率 K 个 token。

    输入参数:
        - probability_map: float32 数组，(Z_full, Y_full, X_full)；一个 PDB 的完整 Stage1 概率图，轴顺序为 Z、Y、X。
        - box_start_zyx: 整数数组，(3,)；当前 F1 候选 80³ 在 probability_map 中的 ZYX 起点。
        - crop_start_zyx: 整数数组，(3,)；当前 48³ 在候选 80³ 内的实际 ZYX 起点，逐轴位于 [0, 32]。
        - voxel_size_xyz: 数组，(3,)；概率图 XYZ 三轴体素尺寸，单位 Å/voxel，用于把相对体素坐标转换为无量纲特征。
        - top_k: int；正整数，最多保留的概率体素数。

    排序规则:
        - selected_subset: int64 数组，(K,)；用 np.argpartition 选出 K=min(top_k, 48³) 个展平体素；第 K 名概率并列时保留 NumPy 当前既有入选子集。
        - selected_order: int64 数组，(K,)；只对已入选体素按概率降序、NumPy C-order（最后一轴变化最快）展平索引升序排列。

    返回值:
        - tokens: float32 数组，(K, 4)；四列依次为概率原值、relative_x、relative_y、relative_z。
        - local_index_zyx: int64 数组，(K, 3)；与 tokens 第一维逐 token 对齐的 48³ 内 ZYX 下标。

    改变并列入选规则或 token 排序时必须提升 PP_CACHE_SCHEMA_VERSION，防止复用语义不同的既有缓存。
    """

    # int64, (3,), 当前 48³ 在候选 80³ 内的实际 ZYX 起点。
    local_start = np.asarray(crop_start_zyx, dtype=np.int64)
    # int64, (3,), 当前 48³ 在完整 probability_map 中的 ZYX 起点。
    start = np.asarray(box_start_zyx, dtype=np.int64) + local_start
    # int64, (3,), 当前 48³ 在完整 probability_map 中的半开 ZYX 终点。
    stop = start + LOCAL_BOX_SIZE
    if start.shape != (3,) or np.any(start < 0) or np.any(stop > probability_map.shape):
        raise ValueError("PP 的 48³ 概率图裁剪越界。")
    # tuple[slice, slice, slice], 依次切分 probability_map 的 Z、Y、X 三轴。
    slices = tuple(slice(int(first), int(last)) for first, last in zip(start, stop))
    # float32, (48, 48, 48), 当前候选实际局部裁剪的 Stage1 概率原值。
    crop = np.asarray(probability_map[slices], dtype=np.float32)
    # float32, (48³,), 按 NumPy C-order 展平，最后一轴 X 变化最快。
    flat = crop.reshape(-1)
    # int, 本候选固定 token 数 K=min(top_k, 48³)。
    count = min(top_k, flat.size)
    # int64, (K,), np.argpartition 选择的展平体素下标；第 K 名并列的子集沿用 NumPy 既有语义。
    selected = np.argpartition(flat, flat.size - count)[-count:]
    # int64, (K,), 对 selected 按概率降序、展平下标升序排列的间接顺序。
    order = np.lexsort((selected, -flat[selected]))
    # [K] -> [K]；重排后 tokens 与 local_index 都按固定概率/下标顺序构造。
    selected = selected[order]
    # int64, (K, 3), 入选体素在 48³ crop 内的 ZYX 下标，与 selected 逐位置对齐。
    local_index = np.stack(np.unravel_index(selected, crop.shape), axis=1).astype(np.int64)
    # int64, (K, 3), 入选体素在候选 F1 80³ 内的 ZYX 下标。
    index_in_80 = local_index + local_start
    # float32, (K, 3), 入选体素中心相对实际裁剪中心的 XYZ 物理坐标，单位 Å。
    coordinates = voxel_zyx_to_crop_centered_world(
        index_in_80,
        local_start,
        voxel_size_xyz,
    )
    # float32, (3,), 实际裁剪 XYZ 三轴各 24 个体素对应的物理半边长，单位 Å。
    coordinate_scale = (
        np.asarray(voxel_size_xyz, dtype=np.float32) * LOCAL_CONTEXT_HALF_EXTENT_VOXELS
    )
    # float32, (K, 4), [Stage1 概率原值, relative_x, relative_y, relative_z]；相对坐标除以逐轴物理半边长后无量纲。
    tokens = np.concatenate(
        (flat[selected, None], coordinates / coordinate_scale[None, :]),
        axis=1,
    ).astype(np.float32, copy=False)
    if tokens.shape != (count, PP_TOKEN_DIM) or not np.isfinite(tokens).all():
        raise ValueError(f"PP token 必须为有限 ({count},{PP_TOKEN_DIM})，实际为 {tokens.shape}。")
    return tokens, local_index


def _load_pdb_cache(path: Path, top_k: int) -> _PpPdbCache:
    """
    读取并完整验证一个已经原子发布的 PDB PP 缓存。

    输入参数:
        - path: Path；一个 PDB 的版本化 PP NPZ 路径。
        - top_k: int；调用者期望的每候选 token 上限，必须与 NPZ 顶层 top_k 标量相同。

    文件字段:
        - schema_version: int64 标量；必须等于 PP_CACHE_SCHEMA_VERSION。
        - top_k: int64 标量；必须等于输入参数 top_k。
        - candidate_indices: int64 数组，(C,)；必须严格等于 0, 1, ..., C-1，C 必须大于 0。
        - tokens: float32 数组，(C, K, 4)；K=min(top_k, 48³)，全部元素必须有限。
        - index_zyx: int16 数组，(C, K, 3)；与 tokens 前两维逐候选逐 token 对齐，数值逐轴位于 [0, 48)。
        - crop_start_zyx: int16 数组，(C, 3)；与 candidate_indices 逐候选对齐，数值逐轴位于 [0, 32]。
        - probability_origin: float32 数组，(3,)；完整概率图角点的世界 XYZ 坐标，单位 Å，全部有限。
        - probability_voxel_size: float32 数组，(3,)；完整概率图 XYZ 三轴体素尺寸，单位 Å/voxel，全部有限且逐轴大于 0。

    返回值:
        - cached: _PpPdbCache；保留六个 NPZ 数组字段的存储 dtype。
        - cached.candidate_indices: int64 数组，(C,)；严格等于 0, 1, ..., C-1。
        - cached.tokens: float32 数组，(C, K, 4)；最后一维依次为概率原值和三个无量纲 relative_xyz。
        - cached.index_zyx: int16 数组，(C, K, 3)；与 tokens 前两维对齐的 48³ 内 ZYX 索引。
        - cached.crop_start_zyx: int16 数组，(C, 3)；与候选轴对齐的实际 48³ ZYX 起点。
        - cached.probability_origin: float32 数组，(3,)；完整概率图低角点世界 XYZ 坐标，单位 Å。
        - cached.probability_voxel_size: float32 数组，(3,)；完整概率图 XYZ 体素尺寸，单位 Å/voxel。
    """

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != _CACHE_FIELDS:
            raise ValueError(f"PP 缓存 {path} 的字段集合不符合 schema v{PP_CACHE_SCHEMA_VERSION}。")
        # int64 标量数组，缓存格式版本；先保留 ndarray 以验证 shape 和 dtype。
        raw_schema_version = np.asarray(archive["schema_version"])
        # int64 标量数组，缓存构建时的每候选 token 上限。
        raw_top_k = np.asarray(archive["top_k"])
        if (
            raw_schema_version.shape != ()
            or raw_schema_version.dtype != np.int64
            or raw_top_k.shape != ()
            or raw_top_k.dtype != np.int64
        ):
            raise ValueError(f"PP 缓存 {path} 的 schema_version/top_k 必须是 int64 标量。")
        # int，NPZ 记录的缓存 schema 版本，关闭归档后与 PP_CACHE_SCHEMA_VERSION 核对。
        schema_version = int(raw_schema_version.item())
        # int，NPZ 记录的每候选 token 上限，关闭归档后与调用参数 top_k 核对。
        stored_top_k = int(raw_top_k.item())
        # _PpPdbCache, NPZ 中六个数组字段；关闭 archive 后这些 ndarray 仍可独立使用。
        cached: _PpPdbCache = {
            "candidate_indices": np.asarray(archive["candidate_indices"]),
            "tokens": np.asarray(archive["tokens"]),
            "index_zyx": np.asarray(archive["index_zyx"]),
            "crop_start_zyx": np.asarray(archive["crop_start_zyx"]),
            "probability_origin": np.asarray(archive["probability_origin"]),
            "probability_voxel_size": np.asarray(archive["probability_voxel_size"]),
        }
    if schema_version != PP_CACHE_SCHEMA_VERSION or stored_top_k != top_k:
        raise ValueError(
            f"PP 缓存 {path} 的 schema/top_k={schema_version}/{stored_top_k}，"
            f"预期为 {PP_CACHE_SCHEMA_VERSION}/{top_k}。"
        )
    # int64, (C,), 期望从 0 连续递增的候选编号。
    candidate_indices = cached["candidate_indices"]
    # float32, (C, K, 4), 全部候选 PP token。
    tokens = cached["tokens"]
    # int16, (C, K, 3), 与 tokens 前两维对齐的候选内 ZYX 下标。
    indices = cached["index_zyx"]
    # int16, (C, 3), 与 candidate_indices 对齐的 48³ 实际 ZYX 起点。
    crop_starts = cached["crop_start_zyx"]
    # int，缓存候选轴长度 C，由 candidate_indices 第一维给出且必须大于 0。
    candidate_count = candidate_indices.shape[0]
    # int，每个候选固定 token 数 K=min(top_k, 48³)，用于同时核验 tokens 与 index_zyx 的第二维。
    token_count = min(top_k, LOCAL_BOX_SIZE**3)
    if (
        candidate_indices.ndim != 1
        or candidate_indices.dtype != np.int64
        or candidate_count == 0
        or not np.array_equal(candidate_indices, np.arange(candidate_count, dtype=np.int64))
    ):
        raise ValueError(f"PP 缓存 {path} 的 candidate_indices 必须从 0 连续递增。")
    if tokens.dtype != np.float32 or tokens.shape != (candidate_count, token_count, PP_TOKEN_DIM):
        raise ValueError(f"PP 缓存 {path}:tokens 的 dtype 或形状非法。")
    if not np.isfinite(tokens).all():
        raise ValueError(f"PP 缓存 {path}:tokens 含 NaN 或 Inf。")
    if indices.dtype != np.int16 or indices.shape != (candidate_count, token_count, 3):
        raise ValueError(f"PP 缓存 {path}:index_zyx 的 dtype 或形状非法。")
    if np.any(indices < 0) or np.any(indices >= LOCAL_BOX_SIZE):
        raise ValueError(f"PP 缓存 {path}:index_zyx 超出 [0,{LOCAL_BOX_SIZE})。")
    if (
        crop_starts.dtype != np.int16
        or crop_starts.shape != (candidate_count, 3)
        or np.any(crop_starts < 0)
        or np.any(crop_starts > 32)
    ):
        raise ValueError(f"PP 缓存 {path}:crop_start_zyx 必须为位于 [0,32] 的 int16 (C,3)。")
    # float32, (3,), 完整概率图角点的世界 XYZ 坐标，单位 Å。
    origin = cached["probability_origin"]
    # float32, (3,), 完整概率图 XYZ 三轴体素尺寸，单位 Å/voxel。
    voxel_size = cached["probability_voxel_size"]
    if (
        origin.shape != (3,)
        or voxel_size.shape != (3,)
        or origin.dtype != np.float32
        or voxel_size.dtype != np.float32
        or not np.isfinite(origin).all()
        or not np.isfinite(voxel_size).all()
        or np.any(voxel_size <= 0)
    ):
        raise ValueError(f"PP 缓存 {path} 的 probability_origin/voxel_size 非法。")
    return cached


def _atomic_save_pdb_cache(path: Path, values: _PpPdbCache, top_k: int) -> None:
    """
    把一个 PDB 的 PP 缓存写入同目录临时文件，刷盘后原子替换目标路径。

    输入参数:
        - path: Path；最终 NPZ 路径，父目录不存在时创建。
        - values: _PpPdbCache；由同一 PDB 全部候选构造的六数组对象。
        - values.candidate_indices: int64 数组，(C,)；严格等于 0, 1, ..., C-1。
        - values.tokens: float32 数组，(C, K, 4)；按候选和 token 对齐的 PP 特征。
        - values.index_zyx: int16 数组，(C, K, 3)；与 tokens 前两维对齐的 48³ 内 ZYX 索引。
        - values.crop_start_zyx: int16 数组，(C, 3)；与候选轴对齐的实际裁剪起点。
        - values.probability_origin: float32 数组，(3,)；当前 PDB 概率图的世界 XYZ 原点，单位 Å。
        - values.probability_voxel_size: float32 数组，(3,)；当前 PDB 概率图的 XYZ 体素尺寸，单位 Å/voxel。
        - top_k: int；写入 NPZ 的 int64 标量，参与缓存路径和 K 的解释。

    写入字段:
        - schema_version: int64 标量；当前 PP_CACHE_SCHEMA_VERSION。
        - top_k: int64 标量；输入参数 top_k。
        - candidate_indices: int64 数组，(C,)；来自 values 同名字段。
        - tokens: float32 数组，(C, K, 4)；来自 values 同名字段。
        - index_zyx: int16 数组，(C, K, 3)；来自 values 同名字段。
        - crop_start_zyx: int16 数组，(C, 3)；来自 values 同名字段。
        - probability_origin: float32 数组，(3,)；来自 values 同名字段。
        - probability_voxel_size: float32 数组，(3,)；来自 values 同名字段。

    副作用:
        - temporary_file: Path；在目标同目录创建 .{name}.tmp.{pid}，写入并 fsync 后由 os.replace 原子替换 path。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # Path, 与最终 NPZ 同目录的一次写入临时文件；同一进程编号进入文件名。
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            schema_version=np.asarray(PP_CACHE_SCHEMA_VERSION, dtype=np.int64),
            top_k=np.asarray(top_k, dtype=np.int64),
            **values,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


# ================================================================================================


class PpInputStore:
    """
    按 PDB 缓存固定 top_k 的 PP token，并向 CandidateDataset 返回一个候选切片。

    构造参数:
        - roots: SourceRoots；stage1_root 提供 F1_centered.npz 与 probability_map.npz，output_root 提供可写版本化缓存根目录。
        - roots.data_root: Path；本类当前不读取该根，但随 SourceRoots 一并保留以维持统一来源对象。
        - roots.stage1_root: Path；读取 F1_centered.npz 与 probability_map.npz 的 Stage1 根。
        - roots.output_root: Path；派生版本化 PP 缓存目录的 Matcher 输出根。
        - top_k: int；每个候选最多保留的最高概率体素数，必须为正，并进入缓存路径和 NPZ 顶层字段。
        - crop_strategy: LocalCropStrategy；决定候选 48³ 起点，并进入缓存路径以隔离不同空间语义。
        - cache_bytes: int；单个 DataLoader worker 的 PDB PP 数组 LRU 最大字节数，必须非负。

    缓存字段:
        - _cache: ByteBudgetLru[tuple[str, str], _PpPdbCache]；键依次为 split_name 和 pdb_id，值是一个 PDB 的全部候选 PP 数组。
        - _cache.value.candidate_indices: int64 数组，(C,)；连续候选编号。
        - _cache.value.tokens: float32 数组，(C, K, 4)；全部候选 PP token。
        - _cache.value.index_zyx: int16 数组，(C, K, 3)；逐候选逐 token 的 48³ 内 ZYX 索引。
        - _cache.value.crop_start_zyx: int16 数组，(C, 3)；逐候选实际裁剪起点。
        - _cache.value.probability_origin: float32 数组，(3,)；PDB 共用的概率图世界 XYZ 原点。
        - _cache.value.probability_voxel_size: float32 数组，(3,)；PDB 共用的概率图 XYZ 体素尺寸。

    本类可能读取 Stage1 来源并写入 Matcher 缓存；不修改 Stage1 文件，不决定候选监督或身份字段。
    """

    def __init__(
        self,
        roots: SourceRoots,
        top_k: int,
        crop_strategy: LocalCropStrategy,
        cache_bytes: int,
    ) -> None:
        """
        建立绑定 top_k 与局部裁剪策略的 PP 候选缓存读取器。

        输入参数:
            - roots: SourceRoots；提供 Stage1 概率来源和可写 PP 缓存根。
            - roots.data_root: Path；当前 PP 构造不读取该根，但实例保留完整 SourceRoots。
            - roots.stage1_root: Path；Stage1 F1 与完整概率图来源根。
            - roots.output_root: Path；版本化 PP 缓存输出根。
            - top_k: int；每个候选最多保留的 PP 概率 token 数，必须大于 0，并进入缓存目录。
            - crop_strategy: LocalCropStrategy；fixed_center 或 v_centroid，决定 48³ 起点与缓存目录分区。
            - cache_bytes: int；当前进程 PDB 级 _PpPdbCache LRU 的最大估算字节数，必须非负；0 表示 put 后不能长期保留条目。

        实例字段:
            - roots: SourceRoots；冻结来源与版本化输出路径集合。
            - roots.data_root: Path；统一来源对象中的冻结 A–G 根，当前类不直接读取。
            - roots.stage1_root: Path；Stage1 F1 与概率图来源根。
            - roots.output_root: Path；版本化 PP 缓存输出根。
            - top_k: int；规范化后的每模态 token 上限。
            - crop_strategy: LocalCropStrategy；当前缓存实例唯一的裁剪策略。
            - cache_bytes: int；规范化后的进程内缓存预算。
            - _cache: ByteBudgetLru[tuple[str, str], _PpPdbCache]；键依次为 split_name、pdb_id，值为该 PDB 全部候选的落盘缓存数组。
            - _cache.value.candidate_indices: int64 数组，(C,)；连续候选编号。
            - _cache.value.tokens: float32 数组，(C, K, 4)；全部候选 PP token。
            - _cache.value.index_zyx: int16 数组，(C, K, 3)；逐候选逐 token 的局部 ZYX 索引。
            - _cache.value.crop_start_zyx: int16 数组，(C, 3)；逐候选实际裁剪起点。
            - _cache.value.probability_origin: float32 数组，(3,)；概率图世界 XYZ 原点。
            - _cache.value.probability_voxel_size: float32 数组，(3,)；概率图 XYZ 体素尺寸。
        """

        if top_k <= 0:
            raise ValueError("top_k 必须为正整数。")
        if cache_bytes < 0:
            raise ValueError("cache_bytes 不能为负数。")
        if crop_strategy not in LOCAL_CROP_STRATEGIES:
            raise ValueError(f"不支持 local_crop_strategy={crop_strategy!r}。")
        self.roots = roots
        self.top_k = int(top_k)
        self.crop_strategy = crop_strategy
        self.cache_bytes = int(cache_bytes)
        self._cache: ByteBudgetLru[tuple[str, str], _PpPdbCache] = ByteBudgetLru(
            self.cache_bytes
        )

    def load(self, split_name: str, pdb_id: str, candidate_index: int) -> PpCandidateInput:
        """
        返回一个候选的 PP token、候选内索引、实际裁剪起点和概率图几何。

        输入参数:
            - split_name: str；Stage1 与缓存中的数据划分目录名。
            - pdb_id: str；当前 PDB 目录名和缓存文件名主体。
            - candidate_index: int；索引 F1_centered.npz 候选第一维，必须存在于连续 candidate_indices。

        返回值:
            - tokens: float32 数组，(K, 4)；缓存 tokens[position] 的候选视图。
            - index_zyx: int64 数组，(K, 3)；由缓存 int16 index_zyx[position] 转换，数值逐轴位于 [0, 48)。
            - crop_start_zyx: int64 数组，(3,)；由缓存 int16 crop_start_zyx[position] 转换，数值逐轴位于 [0, 32]。
            - probability_origin: float32 数组，(3,)；缓存的完整概率图世界 XYZ 角点原点，单位 Å。
            - probability_voxel_size: float32 数组，(3,)；缓存的完整概率图 XYZ 体素尺寸，单位 Å/voxel。
        """

        # _PpPdbCache, 当前 PDB 全部候选 PP 数组；可能来自 worker LRU、磁盘缓存或本次新构建。
        cached = self._load_or_build_pdb(split_name, pdb_id)
        # int64, (C,), 从 0 连续递增的候选编号。
        candidate_indices = cached["candidate_indices"]
        # int, candidate_index 在 candidate_indices 中的候选轴位置。
        position = int(np.searchsorted(candidate_indices, candidate_index))
        if (
            position >= candidate_indices.size
            or int(candidate_indices[position]) != candidate_index
        ):
            raise IndexError(f"{split_name}/{pdb_id} 不含 candidate_index={candidate_index}。")
        return {
            "tokens": cached["tokens"][position],
            "index_zyx": cached["index_zyx"][position].astype(np.int64),
            "crop_start_zyx": cached["crop_start_zyx"][position].astype(np.int64),
            "probability_origin": cached["probability_origin"],
            "probability_voxel_size": cached["probability_voxel_size"],
        }

    def precompute_pdb(self, split_name: str, pdb_id: str) -> int:
        """
        构建或严格复用一个 PDB 的 PP 缓存，并返回候选数量。

        输入参数:
            - split_name: str；Stage1 与缓存中的数据划分目录名。
            - pdb_id: str；当前 PDB 目录名和缓存文件名主体。

        返回值:
            - candidate_count: int；缓存 candidate_indices 的长度。
        """

        return int(self._load_or_build_pdb(split_name, pdb_id)["candidate_indices"].size)

    def _load_or_build_pdb(self, split_name: str, pdb_id: str) -> _PpPdbCache:
        """
        命中当前 worker 的 PDB LRU，或在进程文件锁内构建并严格读取磁盘缓存。

        输入参数:
            - split_name: str；与 pdb_id 共同组成 worker LRU 键和缓存相对目录。
            - pdb_id: str；与 split_name 共同组成 worker LRU 键和缓存文件名。

        返回值:
            - cached: _PpPdbCache；无论磁盘文件原先存在还是本次构建，返回前都完成严格核验。
            - cached.candidate_indices: int64 数组，(C,)；严格等于 0, 1, ..., C-1。
            - cached.tokens: float32 数组，(C, K, 4)；按候选和 token 对齐的 PP 特征。
            - cached.index_zyx: int16 数组，(C, K, 3)；与 tokens 前两维对齐的 48³ 内 ZYX 索引。
            - cached.crop_start_zyx: int16 数组，(C, 3)；与候选轴对齐的实际裁剪起点。
            - cached.probability_origin: float32 数组，(3,)；完整概率图世界 XYZ 原点，单位 Å。
            - cached.probability_voxel_size: float32 数组，(3,)；完整概率图 XYZ 体素尺寸，单位 Å/voxel。

        并发规则:
            - lock_path: Path；最终 NPZ 同路径但后缀为 .lock，第一次 is_file 检查后在锁内再次检查，避免多个进程重复发布。
        """

        # tuple[str, str], 当前 worker LRU 键，依次为划分名和 PDB 标识。
        key = (split_name, pdb_id)
        # _PpPdbCache | None，命中时包含当前 PDB 固定 (C, K, ...) 布局的六个数组；None 表示需要检查磁盘或构建。
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        # Path, 与 schema、crop_strategy、top_k、split_name 和 pdb_id 共同绑定的最终 NPZ 路径。
        path = self._cache_path(split_name, pdb_id)
        if not path.is_file():
            with exclusive_file_lock(path.with_suffix(".lock")):
                if not path.is_file():
                    # _PpPdbCache，按候选轴堆叠的六个待写数组；候选索引固定为 0..C-1，不使用 offsets。
                    values = self._build_pdb(split_name, pdb_id)
                    _atomic_save_pdb_cache(path, values, self.top_k)
        # _PpPdbCache，从最终 NPZ 重新读取并完成 schema、dtype、形状、范围与有限值核验的六数组对象。
        cached = _load_pdb_cache(path, self.top_k)
        self._cache.put(key, cached)
        return cached

    def _build_pdb(self, split_name: str, pdb_id: str) -> _PpPdbCache:
        """
        只解压一次完整概率图，并按 F1_centered.npz 的候选顺序构造该 PDB 的全部 PP 输入。

        输入参数:
            - split_name: str；Stage1 来源划分目录名。
            - pdb_id: str；Stage1 PDB 目录名。

        来源文件:  # NOTE:统一在一个.npz文件中看起来并不难，且I/O仍然很快，对嘛？
            - {stage1_root}/{split_name}/{pdb_id}/centered/F1_centered.npz: NPZ；提供候选 80³ 起点、体素尺寸和完整权威 V 稀疏表。
            - F1_centered.npz/box_start_zyx: 整数数组，(C, 3)；第 c 项是候选 c 在完整概率图中的 80³ ZYX 起点。
            - F1_centered.npz/voxel_size_world: float32 数组，(C, 3)；第 c 项是候选 c 的 XYZ 体素尺寸，单位 Å/voxel。
            - F1_centered.npz/voxel_offsets: int64 数组，(C + 1,)；区间 [voxel_offsets[c], voxel_offsets[c + 1]) 同步切分 voxel_index_local_zyx、centered_probability 和 voxel_final，首值为 0，末值等于三个并列数组的第一维长度；本模块只读取 voxel_index_local_zyx。
            - F1_centered.npz/voxel_index_local_zyx: 整数数组，(L_v, 3)；全部 C 个候选串联的权威 V 体素，最后一维依次为 Z、Y、X，候选内数值逐轴位于 [0, 80)。
            - {stage1_root}/{split_name}/{pdb_id}/probability/probability_map.npz: NPZ；提供完整概率图和空间几何。
            - probability_map.npz/probability_map: float32 数组，(Z_full, Y_full, X_full)；完整 Stage1 融合概率，轴顺序为 Z、Y、X，全部有限。
            - probability_map.npz/origin_xyz: float32 数组，(3,)；完整概率图角点的世界 XYZ 坐标，单位 Å。
            - probability_map.npz/voxel_size_xyz: float32 数组，(3,)；完整概率图 XYZ 三轴体素尺寸，单位 Å/voxel，逐轴大于 0。

        返回值:
            - candidate_indices: int64 数组，(C,)；严格等于 0, 1, ..., C-1。
            - tokens: float32 数组，(C, K, 4)；按候选堆叠的 PP token。
            - index_zyx: int16 数组，(C, K, 3)；按候选堆叠的 48³ 内 token 下标。
            - crop_start_zyx: int16 数组，(C, 3)；每个候选由完整 V 和 crop_strategy 解析的实际起点。
            - probability_origin: float32 数组，(3,)；原样保存 probability_map.npz/origin_xyz。
            - probability_voxel_size: float32 数组，(3,)；原样保存 probability_map.npz/voxel_size_xyz。

        本函数要求每个候选的 voxel_size_world 与 probability_voxel_size 逐元素相同；候选框世界原点与完整概率图原点的进一步对齐由 CandidateDataset 的几何校验负责。
        """

        # Path, 当前 PDB 的 F1 候选几何和权威 V 稀疏表。
        centered_path = (
            self.roots.stage1_root / split_name / pdb_id / "centered/F1_centered.npz"
        )
        with np.load(centered_path, allow_pickle=False) as archive:
            # int64, (C, 3), 每个候选 80³ 在完整概率图中的 ZYX 起点。
            box_starts = np.asarray(archive["box_start_zyx"], dtype=np.int64)
            # float32, (C, 3), 每个候选 XYZ 体素尺寸，单位 Å/voxel。
            voxel_sizes = np.asarray(archive["voxel_size_world"], dtype=np.float32)
            # int64, (C + 1,), 来源契约同步切分 voxel_index_local_zyx、centered_probability 和 voxel_final；本函数只用它切分 voxel_indices。
            voxel_offsets = np.asarray(archive["voxel_offsets"], dtype=np.int64)
            # int64, (L_v, 3), 全部候选串联的权威 V 体素在各自 80³ 内的 ZYX 下标。
            voxel_indices = np.asarray(archive["voxel_index_local_zyx"], dtype=np.int64)
        # Path, 当前 PDB 的完整 Stage1 概率图及 XYZ 几何。
        probability_path = (
            self.roots.stage1_root
            / split_name
            / pdb_id
            / "probability/probability_map.npz"
        )
        with np.load(probability_path, allow_pickle=False) as archive:
            # float32, (Z_full, Y_full, X_full), 完整概率图，轴顺序为 Z、Y、X。
            probability_map = np.asarray(archive["probability_map"], dtype=np.float32)
            # float32, (3,), 完整概率图角点的世界 XYZ 坐标，单位 Å。
            probability_origin = np.asarray(archive["origin_xyz"], dtype=np.float32)
            # float32, (3,), 完整概率图 XYZ 三轴体素尺寸，单位 Å/voxel。
            probability_voxel_size = np.asarray(archive["voxel_size_xyz"], dtype=np.float32)
        # int，F1_centered.npz 候选轴长度 C；后续三个列表都按 0..C-1 各追加一项。
        candidate_count = box_starts.shape[0]
        if box_starts.shape != (candidate_count, 3) or voxel_sizes.shape != (candidate_count, 3):
            raise ValueError(f"{centered_path} 的候选几何字段形状不一致。")
        if probability_map.ndim != 3 or not np.isfinite(probability_map).all():
            raise ValueError(f"{probability_path}:probability_map 必须为有限三维数组。")
        if (
            probability_origin.shape != (3,)
            or probability_voxel_size.shape != (3,)
            or not np.isfinite(probability_origin).all()
            or not np.isfinite(probability_voxel_size).all()
            or np.any(probability_voxel_size <= 0)
        ):
            raise ValueError(f"{probability_path} 的 origin_xyz/voxel_size_xyz 非法。")
        if not np.isfinite(voxel_sizes).all() or np.any(voxel_sizes <= 0):
            raise ValueError(f"{centered_path}:voxel_size_world 必须为有限正数。")
        # list[np.ndarray], 长度最终为 C；每项 float32 (K, 4)，按候选编号顺序追加。
        token_items: list[np.ndarray] = []
        # list[np.ndarray], 长度最终为 C；每项 int16 (K, 3)，与 token_items 同位置候选逐 token 对齐。
        index_items: list[np.ndarray] = []
        # list[np.ndarray], 长度最终为 C；每项 int16 (3,)，与候选编号逐位置对齐。
        crop_start_items: list[np.ndarray] = []
        for candidate_index in range(candidate_count):
            if not np.array_equal(voxel_sizes[candidate_index], probability_voxel_size):
                raise ValueError(
                    f"{split_name}/{pdb_id}/candidate_{candidate_index} 的概率图体素尺寸不一致。"
                )
            # 两个 int，切分 voxel_indices 中当前候选完整权威 V 的半开区间。
            voxel_start = int(voxel_offsets[candidate_index])
            voxel_stop = int(voxel_offsets[candidate_index + 1])
            # CandidateCropGeometry, 从当前候选完整 V 解析的 48³ 实际起点和采样统计；PP 只消费 start_zyx。
            crop_geometry = resolve_candidate_crop(
                voxel_indices[voxel_start:voxel_stop],
                self.crop_strategy,
            )
            # float32，(K, 4)，当前候选按概率与展平索引稳定排序的 PP token。
            # int64，(K, 3)，与 tokens 第一维逐 token 对齐的 48³ 内 ZYX 索引。
            tokens, indices = _build_pp_tokens(
                probability_map,
                box_starts[candidate_index],
                crop_geometry.start_zyx,
                voxel_sizes[candidate_index],
                self.top_k,
            )
            token_items.append(tokens)
            index_items.append(indices.astype(np.int16))
            crop_start_items.append(crop_geometry.start_zyx.astype(np.int16))
        # candidate_indices 为 int64 (C,)，与 F1 候选轴逐位置一致且固定为 0..C-1。
        # token_items 沿新候选轴堆叠为 float32 (C, K, 4)。
        # index_items 沿同一候选轴堆叠为 int16 (C, K, 3)，前两维与 tokens 对齐。
        # crop_start_items 沿候选轴堆叠为 int16 (C, 3)，第一维与 candidate_indices 对齐。
        # probability_origin 与 probability_voxel_size 均为当前 PDB 共用的 float32 (3,) XYZ 几何，不沿候选轴复制。
        return {
            "candidate_indices": np.arange(candidate_count, dtype=np.int64),
            "tokens": np.stack(token_items, axis=0),
            "index_zyx": np.stack(index_items, axis=0),
            "crop_start_zyx": np.stack(crop_start_items, axis=0),
            "probability_origin": probability_origin,
            "probability_voxel_size": probability_voxel_size,
        }

    def _cache_path(self, split_name: str, pdb_id: str) -> Path:
        """
        返回不依赖训练运行编号的版本化 PDB PP 缓存路径。

        输入参数:
            - split_name: str；缓存中的数据划分目录名。
            - pdb_id: str；缓存 NPZ 文件名主体。

        返回值:
            - cache_path: Path；output_root/cache/pp_input_v{PP_CACHE_SCHEMA_VERSION}/{crop_strategy}/top_{top_k}/{split_name}/{pdb_id}.npz。
        """

        return (
            self.roots.pp_cache_root(self.top_k, self.crop_strategy)
            / split_name
            / f"{pdb_id}.npz"
        )

    def __getstate__(self) -> dict[str, object]:
        """
        为 DataLoader spawn 序列化实例状态，并把父进程已读取的 PDB PP 数组替换为空缓存。

        返回值:
            - state: dict[str, object]；可由 DataLoader spawn 序列化的实例状态。
            - state.roots: SourceRoots；与实例 roots 相同的三个来源根。
            - state.roots.data_root: Path；统一来源对象中的冻结 A–G 根，当前类不直接读取。
            - state.roots.stage1_root: Path；Stage1 F1 与完整概率图来源根。
            - state.roots.output_root: Path；版本化 PP 缓存的 Matcher 输出根。
            - state.top_k: int；与实例 top_k 相同的每候选 token 上限。
            - state.crop_strategy: LocalCropStrategy；与实例 crop_strategy 相同的裁剪策略。
            - state.cache_bytes: int；与实例 cache_bytes 相同的字节预算。
            - state._cache: ByteBudgetLru[tuple[str, str], _PpPdbCache]；使用 state.cache_bytes 新建的空缓存，不包含父进程 PP 数组。

        该方法不删除磁盘 NPZ；各 worker 仍可独立读取同一已发布缓存。
        """

        # dict[str, object], 当前实例属性浅拷贝；替换其中 _cache 不修改父进程实例。
        state = self.__dict__.copy()
        state["_cache"] = ByteBudgetLru(self.cache_bytes)
        return state
