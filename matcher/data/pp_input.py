"""预计算并读取候选级 ``48³`` 的 PP token。

``PpInputStore`` 以 PDB 为构建单位，把完整 probability map 转换成固定 ``top_k`` 的候选
PP token，并原子写入
``<output_root>/cache/pp_input_v2/<strategy>/top_<k>/<split>/<pdb>.npz``。
缓存命中后，训练 worker 只读取该小型产物；缓存缺失时，第一个持锁进程一次构建该
PDB，其余进程随后复用，因此不再由每个 worker 重复解压完整 probability map。
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
    """一个候选的 PP 输入及其源几何；K 是实际保留的 PP 体素数。

    ``tokens`` 是有限 float32 ``(K,4)``，第 0 项是未重新缩放的 Stage1 固化概率原值，后三项是相对实际裁剪中心并按各轴 24 个体素的物理半边长归一化的 XYZ，四项均无量纲且没有缺失值。
    ``index_zyx`` 是 int64 ``(K,3)``，第 k 项与 ``tokens[k]`` 对齐且逐轴位于 ``[0,48)``。
    ``crop_start_zyx`` 是该候选在 F1 ``80³`` 内逐轴位于 ``[0,32]`` 的 int64 ``(3,)`` 起点。
    ``probability_origin`` 是有限且无缺失值的 float32 ``(3,)`` 完整概率图 XYZ 角点原点，单位 Å。
    ``probability_voxel_size`` 是有限、无缺失且逐轴为正的 float32 ``(3,)`` XYZ 体素尺寸，单位 Å/voxel。
    """

    tokens: np.ndarray
    index_zyx: np.ndarray
    crop_start_zyx: np.ndarray
    probability_origin: np.ndarray
    probability_voxel_size: np.ndarray


class _PpPdbCache(TypedDict):
    """一个 PDB 的全部候选 PP 输入；C 是候选数，K 是每个候选的 PP 体素数。

    ``candidate_indices`` 是从 0 开始连续递增的 int64 ``(C,)``。
    ``tokens`` 是有限 float32 ``(C,K,4)``，第 0 项是未重新缩放的 Stage1 固化概率原值，后三项是相对各候选实际裁剪中心并按各轴 24 个体素的物理半边长归一化的 XYZ，四项均无量纲且没有缺失值。
    ``index_zyx`` 是 int16 ``(C,K,3)``，``[c,k]`` 与 ``tokens[c,k]`` 对齐且逐轴位于 ``[0,48)``。
    ``crop_start_zyx`` 是 int16 ``(C,3)``，第 c 项与 ``candidate_indices[c]`` 对齐且逐轴位于 ``[0,32]``。
    ``probability_origin`` 是有限且无缺失值的 float32 ``(3,)`` XYZ 原点，单位 Å。
    ``probability_voxel_size`` 是有限、无缺失且逐轴为正的 float32 ``(3,)`` XYZ 体素尺寸，单位 Å/voxel。
    """

    candidate_indices: np.ndarray
    tokens: np.ndarray
    index_zyx: np.ndarray
    crop_start_zyx: np.ndarray
    probability_origin: np.ndarray
    probability_voxel_size: np.ndarray


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


def _build_pp_tokens(
    probability_map: np.ndarray,
    box_start_zyx: np.ndarray,
    crop_start_zyx: np.ndarray,
    voxel_size_xyz: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按 PP cache schema v2 构造 top-k，并返回 ``(K,4)`` token 与 ``(K,3)`` ZYX 索引。

    第 K 名概率并列时，入选子集沿用 NumPy ``argpartition`` 的既有结果；只对已经
    入选的体素按概率降序、展平索引升序排列。改变并列入选规则时必须提升
    ``PP_CACHE_SCHEMA_VERSION``，防止复用语义不同的缓存。
    """

    local_start = np.asarray(crop_start_zyx, dtype=np.int64)
    start = np.asarray(box_start_zyx, dtype=np.int64) + local_start
    stop = start + LOCAL_BOX_SIZE
    if start.shape != (3,) or np.any(start < 0) or np.any(stop > probability_map.shape):
        raise ValueError("PP 的 48³ 概率图裁剪越界。")
    slices = tuple(slice(int(first), int(last)) for first, last in zip(start, stop))
    crop = np.asarray(probability_map[slices], dtype=np.float32)
    flat = crop.reshape(-1)
    count = min(top_k, flat.size)
    selected = np.argpartition(flat, flat.size - count)[-count:]
    order = np.lexsort((selected, -flat[selected]))
    selected = selected[order]
    local_index = np.stack(np.unravel_index(selected, crop.shape), axis=1).astype(np.int64)
    index_in_80 = local_index + local_start
    coordinates = voxel_zyx_to_crop_centered_world(
        index_in_80,
        local_start,
        voxel_size_xyz,
    )
    coordinate_scale = (
        np.asarray(voxel_size_xyz, dtype=np.float32) * LOCAL_CONTEXT_HALF_EXTENT_VOXELS
    )
    tokens = np.concatenate(
        (flat[selected, None], coordinates / coordinate_scale[None, :]),
        axis=1,
    ).astype(np.float32, copy=False)
    if tokens.shape != (count, PP_TOKEN_DIM) or not np.isfinite(tokens).all():
        raise ValueError(f"PP token 必须为有限 ({count},{PP_TOKEN_DIM})，实际为 {tokens.shape}。")
    return tokens, local_index


def _load_pdb_cache(path: Path, top_k: int) -> _PpPdbCache:
    """读取并完整验证一个已经原子发布的 PDB PP 缓存。"""

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != _CACHE_FIELDS:
            raise ValueError(f"PP 缓存 {path} 的字段集合不符合 schema v{PP_CACHE_SCHEMA_VERSION}。")
        raw_schema_version = np.asarray(archive["schema_version"])
        raw_top_k = np.asarray(archive["top_k"])
        if (
            raw_schema_version.shape != ()
            or raw_schema_version.dtype != np.int64
            or raw_top_k.shape != ()
            or raw_top_k.dtype != np.int64
        ):
            raise ValueError(f"PP 缓存 {path} 的 schema_version/top_k 必须是 int64 标量。")
        schema_version = int(raw_schema_version.item())
        stored_top_k = int(raw_top_k.item())
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
    candidate_indices = cached["candidate_indices"]
    tokens = cached["tokens"]
    indices = cached["index_zyx"]
    crop_starts = cached["crop_start_zyx"]
    candidate_count = candidate_indices.shape[0]
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
    origin = cached["probability_origin"]
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
    """把一个 PDB 缓存写入同目录临时文件，刷盘后原子替换目标。"""

    path.parent.mkdir(parents=True, exist_ok=True)
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
    """按 PDB 缓存固定 ``top_k`` 的 PP token，并向 Dataset 返回候选切片。

    ``roots`` 给出只读 probability/F1 输入与可写 Matcher 缓存根。
    ``top_k`` 决定每个候选固定保留的最高概率体素数，并进入缓存路径。
    ``cache_bytes`` 是单个 DataLoader worker 的 PDB PP 数组 LRU 字节上限。
    """

    def __init__(
        self,
        roots: SourceRoots,
        top_k: int,
        crop_strategy: LocalCropStrategy,
        cache_bytes: int,
    ) -> None:
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
        """返回一个候选的 PP token、中心索引和构建时核验过的概率图几何。"""

        cached = self._load_or_build_pdb(split_name, pdb_id)
        candidate_indices = cached["candidate_indices"]
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
        """构建或严格复用一个 PDB 的缓存，并返回其中的候选数。"""

        return int(self._load_or_build_pdb(split_name, pdb_id)["candidate_indices"].size)

    def _load_or_build_pdb(self, split_name: str, pdb_id: str) -> _PpPdbCache:
        """命中 worker LRU，或在进程锁内构建并读取 PDB 缓存。"""

        key = (split_name, pdb_id)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        path = self._cache_path(split_name, pdb_id)
        if not path.is_file():
            with exclusive_file_lock(path.with_suffix(".lock")):
                if not path.is_file():
                    values = self._build_pdb(split_name, pdb_id)
                    _atomic_save_pdb_cache(path, values, self.top_k)
        cached = _load_pdb_cache(path, self.top_k)
        self._cache.put(key, cached)
        return cached

    def _build_pdb(self, split_name: str, pdb_id: str) -> _PpPdbCache:
        """只解压一次完整 probability map，顺序构造该 PDB 的全部候选 PP 输入。"""

        centered_path = (
            self.roots.stage1_root / split_name / pdb_id / "centered/F1_centered.npz"
        )
        with np.load(centered_path, allow_pickle=False) as archive:
            box_starts = np.asarray(archive["box_start_zyx"], dtype=np.int64)
            voxel_sizes = np.asarray(archive["voxel_size_world"], dtype=np.float32)
            voxel_offsets = np.asarray(archive["voxel_offsets"], dtype=np.int64)
            voxel_indices = np.asarray(archive["voxel_index_local_zyx"], dtype=np.int64)
        probability_path = (
            self.roots.stage1_root
            / split_name
            / pdb_id
            / "probability/probability_map.npz"
        )
        with np.load(probability_path, allow_pickle=False) as archive:
            probability_map = np.asarray(archive["probability_map"], dtype=np.float32)
            probability_origin = np.asarray(archive["origin_xyz"], dtype=np.float32)
            probability_voxel_size = np.asarray(archive["voxel_size_xyz"], dtype=np.float32)
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
        token_items: list[np.ndarray] = []
        index_items: list[np.ndarray] = []
        crop_start_items: list[np.ndarray] = []
        for candidate_index in range(candidate_count):
            if not np.array_equal(voxel_sizes[candidate_index], probability_voxel_size):
                raise ValueError(
                    f"{split_name}/{pdb_id}/candidate_{candidate_index} 的概率图体素尺寸不一致。"
                )
            voxel_start = int(voxel_offsets[candidate_index])
            voxel_stop = int(voxel_offsets[candidate_index + 1])
            crop_geometry = resolve_candidate_crop(
                voxel_indices[voxel_start:voxel_stop],
                self.crop_strategy,
            )
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
        return {
            "candidate_indices": np.arange(candidate_count, dtype=np.int64),
            "tokens": np.stack(token_items, axis=0),
            "index_zyx": np.stack(index_items, axis=0),
            "crop_start_zyx": np.stack(crop_start_items, axis=0),
            "probability_origin": probability_origin,
            "probability_voxel_size": probability_voxel_size,
        }

    def _cache_path(self, split_name: str, pdb_id: str) -> Path:
        """返回不依赖运行编号的版本化 PDB PP 缓存路径。"""

        return (
            self.roots.pp_cache_root(self.top_k, self.crop_strategy)
            / split_name
            / f"{pdb_id}.npz"
        )

    def __getstate__(self) -> dict[str, object]:
        """DataLoader 派生 worker 时不复制父进程已经读取的 PDB PP 数组。"""

        state = self.__dict__.copy()
        state["_cache"] = ByteBudgetLru(self.cache_bytes)
        return state
