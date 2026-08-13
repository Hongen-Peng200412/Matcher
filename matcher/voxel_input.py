"""构造并共享缓存 Matcher 候选级 ``48³`` 的 Find_0 体素输入。

每个候选缓存位于
``<output_root>/cache/voxel_input_v2/<strategy>/<split>/<pdb>/candidate_<index>_start_<z>_<y>_<x>.npy``。
顶层对象是 ``(105,48,48,48)`` float16 NPY：前 56 通道严格沿用 Find_0
密度顺序，后 49 通道把完整受体原子坐标向下取整到所属体素并在同体素内求和。
float16 是正式
缓存量化契约；文件名把确定性裁剪起点绑定到数组，禁止错误起点静默复用。
首次构造还会显式核对完整密度图与 F1-centered 候选框的原点、轴序和体素尺寸。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from .cache import ByteBudgetLru
from .density import DENSITY_CHANNEL_NAMES, build_density_channels
from .geometry import crop_local_48, hard_scatter_receptor, validate_crop_start
from .locking import exclusive_file_lock
from .schema import (
    BOX_ORIGIN_ABSOLUTE_TOLERANCE_ANGSTROM,
    LOCAL_BOX_SIZE,
    LOCAL_CROP_STRATEGIES,
    STAGE1_BOX_SIZE,
    LocalCropStrategy,
    SourceRoots,
)


VOXEL_INPUT_CHANNELS = len(DENSITY_CHANNEL_NAMES) + 49
VOXEL_INPUT_SHAPE = (
    VOXEL_INPUT_CHANNELS,
    LOCAL_BOX_SIZE,
    LOCAL_BOX_SIZE,
    LOCAL_BOX_SIZE,
)


class _CenteredGeometry(TypedDict):
    """保存 F1 候选轴上经过核验并规范化的几何字段；C 是候选数。

    ``box_start_zyx`` 是无缺失的 int64 ``(C,3)`` 完整密度图体素起点，轴序为 ZYX。
    ``box_origin_world`` 是有限且无缺失的 float32 ``(C,3)`` 候选角点原点，轴序为 XYZ，单位 Å。
    ``voxel_size_world`` 是有限、无缺失且逐轴为正的 float32 ``(C,3)`` 体素尺寸，轴序为 XYZ，单位 Å/voxel。
    """

    box_start_zyx: np.ndarray
    box_origin_world: np.ndarray
    voxel_size_world: np.ndarray


class _VoxelSources(TypedDict):
    """保存一个 PDB 构造 105 通道缓存所需的完整只读数组。

    ``centered`` 是符合 ``_CenteredGeometry`` 的 C 候选 F1 几何。
    ``experimental`` 是有限且无缺失的 float32 ``(Z,Y,X)`` 实验密度图，数值单位继承冻结源。
    ``simulated`` 是有限且无缺失的 float32 ``(Z,Y,X)`` 模拟密度图，形状与实验密度图相同，数值单位继承冻结源。
    ``voxel_size_xyz`` 是有限、无缺失且逐轴为正的 float32 ``(3,)`` 完整密度图体素尺寸，单位 Å/voxel。
    ``origin_xyz`` 是有限且无缺失的 float32 ``(3,)`` 完整密度图角点原点，单位 Å。
    ``receptor_coords`` 是有限且无缺失的 float32 ``(N,3)`` 受体原子 XYZ 坐标，单位 Å。
    ``receptor_features`` 是有限且无缺失的 float32 ``(N,49)`` 受体特征，第 n 项与 ``receptor_coords[n]`` 对齐。
    """

    centered: _CenteredGeometry
    experimental: np.ndarray
    simulated: np.ndarray
    voxel_size_xyz: np.ndarray
    origin_xyz: np.ndarray
    receptor_coords: np.ndarray
    receptor_features: np.ndarray


class VoxelInputStore:
    """按候选读取一种显式裁剪策略的 float16 缓存，缺失时从冻结源构造。

    ``roots`` 给出只读源与可写缓存根；``crop_strategy`` 进入缓存路径；
    ``source_cache_bytes`` 是单个进程完整密度图和受体数组的 LRU 字节上限。
    """

    def __init__(
        self,
        roots: SourceRoots,
        crop_strategy: LocalCropStrategy,
        source_cache_bytes: int = 1024**3,
    ) -> None:
        if source_cache_bytes < 0:
            raise ValueError("source_cache_bytes 不能为负数。")
        if crop_strategy not in LOCAL_CROP_STRATEGIES:
            raise ValueError(f"不支持 local_crop_strategy={crop_strategy!r}。")
        self.roots = roots
        self.crop_strategy = crop_strategy
        self.source_cache_bytes = int(source_cache_bytes)
        self._source_cache: ByteBudgetLru[tuple[str, str], _VoxelSources] = ByteBudgetLru(
            self.source_cache_bytes
        )

    def load(
        self,
        split_name: str,
        pdb_id: str,
        candidate_index: int,
        crop_start_zyx: np.ndarray,
    ) -> np.ndarray:
        """按调用方解析的候选起点返回 ``(105,48,48,48)`` float16 数组。"""

        validated_start = validate_crop_start(crop_start_zyx)
        path = self._cache_path(
            split_name,
            pdb_id,
            candidate_index,
            validated_start,
        )
        if path.is_file():
            return _load_cached(path)
        lock_path = path.with_suffix(".lock")
        with exclusive_file_lock(lock_path):
            if not path.is_file():
                value = self._build(
                    split_name,
                    pdb_id,
                    candidate_index,
                    validated_start,
                )
                _atomic_save_npy(path, value)
        return _load_cached(path)

    def _build(
        self,
        split_name: str,
        pdb_id: str,
        candidate_index: int,
        crop_start_zyx: np.ndarray,
    ) -> np.ndarray:
        """在完整 80³ 内复现 Find_0 通道，再按候选起点裁取并量化。"""

        sources = self._load_sources(split_name, pdb_id)
        centered = sources["centered"]
        candidate_count = int(centered["box_start_zyx"].shape[0])
        if not 0 <= candidate_index < candidate_count:
            raise IndexError(
                f"{split_name}/{pdb_id} candidate_index={candidate_index} 超出 [0,{candidate_count})。"
            )
        box_start = np.asarray(centered["box_start_zyx"][candidate_index], dtype=np.int64)
        experimental = _crop_source_grid(sources["experimental"], box_start)
        simulated = _crop_source_grid(sources["simulated"], box_start)

        box_origin_xyz = sources["origin_xyz"] + box_start[[2, 1, 0]] * sources["voxel_size_xyz"]
        atom_local_xyz = (
            (sources["receptor_coords"] - box_origin_xyz[None, :])
            / sources["voxel_size_xyz"][None, :]
        ).astype(np.float32, copy=False)
        atom_features = sources["receptor_features"]
        receptor_grid, receptor_hardmask = hard_scatter_receptor(atom_local_xyz, atom_features)
        density_grid = build_density_channels(experimental, simulated, receptor_hardmask)
        input_80 = np.concatenate((density_grid, receptor_grid), axis=0)
        input_48 = crop_local_48(input_80, crop_start_zyx)
        cached = input_48.astype(np.float16)
        if cached.shape != VOXEL_INPUT_SHAPE or not np.isfinite(cached).all():
            raise ValueError(
                f"{split_name}/{pdb_id}/{candidate_index} 的 105 通道量化结果非法："
                f"shape={cached.shape}, finite={bool(np.isfinite(cached).all())}。"
            )
        return cached

    def _load_sources(self, split_name: str, pdb_id: str) -> _VoxelSources:
        """按 ``_VoxelSources`` 契约缓存最近使用的 PDB 密度、F1 几何和受体字段。"""

        key = (split_name, pdb_id)
        cached = self._source_cache.get(key)
        if cached is not None:
            return cached
        centered_path = (
            self.roots.stage1_root / split_name / pdb_id / "centered/F1_centered.npz"
        )
        with np.load(centered_path, allow_pickle=False) as archive:
            raw_box_start_zyx = np.asarray(archive["box_start_zyx"]).copy()
            raw_box_origin_world = np.asarray(archive["box_origin_world"]).copy()
            raw_voxel_size_world = np.asarray(archive["voxel_size_world"]).copy()
        experimental, voxel_size_xyz, origin_xyz = _load_full_grid(
            self.roots.data_root / "density" / pdb_id / "exp.npz"
        )
        simulated, sim_voxel_size_xyz, sim_origin_xyz = _load_full_grid(
            self.roots.data_root / "density" / pdb_id / "sim.npz"
        )
        if (
            experimental.shape != simulated.shape
            or not np.array_equal(voxel_size_xyz, sim_voxel_size_xyz)
            or not np.array_equal(origin_xyz, sim_origin_xyz)
        ):
            raise ValueError(
                f"{pdb_id} 的 exp/sim 完整密度图几何不一致。"
            )
        box_start_zyx = np.asarray(raw_box_start_zyx)
        box_origin_world = np.asarray(raw_box_origin_world, dtype=np.float32)
        centered_voxel_size = np.asarray(raw_voxel_size_world, dtype=np.float32)
        if (
            box_start_zyx.ndim != 2
            or box_start_zyx.shape[1:] != (3,)
            or not np.issubdtype(box_start_zyx.dtype, np.integer)
        ):
            raise ValueError(f"{centered_path}:box_start_zyx 必须为整数 (C,3)。")
        candidate_count = int(box_start_zyx.shape[0])
        if box_origin_world.shape != (candidate_count, 3):
            raise ValueError(f"{centered_path}:box_origin_world 必须为 (C,3)。")
        if centered_voxel_size.shape != (candidate_count, 3):
            raise ValueError(f"{centered_path}:voxel_size_world 必须为 (C,3)。")
        if (
            not np.isfinite(box_origin_world).all()
            or not np.isfinite(centered_voxel_size).all()
            or np.any(centered_voxel_size <= 0)
        ):
            raise ValueError(f"{centered_path} 的候选原点或体素尺寸含非法值。")
        expected_voxel_size = np.broadcast_to(voxel_size_xyz, centered_voxel_size.shape)
        if not np.array_equal(centered_voxel_size, expected_voxel_size):
            raise ValueError(f"{pdb_id} 的完整密度图与 F1-centered voxel_size 不一致。")
        expected_box_origin = (
            origin_xyz.astype(np.float64)[None, :]
            + box_start_zyx[:, [2, 1, 0]].astype(np.float64)
            * voxel_size_xyz.astype(np.float64)[None, :]
        )
        maximum_absolute_error = float(
            np.max(np.abs(box_origin_world.astype(np.float64) - expected_box_origin), initial=0.0)
        )
        if not np.allclose(
            box_origin_world,
            expected_box_origin,
            rtol=0.0,
            atol=BOX_ORIGIN_ABSOLUTE_TOLERANCE_ANGSTROM,
        ):
            raise ValueError(
                f"{pdb_id} 的完整密度图与 F1-centered box_origin 不一致；"
                f"最大绝对误差为 {maximum_absolute_error:.9g} Å，允许上限为"
                f" {BOX_ORIGIN_ABSOLUTE_TOLERANCE_ANGSTROM:.9g} Å。"
            )
        centered: _CenteredGeometry = {
            "box_start_zyx": box_start_zyx.astype(np.int64, copy=False),
            "box_origin_world": box_origin_world,
            "voxel_size_world": centered_voxel_size,
        }
        receptor_path = self.roots.data_root / "parse" / pdb_id / "receptor_tokens.npz"
        with np.load(receptor_path, allow_pickle=False) as archive:
            receptor_coords = np.asarray(archive["coords"], dtype=np.float32)
            receptor_features = np.asarray(archive["feat"], dtype=np.float32)
        if receptor_coords.ndim != 2 or receptor_coords.shape[1] != 3:
            raise ValueError(f"{receptor_path}:coords 必须为 (N,3)。")
        if receptor_features.shape != (receptor_coords.shape[0], 49):
            raise ValueError(f"{receptor_path}:feat 必须为 (N,49)。")
        if not np.isfinite(receptor_coords).all() or not np.isfinite(receptor_features).all():
            raise ValueError(f"{receptor_path} 的受体坐标或特征含 NaN/Inf。")
        sources: _VoxelSources = {
            "centered": centered,
            "experimental": experimental,
            "simulated": simulated,
            "voxel_size_xyz": voxel_size_xyz,
            "origin_xyz": origin_xyz,
            "receptor_coords": receptor_coords,
            "receptor_features": receptor_features,
        }
        self._source_cache.put(key, sources)
        return sources

    def _cache_path(
        self,
        split_name: str,
        pdb_id: str,
        candidate_index: int,
        crop_start_zyx: np.ndarray,
    ) -> Path:
        """返回同时冻结候选编号和 ZYX 裁剪起点的版本化缓存路径。"""

        start = validate_crop_start(crop_start_zyx)
        start_suffix = "_".join(str(int(value)) for value in start)
        return (
            self.roots.voxel_cache_root(self.crop_strategy)
            / split_name
            / pdb_id
            / f"candidate_{candidate_index}_start_{start_suffix}.npy"
        )

    def __getstate__(self) -> dict[str, Any]:
        """DataLoader 派生 worker 时不复制父进程的大数组缓存。"""

        state = self.__dict__.copy()
        state["_source_cache"] = ByteBudgetLru(self.source_cache_bytes)
        return state


# ================================================================================================


def _load_full_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """依次返回 float32 ZYX 密度图、XYZ Å/voxel 尺寸和 XYZ Å 角点原点。"""

    with np.load(path, allow_pickle=False) as archive:
        value = np.asarray(archive["grid"], dtype=np.float32)
        voxel_size = np.asarray(archive["voxel_size"], dtype=np.float32)
        origin = np.asarray(archive["origin"], dtype=np.float32)
    if value.ndim != 4 or value.shape[0] != 1 or not np.isfinite(value).all():
        raise ValueError(f"{path}:grid 必须为有限 (1,Z,Y,X)，实际为 {value.shape}。")
    if (
        voxel_size.shape != (3,)
        or origin.shape != (3,)
        or not np.isfinite(voxel_size).all()
        or not np.isfinite(origin).all()
        or np.any(voxel_size <= 0)
    ):
        raise ValueError(f"{path} 的 voxel_size/origin 空间字段非法。")
    return value[0], voxel_size, origin


def _crop_source_grid(grid: np.ndarray, start_zyx: np.ndarray) -> np.ndarray:
    """从完整 ZYX 密度图裁取严格位于边界内的 80³ 候选框。"""

    start = np.asarray(start_zyx, dtype=np.int64)
    stop = start + STAGE1_BOX_SIZE
    if start.shape != (3,) or np.any(start < 0) or np.any(stop > np.asarray(grid.shape)):
        raise ValueError(
            f"80³ 裁剪越界：start={start.tolist()}, stop={stop.tolist()}, shape={grid.shape}。"
        )
    slices = tuple(slice(int(first), int(last)) for first, last in zip(start, stop))
    return np.asarray(grid[slices], dtype=np.float32)


def _load_cached(path: Path) -> np.ndarray:
    """读取并核验一个已经原子发布的候选体素缓存。"""

    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float16 or value.shape != VOXEL_INPUT_SHAPE:
        raise ValueError(
            f"体素缓存 {path} 必须为 float16 {VOXEL_INPUT_SHAPE}，"
            f"实际为 {value.dtype} {value.shape}。"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"体素缓存 {path} 含 NaN 或 Inf。")
    return value


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    """把 NPY 写入同目录临时文件并原子替换目标。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
