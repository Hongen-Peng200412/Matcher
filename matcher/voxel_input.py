"""构造并共享缓存 Matcher 候选中心 48³ 的 Find_0 105 通道体素输入。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .density import DENSITY_CHANNEL_NAMES, build_density_channels
from .geometry import crop_center_48, hard_scatter_receptor
from .locking import exclusive_file_lock
from .schema import LOCAL_BOX_SIZE, STAGE1_BOX_SIZE, SourceRoots


VOXEL_INPUT_CHANNELS = len(DENSITY_CHANNEL_NAMES) + 49
VOXEL_INPUT_SHAPE = (
    VOXEL_INPUT_CHANNELS,
    LOCAL_BOX_SIZE,
    LOCAL_BOX_SIZE,
    LOCAL_BOX_SIZE,
)


class VoxelInputStore:
    """按候选读取 float16 缓存，首次缺失时从 A–G 和 F1-centered 原子构造。"""

    def __init__(self, roots: SourceRoots) -> None:
        self.roots = roots
        self._source_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def load(self, split_name: str, pdb_id: str, candidate_index: int) -> np.ndarray:
        """返回一个只读语义的 ``(105,48,48,48)`` float16 数组。"""

        path = self._cache_path(split_name, pdb_id, candidate_index)
        if path.is_file():
            return _load_cached(path)
        lock_path = path.with_suffix(".lock")
        with exclusive_file_lock(lock_path):
            if not path.is_file():
                value = self._build(split_name, pdb_id, candidate_index)
                _atomic_save_npy(path, value)
        return _load_cached(path)

    def _build(self, split_name: str, pdb_id: str, candidate_index: int) -> np.ndarray:
        """在 80³ 内复现 Find_0 输入，再裁取中心 48³ 并量化成 float16。"""

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
        input_48 = crop_center_48(input_80)
        cached = input_48.astype(np.float16)
        if cached.shape != VOXEL_INPUT_SHAPE or not np.isfinite(cached).all():
            raise ValueError(
                f"{split_name}/{pdb_id}/{candidate_index} 的 105 通道量化结果非法："
                f"shape={cached.shape}, finite={bool(np.isfinite(cached).all())}。"
            )
        return cached

    def _load_sources(self, split_name: str, pdb_id: str) -> dict[str, Any]:
        """在当前进程缓存最近两个 PDB 的完整密度图与 F1-centered 必要字段。"""

        key = (split_name, pdb_id)
        cached = self._source_cache.get(key)
        if cached is not None:
            return cached
        centered_path = (
            self.roots.stage1_root / split_name / pdb_id / "centered/F1_centered.npz"
        )
        with np.load(centered_path, allow_pickle=False) as archive:
            centered = {
                name: np.asarray(archive[name]).copy()
                for name in (
                    "box_start_zyx",
                )
            }
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
        receptor_path = self.roots.data_root / "parse" / pdb_id / "receptor_tokens.npz"
        with np.load(receptor_path, allow_pickle=False) as archive:
            receptor_coords = np.asarray(archive["coords"], dtype=np.float32)
            receptor_features = np.asarray(archive["feat"], dtype=np.float32)
        if receptor_coords.ndim != 2 or receptor_coords.shape[1] != 3:
            raise ValueError(f"{receptor_path}:coords 必须为 (N,3)。")
        if receptor_features.shape != (receptor_coords.shape[0], 49):
            raise ValueError(f"{receptor_path}:feat 必须为 (N,49)。")
        cached = {
            "centered": centered,
            "experimental": experimental,
            "simulated": simulated,
            "voxel_size_xyz": voxel_size_xyz,
            "origin_xyz": origin_xyz,
            "receptor_coords": receptor_coords,
            "receptor_features": receptor_features,
        }
        self._source_cache[key] = cached
        while len(self._source_cache) > 2:
            del self._source_cache[next(iter(self._source_cache))]
        return cached

    def _cache_path(self, split_name: str, pdb_id: str, candidate_index: int) -> Path:
        """返回不依赖运行编号的版本化候选缓存路径。"""

        return (
            self.roots.voxel_cache_root
            / split_name
            / pdb_id
            / f"candidate_{candidate_index}.npy"
        )

    def __getstate__(self) -> dict[str, Any]:
        """DataLoader 派生 worker 时不复制父进程的大数组缓存。"""

        state = self.__dict__.copy()
        state["_source_cache"] = {}
        return state


def _load_full_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 A–G 单通道完整密度图及其 XYZ 体素尺寸和角点原点。"""

    with np.load(path, allow_pickle=False) as archive:
        value = np.asarray(archive["grid"], dtype=np.float32)
        voxel_size = np.asarray(archive["voxel_size"], dtype=np.float32)
        origin = np.asarray(archive["origin"], dtype=np.float32)
    if value.ndim != 4 or value.shape[0] != 1 or not np.isfinite(value).all():
        raise ValueError(f"{path}:grid 必须为有限 (1,D,H,W)，实际为 {value.shape}。")
    if voxel_size.shape != (3,) or origin.shape != (3,) or np.any(voxel_size <= 0):
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
