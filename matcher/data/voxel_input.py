"""
从完整密度、F1 候选几何和受体原子表构造 Matcher 候选级 105 通道 48³ 体素输入，并原子发布可共享缓存。

主要入口:
    - VoxelInputStore.load: 按 split、PDB、candidate_index、局部裁剪策略和实际 ZYX 起点读取或构建一个候选体素数组。

落盘产物:
    - output_root/cache/voxel_input_v{VOXEL_CACHE_SCHEMA_VERSION}/{crop_strategy}/{split_name}/{pdb_id}/candidate_{candidate_index}_start_{z}_{y}_{x}.npy: float16 NPY，形状固定为 (105, 48, 48, 48)。

输出前 56 通道与 DENSITY_CHANNEL_NAMES 逐位置一致，后 49 通道把 80³ 内受体原子的原始 49 维特征按 floor 后的体素归属逐通道求和。float16 是正式缓存量化契约；文件名同时绑定候选编号和实际裁剪起点，防止不同起点静默复用同一数组。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from ..utils.cache import ByteBudgetLru
from .density import DENSITY_CHANNEL_NAMES, build_density_channels
from .geometry import crop_local_48, hard_scatter_receptor, validate_crop_start
from ..utils.locking import exclusive_file_lock
from .schema import (
    BOX_ORIGIN_ABSOLUTE_TOLERANCE_ANGSTROM,
    LOCAL_BOX_SIZE,
    LOCAL_CROP_STRATEGIES,
    STAGE1_BOX_SIZE,
    LocalCropStrategy,
    SourceRoots,
)


# int, 候选体素输入总通道数 105；前 56 个密度通道加后 49 个受体特征通道。
VOXEL_INPUT_CHANNELS = len(DENSITY_CHANNEL_NAMES) + 49
# tuple[int, int, int, int], 正式缓存形状；轴顺序依次为通道、Z、Y、X。
VOXEL_INPUT_SHAPE = (
    VOXEL_INPUT_CHANNELS,
    LOCAL_BOX_SIZE,
    LOCAL_BOX_SIZE,
    LOCAL_BOX_SIZE,
)


class _CenteredGeometry(TypedDict):
    """
    保存 F1_centered.npz 候选轴上经过核验和 dtype 规范化的三个几何字段。

    形状符号:
        - C: 当前 PDB 的 F1 候选数量。

    字段:
        - box_start_zyx: int64 数组，(C, 3)；每个候选 80³ 在完整密度图中的离散起点，最后一维依次为 Z、Y、X，无缺失值。
        - box_origin_world: float32 数组，(C, 3)；每个候选 80³ 低角点的世界坐标，最后一维依次为 X、Y、Z，单位 Å，全部有限且无缺失值。
        - voxel_size_world: float32 数组，(C, 3)；每个候选 XYZ 三轴体素尺寸，单位 Å/voxel，全部有限、无缺失且逐轴大于 0。

    box_origin_world 必须在绝对误差容限内等于完整密度 origin_xyz + box_start_zyx 转为 XYZ后乘 voxel_size_xyz；voxel_size_world 必须与完整密度体素尺寸逐元素相同。
    """

    box_start_zyx: np.ndarray
    box_origin_world: np.ndarray
    voxel_size_world: np.ndarray


class _VoxelSources(TypedDict):
    """
    保存一个 PDB 构造所有候选 105 通道缓存所需的完整数组；本模块只读取这些数组，但数组本身未设置写保护。

    形状符号:
        - C: 当前 PDB 的 F1 候选数量。
        - Z_full: 完整密度图 Z 轴长度。
        - Y_full: 完整密度图 Y 轴长度。
        - X_full: 完整密度图 X 轴长度。
        - N_atom: 当前 PDB 受体重原子数量。

    字段:
        - centered: _CenteredGeometry；C 个 F1 候选的 80³ 起点、世界原点和体素尺寸。
        - centered.box_start_zyx: int64 数组，(C, 3)；候选 80³ 在完整密度图中的 ZYX 起点。
        - centered.box_origin_world: float32 数组，(C, 3)；候选 80³ 低角点的世界 XYZ 坐标，单位 Å。
        - centered.voxel_size_world: float32 数组，(C, 3)；候选 XYZ 体素尺寸，单位 Å/voxel。
        - experimental: float32 数组，(Z_full, Y_full, X_full)；实验密度原始幅值，轴顺序为 Z、Y、X，全部有限且无缺失值。
        - simulated: float32 数组，(Z_full, Y_full, X_full)；受体模拟密度，形状、体素尺寸和原点与 experimental 完全相同，全部有限且无缺失值。
        - voxel_size_xyz: float32 数组，(3,)；完整密度图 XYZ 三轴体素尺寸，单位 Å/voxel，全部有限、无缺失且逐轴大于 0。
        - origin_xyz: float32 数组，(3,)；完整密度图低角点的世界 XYZ 坐标，单位 Å，全部有限且无缺失值。
        - receptor_coords: float32 数组，(N_atom, 3)；受体重原子世界坐标，最后一维依次为 X、Y、Z，单位 Å，全部有限且无缺失值。
        - receptor_features: float32 数组，(N_atom, 49)；与 receptor_coords 第一维逐原子对齐。
        - receptor_features[:, 0:6]: float32 数组，(N_atom, 6)；C、N、O、S、P、X 元素独热。
        - receptor_features[:, 6:31]: float32 数组，(N_atom, 25)；残基独热顺序为 ALA、ARG、ASN、ASP、CYS、GLN、GLU、GLY、HIS、ILE、LEU、LYS、MET、PHE、PRO、SER、THR、TRP、TYR、VAL、A、U、C、G、X。
        - receptor_features[:, 31:39]: float32 数组，(N_atom, 8)；依次为极性、非极性、酸性、碱性、中性、正电、负电、无电标志。
        - receptor_features[:, 39]: float32 数组，(N_atom,)；元素质量除以 32 的标量，未知 X 使用 14.0/32.0。
        - receptor_features[:, 40]: float32 数组，(N_atom,)；2 Å 内其它受体原子数的 log1p，已扣除自身。
        - receptor_features[:, 41:49]: float32 数组，(N_atom, 8)；依次为 (2,4] 至 (16,18] Å 八个 2 Å 壳层计数的 log1p。
    """

    centered: _CenteredGeometry
    experimental: np.ndarray
    simulated: np.ndarray
    voxel_size_xyz: np.ndarray
    origin_xyz: np.ndarray
    receptor_coords: np.ndarray
    receptor_features: np.ndarray


class VoxelInputStore:
    """
    按候选读取一种显式裁剪策略的 float16 体素缓存，缺失时从冻结来源构造。

    构造参数:
        - roots: SourceRoots；data_root 提供完整密度和受体原子表，stage1_root 提供 F1 候选几何，output_root 提供可写缓存根。
        - roots.data_root: Path；读取 density/{pdb_id}/exp.npz、sim.npz 和 parse/{pdb_id}/receptor_tokens.npz 的冻结 A–G 根。
        - roots.stage1_root: Path；读取 {split_name}/{pdb_id}/centered/F1_centered.npz 的 Stage1 根。
        - roots.output_root: Path；派生版本化候选体素缓存目录的可写 Matcher 输出根。
        - crop_strategy: LocalCropStrategy；进入缓存目录，用于隔离 fixed_center 与 v_centroid 的 48³ 输入语义。
        - source_cache_bytes: int；当前进程完整 PDB 来源数组的 LRU 最大字节数，默认 1 GiB，必须非负。

    缓存字段:
        - _source_cache: ByteBudgetLru[tuple[str, str], _VoxelSources]；键依次为 split_name 和 pdb_id，值包含该 PDB 可被全部候选复用的完整密度、F1 几何与受体数组。
        - _source_cache.value.centered: _CenteredGeometry；候选轴上的 F1 起点、世界原点和体素尺寸。
        - _source_cache.value.centered.box_start_zyx: int64 数组，(C, 3)；候选 80³ 的完整网格 ZYX 起点。
        - _source_cache.value.centered.box_origin_world: float32 数组，(C, 3)；候选低角点世界 XYZ 坐标，单位 Å。
        - _source_cache.value.centered.voxel_size_world: float32 数组，(C, 3)；候选 XYZ 体素尺寸，单位 Å/voxel。
        - _source_cache.value.experimental: float32 数组，(Z_full, Y_full, X_full)；完整实验密度。
        - _source_cache.value.simulated: float32 数组，(Z_full, Y_full, X_full)；完整受体模拟密度。
        - _source_cache.value.voxel_size_xyz: float32 数组，(3,)；完整密度 XYZ 体素尺寸，单位 Å/voxel。
        - _source_cache.value.origin_xyz: float32 数组，(3,)；完整密度低角点世界 XYZ 坐标，单位 Å。
        - _source_cache.value.receptor_coords: float32 数组，(N_atom, 3)；受体重原子世界 XYZ 坐标，单位 Å。
        - _source_cache.value.receptor_features: float32 数组，(N_atom, 49)；与受体坐标逐原子对齐的特征。

    本类读取冻结来源并写入版本化 NPY；不决定 48³ 起点，调用者必须传入由 resolve_candidate_crop 解析的 crop_start_zyx。
    """

    def __init__(
        self,
        roots: SourceRoots,
        crop_strategy: LocalCropStrategy,
        source_cache_bytes: int = 1024**3,
    ) -> None:
        """
        建立绑定局部裁剪策略的候选体素缓存读取器。

        输入参数:
            - roots: SourceRoots；提供完整密度、受体原子表、F1 候选几何和可写缓存根。
            - roots.data_root: Path；冻结完整密度与受体原子表的共同根。
            - roots.stage1_root: Path；Stage1 F1 候选几何根。
            - roots.output_root: Path；版本化候选体素缓存的可写输出根。
            - crop_strategy: LocalCropStrategy；fixed_center 或 v_centroid，决定缓存目录分区；实际起点仍由每次 load 显式传入。
            - source_cache_bytes: int；当前进程完整 PDB 来源数组 LRU 的最大估算字节数，默认 1 GiB，必须非负；0 表示 put 后不能长期保留条目。

        实例字段:
            - roots: SourceRoots；冻结来源与版本化输出路径集合。
            - roots.data_root: Path；冻结完整密度与受体原子表的共同根。
            - roots.stage1_root: Path；Stage1 F1 候选几何根。
            - roots.output_root: Path；版本化候选体素缓存的可写输出根。
            - crop_strategy: LocalCropStrategy；当前实例唯一的缓存裁剪策略。
            - source_cache_bytes: int；规范化后的进程内来源缓存预算。
            - _source_cache: ByteBudgetLru[tuple[str, str], _VoxelSources]；键依次为 split_name、pdb_id，值为该 PDB 的完整密度、F1 几何和受体数组。
            - _source_cache.value.centered: _CenteredGeometry；候选轴上的三个 F1 几何数组。
            - _source_cache.value.centered.box_start_zyx: int64 数组，(C, 3)；候选 80³ 的完整网格 ZYX 起点。
            - _source_cache.value.centered.box_origin_world: float32 数组，(C, 3)；候选低角点世界 XYZ 坐标，单位 Å。
            - _source_cache.value.centered.voxel_size_world: float32 数组，(C, 3)；候选 XYZ 体素尺寸，单位 Å/voxel。
            - _source_cache.value.experimental: float32 数组，(Z_full, Y_full, X_full)；完整实验密度。
            - _source_cache.value.simulated: float32 数组，(Z_full, Y_full, X_full)；完整模拟密度。
            - _source_cache.value.voxel_size_xyz: float32 数组，(3,)；完整密度 XYZ 体素尺寸。
            - _source_cache.value.origin_xyz: float32 数组，(3,)；完整密度世界 XYZ 原点。
            - _source_cache.value.receptor_coords: float32 数组，(N_atom, 3)；受体世界 XYZ 坐标。
            - _source_cache.value.receptor_features: float32 数组，(N_atom, 49)；逐受体原子特征。
        """

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
        """
        按调用者解析的候选起点返回一个 105 通道 48³ float16 数组。

        输入参数:
            - split_name: str；Stage1 来源和缓存中的数据划分目录名。
            - pdb_id: str；当前 PDB 的来源目录名和缓存父目录名。
            - candidate_index: int；索引 F1_centered.npz 的候选第一维。
            - crop_start_zyx: 整数数组，(3,)；当前候选 48³ 在 F1 80³ 内的实际 ZYX 起点，必须逐轴位于 [0, 32]。

        返回值:
            - voxel_input: float16 数组，(105, 48, 48, 48)；轴顺序为通道、Z、Y、X，全部有限。

        并发与路径:
            - cache_path: Path；文件名同时包含 candidate_index 和 crop_start_zyx，命中时直接严格读取。
            - lock_path: Path；与 cache_path 同名但后缀为 .lock，缺失缓存时在进程锁内二次检查并原子发布。
        """

        # int64, (3,), 已验证逐轴位于 [0, 32] 的实际 ZYX 裁剪起点。
        validated_start = validate_crop_start(crop_start_zyx)
        # Path, 与 schema、策略、划分、PDB、候选编号和实际裁剪起点共同绑定的最终 NPY 路径。
        path = self._cache_path(
            split_name,
            pdb_id,
            candidate_index,
            validated_start,
        )
        if path.is_file():
            return _load_cached(path)
        # Path, 当前候选缓存的进程级锁文件；不承载科学字段。
        lock_path = path.with_suffix(".lock")
        with exclusive_file_lock(lock_path):
            if not path.is_file():
                # float16，(105, 48, 48, 48)，本次准备原子发布的候选体素输入，轴顺序为通道、Z、Y、X。
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
        """
        从冻结来源构造一个候选的 80³ 通道，再裁成 48³ 并量化为正式缓存。

        输入参数:
            - split_name: str；Stage1 候选几何所在的数据划分目录名。
            - pdb_id: str；当前 PDB 的密度、受体原子表和 F1 候选目录名。
            - candidate_index: int；当前候选在 F1_centered.npz 候选第一维中的零基索引，必须位于 [0, C)。
            - crop_start_zyx: int64 数组，(3,)；48³ 在当前候选 80³ 内的实际 ZYX 起点，调用前已经通过 validate_crop_start 核验。

        返回值:
            - cached: float16 数组，(105, 48, 48, 48)；前 56 个通道是 build_density_channels 的密度派生量，后 49 个通道是受体原子特征的硬体素求和，全部有限。

        构造关系:
            - experimental: float32 数组，(80, 80, 80)；从完整实验密度图按当前候选 box_start_zyx 裁得。
            - simulated: float32 数组，(80, 80, 80)；与 experimental 使用相同起点裁得的受体模拟密度。
            - receptor_hardmask: bool 数组，(80, 80, 80)；80³ 内至少含一个受体原子的体素标记，供密度归一化统计使用。
            - input_80: float32 数组，(105, 80, 80, 80)；密度与受体通道沿通道轴拼接的未量化候选输入。
        """

        # _VoxelSources, 包含当前 PDB 全部候选共享的完整密度、候选几何和受体原子数组。
        sources = self._load_sources(split_name, pdb_id)
        # _CenteredGeometry, C 个候选的 F1 80³ 起点、世界原点和体素尺寸。
        centered = sources["centered"]
        # int, F1_centered.npz 候选第一维的长度 C。
        candidate_count = int(centered["box_start_zyx"].shape[0])
        if not 0 <= candidate_index < candidate_count:
            raise IndexError(
                f"{split_name}/{pdb_id} candidate_index={candidate_index} 超出 [0,{candidate_count})。"
            )
        # int64, (3,), 当前候选 80³ 在完整密度图中的 ZYX 起点。
        box_start = np.asarray(centered["box_start_zyx"][candidate_index], dtype=np.int64)
        # float32, (80, 80, 80), 当前候选实验密度；轴顺序为 Z、Y、X。
        experimental = _crop_source_grid(sources["experimental"], box_start)
        # float32, (80, 80, 80), 与 experimental 空间对齐的受体模拟密度。
        simulated = _crop_source_grid(sources["simulated"], box_start)

        # float32, (3,), 当前 80³ 低角点的世界 XYZ 坐标，单位 Å。
        box_origin_xyz = sources["origin_xyz"] + box_start[[2, 1, 0]] * sources["voxel_size_xyz"]
        # float32, (N_atom, 3), 受体原子相对 80³ 低角点的连续 XYZ 体素坐标。
        atom_local_xyz = (
            (sources["receptor_coords"] - box_origin_xyz[None, :])
            / sources["voxel_size_xyz"][None, :]
        ).astype(np.float32, copy=False)
        # float32, (N_atom, 49), 与 atom_local_xyz 第一维逐原子对齐的受体特征。
        atom_features = sources["receptor_features"]
        # float32，(49, 80, 80, 80)，受体原子特征按离散体素逐通道求和的网格。
        # bool，(80, 80, 80)，至少落入一个受体原子的体素标记，供密度拟合选择受体区域。
        receptor_grid, receptor_hardmask = hard_scatter_receptor(atom_local_xyz, atom_features)
        # float32, (56, 80, 80, 80), 顺序与 DENSITY_CHANNEL_NAMES 完全一致。
        density_grid = build_density_channels(experimental, simulated, receptor_hardmask)
        # float32, (105, 80, 80, 80), 沿通道轴拼接密度 56 通道与受体 49 通道。
        input_80 = np.concatenate((density_grid, receptor_grid), axis=0)
        # float32, (105, 48, 48, 48), 按 crop_start_zyx 从 input_80 裁取。
        input_48 = crop_local_48(input_80, crop_start_zyx)
        # float16, (105, 48, 48, 48), 正式缓存量化结果；量化后重新核验有限性。
        cached = input_48.astype(np.float16)
        if cached.shape != VOXEL_INPUT_SHAPE or not np.isfinite(cached).all():
            raise ValueError(
                f"{split_name}/{pdb_id}/{candidate_index} 的 105 通道量化结果非法："
                f"shape={cached.shape}, finite={bool(np.isfinite(cached).all())}。"
            )
        return cached

    def _load_sources(self, split_name: str, pdb_id: str) -> _VoxelSources:
        """
        读取并交叉核验一个 PDB 的完整密度、F1 候选几何与受体原子表。

        输入参数:
            - split_name: str；F1_centered.npz 所在的数据划分目录名，同时参与进程内 LRU 键。
            - pdb_id: str；密度、F1 候选和受体原子表的共同 PDB 目录名，同时参与进程内 LRU 键。

        来源字段:
            - F1_centered.npz.box_start_zyx: 整数数组，(C, 3)；C 个候选 80³ 在完整密度图中的 ZYX 起点。
            - F1_centered.npz.box_origin_world: 数值数组，(C, 3)；C 个候选低角点的世界 XYZ 坐标，单位 Å。
            - F1_centered.npz.voxel_size_world: 数值数组，(C, 3)；C 个候选的 XYZ 体素尺寸，单位 Å/voxel。
            - exp.npz.grid: 数值数组，(1, Z_full, Y_full, X_full)；完整实验密度。
            - exp.npz.voxel_size: 数值数组，(3,)；完整实验密度的 XYZ 体素尺寸，单位 Å/voxel。
            - exp.npz.origin: 数值数组，(3,)；完整实验密度低角点的世界 XYZ 坐标，单位 Å。
            - sim.npz.grid: 数值数组，(1, Z_full, Y_full, X_full)；完整受体模拟密度。
            - sim.npz.voxel_size: 数值数组，(3,)；完整模拟密度的 XYZ 体素尺寸，必须与实验密度相同。
            - sim.npz.origin: 数值数组，(3,)；完整模拟密度低角点的世界 XYZ 坐标，必须与实验密度相同。
            - receptor_tokens.npz.coords: 数值数组，(N_atom, 3)；受体重原子世界 XYZ 坐标，单位 Å。
            - receptor_tokens.npz.feat: 数值数组，(N_atom, 49)；与 coords 第一维逐原子对齐的受体特征。

        返回值:
            - sources: _VoxelSources；完整密度几何、F1 候选几何和受体数组均已核验，可供该 PDB 全部候选共享。
            - sources.centered: _CenteredGeometry；候选轴上的 F1 几何对象。
            - sources.centered.box_start_zyx: int64 数组，(C, 3)；候选 80³ 的完整网格 ZYX 起点。
            - sources.centered.box_origin_world: float32 数组，(C, 3)；候选低角点世界 XYZ 坐标，单位 Å。
            - sources.centered.voxel_size_world: float32 数组，(C, 3)；候选 XYZ 体素尺寸，单位 Å/voxel。
            - sources.experimental: float32 数组，(Z_full, Y_full, X_full)；完整实验密度。
            - sources.simulated: float32 数组，(Z_full, Y_full, X_full)；与实验密度几何一致的完整模拟密度。
            - sources.voxel_size_xyz: float32 数组，(3,)；完整密度 XYZ 体素尺寸，单位 Å/voxel。
            - sources.origin_xyz: float32 数组，(3,)；完整密度低角点世界 XYZ 坐标，单位 Å。
            - sources.receptor_coords: float32 数组，(N_atom, 3)；受体重原子世界 XYZ 坐标，单位 Å。
            - sources.receptor_features: float32 数组，(N_atom, 49)；与受体坐标逐原子对齐的特征。

        几何门禁:
            exp.npz 与 sim.npz 的网格形状、voxel_size 和 origin 必须完全相同。
            F1 的 voxel_size_world 必须逐候选等于完整密度 voxel_size。
            F1 的 box_origin_world 必须在 BOX_ORIGIN_ABSOLUTE_TOLERANCE_ANGSTROM 内等于 origin_xyz 加 box_start_zyx 转为 XYZ 后乘 voxel_size_xyz。
        """

        # tuple[str, str], 进程内来源缓存键，依次为数据划分名和 PDB 标识。
        key = (split_name, pdb_id)
        # _VoxelSources | None，命中时复用当前 PDB 的完整数组；本模块不修改这些数组。
        cached = self._source_cache.get(key)
        if cached is not None:
            return cached
        # Path, 当前 PDB 的 F1 候选几何产物。
        centered_path = (
            self.roots.stage1_root / split_name / pdb_id / "centered/F1_centered.npz"
        )
        with np.load(centered_path, allow_pickle=False) as archive:
            # 整数数组，(C, 3)，候选 80³ 在完整密度图中的 ZYX 起点；复制后不依赖已关闭的 NPZ。
            raw_box_start_zyx = np.asarray(archive["box_start_zyx"]).copy()
            # 数值数组，(C, 3)，候选 80³ 低角点的世界 XYZ 坐标，单位 Å；稍后规范为 float32。
            raw_box_origin_world = np.asarray(archive["box_origin_world"]).copy()
            # 数值数组，(C, 3)，候选 XYZ 三轴体素尺寸，单位 Å/voxel；稍后规范为 float32。
            raw_voxel_size_world = np.asarray(archive["voxel_size_world"]).copy()
        # float32，(Z_full, Y_full, X_full)，完整实验密度，轴顺序为 Z、Y、X。
        # float32，(3,)，实验密度 XYZ 三轴体素尺寸，单位 Å/voxel。
        # float32，(3,)，实验密度低角点的世界 XYZ 坐标，单位 Å。
        experimental, voxel_size_xyz, origin_xyz = _load_full_grid(
            self.roots.data_root / "density" / pdb_id / "exp.npz"
        )
        # float32，(Z_full, Y_full, X_full)，完整受体模拟密度，预期与 experimental 同形。
        # float32，(3,)，模拟密度 XYZ 三轴体素尺寸，用于与 voxel_size_xyz 逐值核对。
        # float32，(3,)，模拟密度低角点的世界 XYZ 坐标，用于与 origin_xyz 逐值核对。
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
        # 整数数组，(C, 3)，候选 80³ 在完整密度图中的 ZYX 起点；稍后统一为 int64。
        box_start_zyx = np.asarray(raw_box_start_zyx)
        # float32, (C, 3), 候选 80³ 低角点的世界 XYZ 坐标，单位 Å。
        box_origin_world = np.asarray(raw_box_origin_world, dtype=np.float32)
        # float32, (C, 3), 候选 XYZ 体素尺寸，单位 Å/voxel。
        centered_voxel_size = np.asarray(raw_voxel_size_world, dtype=np.float32)
        if (
            box_start_zyx.ndim != 2
            or box_start_zyx.shape[1:] != (3,)
            or not np.issubdtype(box_start_zyx.dtype, np.integer)
        ):
            raise ValueError(f"{centered_path}:box_start_zyx 必须为整数 (C,3)。")
        # int, 三个 F1 几何数组必须共享的候选轴长度 C。
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
        # float32, (C, 3), 把完整密度 XYZ 体素尺寸广播到候选轴后的期望值。
        expected_voxel_size = np.broadcast_to(voxel_size_xyz, centered_voxel_size.shape)
        if not np.array_equal(centered_voxel_size, expected_voxel_size):
            raise ValueError(f"{pdb_id} 的完整密度图与 F1-centered voxel_size 不一致。")
        # float64, (C, 3), 由完整密度原点、候选 ZYX 起点和 XYZ 体素尺寸重建的世界 XYZ 低角点。
        expected_box_origin = (
            origin_xyz.astype(np.float64)[None, :]
            + box_start_zyx[:, [2, 1, 0]].astype(np.float64)
            * voxel_size_xyz.astype(np.float64)[None, :]
        )
        # float, 单位 Å；全部候选和 XYZ 三轴中的最大原点绝对误差，空数组时为 0。
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
        # _CenteredGeometry, 候选轴 C 对齐且完成几何交叉核验的规范化 F1 字段。
        centered: _CenteredGeometry = {
            "box_start_zyx": box_start_zyx.astype(np.int64, copy=False),
            "box_origin_world": box_origin_world,
            "voxel_size_world": centered_voxel_size,
        }
        # Path, 当前 PDB 的受体重原子世界坐标和 49 维原子特征产物。
        receptor_path = self.roots.data_root / "parse" / pdb_id / "receptor_tokens.npz"
        with np.load(receptor_path, allow_pickle=False) as archive:
            # float32, (N_atom, 3), 世界 XYZ 坐标，单位 Å。
            receptor_coords = np.asarray(archive["coords"], dtype=np.float32)
            # float32, (N_atom, 49), 与 receptor_coords 第一维逐原子对齐。
            receptor_features = np.asarray(archive["feat"], dtype=np.float32)
        if receptor_coords.ndim != 2 or receptor_coords.shape[1] != 3:
            raise ValueError(f"{receptor_path}:coords 必须为 (N,3)。")
        if receptor_features.shape != (receptor_coords.shape[0], 49):
            raise ValueError(f"{receptor_path}:feat 必须为 (N,49)。")
        if not np.isfinite(receptor_coords).all() or not np.isfinite(receptor_features).all():
            raise ValueError(f"{receptor_path} 的受体坐标或特征含 NaN/Inf。")
        # _VoxelSources, 一个 PDB 全部候选共享的规范化来源数组。
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
        """
        生成同时绑定 schema、裁剪策略、候选编号和实际裁剪起点的缓存路径。

        输入参数:
            - split_name: str；缓存的数据划分目录名。
            - pdb_id: str；缓存的 PDB 目录名。
            - candidate_index: int；F1 候选轴中的零基索引。
            - crop_start_zyx: 整数数组，(3,)；48³ 在候选 80³ 内的实际 ZYX 起点，必须逐轴位于 [0, 32]。

        返回值:
            - cache_path: Path；位于 output_root/cache/voxel_input_v{schema}/{crop_strategy}/{split_name}/{pdb_id}/ 下，文件名含候选编号和三个起点分量。
        """

        # int64, (3,), 规范化并再次核验的 ZYX 起点。
        start = validate_crop_start(crop_start_zyx)
        # str, 依次连接 Z、Y、X 三个十进制起点分量。
        start_suffix = "_".join(str(int(value)) for value in start)
        return (
            self.roots.voxel_cache_root(self.crop_strategy)
            / split_name
            / pdb_id
            / f"candidate_{candidate_index}_start_{start_suffix}.npy"
        )

    def __getstate__(self) -> dict[str, Any]:
        """
        返回可序列化状态，并用空 LRU 替换父进程已经加载的 PDB 大数组。

        返回值:
            - state: dict[str, Any]；可由 DataLoader spawn 序列化的实例状态。
            - state.roots: SourceRoots；与实例 roots 相同的三个来源根。
            - state.roots.data_root: Path；冻结完整密度与受体原子表的共同根。
            - state.roots.stage1_root: Path；Stage1 F1 候选几何根。
            - state.roots.output_root: Path；版本化候选体素缓存的 Matcher 输出根。
            - state.crop_strategy: LocalCropStrategy；与实例 crop_strategy 相同的局部裁剪策略。
            - state.source_cache_bytes: int；与实例 source_cache_bytes 相同的字节预算。
            - state._source_cache: ByteBudgetLru[tuple[str, str], _VoxelSources]；使用 state.source_cache_bytes 新建的空缓存，不包含父进程大数组。

        该方法只改变派生 worker 接收的进程内性能缓存，不改变任何候选体素的科学字段或落盘路径。
        """

        # dict[str, Any], 当前实例属性的浅拷贝；随后仅替换进程内来源缓存。
        state = self.__dict__.copy()
        state["_source_cache"] = ByteBudgetLru(self.source_cache_bytes)
        return state


# ================================================================================================


def _load_full_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取并核验一份完整实验密度或受体模拟密度 NPZ。

    输入参数:
        - path: Path；exp.npz 或 sim.npz 的文件路径。

    来源字段:
        - grid: 数值数组，(1, Z_full, Y_full, X_full)；完整密度，全部有限且无缺失值。
        - voxel_size: 数值数组，(3,)；XYZ 三轴体素尺寸，单位 Å/voxel，全部有限且逐轴大于 0。
        - origin: 数值数组，(3,)；网格低角点的世界 XYZ 坐标，单位 Å，全部有限。

    返回值:
        - grid_zyx: float32 数组，(Z_full, Y_full, X_full)；去掉来源 grid 的单通道轴，空间轴顺序为 Z、Y、X。
        - voxel_size_xyz: float32 数组，(3,)；XYZ 三轴体素尺寸，单位 Å/voxel。
        - origin_xyz: float32 数组，(3,)；低角点的世界 XYZ 坐标，单位 Å。
    """

    with np.load(path, allow_pickle=False) as archive:
        # float32, (1,Z_full,Y_full,X_full), 仍保留来源单通道轴以便先核验。
        value = np.asarray(archive["grid"], dtype=np.float32)
        # float32, (3,), XYZ 体素尺寸，单位 Å/voxel。
        voxel_size = np.asarray(archive["voxel_size"], dtype=np.float32)
        # float32, (3,), 完整网格低角点的世界 XYZ 坐标，单位 Å。
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
    """
    从完整 ZYX 密度图裁取一个严格位于边界内的 80³ 候选框。

    输入参数:
        - grid: 数值数组，(Z_full, Y_full, X_full)；完整实验密度或模拟密度，空间轴顺序为 Z、Y、X。
        - start_zyx: 整数数组，(3,)；候选 80³ 在 grid 中的 ZYX 起点。

    返回值:
        - crop: float32 数组，(80, 80, 80)；与来源共享数值语义的候选密度裁剪，轴顺序为 Z、Y、X。
    """

    # int64, (3,), 规范化后的候选 ZYX 起点。
    start = np.asarray(start_zyx, dtype=np.int64)
    # int64, (3,), 右开区间终点；逐轴等于 start 加 80。
    stop = start + STAGE1_BOX_SIZE
    if start.shape != (3,) or np.any(start < 0) or np.any(stop > np.asarray(grid.shape)):
        raise ValueError(
            f"80³ 裁剪越界：start={start.tolist()}, stop={stop.tolist()}, shape={grid.shape}。"
        )
    # tuple[slice, slice, slice], 依次对应 Z、Y、X 三个右开裁剪区间。
    slices = tuple(slice(int(first), int(last)) for first, last in zip(start, stop))
    return np.asarray(grid[slices], dtype=np.float32)


def _load_cached(path: Path) -> np.ndarray:
    """
    读取并严格核验一个已经原子发布的候选体素缓存。

    输入参数:
        - path: Path；候选级 NPY 缓存路径。

    返回值:
        - voxel_input: float16 数组，(105, 48, 48, 48)；轴顺序为通道、Z、Y、X，全部有限；返回数组不复制，加载模式由 NumPy 默认行为决定。
    """

    # ndarray, 先保留落盘 dtype 和形状，随后按正式缓存契约核验。
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
    """
    把一个数组写入目标同目录临时 NPY，刷入磁盘后原子替换最终路径。

    输入参数:
        - path: Path；最终候选缓存路径，父目录不存在时递归创建。
        - value: ndarray；由调用者构造并核验的候选缓存数组，np.save 按当前 dtype 和形状写入且禁止对象 pickle。

    副作用:
        - temporary: Path；目标同目录下带当前进程号的临时文件，成功时由 os.replace 移动到 path。
        - path: Path；同文件系统原子替换后的最终 NPY，已存在时会被完整替换。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # Path, 与目标位于同一目录且包含进程号，支持 os.replace 的同文件系统原子发布。
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
