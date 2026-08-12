"""实现 80³ Stage1 候选框与中心 48³ Matcher 空间之间的坐标变换。"""

from __future__ import annotations

import numpy as np

from .schema import LOCAL_BOX_MARGIN, LOCAL_BOX_SIZE, STAGE1_BOX_SIZE


def crop_center_48(array: np.ndarray) -> np.ndarray:
    """裁取数组最后三个 ZYX 维度的中心 48³ 区域。"""

    expected = (STAGE1_BOX_SIZE, STAGE1_BOX_SIZE, STAGE1_BOX_SIZE)
    if array.shape[-3:] != expected:
        raise ValueError(f"待裁数组最后三维必须为 {expected}，实际为 {array.shape[-3:]}。")
    start = LOCAL_BOX_MARGIN
    stop = start + LOCAL_BOX_SIZE
    return array[..., start:stop, start:stop, start:stop]


def local_xyz_to_centered_world(
    local_xyz: np.ndarray,
    voxel_size_xyz: np.ndarray,
) -> np.ndarray:
    """把 80³ 框的连续 XYZ 体素坐标转换为候选中心相对 Å 坐标。"""

    coordinates = np.asarray(local_xyz, dtype=np.float32)
    voxel_size = np.asarray(voxel_size_xyz, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"local_xyz 必须为 (N,3)，实际为 {coordinates.shape}。")
    if voxel_size.shape != (3,) or np.any(voxel_size <= 0):
        raise ValueError(f"voxel_size_xyz 必须为三个正数，实际为 {voxel_size.tolist()}。")
    return ((coordinates - STAGE1_BOX_SIZE / 2) * voxel_size[None, :]).astype(
        np.float32,
        copy=False,
    )


def voxel_zyx_to_centered_world(
    voxel_index_zyx: np.ndarray,
    voxel_size_xyz: np.ndarray,
) -> np.ndarray:
    """把整数 ZYX 体素索引的中心转换为候选中心相对 XYZ Å 坐标。"""

    indices = np.asarray(voxel_index_zyx)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"voxel_index_zyx 必须为 (N,3)，实际为 {indices.shape}。")
    local_xyz = indices[:, [2, 1, 0]].astype(np.float32) + 0.5
    return local_xyz_to_centered_world(local_xyz, voxel_size_xyz)


def keep_local_48(voxel_index_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """筛选 80³ 候选体素中落在中心 48³ 的项，并返回裁后索引。"""

    indices = np.asarray(voxel_index_zyx)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"voxel_index_zyx 必须为 (N,3)，实际为 {indices.shape}。")
    stop = LOCAL_BOX_MARGIN + LOCAL_BOX_SIZE
    keep = np.all((indices >= LOCAL_BOX_MARGIN) & (indices < stop), axis=1)
    return keep, (indices[keep] - LOCAL_BOX_MARGIN).astype(np.int64, copy=False)


def hard_scatter_receptor(
    atom_local_xyz: np.ndarray,
    atom_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把 80³ 内受体原子的 49 维特征按 home voxel 求和散射。"""

    coordinates = np.asarray(atom_local_xyz, dtype=np.float32)
    features = np.asarray(atom_features, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"atom_local_xyz 必须为 (N,3)，实际为 {coordinates.shape}。")
    if features.shape != (coordinates.shape[0], 49):
        raise ValueError(f"atom_features 必须为 (N,49)，实际为 {features.shape}。")

    voxel_xyz = np.floor(coordinates).astype(np.int64)
    valid = np.all((voxel_xyz >= 0) & (voxel_xyz < STAGE1_BOX_SIZE), axis=1)
    voxel_xyz = voxel_xyz[valid]
    features = features[valid]

    flat_size = STAGE1_BOX_SIZE**3
    linear = (
        voxel_xyz[:, 2] * STAGE1_BOX_SIZE * STAGE1_BOX_SIZE
        + voxel_xyz[:, 1] * STAGE1_BOX_SIZE
        + voxel_xyz[:, 0]
    )
    feature_grid = np.zeros((flat_size, 49), dtype=np.float32)
    np.add.at(feature_grid, linear, features)
    hardmask = np.zeros(flat_size, dtype=bool)
    hardmask[linear] = True
    grid_shape = (STAGE1_BOX_SIZE, STAGE1_BOX_SIZE, STAGE1_BOX_SIZE)
    return (
        feature_grid.reshape(*grid_shape, 49).transpose(3, 0, 1, 2),
        hardmask.reshape(grid_shape),
    )
