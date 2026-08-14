"""解析候选级 ``48³`` 裁剪，并统一转换 Stage1 ``80³`` 局部坐标。

``resolve_candidate_crop`` 从完整权威 V 返回 ``CandidateCropGeometry``；其余坐标、
筛选和采样统计入口共享该对象中的实际起点。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import (
    CANDIDATE_SAMPLING_FEATURE_DIM,
    LOCAL_BOX_MARGIN,
    LOCAL_BOX_SIZE,
    LOCAL_CROP_STRATEGIES,
    STAGE1_BOX_SIZE,
    LocalCropStrategy,
)


@dataclass(frozen=True)
class CandidateCropGeometry:
    """保存一个权威 V 组件解析出的候选级裁剪几何。

    ``strategy`` 是 ``fixed_center`` 或 ``v_centroid``。
    ``requested_start_zyx`` 是边界限制前的 int64 ``(3,)`` 请求起点。
    ``start_zyx`` 是逐轴限制到 ``[0,32]`` 的 int64 ``(3,)`` 实际起点。
    ``v_centroid_zyx`` 是完整 V 的 float64 ``(3,)`` 体素下标质心。
    ``crop_center_offset_zyx`` 是实际 ``48³`` 几何中心减 V 体素中心质心，单位 voxel。
    ``clipped_axis_mask`` 是 bool ``(3,)``，真值表示该轴请求起点被边界限制。
    ``full_v_count`` 是裁剪前完整权威 V 的体素数。
    ``retained_v_count`` 是实际 ``48³`` 裁剪保留的权威 V 体素数。
    """

    strategy: LocalCropStrategy
    requested_start_zyx: np.ndarray
    start_zyx: np.ndarray
    v_centroid_zyx: np.ndarray
    crop_center_offset_zyx: np.ndarray
    clipped_axis_mask: np.ndarray
    full_v_count: int
    retained_v_count: int


def resolve_candidate_crop(
    full_v_index_zyx: np.ndarray,
    strategy: LocalCropStrategy,
) -> CandidateCropGeometry:
    """从非空权威 V 体素解析候选级 ``48³`` 起点和采样统计。

    ``v_centroid`` 严格使用 NumPy 最近整数规则
    ``rint(mean(V_zyx)+0.5-24)``，再逐轴限制到 ``[0,32]``；
    ``fixed_center`` 固定请求 ``(16,16,16)``，用于复现第 0/1 批输入语义。
    """

    if strategy not in LOCAL_CROP_STRATEGIES:
        raise ValueError(f"不支持 local_crop_strategy={strategy!r}。")
    indices = np.asarray(full_v_index_zyx)
    if indices.ndim != 2 or indices.shape[1] != 3 or indices.shape[0] == 0:
        raise ValueError("full_v_index_zyx 必须为非空 (N,3) 整数数组。")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("full_v_index_zyx 必须使用整数 dtype。")
    indices = indices.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= STAGE1_BOX_SIZE):
        raise ValueError(f"权威 V 体素必须位于 [0,{STAGE1_BOX_SIZE})³。")

    centroid = indices.astype(np.float64).mean(axis=0)
    if strategy == "v_centroid":
        requested = np.rint(
            centroid + 0.5 - LOCAL_BOX_SIZE / 2.0
        ).astype(np.int64)
    else:
        requested = np.full(3, LOCAL_BOX_MARGIN, dtype=np.int64)
    maximum_start = STAGE1_BOX_SIZE - LOCAL_BOX_SIZE
    start = np.clip(requested, 0, maximum_start).astype(np.int64, copy=False)
    keep, _ = keep_local_48_at(indices, start)
    crop_center = start.astype(np.float64) + LOCAL_BOX_SIZE / 2.0
    voxel_center_centroid = centroid + 0.5
    return CandidateCropGeometry(
        strategy=strategy,
        requested_start_zyx=requested,
        start_zyx=start,
        v_centroid_zyx=centroid,
        crop_center_offset_zyx=crop_center - voxel_center_centroid,
        clipped_axis_mask=requested != start,
        full_v_count=int(indices.shape[0]),
        retained_v_count=int(np.count_nonzero(keep)),
    )


def candidate_sampling_features(geometry: CandidateCropGeometry) -> np.ndarray:
    """返回 float32 ``(4,)`` 的边界限制、L2/L∞ 体素距离和 ``[0,1]`` V 保留比例。"""

    center_offset = np.asarray(geometry.crop_center_offset_zyx, dtype=np.float64)
    retained_fraction = geometry.retained_v_count / geometry.full_v_count
    features = np.asarray(
        (
            float(np.any(geometry.clipped_axis_mask)),
            float(np.linalg.norm(center_offset)),
            float(np.max(np.abs(center_offset))),
            float(retained_fraction),
        ),
        dtype=np.float32,
    )
    if features.shape != (CANDIDATE_SAMPLING_FEATURE_DIM,) or not np.isfinite(features).all():
        raise RuntimeError("候选采样特征不符合 schema 宽度或有限值契约。")
    return features


def crop_local_48(array: np.ndarray, start_zyx: np.ndarray) -> np.ndarray:
    """把数组末三轴的 ``(80,80,80)`` ZYX 空间裁为 ``(48,48,48)``，保持全部前导轴不变。

    ``start_zyx`` 是逐轴位于 ``[0,32]`` 的整数 ``(3,)`` 候选起点；返回值沿用输入
    dtype，形状为 ``array.shape[:-3] + (48,48,48)``。
    """

    expected = (STAGE1_BOX_SIZE, STAGE1_BOX_SIZE, STAGE1_BOX_SIZE)
    if array.shape[-3:] != expected:
        raise ValueError(f"待裁数组最后三维必须为 {expected}，实际为 {array.shape[-3:]}。")
    start = validate_crop_start(start_zyx)
    stop = start + LOCAL_BOX_SIZE
    return array[
        ...,
        int(start[0]) : int(stop[0]),
        int(start[1]) : int(stop[1]),
        int(start[2]) : int(stop[2]),
    ]


def local_xyz_to_crop_centered_world(
    local_xyz: np.ndarray,
    crop_start_zyx: np.ndarray,
    voxel_size_xyz: np.ndarray,
) -> np.ndarray:
    """把 ``80³`` 局部连续 XYZ 体素坐标转换为相对裁剪中心的世界坐标。

    ``local_xyz`` 是 float32 ``(N,3)`` 连续 XYZ 坐标，单位 voxel；
    ``crop_start_zyx`` 是整数 ``(3,)`` ZYX 起点；``voxel_size_xyz`` 是正数
    float32 ``(3,)`` XYZ 尺寸，单位 Å/voxel。返回 float32 ``(N,3)`` 的
    裁剪中心相对 XYZ 坐标，单位 Å。
    """

    coordinates = np.asarray(local_xyz, dtype=np.float32)
    start = validate_crop_start(crop_start_zyx)
    voxel_size = np.asarray(voxel_size_xyz, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"local_xyz 必须为 (N,3)，实际为 {coordinates.shape}。")
    if voxel_size.shape != (3,) or np.any(voxel_size <= 0):
        raise ValueError(f"voxel_size_xyz 必须为三个正数，实际为 {voxel_size.tolist()}。")
    crop_center_xyz = start[[2, 1, 0]].astype(np.float32) + LOCAL_BOX_SIZE / 2.0
    return ((coordinates - crop_center_xyz[None, :]) * voxel_size[None, :]).astype(
        np.float32,
        copy=False,
    )


def voxel_zyx_to_crop_centered_world(
    voxel_index_zyx: np.ndarray,
    crop_start_zyx: np.ndarray,
    voxel_size_xyz: np.ndarray,
) -> np.ndarray:
    """把 ``80³`` 整数 ZYX 体素下标转换为相对裁剪中心的世界坐标。

    ``voxel_index_zyx`` 是整数 ``(N,3)`` 中心下标；``crop_start_zyx`` 是整数
    ``(3,)`` ZYX 起点；``voxel_size_xyz`` 是正数 float32 ``(3,)`` XYZ 尺寸，
    单位 Å/voxel。返回 float32 ``(N,3)`` 的裁剪中心相对 XYZ 坐标，单位 Å。
    """

    indices = np.asarray(voxel_index_zyx)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"voxel_index_zyx 必须为 (N,3)，实际为 {indices.shape}。")
    local_xyz = indices[:, [2, 1, 0]].astype(np.float32) + 0.5
    return local_xyz_to_crop_centered_world(local_xyz, crop_start_zyx, voxel_size_xyz)


def keep_local_48_at(
    voxel_index_zyx: np.ndarray,
    start_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """筛选落在候选 ``48³`` 内的 ``80³`` 整数 ZYX 体素。

    ``voxel_index_zyx`` 是整数 ``(N,3)``；``start_zyx`` 是整数 ``(3,)`` 起点。
    返回 bool ``(N,)`` 保留掩码和与真值项同序的 int64 ``(K,3)`` 裁剪内 ZYX 下标。
    """

    indices = np.asarray(voxel_index_zyx)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"voxel_index_zyx 必须为 (N,3)，实际为 {indices.shape}。")
    start = validate_crop_start(start_zyx)
    stop = start + LOCAL_BOX_SIZE
    keep = np.all((indices >= start) & (indices < stop), axis=1)
    return keep, (indices[keep] - start).astype(np.int64, copy=False)


def validate_crop_start(start_zyx: np.ndarray) -> np.ndarray:
    """要求 ``48³`` 起点是位于 ``[0,32]`` 的 int64 ``(3,)``。"""

    start = np.asarray(start_zyx)
    maximum_start = STAGE1_BOX_SIZE - LOCAL_BOX_SIZE
    if start.shape != (3,) or not np.issubdtype(start.dtype, np.integer):
        raise ValueError("crop_start_zyx 必须为整数 (3,)。")
    start = start.astype(np.int64, copy=False)
    if np.any(start < 0) or np.any(start > maximum_start):
        raise ValueError(f"crop_start_zyx 必须逐轴位于 [0,{maximum_start}]。")
    return start


def hard_scatter_receptor(
    atom_local_xyz: np.ndarray,
    atom_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把 F1 ``80³`` 局部受体坐标散射为特征网格和占据掩码。

    ``atom_local_xyz`` 是单位为 voxel 的连续 XYZ ``(N,3)``，``atom_features`` 是
    逐原子对齐的 ``(N,49)``。越界原子丢弃；返回 float32 C/ZYX
    ``(49,80,80,80)`` 求和网格和 bool ZYX ``(80,80,80)`` 占据掩码。
    """

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
