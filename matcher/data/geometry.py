"""
从 Stage1 80³ 候选空间解析 Matcher 48³ 局部裁剪，并统一 ZYX 体素索引与 XYZ 物理坐标的转换。

主要入口:
    - resolve_candidate_crop: 从完整权威 V 体素解析请求起点、实际起点、边界限制和 V 保留统计。
    - candidate_sampling_features: 把 CandidateCropGeometry 转成四个候选采样特征。
    - crop_local_48: 用同一个实际起点裁剪任意具有 80³ 末三轴的来源数组。
    - local_xyz_to_crop_centered_world: 把连续 XYZ 体素坐标转成以实际裁剪中心为原点的 XYZ 物理坐标。
    - voxel_zyx_to_crop_centered_world: 把整数 ZYX 体素中心转成以实际裁剪中心为原点的 XYZ 物理坐标。
    - keep_local_48_at: 同步筛选稀疏体素并转换为 48³ 内部 ZYX 下标。
    - hard_scatter_receptor: 把 80³ 局部连续受体原子坐标散射为 49 通道网格和占据掩码。

本模块只返回内存数组和 CandidateCropGeometry，不读取或写入文件。除明确写成世界坐标的返回值外，起点与索引均使用 ZYX 顺序，连续原子坐标与物理坐标均使用 XYZ 顺序。
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
    """
    保存一个权威 V 组件解析出的候选级 48³ 裁剪几何。

    字段:
        - strategy: LocalCropStrategy；fixed_center 固定请求 80³ 中心裁剪，v_centroid 按完整 V 体素中心质心请求裁剪。
        - requested_start_zyx: int64 数组，(3,)；边界限制前的 48³ 请求起点，三个元素依次为 Z、Y、X，v_centroid 情况下可以超出合法起点范围。
        - start_zyx: int64 数组，(3,)；逐轴限制到 [0, 32] 的实际 48³ 起点，三个元素依次为 Z、Y、X。
        - v_centroid_zyx: float64 数组，(3,)；完整权威 V 整数下标的算术平均，三个元素依次为 Z、Y、X，数值位于体素下标坐标系而非世界坐标系。
        - crop_center_offset_zyx: float64 数组，(3,)；实际裁剪几何中心减去 V 体素中心质心，三个元素依次为 Z、Y、X，单位为 voxel。
        - clipped_axis_mask: bool 数组，(3,)；依次对应 Z、Y、X，True 表示该轴 requested_start_zyx 被边界限制后才得到 start_zyx。
        - full_v_count: int；裁剪前完整权威 V 的体素数量，也是 V 保留比例的分母，必须大于 0。
        - retained_v_count: int；落入实际 48³ 裁剪的权威 V 体素数量，范围为 [0, full_v_count]。
    """

    strategy: LocalCropStrategy
    requested_start_zyx: np.ndarray
    start_zyx: np.ndarray
    v_centroid_zyx: np.ndarray   # NOTE: 这个字段将会被上游的 stage1 直接保存，且上游将会完全预先裁剪好 48^3 的范围(策略固定为现在的 V-centered)
    crop_center_offset_zyx: np.ndarray   # NOTE: 这个也必须被stage1的centered推理阶段保存。注意我的总设想仍然是： Matcher 只负责读取和极少量适配，产物由stage1推理负责切块、整理、适配。
    clipped_axis_mask: np.ndarray
    full_v_count: int
    retained_v_count: int


def resolve_candidate_crop(
    full_v_index_zyx: np.ndarray,
    strategy: LocalCropStrategy,
) -> CandidateCropGeometry:
    """
    从非空权威 V 体素解析候选级 48³ 起点和采样统计。

    输入参数:
        - full_v_index_zyx: 整数数组，(N_v, 3)；完整权威 V 的 80³ 局部体素下标，最后一维依次为 Z、Y、X，N_v 必须大于 0，数值逐轴位于 [0, 80)。
        - strategy: LocalCropStrategy；v_centroid 使用 np.rint(mean(V_zyx) + 0.5 - 24) 请求起点，fixed_center 固定请求 (16, 16, 16)。

    关键规则:
        - requested_start_zyx: int64 数组，(3,)；np.rint 使用 NumPy 最近偶数舍入规则，随后才逐轴限制到 [0, 32]。
        - retained_v_count: int；使用实际 start_zyx 筛选 full_v_index_zyx 后得到，不使用未限制的 requested_start_zyx。

    返回值:
        - geometry: CandidateCropGeometry；当前候选的完整裁剪几何，后续体素、V token、PP token、区域监督和采样特征必须共享同一对象。
        - geometry.strategy: LocalCropStrategy；与输入 strategy 相同。
        - geometry.requested_start_zyx: int64 数组，(3,)；边界限制前的请求 ZYX 起点，v_centroid 时可以超出合法范围。
        - geometry.start_zyx: int64 数组，(3,)；逐轴限制到 [0, 32] 的实际 ZYX 起点。
        - geometry.v_centroid_zyx: float64 数组，(3,)；完整 V 整数下标在 Z、Y、X 三轴的算术平均，单位 voxel。
        - geometry.crop_center_offset_zyx: float64 数组，(3,)；实际 48³ 中心减去完整 V 体素中心质心，轴顺序为 ZYX，单位 voxel。
        - geometry.clipped_axis_mask: bool 数组，(3,)；依次对应 Z、Y、X，True 表示该轴请求起点受到边界限制。
        - geometry.full_v_count: int；完整权威 V 的体素数量，等于 N_v 且大于 0。
        - geometry.retained_v_count: int；实际 48³ 内保留的 V 体素数量，范围 [0, N_v]。
    """

    if strategy not in LOCAL_CROP_STRATEGIES:
        raise ValueError(f"不支持 local_crop_strategy={strategy!r}。")
    # (N_v, 3), 完整权威 V 的 80³ 局部 ZYX 体素下标；此处先保留输入整数 dtype 以执行契约检查。
    indices = np.asarray(full_v_index_zyx)
    if indices.ndim != 2 or indices.shape[1] != 3 or indices.shape[0] == 0:
        raise ValueError("full_v_index_zyx 必须为非空 (N,3) 整数数组。")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("full_v_index_zyx 必须使用整数 dtype。")
    indices = indices.astype(np.int64, copy=False)
    if np.any(indices < 0) or np.any(indices >= STAGE1_BOX_SIZE):
        raise ValueError(f"权威 V 体素必须位于 [0,{STAGE1_BOX_SIZE})³。")

    # float64, (3,), 完整 V 整数下标在 Z、Y、X 三轴上的算术平均，尚未加 0.5 转到体素中心坐标。
    centroid = indices.astype(np.float64).mean(axis=0)
    if strategy == "v_centroid":
        # int64, (3,), 让 48³ 几何中心对准 V 体素中心质心的请求 ZYX 起点；可能超出 [0, 32]。
        requested = np.rint(
            centroid + 0.5 - LOCAL_BOX_SIZE / 2.0
        ).astype(np.int64)
    else:
        # int64, (3,), 80³ 中心裁剪的固定 ZYX 起点 (16, 16, 16)。
        requested = np.full(3, LOCAL_BOX_MARGIN, dtype=np.int64)
    maximum_start = STAGE1_BOX_SIZE - LOCAL_BOX_SIZE
    # int64, (3,), 实际 ZYX 起点；逐轴限制后保证 [start, start + 48) 完全落在 80³ 内。
    start = np.clip(requested, 0, maximum_start).astype(np.int64, copy=False)
    # bool (N_v,) 与 int64 (K, 3)；这里只用 keep 统计实际裁剪保留的完整 V 体素数。
    keep, _ = keep_local_48_at(indices, start)
    # float64, (3,), 实际 48³ 裁剪的连续 ZYX 几何中心，单位 voxel。
    crop_center = start.astype(np.float64) + LOCAL_BOX_SIZE / 2.0
    # float64, (3,), 完整 V 的体素中心质心；整数下标加 0.5 后才表示各体素中心。
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
    """
    把候选裁剪几何转换为固定顺序的四维采样特征。

    输入参数:
        - geometry: CandidateCropGeometry；必须来自与当前候选体素、V token、PP token 和区域监督相同的 resolve_candidate_crop 调用。
        - geometry.strategy: LocalCropStrategy；裁剪策略名称，本函数不直接编码该字段。
        - geometry.requested_start_zyx: int64 数组，(3,)；边界限制前的请求 ZYX 起点，本函数不直接编码该字段。
        - geometry.start_zyx: int64 数组，(3,)；实际 ZYX 起点，本函数通过其它统计间接反映其边界影响。
        - geometry.v_centroid_zyx: float64 数组，(3,)；完整 V 整数下标质心，本函数不直接编码该字段。
        - geometry.crop_center_offset_zyx: float64 数组，(3,)；实际裁剪中心相对完整 V 体素中心质心的 ZYX 偏移，单位 voxel，用于 features[1:3]。
        - geometry.clipped_axis_mask: bool 数组，(3,)；任一轴为 True 时 features[0] 为 1。
        - geometry.full_v_count: int；完整 V 体素数，必须大于 0，是 features[3] 的分母。
        - geometry.retained_v_count: int；实际 48³ 内保留的 V 体素数，是 features[3] 的分子。

    返回值:
        - features: float32 数组，(4,)；依次为任一轴受边界限制的 0/1 标志、裁剪中心到 V 质心的 L2 距离、同一偏移的 L∞ 距离和 V 保留比例。
        - features[0]: float32 标量；clipped_axis_mask 任一元素为 True 时为 1，否则为 0。
        - features[1]: float32 标量；crop_center_offset_zyx 的欧氏范数，单位 voxel。
        - features[2]: float32 标量；crop_center_offset_zyx 的绝对值最大分量，单位 voxel。
        - features[3]: float32 标量；retained_v_count / full_v_count，范围 [0, 1]。
    """

    # float64, (3,), 实际裁剪中心相对完整 V 体素中心质心的 ZYX 偏移，单位 voxel。
    center_offset = np.asarray(geometry.crop_center_offset_zyx, dtype=np.float64)
    # float, 实际 48³ 裁剪保留的 V 体素比例，分母是非零 full_v_count。
    retained_fraction = geometry.retained_v_count / geometry.full_v_count
    # float32, (4,), 顺序与 CANDIDATE_SAMPLING_FEATURE_NAMES 逐位置一致。
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
    """
    把数组末三轴的 80³ ZYX 空间裁为 48³，并保持全部前导轴和输入 dtype 不变。

    输入参数:
        - array: 数组，形状为 (..., 80, 80, 80)；末三轴依次为 Z、Y、X，前导轴可以是通道或其他调用者定义的维度。
        - start_zyx: 整数数组，(3,)；逐轴位于 [0, 32] 的实际裁剪起点，依次对应 array 的末第三、末第二和末第一轴。

    返回值:
        - cropped: 数组，形状为 array.shape[:-3] + (48, 48, 48)；与输入共享切片视图语义，dtype 不变。
    """

    expected = (STAGE1_BOX_SIZE, STAGE1_BOX_SIZE, STAGE1_BOX_SIZE)
    if array.shape[-3:] != expected:
        raise ValueError(f"待裁数组最后三维必须为 {expected}，实际为 {array.shape[-3:]}。")
    start = validate_crop_start(start_zyx)
    # int64, (3,), 48³ 半开切片的 ZYX 终点；每一轴区间都是 [start, stop)。
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
    """
    把 80³ 局部连续 XYZ 体素坐标转换为相对实际裁剪中心的 XYZ 物理坐标。

    输入参数:
        - local_xyz: 数组，(N_point, 3)；80³ 局部连续坐标，最后一维依次为 X、Y、Z，单位 voxel。
        - crop_start_zyx: 整数数组，(3,)；实际 48³ 起点，元素依次为 Z、Y、X。
        - voxel_size_xyz: 数组，(3,)；三个正数依次为 X、Y、Z 方向的体素尺寸，单位 Å/voxel。

    返回值:
        - centered_world_xyz: float32 数组，(N_point, 3)；以实际 48³ 几何中心为原点，最后一维依次为 X、Y、Z，单位 Å。
    """

    # float32, (N_point, 3), 80³ 局部连续 XYZ 体素坐标。
    coordinates = np.asarray(local_xyz, dtype=np.float32)
    start = validate_crop_start(crop_start_zyx)
    # float32, (3,), XYZ 轴体素尺寸，单位 Å/voxel。
    voxel_size = np.asarray(voxel_size_xyz, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"local_xyz 必须为 (N,3)，实际为 {coordinates.shape}。")
    if voxel_size.shape != (3,) or np.any(voxel_size <= 0):
        raise ValueError(f"voxel_size_xyz 必须为三个正数，实际为 {voxel_size.tolist()}。")
    # float32, (3,), 实际 48³ 裁剪中心在 80³ 连续局部坐标中的 XYZ 位置，单位 voxel。
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
    """
    把 80³ 整数 ZYX 体素下标转换为相对实际裁剪中心的 XYZ 物理坐标。

    输入参数:
        - voxel_index_zyx: 整数数组，(N_voxel, 3)；最后一维依次为 Z、Y、X，每个整数表示体素下标而非体素角点坐标。
        - crop_start_zyx: 整数数组，(3,)；实际 48³ 起点，元素依次为 Z、Y、X。
        - voxel_size_xyz: 数组，(3,)；三个正数依次为 X、Y、Z 方向的体素尺寸，单位 Å/voxel。

    返回值:
        - centered_world_xyz: float32 数组，(N_voxel, 3)；以实际裁剪中心为原点，最后一维依次为 X、Y、Z，单位 Å。
    """

    # (N_voxel, 3), 80³ 局部 ZYX 体素下标；保持输入整数 dtype 只用于形状检查和轴重排。
    indices = np.asarray(voxel_index_zyx)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"voxel_index_zyx 必须为 (N,3)，实际为 {indices.shape}。")
    # float32, (N_voxel, 3), 每个整数 ZYX 体素的连续 XYZ 中心坐标；加 0.5 从低角点移到体素中心。
    local_xyz = indices[:, [2, 1, 0]].astype(np.float32) + 0.5
    return local_xyz_to_crop_centered_world(local_xyz, crop_start_zyx, voxel_size_xyz)


def keep_local_48_at(
    voxel_index_zyx: np.ndarray,
    start_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    筛选落在指定 48³ 半开区间内的 80³ 整数 ZYX 体素，并转换为裁剪内局部下标。

    输入参数:
        - voxel_index_zyx: 整数数组，(N_voxel, 3)；80³ 局部体素下标，最后一维依次为 Z、Y、X。
        - start_zyx: 整数数组，(3,)；实际 48³ 起点，逐轴位于 [0, 32]。

    返回值:
        - keep_mask: bool 数组，(N_voxel,)；True 表示 voxel_index_zyx 同位置体素逐轴位于 [start_zyx, start_zyx + 48)。
        - local_index_zyx: int64 数组，(N_kept, 3)；只保留 keep_mask 为 True 的体素并减去 start_zyx，最后一维依次为 Z、Y、X，数值逐轴位于 [0, 48)。
    """

    # (N_voxel, 3), 待筛选的 80³ 局部 ZYX 体素下标。
    indices = np.asarray(voxel_index_zyx)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"voxel_index_zyx 必须为 (N,3)，实际为 {indices.shape}。")
    start = validate_crop_start(start_zyx)
    # int64, (3,), 48³ 半开区间的 ZYX 终点。
    stop = start + LOCAL_BOX_SIZE
    # bool, (N_voxel,), 同时通过 Z、Y、X 三轴半开区间检查的体素位置。
    keep = np.all((indices >= start) & (indices < stop), axis=1)
    return keep, (indices[keep] - start).astype(np.int64, copy=False)


def validate_crop_start(start_zyx: np.ndarray) -> np.ndarray:
    """
    验证并规范化 48³ 裁剪起点。

    输入参数:
        - start_zyx: 整数数组，(3,)；元素依次为 Z、Y、X，必须逐轴位于 [0, 32]。

    返回值:
        - validated_start_zyx: int64 数组，(3,)；数值与输入相同，在可能时复用输入内存。
    """

    # (3,), 候选 48³ 裁剪起点；此处先保留输入整数 dtype 执行契约检查。
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
    """
    把 F1 80³ 局部受体原子坐标散射为 49 通道特征网格和受体占据掩码。

    输入参数:
        - atom_local_xyz: 数组，(N_atom, 3)；受体原子在 80³ 局部空间中的连续坐标，最后一维依次为 X、Y、Z，单位 voxel。
        - atom_features: 数组，(N_atom, 49)；与 atom_local_xyz 第一维逐原子对齐的受体特征。
        - atom_features[:, 0:6]: 数组，(N_atom, 6)；C、N、O、S、P、X 六类元素独热特征。
        - atom_features[:, 6:31]: 数组，(N_atom, 25)；残基独热顺序为 ALA、ARG、ASN、ASP、CYS、GLN、GLU、GLY、HIS、ILE、LEU、LYS、MET、PHE、PRO、SER、THR、TRP、TYR、VAL、A、U、C、G、X。
        - atom_features[:, 31:39]: 数组，(N_atom, 8)；依次为极性、非极性、酸性、碱性、中性、正电、负电、无电标志。
        - atom_features[:, 39]: 数组，(N_atom,)；元素质量除以 32，未知元素 X 使用 14.0/32.0。
        - atom_features[:, 40]: 数组，(N_atom,)；2 Å 内其它受体原子数的 log1p，已扣除查询原子自身。
        - atom_features[:, 41:49]: 数组，(N_atom, 8)；依次为 (2,4]、(4,6]、(6,8]、(8,10]、(10,12]、(12,14]、(14,16]、(16,18] Å 壳层内其它原子数的 log1p。

    处理规则:
        - voxel_assignment: int64 数组，(N_atom, 3)；对连续 XYZ 坐标逐元素向下取整，位于 [0, 80)³ 外的原子及其特征同时丢弃。
        - collision_rule: 同一体素中的多枚原子按 49 个通道逐元素求和；占据掩码只记录是否至少有一枚有效原子。

    返回值:
        - feature_grid: float32 数组，(49, 80, 80, 80)；轴顺序为通道、Z、Y、X，同体素原子特征已经求和。
        - receptor_hardmask: bool 数组，(80, 80, 80)；轴顺序为 Z、Y、X，True 表示至少有一枚有效受体原子落入该体素。
    """

    # float32, (N_atom, 3), 80³ 局部连续 XYZ 原子坐标，单位 voxel。
    coordinates = np.asarray(atom_local_xyz, dtype=np.float32)
    # float32, (N_atom, 49), 与 coordinates 第一维逐原子对齐的受体特征。
    features = np.asarray(atom_features, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"atom_local_xyz 必须为 (N,3)，实际为 {coordinates.shape}。")
    if features.shape != (coordinates.shape[0], 49):
        raise ValueError(f"atom_features 必须为 (N,49)，实际为 {features.shape}。")

    # int64, (N_atom, 3), 原子归属的 XYZ 整数体素下标；floor 将连续坐标分配到包含它的半开体素。
    voxel_xyz = np.floor(coordinates).astype(np.int64)
    # bool, (N_atom,), True 表示原子 XYZ 体素下标三轴都位于 [0, 80)。
    valid = np.all((voxel_xyz >= 0) & (voxel_xyz < STAGE1_BOX_SIZE), axis=1)
    # [N_atom, 3] -> [N_valid_atom, 3]；与 features 使用同一个 valid 掩码保持逐原子对齐。
    voxel_xyz = voxel_xyz[valid]
    # [N_atom, 49] -> [N_valid_atom, 49]；只保留位于 80³ 内的受体原子特征。
    features = features[valid]

    flat_size = STAGE1_BOX_SIZE**3
    # int64, (N_valid_atom,), 每个 XYZ 体素映射到展平 ZYX 网格的下标 z*80² + y*80 + x。
    linear = (
        voxel_xyz[:, 2] * STAGE1_BOX_SIZE * STAGE1_BOX_SIZE
        + voxel_xyz[:, 1] * STAGE1_BOX_SIZE
        + voxel_xyz[:, 0]
    )
    # float32, (80³, 49), 展平空间在前、通道在后的临时网格；np.add.at 对重复 linear 下标逐通道累加。
    feature_grid = np.zeros((flat_size, 49), dtype=np.float32)
    np.add.at(feature_grid, linear, features)
    # bool, (80³,), 展平受体占据掩码；重复 linear 下标仍只产生一个 True。
    hardmask = np.zeros(flat_size, dtype=bool)
    hardmask[linear] = True
    # tuple[int, int, int], 受体特征网格与占据掩码共同的 ZYX 空间形状 (80,80,80)。
    grid_shape = (STAGE1_BOX_SIZE, STAGE1_BOX_SIZE, STAGE1_BOX_SIZE)
    return (
        feature_grid.reshape(*grid_shape, 49).transpose(3, 0, 1, 2),
        hardmask.reshape(grid_shape),
    )
