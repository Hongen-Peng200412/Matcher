"""
按 Pocket Plus Find_0 契约把实验密度、模拟密度和受体占据掩码构造成 56 个密度通道.

主要入口:
- build_density_channels: 对三个同形非空三维网格依次组合 4 种基础运算、2 种归一化和 7 种后处理.

返回对象:
- density_channels: float32 数组, (56, Z, Y, X); 第一维与 DENSITY_CHANNEL_NAMES 逐通道对齐, 末三维沿用输入空间形状.

低层函数接受任意同形非空三维网格; 当前正式调用位置 VoxelInputStore._build 传入 80³ ZYX 网格. 本模块只计算内存数组, 不读取或写入文件.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage  # type: ignore[import-untyped]


# tuple[str, ...], 长度 56; 按 operation、normalization、postprocess 三层嵌套顺序生成, 并与 build_density_channels 返回数组第一维逐通道对齐.
DENSITY_CHANNEL_NAMES = tuple(
    f"{operation}_{normalization}_{postprocess}"
    for operation in ("exp", "sim", "diff", "posdiff")
    for normalization in ("nonorm", "clipnorm")
    for postprocess in (
        "nopost",
        "gauss1",
        "gauss2",
        "DoG1",
        "DoG2",
        "smooth1",
        "smooth2",
    )
)


def build_experimental_density_channel(experimental: np.ndarray) -> np.ndarray:
    """
    构造 density_blob_only 唯一允许读取的实验密度通道.

    - experimental: 数组, 形状为 (Z,Y,X); Stage1 supplied crop 内的实验密度.
    - return: float32 数组, 形状为 (1,Z,Y,X); 按现有 exp_clipnorm_nopost 的 0.1%/99.9% 分位截断与全网格 z-score 规则归一化.

    本函数不读取模拟密度或受体 hardmask, 因而不会为单密度变体触发被禁用输入的
    文件访问和派生计算.
    """
    grid = _validated_grid(experimental, "experimental")
    return _clip_normalize(grid)[None].astype(np.float32, copy=False)


def build_density_channels(
    experimental: np.ndarray,
    simulated: np.ndarray,
    receptor_hardmask: np.ndarray,
) -> np.ndarray:
    """
    生成通道顺序固定为基础运算、归一化和后处理的 56 通道密度数组.

    形状符号:
    - Z: 输入三维网格的第一轴长度.
    - Y: 输入三维网格的第二轴长度.
    - X: 输入三维网格的第三轴长度.

    输入参数:
    - experimental: 数组, (Z, Y, X); 有限实验密度, 当前 VoxelInputStore 调用传入 80³ ZYX 网格.
    - simulated: 数组, (Z, Y, X); 与 experimental 逐体素对齐的有限模拟密度.
    - receptor_hardmask: 数组, (Z, Y, X); 转为 bool 后 True 表示该体素至少含一枚受体原子, 用于优先选择模拟密度线性拟合体素和构造 smooth 后处理.

    基础运算:
    - exp: float32 数组, (Z, Y, X); 原始 experimental.
    - sim: float32 数组, (Z, Y, X); 原始 simulated, 不先应用线性拟合.
    - diff: float32 数组, (Z, Y, X); experimental - (scale * simulated + offset).
    - posdiff: float32 数组, (Z, Y, X); diff 的逐体素非负部分 max(diff, 0).

    归一化:
    - nonorm: 数组, (Z, Y, X); 保持对应基础运算值不变.
    - clipnorm: 数组, (Z, Y, X); 按当前候选的 0.1% 和 99.9% 分位截断后执行 z-score.

    后处理:
    - nopost: 数组, (Z, Y, X); 保持归一化后的基础值.
    - gauss1: 数组, (Z, Y, X); 反射边界 Gaussian 滤波, sigma=1 voxel.
    - gauss2: 数组, (Z, Y, X); 反射边界 Gaussian 滤波, sigma=2 voxels.
    - DoG1: 数组, (Z, Y, X); Gaussian(sigma=1.6) - Gaussian(sigma=1.0).
    - DoG2: 数组, (Z, Y, X); Gaussian(sigma=3.2) - Gaussian(sigma=2.0).
    - smooth1: 数组, (Z, Y, X); 基础值乘以 1 - receptor_strength, strength 使用 sigma=1 voxel、radius=2 voxels.
    - smooth2: 数组, (Z, Y, X); 基础值乘以 1 - receptor_strength, strength 使用 sigma=2 voxels、radius=2 voxels.

    返回值:
    - density_channels: float32 数组, (56, Z, Y, X); 第一维名称和顺序由 DENSITY_CHANNEL_NAMES 固定.
    """
    # float32, (Z, Y, X), 已验证为非空且全部有限的实验密度.
    exp = _validated_grid(experimental, "experimental")
    # float32, (Z, Y, X), 已验证为非空且全部有限的模拟密度.
    sim = _validated_grid(simulated, "simulated")
    # bool, (Z, Y, X), 与两套密度逐体素对齐的受体占据掩码.
    mask = np.asarray(receptor_hardmask, dtype=bool)
    if exp.shape != sim.shape or mask.shape != exp.shape:
        raise ValueError(
            "experimental、simulated 和 receptor_hardmask 必须共享形状，"
            f"实际为 {exp.shape}、{sim.shape}、{mask.shape}。"
        )

    # 两个 float 标量, 把模拟密度线性映射到实验密度量级; 拟合退化时分别为 1 和 0.
    scale, offset = _fit_scale(exp, sim, mask)
    # float32, (Z, Y, X), 实验密度减去线性拟合后的模拟密度.
    difference = exp - (scale * sim + offset)
    # dict[str, np.ndarray], 四个同形 (Z, Y, X) 基础密度运算; 键是 DENSITY_CHANNEL_NAMES 的 operation 部分.
    operations = {
        "exp": exp,
        "sim": sim,
        "diff": difference,
        "posdiff": np.maximum(difference, 0.0),
    }
    # dict[(operation, normalization), np.ndarray], 懒计算缓存; 每个值形状为 (Z, Y, X).
    normalized: dict[tuple[str, str], np.ndarray] = {}
    # dict[(operation, normalization, sigma), np.ndarray], Gaussian 结果缓存; 每个值形状为 (Z, Y, X).
    gaussian: dict[tuple[str, str, float], np.ndarray] = {}
    # dict[float, np.ndarray], sigma 1 和 2 对应的受体邻域扣除强度; 每个值形状为 (Z, Y, X), 范围 [0, 1].
    smooth = {
        sigma: _smooth_strength(mask, sigma=sigma, radius=2.0)
        for sigma in (1.0, 2.0)
    }

    def get_normalized(operation: str, normalization: str) -> np.ndarray:
        """
        返回一个基础运算及归一化组合, 并在首次请求后复用同一数组.

        输入参数:
        - operation: str; operations 的键之一, 决定 exp、sim、diff 或 posdiff 基础网格.
        - normalization: str; nonorm 保持基础网格, clipnorm 调用 _clip_normalize.

        返回值:
        - normalized_grid: 数组, (Z, Y, X); 与输入密度逐体素对齐.
        """
        key = (operation, normalization)
        if key not in normalized:
            value = operations[operation]
            normalized[key] = value if normalization == "nonorm" else _clip_normalize(value)
        return normalized[key]

    def get_gaussian(operation: str, normalization: str, sigma: float) -> np.ndarray:
        """
        返回指定基础运算、归一化和 sigma 的反射边界 Gaussian 结果, 并在首次请求后复用.

        输入参数:
        - operation: str; operations 的键之一.
        - normalization: str; nonorm 或 clipnorm.
        - sigma: float; 三个空间轴共用的 Gaussian 标准差, 单位 voxel.

        返回值:
        - gaussian_grid: 数组, (Z, Y, X); 与输入密度逐体素对齐.
        """
        key = (operation, normalization, sigma)
        if key not in gaussian:
            gaussian[key] = ndimage.gaussian_filter(
                get_normalized(operation, normalization),
                sigma=sigma,
                mode="reflect",
            )
        return gaussian[key]

    # list[np.ndarray], 按 DENSITY_CHANNEL_NAMES 依次追加 56 个 (Z, Y, X) 通道.
    channels: list[np.ndarray] = []
    for name in DENSITY_CHANNEL_NAMES:
        operation, normalization, postprocess = name.split("_")
        # (Z, Y, X), 当前 operation 与 normalization 组合的基础网格.
        base = get_normalized(operation, normalization)
        if postprocess == "nopost":
            value = base
        elif postprocess == "gauss1":
            value = get_gaussian(operation, normalization, 1.0)
        elif postprocess == "gauss2":
            value = get_gaussian(operation, normalization, 2.0)
        elif postprocess == "DoG1":
            value = get_gaussian(operation, normalization, 1.6) - get_gaussian(
                operation,
                normalization,
                1.0,
            )
        elif postprocess == "DoG2":
            value = get_gaussian(operation, normalization, 3.2) - get_gaussian(
                operation,
                normalization,
                2.0,
            )
        elif postprocess == "smooth1":
            value = base * (1.0 - smooth[1.0])
        elif postprocess == "smooth2":
            value = base * (1.0 - smooth[2.0])
        else:
            raise AssertionError(f"未覆盖的 Find_0 后处理名称：{postprocess}")
        channels.append(value)
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def _validated_grid(value: np.ndarray, field_name: str) -> np.ndarray:
    """
    检查单个三维密度网格并规范化为 float32 数组.

    输入参数:
    - value: 数组, (Z, Y, X); 三个轴长度都必须大于 0, 全部元素必须有限.
    - field_name: str; 仅用于错误信息中点名 experimental 或 simulated 来源字段.

    返回值:
    - grid: float32 数组, (Z, Y, X); 可能与输入共享内存, 也可能因 dtype 转换产生副本.
    """
    # float32, (Z, Y, X), 待验证密度网格; 不要求固定为当前正式调用的 80³.
    grid = np.asarray(value, dtype=np.float32)
    if grid.ndim != 3 or min(grid.shape) <= 0:
        raise ValueError(f"{field_name} 必须为非空三维数组，实际为 {grid.shape}。")
    if not np.isfinite(grid).all():
        raise ValueError(f"{field_name} 含 NaN 或 Inf。")
    return grid


def _fit_scale(
    experimental: np.ndarray,
    simulated: np.ndarray,
    receptor_hardmask: np.ndarray,
) -> tuple[float, float]:
    """
    复现 Find_0 的受体优先、高密度体素线性最小二乘拟合.

    输入参数:
    - experimental: 数组, (Z, Y, X); 有限实验密度.
    - simulated: 数组, (Z, Y, X); 与 experimental 逐体素对齐的有限模拟密度.
    - receptor_hardmask: bool 数组, (Z, Y, X); 与密度逐体素对齐, True 表示受体占据体素.

    拟合体素:
    - density_filter: bool 数组, (Z, Y, X); 实验和模拟密度都位于各自最高 1.5% 分位范围的体素.
    - receptor_priority: bool 数组, (Z, Y, X); density_filter 与 receptor_hardmask 的交集.
    - fit_mask: bool 数组, (Z, Y, X); receptor_priority 的体素数达到 max(floor(0.003 * 总体素数), 10) 时优先使用, 否则使用 density_filter.

    返回值:
    - scale: float; 最小化 experimental 与 scale * simulated + offset 残差的模拟密度斜率.
    - offset: float; 对应截距 exp_mean - scale * sim_mean.

    有效拟合体素少于 10、模拟密度方差过小或回归分母过小时返回 scale=1、offset=0.
    """
    # float, 总体素数中受体优先集合需要达到的最小比例 0.3%.
    percentile = 0.003
    # int, 受体优先集合的最小体素数; 取总网格 0.3% 向下取整与 10 的较大值.
    minimum_voxels = max(int(experimental.size * percentile), 10)
    # float, 每套密度各自用于高值筛选的最高比例 1.5%.
    density_percentile = min(5.0 * percentile, 1.0)
    exp_threshold = float(np.percentile(experimental, (1.0 - density_percentile) * 100))
    sim_threshold = float(np.percentile(simulated, (1.0 - density_percentile) * 100))
    # bool, (Z, Y, X), 实验和模拟密度同时达到各自高值阈值的体素.
    density_filter = (experimental >= exp_threshold) & (simulated >= sim_threshold)
    # bool, (Z, Y, X), 高密度体素中由受体原子占据的优先拟合集合.
    receptor_priority = receptor_hardmask & density_filter
    # bool, (Z, Y, X), 最终线性回归体素; 优先集合不足 minimum_voxels 时退回全部高密度交集.
    fit_mask = receptor_priority if int(receptor_priority.sum()) >= minimum_voxels else density_filter
    if int(fit_mask.sum()) < 10:
        return 1.0, 0.0

    # (N_fit,), 最终拟合体素中的模拟密度.
    sim_values = simulated[fit_mask]
    # (N_fit,), 与 sim_values 逐位置对齐的实验密度.
    exp_values = experimental[fit_mask]
    if float(sim_values.var()) < 1e-12:
        return 1.0, 0.0
    sim_mean = float(sim_values.mean(dtype=np.float64))
    exp_mean = float(exp_values.mean(dtype=np.float64))
    # float64, (N_fit,), 去均值模拟密度; float64 用于稳定累计回归分子和分母.
    centered_sim = sim_values.astype(np.float64) - sim_mean
    # float64, (N_fit,), 与 centered_sim 逐位置对齐的去均值实验密度.
    centered_exp = exp_values.astype(np.float64) - exp_mean
    denominator = float(np.sum(centered_sim * centered_sim, dtype=np.float64))
    if denominator < 1e-12:
        return 1.0, 0.0
    scale = float(np.sum(centered_sim * centered_exp, dtype=np.float64) / denominator)
    return scale, exp_mean - scale * sim_mean


def _clip_normalize(value: np.ndarray) -> np.ndarray:
    """
    按 Find_0 的 0.1% 和 99.9% 分位截断后对整个三维网格执行 z-score.

    输入参数:
    - value: 数组, (Z, Y, X); 一个基础密度运算的有限网格.

    返回值:
    - normalized: 数组, (Z, Y, X); 低于和高于分位阈值的元素先截断, 再减去全网格均值并除以全网格标准差加 1e-8.
    """
    lower = float(np.percentile(value, 0.1))
    upper = float(np.percentile(value, 99.9))
    # (Z, Y, X), 逐元素限制到当前网格 0.1% 和 99.9% 分位之间的密度.
    clipped = np.clip(value, lower, upper)
    return (clipped - float(clipped.mean())) / (float(clipped.std()) + 1e-8)


def _smooth_strength(mask: np.ndarray, sigma: float, radius: float) -> np.ndarray:
    """
    生成受体占据体素周围的截断 Gaussian 扣除强度.

    输入参数:
    - mask: bool 数组, (Z, Y, X); True 表示受体占据体素; 函数不要求至少存在一个 True.
    - sigma: float; 距离衰减的 Gaussian 标准差, 单位 voxel.
    - radius: float; 保留扣除强度的最大欧氏距离, 单位 voxel, 距离更远的体素强制为 0.

    返回值:
    - strength: float32 数组, (Z, Y, X); mask 含受体时, 占据体素为 1、radius 内按距离衰减、radius 外为 0, 范围 [0, 1]; mask 全为 False 时保留 SciPy 对无内部零点距离变换的既有结果.
    """
    # float64, (Z, Y, X), 对 ~mask 执行的欧氏距离变换; mask 含 True 时表示到最近受体占据体素的距离, 采样间隔为 1 voxel.
    distance = ndimage.distance_transform_edt(~mask)
    # float64, (Z, Y, X), 由距离计算的 Gaussian 强度, 截断前范围 (0, 1].
    strength = np.exp(-(distance**2) / (2.0 * sigma**2))
    strength[distance > radius] = 0.0
    return strength.astype(np.float32, copy=False)
