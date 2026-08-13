"""按 Pocket Plus Find_0 契约在 80³ 候选框内构造 56 个密度通道。"""

from __future__ import annotations

import numpy as np
from scipy import ndimage  # type: ignore[import-untyped]


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


def build_density_channels(
    experimental: np.ndarray,
    simulated: np.ndarray,
    receptor_hardmask: np.ndarray,
) -> np.ndarray:
    """生成顺序固定为 operation、normalization、postprocess 的 56 通道数组。"""

    exp = _validated_grid(experimental, "experimental")
    sim = _validated_grid(simulated, "simulated")
    mask = np.asarray(receptor_hardmask, dtype=bool)
    if exp.shape != sim.shape or mask.shape != exp.shape:
        raise ValueError(
            "experimental、simulated 和 receptor_hardmask 必须共享形状，"
            f"实际为 {exp.shape}、{sim.shape}、{mask.shape}。"
        )

    scale, offset = _fit_scale(exp, sim, mask)
    difference = exp - (scale * sim + offset)
    operations = {
        "exp": exp,
        "sim": sim,
        "diff": difference,
        "posdiff": np.maximum(difference, 0.0),
    }
    normalized: dict[tuple[str, str], np.ndarray] = {}
    gaussian: dict[tuple[str, str, float], np.ndarray] = {}
    smooth = {
        sigma: _smooth_strength(mask, sigma=sigma, radius=2.0)
        for sigma in (1.0, 2.0)
    }

    def get_normalized(operation: str, normalization: str) -> np.ndarray:
        key = (operation, normalization)
        if key not in normalized:
            value = operations[operation]
            normalized[key] = value if normalization == "nonorm" else _clip_normalize(value)
        return normalized[key]

    def get_gaussian(operation: str, normalization: str, sigma: float) -> np.ndarray:
        key = (operation, normalization, sigma)
        if key not in gaussian:
            gaussian[key] = ndimage.gaussian_filter(
                get_normalized(operation, normalization),
                sigma=sigma,
                mode="reflect",
            )
        return gaussian[key]

    channels: list[np.ndarray] = []
    for name in DENSITY_CHANNEL_NAMES:
        operation, normalization, postprocess = name.split("_")
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
    """检查单个三维密度图并返回 float32 只读视图。"""

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
    """复现 Find_0 的受体优先、高密度体素最小二乘尺度拟合。"""

    percentile = 0.003
    minimum_voxels = max(int(experimental.size * percentile), 10)
    density_percentile = min(5.0 * percentile, 1.0)
    exp_threshold = float(np.percentile(experimental, (1.0 - density_percentile) * 100))
    sim_threshold = float(np.percentile(simulated, (1.0 - density_percentile) * 100))
    density_filter = (experimental >= exp_threshold) & (simulated >= sim_threshold)
    receptor_priority = receptor_hardmask & density_filter
    fit_mask = receptor_priority if int(receptor_priority.sum()) >= minimum_voxels else density_filter
    if int(fit_mask.sum()) < 10:
        return 1.0, 0.0

    sim_values = simulated[fit_mask]
    exp_values = experimental[fit_mask]
    if float(sim_values.var()) < 1e-12:
        return 1.0, 0.0
    sim_mean = float(sim_values.mean(dtype=np.float64))
    exp_mean = float(exp_values.mean(dtype=np.float64))
    centered_sim = sim_values.astype(np.float64) - sim_mean
    centered_exp = exp_values.astype(np.float64) - exp_mean
    denominator = float(np.sum(centered_sim * centered_sim, dtype=np.float64))
    if denominator < 1e-12:
        return 1.0, 0.0
    scale = float(np.sum(centered_sim * centered_exp, dtype=np.float64) / denominator)
    return scale, exp_mean - scale * sim_mean


def _clip_normalize(value: np.ndarray) -> np.ndarray:
    """按 Find_0 的 0.1%/99.9% 截断后执行逐候选 z-score。"""

    lower = float(np.percentile(value, 0.1))
    upper = float(np.percentile(value, 99.9))
    clipped = np.clip(value, lower, upper)
    return (clipped - float(clipped.mean())) / (float(clipped.std()) + 1e-8)


def _smooth_strength(mask: np.ndarray, sigma: float, radius: float) -> np.ndarray:
    """生成受体原子所属体素周围的截断高斯扣除强度。"""

    distance = ndimage.distance_transform_edt(~mask)
    strength = np.exp(-(distance**2) / (2.0 * sigma**2))
    strength[distance > radius] = 0.0
    return strength.astype(np.float32, copy=False)
