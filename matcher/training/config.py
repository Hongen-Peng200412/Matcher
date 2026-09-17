"""解析四个正式 Matcher 变体共用的紧凑训练 YAML.

主要入口是 load_smiles_experiment. 本模块把一个 YAML 严格转换为数据、模型、损失、优化和
追踪 dataclass, 拒绝未知字段与非法值. 它只读取指定配置文件, 不创建运行目录或写产物.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..model.config import MatcherModelConfig
from .losses import LossWeights


@dataclass(frozen=True)
class DataConfig:
    """
    保存唯一 manifest、候选过滤和 DataLoader 字段.

    - manifest_path: 冻结 manifest.json 路径.
    - train_split / validation_split: manifest.splits 中的训练和验证键.
    - selected_only: True 时训练只保留 candidate_selected=True; validation 始终由 engine 固定为已选候选.
    - batch_size: 每个物理批次的候选数.
    - accumulation_steps: 每次参数更新累计的物理批次数; 两者乘积是名义 global batch.
    - num_workers / prefetch_factor: DataLoader 持久工作进程数和每进程预取批次数.
    - pp_top_k: strongest 与 trunk_only 每候选保留的 PP token 上限.
    - cache_gib_per_worker: 每个 Dataset 副本的 PDB 载荷最近使用缓存预算.
    """
    manifest_path: Path
    train_split: str
    validation_split: str
    selected_only: bool
    batch_size: int
    accumulation_steps: int
    num_workers: int
    prefetch_factor: int
    pp_top_k: int
    cache_gib_per_worker: float


@dataclass(frozen=True)
class OptimizationConfig:
    """
    保存 AdamW、梯度裁剪和停止条件.

    - learning_rate / weight_decay: AdamW 初始学习率和权重衰减.
    - gradient_clip_norm: 每次参数更新前的全局梯度范数上限.
    - max_epochs: 最多完成的训练与完整 validation 轮数.
    - patience_bad_epochs: 连续多少次完整 validation 未严格改善 selection 后早停.
    """
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    max_epochs: int
    patience_bad_epochs: int


@dataclass(frozen=True)
class TrackingConfig:
    """
    保存 W&B 定位和记录间隔.

    - mode: online、offline 或 disabled.
    - entity: W&B 实体名称; None 使用登录账户默认实体.
    - project / group: 四项正式实验共同使用的 W&B 项目和分组.
    - log_every_updates: 每多少次参数更新记录训练标量.
    """
    mode: str
    entity: str | None
    project: str
    group: str
    log_every_updates: int


@dataclass(frozen=True)
class SmilesExperimentConfig:
    """
    保存一次可独立从零运行的正式实验.

    - name: 输出子目录和 W&B 运行名前缀.
    - output_root: 服务器训练产物根目录.
    - seed: Python、NumPy、PyTorch、采样器和 DataLoader 共用随机种子.
    - data / model / loss_weights / optimization / tracking: 经过各自 dataclass 固定的 数据、结构、损失、优化和追踪字段.
    """
    name: str
    output_root: Path
    seed: int
    data: DataConfig
    model: MatcherModelConfig
    loss_weights: LossWeights
    optimization: OptimizationConfig
    tracking: TrackingConfig


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """
    核对一个 YAML 映射.

    - value: yaml.safe_load 产生的任意值.
    - name: 错误信息使用的字段或配置块名称.
    - return: 字符串键映射的浅拷贝; 非映射或非字符串键立即失败.
    """
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} 必须是字符串键映射。")
    return dict(value)


def _construct(cls: type[Any], value: Any, path_fields: tuple[str, ...] = ()) -> Any:
    """
    从精确字段映射构造一个配置 dataclass.

    - cls: 目标 dataclass 类型.
    - value: YAML 中对应配置块.
    - path_fields: 构造前需要从字符串转换成 Path 的字段名.
    - return: cls 实例; 缺字段、额外字段或键类型错误时失败关闭.
    """
    mapping = _mapping(value, cls.__name__)
    expected = set(cls.__dataclass_fields__)
    if set(mapping) != expected:
        raise ValueError(f"{cls.__name__} 字段必须精确为 {sorted(expected)}。")
    for field in path_fields:
        mapping[field] = Path(mapping[field])
    return cls(**mapping)


def load_smiles_experiment(path: Path) -> SmilesExperimentConfig:
    """
    读取并核对当前版本训练 YAML.

    - path: 四份正式 YAML 之一.
    - return: 字段集合、类型、有限性和跨字段边界均已核对的 SmilesExperimentConfig.

    加载器不迁移历史模型、损失或初始化开关; 活动 YAML 出现未知字段时立即失败.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {"name", "output_root", "seed", "data", "model", "loss", "optimization", "tracking"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f"训练 YAML 顶层字段必须精确为 {sorted(expected)}。")
    if not isinstance(raw["name"], str) or not raw["name"] or Path(raw["name"]).name != raw["name"]:
        raise ValueError("name 必须是非空单级目录名。")
    if not isinstance(raw["output_root"], str) or not raw["output_root"]:
        raise TypeError("output_root 必须是非空路径字符串。")
    if type(raw["seed"]) is not int:
        raise TypeError("seed 必须是整数。")
    data = _construct(DataConfig, raw["data"], ("manifest_path",))
    model = MatcherModelConfig.from_dict(_mapping(raw["model"], "model"))
    config = SmilesExperimentConfig(
        name=raw["name"],
        output_root=Path(raw["output_root"]),
        seed=raw["seed"],
        data=data,
        model=model,
        loss_weights=_construct(LossWeights, raw["loss"]),
        optimization=_construct(OptimizationConfig, raw["optimization"]),
        tracking=_construct(TrackingConfig, raw["tracking"]),
    )
    text_fields = (config.data.train_split, config.data.validation_split)
    if any(not isinstance(value, str) or not value for value in text_fields):
        raise TypeError("train_split 和 validation_split 必须是非空字符串。")
    if type(config.data.selected_only) is not bool:
        raise TypeError("data.selected_only 必须是布尔值。")
    positive_integers = (
        config.data.batch_size,
        config.data.accumulation_steps,
        config.data.prefetch_factor,
        config.data.pp_top_k,
        config.optimization.max_epochs,
        config.optimization.patience_bad_epochs,
        config.tracking.log_every_updates,
    )
    if any(type(value) is not int or value <= 0 for value in positive_integers):
        raise ValueError("批量、累积、预取、PP、epoch 和记录间隔必须是正整数。")
    if type(config.data.num_workers) is not int or config.data.num_workers < 0:
        raise ValueError("num_workers 必须是非负整数。")
    positive_floats = (
        config.optimization.learning_rate,
        config.optimization.gradient_clip_norm,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in positive_floats
    ):
        raise ValueError("学习率和梯度裁剪阈值必须是有限正数。")
    if (
        isinstance(config.optimization.weight_decay, bool)
        or not isinstance(config.optimization.weight_decay, (int, float))
        or not math.isfinite(float(config.optimization.weight_decay))
        or float(config.optimization.weight_decay) < 0
    ):
        raise ValueError("weight_decay 必须是有限非负数。")
    if (
        isinstance(config.data.cache_gib_per_worker, bool)
        or not isinstance(config.data.cache_gib_per_worker, (int, float))
        or not math.isfinite(float(config.data.cache_gib_per_worker))
        or float(config.data.cache_gib_per_worker) < 0
    ):
        raise ValueError("cache_gib_per_worker 必须是有限非负数。")
    if config.tracking.mode not in ("online", "offline", "disabled"):
        raise ValueError("tracking.mode 必须是 online、offline 或 disabled。")
    if config.tracking.entity is not None and not isinstance(config.tracking.entity, str):
        raise TypeError("tracking.entity 必须是字符串或 null。")
    if not isinstance(config.tracking.project, str) or not config.tracking.project:
        raise TypeError("tracking.project 必须是非空字符串。")
    if not isinstance(config.tracking.group, str) or not config.tracking.group:
        raise TypeError("tracking.group 必须是非空字符串。")
    if config.model.variant == "pocket_only" and (
        config.loss_weights.ligand_area != 0 or config.loss_weights.binding_area != 0
    ):
        raise ValueError("pocket_only 的两个区域辅助损失权重必须为 0。")
    return config
