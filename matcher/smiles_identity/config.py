"""解析单一 SMILES 身份训练所需的紧凑 YAML 配置. 

主要入口 load_smiles_experiment 把正式 YAML 的 data、model、loss、optimization 和
tracking 映射转换为冻结 dataclass, 并在模型与 Dataset 构造前检查字段集合、类型和范围. 
本模块不读写训练产物. 
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .losses import IdentityLossVariant, LossWeights
from .model_config import MatcherModelConfig


@dataclass(frozen=True)
class DataConfig:
    """保存清单划分、物理批量和每个工作进程的 I/O 资源. 

    - manifest_path: Path, smiles_identity_v1 入口清单. 
    - train_split: str, 训练候选清单的划分名称; 驻留任务显式写 validation. 
    - validation_split: str, 完整验证候选清单的划分名称; 可与 train_split 相同. 
    - batch_size: int, 正整数; 单卡物理批量. 
    - accumulation_steps: int, 正整数; 每次参数更新累计的物理批次数. 
    - num_workers: int, 非负整数; DataLoader 持久工作进程数, 0 表示在主进程读取. 
    - prefetch_factor: int, 正整数; 每个 DataLoader 工作进程预取的物理批次数. 
    - pp_top_k: int, 正整数; 每个候选的固定 PP token 数上限. 
    - cache_gib_per_worker: 非负有限 float, 单个 Dataset 副本的 PDB 数组缓存预算, 单位 GiB. 
    """

    manifest_path: Path
    train_split: str
    validation_split: str
    batch_size: int
    accumulation_steps: int
    num_workers: int
    prefetch_factor: int
    pp_top_k: int
    cache_gib_per_worker: float


@dataclass(frozen=True)
class OptimizationConfig:
    """保存 AdamW、梯度裁剪和驻留停止条件. 

    - learning_rate: 正有限 float, AdamW 的共同学习率. 
    - weight_decay: 非负有限 float, AdamW 的解耦权重衰减. 
    - gradient_clip_norm: 正有限 float, 全局梯度范数阈值; 设置得足够宽松时近似不裁剪. 
    - max_epochs: int, 正整数; 最大完整数据遍历次数. 
    - max_hours: 正有限 float, 从训练入口开始计算的最长墙钟时间, 单位小时. 
    - validation_every_updates: int, 正整数; 两次完整验证之间的参数更新数. 
    - save_checkpoints: bool, 是否保存模型; 驻留任务固定为 false. 
    """

    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    max_epochs: int
    max_hours: float
    validation_every_updates: int
    save_checkpoints: bool


@dataclass(frozen=True)
class TrackingConfig:
    """保存 W&B 运行位置、模式和曲线记录间隔. 

    - mode: str, online、offline 或 disabled; disabled 不创建 W&B Run. 
    - entity: str|None, W&B 账户或团队名称; None 交由 W&B 当前登录配置决定. 
    - project: str, W&B 项目名称. 
    - group: str, W&B 运行分组名称. 
    - log_every_updates: int, 两次训练标量记录之间的参数更新数. 
    """

    mode: str
    entity: str | None
    project: str
    group: str
    log_every_updates: int


@dataclass(frozen=True)
class SmilesExperimentConfig:
    """保存一次可独立运行的单卡 Matcher 训练配置. 

    - name: str, 当前实验的单级目录名和 W&B 运行名前缀. 
    - output_root: Path, 本地运行目录共同根路径. 
    - seed: int, Python、NumPy、PyTorch、采样器和 DataLoader 随机源的基准种子. 
    - data: DataConfig, 清单划分、物理批量、worker 与缓存配置. 
    - model: MatcherModelConfig, 候选编码器与身份匹配器的完整结构配置. 
    - loss_weights: LossWeights, 六分类、两个区域和身份损失的四项系数. 
    - identity_variant: IdentityLossVariant, oracle、detached 或 joint 身份损失模式. 
    - optimization: OptimizationConfig, AdamW、梯度裁剪、验证间隔和停止条件. 
    - tracking: TrackingConfig, W&B 模式、命名和训练记录间隔. 
    """

    name: str
    output_root: Path
    seed: int
    data: DataConfig
    model: MatcherModelConfig
    loss_weights: LossWeights
    identity_variant: IdentityLossVariant
    optimization: OptimizationConfig
    tracking: TrackingConfig


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """要求一个 YAML 值是字符串键映射, 并返回浅拷贝. 

    - value: Any, YAML 顶层或子层解码值. 
    - name: str, 失败消息中用于定位该映射的配置名称. 
    - return: dict[str, Any], 与输入拥有相同键和值引用的浅拷贝. 
    """

    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} 必须是字符串键映射。")
    return dict(value)


def _construct(cls: type[Any], value: Any, converters: dict[str, type[Any]]) -> Any:
    """按 dataclass 声明字段检查映射并转换指定路径字段. 

    - cls: 目标 dataclass 类型. 
    - value: YAML 子映射; 键必须与 dataclass 字段精确相同. 
    - converters: 字段名到单参数转换器的映射, 本配置只用它构造 Path. 
    - return: cls 实例; 具体数值范围由 load_smiles_experiment 在完成构造后检查. 
    """

    mapping = _mapping(value, cls.__name__)
    expected = set(cls.__dataclass_fields__)
    if set(mapping) != expected:
        raise ValueError(f"{cls.__name__} 字段必须精确为 {sorted(expected)}。")
    converted = dict(mapping)
    for field, converter in converters.items():
        converted[field] = converter(converted[field])
    return cls(**converted)


# ================================================================================================


def load_smiles_experiment(path: Path) -> SmilesExperimentConfig:
    """从 YAML 读取全部显式字段, 并拒绝未知或缺失顶层键. 

    - path: 正式训练 YAML 路径. 
    - return: SmilesExperimentConfig, 完整嵌套字段见五个 dataclass 的字段契约; manifest_path 和 output_root 转为 Path, RAUNet 通道列表转为元组. 

    本函数不提供历史配置迁移或隐式默认值; 字段集合、枚举、类型和范围不满足当前
    契约时在构造模型或 Dataset 前失败. 
    """

    # Any, YAML 解码后的顶层值; 后续要求精确为八键映射.
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("训练 YAML 顶层必须是映射。")
    expected = {"name", "output_root", "seed", "data", "model", "loss", "optimization", "tracking"}
    if set(raw) != expected:
        raise ValueError(f"训练 YAML 顶层字段必须精确为 {sorted(expected)}。")
    if not isinstance(raw["name"], str) or not raw["name"]:
        raise TypeError("name 必须是非空字符串。")
    if not isinstance(raw["output_root"], str) or not raw["output_root"]:
        raise TypeError("output_root 必须是非空路径字符串。")
    if type(raw["seed"]) is not int:
        raise TypeError("seed 必须是整数。")
    # DataConfig, data 映射的全部显式字段; manifest_path 在构造前转为 Path.
    data = _construct(DataConfig, raw["data"], {"manifest_path": Path})
    optimization = _construct(OptimizationConfig, raw["optimization"], {})
    tracking = _construct(TrackingConfig, raw["tracking"], {})
    # dict[str, Any], loss 子映射的浅拷贝; variant 取出后其余四键构造 LossWeights.
    loss_mapping = _mapping(raw["loss"], "loss")
    if set(loss_mapping) != {"classification", "ligand_area", "binding_area", "identity", "identity_variant"}:
        raise ValueError("loss 必须含四项权重和 identity_variant。")
    variant = loss_mapping.pop("identity_variant")
    if variant not in ("oracle", "detached", "joint"):
        raise ValueError("loss.identity_variant 必须是 oracle、detached 或 joint。")
    # dict[str, Any], model 子映射的浅拷贝; MatcherModelConfig.from_dict 负责完整字段门禁.
    model_mapping = _mapping(raw["model"], "model")
    model = MatcherModelConfig.from_dict(model_mapping)
    config = SmilesExperimentConfig(
        name=raw["name"],
        output_root=Path(raw["output_root"]),
        seed=raw["seed"],
        data=data,
        model=model,
        loss_weights=LossWeights(**loss_mapping),
        identity_variant=variant,
        optimization=optimization,
        tracking=tracking,
    )
    if not config.name or Path(config.name).name != config.name:
        raise ValueError("name 必须是非空单级目录名。")
    if not isinstance(config.data.train_split, str) or not config.data.train_split:
        raise TypeError("data.train_split 必须是非空字符串。")
    if not isinstance(config.data.validation_split, str) or not config.data.validation_split:
        raise TypeError("data.validation_split 必须是非空字符串。")
    integers = (
        config.data.batch_size,
        config.data.accumulation_steps,
        config.data.prefetch_factor,
        config.data.pp_top_k,
        config.optimization.max_epochs,
        config.optimization.validation_every_updates,
        config.tracking.log_every_updates,
    )
    if any(type(value) is not int or value <= 0 for value in integers):
        raise ValueError("批量、累积、预取、PP、epoch 和记录间隔必须是正整数。")
    if type(config.data.num_workers) is not int or config.data.num_workers < 0:
        raise ValueError("num_workers 必须是非负整数。")
    finite_positive = (
        config.optimization.learning_rate,
        config.optimization.gradient_clip_norm,
        config.optimization.max_hours,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in finite_positive
    ):
        raise ValueError("学习率、梯度裁剪阈值和最长小时数必须是有限正数。")
    if (
        isinstance(config.optimization.weight_decay, bool)
        or not isinstance(config.optimization.weight_decay, (int, float))
        or not math.isfinite(config.optimization.weight_decay)
        or config.optimization.weight_decay < 0
    ):
        raise ValueError("weight_decay 必须是有限非负数。")
    if (
        isinstance(config.data.cache_gib_per_worker, bool)
        or not isinstance(config.data.cache_gib_per_worker, (int, float))
        or not math.isfinite(config.data.cache_gib_per_worker)
        or config.data.cache_gib_per_worker < 0
    ):
        raise ValueError("cache_gib_per_worker 必须是有限非负数。")
    if type(config.optimization.save_checkpoints) is not bool:
        raise TypeError("save_checkpoints 必须是布尔值。")
    if config.tracking.mode not in ("online", "offline", "disabled"):
        raise ValueError("tracking.mode 必须是 online、offline 或 disabled。")
    if config.tracking.entity is not None and not isinstance(config.tracking.entity, str):
        raise TypeError("tracking.entity 必须是字符串或 null。")
    if not isinstance(config.tracking.project, str) or not config.tracking.project:
        raise TypeError("tracking.project 必须是非空字符串。")
    if not isinstance(config.tracking.group, str) or not config.tracking.group:
        raise TypeError("tracking.group 必须是非空字符串。")
    return config
