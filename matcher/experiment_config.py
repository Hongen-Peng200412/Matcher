"""读取、验证并展开正式实验 YAML。

``load_experiment_plan`` 是主要入口，返回冻结数据来源、顺序阶段、模型和训练配置组成的
``ExperimentPlan``；``validate_run_stamp`` 验证外部分配的运行目录名。解析器拒绝未知字段、
静默默认、批量不一致、相对 checkpoint 和越过 `/storage/penghongen/tmp/Matcher/` 的产物路径。
"""

from __future__ import annotations

import math
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]

from .losses import IdentityLossVariant, LossWeights
from .model import MatcherModelConfig
from .schema import LANGUAGE_FAMILIES, SourceRoots


TaskName = Literal["candidate_pretrain", "full_matcher"]
LanguageFamily = Literal["molformer", "smi_ted_289m"]
Precision = Literal["bf16", "fp16", "fp32"]


@dataclass(frozen=True)
class DataConfig:
    """保存冻结数据来源、清单参数和 DataLoader 资源参数。

    ``roots`` 分别保存只读 AdaLigand 输入、只读 Stage1 Find_0 输入和可写 Matcher 产物根。
    ``coverage_threshold`` 是候选与 occurrence 双向覆盖率共同采用的命中阈值。
    ``selection_seed`` 决定每个身份固定选择哪一个有效 occurrence 语言表示。
    ``pp_top_k`` 是每个候选从完整概率图采样的 PP token 最大数。
    ``num_workers`` 是每个训练进程的 DataLoader 工作进程数。
    ``prefetch_factor`` 是每个 DataLoader 工作进程预取的物理批次数。
    ``cache_gib_per_worker`` 是单个工作进程四类 LRU 缓存的 GiB 总上限，不预分配内存。
    """

    roots: SourceRoots
    coverage_threshold: float
    selection_seed: int
    pp_top_k: int
    num_workers: int
    prefetch_factor: int
    cache_gib_per_worker: float


@dataclass(frozen=True)
class TrainingConfig:
    """保存单个阶段的优化、批量、早停和数值精度参数。

    ``seed`` 控制候选洗牌、模型初始化和训练随机源。
    ``physical_batch_size`` 是一次前向的候选数。
    ``gradient_accumulation`` 是一次参数更新包含的物理批次数。
    ``global_batch_size`` 必须等于前两项乘积，首轮正式实验固定为 32。
    ``learning_rate`` 是 AdamW 初始学习率。
    ``weight_decay`` 是 AdamW 解耦权重衰减系数。
    ``maximum_epochs`` 是可选轮数上限；None 表示不设上限。
    ``patience_bad_epochs`` 是连续完整 validation 无严格改善的停止轮数，固定为 2。
    ``precision`` 是 bf16、fp16 或 fp32 自动混合精度模式。
    ``gradient_clip_norm`` 是每个 global batch 的全模型 L2 梯度裁剪阈值。
    ``class_weights`` 是六分类权重；None 表示交叉熵不额外加权。
    ``loss_weights`` 是六分类、两个区域与闭集身份四项损失的显式权重。
    """

    seed: int
    physical_batch_size: int
    gradient_accumulation: int
    global_batch_size: int
    learning_rate: float
    weight_decay: float
    maximum_epochs: int | None
    patience_bad_epochs: int
    precision: Precision
    gradient_clip_norm: float
    class_weights: tuple[float, ...] | None
    loss_weights: LossWeights


@dataclass(frozen=True)
class StageConfig:
    """描述顺序计划中的一个训练阶段。

    ``name`` 是阶段产物目录的安全单级名称。
    ``task_name`` 选择 candidate_pretrain 或 full_matcher。
    ``language_family`` 选择当前配体语言表示；候选预训练必须为 None。
    ``identity_variant`` 选择 oracle、detached 或 joint；候选预训练必须为 None。
    ``model`` 是已完成跨字段验证的 MatcherModelConfig。
    ``training`` 是本阶段完整 TrainingConfig。
    ``initialize_candidate_from`` 是绝对 checkpoint 或此前 stage:name:BEST|LAST；None 表示从零。
    """

    name: str
    task_name: TaskName
    language_family: LanguageFamily | None
    identity_variant: IdentityLossVariant | None
    model: MatcherModelConfig
    training: TrainingConfig
    initialize_candidate_from: str | None


@dataclass(frozen=True)
class ExperimentPlan:
    """保存一个 Slurm allocation 内顺序执行的一个或多个训练阶段。

    ``plan_name`` 是实验结果目录的安全单级名称。
    ``output_root`` 是运行、阶段和 checkpoint 的根目录，必须位于 Matcher 临时产物根。
    ``data`` 是全部阶段共享的 DataConfig；其中 data.output_root 用于清单与体素缓存。
    ``stages`` 是非空有序 StageConfig 元组，stage:name 引用只能指向此前阶段。
    """

    plan_name: str
    output_root: Path
    data: DataConfig
    stages: tuple[StageConfig, ...]


def _parse_stage(values: dict[str, Any]) -> StageConfig:
    """解析并验证一个训练阶段。"""

    _require_keys(
        values,
        {
            "name",
            "task_name",
            "language_family",
            "identity_variant",
            "model",
            "training",
            "initialize_candidate_from",
        },
        "stage",
    )
    raw_task_name = values["task_name"]
    raw_language_family = values["language_family"]
    raw_identity_variant = values["identity_variant"]
    if raw_task_name not in ("candidate_pretrain", "full_matcher"):
        raise ValueError(f"不支持 task_name={raw_task_name!r}。")
    task_name = cast(TaskName, raw_task_name)
    if task_name == "candidate_pretrain":
        if raw_language_family is not None or raw_identity_variant is not None:
            raise ValueError("candidate_pretrain 不得配置语言模型或身份损失。")
        language_family = None
        identity_variant = None
    else:
        if raw_language_family not in LANGUAGE_FAMILIES:
            raise ValueError(f"full_matcher 的 language_family 必须属于 {LANGUAGE_FAMILIES}。")
        if raw_identity_variant not in ("oracle", "detached", "joint"):
            raise ValueError("full_matcher 必须选择 oracle/detached/joint。")
        language_family = cast(LanguageFamily, raw_language_family)
        identity_variant = cast(IdentityLossVariant, raw_identity_variant)
    model = MatcherModelConfig.from_dict(_mapping(values["model"], "stage.model"))
    training_values = _mapping(values["training"], "stage.training")
    expected_training = {
        "seed",
        "physical_batch_size",
        "gradient_accumulation",
        "global_batch_size",
        "learning_rate",
        "weight_decay",
        "maximum_epochs",
        "patience_bad_epochs",
        "precision",
        "gradient_clip_norm",
        "class_weights",
        "loss_weights",
    }
    _require_keys(training_values, expected_training, "stage.training")
    loss_values = _mapping(training_values["loss_weights"], "stage.training.loss_weights")
    _require_keys(
        loss_values,
        {"classification", "ligand_area", "binding_area", "identity"},
        "stage.training.loss_weights",
    )
    class_weights_value = training_values["class_weights"]
    class_weights = (
        None
        if class_weights_value is None
        else tuple(
            _finite_float(value, f"class_weights[{index}]")
            for index, value in enumerate(_sequence(class_weights_value, "class_weights"))
        )
    )
    if class_weights is not None and (
        len(class_weights) != 6
        or any(value < 0 for value in class_weights)
        or sum(class_weights) <= 0
    ):
        raise ValueError("class_weights 必须为六个非负且总和为正的值，或为 null。")
    maximum_epochs_value = training_values["maximum_epochs"]
    training = TrainingConfig(
        seed=_integer(training_values["seed"], "seed"),
        physical_batch_size=_integer(training_values["physical_batch_size"], "physical_batch_size"),
        gradient_accumulation=_integer(
            training_values["gradient_accumulation"],
            "gradient_accumulation",
        ),
        global_batch_size=_integer(training_values["global_batch_size"], "global_batch_size"),
        learning_rate=_finite_float(training_values["learning_rate"], "learning_rate"),
        weight_decay=_finite_float(training_values["weight_decay"], "weight_decay"),
        maximum_epochs=(
            None
            if maximum_epochs_value is None
            else _integer(maximum_epochs_value, "maximum_epochs")
        ),
        patience_bad_epochs=_integer(training_values["patience_bad_epochs"], "patience_bad_epochs"),
        precision=cast(Precision, _string(training_values["precision"], "precision")),
        gradient_clip_norm=_finite_float(
            training_values["gradient_clip_norm"],
            "gradient_clip_norm",
        ),
        class_weights=class_weights,
        loss_weights=LossWeights(
            classification=_finite_float(loss_values["classification"], "loss.classification"),
            ligand_area=_finite_float(loss_values["ligand_area"], "loss.ligand_area"),
            binding_area=_finite_float(loss_values["binding_area"], "loss.binding_area"),
            identity=_finite_float(loss_values["identity"], "loss.identity"),
        ),
    )
    if training.physical_batch_size * training.gradient_accumulation != training.global_batch_size:
        raise ValueError("physical_batch_size × gradient_accumulation 必须等于 global_batch_size。")
    if training.physical_batch_size <= 0 or training.gradient_accumulation <= 0:
        raise ValueError("physical_batch_size 和 gradient_accumulation 必须为正整数。")
    if training.global_batch_size != 32:
        raise ValueError("当前正式实验的 global_batch_size 固定为 32。")
    if training.seed < 0:
        raise ValueError("seed 不能为负数。")
    if training.learning_rate <= 0:
        raise ValueError("learning_rate 必须为正数。")
    if training.weight_decay < 0 or training.gradient_clip_norm <= 0:
        raise ValueError("weight_decay 不能为负，gradient_clip_norm 必须为正数。")
    if training.maximum_epochs is not None and training.maximum_epochs <= 0:
        raise ValueError("maximum_epochs 必须为正数或 null。")
    if training.patience_bad_epochs != 2:
        raise ValueError("两次完整 validation 无改善的契约要求 patience_bad_epochs=2。")
    if training.precision not in ("bf16", "fp16", "fp32"):
        raise ValueError("precision 只支持 bf16/fp16/fp32。")
    if task_name == "candidate_pretrain" and training.loss_weights.identity != 0.0:
        raise ValueError("candidate_pretrain 的 identity 损失权重必须为 0。")
    initialize_candidate_from = values["initialize_candidate_from"]
    if initialize_candidate_from is not None and not isinstance(initialize_candidate_from, str):
        raise ValueError("initialize_candidate_from 必须是字符串或 null。")
    return StageConfig(
        name=_path_component(values["name"], "stage.name"),
        task_name=task_name,
        language_family=language_family,
        identity_variant=identity_variant,
        model=model,
        training=training,
        initialize_candidate_from=initialize_candidate_from,
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """要求一个 YAML 节点是字符串键映射。"""

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} 必须是字符串键 YAML 映射。")
    return value


def _sequence(value: Any, name: str) -> list[Any] | tuple[Any, ...]:
    """要求一个 YAML 节点是列表或元组。"""

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} 必须是 YAML 列表。")
    return value


def _string(value: Any, name: str) -> str:
    """要求配置值是非空字符串，不把其它类型静默字符串化。"""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} 必须是非空字符串。")
    return value


def _path(value: Any, name: str) -> Path:
    """要求路径配置使用非空字符串。"""

    return Path(_string(value, name))


def _integer(value: Any, name: str) -> int:
    """要求整数配置不使用布尔值或可截断浮点数。"""

    if type(value) is not int:
        raise ValueError(f"{name} 必须是整数。")
    return value


def _finite_float(value: Any, name: str) -> float:
    """读取有限数值，同时拒绝布尔值、字符串、NaN 和无穷值。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数值。")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限数值。")
    return result


def _path_component(value: Any, name: str) -> str:
    """要求计划、阶段和运行名称是不能改变目录层级的单个路径组件。"""

    result = _string(value, name)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", result) is None:
        raise ValueError(f"{name} 必须是以字母或数字开头的安全路径组件。")
    return result


def _require_keys(values: dict[str, Any], expected: set[str], name: str) -> None:
    """同时拒绝缺失键和未知键，避免拼写错误被默认值静默吞掉。"""

    actual = set(values)
    if actual != expected:
        raise ValueError(
            f"{name} 字段不一致：missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}。"
        )


def _validate_output_root(path: Path, name: str) -> Path:
    """规范化路径并限制业务产物位于 Matcher 临时根或其子目录。"""

    posix_path = PurePosixPath(posixpath.normpath(path.as_posix()))
    if not posix_path.is_absolute():
        raise ValueError(f"{name}={path} 必须是绝对路径。")
    required_root = PurePosixPath("/storage/penghongen/tmp/Matcher")
    if not posix_path.is_relative_to(required_root):
        raise ValueError(f"{name}={path} 不在允许的服务器产物根 {required_root} 内。")
    return Path(posix_path.as_posix())


def _validate_stage_initialization(stages: tuple[StageConfig, ...]) -> None:
    """限制 checkpoint 初始化为绝对路径或此前已声明阶段的 BEST/LAST。"""

    previous_names: set[str] = set()
    for stage in stages:
        specification = stage.initialize_candidate_from
        if specification is not None:
            if specification.startswith("stage:"):
                parts = specification.split(":")
                if (
                    len(parts) != 3
                    or _path_component(parts[1], "initialize stage") != parts[1]
                    or parts[1] not in previous_names
                    or parts[2] not in ("BEST", "LAST")
                ):
                    raise ValueError("阶段初始化必须引用此前 stage:<name>:BEST|LAST。")
            elif not PurePosixPath(specification).is_absolute():
                raise ValueError("外部 initialize_candidate_from 必须是绝对路径。")
        previous_names.add(stage.name)


# ================================================================================================


def validate_run_stamp(value: str) -> str:
    """验证来自 allocation 的运行标识，防止结果目录越过计划根。"""

    return _path_component(value, "run_stamp")


def load_experiment_plan(path: Path) -> ExperimentPlan:
    """读取 YAML 并拒绝未知结构、批量不一致和违反任务边界的字段。"""

    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"{path} 顶层必须是 YAML 映射。")
    _require_keys(values, {"plan_name", "output_root", "data", "stages"}, "plan")
    data_values = _mapping(values["data"], "data")
    _require_keys(
        data_values,
        {
            "data_root",
            "stage1_root",
            "output_root",
            "coverage_threshold",
            "selection_seed",
            "pp_top_k",
            "num_workers",
            "prefetch_factor",
            "cache_gib_per_worker",
        },
        "data",
    )
    canonical_data_output = _validate_output_root(
        _path(data_values["output_root"], "data.output_root"),
        "data.output_root",
    )
    data = DataConfig(
        roots=SourceRoots(
            _path(data_values["data_root"], "data.data_root"),
            _path(data_values["stage1_root"], "data.stage1_root"),
            canonical_data_output,
        ),
        coverage_threshold=_finite_float(data_values["coverage_threshold"], "coverage_threshold"),
        selection_seed=_integer(data_values["selection_seed"], "selection_seed"),
        pp_top_k=_integer(data_values["pp_top_k"], "pp_top_k"),
        num_workers=_integer(data_values["num_workers"], "num_workers"),
        prefetch_factor=_integer(data_values["prefetch_factor"], "prefetch_factor"),
        cache_gib_per_worker=_finite_float(
            data_values["cache_gib_per_worker"],
            "cache_gib_per_worker",
        ),
    )
    if not 0.0 < data.coverage_threshold <= 1.0:
        raise ValueError("coverage_threshold 必须位于 (0,1]。")
    if data.pp_top_k <= 0:
        raise ValueError("pp_top_k 必须为正整数。")
    if data.num_workers <= 0 or data.prefetch_factor <= 0 or data.cache_gib_per_worker <= 0:
        raise ValueError("num_workers/prefetch_factor/cache_gib_per_worker 必须为正数。")

    raw_stages = values["stages"]
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("stages 必须为非空 YAML 列表。")
    stages = tuple(_parse_stage(_mapping(stage, "stage")) for stage in raw_stages)
    names = [stage.name for stage in stages]
    if len(names) != len(set(names)):
        raise ValueError("同一计划中的 stage.name 不能重复。")
    _validate_stage_initialization(stages)
    return ExperimentPlan(
        plan_name=_path_component(values["plan_name"], "plan_name"),
        output_root=_validate_output_root(
            _path(values["output_root"], "output_root"),
            "output_root",
        ),
        data=data,
        stages=stages,
    )
