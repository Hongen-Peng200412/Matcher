"""验证候选预训练 BEST，并只初始化完整 Matcher 的候选编码器。

``initialize_candidate_encoder_from_pretraining`` 是唯一入口：它核对 checkpoint、阶段解析配置、
阶段摘要、运行摘要、训练计划、科学数据字段和候选编码器结构，随后严格加载候选编码器参数。
已发布 release 与 launch 的路径关系及文件字节摘要由一次性运维审计留证，不进入本模块或训练主流程。
本模块不构造优化器，不选择 validation 最佳方案，也不读取 calibration 清单或特征。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, TypedDict, cast

import torch
import yaml  # type: ignore[import-untyped]
from torch import nn

from .checkpointing import load_checkpoint
from .experiment_config import DataConfig, StageConfig
from .model.config import AggregationName, MatcherModelConfig
from .schema import LocalCropStrategy


_CANDIDATE_ENCODER_CONFIG_FIELDS = (
    "aggregation",
    "use_manual_features",
    "use_sampling_features",
    "use_candidate_token_attention_bias",
    "model_dim",
    "attention_heads",
    "dropout",
    "voxel_feature_channels",
    "raunet_planes",
    "recycle_iterations",
    "gradient_checkpointing",
    "bottleneck_attention_layers",
)
_SCIENTIFIC_DATA_FIELDS = (
    "data_root",
    "stage1_root",
    "output_root",
    "coverage_threshold",
    "selection_seed",
    "pp_top_k",
    "local_crop_strategy",
)
_DATA_CONFIG_FIELDS = (
    *_SCIENTIFIC_DATA_FIELDS,
    "train_num_workers",
    "validation_num_workers",
    "prefetch_factor",
    "cache_gib_per_worker",
)


class CandidateEncoderConfigRecord(TypedDict):
    """保存与候选编码器参数和前向语义有关的规范化模型配置。

    ``aggregation`` 是 ``mean_max`` 或 ``candidate_cross_attention``。
    ``use_manual_features`` 控制 12 维无量纲候选统计及身份类别概率特征。
    ``use_sampling_features`` 控制 4 维无量纲候选裁剪统计。
    ``use_candidate_token_attention_bias`` 控制四模态实体注意力偏置。
    ``model_dim`` 是正整数候选表示宽度。
    ``attention_heads`` 是能整除模型宽度的正整数注意力头数。
    ``dropout`` 是位于 ``[0,1)`` 的无量纲丢弃概率。
    ``voxel_feature_channels`` 是正整数 RAUNet 输出通道数。
    ``raunet_planes`` 依次是 ``enc0,enc1,enc2,enc3,bottleneck,dec3,dec2,dec1,dec0`` 九个正整数。
    ``recycle_iterations`` 是正整数 RAUNet 回收轮数。
    ``gradient_checkpointing`` 控制训练时是否以重计算节省显存。
    ``bottleneck_attention_layers`` 是正整数 RAUNet 瓶颈注意力层数。
    """

    aggregation: AggregationName
    use_manual_features: bool
    use_sampling_features: bool
    use_candidate_token_attention_bias: bool
    model_dim: int
    attention_heads: int
    dropout: float
    voxel_feature_channels: int
    raunet_planes: tuple[int, ...]
    recycle_iterations: int
    gradient_checkpointing: bool
    bottleneck_attention_layers: int


class ScientificDataRecord(TypedDict):
    """保存来源与目标必须相等的七项科学数据配置。

    ``data_root`` 是非空原始数据绝对路径字符串。
    ``stage1_root`` 是非空 Stage1 固化推理产物绝对路径字符串。
    ``output_root`` 是非空 Matcher 清单与缓存绝对路径字符串。
    ``coverage_threshold`` 是位于 ``(0,1]`` 的无量纲双向覆盖率阈值。
    ``selection_seed`` 是非负 occurrence 选择随机种子。
    ``pp_top_k`` 是正整数 PP token 上限。
    ``local_crop_strategy`` 是 ``fixed_center`` 或 ``v_centroid``。
    """

    data_root: str
    stage1_root: str
    output_root: str
    coverage_threshold: float
    selection_seed: int
    pp_top_k: int
    local_crop_strategy: LocalCropStrategy


class CandidateInitializationRecord(TypedDict):
    """写入目标阶段 ``resolved.json.initialization`` 的候选预训练来源记录。

    ``checkpoint_path`` 是经过解析且文件名为 ``BEST.pt`` 的来源 checkpoint 绝对路径。
    ``source_plan_name`` 是来源候选预训练运行的计划名称。
    ``source_stage_name`` 是来源候选预训练阶段名称。
    ``source_run_stamp`` 是来源运行目录名称。
    ``source_slurm_job_id`` 是来源运行记录中的 Slurm Job ID。
    ``source_epoch`` 是来源 BEST checkpoint 的零起始 validation 最佳轮次。
    ``source_global_step`` 是来源 BEST checkpoint 保存时累计完成的参数更新数。
    ``source_best_score`` 是来源候选预训练的 validation ``evaluation.metrics.six_class.macro_f1``；该兼容字段只平均真实支持数大于 0 的类别。
    ``source_tracking_run_id`` 是来源阶段已收口的 W&B online run ID。
    ``source_tracking_url`` 是来源阶段已收口的 W&B online 曲线地址。
    ``source_candidate_model_config`` 是 ``CandidateEncoderConfigRecord``，不含身份图层数。
    ``source_scientific_data`` 是 ``ScientificDataRecord``。
    ``resolved_path`` 是来源阶段 ``resolved.json`` 绝对路径。
    ``stage_summary_path`` 是来源阶段完成后发布的 ``summary.json`` 绝对路径。
    ``run_record_path`` 是来源运行开始前发布的 ``run.json`` 绝对路径。
    ``run_summary_path`` 是来源计划全部阶段和 W&B 收口后发布的根 ``summary.json`` 绝对路径。
    ``training_plan_path`` 是来源运行记录保存的训练 YAML 绝对路径。
    ``missing_keys`` 是候选编码器严格加载后缺失的参数名；成功时为空列表。
    ``unexpected_keys`` 是候选编码器严格加载后多出的参数名；成功时为空列表。
    """

    checkpoint_path: str
    source_plan_name: str
    source_stage_name: str
    source_run_stamp: str
    source_slurm_job_id: str
    source_epoch: int
    source_global_step: int
    source_best_score: float
    source_tracking_run_id: str
    source_tracking_url: str
    source_candidate_model_config: CandidateEncoderConfigRecord
    source_scientific_data: ScientificDataRecord
    resolved_path: str
    stage_summary_path: str
    run_record_path: str
    run_summary_path: str
    training_plan_path: str
    missing_keys: list[str]
    unexpected_keys: list[str]


def _reject_json_constant(token: str) -> NoReturn:
    """作为 JSON 解码回调拒绝 NaN 与正负 Inf。"""

    raise ValueError(f"严格 JSON 不得包含非有限常量 {token}。")


def _read_json_mapping(path: Path, name: str) -> dict[str, Any]:
    """读取严格 UTF-8 JSON 对象；``name`` 只用于形成具体错误上下文。"""

    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} 必须是字符串键 JSON 对象。")
    return cast(dict[str, Any], value)


def _validate_training_plan_semantics(
    training_plan_path: Path,
    source_plan_name: str,
    source_stage_name: str,
    source_model_config: dict[str, Any],
    source_training_config: dict[str, Any],
    source_data_config: dict[str, Any],
    source_tracking_config: dict[str, Any],
) -> None:
    """核对来源运行记录指向的 YAML 与 checkpoint、resolved 和运行记录。

    ``training_plan_path`` 是来源运行摘要记录的现存训练 YAML 路径。
    ``source_plan_name`` 是来源运行根摘要中的计划名。
    ``source_stage_name`` 是来源 BEST checkpoint 中的阶段名。
    ``source_model_config`` 是 checkpoint 与 resolved 已相互核对的规范化模型配置。
    ``source_training_config`` 是 checkpoint 与 resolved 已相互核对的训练配置。
    ``source_data_config`` 是来源 resolved 的数据配置；运行时派生字段不参与 YAML 比较。
    ``source_tracking_config`` 是 resolved 与运行记录已相互核对的 W&B online 配置。
    缺少模型特征开关时迁移为 ``False``；缺少局部裁剪策略时迁移为 ``fixed_center``。
    本函数只做语义核对；已发布 release 与 launch 的路径关系及字节摘要由一次性运维审计留证。
    """

    values = yaml.safe_load(training_plan_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or not all(isinstance(key, str) for key in values):
        raise ValueError("来源训练 YAML 顶层必须是字符串键映射。")
    if values.get("plan_name") != source_plan_name:
        raise ValueError("来源训练 YAML plan_name 与运行摘要不一致。")

    tracking_value = values.get("tracking")
    if not isinstance(tracking_value, Mapping) or not all(
        isinstance(key, str) for key in tracking_value
    ):
        raise ValueError("来源训练 YAML tracking 必须是字符串键映射。")
    yaml_tracking = {str(key): value for key, value in tracking_value.items()}
    if yaml_tracking != source_tracking_config:
        raise ValueError("来源训练 YAML tracking 与 resolved/run 记录不一致。")

    data_value = values.get("data")
    if not isinstance(data_value, Mapping) or not all(isinstance(key, str) for key in data_value):
        raise ValueError("来源训练 YAML data 必须是字符串键映射。")
    yaml_data = {str(key): value for key, value in data_value.items()}
    normalized_source_data = dict(source_data_config)
    if "local_crop_strategy" not in yaml_data:
        yaml_data["local_crop_strategy"] = "fixed_center"
    for path_field in ("data_root", "stage1_root", "output_root"):
        yaml_path = yaml_data.get(path_field)
        resolved_path = normalized_source_data.get(path_field)
        if isinstance(yaml_path, str):
            yaml_data[path_field] = PurePosixPath(yaml_path.replace("\\", "/")).as_posix()
        if isinstance(resolved_path, str):
            normalized_source_data[path_field] = PurePosixPath(
                resolved_path.replace("\\", "/")
            ).as_posix()
    data_mismatches = {
        name: {
            "yaml": yaml_data.get(name),
            "resolved": normalized_source_data.get(name),
        }
        for name in _DATA_CONFIG_FIELDS
        if yaml_data.get(name) != normalized_source_data.get(name)
    }
    if data_mismatches:
        raise ValueError(f"来源训练 YAML data 与 resolved.json.data 不一致：{data_mismatches}")

    stages_value = values.get("stages")
    if not isinstance(stages_value, list):
        raise ValueError("来源训练 YAML stages 必须是列表。")
    matching_stages = [
        stage
        for stage in stages_value
        if isinstance(stage, Mapping) and stage.get("name") == source_stage_name
    ]
    if len(matching_stages) != 1:
        raise ValueError("来源训练 YAML 必须恰好含一个与 checkpoint 同名的阶段。")
    yaml_stage = matching_stages[0]
    model_value = yaml_stage.get("model")
    training_value = yaml_stage.get("training")
    if not isinstance(model_value, Mapping) or not all(isinstance(key, str) for key in model_value):
        raise ValueError("来源训练 YAML stage.model 必须是字符串键映射。")
    if not isinstance(training_value, Mapping) or not all(
        isinstance(key, str) for key in training_value
    ):
        raise ValueError("来源训练 YAML stage.training 必须是字符串键映射。")
    yaml_model_config = MatcherModelConfig.from_dict(
        {str(key): value for key, value in model_value.items()}
    ).to_dict()
    yaml_training_config = json.loads(
        json.dumps(
            {str(key): value for key, value in training_value.items()},
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    if yaml_model_config != source_model_config:
        raise ValueError("来源训练 YAML 模型配置与 checkpoint/resolved 不一致。")
    if yaml_training_config != source_training_config:
        raise ValueError("来源训练 YAML 训练配置与 checkpoint/resolved 不一致。")
    if "language_family" not in yaml_stage or "identity_variant" not in yaml_stage:
        raise ValueError("来源训练 YAML 必须显式提供 language_family 与 identity_variant。")
    if (
        yaml_stage.get("task_name") != "candidate_pretrain"
        or yaml_stage.get("language_family") is not None
        or yaml_stage.get("identity_variant") is not None
        or yaml_stage.get("identity_scope") is not None
        or "initialize_candidate_from" not in yaml_stage
        or yaml_stage.get("initialize_candidate_from") is not None
    ):
        raise ValueError("来源训练 YAML 未证明该阶段是从零训练的候选预训练。")


# ================================================================================================


def initialize_candidate_encoder_from_pretraining(
    model: nn.Module,
    checkpoint_path: Path,
    target_stage: StageConfig,
    target_data: DataConfig,
) -> CandidateInitializationRecord:
    """核对候选预训练来源并严格加载参数；任一失败都发生在优化器构造前。

    ``model`` 是尚未迁移到 GPU 的完整 Matcher，必须含 ``candidate_encoder`` 子模块。
    ``checkpoint_path`` 必须指向已完成候选预训练运行发布的 ``BEST.pt``。
    ``target_stage`` 必须是准备从该候选编码器初始化的完整 Matcher 阶段。
    ``target_data`` 提供必须与来源相等的七项科学数据配置；工作进程与缓存参数不参与比较。
    返回 ``CandidateInitializationRecord`` 并只修改 ``model.candidate_encoder`` 参数。
    """

    if target_stage.task_name != "full_matcher":
        raise ValueError("候选预训练初始化只允许用于 full_matcher 阶段。")
    resolved_checkpoint_path = checkpoint_path.resolve(strict=True)
    if resolved_checkpoint_path.name != "BEST.pt":
        raise ValueError("候选预训练初始化只接受文件名为 BEST.pt 的 checkpoint。")
    if target_stage.initialize_candidate_from is None:
        raise ValueError("目标阶段 initialize_candidate_from 必须显式指向来源 BEST.pt。")
    configured_checkpoint_path = Path(target_stage.initialize_candidate_from).resolve(strict=True)
    if configured_checkpoint_path != resolved_checkpoint_path:
        raise ValueError("目标阶段 initialize_candidate_from 与实际来源 BEST.pt 不一致。")

    checkpoint_values = cast(Mapping[str, object], load_checkpoint(resolved_checkpoint_path))
    if checkpoint_values.get("task_name") != "candidate_pretrain":
        raise ValueError("候选编码器来源 checkpoint.task_name 必须为 candidate_pretrain。")
    source_stage_name_value = checkpoint_values.get("stage_name")
    if not isinstance(source_stage_name_value, str) or not source_stage_name_value:
        raise ValueError("来源 checkpoint.stage_name 必须是非空字符串。")
    source_stage_name = source_stage_name_value

    checkpoint_integers: dict[str, int] = {}
    for name in ("epoch", "best_epoch", "global_step"):
        value = checkpoint_values.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"来源 checkpoint.{name} 必须是非负整数。")
        checkpoint_integers[name] = value
    if checkpoint_integers["epoch"] != checkpoint_integers["best_epoch"]:
        raise ValueError("来源 BEST.pt 必须满足 checkpoint.epoch == checkpoint.best_epoch。")
    best_score_value = checkpoint_values.get("best_score")
    if isinstance(best_score_value, bool) or not isinstance(best_score_value, (int, float)):
        raise ValueError("来源 checkpoint.best_score 必须是有限数值。")
    source_best_score = float(best_score_value)
    if not math.isfinite(source_best_score):
        raise ValueError("来源 checkpoint.best_score 必须是有限数值。")

    checkpoint_mappings: dict[str, dict[str, Any]] = {}
    for name in ("model_config", "training_config", "epoch_record"):
        value = checkpoint_values.get(name)
        if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
            raise ValueError(f"来源 checkpoint.{name} 必须是字符串键映射。")
        checkpoint_mappings[name] = {str(key): item for key, item in value.items()}
    epoch_record = checkpoint_mappings["epoch_record"]
    if (
        epoch_record.get("epoch") != checkpoint_integers["epoch"]
        or epoch_record.get("global_step") != checkpoint_integers["global_step"]
        or epoch_record.get("selection_score") != source_best_score
        or epoch_record.get("improved") is not True
    ):
        raise ValueError("来源 BEST.pt 的 epoch_record 必须对应同一最佳轮次、更新数和选择分数。")

    source_model = MatcherModelConfig.from_dict(checkpoint_mappings["model_config"])
    source_model_config = source_model.to_dict()
    source_training_config = json.loads(
        json.dumps(
            checkpoint_mappings["training_config"],
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    target_model_values = target_stage.model.to_dict()
    source_candidate_model_config = cast(
        CandidateEncoderConfigRecord,
        {name: source_model_config[name] for name in _CANDIDATE_ENCODER_CONFIG_FIELDS},
    )
    target_candidate_model_config = cast(
        CandidateEncoderConfigRecord,
        {name: target_model_values[name] for name in _CANDIDATE_ENCODER_CONFIG_FIELDS},
    )
    source_candidate_plain: dict[str, object] = dict(source_candidate_model_config)
    target_candidate_plain: dict[str, object] = dict(target_candidate_model_config)
    candidate_model_mismatches = {
        name: {
            "source": source_candidate_plain[name],
            "target": target_candidate_plain[name],
        }
        for name in _CANDIDATE_ENCODER_CONFIG_FIELDS
        if source_candidate_plain[name] != target_candidate_plain[name]
    }
    if candidate_model_mismatches:
        raise ValueError(f"来源与目标候选编码器结构不一致：{candidate_model_mismatches}")

    resolved_path = resolved_checkpoint_path.parent / "resolved.json"
    resolved = _read_json_mapping(resolved_path, "来源 resolved.json")
    resolved_stage_value = resolved.get("stage")
    if not isinstance(resolved_stage_value, Mapping) or not all(
        isinstance(key, str) for key in resolved_stage_value
    ):
        raise ValueError("来源 resolved.json.stage 必须是字符串键映射。")
    resolved_stage = {str(key): value for key, value in resolved_stage_value.items()}
    resolved_model_value = resolved_stage.get("model")
    resolved_training_value = resolved_stage.get("training")
    if not isinstance(resolved_model_value, Mapping) or not all(
        isinstance(key, str) for key in resolved_model_value
    ):
        raise ValueError("来源 resolved.json.stage.model 必须是字符串键映射。")
    if not isinstance(resolved_training_value, Mapping) or not all(
        isinstance(key, str) for key in resolved_training_value
    ):
        raise ValueError("来源 resolved.json.stage.training 必须是字符串键映射。")
    resolved_model_config = MatcherModelConfig.from_dict(
        {str(key): value for key, value in resolved_model_value.items()}
    ).to_dict()
    resolved_training_config = {
        str(key): value for key, value in resolved_training_value.items()
    }
    resolved_tracking_value = resolved.get("tracking")
    if not isinstance(resolved_tracking_value, Mapping) or not all(
        isinstance(key, str) for key in resolved_tracking_value
    ):
        raise ValueError("来源 resolved.json.tracking 必须是字符串键映射。")
    resolved_tracking_config = {
        str(key): value for key, value in resolved_tracking_value.items()
    }
    if "identity_scope" not in resolved_stage:
        resolved_stage["identity_scope"] = None
    if "language_family" not in resolved_stage or "identity_variant" not in resolved_stage:
        raise ValueError(
            "来源 resolved.json.stage 必须显式提供 language_family 与 identity_variant。"
        )
    if (
        resolved_stage.get("name") != source_stage_name
        or resolved_stage.get("task_name") != "candidate_pretrain"
        or resolved_stage.get("language_family") is not None
        or resolved_stage.get("identity_variant") is not None
        or resolved_stage.get("identity_scope") is not None
        or "initialize_candidate_from" not in resolved_stage
        or resolved_stage.get("initialize_candidate_from") is not None
    ):
        raise ValueError("来源 resolved.json.stage 未证明该阶段是同名、从零训练的候选预训练。")
    if resolved_model_config != source_model_config:
        raise ValueError("来源 checkpoint.model_config 与 resolved.json.stage.model 不一致。")
    if resolved_training_config != source_training_config:
        raise ValueError("来源 checkpoint.training_config 与 resolved.json.stage.training 不一致。")

    resolved_data_value = resolved.get("data")
    if not isinstance(resolved_data_value, Mapping) or not all(
        isinstance(key, str) for key in resolved_data_value
    ):
        raise ValueError("来源 resolved.json.data 必须是字符串键映射。")
    resolved_data = {str(key): value for key, value in resolved_data_value.items()}
    if "local_crop_strategy" not in resolved_data:
        resolved_data["local_crop_strategy"] = "fixed_center"
    source_path_values: dict[str, str] = {}
    for name in ("data_root", "stage1_root", "output_root"):
        value = resolved_data.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"来源 resolved.json.data.{name} 必须是非空绝对路径字符串。")
        normalized_path = PurePosixPath(value.replace("\\", "/"))
        if not normalized_path.is_absolute():
            raise ValueError(f"来源 resolved.json.data.{name} 必须是 POSIX 绝对路径。")
        source_path_values[name] = normalized_path.as_posix()
    coverage_threshold_value = resolved_data.get("coverage_threshold")
    selection_seed_value = resolved_data.get("selection_seed")
    pp_top_k_value = resolved_data.get("pp_top_k")
    local_crop_strategy_value = resolved_data.get("local_crop_strategy")
    if (
        isinstance(coverage_threshold_value, bool)
        or not isinstance(coverage_threshold_value, (int, float))
        or not math.isfinite(float(coverage_threshold_value))
        or not 0.0 < float(coverage_threshold_value) <= 1.0
    ):
        raise ValueError("来源 resolved.json.data.coverage_threshold 必须位于 (0,1]。")
    if type(selection_seed_value) is not int or selection_seed_value < 0:
        raise ValueError("来源 resolved.json.data.selection_seed 必须是非负整数。")
    if type(pp_top_k_value) is not int or pp_top_k_value <= 0:
        raise ValueError("来源 resolved.json.data.pp_top_k 必须是正整数。")
    if local_crop_strategy_value not in ("fixed_center", "v_centroid"):
        raise ValueError("来源 resolved.json.data.local_crop_strategy 不受支持。")
    source_scientific_data: ScientificDataRecord = {
        "data_root": source_path_values["data_root"],
        "stage1_root": source_path_values["stage1_root"],
        "output_root": source_path_values["output_root"],
        "coverage_threshold": float(coverage_threshold_value),
        "selection_seed": selection_seed_value,
        "pp_top_k": pp_top_k_value,
        "local_crop_strategy": cast(LocalCropStrategy, local_crop_strategy_value),
    }
    target_scientific_data: ScientificDataRecord = {
        "data_root": PurePosixPath(str(target_data.roots.data_root).replace("\\", "/")).as_posix(),
        "stage1_root": PurePosixPath(
            str(target_data.roots.stage1_root).replace("\\", "/")
        ).as_posix(),
        "output_root": PurePosixPath(
            str(target_data.roots.output_root).replace("\\", "/")
        ).as_posix(),
        "coverage_threshold": float(target_data.coverage_threshold),
        "selection_seed": target_data.selection_seed,
        "pp_top_k": target_data.pp_top_k,
        "local_crop_strategy": target_data.local_crop_strategy,
    }
    source_scientific_plain: dict[str, object] = dict(source_scientific_data)
    target_scientific_plain: dict[str, object] = dict(target_scientific_data)
    scientific_data_mismatches = {
        name: {
            "source": source_scientific_plain[name],
            "target": target_scientific_plain[name],
        }
        for name in _SCIENTIFIC_DATA_FIELDS
        if source_scientific_plain[name] != target_scientific_plain[name]
    }
    if scientific_data_mismatches:
        raise ValueError(f"来源与目标科学数据配置不一致：{scientific_data_mismatches}")

    stage_summary_path = resolved_checkpoint_path.parent / "summary.json"
    stage_summary = _read_json_mapping(stage_summary_path, "来源阶段 summary.json")
    stage_summary_best_checkpoint = stage_summary.get("best_checkpoint")
    stage_summary_last_checkpoint = stage_summary.get("last_checkpoint")
    if not isinstance(stage_summary_best_checkpoint, str):
        raise ValueError("来源阶段 summary.best_checkpoint 必须是字符串路径。")
    if not isinstance(stage_summary_last_checkpoint, str):
        raise ValueError("来源阶段 summary.last_checkpoint 必须是字符串路径。")
    stage_summary_best_path = Path(stage_summary_best_checkpoint)
    stage_summary_last_path = Path(stage_summary_last_checkpoint)
    if not stage_summary_best_path.is_absolute() or not stage_summary_last_path.is_absolute():
        raise ValueError("来源阶段 summary 的 BEST/LAST checkpoint 必须是绝对路径。")
    expected_last_checkpoint = resolved_checkpoint_path.parent / "LAST.pt"
    last_epoch = stage_summary.get("last_epoch")
    stage_global_step = stage_summary.get("global_step")
    source_tracking_run_id = stage_summary.get("tracking_run_id")
    source_tracking_url = stage_summary.get("tracking_url")
    if (
        stage_summary.get("stage") != source_stage_name
        or stage_summary.get("best_epoch") != checkpoint_integers["epoch"]
        or stage_summary.get("best_score") != source_best_score
        or type(last_epoch) is not int
        or last_epoch < checkpoint_integers["epoch"]
        or type(stage_global_step) is not int
        or stage_global_step < checkpoint_integers["global_step"]
        or stage_summary_best_path.resolve(strict=True) != resolved_checkpoint_path
        or stage_summary_last_path.resolve(strict=True) != expected_last_checkpoint
        or not isinstance(source_tracking_run_id, str)
        or not source_tracking_run_id
        or not isinstance(source_tracking_url, str)
        or not source_tracking_url.startswith("https://")
    ):
        raise ValueError("来源阶段 summary.json 未证明该 BEST checkpoint 与 W&B run 已完成收口。")

    run_directory = resolved_checkpoint_path.parent.parent
    run_record_path = run_directory / "run.json"
    run_summary_path = run_directory / "summary.json"
    run_record = _read_json_mapping(run_record_path, "来源 run.json")
    run_summary = _read_json_mapping(run_summary_path, "来源运行 summary.json")
    source_run_stamp = run_summary.get("run_stamp")
    source_slurm_job_id = run_summary.get("slurm_job_id")
    source_plan_name = run_summary.get("plan_name")
    training_plan_value = run_summary.get("plan_path")
    finished_at_utc = run_summary.get("finished_at_utc")
    root_stages = run_summary.get("stages")
    if (
        not isinstance(source_run_stamp, str)
        or source_run_stamp != run_directory.name
        or not isinstance(source_slurm_job_id, str)
        or not source_slurm_job_id
        or not isinstance(source_plan_name, str)
        or not source_plan_name
        or not isinstance(training_plan_value, str)
        or not isinstance(finished_at_utc, str)
        or not finished_at_utc
        or not isinstance(root_stages, list)
        or stage_summary not in root_stages
    ):
        raise ValueError("来源运行 summary.json 未证明计划、阶段和 Slurm 运行已经完成。")
    for name in (
        "plan_path",
        "plan_name",
        "run_stamp",
        "hostname",
        "slurm_job_id",
        "task_launch_dir",
        "tracking",
        "started_at_utc",
    ):
        if run_record.get(name) != run_summary.get(name):
            raise ValueError(f"来源 run.json.{name} 与运行 summary.json 不一致。")
    source_tracking = run_summary.get("tracking")
    if not isinstance(source_tracking, Mapping) or not all(
        isinstance(key, str) for key in source_tracking
    ):
        raise ValueError("来源运行 tracking 必须是字符串键映射。")
    source_tracking_config = {
        str(key): value for key, value in source_tracking.items()
    }
    if source_tracking_config.get("mode") != "online":
        raise ValueError("来源运行必须使用 W&B online tracking。")
    if source_tracking_config != resolved_tracking_config:
        raise ValueError("来源 resolved.json.tracking 与运行记录不一致。")
    training_plan_candidate = Path(training_plan_value)
    if not training_plan_candidate.is_absolute():
        raise ValueError("来源运行 summary.plan_path 必须是绝对路径。")
    training_plan_path = training_plan_candidate.resolve(strict=True)
    _validate_training_plan_semantics(
        training_plan_path,
        source_plan_name,
        source_stage_name,
        source_model_config,
        source_training_config,
        resolved_data,
        source_tracking_config,
    )

    target = getattr(model, "candidate_encoder", None)
    if not isinstance(target, nn.Module):
        raise TypeError("完整 Matcher 必须提供 candidate_encoder 子模块。")
    source_state = cast(dict[str, torch.Tensor], checkpoint_values["model_state"])
    if any(key.startswith("candidate_encoder.") for key in source_state):
        raise ValueError("candidate_pretrain checkpoint.model_state 必须直接保存候选编码器参数。")
    target_state = target.state_dict()
    if source_state.keys() != target_state.keys():
        raise ValueError("来源候选编码器参数名与目标结构不一致。")
    incompatible_tensors = {
        name: {
            "source_shape": tuple(source_state[name].shape),
            "target_shape": tuple(target_state[name].shape),
            "source_dtype": str(source_state[name].dtype),
            "target_dtype": str(target_state[name].dtype),
        }
        for name in source_state
        if source_state[name].shape != target_state[name].shape
        or source_state[name].dtype != target_state[name].dtype
    }
    if incompatible_tensors:
        raise ValueError(f"来源候选编码器参数形状或 dtype 与目标不一致：{incompatible_tensors}")
    load_result = target.load_state_dict(source_state, strict=True)
    return {
        "checkpoint_path": resolved_checkpoint_path.as_posix(),
        "source_plan_name": source_plan_name,
        "source_stage_name": source_stage_name,
        "source_run_stamp": source_run_stamp,
        "source_slurm_job_id": source_slurm_job_id,
        "source_epoch": checkpoint_integers["epoch"],
        "source_global_step": checkpoint_integers["global_step"],
        "source_best_score": source_best_score,
        "source_tracking_run_id": source_tracking_run_id,
        "source_tracking_url": source_tracking_url,
        "source_candidate_model_config": source_candidate_model_config,
        "source_scientific_data": source_scientific_data,
        "resolved_path": resolved_path.as_posix(),
        "stage_summary_path": stage_summary_path.as_posix(),
        "run_record_path": run_record_path.as_posix(),
        "run_summary_path": run_summary_path.as_posix(),
        "training_plan_path": training_plan_path.as_posix(),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
