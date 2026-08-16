"""验证已完成候选预训练的 BEST.pt，并只初始化完整 Matcher 的候选编码器。

initialize_candidate_encoder_from_pretraining 交叉核对 checkpoint、resolved.json、阶段 summary、
根 run.json、根 summary.json、原训练 YAML、候选编码器结构和七项科学数据配置。全部门禁通过
后才调用 load_state_dict，因此预检查失败不会修改目标参数。本模块不构造优化器、不重新选取
validation 最佳轮次，也不读取 calibration。release、launch 与文件字节摘要只属于一次性运维
审计，不进入训练可执行主链或返回记录。
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
from ..data.schema import LocalCropStrategy
from ..model.config import AggregationName, MatcherModelConfig
from .config import DataConfig, StageConfig


# tuple[str,...]，影响候选编码器参数形状或前向语义、必须在来源与目标间逐值相等的模型字段顺序。
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
# tuple[str,...]，来源候选预训练与目标完整Matcher必须逐值相等的七项科学数据字段顺序。
_SCIENTIFIC_DATA_FIELDS = (
    "data_root",
    "stage1_root",
    "output_root",
    "coverage_threshold",
    "selection_seed",
    "pp_top_k",
    "local_crop_strategy",
)
# tuple[str,...]，核对来源YAML data与resolved.json.data时使用的全部十一项字段顺序。
_DATA_CONFIG_FIELDS = (
    *_SCIENTIFIC_DATA_FIELDS,
    "train_num_workers",
    "validation_num_workers",
    "prefetch_factor",
    "cache_gib_per_worker",
)


class CandidateEncoderConfigRecord(TypedDict):
    """保存来源与目标必须相等的候选编码器模型字段。

    - aggregation: mean_max、candidate_cross_attention 或 mean_max_candidate_cross_attention_concat；第三项把前两种完整摘要拼接为零输出残差。
    - use_manual_features: 精确 bool；控制 12 维候选手工特征以及完整模型配对中的类别概率辅助特征。
    - use_sampling_features: 精确 bool；控制 4 维候选裁剪统计。
    - use_candidate_token_attention_bias: 精确 bool；控制四模态候选—token 注意力偏置。
    - model_dim: 正整数候选表示宽度 D_m。
    - attention_heads: 正整数候选注意力头数，必须整除 model_dim。
    - dropout: [0,1) 内有限无量纲丢弃概率。
    - voxel_feature_channels: 正整数 RAUNet 输出通道数。
    - raunet_planes: 九项正整数元组，依次为 enc0、enc1、enc2、enc3、bottleneck、dec3、dec2、dec1、dec0。
    - recycle_iterations: 正整数 RAUNet 回收轮数。
    - gradient_checkpointing: 精确 bool；控制训练且外层梯度启用时的激活重计算。
    - bottleneck_attention_layers: 正整数 RAUNet 瓶颈注意力层数。

    身份门初值和 graph_layers 不属于候选编码器，故不参与来源—目标候选结构相等门禁。
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
    """保存来源与目标必须逐值相等的七项科学数据配置。

    - data_root: POSIX 绝对路径字符串；AdaLigand 数据来源根。
    - stage1_root: POSIX 绝对路径字符串；Stage1 Find_0 产物根。
    - output_root: POSIX 绝对路径字符串；Matcher 清单与缓存根。
    - coverage_threshold: (0,1] 内有限无量纲双向覆盖率阈值。
    - selection_seed: 非负 Python 整数；联合身份 occurrence 语言表示选择种子。
    - pp_top_k: 正整数；每个候选 PP token 上限。
    - local_crop_strategy: fixed_center 或 v_centroid。
    """

    data_root: str
    stage1_root: str
    output_root: str
    coverage_threshold: float
    selection_seed: int
    pp_top_k: int
    local_crop_strategy: LocalCropStrategy


class CandidateInitializationRecord(TypedDict):
    """声明目标 resolved.json.initialization 保存的来源追溯记录。

    - checkpoint_path: resolve(strict=True) 后文件名为 BEST.pt 的来源 checkpoint 绝对 POSIX 路径。
    - source_plan_name: 根运行 summary 中的非空计划名。
    - source_stage_name: checkpoint、resolved、阶段 summary 与训练 YAML 共同确认的候选预训练阶段名。
    - source_run_stamp: 与来源运行目录名相等的运行标识。
    - source_slurm_job_id: 根运行 summary 中的非空字符串 Slurm Job ID。
    - source_epoch: BEST checkpoint 的非负零起始 epoch，且等于 best_epoch。
    - source_global_step: BEST checkpoint 保存时累计完成的非负参数更新数。
    - source_best_score: 有限 validation 六分类 macro_f1；该兼容口径只平均支持数大于零的类别。
    - source_tracking_run_id: 来源阶段 summary 保存的非空 W&B run ID；本记录不查询或证明远端 run 状态。
    - source_tracking_url: 来源阶段 summary 保存且以 https:// 开头的 W&B 地址；本记录不验证远端可访问性。
    - source_candidate_model_config: 来源候选编码器配置，不含两个身份门初值和 graph_layers。
    - source_candidate_model_config.aggregation: 来源聚合方式。
    - source_candidate_model_config.use_manual_features: 来源手工特征开关。
    - source_candidate_model_config.use_sampling_features: 来源裁剪统计开关。
    - source_candidate_model_config.use_candidate_token_attention_bias: 来源候选—token 注意力偏置开关。
    - source_candidate_model_config.model_dim: 来源候选表示宽度。
    - source_candidate_model_config.attention_heads: 来源候选注意力头数。
    - source_candidate_model_config.dropout: 来源丢弃概率。
    - source_candidate_model_config.voxel_feature_channels: 来源 RAUNet 输出通道数。
    - source_candidate_model_config.raunet_planes: 来源九项 RAUNet 通道元组。
    - source_candidate_model_config.recycle_iterations: 来源回收轮数。
    - source_candidate_model_config.gradient_checkpointing: 来源激活重计算开关。
    - source_candidate_model_config.bottleneck_attention_layers: 来源瓶颈注意力层数。
    - source_scientific_data: 来源七项科学数据配置。
    - source_scientific_data.data_root: 来源 AdaLigand 数据根 POSIX 绝对路径。
    - source_scientific_data.stage1_root: 来源 Stage1 Find_0 根 POSIX 绝对路径。
    - source_scientific_data.output_root: 来源 Matcher 清单与缓存根 POSIX 绝对路径。
    - source_scientific_data.coverage_threshold: 来源双向覆盖率阈值。
    - source_scientific_data.selection_seed: 来源 occurrence 选择种子。
    - source_scientific_data.pp_top_k: 来源 PP token 上限。
    - source_scientific_data.local_crop_strategy: 来源局部裁剪策略。
    - resolved_path: 来源阶段 resolved.json 绝对 POSIX 路径。
    - stage_summary_path: 来源阶段 summary.json 绝对 POSIX 路径。
    - run_record_path: 来源根 run.json 绝对 POSIX 路径。
    - run_summary_path: 来源根 summary.json 绝对 POSIX 路径。
    - training_plan_path: 根 summary.plan_path 指向且 resolve(strict=True) 成功的训练 YAML 绝对 POSIX 路径。
    - missing_keys: strict=True 加载成功后的缺失参数名列表，成功路径为空。
    - unexpected_keys: strict=True 加载成功后的额外参数名列表，成功路径为空。
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
    """作为 json.loads 回调拒绝非标准非有限常量。

    - token: 解码器遇到的 NaN、Infinity 或 -Infinity 原文字串。

    本函数总是抛出 ValueError，不返回控制流。
    """

    raise ValueError(f"严格 JSON 不得包含非有限常量 {token}。")


def _read_json_mapping(path: Path, name: str) -> dict[str, Any]:
    """读取 UTF-8 严格 JSON，并要求根值是字符串键对象。

    - path: 必须存在的 JSON 文件路径。
    - name: 错误上下文名称，只用于根结构异常消息。

    - 返回值: 字符串键到未递归验证 JSON 值的新普通 dict；嵌套结构由调用方逐项核验。

    parse_constant 回调拒绝 NaN 与正负 Infinity；重复键仍遵循 Python json.loads 的后值覆盖语义。
    """

    if not path.is_file():
        raise FileNotFoundError(path)
    # object，严格常量门禁后的JSON根值；随后要求为字符串键普通dict。
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
    """核对来源根记录指向的 YAML 与已交叉验证运行语义。

    - training_plan_path: 根 summary.plan_path 指向且已经 resolve(strict=True) 的 UTF-8 YAML 路径。
    - source_plan_name: 根 summary 的非空计划名；必须等于 YAML plan_name。
    - source_stage_name: BEST checkpoint 的非空阶段名；YAML stages 必须恰有一个同名映射。
    - source_model_config: checkpoint 与 resolved 已相等的完整模型配置普通 dict。
    - source_model_config.aggregation: 聚合方式。
    - source_model_config.use_manual_features: 手工特征开关；历史缺失值由 MatcherModelConfig.from_dict 迁移。
    - source_model_config.use_sampling_features: 裁剪统计开关；历史缺失值由 MatcherModelConfig.from_dict 迁移。
    - source_model_config.use_candidate_token_attention_bias: 候选—token 注意力偏置开关；历史缺失值由模型配置迁移。
    - source_model_config.initial_graph_gate: 图门初值。
    - source_model_config.initial_inventory_gate: 清单门初值。
    - source_model_config.model_dim: 表示宽度。
    - source_model_config.attention_heads: 候选与身份注意力头数。
    - source_model_config.dropout: 丢弃概率。
    - source_model_config.voxel_feature_channels: RAUNet 输出通道数。
    - source_model_config.raunet_planes: 九项 RAUNet 通道序列。
    - source_model_config.recycle_iterations: 回收轮数。
    - source_model_config.gradient_checkpointing: 激活重计算开关。
    - source_model_config.bottleneck_attention_layers: 瓶颈注意力层数。
    - source_model_config.graph_layers: 图消息层数。
    - source_training_config: checkpoint 与 resolved 已相等、可由严格 JSON 表示的训练配置普通 dict。
    - source_training_config.seed: 随机种子。
    - source_training_config.physical_batch_size: 物理批量。
    - source_training_config.gradient_accumulation: 梯度累积次数。
    - source_training_config.global_batch_size: 全局批量。
    - source_training_config.learning_rate: 初始学习率。
    - source_training_config.weight_decay: 权重衰减。
    - source_training_config.maximum_epochs: 轮数上限或 None。
    - source_training_config.patience_bad_epochs: 早停坏轮数。
    - source_training_config.precision: bf16、fp16 或 fp32。
    - source_training_config.gradient_clip_norm: 梯度裁剪阈值。
    - source_training_config.class_weights: 六项类别权重列表或 None。
    - source_training_config.loss_weights: 四项损失权重映射。
    - source_training_config.loss_weights.classification: 六分类损失系数。
    - source_training_config.loss_weights.ligand_area: 配体区域损失系数。
    - source_training_config.loss_weights.binding_area: 结合区域损失系数。
    - source_training_config.loss_weights.identity: 身份损失系数。
    - source_data_config: resolved.json.data 完整映射；只比较 _DATA_CONFIG_FIELDS 十一项，运行时派生字段忽略。
    - source_data_config.data_root: AdaLigand 数据根路径字符串。
    - source_data_config.stage1_root: Stage1 Find_0 根路径字符串。
    - source_data_config.output_root: Matcher 清单与缓存根路径字符串。
    - source_data_config.coverage_threshold: 双向覆盖率阈值。
    - source_data_config.selection_seed: occurrence 选择种子。
    - source_data_config.pp_top_k: PP token 上限。
    - source_data_config.local_crop_strategy: fixed_center、v_centroid，或历史记录缺失后迁移为 fixed_center。
    - source_data_config.train_num_workers: train worker 数。
    - source_data_config.validation_num_workers: validation worker 数。
    - source_data_config.prefetch_factor: 每 worker 预取批次数。
    - source_data_config.cache_gib_per_worker: 每 Dataset 实例缓存 GiB 预算。
    - source_tracking_config: resolved 与根运行记录已相等的七字段 W&B 配置。
    - source_tracking_config.provider: wandb。
    - source_tracking_config.mode: online。
    - source_tracking_config.entity: W&B 账户或团队名。
    - source_tracking_config.project: W&B 项目名。
    - source_tracking_config.group: W&B 比较组名。
    - source_tracking_config.log_every_global_steps: 参数更新写入间隔。
    - source_tracking_config.init_timeout_seconds: 在线初始化超时秒数。

    YAML 路径字符串在比较前统一反斜杠并按 PurePosixPath 词法表示；模型配置通过
    MatcherModelConfig.from_dict 执行历史字段迁移，local_crop_strategy 缺失时迁移为
    fixed_center。训练配置经 JSON 往返以统一 tuple/list 表示。该函数只比较语义，不读取
    release、launch、文件摘要或 calibration。
    """

    # object，来源训练YAML根节点；要求为字符串键普通dict。
    values = yaml.safe_load(training_plan_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or not all(isinstance(key, str) for key in values):
        raise ValueError("来源训练 YAML 顶层必须是字符串键映射。")
    if values.get("plan_name") != source_plan_name:
        raise ValueError("来源训练 YAML plan_name 与运行摘要不一致。")

    # object，YAML tracking未验证节点。
    tracking_value = values.get("tracking")
    if not isinstance(tracking_value, Mapping) or not all(
        isinstance(key, str) for key in tracking_value
    ):
        raise ValueError("来源训练 YAML tracking 必须是字符串键映射。")
    # dict[str,Any]，保留YAML tracking值并规范为普通字符串键dict。
    yaml_tracking = {str(key): value for key, value in tracking_value.items()}
    if yaml_tracking != source_tracking_config:
        raise ValueError("来源训练 YAML tracking 与 resolved/run 记录不一致。")

    # object，YAML data未验证节点。
    data_value = values.get("data")
    if not isinstance(data_value, Mapping) or not all(isinstance(key, str) for key in data_value):
        raise ValueError("来源训练 YAML data 必须是字符串键映射。")
    # dict[str,Any]，将要做历史裁剪字段迁移与POSIX路径规范化的YAML data副本。
    yaml_data = {str(key): value for key, value in data_value.items()}
    # dict[str,Any]，来源resolved data副本；仅修改三个路径字段的比较表示。
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
    # dict[str,dict[str,Any]]，十一项正式data字段中YAML与resolved不相等的逐字段双值记录。
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

    # object，YAML stages未验证节点；正式契约要求list。
    stages_value = values.get("stages")
    if not isinstance(stages_value, list):
        raise ValueError("来源训练 YAML stages 必须是列表。")
    # list[Mapping[Any,Any]]，stage.name与来源checkpoint阶段名相等的YAML阶段映射。
    matching_stages = [
        stage
        for stage in stages_value
        if isinstance(stage, Mapping) and stage.get("name") == source_stage_name
    ]
    if len(matching_stages) != 1:
        raise ValueError("来源训练 YAML 必须恰好含一个与 checkpoint 同名的阶段。")
    # Mapping[Any,Any]，唯一同名来源候选预训练阶段。
    yaml_stage = matching_stages[0]
    # object，同名YAML阶段的未验证model节点。
    model_value = yaml_stage.get("model")
    # object，同名YAML阶段的未验证training节点。
    training_value = yaml_stage.get("training")
    if not isinstance(model_value, Mapping) or not all(isinstance(key, str) for key in model_value):
        raise ValueError("来源训练 YAML stage.model 必须是字符串键映射。")
    if not isinstance(training_value, Mapping) or not all(
        isinstance(key, str) for key in training_value
    ):
        raise ValueError("来源训练 YAML stage.training 必须是字符串键映射。")
    # dict[str,object]，经模型门禁和历史字段迁移后的规范化YAML模型配置。
    yaml_model_config = MatcherModelConfig.from_dict(
        {str(key): value for key, value in model_value.items()}
    ).to_dict()
    # object，JSON往返后的YAML训练配置；tuple等容器被统一为JSON列表表示。
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
    """交叉核对候选预训练来源，并在全部显式门禁通过后严格加载候选编码器。

    - model: 必须提供 candidate_encoder 属性的 nn.Module；正式训练在 model.to(device) 与优化器构造前调用，本函数不检查模型当前设备。
    - checkpoint_path: 必须 resolve(strict=True) 成功且文件名恰为 BEST.pt。
    - target_stage: 当前完整 Matcher 阶段的 StageConfig。
    - target_stage.name: 目标阶段名，本函数不参与来源相等比较。
    - target_stage.task_name: 必须为 full_matcher。
    - target_stage.language_family: 本函数不读取。
    - target_stage.identity_variant: 本函数不读取。
    - target_stage.identity_scope: 本函数不读取。
    - target_stage.model: 提供目标候选编码器结构；只比较 _CANDIDATE_ENCODER_CONFIG_FIELDS 十二项。
    - target_stage.model.aggregation: 目标聚合方式。
    - target_stage.model.use_manual_features: 目标手工特征开关。
    - target_stage.model.use_sampling_features: 目标裁剪统计开关。
    - target_stage.model.use_candidate_token_attention_bias: 目标候选—token 注意力偏置开关。
    - target_stage.model.initial_graph_gate: 本函数不比较的目标图门初值。
    - target_stage.model.initial_inventory_gate: 本函数不比较的目标清单门初值。
    - target_stage.model.model_dim: 目标候选表示宽度。
    - target_stage.model.attention_heads: 目标候选注意力头数。
    - target_stage.model.dropout: 目标丢弃概率。
    - target_stage.model.voxel_feature_channels: 目标 RAUNet 输出通道数。
    - target_stage.model.raunet_planes: 目标九项 RAUNet 通道元组。
    - target_stage.model.recycle_iterations: 目标回收轮数。
    - target_stage.model.gradient_checkpointing: 目标激活重计算开关。
    - target_stage.model.bottleneck_attention_layers: 目标瓶颈注意力层数。
    - target_stage.model.graph_layers: 本函数不比较的目标配体图消息层数。
    - target_stage.training: 本函数不读取的完整目标训练配置；优化器尚未构造。
    - target_stage.training.seed: 本函数不读取的目标随机种子。
    - target_stage.training.physical_batch_size: 本函数不读取的目标物理批量。
    - target_stage.training.gradient_accumulation: 本函数不读取的目标梯度累计次数。
    - target_stage.training.global_batch_size: 本函数不读取的目标每次参数更新候选数。
    - target_stage.training.learning_rate: 本函数不读取的目标初始学习率。
    - target_stage.training.weight_decay: 本函数不读取的目标权重衰减。
    - target_stage.training.maximum_epochs: 本函数不读取的目标轮数上限。
    - target_stage.training.patience_bad_epochs: 本函数不读取的目标早停坏轮数阈值。
    - target_stage.training.precision: 本函数不读取的目标混合精度模式。
    - target_stage.training.gradient_clip_norm: 本函数不读取的目标梯度裁剪阈值。
    - target_stage.training.class_weights: 本函数不读取的目标六分类权重或 None。
    - target_stage.training.loss_weights: 本函数不读取的目标四项损失系数映射。
    - target_stage.training.loss_weights.classification: 本函数不读取的目标六分类损失系数。
    - target_stage.training.loss_weights.ligand_area: 本函数不读取的目标配体区域损失系数。
    - target_stage.training.loss_weights.binding_area: 本函数不读取的目标结合区域损失系数。
    - target_stage.training.loss_weights.identity: 本函数不读取的目标身份损失系数。
    - target_stage.initialize_candidate_from: 必须非空，resolve(strict=True) 后与 checkpoint_path 相同。
    - target_data: 当前实验的完整 DataConfig。
    - target_data.roots: 目标数据三根路径。
    - target_data.roots.data_root: 必须与来源 resolved data_root 的规范化 POSIX 字符串相等。
    - target_data.roots.stage1_root: 必须与来源 resolved stage1_root 的规范化 POSIX 字符串相等。
    - target_data.roots.output_root: 必须与来源 resolved output_root 的规范化 POSIX 字符串相等。
    - target_data.coverage_threshold: 必须与来源有限 (0,1] 值精确相等。
    - target_data.selection_seed: 必须与来源非负整数精确相等。
    - target_data.pp_top_k: 必须与来源正整数精确相等。
    - target_data.local_crop_strategy: 必须与来源 fixed_center 或 v_centroid 精确相等。
    - target_data.train_num_workers: 资源字段，不参与来源相等比较。
    - target_data.validation_num_workers: 资源字段，不参与来源相等比较。
    - target_data.prefetch_factor: 资源字段，不参与来源相等比较。
    - target_data.cache_gib_per_worker: 资源字段，不参与来源相等比较。

    - 来源 BEST.pt.schema_version: 由 load_checkpoint 要求等于 CHECKPOINT_SCHEMA_VERSION。
    - 来源 BEST.pt.task_name: 必须为 candidate_pretrain。
    - 来源 BEST.pt.stage_name: 非空候选预训练阶段名。
    - 来源 BEST.pt.epoch: 非负零起始轮次，必须等于 best_epoch。
    - 来源 BEST.pt.best_epoch: 非负最佳轮次，必须等于 epoch。
    - 来源 BEST.pt.global_step: 来源保存时的非负累计参数更新数。
    - 来源 BEST.pt.best_score: 有限 validation 选择分数。
    - 来源 BEST.pt.model_config: 完整模型配置字符串键映射；经 MatcherModelConfig.from_dict 规范化。
    - 来源 BEST.pt.model_config.aggregation: 候选摘要聚合方式。
    - 来源 BEST.pt.model_config.use_manual_features: 12 维手工特征开关。
    - 来源 BEST.pt.model_config.use_sampling_features: 4 维裁剪统计开关。
    - 来源 BEST.pt.model_config.use_candidate_token_attention_bias: 候选—token 注意力偏置开关。
    - 来源 BEST.pt.model_config.initial_graph_gate: 来源候选阶段必须为 0 的图门初值。
    - 来源 BEST.pt.model_config.initial_inventory_gate: 来源候选阶段必须为 0 的清单门初值。
    - 来源 BEST.pt.model_config.model_dim: 候选与身份共同表示宽度。
    - 来源 BEST.pt.model_config.attention_heads: 候选、图与清单注意力共用头数。
    - 来源 BEST.pt.model_config.dropout: 模型丢弃概率。
    - 来源 BEST.pt.model_config.voxel_feature_channels: RAUNet 输出通道数。
    - 来源 BEST.pt.model_config.raunet_planes: 九项 RAUNet 通道序列。
    - 来源 BEST.pt.model_config.recycle_iterations: RAUNet 回收轮数。
    - 来源 BEST.pt.model_config.gradient_checkpointing: 激活重计算开关。
    - 来源 BEST.pt.model_config.bottleneck_attention_layers: RAUNet 瓶颈注意力层数。
    - 来源 BEST.pt.model_config.graph_layers: 配体图消息传递层数；不参与候选编码器结构比较。
    - 来源 BEST.pt.training_config: 完整训练配置字符串键映射；经严格 JSON 往返统一容器表示。
    - 来源 BEST.pt.training_config.seed: 来源训练随机种子。
    - 来源 BEST.pt.training_config.physical_batch_size: 来源物理批量。
    - 来源 BEST.pt.training_config.gradient_accumulation: 来源梯度累计次数。
    - 来源 BEST.pt.training_config.global_batch_size: 来源每次参数更新候选数。
    - 来源 BEST.pt.training_config.learning_rate: 来源阶段初始学习率。
    - 来源 BEST.pt.training_config.weight_decay: 来源 AdamW 权重衰减。
    - 来源 BEST.pt.training_config.maximum_epochs: 来源轮数上限或 null。
    - 来源 BEST.pt.training_config.patience_bad_epochs: 来源早停坏轮数阈值。
    - 来源 BEST.pt.training_config.precision: 来源 bf16、fp16 或 fp32 模式。
    - 来源 BEST.pt.training_config.gradient_clip_norm: 来源全模型梯度裁剪阈值。
    - 来源 BEST.pt.training_config.class_weights: 来源六项分类权重列表或 null。
    - 来源 BEST.pt.training_config.loss_weights: 来源四项损失系数映射。
    - 来源 BEST.pt.training_config.loss_weights.classification: 来源六分类损失系数。
    - 来源 BEST.pt.training_config.loss_weights.ligand_area: 来源配体区域损失系数。
    - 来源 BEST.pt.training_config.loss_weights.binding_area: 来源结合区域损失系数。
    - 来源 BEST.pt.training_config.loss_weights.identity: 来源候选预训练身份损失系数，正式值为 0。
    - 来源 BEST.pt.model_state: 字符串参数名到张量的候选编码器直属状态；不得含 candidate_encoder. 前缀。
    - 来源 BEST.pt.epoch_record: BEST 保存时的字符串键 EpochRecord；本函数只交叉核对四个字段。
    - 来源 BEST.pt.epoch_record.epoch: 必须等于来源 BEST.pt.epoch。
    - 来源 BEST.pt.epoch_record.global_step: 必须等于来源 BEST.pt.global_step。
    - 来源 BEST.pt.epoch_record.selection_score: 必须等于来源 BEST.pt.best_score。
    - 来源 BEST.pt.epoch_record.improved: 必须精确为 True。
    - 来源 resolved.json.stage: 来源阶段配置字符串键映射。
    - 来源 resolved.json.stage.name: 必须等于来源 BEST.pt.stage_name。
    - 来源 resolved.json.stage.task_name: 必须为 candidate_pretrain。
    - 来源 resolved.json.stage.language_family: 必须显式为 null。
    - 来源 resolved.json.stage.identity_variant: 必须显式为 null。
    - 来源 resolved.json.stage.identity_scope: 必须为 null；历史缺失时迁移为 null。
    - 来源 resolved.json.stage.model: 经模型门禁和历史迁移后必须等于来源 BEST.pt.model_config。
    - 来源 resolved.json.stage.model.aggregation: 候选摘要聚合方式。
    - 来源 resolved.json.stage.model.use_manual_features: 手工特征开关。
    - 来源 resolved.json.stage.model.use_sampling_features: 裁剪统计开关。
    - 来源 resolved.json.stage.model.use_candidate_token_attention_bias: 候选—token 注意力偏置开关。
    - 来源 resolved.json.stage.model.initial_graph_gate: 图门初值。
    - 来源 resolved.json.stage.model.initial_inventory_gate: 清单门初值。
    - 来源 resolved.json.stage.model.model_dim: 共同表示宽度。
    - 来源 resolved.json.stage.model.attention_heads: 候选、图与清单注意力共用头数。
    - 来源 resolved.json.stage.model.dropout: 模型丢弃概率。
    - 来源 resolved.json.stage.model.voxel_feature_channels: RAUNet 输出通道数。
    - 来源 resolved.json.stage.model.raunet_planes: 九项 RAUNet 通道序列。
    - 来源 resolved.json.stage.model.recycle_iterations: RAUNet 回收轮数。
    - 来源 resolved.json.stage.model.gradient_checkpointing: 激活重计算开关。
    - 来源 resolved.json.stage.model.bottleneck_attention_layers: RAUNet 瓶颈注意力层数。
    - 来源 resolved.json.stage.model.graph_layers: 配体图消息传递层数。
    - 来源 resolved.json.stage.training: 必须等于来源 BEST.pt.training_config。
    - 来源 resolved.json.stage.training.seed: 来源训练随机种子。
    - 来源 resolved.json.stage.training.physical_batch_size: 来源物理批量。
    - 来源 resolved.json.stage.training.gradient_accumulation: 来源梯度累计次数。
    - 来源 resolved.json.stage.training.global_batch_size: 来源每次参数更新候选数。
    - 来源 resolved.json.stage.training.learning_rate: 来源阶段初始学习率。
    - 来源 resolved.json.stage.training.weight_decay: 来源权重衰减。
    - 来源 resolved.json.stage.training.maximum_epochs: 来源轮数上限或 null。
    - 来源 resolved.json.stage.training.patience_bad_epochs: 来源早停坏轮数阈值。
    - 来源 resolved.json.stage.training.precision: 来源精度模式。
    - 来源 resolved.json.stage.training.gradient_clip_norm: 来源梯度裁剪阈值。
    - 来源 resolved.json.stage.training.class_weights: 来源六项分类权重列表或 null。
    - 来源 resolved.json.stage.training.loss_weights: 来源四项损失系数映射。
    - 来源 resolved.json.stage.training.loss_weights.classification: 来源六分类损失系数。
    - 来源 resolved.json.stage.training.loss_weights.ligand_area: 来源配体区域损失系数。
    - 来源 resolved.json.stage.training.loss_weights.binding_area: 来源结合区域损失系数。
    - 来源 resolved.json.stage.training.loss_weights.identity: 来源身份损失系数。
    - 来源 resolved.json.stage.initialize_candidate_from: 必须显式为 null，证明来源候选阶段从零训练。
    - 来源 resolved.json.data: 来源数据与 worker 字段字符串键映射；科学相等只取七项。
    - 来源 resolved.json.data.data_root: 非空 POSIX 绝对路径，必须与目标相等。
    - 来源 resolved.json.data.stage1_root: 非空 POSIX 绝对路径，必须与目标相等。
    - 来源 resolved.json.data.output_root: 非空 POSIX 绝对路径，必须与目标相等。
    - 来源 resolved.json.data.coverage_threshold: (0,1] 内有限数值，必须与目标相等。
    - 来源 resolved.json.data.selection_seed: 非负精确整数，必须与目标相等。
    - 来源 resolved.json.data.pp_top_k: 正的精确整数，必须与目标相等。
    - 来源 resolved.json.data.local_crop_strategy: fixed_center 或 v_centroid，必须与目标相等；历史缺失时迁移为 fixed_center。
    - 来源 resolved.json.tracking: 七字段 W&B 配置字符串键映射，必须与根运行记录相等。
    - 来源 resolved.json.tracking.provider: 正式值为 wandb。
    - 来源 resolved.json.tracking.mode: 必须为 online。
    - 来源 resolved.json.tracking.entity: W&B 账户或团队名。
    - 来源 resolved.json.tracking.project: W&B 项目名。
    - 来源 resolved.json.tracking.group: W&B 比较组名。
    - 来源 resolved.json.tracking.log_every_global_steps: 参数更新曲线记录间隔。
    - 来源 resolved.json.tracking.init_timeout_seconds: W&B 初始化超时秒数。
    - 来源阶段 summary.json.stage: 必须等于来源阶段名。
    - 来源阶段 summary.json.best_epoch: 必须等于来源 BEST.pt.epoch。
    - 来源阶段 summary.json.best_score: 必须等于来源 BEST.pt.best_score。
    - 来源阶段 summary.json.last_epoch: 精确非负整数且不得早于 best_epoch。
    - 来源阶段 summary.json.global_step: 精确非负整数且不得小于 BEST 保存时的 global_step。
    - 来源阶段 summary.json.best_checkpoint: 绝对路径，解析后必须等于实际来源 BEST.pt。
    - 来源阶段 summary.json.last_checkpoint: 绝对路径，解析后必须等于同阶段目录的 LAST.pt。
    - 来源阶段 summary.json.tracking_run_id: 本地文件保存的非空 W&B run ID；阶段 summary 自身不证明 finish(0) 返回或远端同步完成。
    - 来源阶段 summary.json.tracking_url: 本地文件保存且以 https:// 开头的 W&B 地址；本函数不访问该 URL。
    - 来源 run.json.plan_path: 与根 summary.json.plan_path 相等的来源训练 YAML 路径字符串。
    - 来源 run.json.plan_name: 与根 summary.json.plan_name 相等的计划名。
    - 来源 run.json.run_stamp: 与根 summary.json.run_stamp 相等的运行标识。
    - 来源 run.json.hostname: 与根 summary.json.hostname 相等的主机名事实。
    - 来源 run.json.slurm_job_id: 与根 summary.json.slurm_job_id 相等的 Slurm Job ID。
    - 来源 run.json.task_launch_dir: 与根 summary.json.task_launch_dir 相等的启动目录事实或 null。
    - 来源 run.json.tracking: 与根 summary.json.tracking 相等的七字段 W&B 配置映射。
    - 来源 run.json.started_at_utc: 与根 summary.json.started_at_utc 相等的 UTC 时间字符串。
    - 来源根 summary.json.plan_path: 必须是可解析到既有文件的绝对训练 YAML 路径。
    - 来源根 summary.json.plan_name: 必须是非空字符串，并与来源训练 YAML plan_name 相等。
    - 来源根 summary.json.run_stamp: 必须等于来源运行目录名。
    - 来源根 summary.json.hostname: 只与 run.json 对应字段比较，本函数不另做类型或非空门禁。
    - 来源根 summary.json.slurm_job_id: 必须是非空字符串，并与 run.json 对应字段相等。
    - 来源根 summary.json.task_launch_dir: 只与 run.json 对应字段比较，可为 null。
    - 来源根 summary.json.tracking: 字符串键映射，mode 必须为 online，且与 resolved.json.tracking 和 run.json 相等。
    - 来源根 summary.json.started_at_utc: 只与 run.json 对应字段比较，本函数不解析时间。
    - 来源根 summary.json.finished_at_utc: 必须是非空字符串，本函数不解析时间。
    - 来源根 summary.json.stages: 必须是列表且包含与来源阶段 summary.json 完全相等的元素。
    - 来源训练 YAML.plan_name: 必须等于来源根 summary.json.plan_name。
    - 来源训练 YAML.tracking: 必须与 resolved 和根运行记录的七字段 W&B 配置映射相等。
    - 来源训练 YAML.tracking.provider: 正式值为 wandb。
    - 来源训练 YAML.tracking.mode: 必须为 online。
    - 来源训练 YAML.tracking.entity: W&B 账户或团队名。
    - 来源训练 YAML.tracking.project: W&B 项目名。
    - 来源训练 YAML.tracking.group: W&B 比较组名。
    - 来源训练 YAML.tracking.log_every_global_steps: 参数更新曲线记录间隔。
    - 来源训练 YAML.tracking.init_timeout_seconds: W&B 初始化超时秒数。
    - 来源训练 YAML.data: 比较十一项正式 DataConfig 字段；三个路径先统一为 POSIX 词法表示。
    - 来源训练 YAML.data.data_root: 来源 AdaLigand 数据根。
    - 来源训练 YAML.data.stage1_root: 来源 Stage1 Find_0 根。
    - 来源训练 YAML.data.output_root: 来源 Matcher 清单与缓存根。
    - 来源训练 YAML.data.coverage_threshold: 来源双向覆盖率阈值。
    - 来源训练 YAML.data.selection_seed: 来源 occurrence 选择种子。
    - 来源训练 YAML.data.pp_top_k: 来源 PP token 上限。
    - 来源训练 YAML.data.local_crop_strategy: 来源局部裁剪策略；历史缺失时迁移为 fixed_center。
    - 来源训练 YAML.data.train_num_workers: 来源 train worker 数。
    - 来源训练 YAML.data.validation_num_workers: 来源 validation worker 数。
    - 来源训练 YAML.data.prefetch_factor: 来源每 worker 预取批次数。
    - 来源训练 YAML.data.cache_gib_per_worker: 来源每 Dataset 实例缓存 GiB 预算。
    - 来源训练 YAML.stages: 必须恰有一个 name 等于来源阶段名的映射。
    - 来源训练 YAML.stages[*].task_name: 同名来源阶段必须为 candidate_pretrain。
    - 来源训练 YAML.stages[*].language_family: 同名来源阶段必须显式为 null。
    - 来源训练 YAML.stages[*].identity_variant: 同名来源阶段必须显式为 null。
    - 来源训练 YAML.stages[*].identity_scope: 同名来源阶段必须为 null；历史 YAML 缺失该键也按 null 接受。
    - 来源训练 YAML.stages[*].model: 经模型门禁和历史迁移后必须等于 checkpoint 与 resolved 模型配置。
    - 来源训练 YAML.stages[*].model.aggregation: 候选摘要聚合方式。
    - 来源训练 YAML.stages[*].model.use_manual_features: 手工特征开关。
    - 来源训练 YAML.stages[*].model.use_sampling_features: 裁剪统计开关。
    - 来源训练 YAML.stages[*].model.use_candidate_token_attention_bias: 候选—token 注意力偏置开关。
    - 来源训练 YAML.stages[*].model.initial_graph_gate: 图门初值。
    - 来源训练 YAML.stages[*].model.initial_inventory_gate: 清单门初值。
    - 来源训练 YAML.stages[*].model.model_dim: 共同表示宽度。
    - 来源训练 YAML.stages[*].model.attention_heads: 候选、图与清单注意力共用头数。
    - 来源训练 YAML.stages[*].model.dropout: 模型丢弃概率。
    - 来源训练 YAML.stages[*].model.voxel_feature_channels: RAUNet 输出通道数。
    - 来源训练 YAML.stages[*].model.raunet_planes: 九项 RAUNet 通道序列。
    - 来源训练 YAML.stages[*].model.recycle_iterations: RAUNet 回收轮数。
    - 来源训练 YAML.stages[*].model.gradient_checkpointing: 激活重计算开关。
    - 来源训练 YAML.stages[*].model.bottleneck_attention_layers: RAUNet 瓶颈注意力层数。
    - 来源训练 YAML.stages[*].model.graph_layers: 配体图消息传递层数。
    - 来源训练 YAML.stages[*].training: 经严格 JSON 往返后必须等于 checkpoint 与 resolved 训练配置。
    - 来源训练 YAML.stages[*].training.seed: 来源训练随机种子。
    - 来源训练 YAML.stages[*].training.physical_batch_size: 来源物理批量。
    - 来源训练 YAML.stages[*].training.gradient_accumulation: 来源梯度累计次数。
    - 来源训练 YAML.stages[*].training.global_batch_size: 来源每次参数更新候选数。
    - 来源训练 YAML.stages[*].training.learning_rate: 来源初始学习率。
    - 来源训练 YAML.stages[*].training.weight_decay: 来源权重衰减。
    - 来源训练 YAML.stages[*].training.maximum_epochs: 来源轮数上限或 null。
    - 来源训练 YAML.stages[*].training.patience_bad_epochs: 来源早停坏轮数阈值。
    - 来源训练 YAML.stages[*].training.precision: 来源精度模式。
    - 来源训练 YAML.stages[*].training.gradient_clip_norm: 来源梯度裁剪阈值。
    - 来源训练 YAML.stages[*].training.class_weights: 来源六项分类权重列表或 null。
    - 来源训练 YAML.stages[*].training.loss_weights: 来源四项损失系数映射。
    - 来源训练 YAML.stages[*].training.loss_weights.classification: 来源六分类损失系数。
    - 来源训练 YAML.stages[*].training.loss_weights.ligand_area: 来源配体区域损失系数。
    - 来源训练 YAML.stages[*].training.loss_weights.binding_area: 来源结合区域损失系数。
    - 来源训练 YAML.stages[*].training.loss_weights.identity: 来源候选预训练身份损失系数，正式值为 0。
    - 来源训练 YAML.stages[*].initialize_candidate_from: 必须显式为 null。

    - 返回值.checkpoint_path: 解析后的来源 BEST.pt 绝对 POSIX 路径。
    - 返回值.source_plan_name: 已由根记录与训练 YAML 共同确认的来源计划名。
    - 返回值.source_stage_name: 已由 checkpoint、resolved、阶段 summary 与 YAML 共同确认的来源阶段名。
    - 返回值.source_run_stamp: 与来源运行目录名相等的运行标识。
    - 返回值.source_slurm_job_id: 来源根 summary 的非空 Slurm Job ID。
    - 返回值.source_epoch: 来源 BEST 的非负最佳 epoch。
    - 返回值.source_global_step: 来源 BEST 保存时的非负累计更新数。
    - 返回值.source_best_score: 来源 BEST 的有限 validation 选择分数。
    - 返回值.source_tracking_run_id: 来源阶段 summary 保存的非空 W&B run ID；不代表本函数核验过远端 run 状态。
    - 返回值.source_tracking_url: 来源阶段 summary 保存的 HTTPS W&B URL；不代表本函数访问过该地址。
    - 返回值.source_candidate_model_config: 与目标逐值相等的十二项候选编码器配置。
    - 返回值.source_candidate_model_config.aggregation: 来源聚合方式。
    - 返回值.source_candidate_model_config.use_manual_features: 来源手工特征开关。
    - 返回值.source_candidate_model_config.use_sampling_features: 来源裁剪统计开关。
    - 返回值.source_candidate_model_config.use_candidate_token_attention_bias: 来源注意力偏置开关。
    - 返回值.source_candidate_model_config.model_dim: 来源候选表示宽度。
    - 返回值.source_candidate_model_config.attention_heads: 来源候选注意力头数。
    - 返回值.source_candidate_model_config.dropout: 来源丢弃概率。
    - 返回值.source_candidate_model_config.voxel_feature_channels: 来源 RAUNet 输出通道数。
    - 返回值.source_candidate_model_config.raunet_planes: 来源九项 RAUNet 通道元组。
    - 返回值.source_candidate_model_config.recycle_iterations: 来源回收轮数。
    - 返回值.source_candidate_model_config.gradient_checkpointing: 来源激活重计算开关。
    - 返回值.source_candidate_model_config.bottleneck_attention_layers: 来源瓶颈注意力层数。
    - 返回值.source_scientific_data: 与目标逐值相等的七项科学数据配置。
    - 返回值.source_scientific_data.data_root: 来源 AdaLigand 数据根。
    - 返回值.source_scientific_data.stage1_root: 来源 Stage1 Find_0 根。
    - 返回值.source_scientific_data.output_root: 来源 Matcher 清单与缓存根。
    - 返回值.source_scientific_data.coverage_threshold: 来源覆盖阈值。
    - 返回值.source_scientific_data.selection_seed: 来源 occurrence 选择种子。
    - 返回值.source_scientific_data.pp_top_k: 来源 PP token 上限。
    - 返回值.source_scientific_data.local_crop_strategy: 来源局部裁剪策略。
    - 返回值.resolved_path: 来源阶段 resolved.json 绝对 POSIX 路径。
    - 返回值.stage_summary_path: 来源阶段 summary.json 绝对 POSIX 路径。
    - 返回值.run_record_path: 来源根 run.json 绝对 POSIX 路径。
    - 返回值.run_summary_path: 来源根 summary.json 绝对 POSIX 路径。
    - 返回值.training_plan_path: 来源训练 YAML 解析后的绝对 POSIX 路径。
    - 返回值.missing_keys: 最终 strict=True 加载结果的缺失键列表，成功时为空。
    - 返回值.unexpected_keys: 最终 strict=True 加载结果的额外键列表，成功时为空。

    来源训练 YAML 的顶层 output_root 不参与本函数的来源等价比较；科学数据的 output_root
    指的是 YAML data.output_root，并与 resolved 和目标逐值比较。显式来源、字段、键、shape
    与 dtype 门禁全部发生在最终 load_state_dict 之前；这些预检查
    失败不会修改目标。最终调用只修改 model.candidate_encoder 的参数与缓冲区，不修改目标
    身份分支。阶段 summary 单独存在只证明本地字段与路径关系；根 summary 含该阶段记录则说明
    run_plan 曾在 finish(0) 正常返回后接收阶段结果，但本函数不查询 W&B 远端状态。本函数在
    优化器构造前由训练引擎调用，不读取 calibration 或网络资源。
    """

    if target_stage.task_name != "full_matcher":
        raise ValueError("候选预训练初始化只允许用于 full_matcher 阶段。")
    # Path，必须存在且已解析符号链接的来源checkpoint绝对路径。
    resolved_checkpoint_path = checkpoint_path.resolve(strict=True)
    if resolved_checkpoint_path.name != "BEST.pt":
        raise ValueError("候选预训练初始化只接受文件名为 BEST.pt 的 checkpoint。")
    if target_stage.initialize_candidate_from is None:
        raise ValueError("目标阶段 initialize_candidate_from 必须显式指向来源 BEST.pt。")
    # Path，目标YAML声明路径解析后的绝对文件，必须与实际来源完全相同。
    configured_checkpoint_path = Path(target_stage.initialize_candidate_from).resolve(strict=True)
    if configured_checkpoint_path != resolved_checkpoint_path:
        raise ValueError("目标阶段 initialize_candidate_from 与实际来源 BEST.pt 不一致。")

    # Mapping[str,object]，通过最小checkpoint schema门禁的完整来源顶层映射。
    checkpoint_values = cast(Mapping[str, object], load_checkpoint(resolved_checkpoint_path))
    if checkpoint_values.get("task_name") != "candidate_pretrain":
        raise ValueError("候选编码器来源 checkpoint.task_name 必须为 candidate_pretrain。")
    # object，来源checkpoint.stage_name的未验证值。
    source_stage_name_value = checkpoint_values.get("stage_name")
    if not isinstance(source_stage_name_value, str) or not source_stage_name_value:
        raise ValueError("来源 checkpoint.stage_name 必须是非空字符串。")
    # str，来源候选预训练阶段非空名称。
    source_stage_name = source_stage_name_value

    # dict[str,int]，来源checkpoint的epoch、best_epoch与global_step三个非负整数。
    checkpoint_integers: dict[str, int] = {}
    for name in ("epoch", "best_epoch", "global_step"):
        value = checkpoint_values.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"来源 checkpoint.{name} 必须是非负整数。")
        checkpoint_integers[name] = value
    if checkpoint_integers["epoch"] != checkpoint_integers["best_epoch"]:
        raise ValueError("来源 BEST.pt 必须满足 checkpoint.epoch == checkpoint.best_epoch。")
    # object，来源checkpoint.best_score的未验证标量。
    best_score_value = checkpoint_values.get("best_score")
    if isinstance(best_score_value, bool) or not isinstance(best_score_value, (int, float)):
        raise ValueError("来源 checkpoint.best_score 必须是有限数值。")
    # float，来源BEST的validation选择分数；随后拒绝NaN与正负无穷。
    source_best_score = float(best_score_value)
    if not math.isfinite(source_best_score):
        raise ValueError("来源 checkpoint.best_score 必须是有限数值。")

    # dict[str,dict[str,Any]]，来源checkpoint中model_config、training_config与epoch_record的普通dict副本。
    checkpoint_mappings: dict[str, dict[str, Any]] = {}
    for name in ("model_config", "training_config", "epoch_record"):
        value = checkpoint_values.get(name)
        if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
            raise ValueError(f"来源 checkpoint.{name} 必须是字符串键映射。")
        checkpoint_mappings[name] = {str(key): item for key, item in value.items()}
    # dict[str,Any]，BEST保存时的完整EpochRecord；此处只交叉核对四个最佳轮次关键字段。
    epoch_record = checkpoint_mappings["epoch_record"]
    if (
        epoch_record.get("epoch") != checkpoint_integers["epoch"]
        or epoch_record.get("global_step") != checkpoint_integers["global_step"]
        or epoch_record.get("selection_score") != source_best_score
        or epoch_record.get("improved") is not True
    ):
        raise ValueError("来源 BEST.pt 的 epoch_record 必须对应同一最佳轮次、更新数和选择分数。")

    # MatcherModelConfig，来源checkpoint模型配置经过类型、范围和历史迁移后的对象。
    source_model = MatcherModelConfig.from_dict(checkpoint_mappings["model_config"])
    # dict[str,object]，来源完整模型配置的规范化普通字典。
    source_model_config = source_model.to_dict()
    # object，来源训练配置经严格JSON往返后的值；预期并随后作为dict参与相等比较。
    source_training_config = json.loads(
        json.dumps(
            checkpoint_mappings["training_config"],
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    # dict[str,object]，目标完整模型配置的规范化普通字典。
    target_model_values = target_stage.model.to_dict()
    # CandidateEncoderConfigRecord，从来源完整模型抽取的十二项候选编码器配置。
    source_candidate_model_config = cast(
        CandidateEncoderConfigRecord,
        {name: source_model_config[name] for name in _CANDIDATE_ENCODER_CONFIG_FIELDS},
    )
    # CandidateEncoderConfigRecord，从目标完整模型抽取的十二项候选编码器配置。
    target_candidate_model_config = cast(
        CandidateEncoderConfigRecord,
        {name: target_model_values[name] for name in _CANDIDATE_ENCODER_CONFIG_FIELDS},
    )
    # dict[str,object]，用于字段差异推导的来源候选配置普通映射。
    source_candidate_plain: dict[str, object] = dict(source_candidate_model_config)
    # dict[str,object]，用于字段差异推导的目标候选配置普通映射。
    target_candidate_plain: dict[str, object] = dict(target_candidate_model_config)
    # dict[str,dict[str,object]]，来源与目标不相等的候选编码器字段及双侧值。
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

    # Path，与来源BEST.pt同阶段目录的resolved.json。
    resolved_path = resolved_checkpoint_path.parent / "resolved.json"
    # dict[str,Any]，来源阶段完整解析配置记录。
    resolved = _read_json_mapping(resolved_path, "来源 resolved.json")
    # object，resolved.json.stage未验证节点。
    resolved_stage_value = resolved.get("stage")
    if not isinstance(resolved_stage_value, Mapping) or not all(
        isinstance(key, str) for key in resolved_stage_value
    ):
        raise ValueError("来源 resolved.json.stage 必须是字符串键映射。")
    # dict[str,Any]，来源阶段配置的普通dict副本；可能应用历史identity_scope迁移。
    resolved_stage = {str(key): value for key, value in resolved_stage_value.items()}
    # object，resolved stage.model未验证节点。
    resolved_model_value = resolved_stage.get("model")
    # object，resolved stage.training未验证节点。
    resolved_training_value = resolved_stage.get("training")
    if not isinstance(resolved_model_value, Mapping) or not all(
        isinstance(key, str) for key in resolved_model_value
    ):
        raise ValueError("来源 resolved.json.stage.model 必须是字符串键映射。")
    if not isinstance(resolved_training_value, Mapping) or not all(
        isinstance(key, str) for key in resolved_training_value
    ):
        raise ValueError("来源 resolved.json.stage.training 必须是字符串键映射。")
    # dict[str,object]，来源resolved模型配置经模型门禁与历史迁移后的规范化字典。
    resolved_model_config = MatcherModelConfig.from_dict(
        {str(key): value for key, value in resolved_model_value.items()}
    ).to_dict()
    # dict[str,Any]，来源resolved训练配置的普通字符串键副本。
    resolved_training_config = {
        str(key): value for key, value in resolved_training_value.items()
    }
    # object，来源resolved tracking未验证节点。
    resolved_tracking_value = resolved.get("tracking")
    if not isinstance(resolved_tracking_value, Mapping) or not all(
        isinstance(key, str) for key in resolved_tracking_value
    ):
        raise ValueError("来源 resolved.json.tracking 必须是字符串键映射。")
    # dict[str,Any]，来源resolved七字段W&B配置普通字符串键副本。
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

    # object，来源resolved data未验证节点。
    resolved_data_value = resolved.get("data")
    if not isinstance(resolved_data_value, Mapping) or not all(
        isinstance(key, str) for key in resolved_data_value
    ):
        raise ValueError("来源 resolved.json.data 必须是字符串键映射。")
    # dict[str,Any]，来源resolved数据配置副本；保留运行时派生字段并迁移历史裁剪策略。
    resolved_data = {str(key): value for key, value in resolved_data_value.items()}
    if "local_crop_strategy" not in resolved_data:
        resolved_data["local_crop_strategy"] = "fixed_center"
    # dict[str,str]，来源data_root、stage1_root、output_root的规范化POSIX绝对路径字符串。
    source_path_values: dict[str, str] = {}
    for name in ("data_root", "stage1_root", "output_root"):
        value = resolved_data.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"来源 resolved.json.data.{name} 必须是非空绝对路径字符串。")
        # PurePosixPath，将历史反斜杠统一后得到的来源路径词法表示。
        normalized_path = PurePosixPath(value.replace("\\", "/"))
        if not normalized_path.is_absolute():
            raise ValueError(f"来源 resolved.json.data.{name} 必须是 POSIX 绝对路径。")
        source_path_values[name] = normalized_path.as_posix()
    # object，来源双向覆盖率阈值未验证值。
    coverage_threshold_value = resolved_data.get("coverage_threshold")
    # object，来源occurrence选择种子未验证值。
    selection_seed_value = resolved_data.get("selection_seed")
    # object，来源PP token上限未验证值。
    pp_top_k_value = resolved_data.get("pp_top_k")
    # object，来源局部裁剪策略未验证值。
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
    # ScientificDataRecord，经过路径、类型和范围门禁的来源七项科学数据配置。
    source_scientific_data: ScientificDataRecord = {
        "data_root": source_path_values["data_root"],
        "stage1_root": source_path_values["stage1_root"],
        "output_root": source_path_values["output_root"],
        "coverage_threshold": float(coverage_threshold_value),
        "selection_seed": selection_seed_value,
        "pp_top_k": pp_top_k_value,
        "local_crop_strategy": cast(LocalCropStrategy, local_crop_strategy_value),
    }
    # ScientificDataRecord，目标DataConfig规范化为POSIX字符串后的七项科学数据配置。
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
    # dict[str,object]，用于差异推导的来源科学数据普通映射。
    source_scientific_plain: dict[str, object] = dict(source_scientific_data)
    # dict[str,object]，用于差异推导的目标科学数据普通映射。
    target_scientific_plain: dict[str, object] = dict(target_scientific_data)
    # dict[str,dict[str,object]]，来源与目标七项科学字段中不相等的字段及双侧值。
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

    # Path，与来源BEST.pt同阶段目录的阶段summary.json。
    stage_summary_path = resolved_checkpoint_path.parent / "summary.json"
    # dict[str,Any]，来源训练循环结束后、W&B finish(0)前写出的StageSummary本地记录。
    stage_summary = _read_json_mapping(stage_summary_path, "来源阶段 summary.json")
    # object，阶段summary.best_checkpoint未验证路径字符串。
    stage_summary_best_checkpoint = stage_summary.get("best_checkpoint")
    # object，阶段summary.last_checkpoint未验证路径字符串。
    stage_summary_last_checkpoint = stage_summary.get("last_checkpoint")
    if not isinstance(stage_summary_best_checkpoint, str):
        raise ValueError("来源阶段 summary.best_checkpoint 必须是字符串路径。")
    if not isinstance(stage_summary_last_checkpoint, str):
        raise ValueError("来源阶段 summary.last_checkpoint 必须是字符串路径。")
    # Path，阶段summary声明的BEST checkpoint，随后要求绝对且解析为实际来源文件。
    stage_summary_best_path = Path(stage_summary_best_checkpoint)
    # Path，阶段summary声明的LAST checkpoint，随后要求绝对且解析为同目录LAST.pt。
    stage_summary_last_path = Path(stage_summary_last_checkpoint)
    if not stage_summary_best_path.is_absolute() or not stage_summary_last_path.is_absolute():
        raise ValueError("来源阶段 summary 的 BEST/LAST checkpoint 必须是绝对路径。")
    # Path，来源阶段目录中必须存在的LAST.pt预期路径。
    expected_last_checkpoint = resolved_checkpoint_path.parent / "LAST.pt"
    # object，阶段summary.last_epoch未验证值。
    last_epoch = stage_summary.get("last_epoch")
    # object，阶段summary.global_step未验证值。
    stage_global_step = stage_summary.get("global_step")
    # object，阶段summary.tracking_run_id未验证值。
    source_tracking_run_id = stage_summary.get("tracking_run_id")
    # object，阶段summary.tracking_url未验证值。
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
        # 下方既有异常中的“W&B run已完成收口”仅概括本地字段门禁；本处没有远端查询。
        raise ValueError("来源阶段 summary.json 未证明该 BEST checkpoint 与 W&B run 已完成收口。")

    # Path，来源阶段目录的父目录，即来源计划单次运行根。
    run_directory = resolved_checkpoint_path.parent.parent
    # Path，来源运行开始前发布的根run.json。
    run_record_path = run_directory / "run.json"
    # Path，来源计划全部阶段成功后发布的根summary.json。
    run_summary_path = run_directory / "summary.json"
    # dict[str,Any]，来源根运行开始记录。
    run_record = _read_json_mapping(run_record_path, "来源 run.json")
    # dict[str,Any]，各阶段finish(0)调用正常返回后由run_plan发布的来源根运行本地摘要。
    run_summary = _read_json_mapping(run_summary_path, "来源运行 summary.json")
    # object，根summary.run_stamp未验证值。
    source_run_stamp = run_summary.get("run_stamp")
    # object，根summary.slurm_job_id未验证值。
    source_slurm_job_id = run_summary.get("slurm_job_id")
    # object，根summary.plan_name未验证值。
    source_plan_name = run_summary.get("plan_name")
    # object，根summary.plan_path未验证值。
    training_plan_value = run_summary.get("plan_path")
    # object，根summary.finished_at_utc未验证值。
    finished_at_utc = run_summary.get("finished_at_utc")
    # object，根summary.stages未验证值；正式契约要求list且包含当前阶段summary的相等对象。
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
    # object，根summary.tracking未验证节点。
    source_tracking = run_summary.get("tracking")
    if not isinstance(source_tracking, Mapping) or not all(
        isinstance(key, str) for key in source_tracking
    ):
        raise ValueError("来源运行 tracking 必须是字符串键映射。")
    # dict[str,Any]，来源根运行的七字段W&B配置普通字符串键副本。
    source_tracking_config = {
        str(key): value for key, value in source_tracking.items()
    }
    if source_tracking_config.get("mode") != "online":
        raise ValueError("来源运行必须使用 W&B online tracking。")
    if source_tracking_config != resolved_tracking_config:
        raise ValueError("来源 resolved.json.tracking 与运行记录不一致。")
    # Path，由根summary.plan_path字符串构造的来源YAML候选路径。
    training_plan_candidate = Path(training_plan_value)
    if not training_plan_candidate.is_absolute():
        raise ValueError("来源运行 summary.plan_path 必须是绝对路径。")
    # Path，必须存在的来源训练YAML解析后绝对路径。
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

    # object，目标完整Matcher的candidate_encoder属性；随后要求nn.Module。
    target = getattr(model, "candidate_encoder", None)
    if not isinstance(target, nn.Module):
        raise TypeError("完整 Matcher 必须提供 candidate_encoder 子模块。")
    # dict[str,Tensor]，来源candidate_pretrain直属候选编码器状态；load_checkpoint已核验键与值类型。
    source_state = cast(dict[str, torch.Tensor], checkpoint_values["model_state"])
    if any(key.startswith("candidate_encoder.") for key in source_state):
        raise ValueError("candidate_pretrain checkpoint.model_state 必须直接保存候选编码器参数。")
    # Mapping[str,Tensor]，目标候选编码器当前参数与缓冲区状态快照；用于键、shape和dtype预检查。
    target_state = target.state_dict()
    if source_state.keys() != target_state.keys():
        raise ValueError("来源候选编码器参数名与目标结构不一致。")
    # dict[str,dict[str,object]]，同名来源与目标张量中shape或dtype不相等的逐参数记录。
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
    # _IncompatibleKeys，所有显式门禁通过后执行的严格加载结果；此行才修改目标候选编码器状态。
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
