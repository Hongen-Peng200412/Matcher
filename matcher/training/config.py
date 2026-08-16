"""读取、验证并展开正式 Matcher 实验 YAML。

load_experiment_plan 是唯一完整解析入口：它要求计划、数据、W&B、阶段、模型、训练和损失
映射的键集合精确匹配 schema，再建立不可变 dataclass。validate_run_stamp 只验证外部分配的
运行目录名，validate_matcher_output_path 只规范化并限制服务器业务产物路径。解析器不提供
缺失字段默认值，并拒绝批量关系不一致、相对初始化 checkpoint 以及越过
/storage/penghongen/tmp/Matcher 的计划与缓存输出路径。
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
from ..data.schema import (
    IDENTITY_SCOPES,
    LANGUAGE_FAMILIES,
    LOCAL_CROP_STRATEGIES,
    IdentityScope,
    LocalCropStrategy,
    SourceRoots,
)
from ..model import MatcherModelConfig


# Literal类型，候选六分类预训练与完整闭集Matcher两种阶段任务。
TaskName = Literal["candidate_pretrain", "full_matcher"]
# Literal类型，完整Matcher可读取的两个配体语言表示家族。
LanguageFamily = Literal["molformer", "smi_ted_289m"]
# Literal类型，CUDA autocast与GradScaler支持的三种训练精度模式。
Precision = Literal["bf16", "fp16", "fp32"]
# Literal类型，正式指标追踪唯一提供方。
TrackingProvider = Literal["wandb"]
# Literal类型，正式指标追踪唯一运行模式。
TrackingMode = Literal["online"]
# int，np.random.seed可接受的最大无符号32位种子。
_MAXIMUM_NUMPY_RANDOM_SEED = 2**32 - 1


@dataclass(frozen=True)
class DataConfig:
    """保存全部阶段共享的数据来源、清单参数和 DataLoader 资源参数。

    - roots: SourceRoots；三个路径只组合成结构，本 dataclass 不检查文件存在性。
    - roots.data_root: AdaLigand 解析、配体图、区域与语言来源根路径；训练 Dataset 和其 Store 按需读取。
    - roots.stage1_root: Stage1 Find_0 候选、密度与概率产物根路径；训练 Dataset 和其 Store 按需读取。
    - roots.output_root: Matcher 清单、候选体素和 PP token 缓存的可写根路径；解析时必须位于允许的服务器产物根。
    - coverage_threshold: 浮点数，范围 (0,1]；候选与 occurrence 双向覆盖率共用的清单命中阈值。
    - selection_seed: Python 整数；传给清单构建器固定选择同一联合身份的 occurrence 语言表示，本解析器没有额外限制其范围。
    - pp_top_k: 正整数；每个候选保留的 PP token 上限 K。
    - local_crop_strategy: fixed_center 或 v_centroid；决定 80³ 来源到 48³ 模型输入的裁剪起点。
    - train_num_workers: 正整数；每次完整 train loader 迭代建立的 DataLoader worker 数。
    - validation_num_workers: 正整数；每次完整 validation loader 迭代建立的 DataLoader worker 数。
    - prefetch_factor: 正整数；每个 worker 预取的物理批次数。
    - cache_gib_per_worker: 正有限浮点数；单个 Dataset 实例五类 ByteBudgetLru 的总 GiB 预算，不预分配内存，也不是进程 RSS 硬上限。
    """

    roots: SourceRoots
    coverage_threshold: float
    selection_seed: int
    pp_top_k: int
    local_crop_strategy: LocalCropStrategy
    train_num_workers: int
    validation_num_workers: int
    prefetch_factor: int
    cache_gib_per_worker: float


@dataclass(frozen=True)
class TrackingConfig:
    """保存每个正式训练阶段共用的 W&B online 追踪配置。

    - provider: 固定字符串 wandb；其它提供方在解析时拒绝。
    - mode: 固定字符串 online；初始化失败不得静默降级为离线模式。
    - entity: 非空字符串；已有登录凭据可写入的 W&B 账户或团队，本地解析不验证远端权限。
    - project: 非空字符串；当前实验曲线写入的 W&B 项目，本地解析不访问远端。
    - group: 非空字符串；同一科学批次的可比阶段使用的 W&B 分组名。
    - log_every_global_steps: 正整数；逐参数更新曲线的写入间隔，WandbTracker 仍强制记录首步。
    - init_timeout_seconds: 正整数；传给 W&B Settings 的在线初始化墙钟超时秒数。
    """

    provider: TrackingProvider
    mode: TrackingMode
    entity: str
    project: str
    group: str
    log_every_global_steps: int
    init_timeout_seconds: int


@dataclass(frozen=True)
class TrainingConfig:
    """保存单个阶段的批量、优化、精度、早停和损失配置。

    - seed: [0,2**32-1] 内的 Python 整数；用于模型与进程随机源、候选洗牌和两个 loader worker 生成器派生种子。
    - physical_batch_size: 正整数；一次模型前向的候选数。
    - gradient_accumulation: 正整数；一次 AdamW 参数更新包含的物理批次数。
    - global_batch_size: 固定整数 32，且必须等于 physical_batch_size * gradient_accumulation。
    - learning_rate: 正有限浮点数；AdamW 初始学习率，随后可由 ReduceLROnPlateau 下调。
    - weight_decay: 非负有限浮点数；AdamW 解耦权重衰减系数。
    - maximum_epochs: 正整数或 None；None 表示不设轮数上限，训练仍可由 patience 停止。
    - patience_bad_epochs: 固定整数 2；连续两次完整 validation 未严格改善后停止。
    - precision: bf16、fp16 或 fp32；选择 CUDA autocast dtype，只有 fp16 启用 GradScaler。
    - gradient_clip_norm: 正有限浮点数；每个 global batch 在 unscale 后使用的全模型 L2 梯度裁剪阈值。
    - class_weights: None，或与六个 CLASS_NAMES 固定顺序对齐的六个非负有限浮点数且总和为正。
    - loss_weights: LossWeights；四个无量纲非负有限系数。
    - loss_weights.classification: 六分类交叉熵系数。
    - loss_weights.ligand_area: 配体区域损失系数。
    - loss_weights.binding_area: 结合区域损失系数。
    - loss_weights.identity: 闭集身份损失系数；candidate_pretrain 必须为 0。
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
    """描述顺序计划中的一个训练阶段及其完整科学配置。

    - name: 以字母或数字开头的安全单级目录名，在同一计划内唯一。
    - task_name: candidate_pretrain 或 full_matcher。
    - language_family: full_matcher 为 molformer 或 smi_ted_289m；candidate_pretrain 必须为 None。
    - identity_variant: full_matcher 为 oracle、detached 或 joint；candidate_pretrain 必须为 None。
    - identity_scope: full_matcher 为 small_molecule 或 all_non_background；candidate_pretrain 必须为 None。
    - model: 已由 MatcherModelConfig.from_dict 核验的模型配置。
    - model.aggregation: mean_max 或 candidate_cross_attention。
    - model.use_manual_features: 精确 bool；控制 12 维候选手工特征及身份配对中的类别概率特征。
    - model.use_sampling_features: 精确 bool；控制 4 维候选裁剪统计。
    - model.use_candidate_token_attention_bias: 精确 bool；控制四模态候选—token 注意力偏置。
    - model.initial_graph_gate: [0,1] 内有限浮点门初值；candidate_pretrain 必须为 0。
    - model.initial_inventory_gate: [0,1] 内有限浮点门初值；candidate_pretrain 必须为 0。
    - model.model_dim: 正整数候选与身份表示宽度。
    - model.attention_heads: 正整数，必须整除 model_dim；不控制 RAUNet 内部自选头数。
    - model.dropout: [0,1) 内有限浮点丢弃概率。
    - model.voxel_feature_channels: 正整数 RAUNet 输出通道数。
    - model.raunet_planes: 九个正整数，顺序为 enc0、enc1、enc2、enc3、bottleneck、dec3、dec2、dec1、dec0。
    - model.recycle_iterations: 正整数 RAUNet 回收轮数。
    - model.gradient_checkpointing: 精确 bool；控制训练且梯度启用时的 RAUNet 激活重计算。
    - model.bottleneck_attention_layers: 正整数 RAUNet 瓶颈注意力层数。
    - model.graph_layers: 正整数 LigandObject 图消息层数。
    - training: 已完成批量、范围与任务关系门禁的 TrainingConfig。
    - training.seed: [0,2**32-1] 内的 Python 整数随机种子。
    - training.physical_batch_size: 正整数物理批量。
    - training.gradient_accumulation: 正整数梯度累积次数。
    - training.global_batch_size: 固定为 32，且等于物理批量乘梯度累积次数。
    - training.learning_rate: 正有限 AdamW 初始学习率。
    - training.weight_decay: 非负有限 AdamW 权重衰减系数。
    - training.maximum_epochs: 正整数轮数上限或 None。
    - training.patience_bad_epochs: 固定为 2。
    - training.precision: bf16、fp16 或 fp32。
    - training.gradient_clip_norm: 正有限全模型 L2 梯度裁剪阈值。
    - training.class_weights: 六项非负有限权重元组且总和为正，或 None。
    - training.loss_weights: 四项非负有限 LossWeights。
    - training.loss_weights.classification: 六分类损失系数。
    - training.loss_weights.ligand_area: 配体区域损失系数。
    - training.loss_weights.binding_area: 结合区域损失系数。
    - training.loss_weights.identity: 身份损失系数；candidate_pretrain 必须为 0。
    - initialize_candidate_from: full_matcher 可用的 POSIX 绝对 BEST.pt 路径字符串或 None；candidate_pretrain 必须为 None。
    """

    name: str
    task_name: TaskName
    language_family: LanguageFamily | None
    identity_variant: IdentityLossVariant | None
    identity_scope: IdentityScope | None
    model: MatcherModelConfig
    training: TrainingConfig
    initialize_candidate_from: str | None


@dataclass(frozen=True)
class ExperimentPlan:
    """保存一个计划中共享配置与非空有序训练阶段。

    - plan_name: 以字母或数字开头的安全单级名称；构成运行结果目录第二层。
    - output_root: 规范化绝对 Path；必须位于 /storage/penghongen/tmp/Matcher 或其子目录，承载计划、运行与阶段结果。
    - data: 全部阶段共享的 DataConfig。
    - data.roots: SourceRoots；三个 Path 的组合。
    - data.roots.data_root: AdaLigand 数据来源根路径。
    - data.roots.stage1_root: Stage1 Find_0 产物根路径。
    - data.roots.output_root: Matcher 清单与缓存可写根路径。
    - data.coverage_threshold: 候选与 occurrence 双向覆盖阈值。
    - data.selection_seed: occurrence 语言表示固定选择种子。
    - data.pp_top_k: 每个候选的 PP token 上限。
    - data.local_crop_strategy: fixed_center 或 v_centroid。
    - data.train_num_workers: train DataLoader worker 数。
    - data.validation_num_workers: validation DataLoader worker 数。
    - data.prefetch_factor: 每 worker 预取物理批次数。
    - data.cache_gib_per_worker: 每 Dataset 实例五类 LRU 的总 GiB 预算。
    - tracking: 每个阶段分别建立 W&B online run 时共用的 TrackingConfig。
    - tracking.provider: 固定为 wandb。
    - tracking.mode: 固定为 online。
    - tracking.entity: W&B 账户或团队名。
    - tracking.project: W&B 项目名。
    - tracking.group: W&B 比较分组名。
    - tracking.log_every_global_steps: 参数更新曲线写入间隔。
    - tracking.init_timeout_seconds: 在线初始化超时秒数。
    - stages: 非空 StageConfig 元组，顺序与 YAML 列表相同且 stage.name 不重复。
    - stages[*].name: 阶段安全单级名称。
    - stages[*].task_name: candidate_pretrain 或 full_matcher。
    - stages[*].language_family: 配体语言家族或 None。
    - stages[*].identity_variant: 身份损失变体或 None。
    - stages[*].identity_scope: 身份监督范围或 None。
    - stages[*].model: 完整 MatcherModelConfig。
    - stages[*].model.aggregation: mean_max 或 candidate_cross_attention。
    - stages[*].model.use_manual_features: 精确 bool。
    - stages[*].model.use_sampling_features: 精确 bool。
    - stages[*].model.use_candidate_token_attention_bias: 精确 bool。
    - stages[*].model.initial_graph_gate: [0,1] 内有限浮点门初值。
    - stages[*].model.initial_inventory_gate: [0,1] 内有限浮点门初值。
    - stages[*].model.model_dim: 正整数表示宽度。
    - stages[*].model.attention_heads: 正整数候选与身份注意力头数。
    - stages[*].model.dropout: [0,1) 内有限丢弃概率。
    - stages[*].model.voxel_feature_channels: 正整数 RAUNet 输出通道数。
    - stages[*].model.raunet_planes: 九项正整数通道宽度元组。
    - stages[*].model.recycle_iterations: 正整数回收轮数。
    - stages[*].model.gradient_checkpointing: 精确 bool。
    - stages[*].model.bottleneck_attention_layers: 正整数瓶颈注意力层数。
    - stages[*].model.graph_layers: 正整数图消息层数。
    - stages[*].training: 完整 TrainingConfig。
    - stages[*].training.seed: [0,2**32-1] 内的整数随机种子。
    - stages[*].training.physical_batch_size: 正整数物理批量。
    - stages[*].training.gradient_accumulation: 正整数梯度累积次数。
    - stages[*].training.global_batch_size: 固定为 32 的全局批量。
    - stages[*].training.learning_rate: 正有限初始学习率。
    - stages[*].training.weight_decay: 非负有限权重衰减。
    - stages[*].training.maximum_epochs: 正整数轮数上限或 None。
    - stages[*].training.patience_bad_epochs: 固定为 2。
    - stages[*].training.precision: bf16、fp16 或 fp32。
    - stages[*].training.gradient_clip_norm: 正有限梯度裁剪阈值。
    - stages[*].training.class_weights: 六项权重元组或 None。
    - stages[*].training.loss_weights: 四项 LossWeights。
    - stages[*].training.loss_weights.classification: 六分类损失系数。
    - stages[*].training.loss_weights.ligand_area: 配体区域损失系数。
    - stages[*].training.loss_weights.binding_area: 结合区域损失系数。
    - stages[*].training.loss_weights.identity: 身份损失系数。
    - stages[*].initialize_candidate_from: 外部候选预训练 BEST.pt 绝对路径字符串或 None。
    """

    plan_name: str
    output_root: Path
    data: DataConfig
    tracking: TrackingConfig
    stages: tuple[StageConfig, ...]


def _parse_stage(values: dict[str, Any]) -> StageConfig:
    """把一个 YAML 阶段映射解析为不可变 StageConfig。

    - values: 从 YAML stages 列表取得的一个阶段映射；本函数精确核验顶层键与全部嵌套字段。
    - values.name: 阶段安全单级名称。
    - values.task_name: candidate_pretrain 或 full_matcher。
    - values.language_family: full_matcher 的 molformer 或 smi_ted_289m；candidate_pretrain 为 null。
    - values.identity_variant: full_matcher 的 oracle、detached 或 joint；candidate_pretrain 为 null。
    - values.identity_scope: full_matcher 的 small_molecule 或 all_non_background；candidate_pretrain 为 null。
    - values.model: 精确包含十五个模型字段的字符串键映射。
    - values.model.aggregation: mean_max 或 candidate_cross_attention。
    - values.model.use_manual_features: 精确 bool。
    - values.model.use_sampling_features: 精确 bool。
    - values.model.use_candidate_token_attention_bias: 精确 bool。
    - values.model.initial_graph_gate: [0,1] 内有限数值。
    - values.model.initial_inventory_gate: [0,1] 内有限数值。
    - values.model.model_dim: 正的精确整数。
    - values.model.attention_heads: 正的精确整数且整除 model_dim。
    - values.model.dropout: [0,1) 内有限数值。
    - values.model.voxel_feature_channels: 正的精确整数。
    - values.model.raunet_planes: 九项正的精确整数列表或元组。
    - values.model.recycle_iterations: 正的精确整数。
    - values.model.gradient_checkpointing: 精确 bool。
    - values.model.bottleneck_attention_layers: 正的精确整数。
    - values.model.graph_layers: 正的精确整数。
    - values.training: 精确包含下列十二个字段的字符串键映射，其中 loss_weights 自身是嵌套映射。
    - values.training.seed: [0,2**32-1] 内的精确整数。
    - values.training.physical_batch_size: 正的精确整数。
    - values.training.gradient_accumulation: 正的精确整数。
    - values.training.global_batch_size: 精确整数 32，等于物理批量乘梯度累积次数。
    - values.training.learning_rate: 正有限数值。
    - values.training.weight_decay: 非负有限数值。
    - values.training.maximum_epochs: 正的精确整数或 null。
    - values.training.patience_bad_epochs: 精确整数 2。
    - values.training.precision: bf16、fp16 或 fp32。
    - values.training.gradient_clip_norm: 正有限数值。
    - values.training.class_weights: null，或六项非负有限数值列表或元组且总和为正。
    - values.training.loss_weights: 精确含 classification、ligand_area、binding_area、identity 的映射。
    - values.training.loss_weights.classification: 非负有限数值。
    - values.training.loss_weights.ligand_area: 非负有限数值。
    - values.training.loss_weights.binding_area: 非负有限数值。
    - values.training.loss_weights.identity: 非负有限数值；candidate_pretrain 必须为 0。
    - values.initialize_candidate_from: full_matcher 可用的绝对 BEST.pt 路径字符串或 null；candidate_pretrain 必须为 null。

    - 返回值.name: 解析后的安全阶段名。
    - 返回值.task_name: 经过枚举门禁的任务名。
    - 返回值.language_family: 经过任务关系门禁的语言家族或 None。
    - 返回值.identity_variant: 经过任务关系门禁的身份损失变体或 None。
    - 返回值.identity_scope: 经过任务关系门禁的身份范围或 None。
    - 返回值.model: MatcherModelConfig.from_dict 返回的完整模型配置。
    - 返回值.model.aggregation: mean_max 或 candidate_cross_attention。
    - 返回值.model.use_manual_features: 精确 bool。
    - 返回值.model.use_sampling_features: 精确 bool。
    - 返回值.model.use_candidate_token_attention_bias: 精确 bool。
    - 返回值.model.initial_graph_gate: [0,1] 内有限浮点门初值。
    - 返回值.model.initial_inventory_gate: [0,1] 内有限浮点门初值。
    - 返回值.model.model_dim: 正整数表示宽度。
    - 返回值.model.attention_heads: 正整数候选与身份注意力头数。
    - 返回值.model.dropout: [0,1) 内有限丢弃概率。
    - 返回值.model.voxel_feature_channels: 正整数 RAUNet 输出通道数。
    - 返回值.model.raunet_planes: 九项正整数通道元组。
    - 返回值.model.recycle_iterations: 正整数回收轮数。
    - 返回值.model.gradient_checkpointing: 精确 bool。
    - 返回值.model.bottleneck_attention_layers: 正整数瓶颈注意力层数。
    - 返回值.model.graph_layers: 正整数图消息层数。
    - 返回值.training: 完成类型、范围、批量与任务关系门禁的 TrainingConfig。
    - 返回值.training.seed: [0,2**32-1] 内的整数随机种子。
    - 返回值.training.physical_batch_size: 正整数物理批量。
    - 返回值.training.gradient_accumulation: 正整数梯度累积次数。
    - 返回值.training.global_batch_size: 固定为 32 的全局批量。
    - 返回值.training.learning_rate: 正有限初始学习率。
    - 返回值.training.weight_decay: 非负有限权重衰减。
    - 返回值.training.maximum_epochs: 正整数或 None。
    - 返回值.training.patience_bad_epochs: 固定为 2。
    - 返回值.training.precision: bf16、fp16 或 fp32。
    - 返回值.training.gradient_clip_norm: 正有限梯度裁剪阈值。
    - 返回值.training.class_weights: 六项非负有限权重元组或 None。
    - 返回值.training.loss_weights: 四项非负有限 LossWeights。
    - 返回值.training.loss_weights.classification: 六分类损失系数。
    - 返回值.training.loss_weights.ligand_area: 配体区域损失系数。
    - 返回值.training.loss_weights.binding_area: 结合区域损失系数。
    - 返回值.training.loss_weights.identity: 身份损失系数。
    - 返回值.initialize_candidate_from: 未解析文件系统的来源路径字符串或 None；绝对路径与文件名关系由计划级门禁继续检查。

    本函数要求 stage、stage.model、stage.training 和 loss_weights 的键集合精确匹配；不接受
    未知键，也不为缺失键提供默认值。
    """

    _require_keys(
        values,
        {
            "name",
            "task_name",
            "language_family",
            "identity_variant",
            "identity_scope",
            "model",
            "training",
            "initialize_candidate_from",
        },
        "stage",
    )
    # object，YAML stage.task_name 的未验证标量。
    raw_task_name = values["task_name"]
    # object，YAML stage.language_family 的未验证标量或null。
    raw_language_family = values["language_family"]
    # object，YAML stage.identity_variant 的未验证标量或null。
    raw_identity_variant = values["identity_variant"]
    # object，YAML stage.identity_scope 的未验证标量或null。
    raw_identity_scope = values["identity_scope"]
    if raw_task_name not in ("candidate_pretrain", "full_matcher"):
        raise ValueError(f"不支持 task_name={raw_task_name!r}。")
    task_name = cast(TaskName, raw_task_name)
    if task_name == "candidate_pretrain":
        if (
            raw_language_family is not None
            or raw_identity_variant is not None
            or raw_identity_scope is not None
        ):
            raise ValueError("candidate_pretrain 不得配置语言模型、身份损失或身份范围。")
        language_family = None
        identity_variant = None
        identity_scope = None
    else:
        if raw_language_family not in LANGUAGE_FAMILIES:
            raise ValueError(f"full_matcher 的 language_family 必须属于 {LANGUAGE_FAMILIES}。")
        if raw_identity_variant not in ("oracle", "detached", "joint"):
            raise ValueError("full_matcher 必须选择 oracle/detached/joint。")
        if raw_identity_scope not in IDENTITY_SCOPES:
            raise ValueError(f"full_matcher 的 identity_scope 必须属于 {IDENTITY_SCOPES}。")
        language_family = cast(LanguageFamily, raw_language_family)
        identity_variant = cast(IdentityLossVariant, raw_identity_variant)
        identity_scope = cast(IdentityScope, raw_identity_scope)
    # dict[str,Any]，具有字符串键的未解析模型配置映射。
    model_values = _mapping(values["model"], "stage.model")
    _require_keys(
        model_values,
        {
            "aggregation",
            "use_manual_features",
            "use_sampling_features",
            "use_candidate_token_attention_bias",
            "initial_graph_gate",
            "initial_inventory_gate",
            "model_dim",
            "attention_heads",
            "dropout",
            "voxel_feature_channels",
            "raunet_planes",
            "recycle_iterations",
            "gradient_checkpointing",
            "bottleneck_attention_layers",
            "graph_layers",
        },
        "stage.model",
    )
    # MatcherModelConfig，已完成类型、范围和模型跨字段关系门禁的不可变配置。
    model = MatcherModelConfig.from_dict(model_values)
    if task_name == "candidate_pretrain" and (
        model.initial_graph_gate != 0.0 or model.initial_inventory_gate != 0.0
    ):
        raise ValueError("candidate_pretrain 不构造身份打分分支，两个初始身份门必须为 0。")
    # dict[str,Any]，具有字符串键的未解析训练配置映射。
    training_values = _mapping(values["training"], "stage.training")
    # set[str]，stage.training 必须精确提供的十二个字段，其中loss_weights为嵌套映射。
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
    # dict[str,Any]，四项未解析损失系数映射。
    loss_values = _mapping(training_values["loss_weights"], "stage.training.loss_weights")
    _require_keys(
        loss_values,
        {"classification", "ligand_area", "binding_area", "identity"},
        "stage.training.loss_weights",
    )
    # object，YAML class_weights 的null或未解析序列。
    class_weights_value = training_values["class_weights"]
    # tuple[float,...] | None，逐项转换为有限float后的六分类权重。
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
    # object，YAML maximum_epochs 的null或未解析整数。
    maximum_epochs_value = training_values["maximum_epochs"]
    # TrainingConfig，尚待下方跨字段、范围和任务关系检查的不可变训练配置。
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
    if not 0 <= training.seed <= _MAXIMUM_NUMPY_RANDOM_SEED:
        raise ValueError(f"seed 必须位于 [0,{_MAXIMUM_NUMPY_RANDOM_SEED}]。")
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
    # object，YAML初始化来源的字符串或null；计划级门禁稍后验证绝对BEST.pt形式。
    initialize_candidate_from = values["initialize_candidate_from"]
    if initialize_candidate_from is not None and not isinstance(initialize_candidate_from, str):
        raise ValueError("initialize_candidate_from 必须是字符串或 null。")
    return StageConfig(
        name=_path_component(values["name"], "stage.name"),
        task_name=task_name,
        language_family=language_family,
        identity_variant=identity_variant,
        identity_scope=identity_scope,
        model=model,
        training=training,
        initialize_candidate_from=initialize_candidate_from,
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """要求 YAML 节点是普通 dict 且每个键都是字符串。

    - value: yaml.safe_load 产生的任意节点；只有普通 dict 且全部键为 str 时接受。
    - name: 非空性未核验的错误上下文字符串，只用于异常消息。

    - 返回值: 与 value 同一个 dict 对象，值保持未解析的 Any；本函数不复制或递归验证嵌套节点。
    """

    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} 必须是字符串键 YAML 映射。")
    return value


def _sequence(value: Any, name: str) -> list[Any] | tuple[Any, ...]:
    """要求配置节点是 list 或 tuple，并原样返回容器。

    - value: 任意配置值；正式 yaml.safe_load 通常产生 list，tuple 供 Python 直接调用路径使用。
    - name: 错误上下文字符串，只用于异常消息。

    - 返回值: 与 value 同一个 list 或 tuple；元素类型、数量和取值由调用方继续验证。
    """

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} 必须是 YAML 列表。")
    return value


def _string(value: Any, name: str) -> str:
    """要求配置值已经是非空 str，不执行静默字符串化。

    - value: 任意配置标量。
    - name: 错误上下文字符串，只用于异常消息。

    - 返回值: 与 value 相同的非空字符串；只含空白的字符串仍会被接受。
    """

    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} 必须是非空字符串。")
    return value


def _path(value: Any, name: str) -> Path:
    """把非空字符串配置包装为 Path，不解析存在性或绝对性。

    - value: 必须由 _string 接受的非空字符串。
    - name: 错误上下文字符串，只用于异常消息。

    - 返回值: 按当前操作系统 Path 语义构造的路径对象；不调用 resolve 或 expanduser。
    """

    return Path(_string(value, name))


def _integer(value: Any, name: str) -> int:
    """要求配置值满足 type(value) is int。

    - value: 任意配置标量；bool、NumPy 整数和可截断浮点数均拒绝。
    - name: 错误上下文字符串，只用于异常消息。

    - 返回值: 与 value 相同的 Python int；范围由调用方继续验证。
    """

    if type(value) is not int:
        raise ValueError(f"{name} 必须是整数。")
    return value


def _finite_float(value: Any, name: str) -> float:
    """把通过 int/float isinstance 门禁且不是 bool 的值转为有限 float。

    - value: int 或 float 实例及能通过 isinstance(value,(int,float)) 的子类；bool、字符串、NaN 与正负无穷拒绝；YAML 正式入口通常给出内置数值，直接 Python 调用可传入符合该 isinstance 关系的子类。
    - name: 错误上下文字符串，只用于异常消息。

    - 返回值: 有限 Python float；符号与业务范围由调用方继续验证。
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数值。")
    # float，由通过int/float isinstance门禁且非bool的值显式转换；随后拒绝NaN与正负无穷。
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限数值。")
    return result


def _path_component(value: Any, name: str) -> str:
    """验证计划、阶段或运行标识是安全单级路径组件。

    - value: 必须由 _string 接受的非空字符串。
    - name: 错误上下文字符串，只用于异常消息。

    - 返回值: 以 ASCII 字母或数字开头、后续只含 ASCII 字母数字、下划线、点或连字符的原字符串。
    """

    # str，尚待正则门禁的非空配置字符串。
    result = _string(value, name)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", result) is None:
        raise ValueError(f"{name} 必须是以字母或数字开头的安全路径组件。")
    return result


def _require_keys(values: dict[str, Any], expected: set[str], name: str) -> None:
    """要求映射键集合与期望集合精确相等。

    - values: 字符串键配置映射；本函数只检查顶层键，不读取或复制值。
    - expected: 必需且唯一允许的字段名集合。
    - name: 错误上下文字符串，只用于异常消息。

    缺失键与未知键会在同一个 ValueError 中分别按字典序列出。
    """

    # set[str]，调用方映射的实际顶层键集合。
    actual = set(values)
    if actual != expected:
        raise ValueError(
            f"{name} 字段不一致：missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}。"
        )


def _validate_stage_initialization(stages: tuple[StageConfig, ...]) -> None:
    """验证有序阶段中的候选预训练初始化字段只采用绝对 BEST.pt 形式。

    - stages: 已完成单阶段解析的 StageConfig 元组。
    - stages[*].name: 本函数不读取的阶段名。
    - stages[*].task_name: initialize_candidate_from 非空时必须为 full_matcher。
    - stages[*].language_family: 本函数不读取。
    - stages[*].identity_variant: 本函数不读取。
    - stages[*].identity_scope: 本函数不读取。
    - stages[*].model: 本函数不读取。
    - stages[*].training: 本函数不读取。
    - stages[*].initialize_candidate_from: None，或文件名恰为 BEST.pt 的 POSIX 绝对路径字符串。

    本函数只检查路径文字形式，不检查文件是否存在、是否位于允许输出根或是否真是候选预训练。
    """

    for stage in stages:
        # str | None，当前阶段声明的外部候选编码器初始化checkpoint。
        specification = stage.initialize_candidate_from
        if specification is not None:
            if stage.task_name != "full_matcher":
                raise ValueError("initialize_candidate_from 只允许用于 full_matcher 阶段。")
            # PurePosixPath，不访问文件系统的服务器路径语法视图。
            external_path = PurePosixPath(specification)
            if not external_path.is_absolute() or external_path.name != "BEST.pt":
                raise ValueError("initialize_candidate_from 必须是以 BEST.pt 结尾的绝对路径。")


# ================================================================================================


def validate_matcher_output_path(path: Path, name: str) -> Path:
    """按 POSIX 词法规范化路径并限制其位于 Matcher 服务器产物根。

    - path: 待核验 Path；仅使用 as_posix 文字，不访问文件系统、不解析符号链接。
    - name: 错误上下文字符串，只用于异常消息。

    - 返回值: 词法消除点与双点组件后的绝对 Path，位于 /storage/penghongen/tmp/Matcher 或其子目录。

    本函数不创建目录、不检查存在性，也不防止已存在路径中的符号链接在文件系统解析后指向根外。
    """

    # PurePosixPath，按posixpath.normpath完成词法规范化的服务器路径。
    posix_path = PurePosixPath(posixpath.normpath(path.as_posix()))
    if not posix_path.is_absolute():
        raise ValueError(f"{name}={path} 必须是绝对路径。")
    # PurePosixPath，计划结果与Matcher缓存唯一允许的服务器业务产物根。
    required_root = PurePosixPath("/storage/penghongen/tmp/Matcher")
    if not posix_path.is_relative_to(required_root):
        raise ValueError(f"{name}={path} 不在允许的服务器产物根 {required_root} 内。")
    return Path(posix_path.as_posix())


def validate_run_stamp(value: str) -> str:
    """验证 allocation 提供的运行标识是安全单级目录名。

    - value: 预期为非空字符串；通常来自 TASK_RUN_STAMP 或 UTC 时间字符串。

    - 返回值: 经 _path_component 门禁后与输入相同的字符串。
    """

    return _path_component(value, "run_stamp")


def load_experiment_plan(path: Path) -> ExperimentPlan:
    """读取 UTF-8 YAML，完成全计划门禁并返回不可变 ExperimentPlan。

    - path: 计划 YAML 路径；read_text 负责存在性和 UTF-8 解码，本函数不要求特定扩展名。

    - 返回值.plan_name: 安全单级计划名。
    - 返回值.output_root: 位于 /storage/penghongen/tmp/Matcher 下的规范化绝对 Path。
    - 返回值.data: 全部阶段共享的 DataConfig。
    - 返回值.data.roots: 三个数据路径组成的 SourceRoots。
    - 返回值.data.roots.data_root: YAML data.data_root 转换的 Path，不检查绝对性或存在性。
    - 返回值.data.roots.stage1_root: YAML data.stage1_root 转换的 Path，不检查绝对性或存在性。
    - 返回值.data.roots.output_root: 经业务产物根门禁的规范化绝对 Path。
    - 返回值.data.coverage_threshold: (0,1] 内有限浮点数。
    - 返回值.data.selection_seed: 精确 Python 整数，本函数未限制范围。
    - 返回值.data.pp_top_k: 正整数。
    - 返回值.data.local_crop_strategy: fixed_center 或 v_centroid。
    - 返回值.data.train_num_workers: 正整数。
    - 返回值.data.validation_num_workers: 正整数。
    - 返回值.data.prefetch_factor: 正整数。
    - 返回值.data.cache_gib_per_worker: 正有限浮点数。
    - 返回值.tracking: 全部阶段共享的 TrackingConfig。
    - 返回值.tracking.provider: 固定字符串 wandb。
    - 返回值.tracking.mode: 固定字符串 online。
    - 返回值.tracking.entity: 非空字符串。
    - 返回值.tracking.project: 非空字符串。
    - 返回值.tracking.group: 非空字符串。
    - 返回值.tracking.log_every_global_steps: 正整数。
    - 返回值.tracking.init_timeout_seconds: 正整数。
    - 返回值.stages: 非空 StageConfig 元组，顺序与 YAML stages 列表一致且名称唯一。
    - 返回值.stages[*].name: 安全且计划内唯一的阶段名。
    - 返回值.stages[*].task_name: candidate_pretrain 或 full_matcher。
    - 返回值.stages[*].language_family: full_matcher 语言家族或 None。
    - 返回值.stages[*].identity_variant: full_matcher 身份损失变体或 None。
    - 返回值.stages[*].identity_scope: full_matcher 身份范围或 None。
    - 返回值.stages[*].model: 完整 MatcherModelConfig。
    - 返回值.stages[*].model.aggregation: mean_max 或 candidate_cross_attention。
    - 返回值.stages[*].model.use_manual_features: 精确 bool。
    - 返回值.stages[*].model.use_sampling_features: 精确 bool。
    - 返回值.stages[*].model.use_candidate_token_attention_bias: 精确 bool。
    - 返回值.stages[*].model.initial_graph_gate: [0,1] 内有限浮点门初值。
    - 返回值.stages[*].model.initial_inventory_gate: [0,1] 内有限浮点门初值。
    - 返回值.stages[*].model.model_dim: 正整数表示宽度。
    - 返回值.stages[*].model.attention_heads: 正整数候选与身份注意力头数。
    - 返回值.stages[*].model.dropout: [0,1) 内有限丢弃概率。
    - 返回值.stages[*].model.voxel_feature_channels: 正整数 RAUNet 输出通道数。
    - 返回值.stages[*].model.raunet_planes: 九项正整数通道元组。
    - 返回值.stages[*].model.recycle_iterations: 正整数回收轮数。
    - 返回值.stages[*].model.gradient_checkpointing: 精确 bool。
    - 返回值.stages[*].model.bottleneck_attention_layers: 正整数瓶颈注意力层数。
    - 返回值.stages[*].model.graph_layers: 正整数图消息层数。
    - 返回值.stages[*].training: 完整 TrainingConfig。
    - 返回值.stages[*].training.seed: [0,2**32-1] 内的整数随机种子。
    - 返回值.stages[*].training.physical_batch_size: 正整数物理批量。
    - 返回值.stages[*].training.gradient_accumulation: 正整数梯度累积次数。
    - 返回值.stages[*].training.global_batch_size: 固定为 32 的全局批量。
    - 返回值.stages[*].training.learning_rate: 正有限初始学习率。
    - 返回值.stages[*].training.weight_decay: 非负有限权重衰减。
    - 返回值.stages[*].training.maximum_epochs: 正整数或 None。
    - 返回值.stages[*].training.patience_bad_epochs: 固定为 2。
    - 返回值.stages[*].training.precision: bf16、fp16 或 fp32。
    - 返回值.stages[*].training.gradient_clip_norm: 正有限梯度裁剪阈值。
    - 返回值.stages[*].training.class_weights: 六项权重元组或 None。
    - 返回值.stages[*].training.loss_weights: 四项 LossWeights。
    - 返回值.stages[*].training.loss_weights.classification: 六分类损失系数。
    - 返回值.stages[*].training.loss_weights.ligand_area: 配体区域损失系数。
    - 返回值.stages[*].training.loss_weights.binding_area: 结合区域损失系数。
    - 返回值.stages[*].training.loss_weights.identity: 身份损失系数。
    - 返回值.stages[*].initialize_candidate_from: POSIX 绝对 BEST.pt 路径字符串或 None。

    顶层 plan、data、tracking、每个 stage、model、training 和 loss_weights 都采用精确键集合。
    yaml.safe_load 解析错误、文件错误与所有字段错误直接传播；本函数不创建产物或访问 W&B。
    """

    # object，yaml.safe_load产生的计划根节点；随后要求为普通dict。
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"{path} 顶层必须是 YAML 映射。")
    _require_keys(values, {"plan_name", "output_root", "data", "tracking", "stages"}, "plan")
    # dict[str,Any]，YAML data映射的未解析字段。
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
            "local_crop_strategy",
            "train_num_workers",
            "validation_num_workers",
            "prefetch_factor",
            "cache_gib_per_worker",
        },
        "data",
    )
    # Path，词法规范化且限制在Matcher业务产物根下的数据缓存输出根。
    canonical_data_output = validate_matcher_output_path(
        _path(data_values["output_root"], "data.output_root"),
        "data.output_root",
    )
    # DataConfig，尚待范围和资源正值门禁的共享数据配置。
    data = DataConfig(
        roots=SourceRoots(
            _path(data_values["data_root"], "data.data_root"),
            _path(data_values["stage1_root"], "data.stage1_root"),
            canonical_data_output,
        ),
        coverage_threshold=_finite_float(data_values["coverage_threshold"], "coverage_threshold"),
        selection_seed=_integer(data_values["selection_seed"], "selection_seed"),
        pp_top_k=_integer(data_values["pp_top_k"], "pp_top_k"),
        local_crop_strategy=cast(
            LocalCropStrategy,
            _string(data_values["local_crop_strategy"], "local_crop_strategy"),
        ),
        train_num_workers=_integer(
            data_values["train_num_workers"],
            "train_num_workers",
        ),
        validation_num_workers=_integer(
            data_values["validation_num_workers"],
            "validation_num_workers",
        ),
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
    if data.local_crop_strategy not in LOCAL_CROP_STRATEGIES:
        raise ValueError(
            f"local_crop_strategy 必须属于 {LOCAL_CROP_STRATEGIES}。"
        )
    if (
        data.train_num_workers <= 0
        or data.validation_num_workers <= 0
        or data.prefetch_factor <= 0
        or data.cache_gib_per_worker <= 0
    ):
        raise ValueError(
            "train_num_workers/validation_num_workers/prefetch_factor/"
            "cache_gib_per_worker 必须为正数。"
        )

    # dict[str,Any]，YAML tracking映射的未解析字段。
    tracking_values = _mapping(values["tracking"], "tracking")
    _require_keys(
        tracking_values,
        {
            "provider",
            "mode",
            "entity",
            "project",
            "group",
            "log_every_global_steps",
            "init_timeout_seconds",
        },
        "tracking",
    )
    # str，尚待固定wandb枚举门禁的provider。
    provider = _string(tracking_values["provider"], "tracking.provider")
    # str，尚待固定online枚举门禁的mode。
    mode = _string(tracking_values["mode"], "tracking.mode")
    if provider != "wandb":
        raise ValueError("tracking.provider 当前必须为 wandb。")
    if mode != "online":
        raise ValueError("tracking.mode 当前必须为 online，不能静默降级为离线模式。")
    # TrackingConfig，已完成字段类型门禁，随后检查两个正整数资源字段。
    tracking = TrackingConfig(
        provider=cast(TrackingProvider, provider),
        mode=cast(TrackingMode, mode),
        entity=_string(tracking_values["entity"], "tracking.entity"),
        project=_string(tracking_values["project"], "tracking.project"),
        group=_string(tracking_values["group"], "tracking.group"),
        log_every_global_steps=_integer(
            tracking_values["log_every_global_steps"],
            "tracking.log_every_global_steps",
        ),
        init_timeout_seconds=_integer(
            tracking_values["init_timeout_seconds"],
            "tracking.init_timeout_seconds",
        ),
    )
    if tracking.log_every_global_steps <= 0 or tracking.init_timeout_seconds <= 0:
        raise ValueError("tracking 的步间隔和初始化超时必须为正整数。")

    # object，YAML stages根节点；正式契约要求为非空list。
    raw_stages = values["stages"]
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("stages 必须为非空 YAML 列表。")
    # tuple[StageConfig,...]，保持YAML列表顺序的已解析阶段序列。
    stages = tuple(_parse_stage(_mapping(stage, "stage")) for stage in raw_stages)
    # list[str]，用于计划内阶段名唯一性检查的有序名称列表。
    names = [stage.name for stage in stages]
    if len(names) != len(set(names)):
        raise ValueError("同一计划中的 stage.name 不能重复。")
    _validate_stage_initialization(stages)
    return ExperimentPlan(
        plan_name=_path_component(values["plan_name"], "plan_name"),
        output_root=validate_matcher_output_path(
            _path(values["output_root"], "output_root"),
            "output_root",
        ),
        data=data,
        tracking=tracking,
        stages=stages,
    )
