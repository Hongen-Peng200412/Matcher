"""把正式训练的参数更新与 validation 指标发送到 W&B online。

start_wandb_tracking 为每个训练阶段建立独立在线 run，并在阶段目录发布 tracking.json。
WandbTracker 负责按 global_step 记录更新曲线、按 epoch 记录训练与完整 validation 汇总、
更新最终摘要并结束后台同步。明确由零支持分母解释的 NaN 会转换成 W&B 空值；Inf、其它
NaN、初始化失败和在线写入失败不会降级为离线模式。
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias, TypedDict, cast

from ..data.schema import CLASS_NAMES, FOREGROUND_CLASS_NAMES
from ..utils.json_artifacts import write_json
from .config import StageConfig, TrackingConfig


# TypeAlias，W&B扁平指标进入有限性与零支持NaN门禁前允许的Python标量及空值集合。
MetricValue: TypeAlias = float | int | bool | None


class TrackingMetadata(TypedDict):
    """声明阶段目录 tracking.json 的十个字段。

    - provider: 字符串 wandb；在线追踪实现。
    - mode: 字符串 online；W&B 运行模式。
    - entity: 配置中的 W&B 账户或团队名。
    - project: 配置中的 W&B 项目名。
    - group: 配置中的批次比较分组名。
    - run_id: W&B run.id 返回的非空字符串。
    - run_name: stage.name、两个下划线和 plan_directory.name 拼成的字符串。
    - run_url: W&B run.get_url 返回且以 https:// 开头的字符串。
    - wandb_version: 导入的 W&B SDK __version__ 非空字符串。
    - local_directory: stage_directory / wandb 的路径字符串；本模块记录预期位置，不另行检查目录存在。
    """

    provider: str
    mode: str
    entity: str
    project: str
    group: str
    run_id: str
    run_name: str
    run_url: str
    wandb_version: str
    local_directory: str


class _WandbSummary(Protocol):
    """声明适配层使用的 W&B Summary 最小结构。

    Protocol 没有运行时 isinstance 门禁，只供静态类型检查和测试替身实现接口。
    """

    def update(self, values: Mapping[str, MetricValue]) -> None:
        """合并一组扁平最终摘要值。

        - values: 非空性未要求的字符串指标名到 float、int、bool 或 None 的映射。
        """


class _WandbRun(Protocol):
    """声明适配层实际读取和调用的 W&B Run 最小结构。

    - id: W&B 分配的运行标识；启动函数要求为非空字符串。
    - summary: 提供 update 方法的最终摘要对象。
    """

    id: object
    summary: _WandbSummary

    def get_url(self) -> object:
        """返回未验证的运行地址对象。

        - 返回值: W&B SDK 提供的地址对象；启动函数随后要求它是以 https:// 开头的字符串。
        """

    def define_metric(self, name: str, *, step_metric: str | None = None) -> object:
        """声明指标名或通配模式及其可选横轴。

        - name: W&B 指标名称或通配模式字符串。
        - step_metric: 自定义横轴名称；None 表示只声明该指标自身。

        - 返回值: W&B SDK 的未使用返回对象。
        """

    def log(self, values: Mapping[str, MetricValue]) -> None:
        """写入一组含显式横轴的扁平指标。

        - values: 字符串指标名到 float、int、bool 或 None 的映射。
        """

    def finish(self, *, exit_code: int) -> None:
        """结束后台同步并保存阶段退出语义。

        - exit_code: 非负整数；0 表示正常收口，1 用于本模块已创建 run 后的失败路径。
        """


class _WandbModule(Protocol):
    """声明延迟导入 W&B 模块时使用的最小结构。

    - __version__: SDK 版本对象；启动函数要求为非空字符串。
    """

    __version__: object

    def Settings(self, *, init_timeout: int) -> object:
        """构造 W&B SDK 设置对象。

        - init_timeout: 正整数在线初始化墙钟超时秒数。

        - 返回值: 直接传给 wandb.init 的 SDK 设置对象。
        """

    def init(
        self,
        *,
        project: str,
        entity: str,
        group: str,
        name: str,
        job_type: str,
        tags: list[str],
        config: dict[str, object],
        dir: str,
        mode: str,
        resume: str,
        reinit: bool,
        anonymous: str,
        save_code: bool,
        force: bool,
        settings: object,
    ) -> _WandbRun | None:
        """创建阶段级 W&B run。

        - project: W&B 项目名。
        - entity: W&B 账户或团队名。
        - group: W&B 比较分组名。
        - name: 阶段名与计划运行标识组成的 run 名。
        - job_type: candidate_pretrain 或 full_matcher。
        - tags: 任务、聚合、语言、身份变体与身份范围五个字符串标签。
        - config: resolved.json 同源的完整解析配置普通 dict。
        - dir: W&B 本地同步文件写入的阶段目录字符串。
        - mode: 固定 online。
        - resume: 固定 never，不复用已有 run ID。
        - reinit: 固定 True，允许同一进程顺序建立多个阶段 run。
        - anonymous: 固定 never，要求已认证账户。
        - save_code: 固定 False，不上传源码快照。
        - force: 固定 True；当前 W&B 0.17.3 把该值交给登录流程，用于要求登录成功而不是保证在线 run 必然创建。
        - settings: Settings 返回的 SDK 设置对象。

        - 返回值: 满足 _WandbRun 的对象；SDK 返回 None 时启动函数失败关闭。
        """


# tuple[str,...]，W&B中需要相同零支持NaN规则的配置范围、固定范围与五个前景类别前缀。
_IDENTITY_REPORT_PREFIXES = (
    "identity_configured",
    "identity_small_molecule",
    "identity_all_non_background",
    *(f"identity_{name}" for name in FOREGROUND_CLASS_NAMES),
)
# dict[str,str]，允许NaN转None的指标名到同批整数支持数字段名的精确映射。
_NULLABLE_METRIC_SUPPORT: dict[str, str] = {
    "identity_configured/conditional/seen/top1": (
        "identity_configured/conditional/seen/count"
    ),
    "identity_configured/conditional/seen/topk": (
        "identity_configured/conditional/seen/count"
    ),
    "identity_configured/conditional/unseen/top1": (
        "identity_configured/conditional/unseen/count"
    ),
    "identity_configured/conditional/unseen/topk": (
        "identity_configured/conditional/unseen/count"
    ),
    "area/ligand/micro_dice": "area/ligand/dice_denominator",
    "area/binding/micro_dice": "area/binding/dice_denominator",
    **{
        f"classification_per_class/{name}/precision": (
            f"classification_per_class/{name}/predicted_count"
        )
        for name in CLASS_NAMES
    },
    **{
        f"classification_per_class/{name}/recall": (
            f"classification_per_class/{name}/support"
        )
        for name in CLASS_NAMES
    },
    **{
        f"{prefix}/{metric}": f"{prefix}/{support}"
        for prefix in _IDENTITY_REPORT_PREFIXES
        for metric, support in (
            ("intrinsic/top1", "intrinsic/eligible_count"),
            ("intrinsic/topk", "intrinsic/eligible_count"),
            ("conditional/top1", "conditional/eligible_count"),
            ("conditional/topk", "conditional/eligible_count"),
            ("end_to_end/precision", "end_to_end/predicted_positive_count"),
            ("end_to_end/recall", "end_to_end/actual_positive_count"),
        )
    },
}


def _normalize_metrics(metrics: Mapping[str, MetricValue]) -> dict[str, MetricValue]:
    """验证扁平指标，并把有明确零支持分母的 NaN 转为 None。

    - metrics: 指标名到 float、int、bool 或 None 的映射；每个名称必须是非空字符串。

    - 返回值: 保持输入迭代顺序的新普通 dict；有限值、bool、int 与已有 None 原样保留，合规 NaN 改为 None。

    正无穷与负无穷总是失败。NaN 只有在 _NULLABLE_METRIC_SUPPORT 为该名称登记支持字段，
    且同一 metrics 映射中的支持值满足 type(value) is int 且等于 0 时才转为空值；浮点 0、
    布尔 False 或缺失支持字段都不能解释 NaN。
    """

    # dict[str,MetricValue]，按输入顺序写入验证后的扁平指标。
    normalized: dict[str, MetricValue] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise TypeError("W&B 指标名称必须是非空字符串。")
        if value is not None and not isinstance(value, (bool, int, float)):
            raise TypeError(f"W&B 指标 {name!r} 只能是数值、布尔值或 None。")
        if isinstance(value, float) and not math.isfinite(value):
            if math.isinf(value):
                raise FloatingPointError(f"W&B 指标 {name!r} 含 Inf。")
            # str | None，允许该NaN为空值时必须同批提供的支持计数字段名。
            support_name = _NULLABLE_METRIC_SUPPORT.get(name)
            # MetricValue | None，支持字段当前值；缺失与显式None在此都不能解释NaN。
            support = metrics.get(support_name) if support_name is not None else None
            if type(support) is not int or support != 0:
                raise FloatingPointError(
                    f"W&B 指标 {name!r} 含未被零支持分母解释的 NaN。"
                )
            normalized[name] = None
        else:
            normalized[name] = value
    return normalized


@dataclass
class WandbTracker:
    """保存一个已初始化 W&B online run 及其写入间隔。

    - run_id: W&B 分配的非空字符串运行标识。
    - run_url: 以 https:// 开头的在线曲线地址。
    - log_every_global_steps: 正整数参数更新写入间隔；global_step=1 始终写入。
    - _run: 满足 _WandbRun 最小协议的第三方对象；本 dataclass 不负责构造或关闭它。
    """

    run_id: str
    run_url: str
    log_every_global_steps: int
    _run: _WandbRun

    def log_global_step(
        self,
        global_step: int,
        metrics: Mapping[str, MetricValue],
    ) -> None:
        """按配置间隔记录一次完整参数更新后的训练曲线。

        - global_step: 正整数语义的累计参数更新序号；1 始终记录，其余只记录间隔整倍数；本地只执行 <= 0、!= 1 与取模运算，不做 type(value) is int 门禁，bool、float 或其它数值兼容对象的边界由这些运算决定。
        - metrics: 调用方扁平指标映射；实际写入时不得含 global_step 或 epoch 键。

        未到写入间隔时函数在检查保留键和指标值之前直接返回，因此该次 metrics 不会被验证。
        到达间隔时先用 _normalize_metrics 处理值，再加入 global_step 横轴并调用 W&B log。
        """

        if global_step <= 0:
            raise ValueError("W&B global_step 必须为正整数。")
        if global_step != 1 and global_step % self.log_every_global_steps != 0:
            return
        if "global_step" in metrics or "epoch" in metrics:
            raise ValueError("调用方指标不得覆盖 W&B 的 global_step 或 epoch 轴。")
        self._run.log({"global_step": global_step, **_normalize_metrics(metrics)})

    def log_epoch(
        self,
        epoch: int,
        global_step: int,
        metrics: Mapping[str, MetricValue],
    ) -> None:
        """记录一个已完成 epoch 的训练、validation 与选模汇总。

        - epoch: 非负整数语义的零起始轮次；本地只用数值比较，不做严格 type 门禁。
        - global_step: 正整数语义的累计参数更新数。
        - metrics: 不得含 global_step 或 epoch 的扁平指标映射。

        本方法不受 log_every_global_steps 影响；每个完成轮次都加入两条显式横轴并写入 W&B。
        """

        if epoch < 0 or global_step <= 0:
            raise ValueError("W&B epoch 必须非负，global_step 必须为正。")
        if "global_step" in metrics or "epoch" in metrics:
            raise ValueError("调用方指标不得覆盖 W&B 的 global_step 或 epoch 轴。")
        self._run.log(
            {
                "epoch": epoch,
                "global_step": global_step,
                **_normalize_metrics(metrics),
            }
        )

    def update_summary(self, values: Mapping[str, MetricValue]) -> None:
        """验证并合并阶段最终摘要字段。

        - values: 字符串名称到 float、int、bool 或 None 的映射；调用方通常提供最佳轮次、最佳分数、最后轮次、global_step 和早停状态。

        本方法不限制字段名集合，也不自动加入横轴。
        """

        self._run.summary.update(_normalize_metrics(values))

    def finish(self, exit_code: int) -> None:
        """结束后台在线同步并传递退出语义。

        - exit_code: 数值上不小于 0 的整数语义值；正式调用使用 0 表示成功、1 表示失败，本地未使用 type(...) is int 门禁。
        """

        if exit_code < 0:
            raise ValueError("W&B exit_code 不能为负数。")
        self._run.finish(exit_code=exit_code)


# ================================================================================================


def start_wandb_tracking(
    config: TrackingConfig,
    stage: StageConfig,
    stage_directory: Path,
    plan_directory: Path,
    resolved_config: Mapping[str, object],
) -> WandbTracker:
    """建立阶段级 W&B online run、声明横轴并发布 tracking.json。

    - config: 当前实验的完整 TrackingConfig。
    - config.provider: 必须为 wandb。
    - config.mode: 必须为 online。
    - config.entity: W&B 账户或团队名，传给 wandb.init。
    - config.project: W&B 项目名，传给 wandb.init。
    - config.group: W&B 比较分组名，传给 wandb.init。
    - config.log_every_global_steps: 正式解析保证为正；保存进返回 tracker。
    - config.init_timeout_seconds: 正式解析保证为正；传给 W&B Settings。
    - stage: 当前阶段的完整 StageConfig；除具名标签字段外不在本函数解释科学配置。
    - stage.name: 安全阶段名；与 plan_directory.name 组成 run_name。
    - stage.task_name: 作为 W&B job_type 和首个标签。
    - stage.language_family: 作为标签；None 转为 no_language_family。
    - stage.identity_variant: 作为标签；None 转为 no_identity_variant。
    - stage.identity_scope: 作为标签；None 转为 no_identity_scope。
    - stage.model: 提供聚合方式标签；其余模型字段不由本函数读取。
    - stage.model.aggregation: 作为第二个标签。
    - stage.model.use_manual_features: 本函数不单独读取。
    - stage.model.use_sampling_features: 本函数不单独读取。
    - stage.model.use_candidate_token_attention_bias: 本函数不单独读取。
    - stage.model.initial_graph_gate: 本函数不单独读取。
    - stage.model.initial_inventory_gate: 本函数不单独读取。
    - stage.model.model_dim: 本函数不单独读取。
    - stage.model.attention_heads: 本函数不单独读取。
    - stage.model.dropout: 本函数不单独读取。
    - stage.model.voxel_feature_channels: 本函数不单独读取。
    - stage.model.raunet_planes: 本函数不单独读取。
    - stage.model.recycle_iterations: 本函数不单独读取。
    - stage.model.gradient_checkpointing: 本函数不单独读取。
    - stage.model.bottleneck_attention_layers: 本函数不单独读取。
    - stage.model.graph_layers: 本函数不单独读取。
    - stage.training: 本函数不直接读取的完整训练配置。
    - stage.training.seed: 本函数不单独读取。
    - stage.training.physical_batch_size: 本函数不单独读取。
    - stage.training.gradient_accumulation: 本函数不单独读取。
    - stage.training.global_batch_size: 本函数不单独读取。
    - stage.training.learning_rate: 本函数不单独读取。
    - stage.training.weight_decay: 本函数不单独读取。
    - stage.training.maximum_epochs: 本函数不单独读取。
    - stage.training.patience_bad_epochs: 本函数不单独读取。
    - stage.training.precision: 本函数不单独读取。
    - stage.training.gradient_clip_norm: 本函数不单独读取。
    - stage.training.class_weights: 本函数不单独读取。
    - stage.training.loss_weights: 四项损失系数映射；本函数不直接读取。
    - stage.training.loss_weights.classification: 本函数不单独读取。
    - stage.training.loss_weights.ligand_area: 本函数不单独读取。
    - stage.training.loss_weights.binding_area: 本函数不单独读取。
    - stage.training.loss_weights.identity: 本函数不单独读取。
    - stage.initialize_candidate_from: 本函数不单独读取，但保留在 resolved_config 中发送给 W&B。
    - stage_directory: 已创建的阶段目录；用作 W&B 本地目录并承载 tracking.json。
    - plan_directory: 计划运行根；只读取 name 组成 run_name。
    - resolved_config: resolved.json 同源的字符串键映射；浅复制为普通 dict 传给 W&B，不在本函数递归验证字段。

    - 返回值.run_id: W&B 返回的非空字符串 run ID。
    - 返回值.run_url: W&B 返回且以 https:// 开头的 URL。
    - 返回值.log_every_global_steps: config.log_every_global_steps 原值。
    - 返回值._run: 已声明 global_step、epoch 和各指标通配模式的在线 W&B Run 对象。

    - tracking.json.provider: config.provider。
    - tracking.json.mode: config.mode。
    - tracking.json.entity: config.entity。
    - tracking.json.project: config.project。
    - tracking.json.group: config.group。
    - tracking.json.run_id: 返回 tracker.run_id。
    - tracking.json.run_name: stage.name__plan_directory.name。
    - tracking.json.run_url: 返回 tracker.run_url。
    - tracking.json.wandb_version: SDK 的非空字符串版本号。
    - tracking.json.local_directory: stage_directory / wandb 的路径字符串。

    正式训练由 _build_stage_runtime 把同一 stage 的全部模型与训练字段写入 resolved_config，
    但本函数接受一般 Mapping，不交叉核验 stage 与 resolved_config 是否相等。本入口不显式接收、
    传入或落盘 API key；凭据来源、singleton、Settings 与托管环境状态的解析均交给已安装 W&B SDK，
    本函数不枚举或约束其来源。模块缺失、
    在线初始化失败、空 run、非法 ID、非 HTTPS URL、版本缺失、横轴声明或 tracking.json 发布
    失败都会失败关闭。run 一旦创建，后续异常路径会尝试 finish(exit_code=1)，忽略该清理动作
    自身的异常，再重新抛出原始异常。
    """

    if config.provider != "wandb" or config.mode != "online":
        raise ValueError("正式追踪只支持 provider=wandb、mode=online。")
    try:
        # _WandbModule，运行时延迟导入的W&B SDK；cast只影响静态类型，不执行协议校验。
        wandb_module = cast(_WandbModule, importlib.import_module("wandb"))
    except ImportError as error:
        raise RuntimeError("正式环境缺少 wandb，不能启动 online 指标追踪。") from error

    # str，阶段名和计划运行目录名以两个下划线连接的在线run名称。
    run_name = f"{stage.name}__{plan_directory.name}"
    # list[str]，固定五项科学标签，None字段用具名占位字符串保留缺失语义。
    tags: list[str] = [
        stage.task_name,
        stage.model.aggregation,
        stage.language_family or "no_language_family",
        stage.identity_variant or "no_identity_variant",
        stage.identity_scope or "no_identity_scope",
    ]
    try:
        # _WandbRun | None，新建且禁止resume的在线阶段run；None由下方失败关闭。
        run = wandb_module.init(
            project=config.project,
            entity=config.entity,
            group=config.group,
            name=run_name,
            job_type=stage.task_name,
            tags=tags,
            config=dict(resolved_config),
            dir=str(stage_directory),
            mode=config.mode,
            resume="never",
            reinit=True,
            anonymous="never",
            save_code=False,
            force=True,
            settings=wandb_module.Settings(init_timeout=config.init_timeout_seconds),
        )
    except Exception as error:
        raise RuntimeError("W&B online 初始化失败；正式训练未降级为离线模式。") from error
    if run is None:
        raise RuntimeError("W&B online 初始化未返回 run。")

    try:
        # object，SDK返回的未验证run标识；正式契约要求非空str。
        raw_run_id = run.id
        if not isinstance(raw_run_id, str) or not raw_run_id:
            raise RuntimeError("W&B online run 没有返回非空字符串 run ID。")
        # object，SDK返回的未验证在线地址；正式契约要求以https://开头的str。
        raw_run_url = run.get_url()
        if not isinstance(raw_run_url, str) or not raw_run_url.startswith("https://"):
            raise RuntimeError(f"W&B online run URL 非法：{raw_run_url!r}。")
        # object，SDK模块的未验证版本字段；正式契约要求非空str。
        raw_version = wandb_module.__version__
        if not isinstance(raw_version, str) or not raw_version:
            raise RuntimeError("W&B SDK 没有提供非空字符串版本号。")
        run.define_metric("global_step")
        for section in (
            "loss",
            "gradient",
            "gate",
            "attention_bias",
            "learning_rate",
            "performance",
            "memory",
            "statistics",
        ):
            run.define_metric(f"{section}/train_step/*", step_metric="global_step")
        run.define_metric("epoch")
        # tuple[str,...]，全部使用epoch自定义横轴的训练、validation、身份与选模指标模式。
        epoch_patterns = (
            "loss/train_epoch/*",
            "loss/validation/*",
            "gradient/train_epoch/*",
            "gate/train_epoch/*",
            "learning_rate/train_epoch/*",
            "performance/train_epoch/*",
            "performance/validation/*",
            "memory/train_epoch/*",
            "statistics/train_epoch/*",
            "statistics/validation/*",
            "classification_summary/*",
            "classification_per_class/*",
            "classification_confusion/*",
            "identity_contribution/validation/*",
            *(f"{prefix}/*" for prefix in _IDENTITY_REPORT_PREFIXES),
            "area/*",
            "selection/*",
        )
        for pattern in epoch_patterns:
            run.define_metric(pattern, step_metric="epoch")
        # WandbTracker，封装已验证ID、URL、写入间隔和底层在线run。
        tracker = WandbTracker(
            run_id=raw_run_id,
            run_url=raw_run_url,
            log_every_global_steps=config.log_every_global_steps,
            _run=run,
        )
        # TrackingMetadata，写入tracking.json的十字段在线追踪事实记录。
        metadata: TrackingMetadata = {
            "provider": config.provider,
            "mode": config.mode,
            "entity": config.entity,
            "project": config.project,
            "group": config.group,
            "run_id": tracker.run_id,
            "run_name": run_name,
            "run_url": tracker.run_url,
            "wandb_version": raw_version,
            "local_directory": str(stage_directory / "wandb"),
        }
        write_json(stage_directory / "tracking.json", metadata)
        print(
            json.dumps(
                {
                    "event": "wandb_online_started",
                    "stage": stage.name,
                    "run_id": tracker.run_id,
                    "run_url": tracker.run_url,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException:
        try:
            run.finish(exit_code=1)
        except BaseException:
            pass
        raise
    return tracker
