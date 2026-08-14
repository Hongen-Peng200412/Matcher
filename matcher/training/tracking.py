"""把正式训练的 global-batch 与 validation 指标发送到 W&B online。

``start_wandb_tracking`` 为每个训练阶段建立独立在线 run，并在阶段目录写入由 ``TrackingMetadata`` 定义十字段的 ``tracking.json``。
``WandbTracker`` 记录逐 global-batch 曲线、逐 epoch 汇总和最终摘要。
仅有明确零支持分母的评估指标可转为 W&B 空值；其它非有限值与在线写入错误直接传播。
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


MetricValue: TypeAlias = float | int | bool | None


class TrackingMetadata(TypedDict):
    """``tracking.json`` 的完整字段契约。

    ``provider`` 是在线追踪实现，本轮固定为 ``wandb``。
    ``mode`` 是 W&B 运行模式，本轮固定为 ``online``。
    ``entity`` 是拥有在线项目的 W&B 账户或团队。
    ``project`` 是收纳本轮曲线的 W&B 项目名称。
    ``group`` 是用于比较同一实验批次内任务的 W&B 分组名称。
    ``run_id`` 是 W&B 分配的不可变在线运行标识。
    ``run_name`` 由训练阶段名和计划运行目录名组成。
    ``run_url`` 是用户可直接打开的 HTTPS 曲线地址。
    ``wandb_version`` 是服务器实际导入的 W&B SDK 版本。
    ``local_directory`` 是阶段目录内由 W&B 管理的本地同步目录。
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
    """限定适配层需要的 W&B Summary 写入接口。"""

    def update(self, values: Mapping[str, MetricValue]) -> None:
        """合并一组扁平的最终摘要值。"""


class _WandbRun(Protocol):
    """限定适配层实际使用的 W&B Run 最小接口。"""

    id: object
    summary: _WandbSummary

    def get_url(self) -> object:
        """返回在线运行地址；离线运行可能返回空值。"""

    def define_metric(self, name: str, *, step_metric: str | None = None) -> object:
        """声明一个指标或其自定义横轴。"""

    def log(self, values: Mapping[str, MetricValue]) -> None:
        """写入一组使用显式横轴的扁平指标。"""

    def finish(self, *, exit_code: int) -> None:
        """结束后台同步并保存进程退出语义。"""


class _WandbModule(Protocol):
    """限定延迟导入的 W&B 模块入口，避免第三方 ``Any`` 进入训练主链。"""

    __version__: object

    def Settings(self, *, init_timeout: int) -> object:
        """构造只冻结在线握手等待时间的 SDK 设置。"""

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
        """创建一个必须在线且不复用既有标识的阶段运行。"""


_IDENTITY_REPORT_PREFIXES = (
    "identity_configured",
    "identity_small_molecule",
    "identity_all_non_background",
    *(f"identity_{name}" for name in FOREGROUND_CLASS_NAMES),
)
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
    """验证扁平指标，只把同批记录中分母为零的预期 NaN 转为空值。"""

    normalized: dict[str, MetricValue] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise TypeError("W&B 指标名称必须是非空字符串。")
        if value is not None and not isinstance(value, (bool, int, float)):
            raise TypeError(f"W&B 指标 {name!r} 只能是数值、布尔值或 None。")
        if isinstance(value, float) and not math.isfinite(value):
            if math.isinf(value):
                raise FloatingPointError(f"W&B 指标 {name!r} 含 Inf。")
            support_name = _NULLABLE_METRIC_SUPPORT.get(name)
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
    """保存一个已初始化 W&B online run 及其正式写入契约。

    ``run_id`` 是 W&B 分配的不可变 run 标识。
    ``run_url`` 是用户查看在线曲线的 HTTPS 地址。
    ``log_every_global_steps`` 控制逐参数更新曲线的写入间隔；首步始终写入。
    ``_run`` 是满足最小协议的第三方 W&B Run 对象，只由本适配层调用。
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
        """按 ``global_step`` 记录一次完整参数更新后的训练曲线。"""

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
        """按零起始 ``epoch`` 记录训练汇总、validation 指标和选模状态。"""

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
        """更新 W&B run 的最终最佳轮次、分数、步数和停止状态。"""

        self._run.summary.update(_normalize_metrics(values))

    def finish(self, exit_code: int) -> None:
        """结束后台在线同步；零表示正常完成，非零表示阶段失败。"""

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
    """建立阶段级 W&B online run，并写出 ``stage_directory/tracking.json``。

    run 名称由阶段名和 allocation 的运行目录名组成；同一批次的可比任务使用配置中的 group。
    本入口只从现有环境或 ``~/.netrc`` 读取认证，不接收或落盘 API key。W&B 不可导入、
    未登录、无法在线初始化或没有返回 HTTPS URL 时立即失败，不切换为 offline。产物完整保存
    ``TrackingMetadata`` 的十个字段；初始化后任一字段、横轴或落盘检查失败都会结束该 run。
    """

    if config.provider != "wandb" or config.mode != "online":
        raise ValueError("正式追踪只支持 provider=wandb、mode=online。")
    try:
        wandb_module = cast(_WandbModule, importlib.import_module("wandb"))
    except ImportError as error:
        raise RuntimeError("正式环境缺少 wandb，不能启动 online 指标追踪。") from error

    run_name = f"{stage.name}__{plan_directory.name}"
    tags: list[str] = [
        stage.task_name,
        stage.model.aggregation,
        stage.language_family or "no_language_family",
        stage.identity_variant or "no_identity_variant",
        stage.identity_scope or "no_identity_scope",
    ]
    try:
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
        raw_run_id = run.id
        if not isinstance(raw_run_id, str) or not raw_run_id:
            raise RuntimeError("W&B online run 没有返回非空字符串 run ID。")
        raw_run_url = run.get_url()
        if not isinstance(raw_run_url, str) or not raw_run_url.startswith("https://"):
            raise RuntimeError(f"W&B online run URL 非法：{raw_run_url!r}。")
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
            *(f"{prefix}/*" for prefix in _IDENTITY_REPORT_PREFIXES),
            "area/*",
            "selection/*",
        )
        for pattern in epoch_patterns:
            run.define_metric(pattern, step_metric="epoch")
        tracker = WandbTracker(
            run_id=raw_run_id,
            run_url=raw_run_url,
            log_every_global_steps=config.log_every_global_steps,
            _run=run,
        )
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
