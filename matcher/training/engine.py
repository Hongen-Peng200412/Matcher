"""执行 Matcher 具名变体共用的单卡训练、完整验证和 BEST/LAST 治理.

主要入口是 run_smiles_training. 该入口建立 Dataset、DataLoader、模型和优化器, 每个 epoch
运行完整 validation, 按唯一 selection 更新 BEST 并执行自然早停. 运行目录保存来源路径、
manifest 快照、逐 epoch 历史、BEST/LAST checkpoint 和最终摘要, 可选向 W&B 写入在线曲线.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import socket
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from ..data.manifest import load_manifest
from ..datasets.batching import ShuffledBatchSampler, build_candidate_loader
from ..datasets.candidate_data import load_manifest_dataset
from ..datasets.contracts import CandidateBatch
from ..evaluation.metrics import ValidationAccumulator, fit_foreground_threshold
from ..model.ligand_matcher import ClosedSetMatcher
from ..utils.json_artifacts import append_jsonl, write_json
from .checkpointing import CHECKPOINT_SCHEMA_VERSION, load_checkpoint, save_checkpoint
from .config import SmilesExperimentConfig
from .losses import compute_matcher_loss


def _write_manifest_snapshot(source_path: Path, snapshot_path: Path) -> dict[str, Any]:
    """
    将 source_path 返回的 manifest 进行load, 并写入 snapshot_path.

    - source_path: 构建训练进程时解析的正式 manifest.json 路径.
    - snapshot_path: 当前运行目录内的新 JSON 路径.
    - return: 已写入 snapshot_path 的顶层 manifest 映射.

    快照把每个 split 的候选和身份文件名解析为绝对路径. 构建器为这些 JSONL 使用唯一生成
    文件名, 因此原始 manifest 后续发布新代际不会改变本次 Dataset 实际读取的记录.
    """
    source_path = Path(source_path).resolve()
    snapshot = load_manifest(source_path)
    frozen_splits: dict[str, dict[str, str]] = {}
    for split, files in snapshot["splits"].items():
        if not isinstance(files, dict) or set(files) != {"candidates", "identities"}:
            raise ValueError(f"manifest.splits.{split} 必须只含 candidates 和 identities。")
        frozen_files: dict[str, str] = {}
        for field in ("candidates", "identities"):
            value = files[field]
            if not isinstance(value, str) or not value:
                raise ValueError(f"manifest.splits.{split}.{field} 必须是非空路径字符串。")
            resolved = (source_path.parent / value).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            frozen_files[field] = str(resolved)
        frozen_splits[str(split)] = frozen_files
    snapshot["splits"] = frozen_splits
    write_json(Path(snapshot_path), snapshot)
    return snapshot


def _set_seed(seed: int) -> None:
    """
    固定当前训练进程的随机源与矩阵精度策略.

    - seed: 同时传给 Python、NumPy、PyTorch CPU 和全部可见 CUDA 设备的整数种子.
    - return: None; 同时把 float32 矩阵乘法精度设为 high.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


def _move_batch(batch: CandidateBatch, device: torch.device) -> CandidateBatch:
    """
    把批次张量迁移到训练设备.

    - batch: DataLoader 产生的 CandidateBatch.
    - device: 当前单卡训练设备.
    - return: 键集合不变的新映射; 张量使用 non_blocking=True 迁移, 字符串列表和 metadata 保留在 CPU.
    """
    return cast(
        CandidateBatch,
        {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        },
    )


def _start_tracking(config: SmilesExperimentConfig, provenance: dict[str, Any]) -> Any:
    """
    按正式配置建立一次 W&B 运行.

    - config: 当前完整实验配置.
    - provenance: 已写入 run.json 的来源和工作进程证据.
    - return: wandb.Run; tracking.mode 为 disabled 时返回 None.
    """
    if config.tracking.mode == "disabled":
        return None
    import wandb

    return wandb.init(
        entity=config.tracking.entity,
        project=config.tracking.project,
        group=config.tracking.group,
        name=f"{config.name}-{provenance['run_stamp']}",
        mode=config.tracking.mode,
        config={
            "experiment": json.loads(json.dumps(asdict(config), default=str)),
            "provenance": provenance,
        },
    )


def _track(tracker: Any, values: dict[str, Any], step: int) -> None:
    """
    记录同一参数更新步的一组 W&B 值.

    - tracker: wandb.Run 或 None.
    - values: 字符串键到标量或图像的映射.
    - step: 参数更新后的全局步数.
    - return: None; tracker 为 None 时不产生副作用.
    """
    if tracker is not None:
        tracker.log(values, step=step)


def _validation_images(
    outputs: dict[str, Any], batch: CandidateBatch, tracker: Any
) -> dict[str, Any]:
    """
    生成首个验证候选的区域最大投影图.

    - outputs: 模型输出; 两个区域 logit 有效时为浮点 (B,48,48,48).
    - batch: 已迁移到模型设备的 CandidateBatch, 含同形 bool 区域目标.
    - tracker: wandb.Run 或 None.
    - return: 四张 W&B 图像映射; 没有 tracker 或没有区域头时返回空映射.
    """
    ligand_logits = outputs["ligand_area_logits"]
    binding_logits = outputs["binding_area_logits"]
    if tracker is None or ligand_logits is None or binding_logits is None:
        return {}
    import wandb

    return {
        "validation_image/ligand_probability_max_z": wandb.Image(
            torch.sigmoid(ligand_logits[0]).amax(dim=0).float().cpu().numpy()
        ),
        "validation_image/ligand_target_max_z": wandb.Image(
            batch["ligand_area_target"][0].amax(dim=0).float().cpu().numpy()
        ),
        "validation_image/binding_probability_max_z": wandb.Image(
            torch.sigmoid(binding_logits[0]).amax(dim=0).float().cpu().numpy()
        ),
        "validation_image/binding_target_max_z": wandb.Image(
            batch["binding_area_target"][0].amax(dim=0).float().cpu().numpy()
        ),
    }


@dataclass
class _ValidationPolicyState:
    """
    保存跨 validation 检查累计的 BEST、平台计数和学习率降低次数.

    - best_score: 最近一次达到最小改善量并写入 BEST 的 selection.
    - best_epoch: BEST 所属的零起始 epoch 编号.
    - best_validation_check: BEST 所属的全局 validation 检查编号, 编号从一开始.
    - bad_validation_checks: 自上次有效改善或平台动作后连续未改善的检查数.
    - learning_rate_reductions: 整次训练已经实际降低学习率的累计次数.
    """

    best_score: float = -math.inf
    best_epoch: int = -1
    best_validation_check: int = -1
    bad_validation_checks: int = 0
    learning_rate_reductions: int = 0

    def observe(
        self,
        score: float,
        epoch: int,
        validation_check: int,
        optimizer: torch.optim.Optimizer,
        config: SmilesExperimentConfig,
    ) -> dict[str, bool]:
        """
        应用一次统一的 BEST、平台降学习率与早停规则.

        - score: 本次完整 validation 的有限 selection.
        - epoch: 本次检查所属的零起始 epoch 编号.
        - validation_check: 本次检查的一起始全局编号.
        - optimizer: 当前 AdamW; 平台动作直接更新全部参数组的学习率.
        - config: 提供最小改善量、平台耐心、降低乘数和累计降低上限.
        - return: improved、plateau、learning_rate_reduced 和 should_stop 四个布尔状态.

        patience_bad_validation_checks=3 表示前三次未改善只累计计数，第四次未改善才执行
        平台动作。有效改善清零坏检查数，但不清零累计学习率降低次数。
        """
        if not math.isfinite(score):
            raise ValueError("validation selection 必须是有限值。")
        improved = score >= (
            self.best_score + config.optimization.selection_improvement_minimum
        )
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.best_validation_check = validation_check
            self.bad_validation_checks = 0
            return {
                "improved": True,
                "plateau": False,
                "learning_rate_reduced": False,
                "should_stop": False,
            }

        self.bad_validation_checks += 1
        plateau = (
            self.bad_validation_checks
            > config.optimization.patience_bad_validation_checks
        )
        if not plateau:
            return {
                "improved": False,
                "plateau": False,
                "learning_rate_reduced": False,
                "should_stop": False,
            }

        self.bad_validation_checks = 0
        if (
            self.learning_rate_reductions
            >= config.optimization.max_learning_rate_reductions
        ):
            return {
                "improved": False,
                "plateau": True,
                "learning_rate_reduced": False,
                "should_stop": True,
            }
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] *= config.optimization.learning_rate_reduction_factor
        self.learning_rate_reductions += 1
        return {
            "improved": False,
            "plateau": True,
            "learning_rate_reduced": True,
            "should_stop": False,
        }


def _partition_optimizer_updates(total_updates: int, part_count: int) -> tuple[int, ...]:
    """
    把一个 epoch 的优化器更新划成顺序连续、大小最多相差一次更新的区段.

    - total_updates: 当前 epoch 实际执行的优化器更新总数.
    - part_count: 连续训练区段数, 也就是该 epoch 的完整 validation 次数.
    - return: 每个区段的正整数更新数; 总和严格等于 total_updates.
    """
    if type(total_updates) is not int or type(part_count) is not int:
        raise TypeError("total_updates 和 part_count 必须是整数。")
    if part_count <= 0 or total_updates < part_count:
        raise ValueError("优化器更新总数必须不小于正的区段数。")
    quotient, remainder = divmod(total_updates, part_count)
    return tuple(
        quotient + (1 if index < remainder else 0)
        for index in range(part_count)
    )


def _partition_epoch_batch_plan(
    batch_plan: list[list[int]],
    accumulation_steps: int,
    part_count: int,
) -> tuple[list[list[int]], ...]:
    """
    按完整梯度累积窗口切分一个 epoch 已冻结的物理批次计划。

    - batch_plan: list[list[int]]，当前 epoch 唯一的候选下标批次顺序。
    - accumulation_steps: int，每次优化器更新包含的连续物理批次数。
    - part_count: int，连续训练区段数，也就是完整 validation 次数。
    - return: tuple[list[list[int]], ...]，依次连接后与 batch_plan 逐批相同的非空区段。

    本函数只切片已经确定的批次计划，不重新调用采样器。区段边界严格落在完整梯度累积
    窗口之间，因此关闭和重建 DataLoader 不会遗漏、重复或改变训练样本顺序。
    """
    if type(accumulation_steps) is not int or accumulation_steps <= 0:
        raise ValueError("accumulation_steps 必须是正整数。")
    if len(batch_plan) % accumulation_steps:
        raise ValueError("物理批次计划必须形成完整梯度累积窗口。")
    update_counts = _partition_optimizer_updates(
        len(batch_plan) // accumulation_steps,
        part_count,
    )
    segments: list[list[list[int]]] = []
    start = 0
    for update_count in update_counts:
        stop = start + update_count * accumulation_steps
        segments.append(batch_plan[start:stop])
        start = stop
    if start != len(batch_plan):
        raise RuntimeError("训练区段未完整覆盖当前 epoch 批次计划。")
    return tuple(segments)






# ================================================================================================
# ================================================================================================
@torch.no_grad()
def _validate(
    model: ClosedSetMatcher,
    loader: Any,
    config: SmilesExperimentConfig,
    device: torch.device,
    global_step: int,
    tracker: Any,
) -> tuple[dict[str, Any], ValidationAccumulator]:
    """
    遍历完整 validation.

    - model: 当前 ClosedSetMatcher; 函数入口切到 eval, 返回前恢复 train.
    - loader: 不打乱、不丢尾的完整 validation DataLoader.
    - config: 损失权重和追踪设置来源.
    - device: 单卡设备.
    - global_step: 当前参数更新步, 只用于 W&B 横轴和返回记录.
    - tracker: wandb.Run 或 None.
    - return[0]: 按真实候选/身份分母汇总的损失与科学指标.
    - return[1]: 保存全部候选预测的 ValidationAccumulator, 供最终 BEST 拟合阈值.
    """
    model.eval()
    loss_numerators = {name: 0.0 for name in ("foreground", "ligand_area", "binding_area", "identity")}
    candidate_count = 0
    identity_count = 0
    accumulator = ValidationAccumulator.empty()
    image_payload: dict[str, Any] | None = None
    for cpu_batch in loader:
        batch = _move_batch(cpu_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(batch)
            losses = compute_matcher_loss(cast(dict[str, Any], outputs), cast(dict[str, torch.Tensor], batch), config.loss_weights)
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError("validation 总损失出现 NaN 或 Inf。")
        batch_count = int(batch["foreground_target"].numel())
        batch_identity_count = int(losses["identity_eligible_count"])
        candidate_count += batch_count
        identity_count += batch_identity_count
        for name in ("foreground", "ligand_area", "binding_area"):
            loss_numerators[name] += float(losses[name]) * batch_count
        loss_numerators["identity"] += float(losses["identity"]) * batch_identity_count
        probability = torch.sigmoid(outputs["foreground_logit"]).float().cpu().numpy()
        predicted_index = outputs["identity_logits"].argmax(dim=1).cpu().tolist() if outputs["identity_logits"].shape[1] else [-1] * batch_count
        for row, metadata in enumerate(cpu_batch["metadata"]):
            smiles_axis = cpu_batch["identity_smiles"][row]
            index = int(predicted_index[row])
            accumulator.foreground_target.append(bool(metadata["foreground_target"]))
            accumulator.foreground_probability.append(float(probability[row]))
            accumulator.predicted_smiles.append(smiles_axis[index] if 0 <= index < len(smiles_axis) else None)
            accumulator.target_smiles.append(metadata["target_smiles"])
            accumulator.target_category.append(metadata["target_category"])
            accumulator.identity_supported.append(bool(metadata["identity_supported"]))
        if image_payload is None:
            image_payload = _validation_images(cast(dict[str, Any], outputs), batch, tracker)
    if candidate_count == 0:
        raise ValueError("validation 候选集合不能为空。")
    metrics = accumulator.finalize()
    losses = {
        "foreground": loss_numerators["foreground"] / candidate_count,
        "ligand_area": loss_numerators["ligand_area"] / candidate_count,
        "binding_area": loss_numerators["binding_area"] / candidate_count,
        "identity": loss_numerators["identity"] / identity_count if identity_count else 0.0,
    }
    losses["total"] = (
        config.loss_weights.foreground * losses["foreground"]
        + config.loss_weights.ligand_area * losses["ligand_area"]
        + config.loss_weights.binding_area * losses["binding_area"]
        + config.loss_weights.identity * losses["identity"]
    )
    result = {"global_step": global_step, "loss": losses, "metrics": metrics}
    tracking = {
        "validation/foreground/prauc": metrics["foreground_prauc"],
        "validation/smiles_match/overall_accuracy": metrics["smiles_match"]["overall"]["accuracy"],
        "selection/score": metrics["selection"],
        **(image_payload or {}),
    }
    for category, values in metrics["smiles_match"].items():
        tracking[f"validation/smiles_match/{category}_accuracy"] = values["accuracy"]
        tracking[f"validation/smiles_match/{category}_eligible_count"] = values["eligible_count"]
    _track(tracker, tracking, global_step)
    model.train()
    return result, accumulator


def _train_segment(
    model: ClosedSetMatcher,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    config: SmilesExperimentConfig,
    device: torch.device,
    global_step: int,
    tracker: Any,
) -> tuple[dict[str, float], int]:
    """
    穷尽一个连续训练区段，并返回区段汇总与累计参数更新数。

    - model: 当前 ClosedSetMatcher. 函数入口切换到 train 模式.
    - loader: 当前区段的 DataLoader；批次数必须形成完整累积窗口，遍历结束后工作进程退出。
    - optimizer: 当前 AdamW 优化器. 每个完整窗口执行一次 step.
    - config: 完整实验配置. 提供损失权重、累积步数、裁剪阈值和追踪频率.
    - device: 当前单卡 CUDA 设备.
    - global_step: 进入本 epoch 前已经完成的参数更新数.
    - tracker: wandb.Run 或 None. None 时不写在线指标.
    - return[0]: dict[str, float]，当前区段五项物理批次均值损失、身份支持数和更新数。
    - return[1]: int，完成当前区段后的累计参数更新数。

    当前 loader 的下标批次来自当前 epoch 已冻结计划的连续切片。每个物理批次候选数相同，
    因此前景和区域均除以 accumulation_steps。身份候选数会在
    物理批次之间变化, 先读取一个完整累积窗口的真值计数, 再让各批身份均值按其合格候选
    数占窗口总数的比例回传. 这样 32x1、16x2 与 8x4 使用相同的身份损失定义.
    """
    physical_batch_count = len(loader)
    if physical_batch_count % config.data.accumulation_steps:
        raise RuntimeError("训练批次数必须形成完整梯度累积窗口。")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {name: 0.0 for name in ("total", "foreground", "ligand_area", "binding_area", "identity")}
    physical_batches = 0
    identity_eligible = 0
    optimizer_updates = 0
    window: list[CandidateBatch] = []
    for cpu_batch in loader:
        window.append(cpu_batch)
        if len(window) < config.data.accumulation_steps:
            continue
        # int 标量, 当前完整累积窗口中真实前景且 identity_target>=0 的候选总数.
        window_identity_count = sum(
            int((batch["foreground_target"] & (batch["identity_target"] >= 0)).sum())
            for batch in window
        )
        latest_losses: dict[str, torch.Tensor] | None = None
        for window_batch in window:
            batch = _move_batch(window_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = model(batch)
                losses = compute_matcher_loss(
                    cast(dict[str, Any], outputs),
                    cast(dict[str, torch.Tensor], batch),
                    config.loss_weights,
                )
                # int 标量, 当前物理批次实际进入直接 SMILES 身份交叉熵的候选数.
                batch_identity_count = int(losses["identity_eligible_count"])
                # float 标量, 当前批身份均值在全局窗口身份均值中的真实分母权重.
                identity_scale = (
                    batch_identity_count / window_identity_count
                    if window_identity_count
                    else 0.0
                )
                # 浮点标量张量, 前景/区域按物理批次数缩放, 身份按 eligible 数缩放后的反向目标.
                scaled_loss = (
                    config.loss_weights.foreground
                    * losses["foreground"]
                    / config.data.accumulation_steps
                    + config.loss_weights.ligand_area
                    * losses["ligand_area"]
                    / config.data.accumulation_steps
                    + config.loss_weights.binding_area
                    * losses["binding_area"]
                    / config.data.accumulation_steps
                    + config.loss_weights.identity * losses["identity"] * identity_scale
                )
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("训练总损失出现 NaN 或 Inf。")
            scaled_loss.backward()
            physical_batches += 1
            identity_eligible += batch_identity_count
            for name in totals:
                totals[name] += float(losses[name])
            latest_losses = cast(dict[str, torch.Tensor], losses)
        # 浮点标量张量, 当前全局批次在裁剪前的全部参数梯度 L2 范数.
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.optimization.gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("梯度范数出现 NaN 或 Inf。")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        optimizer_updates += 1
        if global_step % config.tracking.log_every_updates == 0:
            if latest_losses is None:
                raise RuntimeError("完整梯度累积窗口未产生损失。")
            _track(
                tracker,
                {
                    "training/loss_total": float(latest_losses["total"]),
                    "training/gradient_norm_before_clip": float(gradient_norm),
                    "training/learning_rate": float(optimizer.param_groups[0]["lr"]),
                },
                global_step,
            )
        window.clear()
    if window or physical_batches != physical_batch_count:
        raise RuntimeError("训练区段未严格穷尽全部物理批次。")
    summary = {name: value / physical_batches for name, value in totals.items()}
    summary["identity_eligible_count"] = float(identity_eligible)
    summary["optimizer_update_count"] = float(optimizer_updates)
    return summary, global_step


# ================================================================================================
# ================================================================================================


def run_smiles_training(config: SmilesExperimentConfig, config_path: Path) -> Path:
    """
    执行一次正式单卡从零训练.

    - config: 已严格加载的一份完整实验配置.
    - config_path: 与 config 对应的 YAML 路径; 路径写入来源证据.
    - return: 新建的运行目录, 内含 run.json、history.jsonl、BEST.pt、LAST.pt 和 summary.json.

    每个 epoch 连续训练三个区段, 每段后遍历同一完整 validation. selection 至少提高配置中的最小改善量才更新 BEST; 平台最多实际降低两次学习率, 第三次平台直接早停. 
    训练结束后只用已保存的 BEST validation 预测拟合一次前景 F1 阈值, 不额外执行第 46 次 validation.
    """
    _set_seed(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("正式 Matcher 训练要求 CUDA GPU。")
    device = torch.device("cuda")
    run_stamp = os.environ.get("TASK_RUN_STAMP") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_directory = config.output_root / config.name / run_stamp
    run_directory.mkdir(parents=True, exist_ok=False)
    config_path = Path(config_path).resolve()
    manifest_path = config.data.manifest_path.resolve()
    project_root = Path(__file__).resolve().parents[2]
    release_manifest_path = project_root.parent / "manifest.json"
    resolved_config = json.loads(json.dumps(asdict(config), default=str))
    resolved_config_path = run_directory / "resolved_config.json"
    manifest_snapshot_path = run_directory / "manifest.snapshot.json"
    write_json(resolved_config_path, resolved_config)
    _write_manifest_snapshot(manifest_path, manifest_snapshot_path)
    provenance = {
        "schema": "matcher_smiles_training_v1",
        "run_stamp": run_stamp,
        "name": config.name,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "task_launch_dir": os.environ.get("TASK_LAUNCH_DIR"),
        "source_project_root": str(project_root),
        "source_release_manifest_path": (
            str(release_manifest_path) if release_manifest_path.is_file() else None
        ),
        "config_path": str(config_path),
        "resolved_config_path": str(resolved_config_path),
        "manifest_path": str(manifest_path),
        "manifest_snapshot_path": str(manifest_snapshot_path),
        "worker_provenance": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "num_workers": config.data.num_workers,
            "loader_lifecycle": "stage_exclusive_nonpersistent",
        },
    }
    write_json(run_directory / "run.json", provenance)
    tracker = _start_tracking(config, provenance)
    try:
        train_dataset = load_manifest_dataset(
            manifest_snapshot_path,
            split=config.data.train_split,
            selected_only=config.data.selected_only,
            mode="train",
            variant=config.model.variant,
            pp_top_k=config.data.pp_top_k,
            cache_gib_per_worker=config.data.cache_gib_per_worker,
        )
        validation_dataset = load_manifest_dataset(
            manifest_snapshot_path,
            split=config.data.validation_split,
            selected_only=True,
            mode="validation",
            variant=config.model.variant,
            pp_top_k=config.data.pp_top_k,
            cache_gib_per_worker=config.data.cache_gib_per_worker,
        )
        multiple = config.data.batch_size * config.data.accumulation_steps
        train_sampler = ShuffledBatchSampler(train_dataset, config.data.batch_size, True, True, config.seed, multiple)
        validation_sampler = ShuffledBatchSampler(validation_dataset, config.data.batch_size, False, False, config.seed, None)
        model = ClosedSetMatcher(config.model).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
        )
        policy = _ValidationPolicyState()
        best_validation: dict[str, Any] | None = None
        best_accumulator: ValidationAccumulator | None = None
        global_step = 0
        last_epoch = -1
        validation_check = 0
        stopped_by_early_stopping = False
        for epoch in range(config.optimization.max_epochs):
            train_sampler.set_epoch(epoch)
            # list[list[int]]，当前 epoch 唯一的物理批次顺序；后续只连续切片，不再次采样。
            epoch_batch_plan = list(train_sampler)
            segment_batch_plans = _partition_epoch_batch_plan(
                epoch_batch_plan,
                config.data.accumulation_steps,
                config.optimization.validation_checks_per_epoch,
            )
            for segment_offset, segment_batch_plan in enumerate(segment_batch_plans):
                epoch_segment = segment_offset + 1
                train_loader = build_candidate_loader(
                    train_dataset,
                    segment_batch_plan,
                    config.data.num_workers,
                    config.data.prefetch_factor,
                    config.seed + 1 + epoch * len(segment_batch_plans) + segment_offset,
                )
                training, global_step = _train_segment(
                    model,
                    train_loader,
                    optimizer,
                    config,
                    device,
                    global_step,
                    tracker,
                )
                del train_loader
                validation_check += 1
                validation_loader = build_candidate_loader(
                    validation_dataset,
                    validation_sampler,
                    config.data.num_workers,
                    config.data.prefetch_factor,
                    config.seed + 2 + validation_check,
                )
                validation, accumulator = _validate(
                    model, validation_loader, config, device, global_step, tracker
                )
                del validation_loader
                score = float(validation["metrics"]["selection"])
                decision = policy.observe(
                    score,
                    epoch,
                    validation_check,
                    optimizer,
                    config,
                )
                if decision["improved"]:
                    best_validation = validation
                    best_accumulator = accumulator
                validation_record = {
                    "event": "validation_check",
                    "epoch": epoch,
                    "epoch_segment": epoch_segment,
                    "validation_check": validation_check,
                    "global_step": global_step,
                    "training": training,
                    "validation": validation,
                    "selection": score,
                    **decision,
                    "best_score": policy.best_score,
                    "bad_validation_checks": policy.bad_validation_checks,
                    "learning_rate_reductions": policy.learning_rate_reductions,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                append_jsonl(run_directory / "history.jsonl", validation_record)
                checkpoint = {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "epoch": epoch,
                    "epoch_segment": epoch_segment,
                    "validation_check": validation_check,
                    "global_step": global_step,
                    "best_epoch": policy.best_epoch,
                    "best_validation_check": policy.best_validation_check,
                    "best_score": policy.best_score,
                    "bad_validation_checks": policy.bad_validation_checks,
                    "learning_rate_reductions": policy.learning_rate_reductions,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "model_config": config.model.to_dict(),
                    "experiment_config": resolved_config,
                    "manifest_path": str(manifest_path),
                    "manifest_snapshot_path": str(manifest_snapshot_path),
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "validation": validation,
                }
                checkpoint_path = (
                    run_directory
                    / "checkpoints"
                    / f"epoch_{epoch + 1:02d}_validation_{validation_check:02d}.pt"
                )
                save_checkpoint(checkpoint_path, checkpoint)
                save_checkpoint(run_directory / "LAST.pt", checkpoint)
                if decision["improved"]:
                    save_checkpoint(run_directory / "BEST.pt", checkpoint)
                _track(
                    tracker,
                    {
                        "selection/best_score": policy.best_score,
                        "selection/improved": int(decision["improved"]),
                        "selection/bad_validation_checks": policy.bad_validation_checks,
                        "selection/learning_rate_reductions": policy.learning_rate_reductions,
                        "selection/plateau": int(decision["plateau"]),
                    },
                    global_step,
                )
                last_epoch = epoch
                if decision["should_stop"]:
                    stopped_by_early_stopping = True
                    break
            if stopped_by_early_stopping:
                break
        if policy.best_epoch < 0 or best_validation is None or best_accumulator is None:
            raise RuntimeError("训练未产生有限 validation selection。")
        best_checkpoint = dict(load_checkpoint(run_directory / "BEST.pt", map_location=device))
        threshold, threshold_metrics = fit_foreground_threshold(
            np.asarray(best_accumulator.foreground_target, dtype=np.bool_),
            np.asarray(best_accumulator.foreground_probability, dtype=np.float64),
        )
        best_checkpoint["foreground_threshold"] = threshold
        best_checkpoint["positive_class_definition"] = "five_true_ligand_categories"
        best_checkpoint["best_validation"] = {
            "foreground_f1": threshold_metrics["f1"],
            "foreground_prauc": best_validation["metrics"]["foreground_prauc"],
            "smiles_match_overall_accuracy": best_validation["metrics"]["smiles_match"]["overall"]["accuracy"],
            "selection": best_validation["metrics"]["selection"],
        }
        save_checkpoint(run_directory / "BEST.pt", best_checkpoint)
        threshold_record = {
            "event": "best_threshold",
            "best_epoch": policy.best_epoch,
            "best_validation_check": policy.best_validation_check,
            "foreground_threshold": threshold,
            "foreground": threshold_metrics,
            "validation": best_validation,
        }
        append_jsonl(run_directory / "history.jsonl", threshold_record)
        _track(
            tracker,
            {
                "validation/foreground/f1": threshold_metrics["f1"],
                "validation/foreground/threshold": threshold,
            },
            global_step,
        )
        summary = {
            **provenance,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "best_epoch": policy.best_epoch,
            "best_validation_check": policy.best_validation_check,
            "best_score": policy.best_score,
            "last_epoch": last_epoch,
            "global_step": global_step,
            "validation_check_count": validation_check,
            "learning_rate_reductions": policy.learning_rate_reductions,
            "stopped_by_early_stopping": stopped_by_early_stopping,
            "stopped_by_max_epochs": not stopped_by_early_stopping,
            "foreground_threshold": threshold,
            "tracking_url": getattr(tracker, "url", None),
        }
        write_json(run_directory / "summary.json", summary)
        return run_directory
    finally:
        if tracker is not None:
            tracker.finish()
