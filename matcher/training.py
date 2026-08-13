"""提供单卡、梯度累积、逐 epoch 验证和 BEST/LAST 保存的正式训练引擎。

``evaluate_model`` 汇总一个完整数据划分；``run_training_stage`` 从头执行一个正式阶段。
下列产物全部写入调用方传入的 ``stage_directory``，不会使用隐式输出位置。
``resolved.json`` 保存阶段、数据、模型、训练、初始化、数值后端和运行环境的解析后配置。
``tracking.json`` 保存 W&B online run 标识、曲线地址和本地同步目录，不保存 API key。
``history.jsonl`` 逐 epoch 保存训练统计、完整 validation、选模分数和学习率时序。
``BEST.pt`` 保存严格改善 epoch 的模型、优化器、调度器、缩放器和早停状态。
``LAST.pt`` 每个 epoch 覆盖保存同一组状态，用于保留最后一次完整训练状态。
``summary.json`` 保存阶段名、最佳与最后 epoch、最佳分数、累计更新数、早停状态、BEST/LAST 路径以及 W&B run ID 和在线曲线地址。
零支持指标在内存中为 NaN，严格 JSON 边界统一写为 ``null``。
"""

from __future__ import annotations

import itertools
import json
import math
import random
import re
import sys
import time
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypedDict, cast

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .batching import ShuffledBatchSampler, collate_candidates
from .candidate_data import CandidateDataset
from .checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    load_candidate_encoder_checkpoint,
    save_checkpoint,
)
from .contracts import CandidateBatch
from .experiment_config import DataConfig, Precision, StageConfig, TrackingConfig
from .json_artifacts import append_jsonl, write_json
from .losses import LossOutput, LossWeights, compute_matcher_loss
from .manifest import ensure_manifest
from .metrics import EvaluationAccumulator, EvaluationReport, selection_score
from .model import CandidateEncoder, ClosedSetMatcher
from .schema import SMALL_MOLECULE_INDEX
from .tracking import MetricValue, WandbTracker, start_wandb_tracking


ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
SCHEDULER_FACTOR = 0.5
SCHEDULER_MINIMUM_LEARNING_RATE = 1e-7
SCHEDULER_EPSILON = 1e-8
GRAD_SCALER_INITIAL_SCALE = 65536.0
GRAD_SCALER_GROWTH_FACTOR = 2.0
GRAD_SCALER_BACKOFF_FACTOR = 0.5
GRAD_SCALER_GROWTH_INTERVAL = 2000
_DATALOADER_MULTIPROCESSING_CONTEXT = "spawn"
_TENSOR_BATCH_KEYS = (
    "voxel_input",
    "class_target",
    "ligand_area_target",
    "binding_area_target",
    "full_candidate_voxel_count",
    "a_tokens",
    "a_mask",
    "p_tokens",
    "p_mask",
    "v_tokens",
    "v_mask",
    "v_index_zyx",
    "pp_tokens",
    "pp_mask",
    "pp_index_zyx",
    "language_embeddings",
    "identity_mask",
    "identity_target",
    "graph_to_sample",
    "graph_to_identity",
    "graph_element",
    "graph_charge",
    "graph_residue_id",
    "graph_node_continuous",
    "graph_node_graph_index",
    "graph_edge_index",
    "graph_edge_features",
)


class LossValues(TypedDict):
    """五个全局归一后的有限损失标量。

    ``total`` 是按 LossWeights 合并后的标量。
    ``classification`` 是按候选或类别权重和归一的六分类交叉熵。
    ``ligand_area`` 是按候选平均的配体区域损失。
    ``binding_area`` 是按候选平均的受体结合区域损失。
    ``identity`` 是按真实且身份目标有效的小分子候选平均的身份损失。
    """

    total: float
    classification: float
    ligand_area: float
    binding_area: float
    identity: float


class IdentityDiagnostics(TypedDict):
    """完整 Matcher 两个零初始化身份出口门在 epoch 结束时的值。

    ``graph_gate`` 是配体原子图身份分数出口的可训练标量门。
    ``inventory_gate`` 是排除自身清单身份分数出口的可训练标量门。
    """

    graph_gate: float
    inventory_gate: float


class TrainingEpochResult(TypedDict):
    """一个训练 epoch 的结构化结果。

    ``loss`` 是按完整 epoch 分母汇总的五项损失。
    ``sample_count`` 是本轮实际参与训练的候选数。
    ``identity_eligible_count`` 是有当前语言身份目标的真实小分子候选数。
    ``optimizer_steps`` 是本轮完成的 global-batch 参数更新次数。
    ``skipped_optimizer_steps`` 保留统一训练产物字段；当前 fail-fast 策略下恒为 0。
    ``mean_preclip_gradient_norm`` 是各更新裁剪前全模型 L2 梯度范数均值。
    ``maximum_preclip_gradient_norm`` 是各更新裁剪前全模型 L2 梯度范数最大值。
    ``mean_graph_branch_gradient_norm`` 是裁剪后图配对打分分支范数均值；无分支时为 None。
    ``mean_inventory_branch_gradient_norm`` 是裁剪后清单打分分支范数均值；无分支时为 None。
    ``identity_diagnostics`` 是身份门值；候选预训练没有身份分支时为 None。
    ``elapsed_seconds`` 是整轮墙钟秒数。
    ``data_wait_seconds`` 是等待 CPU DataLoader 产生完整累积组的秒数。
    ``compute_seconds`` 是 GPU 前向、反向和优化器更新的秒数。
    ``samples_per_second`` 是 sample_count 除以 elapsed_seconds。
    ``peak_cuda_allocated_bytes`` 是本轮 PyTorch 已分配显存峰值，单位 byte。
    ``peak_cuda_reserved_bytes`` 是本轮 PyTorch 保留显存峰值，单位 byte。
    ``global_step`` 是本轮结束后的累计 global-batch 更新序号。
    """

    loss: LossValues
    sample_count: int
    identity_eligible_count: int
    optimizer_steps: int
    skipped_optimizer_steps: int
    mean_preclip_gradient_norm: float
    maximum_preclip_gradient_norm: float
    mean_graph_branch_gradient_norm: float | None
    mean_inventory_branch_gradient_norm: float | None
    identity_diagnostics: IdentityDiagnostics | None
    elapsed_seconds: float
    data_wait_seconds: float
    compute_seconds: float
    samples_per_second: float
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    global_step: int


class EvaluationResult(TypedDict):
    """一个完整评估划分的结构化结果。

    ``loss`` 是按整个划分真实分母汇总的五项损失。
    ``metrics`` 是六分类、身份与区域指标的完整 EvaluationReport。
    ``sample_count`` 是该划分实际纳入评估的候选数。
    ``identity_eligible_count`` 是有当前语言身份目标的真实小分子候选数。
    ``elapsed_seconds`` 是完整评估墙钟秒数。
    ``samples_per_second`` 是 sample_count 除以 elapsed_seconds。
    """

    loss: LossValues
    metrics: EvaluationReport
    sample_count: int
    identity_eligible_count: int
    elapsed_seconds: float
    samples_per_second: float


class StageSummary(TypedDict):
    """一个正式训练阶段的收口摘要。

    ``stage`` 是 YAML 中的安全阶段名称。
    ``best_epoch`` 是最佳 validation 分数所在的零起始 epoch。
    ``best_score`` 是有限的最佳 validation 选择分数。
    ``last_epoch`` 是实际完成的最后一个零起始 epoch。
    ``global_step`` 是阶段结束后的累计 global-batch 更新次数。
    ``stopped_by_patience`` 表示是否因连续两轮 validation 无改善停止。
    ``best_checkpoint`` 是严格改善时原子更新的 BEST.pt 路径。
    ``last_checkpoint`` 是每轮原子更新的 LAST.pt 路径。
    ``tracking_run_id`` 是本阶段 W&B online run 的不可变标识。
    ``tracking_url`` 是本阶段在线曲线地址。
    """

    stage: str
    best_epoch: int
    best_score: float
    last_epoch: int
    global_step: int
    stopped_by_patience: bool
    best_checkpoint: str
    last_checkpoint: str
    tracking_run_id: str
    tracking_url: str


class EpochRecord(TypedDict):
    """写入 history 与 checkpoint 的单轮记录。

    ``epoch`` 是零起始轮次。
    ``global_step`` 是本轮训练结束后的累计参数更新次数。
    ``learning_rate_used`` 是本轮训练开始前读取的实际学习率。
    ``next_learning_rate`` 是本轮 validation 调度器更新后的下一轮学习率。
    ``selection_score`` 是只由本轮完整 validation 计算的有限选模分数。
    ``improved`` 表示 selection_score 是否严格大于此前最佳值。
    ``bad_validation_epochs`` 是截至本轮连续未严格改善的轮数。
    ``training`` 是本轮完整 TrainingEpochResult。
    ``validation`` 是本轮完整 EvaluationResult。
    ``timestamp_unix`` 是记录完成时的 Unix 墙钟秒数。
    """

    epoch: int
    global_step: int
    learning_rate_used: float
    next_learning_rate: float
    selection_score: float
    improved: bool
    bad_validation_epochs: int
    training: TrainingEpochResult
    validation: EvaluationResult
    timestamp_unix: float


@dataclass
class _LossStatistics:
    """按样本累积三个基础损失，按有效真实小分子累积身份损失分子。"""

    sample_count: int = 0
    classification_numerator: float = 0.0
    classification_denominator: float = 0.0
    ligand_area_sum: float = 0.0
    binding_area_sum: float = 0.0
    identity_numerator: float = 0.0
    identity_eligible_count: int = 0

    def update(self, losses: LossOutput, batch_size: int) -> None:
        """累加一个物理批次；身份项乘回该批次真实有效数后才加入总分子。"""

        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数。")
        finite_components = (
            ("total", losses["total"]),
            ("classification", losses["classification"]),
            ("ligand_area", losses["ligand_area"]),
            ("binding_area", losses["binding_area"]),
            ("identity", losses["identity"]),
            ("classification_denominator", losses["classification_denominator"]),
        )
        for name, value in finite_components:
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"损失分量 {name} 含 NaN 或 Inf。")
        eligible_count = int(losses["identity_eligible_count"].detach())
        if eligible_count < 0:
            raise ValueError("identity_eligible_count 不能为负数。")
        self.sample_count += batch_size
        classification_denominator = float(losses["classification_denominator"].detach())
        self.classification_numerator += (
            float(losses["classification"].detach()) * classification_denominator
        )
        self.classification_denominator += classification_denominator
        self.ligand_area_sum += float(losses["ligand_area"].detach()) * batch_size
        self.binding_area_sum += float(losses["binding_area"].detach()) * batch_size
        self.identity_numerator += float(losses["identity"].detach()) * eligible_count
        self.identity_eligible_count += eligible_count

    def compute(self, weights: LossWeights) -> LossValues:
        """返回三个样本均值、全局有效小分子均值与按配置加权的总损失。"""

        if self.sample_count == 0:
            raise RuntimeError("不能汇总空损失统计。")
        if self.classification_denominator <= 0:
            raise RuntimeError("六分类损失的全局归一分母必须为正数。")
        classification = self.classification_numerator / self.classification_denominator
        ligand_area = self.ligand_area_sum / self.sample_count
        binding_area = self.binding_area_sum / self.sample_count
        identity = (
            self.identity_numerator / self.identity_eligible_count
            if self.identity_eligible_count > 0
            else 0.0
        )
        total = (
            weights.classification * classification
            + weights.ligand_area * ligand_area
            + weights.binding_area * binding_area
            + weights.identity * identity
        )
        return {
            "total": total,
            "classification": classification,
            "ligand_area": ligand_area,
            "binding_area": binding_area,
            "identity": identity,
        }


@dataclass
class _StageRuntime:
    """保存一次正式阶段构造完成后，各 epoch 共同使用的运行对象。

    ``model`` 是已放入 CUDA 的 CandidateEncoder 或 ClosedSetMatcher。
    ``train_loader`` 只读取 train，并且只产生完整 global batch 所需的物理批次。
    ``validation_loader`` 只读取 validation，保留划分尾部不满物理批次的候选。
    ``train_sampler`` 在每轮按 seed+epoch 全候选洗牌。
    ``optimizer`` 是显式冻结全部语义的 AdamW。
    ``scheduler`` 是根据完整 validation 选择分数更新的 ReduceLROnPlateau。
    ``scaler`` 仅 precision=fp16 时启用，全部缩放参数已显式冻结。
    ``class_weights`` 是 CUDA float32 六分类权重；未配置时为 None。
    ``device`` 是正式运行使用的 CUDA 设备。
    ``training_config`` 是写入 checkpoint 的 JSON 可序列化训练配置字典。
    ``resolved_config`` 是已写入 resolved.json 并发送给 W&B 的完整解析配置。
    """

    model: nn.Module
    train_loader: DataLoader[CandidateBatch]
    validation_loader: DataLoader[CandidateBatch]
    train_sampler: ShuffledBatchSampler
    optimizer: AdamW
    scheduler: ReduceLROnPlateau
    scaler: torch.amp.GradScaler
    class_weights: torch.Tensor | None
    device: torch.device
    training_config: dict[str, Any]
    resolved_config: dict[str, object]


def _build_stage_runtime(
    stage: StageConfig,
    data: DataConfig,
    tracking: TrackingConfig,
    stage_directory: Path,
    plan_directory: Path,
) -> _StageRuntime:
    """一次性构造数据、模型和优化状态，并写出可审计的 resolved.json。"""

    _configure_random_seeds_and_numeric_backend(stage.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("正式 Matcher 训练必须在 CUDA GPU 上运行。")
    manifest_path = ensure_manifest(
        data.roots,
        data.coverage_threshold,
        data.selection_seed,
    )
    cache_bytes_per_worker = int(data.cache_gib_per_worker * 1024**3)
    train_dataset = CandidateDataset(
        roots=data.roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name=stage.task_name,
        language_family=stage.language_family,
        pp_top_k=data.pp_top_k,
        cache_bytes_per_worker=cache_bytes_per_worker,
    )
    validation_dataset = CandidateDataset(
        roots=data.roots,
        manifest_path=manifest_path,
        split_name="validation",
        task_name=stage.task_name,
        language_family=stage.language_family,
        pp_top_k=data.pp_top_k,
        cache_bytes_per_worker=cache_bytes_per_worker,
    )
    train_sampler = ShuffledBatchSampler(
        train_dataset,
        batch_size=stage.training.physical_batch_size,
        shuffle=True,
        drop_last=True,
        seed=stage.training.seed,
        sample_count_multiple=stage.training.global_batch_size,
    )
    validation_sampler = ShuffledBatchSampler(
        validation_dataset,
        batch_size=stage.training.physical_batch_size,
        shuffle=False,
        drop_last=False,
        seed=stage.training.seed,
        sample_count_multiple=None,
    )
    train_loader = cast(
        DataLoader[CandidateBatch],
        DataLoader(
            cast(Any, train_dataset),
            batch_sampler=train_sampler,
            num_workers=data.num_workers,
            collate_fn=collate_candidates,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=data.prefetch_factor,
            multiprocessing_context=_DATALOADER_MULTIPROCESSING_CONTEXT,
        ),
    )
    validation_loader = cast(
        DataLoader[CandidateBatch],
        DataLoader(
            cast(Any, validation_dataset),
            batch_sampler=validation_sampler,
            num_workers=data.num_workers,
            collate_fn=collate_candidates,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=data.prefetch_factor,
            multiprocessing_context=_DATALOADER_MULTIPROCESSING_CONTEXT,
        ),
    )

    model: nn.Module
    if stage.task_name == "candidate_pretrain":
        model = CandidateEncoder(stage.model)
    else:
        model = ClosedSetMatcher(stage.model)
    initialization: dict[str, Any] | None = None
    if stage.initialize_candidate_from is not None:
        checkpoint_path = _resolve_initial_checkpoint(
            stage.initialize_candidate_from,
            plan_directory,
        )
        initialization = {
            "checkpoint": str(checkpoint_path),
            **load_candidate_encoder_checkpoint(model, checkpoint_path),
        }
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=stage.training.learning_rate,
        weight_decay=stage.training.weight_decay,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
        amsgrad=False,
        maximize=False,
        foreach=True,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=0,
        threshold=0.0,
        threshold_mode="abs",
        cooldown=0,
        min_lr=SCHEDULER_MINIMUM_LEARNING_RATE,
        eps=SCHEDULER_EPSILON,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=stage.training.precision == "fp16",
        init_scale=GRAD_SCALER_INITIAL_SCALE,
        growth_factor=GRAD_SCALER_GROWTH_FACTOR,
        backoff_factor=GRAD_SCALER_BACKOFF_FACTOR,
        growth_interval=GRAD_SCALER_GROWTH_INTERVAL,
    )
    class_weights = (
        None
        if stage.training.class_weights is None
        else torch.tensor(stage.training.class_weights, device=device, dtype=torch.float32)
    )
    stage_record = asdict(stage)
    stage_record["model"] = stage.model.to_dict()
    data_record = {
        "data_root": str(data.roots.data_root),
        "stage1_root": str(data.roots.stage1_root),
        "output_root": str(data.roots.output_root),
        "coverage_threshold": data.coverage_threshold,
        "selection_seed": data.selection_seed,
        "pp_top_k": data.pp_top_k,
        "num_workers": data.num_workers,
        "prefetch_factor": data.prefetch_factor,
        "cache_gib_per_worker": data.cache_gib_per_worker,
        "multiprocessing_context": _DATALOADER_MULTIPROCESSING_CONTEXT,
    }
    training_record = stage_record["training"]
    if not isinstance(training_record, dict):
        raise RuntimeError("阶段训练配置序列化后必须是字典。")
    resolved: dict[str, object] = {
        "stage": stage_record,
        "data": data_record,
        "tracking": asdict(tracking),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "frozen_optimizer_semantics": {
            "adam_betas": ADAM_BETAS,
            "adam_epsilon": ADAM_EPSILON,
            "adam_amsgrad": False,
            "adam_maximize": False,
            "adam_foreach": True,
            "adam_capturable": False,
            "adam_differentiable": False,
            "adam_fused": False,
            "scheduler": "ReduceLROnPlateau",
            "scheduler_mode": "max",
            "scheduler_factor": SCHEDULER_FACTOR,
            "scheduler_patience": 0,
            "scheduler_threshold": 0.0,
            "scheduler_threshold_mode": "abs",
            "scheduler_cooldown": 0,
            "scheduler_minimum_learning_rate": SCHEDULER_MINIMUM_LEARNING_RATE,
            "scheduler_epsilon": SCHEDULER_EPSILON,
            "grad_scaler_initial_scale": GRAD_SCALER_INITIAL_SCALE,
            "grad_scaler_growth_factor": GRAD_SCALER_GROWTH_FACTOR,
            "grad_scaler_backoff_factor": GRAD_SCALER_BACKOFF_FACTOR,
            "grad_scaler_growth_interval": GRAD_SCALER_GROWTH_INTERVAL,
        },
        "numeric_backend": {
            "python_numpy_torch_seed": stage.training.seed,
            "cudnn_benchmark": True,
            "cudnn_deterministic": False,
            "deterministic_algorithms": False,
            "float32_matmul_precision": "high",
        },
        "initialization": initialization,
        "train_candidate_count": len(train_dataset),
        "train_candidates_used_per_epoch": len(train_sampler)
        * stage.training.physical_batch_size,
        "validation_candidate_count": len(validation_dataset),
    }
    write_json(stage_directory / "resolved.json", resolved)
    return _StageRuntime(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_sampler=train_sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        class_weights=class_weights,
        device=device,
        training_config=training_record,
        resolved_config=resolved,
    )


def _train_epoch(
    model: nn.Module,
    loader: DataLoader[CandidateBatch],
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    stage: StageConfig,
    class_weights: torch.Tensor | None,
    device: torch.device,
    initial_global_step: int,
    tracker: WandbTracker,
) -> TrainingEpochResult:
    """执行完整 global batch，并把每次参数更新后的有限指标发送到阶段 W&B run。"""

    started_at = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = stage.training.gradient_accumulation
    statistics = _LossStatistics()
    global_step = initial_global_step
    batch_count = len(loader)
    if batch_count == 0:
        raise RuntimeError("训练 DataLoader 没有产生任何批次。")
    if batch_count % accumulation != 0:
        raise RuntimeError("训练 sampler 必须产生 gradient_accumulation 的整数倍物理批次。")
    complete_group_count = batch_count // accumulation
    if complete_group_count == 0:
        raise RuntimeError("训练批次数不足以构成一个 global batch。")
    torch.cuda.reset_peak_memory_stats(device)
    data_wait_seconds = 0.0
    compute_seconds = 0.0
    gradient_norm_sum = 0.0
    maximum_gradient_norm = 0.0
    graph_gradient_norm_sum = 0.0
    inventory_gradient_norm_sum = 0.0
    identity_gradient_step_count = 0
    # 非有限反向损失与梯度在 optimizer.step 前直接报错，因此正式策略不跳过更新。
    skipped_optimizer_steps = 0
    loader_iterator = iter(loader)
    for _ in range(complete_group_count):
        wait_started_at = time.perf_counter()
        cpu_group = list(itertools.islice(loader_iterator, accumulation))
        group_data_wait_seconds = time.perf_counter() - wait_started_at
        data_wait_seconds += group_data_wait_seconds
        if len(cpu_group) != accumulation:
            raise RuntimeError("DataLoader 在完整 global batch 中途提前结束。")
        group_sample_count = sum(int(batch["class_target"].shape[0]) for batch in cpu_group)
        if group_sample_count != stage.training.global_batch_size:
            raise RuntimeError(
                f"global batch 应含 {stage.training.global_batch_size} 个候选，"
                f"实际为 {group_sample_count}。"
            )
        group_eligible_count = sum(
            int(
                (
                    (batch["class_target"] == SMALL_MOLECULE_INDEX)
                    & (batch["identity_target"] >= 0)
                ).sum()
            )
            for batch in cpu_group
        )
        group_classification_denominator = _classification_denominator(
            cpu_group,
            stage.training.class_weights,
        )
        group_statistics = _LossStatistics()
        compute_started_at = time.perf_counter()
        for cpu_batch in cpu_group:
            batch = _move_batch(cpu_batch, device)
            batch_size = int(batch["class_target"].shape[0])
            with _autocast_context(stage.training.precision):
                outputs = model(batch)
                losses = compute_matcher_loss(
                    outputs,
                    batch,
                    stage.training.loss_weights,
                    stage.identity_variant,
                    class_weights,
                )
                scaled_loss = _accumulation_backward_loss(
                    losses,
                    stage.training.loss_weights,
                    sample_fraction=batch_size / group_sample_count,
                    group_classification_denominator=group_classification_denominator,
                    group_identity_eligible_count=group_eligible_count,
                )
                if not torch.isfinite(scaled_loss):
                    raise FloatingPointError("global batch 反向损失含 NaN 或 Inf。")
            scaler.scale(scaled_loss).backward()
            statistics.update(losses, batch_size)
            group_statistics.update(losses, batch_size)
        scaler.unscale_(optimizer)
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                stage.training.gradient_clip_norm,
                error_if_nonfinite=True,
            ).detach()
        )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("裁剪前梯度范数含 NaN 或 Inf。")
        gradient_norm_sum += gradient_norm
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        graph_gradient_norm = _named_gradient_norm(model, "identity_scorer.graph.")
        inventory_gradient_norm = _named_gradient_norm(model, "identity_scorer.inventory.")
        if graph_gradient_norm is not None and inventory_gradient_norm is not None:
            graph_gradient_norm_sum += graph_gradient_norm
            inventory_gradient_norm_sum += inventory_gradient_norm
            identity_gradient_step_count += 1
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        is_first_stage_step = global_step == 1
        # CUDA 前向、反向与 AdamW 更新均可能异步；每步同步后才能得到可跨硬件比较的完整耗时。
        torch.cuda.synchronize(device)
        global_step_compute_seconds = time.perf_counter() - compute_started_at
        compute_seconds += global_step_compute_seconds
        if is_first_stage_step:
            print(
                json.dumps(
                    {
                        "event": "first_global_step_accepted",
                        "stage": stage.name,
                        "global_step": global_step,
                        "global_batch_size": group_sample_count,
                        "identity_eligible_count": group_eligible_count,
                        "preclip_gradient_norm": gradient_norm,
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                        "global_step_compute_seconds": global_step_compute_seconds,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        group_loss = group_statistics.compute(stage.training.loss_weights)
        if isinstance(model, ClosedSetMatcher):
            graph_gate: float | None = float(model.identity_scorer.graph_gate.detach())
            inventory_gate: float | None = float(model.identity_scorer.inventory_gate.detach())
        else:
            graph_gate = None
            inventory_gate = None
        step_metrics: dict[str, MetricValue] = {
            "train/step/loss_total": group_loss["total"],
            "train/step/loss_classification": group_loss["classification"],
            "train/step/loss_ligand_area": group_loss["ligand_area"],
            "train/step/loss_binding_area": group_loss["binding_area"],
            "train/step/loss_identity": group_loss["identity"],
            "train/step/identity_eligible_count": group_eligible_count,
            "train/step/preclip_gradient_norm": gradient_norm,
            "train/step/graph_branch_gradient_norm": graph_gradient_norm,
            "train/step/inventory_branch_gradient_norm": inventory_gradient_norm,
            "train/step/graph_gate": graph_gate,
            "train/step/inventory_gate": inventory_gate,
            "train/step/learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train/step/data_wait_seconds": group_data_wait_seconds,
            "train/step/compute_seconds": global_step_compute_seconds,
            "train/step/samples_per_second": group_sample_count
            / max(group_data_wait_seconds + global_step_compute_seconds, 1e-12),
            "train/step/peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "train/step/peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        tracker.log_global_step(global_step, step_metrics)
    elapsed_seconds = time.perf_counter() - started_at
    identity_diagnostics: IdentityDiagnostics | None
    if isinstance(model, ClosedSetMatcher):
        identity_diagnostics = {
            "graph_gate": float(model.identity_scorer.graph_gate.detach()),
            "inventory_gate": float(model.identity_scorer.inventory_gate.detach()),
        }
    else:
        identity_diagnostics = None
    return {
        "loss": statistics.compute(stage.training.loss_weights),
        "sample_count": statistics.sample_count,
        "identity_eligible_count": statistics.identity_eligible_count,
        "optimizer_steps": complete_group_count,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "mean_preclip_gradient_norm": gradient_norm_sum / complete_group_count,
        "maximum_preclip_gradient_norm": maximum_gradient_norm,
        "mean_graph_branch_gradient_norm": (
            graph_gradient_norm_sum / identity_gradient_step_count
            if identity_gradient_step_count > 0
            else None
        ),
        "mean_inventory_branch_gradient_norm": (
            inventory_gradient_norm_sum / identity_gradient_step_count
            if identity_gradient_step_count > 0
            else None
        ),
        "identity_diagnostics": identity_diagnostics,
        "elapsed_seconds": elapsed_seconds,
        "data_wait_seconds": data_wait_seconds,
        "compute_seconds": compute_seconds,
        "samples_per_second": statistics.sample_count / max(elapsed_seconds, 1e-12),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "global_step": global_step,
    }


def _accumulation_backward_loss(
    losses: LossOutput,
    weights: LossWeights,
    sample_fraction: float,
    group_classification_denominator: float,
    group_identity_eligible_count: int,
) -> torch.Tensor:
    """按 global batch 样本数缩放基础项，并按同一 global batch 的真实有效小分子数缩放身份项。"""

    classification_fraction = (
        losses["classification_denominator"] / group_classification_denominator
    )
    base = (
        weights.classification * losses["classification"] * classification_fraction
        + weights.ligand_area * losses["ligand_area"] * sample_fraction
        + weights.binding_area * losses["binding_area"]
        * sample_fraction
    )
    local_eligible_count = losses["identity_eligible_count"].to(dtype=base.dtype)
    identity_fraction = local_eligible_count / max(group_identity_eligible_count, 1)
    return base + weights.identity * losses["identity"] * identity_fraction


def _classification_denominator(
    batches: list[CandidateBatch],
    class_weights: tuple[float, ...] | None,
) -> float:
    """返回一个 global batch 的六分类真实归一分母；无类别权重时就是候选数。"""

    if class_weights is None:
        return float(sum(int(batch["class_target"].numel()) for batch in batches))
    weights = torch.tensor(class_weights, dtype=torch.float64)
    denominator = sum(float(weights[batch["class_target"]].sum()) for batch in batches)
    if denominator <= 0:
        raise ValueError("一个 global batch 的六分类权重和必须为正数。")
    return denominator


def _named_gradient_norm(model: nn.Module, prefix: str) -> float | None:
    """返回具名前缀下全部现有梯度的 L2 范数；不存在该分支时返回 ``None``。"""

    squared_sum = 0.0
    found = False
    for name, parameter in model.named_parameters():
        if name.startswith(prefix) and parameter.grad is not None:
            squared_sum += float(parameter.grad.detach().float().square().sum())
            found = True
    return math.sqrt(squared_sum) if found else None


def _move_batch(batch: CandidateBatch, device: torch.device) -> CandidateBatch:
    """把所有张量异步移动到 GPU，并保留 CPU 元数据与 object_key 列表。"""

    moved: dict[str, Any] = dict(batch)
    for key in _TENSOR_BATCH_KEYS:
        value = moved[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"CandidateBatch[{key!r}] 必须是 torch.Tensor。")
        moved[key] = value.to(device, non_blocking=True)
    return cast(CandidateBatch, moved)


def _autocast_context(precision: Precision) -> AbstractContextManager[None]:
    """按配置返回 CUDA 自动混合精度上下文；fp32 使用禁用状态。"""

    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cuda", enabled=False)


def _resolve_initial_checkpoint(specification: str, plan_directory: Path) -> Path:
    """解析绝对路径或 ``stage:<name>:BEST`` 的同计划阶段 checkpoint 引用。"""

    if specification.startswith("stage:"):
        parts = specification.split(":")
        if (
            len(parts) != 3
            or parts[2] not in ("BEST", "LAST")
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", parts[1]) is None
        ):
            raise ValueError("阶段 checkpoint 引用必须写为 stage:<name>:BEST|LAST。")
        return plan_directory / parts[1] / f"{parts[2]}.pt"
    path = PurePosixPath(specification)
    if not path.is_absolute():
        raise ValueError("外部 checkpoint 必须使用绝对路径。")
    return Path(path.as_posix())


def _configure_random_seeds_and_numeric_backend(seed: int) -> None:
    """设置随机种子和高吞吐数值后端；不承诺跨硬件逐位可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("high")


# ================================================================================================


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[CandidateBatch],
    stage: StageConfig,
    class_weights: torch.Tensor | None,
    device: torch.device,
) -> EvaluationResult:
    """在调用方指定的 train 或 validation loader 上返回 ``EvaluationResult``。

    损失按全划分真实分母汇总；指标由 ``EvaluationAccumulator`` 跨物理批次合并。
    任一非有限模型输出或损失立即终止，不允许产生 BEST checkpoint。
    """

    started_at = time.perf_counter()
    model.eval()
    metrics = EvaluationAccumulator(top_k=5)
    statistics = _LossStatistics()
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = _move_batch(cpu_batch, device)
            with _autocast_context(stage.training.precision):
                outputs = model(batch)
                losses = compute_matcher_loss(
                    outputs,
                    batch,
                    stage.training.loss_weights,
                    stage.identity_variant,
                    class_weights,
                )
            batch_size = int(batch["class_target"].shape[0])
            statistics.update(losses, batch_size)
            metrics.update(outputs, batch)
    if statistics.sample_count == 0:
        raise RuntimeError("评估 DataLoader 没有产生任何样本。")
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "loss": statistics.compute(stage.training.loss_weights),
        "metrics": metrics.compute(),
        "sample_count": statistics.sample_count,
        "identity_eligible_count": statistics.identity_eligible_count,
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": statistics.sample_count / max(elapsed_seconds, 1e-12),
    }


def run_training_stage(
    stage: StageConfig,
    data: DataConfig,
    tracking: TrackingConfig,
    stage_directory: Path,
    plan_directory: Path,
) -> StageSummary:
    """从头执行一个正式阶段，并返回已写入 ``stage_directory/summary.json`` 的 StageSummary。

    本入口只构造 train 与 validation，不读取 calibration。它先创建不可复用的阶段目录，
    再冻结运行环境，并写出 resolved、tracking、history、BEST、LAST 和 summary 六类产物。
    W&B online 初始化发生在任何 DataLoader worker 启动前；worker 使用 ``spawn``，不会继承
    主进程的 W&B 客户端或 TLS 会话。初始化或同步失败不降级为离线。
    """

    stage_directory.mkdir(parents=True, exist_ok=False)
    runtime = _build_stage_runtime(
        stage,
        data,
        tracking,
        stage_directory,
        plan_directory,
    )
    tracker = start_wandb_tracking(
        tracking,
        stage,
        stage_directory,
        plan_directory,
        runtime.resolved_config,
    )
    try:
        best_score = -math.inf
        best_epoch = -1
        bad_validation_epochs = 0
        global_step = 0
        history_path = stage_directory / "history.jsonl"
        epoch = 0
        last_epoch = -1
        while stage.training.maximum_epochs is None or epoch < stage.training.maximum_epochs:
            last_epoch = epoch
            runtime.train_sampler.set_epoch(epoch)
            learning_rate_used = float(runtime.optimizer.param_groups[0]["lr"])
            training = _train_epoch(
                runtime.model,
                runtime.train_loader,
                runtime.optimizer,
                runtime.scaler,
                stage,
                runtime.class_weights,
                runtime.device,
                global_step,
                tracker,
            )
            global_step = int(training["global_step"])
            validation = evaluate_model(
                runtime.model,
                runtime.validation_loader,
                stage,
                runtime.class_weights,
                runtime.device,
            )
            validation_metrics = validation["metrics"]
            score = selection_score(
                stage.task_name,
                validation_metrics["six_class"]["macro_f1"],
                validation_metrics["conditional_identity"]["top1"],
                validation_metrics["end_to_end_identity"]["f1"],
            )
            if not math.isfinite(score):
                raise FloatingPointError("validation 选择分数不是有限值。")
            runtime.scheduler.step(score)
            next_learning_rate = float(runtime.optimizer.param_groups[0]["lr"])
            improved = score > best_score
            if improved:
                best_score = score
                best_epoch = epoch
                bad_validation_epochs = 0
            else:
                bad_validation_epochs += 1
            epoch_record: EpochRecord = {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate_used": learning_rate_used,
                "next_learning_rate": next_learning_rate,
                "selection_score": score,
                "improved": improved,
                "bad_validation_epochs": bad_validation_epochs,
                "training": training,
                "validation": validation,
                "timestamp_unix": time.time(),
            }
            append_jsonl(history_path, epoch_record)
            checkpoint_state: dict[str, object] = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "stage_name": stage.name,
                "task_name": stage.task_name,
                "epoch": epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "bad_validation_epochs": bad_validation_epochs,
                "model_config": stage.model.to_dict(),
                "training_config": runtime.training_config,
                "model_state": runtime.model.state_dict(),
                "optimizer_state": runtime.optimizer.state_dict(),
                "scheduler_state": runtime.scheduler.state_dict(),
                "scaler_state": runtime.scaler.state_dict(),
                "epoch_record": epoch_record,
            }
            save_checkpoint(stage_directory / "LAST.pt", checkpoint_state)
            if improved:
                save_checkpoint(stage_directory / "BEST.pt", checkpoint_state)
            validation_report = validation["metrics"]
            six_class = validation_report["six_class"]
            intrinsic_identity = validation_report["intrinsic_identity"]
            conditional_identity = validation_report["conditional_identity"]
            end_to_end_identity = validation_report["end_to_end_identity"]
            area = validation_report["area"]
            epoch_metrics: dict[str, MetricValue] = {
                "train/epoch/loss_total": training["loss"]["total"],
                "train/epoch/loss_classification": training["loss"]["classification"],
                "train/epoch/loss_ligand_area": training["loss"]["ligand_area"],
                "train/epoch/loss_binding_area": training["loss"]["binding_area"],
                "train/epoch/loss_identity": training["loss"]["identity"],
                "train/epoch/samples_per_second": training["samples_per_second"],
                "train/epoch/data_wait_seconds": training["data_wait_seconds"],
                "train/epoch/compute_seconds": training["compute_seconds"],
                "train/epoch/mean_preclip_gradient_norm": training[
                    "mean_preclip_gradient_norm"
                ],
                "train/epoch/maximum_preclip_gradient_norm": training[
                    "maximum_preclip_gradient_norm"
                ],
                "train/epoch/learning_rate_used": learning_rate_used,
                "validation/loss_total": validation["loss"]["total"],
                "validation/loss_classification": validation["loss"]["classification"],
                "validation/loss_ligand_area": validation["loss"]["ligand_area"],
                "validation/loss_binding_area": validation["loss"]["binding_area"],
                "validation/loss_identity": validation["loss"]["identity"],
                "validation/samples_per_second": validation["samples_per_second"],
                "validation/six_class_accuracy": six_class["accuracy"],
                "validation/six_class_macro_f1": six_class["macro_f1"],
                "validation/intrinsic_identity_top1": intrinsic_identity["top1"],
                "validation/intrinsic_identity_topk": intrinsic_identity["topk"],
                "validation/intrinsic_identity_eligible_count": intrinsic_identity[
                    "eligible_count"
                ],
                "validation/conditional_identity_top1": conditional_identity["top1"],
                "validation/conditional_identity_topk": conditional_identity["topk"],
                "validation/conditional_identity_eligible_count": conditional_identity[
                    "eligible_count"
                ],
                "validation/conditional_seen_top1": conditional_identity["seen_top1"],
                "validation/conditional_seen_topk": conditional_identity["seen_topk"],
                "validation/conditional_seen_count": conditional_identity["seen_count"],
                "validation/conditional_unseen_top1": conditional_identity["unseen_top1"],
                "validation/conditional_unseen_topk": conditional_identity["unseen_topk"],
                "validation/conditional_unseen_count": conditional_identity["unseen_count"],
                "validation/end_to_end_identity_precision": end_to_end_identity["precision"],
                "validation/end_to_end_identity_recall": end_to_end_identity["recall"],
                "validation/end_to_end_identity_f1": end_to_end_identity["f1"],
                "validation/end_to_end_true_positive_count": end_to_end_identity[
                    "true_positive"
                ],
                "validation/end_to_end_predicted_small_molecule_count": end_to_end_identity[
                    "predicted_small_molecule"
                ],
                "validation/end_to_end_actual_small_molecule_count": end_to_end_identity[
                    "actual_small_molecule"
                ],
                "validation/ligand_micro_dice": area["ligand_micro_dice"],
                "validation/binding_micro_dice": area["binding_micro_dice"],
                "validation/ligand_dice_denominator": area["ligand_dice_denominator"],
                "validation/binding_dice_denominator": area["binding_dice_denominator"],
                "selection/score": score,
                "selection/best_score": best_score,
                "selection/improved": improved,
                "selection/bad_validation_epochs": bad_validation_epochs,
                "selection/next_learning_rate": next_learning_rate,
            }
            for class_name, class_metrics in six_class["per_class"].items():
                epoch_metrics[f"validation/six_class_f1/{class_name}"] = class_metrics["f1"]
                epoch_metrics[f"validation/six_class_support/{class_name}"] = class_metrics[
                    "support"
                ]
            tracker.log_epoch(epoch, global_step, epoch_metrics)
            print(
                json.dumps(
                    {
                        "stage": stage.name,
                        "epoch": epoch,
                        "score": score,
                        "best_score": best_score,
                        "bad_validation_epochs": bad_validation_epochs,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if bad_validation_epochs >= stage.training.patience_bad_epochs:
                break
            epoch += 1
        if best_epoch < 0:
            raise RuntimeError(f"阶段 {stage.name} 没有产生有限 validation 选择分数。")
        summary: StageSummary = {
            "stage": stage.name,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "last_epoch": last_epoch,
            "global_step": global_step,
            "stopped_by_patience": (
                bad_validation_epochs >= stage.training.patience_bad_epochs
            ),
            "best_checkpoint": str(stage_directory / "BEST.pt"),
            "last_checkpoint": str(stage_directory / "LAST.pt"),
            "tracking_run_id": tracker.run_id,
            "tracking_url": tracker.run_url,
        }
        tracker.update_summary(
            {
                "best_epoch": best_epoch,
                "best_score": best_score,
                "last_epoch": last_epoch,
                "global_step": global_step,
                "stopped_by_patience": summary["stopped_by_patience"],
            }
        )
        write_json(stage_directory / "summary.json", summary)
    except BaseException as training_error:
        try:
            tracker.finish(exit_code=1)
        except BaseException as finish_error:
            print(
                json.dumps(
                    {
                        "event": "wandb_finish_failed_after_training_error",
                        "stage": stage.name,
                        "training_error_type": type(training_error).__name__,
                        "finish_error_type": type(finish_error).__name__,
                        "finish_error": str(finish_error),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        raise
    tracker.finish(exit_code=0)
    return summary
