"""当前 Anchor O/O′ Matcher 的单卡 Phase1/Phase2 训练入口。"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .anchor_data import AnchorDataConfig, AnchorPocketDataset, AnchorSample
from .batching import OccurrenceBudgetBatchSampler, collate_anchor_batch
from .consistency import O_O_prime_consistency
from .model import Matcher, PDBMatcherOutput
from .objectives import MatcherLoss, matcher_loss
from .oo_prime import OOPrimeEvaluation


class WarmupPlateau:
    """先按 optimizer step 线性 warmup，再按完整验证事件推进 plateau。"""

    def __init__(
        self,
        optimizer: AdamW,
        *,
        warmup_steps: int,
        warmup_start_factor: float,
        factor: float,
        patience: int,
        threshold: float,
        stop_after_lr_reductions: int,
        cooldown: int = 0,
        min_lr: float = 0.0,
        eps: float = 1.0e-8,
    ) -> None:
        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.warmup_steps = warmup_steps
        self.warmup_start_factor = warmup_start_factor
        self.warmup_step = 0
        self.reduction_count = 0
        self.stop_after_lr_reductions = stop_after_lr_reductions
        self.plateau = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode="abs",
            cooldown=cooldown,
            min_lr=min_lr,
            eps=eps,
        )
        if warmup_steps:
            for group, base_lr in zip(optimizer.param_groups, self.base_lrs, strict=True):
                group["lr"] = base_lr * warmup_start_factor

    def step_optimizer(self) -> None:
        if self.warmup_step >= self.warmup_steps:
            return
        self.warmup_step += 1
        progress = self.warmup_step / self.warmup_steps
        factor = self.warmup_start_factor + progress * (1.0 - self.warmup_start_factor)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * factor

    def step_validation(self, metric: float) -> None:
        if self.warmup_step < self.warmup_steps:
            return
        before = [group["lr"] for group in self.optimizer.param_groups]
        self.plateau.step(metric)
        after = [group["lr"] for group in self.optimizer.param_groups]
        self.reduction_count += int(any(new < old - 1.0e-8 for old, new in zip(before, after)))

    @property
    def should_stop(self) -> bool:
        return self.reduction_count >= self.stop_after_lr_reductions

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_lrs": self.base_lrs,
            "warmup_steps": self.warmup_steps,
            "warmup_start_factor": self.warmup_start_factor,
            "warmup_step": self.warmup_step,
            "reduction_count": self.reduction_count,
            "stop_after_lr_reductions": self.stop_after_lr_reductions,
            "plateau": self.plateau.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.base_lrs = state["base_lrs"]
        self.warmup_steps = state["warmup_steps"]
        self.warmup_start_factor = state["warmup_start_factor"]
        self.warmup_step = state["warmup_step"]
        self.reduction_count = state["reduction_count"]
        self.stop_after_lr_reductions = state["stop_after_lr_reductions"]
        self.plateau.load_state_dict(state["plateau"])


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _data_config(config: DictConfig, split: str, training: bool) -> AnchorDataConfig:
    phase = int(config.run.phase)
    return AnchorDataConfig(
        data_root=Path(config.data.data_root),
        stage1_preparation_root=Path(config.data.stage1_preparation_root),
        experiment_manifest=Path(config.data.experiment_manifest),
        split=split,
        training=training,
        seed=int(config.run.seed),
        p_miss=float(config.data.p_miss),
        p_split=float(config.data.p_split),
        p_hit=float(config.data.p_hit),
        max_context_ratio=float(config.data.max_context_ratio),
        empty_A_context_skip_probability=float(
            config.data.empty_A_context_skip_probability
        ),
        receptor_radius=float(config.data.receptor_radius),
        O_radius=float(config.objective.O_radius),
        O_prime_scale=float(config.objective.O_prime_scale),
        O_prime_observed_radius=float(config.objective.O_prime_observed_radius),
        load_entity_auxiliary_labels=float(config.objective.entity_aux_weight) > 0,
        load_fine_pair_labels=phase == 2 and float(config.objective.fine_pair_aux_weight) > 0,
        A_node_feature_sources=tuple(str(value) for value in config.data.A_node_feature_sources),
        graph_radius=float(config.data.graph_radius),
        graph_max_radius_neighbors=int(config.data.graph_max_radius_neighbors),
        graph_rbf_bins=int(config.data.graph_rbf_bins),
    )


def _model(config: DictConfig) -> Matcher:
    model = config.model
    if str(config.evaluation.decoder_name) != "O_O_prime":
        raise ValueError(f"当前路线不支持 decoder：{config.evaluation.decoder_name}")
    if int(model.edge_input_dim) != 15 + int(config.data.graph_rbf_bins):
        raise ValueError("model.edge_input_dim 必须等于 15 + data.graph_rbf_bins。")
    return Matcher(
        phase=int(config.run.phase),
        A_node_input_dim=int(model.A_node_input_dim),
        edge_input_dim=int(model.edge_input_dim),
        node_dim=int(model.node_dim),
        edge_dim=int(model.edge_dim),
        repr_dim=int(model.repr_dim),
        num_heads=int(model.num_heads),
        coarse_layers_per_block=int(model.coarse_layers_per_block),
        coarse_ffn_dim=int(model.coarse_ffn_dim),
        dropout=float(model.dropout),
        mode=str(model.mode),
        phase1_blocks=int(model.phase1_blocks),
        phase2_blocks=int(model.phase2_blocks),
        map_channels=tuple(int(value) for value in model.map_channels),
        map_candidate_chunk_size=(
            int(model.map_candidate_chunk_size)
            if model.map_candidate_chunk_size is not None
            else None
        ),
        map_input_channels=int(model.map_input_channels),
        map_input_shape_zyx=tuple(int(value) for value in model.map_input_shape_zyx),
        map_norm_groups=int(model.map_norm_groups),
        map_bottleneck_heads=int(model.map_bottleneck_heads),
        map_bottleneck_layers=int(model.map_bottleneck_layers),
        map_bottleneck_ffn_dim=int(model.map_bottleneck_ffn_dim),
        map_activation_checkpoint=bool(model.map_activation_checkpoint),
        map_scale_summary_dim=int(model.map_scale_summary_dim),
        map_summary_hidden_dim=int(model.map_summary_hidden_dim),
        candidate_bias_rbf_bins=int(model.candidate_bias_rbf_bins),
        candidate_bias_hidden_dim=int(model.candidate_bias_hidden_dim),
        candidate_bias_distance_scale=float(model.candidate_bias_distance_scale),
        O_initial_prior=float(model.O_initial_prior),
        O_prime_initial_value=float(model.O_prime_initial_value),
        graph_update_edge=bool(model.graph_update_edge),
        graph_activation_checkpoint=bool(model.graph_activation_checkpoint),
        phase2_use_FiLM_plus=bool(model.phase2_use_FiLM_plus),
        phase2_condition_mlp_ratio=float(model.phase2_condition_mlp_ratio),
        phase2_condition_dropout=float(model.phase2_condition_dropout),
        entity_aux_weight=float(config.objective.entity_aux_weight),
        fine_pair_aux_weight=float(config.objective.fine_pair_aux_weight),
        fine_pair_chunk_size=(
            int(model.fine_pair_chunk_size) if model.fine_pair_chunk_size is not None else None
        ),
        fine_pair_activation_checkpoint=bool(model.fine_pair_activation_checkpoint),
        fine_attention_use_ffn=bool(model.fine_attention_use_ffn),
        fine_attention_ffn_hidden_dim=int(model.fine_attention_ffn_hidden_dim),
        fine_readout_dim=int(model.fine_readout_dim),
    )


def _load_phase1_for_phase2(model: Matcher, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = [
        "phase2_entity_stem.",
        "ligand_FiLM_plus.",
        "A_FiLM_plus.",
        "fine_pair_branch.",
        "fine_adapter.",
    ]
    for block_index in range(model.phase1_blocks, model.num_blocks):
        allowed_missing.extend(
            f"blocks.{block_index}.{name}."
            for name in ("ligand_gnn_layer", "A_gnn_layer", "CCD_readback", "A_readback")
        )
    unexpected_missing = [
        key for key in incompatible.missing_keys if not key.startswith(tuple(allowed_missing))
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Phase2 checkpoint mismatch: missing={unexpected_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.freeze_phase1_for_phase2()


def _optimizer(model: Matcher, config: DictConfig) -> AdamW:
    if str(config.optimizer.name) != "AdamW":
        raise ValueError(f"当前训练入口不支持 optimizer：{config.optimizer.name}")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return AdamW(
        parameters,
        lr=float(config.optimizer.lr),
        betas=tuple(float(value) for value in config.optimizer.betas),
        eps=float(config.optimizer.eps),
        weight_decay=float(config.optimizer.weight_decay),
    )


def _scheduler(optimizer: AdamW, config: DictConfig, expected_steps: int) -> WarmupPlateau:
    scheduler = config.scheduler
    expected_contract = {
        "name": "warmup_plateau",
        "monitor": "val/F1_O_O_prime",
        "mode": "max",
        "threshold_mode": "abs",
    }
    for key, expected in expected_contract.items():
        if str(scheduler[key]) != expected:
            raise ValueError(f"scheduler.{key} 必须为 {expected}。")
    explicit = scheduler.get("warmup_steps")
    warmup_steps = int(explicit) if explicit is not None else max(
        1, round(float(scheduler.warmup_ratio) * expected_steps)
    )
    return WarmupPlateau(
        optimizer,
        warmup_steps=warmup_steps,
        warmup_start_factor=float(scheduler.warmup_start_factor),
        factor=float(scheduler.factor),
        patience=int(scheduler.patience),
        threshold=float(scheduler.threshold),
        stop_after_lr_reductions=int(scheduler.stop_after_lr_reductions),
        cooldown=int(scheduler.cooldown),
        min_lr=float(scheduler.min_lr),
        eps=float(scheduler.eps),
    )


def _loss(
    outputs: tuple[PDBMatcherOutput, ...],
    samples: tuple[AnchorSample, ...],
    config: DictConfig,
) -> MatcherLoss:
    return matcher_loss(
        outputs,
        samples,
        phase=int(config.run.phase),
        phase1_blocks=int(config.model.phase1_blocks),
        occurrence_budget_per_batch=int(config.data.occurrence_budget_per_batch),
        coarse_deep_total_weight=float(config.objective.coarse_deep_total_weight),
        coarse_deep_schedule=str(config.objective.coarse_deep_schedule),
        lambda_O=float(config.objective.lambda_O),
        lambda_O_prime=float(config.objective.lambda_O_prime),
        entity_aux_weight=float(config.objective.entity_aux_weight),
        fine_pair_aux_weight=float(config.objective.fine_pair_aux_weight),
        O_prime_truncation_value=float(config.objective.O_prime_truncation_value),
    )


@torch.no_grad()
def validate(
    model: Matcher,
    loader: DataLoader,
    config: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    """累计完整验证集损失、冻结语义的 O/O′ 指标与一致性诊断。"""

    model.eval()
    evaluation = OOPrimeEvaluation(float(config.evaluation.threshold_step))
    total_loss = 0.0
    equal_count = 0
    consistency_count = 0
    for batch in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and bool(config.run.bf16),
        ):
            outputs = model(batch)
        loss = _loss(outputs, batch.samples, config)
        total_loss += float(loss.loss_total)
        for output, sample, assignment in zip(
            outputs, batch.samples, loss.final_assignment, strict=True
        ):
            final = output.block_outputs[model.num_blocks - 1]
            evaluation.update(final.O_logit.float(), final.O_prime.float(), sample.O_target, assignment)
            consistency = O_O_prime_consistency(
                final.O_logit,
                final.O_prime,
                O_radius=float(config.objective.O_radius),
                O_prime_scale=float(config.objective.O_prime_scale),
            )
            equal_count += consistency.equal_count
            consistency_count += consistency.total_count
    metrics = evaluation.best()
    return {
        "val/loss_total": total_loss,
        "val/precision_O_O_prime": metrics.precision,
        "val/recall_O_O_prime": metrics.recall,
        "val/F1_O_O_prime": metrics.F1,
        "val/decode_O_threshold": metrics.threshold,
        "val/O_O_prime_consistency": equal_count / consistency_count,
    }


def _random_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    try:
        torch.save(value, temporary_name)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_json_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _save_checkpoint(
    path: Path,
    *,
    model: Matcher,
    optimizer: AdamW,
    scheduler: WarmupPlateau,
    epoch: int,
    batch_index_in_epoch: int,
    global_step: int,
    metrics: dict[str, float],
    decoder_threshold: float,
    best_F1: float,
    best_decoder_threshold: float,
    config: DictConfig,
) -> None:
    _atomic_torch_save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "batch_index_in_epoch": batch_index_in_epoch,
            "global_step": global_step,
            "metrics": metrics,
            "decoder": {"name": "O_O_prime", "O_threshold": decoder_threshold},
            "best": {
                "F1_O_O_prime": best_F1,
                "decoder_threshold": best_decoder_threshold,
            },
            "random_state": _random_state(),
            "config": OmegaConf.to_container(config, resolve=True),
        },
        path,
    )


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _start_wandb(config: DictConfig, output_dir: Path) -> Any | None:
    """为正式训练建立在线 W&B run；未配置的画像与测试入口保持无网络依赖。"""

    wandb_config = config.get("wandb")
    if wandb_config is None or not bool(wandb_config.enabled):
        return None
    import wandb

    run = wandb.init(
        project=str(wandb_config.project),
        group=str(wandb_config.group),
        name=str(wandb_config.name),
        mode=str(wandb_config.mode),
        dir=output_dir.as_posix(),
        config=OmegaConf.to_container(config, resolve=True),
        resume="never",
    )
    run.define_metric("global_step")
    run.define_metric("*", step_metric="global_step")
    return run


def run_training(config: DictConfig) -> None:
    """执行一段显式 Phase1 或 Phase2 单卡训练。"""

    seed = int(config.run.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(str(config.run.device))
    num_workers = int(config.data.num_workers)
    if num_workers > 0:
        # AnchorBatch 包含较多独立张量；文件系统共享避免 worker 长时间耗尽文件描述符。
        torch.multiprocessing.set_sharing_strategy("file_system")
    output_dir = Path(config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "resolved_config.yaml", resolve=True)
    events_path = output_dir / "events.jsonl"
    wandb_run = _start_wandb(config, output_dir)
    wandb_log_every_n_steps = int(config.wandb.log_every_n_steps) if wandb_run else 0
    # 第三方在线初始化完成后重置随机数，避免它影响 Dataset、模型和优化器的正式 seed。
    _seed_everything(seed)

    train_dataset = AnchorPocketDataset(_data_config(config, "train", True))
    validation_dataset = AnchorPocketDataset(_data_config(config, "validation", False))
    train_dataset.set_epoch(0)
    prepared_train_epoch = 0
    train_sampler = OccurrenceBudgetBatchSampler(
        [entry["num_occurrences"] for entry in train_dataset.entries],
        budget=int(config.data.occurrence_budget_per_batch),
        seed=seed,
        shuffle=True,
        indices=train_dataset.available_indices,
    )
    validation_sampler = OccurrenceBudgetBatchSampler(
        [entry["num_occurrences"] for entry in validation_dataset.entries],
        budget=int(config.data.occurrence_budget_per_batch),
        seed=seed,
        shuffle=False,
    )
    loader_options = {
        "num_workers": num_workers,
        "collate_fn": collate_anchor_batch,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        generator=torch.Generator().manual_seed(seed),
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        generator=torch.Generator().manual_seed(seed + 1),
        **loader_options,
    )

    model = _model(config)
    if int(config.run.phase) == 2:
        _load_phase1_for_phase2(model, Path(config.run.phase1_checkpoint))
    model.to(device)
    optimizer = _optimizer(model, config)
    max_epochs = int(config.run.max_epochs)
    scheduler = _scheduler(optimizer, config, len(train_loader) * max_epochs)

    start_epoch = 0
    completed_batch_index = 0
    global_step = 0
    best_F1 = -1.0
    best_decoder_threshold = float(config.evaluation.fallback_O_threshold)
    resume_path = config.run.get("resume_checkpoint")
    if resume_path:
        checkpoint_value = torch.load(Path(resume_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_value["model"], strict=True)
        optimizer.load_state_dict(checkpoint_value["optimizer"])
        scheduler.load_state_dict(checkpoint_value["scheduler"])
        start_epoch = int(checkpoint_value["epoch"])
        completed_batch_index = int(checkpoint_value["batch_index_in_epoch"])
        global_step = int(checkpoint_value["global_step"])
        best_F1 = float(checkpoint_value["best"]["F1_O_O_prime"])
        best_decoder_threshold = float(checkpoint_value["best"]["decoder_threshold"])
        _restore_random_state(checkpoint_value["random_state"])
        if scheduler.should_stop:
            _append_jsonl(
                events_path,
                {"event": "resume_already_stopped", "global_step": global_step},
            )
            if wandb_run:
                wandb_run.finish()
            return

    for epoch in range(start_epoch, max_epochs):
        if epoch != prepared_train_epoch:
            train_dataset.set_epoch(epoch)
            prepared_train_epoch = epoch
        train_sampler.set_epoch(epoch)
        train_sampler.set_indices(train_dataset.available_indices)
        candidate_record = {
            "event": "candidate_sampling",
            "epoch": epoch,
            **train_dataset.candidate_audit,
        }
        _append_jsonl(events_path, candidate_record)
        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    **{
                        f"data/{name}": value
                        for name, value in train_dataset.candidate_audit.items()
                    },
                }
            )
        validations_per_epoch = int(config.run.val_per_epoch)
        if len(train_loader) < validations_per_epoch:
            raise ValueError(
                f"每 epoch 只有 {len(train_loader)} 个 optimizer step，"
                f"无法执行 {validations_per_epoch} 次互不重合的验证。"
            )
        validation_steps = [
            math.ceil(position * len(train_loader) / validations_per_epoch)
            for position in range(1, validations_per_epoch + 1)
        ]
        model.train()
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < completed_batch_index:
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and bool(config.run.bf16),
            ):
                outputs = model(batch)
                loss = _loss(outputs, batch.samples, config)
            loss.loss_total.backward()
            clip_grad_norm_(model.parameters(), float(config.optimizer.gradient_clip_norm))
            optimizer.step()
            scheduler.step_optimizer()
            global_step += 1
            train_record = {
                "event": "train",
                "epoch": epoch,
                "batch_index_in_epoch": batch_index,
                "global_step": global_step,
                "loss_total": float(loss.loss_total),
                "loss_O": float(loss.loss_O),
                "loss_O_prime": float(loss.loss_O_prime),
                "loss_aux": float(loss.loss_aux),
                "lr": optimizer.param_groups[0]["lr"],
            }
            _append_jsonl(events_path, train_record)
            if wandb_run and global_step % wandb_log_every_n_steps == 0:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "batch_index_in_epoch": batch_index,
                        "train/loss_total": train_record["loss_total"],
                        "train/loss_O": train_record["loss_O"],
                        "train/loss_O_prime": train_record["loss_O_prime"],
                        "train/loss_aux": train_record["loss_aux"],
                        "train/lr": train_record["lr"],
                    }
                )

            if batch_index + 1 not in validation_steps:
                continue
            metrics = validate(model, validation_loader, config, device)
            decoder_threshold = metrics["val/decode_O_threshold"]
            scheduler.step_validation(metrics["val/F1_O_O_prime"])
            is_best = metrics["val/F1_O_O_prime"] > best_F1
            if is_best:
                best_F1 = metrics["val/F1_O_O_prime"]
                best_decoder_threshold = decoder_threshold
            checkpoint_path = output_dir / "checkpoints" / f"checkpoint_step_{global_step:08d}.pt"
            _save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                batch_index_in_epoch=batch_index + 1,
                global_step=global_step,
                metrics=metrics,
                decoder_threshold=decoder_threshold,
                best_F1=best_F1,
                best_decoder_threshold=best_decoder_threshold,
                config=config,
            )
            if is_best:
                _atomic_json_save(
                    {
                        "checkpoint": checkpoint_path.as_posix(),
                        "global_step": global_step,
                        "F1_O_O_prime": best_F1,
                        "decoder": {
                            "name": "O_O_prime",
                            "O_threshold": best_decoder_threshold,
                        },
                    },
                    output_dir / "BEST.json",
                )
            validation_record = {
                "event": "validation",
                "epoch": epoch,
                "global_step": global_step,
                **metrics,
                "lr_reduction_count": scheduler.reduction_count,
                "checkpoint": checkpoint_path.as_posix(),
            }
            _append_jsonl(events_path, validation_record)
            if wandb_run:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        **metrics,
                        "val/lr_reduction_count": scheduler.reduction_count,
                        "val/best_F1_O_O_prime": best_F1,
                    }
                )
            print(json.dumps({"global_step": global_step, **metrics}, ensure_ascii=False))
            model.train()
            if scheduler.should_stop:
                if wandb_run:
                    wandb_run.finish()
                return
        completed_batch_index = 0
    if wandb_run:
        wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 Anchor O/O′ Matcher。")
    parser.add_argument("--config", type=Path, required=True, help="Phase1 或 Phase2 YAML。")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf 点号覆盖，例如 model.map_channels=[24,48,72,96]。",
    )
    args = parser.parse_args()
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    run_training(config)


if __name__ == "__main__":
    main()
