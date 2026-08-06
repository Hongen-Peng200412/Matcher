"""Matcher v2 模式一的单卡 Phase1/Phase2 正式训练入口。"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from matcher.train import WarmupPlateau

from .batching import OccurrenceBudgetBatchSampler, collate_matcher_batch
from .data_ground_truth import GroundTruthDataConfig, GroundTruthMatcherDataset
from .data_stage1 import Stage1DataConfig, Stage1MatcherDataset
from .decode_ground_truth import MatcherMetrics, prepare_evaluation, select_threshold
from .model import Matcher
from .objectives import matcher_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_ground_truth_dataset(config: DictConfig, split: str, training: bool) -> GroundTruthMatcherDataset:
    """公开的模式一 Dataset 装配入口，供训练、推理和画像复用。"""

    data = config.data
    return GroundTruthMatcherDataset(
        GroundTruthDataConfig(
            data_root=Path(data.data_root),
            manifest_path=Path(data.manifest),
            split=split,
            training=training,
            seed=int(config.run.seed),
            map_size=int(data.map_size),
            receptor_envelope_angstrom=float(data.receptor_envelope_angstrom),
            graph_radius=float(data.graph_radius),
            graph_max_radius_neighbors=int(data.graph_max_radius_neighbors),
            graph_rbf_bins=int(data.graph_rbf_bins),
            load_binding_labels=float(config.objective.entity_auxiliary_weight) > 0,
        )
    )


def build_stage1_dataset(config: DictConfig, split: str, training: bool) -> Stage1MatcherDataset:
    """公开的模式二 Dataset 装配入口；当前训练入口与模式一共用。"""

    del training
    data = config.data
    return Stage1MatcherDataset(
        Stage1DataConfig(
            data_root=Path(data.data_root),
            pointer_path=Path(data.pointer),
            split=split,
            map_size=int(data.map_size),
            PP_top_k=int(data.PP_top_k),
            V_feature_dim=int(config.model.stage1_V_feature_dim),
            P_feature_dim=int(config.model.stage1_P_feature_dim),
            A_feature_dim=int(config.model.stage1_A_feature_dim),
            graph_radius=float(data.graph_radius),
            graph_max_radius_neighbors=int(data.graph_max_radius_neighbors),
            graph_rbf_bins=int(data.graph_rbf_bins),
            load_binding_labels=float(config.objective.entity_auxiliary_weight) > 0,
        )
    )


def build_dataset(config: DictConfig, split: str, training: bool) -> Dataset:
    """只在此处选择两个已定义的数据契约，不建立多模态注册系统。"""

    mode = str(config.data.mode)
    if mode == "ground_truth_context":
        return build_ground_truth_dataset(config, split, training)
    if mode == "stage1_context":
        return build_stage1_dataset(config, split, training)
    raise ValueError(f"未知 Matcher 数据模式：{mode}")


def build_matcher(config: DictConfig) -> Matcher:
    """由一份 YAML 建立 Phase1 或 Phase2 模型。"""

    model = config.model
    edge_input_dim = 15 + int(config.data.graph_rbf_bins)
    return Matcher(
        phase=int(config.run.phase),
        A_node_input_dim=int(model.A_node_input_dim),
        edge_input_dim=edge_input_dim,
        node_dim=int(model.node_dim),
        edge_dim=int(model.edge_dim),
        repr_dim=int(model.repr_dim),
        num_heads=int(model.num_heads),
        coarse_layers_per_block=int(model.coarse_layers_per_block),
        coarse_ffn_dim=int(model.coarse_ffn_dim),
        dropout=float(model.dropout),
        mode="coupled_plus_modal",
        phase1_blocks=int(model.phase1_blocks),
        phase2_blocks=int(model.phase2_blocks),
        map_channels=tuple(int(value) for value in model.map_channels),
        map_candidate_chunk_size=int(model.map_candidate_chunk_size),
        map_input_channels=1,
        map_input_shape_zyx=(int(config.data.map_size),) * 3,
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
        graph_update_edge=True,
        graph_activation_checkpoint=bool(model.graph_activation_checkpoint),
        phase2_use_FiLM_plus=bool(model.phase2_use_FiLM_plus),
        phase2_condition_mlp_ratio=float(model.phase2_condition_mlp_ratio),
        phase2_condition_dropout=float(model.phase2_condition_dropout),
        entity_aux_weight=float(config.objective.entity_auxiliary_weight),
        fine_pair_aux_weight=float(config.objective.fine_pair_auxiliary_weight),
        fine_pair_chunk_size=int(model.fine_pair_chunk_size),
        fine_pair_activation_checkpoint=bool(model.fine_pair_activation_checkpoint),
        fine_attention_use_ffn=bool(model.fine_attention_use_ffn),
        fine_attention_ffn_hidden_dim=int(model.fine_attention_ffn_hidden_dim),
        fine_readout_dim=int(model.fine_readout_dim),
        V_feature_dim=int(model.stage1_V_feature_dim),
        P_feature_dim=int(model.stage1_P_feature_dim),
        A_feature_dim=int(model.stage1_A_feature_dim),
        initial_prior=float(model.initial_prior),
    )


def load_phase1(model: Matcher, checkpoint_path: Path) -> None:
    """把 Phase1 基础权重装入 Phase2，并冻结前四个 Block。"""

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    allowed = (
        "phase2_entity_stem.", "ligand_FiLM_plus.", "A_FiLM_plus.",
        "fine_pair_branch.", "fine_adapter.",
    )
    phase2_blocks = tuple(
        f"blocks.{index}.{name}."
        for index in range(model.phase1_blocks, model.num_blocks)
        for name in ("ligand_gnn_layer", "A_gnn_layer", "CCD_readback", "A_readback")
    )
    missing = [
        name for name in incompatible.missing_keys
        if not name.startswith((*allowed, *phase2_blocks))
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(f"Phase1→Phase2 权重不兼容：missing={missing}, unexpected={incompatible.unexpected_keys}")
    model.freeze_phase1_for_phase2()


def load_checkpoint(model: Matcher, path: Path) -> dict[str, Any]:
    """推理、评估和续训共用的 checkpoint 读取。"""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def _load_training_weights(
    model: Matcher,
    *,
    phase: int,
    resume_checkpoint: Path | None,
    phase1_checkpoint: Path | None,
) -> dict[str, Any] | None:
    """区分同阶段续训与 Phase1→Phase2 初始化，并恢复正确冻结边界。"""

    if resume_checkpoint is not None:
        state = load_checkpoint(model, resume_checkpoint)
        if phase == 2:
            model.freeze_phase1_for_phase2()
        return state
    if phase == 2 and phase1_checkpoint is not None:
        load_phase1(model, phase1_checkpoint)
    return None


def build_loader(config: DictConfig, split: str, training: bool) -> tuple[Dataset, DataLoader]:
    """训练、验证、推理共用的 occurrence 预算装箱入口。"""

    dataset = build_dataset(config, split, training)
    sampler = OccurrenceBudgetBatchSampler(
        dataset.occurrence_counts,
        budget=int(config.data.occurrence_budget_per_batch),
        seed=int(config.run.seed),
        shuffle=training,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_matcher_batch,
        num_workers=int(config.data.num_workers),
        persistent_workers=bool(config.data.num_workers),
        pin_memory=True,
    )
    return dataset, loader


def build_ground_truth_loader(
    config: DictConfig, split: str, training: bool
) -> tuple[GroundTruthMatcherDataset, DataLoader]:
    """模式一推理的显式边界，防止未来模式二后处理混入。"""

    if str(config.data.mode) != "ground_truth_context":
        raise ValueError("模式一推理只接受 ground_truth_context 配置。")
    dataset, loader = build_loader(config, split, training)
    return dataset, loader


@torch.no_grad()
def validate(
    model: Matcher, loader: DataLoader, *, threshold_step: float, use_bf16: bool
) -> MatcherMetrics:
    model.eval()
    records = []
    device = next(model.parameters()).device
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            output = model(batch)
        records.extend(
            prepare_evaluation(item.block_outputs[-1], sample)
            for item, sample in zip(output, batch.samples, strict=True)
        )
    return select_threshold(tuple(records), step=threshold_step)


def _atomic_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _scheduler(optimizer: AdamW, config: DictConfig, expected_steps: int) -> WarmupPlateau:
    scheduler = config.scheduler
    warmup_steps = scheduler.warmup_steps
    if warmup_steps is None:
        warmup_steps = max(1, round(float(scheduler.warmup_ratio) * expected_steps))
    return WarmupPlateau(
        optimizer,
        warmup_steps=int(warmup_steps),
        warmup_start_factor=float(scheduler.warmup_start_factor),
        factor=float(scheduler.factor),
        patience=int(scheduler.patience),
        threshold=float(scheduler.threshold),
        stop_after_lr_reductions=int(scheduler.stop_after_lr_reductions),
        min_lr=float(scheduler.min_lr),
    )


def _validation_batches(batch_count: int, validations: int) -> set[int]:
    """返回一个 epoch 内均匀分布且不重复的验证批次编号。"""

    count = min(batch_count, validations)
    return {math.ceil(index * batch_count / count) for index in range(1, count + 1)}


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _append_event(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _finish_run(output_dir: Path, *, reason: str, global_step: int, best_F1: float) -> None:
    value = {
        "time_unix": time.time(), "event": "completed", "reason": reason,
        "global_step": global_step, "best_F1": best_F1,
    }
    _append_event(output_dir / "events.jsonl", value)
    _atomic_json(output_dir / "COMPLETED.json", value)


def run_training(config: DictConfig, resume_checkpoint: Path | None = None) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    seed_everything(int(config.run.seed))
    device = torch.device(str(config.run.device))
    model = build_matcher(config)
    configured_resume = config.run.get("resume_checkpoint")
    resume_checkpoint = resume_checkpoint or (Path(configured_resume) if configured_resume else None)
    resume_state = _load_training_weights(
        model,
        phase=int(config.run.phase),
        resume_checkpoint=resume_checkpoint,
        phase1_checkpoint=(
            Path(config.run.phase1_checkpoint) if config.run.phase1_checkpoint else None
        ),
    )
    model.to(device)
    train_dataset, train_loader = build_loader(config, "train", True)
    _, validation_loader = build_loader(config, "validation", False)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config.optimizer.lr), weight_decay=float(config.optimizer.weight_decay),
        betas=tuple(float(value) for value in config.optimizer.betas),
        eps=float(config.optimizer.eps),
    )
    scheduler = _scheduler(optimizer, config, len(train_loader) * int(config.run.max_epochs))
    use_bf16 = bool(config.run.bf16)
    output_dir = Path(config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "resolved_config.yaml", resolve=True)

    wandb = None
    if bool(config.wandb.enabled):
        import wandb as wandb_module
        wandb = wandb_module.init(
            project=str(config.wandb.project), group=str(config.wandb.group),
            name=str(config.wandb.name), id=str(config.wandb.run_id), resume="allow",
            mode="online", config=OmegaConf.to_container(config, resolve=True),
        )

    seed_everything(int(config.run.seed))
    global_step = int(resume_state["global_step"]) if resume_state else 0
    best_F1 = float(resume_state.get("best_F1", -1.0)) if resume_state else -1.0
    start_epoch = int(resume_state["epoch"]) if resume_state else 0
    completed_batch = int(resume_state.get("batch_index", 0)) if resume_state else 0
    if resume_state:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        _restore_rng(resume_state["rng_state"])
    for epoch in range(start_epoch, int(config.run.max_epochs)):
        train_dataset.set_epoch(epoch)
        train_loader.batch_sampler.set_epoch(epoch)
        validation_batches = _validation_batches(len(train_loader), int(config.run.val_per_epoch))
        model.train()
        for batch_index, batch in enumerate(train_loader, start=1):
            if epoch == start_epoch and batch_index <= completed_batch:
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(batch)
                loss = matcher_loss(
                    outputs, batch.samples,
                    phase=int(config.run.phase),
                    phase1_blocks=int(config.model.phase1_blocks),
                    occurrence_budget_per_batch=int(config.data.occurrence_budget_per_batch),
                    deep_supervision_weight=float(config.objective.deep_supervision_weight),
                    continuous_weight=float(config.objective.continuous_weight),
                    hard_weight=float(config.objective.hard_weight),
                    entity_auxiliary_weight=float(config.objective.entity_auxiliary_weight),
                    fine_pair_auxiliary_weight=float(config.objective.fine_pair_auxiliary_weight),
                )
            loss.total.backward()
            clip_grad_norm_(model.parameters(), float(config.optimizer.gradient_clip_norm))
            optimizer.step()
            scheduler.step_optimizer()
            global_step += 1
            if wandb and global_step % int(config.wandb.log_every_n_steps) == 0:
                wandb.log(
                    {
                        "train/loss": float(loss.total),
                        "train/continuous": float(loss.continuous),
                        "train/hard": float(loss.hard),
                        "train/entity_auxiliary": float(loss.entity_auxiliary),
                        "train/fine_pair_auxiliary": float(loss.fine_pair_auxiliary),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )
            if batch_index not in validation_batches:
                continue
            metrics = validate(
                model, validation_loader,
                threshold_step=float(config.evaluation.threshold_step), use_bf16=use_bf16,
            )
            scheduler.step_validation(metrics.F1)
            state = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch,
                "batch_index": batch_index, "best_F1": max(best_F1, metrics.F1),
                "global_step": global_step, "validation": asdict(metrics),
                "rng_state": _rng_state(),
                "config": OmegaConf.to_container(config, resolve=True),
            }
            _atomic_checkpoint(output_dir / "latest.pt", state)
            if metrics.F1 > best_F1:
                best_F1 = metrics.F1
                _atomic_checkpoint(output_dir / "best.pt", state)
                _atomic_json(
                    output_dir / "BEST.json",
                    {"checkpoint": "best.pt", **asdict(metrics)},
                )
            event = {
                "time_unix": time.time(), "event": "validation", "phase": int(config.run.phase),
                "epoch": epoch, "batch_index": batch_index, "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"], **asdict(metrics),
            }
            _append_event(output_dir / "events.jsonl", event)
            print(json.dumps(event, ensure_ascii=False), flush=True)
            if wandb:
                wandb.log({f"validation/{key}": value for key, value in asdict(metrics).items()}, step=global_step)
            model.train()
            if scheduler.should_stop:
                _finish_run(
                    output_dir, reason="scheduler_stop",
                    global_step=global_step, best_F1=best_F1,
                )
                if wandb:
                    wandb.finish()
                return
        completed_batch = 0
    _finish_run(
        output_dir, reason="max_epochs", global_step=global_step, best_F1=best_F1
    )
    if wandb:
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 Matcher v2 的显式双模式模型。")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    run_training(OmegaConf.load(args.config), args.resume)


if __name__ == "__main__":
    main()
