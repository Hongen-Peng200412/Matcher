"""用一个真实 Matcher batch 测量完整 forward/loss/backward/AdamW step 峰值显存。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from matcher.anchor_data import AnchorPocketDataset
from matcher.batching import OccurrenceBudgetBatchSampler, collate_anchor_batch
from matcher.train import _data_config, _load_phase1_for_phase2, _loss, _model, _optimizer


def _write_json(path: Path, value: dict) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="测量完整 Phase2 真实训练步显存；脚本不自动搜索或修改配置。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python -m ops.profile_matcher_memory --config configs/matcher/anchor_O_O_prime_phase2.yaml --channels 40,80,120,160 --output tmp/matcher_profile/memory_profile.json
  python -m ops.profile_matcher_memory --config configs/matcher/anchor_O_O_prime_phase2.yaml --channels 40,80,120,160 --memory-limit-gib 72 --output tmp/matcher_profile/memory_profile.json
""",
    )
    parser.add_argument("--config", type=Path, required=True, help="Phase2 profile YAML。")
    parser.add_argument(
        "--experiment-manifest",
        type=Path,
        default=None,
        help="可选：已冻结 candidate_start_zyx 的画像清单；省略时准备正式训练清单的 epoch 0。",
    )
    parser.add_argument(
        "--channels",
        required=True,
        help="四个逗号分隔 U-Net 通道，例如 16,32,48,64。",
    )
    parser.add_argument(
        "--phase1-checkpoint",
        type=Path,
        default=None,
        help="可选：与待测通道相同的 Phase1 checkpoint；省略时使用随机初始化权重。",
    )
    parser.add_argument("--output", type=Path, required=True, help="memory_profile.json 输出路径。")
    parser.add_argument(
        "--memory-limit-gib",
        type=float,
        default=None,
        help="可选人工显存上限；A800 80 GiB 的 90%% 示例为 72。",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="可选 OmegaConf 点号覆盖，例如 model.fine_pair_chunk_size=1024。",
    )
    args = parser.parse_args()

    preparation_started = time.perf_counter()
    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    config.model.map_channels = [int(value) for value in args.channels.split(",")]
    if args.experiment_manifest is not None:
        config.data.experiment_manifest = str(args.experiment_manifest)
    config.run.phase1_checkpoint = (
        str(args.phase1_checkpoint) if args.phase1_checkpoint is not None else None
    )
    fixed_candidates = args.experiment_manifest is not None
    dataset = AnchorPocketDataset(_data_config(config, "train", not fixed_candidates))
    if not fixed_candidates:
        dataset.set_epoch(0)
    sampler = OccurrenceBudgetBatchSampler(
        [entry["num_occurrences"] for entry in dataset.entries],
        budget=int(config.data.occurrence_budget_per_batch),
        shuffle=False,
        indices=dataset.available_indices,
    )
    indices = next(iter(sampler))
    samples = [dataset[index] for index in indices]
    batch = collate_anchor_batch(samples)
    data_preparation_seconds = time.perf_counter() - preparation_started

    device = torch.device("cuda")
    status = "ok"
    oom_stage = None
    error = None
    optimization_step_seconds = None
    stage = "model_setup"
    try:
        model = _model(config)
        if args.phase1_checkpoint is None:
            model.freeze_phase1_for_phase2()
        else:
            _load_phase1_for_phase2(model, args.phase1_checkpoint)
        model.to(device).train()
        optimizer = _optimizer(model, config)

        def run_step() -> None:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(config.run.bf16)):
                outputs = model(batch)
                loss = _loss(outputs, batch.samples, config)
            loss.loss_total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.optimizer.gradient_clip_norm)
            )
            optimizer.step()

        stage = "warmup"
        torch.cuda.reset_peak_memory_stats(device)
        run_step()
        torch.cuda.synchronize(device)

        stage = "measurement"
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        optimization_started = time.perf_counter()
        run_step()
        torch.cuda.synchronize(device)
        optimization_step_seconds = time.perf_counter() - optimization_started
    except torch.cuda.OutOfMemoryError as exception:
        status = "oom"
        oom_stage = stage
        error = str(exception)

    reserved_gib = torch.cuda.max_memory_reserved(device) / 1024**3
    allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    if status == "oom":
        torch.cuda.empty_cache()
    result = {
        "status": status,
        "oom_stage": oom_stage,
        "error": error,
        "GPU": torch.cuda.get_device_name(device),
        "GPU_total_gib": torch.cuda.get_device_properties(device).total_memory / 1024**3,
        "numeric_precision": "bfloat16_autocast" if bool(config.run.bf16) else "float32",
        "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "max_memory_allocated_gib": allocated_gib,
        "max_memory_reserved_gib": reserved_gib,
        "memory_limit_gib": args.memory_limit_gib,
        "phase1_checkpoint": (
            str(args.phase1_checkpoint) if args.phase1_checkpoint is not None else None
        ),
        "experiment_manifest": str(config.data.experiment_manifest),
        "data_preparation_seconds": data_preparation_seconds,
        "optimization_step_seconds": optimization_step_seconds,
        "within_memory_limit": (
            status == "ok" and reserved_gib <= args.memory_limit_gib
            if args.memory_limit_gib is not None
            else None
        ),
        "map_channels": list(config.model.map_channels),
        "occurrence_budget_per_batch": int(config.data.occurrence_budget_per_batch),
        "map_candidate_chunk_size": config.model.map_candidate_chunk_size,
        "fine_pair_chunk_size": config.model.fine_pair_chunk_size,
        "graph_activation_checkpoint": bool(config.model.graph_activation_checkpoint),
        "batch_indices": indices,
        "pdb_ids": [sample.pdb_id for sample in batch.samples],
        "num_occurrences": sum(sample.num_occurrences for sample in batch.samples),
        "num_candidates": sum(sample.num_candidates for sample in batch.samples),
        "num_A_atoms": sum(
            candidate.A_graph.node_input.shape[0]
            for sample in batch.samples
            for candidate in sample.candidates
        ),
    }
    _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    if status == "oom":
        sys.exit(1)


if __name__ == "__main__":
    main()
