"""在真实训练数据流上限时测量 Matcher Phase2 的平均单步耗时。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from matcher.anchor_data import AnchorPocketDataset
from matcher.batching import OccurrenceBudgetBatchSampler, collate_anchor_batch
from matcher.train import _data_config, _load_phase1_for_phase2, _loss, _model, _optimizer
from ops.profile_matcher_memory import _write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在普通训练 Dataset 上运行指定分钟数，记录端到端平均 batch 时间。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python -m ops.profile_matcher_throughput --config configs/matcher/anchor_O_O_prime_phase2.yaml --channels 24,48,72,96 --minutes 60 --output tmp/matcher_profile/normal_throughput.json
""",
    )
    parser.add_argument("--config", type=Path, required=True, help="Phase2 profile YAML。")
    parser.add_argument(
        "--channels",
        required=True,
        help="四个逗号分隔 U-Net 通道，例如 24,48,72,96。",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=60.0,
        help="测量时长，默认 60 分钟；最后一个训练步允许越过截止时刻完成。",
    )
    parser.add_argument(
        "--phase1-checkpoint",
        type=Path,
        default=None,
        help="可选：与待测通道相同的 Phase1 checkpoint；省略时使用随机初始化权重。",
    )
    parser.add_argument("--output", type=Path, required=True, help="吞吐画像 JSON 输出路径。")
    parser.add_argument(
        "--memory-limit-gib",
        type=float,
        default=None,
        help="可选人工显存上限；80 GiB 显卡的 90%% 为 72。",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="可选 OmegaConf 点号覆盖，例如 model.fine_pair_chunk_size=1024。",
    )
    args = parser.parse_args()
    if args.minutes <= 0:
        parser.error("--minutes 必须大于 0。")

    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    config.model.map_channels = [int(value) for value in args.channels.split(",")]
    if int(config.run.phase) != 2:
        parser.error("吞吐画像只接受 run.phase=2 的配置。")
    config.run.phase1_checkpoint = (
        str(args.phase1_checkpoint) if args.phase1_checkpoint is not None else None
    )
    seed = int(config.run.seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    num_workers = int(config.data.num_workers)
    if num_workers > 0:
        # 与正式训练一致：避免长时间传递大量小张量时耗尽文件描述符。
        torch.multiprocessing.set_sharing_strategy("file_system")

    dataset = AnchorPocketDataset(_data_config(config, "train", True))
    dataset.set_epoch(0)
    sampler = OccurrenceBudgetBatchSampler(
        [entry["num_occurrences"] for entry in dataset.entries],
        budget=int(config.data.occurrence_budget_per_batch),
        seed=seed,
        shuffle=True,
        indices=dataset.available_indices,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_anchor_batch,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )
    iterator = iter(loader)

    device = torch.device("cuda")
    status = "ok"
    oom_stage = None
    error = None
    measured_steps = 0
    measured_pdbs = 0
    measured_occurrences = 0
    data_loading_seconds = 0.0
    optimization_seconds = 0.0
    measurement_seconds = 0.0
    last_pdb_ids: list[str] = []
    epoch = 0
    stage = "model_setup"
    measurement_started = None
    try:
        model = _model(config)
        if args.phase1_checkpoint is None:
            model.freeze_phase1_for_phase2()
        else:
            _load_phase1_for_phase2(model, args.phase1_checkpoint)
        model.to(device).train()
        optimizer = _optimizer(model, config)

        def run_step(batch) -> None:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=bool(config.run.bf16)
            ):
                outputs = model(batch)
                loss = _loss(outputs, batch.samples, config)
            loss.loss_total.backward()
            clip_grad_norm_(model.parameters(), float(config.optimizer.gradient_clip_norm))
            optimizer.step()

        stage = "warmup_data"
        torch.cuda.reset_peak_memory_stats(device)
        warmup_batch = next(iterator)
        stage = "warmup_step"
        run_step(warmup_batch)
        torch.cuda.synchronize(device)

        stage = "measurement"
        torch.cuda.reset_peak_memory_stats(device)
        measurement_started = time.perf_counter()
        deadline = measurement_started + args.minutes * 60.0
        while time.perf_counter() < deadline:
            data_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                dataset.set_epoch(epoch)
                sampler.set_epoch(epoch)
                sampler.set_indices(dataset.available_indices)
                iterator = iter(loader)
                batch = next(iterator)
            data_loading_seconds += time.perf_counter() - data_started

            optimization_started = time.perf_counter()
            run_step(batch)
            torch.cuda.synchronize(device)
            optimization_seconds += time.perf_counter() - optimization_started
            measured_steps += 1
            measured_pdbs += len(batch.samples)
            measured_occurrences += sum(sample.num_occurrences for sample in batch.samples)
            last_pdb_ids = [sample.pdb_id for sample in batch.samples]
        measurement_seconds = time.perf_counter() - measurement_started
    except torch.cuda.OutOfMemoryError as exception:
        status = "oom"
        oom_stage = stage
        error = str(exception)
        if stage == "measurement" and measurement_started is not None:
            measurement_seconds = time.perf_counter() - measurement_started

    reserved_gib = torch.cuda.max_memory_reserved(device) / 1024**3
    allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    result = {
        "status": status,
        "oom_stage": oom_stage,
        "error": error,
        "GPU": torch.cuda.get_device_name(device),
        "GPU_total_gib": torch.cuda.get_device_properties(device).total_memory / 1024**3,
        "numeric_precision": "bfloat16_autocast" if bool(config.run.bf16) else "float32",
        "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "duration_minutes_requested": args.minutes,
        "measurement_seconds": measurement_seconds,
        "measured_steps": measured_steps,
        "mean_batch_wall_seconds": (
            measurement_seconds / measured_steps if measured_steps else None
        ),
        "mean_optimization_step_seconds": (
            optimization_seconds / measured_steps if measured_steps else None
        ),
        "mean_data_loading_seconds": (
            data_loading_seconds / measured_steps if measured_steps else None
        ),
        "mean_wall_seconds_per_pdb": (
            measurement_seconds / measured_pdbs if measured_pdbs else None
        ),
        "mean_optimization_seconds_per_pdb": (
            optimization_seconds / measured_pdbs if measured_pdbs else None
        ),
        "mean_wall_seconds_per_occurrence": (
            measurement_seconds / measured_occurrences if measured_occurrences else None
        ),
        "measured_pdbs": measured_pdbs,
        "measured_occurrences": measured_occurrences,
        "max_memory_allocated_gib": allocated_gib,
        "max_memory_reserved_gib": reserved_gib,
        "memory_limit_gib": args.memory_limit_gib,
        "within_memory_limit": (
            status == "ok" and reserved_gib <= args.memory_limit_gib
            if args.memory_limit_gib is not None
            else None
        ),
        "map_channels": list(config.model.map_channels),
        "occurrence_budget_per_batch": int(config.data.occurrence_budget_per_batch),
        "map_candidate_chunk_size": config.model.map_candidate_chunk_size,
        "fine_pair_chunk_size": config.model.fine_pair_chunk_size,
        "phase1_checkpoint": (
            str(args.phase1_checkpoint) if args.phase1_checkpoint is not None else None
        ),
        "last_pdb_ids": last_pdb_ids,
    }
    _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    if status == "oom":
        torch.cuda.empty_cache()
        sys.exit(1)


if __name__ == "__main__":
    main()
