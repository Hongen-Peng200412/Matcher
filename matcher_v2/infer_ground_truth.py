"""模式一独立推理与评估；只使用 checkpoint 内冻结的验证阈值。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .decode_ground_truth import decode_pairs, evaluate, pair_score, prepare_evaluation
from .train import _atomic_json, build_ground_truth_loader, build_matcher, load_checkpoint


@torch.no_grad()
def run(config_path: Path, checkpoint_path: Path, split: str, output_dir: Path) -> None:
    config = OmegaConf.load(config_path)
    model = build_matcher(config)
    checkpoint = load_checkpoint(model, checkpoint_path)
    model.to(torch.device(str(config.run.device))).eval()
    _, loader = build_ground_truth_loader(config, split, False)
    threshold = float(checkpoint["validation"]["threshold"])
    records = []
    predictions = []
    device = next(model.parameters()).device
    for batch in loader:
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=bool(config.run.bf16)
        ):
            outputs = model(batch)
        for sample, output in zip(batch.samples, outputs, strict=True):
            head = output.block_outputs[-1]
            score = pair_score(head)
            pairs = decode_pairs(score, threshold)
            records.append(prepare_evaluation(head, sample))
            predictions.append(
                {
                    "pdb_id": sample.pdb_id,
                    "threshold": threshold,
                    "pairs": [asdict(pair) for pair in pairs],
                }
            )
    metrics = evaluate(
        tuple(records), threshold=threshold, step=float(config.evaluation.threshold_step)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".predictions.", dir=output_dir)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in predictions:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, output_dir / "predictions.jsonl")
    finally:
        Path(temporary).unlink(missing_ok=True)
    _atomic_json(output_dir / "metrics.json", asdict(metrics))


def main() -> None:
    parser = argparse.ArgumentParser(description="Matcher v2 模式一推理与评估。")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "calibration"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.split, args.output_dir)


if __name__ == "__main__":
    main()
