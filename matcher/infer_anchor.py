"""当前 Anchor O/O′ 路线的独立推理与可选 validation 评估入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .anchor_data import AnchorPocketDataset
from .batching import OccurrenceBudgetBatchSampler, collate_anchor_batch
from .objectives import identity_hungarian_assignment
from .oo_prime import OOPrimeDecoder, OOPrimeEvaluation
from .train import _atomic_json_save, _data_config, _model


def main() -> None:
    parser = argparse.ArgumentParser(description="运行当前 Anchor O/O′ Matcher 推理。")
    parser.add_argument("--config", type=Path, required=True, help="与 checkpoint 同构的固定 YAML。")
    parser.add_argument("--checkpoint", type=Path, required=True, help="带 decoder 阈值元数据的 checkpoint。")
    parser.add_argument("--output-dir", type=Path, required=True, help="正式预测与评估输出目录。")
    parser.add_argument("--split", default="validation", choices=("validation",))
    parser.add_argument("--evaluate", action="store_true", help="使用真实 O 标签累计当前路线 TP/P/G。")
    parser.add_argument("overrides", nargs="*", help="OmegaConf 点号覆盖。")
    args = parser.parse_args()

    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    device = torch.device(str(config.run.device))
    num_workers = int(config.data.num_workers)
    if num_workers > 0:
        # AnchorBatch 包含较多独立张量；文件系统共享避免 worker 耗尽文件描述符。
        torch.multiprocessing.set_sharing_strategy("file_system")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = _model(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    decoder = OOPrimeDecoder(float(checkpoint["decoder"]["O_threshold"]))
    evaluation = (
        OOPrimeEvaluation(None, frozen_threshold=decoder.O_threshold)
        if args.evaluate
        else None
    )

    dataset = AnchorPocketDataset(_data_config(config, args.split, False))
    sampler = OccurrenceBudgetBatchSampler(
        [entry["num_occurrences"] for entry in dataset.entries],
        budget=int(config.data.occurrence_budget_per_batch),
        seed=int(config.run.seed),
        shuffle=False,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_anchor_batch,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
        with torch.no_grad():
            for batch in loader:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda" and bool(config.run.bf16),
                ):
                    outputs = model(batch)
                for sample, output in zip(batch.samples, outputs, strict=True):
                    final = output.block_outputs[model.num_blocks - 1]
                    decoded = decoder.decode(final.O_logit, final.O_prime)
                    handle.write(
                        json.dumps(
                            {
                                "pdb_id": sample.pdb_id,
                                "selected_pairs": [pair.__dict__ for pair in decoded.selected_pairs],
                                "unmatched_slot_indices": decoded.unmatched_slot_indices,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    if evaluation is not None:
                        assignment = identity_hungarian_assignment(final, sample)
                        evaluation.update(
                            final.O_logit, final.O_prime, sample.O_target, assignment
                        )

    if evaluation is not None:
        metrics = evaluation.best()
        _atomic_json_save(metrics.__dict__, args.output_dir / "evaluation.json")


if __name__ == "__main__":
    main()
