"""核验一次性 calibration 入口的产物、绑定和重入边界。"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from matcher import calibration as calibration_module
from matcher.calibration import run_calibration
from matcher.experiment_config import load_experiment_plan
from matcher.json_artifacts import write_json
from matcher.metrics import EvaluationAccumulator, EvaluationBatch
from matcher.numeric import NumericBackendRecord
from matcher.schema import SourceRoots
from matcher.training import EvaluationResult


class _CalibrationDatasetStub:
    """只提供正式入口构造采样器与记录候选数所需的最小 Dataset 契约。"""

    def __init__(self, **values: object) -> None:
        if values.get("split_name") != "calibration":
            raise AssertionError("一次性评估只能构造 calibration Dataset。")

    def __len__(self) -> int:
        return 1


def test_run_calibration_writes_evidence_and_rejects_second_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """编排主流程必须完成 claim，并拒绝同一数据指纹换输出路径重入。"""

    source_plan_path = (
        Path(__file__).parents[2]
        / "configs"
        / "experiments"
        / "initial"
        / "full_scratch_molformer_oracle.yaml"
    )
    source_plan = load_experiment_plan(source_plan_path)
    roots = SourceRoots(tmp_path / "data", tmp_path / "stage1", tmp_path)
    plan = replace(source_plan, data=replace(source_plan.data, roots=roots))
    stage = plan.stages[0]

    checkpoint_path = tmp_path / "run" / stage.name / "BEST.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"selected-checkpoint")
    stage_record = asdict(stage)
    stage_record["model"] = stage.model.to_dict()
    (checkpoint_path.parent / "resolved.json").write_text(
        json.dumps(
            {
                "stage": stage_record,
                "data": {
                    "data_root": str(plan.data.roots.data_root),
                    "stage1_root": str(plan.data.roots.stage1_root),
                    "output_root": str(plan.data.roots.output_root),
                    "coverage_threshold": plan.data.coverage_threshold,
                    "selection_seed": plan.data.selection_seed,
                    "pp_top_k": plan.data.pp_top_k,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    checkpoint: dict[str, object] = {
        "schema_version": 1,
        "model_state": {},
        "stage_name": stage.name,
        "task_name": stage.task_name,
        "epoch": 4,
        "best_epoch": 4,
        "global_step": 10,
        "best_score": 0.6,
        "model_config": stage.model.to_dict(),
        "training_config": asdict(stage.training),
        "epoch_record": {
            "epoch": 4,
            "global_step": 10,
            "selection_score": 0.6,
            "improved": True,
        },
    }
    write_json(
        checkpoint_path.parent / "summary.json",
        {
            "stage": stage.name,
            "best_epoch": 4,
            "best_score": 0.6,
            "last_epoch": 6,
            "best_checkpoint": checkpoint_path.as_posix(),
        },
    )
    write_json(
        checkpoint_path.parent.parent / "run.json",
        {"plan_name": plan.plan_name, "plan_path": source_plan_path.resolve().as_posix()},
    )

    manifest_root = roots.manifest_root
    manifest_root.mkdir(parents=True)
    (manifest_root / "calibration_candidates.jsonl").write_text(
        '{"candidate_index":0}\n',
        encoding="utf-8",
    )
    (manifest_root / "calibration_identities.jsonl").write_text(
        '{"object_key":"LIG"}\n',
        encoding="utf-8",
    )
    manifest_path = manifest_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "splits": {
                    "calibration": {
                        "candidates": "calibration_candidates.jsonl",
                        "identities": "calibration_identities.jsonl",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    numeric_backend: NumericBackendRecord = {
        "python_numpy_torch_seed": stage.training.seed,
        "cudnn_benchmark": True,
        "cudnn_deterministic": False,
        "deterministic_algorithms": False,
        "float32_matmul_precision": "high",
    }
    accumulator = EvaluationAccumulator(top_k=5)
    accumulator.update(
        {
            "class_logits": torch.tensor([[10.0, -1.0, -1.0, -1.0, -1.0, -1.0]]),
            "identity_logits": torch.zeros((1, 1)),
            "ligand_area_logits": torch.full((1, 1, 1, 1), -1.0),
            "binding_area_logits": torch.full((1, 1, 1, 1), -1.0),
        },
        cast(
            EvaluationBatch,
            {
                "class_target": torch.tensor([0]),
                "identity_target": torch.tensor([-1]),
                "ligand_area_target": torch.zeros((1, 1, 1, 1), dtype=torch.bool),
                "binding_area_target": torch.zeros((1, 1, 1, 1), dtype=torch.bool),
                "identity_mask": torch.zeros((1, 1), dtype=torch.bool),
                "metadata": [
                    {
                        "split": "calibration",
                        "pdb_id": "1abc",
                        "candidate_index": 0,
                        "matched_object_key": None,
                        "identity_seen_in_train": None,
                    }
                ],
            },
        ),
    )
    evaluation: EvaluationResult = {
        "loss": {
            "total": 0.0,
            "classification": 0.0,
            "ligand_area": 0.0,
            "binding_area": 0.0,
            "identity": 0.0,
        },
        "metrics": accumulator.compute(),
        "sample_count": 1,
        "identity_eligible_count": 0,
        "elapsed_seconds": 1.0,
        "samples_per_second": 1.0,
    }
    loader_stub = object()
    evaluation_calls = 0

    def evaluate_once(*_: object) -> EvaluationResult:
        nonlocal evaluation_calls
        evaluation_calls += 1
        return evaluation

    monkeypatch.setattr(calibration_module, "load_experiment_plan", lambda _: plan)
    monkeypatch.setattr(calibration_module, "validate_matcher_output_path", lambda path, _: path)
    monkeypatch.setattr(calibration_module, "load_checkpoint", lambda _: checkpoint)
    monkeypatch.setattr(calibration_module, "require_manifest", lambda *_: manifest_path)
    monkeypatch.setattr(calibration_module, "CandidateDataset", _CalibrationDatasetStub)
    monkeypatch.setattr(calibration_module, "build_candidate_loader", lambda *_: loader_stub)
    monkeypatch.setattr(calibration_module, "ClosedSetMatcher", lambda _: nn.Identity())
    monkeypatch.setattr(
        calibration_module,
        "configure_random_seeds_and_numeric_backend",
        lambda _: numeric_backend,
    )
    monkeypatch.setattr(calibration_module, "evaluate_model", evaluate_once)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "test-cuda-device")

    output_path = tmp_path / "calibration" / "result.json"
    assert run_calibration(source_plan_path, stage.name, checkpoint_path, output_path) == output_path
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["split"] == "calibration"
    assert artifact["checkpoint"]["sha256"]
    assert artifact["data"]["fingerprint"]
    six_class = artifact["evaluation"]["metrics"]["six_class"]
    assert six_class["macro_f1_supported_classes"] == 1.0
    assert six_class["macro_f1_all_classes"] == pytest.approx(1.0 / 6.0)
    claim_path = Path(artifact["claim_path"])
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    assert claim["status"] == "completed"
    assert claim["output_path"] == output_path.as_posix()
    assert claim["output_sha256"]
    assert evaluation_calls == 1

    claimed_after_publish = {
        **claim,
        "status": "claimed",
        "output_sha256": None,
        "completed_at_utc": None,
    }
    write_json(claim_path, claimed_after_publish)
    assert run_calibration(source_plan_path, stage.name, checkpoint_path, output_path) == output_path
    recovered = json.loads(claim_path.read_text(encoding="utf-8"))
    assert recovered["status"] == "completed"
    assert recovered["output_sha256"]
    assert evaluation_calls == 1

    with pytest.raises(ValueError, match="不得与一次性 claim"):
        run_calibration(source_plan_path, stage.name, checkpoint_path, claim_path)

    with pytest.raises(RuntimeError, match="已绑定另一项选择|已经完成一次评估"):
        run_calibration(
            source_plan_path,
            stage.name,
            checkpoint_path,
            tmp_path / "calibration" / "second.json",
        )
    claim_path.unlink()
    with pytest.raises(FileExistsError, match="没有对应 claim"):
        run_calibration(source_plan_path, stage.name, checkpoint_path, output_path)
