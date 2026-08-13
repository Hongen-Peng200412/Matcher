"""验证测评、损失累积、配置、checkpoint、校准留证、严格 JSON 和计划执行边界。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pytest
import torch
import yaml  # type: ignore[import-untyped]
from torch import nn

from matcher import experiment, training as training_module
from matcher.batching import collate_candidates
from matcher.calibration import (
    CalibrationCheckpointEvidence,
    _calibration_data_evidence,
    _claim_calibration,
    _select_stage,
    _validate_checkpoint_for_stage,
)
from matcher.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    load_candidate_encoder_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from matcher.experiment_config import (
    DataConfig,
    StageConfig,
    TrackingConfig,
    load_experiment_plan,
    validate_matcher_output_path,
)
from matcher.json_artifacts import append_jsonl, write_json
from matcher.losses import LossOutput, LossWeights, MatcherLossBatch, compute_matcher_loss
from matcher.metrics import EvaluationAccumulator, EvaluationBatch, selection_score
from matcher.schema import SMALL_MOLECULE_INDEX
from matcher.training import (
    _LossStatistics,
    StageSummary,
    _accumulation_backward_loss,
    _resolve_initial_checkpoint,
)

from .factories import build_synthetic_roots


def test_identity_metrics_exclude_candidates_without_current_language_representation() -> None:
    """没有当前语言表示的真实小分子仅进入六分类指标。"""

    class_logits = torch.full((3, 6), -10.0)
    class_logits[0, SMALL_MOLECULE_INDEX] = 10.0
    class_logits[1, SMALL_MOLECULE_INDEX] = 10.0
    class_logits[2, 0] = 10.0
    outputs = {
        "class_logits": class_logits,
        "identity_logits": torch.tensor([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]]),
        "ligand_area_logits": torch.full((3, 1, 1, 1), -1.0),
        "binding_area_logits": torch.full((3, 1, 1, 1), -1.0),
    }
    batch = cast(
        EvaluationBatch,
        {
            "class_target": torch.tensor([SMALL_MOLECULE_INDEX, SMALL_MOLECULE_INDEX, 0]),
            "identity_target": torch.tensor([0, -1, -1]),
            "ligand_area_target": torch.zeros((3, 1, 1, 1), dtype=torch.bool),
            "binding_area_target": torch.zeros((3, 1, 1, 1), dtype=torch.bool),
            "identity_mask": torch.tensor([[True, True], [True, True], [True, True]]),
            "metadata": [
                {
                    "split": "validation",
                    "pdb_id": "a",
                    "candidate_index": 0,
                    "matched_object_key": "A",
                    "identity_seen_in_train": True,
                },
                {
                    "split": "validation",
                    "pdb_id": "b",
                    "candidate_index": 0,
                    "matched_object_key": "B",
                    "identity_seen_in_train": None,
                },
                {
                    "split": "validation",
                    "pdb_id": "c",
                    "candidate_index": 0,
                    "matched_object_key": None,
                    "identity_seen_in_train": None,
                },
            ],
        },
    )

    accumulator = EvaluationAccumulator(top_k=5)
    accumulator.update(outputs, batch)
    metrics = accumulator.compute()

    assert metrics["six_class"]["accuracy"] == 1.0
    assert metrics["six_class"]["macro_f1"] == 1.0
    assert metrics["six_class"]["macro_f1_supported_classes"] == 1.0
    assert metrics["six_class"]["macro_f1_all_classes"] == pytest.approx(2.0 / 6.0)
    assert metrics["conditional_identity"]["eligible_count"] == 1
    assert metrics["conditional_identity"]["top1"] == 1.0
    assert metrics["end_to_end_identity"]["true_positive"] == 1
    assert metrics["end_to_end_identity"]["predicted_small_molecule"] == 1
    assert metrics["end_to_end_identity"]["actual_small_molecule"] == 1


def test_end_to_end_identity_counts_background_small_molecule_prediction_as_false_positive() -> None:
    """有身份清单的背景候选若被预测为 small_molecule，必须降低端到端精确率。"""

    class_logits = torch.full((2, 6), -10.0)
    class_logits[:, SMALL_MOLECULE_INDEX] = 10.0
    outputs = {
        "class_logits": class_logits,
        "identity_logits": torch.tensor([[5.0, 0.0], [5.0, 0.0]]),
        "ligand_area_logits": torch.full((2, 1, 1, 1), -1.0),
        "binding_area_logits": torch.full((2, 1, 1, 1), -1.0),
    }
    batch = cast(
        EvaluationBatch,
        {
            "class_target": torch.tensor([SMALL_MOLECULE_INDEX, 0]),
            "identity_target": torch.tensor([0, -1]),
            "identity_mask": torch.tensor([[True, True], [True, True]]),
            "ligand_area_target": torch.zeros((2, 1, 1, 1), dtype=torch.bool),
            "binding_area_target": torch.zeros((2, 1, 1, 1), dtype=torch.bool),
            "metadata": [
                {
                    "split": "validation",
                    "pdb_id": "a",
                    "candidate_index": 0,
                    "matched_object_key": "A",
                    "identity_seen_in_train": True,
                },
                {
                    "split": "validation",
                    "pdb_id": "b",
                    "candidate_index": 0,
                    "matched_object_key": None,
                    "identity_seen_in_train": None,
                },
            ],
        },
    )
    accumulator = EvaluationAccumulator(top_k=5)
    accumulator.update(outputs, batch)
    end_to_end = accumulator.compute()["end_to_end_identity"]
    assert end_to_end["true_positive"] == 1
    assert end_to_end["predicted_small_molecule"] == 2
    assert end_to_end["actual_small_molecule"] == 1
    assert end_to_end["precision"] == 0.5
    assert end_to_end["recall"] == 1.0


def test_conditional_identity_requires_correct_six_class_prediction() -> None:
    """条件身份只统计六分类判对的小分子，固有身份诊断仍覆盖全部可监督目标。"""

    class_logits = torch.full((2, 6), -10.0)
    class_logits[0, 0] = 10.0
    class_logits[1, SMALL_MOLECULE_INDEX] = 10.0
    outputs = {
        "class_logits": class_logits,
        "identity_logits": torch.tensor([[5.0, 0.0], [0.0, 5.0]]),
        "ligand_area_logits": torch.zeros((2, 1, 1, 1)),
        "binding_area_logits": torch.zeros((2, 1, 1, 1)),
    }
    batch = cast(
        EvaluationBatch,
        {
            "class_target": torch.tensor([SMALL_MOLECULE_INDEX, SMALL_MOLECULE_INDEX]),
            "identity_target": torch.tensor([0, 0]),
            "identity_mask": torch.ones((2, 2), dtype=torch.bool),
            "ligand_area_target": torch.zeros((2, 1, 1, 1), dtype=torch.bool),
            "binding_area_target": torch.zeros((2, 1, 1, 1), dtype=torch.bool),
            "metadata": [
                {
                    "split": "validation",
                    "pdb_id": "a",
                    "candidate_index": 0,
                    "matched_object_key": "A",
                    "identity_seen_in_train": True,
                },
                {
                    "split": "validation",
                    "pdb_id": "b",
                    "candidate_index": 0,
                    "matched_object_key": "B",
                    "identity_seen_in_train": False,
                },
            ],
        },
    )
    accumulator = EvaluationAccumulator(top_k=2)
    accumulator.update(outputs, batch)
    metrics = accumulator.compute()
    assert metrics["intrinsic_identity"]["top1"] == 0.5
    assert metrics["intrinsic_identity"]["eligible_count"] == 2
    assert metrics["conditional_identity"]["top1"] == 0.0
    assert metrics["conditional_identity"]["topk"] == 1.0
    assert metrics["conditional_identity"]["top_k"] == 2
    assert metrics["conditional_identity"]["eligible_count"] == 1
    assert metrics["conditional_identity"]["seen_count"] == 0
    assert metrics["conditional_identity"]["unseen_count"] == 1


def test_evaluator_and_loss_statistics_reject_nonfinite_values() -> None:
    """NaN 模型输出或损失必须立即失败，不能被 argmax 或选模逻辑解释。"""

    outputs = {
        "class_logits": torch.full((1, 6), float("nan")),
        "ligand_area_logits": torch.zeros((1, 1, 1, 1)),
        "binding_area_logits": torch.zeros((1, 1, 1, 1)),
    }
    batch = cast(
        EvaluationBatch,
        {
            "class_target": torch.tensor([0]),
            "identity_target": torch.tensor([-1]),
            "identity_mask": torch.empty((1, 0), dtype=torch.bool),
            "ligand_area_target": torch.zeros((1, 1, 1, 1), dtype=torch.bool),
            "binding_area_target": torch.zeros((1, 1, 1, 1), dtype=torch.bool),
            "metadata": [
                {
                    "split": "validation",
                    "pdb_id": "a",
                    "candidate_index": 0,
                    "matched_object_key": None,
                    "identity_seen_in_train": None,
                }
            ],
        },
    )
    with pytest.raises(FloatingPointError, match="class_logits"):
        EvaluationAccumulator().update(outputs, batch)

    losses = cast(
        LossOutput,
        {
            "total": torch.tensor(float("nan")),
            "classification": torch.tensor(0.0),
            "ligand_area": torch.tensor(0.0),
            "binding_area": torch.tensor(0.0),
            "identity": torch.tensor(0.0),
            "classification_denominator": torch.tensor(1.0),
            "identity_eligible_count": torch.tensor(0),
        },
    )
    with pytest.raises(FloatingPointError, match="total"):
        _LossStatistics().update(losses, batch_size=1)


def test_evaluator_rejects_task_outputs_with_misaligned_batch_axes() -> None:
    """区域或身份输出若不与六分类批次轴对齐，评估器必须在累计前拒绝。"""

    metadata = [
        {
            "split": "validation",
            "pdb_id": f"pdb_{index}",
            "candidate_index": index,
            "matched_object_key": None,
            "identity_seen_in_train": None,
        }
        for index in range(2)
    ]
    area_mismatch_batch = cast(
        EvaluationBatch,
        {
            "class_target": torch.tensor([0, 0]),
            "identity_target": torch.tensor([-1, -1]),
            "identity_mask": torch.empty((2, 0), dtype=torch.bool),
            "ligand_area_target": torch.zeros((1, 1, 1, 1), dtype=torch.bool),
            "binding_area_target": torch.zeros((2, 1, 1, 1), dtype=torch.bool),
            "metadata": metadata,
        },
    )
    area_mismatch_outputs = {
        "class_logits": torch.zeros((2, 6)),
        "ligand_area_logits": torch.zeros((1, 1, 1, 1)),
        "binding_area_logits": torch.zeros((2, 1, 1, 1)),
    }
    with pytest.raises(ValueError, match="ligand.*class_logits"):
        EvaluationAccumulator().update(area_mismatch_outputs, area_mismatch_batch)

    identity_mismatch_batch = cast(
        EvaluationBatch,
        {
            "class_target": torch.tensor([0, 0]),
            "identity_target": torch.tensor([-1]),
            "identity_mask": torch.zeros((1, 1), dtype=torch.bool),
            "ligand_area_target": torch.zeros((2, 1, 1, 1), dtype=torch.bool),
            "binding_area_target": torch.zeros((2, 1, 1, 1), dtype=torch.bool),
            "metadata": metadata,
        },
    )
    identity_mismatch_outputs = {
        "class_logits": torch.zeros((2, 6)),
        "identity_logits": torch.zeros((1, 1)),
        "ligand_area_logits": torch.zeros((2, 1, 1, 1)),
        "binding_area_logits": torch.zeros((2, 1, 1, 1)),
    }
    with pytest.raises(ValueError, match="identity_logits.*class_logits"):
        EvaluationAccumulator().update(identity_mismatch_outputs, identity_mismatch_batch)


def test_selection_score_uses_only_validation_report_components() -> None:
    """候选预训练只用六分类，完整 Matcher 等权合并三项指标。"""

    assert selection_score("candidate_pretrain", 0.6, 0.3, 0.9) == 0.6
    assert abs(selection_score("full_matcher", 0.6, 0.3, 0.9) - 0.6) < 1e-12


def test_all_initial_experiment_plans_preserve_resource_and_stopping_contracts() -> None:
    """七个首轮 YAML 分别冻结 train/validation worker 与 global batch 32。"""

    config_root = Path(__file__).parents[2] / "configs" / "experiments" / "initial"
    paths = sorted(config_root.glob("*.yaml"))
    assert len(paths) == 7
    plans = [load_experiment_plan(path) for path in paths]

    for plan in plans:
        assert plan.output_root.is_relative_to(Path("/storage/penghongen/tmp/Matcher"))
        assert plan.data.train_num_workers == 4
        assert plan.data.validation_num_workers == 4
        assert plan.tracking == TrackingConfig(
            provider="wandb",
            mode="online",
            entity="pencounkdual-111",
            project="AdaLigand-Matcher",
            group="matcher-initial-round1",
            log_every_global_steps=1,
            init_timeout_seconds=120,
        )
        for stage in plan.stages:
            assert stage.training.global_batch_size == 32
            assert stage.training.maximum_epochs is None
            assert stage.training.patience_bad_epochs == 2

    pair = next(plan for plan in plans if plan.plan_name == "candidate_pretrain_pair")
    assert [stage.model.aggregation for stage in pair.stages] == [
        "mean_max",
        "candidate_cross_attention",
    ]
    assert [stage.training.physical_batch_size for stage in pair.stages] == [8, 8]

    full_plans = [plan for plan in plans if plan.plan_name != "candidate_pretrain_pair"]
    assert all(len(plan.stages) == 1 for plan in full_plans)
    assert all(plan.stages[0].model.aggregation == "mean_max" for plan in full_plans)
    assert {
        (plan.stages[0].language_family, plan.stages[0].identity_variant)
        for plan in full_plans
    } == {
        (family, variant)
        for family in ("molformer", "smi_ted_289m")
        for variant in ("oracle", "detached", "joint")
    }
    for plan in full_plans:
        stage = plan.stages[0]
        if stage.language_family == "smi_ted_289m":
            assert stage.training.physical_batch_size == 16
            assert stage.training.gradient_accumulation == 2
            assert plan.data.cache_gib_per_worker == 32.0
        else:
            assert stage.training.physical_batch_size == 8
            assert stage.training.gradient_accumulation == 4
            assert plan.data.cache_gib_per_worker == 16.0


def test_experiment_config_rejects_output_escape_and_relative_initialization(
    tmp_path: Path,
) -> None:
    """规范化后越过业务产物根的路径，以及相对 checkpoint，都必须在启动前失败。"""

    source = (
        Path(__file__).parents[2]
        / "configs"
        / "experiments"
        / "initial"
        / "full_scratch_molformer_oracle.yaml"
    )
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    values["output_root"] = "/storage/penghongen/tmp/Matcher/../../outside"
    escaped = tmp_path / "escaped.yaml"
    escaped.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="不在允许的服务器产物根"):
        load_experiment_plan(escaped)

    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    values["stages"][0]["initialize_candidate_from"] = "relative/BEST.pt"
    relative = tmp_path / "relative.yaml"
    relative.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="必须是绝对路径"):
        load_experiment_plan(relative)


def test_calibration_requires_matching_stage_checkpoint_and_safe_output(tmp_path: Path) -> None:
    """保留集入口必须绑定同阶段配置，并拒绝规范化后越过 Matcher 根的产物路径。"""

    plan_path = (
        Path(__file__).parents[2]
        / "configs"
        / "experiments"
        / "initial"
        / "full_scratch_molformer_oracle.yaml"
    )
    plan = load_experiment_plan(plan_path)
    stage = _select_stage(plan, "full_scratch_molformer_oracle")
    checkpoint_path = tmp_path / "run" / stage.name / "BEST.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_bytes = b"calibration-checkpoint-evidence"
    checkpoint_path.write_bytes(checkpoint_bytes)
    stage_record = asdict(stage)
    stage_record["model"] = stage.model.to_dict()
    resolved_path = checkpoint_path.parent / "resolved.json"
    resolved_path.write_text(
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
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": {},
        "stage_name": stage.name,
        "task_name": stage.task_name,
        "epoch": 7,
        "best_epoch": 7,
        "global_step": 3576,
        "best_score": 0.65,
        "model_config": stage.model.to_dict(),
        "training_config": asdict(stage.training),
        "epoch_record": {
            "epoch": 7,
            "global_step": 3576,
            "selection_score": 0.65,
            "improved": True,
        },
    }
    write_json(
        checkpoint_path.parent / "summary.json",
        {
            "stage": stage.name,
            "best_epoch": 7,
            "best_score": 0.65,
            "last_epoch": 9,
            "best_checkpoint": checkpoint_path.as_posix(),
        },
    )
    write_json(
        checkpoint_path.parent.parent / "run.json",
        {"plan_name": plan.plan_name, "plan_path": plan_path.resolve().as_posix()},
    )

    evidence = _validate_checkpoint_for_stage(checkpoint_path, checkpoint, plan, stage)
    assert evidence["epoch"] == 7
    assert evidence["global_step"] == 3576
    assert evidence["sha256"] == hashlib.sha256(checkpoint_bytes).hexdigest()
    with pytest.raises(ValueError, match="stage_name"):
        _validate_checkpoint_for_stage(
            checkpoint_path,
            {**checkpoint, "stage_name": "wrong"},
            plan,
            stage,
        )
    with pytest.raises(ValueError, match="epoch == checkpoint.best_epoch"):
        _validate_checkpoint_for_stage(
            checkpoint_path,
            {**checkpoint, "best_epoch": 6},
            plan,
            stage,
        )
    resolved_values = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved_values["data"]["selection_seed"] = 999999
    resolved_path.write_text(json.dumps(resolved_values), encoding="utf-8")
    with pytest.raises(ValueError, match="resolved.json.data 科学字段"):
        _validate_checkpoint_for_stage(checkpoint_path, checkpoint, plan, stage)
    resolved_values["data"]["selection_seed"] = plan.data.selection_seed
    resolved_path.write_text(json.dumps(resolved_values), encoding="utf-8")
    stage_record["language_family"] = "smi_ted_289m"
    resolved_path.write_text(
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
    with pytest.raises(ValueError, match="resolved.json.stage"):
        _validate_checkpoint_for_stage(checkpoint_path, checkpoint, plan, stage)
    assert validate_matcher_output_path(
        Path("/storage/penghongen/tmp/Matcher/calibration/report.json"),
        "output",
    ).as_posix() == "/storage/penghongen/tmp/Matcher/calibration/report.json"
    with pytest.raises(ValueError, match="不在允许的服务器产物根"):
        validate_matcher_output_path(
            Path("/storage/penghongen/tmp/Matcher/../../outside/report.json"),
            "output",
        )


def test_calibration_data_fingerprint_binds_one_selection_while_claim_is_retained(
    tmp_path: Path,
) -> None:
    """claim 保留期间，同一 calibration 记录只能绑定一个 checkpoint 与输出路径。"""

    manifest_path = tmp_path / "manifest.json"
    candidates_path = tmp_path / "calibration_candidates.jsonl"
    identities_path = tmp_path / "calibration_identities.jsonl"
    candidates_path.write_text('{"candidate_index":0}\n', encoding="utf-8")
    identities_path.write_text('{"object_key":"LIG"}\n', encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": "2026-08-14T00:00:00+00:00",
                "split_pdb_counts": {"train": {"included": 2}, "calibration": {"included": 1}},
                "summary": {
                    "splits": {
                        "train": {"candidate_count": 2},
                        "calibration": {"candidate_count": 1},
                    }
                },
                "splits": {
                    "calibration": {
                        "candidates": candidates_path.name,
                        "identities": identities_path.name,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    data = _calibration_data_evidence(manifest_path)
    manifest_values = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_values["created_at_utc"] = "2026-08-14T01:00:00+00:00"
    manifest_values["summary"]["splits"]["train"]["candidate_count"] = 999
    manifest_path.write_text(json.dumps(manifest_values), encoding="utf-8")
    rebuilt_data = _calibration_data_evidence(manifest_path)
    assert rebuilt_data["manifest_sha256"] != data["manifest_sha256"]
    assert rebuilt_data["fingerprint"] == data["fingerprint"]
    checkpoint: CalibrationCheckpointEvidence = {
        "path": "/storage/penghongen/tmp/Matcher/run/stage/BEST.pt",
        "sha256": "a" * 64,
        "epoch": 1,
        "global_step": 2,
        "best_score": 0.5,
        "resolved_path": "/storage/penghongen/tmp/Matcher/run/stage/resolved.json",
        "resolved_sha256": "c" * 64,
        "summary_path": "/storage/penghongen/tmp/Matcher/run/stage/summary.json",
        "summary_sha256": "d" * 64,
        "run_path": "/storage/penghongen/tmp/Matcher/run/run.json",
        "run_sha256": "e" * 64,
        "training_plan_path": "/storage/penghongen/tmp/Matcher/release/plan.yaml",
        "training_plan_sha256": "f" * 64,
        "model_config": {},
        "training_config": {},
    }
    claim_path = tmp_path / "claim.json"
    output_path = Path("/storage/penghongen/tmp/Matcher/calibration/result.json")

    claim = _claim_calibration(
        claim_path,
        data["fingerprint"],
        "b" * 64,
        "selected_stage",
        checkpoint,
        output_path,
        "2026-08-14T00:00:00+00:00",
    )
    assert claim["status"] == "claimed"
    assert _claim_calibration(
        claim_path,
        data["fingerprint"],
        "b" * 64,
        "selected_stage",
        checkpoint,
        output_path,
        "2026-08-14T01:00:00+00:00",
    ) == claim
    with pytest.raises(RuntimeError, match="已绑定另一项选择"):
        _claim_calibration(
            claim_path,
            data["fingerprint"],
            "b" * 64,
            "another_stage",
            checkpoint,
            output_path,
            "2026-08-14T01:00:00+00:00",
        )
    write_json(
        claim_path,
        {
            **claim,
            "status": "completed",
            "output_sha256": "9" * 64,
            "completed_at_utc": "2026-08-14T02:00:00+00:00",
        },
    )
    with pytest.raises(RuntimeError, match="已经完成一次评估"):
        _claim_calibration(
            claim_path,
            data["fingerprint"],
            "b" * 64,
            "selected_stage",
            checkpoint,
            output_path,
            "2026-08-14T03:00:00+00:00",
        )


def test_experiment_config_rejects_seed_outside_numpy_range(tmp_path: Path) -> None:
    """训练种子必须同时落在 Python、NumPy、Torch 和 worker 派生公式的共同范围。"""

    source = (
        Path(__file__).parents[2]
        / "configs"
        / "experiments"
        / "initial"
        / "full_scratch_molformer_oracle.yaml"
    )
    for value in (-1, 2**32):
        values = yaml.safe_load(source.read_text(encoding="utf-8"))
        values["stages"][0]["training"]["seed"] = value
        invalid = tmp_path / f"seed_{value}.yaml"
        invalid.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError, match=r"seed 必须位于 \[0,4294967295\]"):
            load_experiment_plan(invalid)


def test_initial_checkpoint_resolver_accepts_only_stage_dsl_or_posix_absolute_path() -> None:
    """运行时解析器只接受同计划阶段引用或服务器 POSIX 绝对路径。"""

    plan_directory = Path("/tmp/matcher-plan")
    assert _resolve_initial_checkpoint(
        "stage:candidate_weak:BEST",
        plan_directory,
    ) == plan_directory / "candidate_weak" / "BEST.pt"
    external = _resolve_initial_checkpoint(
        "/storage/penghongen/tmp/Matcher/checkpoints/BEST.pt",
        plan_directory,
    )
    assert external.as_posix() == "/storage/penghongen/tmp/Matcher/checkpoints/BEST.pt"
    with pytest.raises(ValueError, match="绝对路径"):
        _resolve_initial_checkpoint("checkpoints/BEST.pt", plan_directory)
    for unsafe_reference in ("stage:..:BEST", "stage:.hidden:BEST", "stage:-dash:LAST"):
        with pytest.raises(ValueError, match="stage:<name>"):
            _resolve_initial_checkpoint(unsafe_reference, plan_directory)


def test_stage_runtime_uses_spawn_loaders_and_records_process_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train/validation 分别配置非持久 spawn 进程池，并把随机隔离契约写入产物。"""

    class CpuOnlyModel(nn.Module):
        """提供不触发真实 CUDA 分配的最小可优化模型替身。"""

        def __init__(self, _config: object) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def to(self, *_args: object, **_kwargs: object) -> CpuOnlyModel:
            """保留 CPU 参数并模拟正式模型的设备迁移接口。"""

            return self

    plan = load_experiment_plan(
        Path(__file__).parents[2] / "configs/experiments/initial/candidate_pretrain_pair.yaml"
    )
    roots = build_synthetic_roots(tmp_path / "sources")
    data = replace(
        plan.data,
        roots=roots,
        train_num_workers=1,
        validation_num_workers=2,
        prefetch_factor=1,
        cache_gib_per_worker=0.01,
    )
    monkeypatch.setattr(training_module, "CandidateEncoder", CpuOnlyModel)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "synthetic-cuda")
    stage_directory = tmp_path / "stage"
    stage_directory.mkdir()

    runtime = training_module._build_stage_runtime(
        plan.stages[0],
        data,
        plan.tracking,
        stage_directory,
        tmp_path,
    )

    for loader, expected_workers in (
        (runtime.train_loader, 1),
        (runtime.validation_loader, 2),
    ):
        assert loader.num_workers == expected_workers
        assert loader.multiprocessing_context is not None
        assert loader.multiprocessing_context.get_start_method() == "spawn"
        assert loader.pin_memory is True
        assert loader.persistent_workers is False
        assert loader.collate_fn is collate_candidates
        assert loader.generator is not None
        # 正式属性已在上方断言；CPU-only 门禁不具备驱动，迭代探针只关闭测试对象的页锁定线程。
        loader.pin_memory = False
    assert runtime.train_loader.generator is not runtime.validation_loader.generator
    assert runtime.train_loader.generator.initial_seed() == 2 * plan.stages[0].training.seed
    assert (
        runtime.validation_loader.generator.initial_seed()
        == 2 * plan.stages[0].training.seed + 1
    )
    torch.manual_seed(71)
    expected_global_draw = torch.rand(4)
    torch.manual_seed(71)
    for loader in (runtime.train_loader, runtime.validation_loader):
        loader_iterator = iter(loader)
        worker_processes = tuple(cast(Any, loader_iterator)._workers)
        assert list(loader_iterator) == []
        for process in worker_processes:
            process.join(timeout=5.0)
            assert not process.is_alive()
    assert torch.equal(torch.rand(4), expected_global_draw)
    resolved = json.loads((stage_directory / "resolved.json").read_text(encoding="utf-8"))
    assert resolved["data"]["multiprocessing_context"] == "spawn"
    assert resolved["data"]["train_num_workers"] == 1
    assert resolved["data"]["validation_num_workers"] == 2
    assert resolved["data"]["persistent_workers"] is False
    assert resolved["data"]["worker_pool_lifecycle"] == "per_complete_loader_iteration"
    assert resolved["data"]["train_worker_generator_seed"] == 6814
    assert resolved["data"]["validation_worker_generator_seed"] == 6815
    assert resolved["data"]["worker_generator_lifecycle"] == (
        "one generator per loader; iterator creation advances only that generator"
    )


def test_checkpoint_schema_and_candidate_prefix_loading(tmp_path: Path) -> None:
    """候选参数可从直接模型或完整模型前缀加载，损坏 model_state 会给出契约错误。"""

    source = nn.Linear(3, 2)
    direct_path = tmp_path / "direct.pt"
    save_checkpoint(
        direct_path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state": source.state_dict(),
        },
    )
    target = nn.Linear(3, 2)
    assert load_candidate_encoder_checkpoint(target, direct_path) == {
        "missing_keys": [],
        "unexpected_keys": [],
    }
    for expected, actual in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.equal(expected, actual)

    class FullModel(nn.Module):
        """为测试模拟含 candidate_encoder 子模块的完整 Matcher。"""

        def __init__(self) -> None:
            super().__init__()
            self.candidate_encoder = nn.Linear(3, 2)

    full_source = FullModel()
    prefixed_path = tmp_path / "prefixed.pt"
    save_checkpoint(
        prefixed_path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state": full_source.state_dict(),
        },
    )
    full_target = FullModel()
    load_candidate_encoder_checkpoint(full_target, prefixed_path)
    for expected, actual in zip(
        full_source.candidate_encoder.parameters(),
        full_target.candidate_encoder.parameters(),
        strict=True,
    ):
        assert torch.equal(expected, actual)

    malformed_path = tmp_path / "malformed.pt"
    torch.save(
        {"schema_version": CHECKPOINT_SCHEMA_VERSION, "model_state": ["weight"]},
        malformed_path,
    )
    with pytest.raises(ValueError, match="model_state.*映射"):
        load_checkpoint(malformed_path)


def test_json_artifacts_are_strict_and_run_plan_preserves_stage_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON 边界写 null 而非 NaN，计划按 YAML 阶段顺序写出起止产物。"""

    json_path = tmp_path / "strict.json"
    jsonl_path = tmp_path / "strict.jsonl"
    write_json(json_path, {"zero_support": float("nan")})
    append_jsonl(jsonl_path, MappingProxyType({"zero_support": float("inf")}))
    assert json.loads(json_path.read_text(encoding="utf-8"))["zero_support"] is None
    assert json.loads(jsonl_path.read_text(encoding="utf-8"))["zero_support"] is None
    with pytest.raises(TypeError, match="字符串"):
        write_json(tmp_path / "bad-key.json", cast(Any, {1: "collision"}))

    source = (
        Path(__file__).parents[2]
        / "configs"
        / "experiments"
        / "initial"
        / "candidate_pretrain_pair.yaml"
    )
    plan = replace(load_experiment_plan(source), output_root=tmp_path)
    visited: list[str] = []

    def fake_run_training_stage(
        stage: StageConfig,
        data: DataConfig,
        tracking: TrackingConfig,
        stage_directory: Path,
        plan_directory: Path,
    ) -> StageSummary:
        """记录调用顺序并返回满足正式结构的最小阶段摘要。"""

        del data, tracking
        visited.append(stage.name)
        return {
            "stage": stage.name,
            "best_epoch": 0,
            "best_score": 0.5,
            "last_epoch": 0,
            "global_step": 1,
            "stopped_by_patience": False,
            "best_checkpoint": str(stage_directory / "BEST.pt"),
            "last_checkpoint": str(plan_directory / stage.name / "LAST.pt"),
            "tracking_run_id": f"run-{stage.name}",
            "tracking_url": f"https://wandb.ai/test/{stage.name}",
        }

    monkeypatch.setattr(experiment, "load_experiment_plan", lambda _: plan)
    monkeypatch.setattr(experiment, "run_training_stage", fake_run_training_stage)
    monkeypatch.setenv("TASK_RUN_STAMP", "test-run")
    run_directory = experiment.run_plan(source)
    assert visited == [stage.name for stage in plan.stages]
    assert (run_directory / "run.json").is_file()
    assert (run_directory / "summary.json").is_file()
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    assert [stage["stage"] for stage in summary["stages"]] == visited


def test_accumulated_joint_loss_matches_one_global_batch_with_uneven_identity_counts() -> None:
    """缩小例中有效身份数不均匀时，微批累积仍等于一次计算整个 global batch。"""

    weights = LossWeights(classification=1.0, ligand_area=0.1, binding_area=0.1, identity=1.0)
    class_logits = torch.tensor(
        [[2.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 2.0]],
        requires_grad=True,
    )
    identity_logits = torch.tensor(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [2.0, 0.0, 0.0]],
        requires_grad=True,
    )
    class_target = torch.tensor([0, 4, 4, 4])
    identity_target = torch.tensor([-1, -1, 2, 0])
    area_target = torch.zeros((4, 1, 1, 1), dtype=torch.bool)
    area_logits = torch.zeros((4, 1, 1, 1), requires_grad=True)

    def calculate(start: int, end: int) -> LossOutput:
        outputs = {
            "class_logits": class_logits[start:end],
            "identity_logits": identity_logits[start:end],
            "ligand_area_logits": area_logits[start:end],
            "binding_area_logits": area_logits[start:end],
        }
        batch = cast(
            MatcherLossBatch,
            {
                "class_target": class_target[start:end],
                "identity_target": identity_target[start:end],
                "identity_mask": torch.ones((end - start, 3), dtype=torch.bool),
                "ligand_area_target": area_target[start:end],
                "binding_area_target": area_target[start:end],
            },
        )
        return compute_matcher_loss(
            outputs,
            batch,
            weights,
            identity_variant="joint",
            class_weights=None,
        )

    global_loss = calculate(0, 4)["total"]
    first = calculate(0, 2)
    second = calculate(2, 4)
    accumulated = _accumulation_backward_loss(first, weights, 0.5, 4.0, 2)
    accumulated = accumulated + _accumulation_backward_loss(second, weights, 0.5, 4.0, 2)
    assert torch.allclose(accumulated, global_loss, atol=1e-7, rtol=1e-7)
