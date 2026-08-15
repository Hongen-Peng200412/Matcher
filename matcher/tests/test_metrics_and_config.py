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

from matcher.data.schema import CLASS_TO_INDEX, SMALL_MOLECULE_INDEX
from matcher.datasets.batching import collate_candidates
from matcher.evaluation.calibration import (
    CalibrationCheckpointEvidence,
    _calibration_data_evidence,
    _claim_calibration,
    _select_stage,
    _validate_checkpoint_for_stage,
)
from matcher.evaluation.metrics import EvaluationAccumulator, EvaluationBatch, selection_score
from matcher.model import IdentityScoreComponents
from matcher.training import engine as training_module
from matcher.training import experiment
from matcher.training import tracking as tracking_module
from matcher.training.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    load_candidate_encoder_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from matcher.training.config import (
    DataConfig,
    StageConfig,
    TrackingConfig,
    load_experiment_plan,
    validate_matcher_output_path,
)
from matcher.training.engine import (
    _LossStatistics,
    StageSummary,
    _accumulation_backward_loss,
    _resolve_initial_checkpoint,
)
from matcher.training.losses import LossOutput, LossWeights, MatcherLossBatch, compute_matcher_loss
from matcher.utils.json_artifacts import append_jsonl, write_json

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
            "identity_class_index": torch.full((3, 2), 4, dtype=torch.long),
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

    accumulator = EvaluationAccumulator(identity_scope="small_molecule", top_k=5)
    accumulator.update(outputs, batch)
    metrics = accumulator.compute()

    assert metrics["six_class"]["accuracy"] == 1.0
    assert metrics["six_class"]["macro_f1"] == 1.0
    assert metrics["six_class"]["macro_f1_supported_classes"] == 1.0
    assert metrics["six_class"]["macro_f1_all_classes"] == pytest.approx(2.0 / 6.0)
    assert metrics["conditional_identity"]["eligible_count"] == 1
    assert metrics["conditional_identity"]["top1"] == 1.0
    assert metrics["end_to_end_identity"]["true_positive"] == 1
    assert metrics["end_to_end_identity"]["predicted_positive"] == 1
    assert metrics["end_to_end_identity"]["actual_positive"] == 1


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
            "identity_class_index": torch.full((2, 2), 4, dtype=torch.long),
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
    accumulator = EvaluationAccumulator(identity_scope="small_molecule", top_k=5)
    accumulator.update(outputs, batch)
    end_to_end = accumulator.compute()["end_to_end_identity"]
    assert end_to_end["true_positive"] == 1
    assert end_to_end["predicted_positive"] == 2
    assert end_to_end["actual_positive"] == 1
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
            "identity_class_index": torch.full((2, 2), 4, dtype=torch.long),
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
    accumulator = EvaluationAccumulator(identity_scope="small_molecule", top_k=2)
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


def test_five_class_identity_metrics_keep_small_compatibility_and_per_class_counts() -> None:
    """五类总体按共享清单排名，小分子兼容指标只在旧小分子身份子清单中排名。"""

    ion = CLASS_TO_INDEX["ion"]
    sugar = CLASS_TO_INDEX["sugar"]
    class_target = torch.tensor([ion, SMALL_MOLECULE_INDEX, sugar, 0])
    class_logits = torch.full((4, 6), -10.0)
    class_logits[torch.arange(4), class_target] = 10.0
    outputs = {
        "class_logits": class_logits,
        # 第二个候选在五类共享清单中误选 ion；只屏蔽到小分子槽后则正确。
        "identity_logits": torch.tensor(
            [[9.0, 1.0, 0.0], [9.0, 5.0, 0.0], [0.0, 1.0, 9.0], [0.0, 0.0, 0.0]]
        ),
        "ligand_area_logits": torch.full((4, 1, 1, 1), -1.0),
        "binding_area_logits": torch.full((4, 1, 1, 1), -1.0),
    }
    batch = cast(
        EvaluationBatch,
        {
            "class_target": class_target,
            "identity_target": torch.tensor([0, 1, 2, -1]),
            "identity_mask": torch.ones((4, 3), dtype=torch.bool),
            "identity_class_index": torch.tensor(
                [[ion, SMALL_MOLECULE_INDEX, sugar]] * 4
            ),
            "ligand_area_target": torch.zeros((4, 1, 1, 1), dtype=torch.bool),
            "binding_area_target": torch.zeros((4, 1, 1, 1), dtype=torch.bool),
            "metadata": [
                {
                    "split": "validation",
                    "pdb_id": f"pdb-{index}",
                    "candidate_index": 0,
                    "matched_object_key": None if index == 3 else f"identity-{index}",
                    "identity_seen_in_train": None if index == 3 else index != 2,
                }
                for index in range(4)
            ],
        },
    )
    accumulator = EvaluationAccumulator(identity_scope="all_non_background", top_k=2)
    accumulator.update(outputs, batch)
    report = accumulator.compute()

    assert report["intrinsic_identity"]["eligible_count"] == 3
    assert report["intrinsic_identity"]["top1"] == pytest.approx(2.0 / 3.0)
    assert report["small_molecule_identity"]["intrinsic"] == {
        "top1": 1.0,
        "topk": 1.0,
        "top_k": 2,
        "eligible_count": 1,
    }
    assert report["all_non_background_identity"]["intrinsic"]["top1"] == pytest.approx(
        2.0 / 3.0
    )
    assert report["identity_per_class"]["ion"]["intrinsic"]["eligible_count"] == 1
    assert report["identity_per_class"]["ion"]["intrinsic"]["top1"] == 1.0
    assert report["identity_per_class"]["small_molecule"]["intrinsic"]["top1"] == 0.0
    assert report["identity_per_class"]["sugar"]["intrinsic"]["top1"] == 1.0
    assert report["identity_per_class"]["nucleotide_like"]["intrinsic"][
        "eligible_count"
    ] == 0
    assert report["conditional_identity"]["seen_count"] == 2
    assert report["conditional_identity"]["unseen_count"] == 1

    evaluation = cast(
        training_module.EvaluationResult,
        {
            "loss": {
                "total": 1.0,
                "classification": 0.5,
                "ligand_area": 0.2,
                "binding_area": 0.2,
                "identity": 0.1,
            },
            "metrics": report,
            "sample_count": 4,
            "identity_eligible_count": 3,
            "elapsed_seconds": 2.0,
            "samples_per_second": 2.0,
        },
    )
    wandb_metrics = training_module._evaluation_wandb_metrics(
        evaluation,
        "all_non_background",
    )
    normalized = tracking_module._normalize_metrics(wandb_metrics)

    assert normalized["classification_summary/macro_f1"] == report["six_class"][
        "macro_f1"
    ]
    assert normalized[
        "identity_small_molecule/intrinsic/eligible_count"
    ] == 1
    assert normalized[
        "identity_all_non_background/intrinsic/eligible_count"
    ] == 3
    assert normalized[
        "identity_nucleotide_like/intrinsic/top1"
    ] is None
    assert normalized[
        "classification_per_class/nucleotide_like/recall"
    ] is None
    assert normalized["classification_confusion/ion/ion"] == 1
    assert "validation/macro_f1" not in normalized
    candidate_only_metrics = training_module._evaluation_wandb_metrics(evaluation, None)
    assert not any(name.startswith("identity_") for name in candidate_only_metrics)


def test_validation_identity_contribution_uses_centered_multi_identity_ranking_scale() -> None:
    """贡献诊断只统计有目标的多身份候选，并以去均值 L2 排名尺度计算门控增量比例。"""

    components = cast(
        IdentityScoreComponents,
        {
            "native": torch.tensor(
                [
                    [0.0, 2.0, 4.0],
                    [1.0, 3.0, float("nan")],
                    [2.0, 4.0, 6.0],
                    [1.0, 2.0, 3.0],
                ]
            ),
            "graph_increment": torch.tensor(
                [
                    [0.0, 1.0, 2.0],
                    [0.0, 2.0, float("nan")],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ),
            "inventory_increment": torch.tensor(
                [
                    [0.0, -1.0, -2.0],
                    [1.0, 1.0, float("nan")],
                    [2.0, 2.0, 2.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        },
    )
    batch = cast(
        MatcherLossBatch,
        {
            "class_target": torch.tensor(
                [SMALL_MOLECULE_INDEX, 1, SMALL_MOLECULE_INDEX, 0]
            ),
            "identity_target": torch.tensor([0, 1, -1, -1]),
            "identity_mask": torch.tensor(
                [
                    [True, True, True],
                    [True, True, False],
                    [True, True, True],
                    [True, True, True],
                ]
            ),
        },
    )
    statistics = training_module._IdentityContributionStatistics()
    statistics.update(components, batch, "all_non_background")
    diagnostics = statistics.compute()
    assert diagnostics["multi_identity_eligible_count"] == 2
    assert diagnostics["nonzero_native_count"] == 2
    assert diagnostics["native_centered_l2_mean"] == pytest.approx(3.0 / 2.0**0.5)
    assert diagnostics["graph_increment_centered_l2_mean"] == pytest.approx(2.0**0.5)
    assert diagnostics["inventory_increment_centered_l2_mean"] == pytest.approx(
        1.0 / 2.0**0.5
    )
    assert diagnostics["graph_to_native_ratio_mean"] == pytest.approx(0.75)
    assert diagnostics["inventory_to_native_ratio_mean"] == pytest.approx(0.25)
    assert diagnostics["combined_to_native_ratio_mean"] == pytest.approx(0.5)
    wandb = training_module._identity_contribution_wandb_metrics(diagnostics)
    assert wandb[
        "identity_contribution/validation/graph_to_native_ratio_mean"
    ] == pytest.approx(0.75)
    assert wandb["identity_contribution/validation/multi_identity_eligible_count"] == 2


def test_validation_identity_contribution_omits_ratios_without_native_scale() -> None:
    """语言基线没有候选内排序尺度时，W&B 只保留支持数与可定义的绝对范数。"""

    components = cast(
        IdentityScoreComponents,
        {
            "native": torch.tensor([[2.0, 2.0]]),
            "graph_increment": torch.tensor([[0.0, 1.0]]),
            "inventory_increment": torch.tensor([[0.0, 0.0]]),
        },
    )
    batch = cast(
        MatcherLossBatch,
        {
            "class_target": torch.tensor([SMALL_MOLECULE_INDEX]),
            "identity_target": torch.tensor([0]),
            "identity_mask": torch.tensor([[True, True]]),
        },
    )
    statistics = training_module._IdentityContributionStatistics()
    statistics.update(components, batch, "all_non_background")
    diagnostics = statistics.compute()
    metrics = training_module._identity_contribution_wandb_metrics(diagnostics)
    assert diagnostics["multi_identity_eligible_count"] == 1
    assert diagnostics["nonzero_native_count"] == 0
    assert diagnostics["graph_to_native_ratio_mean"] is None
    assert "identity_contribution/validation/graph_to_native_ratio_mean" not in metrics


def test_all_foreground_end_to_end_treats_subclass_as_an_auxiliary_label() -> None:
    """五类端到端只以背景/前景为门，逐类预测由最终身份所属类别归桶。"""

    ion = CLASS_TO_INDEX["ion"]
    sugar = CLASS_TO_INDEX["sugar"]
    class_target = torch.tensor(
        [ion, SMALL_MOLECULE_INDEX, sugar, ion, 0, sugar]
    )
    class_prediction = torch.tensor(
        [sugar, ion, 0, sugar, ion, ion]
    )
    class_logits = torch.full((6, 6), -10.0)
    class_logits[torch.arange(6), class_prediction] = 10.0
    outputs = {
        "class_logits": class_logits,
        # 身份槽依次是 ion、small_molecule、sugar；第0/1个跨子类预测仍命中真实身份。
        "identity_logits": torch.tensor(
            [
                [9.0, 0.0, 0.0],
                [0.0, 9.0, 0.0],
                [0.0, 0.0, 9.0],
                [0.0, 0.0, 9.0],
                [0.0, 9.0, 0.0],
                [9.0, 0.0, 0.0],
            ]
        ),
        "ligand_area_logits": torch.full((6, 1, 1, 1), -1.0),
        "binding_area_logits": torch.full((6, 1, 1, 1), -1.0),
    }
    batch = cast(
        EvaluationBatch,
        {
            "class_target": class_target,
            "identity_target": torch.tensor([0, 1, 2, 0, -1, -1]),
            "identity_mask": torch.ones((6, 3), dtype=torch.bool),
            "identity_class_index": torch.tensor(
                [[ion, SMALL_MOLECULE_INDEX, sugar]] * 6
            ),
            "ligand_area_target": torch.zeros((6, 1, 1, 1), dtype=torch.bool),
            "binding_area_target": torch.zeros((6, 1, 1, 1), dtype=torch.bool),
            "metadata": [
                {
                    "split": "validation",
                    "pdb_id": f"pdb-{index}",
                    "candidate_index": 0,
                    "matched_object_key": None if target < 0 else f"identity-{index}",
                    "identity_seen_in_train": None if target < 0 else True,
                }
                for index, target in enumerate([0, 1, 2, 0, -1, -1])
            ],
        },
    )
    accumulator = EvaluationAccumulator(identity_scope="all_non_background", top_k=2)
    accumulator.update(outputs, batch)
    report = accumulator.compute()
    overall = report["all_non_background_identity"]["end_to_end"]
    per_class = report["identity_per_class"]

    assert report["conditional_identity"]["eligible_count"] == 0
    assert overall == {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "true_positive": 2,
        "predicted_positive": 4,
        "actual_positive": 4,
    }
    assert report["small_molecule_identity"]["end_to_end"]["true_positive"] == 0
    assert per_class["ion"]["end_to_end"]["true_positive"] == 1
    assert per_class["small_molecule"]["end_to_end"]["true_positive"] == 1
    assert per_class["sugar"]["end_to_end"]["true_positive"] == 0
    assert sum(
        metrics["end_to_end"]["true_positive"] for metrics in per_class.values()
    ) == overall["true_positive"]
    assert sum(
        metrics["end_to_end"]["predicted_positive"] for metrics in per_class.values()
    ) == overall["predicted_positive"]
    assert sum(
        metrics["end_to_end"]["actual_positive"] for metrics in per_class.values()
    ) == overall["actual_positive"]


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
            "identity_class_index": torch.empty((1, 0), dtype=torch.long),
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
        EvaluationAccumulator(identity_scope=None).update(outputs, batch)

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
            "identity_class_index": torch.empty((2, 0), dtype=torch.long),
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
        EvaluationAccumulator(identity_scope=None).update(
            area_mismatch_outputs,
            area_mismatch_batch,
        )

    identity_mismatch_batch = cast(
        EvaluationBatch,
        {
            "class_target": torch.tensor([0, 0]),
            "identity_target": torch.tensor([-1]),
            "identity_mask": torch.zeros((1, 1), dtype=torch.bool),
            "identity_class_index": torch.full((1, 1), -1, dtype=torch.long),
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
        EvaluationAccumulator(identity_scope="small_molecule").update(
            identity_mismatch_outputs,
            identity_mismatch_batch,
        )


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
            assert plan.data.local_crop_strategy == "fixed_center"
            assert stage.model.use_sampling_features is False
            assert stage.model.use_candidate_token_attention_bias is False
            assert stage.model.initial_graph_gate == 0.0
            assert stage.model.initial_inventory_gate == 0.0
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


def test_all_batch1_plans_freeze_five_class_manual_feature_comparison() -> None:
    """第1批三个 YAML 只改变身份损失和硬件微批，保持同一科学对照。"""

    config_root = Path(__file__).parents[2] / "configs" / "experiments" / "batch1"
    paths = sorted(config_root.glob("*.yaml"))
    assert len(paths) == 3
    plans = [load_experiment_plan(path) for path in paths]
    assert {plan.stages[0].identity_variant for plan in plans} == {
        "oracle",
        "detached",
        "joint",
    }
    for plan in plans:
        assert plan.output_root.is_relative_to(Path("/storage/penghongen/tmp/Matcher"))
        assert plan.tracking.project == "AdaLigand-Matcher-Batch1"
        assert plan.tracking.group == "matcher-batch1-five-class-manual-features"
        assert plan.data.train_num_workers == 4
        assert plan.data.validation_num_workers == 4
        assert plan.data.local_crop_strategy == "fixed_center"
        stage = plan.stages[0]
        assert stage.task_name == "full_matcher"
        assert stage.language_family == "smi_ted_289m"
        assert stage.identity_scope == "all_non_background"
        assert stage.model.aggregation == "mean_max"
        assert stage.model.use_manual_features is True
        assert stage.model.use_sampling_features is False
        assert stage.model.use_candidate_token_attention_bias is False
        assert stage.model.initial_graph_gate == 0.0
        assert stage.model.initial_inventory_gate == 0.0
        assert stage.training.global_batch_size == 32
        assert stage.training.maximum_epochs is None
        assert stage.training.patience_bad_epochs == 2
        expected_physical = 8 if stage.identity_variant == "joint" else 16
        expected_accumulation = 4 if stage.identity_variant == "joint" else 2
        expected_cache = 16.0 if stage.identity_variant == "joint" else 32.0
        assert stage.training.physical_batch_size == expected_physical
        assert stage.training.gradient_accumulation == expected_accumulation
        assert plan.data.cache_gib_per_worker == expected_cache


def test_all_batch2_plans_freeze_v_centroid_sampling_comparisons() -> None:
    """第2批统一新视野与特征，并冻结两个候选聚合及三个完整 Matcher 损失对照。"""

    config_root = Path(__file__).parents[2] / "configs" / "experiments" / "batch2"
    paths = sorted(config_root.glob("*.yaml"))
    assert len(paths) == 5
    plans = [load_experiment_plan(path) for path in paths]
    for plan in plans:
        assert plan.output_root.is_relative_to(Path("/storage/penghongen/tmp/Matcher"))
        assert plan.tracking.project == "AdaLigand-Matcher-Batch2"
        assert plan.data.local_crop_strategy == "v_centroid"
        assert plan.data.train_num_workers == 4
        assert plan.data.validation_num_workers == 4
        assert len(plan.stages) == 1
        stage = plan.stages[0]
        assert stage.model.use_manual_features is True
        assert stage.model.use_sampling_features is True
        assert stage.model.use_candidate_token_attention_bias is False
        assert stage.model.initial_graph_gate == 0.0
        assert stage.model.initial_inventory_gate == 0.0
        assert stage.training.global_batch_size == 32
        assert stage.training.maximum_epochs is None
        assert stage.training.patience_bad_epochs == 2

    full_plans = [plan for plan in plans if plan.stages[0].task_name == "full_matcher"]
    assert len(full_plans) == 3
    assert {plan.stages[0].identity_variant for plan in full_plans} == {
        "oracle",
        "detached",
        "joint",
    }
    for plan in full_plans:
        stage = plan.stages[0]
        assert plan.tracking.group == "matcher-batch2-full-v-centroid-sampling"
        assert plan.data.cache_gib_per_worker == 16.0
        assert stage.language_family == "smi_ted_289m"
        assert stage.identity_scope == "all_non_background"
        assert stage.model.aggregation == "mean_max"
        assert stage.training.physical_batch_size == 8
        assert stage.training.gradient_accumulation == 4

    candidate_plans = [
        plan for plan in plans if plan.stages[0].task_name == "candidate_pretrain"
    ]
    assert len(candidate_plans) == 2
    assert {plan.stages[0].model.aggregation for plan in candidate_plans} == {
        "mean_max",
        "candidate_cross_attention",
    }
    for plan in candidate_plans:
        stage = plan.stages[0]
        assert plan.tracking.group == "matcher-batch2-candidate-v-centroid-sampling"
        assert plan.data.cache_gib_per_worker == 32.0
        assert stage.language_family is None
        assert stage.identity_variant is None
        assert stage.identity_scope is None
        assert stage.training.physical_batch_size == 16
        assert stage.training.gradient_accumulation == 2
        assert stage.training.loss_weights.identity == 0.0


def test_all_batch3_plans_freeze_enhanced_attention_comparisons() -> None:
    """第3批四份 YAML 统一零初始化实体偏置，并冻结三损失与候选直接对照。"""

    config_root = Path(__file__).parents[2] / "configs" / "experiments" / "batch3"
    paths = sorted(config_root.glob("*.yaml"))
    assert len(paths) == 4
    plans = [load_experiment_plan(path) for path in paths]
    for plan in plans:
        assert plan.output_root.is_relative_to(Path("/storage/penghongen/tmp/Matcher"))
        assert plan.tracking.project == "AdaLigand-Matcher-Batch3"
        assert plan.data.local_crop_strategy == "v_centroid"
        assert plan.data.train_num_workers == 4
        assert plan.data.validation_num_workers == 4
        assert len(plan.stages) == 1
        stage = plan.stages[0]
        assert stage.model.aggregation == "candidate_cross_attention"
        assert stage.model.use_manual_features is True
        assert stage.model.use_sampling_features is True
        assert stage.model.use_candidate_token_attention_bias is True
        assert stage.model.initial_graph_gate == 0.0
        assert stage.model.initial_inventory_gate == 0.0
        assert stage.training.global_batch_size == 32
        assert stage.training.maximum_epochs is None
        assert stage.training.patience_bad_epochs == 2

    full_plans = [plan for plan in plans if plan.stages[0].task_name == "full_matcher"]
    assert len(full_plans) == 3
    assert {plan.stages[0].identity_variant for plan in full_plans} == {
        "oracle",
        "detached",
        "joint",
    }
    for plan in full_plans:
        stage = plan.stages[0]
        assert plan.tracking.group == "matcher-batch3-full-enhanced-cross-attention"
        assert plan.data.cache_gib_per_worker == 16.0
        assert stage.language_family == "smi_ted_289m"
        assert stage.identity_scope == "all_non_background"
        assert stage.training.physical_batch_size == 8
        assert stage.training.gradient_accumulation == 4

    candidate_plans = [
        plan for plan in plans if plan.stages[0].task_name == "candidate_pretrain"
    ]
    assert len(candidate_plans) == 1
    candidate_plan = candidate_plans[0]
    candidate_stage = candidate_plan.stages[0]
    assert candidate_plan.tracking.group == "matcher-batch3-candidate-enhanced-cross-attention"
    assert candidate_plan.data.cache_gib_per_worker == 32.0
    assert candidate_stage.language_family is None
    assert candidate_stage.identity_variant is None
    assert candidate_stage.identity_scope is None
    assert candidate_stage.training.physical_batch_size == 16
    assert candidate_stage.training.gradient_accumulation == 2
    assert candidate_stage.training.loss_weights.identity == 0.0

    batch2_root = Path(__file__).parents[2] / "configs" / "experiments" / "batch2"
    batch2_candidate = load_experiment_plan(
        batch2_root / "candidate_v_centroid_cross_attention_a800.yaml"
    )
    baseline_candidate_stage = batch2_candidate.stages[0]
    assert asdict(candidate_plan.data) == asdict(batch2_candidate.data)
    assert asdict(candidate_stage.training) == asdict(baseline_candidate_stage.training)
    candidate_model = candidate_stage.model.to_dict()
    baseline_candidate_model = baseline_candidate_stage.model.to_dict()
    assert candidate_model.pop("use_candidate_token_attention_bias") is True
    assert baseline_candidate_model.pop("use_candidate_token_attention_bias") is False
    assert candidate_model == baseline_candidate_model

    batch2_full_plans = [
        load_experiment_plan(path) for path in batch2_root.glob("full_*.yaml")
    ]
    for enhanced_plan in full_plans:
        enhanced_stage = enhanced_plan.stages[0]
        baseline_plan = next(
            plan
            for plan in batch2_full_plans
            if plan.stages[0].identity_variant == enhanced_stage.identity_variant
        )
        baseline_stage = baseline_plan.stages[0]
        assert asdict(enhanced_plan.data) == asdict(baseline_plan.data)
        assert asdict(enhanced_stage.training) == asdict(baseline_stage.training)
        enhanced_model = enhanced_stage.model.to_dict()
        baseline_model = baseline_stage.model.to_dict()
        assert enhanced_model.pop("aggregation") == "candidate_cross_attention"
        assert baseline_model.pop("aggregation") == "mean_max"
        assert enhanced_model.pop("use_candidate_token_attention_bias") is True
        assert baseline_model.pop("use_candidate_token_attention_bias") is False
        assert enhanced_model == baseline_model


def test_all_batch4_plans_freeze_mean_max_candidate_pretraining_initialization() -> None:
    """第4批只增加候选初始化，并锁定批次命名、W&B与A800资源适配。"""

    project_root = Path(__file__).parents[2]
    config_root = project_root / "configs" / "experiments"
    batch4_paths = sorted((config_root / "batch4").glob("*.yaml"))
    assert len(batch4_paths) == 3
    plans = [load_experiment_plan(path) for path in batch4_paths]
    source_plan = load_experiment_plan(
        config_root / "batch2" / "candidate_v_centroid_mean_max_a800.yaml"
    )
    source_stage = source_plan.stages[0]
    baseline_plans = {
        plan.stages[0].identity_variant: plan
        for plan in (
            load_experiment_plan(
                config_root / "batch2" / "full_smi_ted_v_centroid_oracle_a100.yaml"
            ),
            load_experiment_plan(
                config_root / "batch2" / "full_smi_ted_v_centroid_detached_a100.yaml"
            ),
            load_experiment_plan(
                config_root / "batch2" / "full_smi_ted_v_centroid_joint_a100.yaml"
            ),
        )
    }
    source_checkpoint = (
        "/storage/penghongen/tmp/Matcher/experiments/"
        "batch2_candidate_v_centroid_mean_max_a800/"
        "train_experiment_job340894_20260814T093012_a7/"
        "batch2_candidate_v_centroid_mean_max_a800/BEST.pt"
    )
    expected_names = {
        "oracle": "batch4_smi_ted_v_centroid_pretrained_mean_max_oracle_a100",
        "detached": "batch4_smi_ted_v_centroid_pretrained_mean_max_detached_a800",
        "joint": "batch4_smi_ted_v_centroid_pretrained_mean_max_joint_a800",
    }
    assert {plan.stages[0].identity_variant for plan in plans} == {
        "oracle",
        "detached",
        "joint",
    }
    for plan in plans:
        assert len(plan.stages) == 1
        stage = plan.stages[0]
        identity_variant = stage.identity_variant
        assert identity_variant is not None
        baseline_plan = baseline_plans[identity_variant]
        baseline_stage = baseline_plan.stages[0]
        assert plan.plan_name == expected_names[identity_variant]
        assert stage.name == plan.plan_name
        assert plan.tracking.project == "AdaLigand-Matcher-Batch4"
        assert plan.tracking.group == "matcher-batch4-pretrained-mean-max"
        target_tracking = asdict(plan.tracking)
        baseline_tracking = asdict(baseline_plan.tracking)
        for field in ("project", "group"):
            target_tracking.pop(field)
            baseline_tracking.pop(field)
        assert target_tracking == baseline_tracking
        assert plan.output_root == baseline_plan.output_root
        assert stage.task_name == "full_matcher"
        assert stage.language_family == "smi_ted_289m"
        assert stage.identity_scope == "all_non_background"
        assert stage.identity_variant == baseline_stage.identity_variant
        assert stage.model.to_dict() == source_stage.model.to_dict()
        assert stage.model.to_dict() == baseline_stage.model.to_dict()
        assert stage.model.initial_graph_gate == 0.0
        assert stage.model.initial_inventory_gate == 0.0
        assert stage.initialize_candidate_from == source_checkpoint

        target_data = asdict(plan.data)
        baseline_data = asdict(baseline_plan.data)
        target_cache_gib = target_data.pop("cache_gib_per_worker")
        baseline_data.pop("cache_gib_per_worker")
        assert target_data == baseline_data

        target_training = asdict(stage.training)
        baseline_training = asdict(baseline_stage.training)
        target_physical_batch = target_training.pop("physical_batch_size")
        target_accumulation = target_training.pop("gradient_accumulation")
        baseline_training.pop("physical_batch_size")
        baseline_training.pop("gradient_accumulation")
        assert target_training == baseline_training

        if plan.plan_name.endswith("_a100"):
            assert stage.identity_variant == "oracle"
            assert target_physical_batch == 8
            assert target_accumulation == 4
            assert target_cache_gib == 16.0
        else:
            assert plan.plan_name.endswith("_a800")
            assert stage.identity_variant in ("detached", "joint")
            assert target_physical_batch == 16
            assert target_accumulation == 2
            assert target_cache_gib == 32.0


@pytest.mark.parametrize(
    (
        "plan_name",
        "graph_gate",
        "inventory_gate",
        "clip_norm",
        "physical_batch_size",
        "gradient_accumulation",
        "cache_gib_per_worker",
        "tracking_group",
    ),
    (
        (
            "batch5_smi_ted_v_centroid_gate_002_oracle_a100",
            0.02,
            0.02,
            1.0,
            8,
            4,
            16.0,
            "matcher-batch5-gate-initialization",
        ),
        (
            "batch5_smi_ted_v_centroid_gate_010_oracle_a100",
            0.1,
            0.1,
            1.0,
            8,
            4,
            16.0,
            "matcher-batch5-gate-initialization",
        ),
        (
            "batch5_smi_ted_v_centroid_graph_gate_002_oracle_a100",
            0.02,
            0.0,
            1.0,
            8,
            4,
            16.0,
            "matcher-batch5-gate-initialization",
        ),
        (
            "batch5_smi_ted_v_centroid_inventory_gate_002_oracle_a100",
            0.0,
            0.02,
            1.0,
            8,
            4,
            16.0,
            "matcher-batch5-gate-initialization",
        ),
        (
            "batch5_smi_ted_v_centroid_clip_5_oracle_a800",
            0.0,
            0.0,
            5.0,
            16,
            2,
            32.0,
            "matcher-batch5-gradient-clipping",
        ),
        (
            "batch5_smi_ted_v_centroid_clip_200_oracle_a800",
            0.0,
            0.0,
            200.0,
            16,
            2,
            32.0,
            "matcher-batch5-gradient-clipping",
        ),
    ),
)
def test_batch5_oracle_matrix_changes_only_declared_variables(
    plan_name: str,
    graph_gate: float,
    inventory_gate: float,
    clip_norm: float,
    physical_batch_size: int,
    gradient_accumulation: int,
    cache_gib_per_worker: float,
    tracking_group: str,
) -> None:
    """第 5 批六项锁定 0.02 四格、0.1 尺度探针和独立裁剪对照的声明变量。"""

    config_root = Path(__file__).parents[2] / "configs" / "experiments"
    baseline = load_experiment_plan(
        config_root / "batch2" / "full_smi_ted_v_centroid_oracle_a100.yaml"
    )
    target = load_experiment_plan(config_root / "batch5" / f"{plan_name}.yaml")
    assert len(target.stages) == 1
    baseline_stage = baseline.stages[0]
    target_stage = target.stages[0]
    assert target.plan_name == target_stage.name == plan_name
    assert target.output_root == baseline.output_root
    assert target_stage.task_name == baseline_stage.task_name == "full_matcher"
    assert target_stage.language_family == baseline_stage.language_family == "smi_ted_289m"
    assert target_stage.identity_variant == baseline_stage.identity_variant == "oracle"
    assert target_stage.identity_scope == baseline_stage.identity_scope == "all_non_background"
    assert target_stage.initialize_candidate_from is None
    assert baseline_stage.initialize_candidate_from is None

    target_data = asdict(target.data)
    baseline_data = asdict(baseline.data)
    assert target_data.pop("cache_gib_per_worker") == cache_gib_per_worker
    baseline_data.pop("cache_gib_per_worker")
    assert target_data == baseline_data

    target_model = target_stage.model.to_dict()
    baseline_model = baseline_stage.model.to_dict()
    assert target_model.pop("initial_graph_gate") == graph_gate
    assert target_model.pop("initial_inventory_gate") == inventory_gate
    assert baseline_model.pop("initial_graph_gate") == 0.0
    assert baseline_model.pop("initial_inventory_gate") == 0.0
    assert target_model == baseline_model

    target_training = asdict(target_stage.training)
    baseline_training = asdict(baseline_stage.training)
    assert target_training.pop("physical_batch_size") == physical_batch_size
    assert target_training.pop("gradient_accumulation") == gradient_accumulation
    assert target_training.pop("gradient_clip_norm") == clip_norm
    baseline_training.pop("physical_batch_size")
    baseline_training.pop("gradient_accumulation")
    assert baseline_training.pop("gradient_clip_norm") == 1.0
    assert target_training == baseline_training
    assert target_stage.training.global_batch_size == 32

    target_tracking = asdict(target.tracking)
    baseline_tracking = asdict(baseline.tracking)
    assert target_tracking.pop("project") == "AdaLigand-Matcher-Batch5"
    assert target_tracking.pop("group") == tracking_group
    baseline_tracking.pop("project")
    baseline_tracking.pop("group")
    assert target_tracking == baseline_tracking


def test_joint_identity_v4_zero_gate_config_matches_batch2_oracle() -> None:
    """v4 零门计划的解析配置仅在任务命名和 W&B 隔离字段上不同。"""

    config_root = Path(__file__).parents[2] / "configs" / "experiments"
    baseline = load_experiment_plan(
        config_root / "batch2" / "full_smi_ted_v_centroid_oracle_a100.yaml"
    )
    target = load_experiment_plan(
        config_root
        / "joint_identity_v4"
        / "joint_identity_v4_zero_gate_oracle_a100.yaml"
    )
    assert len(baseline.stages) == len(target.stages) == 1
    baseline_stage = baseline.stages[0]
    target_stage = target.stages[0]
    assert baseline.plan_name == baseline_stage.name == "batch2_smi_ted_v_centroid_oracle_a100"
    assert target.plan_name == target_stage.name == "joint_identity_v4_zero_gate_oracle_a100"
    assert target.output_root == baseline.output_root
    assert target.data == baseline.data
    assert target_stage.task_name == baseline_stage.task_name == "full_matcher"
    assert target_stage.language_family == baseline_stage.language_family == "smi_ted_289m"
    assert target_stage.identity_variant == baseline_stage.identity_variant == "oracle"
    assert target_stage.identity_scope == baseline_stage.identity_scope == "all_non_background"
    assert target_stage.model == baseline_stage.model
    assert target_stage.training == baseline_stage.training
    assert target_stage.initialize_candidate_from is baseline_stage.initialize_candidate_from is None

    target_tracking = asdict(target.tracking)
    baseline_tracking = asdict(baseline.tracking)
    assert target_tracking.pop("project") == "AdaLigand-Matcher-JointIdentityV4"
    assert target_tracking.pop("group") == "joint-identity-v4-zero-gate-baseline"
    baseline_tracking.pop("project")
    baseline_tracking.pop("group")
    assert target_tracking == baseline_tracking


def test_joint_identity_v4_concat_anchor_changes_only_aggregation_and_run_names() -> None:
    """A800 摘要拼接锚点只改变聚合结构与运行隔离字段。"""

    config_directory = (
        Path(__file__).parents[2] / "configs" / "experiments" / "joint_identity_v4"
    )
    baseline = load_experiment_plan(
        config_directory / "joint_identity_v4_zero_gate_oracle_a100.yaml"
    )
    target = load_experiment_plan(
        config_directory
        / "joint_identity_v4_concat_zero_gate_seed_3407_lr_1e4_oracle_a800.yaml"
    )
    assert len(baseline.stages) == len(target.stages) == 1
    baseline_stage = baseline.stages[0]
    target_stage = target.stages[0]
    expected_name = "joint_identity_v4_concat_zero_gate_seed_3407_lr_1e4_oracle_a800"
    assert target.plan_name == target_stage.name == expected_name
    assert target.output_root == baseline.output_root
    assert target.data == baseline.data
    assert target_stage.task_name == baseline_stage.task_name == "full_matcher"
    assert target_stage.language_family == baseline_stage.language_family == "smi_ted_289m"
    assert target_stage.identity_variant == baseline_stage.identity_variant == "oracle"
    assert target_stage.identity_scope == baseline_stage.identity_scope == "all_non_background"
    assert target_stage.initialize_candidate_from is baseline_stage.initialize_candidate_from is None

    baseline_model = baseline_stage.model.to_dict()
    target_model = target_stage.model.to_dict()
    assert baseline_model.pop("aggregation") == "mean_max"
    assert target_model.pop("aggregation") == "mean_max_candidate_cross_attention_concat"
    assert target_model == baseline_model
    assert target_stage.training == baseline_stage.training
    assert target_stage.training.seed == 3407
    assert target_stage.training.learning_rate == 1.0e-4
    assert target_stage.training.physical_batch_size == 8
    assert target_stage.training.gradient_accumulation == 4
    assert target_stage.training.global_batch_size == 32
    assert target_stage.training.gradient_clip_norm == 1.0

    baseline_tracking = asdict(baseline.tracking)
    target_tracking = asdict(target.tracking)
    baseline_tracking.pop("project")
    baseline_tracking.pop("group")
    assert target_tracking.pop("project") == "AdaLigand-Matcher-JointIdentityV4-SummaryConcat"
    assert target_tracking.pop("group") == "joint-identity-v4-summary-concat-seed-3407"
    assert target_tracking == baseline_tracking


@pytest.mark.parametrize(
    (
        "file_name",
        "training_seed",
        "graph_gate",
        "inventory_gate",
        "clip_norm",
        "tracking_group",
    ),
    (
        (
            "joint_identity_v4_concat_zero_gate_seed_3407_lr_1e4_oracle_a800.yaml",
            3407,
            0.0,
            0.0,
            1.0,
            "joint-identity-v4-summary-concat-seed-3407",
        ),
        (
            "joint_identity_v4_concat_zero_gate_clip_200_seed_3407_lr_1e4_oracle_a800.yaml",
            3407,
            0.0,
            0.0,
            200.0,
            "joint-identity-v4-summary-concat-seed-3407",
        ),
        (
            "joint_identity_v4_concat_gate_002_seed_3407_lr_1e4_oracle_a800.yaml",
            3407,
            0.02,
            0.02,
            1.0,
            "joint-identity-v4-summary-concat-seed-3407",
        ),
        (
            "joint_identity_v4_concat_zero_gate_seed_3408_lr_1e4_oracle_a100.yaml",
            3408,
            0.0,
            0.0,
            1.0,
            "joint-identity-v4-summary-concat-a100-seed-3408",
        ),
        (
            "joint_identity_v4_concat_gate_002_seed_3408_lr_1e4_oracle_a100.yaml",
            3408,
            0.02,
            0.02,
            1.0,
            "joint-identity-v4-summary-concat-a100-seed-3408",
        ),
        (
            "joint_identity_v4_concat_zero_gate_clip_200_seed_3408_lr_1e4_oracle_a100.yaml",
            3408,
            0.0,
            0.0,
            200.0,
            "joint-identity-v4-summary-concat-a100-seed-3408",
        ),
        (
            "joint_identity_v4_concat_gate_002_clip_200_seed_3408_lr_1e4_oracle_a100.yaml",
            3408,
            0.02,
            0.02,
            200.0,
            "joint-identity-v4-summary-concat-a100-seed-3408",
        ),
    ),
)
def test_joint_identity_v4_concat_matrix_changes_only_declared_axes(
    file_name: str,
    training_seed: int,
    graph_gate: float,
    inventory_gate: float,
    clip_norm: float,
    tracking_group: str,
) -> None:
    """七项摘要拼接配置只改变聚合结构、两个门、裁剪、训练种子与运行命名。"""

    config_directory = (
        Path(__file__).parents[2] / "configs" / "experiments" / "joint_identity_v4"
    )
    baseline = load_experiment_plan(
        config_directory / "joint_identity_v4_zero_gate_oracle_a100.yaml"
    )
    target = load_experiment_plan(config_directory / file_name)
    assert len(baseline.stages) == len(target.stages) == 1
    baseline_stage = baseline.stages[0]
    target_stage = target.stages[0]
    expected_name = Path(file_name).stem
    assert target.plan_name == target_stage.name == expected_name
    assert target.output_root == baseline.output_root
    assert target.data == baseline.data
    assert target_stage.task_name == baseline_stage.task_name == "full_matcher"
    assert target_stage.language_family == baseline_stage.language_family == "smi_ted_289m"
    assert target_stage.identity_variant == baseline_stage.identity_variant == "oracle"
    assert target_stage.identity_scope == baseline_stage.identity_scope == "all_non_background"
    assert target_stage.initialize_candidate_from is baseline_stage.initialize_candidate_from is None

    baseline_model = baseline_stage.model.to_dict()
    target_model = target_stage.model.to_dict()
    assert baseline_model.pop("aggregation") == "mean_max"
    assert target_model.pop("aggregation") == "mean_max_candidate_cross_attention_concat"
    assert target_model.pop("initial_graph_gate") == graph_gate
    assert target_model.pop("initial_inventory_gate") == inventory_gate
    baseline_model.pop("initial_graph_gate")
    baseline_model.pop("initial_inventory_gate")
    assert target_model == baseline_model

    baseline_training = asdict(baseline_stage.training)
    target_training = asdict(target_stage.training)
    assert target_training.pop("seed") == training_seed
    assert target_training.pop("gradient_clip_norm") == clip_norm
    baseline_training.pop("seed")
    baseline_training.pop("gradient_clip_norm")
    assert target_training == baseline_training
    assert target_stage.training.learning_rate == 1.0e-4
    assert target_stage.training.physical_batch_size == 8
    assert target_stage.training.gradient_accumulation == 4
    assert target_stage.training.global_batch_size == 32

    baseline_tracking = asdict(baseline.tracking)
    target_tracking = asdict(target.tracking)
    baseline_tracking.pop("project")
    baseline_tracking.pop("group")
    assert target_tracking.pop("project") == "AdaLigand-Matcher-JointIdentityV4-SummaryConcat"
    assert target_tracking.pop("group") == tracking_group
    assert target_tracking == baseline_tracking


def test_joint_identity_v4_concat_a800_gate_probe_changes_only_gates_and_names() -> None:
    """A800 双门 0.02 任务相对同卡零门任务只改变两个门和运行命名。"""

    config_directory = (
        Path(__file__).parents[2] / "configs" / "experiments" / "joint_identity_v4"
    )
    baseline = load_experiment_plan(
        config_directory
        / "joint_identity_v4_concat_zero_gate_seed_3407_lr_1e4_oracle_a800.yaml"
    )
    target = load_experiment_plan(
        config_directory
        / "joint_identity_v4_concat_gate_002_seed_3407_lr_1e4_oracle_a800.yaml"
    )
    assert len(baseline.stages) == len(target.stages) == 1
    target_stage = target.stages[0]
    expected_name = "joint_identity_v4_concat_gate_002_seed_3407_lr_1e4_oracle_a800"
    assert target.plan_name == target_stage.name == expected_name

    baseline_structure = asdict(baseline)
    target_structure = asdict(target)
    baseline_structure["plan_name"] = target_structure["plan_name"]
    baseline_structure["stages"][0]["name"] = target_structure["stages"][0]["name"]
    baseline_structure["stages"][0]["model"]["initial_graph_gate"] = 0.02
    baseline_structure["stages"][0]["model"]["initial_inventory_gate"] = 0.02
    assert target_structure == baseline_structure


def test_joint_identity_v4_learning_rate_matrix_has_three_seed_pairs() -> None:
    """三个联合身份 v4 随机种子配对只改变学习率、种子和隔离命名，并保持计划名全局唯一。"""

    config_directory = (
        Path(__file__).parents[2] / "configs" / "experiments" / "joint_identity_v4"
    )
    pairs = (
        (
            3407,
            "joint_identity_v4_zero_gate_oracle_a100.yaml",
            "joint_identity_v4_lr_2e4_oracle_a100.yaml",
            "joint-identity-v4-zero-gate-baseline",
            "joint-identity-v4-learning-rate",
        ),
        (
            3408,
            "joint_identity_v4_seed_3408_lr_1e4_oracle_a100.yaml",
            "joint_identity_v4_seed_3408_lr_2e4_oracle_a100.yaml",
            "joint-identity-v4-learning-rate-seed-3408",
            "joint-identity-v4-learning-rate-seed-3408",
        ),
        (
            3409,
            "joint_identity_v4_seed_3409_lr_1e4_oracle_a100.yaml",
            "joint_identity_v4_seed_3409_lr_2e4_oracle_a100.yaml",
            "joint-identity-v4-learning-rate-seed-3409",
            "joint-identity-v4-learning-rate-seed-3409",
        ),
    )
    normalized_pairs: list[dict[str, object]] = []

    for seed, lr_1e4_file, lr_2e4_file, lr_1e4_group, lr_2e4_group in pairs:
        lr_1e4 = load_experiment_plan(config_directory / lr_1e4_file)
        lr_2e4 = load_experiment_plan(config_directory / lr_2e4_file)
        assert len(lr_1e4.stages) == len(lr_2e4.stages) == 1
        lr_1e4_stage = lr_1e4.stages[0]
        lr_2e4_stage = lr_2e4.stages[0]
        assert lr_1e4.plan_name == lr_1e4_stage.name == Path(lr_1e4_file).stem
        assert lr_2e4.plan_name == lr_2e4_stage.name == Path(lr_2e4_file).stem
        assert lr_1e4.output_root == lr_2e4.output_root
        assert lr_1e4.data == lr_2e4.data
        assert lr_1e4.data.selection_seed == lr_2e4.data.selection_seed == 3407
        assert lr_1e4.data.roots.manifest_root.name == "closed_set_v4"
        assert lr_2e4.data.roots.manifest_root.name == "closed_set_v4"
        assert lr_1e4_stage.task_name == lr_2e4_stage.task_name == "full_matcher"
        assert lr_1e4_stage.language_family == lr_2e4_stage.language_family == "smi_ted_289m"
        assert lr_1e4_stage.identity_variant == lr_2e4_stage.identity_variant == "oracle"
        assert lr_1e4_stage.identity_scope == lr_2e4_stage.identity_scope == "all_non_background"
        assert lr_1e4_stage.model == lr_2e4_stage.model
        assert lr_1e4_stage.model.initial_graph_gate == 0.0
        assert lr_1e4_stage.model.initial_inventory_gate == 0.0
        assert lr_1e4_stage.initialize_candidate_from is lr_2e4_stage.initialize_candidate_from is None

        lr_1e4_training = asdict(lr_1e4_stage.training)
        lr_2e4_training = asdict(lr_2e4_stage.training)
        assert lr_1e4_training.pop("learning_rate") == 1.0e-4
        assert lr_2e4_training.pop("learning_rate") == 2.0e-4
        assert lr_1e4_training == lr_2e4_training
        assert lr_1e4_stage.training.seed == lr_2e4_stage.training.seed == seed
        assert lr_1e4_stage.training.physical_batch_size == 8
        assert lr_1e4_stage.training.gradient_accumulation == 4
        assert lr_1e4_stage.training.global_batch_size == 32
        assert lr_1e4_stage.training.gradient_clip_norm == 1.0

        lr_1e4_tracking = asdict(lr_1e4.tracking)
        lr_2e4_tracking = asdict(lr_2e4.tracking)
        assert lr_1e4_tracking.pop("project") == "AdaLigand-Matcher-JointIdentityV4"
        assert lr_2e4_tracking.pop("project") == "AdaLigand-Matcher-JointIdentityV4"
        assert lr_1e4_tracking.pop("group") == lr_1e4_group
        assert lr_2e4_tracking.pop("group") == lr_2e4_group
        assert lr_1e4_tracking == lr_2e4_tracking

        normalized_data = asdict(lr_1e4.data)
        normalized_training = dict(lr_1e4_training)
        normalized_training.pop("seed")
        normalized_pairs.append(
            {
                "output_root": lr_1e4.output_root,
                "data": normalized_data,
                "task": (
                    lr_1e4_stage.task_name,
                    lr_1e4_stage.language_family,
                    lr_1e4_stage.identity_variant,
                    lr_1e4_stage.identity_scope,
                ),
                "model": lr_1e4_stage.model.to_dict(),
                "training_without_seed_or_learning_rate": normalized_training,
                "initialize_candidate_from": lr_1e4_stage.initialize_candidate_from,
                "tracking_without_project_or_group": lr_1e4_tracking,
            }
        )

    assert normalized_pairs[1:] == [normalized_pairs[0], normalized_pairs[0]]
    all_plans = [
        load_experiment_plan(path)
        for path in sorted(config_directory.glob("*.yaml"))
    ]
    assert all(len(plan.stages) == 1 for plan in all_plans)
    all_plan_names = [plan.plan_name for plan in all_plans]
    all_stage_names = [plan.stages[0].name for plan in all_plans]
    assert all_plan_names == all_stage_names
    assert len(all_plan_names) == len(set(all_plan_names))


@pytest.mark.parametrize(
    "missing_key",
    (
        "use_manual_features",
        "use_sampling_features",
        "use_candidate_token_attention_bias",
        "initial_graph_gate",
        "initial_inventory_gate",
    ),
)
def test_formal_yaml_requires_all_model_scientific_fields(
    tmp_path: Path,
    missing_key: str,
) -> None:
    """正式 YAML 缺任一特征开关或身份门初值都必须失败，旧产物迁移不得进入计划解析。"""

    source = (
        Path(__file__).parents[2]
        / "configs"
        / "experiments"
        / "batch2"
        / "full_smi_ted_v_centroid_oracle_a100.yaml"
    )
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    del values["stages"][0]["model"][missing_key]
    invalid = tmp_path / f"missing_{missing_key}.yaml"
    invalid.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"stage\.model 字段不一致.*{missing_key}"):
        load_experiment_plan(invalid)


def test_experiment_config_rejects_output_escape_and_non_best_initialization(
    tmp_path: Path,
) -> None:
    """越界输出、相对或 LAST 来源，以及候选预训练继承参数，都必须在启动前失败。"""

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
    with pytest.raises(ValueError, match="绝对路径"):
        load_experiment_plan(relative)

    values["stages"][0]["initialize_candidate_from"] = (
        "/storage/penghongen/tmp/Matcher/checkpoints/LAST.pt"
    )
    last = tmp_path / "last.yaml"
    last.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="BEST.pt"):
        load_experiment_plan(last)

    candidate_source = (
        Path(__file__).parents[2]
        / "configs/experiments/batch3/candidate_enhanced_cross_attention_a800.yaml"
    )
    candidate_values = yaml.safe_load(candidate_source.read_text(encoding="utf-8"))
    candidate_values["stages"][0]["initialize_candidate_from"] = (
        "/storage/penghongen/tmp/Matcher/checkpoints/BEST.pt"
    )
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(yaml.safe_dump(candidate_values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="只允许用于 full_matcher"):
        load_experiment_plan(candidate)

    candidate_values = yaml.safe_load(candidate_source.read_text(encoding="utf-8"))
    candidate_values["stages"][0]["model"]["initial_graph_gate"] = 0.02
    nonzero_candidate_gate = tmp_path / "nonzero_candidate_gate.yaml"
    nonzero_candidate_gate.write_text(
        yaml.safe_dump(candidate_values, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="candidate_pretrain.*两个初始身份门必须为 0"):
        load_experiment_plan(nonzero_candidate_gate)


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
    historical_model_config = stage.model.to_dict()
    historical_model_config.pop("initial_graph_gate")
    historical_model_config.pop("initial_inventory_gate")
    stage_record = asdict(stage)
    stage_record["model"] = historical_model_config
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
        "model_config": historical_model_config,
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
    assert evidence["model_config"]["initial_graph_gate"] == 0.0
    assert evidence["model_config"]["initial_inventory_gate"] == 0.0
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
    resolved_values["data"]["local_crop_strategy"] = "v_centroid"
    resolved_path.write_text(json.dumps(resolved_values), encoding="utf-8")
    with pytest.raises(ValueError, match="resolved.json.data 科学字段"):
        _validate_checkpoint_for_stage(checkpoint_path, checkpoint, plan, stage)
    resolved_values["data"]["local_crop_strategy"] = plan.data.local_crop_strategy
    resolved_path.write_text(json.dumps(resolved_values), encoding="utf-8")
    restored_evidence = _validate_checkpoint_for_stage(
        checkpoint_path,
        checkpoint,
        plan,
        stage,
    )
    assert restored_evidence["epoch"] == 7
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
                "split_pdb_rule": (
                    "同一小写 pdb_id 的候选记录不得同时发布到多个划分；"
                    "仅出现在 excluded_pdb.jsonl 的整 PDB 排除记录不构成候选重叠。"
                ),
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
    manifest_values["split_pdb_rule"] = "被篡改的划分互斥规则"
    manifest_path.write_text(json.dumps(manifest_values), encoding="utf-8")
    changed_rule_data = _calibration_data_evidence(manifest_path)
    assert changed_rule_data["manifest_sha256"] != rebuilt_data["manifest_sha256"]
    assert changed_rule_data["fingerprint"] != rebuilt_data["fingerprint"]
    manifest_values["split_pdb_rule"] = (
        "同一小写 pdb_id 的候选记录不得同时发布到多个划分；"
        "仅出现在 excluded_pdb.jsonl 的整 PDB 排除记录不构成候选重叠。"
    )
    manifest_path.write_text(json.dumps(manifest_values), encoding="utf-8")
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


def test_initial_checkpoint_resolver_accepts_only_posix_absolute_best_path() -> None:
    """运行时解析器只接受 validation 选定的外部 ``BEST.pt`` 绝对路径。"""

    external = _resolve_initial_checkpoint(
        "/storage/penghongen/tmp/Matcher/checkpoints/BEST.pt",
    )
    assert external.as_posix() == "/storage/penghongen/tmp/Matcher/checkpoints/BEST.pt"
    with pytest.raises(ValueError, match="BEST.pt.*绝对路径"):
        _resolve_initial_checkpoint("checkpoints/BEST.pt")
    with pytest.raises(ValueError, match="BEST.pt.*绝对路径"):
        _resolve_initial_checkpoint(
            "/storage/penghongen/tmp/Matcher/checkpoints/LAST.pt",
        )
    with pytest.raises(ValueError, match="绝对路径"):
        _resolve_initial_checkpoint("stage:candidate_weak:BEST")


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
                "identity_class_index": torch.full(
                    (end - start, 3),
                    4,
                    dtype=torch.long,
                ),
                "ligand_area_target": area_target[start:end],
                "binding_area_target": area_target[start:end],
            },
        )
        return compute_matcher_loss(
            outputs,
            batch,
            weights,
            identity_variant="joint",
            identity_scope="small_molecule",
            class_weights=None,
        )

    global_loss = calculate(0, 4)["total"]
    first = calculate(0, 2)
    second = calculate(2, 4)
    accumulated = _accumulation_backward_loss(first, weights, 0.5, 4.0, 2)
    accumulated = accumulated + _accumulation_backward_loss(second, weights, 0.5, 4.0, 2)
    assert torch.allclose(accumulated, global_loss, atol=1e-7, rtol=1e-7)
