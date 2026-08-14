"""验证第 4 批只从已完成、科学配置兼容的候选预训练 BEST 初始化完整 Matcher。"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
import yaml  # type: ignore[import-untyped]
from torch import nn

from matcher.training.checkpointing import CHECKPOINT_SCHEMA_VERSION, save_checkpoint
from matcher.training.config import DataConfig, StageConfig, load_experiment_plan
from matcher.training.initialization import initialize_candidate_encoder_from_pretraining
from matcher.utils.json_artifacts import write_json


class _FullModelStub(nn.Module):
    """用最小候选子模块和非候选哨兵隔离测试初始化编排。"""

    def __init__(self) -> None:
        super().__init__()
        self.candidate_encoder = nn.Linear(3, 2)
        self.identity_sentinel = nn.Linear(2, 2)


def _batch3_plans() -> tuple[Path, StageConfig, DataConfig, StageConfig]:
    """返回第 3 批候选 YAML、来源阶段、共享数据配置和完整 Matcher 阶段。"""

    config_root = Path(__file__).parents[2] / "configs/experiments/batch3"
    source_path = config_root / "candidate_enhanced_cross_attention_a800.yaml"
    source_plan = load_experiment_plan(source_path)
    target_plan = load_experiment_plan(
        config_root / "full_smi_ted_enhanced_cross_attention_oracle_a100.yaml"
    )
    return source_path, source_plan.stages[0], source_plan.data, target_plan.stages[0]


def _batch2_plans() -> tuple[Path, StageConfig, DataConfig, StageConfig]:
    """返回第 2 批 mean/max 候选来源与同结构完整 Matcher 目标。"""

    config_root = Path(__file__).parents[2] / "configs/experiments/batch2"
    source_path = config_root / "candidate_v_centroid_mean_max_a800.yaml"
    source_plan = load_experiment_plan(source_path)
    target_plan = load_experiment_plan(
        config_root / "full_smi_ted_v_centroid_oracle_a100.yaml"
    )
    return source_path, source_plan.stages[0], source_plan.data, target_plan.stages[0]


def _batch0_plans() -> tuple[Path, StageConfig, DataConfig, StageConfig]:
    """返回固定中心、无手动或采样特征的历史字段迁移对照。"""

    config_root = Path(__file__).parents[2] / "configs/experiments/initial"
    source_path = config_root / "candidate_pretrain_pair.yaml"
    source_plan = load_experiment_plan(source_path)
    target_plan = load_experiment_plan(
        config_root / "full_scratch_smi_ted_light_289m_oracle.yaml"
    )
    return source_path, source_plan.stages[0], source_plan.data, target_plan.stages[0]


def _write_completed_candidate_run(
    root: Path,
    source_plan_path: Path,
    source_stage: StageConfig,
    source_data: DataConfig,
    source_model: nn.Linear,
    *,
    checkpoint_updates: dict[str, object],
    data_updates: dict[str, object],
    model_config_omissions: tuple[str, ...],
    stage_field_omissions: tuple[str, ...],
    data_field_omissions: tuple[str, ...],
    publish_run_summary: bool,
) -> Path:
    """只构造初始化器直接核对的合成训练产物；具名参数用于单项反例。"""

    run_directory = root / "20260814T120000Z_a1"
    stage_directory = run_directory / source_stage.name
    stage_directory.mkdir(parents=True)
    checkpoint_path = stage_directory / "BEST.pt"
    training_config = asdict(source_stage.training)
    epoch_record = {
        "epoch": 2,
        "global_step": 120,
        "selection_score": 0.53,
        "improved": True,
    }
    model_config = source_stage.model.to_dict()
    for name in model_config_omissions:
        model_config.pop(name)
    checkpoint: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "stage_name": source_stage.name,
        "task_name": source_stage.task_name,
        "epoch": 2,
        "global_step": 120,
        "best_epoch": 2,
        "best_score": 0.53,
        "model_config": model_config,
        "training_config": training_config,
        "model_state": source_model.state_dict(),
        "epoch_record": epoch_record,
    }
    checkpoint.update(checkpoint_updates)
    save_checkpoint(checkpoint_path, checkpoint)
    save_checkpoint(stage_directory / "LAST.pt", checkpoint)

    stage_record = asdict(source_stage)
    stage_record["model"] = model_config
    for name in stage_field_omissions:
        stage_record.pop(name)
    data_record: dict[str, object] = {
        "data_root": str(source_data.roots.data_root),
        "stage1_root": str(source_data.roots.stage1_root),
        "output_root": str(source_data.roots.output_root),
        "coverage_threshold": source_data.coverage_threshold,
        "selection_seed": source_data.selection_seed,
        "pp_top_k": source_data.pp_top_k,
        "local_crop_strategy": source_data.local_crop_strategy,
        "train_num_workers": source_data.train_num_workers,
        "validation_num_workers": source_data.validation_num_workers,
        "prefetch_factor": source_data.prefetch_factor,
        "cache_gib_per_worker": source_data.cache_gib_per_worker,
    }
    data_record.update(data_updates)
    for name in data_field_omissions:
        data_record.pop(name)
    source_plan_values = yaml.safe_load(source_plan_path.read_text(encoding="utf-8"))
    if not isinstance(source_plan_values, dict):
        raise TypeError("测试来源 YAML 顶层必须是映射。")
    source_plan_name = source_plan_values["plan_name"]
    source_tracking = source_plan_values["tracking"]
    if not isinstance(source_plan_name, str) or not isinstance(source_tracking, dict):
        raise TypeError("测试来源 YAML 必须提供 plan_name 与 tracking。")
    write_json(
        stage_directory / "resolved.json",
        {
            "stage": stage_record,
            "data": data_record,
            "tracking": source_tracking,
        },
    )

    stage_summary = {
        "stage": source_stage.name,
        "best_epoch": 2,
        "best_score": 0.53,
        "last_epoch": 4,
        "global_step": 200,
        "stopped_by_patience": True,
        "best_checkpoint": checkpoint_path.resolve().as_posix(),
        "last_checkpoint": (stage_directory / "LAST.pt").resolve().as_posix(),
        "tracking_run_id": "candidate-run-id",
        "tracking_url": "https://wandb.ai/entity/project/runs/candidate-run-id",
    }
    write_json(stage_directory / "summary.json", stage_summary)
    metadata = {
        "plan_path": source_plan_path.resolve().as_posix(),
        "plan_name": source_plan_name,
        "run_stamp": run_directory.name,
        "hostname": "synthetic-node",
        "slurm_job_id": "341999",
        "task_launch_dir": "/home/penghongen/Feedback/task_launcher/launches/341999/example",
        "tracking": source_tracking,
        "started_at_utc": "2026-08-14T12:00:00+00:00",
    }
    write_json(run_directory / "run.json", metadata)
    if publish_run_summary:
        write_json(
            run_directory / "summary.json",
            {
                **metadata,
                "finished_at_utc": "2026-08-14T13:00:00+00:00",
                "stages": [stage_summary],
            },
        )
    return checkpoint_path


def _target_stage_with_checkpoint(stage: StageConfig, checkpoint_path: Path) -> StageConfig:
    """让测试目标阶段显式引用刚构造的来源 BEST。"""

    return replace(
        stage,
        initialize_candidate_from=checkpoint_path.resolve().as_posix(),
    )


def test_candidate_initialization_loads_only_candidate_encoder_from_completed_best(
    tmp_path: Path,
) -> None:
    """成功路径只加载候选编码器，并记录已完成来源的语义证据。"""

    source_plan_path, source_stage, source_data, target_stage = _batch3_plans()
    target_stage = replace(
        target_stage,
        model=replace(target_stage.model, graph_layers=target_stage.model.graph_layers + 1),
    )
    torch.manual_seed(4)
    source_model = nn.Linear(3, 2)
    checkpoint_path = _write_completed_candidate_run(
        tmp_path / "source",
        source_plan_path,
        source_stage,
        source_data,
        source_model,
        checkpoint_updates={},
        data_updates={},
        model_config_omissions=(),
        stage_field_omissions=(),
        data_field_omissions=(),
        publish_run_summary=True,
    )
    target_stage = _target_stage_with_checkpoint(target_stage, checkpoint_path)
    target_model = _FullModelStub()
    identity_before = {
        name: value.detach().clone()
        for name, value in target_model.identity_sentinel.state_dict().items()
    }
    rng_before = torch.random.get_rng_state().clone()

    record = initialize_candidate_encoder_from_pretraining(
        target_model,
        checkpoint_path,
        target_stage,
        source_data,
    )

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert record["source_plan_name"] == load_experiment_plan(source_plan_path).plan_name
    assert record["source_stage_name"] == source_stage.name
    assert record["source_slurm_job_id"] == "341999"
    assert record["source_best_score"] == 0.53
    assert record["missing_keys"] == []
    assert record["unexpected_keys"] == []
    assert "graph_layers" not in record["source_candidate_model_config"]
    for expected, actual in zip(
        source_model.parameters(),
        target_model.candidate_encoder.parameters(),
        strict=True,
    ):
        assert torch.equal(expected, actual)
    for name, value in target_model.identity_sentinel.state_dict().items():
        assert torch.equal(value, identity_before[name])


def test_candidate_initialization_migrates_missing_historical_fields(tmp_path: Path) -> None:
    """字段缺失时迁移三个模型开关、两个身份门、固定中心裁剪和空身份范围。"""

    source_plan_path, source_stage, source_data, target_stage = _batch0_plans()
    historical_values = yaml.safe_load(source_plan_path.read_text(encoding="utf-8"))
    historical_stage = historical_values["stages"][0]
    for name in (
        "use_manual_features",
        "use_sampling_features",
        "use_candidate_token_attention_bias",
        "initial_graph_gate",
        "initial_inventory_gate",
    ):
        historical_stage["model"].pop(name)
    historical_stage.pop("identity_scope")
    historical_values["data"].pop("local_crop_strategy")
    historical_plan_path = tmp_path / "historical_candidate.yaml"
    historical_plan_path.write_text(
        yaml.safe_dump(historical_values, sort_keys=False),
        encoding="utf-8",
    )
    checkpoint_path = _write_completed_candidate_run(
        tmp_path / "source",
        historical_plan_path,
        source_stage,
        source_data,
        nn.Linear(3, 2),
        checkpoint_updates={},
        data_updates={},
        model_config_omissions=(
            "use_manual_features",
            "use_sampling_features",
            "use_candidate_token_attention_bias",
            "initial_graph_gate",
            "initial_inventory_gate",
        ),
        stage_field_omissions=("identity_scope",),
        data_field_omissions=("local_crop_strategy",),
        publish_run_summary=True,
    )
    target_stage = _target_stage_with_checkpoint(target_stage, checkpoint_path)

    record = initialize_candidate_encoder_from_pretraining(
        _FullModelStub(),
        checkpoint_path,
        target_stage,
        source_data,
    )

    assert record["source_candidate_model_config"]["use_manual_features"] is False
    assert record["source_candidate_model_config"]["use_sampling_features"] is False
    assert record["source_candidate_model_config"]["use_candidate_token_attention_bias"] is False
    assert record["source_scientific_data"]["local_crop_strategy"] == "fixed_center"


def test_candidate_initialization_migrates_batch2_attention_and_identity_gates(
    tmp_path: Path,
) -> None:
    """已有手动与采样统计的第 2 批来源迁移缺失的注意力偏置开关和两个身份门。"""

    source_plan_path, source_stage, source_data, target_stage = _batch2_plans()
    historical_values = yaml.safe_load(source_plan_path.read_text(encoding="utf-8"))
    historical_model = historical_values["stages"][0]["model"]
    for name in (
        "use_candidate_token_attention_bias",
        "initial_graph_gate",
        "initial_inventory_gate",
    ):
        del historical_model[name]
    historical_plan_path = tmp_path / "historical_batch2.yaml"
    historical_plan_path.write_text(
        yaml.safe_dump(historical_values, sort_keys=False),
        encoding="utf-8",
    )
    checkpoint_path = _write_completed_candidate_run(
        tmp_path / "source",
        historical_plan_path,
        source_stage,
        source_data,
        nn.Linear(3, 2),
        checkpoint_updates={},
        data_updates={},
        model_config_omissions=(
            "use_candidate_token_attention_bias",
            "initial_graph_gate",
            "initial_inventory_gate",
        ),
        stage_field_omissions=(),
        data_field_omissions=(),
        publish_run_summary=True,
    )
    target_stage = _target_stage_with_checkpoint(target_stage, checkpoint_path)

    record = initialize_candidate_encoder_from_pretraining(
        _FullModelStub(),
        checkpoint_path,
        target_stage,
        source_data,
    )

    assert record["source_candidate_model_config"]["use_manual_features"] is True
    assert record["source_candidate_model_config"]["use_sampling_features"] is True
    assert record["source_candidate_model_config"]["use_candidate_token_attention_bias"] is False
    assert "initial_graph_gate" not in record["source_candidate_model_config"]
    assert "initial_inventory_gate" not in record["source_candidate_model_config"]


@pytest.mark.parametrize(
    ("artifact", "field_name"),
    (
        ("yaml", "language_family"),
        ("yaml", "identity_variant"),
        ("resolved", "language_family"),
        ("resolved", "identity_variant"),
    ),
)
def test_candidate_initialization_requires_explicit_null_identity_fields(
    tmp_path: Path,
    artifact: str,
    field_name: str,
) -> None:
    """来源 YAML 与 resolved 必须显式记录候选预训练没有身份分支。"""

    source_plan_path, source_stage, source_data, target_stage = _batch3_plans()
    copied_plan_path = tmp_path / "source_plan.yaml"
    plan_values = yaml.safe_load(source_plan_path.read_text(encoding="utf-8"))
    if artifact == "yaml":
        plan_values["stages"][0].pop(field_name)
    copied_plan_path.write_text(
        yaml.safe_dump(plan_values, sort_keys=False),
        encoding="utf-8",
    )
    checkpoint_path = _write_completed_candidate_run(
        tmp_path / "source",
        copied_plan_path,
        source_stage,
        source_data,
        nn.Linear(3, 2),
        checkpoint_updates={},
        data_updates={},
        model_config_omissions=(),
        stage_field_omissions=(field_name,) if artifact == "resolved" else (),
        data_field_omissions=(),
        publish_run_summary=True,
    )
    target_stage = _target_stage_with_checkpoint(target_stage, checkpoint_path)

    with pytest.raises(ValueError, match="language_family.*identity_variant"):
        initialize_candidate_encoder_from_pretraining(
            _FullModelStub(),
            checkpoint_path,
            target_stage,
            source_data,
        )


@pytest.mark.parametrize(
    ("artifact", "field_state"),
    (
        ("yaml", "missing"),
        ("yaml", "non_null"),
        ("resolved", "missing"),
        ("resolved", "non_null"),
    ),
)
def test_candidate_initialization_requires_explicit_zero_source(
    tmp_path: Path,
    artifact: str,
    field_state: str,
) -> None:
    """来源 YAML 与 resolved 都必须显式记录从零候选预训练。"""

    source_plan_path, source_stage, source_data, target_stage = _batch3_plans()
    copied_plan_path = tmp_path / "source_plan.yaml"
    plan_values = yaml.safe_load(source_plan_path.read_text(encoding="utf-8"))
    if artifact == "yaml":
        if field_state == "missing":
            plan_values["stages"][0].pop("initialize_candidate_from")
        else:
            plan_values["stages"][0]["initialize_candidate_from"] = (
                "/storage/penghongen/tmp/Matcher/other/BEST.pt"
            )
    copied_plan_path.write_text(
        yaml.safe_dump(plan_values, sort_keys=False),
        encoding="utf-8",
    )
    checkpoint_path = _write_completed_candidate_run(
        tmp_path / "source",
        copied_plan_path,
        source_stage,
        source_data,
        nn.Linear(3, 2),
        checkpoint_updates={},
        data_updates={},
        model_config_omissions=(),
        stage_field_omissions=("initialize_candidate_from",)
        if artifact == "resolved" and field_state == "missing"
        else (),
        data_field_omissions=(),
        publish_run_summary=True,
    )
    if artifact == "resolved" and field_state == "non_null":
        resolved_path = checkpoint_path.parent / "resolved.json"
        resolved_values = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        resolved_values["stage"]["initialize_candidate_from"] = (
            "/storage/penghongen/tmp/Matcher/other/BEST.pt"
        )
        write_json(resolved_path, resolved_values)
    target_stage = _target_stage_with_checkpoint(target_stage, checkpoint_path)

    with pytest.raises(ValueError, match="从零训练"):
        initialize_candidate_encoder_from_pretraining(
            _FullModelStub(),
            checkpoint_path,
            target_stage,
            source_data,
        )


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("not_best", "epoch == checkpoint.best_epoch"),
        ("data_mismatch", "科学数据配置不一致"),
        ("model_mismatch", "候选编码器结构不一致"),
        ("incomplete_run", "summary.json"),
        ("plan_mismatch", "YAML plan_name"),
        ("source_initialized", "从零训练"),
        ("state_mismatch", "参数名与目标结构不一致"),
        ("state_non_string_key", "字符串键和张量值"),
        ("state_non_tensor", "字符串键和张量值"),
        ("state_shape_mismatch", "参数形状或 dtype"),
        ("state_dtype_mismatch", "参数形状或 dtype"),
        ("relative_last_checkpoint", "绝对路径"),
        ("wrong_last_checkpoint", "未证明"),
        ("relative_training_plan", "绝对路径"),
    ),
)
def test_candidate_initialization_rejects_before_changing_target_parameters(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    """来源证据或结构不一致时，在改动候选及非候选参数前失败。"""

    source_plan_path, source_stage, source_data, target_stage = _batch3_plans()
    copied_plan_path = tmp_path / "source_plan.yaml"
    copied_plan_path.write_text(
        source_plan_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    source_plan_path = copied_plan_path
    checkpoint_updates: dict[str, object] = {}
    data_updates: dict[str, object] = {}
    publish_run_summary = corruption != "incomplete_run"
    if corruption == "not_best":
        checkpoint_updates["best_epoch"] = 1
    if corruption == "data_mismatch":
        data_updates["selection_seed"] = source_data.selection_seed + 1
    if corruption == "model_mismatch":
        target_stage = replace(
            target_stage,
            model=replace(target_stage.model, dropout=target_stage.model.dropout + 0.1),
        )
    source_model = nn.Linear(3, 2)
    if corruption == "state_mismatch":
        source_state = source_model.state_dict()
        source_state["unexpected"] = torch.zeros(1)
        checkpoint_updates["model_state"] = source_state
    if corruption == "state_non_string_key":
        source_state = dict(source_model.state_dict())
        source_state[1] = torch.zeros(1)  # type: ignore[index]
        checkpoint_updates["model_state"] = source_state
    if corruption == "state_non_tensor":
        source_state = dict(source_model.state_dict())
        source_state["weight"] = 1
        checkpoint_updates["model_state"] = source_state
    if corruption == "state_shape_mismatch":
        source_state = source_model.state_dict()
        source_state["weight"] = torch.zeros(3, 3)
        checkpoint_updates["model_state"] = source_state
    if corruption == "state_dtype_mismatch":
        source_state = source_model.state_dict()
        source_state["weight"] = source_state["weight"].to(dtype=torch.float64)
        checkpoint_updates["model_state"] = source_state
    checkpoint_path = _write_completed_candidate_run(
        tmp_path / "source",
        source_plan_path,
        source_stage,
        source_data,
        source_model,
        checkpoint_updates=checkpoint_updates,
        data_updates=data_updates,
        model_config_omissions=(),
        stage_field_omissions=(),
        data_field_omissions=(),
        publish_run_summary=publish_run_summary,
    )
    if corruption == "plan_mismatch":
        plan_values = yaml.safe_load(source_plan_path.read_text(encoding="utf-8"))
        plan_values["plan_name"] = "changed_after_training"
        source_plan_path.write_text(
            yaml.safe_dump(plan_values, sort_keys=False),
            encoding="utf-8",
        )
    if corruption == "source_initialized":
        resolved_path = checkpoint_path.parent / "resolved.json"
        resolved_values = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        resolved_values["stage"]["initialize_candidate_from"] = (
            "/storage/penghongen/tmp/Matcher/other/BEST.pt"
        )
        write_json(resolved_path, resolved_values)
    if corruption in ("relative_last_checkpoint", "wrong_last_checkpoint"):
        stage_summary_path = checkpoint_path.parent / "summary.json"
        stage_summary = yaml.safe_load(stage_summary_path.read_text(encoding="utf-8"))
        stage_summary["last_checkpoint"] = (
            "LAST.pt"
            if corruption == "relative_last_checkpoint"
            else checkpoint_path.resolve().as_posix()
        )
        write_json(stage_summary_path, stage_summary)
    if corruption == "relative_training_plan":
        run_directory = checkpoint_path.parent.parent
        run_record_path = run_directory / "run.json"
        run_summary_path = run_directory / "summary.json"
        run_record = yaml.safe_load(run_record_path.read_text(encoding="utf-8"))
        run_summary = yaml.safe_load(run_summary_path.read_text(encoding="utf-8"))
        run_record["plan_path"] = source_plan_path.name
        run_summary["plan_path"] = source_plan_path.name
        write_json(run_record_path, run_record)
        write_json(run_summary_path, run_summary)
    target_stage = _target_stage_with_checkpoint(target_stage, checkpoint_path)
    target_model = _FullModelStub()
    before = {
        name: value.detach().clone()
        for name, value in target_model.state_dict().items()
    }

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        initialize_candidate_encoder_from_pretraining(
            target_model,
            checkpoint_path,
            target_stage,
            source_data,
        )

    for name, value in target_model.state_dict().items():
        assert torch.equal(value, before[name])
