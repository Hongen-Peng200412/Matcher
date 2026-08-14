"""验证 W&B online 初始化、完整元数据、曲线间隔、空值、摘要和失败收口。"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import torch

import matcher.training.tracking as tracking_module
from matcher.model.fusion import CandidateConditionedAggregator
from matcher.training.config import load_experiment_plan
from matcher.training.engine import _attention_bias_weight_metrics
from matcher.training.tracking import MetricValue, start_wandb_tracking


class _FakeWandbSummary:
    """保存适配层写入的 W&B 最终摘要。"""

    def __init__(self) -> None:
        self.values: dict[str, MetricValue] = {}

    def update(self, values: Mapping[str, MetricValue]) -> None:
        """合并一次摘要更新。"""

        self.values.update(values)


class _FakeWandbRun:
    """记录适配层发出的横轴定义、指标、摘要与结束码。"""

    def __init__(
        self,
        run_id: object = "fake-run-id",
        run_url: object = (
            "https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/fake-run-id"
        ),
    ) -> None:
        self.id = run_id
        self._run_url = run_url
        self.summary = _FakeWandbSummary()
        self.metric_definitions: list[tuple[str, str | None]] = []
        self.logs: list[dict[str, MetricValue]] = []
        self.exit_codes: list[int] = []

    def get_url(self) -> object:
        """返回测试指定的在线地址或非法替身值。"""

        return self._run_url

    def define_metric(self, name: str, *, step_metric: str | None = None) -> None:
        """保存一项自定义横轴声明。"""

        self.metric_definitions.append((name, step_metric))

    def log(self, values: Mapping[str, MetricValue]) -> None:
        """保存一次扁平曲线写入。"""

        self.logs.append(dict(values))

    def finish(self, *, exit_code: int) -> None:
        """保存阶段结束码。"""

        self.exit_codes.append(exit_code)


class _FakeWandbModule(ModuleType):
    """提供与正式适配器最小协议一致的可注入 W&B 模块。"""

    __version__ = "0.17.3"

    def __init__(self, run: _FakeWandbRun) -> None:
        super().__init__("wandb")
        self.run = run
        self.initialization: dict[str, object] = {}
        self.settings: dict[str, int] = {}

    def Settings(self, *, init_timeout: int) -> object:
        """保存在线初始化等待时间并返回设置替身。"""

        self.settings = {"init_timeout": init_timeout}
        return self.settings

    def init(
        self,
        *,
        project: str,
        entity: str,
        group: str,
        name: str,
        job_type: str,
        tags: list[str],
        config: dict[str, object],
        dir: str,
        mode: str,
        resume: str,
        reinit: bool,
        anonymous: str,
        save_code: bool,
        force: bool,
        settings: object,
    ) -> _FakeWandbRun:
        """保存全部正式初始化参数并返回预先构造的 run。"""

        self.initialization = {
            "project": project,
            "entity": entity,
            "group": group,
            "name": name,
            "job_type": job_type,
            "tags": tags,
            "config": config,
            "dir": dir,
            "mode": mode,
            "resume": resume,
            "reinit": reinit,
            "anonymous": anonymous,
            "save_code": save_code,
            "force": force,
            "settings": settings,
        }
        return self.run


def test_wandb_tracker_records_complete_metadata_intervals_nulls_summary_and_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """在线 run 必须完整落盘契约，并按 global step 记录曲线与零支持空值。"""

    plan_path = (
        Path(__file__).parents[2]
        / "configs/experiments/initial/full_scratch_molformer_oracle.yaml"
    )
    plan = load_experiment_plan(plan_path)
    tracking = replace(plan.tracking, log_every_global_steps=2)
    stage = plan.stages[0]
    stage_directory = tmp_path / stage.name
    stage_directory.mkdir()
    run = _FakeWandbRun()
    fake_wandb = _FakeWandbModule(run)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    tracker = start_wandb_tracking(
        tracking,
        stage,
        stage_directory,
        tmp_path,
        {"stage": {"name": stage.name}},
    )

    metadata = json.loads((stage_directory / "tracking.json").read_text(encoding="utf-8"))
    assert metadata == {
        "provider": "wandb",
        "mode": "online",
        "entity": "pencounkdual-111",
        "project": "AdaLigand-Matcher",
        "group": "matcher-initial-round1",
        "run_id": "fake-run-id",
        "run_name": f"{stage.name}__{tmp_path.name}",
        "run_url": "https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/fake-run-id",
        "wandb_version": "0.17.3",
        "local_directory": str(stage_directory / "wandb"),
    }
    assert "api_key" not in json.dumps(metadata).lower()
    assert fake_wandb.initialization["mode"] == "online"
    assert fake_wandb.initialization["resume"] == "never"
    assert fake_wandb.initialization["force"] is True
    assert fake_wandb.initialization["project"] == "AdaLigand-Matcher"
    assert fake_wandb.initialization["entity"] == "pencounkdual-111"
    assert fake_wandb.settings == {"init_timeout": 120}
    assert ("global_step", None) in run.metric_definitions
    assert ("loss/train_step/*", "global_step") in run.metric_definitions
    assert ("gradient/train_step/*", "global_step") in run.metric_definitions
    assert ("attention_bias/train_step/*", "global_step") in run.metric_definitions
    assert ("epoch", None) in run.metric_definitions
    assert ("classification_summary/*", "epoch") in run.metric_definitions
    assert ("identity_all_non_background/*", "epoch") in run.metric_definitions
    assert ("selection/*", "epoch") in run.metric_definitions

    tracker.log_global_step(1, {"loss/train_step/total": 1.0})
    tracker.log_global_step(2, {"loss/train_step/total": 0.75})
    tracker.log_global_step(3, {"loss/train_step/total": 0.5})
    tracker.log_epoch(
        0,
        2,
        {
            "identity_configured/conditional/seen/top1": float("nan"),
            "identity_configured/conditional/seen/count": 0,
        },
    )
    tracker.update_summary({"best_epoch": 0, "best_score": 0.4})
    tracker.finish(exit_code=0)

    assert [values["global_step"] for values in run.logs[:2]] == [1, 2]
    assert run.logs[2]["epoch"] == 0
    assert run.logs[2]["identity_configured/conditional/seen/top1"] is None
    assert run.summary.values == {"best_epoch": 0, "best_score": 0.4}
    assert run.exit_codes == [0]


def test_wandb_tracker_rejects_unexplained_nonfinite_training_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """逐步损失的 NaN 或 Inf 不能伪装成合法的零支持空值。"""

    plan = load_experiment_plan(
        Path(__file__).parents[2]
        / "configs/experiments/initial/full_scratch_molformer_oracle.yaml"
    )
    stage = plan.stages[0]
    stage_directory = tmp_path / stage.name
    stage_directory.mkdir()
    run = _FakeWandbRun()
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandbModule(run))
    tracker = start_wandb_tracking(
        plan.tracking,
        stage,
        stage_directory,
        tmp_path,
        {"stage": {"name": stage.name}},
    )

    with pytest.raises(FloatingPointError, match="未被零支持分母解释"):
        tracker.log_global_step(1, {"loss/train_step/total": float("nan")})
    with pytest.raises(FloatingPointError, match="含 Inf"):
        tracker.log_global_step(1, {"loss/train_step/total": float("inf")})


def test_attention_bias_telemetry_records_postclip_weight_gradient_and_poststep_weight(
) -> None:
    """诊断记录裁剪后梯度与更新后权重，并省略整批无实体的 PP 模态。"""

    model = torch.nn.Module()
    aggregator = CandidateConditionedAggregator(16, 4, 0.0, False, False, True)
    model.add_module("aggregator", aggregator)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    query = torch.randn(2, 16)
    modalities = {
        name: (torch.randn(2, 3, 16), torch.ones((2, 3), dtype=torch.bool))
        for name in ("a", "p", "v")
    }
    modalities["pp"] = (
        torch.empty((2, 0, 16)),
        torch.empty((2, 0), dtype=torch.bool),
    )
    bias_features = {
        name: torch.randn(tokens.shape[0], tokens.shape[1], 4)
        for name, (tokens, _) in modalities.items()
    }
    output = aggregator(query, modalities, None, None, bias_features)
    output.square().sum().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    assert aggregator.attention_biases is not None
    a_final = aggregator.attention_biases["a"].layers[2]
    assert isinstance(a_final, torch.nn.Linear)
    a_weight = a_final.weight
    assert a_weight.grad is not None
    expected_postclip = float(a_weight.grad.float().square().sum().sqrt())
    optimizer.step()
    expected_poststep = float(a_weight.detach().float().square().sum().sqrt())

    metrics = _attention_bias_weight_metrics(model, "")
    expected_keys = {
        f"{root}/train_step/{name}_{suffix}"
        for name in ("a", "p", "v")
        for root, suffix in (
            ("gradient", "attention_bias_final_weight_postclip_norm"),
            ("attention_bias", "final_weight_poststep_norm"),
        )
    }
    assert set(metrics) == expected_keys
    assert metrics[
        "gradient/train_step/a_attention_bias_final_weight_postclip_norm"
    ] == pytest.approx(expected_postclip)
    assert metrics[
        "attention_bias/train_step/a_final_weight_poststep_norm"
    ] == pytest.approx(expected_poststep)


def test_wandb_start_rejects_empty_run_id_and_finishes_created_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """初始化后发现空 run ID 时必须失败收口，不能把空值字符串化。"""

    plan = load_experiment_plan(
        Path(__file__).parents[2]
        / "configs/experiments/initial/full_scratch_molformer_oracle.yaml"
    )
    stage = plan.stages[0]
    stage_directory = tmp_path / stage.name
    stage_directory.mkdir()
    run = _FakeWandbRun(run_id=None)
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandbModule(run))

    with pytest.raises(RuntimeError, match="run ID"):
        start_wandb_tracking(
            plan.tracking,
            stage,
            stage_directory,
            tmp_path,
            {"stage": {"name": stage.name}},
        )
    assert run.exit_codes == [1]
    assert not (stage_directory / "tracking.json").exists()


def test_wandb_start_preserves_artifact_error_while_finishing_created_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tracking.json 写入失败时必须保留原异常并以失败码结束已创建 run。"""

    plan = load_experiment_plan(
        Path(__file__).parents[2]
        / "configs/experiments/initial/full_scratch_molformer_oracle.yaml"
    )
    stage = plan.stages[0]
    stage_directory = tmp_path / stage.name
    stage_directory.mkdir()
    run = _FakeWandbRun()
    monkeypatch.setitem(sys.modules, "wandb", _FakeWandbModule(run))

    def fail_write_json(path: Path, value: object) -> None:
        """模拟元数据落盘失败。"""

        del path, value
        raise OSError("tracking artifact failed")

    monkeypatch.setattr(tracking_module, "write_json", fail_write_json)
    with pytest.raises(OSError, match="tracking artifact failed"):
        start_wandb_tracking(
            plan.tracking,
            stage,
            stage_directory,
            tmp_path,
            {"stage": {"name": stage.name}},
        )
    assert run.exit_codes == [1]
