"""在 validation 选定方案后，对一个 BEST checkpoint 执行一次 calibration 评估。

``run_calibration`` 是编程入口，``main`` 是 ``python -m matcher.evaluation.calibration`` 命令入口。
入口只读取 calibration 清单，不训练、不更新 checkpoint，也不计算新的选模分数。
评估输出是调用方指定的严格 JSON 文件；数据根还会保存按 calibration 数据指纹命名的
一次性 claim 和空互斥锁文件。claim 的 ``status`` 才是完成状态，锁文件没有科学语义。
结果包含计划、解析配置、checkpoint 与清单文件哈希、完整 ``EvaluationResult``、固定六类
与支持类 macro-F1，以及运行环境证据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, NoReturn, TypedDict, cast

import torch

from ..data.manifest import require_manifest
from ..data.schema import LocalCropStrategy
from ..datasets.batching import (
    DATALOADER_MULTIPROCESSING_CONTEXT,
    ShuffledBatchSampler,
    build_candidate_loader,
)
from ..datasets.candidate_data import CandidateDataset
from ..model import CandidateEncoder, ClosedSetMatcher, MatcherModelConfig
from ..training.checkpointing import load_checkpoint
from ..training.config import (
    ExperimentPlan,
    StageConfig,
    load_experiment_plan,
    validate_matcher_output_path,
)
from ..training.engine import EvaluationResult, evaluate_model
from ..training.numeric import NumericBackendRecord, configure_random_seeds_and_numeric_backend
from ..utils.json_artifacts import write_json
from ..utils.locking import exclusive_file_lock


CALIBRATION_SCHEMA_VERSION = 2


class CalibrationCheckpointEvidence(TypedDict):
    """证明 calibration 使用了与所选阶段一致的 BEST checkpoint。

    ``path`` 是服务器绝对路径，文件名必须为 BEST.pt。
    ``sha256`` 是评估前读取完整文件得到的小写十六进制摘要。
    ``epoch`` 是 checkpoint 保存的零起始最佳轮次。
    ``global_step`` 是 checkpoint 保存时累计完成的参数更新数。
    ``best_score`` 是 checkpoint 保存的 validation 最佳选择分数。
    ``resolved_path`` 是训练阶段解析配置路径。
    ``resolved_sha256`` 是训练阶段解析配置摘要。
    ``summary_path`` 是只在阶段完成后发布的阶段摘要路径。
    ``summary_sha256`` 是阶段摘要内容摘要。
    ``run_path`` 是训练开始前发布的运行记录路径。
    ``run_sha256`` 是运行记录内容摘要。
    ``training_plan_path`` 是训练实际读取的 YAML 路径。
    ``training_plan_sha256`` 是评估时重新读取训练 release YAML 得到的内容摘要。
    ``model_config`` 是已与 YAML 阶段逐字段相等的模型结构字典。
    ``training_config`` 是已与 YAML 阶段逐字段相等的训练配置字典。
    """

    path: str
    sha256: str
    epoch: int
    global_step: int
    best_score: float
    resolved_path: str
    resolved_sha256: str
    summary_path: str
    summary_sha256: str
    run_path: str
    run_sha256: str
    training_plan_path: str
    training_plan_sha256: str
    model_config: dict[str, Any]
    training_config: dict[str, Any]


class CalibrationDataEvidence(TypedDict):
    """冻结一次 calibration 实际使用的清单文件。

    ``manifest_path`` 是现存清单入口路径。
    ``manifest_sha256`` 是现存清单全文件内容摘要。
    ``training_manifest_sha256`` 是训练开始时留存的清单摘要；既有首轮运行未记录时为 None。
    ``manifest_semantic_sha256`` 摘要 calibration 相关的稳定清单字段，不含创建时间与其它划分。
    ``candidates_path`` 是 calibration 候选 JSONL 路径。
    ``candidates_sha256`` 是 calibration 候选 JSONL 内容摘要。
    ``identities_path`` 是 calibration 身份 JSONL 路径。
    ``identities_sha256`` 是 calibration 身份 JSONL 内容摘要。
    ``fingerprint`` 是三个摘要按 README 公式连接后的 SHA-256。
    """

    manifest_path: str
    manifest_sha256: str
    training_manifest_sha256: str | None
    manifest_semantic_sha256: str
    candidates_path: str
    candidates_sha256: str
    identities_path: str
    identities_sha256: str
    fingerprint: str


class CalibrationRuntimeEvidence(TypedDict):
    """记录 calibration 数据加载和数值运行环境。

    ``candidate_count`` 是冻结 calibration 清单中的候选数。
    ``physical_batch_size`` 沿用所选阶段的单次前向候选数。
    ``num_workers`` 沿用 YAML 的 validation 工作进程数。
    ``prefetch_factor`` 是每个工作进程预取的物理批次数。
    ``cache_gib_per_worker`` 是单个工作进程五类 LRU 缓存的 GiB 上限。
    ``local_crop_strategy`` 是 checkpoint 训练时冻结的候选 ``48³`` 裁剪策略。
    ``multiprocessing_context`` 固定为 spawn。
    ``persistent_workers`` 固定为 False，完整迭代结束后退出工作进程池。
    ``worker_generator_seed`` 是只供 calibration 工作进程播种的独立 Torch 种子。
    ``numeric_backend`` 是训练与评估共用的后端配置。
    ``device`` 是服务器实际 CUDA 设备名称。
    ``torch_version`` 是执行评估的 PyTorch 版本。
    """

    candidate_count: int
    physical_batch_size: int
    num_workers: int
    prefetch_factor: int
    cache_gib_per_worker: float
    local_crop_strategy: LocalCropStrategy
    multiprocessing_context: str
    persistent_workers: bool
    worker_generator_seed: int
    numeric_backend: NumericBackendRecord
    device: str
    torch_version: str


class CalibrationArtifact(TypedDict):
    """一次性 calibration JSON 的完整顶层字段。

    ``schema_version`` 是本产物格式版本。
    ``split`` 固定为 calibration。
    ``plan_path`` 是本次 calibration 读取的实验 YAML 绝对路径。
    ``plan_sha256`` 是本次 calibration YAML 的内容摘要。
    ``plan_name`` 是本次 YAML 声明的安全名称。
    ``stage_name`` 是 validation 选定的阶段名称。
    ``task_name`` 是 candidate_pretrain 或 full_matcher。
    ``language_family`` 是完整 Matcher 使用的语言表示；候选预训练为 None。
    ``identity_variant`` 是完整 Matcher 使用的身份损失；候选预训练为 None。
    ``checkpoint`` 保存模型文件及配置证据。
    ``data`` 保存 calibration 清单与两个记录文件的内容证据。
    ``claim_path`` 是按 data.fingerprint 固定的一次性选择记录。
    ``runtime`` 保存数据加载、数值后端和设备证据。
    ``evaluation`` 是完整 calibration 损失、指标、分母、耗时和吞吐。
    ``hostname`` 是实际执行节点名称。
    ``slurm_job_id`` 是 Slurm Job ID；非 Slurm 运行时为 None。
    ``task_launch_dir`` 是 launch 留证目录；调用方未提供时为 None。
    ``started_at_utc`` 是开始评估时带时区的 ISO 8601 时间。
    ``finished_at_utc`` 是完成评估时带时区的 ISO 8601 时间。
    """

    schema_version: int
    split: Literal["calibration"]
    plan_path: str
    plan_sha256: str
    plan_name: str
    stage_name: str
    task_name: str
    language_family: str | None
    identity_variant: str | None
    checkpoint: CalibrationCheckpointEvidence
    data: CalibrationDataEvidence
    claim_path: str
    runtime: CalibrationRuntimeEvidence
    evaluation: EvaluationResult
    hostname: str
    slurm_job_id: str | None
    task_launch_dir: str | None
    started_at_utc: str
    finished_at_utc: str


class CalibrationClaim(TypedDict):
    """在 claim 文件保留期间，把一个 calibration 数据指纹绑定到唯一所选模型。

    ``schema_version`` 是结果与 claim 共用且联动提升的格式版本。
    ``status`` 为 claimed 时允许同一选择失败后重试，为 completed 时禁止再次评估。
    ``data_fingerprint`` 是三个 calibration 清单文件的组合摘要。
    ``plan_sha256`` 是已选实验计划的内容摘要。
    ``stage_name`` 是已选阶段名称。
    ``checkpoint_sha256`` 是已选 BEST.pt 的内容摘要。
    ``checkpoint_path`` 是已选 BEST.pt 的服务器绝对路径。
    ``output_path`` 是首次 claim 时绑定的结果 JSON 路径。
    ``output_sha256`` 是成功发布结果后的内容摘要；claimed 状态为 None。
    ``claimed_at_utc`` 是首次绑定时带时区的 ISO 8601 时间。
    ``completed_at_utc`` 是成功完成时间；claimed 状态为 None。
    """

    schema_version: int
    status: Literal["claimed", "completed"]
    data_fingerprint: str
    plan_sha256: str
    stage_name: str
    checkpoint_sha256: str
    checkpoint_path: str
    output_path: str
    output_sha256: str | None
    claimed_at_utc: str
    completed_at_utc: str | None


def _select_stage(plan: ExperimentPlan, stage_name: str) -> StageConfig:
    """按完整名称选择唯一阶段；不存在或重复时拒绝继续。"""

    matches = [stage for stage in plan.stages if stage.name == stage_name]
    if len(matches) != 1:
        raise ValueError(f"计划中必须恰好存在一个 stage={stage_name!r}，实际为 {len(matches)}。")
    return matches[0]


def _sha256_file(path: Path) -> str:
    """流式读取文件并返回 SHA-256，避免大型文件产生第二份内存副本。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_json_constant(token: str) -> NoReturn:
    """拒绝 Python JSON 解码器默认接受的 NaN 与正负 Inf。"""

    raise ValueError(f"严格 JSON 不得包含非有限常量 {token}。")


def _read_json_mapping(path: Path, name: str) -> dict[str, Any]:
    """读取严格 JSON 对象；拒绝 NaN、Inf、非对象顶层和非字符串键。"""

    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} 必须是字符串键 JSON 对象。")
    return cast(dict[str, Any], value)


def _calibration_data_evidence(manifest_path: Path) -> CalibrationDataEvidence:
    """计算主清单及 calibration 候选、身份记录的稳定内容指纹。"""

    manifest = _read_json_mapping(manifest_path, "manifest.json")
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("manifest.json.splits 必须是 JSON 对象。")
    calibration = splits.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("manifest.json.splits.calibration 必须是 JSON 对象。")
    candidate_name = calibration.get("candidates")
    identity_name = calibration.get("identities")
    if not isinstance(candidate_name, str) or Path(candidate_name).name != candidate_name:
        raise ValueError("calibration candidates 必须是清单目录内的单级文件名。")
    if not isinstance(identity_name, str) or Path(identity_name).name != identity_name:
        raise ValueError("calibration identities 必须是清单目录内的单级文件名。")
    candidates_path = manifest_path.parent / candidate_name
    identities_path = manifest_path.parent / identity_name
    manifest_sha256 = _sha256_file(manifest_path)
    split_pdb_counts = manifest.get("split_pdb_counts")
    summary = manifest.get("summary")
    summary_splits = summary.get("splits") if isinstance(summary, Mapping) else None
    semantic_manifest = {
        "schema_version": manifest.get("schema_version"),
        "data_root": manifest.get("data_root"),
        "stage1_root": manifest.get("stage1_root"),
        "output_root": manifest.get("output_root"),
        "coverage_threshold": manifest.get("coverage_threshold"),
        "selection_seed": manifest.get("selection_seed"),
        "language_selection_rule": manifest.get("language_selection_rule"),
        "seen_identity_rule": manifest.get("seen_identity_rule"),
        "class_names": manifest.get("class_names"),
        "calibration_files": calibration,
        "calibration_pdb_counts": (
            split_pdb_counts.get("calibration")
            if isinstance(split_pdb_counts, Mapping)
            else None
        ),
        "calibration_summary": (
            summary_splits.get("calibration") if isinstance(summary_splits, Mapping) else None
        ),
    }
    semantic_bytes = json.dumps(
        semantic_manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    manifest_semantic_sha256 = hashlib.sha256(semantic_bytes).hexdigest()
    candidates_sha256 = _sha256_file(candidates_path)
    identities_sha256 = _sha256_file(identities_path)
    fingerprint = hashlib.sha256(
        (
            f"{manifest_semantic_sha256}\n"
            f"{candidates_sha256}\n"
            f"{identities_sha256}\n"
        ).encode("ascii")
    ).hexdigest()
    return {
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": manifest_sha256,
        "training_manifest_sha256": None,
        "manifest_semantic_sha256": manifest_semantic_sha256,
        "candidates_path": candidates_path.as_posix(),
        "candidates_sha256": candidates_sha256,
        "identities_path": identities_path.as_posix(),
        "identities_sha256": identities_sha256,
        "fingerprint": fingerprint,
    }


def _validate_checkpoint_for_stage(
    checkpoint_path: Path,
    checkpoint: Mapping[str, object],
    plan: ExperimentPlan,
    stage: StageConfig,
) -> CalibrationCheckpointEvidence:
    """核对已完成阶段、BEST 自洽性和科学数据配置，并返回文件证据。"""

    if checkpoint.get("stage_name") != stage.name:
        raise ValueError("checkpoint.stage_name 与所选 YAML 阶段不一致。")
    if checkpoint.get("task_name") != stage.task_name:
        raise ValueError("checkpoint.task_name 与所选 YAML 阶段不一致。")

    integers: dict[str, int] = {}
    for name in ("epoch", "best_epoch", "global_step"):
        value = checkpoint.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"checkpoint.{name} 必须是非负整数。")
        integers[name] = value
    if integers["epoch"] != integers["best_epoch"]:
        raise ValueError("BEST.pt 必须满足 checkpoint.epoch == checkpoint.best_epoch。")
    score_value = checkpoint.get("best_score")
    if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
        raise ValueError("checkpoint.best_score 必须是有限数值。")
    best_score = float(score_value)
    if not math.isfinite(best_score):
        raise ValueError("checkpoint.best_score 必须是有限数值。")
    checkpoint_mappings: dict[str, dict[str, Any]] = {}
    for name in ("model_config", "training_config", "epoch_record"):
        value = checkpoint.get(name)
        if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
            raise ValueError(f"checkpoint.{name} 必须是字符串键映射。")
        checkpoint_mappings[name] = {str(key): item for key, item in value.items()}
    epoch_record = checkpoint_mappings["epoch_record"]
    if (
        epoch_record.get("epoch") != integers["epoch"]
        or epoch_record.get("global_step") != integers["global_step"]
        or epoch_record.get("selection_score") != best_score
        or epoch_record.get("improved") is not True
    ):
        raise ValueError("BEST.pt 的 epoch_record 必须对应同一最佳轮次、更新数和选择分数。")

    resolved_path = checkpoint_path.parent / "resolved.json"
    resolved = _read_json_mapping(resolved_path, "resolved.json")
    expected_stage = asdict(stage)
    expected_stage["model"] = stage.model.to_dict()
    expected_stage_json = json.loads(
        json.dumps(expected_stage, ensure_ascii=False, allow_nan=False)
    )
    resolved_stage_value = resolved.get("stage")
    if not isinstance(resolved_stage_value, Mapping):
        raise ValueError("resolved.json.stage 必须是映射。")
    resolved_stage = {str(key): value for key, value in resolved_stage_value.items()}
    resolved_model_value = resolved_stage.get("model")
    if not isinstance(resolved_model_value, Mapping) or not all(
        isinstance(key, str) for key in resolved_model_value
    ):
        raise ValueError("resolved.json.stage.model 必须是字符串键映射。")
    normalized_resolved_model = MatcherModelConfig.from_dict(
        {str(key): value for key, value in resolved_model_value.items()}
    ).to_dict()
    resolved_stage["model"] = json.loads(
        json.dumps(normalized_resolved_model, ensure_ascii=False, allow_nan=False)
    )
    if "identity_scope" not in resolved_stage:
        task_name = resolved_stage.get("task_name")
        if task_name == "full_matcher":
            resolved_stage["identity_scope"] = "small_molecule"
        elif task_name == "candidate_pretrain":
            resolved_stage["identity_scope"] = None
    if resolved_stage != expected_stage_json:
        raise ValueError("resolved.json.stage 与所选 YAML 阶段不一致。")
    resolved_data_value = resolved.get("data")
    if not isinstance(resolved_data_value, Mapping):
        raise ValueError("resolved.json.data 必须是映射。")
    resolved_data = {str(key): value for key, value in resolved_data_value.items()}
    if "local_crop_strategy" not in resolved_data:
        resolved_data["local_crop_strategy"] = "fixed_center"
    expected_scientific_data = {
        "data_root": str(plan.data.roots.data_root),
        "stage1_root": str(plan.data.roots.stage1_root),
        "output_root": str(plan.data.roots.output_root),
        "coverage_threshold": plan.data.coverage_threshold,
        "selection_seed": plan.data.selection_seed,
        "pp_top_k": plan.data.pp_top_k,
        "local_crop_strategy": plan.data.local_crop_strategy,
    }
    data_mismatches = {
        name: {"actual": resolved_data.get(name), "expected": expected}
        for name, expected in expected_scientific_data.items()
        if resolved_data.get(name) != expected
    }
    if data_mismatches:
        raise ValueError(f"resolved.json.data 科学字段与 calibration 计划不一致：{data_mismatches}")
    model_config = MatcherModelConfig.from_dict(
        checkpoint_mappings["model_config"]
    ).to_dict()
    training_config = checkpoint_mappings["training_config"]
    if model_config != stage.model.to_dict():
        raise ValueError("checkpoint.model_config 与所选 YAML 阶段不一致。")
    if training_config != asdict(stage.training):
        raise ValueError("checkpoint.training_config 与所选 YAML 阶段不一致。")

    summary_path = checkpoint_path.parent / "summary.json"
    summary = _read_json_mapping(summary_path, "stage summary.json")
    summary_best_path = summary.get("best_checkpoint")
    if not isinstance(summary_best_path, str):
        raise ValueError("stage summary.best_checkpoint 必须是字符串路径。")
    summary_score = summary.get("best_score")
    if (
        summary.get("stage") != stage.name
        or summary.get("best_epoch") != integers["epoch"]
        or summary.get("last_epoch") is None
        or type(summary.get("last_epoch")) is not int
        or cast(int, summary["last_epoch"]) < integers["epoch"]
        or summary_score != best_score
        or Path(summary_best_path).as_posix() != checkpoint_path.as_posix()
    ):
        raise ValueError("阶段 summary.json 未证明该路径是已完成阶段的最佳 checkpoint。")

    run_path = checkpoint_path.parent.parent / "run.json"
    run_record = _read_json_mapping(run_path, "run.json")
    if run_record.get("plan_name") != plan.plan_name:
        raise ValueError("run.json.plan_name 与 calibration 计划不一致。")
    training_plan_value = run_record.get("plan_path")
    if not isinstance(training_plan_value, str):
        raise ValueError("run.json.plan_path 必须是字符串路径。")
    training_plan_path = Path(training_plan_value)
    if not training_plan_path.is_file():
        raise FileNotFoundError(f"训练时 YAML 已不存在，无法完成留证：{training_plan_path}")
    return {
        "path": checkpoint_path.as_posix(),
        "sha256": _sha256_file(checkpoint_path),
        "epoch": integers["epoch"],
        "global_step": integers["global_step"],
        "best_score": best_score,
        "resolved_path": resolved_path.as_posix(),
        "resolved_sha256": _sha256_file(resolved_path),
        "summary_path": summary_path.as_posix(),
        "summary_sha256": _sha256_file(summary_path),
        "run_path": run_path.as_posix(),
        "run_sha256": _sha256_file(run_path),
        "training_plan_path": training_plan_path.as_posix(),
        "training_plan_sha256": _sha256_file(training_plan_path),
        "model_config": model_config,
        "training_config": training_config,
    }


def _claim_calibration(
    claim_path: Path,
    data_fingerprint: str,
    plan_sha256: str,
    stage_name: str,
    checkpoint: CalibrationCheckpointEvidence,
    output_path: Path,
    claimed_at_utc: str,
) -> CalibrationClaim:
    """创建一次性 claim，或只允许同一模型在未完成时原路径重试。"""

    immutable = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "data_fingerprint": data_fingerprint,
        "plan_sha256": plan_sha256,
        "stage_name": stage_name,
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_path": checkpoint["path"],
        "output_path": output_path.as_posix(),
    }
    if claim_path.exists():
        existing = _read_json_mapping(claim_path, "calibration claim")
        mismatches = {
            name: {"actual": existing.get(name), "expected": expected}
            for name, expected in immutable.items()
            if existing.get(name) != expected
        }
        if mismatches:
            raise RuntimeError(f"该 calibration 数据已绑定另一项选择：{mismatches}")
        if existing.get("status") == "completed":
            raise RuntimeError("该 calibration 数据已经完成一次评估，禁止再次运行。")
        if existing.get("status") != "claimed" or not isinstance(
            existing.get("claimed_at_utc"), str
        ):
            raise ValueError("既有 calibration claim 的状态或时间字段无效。")
        if existing.get("completed_at_utc") is not None:
            raise ValueError("claimed 状态的 completed_at_utc 必须为 null。")
        if existing.get("output_sha256") is not None:
            raise ValueError("claimed 状态的 output_sha256 必须为 null。")
        return cast(CalibrationClaim, existing)
    claim: CalibrationClaim = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": "claimed",
        "data_fingerprint": data_fingerprint,
        "plan_sha256": plan_sha256,
        "stage_name": stage_name,
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_path": checkpoint["path"],
        "output_path": output_path.as_posix(),
        "output_sha256": None,
        "claimed_at_utc": claimed_at_utc,
        "completed_at_utc": None,
    }
    write_json(claim_path, claim)
    return claim


def _recover_claim_from_published_output(
    output_path: Path,
    claim_path: Path,
    claim: CalibrationClaim,
) -> bool:
    """验证已原子发布的结果并把崩溃窗口中的 claimed 状态收口为 completed。"""

    if not output_path.exists():
        return False
    artifact = _read_json_mapping(output_path, "calibration output")
    checkpoint = artifact.get("checkpoint")
    data = artifact.get("data")
    if not isinstance(checkpoint, Mapping) or not isinstance(data, Mapping):
        raise ValueError("既有 calibration output 缺少 checkpoint 或 data 对象。")
    expected = {
        "schema_version": claim["schema_version"],
        "split": "calibration",
        "plan_sha256": claim["plan_sha256"],
        "stage_name": claim["stage_name"],
        "claim_path": claim_path.as_posix(),
    }
    mismatches = {
        name: {"actual": artifact.get(name), "expected": value}
        for name, value in expected.items()
        if artifact.get(name) != value
    }
    if checkpoint.get("sha256") != claim["checkpoint_sha256"]:
        mismatches["checkpoint.sha256"] = {
            "actual": checkpoint.get("sha256"),
            "expected": claim["checkpoint_sha256"],
        }
    if data.get("fingerprint") != claim["data_fingerprint"]:
        mismatches["data.fingerprint"] = {
            "actual": data.get("fingerprint"),
            "expected": claim["data_fingerprint"],
        }
    if mismatches:
        raise RuntimeError(f"既有 calibration output 与 claimed 选择不一致：{mismatches}")
    finished_at_utc = artifact.get("finished_at_utc")
    if not isinstance(finished_at_utc, str):
        raise ValueError("calibration output.finished_at_utc 必须是字符串。")
    completed_claim: CalibrationClaim = {
        **claim,
        "status": "completed",
        "output_sha256": _sha256_file(output_path),
        "completed_at_utc": finished_at_utc,
    }
    write_json(claim_path, completed_claim)
    return True


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    """解析计划、阶段、BEST checkpoint 和唯一输出 JSON。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="包含所选阶段的实验 YAML。")
    parser.add_argument("--stage", required=True, help="YAML 中已完成训练阶段的完整名称。")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="同一已完成阶段在 Matcher 产物根下的 BEST.pt。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Matcher 产物根下尚不存在的 calibration 结果 .json。",
    )
    return parser.parse_args(argv)


# ================================================================================================


def run_calibration(
    plan_path: Path,
    stage_name: str,
    checkpoint_path: Path,
    output_path: Path,
) -> Path:
    """用所选 BEST checkpoint 评估一次 calibration，并原子写出唯一 JSON。

    四个路径参数均由调用方显式提供。checkpoint 与输出必须位于
    ``/storage/penghongen/tmp/Matcher/``；claim 保留期间，相同数据指纹只能绑定一个选择。
    本函数不初始化优化器、调度器或 W&B，不修改 checkpoint，也不返回选模分数。
    """

    plan = load_experiment_plan(plan_path)
    stage = _select_stage(plan, stage_name)
    checkpoint_path = validate_matcher_output_path(checkpoint_path, "checkpoint")
    output_path = validate_matcher_output_path(output_path, "output")
    if checkpoint_path.name != "BEST.pt":
        raise ValueError("calibration 只接受文件名为 BEST.pt 的所选最佳 checkpoint。")
    if output_path.suffix != ".json":
        raise ValueError("calibration 输出必须是 .json 文件。")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    resolved_plan_path = plan_path.resolve(strict=True)
    plan_sha256 = _sha256_file(resolved_plan_path)
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_evidence = _validate_checkpoint_for_stage(
        checkpoint_path,
        checkpoint,
        plan,
        stage,
    )
    manifest_path = require_manifest(
        plan.data.roots,
        plan.data.coverage_threshold,
        plan.data.selection_seed,
        ("calibration",),
    )
    data_evidence = _calibration_data_evidence(manifest_path)
    claim_path = validate_matcher_output_path(
        plan.data.roots.output_root
        / "calibration"
        / f"claim_{data_evidence['fingerprint']}.json",
        "calibration claim",
    )
    lock_path = claim_path.with_suffix(f"{claim_path.suffix}.lock")
    if output_path in (claim_path, lock_path):
        raise ValueError("calibration 输出不得与一次性 claim 或其互斥锁使用同一路径。")
    with exclusive_file_lock(lock_path):
        claim_existed_before = claim_path.exists()
        if output_path.exists() and not claim_existed_before:
            raise FileExistsError(
                f"calibration 输出已存在但没有对应 claim，禁止把它当作崩溃恢复：{output_path}"
            )
        started_at_utc = datetime.now(timezone.utc).isoformat()
        claim = _claim_calibration(
            claim_path,
            data_evidence["fingerprint"],
            plan_sha256,
            stage.name,
            checkpoint_evidence,
            output_path,
            started_at_utc,
        )
        if _recover_claim_from_published_output(output_path, claim_path, claim):
            print(
                json.dumps(
                    {
                        "event": "calibration_claim_recovered",
                        "stage": stage.name,
                        "checkpoint_sha256": checkpoint_evidence["sha256"],
                        "output": output_path.as_posix(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            return output_path
        numeric_backend = configure_random_seeds_and_numeric_backend(stage.training.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("正式 calibration 必须在 CUDA GPU 上运行。")

        model: torch.nn.Module
        if stage.task_name == "candidate_pretrain":
            model = CandidateEncoder(stage.model)
        else:
            model = ClosedSetMatcher(stage.model)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device)

        calibration_dataset = CandidateDataset(
            roots=plan.data.roots,
            manifest_path=manifest_path,
            split_name="calibration",
            task_name=stage.task_name,
            language_family=stage.language_family,
            identity_scope=stage.identity_scope,
            pp_top_k=plan.data.pp_top_k,
            local_crop_strategy=plan.data.local_crop_strategy,
            cache_bytes_per_worker=int(plan.data.cache_gib_per_worker * 1024**3),
        )
        calibration_sampler = ShuffledBatchSampler(
            calibration_dataset,
            batch_size=stage.training.physical_batch_size,
            shuffle=False,
            drop_last=False,
            seed=stage.training.seed,
            sample_count_multiple=None,
        )
        worker_generator_seed = 2 * stage.training.seed + 2
        calibration_loader = build_candidate_loader(
            calibration_dataset,
            calibration_sampler,
            plan.data.validation_num_workers,
            plan.data.prefetch_factor,
            worker_generator_seed,
        )
        class_weights = (
            None
            if stage.training.class_weights is None
            else torch.tensor(stage.training.class_weights, device=device, dtype=torch.float32)
        )
        evaluation = evaluate_model(model, calibration_loader, stage, class_weights, device)
        if evaluation["sample_count"] != len(calibration_dataset):
            raise RuntimeError(
                "calibration EvaluationResult.sample_count 与冻结 Dataset 候选数不一致。"
            )
        finished_at_utc = datetime.now(timezone.utc).isoformat()
        artifact: CalibrationArtifact = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "split": "calibration",
            "plan_path": resolved_plan_path.as_posix(),
            "plan_sha256": plan_sha256,
            "plan_name": plan.plan_name,
            "stage_name": stage.name,
            "task_name": stage.task_name,
            "language_family": stage.language_family,
            "identity_variant": stage.identity_variant,
            "checkpoint": checkpoint_evidence,
            "data": data_evidence,
            "claim_path": claim_path.as_posix(),
            "runtime": {
                "candidate_count": len(calibration_dataset),
                "physical_batch_size": stage.training.physical_batch_size,
                "num_workers": plan.data.validation_num_workers,
                "prefetch_factor": plan.data.prefetch_factor,
                "cache_gib_per_worker": plan.data.cache_gib_per_worker,
                "local_crop_strategy": plan.data.local_crop_strategy,
                "multiprocessing_context": DATALOADER_MULTIPROCESSING_CONTEXT,
                "persistent_workers": False,
                "worker_generator_seed": worker_generator_seed,
                "numeric_backend": numeric_backend,
                "device": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__,
            },
            "evaluation": evaluation,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "task_launch_dir": os.environ.get("TASK_LAUNCH_DIR"),
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
        }
        if output_path.exists():
            raise FileExistsError(f"calibration 输出在评估期间出现，禁止覆盖：{output_path}")
        write_json(output_path, artifact)
        completed_claim: CalibrationClaim = {
            **claim,
            "status": "completed",
            "output_sha256": _sha256_file(output_path),
            "completed_at_utc": finished_at_utc,
        }
        write_json(claim_path, completed_claim)
        print(
            json.dumps(
                {
                    "event": "calibration_completed",
                    "stage": stage.name,
                    "checkpoint_sha256": checkpoint_evidence["sha256"],
                    "sample_count": evaluation["sample_count"],
                    "output": output_path.as_posix(),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    return output_path


def main(argv: list[str] | None = None) -> int:
    """执行一次 calibration，并把严格 JSON 绝对路径写到标准输出。"""

    arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
    output_path = run_calibration(
        arguments.plan,
        arguments.stage,
        arguments.checkpoint,
        arguments.output,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
