"""一次性记录第 4 批候选预训练来源的发布关系与关键文件字节摘要。

运行方法：
``python audit_candidate_pretrain_source.py --checkpoint /abs/run/stage/BEST.pt --report /storage/penghongen/tmp/Matcher/diagnostics/batch4_source_audit/<name>.json``

脚本只读取来源 checkpoint、训练记录、launch 和 release，唯一写入是调用方指定的 diagnostics JSON。
release manifest 的内容摘要按声明值记录；脚本不独立重算或证明整棵 release 目录的摘要。
它不导入 ``matcher`` 正式包，不读取 calibration，不加载模型，也不参与训练入口或 ``resolved.json``。
输出中的 SHA-256 只服务选定第 4 批来源 BEST 时的一次性运维留证。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, NoReturn, cast


DIAGNOSTICS_ROOT = Path("/storage/penghongen/tmp/Matcher/diagnostics")


def reject_nonfinite_constant(token: str) -> NoReturn:
    raise ValueError(f"严格 JSON 不得包含非有限常量 {token}。")


def read_json_and_sha256(path: Path, name: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} 必须使用 UTF-8。") from error
    value = json.loads(text, parse_constant=reject_nonfinite_constant)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} 必须是字符串键 JSON 对象。")
    return cast(dict[str, Any], value), hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def require_string(mapping: dict[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name}.{key} 必须是非空字符串。")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="已选 candidate_pretrain 阶段 BEST.pt 的绝对路径。")
    parser.add_argument("--report", required=True, help="Matcher diagnostics 根内尚不存在的 JSON 报告绝对路径。")
    arguments = parser.parse_args()

    checkpoint_argument = Path(arguments.checkpoint)
    report_argument = Path(arguments.report)
    if not checkpoint_argument.is_absolute():
        raise ValueError("--checkpoint 必须是绝对 BEST.pt。")
    if not report_argument.is_absolute():
        raise ValueError(f"--report 必须位于 {DIAGNOSTICS_ROOT}。")
    checkpoint_path = checkpoint_argument.resolve(strict=True)
    report_path = report_argument.resolve(strict=False)
    diagnostics_root = DIAGNOSTICS_ROOT.resolve(strict=True)
    if checkpoint_path.name != "BEST.pt":
        raise ValueError("--checkpoint 必须是绝对 BEST.pt。")
    if not report_path.is_relative_to(diagnostics_root):
        raise ValueError(f"--report 必须位于 {diagnostics_root}。")
    if report_path.suffix != ".json" or report_path.exists():
        raise ValueError("--report 必须是尚不存在的 .json 文件。")

    stage_directory = checkpoint_path.parent
    run_directory = stage_directory.parent
    resolved_path = stage_directory / "resolved.json"
    stage_summary_path = stage_directory / "summary.json"
    run_record_path = run_directory / "run.json"
    run_summary_path = run_directory / "summary.json"
    resolved, resolved_sha256 = read_json_and_sha256(resolved_path, "resolved.json")
    stage_summary, stage_summary_sha256 = read_json_and_sha256(
        stage_summary_path,
        "阶段 summary.json",
    )
    run_record, run_record_sha256 = read_json_and_sha256(run_record_path, "run.json")
    run_summary, run_summary_sha256 = read_json_and_sha256(
        run_summary_path,
        "运行 summary.json",
    )

    plan_argument = Path(require_string(run_summary, "plan_path", "运行 summary.json"))
    launch_directory_argument = Path(
        require_string(run_summary, "task_launch_dir", "运行 summary.json")
    )
    if not plan_argument.is_absolute() or not launch_directory_argument.is_absolute():
        raise ValueError("运行 summary.json 的 plan_path 与 task_launch_dir 必须是绝对路径。")
    plan_path = plan_argument.resolve(strict=True)
    task_launch_dir = launch_directory_argument.resolve(strict=True)
    run_stamp = require_string(run_summary, "run_stamp", "运行 summary.json")
    slurm_job_id = require_string(run_summary, "slurm_job_id", "运行 summary.json")
    if run_stamp != run_directory.name:
        raise ValueError("运行 summary.json.run_stamp 与运行目录名不一致。")
    for key in (
        "plan_path",
        "plan_name",
        "run_stamp",
        "hostname",
        "slurm_job_id",
        "task_launch_dir",
        "tracking",
        "started_at_utc",
    ):
        if run_record.get(key) != run_summary.get(key):
            raise ValueError(f"run.json.{key} 与运行 summary.json 不一致。")
    stages = run_summary.get("stages")
    if not isinstance(stages, list) or stage_summary not in stages:
        raise ValueError("运行 summary.json 未逐值包含来源阶段 summary.json。")

    launch_path = task_launch_dir / "launch.json"
    launch_command_path = task_launch_dir / "run_cmd.sh"
    launch, launch_sha256 = read_json_and_sha256(launch_path, "launch.json")
    launch_command_sha256 = sha256_file(launch_command_path)
    if (
        launch.get("schema_version") != 1
        or launch.get("launch_id") != task_launch_dir.name
        or launch.get("task_run_stamp") != run_stamp
        or launch.get("slurm_job_id") != slurm_job_id
        or launch.get("task_script") != "训练与运行/sh/train_experiment.sh"
    ):
        raise ValueError("launch.json 未绑定同一训练命令、运行名称与 Slurm Job。")

    release_project_argument = Path(
        require_string(launch, "release_project_root", "launch.json")
    )
    if not release_project_argument.is_absolute():
        raise ValueError("launch.json.release_project_root 必须是绝对路径。")
    release_project_root = release_project_argument.resolve(strict=True)
    if not plan_path.is_relative_to(release_project_root):
        raise ValueError("训练 YAML 不在 launch.json 记录的冻结 release 项目根内。")
    release_manifest_path = release_project_root.parent / "manifest.json"
    release_manifest, release_manifest_sha256 = read_json_and_sha256(
        release_manifest_path,
        "release manifest.json",
    )
    release_manifest_declared_content_sha256 = require_string(
        release_manifest,
        "content_sha256",
        "release manifest.json",
    )
    project_name = require_string(release_manifest, "project_name", "release manifest.json")
    release_id = require_string(release_manifest, "release_id", "release manifest.json")
    if re.fullmatch(r"[0-9a-f]{64}", release_manifest_declared_content_sha256) is None:
        raise ValueError("release manifest.json.content_sha256 必须是 64 位小写十六进制。")
    if (
        release_manifest.get("schema_version") != 1
        or project_name != release_project_root.name
        or release_id != f"{project_name}_{release_manifest_declared_content_sha256[:12]}"
        or release_project_root.parent.name != release_id
    ):
        raise ValueError("release manifest.json 与内容寻址 release 目录不一致。")

    report = {
        "schema_version": 1,
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "resolved_path": resolved_path.as_posix(),
        "resolved_sha256": resolved_sha256,
        "stage_summary_path": stage_summary_path.as_posix(),
        "stage_summary_sha256": stage_summary_sha256,
        "run_record_path": run_record_path.as_posix(),
        "run_record_sha256": run_record_sha256,
        "run_summary_path": run_summary_path.as_posix(),
        "run_summary_sha256": run_summary_sha256,
        "training_plan_path": plan_path.as_posix(),
        "training_plan_sha256": sha256_file(plan_path),
        "launch_path": launch_path.as_posix(),
        "launch_sha256": launch_sha256,
        "launch_command_path": launch_command_path.as_posix(),
        "launch_command_sha256": launch_command_sha256,
        "release_project_root": release_project_root.as_posix(),
        "release_manifest_path": release_manifest_path.as_posix(),
        "release_manifest_sha256": release_manifest_sha256,
        "release_manifest_declared_content_sha256": release_manifest_declared_content_sha256,
        "resolved_stage_name": resolved.get("stage", {}).get("name")
        if isinstance(resolved.get("stage"), dict)
        else None,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, report_path)
        except FileExistsError as error:
            raise FileExistsError(f"报告已存在，拒绝覆盖：{report_path}") from error
    finally:
        temporary_path.unlink(missing_ok=True)
    print(report_path.as_posix())


if __name__ == "__main__":
    main()
