"""执行一个正式 YAML 计划，并在唯一运行目录内顺序训练全部阶段。

``run_plan`` 是程序化主入口，``main`` 是 ``python -m matcher.experiment`` 命令入口。
运行目录固定为 ``plan.output_root / plan.plan_name / run_stamp``。
``run.json`` 保存计划路径、计划名、运行时间戳、主机、Slurm 任务和 launch 目录信息。
``summary.json`` 复用 ``run.json`` 字段，并增加完成时间和各阶段 ``StageSummary``。
每个阶段由 ``run_training_stage`` 管理自己的 resolved、history、BEST、LAST 与 summary 产物。
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

from .experiment_config import load_experiment_plan, validate_run_stamp
from .json_artifacts import write_json
from .training import run_training_stage


def run_plan(plan_path: Path) -> Path:
    """创建计划运行目录，顺序训练所有阶段并写入最终摘要。"""

    plan = load_experiment_plan(plan_path)
    run_stamp = validate_run_stamp(
        os.environ.get("TASK_RUN_STAMP")
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_directory = plan.output_root / plan.plan_name / run_stamp
    run_directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "plan_path": str(plan_path.resolve()),
        "plan_name": plan.plan_name,
        "run_stamp": run_stamp,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "task_launch_dir": os.environ.get("TASK_LAUNCH_DIR"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_directory / "run.json", metadata)
    summaries = []
    for stage in plan.stages:
        summary = run_training_stage(
            stage,
            plan.data,
            run_directory / stage.name,
            run_directory,
        )
        summaries.append(summary)
    final = {
        **metadata,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "stages": summaries,
    }
    write_json(run_directory / "summary.json", final)
    return run_directory


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    """解析唯一正式入口所需的计划路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """执行计划并把唯一结果目录写到标准输出。"""

    arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
    run_directory = run_plan(arguments.plan)
    print(run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
