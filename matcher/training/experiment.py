"""执行一个正式 YAML 计划，并在唯一运行目录内顺序训练全部阶段。

run_plan 是程序化入口，main 是 python -m matcher.training.experiment 的命令入口。运行目录
由计划输出根、计划名和运行标识三层组成；根 run.json 在任何阶段开始前发布，根 summary.json
只在全部阶段成功后发布。每个阶段的 resolved、tracking、history、BEST、LAST 和 summary
产物由 run_training_stage 管理。本模块不并行阶段，也不读取 calibration。
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from ..utils.json_artifacts import write_json
from .config import load_experiment_plan, validate_run_stamp
from .engine import run_training_stage


def run_plan(plan_path: Path) -> Path:
    """解析计划、创建不可复用运行目录、顺序执行阶段并发布根摘要。

    - plan_path: 正式实验 YAML 路径；load_experiment_plan 负责字段和业务输出路径门禁。

    - 返回值: 全部阶段成功后返回新建运行目录 Path，等于 plan.output_root / plan.plan_name / run_stamp。

    - run.json.plan_path: plan_path.resolve() 的本机绝对路径字符串。
    - run.json.plan_name: 解析后的安全单级计划名。
    - run.json.run_stamp: TASK_RUN_STAMP 环境变量的安全单级值；环境变量为空或缺失时使用当前 UTC 秒级时间。
    - run.json.hostname: socket.gethostname 返回的主机名字符串，本函数不另做非空门禁。
    - run.json.slurm_job_id: SLURM_JOB_ID 环境变量字符串；缺失时为 null。
    - run.json.task_launch_dir: TASK_LAUNCH_DIR 环境变量字符串；缺失时为 null。
    - run.json.tracking: TrackingConfig 经 dataclasses.asdict 得到的七字段映射。
    - run.json.tracking.provider: 固定字符串 wandb。
    - run.json.tracking.mode: 固定字符串 online。
    - run.json.tracking.entity: W&B 账户或团队名。
    - run.json.tracking.project: W&B 项目名。
    - run.json.tracking.group: W&B 比较组名。
    - run.json.tracking.log_every_global_steps: 参数更新曲线记录间隔。
    - run.json.tracking.init_timeout_seconds: W&B 初始化超时秒数。
    - run.json.started_at_utc: 根运行记录构造时的 UTC ISO 8601 时间字符串。
    - summary.json.plan_path: 与 run.json.plan_path 相同。
    - summary.json.plan_name: 与 run.json.plan_name 相同。
    - summary.json.run_stamp: 与 run.json.run_stamp 相同。
    - summary.json.hostname: 与 run.json.hostname 相同。
    - summary.json.slurm_job_id: 与 run.json.slurm_job_id 相同。
    - summary.json.task_launch_dir: 与 run.json.task_launch_dir 相同。
    - summary.json.tracking: 与 run.json.tracking 同源的七字段映射。
    - summary.json.tracking.provider: 固定字符串 wandb。
    - summary.json.tracking.mode: 固定字符串 online。
    - summary.json.tracking.entity: W&B 账户或团队名。
    - summary.json.tracking.project: W&B 项目名。
    - summary.json.tracking.group: W&B 比较组名。
    - summary.json.tracking.log_every_global_steps: 参数更新曲线记录间隔。
    - summary.json.tracking.init_timeout_seconds: W&B 初始化超时秒数。
    - summary.json.started_at_utc: 与 run.json.started_at_utc 相同。
    - summary.json.finished_at_utc: 最后阶段成功后生成的 UTC ISO 8601 时间字符串。
    - summary.json.stages: 与 YAML stages 顺序相同的 StageSummary 列表。
    - summary.json.stages[*].stage: 阶段安全名称。
    - summary.json.stages[*].best_epoch: 最佳 validation 分数所在的零起始 epoch。
    - summary.json.stages[*].best_score: 有限最佳 validation 选择分数。
    - summary.json.stages[*].last_epoch: 实际完成的最后一个 epoch。
    - summary.json.stages[*].global_step: 阶段结束时累计参数更新数。
    - summary.json.stages[*].stopped_by_patience: 是否因连续坏轮数达到阈值停止。
    - summary.json.stages[*].best_checkpoint: 阶段 BEST.pt 路径字符串。
    - summary.json.stages[*].last_checkpoint: 阶段 LAST.pt 路径字符串。
    - summary.json.stages[*].tracking_run_id: 阶段 W&B run ID。
    - summary.json.stages[*].tracking_url: 阶段 W&B HTTPS 地址。

    mkdir 使用 exist_ok=False，既有同名运行目录会失败。run.json 发布后任一阶段失败时异常直接
    传播，后续阶段和根 summary.json 不执行；已完成阶段与失败阶段已经写出的文件不会回滚。
    """

    # ExperimentPlan，YAML 经严格字段、枚举、批量和路径门禁后的不可变计划对象。
    plan = load_experiment_plan(plan_path)
    # str，安全单级运行目录名；优先采用外部 allocation 提供的 TASK_RUN_STAMP。
    run_stamp = validate_run_stamp(
        os.environ.get("TASK_RUN_STAMP")
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    # Path，本次计划唯一运行根；exist_ok=False 防止复用并覆盖已有训练记录。
    run_directory = plan.output_root / plan.plan_name / run_stamp
    run_directory.mkdir(parents=True, exist_ok=False)
    # dict[str,object]，run.json 的完整根记录，也是最终 summary.json 的公共字段来源。
    metadata = {
        "plan_path": str(plan_path.resolve()),
        "plan_name": plan.plan_name,
        "run_stamp": run_stamp,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "task_launch_dir": os.environ.get("TASK_LAUNCH_DIR"),
        "tracking": asdict(plan.tracking),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_directory / "run.json", metadata)
    # list[StageSummary]，按 YAML stages 顺序追加已成功收口的阶段摘要。
    summaries = []
    for stage in plan.stages:
        # StageSummary，当前阶段已完成训练、W&B 收口并发布 summary.json 后返回的十字段记录。
        summary = run_training_stage(
            stage,
            plan.data,
            plan.tracking,
            run_directory / stage.name,
            run_directory,
        )
        summaries.append(summary)
    # dict[str,object]，根 summary.json；公共运行字段之外增加完成时刻和有序阶段摘要。
    final = {
        **metadata,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "stages": summaries,
    }
    write_json(run_directory / "summary.json", final)
    return run_directory


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    """解析命令参数，并返回只含 plan 路径的 argparse Namespace。

    - argv: 不含程序名的命令参数字符串列表。

    - 返回值.plan: --plan 对应的 Path；该选项必填，但此处不检查文件存在或 YAML 内容。

    argparse 在缺少 --plan、出现未知参数或转换失败时打印帮助并抛出 SystemExit。
    """

    # ArgumentParser，使用模块 Docstring 作为 description；本轮只允许这段非科学帮助文字变化。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """执行命令入口，把成功运行目录打印到标准输出并返回零退出码。

    - argv: 显式命令参数列表；None 时读取 sys.argv[1:]，不含程序名。

    - 返回值: 全部阶段成功后固定为整数 0。

    解析、训练或写盘异常不在此捕获，因此不会打印成功目录，也不会转换为零退出码。
    """

    # argparse.Namespace，只含经 Path 转换的 arguments.plan。
    arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
    # Path，run_plan 完整成功后返回的唯一运行目录。
    run_directory = run_plan(arguments.plan)
    print(run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
