"""从冻结 test_smallmolecule manifest 生成 Stage1-only 正式统计.

主要入口是 evaluate_stage1.它只读取 manifest 中的三套候选, 不加载 Matcher checkpoint,
不生成图像, 也不重新计算候选与 occurrence 的覆盖关系.
"""

from __future__ import annotations

import argparse
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import load_config, read_json, read_jsonl, sha256_file, write_json
from .ranking import build_topk_curve


def _checkpoint_counts(curve: dict[str, Any]) -> list[int]:
    """提取 k=1、3、5、10、20 的命中数.

    输入参数:
        - curve: dict[str, Any], 含 k=1 至 20 顺序命中数的单条冻结曲线.

    返回值:
        - counts: list[int], 五个核对位置的命中数, 用于复现计划阶段只读预演.
    """
    positions = [0, 2, 4, 9, 19]
    return [int(curve["hit_count"][position]) for position in positions]


def evaluate_stage1(config_path: Path) -> Path:
    """生成固定 77-PDB 分母的 Find_1 与 Emap2lig Stage1-only 结果.

    输入参数:
        - config_path: Path, 显式给出 manifest、正式结果和预期计数的 YAML.

    返回值:
        - output_path: Path, 已原子发布的 stage1_only.json.

    输出字段:
        - schema: str, 固定为 matcher_test_smallmolecule_stage1_v1.
        - manifest: object, 保存 manifest 绝对路径和 SHA-256.
        - curves.gaussian: object, 真实受体与 CryoAtom2 按 Gaussian 分数得到的曲线.
        - curves.source_probability_mean: object, 两种 Find_1 受体按 blob 内概率均值得到的曲线.
        - emap2lig: object, 唯一的 receptor-independent Emap2lig 原生均值曲线.
        - traces: object, 每种来源和排序下逐 PDB 的全部候选尝试轨迹.
    """
    config = load_config(config_path)
    manifest_path = Path(config["outputs"]["manifest_root"]) / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest["schema"] != "matcher_test_smallmolecule_v1":
        raise ValueError("输入 manifest schema 不正确.")
    root = manifest_path.parent
    pdb_ids = [str(pdb_id) for pdb_id in manifest["pdb_ids"]]
    real_path = root / manifest["files"]["real_candidates"]
    cryo_path = root / manifest["files"]["cryoatom2_candidates"]
    emap_path = root / manifest["files"]["emap2lig_candidates"]
    real = read_jsonl(real_path)
    cryo = read_jsonl(cryo_path)
    emap = read_jsonl(emap_path)

    curves: dict[str, dict[str, dict[str, Any]]] = {
        "gaussian": {},
        "source_probability_mean": {},
    }
    traces: dict[str, Any] = {
        "gaussian": {},
        "source_probability_mean": {},
    }
    for sorting in ("gaussian_score", "source_probability_mean"):
        output_key = "gaussian" if sorting == "gaussian_score" else sorting
        for condition, path, candidates, label in (
            ("real_receptor", real_path, real, "Find_1 real-receptor Stage1-only"),
            (
                "cryoatom2_receptor",
                cryo_path,
                cryo,
                "Find_1 CryoAtom2-receptor Stage1-only",
            ),
        ):
            curve, condition_traces = build_topk_curve(
                label,
                path,
                pdb_ids,
                candidates,
                sorting,
                lambda candidate: candidate["target_scope"] == "stage3_foreground",
            )
            curves[output_key][condition] = curve
            traces[output_key][condition] = condition_traces

    emap_curve, emap_traces = build_topk_curve(
        "Emap2lig Find Stage1-only (density-only, receptor-independent)",
        emap_path,
        pdb_ids,
        emap,
        "source_probability_mean",
        lambda candidate: candidate["target_scope"] == "stage3_foreground",
    )

    observed = {
        "real_gaussian": _checkpoint_counts(curves["gaussian"]["real_receptor"]),
        "real_probability_mean": _checkpoint_counts(
            curves["source_probability_mean"]["real_receptor"]
        ),
        "cryoatom2_gaussian": _checkpoint_counts(
            curves["gaussian"]["cryoatom2_receptor"]
        ),
        "cryoatom2_probability_mean": _checkpoint_counts(
            curves["source_probability_mean"]["cryoatom2_receptor"]
        ),
        "emap2lig_probability_mean": _checkpoint_counts(emap_curve),
    }
    expected = {
        key: [int(value) for value in values]
        for key, values in config["expected"]["stage1_checkpoint_hit_counts"].items()
    }
    if observed != expected:
        raise RuntimeError(f"Stage1-only 只读预演未复现: observed={observed}, expected={expected}")

    artifact = {
        "schema": "matcher_test_smallmolecule_stage1_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "runtime": {
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "task_project_root": os.environ.get("TASK_PROJECT_ROOT"),
            "task_launch_dir": os.environ.get("TASK_LAUNCH_DIR"),
        },
        "coverage_threshold": 0.30,
        "denominator_pdb_count": len(pdb_ids),
        "candidate_order": (
            "candidate_selected=true 候选按分数降序和 source_blob_index 升序排列, "
            "随后追加按相同规则排列的 candidate_selected=false 候选"
        ),
        "attempt_rule": (
            "背景与 Stage3 目标候选消耗一次尝试; 非 Stage3 前景候选免费跳过"
        ),
        "curves": curves,
        "emap2lig": emap_curve,
        "traces": {**traces, "emap2lig": emap_traces},
        "checkpoint_hit_counts": observed,
    }
    output_path = Path(config["outputs"]["evaluation_root"]) / "stage1_only/stage1_only.json"
    if output_path.exists():
        raise FileExistsError(f"Stage1-only 正式结果已经存在, 禁止覆盖: {output_path}")
    return write_json(output_path, artifact)


def main(argv: list[str] | None = None) -> int:
    """解析命令行并写出 Stage1-only 正式统计.

    输入参数:
        - argv: list[str]|None, 可测试注入的参数; None 时读取当前进程参数.

    返回值:
        - exit_code: int, 正式 JSON 成功发布后固定为 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(evaluate_stage1(arguments.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
