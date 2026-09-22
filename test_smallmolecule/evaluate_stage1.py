"""从小分子测试清单生成 Stage1-only top-k 曲线与候选轨迹.

主要入口是 evaluate_stage1. 本模块只读取清单中的三套候选, 不加载 Matcher
checkpoint, 不重新计算候选与 occurrence 的覆盖关系.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .artifacts import load_config, read_json, read_jsonl, write_json
from .ranking import build_topk_curve


def evaluate_stage1(config_path: Path) -> Path:
    """生成 Find_1 与 Emap2lig 的 Stage1-only 正式结果.

    输入参数:
        - config_path: Path, 小分子测试 YAML.

    返回值:
        - output_path: Path, 写出的 stage1_only.json.

    输出字段:
        - coverage_threshold: float, 候选与 occurrence 共用的双向覆盖阈值.
        - curves: object, 真实受体与 CryoAtom2 在 Gaussian 和来源概率均值排序下的 Stage1-only 曲线.
        - emap2lig: object, Emap2lig 使用原生概率均值排序的单条 receptor-independent 曲线.
        - traces: object, 每种来源与排序下逐 PDB 的完整候选顺序和计费状态.
    """
    config = load_config(config_path)
    manifest_path = Path(config["outputs"]["manifest_root"]) / "manifest.json"
    manifest = read_json(manifest_path)
    root = manifest_path.parent
    pdb_ids = [str(pdb_id) for pdb_id in manifest["pdb_ids"]]
    real_candidates = read_jsonl(root / manifest["files"]["real_candidates"])
    cryoatom2_candidates = read_jsonl(root / manifest["files"]["cryoatom2_candidates"])
    emap2lig_candidates = read_jsonl(root / manifest["files"]["emap2lig_candidates"])

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
        for condition, candidates in (
            ("real_receptor", real_candidates),
            ("cryoatom2_receptor", cryoatom2_candidates),
        ):
            curve, condition_traces = build_topk_curve(
                pdb_ids,
                candidates,
                sorting,
                lambda candidate: candidate["target_scope"] == "stage3_foreground",
            )
            curves[output_key][condition] = curve
            traces[output_key][condition] = condition_traces

    emap2lig_curve, emap2lig_traces = build_topk_curve(
        pdb_ids,
        emap2lig_candidates,
        "source_probability_mean",
        lambda candidate: candidate["target_scope"] == "stage3_foreground",
    )
    output_path = (Path(config["outputs"]["evaluation_root"]) / "stage1_only/stage1_only.json")
    return write_json(
        output_path,
        {
            "coverage_threshold": float(manifest["coverage_threshold"]),
            "curves": curves,
            "emap2lig": emap2lig_curve,
            "traces": {**traces, "emap2lig": emap2lig_traces},
        },
    )


def main(argv: list[str] | None = None) -> int:
    """解析配置路径并写出 Stage1-only 结果."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(evaluate_stage1(arguments.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
