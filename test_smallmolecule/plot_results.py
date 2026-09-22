"""从冻结 JSON 绘制 Stage3 小分子测试集的 top-k 曲线.

主要入口是 plot_results. 本模块只读取 stage1_only.json 或 evaluation.json 中的
top_k 与 hit_ratio, 不加载模型, 也不重新计算候选顺序或命中.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .artifacts import load_config, read_json


def _plot_curves(
    curves: list[tuple[str, dict[str, Any]]],
    title: str,
    output_path: Path,
) -> None:
    """把若干冻结 top-k 曲线绘制到一张 PNG.

    输入参数:
        - curves: list[tuple[str,dict]], 每项包含图例文字和具有 top_k、hit_ratio 的曲线.
        - title: str, 图像标题.
        - output_path: Path, PNG 输出路径.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9.2, 6.2), dpi=160)
    for label, curve in curves:
        axis.plot(
            curve["top_k"],
            curve["hit_ratio"],
            marker="o",
            markersize=3,
            label=label,
        )
    axis.set_xlabel("Number of charged attempts k")
    axis.set_ylabel("PDB hit ratio (denominator = 77)")
    axis.set_title(title)
    axis.set_xlim(1, 20)
    axis.set_xticks([1, 3, 5, 10, 15, 20])
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def plot_results(config_path: Path, target: str) -> list[Path]:
    """为 Stage1-only 或一个 Matcher 实验生成两张排序图.

    输入参数:
        - config_path: Path, 小分子测试 YAML.
        - target: str, stage1_only 或 config.models 中的稳定实验标识.

    返回值:
        - output_paths: list[Path], Gaussian 与来源概率均值 PNG 路径.
    """
    config = load_config(config_path)
    evaluation_root = Path(config["outputs"]["evaluation_root"])
    outputs: list[Path] = []
    if target == "stage1_only":
        artifact = read_json(evaluation_root / "stage1_only/stage1_only.json")
        figure_root = evaluation_root / "stage1_only/figures"
        for sorting, filename, title in (
            (
                "gaussian",
                "topk_gaussian_stage1_only.png",
                "Stage1-only: Gaussian ranking",
            ),
            (
                "source_probability_mean",
                "topk_probability_mean_stage1_only.png",
                "Stage1-only: probability-mean ranking",
            ),
        ):
            _plot_curves(
                [
                    (
                        "Find_1 real receptor",
                        artifact["curves"][sorting]["real_receptor"],
                    ),
                    (
                        "Find_1 CryoAtom2 receptor",
                        artifact["curves"][sorting]["cryoatom2_receptor"],
                    ),
                    ("Emap2lig Find (density-only)", artifact["emap2lig"]),
                ],
                title,
                figure_root / filename,
            )
            outputs.append(figure_root / filename)
        return outputs

    model_record = config["models"][target]
    result_root = evaluation_root / target / str(model_record["training_run_name"])
    artifact = read_json(result_root / "evaluation.json")
    stage1 = read_json(evaluation_root / "stage1_only/stage1_only.json")
    for sorting, filename, title in (
        (
            "gaussian",
            "topk_gaussian_stage1_stage2.png",
            f"{target}: Gaussian ranking",
        ),
        (
            "source_probability_mean",
            "topk_probability_mean_stage1_stage2.png",
            f"{target}: probability-mean ranking",
        ),
    ):
        curves: list[tuple[str, dict[str, Any]]] = []
        for condition, label in (
            ("real_receptor", "real receptor"),
            ("cryoatom2_receptor", "CryoAtom2 receptor"),
        ):
            curves.extend(
                [
                    (
                        f"Stage1-only, {label}",
                        stage1["curves"][sorting][condition],
                    ),
                    (
                        f"all-ligand answers, {label}",
                        artifact["curves"][sorting][condition]["all_ligand"],
                    ),
                    (
                        f"small-molecule answers, {label}",
                        artifact["curves"][sorting][condition]["small_molecule"],
                    ),
                ]
            )
        output_path = result_root / "figures" / filename
        _plot_curves(curves, title, output_path)
        outputs.append(output_path)
    return outputs


def main(argv: list[str] | None = None) -> int:
    """解析配置和绘图目标, 并生成两张 top-k 图片."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", required=True)
    arguments = parser.parse_args(argv)
    for path in plot_results(arguments.config, arguments.target):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
