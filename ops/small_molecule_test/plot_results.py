"""从冻结 JSON 绘制 test_smallmolecule Stage1 与 Stage1+Stage2 曲线.

主要入口是 plot_results.本模块不读取 Stage1 原始 NPZ、不加载 Matcher checkpoint,
也不重新计算候选顺序或命中; 图中的每个点都直接来自正式 JSON 的 hit_ratio 数组.
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
    """把若干已经冻结的 top-k 数组绘制到一张 PNG.

    输入参数:
        - curves: list[(label, curve)], 每个 curve 含同长度 top_k 和 hit_ratio 数组.
        - title: str, 图像标题, 明确受体、排序或模型比较范围.
        - output_path: Path, 目标 PNG; 父目录不存在时创建, 现存文件禁止覆盖.

    返回值:
        - None: 图像保存并关闭后返回; 本函数不修改输入曲线.
    """
    if output_path.exists():
        raise FileExistsError(f"派生图已经存在, 禁止覆盖: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9.2, 6.2), dpi=160)
    for label, curve in curves:
        axis.plot(curve["top_k"], curve["hit_ratio"], marker="o", markersize=3, label=label)
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
    """为 Stage1-only 或一个 Matcher 模型生成两张独立排序图.

    输入参数:
        - config_path: Path, test_smallmolecule.yaml 路径.
        - target: str, stage1_only 或配置 models 下的稳定实验标识.

    返回值:
        - output_paths: list[Path], 依次为 Gaussian 与概率均值 PNG 路径.
    """
    config = load_config(config_path)
    evaluation_root = Path(config["outputs"]["evaluation_root"])
    outputs: list[Path] = []
    if target == "stage1_only":
        artifact = read_json(evaluation_root / "stage1_only/stage1_only.json")
        figure_root = evaluation_root / "stage1_only/figures"
        for sorting, filename, title in (
            ("gaussian", "topk_gaussian_stage1_only.png", "Stage1-only: Gaussian ranking"),
            (
                "source_probability_mean",
                "topk_probability_mean_stage1_only.png",
                "Stage1-only: probability-mean ranking",
            ),
        ):
            curves = [
                ("Find_1 real receptor", artifact["curves"][sorting]["real_receptor"]),
                (
                    "Find_1 CryoAtom2 receptor",
                    artifact["curves"][sorting]["cryoatom2_receptor"],
                ),
                (
                    "Emap2lig Find (density-only)",
                    artifact["emap2lig"],
                ),
            ]
            output_path = figure_root / filename
            _plot_curves(curves, title, output_path)
            outputs.append(output_path)
        return outputs

    model_record = config["models"][target]
    result_root = evaluation_root / target / str(model_record["training_run_name"])
    artifact = read_json(result_root / "evaluation.json")
    stage1 = read_json(evaluation_root / "stage1_only/stage1_only.json")
    for sorting, filename, title in (
        ("gaussian", "topk_gaussian_stage1_stage2.png", f"{target}: Gaussian ranking"),
        (
            "source_probability_mean",
            "topk_probability_mean_stage1_stage2.png",
            f"{target}: probability-mean ranking",
        ),
    ):
        curves = []
        for condition, condition_label in (
            ("real_receptor", "real receptor"),
            ("cryoatom2_receptor", "CryoAtom2 receptor"),
        ):
            curves.append(
                (f"Stage1-only, {condition_label}", stage1["curves"][sorting][condition])
            )
            curves.append(
                (
                    f"all-ligand answers, {condition_label}",
                    artifact["curves"][sorting][condition]["all_ligand"],
                )
            )
            curves.append(
                (
                    f"small-molecule answers, {condition_label}",
                    artifact["curves"][sorting][condition]["small_molecule"],
                )
            )
        output_path = result_root / "figures" / filename
        _plot_curves(curves, title, output_path)
        outputs.append(output_path)
    return outputs


def main(argv: list[str] | None = None) -> int:
    """解析命令行, 并只从冻结 JSON 绘制图像.

    输入参数:
        - argv: list[str]|None, 可测试注入的配置与目标名称; None 时读取当前进程参数.

    返回值:
        - exit_code: int, 两张目标图成功保存后固定为 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", required=True)
    arguments = parser.parse_args(argv)
    for path in plot_results(arguments.config, arguments.target):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
