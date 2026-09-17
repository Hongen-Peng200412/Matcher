"""只读取冻结评估 JSON, 绘制 Matcher 指标图和两种排序的 top-k 图. """

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    """
    读取冻结评估 JSON.

    - path: UTF-8 JSON 文件路径.
    - return: 顶层对象映射; 顶层不是对象时失败关闭.
    """
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 必须是 JSON 对象。")
    return value


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    """
    保存并关闭一张 Matplotlib 图.

    - figure: 已完成布局和标注的 Figure.
    - path: PNG 输出路径; 父目录不存在时建立.
    - return: 已写入的 path; 固定使用 180 dpi 和紧边界.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


# ================================================================================================


def plot_stage1_only_preview(
    metadata_path: Path, sorting: str, output_path: Path
) -> Path:
    """
    绘制不含任何 Matcher 数值的 Stage1-only 预览图.

    - metadata_path: matcher_stage1_only_preview_v1 JSON.
    - sorting: gaussian 或 source_probability_mean, 只改变 Find_1 的排序字段.
    - output_path: 文件名必须含 stage1_only_preview, 避免被误称为最终图.
    - return: 已保存的 PNG 路径.
    """
    if sorting not in ("gaussian", "source_probability_mean"):
        raise ValueError("sorting 必须是 gaussian 或 source_probability_mean。")
    if "stage1_only_preview" not in Path(output_path).name:
        raise ValueError("预览图文件名必须明确包含 stage1_only_preview。")
    artifact = _read_json(metadata_path)
    if artifact.get("status") != "stage1_only_preview":
        raise ValueError("输入不是 Stage1-only 预览元数据。")
    curves = artifact["curves"][sorting]
    emap_gaussian = artifact["curves"]["gaussian"]["emap2lig"]["hit_count"]
    emap_probability = artifact["curves"]["source_probability_mean"]["emap2lig"]["hit_count"]
    if emap_gaussian != emap_probability:
        raise ValueError("两张预览图引用的 Emap2lig 单曲线必须逐点相同。")
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    styles = (
        ("real_receptor", "Find_1 real-receptor Stage1-only", "#2563eb", "o"),
        ("cryoatom2_receptor", "Find_1 CryoAtom2-receptor Stage1-only", "#ea580c", "s"),
        (
            "emap2lig",
            "Emap2lig Find Stage1-only (density-only, receptor-independent)",
            "#16803c",
            "^",
        ),
    )
    for key, label, color, marker in styles:
        curve = curves[key]
        axis.plot(
            curve["top_k"],
            curve["hit_ratio"],
            label=label,
            color=color,
            marker=marker,
            markevery=2,
            linewidth=2,
        )
    sorting_label = "Gaussian candidate_score" if sorting == "gaussian" else "source_probability_mean"
    axis.set_title(f"stage1_only_preview — Find_1 ranked by {sorting_label}")
    axis.set_xlabel("Top-k candidate count")
    axis.set_ylabel("PDB hit ratio (fixed denominator n=179)")
    axis.set_xticks(range(1, 21))
    axis.set_xlim(1, 20)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8)
    return _save_figure(figure, output_path)


def plot_topk_final(result_path: Path, sorting: str, output_path: Path) -> Path:
    """
    绘制阶段 E 的一张正式 top-k 图.

    - result_path: 状态为 final_with_matcher 的冻结测试 JSON.
    - sorting: gaussian 或 source_probability_mean; 输出文件名必须含该字段.
    - output_path: 正式 PNG 路径.
    - return: 已保存的 PNG 路径.

    图中包含真实受体和 CryoAtom2 各自的 Stage1-only、Stage1+Matcher 四条 Find_1 曲线,
    以及一条 Emap2lig receptor-independent 基线.
    """
    if sorting not in ("gaussian", "source_probability_mean"):
        raise ValueError("sorting 必须是 gaussian 或 source_probability_mean。")
    if sorting not in Path(output_path).name:
        raise ValueError("最终 top-k 图文件名必须明确包含 Find_1 排序字段。")
    artifact = _read_json(result_path)
    if artifact.get("status") != "final_with_matcher":
        raise ValueError("最终 top-k 图必须读取真实 Matcher 冻结评估产物。")
    curves = artifact["curves"][sorting]
    required = {
        "real_stage1_only",
        "real_stage1_matcher",
        "cryo_stage1_only",
        "cryo_stage1_matcher",
        "emap2lig",
    }
    if set(curves) != required:
        raise ValueError(f"最终 top-k 曲线必须精确为 {sorted(required)}。")
    other_sorting = "source_probability_mean" if sorting == "gaussian" else "gaussian"
    if curves["emap2lig"]["hit_count"] != artifact["curves"][other_sorting]["emap2lig"]["hit_count"]:
        raise ValueError("两张最终图引用的 Emap2lig 单曲线必须逐点相同。")
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    for key in sorted(required):
        curve = curves[key]
        axis.plot(curve["top_k"], curve["hit_ratio"], label=curve["name"], linewidth=2)
    sorting_label = "Gaussian candidate_score" if sorting == "gaussian" else "source_probability_mean"
    axis.set_title(f"Find_1 ranked by {sorting_label}")
    axis.set_xlabel("Top-k candidate count")
    axis.set_ylabel("PDB hit ratio (fixed denominator n=179)")
    axis.set_xticks(range(1, 21))
    axis.set_xlim(1, 20)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    return _save_figure(figure, output_path)


def plot_metric_bars(result_path: Path, output_path: Path) -> Path:
    """
    绘制真实受体与 CryoAtom2 的正式指标图.

    - result_path: 状态为 final_with_matcher 的冻结测试 JSON.
    - output_path: 七栏图 PNG 输出路径.
    - return: 已保存的 PNG 路径.

    主轴依次显示总体和五类 SMILES 准确率、前景 F1; 右侧独立轴显示前景 PRAUC. 零支持
    类别绘制为 N/A 并明确标注 n=0, 不把它转换成零准确率.
    """
    artifact = _read_json(result_path)
    if artifact.get("status") != "final_with_matcher":
        raise ValueError("七栏图必须读取真实 Matcher 冻结评估产物。")
    labels = [
        "SMILES overall",
        "small molecule",
        "sugar",
        "metal/ion",
        "peptide-like",
        "nucleotide-like",
        "foreground F1",
    ]
    conditions = ("real_receptor", "cryoatom2_receptor")
    figure, (axis, pr_axis) = plt.subplots(
        1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [7, 1]}
    )
    positions = np.arange(len(labels))
    width = 0.36
    for offset, condition in zip((-width / 2, width / 2), conditions, strict=True):
        metrics = artifact["metrics"][condition]
        smiles = metrics["smiles_match"]
        groups = (
            "overall",
            "small_molecule",
            "sugar",
            "ion",
            "peptide_like",
            "nucleotide_like",
        )
        values = [smiles[group]["accuracy"] for group in groups]
        values.append(metrics["foreground"]["f1"])
        supports = [smiles[group]["eligible_count"] for group in groups]
        rendered = [np.nan if value is None else value for value in values]
        bars = axis.bar(positions + offset, rendered, width, label=condition)
        for bar, value, support in zip(bars, values[:6], supports, strict=False):
            text = f"N/A\n(n={support})" if value is None else f"n={support}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                0.02 if value is None else bar.get_height(),
                text,
                ha="center",
                va="bottom",
                fontsize=7,
            )
        pr_axis.scatter([condition], [metrics["foreground_prauc"]], marker="D", s=55)
    axis.set_xticks(positions, labels, rotation=24, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy or F1")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    pr_axis.set_title("PRAUC")
    pr_axis.set_ylim(0, 1)
    pr_axis.grid(axis="y", alpha=0.25)
    return _save_figure(figure, output_path)
