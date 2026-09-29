"""从一项冻结查询的三方法逐点预测派生两个身份结果视图与图片。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("density_only-0", "density_only-106", "EMERALD-ID")
CATEGORIES = ("all", "small_molecule", "sugar", "ion", "peptide_like", "nucleotide_like")


def read_jsonl(path: Path) -> list[dict]:
    """按冻结文件顺序读出一个任务的 PDB 或 occurrence 记录。

    - path: pdbs.jsonl 或 queries.jsonl。
    - return: 原顺序的字典列表。
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def summarize_view(queries: list[dict], pdbs: list[dict], prediction_root: Path,
                   density0_root: Path,
                   task_name: str, view_name: str) -> dict:
    """以固定查询点为分母，聚合三方法的逐点身份正确率和评分覆盖。

    - queries: 当前视图的原始查询点；不随方法的失败或资产缺失删除。
    - pdbs: 当前视图的 PDB 记录，保留完整候选 SMILES 集合。
    - prediction_root: 三方法逐查询结果的共同根。
    - density0_root: 单通道模型逐点结果根；CryoAtom2 的同输入预测直接读取真实受体冻结结果。
    - task_name: NxU 或 nxu；两项任务的 PDB 与身份集合独立。
    - view_name: all_retained 或 multi_identity；只改变汇总筛选。
    - return: 三方法可从逐点 JSON 重建的正确数、类别、PDB 平均与覆盖率。
    """
    identity_by_pdb = {pdb["pdb_id"]: set(pdb["identity_smiles"]) for pdb in pdbs}
    report = {"task": task_name, "view": view_name, "pdb_count": len(pdbs),
              "query_count": len(queries), "methods": {}}
    for method in METHODS:
        category: dict[str, dict[str, int]] = {
            name: {"correct": 0, "support": 0} for name in CATEGORIES
        }
        pdb_correct: dict[str, int] = defaultdict(int)
        pdb_total: dict[str, int] = defaultdict(int)
        scored_queries = 0
        fully_scored_queries = 0
        scored_pairs = 0
        total_pairs = 0
        for query in queries:
            pdb_id = query["pdb_id"]
            occurrence_id = query["occurrence_id"]
            if method == "density_only-0":
                path = density0_root / method / task_name / pdb_id / f"{occurrence_id}.json"
            elif method == "EMERALD-ID":
                path = prediction_root / method / task_name / pdb_id / str(occurrence_id) / "result.json"
            else:
                path = prediction_root / method / task_name / pdb_id / f"{occurrence_id}.json"
            # 缺失文件不是一次有记录的评分失败；结果发布前必须先补齐或明确记录失败。
            result = json.loads(path.read_text(encoding="utf-8"))
            successful = set(result["scored_smiles"])
            expected = identity_by_pdb[pdb_id]
            scored_pairs += len(successful)
            total_pairs += len(expected)
            scored_queries += bool(successful)
            fully_scored_queries += successful == expected
            correct = result["predicted_smiles"] == query["target_smiles"]
            pdb_correct[pdb_id] += correct
            pdb_total[pdb_id] += 1
            for category_name in ("all", query["type_tag"]):
                category[category_name]["support"] += 1
                category[category_name]["correct"] += correct
        for values in category.values():
            values["accuracy"] = values["correct"] / values["support"] if values["support"] else None
        report["methods"][method] = {
            "correct": category["all"]["correct"],
            "denominator": len(queries),
            "micro_top1": category["all"]["accuracy"],
            "pdb_macro_top1": float(np.mean([pdb_correct[pdb] / pdb_total[pdb] for pdb in pdb_total])),
            "scored_query_count": scored_queries,
            "fully_scored_query_count": fully_scored_queries,
            "scored_identity_pairs": scored_pairs,
            "total_identity_pairs": total_pairs,
            "scored_identity_pair_coverage": scored_pairs / total_pairs,
            "categories": category,
        }
    return report


def draw_metrics(report: dict, path: Path) -> None:
    """绘制整体与五类配体的逐点 Top-1，标出各自支持点数。

    - report: summarize_view 的冻结聚合值。
    - path: 本视图的 PNG 输出位置。
    - return: None；无支持类别显示 N/A，不当作零正确率。
    """
    positions = np.arange(len(CATEGORIES))
    width = 0.24
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for index, method in enumerate(METHODS):
        values = report["methods"][method]["categories"]
        heights = [values[name]["accuracy"] if values[name]["accuracy"] is not None else 0.0
                   for name in CATEGORIES]
        bars = ax.bar(positions + (index - 1) * width, heights, width, label=method)
        for bar, name in zip(bars, CATEGORIES):
            item = values[name]
            label = (f"{item['correct']}/{item['support']}" if item["support"] else "N/A")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    label, ha="center", va="bottom", rotation=90, fontsize=7)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("Known-site identity micro Top-1")
    ax.set_xticks(positions, CATEGORIES)
    ax.set_title(f"{report['task']} / {report['view']} — {report['pdb_count']} PDB, "
                 f"{report['query_count']} fixed queries")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    """从指定任务的相同预测派生主视图与多身份视图，不触发模型或对接。

    - --manifest-root: build_manifest.py 的正式冻结清单。
    - --task-name: 本次只汇总 NxU 或 nxu；小分子可先单独收口。
    - --prediction-root: 三方法正式逐点预测的共同根。
    - --output-root: 两项任务各自两个 summary.json 与分类图的共同根。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--task-name", required=True, choices=("NxU", "nxu"))
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--density0-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    density0_root = args.density0_root if args.density0_root is not None else args.prediction_root
    task_name = args.task_name
    task_root = args.manifest_root / task_name
    pdbs = read_jsonl(task_root / "pdbs.jsonl")
    queries = read_jsonl(task_root / "queries.jsonl")
    for view_name in ("all_retained", "multi_identity"):
        chosen_pdbs = (pdbs if view_name == "all_retained"
                       else [pdb for pdb in pdbs if len(pdb["identity_smiles"]) > 1])
        chosen_ids = {pdb["pdb_id"] for pdb in chosen_pdbs}
        chosen_queries = [query for query in queries if query["pdb_id"] in chosen_ids]
        report = summarize_view(chosen_queries, chosen_pdbs, args.prediction_root, density0_root,
                                task_name, view_name)
        output = args.output_root / task_name / view_name
        output.mkdir(parents=True, exist_ok=False)
        (output / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        draw_metrics(report, output / "identity_metrics.png")


if __name__ == "__main__":
    main()
