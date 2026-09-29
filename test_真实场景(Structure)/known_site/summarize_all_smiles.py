"""读取冻结的 nxu 两种身份输入预测，汇总固定与完整可用分母并绘制比较图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    "density_only-0/small_input",
    "density_only-0/all_input_unmasked",
    "density_only-0/all_input_small_masked",
    "density_only-106/small_input",
    "density_only-106/all_input_unmasked",
    "density_only-106/all_input_small_masked",
    "EMERALD-ID/small_input",
)
GROUPS = ("1", "2", "3", "4", "5", ">5")


def measure(keys: list[str], predictions: dict[str, dict[str, dict]], method: str) -> dict:
    """从指定查询键集合计算一个方法的逐点正确数、支持数和准确率。

    - keys: 冻结 nxu 查询键，格式为 pdb_id/occurrence_id；列表决定本次分母。
    - predictions: 方法名到查询键、正确与完整可用布尔值的两层映射。
    - method: METHODS 中的方法与身份输入读出组合。
    - return: correct 为正确点数，support 为 len(keys)，accuracy 在无点时为 None。
    """
    correct = sum(predictions[method][key]["correct"] for key in keys)
    return {"correct": correct, "support": len(keys),
            "accuracy": correct / len(keys) if keys else None}


def draw_curves(report: dict, readout: str, comparison: str, output_path: Path) -> None:
    """为一种受体、完整身份读出和分母口径绘制两方法的分组准确率。

    - report: 当前受体条件的汇总；grouped 中按小分子身份数保存正确数与支持数。
    - readout: unmasked 或 small_masked，决定 106 通道模型读取哪个完整身份答案。
    - comparison: overall 比较固定和各自完整子集；paired 比较同一共同子集。
    - output_path: PNG 目标，坐标横轴固定为 1、2、3、4、5、>5。
    """
    matcher = f"density_only-106/all_input_{readout}"
    emerald = "EMERALD-ID/small_input"
    curves = (
        [(matcher, "fixed", "#1f77b4", "-"), (matcher, "own_complete", "#1f77b4", "--"),
         (emerald, "fixed", "#ff7f0e", "-"), (emerald, "own_complete", "#ff7f0e", "--")]
        if comparison == "overall" else
        [(matcher, "common", "#1f77b4", "-"), (emerald, "common", "#ff7f0e", "-")]
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    positions = list(range(len(GROUPS)))
    for curve_index, (method, denominator, color, line_style) in enumerate(curves):
        values = [report["grouped"][group][method][denominator] for group in GROUPS]
        heights = [value["accuracy"] if value["support"] else np.nan for value in values]
        label = f"{method.split('/')[0]} — {denominator.replace('_', ' ')}"
        ax.plot(positions, heights, marker="o", color=color, linestyle=line_style, label=label)
        for position, value in zip(positions, values):
            if value["support"]:
                ax.annotate(f"{value['correct']}/{value['support']}",
                            (position, value["accuracy"]), textcoords="offset points",
                            xytext=(0, 7 + 9 * (curve_index % 2)), ha="center", fontsize=8, color=color)
            else:
                ax.text(position, 0.025 + 0.055 * curve_index, "N/A", ha="center", fontsize=8, color=color)
    ax.set_xticks(positions, GROUPS)
    ax.set_xlim(-0.35, len(GROUPS) - 0.65)
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("Distinct small-molecule SMILES in the frozen PDB")
    ax.set_ylabel("Known-site identity micro Top-1")
    ax.set_title(f"{report['condition']} / all-SMILES {readout} / {comparison}")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# ================================================================================================


def main() -> None:
    """从原 357 查询键读取七种方法与读出，保存两层分母及四张图。

    - --manifest-root: 原冻结 nxu 查询文件和小分子 PDB 身份文件的共同根。
    - --all-identity-manifest: 扩展后的 84 个 PDB 全类别身份轴。
    - --prediction-root: 原真实受体小分子输入 Matcher 与 EMERALD-ID 逐点根。
    - --cryo-root: 原 CryoAtom2 小分子输入 106 通道 Matcher 与 EMERALD-ID 逐点根。
    - --all-prediction-root: 本次完整身份输入的真实受体与 CryoAtom2 逐点根。
    - --condition: real 或 CryoAtom2；单通道模型在两者中读取同一正式预测。
    - --output-root: 当前受体条件 summary.json 与四张 PNG 的新目录。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--all-identity-manifest", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--cryo-root", required=True, type=Path)
    parser.add_argument("--all-prediction-root", required=True, type=Path)
    parser.add_argument("--condition", required=True, choices=("real", "CryoAtom2"))
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    small_pdbs = {
        item["pdb_id"]: item for item in (
            json.loads(line) for line in (args.manifest_root / "nxu/pdbs.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    all_pdbs = {
        item["pdb_id"]: item for item in (
            json.loads(line) for line in args.all_identity_manifest.read_text(encoding="utf-8").splitlines()
        )
    }
    queries = [
        json.loads(line)
        for line in (args.manifest_root / "nxu/queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    predictions: dict[str, dict[str, dict]] = {method: {} for method in METHODS}
    small_count_by_key: dict[str, int] = {}
    frozen_keys: list[str] = []
    for query in queries:
        pdb_id = query["pdb_id"]
        occurrence_id = query["occurrence_id"]
        key = f"{pdb_id}/{occurrence_id}"
        frozen_keys.append(key)
        small_count_by_key[key] = len(small_pdbs[pdb_id]["identity_smiles"])
        expected_small = set(small_pdbs[pdb_id]["identity_smiles"])
        expected_all = set(all_pdbs[pdb_id]["identity_smiles"])
        small_root = args.prediction_root if args.condition == "real" else args.cryo_root
        all_real = args.all_prediction_root / "real"
        all_current = args.all_prediction_root / args.condition
        # 同一 357 点分别读取原小分子输入、完整身份输入及官方评分；没有前向或对接副作用。
        files = {
            "density_only-0/small_input": args.prediction_root / "density_only-0/nxu" / pdb_id / f"{occurrence_id}.json",
            "density_only-0/all_input": all_real / "density_only-0/nxu" / pdb_id / f"{occurrence_id}.json",
            "density_only-106/small_input": small_root / "density_only-106/nxu" / pdb_id / f"{occurrence_id}.json",
            "density_only-106/all_input": all_current / "density_only-106/nxu" / pdb_id / f"{occurrence_id}.json",
            "EMERALD-ID/small_input": small_root / "EMERALD-ID/nxu" / pdb_id / str(occurrence_id) / "result.json",
        }
        results = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in files.items()}
        for model in ("density_only-0", "density_only-106"):
            small_result = results[f"{model}/small_input"]
            all_result = results[f"{model}/all_input"]
            for method, result, answer, expected in (
                (f"{model}/small_input", small_result, small_result["predicted_smiles"], expected_small),
                (f"{model}/all_input_unmasked", all_result, all_result["predicted_smiles"], expected_all),
                (f"{model}/all_input_small_masked", all_result,
                 all_result["predicted_small_molecule_smiles"], expected_all),
            ):
                predictions[method][key] = {
                    "correct": answer == query["target_smiles"],
                    "complete": set(result["scored_smiles"]) == expected,
                }
        emerald = results["EMERALD-ID/small_input"]
        predictions["EMERALD-ID/small_input"][key] = {
            "correct": emerald["predicted_smiles"] == query["target_smiles"],
            "complete": set(emerald["scored_smiles"]) == expected_small,
        }

    # 共同子集取两个模型各三种读出与官方评分的交集，按冻结查询顺序存储可复核键。
    common_keys = [key for key in frozen_keys
                   if all(predictions[method][key]["complete"] for method in METHODS)]
    report: dict = {
        "condition": args.condition,
        "query_count": len(frozen_keys),
        "common_query_keys": common_keys,
        "methods": {},
        "grouped": {},
    }
    for method in METHODS:
        own_keys = [key for key in frozen_keys if predictions[method][key]["complete"]]
        report["methods"][method] = {
            "fixed": measure(frozen_keys, predictions, method),
            "own_complete": measure(own_keys, predictions, method),
            "common": measure(common_keys, predictions, method),
            "own_complete_query_keys": own_keys,
        }
    for group in GROUPS:
        group_keys = [key for key in frozen_keys if (
            str(small_count_by_key[key]) if small_count_by_key[key] <= 5 else ">5") == group]
        report["grouped"][group] = {}
        for method in METHODS:
            own = set(report["methods"][method]["own_complete_query_keys"])
            common = set(common_keys)
            report["grouped"][group][method] = {
                "fixed": measure(group_keys, predictions, method),
                "own_complete": measure([key for key in group_keys if key in own], predictions, method),
                "common": measure([key for key in group_keys if key in common], predictions, method),
            }

    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for readout in ("unmasked", "small_masked"):
        for comparison in ("overall", "paired"):
            draw_curves(report, readout, comparison,
                        args.output_root / f"{readout}_{comparison}.png")


if __name__ == "__main__":
    main()
