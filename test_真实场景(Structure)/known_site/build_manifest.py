"""冻结真实配体质心上的 NxU 与 nxu 闭集身份测试清单.

正式入口直接执行本文件. 输出根由命令行参数指定, 每项任务分别保存 PDB 身份集合、
查询点和成本截尾的 PDB. 查询点仅使用沉积配体坐标确定世界位置与评价答案.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-manifest", required=True, type=Path)
parser.add_argument("--data-root", required=True, type=Path)
parser.add_argument("--output-root", required=True, type=Path)
parser.add_argument("--trim-percent", type=float, default=10.0)
args = parser.parse_args()

# 179 个 PDB 的权威 held_out_test_0 集合; 不以 Stage1 selected 候选重定义测试结构.
source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
pdb_ids = source_manifest["summary"]["held_out_test_0"]["pdb_ids"]
pair_list = {
    record["pdb_id"]: record
    for record in (
        json.loads(line)
        for line in (args.data_root / "raw/pair_list.jsonl").read_text(encoding="utf-8").splitlines()
    )
}

# task_records 的每个元素是一个 PDB 的查询 occurrence 和精确顶层 SMILES 去重集合.
task_records: dict[str, list[dict]] = {"NxU": [], "nxu": []}
for pdb_id in pdb_ids:
    occurrences = [
        json.loads(line)
        for line in (args.data_root / "parse" / pdb_id / "occurrences.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    prepared_smiles = {
        record["candidate_id"]: record["smiles"]
        for record in (
            json.loads(line)
            for line in (
                args.data_root
                / "stage1_preparation_box_pool_2/ligand_language_models/prepared"
                / pdb_id
                / "prepared_smiles.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    for task_name in ("NxU", "nxu"):
        # NxU 包含所有原始类别; nxu 只含 type_tag=small_molecule 的 occurrence.
        selected = [
            occurrence
            for occurrence in occurrences
            if task_name == "NxU" or occurrence["type_tag"] == "small_molecule"
        ]
        if not selected:
            continue
        identity_smiles = sorted({prepared_smiles[row["candidate_id"]] for row in selected})
        task_records[task_name].append(
            {
                "pdb_id": pdb_id,
                "occurrences": selected,
                "prepared_smiles": prepared_smiles,
                "identity_smiles": identity_smiles,
                "cost": len(selected) * len(identity_smiles),
                "emdb_id": pair_list[pdb_id]["emdb_id"],
                "resolution_angstrom": pair_list[pdb_id]["resolution"],
            }
        )

# 先完成两项成本排序, 再落盘; 并列时按 PDB ID 升序选择被截去的结构.
frozen: dict[str, dict] = {}
for task_name, records in task_records.items():
    removed_count = math.ceil(len(records) * args.trim_percent / 100)
    descending = sorted(records, key=lambda record: (-record["cost"], record["pdb_id"]))
    removed = descending[:removed_count]
    removed_ids = {record["pdb_id"] for record in removed}
    retained = sorted(
        (record for record in records if record["pdb_id"] not in removed_ids),
        key=lambda record: record["pdb_id"],
    )
    pdb_records = [
        {
            "pdb_id": record["pdb_id"],
            "occurrence_count": len(record["occurrences"]),
            "identity_smiles": record["identity_smiles"],
            "docking_count": record["cost"],
            "emdb_id": record["emdb_id"],
            "resolution_angstrom": record["resolution_angstrom"],
        }
        for record in retained
    ]
    query_records = []
    for record in retained:
        pdb_id = record["pdb_id"]
        with np.load(
            args.data_root / "parse" / pdb_id / "ligand_coords.npz", allow_pickle=False
        ) as coordinates:
            for occurrence in sorted(record["occurrences"], key=lambda row: row["candidate_id"]):
                occurrence_id = occurrence["candidate_id"]
                # (3,), 沉积 present 原子的 XYZ 世界坐标算术平均, 单位 Å; 只定义已知查询点.
                center = coordinates[f"coords_{occurrence_id}"][
                    coordinates[f"present_{occurrence_id}"]
                ].mean(axis=0, dtype=np.float64)
                query_records.append(
                    {
                        "pdb_id": pdb_id,
                        "occurrence_id": occurrence_id,
                        "type_tag": occurrence["type_tag"],
                        "target_smiles": record["prepared_smiles"][occurrence_id],
                        "center_xyz_angstrom": center.tolist(),
                    }
                )
    frozen[task_name] = {
        "pdbs": pdb_records,
        "queries": query_records,
        "excluded": [
            {
                "pdb_id": record["pdb_id"],
                "occurrence_count": len(record["occurrences"]),
                "identity_count": len(record["identity_smiles"]),
                "docking_count": record["cost"],
            }
            for record in removed
        ],
        "summary": {
            "source_pdb_count": len(records),
            "excluded_pdb_count": removed_count,
            "pdb_count": len(pdb_records),
            "query_count": len(query_records),
            "identity_slot_count": sum(len(record["identity_smiles"]) for record in pdb_records),
            "docking_count": sum(record["docking_count"] for record in pdb_records),
            "multi_identity_pdb_count": sum(len(record["identity_smiles"]) > 1 for record in pdb_records),
            "multi_identity_query_count": sum(
                record["occurrence_count"]
                for record in pdb_records
                if len(record["identity_smiles"]) > 1
            ),
        },
    }

args.output_root.mkdir(parents=True, exist_ok=False)
manifest = {
    "schema": "matcher_known_site_v1",
    "source_manifest": str(args.source_manifest),
    "data_root": str(args.data_root),
    "trim_percent": args.trim_percent,
    "tasks": {task_name: value["summary"] for task_name, value in frozen.items()},
}
(args.output_root / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
for task_name, value in frozen.items():
    task_root = args.output_root / task_name
    task_root.mkdir()
    for name, records in (
        ("pdbs.jsonl", value["pdbs"]),
        ("queries.jsonl", value["queries"]),
        ("excluded_pdbs.jsonl", value["excluded"]),
    ):
        with (task_root / name).open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
