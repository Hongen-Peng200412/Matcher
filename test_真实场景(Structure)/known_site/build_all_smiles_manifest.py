"""为冻结 nxu 查询点建立 PDB 全类别精确 SMILES 身份轴，不改变查询点。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--query-manifest-root", required=True, type=Path)
parser.add_argument("--data-root", required=True, type=Path)
parser.add_argument("--output-root", required=True, type=Path)
args = parser.parse_args()

# list[dict], 长度 84; 每项是原 nxu 清单的一个 PDB 及其小分子身份轴, 如 9llg.
# 新清单只扩展候选身份, 不复制或重排原 queries.jsonl 的 357 个查询点.
small_pdbs = [
    json.loads(line)
    for line in (args.query_manifest_root / "nxu/pdbs.jsonl").read_text(encoding="utf-8").splitlines()
]
# list[dict], 与 small_pdbs 的 PDB 顺序一致; 每项最终成为新 pdbs.jsonl 的一行.
records = []
for pdb in small_pdbs:
    pdb_id = pdb["pdb_id"]
    occurrences = [
        json.loads(line)
        for line in (args.data_root / "parse" / pdb_id / "occurrences.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    prepared_path = (
        args.data_root
        / "stage1_preparation_box_pool_2/ligand_language_models/prepared"
        / pdb_id
        / "prepared_smiles.jsonl"
    )
    # candidate_id -> 顶层 PreparedLigand.smiles; 前者定位 occurrence, 后者定义身份.
    prepared = {
        item["candidate_id"]: item["smiles"]
        for item in (json.loads(line) for line in prepared_path.read_text(encoding="utf-8").splitlines())
    }
    # 精确 SMILES -> 该 PDB 内全部 occurrence 类别; 同一字符串跨类别时不能定义唯一掩码.
    categories_by_smiles: dict[str, set[str]] = {}
    for occurrence in occurrences:
        smiles = prepared[occurrence["candidate_id"]]
        categories_by_smiles.setdefault(smiles, set()).add(occurrence["type_tag"])
    # 仅收集类别不唯一的 SMILES; 如同一字符串同时标为 small_molecule 和 sugar.
    shared = {smiles: sorted(categories) for smiles, categories in categories_by_smiles.items()
              if len(categories) != 1}
    if shared:
        raise ValueError(f"{pdb_id} 的精确 SMILES 跨类别：{shared}")

    # 两条身份轴都按精确字符串排序; identity_categories[i] 属于 identity_smiles[i].
    identity_smiles = sorted(categories_by_smiles)
    # 原小分子身份轴是完整身份轴的子集; 不是重新筛选查询 occurrence.
    small_smiles = [smiles for smiles in identity_smiles
                    if "small_molecule" in categories_by_smiles[smiles]]
    if small_smiles != pdb["identity_smiles"]:
        raise ValueError(f"{pdb_id} 的原冻结小分子身份轴与来源不一致")
    records.append({
        "pdb_id": pdb_id,
        "identity_smiles": identity_smiles,
        "identity_categories": [next(iter(categories_by_smiles[smiles])) for smiles in identity_smiles],
        "small_molecule_identity_smiles": small_smiles,
    })

args.output_root.mkdir(parents=True, exist_ok=False)
with (args.output_root / "pdbs.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
print(f"冻结 PDB={len(records)}, 全类别身份槽={sum(len(x['identity_smiles']) for x in records)}", flush=True)
