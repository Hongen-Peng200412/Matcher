"""在 9nal 多身份点重复官方固定种子运行，比较两次有效评分与排名。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_emerald_query import run_query


def main() -> None:
    """同输入独立运行两次，只有两次均完整评分且排名一致才通过。

    - --manifest-root: 本轮 NxU/nxu 清单根；门禁选择 nxu 的 9nal occurrence 0。
    - --data-root: AdaLigand 原始资料根。
    - --output-root: 仅用于门禁的独立目录，不是正式结果根。
    - --tutorial-root: 冻结的官方 XML 与排名脚本位置。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tutorial-root", required=True, type=Path)
    args = parser.parse_args()
    task_root = args.manifest_root / "nxu"
    pdb = next(json.loads(line) for line in (task_root / "pdbs.jsonl").read_text(encoding="utf-8").splitlines()
               if json.loads(line)["pdb_id"] == "9nal")
    query = next(json.loads(line) for line in (task_root / "queries.jsonl").read_text(encoding="utf-8").splitlines()
                 if json.loads(line)["pdb_id"] == "9nal" and json.loads(line)["occurrence_id"] == 0)
    results = []
    ranks = []
    for attempt in (1, 2):
        root = args.output_root / f"attempt_{attempt}"
        results.append(run_query(query, pdb, args.data_root, args.data_root, root, args.tutorial_root))
        rank_path = root / "9nal/0/official_rank.txt"
        ranks.append(rank_path.read_text(encoding="utf-8") if rank_path.is_file() else None)
    expected = set(pdb["identity_smiles"])
    if any(result["status"] != "scored" or set(result["scored_smiles"]) != expected for result in results):
        raise RuntimeError(f"固定种子门禁未完整评分: {results}")
    if ranks[0] != ranks[1] or results[0]["predicted_smiles"] != results[1]["predicted_smiles"]:
        raise RuntimeError("两次独立固定种子评分或官方排名不同")
    print(f"固定种子门禁 PASS: 9nal/0, 身份数={len(expected)}", flush=True)


if __name__ == "__main__":
    main()
