"""验证官方参数生成器未写出参数时，不会污染同点其他身份的评分。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_emerald_query import run_query


def main() -> None:
    """在冻结的 9nal/1 全配体查询点上执行完整官方评分门禁。

    - --manifest-root: 正式冻结清单；从 NxU 读取完整 PDB 内身份集合。
    - --data-root: AdaLigand 原始受体、密度图与 occurrence 数据根。
    - --output-root: 与正式逐点目录隔离的门禁结果根。
    - --tutorial-root: 官方 XML 与排名脚本的冻结目录。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tutorial-root", required=True, type=Path)
    args = parser.parse_args()
    task_root = args.manifest_root / "NxU"
    pdb = next(json.loads(line) for line in (task_root / "pdbs.jsonl").read_text(encoding="utf-8").splitlines()
               if json.loads(line)["pdb_id"] == "9nal")
    query = next(json.loads(line) for line in (task_root / "queries.jsonl").read_text(encoding="utf-8").splitlines()
                 if json.loads(line)["pdb_id"] == "9nal" and json.loads(line)["occurrence_id"] == 1)
    result = run_query(query, pdb, args.data_root, args.data_root, args.output_root, args.tutorial_root)
    if result["status"] != "scored":
        raise RuntimeError(f"可准备身份未得到有效官方评分: {result}")
    if set(result["failed_smiles"]) != {"[Mg+2]"}:
        raise RuntimeError(f"未正确记录无 .params 的镁离子身份: {result}")
    if set(result["scored_smiles"]) != set(pdb["identity_smiles"]) - {"[Mg+2]"}:
        raise RuntimeError(f"同点其余身份未完整评分: {result}")
    print("缺失 .params 门禁 PASS：镁离子保留为准备失败，其余两个身份完成官方评分。", flush=True)


if __name__ == "__main__":
    main()
