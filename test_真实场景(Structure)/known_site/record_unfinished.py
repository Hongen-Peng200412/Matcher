"""为已结束的 CPU 数组中没有正式结果的查询点补写失败记录。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    """只补齐冻结清单中缺少 result.json 的查询点，不覆盖已有评分。

    - --manifest-root: NxU 与 nxu 冻结清单的共同根。
    - --task-name: 本次已经结束的身份任务名称。
    - --output-root: EMERALD-ID 的正式逐查询结果根。

    调用前须从 Slurm 和 allocation 日志确认该任务的所有工作进程均已结束。
    保留查询点原目录中的 Rosetta 部分输出，不从未完成的评分文件拼排名。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--task-name", required=True, choices=("NxU", "nxu"))
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    queries_path = args.manifest_root / args.task_name / "queries.jsonl"
    # 一个查询点的 PDB 和 occurrence ID 共同确定唯一结果路径。
    queries = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines()]
    written = 0
    for query in queries:
        output = args.output_root / args.task_name / query["pdb_id"] / str(query["occurrence_id"])
        result_path = output / "result.json"
        if result_path.exists():
            continue
        output.mkdir(parents=True, exist_ok=True)
        result = {
            "pdb_id": query["pdb_id"],
            "occurrence_id": query["occurrence_id"],
            "status": "allocation_ended_before_result",
            "predicted_smiles": None,
            "scored_smiles": [],
            "failed_smiles": {},
            "rosetta_exit_code": None,
        }
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        written += 1
    print(f"记录未完成查询点：{written}；冻结查询点总数：{len(queries)}。", flush=True)


if __name__ == "__main__":
    main()
