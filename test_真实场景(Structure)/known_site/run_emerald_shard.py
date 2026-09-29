"""把一项冻结身份任务的查询点分给 Slurm 数组中的单核 Rosetta 进程。"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from run_emerald_query import run_query


def read_jsonl(path: Path) -> list[dict]:
    """读取一项冻结清单，保持查询点原有顺序。

    - path: pdbs.jsonl 或 queries.jsonl 的路径。
    - return: JSON 对象列表。
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    """每个数组任务启动固定数量单核子进程，逐点执行完整候选集合。

    - --manifest-root: NxU 与 nxu 两项清单的共同根。
    - --task-name: 本次只运行 NxU 或 nxu；两项不在同一批任务中混跑。
    - --data-root: 只读 AdaLigand Ori_Data。
    - --receptor-root: 每个 PDB 的受体 mmCIF 根；省略时使用原始 --data-root。
    - --output-root: 本方法正式逐查询输出根。
    - --tutorial-root: 官方 XML 与官方排名脚本的冻结目录。
    - --shard-index: 当前 Slurm 数组下标，从零开始。
    - --shard-count: 数组元素数；与 CPU 配额共同限制并发总量。
    - --workers: 当前数组元素可同时运行的单核 Rosetta 进程数。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--task-name", required=True, choices=("NxU", "nxu"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--receptor-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tutorial-root", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--workers", required=True, type=int)
    args = parser.parse_args()
    receptor_root = args.receptor_root if args.receptor_root is not None else args.data_root

    # 大身份集合的点先分配给当前理论工作量最低的 shard，不拆开同一点身份。
    work = []
    task_name = args.task_name
    task_root = args.manifest_root / task_name
    pdbs = {record["pdb_id"]: record for record in read_jsonl(task_root / "pdbs.jsonl")}
    for query in read_jsonl(task_root / "queries.jsonl"):
        pdb = pdbs[query["pdb_id"]]
        work.append((query, pdb))
    work.sort(key=lambda item: (-len(item[1]["identity_smiles"]),
                                item[0]["pdb_id"], item[0]["occurrence_id"]))
    shards: list[list[tuple[dict, dict]]] = [[] for _ in range(args.shard_count)]
    costs = [0] * args.shard_count
    for item in work:
        position = min(range(args.shard_count), key=lambda index: (costs[index], index))
        shards[position].append(item)
        costs[position] += len(item[1]["identity_smiles"])

    selected = shards[args.shard_index]
    print(f"shard={args.shard_index}/{args.shard_count} queries={len(selected)} "
          f"theoretical_scores={costs[args.shard_index]}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for query, pdb in selected:
            point_root = args.output_root / task_name
            result_path = point_root / query["pdb_id"] / str(query["occurrence_id"]) / "result.json"
            if result_path.is_file():
                continue
            future = executor.submit(run_query, query, pdb, args.data_root, receptor_root, point_root,
                                     args.tutorial_root)
            futures[future] = (task_name, query["pdb_id"], query["occurrence_id"])
        for future in as_completed(futures):
            task_name, pdb_id, occurrence_id = futures[future]
            result = future.result()
            print(f"{task_name}/{pdb_id}/{occurrence_id}: {result['status']}", flush=True)


if __name__ == "__main__":
    main()
