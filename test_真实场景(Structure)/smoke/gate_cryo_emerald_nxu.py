"""在真实 CPU 上检查 CryoAtom2 最终受体进入官方单点对接入口。"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from run_emerald_query import run_query


def main() -> None:
    """用冻结 nxu 清单的指定查询点测试受体替换与逐点结果落盘。

    - --manifest-root: 已冻结的 nxu/pdbs.jsonl 与 nxu/queries.jsonl 所在根。
    - --data-root: 原始 AdaLigand Ori_Data；保留原始实验图和配体身份。
    - --receptor-root: CryoAtom2 test_0 最终受体适配 Ori_Data。
    - --tutorial-root: 官方 EMERALD-ID XML 和排名脚本的冻结目录。
    - --scratch-root: 门禁专用临时目录的父目录；不写正式逐点根。
    - --pdb-id / --occurrence-id: 门禁查询点，默认已验证的 9nal/0。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--receptor-root", required=True, type=Path)
    parser.add_argument("--tutorial-root", required=True, type=Path)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--pdb-id", default="9nal")
    parser.add_argument("--occurrence-id", type=int, default=0)
    args = parser.parse_args()

    task_root = args.manifest_root / "nxu"
    pdbs = [json.loads(line) for line in (task_root / "pdbs.jsonl").read_text(encoding="utf-8").splitlines()]
    queries = [json.loads(line) for line in (task_root / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    pdb = next(record for record in pdbs if record["pdb_id"] == args.pdb_id)
    query = next(record for record in queries if record["pdb_id"] == args.pdb_id
                 and record["occurrence_id"] == args.occurrence_id)
    scratch = Path(tempfile.mkdtemp(prefix="gate_cryo_emerald_nxu_", dir=args.scratch_root))
    result = run_query(query, pdb, args.data_root, args.receptor_root, scratch, args.tutorial_root)
    if result["status"] != "scored":
        raise RuntimeError(f"CryoAtom2 官方单点门禁失败: {scratch / args.pdb_id / str(args.occurrence_id) / 'result.json'}")
    print(f"CryoAtom2 EMERALD-ID 门禁 PASS: status={result['status']} scratch={scratch}", flush=True)


if __name__ == "__main__":
    main()
