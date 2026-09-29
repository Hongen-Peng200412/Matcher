"""一次性记录已知位点比较读取的冻结文件字节摘要；不参与模型或评分。"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    """读取冻结清单中的 PDB 与语言资产索引。

    - path: 逐行 JSON 文件。
    - return: 原文件顺序的记录列表。
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def digest(path: Path) -> dict:
    """逐块散列一个只读输入文件，保存相对分析所需的文件状态。

    - path: 来源文件绝对路径。
    - return: 绝对路径、字节数、修改时间纳秒和 SHA-256。
    """
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sha256": checksum.hexdigest()}


def main() -> None:
    """把清单、模型身份资产、图、密度、受体与原始 EMERALD 输入逐文件快照。

    - --manifest-root: 正式 NxU/nxu 冻结清单。
    - --source-manifest: 既有 held_out_test_0 身份资产清单。
    - --data-root: AdaLigand Ori_Data。
    - --checkpoint: 两个既有训练 BEST，各提供一次该选项。
    - --tutorial-root: 官方 XML 与排名脚本的只读来源目录。
    - --output-root: 一次性追溯资料目录，不是模型科学产物。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, type=Path)
    parser.add_argument("--tutorial-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    files = {args.source_manifest, args.data_root / "raw/pair_list.jsonl",
             args.tutorial_root / "ligand_screen.xml",
             args.tutorial_root / "rank_ligand_identity.py", *args.checkpoint}
    pdb_records = {}
    smiles_by_pdb: dict[str, set[str]] = {}
    for task_name in ("NxU", "nxu"):
        task_root = args.manifest_root / task_name
        files.update(task_root / name for name in ("pdbs.jsonl", "queries.jsonl", "excluded_pdbs.jsonl"))
        for pdb in read_jsonl(task_root / "pdbs.jsonl"):
            pdb_records[pdb["pdb_id"]] = pdb
            smiles_by_pdb.setdefault(pdb["pdb_id"], set()).update(pdb["identity_smiles"])
    files.add(args.manifest_root / "manifest.json")
    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    files.add(Path(source["graph_asset_path"]))
    identities = read_jsonl(args.source_manifest.parent / source["splits"]["held_out_test_0"]["identities"])
    files.add(args.source_manifest.parent / source["splits"]["held_out_test_0"]["identities"])
    for identity in identities:
        if identity["smiles"] in smiles_by_pdb.get(identity["pdb_id"], set()):
            files.add(args.data_root / identity["language_relative_path"])
    for pdb_id, pdb in pdb_records.items():
        files.update((
            args.data_root / "parse" / pdb_id / "occurrences.jsonl",
            args.data_root / "parse" / pdb_id / "ligand_coords.npz",
            args.data_root / "parse" / pdb_id / "receptor_tokens.npz",
            args.data_root / "density" / pdb_id / "exp.npy",
            args.data_root / "density" / pdb_id / "exp.npz",
            args.data_root / "density" / pdb_id / "sim.npy",
            args.data_root / "density" / pdb_id / "sim.npz",
            args.data_root / "stage1_preparation_box_pool_2/ligand_language_models/prepared" / pdb_id / "prepared_smiles.jsonl",
            args.data_root / "raw/rcsb_mmcif" / f"{pdb_id}.cif",
            args.data_root / "raw/emdb_maps" / f"emd_{pdb['emdb_id'].split('-')[1]}.map.gz",
        ))
    ordered = sorted(files)
    with ThreadPoolExecutor(max_workers=8) as executor:
        records = list(executor.map(digest, ordered))
    args.output_root.mkdir(parents=True, exist_ok=False)
    with (args.output_root / "input_files.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"冻结输入文件 {len(records)} 个；追溯根 {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
