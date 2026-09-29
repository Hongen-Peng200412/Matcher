"""一次性散列固定 nxu 全类别身份前向读取的文件；只生成追溯清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(path: Path) -> dict:
    """记录一个实际输入文件的绝对路径、字节数与 SHA-256。"""
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": checksum.hexdigest()}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest-root", required=True, type=Path)
parser.add_argument("--all-identity-manifest", required=True, type=Path)
parser.add_argument("--source-manifest", required=True, type=Path)
parser.add_argument("--data-root", required=True, type=Path)
parser.add_argument("--cryo-root", required=True, type=Path)
parser.add_argument("--checkpoint", required=True, type=Path, action="append")
parser.add_argument("--output-root", required=True, type=Path)
args = parser.parse_args()

source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
identity_index = args.source_manifest.parent / source["splits"]["held_out_test_0"]["identities"]
pdbs = [json.loads(line) for line in args.all_identity_manifest.read_text(encoding="utf-8").splitlines()]
needed = {(pdb["pdb_id"], smiles) for pdb in pdbs for smiles in pdb["identity_smiles"]}
files = {
    args.manifest_root / "nxu/pdbs.jsonl",
    args.manifest_root / "nxu/queries.jsonl",
    args.all_identity_manifest,
    args.source_manifest,
    identity_index,
    Path(source["graph_asset_path"]),
    *args.checkpoint,
}
for identity in (json.loads(line) for line in identity_index.read_text(encoding="utf-8").splitlines()):
    if (identity["pdb_id"], identity["smiles"]) in needed:
        files.add(args.data_root / identity["language_relative_path"])
for pdb in pdbs:
    pdb_id = pdb["pdb_id"]
    files.update((
        args.data_root / "parse" / pdb_id / "occurrences.jsonl",
        args.data_root / "stage1_preparation_box_pool_2/ligand_language_models/prepared"
        / pdb_id / "prepared_smiles.jsonl",
        args.data_root / "density" / pdb_id / "exp.npy",
        args.data_root / "density" / pdb_id / "exp.npz",
        args.data_root / "density" / pdb_id / "sim.npy",
        args.data_root / "parse" / pdb_id / "receptor_tokens.npz",
        args.cryo_root / "density" / pdb_id / "sim.npy",
        args.cryo_root / "parse" / pdb_id / "receptor_tokens.npz",
    ))
with ThreadPoolExecutor(max_workers=8) as executor:
    records = list(executor.map(digest, sorted(files)))
args.output_root.mkdir(parents=True, exist_ok=False)
with (args.output_root / "input_files.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
print(f"输入快照文件数={len(records)}", flush=True)
