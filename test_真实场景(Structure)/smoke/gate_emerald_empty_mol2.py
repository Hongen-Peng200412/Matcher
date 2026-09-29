"""确认官方配体准备产生空 MOL2 时，该身份作为准备失败而非整点异常。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_emerald_query import prepare_identity


def main() -> None:
    """用冻结 9vap 全配体身份中的铁硫 SMILES 复核空 MOL2 边界。

    - --manifest-root: 含 NxU/pdbs.jsonl 的正式冻结清单根。
    - --output-root: 独立的临时门禁目录，不进入正式评分分母。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    pdb_path = args.manifest_root / "NxU/pdbs.jsonl"
    pdb = next(json.loads(line) for line in pdb_path.read_text(encoding="utf-8").splitlines()
               if json.loads(line)["pdb_id"] == "9vap")
    iron_sulfur = next(smiles for smiles in pdb["identity_smiles"] if "[Fe]" in smiles)
    ligand_dir = args.output_root / "ligands"
    ligand_dir.mkdir(parents=True, exist_ok=False)
    try:
        prepare_identity(iron_sulfur, "L02", ligand_dir)
    except ValueError as error:
        if str(error) != "MOL2 回读未产生 SMILES":
            raise
    else:
        raise RuntimeError("空 MOL2 被误认为可供 Rosetta 评分的身份")
    print("空 MOL2 门禁 PASS：原身份保留在清单，准备失败不成为整点异常。", flush=True)


if __name__ == "__main__":
    main()
