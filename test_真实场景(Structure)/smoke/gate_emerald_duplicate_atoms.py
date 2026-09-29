"""验证官方 .params 的重复原子名被记作单个身份的准备失败。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_emerald_query import prepare_identity


def main() -> None:
    """用冻结 9uat 身份复核 Rosetta 已报告的 H2 名称冲突。

    - --manifest-root: 含 NxU/pdbs.jsonl 的正式冻结清单根。
    - --output-root: 独立的临时门禁目录，不进入正式评分分母。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    pdb_path = args.manifest_root / "NxU/pdbs.jsonl"
    pdb = next(json.loads(line) for line in pdb_path.read_text(encoding="utf-8").splitlines()
               if json.loads(line)["pdb_id"] == "9uat")
    smiles = pdb["identity_smiles"][2]
    ligand_dir = args.output_root / "ligands"
    ligand_dir.mkdir(parents=True, exist_ok=False)
    try:
        prepare_identity(smiles, "L02", ligand_dir)
    except ValueError as error:
        if str(error) != "官方 .params 的原子名重复":
            raise
    else:
        raise RuntimeError("重复原子名的 .params 被误认为可评分身份")
    print("重复原子名门禁 PASS：不改官方 .params，身份按准备失败保留。", flush=True)


if __name__ == "__main__":
    main()
