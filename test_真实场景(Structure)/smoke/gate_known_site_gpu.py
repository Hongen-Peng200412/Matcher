"""在实际 GPU 上对已知位点正式前向入口做小样本门禁，不产生正式指标。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    """从冻结清单选取多身份 PDB，调用正式预测入口并检查逐点输出。

    - --manifest-root: 正式或待冻结的完整 NxU/nxu 清单根。
    - --identity-manifest: 可选的全类别 PDB 身份轴；门禁选择含额外类别的 PDB。
    - --source-manifest: 模型已有的 held_out_test_0 身份资产来源清单。
    - --checkpoint: 与门禁硬件及模型相对应的训练 BEST。
    - --model-name: density_only-0 或 density_only-106。
    - --data-root: AdaLigand 原始资料根。
    - --receptor-root: 106 通道变体读取的受体与模拟密度根；省略时使用原始资料。
    - --task-names: 本次门禁选取的冻结任务；CryoAtom2 小分子只取 nxu。
    - --scratch-root: 当前门禁输出的临时父目录。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--identity-manifest", type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--receptor-root", type=Path)
    parser.add_argument("--task-names", nargs="+", choices=("NxU", "nxu"), default=("NxU", "nxu"))
    parser.add_argument("--scratch-root", required=True, type=Path)
    args = parser.parse_args()

    scratch = Path(tempfile.mkdtemp(prefix=f"gate_{args.model_name}_", dir=args.scratch_root))
    selected = {}
    for task_name in args.task_names:
        source = args.manifest_root / task_name
        identity_path = args.identity_manifest if args.identity_manifest else source / "pdbs.jsonl"
        pdbs = [json.loads(line) for line in identity_path.read_text(encoding="utf-8").splitlines()]
        if args.identity_manifest:
            pdb = next(record for record in pdbs if
                       len(record["identity_smiles"]) > len(record["small_molecule_identity_smiles"]))
        else:
            pdb = next(record for record in pdbs if len(record["identity_smiles"]) > 1)
        queries = [query for query in (
            json.loads(line) for line in (source / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        ) if query["pdb_id"] == pdb["pdb_id"]][:2]
        selected[task_name] = (pdb, queries)
        target = scratch / "manifest" / task_name
        target.mkdir(parents=True)
        (target / "pdbs.jsonl").write_text(json.dumps(pdb, ensure_ascii=False) + "\n", encoding="utf-8")
        (target / "queries.jsonl").write_text(
            "".join(json.dumps(query, ensure_ascii=False) + "\n" for query in queries), encoding="utf-8"
        )
    entry = Path(__file__).resolve().parents[1] / "known_site" / "predict_matcher.py"
    command = [sys.executable, str(entry), "--manifest-root", str(scratch / "manifest"),
               "--source-manifest", str(args.source_manifest), "--checkpoint", str(args.checkpoint),
               "--model-name", args.model_name, "--output-root", str(scratch / "predictions"),
               "--data-root", str(args.data_root), "--task-names", *args.task_names,
               "--batch-size", "2"]
    if args.receptor_root is not None:
        command += ["--receptor-root", str(args.receptor_root)]
    if args.identity_manifest is not None:
        command += ["--identity-manifest", str(scratch / "manifest" / args.task_names[0] / "pdbs.jsonl")]
    subprocess.run(command, check=True)
    for task_name, (pdb, queries) in selected.items():
        for query in queries:
            path = scratch / "predictions" / args.model_name / task_name / pdb["pdb_id"] / f"{query['occurrence_id']}.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["status"] != "scored" or result["predicted_smiles"] not in result["scored_smiles"]:
                raise RuntimeError(f"门禁没有得到有效身份前向: {path}")
            if args.identity_manifest is not None:
                scores = result["identity_logits"]
                smiles = result["scored_smiles"]
                small = set(pdb["small_molecule_identity_smiles"])
                if not any(value not in small for value in smiles):
                    raise RuntimeError(f"门禁没有实际编码扩展的非小分子身份: {path}")
                if result["predicted_smiles"] != smiles[max(range(len(smiles)), key=lambda index: scores[index])]:
                    raise RuntimeError(f"门禁的不掩码读出与完整身份 logits 不一致: {path}")
                small_indices = [index for index, value in enumerate(smiles) if value in small]
                expected = smiles[max(small_indices, key=lambda index: scores[index])] if small_indices else None
                if result["predicted_small_molecule_smiles"] != expected:
                    raise RuntimeError(f"门禁的掩码读出与完整身份 logits 不一致: {path}")
    print(f"GPU 门禁 PASS: {args.model_name}, scratch={scratch}", flush=True)


if __name__ == "__main__":
    main()
