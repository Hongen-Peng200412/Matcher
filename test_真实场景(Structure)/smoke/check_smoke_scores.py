"""检查一个查询点的 EMERALD-ID scorefile, 再调用官方身份排名脚本.

主入口 main 只消费一个查询点专属目录中的 .sc 文件; 不在此实现替代性排名公式.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path


def main() -> None:
    """核对每个成功身份仅有一条完整评分, 并保存官方脚本的原始排名.

    输入参数:
        - score_dir: 当前唯一查询点的 Rosetta .sc 所在目录, 不得混入其他点.
        - rank_script: 作者提供的 rank_ligand_identity.py, 读取 zscore 与 lig_dens.
        - output_dir: 当前查询点的门禁输出目录.
        - mapping: 项目查询的 identity_map.json, 内有 names_to_smiles; 教程不用.
        - expected_names: 教程的原生配体名称, 例如 ADP,ATP,CLR,Y01.

    输出文件:
        - official_rank.txt: 官方脚本按本查询全部有效评分生成的原始表格.
        - score_gate.json: 每个成功名称的原始 zscore、lig_dens 和官方概率, 及缺分名称.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score_dir", type=Path)
    parser.add_argument("rank_script", type=Path)
    parser.add_argument("output_dir", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mapping", type=Path)
    source.add_argument("--expected-names")
    args = parser.parse_args()

    if args.mapping is None:
        names_to_smiles = {name: None for name in args.expected_names.split(",")}
    else:
        names_to_smiles = json.loads(args.mapping.read_text(encoding="utf-8"))["names_to_smiles"]

    scores: dict[str, dict[str, float]] = {}
    scorefiles = sorted(args.score_dir.glob("*.sc"))
    for scorefile in scorefiles:
        fields = None
        for line in scorefile.read_text(encoding="utf-8").splitlines():
            pieces = line.split()
            if not pieces or pieces[0] != "SCORE:":
                continue
            if "ligandname" in pieces and "zscore" in pieces and "lig_dens" in pieces:
                fields = pieces[1:]
                continue
            if fields is None:
                continue
            values = dict(zip(fields, pieces[1:], strict=True))
            name = values["ligandname"]
            if name not in names_to_smiles or name in scores:
                raise ValueError(f"{scorefile}: 配体名称未映射或出现重复有效评分: {name}")
            zscore = float(values["zscore"])
            ligand_density = float(values["lig_dens"])
            if not math.isfinite(zscore) or not math.isfinite(ligand_density):
                raise ValueError(f"{scorefile}: {name} 的 zscore/lig_dens 非有限数")
            scores[name] = {"zscore": zscore, "lig_dens": ligand_density}

    if not scores:
        raise ValueError(f"{args.score_dir}: 未得到任何有效的官方身份评分")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    official_rank = args.output_dir / "official_rank.txt"
    subprocess.run(
        [
            "/home/penghongen/anaconda3/envs/pxm_phase1/bin/python",
            str(args.rank_script),
            "-d",
            str(args.score_dir),
            "-o",
            str(official_rank),
        ],
        check=True,
    )
    ranked = [line.split() for line in official_rank.read_text(encoding="utf-8").splitlines()[1:]]
    ranked_names = [fields[0] for fields in ranked]
    if set(ranked_names) != set(scores) or len(ranked_names) != len(scores):
        raise ValueError("官方排名输出与唯一有效 scorefile 身份不一致")

    for fields in ranked:
        scores[fields[0]]["official_probability"] = float(fields[-1])
    report = {
        "expected_names": list(names_to_smiles),
        "successful_names": ranked_names,
        "missing_names": sorted(set(names_to_smiles) - set(scores)),
        "scorefiles": [str(path) for path in scorefiles],
        "scores": {name: {"original_smiles": names_to_smiles[name], **score} for name, score in scores.items()},
    }
    (args.output_dir / "score_gate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
