"""为一个真实配体质心准备 EMERALD-ID 小规模试运行输入.

主入口 main 读取 AdaLigand 的 occurrence 与顶层 SMILES, 从同一 PDB 生成候选配体的
MOL2/params 和只含受体 ATOM 记录的 PDB. 输出只供 CPU smoke; 不改写来源资料.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
from pathlib import Path

import gemmi
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


ROSETTA_PARAMS = Path(
    "/home/penghongen/software/rosetta/source_build/rosetta.source.release-430/main/"
    "source/scripts/python/public/generic_potential/mol2genparams.py"
)
OBABEL = Path("/home/penghongen/anaconda3/envs/pxm_phase1/bin/obabel")


def main() -> None:
    """读取一个 PDB 的给定点和闭集身份, 生成化学门禁通过后的教程式输入.

    输入参数:
        - data_root: AdaLigand Ori_Data 根, 内有 parse、raw、prepared 三组来源.
        - pdb_id: 小写结构编号, 如 9nal.
        - occurrence_id: 当前 PDB 的 candidate_id, 只决定查询点和隔离保存的评价答案.
        - task: small 时仅取小分子 SMILES, all 时取全部目标 SMILES.
        - output_root: 当前查询点专属目录, 不得与另一查询点共用 Rosetta scorefile.

    输出文件:
        - input.pdb: 受体 ATOM 原子和位于查询质心的最后一条 C3/LG2 虚拟 HETATM.
        - map.map: 与当前 PDB 配对的原始 EMDB map.gz 无损解压, 未重采样.
        - ligands/Lxx.sdf、Lxx.mol2、Lxx.params: 每个不同顶层 SMILES 一套计算输入; xx 是当前 PDB 的确定性升序编号.
        - identity_map.json: 候选 Rosetta 三字母名称到原始精确 SMILES 的映射, 以及当前图的全局分辨率.
        - truth_for_gate.json: 只供事后评价的 occurrence 编号、真实 SMILES 和查询 XYZ 质心; 对接进程不得读取.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("pdb_id")
    parser.add_argument("occurrence_id", type=int)
    parser.add_argument("task", choices=("all", "small"))
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    root = args.data_root
    pdb_id = args.pdb_id
    output = args.output_root
    ligand_dir = output / "ligands"
    ligand_dir.mkdir(parents=True, exist_ok=True)

    # candidate_id 指向同一 PDB 的 occurrence, 不指向候选 SMILES 列表的下标.
    occurrences = [
        json.loads(line)
        for line in (root / "parse" / pdb_id / "occurrences.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    prepared = {
        record["candidate_id"]: record["smiles"]
        for record in map(
            json.loads,
            (root / "stage1_preparation_box_pool_2/ligand_language_models/prepared" / pdb_id / "prepared_smiles.jsonl")
            .read_text(encoding="utf-8")
            .splitlines(),
        )
    }
    with np.load(root / "parse" / pdb_id / "ligand_coords.npz", allow_pickle=False) as coordinates:
        atom_positions = coordinates[f"coords_{args.occurrence_id}"]
        present = coordinates[f"present_{args.occurrence_id}"]
        # float32 (3,), 沉积重原子 XYZ 均值, 单位 Å; 只作为给定查询位置.
        center = atom_positions[present].mean(axis=0, dtype=np.float64).astype(np.float32)

    # 一个 PDB 的 exact smiles 字符串按字典序去重; 不用 object_key 或准备后的分子名称重定义身份.
    identities = sorted(
        {
            prepared[occurrence["candidate_id"]]
            for occurrence in occurrences
            if args.task == "all" or occurrence["type_tag"] == "small_molecule"
        }
    )
    target = prepared[args.occurrence_id]

    # 与项目 pair_list 配对的原始 EMDB 图和实验全局分辨率; 不使用约 1 Å 的 Matcher 重采样图.
    pair = next(
        record
        for record in map(json.loads, (root / "raw/pair_list.jsonl").read_text(encoding="utf-8").splitlines())
        if record["pdb_id"] == pdb_id
    )
    map_source = root / "raw/emdb_maps" / f"emd_{pair['emdb_id'].split('-')[1]}.map.gz"
    with gzip.open(map_source, "rb") as source, (output / "map.map").open("wb") as target_map:
        shutil.copyfileobj(source, target_map)

    # Gemmi 把 mmCIF 转成 PDB, 只保留第一个 model 的 ATOM 重原子, 避免把 HETATM 目标配体当受体.
    structure = gemmi.read_structure(str(root / "raw/rcsb_mmcif" / f"{pdb_id}.cif"))
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    structure.remove_ligands_and_waters()
    receptor_lines = [line for line in structure.make_pdb_string().splitlines() if line.startswith("ATOM  ")]
    x, y, z = (float(value) for value in center)
    dummy = f"HETATM99999  C3  LG2 X   1    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          C"
    (output / "input.pdb").write_text("\n".join((*receptor_lines, dummy, "END")) + "\n", encoding="ascii")

    names_to_smiles: dict[str, str] = {}
    for index, smiles in enumerate(identities):
        ligand_name = f"L{index:02d}"
        # 显式氢只用于三维准备; 比较身份时先移除氢, 保留全部重原子、键和形式电荷.
        parsed = Chem.MolFromSmiles(smiles)
        if parsed is None:
            raise ValueError(f"{pdb_id}: 无法解析候选 {ligand_name} 的原始 SMILES")
        molecule = Chem.AddHs(parsed)
        embedded = AllChem.EmbedMolecule(molecule, randomSeed=3409)
        if embedded != 0:
            raise ValueError(f"{pdb_id}: 候选 {ligand_name} 无法从 SMILES 生成初始三维构象")
        AllChem.MMFFOptimizeMolecule(molecule)
        sdf_path = ligand_dir / f"{ligand_name}.sdf"
        mol2_path = ligand_dir / f"{ligand_name}.mol2"
        Chem.MolToMolFile(molecule, str(sdf_path))
        subprocess.run(
            [str(OBABEL), str(sdf_path), "-O", str(mol2_path), "--partialcharge", "mmff94"],
            check=True,
        )
        # Open Babel 回读它写出的 MOL2, 再由 RDKit 统一比较非立体重原子图; RDKit 的 MOL2 芳香环读取器不能解析本次 9nal 第二个候选.
        roundtrip_smiles = subprocess.check_output(
            [str(OBABEL), "-imol2", str(mol2_path), "-osmi"], text=True
        ).split()[0]
        roundtrip = Chem.MolFromSmiles(roundtrip_smiles)
        if roundtrip is None or Chem.MolToSmiles(Chem.RemoveHs(roundtrip), isomericSmiles=False) != Chem.MolToSmiles(Chem.RemoveHs(parsed), isomericSmiles=False):
            raise ValueError(f"{pdb_id}: 候选 {ligand_name} 在 MOL2 转换后改变了重原子化学图")
        subprocess.run(
            [
                "/home/penghongen/anaconda3/envs/pxm_phase1/bin/python",
                str(ROSETTA_PARAMS),
                "--amide_chi",
                "-s",
                str(mol2_path),
                "--nm",
                ligand_name,
                "--puckering_chi",
                "--outdir",
                str(ligand_dir),
            ],
            check=True,
        )
        names_to_smiles[ligand_name] = smiles

    (output / "identity_map.json").write_text(
        json.dumps({"pdb_id": pdb_id, "task": args.task, "resolution_angstrom": pair["resolution"], "names_to_smiles": names_to_smiles}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "truth_for_gate.json").write_text(
        json.dumps({"occurrence_id": args.occurrence_id, "target_smiles": target, "query_center_xyz_angstrom": center.tolist()}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
