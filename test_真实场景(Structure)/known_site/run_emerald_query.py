"""为一个冻结的 occurrence 查询点运行一次官方 EMERALD-ID 身份筛选。

当前 PDB 的全部候选精确 SMILES 进入同一次 multiple_ligands 调用；一个查询点
只有自己的输入、Rosetta 输出和官方排名。化学准备或评分失败也保存逐点结果。
"""

from __future__ import annotations

import gzip
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem


PYTHON = "/home/penghongen/anaconda3/envs/pxm_phase1/bin/python"
OBABEL = "/home/penghongen/anaconda3/envs/pxm_phase1/bin/obabel"
ROSETTA_ROOT = Path("/home/penghongen/software/rosetta/source_build/rosetta.source.release-430/main")
ROSETTA = "/home/penghongen/software/rosetta/bin/rosetta_scripts_430"
GEN_PARAMS = ROSETTA_ROOT / "source/scripts/python/public/generic_potential/mol2genparams.py"


def prepare_identity(smiles: str, name: str, ligand_dir: Path) -> Path:
    """从原始 SMILES 生成一套官方 Rosetta 配体输入。

    - smiles: 精确顶层 PreparedLigand.smiles，不按 pH 重写。
    - name: 当前 PDB 的三字符 Rosetta 配体名，例如 L00。
    - ligand_dir: 当前查询点的独立配体文件夹。
    - return: 官方 mol2genparams.py 生成的 .params 路径。

    RDKit 加氢与 3D 嵌入只用于对接初态。Open Babel 写 MOL2 后，把其回读
    的非立体重原子化学图与原始 SMILES 比较，防止键级、形式电荷或组分变化。
    """
    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise ValueError("原始 SMILES 无法通过 RDKit 解析")
    molecule = Chem.AddHs(parsed)
    if AllChem.EmbedMolecule(molecule, randomSeed=3409) != 0:
        raise ValueError("SMILES 无法生成初始三维构象")
    AllChem.MMFFOptimizeMolecule(molecule)
    sdf = ligand_dir / f"{name}.sdf"
    mol2 = ligand_dir / f"{name}.mol2"
    Chem.MolToMolFile(molecule, str(sdf))
    subprocess.run([OBABEL, str(sdf), "-O", str(mol2), "--partialcharge", "mmff94"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    roundtrip_words = subprocess.check_output([OBABEL, "-imol2", str(mol2), "-osmi"], text=True).split()
    if not roundtrip_words:
        raise ValueError("MOL2 回读未产生 SMILES")
    roundtrip = Chem.MolFromSmiles(roundtrip_words[0])
    original_graph = Chem.MolToSmiles(Chem.RemoveHs(parsed), isomericSmiles=False)
    if roundtrip is None or Chem.MolToSmiles(Chem.RemoveHs(roundtrip), isomericSmiles=False) != original_graph:
        raise ValueError("MOL2 回读后的重原子化学图与原始 SMILES 不同")
    subprocess.run([PYTHON, str(GEN_PARAMS), "--amide_chi", "-s", str(mol2),
                    "--nm", name, "--puckering_chi", "--outdir", str(ligand_dir)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    params = ligand_dir / f"{name}.params"
    # 官方生成器对部分单原子离子会以零退出码结束，却不写出可供 Rosetta 读取的参数。
    if not params.is_file():
        raise ValueError("官方参数生成器未写出 .params")
    # 官方生成器也可能给不同原子写出相同名称；Rosetta 会拒绝整个参数集合。
    atom_names = [line.split()[1] for line in params.read_text(encoding="utf-8").splitlines()
                  if line.startswith("ATOM ")]
    if len(atom_names) != len(set(atom_names)):
        raise ValueError("官方 .params 的原子名重复")
    return params


def rank_scores(score_dir: Path, names_to_smiles: dict[str, str], output_dir: Path,
                rank_script: Path) -> tuple[str | None, list[str], list[str]]:
    """核对逐身份唯一有效评分，再调用官方 rank_ligand_identity.py。

    - score_dir: 当前查询点唯一的 .sc 文件夹。
    - names_to_smiles: Rosetta 名称到当前任务精确 SMILES 的映射。
    - output_dir: 当前查询点根目录，保存官方原始排名表。
    - rank_script: 官方脚本的冻结路径。
    - return[0]: 官方首位对应的精确 SMILES；没有有效评分则为 None。
    - return[1]: 成功获得有效 zscore 和 lig_dens 的 SMILES，按官方排名顺序。
    - return[2]: 有 .sc 但未能进入唯一有效评分的名称。
    """
    scored: dict[str, tuple[float, float]] = {}
    rejected: list[str] = []
    for path in sorted(score_dir.glob("*.sc")):
        columns = None
        for line in path.read_text(encoding="utf-8").splitlines():
            pieces = line.split()
            if not pieces or pieces[0] != "SCORE:":
                continue
            if "ligandname" in pieces and "zscore" in pieces and "lig_dens" in pieces:
                columns = pieces[1:]
                continue
            if columns is None:
                continue
            fields = dict(zip(columns, pieces[1:]))
            name = fields["ligandname"]
            values = (float(fields["zscore"]), float(fields["lig_dens"]))
            if name not in names_to_smiles or name in scored or not all(map(math.isfinite, values)):
                rejected.append(name)
                continue
            scored[name] = values
    if not scored or rejected:
        return None, [], rejected
    rank_path = output_dir / "official_rank.txt"
    subprocess.run([PYTHON, str(rank_script), "-d", str(score_dir), "-o", str(rank_path)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    ranked_names = [line.split()[0] for line in rank_path.read_text(encoding="utf-8").splitlines()[1:]]
    if len(ranked_names) != len(scored) or set(ranked_names) != set(scored):
        return None, [], ranked_names
    return names_to_smiles[ranked_names[0]], [names_to_smiles[name] for name in ranked_names], []


def run_query(query: dict, pdb: dict, data_root: Path, receptor_root: Path, output_root: Path,
              tutorial_root: Path) -> dict:
    """准备并评分一个完整查询点，结果或失败写入专属 result.json。

    - query: manifest 查询点记录，含 PDB、occurrence 和世界 XYZ 质心。
    - pdb: 同任务该 PDB 的完整候选 SMILES、EMDB 编号及全局分辨率。
    - data_root: AdaLigand Ori_Data 根。
    - receptor_root: 受体 mmCIF 所在的 Ori_Data 根；CryoAtom2 条件取最终预测受体。
    - output_root: 当前方法与任务的正式产物根。
    - tutorial_root: 官方 ligand_screen.xml 和 rank_ligand_identity.py 目录。
    - return: 当前点的预测、逐身份成功覆盖及明确失败原因。
    """
    tutorial_root = tutorial_root.resolve()
    pdb_id = query["pdb_id"]
    occurrence_id = query["occurrence_id"]
    output = output_root / pdb_id / str(occurrence_id)
    output.mkdir(parents=True, exist_ok=False)
    result = {"pdb_id": pdb_id, "occurrence_id": occurrence_id,
              "status": "not_scored", "predicted_smiles": None,
              "scored_smiles": [], "failed_smiles": {}, "rosetta_exit_code": None}
    try:
        ligand_dir = output / "ligands"
        ligand_dir.mkdir()
        names_to_smiles = {}
        params = []
        for index, smiles in enumerate(pdb["identity_smiles"]):
            name = f"L{index:02d}"
            try:
                params.append(prepare_identity(smiles, name, ligand_dir))
                names_to_smiles[name] = smiles
            except (ValueError, subprocess.CalledProcessError) as error:
                result["failed_smiles"][smiles] = f"prepare: {error}"
        (output / "identity_map.json").write_text(
            json.dumps(names_to_smiles, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if not params:
            result["status"] = "no_prepared_identity"
            return result

        # 输入 PDB 只含受体 ATOM 和质心处的虚拟查询原子，不含目标配体坐标。
        structure = gemmi.read_structure(str(receptor_root / "raw/rcsb_mmcif" / f"{pdb_id}.cif"))
        structure.remove_alternative_conformations()
        structure.remove_hydrogens()
        # CryoAtom2 最终 CIF 没有 entity_type; 两种受体都只把 ATOM 行交给 Rosetta。
        if receptor_root == data_root:
            structure.remove_ligands_and_waters()
        else:
            # PDB 只能写入至多两字符的链名；末字符 X 留给虚拟查询原子所在链。
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwxyz0123456789"
            for model in structure:
                if any(len(chain.name) > 2 for chain in model):
                    for index, chain in enumerate(model):
                        chain.name = alphabet[index // len(alphabet)] + alphabet[index % len(alphabet)]
        receptor_lines = [line for line in structure.make_pdb_string().splitlines() if line.startswith("ATOM  ")]
        x, y, z = query["center_xyz_angstrom"]
        dummy = f"HETATM99999  C3  LG2 X   1    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          C"
        input_pdb = output / "input.pdb"
        input_pdb.write_text("\n".join((*receptor_lines, dummy, "END")) + "\n", encoding="ascii")
        map_source = data_root / "raw/emdb_maps" / f"emd_{pdb['emdb_id'].split('-')[1]}.map.gz"
        map_path = output / "map.map"
        with gzip.open(map_source, "rb") as source, map_path.open("wb") as target:
            shutil.copyfileobj(source, target)

        score_dir = output / "outputs"
        score_dir.mkdir()
        resolution = str(pdb["resolution_angstrom"])
        ligands = ",".join(path.stem for path in params)
        command = [
            ROSETTA, "-in:file:extra_res_fa", *(str(path) for path in params),
            "-in:file:override_database_params", "-out:path:all", str(score_dir),
            "-out:prefix", "emerald-id_", "-gen_potential", "-database", str(ROSETTA_ROOT / "database"),
            "-s", str(input_pdb), "-multi_cool_annealer", "10", "-parser:protocol",
            str(tutorial_root / "ligand_screen.xml"), "-no_autogen_cart_improper", "-atom_mask", "2",
            "-parser:script_vars", f"map={map_path}", "-parser:script_vars", f"ligands={ligands}",
            "-parser:script_vars", f"local_res={resolution}", "-sliding_window", "1",
            "-edensity::score_sliding_window_context", "-edensity::mapreso", resolution,
            "-edensity::grid_spacing", "1.0", "-constant_seed", "-jran", "3409",
        ]
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        with (output / "rosetta.stdout.log").open("w") as stdout, (output / "rosetta.stderr.log").open("w") as stderr:
            completed = subprocess.run(command, cwd=output, env=environment, stdout=stdout, stderr=stderr,
                                       check=False)
        result["rosetta_exit_code"] = completed.returncode
        try:
            predicted, scored, rejected = rank_scores(score_dir, names_to_smiles, output,
                                                       tutorial_root / "rank_ligand_identity.py")
            result["predicted_smiles"] = predicted
            result["scored_smiles"] = scored
            result["status"] = "scored" if predicted is not None else "no_valid_score"
            if rejected:
                result["failure_reason"] = f"重复、非有限或无法映射的评分名称: {rejected}"
        except (ValueError, subprocess.CalledProcessError) as error:
            result["status"] = "ranking_failed"
            result["failure_reason"] = str(error)
    except Exception as error:
        # 单点失败必须留在固定分母；异常只终止当前点，不能吞掉其他点的任务。
        result["status"] = "query_failed"
        result["failure_reason"] = f"{type(error).__name__}: {error}"
    finally:
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    return result
