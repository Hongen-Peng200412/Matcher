"""从 PreparedLigand 顶层 smiles 构建版本化全局图包和自同构包.

正式命令示例:
python -m ops.matcher.build_smiles_assets --data-root /storage/penghongen/AdaLigand/Ori_Data

构建器只扫描 prepared_smiles.jsonl, 不读取 LigandObject. 无法通过 RDKit 严格解析与
sanitize 的精确字符串统一排除, 并在 data_root/smiles_assets/smiles_assets_v1_report.json
中列明; 合法的多片段字符串保留. 主要入口 build_smiles_assets 同时发布 smiles_graphs_v1.npz 和
smiles_symmetries_v1.npz, 两份资产共用严格递增的精确 SMILES 第一轴.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, rdBase


BOND_TYPES = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.DATIVE,
    Chem.BondType.AROMATIC,
)
RING_SIZES = (3, 4, 5, 6)


def read_exact_smiles(prepared_root: Path) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    """收集所有非空顶层 smiles, 并保留来源位置用于排除报告.

    - prepared_root: ligand_language_models/prepared 目录.
    - return[0]: 按精确字符串递增的唯一 SMILES 列表.
    - return[1]: dict[str, list[dict[str, Any]]], 精确 SMILES 到全部来源记录的映射.
    - return[1][smiles][*].pdb_id: str, prepared_smiles.jsonl 中保存该字符串的小写 PDB 标识.
    - return[1][smiles][*].occurrence_id: int, prepared_smiles.jsonl 的 candidate_id, 在新管线中解释为 occurrence 编号.
    - return[1][smiles][*].path: str, 当前来源 prepared_smiles.jsonl 的完整路径.
    - return[1][smiles][*].line_number: int, 当前来源记录在 JSONL 中从 1 开始的物理行号.
"""

    sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(prepared_root.glob("*/prepared_smiles.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                smiles = record.get("smiles")
                if not isinstance(smiles, str) or not smiles:
                    continue
                sources[smiles].append(
                    {
                        "pdb_id": str(record["pdb_id"]),
                        "occurrence_id": int(record["candidate_id"]),
                        "path": str(path),
                        "line_number": line_number,
                    }
                )
    return sorted(sources), sources


def build_arrays(smiles_values: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    """解析精确 SMILES, 返回图、自同构连接数组和统一排除记录.

    - smiles_values: 唯一且已排序的精确顶层字符串.
    - return[0].schema_version: int64 标量, 固定为 1.
    - return[0].smiles: Unicode 数组, (K,); K 个合法精确字符串按输入顺序保存.
    - return[0].atom_offsets: int64 数组, (K + 1,); 同步切分 element、charge 和 atom_in_ring, 首值为 0, 末值为总原子数.
    - return[0].element: int16 数组, (N_atom_total,); K 张图连接后的重原子序数.
    - return[0].charge: int8 数组, (N_atom_total,); 与 element 逐原子对齐的形式电荷.
    - return[0].atom_in_ring: bool 数组, (N_atom_total, 4); 与 element 逐原子对齐的 3、4、5、6 元环标志.
    - return[0].bond_offsets: int64 数组, (K + 1,); 同步切分 bond_index、bond_type 和 bond_in_ring, 首值为 0, 末值为无向键总数.
    - return[0].bond_index: int32 数组, (2, N_bond_total); 每张图内部的 RDKit 无向键起始与终止原子下标.
    - return[0].bond_type: uint8 数组, (N_bond_total,); 0:5 依次表示 single、double、triple、dative、aromatic.
    - return[0].bond_in_ring: bool 数组, (N_bond_total, 4); 与 bond_index 第二轴逐键对齐的 3、4、5、6 元环标志.
    - return[1].schema_version: int64 标量, 固定为 1.
    - return[1].smiles: Unicode 数组, (K,); 与 return[0].smiles 逐项相同.
    - return[1].atom_count: int32 数组, (K,); 每个精确 SMILES 的 RDKit 重原子数.
    - return[1].matches_shape: int32 数组, (K, 2); 每项依次为自同构数量 N_match 和可变原子列数 N_variable.
    - return[1].matches_offsets: int64 数组, (K + 1,); matches_offsets[i]:matches_offsets[i + 1] 切分第 i 个精确 SMILES 展平后的紧凑自同构矩阵.
    - return[1].matches_iso: int32 数组, (N_match_value_total,); K 个紧凑自同构矩阵按行展平后连接的原子下标.
    - return[2]: list[dict[str, str]], 严格解析失败记录.
    - return[2][*].smiles: str, 未进入两份资产的精确顶层 PreparedLigand.smiles.
    - return[2][*].error: str, RDKit 解析、去氢或 sanitize 异常的类型与消息.

    自同构沿用 PocketXMol 的 useChirality=True、maxMatches=10000 规则. 每个排列只保存
    至少一种自同构会改变的原子列; matches_shape[:, 1] 给出该可变列数. 解析后出现 BOND_TYPES
    未定义的键型时中止整次构建, 不把该错误降级为 return[2] 排除记录.
"""

    # list[str], 长度 K; 只有完整通过解析、去氢和 sanitize 的精确字符串才进入两份同序资产.
    accepted: list[str] = []
    # list[dict[str, str]], 每项保存被排除的精确字符串和失败消息.
    excluded: list[dict[str, Any]] = []
    # 下列原子列表沿 accepted 中各分子的 RDKit 重原子顺序连接; atom_offsets 记录每张图的切片边界.
    elements: list[int] = []
    charges: list[int] = []
    atom_rings: list[list[bool]] = []
    # 下列键列表沿 accepted 中各分子的 RDKit 键顺序连接; bond_offsets 记录每张图的切片边界.
    bond_sources: list[int] = []
    bond_targets: list[int] = []
    bond_types: list[int] = []
    bond_rings: list[list[bool]] = []
    # list[int], 最终长度 K + 1; 首值为 0, 末值为连接原子总数.
    atom_offsets = [0]
    # list[int], 最终长度 K + 1; 首值为 0, 末值为连接无向键总数.
    bond_offsets = [0]
    # 下列自同构列表沿 accepted 顺序保存原子数、紧凑矩阵形状和展平值.
    symmetry_atom_count: list[int] = []
    symmetry_shapes: list[tuple[int, int]] = []
    symmetry_offsets = [0]
    symmetry_values: list[int] = []
    for smiles in smiles_values:
        try:
            molecule = Chem.MolFromSmiles(smiles, sanitize=True)
            if molecule is None:
                raise ValueError("MolFromSmiles 返回 None")
            molecule = Chem.RemoveHs(molecule, sanitize=True)
            if molecule.GetNumAtoms() == 0:
                raise ValueError("RemoveHs 后没有重原子")
            Chem.SanitizeMol(molecule)
        except Exception as error:  # RDKit 会抛出多种 C++ 映射异常, 统一记录为科学排除.
            excluded.append({"smiles": smiles, "error": f"{type(error).__name__}: {error}"})
            continue
        # RDKit Mol 的当前重原子顺序是图包与未来 occurrence 坐标迁移的共同基础顺序.
        accepted.append(smiles)
        for atom in molecule.GetAtoms():
            elements.append(atom.GetAtomicNum())
            charges.append(atom.GetFormalCharge())
            atom_rings.append([atom.IsInRingSize(size) for size in RING_SIZES])
        for bond in molecule.GetBonds():
            try:
                bond_type = BOND_TYPES.index(bond.GetBondType())
            except ValueError as error:
                raise ValueError(f"{smiles!r} 含未定义键型 {bond.GetBondType()}。") from error
            bond_sources.append(bond.GetBeginAtomIdx())
            bond_targets.append(bond.GetEndAtomIdx())
            bond_types.append(bond_type)
            bond_rings.append([bond.IsInRingSize(size) for size in RING_SIZES])
        atom_offsets.append(len(elements))
        bond_offsets.append(len(bond_types))
        # int32 (N_atom,); 身份排列, 用于确认自同构集合包含不改变原子编号的情况.
        natural = np.arange(molecule.GetNumAtoms(), dtype=np.int32)
        # int32 (N_match, N_atom); 每行是保持元素、电荷和化学键的一个有序原子置换.
        matches = np.asarray(
            molecule.GetSubstructMatches(
                molecule,
                uniquify=False,
                useChirality=True,
                maxMatches=10000,
            ),
            dtype=np.int32,
        )
        if matches.ndim != 2 or matches.shape[1] != molecule.GetNumAtoms():
            raise ValueError(f"{smiles!r} 的自同构结果形状非法: {matches.shape}")
        if not np.any(np.all(matches == natural, axis=1)):
            matches = np.concatenate((natural[None], matches), axis=0)
        # bool (N_atom,); 始终固定不动的原子列不参与 Stage3 的坐标重分配.
        variable_atoms = np.any(matches != natural, axis=0)
        # int32 (N_match, N_variable); 保留会在至少一个自同构中改变的原子列.
        compact = matches[:, variable_atoms]
        symmetry_atom_count.append(molecule.GetNumAtoms())
        symmetry_shapes.append(compact.shape)
        symmetry_values.extend(compact.reshape(-1).tolist())
        symmetry_offsets.append(len(symmetry_values))
    # 图资产按 accepted 的共同顺序用 offsets 连接变长原子轴和无向键轴.
    graph_arrays = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "smiles": np.asarray(accepted, dtype=np.str_),
        "atom_offsets": np.asarray(atom_offsets, dtype=np.int64),
        "element": np.asarray(elements, dtype=np.int16),
        "charge": np.asarray(charges, dtype=np.int8),
        "atom_in_ring": np.asarray(atom_rings, dtype=np.bool_).reshape(-1, 4),
        "bond_offsets": np.asarray(bond_offsets, dtype=np.int64),
        "bond_index": np.asarray((bond_sources, bond_targets), dtype=np.int32).reshape(2, -1),
        "bond_type": np.asarray(bond_types, dtype=np.uint8),
        "bond_in_ring": np.asarray(bond_rings, dtype=np.bool_).reshape(-1, 4),
    }
    # 自同构资产复用同一 accepted 顺序; matches_offsets 连接每个 SMILES 的紧凑矩阵.
    symmetry_arrays = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "smiles": np.asarray(accepted, dtype=np.str_),
        "atom_count": np.asarray(symmetry_atom_count, dtype=np.int32),
        "matches_shape": np.asarray(symmetry_shapes, dtype=np.int32).reshape(-1, 2),
        "matches_offsets": np.asarray(symmetry_offsets, dtype=np.int64),
        "matches_iso": np.asarray(symmetry_values, dtype=np.int32),
    }
    return graph_arrays, symmetry_arrays, excluded


def save_npz_atomic(path: Path, arrays: dict[str, np.ndarray], compressed: bool) -> None:
    """在目标目录写临时文件并用 os.replace 原子发布一个 NPZ.

    - path: 最终文件路径.
    - arrays: dict[str, np.ndarray], 要写入 NPZ 的正式字段映射; 字段语义由 build_arrays 的 return[0] 或 return[1] 契约定义.
    - compressed: True 使用压缩 NPZ; False 保存未压缩 NPZ, 供 I/O 对照使用.
    - return: None; 创建父目录, 在同目录写临时 NPZ, 经 os.replace 发布到 path, 并在成功或失败后清理残留临时文件.
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        writer = np.savez_compressed if compressed else np.savez
        writer(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """在目标目录写临时 UTF-8 JSON, 并用 os.replace 原子发布.

    - path: 最终 JSON 文件路径.
    - value: dict[str, Any], 可由标准 JSON 编码器直接序列化的报告字段映射.
    - return: None; 创建父目录, 在同目录写临时 UTF-8 JSON, 经 os.replace 发布到 path, 并在成功或失败后清理残留临时文件.
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# ================================================================================================


def build_smiles_assets(data_root: Path, uncompressed: bool) -> int:
    """构建两个正式图包和一份 JSON 排除报告.

    - data_root: Path, AdaLigand Ori_Data 根目录; prepared_smiles.jsonl 从该目录读取, 三项资产写入该目录.
    - uncompressed: bool, True 使用未压缩 NPZ 供 I/O 对照, False 使用压缩 NPZ 作为正式资产.
    - return: int, 成功发布全部三项产物时返回 0.

    文件产物:
    - data_root/smiles_assets/smiles_graphs_v1.npz: NPZ, build_arrays 的 return[0] 图字段.
    - data_root/smiles_assets/smiles_symmetries_v1.npz: NPZ, build_arrays 的 return[1] 自同构字段.
    - data_root/smiles_assets/smiles_assets_v1_report.json: JSON object, 保存下列构建来源与排除统计.
    - schema_version: int, 固定为 1.
    - rdkit_version: str, 本次构建使用的 RDKit 版本.
    - exact_smiles_count: int, 扫描得到的唯一非空精确字符串数.
    - accepted_smiles_count: int, 同时进入图包与自同构包的字符串数.
    - excluded_smiles_count: int, 未通过严格化学构建的字符串数.
    - excluded: list[dict[str, Any]], build_arrays 返回的排除记录; 每项由 build_smiles_assets 追加全部 sources 来源记录.
    - excluded[*].smiles: str, 未进入两份资产的精确顶层 PreparedLigand.smiles.
    - excluded[*].error: str, RDKit 解析、去氢或 sanitize 异常的类型与消息.
    - excluded[*].sources: list[dict[str, Any]], 该精确 SMILES 在 prepared_smiles.jsonl 中的全部来源记录.
    - excluded[*].sources[*].pdb_id: str, 保存该字符串的小写 PDB 标识.
    - excluded[*].sources[*].occurrence_id: int, prepared_smiles.jsonl 的 candidate_id, 在新管线中解释为 occurrence 编号.
    - excluded[*].sources[*].path: str, 当前来源 prepared_smiles.jsonl 的完整路径.
    - excluded[*].sources[*].line_number: int, 当前来源记录在 JSONL 中从 1 开始的物理行号.
"""

    prepared_root = data_root / "stage1_preparation_box_pool_2/ligand_language_models/prepared"
    output_root = data_root / "smiles_assets"
    smiles_values, sources = read_exact_smiles(prepared_root)
    graph_arrays, symmetry_arrays, excluded = build_arrays(smiles_values)
    for record in excluded:
        record["sources"] = sources[record["smiles"]]
    save_npz_atomic(output_root / "smiles_graphs_v1.npz", graph_arrays, not uncompressed)
    save_npz_atomic(output_root / "smiles_symmetries_v1.npz", symmetry_arrays, not uncompressed)
    report = {
        "schema_version": 1,
        "rdkit_version": rdBase.rdkitVersion,
        "exact_smiles_count": len(smiles_values),
        "accepted_smiles_count": int(graph_arrays["smiles"].shape[0]),
        "excluded_smiles_count": len(excluded),
        "excluded": excluded,
    }
    report_path = output_root / "smiles_assets_v1_report.json"
    save_json_atomic(report_path, report)
    print(json.dumps({"graph": str(output_root / "smiles_graphs_v1.npz"), "symmetry": str(output_root / "smiles_symmetries_v1.npz"), **{key: value for key, value in report.items() if key != "excluded"}}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    """解析命令行并把两个参数交给正式资产构建入口.

    - argv: 可选命令行参数列表; None 表示读取当前进程命令行.
    - return: int, build_smiles_assets 返回的进程退出码.
"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--uncompressed", action="store_true")
    arguments = parser.parse_args(argv)
    return build_smiles_assets(arguments.data_root, arguments.uncompressed)


if __name__ == "__main__":
    raise SystemExit(main())
