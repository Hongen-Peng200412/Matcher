"""为新 Matcher 数据契约测试建立最小但可端到端读取的冻结产物。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from matcher.schema import LANGUAGE_MODEL_NAMES, SourceRoots


def build_synthetic_roots(root: Path) -> SourceRoots:
    """创建一个完整 train PDB、一个 blob_exceed 和一个缺失 F1 的三划分样例。"""

    data_root = root / "source"
    stage1_root = root / "stage1"
    output_root = root / "output"
    for split_name in ("train", "validation", "calibration"):
        (stage1_root / split_name).mkdir(parents=True)
    (stage1_root / "validation/2bad").mkdir()
    (stage1_root / "validation/2bad/_BLOB_EXCEED").touch()
    (stage1_root / "calibration/3bad").mkdir()

    pdb_id = "1abc"
    stage1_pdb = stage1_root / "train" / pdb_id
    for relative in ("centered", "probability"):
        (stage1_pdb / relative).mkdir(parents=True)
    for relative in (f"parse/{pdb_id}", f"density/{pdb_id}", f"labels/{pdb_id}", "ligand_objects"):
        (data_root / relative).mkdir(parents=True)

    occurrences = [
        {
            "pdb_id": pdb_id,
            "candidate_id": candidate_id,
            "object_key": "CCD:LIG",
            "type_tag": "small_molecule",
        }
        for candidate_id in (0, 1)
    ]
    occurrence_text = "".join(json.dumps(item) + "\n" for item in occurrences)
    (data_root / f"parse/{pdb_id}/occurrences.jsonl").write_text(
        occurrence_text,
        encoding="utf-8",
    )

    receptor_dtype = np.float32
    np.savez(
        data_root / f"parse/{pdb_id}/receptor_tokens.npz",
        coords=np.asarray([[40.5, 40.5, 40.5]], dtype=receptor_dtype),
        feat=np.ones((1, 49), dtype=receptor_dtype),
    )
    grid = np.linspace(-1.0, 1.0, 80**3, dtype=np.float32).reshape(1, 80, 80, 80)
    geometry = {
        "grid": grid,
        "voxel_size": np.ones(3, dtype=np.float32),
        "origin": np.zeros(3, dtype=np.float32),
    }
    np.savez(data_root / f"density/{pdb_id}/exp.npz", **geometry)
    np.savez(data_root / f"density/{pdb_id}/sim.npz", **geometry)
    candidate_mask = np.asarray([[0, 0, 0], [40, 40, 40]], dtype=np.int32)
    np.savez(
        data_root / f"density/{pdb_id}/ligand_area.npz",
        mask_0=candidate_mask,
        mask_1=np.asarray([[40, 40, 40]], dtype=np.int32),
    )
    np.savez(
        data_root / f"labels/{pdb_id}/atom_labels.npz",
        binding_atom=np.asarray([True]),
    )
    _write_centered(stage1_pdb / "centered/F1_centered.npz", candidate_mask)
    probability = np.zeros((80, 80, 80), dtype=np.float32)
    probability[40, 40, 40] = 1.0
    np.savez(
        stage1_pdb / "probability/probability_map.npz",
        probability_map=probability,
        origin_xyz=np.zeros(3, dtype=np.float32),
        voxel_size_xyz=np.ones(3, dtype=np.float32),
    )
    _write_language_outputs(data_root, pdb_id)
    _write_ligand_object(data_root / "ligand_objects/CCD_LIG.npz")
    return SourceRoots(data_root, stage1_root, output_root)


def _write_centered(path: Path, candidate_mask: np.ndarray) -> None:
    """写入一个候选及 A/P/V 最小字段。"""

    np.savez(
        path,
        centered_box_index=np.asarray([0], dtype=np.int32),
        box_start_zyx=np.zeros((1, 3), dtype=np.int32),
        box_origin_world=np.zeros((1, 3), dtype=np.float32),
        voxel_size_world=np.ones((1, 3), dtype=np.float32),
        source_tree_id=np.asarray([1], dtype=np.int32),
        source_node_id=np.asarray([2], dtype=np.int32),
        source_threshold_grid_index=np.asarray([3], dtype=np.int32),
        voxel_offsets=np.asarray([0, 2], dtype=np.int64),
        voxel_index_local_zyx=candidate_mask.astype(np.int16),
        centered_probability=np.asarray([0.7, 0.9], dtype=np.float32),
        voxel_final=np.ones((2, 48), dtype=np.float16),
        P_offsets=np.asarray([0, 2], dtype=np.int64),
        P_coord_local_xyz=np.asarray([[0.5, 0.5, 0.5], [40.5, 40.5, 40.5]], dtype=np.float32),
        P_probability=np.asarray([0.1, 0.9], dtype=np.float32),
        P_feat_L2=np.ones((2, 64), dtype=np.float16),
        P_feat_L3=np.ones((2, 48), dtype=np.float16),
        A_offsets=np.asarray([0, 1], dtype=np.int64),
        A_global_index=np.asarray([0], dtype=np.int64),
        A_coord_local_xyz=np.asarray([[40.5, 40.5, 40.5]], dtype=np.float32),
        A_coord_centered_world=np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32),
        A_probability=np.asarray([0.8], dtype=np.float32),
        A_feat_L0=np.ones((1, 49), dtype=np.float32),
        A_feat_L1=np.ones((1, 64), dtype=np.float16),
        A_feat_L2=np.ones((1, 64), dtype=np.float16),
        A_feat_L3=np.ones((1, 48), dtype=np.float16),
    )


def _write_language_outputs(data_root: Path, pdb_id: str) -> None:
    """写入两套语言模型的有效/缺失 occurrence 组合。"""

    language_root = data_root / "stage1_preparation_box_pool_2/ligand_language_models"
    for family, candidate_ids in (("molformer", (0, 1)), ("smi_ted_289m", (1,))):
        directory = language_root / family / pdb_id
        directory.mkdir(parents=True)
        for candidate_id in candidate_ids:
            np.savez(
                directory / f"candidate_{candidate_id}.npz",
                pdb_id=np.asarray(pdb_id),
                candidate_id=np.asarray(candidate_id, dtype=np.int32),
                object_key=np.asarray("CCD:LIG"),
                model_name=np.asarray(LANGUAGE_MODEL_NAMES[family]),
                model_smiles=np.asarray("CC" if candidate_id == 0 else "C=C"),
                embedding=np.full(768, candidate_id + 1, dtype=np.float32),
            )


def _write_ligand_object(path: Path) -> None:
    """写入含两个原子和一条单键的最小 LigandObject。"""

    atom_dtype = np.dtype(
        [
            ("element", "i1"),
            ("charge", "i1"),
            ("ref_pos", "3f4"),
            ("is_present", "?"),
            ("chirality", "7?"),
            ("in_ring", "4?"),
            ("residue_id", "i4"),
        ]
    )
    bond_dtype = np.dtype(
        [("atom_1", "i4"), ("atom_2", "i4"), ("type", "5?"), ("in_ring", "4?")]
    )
    atoms = np.zeros(2, dtype=atom_dtype)
    atoms["element"] = [6, 6]
    atoms["ref_pos"] = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
    atoms["is_present"] = True
    atoms["residue_id"] = 1
    bonds = np.zeros(1, dtype=bond_dtype)
    bonds["atom_1"] = 0
    bonds["atom_2"] = 1
    bonds["type"][0, 0] = True
    np.savez(path, atoms=atoms, bonds=bonds, name=np.asarray("CCD:LIG"))
