from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest

from matcher.anchor_data import (
    AnchorDataConfig,
    AnchorPocketDataset,
    _ligand_node_input,
    _load_ligand,
    sample_candidate_starts,
)
from matcher.element_properties import ELEMENT_PROPERTIES


def _config(tmp_path: Path) -> AnchorDataConfig:
    return AnchorDataConfig(
        data_root=tmp_path,
        stage1_preparation_root=tmp_path,
        experiment_manifest=tmp_path / "manifest.json",
        split="train",
        training=True,
        p_miss=0.0,
        p_split=0.0,
        p_hit=1.0,
        max_context_ratio=0.0,
    )


def test_candidate_sampling_takes_one_bias_per_occurrence(tmp_path: Path) -> None:
    box = {
        "occurrence_id": np.asarray([3, 8], dtype=np.int32),
        "bias_start_zyx": np.asarray(
            [
                [[0, 0, 0], [1, 1, 1]],
                [[2, 2, 2], [3, 3, 3]],
            ],
            dtype=np.int32,
        ),
        "context_start_zyx": np.empty((0, 3), dtype=np.int32),
    }

    starts, audit = sample_candidate_starts(
        box,
        np.empty((0, 3), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
        np.random.default_rng(3407),
        _config(tmp_path),
    )

    assert len(starts) == 2
    assert audit == {"bias_count": 2, "context_requested": 0, "context_accepted": 0}


def test_empty_context_is_skipped_without_fabricating_candidate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = replace(
        config,
        p_miss=1.0,
        p_hit=0.0,
        max_context_ratio=1.0,
        empty_A_context_skip_probability=1.0,
    )
    box = {
        "occurrence_id": np.asarray([0], dtype=np.int32),
        "bias_start_zyx": np.zeros((1, 2, 3), dtype=np.int32),
        "context_start_zyx": np.asarray([[0, 0, 0]], dtype=np.int32),
    }
    rng = np.random.default_rng(7)

    starts, audit = sample_candidate_starts(
        box,
        np.empty((0, 3), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
        rng,
        config,
    )

    assert starts == []
    assert audit["context_accepted"] == 0


def test_dataset_rejects_manifest_sampling_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest = {
        "schema_version": 1,
        "route": "anchor_O_O_prime",
        "seed": 3407,
        "source_box_manifest": "box_pool/manifest.json",
        "source_box_manifest_sha256": "manifest-sha256",
        "source_box_config": "box_pool/config.json",
        "source_box_config_sha256": "config-sha256",
        "sampling": {
            "p_miss": 0.0,
            "p_split": 0.0,
            "p_hit": 1.0,
            "max_context_ratio": 0.0,
            "empty_A_context_skip_probability": 0.8,
            "receptor_radius_angstrom": 18.0,
        },
        "splits": {"train": []},
    }
    config.experiment_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    AnchorPocketDataset(config)
    with pytest.raises(ValueError, match="sampling.p_hit"):
        AnchorPocketDataset(replace(config, p_hit=0.5))

    del manifest["source_box_config_sha256"]
    config.experiment_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source_box_config_sha256"):
        AnchorPocketDataset(config)


def test_ligand_node_input_places_fixed_element_properties_at_145_to_149() -> None:
    assert len(ELEMENT_PROPERTIES) == 128
    atoms = np.zeros(
        1,
        dtype=[
            ("name", np.float32, (4,)),
            ("element", np.int64),
            ("charge", np.float32),
            ("chirality", np.float32, (7,)),
            ("in_ring", np.float32, (4,)),
            ("residue_id", np.float32),
        ],
    )
    atoms["element"] = 6

    features = _ligand_node_input(atoms)

    assert features.shape == (1, 149)
    assert features[0, 10] == 1.0
    assert features[0, 145:].tolist() == pytest.approx(ELEMENT_PROPERTIES[6])


def test_ligand_loader_copies_unaligned_structured_fields(tmp_path: Path) -> None:
    atom_dtype = np.dtype(
        [
            ("padding", np.uint8),
            ("ref_pos", np.float32, (3,)),
            ("name", np.float32, (4,)),
            ("element", np.int64),
            ("charge", np.float32),
            ("chirality", np.float32, (7,)),
            ("in_ring", np.float32, (4,)),
            ("residue_id", np.int64),
        ],
        align=False,
    )
    atoms = np.zeros(1, dtype=atom_dtype)
    atoms["element"] = 6
    bond_dtype = np.dtype(
        [
            ("atom_1", np.int64),
            ("atom_2", np.int64),
            ("type", np.bool_, (5,)),
            ("in_ring", np.bool_, (4,)),
        ]
    )
    ligand_root = tmp_path / "ligand_objects"
    ligand_root.mkdir()
    np.savez(ligand_root / "CCD_TST.npz", atoms=atoms, bonds=np.empty(0, dtype=bond_dtype))

    ligand = _load_ligand(tmp_path, "CCD:TST", _config(tmp_path))

    assert ligand.raw_graph.coordinates.shape == (1, 3)
    assert ligand.raw_graph.element.tolist() == [6]
