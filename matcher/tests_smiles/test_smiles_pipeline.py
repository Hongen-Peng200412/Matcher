from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from matcher.smiles_identity.batching import collate_candidates
from matcher.smiles_identity.config import load_smiles_experiment
from matcher.smiles_identity.dataset import CandidateDataset
from matcher.smiles_identity.engine import _summarize_training_window, _validate
from matcher.smiles_identity.graph import SmilesGraphStore
from matcher.smiles_identity.labels import OccurrenceMask, assign_candidate_target
from matcher.smiles_identity.losses import LossWeights, compute_matcher_loss
from matcher.smiles_identity.manifest import _build_pdb_identities, read_split_records
from matcher.smiles_identity.model import ClosedSetMatcher, GraphIdentityAttention, LigandGraphEncoder
from matcher.smiles_identity.model_config import MatcherModelConfig
from matcher.smiles_identity.schema import SMI_TED_MODEL_NAME, SourceRoots


def test_label_assignment_uses_occurrence_id_tie_break() -> None:
    candidate = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int64)
    occurrences = [
        OccurrenceMask(8, "ion", candidate.copy()),
        OccurrenceMask(3, "small_molecule", candidate.copy()),
    ]
    target = assign_candidate_target(candidate, occurrences, 0.5)
    assert target.matched_occurrence_id == 3
    assert target.class_index == 4
    assert target.overlap_score == 1.0


def test_dataset_batch_and_graph_encoder(tmp_path: Path) -> None:
    manifest_path = _write_minimal_dataset(tmp_path)
    dataset = CandidateDataset(manifest_path, "validation", 8, 0.1)
    item = dataset[0]
    assert item["voxel_input"].shape == (106, 48, 48, 48)
    assert item["a_tokens"].shape == (1, 230)
    assert item["p_tokens"].shape == (1, 116)
    assert item["v_tokens"].shape == (1, 68)
    assert item["identity_smiles"] == ["C"]
    assert item["identity_target"] == 0
    batch = collate_candidates([item])
    assert batch["graph_atom_in_ring"].shape == (1, 4)
    config = MatcherModelConfig(
        aggregation="mean_max",
        use_manual_features=True,
        use_sampling_features=True,
        use_candidate_token_attention_bias=False,
        initial_graph_gate=0.0,
        initial_inventory_gate=0.0,
        model_dim=16,
        attention_heads=4,
        dropout=0.0,
        voxel_feature_channels=8,
        raunet_planes=(8, 8, 8, 8, 8, 8, 8, 8, 8),
        recycle_iterations=1,
        gradient_checkpointing=False,
        bottleneck_attention_layers=1,
        graph_layers=1,
    )
    nodes = LigandGraphEncoder(config.model_dim, config.graph_layers, config.dropout)(batch)
    assert nodes.shape == (1, 16)
    assert torch.isfinite(nodes).all()
    model = ClosedSetMatcher(config).eval()
    with torch.no_grad():
        outputs = model(batch)
    assert outputs["class_logits"].shape == (1, 6)
    assert outputs["identity_logits"].shape == (1, 1)
    assert torch.isfinite(outputs["identity_logits"]).all()


def test_loss_uses_all_foreground_identity_targets() -> None:
    outputs = {
        "class_logits": torch.tensor([[0.0, 0.0, 0.0, 0.0, 2.0, 0.0]]),
        "ligand_area_logits": torch.zeros((1, 2, 2, 2)),
        "binding_area_logits": torch.zeros((1, 2, 2, 2)),
        "identity_logits": torch.tensor([[1.0, 0.0]]),
    }
    batch = {
        "class_target": torch.tensor([4]),
        "ligand_area_target": torch.zeros((1, 2, 2, 2), dtype=torch.bool),
        "binding_area_target": torch.zeros((1, 2, 2, 2), dtype=torch.bool),
        "identity_target": torch.tensor([0]),
        "identity_mask": torch.tensor([[True, True]]),
        "identity_class_index": torch.tensor([[4, 4]]),
    }
    result = compute_matcher_loss(outputs, batch, LossWeights(1.0, 0.1, 0.1, 1.0), "oracle")
    assert int(result["identity_eligible_count"]) == 1
    assert torch.isfinite(result["total"])


def test_formal_yaml_is_complete() -> None:
    path = Path("configs/experiments/smiles_identity/smiles_identity_find1_validation_resident_a800.yaml")
    config = load_smiles_experiment(path)
    assert config.data.train_split == config.data.validation_split == "validation"
    assert config.optimization.save_checkpoints is False
    assert config.tracking.mode == "online"


def test_duplicate_language_vectors_use_smallest_occurrence_with_strict_tolerance(
    tmp_path: Path,
) -> None:
    roots = SourceRoots(tmp_path, tmp_path / "find1", tmp_path / "output")
    language_root = roots.language_root / "test"
    language_root.mkdir(parents=True)
    for occurrence_id, offset in ((1, 0.0), (2, 4e-6)):
        np.savez_compressed(
            language_root / f"candidate_{occurrence_id}.npz",
            pdb_id=np.asarray("test"),
            candidate_id=np.asarray(occurrence_id, dtype=np.int64),
            prepared_smiles=np.asarray("C"),
            model_name=np.asarray(SMI_TED_MODEL_NAME),
            embedding=np.full(768, 1.0 + offset, dtype=np.float32),
        )
    records = _build_pdb_identities(
        roots,
        "test",
        {1: {"type_tag": "small_molecule"}, 2: {"type_tag": "small_molecule"}},
        {1: "C", 2: "C"},
        {"C"},
    )
    assert records[0]["occurrence_ids"] == [1, 2]
    assert records[0]["language_relative_path"].endswith("candidate_1.npz")

    with np.load(language_root / "candidate_2.npz", allow_pickle=False) as archive:
        fields = {name: np.asarray(archive[name]) for name in archive.files}
    fields["embedding"] = np.full(768, 1.0 + 6e-6, dtype=np.float32)
    np.savez_compressed(language_root / "candidate_2.npz", **fields)
    with pytest.raises(ValueError, match="最大绝对差"):
        _build_pdb_identities(
            roots,
            "test",
            {1: {"type_tag": "small_molecule"}, 2: {"type_tag": "small_molecule"}},
            {1: "C", 2: "C"},
            {"C"},
        )


def test_smiles_identity_keeps_all_occurrences_when_one_language_file_exists(
    tmp_path: Path,
) -> None:
    roots = SourceRoots(tmp_path, tmp_path / "find1", tmp_path / "output")
    language_root = roots.language_root / "test"
    language_root.mkdir(parents=True)
    np.savez_compressed(
        language_root / "candidate_2.npz",
        pdb_id=np.asarray("test"),
        candidate_id=np.asarray(2, dtype=np.int64),
        prepared_smiles=np.asarray("C"),
        model_name=np.asarray(SMI_TED_MODEL_NAME),
        embedding=np.ones(768, dtype=np.float32),
    )
    records = _build_pdb_identities(
        roots,
        "test",
        {1: {"type_tag": "small_molecule"}, 2: {"type_tag": "small_molecule"}},
        {1: "C", 2: "C"},
        {"C"},
    )
    assert records[0]["occurrence_ids"] == [1, 2]
    assert records[0]["language_relative_path"].endswith("candidate_2.npz")


def test_all_background_and_mixed_identity_batches_support_backward(tmp_path: Path) -> None:
    manifest_path = _write_minimal_dataset(tmp_path)
    item = CandidateDataset(manifest_path, "validation", 8, 0.1)[0]
    background = dict(item)
    background["metadata"] = {**item["metadata"], "matched_occurrence_id": None, "matched_smiles": None}
    background["class_target"] = 0
    background["identity_smiles"] = []
    background["identity_class_index"] = np.empty(0, dtype=np.int64)
    background["language_embeddings"] = np.empty((0, 768), dtype=np.float32)
    background["smiles_graphs"] = []
    background["identity_target"] = -1
    config = MatcherModelConfig(
        aggregation="mean_max",
        use_manual_features=True,
        use_sampling_features=True,
        use_candidate_token_attention_bias=False,
        initial_graph_gate=0.0,
        initial_inventory_gate=0.0,
        model_dim=16,
        attention_heads=4,
        dropout=0.0,
        voxel_feature_channels=8,
        raunet_planes=(8, 8, 8, 8, 8, 8, 8, 8, 8),
        recycle_iterations=1,
        gradient_checkpointing=False,
        bottleneck_attention_layers=1,
        graph_layers=1,
    )
    model = ClosedSetMatcher(config)
    for items in ([background], [item, background]):
        batch = collate_candidates(items)
        outputs = model(batch)
        losses = compute_matcher_loss(
            outputs,
            batch,
            LossWeights(1.0, 0.1, 0.1, 1.0),
            "oracle",
        )
        assert int(losses["identity_eligible_count"]) == len(items) - 1
        assert torch.isfinite(losses["total"])
        model.zero_grad(set_to_none=True)
        losses["total"].backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_validation_loss_is_invariant_to_batch_partition() -> None:
    class BatchBackedModel:
        def eval(self) -> None:
            return None

        def train(self) -> None:
            return None

        def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {
                "class_logits": batch["mock_class_logits"],
                "ligand_area_logits": batch["mock_ligand_area_logits"],
                "binding_area_logits": batch["mock_binding_area_logits"],
                "identity_logits": batch["mock_identity_logits"],
            }

    complete = {
        "class_target": torch.tensor([4, 0, 0, 4]),
        "ligand_area_target": torch.tensor([[[[1]]], [[[0]]], [[[0]]], [[[1]]]], dtype=torch.bool),
        "binding_area_target": torch.tensor([[[[0]]], [[[1]]], [[[0]]], [[[1]]]], dtype=torch.bool),
        "identity_target": torch.tensor([0, -1, -1, 1]),
        "identity_mask": torch.ones((4, 2), dtype=torch.bool),
        "identity_class_index": torch.full((4, 2), 4, dtype=torch.long),
        "mock_class_logits": torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -2.0, 0.0],
            ]
        ),
        "mock_ligand_area_logits": torch.tensor([[[[2.0]]], [[[1.0]]], [[[-1.0]]], [[[-2.0]]]]),
        "mock_binding_area_logits": torch.tensor([[[[-1.0]]], [[[2.0]]], [[[3.0]]], [[[-3.0]]]]),
        "mock_identity_logits": torch.tensor([[3.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2.0, -1.0]]),
    }

    def partition(groups: list[list[int]]) -> list[dict[str, torch.Tensor]]:
        return [
            {key: value[indices] for key, value in complete.items()}
            for indices in groups
        ]

    config = SimpleNamespace(
        loss_weights=LossWeights(1.0, 0.1, 0.1, 1.0),
        identity_variant="oracle",
    )
    model = BatchBackedModel()
    first = _validate(model, partition([[0, 1, 2, 3]]), config, torch.device("cpu"), 0, None)
    second = _validate(model, partition([[0], [1, 2, 3]]), config, torch.device("cpu"), 0, None)
    for field in ("classification_loss", "ligand_area_loss", "binding_area_loss", "identity_loss", "total_loss"):
        assert second[field] == pytest.approx(first[field], abs=1e-7)


def test_training_window_reports_all_microbatches() -> None:
    records = [
        {
            "total": float(index),
            "classification": float(index + 1),
            "ligand_area": float(index + 2),
            "binding_area": float(index + 3),
            "identity": float(index + 4),
            "identity_eligible_count": float(index % 2),
        }
        for index in range(4)
    ]
    result = _summarize_training_window(records)
    assert result["total"] == pytest.approx(1.5)
    assert result["classification"] == pytest.approx(2.5)
    assert result["identity"] == pytest.approx(5.5)
    assert result["identity_eligible_count"] == 2.0


def test_graph_store_rejects_bond_endpoint_outside_local_graph(tmp_path: Path) -> None:
    graph_path = tmp_path / "invalid_graph.npz"
    np.savez_compressed(
        graph_path,
        schema_version=np.asarray(1, dtype=np.int64),
        smiles=np.array(["C"]),
        atom_offsets=np.array([0, 1], dtype=np.int64),
        element=np.array([6], dtype=np.int16),
        charge=np.array([0], dtype=np.int8),
        atom_in_ring=np.zeros((1, 4), dtype=np.bool_),
        bond_offsets=np.array([0, 1], dtype=np.int64),
        bond_index=np.array([[0], [1]], dtype=np.int32),
        bond_type=np.array([0], dtype=np.uint8),
        bond_in_ring=np.zeros((1, 4), dtype=np.bool_),
    )
    with pytest.raises(ValueError, match="bond_index"):
        SmilesGraphStore(graph_path).initialize()


def test_manifest_rejects_old_identity_field(tmp_path: Path) -> None:
    manifest_path = _write_minimal_dataset(tmp_path)
    candidate_path = manifest_path.parent / "validation_candidates.jsonl"
    candidate = json.loads(candidate_path.read_text())
    candidate["object_key"] = "CCD:C"
    candidate_path.write_text(json.dumps(candidate) + "\n")
    with pytest.raises(ValueError, match="废止字段"):
        read_split_records(manifest_path, "validation")


def test_smiles_graph_attention_is_invariant_to_atom_renumbering() -> None:
    torch.manual_seed(7)
    graph_encoder = LigandGraphEncoder(8, 2, 0.0).eval()
    graph_attention = GraphIdentityAttention(8, 2, 0.0).eval()
    element = torch.tensor([6, 7, 8])
    charge = torch.tensor([0, 1, -1])
    atom_ring = torch.zeros((3, 4))
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_features = torch.arange(36, dtype=torch.float32).reshape(4, 9) / 36
    native = torch.randn(1, 1, 8)

    def encode(
        node_order: torch.Tensor,
        remapped_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        batch = {
            "graph_element": element[node_order],
            "graph_charge": charge[node_order],
            "graph_atom_in_ring": atom_ring[node_order],
            "graph_edge_index": remapped_edge_index,
            "graph_edge_features": edge_features,
        }
        nodes = graph_encoder(batch)
        return graph_attention(
            native,
            nodes,
            torch.zeros(3, dtype=torch.long),
            torch.tensor([0]),
            torch.tensor([0]),
        )

    natural = torch.arange(3)
    permutation = torch.tensor([2, 0, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = natural
    with torch.no_grad():
        expected = encode(natural, edge_index)
        actual = encode(permutation, inverse[edge_index])
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


def _write_minimal_dataset(root: Path) -> Path:
    data_root = root / "data"
    stage1_root = root / "find1"
    output_root = root / "output"
    pdb_id = "test"
    centered_root = stage1_root / "validation" / pdb_id / "centered"
    centered_root.mkdir(parents=True)
    arrays = {
        "source_blob_index": np.array([7], dtype=np.int32),
        "box_start_zyx": np.array([[0, 0, 0]], dtype=np.int32),
        "box_shape_zyx": np.array([[80, 80, 80]], dtype=np.uint8),
        "box_origin_world": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "voxel_size_world": np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        "source_voxel_count": np.array([1], dtype=np.int32),
        "centered_box_index": np.array([0], dtype=np.int32),
        "voxel_offsets": np.array([0, 1], dtype=np.int64),
        "voxel_index_local_zyx": np.array([[40, 40, 40]], dtype=np.int16),
        "centered_probability": np.array([0.9], dtype=np.float32),
        "A_offsets": np.array([0, 1], dtype=np.int64),
        "A_coord_local_xyz": np.array([[40.0, 40.0, 40.0]], dtype=np.float32),
        "A_probability": np.array([0.8], dtype=np.float32),
        "A_feat_L0": np.zeros((1, 50), dtype=np.float32),
        "A_feat_L1": np.zeros((1, 64), dtype=np.float16),
        "A_feat_L2": np.zeros((1, 64), dtype=np.float16),
        "A_feat_L3": np.zeros((1, 48), dtype=np.float16),
        "P_offsets": np.array([0, 1], dtype=np.int64),
        "P_coord_local_xyz": np.array([[40.0, 40.0, 40.0]], dtype=np.float32),
        "P_probability": np.array([0.7], dtype=np.float32),
        "P_feat_L2": np.zeros((1, 64), dtype=np.float16),
        "P_feat_L3": np.zeros((1, 48), dtype=np.float16),
        "voxel_final": np.zeros((1, 64), dtype=np.float16),
        "crop_start_local_zyx": np.array([[16, 16, 16]], dtype=np.int16),
        "crop_center_offset_zyx": np.zeros((1, 3), dtype=np.float32),
        "crop_clipped_axis_mask": np.zeros((1, 3), dtype=np.bool_),
        "experimental_density_48": np.zeros((1, 48, 48, 48), dtype=np.float32),
        "simulated_density_48": np.zeros((1, 48, 48, 48), dtype=np.float32),
        "source_probability_48": np.zeros((1, 48, 48, 48), dtype=np.float32),
        "selected": np.array([True], dtype=np.bool_),
    }
    np.savez_compressed(centered_root / "F2_centered.npz", **arrays)
    receptor_root = data_root / "parse" / pdb_id
    receptor_root.mkdir(parents=True)
    np.savez_compressed(
        receptor_root / "receptor_tokens.npz",
        coords=np.array([[40.0, 40.0, 40.0]], dtype=np.float32),
        feat=np.ones((1, 49), dtype=np.float32),
        is_backbone=np.array([True]),
    )
    label_root = data_root / "labels" / pdb_id
    label_root.mkdir(parents=True)
    np.savez_compressed(label_root / "atom_labels.npz", binding_atom=np.array([True]))
    ligand_root = data_root / "density" / pdb_id
    ligand_root.mkdir(parents=True)
    np.savez_compressed(
        ligand_root / "ligand_area.npz",
        mask_0=np.array([[40, 40, 40]], dtype=np.int32),
        mask_index_order=np.asarray("zyx"),
        small_molecule_mask=np.array([[40, 40, 40]], dtype=np.int32),
    )
    language_root = data_root / "language" / pdb_id
    language_root.mkdir(parents=True)
    np.savez_compressed(language_root / "candidate_0.npz", embedding=np.ones(768, dtype=np.float32))
    asset_root = data_root / "smiles_assets"
    asset_root.mkdir(parents=True)
    graph_path = asset_root / "smiles_graphs_v1.npz"
    np.savez_compressed(
        graph_path,
        schema_version=np.asarray(1, dtype=np.int64),
        smiles=np.array(["C"]),
        atom_offsets=np.array([0, 1], dtype=np.int64),
        element=np.array([6], dtype=np.int16),
        charge=np.array([0], dtype=np.int8),
        atom_in_ring=np.zeros((1, 4), dtype=np.bool_),
        bond_offsets=np.array([0, 0], dtype=np.int64),
        bond_index=np.empty((2, 0), dtype=np.int32),
        bond_type=np.empty(0, dtype=np.uint8),
        bond_in_ring=np.empty((0, 4), dtype=np.bool_),
    )
    manifest_root = output_root / "manifests" / "smiles_identity"
    manifest_root.mkdir(parents=True)
    candidate = {
        "split": "validation",
        "pdb_id": pdb_id,
        "centered_box_index": 0,
        "source_blob_index": 7,
        "stage1_selected": True,
        "class_index": 4,
        "matched_occurrence_id": 0,
        "matched_smiles": "C",
        "candidate_coverage": 1.0,
        "occurrence_coverage": 1.0,
        "overlap_score": 1.0,
    }
    identity = {
        "pdb_id": pdb_id,
        "smiles": "C",
        "class_index": 4,
        "occurrence_ids": [0],
        "language_relative_path": "language/test/candidate_0.npz",
    }
    (manifest_root / "validation_candidates.jsonl").write_text(json.dumps(candidate) + "\n")
    (manifest_root / "validation_identities.jsonl").write_text(json.dumps(identity) + "\n")
    candidate_path = manifest_root / "validation_candidates.jsonl"
    identity_path = manifest_root / "validation_identities.jsonl"
    manifest = {
        "schema": "smiles_identity_v1",
        "class_names": ["background", "ion", "nucleotide_like", "peptide_like", "small_molecule", "sugar"],
        "identity_key": "PreparedLigand.smiles",
        "language_model": "SMI-TED Light 289M",
        "data_root": str(data_root),
        "stage1_root": str(stage1_root),
        "graph_asset_path": str(graph_path),
        "splits": {
            "validation": {
                "candidates": candidate_path.name,
                "identities": identity_path.name,
            }
        },
        "summary": {
            "validation": {
                "pdb_count": 1,
                "candidate_count": 1,
                "foreground_candidate_count": 1,
                "identity_count": 1,
                "unique_smiles_count": 1,
                "skipped_positive_candidate_count": 0,
            }
        },
    }
    manifest_path = manifest_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path
