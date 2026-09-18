from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import ops.matcher.build_stage1_manifest as manifest_builder

from matcher.data.density import (
    DENSITY_CHANNEL_NAMES,
    build_density_channels,
    build_experimental_density_channel,
)
from matcher.data.labels import OccurrenceMask, assign_candidate_target
from matcher.data.smiles_graph import SmilesGraph, SmilesGraphStore
from matcher.datasets.batching import collate_candidates
from matcher.datasets.candidate_data import load_manifest_dataset
from matcher.evaluation.checkpoint import load_matcher_from_training_release
from matcher.evaluation.metrics import (
    fit_foreground_threshold,
    foreground_average_precision,
    smiles_match_metrics,
)
from matcher.evaluation.test_set import (
    _candidate_hit_mask,
    _matcher_curve,
    _metrics_from_rows,
    _ranked_hit_curve,
)
from matcher.model.config import MatcherModelConfig
from matcher.model.ligand_matcher import ClosedSetMatcher
from matcher.training.config import load_smiles_experiment
from matcher.training.engine import (
    _ValidationPolicyState,
    _partition_optimizer_updates,
    _train_epoch_segments,
    _write_manifest_snapshot,
)
from matcher.training.losses import (
    LossWeights,
    compute_matcher_loss,
    smiles_identity_cross_entropy,
)


def test_checkpoint_loader_uses_training_release_package(tmp_path: Path) -> None:
    project_root = tmp_path / "Matcher_snapshot" / "Matcher"
    for relative in ("matcher", "matcher/model", "matcher/training"):
        directory = project_root / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("", encoding="utf-8")
    (project_root / "matcher/model/config.py").write_text(
        "class MatcherModelConfig:\n"
        "    def __init__(self, variant):\n"
        "        self.variant = variant\n"
        "    @classmethod\n"
        "    def from_dict(cls, values):\n"
        "        return cls(values['variant'])\n",
        encoding="utf-8",
    )
    (project_root / "matcher/model/ligand_matcher.py").write_text(
        "from torch import nn\n"
        "class ClosedSetMatcher(nn.Module):\n"
        "    def __init__(self, config):\n"
        "        super().__init__()\n"
        "        self.config = config\n"
        "        self.projection = nn.Linear(1, 1, bias=False)\n",
        encoding="utf-8",
    )
    (project_root / "matcher/training/checkpointing.py").write_text(
        "import torch\n"
        "def load_checkpoint(path, map_location='cpu'):\n"
        "    return torch.load(path, map_location=map_location, weights_only=False)\n",
        encoding="utf-8",
    )
    release_manifest = project_root.parent / "manifest.json"
    release_manifest.write_text("{}\n", encoding="utf-8")
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "run.json").write_text(
        json.dumps(
            {
                "source_project_root": str(project_root),
                "source_release_manifest_path": str(release_manifest),
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = run_directory / "BEST.pt"
    torch.save(
        {
            "model_config": {"variant": "snapshot_variant"},
            "model_state": {"projection.weight": torch.tensor([[2.0]])},
        },
        checkpoint_path,
    )

    model, checkpoint, config, provenance = load_matcher_from_training_release(
        checkpoint_path,
        torch.device("cpu"),
    )

    assert config.variant == "snapshot_variant"
    assert model.__class__.__module__.startswith("_matcher_training_snapshot_")
    assert torch.equal(
        model.projection.weight,
        checkpoint["model_state"]["projection.weight"],
    )
    assert provenance["training_project_root"] == str(project_root.resolve())


def test_manifest_source_mismatch_does_not_touch_published_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "manifest.json"
    candidate_path = tmp_path / "old_candidates.jsonl"
    identity_path = tmp_path / "old_identities.jsonl"
    candidate_path.write_text("candidate sentinel\n", encoding="utf-8")
    identity_path.write_text("identity sentinel\n", encoding="utf-8")
    output_path.write_text(
        json.dumps(
            {
                "schema": "matcher_stage1_manifest_v1",
                "class_names": [
                    "background",
                    "ion",
                    "nucleotide_like",
                    "peptide_like",
                    "small_molecule",
                    "sugar",
                ],
                "identity_key": "PreparedLigand.smiles",
                "language_model": "SMI-TED Light 289M",
                "data_root": str(tmp_path / "old_data"),
                "stage1_root": str(tmp_path / "stage1"),
                "graph_asset_path": str(tmp_path / "graph.npz"),
                "splits": {
                    "validation": {
                        "candidates": candidate_path.name,
                        "identities": identity_path.name,
                    }
                },
                "summary": {"validation": {}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="固定来源"):
        manifest_builder.build_stage1_manifest(
            tmp_path / "stage1",
            "validation",
            output_path,
            data_root=tmp_path / "different_data",
            graph_asset_path=tmp_path / "graph.npz",
        )
    assert candidate_path.read_text(encoding="utf-8") == "candidate sentinel\n"
    assert identity_path.read_text(encoding="utf-8") == "identity sentinel\n"


def _graph() -> SmilesGraph:
    return SmilesGraph(
        element=np.asarray([6, 6], dtype=np.int64),
        charge=np.asarray([0, 0], dtype=np.int64),
        atom_in_ring=np.zeros((2, 4), dtype=np.float32),
        edge_index=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        edge_features=np.asarray(
            [[1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]],
            dtype=np.float32,
        ),
    )


def _item(variant: str) -> dict[str, object]:
    spatial = 48
    if variant == "pocket_only":
        voxel = np.empty((0, spatial, spatial, spatial), dtype=np.float16)
        a_tokens = np.ones((2, 50), dtype=np.float32)
    else:
        voxel_channels = 1 if variant == "density_blob_only" else 106
        voxel = np.zeros((voxel_channels, spatial, spatial, spatial), dtype=np.float16)
        a_tokens = (
            np.empty((0, 230), dtype=np.float32)
            if variant == "density_blob_only"
            else np.ones((1, 230), dtype=np.float32)
        )
    v_tokens = (
        np.ones((1, 68), dtype=np.float32)
        if variant != "pocket_only"
        else np.empty((0, 68), dtype=np.float32)
    )
    return {
        "metadata": {
            "split": "validation",
            "pdb_id": "1abc",
            "centered_box_index": 0,
            "source_blob_index": 0,
            "candidate_selected": True,
            "gaussian_score": 1.0,
            "source_probability_mean": 0.5,
            "foreground_target": True,
            "target_category": "small_molecule",
            "matched_occurrence_id": 0,
            "target_smiles": "CC",
            "identity_supported": True,
        },
        "voxel_input": voxel,
        "a_tokens": a_tokens,
        "p_tokens": np.empty((0, 116), dtype=np.float32),
        "v_tokens": v_tokens,
        "v_index_zyx": np.zeros((len(v_tokens), 3), dtype=np.int64),
        "pp_tokens": np.empty((0, 4), dtype=np.float32),
        "pp_index_zyx": np.empty((0, 3), dtype=np.int64),
        "foreground_target": True,
        "ligand_area_target": np.zeros((spatial,) * 3, dtype=np.bool_),
        "binding_area_target": np.zeros((spatial,) * 3, dtype=np.bool_),
        "full_candidate_voxel_count": 1,
        "candidate_manual_features": np.zeros(12, dtype=np.float32),
        "candidate_sampling_features": np.zeros(4, dtype=np.float32),
        "identity_smiles": ["CC"],
        "language_embeddings": np.ones((1, 768), dtype=np.float32),
        "smiles_graphs": [] if variant == "trunk_only" else [_graph()],
        "identity_target": 0,
    }


def _config(variant: str) -> MatcherModelConfig:
    return MatcherModelConfig(
        variant=variant,
        model_dim=24,
        attention_heads=2,
        dropout=0.0,
        voxel_feature_channels=8,
        raunet_planes=(8, 8, 8, 8, 8, 8, 8, 8, 8),
        recycle_iterations=1,
        gradient_checkpointing=False,
        bottleneck_attention_layers=1,
        graph_layers=1,
    )


def test_maximum_overlap_target_is_unique_and_uses_smaller_occurrence_id() -> None:
    candidate = np.asarray([[0, 0, 0], [0, 0, 1]], dtype=np.int64)
    occurrences = [
        OccurrenceMask(2, "small_molecule", candidate.copy()),
        OccurrenceMask(1, "sugar", candidate.copy()),
    ]
    target = assign_candidate_target(candidate, occurrences, 0.30)
    assert target.foreground_target is True
    assert target.matched_occurrence_id == 1
    assert target.target_category == "sugar"


def test_single_density_channel_matches_exp_clipnorm_nopost() -> None:
    experimental = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    simulated = np.flip(experimental).copy()
    hardmask = np.zeros_like(experimental, dtype=np.bool_)
    complete = build_density_channels(experimental, simulated, hardmask)
    index = DENSITY_CHANNEL_NAMES.index("exp_clipnorm_nopost")
    single = build_experimental_density_channel(experimental)
    np.testing.assert_allclose(single[0], complete[index], rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["strongest", "density_blob_only", "pocket_only", "trunk_only"])
def test_four_model_variants_forward_and_direct_identity_loss(variant: str) -> None:
    batch = collate_candidates([_item(variant)])
    model = ClosedSetMatcher(_config(variant)).eval()
    output = model(batch)
    assert output["foreground_logit"].shape == (1,)
    assert output["identity_logits"].shape == (1, 1)
    assert torch.isfinite(output["foreground_logit"]).all()
    weights = LossWeights(1.0, 0.0 if variant == "pocket_only" else 0.1, 0.0 if variant == "pocket_only" else 0.1, 1.0)
    loss = compute_matcher_loss(output, batch, weights)
    assert torch.isfinite(loss["total"])
    assert int(loss["identity_eligible_count"]) == 1


def test_trunk_only_parameter_tree_has_no_identity_increment_modules() -> None:
    names = list(dict(ClosedSetMatcher(_config("trunk_only")).named_parameters()))
    assert not any("graph_encoder" in name for name in names)
    assert not any("graph_attention" in name for name in names)
    assert not any("inventory_attention" in name for name in names)
    assert not any("graph_gate" in name for name in names)
    assert not any("inventory_gate" in name for name in names)


def test_four_variants_share_identical_initial_values_for_common_parameters() -> None:
    """
    同一实验种子下, 四变体的同名同形参数必须逐值相同.
    """
    models: dict[str, dict[str, torch.Tensor]] = {}
    for variant in ("strongest", "density_blob_only", "pocket_only", "trunk_only"):
        torch.manual_seed(3409)
        rng_state = torch.random.get_rng_state()
        model = ClosedSetMatcher(_config(variant))
        assert torch.equal(torch.random.get_rng_state(), rng_state)
        models[variant] = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
    reference = models["strongest"]
    for variant, parameters in models.items():
        reference_modules = {
            name.rsplit(".", 1)[0]: {
                parameter_name.rsplit(".", 1)[1]: tuple(parameter.shape)
                for parameter_name, parameter in reference.items()
                if parameter_name.rsplit(".", 1)[0] == name.rsplit(".", 1)[0]
            }
            for name in reference
        }
        variant_modules = {
            name.rsplit(".", 1)[0]: {
                parameter_name.rsplit(".", 1)[1]: tuple(parameter.shape)
                for parameter_name, parameter in parameters.items()
                if parameter_name.rsplit(".", 1)[0] == name.rsplit(".", 1)[0]
            }
            for name in parameters
        }
        common = {
            name
            for name, value in reference.items()
            if name in parameters
            and parameters[name].shape == value.shape
            and reference_modules[name.rsplit(".", 1)[0]]
            == variant_modules[name.rsplit(".", 1)[0]]
        }
        assert common
        for name in common:
            torch.testing.assert_close(reference[name], parameters[name], rtol=0, atol=0)
    strongest = models["strongest"]
    for name in (
        "candidate_encoder.aggregator.conditioned.branches.pp.output.weight",
        "candidate_encoder.aggregator.conditioned.branches.pp.output.bias",
        "candidate_encoder.aggregator.residual.4.weight",
        "candidate_encoder.aggregator.residual.4.bias",
    ):
        assert torch.count_nonzero(strongest[name]) == 0


def test_training_manifest_snapshot_keeps_original_generation_after_entry_moves(
    tmp_path: Path,
) -> None:
    old_candidates = tmp_path / "generation_old_candidates.jsonl"
    old_identities = tmp_path / "generation_old_identities.jsonl"
    new_candidates = tmp_path / "generation_new_candidates.jsonl"
    new_identities = tmp_path / "generation_new_identities.jsonl"
    for path, text in (
        (old_candidates, "old candidates\n"),
        (old_identities, "old identities\n"),
        (new_candidates, "new candidates\n"),
        (new_identities, "new identities\n"),
    ):
        path.write_text(text, encoding="utf-8")
    manifest = {
        "schema": "matcher_stage1_manifest_v1",
        "class_names": [
            "background",
            "ion",
            "nucleotide_like",
            "peptide_like",
            "small_molecule",
            "sugar",
        ],
        "identity_key": "PreparedLigand.smiles",
        "language_model": "SMI-TED Light 289M",
        "data_root": str(tmp_path),
        "stage1_root": str(tmp_path),
        "graph_asset_path": str(tmp_path / "graph.npz"),
        "splits": {
            "train": {
                "candidates": old_candidates.name,
                "identities": old_identities.name,
            }
        },
        "summary": {"train": {}},
    }
    manifest_path = tmp_path / "manifest.json"
    snapshot_path = tmp_path / "run" / "manifest.snapshot.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _write_manifest_snapshot(manifest_path, snapshot_path)
    manifest["splits"]["train"] = {
        "candidates": new_candidates.name,
        "identities": new_identities.name,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert Path(snapshot["splits"]["train"]["candidates"]).read_text(encoding="utf-8") == "old candidates\n"
    assert Path(snapshot["splits"]["train"]["identities"]).read_text(encoding="utf-8") == "old identities\n"


@pytest.mark.parametrize(
    ("variant", "disabled_fields"),
    [
        ("strongest", ()),
        ("density_blob_only", ("a_tokens", "p_tokens", "pp_tokens")),
        (
            "pocket_only",
            (
                "voxel_input",
                "p_tokens",
                "v_tokens",
                "pp_tokens",
                "candidate_manual_features",
                "candidate_sampling_features",
            ),
        ),
        (
            "trunk_only",
            (
                "graph_element",
                "graph_charge",
                "graph_atom_in_ring",
                "graph_edge_features",
            ),
        ),
    ],
)
def test_disabled_inputs_do_not_change_model_output(
    variant: str,
    disabled_fields: tuple[str, ...],
) -> None:
    """
    扰动监督字段和当前变体禁用的模态，不得改变任何模型输出。
    """
    batch = collate_candidates([_item(variant)])
    changed = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    changed["foreground_target"] = ~changed["foreground_target"]
    changed["identity_target"] = torch.full_like(changed["identity_target"], -1)
    changed["ligand_area_target"] = ~changed["ligand_area_target"]
    changed["binding_area_target"] = ~changed["binding_area_target"]
    for field in disabled_fields:
        if field == "voxel_input":
            changed[field] = torch.full((1, 2, 48, 48, 48), 7.0)
        elif field in ("a_tokens", "p_tokens", "v_tokens", "pp_tokens"):
            width = {"a_tokens": 230, "p_tokens": 116, "v_tokens": 68, "pp_tokens": 4}[field]
            changed[field] = torch.full((1, 2, width), 7.0)
            prefix = field.removesuffix("_tokens")
            changed[f"{prefix}_mask"] = torch.ones((1, 2), dtype=torch.bool)
            if prefix in ("v", "pp"):
                changed[f"{prefix}_index_zyx"] = torch.zeros((1, 2, 3), dtype=torch.int64)
        elif field.startswith("graph_"):
            changed["graph_element"] = torch.asarray([6, 8], dtype=torch.int64)
            changed["graph_charge"] = torch.asarray([0, -1], dtype=torch.int64)
            changed["graph_atom_in_ring"] = torch.ones((2, 4))
            changed["graph_node_graph_index"] = torch.zeros(2, dtype=torch.int64)
            changed["graph_edge_index"] = torch.asarray([[0, 1], [1, 0]], dtype=torch.int64)
            changed["graph_edge_features"] = torch.ones((2, 9))
            changed["graph_to_sample"] = torch.zeros(1, dtype=torch.int64)
            changed["graph_to_identity"] = torch.zeros(1, dtype=torch.int64)
        else:
            changed[field] = torch.full_like(changed[field], 7)
    model = ClosedSetMatcher(_config(variant)).eval()
    with torch.no_grad():
        reference = model(batch)
        perturbed = model(changed)
    assert set(reference) == set(perturbed)
    for field in reference:
        if reference[field] is None:
            assert perturbed[field] is None
        else:
            torch.testing.assert_close(reference[field], perturbed[field], rtol=0, atol=0)


def test_smiles_graph_store_opens_package_only_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    同一个工作进程内重复初始化和取图只允许打开一次图包。
    """
    path = tmp_path / "smiles_graphs_v1.npz"
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        smiles=np.asarray(["CC"]),
        atom_offsets=np.asarray([0, 2], dtype=np.int64),
        element=np.asarray([6, 6], dtype=np.int16),
        charge=np.asarray([0, 0], dtype=np.int8),
        atom_in_ring=np.zeros((2, 4), dtype=np.bool_),
        bond_offsets=np.asarray([0, 1], dtype=np.int64),
        bond_index=np.asarray([[0], [1]], dtype=np.int32),
        bond_type=np.asarray([0], dtype=np.uint8),
        bond_in_ring=np.zeros((1, 4), dtype=np.bool_),
    )
    original_load = np.load
    open_count = 0

    def counted_load(*args: object, **kwargs: object) -> object:
        nonlocal open_count
        open_count += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counted_load)
    store = SmilesGraphStore(path)
    store.initialize()
    store.initialize()
    assert store.get("CC").element.tolist() == [6, 6]
    assert "CC" in store
    assert open_count == 1


def test_manifest_dataset_skips_only_unsupported_foreground_outside_test(tmp_path: Path) -> None:
    """
    训练和 validation 跳过不受支持真前景，test 保留且背景始终保留。
    """
    candidates = [
        {
            "split": "validation",
            "pdb_id": "1abc",
            "centered_box_index": index,
            "source_blob_index": index,
            "candidate_selected": True,
            "gaussian_score": 1.0 - index / 10,
            "source_probability_mean": 0.5,
            "foreground_target": foreground,
            "target_category": "small_molecule" if foreground else None,
            "matched_occurrence_id": index if foreground else None,
            "target_smiles": smiles,
            "candidate_coverage": 0.5 if foreground else 0.0,
            "occurrence_coverage": 0.5 if foreground else 0.0,
            "overlap_score": 0.5 if foreground else 0.0,
        }
        for index, (foreground, smiles) in enumerate(
            ((True, "CC"), (True, "unsupported"), (False, None))
        )
    ]
    identities = [
        {
            "pdb_id": "1abc",
            "smiles": "CC",
            "language_relative_path": "language/1abc/candidate_0.npz",
        }
    ]
    (tmp_path / "validation_candidates.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in candidates),
        encoding="utf-8",
    )
    (tmp_path / "validation_identities.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in identities),
        encoding="utf-8",
    )
    manifest = {
        "schema": "matcher_stage1_manifest_v1",
        "class_names": [
            "background",
            "ion",
            "nucleotide_like",
            "peptide_like",
            "small_molecule",
            "sugar",
        ],
        "identity_key": "PreparedLigand.smiles",
        "language_model": "SMI-TED Light 289M",
        "data_root": str(tmp_path),
        "stage1_root": str(tmp_path),
        "graph_asset_path": str(tmp_path / "smiles_graphs_v1.npz"),
        "splits": {
            "validation": {
                "candidates": "validation_candidates.jsonl",
                "identities": "validation_identities.jsonl",
            }
        },
        "summary": {},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    common = {
        "split": "validation",
        "selected_only": True,
        "variant": "trunk_only",
        "pp_top_k": 1,
        "cache_gib_per_worker": 0.0,
    }
    validation = load_manifest_dataset(manifest_path, mode="validation", **common)
    test = load_manifest_dataset(manifest_path, mode="test", **common)
    assert len(validation) == 2
    assert len(test) == 3


def test_identity_accumulation_weight_matches_one_global_batch() -> None:
    """
    物理批次身份数不等时，按合格数加权必须等于一次 global-batch CE。
    """
    first_logits = torch.asarray([[2.0, 0.0]])
    second_logits = torch.asarray([[0.0, 1.0], [1.0, 0.0]])
    first_loss, first_count = smiles_identity_cross_entropy(
        first_logits,
        torch.ones_like(first_logits, dtype=torch.bool),
        torch.asarray([0]),
        torch.asarray([True]),
    )
    second_loss, second_count = smiles_identity_cross_entropy(
        second_logits,
        torch.ones_like(second_logits, dtype=torch.bool),
        torch.asarray([1, 0]),
        torch.asarray([True, True]),
    )
    merged_loss, _ = smiles_identity_cross_entropy(
        torch.cat((first_logits, second_logits)),
        torch.ones((3, 2), dtype=torch.bool),
        torch.asarray([0, 1, 0]),
        torch.asarray([True, True, True]),
    )
    accumulated = (
        first_loss * first_count / (first_count + second_count)
        + second_loss * second_count / (first_count + second_count)
    )
    torch.testing.assert_close(accumulated, merged_loss, rtol=0, atol=1e-7)


def test_identity_loss_does_not_depend_on_foreground_prediction() -> None:
    batch = collate_candidates([_item("trunk_only")])
    output = ClosedSetMatcher(_config("trunk_only")).eval()(batch)
    first = compute_matcher_loss(output, batch, LossWeights(0.0, 0.0, 0.0, 1.0))["identity"]
    changed = dict(output)
    changed["foreground_logit"] = output["foreground_logit"] + 100
    second = compute_matcher_loss(changed, batch, LossWeights(0.0, 0.0, 0.0, 1.0))["identity"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_metrics_use_tied_average_precision_and_highest_best_threshold() -> None:
    target = np.asarray([True, False, True, False])
    probability = np.asarray([0.8, 0.8, 0.2, 0.1])
    assert foreground_average_precision(target, probability) == pytest.approx(7 / 12)
    threshold, metrics = fit_foreground_threshold(target, probability)
    assert threshold == pytest.approx(0.2)
    assert metrics["f1"] == pytest.approx(0.8)


def test_smiles_metrics_return_none_for_zero_support() -> None:
    metrics = smiles_match_metrics(
        ["CC"], ["CC"], ["small_molecule"], np.asarray([True]), np.asarray([True])
    )
    assert metrics["overall"]["accuracy"] == 1.0
    assert metrics["nucleotide_like"] == {
        "correct_count": 0,
        "eligible_count": 0,
        "accuracy": None,
    }


def test_stage1_hit_and_ranking_append_unselected_after_selected_prefix() -> None:
    intersections = np.asarray([[3, 0], [0, 2], [3, 0]])
    pred_sizes = np.asarray([10, 10, 10])
    gt_sizes = np.asarray([10, 10])
    hit = _candidate_hit_mask(intersections, pred_sizes, gt_sizes)
    curve = _ranked_hit_curve(
        np.asarray([0.5, 0.9, 0.5]),
        np.asarray([2, 1, 0]),
        np.asarray([True, False, True]),
        hit,
    )
    assert curve[:2] == [True, True]
    assert all(curve)


def test_stage1_unselected_candidate_cannot_displace_selected_prefix() -> None:
    curve = _ranked_hit_curve(
        np.asarray([0.1, 0.2, 0.99]),
        np.asarray([2, 1, 0]),
        np.asarray([True, True, False]),
        np.asarray([False, False, True]),
    )
    assert curve[:3] == [False, False, True]


def test_matcher_curve_uses_the_same_selected_priority_order(tmp_path: Path) -> None:
    pdb_ids = [f"pdb{index:03d}" for index in range(179)]
    rows = {
        pdb_ids[0]: [
            {
                "candidate_selected": True,
                "gaussian_score": 0.2,
                "source_blob_index": 2,
                "foreground_target": False,
                "identity_supported": False,
                "predicted_smiles": None,
                "target_smiles": None,
            },
            {
                "candidate_selected": True,
                "gaussian_score": 0.1,
                "source_blob_index": 1,
                "foreground_target": False,
                "identity_supported": False,
                "predicted_smiles": None,
                "target_smiles": None,
            },
            {
                "candidate_selected": False,
                "gaussian_score": 0.99,
                "source_blob_index": 0,
                "foreground_target": True,
                "identity_supported": True,
                "predicted_smiles": "CC",
                "target_smiles": "CC",
            },
        ]
    }
    curve = _matcher_curve("test", rows, pdb_ids, "gaussian_score", tmp_path / "best.pt")
    assert curve["hit_count"][:3] == [0, 0, 1]


def test_final_metrics_and_topk_keep_empty_pdb_in_fixed_denominator(tmp_path: Path) -> None:
    rows = {
        "nonempty": [
            {
                "foreground_target": True,
                "candidate_selected": True,
                "foreground_probability": 0.9,
                "predicted_smiles": "CC",
                "target_smiles": "CC",
                "target_category": "small_molecule",
                "identity_supported": True,
                "centered_box_index": 0,
                "source_blob_index": 0,
                "matched_occurrence_id": 0,
            },
            {
                "foreground_target": False,
                "candidate_selected": True,
                "foreground_probability": 0.1,
                "predicted_smiles": None,
                "target_smiles": None,
                "target_category": None,
                "identity_supported": False,
                "centered_box_index": 1,
                "source_blob_index": 1,
                "matched_occurrence_id": None,
            },
        ]
    }
    metrics = _metrics_from_rows(rows, 0.5, {"nonempty", "empty"})
    assert metrics["pdb_count"] == 2
    assert "selection" not in metrics
    pdb_ids = [f"pdb{index:03d}" for index in range(179)]
    curve = _matcher_curve("test", {}, pdb_ids, "gaussian_score", tmp_path / "best.pt")
    assert curve["pdb_count"] == 179
    assert curve["hit_count"] == [0] * 20


def test_four_formal_yaml_files_differ_only_by_registered_variant_and_area_boundary() -> None:
    root = Path("configs/experiments/final_smiles")
    configs = {path.stem: load_smiles_experiment(path) for path in sorted(root.glob("*.yaml"))}
    assert set(configs) == {"strongest", "density_blob_only", "pocket_only", "trunk_only"}
    assert {config.seed for config in configs.values()} == {3409}
    assert {config.data.selected_only for config in configs.values()} == {True}
    for name, config in configs.items():
        assert config.model.variant == name
        assert config.optimization.gradient_clip_norm == 200.0
        assert config.data.batch_size == 32
        assert config.data.accumulation_steps == 1
        assert config.optimization.validation_checks_per_epoch == 3
        assert config.optimization.selection_improvement_minimum == 0.001
        assert config.optimization.patience_bad_validation_checks == 2
        assert config.optimization.learning_rate_reduction_factor == 0.8
        assert config.optimization.max_learning_rate_reductions == 2
        assert config.optimization.max_epochs == 15
        if name == "pocket_only":
            assert config.loss_weights.ligand_area == config.loss_weights.binding_area == 0.0
        else:
            assert config.loss_weights.ligand_area == config.loss_weights.binding_area == 0.1


def test_optimizer_updates_are_partitioned_without_omission_or_repetition() -> None:
    assert _partition_optimizer_updates(10, 3) == (4, 3, 3)
    assert _partition_optimizer_updates(9, 3) == (3, 3, 3)
    with pytest.raises(ValueError, match="不小于"):
        _partition_optimizer_updates(2, 3)


def test_training_segments_consume_one_loader_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_smiles_experiment(Path("configs/experiments/final_smiles/trunk_only.yaml"))

    class CountingLoader:
        def __init__(self) -> None:
            self.iteration_count = 0

        def __len__(self) -> int:
            return 10

        def __iter__(self):
            self.iteration_count += 1
            for index in range(10):
                yield {
                    "feature": torch.asarray([[float(index)]]),
                    "foreground_target": torch.asarray([True]),
                    "identity_target": torch.asarray([0]),
                }

    class ScalarModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.asarray(1.0))

        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {"prediction": batch["feature"].mean() * self.weight}

    def fake_loss(
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        weights: LossWeights,
    ) -> dict[str, torch.Tensor]:
        del batch, weights
        value = outputs["prediction"].square() + 1.0
        return {
            "total": value,
            "foreground": value,
            "ligand_area": value,
            "binding_area": value,
            "identity": value,
            "identity_eligible_count": torch.asarray(1),
        }

    monkeypatch.setattr("matcher.training.engine.compute_matcher_loss", fake_loss)
    loader = CountingLoader()
    model = ScalarModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    segments = list(
        _train_epoch_segments(
            model,
            loader,
            optimizer,
            config,
            torch.device("cpu"),
            0,
            None,
        )
    )
    assert loader.iteration_count == 1
    assert [segment for segment, _, _ in segments] == [1, 2, 3]
    assert [step for _, _, step in segments] == [4, 7, 10]
    assert [summary["optimizer_update_count"] for _, summary, _ in segments] == [4.0, 3.0, 3.0]


def test_validation_policy_uses_one_threshold_and_stops_after_two_real_lr_reductions() -> None:
    config = load_smiles_experiment(Path("configs/experiments/final_smiles/strongest.yaml"))
    parameter = torch.nn.Parameter(torch.asarray(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=config.optimization.learning_rate)
    state = _ValidationPolicyState()

    first = state.observe(0.5000, 0, 1, optimizer, config)
    assert first["improved"] is True
    assert state.best_score == pytest.approx(0.5000)

    for validation_check, score in enumerate((0.5009, 0.5008), start=2):
        decision = state.observe(score, 0, validation_check, optimizer, config)
        assert decision["plateau"] is False
    first_plateau = state.observe(0.5007, 0, 4, optimizer, config)
    assert first_plateau["learning_rate_reduced"] is True
    assert state.learning_rate_reductions == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(8.0e-5)

    improvement = state.observe(0.5010, 1, 5, optimizer, config)
    assert improvement["improved"] is True
    assert state.bad_validation_checks == 0
    assert state.learning_rate_reductions == 1

    for validation_check in (6, 7):
        state.observe(0.5015, 1, validation_check, optimizer, config)
    second_plateau = state.observe(0.5014, 2, 8, optimizer, config)
    assert second_plateau["learning_rate_reduced"] is True
    assert state.learning_rate_reductions == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(6.4e-5)

    for validation_check in (9, 10):
        state.observe(0.5012, 2, validation_check, optimizer, config)
    final_plateau = state.observe(0.5011, 3, 11, optimizer, config)
    assert final_plateau == {
        "improved": False,
        "plateau": True,
        "learning_rate_reduced": False,
        "should_stop": True,
    }
    assert state.learning_rate_reductions == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(6.4e-5)
