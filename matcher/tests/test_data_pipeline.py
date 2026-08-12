"""验证新数据适配层的覆盖率、空间边界、缺失策略和批次字段。"""

from __future__ import annotations

import json

import numpy as np

from matcher.batching import PdbGroupedBatchSampler, collate_candidates
from matcher.candidate_data import CandidateDataset
from matcher.geometry import crop_center_48, hard_scatter_receptor, keep_local_48
from matcher.labels import OccurrenceMask, assign_candidate_target
from matcher.manifest import build_manifest, load_jsonl
from matcher.schema import SourceRoots

from .factories import build_synthetic_roots


def test_bidirectional_coverage_uses_score_then_occurrence_id() -> None:
    """覆盖阈值命中后按几何平均分，完全并列时选择较小 occurrence 编号。"""

    candidate = np.asarray([[1, 1, 1], [1, 1, 2]], dtype=np.int64)
    occurrences = [
        OccurrenceMask(4, "CCD:B", "ion", candidate.copy()),
        OccurrenceMask(2, "CCD:A", "small_molecule", candidate.copy()),
    ]
    target = assign_candidate_target(candidate, occurrences, coverage_threshold=0.30)
    assert target.matched_candidate_id == 2
    assert target.class_name == "small_molecule"
    assert target.candidate_coverage == 1.0
    assert target.occurrence_coverage == 1.0


def test_geometry_keeps_only_center_48_and_sums_receptor_features() -> None:
    """V 截断使用 [16,64)；同 home voxel 的受体特征执行求和。"""

    full = np.arange(80**3, dtype=np.int64).reshape(80, 80, 80)
    assert crop_center_48(full).shape == (48, 48, 48)
    keep, cropped = keep_local_48(np.asarray([[15, 16, 16], [16, 16, 16], [63, 63, 63], [64, 63, 63]]))
    assert keep.tolist() == [False, True, True, False]
    assert cropped.tolist() == [[0, 0, 0], [47, 47, 47]]
    grid, mask = hard_scatter_receptor(
        np.asarray([[20.2, 21.2, 22.2], [20.8, 21.8, 22.8]], dtype=np.float32),
        np.ones((2, 49), dtype=np.float32),
    )
    assert mask[22, 21, 20]
    assert np.all(grid[:, 22, 21, 20] == 2.0)


def test_manifest_and_dataset_keep_pdb_but_drop_missing_smi_occurrence(tmp_path) -> None:
    """SMI-TED 单 occurrence 缺失不排除 PDB，并改选同身份的有效表示。"""

    roots = build_synthetic_roots(tmp_path)
    manifest_path = build_manifest(roots, coverage_threshold=0.30, selection_seed=3407)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["split_pdb_counts"]["train"] == {
        "discovered": 1,
        "included": 1,
        "excluded": 0,
    }
    exclusions = load_jsonl(manifest_path.parent / "excluded_pdb.jsonl")
    assert {item["reason"] for item in exclusions} == {"blob_exceed", "missing_required_product"}
    identities = load_jsonl(manifest_path.parent / "train_identities.jsonl")
    smi = identities[0]["language"]["smi_ted_289m"]
    assert smi["selected_candidate_id"] == 1
    assert smi["valid_candidate_ids"] == [1]
    assert smi["failed_occurrences"] == [{"candidate_id": 0, "reason": "missing_file"}]

    dataset = CandidateDataset(
        roots=SourceRoots(roots.data_root, roots.stage1_root, roots.output_root),
        manifest_path=manifest_path,
        split_name="train",
        task_name="full_matcher",
        language_family="smi_ted_289m",
        pp_top_k=4,
    )
    item = dataset[0]
    assert item["class_target"] == 4
    assert item["full_candidate_voxel_count"] == 2
    assert item["v_tokens"].shape == (1, 52)
    assert item["p_tokens"].shape == (2, 116)
    assert item["a_tokens"].shape == (1, 229)
    assert item["pp_tokens"].shape == (4, 4)
    assert item["identity_object_keys"] == ["CCD:LIG"]
    assert item["identity_target"] == 0
    assert item["language_embeddings"].shape == (1, 768)
    assert item["ligand_area_target"][24, 24, 24]
    assert item["binding_area_target"][24, 24, 24]

    batch = collate_candidates([item])
    assert batch["voxel_input"].shape == (1, 105, 48, 48, 48)
    assert batch["identity_mask"].tolist() == [[True]]
    assert batch["graph_edge_index"].shape == (2, 2)
    sampler = PdbGroupedBatchSampler(dataset, batch_size=1, shuffle=True, drop_last=True, seed=7)
    assert list(sampler) == [[0]]
