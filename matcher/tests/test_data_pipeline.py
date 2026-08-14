"""验证新数据适配层的覆盖率、空间边界、缺失策略和批次字段。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from matcher.data.geometry import (
    candidate_sampling_features,
    crop_local_48,
    hard_scatter_receptor,
    keep_local_48_at,
    resolve_candidate_crop,
)
from matcher.data.labels import OccurrenceMask, assign_candidate_target
from matcher.data.ligand_graph import LigandGraphStore
from matcher.data.manifest import (
    _annotate_seen,
    _joint_identity_cross_pdb_class_variants,
    _validate_joint_identity_classes,
    build_manifest,
    ensure_manifest,
    load_jsonl,
    require_manifest,
)
from matcher.data.pp_input import _build_pp_tokens
from matcher.data.precompute import precompute_inputs
from matcher.data.schema import LocalCropStrategy, SourceRoots
from matcher.data.voxel_input import VoxelInputStore
from matcher.datasets.batching import ShuffledBatchSampler, collate_candidates
from matcher.datasets.candidate_data import CandidateDataset
from matcher.datasets.contracts import CandidateItem
from matcher.utils import locking
from matcher.utils.cache import ByteBudgetLru

from .factories import build_synthetic_roots


def _replace_prepared_smiles(
    roots: SourceRoots,
    values_by_candidate: dict[int, str | None],
) -> None:
    """替换合成 PDB 的顶层 PreparedLigand.smiles，不改动其它来源字段。"""

    path = roots.language_root / "prepared/1abc/prepared_smiles.jsonl"
    records = load_jsonl(path)
    for record in records:
        candidate_id = int(record["candidate_id"])
        if candidate_id in values_by_candidate:
            record["smiles"] = values_by_candidate[candidate_id]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _replace_language_prepared_smiles(
    roots: SourceRoots,
    family: str,
    candidate_id: int,
    prepared_smiles: str,
) -> None:
    """替换一个合成语言 NPZ 的 public API 输入字符串并保留向量。"""

    path = roots.language_root / family / "1abc" / f"candidate_{candidate_id}.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]).copy() for name in archive.files}
    values["prepared_smiles"] = np.asarray(prepared_smiles)
    np.savez(path, **values)


def test_read_only_manifest_entry_never_builds_missing_files(tmp_path: Path) -> None:
    """保留评估要求既有清单，缺失时不得产生目录或清单副作用。"""

    roots = build_synthetic_roots(tmp_path)
    assert not roots.manifest_root.exists()
    with pytest.raises(FileNotFoundError, match="禁止在只读评估入口重建"):
        require_manifest(
            roots,
            coverage_threshold=0.30,
            selection_seed=3407,
            required_splits=("calibration",),
        )
    assert not roots.manifest_root.exists()


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "expected_error"),
    (
        ("schema_version", 3, "schema_version"),
        ("identity_key_fields", ["object_key"], "联合身份字段顺序"),
    ),
)
def test_dataset_requires_explicit_v4_joint_identity_contract(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
    expected_error: str,
) -> None:
    """Dataset 不得把 v3 或字段不完整的清单静默解释为联合身份。"""

    roots = build_synthetic_roots(tmp_path)
    manifest_path = build_manifest(roots, 0.30, 3407, ("train", "validation"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field_name] = invalid_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=expected_error):
        CandidateDataset(
            roots=roots,
            manifest_path=manifest_path,
            split_name="train",
            task_name="candidate_pretrain",
            language_family=None,
            identity_scope=None,
            pp_top_k=4,
            local_crop_strategy="fixed_center",
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "expected_error"),
    (
        ("prepared_smiles", None, "prepared_smiles 必须是非空字符串"),
        ("prepared_smiles", 7, "prepared_smiles 必须是非空字符串"),
        ("object_key", 7, "object_key 必须是非空字符串"),
        ("class_index", 5, "class_index 与 class_name 不一致"),
    ),
)
def test_dataset_rejects_invalid_joint_identity_fields(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
    expected_error: str,
) -> None:
    """身份 JSONL 的联合键与类别字段不得通过 str/int 强制转换迁移。"""

    roots = build_synthetic_roots(tmp_path)
    manifest_path = build_manifest(roots, 0.30, 3407, ("train", "validation"))
    identity_path = manifest_path.parent / "train_identities.jsonl"
    identities = load_jsonl(identity_path)
    identities[0][field_name] = invalid_value
    identity_path.write_text(
        "".join(json.dumps(identity) + "\n" for identity in identities),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=expected_error):
        CandidateDataset(
            roots=roots,
            manifest_path=manifest_path,
            split_name="train",
            task_name="candidate_pretrain",
            language_family=None,
            identity_scope=None,
            pp_top_k=4,
            local_crop_strategy="fixed_center",
        )


def test_dataset_rejects_duplicate_joint_identity_records(tmp_path: Path) -> None:
    """同一 PDB 的重复联合身份不得形成两个清单槽。"""

    roots = build_synthetic_roots(tmp_path)
    manifest_path = build_manifest(roots, 0.30, 3407, ("train", "validation"))
    identity_path = manifest_path.parent / "train_identities.jsonl"
    identities = load_jsonl(identity_path)
    identity_path.write_text(
        "".join(json.dumps(identity) + "\n" for identity in identities + identities),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="既有联合身份重复"):
        CandidateDataset(
            roots=roots,
            manifest_path=manifest_path,
            split_name="train",
            task_name="candidate_pretrain",
            language_family=None,
            identity_scope=None,
            pp_top_k=4,
            local_crop_strategy="fixed_center",
        )


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
    """V 截断使用 [16,64)；向下取整到同一体素的受体特征执行求和。"""

    full = np.arange(80**3, dtype=np.int64).reshape(80, 80, 80)
    fixed_start = np.asarray([16, 16, 16], dtype=np.int64)
    assert crop_local_48(full, fixed_start).shape == (48, 48, 48)
    keep, cropped = keep_local_48_at(
        np.asarray([[15, 16, 16], [16, 16, 16], [63, 63, 63], [64, 63, 63]]),
        fixed_start,
    )
    assert keep.tolist() == [False, True, True, False]
    assert cropped.tolist() == [[0, 0, 0], [47, 47, 47]]
    grid, mask = hard_scatter_receptor(
        np.asarray([[20.2, 21.2, 22.2], [20.8, 21.8, 22.8]], dtype=np.float32),
        np.ones((2, 49), dtype=np.float32),
    )
    assert mask[22, 21, 20]
    assert np.all(grid[:, 22, 21, 20] == 2.0)


@pytest.mark.parametrize(
    ("axis", "edge", "expected_start", "expected_offset"),
    (
        (0, "low", 0, 23.0),
        (0, "high", 32, -23.0),
        (1, "low", 0, 23.0),
        (1, "high", 32, -23.0),
        (2, "low", 0, 23.0),
        (2, "high", 32, -23.0),
    ),
)
def test_v_centroid_crop_resolves_interior_and_six_boundary_faces(
    axis: int,
    edge: str,
    expected_start: int,
    expected_offset: float,
) -> None:
    """内部 V 保持中心视野；六面请求逐轴限制到 0/32，且保留完整 V。"""

    interior = np.asarray([[39, 39, 39], [40, 40, 40]], dtype=np.int64)
    interior_geometry = resolve_candidate_crop(interior, "v_centroid")
    np.testing.assert_array_equal(interior_geometry.start_zyx, [16, 16, 16])
    np.testing.assert_array_equal(interior_geometry.crop_center_offset_zyx, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        candidate_sampling_features(interior_geometry),
        [0.0, 0.0, 0.0, 1.0],
    )

    boundary = interior.copy()
    boundary[:, axis] = [0, 1] if edge == "low" else [78, 79]
    geometry = resolve_candidate_crop(boundary, "v_centroid")
    expected = np.full(3, 16, dtype=np.int64)
    expected[axis] = expected_start
    np.testing.assert_array_equal(geometry.start_zyx, expected)
    assert geometry.clipped_axis_mask.tolist() == [index == axis for index in range(3)]
    assert geometry.crop_center_offset_zyx[axis] == expected_offset
    assert geometry.retained_v_count == geometry.full_v_count == 2
    features = candidate_sampling_features(geometry)
    assert features[0] == 1.0
    assert features[1] == pytest.approx(abs(expected_offset))
    assert features[2] == pytest.approx(abs(expected_offset))
    assert features[3] == 1.0


def test_v_centroid_crop_uses_ties_to_even_and_reports_partial_retention() -> None:
    """半整数请求沿用 NumPy 最近偶数取整，宽组件的保留比例按完整 V 计数。"""

    low_half = resolve_candidate_crop(
        np.asarray([[24, 39, 39]], dtype=np.int64),
        "v_centroid",
    )
    high_half = resolve_candidate_crop(
        np.asarray([[25, 39, 39]], dtype=np.int64),
        "v_centroid",
    )
    assert low_half.requested_start_zyx[0] == 0
    assert high_half.requested_start_zyx[0] == 2

    wide_component = np.stack(
        (
            np.arange(80, dtype=np.int64),
            np.full(80, 39, dtype=np.int64),
            np.full(80, 40, dtype=np.int64),
        ),
        axis=1,
    )
    geometry = resolve_candidate_crop(wide_component, "v_centroid")
    assert geometry.start_zyx.tolist() == [16, 16, 16]
    assert geometry.full_v_count == 80
    assert geometry.retained_v_count == 48
    assert candidate_sampling_features(geometry)[3] == pytest.approx(0.6)


def test_v_centroid_crop_drives_voxel_tokens_pp_and_area_targets_together(tmp_path) -> None:
    """高 X 边界候选的体素、V、PP、配体和结合区域严格共享一个起点。"""

    roots = build_synthetic_roots(tmp_path)
    centered_path = roots.stage1_root / "train/1abc/centered/F1_centered.npz"
    with np.load(centered_path, allow_pickle=False) as archive:
        centered = {name: np.asarray(archive[name]).copy() for name in archive.files}
    full_v = np.asarray([[39, 39, 78], [40, 40, 79]], dtype=np.int16)
    marker_zyx = np.asarray([40, 40, 79], dtype=np.int64)
    centered["voxel_index_local_zyx"] = full_v
    centered["A_coord_local_xyz"] = (marker_zyx[[2, 1, 0]] + 0.5)[None].astype(np.float32)
    centered["P_coord_local_xyz"] = np.stack(
        (
            (marker_zyx[[2, 1, 0]] + 0.5).astype(np.float32),
            np.asarray([40.5, 40.5, 40.5], dtype=np.float32),
        )
    )
    np.savez(centered_path, **centered)

    receptor_path = roots.data_root / "parse/1abc/receptor_tokens.npz"
    np.savez(
        receptor_path,
        coords=(marker_zyx[[2, 1, 0]] + 0.5)[None].astype(np.float32),
        feat=np.ones((1, 49), dtype=np.float32),
    )
    np.savez(
        roots.data_root / "labels/1abc/atom_labels.npz",
        binding_atom=np.asarray([True]),
    )
    np.savez(
        roots.data_root / "density/1abc/ligand_area.npz",
        mask_0=marker_zyx[None].astype(np.int32),
        mask_1=marker_zyx[None].astype(np.int32),
    )
    probability = np.zeros((80, 80, 80), dtype=np.float32)
    probability[tuple(marker_zyx)] = 1.0
    np.savez(
        roots.stage1_root / "train/1abc/probability/probability_map.npz",
        probability_map=probability,
        origin_xyz=np.zeros(3, dtype=np.float32),
        voxel_size_xyz=np.ones(3, dtype=np.float32),
    )

    manifest_path = build_manifest(
        roots,
        coverage_threshold=0.30,
        selection_seed=3407,
        split_names=("train", "validation"),
    )
    item = CandidateDataset(
        roots=roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name="candidate_pretrain",
        language_family=None,
        identity_scope=None,
        pp_top_k=4,
        local_crop_strategy="v_centroid",
        cache_bytes_per_worker=64 * 1024**2,
    )[0]
    marker_local = (marker_zyx - np.asarray([16, 16, 32])).tolist()
    assert marker_local == [24, 24, 47]
    assert marker_local in item["v_index_zyx"].tolist()
    assert item["pp_index_zyx"][0].tolist() == marker_local
    assert item["ligand_area_target"][tuple(marker_local)]
    assert item["binding_area_target"][tuple(marker_local)]
    expected_exp = np.load(
        roots.data_root / "density/1abc/exp.npz",
        allow_pickle=False,
    )["grid"][0][tuple(marker_zyx)]
    assert item["voxel_input"][0][tuple(marker_local)] == np.float16(expected_exp)
    assert item["candidate_sampling_features"].tolist() == pytest.approx(
        [1.0, 23.0, 23.0, 1.0]
    )
    assert item["a_tokens"][0, -3:].tolist() == pytest.approx(
        [23.5 / 40.0, 0.5 / 40.0, 0.5 / 40.0]
    )
    assert item["p_tokens"][0, -3:].tolist() == pytest.approx(
        [23.5 / 40.0, 0.5 / 40.0, 0.5 / 40.0]
    )


def test_manifest_and_dataset_keep_pdb_but_drop_missing_smi_occurrence(tmp_path) -> None:
    """SMI-TED 单 occurrence 缺失不排除 PDB，并改选同身份的有效表示。"""

    roots = build_synthetic_roots(tmp_path)
    manifest_path = build_manifest(
        roots,
        coverage_threshold=0.30,
        selection_seed=3407,
        split_names=("train", "validation"),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["split_pdb_counts"]["train"] == {
        "discovered": 1,
        "included": 1,
        "excluded": 0,
    }
    exclusions = load_jsonl(manifest_path.parent / "excluded_pdb.jsonl")
    assert {item["reason"] for item in exclusions} == {"blob_exceed"}
    assert manifest["available_splits"] == ["train", "validation"]
    assert manifest["deferred_splits"] == ["calibration"]
    assert manifest["cross_pdb_joint_identity_class_variants"] == []
    identities = load_jsonl(manifest_path.parent / "train_identities.jsonl")
    candidates = load_jsonl(manifest_path.parent / "train_candidates.jsonl")
    smi = identities[0]["language"]["smi_ted_289m"]
    assert identities[0]["prepared_smiles"] == "CC"
    assert identities[0]["language"]["molformer"]["selected_candidate_id"] == 0
    assert smi["selected_candidate_id"] == 1
    assert smi["valid_candidate_ids"] == [1]
    assert smi["failed_occurrences"] == [{"candidate_id": 0, "reason": "missing_file"}]
    assert identities[0]["seen_in_train_by_language"] == {
        "molformer": True,
        "smi_ted_289m": True,
    }
    assert candidates[0]["identity_seen_in_train_by_language"] == {
        "molformer": True,
        "smi_ted_289m": True,
    }
    assert candidates[0]["matched_prepared_smiles"] == "CC"
    assert candidates[0]["matched_prepared_smiles_status"] == "available"

    dataset = CandidateDataset(
        roots=SourceRoots(roots.data_root, roots.stage1_root, roots.output_root),
        manifest_path=manifest_path,
        split_name="train",
        task_name="full_matcher",
        language_family="smi_ted_289m",
        identity_scope="small_molecule",
        pp_top_k=4,
        local_crop_strategy="fixed_center",
    )
    item = dataset[0]
    assert item["class_target"] == 4
    assert item["full_candidate_voxel_count"] == 2
    assert item["candidate_manual_features"].tolist() == pytest.approx(
        [1.0, 0.8, 0.8, 2.0, 0.5, 0.9, 1.0, 0.9, 0.9, 4.0, 0.25, 1.0]
    )
    assert item["v_tokens"].shape == (1, 52)
    assert item["p_tokens"].shape == (2, 116)
    assert item["a_tokens"].shape == (1, 229)
    assert item["pp_tokens"].shape == (4, 4)
    assert item["identity_object_keys"] == ["CCD:LIG"]
    assert item["identity_prepared_smiles"] == ["CC"]
    assert item["identity_target"] == 0
    assert item["metadata"]["matched_prepared_smiles"] == "CC"
    assert item["metadata"]["identity_seen_in_train"] is True
    assert item["language_embeddings"].shape == (1, 768)
    assert item["ligand_area_target"][24, 24, 24]
    assert item["binding_area_target"][24, 24, 24]
    assert item["binding_area_target"][24, 24, 25]
    assert int(item["binding_area_target"].sum()) == 2

    batch = collate_candidates([cast(CandidateItem, item)])
    assert batch["voxel_input"].shape == (1, 105, 48, 48, 48)
    assert batch["identity_mask"].tolist() == [[True]]
    assert batch["identity_class_index"].tolist() == [[4]]
    assert batch["identity_prepared_smiles"] == [["CC"]]
    assert batch["graph_edge_index"].shape == (2, 2)
    assert batch["graph_edge_features"][:, -1].tolist() == pytest.approx([0.15, 0.15])


def test_joint_identity_splits_same_object_key_by_exact_prepared_smiles(tmp_path) -> None:
    """同一模板的两个顶层 SMILES 必须成为两个身份槽并读取各自 occurrence 向量。"""

    roots = build_synthetic_roots(tmp_path)
    _replace_prepared_smiles(roots, {0: "CC", 1: "C=C"})
    _replace_language_prepared_smiles(roots, "molformer", 1, "C=C")
    _replace_language_prepared_smiles(roots, "smi_ted_289m", 1, "C=C")
    manifest_path = build_manifest(roots, 0.30, 3407, ("train", "validation"))

    identities = load_jsonl(manifest_path.parent / "train_identities.jsonl")
    assert [(item["object_key"], item["prepared_smiles"]) for item in identities] == [
        ("CCD:LIG", "C=C"),
        ("CCD:LIG", "CC"),
    ]
    item = CandidateDataset(
        roots=roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name="full_matcher",
        language_family="molformer",
        identity_scope="small_molecule",
        pp_top_k=4,
        local_crop_strategy="fixed_center",
    )[0]
    assert item["identity_object_keys"] == ["CCD:LIG", "CCD:LIG"]
    assert item["identity_prepared_smiles"] == ["C=C", "CC"]
    assert item["identity_target"] == 1
    np.testing.assert_array_equal(item["language_embeddings"][:, 0], [2.0, 1.0])
    assert len(item["ligand_graphs"]) == 2
    np.testing.assert_array_equal(
        item["ligand_graphs"][0].node_continuous,
        item["ligand_graphs"][1].node_continuous,
    )
    batch = collate_candidates([cast(CandidateItem, item)])
    assert batch["graph_to_sample"].tolist() == [0, 0]
    assert batch["graph_to_identity"].tolist() == [0, 1]


def test_manifest_preserves_valid_unsupported_prepared_kind(tmp_path: Path) -> None:
    """非标准 kind 的合法 unsupported/null 记录必须保留候选并关闭该身份目标。"""

    roots = build_synthetic_roots(tmp_path)
    prepared_path = roots.language_root / "prepared/1abc/prepared_smiles.jsonl"
    records = load_jsonl(prepared_path)
    records[0]["kind"] = "OTHER_KIND"
    records[0]["smiles"] = None
    records[0]["smiles_source"] = "unsupported"
    prepared_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest_path = build_manifest(roots, 0.30, 3407, ("train", "validation"))
    candidate = load_jsonl(manifest_path.parent / "train_candidates.jsonl")[0]
    assert candidate["class_name"] == "small_molecule"
    assert candidate["matched_prepared_smiles_status"] == "explicit_null"
    assert candidate["language_target_available"] == {
        "molformer": False,
        "smi_ted_289m": False,
    }


def test_joint_identity_selector_keeps_v3_salt(tmp_path: Path) -> None:
    """未拆分身份必须沿用不含 PreparedLigand.smiles 的 v3 selector salt。"""

    roots = build_synthetic_roots(tmp_path)
    _replace_prepared_smiles(roots, {0: "CCC", 1: "CCC"})
    for candidate_id in (0, 1):
        _replace_language_prepared_smiles(roots, "molformer", candidate_id, "CCC")
    manifest_path = build_manifest(roots, 0.30, 3407, ("train", "validation"))
    identity = load_jsonl(manifest_path.parent / "train_identities.jsonl")[0]
    selected_candidate_id = identity["language"]["molformer"]["selected_candidate_id"]

    def selector_index(text: str) -> int:
        weighted_bytes = sum(
            (position + 1) * byte
            for position, byte in enumerate(text.encode("utf-8"))
        )
        return (3407 + weighted_bytes) % 2

    v3_index = selector_index("1abc|CCD:LIG|molformer")
    wrong_v4_index = selector_index("1abc|CCD:LIG|CCC|molformer")
    assert v3_index == 0
    assert wrong_v4_index == 1
    assert selected_candidate_id == v3_index


def test_joint_seen_does_not_cross_prepared_smiles() -> None:
    """同一 object_key 的不同顶层字符串不得跨 train/validation 继承 seen。"""

    availability = {"molformer": True, "smi_ted_289m": True}
    split_candidates = {
        "train": [
            {
                "class_name": "small_molecule",
                "matched_object_key": "CCD:LIG",
                "matched_prepared_smiles": "CC",
                "language_target_available": availability,
            }
        ],
        "validation": [
            {
                "class_name": "small_molecule",
                "matched_object_key": "CCD:LIG",
                "matched_prepared_smiles": "C=C",
                "language_target_available": availability,
            },
            {
                "class_name": "small_molecule",
                "matched_object_key": "CCD:LIG",
                "matched_prepared_smiles": "CC",
                "language_target_available": availability,
            },
        ],
    }
    split_identities = {
        "train": [{"object_key": "CCD:LIG", "prepared_smiles": "CC"}],
        "validation": [
            {"object_key": "CCD:LIG", "prepared_smiles": "C=C"},
            {"object_key": "CCD:LIG", "prepared_smiles": "CC"},
        ],
    }
    _annotate_seen(split_candidates, split_identities)
    assert split_candidates["validation"][0]["identity_seen_in_train_by_language"] == {
        "molformer": False,
        "smi_ted_289m": False,
    }
    assert split_candidates["validation"][1]["identity_seen_in_train_by_language"] == {
        "molformer": True,
        "smi_ted_289m": True,
    }


def test_null_prepared_smiles_never_borrows_same_object_identity(tmp_path) -> None:
    """命中显式 null occurrence 时保留分类样本，但不得借用同模板的非空身份。"""

    roots = build_synthetic_roots(tmp_path)
    _replace_prepared_smiles(roots, {0: None})
    manifest_path = build_manifest(roots, 0.30, 3407, ("train", "validation"))
    candidate = load_jsonl(manifest_path.parent / "train_candidates.jsonl")[0]
    assert candidate["class_name"] == "small_molecule"
    assert candidate["matched_prepared_smiles"] is None
    assert candidate["matched_prepared_smiles_status"] == "explicit_null"
    assert candidate["language_target_available"] == {
        "molformer": False,
        "smi_ted_289m": False,
    }

    item = CandidateDataset(
        roots=roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name="full_matcher",
        language_family="molformer",
        identity_scope="small_molecule",
        pp_top_k=4,
        local_crop_strategy="fixed_center",
    )[0]
    assert item["class_target"] == 4
    assert item["identity_prepared_smiles"] == ["CC"]
    assert item["identity_target"] == -1


def test_all_null_prepared_smiles_produces_no_identity_slots(tmp_path) -> None:
    """PDB 内所有顶层 SMILES 均为 null 时仍保留候选并返回空身份轴。"""

    roots = build_synthetic_roots(tmp_path)
    _replace_prepared_smiles(roots, {0: None, 1: None})
    manifest_path = build_manifest(roots, 0.30, 3407, ("train", "validation"))
    assert load_jsonl(manifest_path.parent / "train_identities.jsonl") == []
    item = CandidateDataset(
        roots=roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name="full_matcher",
        language_family="molformer",
        identity_scope="small_molecule",
        pp_top_k=4,
        local_crop_strategy="fixed_center",
    )[0]
    assert item["class_target"] == 4
    assert item["identity_object_keys"] == []
    assert item["identity_prepared_smiles"] == []
    assert item["identity_target"] == -1


def test_missing_prepared_file_is_distinct_from_explicit_null(tmp_path) -> None:
    """缺失 PreparedLigand 文件时保留候选，并用独立状态关闭身份监督。"""

    roots = build_synthetic_roots(tmp_path)
    (roots.language_root / "prepared/1abc/prepared_smiles.jsonl").unlink()
    manifest_path = build_manifest(roots, 0.30, 3407, ("train", "validation"))
    candidate = load_jsonl(manifest_path.parent / "train_candidates.jsonl")[0]
    assert candidate["matched_prepared_smiles"] is None
    assert candidate["matched_prepared_smiles_status"] == "missing_file"
    assert load_jsonl(manifest_path.parent / "train_identities.jsonl") == []


def test_language_archive_must_match_joint_prepared_smiles(tmp_path) -> None:
    """语言 NPZ 的 prepared_smiles 不一致时不得跨联合身份借用该向量。"""

    roots = build_synthetic_roots(tmp_path)
    _replace_prepared_smiles(roots, {0: "C=C", 1: "CC"})
    manifest_path = build_manifest(roots, 0.30, 3407, ("train", "validation"))
    identities = load_jsonl(manifest_path.parent / "train_identities.jsonl")
    split_identity = next(item for item in identities if item["prepared_smiles"] == "C=C")
    assert split_identity["language"]["molformer"]["available"] is False
    assert split_identity["language"]["molformer"]["failed_occurrences"][0][
        "reason"
    ] == "invalid_archive"
    assert split_identity["language"]["smi_ted_289m"]["available"] is False
    assert split_identity["language"]["smi_ted_289m"]["failed_occurrences"][0][
        "reason"
    ] == "missing_file"
    candidate = load_jsonl(manifest_path.parent / "train_candidates.jsonl")[0]
    assert candidate["language_target_available"] == {
        "molformer": False,
        "smi_ted_289m": False,
    }


def test_manifest_rejects_non_string_language_prepared_smiles(tmp_path: Path) -> None:
    """语言 NPZ 的 prepared_smiles 非字符串时必须记为无效归档，不能字符串化。"""

    roots = build_synthetic_roots(tmp_path)
    path = roots.language_root / "molformer/1abc/candidate_0.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]).copy() for name in archive.files}
    values["prepared_smiles"] = np.asarray(7, dtype=np.int64)
    np.savez(path, **values)
    manifest_path = build_manifest(roots, 0.30, 3407, ("train", "validation"))
    identity = load_jsonl(manifest_path.parent / "train_identities.jsonl")[0]
    evidence = identity["language"]["molformer"]
    assert evidence["valid_candidate_ids"] == [1]
    assert evidence["failed_occurrences"][0]["candidate_id"] == 0
    assert evidence["failed_occurrences"][0]["reason"] == "invalid_archive"
    assert "Unicode 标量字符串" in evidence["failed_occurrences"][0]["detail"]


def test_dataset_rechecks_language_prepared_smiles_scalar_type(tmp_path: Path) -> None:
    """清单发布后归档类型漂移时，Dataset 二次读取必须失败关闭。"""

    roots = build_synthetic_roots(tmp_path)
    manifest_path = build_manifest(roots, 0.30, 3407, ("train", "validation"))
    path = roots.language_root / "molformer/1abc/candidate_0.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]).copy() for name in archive.files}
    values["prepared_smiles"] = np.asarray(7, dtype=np.int64)
    np.savez(path, **values)
    dataset = CandidateDataset(
        roots=roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name="full_matcher",
        language_family="molformer",
        identity_scope="small_molecule",
        pp_top_k=4,
        local_crop_strategy="fixed_center",
    )
    with pytest.raises(ValueError, match="prepared_smiles 必须是非空 Unicode 标量字符串"):
        dataset[0]


def test_joint_identity_allows_cross_pdb_class_variants_as_diagnostics() -> None:
    """不同 PDB 不共享身份轴，联合键的局部类别差异只形成诊断。"""

    records = {
        "train": [
            {
                "pdb_id": "1abc",
                "object_key": "CCD:LIG",
                "prepared_smiles": "CC",
                "class_name": "peptide_like",
                "class_index": 3,
                "occurrence_candidate_ids": [0, 1],
            }
        ],
        "validation": [
            {
                "pdb_id": "2def",
                "object_key": "CCD:LIG",
                "prepared_smiles": "CC",
                "class_name": "small_molecule",
                "class_index": 4,
                "occurrence_candidate_ids": [2],
            }
        ],
    }
    _validate_joint_identity_classes(records)
    assert _joint_identity_cross_pdb_class_variants(records) == [
        {
            "object_key": "CCD:LIG",
            "prepared_smiles": "CC",
            "classes": [
                {"class_name": "peptide_like", "class_index": 3},
                {"class_name": "small_molecule", "class_index": 4},
            ],
            "locations": [
                {
                    "split": "train",
                    "pdb_id": "1abc",
                    "class_name": "peptide_like",
                    "class_index": 3,
                    "occurrence_count": 2,
                },
                {
                    "split": "validation",
                    "pdb_id": "2def",
                    "class_name": "small_molecule",
                    "class_index": 4,
                    "occurrence_count": 1,
                },
            ],
        }
    ]


def test_joint_identity_rejects_class_drift_within_one_pdb() -> None:
    """同一 PDB 的一个联合身份不能同时占用多个六分类类别。"""

    with pytest.raises(ValueError, match="同一 PDB 内的联合身份类别不一致"):
        _validate_joint_identity_classes(
            {
                "train": [
                    {
                        "pdb_id": "1abc",
                        "object_key": "CCD:LIG",
                        "prepared_smiles": "CC",
                        "class_name": "small_molecule",
                        "class_index": 4,
                    },
                    {
                        "pdb_id": "1abc",
                        "object_key": "CCD:LIG",
                        "prepared_smiles": "CC",
                        "class_name": "sugar",
                        "class_index": 5,
                    },
                ]
            }
        )


def test_joint_identity_rejects_class_drift_for_the_same_pdb_across_splits() -> None:
    """split 不得成为同一物理 PDB 的联合身份类别豁免项。"""

    with pytest.raises(ValueError, match="同一 PDB 内的联合身份类别不一致"):
        _validate_joint_identity_classes(
            {
                "train": [
                    {
                        "pdb_id": "1abc",
                        "object_key": "CCD:LIG",
                        "prepared_smiles": "CC",
                        "class_name": "peptide_like",
                        "class_index": 3,
                    }
                ],
                "validation": [
                    {
                        "pdb_id": "1abc",
                        "object_key": "CCD:LIG",
                        "prepared_smiles": "CC",
                        "class_name": "small_molecule",
                        "class_index": 4,
                    }
                ],
            }
        )


def test_cross_pdb_diagnostic_ignores_a_single_pdb_across_splits() -> None:
    """诊断 helper 不得把同一 PDB 的跨划分漂移伪装成跨 PDB 变体。"""

    assert _joint_identity_cross_pdb_class_variants(
        {
            "train": [
                {
                    "pdb_id": "1abc",
                    "object_key": "CCD:LIG",
                    "prepared_smiles": "CC",
                    "class_name": "peptide_like",
                    "class_index": 3,
                    "occurrence_candidate_ids": [0],
                }
            ],
            "validation": [
                {
                    "pdb_id": "1abc",
                    "object_key": "CCD:LIG",
                    "prepared_smiles": "CC",
                    "class_name": "small_molecule",
                    "class_index": 4,
                    "occurrence_candidate_ids": [1],
                }
            ],
        }
    ) == []


def test_manifest_rejects_pdb_overlap_across_materialized_splits(tmp_path: Path) -> None:
    """train/validation 即使类别相同也不得复用同一 PDB 标识。"""

    roots = build_synthetic_roots(tmp_path)
    shutil.copytree(
        roots.stage1_root / "train/1abc",
        roots.stage1_root / "validation/1abc",
    )
    with pytest.raises(ValueError, match="同一 PDB 的候选记录不得同时发布到多个划分"):
        build_manifest(roots, 0.30, 3407, ("train", "validation"))


def test_manifest_allows_excluded_pdb_overlap_across_splits(tmp_path: Path) -> None:
    """两个划分中均被整 PDB 排除的同名源目录不构成已物化数据重叠。"""

    roots = build_synthetic_roots(tmp_path)
    (roots.stage1_root / "train/2bad").mkdir()
    (roots.stage1_root / "train/2bad/_BLOB_EXCEED").touch()
    manifest_path = build_manifest(roots, 0.30, 3407, ("train", "validation"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["split_pdb_counts"]["train"]["excluded"] == 1
    assert manifest["split_pdb_counts"]["validation"]["excluded"] == 1


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    (
        ("object_key", 7, "object_key 必须是非空字符串"),
        ("prepared_smiles", 7, "prepared_smiles 必须是非空字符串"),
        ("class_name", 7, "class_name 必须属于五类非背景类别"),
        ("class_index", "4", "class_index 与 class_name 不一致"),
    ),
)
def test_manifest_extension_rejects_invalid_persisted_identity_types(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
    message: str,
) -> None:
    """扩展既有 v4 时必须拒绝身份 JSONL 类型漂移，不得隐式迁移后重发。"""

    roots = build_synthetic_roots(tmp_path)
    manifest_path = build_manifest(roots, 0.30, 3407, ("train",))
    identity_path = manifest_path.parent / "train_identities.jsonl"
    identities = load_jsonl(identity_path)
    identities[0][field_name] = invalid_value
    identity_path.write_text(
        "".join(json.dumps(record) + "\n" for record in identities),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        ensure_manifest(roots, 0.30, 3407, ("train", "validation"))
    assert not (manifest_path.parent / "validation_candidates.jsonl").exists()
    assert not (manifest_path.parent / "validation_identities.jsonl").exists()


@pytest.mark.parametrize(
    ("modality", "field_name", "probability_index"),
    (
        ("A", "A_probability", 0),
        ("P", "P_probability", 0),
        ("V", "centered_probability", 1),
        ("PP", None, None),
    ),
)
def test_dataset_rejects_stage1_probability_outside_unit_interval(
    tmp_path: Path,
    modality: str,
    field_name: str | None,
    probability_index: int | None,
) -> None:
    """A/P/V/PP 注意力偏置与手动统计共享的概率都必须位于 [0,1]。"""

    roots = build_synthetic_roots(tmp_path)
    manifest_path = build_manifest(
        roots,
        coverage_threshold=0.30,
        selection_seed=3407,
        split_names=("train", "validation"),
    )
    if field_name is not None:
        assert probability_index is not None
        centered_path = roots.stage1_root / "train/1abc/centered/F1_centered.npz"
        with np.load(centered_path, allow_pickle=False) as archive:
            centered = {name: np.asarray(archive[name]).copy() for name in archive.files}
        centered[field_name][probability_index] = np.float32(1.01)
        np.savez(centered_path, **centered)
    else:
        probability_path = roots.stage1_root / "train/1abc/probability/probability_map.npz"
        with np.load(probability_path, allow_pickle=False) as archive:
            probability = {name: np.asarray(archive[name]).copy() for name in archive.files}
        probability["probability_map"][40, 40, 40] = np.float32(1.01)
        np.savez(probability_path, **probability)
    dataset = CandidateDataset(
        roots=roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name="candidate_pretrain",
        language_family=None,
        identity_scope=None,
        pp_top_k=4,
        local_crop_strategy="fixed_center",
        cache_bytes_per_worker=64 * 1024**2,
    )
    with pytest.raises(ValueError, match=rf"{modality}.*\[0,1\]"):
        dataset[0]
    sampler = ShuffledBatchSampler(
        dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        seed=7,
        sample_count_multiple=1,
    )
    assert list(sampler) == [[0]]


def test_manifest_v4_scope_distinguishes_small_from_multiple_foreground_classes(tmp_path) -> None:
    """合成 ion 与 small_molecule 后，两种身份范围读取不同身份轴。"""

    roots = build_synthetic_roots(tmp_path)
    occurrence_path = roots.data_root / "parse/1abc/occurrences.jsonl"
    occurrences = load_jsonl(occurrence_path)
    occurrences.append(
        {
            "pdb_id": "1abc",
            "candidate_id": 2,
            "object_key": "ION:ZN",
            "type_tag": "ion",
        }
    )
    occurrence_path.write_text(
        "".join(json.dumps(item) + "\n" for item in occurrences),
        encoding="utf-8",
    )
    prepared_path = roots.language_root / "prepared/1abc/prepared_smiles.jsonl"
    prepared_records = load_jsonl(prepared_path)
    prepared_records.append(
        {
            "pdb_id": "1abc",
            "candidate_id": 2,
            "object_key": "ION:ZN",
            "kind": "CCD",
            "type_tag": "ion",
            "is_covalent": False,
            "smiles": "CC",
            "smiles_source": "ccd",
            "diagnostics": {},
        }
    )
    prepared_path.write_text(
        "".join(json.dumps(item) + "\n" for item in prepared_records),
        encoding="utf-8",
    )
    ligand_area_path = roots.data_root / "density/1abc/ligand_area.npz"
    with np.load(ligand_area_path, allow_pickle=False) as archive:
        masks = {name: np.asarray(archive[name]).copy() for name in archive.files}
    masks["mask_2"] = np.asarray([[79, 79, 79]], dtype=np.int32)
    np.savez(ligand_area_path, **masks)
    with np.load(
        roots.data_root / "ligand_objects/CCD_LIG.npz",
        allow_pickle=False,
    ) as archive:
        ligand_object = {name: np.asarray(archive[name]).copy() for name in archive.files}
    ligand_object["name"] = np.asarray("ION:ZN")
    np.savez(roots.data_root / "ligand_objects/ION_ZN.npz", **ligand_object)
    for family in ("molformer", "smi_ted_289m"):
        reference_path = roots.language_root / f"{family}/1abc/candidate_1.npz"
        with np.load(reference_path, allow_pickle=False) as archive:
            language = {name: np.asarray(archive[name]).copy() for name in archive.files}
        language["candidate_id"] = np.asarray(2, dtype=np.int32)
        language["object_key"] = np.asarray("ION:ZN")
        language["prepared_smiles"] = np.asarray("CC")
        np.savez(reference_path.parent / "candidate_2.npz", **language)

    manifest_path = build_manifest(
        roots,
        coverage_threshold=0.30,
        selection_seed=3407,
        split_names=("train", "validation"),
    )
    identities = load_jsonl(manifest_path.parent / "train_identities.jsonl")
    assert [(item["class_name"], item["object_key"]) for item in identities] == [
        ("ion", "ION:ZN"),
        ("small_molecule", "CCD:LIG"),
    ]
    all_foreground = CandidateDataset(
        roots=roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name="full_matcher",
        language_family="smi_ted_289m",
        identity_scope="all_non_background",
        pp_top_k=4,
        local_crop_strategy="fixed_center",
    )[0]
    assert all_foreground["identity_class_index"].tolist() == [1, 4]
    assert all_foreground["identity_object_keys"] == ["ION:ZN", "CCD:LIG"]
    assert all_foreground["identity_prepared_smiles"] == ["CC", "CC"]
    assert all_foreground["identity_target"] == 1

    small_only = CandidateDataset(
        roots=roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name="full_matcher",
        language_family="smi_ted_289m",
        identity_scope="small_molecule",
        pp_top_k=4,
        local_crop_strategy="fixed_center",
    )[0]
    assert small_only["identity_class_index"].tolist() == [4]
    assert small_only["identity_target"] == 0


def test_manifest_explicitly_extends_calibration_without_rewriting_train_semantics(
    tmp_path,
) -> None:
    """准备入口只在显式请求后补建 calibration，且既有 train 记录逐字节不变。"""

    roots = build_synthetic_roots(tmp_path)
    manifest_path = ensure_manifest(
        roots,
        coverage_threshold=0.30,
        selection_seed=3407,
        required_splits=("train", "validation"),
    )
    train_candidates_path = manifest_path.parent / "train_candidates.jsonl"
    train_before = train_candidates_path.read_bytes()
    with pytest.raises(FileNotFoundError, match="尚未物化"):
        require_manifest(
            roots,
            coverage_threshold=0.30,
            selection_seed=3407,
            required_splits=("calibration",),
        )

    ensure_manifest(
        roots,
        coverage_threshold=0.30,
        selection_seed=3407,
        required_splits=("calibration",),
    )
    assert train_candidates_path.read_bytes() == train_before
    require_manifest(
        roots,
        coverage_threshold=0.30,
        selection_seed=3407,
        required_splits=("calibration",),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["available_splits"] == ["train", "validation", "calibration"]
    assert manifest["deferred_splits"] == []


def test_dataset_accepts_float32_box_origin_recomputation_but_rejects_real_shift(tmp_path) -> None:
    """候选原点允许冻结产物的一个 float32 间隔误差，但必须拒绝毫埃级真实错位。"""

    roots = build_synthetic_roots(tmp_path)
    manifest_path = build_manifest(
        roots,
        coverage_threshold=0.30,
        selection_seed=3407,
        split_names=("train", "validation"),
    )
    centered_path = roots.stage1_root / "train/1abc/centered/F1_centered.npz"
    probability_path = roots.stage1_root / "train/1abc/probability/probability_map.npz"
    with np.load(centered_path, allow_pickle=False) as archive:
        centered = {name: np.asarray(archive[name]).copy() for name in archive.files}
    with np.load(probability_path, allow_pickle=False) as archive:
        probability = {name: np.asarray(archive[name]).copy() for name in archive.files}
    lower_origin = np.float32(333.54345703125)
    adjacent_origin = np.nextafter(lower_origin, np.float32(np.inf))
    assert adjacent_origin - lower_origin == np.float32(3.0517578125e-5)
    centered["box_origin_world"][0, 0] = lower_origin
    probability["origin_xyz"][0] = adjacent_origin
    np.savez(centered_path, **centered)
    np.savez(probability_path, **probability)
    for density_name in ("exp", "sim"):
        density_path = roots.data_root / f"density/1abc/{density_name}.npz"
        with np.load(density_path, allow_pickle=False) as archive:
            density = {name: np.asarray(archive[name]).copy() for name in archive.files}
        density["origin"][0] = adjacent_origin
        np.savez(density_path, **density)

    accepted = CandidateDataset(
        roots=roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name="candidate_pretrain",
        language_family=None,
        identity_scope=None,
        pp_top_k=4,
        local_crop_strategy="fixed_center",
    )
    assert accepted[0]["class_target"] == 4

    centered["box_origin_world"][0, 0] += np.float32(1e-3)
    np.savez(centered_path, **centered)
    rejected = CandidateDataset(
        roots=roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name="candidate_pretrain",
        language_family=None,
        identity_scope=None,
        pp_top_k=4,
        local_crop_strategy="fixed_center",
    )
    with pytest.raises(ValueError, match="最大绝对误差"):
        rejected[0]


def test_pp_cache_removes_runtime_probability_map_dependency(
    tmp_path,
) -> None:
    """PP 缓存命中后逐值复用首次构建结果，且不再读取完整概率图。"""

    roots = build_synthetic_roots(tmp_path)
    manifest_path = build_manifest(
        roots,
        coverage_threshold=0.30,
        selection_seed=3407,
        split_names=("train", "validation"),
    )
    dataset = CandidateDataset(
        roots=roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name="candidate_pretrain",
        language_family=None,
        identity_scope=None,
        pp_top_k=4,
        local_crop_strategy="fixed_center",
        cache_bytes_per_worker=64 * 1024**2,
    )
    first = dataset[0]
    assert first["pp_index_zyx"][0].tolist() == [24, 24, 24]
    assert first["pp_tokens"][0, 0].item() == 1.0

    cache_path = roots.pp_cache_root(4, "fixed_center") / "train/1abc.npz"
    assert cache_path.is_file()
    with np.load(cache_path, allow_pickle=False) as archive:
        assert int(archive["schema_version"].item()) == 2
        assert int(archive["top_k"].item()) == 4
        assert archive["tokens"].dtype == np.float32
        assert archive["index_zyx"].dtype == np.int16
        assert archive["crop_start_zyx"].dtype == np.int16
        np.testing.assert_array_equal(archive["crop_start_zyx"], [[16, 16, 16]])

    probability_path = roots.stage1_root / "train/1abc/probability/probability_map.npz"
    probability_path.unlink()
    reloaded = CandidateDataset(
        roots=roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name="candidate_pretrain",
        language_family=None,
        identity_scope=None,
        pp_top_k=4,
        local_crop_strategy="fixed_center",
        cache_bytes_per_worker=64 * 1024**2,
    )[0]
    np.testing.assert_array_equal(reloaded["pp_tokens"], first["pp_tokens"])
    np.testing.assert_array_equal(reloaded["pp_index_zyx"], first["pp_index_zyx"])


def test_crop_strategies_use_disjoint_voxel_and_pp_cache_paths(tmp_path) -> None:
    """固定中心和 V 质心视野不得命中彼此的 voxel/PP 缓存。"""

    roots = build_synthetic_roots(tmp_path)
    manifest_path = build_manifest(
        roots,
        coverage_threshold=0.30,
        selection_seed=3407,
        split_names=("train", "validation"),
    )
    strategies: tuple[LocalCropStrategy, ...] = ("fixed_center", "v_centroid")
    for strategy in strategies:
        CandidateDataset(
            roots=roots,
            manifest_path=manifest_path,
            split_name="train",
            task_name="candidate_pretrain",
            language_family=None,
            identity_scope=None,
            pp_top_k=4,
            local_crop_strategy=strategy,
            cache_bytes_per_worker=64 * 1024**2,
        )[0]
    fixed_voxel = (
        roots.voxel_cache_root("fixed_center")
        / "train/1abc/candidate_0_start_16_16_16.npy"
    )
    centroid_voxel = (
        roots.voxel_cache_root("v_centroid")
        / "train/1abc/candidate_0_start_0_0_0.npy"
    )
    fixed_pp = roots.pp_cache_root(4, "fixed_center") / "train/1abc.npz"
    centroid_pp = roots.pp_cache_root(4, "v_centroid") / "train/1abc.npz"
    assert all(path.is_file() for path in (fixed_voxel, centroid_voxel, fixed_pp, centroid_pp))
    assert fixed_voxel != centroid_voxel
    assert fixed_pp != centroid_pp
    with np.load(fixed_pp, allow_pickle=False) as fixed_archive:
        np.testing.assert_array_equal(fixed_archive["crop_start_zyx"], [[16, 16, 16]])
    with np.load(centroid_pp, allow_pickle=False) as centroid_archive:
        np.testing.assert_array_equal(centroid_archive["crop_start_zyx"], [[0, 0, 0]])


def test_voxel_cache_path_binds_crop_start_and_prevents_silent_reuse(tmp_path) -> None:
    """同一候选的两个裁剪起点必须物化到不同 NPY，且各自命中自身数组。"""

    roots = build_synthetic_roots(tmp_path)
    store = VoxelInputStore(roots, "v_centroid", source_cache_bytes=0)
    first_start = np.asarray([0, 0, 0], dtype=np.int64)
    second_start = np.asarray([16, 16, 16], dtype=np.int64)
    first = store.load("train", "1abc", 0, first_start)
    second = store.load("train", "1abc", 0, second_start)
    first_path = (
        roots.voxel_cache_root("v_centroid")
        / "train/1abc/candidate_0_start_0_0_0.npy"
    )
    second_path = (
        roots.voxel_cache_root("v_centroid")
        / "train/1abc/candidate_0_start_16_16_16.npy"
    )
    assert first_path.is_file()
    assert second_path.is_file()
    assert first_path != second_path
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(store.load("train", "1abc", 0, first_start), first)
    np.testing.assert_array_equal(store.load("train", "1abc", 0, second_start), second)


def test_voxel_input_bridges_anisotropic_density_and_f1_geometry(tmp_path) -> None:
    """非零原点、各向异性尺寸和非对称 ZYX 起点必须共同解析到同一候选框。"""

    roots = build_synthetic_roots(tmp_path)
    density_shape = (84, 84, 84)
    box_start_zyx = np.asarray([1, 2, 3], dtype=np.int32)
    origin_xyz = np.asarray([10.0, -20.0, 30.0], dtype=np.float32)
    voxel_size_xyz = np.asarray([1.25, 1.5, 2.0], dtype=np.float32)
    box_origin_xyz = origin_xyz + box_start_zyx[[2, 1, 0]] * voxel_size_xyz
    marker_zyx = box_start_zyx + np.asarray([16, 16, 16], dtype=np.int32)
    experimental = np.zeros((1, *density_shape), dtype=np.float32)
    simulated = np.zeros_like(experimental)
    experimental[(0, *marker_zyx.tolist())] = np.float32(0.75)
    for density_name, grid in (("exp", experimental), ("sim", simulated)):
        np.savez(
            roots.data_root / f"density/1abc/{density_name}.npz",
            grid=grid,
            voxel_size=voxel_size_xyz,
            origin=origin_xyz,
        )

    centered_path = roots.stage1_root / "train/1abc/centered/F1_centered.npz"
    with np.load(centered_path, allow_pickle=False) as archive:
        centered = {name: np.asarray(archive[name]).copy() for name in archive.files}
    centered["box_start_zyx"] = box_start_zyx[None, :]
    centered["box_origin_world"] = box_origin_xyz[None, :]
    centered["voxel_size_world"] = voxel_size_xyz[None, :]
    np.savez(centered_path, **centered)

    receptor_local_xyz = np.asarray([[16.5, 16.5, 16.5]], dtype=np.float32)
    np.savez(
        roots.data_root / "parse/1abc/receptor_tokens.npz",
        coords=box_origin_xyz[None, :] + receptor_local_xyz * voxel_size_xyz[None, :],
        feat=np.ones((1, 49), dtype=np.float32),
    )
    value = VoxelInputStore(roots, "fixed_center", source_cache_bytes=0).load(
        "train",
        "1abc",
        0,
        np.asarray([16, 16, 16], dtype=np.int64),
    )
    assert value[0, 0, 0, 0] == np.float16(0.75)
    assert value[56, 0, 0, 0] == np.float16(1.0)


@pytest.mark.parametrize(
    ("field_name", "expected_error"),
    (("origin", "box_origin"), ("voxel_size", "voxel_size")),
)
def test_voxel_input_rejects_density_geometry_shift_before_cache_write(
    tmp_path,
    field_name: str,
    expected_error: str,
) -> None:
    """完整密度图原点或尺寸漂移时必须在发布体素缓存前失败。"""

    roots = build_synthetic_roots(tmp_path)
    for density_name in ("exp", "sim"):
        density_path = roots.data_root / f"density/1abc/{density_name}.npz"
        with np.load(density_path, allow_pickle=False) as archive:
            density = {name: np.asarray(archive[name]).copy() for name in archive.files}
        density[field_name][0] += np.float32(1e-3)
        np.savez(density_path, **density)

    cache_path = (
        roots.voxel_cache_root("fixed_center")
        / "train/1abc/candidate_0_start_16_16_16.npy"
    )
    with pytest.raises(ValueError, match=expected_error):
        VoxelInputStore(roots, "fixed_center", source_cache_bytes=0).load(
            "train",
            "1abc",
            0,
            np.asarray([16, 16, 16], dtype=np.int64),
        )
    assert not cache_path.exists()


def test_pp_top_k_matches_legacy_argpartition_selection() -> None:
    """PP top-k 必须逐值复现首批任务使用的旧 ``argpartition`` 选择。"""

    random_values = np.random.default_rng(3407).integers(
        0,
        4,
        size=(80, 80, 80),
        dtype=np.int16,
    ).astype(np.float32)
    tokens, index_zyx = _build_pp_tokens(
        random_values,
        np.zeros(3, dtype=np.int64),
        np.full(3, 16, dtype=np.int64),
        np.ones(3, dtype=np.float32),
        1024,
    )
    flat = random_values[16:64, 16:64, 16:64].reshape(-1)
    expected = np.argpartition(flat, flat.size - 1024)[-1024:]
    expected = expected[np.lexsort((expected, -flat[expected]))]
    actual = np.ravel_multi_index(
        (index_zyx[:, 0], index_zyx[:, 1], index_zyx[:, 2]),
        (48, 48, 48),
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(tokens[:, 0], flat[expected])


def test_precompute_entry_requires_explicit_products_and_builds_pp(tmp_path) -> None:
    """正式预计算入口拒绝含糊参数，并能按 PDB 物化显式选择的 PP 缓存。"""

    roots = build_synthetic_roots(tmp_path)
    manifest_path = build_manifest(
        roots,
        coverage_threshold=0.30,
        selection_seed=3407,
        split_names=("train", "validation"),
    )
    with pytest.raises(ValueError, match="非空且不重复"):
        precompute_inputs(roots, manifest_path, ("train",), 1, (), None, "fixed_center")
    with pytest.raises(ValueError, match="必须提供正整数"):
        precompute_inputs(
            roots, manifest_path, ("train",), 1, ("pp",), None, "fixed_center"
        )
    with pytest.raises(ValueError, match="不得提供"):
        precompute_inputs(
            roots, manifest_path, ("train",), 1, ("voxel",), 4, "fixed_center"
        )

    precompute_inputs(roots, manifest_path, ("train",), 1, ("pp",), 4, "fixed_center")
    cache_path = roots.pp_cache_root(4, "fixed_center") / "train/1abc.npz"
    assert cache_path.is_file()


def test_seen_identity_requires_train_positive_for_current_language_family(tmp_path) -> None:
    """某语言模型没有可监督正目标时，同一 object_key 在该语言下必须标为 unseen。"""

    roots = build_synthetic_roots(tmp_path)
    missing_smi = (
        roots.language_root
        / "smi_ted_289m/1abc/candidate_1.npz"
    )
    missing_smi.unlink()
    manifest_path = build_manifest(
        roots,
        coverage_threshold=0.30,
        selection_seed=3407,
        split_names=("train", "validation"),
    )
    candidate = load_jsonl(manifest_path.parent / "train_candidates.jsonl")[0]
    assert candidate["language_target_available"] == {
        "molformer": True,
        "smi_ted_289m": False,
    }
    assert candidate["identity_seen_in_train_by_language"] == {
        "molformer": True,
        "smi_ted_289m": False,
    }


def test_ligand_graph_rejects_residue_id_outside_embedding_table(tmp_path) -> None:
    """LigandObject 的 residue_id 必须在模型 512 类嵌入表范围内。"""

    roots = build_synthetic_roots(tmp_path)
    graph_path = roots.data_root / "ligand_objects/CCD_LIG.npz"
    with np.load(graph_path, allow_pickle=False) as archive:
        atoms = archive["atoms"].copy()
        bonds = archive["bonds"].copy()
        name = archive["name"].copy()
    atoms["residue_id"][0] = 512
    np.savez_compressed(graph_path, atoms=atoms, bonds=bonds, name=name)
    store = LigandGraphStore(roots.data_root, cache_bytes=0)
    with pytest.raises(ValueError, match="residue_id.*512"):
        store.load("CCD:LIG")


def test_branched_graph_does_not_invent_cross_residue_geometry(tmp_path) -> None:
    """多 CCD 组件配体按 ``residue_id`` 原子组中心化，跨组化学键距离必须为零。"""

    roots = build_synthetic_roots(tmp_path)
    graph_path = roots.data_root / "ligand_objects/CCD_LIG.npz"
    with np.load(graph_path, allow_pickle=False) as archive:
        atoms = archive["atoms"].copy()
        bonds = archive["bonds"].copy()
        name = archive["name"].copy()
    expanded_atoms = np.zeros(4, dtype=atoms.dtype)
    expanded_atoms["element"] = 6
    expanded_atoms["is_present"] = True
    expanded_atoms["residue_id"] = [1, 1, 2, 2]
    expanded_atoms["ref_pos"] = [
        [10.0, 0.0, 0.0],
        [12.0, 0.0, 0.0],
        [-20.0, 0.0, 0.0],
        [-16.0, 0.0, 0.0],
    ]
    expanded_bonds = np.zeros(3, dtype=bonds.dtype)
    expanded_bonds["atom_1"] = [0, 1, 2]
    expanded_bonds["atom_2"] = [1, 2, 3]
    expanded_bonds["type"][:, 0] = True
    np.savez_compressed(
        graph_path,
        atoms=expanded_atoms,
        bonds=expanded_bonds,
        name=name,
    )

    graph = LigandGraphStore(roots.data_root, cache_bytes=0).load("CCD:LIG")
    np.testing.assert_allclose(
        graph.node_continuous[:, -3:],
        [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]],
    )
    np.testing.assert_allclose(graph.edge_features[:, -1], [0.2, 0.0, 0.4] * 2)


def test_shuffled_sampler_ignores_pdb_groups_and_is_epoch_deterministic() -> None:
    """候选全局洗牌，不读取 Dataset 上可能存在的 PDB 分组字段。"""

    class SixCandidates:
        indices_by_pdb = {"first": [0, 1, 2], "second": [3, 4, 5]}

        def __len__(self) -> int:
            return 6

    dataset = SixCandidates()
    sampler = ShuffledBatchSampler(
        dataset,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        seed=11,
        sample_count_multiple=2,
    )
    first = list(sampler)
    assert sorted(index for batch in first for index in batch) == list(range(6))
    assert first != [[0, 1], [2, 3], [4, 5]]
    assert list(sampler) == first
    sampler.set_epoch(1)
    assert list(sampler) != first


def test_a100_and_a800_samplers_use_the_same_global_batch_candidates() -> None:
    """物理批次 8/16 都先把全局洗牌候选截断到 32 的整倍，因而每轮样本序列完全一致。"""

    class FortyFiveCandidates:
        def __len__(self) -> int:
            return 45

    dataset = FortyFiveCandidates()
    a100 = ShuffledBatchSampler(dataset, 8, True, True, 3407, 32)
    a800 = ShuffledBatchSampler(dataset, 16, True, True, 3407, 32)
    a100_indices = [index for batch in a100 for index in batch]
    a800_indices = [index for batch in a800 for index in batch]
    assert len(a100_indices) == 32
    assert a100_indices == a800_indices


def test_byte_budget_lru_promotes_hits_and_evicts_by_array_bytes() -> None:
    """LRU 按实际数组字节数限制每个 worker 的 PDB 缓存。"""

    cache: ByteBudgetLru[str, np.ndarray] = ByteBudgetLru(maximum_bytes=16)
    cache.put("a", np.zeros(2, dtype=np.float32))
    cache.put("b", np.zeros(2, dtype=np.float32))
    assert cache.get("a") is not None
    cache.put("c", np.zeros(2, dtype=np.float32))
    assert list(cache.keys()) == ["a", "c"]
    assert cache.get("b") is None


def test_file_lock_does_not_unlock_after_lock_failure(tmp_path, monkeypatch) -> None:
    """取得锁失败时只关闭句柄，不调用尚未成立的解锁操作。"""

    unlock_called = False

    def fail_lock(_handle) -> None:
        raise RuntimeError("lock failed")

    def record_unlock(_handle) -> None:
        nonlocal unlock_called
        unlock_called = True

    monkeypatch.setattr(locking, "_lock", fail_lock)
    monkeypatch.setattr(locking, "_unlock", record_unlock)
    with pytest.raises(RuntimeError, match="lock failed"):
        with locking.exclusive_file_lock(tmp_path / "failed.lock"):
            raise AssertionError("不应进入锁内代码。")
    assert not unlock_called


def test_file_lock_preserves_body_error_when_unlock_also_fails(tmp_path, monkeypatch) -> None:
    """锁内原始异常优先于释放锁时的次生异常。"""

    monkeypatch.setattr(locking, "_lock", lambda _handle: None)

    def fail_unlock(_handle) -> None:
        raise RuntimeError("unlock failed")

    monkeypatch.setattr(locking, "_unlock", fail_unlock)
    with pytest.raises(ValueError, match="body failed"):
        with locking.exclusive_file_lock(tmp_path / "body.lock"):
            raise ValueError("body failed")
