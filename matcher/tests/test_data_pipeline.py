"""验证新数据适配层的覆盖率、空间边界、缺失策略和批次字段。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from matcher import locking
from matcher.batching import ShuffledBatchSampler, collate_candidates
from matcher.cache import ByteBudgetLru
from matcher.candidate_data import CandidateDataset
from matcher.contracts import CandidateItem
from matcher.geometry import crop_center_48, hard_scatter_receptor, keep_local_48
from matcher.labels import OccurrenceMask, assign_candidate_target
from matcher.ligand_graph import LigandGraphStore
from matcher.manifest import build_manifest, ensure_manifest, load_jsonl, require_manifest
from matcher.precompute import precompute_inputs
from matcher.pp_input import _build_pp_tokens
from matcher.schema import SourceRoots

from .factories import build_synthetic_roots


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
    identities = load_jsonl(manifest_path.parent / "train_identities.jsonl")
    candidates = load_jsonl(manifest_path.parent / "train_candidates.jsonl")
    smi = identities[0]["language"]["smi_ted_289m"]
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

    dataset = CandidateDataset(
        roots=SourceRoots(roots.data_root, roots.stage1_root, roots.output_root),
        manifest_path=manifest_path,
        split_name="train",
        task_name="full_matcher",
        language_family="smi_ted_289m",
        identity_scope="small_molecule",
        pp_top_k=4,
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
    assert item["identity_target"] == 0
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
    assert batch["graph_edge_index"].shape == (2, 2)
    assert batch["graph_edge_features"][:, -1].tolist() == pytest.approx([0.15, 0.15])
    sampler = ShuffledBatchSampler(
        dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        seed=7,
        sample_count_multiple=1,
    )
    assert list(sampler) == [[0]]


def test_manifest_v3_scope_distinguishes_small_from_multiple_foreground_classes(tmp_path) -> None:
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
    )[0]
    assert all_foreground["identity_class_index"].tolist() == [1, 4]
    assert all_foreground["identity_object_keys"] == ["ION:ZN", "CCD:LIG"]
    assert all_foreground["identity_target"] == 1

    small_only = CandidateDataset(
        roots=roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name="full_matcher",
        language_family="smi_ted_289m",
        identity_scope="small_molecule",
        pp_top_k=4,
    )[0]
    assert small_only["identity_class_index"].tolist() == [4]
    assert small_only["identity_target"] == 0


def test_manifest_explicitly_extends_calibration_without_rewriting_train_semantics(
    tmp_path,
) -> None:
    """准备入口只在显式请求后补建 calibration，且既有 train 记录逐字节不变。"""

    roots = build_synthetic_roots(tmp_path)
    shutil.copytree(
        roots.stage1_root / "train/1abc",
        roots.stage1_root / "calibration/1abc",
    )
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

    accepted = CandidateDataset(
        roots=roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name="candidate_pretrain",
        language_family=None,
        identity_scope=None,
        pp_top_k=4,
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
        cache_bytes_per_worker=64 * 1024**2,
    )
    first = dataset[0]
    assert first["pp_index_zyx"][0].tolist() == [24, 24, 24]
    assert first["pp_tokens"][0, 0].item() == 1.0

    cache_path = roots.pp_cache_root(4) / "train/1abc.npz"
    assert cache_path.is_file()
    with np.load(cache_path, allow_pickle=False) as archive:
        assert int(archive["schema_version"].item()) == 1
        assert int(archive["top_k"].item()) == 4
        assert archive["tokens"].dtype == np.float32
        assert archive["index_zyx"].dtype == np.int16

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
        cache_bytes_per_worker=64 * 1024**2,
    )[0]
    np.testing.assert_array_equal(reloaded["pp_tokens"], first["pp_tokens"])
    np.testing.assert_array_equal(reloaded["pp_index_zyx"], first["pp_index_zyx"])


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
        precompute_inputs(roots, manifest_path, ("train",), 1, (), None)
    with pytest.raises(ValueError, match="必须提供正整数"):
        precompute_inputs(roots, manifest_path, ("train",), 1, ("pp",), None)
    with pytest.raises(ValueError, match="不得提供"):
        precompute_inputs(roots, manifest_path, ("train",), 1, ("voxel",), 4)

    precompute_inputs(roots, manifest_path, ("train",), 1, ("pp",), 4)
    cache_path = roots.pp_cache_root(4) / "train/1abc.npz"
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

    dataset = cast(CandidateDataset, SixCandidates())
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

    dataset = cast(CandidateDataset, FortyFiveCandidates())
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
