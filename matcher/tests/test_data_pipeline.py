"""验证新数据适配层的覆盖率、空间边界、缺失策略和批次字段。"""

from __future__ import annotations

import json
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
    assert item["metadata"]["identity_seen_in_train"] is True
    assert item["language_embeddings"].shape == (1, 768)
    assert item["ligand_area_target"][24, 24, 24]
    assert item["binding_area_target"][24, 24, 24]
    assert item["binding_area_target"][24, 24, 25]
    assert int(item["binding_area_target"].sum()) == 2

    batch = collate_candidates([cast(CandidateItem, item)])
    assert batch["voxel_input"].shape == (1, 105, 48, 48, 48)
    assert batch["identity_mask"].tolist() == [[True]]
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


def test_seen_identity_requires_train_positive_for_current_language_family(tmp_path) -> None:
    """某语言模型没有可监督正目标时，同一 object_key 在该语言下必须标为 unseen。"""

    roots = build_synthetic_roots(tmp_path)
    missing_smi = (
        roots.language_root
        / "smi_ted_289m/1abc/candidate_1.npz"
    )
    missing_smi.unlink()
    manifest_path = build_manifest(roots, coverage_threshold=0.30, selection_seed=3407)
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
