"""把冻结清单中的单个 F1 候选转换为新 Matcher 的体素、四模态和身份输入。"""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from .geometry import keep_local_48, local_xyz_to_centered_world, voxel_zyx_to_centered_world
from .ligand_graph import LigandGraphStore
from .manifest import load_jsonl
from .schema import (
    CLASS_NAMES,
    LANGUAGE_EMBEDDING_DIM,
    LOCAL_BOX_MARGIN,
    LOCAL_BOX_SIZE,
    SourceRoots,
)
from .voxel_input import VoxelInputStore


TASK_NAMES = ("candidate_pretrain", "full_matcher")


class CandidateDataset(Dataset[dict[str, Any]]):
    """按候选粒度读取六分类样本，并按需附加 PDB 内小分子身份清单。"""

    def __init__(
        self,
        roots: SourceRoots,
        manifest_path: Path,
        split_name: str,
        task_name: str,
        language_family: str | None,
        pp_top_k: int,
    ) -> None:
        if task_name not in TASK_NAMES:
            raise ValueError(f"task_name 必须属于 {TASK_NAMES}，实际为 {task_name!r}。")
        if task_name == "full_matcher" and language_family is None:
            raise ValueError("full_matcher 必须明确指定 language_family。")
        if pp_top_k <= 0:
            raise ValueError("pp_top_k 必须为正整数。")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tuple(manifest["class_names"]) != CLASS_NAMES:
            raise ValueError(f"{manifest_path} 的六分类顺序与代码不一致。")
        split_files = manifest["splits"].get(split_name)
        if split_files is None:
            raise KeyError(f"{manifest_path} 不含划分 {split_name!r}。")

        self.roots = roots
        self.split_name = split_name
        self.task_name = task_name
        self.language_family = language_family
        self.pp_top_k = pp_top_k
        self.records = load_jsonl(manifest_path.parent / split_files["candidates"])
        identity_records = load_jsonl(manifest_path.parent / split_files["identities"])
        self.identities_by_pdb: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for identity in identity_records:
            self.identities_by_pdb[str(identity["pdb_id"])].append(identity)
        for identities in self.identities_by_pdb.values():
            identities.sort(key=lambda item: str(item["object_key"]))

        self.indices_by_pdb: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            self.indices_by_pdb[str(record["pdb_id"])].append(index)
        self.voxel_store = VoxelInputStore(roots)
        self.graph_store = LigandGraphStore(roots.data_root)
        self._artifact_cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        """返回当前划分中未被整 PDB 排除的候选数量。"""

        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """物化一个候选的所有输入、辅助目标和可用身份清单。"""

        record = self.records[index]
        pdb_id = str(record["pdb_id"])
        candidate_index = int(record["candidate_index"])
        artifacts = self._load_pdb_artifacts(pdb_id)
        centered = artifacts["centered"]
        voxel_size = np.asarray(centered["voxel_size_world"][candidate_index], dtype=np.float32)
        box_start = np.asarray(centered["box_start_zyx"][candidate_index], dtype=np.int64)
        _validate_candidate_geometry(centered, artifacts, candidate_index)

        a_tokens, binding_target = _build_a_tokens_and_target(
            centered,
            artifacts["binding_atom"],
            candidate_index,
        )
        p_tokens = _build_p_tokens(centered, candidate_index, voxel_size)
        v_tokens, v_index_zyx = _build_v_tokens(centered, candidate_index, voxel_size)
        pp_tokens, pp_index_zyx = _build_pp_tokens(
            artifacts["probability_map"],
            box_start,
            voxel_size,
            self.pp_top_k,
        )
        ligand_target = _build_ligand_target(artifacts["occurrence_masks"], box_start)

        item: dict[str, Any] = {
            "metadata": {
                "split": self.split_name,
                "pdb_id": pdb_id,
                "candidate_index": candidate_index,
                "matched_object_key": record["matched_object_key"],
                "identity_seen_in_train": record["identity_seen_in_train"],
            },
            "voxel_input": self.voxel_store.load(self.split_name, pdb_id, candidate_index),
            "a_tokens": a_tokens,
            "p_tokens": p_tokens,
            "v_tokens": v_tokens,
            "v_index_zyx": v_index_zyx,
            "pp_tokens": pp_tokens,
            "pp_index_zyx": pp_index_zyx,
            "class_target": int(record["class_index"]),
            "ligand_area_target": ligand_target,
            "binding_area_target": binding_target,
            "full_candidate_voxel_count": int(record["full_candidate_voxel_count"]),
        }
        if self.task_name == "full_matcher":
            item.update(self._load_identity_inputs(pdb_id, record["matched_object_key"]))
        else:
            item.update(
                identity_object_keys=[],
                language_embeddings=np.empty((0, LANGUAGE_EMBEDDING_DIM), dtype=np.float32),
                ligand_graphs=[],
                identity_target=-1,
            )
        return item

    def _load_pdb_artifacts(self, pdb_id: str) -> dict[str, Any]:
        """缓存最近两个 PDB 的 F1-centered、概率图和稀疏辅助标签。"""

        key = (self.split_name, pdb_id)
        cached = self._artifact_cache.get(key)
        if cached is not None:
            self._artifact_cache.move_to_end(key)
            return cached
        centered_path = (
            self.roots.stage1_root / self.split_name / pdb_id / "centered/F1_centered.npz"
        )
        with np.load(centered_path, allow_pickle=False) as archive:
            centered = {name: np.asarray(archive[name]).copy() for name in _CENTERED_FIELDS}
        probability_path = (
            self.roots.stage1_root
            / self.split_name
            / pdb_id
            / "probability/probability_map.npz"
        )
        with np.load(probability_path, allow_pickle=False) as archive:
            probability_map = np.asarray(archive["probability_map"], dtype=np.float32)
            probability_origin = np.asarray(archive["origin_xyz"], dtype=np.float32)
            probability_voxel_size = np.asarray(archive["voxel_size_xyz"], dtype=np.float32)
        label_path = self.roots.data_root / "labels" / pdb_id / "atom_labels.npz"
        with np.load(label_path, allow_pickle=False) as archive:
            binding_atom = np.asarray(archive["binding_atom"], dtype=bool)
        ligand_path = self.roots.data_root / "density" / pdb_id / "ligand_area.npz"
        occurrence_masks: list[np.ndarray] = []
        with np.load(ligand_path, allow_pickle=False) as archive:
            for name in archive.files:
                suffix = name.removeprefix("mask_")
                if name.startswith("mask_") and suffix.isdigit():
                    occurrence_masks.append(np.asarray(archive[name], dtype=np.int64))

        cached = {
            "centered": centered,
            "probability_map": probability_map,
            "probability_origin": probability_origin,
            "probability_voxel_size": probability_voxel_size,
            "binding_atom": binding_atom,
            "occurrence_masks": occurrence_masks,
        }
        self._artifact_cache[key] = cached
        while len(self._artifact_cache) > 2:
            self._artifact_cache.popitem(last=False)
        return cached

    def _load_identity_inputs(
        self,
        pdb_id: str,
        matched_object_key: str | None,
    ) -> dict[str, Any]:
        """读取当前语言模型和配体图都可用的去重小分子身份。"""

        if self.language_family is None:
            raise AssertionError("full_matcher 的 language_family 不应为空。")
        object_keys: list[str] = []
        embeddings: list[np.ndarray] = []
        graphs = []
        for identity in self.identities_by_pdb.get(pdb_id, []):
            language = identity["language"][self.language_family]
            if not identity["graph_available"] or not language["available"]:
                continue
            object_key = str(identity["object_key"])
            relative_path = str(language["selected_relative_path"])
            path = self.roots.language_root / relative_path
            with np.load(path, allow_pickle=False) as archive:
                embedding = np.asarray(archive["embedding"], dtype=np.float32)
            if embedding.shape != (LANGUAGE_EMBEDDING_DIM,) or not np.isfinite(embedding).all():
                raise ValueError(f"{path}:embedding 必须为有限 ({LANGUAGE_EMBEDDING_DIM},)。")
            object_keys.append(object_key)
            embeddings.append(embedding)
            graphs.append(self.graph_store.load(object_key))
        target = object_keys.index(matched_object_key) if matched_object_key in object_keys else -1
        embedding_array = (
            np.stack(embeddings, axis=0)
            if embeddings
            else np.empty((0, LANGUAGE_EMBEDDING_DIM), dtype=np.float32)
        )
        return {
            "identity_object_keys": object_keys,
            "language_embeddings": embedding_array,
            "ligand_graphs": graphs,
            "identity_target": target,
        }

    def __getstate__(self) -> dict[str, Any]:
        """DataLoader 派生 worker 时清空父进程 PDB 大数组缓存。"""

        state = self.__dict__.copy()
        state["_artifact_cache"] = OrderedDict()
        return state


_CENTERED_FIELDS = (
    "box_start_zyx",
    "box_origin_world",
    "voxel_size_world",
    "voxel_offsets",
    "voxel_index_local_zyx",
    "centered_probability",
    "voxel_final",
    "P_offsets",
    "P_coord_local_xyz",
    "P_probability",
    "P_feat_L2",
    "P_feat_L3",
    "A_offsets",
    "A_global_index",
    "A_coord_local_xyz",
    "A_coord_centered_world",
    "A_probability",
    "A_feat_L0",
    "A_feat_L1",
    "A_feat_L2",
    "A_feat_L3",
)


def _validate_candidate_geometry(
    centered: dict[str, np.ndarray],
    artifacts: dict[str, Any],
    candidate_index: int,
) -> None:
    """核对 F1-centered 和概率图保存的候选框空间字段。"""

    voxel_size = np.asarray(centered["voxel_size_world"][candidate_index], dtype=np.float32)
    box_origin = np.asarray(centered["box_origin_world"][candidate_index], dtype=np.float32)
    box_start = np.asarray(centered["box_start_zyx"][candidate_index], dtype=np.float32)
    expected_origin = artifacts["probability_origin"] + box_start[[2, 1, 0]] * voxel_size
    if not np.array_equal(voxel_size, artifacts["probability_voxel_size"]):
        raise ValueError("F1-centered 与 probability_map 的 voxel_size 不一致。")
    if not np.allclose(box_origin, expected_origin, rtol=0.0, atol=2e-5):
        raise ValueError(
            f"F1-centered box_origin={box_origin.tolist()} 与离散起点推导值"
            f" {expected_origin.tolist()} 不一致。"
        )


def _build_a_tokens_and_target(
    centered: dict[str, np.ndarray],
    binding_atom: np.ndarray,
    candidate_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """保留完整 A token，并把真实 binding 原子散射到中心 48³。"""

    start, stop = _offset_range(centered["A_offsets"], candidate_index)
    global_index = np.asarray(centered["A_global_index"][start:stop], dtype=np.int64)
    if np.any(global_index < 0) or np.any(global_index >= binding_atom.shape[0]):
        raise ValueError("A_global_index 超出 binding_atom 范围。")
    centered_coordinates = np.asarray(
        centered["A_coord_centered_world"][start:stop],
        dtype=np.float32,
    )
    features = np.concatenate(
        (
            np.asarray(centered["A_feat_L0"][start:stop], dtype=np.float32),
            np.asarray(centered["A_feat_L1"][start:stop], dtype=np.float32),
            np.asarray(centered["A_feat_L2"][start:stop], dtype=np.float32),
            np.asarray(centered["A_feat_L3"][start:stop], dtype=np.float32),
            np.asarray(centered["A_probability"][start:stop], dtype=np.float32)[:, None],
            centered_coordinates / 24.0,
        ),
        axis=1,
    )
    local_xyz = np.asarray(centered["A_coord_local_xyz"][start:stop], dtype=np.float32)
    local_index_xyz = np.floor(local_xyz).astype(np.int64) - LOCAL_BOX_MARGIN
    selected_binding = binding_atom[global_index]
    valid = selected_binding & np.all(
        (local_index_xyz >= 0) & (local_index_xyz < LOCAL_BOX_SIZE),
        axis=1,
    )
    target = np.zeros((LOCAL_BOX_SIZE, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE), dtype=bool)
    coordinates = local_index_xyz[valid]
    target[coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]] = True
    return features.astype(np.float32, copy=False), target


def _build_p_tokens(
    centered: dict[str, np.ndarray],
    candidate_index: int,
    voxel_size: np.ndarray,
) -> np.ndarray:
    """保留 80³ 内全部 P token，并附加概率与候选中心相对坐标。"""

    start, stop = _offset_range(centered["P_offsets"], candidate_index)
    coordinates = local_xyz_to_centered_world(
        centered["P_coord_local_xyz"][start:stop],
        voxel_size,
    )
    return np.concatenate(
        (
            np.asarray(centered["P_feat_L2"][start:stop], dtype=np.float32),
            np.asarray(centered["P_feat_L3"][start:stop], dtype=np.float32),
            np.asarray(centered["P_probability"][start:stop], dtype=np.float32)[:, None],
            coordinates / 40.0,
        ),
        axis=1,
    ).astype(np.float32, copy=False)


def _build_v_tokens(
    centered: dict[str, np.ndarray],
    candidate_index: int,
    voxel_size: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """只保留候选完整组件中位于中心 48³ 的 V token。"""

    start, stop = _offset_range(centered["voxel_offsets"], candidate_index)
    full_index = np.asarray(centered["voxel_index_local_zyx"][start:stop], dtype=np.int64)
    keep, cropped_index = keep_local_48(full_index)
    coordinates = voxel_zyx_to_centered_world(full_index[keep], voxel_size)
    tokens = np.concatenate(
        (
            np.asarray(centered["voxel_final"][start:stop][keep], dtype=np.float32),
            np.asarray(centered["centered_probability"][start:stop][keep], dtype=np.float32)[:, None],
            coordinates / 24.0,
        ),
        axis=1,
    )
    return tokens.astype(np.float32, copy=False), cropped_index


def _build_pp_tokens(
    probability_map: np.ndarray,
    box_start_zyx: np.ndarray,
    voxel_size: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """从候选中心 48³ 概率图稳定选择概率最高的 PP token。"""

    start = np.asarray(box_start_zyx, dtype=np.int64) + LOCAL_BOX_MARGIN
    stop = start + LOCAL_BOX_SIZE
    if np.any(start < 0) or np.any(stop > np.asarray(probability_map.shape)):
        raise ValueError("PP 的 48³ 概率图裁剪越界。")
    slices = tuple(slice(int(first), int(last)) for first, last in zip(start, stop))
    crop = np.asarray(probability_map[slices], dtype=np.float32)
    flat = crop.reshape(-1)
    count = min(top_k, flat.size)
    selected = np.argpartition(flat, flat.size - count)[-count:]
    order = np.lexsort((selected, -flat[selected]))
    selected = selected[order]
    local_index = np.stack(np.unravel_index(selected, crop.shape), axis=1).astype(np.int64)
    index_in_80 = local_index + LOCAL_BOX_MARGIN
    coordinates = voxel_zyx_to_centered_world(index_in_80, voxel_size)
    tokens = np.concatenate((flat[selected, None], coordinates / 24.0), axis=1)
    return tokens.astype(np.float32, copy=False), local_index


def _build_ligand_target(
    occurrence_masks: list[np.ndarray],
    box_start_zyx: np.ndarray,
) -> np.ndarray:
    """由全部 occurrence 稀疏掩码构造中心 48³ 配体区域并集。"""

    crop_start = np.asarray(box_start_zyx, dtype=np.int64) + LOCAL_BOX_MARGIN
    target = np.zeros((LOCAL_BOX_SIZE, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE), dtype=bool)
    for mask in occurrence_masks:
        local = np.asarray(mask, dtype=np.int64) - crop_start[None, :]
        keep = np.all((local >= 0) & (local < LOCAL_BOX_SIZE), axis=1)
        selected = local[keep]
        target[selected[:, 0], selected[:, 1], selected[:, 2]] = True
    return target


def _offset_range(offsets: np.ndarray, candidate_index: int) -> tuple[int, int]:
    """从单调偏移数组读取一个候选的半开切片边界。"""

    values = np.asarray(offsets, dtype=np.int64)
    if candidate_index < 0 or candidate_index + 1 >= values.shape[0]:
        raise IndexError(f"candidate_index={candidate_index} 不在 offsets 范围内。")
    start = int(values[candidate_index])
    stop = int(values[candidate_index + 1])
    if start < 0 or stop < start:
        raise ValueError("候选 offsets 不是合法单调偏移。")
    return start, stop
