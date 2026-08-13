"""把冻结清单中的单个 F1 候选转换为新 Matcher 的体素、四模态和身份输入。

``CandidateDataset`` 按候选返回结构化 ``CandidateItem``；105 通道和 PP 缓存的
原子落盘分别由 ``VoxelInputStore`` 与 ``PpInputStore`` 负责。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from torch.utils.data import Dataset

from .cache import ByteBudgetLru
from .contracts import CandidateItem, CandidateMetadata
from .geometry import (
    candidate_sampling_features,
    keep_local_48_at,
    local_xyz_to_crop_centered_world,
    resolve_candidate_crop,
    voxel_zyx_to_crop_centered_world,
)
from .ligand_graph import LigandGraphStore
from .manifest import load_jsonl
from .pp_input import PpCandidateInput, PpInputStore
from .schema import (
    CLASS_NAMES,
    A_TOKEN_DIM,
    BOX_ORIGIN_ABSOLUTE_TOLERANCE_ANGSTROM,
    CANDIDATE_MANUAL_FEATURE_DIM,
    FULL_CONTEXT_HALF_EXTENT_VOXELS,
    LANGUAGE_EMBEDDING_DIM,
    LANGUAGE_FAMILIES,
    LOCAL_CROP_STRATEGIES,
    LOCAL_BOX_SIZE,
    LOCAL_CONTEXT_HALF_EXTENT_VOXELS,
    P_TOKEN_DIM,
    SourceRoots,
    V_TOKEN_DIM,
    IdentityScope,
    LocalCropStrategy,
    identity_scope_class_indices,
)
from .voxel_input import VoxelInputStore


TASK_NAMES = ("candidate_pretrain", "full_matcher")


class CandidateDataset(Dataset[CandidateItem]):
    """按候选粒度读取六分类样本，并按需附加 PDB 内指定范围的身份清单。

    ``roots`` 给出冻结数据、Stage1 产物与 Matcher 缓存根；``manifest_path`` 指向
    版本化清单；``split_name`` 选择其物化划分。``task_name`` 决定只构造候选任务或
    完整 Matcher；后者由 ``language_family`` 和 ``identity_scope`` 选择语言表示与身份范围。
    ``pp_top_k`` 限制每候选 PP token 数；``local_crop_strategy`` 决定 105 通道、V、PP
    和两个区域目标共享的 ``48³`` 起点，并同时移动 A/P 相对坐标原点。
    ``cache_bytes_per_worker`` 是每个 DataLoader 工作进程自己的总缓存上限，
    默认 16 GiB，不会预先分配。预算覆盖 PDB 级 F1-centered/辅助标签、尚未
    预计算的 105 通道源数组、PP 缓存、身份输入和 LigandObject 图；五部分分别
    使用 13/16、1/16、1/16、1/32 和 1/32 的上限。
    """

    def __init__(
        self,
        roots: SourceRoots,
        manifest_path: Path,
        split_name: str,
        task_name: str,
        language_family: str | None,
        identity_scope: IdentityScope | None,
        pp_top_k: int,
        local_crop_strategy: LocalCropStrategy,
        cache_bytes_per_worker: int = 16 * 1024**3,
    ) -> None:
        if task_name not in TASK_NAMES:
            raise ValueError(f"task_name 必须属于 {TASK_NAMES}，实际为 {task_name!r}。")
        if task_name == "full_matcher" and language_family is None:
            raise ValueError("full_matcher 必须明确指定 language_family。")
        if task_name == "full_matcher" and identity_scope is None:
            raise ValueError("full_matcher 必须明确指定 identity_scope。")
        if task_name == "candidate_pretrain" and identity_scope is not None:
            raise ValueError("candidate_pretrain 的 identity_scope 必须为 None。")
        if language_family is not None and language_family not in LANGUAGE_FAMILIES:
            raise ValueError(
                f"language_family 必须属于 {LANGUAGE_FAMILIES} 或为 None，"
                f"实际为 {language_family!r}。"
            )
        if pp_top_k <= 0:
            raise ValueError("pp_top_k 必须为正整数。")
        if local_crop_strategy not in LOCAL_CROP_STRATEGIES:
            raise ValueError(
                f"local_crop_strategy 必须属于 {LOCAL_CROP_STRATEGIES}。"
            )
        if cache_bytes_per_worker < 0:
            raise ValueError("cache_bytes_per_worker 不能为负数。")
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
        self.identity_scope = identity_scope
        self.pp_top_k = pp_top_k
        self.local_crop_strategy = local_crop_strategy
        self.cache_bytes_per_worker = int(cache_bytes_per_worker)
        self.records = load_jsonl(manifest_path.parent / split_files["candidates"])
        identity_records = load_jsonl(manifest_path.parent / split_files["identities"])
        self.identities_by_pdb: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for identity in identity_records:
            self.identities_by_pdb[str(identity["pdb_id"])].append(identity)
        for identities in self.identities_by_pdb.values():
            identities.sort(
                key=lambda item: (int(item["class_index"]), str(item["object_key"]))
            )

        budgets = _split_cache_budget(self.cache_bytes_per_worker)
        source_cache_bytes = budgets["source"]
        artifact_cache_bytes = budgets["artifact"]
        self.voxel_store = VoxelInputStore(
            roots,
            local_crop_strategy,
            source_cache_bytes=source_cache_bytes,
        )
        self.pp_store = PpInputStore(
            roots,
            pp_top_k,
            local_crop_strategy,
            cache_bytes=budgets["pp"],
        )
        self.graph_store = LigandGraphStore(roots.data_root, cache_bytes=budgets["graph"])
        self._artifact_cache: ByteBudgetLru[tuple[str, str], dict[str, Any]] = ByteBudgetLru(
            artifact_cache_bytes
        )
        self._identity_cache: ByteBudgetLru[tuple[str, str], dict[str, Any]] = ByteBudgetLru(
            budgets["identity"]
        )

    def __len__(self) -> int:
        """返回当前划分中未被整 PDB 排除的候选数量。"""

        return len(self.records)

    def __getitem__(self, index: int) -> CandidateItem:
        """物化一个候选；字段形状与掩码语义由 ``CandidateItem`` 固定。"""

        record = self.records[index]
        pdb_id = str(record["pdb_id"])
        candidate_index = int(record["candidate_index"])
        artifacts = self._load_pdb_artifacts(pdb_id)
        centered = artifacts["centered"]
        voxel_size = np.asarray(centered["voxel_size_world"][candidate_index], dtype=np.float32)
        box_start = np.asarray(centered["box_start_zyx"][candidate_index], dtype=np.int64)
        v_start, v_stop = _offset_range(centered["voxel_offsets"], candidate_index)
        full_v_index = np.asarray(
            centered["voxel_index_local_zyx"][v_start:v_stop],
            dtype=np.int64,
        )
        crop_geometry = resolve_candidate_crop(full_v_index, self.local_crop_strategy)
        if crop_geometry.full_v_count != int(record["full_candidate_voxel_count"]):
            raise ValueError(
                f"{self.split_name}/{pdb_id}/candidate_{candidate_index} 的权威 V 数量"
                "与冻结清单不一致。"
            )
        pp_input = self.pp_store.load(self.split_name, pdb_id, candidate_index)
        _validate_candidate_geometry(
            centered,
            pp_input,
            candidate_index,
            crop_geometry.start_zyx,
        )

        a_tokens = _build_a_tokens(
            centered,
            candidate_index,
            voxel_size,
            crop_geometry.start_zyx,
        )
        binding_target = _build_binding_target(
            artifacts["receptor_coords"],
            artifacts["binding_atom"],
            np.asarray(centered["box_origin_world"][candidate_index], dtype=np.float32),
            voxel_size,
            crop_geometry.start_zyx,
        )
        p_tokens = _build_p_tokens(
            centered,
            candidate_index,
            voxel_size,
            crop_geometry.start_zyx,
        )
        v_tokens, v_index_zyx = _build_v_tokens(
            centered,
            v_start,
            v_stop,
            full_v_index,
            voxel_size,
            crop_geometry.start_zyx,
        )
        ligand_target = _build_ligand_target(
            artifacts["occurrence_masks"],
            box_start,
            crop_geometry.start_zyx,
        )

        metadata: CandidateMetadata = {
            "split": self.split_name,
            "pdb_id": pdb_id,
            "candidate_index": candidate_index,
            "matched_object_key": record["matched_object_key"],
            "identity_seen_in_train": (
                record["identity_seen_in_train_by_language"][self.language_family]
                if self.language_family is not None
                else None
            ),
        }
        item: dict[str, Any] = {
            "metadata": metadata,
            "voxel_input": self.voxel_store.load(
                self.split_name,
                pdb_id,
                candidate_index,
                crop_geometry.start_zyx,
            ),
            "a_tokens": a_tokens,
            "p_tokens": p_tokens,
            "v_tokens": v_tokens,
            "v_index_zyx": v_index_zyx,
            "pp_tokens": pp_input["tokens"],
            "pp_index_zyx": pp_input["index_zyx"],
            "class_target": int(record["class_index"]),
            "ligand_area_target": ligand_target,
            "binding_area_target": binding_target,
            "full_candidate_voxel_count": int(record["full_candidate_voxel_count"]),
            "candidate_manual_features": _candidate_manual_features(
                a_tokens,
                p_tokens,
                v_tokens,
                pp_input["tokens"],
            ),
            "candidate_sampling_features": candidate_sampling_features(crop_geometry),
        }
        if self.task_name == "full_matcher":
            item.update(
                self._load_identity_inputs(
                    pdb_id,
                    int(record["class_index"]),
                    record["matched_object_key"],
                )
            )
        else:
            item.update(
                identity_object_keys=[],
                identity_class_index=np.empty(0, dtype=np.int64),
                language_embeddings=np.empty((0, LANGUAGE_EMBEDDING_DIM), dtype=np.float32),
                ligand_graphs=[],
                identity_target=-1,
            )
        return cast(CandidateItem, item)

    def _load_pdb_artifacts(self, pdb_id: str) -> dict[str, Any]:
        """按字节预算缓存最近使用的 PDB 级 F1-centered、受体和监督数组。"""

        key = (self.split_name, pdb_id)
        cached = self._artifact_cache.get(key)
        if cached is not None:
            return cached
        centered_path = (
            self.roots.stage1_root / self.split_name / pdb_id / "centered/F1_centered.npz"
        )
        with np.load(centered_path, allow_pickle=False) as archive:
            centered = {name: np.asarray(archive[name]).copy() for name in _CENTERED_FIELDS}
        label_path = self.roots.data_root / "labels" / pdb_id / "atom_labels.npz"
        with np.load(label_path, allow_pickle=False) as archive:
            binding_atom = np.asarray(archive["binding_atom"], dtype=bool)
        receptor_path = self.roots.data_root / "parse" / pdb_id / "receptor_tokens.npz"
        with np.load(receptor_path, allow_pickle=False) as archive:
            receptor_coords = np.asarray(archive["coords"], dtype=np.float32)
        if receptor_coords.ndim != 2 or receptor_coords.shape[1] != 3:
            raise ValueError(f"{receptor_path}:coords 必须为有限 (N,3)。")
        if not np.isfinite(receptor_coords).all():
            raise ValueError(f"{receptor_path}:coords 含 NaN 或 Inf。")
        if binding_atom.shape != (receptor_coords.shape[0],):
            raise ValueError(f"{label_path}:binding_atom 必须与完整受体原子表逐项对齐。")
        ligand_path = self.roots.data_root / "density" / pdb_id / "ligand_area.npz"
        occurrence_masks: list[np.ndarray] = []
        with np.load(ligand_path, allow_pickle=False) as archive:
            for name in archive.files:
                suffix = name.removeprefix("mask_")
                if name.startswith("mask_") and suffix.isdigit():
                    occurrence_masks.append(np.asarray(archive[name], dtype=np.int64))

        cached = {
            "centered": centered,
            "binding_atom": binding_atom,
            "receptor_coords": receptor_coords,
            "occurrence_masks": occurrence_masks,
        }
        self._artifact_cache.put(key, cached)
        return cached

    def _load_identity_inputs(
        self,
        pdb_id: str,
        matched_class_index: int,
        matched_object_key: str | None,
    ) -> dict[str, Any]:
        """读取当前范围内语言表示和配体图都可用的去重前景身份。"""

        if self.language_family is None or self.identity_scope is None:
            raise AssertionError("full_matcher 的语言模型和身份范围不应为空。")
        cache_key = (pdb_id, f"{self.language_family}:{self.identity_scope}")
        cached = self._identity_cache.get(cache_key)
        if cached is not None:
            target = _identity_target_index(
                cached["identity_object_keys"],
                cached["identity_class_index"],
                matched_class_index,
                matched_object_key,
            )
            return {**cached, "identity_target": target}
        object_keys: list[str] = []
        embeddings: list[np.ndarray] = []
        class_indices: list[int] = []
        graphs = []
        allowed_classes = set(identity_scope_class_indices(self.identity_scope))
        for identity in self.identities_by_pdb.get(pdb_id, []):
            class_index = int(identity["class_index"])
            if class_index not in allowed_classes:
                continue
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
            class_indices.append(class_index)
            graphs.append(self.graph_store.load(object_key))
        embedding_array = (
            np.stack(embeddings, axis=0)
            if embeddings
            else np.empty((0, LANGUAGE_EMBEDDING_DIM), dtype=np.float32)
        )
        cached = {
            "identity_object_keys": object_keys,
            "language_embeddings": embedding_array,
            "identity_class_index": np.asarray(class_indices, dtype=np.int64),
            "ligand_graphs": graphs,
        }
        self._identity_cache.put(cache_key, cached)
        target = _identity_target_index(
            object_keys,
            cached["identity_class_index"],
            matched_class_index,
            matched_object_key,
        )
        return {**cached, "identity_target": target}

    def __getstate__(self) -> dict[str, Any]:
        """DataLoader 派生 worker 时清空父进程 PDB 大数组缓存。"""

        state = self.__dict__.copy()
        budgets = _split_cache_budget(self.cache_bytes_per_worker)
        state["_artifact_cache"] = ByteBudgetLru(budgets["artifact"])
        state["_identity_cache"] = ByteBudgetLru(budgets["identity"])
        return state


# ================================================================================================


def _candidate_manual_features(
    a_tokens: np.ndarray,
    p_tokens: np.ndarray,
    v_tokens: np.ndarray,
    pp_tokens: np.ndarray,
) -> np.ndarray:
    """返回 A/P/V/PP 的实体数、固化 Stage1 概率均值和最大值。"""

    result: list[float] = []
    for name, tokens, probability_column in (
        ("A", a_tokens, -4),
        ("P", p_tokens, -4),
        ("V", v_tokens, -4),
        ("PP", pp_tokens, 0),
    ):
        if tokens.ndim != 2:
            raise ValueError(f"{name} token 必须是二维数组。")
        count = int(tokens.shape[0])
        if count == 0:
            result.extend((0.0, 0.0, 0.0))
            continue
        probability = np.asarray(tokens[:, probability_column], dtype=np.float32)
        if not np.isfinite(probability).all():
            raise ValueError(f"{name} 的 Stage1 固化概率含 NaN 或 Inf。")
        result.extend((float(count), float(probability.mean()), float(probability.max())))
    features = np.asarray(result, dtype=np.float32)
    if features.shape != (CANDIDATE_MANUAL_FEATURE_DIM,):
        raise RuntimeError("候选手动特征宽度与 schema 常量不一致。")
    return features


def _identity_target_index(
    object_keys: list[str],
    class_indices: np.ndarray,
    matched_class_index: int,
    matched_object_key: str | None,
) -> int:
    """按类别与 object_key 唯一定位真实身份；不可用时返回 -1。"""

    if matched_object_key is None:
        return -1
    matches = [
        index
        for index, (class_index, object_key) in enumerate(zip(class_indices, object_keys))
        if int(class_index) == matched_class_index and object_key == matched_object_key
    ]
    if len(matches) > 1:
        raise ValueError("同一 PDB 身份清单含重复 (class_index, object_key)。")
    return matches[0] if matches else -1


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
    "A_probability",
    "A_feat_L0",
    "A_feat_L1",
    "A_feat_L2",
    "A_feat_L3",
)


def _validate_candidate_geometry(
    centered: dict[str, np.ndarray],
    pp_input: PpCandidateInput,
    candidate_index: int,
    crop_start_zyx: np.ndarray,
) -> None:
    """核对 F1-centered 和概率图保存的候选框空间字段。

    先要求 ``F1_centered.npz:voxel_size_world`` 与
    ``probability_map.npz:voxel_size_xyz`` 逐元素相等，再比较保存的
    ``box_origin_world`` 与由 ``origin_xyz``、``box_start_zyx`` 和体素尺寸
    重算的候选框原点。float32 重算绝对误差允许不超过 ``5e-5 Å``。
    """

    voxel_size = np.asarray(centered["voxel_size_world"][candidate_index], dtype=np.float32)
    box_origin = np.asarray(centered["box_origin_world"][candidate_index], dtype=np.float32)
    box_start = np.asarray(centered["box_start_zyx"][candidate_index], dtype=np.float32)
    probability_origin = np.asarray(pp_input["probability_origin"], dtype=np.float32)
    probability_voxel_size = np.asarray(
        pp_input["probability_voxel_size"],
        dtype=np.float32,
    )
    cached_crop_start = np.asarray(pp_input["crop_start_zyx"], dtype=np.int64)
    named_vectors: dict[str, np.ndarray] = {
        "F1-centered voxel_size_world": voxel_size,
        "F1-centered box_origin_world": box_origin,
        "F1-centered box_start_zyx": box_start,
        "probability_map origin_xyz": probability_origin,
        "probability_map voxel_size_xyz": probability_voxel_size,
        "PP cache crop_start_zyx": cached_crop_start,
    }
    for field_name, vector in named_vectors.items():
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"{field_name} 必须是有限 (3,) 空间字段。")
    expected_origin = probability_origin + box_start[[2, 1, 0]] * voxel_size
    if not np.array_equal(voxel_size, probability_voxel_size):
        raise ValueError("F1-centered 与 probability_map 的 voxel_size 不一致。")
    if not np.array_equal(cached_crop_start, np.asarray(crop_start_zyx, dtype=np.int64)):
        raise ValueError("PP 缓存与 CandidateDataset 解析的候选裁剪起点不一致。")
    maximum_absolute_error = float(np.max(np.abs(box_origin - expected_origin)))
    if not np.allclose(
        box_origin,
        expected_origin,
        rtol=0.0,
        atol=BOX_ORIGIN_ABSOLUTE_TOLERANCE_ANGSTROM,
    ):
        raise ValueError(
            f"F1-centered box_origin={box_origin.tolist()} 与离散起点推导值"
            f" {expected_origin.tolist()} 不一致；最大绝对误差为"
            f" {maximum_absolute_error:.9g} Å，允许上限为"
            f" {BOX_ORIGIN_ABSOLUTE_TOLERANCE_ANGSTROM:.9g} Å。"
        )


def _build_a_tokens(
    centered: dict[str, np.ndarray],
    candidate_index: int,
    voxel_size: np.ndarray,
    crop_start_zyx: np.ndarray,
) -> np.ndarray:
    """保留完整 A token，不用经过筛选的 A 子集构造区域监督。

    229 维依次为 L0(49)、L1(64)、L2(64)、L3(48)、概率(1)和相对
    XYZ(3)。A 覆盖完整 80³，因此坐标按该空间的物理半边长归一化。
    """

    start, stop = _offset_range(centered["A_offsets"], candidate_index)
    centered_coordinates = local_xyz_to_crop_centered_world(
        centered["A_coord_local_xyz"][start:stop],
        crop_start_zyx,
        voxel_size,
    )
    coordinate_scale = _coordinate_scale_xyz(voxel_size, FULL_CONTEXT_HALF_EXTENT_VOXELS)
    features = np.concatenate(
        (
            np.asarray(centered["A_feat_L0"][start:stop], dtype=np.float32),
            np.asarray(centered["A_feat_L1"][start:stop], dtype=np.float32),
            np.asarray(centered["A_feat_L2"][start:stop], dtype=np.float32),
            np.asarray(centered["A_feat_L3"][start:stop], dtype=np.float32),
            np.asarray(centered["A_probability"][start:stop], dtype=np.float32)[:, None],
            centered_coordinates / coordinate_scale[None, :],
        ),
        axis=1,
    )
    return _validate_tokens(features, A_TOKEN_DIM, "A")


def _build_binding_target(
    receptor_coords: np.ndarray,
    binding_atom: np.ndarray,
    box_origin_xyz: np.ndarray,
    voxel_size_xyz: np.ndarray,
    crop_start_zyx: np.ndarray,
) -> np.ndarray:
    """把完整受体表中的结合原子独立散射到候选实际裁剪的 ``48³``。

    A token 是来源组件周围的筛选子集，不能作为辅助监督的原子全集。世界 XYZ
    先按候选 80³ 原点换算为连续局部坐标，向下取整得到所属体素，再减去该候选
    的 ``48³`` XYZ 起点。
    """

    coordinates = np.asarray(receptor_coords, dtype=np.float32)
    labels = np.asarray(binding_atom, dtype=bool)
    origin = np.asarray(box_origin_xyz, dtype=np.float32)
    voxel_size = np.asarray(voxel_size_xyz, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("receptor_coords 必须为 (N,3)。")
    if labels.shape != (coordinates.shape[0],):
        raise ValueError("binding_atom 必须与 receptor_coords 第 0 维逐原子对齐。")
    if origin.shape != (3,) or voxel_size.shape != (3,) or np.any(voxel_size <= 0):
        raise ValueError("候选 box_origin_xyz/voxel_size_xyz 必须是合法 (3,) 空间字段。")
    if not np.isfinite(coordinates).all() or not np.isfinite(origin).all() or not np.isfinite(voxel_size).all():
        raise ValueError("binding-area 空间字段含 NaN 或 Inf。")
    local_xyz = (coordinates - origin[None, :]) / voxel_size[None, :]
    local_index_xyz = (
        np.floor(local_xyz).astype(np.int64)
        - np.asarray(crop_start_zyx, dtype=np.int64)[[2, 1, 0]]
    )
    valid = labels & np.all(
        (local_index_xyz >= 0) & (local_index_xyz < LOCAL_BOX_SIZE),
        axis=1,
    )
    selected = local_index_xyz[valid]
    target = np.zeros((LOCAL_BOX_SIZE, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE), dtype=bool)
    target[selected[:, 2], selected[:, 1], selected[:, 0]] = True
    return target


def _build_p_tokens(
    centered: dict[str, np.ndarray],
    candidate_index: int,
    voxel_size: np.ndarray,
    crop_start_zyx: np.ndarray,
) -> np.ndarray:
    """构造完整 80³ 的 116 维 P token。

    字段依次为 L2(64)、L3(48)、概率(1)和按完整空间物理半边长归一化的
    相对 XYZ(3)。
    """

    start, stop = _offset_range(centered["P_offsets"], candidate_index)
    coordinates = local_xyz_to_crop_centered_world(
        centered["P_coord_local_xyz"][start:stop],
        crop_start_zyx,
        voxel_size,
    )
    coordinate_scale = _coordinate_scale_xyz(voxel_size, FULL_CONTEXT_HALF_EXTENT_VOXELS)
    tokens = np.concatenate(
        (
            np.asarray(centered["P_feat_L2"][start:stop], dtype=np.float32),
            np.asarray(centered["P_feat_L3"][start:stop], dtype=np.float32),
            np.asarray(centered["P_probability"][start:stop], dtype=np.float32)[:, None],
            coordinates / coordinate_scale[None, :],
        ),
        axis=1,
    )
    return _validate_tokens(tokens, P_TOKEN_DIM, "P")


def _build_v_tokens(
    centered: dict[str, np.ndarray],
    start: int,
    stop: int,
    full_index: np.ndarray,
    voxel_size: np.ndarray,
    crop_start_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """构造候选级 48³ 内的 52 维 V token。

    字段依次为 Stage1 voxel_final(48)、候选概率(1)和按局部空间物理
    半边长归一化的相对 XYZ(3)。监督覆盖率仍使用清单中的完整组件。
    """

    keep, cropped_index = keep_local_48_at(full_index, crop_start_zyx)
    coordinates = voxel_zyx_to_crop_centered_world(
        full_index[keep],
        crop_start_zyx,
        voxel_size,
    )
    coordinate_scale = _coordinate_scale_xyz(voxel_size, LOCAL_CONTEXT_HALF_EXTENT_VOXELS)
    tokens = np.concatenate(
        (
            np.asarray(centered["voxel_final"][start:stop][keep], dtype=np.float32),
            np.asarray(centered["centered_probability"][start:stop][keep], dtype=np.float32)[:, None],
            coordinates / coordinate_scale[None, :],
        ),
        axis=1,
    )
    return _validate_tokens(tokens, V_TOKEN_DIM, "V"), cropped_index


def _build_ligand_target(
    occurrence_masks: list[np.ndarray],
    box_start_zyx: np.ndarray,
    local_crop_start_zyx: np.ndarray,
) -> np.ndarray:
    """由全部 occurrence 稀疏掩码构造候选 ``48³`` 配体区域并集。"""

    crop_start = np.asarray(box_start_zyx, dtype=np.int64) + np.asarray(
        local_crop_start_zyx,
        dtype=np.int64,
    )
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


def _coordinate_scale_xyz(voxel_size: np.ndarray, half_extent_voxels: float) -> np.ndarray:
    """把网格半边长转换成 XYZ 三轴的物理坐标归一化尺度。"""

    scale = np.asarray(voxel_size, dtype=np.float32) * float(half_extent_voxels)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(f"坐标归一化尺度必须为有限正 (3,)，实际为 {scale}。")
    return scale


def _validate_tokens(tokens: np.ndarray, expected_width: int, modality: str) -> np.ndarray:
    """固定一种模态的二维宽度和有限值契约。"""

    values = np.asarray(tokens, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != expected_width:
        raise ValueError(
            f"{modality} token 必须为 (N,{expected_width})，实际为 {values.shape}。"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{modality} token 含 NaN 或 Inf。")
    return values


def _split_cache_budget(total_bytes: int) -> dict[str, int]:
    """把单 worker 总预算无重叠地分配给五类可淘汰缓存。"""

    graph = total_bytes // 32
    identity = total_bytes // 32
    source = total_bytes // 16
    pp = total_bytes // 16
    artifact = total_bytes - graph - identity - source - pp
    return {
        "artifact": artifact,
        "source": source,
        "pp": pp,
        "identity": identity,
        "graph": graph,
    }
