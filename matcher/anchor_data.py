"""不依赖 Stage1 推理产物的 Anchor Matcher 数据入口。

一个 Dataset 样本对应一个完整 PDB：全部真实 occurrence slot、一次 synthetic-anchor
候选集合、每个候选的中心 48³ Map 和 18 Å A 图。候选来源只参与抽样，不进入返回值。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .element_properties import ELEMENT_PROPERTIES
from .graph import MolecularGraph, build_molecular_graph


@dataclass(frozen=True)
class RawGraph:
    """一个实体的原始图。

    ``node_input [N,F]``、``coordinates [N,3]`` 与 ``element [N]`` 共享节点轴；
    ``binding_atom [N]`` 仅用于 A 的可选辅助监督。边字段见 ``MolecularGraph``。
    """

    node_input: Tensor
    coordinates: Tensor
    graph: MolecularGraph
    element: Tensor
    binding_atom: Tensor | None = None


@dataclass(frozen=True)
class LigandEntity:
    """一个 PDB 内按 object_key 去重的配体模板图。"""

    object_key: str
    raw_graph: RawGraph


@dataclass(frozen=True)
class CandidateEntity:
    """一个候选的 ``center_xyz [3]``、``density [1,48,48,48]`` 和 A 图。"""

    center_xyz: Tensor
    density: Tensor
    A_graph: RawGraph
    A_global_index: Tensor


@dataclass(frozen=True)
class AnchorSample:
    """一个完整 PDB 样本。

    ``ligands`` 按首次出现的 ``object_key`` 去重，``candidates`` 至少含一项；
    ``occurrence_to_ligand [S]`` 把全部真实 slot 映射到配体身份。O/O′ 三个标签
    都按 ``[C,S]`` 排列；可选真实配体坐标只供细辅助损失生成标签。
    """

    pdb_id: str
    manifest_index: int
    ligands: tuple[LigandEntity, ...]
    candidates: tuple[CandidateEntity, ...]
    occurrence_to_ligand: Tensor
    occurrence_candidate_id: Tensor
    occurrence_centroid_xyz: Tensor
    ligand_gt_coordinates: tuple[Tensor, ...] | None
    ligand_gt_present: tuple[Tensor, ...] | None
    O_target: Tensor
    O_prime_target: Tensor
    O_prime_exact: Tensor

    @property
    def num_occurrences(self) -> int:
        return int(self.occurrence_candidate_id.numel())

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)


@dataclass(frozen=True)
class AnchorDataConfig:
    """Anchor 数据读取和 synthetic-anchor 抽样参数。"""

    data_root: Path
    stage1_preparation_root: Path
    experiment_manifest: Path
    split: str
    training: bool
    seed: int = 3407
    p_miss: float = 0.30
    p_split: float = 0.20
    p_hit: float = 0.50
    max_context_ratio: float = 2.0
    empty_A_context_skip_probability: float = 0.80
    receptor_radius: float = 18.0
    O_radius: float = 12.0
    O_prime_scale: float = 6.0
    O_prime_observed_radius: float = 24.0
    load_entity_auxiliary_labels: bool = True
    load_fine_pair_labels: bool = False
    A_node_feature_sources: tuple[str, ...] = ("feat_l0",)
    graph_radius: float = 4.0
    graph_max_radius_neighbors: int = 48
    graph_rbf_bins: int = 16


def _safe_object_filename(object_key: str) -> str:
    return "".join(character if character.isalnum() or character in "_.-" else "_" for character in object_key)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ligand_node_input(atoms: np.ndarray) -> Tensor:
    """按 Emap2lig 顺序编码 149 维配体原子字段。

    切片依次为 4 维 atom name、128 维元素 one-hot、charge、7 维 chirality、
    4 维 ring、residue_id，以及 4 维元素属性。
    """

    features = np.zeros((len(atoms), 149), dtype=np.float32)
    for atom_index, atom in enumerate(atoms):
        atomic_number = int(atom["element"])
        features[atom_index, :4] = atom["name"]
        if 0 <= atomic_number < 128:
            features[atom_index, 4 + atomic_number] = 1.0
        features[atom_index, 132] = atom["charge"]
        features[atom_index, 133:140] = atom["chirality"]
        features[atom_index, 140:144] = atom["in_ring"]
        features[atom_index, 144] = atom["residue_id"]
        features[atom_index, 145:] = ELEMENT_PROPERTIES[atomic_number]
    return torch.from_numpy(features)


def _load_ligand(
    data_root: Path, object_key: str, config: AnchorDataConfig
) -> LigandEntity:
    path = data_root / "ligand_objects" / f"{_safe_object_filename(object_key)}.npz"
    with np.load(path, allow_pickle=True) as arrays:
        atoms = arrays["atoms"]
        bonds = arrays["bonds"]

    coordinates = torch.from_numpy(np.array(atoms["ref_pos"], dtype=np.float32, copy=True))
    residue_id = torch.from_numpy(np.array(atoms["residue_id"], dtype=np.int64, copy=True))
    bond_index = torch.from_numpy(
        np.stack((bonds["atom_1"], bonds["atom_2"])).astype(np.int64, copy=False)
    )
    graph = build_molecular_graph(
        coordinates,
        residue_id,
        bond_index,
        torch.from_numpy(np.asarray(bonds["type"], dtype=bool)),
        torch.zeros((len(bonds), 3), dtype=torch.bool),
        torch.from_numpy(np.asarray(bonds["in_ring"], dtype=bool)),
        radius=config.graph_radius,
        max_radius_neighbors=config.graph_max_radius_neighbors,
        rbf_bins=config.graph_rbf_bins,
    )
    raw_graph = RawGraph(
        node_input=_ligand_node_input(atoms),
        coordinates=coordinates,
        graph=graph,
        element=torch.from_numpy(np.array(atoms["element"], dtype=np.int64, copy=True)),
    )
    return LigandEntity(object_key, raw_graph)


def _receptor_bond_fields(bond_type: np.ndarray) -> tuple[Tensor, Tensor]:
    order = torch.zeros((len(bond_type), 5), dtype=torch.bool)
    role = torch.zeros((len(bond_type), 3), dtype=torch.bool)
    mapping = {0: 0, 1: 1, 2: 4, 3: 0, 4: 0, 5: 0, 6: 2}
    for edge_index, code in enumerate(bond_type.tolist()):
        order[edge_index, mapping[int(code)]] = True
        if code in (3, 4, 5):
            role[edge_index, int(code) - 3] = True
    return order, role


def _A_node_input(
    receptor: dict[str, np.ndarray],
    atom_index: np.ndarray,
    sources: tuple[str, ...],
) -> Tensor:
    """按配置顺序拼接当前路线已实现的 A 原子特征来源。"""

    fields = []
    for source in sources:
        if source != "feat_l0":
            raise ValueError(f"AnchorPocketDataset 尚未实现 A 特征来源：{source}")
        fields.append(np.asarray(receptor["feat"][atom_index], dtype=np.float32))
    atom_counts = {field.shape[0] for field in fields}
    if len(atom_counts) != 1:
        raise ValueError("A 特征来源的原子数不一致。")
    return torch.from_numpy(np.concatenate(fields, axis=-1))


def _candidate_center_xyz(start_zyx: np.ndarray, origin_xyz: np.ndarray, voxel_size_xyz: np.ndarray) -> np.ndarray:
    start_xyz = np.asarray(start_zyx, dtype=np.float32)[::-1]
    return origin_xyz + (start_xyz + 40.0) * voxel_size_xyz


def _A_indices(receptor_coordinates: np.ndarray, center_xyz: np.ndarray, radius: float) -> np.ndarray:
    squared_distance = np.sum((receptor_coordinates - center_xyz[None]) ** 2, axis=1)
    return np.flatnonzero(squared_distance <= radius * radius).astype(np.int64, copy=False)


def sample_candidate_starts(
    box_arrays: dict[str, np.ndarray],
    receptor_coordinates: np.ndarray,
    origin_xyz: np.ndarray,
    voxel_size_xyz: np.ndarray,
    rng: np.random.Generator,
    config: AnchorDataConfig,
) -> tuple[list[list[int]], dict[str, int]]:
    """按 miss/split/hit 与 context 规则返回候选起点和数量审计。"""

    starts: list[np.ndarray] = []
    probabilities = (config.p_miss, config.p_split, config.p_hit)
    for occurrence_index in range(len(box_arrays["occurrence_id"])):
        outcome = int(rng.choice(3, p=probabilities))
        count = (0, 2, 1)[outcome]
        if count:
            chosen = rng.choice(box_arrays["bias_start_zyx"].shape[1], size=count, replace=False)
            starts.extend(box_arrays["bias_start_zyx"][occurrence_index, chosen])

    occurrence_count = len(box_arrays["occurrence_id"])
    context_target = int(rng.integers(0, int(config.max_context_ratio * occurrence_count) + 1))
    context_order = rng.permutation(len(box_arrays["context_start_zyx"]))
    accepted = 0
    for context_index in context_order:
        if accepted >= context_target:
            break
        start = box_arrays["context_start_zyx"][context_index]
        center = _candidate_center_xyz(start, origin_xyz, voxel_size_xyz)
        empty_A = _A_indices(receptor_coordinates, center, config.receptor_radius).size == 0
        if empty_A and rng.random() < config.empty_A_context_skip_probability:
            continue
        starts.append(start)
        accepted += 1

    if starts:
        order = rng.permutation(len(starts))
        starts = [starts[index] for index in order]
    return [np.asarray(start, dtype=np.int32).tolist() for start in starts], {
        "bias_count": len(starts) - accepted,
        "context_requested": context_target,
        "context_accepted": accepted,
    }


def _load_A_graph(
    receptor: dict[str, np.ndarray],
    binding_atom: np.ndarray | None,
    A_global_index: np.ndarray,
    config: AnchorDataConfig,
) -> RawGraph:
    coordinates = torch.from_numpy(
        np.asarray(receptor["coords"][A_global_index], dtype=np.float32)
    )
    global_to_local = {int(global_index): local for local, global_index in enumerate(A_global_index)}
    source, target = receptor["bond_index"]
    keep = np.isin(source, A_global_index) & np.isin(target, A_global_index)
    global_edges = receptor["bond_index"][:, keep]
    if global_edges.shape[1]:
        local_edges = np.asarray(
            [[global_to_local[int(value)] for value in row] for row in global_edges],
            dtype=np.int64,
        )
    else:
        local_edges = np.empty((2, 0), dtype=np.int64)
    bond_order, bond_role = _receptor_bond_fields(receptor["bond_type"][keep])
    graph = build_molecular_graph(
        coordinates,
        torch.zeros(len(A_global_index), dtype=torch.long),
        torch.from_numpy(local_edges),
        bond_order,
        bond_role,
        torch.zeros((local_edges.shape[1], 4), dtype=torch.bool),
        radius=config.graph_radius,
        max_radius_neighbors=config.graph_max_radius_neighbors,
        rbf_bins=config.graph_rbf_bins,
    )
    return RawGraph(
        node_input=_A_node_input(
            receptor, A_global_index, config.A_node_feature_sources
        ),
        coordinates=coordinates,
        graph=graph,
        element=torch.from_numpy(
            np.asarray(receptor["element"][A_global_index], dtype=np.int64)
        ),
        binding_atom=(
            torch.from_numpy(np.asarray(binding_atom[A_global_index], dtype=bool))
            if binding_atom is not None
            else None
        ),
    )


def _density_crop(
    grid: np.ndarray,
    start_zyx: np.ndarray,
    rotation: tuple[tuple[int, int], int] | None,
) -> Tensor:
    start = np.asarray(start_zyx, dtype=np.int64) + 16
    z, y, x = start.tolist()
    crop = np.asarray(grid[0, z : z + 48, y : y + 48, x : x + 48], dtype=np.float32)
    lower, upper = np.quantile(crop, (0.001, 0.999))
    crop = np.clip(crop, lower, upper)
    crop = (crop - crop.mean()) / max(float(crop.std()), 1.0e-8)
    if rotation is not None:
        crop = np.rot90(crop, k=rotation[1], axes=rotation[0])
    return torch.from_numpy(np.ascontiguousarray(crop[None]))


class AnchorPocketDataset(Dataset[AnchorSample]):
    """读取版本化实验清单，并在训练期逐 epoch 重采 synthetic anchor。

    训练期在 ``prepare_epoch`` 中先剔除零候选 PDB；进入 ``__getitem__`` 后，
    训练和验证样本都保证至少含一个候选。
    """

    def __init__(self, config: AnchorDataConfig) -> None:
        self.config = config
        if not config.A_node_feature_sources:
            raise ValueError("A_node_feature_sources 不能为空。")
        with config.experiment_manifest.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != 1:
            raise ValueError("Matcher 实验清单 schema_version 必须为 1。")
        if manifest.get("route") != "anchor_O_O_prime":
            raise ValueError("AnchorPocketDataset 只读取 anchor_O_O_prime 路线清单。")
        if config.split not in manifest.get("splits", {}):
            raise ValueError(f"Matcher 实验清单不含 split：{config.split}")
        if int(manifest.get("seed", -1)) != config.seed:
            raise ValueError(
                f"Matcher 清单 seed={manifest.get('seed')}，与运行配置 {config.seed} 不一致。"
            )
        for name in (
            "source_box_manifest",
            "source_box_manifest_sha256",
            "source_box_config",
            "source_box_config_sha256",
        ):
            if not manifest.get(name):
                raise ValueError(f"Matcher 实验清单缺少来源身份字段：{name}")
        expected_sampling = {
            "p_miss": config.p_miss,
            "p_split": config.p_split,
            "p_hit": config.p_hit,
            "max_context_ratio": config.max_context_ratio,
            "empty_A_context_skip_probability": config.empty_A_context_skip_probability,
            "receptor_radius_angstrom": config.receptor_radius,
        }
        manifest_sampling = manifest.get("sampling", {})
        for name, expected in expected_sampling.items():
            actual = manifest_sampling.get(name)
            if actual is None or not np.isclose(float(actual), float(expected)):
                raise ValueError(
                    f"Matcher 清单 sampling.{name}={actual}，与运行配置 {expected} 不一致。"
                )
        self.entries: list[dict[str, Any]] = manifest["splits"][config.split]
        if not config.training and any(
            not entry.get("candidate_start_zyx") for entry in self.entries
        ):
            raise ValueError("冻结验证清单中的每个 PDB 都必须至少有一个候选。")
        self.epoch = -1
        self._candidate_starts_by_index: dict[int, list[list[int]]] = {}
        self._rotation_by_index: dict[int, tuple[tuple[int, int], int]] = {}
        self._epoch_prepared = False
        self.candidate_audit: dict[str, int] = {}
        self.available_indices = list(range(len(self.entries)))

    def set_epoch(self, epoch: int) -> None:
        if self.epoch == epoch and self._epoch_prepared:
            return
        self.epoch = epoch
        if self.config.training:
            self.prepare_epoch()

    def prepare_epoch(self) -> None:
        """先固定本 epoch 候选并剔除零候选 PDB，供 occurrence 装箱读取。"""

        self._candidate_starts_by_index = {}
        self._rotation_by_index = {}
        self._epoch_prepared = True
        audit: dict[str, int] = {}
        for manifest_index in range(len(self.entries)):
            starts, rotation, sample_audit = self._sample_candidate_starts(manifest_index)
            if starts:
                self._candidate_starts_by_index[manifest_index] = starts
                self._rotation_by_index[manifest_index] = rotation
            for name, value in sample_audit.items():
                audit[name] = audit.get(name, 0) + int(value)
        self.available_indices = list(self._candidate_starts_by_index)
        audit["PDB_kept"] = len(self.available_indices)
        audit["PDB_zero_candidate"] = len(self.entries) - len(self.available_indices)
        self.candidate_audit = audit

    def _sample_candidate_starts(
        self, manifest_index: int
    ) -> tuple[list[list[int]], tuple[tuple[int, int], int], dict[str, int]]:
        entry = self.entries[manifest_index]
        pdb_id = entry["pdb_id"]
        with np.load(
            self.config.stage1_preparation_root / entry["box_path"], allow_pickle=False
        ) as arrays:
            box = {key: arrays[key] for key in arrays.files}
        with np.load(
            self.config.data_root / "parse" / pdb_id / "receptor_tokens.npz",
            allow_pickle=False,
        ) as arrays:
            receptor_coordinates = arrays["coords"]
        with np.load(
            self.config.data_root / "density" / pdb_id / "exp.npz", allow_pickle=False
        ) as arrays:
            voxel_size_xyz = np.asarray(arrays["voxel_size"], dtype=np.float32)
            origin_xyz = np.asarray(arrays["origin"], dtype=np.float32)
        seed = self.config.seed + (self.epoch + 1) * (len(self.entries) + 1) + manifest_index
        rng = np.random.default_rng(seed)
        starts, audit = sample_candidate_starts(
            box,
            receptor_coordinates,
            origin_xyz,
            voxel_size_xyz,
            rng,
            self.config,
        )
        planes = ((1, 2), (0, 2), (0, 1))
        rotation = (planes[int(rng.integers(0, 3))], int(rng.integers(0, 4)))
        return starts, rotation, audit

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, manifest_index: int) -> AnchorSample:
        entry = self.entries[manifest_index]
        pdb_id = entry["pdb_id"]
        box_path = self.config.stage1_preparation_root / entry["box_path"]
        with np.load(box_path, allow_pickle=False) as arrays:
            box = {key: arrays[key] for key in arrays.files}
        parse_dir = self.config.data_root / "parse" / pdb_id
        occurrences = _read_jsonl(parse_dir / "occurrences.jsonl")
        occurrence_by_id = {int(item["candidate_id"]): item for item in occurrences}
        ordered_occurrences = [occurrence_by_id[int(value)] for value in box["occurrence_id"]]

        with np.load(parse_dir / "receptor_tokens.npz", allow_pickle=False) as arrays:
            receptor = {key: arrays[key] for key in arrays.files}
        with np.load(self.config.data_root / "density" / pdb_id / "exp.npz") as arrays:
            grid = arrays["grid"]
            voxel_size_xyz = np.asarray(arrays["voxel_size"], dtype=np.float32)
            origin_xyz = np.asarray(arrays["origin"], dtype=np.float32)

        if self.config.training:
            if not self._epoch_prepared:
                raise RuntimeError("训练 Dataset 必须先调用 set_epoch() 固定并过滤候选。")
            candidate_starts = self._candidate_starts_by_index[manifest_index]
            rotation = self._rotation_by_index[manifest_index]
        else:
            candidate_starts = entry["candidate_start_zyx"]
            rotation = None

        binding_atom = None
        if self.config.load_entity_auxiliary_labels:
            with np.load(
                self.config.data_root / "labels" / pdb_id / "atom_labels.npz",
                allow_pickle=False,
            ) as arrays:
                binding_atom = arrays["binding_atom"]

        candidates = []
        centers = []
        for start_list in candidate_starts:
            start = np.asarray(start_list, dtype=np.int32)
            center = _candidate_center_xyz(start, origin_xyz, voxel_size_xyz)
            A_global_index = _A_indices(
                receptor["coords"], center, self.config.receptor_radius
            )
            candidates.append(
                CandidateEntity(
                    center_xyz=torch.from_numpy(center.astype(np.float32, copy=False)),
                    density=_density_crop(grid, start, rotation),
                    A_graph=_load_A_graph(
                        receptor, binding_atom, A_global_index, self.config
                    ),
                    A_global_index=torch.from_numpy(A_global_index),
                )
            )
            centers.append(center)

        object_keys: list[str] = []
        for occurrence in ordered_occurrences:
            if occurrence["object_key"] not in object_keys:
                object_keys.append(occurrence["object_key"])
        ligands = tuple(
            _load_ligand(self.config.data_root, key, self.config) for key in object_keys
        )
        object_to_index = {key: index for index, key in enumerate(object_keys)}
        occurrence_to_ligand = torch.tensor(
            [object_to_index[item["object_key"]] for item in ordered_occurrences], dtype=torch.long
        )

        with np.load(parse_dir / "ligand_coords.npz", allow_pickle=False) as arrays:
            occurrence_ids = [int(item["candidate_id"]) for item in ordered_occurrences]
            centroids = np.stack([arrays[f"centroid_atom_{cid}"] for cid in occurrence_ids])
            if self.config.load_fine_pair_labels:
                gt_coordinates = tuple(
                    torch.from_numpy(np.asarray(arrays[f"coords_{cid}"], dtype=np.float32))
                    for cid in occurrence_ids
                )
                gt_present = tuple(
                    torch.from_numpy(np.asarray(arrays[f"present_{cid}"], dtype=bool))
                    for cid in occurrence_ids
                )
            else:
                gt_coordinates = None
                gt_present = None

        distances = torch.cdist(
            torch.from_numpy(np.asarray(centers, dtype=np.float32)),
            torch.from_numpy(np.asarray(centroids, dtype=np.float32)),
        )
        O_target = distances < self.config.O_radius
        O_prime_target = 1.0 / (1.0 + distances / self.config.O_prime_scale)
        O_prime_exact = distances < self.config.O_prime_observed_radius
        return AnchorSample(
            pdb_id=pdb_id,
            manifest_index=manifest_index,
            ligands=ligands,
            candidates=tuple(candidates),
            occurrence_to_ligand=occurrence_to_ligand,
            occurrence_candidate_id=torch.tensor(occurrence_ids, dtype=torch.long),
            occurrence_centroid_xyz=torch.from_numpy(centroids.astype(np.float32, copy=False)),
            ligand_gt_coordinates=gt_coordinates,
            ligand_gt_present=gt_present,
            O_target=O_target,
            O_prime_target=O_prime_target,
            O_prime_exact=O_prime_exact,
        )
