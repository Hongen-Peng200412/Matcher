"""读取不依赖 Stage1 推理产物的 `ground_truth_context` Matcher 样本。

主要入口 `GroundTruthMatcherDataset` 读取正式 PDB 清单和 A–G 产物。每个真实
occurrence 直接产生一个 GT ligand-area 候选、10 Å A 图和中心 48³ 实验密度。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from .contracts import MatcherSample, PocketInput
from .data_common import (
    build_receptor_graph,
    centered_crop,
    coverage_targets,
    load_ligand_graph,
    read_jsonl,
)


@dataclass(frozen=True)
class GroundTruthDataConfig:
    """模式一正式清单、A–G 读取和图构造参数。"""

    data_root: Path
    manifest_path: Path
    split: str
    training: bool
    seed: int
    map_size: int
    receptor_envelope_angstrom: float
    graph_radius: float
    graph_max_radius_neighbors: int
    graph_rbf_bins: int
    load_binding_labels: bool


class GroundTruthMatcherDataset(Dataset[MatcherSample]):
    """把一个 PDB 的全部 GT occurrence 物化为一个 Matcher 样本。"""

    def __init__(self, config: GroundTruthDataConfig) -> None:
        self.config = config
        manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1 or manifest["mode"] != "ground_truth_context":
            raise ValueError("模式一 Dataset 只读取 ground_truth_context schema 1 清单。")
        self.entries = manifest["splits"][config.split]
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def occurrence_counts(self) -> tuple[int, ...]:
        """供 occurrence 预算批次采样器使用，不触发重型数据读取。"""

        return tuple(int(entry["num_occurrences"]) for entry in self.entries)

    def __getitem__(self, manifest_index: int) -> MatcherSample:
        entry = self.entries[manifest_index]
        pdb_id = entry["pdb_id"]
        parse_dir = self.config.data_root / "parse" / pdb_id
        density_dir = self.config.data_root / "density" / pdb_id
        occurrence_by_id = {
            int(item["candidate_id"]): item
            for item in read_jsonl(parse_dir / "occurrences.jsonl")
        }
        occurrence_ids = [int(value) for value in entry["occurrence_ids"]]
        occurrences = [occurrence_by_id[value] for value in occurrence_ids]

        with np.load(parse_dir / "ligand_coords.npz", allow_pickle=False) as arrays:
            coordinates = tuple(
                torch.from_numpy(np.asarray(arrays[f"coords_{cid}"], dtype=np.float32))
                for cid in occurrence_ids
            )
            present = tuple(
                torch.from_numpy(np.asarray(arrays[f"present_{cid}"], dtype=bool))
                for cid in occurrence_ids
            )
            centroids = np.stack(
                [np.asarray(arrays[f"centroid_atom_{cid}"], dtype=np.float32) for cid in occurrence_ids]
            )
        with np.load(density_dir / "ligand_area.npz", allow_pickle=False) as arrays:
            masks = tuple(
                np.asarray(arrays[f"mask_{cid}"], dtype=np.int32) for cid in occurrence_ids
            )
            voxel_size_xyz = np.asarray(arrays["voxel_size_xyz"], dtype=np.float32)
            origin_xyz = np.asarray(arrays["origin_xyz"], dtype=np.float32)
        with np.load(density_dir / "exp.npz", allow_pickle=False) as arrays:
            density_grid = np.asarray(arrays["grid"], dtype=np.float32)
        with np.load(parse_dir / "receptor_tokens.npz", allow_pickle=False) as arrays:
            receptor = {name: arrays[name] for name in arrays.files}

        binding_atom = None
        if self.config.load_binding_labels:
            with np.load(
                self.config.data_root / "labels" / pdb_id / "atom_labels.npz",
                allow_pickle=False,
            ) as arrays:
                binding_atom = np.asarray(arrays["binding_atom"], dtype=bool)

        rotation = None
        if self.config.training:
            rng = np.random.default_rng(
                self.config.seed + (self.epoch + 1) * (len(self.entries) + 1) + manifest_index
            )
            planes = ((1, 2), (0, 2), (0, 1))
            rotation = (planes[int(rng.integers(0, 3))], int(rng.integers(0, 4)))

        pockets = []
        for center_xyz, mask_zyx, atom_xyz, atom_present in zip(
            centroids, masks, coordinates, present, strict=True
        ):
            nearest_distance, _ = cKDTree(atom_xyz[atom_present].numpy()).query(
                receptor["coords"], distance_upper_bound=self.config.receptor_envelope_angstrom
            )
            A_global_index = np.flatnonzero(np.isfinite(nearest_distance)).astype(np.int64)
            density, start_zyx = centered_crop(
                density_grid,
                center_xyz,
                origin_xyz,
                voxel_size_xyz,
                size=self.config.map_size,
                rotation=rotation,
            )
            pockets.append(
                PocketInput(
                    torch.from_numpy(center_xyz),
                    density,
                    torch.from_numpy(start_zyx),
                    torch.from_numpy(voxel_size_xyz),
                    torch.from_numpy(origin_xyz),
                    build_receptor_graph(
                        receptor,
                        A_global_index,
                        binding_atom,
                        graph_radius=self.config.graph_radius,
                        graph_max_radius_neighbors=self.config.graph_max_radius_neighbors,
                        graph_rbf_bins=self.config.graph_rbf_bins,
                    ),
                    torch.from_numpy(A_global_index),
                    torch.from_numpy(mask_zyx),
                    rotation[0] if rotation else None,
                    rotation[1] if rotation else 0,
                )
            )

        object_keys: list[str] = []
        for occurrence in occurrences:
            if occurrence["object_key"] not in object_keys:
                object_keys.append(occurrence["object_key"])
        ligands = tuple(
            load_ligand_graph(
                self.config.data_root,
                key,
                graph_radius=self.config.graph_radius,
                graph_max_radius_neighbors=self.config.graph_max_radius_neighbors,
                graph_rbf_bins=self.config.graph_rbf_bins,
            )
            for key in object_keys
        )
        object_to_index = {key: index for index, key in enumerate(object_keys)}
        targets = coverage_targets(masks, masks, centroids, centroids)
        return MatcherSample(
            pdb_id,
            manifest_index,
            ligands,
            tuple(pockets),
            torch.tensor(
                [object_to_index[item["object_key"]] for item in occurrences], dtype=torch.long
            ),
            torch.tensor(occurrence_ids, dtype=torch.long),
            torch.from_numpy(centroids),
            coordinates,
            present,
            *targets,
        )
