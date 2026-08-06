"""读取依赖 Stage1 Find F1-centered 产物的 `stage1_context` 样本。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import MatcherSample, PocketInput, Stage1Context
from .data_common import (
    build_receptor_graph,
    centered_crop,
    coverage_targets,
    load_ligand_graph,
    read_jsonl,
)
from .context import topk_probability_points


@dataclass(frozen=True)
class Stage1DataConfig:
    """模式二指针、A–G 读取和显式特征宽度。"""

    data_root: Path
    pointer_path: Path
    split: str
    map_size: int = 48
    PP_top_k: int = 256
    V_feature_dim: int = 0
    P_feature_dim: int = 0
    A_feature_dim: int = 0
    graph_radius: float = 4.5
    graph_max_radius_neighbors: int = 32
    graph_rbf_bins: int = 16
    load_binding_labels: bool = False


def _slice(arrays: dict[str, np.ndarray], prefix: str, index: int) -> slice:
    offsets = arrays[f"{prefix}_offsets"]
    return slice(int(offsets[index]), int(offsets[index + 1]))


def _feature(arrays: dict[str, np.ndarray], names: tuple[str, ...], part: slice) -> np.ndarray:
    values = [np.asarray(arrays[name][part], dtype=np.float32) for name in names]
    return np.concatenate(values, axis=-1) if values else np.empty((part.stop - part.start, 0), np.float32)


class Stage1MatcherDataset(Dataset[MatcherSample]):
    """按显式指针读取一个 PDB 的全部 F1-centered 候选。"""

    def __init__(self, config: Stage1DataConfig) -> None:
        self.config = config
        pointer = json.loads(config.pointer_path.read_text(encoding="utf-8"))
        if pointer["schema_version"] != 1 or pointer["mode"] != "stage1_context":
            raise ValueError("模式二 Dataset 只读取 stage1_context schema 1 指针。")
        if not pointer["producer"].startswith("Find_"):
            raise ValueError("当前模式二要求 P 存在，只接受 Find 系列产物。")
        if pointer["role"] != "F1_centered":
            raise ValueError("当前模式二只实现 F1_centered 候选语义。")
        self.entries = [row for row in pointer["records"] if row["split"] == config.split]
        self._occurrence_counts = [int(row["num_occurrences"]) for row in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def set_epoch(self, epoch: int) -> None:
        """模式二当前不做随机增强；保留与共同训练循环一致的轻量入口。"""

        del epoch

    @property
    def occurrence_counts(self) -> tuple[int, ...]:
        return tuple(self._occurrence_counts)

    def __getitem__(self, manifest_index: int) -> MatcherSample:
        entry = self.entries[manifest_index]
        pdb_id = entry["pdb_id"]
        parse_dir = self.config.data_root / "parse" / pdb_id
        density_dir = self.config.data_root / "density" / pdb_id
        occurrences = read_jsonl(parse_dir / "occurrences.jsonl")
        occurrence_ids = [int(row["candidate_id"]) for row in occurrences]

        with np.load(parse_dir / "ligand_coords.npz", allow_pickle=False) as source:
            coordinates = tuple(
                torch.from_numpy(np.asarray(source[f"coords_{cid}"], dtype=np.float32))
                for cid in occurrence_ids
            )
            present = tuple(
                torch.from_numpy(np.asarray(source[f"present_{cid}"], dtype=bool))
                for cid in occurrence_ids
            )
            centroids = np.stack(
                [np.asarray(source[f"centroid_atom_{cid}"], dtype=np.float32) for cid in occurrence_ids]
            )
        with np.load(density_dir / "ligand_area.npz", allow_pickle=False) as source:
            occurrence_masks = tuple(
                np.asarray(source[f"mask_{cid}"], dtype=np.int32) for cid in occurrence_ids
            )
        with np.load(density_dir / "exp.npz", allow_pickle=False) as source:
            density_grid = np.asarray(source["grid"], dtype=np.float32)
            origin_name = "origin_xyz" if "origin_xyz" in source.files else "origin"
            voxel_name = "voxel_size_xyz" if "voxel_size_xyz" in source.files else "voxel_size"
            origin_xyz = np.asarray(source[origin_name], dtype=np.float32)
            voxel_size_xyz = np.asarray(source[voxel_name], dtype=np.float32)
        with np.load(parse_dir / "receptor_tokens.npz", allow_pickle=False) as source:
            receptor = {name: source[name] for name in source.files}
        binding_atom = None
        if self.config.load_binding_labels:
            with np.load(self.config.data_root / "labels" / pdb_id / "atom_labels.npz") as source:
                binding_atom = np.asarray(source["binding_atom"], dtype=bool)
        with np.load(entry["centered_path"], allow_pickle=False) as source:
            centered = {name: source[name] for name in source.files}
        with np.load(entry["probability_path"], allow_pickle=False) as source:
            probability_map = np.asarray(source["probability_map"], dtype=np.float32)

        pockets = []
        candidate_masks = []
        candidate_centers = []
        for index in range(len(centered["centered_box_index"])):
            box_start = np.asarray(centered["box_start_zyx"][index], dtype=np.int32)
            center_xyz = np.asarray(centered["box_origin_world"][index], dtype=np.float32)
            center_xyz = center_xyz + 40.0 * np.asarray(centered["voxel_size_world"][index])
            density, map_start = centered_crop(
                density_grid, center_xyz, origin_xyz, voxel_size_xyz,
                size=self.config.map_size, rotation=None,
            )
            voxel_part = _slice(centered, "voxel", index)
            voxel_local = np.asarray(centered["voxel_index_local_zyx"][voxel_part], dtype=np.int32)
            candidate_mask = voxel_local + box_start
            candidate_masks.append(candidate_mask)
            candidate_centers.append(center_xyz)

            A_part = _slice(centered, "A", index)
            P_part = _slice(centered, "P", index)
            A_index = np.asarray(centered["A_global_index"][A_part], dtype=np.int64)
            A_extra = _feature(centered, ("A_feat_L1", "A_feat_L2", "A_feat_L3"), A_part)
            V_feature = np.asarray(centered["voxel_final"][voxel_part], dtype=np.float32)
            P_feature = _feature(centered, ("P_feat_L2", "P_feat_L3"), P_part)
            self._check_widths(V_feature, P_feature, A_extra)

            PP_local_xyz, PP_probability = topk_probability_points(
                probability_map, map_start, map_size=self.config.map_size,
                voxel_size_xyz=voxel_size_xyz, top_k=self.config.PP_top_k,
            )

            stage1 = Stage1Context(
                torch.from_numpy((voxel_local[..., ::-1] + 0.5) * voxel_size_xyz - 16.0 * voxel_size_xyz),
                torch.from_numpy(np.asarray(centered["centered_probability"][voxel_part], np.float32)),
                torch.from_numpy(V_feature),
                torch.from_numpy(np.asarray(centered["P_coord_local_xyz"][P_part], np.float32) - 16.0 * voxel_size_xyz),
                torch.from_numpy(np.asarray(centered["P_probability"][P_part], np.float32)),
                torch.from_numpy(P_feature),
                PP_local_xyz,
                PP_probability,
                torch.from_numpy(np.asarray(centered["A_probability"][A_part], np.float32)),
                torch.from_numpy(A_extra),
            )
            pockets.append(
                PocketInput(
                    torch.from_numpy(center_xyz), density, torch.from_numpy(map_start),
                    torch.from_numpy(voxel_size_xyz), torch.from_numpy(origin_xyz),
                    build_receptor_graph(
                        receptor, A_index, binding_atom,
                        graph_radius=self.config.graph_radius,
                        graph_max_radius_neighbors=self.config.graph_max_radius_neighbors,
                        graph_rbf_bins=self.config.graph_rbf_bins,
                        node_input=np.asarray(centered["A_feat_L0"][A_part], np.float32),
                    ),
                    torch.from_numpy(A_index), torch.from_numpy(candidate_mask), None, 0, stage1,
                )
            )
        if not pockets:
            raise ValueError(f"{pdb_id} 没有 Stage1 候选，应在指针生成后、Dataset 外过滤。")

        object_keys = list(dict.fromkeys(row["object_key"] for row in occurrences))
        ligands = tuple(
            load_ligand_graph(
                self.config.data_root, key,
                graph_radius=self.config.graph_radius,
                graph_max_radius_neighbors=self.config.graph_max_radius_neighbors,
                graph_rbf_bins=self.config.graph_rbf_bins,
            )
            for key in object_keys
        )
        object_to_index = {key: index for index, key in enumerate(object_keys)}
        targets = coverage_targets(
            tuple(candidate_masks), occurrence_masks, np.stack(candidate_centers), centroids
        )
        return MatcherSample(
            pdb_id, manifest_index, ligands, tuple(pockets),
            torch.tensor([object_to_index[row["object_key"]] for row in occurrences]),
            torch.tensor(occurrence_ids), torch.from_numpy(centroids), coordinates, present, *targets,
        )

    def _check_widths(self, V: np.ndarray, P: np.ndarray, A: np.ndarray) -> None:
        expected = (self.config.V_feature_dim, self.config.P_feature_dim, self.config.A_feature_dim)
        actual = (V.shape[1], P.shape[1], A.shape[1])
        if actual != expected:
            raise ValueError(f"Stage1 特征宽度与配置不符：实际 {actual}，配置 {expected}。")
