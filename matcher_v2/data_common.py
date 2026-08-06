"""读取配体、受体图和中心 48³ 实验密度的共同变换。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from matcher.element_properties import ELEMENT_PROPERTIES
from matcher.graph import build_molecular_graph

from .contracts import LigandInput, RawGraph


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取非空 JSONL 记录并保持文件顺序。"""

    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_object_filename(object_key: str) -> str:
    """把配体 `object_key` 转成 A–G 使用的安全 NPZ 文件名。"""

    return "".join(
        character if character.isalnum() or character in "_.-" else "_"
        for character in object_key
    )


def load_ligand_graph(
    data_root: Path,
    object_key: str,
    *,
    graph_radius: float,
    graph_max_radius_neighbors: int,
    graph_rbf_bins: int,
) -> LigandInput:
    """读取一个去重配体模板并构造化学键与半径边的同构分子图。"""

    path = data_root / "ligand_objects" / f"{safe_object_filename(object_key)}.npz"
    with np.load(path, allow_pickle=True) as arrays:
        atoms = arrays["atoms"]
        bonds = arrays["bonds"]

    # (M,149), Emap2lig 顺序：原子名、元素、形式电荷、手性、环、残基和元素属性。
    node_input = np.zeros((len(atoms), 149), dtype=np.float32)
    for atom_index, atom in enumerate(atoms):
        atomic_number = int(atom["element"])
        node_input[atom_index, :4] = atom["name"]
        if 0 <= atomic_number < 128:
            node_input[atom_index, 4 + atomic_number] = 1.0
        node_input[atom_index, 132] = atom["charge"]
        node_input[atom_index, 133:140] = atom["chirality"]
        node_input[atom_index, 140:144] = atom["in_ring"]
        node_input[atom_index, 144] = atom["residue_id"]
        node_input[atom_index, 145:] = ELEMENT_PROPERTIES[atomic_number]

    coordinates = torch.from_numpy(np.array(atoms["ref_pos"], dtype=np.float32, copy=True))
    bond_index = torch.from_numpy(
        np.stack((bonds["atom_1"], bonds["atom_2"])).astype(np.int64, copy=False)
    )
    graph = build_molecular_graph(
        coordinates,
        torch.from_numpy(np.array(atoms["residue_id"], dtype=np.int64, copy=True)),
        bond_index,
        torch.from_numpy(np.asarray(bonds["type"], dtype=bool)),
        torch.zeros((len(bonds), 3), dtype=torch.bool),
        torch.from_numpy(np.asarray(bonds["in_ring"], dtype=bool)),
        radius=graph_radius,
        max_radius_neighbors=graph_max_radius_neighbors,
        rbf_bins=graph_rbf_bins,
    )
    return LigandInput(
        object_key,
        RawGraph(
            torch.from_numpy(node_input),
            coordinates,
            graph,
            torch.from_numpy(np.array(atoms["element"], dtype=np.int64, copy=True)),
        ),
    )


def build_receptor_graph(
    receptor: dict[str, np.ndarray],
    atom_index: np.ndarray,
    binding_atom: np.ndarray | None,
    *,
    graph_radius: float,
    graph_max_radius_neighbors: int,
    graph_rbf_bins: int,
    node_input: np.ndarray | None = None,
) -> RawGraph:
    """从完整受体原子编号构造 A 图，并保留原始编号用于身份追踪。"""

    coordinates = torch.from_numpy(np.asarray(receptor["coords"][atom_index], dtype=np.float32))
    global_to_local = {int(global_index): local for local, global_index in enumerate(atom_index)}
    source, target = receptor["bond_index"]
    keep = np.isin(source, atom_index) & np.isin(target, atom_index)
    global_edges = receptor["bond_index"][:, keep]
    local_edges = (
        np.asarray(
            [[global_to_local[int(value)] for value in endpoint] for endpoint in global_edges],
            dtype=np.int64,
        )
        if global_edges.shape[1]
        else np.empty((2, 0), dtype=np.int64)
    )

    bond_type = receptor["bond_type"][keep]
    bond_order = torch.zeros((len(bond_type), 5), dtype=torch.bool)
    bond_role = torch.zeros((len(bond_type), 3), dtype=torch.bool)
    order_mapping = {0: 0, 1: 1, 2: 4, 3: 0, 4: 0, 5: 0, 6: 2}
    for edge_index, code in enumerate(bond_type.tolist()):
        bond_order[edge_index, order_mapping[int(code)]] = True
        if code in (3, 4, 5):
            bond_role[edge_index, int(code) - 3] = True

    graph = build_molecular_graph(
        coordinates,
        torch.zeros(len(atom_index), dtype=torch.long),
        torch.from_numpy(local_edges),
        bond_order,
        bond_role,
        torch.zeros((local_edges.shape[1], 4), dtype=torch.bool),
        radius=graph_radius,
        max_radius_neighbors=graph_max_radius_neighbors,
        rbf_bins=graph_rbf_bins,
    )
    actual_node_input = (
        np.asarray(receptor["feat"][atom_index], dtype=np.float32)
        if node_input is None
        else np.asarray(node_input, dtype=np.float32)
    )
    return RawGraph(
        torch.from_numpy(actual_node_input),
        coordinates,
        graph,
        torch.from_numpy(np.asarray(receptor["element"][atom_index], dtype=np.int64)),
        (
            torch.from_numpy(np.asarray(binding_atom[atom_index], dtype=bool))
            if binding_atom is not None
            else None
        ),
    )


def centered_crop(
    grid: np.ndarray,
    center_xyz: np.ndarray,
    origin_xyz: np.ndarray,
    voxel_size_xyz: np.ndarray,
    *,
    size: int,
    rotation: tuple[tuple[int, int], int] | None,
) -> tuple[Tensor, np.ndarray]:
    """保持物理中心不移动地裁剪密度，越出完整图的体素以零填充。

    `grid` 为 `[C,Z,Y,X]`；返回密度为 `float32 [C,size,size,size]`，起点为完整图
    ZYX 整数坐标。归一化只使用裁剪后有限值的 0.1%–99.9% 分位区间。
    """

    center_index_xyz = (center_xyz - origin_xyz) / voxel_size_xyz - 0.5
    start_zyx = np.rint(center_index_xyz[::-1] - (size - 1) / 2).astype(np.int64)
    output = np.zeros((grid.shape[0], size, size, size), dtype=np.float32)
    source_start = np.maximum(start_zyx, 0)
    source_end = np.minimum(start_zyx + size, np.asarray(grid.shape[1:], dtype=np.int64))
    target_start = source_start - start_zyx
    target_end = target_start + np.maximum(source_end - source_start, 0)
    if np.all(source_end > source_start):
        output[
            :,
            target_start[0] : target_end[0],
            target_start[1] : target_end[1],
            target_start[2] : target_end[2],
        ] = grid[
            :,
            source_start[0] : source_end[0],
            source_start[1] : source_end[1],
            source_start[2] : source_end[2],
        ]
    lower, upper = np.quantile(output, (0.001, 0.999))
    output = np.clip(output, lower, upper)
    output = (output - output.mean()) / max(float(output.std()), 1.0e-8)
    if rotation is not None:
        output = np.rot90(output, k=rotation[1], axes=(rotation[0][0] + 1, rotation[0][1] + 1))
    return torch.from_numpy(np.ascontiguousarray(output)), start_zyx.astype(np.int32)


def coverage_targets(
    candidate_masks: tuple[np.ndarray, ...],
    occurrence_masks: tuple[np.ndarray, ...],
    candidate_centers_xyz: np.ndarray,
    occurrence_centroids_xyz: np.ndarray,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """按真实稀疏 mask 交集和中心距离生成 A/B/O 与三个硬截断标签。"""

    shape = np.max(
        np.concatenate((*candidate_masks, *occurrence_masks), axis=0), axis=0
    ) + 1
    candidate_linear = [np.ravel_multi_index(mask.T, shape) for mask in candidate_masks]
    occurrence_linear = [np.ravel_multi_index(mask.T, shape) for mask in occurrence_masks]
    intersection = np.asarray(
        [
            [np.intersect1d(candidate, occurrence, assume_unique=True).size for occurrence in occurrence_linear]
            for candidate in candidate_linear
        ],
        dtype=np.float32,
    )
    A = intersection / np.asarray([len(mask) for mask in occurrence_masks], dtype=np.float32)[None]
    B = intersection / np.asarray([len(mask) for mask in candidate_masks], dtype=np.float32)[:, None]
    distance = np.linalg.norm(
        candidate_centers_xyz[:, None] - occurrence_centroids_xyz[None], axis=-1
    ).astype(np.float32)
    O = 1.0 / (1.0 + distance)
    return (
        torch.from_numpy(A),
        torch.from_numpy(B),
        torch.from_numpy(O),
        torch.from_numpy(A >= 0.10),
        torch.from_numpy(B >= 0.10),
        torch.from_numpy(distance < 10.0),
    )

