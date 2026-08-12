"""按 PDB 保持读取局部性，并把候选级变长 token 与配体图拼成固定物理批次。"""

from __future__ import annotations

import math
import random
from typing import Any, Iterator

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler

from .candidate_data import CandidateDataset
from .schema import LANGUAGE_EMBEDDING_DIM


class PdbGroupedBatchSampler(Sampler[list[int]]):
    """先连续排列每个 PDB 的候选，再切成固定大小批次以复用大体积源文件。"""

    def __init__(
        self,
        dataset: CandidateDataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数。")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """设置决定当前训练周期 PDB 顺序和 PDB 内候选顺序的整数。"""

        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        """生成物理批次；训练批次在 drop_last=True 时始终具有固定样本数。"""

        generator = random.Random(self.seed + self.epoch)
        pdb_ids = sorted(self.dataset.indices_by_pdb)
        if self.shuffle:
            generator.shuffle(pdb_ids)
        ordered_indices: list[int] = []
        for pdb_id in pdb_ids:
            indices = list(self.dataset.indices_by_pdb[pdb_id])
            if self.shuffle:
                generator.shuffle(indices)
            ordered_indices.extend(indices)
        for start in range(0, len(ordered_indices), self.batch_size):
            batch = ordered_indices[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        """返回当前划分的物理批次数量。"""

        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)


def collate_candidates(items: list[dict[str, Any]]) -> dict[str, Any]:
    """填充 A/P/V/PP 与身份维度，并把所有 LigandObject 图连接成一个图批次。"""

    if not items:
        raise ValueError("候选批次不能为空。")
    batch: dict[str, Any] = {
        "metadata": [item["metadata"] for item in items],
        "voxel_input": torch.from_numpy(np.stack([item["voxel_input"] for item in items])),
        "class_target": torch.tensor([item["class_target"] for item in items], dtype=torch.long),
        "ligand_area_target": torch.from_numpy(
            np.stack([item["ligand_area_target"] for item in items])
        ),
        "binding_area_target": torch.from_numpy(
            np.stack([item["binding_area_target"] for item in items])
        ),
        "full_candidate_voxel_count": torch.tensor(
            [item["full_candidate_voxel_count"] for item in items],
            dtype=torch.long,
        ),
    }
    for prefix in ("a", "p", "v", "pp"):
        padded, mask = _pad_float_sequences([item[f"{prefix}_tokens"] for item in items])
        batch[f"{prefix}_tokens"] = padded
        batch[f"{prefix}_mask"] = mask
    for prefix in ("v", "pp"):
        batch[f"{prefix}_index_zyx"] = _pad_index_sequences(
            [item[f"{prefix}_index_zyx"] for item in items]
        )
    batch.update(_collate_identities(items))
    return batch


def _pad_float_sequences(arrays: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """把一组 ``(N_i,F)`` float 数组补零到共同 token 数并返回有效掩码。"""

    tensors = [torch.from_numpy(np.asarray(array, dtype=np.float32)) for array in arrays]
    feature_dims = {int(tensor.shape[1]) for tensor in tensors if tensor.ndim == 2}
    if len(feature_dims) != 1 or any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("同一模态的 token 必须都是具有相同特征宽度的二维数组。")
    padded = pad_sequence(tensors, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([tensor.shape[0] for tensor in tensors], dtype=torch.long)
    positions = torch.arange(padded.shape[1], dtype=torch.long)[None, :]
    return padded, positions < lengths[:, None]


def _pad_index_sequences(arrays: list[np.ndarray]) -> torch.Tensor:
    """把一组 ``(N_i,3)`` ZYX 索引补零到对应模态共同 token 数。"""

    tensors = [torch.from_numpy(np.asarray(array, dtype=np.int64)) for array in arrays]
    if any(tensor.ndim != 2 or tensor.shape[1] != 3 for tensor in tensors):
        raise ValueError("体素 token 索引必须为 (N,3)。")
    return pad_sequence(tensors, batch_first=True, padding_value=0)


def _collate_identities(items: list[dict[str, Any]]) -> dict[str, Any]:
    """填充语言表示并连接与每个有效身份一一对应的配体图。"""

    batch_size = len(items)
    identity_count = max(len(item["identity_object_keys"]) for item in items)
    language = torch.zeros(
        (batch_size, identity_count, LANGUAGE_EMBEDDING_DIM),
        dtype=torch.float32,
    )
    identity_mask = torch.zeros((batch_size, identity_count), dtype=torch.bool)
    identity_object_keys: list[list[str]] = []

    elements: list[torch.Tensor] = []
    charges: list[torch.Tensor] = []
    residues: list[torch.Tensor] = []
    node_continuous: list[torch.Tensor] = []
    node_graph_index: list[torch.Tensor] = []
    edge_indices: list[torch.Tensor] = []
    edge_features: list[torch.Tensor] = []
    graph_to_sample: list[int] = []
    graph_to_identity: list[int] = []
    node_offset = 0
    graph_index = 0

    for sample_index, item in enumerate(items):
        object_keys = list(item["identity_object_keys"])
        graphs = list(item["ligand_graphs"])
        embeddings = np.asarray(item["language_embeddings"], dtype=np.float32)
        if len(object_keys) != len(graphs) or embeddings.shape != (
            len(object_keys),
            LANGUAGE_EMBEDDING_DIM,
        ):
            raise ValueError("身份名称、语言表示和配体图数量不一致。")
        identity_object_keys.append(object_keys)
        if object_keys:
            language[sample_index, : len(object_keys)] = torch.from_numpy(embeddings)
            identity_mask[sample_index, : len(object_keys)] = True
        for identity_index, graph in enumerate(graphs):
            node_count = int(graph.element.shape[0])
            elements.append(torch.from_numpy(graph.element))
            charges.append(torch.from_numpy(graph.charge))
            residues.append(torch.from_numpy(graph.residue_id))
            node_continuous.append(torch.from_numpy(graph.node_continuous))
            node_graph_index.append(torch.full((node_count,), graph_index, dtype=torch.long))
            edge_indices.append(torch.from_numpy(graph.edge_index) + node_offset)
            edge_features.append(torch.from_numpy(graph.edge_features))
            graph_to_sample.append(sample_index)
            graph_to_identity.append(identity_index)
            node_offset += node_count
            graph_index += 1

    graph_batch = _empty_or_concatenated_graph(
        elements,
        charges,
        residues,
        node_continuous,
        node_graph_index,
        edge_indices,
        edge_features,
    )
    return {
        "language_embeddings": language,
        "identity_mask": identity_mask,
        "identity_target": torch.tensor(
            [item["identity_target"] for item in items],
            dtype=torch.long,
        ),
        "identity_object_keys": identity_object_keys,
        "graph_to_sample": torch.tensor(graph_to_sample, dtype=torch.long),
        "graph_to_identity": torch.tensor(graph_to_identity, dtype=torch.long),
        **graph_batch,
    }


def _empty_or_concatenated_graph(
    elements: list[torch.Tensor],
    charges: list[torch.Tensor],
    residues: list[torch.Tensor],
    node_continuous: list[torch.Tensor],
    node_graph_index: list[torch.Tensor],
    edge_indices: list[torch.Tensor],
    edge_features: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """连接图字段；候选预训练没有身份时返回形状明确的空张量。"""

    if not elements:
        return {
            "graph_element": torch.empty(0, dtype=torch.long),
            "graph_charge": torch.empty(0, dtype=torch.long),
            "graph_residue_id": torch.empty(0, dtype=torch.long),
            "graph_node_continuous": torch.empty((0, 15), dtype=torch.float32),
            "graph_node_graph_index": torch.empty(0, dtype=torch.long),
            "graph_edge_index": torch.empty((2, 0), dtype=torch.long),
            "graph_edge_features": torch.empty((0, 10), dtype=torch.float32),
        }
    return {
        "graph_element": torch.cat(elements, dim=0).long(),
        "graph_charge": torch.cat(charges, dim=0).long(),
        "graph_residue_id": torch.cat(residues, dim=0).long(),
        "graph_node_continuous": torch.cat(node_continuous, dim=0).float(),
        "graph_node_graph_index": torch.cat(node_graph_index, dim=0),
        "graph_edge_index": torch.cat(edge_indices, dim=1).long(),
        "graph_edge_features": torch.cat(edge_features, dim=0).float(),
    }
