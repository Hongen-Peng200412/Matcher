"""按可复现顺序采样候选, 并补齐 token、身份和 SMILES 图字段.

DataLoader 使用 spawn 且不保留跨迭代工作进程. 每个完整训练区段或 validation 遍历结束后
工作进程退出, 使训练与 validation 工作进程不并存.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sized
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, get_worker_info

from .candidate_data import CandidateDataset
from .contracts import CandidateBatch, CandidateItem
from ..data.schema import LANGUAGE_EMBEDDING_DIM


def _pad_float(values: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    沿第一维补零一组同宽浮点二维数组, 并返回有效位置掩码.

    - values: 非空二维数组列表; 第二维宽度必须相同, 第一维允许为零.
    - return[0]: float32 张量, (B, N_max, D), 无效位置补零, 特征.
    - return[1]: bool 张量, (B, N_max), 原数组覆盖位置为 True, mask.
    """
    # int, N_max; 批内数组第一轴的最大长度.
    maximum = max(len(value) for value in values)
    # int, D; 批内所有二维数组必须共享的特征宽度.
    width = values[0].shape[1]
    # float32 张量, (B, N_max, D); 每个输入数组写入对应样本的前缀.
    output = torch.zeros((len(values), maximum, width), dtype=torch.float32)
    # bool 张量, (B, N_max); True 表示 output 第二轴来自原数组, False 表示补零.
    mask = torch.zeros((len(values), maximum), dtype=torch.bool)
    for index, value in enumerate(values):
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError("批内 token 数组必须共享二维特征宽度。")
        output[index, : len(value)] = torch.from_numpy(value).float()
        mask[index, : len(value)] = True
    return output, mask


def _pad_integer(values: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    沿第一维补零一组 int64 二维索引, 并返回有效位置掩码.

    - values: 非空二维索引数组列表; 第二维宽度必须相同, 第一维允许为零.
    - return[0]: int64 张量, (B, N_max, D), 无效位置补零, 特征.
    - return[1]: bool 张量, (B, N_max), 原数组覆盖位置为 True, mask.
    """
    # int, N_max; 批内索引数组第一轴的最大长度.
    maximum = max(len(value) for value in values)
    # int, D; 批内所有二维索引数组必须共享的宽度.
    width = values[0].shape[1]
    # int64 张量, (B, N_max, D); 每个输入索引数组写入对应样本的前缀.
    output = torch.zeros((len(values), maximum, width), dtype=torch.long)
    # bool 张量, (B, N_max); True 表示 output 第二轴来自原数组, False 表示补零.
    mask = torch.zeros((len(values), maximum), dtype=torch.bool)
    for index, value in enumerate(values):
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError("批内索引数组必须共享二维特征宽度。")
        output[index, : len(value)] = torch.from_numpy(value).long()
        mask[index, : len(value)] = True
    return output, mask


# ================================================================================================


class ShuffledBatchSampler(Sampler[list[int]]):
    """
    用 seed+epoch 可复现洗牌, 再切成固定物理批次.

    - dataset: 只通过 len 读取候选总数.
    - batch_size: 每个物理批次的最大候选数.
    - shuffle: 是否洗牌.
    - drop_last: 是否丢弃不足一个物理批次的末批.
    - seed: 基准随机种子.
    - sample_count_multiple: 可选截短倍数; 用于保持梯度累积边界完整.
    """
    def __init__(
        self,
        dataset: Sized,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int,
        sample_count_multiple: int | None,
    ) -> None:
        """
        验证批量约束并把首次 epoch 固定为零.

        - dataset: 只要求实现 len 的候选集合.
        - batch_size: 每个物理批次的最大候选数.
        - shuffle: True 时按 seed+epoch 洗牌全部候选下标.
        - drop_last: True 时不生成不足 batch_size 的末批.
        - seed: 洗牌基准种子.
        - sample_count_multiple: 可选候选截短倍数; 正式训练用它保持梯度累积完整.
        """
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数。")
        if sample_count_multiple is not None and (
            sample_count_multiple <= 0 or sample_count_multiple % batch_size != 0
        ):
            raise ValueError("sample_count_multiple 必须是 batch_size 的正整数倍。")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.sample_count_multiple = sample_count_multiple
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """
        更新下一次迭代使用的洗牌周期.

        - epoch: int, 与基准 seed 相加后播种当前周期的独立洗牌, 通常从 0 递增.
        - return: None, 只更新 self.epoch.
        """
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        """
        生成当前 epoch 的 Dataset 下标批次.

        - return: 依次产生 list[int]; 每个列表是一个物理批次的 Dataset 下标.

        洗牌只依赖固定 seed 和当前 epoch; 可选截短发生在洗牌后, 因此每个 epoch
        可能舍弃不同候选, 但总样本数和梯度累积边界保持不变.
        """
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        if self.sample_count_multiple is not None:
            usable = len(indices) // self.sample_count_multiple * self.sample_count_multiple
            indices = indices[:usable]
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        """
        返回应用截短与末批规则后的物理批次数.

        - return: int, sample_count_multiple 截短后按 batch_size 切分的批次数; drop_last=False 时向上取整.
        """
        sample_count = len(self.dataset)
        if self.sample_count_multiple is not None:
            sample_count = sample_count // self.sample_count_multiple * self.sample_count_multiple
        return sample_count // self.batch_size if self.drop_last else math.ceil(sample_count / self.batch_size)


def initialize_candidate_worker(_: int) -> None:
    """
    在 DataLoader 工作进程中初始化该 Dataset 副本的全局图包.

    - _: int, PyTorch 提供的工作进程编号; 初始化逻辑不依赖具体编号.
    - return: None, 当前工作进程的 CandidateDataset.graph_store 完整载入一次图包.
    """
    information = get_worker_info()
    if information is None or not isinstance(information.dataset, CandidateDataset):
        raise RuntimeError("initialize_candidate_worker 只能由 CandidateDataset DataLoader 调用。")
    information.dataset.initialize_worker()


def collate_candidates(items: list[CandidateItem]) -> CandidateBatch:
    """
    把非空 CandidateItem 列表补齐并连接全部有效身份图.

    - items: list[CandidateItem], 非空候选列表; 列表顺序定义 CandidateBatch 的 B 轴.
    - return: CandidateBatch, 完整字段、形状、补零与连接图索引契约见 contracts.CandidateBatch.

    token 补零仅发生在批内最大长度之后; 对应 mask 标记有效位置. 每个有效身份恰好
    连接一张图, graph_to_sample 和 graph_to_identity 把图表示写回身份槽.
    """
    if not items:
        raise ValueError("collate_candidates 不接受空列表。")
    a_tokens, a_mask = _pad_float([item["a_tokens"] for item in items])
    p_tokens, p_mask = _pad_float([item["p_tokens"] for item in items])
    v_tokens, v_mask = _pad_float([item["v_tokens"] for item in items])
    v_index, _ = _pad_integer([item["v_index_zyx"] for item in items])
    pp_tokens, pp_mask = _pad_float([item["pp_tokens"] for item in items])
    pp_index, _ = _pad_integer([item["pp_index_zyx"] for item in items])
    # int, I_max; 批内单个候选所属 PDB 的最大有效身份数.
    identity_count = max(len(item["identity_smiles"]) for item in items)
    # float32 张量, (B, I_max, 768); 每个候选的有效语言向量写入前缀, 其余槽补零.
    language = torch.zeros((len(items), identity_count, LANGUAGE_EMBEDDING_DIM), dtype=torch.float32)
    # bool 张量, (B, I_max); True 表示对应 language 的真实身份槽.
    identity_mask = torch.zeros((len(items), identity_count), dtype=torch.bool)

    # 下列两个整数列表逐图对齐, 分别保存每张图所属样本和该样本内身份槽.
    graph_to_sample: list[int] = []
    graph_to_identity: list[int] = []
    # 下列节点与边张量列表按样本、身份槽顺序连接; 同一列表位置对应一张精确 SMILES 图.
    elements: list[torch.Tensor] = []
    charges: list[torch.Tensor] = []
    atom_rings: list[torch.Tensor] = []
    node_graph_indices: list[torch.Tensor] = []
    edge_indices: list[torch.Tensor] = []
    edge_features: list[torch.Tensor] = []
    # int, 当前图首个节点在连接节点轴上的偏移; 加到该图局部 edge_index.
    node_offset = 0
    # int, 当前图在 G 轴上的下标; 写入每个节点的 graph_node_graph_index.
    graph_index = 0

    for sample_index, item in enumerate(items):
        smiles_count = len(item["identity_smiles"])
        if item["language_embeddings"].shape != (smiles_count, LANGUAGE_EMBEDDING_DIM):
            raise ValueError("language_embeddings 未与 identity_smiles 对齐。")
        if len(item["smiles_graphs"]) not in (0, smiles_count):
            raise ValueError("SMILES 图列表必须为空，或与 identity_smiles 逐槽对齐。")
        if not -1 <= item["identity_target"] < smiles_count:
            raise ValueError("identity_target 超出当前 PDB 身份轴。")
        if smiles_count:
            language[sample_index, :smiles_count] = torch.from_numpy(item["language_embeddings"])
            identity_mask[sample_index, :smiles_count] = True

        for identity_index, graph in enumerate(item["smiles_graphs"]):
            node_count = len(graph.element)
            if node_count <= 0 or graph.atom_in_ring.shape != (node_count, 4):
                raise ValueError("每张 SMILES 图必须含原子且 atom_in_ring 与原子轴对齐。")
            if graph.edge_index.shape != (2, len(graph.edge_features)) or graph.edge_features.shape[1:] != (9,):
                raise ValueError("SMILES 图的有向边和 9 维边特征未对齐。")
            graph_to_sample.append(sample_index)
            graph_to_identity.append(identity_index)
            elements.append(torch.from_numpy(graph.element))
            charges.append(torch.from_numpy(graph.charge))
            atom_rings.append(torch.from_numpy(graph.atom_in_ring))
            node_graph_indices.append(torch.full((node_count,), graph_index, dtype=torch.long))
            edge_indices.append(torch.from_numpy(graph.edge_index) + node_offset)
            edge_features.append(torch.from_numpy(graph.edge_features))
            node_offset += node_count
            graph_index += 1
    if elements:
        # dict[str, torch.Tensor], 六个连接图批次字段; 节点与边均按前述图顺序连接.
        graph_fields = {
            "graph_element": torch.cat(elements).long(),
            "graph_charge": torch.cat(charges).long(),
            "graph_atom_in_ring": torch.cat(atom_rings).float(),
            "graph_node_graph_index": torch.cat(node_graph_indices).long(),
            "graph_edge_index": torch.cat(edge_indices, dim=1).long(),
            "graph_edge_features": torch.cat(edge_features).float(),
        }
    else:
        # dict[str, torch.Tensor], G=0 时使用的同 dtype、同固定尾维空连接图字段.
        graph_fields = {
            "graph_element": torch.empty(0, dtype=torch.long),
            "graph_charge": torch.empty(0, dtype=torch.long),
            "graph_atom_in_ring": torch.empty((0, 4), dtype=torch.float32),
            "graph_node_graph_index": torch.empty(0, dtype=torch.long),
            "graph_edge_index": torch.empty((2, 0), dtype=torch.long),
            "graph_edge_features": torch.empty((0, 9), dtype=torch.float32),
        }
    # dict[str, Any], contracts.CandidateBatch 的全部字段; token 与身份轴已补齐, 图轴已连接.
    batch: dict[str, Any] = {
        "metadata": [item["metadata"] for item in items],
        "voxel_input": torch.from_numpy(np.stack([item["voxel_input"] for item in items])),
        "foreground_target": torch.tensor([item["foreground_target"] for item in items], dtype=torch.bool),
        "ligand_area_target": torch.from_numpy(np.stack([item["ligand_area_target"] for item in items])),
        "binding_area_target": torch.from_numpy(np.stack([item["binding_area_target"] for item in items])),
        "full_candidate_voxel_count": torch.tensor([item["full_candidate_voxel_count"] for item in items], dtype=torch.long),
        "candidate_manual_features": torch.from_numpy(np.stack([item["candidate_manual_features"] for item in items])).float(),
        "candidate_sampling_features": torch.from_numpy(np.stack([item["candidate_sampling_features"] for item in items])).float(),
        "a_tokens": a_tokens,
        "a_mask": a_mask,
        "p_tokens": p_tokens,
        "p_mask": p_mask,
        "v_tokens": v_tokens,
        "v_mask": v_mask,
        "v_index_zyx": v_index,
        "pp_tokens": pp_tokens,
        "pp_mask": pp_mask,
        "pp_index_zyx": pp_index,
        "language_embeddings": language,
        "identity_mask": identity_mask,
        "identity_target": torch.tensor([item["identity_target"] for item in items], dtype=torch.long),
        "identity_smiles": [item["identity_smiles"] for item in items],
        "graph_to_sample": torch.tensor(graph_to_sample, dtype=torch.long),
        "graph_to_identity": torch.tensor(graph_to_identity, dtype=torch.long),
        **graph_fields,
    }
    return cast(CandidateBatch, batch)


def build_candidate_loader(
    dataset: CandidateDataset,
    sampler: ShuffledBatchSampler | list[list[int]],
    num_workers: int,
    prefetch_factor: int,
    worker_generator_seed: int,
) -> DataLoader[CandidateBatch]:
    """
    构造候选 DataLoader; 始终启用页锁定内存, num_workers>0 时使用 spawn 且不跨遍历保留工作进程.

    - dataset: 单一 SMILES 身份 CandidateDataset.
    - sampler: ShuffledBatchSampler 或已经冻结顺序的物理批次下标列表; 唯一决定候选顺序和物理批次.
    - num_workers: 非负工作进程数; 为零时在主进程读取.
    - prefetch_factor: 工作进程预取批次数; 只在 num_workers>0 时传给 PyTorch.
    - worker_generator_seed: DataLoader 工作进程播种使用的独立种子.
    - return: DataLoader[CandidateBatch], sampler 决定批次顺序, collate_candidates 生成每个 CandidateBatch.
    """
    if num_workers < 0:
        raise ValueError("num_workers 不能为负数。")
    if num_workers > 0 and prefetch_factor <= 0:
        raise ValueError("prefetch_factor 必须为正整数。")
    generator = torch.Generator().manual_seed(worker_generator_seed)
    keyword: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "collate_fn": collate_candidates,
        "pin_memory": True,
        "generator": generator,
    }
    if num_workers > 0:
        keyword.update(
            {
                "persistent_workers": False,
                "prefetch_factor": prefetch_factor,
                "multiprocessing_context": "spawn",
                "worker_init_fn": initialize_candidate_worker,
            }
        )
    else:
        dataset.initialize_worker()
    return cast(DataLoader[CandidateBatch], DataLoader(**keyword))
