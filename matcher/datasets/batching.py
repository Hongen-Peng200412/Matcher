"""排列全局候选轴, 并把变长 token 与配体图整理成固定物理批次. 

ShuffledBatchSampler 只决定候选下标顺序和物理批次边界; collate_candidates 为四种
token 和身份轴补零、生成有效掩码并连接配体图; build_candidate_loader 使用 spawn
非持久工作进程和独立 Torch 随机生成器. 该模块不改变单候选的科学字段或监督口径. 
"""

from __future__ import annotations

import math
import random
from collections.abc import Sized
from typing import Any, Iterator, cast

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler

from .candidate_data import CandidateDataset
from .contracts import CandidateBatch, CandidateItem
from ..data.schema import (
    CANDIDATE_MANUAL_FEATURE_DIM,
    CANDIDATE_SAMPLING_FEATURE_DIM,
    LANGUAGE_EMBEDDING_DIM,
    PP_TOKEN_DIM,
    V_TOKEN_DIM,
)


DATALOADER_MULTIPROCESSING_CONTEXT = "spawn"


def _pad_float_sequences(arrays: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """把一批同宽浮点 token 表补零到共同长度并生成有效掩码. 

    输入参数：
        - arrays: 非空 NumPy 数组列表; 第 i 项形状必须为 (N_i, F), 所有项的 F 相同, N_i 可以为零. 

    中间字段：  # NOTE: 函数的docstring可以不用写"中间字段"。专心写输入、输出，以及可能的运算过程和示例
        - tensors: float32 张量列表; 第 i 项形状为 (N_i, F), 由对应 NumPy 数组转换. 
        - feature_dims: 整数集合; 收集所有二维张量的特征宽度, 合法输入恰有一个元素 F. 
        - lengths: int64 张量, (B,); 每个样本补零前的 token 数 N_i. 
        - positions: int64 张量, (1, N_max); 批内统一位置下标, 用于和 lengths 比较生成掩码. 

    返回字段：
        - padded: float32 张量, (B, N_max, F); 每项前 N_i 个位置保存原 token, 其余位置为 0. 
        - mask: bool 张量, (B, N_max); 第 i 项前 N_i 个位置为 True, 补零位置为 False. 
    """

    # list[torch.Tensor], 每项为 float32 (N_i, F), 保留输入列表顺序. 
    tensors = [torch.from_numpy(np.asarray(array, dtype=np.float32)) for array in arrays]
    # set[int], 全部二维 token 表出现过的特征宽度; 合法批次必须恰好一种宽度. 
    feature_dims = {int(tensor.shape[1]) for tensor in tensors if tensor.ndim == 2}
    if len(feature_dims) != 1 or any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("同一模态的 token 必须都是具有相同特征宽度的二维数组. ")
    # float32, (B, N_max, F), 沿 token 轴右侧补零后的批次张量. 
    padded = pad_sequence(tensors, batch_first=True, padding_value=0.0)
    # int64, (B,), 每个样本在补零前的有效 token 数. 
    lengths = torch.tensor([tensor.shape[0] for tensor in tensors], dtype=torch.long)
    # int64, (1, N_max), 统一 token 位置, 用于广播生成逐样本有效掩码. 
    positions = torch.arange(padded.shape[1], dtype=torch.long)[None, :]
    return padded, positions < lengths[:, None]


def _pad_index_sequences(arrays: list[np.ndarray]) -> torch.Tensor:
    """把一批 Z/Y/X 索引表补零到共同 token 长度. 

    输入参数：
        - arrays: 非空 NumPy 数组列表; 第 i 项必须为 (N_i, 3), 与同一样本的 V 或 PP token 第一轴对齐. 

    中间字段：
        - tensors: int64 张量列表; 第 i 项形状为 (N_i, 3), 最后一轴依次为裁剪内 Z/Y/X. 

    返回值：
        - padded_indices: int64 张量, (B, N_max, 3); 每项前 N_i 个位置保存原索引, 其余位置为 0; 只有对应模态 mask 为 True 的位置可读. 
    """

    # list[torch.Tensor], 每项为 int64 (N_i, 3) 的裁剪内 Z/Y/X 索引. 
    tensors = [torch.from_numpy(np.asarray(array, dtype=np.int64)) for array in arrays]
    if any(tensor.ndim != 2 or tensor.shape[1] != 3 for tensor in tensors):
        raise ValueError("体素 token 索引必须为 (N,3). ")
    return pad_sequence(tensors, batch_first=True, padding_value=0)


def _collate_identities(items: list[CandidateItem]) -> dict[str, Any]:
    """补齐数值身份轴, 并连接与每个有效联合身份逐槽对应的配体图. 

    输入参数：
        - items: 非空 CandidateItem 列表; 顺序定义物理批次的样本轴. 
        - items[*].identity_object_keys: 长度 I_b 的字符串列表; 第 b 个样本的 LigandObject 模板键. 
        - items[*].identity_prepared_smiles: 长度 I_b 的非空字符串列表; 与 object_key 共同定义联合身份. 
        - items[*].language_embeddings: 数组, (I_b, 768); 当前语言家族的身份表示, 转换为 float32. 
        - items[*].identity_class_index: 数组, (I_b,); 身份前景类别, 转换为 int64. 
        - items[*].identity_target: 整数; 必须位于 [-1, I_b), -1 表示当前样本没有可监督身份目标. 
        - items[*].ligand_graphs: 长度 I_b 的 LigandGraph 列表; 每个身份槽恰好提供一张模板图. 
        - items[*].ligand_graphs[*].element: int64 数组, (N_atom_g,), 模板原子序数. 
        - items[*].ligand_graphs[*].charge: int64 数组, (N_atom_g,), 模板原子形式电荷. 
        - items[*].ligand_graphs[*].residue_id: int64 数组, (N_atom_g,), 模板内部残基编号. 
        - items[*].ligand_graphs[*].node_continuous: float32 数组, (N_atom_g, 15), 节点连续特征. 
        - items[*].ligand_graphs[*].edge_index: int64 数组, (2, N_edge_g), 图内有向边端点. 
        - items[*].ligand_graphs[*].edge_features: float32 数组, (N_edge_g, 10), 与边逐项对齐的特征. 

    返回字段：
        - language_embeddings: float32 张量, (B, I_max, 768); 有效身份槽保存输入向量, 补零槽为 0. 
        - identity_mask: bool 张量, (B, I_max); 每个样本前 I_b 个位置为 True, 补零位置为 False. 
        - identity_class_index: int64 张量, (B, I_max); 有效槽保存输入类别, 补零槽为 -1. 
        - identity_target: int64 张量, (B,); 逐样本复制输入目标下标. 
        - identity_object_keys: 长度 B 的嵌套字符串列表; 第 b 项保持原长 I_b, 不补到 I_max. 
        - identity_prepared_smiles: 长度 B 的嵌套字符串列表; 第 b 项保持原长 I_b, 并与 object_key 逐槽对齐. 

        - graph_to_sample: int64 张量, (G,); 第 g 项是该图所属样本下标. 
        - graph_to_identity: int64 张量, (G,); 第 g 项是该图在所属样本中的身份下标. 

        - graph_element: int64 张量, (N_node,); 所有图连接后的原子序数. 
        - graph_charge: int64 张量, (N_node,); 与 graph_element 逐节点对齐的形式电荷. 
        - graph_residue_id: int64 张量, (N_node,); 与 graph_element 逐节点对齐的模板内部残基编号. 
        - graph_node_continuous: float32 张量, (N_node, 15); 所有图连接后的节点连续特征. 
        - graph_node_graph_index: int64 张量, (N_node,); 把每个节点映射到连接前图下标 [0, G). 
        - graph_edge_index: int64 张量, (2, N_directed_edge); 每张图端点加 node_offset 后沿第二轴连接. 
        - graph_edge_features: float32 张量, (N_directed_edge, 10); 与 graph_edge_index 第二轴逐边对齐. 

    其中 I_max=max(I_b), G=sum(I_b), N_node 和 N_directed_edge 是全部图的节点与有向边总数. 相同 object_key、不同 prepared_smiles 仍是两个身份槽和两个图映射项; 图缓存预算允许且写入成功时, 两项可复用同一实例. 该函数核验身份列表长度、数值数组形状、smiles 非空和目标范围, 但不再次核验目标槽的类别或联合键是否与 metadata 相同. 
    # NOTE: "图缓存预算允许且写入成功时, 两项可复用同一实例"————没有必要为此写复杂逻辑！
    """

    # int, 物理批次候选数 B. 
    batch_size = len(items)
    # int, 批内最大身份数 I_max; 候选预训练批次可为 0. 
    identity_count = max(len(item["identity_object_keys"]) for item in items)
    # float32, (B, I_max, 768), 身份语言表示的补零容器. 
    language = torch.zeros(
        (batch_size, identity_count, LANGUAGE_EMBEDDING_DIM),
        dtype=torch.float32,
    )
    # bool, (B, I_max), 有效身份槽掩码; 填充前全部为 False. 
    identity_mask = torch.zeros((batch_size, identity_count), dtype=torch.bool)
    # int64, (B, I_max), 身份类别补零容器; 无效槽固定为 -1. 
    identity_class_index = torch.full((batch_size, identity_count), -1, dtype=torch.long)
    # list[list[str]], 逐样本保留原长 I_b 的 object_key, 不做字符串补齐. 
    identity_object_keys: list[list[str]] = []
    # list[list[str]], 逐样本保留原长 I_b 的顶层 PreparedLigand.smiles. 
    identity_prepared_smiles: list[list[str]] = []

    # 以下七个列表逐图累积节点或边张量; 相同列表位置属于同一身份图. 
    # list[torch.Tensor], 每项为 int64 (N_atom_g,) 的模板原子序数. 
    elements: list[torch.Tensor] = []
    # list[torch.Tensor], 每项为 int64 (N_atom_g,) 的模板原子形式电荷. 
    charges: list[torch.Tensor] = []
    # list[torch.Tensor], 每项为 int64 (N_atom_g,) 的模板内部残基编号. 
    residues: list[torch.Tensor] = []
    # list[torch.Tensor], 每项为 float32 (N_atom_g, 15) 的节点连续特征. 
    node_continuous: list[torch.Tensor] = []
    # list[torch.Tensor], 每项为 int64 (N_atom_g,) 且全部填当前全局图下标. 
    node_graph_index: list[torch.Tensor] = []
    # list[torch.Tensor], 每项为 int64 (2, N_edge_g); 端点已加此前图的节点总数. 
    edge_indices: list[torch.Tensor] = []
    # list[torch.Tensor], 每项为 float32 (N_edge_g, 10) 的有向边特征. 
    edge_features: list[torch.Tensor] = []
    # list[int], 逐全局图保存所属样本下标. 
    graph_to_sample: list[int] = []
    # list[int], 逐全局图保存所属样本内身份下标. 
    graph_to_identity: list[int] = []
    # int, 已经累积的节点总数; 用于把下一张图的局部端点变成连接后全局端点. 
    node_offset = 0
    # int, 下一张图的批内全局图下标, 同时写入 graph_node_graph_index. 
    graph_index = 0

    for sample_index, item in enumerate(items):
        # list[str], 当前样本原长 I_b 的 LigandObject 模板键副本. 
        object_keys = list(item["identity_object_keys"])
        # list[str], 当前样本原长 I_b 的顶层 PreparedLigand.smiles 副本. 
        prepared_smiles_values = list(item["identity_prepared_smiles"])
        # list[LigandGraph], 当前样本每个身份槽对应的模板图引用. 
        graphs = list(item["ligand_graphs"])
        # float32, (I_b, 768), 当前样本的身份语言表示. 
        embeddings = np.asarray(item["language_embeddings"], dtype=np.float32)
        # int64, (I_b,), 当前样本身份轴的前景类别. 
        class_indices = np.asarray(item["identity_class_index"], dtype=np.int64)
        if (
            len(object_keys) != len(prepared_smiles_values)
            or len(object_keys) != len(graphs)
            or embeddings.shape != (len(object_keys), LANGUAGE_EMBEDDING_DIM)
            or class_indices.shape != (len(object_keys),)
        ):
            raise ValueError(
                "身份模板键、PreparedLigand.smiles、类别、语言表示和配体图数量不一致. "
            )
        if any(not value.strip() for value in prepared_smiles_values):
            raise ValueError("有效身份的 PreparedLigand.smiles 必须是非空字符串. ")
        # int, 当前样本身份真值下标; -1 或 [0, I_b) 内的位置. 
        target = int(item["identity_target"])
        if target < -1 or target >= len(object_keys):
            raise ValueError(
                f"identity_target={target} 超出当前样本的 {len(object_keys)} 个身份. "
            )
        identity_object_keys.append(object_keys)
        identity_prepared_smiles.append(prepared_smiles_values)
        if object_keys:
            language[sample_index, : len(object_keys)] = torch.from_numpy(embeddings)
            identity_mask[sample_index, : len(object_keys)] = True
            identity_class_index[sample_index, : len(object_keys)] = torch.from_numpy(
                class_indices
            )
        for identity_index, graph in enumerate(graphs):
            # int, 当前身份图的节点数 N_atom_g, 用于节点映射和下一图端点偏移. 
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

    # dict[str, torch.Tensor], 连接后的七个图字段; G=0 时为形状明确的空张量. 
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
        "identity_class_index": identity_class_index,
        "identity_target": torch.tensor(
            [item["identity_target"] for item in items],
            dtype=torch.long,
        ),
        "identity_object_keys": identity_object_keys,
        "identity_prepared_smiles": identity_prepared_smiles,
        "graph_to_sample": torch.tensor(graph_to_sample, dtype=torch.long),
        "graph_to_identity": torch.tensor(graph_to_identity, dtype=torch.long),
        **graph_batch,
    }


def _validate_token_index_alignment(items: list[CandidateItem]) -> None:
    """核对每个样本 V、PP token 与局部索引的长度和固定宽度. 

    输入参数：
        - items: CandidateItem 列表; collate_candidates 已要求列表非空. 
        - items[*].v_tokens: NumPy 数组; 必须为 (N_V, V_TOKEN_DIM), 当前 V_TOKEN_DIM 为 52. 
        - items[*].v_index_zyx: NumPy 数组; 必须为 (N_V, 3), 与 v_tokens 第一轴逐项对齐. 
        - items[*].pp_tokens: NumPy 数组; 必须为 (N_PP, PP_TOKEN_DIM), 当前 PP_TOKEN_DIM 为 4. 
        - items[*].pp_index_zyx: NumPy 数组; 必须为 (N_PP, 3), 与 pp_tokens 第一轴逐项对齐. 

    返回值：
        - return: None; 全部长度和宽度相容时返回, 失败信息包含样本位置和 V 或 PP 名称. 

    本函数不转换 dtype, 也不核验有限值、索引范围、A/P 固定宽度或体素与区域目标形状; 这些字段依赖正式 CandidateDataset 契约. 通用补齐和堆叠操作可能接受批内统一但科学宽度错误的输入, 不能替代固定 schema 门禁. 
    """

    for item_index, item in enumerate(items):
        # tuple[tuple[str, np.ndarray, np.ndarray, int], ...], 当前样本 V/PP 的 token、索引和固定宽度配对. 
        pairs = (
            ("V", item["v_tokens"], item["v_index_zyx"], V_TOKEN_DIM),
            ("PP", item["pp_tokens"], item["pp_index_zyx"], PP_TOKEN_DIM),
        )
        for name, tokens, indices, width in pairs:
            if tokens.ndim != 2 or tokens.shape[1] != width:
                raise ValueError(
                    f"items[{item_index}] 的 {name} token 必须为 (N,{width}). "
                )
            if indices.shape != (tokens.shape[0], 3):
                raise ValueError(
                    f"items[{item_index}] 的 {name} token 与 ZYX 索引数量不一致. "
                )


def _empty_or_concatenated_graph(
    elements: list[torch.Tensor],
    charges: list[torch.Tensor],
    residues: list[torch.Tensor],
    node_continuous: list[torch.Tensor],
    node_graph_index: list[torch.Tensor],
    edge_indices: list[torch.Tensor],
    edge_features: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """沿节点轴或边轴连接逐图字段, 并处理没有身份图的批次. 

    输入参数：
        - elements: 张量列表; 第 g 项为 (N_atom_g,) 的模板原子序数. 
        - charges: 张量列表; 第 g 项为 (N_atom_g,) 的模板原子形式电荷. 
        - residues: 张量列表; 第 g 项为 (N_atom_g,) 的模板内部残基编号. 
        - node_continuous: 张量列表; 第 g 项为 (N_atom_g, 15) 的节点连续特征. 
        - node_graph_index: 张量列表; 第 g 项为 (N_atom_g,) 且值为全局图下标 g. 
        - edge_indices: 张量列表; 第 g 项为 (2, N_edge_g), 端点已经加该图之前的节点总数. 
        - edge_features: 张量列表; 第 g 项为 (N_edge_g, 10), 与 edge_indices[g] 第二轴对齐. 

    返回字段：
        - graph_element: int64 张量, (N_node,); 沿节点轴连接 elements. 
        - graph_charge: int64 张量, (N_node,); 沿节点轴连接 charges. 
        - graph_residue_id: int64 张量, (N_node,); 沿节点轴连接 residues. 
        - graph_node_continuous: float32 张量, (N_node, 15); 沿节点轴连接 node_continuous. 
        - graph_node_graph_index: int64 张量, (N_node,); 沿节点轴连接 node_graph_index. 
        - graph_edge_index: int64 张量, (2, N_directed_edge); 沿边轴连接 edge_indices. 
        - graph_edge_features: float32 张量, (N_directed_edge, 10); 沿边轴连接 edge_features. 

    空图边界：
        - graph_element: elements 为空时返回 int64 (0,). 
        - graph_charge: elements 为空时返回 int64 (0,). 
        - graph_residue_id: elements 为空时返回 int64 (0,). 
        - graph_node_continuous: elements 为空时返回 float32 (0, 15). 
        - graph_node_graph_index: elements 为空时返回 int64 (0,). 
        - graph_edge_index: elements 为空时返回 int64 (2, 0). 
        - graph_edge_features: elements 为空时返回 float32 (0, 10). 

    该内部函数依赖调用者保证七个列表按图对齐, 不单独比较列表长度或逐图形状. 
    """

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


# ================================================================================================


class ShuffledBatchSampler(Sampler[list[int]]):
    """在完整候选下标轴上可复现地排列样本, 再切分固定物理批次. 

    采样器不读取 CandidateMetadata, 也不按 PDB 分组; 因此同一 PDB 候选既不会被主动聚集, 
    也不保证彼此分离. 预算允许且缓存写入成功时, 每个 DataLoader 工作进程可复用 I/O 载荷, 不改变
    全局下标顺序. sample_count_multiple 可把一个 epoch 截短到全局批量等约束的整数倍. 

    实例状态：
        - dataset: Sized; 只通过 len(dataset) 确定全局候选下标 0 到 N-1. 
        - batch_size: 正整数; 每个物理批次的最大样本数. 
        - shuffle: 布尔值; 为真时用 seed+epoch 洗牌, 为假时保持递增下标. 
        - drop_last: 布尔值; 为真时丢弃不足 batch_size 的末批, 否则保留. 
        - seed: 整数; 和 epoch 相加后初始化独立 Python random.Random. 
        - sample_count_multiple: 正整数或 None; 非 None 时必须是 batch_size 的整数倍, 并在切批前截短样本轴. 
        - epoch: 整数; 当前排列周期, 构造后为 0, 可由 set_epoch 更新. 
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
        """保存候选轴和物理批次规则, 并验证两个正整数约束. 

        输入参数：
            - dataset: Sized; 任何实现 len 的候选集合, 采样器不调用 getitem. 
            - batch_size: 整数; 必须大于零, 定义每次 yield 的最大下标数. 
            - shuffle: 布尔值; 控制是否对完整下标列表执行原地洗牌. 
            - drop_last: 布尔值; 控制是否产出不足 batch_size 的最终批次. 
            - seed: 整数; 可复现洗牌的基准种子. 
            - sample_count_multiple: 整数或 None; 非 None 时必须为 batch_size 的正整数倍. 

        返回状态：
            - self.dataset: 原 Sized 对象引用. 
            - self.batch_size: 输入物理批量. 
            - self.shuffle: 输入洗牌开关. 
            - self.drop_last: 输入末批开关. 
            - self.seed: 输入基准种子. 
            - self.sample_count_multiple: 输入截短倍数或 None. 
            - self.epoch: 整数 0; 首次迭代使用的周期偏移. 
        """

        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数. ")
        if sample_count_multiple is not None and (
            sample_count_multiple <= 0 or sample_count_multiple % batch_size != 0
        ):
            raise ValueError("sample_count_multiple 必须为 batch_size 的正整数倍. ")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.sample_count_multiple = sample_count_multiple
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """更新后续迭代使用的洗牌周期. 

        输入参数：
            - epoch: 可由 int 转换的值; 转换结果与 seed 相加后决定 shuffle=True 时的局部随机序列. 

        返回值：
            - return: None; 只更新 self.epoch, 不立即构造或缓存候选顺序. 
        """

        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        """生成当前 epoch 的全局候选下标批次. 

        中间字段：
            - generator: random.Random; 只由 self.seed+self.epoch 初始化, 不读取或推进 Python 全局随机状态. 
            - ordered_indices: 整数列表; 初始为 0 到 len(dataset)-1, 按 shuffle 选择保持递增或原地洗牌. 
            - usable_count: 整数; 启用 sample_count_multiple 时可保留的完整倍数数量, 而不是样本数量. 
            - batch: 整数列表; ordered_indices 的一个连续切片, 长度为 batch_size 或允许保留的末批长度. 

        产出值：
            - candidate_indices: 整数列表; 一个物理批次的 Dataset 全局下标, 顺序与本 epoch 的 ordered_indices 一致. 
        """

        # random.Random, 当前 epoch 私有的 Python 随机生成器, 不改变模块级随机状态. 
        generator = random.Random(self.seed + self.epoch)
        # list[int], 完整候选轴 0..N-1; shuffle=True 时在本列表内原地洗牌. 
        ordered_indices = list(range(len(self.dataset)))
        if self.shuffle:
            generator.shuffle(ordered_indices)
        if self.sample_count_multiple is not None:
            # int, 可保留的 sample_count_multiple 完整组数; 下一行把组数换算回样本数量. 
            usable_count = len(ordered_indices) // self.sample_count_multiple
            ordered_indices = ordered_indices[: usable_count * self.sample_count_multiple]
        for start in range(0, len(ordered_indices), self.batch_size):
            # list[int], 当前物理批次的全局候选下标; 末批可能短于 batch_size. 
            batch = ordered_indices[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        """按截短倍数和末批规则计算一次迭代产出的物理批次数. 

        中间字段：
            - sample_count: 整数; 初始为 len(dataset), 启用 sample_count_multiple 时向下截短到该值的最大整数倍. 

        返回值：
            - batch_count: 整数; drop_last 为真时向下整除 batch_size, 否则向上取整. 

        返回值不依赖 shuffle、seed 或 epoch, 因为这些字段只改变顺序, 不改变保留样本数. 
        """

        # int, 应用 sample_count_multiple 截短前的候选总数. 
        sample_count = len(self.dataset)
        if self.sample_count_multiple is not None:
            sample_count = sample_count // self.sample_count_multiple * self.sample_count_multiple
        if self.drop_last:
            return sample_count // self.batch_size
        return math.ceil(sample_count / self.batch_size)


def build_candidate_loader(
    dataset: CandidateDataset,
    sampler: ShuffledBatchSampler,
    num_workers: int,
    prefetch_factor: int,
    worker_generator_seed: int,
) -> DataLoader[CandidateBatch]:
    """构造采用 spawn、非持久工作进程和专用随机生成器的 DataLoader. 

    输入参数：
        - dataset: CandidateDataset; 按全局下标物化 CandidateItem, 工作进程会通过 spawn 获得序列化副本. 
        - sampler: ShuffledBatchSampler; 作为 batch_sampler 唯一决定样本顺序、物理批量和末批处理. 
        - num_workers: 整数; 传给 torch.utils.data.DataLoader 的工作进程数, 合法范围由 PyTorch 验证. 
        - prefetch_factor: 整数; 每个工作进程预取的批次数, 和 num_workers 的组合约束由 PyTorch 验证. 
        - worker_generator_seed: 整数; 初始化专用 torch.Generator, 用于 DataLoader 工作进程基准种子. 

    返回对象：
        - loader: DataLoader[CandidateBatch]; 每次迭代按 sampler 产生批次并调用 collate_candidates. 
        - loader.batch_sampler: 输入 sampler; DataLoader 不另行构造逐样本 sampler 或 batch_size. 
        - loader.collate_fn: collate_candidates; 生成完整 CandidateBatch. 
        - loader.pin_memory: True; 主进程尝试把返回张量放入页锁定内存. 
        - loader.persistent_workers: False; 一次完整迭代结束后工作进程退出, 下次迭代重新 spawn. 
        - loader.multiprocessing_context: spawn; 不继承父进程已经填充的 Dataset 大数组缓存. 
        - loader.generator: 专用 torch.Generator; 创建迭代器时不推进模型使用的全局 Torch 随机状态. 

    专用生成器只隔离 DataLoader 的工作进程播种; 候选洗牌仍完全由 ShuffledBatchSampler 的 Python random.Random 决定. 
    """

    # torch.Generator, DataLoader 私有的工作进程播种来源, 不与模型全局 Torch RNG 共用状态. 
    worker_generator = torch.Generator()
    worker_generator.manual_seed(worker_generator_seed)
    return cast(
        DataLoader[CandidateBatch],
        DataLoader(
            cast(Any, dataset),
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_candidates,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=DATALOADER_MULTIPROCESSING_CONTEXT,
            generator=worker_generator,
        ),
    )


def collate_candidates(items: list[CandidateItem]) -> CandidateBatch:
    """把非空 CandidateItem 列表整理成模型和评估共同使用的 CandidateBatch. 

    输入参数：
        - items: 非空 CandidateItem 列表; 列表顺序定义批次样本轴 B. 
        - items[*].metadata: CandidateMetadata; 保持为长度 B 的 Python 对象列表, 不转换为张量. 
        - items[*].voxel_input: 数组, (105, 48, 48, 48); 沿新样本轴直接堆叠. 
        - items[*].class_target: 整数; 转换为 int64 批次张量. 
        - items[*].ligand_area_target: bool 数组, (48, 48, 48); 沿样本轴堆叠. 
        - items[*].binding_area_target: bool 数组, (48, 48, 48); 沿样本轴堆叠. 
        - items[*].full_candidate_voxel_count: 整数; 转换为 int64 批次张量. 
        - items[*].candidate_manual_features: 数组, (12,); 堆叠并转换为 float32. 
        - items[*].candidate_sampling_features: 数组, (4,); 堆叠并转换为 float32. 
        - items[*].a_tokens: 数组, (N_A_b, 229); 沿 token 轴右侧补零. 
        - items[*].p_tokens: 数组, (N_P_b, 116); 沿 token 轴右侧补零. 
        - items[*].v_tokens: 数组, (N_V_b, 52); 沿 token 轴右侧补零并与 v_index_zyx 核对长度. 
        - items[*].v_index_zyx: 数组, (N_V_b, 3); 沿 token 轴右侧补零. 
        - items[*].pp_tokens: 数组, (K, 4); 正式 Dataset 的 K=min(pp_top_k, 48³), 沿 token 轴交给通用补齐函数处理. 
        - items[*].pp_index_zyx: 数组, (K, 3); 与 pp_tokens 逐项对齐并交给通用补齐函数处理. 
        - items[*].identity_object_keys: 长度 I_b 的字符串列表; 由 _collate_identities 保持原长度. 
        - items[*].identity_prepared_smiles: 长度 I_b 的字符串列表; 与 object_key 逐槽对齐. 
        - items[*].identity_class_index: 数组, (I_b,); 补到批内 I_max. 
        - items[*].language_embeddings: 数组, (I_b, 768); 补到批内 I_max. 
        - items[*].ligand_graphs: 长度 I_b 的 LigandGraph 列表; 连接为一组批次图张量. 
        - items[*].identity_target: 整数; 逐样本转换为 int64. 

    返回字段：
        - batch.metadata: 长度 B 的 CandidateMetadata 列表; 与 items 顺序相同. 
        - batch.metadata[*].split: 字符串; 对应候选的物化划分. 
        - batch.metadata[*].pdb_id: 字符串; 对应候选所属 PDB. 
        - batch.metadata[*].candidate_index: 整数; 对应 PDB 内候选编号. 
        - batch.metadata[*].matched_object_key: 字符串或 None; 通过覆盖阈值的 occurrence 模板键; type_tag=other 背景也可非空, 未命中时为 None. 
        - batch.metadata[*].matched_prepared_smiles: 字符串或 None; 仅五类前景命中的顶层 smiles 可非空; type_tag=other 背景和未命中为 None. 
        - batch.metadata[*].matched_prepared_smiles_status: PreparedSmilesStatus; 顶层字符串可用状态. 
        - batch.metadata[*].identity_seen_in_train: 布尔值或 None; 具备完整前景联合身份时是当前语言家族的 train 可见性; 背景、未命中、联合键不完整或未选语言家族时为 None. 
        - batch.voxel_input: 张量, (B, 105, 48, 48, 48); 保留 np.stack 产生的输入 dtype, 正式 Dataset 输出为 float16. 
        - batch.class_target: int64 张量, (B,); 六分类下标. 
        - batch.ligand_area_target: 张量, (B, 48, 48, 48); 保留输入 bool dtype. 
        - batch.binding_area_target: 张量, (B, 48, 48, 48); 保留输入 bool dtype. 
        - batch.full_candidate_voxel_count: int64 张量, (B,); 裁剪前完整 V 数量. 
        - batch.candidate_manual_features: float32 张量, (B, 12); A/P/V/PP 的 count/mean/max 四组三元组. 
        - batch.candidate_sampling_features: float32 张量, (B, 4); 边界标志、L2/L∞ 体素距离和 V 保留比例. 
        - batch.a_tokens: float32 张量, (B, N_A_max, 229); 右侧补零. 
        - batch.a_mask: bool 张量, (B, N_A_max); True 表示有效 A token. 
        - batch.p_tokens: float32 张量, (B, N_P_max, 116); 右侧补零. 
        - batch.p_mask: bool 张量, (B, N_P_max); True 表示有效 P token. 
        - batch.v_tokens: float32 张量, (B, N_V_max, 52); 右侧补零. 
        - batch.v_mask: bool 张量, (B, N_V_max); True 表示有效 V token 和索引. 
        - batch.v_index_zyx: int64 张量, (B, N_V_max, 3); 最后一轴为 Z/Y/X, 补零位置仅在 v_mask 为 False 时出现. 
        - batch.pp_tokens: float32 张量, (B, K, 4); 正式 Dataset 在同一 loader 内固定 K. 
        - batch.pp_mask: bool 张量, (B, K); 正式 CandidateItem 的全部 K 个位置均为 True. 
        - batch.pp_index_zyx: int64 张量, (B, K, 3); 最后一轴为 Z/Y/X, 只有 pp_mask 为 True 的位置可读. 
        - batch.language_embeddings: float32 张量, (B, I_max, 768); 身份语言向量, 补零槽为 0. 
        - batch.identity_mask: bool 张量, (B, I_max); True 表示有效身份槽. 
        - batch.identity_class_index: int64 张量, (B, I_max); 有效槽为前景类别, 补零槽为 -1. 
        - batch.identity_target: int64 张量, (B,); 有效身份下标或 -1. 
        - batch.identity_object_keys: 长度 B 的嵌套字符串列表; 每项保持原长 I_b. 
        - batch.identity_prepared_smiles: 长度 B 的嵌套字符串列表; 每项保持原长 I_b. 
        - batch.graph_to_sample: int64 张量, (G,); 每张连接前图所属的样本下标. 
        - batch.graph_to_identity: int64 张量, (G,); 每张连接前图所属样本内的身份下标. 
        - batch.graph_element: int64 张量, (N_node,); 连接后的模板原子序数. 
        - batch.graph_charge: int64 张量, (N_node,); 连接后的模板原子形式电荷. 
        - batch.graph_residue_id: int64 张量, (N_node,); 连接后的模板内部残基编号. 
        - batch.graph_node_continuous: float32 张量, (N_node, 15); 连接后的节点连续特征. 
        - batch.graph_node_graph_index: int64 张量, (N_node,); 每个节点所属图的批内全局下标. 
        - batch.graph_edge_index: int64 张量, (2, N_directed_edge); 已经加节点偏移的连接后边端点. 
        - batch.graph_edge_features: float32 张量, (N_directed_edge, 10); 与 graph_edge_index 逐边对齐. 

    本函数明确拒绝空 items, 并核验 V/PP token-index 对齐、V/PP 固定宽度、两种候选特征宽度以及身份数量与目标范围. A/P 固定宽度、voxel/area 固定形状和若干 dtype/range 依赖正式 CandidateDataset 契约; np.stack、补齐函数或身份连接只会暴露部分不相容输入, 批内统一但科学形状错误的值可能被接受. 
    """

    if not items:
        raise ValueError("候选批次不能为空. ")
    _validate_token_index_alignment(items)   # NOTE: 这里声明我的态度：纯检验函数要么放到隔离的 utils/valid_.py, 要么不写——————核心是将检验函数和主线代码分到不同的地方，以方便人类阅读。
    # dict[str, Any], 先装入固定形状候选字段; 随后补入四模态变长字段和身份图字段. 
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
        "candidate_manual_features": torch.from_numpy(
            np.stack([item["candidate_manual_features"] for item in items])
        ).float(),
        "candidate_sampling_features": torch.from_numpy(
            np.stack([item["candidate_sampling_features"] for item in items])
        ).float(),
    }
    if batch["candidate_manual_features"].shape != (
        len(items),
        CANDIDATE_MANUAL_FEATURE_DIM,
    ):
        raise ValueError(
            "candidate_manual_features 必须为"
            f" (B,{CANDIDATE_MANUAL_FEATURE_DIM}). "
        )
    if batch["candidate_sampling_features"].shape != (
        len(items),
        CANDIDATE_SAMPLING_FEATURE_DIM,
    ):
        raise ValueError(
            "candidate_sampling_features 必须为"
            f" (B,{CANDIDATE_SAMPLING_FEATURE_DIM}). "
        )
    # float32 (B, N_A_max, 229) 与 bool (B, N_A_max), A token 补零结果及有效掩码. 
    batch["a_tokens"], batch["a_mask"] = _pad_float_sequences(
        [item["a_tokens"] for item in items]
    )
    # float32 (B, N_P_max, 116) 与 bool (B, N_P_max), P token 补零结果及有效掩码. 
    batch["p_tokens"], batch["p_mask"] = _pad_float_sequences(
        [item["p_tokens"] for item in items]
    )
    # float32 (B, N_V_max, 52) 与 bool (B, N_V_max), V token 补零结果及有效掩码. 
    batch["v_tokens"], batch["v_mask"] = _pad_float_sequences(
        [item["v_tokens"] for item in items]
    )
    # float32 (B, K, 4) 与 bool (B, K), 正式 Dataset 的固定 PP token 及有效掩码. 
    batch["pp_tokens"], batch["pp_mask"] = _pad_float_sequences(
        [item["pp_tokens"] for item in items]
    )
    # int64, (B, N_V_max, 3), 与 v_tokens 同一补零长度的裁剪内 Z/Y/X 索引. 
    batch["v_index_zyx"] = _pad_index_sequences(
        [item["v_index_zyx"] for item in items]
    )
    # int64, (B, K, 3), 与 pp_tokens 逐项对齐的裁剪内 Z/Y/X 索引. 
    batch["pp_index_zyx"] = _pad_index_sequences(
        [item["pp_index_zyx"] for item in items]
    )
    # 加入补齐身份轴、未补齐字符串轴、图映射和七个连接后图字段. 
    batch.update(_collate_identities(items))
    return cast(CandidateBatch, batch)
