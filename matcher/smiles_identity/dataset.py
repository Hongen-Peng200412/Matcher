"""把 Find_1 centered 候选物化为单一 SMILES 身份 Matcher 样本.

每个持久 DataLoader 工作进程完整载入一次全局 SMILES 图包, 并用一个按字节预算的
PDB 最近使用缓存保存 centered、受体、occurrence 掩码和语言向量. 106 通道
体素输入只在内存构造, 不额外落盘.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from torch.utils.data import Dataset

from ..data.density import build_density_channels
from ..utils.cache import ByteBudgetLru
from .contracts import CandidateItem
from .graph import SmilesGraphStore
from .manifest import load_manifest, read_split_records
from .schema import (
    A_TOKEN_DIM,
    CANDIDATE_MANUAL_FEATURE_DIM,
    CANDIDATE_SAMPLING_FEATURE_DIM,
    FULL_CONTEXT_HALF_EXTENT_VOXELS,
    LANGUAGE_EMBEDDING_DIM,
    LOCAL_BOX_SIZE,
    LOCAL_CONTEXT_HALF_EXTENT_VOXELS,
    P_TOKEN_DIM,
    PP_TOKEN_DIM,
    V_TOKEN_DIM,
    VOXEL_INPUT_CHANNELS,
)


def _scatter_receptor(
    coordinates_xyz: np.ndarray,
    features: np.ndarray,
    box_origin_xyz: np.ndarray,
    voxel_size_xyz: np.ndarray,
    crop_start_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把完整受体的 50 维特征求和散射到给定 48 立方体.

    - coordinates_xyz: float32 (N, 3), 受体原子的世界 X/Y/Z 坐标.
    - features: float32 (N, 50), 与原子轴对齐的受体特征.
    - box_origin_xyz: float32 (3,), 80 立方体的世界坐标原点, 轴顺序为 XYZ, 单位 Å.
    - voxel_size_xyz: float32 (3,), 80 立方体每轴体素尺寸, 轴顺序为 XYZ, 单位 Å.
    - crop_start_zyx: int64 (3,), 48 立方体在 80 立方体内的 Z/Y/X 起点.
    - return[0]: float32 (50, 48, 48, 48); 落入同一体素的原子特征按通道求和.
    - return[1]: bool (48, 48, 48); 至少含一个受体原子的体素为 True.
    """

    # int64 数组, (N, 3); 世界 XYZ 坐标减候选原点并除以各轴 spacing 后向下取整, 得到 80 立方体局部 XYZ 下标.
    local_xyz = np.floor((coordinates_xyz - box_origin_xyz) / voxel_size_xyz).astype(np.int64)
    # int64 数组, (N, 3); 从 80 立方体局部 XYZ 下标减去 supplied crop 的 XYZ 起点.
    crop_xyz = local_xyz - crop_start_zyx[::-1]
    # bool 数组, (N,); True 表示对应受体原子落入 supplied 48 立方体, 并遮盖 coordinates_xyz 与 features 第一轴.
    keep = np.all((crop_xyz >= 0) & (crop_xyz < LOCAL_BOX_SIZE), axis=1)
    # int64 数组, (N_kept, 3); 保留受体原子在 supplied crop 内的 XYZ 下标.
    selected = crop_xyz[keep]
    grid = np.zeros((50, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE), dtype=np.float32)
    np.add.at(grid, (slice(None), selected[:, 2], selected[:, 1], selected[:, 0]), features[keep].T)
    hardmask = np.zeros((LOCAL_BOX_SIZE, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE), dtype=np.bool_)
    hardmask[selected[:, 2], selected[:, 1], selected[:, 0]] = True
    return grid, hardmask


def _build_dense_tokens(
    centered: dict[str, np.ndarray],
    position: int,
    crop_start_zyx: np.ndarray,
    modality: str,
) -> np.ndarray:
    """构造保留完整候选范围的 A 或 P token, 并以 supplied crop 中心定义相对坐标.

    - centered: 当前 PDB 的完整 centered 数组映射.
    - position: 当前候选在 centered 第一轴的位置.
    - crop_start_zyx: supplied crop 在 80 立方体内的 Z/Y/X 起点.
    - modality: A 或 P; 决定读取字段集合和输出宽度.
    - return: float32 (N_A, 230) 或 (N_P, 116); 最后四列为概率、相对 X/Y/Z.
    """

    # 两个整数, 当前候选在对应模态连接 token 轴上的左闭右开区间.
    start, stop = centered[f"{modality}_offsets"][position : position + 2]
    # float32 数组, (3,); supplied crop 中心在完整 80 立方体内的局部 X/Y/Z 坐标, 单位为体素.
    crop_center_xyz = crop_start_zyx[::-1].astype(np.float32) + LOCAL_BOX_SIZE / 2.0
    # float32 数组, (N_modality, 3); token 相对 supplied crop 中心的 X/Y/Z, 除以完整框半边长 40 体素.
    relative_xyz = (
        np.asarray(centered[f"{modality}_coord_local_xyz"][start:stop], dtype=np.float32)
        - crop_center_xyz
    ) / FULL_CONTEXT_HALF_EXTENT_VOXELS
    if modality == "A":
        tokens = np.concatenate(
            (
                np.asarray(centered["A_feat_L0"][start:stop], dtype=np.float32),
                np.asarray(centered["A_feat_L1"][start:stop], dtype=np.float32),
                np.asarray(centered["A_feat_L2"][start:stop], dtype=np.float32),
                np.asarray(centered["A_feat_L3"][start:stop], dtype=np.float32),
                np.asarray(centered["A_probability"][start:stop], dtype=np.float32)[:, None],
                relative_xyz,
            ),
            axis=1,
        )
        expected_width = A_TOKEN_DIM
    elif modality == "P":
        tokens = np.concatenate(
            (
                np.asarray(centered["P_feat_L2"][start:stop], dtype=np.float32),
                np.asarray(centered["P_feat_L3"][start:stop], dtype=np.float32),
                np.asarray(centered["P_probability"][start:stop], dtype=np.float32)[:, None],
                relative_xyz,
            ),
            axis=1,
        )
        expected_width = P_TOKEN_DIM
    else:
        raise ValueError(f"只支持 A 或 P，实际为 {modality!r}。")
    if tokens.shape != (stop - start, expected_width) or not np.isfinite(tokens).all():
        raise ValueError(f"{modality} token 形状或有限值错误: {tokens.shape}。")
    return tokens.astype(np.float32, copy=False)


def _build_v_tokens(
    centered: dict[str, np.ndarray],
    position: int,
    crop_start_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """筛选 supplied crop 内的 V, 并拼接 64 维特征、概率和相对坐标.

    - centered: 当前 PDB 的完整 centered 数组映射.
    - position: 当前候选在 centered 第一轴的位置.
    - crop_start_zyx: supplied crop 在 80 立方体内的 Z/Y/X 起点.
    - return[0]: float32 (N_V, 68); 64 维 V 特征、概率和相对 X/Y/Z.
    - return[1]: int64 (N_V, 3); 同一 token 在 48 立方体内的 Z/Y/X 下标.
    """

    start, stop = centered["voxel_offsets"][position : position + 2]
    # int64 数组, (N_source_V, 3); 当前候选全部 V token 在完整 80 立方体内的局部 Z/Y/X 下标.
    full_index = np.asarray(centered["voxel_index_local_zyx"][start:stop], dtype=np.int64)
    # int64 数组, (N_source_V, 3); 全部 V token 相对 supplied crop 起点的 Z/Y/X 下标.
    crop_index = full_index - crop_start_zyx
    # bool 数组, (N_source_V,); True 表示对应 V token 落入 supplied 48 立方体.
    keep = np.all((crop_index >= 0) & (crop_index < LOCAL_BOX_SIZE), axis=1)
    crop_index = crop_index[keep]
    # float32 数组, (N_V, 3); V 体素中心相对 crop 中心的 X/Y/Z, 除以局部半边长 24 体素.
    relative_xyz = (
        crop_index[:, ::-1].astype(np.float32) + 0.5 - LOCAL_BOX_SIZE / 2.0
    ) / LOCAL_CONTEXT_HALF_EXTENT_VOXELS
    tokens = np.concatenate(
        (
            np.asarray(centered["voxel_final"][start:stop][keep], dtype=np.float32),
            np.asarray(centered["centered_probability"][start:stop][keep], dtype=np.float32)[:, None],
            relative_xyz,
        ),
        axis=1,
    )
    if tokens.shape != (len(crop_index), V_TOKEN_DIM) or not np.isfinite(tokens).all():
        raise ValueError(f"V token 形状或有限值错误: {tokens.shape}。")
    return tokens.astype(np.float32, copy=False), crop_index


def _build_pp_tokens(probability: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """按概率降序、C 顺序下标升序稳定选择 supplied 48 立方体的 PP token.

    - probability: float32 (48, 48, 48), Find_1 supplied crop 的源概率.
    - top_k: 正整数; 最多保留的体素数.
    - return[0]: float32 (K, 4); 概率和相对 X/Y/Z, K=min(top_k, 48^3).
    - return[1]: int64 (K, 3); 与 token 对齐的局部 Z/Y/X 下标.
    """

    grid = np.asarray(probability, dtype=np.float32)
    if grid.shape != (LOCAL_BOX_SIZE, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE) or not np.isfinite(grid).all():
        raise ValueError("source_probability_48 必须是有限 float32 (48,48,48)。")
    # float32 数组, (48^3,); 概率按 NumPy C 顺序展平, X 轴变化最快.
    flat = grid.reshape(-1)
    count = min(top_k, flat.size)
    # int64 数组, (K,); 概率降序的 C 顺序平面下标, 并列时保留较小下标.
    selected_flat = np.argsort(-flat, kind="stable")[:count]
    # int64 数组, (K, 3); selected_flat 还原为 supplied crop 内 Z/Y/X 下标.
    index_zyx = np.stack(np.unravel_index(selected_flat, grid.shape), axis=1).astype(np.int64)
    # float32 数组, (K, 3); PP 体素中心相对 crop 中心的 X/Y/Z, 除以局部半边长 24 体素.
    relative_xyz = (
        index_zyx[:, ::-1].astype(np.float32) + 0.5 - LOCAL_BOX_SIZE / 2.0
    ) / LOCAL_CONTEXT_HALF_EXTENT_VOXELS
    tokens = np.concatenate((flat[selected_flat, None], relative_xyz), axis=1)
    if tokens.shape != (count, PP_TOKEN_DIM):
        raise RuntimeError("PP token 宽度不符合固定契约。")
    return tokens.astype(np.float32, copy=False), index_zyx


def _build_ligand_target(
    occurrence_masks: list[np.ndarray],
    box_start_zyx: np.ndarray,
    crop_start_zyx: np.ndarray,
) -> np.ndarray:
    """把当前 PDB 全部 occurrence 掩码在 supplied crop 内取并集.

    - occurrence_masks: 每项为 int64 (N_i, 3) 的全局 Z/Y/X 稀疏掩码.
    - box_start_zyx: 当前 80 立方体在全局密度图中的 Z/Y/X 起点.
    - crop_start_zyx: supplied 48 立方体在当前 80 立方体内的 Z/Y/X 起点.
    - return: bool (48, 48, 48); 所有 occurrence 落入当前 crop 的体素并集.
    """

    # int64 数组, (3,); supplied crop 在全局密度图中的 Z/Y/X 起点.
    crop_global_start = box_start_zyx + crop_start_zyx
    target = np.zeros((LOCAL_BOX_SIZE, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE), dtype=np.bool_)
    for mask in occurrence_masks:
        # int64 数组, (N_i, 3); 当前 occurrence 全局体素相对 crop 起点的 Z/Y/X 下标.
        local = np.asarray(mask, dtype=np.int64) - crop_global_start
        # bool 数组, (N_i,); True 表示对应 occurrence 体素落入 supplied crop.
        keep = np.all((local >= 0) & (local < LOCAL_BOX_SIZE), axis=1)
        selected = local[keep]
        target[selected[:, 0], selected[:, 1], selected[:, 2]] = True
    return target


def _build_binding_target(
    coordinates_xyz: np.ndarray,
    binding_atom: np.ndarray,
    box_origin_xyz: np.ndarray,
    voxel_size_xyz: np.ndarray,
    crop_start_zyx: np.ndarray,
) -> np.ndarray:
    """把完整受体中的 binding_atom 散射到 supplied crop.

    - coordinates_xyz: float32 (N, 3), 受体原子的世界 X/Y/Z 坐标.
    - binding_atom: bool (N,), 与受体原子轴对齐的结合标签.
    - box_origin_xyz: float32 (3,), 80 立方体的世界坐标原点, 轴顺序为 XYZ, 单位 Å.
    - voxel_size_xyz: float32 (3,), 80 立方体每轴体素尺寸, 轴顺序为 XYZ, 单位 Å.
    - crop_start_zyx: supplied 48 立方体在 80 立方体内的 Z/Y/X 起点.
    - return: bool (48, 48, 48); 含至少一个 binding_atom 的局部体素为 True.
    """

    # int64 数组, (N, 3); 受体世界 XYZ 坐标转换为 80 立方体局部 XYZ 下标.
    local_xyz = np.floor((coordinates_xyz - box_origin_xyz) / voxel_size_xyz).astype(np.int64)
    # int64 数组, (N, 3); 受体原子相对 supplied crop 起点的 XYZ 下标.
    crop_xyz = local_xyz - crop_start_zyx[::-1]
    # bool 数组, (N,); 同时要求 binding_atom=True 且对应受体原子落入 supplied crop.
    keep = binding_atom & np.all((crop_xyz >= 0) & (crop_xyz < LOCAL_BOX_SIZE), axis=1)
    selected = crop_xyz[keep]
    target = np.zeros((LOCAL_BOX_SIZE, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE), dtype=np.bool_)
    target[selected[:, 2], selected[:, 1], selected[:, 0]] = True
    return target


def _manual_features(
    a_tokens: np.ndarray,
    p_tokens: np.ndarray,
    v_tokens: np.ndarray,
    pp_tokens: np.ndarray,
) -> np.ndarray:
    """按 A、P、V、PP 顺序汇总数量、概率均值和概率最大值.

    - a_tokens: float32 (N_A, 230), 当前候选的 A token; 第 226 列为 Stage1 概率.
    - p_tokens: float32 (N_P, 116), 当前候选的 P token; 第 112 列为 Stage1 概率.
    - v_tokens: float32 (N_V, 68), 当前候选的 V token; 第 64 列为 Stage1 概率.
    - pp_tokens: float32 (N_PP, 4), 当前候选的 PP token; 第 0 列为 Stage1 概率.
    - return: float32 (12,); 每类依次保存 token 数、概率均值和概率最大值.

    空 token 集合的均值和最大值统一写零; 计数保留原始数值, 模型侧再执行 log1p.
    """

    values: list[float] = []
    for tokens, probability_column in ((a_tokens, 226), (p_tokens, 112), (v_tokens, 64), (pp_tokens, 0)):
        # float32 数组, (N_modality,); 当前模态每个 token 的 Stage1 概率.
        probabilities = tokens[:, probability_column]
        values.extend(
            (
                float(len(tokens)),
                float(probabilities.mean()) if len(tokens) else 0.0,
                float(probabilities.max()) if len(tokens) else 0.0,
            )
        )
    features = np.asarray(values, dtype=np.float32)
    if features.shape != (CANDIDATE_MANUAL_FEATURE_DIM,) or not np.isfinite(features).all():
        raise RuntimeError("候选手工统计不符合固定宽度或有限值契约。")
    return features


# ================================================================================================


class CandidateDataset(Dataset[CandidateItem]):
    """按清单顺序读取 Find_1 候选和同 PDB 精确 SMILES 身份轴.

    构造参数:
        - manifest_path: smiles_identity_v1 的 manifest.json.
        - split: 入口清单中存在的划分名称.
        - pp_top_k: 每个候选从 source_probability_48 选择的固定 token 上限.
        - cache_gib_per_worker: 当前 Dataset 副本的 PDB 数组缓存预算; 不是进程 RSS 上限.

    初始化只读取小型 JSON/JSONL. 图包和 PDB NPZ 在主进程直接索引或工作进程初始化后
    首次需要时读取. 缺少图或语言向量的正类候选已经在清单构建阶段整条排除.
    """

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        pp_top_k: int,
        cache_gib_per_worker: float,
    ) -> None:
        """冻结候选顺序、PDB 身份记录与两个内存存储.

        - manifest_path: manifest.json 路径; 文件同时绑定图包、数据根和 Find_1 根.
        - split: 要读取的候选与身份 JSONL 划分名.
        - pp_top_k: 每个候选保留的 PP token 上限, 必须为正.
        - cache_gib_per_worker: 当前 Dataset 副本的 PDB 数组缓存预算, 必须为非负有限数.

        - self.candidates: 清单顺序的候选记录列表, 决定 Dataset 下标语义.
        - self.identities_by_pdb: PDB 到身份记录列表; 每个列表按类别、精确 SMILES 排序.
        - self.graph_store: 延迟完整载入的全局 SMILES 图存储.
        - self.pdb_cache: 按估算数组字节数淘汰的 PDB 级最近使用缓存.
        """

        if pp_top_k <= 0:
            raise ValueError("pp_top_k 必须为正整数。")
        if not np.isfinite(cache_gib_per_worker) or cache_gib_per_worker < 0:
            raise ValueError("cache_gib_per_worker 必须是非负有限数。")
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.pp_top_k = int(pp_top_k)
        manifest = load_manifest(self.manifest_path)
        self.data_root = Path(manifest["data_root"])
        self.stage1_root = Path(manifest["stage1_root"])
        self.candidates, identities = read_split_records(self.manifest_path, split)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for identity in identities:
            grouped[str(identity["pdb_id"])].append(identity)
        self.identities_by_pdb = {
            pdb_id: sorted(records, key=lambda item: (int(item["class_index"]), str(item["smiles"])))
            for pdb_id, records in grouped.items()
        }
        self.graph_store = SmilesGraphStore(Path(manifest["graph_asset_path"]))
        maximum_bytes = int(cache_gib_per_worker * 1024**3)
        self.pdb_cache: ByteBudgetLru[str, dict[str, Any]] = ByteBudgetLru(maximum_bytes)

    def __len__(self) -> int:
        """返回清单中已通过正类完整性门禁的候选数.

        - return: int, self.candidates 的长度; Dataset 下标取值属于 [0, return).
        """

        return len(self.candidates)

    def initialize_worker(self) -> None:
        """在当前进程完整载入图包.

        - return: None, 供 DataLoader worker_init_fn 调用; 重复调用复用 self.graph_store 已有内存.
        """

        self.graph_store.initialize()

    def __getitem__(self, index: int) -> CandidateItem:
        """物化一个候选的 106 通道体素、四类 token、区域目标和身份轴.

        - index: self.candidates 的零基下标.
        - return: CandidateItem, 候选字段只描述当前 centered 框, 身份字段覆盖当前 PDB 的全部有效精确 SMILES 槽; 完整字段契约见 contracts.CandidateItem.

        PDB 级数组只在缓存未命中时打开. 当前候选的 supplied crop 决定受体散射、密度
        通道、V/PP token 和两个区域目标的共同 48 立方体坐标系.
        """

        # 候选清单对象; centered_box_index 是跨文件对齐键, 不假定等于当前列表下标.
        record = self.candidates[index]
        pdb_id = str(record["pdb_id"])
        payload = self.pdb_cache.get(pdb_id)
        if payload is None:
            payload = self._load_pdb(pdb_id)
            self.pdb_cache.put(pdb_id, payload)
        # 字段名到完整 PDB centered 数组的映射; 第一轴由 centered_box_index 标识.
        centered = cast(dict[str, np.ndarray], payload["centered"])
        try:
            position = cast(dict[int, int], payload["position_by_centered_index"])[
                int(record["centered_box_index"])
            ]
        except KeyError as error:
            raise ValueError(f"{pdb_id}: 清单 centered_box_index 不在 centered NPZ 中。") from error
        # int64 (3,); supplied 48 立方体在当前 80 立方体候选框内的 Z/Y/X 起点.
        crop_start_zyx = np.asarray(centered["crop_start_local_zyx"][position], dtype=np.int64)
        # float32 (3,); 80 立方体零号体素的世界 X/Y/Z 原点, 单位 Å.
        box_origin_xyz = np.asarray(centered["box_origin_world"][position], dtype=np.float32)
        # float32 (3,); 80 立方体 X/Y/Z 各轴的体素尺寸, 单位 Å/voxel.
        voxel_size_xyz = np.asarray(centered["voxel_size_world"][position], dtype=np.float32)
        if crop_start_zyx.shape != (3,) or np.any(crop_start_zyx < 0) or np.any(
            crop_start_zyx + LOCAL_BOX_SIZE > centered["box_shape_zyx"][position]
        ):
            raise ValueError(f"{pdb_id}: Find_1 crop_start_local_zyx 超出候选框。")
        # receptor_grid 为 float32 (50, 48, 48, 48), hardmask 为同空间 bool 占据掩码.
        receptor_grid, receptor_hardmask = _scatter_receptor(
            cast(np.ndarray, payload["receptor_coords"]),
            cast(np.ndarray, payload["receptor_features"]),
            box_origin_xyz,
            voxel_size_xyz,
            crop_start_zyx,
        )
        # float32 (56, 48, 48, 48); 实验密度、模拟密度和受体占据派生通道.
        density_channels = build_density_channels(
            centered["experimental_density_48"][position],
            centered["simulated_density_48"][position],
            receptor_hardmask,
        )
        # float16 数组, (106, 48, 48, 48); 第一轴连接 56 个密度通道和 50 个受体通道, 空间轴为 Z/Y/X.
        voxel_input = np.concatenate((density_channels, receptor_grid), axis=0).astype(np.float16)
        if voxel_input.shape != (VOXEL_INPUT_CHANNELS, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE):
            raise RuntimeError(f"{pdb_id}: voxel_input 形状错误 {voxel_input.shape}。")
        a_tokens = _build_dense_tokens(centered, position, crop_start_zyx, "A")
        p_tokens = _build_dense_tokens(centered, position, crop_start_zyx, "P")
        v_tokens, v_index_zyx = _build_v_tokens(centered, position, crop_start_zyx)
        pp_tokens, pp_index_zyx = _build_pp_tokens(
            centered["source_probability_48"][position],
            self.pp_top_k,
        )
        ligand_target = _build_ligand_target(
            cast(list[np.ndarray], payload["occurrence_masks"]),
            np.asarray(centered["box_start_zyx"][position], dtype=np.int64),
            crop_start_zyx,
        )
        binding_target = _build_binding_target(
            cast(np.ndarray, payload["receptor_coords"]),
            cast(np.ndarray, payload["binding_atom"]),
            box_origin_xyz,
            voxel_size_xyz,
            crop_start_zyx,
        )
        # 身份记录列表; 顺序同时定义字符串、类别、语言向量、图和监督下标的 I 轴.
        identities = self.identities_by_pdb.get(pdb_id, [])
        identity_smiles = [str(identity["smiles"]) for identity in identities]
        matched_smiles = record["matched_smiles"]
        # 整数; 背景为 -1, 前景为匹配精确 SMILES 在当前 PDB 身份轴中的唯一位置.
        identity_target = -1 if matched_smiles is None else identity_smiles.index(str(matched_smiles))
        manual_features = _manual_features(a_tokens, p_tokens, v_tokens, pp_tokens)
        full_voxel_count = int(centered["source_voxel_count"][position])
        # float32 数组, (4,); 依次为边界限制标志、中心到 V 质心的 L2 与 L∞ 体素距离、V 保留比例.
        sampling_features = np.asarray(
            (
                float(np.any(centered["crop_clipped_axis_mask"][position])),
                float(np.linalg.norm(centered["crop_center_offset_zyx"][position])),
                float(np.max(np.abs(centered["crop_center_offset_zyx"][position]))),
                len(v_tokens) / full_voxel_count,
            ),
            dtype=np.float32,
        )
        # CandidateMetadata, 当前候选的跨文件编号、Stage1 选择状态和真实匹配.
        metadata = {
            "split": self.split,
            "pdb_id": pdb_id,
            "centered_box_index": int(record["centered_box_index"]),
            "source_blob_index": int(record["source_blob_index"]),
            "stage1_selected": bool(record["stage1_selected"]),
            "matched_occurrence_id": record["matched_occurrence_id"],
            "matched_smiles": matched_smiles,
        }
        return CandidateItem(
            metadata=metadata,
            voxel_input=voxel_input,
            a_tokens=a_tokens,
            p_tokens=p_tokens,
            v_tokens=v_tokens,
            v_index_zyx=v_index_zyx,
            pp_tokens=pp_tokens,
            pp_index_zyx=pp_index_zyx,
            class_target=int(record["class_index"]),
            ligand_area_target=ligand_target,
            binding_area_target=binding_target,
            full_candidate_voxel_count=full_voxel_count,
            candidate_manual_features=manual_features,
            candidate_sampling_features=sampling_features,
            identity_smiles=identity_smiles,
            identity_class_index=np.asarray([identity["class_index"] for identity in identities], dtype=np.int64),
            language_embeddings=np.stack([payload["embedding_by_smiles"][smiles] for smiles in identity_smiles], axis=0)
            if identity_smiles
            else np.empty((0, LANGUAGE_EMBEDDING_DIM), dtype=np.float32),
            smiles_graphs=[self.graph_store.get(smiles) for smiles in identity_smiles],
            identity_target=identity_target,
        )

    def _load_pdb(self, pdb_id: str) -> dict[str, Any]:
        """一次打开一个 PDB 的 centered、受体、occurrence 掩码和身份向量.

        - pdb_id: 当前候选所属的小写 PDB 标识.
        - return.centered: dict[str, np.ndarray], F2_centered.npz 的全部数组副本; 下列字段由当前 Dataset 读取.
            - return.centered.centered_box_index: 整数数组, (N_candidate,); 每个 centered 候选的稳定编号.
            - return.centered.crop_start_local_zyx: 整数数组, (N_candidate, 3); supplied 48 立方体在各 80 立方体候选内的 Z/Y/X 起点.
            - return.centered.box_origin_world: float32 数组, (N_candidate, 3); 各 80 立方体零号体素的世界 X/Y/Z 原点.
            - return.centered.voxel_size_world: float32 数组, (N_candidate, 3); 各候选 X/Y/Z 三轴的世界体素尺寸.
            - return.centered.box_shape_zyx: 整数数组, (N_candidate, 3); 各候选完整框的 Z/Y/X 单轴体素数.
            - return.centered.experimental_density_48: 数值数组, (N_candidate, 48, 48, 48); supplied crop 内实验密度, 空间轴为 Z/Y/X.
            - return.centered.simulated_density_48: 数值数组, (N_candidate, 48, 48, 48); supplied crop 内模拟密度, 空间轴为 Z/Y/X.
            - return.centered.A_offsets: 整数数组, (N_candidate + 1,); 同步切分 A 坐标、四层特征和概率的候选区间.
            - return.centered.A_coord_local_xyz: 数值数组, (N_A_total, 3); 每个 A token 在所属 80 立方体中的局部 X/Y/Z 坐标.
            - return.centered.A_feat_L0: 数值数组, (N_A_total, 50); A token 的 L0 特征.
            - return.centered.A_feat_L1: 数值数组, (N_A_total, 64); A token 的 L1 特征.
            - return.centered.A_feat_L2: 数值数组, (N_A_total, 64); A token 的 L2 特征.
            - return.centered.A_feat_L3: 数值数组, (N_A_total, 48); A token 的 L3 特征.
            - return.centered.A_probability: 数值数组, (N_A_total,); 与 A token 逐项对齐的 Stage1 概率.
            - return.centered.P_offsets: 整数数组, (N_candidate + 1,); 同步切分 P 坐标、两层特征和概率的候选区间.
            - return.centered.P_coord_local_xyz: 数值数组, (N_P_total, 3); 每个 P token 在所属 80 立方体中的局部 X/Y/Z 坐标.
            - return.centered.P_feat_L2: 数值数组, (N_P_total, 64); P token 的 L2 特征.
            - return.centered.P_feat_L3: 数值数组, (N_P_total, 48); P token 的 L3 特征.
            - return.centered.P_probability: 数值数组, (N_P_total,); 与 P token 逐项对齐的 Stage1 概率.
            - return.centered.voxel_offsets: 整数数组, (N_candidate + 1,); 同步切分 V 下标、特征和概率的候选区间.
            - return.centered.voxel_index_local_zyx: 整数数组, (N_V_total, 3); 每个 V token 在所属 80 立方体中的局部 Z/Y/X 下标.
            - return.centered.voxel_final: 数值数组, (N_V_total, 64); 与 V token 逐项对齐的 64 维特征.
            - return.centered.centered_probability: 数值数组, (N_V_total,); 与 V token 逐项对齐的 Stage1 概率.
            - return.centered.source_probability_48: 数值数组, (N_candidate, 48, 48, 48); supplied crop 内用于 PP top-k 的 Stage1 概率, 空间轴为 Z/Y/X.
            - return.centered.box_start_zyx: 整数数组, (N_candidate, 3); 各 80 立方体候选在全局密度图中的 Z/Y/X 起点.
            - return.centered.source_voxel_count: 整数数组, (N_candidate,); 各候选裁剪前 source blob 的体素数.
            - return.centered.crop_clipped_axis_mask: bool 数组, (N_candidate, 3); True 表示 supplied crop 对应 Z/Y/X 起点曾受候选边界限制.
            - return.centered.crop_center_offset_zyx: 数值数组, (N_candidate, 3); supplied crop 中心到 V 质心的 Z/Y/X 偏移, 单位为体素.
        - return.position_by_centered_index: centered_box_index 到第一轴位置的唯一映射.
        - return.receptor_coords: float32 数组, (N_receptor, 3); 完整受体原子的世界 X/Y/Z 坐标.
        - return.receptor_features: float32 数组, (N_receptor, 50); 49 维 receptor feat 后连接 is_backbone 的 0/1 列.
        - return.binding_atom: bool (N_receptor,), 受体原子结合标签.
        - return.occurrence_masks: list[np.ndarray], 按 mask_数字中的 occurrence 编号递增; 每项为 int64 (N_i, 3) 全局 Z/Y/X 稀疏体素下标, 仅用于取全部 occurrence 并集.
        - return.embedding_by_smiles: dict[str, np.ndarray], 精确 SMILES 到有限 float32 (768,) SMI-TED 向量的映射.

        返回值交给 PDB 缓存整体保存; 所有 NPZ 上下文在返回前关闭.
        """

        centered_path = self.stage1_root / self.split / pdb_id / "centered/F2_centered.npz"
        with np.load(centered_path, allow_pickle=False) as archive:
            centered = {name: np.asarray(archive[name]).copy() for name in archive.files}
        centered_indices = np.asarray(centered["centered_box_index"], dtype=np.int64)
        position_by_centered_index = {int(value): index for index, value in enumerate(centered_indices)}
        if len(position_by_centered_index) != len(centered_indices):
            raise ValueError(f"{pdb_id}: centered_box_index 存在重复值。")
        with np.load(self.data_root / "parse" / pdb_id / "receptor_tokens.npz", allow_pickle=False) as archive:
            receptor_coords = np.asarray(archive["coords"], dtype=np.float32).copy()
            receptor_base_features = np.asarray(archive["feat"], dtype=np.float32).copy()
            receptor_is_backbone = np.asarray(archive["is_backbone"], dtype=np.bool_).copy()
        if receptor_base_features.shape != (len(receptor_coords), 49):
            raise ValueError(
                f"{pdb_id}: receptor feat 必须为 (N,49)，实际为 "
                f"{receptor_base_features.shape}。"
            )
        if receptor_is_backbone.shape != (len(receptor_coords),):
            raise ValueError(f"{pdb_id}: receptor is_backbone 未与完整受体原子轴对齐。")
        receptor_features = np.concatenate(
            (receptor_base_features, receptor_is_backbone.astype(np.float32)[:, None]),
            axis=1,
        )
        if receptor_features.shape != (len(receptor_coords), 50):
            raise RuntimeError(f"{pdb_id}: 受体特征拼接结果不是 (N,50)。")
        with np.load(self.data_root / "labels" / pdb_id / "atom_labels.npz", allow_pickle=False) as archive:
            binding_atom = np.asarray(archive["binding_atom"], dtype=np.bool_).copy()
        if binding_atom.shape != (len(receptor_coords),):
            raise ValueError(f"{pdb_id}: binding_atom 未与完整受体原子轴对齐。")
        with np.load(self.data_root / "density" / pdb_id / "ligand_area.npz", allow_pickle=False) as archive:
            mask_names = sorted(
                (name for name in archive.files if name.startswith("mask_") and name[5:].isdigit()),
                key=lambda name: int(name[5:]),
            )
            occurrence_masks = [
                np.asarray(archive[name], dtype=np.int64).copy() for name in mask_names
            ]
        if any(mask.ndim != 2 or mask.shape[1] != 3 for mask in occurrence_masks):
            raise ValueError(f"{pdb_id}: occurrence mask 必须全部为 (N,3) Z/Y/X 索引。")
        embedding_by_smiles: dict[str, np.ndarray] = {}
        for identity in self.identities_by_pdb.get(pdb_id, []):
            path = self.data_root / str(identity["language_relative_path"])
            with np.load(path, allow_pickle=False) as archive:
                embedding = np.asarray(archive["embedding"], dtype=np.float32).copy()
            if embedding.shape != (LANGUAGE_EMBEDDING_DIM,) or not np.isfinite(embedding).all():
                raise ValueError(f"{path}: embedding 不符合 (768,) 有限值契约。")
            embedding_by_smiles[str(identity["smiles"])] = embedding
        return {
            "centered": centered,
            "position_by_centered_index": position_by_centered_index,
            "receptor_coords": receptor_coords,
            "receptor_features": receptor_features,
            "binding_atom": binding_atom,
            "occurrence_masks": occurrence_masks,
            "embedding_by_smiles": embedding_by_smiles,
        }
