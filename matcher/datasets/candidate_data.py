"""把 Stage1 centered 候选物化为四个正式 Matcher 变体的单一 SMILES 样本。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from torch.utils.data import Dataset

from ..data.density import build_density_channels, build_experimental_density_channel
from ..data.manifest import load_manifest, read_split_records
from ..data.schema import (
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
from ..data.smiles_graph import SmilesGraphStore
from ..model.config import MODEL_VARIANTS, ModelVariant
from ..utils.cache import ByteBudgetLru
from .contracts import CandidateItem


DatasetMode = Literal["train", "validation", "test"]


def _load_npz_fields(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """只解压当前模型变体实际消费的 NPZ 字段。

    - path: 待读取的 NPZ 文件。
    - names: 必须存在且需要复制到内存的字段名。
    - return: 字段名到独立 NumPy 数组副本的映射。
    """

    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(names) - set(archive.files))
        if missing:
            raise ValueError(f"{path}: 缺少字段 {missing}。")
        return {name: np.asarray(archive[name]).copy() for name in names}


def _scatter_receptor(
    coordinates_xyz: np.ndarray,
    features: np.ndarray,
    box_origin_xyz: np.ndarray,
    voxel_size_xyz: np.ndarray,
    crop_start_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把受体原子特征散射到 supplied 48 立方体。

    - coordinates_xyz: float 数组，(N_atom,3)；受体原子世界坐标。
    - features: float32 数组，(N_atom,50)；49 维 L0 与 backbone 标志。
    - box_origin_xyz: float 数组，(3,)；完整 centered 框世界坐标原点。
    - voxel_size_xyz: float 数组，(3,)；X/Y/Z 体素边长。
    - crop_start_zyx: int64 数组，(3,)；supplied 48 crop 在完整框内的起点。
    - return[0]: float32 数组，(50,48,48,48)；同体素原子特征求和。
    - return[1]: bool 数组，(48,48,48)；至少含一个受体原子的体素为 True。
    """

    local_xyz = np.floor((coordinates_xyz - box_origin_xyz) / voxel_size_xyz).astype(np.int64)
    crop_xyz = local_xyz - crop_start_zyx[::-1]
    keep = np.all((crop_xyz >= 0) & (crop_xyz < LOCAL_BOX_SIZE), axis=1)
    selected = crop_xyz[keep]
    grid = np.zeros((50, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE), dtype=np.float32)
    np.add.at(grid, (slice(None), selected[:, 2], selected[:, 1], selected[:, 0]), features[keep].T)
    hardmask = np.zeros((LOCAL_BOX_SIZE, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE), dtype=np.bool_)
    hardmask[selected[:, 2], selected[:, 1], selected[:, 0]] = True
    return grid, hardmask


def _build_full_a_tokens(
    centered: dict[str, np.ndarray], position: int, crop_start_zyx: np.ndarray
) -> np.ndarray:
    """构造 strongest/trunk_only 使用的完整 230 维 A token。

    - centered: 当前 PDB 的 centered 字段；读取 A offsets、L0–L3、概率和局部坐标。
    - position: 当前候选在 centered 第一轴的位置。
    - crop_start_zyx: int64 (3,)；supplied crop 起点。
    - return: float32 (N_A,230)；四级特征、概率和归一化相对 XYZ。
    """

    start, stop = centered["A_offsets"][position : position + 2]
    center_xyz = crop_start_zyx[::-1].astype(np.float32) + LOCAL_BOX_SIZE / 2.0
    relative_xyz = (
        np.asarray(centered["A_coord_local_xyz"][start:stop], dtype=np.float32) - center_xyz
    ) / FULL_CONTEXT_HALF_EXTENT_VOXELS
    tokens = np.concatenate(
        (
            centered["A_feat_L0"][start:stop],
            centered["A_feat_L1"][start:stop],
            centered["A_feat_L2"][start:stop],
            centered["A_feat_L3"][start:stop],
            centered["A_probability"][start:stop, None],
            relative_xyz,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    if tokens.shape != (stop - start, A_TOKEN_DIM) or not np.isfinite(tokens).all():
        raise ValueError(f"A token 形状或有限值错误: {tokens.shape}。")
    return tokens


def _build_pocket_tokens(centered: dict[str, np.ndarray], position: int) -> np.ndarray:
    """读取 pocket_only 唯一允许使用的原始 50 维 A_feat_L0。

    - centered: 当前 PDB 的 centered 字段；只读取 A_offsets 与 A_feat_L0。
    - position: 当前候选在 centered 第一轴的位置。
    - return: float32 (N_A,50)；不含概率、坐标或 L1–L3。
    """

    start, stop = centered["A_offsets"][position : position + 2]
    tokens = np.asarray(centered["A_feat_L0"][start:stop], dtype=np.float32)
    if tokens.shape != (stop - start, 50) or not np.isfinite(tokens).all():
        raise ValueError(f"A_feat_L0 token 形状或有限值错误: {tokens.shape}。")
    return tokens


def _build_p_tokens(
    centered: dict[str, np.ndarray], position: int, crop_start_zyx: np.ndarray
) -> np.ndarray:
    """构造 strongest/trunk_only 使用的 116 维 P token。

    - centered: 当前 PDB 的 centered 字段；读取 P offsets、L2/L3、概率和坐标。
    - position: 当前候选在 centered 第一轴的位置。
    - crop_start_zyx: int64 (3,)；supplied crop 起点。
    - return: float32 (N_P,116)；两级特征、概率和归一化相对 XYZ。
    """

    start, stop = centered["P_offsets"][position : position + 2]
    center_xyz = crop_start_zyx[::-1].astype(np.float32) + LOCAL_BOX_SIZE / 2.0
    relative_xyz = (
        np.asarray(centered["P_coord_local_xyz"][start:stop], dtype=np.float32) - center_xyz
    ) / FULL_CONTEXT_HALF_EXTENT_VOXELS
    tokens = np.concatenate(
        (
            centered["P_feat_L2"][start:stop],
            centered["P_feat_L3"][start:stop],
            centered["P_probability"][start:stop, None],
            relative_xyz,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    if tokens.shape != (stop - start, P_TOKEN_DIM) or not np.isfinite(tokens).all():
        raise ValueError(f"P token 形状或有限值错误: {tokens.shape}。")
    return tokens


def _build_v_tokens(
    centered: dict[str, np.ndarray], position: int, crop_start_zyx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """筛选 supplied crop 内的 V token，并拼接特征、概率和相对坐标。

    - centered: 当前 PDB 的 centered 字段；读取 V offsets、64 维特征、概率和局部索引。
    - position: 当前候选在 centered 第一轴的位置。
    - crop_start_zyx: int64 (3,)；supplied crop 起点。
    - return[0]: float32 (N_V,68)；64 维特征、概率和相对 XYZ。
    - return[1]: int64 (N_V,3)；supplied crop 内 Z/Y/X 索引。
    """

    start, stop = centered["voxel_offsets"][position : position + 2]
    full_index = np.asarray(centered["voxel_index_local_zyx"][start:stop], dtype=np.int64)
    crop_index = full_index - crop_start_zyx
    keep = np.all((crop_index >= 0) & (crop_index < LOCAL_BOX_SIZE), axis=1)
    crop_index = crop_index[keep]
    relative_xyz = (
        crop_index[:, ::-1].astype(np.float32) + 0.5 - LOCAL_BOX_SIZE / 2.0
    ) / LOCAL_CONTEXT_HALF_EXTENT_VOXELS
    tokens = np.concatenate(
        (
            centered["voxel_final"][start:stop][keep],
            centered["centered_probability"][start:stop][keep, None],
            relative_xyz,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    if tokens.shape != (len(crop_index), V_TOKEN_DIM) or not np.isfinite(tokens).all():
        raise ValueError(f"V token 形状或有限值错误: {tokens.shape}。")
    return tokens, crop_index


def _build_pp_tokens(probability: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """按概率降序和 C 顺序下标升序稳定选择 PP token。

    - probability: float 数组，(48,48,48)；supplied Stage1 probability map。
    - top_k: 正整数；最多保留的体素数。
    - return[0]: float32 (N_PP,4)；概率和归一化相对 XYZ。
    - return[1]: int64 (N_PP,3)；对应 Z/Y/X 索引。
    """

    grid = np.asarray(probability, dtype=np.float32)
    if grid.shape != (LOCAL_BOX_SIZE, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE):
        raise ValueError(f"PP 概率网格必须是 (48,48,48)，实际为 {grid.shape}。")
    if not np.isfinite(grid).all():
        raise ValueError("PP 概率网格含非有限值。")
    flat = grid.reshape(-1)
    selected_flat = np.argsort(-flat, kind="stable")[: min(top_k, flat.size)]
    index_zyx = np.stack(np.unravel_index(selected_flat, grid.shape), axis=1).astype(np.int64)
    relative_xyz = (
        index_zyx[:, ::-1].astype(np.float32) + 0.5 - LOCAL_BOX_SIZE / 2.0
    ) / LOCAL_CONTEXT_HALF_EXTENT_VOXELS
    tokens = np.concatenate((flat[selected_flat, None], relative_xyz), axis=1)
    tokens = tokens.astype(np.float32, copy=False)
    if tokens.shape != (len(index_zyx), PP_TOKEN_DIM):
        raise RuntimeError(f"PP token 形状错误: {tokens.shape}。")
    return tokens, index_zyx


def _build_ligand_target(
    occurrence_masks: list[np.ndarray], box_start_zyx: np.ndarray, crop_start_zyx: np.ndarray
) -> np.ndarray:
    """把全部真实 occurrence 的稀疏全局体素并集投影到 supplied 48 立方体。

    - occurrence_masks: 每项为 int64 (N_i,3) 全局 Z/Y/X 稀疏体素。
    - box_start_zyx: int 数组，(3,)；完整 centered 框的全局起点。
    - crop_start_zyx: int 数组，(3,)；48 crop 在完整框内的起点。
    - return: bool (48,48,48)；任一 occurrence 占据的区域辅助目标。
    """

    target = np.zeros((LOCAL_BOX_SIZE,) * 3, dtype=np.bool_)
    crop_origin = box_start_zyx + crop_start_zyx
    for mask in occurrence_masks:
        local = mask - crop_origin
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
    """把受体 binding_atom 真值投影到 supplied 48 立方体。

    - coordinates_xyz: float (N_atom,3)；受体原子世界坐标。
    - binding_atom: bool (N_atom,)；受体结合区域真值。
    - box_origin_xyz / voxel_size_xyz: float (3,)；完整框原点和体素边长。
    - crop_start_zyx: int64 (3,)；48 crop 在完整框内的起点。
    - return: bool (48,48,48)；结合原子占据的区域辅助目标。
    """

    local_xyz = np.floor((coordinates_xyz - box_origin_xyz) / voxel_size_xyz).astype(np.int64)
    crop_xyz = local_xyz - crop_start_zyx[::-1]
    keep = binding_atom & np.all((crop_xyz >= 0) & (crop_xyz < LOCAL_BOX_SIZE), axis=1)
    selected = crop_xyz[keep]
    target = np.zeros((LOCAL_BOX_SIZE,) * 3, dtype=np.bool_)
    target[selected[:, 2], selected[:, 1], selected[:, 0]] = True
    return target


def _manual_features(
    a_tokens: np.ndarray,
    p_tokens: np.ndarray,
    v_tokens: np.ndarray,
    pp_tokens: np.ndarray,
) -> np.ndarray:
    """按 A、P、V、PP 顺序汇总 token 数、概率均值和概率最大值。

    - a_tokens / p_tokens / v_tokens / pp_tokens: 当前候选四类二维 token 数组。
    - return: float32 (12,)；每类依次为 count、probability_mean、probability_max。
    """

    values: list[float] = []
    for tokens, probability_column in (
        (a_tokens, 226),
        (p_tokens, 112),
        (v_tokens, 64),
        (pp_tokens, 0),
    ):
        probabilities = tokens[:, probability_column]
        values.extend(
            (
                float(len(tokens)),
                float(probabilities.mean()) if len(tokens) else 0.0,
                float(probabilities.max()) if len(tokens) else 0.0,
            )
        )
    return np.asarray(values, dtype=np.float32)


def _centered_fields(variant: ModelVariant) -> tuple[str, ...]:
    """返回一个模型变体允许从 F2_centered.npz 解压的字段集合。

    - variant: 四个正式模型变体之一。
    - return: 去重前已按读取用途排列的字段名元组；禁用模态字段不在其中。
    """

    common = ("centered_box_index",)
    if variant == "pocket_only":
        return common + ("A_offsets", "A_feat_L0")
    spatial = (
        "crop_start_local_zyx",
        "box_origin_world",
        "voxel_size_world",
        "experimental_density_48",
        "voxel_offsets",
        "voxel_index_local_zyx",
        "voxel_final",
        "centered_probability",
        "box_start_zyx",
        "source_voxel_count",
        "crop_clipped_axis_mask",
        "crop_center_offset_zyx",
    )
    if variant == "density_blob_only":
        return common + spatial
    return common + spatial + (
        "simulated_density_48",
        "source_probability_48",
        "A_offsets",
        "A_coord_local_xyz",
        "A_feat_L0",
        "A_feat_L1",
        "A_feat_L2",
        "A_feat_L3",
        "A_probability",
        "P_offsets",
        "P_coord_local_xyz",
        "P_feat_L2",
        "P_feat_L3",
        "P_probability",
    )


class CandidateDataset(Dataset[CandidateItem]):
    """读取一份 manifest，并按同一变体构造训练、验证或测试候选。

    manifest 冻结候选顺序和 PDB 内精确 SMILES 身份轴；Dataset 只应用 selected 过滤、
    train/validation 的身份完整性过滤和当前变体的按需字段读取。
    """

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        selected_only: bool,
        mode: DatasetMode,
        variant: ModelVariant,
        pp_top_k: int,
        cache_gib_per_worker: float,
    ) -> None:
        """冻结候选顺序、受支持身份轴和工作进程内缓存策略。

        - manifest_path: `matcher_stage1_manifest_v1` 的 manifest.json。
        - split: manifest.splits 中的划分键。
        - selected_only: True 时只保留 Stage1 已选候选。
        - mode: train/validation 跳过不受支持真前景；test 保留原候选。
        - variant: 四个正式模型变体之一，决定实际解压字段和样本张量。
        - pp_top_k: strongest/trunk_only 的 PP token 上限。
        - cache_gib_per_worker: 当前 Dataset 副本的 PDB 载荷缓存预算。
        """

        if mode not in ("train", "validation", "test"):
            raise ValueError(f"不支持 Dataset mode={mode!r}。")
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"不支持模型变体 {variant!r}。")
        if pp_top_k <= 0:
            raise ValueError("pp_top_k 必须为正整数。")
        if not np.isfinite(cache_gib_per_worker) or cache_gib_per_worker < 0:
            raise ValueError("cache_gib_per_worker 必须是非负有限数。")
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.mode = mode
        self.variant = variant
        self.pp_top_k = int(pp_top_k)
        manifest = load_manifest(self.manifest_path)
        self.data_root = Path(manifest["data_root"])
        self.stage1_root = Path(manifest["stage1_root"])
        candidates, identities = read_split_records(self.manifest_path, split)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for identity in identities:
            grouped[str(identity["pdb_id"])].append(identity)
        self.identities_by_pdb = {
            pdb_id: sorted(records, key=lambda item: str(item["smiles"]))
            for pdb_id, records in grouped.items()
        }
        supported = {
            (pdb_id, str(identity["smiles"]))
            for pdb_id, records in self.identities_by_pdb.items()
            for identity in records
        }
        self.candidates = []
        for record in candidates:
            if selected_only and not bool(record["candidate_selected"]):
                continue
            target = record["target_smiles"]
            target_supported = target is not None and (str(record["pdb_id"]), str(target)) in supported
            if mode != "test" and bool(record["foreground_target"]) and not target_supported:
                continue
            self.candidates.append(record)
        self.graph_store = SmilesGraphStore(Path(manifest["graph_asset_path"]))
        self.pdb_cache: ByteBudgetLru[str, dict[str, Any]] = ByteBudgetLru(
            int(cache_gib_per_worker * 1024**3)
        )

    def __len__(self) -> int:
        """返回当前 Dataset 的候选数。

        - return: 应用 selected_only 及 train/validation 身份完整性规则后的整数候选数。
        """

        return len(self.candidates)

    def initialize_worker(self) -> None:
        """初始化当前 DataLoader 工作进程的全局图包。

        - return: None；strongest、density_blob_only 和 pocket_only 完整打开一次 NPZ，
          trunk_only 不读取图包。

        SmilesGraphStore 会忽略同一进程中的重复调用，因此训练样本不会反复打开或解压图文件。
        """

        if self.variant != "trunk_only":
            self.graph_store.initialize()

    def __getitem__(self, index: int) -> CandidateItem:
        """物化一个候选，并保持 manifest 定义的 PDB 内 SMILES 身份顺序。

        - index: 过滤后候选列表的零基位置。
        - return: CandidateItem；候选张量与 variant 对齐，身份轴按精确 SMILES 排序。
        """

        record = self.candidates[index]
        pdb_id = str(record["pdb_id"])
        payload = self.pdb_cache.get(pdb_id)
        if payload is None:
            payload = self._load_pdb(pdb_id)
            self.pdb_cache.put(pdb_id, payload)
        centered = cast(dict[str, np.ndarray], payload["centered"])
        position = cast(dict[int, int], payload["position_by_centered_index"])[
            int(record["centered_box_index"])
        ]
        arrays = self._build_candidate_arrays(centered, payload, position)
        identities = self.identities_by_pdb.get(pdb_id, [])
        identity_smiles = [str(identity["smiles"]) for identity in identities]
        target_smiles = record["target_smiles"]
        identity_supported = target_smiles is not None and str(target_smiles) in identity_smiles
        identity_target = identity_smiles.index(str(target_smiles)) if identity_supported else -1
        metadata = {
            "split": self.split,
            "pdb_id": pdb_id,
            "centered_box_index": int(record["centered_box_index"]),
            "source_blob_index": int(record["source_blob_index"]),
            "candidate_selected": bool(record["candidate_selected"]),
            "gaussian_score": float(record["gaussian_score"]),
            "source_probability_mean": float(record["source_probability_mean"]),
            "foreground_target": bool(record["foreground_target"]),
            "target_category": record["target_category"],
            "matched_occurrence_id": record["matched_occurrence_id"],
            "target_smiles": target_smiles,
            "identity_supported": identity_supported,
        }
        embeddings = cast(dict[str, np.ndarray], payload["embedding_by_smiles"])
        return CandidateItem(
            metadata=metadata,
            voxel_input=arrays["voxel_input"],
            a_tokens=arrays["a_tokens"],
            p_tokens=arrays["p_tokens"],
            v_tokens=arrays["v_tokens"],
            v_index_zyx=arrays["v_index_zyx"],
            pp_tokens=arrays["pp_tokens"],
            pp_index_zyx=arrays["pp_index_zyx"],
            foreground_target=bool(record["foreground_target"]),
            ligand_area_target=arrays["ligand_area_target"],
            binding_area_target=arrays["binding_area_target"],
            full_candidate_voxel_count=int(arrays["full_candidate_voxel_count"]),
            candidate_manual_features=arrays["candidate_manual_features"],
            candidate_sampling_features=arrays["candidate_sampling_features"],
            identity_smiles=identity_smiles,
            language_embeddings=np.stack([embeddings[smiles] for smiles in identity_smiles])
            if identity_smiles
            else np.empty((0, LANGUAGE_EMBEDDING_DIM), dtype=np.float32),
            smiles_graphs=[]
            if self.variant == "trunk_only"
            else [self.graph_store.get(smiles) for smiles in identity_smiles],
            identity_target=identity_target,
        )

    def _build_candidate_arrays(
        self, centered: dict[str, np.ndarray], payload: dict[str, Any], position: int
    ) -> dict[str, Any]:
        """按变体构造候选输入，并让禁用模态保持空数组或零统计。

        - centered: 当前 PDB 按 variant 解压的 centered 字段。
        - payload: 当前 PDB 的 centered、受体、区域真值和语言向量载荷。
        - position: 当前候选在 centered 第一轴的位置。
        - return: 候选输入、区域目标、统计和完整 V 体素数的字段映射。
        """

        zeros_target = np.zeros((LOCAL_BOX_SIZE,) * 3, dtype=np.bool_)
        empty_p = np.empty((0, P_TOKEN_DIM), dtype=np.float32)
        empty_v = np.empty((0, V_TOKEN_DIM), dtype=np.float32)
        empty_v_index = np.empty((0, 3), dtype=np.int64)
        empty_pp = np.empty((0, PP_TOKEN_DIM), dtype=np.float32)
        if self.variant == "pocket_only":
            return {
                "voxel_input": np.empty((0, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE), dtype=np.float16),
                "a_tokens": _build_pocket_tokens(centered, position),
                "p_tokens": empty_p,
                "v_tokens": empty_v,
                "v_index_zyx": empty_v_index,
                "pp_tokens": empty_pp,
                "pp_index_zyx": empty_v_index.copy(),
                "ligand_area_target": zeros_target,
                "binding_area_target": zeros_target.copy(),
                "full_candidate_voxel_count": 0,
                "candidate_manual_features": np.zeros(CANDIDATE_MANUAL_FEATURE_DIM, dtype=np.float32),
                "candidate_sampling_features": np.zeros(CANDIDATE_SAMPLING_FEATURE_DIM, dtype=np.float32),
            }
        crop_start = np.asarray(centered["crop_start_local_zyx"][position], dtype=np.int64)
        box_origin = np.asarray(centered["box_origin_world"][position], dtype=np.float32)
        voxel_size = np.asarray(centered["voxel_size_world"][position], dtype=np.float32)
        v_tokens, v_index = _build_v_tokens(centered, position, crop_start)
        if self.variant == "density_blob_only":
            voxel_input = build_experimental_density_channel(centered["experimental_density_48"][position]).astype(np.float16)
            a_tokens = np.empty((0, A_TOKEN_DIM), dtype=np.float32)
            p_tokens = empty_p
            pp_tokens, pp_index = empty_pp, empty_v_index.copy()
        else:
            receptor_grid, receptor_hardmask = _scatter_receptor(
                payload["receptor_coords"], payload["receptor_features"], box_origin, voxel_size, crop_start
            )
            density = build_density_channels(
                centered["experimental_density_48"][position],
                centered["simulated_density_48"][position],
                receptor_hardmask,
            )
            voxel_input = np.concatenate((density, receptor_grid), axis=0).astype(np.float16)
            if voxel_input.shape[0] != VOXEL_INPUT_CHANNELS:
                raise RuntimeError("完整变体的体素输入必须为 106 通道。")
            a_tokens = _build_full_a_tokens(centered, position, crop_start)
            p_tokens = _build_p_tokens(centered, position, crop_start)
            pp_tokens, pp_index = _build_pp_tokens(
                centered["source_probability_48"][position], self.pp_top_k
            )
        ligand_target = _build_ligand_target(
            payload["occurrence_masks"], centered["box_start_zyx"][position], crop_start
        )
        binding_target = _build_binding_target(
            payload["receptor_coords"], payload["binding_atom"], box_origin, voxel_size, crop_start
        )
        manual = _manual_features(a_tokens, p_tokens, v_tokens, pp_tokens)
        full_count = int(centered["source_voxel_count"][position])
        if full_count <= 0:
            raise ValueError("source_voxel_count 必须是正整数。")
        sampling = np.asarray(
            (
                float(np.any(centered["crop_clipped_axis_mask"][position])),
                float(np.linalg.norm(centered["crop_center_offset_zyx"][position])),
                float(np.max(np.abs(centered["crop_center_offset_zyx"][position]))),
                len(v_tokens) / full_count,
            ),
            dtype=np.float32,
        )
        return {
            "voxel_input": voxel_input,
            "a_tokens": a_tokens,
            "p_tokens": p_tokens,
            "v_tokens": v_tokens,
            "v_index_zyx": v_index,
            "pp_tokens": pp_tokens,
            "pp_index_zyx": pp_index,
            "ligand_area_target": ligand_target,
            "binding_area_target": binding_target,
            "full_candidate_voxel_count": full_count,
            "candidate_manual_features": manual,
            "candidate_sampling_features": sampling,
        }

    def _load_pdb(self, pdb_id: str) -> dict[str, Any]:
        """一次读取当前变体实际需要的 PDB 数组和全部受支持语言向量。

        - pdb_id: manifest 中的小写 PDB 标识。
        - return: 可放入字节预算缓存的 PDB 载荷；不含 SMILES 图包副本。
        """

        centered_path = self.stage1_root / self.split / pdb_id / "centered/F2_centered.npz"
        centered = _load_npz_fields(centered_path, _centered_fields(self.variant))
        centered_indices = np.asarray(centered["centered_box_index"], dtype=np.int64)
        positions = {int(value): index for index, value in enumerate(centered_indices)}
        if len(positions) != len(centered_indices):
            raise ValueError(f"{pdb_id}: centered_box_index 存在重复值。")
        payload: dict[str, Any] = {
            "centered": centered,
            "position_by_centered_index": positions,
        }
        if self.variant != "pocket_only":
            receptor_path = self.data_root / "parse" / pdb_id / "receptor_tokens.npz"
            receptor_fields = ("coords",) if self.variant == "density_blob_only" else ("coords", "feat", "is_backbone")
            receptor = _load_npz_fields(receptor_path, receptor_fields)
            payload["receptor_coords"] = np.asarray(receptor["coords"], dtype=np.float32)
            if self.variant != "density_blob_only":
                payload["receptor_features"] = np.concatenate(
                    (
                        np.asarray(receptor["feat"], dtype=np.float32),
                        np.asarray(receptor["is_backbone"], dtype=np.float32)[:, None],
                    ),
                    axis=1,
                )
            labels = _load_npz_fields(
                self.data_root / "labels" / pdb_id / "atom_labels.npz", ("binding_atom",)
            )
            payload["binding_atom"] = np.asarray(labels["binding_atom"], dtype=np.bool_)
            ligand_path = self.data_root / "density" / pdb_id / "ligand_area.npz"
            with np.load(ligand_path, allow_pickle=False) as archive:
                names = sorted(
                    (name for name in archive.files if name.startswith("mask_") and name[5:].isdigit()),
                    key=lambda name: int(name[5:]),
                )
                payload["occurrence_masks"] = [
                    np.asarray(archive[name], dtype=np.int64).copy() for name in names
                ]
        embeddings: dict[str, np.ndarray] = {}
        for identity in self.identities_by_pdb.get(pdb_id, []):
            path = self.data_root / str(identity["language_relative_path"])
            embedding = _load_npz_fields(path, ("embedding",))["embedding"].astype(np.float32)
            if embedding.shape != (LANGUAGE_EMBEDDING_DIM,) or not np.isfinite(embedding).all():
                raise ValueError(f"{path}: embedding 不符合 (768,) 有限值契约。")
            embeddings[str(identity["smiles"])] = embedding
        payload["embedding_by_smiles"] = embeddings
        return payload


def load_manifest_dataset(
    manifest_path: Path,
    *,
    split: str,
    selected_only: bool,
    mode: DatasetMode,
    variant: ModelVariant,
    pp_top_k: int,
    cache_gib_per_worker: float,
) -> CandidateDataset:
    """建立正式管线唯一的 manifest Dataset。

    - manifest_path: 由 build_stage1_manifest 生成的版本化 manifest.json。
    - split: manifest 中的 train、validation 或测试划分名。
    - selected_only: True 时只保留 Stage1 candidate_selected=True 的候选。
    - mode: train/validation 会跳过缺少受支持身份的真前景；test 保留候选及原始排名。
    - variant: 四个具名模型变体之一，同时约束实际文件读取和模型输入字段。
    - pp_top_k: strongest/trunk_only 的 PP token 上限。
    - cache_gib_per_worker: 每个 Dataset 副本的 PDB 数组最近使用缓存预算。
    - return: 与模型变体同源的 CandidateDataset。
    """

    return CandidateDataset(
        manifest_path=manifest_path,
        split=split,
        selected_only=selected_only,
        mode=mode,
        variant=variant,
        pp_top_k=pp_top_k,
        cache_gib_per_worker=cache_gib_per_worker,
    )
