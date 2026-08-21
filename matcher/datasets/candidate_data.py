"""把冻结清单中的单个 Stage1 F1 候选物化为 Matcher 样本. 

CandidateDataset 以候选 JSONL 的顺序建立样本轴, 为每个候选组合 105 通道体素、
A/P/V/PP 四种变长 token、配体与结合区域监督以及当前 PDB 的联合身份轴. 体素与
PP 缓存分别委托 VoxelInputStore 和 PpInputStore; 本模块只按清单字段定位来源, 
不重新生成 manifest、LigandObject 或 occurrence 级语言向量. 
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from torch.utils.data import Dataset

from ..data.geometry import (
    candidate_sampling_features,
    keep_local_48_at,
    local_xyz_to_crop_centered_world,
    resolve_candidate_crop,
    voxel_zyx_to_crop_centered_world,
)
from ..data.ligand_graph import LigandGraphStore
from ..data.manifest import load_jsonl, read_npz_string_scalar
from ..data.pp_input import PpCandidateInput, PpInputStore
from ..data.schema import (
    A_TOKEN_DIM,
    BOX_ORIGIN_ABSOLUTE_TOLERANCE_ANGSTROM,
    CANDIDATE_MANUAL_FEATURE_DIM,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    FOREGROUND_CLASS_NAMES,
    FULL_CONTEXT_HALF_EXTENT_VOXELS,
    IDENTITY_KEY_FIELDS,
    LANGUAGE_EMBEDDING_DIM,
    LANGUAGE_FAMILIES,
    LOCAL_CROP_STRATEGIES,
    LOCAL_BOX_SIZE,
    LOCAL_CONTEXT_HALF_EXTENT_VOXELS,
    MANIFEST_SCHEMA_VERSION,
    P_TOKEN_DIM,
    PREPARED_SMILES_STATUSES,
    SourceRoots,
    V_TOKEN_DIM,
    IdentityScope,
    LocalCropStrategy,
    identity_scope_class_indices,
)
from ..data.voxel_input import VoxelInputStore
from ..utils.cache import ByteBudgetLru
from .contracts import CandidateItem, CandidateMetadata


TASK_NAMES = ("candidate_pretrain", "full_matcher")


class CandidateDataset(Dataset[CandidateItem]):
    """按候选粒度读取六分类样本, 并按需附加 PDB 内的联合身份轴. 

    生命周期：
        - 初始化: 读取 schema v4 manifest、一个划分的候选 JSONL 和身份 JSONL, 建立只读记录与按 PDB 分组的身份索引. 
        - 索引: 每次按全局候选下标物化一个 CandidateItem; 预算允许且写入缓存成功时, PDB 级大数组、身份输入和 LigandObject 图可由后续访问复用. 
        - 多进程复制: DataLoader 使用 spawn 序列化 Dataset 时, worker 副本不携带父进程已填充的两个本地缓存; 嵌套 Store 的序列化副本也不携带各自缓存, 父进程对象不被就地清空.  # NOTE：虽然我们改了stage1的落盘字段，但是"缓存什么,如何缓存以使I/O更快"这个问题仍然是通用的。过去的思路可以借鉴式地迭代改良。

    实例状态：
        - roots: SourceRoots; 冻结科学输入、Stage1 产物和 Matcher 可写缓存的根目录集合. 
        - split_name: 字符串; 当前物化划分名称. 
        - task_name: 字符串; candidate_pretrain 或 full_matcher. 
        - language_family: 字符串或 None; molformer、smi_ted_289m 或未选择语言家族. 
        - identity_scope: IdentityScope 或 None; 类型契约允许 small_molecule、all_non_background, 候选预训练使用 None; 构造器对 full_matcher 只检查非 None, 非法非空字符串会在首次物化身份轴时失败. 
        - pp_top_k: 正整数; 每个候选 PP token 数 K 等于 min(pp_top_k, 48³). 
        - local_crop_strategy: LocalCropStrategy; fixed_center 或 v_centroid, 统一决定 105 通道、V、PP 和两个区域目标的 48³ 起点.  # NOTE：新版本只支持后者，以减少复杂度
        - cache_bytes_per_worker: 非负整数; 单个 DataLoader 工作进程五类 ByteBudgetLru 的 array_nbytes 记账总上限, 不是进程 RSS 或全部 Python 对象内存上限, 也不是预分配量. 
        - records: JSON 对象列表; 候选 JSONL 的完整顺序, 每个元素对应一个 Dataset 下标. 
        - identities_by_pdb: PDB 小写标识到身份对象列表的映射; 每个列表按 class_index、object_key、prepared_smiles 排序. 
        - voxel_store: VoxelInputStore; 读取或构建当前裁剪策略的 105 通道体素缓存. 
        - pp_store: PpInputStore; 读取或构建当前裁剪策略和 pp_top_k 的 PP token 缓存. 
        - graph_store: LigandGraphStore; 按 object_key 读取 LigandObject 模板图, 并在字节预算允许时尝试缓存. 
        - _artifact_cache: ByteBudgetLru; 键为 split_name 与 pdb_id, 值为 PDB 级 centered、受体和监督数组. 
        - _identity_cache: ByteBudgetLru; 键为 pdb_id 与 language_family:identity_scope, 值为不含候选目标下标的可用身份轴. 
    """

    def __init__(
        self,
        roots: SourceRoots,
        manifest_path: Path,
        split_name: str,
        task_name: str,
        language_family: str | None,
        identity_scope: IdentityScope | None,   # NOTE: 这又是一个令人高度恶心高度冗余的字段！：只需要多分类的情况！
        pp_top_k: int,
        local_crop_strategy: LocalCropStrategy,
        cache_bytes_per_worker: int = 16 * 1024**3,
    ) -> None:
        """执行构造期现有配置门禁, 并建立当前划分的候选轴、身份索引和缓存读取器. 

        输入参数：
            - roots: SourceRoots; 三个根目录由当前 Dataset 与内部 Store 共同用于读取来源、构建缓存和定位产物. 
            - roots.data_root: Path; 本 Dataset 直接读取 atom_labels.npz:binding_atom、receptor_tokens.npz:coords 和 ligand_area.npz, 并委托 VoxelInputStore 读取 exp.npz、sim.npz 与 receptor_tokens.npz:feat, 委托 LigandGraphStore 读取 LigandObject, 再通过派生 language_root 读取语言 NPZ; 不直接读取 occurrences.jsonl. 
            - roots.stage1_root: Path; 提供按划分和 PDB 组织的 F1_centered.npz 与 probability_map.npz. 
            - roots.output_root: Path; 供 VoxelInputStore 和 PpInputStore 定位版本化 105 通道体素缓存与 PP token 缓存; 缓存缺失时对应 Store 会从来源构建并写入, 不用于推导本次 manifest_path. 
            - manifest_path: Path; 显式传入的 schema v4 manifest.json, 独立于 roots.output_root 推导; 父目录同时承载该 manifest 声明的候选和身份 JSONL. 
            - split_name: 字符串; 必须是 manifest.splits 已声明的划分键. 
            - task_name: 字符串; candidate_pretrain 只返回候选任务所需的空身份轴, full_matcher 还物化语言表示和配体图. 
            - language_family: 字符串或 None; 非 None 时必须是 molformer 或 smi_ted_289m; full_matcher 必须指定, candidate_pretrain 未强制为 None. 
            - identity_scope: IdentityScope 或 None; API 类型契约允许 full_matcher 使用 small_molecule 或 all_non_background, candidate_pretrain 必须为 None; 运行时构造器只检查 full_matcher 非 None, 不在此处检查非空字符串是否属于枚举. 
            - pp_top_k: 正整数; 每个候选保留 K=min(pp_top_k, 48³) 个最高概率 PP token, 并参与缓存路径. 
            - local_crop_strategy: LocalCropStrategy; fixed_center 使用 80³ 中心裁剪, v_centroid 以完整 V 质心请求裁剪中心并限制到有效起点. 
            - cache_bytes_per_worker: 非负整数; 单个 Dataset 实例五类 LRU 的 array_nbytes 记账总上限, 默认 16 GiB; 不统计 records、identities_by_pdb、字符串与容器开销或批次张量, 也不立即分配对应内存. 

        manifest 读取字段：
            - manifest.schema_version: 整数; 必须等于 MANIFEST_SCHEMA_VERSION, 即当前联合身份 schema 版本. 
            - manifest.identity_key_fields: 字符串列表; 必须按 object_key、prepared_smiles 的固定顺序出现. 
            - manifest.class_names: 字符串列表; 必须与 CLASS_NAMES 的六分类顺序完全一致. 
            - manifest.splits: 映射; 必须含 split_name. 
            - manifest.splits[split_name].candidates: 相对路径字符串; 定位候选 JSONL. 
            - manifest.splits[split_name].identities: 相对路径字符串; 定位身份 JSONL. 

        缓存预算：
            - artifact: 总预算减去 graph、identity、source、pp 四项向下取整份额后的全部余量; 只有总字节数可被 32 整除时才恰为 13/16; 供本 Dataset 尝试保存 centered、受体坐标、binding_atom 和 occurrence 稀疏掩码. 
            - source: 总预算的 1/16; 供 VoxelInputStore 尝试保存构建 105 通道所需的来源数组. 
            - pp: 总预算的 1/16; 供 PpInputStore 尝试保存 PDB 级 PP 缓存. 
            - identity: 总预算的 1/32; 供本 Dataset 尝试保存已按 PDB、语言家族和身份范围筛选的身份输入. 
            - graph: 总预算的 1/32; 供 LigandGraphStore 尝试保存 object_key 对应的模板图. 

        返回状态：
            - self.records: 候选 JSON 对象列表; 顺序与 candidates JSONL 一致. 
            - self.identities_by_pdb: PDB 标识到已验证并排序身份对象列表的映射. 
        """

        if task_name not in TASK_NAMES:
            raise ValueError(f"task_name 必须属于 {TASK_NAMES}, 实际为 {task_name!r}. ")
        if task_name == "full_matcher" and language_family is None:
            raise ValueError("full_matcher 必须明确指定 language_family. ")
        if task_name == "full_matcher" and identity_scope is None:
            raise ValueError("full_matcher 必须明确指定 identity_scope. ")
        if task_name == "candidate_pretrain" and identity_scope is not None:
            raise ValueError("candidate_pretrain 的 identity_scope 必须为 None. ")
        if language_family is not None and language_family not in LANGUAGE_FAMILIES:
            raise ValueError(
                f"language_family 必须属于 {LANGUAGE_FAMILIES} 或为 None, "
                f"实际为 {language_family!r}. "
            )
        if pp_top_k <= 0:
            raise ValueError("pp_top_k 必须为正整数. ")
        if local_crop_strategy not in LOCAL_CROP_STRATEGIES:
            raise ValueError(
                f"local_crop_strategy 必须属于 {LOCAL_CROP_STRATEGIES}. "
            )
        if cache_bytes_per_worker < 0:
            raise ValueError("cache_bytes_per_worker 不能为负数. ")
        # JSON 对象, manifest.json 的完整顶层内容; 这里只核验 Dataset 直接依赖的版本、联合键、类别和划分字段. 
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"{manifest_path} 的 schema_version 必须为 {MANIFEST_SCHEMA_VERSION}. "
            )
        if manifest.get("identity_key_fields") != list(IDENTITY_KEY_FIELDS):
            raise ValueError(f"{manifest_path} 的联合身份字段顺序与代码不一致. ")
        if tuple(manifest["class_names"]) != CLASS_NAMES:
            raise ValueError(f"{manifest_path} 的六分类顺序与代码不一致. ")
        # JSON 对象或 None, 当前划分声明的 candidates/identities 相对路径; None 表示 manifest 未物化该划分. 
        split_files = manifest["splits"].get(split_name)
        if split_files is None:
            raise KeyError(f"{manifest_path} 不含划分 {split_name!r}. ")

        self.roots = roots
        self.split_name = split_name
        self.task_name = task_name
        self.language_family = language_family
        self.identity_scope = identity_scope
        self.pp_top_k = pp_top_k
        self.local_crop_strategy = local_crop_strategy
        self.cache_bytes_per_worker = int(cache_bytes_per_worker)
        # JSON 对象列表, 候选 JSONL 的文件顺序就是 Dataset 全局样本下标顺序. 
        self.records = load_jsonl(manifest_path.parent / split_files["candidates"])
        # JSON 对象列表, 当前划分全部 PDB 内联合身份记录; 验证后按 PDB 分组但不跨 PDB 合并身份轴. 
        identity_records = load_jsonl(manifest_path.parent / split_files["identities"])
        _validate_identity_records(identity_records, split_name)
        # defaultdict[str, list[dict[str, Any]]], 把身份 JSON 对象按 PDB 局部化; 每个列表随后按类别和联合键排序. 
        self.identities_by_pdb: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for identity in identity_records:
            self.identities_by_pdb[identity["pdb_id"]].append(identity)
        for identities in self.identities_by_pdb.values():
            identities.sort(
                key=lambda item: (
                    item["class_index"],
                    item["object_key"],
                    item["prepared_smiles"],
                )
            )

        # str 到 int 的映射, 把单 worker 总上限无重叠地分给 artifact/source/pp/identity/graph 五类缓存. 
        budgets = _split_cache_budget(self.cache_bytes_per_worker)
        # int, VoxelInputStore 构建体素输入时复用来源数组的字节上限. 
        source_cache_bytes = budgets["source"]
        # int, PDB 级 centered、受体坐标和监督数组的字节上限. 
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
        """返回当前物化划分的候选数量. 

        返回值：
            - candidate_count: 整数; 等于 candidates JSONL 的对象数, 整 PDB 排除项不出现在该文件中. 
        """

        return len(self.records)

    def __getitem__(self, index: int) -> CandidateItem:
        """按全局候选下标物化体素、四模态、区域监督和可选身份输入. 

        输入参数：
            - index: 整数; 索引 self.records 的全局候选位置, 越界语义沿用 Python 列表. 

        候选记录字段：
            - record.pdb_id: JSON 值; 本函数执行 str 转换后用于来源路径和缓存键, 不在此重新核验非空或小写, 正式 manifest 负责该契约. 
            - record.candidate_index: JSON 值; 本函数执行 int 转换后索引 centered 候选轴、PP 缓存和体素缓存. 
            - record.full_candidate_voxel_count: JSON 值; 本函数执行 int 转换, 与实际完整 V 数量核对后写入输出. 
            - record.class_index: JSON 值; 本函数执行 int 转换后写入六分类目标并参与身份目标定位, 不在此重新核验 [0, 5] 范围. 
            - record.matched_object_key: 字符串或 None; 本函数不另做类型检查; 有 occurrence 通过阈值时保留其模板键, type_tag=other 背景也可非空, 未命中时为 None. 
            - record.matched_prepared_smiles: JSON 值; available 状态下必须是非空精确字符串, 其余三种状态必须为 None. 
            - record.matched_prepared_smiles_status: JSON 值; 本函数要求属于 available、explicit_null、missing_file、not_applicable. 
            - record.identity_seen_in_train_by_language: JSON 对象; 选择 language_family 时读取对应值, 不在此重新核验对象键或 bool/None 类型; 缺少完整前景联合身份时正式值为 None. 
            - record.identity_seen_in_train_by_language[language_family]: 布尔值或 None; 具备完整前景 object_key+prepared_smiles 时为 train 可见性, 否则为 None. 

        返回字段：
            - item.metadata: CandidateMetadata; 候选标识、命中联合身份字段及当前语言家族的 train 可见性. 
            - item.metadata.split: 字符串; self.split_name. 
            - item.metadata.pdb_id: 非空小写字符串; 候选记录所属 PDB. 
            - item.metadata.candidate_index: 整数; 当前 PDB 内的 Stage1 F1 候选编号. 
            - item.metadata.matched_object_key: 字符串或 None; 通过覆盖阈值的 occurrence 模板键; type_tag=other 映射成 background 时也可非空, 未命中时为 None. 
            - item.metadata.matched_prepared_smiles: 字符串或 None; 仅五类前景命中的顶层 PreparedLigand.smiles 可非空; type_tag=other 背景和未命中为 None. 
            - item.metadata.matched_prepared_smiles_status: PreparedSmilesStatus; available、explicit_null、missing_file 或 not_applicable. 
            - item.metadata.identity_seen_in_train: 布尔值或 None; 选择语言家族且候选具备完整前景联合身份时读取对应 train 可见性; 背景、未命中、联合键不完整或未选语言家族时为 None. 
            - item.voxel_input: float16 数组, (105, 48, 48, 48); 当前 crop_start_zyx 的候选体素输入, 空间轴依次为 Z/Y/X. 
            - item.a_tokens: float32 数组, (N_A, 229); 完整 80³ A 表, 不按 48³ 裁剪筛选. 
            - item.p_tokens: float32 数组, (N_P, 116); 完整 80³ P 表, 不按 48³ 裁剪筛选. 
            - item.v_tokens: float32 数组, (N_V, 52); 仅保留当前 48³ 裁剪内的 V 表. 
            - item.v_index_zyx: int64 数组, (N_V, 3); 与 v_tokens 逐项对齐的裁剪内 Z/Y/X 下标. 
            - item.pp_tokens: float32 数组, (K, 4); K=min(pp_top_k, 48³), 四列依次为概率和相对 X/Y/Z. 
            - item.pp_index_zyx: int64 数组, (K, 3); 与 pp_tokens 逐项对齐的裁剪内 Z/Y/X 下标. 
            - item.class_target: 整数; 候选 JSONL 中的六分类下标. 
            - item.ligand_area_target: bool 数组, (48, 48, 48); 全部 occurrence 稀疏掩码落入当前裁剪的并集. 
            - item.binding_area_target: bool 数组, (48, 48, 48); 完整受体表中 binding_atom 落入当前裁剪的并集. 
            - item.full_candidate_voxel_count: 整数; 候选 JSONL 冻结的裁剪前完整 V 数量, 并与实际 centered 切片数量核对. 
            - item.candidate_manual_features: float32 数组, (12,); A/P/V/PP 各自的 token 数、概率均值和概率最大值. 
            - item.candidate_sampling_features: float32 数组, (4,); 边界限制标志、裁剪中心到 V 质心的 L2/L∞ 体素距离和 V 保留比例. 
            - item.identity_object_keys: 长度 I 的字符串列表; 当前身份轴的 LigandObject 模板键. 
            - item.identity_prepared_smiles: 长度 I 的非空字符串列表; 与 object_key 共同定义每个身份槽. 
            - item.identity_class_index: int64 数组, (I,); 身份轴的前景类别下标. 
            - item.language_embeddings: float32 数组, (I, 768); 当前语言家族的身份表示. 
            - item.ligand_graphs: 长度 I 的 LigandGraph 列表; 每个身份槽一张模板图. 
            - item.ligand_graphs[*].element: int64 数组, (N_atom,), 模板原子序数. 
            - item.ligand_graphs[*].charge: int64 数组, (N_atom,), 模板原子形式电荷. 
            - item.ligand_graphs[*].residue_id: int64 数组, (N_atom,), 模板内部残基编号. 
            - item.ligand_graphs[*].node_continuous: float32 数组, (N_atom, 15), 节点连续特征. 
            - item.ligand_graphs[*].edge_index: int64 数组, (2, 2*N_bond), 模板双向边端点. 
            - item.ligand_graphs[*].edge_features: float32 数组, (2*N_bond, 10), 与边逐项对齐的特征. 
            - item.identity_target: 整数; 真值在当前身份轴的下标, 身份不可监督时为 -1; type_tag=other 背景即使保留非空 object_key 也因 prepared_smiles=None 固定为 -1. 

        边界：
            - candidate_pretrain: 返回 I=0 的空身份数组、空列表和 identity_target=-1. 
            - full_matcher: 只把当前范围内 graph_available 与所选语言 available 都为真的身份加入轴, 目标下标仍按候选联合真值重新计算; 非法非空 identity_scope 会在首次执行该分支时由 identity_scope_class_indices 拒绝. 
        副作用：
            - output_root 缓存: 首次访问缺失的体素或 PP 缓存时, voxel_store 或 pp_store 会按既有 Store 契约加锁构建并原子发布对应文件. 
            - 进程内 LRU: artifact、PP PDB、identity、graph 与体素构建来源会在当前 Dataset 实例所在进程中按各自字节预算尝试写入 LRU; 预算为零或载荷超出预算时不保留, DataLoader 场景中的该进程是 worker. 
            - 候选体素输入: voxel_store 以 output_root 中的 NPY 作为候选级缓存; 命中已有 NPY 时不填充来源 LRU, 只有构建缺失 NPY 时才读取并尝试缓存来源数组. 
        """

        # JSON 对象, 当前全局 Dataset 下标对应的候选记录; 后续候选级索引均以其中的 pdb_id/candidate_index 为准. 
        record = self.records[index]
        # str, 正式 manifest 契约规定为小写的 PDB 标识; 本行只执行 str 转换而不重新核验, 用于定位来源和缓存键. 
        pdb_id = str(record["pdb_id"])
        # int, 当前 PDB 内从零开始的候选位置, 用于 offsets、PP 缓存和体素缓存. 
        candidate_index = int(record["candidate_index"])
        # dict[str, Any], 当前 PDB 的 centered 字段、完整受体坐标、binding_atom 和 occurrence 稀疏掩码. 
        artifacts = self._load_pdb_artifacts(pdb_id)
        # dict[str, np.ndarray], F1_centered.npz 中 Dataset 需要的全部 PDB 级数组副本. 
        centered = artifacts["centered"]
        # float32, (3,), 当前候选世界 X/Y/Z 单体素尺寸, 单位 Å. 
        voxel_size = np.asarray(centered["voxel_size_world"][candidate_index], dtype=np.float32)
        # int64, (3,), 当前候选 80³ 框在全局密度网格中的 Z/Y/X 起点. 
        box_start = np.asarray(centered["box_start_zyx"][candidate_index], dtype=np.int64)
        # 两个 int, 当前候选完整 V 在拼接 voxel 数组中的半开切片边界. 
        v_start, v_stop = _offset_range(centered["voxel_offsets"], candidate_index)
        # int64, (N_V_full, 3), 裁剪前完整 V 在当前 80³ 框内的 Z/Y/X 下标. 
        full_v_index = np.asarray(
            centered["voxel_index_local_zyx"][v_start:v_stop],
            dtype=np.int64,
        )
        # CandidateCropGeometry, 固定中心或 V 质心策略解析出的 48³ 起点、V 质心、保留数与距离诊断. 
        crop_geometry = resolve_candidate_crop(full_v_index, self.local_crop_strategy)
        if crop_geometry.full_v_count != int(record["full_candidate_voxel_count"]):
            raise ValueError(
                f"{self.split_name}/{pdb_id}/candidate_{candidate_index} 的权威 V 数量"
                "与冻结清单不一致. "
            )
        # PpCandidateInput, 当前候选 K 个 PP token、局部索引及用于跨缓存核验的空间字段. 
        pp_input = self.pp_store.load(self.split_name, pdb_id, candidate_index)
        _validate_candidate_geometry(
            centered,
            pp_input,
            candidate_index,
            crop_geometry.start_zyx,
        )

        # NOTE: 我注意到，这个版本的代码直接把原本的多尺度特征进行了粗暴的拼接。粗暴拼接+MLP的融合方式太糙了！
        # float32, (N_A, 229), 完整 80³ A 表及相对当前 48³ 中心的坐标特征. 
        a_tokens = _build_a_tokens(
            centered,
            candidate_index,
            voxel_size,
            crop_geometry.start_zyx,
        )
        # bool, (48, 48, 48), 完整受体 binding_atom 散射到当前裁剪后的 Z/Y/X 体素掩码. 
        binding_target = _build_binding_target(
            artifacts["receptor_coords"],
            artifacts["binding_atom"],
            np.asarray(centered["box_origin_world"][candidate_index], dtype=np.float32),
            voxel_size,
            crop_geometry.start_zyx,
        )
        # float32, (N_P, 116), 完整 80³ P 表及相对当前 48³ 中心的坐标特征. 
        p_tokens = _build_p_tokens(
            centered,
            candidate_index,
            voxel_size,
            crop_geometry.start_zyx,
        )
        # float32 (N_V, 52) 与 int64 (N_V, 3), 只保留当前 48³ 内的 V 特征及逐 token Z/Y/X 下标. 
        v_tokens, v_index_zyx = _build_v_tokens(
            centered,
            v_start,
            v_stop,
            full_v_index,
            voxel_size,
            crop_geometry.start_zyx,
        )
        # bool, (48, 48, 48), 当前 PDB 全部 occurrence 掩码在候选裁剪内的并集. 
        ligand_target = _build_ligand_target(
            artifacts["occurrence_masks"],
            box_start,
            crop_geometry.start_zyx,
        )
        # PreparedSmilesStatus, 候选顶层 PreparedLigand.smiles 可用性; 未命中或含 type_tag=other 的背景均为 not_applicable. 
        matched_prepared_smiles_status = record["matched_prepared_smiles_status"]
        if matched_prepared_smiles_status not in PREPARED_SMILES_STATUSES:
            raise ValueError(
                f"{self.split_name}/{pdb_id}/candidate_{candidate_index} 的 "
                "matched_prepared_smiles_status 无效. "
            )
        # str 或 None; available 状态必须是非空顶层字符串, 其它三种状态必须为 None. 
        matched_prepared_smiles = record["matched_prepared_smiles"]
        if matched_prepared_smiles_status == "available":
            if (
                type(matched_prepared_smiles) is not str
                or not matched_prepared_smiles.strip()
            ):
                raise ValueError("available 状态必须对应非空 matched_prepared_smiles. ")
        elif matched_prepared_smiles is not None:
            raise ValueError("非 available 状态的 matched_prepared_smiles 必须为 null. ")

        # CandidateMetadata, 保持候选定位、联合真值和所选语言家族的 train 可见性, 不参与数值补零. 
        metadata: CandidateMetadata = {
            "split": self.split_name,
            "pdb_id": pdb_id,
            "candidate_index": candidate_index,
            "matched_object_key": record["matched_object_key"],
            "matched_prepared_smiles": matched_prepared_smiles,
            "matched_prepared_smiles_status": matched_prepared_smiles_status,
            "identity_seen_in_train": (
                record["identity_seen_in_train_by_language"][self.language_family]
                if self.language_family is not None
                else None
            ),
        }
        # dict[str, Any], 先装配候选分类、四模态、体素和区域监督, 随后按 task_name 补充身份字段. 
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
                    record["matched_prepared_smiles"],
                )
            )
        else:
            item.update(
                identity_object_keys=[],
                identity_prepared_smiles=[],
                identity_class_index=np.empty(0, dtype=np.int64),
                language_embeddings=np.empty((0, LANGUAGE_EMBEDDING_DIM), dtype=np.float32),
                ligand_graphs=[],
                identity_target=-1,
            )
        return cast(CandidateItem, item)

    def _load_pdb_artifacts(self, pdb_id: str) -> dict[str, Any]:
        """读取一个 PDB 的 centered、受体和监督数组, 并按字节预算尝试缓存. 

        输入参数：
            - pdb_id: 非空小写字符串; 与 self.split_name 共同定位 Stage1 来源, 与 data_root 共同定位受体和标签来源. 

        返回字段：
            - cached.centered: 字符串到 NumPy 数组的映射; 每个值都是 F1_centered.npz 对应字段的内存副本, 加载时不统一转换 dtype. 
            - cached.centered.box_start_zyx: 来源契约为 int32 (C, 3); 每个 80³ 候选在完整网格中的 Z/Y/X 起点. 
            - cached.centered.box_origin_world: 来源契约为 float32 (C, 3); 每个 80³ 候选低角点的世界 X/Y/Z 坐标, 单位 Å. 
            - cached.centered.voxel_size_world: 来源契约为 float32 (C, 3); 每个候选 X/Y/Z 单体素尺寸, 单位 Å/voxel. 
            - cached.centered.voxel_offsets: 来源契约为 int64 (C+1,); 相邻值同时切分 voxel_index_local_zyx、centered_probability 和 voxel_final. 
            - cached.centered.voxel_index_local_zyx: 来源契约为 int16 (V_total, 3); 拼接后的候选 80³ 内 Z/Y/X 索引. 
            - cached.centered.centered_probability: 来源契约为 float32 (V_total,); 与 voxel_index_local_zyx 逐体素对齐的 Stage1 概率. 
            - cached.centered.voxel_final: 来源契约为 float16 (V_total, 48); 与 voxel_index_local_zyx 逐体素对齐的 V token 主体. 
            - cached.centered.P_offsets: 来源契约为 int64 (C+1,); 相邻值同时切分全部 P_ 前缀拼接数组. 
            - cached.centered.P_coord_local_xyz: 来源契约为 float32 (P_total, 3); 各 P 在所属 80³ 候选内的连续 X/Y/Z 体素坐标. 
            - cached.centered.P_probability: 来源契约为 float32 (P_total,); 与 P 坐标逐项对齐的 Stage1 概率. 
            - cached.centered.P_feat_L2: 来源契约为 float16 (P_total, 64); P 的 L2 特征. 
            - cached.centered.P_feat_L3: 来源契约为 float16 (P_total, 48); P 的 L3 特征. 
            - cached.centered.A_offsets: 来源契约为 int64 (C+1,); 相邻值同时切分全部 A_ 前缀拼接数组. 
            - cached.centered.A_global_index: 来源契约为 int64 (A_total,); A 在完整受体原子表中的下标, 本 Dataset 当前不把它写入 CandidateItem. 
            - cached.centered.A_coord_local_xyz: 来源契约为 float32 (A_total, 3); 各 A 在所属 80³ 候选内的连续 X/Y/Z 体素坐标. 
            - cached.centered.A_probability: 来源契约为 float32 (A_total,); 与 A 坐标逐项对齐的 Stage1 概率. 
            - cached.centered.A_feat_L0: 来源契约为 float32 (A_total, 49); A 的受体基础特征. 
            - cached.centered.A_feat_L1: 来源契约为 float16 (A_total, 64); A 的 L1 特征. 
            - cached.centered.A_feat_L2: 来源契约为 float16 (A_total, 64); A 的 L2 特征. 
            - cached.centered.A_feat_L3: 来源契约为 float16 (A_total, 48); A 的 L3 特征. 
            - cached.binding_atom: bool 数组, (N_receptor,); atom_labels.npz 中与完整受体表逐原子对齐的结合原子标志. 
            - cached.receptor_coords: float32 数组, (N_receptor, 3); receptor_tokens.npz 中有限世界 X/Y/Z 坐标, 单位 Å. 
            - cached.occurrence_masks: int64 数组列表; 每项形状预期为 (V_occurrence, 3), 保存一个 mask_数字 字段的完整网格 Z/Y/X 稀疏下标. 

        读取边界：
            - centered: 只复制 _CENTERED_FIELDS 列出的二十个字段; NPZ 缺字段时由 NumPy 访问失败. 
            - ligand_area: 只收集字段名满足 mask_ 后接十进制数字的数组; 字段遍历顺序不用于监督语义, 因为最终只构造并集. 
            - validation: 本函数核验 receptor_coords 的形状和有限值, 以及 binding_atom 的逐原子长度; centered 与 occurrence mask 的详细形状在实际消费者中按需暴露错误. 
        """

        # tuple[str, str], PDB 级 artifact 缓存键; 不同划分的同名 PDB 不共享 Stage1 centered 数组. 
        key = (self.split_name, pdb_id)
        # dict[str, Any] 或 None; 命中时直接复用同一组 PDB 级数组对象. 
        cached = self._artifact_cache.get(key)
        if cached is not None:
            return cached
        # Path, 当前划分和 PDB 的 Stage1 F1-centered 归档. 
        centered_path = (
            self.roots.stage1_root / self.split_name / pdb_id / "centered/F1_centered.npz"
        )
        with np.load(centered_path, allow_pickle=False) as archive:
            # dict[str, np.ndarray], 二十个指定字段的独立内存副本; 不依赖已关闭归档, 预算允许时可由缓存持有. 
            centered = {name: np.asarray(archive[name]).copy() for name in _CENTERED_FIELDS}
        # Path, 当前 PDB 的完整受体逐原子监督归档. 
        label_path = self.roots.data_root / "labels" / pdb_id / "atom_labels.npz"
        with np.load(label_path, allow_pickle=False) as archive:
            # bool, (N_receptor,), 与 receptor_tokens.npz 的完整原子顺序对齐. 
            binding_atom = np.asarray(archive["binding_atom"], dtype=bool)
        # Path, 当前 PDB 的完整受体 token 归档; 本函数只读取世界坐标. 
        receptor_path = self.roots.data_root / "parse" / pdb_id / "receptor_tokens.npz"
        with np.load(receptor_path, allow_pickle=False) as archive:
            # float32, (N_receptor, 3), 完整受体原子的世界 X/Y/Z, 单位 Å. 
            receptor_coords = np.asarray(archive["coords"], dtype=np.float32)
        if receptor_coords.ndim != 2 or receptor_coords.shape[1] != 3:
            raise ValueError(f"{receptor_path}:coords 必须为有限 (N,3). ")
        if not np.isfinite(receptor_coords).all():
            raise ValueError(f"{receptor_path}:coords 含 NaN 或 Inf. ")
        if binding_atom.shape != (receptor_coords.shape[0],):
            raise ValueError(f"{label_path}:binding_atom 必须与完整受体原子表逐项对齐. ")
        # Path, 当前 PDB 全部 occurrence 的稀疏配体区域归档. 
        ligand_path = self.roots.data_root / "density" / pdb_id / "ligand_area.npz"
        # list[np.ndarray], 每项为 int64 (V_occurrence, 3) 的完整网格 Z/Y/X 稀疏下标; 空列表产生全 False 目标. 
        occurrence_masks: list[np.ndarray] = []
        with np.load(ligand_path, allow_pickle=False) as archive:
            for name in archive.files:
                suffix = name.removeprefix("mask_")
                if name.startswith("mask_") and suffix.isdigit():
                    occurrence_masks.append(np.asarray(archive[name], dtype=np.int64))

        # dict[str, Any], 一个 PDB 的 centered、完整受体坐标和两类区域监督来源. 
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
        matched_prepared_smiles: str | None,
    ) -> dict[str, Any]:
        """物化当前 PDB 中语言表示与模板图都可用的联合身份轴. 

        输入参数：
            - pdb_id: 非空小写字符串; 只从 identities_by_pdb[pdb_id] 构造一个 PDB 内的身份轴, 不与其它 PDB 共同归一化分数. 
            - matched_class_index: 整数; 当前候选的六分类下标, 用于和联合键一起定位真实身份. 
            - matched_object_key: 字符串或 None; 通过覆盖阈值的 occurrence 模板键; type_tag=other 映射成 background 时仍可非空, 未命中时为 None. 
            - matched_prepared_smiles: 字符串或 None; 仅五类前景命中的顶层 PreparedLigand.smiles 可非空; 不可用、type_tag=other 背景或未命中时为 None. 

        使用的身份记录字段：
            - identity.class_index: 前景类别下标; 必须属于 identity_scope_class_indices 的返回集合才进入后续筛选. 
            - identity.object_key: 非空字符串; 定位 LigandObject 图并作为联合身份第一分量. 
            - identity.prepared_smiles: 非空字符串; 作为联合身份第二分量, 并与语言 NPZ 的 prepared_smiles 逐字核对. 
            - identity.graph_available: 布尔值; 为真才尝试由 graph_store 读取 object_key 对应文件. 
            - identity.language[language_family].available: 布尔值; 为真才把该身份加入当前语言家族的轴.   # NOTE: 你的代码又臭又长，令我恶心。language_family 不需要增加单独的字段————直接用目录名定义就好！类似的冗余定义还有很多，你必须广泛地全库自查并精简————我没有义务批注你这烂代码！
            - identity.language[language_family].selected_relative_path: 非空字符串; 相对 roots.language_root 定位被选中的 occurrence 语言 NPZ. 

        返回字段：
            - identity_object_keys: 长度 I 的字符串列表; 按 identities_by_pdb 已冻结顺序保留的 LigandObject 模板键. 
            - identity_prepared_smiles: 长度 I 的非空字符串列表; 与 object_key 逐槽共同定义身份. 
            - language_embeddings: float32 数组, (I, 768); 每个所选 NPZ 的 embedding, 必须有限. 
            - identity_class_index: int64 数组, (I,); 与联合身份字符串和语言向量逐槽对齐的前景类别. 
            - ligand_graphs: 长度 I 的 LigandGraph 列表; 每槽由对应 object_key 加载, 图缓存预算允许且写入成功时, 同键不同 smiles 可以复用同一个 dataclass 实例. 
            - ligand_graphs[*].element: int64 数组, (N_atom,), 模板原子序数. 
            - ligand_graphs[*].charge: int64 数组, (N_atom,), 模板原子形式电荷. 
            - ligand_graphs[*].residue_id: int64 数组, (N_atom,), 模板内部残基编号. 
            - ligand_graphs[*].node_continuous: float32 数组, (N_atom, 15), 模板节点连续特征. 
            - ligand_graphs[*].edge_index: int64 数组, (2, 2*N_bond), 模板双向边端点. 
            - ligand_graphs[*].edge_features: float32 数组, (2*N_bond, 10), 模板双向边特征. 
            - identity_target: 整数; 当前候选真值在以上平行身份轴中的唯一位置, 顺序按照(class_index, ligand_object, smiles)顺序排列, 缺少完整联合键或轴中无真值时为 -1; type_tag=other 背景因 prepared_smiles=None 固定为 -1. 

        缓存：
            - cache_key: pdb_id 与 language_family:identity_scope; 缓存的五个身份输入字段不含候选相关的 identity_target. 
            - identity_target: 无论缓存命中或新建身份轴都重新根据当前候选计算, 避免同一 PDB 不同候选共享真值下标. 
        """

        if self.language_family is None or self.identity_scope is None:
            raise AssertionError("full_matcher 的语言模型和身份范围不应为空. ")
        # tuple[str, str], 身份轴缓存键; 同一 PDB 的同语言家族和范围共享可用身份顺序. 
        cache_key = (pdb_id, f"{self.language_family}:{self.identity_scope}")
        # dict[str, Any] 或 None; 缓存值含五个候选无关身份字段, 不含 identity_target. 
        cached = self._identity_cache.get(cache_key)
        if cached is not None:
            # int, 当前候选真值在缓存身份轴中的位置; -1 表示无完整真值或不在可用轴. 
            target = _identity_target_index(
                cached["identity_object_keys"],
                cached["identity_prepared_smiles"],
                cached["identity_class_index"],
                matched_class_index,
                matched_object_key,
                matched_prepared_smiles,
            )
            return {**cached, "identity_target": target}
        # list[str], 按已排序 identity 记录累积的 LigandObject 模板键. 
        object_keys: list[str] = []
        # list[str], 与 object_keys 逐槽对齐的非空顶层 PreparedLigand.smiles. 
        prepared_smiles_values: list[str] = []
        # list[np.ndarray], 每项为 float32 (768,) 且与联合身份逐槽对齐的语言向量. 
        embeddings: list[np.ndarray] = []
        # list[int], 每个保留身份的前景六分类下标. 
        class_indices: list[int] = []
        # list[LigandGraph], 每个保留身份一张 object_key 定位的模板图. 
        graphs = []
        # set[int], identity_scope 允许进入身份轴的前景类别下标. 
        allowed_classes = set(identity_scope_class_indices(self.identity_scope))
        for identity in self.identities_by_pdb.get(pdb_id, []):
            class_index = identity["class_index"]
            if class_index not in allowed_classes:
                continue
            # JSON 对象, 当前身份在所选语言家族下的可用性和被选 occurrence 归档路径. 
            language = identity["language"][self.language_family]
            if not identity["graph_available"] or not language["available"]:
                continue
            object_key = identity["object_key"]
            prepared_smiles = identity["prepared_smiles"]
            # str, 所选 occurrence 语言 NPZ 相对 language_root 的路径; available=true 时必须非空. 
            relative_path = language["selected_relative_path"]
            if type(relative_path) is not str or not relative_path:    # NOTE: 令人恶心的检验！
                raise ValueError("可用语言身份必须保存非空 selected_relative_path. ")
            # Path, 当前联合身份实际读取的 occurrence 语言 NPZ. 
            path = self.roots.language_root / relative_path
            with np.load(path, allow_pickle=False) as archive:
                # float32, (768,), 语言模型公开接口为顶层 smiles 产生的冻结向量. 
                embedding = np.asarray(archive["embedding"], dtype=np.float32)
                # str, 语言 NPZ 保存的顶层 PreparedLigand.smiles, 用于防止向量与身份槽错配. 
                archive_prepared_smiles = read_npz_string_scalar(
                    archive["prepared_smiles"],
                    f"{path}:prepared_smiles",
                )
            if embedding.shape != (LANGUAGE_EMBEDDING_DIM,) or not np.isfinite(embedding).all():  # NOTE: 令人恶心的检验！
                raise ValueError(f"{path}:embedding 必须为有限 ({LANGUAGE_EMBEDDING_DIM},). ")
            if archive_prepared_smiles != prepared_smiles:  # NOTE: 令人恶心的检验！
                raise ValueError(
                    f"{path}:prepared_smiles={archive_prepared_smiles!r}, "
                    f"与联合身份 {prepared_smiles!r} 不一致. "
                )
            object_keys.append(object_key)   # NOTE: 所有的这些变量都没有举例子、都没注释。恶心！
            prepared_smiles_values.append(prepared_smiles)
            embeddings.append(embedding)
            class_indices.append(class_index)
            graphs.append(self.graph_store.load(object_key))
        # float32, (I, 768), 按身份轴堆叠; I=0 时保持明确的二维空形状. 
        embedding_array = (
            np.stack(embeddings, axis=0)
            if embeddings
            else np.empty((0, LANGUAGE_EMBEDDING_DIM), dtype=np.float32)
        )
        # dict[str, Any], 候选无关的身份轴字段; 预算允许且 put 保留后, 后续同 PDB 候选可复用数组、列表和图实例. 
        cached = {
            "identity_object_keys": object_keys,
            "identity_prepared_smiles": prepared_smiles_values,
            "language_embeddings": embedding_array,
            "identity_class_index": np.asarray(class_indices, dtype=np.int64),
            "ligand_graphs": graphs,
        }
        if len(set(zip(object_keys, prepared_smiles_values))) != len(object_keys):
            raise ValueError("同一 PDB 身份轴含重复联合身份. ")
        self._identity_cache.put(cache_key, cached)
        # int, 当前候选真值在新建身份轴中的位置; 不写入缓存. 
        target = _identity_target_index(
            object_keys,
            prepared_smiles_values,
            cached["identity_class_index"],
            matched_class_index,
            matched_object_key,
            matched_prepared_smiles,
        )
        return {**cached, "identity_target": target}

    def __getstate__(self) -> dict[str, Any]:
        """返回供 spawn 序列化的 Dataset 状态, 使 worker 副本不携带父进程已填充的本地缓存. 

        返回字段：
            - state.roots: SourceRoots; 来源和 Matcher 缓存根目录. 
            - state.split_name: 字符串; 当前物化划分. 
            - state.task_name: 字符串; candidate_pretrain 或 full_matcher. 
            - state.language_family: 字符串或 None; 当前语言家族. 
            - state.identity_scope: IdentityScope 或 None; 当前身份范围. 
            - state.pp_top_k: 正整数; PP token 上限. 
            - state.local_crop_strategy: LocalCropStrategy; 当前 48³ 裁剪策略. 
            - state.cache_bytes_per_worker: 非负整数; 单 worker 五类 ByteBudgetLru 的 array_nbytes 记账总上限, 不是进程 RSS 上限. 
            - state.records: JSON 对象列表; 候选轴完整记录, 不清空并参与 spawn 序列化. 
            - state.identities_by_pdb: PDB 标识到已排序身份对象列表的映射, 不清空并参与 spawn 序列化. 
            - state.voxel_store: VoxelInputStore; 其序列化状态由该对象自己的钩子替换为空缓存, 原对象不就地修改. 
            - state.pp_store: PpInputStore; 其序列化状态由该对象自己的钩子替换为空缓存, 原对象不就地修改. 
            - state.graph_store: LigandGraphStore; 其序列化状态由该对象自己的钩子替换为空缓存, 原对象不就地修改. 
            - state._artifact_cache: 新建空 ByteBudgetLru; 字节上限为总预算的 artifact 份额. 
            - state._identity_cache: 新建空 ByteBudgetLru; 字节上限为总预算的 identity 份额. 

        本方法不修改 self 中现有缓存; 它只替换返回字典中的两个缓存对象. 
        """

        # dict[str, Any], self.__dict__ 的浅拷贝; 嵌套 Store 仍由各自序列化方法处理. 
        state = self.__dict__.copy()
        # dict[str, int], 根据总预算重新计算当前 worker 应持有的五类缓存上限. 
        budgets = _split_cache_budget(self.cache_bytes_per_worker)
        state["_artifact_cache"] = ByteBudgetLru(budgets["artifact"])
        state["_identity_cache"] = ByteBudgetLru(budgets["identity"])
        return state


# ================================================================================================


def _validate_identity_records(
    records: list[dict[str, Any]],
    split_name: str,
) -> None:
    """核验身份 JSONL 中 Dataset 建轴直接依赖的字段和联合键唯一性. 

    输入参数：
        - records: JSON 对象列表; 当前划分 identity JSONL 的完整对象序列, 函数不修改对象或顺序. 
        - split_name: 字符串; Dataset 请求的划分名称, 每条记录的 split 必须逐字相同. 

    逐记录字段：
        - record.split: 字符串; 必须等于 split_name. 
        - record.pdb_id: 非空小写字符串; 与 object_key、prepared_smiles 一起参与本函数的重复键检查. 
        - record.class_name: 字符串; 必须是 ion、nucleotide_like、peptide_like、small_molecule、sugar 之一. 
        - record.class_index: 精确 int; 必须等于 CLASS_TO_INDEX[class_name], 不接受可转换字符串或布尔值. 
        - record.object_key: 非空精确字符串; 联合身份第一分量并定位 LigandObject 图. 
        - record.prepared_smiles: 非空精确字符串; 联合身份第二分量, 逐字采用 manifest 已冻结的顶层值. 
        - record.occurrence_candidate_ids: 非空整数列表; 元素必须为非负精确 int, 并已升序去重. 
        - record.graph_available: 精确 bool; 表示 manifest 扫描时对应 LigandObject 文件存在. 
        - record.graph_relative_path: 字符串; 本函数只核验 JSON 类型, 不核验路径存在或与 object_key 的对应关系. 
        - record.language: JSON 对象; 键集合必须恰好是 molformer 和 smi_ted_289m. 
        - record.language.molformer.available: 精确 bool; 本函数不继续核验该证据对象的路径、计数或失败 occurrence 字段. 
        - record.language.smi_ted_289m.available: 精确 bool; 本函数不继续核验该证据对象的路径、计数或失败 occurrence 字段. 
        - record.seen_in_train_by_language: JSON 对象; 键集合必须恰好是两个语言家族. 
        - record.seen_in_train_by_language.molformer: 精确 bool; 联合键在 molformer 可用前提下是否曾是 train 正目标. 
        - record.seen_in_train_by_language.smi_ted_289m: 精确 bool; 联合键在 smi_ted_289m 可用前提下是否曾是 train 正目标. 

    状态与返回：
        - seen_keys: 三元组集合; 元素为 pdb_id、object_key、prepared_smiles, 只用于拒绝同一 PDB 内重复联合身份. 
        - return: None; 全部检查通过时不产生新对象, 失败时抛出带 JSONL 位置前缀的 ValueError. 

    本函数不是完整 identity schema 验证器; selected_relative_path 等字段只在身份实际进入当前语言轴时由 _load_identity_inputs 检查. 
    """

    # set[tuple[str, str, str]], 已经通过基本字段检查的 PDB 局部联合键. 
    seen_keys: set[tuple[str, str, str]] = set()
    for record_index, record in enumerate(records):
        # str, 错误位置前缀, 使用当前划分和 JSONL 对象序号定位失败字段. 
        prefix = f"{split_name}_identities.jsonl[{record_index}]"
        if record.get("split") != split_name:
            raise ValueError(f"{prefix}.split 与请求划分不一致. ")
        # Any, 身份所属 PDB; 必须是非空小写精确字符串, 不能依靠 str 转换. 
        pdb_id = record.get("pdb_id")
        if type(pdb_id) is not str or not pdb_id or pdb_id != pdb_id.lower():
            raise ValueError(f"{prefix}.pdb_id 必须是非空小写字符串. ")
        # Any, 五类前景类别名称; 背景不建立身份记录. 
        class_name = record.get("class_name")
        # Any, 必须是与 class_name 固定映射一致的精确 int. 
        class_index = record.get("class_index")
        if class_name not in FOREGROUND_CLASS_NAMES:
            raise ValueError(f"{prefix}.class_name 必须属于五类非背景类别. ")
        if type(class_index) is not int or class_index != CLASS_TO_INDEX[class_name]:
            raise ValueError(f"{prefix}.class_index 与 class_name 不一致. ")
        # Any, 联合身份第一分量; 通过后同时用于 LigandObject 图定位. 
        object_key = record.get("object_key")
        # Any, 联合身份第二分量; 必须是非空顶层 PreparedLigand.smiles. 
        prepared_smiles = record.get("prepared_smiles")
        if type(object_key) is not str or not object_key:
            raise ValueError(f"{prefix}.object_key 必须是非空字符串. ")
        if type(prepared_smiles) is not str or not prepared_smiles.strip():
            raise ValueError(f"{prefix}.prepared_smiles 必须是非空字符串. ")
        # tuple[str, str, str], 在一个 PDB 内唯一的完整监督身份键. 
        joint_key = (pdb_id, object_key, prepared_smiles)
        if joint_key in seen_keys:
            raise ValueError(f"{prefix} 与同一 PDB 的既有联合身份重复. ")
        seen_keys.add(joint_key)

        # Any, 贡献该联合身份的 occurrence candidate_id 列表; 必须为非空升序非负 int. 
        occurrence_ids = record.get("occurrence_candidate_ids")
        if (
            not isinstance(occurrence_ids, list)
            or not occurrence_ids
            or any(type(value) is not int or value < 0 for value in occurrence_ids)
            or occurrence_ids != sorted(set(occurrence_ids))
        ):
            raise ValueError(
                f"{prefix}.occurrence_candidate_ids 必须是非空、升序且不重复的非负整数列表. "
            )
        if type(record.get("graph_available")) is not bool:
            raise ValueError(f"{prefix}.graph_available 必须是布尔值. ")
        if type(record.get("graph_relative_path")) is not str:
            raise ValueError(f"{prefix}.graph_relative_path 必须是字符串. ")
        # Any, 两个语言家族到证据对象的映射; 此处只深入核验 available 布尔值. 
        language = record.get("language")
        if not isinstance(language, dict) or set(language) != set(LANGUAGE_FAMILIES):
            raise ValueError(f"{prefix}.language 必须逐项包含两个语言家族. ")
        for family in LANGUAGE_FAMILIES:
            # Any, 单一语言家族的身份证据对象; 路径等条件字段在实际读取时核验. 
            evidence = language[family]
            if not isinstance(evidence, dict) or type(evidence.get("available")) is not bool:
                raise ValueError(f"{prefix}.language.{family} 必须保存布尔 available. ")
        # Any, 两个语言家族各自的 train 可见性; 联合键集合本身不含 pdb_id, 因此该值可跨 PDB 共享. 
        seen = record.get("seen_in_train_by_language")
        if (
            not isinstance(seen, dict)
            or set(seen) != set(LANGUAGE_FAMILIES)
            or any(type(seen[family]) is not bool for family in LANGUAGE_FAMILIES)
        ):
            raise ValueError(
                f"{prefix}.seen_in_train_by_language 必须保存两个语言家族的布尔值. "
            )


def _candidate_manual_features(
    a_tokens: np.ndarray,
    p_tokens: np.ndarray,
    v_tokens: np.ndarray,
    pp_tokens: np.ndarray,
) -> np.ndarray:
    """汇总 A、P、V、PP 的 token 数和 Stage1 概率统计. 

    输入参数：
        - a_tokens: NumPy 数组, (N_A, F_A); 概率列位于倒数第四个特征分量. 
        - p_tokens: NumPy 数组, (N_P, F_P); 概率列位于倒数第四个特征分量. 
        - v_tokens: NumPy 数组, (N_V, F_V); 概率列位于倒数第四个特征分量. 
        - pp_tokens: NumPy 数组, (N_PP, F_PP); 概率列位于第 0 个特征分量. 

    返回字段：
        - features: float32 数组, (12,); 四种模态各占连续三个值. 
        - features[0]: float32 标量; A token 数, 单位为 token 个数. 
        - features[1]: float32 标量; A 的有限 Stage1 概率均值, 无量纲. 
        - features[2]: float32 标量; A 的有限 Stage1 概率最大值, 无量纲. 
        - features[3]: float32 标量; P token 数, 单位为 token 个数. 
        - features[4]: float32 标量; P 的有限 Stage1 概率均值, 无量纲. 
        - features[5]: float32 标量; P 的有限 Stage1 概率最大值, 无量纲. 
        - features[6]: float32 标量; V token 数, 单位为 token 个数. 
        - features[7]: float32 标量; V 的有限 Stage1 概率均值, 无量纲. 
        - features[8]: float32 标量; V 的有限 Stage1 概率最大值, 无量纲. 
        - features[9]: float32 标量; PP token 数, 单位为 token 个数. 
        - features[10]: float32 标量; PP 的有限 Stage1 概率均值, 无量纲. 
        - features[11]: float32 标量; PP 的有限 Stage1 概率最大值, 无量纲. 

    每个输入必须是二维数组; 非空模态的概率必须有限并位于 [0, 1], 空模态对应的三个输出均为 0. 
    """

    # list[float], 按 A/P/V/PP 顺序累积每种模态的 count/mean/max 三元组. 
    result: list[float] = []
    for name, tokens, probability_column in (
        ("A", a_tokens, -4),
        ("P", p_tokens, -4),
        ("V", v_tokens, -4),
        ("PP", pp_tokens, 0),
    ):
        if tokens.ndim != 2:
            raise ValueError(f"{name} token 必须是二维数组. ")
        # int, 当前模态第一轴 token 数; 空模态不执行均值和最大值归约. 
        count = int(tokens.shape[0])
        if count == 0:
            result.extend((0.0, 0.0, 0.0))
            continue
        # float32, (N_modality,), 当前模态固化 Stage1 概率; A/P/V 取倒数第四列, PP 取第 0 列. 
        probability = np.asarray(tokens[:, probability_column], dtype=np.float32)
        if not np.isfinite(probability).all():
            raise ValueError(f"{name} 的 Stage1 固化概率含 NaN 或 Inf. ")
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError(f"{name} 的 Stage1 固化概率必须位于 [0,1]. ")
        result.extend((float(count), float(probability.mean()), float(probability.max())))
    # float32, (12,), 四个 count/mean/max 三元组的固定顺序拼接结果. 
    features = np.asarray(result, dtype=np.float32)
    if features.shape != (CANDIDATE_MANUAL_FEATURE_DIM,):
        raise RuntimeError("候选手动特征宽度与 schema 常量不一致. ")
    return features


def _identity_target_index(  # NOTE: 又来制造重复的垃圾了。你看看你自己写的 C:\Users\15919\Desktop\Matcher\matcher\data\labels.py ，它已经定义了一个 candidate 匹配的 occurrence————那么直接记录，到时候直接读取这个occurrence的(ligand_object, smiles)自然地作为identity就好了————何必又来设计一套规则重复制造垃圾呢？ 另外，我怀疑可能发生了契约漂移，因为它的代码和这里本质是两套逻辑。
    object_keys: list[str],
    prepared_smiles_values: list[str],
    class_indices: np.ndarray,
    matched_class_index: int,
    matched_object_key: str | None,
    matched_prepared_smiles: str | None,
) -> int:
    """在 PDB 局部身份轴中按类别和联合键唯一定位候选真值. 

    输入参数：
        - object_keys: 长度 I 的字符串列表; 身份槽的 LigandObject 模板键. 
        - prepared_smiles_values: 长度 I 的非空字符串列表; 与 object_keys 共同定义联合身份. 
        - class_indices: 整数数组, (I,); 与两个字符串列表逐槽对齐的前景六分类下标. 
        - matched_class_index: 整数; 当前候选的六分类真值下标. 
        - matched_object_key: 字符串或 None; 通过覆盖阈值的 occurrence 模板键; type_tag=other 背景也可非空, 未命中时为 None. 
        - matched_prepared_smiles: 字符串或 None; 仅五类前景命中的顶层 PreparedLigand.smiles 可非空; 不可用、type_tag=other 背景或未命中时为 None. 

    中间字段：
        - matches: 整数列表; 类别、object_key 和 prepared_smiles 三者都与候选真值相同的身份槽下标. 

    返回值：
        - identity_target: 整数; 唯一匹配槽的下标; 任一联合键分量缺失或轴中没有匹配时为 -1, type_tag=other 背景因 prepared_smiles=None 也返回 -1; 多于一个匹配时抛出 ValueError. 
    """

    if matched_object_key is None or matched_prepared_smiles is None:
        return -1
    if len(object_keys) != len(prepared_smiles_values) or class_indices.shape != (
        len(object_keys),
    ):
        raise ValueError("身份模板键、PreparedLigand.smiles 与类别数量不一致. ")
    # list[int], 同时匹配类别与联合键两个分量的身份槽下标, 合法轴最多含一个. 
    matches = [
        index
        for index, (class_index, object_key, prepared_smiles) in enumerate(
            zip(class_indices, object_keys, prepared_smiles_values)
        )
        if int(class_index) == matched_class_index
        and object_key == matched_object_key
        and prepared_smiles == matched_prepared_smiles
    ]
    if len(matches) > 1:
        raise ValueError("同一 PDB 身份清单含重复联合身份. ")
    return matches[0] if matches else -1


# tuple[str, ...], _load_pdb_artifacts 从 F1_centered.npz 复制的二十个字段; offsets 与对应拼接数组必须保持共同候选切分关系. 
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
    """核对 F1-centered、PP 缓存和完整概率图的候选空间字段. 

    输入参数：
        - centered: 字符串到 NumPy 数组的映射; 当前 PDB 的 F1_centered.npz 指定字段. 
        - centered.voxel_size_world: 来源契约为 float32 (C, 3); 第 candidate_index 项是世界 X/Y/Z 单体素尺寸, 单位 Å/voxel. 
        - centered.box_origin_world: 来源契约为 float32 (C, 3); 第 candidate_index 项是 80³ 候选低角点的世界 X/Y/Z, 单位 Å. 
        - centered.box_start_zyx: 来源契约为 int32 (C, 3); 第 candidate_index 项是候选在完整概率图中的 Z/Y/X 起点. 
        - pp_input: PpCandidateInput; 当前候选 PP 缓存切片及其来源概率图空间字段. 
        - pp_input.probability_origin: float32 数组, (3,); 完整 probability_map 的世界 X/Y/Z 低角点, 单位 Å. 
        - pp_input.probability_voxel_size: float32 数组, (3,); 完整 probability_map 的世界 X/Y/Z 单体素尺寸, 单位 Å/voxel. 
        - pp_input.crop_start_zyx: int64 数组, (3,); PP 缓存绑定的 48³ 裁剪在候选 80³ 内的 Z/Y/X 起点. 
        - candidate_index: 整数; 索引 centered 的候选第一轴. 
        - crop_start_zyx: 整数数组, (3,); CandidateDataset 由当前裁剪策略重新解析的候选内 Z/Y/X 起点. 

    派生字段：
        - expected_origin: float32 数组, (3,); probability_origin 加 box_start_zyx 按 X/Y/Z 重排后乘 voxel_size 得到的候选世界原点. 
        - maximum_absolute_error: 浮点标量; box_origin_world 与 expected_origin 三轴绝对误差的最大值, 单位 Å. 

    返回值：
        - return: None; 空间字段必须为有限三元组、两处体素尺寸逐元素相等、两处裁剪起点逐元素相等, 并且候选原点绝对误差不超过 BOX_ORIGIN_ABSOLUTE_TOLERANCE_ANGSTROM. 
    """

    # float32, (3,), F1-centered 保存的当前候选世界 X/Y/Z 单体素尺寸. 
    voxel_size = np.asarray(centered["voxel_size_world"][candidate_index], dtype=np.float32)
    # float32, (3,), F1-centered 保存的当前 80³ 候选世界 X/Y/Z 低角点. 
    box_origin = np.asarray(centered["box_origin_world"][candidate_index], dtype=np.float32)
    # float32, (3,), 当前候选在完整概率图中的 Z/Y/X 整数起点; 转成 float32 仅用于原点重算. 
    box_start = np.asarray(centered["box_start_zyx"][candidate_index], dtype=np.float32)
    # float32, (3,), PP 缓存记录的完整概率图世界 X/Y/Z 低角点. 
    probability_origin = np.asarray(pp_input["probability_origin"], dtype=np.float32)
    # float32, (3,), PP 缓存记录的完整概率图 X/Y/Z 单体素尺寸. 
    probability_voxel_size = np.asarray(
        pp_input["probability_voxel_size"],
        dtype=np.float32,
    )
    # int64, (3,), PP 缓存构建时使用的候选内 48³ Z/Y/X 起点. 
    cached_crop_start = np.asarray(pp_input["crop_start_zyx"], dtype=np.int64)
    # dict[str, np.ndarray], 六个必须为有限三元组的空间字段; 字段名用于错误定位. 
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
            raise ValueError(f"{field_name} 必须是有限 (3,) 空间字段. ")
    # float32, (3,), 由完整概率图原点和候选 Z/Y/X 起点重算的世界 X/Y/Z 候选原点. 
    expected_origin = probability_origin + box_start[[2, 1, 0]] * voxel_size
    if not np.array_equal(voxel_size, probability_voxel_size):
        raise ValueError("F1-centered 与 probability_map 的 voxel_size 不一致. ")
    if not np.array_equal(cached_crop_start, np.asarray(crop_start_zyx, dtype=np.int64)):
        raise ValueError("PP 缓存与 CandidateDataset 解析的候选裁剪起点不一致. ")
    # float, 保存原点和重算原点的三轴最大绝对误差, 单位 Å; 仅用于失败诊断. 
    maximum_absolute_error = float(np.max(np.abs(box_origin - expected_origin)))
    if not np.allclose(
        box_origin,
        expected_origin,
        rtol=0.0,
        atol=BOX_ORIGIN_ABSOLUTE_TOLERANCE_ANGSTROM,
    ):
        raise ValueError(
            f"F1-centered box_origin={box_origin.tolist()} 与离散起点推导值"
            f" {expected_origin.tolist()} 不一致; 最大绝对误差为"
            f" {maximum_absolute_error:.9g} Å, 允许上限为"
            f" {BOX_ORIGIN_ABSOLUTE_TOLERANCE_ANGSTROM:.9g} Å. "
        )


def _build_a_tokens(
    centered: dict[str, np.ndarray],
    candidate_index: int,
    voxel_size: np.ndarray,
    crop_start_zyx: np.ndarray,
) -> np.ndarray:
    """构造当前候选完整 80³ A 表的 229 维 token. 

    输入参数：
        - centered: 字符串到 NumPy 数组的映射; 当前 PDB 的 F1-centered 字段. 
        - centered.A_offsets: 来源契约为 int64 (C+1,); 相邻值共同切分全部 A 拼接数组. 
        - centered.A_coord_local_xyz: 来源契约为 float32 (A_total, 3); 所属 80³ 候选内连续 X/Y/Z 体素坐标. 
        - centered.A_probability: 来源契约为 float32 (A_total,); 与 A 坐标逐项对齐的 Stage1 概率. 
        - centered.A_feat_L0: 来源契约为 float32 (A_total, 49); A 的受体基础特征. 
        - centered.A_feat_L1: 来源契约为 float16 (A_total, 64); A 的 L1 特征. 
        - centered.A_feat_L2: 来源契约为 float16 (A_total, 64); A 的 L2 特征. 
        - centered.A_feat_L3: 来源契约为 float16 (A_total, 48); A 的 L3 特征. 
        - candidate_index: 整数; 选择 A_offsets 中当前候选的半开切片. 
        - voxel_size: 数组, (3,); 当前候选世界 X/Y/Z 单体素尺寸, 单位 Å/voxel. 
        - crop_start_zyx: 整数数组, (3,); 当前 48³ 裁剪在 80³ 内的 Z/Y/X 起点, 用于移动相对坐标原点但不筛除 A. 

    返回字段：
        - a_tokens: float32 数组, (N_A, 229); 全部元素有限, N_A 为 offsets 切片长度. 
        - a_tokens[:, 0:49]: float32 数组, (N_A, 49); A_feat_L0. 
        - a_tokens[:, 49:113]: float32 数组, (N_A, 64); A_feat_L1. 
        - a_tokens[:, 113:177]: float32 数组, (N_A, 64); A_feat_L2. 
        - a_tokens[:, 177:225]: float32 数组, (N_A, 48); A_feat_L3. 
        - a_tokens[:, 225]: float32 数组, (N_A,); Stage1 固化概率原值. 
        - a_tokens[:, 226:229]: float32 数组, (N_A, 3); 相对当前 48³ 中心的世界 X/Y/Z, 再按完整 80³ 的 40 体素物理半边长逐轴归一化. 

    A token 保留完整 80³ 表; binding_area_target 必须由完整受体坐标和 binding_atom 独立构造, 不能把 A 表当作监督原子全集. 
    """

    # 两个 int, 当前候选 A 在全部 PDB 拼接数组中的半开切片边界. 
    start, stop = _offset_range(centered["A_offsets"], candidate_index)
    # float32, (N_A, 3), A 相对实际 48³ 裁剪中心的世界 X/Y/Z 位移, 单位 Å. 
    centered_coordinates = local_xyz_to_crop_centered_world(
        centered["A_coord_local_xyz"][start:stop],
        crop_start_zyx,
        voxel_size,
    )
    # float32, (3,), 完整 80³ 的 X/Y/Z 物理半边长, 即各轴 40 个体素, 单位 Å. 
    coordinate_scale = _coordinate_scale_xyz(voxel_size, FULL_CONTEXT_HALF_EXTENT_VOXELS)
    # float32, (N_A, 229), 四级特征、概率与归一化相对坐标按固定列顺序拼接. 
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
    """把完整受体表中的结合原子散射到当前候选的 48³ 裁剪. 

    输入参数：
        - receptor_coords: 数组, (N_receptor, 3); 完整受体原子的有限世界 X/Y/Z, 单位 Å. 
        - binding_atom: 数组, (N_receptor,); 与 receptor_coords 逐原子对齐的布尔监督标志. 
        - box_origin_xyz: 数组, (3,); 当前 80³ 候选低角点的世界 X/Y/Z, 单位 Å. 
        - voxel_size_xyz: 数组, (3,); 当前候选有限正 X/Y/Z 单体素尺寸, 单位 Å/voxel. 
        - crop_start_zyx: 整数数组, (3,); 48³ 裁剪在候选 80³ 内的 Z/Y/X 起点. 

    中间字段：
        - local_xyz: float32 数组, (N_receptor, 3); 受体坐标相对 80³ 候选低角点的连续 X/Y/Z 体素坐标. 
        - local_index_xyz: int64 数组, (N_receptor, 3); 向下取整后减去裁剪起点的 48³ 内 X/Y/Z 下标, 范围外值暂时保留. 
        - valid: bool 数组, (N_receptor,); 同时要求 binding_atom 为真且三个局部下标都位于 [0, 48). 
        - selected: int64 数组, (N_selected, 3); 通过 valid 选择的裁剪内 X/Y/Z 下标. 

    返回值：
        - binding_target: bool 数组, (48, 48, 48); 轴依次为 Z/Y/X, 至少一个有效结合原子落入的体素为 True, 其余为 False. 

    A token 是来源组件周围的筛选表, 不能替代这里使用的完整受体原子表. 
    """

    # float32, (N_receptor, 3), 完整受体原子的世界 X/Y/Z. 
    coordinates = np.asarray(receptor_coords, dtype=np.float32)
    # bool, (N_receptor,), 与完整受体表逐原子对齐的结合监督. 
    labels = np.asarray(binding_atom, dtype=bool)
    # float32, (3,), 当前 80³ 候选的世界 X/Y/Z 低角点. 
    origin = np.asarray(box_origin_xyz, dtype=np.float32)
    # float32, (3,), X/Y/Z 单体素尺寸, 单位 Å/voxel. 
    voxel_size = np.asarray(voxel_size_xyz, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("receptor_coords 必须为 (N,3). ")
    if labels.shape != (coordinates.shape[0],):
        raise ValueError("binding_atom 必须与 receptor_coords 第 0 维逐原子对齐. ")
    if origin.shape != (3,) or voxel_size.shape != (3,) or np.any(voxel_size <= 0):
        raise ValueError("候选 box_origin_xyz/voxel_size_xyz 必须是合法 (3,) 空间字段. ")
    if not np.isfinite(coordinates).all() or not np.isfinite(origin).all() or not np.isfinite(voxel_size).all():
        raise ValueError("binding-area 空间字段含 NaN 或 Inf. ")
    # float32, (N_receptor, 3), 相对 80³ 候选低角点的连续 X/Y/Z 体素坐标. 
    local_xyz = (coordinates - origin[None, :]) / voxel_size[None, :]
    # int64, (N_receptor, 3), 向下取整并平移到实际 48³ 裁剪后的 X/Y/Z 下标. 
    local_index_xyz = (
        np.floor(local_xyz).astype(np.int64)
        - np.asarray(crop_start_zyx, dtype=np.int64)[[2, 1, 0]]
    )
    # bool, (N_receptor,), True 表示该原子有结合标签且落在实际 48³ 三轴范围内. 
    valid = labels & np.all(
        (local_index_xyz >= 0) & (local_index_xyz < LOCAL_BOX_SIZE),
        axis=1,
    )
    # int64, (N_selected, 3), 需要散射为 True 的裁剪内 X/Y/Z 下标. 
    selected = local_index_xyz[valid]
    # bool, (48, 48, 48), Z/Y/X 顺序的结合区域目标; 重复原子落点自然取并集. 
    target = np.zeros((LOCAL_BOX_SIZE, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE), dtype=bool)
    target[selected[:, 2], selected[:, 1], selected[:, 0]] = True
    return target


def _build_p_tokens(
    centered: dict[str, np.ndarray],
    candidate_index: int,
    voxel_size: np.ndarray,
    crop_start_zyx: np.ndarray,
) -> np.ndarray:
    """构造当前候选完整 80³ P 表的 116 维 token. 

    输入参数：
        - centered: 字符串到 NumPy 数组的映射; 当前 PDB 的 F1-centered 字段. 
        - centered.P_offsets: 来源契约为 int64 (C+1,); 相邻值共同切分全部 P 拼接数组. 
        - centered.P_coord_local_xyz: 来源契约为 float32 (P_total, 3); 所属 80³ 候选内连续 X/Y/Z 体素坐标. 
        - centered.P_probability: 来源契约为 float32 (P_total,); 与 P 坐标逐项对齐的 Stage1 概率. 
        - centered.P_feat_L2: 来源契约为 float16 (P_total, 64); P 的 L2 特征. 
        - centered.P_feat_L3: 来源契约为 float16 (P_total, 48); P 的 L3 特征. 
        - candidate_index: 整数; 选择 P_offsets 中当前候选的半开切片. 
        - voxel_size: 数组, (3,); 当前候选世界 X/Y/Z 单体素尺寸, 单位 Å/voxel. 
        - crop_start_zyx: 整数数组, (3,); 当前 48³ 裁剪在 80³ 内的 Z/Y/X 起点, 用于移动相对坐标原点但不筛除 P. 

    返回字段：
        - p_tokens: float32 数组, (N_P, 116); 全部元素有限, N_P 为 offsets 切片长度. 
        - p_tokens[:, 0:64]: float32 数组, (N_P, 64); P_feat_L2. 
        - p_tokens[:, 64:112]: float32 数组, (N_P, 48); P_feat_L3. 
        - p_tokens[:, 112]: float32 数组, (N_P,); Stage1 固化概率原值. 
        - p_tokens[:, 113:116]: float32 数组, (N_P, 3); 相对当前 48³ 中心的世界 X/Y/Z, 再按完整 80³ 的 40 体素物理半边长逐轴归一化. 
    """

    # 两个 int, 当前候选 P 在全部 PDB 拼接数组中的半开切片边界. 
    start, stop = _offset_range(centered["P_offsets"], candidate_index)
    # float32, (N_P, 3), P 相对实际 48³ 裁剪中心的世界 X/Y/Z 位移, 单位 Å. 
    coordinates = local_xyz_to_crop_centered_world(
        centered["P_coord_local_xyz"][start:stop],
        crop_start_zyx,
        voxel_size,
    )
    # float32, (3,), 完整 80³ 的 X/Y/Z 物理半边长, 即各轴 40 个体素, 单位 Å. 
    coordinate_scale = _coordinate_scale_xyz(voxel_size, FULL_CONTEXT_HALF_EXTENT_VOXELS)
    # float32, (N_P, 116), L2、L3、概率与归一化相对坐标按固定列顺序拼接. 
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
    """从完整候选 V 切片构造实际 48³ 内的 52 维 V token. 

    输入参数：
        - centered: 字符串到 NumPy 数组的映射; 当前 PDB 的 F1-centered 字段. 
        - centered.voxel_final: 来源契约为 float16 (V_total, 48); 全部候选拼接的 V 主体特征. 
        - centered.centered_probability: 来源契约为 float32 (V_total,); 与 voxel_final 第一轴逐体素对齐的 Stage1 概率. 
        - start: 整数; 当前候选在两个 V 拼接数组中的半开切片起点. 
        - stop: 整数; 当前候选在两个 V 拼接数组中的半开切片终点. 
        - full_index: int64 数组, (N_V_full, 3); 与 start:stop 切片逐项对齐的 80³ 内 Z/Y/X 下标. 
        - voxel_size: 数组, (3,); 当前候选世界 X/Y/Z 单体素尺寸, 单位 Å/voxel. 
        - crop_start_zyx: 整数数组, (3,); 实际 48³ 裁剪在候选 80³ 内的 Z/Y/X 起点. 

    中间字段：
        - keep: bool 数组, (N_V_full,); True 表示完整 V 下标落在当前 48³ 裁剪内. 
        - cropped_index: int64 数组, (N_V, 3); full_index[keep] 减 crop_start_zyx 后的裁剪内 Z/Y/X. 
        - coordinates: float32 数组, (N_V, 3); 保留 V 相对裁剪中心的世界 X/Y/Z, 单位 Å. 

    返回字段：
        - v_tokens: float32 数组, (N_V, 52); 全部元素有限. 
        - v_tokens[:, 0:48]: float32 数组, (N_V, 48); voxel_final 的裁剪内切片. 
        - v_tokens[:, 48]: float32 数组, (N_V,); 裁剪内 Stage1 固化概率原值. 
        - v_tokens[:, 49:52]: float32 数组, (N_V, 3); 相对裁剪中心的世界 X/Y/Z, 再按 48³ 的 24 体素物理半边长逐轴归一化. 
        - v_index_zyx: int64 数组, (N_V, 3); 与 v_tokens 逐项对齐, 数值逐轴位于 [0, 48). 

    full_candidate_voxel_count 与监督覆盖率继续使用 N_V_full; 本函数只改变模型可见的局部 V 表. 
    """

    # bool (N_V_full,) 与 int64 (N_V, 3), 标识裁剪内 V 并返回平移后的 Z/Y/X 下标. 
    keep, cropped_index = keep_local_48_at(full_index, crop_start_zyx)
    # float32, (N_V, 3), 保留 V 相对实际 48³ 中心的世界 X/Y/Z 位移, 单位 Å. 
    coordinates = voxel_zyx_to_crop_centered_world(
        full_index[keep],
        crop_start_zyx,
        voxel_size,
    )
    # float32, (3,), 实际 48³ 的 X/Y/Z 物理半边长, 即各轴 24 个体素, 单位 Å. 
    coordinate_scale = _coordinate_scale_xyz(voxel_size, LOCAL_CONTEXT_HALF_EXTENT_VOXELS)
    # float32, (N_V, 52), V 主体、概率和归一化相对坐标按固定列顺序拼接. 
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
    """由当前 PDB 全部 occurrence 稀疏掩码构造候选 48³ 配体区域并集. 

    输入参数：
        - occurrence_masks: NumPy 数组列表; 每项预期为 int64 (V_occurrence, 3) 的完整网格 Z/Y/X 稀疏下标, 列表可为空. 
        - box_start_zyx: 数组, (3,); 当前 80³ 候选在完整网格中的 Z/Y/X 起点. 
        - local_crop_start_zyx: 数组, (3,); 当前 48³ 裁剪在候选 80³ 内的 Z/Y/X 起点. 

    中间字段：
        - crop_start: int64 数组, (3,); 当前 48³ 裁剪在完整网格中的 Z/Y/X 起点. 
        - local: int64 数组, (V_occurrence, 3); 一个 occurrence 掩码平移到裁剪坐标后的 Z/Y/X. 
        - keep: bool 数组, (V_occurrence,); 三个下标都位于 [0, 48) 的体素为 True. 
        - selected: int64 数组, (V_selected, 3); 需要写入目标的裁剪内 Z/Y/X 下标. 

    返回值：
        - ligand_target: bool 数组, (48, 48, 48); 轴依次为 Z/Y/X, 任一 occurrence 覆盖的裁剪内体素为 True; 空列表返回全 False. 

    本函数不按 matched occurrence 过滤掩码, 也不显式验证每个 mask 的二维三列形状; 部分畸形输入可能通过 NumPy 广播被接受, 因此正式的 int64 (V_occurrence, 3) 形状必须由来源生产契约保证, 不能依赖本函数失败关闭. 
    """

    # int64, (3,), 当前裁剪相对完整密度网格的绝对 Z/Y/X 起点. 
    crop_start = np.asarray(box_start_zyx, dtype=np.int64) + np.asarray(
        local_crop_start_zyx,
        dtype=np.int64,
    )
    # bool, (48, 48, 48), 按 Z/Y/X 保存全部 occurrence 在候选裁剪内的并集. 
    target = np.zeros((LOCAL_BOX_SIZE, LOCAL_BOX_SIZE, LOCAL_BOX_SIZE), dtype=bool)
    for mask in occurrence_masks:
        # int64, (V_occurrence, 3), 一个完整网格稀疏掩码平移到裁剪内的 Z/Y/X. 
        local = np.asarray(mask, dtype=np.int64) - crop_start[None, :]
        # bool, (V_occurrence,), 三个局部下标都在 48³ 内的体素选择掩码. 
        keep = np.all((local >= 0) & (local < LOCAL_BOX_SIZE), axis=1)
        # int64, (V_selected, 3), 当前 occurrence 对目标并集有贡献的 Z/Y/X 下标. 
        selected = local[keep]
        target[selected[:, 0], selected[:, 1], selected[:, 2]] = True
    return target


def _offset_range(offsets: np.ndarray, candidate_index: int) -> tuple[int, int]:  # NOTE:就是一个 return values[candidate_index], values[candidate_index+1]的东西，有必要单独写个函数嘛？？
    """从候选偏移数组读取一个指定位置的半开切片边界. 

    输入参数：
        - offsets: NumPy 数组; 转换为 int64 后按一维索引读取, 相邻值分别是候选切片起止位置. 
        - candidate_index: 整数; 候选位置, 必须满足 0 <= candidate_index 且 candidate_index+1 小于 offsets 第一维长度. 

    返回字段：
        - start: 整数; 当前候选在对应拼接数组中的包含式起点, 必须非负. 
        - stop: 整数; 当前候选在对应拼接数组中的排除式终点, 必须大于或等于 start. 

    本函数只核验被选择的两个边界, 不全局核验 offsets 单调性、首值、末值或与拼接数组长度的关系. 
    """

    # int64 数组, offsets 的转换视图或副本; 后续只读取 candidate_index 和相邻位置. 
    values = np.asarray(offsets, dtype=np.int64)
    if candidate_index < 0 or candidate_index + 1 >= values.shape[0]:
        raise IndexError(f"candidate_index={candidate_index} 不在 offsets 范围内. ")
    # int, 当前候选在对应拼接表中的包含式起点. 
    start = int(values[candidate_index])
    # int, 当前候选在对应拼接表中的排除式终点. 
    stop = int(values[candidate_index + 1])
    if start < 0 or stop < start:
        raise ValueError("候选 offsets 不是合法单调偏移. ")
    return start, stop


def _coordinate_scale_xyz(voxel_size: np.ndarray, half_extent_voxels: float) -> np.ndarray:  # NOTE:和上面同样的问题，又来制造几行函数的细小垃圾了
    """把单体素尺寸与网格半边长转换为 X/Y/Z 物理归一化尺度. 

    输入参数：
        - voxel_size: 数组, (3,); X/Y/Z 单体素尺寸, 单位 Å/voxel. 
        - half_extent_voxels: 浮点数; 每个轴从裁剪中心到边界的体素数, A/P 使用 40, V/PP 使用 24. 

    返回值：
        - scale: float32 数组, (3,); 有限正 X/Y/Z 物理半边长, 单位 Å, 用作相对世界坐标的逐轴除数. 
    """

    # float32, (3,), X/Y/Z 单体素尺寸乘无量纲半边体素数得到的物理半边长. 
    scale = np.asarray(voxel_size, dtype=np.float32) * float(half_extent_voxels)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError(f"坐标归一化尺度必须为有限正 (3,), 实际为 {scale}. ")
    return scale


def _validate_tokens(tokens: np.ndarray, expected_width: int, modality: str) -> np.ndarray:
    """把一种模态转换为 float32, 并核验二维宽度与有限值. 

    输入参数：
        - tokens: NumPy 数组; 待验证的模态 token 表, 第一轴是 token, 第二轴是特征. 
        - expected_width: 整数; 该模态固定特征宽度, A/P/V 分别为 229、116、52. 
        - modality: 字符串; 仅用于错误信息中的 A、P 或 V 名称. 

    返回值：
        - values: float32 数组, (N_token, expected_width); 全部元素有限, 可能与输入共享内存. 

    本函数不核验概率列范围或各列内部语义; 候选手动特征函数会对使用到的概率列另行检查 [0, 1]. 
    """

    # float32 数组, 把待验证 token 转为模型使用的统一 dtype. 
    values = np.asarray(tokens, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != expected_width:
        raise ValueError(
            f"{modality} token 必须为 (N,{expected_width}), 实际为 {values.shape}. "
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{modality} token 含 NaN 或 Inf. ")
    return values


def _split_cache_budget(total_bytes: int) -> dict[str, int]:
    """把单个 Dataset 实例的总字节上限分配给五类可淘汰缓存. 

    输入参数：
        - total_bytes: 整数; 单 worker 缓存总字节上限, CandidateDataset 构造器已要求非负; 本函数自身不重复检查符号. 

    返回字段：
        - budgets.artifact: 整数; PDB 级 centered、受体坐标和监督数组预算, 接收其余四项向下取整后的全部余量; 只有 total_bytes 可被 32 整除时才恰为总量 13/16. 
        - budgets.source: 整数; VoxelInputStore 来源数组预算, 等于 total_bytes//16. 
        - budgets.pp: 整数; PpInputStore 的 PDB 级 PP 缓存预算, 等于 total_bytes//16. 
        - budgets.identity: 整数; 当前 PDB 身份轴输入预算, 等于 total_bytes//32. 
        - budgets.graph: 整数; LigandGraphStore 模板图预算, 等于 total_bytes//32. 

    五个整数值之和严格等于 total_bytes; 它们只约束 ByteBudgetLru 通过 array_nbytes 递归统计到的 NumPy、映射、序列和 dataclass 载荷, 不统计进程 RSS、字符串与容器开销、清单记录或批次张量, 也不触发等量内存预分配. 
    """

    # int, LigandObject 图缓存上限, 取总预算的向下取整 1/32. 
    graph = total_bytes // 32
    # int, 已筛选身份字符串、类别、语言向量和图列表缓存上限, 取总预算的向下取整 1/32. 
    identity = total_bytes // 32
    # int, VoxelInputStore 构建 105 通道时使用的来源数组缓存上限, 取总预算的向下取整 1/16. 
    source = total_bytes // 16
    # int, PpInputStore 的 PDB 级 PP 缓存上限, 取总预算的向下取整 1/16. 
    pp = total_bytes // 16
    # int, 剩余全部字节归 PDB 级 artifact 缓存, 吸收整数除法余数以保持总和不变. 
    artifact = total_bytes - graph - identity - source - pp
    return {
        "artifact": artifact,
        "source": source,
        "pp": pp,
        "identity": identity,
        "graph": graph,
    }
