"""
扫描冻结 Stage1 F1 候选，并发布 Matcher 联合身份监督清单。

主要入口:
    - ensure_manifest: 在跨任务锁内构建、复用或扩展调用者显式请求的划分。
    - require_manifest: 只读核验既有清单，缺失划分时失败而不补建。
    - build_manifest: 首次扫描来源并发布候选、身份、排除记录、摘要和主清单。

身份边界:
    - object_key: 定位可复用 LigandObject 模板图。
    - prepared_smiles: 逐字采用交给语言模型公开接口的 PreparedLigand 顶层 smiles；通常为 canonical non-isomeric 字符串，diagnostics.used_raw_smiles=true 时为原文回退。
    - joint_identity: 同一 PDB 内由 object_key 与 prepared_smiles 的有序二元组定义；任一 occurrence 的 prepared_smiles 为 null 或来源文件缺失时不建立身份。
    - pdb_id: 身份轴只在单个 PDB 内构造；不同 PDB 中相同二元组不会直接竞争。

正式产物:
    - manifest.json: 固定 schema、来源根、监督参数、划分文件名、身份规则和汇总计数。
    - {split_name}_candidates.jsonl: 每个非空物理行对应一个 Stage1 F1 候选及其覆盖率目标。
    - {split_name}_identities.jsonl: 每个非空物理行对应当前 PDB 内一个可建立的前景联合身份。
    - excluded_pdb.jsonl: 每个非空物理行对应一个因允许原因整 PDB 排除的划分目录。

所有文件位于 output_root/manifests/closed_set_v{MANIFEST_SCHEMA_VERSION}。覆盖阈值只决定候选与 occurrence 的覆盖匹配，选择种子只决定同一联合身份采用哪个已经通过归档核验的 occurrence 语言向量。训练通常只物化 train 和 validation；calibration 必须由独立命令显式请求，评估入口不会补建。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .labels import OccurrenceMask, assign_candidate_target
from ..utils.locking import exclusive_file_lock
from .schema import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    FOREGROUND_CLASS_NAMES,
    IDENTITY_KEY_FIELDS,
    LANGUAGE_EMBEDDING_DIM,
    LANGUAGE_FAMILIES,
    LANGUAGE_MODEL_NAMES,
    MANIFEST_SCHEMA_VERSION,
    OCCURRENCE_TYPE_TO_CLASS,
    SourceRoots,
    safe_object_filename,
)


# tuple[str, str, str], 受支持且用于稳定发布顺序的数据划分；顺序为训练、验证、校准。
SPLIT_NAMES = ("train", "validation", "calibration")
# str, 联合身份类别门禁的人类可读契约；同一 PDB 内强制唯一，跨 PDB 仅诊断。
IDENTITY_CLASS_RULE = (
    "同一 PDB 内的 (object_key, prepared_smiles) 只能映射到一个局部类别；"
    "不同 PDB 不共享身份轴，跨 PDB 类别差异只作诊断。"
)
# str, 已物化候选记录的 PDB 互斥契约；排除记录不属于候选集合。
SPLIT_PDB_RULE = (
    "同一小写 pdb_id 的候选记录不得同时发布到多个划分；"
    "仅出现在 excluded_pdb.jsonl 的整 PDB 排除记录不构成候选重叠。"
)

# NOTE: 本文件中很多函数没必要（比如这个函数）。大的主函数其实只要2个就好：提供地址读取manifest; 给定相关目录(如stage1推理产物目录), 按照它们自己的布局进行读取, 建立新manifest。至于"清单还是否有效/合法性"等检验，可以放到临时脚本中，或者由AI当场检查。
def ensure_manifest(
    roots: SourceRoots,
    coverage_threshold: float,
    selection_seed: int,
    required_splits: tuple[str, ...],
) -> Path:
    """
    在跨任务文件锁内构建、复用或扩展显式请求的划分。

    输入参数:
        - roots: SourceRoots；提供冻结数据根、Stage1 根和可写输出根，manifest_root 由 schema 版本派生。
        - roots.data_root: Path；冻结 occurrence、密度、LigandObject、PreparedLigand 和语言归档的 A–G 根。
        - roots.stage1_root: Path；冻结 F1 候选和完整概率图的 Stage1 根。
        - roots.output_root: Path；派生当前 schema 清单目录的 Matcher 可写输出根。
        - coverage_threshold: float；候选命中 occurrence 的最小双向覆盖阈值，必须位于 [0, 1]。
        - selection_seed: int；同一联合身份存在多个有效 occurrence 语言归档时的确定性选择种子。
        - required_splits: tuple[str, ...]；本次必须已经物化的划分名，非空、不重复且只含 train、validation、calibration。

    返回值:
        - manifest_path: Path；经核验并包含 required_splits 的 manifest.json 路径。

    写入规则:
        - manifest.json: 已存在时先核验来源根和监督参数，只为缺失的显式划分扩展正式文件；不存在时首次请求必须包含 train，使 seen 标记只由真实训练正目标定义。
        - .build.lock: 只协调跨任务构建，不进入任何科学字段。
    """

    _validate_split_names(required_splits)
    # Path, 当前 schema 版本目录中的主清单。
    manifest_path = roots.manifest_root / "manifest.json"
    with exclusive_file_lock(roots.manifest_root / ".build.lock"):
        if manifest_path.is_file():
            manifest = _validate_existing_manifest(
                manifest_path,
                roots,
                coverage_threshold,
                selection_seed,
            )
            # tuple[str, ...], 调用者请求但既有 manifest.splits 尚未映射的划分。
            missing_splits = tuple(
                split_name
                for split_name in required_splits
                if split_name not in manifest["splits"]
            )
            if missing_splits:
                _extend_manifest(
                    roots,
                    manifest,
                    coverage_threshold,
                    selection_seed,
                    missing_splits,
                )
        else:
            if "train" not in required_splits:
                raise FileNotFoundError(
                    "首次构建清单必须显式包含 train；不得从其它划分推导 seen 身份。"
                )
            build_manifest(roots, coverage_threshold, selection_seed, required_splits)
    return manifest_path


def require_manifest(
    roots: SourceRoots,
    coverage_threshold: float,
    selection_seed: int,
    required_splits: tuple[str, ...],
) -> Path:
    """
    只读要求既有清单的关键字段与当前来源和参数相容，并声明全部显式请求划分。

    输入参数:
        - roots: SourceRoots；用于定位 manifest_root，并提供必须与 manifest.json 逐值相同的三个来源根。
        - roots.data_root: Path；必须与 manifest.data_root 的路径文本相同。
        - roots.stage1_root: Path；必须与 manifest.stage1_root 的路径文本相同。
        - roots.output_root: Path；必须与 manifest.output_root 的路径文本相同。
        - coverage_threshold: float；必须与既有主清单记录的候选覆盖阈值相同。
        - selection_seed: int；必须与既有主清单记录的语言表示选择种子相同。
        - required_splits: tuple[str, ...]；只读消费者要求的划分名，非空、不重复且属于受支持集合。

    返回值:
        - manifest_path: Path；已经存在、已通过关键字段核验且 manifest.splits 声明全部 required_splits 的主清单路径。

    本函数不创建目录、不取得构建锁、不补写 calibration 或其它缺失划分，也不核验划分 JSONL 文件是否存在或其记录内容；后续消费者打开对应文件时才会发现这类错误。
    """

    _validate_split_names(required_splits)
    manifest_path = roots.manifest_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"冻结清单不存在，禁止在只读评估入口重建：{manifest_path}")
    manifest = _validate_existing_manifest(
        manifest_path,
        roots,
        coverage_threshold,
        selection_seed,
    )
    # list[str], 只读消费者请求但既有主清单未物化的划分，按字典序报告。
    missing_splits = sorted(set(required_splits).difference(manifest["splits"]))
    if missing_splits:
        raise FileNotFoundError(
            f"冻结清单尚未物化请求划分 {missing_splits}；只读入口不得补建。"
        )
    return manifest_path


def build_manifest(
    roots: SourceRoots,
    coverage_threshold: float,
    selection_seed: int,
    split_names: tuple[str, ...],
) -> Path:
    """
    首次扫描显式划分，并原子发布候选、联合身份、排除原因、seen 标记和主清单。

    输入参数:
        - roots: SourceRoots；冻结来源根和当前 schema 的可写清单目录。
        - roots.data_root: Path；读取 occurrence、真实配体体素、LigandObject 和语言产物的冻结 A–G 根。
        - roots.stage1_root: Path；读取各划分 F1 候选和完整概率图的 Stage1 根。
        - roots.output_root: Path；派生 closed_set_v4 清单目录的可写 Matcher 输出根。
        - coverage_threshold: float；assign_candidate_target 使用的候选覆盖率与 occurrence 覆盖率共同下限，必须位于 [0, 1]。
        - selection_seed: int；每个联合身份、每个语言家族选择有效 occurrence 归档的确定性种子。
        - split_names: tuple[str, ...]；本次允许读取并物化的划分，必须包含 train，且不得重复或超出 SPLIT_NAMES。

    返回值:
        - manifest_path: Path；原子发布完成的 manifest.json 路径。

    中间集合:
        - split_candidates: dict[str, list[dict[str, Any]]]；划分名到候选记录列表的映射。
        - split_candidates.{split_name}[*].split: str；当前 Stage1 划分名。
        - split_candidates.{split_name}[*].pdb_id: str；当前小写 PDB 标识。
        - split_candidates.{split_name}[*].candidate_index: int；F1 候选轴零基索引。
        - split_candidates.{split_name}[*].source_tree_id: int；候选来源连通树编号。
        - split_candidates.{split_name}[*].source_node_id: int；候选来源树节点编号。
        - split_candidates.{split_name}[*].source_threshold_grid_index: int；候选来源阈值网格索引。
        - split_candidates.{split_name}[*].full_candidate_voxel_count: int；候选 80³ 稀疏体素数。
        - split_candidates.{split_name}[*].class_name: str；六分类名称。
        - split_candidates.{split_name}[*].class_index: int；与 class_name 对齐的六分类索引。
        - split_candidates.{split_name}[*].matched_candidate_id: int 或 null；最佳 occurrence 编号，未命中为 null。
        - split_candidates.{split_name}[*].matched_object_key: str 或 null；最佳 occurrence 模板键，未命中为 null。
        - split_candidates.{split_name}[*].matched_prepared_smiles: str 或 null；最佳五类前景 occurrence 的顶层公开接口字符串，不可用时为 null。
        - split_candidates.{split_name}[*].matched_prepared_smiles_status: str；available、explicit_null、missing_file 或 not_applicable。
        - split_candidates.{split_name}[*].candidate_coverage: float；候选方向覆盖率，范围 [0, 1]。
        - split_candidates.{split_name}[*].occurrence_coverage: float；occurrence 方向覆盖率，范围 [0, 1]。
        - split_candidates.{split_name}[*].overlap_score: float；两个覆盖率的几何平均。
        - split_candidates.{split_name}[*].language_target_available: dict[str, bool]；两个语言家族的监督可用性。
        - split_candidates.{split_name}[*].language_target_available.molformer: bool；LigandObject 文件存在且 MoLFormer 归档可用时为 true。
        - split_candidates.{split_name}[*].language_target_available.smi_ted_289m: bool；LigandObject 文件存在且 SMI-TED 归档可用时为 true。
        - split_candidates.{split_name}[*].identity_seen_in_train_by_language: dict[str, bool | null]；发布前补充的逐语言训练可见性。
        - split_candidates.{split_name}[*].identity_seen_in_train_by_language.molformer: bool 或 null；MoLFormer 口径 seen；不具备前景身份时为 null。
        - split_candidates.{split_name}[*].identity_seen_in_train_by_language.smi_ted_289m: bool 或 null；SMI-TED 口径 seen；不具备前景身份时为 null。
        - split_identities: dict[str, list[dict[str, Any]]]；划分名到 PDB 内联合身份记录列表的映射。
        - split_identities.{split_name}[*].split: str；当前 Stage1 划分名。
        - split_identities.{split_name}[*].pdb_id: str；限定本地身份轴的小写 PDB 标识。
        - split_identities.{split_name}[*].class_name: str；当前联合身份唯一的五类前景类别。
        - split_identities.{split_name}[*].class_index: int；与 class_name 对齐的六分类索引。
        - split_identities.{split_name}[*].object_key: str；联合身份模板图分量。
        - split_identities.{split_name}[*].prepared_smiles: str；联合身份顶层公开接口字符串分量。
        - split_identities.{split_name}[*].occurrence_candidate_ids: list[int]；共享联合身份的 occurrence 编号递增列表。
        - split_identities.{split_name}[*].graph_available: bool；LigandObject 文件是否存在，不代表图内容已经验证。
        - split_identities.{split_name}[*].graph_relative_path: str；相对 data_root 的 LigandObject NPZ 路径。
        - split_identities.{split_name}[*].language: dict[str, dict[str, Any]]；两个语言家族的归档选择证据。
        - split_identities.{split_name}[*].language.molformer: dict[str, Any]；MoLFormer 归档选择证据。
        - split_identities.{split_name}[*].language.smi_ted_289m: dict[str, Any]；SMI-TED 归档选择证据。
        - split_identities.{split_name}[*].language.{family}.available: bool；至少一份 occurrence 归档有效时为 true。
        - split_identities.{split_name}[*].language.{family}.selected_candidate_id: int 或 null；确定性选中的 occurrence 编号。
        - split_identities.{split_name}[*].language.{family}.selected_relative_path: str 或 null；选中归档相对 language_root 的路径。
        - split_identities.{split_name}[*].language.{family}.valid_candidate_ids: list[int]；全部有效归档 occurrence 编号。
        - split_identities.{split_name}[*].language.{family}.failed_occurrences: list[dict[str, Any]]；缺失或无效归档的诊断列表。
        - split_identities.{split_name}[*].language.{family}.failed_occurrences[*].candidate_id: int；失败归档的 occurrence 编号。
        - split_identities.{split_name}[*].language.{family}.failed_occurrences[*].reason: str；missing_file 或 invalid_archive。
        - split_identities.{split_name}[*].language.{family}.failed_occurrences[*].detail: str；仅 invalid_archive 时出现的异常文本。
        - split_identities.{split_name}[*].language.{family}.distinct_model_smiles: list[str]；有效归档 tokenizer 输入诊断字符串的排序去重列表。
        - split_identities.{split_name}[*].seen_in_train_by_language: dict[str, bool]；发布前补充的逐语言训练可见性。
        - split_identities.{split_name}[*].seen_in_train_by_language.molformer: bool；MoLFormer 口径 seen。
        - split_identities.{split_name}[*].seen_in_train_by_language.smi_ted_289m: bool；SMI-TED 口径 seen。
        - exclusions: list[dict[str, Any]]；按扫描顺序保存允许的整 PDB 排除记录。
        - exclusions[*].split: str；被排除 PDB 所在划分。
        - exclusions[*].pdb_id: str；被排除的小写 PDB 标识。
        - exclusions[*].reason: str；blob_exceed 或 missing_required_product。
        - exclusions[*].missing_paths: list[str]；仅缺必要产物时出现，由 roots 拼出的不存在路径文本。
        - split_pdb_counts: dict[str, dict[str, int]]；划分名到 PDB 目录计数的映射。
        - split_pdb_counts.{split_name}.discovered: int；发现的全部 PDB 目录数。
        - split_pdb_counts.{split_name}.included: int；成功扫描并发布候选的 PDB 数。
        - split_pdb_counts.{split_name}.excluded: int；因允许原因整 PDB 排除的目录数。
    """

    if not 0.0 <= coverage_threshold <= 1.0:
        raise ValueError("coverage_threshold 必须位于 [0,1]。")
    _validate_split_names(split_names)
    if "train" not in split_names:
        raise ValueError("首次构建清单必须包含 train。")
    roots.manifest_root.mkdir(parents=True, exist_ok=True)

    # dict[str, list[dict[str, Any]]]，每个显式划分到候选记录列表的映射；记录含 PDB/F1 索引、六分类目标、匹配联合键、覆盖率、语言可用性和 seen 字段。
    split_candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in split_names}
    # dict[str, list[dict[str, Any]]]，每个显式划分到 PDB 内联合身份记录列表的映射；记录含类别、联合键、occurrence 编号、图路径、两家语言证据和 seen 字段。
    split_identities: dict[str, list[dict[str, Any]]] = {name: [] for name in split_names}
    # list[dict[str, Any]], 因 blob_exceed 或必要产物缺失而整 PDB 排除的记录。
    exclusions: list[dict[str, Any]] = []
    # dict[str, dict[str, int]], 每个划分发现、纳入和排除的 PDB 目录计数。
    split_pdb_counts: dict[str, dict[str, int]] = {}

    for split_name in split_names:
        split_root = roots.stage1_root / split_name
        if not split_root.is_dir():
            raise FileNotFoundError(f"Stage1 划分目录不存在：{split_root}")
        # list[Path], 当前 Stage1 划分下按路径排序的全部 PDB 目录。
        pdb_directories = sorted(path for path in split_root.iterdir() if path.is_dir())
        # int, 当前划分成功扫描并进入候选/身份文件的 PDB 数。
        included_pdb = 0
        for pdb_directory in pdb_directories:
            # str，当前 Stage1 PDB 目录名规范为小写后的清单标识。
            pdb_id = pdb_directory.name.lower()
            # dict[str, Any] | None, 允许的整 PDB 排除事实；None 表示必须严格扫描。
            exclusion = _allowed_exclusion(roots, split_name, pdb_id, pdb_directory)
            if exclusion is not None:
                exclusions.append(exclusion)
                continue
            # list[dict[str, Any]]，当前 PDB 按 candidate_index 递增的 F1 候选监督记录。
            # list[dict[str, Any]]，当前 PDB 按类别和联合键稳定排序的前景身份记录。
            candidates, identities = _scan_pdb(
                roots=roots,
                split_name=split_name,
                pdb_id=pdb_id,
                coverage_threshold=coverage_threshold,
                selection_seed=selection_seed,
            )
            split_candidates[split_name].extend(candidates)
            split_identities[split_name].extend(identities)
            included_pdb += 1
        split_pdb_counts[split_name] = {
            "discovered": len(pdb_directories),
            "included": included_pdb,
            "excluded": len(pdb_directories) - included_pdb,
        }

    _validate_materialized_split_pdb_disjointness(split_candidates)
    _validate_joint_identity_classes(split_identities)
    _annotate_seen(split_candidates, split_identities)
    return _publish_manifest(
        roots=roots,
        coverage_threshold=coverage_threshold,
        selection_seed=selection_seed,
        split_candidates=split_candidates,
        split_identities=split_identities,
        exclusions=exclusions,
        split_pdb_counts=split_pdb_counts,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def _extend_manifest(
    roots: SourceRoots,
    manifest: dict[str, Any],
    coverage_threshold: float,
    selection_seed: int,
    missing_splits: tuple[str, ...],
) -> Path:
    """
    在构建锁内补建缺失划分，并重新发布所有已物化划分的一致主清单。

    输入参数:
        - roots: SourceRoots；用于读取既有清单文件、扫描新划分并原子覆盖同一 schema 目录。
        - roots.data_root: Path；读取新增划分 occurrence、配体掩码、PreparedLigand、LigandObject 和语言归档的冻结 A–G 根。
        - roots.stage1_root: Path；读取新增划分 F1 候选和完整概率图的 Stage1 根。
        - roots.output_root: Path；派生当前 schema 清单目录的 Matcher 输出根。
        - manifest: dict[str, Any]；已经通过复用门禁的主清单 JSON 对象。
        - manifest.schema_version: int；清单 schema 版本。
        - manifest.created_at_utc: str；首次建立当前 schema 清单的 UTC ISO 8601 时间，本函数发布时原样保留。
        - manifest.updated_at_utc: str；上次发布的 UTC ISO 8601 时间，本函数不把它作为本次发布时间。
        - manifest.data_root: str；冻结 A–G 数据根路径文本。
        - manifest.stage1_root: str；Stage1 F1 候选根路径文本。
        - manifest.output_root: str；Matcher 输出根路径文本。
        - manifest.coverage_threshold: float；既有候选与 occurrence 双向覆盖率下限。
        - manifest.selection_seed: int；既有身份内语言归档选择种子。
        - manifest.identity_key_fields: list[str]；联合身份字段顺序 object_key、prepared_smiles。
        - manifest.prepared_smiles_rule: str；顶层字符串、显式 null 和缺失 PreparedLigand 文件的规则说明。
        - manifest.identity_class_rule: str；同一 PDB 内联合身份类别唯一规则。
        - manifest.split_pdb_rule: str；候选 PDB 的划分互斥规则。
        - manifest.cross_pdb_joint_identity_class_variants: list[dict[str, Any]]；既有跨 PDB 联合键多类别诊断，本函数重新发布时重算。
        - manifest.cross_pdb_joint_identity_class_variants[*].object_key: str；跨 PDB 比较的模板图键。
        - manifest.cross_pdb_joint_identity_class_variants[*].prepared_smiles: str；跨 PDB 比较的顶层公开接口字符串。
        - manifest.cross_pdb_joint_identity_class_variants[*].classes: list[dict[str, Any]]；观察到的不同类别列表。
        - manifest.cross_pdb_joint_identity_class_variants[*].classes[*].class_name: str；五类前景类别名。
        - manifest.cross_pdb_joint_identity_class_variants[*].classes[*].class_index: int；与类别名对齐的六分类索引。
        - manifest.cross_pdb_joint_identity_class_variants[*].locations: list[dict[str, Any]]；联合键的划分/PDB 位置列表。
        - manifest.cross_pdb_joint_identity_class_variants[*].locations[*].split: str；身份记录所在划分。
        - manifest.cross_pdb_joint_identity_class_variants[*].locations[*].pdb_id: str；身份记录所在 PDB。
        - manifest.cross_pdb_joint_identity_class_variants[*].locations[*].class_name: str；该 PDB 内类别名。
        - manifest.cross_pdb_joint_identity_class_variants[*].locations[*].class_index: int；该 PDB 内类别索引。
        - manifest.cross_pdb_joint_identity_class_variants[*].locations[*].occurrence_count: int；共享该 PDB 内身份的 occurrence 数。
        - manifest.language_selection_rule: str；语言归档确定性选择规则说明。
        - manifest.seen_identity_rule: str；逐语言家族训练可见性规则说明。
        - manifest.class_names: list[str]；固定六分类顺序。
        - manifest.available_splits: list[str]；既有物化划分的稳定排序列表，本函数重新发布时重算。
        - manifest.deferred_splits: list[str]；既有未物化划分的稳定排序列表，本函数重新发布时重算。
        - manifest.splits: dict[str, dict[str, str]]；既有物化划分到候选和身份 JSONL 文件名的映射。
        - manifest.splits.{split_name}.candidates: str；当前既有划分候选 JSONL 的相对文件名。
        - manifest.splits.{split_name}.identities: str；当前既有划分身份 JSONL 的相对文件名。
        - manifest.excluded_pdb: str；整 PDB 排除 JSONL 的相对文件名。
        - manifest.split_pdb_counts: dict[str, dict[str, int]]；既有物化划分的目录计数。
        - manifest.split_pdb_counts.{split_name}.discovered: int；该划分发现的 PDB 目录数。
        - manifest.split_pdb_counts.{split_name}.included: int；该划分成功扫描并发布候选的 PDB 数。
        - manifest.split_pdb_counts.{split_name}.excluded: int；该划分因允许原因整 PDB 排除的目录数。
        - manifest.summary: dict[str, Any]；既有候选、类别、身份、语言和排除原因汇总，本函数重新发布时重算。
        - manifest.summary.splits: dict[str, dict[str, Any]]；既有物化划分到汇总计数的映射。
        - manifest.summary.splits.{split_name}.candidate_count: int；候选记录总数。
        - manifest.summary.splits.{split_name}.class_counts: dict[str, int]；实际出现的六分类类别计数。
        - manifest.summary.splits.{split_name}.class_counts.{class_name}: int；当前实际出现类别的候选数。
        - manifest.summary.splits.{split_name}.identity_count: int；PDB 内联合身份记录数。
        - manifest.summary.splits.{split_name}.unique_object_key_count: int；不同模板键数量。
        - manifest.summary.splits.{split_name}.same_pdb_object_key_multi_prepared_smiles_group_count: int；同 PDB、同模板键的多字符串分组数。
        - manifest.summary.splits.{split_name}.matched_foreground_without_prepared_smiles_candidate_count: int；前景命中但顶层字符串不可用的候选数。
        - manifest.summary.splits.{split_name}.matched_foreground_explicit_null_candidate_count: int；前景命中且顶层字符串显式为 null 的候选数。
        - manifest.summary.splits.{split_name}.matched_foreground_missing_prepared_file_candidate_count: int；前景命中但 PreparedLigand 文件缺失的候选数。
        - manifest.summary.splits.{split_name}.language_available_identity_count: dict[str, int]；逐语言家族可用身份数。
        - manifest.summary.splits.{split_name}.language_available_identity_count.molformer: int；MoLFormer 可用身份数。
        - manifest.summary.splits.{split_name}.language_available_identity_count.smi_ted_289m: int；SMI-TED 可用身份数。
        - manifest.summary.splits.{split_name}.language_multi_model_smiles_identity_count: dict[str, int]；逐语言家族多 tokenizer 字符串身份数。
        - manifest.summary.splits.{split_name}.language_multi_model_smiles_identity_count.molformer: int；MoLFormer 多字符串身份数。
        - manifest.summary.splits.{split_name}.language_multi_model_smiles_identity_count.smi_ted_289m: int；SMI-TED 多字符串身份数。
        - manifest.summary.splits.{split_name}.seen_positive_identity_count: dict[str, int]；逐语言家族训练可见身份数。
        - manifest.summary.splits.{split_name}.seen_positive_identity_count.molformer: int；MoLFormer 口径训练可见身份数。
        - manifest.summary.splits.{split_name}.seen_positive_identity_count.smi_ted_289m: int；SMI-TED 口径训练可见身份数。
        - manifest.summary.excluded_pdb_reason_counts: dict[str, int]；实际出现的整 PDB 排除原因计数。
        - manifest.summary.excluded_pdb_reason_counts.blob_exceed: int；条件字段，blob_exceed PDB 数。
        - manifest.summary.excluded_pdb_reason_counts.missing_required_product: int；条件字段，缺必要产物 PDB 数。
        - coverage_threshold: float；与 manifest.coverage_threshold 相同的候选覆盖阈值。
        - selection_seed: int；与 manifest.selection_seed 相同的语言表示选择种子。
        - missing_splits: tuple[str, ...]；调用者显式请求且 manifest.splits 尚未包含的划分名。

    返回值:
        - manifest_path: Path；保留 created_at_utc、更新物化划分和摘要后的 manifest.json 路径。

    既有 {split}_candidates.jsonl 和 {split}_identities.jsonl 先完整读回内存，新增划分扫描完成后统一重算跨划分门禁和 seen 标记；既有候选与身份的来源字段不重新扫描。
    """

    # dict[str, list[dict[str, Any]]]，从 manifest.splits.{split}.candidates 逐划分读回的既有候选记录；每条保留 PDB/F1 索引、监督、联合键、语言和 seen 字段。
    split_candidates = {
        split_name: load_jsonl(roots.manifest_root / files["candidates"])
        for split_name, files in manifest["splits"].items()
    }
    # dict[str, list[dict[str, Any]]]，从 manifest.splits.{split}.identities 逐划分读回的既有联合身份记录；每条保留类别、联合键、occurrence、图、语言和 seen 字段。
    split_identities = {
        split_name: load_jsonl(roots.manifest_root / files["identities"])
        for split_name, files in manifest["splits"].items()
    }
    # list[dict[str, Any]]，既有 excluded_pdb.jsonl 的全部整 PDB 排除记录；missing_paths 只在缺产物原因下出现。
    exclusions = load_jsonl(roots.manifest_root / manifest["excluded_pdb"])
    # dict[str, dict[str, int]]，既有划分到 discovered、included、excluded 三项 PDB 目录计数的浅拷贝。
    split_pdb_counts = dict(manifest["split_pdb_counts"])
    for split_name in missing_splits:
        split_candidates[split_name] = []
        split_identities[split_name] = []
        split_root = roots.stage1_root / split_name
        if not split_root.is_dir():
            raise FileNotFoundError(f"Stage1 划分目录不存在：{split_root}")
        # list[Path], 当前新增划分下按路径排序的全部 PDB 目录。
        pdb_directories = sorted(path for path in split_root.iterdir() if path.is_dir())
        # int，当前新增划分中成功扫描并进入候选/身份文件的 PDB 数。
        included_pdb = 0
        for pdb_directory in pdb_directories:
            # str，当前新增划分的 Stage1 PDB 目录名规范为小写后的清单标识。
            pdb_id = pdb_directory.name.lower()
            # dict[str, Any] | None，允许的整 PDB 排除记录；None 表示必须严格扫描来源。
            exclusion = _allowed_exclusion(roots, split_name, pdb_id, pdb_directory)
            if exclusion is not None:
                exclusions.append(exclusion)
                continue
            # list[dict[str, Any]]，新增 PDB 按 candidate_index 递增的 F1 候选监督记录。
            # list[dict[str, Any]]，新增 PDB 按类别和联合键稳定排序的前景身份记录。
            candidates, identities = _scan_pdb(
                roots=roots,
                split_name=split_name,
                pdb_id=pdb_id,
                coverage_threshold=coverage_threshold,
                selection_seed=selection_seed,
            )
            split_candidates[split_name].extend(candidates)
            split_identities[split_name].extend(identities)
            included_pdb += 1
        split_pdb_counts[split_name] = {
            "discovered": len(pdb_directories),
            "included": included_pdb,
            "excluded": len(pdb_directories) - included_pdb,
        }
    _validate_materialized_split_pdb_disjointness(split_candidates)
    _validate_joint_identity_classes(split_identities)
    _annotate_seen(split_candidates, split_identities)
    return _publish_manifest(
        roots=roots,
        coverage_threshold=coverage_threshold,
        selection_seed=selection_seed,
        split_candidates=split_candidates,
        split_identities=split_identities,
        exclusions=exclusions,
        split_pdb_counts=split_pdb_counts,
        created_at_utc=str(manifest["created_at_utc"]),
    )


def _annotate_seen(
    split_candidates: dict[str, list[dict[str, Any]]],
    split_identities: dict[str, list[dict[str, Any]]],
) -> None:
    """
    按语言家族与联合身份标记是否在 train 候选中作为可监督前景正目标出现。

    输入参数:
        - split_candidates: dict[str, list[dict[str, Any]]]；至少包含 train 的候选记录，键为已物化划分名。
        - split_candidates.{split_name}[*].class_name: str；六分类名称，只有五类前景进入训练正目标集合。
        - split_candidates.{split_name}[*].matched_object_key: str 或 null；最佳命中 occurrence 的模板键，未命中时为 null。
        - split_candidates.{split_name}[*].matched_prepared_smiles: str 或 null；最佳前景命中的顶层公开接口字符串，不可用或不适用时为 null。
        - split_candidates.{split_name}[*].language_target_available: dict[str, bool]；当前候选逐语言家族是否具备完整语言目标。
        - split_candidates.{split_name}[*].language_target_available.molformer: bool；MoLFormer 语言目标可用性。
        - split_candidates.{split_name}[*].language_target_available.smi_ted_289m: bool；SMI-TED 语言目标可用性。
        - split_candidates.{split_name}[*].identity_seen_in_train_by_language: dict[str, bool | null]；本函数原位新增或覆盖的逐语言训练可见性对象。
        - split_candidates.{split_name}[*].identity_seen_in_train_by_language.molformer: bool 或 null；MoLFormer 口径 seen，缺前景联合身份时为 null。
        - split_candidates.{split_name}[*].identity_seen_in_train_by_language.smi_ted_289m: bool 或 null；SMI-TED 口径 seen，缺前景联合身份时为 null。
        - split_identities: dict[str, list[dict[str, Any]]]；与候选划分键一致的身份记录。
        - split_identities.{split_name}[*].object_key: str；联合身份模板图分量。
        - split_identities.{split_name}[*].prepared_smiles: str；联合身份顶层公开接口字符串分量。
        - split_identities.{split_name}[*].seen_in_train_by_language: dict[str, bool]；本函数原位新增或覆盖的逐语言训练可见性对象。
        - split_identities.{split_name}[*].seen_in_train_by_language.molformer: bool；MoLFormer 口径 seen。
        - split_identities.{split_name}[*].seen_in_train_by_language.smi_ted_289m: bool；SMI-TED 口径 seen。

    新增身份字段:
        - seen_in_train_by_language: dict[str, bool]；键为 molformer 与 smi_ted_289m，值表示该联合身份是否在 train 候选中成为相应语言家族可监督的前景正目标。
        - seen_in_train_by_language.molformer: bool；MoLFormer 监督口径下的训练可见性。
        - seen_in_train_by_language.smi_ted_289m: bool；SMI-TED 监督口径下的训练可见性。

    新增候选字段:
        - identity_seen_in_train_by_language: dict[str, bool | null]；背景候选、未匹配候选或缺少 prepared_smiles 的前景候选写 null，其余按联合身份查询训练可见性。
        - identity_seen_in_train_by_language.molformer: bool 或 null；MoLFormer 监督口径下的可见性。
        - identity_seen_in_train_by_language.smi_ted_289m: bool 或 null；SMI-TED 监督口径下的可见性。

    本函数原位补充记录，不返回新对象；不同 PDB 的相同联合键在 seen 判断中共享训练可见事实，但不共享同一次 Matcher 身份轴。
    """

    # dict[str, set[tuple[str, str]]], 每个语言家族在 train 候选中实际可监督的前景联合键。
    train_positive_keys = {
        family: {
            (candidate["matched_object_key"], candidate["matched_prepared_smiles"])
            for candidate in split_candidates["train"]
            if candidate["class_name"] in FOREGROUND_CLASS_NAMES
            and candidate["matched_object_key"] is not None
            and candidate["matched_prepared_smiles"] is not None
            and candidate["language_target_available"][family]
        }
        for family in LANGUAGE_FAMILIES
    }
    for split_name in split_candidates:
        for identity in split_identities[split_name]:
            # tuple[str, str], 当前身份的 object_key 与非空 prepared_smiles。
            identity_key = (identity["object_key"], identity["prepared_smiles"])
            identity["seen_in_train_by_language"] = {
                family: identity_key in train_positive_keys[family]
                for family in LANGUAGE_FAMILIES
            }
        for candidate in split_candidates[split_name]:
            # str | None, 当前候选最佳匹配 occurrence 的模板键；未命中时为 None，命中 other 时即使类别为 background 也可非空。
            object_key = candidate["matched_object_key"]
            # str | None, 当前候选匹配 occurrence 的顶层 PreparedLigand.smiles。
            prepared_smiles = candidate["matched_prepared_smiles"]
            # tuple[str | None, str | None], 仅两个分量均非空且为前景时才用于 seen 集合查询。
            identity_key = (object_key, prepared_smiles)
            candidate["identity_seen_in_train_by_language"] = {
                family: (
                    identity_key in train_positive_keys[family]
                    if object_key is not None
                    and prepared_smiles is not None
                    and candidate["class_name"] in FOREGROUND_CLASS_NAMES
                    else None
                )
                for family in LANGUAGE_FAMILIES
            }

# NOTE: 同样看法——————主代码链路不负责写 _valid* 这些检查函数（这是个广泛问题，不局限于这一个函数，比如还包括下面一个函数）
def _validate_materialized_split_pdb_disjointness(
    split_candidates: dict[str, list[dict[str, Any]]],
) -> None:
    """
    拒绝同一小写 PDB 标识的候选记录同时进入多个准备发布的划分。

    输入参数:
        - split_candidates: dict[str, list[dict[str, Any]]]；划分名到候选记录列表的映射。
        - split_candidates.{split_name}[*].pdb_id: str；非空 PDB 标识，转为小写后用于判断不同划分是否发布了同一 PDB。

    中间字段:
        - splits_by_pdb: dict[str, set[str]]；小写 pdb_id 到含候选记录的划分名集合。
        - overlaps: dict[str, list[str]]；只保留出现在两个或更多划分的 PDB，并把划分名排序。

    仅存在于 excluded_pdb.jsonl 而没有候选记录的 PDB 不参与此门禁。
    """

    # defaultdict[str, set[str]], 汇总每个小写 PDB 实际发布候选的划分集合。
    splits_by_pdb: dict[str, set[str]] = defaultdict(set)
    for split_name, candidates in split_candidates.items():
        for record_index, candidate in enumerate(candidates):
            pdb_id = candidate.get("pdb_id")
            if type(pdb_id) is not str or not pdb_id:
                raise ValueError(
                    f"{split_name}_candidates.jsonl[{record_index}].pdb_id 必须是非空字符串。"
                )
            splits_by_pdb[pdb_id.lower()].add(split_name)
    # dict[str, list[str]], 跨划分候选重叠；键和每个划分列表均稳定排序以便追错。
    overlaps = {
        pdb_id: sorted(split_names_for_pdb)
        for pdb_id, split_names_for_pdb in sorted(splits_by_pdb.items())
        if len(split_names_for_pdb) > 1
    }
    if overlaps:
        raise ValueError(f"同一 PDB 的候选记录不得同时发布到多个划分：{overlaps}")


def _validate_joint_identity_classes(
    split_identities: dict[str, list[dict[str, Any]]],
) -> None:
    """
    要求每个 PDB 内的同一联合身份只映射到一个六分类类别。

    输入参数:
        - split_identities: dict[str, list[dict[str, Any]]]；本次准备发布的全部划分身份记录。
        - split_identities.{split_name}[*].pdb_id: str；非空字符串，限定身份轴所在 PDB。
        - split_identities.{split_name}[*].object_key: str；非空联合身份模板图分量。
        - split_identities.{split_name}[*].prepared_smiles: str；非空联合身份语言公开接口分量。
        - split_identities.{split_name}[*].class_name: str；五类非背景类别之一。
        - split_identities.{split_name}[*].class_index: int；必须等于 CLASS_TO_INDEX[class_name]。

    函数按 pdb_id、object_key、prepared_smiles 三元组聚合；同一键出现多个 class_name、class_index 二元组时，在写任何正式清单前失败。不同 PDB 不共享身份轴，跨 PDB 的类别差异由 _joint_identity_cross_pdb_class_variants 记录而不阻断清单。
    """

    # defaultdict[tuple[str,str,str], set[tuple[str,int]]], 每个 PDB 内联合身份观察到的类别集合。
    classes_by_identity: dict[tuple[str, str, str], set[tuple[str, int]]] = defaultdict(set)
    # defaultdict[tuple[str,str,str], list[str]], 每条重复身份记录的划分/PDB 位置，供错误报告。
    locations_by_identity: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for split_name, identities in split_identities.items():
        for record_index, identity in enumerate(identities):
            prefix = f"{split_name}_identities.jsonl[{record_index}]"
            pdb_id = identity.get("pdb_id")
            object_key = identity.get("object_key")
            prepared_smiles = identity.get("prepared_smiles")
            class_name = identity.get("class_name")
            class_index = identity.get("class_index")
            if type(pdb_id) is not str or not pdb_id:
                raise ValueError(f"{prefix}.pdb_id 必须是非空字符串。")
            if type(object_key) is not str or not object_key:
                raise ValueError(f"{prefix}.object_key 必须是非空字符串。")
            if type(prepared_smiles) is not str or not prepared_smiles.strip():
                raise ValueError(f"{prefix}.prepared_smiles 必须是非空字符串。")
            if class_name not in FOREGROUND_CLASS_NAMES:
                raise ValueError(f"{prefix}.class_name 必须属于五类非背景类别。")
            if type(class_index) is not int or class_index != CLASS_TO_INDEX[class_name]:
                raise ValueError(f"{prefix}.class_index 与 class_name 不一致。")
            # tuple[str, str, str], PDB 内联合身份键；不跨 PDB 合并身份轴。
            key = (pdb_id, object_key, prepared_smiles)
            classes_by_identity[key].add((class_name, class_index))
            locations_by_identity[key].append(f"{split_name}/{pdb_id}")
    # dict[tuple[str,str,str], dict[str,Any]], 映射到多个类别的 PDB 内联合身份及其位置。
    conflicts = {
        key: {
            "classes": sorted(classes),
            "locations": sorted(locations_by_identity[key]),
        }
        for key, classes in classes_by_identity.items()
        if len(classes) > 1
    }
    if conflicts:
        raise ValueError(f"同一 PDB 内的联合身份类别不一致：{conflicts}")


# NOTE: 这种不通用的诊断脚本可以边缘化放置（比如放到 ops 甚至 tmp 目录下），不要污染科学主线！
def _joint_identity_cross_pdb_class_variants(
    split_identities: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    列出同一联合键在不同 PDB 中采用多个局部六分类类别的诊断记录。

    输入参数:
        - split_identities: dict[str, list[dict[str, Any]]]；已经通过 PDB 内类别唯一门禁的全部划分身份记录。
        - split_identities.{split_name}[*].object_key: str；跨 PDB 诊断联合键的模板图分量。
        - split_identities.{split_name}[*].prepared_smiles: str；跨 PDB 诊断联合键的顶层公开接口字符串分量。
        - split_identities.{split_name}[*].class_name: str；当前 PDB 内身份的五类前景类别名。
        - split_identities.{split_name}[*].class_index: int；与 class_name 对齐的六分类索引。
        - split_identities.{split_name}[*].pdb_id: str；身份所在 PDB 标识，用于要求诊断至少跨两个 PDB。
        - split_identities.{split_name}[*].occurrence_candidate_ids: list[int]；共享该 PDB 内身份的 occurrence 编号，用于计算 occurrence_count。

    返回记录字段:
        - object_key: str；跨 PDB 比较的模板图键。
        - prepared_smiles: str；跨 PDB 比较的顶层 PreparedLigand.smiles。
        - classes: list[dict[str, Any]]；该联合键跨 PDB 观察到的不同类别，按 class_name、class_index 排序。
        - classes[*].class_name: str；观察到的五类非背景类别名。
        - classes[*].class_index: int；与 class_name 对齐的六分类索引。
        - locations: list[dict[str, Any]]；该联合键的每个划分/PDB 位置，按 split、pdb_id 排序。
        - locations[*].split: str；身份记录所在划分。
        - locations[*].pdb_id: str；身份记录所在 PDB。
        - locations[*].class_name: str；该 PDB 内联合身份的类别名。
        - locations[*].class_index: int；该 PDB 内联合身份的类别索引。
        - locations[*].occurrence_count: int；共享该 PDB 内联合身份的 occurrence 数。

    只有类别数大于一且 PDB 数大于一的联合键进入返回列表；该结果是主清单诊断字段，不改变训练身份轴。
    """

    # defaultdict[tuple[str,str], set[tuple[str,int]]], 跨 PDB 联合键观察到的类别集合。
    classes_by_identity: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    # defaultdict[tuple[str,str], set[str]], 跨 PDB 联合键出现的 PDB 集合。
    pdb_ids_by_identity: dict[tuple[str, str], set[str]] = defaultdict(set)
    # defaultdict[tuple[str,str], list[dict[str,Any]]], 每个联合键的划分/PDB/类别位置记录。
    locations_by_identity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for split_name, identities in split_identities.items():
        for identity in identities:
            # tuple[str, str], 不含 PDB 的联合键，仅用于生成跨 PDB 诊断。
            key = (identity["object_key"], identity["prepared_smiles"])
            classes_by_identity[key].add((identity["class_name"], identity["class_index"]))
            pdb_ids_by_identity[key].add(identity["pdb_id"])
            locations_by_identity[key].append(
                {
                    "split": split_name,
                    "pdb_id": identity["pdb_id"],
                    "class_name": identity["class_name"],
                    "class_index": identity["class_index"],
                    "occurrence_count": len(identity["occurrence_candidate_ids"]),
                }
            )
    return [
        {
            "object_key": key[0],
            "prepared_smiles": key[1],
            "classes": [
                {"class_name": class_name, "class_index": class_index}
                for class_name, class_index in sorted(classes_by_identity[key])
            ],
            "locations": sorted(
                locations_by_identity[key],
                key=lambda item: (item["split"], item["pdb_id"]),
            ),
        }
        for key in sorted(classes_by_identity)
        if len(classes_by_identity[key]) > 1 and len(pdb_ids_by_identity[key]) > 1
    ]


def _publish_manifest(
    roots: SourceRoots,
    coverage_threshold: float,
    selection_seed: int,
    split_candidates: dict[str, list[dict[str, Any]]],
    split_identities: dict[str, list[dict[str, Any]]],
    exclusions: list[dict[str, Any]],
    split_pdb_counts: dict[str, dict[str, int]],
    created_at_utc: str,
) -> Path:
    """
    原子发布当前已物化划分的 JSONL、汇总和主清单。

    输入参数:
        - roots: SourceRoots；提供当前 schema 版本的 manifest_root 和三个来源根字符串。
        - roots.data_root: Path；写入 manifest.data_root 的冻结 A–G 根。
        - roots.stage1_root: Path；写入 manifest.stage1_root 的 Stage1 根。
        - roots.output_root: Path；写入 manifest.output_root 并派生正式清单目录的 Matcher 输出根。
        - coverage_threshold: float；候选与 occurrence 双向覆盖的正式阈值。
        - selection_seed: int；联合身份语言表示的确定性选择种子。
        - split_candidates: dict[str, list[dict[str, Any]]]；已完成 seen 标注的候选记录，键为已物化划分。
        - split_candidates.{split_name}[*].split: str；当前 Stage1 划分名。
        - split_candidates.{split_name}[*].pdb_id: str；当前小写 PDB 标识。
        - split_candidates.{split_name}[*].candidate_index: int；F1 候选轴零基索引。
        - split_candidates.{split_name}[*].source_tree_id: int；候选来源连通树编号。
        - split_candidates.{split_name}[*].source_node_id: int；候选来源树节点编号。
        - split_candidates.{split_name}[*].source_threshold_grid_index: int；候选来源阈值网格索引。
        - split_candidates.{split_name}[*].full_candidate_voxel_count: int；完整 80³ 候选的稀疏体素数。
        - split_candidates.{split_name}[*].class_name: str；六分类名称。
        - split_candidates.{split_name}[*].class_index: int；与 class_name 对齐的六分类索引。
        - split_candidates.{split_name}[*].matched_candidate_id: int 或 null；最佳 occurrence 编号，未命中时为 null。
        - split_candidates.{split_name}[*].matched_object_key: str 或 null；最佳 occurrence 模板键，未命中时为 null。
        - split_candidates.{split_name}[*].matched_prepared_smiles: str 或 null；最佳前景命中的顶层公开接口字符串，不可用或不适用时为 null。
        - split_candidates.{split_name}[*].matched_prepared_smiles_status: str；available、explicit_null、missing_file 或 not_applicable。
        - split_candidates.{split_name}[*].candidate_coverage: float；候选方向覆盖率，范围 [0, 1]。
        - split_candidates.{split_name}[*].occurrence_coverage: float；occurrence 方向覆盖率，范围 [0, 1]。
        - split_candidates.{split_name}[*].overlap_score: float；两个覆盖率乘积的平方根。
        - split_candidates.{split_name}[*].language_target_available: dict[str, bool]；逐语言家族的监督可用性。
        - split_candidates.{split_name}[*].language_target_available.molformer: bool；MoLFormer 语言目标可用性。
        - split_candidates.{split_name}[*].language_target_available.smi_ted_289m: bool；SMI-TED 语言目标可用性。
        - split_candidates.{split_name}[*].identity_seen_in_train_by_language: dict[str, bool | null]；逐语言家族的训练可见性。
        - split_candidates.{split_name}[*].identity_seen_in_train_by_language.molformer: bool 或 null；MoLFormer 口径 seen，缺前景联合身份时为 null。
        - split_candidates.{split_name}[*].identity_seen_in_train_by_language.smi_ted_289m: bool 或 null；SMI-TED 口径 seen，缺前景联合身份时为 null。
        - split_identities: dict[str, list[dict[str, Any]]]；已完成 seen 标注的联合身份记录，键与 split_candidates 一致。
        - split_identities.{split_name}[*].split: str；当前 Stage1 划分名。
        - split_identities.{split_name}[*].pdb_id: str；限定本地身份轴的小写 PDB 标识。
        - split_identities.{split_name}[*].class_name: str；该联合身份唯一的五类前景类别。
        - split_identities.{split_name}[*].class_index: int；与 class_name 对齐的六分类索引。
        - split_identities.{split_name}[*].object_key: str；联合身份模板图分量。
        - split_identities.{split_name}[*].prepared_smiles: str；联合身份顶层公开接口字符串分量。
        - split_identities.{split_name}[*].occurrence_candidate_ids: list[int]；共享身份的 occurrence 编号递增列表。
        - split_identities.{split_name}[*].graph_available: bool；LigandObject 文件是否存在，不代表图内容已核验。
        - split_identities.{split_name}[*].graph_relative_path: str；相对 data_root 的 LigandObject NPZ 路径。
        - split_identities.{split_name}[*].language: dict[str, dict[str, Any]]；molformer 与 smi_ted_289m 的归档选择证据。
        - split_identities.{split_name}[*].language.molformer: dict[str, Any]；MoLFormer 归档选择证据。
        - split_identities.{split_name}[*].language.smi_ted_289m: dict[str, Any]；SMI-TED 归档选择证据。
        - split_identities.{split_name}[*].language.{family}.available: bool；至少一个 occurrence 归档有效时为 true。
        - split_identities.{split_name}[*].language.{family}.selected_candidate_id: int 或 null；确定性选中的 occurrence 编号。
        - split_identities.{split_name}[*].language.{family}.selected_relative_path: str 或 null；选中归档相对 language_root 的路径。
        - split_identities.{split_name}[*].language.{family}.valid_candidate_ids: list[int]；全部有效归档的 occurrence 编号。
        - split_identities.{split_name}[*].language.{family}.failed_occurrences: list[dict[str, Any]]；缺失或无效归档诊断。
        - split_identities.{split_name}[*].language.{family}.failed_occurrences[*].candidate_id: int；失败归档的 occurrence 编号。
        - split_identities.{split_name}[*].language.{family}.failed_occurrences[*].reason: str；missing_file 或 invalid_archive。
        - split_identities.{split_name}[*].language.{family}.failed_occurrences[*].detail: str；仅 invalid_archive 时出现的错误文本。
        - split_identities.{split_name}[*].language.{family}.distinct_model_smiles: list[str]；有效归档 tokenizer 输入字符串的排序去重列表。
        - split_identities.{split_name}[*].seen_in_train_by_language: dict[str, bool]；逐语言家族训练可见性。
        - split_identities.{split_name}[*].seen_in_train_by_language.molformer: bool；MoLFormer 口径 seen。
        - split_identities.{split_name}[*].seen_in_train_by_language.smi_ted_289m: bool；SMI-TED 口径 seen。
        - exclusions: list[dict[str, Any]]；所有已扫描划分的整 PDB 排除记录。
        - exclusions[*].split: str；被排除 PDB 所在划分。
        - exclusions[*].pdb_id: str；被排除的小写 PDB 标识。
        - exclusions[*].reason: str；blob_exceed 或 missing_required_product。
        - exclusions[*].missing_paths: list[str]；仅缺必要产物时出现，由来源根拼出的不存在路径文本。
        - split_pdb_counts: dict[str, dict[str, int]]；每个已物化划分的 PDB 目录计数。
        - split_pdb_counts.{split_name}.discovered: int；发现的全部 PDB 目录数。
        - split_pdb_counts.{split_name}.included: int；成功扫描并发布候选的 PDB 数。
        - split_pdb_counts.{split_name}.excluded: int；因允许原因整 PDB 排除的目录数。
        - created_at_utc: str；首次建立该 schema 清单的 UTC 时间；扩展划分时原样保留。

    manifest.json 顶层字段:
        - schema_version: int；MANIFEST_SCHEMA_VERSION，决定正式目录和读取契约。
        - created_at_utc: str；首次建立清单的 UTC ISO 8601 时间。
        - updated_at_utc: str；本次发布的 UTC ISO 8601 时间。
        - data_root: str；A–G 冻结数据根的绝对或调用者提供路径文本。
        - stage1_root: str；Stage1 F1 候选根的路径文本。
        - output_root: str；Matcher 输出根的路径文本。
        - coverage_threshold: float；assign_candidate_target 使用的正式双向覆盖下限。
        - selection_seed: int；语言表示选择种子。
        - identity_key_fields: list[str]；按顺序固定为 object_key、prepared_smiles。
        - prepared_smiles_rule: str；说明顶层 smiles 的逐字身份语义、null 和缺失文件处理。
        - identity_class_rule: str；与 IDENTITY_CLASS_RULE 相同的 PDB 内类别唯一规则。
        - split_pdb_rule: str；与 SPLIT_PDB_RULE 相同的候选 PDB 划分互斥规则。
        - cross_pdb_joint_identity_class_variants: list[dict[str, Any]]；跨 PDB 同联合键出现多个局部类别时的条件诊断列表。
        - cross_pdb_joint_identity_class_variants[*].object_key: str；跨 PDB 比较的模板图键。
        - cross_pdb_joint_identity_class_variants[*].prepared_smiles: str；跨 PDB 比较的顶层公开接口字符串。
        - cross_pdb_joint_identity_class_variants[*].classes: list[dict[str, Any]]；观察到的不同类别列表。
        - cross_pdb_joint_identity_class_variants[*].classes[*].class_name: str；五类前景类别名。
        - cross_pdb_joint_identity_class_variants[*].classes[*].class_index: int；与 class_name 对齐的六分类索引。
        - cross_pdb_joint_identity_class_variants[*].locations: list[dict[str, Any]]；该联合键的划分/PDB 位置列表。
        - cross_pdb_joint_identity_class_variants[*].locations[*].split: str；身份记录所在划分。
        - cross_pdb_joint_identity_class_variants[*].locations[*].pdb_id: str；身份记录所在 PDB。
        - cross_pdb_joint_identity_class_variants[*].locations[*].class_name: str；该 PDB 内的类别名。
        - cross_pdb_joint_identity_class_variants[*].locations[*].class_index: int；该 PDB 内的类别索引。
        - cross_pdb_joint_identity_class_variants[*].locations[*].occurrence_count: int；共享该 PDB 内联合身份的 occurrence 数。
        - language_selection_rule: str；说明有效 occurrence 排序、v3 兼容盐值和循环选择方式。
        - seen_identity_rule: str；说明每个语言家族按 train 可监督正目标判断 seen。
        - class_names: list[str]；六分类顺序，与 CLASS_NAMES 逐位置一致。
        - available_splits: list[str]；已物化划分，按 SPLIT_NAMES 的稳定顺序排列。
        - deferred_splits: list[str]；尚未物化划分，按 SPLIT_NAMES 的稳定顺序排列。
        - splits: dict[str, dict[str, str]]；每个已物化划分的候选和身份 JSONL 文件名映射。
        - splits.{split_name}.candidates: str；当前划分候选 JSONL 的相对文件名。
        - splits.{split_name}.identities: str；当前划分身份 JSONL 的相对文件名。
        - excluded_pdb: str；固定为 excluded_pdb.jsonl。
        - split_pdb_counts: dict[str, dict[str, int]]；每个已物化划分的目录扫描计数。
        - split_pdb_counts.{split_name}.discovered: int；Stage1 划分下发现的全部 PDB 目录数。
        - split_pdb_counts.{split_name}.included: int；成功扫描并进入候选/身份产物的 PDB 数。
        - split_pdb_counts.{split_name}.excluded: int；因允许原因整 PDB 排除的目录数。
        - summary: dict[str, Any]；类别、身份、语言监督和排除计数。
        - summary.splits: dict[str, dict[str, Any]]；已物化划分到汇总计数的映射。
        - summary.splits.{split_name}.candidate_count: int；当前划分候选记录总数。
        - summary.splits.{split_name}.class_counts: dict[str, int]；实际出现的候选类别计数，未出现类别省略。
        - summary.splits.{split_name}.class_counts.{class_name}: int；当前六分类类别的候选数。
        - summary.splits.{split_name}.identity_count: int；当前划分联合身份记录数。
        - summary.splits.{split_name}.unique_object_key_count: int；身份记录中不同 object_key 数。
        - summary.splits.{split_name}.same_pdb_object_key_multi_prepared_smiles_group_count: int；同一 PDB、同一 object_key 拆成多个顶层字符串身份的分组数。
        - summary.splits.{split_name}.matched_foreground_without_prepared_smiles_candidate_count: int；前景命中但顶层字符串不可用的候选数。
        - summary.splits.{split_name}.matched_foreground_explicit_null_candidate_count: int；前景命中且顶层 smiles 显式为 null 的候选数。
        - summary.splits.{split_name}.matched_foreground_missing_prepared_file_candidate_count: int；前景命中但整个 PreparedLigand 文件缺失的候选数。
        - summary.splits.{split_name}.language_available_identity_count: dict[str, int]；逐语言家族的可用身份数。
        - summary.splits.{split_name}.language_available_identity_count.molformer: int；MoLFormer 可用身份数。
        - summary.splits.{split_name}.language_available_identity_count.smi_ted_289m: int；SMI-TED 可用身份数。
        - summary.splits.{split_name}.language_multi_model_smiles_identity_count: dict[str, int]；逐语言家族的多 tokenizer 字符串诊断身份数。
        - summary.splits.{split_name}.language_multi_model_smiles_identity_count.molformer: int；MoLFormer 多字符串身份数。
        - summary.splits.{split_name}.language_multi_model_smiles_identity_count.smi_ted_289m: int；SMI-TED 多字符串身份数。
        - summary.splits.{split_name}.seen_positive_identity_count: dict[str, int]；逐语言家族的训练可见身份数。
        - summary.splits.{split_name}.seen_positive_identity_count.molformer: int；MoLFormer 口径 seen 身份数。
        - summary.splits.{split_name}.seen_positive_identity_count.smi_ted_289m: int；SMI-TED 口径 seen 身份数。
        - summary.excluded_pdb_reason_counts: dict[str, int]；实际出现的整 PDB 排除原因计数。
        - summary.excluded_pdb_reason_counts.blob_exceed: int；条件字段，存在该原因时的 PDB 数。
        - summary.excluded_pdb_reason_counts.missing_required_product: int；条件字段，存在该原因时的 PDB 数。

    返回值:
        - manifest_path: Path；最后原子替换成功的 manifest.json 路径。
    """

    # dict[str, dict[str, str]], 已物化划分到候选/身份 JSONL 相对文件名的映射。
    files: dict[str, dict[str, str]] = {}
    for split_name in SPLIT_NAMES:
        if split_name not in split_candidates:
            continue
        # str, 当前划分候选记录的稳定相对文件名。
        candidate_name = f"{split_name}_candidates.jsonl"
        # str, 当前划分联合身份记录的稳定相对文件名。
        identity_name = f"{split_name}_identities.jsonl"
        _atomic_write_jsonl(roots.manifest_root / candidate_name, split_candidates[split_name])
        _atomic_write_jsonl(roots.manifest_root / identity_name, split_identities[split_name])
        files[split_name] = {"candidates": candidate_name, "identities": identity_name}
    _atomic_write_jsonl(roots.manifest_root / "excluded_pdb.jsonl", exclusions)
    # dict[str, Any], 即将序列化为 manifest.json 的完整主清单对象。
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(roots.data_root),
        "stage1_root": str(roots.stage1_root),
        "output_root": str(roots.output_root),
        "coverage_threshold": coverage_threshold,
        "selection_seed": selection_seed,
        "identity_key_fields": list(IDENTITY_KEY_FIELDS),
        "prepared_smiles_rule": (
            "prepared_smiles 逐字采用 PreparedLigand 顶层 smiles；显式 null 不建立身份；"
            "PreparedLigand 文件缺失单独标记，已有文件不完整时失败。"
        ),
        "identity_class_rule": IDENTITY_CLASS_RULE,
        "split_pdb_rule": SPLIT_PDB_RULE,
        "cross_pdb_joint_identity_class_variants": (
            _joint_identity_cross_pdb_class_variants(split_identities)
        ),
        "language_selection_rule": (
            "对每个 (pdb_id, object_key, prepared_smiles, language_family) 的有效 occurrence "
            "编号排序；沿用 v3 的 (pdb_id, object_key, language_family) 固定字节加权和与 "
            "selection_seed 决定循环起点，选择该位置的 occurrence。"
        ),
        "seen_identity_rule": (
            "按语言模型和 (object_key, prepared_smiles) 分别判断；只有该身份在 train "
            "候选中作为可监督前景正目标出现过，才标记为 seen。"
        ),
        "class_names": list(CLASS_NAMES),
        "available_splits": [name for name in SPLIT_NAMES if name in files],
        "deferred_splits": [name for name in SPLIT_NAMES if name not in files],
        "splits": files,
        "excluded_pdb": "excluded_pdb.jsonl",
        "split_pdb_counts": split_pdb_counts,
        "summary": _summarize(split_candidates, split_identities, exclusions),
    }
    # Path, 当前 schema 版本目录中的最终主清单路径。
    manifest_path = roots.manifest_root / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    return manifest_path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    读取一个每个非空物理行均为 JSON 对象的 UTF-8 JSONL 文件。

    输入参数:
        - path: Path；候选、身份、排除、occurrence 或 PreparedLigand JSONL 文件路径。

    返回值:
        - records: list[dict[str, Any]]；保持非空物理行顺序的 JSON 对象列表，空白行被忽略。
        - records[*]: dict[str, Any]；当前通用读取边界不规定业务字段，物理行中的全部键值由 json.loads 原样保留，具体调用者必须再执行字段门禁。

    JSON 标量、列表和嵌套对象保持 json.loads 的原生 Python 类型；顶层不是对象时按物理行号失败。
    """

    # list[dict[str, Any]], 按非空物理行顺序积累的顶层 JSON 对象。
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            # str, 去掉当前物理行两端空白后的 JSON 文本；空字符串表示可忽略空白行。
            text = line.strip()
            if not text:
                continue
            # Any, 当前物理行的解析结果；正式契约要求顶层必须是 dict。
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON 对象。")
            records.append(value)
    return records


def read_npz_string_scalar(value: Any, field_name: str) -> str:
    """
    读取 NPZ 中的非空 Unicode 标量，拒绝数值、bytes 和非标量数组。

    输入参数:
        - value: Any；通常是 numpy.load 返回的单字段数组，必须具有标量形状且 dtype.kind 为 U。
        - field_name: str；错误消息中的字段定位文本，不参与字符串转换。

    返回值:
        - result: str；数组保存的原始非空 Python 字符串，不执行 strip、大小写转换、规范化或其它隐式类型转换。
    """

    # ndarray, 保留来源标量形状与 Unicode dtype，禁止借由 astype 接受其它类型。
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "U":
        raise ValueError(f"{field_name} 必须是非空 Unicode 标量字符串。")
    # Any, 从零维 Unicode 数组取出的 Python 标量；随后严格要求内建 str 且非空。
    result = array.item()
    if type(result) is not str or not result:
        raise ValueError(f"{field_name} 必须是非空 Unicode 标量字符串。")
    return result


def _allowed_exclusion(
    roots: SourceRoots,
    split_name: str,
    pdb_id: str,
    pdb_directory: Path,
) -> dict[str, Any] | None:
    """
    识别允许整 PDB 跳过的 blob_exceed 标记或必要产物缺失。

    输入参数:
        - roots: SourceRoots；用于定位 Stage1 之外的冻结 A–G 必要产物。
        - roots.data_root: Path；定位 occurrence、受体、密度、真实配体区域和原子标签。
        - roots.stage1_root: Path；定位 F1 候选和完整概率图。
        - roots.output_root: Path；本函数不读取或写入该根，只随统一 SourceRoots 输入保留。
        - split_name: str；当前 Stage1 划分名，写入排除记录。
        - pdb_id: str；小写 PDB 标识，写入排除记录并定位 A–G 目录。
        - pdb_directory: Path；当前 Stage1 PDB 目录，用于检查 _BLOB_EXCEED 和 F1 文件。

    返回记录字段:
        - split: str；发生排除的 Stage1 划分名。
        - pdb_id: str；被整 PDB 排除的小写标识。
        - reason: str；_BLOB_EXCEED 存在时为 blob_exceed，否则必要文件缺失时为 missing_required_product。
        - missing_paths: list[str]；仅 reason 为 missing_required_product 时出现，按必要文件检查顺序列出由调用者来源根拼出的不存在路径文本，不保证为绝对路径。

    返回 None 表示没有允许排除原因，调用者必须进入严格 PDB 扫描；已有但损坏的必要文件不会在这里降级为排除。
    """

    if (pdb_directory / "_BLOB_EXCEED").exists():
        return {"split": split_name, "pdb_id": pdb_id, "reason": "blob_exceed"}
    # tuple[Path, ...], 当前 PDB 建立候选监督所需的八份 Stage1 与 A–G 文件。
    required_paths = _required_pdb_paths(roots, split_name, pdb_id)
    # list[str], 不存在的必要文件路径；被实际读取的文件在对应消费者打开时核验，atom_labels.npz 在本层只检查存在性且不会发现内容错误。
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        return {
            "split": split_name,
            "pdb_id": pdb_id,
            "reason": "missing_required_product",
            "missing_paths": missing,
        }
    return None


def _required_pdb_paths(roots: SourceRoots, split_name: str, pdb_id: str) -> tuple[Path, ...]:
    """
    列出扫描一个 PDB 候选监督所需的 Stage1 与 A–G 文件。

    输入参数:
        - roots: SourceRoots；提供 Stage1、冻结数据和语言产物根，其中语言 prepared 文件故意不属于整 PDB 必要项。
        - roots.data_root: Path；定位六份冻结 A–G 必要文件。
        - roots.stage1_root: Path；定位 F1_centered.npz 和 probability_map.npz。
        - roots.output_root: Path；本函数不消费该字段，只随统一 SourceRoots 输入保留。
        - split_name: str；Stage1 PDB 目录所属划分。
        - pdb_id: str；当前小写 PDB 目录名。

    返回值:
        - required_paths: tuple[Path, ...]；依次为 F1_centered.npz、probability_map.npz、occurrences.jsonl、receptor_tokens.npz、exp.npz、sim.npz、ligand_area.npz、atom_labels.npz。

    PreparedLigand 文件允许整体缺失并在候选字段中显式标记 missing_file，因此不作为整 PDB 排除条件；LigandObject 图和语言向量也按身份级可用性记录，而不触发整 PDB 排除。
    """

    # Path, 当前 Stage1 划分中的 PDB 目录。
    stage1_pdb = roots.stage1_root / split_name / pdb_id
    return (
        stage1_pdb / "centered/F1_centered.npz",
        stage1_pdb / "probability/probability_map.npz",
        roots.data_root / "parse" / pdb_id / "occurrences.jsonl",
        roots.data_root / "parse" / pdb_id / "receptor_tokens.npz",
        roots.data_root / "density" / pdb_id / "exp.npz",
        roots.data_root / "density" / pdb_id / "sim.npz",
        roots.data_root / "density" / pdb_id / "ligand_area.npz",
        roots.data_root / "labels" / pdb_id / "atom_labels.npz",
    )


def _scan_pdb(
    roots: SourceRoots,
    split_name: str,
    pdb_id: str,
    coverage_threshold: float,
    selection_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    扫描一个来源完整的 PDB，构造候选监督记录与五类前景联合身份记录。

    输入参数:
        - roots: SourceRoots；提供 occurrence、真值掩码、PreparedLigand、LigandObject、语言向量和 Stage1 F1 来源。
        - roots.data_root: Path；读取 occurrence、真实配体掩码、PreparedLigand、LigandObject 和语言归档的冻结 A–G 根。
        - roots.stage1_root: Path；读取当前划分 F1_centered.npz 的 Stage1 根。
        - roots.output_root: Path；本函数不消费该字段，只随统一 SourceRoots 输入保留。
        - split_name: str；当前 PDB 所属 Stage1 划分，写入每条候选和身份记录。
        - pdb_id: str；小写 PDB 标识，用于读取来源并写入记录。
        - coverage_threshold: float；候选与 occurrence 双向覆盖率必须共同达到的阈值。
        - selection_seed: int；身份内每个语言家族选择有效 occurrence 归档的确定性种子。

    候选记录字段:
        - split: str；当前 Stage1 划分名。
        - pdb_id: str；当前小写 PDB 标识。
        - candidate_index: int；当前候选在 F1_centered.npz 候选轴中的零基索引。
        - source_tree_id: int；F1 保存的候选来源连通树编号。
        - source_node_id: int；F1 保存的候选来源树节点编号。
        - source_threshold_grid_index: int；F1 保存的阈值网格索引。
        - full_candidate_voxel_count: int；当前候选在 80³ 中保存的稀疏体素数。
        - class_name: str；六分类类别名，未满足阈值或最佳命中 type_tag=other 时为 background。
        - class_index: int；与 class_name 对齐的 CLASS_NAMES 索引。
        - matched_candidate_id: int 或 null；最佳匹配 occurrence 编号；没有 occurrence 通过阈值时为 null，命中 other 时即使 class_name 为 background 也保留编号。
        - matched_object_key: str 或 null；最佳匹配 occurrence 的 LigandObject 模板键；没有 occurrence 通过阈值时为 null。
        - matched_prepared_smiles: str 或 null；仅五类前景最佳命中关联顶层 PreparedLigand.smiles；background、prepared 文件缺失或显式 null 时均为 null。
        - matched_prepared_smiles_status: str；class_name 为 background 或未命中时为 not_applicable，prepared 文件缺失时为 missing_file，顶层 smiles 为 null 时为 explicit_null，非空时为 available。
        - candidate_coverage: float；交集体素数除以候选稀疏体素数，没有 occurrence 通过阈值时为 0。
        - occurrence_coverage: float；交集体素数除以 occurrence 真值体素数，没有 occurrence 通过阈值时为 0。
        - overlap_score: float；candidate_coverage 与 occurrence_coverage 乘积的平方根，没有 occurrence 通过阈值时为 0。
        - language_target_available: dict[str, bool]；联合身份存在、对应 LigandObject 文件存在且相应语言家族至少有一个有效归档时为 true；图内容尚未在本函数核验。
        - language_target_available.molformer: bool；MoLFormer 语言目标是否可用于当前候选。
        - language_target_available.smi_ted_289m: bool；SMI-TED 语言目标是否可用于当前候选。

    F1 稀疏字段:
        - voxel_offsets: 整数数组，(C + 1,)；同时切分 voxel_index_local_zyx、centered_probability 和 voxel_final；本函数只读取前者。
        - voxel_index_local_zyx: 整数数组，(V_total, 3)；拼接全部候选的 80³ 局部稀疏 ZYX 坐标。
        - box_start_zyx: 整数数组，(C, 3)；候选 80³ 在完整密度图中的 ZYX 起点。

    返回值:
        - candidates: list[dict[str, Any]]；按 candidate_index 递增的候选记录。
        - identities: list[dict[str, Any]]；当前 PDB 按 class_index、object_key、prepared_smiles 排序的联合身份记录。
        - identities[*].split: str；当前 Stage1 划分名。
        - identities[*].pdb_id: str；当前小写 PDB 标识。
        - identities[*].class_name: str；联合身份唯一的五类前景类别。
        - identities[*].class_index: int；与 class_name 对齐的六分类索引。
        - identities[*].object_key: str；联合身份模板图分量。
        - identities[*].prepared_smiles: str；联合身份顶层公开接口字符串分量。
        - identities[*].occurrence_candidate_ids: list[int]；共享身份的 occurrence 编号递增列表。
        - identities[*].graph_available: bool；LigandObject 文件是否存在，不代表图内容已经验证。
        - identities[*].graph_relative_path: str；相对 data_root 的 LigandObject NPZ 路径。
        - identities[*].language: dict[str, dict[str, Any]]；两个语言家族的归档选择证据。
        - identities[*].language.molformer: dict[str, Any]；MoLFormer 归档选择证据。
        - identities[*].language.smi_ted_289m: dict[str, Any]；SMI-TED 归档选择证据。
        - identities[*].language.{family}.available: bool；至少一个 occurrence 归档有效时为 true。
        - identities[*].language.{family}.selected_candidate_id: int 或 null；选中的 occurrence 编号。
        - identities[*].language.{family}.selected_relative_path: str 或 null；选中归档相对 language_root 的路径。
        - identities[*].language.{family}.valid_candidate_ids: list[int]；有效 occurrence 归档编号列表。
        - identities[*].language.{family}.failed_occurrences: list[dict[str, Any]]；缺失或无效归档诊断。
        - identities[*].language.{family}.failed_occurrences[*].candidate_id: int；失败归档的 occurrence 编号。
        - identities[*].language.{family}.failed_occurrences[*].reason: str；missing_file 或 invalid_archive。
        - identities[*].language.{family}.failed_occurrences[*].detail: str；仅 invalid_archive 时出现的异常文本。
        - identities[*].language.{family}.distinct_model_smiles: list[str]；有效归档 model_smiles 的排序去重列表。
    """

    # list[dict[str, Any]], 按 candidate_id 排序且完成身份字段核验的 occurrence 主表。
    occurrence_records = _read_occurrences(roots.data_root, pdb_id)
    # dict[int, str | None]，五类前景 occurrence 编号到顶层公开接口字符串或显式 null 的映射。
    # bool，整个 prepared_smiles.jsonl 存在且完整通过门禁时为 true；false 与记录内显式 null 分开解释。
    prepared_smiles_by_candidate, prepared_smiles_file_available = (
        _read_prepared_smiles_by_candidate(
            roots,
            pdb_id,
            occurrence_records,
        )
    )
    # list[OccurrenceMask], 与 occurrence_records 同序的稀疏完整网格 ZYX 真值掩码。
    occurrence_masks = _read_occurrence_masks(roots.data_root, pdb_id, occurrence_records)
    # list[dict[str, Any]], 按联合键分组并完成图/语言可用性选择的 PDB 内身份记录。
    identities = _build_identity_records(
        roots,
        split_name,
        pdb_id,
        occurrence_records,
        prepared_smiles_by_candidate,
        selection_seed,
    )
    # dict[tuple[str,str], dict[str,Any]], object_key、prepared_smiles 联合键到身份记录的映射。
    identity_by_key = {
        (identity["object_key"], identity["prepared_smiles"]): identity
        for identity in identities
    }

    # Path, 当前 PDB 的 F1 候选稀疏体素与来源编号产物。
    centered_path = roots.stage1_root / split_name / pdb_id / "centered/F1_centered.npz"
    # list[dict[str, Any]], 按 F1 候选轴顺序积累的监督记录。
    candidates: list[dict[str, Any]] = []
    with np.load(centered_path, allow_pickle=False) as centered:
        # int, F1 候选轴长度 C；偏移和候选级数组已经通过 _validate_centered_offsets 核验。
        candidate_count = _validate_centered_offsets(centered, centered_path)
        for candidate_index in range(candidate_count):
            # int, 当前候选在拼接稀疏 voxel_index_local_zyx 中的左闭切片起点。
            voxel_start = int(centered["voxel_offsets"][candidate_index])
            # int, 当前候选在拼接稀疏 voxel_index_local_zyx 中的右开切片终点。
            voxel_stop = int(centered["voxel_offsets"][candidate_index + 1])
            # int64, (V_candidate, 3), 当前候选 80³ 内的局部 ZYX 稀疏体素坐标。
            local_voxels = np.asarray(
                centered["voxel_index_local_zyx"][voxel_start:voxel_stop],
                dtype=np.int64,
            )
            # int64, (3,), 当前候选 80³ 在完整密度图中的 ZYX 起点。
            box_start = np.asarray(centered["box_start_zyx"][candidate_index], dtype=np.int64)
            # int64, (V_candidate, 3), 转换到完整密度图坐标系的候选 ZYX 体素坐标。
            global_voxels = local_voxels + box_start[None, :]
            # CandidateTarget, 通过双向覆盖阈值和确定性并列规则选出的六分类监督。
            target = assign_candidate_target(global_voxels, occurrence_masks, coverage_threshold)
            # str | None, 仅前景命中且 prepared 文件存在且顶层 smiles 非空时保存联合身份语言分量。
            matched_prepared_smiles: str | None = None
            if (
                target.class_name not in FOREGROUND_CLASS_NAMES
                or target.matched_candidate_id is None
            ):
                matched_prepared_smiles_status = "not_applicable"
            elif not prepared_smiles_file_available:
                matched_prepared_smiles_status = "missing_file"
            else:
                matched_prepared_smiles = prepared_smiles_by_candidate[
                    target.matched_candidate_id
                ]
                matched_prepared_smiles_status = (
                    "available" if matched_prepared_smiles is not None else "explicit_null"
                )
            # dict[str, Any] | None, 与候选匹配 object_key、prepared_smiles 精确相同的 PDB 内身份。
            identity = identity_by_key.get(
                (target.matched_object_key, matched_prepared_smiles)
            )
            if identity is not None and identity["class_name"] != target.class_name:
                raise ValueError(
                    f"{split_name}/{pdb_id} 的联合身份类别 {identity['class_name']!r} "
                    f"与候选目标类别 {target.class_name!r} 不一致。"
                )
            # dict[str, bool]，每个语言家族同时要求身份存在、LigandObject 文件存在和有效语言归档存在；图内容尚未在此核验。
            language_available = {
                family: bool(
                    identity
                    and identity["graph_available"]
                    and identity["language"][family]["available"]
                )
                for family in LANGUAGE_FAMILIES
            }
            candidates.append(
                {
                    "split": split_name,
                    "pdb_id": pdb_id,
                    "candidate_index": candidate_index,
                    "source_tree_id": int(centered["source_tree_id"][candidate_index]),
                    "source_node_id": int(centered["source_node_id"][candidate_index]),
                    "source_threshold_grid_index": int(
                        centered["source_threshold_grid_index"][candidate_index]
                    ),
                    "full_candidate_voxel_count": int(global_voxels.shape[0]),
                    "class_name": target.class_name,
                    "class_index": target.class_index,
                    "matched_candidate_id": target.matched_candidate_id,
                    "matched_object_key": target.matched_object_key,
                    "matched_prepared_smiles": matched_prepared_smiles,
                    "matched_prepared_smiles_status": matched_prepared_smiles_status,
                    "candidate_coverage": target.candidate_coverage,
                    "occurrence_coverage": target.occurrence_coverage,
                    "overlap_score": target.overlap_score,
                    "language_target_available": language_available,
                }
            )
    return candidates, identities


def _read_occurrences(data_root: Path, pdb_id: str) -> list[dict[str, Any]]:
    """
    读取并验证一个 PDB 的 A–G occurrence 主表中 Matcher 使用的身份字段。

    输入参数:
        - data_root: Path；冻结 A–G 数据根，occurrence 文件位于 parse/{pdb_id}/occurrences.jsonl。
        - pdb_id: str；期望与每条记录顶层 pdb_id 忽略大小写后相同的小写目录标识。

    使用字段:
        - pdb_id: JSON 值；执行 str(value).lower() 后必须与当前目录名一致。
        - candidate_id: 可由 int(value) 转换的 JSON 值；PDB 内 occurrence 编号，转换后必须唯一，并用于稳定排序、掩码字段名和语言归档文件名；本函数不额外要求非负。
        - object_key: 非空字符串；定位 LigandObject 模板图的键。
        - type_tag: JSON 值；执行 str(value) 后必须为 ion、nucleotide_like、peptide_like、small_molecule、sugar 或 other，再由 OCCURRENCE_TYPE_TO_CLASS 映射到六分类。

    返回值:
        - records: list[dict[str, Any]]；保留每条 occurrence 的完整 JSON 对象，按 int(candidate_id) 递增排序；本函数未解释的附加字段原样保留。
        - records[*].pdb_id: JSON 值；执行 str(value).lower() 后必须等于输入 pdb_id。
        - records[*].candidate_id: JSON 值；必须可由 int 转换，转换后的整数在 PDB 内唯一并决定返回排序。
        - records[*].object_key: str；非空 LigandObject 模板键。
        - records[*].type_tag: JSON 值；执行 str(value) 后必须属于六个支持枚举。
    """

    # Path, 当前 PDB 的 occurrence 主表 JSONL。
    path = data_root / "parse" / pdb_id / "occurrences.jsonl"
    # list[dict[str, Any]], 按来源物理行读取的完整 occurrence 对象。
    records = load_jsonl(path)
    for record in records:
        for field in ("candidate_id", "object_key", "type_tag"):
            if field not in record:
                raise KeyError(f"{path} 缺少 occurrence 字段 {field}。")
        if str(record.get("pdb_id", "")).lower() != pdb_id:
            raise ValueError(f"{path} 的 pdb_id 与目录名不一致。")
        try:
            int(record["candidate_id"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path} 的 candidate_id 必须可转换为整数。") from error
        if not isinstance(record["object_key"], str) or not record["object_key"]:
            raise ValueError(f"{path} 的 object_key 必须为非空字符串。")
        if str(record["type_tag"]) not in {"ion", "nucleotide_like", "peptide_like", "small_molecule", "sugar", "other"}:
            raise ValueError(f"{path} 含不支持的 type_tag={record['type_tag']!r}。")
    # list[int], 逐记录规范化的 occurrence 编号，用于唯一性门禁。
    candidate_ids = [int(record["candidate_id"]) for record in records]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"{path} 含重复 candidate_id。")
    return sorted(records, key=lambda record: int(record["candidate_id"]))


def _read_prepared_smiles_by_candidate(
    roots: SourceRoots,
    pdb_id: str,
    occurrence_records: Iterable[dict[str, Any]],
) -> tuple[dict[int, str | None], bool]:
    """
    读取一个 PDB 每个五类前景 occurrence 的顶层 PreparedLigand.smiles。

    输入参数:
        - roots: SourceRoots；language_root 用于定位 prepared/{pdb_id}/prepared_smiles.jsonl。
        - roots.data_root: Path；派生 language_root 的冻结 A–G 根。
        - roots.stage1_root: Path；本函数不消费该字段，只随统一 SourceRoots 输入保留。
        - roots.output_root: Path；本函数不消费该字段，只随统一 SourceRoots 输入保留。
        - pdb_id: str；当前小写 PDB 标识，必须与 PreparedLigand.pdb_id 和 occurrence 目录一致。
        - occurrence_records: Iterable[dict[str, Any]]；已核验并按 candidate_id 排序的 occurrence 表。
        - occurrence_records[*].candidate_id: JSON 值；可由 int 转换的 PDB 内 occurrence 编号。
        - occurrence_records[*].object_key: str；与 PreparedLigand.object_key 逐字核对的非空模板键。
        - occurrence_records[*].type_tag: JSON 值；执行 str(value) 后用于筛选五类前景并与 PreparedLigand.type_tag 核对。

    PreparedLigand 顶层字段:
        - pdb_id: 非空小写字符串；必须逐字等于当前 pdb_id。
        - candidate_id: 非负 int；必须对应 occurrence 表中的五类前景编号且在文件内唯一。
        - object_key: 非空字符串；必须逐字等于同编号 occurrence.object_key。
        - kind: 非空字符串；通常为 CCD 或 BRANCHED，本函数只验证非空而不参与身份。
        - type_tag: str；必须逐字等于同编号 occurrence.type_tag。
        - is_covalent: bool；Stage C 保存的受体共价连接事实，本函数只验证类型而不参与身份。
        - smiles: 非空字符串或 null；逐字采用交给语言模型公开接口的顶层字符串，通常为 canonical non-isomeric 字符串，diagnostics.used_raw_smiles=true 时为原文回退；身份只认该顶层值，null 不建立身份。
        - smiles_source: str；必须属于 ccd、clc、present_graph、unsupported、occurrence_error，本函数只作来源枚举门禁。
        - diagnostics: dict[str, Any]；化学转换、输入缺失和异常事实；本函数只要求顶层为 JSON 对象，不读取任何嵌套值，也不允许其影响联合身份。

    返回值:
        - prepared_smiles_by_candidate: dict[int, str | None]；每个五类前景 candidate_id 到原样非空 smiles 或显式 None 的映射。
        - prepared_smiles_file_available: bool；prepared_smiles.jsonl 存在并通过完整契约时为 true，整个文件不存在时为 false。

    文件不存在时返回空映射与 false。文件一旦存在，九个顶层字段必须恰好齐全且不得增加字段，五类前景 candidate_id 集合必须完整；任何身份漂移或不完整均失败关闭。
    """

    # dict[int, dict[str, Any]], occurrence 编号到完整 occurrence 对象的唯一映射。
    occurrences = {int(record["candidate_id"]): record for record in occurrence_records}
    # set[int], occurrence 主表中映射到五类非背景类别的全部编号。
    formal_candidate_ids = {
        candidate_id
        for candidate_id, record in occurrences.items()
        if OCCURRENCE_TYPE_TO_CLASS[str(record["type_tag"])] in FOREGROUND_CLASS_NAMES
    }
    # Path, 当前 PDB 的 PreparedLigand JSONL；整体缺失是显式允许状态。
    path = roots.language_root / "prepared" / pdb_id / "prepared_smiles.jsonl"
    if not path.is_file():
        return {}, False

    # set[str], PreparedLigand 顶层必须恰好包含的九个字段；额外字段同样失败。
    required_fields = {
        "pdb_id",
        "candidate_id",
        "object_key",
        "kind",
        "type_tag",
        "is_covalent",
        "smiles",
        "smiles_source",
        "diagnostics",
    }
    # dict[int, str | None], 按 candidate_id 收集顶层 smiles，显式 null 保留为 None。
    prepared_smiles: dict[int, str | None] = {}
    for record in load_jsonl(path):
        # list[str], 当前记录缺失的正式顶层字段。
        missing = sorted(required_fields.difference(record))
        # list[str], 当前记录超出正式 PreparedLigand schema 的顶层字段。
        extra = sorted(set(record).difference(required_fields))
        if missing or extra:
            raise ValueError(f"{path} 的 PreparedLigand 字段漂移：missing={missing}, extra={extra}。")
        if type(record["candidate_id"]) is not int or int(record["candidate_id"]) < 0:
            raise ValueError(f"{path}:candidate_id 必须是非负整数。")
        # int, 当前 PreparedLigand 的非负 occurrence 编号。
        candidate_id = int(record["candidate_id"])
        if candidate_id in prepared_smiles:
            raise ValueError(f"{path} 含重复 candidate_id={candidate_id}。")
        if candidate_id not in formal_candidate_ids:
            raise ValueError(f"{path} 含非五类前景 occurrence candidate_id={candidate_id}。")
        # dict[str, Any], 同编号 A–G occurrence，用于交叉核验 PDB、模板键与类别标签。
        occurrence = occurrences[candidate_id]
        # dict[str, str]，必须与 PreparedLigand 顶层逐字相同的三个身份字段。
        # expected.pdb_id 是当前小写 PDB 标识。
        # expected.object_key 来自同编号 occurrence 的模板键。
        # expected.type_tag 来自同编号 occurrence 的类别来源标签。
        expected = {
            "pdb_id": pdb_id,
            "object_key": str(occurrence["object_key"]),
            "type_tag": str(occurrence["type_tag"]),
        }
        for field, expected_value in expected.items():
            if record[field] != expected_value:
                raise ValueError(
                    f"{path}:candidate_id={candidate_id} 的 {field}={record[field]!r}，"
                    f"期望 {expected_value!r}。"
                )
        if type(record["kind"]) is not str or not record["kind"]:
            raise ValueError(f"{path}:kind 必须是非空字符串。")
        if type(record["is_covalent"]) is not bool:
            raise ValueError(f"{path}:is_covalent 必须是布尔值。")
        # str | None, 顶层公开接口输入；非空时逐字成为联合身份语言分量。
        smiles = record["smiles"]
        if smiles is not None and (type(smiles) is not str or not smiles.strip()):
            raise ValueError(f"{path}:smiles 必须是非空字符串或 null。")
        if record["smiles_source"] not in {
            "ccd",
            "clc",
            "present_graph",
            "unsupported",
            "occurrence_error",
        }:
            raise ValueError(f"{path}:smiles_source 不属于 PreparedLigand 枚举。")
        if type(record["diagnostics"]) is not dict:
            raise ValueError(f"{path}:diagnostics 必须是 JSON 对象。")
        prepared_smiles[candidate_id] = smiles

    # list[int], occurrence 主表中存在但 PreparedLigand 文件缺失的五类前景编号。
    missing_candidate_ids = sorted(formal_candidate_ids.difference(prepared_smiles))
    if missing_candidate_ids:
        raise ValueError(f"{path} 缺少五类前景 candidate_id={missing_candidate_ids}。")
    return prepared_smiles, True


def _read_occurrence_masks(
    data_root: Path,
    pdb_id: str,
    occurrence_records: Iterable[dict[str, Any]],
) -> list[OccurrenceMask]:
    """
    读取每个 occurrence 的稀疏真值体素坐标，不解压整个 PDB 的 union_mask。

    输入参数:
        - data_root: Path；冻结 A–G 数据根，掩码位于 density/{pdb_id}/ligand_area.npz。
        - pdb_id: str；当前小写 PDB 标识。
        - occurrence_records: Iterable[dict[str, Any]]；已核验并排序的 occurrence 对象。
        - occurrence_records[*].candidate_id: JSON 值；可由 int 转换，用于定位 mask_{candidate_id}。
        - occurrence_records[*].object_key: str；写入返回 OccurrenceMask.object_key 的模板键。
        - occurrence_records[*].type_tag: JSON 值；执行 str(value) 后写入返回 OccurrenceMask.type_tag。

    来源字段:
        - mask_{candidate_id}: int32 数组，(V_occurrence, 3)；当前 occurrence 在完整密度图中的稀疏 ZYX 体素坐标，读取时规范化为 int64。
        - union_mask: bool 数组，(1, Z_full, Y_full, X_full)；True 表示至少属于一个稀疏 occurrence 掩码，本函数故意不读取以避免不必要的大数组解压。

    返回值:
        - occurrence_masks: list[OccurrenceMask]；与 occurrence_records 顺序一致，每项保存 candidate_id、object_key、type_tag 和规范化 int64 稀疏坐标。
        - occurrence_masks[*].candidate_id: int；PDB 内 occurrence 编号。
        - occurrence_masks[*].object_key: str；LigandObject 模板键。
        - occurrence_masks[*].type_tag: str；用于映射六分类的 occurrence 类型。
        - occurrence_masks[*].voxel_index_zyx: int64 数组，(V_occurrence, 3)；完整密度图坐标系中的稀疏 ZYX 真值体素。
    """

    # Path, 当前 PDB 的 occurrence 稀疏配体区域 NPZ。
    path = data_root / "density" / pdb_id / "ligand_area.npz"
    # list[OccurrenceMask], 与 occurrence_records 同序积累的稀疏真值对象。
    result: list[OccurrenceMask] = []
    with np.load(path, allow_pickle=False) as archive:
        for record in occurrence_records:
            candidate_id = int(record["candidate_id"])
            # str, 当前 occurrence 在 ligand_area.npz 中的候选专属稀疏掩码字段名。
            key = f"mask_{candidate_id}"
            if key not in archive.files:
                raise KeyError(f"{path} 缺少 {key}。")
            # int64, (V_occurrence, 3), 完整密度图坐标系下的 ZYX 真值体素坐标。
            mask = np.asarray(archive[key], dtype=np.int64)
            result.append(
                OccurrenceMask(
                    candidate_id=candidate_id,
                    object_key=str(record["object_key"]),
                    type_tag=str(record["type_tag"]),
                    voxel_index_zyx=mask,
                )
            )
    return result


def _build_identity_records(
    roots: SourceRoots,
    split_name: str,
    pdb_id: str,
    occurrence_records: Iterable[dict[str, Any]],
    prepared_smiles_by_candidate: dict[int, str | None],
    selection_seed: int,
) -> list[dict[str, Any]]:
    """
    按 object_key 与顶层 PreparedLigand.smiles 构造 PDB 内前景联合身份。

    输入参数:
        - roots: SourceRoots；用于检查 LigandObject 图，并定位两个语言家族的 occurrence 归档。
        - roots.data_root: Path；检查 LigandObject 文件是否存在，并派生 language_root。
        - roots.stage1_root: Path；本函数不消费该字段，只随统一 SourceRoots 输入保留。
        - roots.output_root: Path；本函数不消费该字段，只随统一 SourceRoots 输入保留。
        - split_name: str；当前 PDB 所属划分，写入每条身份记录。
        - pdb_id: str；当前小写 PDB 标识，限定身份轴并定位语言归档。
        - occurrence_records: Iterable[dict[str, Any]]；已核验 occurrence 主表。
        - occurrence_records[*].candidate_id: JSON 值；可由 int 转换的 PDB 内 occurrence 编号。
        - occurrence_records[*].object_key: str；联合身份模板图分量。
        - occurrence_records[*].type_tag: JSON 值；执行 str(value) 后映射到六分类，只有五类前景可建立身份。
        - prepared_smiles_by_candidate: dict[int, str | None]；五类前景 occurrence 到顶层 PreparedLigand.smiles 的映射；缺失键或 None 均不建立身份。
        - selection_seed: int；每个身份、每个语言家族选择有效 occurrence 归档的确定性种子。

    返回值:
        - identities: list[dict[str, Any]]；按 class_index、object_key、prepared_smiles 排序的身份记录。
        - identities[*].split: str；当前 Stage1 划分名。
        - identities[*].pdb_id: str；当前小写 PDB 标识。
        - identities[*].class_name: str；共享该联合身份的 occurrence 唯一五类前景类别。
        - identities[*].class_index: int；与 class_name 对齐的 CLASS_NAMES 索引。
        - identities[*].object_key: str；联合身份第一分量，定位 LigandObject 模板图。
        - identities[*].prepared_smiles: str；联合身份第二分量，逐字来自 PreparedLigand 顶层 smiles。
        - identities[*].occurrence_candidate_ids: list[int]；共享该联合身份的 occurrence 编号，严格递增。
        - identities[*].graph_available: bool；graph_relative_path 指向的 LigandObject NPZ 是否存在，不表示图内容已经核验。
        - identities[*].graph_relative_path: str；相对 data_root 的 ligand_objects/{safe_object_filename(object_key)}.npz。
        - identities[*].language: dict[str, dict[str, Any]]；molformer 与 smi_ted_289m 两个家族的归档可用性和选择证据。
        - identities[*].language.molformer: dict[str, Any]；MoLFormer occurrence 归档的核验与选择结果。
        - identities[*].language.smi_ted_289m: dict[str, Any]；SMI-TED occurrence 归档的核验与选择结果。
        - identities[*].language.{family}.available: bool；至少一个 occurrence 归档存在并通过门禁时为 true。
        - identities[*].language.{family}.selected_candidate_id: int 或 null；available 为 true 时确定性选中的 occurrence 编号。
        - identities[*].language.{family}.selected_relative_path: str 或 null；available 为 true 时选中归档相对 language_root 的路径。
        - identities[*].language.{family}.valid_candidate_ids: list[int]；按 occurrence 顺序保留的全部有效归档编号。
        - identities[*].language.{family}.failed_occurrences: list[dict[str, Any]]；按 occurrence 顺序保存的缺失或无效归档诊断。
        - identities[*].language.{family}.failed_occurrences[*].candidate_id: int；失败归档对应的 occurrence 编号。
        - identities[*].language.{family}.failed_occurrences[*].reason: str；missing_file 或 invalid_archive。
        - identities[*].language.{family}.failed_occurrences[*].detail: str；仅 invalid_archive 时出现的异常文本。
        - identities[*].language.{family}.distinct_model_smiles: list[str]；有效归档中非空 model_smiles 的排序去重列表，只作诊断。

    相同 object_key、不同 prepared_smiles 保持为不同身份；相同 prepared_smiles、不同 object_key 也不合并。显式 null 或 PreparedLigand 文件缺失的 occurrence 不建立身份。
    """

    # defaultdict[tuple[str,str], list[tuple[int,str]]], 联合键到 occurrence 编号与类别名的列表。
    grouped: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for record in occurrence_records:
        # str, 当前 occurrence 的六分类类别名。
        class_name = OCCURRENCE_TYPE_TO_CLASS[str(record["type_tag"])]
        # int, 当前 PDB 内 occurrence 编号。
        candidate_id = int(record["candidate_id"])
        # str | None, 顶层 PreparedLigand.smiles；None 或缺失键均不进入联合身份。
        prepared_smiles = prepared_smiles_by_candidate.get(candidate_id)
        if class_name in FOREGROUND_CLASS_NAMES and prepared_smiles is not None:
            grouped[(str(record["object_key"]), prepared_smiles)].append(
                (candidate_id, class_name)
            )

    # list[dict[str, Any]], 按联合键构造后再按类别与联合键稳定排序的身份记录。
    identities: list[dict[str, Any]] = []
    for object_key, prepared_smiles in sorted(grouped):
        # list[tuple[int,str]], 当前联合键下的 occurrence 编号与类别名。
        grouped_occurrences = grouped[(object_key, prepared_smiles)]
        # set[str], 当前 PDB 内联合身份观察到的前景类别，正式契约要求恰好一个。
        class_names = {class_name for _, class_name in grouped_occurrences}
        if len(class_names) != 1:
            raise ValueError(
                f"{pdb_id} 的联合身份 ({object_key!r}, {prepared_smiles!r}) "
                f"对应多个六分类类别：{sorted(class_names)}。"
            )
        # str, 当前联合身份唯一的五类前景类别名。
        class_name = next(iter(class_names))
        # list[int], 共享当前联合身份的 occurrence 编号，严格递增。
        occurrence_ids = sorted(candidate_id for candidate_id, _ in grouped_occurrences)
        # str, 相对 data_root 的 LigandObject 模板图路径。
        graph_relative_path = f"ligand_objects/{safe_object_filename(object_key)}.npz"
        # bool, 模板图文件当前是否存在；内容合法性由后续 LigandGraphStore 核验。
        graph_available = (roots.data_root / graph_relative_path).is_file()
        # dict[str, dict[str, Any]], 两个语言家族分别独立核验和选择 occurrence 表示。
        language = {
            family: _select_language_representation(
                roots,
                pdb_id,
                object_key,
                prepared_smiles,
                occurrence_ids,
                family,
                selection_seed,
            )
            for family in LANGUAGE_FAMILIES
        }
        identities.append(
            {
                "split": split_name,
                "pdb_id": pdb_id,
                "class_name": class_name,
                "class_index": CLASS_TO_INDEX[class_name],
                "object_key": object_key,
                "prepared_smiles": prepared_smiles,
                "occurrence_candidate_ids": occurrence_ids,
                "graph_available": graph_available,
                "graph_relative_path": graph_relative_path,
                "language": language,
            }
        )
    return sorted(
        identities,
        key=lambda item: (
            int(item["class_index"]),
            str(item["object_key"]),
            str(item["prepared_smiles"]),
        ),
    )


def _select_language_representation(
    roots: SourceRoots,
    pdb_id: str,
    object_key: str,
    prepared_smiles: str,
    occurrence_ids: list[int],
    family: str,
    selection_seed: int,
) -> dict[str, Any]:
    """
    核验联合身份内每个 occurrence 的语言归档，并从有效集合确定性选择一个。

    输入参数:
        - roots: SourceRoots；language_root 用于定位 {family}/{pdb_id}/candidate_{candidate_id}.npz。
        - roots.data_root: Path；派生两个语言家族共同的 language_root。
        - roots.stage1_root: Path；本函数不消费该字段，只随统一 SourceRoots 输入保留。
        - roots.output_root: Path；本函数不消费该字段，只随统一 SourceRoots 输入保留。
        - pdb_id: str；当前小写 PDB 标识，必须与归档 pdb_id 一致。
        - object_key: str；当前联合身份模板键，必须与归档 object_key 一致。
        - prepared_smiles: str；当前联合身份顶层公开接口输入，必须与归档 prepared_smiles 逐字一致。
        - occurrence_ids: list[int]；共享该联合身份的 occurrence 编号，调用者提供严格递增顺序。
        - family: str；语言家族键 molformer 或 smi_ted_289m，决定目录与期望 model_name。
        - selection_seed: int；与稳定盐值共同决定有效归档循环索引的整数种子。

    返回记录字段:
        - available: bool；至少一个 occurrence 归档存在并通过联合身份、模型名和向量门禁时为 true。
        - selected_candidate_id: int 或 null；available 为 true 时确定性选中的 occurrence 编号，否则为 null。
        - selected_relative_path: str 或 null；available 为 true 时相对 language_root 的选中 NPZ 路径，否则为 null。
        - valid_candidate_ids: list[int]；按 occurrence_ids 顺序保留的全部有效归档编号。
        - failed_occurrences: list[dict[str, Any]]；缺失或无效 occurrence 归档的诊断列表，保持 occurrence_ids 顺序。
        - failed_occurrences[*].candidate_id: int；失败归档对应的 occurrence 编号。
        - failed_occurrences[*].reason: str；文件不存在时为 missing_file，归档核验失败时为 invalid_archive。
        - failed_occurrences[*].detail: str；仅 reason 为 invalid_archive 时出现，保存捕获异常文本。
        - distinct_model_smiles: list[str]；所有有效归档中通过当前非空 Unicode 标量门禁的 model_smiles 精确去重后按字典序排列；model_smiles 只诊断 tokenizer 输入，不定义身份。

    有效归档必须同时匹配 pdb_id、candidate_id、object_key、prepared_smiles 与语言模型名称。选择盐值继续使用 pdb_id、object_key、family 三段文本，不加入 prepared_smiles，使没有拆分的 v3 身份保持原 occurrence 选择位置。
    """

    # list[dict[str, Any]], 按 occurrence_ids 顺序保存通过归档门禁的元数据。
    valid: list[dict[str, Any]] = []
    # list[dict[str, Any]], 按 occurrence_ids 顺序保存缺失或无效归档的失败证据。
    failures: list[dict[str, Any]] = []
    for candidate_id in occurrence_ids:
        # str, 当前 occurrence 语言归档相对 language_root 的 POSIX 风格路径文本。
        relative_path = f"{family}/{pdb_id}/candidate_{candidate_id}.npz"
        # Path, 当前 occurrence 的实际语言向量 NPZ 路径。
        path = roots.language_root / relative_path
        if not path.is_file():
            failures.append({"candidate_id": candidate_id, "reason": "missing_file"})
            continue
        try:
            # dict[str, Any], 通过门禁的 candidate_id、prepared_smiles 与诊断 model_smiles。
            metadata = _validate_language_archive(
                path,
                pdb_id,
                candidate_id,
                object_key,
                prepared_smiles,
                family,
            )
        except (KeyError, OSError, ValueError) as error:
            failures.append(
                {
                    "candidate_id": candidate_id,
                    "reason": "invalid_archive",
                    "detail": str(error),
                }
            )
            continue
        metadata["relative_path"] = relative_path
        valid.append(metadata)

    if not valid:
        return {
            "available": False,
            "selected_candidate_id": None,
            "selected_relative_path": None,
            "valid_candidate_ids": [],
            "failed_occurrences": failures,
            "distinct_model_smiles": [],
        }

    # str, 与 v3 相同的稳定选择盐值；故意不含 prepared_smiles。
    selector_text = f"{pdb_id}|{object_key}|{family}"
    # int, UTF-8 字节按一基位置加权后的确定性整数，不使用运行时随机状态。
    weighted_bytes = sum(
        (position + 1) * byte for position, byte in enumerate(selector_text.encode("utf-8"))
    )
    # dict[str, Any], 由 selection_seed 和稳定字节和在有效列表上循环选出的归档元数据。
    selected = valid[(selection_seed + weighted_bytes) % len(valid)]
    return {
        "available": True,
        "selected_candidate_id": selected["candidate_id"],
        "selected_relative_path": selected["relative_path"],
        "valid_candidate_ids": [item["candidate_id"] for item in valid],
        "failed_occurrences": failures,
        "distinct_model_smiles": sorted({item["model_smiles"] for item in valid}),
    }


def _validate_language_archive(
    path: Path,
    pdb_id: str,
    candidate_id: int,
    object_key: str,
    prepared_smiles: str,
    family: str,
) -> dict[str, Any]:
    """
    核验一个 occurrence 语言归档的联合身份、模型名称和有限 768 维向量。

    输入参数:
        - path: Path；candidate_{candidate_id}.npz 的文件路径。
        - pdb_id: str；期望的小写 PDB 标识。
        - candidate_id: int；期望的 occurrence 编号。
        - object_key: str；期望的联合身份模板图键。
        - prepared_smiles: str；期望的顶层 PreparedLigand.smiles，必须与归档逐字相同。
        - family: str；语言家族键，用 LANGUAGE_MODEL_NAMES 查得期望 model_name。

    NPZ 字段:
        - pdb_id: Unicode 标量；occurrence 所属 PDB，读取后转小写与输入核对。
        - candidate_id: 标量；必须可由 int 转换为 PDB 内 occurrence 编号，本函数不单独限制来源整数 dtype。
        - object_key: Unicode 标量；CPU 阶段使用的 LigandObject 模板键。
        - model_name: Unicode 标量；生成向量的官方模型名称。
        - prepared_smiles: Unicode 标量；传给模型公开接口的顶层字符串，参与联合身份门禁。
        - model_smiles: 非空 Unicode 标量；已确认进入 tokenizer 的字符串，只作诊断而不定义身份；当前读取器复用 read_npz_string_scalar，因此来源用空字符串表示未知 tokenizer 输入时，该归档会被判为无效。
        - embedding: 数值数组，(768,)；分子级向量，读取为 float32，必须全部有限。

    返回记录字段:
        - candidate_id: int；通过核验的 occurrence 编号。
        - prepared_smiles: str；通过核验的公开接口字符串，与输入逐字相同。
        - model_smiles: str；归档记录的 tokenizer 输入诊断字符串。
    """

    with np.load(path, allow_pickle=False) as archive:
        # set[str], 语言归档必须至少包含的七个字段；附加字段不参与本函数判断。
        required = {
            "pdb_id",
            "candidate_id",
            "object_key",
            "model_name",
            "prepared_smiles",
            "model_smiles",
            "embedding",
        }
        # list[str], 当前归档缺失的必要字段，按字典序报告。
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"缺少字段 {missing}")
        # dict[str, str | int]，从 NPZ 严格标量读取的实际身份与模型元数据。
        # actual.pdb_id 是转为小写的 Unicode 标量。
        # actual.candidate_id 是由 NPZ 标量转换的 occurrence 编号。
        # actual.object_key 是模板图键。
        # actual.model_name 是归档声明的公开模型名。
        # actual.prepared_smiles 是顶层公开接口字符串。
        # actual.model_smiles 是实际 tokenizer 输入诊断字符串。
        actual = {
            "pdb_id": read_npz_string_scalar(archive["pdb_id"], f"{path}:pdb_id").lower(),
            "candidate_id": int(archive["candidate_id"].item()),
            "object_key": read_npz_string_scalar(
                archive["object_key"],
                f"{path}:object_key",
            ),
            "model_name": read_npz_string_scalar(
                archive["model_name"],
                f"{path}:model_name",
            ),
            "prepared_smiles": read_npz_string_scalar(
                archive["prepared_smiles"],
                f"{path}:prepared_smiles",
            ),
            "model_smiles": read_npz_string_scalar(
                archive["model_smiles"],
                f"{path}:model_smiles",
            ),
        }
        # float32, (768,), 分子级语言向量；关闭 NPZ 前读入并在外部核验有限性。
        embedding = np.asarray(archive["embedding"], dtype=np.float32)
    # dict[str, str | int]，当前联合身份与语言家族要求的精确归档元数据。
    # expected.pdb_id、candidate_id、object_key、prepared_smiles 分别来自函数输入。
    # expected.model_name 由 family 在 LANGUAGE_MODEL_NAMES 中查得。
    expected = {
        "pdb_id": pdb_id,
        "candidate_id": candidate_id,
        "object_key": object_key,
        "model_name": LANGUAGE_MODEL_NAMES[family],
        "prepared_smiles": prepared_smiles,
    }
    for field, expected_value in expected.items():
        if actual[field] != expected_value:
            raise ValueError(f"{field}={actual[field]!r}，期望 {expected_value!r}")
    if embedding.shape != (LANGUAGE_EMBEDDING_DIM,) or not np.isfinite(embedding).all():
        raise ValueError(f"embedding 必须为有限 ({LANGUAGE_EMBEDDING_DIM},)，实际为 {embedding.shape}")
    return {
        "candidate_id": candidate_id,
        "prepared_smiles": actual["prepared_smiles"],
        "model_smiles": actual["model_smiles"],
    }


def _validate_centered_offsets(archive: Any, path: Path) -> int:
    """
    核验 F1-centered 候选轴与拼接稀疏体素偏移字段的基本对齐关系。

    输入参数:
        - archive: Any；打开状态的 F1_centered.npz NpzFile，必须提供 files 和字段下标访问。
        - path: Path；错误消息中的 F1_centered.npz 文件路径。

    必要字段:
        - centered_box_index: 数组，第一维长度 C；中心化候选编号，本函数只用其长度定义候选数。
        - box_start_zyx: 数组，第一维长度 C；每个候选 80³ 在完整密度图中的 ZYX 起点。
        - source_tree_id: 数组，第一维长度 C；每个候选的来源连通树编号。
        - source_node_id: 数组，第一维长度 C；每个候选的来源树节点编号。
        - source_threshold_grid_index: 数组，第一维长度 C；每个候选的来源阈值网格索引。
        - voxel_offsets: 整数数组，(C + 1,)；候选级右开偏移，同时切分 voxel_index_local_zyx、centered_probability 和 voxel_final；本函数以 voxel_index_local_zyx 长度核验终点。
        - voxel_index_local_zyx: 数组，(V_total, 3)；拼接全部候选的 80³ 局部 ZYX 稀疏体素坐标。

    返回值:
        - candidate_count: int；centered_box_index 第一维长度 C。

    本函数只验证必要字段存在、四个候选级数组第一维与 C 对齐，以及 voxel_offsets 从 0 开始、单调不减并以 voxel_index_local_zyx 第一维结束；各字段更细的形状和数值门禁由后续消费者承担。
    """

    # set[str], 候选监督扫描需要的 F1 候选级、偏移和稀疏坐标字段。
    required = {
        "centered_box_index",
        "box_start_zyx",
        "source_tree_id",
        "source_node_id",
        "source_threshold_grid_index",
        "voxel_offsets",
        "voxel_index_local_zyx",
    }
    # list[str], 当前 F1 归档缺失的必要字段，按字典序报告。
    missing = sorted(required.difference(archive.files))
    if missing:
        raise KeyError(f"{path} 缺少字段 {missing}。")
    # int, F1 候选轴长度 C。
    candidate_count = int(archive["centered_box_index"].shape[0])
    for key in (
        "box_start_zyx",
        "source_tree_id",
        "source_node_id",
        "source_threshold_grid_index",
    ):
        if archive[key].shape[0] != candidate_count:
            raise ValueError(f"{path}:{key} 第一维与候选数量不一致。")
    # int64, (C+1,), 切分拼接稀疏候选字段的右开偏移。
    offsets = np.asarray(archive["voxel_offsets"], dtype=np.int64)
    if (
        offsets.shape != (candidate_count + 1,)
        or offsets[0] != 0
        or np.any(offsets[1:] < offsets[:-1])
        or offsets[-1] != archive["voxel_index_local_zyx"].shape[0]
    ):
        raise ValueError(f"{path}:voxel_offsets 非法。")
    return candidate_count


def _summarize(
    split_candidates: dict[str, list[dict[str, Any]]],
    split_identities: dict[str, list[dict[str, Any]]],
    exclusions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    汇总每个已物化划分的候选、联合身份、语言监督和整 PDB 排除计数。

    输入参数:
        - split_candidates: dict[str, list[dict[str, Any]]]；已经完成 seen 标注的划分候选记录。
        - split_candidates.{split_name}[*].class_name: str；六分类名称，用于类别计数和前景筛选。
        - split_candidates.{split_name}[*].matched_prepared_smiles: str 或 null；用于统计前景命中但顶层字符串不可用的候选。
        - split_candidates.{split_name}[*].matched_prepared_smiles_status: str；用于分别统计 explicit_null 与 missing_file。
        - split_identities: dict[str, list[dict[str, Any]]]；已经完成 seen 标注的划分身份记录。
        - split_identities.{split_name}[*].pdb_id: str；与 object_key 组成同 PDB 模板键分组。
        - split_identities.{split_name}[*].object_key: str；用于模板键去重和同 PDB 多字符串分组。
        - split_identities.{split_name}[*].language: dict[str, dict[str, Any]]；两个语言家族的归档证据。
        - split_identities.{split_name}[*].language.molformer: dict[str, Any]；MoLFormer 归档证据。
        - split_identities.{split_name}[*].language.smi_ted_289m: dict[str, Any]；SMI-TED 归档证据。
        - split_identities.{split_name}[*].language.{family}.available: bool；用于统计至少一个有效归档的身份。
        - split_identities.{split_name}[*].language.{family}.distinct_model_smiles: list[str]；长度大于一时计入多 tokenizer 字符串诊断。
        - split_identities.{split_name}[*].seen_in_train_by_language: dict[str, bool]；逐语言家族训练可见性。
        - split_identities.{split_name}[*].seen_in_train_by_language.molformer: bool；MoLFormer 口径 seen。
        - split_identities.{split_name}[*].seen_in_train_by_language.smi_ted_289m: bool；SMI-TED 口径 seen。
        - exclusions: list[dict[str, Any]]；全部已物化划分的整 PDB 排除记录。
        - exclusions[*].reason: str；blob_exceed 或 missing_required_product，用于排除原因计数。

    返回顶层字段:
        - splits: dict[str, dict[str, Any]]；只含当前已经物化的划分，按 SPLIT_NAMES 稳定插入。
        - excluded_pdb_reason_counts: dict[str, int]；按 reason 字符串排序的整 PDB 排除计数。
        - excluded_pdb_reason_counts.blob_exceed: int；存在该原因时，带 _BLOB_EXCEED 标记的 PDB 数。
        - excluded_pdb_reason_counts.missing_required_product: int；存在该原因时，缺少至少一份必要产物的 PDB 数。

    每个 summary.splits.{split_name} 字段:
        - candidate_count: int；当前划分候选记录总数。
        - class_counts: dict[str, int]；实际出现的候选类别计数，键按类别名排序，未出现类别省略。
        - class_counts.background: int；存在背景候选时的数量。
        - class_counts.ion: int；存在 ion 候选时的数量。
        - class_counts.nucleotide_like: int；存在 nucleotide_like 候选时的数量。
        - class_counts.peptide_like: int；存在 peptide_like 候选时的数量。
        - class_counts.small_molecule: int；存在 small_molecule 候选时的数量。
        - class_counts.sugar: int；存在 sugar 候选时的数量。
        - identity_count: int；prepared_smiles 非空并建立的 PDB 内联合身份记录总数。
        - unique_object_key_count: int；身份记录中不同 LigandObject 模板键数量。
        - same_pdb_object_key_multi_prepared_smiles_group_count: int；同一 PDB、同一 object_key 拆成两个或更多 prepared_smiles 身份的分组数。
        - matched_foreground_without_prepared_smiles_candidate_count: int；前景命中但 matched_prepared_smiles 为 null 的候选数，合并显式 null 与整个 prepared 文件缺失。
        - matched_foreground_explicit_null_candidate_count: int；前景命中且 PreparedLigand 顶层 smiles 显式为 null 的候选数。
        - matched_foreground_missing_prepared_file_candidate_count: int；前景命中但当前 PDB 的 prepared_smiles.jsonl 整体不存在的候选数。
        - language_available_identity_count: dict[str, int]；每个语言家族至少有一个有效归档的身份数。
        - language_available_identity_count.molformer: int；MoLFormer 可用身份数。
        - language_available_identity_count.smi_ted_289m: int；SMI-TED 可用身份数。
        - language_multi_model_smiles_identity_count: dict[str, int]；有效归档中出现多个精确 model_smiles 的身份数。
        - language_multi_model_smiles_identity_count.molformer: int；MoLFormer tokenizer 输入存在多字符串诊断的身份数。
        - language_multi_model_smiles_identity_count.smi_ted_289m: int；SMI-TED tokenizer 输入存在多字符串诊断的身份数。
        - seen_positive_identity_count: dict[str, int]；按语言家族统计 seen_in_train_by_language 为 true 的身份数。
        - seen_positive_identity_count.molformer: int；MoLFormer 口径训练可见身份数。
        - seen_positive_identity_count.smi_ted_289m: int；SMI-TED 口径训练可见身份数。
    """

    # dict[str, Any], 最终写入 manifest.summary 的嵌套计数对象。
    result: dict[str, Any] = {"splits": {}}
    for split_name in SPLIT_NAMES:
        if split_name not in split_candidates:
            continue
        # list[dict[str, Any]], 当前划分全部候选记录。
        candidates = split_candidates[split_name]
        # list[dict[str, Any]], 当前划分全部联合身份记录。
        identities = split_identities[split_name]
        # dict[str, int], 每个语言家族至少一个有效归档的身份数。
        language_available = {
            family: sum(identity["language"][family]["available"] for identity in identities)
            for family in LANGUAGE_FAMILIES
        }
        # dict[str, int], 每个语言家族有效归档中存在多个 model_smiles 的身份数。
        language_model_smiles_conflicts = {
            family: sum(
                len(identity["language"][family]["distinct_model_smiles"]) > 1
                for identity in identities
            )
            for family in LANGUAGE_FAMILIES
        }
        # dict[str, int], 每个语言家族在 train 可监督正目标中出现过的身份数。
        seen_positive_identities = {
            family: sum(
                identity["seen_in_train_by_language"][family]
                for identity in identities
            )
            for family in LANGUAGE_FAMILIES
        }
        result["splits"][split_name] = {
            "candidate_count": len(candidates),
            "class_counts": dict(sorted(Counter(item["class_name"] for item in candidates).items())),
            "identity_count": len(identities),
            "unique_object_key_count": len({item["object_key"] for item in identities}),
            "same_pdb_object_key_multi_prepared_smiles_group_count": sum(
                count > 1
                for count in Counter(
                    (item["pdb_id"], item["object_key"]) for item in identities
                ).values()
            ),
            "matched_foreground_without_prepared_smiles_candidate_count": sum(
                item["class_name"] in FOREGROUND_CLASS_NAMES
                and item["matched_prepared_smiles"] is None
                for item in candidates
            ),
            "matched_foreground_explicit_null_candidate_count": sum(
                item["matched_prepared_smiles_status"] == "explicit_null"
                for item in candidates
            ),
            "matched_foreground_missing_prepared_file_candidate_count": sum(
                item["matched_prepared_smiles_status"] == "missing_file"
                for item in candidates
            ),
            "language_available_identity_count": language_available,
            "language_multi_model_smiles_identity_count": language_model_smiles_conflicts,
            "seen_positive_identity_count": seen_positive_identities,
        }
    result["excluded_pdb_reason_counts"] = dict(
        sorted(Counter(item["reason"] for item in exclusions).items())
    )
    return result


# NOTE: 我认为在未来的新版正式代码中， _validate_这种验证函数最好不要在主线中出现————你觉得呢？
def _validate_existing_manifest(
    path: Path,
    roots: SourceRoots,
    coverage_threshold: float,
    selection_seed: int,
) -> dict[str, Any]:
    """
    读取既有 manifest.json，并拒绝复用来源根、schema 或监督参数不一致的清单。

    输入参数:
        - path: Path；既有 manifest.json 路径。
        - roots: SourceRoots；当前请求的 data_root、stage1_root、output_root 及 schema 版本。
        - roots.data_root: Path；必须与 manifest.data_root 的路径文本相同。
        - roots.stage1_root: Path；必须与 manifest.stage1_root 的路径文本相同。
        - roots.output_root: Path；必须与 manifest.output_root 的路径文本相同。
        - coverage_threshold: float；当前请求的候选双向覆盖阈值。
        - selection_seed: int；当前请求的语言表示选择种子。

    必须精确匹配的 manifest 字段:
        - schema_version: int；必须等于 MANIFEST_SCHEMA_VERSION。
        - data_root: str；必须等于 str(roots.data_root)。
        - stage1_root: str；必须等于 str(roots.stage1_root)。
        - output_root: str；必须等于 str(roots.output_root)。
        - coverage_threshold: float；必须等于调用参数。
        - selection_seed: int；必须等于调用参数。
        - identity_key_fields: list[str]；必须等于 object_key、prepared_smiles 的正式顺序。
        - identity_class_rule: str；必须等于当前 IDENTITY_CLASS_RULE。
        - split_pdb_rule: str；必须等于当前 SPLIT_PDB_RULE。
        - class_names: list[str]；必须等于当前六分类顺序。

    划分字段:
        - splits: dict[str, dict[str, str]]；必须是对象，键只能属于 SPLIT_NAMES，每个值必须是文件映射对象。
        - splits.{split_name}.candidates: str；当前划分候选 JSONL 文件名；本函数不检查该子字段类型或文件存在性，由后续消费者读取时失败关闭。
        - splits.{split_name}.identities: str；当前划分身份 JSONL 文件名；本函数不检查该子字段类型或文件存在性，由后续消费者读取时失败关闭。

    返回值:
        - manifest: dict[str, Any]；完整既有主清单对象。
        - manifest.schema_version: int；清单 schema 版本，必须等于 MANIFEST_SCHEMA_VERSION。
        - manifest.created_at_utc: str；首次建立当前 schema 清单的 UTC ISO 8601 时间。
        - manifest.updated_at_utc: str；最近一次发布的 UTC ISO 8601 时间。
        - manifest.data_root: str；冻结 A–G 数据根路径文本。
        - manifest.stage1_root: str；Stage1 F1 候选根路径文本。
        - manifest.output_root: str；Matcher 输出根路径文本。
        - manifest.coverage_threshold: float；候选与 occurrence 的双向覆盖率下限。
        - manifest.selection_seed: int；身份内语言归档的确定性选择种子。
        - manifest.identity_key_fields: list[str]；固定顺序 object_key、prepared_smiles。
        - manifest.prepared_smiles_rule: str；顶层字符串、显式 null 和缺失文件的身份规则说明。
        - manifest.identity_class_rule: str；PDB 内联合身份类别唯一规则。
        - manifest.split_pdb_rule: str；候选 PDB 的划分互斥规则。
        - manifest.cross_pdb_joint_identity_class_variants: list[dict[str, Any]]；跨 PDB 同联合键多类别诊断。
        - manifest.cross_pdb_joint_identity_class_variants[*].object_key: str；跨 PDB 比较的模板图键。
        - manifest.cross_pdb_joint_identity_class_variants[*].prepared_smiles: str；跨 PDB 比较的顶层公开接口字符串。
        - manifest.cross_pdb_joint_identity_class_variants[*].classes: list[dict[str, Any]]；观察到的不同类别。
        - manifest.cross_pdb_joint_identity_class_variants[*].classes[*].class_name: str；五类前景类别名。
        - manifest.cross_pdb_joint_identity_class_variants[*].classes[*].class_index: int；与类别名对齐的六分类索引。
        - manifest.cross_pdb_joint_identity_class_variants[*].locations: list[dict[str, Any]]；联合键的划分/PDB 位置。
        - manifest.cross_pdb_joint_identity_class_variants[*].locations[*].split: str；身份记录所在划分。
        - manifest.cross_pdb_joint_identity_class_variants[*].locations[*].pdb_id: str；身份记录所在 PDB。
        - manifest.cross_pdb_joint_identity_class_variants[*].locations[*].class_name: str；该 PDB 内类别名。
        - manifest.cross_pdb_joint_identity_class_variants[*].locations[*].class_index: int；该 PDB 内类别索引。
        - manifest.cross_pdb_joint_identity_class_variants[*].locations[*].occurrence_count: int；共享该 PDB 内身份的 occurrence 数。
        - manifest.language_selection_rule: str；有效 occurrence 排序、稳定盐值和循环选择规则说明。
        - manifest.seen_identity_rule: str；逐语言家族按 train 正目标判断 seen 的规则说明。
        - manifest.class_names: list[str]；固定六分类顺序。
        - manifest.available_splits: list[str]；已经物化的划分，按 SPLIT_NAMES 排序。
        - manifest.deferred_splits: list[str]；尚未物化的划分，按 SPLIT_NAMES 排序。
        - manifest.splits: dict[str, dict[str, str]]；已物化划分到 JSONL 文件名的映射。
        - manifest.splits.{split_name}.candidates: str；当前划分候选 JSONL 文件名。
        - manifest.splits.{split_name}.identities: str；当前划分身份 JSONL 文件名。
        - manifest.excluded_pdb: str；整 PDB 排除 JSONL 文件名。
        - manifest.split_pdb_counts: dict[str, dict[str, int]]；已物化划分的目录计数。
        - manifest.split_pdb_counts.{split_name}.discovered: int；发现的 PDB 目录数。
        - manifest.split_pdb_counts.{split_name}.included: int；成功扫描并发布候选的 PDB 数。
        - manifest.split_pdb_counts.{split_name}.excluded: int；因允许原因整 PDB 排除的目录数。
        - manifest.summary: dict[str, Any]；候选、类别、身份、语言和排除原因汇总。
        - manifest.summary.splits: dict[str, dict[str, Any]]；已物化划分到汇总计数的映射。
        - manifest.summary.splits.{split_name}.candidate_count: int；候选记录总数。
        - manifest.summary.splits.{split_name}.class_counts: dict[str, int]；实际出现的候选类别计数。
        - manifest.summary.splits.{split_name}.class_counts.{class_name}: int；当前实际出现类别的候选数。
        - manifest.summary.splits.{split_name}.identity_count: int；联合身份记录数。
        - manifest.summary.splits.{split_name}.unique_object_key_count: int；不同模板键数量。
        - manifest.summary.splits.{split_name}.same_pdb_object_key_multi_prepared_smiles_group_count: int；同 PDB、同模板键的多字符串分组数。
        - manifest.summary.splits.{split_name}.matched_foreground_without_prepared_smiles_candidate_count: int；前景命中但顶层字符串不可用的候选数。
        - manifest.summary.splits.{split_name}.matched_foreground_explicit_null_candidate_count: int；前景命中且顶层字符串显式为 null 的候选数。
        - manifest.summary.splits.{split_name}.matched_foreground_missing_prepared_file_candidate_count: int；前景命中但整个 PreparedLigand 文件缺失的候选数。
        - manifest.summary.splits.{split_name}.language_available_identity_count: dict[str, int]；逐语言家族可用身份数。
        - manifest.summary.splits.{split_name}.language_available_identity_count.molformer: int；MoLFormer 可用身份数。
        - manifest.summary.splits.{split_name}.language_available_identity_count.smi_ted_289m: int；SMI-TED 可用身份数。
        - manifest.summary.splits.{split_name}.language_multi_model_smiles_identity_count: dict[str, int]；逐语言家族多 tokenizer 字符串身份数。
        - manifest.summary.splits.{split_name}.language_multi_model_smiles_identity_count.molformer: int；MoLFormer 多字符串身份数。
        - manifest.summary.splits.{split_name}.language_multi_model_smiles_identity_count.smi_ted_289m: int；SMI-TED 多字符串身份数。
        - manifest.summary.splits.{split_name}.seen_positive_identity_count: dict[str, int]；逐语言家族训练可见身份数。
        - manifest.summary.splits.{split_name}.seen_positive_identity_count.molformer: int；MoLFormer 口径 seen 身份数。
        - manifest.summary.splits.{split_name}.seen_positive_identity_count.smi_ted_289m: int；SMI-TED 口径 seen 身份数。
        - manifest.summary.excluded_pdb_reason_counts: dict[str, int]；实际出现的整 PDB 排除原因计数。
        - manifest.summary.excluded_pdb_reason_counts.blob_exceed: int；条件字段，blob_exceed PDB 数。
        - manifest.summary.excluded_pdb_reason_counts.missing_required_product: int；条件字段，缺必要产物 PDB 数。
    """

    # dict[str, Any], 从 UTF-8 JSON 读取的既有主清单顶层对象。
    manifest = json.loads(path.read_text(encoding="utf-8"))
    # dict[str, Any], 决定能否安全复用既有候选和身份记录的当前期望字段。
    expected = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "data_root": str(roots.data_root),
        "stage1_root": str(roots.stage1_root),
        "output_root": str(roots.output_root),
        "coverage_threshold": coverage_threshold,
        "selection_seed": selection_seed,
        "identity_key_fields": list(IDENTITY_KEY_FIELDS),
        "identity_class_rule": IDENTITY_CLASS_RULE,
        "split_pdb_rule": SPLIT_PDB_RULE,
        "class_names": list(CLASS_NAMES),
    }
    # dict[str, dict[str, Any]], 实际值与期望值不同的复用关键字段。
    mismatches = {
        key: {"actual": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"既有清单与当前请求不一致：{mismatches}")
    # Any, 已物化划分到候选/身份文件名的映射，随后核验顶层结构与划分键。
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not all(
        split_name in SPLIT_NAMES and isinstance(files, dict)
        for split_name, files in splits.items()
    ):
        raise ValueError(f"{path}:splits 必须只含受支持划分的文件映射。")
    return manifest


# NOTE: 我认为在未来的新版正式代码中， _validate_这种验证函数最好不要在主线中出现————你觉得呢？
def _validate_split_names(split_names: tuple[str, ...]) -> None:
    """
    要求显式划分序列非空、不重复且只含受支持的三个名称。

    输入参数:
        - split_names: tuple[str, ...]；调用者允许读取或要求存在的划分序列，可为 SPLIT_NAMES 的任意非空不重复子序列。

    本函数不重排输入；原始顺序会影响首次扫描顺序，但正式发布文件顺序由 SPLIT_NAMES 固定。
    """

    if (
        not split_names
        or len(set(split_names)) != len(split_names)
        or not set(split_names).issubset(SPLIT_NAMES)
    ):
        raise ValueError(
            f"split_names 必须是非空、不重复的 {SPLIT_NAMES} 子序列。"
        )


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """
    把一个 JSON 对象格式化为稳定 UTF-8 文本并交给原子文本写入器。

    输入参数:
        - path: Path；最终 JSON 文件路径。
        - value: dict[str, Any]；待序列化顶层对象，键排序、中文不转义、缩进为两个空格；该通用写入边界不规定固定业务键，全部嵌套值由 json.dumps 原样编码。

    序列化文本固定以一个换行结束；先在目标目录创建临时文件，写入并刷新到磁盘后再以原子替换发布到目标路径。成功时完成原子替换；写入、刷盘或替换异常直接传播，临时文件可能保留。
    """

    # str, 键排序且以单个换行结尾的 UTF-8 JSON 文本。
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, text)


def _atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """
    把 JSON 对象序列编码为每个对象一行的稳定 UTF-8 JSONL，并原子发布。

    输入参数:
        - path: Path；最终 JSONL 文件路径。
        - records: Iterable[dict[str, Any]]；按迭代顺序写出的顶层对象，每个对象键排序且中文不转义。
        - records[*]: dict[str, Any]；该通用写入边界不规定固定业务键，调用者提供的每个键和值均交给 json.dumps 原样编码。

    空序列写出空文件；非空序列的每条记录后均有一个换行。函数先在目标目录创建临时文件，写入并刷新到磁盘后再以原子替换发布到目标路径。成功时完成原子替换；写入、刷盘或替换异常直接传播，临时文件可能保留。
    """

    # str, 保持记录迭代顺序、每个对象占一个物理行的完整 JSONL 文本。
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    _atomic_write_text(path, text)


def _atomic_write_text(path: Path, text: str) -> None:
    """
    使用目标同目录的进程专属临时文件刷盘并原子替换 UTF-8 文本。

    输入参数:
        - path: Path；最终正式文件路径，父目录不存在时递归创建。
        - text: str；完整待发布文本，以 UTF-8 和显式 LF 换行写入。

    副作用:
        - temporary: Path；目标同目录下带当前进程号的临时文件，成功时由 os.replace 移动为 path；异常直接传播且没有清理处理器，因此该文件可能保留。
        - path: Path；原子替换后的最终文本文件，已有文件会被完整替换。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # Path, 与目标同目录且带当前进程号，支持同文件系统原子发布。
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    """
    解析正式清单构建命令行参数。

    输入参数:
        - argv: list[str]；不含程序名的命令行 token 序列。

    返回字段:
        - data_root: Path；冻结 A–G 数据根，命令行必须显式提供。
        - stage1_root: Path；Stage1 F1 候选根，命令行必须显式提供。
        - output_root: Path；Matcher 输出根，命令行必须显式提供。
        - coverage_threshold: float；候选双向覆盖阈值，命令行必须显式提供，范围由 build_manifest 核验。
        - selection_seed: int；语言表示选择种子，命令行必须显式提供。
        - splits: list[str]；一个或多个显式划分，只能取 train、validation、calibration；首次建立是否包含 train 由 ensure_manifest 核验。

    ArgumentParser 的 description 取自本模块 Docstring，因此学习注释修订允许改变 --help 的说明文字；参数名、类型、必填性、choices、解析和退出语义保持不变。
    """

    # ArgumentParser, 正式清单 CLI；模块 Docstring 仅用于帮助说明。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--coverage-threshold", type=float, required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLIT_NAMES,
        required=True,
        help="显式列出本次允许读取并物化的划分；不得预先加入 calibration。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    构建或核验正式清单，并把最终 manifest.json 路径写到标准输出。

    输入参数:
        - argv: list[str] 或 None；显式列表用于测试或嵌入调用，None 时读取 sys.argv[1:]。

    返回值:
        - exit_code: int；ensure_manifest 成功并打印路径后固定为 0；参数或数据错误通过异常或 argparse 退出传播。
    """

    # argparse.Namespace, 已按 CLI 类型转换并完成必填/choices 检查的参数对象。
    arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
    # SourceRoots, 由三个命令行路径构造的冻结来源与版本化输出路径集合。
    roots = SourceRoots(arguments.data_root, arguments.stage1_root, arguments.output_root)
    # Path, 构建、扩展或复用后包含全部显式划分的主清单路径。
    path = ensure_manifest(
        roots,
        arguments.coverage_threshold,
        arguments.selection_seed,
        tuple(arguments.splits),
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
