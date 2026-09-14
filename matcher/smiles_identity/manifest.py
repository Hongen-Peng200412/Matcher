"""构建并验证 Matcher 的单一精确 SMILES 身份清单. 

正式命令示例:
    python -m matcher.smiles_identity.manifest --data-root /storage/penghongen/AdaLigand/Ori_Data --stage1-root /storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1 --output-root /storage/penghongen/tmp/Matcher --split validation

主要入口 build_manifest 在 output_root/manifests/smiles_identity 下发布 manifest.json、
<split>_candidates.jsonl 和 <split>_identities.jsonl. 候选身份只由 PreparedLigand 顶层
smiles 定义. 清单不保存 object_key、seen/unseen、语言家族、model_smiles 或重新计算的裁剪位置. 
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .schema import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    LANGUAGE_EMBEDDING_DIM,
    MANIFEST_SCHEMA,
    SMI_TED_MODEL_NAME,
    SourceRoots,
)
from .graph import SmilesGraphStore


# 同一精确 SMILES 的历史 occurrence 向量来自相同公开模型输入, 但批次推理留下了
# 最高 3.58e-6 的 float32 数值抖动. 清单固定选择最小可用 occurrence_id 的归档; 
# 若任一重复向量超过该绝对误差, 则视为语言产物内容冲突并停止发布. 
LANGUAGE_DUPLICATE_ABSOLUTE_TOLERANCE = 5e-6


def load_manifest(path: Path) -> dict[str, Any]:
    """读取入口 manifest.json 并验证新管线消费的全部字段. 

    - path: manifest.json 路径. 
    - return: dict[str, Any], 通过 build_manifest 字段契约检查后的 manifest.json 顶层映射; splits 内 JSONL 路径仍保持相对形式. 
    """

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_fields = {
        "schema",
        "class_names",
        "identity_key",
        "language_model",
        "data_root",
        "stage1_root",
        "graph_asset_path",
        "splits",
        "summary",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_fields:
        raise ValueError(f"manifest 顶层字段必须精确为 {sorted(expected_fields)}。")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest.schema 必须是 {MANIFEST_SCHEMA!r}。")
    if manifest.get("class_names") != list(CLASS_NAMES):
        raise ValueError("manifest.class_names 与固定六分类顺序不一致。")
    if manifest.get("identity_key") != "PreparedLigand.smiles":
        raise ValueError("manifest.identity_key 必须是 PreparedLigand.smiles。")
    if manifest.get("language_model") != SMI_TED_MODEL_NAME:
        raise ValueError("manifest.language_model 与固定 SMI-TED 模型不一致。")
    for field in ("data_root", "stage1_root", "graph_asset_path"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"manifest.{field} 必须是非空路径字符串。")
    if not isinstance(manifest.get("splits"), dict) or not manifest["splits"]:
        raise ValueError("manifest.splits 必须是非空映射。")
    return manifest


def read_split_records(path: Path, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """读取一个划分的候选与身份 JSONL, 并检查记录不含旧身份字段. 

    - path: manifest.json 路径. 
    - split: manifest.splits 中的划分名称. 
    - return[0]: list[dict[str, Any]], 按候选 JSONL 文件顺序读取的记录; 每项字段由 _build_split 的 return[0] 契约定义. 
    - return[1]: list[dict[str, Any]], 按身份 JSONL 文件顺序读取的记录; 每项字段由 _build_split 的 return[1] 契约定义. 
    """

    manifest = load_manifest(path)
    try:
        files = manifest["splits"][split]
    except KeyError as error:
        raise KeyError(f"manifest 不含划分 {split!r}。") from error
    root = Path(path).parent
    expected_fields = {"candidates", "identities"}
    if not isinstance(files, dict) or set(files) != expected_fields:
        raise ValueError(f"manifest.splits.{split} 字段必须精确为 {sorted(expected_fields)}。")
    candidate_path = root / files["candidates"]
    identity_path = root / files["identities"]
    candidates = _read_jsonl(candidate_path)
    identities = _read_jsonl(identity_path)
    forbidden = {"object_key", "model_smiles", "language_family", "seen_in_train"}
    for record in candidates + identities:
        overlap = forbidden.intersection(record)
        if overlap:
            raise ValueError(f"新清单记录含废止字段: {sorted(overlap)}")
    return candidates, identities


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL, 并要求每个非空行解析为字符串键映射. 

    - path: 输入 JSONL 路径. 
    - return: list[dict[str, Any]], 按文件顺序保存的 JSON 映射列表; 空白行忽略. 
    """

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON 对象。")
            records.append(record)
    return records


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    """以 UTF-8 JSONL 原子发布一组记录. 

    - path: 最终 JSONL 路径. 
    - records: list[dict[str, Any]], 按发布顺序写出的 JSON 映射列表; 映射键按字典序编码. 
    - return: None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """以缩进 UTF-8 JSON 原子发布一个顶层映射. 

    - path: 最终 JSON 路径. 
    - value: dict[str, Any], 可由标准 JSON 编码器直接序列化的顶层映射. 
    - return: None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _read_language_embedding(
    path: Path,
    pdb_id: str,
    occurrence_id: int,
    smiles: str,
) -> np.ndarray | None:
    """读取并验证一个 occurrence 的固定 SMI-TED 向量; 缺文件时返回 None. 

    - path: candidate_{occurrence_id}.npz 的完整路径. 
    - pdb_id: str, 当前语言向量预期所属的小写 PDB 标识.
    - occurrence_id: int, 当前语言向量预期所属的 occurrence 编号.
    - smiles: str, 当前语言向量预期所属的精确顶层 PreparedLigand.smiles.
    - return: float32 数组|None, (768,); 有限 SMI-TED 向量, 文件不存在时为 None. 

    归档字段:
        - pdb_id: 字符串标量, 必须等于参数 pdb_id. 
        - candidate_id: 整数标量, 在本模块中解释为 occurrence_id, 必须等于参数 occurrence_id. 
        - prepared_smiles: 字符串标量, 必须逐字符等于参数 smiles. 
        - model_name: 字符串标量, 必须等于固定 SMI_TED_MODEL_NAME. 
        - embedding: float32 数组, (768,); 必须全部为有限值. 

    归档必须逐字段匹配 PDB、occurrence、精确 prepared_smiles 和固定模型名称; 本函数
    不读取 model_smiles, 也不重新运行语言模型. 
    """

    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["pdb_id"].item()) != pdb_id:
            raise ValueError(f"{path}: pdb_id 不一致。")
        if int(archive["candidate_id"].item()) != occurrence_id:
            raise ValueError(f"{path}: occurrence_id 不一致。")
        if str(archive["prepared_smiles"].item()) != smiles:
            raise ValueError(f"{path}: prepared_smiles 与顶层字符串不一致。")
        if str(archive["model_name"].item()) != SMI_TED_MODEL_NAME:
            raise ValueError(f"{path}: model_name 不是固定 SMI-TED 模型。")
        embedding = np.asarray(archive["embedding"], dtype=np.float32)
    if embedding.shape != (LANGUAGE_EMBEDDING_DIM,) or not np.isfinite(embedding).all():
        raise ValueError(f"{path}: embedding 必须是有限 float32 ({LANGUAGE_EMBEDDING_DIM},)。")
    return embedding


def _read_occurrences_and_smiles(
    roots: SourceRoots,
    pdb_id: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, str | None]]:
    """按 occurrence_id 对齐 occurrence 类型与顶层精确 SMILES. 

    - roots: 原始数据、Find_1 和 Matcher 输出根目录. 
    - pdb_id: 当前蛋白质结构编号. 
    - return[0]: dict[int, dict[str, Any]], occurrence_id 到 occurrences.jsonl 原始类型记录的完整映射. 
    - return[1]: dict[int, str|None], occurrence_id 到精确顶层 SMILES 的映射; 缺少 PreparedLigand 记录或顶层 smiles 时为 None. 

    PreparedLigand 可以不包含无法准备的 occurrence, 但不得凭空增加未知编号. 
    缺失前景是否跳过由候选清单连接统一处理; 背景不依赖 SMILES. 
    """

    occurrence_path = roots.data_root / "parse" / pdb_id / "occurrences.jsonl"
    prepared_path = roots.prepared_root / pdb_id / "prepared_smiles.jsonl"
    occurrences = {int(record["candidate_id"]): record for record in _read_jsonl(occurrence_path)}
    prepared = (
        {int(record["candidate_id"]): record for record in _read_jsonl(prepared_path)}
        if prepared_path.is_file()
        else {}
    )
    unknown_prepared_ids = sorted(set(prepared) - set(occurrences))
    if unknown_prepared_ids:
        raise ValueError(f"{pdb_id}: PreparedLigand 含未知 occurrence_id {unknown_prepared_ids}。")
    smiles_by_id: dict[int, str | None] = {}
    for occurrence_id in occurrences:
        record = prepared.get(occurrence_id)
        if record is None:
            smiles_by_id[occurrence_id] = None
            continue
        if str(record["pdb_id"]) != pdb_id:
            raise ValueError(f"{prepared_path}: pdb_id 不一致。")
        smiles = record.get("smiles")
        smiles_by_id[occurrence_id] = smiles if isinstance(smiles, str) and smiles else None
    return occurrences, smiles_by_id


def _build_pdb_identities(
    roots: SourceRoots,
    pdb_id: str,
    occurrence_by_id: dict[int, dict[str, Any]],
    smiles_by_id: dict[int, str | None],
    graph_store: SmilesGraphStore,
) -> list[dict[str, Any]]:
    """把同一 PDB 的有效前景 occurrence 按精确 SMILES 合并成身份槽. 

    - roots: 原始数据、Find_1 和 Matcher 输出根目录. 
    - pdb_id: 当前蛋白质结构编号. 
    - occurrence_by_id: occurrence_id 到类别记录的映射. 
    - smiles_by_id: occurrence_id 到精确顶层 SMILES 的映射; 缺失项为 None. 
    - graph_store: 已完整载入内存的全局 SMILES 图资产. 
    - return: list[dict[str, Any]], 按类别和精确 SMILES 排序的身份记录列表; 每项字段与 _build_split 的 return[1][*] 相同. 

    occurrence_ids 保存同一身份下全部图合法成员. 函数读取该 SMILES 的全部可用 occurrence
    SMI-TED 归档并审计向量一致性, 再发布最小可用 occurrence_id 的归档路径; 整组没有语言
    文件时不发布该身份. 相同公开模型输入的历史 float32 向量允许已审计的微小数值抖动;
    超过固定绝对误差时失败关闭. 
    """

    # dict[str, list[int]], 精确 SMILES 到同一 PDB 内全部有效 occurrence_id 的递增列表.
    grouped: dict[str, list[int]] = defaultdict(list)
    # dict[str, int], 精确 SMILES 到唯一前景类别下标的映射.
    class_by_smiles: dict[str, int] = {}
    # dict[str, np.ndarray], 精确 SMILES 到首个可用 SMI-TED float32 (768,) 向量的映射; 仅用于重复向量审计.
    embedding_by_smiles: dict[str, np.ndarray] = {}
    # dict[str, str], 精确 SMILES 到最小可用 occurrence 语言归档相对路径的映射.
    selected_path_by_smiles: dict[str, str] = {}
    for occurrence_id in sorted(occurrence_by_id):
        type_tag = str(occurrence_by_id[occurrence_id]["type_tag"])
        if type_tag == "other":
            continue
        if type_tag not in CLASS_TO_INDEX or CLASS_TO_INDEX[type_tag] == 0:
            raise ValueError(f"{pdb_id}: 未知前景类型 {type_tag!r}。")
        smiles = smiles_by_id[occurrence_id]
        if smiles is None or smiles not in graph_store:
            continue
        class_index = CLASS_TO_INDEX[type_tag]
        if smiles in class_by_smiles and class_by_smiles[smiles] != class_index:
            raise ValueError(f"{pdb_id}: 同一精确 SMILES 对应多个前景类别。")
        class_by_smiles[smiles] = class_index
        grouped[smiles].append(occurrence_id)
        language_path = roots.language_root / pdb_id / f"candidate_{occurrence_id}.npz"
        embedding = _read_language_embedding(language_path, pdb_id, occurrence_id, smiles)
        if embedding is None:
            continue
        if smiles in embedding_by_smiles and not np.allclose(
            embedding_by_smiles[smiles],
            embedding,
            rtol=0.0,
            atol=LANGUAGE_DUPLICATE_ABSOLUTE_TOLERANCE,
        ):
            maximum_difference = float(np.max(np.abs(embedding_by_smiles[smiles] - embedding)))
            raise ValueError(
                f"{pdb_id}: 同一精确 SMILES 的 SMI-TED 向量最大绝对差 "
                f"{maximum_difference:.9g} 超过 "
                f"{LANGUAGE_DUPLICATE_ABSOLUTE_TOLERANCE:.9g}。"
            )
        embedding_by_smiles.setdefault(smiles, embedding)
        selected_path_by_smiles.setdefault(smiles, str(language_path.relative_to(roots.data_root)))
    return [
        {
            "pdb_id": pdb_id,
            "smiles": smiles,
            "class_index": class_by_smiles[smiles],
            "occurrence_ids": grouped[smiles],
            "language_relative_path": selected_path_by_smiles[smiles],
        }
        for smiles in sorted(selected_path_by_smiles, key=lambda value: (class_by_smiles[value], value))
    ]


def _check_label_alignment(
    labels: np.lib.npyio.NpzFile,
    centered_box_index: np.ndarray,
    source_blob_index: np.ndarray,
) -> None:
    """核验标签旁路文件与 centered 候选轴逐项一致. 

    - labels: 已打开的 F2_centered_label.npz. 
    - centered_box_index: centered 产物的候选编号轴. 
    - source_blob_index: centered 产物的源 blob 编号轴. 
    - return: None; 全部标签字段存在、同形且两个连接轴逐项相等时通过. 
    """

    required = {
        "centered_box_index",
        "source_blob_index",
        "class_index",
        "matched_occurrence_id",
        "candidate_coverage",
        "occurrence_coverage",
        "overlap_score",
    }
    if required.difference(labels.files):
        raise ValueError(f"标签文件缺少字段: {sorted(required.difference(labels.files))}")
    if not np.array_equal(labels["centered_box_index"], centered_box_index):
        raise ValueError("标签 centered_box_index 与 centered 产物不一致。")
    if not np.array_equal(labels["source_blob_index"], source_blob_index):
        raise ValueError("标签 source_blob_index 与 centered 产物不一致。")
    if any(np.asarray(labels[name]).shape != centered_box_index.shape for name in required):
        raise ValueError("标签字段必须共享 centered 候选轴。")


def _build_split(
    roots: SourceRoots,
    split: str,
    graph_store: SmilesGraphStore,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """连接一个划分的 centered、标签、PreparedLigand 和语言向量. 

    - roots: 原始数据、Find_1 和 Matcher 输出根目录. 
    - split: stage1_root 下的划分目录名. 
    - graph_store: 已加载的全局图包; 同时充当合法精确 SMILES 集合. 
    - return[0]: list[dict[str, Any]], 候选记录列表; 背景全部保留, 缺图或缺语言的前景整条排除. 
        - return[0][*].split: str, 当前 Find_1 划分名称. 
        - return[0][*].pdb_id: str, 候选所属 PDB 目录名. 
        - return[0][*].centered_box_index: int, F2_centered.npz 中候选的稳定编号. 
        - return[0][*].source_blob_index: int, 候选连接的 F2_blobs.npz blob_index. 
        - return[0][*].stage1_selected: bool, Find_1 是否把候选选为 Stage1 输出. 
        - return[0][*].class_index: int, 六分类下标. 
        - return[0][*].matched_occurrence_id: int|None, 双向覆盖最佳 occurrence 编号; 标签值小于 0 时为 None. 
        - return[0][*].matched_smiles: str|None, 前景候选的精确顶层 PreparedLigand.smiles; 背景为 None. 
        - return[0][*].candidate_coverage: float, 交集唯一体素数除以候选唯一体素数. 
        - return[0][*].occurrence_coverage: float, 交集唯一体素数除以 occurrence 唯一体素数. 
        - return[0][*].overlap_score: float, 两个方向覆盖率乘积的平方根. 
    - return[1]: list[dict[str, Any]], PDB 内身份记录列表; 每项由精确 SMILES 唯一定义. 
        - return[1][*].pdb_id: str, 身份槽所属 PDB. 
        - return[1][*].smiles: str, 身份槽的精确顶层 PreparedLigand.smiles. 
        - return[1][*].class_index: int, 该身份槽唯一的前景类别下标. 
        - return[1][*].occurrence_ids: list[int], 当前 PDB 中共享该精确 SMILES 的全部有效 occurrence 编号, 按整数递增. 
        - return[1][*].language_relative_path: str, 相对 data_root 的 SMI-TED NPZ 路径, 固定选择最小可用 occurrence_id. 
    - return[2]: dict[str, int], 当前划分计数摘要. 
        - return[2].pdb_count: int, 至少保留一个候选的 PDB 数. 
        - return[2].candidate_count: int, 保留的候选总数. 
        - return[2].foreground_candidate_count: int, 保留且 class_index 大于 0 的候选数. 
        - return[2].identity_count: int, 全部 PDB 内身份槽总数. 
        - return[2].unique_smiles_count: int, 跨 PDB 去重后的精确 SMILES 数. 
        - return[2].skipped_positive_candidate_count: int, 因图或语言资产不可用而整条排除的前景候选数. 
    """

    split_root = roots.stage1_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(split_root)
    # list[dict[str, Any]], 按 PDB 目录名和 centered 候选轴连接的候选记录.
    candidates: list[dict[str, Any]] = []
    # list[dict[str, Any]], 按 PDB 目录名以及 PDB 内类别、SMILES 顺序连接的身份记录.
    identities: list[dict[str, Any]] = []
    # int, 当前划分因身份图或语言向量不可用而排除的前景候选累计数.
    skipped_positive = 0
    # PDB 目录名排序固定跨文件系统的清单发布顺序. 
    for pdb_root in sorted(path for path in split_root.iterdir() if path.is_dir()):
        pdb_id = pdb_root.name
        occurrence_by_id, smiles_by_id = _read_occurrences_and_smiles(roots, pdb_id)
        pdb_identities = _build_pdb_identities(
            roots,
            pdb_id,
            occurrence_by_id,
            smiles_by_id,
            graph_store,
        )
        # 精确字符串到当前 PDB 有效身份记录的映射; 用于前景候选完整性门禁. 
        identity_by_smiles = {record["smiles"]: record for record in pdb_identities}
        with np.load(pdb_root / "centered/F2_centered.npz", allow_pickle=False) as centered:
            with np.load(pdb_root / "centered/F2_centered_label.npz", allow_pickle=False) as labels:
                # int32 数组, (N_candidate,); 当前 PDB centered 候选的稳定编号轴.
                centered_box_index = np.asarray(centered["centered_box_index"], dtype=np.int32)
                # int32 数组, (N_candidate,); 与 centered_box_index 逐项对齐的源 blob_index.
                source_blob_index = np.asarray(centered["source_blob_index"], dtype=np.int32)
                # bool 数组, (N_candidate,); 与 centered_box_index 逐项对齐, True 表示 Find_1 选中该候选.
                selected = np.asarray(centered["selected"], dtype=np.bool_)
                _check_label_alignment(labels, centered_box_index, source_blob_index)
                for index in range(len(centered_box_index)):
                    class_index = int(labels["class_index"][index])
                    occurrence_id = int(labels["matched_occurrence_id"][index])
                    matched_smiles: str | None = None
                    if class_index > 0:
                        matched_smiles = smiles_by_id.get(occurrence_id)
                        if matched_smiles not in identity_by_smiles:
                            skipped_positive += 1
                            continue
                        if identity_by_smiles[matched_smiles]["class_index"] != class_index:
                            raise ValueError(f"{pdb_id}: 候选类别与 SMILES 身份类别不一致。")
                    candidates.append(
                        {
                            "split": split,
                            "pdb_id": pdb_id,
                            "centered_box_index": int(centered_box_index[index]),
                            "source_blob_index": int(source_blob_index[index]),
                            "stage1_selected": bool(selected[index]),
                            "class_index": class_index,
                            "matched_occurrence_id": None if occurrence_id < 0 else occurrence_id,
                            "matched_smiles": matched_smiles,
                            "candidate_coverage": float(labels["candidate_coverage"][index]),
                            "occurrence_coverage": float(labels["occurrence_coverage"][index]),
                            "overlap_score": float(labels["overlap_score"][index]),
                        }
                    )
        identities.extend(pdb_identities)
    # dict[str, int], 当前划分写入 manifest.summary[split] 的六项计数.
    summary = {
        "pdb_count": len({record["pdb_id"] for record in candidates}),
        "candidate_count": len(candidates),
        "foreground_candidate_count": sum(record["class_index"] > 0 for record in candidates),
        "identity_count": len(identities),
        "unique_smiles_count": len({record["smiles"] for record in identities}),
        "skipped_positive_candidate_count": skipped_positive,
    }
    return candidates, identities, summary


# ================================================================================================


def build_manifest(roots: SourceRoots, splits: list[str]) -> Path:
    """构建候选与身份 JSONL, 并最后原子发布入口 manifest.json. 

    - roots: 三个正式来源根目录. 
    - splits: 需要物化的 Find_1 划分名称; 调用方顺序会保存在入口文件. 
    - return: Path, 发布后的 output_root/manifests/smiles_identity/manifest.json 路径. 

    文件产物:
        - <split>_candidates.jsonl: JSONL, 每个元素是一个保留的 Find_1 centered 候选; 字段由 _build_split 的 return[0] 契约定义. 
        - <split>_identities.jsonl: JSONL, 每个元素是一个 PDB 内精确 SMILES 身份槽; 字段由 _build_split 的 return[1] 契约定义. 
        - manifest.json: JSON object, 绑定固定类别、身份语言契约、来源根目录、图包位置、各划分 JSONL 位置和计数. 

    manifest.json 字段:
        - schema: str, 固定为 smiles_identity_v1. 
        - class_names: list[str], 长度 6; 与 CLASS_NAMES 顺序逐项相同. 
        - identity_key: str, 固定为 PreparedLigand.smiles. 
        - language_model: str, 固定为 SMI-TED Light 289M. 
        - data_root: str, AdaLigand Ori_Data 根目录. 
        - stage1_root: str, Find_1 artifacts/Find_1 根目录. 
        - graph_asset_path: str, 全局 smiles_graphs_v1.npz 路径. 
        - splits: dict[str, dict[str, str]], 划分名称到两个 JSONL 文件字段的映射. 
            - splits[split].candidates: str, 当前划分候选 JSONL 相对 manifest.json 所在目录的文件名. 
            - splits[split].identities: str, 当前划分身份 JSONL 相对 manifest.json 所在目录的文件名. 
        - summary: dict[str, dict[str, int]], 每个划分的 PDB、候选、前景、身份、唯一 SMILES 和排除计数; 字段由 _build_split 的 return[2] 契约定义. 

    每个正类候选只有在精确 SMILES 同时拥有合法图和有效 SMI-TED 向量时才写入; 
    缺少任一项时整条候选跳过. 背景候选不需要 SMILES, 因此始终保留. 
    """

    if not splits or any(not split for split in splits) or len(set(splits)) != len(splits):
        raise ValueError("splits 必须是非空且不重复的名称列表。")
    graph_store = SmilesGraphStore(roots.graph_asset_path)
    graph_store.initialize()
    roots.manifest_root.mkdir(parents=True, exist_ok=True)
    # dict[str, dict[str, str]], 划分名称到候选与身份 JSONL 相对文件名的映射.
    split_files: dict[str, dict[str, str]] = {}
    # dict[str, dict[str, int]], 划分名称到六项保留与排除计数的映射.
    split_summary: dict[str, dict[str, int]] = {}
    for split in splits:
        candidates, identities, summary = _build_split(roots, split, graph_store)
        candidate_name = f"{split}_candidates.jsonl"
        identity_name = f"{split}_identities.jsonl"
        _write_jsonl_atomic(roots.manifest_root / candidate_name, candidates)
        _write_jsonl_atomic(roots.manifest_root / identity_name, identities)
        split_files[split] = {
            "candidates": candidate_name,
            "identities": identity_name,
        }
        split_summary[split] = summary
    # dict[str, Any], manifest.json 的完整顶层字段; JSONL 文件名保持相对 manifest 所在目录.
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "class_names": list(CLASS_NAMES),
        "identity_key": "PreparedLigand.smiles",
        "language_model": SMI_TED_MODEL_NAME,
        "data_root": str(roots.data_root),
        "stage1_root": str(roots.stage1_root),
        "graph_asset_path": str(roots.graph_asset_path),
        "splits": split_files,
        "summary": split_summary,
    }
    manifest_path = roots.manifest_root / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    """解析三个来源根目录和划分名称, 发布正式 manifest.json. 

    - argv: 可选命令行参数列表; None 表示读取当前进程命令行. 
    - return: int, 入口 manifest.json 成功原子发布后返回 0. 
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", action="append", required=True)
    arguments = parser.parse_args(argv)
    path = build_manifest(
        SourceRoots(arguments.data_root, arguments.stage1_root, arguments.output_root),
        arguments.split,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
