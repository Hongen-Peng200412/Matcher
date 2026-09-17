"""读取版本化 Stage1 manifest, 并核对运行时依赖的最小契约.

主要入口:
- load_manifest: 读取 manifest.json 顶层元数据.
- read_split_records: 读取一个划分的候选与 PDB 内 SMILES 身份记录.

目录扫描、几何目标计算和原子发布属于 ops/matcher/build_stage1_manifest.py. 本模块不扫描
Stage1 目录, 也不在训练时补建缺失记录.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .schema import CLASS_NAMES, MANIFEST_SCHEMA, SMI_TED_MODEL_NAME


MANIFEST_FIELDS = {
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

CANDIDATE_FIELDS = {
    "split",
    "pdb_id",
    "centered_box_index",
    "source_blob_index",
    "candidate_selected",
    "gaussian_score",
    "source_probability_mean",
    "foreground_target",
    "target_category",
    "matched_occurrence_id",
    "target_smiles",
    "candidate_coverage",
    "occurrence_coverage",
    "overlap_score",
}

IDENTITY_FIELDS = {
    "pdb_id",
    "smiles",
    "language_relative_path",
}

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """按文件顺序读取 UTF-8 JSONL.

    - path: JSONL 文件路径.
    - return: 对象列表; 空白行忽略, 非对象记录会立即报错.
"""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON 对象。")
            records.append(value)
    return records


def load_manifest(path: Path) -> dict[str, Any]:
    """读取 manifest.json 并核对稳定顶层字段.

    - path: manifest.json 路径.
    - return: 顶层映射; 划分文件名仍是相对 manifest 所在目录的路径.
"""

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
        raise ValueError(f"manifest 顶层字段必须精确为 {sorted(MANIFEST_FIELDS)}。")
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise ValueError(f"manifest.schema 必须是 {MANIFEST_SCHEMA!r}。")
    if manifest["class_names"] != list(CLASS_NAMES):
        raise ValueError("manifest.class_names 与固定报告类别顺序不一致。")
    if manifest["identity_key"] != "PreparedLigand.smiles":
        raise ValueError("manifest.identity_key 必须是 PreparedLigand.smiles。")
    if manifest["language_model"] != SMI_TED_MODEL_NAME:
        raise ValueError("manifest.language_model 与固定 SMI-TED 模型不一致。")
    for field in ("data_root", "stage1_root", "graph_asset_path"):
        if not isinstance(manifest[field], str) or not manifest[field]:
            raise ValueError(f"manifest.{field} 必须是非空路径字符串。")
    if not isinstance(manifest["splits"], dict) or not manifest["splits"]:
        raise ValueError("manifest.splits 必须是非空映射。")
    if not isinstance(manifest["summary"], dict):
        raise ValueError("manifest.summary 必须是映射。")
    return manifest


def _check_candidate(record: dict[str, Any], split: str, index: int) -> None:
    """核对一个候选记录的字段集合和关键身份状态.

    - record: 候选 JSON 对象.
    - split: 当前划分名称.
    - index: 候选 JSONL 中的零基位置, 只用于错误定位.
    - return: 字段与前景状态一致时返回 None.
"""

    prefix = f"{split}.candidates[{index}]"
    if set(record) != CANDIDATE_FIELDS:
        raise ValueError(f"{prefix} 字段必须精确为 {sorted(CANDIDATE_FIELDS)}。")
    if record["split"] != split:
        raise ValueError(f"{prefix}.split 与请求划分不一致。")
    if not isinstance(record["pdb_id"], str) or not record["pdb_id"]:
        raise ValueError(f"{prefix}.pdb_id 必须是非空字符串。")
    for field in ("centered_box_index", "source_blob_index"):
        if type(record[field]) is not int or record[field] < 0:
            raise ValueError(f"{prefix}.{field} 必须是非负整数。")
    if type(record["candidate_selected"]) is not bool:
        raise ValueError(f"{prefix}.candidate_selected 必须是布尔值。")
    for field in (
        "gaussian_score",
        "source_probability_mean",
        "candidate_coverage",
        "occurrence_coverage",
        "overlap_score",
    ):
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{prefix}.{field} 必须是有限数值。")
        if not math.isfinite(float(value)):
            raise ValueError(f"{prefix}.{field} 必须是有限数值。")
    for field in ("candidate_coverage", "occurrence_coverage", "overlap_score"):
        if not 0.0 <= float(record[field]) <= 1.0:
            raise ValueError(f"{prefix}.{field} 必须位于 [0,1]。")
    if type(record["foreground_target"]) is not bool:
        raise ValueError(f"{prefix}.foreground_target 必须是布尔值。")
    foreground = record["foreground_target"]
    required_foreground_values = (
        record["target_category"],
        record["matched_occurrence_id"],
    )
    if foreground and any(value is None for value in required_foreground_values):
        raise ValueError(f"{prefix} 的前景目标缺少类别或 occurrence。")
    if foreground and (
        type(record["matched_occurrence_id"]) is not int
        or record["matched_occurrence_id"] < 0
    ):
        raise ValueError(f"{prefix}.matched_occurrence_id 必须是非负整数。")
    if not foreground and any(
        value is not None
        for value in (*required_foreground_values, record["target_smiles"])
    ):
        raise ValueError(f"{prefix} 的背景目标不得保存类别、occurrence 或 SMILES。")
    if record["target_category"] is not None and record["target_category"] not in CLASS_NAMES[1:]:
        raise ValueError(f"{prefix}.target_category 不是已知前景类别。")
    target_smiles = record["target_smiles"]
    if target_smiles is not None and (
        not isinstance(target_smiles, str) or not target_smiles
    ):
        raise ValueError(f"{prefix}.target_smiles 必须是非空字符串或 null。")


def _check_identity(record: dict[str, Any], split: str, index: int) -> None:
    """核对一个 PDB 内精确 SMILES 身份记录.

    - record: 身份 JSON 对象.
    - split: 当前划分名称, 只用于错误定位.
    - index: 身份 JSONL 中的零基位置.
    - return: 三个字段均为非空字符串时返回 None.
"""

    prefix = f"{split}.identities[{index}]"
    if set(record) != IDENTITY_FIELDS:
        raise ValueError(f"{prefix} 字段必须精确为 {sorted(IDENTITY_FIELDS)}。")
    for field in IDENTITY_FIELDS:
        if not isinstance(record[field], str) or not record[field]:
            raise ValueError(f"{prefix}.{field} 必须是非空字符串。")


def read_split_records(
    path: Path,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """读取一个划分的全部候选和 PDB 内 SMILES 身份记录.

    - path: manifest.json 路径.
    - split: manifest.splits 中的划分名称.
    - return[0]: 候选记录, 顺序就是 Dataset 的冻结候选顺序.
    - return[1]: 身份记录; 同一 PDB 内由构建器按精确 SMILES 字典序发布.
"""

    manifest = load_manifest(path)
    if split not in manifest["splits"]:
        raise KeyError(f"manifest 不含划分 {split!r}。")
    files = manifest["splits"][split]
    if not isinstance(files, dict) or set(files) != {"candidates", "identities"}:
        raise ValueError(f"manifest.splits.{split} 必须只含 candidates 和 identities。")
    root = Path(path).parent
    candidates = _read_jsonl(root / files["candidates"])
    identities = _read_jsonl(root / files["identities"])
    for index, record in enumerate(candidates):
        _check_candidate(record, split, index)
    for index, record in enumerate(identities):
        _check_identity(record, split, index)
    duplicate_identities = len(identities) != len(
        {(record["pdb_id"], record["smiles"]) for record in identities}
    )
    if duplicate_identities:
        raise ValueError(f"manifest.splits.{split} 存在重复的 PDB 内 SMILES 身份。")
    return candidates, identities
