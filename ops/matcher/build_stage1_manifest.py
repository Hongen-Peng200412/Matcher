"""扫描一个 Find_1 划分并原子发布 Matcher Stage1 manifest.

正式入口 build_stage1_manifest 直接从完整 F2 blob、centered 候选、真实 occurrence、
PreparedLigand 顶层 SMILES、SMI-TED 向量和全局 SMILES 图包建立候选与身份记录. 它不读取
历史 F2_centered_label.npz, 也不生成新的标签旁路文件.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from matcher.data.labels import CandidateTarget, OccurrenceMask, assign_candidate_target
from matcher.data.schema import (
    CLASS_NAMES,
    LANGUAGE_EMBEDDING_DIM,
    MANIFEST_SCHEMA,
    SMI_TED_MODEL_NAME,
    SourceRoots,
)
from matcher.data.smiles_graph import SmilesGraphStore


# float; 完整 test_0 的 240 个重复 SMILES 组最大观测差为 5.9604645e-6.
LANGUAGE_DUPLICATE_ABSOLUTE_TOLERANCE = 1e-5
COVERAGE_THRESHOLD = 0.30


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取一个 UTF-8 JSONL 文件.

    - path: 输入路径.
    - return: 按物理行顺序保存的 JSON 对象; 空白行忽略.
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


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    """在目标目录内原子发布 JSONL.

    - path: 最终路径.
    - records: 按冻结顺序写出的对象列表.
    - return: 成功替换目标文件后返回 None.
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """在目标目录内原子发布缩进 JSON.

    - path: 最终路径.
    - value: 可由标准 JSON 编码器直接写出的对象.
    - return: 成功替换目标文件后返回 None.
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _read_occurrences(
    roots: SourceRoots,
    pdb_id: str,
) -> tuple[dict[int, str], dict[int, str | None]]:
    """按 occurrence_id 对齐真实类别与顶层精确 SMILES.

    - roots: AdaLigand 数据根、Find_1 根和 Matcher 输出根.
    - pdb_id: 当前小写 PDB 标识.
    - return[0]: occurrence_id 到 type_tag 的映射.
    - return[1]: occurrence_id 到顶层 PreparedLigand.smiles 的映射; 缺失时为 None.
"""

    occurrence_path = roots.data_root / "parse" / pdb_id / "occurrences.jsonl"
    prepared_path = roots.prepared_root / pdb_id / "prepared_smiles.jsonl"
    occurrences = {
        int(record["candidate_id"]): str(record["type_tag"])
        for record in _read_jsonl(occurrence_path)
    }
    prepared = {
        int(record["candidate_id"]): record
        for record in _read_jsonl(prepared_path)
    } if prepared_path.is_file() else {}
    unknown = sorted(set(prepared) - set(occurrences))
    if unknown:
        raise ValueError(f"{pdb_id}: PreparedLigand 含未知 occurrence_id {unknown}。")
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


def _read_occurrence_masks(
    data_root: Path,
    pdb_id: str,
    type_by_id: dict[int, str],
) -> list[OccurrenceMask]:
    """读取一个 PDB 的真实 occurrence 稀疏体素掩码.

    - data_root: AdaLigand Ori_Data 根目录.
    - pdb_id: 当前 PDB 标识.
    - type_by_id: occurrence_id 到 type_tag 的映射.
    - return: 按 occurrence_id 递增的 OccurrenceMask 列表.
"""

    path = data_root / "density" / pdb_id / "ligand_area.npz"
    masks: list[OccurrenceMask] = []
    with np.load(path, allow_pickle=False) as archive:
        for occurrence_id in sorted(type_by_id):
            name = f"mask_{occurrence_id}"
            if name not in archive.files:
                raise ValueError(f"{path} 缺少 {name}。")
            masks.append(
                OccurrenceMask(
                    occurrence_id=occurrence_id,
                    type_tag=type_by_id[occurrence_id],
                    voxel_index_zyx=np.asarray(archive[name], dtype=np.int64),
                )
            )
    return masks


def _read_language_embedding(
    path: Path,
    pdb_id: str,
    occurrence_id: int,
    smiles: str,
) -> np.ndarray | None:
    """读取并核对一个 occurrence 的固定 SMI-TED 向量.

    - path: candidate_{occurrence_id}.npz 路径.
    - pdb_id: 预期 PDB 标识.
    - occurrence_id: 预期 occurrence 编号.
    - smiles: 预期精确顶层 SMILES.
    - return: 有限 float32 (768,) 向量; 文件不存在时为 None.
"""

    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["pdb_id"].item()) != pdb_id:
            raise ValueError(f"{path}: pdb_id 不一致。")
        if int(archive["candidate_id"].item()) != occurrence_id:
            raise ValueError(f"{path}: occurrence_id 不一致。")
        if str(archive["prepared_smiles"].item()) != smiles:
            raise ValueError(f"{path}: prepared_smiles 不一致。")
        if str(archive["model_name"].item()) != SMI_TED_MODEL_NAME:
            raise ValueError(f"{path}: model_name 不是固定 SMI-TED 模型。")
        embedding = np.asarray(archive["embedding"], dtype=np.float32)
    if embedding.shape != (LANGUAGE_EMBEDDING_DIM,) or not np.isfinite(embedding).all():
        raise ValueError(f"{path}: embedding 必须是有限 float32 (768,)。")
    return embedding


def _build_identities(
    roots: SourceRoots,
    pdb_id: str,
    type_by_id: dict[int, str],
    smiles_by_id: dict[int, str | None],
    graph_store: SmilesGraphStore,
) -> list[dict[str, Any]]:
    """构造一个 PDB 内可预测的精确 SMILES 身份轴.

    - roots: 正式数据根集合.
    - pdb_id: 当前 PDB 标识.
    - type_by_id: occurrence 类别映射; other 不形成身份槽.
    - smiles_by_id: occurrence 顶层 SMILES 映射.
    - graph_store: 已初始化的全局图包.
    - return: 按精确 SMILES 字典序排列的身份记录.
"""

    occurrence_ids_by_smiles: dict[str, list[int]] = defaultdict(list)
    for occurrence_id in sorted(type_by_id):
        smiles = smiles_by_id[occurrence_id]
        if type_by_id[occurrence_id] != "other" and smiles is not None and smiles in graph_store:
            occurrence_ids_by_smiles[smiles].append(occurrence_id)
    records: list[dict[str, Any]] = []
    for smiles in sorted(occurrence_ids_by_smiles):
        reference: np.ndarray | None = None
        selected_path: Path | None = None
        for occurrence_id in occurrence_ids_by_smiles[smiles]:
            path = roots.language_root / pdb_id / f"candidate_{occurrence_id}.npz"
            embedding = _read_language_embedding(path, pdb_id, occurrence_id, smiles)
            if embedding is None:
                continue
            if reference is not None and not np.allclose(
                reference,
                embedding,
                rtol=0.0,
                atol=LANGUAGE_DUPLICATE_ABSOLUTE_TOLERANCE,
            ):
                difference = float(np.max(np.abs(reference - embedding)))
                raise ValueError(f"{pdb_id}: {smiles!r} 的重复 SMI-TED 向量最大差 {difference:.9g}。")
            if reference is None:
                reference = embedding
                selected_path = path
        if selected_path is not None:
            records.append(
                {
                    "pdb_id": pdb_id,
                    "smiles": smiles,
                    "language_relative_path": str(selected_path.relative_to(roots.data_root)),
                }
            )
    return records


def _read_candidate_ranking(
    pdb_root: Path,
    source_blob_index: np.ndarray,
    centered: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray]:
    """按 centered 候选轴返回冻结 Gaussian 分数和选择掩码.

    - pdb_root: 当前 Stage1 PDB 目录.
    - source_blob_index: int32 (N_candidate,) centered 到 F2 blob 的连接轴.
    - centered: 已打开的 F2_centered.npz.
    - return[0]: float32 (N_candidate,) Gaussian 分数.
    - return[1]: bool (N_candidate,) candidate_selected 掩码.
"""

    if "score" in centered.files and "selected" in centered.files:
        return (
            np.asarray(centered["score"], dtype=np.float32),
            np.asarray(centered["selected"], dtype=np.bool_),
        )
    path = pdb_root / "evaluation/f2_centered_gaussian_macro_selected.npz"
    with np.load(path, allow_pickle=False) as archive:
        ranked_blob = np.asarray(archive["source_blob_index"], dtype=np.int64)
        ranked_score = np.asarray(archive["candidate_score"], dtype=np.float32)
        ranked_selected = np.asarray(archive["candidate_selected"], dtype=np.bool_)
    if len(set(ranked_blob.tolist())) != len(ranked_blob):
        raise ValueError(f"{path}: source_blob_index 存在重复值。")
    position = {int(blob): index for index, blob in enumerate(ranked_blob)}
    try:
        order = np.asarray([position[int(blob)] for blob in source_blob_index], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"{path}: 缺少 centered 候选的 source_blob_index。") from error
    return ranked_score[order], ranked_selected[order]


def _read_candidate_voxels(
    blob_path: Path,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """读取完整 F2 blob 的全局稀疏体素和原生概率均值.

    - blob_path: blobs/F2_blobs.npz 路径.
    - return[0]: source_blob_index 到 int64 (N_voxel, 3) 全局 Z/Y/X 索引的映射.
    - return[1]: source_blob_index 到 blob 内来源概率均值的映射.
"""

    with np.load(blob_path, allow_pickle=False) as archive:
        blob_index = np.asarray(archive["blob_index"], dtype=np.int64)
        offsets = np.asarray(archive["voxel_offsets"], dtype=np.int64)
        voxels = np.asarray(archive["voxel_index_global_zyx"], dtype=np.int64)
        probability_mean = np.asarray(archive["source_probability_mean"], dtype=np.float32)
    if offsets.shape != (len(blob_index) + 1,) or probability_mean.shape != blob_index.shape:
        raise ValueError(f"{blob_path}: 候选轴、offsets 或概率均值未对齐。")
    return (
        {
            int(blob): voxels[int(offsets[index]):int(offsets[index + 1])]
            for index, blob in enumerate(blob_index)
        },
        {int(blob): float(probability_mean[index]) for index, blob in enumerate(blob_index)},
    )


def _build_split(
    roots: SourceRoots,
    split: str,
    graph_store: SmilesGraphStore,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """扫描一个 Stage1 划分并构造全部候选与可预测身份记录.

    - roots: 正式数据根集合.
    - split: stage1_root 下的划分目录名.
    - graph_store: 全局 SMILES 图包.
    - return[0]: list[dict]. 按 PDB 和 centered 轴排序的全部候选记录.
    - return[0][].split: str. 当前 Stage1 划分名称.
    - return[0][].pdb_id: str. 小写 PDB 标识.
    - return[0][].centered_box_index: int. F2_centered.npz 中的稳定候选编号.
    - return[0][].source_blob_index: int. F2_blobs.npz 中的源 blob 编号.
    - return[0][].candidate_selected: bool. Stage1 冻结选择掩码.
    - return[0][].gaussian_score: float. Gaussian 排序分数.
    - return[0][].source_probability_mean: float. 源 blob 内原生 Stage1 概率均值.
    - return[0][].foreground_target: bool. 双向覆盖率均不低于 0.30 时为 True.
    - return[0][].target_category: str 或 None. 最大覆盖 occurrence 的五类标签, 背景为 None.
    - return[0][].matched_occurrence_id: int 或 None. 最大几何平均覆盖 occurrence, 并列取较小编号.
    - return[0][].target_smiles: str 或 None. 目标 occurrence 的精确顶层 PreparedLigand.smiles.
    - return[0][].candidate_coverage: float. 交集体素数除以候选体素数.
    - return[0][].occurrence_coverage: float. 交集体素数除以 occurrence 体素数.
    - return[0][].overlap_score: float. 两项覆盖率的几何平均.
    - return[1]: list[dict]. 按 PDB 和精确 SMILES 排序的受支持身份记录.
    - return[1][].pdb_id: str. 身份所属小写 PDB 标识.
    - return[1][].smiles: str. 同时具有严格图和语言资产的精确顶层 SMILES.
    - return[1][].language_relative_path: str. 相对 data_root 的代表 occurrence SMI-TED NPZ 路径.
    - return[2]: dict. 当前划分的 PDB、候选、选择、前景、支持状态和身份计数.
    - return[2].pdb_ids: list[str]. 即使没有 centered 候选也保留的已发现 PDB 标识顺序.
    - return[2].pdb_count: int. pdb_ids 长度.
    - return[2].candidate_count: int. 全部候选数.
    - return[2].selected_candidate_count: int. candidate_selected=True 的候选数.
    - return[2].foreground_candidate_count: int. 几何真前景候选数.
    - return[2].supported_foreground_candidate_count: int. 具有受支持目标身份的真前景数.
    - return[2].unsupported_foreground_candidate_count: int. 缺少严格图或语言资产的真前景数.
    - return[2].train_validation_candidate_count: int. 训练/验证保留的全部候选数.
    - return[2].selected_train_validation_candidate_count: int. selected_only 训练/验证保留数.
    - return[2].identity_count: int. 按 PDB 计数的身份槽总数.
    - return[2].unique_smiles_count: int. 跨 PDB 去重的精确 SMILES 数.
"""

    split_root = roots.stage1_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(split_root)
    candidates: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    supported_foreground = 0
    selected_supported_foreground = 0
    pdb_roots = sorted(
        path for path in split_root.iterdir()
        if path.is_dir() and (path / "centered/F2_centered.npz").is_file()
    )
    for pdb_root in pdb_roots:
        pdb_id = pdb_root.name.lower()
        type_by_id, smiles_by_id = _read_occurrences(roots, pdb_id)
        occurrences = _read_occurrence_masks(roots.data_root, pdb_id, type_by_id)
        pdb_identities = _build_identities(roots, pdb_id, type_by_id, smiles_by_id, graph_store)
        supported_smiles = {record["smiles"] for record in pdb_identities}
        candidate_voxels, probability_mean_by_blob = _read_candidate_voxels(
            pdb_root / "blobs/F2_blobs.npz"
        )
        with np.load(pdb_root / "centered/F2_centered.npz", allow_pickle=False) as centered:
            centered_box_index = np.asarray(centered["centered_box_index"], dtype=np.int64)
            source_blob_index = np.asarray(centered["source_blob_index"], dtype=np.int64)
            if centered_box_index.shape != source_blob_index.shape:
                raise ValueError(f"{pdb_id}: centered 两个编号轴未对齐。")
            gaussian_score, selected = _read_candidate_ranking(
                pdb_root,
                source_blob_index,
                centered,
            )
        if gaussian_score.shape != centered_box_index.shape or selected.shape != centered_box_index.shape:
            raise ValueError(f"{pdb_id}: Gaussian 分数或选择掩码未与 centered 候选轴对齐。")
        for position, blob_index in enumerate(source_blob_index):
            try:
                voxels = candidate_voxels[int(blob_index)]
                probability_mean = probability_mean_by_blob[int(blob_index)]
            except KeyError as error:
                raise ValueError(f"{pdb_id}: centered 候选无法连接 F2 blob {int(blob_index)}。") from error
            target: CandidateTarget = assign_candidate_target(
                voxels,
                occurrences,
                COVERAGE_THRESHOLD,
            )
            target_smiles = (
                smiles_by_id[target.matched_occurrence_id]
                if target.matched_occurrence_id is not None
                else None
            )
            if target.foreground_target and target_smiles in supported_smiles:
                supported_foreground += 1
                selected_supported_foreground += int(bool(selected[position]))
            candidates.append(
                {
                    "split": split,
                    "pdb_id": pdb_id,
                    "centered_box_index": int(centered_box_index[position]),
                    "source_blob_index": int(blob_index),
                    "candidate_selected": bool(selected[position]),
                    "gaussian_score": float(gaussian_score[position]),
                    "source_probability_mean": probability_mean,
                    "foreground_target": target.foreground_target,
                    "target_category": target.target_category,
                    "matched_occurrence_id": target.matched_occurrence_id,
                    "target_smiles": target_smiles,
                    "candidate_coverage": target.candidate_coverage,
                    "occurrence_coverage": target.occurrence_coverage,
                    "overlap_score": target.overlap_score,
                }
            )
        identities.extend(pdb_identities)
    foreground_count = sum(record["foreground_target"] for record in candidates)
    selected_count = sum(record["candidate_selected"] for record in candidates)
    selected_foreground_count = sum(
        record["candidate_selected"] and record["foreground_target"]
        for record in candidates
    )
    unsupported_foreground_count = foreground_count - supported_foreground
    selected_unsupported_foreground_count = (
        selected_foreground_count - selected_supported_foreground
    )
    summary = {
        "pdb_ids": [path.name.lower() for path in pdb_roots],
        "pdb_count": len(pdb_roots),
        "candidate_count": len(candidates),
        "selected_candidate_count": selected_count,
        "foreground_candidate_count": foreground_count,
        "supported_foreground_candidate_count": supported_foreground,
        "unsupported_foreground_candidate_count": unsupported_foreground_count,
        "train_validation_candidate_count": len(candidates) - unsupported_foreground_count,
        "selected_train_validation_candidate_count": (
            selected_count - selected_unsupported_foreground_count
        ),
        "identity_count": len(identities),
        "unique_smiles_count": len({record["smiles"] for record in identities}),
    }
    return candidates, identities, summary


# ================================================================================================


def build_stage1_manifest(
    stage1_root: Path,
    split: str,
    output_path: Path,
    *,
    data_root: Path,
    graph_asset_path: Path,
) -> Path:
    """扫描一个 Stage1 划分并更新同一版本化 manifest 入口.

    - stage1_root: Find_1 artifacts/Find_1 根目录; 其下包含 split 目录.
    - split: 本次扫描并发布的划分名称.
    - output_path: 最终 manifest.json 路径.
    - data_root: AdaLigand Ori_Data 根目录.
    - graph_asset_path: 全局 smiles_graphs_v1.npz 路径.
    - return: 原子发布后的 output_path.

    落盘文件:
    - manifest.json: 唯一可变入口. 顶层保存 schema、class_names、identity_key、language_model、data_root、stage1_root、graph_asset_path、splits 和 summary.
    - <generation>_candidates.jsonl: 不可变候选快照. 每行字段与 _build_split return[0] 相同.
    - <generation>_identities.jsonl: 不可变身份快照. 每行字段与 _build_split return[1] 相同.

    manifest 嵌套字段:
    - splits[split].candidates: str. 相对 manifest 父目录的候选 JSONL 生成文件名.
    - splits[split].identities: str. 相对 manifest 父目录的身份 JSONL 生成文件名.
    - summary[split]: dict. 字段与 _build_split return[2] 相同.

    每次调用只扫描一个 split. output_path 已存在时只允许来源根和固定契约完全相同. 两份
    JSONL 先写入新的唯一生成文件, 最后只原子替换 manifest 入口. 中途失败最多留下未引用文件,
    不能改变旧入口指向的快照. 其它已发布 split 保持不变.
"""

    if not split or Path(split).name != split or split in {".", ".."}:
        raise ValueError("split 必须是非空的单级目录名称。")
    output_path = Path(output_path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "class_names": list(CLASS_NAMES),
        "identity_key": "PreparedLigand.smiles",
        "language_model": SMI_TED_MODEL_NAME,
        "data_root": str(Path(data_root)),
        "stage1_root": str(Path(stage1_root)),
        "graph_asset_path": str(Path(graph_asset_path)),
        "splits": {},
        "summary": {},
    }
    if output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        fixed_fields = set(manifest) - {"splits", "summary"}
        if any(previous.get(field) != manifest[field] for field in fixed_fields):
            raise ValueError("现有 manifest 的固定来源或科学契约与本次构建不一致。")
        if not isinstance(previous.get("splits"), dict) or not isinstance(
            previous.get("summary"), dict
        ):
            raise ValueError("现有 manifest 的 splits 或 summary 不是映射。")
        manifest["splits"] = dict(previous["splits"])
        manifest["summary"] = dict(previous["summary"])

    roots = SourceRoots(Path(data_root), Path(stage1_root))
    graph_store = SmilesGraphStore(Path(graph_asset_path))
    graph_store.initialize()
    candidates, identities, summary = _build_split(roots, split, graph_store)
    # 唯一生成名使旧入口永远指向不可变文件; manifest 的最后一次原子替换是唯一发布点.
    generation = f"{split}_{time.time_ns()}_{os.getpid()}"
    candidate_name = f"{generation}_candidates.jsonl"
    identity_name = f"{generation}_identities.jsonl"
    _write_jsonl_atomic(output_path.parent / candidate_name, candidates)
    _write_jsonl_atomic(output_path.parent / identity_name, identities)
    manifest["splits"][split] = {
        "candidates": candidate_name,
        "identities": identity_name,
    }
    manifest["summary"][split] = summary
    _write_json_atomic(output_path, manifest)
    return output_path


def main(argv: list[str] | None = None) -> int:
    """解析一个 Stage1 来源、划分和输出 manifest 路径.

    - argv: 可选命令行参数; None 表示读取当前进程命令行.
    - return: manifest 成功发布后返回 0.
"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--graph-asset-path", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(
        build_stage1_manifest(
            arguments.stage1_root,
            arguments.split,
            arguments.output_path,
            data_root=arguments.data_root,
            graph_asset_path=arguments.graph_asset_path,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
