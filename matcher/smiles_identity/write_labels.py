"""为 Find_1 centered 候选生成紧邻的六分类与 occurrence 匹配标签. 

正式命令示例:
    python -m matcher.smiles_identity.write_labels --data-root /storage/penghongen/AdaLigand/Ori_Data --stage1-root /storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1 --split validation --mode skip --workers 16

主要入口 write_pdb_labels 为一个 PDB 写出
stage1_root/<split>/<pdb_id>/centered/F2_centered_label.npz. 标签文件不保存 SMILES、
模板键或体素副本, 只用 source_blob_index 连接原 centered 与 blobs 产物. 
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .labels import OccurrenceMask, assign_candidate_target


def _run_task(arguments: tuple[Path, Path, str, str, float, bool]) -> dict[str, int | str]:
    """把可序列化参数元组展开给进程池调用. 

    - arguments: tuple[Path, Path, str, str, float, bool], 依次为 data_root、stage1_root、split、pdb_id、coverage_threshold 和 overwrite. 
    - return: dict[str, int|str], write_pdb_labels 返回的 PDB 标签计数摘要. 
    """

    return write_pdb_labels(*arguments)


# ================================================================================================


def write_pdb_labels(
    data_root: Path,
    stage1_root: Path,
    split: str,
    pdb_id: str,
    coverage_threshold: float,
    overwrite: bool,
) -> dict[str, int | str]:
    """生成一个 PDB 的标签旁路文件, 并返回候选计数摘要. 

    - data_root: AdaLigand Ori_Data 根目录. 
    - stage1_root: Find_1 artifacts/Find_1 根目录. 
    - split: centered 产物划分名称. 
    - pdb_id: 小写 PDB 标识. 
    - coverage_threshold: 候选和 occurrence 共用的双向覆盖阈值. 
    - overwrite: True 覆盖现有完整标签; False 时直接跳过. 

    文件字段:
        - centered_box_index: int32 数组, (N_candidate,); 与 F2_centered.npz 同名字段逐项相同, 标识标签第一轴上的候选. 
        - source_blob_index: int32 数组, (N_candidate,); 与 F2_centered.npz 同名字段逐项相同, 每项指向 F2_blobs.npz 的 blob_index. 
        - class_index: int8 数组, (N_candidate,); 六分类下标, 0:6 依次为 background、ion、nucleotide_like、peptide_like、small_molecule、sugar. 
        - matched_occurrence_id: int32 数组, (N_candidate,); 最佳真实 occurrence 编号, 没有双向覆盖命中时为 -1. 
        - candidate_coverage: float32 数组, (N_candidate,); 交集唯一体素数除以候选唯一体素数. 
        - occurrence_coverage: float32 数组, (N_candidate,); 交集唯一体素数除以 occurrence 唯一体素数. 
        - overlap_score: float32 数组, (N_candidate,); 两个方向覆盖率乘积的平方根. 

    返回字段:
        - result.pdb_id: str, 当前 PDB 标识. 
        - result.status: str, written 表示本次原子发布文件, skipped 表示 overwrite=False 且文件已存在. 
        - result.candidate_count: int, written 时为 N_candidate, skipped 时为 0. 
        - result.foreground_count: int, written 时为 class_index 大于 0 的候选数, skipped 时为 0. 
    """

    pdb_root = stage1_root / split / pdb_id
    output_path = pdb_root / "centered/F2_centered_label.npz"
    if output_path.exists() and not overwrite:
        return {"pdb_id": pdb_id, "status": "skipped", "candidate_count": 0, "foreground_count": 0}
    with (data_root / "parse" / pdb_id / "occurrences.jsonl").open("r", encoding="utf-8") as handle:
        occurrence_records = [json.loads(line) for line in handle]
    with np.load(data_root / "density" / pdb_id / "ligand_area.npz", allow_pickle=False) as archive:
        occurrences = [
            OccurrenceMask(
                occurrence_id=int(record["candidate_id"]),
                type_tag=str(record["type_tag"]),
                voxel_index_zyx=np.asarray(archive[f"mask_{int(record['candidate_id'])}"], dtype=np.int64),
            )
            for record in occurrence_records
        ]
    with np.load(pdb_root / "blobs/F2_blobs.npz", allow_pickle=False) as archive:
        # int64 (N_blob,), 每个完整 F2 blob 的稳定编号.
        blob_index = np.asarray(archive["blob_index"], dtype=np.int64)
        # int64 (N_blob + 1,), offsets[i]:offsets[i + 1] 切分第 i 个 blob 的全局体素下标.
        offsets = np.asarray(archive["voxel_offsets"], dtype=np.int64)
        # int64 (N_blob_voxel_total, 3), 全部 blob 连接后的全局 Z/Y/X 体素下标.
        voxels = np.asarray(archive["voxel_index_global_zyx"], dtype=np.int64)
    with np.load(pdb_root / "centered/F2_centered.npz", allow_pickle=False) as archive:
        # int32 (N_candidate,), centered 候选的稳定编号轴.
        centered_box_index = np.asarray(archive["centered_box_index"], dtype=np.int32)
        # int32 (N_candidate,), 每个 centered 候选对应的 F2 blob_index.
        source_blob_index = np.asarray(archive["source_blob_index"], dtype=np.int32)
    # dict[int, int], F2 blob_index 到 blob 连接数组位置的唯一映射.
    position_by_blob = {int(value): index for index, value in enumerate(blob_index)}
    if len(position_by_blob) != len(blob_index):
        raise ValueError(f"{pdb_id}: F2_blobs.blob_index 存在重复值。")
    # list[CandidateTarget], 长度 N_candidate; 与 centered_box_index 和 source_blob_index 逐候选对齐.
    targets = []
    for source_index in source_blob_index:
        try:
            position = position_by_blob[int(source_index)]
        except KeyError as error:
            raise ValueError(f"{pdb_id}: centered source_blob_index={int(source_index)} 不在 F2_blobs 中。") from error
        targets.append(
            assign_candidate_target(
                voxels[offsets[position] : offsets[position + 1]],
                occurrences,
                coverage_threshold,
            )
        )
    # dict[str, np.ndarray], F2_centered_label.npz 的七个同轴正式字段.
    arrays = {
        "centered_box_index": centered_box_index,
        "source_blob_index": source_blob_index,
        "class_index": np.asarray([target.class_index for target in targets], dtype=np.int8),
        "matched_occurrence_id": np.asarray(
            [-1 if target.matched_occurrence_id is None else target.matched_occurrence_id for target in targets],
            dtype=np.int32,
        ),
        "candidate_coverage": np.asarray([target.candidate_coverage for target in targets], dtype=np.float32),
        "occurrence_coverage": np.asarray([target.occurrence_coverage for target in targets], dtype=np.float32),
        "overlap_score": np.asarray([target.overlap_score for target in targets], dtype=np.float32),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "pdb_id": pdb_id,
        "status": "written",
        "candidate_count": len(targets),
        "foreground_count": sum(target.class_index > 0 for target in targets),
    }


def run_label_build(
    data_root: Path,
    stage1_root: Path,
    split: str,
    coverage_threshold: float,
    overwrite: bool,
    workers: int,
) -> int:
    """并行处理一个划分的全部 PDB, 并打印汇总 JSON. 

    - data_root: Path, AdaLigand Ori_Data 根目录. 
    - stage1_root: Path, Find_1 artifacts/Find_1 根目录; 标签写入该目录对应 split 的 centered 子目录. 
    - split: str, 要处理的 centered 划分名称, 如 validation. 
    - coverage_threshold: float, 候选和 occurrence 共用的双向覆盖阈值. 
    - overwrite: bool, True 原子覆盖现有标签, False 保留现有标签并把对应 PDB 计为 skipped. 
    - workers: int, ProcessPoolExecutor 并行进程数; 必须为正整数. 
    - return: int, 全部 PDB 标签任务成功完成并输出汇总后返回 0. 

    标准输出 JSON:
        - split: str, 本次处理的 centered 划分名称. 
        - pdb_count: int, stage1_root/<split> 下参与进程池的 PDB 目录数. 
        - written_pdb_count: int, 本次实际发布标签文件的 PDB 数. 
        - candidate_count: int, 本次实际写入的候选总数; skipped PDB 贡献 0. 
        - foreground_count: int, 本次实际写入且 class_index 大于 0 的候选总数. 

    PDB 目录名先排序再交给进程池; 每个任务只写自己的 centered 旁路文件, 不共享
    输出目标. 任一子进程异常会由 executor.map 传播, 不能用较小汇总静默收口. 
    """

    if workers <= 0:
        raise ValueError("workers 必须为正整数。")
    pdb_ids = sorted(path.name for path in (stage1_root / split).iterdir() if path.is_dir())
    tasks = [
        (
            data_root,
            stage1_root,
            split,
            pdb_id,
            coverage_threshold,
            overwrite,
        )
        for pdb_id in pdb_ids
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_run_task, tasks))
    summary = {
        "split": split,
        "pdb_count": len(results),
        "written_pdb_count": sum(item["status"] == "written" for item in results),
        "candidate_count": sum(int(item["candidate_count"]) for item in results),
        "foreground_count": sum(int(item["foreground_count"]) for item in results),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    """解析命令行并把六个参数交给正式标签构建入口. 

    - argv: 可选命令行参数列表; None 表示读取当前进程命令行. 
    - return: int, run_label_build 返回的进程退出码. 
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--coverage-threshold", type=float, required=True)
    parser.add_argument("--mode", choices=("skip", "overwrite"), required=True)
    parser.add_argument("--workers", type=int, required=True)
    arguments = parser.parse_args(argv)
    return run_label_build(
        arguments.data_root,
        arguments.stage1_root,
        arguments.split,
        arguments.coverage_threshold,
        arguments.mode == "overwrite",
        arguments.workers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
