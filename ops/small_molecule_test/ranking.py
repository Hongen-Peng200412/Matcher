"""应用 test_smallmolecule 的候选顺序与尝试次数规则.

主要入口是 build_topk_curve. 候选排序直接复用现有 test_0 的 selected 优先实现,
本模块只增加"非 Stage3 前景免费跳过"的尝试状态机, 不计算覆盖关系或模型输出.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from matcher.evaluation.test_set import _selected_priority_order


TOP_K_VALUES = tuple(range(1, 21))


def build_topk_curve(
    name: str,
    source: Path,
    pdb_ids: list[str],
    candidates: list[dict[str, Any]],
    sorting: str,
    candidate_success: Callable[[dict[str, Any]], bool],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """汇总固定 77-PDB 分母的尝试次数 top-k 曲线与完整候选轨迹.

    输入参数:
        - name: str, 图例与结果表使用的完整曲线名称.
        - source: Path, 当前曲线读取的冻结候选文件或模型结果路径.
        - pdb_ids: list[str], 固定分母中的有序 PDB 标识, 如 11jb.
        - candidates: list[dict[str, Any]], 一种受体条件或 Emap2lig 的全部冻结候选.
          - pdb_id: str, 候选所属 PDB.
          - source_blob_index: int, 同一来源 PDB 内的完整 blob 编号.
          - centered_box_index: int|None, Find_1 centered 候选编号; Emap2lig 为 None.
          - candidate_selected: bool, 决定候选进入排序前段还是追加段.
          - gaussian_score: float|None, Find_1 Gaussian 段内分数.
          - source_probability_mean: float, blob 内来源概率均值.
          - target_scope: str, background、non_stage3_foreground 或 stage3_foreground.
          - matched_occurrence_id: int|None, 双向覆盖门槛下的唯一目标 occurrence.
          - target_smiles: str|None, 唯一目标的精确顶层 SMILES.
        - sorting: str, gaussian_score 或 source_probability_mean, 只改变 selected 与 unselected 两段内部的顺序.
        - candidate_success: Callable, 接收一个候选; 仅 stage3_foreground 候选可能返回 True.

    返回值:
        - curve: dict[str, Any], 一条固定分母的累计命中曲线.
          - name: str, 曲线完整名称.
          - source: str, 冻结候选文件或模型结果路径.
          - sorting: str, 当前段内排序字段.
          - pdb_count: int, 固定分母数量.
          - pdb_ids: list[str], 分母中的有序 PDB 标识.
          - top_k: list[int], 固定为 1 至 20.
          - hit_count: list[int], 与 top_k 逐项对齐的命中数.
          - hit_ratio: list[float], 与 top_k 逐项对齐的固定分母命中比例.
          - pdb_hit: dict[str, list[bool]], 每个 PDB 在 20 个 k 上的累计命中状态.
          - candidate_order_sha256: str, 全部 PDB 候选顺序的内容摘要.
        - traces: dict[str, list[dict[str, Any]]], PDB 到完整候选轨迹的映射.
          - source_blob_index: int, 当前来源 PDB 内完整 blob 编号.
          - centered_box_index: int|None, Find_1 centered 候选编号.
          - raw_rank: int, 完整冻结顺序中的一基位置.
          - attempt_index: int|None, 计费尝试编号; 免费跳过项为 None.
          - action: str, 免费跳过、消耗背景或消耗 Stage3 目标.
          - target_scope: str, 候选三态适用范围.
          - matched_occurrence_id: int|None, 唯一目标 occurrence 编号.
          - target_smiles: str|None, 唯一目标精确 SMILES.
          - success: bool, 当前 Stage1 或 Stage1+Stage2 候选是否成功.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["pdb_id"])].append(candidate)
    if set(grouped) - set(pdb_ids):
        raise ValueError("候选含固定 77-PDB 分母之外的 PDB.")

    pdb_hit: dict[str, list[bool]] = {}
    traces: dict[str, list[dict[str, Any]]] = {}
    frozen_order: list[tuple[str, int]] = []
    for pdb_id in pdb_ids:
        records = grouped.get(pdb_id, [])
        score = np.asarray([float(record[sorting]) for record in records], dtype=np.float64)
        source_blob_index = np.asarray(
            [int(record["source_blob_index"]) for record in records], dtype=np.int64
        )
        selected = np.asarray(
            [bool(record["candidate_selected"]) for record in records], dtype=np.bool_
        )
        # int64 数组, (N_candidate,); selected 段完整排在 unselected 段之前.
        order = _selected_priority_order(score, source_blob_index, selected)
        spent = 0
        first_hit_attempt: int | None = None
        pdb_trace: list[dict[str, Any]] = []
        for raw_rank, position in enumerate(order.tolist(), start=1):
            record = records[position]
            scope = str(record["target_scope"])
            frozen_order.append((pdb_id, int(record["source_blob_index"])))
            if scope == "non_stage3_foreground":
                # 唯一目标不属于 PocketXMol 测试范围, 既不计费也不调用 Matcher.
                action = "skip_non_stage3_foreground"
                attempt_index = None
                success = False
            else:
                # 背景与 Stage3 目标都消耗一次机会; 只有后者可以形成命中.
                spent += 1
                attempt_index = spent
                action = "spend_background" if scope == "background" else "spend_stage3_target"
                success = scope == "stage3_foreground" and bool(candidate_success(record))
                if success and first_hit_attempt is None:
                    first_hit_attempt = spent
            pdb_trace.append(
                {
                    "source_blob_index": int(record["source_blob_index"]),
                    "centered_box_index": record["centered_box_index"],
                    "raw_rank": raw_rank,
                    "attempt_index": attempt_index,
                    "action": action,
                    "target_scope": scope,
                    "matched_occurrence_id": record["matched_occurrence_id"],
                    "target_smiles": record["target_smiles"],
                    "success": success,
                }
            )
        pdb_hit[pdb_id] = [
            first_hit_attempt is not None and first_hit_attempt <= top_k
            for top_k in TOP_K_VALUES
        ]
        traces[pdb_id] = pdb_trace

    counts = [
        sum(int(pdb_hit[pdb_id][index]) for pdb_id in pdb_ids)
        for index in range(len(TOP_K_VALUES))
    ]
    order_payload = json.dumps(frozen_order, ensure_ascii=True, separators=(",", ":"))
    curve = {
        "name": name,
        "source": str(source),
        "sorting": sorting,
        "pdb_count": len(pdb_ids),
        "pdb_ids": pdb_ids,
        "top_k": list(TOP_K_VALUES),
        "hit_count": counts,
        "hit_ratio": [count / len(pdb_ids) for count in counts],
        "pdb_hit": pdb_hit,
        "candidate_order_sha256": hashlib.sha256(order_payload.encode("ascii")).hexdigest(),
    }
    return curve, traces
