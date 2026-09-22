"""计算 Stage3 小分子测试集的尝试次数 top-k 曲线.

主要入口 build_topk_curve 先应用 selected 优先的 Stage1 候选顺序, 再执行
Stage3 专用计费规则. 函数同时返回累计命中曲线和逐 PDB 候选轨迹.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np

from matcher.evaluation.test_set import selected_priority_order


TOP_K_VALUES = tuple(range(1, 21))


def build_topk_curve(
    pdb_ids: list[str],
    candidates: list[dict[str, Any]],
    sorting: str,
    candidate_success: Callable[[dict[str, Any]], bool],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """调用 candidate_success 跑 candidates 的 match, 根据 pdb_ids 汇总逐个 pdb 的完整尝试轨迹 traces(不止于selected或者20)。之后汇总 top-k 曲线.

    输入参数:
        - pdb_ids: list[str], 曲线分母中的有序 PDB 标识.
        - candidates: list[dict[str, Any]], 一种受体条件或 Emap2lig 的全部候选; 每项包含 PDB、blob 编号、selected 标记、排序分数、目标范围、目标 occurrence 和目标 SMILES.
        - sorting: str, gaussian_score 或 source_probability_mean; 只改变 selected 与 unselected 两段内部的顺序.
        - candidate_success: Callable, 作用于某个 candidate(candadates 的一项); 只允许 Stage3 前景候选形成命中.

    返回值:
        - curve: dict[str, Any], 含 pdb_ids、top_k、hit_count、hit_ratio 和逐 PDB 累计命中 pdb_hit.
        - traces: dict[str, list[dict[str, Any]]], 每个 PDB 的完整候选顺序、计费位置、目标和命中状态.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["pdb_id"])].append(candidate)

    pdb_hit: dict[str, list[bool]] = {}
    traces: dict[str, list[dict[str, Any]]] = {}
    for pdb_id in pdb_ids:
        records = grouped.get(pdb_id, [])
        score = np.asarray(
            [float(record[sorting]) for record in records], dtype=np.float64
        )
        source_blob_index = np.asarray(
            [int(record["source_blob_index"]) for record in records], dtype=np.int64
        )
        selected = np.asarray(
            [bool(record["candidate_selected"]) for record in records], dtype=np.bool_
        )
        # int64, (N_candidate,), selected 段整体位于 unselected 段之前; 每段内部按分数降序、blob 编号升序排列.
        order = selected_priority_order(score, source_blob_index, selected)
        spent = 0
        first_hit_attempt: int | None = None
        pdb_trace: list[dict[str, Any]] = []
        for raw_rank, position in enumerate(order.tolist(), start=1):
            record = records[position]
            target_scope = str(record["target_scope"])
            if target_scope == "non_stage3_foreground":
                action = "skip_non_stage3_foreground"
                attempt_index = None
                success = False
            else:
                spent += 1
                attempt_index = spent
                action = ("spend_background" if target_scope == "background" else "spend_stage3_target")
                success = target_scope == "stage3_foreground" and bool(candidate_success(record))
                if success and first_hit_attempt is None:
                    first_hit_attempt = spent
            pdb_trace.append(
                {
                    "source_blob_index": int(record["source_blob_index"]),
                    "centered_box_index": record["centered_box_index"],
                    "raw_rank": raw_rank,
                    "attempt_index": attempt_index,
                    "action": action,
                    "target_scope": target_scope,
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

    hit_count = [
        sum(int(pdb_hit[pdb_id][index]) for pdb_id in pdb_ids)
        for index in range(len(TOP_K_VALUES))
    ]
    curve = {
        "pdb_ids": pdb_ids,
        "top_k": list(TOP_K_VALUES),
        "hit_count": hit_count,
        "hit_ratio": [count / len(pdb_ids) for count in hit_count],
        "pdb_hit": pdb_hit,
    }
    return curve, traces
