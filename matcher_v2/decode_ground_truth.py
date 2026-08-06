"""模式一专用的纯打分、空槽位解码、阈值扫描和指标累计。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .contracts import MatcherSample
from .model import MatcherHeadOutput


@dataclass(frozen=True)
class DecodedPair:
    candidate_index: int
    occurrence_index: int
    score: float


@dataclass(frozen=True)
class MatcherMetrics:
    precision: float
    recall: float
    F1: float
    PR_AUC: float
    threshold: float
    predicted: int
    correct: int
    occurrence: int


@dataclass(frozen=True)
class EvaluationPDB:
    """验证只保留得分、硬正确矩阵和 occurrence 数，不保留图或密度。"""

    score: torch.Tensor
    correct_pair: torch.Tensor
    occurrence: int


def pair_score(head: MatcherHeadOutput) -> torch.Tensor:
    """六项概率等权平均；权重实验只需在这一处替换。"""

    return torch.stack(
        (
            head.A, head.B, head.O,
            torch.sigmoid(head.A_prime_logit),
            torch.sigmoid(head.B_prime_logit),
            torch.sigmoid(head.O_prime_logit),
        )
    ).mean(dim=0)


def decode_pairs(score: torch.Tensor, threshold: float) -> tuple[DecodedPair, ...]:
    """一个 PDB 内联合决定一对一匹配与空槽位。"""

    value = score.detach().float().cpu().numpy()
    candidate_count, occurrence_count = value.shape
    size = candidate_count + occurrence_count
    cost = np.zeros((size, size), dtype=np.float32)
    cost[:candidate_count, :occurrence_count] = -(value - threshold)
    candidate, occurrence = linear_sum_assignment(cost)
    return tuple(
        DecodedPair(int(i), int(o), float(score[i, o]))
        for i, o in zip(candidate, occurrence, strict=True)
        if i < candidate_count and o < occurrence_count and float(score[i, o]) >= threshold
    )


def prepare_evaluation(head: MatcherHeadOutput, sample: MatcherSample) -> EvaluationPDB:
    """从模型出口立即压缩出模式一评估所需的 CPU 张量。"""

    return EvaluationPDB(
        pair_score(head).detach().float().cpu(),
        (sample.A_prime_target & sample.B_prime_target & sample.O_prime_target).cpu(),
        sample.num_occurrences,
    )


def _evaluate_at(
    records: tuple[EvaluationPDB, ...], threshold: float, missing_occurrences: int = 0
) -> MatcherMetrics:
    predicted = 0
    correct = 0
    occurrence = missing_occurrences
    for record in records:
        pairs = decode_pairs(record.score, threshold)
        predicted += len(pairs)
        correct += sum(
            bool(record.correct_pair[pair.candidate_index, pair.occurrence_index])
            for pair in pairs
        )
        occurrence += record.occurrence
    precision = correct / predicted if predicted else 0.0
    recall = correct / occurrence if occurrence else 0.0
    F1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return MatcherMetrics(precision, recall, F1, 0.0, threshold, predicted, correct, occurrence)


def _scan(
    records: tuple[EvaluationPDB, ...], *, step: float, missing_occurrences: int = 0
) -> tuple[MatcherMetrics, ...]:
    values = tuple(
        _evaluate_at(records, float(threshold), missing_occurrences)
        for threshold in np.arange(0.0, 1.0 + step / 2.0, step)
    )
    by_recall: dict[float, float] = {0.0: 1.0}
    for value in values:
        by_recall[value.recall] = max(by_recall.get(value.recall, 0.0), value.precision)
    recall = np.asarray(sorted(by_recall), dtype=np.float64)
    precision = np.asarray([by_recall[value] for value in recall], dtype=np.float64)
    PR_AUC = float(np.trapz(precision, recall))
    return tuple(replace(value, PR_AUC=PR_AUC) for value in values)


def evaluate(
    records: tuple[EvaluationPDB, ...], *, threshold: float,
    step: float = 0.003, missing_occurrences: int = 0,
) -> MatcherMetrics:
    """使用冻结阈值评估；PR-AUC 仍由完整解码曲线计算，不重选阈值。"""

    PR_AUC = _scan(records, step=step, missing_occurrences=missing_occurrences)[0].PR_AUC
    return replace(_evaluate_at(records, threshold, missing_occurrences), PR_AUC=PR_AUC)


def select_threshold(
    records: tuple[EvaluationPDB, ...], *, step: float = 0.003
) -> MatcherMetrics:
    """只在验证时扫描；独立评估必须读取 checkpoint 冻结阈值。"""

    return max(_scan(records, step=step), key=lambda item: item.F1)
