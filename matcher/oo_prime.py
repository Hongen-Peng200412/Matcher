"""只适用于当前 O/O′ 输出的解码、阈值扫描和 occurrence 级评估。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SelectedPair:
    """一个非空 slot 的当前 O/O′ 解码关系。"""

    slot_index: int
    candidate_index: int
    O_probability: float
    O_prime: float
    decode_score: float


@dataclass(frozen=True)
class OOPrimeDecoded:
    """当前 decoder 的稀疏关系结果。"""

    selected_pairs: tuple[SelectedPair, ...]
    unmatched_slot_indices: tuple[int, ...]


@dataclass(frozen=True)
class OOPrimeMetrics:
    """完整验证集最佳阈值对应的 TP/P/G 与 precision/recall/F1。"""

    threshold: float
    true_positive: int
    predicted_nonempty: int
    ground_truth_occurrence: int
    precision: float
    recall: float
    F1: float


class OOPrimeDecoder:
    """逐 slot 过滤 O 概率，再按 ``O_probability * O_prime`` 选择候选。"""

    def __init__(self, O_threshold: float = 0.5) -> None:
        self.O_threshold = float(O_threshold)

    def decode(self, O_logit: Tensor, O_prime: Tensor) -> OOPrimeDecoded:
        O_probability = torch.sigmoid(O_logit.float())
        decode_score = O_probability * O_prime.float()
        selected = []
        unmatched = []
        for slot_index in range(O_logit.shape[1]):
            valid = O_probability[:, slot_index] >= self.O_threshold
            if not valid.any():
                unmatched.append(slot_index)
                continue
            valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
            local_index = int(torch.argmax(decode_score[valid, slot_index]))
            candidate_index = int(valid_indices[local_index])
            selected.append(
                SelectedPair(
                    slot_index,
                    candidate_index,
                    float(O_probability[candidate_index, slot_index]),
                    float(O_prime[candidate_index, slot_index]),
                    float(decode_score[candidate_index, slot_index]),
                )
            )
        return OOPrimeDecoded(tuple(selected), tuple(unmatched))


class OOPrimeEvaluation:
    """复用 decoder 累计全局 TP/P/G；可扫描验证阈值或评估一个冻结阈值。"""

    def __init__(
        self,
        threshold_step: float | None = 0.01,
        *,
        frozen_threshold: float | None = None,
    ) -> None:
        if frozen_threshold is None:
            if threshold_step is None:
                raise ValueError("阈值扫描必须提供 threshold_step。")
            count = round(1.0 / threshold_step)
            self.thresholds = tuple(index * threshold_step for index in range(count + 1))
        else:
            self.thresholds = (float(frozen_threshold),)
        self.true_positive = [0] * len(self.thresholds)
        self.predicted_nonempty = [0] * len(self.thresholds)
        self.ground_truth_occurrence = 0

    @torch.no_grad()
    def update(
        self,
        O_logit: Tensor,
        O_prime: Tensor,
        O_target: Tensor,
        assignment: Tensor,
    ) -> None:
        aligned_target = O_target.to(O_logit.device)[:, assignment]
        for threshold_index, threshold in enumerate(self.thresholds):
            decoded = OOPrimeDecoder(threshold).decode(O_logit, O_prime)
            self.predicted_nonempty[threshold_index] += len(decoded.selected_pairs)
            self.true_positive[threshold_index] += sum(
                int(aligned_target[pair.candidate_index, pair.slot_index])
                for pair in decoded.selected_pairs
            )
        self.ground_truth_occurrence += O_logit.shape[1]

    def best(self) -> OOPrimeMetrics:
        best_metrics = OOPrimeMetrics(0.0, 0, 0, self.ground_truth_occurrence, 0.0, 0.0, 0.0)
        for threshold, true_positive, predicted_nonempty in zip(
            self.thresholds,
            self.true_positive,
            self.predicted_nonempty,
            strict=True,
        ):
            precision = true_positive / predicted_nonempty if predicted_nonempty else 0.0
            recall = (
                true_positive / self.ground_truth_occurrence
                if self.ground_truth_occurrence
                else 0.0
            )
            F1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            if F1 >= best_metrics.F1:
                best_metrics = OOPrimeMetrics(
                    threshold,
                    true_positive,
                    predicted_nonempty,
                    self.ground_truth_occurrence,
                    precision,
                    recall,
                    F1,
                )
        return best_metrics
