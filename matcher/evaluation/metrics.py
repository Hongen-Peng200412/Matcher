"""计算前景二分类和受支持真前景候选上的 SMILES 匹配指标.

主要入口是 ValidationAccumulator.finalize. 本模块也提供同分概率稳定处理的前景 PRAUC、
固定阈值混淆计数、validation 阈值拟合和五类 SMILES 分层准确率. 所有函数只消费内存
数组并返回普通 Python 数值或映射, 不读取或写入磁盘.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.schema import FOREGROUND_CLASS_NAMES


def foreground_average_precision(target: np.ndarray, probability: np.ndarray) -> float:
    """
    按预测概率降序、同分组共同处理计算前景 average precision.

    - target: bool 一维数组; True 是五类真实前景.
    - probability: 有限浮点一维数组; 与 target 逐候选对齐.
    - return: average precision; 没有正类时返回 NaN.
    """
    truth = np.asarray(target, dtype=np.bool_)
    score = np.asarray(probability, dtype=np.float64)
    if truth.ndim != 1 or score.shape != truth.shape or not np.isfinite(score).all():
        raise ValueError("前景真值与概率必须是同形有限一维数组。")
    positive_count = int(truth.sum())
    if positive_count == 0:
        return float("nan")
    # int64 (N,), 概率降序的稳定候选下标, 同分候选保持输入顺序.
    order = np.argsort(-score, kind="stable")
    # bool (N,), 按概率降序重排的前景真值.
    sorted_truth = truth[order]
    # float64 (N,), 按降序排列且与 sorted_truth 逐候选对齐的概率.
    sorted_score = score[order]
    # int 标量, 已纳入完整同分组的累计真阳性数.
    true_positive = 0
    # int 标量, 已纳入完整同分组的累计假阳性数.
    false_positive = 0
    # float 标量, 前一完整同分组结束后的召回率.
    previous_recall = 0.0
    # float 标量, 各同分组召回增量乘组末精确率的累计和.
    average_precision = 0.0
    # int 标量, 当前同分组在排序候选轴上的起点.
    start = 0
    while start < len(order):
        # int 标量, 当前同分组的半开终点, 保证同一概率候选共同进入曲线.
        stop = start + 1
        while stop < len(order) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        # int 标量, 当前同分组包含的真实前景数.
        group_true_positive = int(sorted_truth[start:stop].sum())
        true_positive += group_true_positive
        false_positive += (stop - start) - group_true_positive
        # float 标量, 纳入当前完整同分组后的累计召回率.
        recall = true_positive / positive_count
        # float 标量, 纳入当前完整同分组后的累计精确率.
        precision = true_positive / (true_positive + false_positive)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return average_precision


def foreground_counts(
    target: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, int | float]:
    """
    在固定阈值上计算前景混淆计数和 F1.

    - target: bool (N,); 五类真实配体合并后的前景真值.
    - probability: 浮点 (N,); 与 target 对齐的前景概率.
    - threshold: 浮点标量; 概率大于或等于该值时预测为前景.
    - return: true_positive、false_positive、false_negative、true_negative 和 f1 的映射; F1 分母为零时返回 NaN.
    """
    truth = np.asarray(target, dtype=np.bool_)
    predicted = np.asarray(probability, dtype=np.float64) >= float(threshold)
    true_positive = int(np.sum(truth & predicted))
    false_positive = int(np.sum(~truth & predicted))
    false_negative = int(np.sum(truth & ~predicted))
    true_negative = int(np.sum(~truth & ~predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "f1": 2 * true_positive / denominator if denominator else float("nan"),
    }


def fit_foreground_threshold(
    target: np.ndarray, probability: np.ndarray
) -> tuple[float, dict[str, int | float]]:
    """
    只在 validation 概率上拟合前景 F1 阈值.

    - target: bool (N,); 完整 validation 的前景真值.
    - probability: 有限浮点 (N,); 完整 validation 的前景概率.
    - return[0]: F1 最大的阈值; 多个阈值并列时保留首次遍历到的较高阈值.
    - return[1]: 该阈值对应的混淆计数和 F1.
    """
    score = np.asarray(probability, dtype=np.float64)
    if score.ndim != 1 or not np.isfinite(score).all() or score.size == 0:
        raise ValueError("阈值搜索需要非空有限一维概率。")
    thresholds = np.unique(np.concatenate((score, np.asarray([1.0], dtype=np.float64))))[::-1]
    best_threshold = float(thresholds[0])
    best_metrics = foreground_counts(target, score, best_threshold)
    best_f1 = float(best_metrics["f1"])
    for threshold in thresholds[1:]:
        metrics = foreground_counts(target, score, float(threshold))
        f1 = float(metrics["f1"])
        if np.isnan(best_f1) or (np.isfinite(f1) and f1 > best_f1):
            best_threshold = float(threshold)
            best_metrics = metrics
            best_f1 = f1
    return best_threshold, best_metrics


def _accuracy(correct: int, eligible: int) -> float | None:
    """
    计算允许零支持的准确率.

    - correct: 匹配正确的非负候选数.
    - eligible: 进入当前指标分母的非负候选数.
    - return: correct/eligible; eligible 为零时返回 None, 在 JSON 与图中表示 N/A.
    """
    return correct / eligible if eligible else None


def smiles_match_metrics(
    predicted_smiles: list[str | None],
    target_smiles: list[str | None],
    target_category: list[str | None],
    foreground_target: np.ndarray,
    identity_supported: np.ndarray,
) -> dict[str, Any]:
    """
    汇总受支持真前景候选的 SMILES 匹配准确率.

    - predicted_smiles: 长度 N; 每个候选的闭集 argmax 精确 SMILES, 无法预测时为 None.
    - target_smiles: 长度 N; 最大覆盖 occurrence 的精确 SMILES, 背景为 None.
    - target_category: 长度 N; 五类前景标签之一, 背景为 None.
    - foreground_target: bool (N,); True 表示真实前景.
    - identity_supported: bool (N,); 目标图和 SMI-TED 资产均可用时为 True.
    - return: overall 与五类分层的 correct_count、eligible_count、accuracy; 零支持类别 accuracy 为 None.
    """
    count = len(predicted_smiles)
    if not (
        len(target_smiles) == len(target_category) == count
        and np.asarray(foreground_target).shape == (count,)
        and np.asarray(identity_supported).shape == (count,)
    ):
        raise ValueError("SMILES 指标输入必须沿候选轴逐项对齐。")
    groups = ("overall", *FOREGROUND_CLASS_NAMES)
    correct = {group: 0 for group in groups}
    eligible = {group: 0 for group in groups}
    for index in range(count):
        if not bool(foreground_target[index]) or not bool(identity_supported[index]):
            continue
        category = target_category[index]
        if category not in FOREGROUND_CLASS_NAMES:
            raise ValueError("受支持真前景候选必须具有已知五类标签。")
        matched = predicted_smiles[index] == target_smiles[index]
        for group in ("overall", category):
            eligible[group] += 1
            correct[group] += int(matched)
    return {
        group: {
            "correct_count": correct[group],
            "eligible_count": eligible[group],
            "accuracy": _accuracy(correct[group], eligible[group]),
        }
        for group in groups
    }


def selection_score(foreground_prauc: float, smiles_accuracy: float | None) -> float:
    """
    计算唯一 validation 选模分数.

    - foreground_prauc: 有限浮点数; 完整 validation 的前景 average precision.
    - smiles_accuracy: 有限浮点数; 受支持真前景候选上的总体 SMILES 匹配准确率.
    - return: 两项指标的算术平均; 任一项不可用时失败关闭.
    """
    if not np.isfinite(foreground_prauc) or smiles_accuracy is None or not np.isfinite(smiles_accuracy):
        raise ValueError("selection 需要有限前景 PRAUC 和非空 SMILES 准确率。")
    return (float(foreground_prauc) + float(smiles_accuracy)) / 2.0


@dataclass
class ValidationAccumulator:
    """
    收集完整 validation 的逐候选预测.

    - foreground_target: 每个候选的前景真值.
    - foreground_probability: 每个候选的前景概率.
    - predicted_smiles: 每个候选的闭集 argmax 精确 SMILES.
    - target_smiles: 每个候选的确定目标 SMILES; 背景为 None.
    - target_category: 每个候选的五类前景标签; 背景为 None.
    - identity_supported: 目标图和语言向量是否均受支持.

    所有列表必须沿同一候选轴追加; finalize 在完整 validation 结束后一次性计算指标.
    """
    foreground_target: list[bool]
    foreground_probability: list[float]
    predicted_smiles: list[str | None]
    target_smiles: list[str | None]
    target_category: list[str | None]
    identity_supported: list[bool]

    @classmethod
    def empty(cls) -> "ValidationAccumulator":
        """
        建立空累积器.

        - return: 六个逐候选列表均为空的 ValidationAccumulator.
        """
        return cls([], [], [], [], [], [])

    def finalize(self, threshold: float | None = None) -> dict[str, Any]:
        """
        计算完整候选集合的科学指标.

        - threshold: 可选固定前景阈值; None 时不报告 F1, 有值时只应用而不重新拟合.
        - return: candidate_count、foreground_prauc、smiles_match 和 selection; 提供阈值时 额外含 foreground_threshold 与前景混淆计数/F1.
        """
        truth = np.asarray(self.foreground_target, dtype=np.bool_)
        probability = np.asarray(self.foreground_probability, dtype=np.float64)
        prauc = foreground_average_precision(truth, probability)
        smiles = smiles_match_metrics(
            self.predicted_smiles,
            self.target_smiles,
            self.target_category,
            truth,
            np.asarray(self.identity_supported, dtype=np.bool_),
        )
        result: dict[str, Any] = {
            "candidate_count": len(truth),
            "foreground_prauc": prauc,
            "smiles_match": smiles,
            "selection": selection_score(prauc, smiles["overall"]["accuracy"]),
        }
        if threshold is not None:
            result["foreground_threshold"] = float(threshold)
            result["foreground"] = foreground_counts(truth, probability, threshold)
        return result
