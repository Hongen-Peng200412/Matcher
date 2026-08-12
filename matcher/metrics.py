"""累积并选择 Matcher validation 指标。

``EvaluationAccumulator`` 跨物理批次累积六分类、固有身份、条件身份、端到端身份和
区域计数，再由 ``compute`` 返回 ``EvaluationReport``；``selection_score`` 只接收三项
具名 validation 标量，不读取 calibration 或落盘文件。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Literal, TypedDict

import numpy as np
import torch

from .contracts import CandidateMetadata
from .schema import CLASS_NAMES, SMALL_MOLECULE_INDEX


class EvaluationBatch(TypedDict):
    """声明评估器实际读取且沿批次轴 ``B`` 对齐的字段。

    ``class_target`` 是 int64 ``(B,)`` 六分类目标。
    ``ligand_area_target`` 是 bool ``(B,Z,Y,X)`` 候选配体区域目标。
    ``binding_area_target`` 是 bool ``(B,Z,Y,X)`` 受体结合区域目标。
    ``identity_target`` 是 int64 ``(B,)``；``-1`` 表示当前语言模型没有可监督身份。
    ``identity_mask`` 是 bool ``(B,I)`` 有效身份槽；非负 identity_target 必须指向真值槽。
    ``metadata`` 长度为 ``B``；identity_seen_in_train 按当前语言模型记录训练正目标历史。
    """

    class_target: torch.Tensor
    ligand_area_target: torch.Tensor
    binding_area_target: torch.Tensor
    identity_target: torch.Tensor
    identity_mask: torch.Tensor
    metadata: list[CandidateMetadata]


class PerClassMetrics(TypedDict):
    """一个六分类类别的指标。

    ``precision`` 是该类真阳性除以该类预测数；零预测时为 NaN。
    ``recall`` 是该类真阳性除以该类真实数；零支持时为 NaN。
    ``f1`` 是有限 precision/recall 的调和平均；无法计算时为 0。
    ``support`` 是该类真实候选数。
    """

    precision: float
    recall: float
    f1: float
    support: int


class SixClassMetrics(TypedDict):
    """六分类汇总。

    ``accuracy`` 是全部候选的预测正确率。
    ``macro_f1`` 只平均真实支持数大于零的类别 F1。
    ``per_class`` 按 CLASS_NAMES 保存每类 PerClassMetrics。
    ``confusion_matrix`` 是 ``(6,6)`` 整数嵌套列表，第一轴为真实类，第二轴为预测类。
    """

    accuracy: float
    macro_f1: float
    per_class: dict[str, PerClassMetrics]
    confusion_matrix: list[list[int]]


class IntrinsicIdentityMetrics(TypedDict):
    """忽略六分类预测后的身份诊断。

    ``top1`` 是全部可监督真实小分子的身份 Top-1 正确率。
    ``topk`` 是全部可监督真实小分子的身份 Top-K 正确率。
    ``top_k`` 是本报告实际采用的最大名次 K。
    ``eligible_count`` 是真实类别为 small_molecule 且 identity_target 非负的候选数。
    """

    top1: float
    topk: float
    top_k: int
    eligible_count: int


class ConditionalIdentityMetrics(TypedDict):
    """仅对六分类预测正确的可监督真实小分子计算身份 Top-K。

    ``top1`` 是条件集合上的身份 Top-1 正确率。
    ``topk`` 是条件集合上的身份 Top-K 正确率。
    ``top_k`` 是本报告实际采用的最大名次 K。
    ``eligible_count`` 是条件集合的候选数。
    ``seen_top1`` 是当前语言模型 seen 子集的身份 Top-1 正确率。
    ``seen_topk`` 是当前语言模型 seen 子集的身份 Top-K 正确率。
    ``seen_count`` 是当前语言模型 seen 子集的候选数。
    ``unseen_top1`` 是当前语言模型 unseen 子集的身份 Top-1 正确率。
    ``unseen_topk`` 是当前语言模型 unseen 子集的身份 Top-K 正确率。
    ``unseen_count`` 是当前语言模型 unseen 子集的候选数。
    """

    top1: float
    topk: float
    top_k: int
    eligible_count: int
    seen_top1: float
    seen_topk: float
    seen_count: int
    unseen_top1: float
    unseen_topk: float
    unseen_count: int


class EndToEndIdentityMetrics(TypedDict):
    """同时要求六分类与身份正确的端到端指标。

    ``precision`` 是正确类别且身份正确数除以预测为 small_molecule 的可评估候选数。
    ``recall`` 是正确类别且身份正确数除以有身份目标的真实 small_molecule 数。
    ``f1`` 是有限 precision/recall 的调和平均；无法计算时为 0。
    ``true_positive`` 是类别与身份都正确的候选数。
    ``predicted_small_molecule`` 是有身份清单且被六分类预测为 small_molecule 的候选数。
    ``actual_small_molecule`` 是有当前语言身份目标的真实 small_molecule 候选数。
    """

    precision: float
    recall: float
    f1: float
    true_positive: int
    predicted_small_molecule: int
    actual_small_molecule: int


class AreaMetrics(TypedDict):
    """在 logit 0 阈值下跨样本、跨体素微平均的区域 Dice。

    ``ligand_micro_dice`` 汇总全部候选的配体区域交集与预测/真实体素数。
    ``binding_micro_dice`` 汇总全部候选的受体结合区域交集与预测/真实体素数。
    """

    ligand_micro_dice: float
    binding_micro_dice: float


class EvaluationReport(TypedDict):
    """完整评估报告；零支持分母在内存中为 NaN，JSON 落盘时转为 null。

    ``six_class`` 是全候选六分类报告。
    ``intrinsic_identity`` 是忽略六分类预测的身份诊断。
    ``conditional_identity`` 是以六分类判对为条件的身份报告。
    ``end_to_end_identity`` 是类别与身份联合判定报告。
    ``area`` 是两个辅助区域的微平均 Dice。
    """

    six_class: SixClassMetrics
    intrinsic_identity: IntrinsicIdentityMetrics
    conditional_identity: ConditionalIdentityMetrics
    end_to_end_identity: EndToEndIdentityMetrics
    area: AreaMetrics


def _safe_divide(numerator: float, denominator: float) -> float:
    """分母为零时返回 NaN，避免把没有支持样本误报为零性能。"""

    return numerator / denominator if denominator > 0 else math.nan


def _require_finite(values: torch.Tensor, name: str) -> None:
    """拒绝会被 argmax 或阈值运算静默解释的 NaN/Inf 模型输出。"""

    if not torch.isfinite(values).all():
        raise FloatingPointError(f"{name} 含 NaN 或 Inf。")


# ================================================================================================


@dataclass
class EvaluationAccumulator:
    """把多个物理批次合并为不依赖批次切分的 ``EvaluationReport`` 计数。

    ``class_confusion`` 是 ``(6,6)`` int64 混淆矩阵，第一轴真实类、第二轴预测类。
    ``intrinsic_*`` 累积全部可监督真实小分子的 Top-1、Top-K 和分母。
    ``conditional_*`` 累积六分类判对的可监督真实小分子 Top-1、Top-K 和分母。
    ``seen_*`` 与 ``unseen_*`` 按当前语言模型的训练正目标历史拆分条件身份计数。
    ``end_to_end_*`` 累积联合类别—身份真阳性、预测阳性和真实阳性。
    ``ligand_*`` 与 ``binding_*`` 累积区域交集、预测体素数和真实体素数。
    ``top_k`` 是正整数最大名次，默认 5 只用于可独立构造的零状态累加器。
    """

    class_confusion: np.ndarray = field(
        default_factory=lambda: np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    )
    intrinsic_correct_at_1: int = 0
    intrinsic_correct_at_k: int = 0
    intrinsic_eligible: int = 0
    conditional_correct_at_1: int = 0
    conditional_correct_at_k: int = 0
    conditional_eligible: int = 0
    seen_correct_at_1: int = 0
    seen_correct_at_k: int = 0
    seen_eligible: int = 0
    unseen_correct_at_1: int = 0
    unseen_correct_at_k: int = 0
    unseen_eligible: int = 0
    end_to_end_true_positive: int = 0
    end_to_end_predicted_positive: int = 0
    end_to_end_actual_positive: int = 0
    ligand_intersection: int = 0
    ligand_predicted: int = 0
    ligand_actual: int = 0
    binding_intersection: int = 0
    binding_predicted: int = 0
    binding_actual: int = 0
    top_k: int = 5

    def __post_init__(self) -> None:
        """拒绝无意义的 Top-K 和形状错误的初始混淆矩阵。"""

        if self.top_k <= 0:
            raise ValueError("top_k 必须为正整数。")
        expected = (len(CLASS_NAMES), len(CLASS_NAMES))
        if self.class_confusion.shape != expected:
            raise ValueError(f"class_confusion 必须为 {expected}。")

    def update(
        self,
        outputs: Mapping[str, torch.Tensor],
        batch: EvaluationBatch,
    ) -> None:
        """从一个物理批次更新全部计数。

        ``outputs`` 必须含有限 ``class_logits`` ``(B,6)`` 和两个有限区域 logits
        ``(B,Z,Y,X)``；完整 Matcher 另含有限 ``identity_logits`` ``(B,I)``。
        ``batch`` 遵守 ``EvaluationBatch``，任何形状、目标范围或 metadata 对齐错误
        都立即拒绝，避免生成看似有效但分母错误的报告。
        """

        if outputs["class_logits"].ndim != 2 or outputs["class_logits"].shape[1] != len(
            CLASS_NAMES
        ):
            raise ValueError("class_logits 必须为 (B,6)。")
        _require_finite(outputs["class_logits"], "class_logits")
        if batch["class_target"].shape != (outputs["class_logits"].shape[0],):
            raise ValueError("class_target 必须与 class_logits 的批次轴对齐。")
        if batch["class_target"].dtype != torch.long:
            raise TypeError("class_target 必须是 int64。")
        if torch.any(batch["class_target"] < 0) or torch.any(
            batch["class_target"] >= len(CLASS_NAMES)
        ):
            raise ValueError("class_target 超出六分类下标范围。")
        if len(batch["metadata"]) != outputs["class_logits"].shape[0]:
            raise ValueError("metadata 长度必须与批次轴 B 一致。")
        batch_size = outputs["class_logits"].shape[0]
        class_prediction = outputs["class_logits"].argmax(dim=-1).detach().cpu().numpy()
        class_target = batch["class_target"].detach().cpu().numpy()
        np.add.at(self.class_confusion, (class_target, class_prediction), 1)
        self._update_area(
            outputs["ligand_area_logits"],
            batch["ligand_area_target"],
            "ligand",
            batch_size,
        )
        self._update_area(
            outputs["binding_area_logits"],
            batch["binding_area_target"],
            "binding",
            batch_size,
        )
        if "identity_logits" in outputs:
            self._update_identity(outputs["identity_logits"], class_prediction, batch)

    def _update_identity(
        self,
        identity_logits: torch.Tensor,
        class_prediction: np.ndarray,
        batch: EvaluationBatch,
    ) -> None:
        """更新固有身份、以六分类判对为条件的 Top-K 与端到端精确率/召回率。"""

        if identity_logits.ndim != 2 or identity_logits.shape != batch["identity_mask"].shape:
            raise ValueError("identity_logits 和 identity_mask 必须是同形 (B,I) 张量。")
        if identity_logits.shape[0] != class_prediction.shape[0]:
            raise ValueError("identity_logits 的批次轴必须与 class_logits 对齐。")
        _require_finite(identity_logits, "identity_logits")
        if batch["identity_mask"].dtype != torch.bool:
            raise TypeError("identity_mask 必须是 bool。")
        if batch["identity_target"].dtype != torch.long or batch["identity_target"].shape != (
            identity_logits.shape[0],
        ):
            raise ValueError("identity_target 必须是与批次轴对齐的 int64 (B,) 张量。")
        target_tensor = batch["identity_target"]
        if torch.any(target_tensor < -1) or torch.any(target_tensor >= identity_logits.shape[1]):
            raise ValueError("identity_target 必须为 -1 或身份轴 [0,I) 的下标。")
        target_available = target_tensor >= 0
        if torch.any(target_available) and not torch.all(
            batch["identity_mask"][target_available, target_tensor[target_available]]
        ):
            raise ValueError("非负 identity_target 必须指向 identity_mask 的真值槽。")
        target = batch["identity_target"].detach().cpu().numpy()
        class_target = batch["class_target"].detach().cpu().numpy()
        intrinsic_eligible = (class_target == SMALL_MOLECULE_INDEX) & (target >= 0)
        conditional_eligible = intrinsic_eligible & (
            class_prediction == SMALL_MOLECULE_INDEX
        )
        identity_prediction = np.full(target.shape, -1, dtype=np.int64)
        top_k_correct = np.zeros(target.shape, dtype=bool)
        if identity_logits.shape[1] > 0:
            scores = identity_logits.masked_fill(
                ~batch["identity_mask"],
                torch.finfo(identity_logits.dtype).min,
            ).detach().cpu()
            identity_prediction = scores.argmax(dim=-1).numpy()
            count = min(self.top_k, scores.shape[1])
            top_indices = scores.topk(count, dim=-1).indices.numpy()
            top_k_correct = np.any(top_indices == target[:, None], axis=1)
        top1_correct = identity_prediction == target
        self.intrinsic_eligible += int(intrinsic_eligible.sum())
        self.intrinsic_correct_at_1 += int((intrinsic_eligible & top1_correct).sum())
        self.intrinsic_correct_at_k += int((intrinsic_eligible & top_k_correct).sum())
        self.conditional_eligible += int(conditional_eligible.sum())
        self.conditional_correct_at_1 += int((conditional_eligible & top1_correct).sum())
        self.conditional_correct_at_k += int((conditional_eligible & top_k_correct).sum())

        for index, metadata in enumerate(batch["metadata"]):
            if not conditional_eligible[index]:
                continue
            if type(metadata["identity_seen_in_train"]) is not bool:
                raise ValueError("条件身份样本的 identity_seen_in_train 必须是布尔值。")
            if metadata["identity_seen_in_train"] is True:
                self.seen_eligible += 1
                self.seen_correct_at_1 += int(top1_correct[index])
                self.seen_correct_at_k += int(top_k_correct[index])
            elif metadata["identity_seen_in_train"] is False:
                self.unseen_eligible += 1
                self.unseen_correct_at_1 += int(top1_correct[index])
                self.unseen_correct_at_k += int(top_k_correct[index])

        identity_options = batch["identity_mask"].any(dim=1).detach().cpu().numpy()
        unavailable_true_small = (class_target == SMALL_MOLECULE_INDEX) & (target < 0)
        end_to_end_scope = identity_options & ~unavailable_true_small
        predicted_small = (class_prediction == SMALL_MOLECULE_INDEX) & end_to_end_scope
        actual_small = intrinsic_eligible
        # 真实小分子缺当前 LM 目标时不进身份指标；其它候选的 small-molecule 误报计入端到端假阳性。
        correct_end_to_end = predicted_small & intrinsic_eligible & top1_correct
        self.end_to_end_true_positive += int(correct_end_to_end.sum())
        self.end_to_end_predicted_positive += int(predicted_small.sum())
        self.end_to_end_actual_positive += int(actual_small.sum())

    def _update_area(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        prefix: Literal["ligand", "binding"],
        expected_batch_size: int,
    ) -> None:
        """以 logit 0 为阈值累计包含空真负样本的微平均区域 Dice 计数。"""

        if logits.ndim != 4 or logits.shape != target.shape:
            raise ValueError(f"{prefix} 区域 logits/target 必须为同形 (B,Z,Y,X)。")
        if logits.shape[0] != expected_batch_size:
            raise ValueError(f"{prefix} 区域 logits 的批次轴必须与 class_logits 对齐。")
        _require_finite(logits, f"{prefix}_area_logits")
        if target.dtype != torch.bool:
            raise TypeError(f"{prefix} 区域 target 必须是 bool。")
        prediction = logits.detach() >= 0
        target_bool = target.detach().bool()
        intersection = int((prediction & target_bool).sum().item())
        predicted = int(prediction.sum().item())
        actual = int(target_bool.sum().item())
        if prefix == "ligand":
            self.ligand_intersection += intersection
            self.ligand_predicted += predicted
            self.ligand_actual += actual
        else:
            self.binding_intersection += intersection
            self.binding_predicted += predicted
            self.binding_actual += actual

    def compute(self) -> EvaluationReport:
        """返回完整报告；零支持率在内存中为 NaN，正式 JSON 边界将其写为 null。"""

        support = self.class_confusion.sum(axis=1)
        predicted = self.class_confusion.sum(axis=0)
        true_positive = np.diag(self.class_confusion)
        class_f1 = []
        class_metrics: dict[str, PerClassMetrics] = {}
        for index, name in enumerate(CLASS_NAMES):
            precision = _safe_divide(float(true_positive[index]), float(predicted[index]))
            recall = _safe_divide(float(true_positive[index]), float(support[index]))
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if math.isfinite(precision) and math.isfinite(recall) and precision + recall > 0
                else 0.0
            )
            if support[index] > 0:
                class_f1.append(f1)
            class_metrics[name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support[index]),
            }
        accuracy = _safe_divide(float(true_positive.sum()), float(self.class_confusion.sum()))
        end_precision = _safe_divide(
            self.end_to_end_true_positive,
            self.end_to_end_predicted_positive,
        )
        end_recall = _safe_divide(
            self.end_to_end_true_positive,
            self.end_to_end_actual_positive,
        )
        end_f1 = (
            2.0 * end_precision * end_recall / (end_precision + end_recall)
            if math.isfinite(end_precision)
            and math.isfinite(end_recall)
            and end_precision + end_recall > 0
            else 0.0
        )
        return {
            "six_class": {
                "accuracy": accuracy,
                "macro_f1": float(np.mean(class_f1)) if class_f1 else math.nan,
                "per_class": class_metrics,
                "confusion_matrix": self.class_confusion.tolist(),
            },
            "intrinsic_identity": {
                "top1": _safe_divide(self.intrinsic_correct_at_1, self.intrinsic_eligible),
                "topk": _safe_divide(
                    self.intrinsic_correct_at_k,
                    self.intrinsic_eligible,
                ),
                "top_k": self.top_k,
                "eligible_count": self.intrinsic_eligible,
            },
            "conditional_identity": {
                "top1": _safe_divide(
                    self.conditional_correct_at_1,
                    self.conditional_eligible,
                ),
                "topk": _safe_divide(
                    self.conditional_correct_at_k,
                    self.conditional_eligible,
                ),
                "top_k": self.top_k,
                "eligible_count": self.conditional_eligible,
                "seen_top1": _safe_divide(self.seen_correct_at_1, self.seen_eligible),
                "seen_topk": _safe_divide(
                    self.seen_correct_at_k,
                    self.seen_eligible,
                ),
                "seen_count": self.seen_eligible,
                "unseen_top1": _safe_divide(self.unseen_correct_at_1, self.unseen_eligible),
                "unseen_topk": _safe_divide(
                    self.unseen_correct_at_k,
                    self.unseen_eligible,
                ),
                "unseen_count": self.unseen_eligible,
            },
            "end_to_end_identity": {
                "precision": end_precision,
                "recall": end_recall,
                "f1": end_f1,
                "true_positive": self.end_to_end_true_positive,
                "predicted_small_molecule": self.end_to_end_predicted_positive,
                "actual_small_molecule": self.end_to_end_actual_positive,
            },
            "area": {
                "ligand_micro_dice": _safe_divide(
                    2 * self.ligand_intersection,
                    self.ligand_predicted + self.ligand_actual,
                ),
                "binding_micro_dice": _safe_divide(
                    2 * self.binding_intersection,
                    self.binding_predicted + self.binding_actual,
                ),
            },
        }


def selection_score(
    task_name: str,
    six_class_macro_f1: float,
    conditional_identity_top1: float,
    end_to_end_identity_f1: float,
) -> float:
    """从三个具名 validation 标量返回模型选择分数；校准划分不参与该函数。"""

    if task_name not in ("candidate_pretrain", "full_matcher"):
        raise ValueError(f"不支持 task_name={task_name!r}。")
    if task_name == "candidate_pretrain":
        return six_class_macro_f1
    finite_identity = (
        0.0 if not math.isfinite(end_to_end_identity_f1) else end_to_end_identity_f1
    )
    finite_conditional = (
        0.0 if not math.isfinite(conditional_identity_top1) else conditional_identity_top1
    )
    return (six_class_macro_f1 + finite_identity + finite_conditional) / 3.0
