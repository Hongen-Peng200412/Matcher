"""累积 Matcher 完整划分评估指标，并计算 validation 选模分数。

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

from ..data.schema import (
    CLASS_NAMES,
    FOREGROUND_CLASS_INDICES,
    FOREGROUND_CLASS_NAMES,
    SMALL_MOLECULE_INDEX,
    IdentityScope,
)
from ..datasets.contracts import CandidateMetadata


class EvaluationBatch(TypedDict):
    """声明评估器实际读取且沿批次轴 ``B`` 对齐的字段。

    ``class_target`` 是 int64 ``(B,)`` 六分类目标。
    ``ligand_area_target`` 是 bool ``(B,Z,Y,X)`` 候选配体区域目标。
    ``binding_area_target`` 是 bool ``(B,Z,Y,X)`` 受体结合区域目标。
    ``identity_target`` 是 int64 ``(B,)``；``-1`` 表示当前语言模型没有可监督身份。
    ``identity_mask`` 是 bool ``(B,I)`` 有效身份槽；非负 identity_target 必须指向真值槽。
    ``identity_class_index`` 是 int64 ``(B,I)``；有效槽是五类非背景下标，补零槽为 -1。
    ``metadata`` 长度为 ``B``；identity_seen_in_train 按当前语言模型记录训练正目标历史。
    """

    class_target: torch.Tensor
    ligand_area_target: torch.Tensor
    binding_area_target: torch.Tensor
    identity_target: torch.Tensor
    identity_mask: torch.Tensor
    identity_class_index: torch.Tensor
    metadata: list[CandidateMetadata]


class PerClassMetrics(TypedDict):
    """一个六分类类别的指标。

    ``precision`` 是该类真阳性除以该类预测数；零预测时为 NaN。
    ``recall`` 是该类真阳性除以该类真实数；零支持时为 NaN。
    ``f1`` 是有限 precision/recall 的调和平均；无法计算时为 0。
    ``support`` 是该类真实候选数。
    ``predicted_count`` 是被六分类头预测为该类的候选数。
    """

    precision: float
    recall: float
    f1: float
    support: int
    predicted_count: int


class SixClassMetrics(TypedDict):
    """六分类汇总。

    ``accuracy`` 是全部候选的预测正确率。
    ``macro_f1`` 是与既有 validation 选模兼容的支持类 macro-F1。
    ``macro_f1_supported_classes`` 只平均真实支持数大于零的类别 F1。
    ``macro_f1_all_classes`` 固定平均 CLASS_NAMES 中全部六类，零支持类按 F1=0 计入。
    ``per_class`` 按 CLASS_NAMES 保存每类 PerClassMetrics。
    ``confusion_matrix`` 是 ``(6,6)`` 整数嵌套列表，第一轴为真实类，第二轴为预测类。
    """

    accuracy: float
    macro_f1: float
    macro_f1_supported_classes: float
    macro_f1_all_classes: float
    per_class: dict[str, PerClassMetrics]
    confusion_matrix: list[list[int]]


class IntrinsicIdentityMetrics(TypedDict):
    """忽略六分类预测后的一个指定身份范围诊断。

    ``top1`` 是该范围全部可监督真实候选的身份 Top-1 正确率。
    ``topk`` 是该范围全部可监督真实候选的身份 Top-K 正确率。
    ``top_k`` 是本报告实际采用的最大名次 K。
    ``eligible_count`` 是真实类别属于该范围且 identity_target 非负的候选数。
    """

    top1: float
    topk: float
    top_k: int
    eligible_count: int


class ConditionalIdentityMetrics(TypedDict):
    """仅对六分类预测正确且属于指定身份范围的可监督真实候选计算 Top-K。

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
    """经当前报告规定的背景/前景门后评价最终身份 Top-1 的端到端指标。

    ``precision`` 是端到端身份正确数除以该范围的预测阳性数。
    ``recall`` 是端到端身份正确数除以当前范围内有身份目标的真实候选数。
    ``f1`` 是有限 precision/recall 的调和平均；无法计算时为 0。
    ``true_positive`` 是通过当前前景门且最终身份正确的候选数。
    ``predicted_positive`` 是通过当前前景门且最终身份落入本报告范围的候选数。
    ``actual_positive`` 是当前范围内有语言与配体图身份目标的真实候选数。
    """

    precision: float
    recall: float
    f1: float
    true_positive: int
    predicted_positive: int
    actual_positive: int


class IdentityScopeMetrics(TypedDict):
    """一个身份范围的固有、条件和端到端指标。

    ``intrinsic`` 忽略六分类预测，只评价闭集身份排名。
    ``conditional`` 仅在六分类预测等于真实类别时评价闭集身份排名。
    ``end_to_end`` 经该报告规定的背景/前景门后评价最终身份 Top-1。
    """

    intrinsic: IntrinsicIdentityMetrics
    conditional: IntrinsicIdentityMetrics
    end_to_end: EndToEndIdentityMetrics


class AreaMetrics(TypedDict):
    """在 logit 0 阈值下跨样本、跨体素微平均的区域 Dice。

    ``ligand_micro_dice`` 汇总全部候选的配体区域交集与预测/真实体素数。
    ``binding_micro_dice`` 汇总全部候选的受体结合区域交集与预测/真实体素数。
    ``ligand_dice_denominator`` 是配体区域预测体素数与真实体素数之和。
    ``binding_dice_denominator`` 是受体结合区域预测体素数与真实体素数之和。
    """

    ligand_micro_dice: float
    binding_micro_dice: float
    ligand_dice_denominator: int
    binding_dice_denominator: int


class EvaluationReport(TypedDict):
    """完整评估报告；零支持分母在内存中为 NaN，JSON 落盘时转为 null。

    ``six_class`` 是全候选六分类报告。
    ``intrinsic_identity`` 是忽略六分类预测的身份诊断。
    ``conditional_identity`` 是以六分类判对为条件的身份报告。
    ``end_to_end_identity`` 是当前配置身份范围采用其背景/前景门后的最终身份报告。
    ``small_molecule_identity`` 使用只含小分子身份的清单和严格 small_molecule 分类门，供第0批直接比较。
    ``all_non_background_identity`` 在五类共享身份清单中计算总体指标；端到端只要求预测为任一前景类。
    ``identity_per_class`` 在共享清单中按真实类别拆分固有/条件指标，端到端预测按最终身份类别归桶。
    ``area`` 是两个辅助区域的微平均 Dice。
    """

    six_class: SixClassMetrics
    intrinsic_identity: IntrinsicIdentityMetrics
    conditional_identity: ConditionalIdentityMetrics
    end_to_end_identity: EndToEndIdentityMetrics
    small_molecule_identity: IdentityScopeMetrics
    all_non_background_identity: IdentityScopeMetrics
    identity_per_class: dict[str, IdentityScopeMetrics]
    area: AreaMetrics


def _safe_divide(numerator: float, denominator: float) -> float:
    """分母为零时返回 NaN，避免把没有支持样本误报为零性能。"""

    return numerator / denominator if denominator > 0 else math.nan


def _require_finite(values: torch.Tensor, name: str) -> None:
    """拒绝会被 argmax 或阈值运算静默解释的 NaN/Inf 模型输出。"""

    if not torch.isfinite(values).all():
        raise FloatingPointError(f"{name} 含 NaN 或 Inf。")


@dataclass
class _IdentityCounts:
    """保存一个身份范围跨批次可加的排名与联合判定整数计数。"""

    intrinsic_correct_at_1: int = 0
    intrinsic_correct_at_k: int = 0
    intrinsic_eligible: int = 0
    conditional_correct_at_1: int = 0
    conditional_correct_at_k: int = 0
    conditional_eligible: int = 0
    end_to_end_true_positive: int = 0
    end_to_end_predicted_positive: int = 0
    end_to_end_actual_positive: int = 0

    def update(
        self,
        intrinsic_eligible: np.ndarray,
        conditional_eligible: np.ndarray,
        top1_correct: np.ndarray,
        top_k_correct: np.ndarray,
        end_to_end_correct: np.ndarray,
        predicted_positive: np.ndarray,
    ) -> None:
        """把同一批次的排名、端到端真阳性和预测阳性布尔量累加到当前范围。"""

        self.intrinsic_eligible += int(intrinsic_eligible.sum())
        self.intrinsic_correct_at_1 += int((intrinsic_eligible & top1_correct).sum())
        self.intrinsic_correct_at_k += int((intrinsic_eligible & top_k_correct).sum())
        self.conditional_eligible += int(conditional_eligible.sum())
        self.conditional_correct_at_1 += int((conditional_eligible & top1_correct).sum())
        self.conditional_correct_at_k += int((conditional_eligible & top_k_correct).sum())
        self.end_to_end_true_positive += int(end_to_end_correct.sum())
        self.end_to_end_predicted_positive += int(predicted_positive.sum())
        self.end_to_end_actual_positive += int(intrinsic_eligible.sum())

    def compute(self, top_k: int) -> IdentityScopeMetrics:
        """按当前整数分母返回一个可落盘身份范围报告。"""

        precision = _safe_divide(
            self.end_to_end_true_positive,
            self.end_to_end_predicted_positive,
        )
        recall = _safe_divide(
            self.end_to_end_true_positive,
            self.end_to_end_actual_positive,
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if math.isfinite(precision)
            and math.isfinite(recall)
            and precision + recall > 0
            else 0.0
        )
        return {
            "intrinsic": {
                "top1": _safe_divide(self.intrinsic_correct_at_1, self.intrinsic_eligible),
                "topk": _safe_divide(self.intrinsic_correct_at_k, self.intrinsic_eligible),
                "top_k": top_k,
                "eligible_count": self.intrinsic_eligible,
            },
            "conditional": {
                "top1": _safe_divide(
                    self.conditional_correct_at_1,
                    self.conditional_eligible,
                ),
                "topk": _safe_divide(
                    self.conditional_correct_at_k,
                    self.conditional_eligible,
                ),
                "top_k": top_k,
                "eligible_count": self.conditional_eligible,
            },
            "end_to_end": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positive": self.end_to_end_true_positive,
                "predicted_positive": self.end_to_end_predicted_positive,
                "actual_positive": self.end_to_end_actual_positive,
            },
        }


def _masked_identity_prediction(
    scores: torch.Tensor,
    slot_mask: torch.Tensor,
    target: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """在指定身份槽内返回 Top-1 下标和目标是否进入 Top-K。"""

    prediction = np.full(target.shape, -1, dtype=np.int64)
    top_k_correct = np.zeros(target.shape, dtype=bool)
    if scores.shape[1] == 0:
        return prediction, top_k_correct
    masked = scores.masked_fill(~slot_mask, torch.finfo(scores.dtype).min).detach().cpu()
    prediction = masked.argmax(dim=-1).numpy()
    count = min(top_k, masked.shape[1])
    top_indices = masked.topk(count, dim=-1).indices.numpy()
    top_k_correct = np.any(top_indices == target[:, None], axis=1)
    has_slot = slot_mask.any(dim=1).detach().cpu().numpy()
    prediction[~has_slot] = -1
    top_k_correct[~has_slot] = False
    return prediction, top_k_correct


def _scope_masks(
    class_prediction: np.ndarray,
    class_target: np.ndarray,
    target: np.ndarray,
    slot_options: np.ndarray,
    allowed_classes: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回固有分母、精确子类条件分母、前景门与可评估身份槽掩码。"""

    true_in_scope = np.isin(class_target, allowed_classes)
    predicted_in_scope = np.isin(class_prediction, allowed_classes)
    intrinsic = true_in_scope & (target >= 0)
    conditional = intrinsic & (class_prediction == class_target)
    end_to_end_scope = slot_options & ~(true_in_scope & (target < 0))
    return intrinsic, conditional, predicted_in_scope, end_to_end_scope


# ================================================================================================


@dataclass
class EvaluationAccumulator:
    """把多个物理批次合并为不依赖批次切分的 ``EvaluationReport`` 计数。

    ``class_confusion`` 是 ``(6,6)`` int64 混淆矩阵，第一轴真实类、第二轴预测类。
    ``identity_scope`` 决定旧三项顶层身份字段和 validation 选模使用哪一范围。
    ``small_identity/all_identity/per_class_identity`` 分别保存兼容小分子、五类总体和分类计数。
    ``seen_*`` 与 ``unseen_*`` 按当前语言模型的训练正目标历史拆分条件身份计数。
    ``end_to_end_*`` 累积当前报告前景门后的身份真阳性、预测阳性和真实阳性。
    ``ligand_*`` 与 ``binding_*`` 累积区域交集、预测体素数和真实体素数。
    ``top_k`` 是正整数最大名次，默认 5 只用于可独立构造的零状态累加器。
    """

    identity_scope: IdentityScope | None
    class_confusion: np.ndarray = field(
        default_factory=lambda: np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    )
    small_identity: _IdentityCounts = field(default_factory=_IdentityCounts)
    all_identity: _IdentityCounts = field(default_factory=_IdentityCounts)
    per_class_identity: dict[int, _IdentityCounts] = field(
        default_factory=lambda: {
            class_index: _IdentityCounts() for class_index in FOREGROUND_CLASS_INDICES
        }
    )
    seen_correct_at_1: int = 0
    seen_correct_at_k: int = 0
    seen_eligible: int = 0
    unseen_correct_at_1: int = 0
    unseen_correct_at_k: int = 0
    unseen_eligible: int = 0
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
        if self.identity_scope not in (None, "small_molecule", "all_non_background"):
            raise ValueError("identity_scope 必须为 None、small_molecule 或 all_non_background。")
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
        """更新固有身份、精确子类条件 Top-K 与背景/前景门后的端到端指标。"""

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
        identity_class_index = batch["identity_class_index"]
        if identity_class_index.dtype != torch.long or identity_class_index.shape != (
            identity_logits.shape
        ):
            raise ValueError("identity_class_index 必须是与身份轴对齐的 int64 (B,I) 张量。")
        if torch.any(identity_class_index[batch["identity_mask"]] <= 0) or torch.any(
            identity_class_index[batch["identity_mask"]] >= len(CLASS_NAMES)
        ):
            raise ValueError("有效身份必须属于五类非背景类别。")
        if torch.any(identity_class_index[~batch["identity_mask"]] != -1):
            raise ValueError("补零身份的 identity_class_index 必须为 -1。")
        if torch.any(target_available) and not torch.equal(
            identity_class_index[target_available, target_tensor[target_available]],
            batch["class_target"][target_available],
        ):
            raise ValueError("identity_target 的身份类别必须等于候选真实类别。")
        target = batch["identity_target"].detach().cpu().numpy()
        class_target = batch["class_target"].detach().cpu().numpy()
        full_prediction, full_top_k = _masked_identity_prediction(
            identity_logits,
            batch["identity_mask"],
            target,
            self.top_k,
        )
        full_top1 = full_prediction == target
        small_slot_mask = batch["identity_mask"] & (
            identity_class_index == SMALL_MOLECULE_INDEX
        )
        small_prediction, small_top_k = _masked_identity_prediction(
            identity_logits,
            small_slot_mask,
            target,
            self.top_k,
        )
        small_top1 = small_prediction == target

        small_slot_options = small_slot_mask.any(dim=1).detach().cpu().numpy()
        small_masks = _scope_masks(
            class_prediction,
            class_target,
            target,
            small_slot_options,
            (SMALL_MOLECULE_INDEX,),
        )
        small_predicted_positive = small_masks[2] & small_masks[3]
        small_end_to_end_correct = small_masks[0] & small_predicted_positive & small_top1
        self.small_identity.update(
            *small_masks[:2],
            small_top1,
            small_top_k,
            small_end_to_end_correct,
            small_predicted_positive,
        )

        full_slot_options = batch["identity_mask"].any(dim=1).detach().cpu().numpy()
        all_masks = _scope_masks(
            class_prediction,
            class_target,
            target,
            full_slot_options,
            FOREGROUND_CLASS_INDICES,
        )
        all_predicted_positive = all_masks[2] & all_masks[3]
        all_end_to_end_correct = all_masks[0] & all_predicted_positive & full_top1
        self.all_identity.update(
            *all_masks[:2],
            full_top1,
            full_top_k,
            all_end_to_end_correct,
            all_predicted_positive,
        )

        identity_class_values = identity_class_index.detach().cpu().numpy()
        predicted_identity_class = np.full(target.shape, -1, dtype=np.int64)
        valid_prediction = full_prediction >= 0
        if np.any(valid_prediction):
            row_indices = np.arange(target.shape[0])[valid_prediction]
            predicted_identity_class[valid_prediction] = identity_class_values[
                row_indices,
                full_prediction[valid_prediction],
            ]
        for class_index in FOREGROUND_CLASS_INDICES:
            class_slot_options = (
                batch["identity_mask"] & (identity_class_index == class_index)
            ).any(dim=1).detach().cpu().numpy()
            class_masks = _scope_masks(
                class_prediction,
                class_target,
                target,
                class_slot_options,
                (class_index,),
            )
            class_predicted_positive = (
                all_masks[2]
                & all_masks[3]
                & (predicted_identity_class == class_index)
            )
            class_end_to_end_correct = (
                class_masks[0] & class_predicted_positive & full_top1
            )
            self.per_class_identity[class_index].update(
                *class_masks[:2],
                full_top1,
                full_top_k,
                class_end_to_end_correct,
                class_predicted_positive,
            )

        configured_counts = (
            self.small_identity
            if self.identity_scope in (None, "small_molecule")
            else self.all_identity
        )
        configured_conditional = (
            small_masks[1]
            if self.identity_scope in (None, "small_molecule")
            else all_masks[1]
        )
        configured_top1 = (
            small_top1
            if self.identity_scope in (None, "small_molecule")
            else full_top1
        )
        configured_top_k = (
            small_top_k
            if self.identity_scope in (None, "small_molecule")
            else full_top_k
        )

        for index, metadata in enumerate(batch["metadata"]):
            if not configured_conditional[index]:
                continue
            if type(metadata["identity_seen_in_train"]) is not bool:
                raise ValueError("条件身份样本的 identity_seen_in_train 必须是布尔值。")
            if metadata["identity_seen_in_train"] is True:
                self.seen_eligible += 1
                self.seen_correct_at_1 += int(configured_top1[index])
                self.seen_correct_at_k += int(configured_top_k[index])
            elif metadata["identity_seen_in_train"] is False:
                self.unseen_eligible += 1
                self.unseen_correct_at_1 += int(configured_top1[index])
                self.unseen_correct_at_k += int(configured_top_k[index])
        if configured_counts.conditional_eligible != self.seen_eligible + self.unseen_eligible:
            raise ValueError("条件身份样本必须完整划分为 seen 或 unseen。")

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
                "predicted_count": int(predicted[index]),
            }
        accuracy = _safe_divide(float(true_positive.sum()), float(self.class_confusion.sum()))
        supported_macro_f1 = float(np.mean(class_f1)) if class_f1 else math.nan
        all_classes_macro_f1 = float(
            np.mean([class_metrics[name]["f1"] for name in CLASS_NAMES])
        )
        small_report = self.small_identity.compute(self.top_k)
        all_report = self.all_identity.compute(self.top_k)
        configured_report = (
            small_report
            if self.identity_scope in (None, "small_molecule")
            else all_report
        )
        configured_intrinsic = configured_report["intrinsic"]
        configured_conditional = configured_report["conditional"]
        configured_end_to_end = configured_report["end_to_end"]
        conditional_with_history: ConditionalIdentityMetrics = {
            **configured_conditional,
            "seen_top1": _safe_divide(self.seen_correct_at_1, self.seen_eligible),
            "seen_topk": _safe_divide(self.seen_correct_at_k, self.seen_eligible),
            "seen_count": self.seen_eligible,
            "unseen_top1": _safe_divide(self.unseen_correct_at_1, self.unseen_eligible),
            "unseen_topk": _safe_divide(self.unseen_correct_at_k, self.unseen_eligible),
            "unseen_count": self.unseen_eligible,
        }
        per_class_identity = {
            class_name: self.per_class_identity[class_index].compute(self.top_k)
            for class_index, class_name in zip(
                FOREGROUND_CLASS_INDICES,
                FOREGROUND_CLASS_NAMES,
            )
        }
        return {
            "six_class": {
                "accuracy": accuracy,
                "macro_f1": supported_macro_f1,
                "macro_f1_supported_classes": supported_macro_f1,
                "macro_f1_all_classes": all_classes_macro_f1,
                "per_class": class_metrics,
                "confusion_matrix": self.class_confusion.tolist(),
            },
            "intrinsic_identity": configured_intrinsic,
            "conditional_identity": conditional_with_history,
            "end_to_end_identity": configured_end_to_end,
            "small_molecule_identity": small_report,
            "all_non_background_identity": all_report,
            "identity_per_class": per_class_identity,
            "area": {
                "ligand_micro_dice": _safe_divide(
                    2 * self.ligand_intersection,
                    self.ligand_predicted + self.ligand_actual,
                ),
                "binding_micro_dice": _safe_divide(
                    2 * self.binding_intersection,
                    self.binding_predicted + self.binding_actual,
                ),
                "ligand_dice_denominator": self.ligand_predicted + self.ligand_actual,
                "binding_dice_denominator": self.binding_predicted + self.binding_actual,
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
