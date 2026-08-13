"""用 ``compute_matcher_loss`` 计算候选基础损失与三种身份门控，并以 ``LossOutput`` 返回可记录分量。

符号 ``B`` 表示物理批量，``I`` 表示批内最大身份数，``Z/Y/X`` 表示三个体素空间轴。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypedDict

import torch
from torch.nn import functional as F

from .schema import CLASS_NAMES, IdentityScope, identity_scope_class_indices


IdentityLossVariant = Literal["oracle", "detached", "joint"]


@dataclass(frozen=True)
class LossWeights:
    """保存总损失非负系数：``classification`` 缩放六分类；``ligand_area`` 缩放配体区域；``binding_area`` 缩放结合区域；``identity`` 缩放闭集身份分类。"""

    classification: float
    ligand_area: float
    binding_area: float
    identity: float

    def __post_init__(self) -> None:
        """拒绝非有限值和负权重。"""

        if any(not math.isfinite(value) or value < 0.0 for value in self.__dict__.values()):
            raise ValueError("损失权重必须是有限非负数。")


class MatcherLossBatch(TypedDict):
    """声明损失函数实际读取的目标字段。

    ``class_target`` 和 ``identity_target`` 是 int64 ``(B,)``；后者以 ``-1``
    表示没有身份监督。``ligand_area_target`` 和 ``binding_area_target`` 是 bool
    ``(B,Z,Y,X)``。``identity_mask`` 是 bool ``(B,I)``，真值表示有效身份槽。
    ``identity_class_index`` 是 int64 ``(B,I)``；有效槽保存五类非背景下标，
    补零槽保存 -1，非负 identity_target 指向的类别必须等于 class_target。
    """

    class_target: torch.Tensor
    ligand_area_target: torch.Tensor
    binding_area_target: torch.Tensor
    identity_target: torch.Tensor
    identity_mask: torch.Tensor
    identity_class_index: torch.Tensor


class LossOutput(TypedDict):
    """损失输出：``total/classification/ligand_area/binding_area/identity`` 均是浮点标量；``classification_denominator`` 是六分类权重和浮点标量；``identity_eligible_count`` 是 int64 标量真实有效身份目标数。"""

    total: torch.Tensor
    classification: torch.Tensor
    ligand_area: torch.Tensor
    binding_area: torch.Tensor
    identity: torch.Tensor
    classification_denominator: torch.Tensor
    identity_eligible_count: torch.Tensor


def compute_matcher_loss(
    outputs: dict[str, torch.Tensor],
    batch: MatcherLossBatch,
    weights: LossWeights,
    identity_variant: IdentityLossVariant | None,
    identity_scope: IdentityScope | None,
    class_weights: torch.Tensor | None,
) -> LossOutput:
    """读取 ``outputs`` 的 ``class_logits`` ``(B,6)``、两个 ``(B,Z,Y,X)`` 区域分数和完整 Matcher 才需的 ``identity_logits`` ``(B,I)``，再与 ``MatcherLossBatch`` 对应目标计算 ``LossOutput``。

    身份项覆盖 ``identity_scope`` 中具有有效目标的候选。三个变体都除以真实
    有效候选数，不以目标类别概率权重和作分母。
    """

    per_candidate_classification = F.cross_entropy(
        outputs["class_logits"],
        batch["class_target"],
        weight=class_weights,
        reduction="none",
    )
    if class_weights is None:
        classification_denominator = outputs["class_logits"].new_tensor(
            batch["class_target"].numel()
        )
    else:
        classification_denominator = class_weights[batch["class_target"]].sum()
    classification = per_candidate_classification.sum() / classification_denominator.clamp_min(1.0)
    ligand_area = balanced_voxel_loss(
        outputs["ligand_area_logits"],
        batch["ligand_area_target"],
    )
    binding_area = balanced_voxel_loss(
        outputs["binding_area_logits"],
        batch["binding_area_target"],
    )
    base = (
        weights.classification * classification
        + weights.ligand_area * ligand_area
        + weights.binding_area * binding_area
    )
    identity = base.new_zeros(())
    eligible_count = torch.zeros((), device=base.device, dtype=torch.long)
    if identity_variant is not None:
        if identity_scope is None:
            raise ValueError("启用身份损失时必须指定 identity_scope。")
        if identity_variant not in ("oracle", "detached", "joint"):
            raise ValueError(f"不支持 identity_variant={identity_variant!r}。")
        if "identity_logits" not in outputs:
            raise KeyError("完整 Matcher 身份损失需要 identity_logits。")
        identity, eligible_count = _identity_loss(
            outputs["identity_logits"],
            outputs["class_logits"],
            batch,
            identity_variant,
            identity_scope,
        )
    elif identity_scope is not None:
        raise ValueError("未启用身份损失时 identity_scope 必须为 None。")
    total = base + weights.identity * identity
    return {
        "total": total,
        "classification": classification,
        "ligand_area": ligand_area,
        "binding_area": binding_area,
        "identity": identity,
        "classification_denominator": classification_denominator,
        "identity_eligible_count": eligible_count,
    }


def balanced_voxel_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """对同形 ``(B,Z,Y,X)`` logits/target 分样本平均正负体素 BCE 与软 Dice，返回批次标量均值。"""

    if logits.shape != target.shape or logits.ndim != 4:
        raise ValueError("区域 logits/target 必须具有相同的 (B,Z,Y,X) 形状。")
    target_float = target.to(dtype=logits.dtype)
    element = F.binary_cross_entropy_with_logits(logits, target_float, reduction="none")
    flat_loss = element.flatten(1)
    flat_target = target_float.flatten(1)
    positive_count = flat_target.sum(dim=1)
    negative_count = (1.0 - flat_target).sum(dim=1)
    positive = (flat_loss * flat_target).sum(dim=1) / positive_count.clamp_min(1.0)
    negative = (flat_loss * (1.0 - flat_target)).sum(dim=1) / negative_count.clamp_min(1.0)
    # 没有正体素的样本只使用负项，避免人为加入恒为零的“正类平均”。
    balanced_bce = torch.where(positive_count > 0, 0.5 * (positive + negative), negative)
    probability = torch.sigmoid(logits).flatten(1)
    intersection = (probability * flat_target).sum(dim=1)
    dice = 1.0 - (2.0 * intersection + 1.0) / (
        probability.sum(dim=1) + positive_count + 1.0
    )
    return (balanced_bce + dice).mean()


def _identity_loss(
    identity_logits: torch.Tensor,
    class_logits: torch.Tensor,
    batch: MatcherLossBatch,
    variant: IdentityLossVariant,
    identity_scope: IdentityScope,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按目标身份所属类别门控 ``(B,I)`` 交叉熵，返回损失与有效候选数。"""

    target = batch["identity_target"]
    identity_mask = batch["identity_mask"]
    if (
        identity_mask.dtype != torch.bool
        or identity_mask.ndim != 2
        or identity_mask.shape != identity_logits.shape
    ):
        raise ValueError("identity_mask 必须是与 identity_logits 同形的 bool (B,I) 张量。")
    identity_class_index = batch["identity_class_index"]
    if identity_class_index.shape != identity_mask.shape or identity_class_index.dtype != torch.long:
        raise ValueError("identity_class_index 必须是与 identity_mask 同形的 int64 张量。")
    if torch.any(identity_class_index[identity_mask] <= 0) or torch.any(
        identity_class_index[identity_mask] >= len(CLASS_NAMES)
    ):
        raise ValueError("有效身份必须属于非背景类别。")
    eligible = identity_eligible_mask(batch, identity_scope)
    count = eligible.sum()
    if not torch.any(eligible):
        # 保留身份分支计算图，使分布式或自动混合精度训练的零样本批次行为一致。
        return identity_logits[eligible].sum() * 0.0, count
    selected_logits = identity_logits[eligible]
    selected_target = target[eligible]
    if selected_logits.shape[1] == 0:
        raise ValueError("存在身份监督目标时 identity_logits 的身份轴不能为零。")
    if torch.any(selected_target >= selected_logits.shape[1]):
        raise ValueError("identity_target 超出 identity_logits 的身份轴。")
    if not torch.all(identity_mask[eligible, selected_target]):
        raise ValueError("identity_target 必须指向 identity_mask 中的有效身份。")
    selected_class = identity_class_index[eligible, selected_target]
    if not torch.equal(selected_class, batch["class_target"][eligible]):
        raise ValueError("identity_target 指向的身份类别必须等于候选真实六分类类别。")
    per_candidate = F.cross_entropy(selected_logits, selected_target, reduction="none")
    if variant == "oracle":
        multiplier = torch.ones_like(per_candidate)
    else:
        target_class_probability = torch.softmax(class_logits, dim=-1)[
            eligible,
            selected_class,
        ]
        multiplier = (
            target_class_probability.detach()
            if variant == "detached"
            else target_class_probability
        )
    return (multiplier * per_candidate).sum() / count.to(per_candidate.dtype), count


def identity_eligible_mask(
    batch: MatcherLossBatch,
    identity_scope: IdentityScope,
) -> torch.Tensor:
    """返回 ``(B,)`` 身份损失分母掩码，不读取模型预测。"""

    class_target = batch["class_target"]
    target = batch["identity_target"]
    if class_target.shape != target.shape or class_target.ndim != 1:
        raise ValueError("class_target 与 identity_target 必须是同形 (B,) 张量。")
    allowed = torch.zeros_like(class_target, dtype=torch.bool)
    for class_index in identity_scope_class_indices(identity_scope):
        allowed |= class_target == class_index
    return allowed & (target >= 0)
