"""用 ``compute_matcher_loss`` 计算候选基础损失与三种身份门控，并以 ``LossOutput`` 返回可记录分量。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import torch
from torch.nn import functional as F

from .contracts import CandidateBatch
from .schema import SMALL_MOLECULE_INDEX


IdentityLossVariant = Literal["oracle", "detached", "joint"]


@dataclass(frozen=True)
class LossWeights:
    """保存总损失非负系数：``classification`` 缩放六分类；``ligand_area`` 缩放配体区域；``binding_area`` 缩放结合区域；``identity`` 缩放闭集身份分类。"""

    classification: float
    ligand_area: float
    binding_area: float
    identity: float

    def __post_init__(self) -> None:
        """拒绝负权重。"""

        if any(value < 0.0 for value in self.__dict__.values()):
            raise ValueError("损失权重不能为负数。")


class LossOutput(TypedDict):
    """损失输出：``total`` 是加权总损失；``classification`` 是六分类损失；``ligand_area`` 是配体区域损失；``binding_area`` 是结合区域损失；``identity`` 是门控后身份损失；``identity_eligible_count`` 是有效真实小分子数。"""

    total: torch.Tensor
    classification: torch.Tensor
    ligand_area: torch.Tensor
    binding_area: torch.Tensor
    identity: torch.Tensor
    classification_denominator: torch.Tensor
    identity_eligible_count: torch.Tensor


def compute_matcher_loss(
    outputs: dict[str, torch.Tensor],
    batch: CandidateBatch,
    weights: LossWeights,
    identity_variant: IdentityLossVariant | None,
    class_weights: torch.Tensor | None,
) -> LossOutput:
    """由候选输出与 ``CandidateBatch`` 计算 ``LossOutput``，并按 oracle/detached/joint 规则选择性加入身份损失。

    身份项只覆盖真实六分类标签为 small_molecule 且当前语言模型存在目标表示的
    候选。三个变体都除以这类候选的真实数量，不以 ``p_small`` 权重和作分母。
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
        if identity_variant not in ("oracle", "detached", "joint"):
            raise ValueError(f"不支持 identity_variant={identity_variant!r}。")
        if "identity_logits" not in outputs:
            raise KeyError("完整 Matcher 身份损失需要 identity_logits。")
        identity, eligible_count = _identity_loss(
            outputs["identity_logits"],
            outputs["class_logits"],
            batch,
            identity_variant,
        )
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
    """对同形 ``(B,D,H,W)`` logits/target 分样本平均正负体素 BCE 与软 Dice，返回批次标量均值。"""

    if logits.shape != target.shape or logits.ndim != 4:
        raise ValueError("区域 logits/target 必须具有相同的 (B,D,H,W) 形状。")
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
    batch: CandidateBatch,
    variant: IdentityLossVariant,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 ``(B,I)`` 身份分数实现三种门控交叉熵，返回损失标量与真实有效小分子计数。"""

    target = batch["identity_target"]
    eligible = (batch["class_target"] == SMALL_MOLECULE_INDEX) & (target >= 0)
    count = eligible.sum()
    if not torch.any(eligible):
        # 保留身份分支计算图，使分布式或自动混合精度训练的零样本批次行为一致。
        return identity_logits[eligible].sum() * 0.0, count
    selected_logits = identity_logits[eligible]
    selected_target = target[eligible]
    if selected_logits.shape[1] == 0:
        raise ValueError("存在身份监督目标时 identity_logits 的身份轴不能为零。")
    per_candidate = F.cross_entropy(selected_logits, selected_target, reduction="none")
    if variant == "oracle":
        multiplier = torch.ones_like(per_candidate)
    else:
        p_small = torch.softmax(class_logits, dim=-1)[eligible, SMALL_MOLECULE_INDEX]
        multiplier = p_small.detach() if variant == "detached" else p_small
    return (multiplier * per_candidate).sum() / count.to(per_candidate.dtype), count
