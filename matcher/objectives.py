"""六项 A/B/O 监督、身份内 Hungarian、实体与细配对辅助损失。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.nn import functional as F

from .contracts import MatcherSample
from .model import MatcherHeadOutput, PDBMatcherOutput


@dataclass(frozen=True)
class MatcherLoss:
    """训练总损失与少量直接用于日志的分项。"""

    total: Tensor
    continuous: Tensor
    hard: Tensor
    entity_auxiliary: Tensor
    fine_pair_auxiliary: Tensor


def sigmoid_focal_mean(logit: Tensor, target: Tensor, gamma: float = 2.0) -> Tensor:
    """数值稳定的二分类 focal loss。"""

    target_float = target.float()
    cross_entropy = F.binary_cross_entropy_with_logits(logit.float(), target_float, reduction="none")
    probability = torch.sigmoid(logit.float())
    probability_true = probability * target_float + (1.0 - probability) * (1.0 - target_float)
    return (((1.0 - probability_true) ** gamma) * cross_entropy).mean()


def _pair_cost(head: MatcherHeadOutput, sample: MatcherSample) -> Tensor:
    """返回 `[S_pred,S_true]` 成本；六项只按候选数做相同均值。"""

    continuous_prediction = torch.stack((head.A, head.B, head.O))
    continuous_target = torch.stack(
        (sample.A_target, sample.B_target, sample.O_target)
    ).to(continuous_prediction)
    hard_logit = torch.stack(
        (head.A_prime_logit, head.B_prime_logit, head.O_prime_logit)
    )
    hard_target = torch.stack(
        (sample.A_prime_target, sample.B_prime_target, sample.O_prime_target)
    ).to(hard_logit)
    continuous = (continuous_prediction[:, :, :, None] - continuous_target[:, :, None, :]).abs()
    hard_ce = F.binary_cross_entropy_with_logits(
        hard_logit[:, :, :, None].expand(-1, -1, -1, hard_target.shape[-1]),
        hard_target[:, :, None, :].expand(-1, -1, hard_logit.shape[-1], -1).float(),
        reduction="none",
    )
    probability = torch.sigmoid(hard_logit)[:, :, :, None]
    target = hard_target[:, :, None, :].float()
    probability_true = probability * target + (1.0 - probability) * (1.0 - target)
    hard = ((1.0 - probability_true) ** 2) * hard_ce
    return continuous.mean(dim=(0, 1)) + hard.mean(dim=(0, 1))


@torch.no_grad()
def identity_hungarian_assignment(head: MatcherHeadOutput, sample: MatcherSample) -> Tensor:
    """仅在相同配体身份的槽位中求 `predicted_slot -> true_occurrence`。"""

    device = head.A_logit.device
    identities = sample.occurrence_to_ligand.to(device)
    cost = _pair_cost(head, sample)
    assignment = torch.empty(sample.num_occurrences, dtype=torch.long, device=device)
    for ligand_index in range(len(sample.ligands)):
        indices = torch.nonzero(identities == ligand_index, as_tuple=False).flatten()
        predicted, target = linear_sum_assignment(cost[indices][:, indices].cpu().numpy())
        assignment[indices[torch.as_tensor(predicted, device=device)]] = indices[
            torch.as_tensor(target, device=device)
        ]
    return assignment


def _head_loss(head: MatcherHeadOutput, sample: MatcherSample, assignment: Tensor) -> tuple[Tensor, Tensor]:
    continuous_prediction = (head.A, head.B, head.O)
    continuous_target = (
        sample.A_target.to(head.A_logit.device)[:, assignment],
        sample.B_target.to(head.B_logit.device)[:, assignment],
        sample.O_target.to(head.O_logit.device)[:, assignment],
    )
    hard_prediction = (head.A_prime_logit, head.B_prime_logit, head.O_prime_logit)
    hard_target = (
        sample.A_prime_target.to(head.A_prime_logit.device)[:, assignment],
        sample.B_prime_target.to(head.B_prime_logit.device)[:, assignment],
        sample.O_prime_target.to(head.O_prime_logit.device)[:, assignment],
    )
    continuous = sum(
        F.smooth_l1_loss(prediction.float(), target.to(prediction).float(), reduction="sum")
        for prediction, target in zip(continuous_prediction, continuous_target, strict=True)
    ) / max(1, sample.num_candidates)
    hard = sum(
        sigmoid_focal_mean(prediction, target.to(prediction)) * sample.num_occurrences
        for prediction, target in zip(hard_prediction, hard_target, strict=True)
    )
    return continuous, hard


def _entity_loss(output: PDBMatcherOutput) -> Tensor:
    values = []
    for auxiliary, entities in (
        (output.ligand_auxiliary, output.ligand_entities),
        (output.A_auxiliary, output.A_entities),
    ):
        if auxiliary is None:
            continue
        values.append(F.cross_entropy(auxiliary.element_logit.float(), entities.element))
        if len(entities.edge_class):
            values.append(F.cross_entropy(auxiliary.edge_class_logit.float(), entities.edge_class))
        if auxiliary.binding_logit is not None and entities.binding_atom is not None:
            values.append(sigmoid_focal_mean(auxiliary.binding_logit, entities.binding_atom))
    reference = output.ligand_state.node
    return torch.stack(values).mean() if values else reference.new_zeros(())


def matcher_loss(
    outputs: tuple[PDBMatcherOutput, ...],
    samples: tuple[MatcherSample, ...],
    *,
    phase: int,
    phase1_blocks: int = 4,
    occurrence_budget_per_batch: int = 64,
    deep_supervision_weight: float = 0.5,
    continuous_weight: float = 1.0,
    hard_weight: float = 1.0,
    entity_auxiliary_weight: float = 0.1,
    fine_pair_auxiliary_weight: float = 0.1,
) -> MatcherLoss:
    """先按 PDB 隔离计算，再除以固定预期 occurrence 预算。"""

    reference = outputs[0].ligand_state.node
    continuous = reference.new_zeros(())
    hard = reference.new_zeros(())
    entity = reference.new_zeros(())
    fine = reference.new_zeros(())
    for output, sample in zip(outputs, samples, strict=True):
        active = [
            index for index, value in enumerate(output.block_outputs)
            if value is not None and (phase == 1 or index >= phase1_blocks)
        ]
        final_assignment = None
        for position, block_index in enumerate(active):
            head = output.block_outputs[block_index]
            assignment = identity_hungarian_assignment(head, sample)
            weight = 1.0 if position == len(active) - 1 else deep_supervision_weight / max(1, len(active) - 1)
            block_continuous, block_hard = _head_loss(head, sample, assignment)
            continuous = continuous + weight * block_continuous
            hard = hard + weight * block_hard
            final_assignment = assignment
        entity = entity + _entity_loss(output)
        if final_assignment is not None and output.fine_recall_pair_loss is not None:
            slot = torch.arange(sample.num_occurrences, device=final_assignment.device)
            candidate_divisor = max(1, sample.num_candidates)
            fine = fine + output.fine_recall_pair_loss[:, slot, final_assignment].sum() / candidate_divisor
            fine = fine + output.fine_precision_pair_loss[:, slot, final_assignment].sum() / candidate_divisor
    divisor = float(occurrence_budget_per_batch)
    continuous = continuous / divisor
    hard = hard / divisor
    entity = entity / max(1, len(samples))
    fine = fine / divisor
    total = (
        continuous_weight * continuous + hard_weight * hard
        + entity_auxiliary_weight * entity + fine_pair_auxiliary_weight * fine
    )
    return MatcherLoss(total, continuous, hard, entity, fine)
