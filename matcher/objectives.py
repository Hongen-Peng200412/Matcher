"""Matcher 的身份内 Hungarian、O/O′ 主损失和两类辅助损失。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.nn import functional as F

from .anchor_data import AnchorSample
from .model import EntityAuxiliaryOutput, HeadOutput, PDBMatcherOutput


@dataclass(frozen=True)
class MatcherLoss:
    """四个满足直接相加关系的正式日志标量和最终出口对齐。"""

    loss_total: Tensor
    loss_O: Tensor
    loss_O_prime: Tensor
    loss_aux: Tensor
    final_assignment: tuple[Tensor, ...]


def sigmoid_focal_mean(logit: Tensor, target: Tensor, gamma: float = 2.0) -> Tensor:
    """计算无 alpha 的二值 sigmoid focal 平均。"""

    target_float = target.float()
    cross_entropy = F.binary_cross_entropy_with_logits(
        logit.float(), target_float, reduction="none"
    )
    probability = torch.sigmoid(logit.float())
    target_probability = probability * target_float + (1 - probability) * (1 - target_float)
    return (((1 - target_probability) ** gamma) * cross_entropy).mean()


def _O_cost(O_logit: Tensor, O_target: Tensor) -> Tensor:
    probability = torch.sigmoid(O_logit.float())
    target = O_target.float()
    positive_probability = probability[:, :, None]
    target_columns = target[:, None, :]
    target_probability = (
        positive_probability * target_columns
        + (1 - positive_probability) * (1 - target_columns)
    )
    cross_entropy = -(
        target_columns * torch.log(positive_probability.clamp_min(1.0e-8))
        + (1 - target_columns) * torch.log((1 - positive_probability).clamp_min(1.0e-8))
    )
    focal = (((1 - target_probability) ** 2) * cross_entropy).mean(dim=0)
    intersection = (positive_probability * target_columns).sum(dim=0)
    dice = 1 - (2 * intersection + 1) / (
        positive_probability.sum(dim=0) + target_columns.sum(dim=0) + 1
    )
    return 0.5 * (focal + dice)


def _O_prime_cost(
    O_prime: Tensor,
    O_prime_target: Tensor,
    O_prime_exact: Tensor,
    truncation_value: float,
) -> Tensor:
    prediction = O_prime.float()[:, :, None]
    target = O_prime_target.float()[:, None, :]
    exact = O_prime_exact[:, None, :]
    error = torch.where(
        exact,
        (prediction - target).abs(),
        torch.relu(prediction - truncation_value),
    )
    return error.mean(dim=0)


@torch.no_grad()
def identity_hungarian_assignment(
    head: HeadOutput,
    sample: AnchorSample,
    *,
    truncation_value: float = 0.2,
) -> Tensor:
    """返回 ``predicted_slot -> true occurrence`` 的完整列映射。"""

    device = head.O_logit.device
    occurrence_to_ligand = sample.occurrence_to_ligand.to(device)
    O_target = sample.O_target.to(device)
    O_prime_target = sample.O_prime_target.to(device)
    O_prime_exact = sample.O_prime_exact.to(device)
    assignment = torch.empty(sample.num_occurrences, dtype=torch.long, device=device)
    for ligand_index in range(len(sample.ligands)):
        occurrence_indices = torch.nonzero(
            occurrence_to_ligand == ligand_index, as_tuple=False
        ).flatten()
        O_cost = _O_cost(
            head.O_logit[:, occurrence_indices], O_target[:, occurrence_indices]
        )
        O_prime_cost = _O_prime_cost(
            head.O_prime[:, occurrence_indices],
            O_prime_target[:, occurrence_indices],
            O_prime_exact[:, occurrence_indices],
            truncation_value,
        )
        predicted, target = linear_sum_assignment((O_cost + O_prime_cost).cpu().numpy())
        assignment[occurrence_indices[torch.as_tensor(predicted, device=device)]] = occurrence_indices[
            torch.as_tensor(target, device=device)
        ]
    return assignment


def _head_losses(
    head: HeadOutput,
    sample: AnchorSample,
    assignment: Tensor,
    truncation_value: float,
) -> tuple[Tensor, Tensor]:
    device = head.O_logit.device
    O_target = sample.O_target.to(device)[:, assignment]
    O_prime_target = sample.O_prime_target.to(device)[:, assignment]
    O_prime_exact = sample.O_prime_exact.to(device)[:, assignment]
    losses_O = []
    losses_O_prime = []
    for occurrence_index in range(sample.num_occurrences):
        target = O_target[:, occurrence_index]
        focal = sigmoid_focal_mean(head.O_logit[:, occurrence_index], target)
        if target.any():
            probability = torch.sigmoid(head.O_logit[:, occurrence_index].float())
            intersection = (probability * target.float()).sum()
            dice = 1 - (2 * intersection + 1) / (
                probability.sum() + target.float().sum() + 1
            )
        else:
            dice = focal.new_zeros(())
        losses_O.append(focal + dice)
        exact = O_prime_exact[:, occurrence_index]
        error = torch.where(
            exact,
            (head.O_prime[:, occurrence_index].float() - O_prime_target[:, occurrence_index]).abs(),
            torch.relu(head.O_prime[:, occurrence_index].float() - truncation_value),
        )
        losses_O_prime.append(error.mean())
    return torch.stack(losses_O).sum(), torch.stack(losses_O_prime).sum()


def _multiclass_focal_mean(logit: Tensor, target: Tensor) -> Tensor:
    if logit.shape[0] == 0:
        return logit.new_zeros(())
    cross_entropy = F.cross_entropy(logit.float(), target, reduction="none")
    return (((1 - torch.exp(-cross_entropy)) ** 2) * cross_entropy).mean()


def _entity_losses(
    output: EntityAuxiliaryOutput,
    node_ptr: tuple[int, ...],
    edge_ptr: tuple[int, ...],
    element: Tensor,
    edge_class: Tensor,
    binding_atom: Tensor | None,
) -> Tensor:
    losses = []
    for entity_index in range(len(node_ptr) - 1):
        node_start, node_end = node_ptr[entity_index : entity_index + 2]
        edge_start, edge_end = edge_ptr[entity_index : entity_index + 2]
        if node_start == node_end:
            losses.append(output.element_logit.new_zeros(()))
            continue
        parts = [
            F.cross_entropy(
                output.element_logit[node_start:node_end].float(),
                element[node_start:node_end],
            ),
            _multiclass_focal_mean(
                output.edge_class_logit[edge_start:edge_end],
                edge_class[edge_start:edge_end],
            ),
        ]
        if output.binding_logit is not None and binding_atom is not None:
            parts.append(
                sigmoid_focal_mean(
                    output.binding_logit[node_start:node_end],
                    binding_atom[node_start:node_end],
                )
            )
        losses.append(torch.stack(parts).mean())
    return torch.stack(losses)


def _deep_supervision_weights(active_indices: list[int], total_weight: float) -> dict[int, float]:
    auxiliary = active_indices[:-1]
    denominator = sum(range(1, len(auxiliary) + 1))
    result = (
        {
            index: total_weight * position / denominator
            for position, index in enumerate(auxiliary, start=1)
        }
        if auxiliary
        else {}
    )
    result[active_indices[-1]] = 1.0
    return result


def matcher_loss(
    outputs: tuple[PDBMatcherOutput, ...],
    samples: tuple[AnchorSample, ...],
    *,
    phase: int,
    phase1_blocks: int = 4,
    occurrence_budget_per_batch: int = 64,
    coarse_deep_total_weight: float = 0.5,
    coarse_deep_schedule: str = "linear_depth",
    lambda_O: float = 1.0,
    lambda_O_prime: float = 1.0,
    entity_aux_weight: float = 0.1,
    fine_pair_aux_weight: float = 0.0,
    O_prime_truncation_value: float = 0.2,
) -> MatcherLoss:
    """按 PDB 隔离计算全部损失，最后统一除以固定 occurrence 预算。"""

    if coarse_deep_schedule != "linear_depth":
        raise ValueError(f"未知 coarse 深监督排程：{coarse_deep_schedule}")
    num_blocks = len(outputs[0].block_outputs)
    active_indices = (
        list(range(num_blocks)) if phase == 1 else list(range(phase1_blocks, num_blocks))
    )
    weights = _deep_supervision_weights(active_indices, coarse_deep_total_weight)
    reference = next(
        head.O_logit for output in outputs for head in output.block_outputs if head is not None
    )
    loss_O = reference.new_zeros(())
    loss_O_prime = reference.new_zeros(())
    loss_aux = reference.new_zeros(())
    final_assignments = []
    for output, sample in zip(outputs, samples, strict=True):
        for block_index in active_indices:
            head = output.block_outputs[block_index]
            assignment = identity_hungarian_assignment(
                head, sample, truncation_value=O_prime_truncation_value
            )
            block_loss_O, block_loss_O_prime = _head_losses(
                head, sample, assignment, O_prime_truncation_value
            )
            loss_O = loss_O + weights[block_index] * block_loss_O
            loss_O_prime = loss_O_prime + weights[block_index] * block_loss_O_prime
            if block_index == active_indices[-1]:
                final_assignments.append(assignment)

        if entity_aux_weight > 0:
            ligand_entity_loss = _entity_losses(
                output.ligand_auxiliary,
                output.ligand_entities.node_ptr,
                output.ligand_entities.edge_ptr,
                output.ligand_entities.element,
                output.ligand_entities.edge_class,
                None,
            )
            A_entity_loss = _entity_losses(
                output.A_auxiliary,
                output.A_entities.node_ptr,
                output.A_entities.edge_ptr,
                output.A_entities.element,
                output.A_entities.edge_class,
                output.A_entities.binding_atom,
            )
            occurrence_to_ligand = sample.occurrence_to_ligand.to(reference.device)
            loss_aux = loss_aux + entity_aux_weight * (
                ligand_entity_loss[occurrence_to_ligand].sum()
                + sample.num_occurrences * A_entity_loss.mean()
            )
        if fine_pair_aux_weight > 0:
            assignment = final_assignments[-1]
            predicted_slot = torch.arange(sample.num_occurrences, device=reference.device)
            recall = output.fine_recall_pair_loss[:, predicted_slot, assignment]
            precision = output.fine_precision_pair_loss[:, predicted_slot, assignment]
            loss_aux = loss_aux + fine_pair_aux_weight * (
                recall.mean(dim=0) + precision.mean(dim=0)
            ).sum()

    divisor = float(occurrence_budget_per_batch)
    loss_O = lambda_O * loss_O / divisor
    loss_O_prime = lambda_O_prime * loss_O_prime / divisor
    loss_aux = loss_aux / divisor
    return MatcherLoss(
        loss_O + loss_O_prime + loss_aux,
        loss_O,
        loss_O_prime,
        loss_aux,
        tuple(final_assignments),
    )
