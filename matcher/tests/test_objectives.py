from __future__ import annotations

from dataclasses import replace

import torch

from matcher.model import HeadOutput
from matcher.batching import collate_anchor_batch
from matcher.objectives import _head_losses, identity_hungarian_assignment, matcher_loss
from matcher.tests.sample_factory import sample as make_sample


def test_identity_hungarian_can_swap_repeated_occurrence_slots() -> None:
    sample = make_sample()
    sample = replace(
        sample,
        O_target=torch.tensor([[True, False], [False, True]]),
        O_prime_target=torch.tensor([[0.9, 0.2], [0.2, 0.9]]),
        O_prime_exact=torch.ones(2, 2, dtype=torch.bool),
    )
    O_logit = torch.tensor([[-8.0, 8.0], [8.0, -8.0]])
    O_prime = torch.tensor([[0.2, 0.9], [0.9, 0.2]])
    output = HeadOutput(
        coarse_pair_repr=torch.empty(2, 2, 3),
        O_logit=O_logit,
        O_prime_logit=torch.logit(O_prime),
        O_prime=O_prime,
    )

    assignment = identity_hungarian_assignment(output, sample)

    assert assignment.tolist() == [1, 0]


def test_formal_O_loss_adds_focal_and_dice_without_halving() -> None:
    sample = replace(
        make_sample(),
        O_target=torch.tensor([[True, False], [False, False]]),
        O_prime_target=torch.full((2, 2), 0.5),
        O_prime_exact=torch.ones(2, 2, dtype=torch.bool),
    )
    O_logit = torch.zeros(2, 2)
    head = HeadOutput(torch.empty(2, 2, 3), O_logit, torch.zeros(2, 2), torch.full((2, 2), 0.5))

    loss_O, _ = _head_losses(head, sample, torch.arange(2), 0.2)
    focal_positive = 0.25 * torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros(2), torch.tensor([1.0, 0.0])
    )
    dice_positive = 1 - 2.0 / 3.0
    focal_empty = 0.25 * torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros(2), torch.zeros(2)
    )
    assert torch.allclose(loss_O, focal_positive + dice_positive + focal_empty)


def test_fine_auxiliary_uses_hungarian_aligned_gt_occurrence() -> None:
    from matcher.model import Matcher

    model = Matcher(
        phase=1,
        A_node_input_dim=7,
        node_dim=16,
        edge_dim=8,
        repr_dim=32,
        num_heads=4,
        coarse_layers_per_block=1,
        map_channels=(2, 4, 8, 8),
        map_norm_groups=2,
    ).eval()
    sample = replace(
        make_sample(),
        O_target=torch.tensor([[True, False], [False, True]]),
        O_prime_target=torch.tensor([[0.9, 0.2], [0.2, 0.9]]),
        O_prime_exact=torch.ones(2, 2, dtype=torch.bool),
    )
    output = model(collate_anchor_batch([sample]))[0]
    O_logit = torch.tensor([[-8.0, 8.0], [8.0, -8.0]])
    O_prime = torch.tensor([[0.2, 0.9], [0.9, 0.2]])
    final_head = HeadOutput(
        torch.empty(2, 2, 3), O_logit, torch.logit(O_prime), O_prime
    )
    fine_recall = torch.tensor(
        [[[10.0, 1.0], [2.0, 20.0]], [[10.0, 1.0], [2.0, 20.0]]]
    )
    output = replace(
        output,
        block_outputs=(*output.block_outputs[:-1], final_head),
        fine_recall_pair_loss=fine_recall,
        fine_precision_pair_loss=torch.zeros_like(fine_recall),
    )

    loss = matcher_loss(
        (output,),
        (sample,),
        phase=2,
        phase1_blocks=7,
        entity_aux_weight=0.0,
        fine_pair_aux_weight=1.0,
    )

    assert loss.final_assignment[0].tolist() == [1, 0]
    assert torch.allclose(loss.loss_aux, torch.tensor(3.0 / 64.0))
