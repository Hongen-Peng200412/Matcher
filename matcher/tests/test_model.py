from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from matcher.batching import collate_anchor_batch
import matcher.model as matcher_model
from matcher.model import FinePairBranch, Matcher, build_contact_marginals
from matcher.tests.sample_factory import sample as make_sample


def test_phase1_outputs_eight_independent_heads_and_supports_empty_A() -> None:
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
        map_candidate_chunk_size=1,
    )
    model.eval()

    output = model(collate_anchor_batch([make_sample()]))[0]
    assert len(output.block_outputs) == 8
    assert all(head is not None for head in output.block_outputs)
    assert output.block_outputs[-1].O_logit.shape == (2, 2)
    assert output.block_outputs[-1].O_prime.shape == (2, 2)
    assert model.prediction_heads[0].OHead is not model.prediction_heads[1].OHead


def test_phase1_supports_bfloat16_autocast() -> None:
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
        map_candidate_chunk_size=1,
    )
    model.train()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(collate_anchor_batch([make_sample()]))[0]
        loss = output.block_outputs[-1].O_logit.sum()
    loss.backward()

    assert output.block_outputs[-1].O_logit.dtype == torch.bfloat16
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_graph_checkpoint_preserves_outputs_and_gradients() -> None:
    kwargs = dict(
        phase=1,
        A_node_input_dim=7,
        node_dim=16,
        edge_dim=8,
        repr_dim=32,
        num_heads=4,
        coarse_layers_per_block=1,
        phase1_blocks=2,
        phase2_blocks=1,
        map_channels=(2, 4, 8, 8),
        map_norm_groups=2,
        map_candidate_chunk_size=1,
        dropout=0.0,
    )
    plain = Matcher(**kwargs, graph_activation_checkpoint=False)
    checkpointed = Matcher(**kwargs, graph_activation_checkpoint=True)
    checkpointed.load_state_dict(plain.state_dict())
    batch = collate_anchor_batch([make_sample()])

    plain_output = plain(batch)[0].block_outputs[-1].O_logit
    checkpointed_output = checkpointed(batch)[0].block_outputs[-1].O_logit
    plain_output.sum().backward()
    checkpointed_output.sum().backward()

    assert torch.allclose(plain_output, checkpointed_output)
    checkpointed_parameters = dict(checkpointed.named_parameters())
    for name, parameter in plain.named_parameters():
        checkpointed_gradient = checkpointed_parameters[name].grad
        if parameter.grad is None and checkpointed_gradient is None:
            continue
        assert parameter.grad is not None
        assert checkpointed_gradient is not None
        assert torch.allclose(parameter.grad, checkpointed_gradient)


def test_packed_graph_encoding_keeps_pdb_outputs_isolated() -> None:
    model = Matcher(
        phase=1,
        A_node_input_dim=7,
        node_dim=16,
        edge_dim=8,
        repr_dim=32,
        num_heads=4,
        coarse_layers_per_block=1,
        dropout=0.0,
        map_channels=(2, 4, 8, 8),
        map_norm_groups=2,
    ).eval()
    left = make_sample()
    right = replace(make_sample(), pdb_id="other", manifest_index=1)

    together = model(collate_anchor_batch([left, right]))
    separate = (
        model(collate_anchor_batch([left]))[0],
        model(collate_anchor_batch([right]))[0],
    )

    for packed_output, single_output in zip(together, separate, strict=True):
        assert torch.allclose(
            packed_output.block_outputs[-1].O_logit,
            single_output.block_outputs[-1].O_logit,
            atol=1.0e-6,
            rtol=1.0e-6,
        )


def test_phase2_freeze_keeps_map_trainable() -> None:
    model = Matcher(
        phase=2,
        A_node_input_dim=7,
        node_dim=16,
        edge_dim=8,
        repr_dim=32,
        num_heads=4,
        coarse_layers_per_block=1,
        map_channels=(2, 4, 8, 8),
        map_norm_groups=2,
    )
    model.freeze_phase1_for_phase2()
    model.train()

    assert not any(parameter.requires_grad for parameter in model.blocks[0].parameters())
    assert not model.blocks[0].training
    assert not model.prediction_heads[0].training
    assert all(parameter.requires_grad for parameter in model.MapBackbone.parameters())
    assert all(parameter.requires_grad for parameter in model.blocks[4].parameters())

    output = model(collate_anchor_batch([make_sample()]))[0]
    output.block_outputs[-1].O_logit.sum().backward()
    assert all(parameter.grad is None for parameter in model.blocks[0].parameters())
    assert any(parameter.grad is not None for parameter in model.MapBackbone.parameters())


def test_phase2_runs_fine_pairs_once_at_final_output() -> None:
    model = Matcher(
        phase=2,
        A_node_input_dim=7,
        node_dim=16,
        edge_dim=8,
        repr_dim=32,
        num_heads=4,
        coarse_layers_per_block=1,
        map_channels=(2, 4, 8, 8),
        map_norm_groups=2,
        fine_pair_chunk_size=2,
        fine_pair_activation_checkpoint=False,
        fine_pair_aux_weight=0.1,
        fine_readout_dim=24,
    )
    model.eval()
    sample = make_sample()
    sample = replace(
        sample,
        ligand_gt_coordinates=(torch.zeros(3, 3), torch.zeros(3, 3)),
        ligand_gt_present=(
            torch.ones(3, dtype=torch.bool),
            torch.ones(3, dtype=torch.bool),
        ),
    )

    fine_pair_batches: list[int] = []
    hook = model.fine_pair_branch.register_forward_hook(
        lambda _module, inputs, _output: fine_pair_batches.append(len(inputs[0]))
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(collate_anchor_batch([sample]))[0]
    hook.remove()

    assert output.block_outputs[:4] == (None, None, None, None)
    assert all(head is not None for head in output.block_outputs[4:])
    assert fine_pair_batches == [2, 2]
    assert torch.isfinite(output.block_outputs[-1].O_logit).all()
    assert output.fine_recall_pair_loss.shape == (2, 2, 2)
    assert output.fine_precision_pair_loss.shape == (2, 2, 2)
    assert output.fine_recall_pair_loss.dtype == torch.float32
    assert output.fine_precision_pair_loss.dtype == torch.float32


@pytest.mark.parametrize("use_ffn", [False, True])
def test_fine_pair_batch_masks_padding_and_empty_A(use_ffn: bool) -> None:
    torch.manual_seed(19)
    branch = FinePairBranch(
        node_dim=8,
        repr_dim=12,
        slot_repr_dim=12,
        num_heads=2,
        readout_dim=16,
        dropout=0.0,
        use_ffn=use_ffn,
        auxiliary_enabled=True,
    ).eval()
    ligand_atoms = torch.randn(3, 4, 8)
    A_atoms = torch.randn(3, 3, 8)
    ligand_mask = torch.tensor(
        [[True, True, False, False], [True, True, True, True], [True, False, False, False]]
    )
    A_mask = torch.tensor(
        [[True, True, True], [True, False, False], [False, False, False]]
    )
    slot_embedding = torch.randn(3, 12)

    expected = branch(ligand_atoms, A_atoms, ligand_mask, A_mask, slot_embedding)
    changed_ligand = ligand_atoms.masked_fill(~ligand_mask[:, :, None], 1000.0)
    changed_A = A_atoms.masked_fill(~A_mask[:, :, None], -1000.0)
    actual = branch(changed_ligand, changed_A, ligand_mask, A_mask, slot_embedding)

    assert torch.allclose(expected[0], actual[0], atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(expected[1][ligand_mask], actual[1][ligand_mask])
    assert torch.allclose(expected[2][A_mask], actual[2][A_mask])
    assert all(torch.isfinite(value).all() for value in actual)


def test_fine_pair_batch_matches_independent_pairs() -> None:
    torch.manual_seed(23)
    branch = FinePairBranch(
        node_dim=8,
        repr_dim=12,
        slot_repr_dim=12,
        num_heads=2,
        readout_dim=16,
        dropout=0.0,
        auxiliary_enabled=True,
    ).eval()
    ligand_atoms = torch.randn(3, 4, 8)
    A_atoms = torch.randn(3, 3, 8)
    ligand_mask = torch.tensor(
        [[True, True, False, False], [True, True, True, True], [True, False, False, False]]
    )
    A_mask = torch.tensor(
        [[True, True, True], [True, False, False], [False, False, False]]
    )
    slot_embedding = torch.randn(3, 12)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        batched = branch(ligand_atoms, A_atoms, ligand_mask, A_mask, slot_embedding)
    assert batched[0].dtype == torch.bfloat16
    for pair_index in range(3):
        ligand_width = int(ligand_mask[pair_index].sum())
        A_width = max(int(A_mask[pair_index].sum()), 1)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            independent = branch(
                ligand_atoms[pair_index : pair_index + 1, :ligand_width],
                A_atoms[pair_index : pair_index + 1, :A_width],
                ligand_mask[pair_index : pair_index + 1, :ligand_width],
                A_mask[pair_index : pair_index + 1, :A_width],
                slot_embedding[pair_index : pair_index + 1],
            )
        assert torch.allclose(batched[0][pair_index], independent[0][0], atol=2.0e-2)
        assert torch.allclose(
            batched[1][pair_index, :ligand_width], independent[1][0], atol=2.0e-2
        )
        assert torch.allclose(
            batched[2][pair_index, :A_width], independent[2][0], atol=2.0e-2
        )


def test_batched_contact_marginals_keep_candidate_and_occurrence_axes() -> None:
    A_coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    A_mask = torch.tensor([[True, True], [False, False]])
    ligand_coordinates = torch.tensor(
        [
            [[0.5, 0.0, 0.0], [20.0, 0.0, 0.0], [999.0, 0.0, 0.0]],
            [[9.0, 0.0, 0.0], [50.0, 0.0, 0.0], [60.0, 0.0, 0.0]],
        ]
    )
    ligand_present = torch.tensor([[True, True, False], [True, False, False]])

    recall, precision = build_contact_marginals(
        A_coordinates,
        A_mask,
        ligand_coordinates,
        ligand_present,
        pair_chunk_size=3,
    )

    assert torch.equal(
        recall,
        torch.tensor(
            [
                [[True, False, False], [True, False, False]],
                [[False, False, False], [False, False, False]],
            ]
        ),
    )
    assert torch.equal(
        precision,
        torch.tensor(
            [
                [[True, False], [False, True]],
                [[False, False], [False, False]],
            ]
        ),
    )


def test_fine_pair_auxiliary_keeps_all_same_identity_ground_truth_targets() -> None:
    model = Matcher(
        phase=2,
        A_node_input_dim=7,
        node_dim=16,
        edge_dim=8,
        repr_dim=32,
        num_heads=4,
        coarse_layers_per_block=1,
        dropout=0.0,
        map_channels=(2, 4, 8, 8),
        map_norm_groups=2,
        entity_aux_weight=0.0,
        fine_pair_aux_weight=0.1,
        fine_readout_dim=24,
        fine_pair_chunk_size=None,
        fine_pair_activation_checkpoint=False,
    ).eval()
    assert model.fine_pair_branch is not None
    with torch.no_grad():
        model.fine_pair_branch.recall_atom_head.weight.zero_()
        model.fine_pair_branch.recall_atom_head.bias.fill_(2.0)
        model.fine_pair_branch.precision_atom_head.weight.zero_()
        model.fine_pair_branch.precision_atom_head.bias.fill_(2.0)
    sample = replace(
        make_sample(),
        ligand_gt_coordinates=(torch.zeros(3, 3), torch.full((3, 3), 20.0)),
        ligand_gt_present=(
            torch.ones(3, dtype=torch.bool),
            torch.ones(3, dtype=torch.bool),
        ),
    )

    output = model(collate_anchor_batch([sample]))[0]

    # candidate 1 含 A 原子；两个预测槽位属于同一配体身份，均须面对两个真实 occurrence。
    for grid in (
        output.fine_recall_pair_loss[1],
        output.fine_precision_pair_loss[1],
    ):
        assert torch.all(grid > 0)
        assert torch.allclose(grid[0], grid[1])
        assert torch.all(grid[:, 0] != grid[:, 1])


@pytest.mark.skipif(
    not torch.cuda.is_available() or matcher_model.flash_attn_varlen_func is None,
    reason="需要 CUDA 与 flash-attn",
)
def test_cuda_varlen_flash_matches_sdpa_outputs_and_gradients(monkeypatch) -> None:
    torch.manual_seed(29)
    flash_branch = FinePairBranch(
        node_dim=32,
        repr_dim=48,
        slot_repr_dim=48,
        num_heads=4,
        readout_dim=64,
        dropout=0.0,
        auxiliary_enabled=True,
    ).cuda().train()
    sdpa_branch = copy.deepcopy(flash_branch)
    ligand_atoms = torch.randn(3, 5, 32, device="cuda")
    A_atoms = torch.randn(3, 4, 32, device="cuda")
    ligand_mask = torch.tensor(
        [[True, True, False, False, False], [True] * 5, [True, True, True, False, False]],
        device="cuda",
    )
    A_mask = torch.tensor(
        [[True, True, True, False], [True, False, False, False], [False] * 4],
        device="cuda",
    )
    slot_embedding = torch.randn(3, 48, device="cuda")
    actual_flash = matcher_model.flash_attn_varlen_func
    flash_calls: list[int] = []

    def counted_flash(*args, **kwargs):
        flash_calls.append(len(args[0]))
        return actual_flash(*args, **kwargs)

    monkeypatch.setattr(matcher_model, "flash_attn_varlen_func", counted_flash)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        flash_output = flash_branch(
            ligand_atoms, A_atoms, ligand_mask, A_mask, slot_embedding
        )
        sum(value.float().sum() for value in flash_output).backward()
    monkeypatch.setattr(matcher_model, "flash_attn_varlen_func", None)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sdpa_output = sdpa_branch(
            ligand_atoms, A_atoms, ligand_mask, A_mask, slot_embedding
        )
        sum(value.float().sum() for value in sdpa_output).backward()

    assert flash_calls
    for flash_value, sdpa_value in zip(flash_output, sdpa_output, strict=True):
        torch.testing.assert_close(flash_value, sdpa_value, atol=5.0e-2, rtol=5.0e-2)
    sdpa_parameters = dict(sdpa_branch.named_parameters())
    for name, parameter in flash_branch.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            sdpa_parameters[name].grad,
            atol=7.0e-2,
            rtol=7.0e-2,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_fine_pair_chunk_and_checkpoint_are_numerically_equivalent() -> None:
    torch.manual_seed(11)
    base = Matcher(
        phase=2,
        A_node_input_dim=7,
        node_dim=16,
        edge_dim=8,
        repr_dim=32,
        num_heads=4,
        coarse_layers_per_block=1,
        dropout=0.0,
        map_channels=(2, 4, 8, 8),
        map_norm_groups=2,
        entity_aux_weight=0.0,
        fine_pair_aux_weight=0.1,
        fine_readout_dim=24,
        fine_pair_chunk_size=None,
        fine_pair_activation_checkpoint=False,
    )
    chunked = Matcher(
        phase=2,
        A_node_input_dim=7,
        node_dim=16,
        edge_dim=8,
        repr_dim=32,
        num_heads=4,
        coarse_layers_per_block=1,
        dropout=0.0,
        map_channels=(2, 4, 8, 8),
        map_norm_groups=2,
        entity_aux_weight=0.0,
        fine_pair_aux_weight=0.1,
        fine_readout_dim=24,
        fine_pair_chunk_size=2,
        fine_pair_activation_checkpoint=False,
    )
    checked = Matcher(
        phase=2,
        A_node_input_dim=7,
        node_dim=16,
        edge_dim=8,
        repr_dim=32,
        num_heads=4,
        coarse_layers_per_block=1,
        dropout=0.0,
        map_channels=(2, 4, 8, 8),
        map_norm_groups=2,
        entity_aux_weight=0.0,
        fine_pair_aux_weight=0.1,
        fine_readout_dim=24,
        fine_pair_chunk_size=2,
        fine_pair_activation_checkpoint=True,
    )
    chunked.load_state_dict(base.state_dict())
    checked.load_state_dict(base.state_dict())
    sample = replace(
        make_sample(),
        ligand_gt_coordinates=(torch.zeros(3, 3), torch.ones(3, 3)),
        ligand_gt_present=(
            torch.ones(3, dtype=torch.bool),
            torch.ones(3, dtype=torch.bool),
        ),
    )
    batch = collate_anchor_batch([sample])

    def run(model: Matcher) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        model.freeze_phase1_for_phase2()
        model.train()
        output = model(batch)[0]
        final = output.block_outputs[-1]
        value = (
            final.O_logit.sum()
            + final.O_prime.sum()
            + output.fine_recall_pair_loss.sum()
            + output.fine_precision_pair_loss.sum()
        )
        value.backward()
        gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return final.O_logit.detach(), output.fine_recall_pair_loss.detach(), gradients

    plain_O, plain_aux, plain_grad = run(base)
    for model in (chunked, checked):
        other_O, other_aux, other_grad = run(model)
        assert torch.allclose(plain_O, other_O, atol=1.0e-6, rtol=1.0e-6)
        assert torch.allclose(plain_aux, other_aux, atol=1.0e-6, rtol=1.0e-6)
        assert other_grad.keys() == plain_grad.keys()
        for name in plain_grad:
            assert torch.allclose(
                plain_grad[name], other_grad[name], atol=2.0e-5, rtol=2.0e-5
            ), name
