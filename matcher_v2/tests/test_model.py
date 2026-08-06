"""验证双模式共同模型的六头、二阶段和严格零增量。"""

from __future__ import annotations

from dataclasses import replace

import torch

from matcher.tests.sample_factory import raw_graph

from matcher_v2.batching import collate_matcher_batch
from matcher_v2.contracts import LigandInput, MatcherSample, PocketInput, Stage1Context
from matcher_v2.model import Matcher, MatcherHeadOutput
from matcher_v2.objectives import identity_hungarian_assignment
from matcher_v2.train import load_phase1


def _sample(stage1: Stage1Context | None = None) -> MatcherSample:
    A_graph = raw_graph(2, 7, binding=True)
    pocket = PocketInput(
        center_xyz=torch.zeros(3),
        density=torch.randn(1, 48, 48, 48),
        map_start_zyx=torch.zeros(3, dtype=torch.int32),
        voxel_size_xyz=torch.ones(3),
        origin_xyz=torch.zeros(3),
        A_graph=A_graph,
        A_global_index=torch.arange(2),
        candidate_mask_zyx=torch.zeros(1, 3, dtype=torch.int32),
        rotation_axes_zyx=None,
        rotation_k=0,
        stage1=stage1,
    )
    one = torch.ones(1, 1)
    return MatcherSample(
        pdb_id="test",
        manifest_index=0,
        ligands=(LigandInput("CCD:TST", raw_graph(3, 149, binding=False)),),
        pockets=(pocket,),
        occurrence_to_ligand=torch.zeros(1, dtype=torch.long),
        occurrence_candidate_id=torch.zeros(1, dtype=torch.long),
        occurrence_centroid_xyz=torch.zeros(1, 3),
        occurrence_coordinates=(torch.zeros(3, 3),),
        occurrence_present=(torch.ones(3, dtype=torch.bool),),
        A_target=one,
        B_target=one,
        O_target=one,
        A_prime_target=one.bool(),
        B_prime_target=one.bool(),
        O_prime_target=one.bool(),
    )


def _stage1() -> Stage1Context:
    return Stage1Context(
        V_coord_local_xyz=torch.ones(2, 3),
        V_probability=torch.ones(2),
        V_feature=torch.ones(2, 2),
        P_coord_local_xyz=torch.ones(2, 3),
        P_probability=torch.ones(2),
        P_feature=torch.ones(2, 3),
        PP_coord_local_xyz=torch.ones(2, 3),
        PP_probability=torch.ones(2),
        A_probability=torch.ones(2),
        A_feature=torch.ones(2, 4),
    )


def _model(phase: int, *, fine_auxiliary: float = 0.0) -> Matcher:
    return Matcher(
        phase=phase,
        A_node_input_dim=7,
        node_dim=16,
        edge_dim=8,
        repr_dim=32,
        num_heads=4,
        coarse_layers_per_block=1,
        coarse_ffn_dim=64,
        phase1_blocks=1,
        phase2_blocks=1,
        map_channels=(2, 4, 8, 8),
        map_norm_groups=2,
        map_candidate_chunk_size=1,
        fine_readout_dim=32,
        fine_pair_chunk_size=4,
        fine_pair_aux_weight=fine_auxiliary,
        V_feature_dim=2,
        P_feature_dim=3,
        A_feature_dim=4,
        dropout=0.0,
    )


def test_phase1_stage1_context_is_initially_exact_zero_increment() -> None:
    model = _model(1).eval()
    plain_sample = _sample()
    conditioned_sample = replace(
        plain_sample,
        pockets=(replace(plain_sample.pockets[0], stage1=_stage1()),),
    )
    plain = model(collate_matcher_batch([plain_sample]))[0].block_outputs[-1]
    conditioned = model(collate_matcher_batch([conditioned_sample]))[0].block_outputs[-1]

    for name in (
        "A_logit", "B_logit", "O_logit",
        "A_prime_logit", "B_prime_logit", "O_prime_logit",
    ):
        assert torch.equal(getattr(plain, name), getattr(conditioned, name))


def test_phase2_loads_phase1_and_all_six_heads_receive_fine_gradient(tmp_path) -> None:
    phase1 = _model(1)
    checkpoint = tmp_path / "phase1.pt"
    torch.save({"model": phase1.state_dict()}, checkpoint)
    phase2 = _model(2, fine_auxiliary=0.1)
    load_phase1(phase2, checkpoint)
    phase2.train()

    output = phase2(collate_matcher_batch([_sample()]))[0].block_outputs[-1]
    loss = sum(
        getattr(output, name).sum()
        for name in (
            "A_logit", "B_logit", "O_logit",
            "A_prime_logit", "B_prime_logit", "O_prime_logit",
        )
    )
    loss.backward()

    assert output.A_logit.shape == (1, 1)
    assert phase2.fine_adapter[-1].weight.grad is not None
    assert not any(parameter.requires_grad for parameter in phase2.blocks[0].parameters())


def test_six_head_hungarian_can_exchange_same_identity_occurrences() -> None:
    base = _sample()
    sample = replace(
        base,
        pockets=(base.pockets[0], base.pockets[0]),
        occurrence_to_ligand=torch.zeros(2, dtype=torch.long),
        occurrence_candidate_id=torch.arange(2),
        occurrence_centroid_xyz=torch.zeros(2, 3),
        occurrence_coordinates=(torch.zeros(3, 3), torch.zeros(3, 3)),
        occurrence_present=(torch.ones(3, dtype=torch.bool),) * 2,
        A_target=torch.eye(2), B_target=torch.eye(2), O_target=torch.eye(2),
        A_prime_target=torch.eye(2, dtype=torch.bool),
        B_prime_target=torch.eye(2, dtype=torch.bool),
        O_prime_target=torch.eye(2, dtype=torch.bool),
    )
    swapped_logit = torch.tensor([[-8.0, 8.0], [8.0, -8.0]])
    head = MatcherHeadOutput(
        coarse_pair_repr=torch.empty(2, 2, 1),
        A_logit=swapped_logit,
        B_logit=swapped_logit,
        O_logit=swapped_logit,
        A_prime_logit=swapped_logit,
        B_prime_logit=swapped_logit,
        O_prime_logit=swapped_logit,
    )

    assert torch.equal(identity_hungarian_assignment(head, sample), torch.tensor([1, 0]))
