"""共享双模式 Matcher：复用稳定算子，新增六头监督与严格零条件增量。"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

from matcher.model import (
    CoarsePredictionHead,
    CoarseState,
    EntityAuxiliaryOutput,
    Matcher as _AnchorMatcher,
    PDBMatcherOutput,
    _masked_sigmoid_focal_mean,
    _pad_packed_values,
    build_contact_marginals,
)

from .contracts import MatcherSample
from .context import DensityPointBuilder


@dataclass(frozen=True)
class MatcherHeadOutput:
    """一个 Block 的六项候选—slot 输出，所有矩阵均为 `[C,S]`。"""

    coarse_pair_repr: Tensor
    A_logit: Tensor
    B_logit: Tensor
    O_logit: Tensor
    A_prime_logit: Tensor
    B_prime_logit: Tensor
    O_prime_logit: Tensor

    @property
    def A(self) -> Tensor:
        return torch.sigmoid(self.A_logit)

    @property
    def B(self) -> Tensor:
        return torch.sigmoid(self.B_logit)

    @property
    def O(self) -> Tensor:
        return torch.sigmoid(self.O_logit)

    @property
    def O_prime(self) -> Tensor:
        return torch.sigmoid(self.O_prime_logit)


class MatcherPredictionHead(CoarsePredictionHead):
    """沿用已验证的 pair 表示，只把出口扩展为六个独立小头。"""

    def __init__(self, *args, initial_prior: float = 0.05, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        width = self.OHead.in_features
        self.continuous_heads = nn.ModuleDict(
            {name: nn.Linear(width, 1) for name in ("A", "B", "O")}
        )
        self.hard_heads = nn.ModuleDict(
            {name: nn.Linear(width, 1) for name in ("A_prime", "B_prime", "O_prime")}
        )
        del self.OHead
        del self.OPrimeHead
        bias = math.log(initial_prior / (1.0 - initial_prior))
        for head in (*self.continuous_heads.values(), *self.hard_heads.values()):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.constant_(head.bias, bias)

    def predict(self, pair_repr: Tensor) -> MatcherHeadOutput:
        values = {
            name: head(pair_repr).squeeze(-1)
            for name, head in (*self.continuous_heads.items(), *self.hard_heads.items())
        }
        return MatcherHeadOutput(pair_repr, **{f"{name}_logit": value for name, value in values.items()})


class Stage1ContextDelta(nn.Module):
    """把 V/P/PP 压成 BOX 候选增量；条件缺席时显式返回逐元素零。"""

    def __init__(
        self, *, V_feature_dim: int, P_feature_dim: int, repr_dim: int
    ) -> None:
        super().__init__()
        input_dim = 4 + V_feature_dim + 4 + P_feature_dim + 4 + repr_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, repr_dim), nn.SiLU(),
            nn.Linear(repr_dim, repr_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    @staticmethod
    def _mean(coordinate: Tensor, probability: Tensor, feature: Tensor | None = None) -> Tensor:
        width = 4 + (0 if feature is None else feature.shape[-1])
        if coordinate.shape[0] == 0:
            return coordinate.new_zeros(width)
        fields = (coordinate, probability[:, None]) if feature is None else (coordinate, probability[:, None], feature)
        return torch.cat(fields, dim=-1).mean(dim=0)

    def forward(self, sample: MatcherSample, reference: Tensor, PP_map_summary: Tensor) -> Tensor:
        rows = []
        for pocket in sample.pockets:
            context = pocket.stage1
            if context is None:
                rows.append(reference.new_zeros(reference.shape[-1]))
                continue
            value = torch.cat(
                (
                    self._mean(context.V_coord_local_xyz.to(reference), context.V_probability.to(reference), context.V_feature.to(reference)),
                    self._mean(context.P_coord_local_xyz.to(reference), context.P_probability.to(reference), context.P_feature.to(reference)),
                    self._mean(context.PP_coord_local_xyz.to(reference), context.PP_probability.to(reference)),
                    PP_map_summary[len(rows)],
                )
            )
            rows.append(self.network(value))
        return torch.stack(rows)


class Stage1ANodeDelta(nn.Module):
    """以 A 概率和 L1–L3 特征调制基础 49 维 A 输入。"""

    def __init__(self, A_feature_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(1 + A_feature_dim), nn.Linear(1 + A_feature_dim, output_dim),
            nn.SiLU(), nn.Linear(output_dim, output_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, probability: Tensor, feature: Tensor) -> Tensor:
        return self.network(torch.cat((probability[:, None], feature), dim=-1))


class Matcher(_AnchorMatcher):
    """两个显式 Dataset 共用的 Matcher；旧 Anchor 类型不会进入公开接口。"""

    def __init__(
        self, *args, V_feature_dim: int = 0, P_feature_dim: int = 0,
        A_feature_dim: int = 0, initial_prior: float = 0.05, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.prediction_heads = nn.ModuleList(
            MatcherPredictionHead(
                node_dim=kwargs.get("node_dim", 128),
                repr_dim=kwargs.get("repr_dim", 256),
                num_heads=kwargs.get("num_heads", 8),
                dropout=kwargs.get("dropout", 0.1),
                mode=kwargs.get("mode", "coupled_plus_modal"),
                initial_prior=initial_prior,
            )
            for _ in range(self.num_blocks)
        )
        repr_dim = kwargs.get("repr_dim", 256)
        self.density_point_builder = DensityPointBuilder(
            tuple(kwargs.get("map_channels", (16, 32, 48, 64))), repr_dim
        )
        self.stage1_delta = Stage1ContextDelta(
            V_feature_dim=V_feature_dim,
            P_feature_dim=P_feature_dim,
            repr_dim=repr_dim,
        )
        self.stage1_A_node_delta = Stage1ANodeDelta(A_feature_dim, kwargs.get("A_node_input_dim", 49))
        self._PP_map_summary: dict[int, Tensor] = {}

    def _Map_repr(self, batch, device: torch.device) -> list[Tensor]:
        """同一次 U-Net 前向同时产生 Map 摘要和模式二 PP 采样特征。"""

        counts = [sample.num_candidates for sample in batch.samples]
        pockets = [pocket for sample in batch.samples for pocket in sample.pockets]
        density = batch.density.to(device)
        chunk_size = self.map_candidate_chunk_size or len(density)
        summaries = []
        point_summaries = []
        for start in range(0, len(density), chunk_size):
            end = min(start + chunk_size, len(density))
            features = self.MapBackbone(density[start:end])
            summaries.append(self.MapSummaryHead(features))
            current = pockets[start:end]
            points = self.density_point_builder(
                features,
                tuple(
                    pocket.stage1.PP_coord_local_xyz.to(device)
                    if pocket.stage1 is not None else density.new_empty((0, 3))
                    for pocket in current
                ),
                tuple(
                    pocket.stage1.PP_probability.to(device)
                    if pocket.stage1 is not None else density.new_empty(0)
                    for pocket in current
                ),
                tuple(pocket.voxel_size_xyz.to(device) * density.shape[-1] for pocket in current),
            )
            point_summaries.extend(
                point.feature.mean(dim=0)
                if len(point.feature) else density.new_zeros(self.stage1_delta.network[-1].out_features)
                for point in points
            )
        Map_by_pdb = list(torch.cat(summaries).split(counts))
        PP_by_pdb = list(torch.stack(point_summaries).split(counts))
        self._PP_map_summary = {
            id(sample): value for sample, value in zip(batch.samples, PP_by_pdb, strict=True)
        }
        return Map_by_pdb

    def _initial_coarse(self, sample, *args, **kwargs) -> CoarseState:
        coarse = super()._initial_coarse(sample, *args, **kwargs)
        delta = self.stage1_delta(sample, coarse.BOX_repr, self._PP_map_summary[id(sample)])
        return CoarseState(
            coarse.CCD_repr,
            coarse.BOX_repr + delta,
            coarse.A_repr,
            coarse.Map_repr,
        )

    def _run_fine_pairs(self, sample, coarse_pair_repr, prediction_head, *args, **kwargs):
        ligand_state, A_state, ligand, A, device = args
        if self.fine_pair_branch is None or self.fine_adapter is None:
            raise RuntimeError("FinePair 仅能在 Phase2 中运行。")
        candidate_count, occurrence_count = coarse_pair_repr.shape[:2]
        candidate_index = torch.arange(candidate_count, device=device).repeat_interleave(occurrence_count)
        occurrence_index = torch.arange(occurrence_count, device=device).repeat(candidate_count)
        occurrence_to_ligand = sample.occurrence_to_ligand.to(device)
        slot_index, _ = self._slot_indices(occurrence_to_ligand, len(ligand.node_ptr) - 1)
        ligand_atoms, ligand_mask = _pad_packed_values(ligand_state.node, ligand.node_ptr)
        A_atoms, A_mask = _pad_packed_values(A_state.node, A.node_ptr)
        A_coordinates, _ = _pad_packed_values(A.coordinates, A.node_ptr)
        pair_chunk_size = self.fine_pair_chunk_size or len(candidate_index)
        auxiliary_enabled = self.fine_pair_branch.recall_atom_head is not None
        gt_coordinates = (
            pad_sequence(
                tuple(value.to(device) for value in sample.ligand_gt_coordinates),
                batch_first=True,
            )
            if auxiliary_enabled else None
        )
        gt_present = (
            pad_sequence(
                tuple(value.to(device) for value in sample.ligand_gt_present),
                batch_first=True,
            )
            if auxiliary_enabled else None
        )
        contact_recall = None
        contact_precision = None
        if gt_coordinates is not None and gt_present is not None:
            contact_recall, contact_precision = build_contact_marginals(
                A_coordinates, A_mask, gt_coordinates, gt_present, pair_chunk_size
            )

        def compute_chunk(coarse_chunk, candidate_chunk, occurrence_chunk):
            ligand_index = occurrence_to_ligand[occurrence_chunk]
            chunk_ligand_mask = ligand_mask[ligand_index]
            chunk_A_mask = A_mask[candidate_chunk]
            ligand_width = int(chunk_ligand_mask.sum(dim=1).max().item())
            A_width = max(int(chunk_A_mask.sum(dim=1).max().item()), 1)
            chunk_ligand_mask = chunk_ligand_mask[:, :ligand_width]
            chunk_A_mask = chunk_A_mask[:, :A_width]
            fine_repr, recall_logit, precision_logit = self.fine_pair_branch(
                ligand_atoms[ligand_index, :ligand_width],
                A_atoms[candidate_chunk, :A_width],
                chunk_ligand_mask, chunk_A_mask,
                self.slot_embedding(slot_index[occurrence_chunk]),
            )
            recall_by_gt = coarse_chunk.new_zeros((len(coarse_chunk), occurrence_count), dtype=torch.float32)
            precision_by_gt = torch.zeros_like(recall_by_gt)
            if contact_recall is not None and contact_precision is not None:
                same_identity = (
                    occurrence_to_ligand[occurrence_chunk, None]
                    == occurrence_to_ligand[None, :]
                )
                pair_row, gt_index = torch.nonzero(same_identity, as_tuple=True)
                target_candidate = candidate_chunk[pair_row]
                recall_by_gt[pair_row, gt_index] = _masked_sigmoid_focal_mean(
                    recall_logit[pair_row],
                    contact_recall[target_candidate, gt_index, :ligand_width],
                    gt_present[gt_index, :ligand_width],
                )
                precision_by_gt[pair_row, gt_index] = _masked_sigmoid_focal_mean(
                    precision_logit[pair_row],
                    contact_precision[target_candidate, gt_index, :A_width],
                    A_mask[target_candidate, :A_width],
                )
            prediction = prediction_head.predict(coarse_chunk + self.fine_adapter(fine_repr))
            return (*(
                prediction.A_logit, prediction.B_logit, prediction.O_logit,
                prediction.A_prime_logit, prediction.B_prime_logit, prediction.O_prime_logit,
            ), recall_by_gt, precision_by_gt)

        chunks = []
        flat_coarse = coarse_pair_repr.flatten(0, 1)
        for start in range(0, len(candidate_index), pair_chunk_size):
            arguments = (
                flat_coarse[start : start + pair_chunk_size],
                candidate_index[start : start + pair_chunk_size],
                occurrence_index[start : start + pair_chunk_size],
            )
            chunks.append(
                checkpoint(compute_chunk, *arguments, use_reentrant=False, preserve_rng_state=True)
                if self.training and self.fine_pair_activation_checkpoint
                else compute_chunk(*arguments)
            )
        merged = [torch.cat([chunk[index] for chunk in chunks]) for index in range(8)]
        shape = (candidate_count, occurrence_count)
        head = MatcherHeadOutput(coarse_pair_repr, *(value.reshape(shape) for value in merged[:6]))
        pair_loss_shape = (candidate_count, occurrence_count, occurrence_count)
        return head, merged[6].reshape(pair_loss_shape), merged[7].reshape(pair_loss_shape)

    def forward(self, batch):
        """在旧稳定主干前只装配 Stage1 A 节点增量；模式一不经过该模块。"""

        device = next(self.parameters()).device
        A_graphs = []
        for sample in batch.samples:
            for pocket in sample.pockets:
                graph = pocket.A_graph
                node_input = graph.node_input.to(device)
                if pocket.stage1 is not None:
                    node_input = node_input + self.stage1_A_node_delta(
                        pocket.stage1.A_probability.to(device), pocket.stage1.A_feature.to(device)
                    )
                A_graphs.append(replace(graph, node_input=node_input))
        return super().forward(replace(batch, A_graphs=tuple(A_graphs)))


__all__ = [
    "EntityAuxiliaryOutput", "Matcher", "MatcherHeadOutput", "PDBMatcherOutput",
    "Stage1ContextDelta",
]
