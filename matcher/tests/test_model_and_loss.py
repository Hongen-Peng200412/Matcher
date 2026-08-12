"""验证候选模型、完整身份匹配和三种监督门控的关键不变量。"""

from __future__ import annotations

import json
from typing import cast

import torch

from matcher.batching import collate_candidates
from matcher.candidate_data import CandidateDataset
from matcher.contracts import CandidateBatch, CandidateItem
from matcher.losses import LossWeights, compute_matcher_loss
from matcher.manifest import build_manifest
from matcher.model import CandidateEncoder, ClosedSetMatcher, MatcherModelConfig
from matcher.model.fusion import CandidateConditionedAggregator

from .factories import build_synthetic_roots


def _small_config(aggregation: str = "mean_max") -> MatcherModelConfig:
    """返回适合 CPU 契约测试的窄 RAUNet，空间层级与正式模型相同。"""

    return MatcherModelConfig(
        aggregation=aggregation,
        model_dim=16,
        attention_heads=4,
        dropout=0.0,
        voxel_feature_channels=8,
        raunet_planes=(4, 16, 16, 16, 16, 16, 8, 8, 8),
        recycle_iterations=1,
        gradient_checkpointing=False,
        bottleneck_attention_layers=1,
        graph_layers=1,
    )


def _batch(tmp_path, task_name: str) -> CandidateBatch:
    """从合成正式产物建立候选预训练或完整 Matcher 单样本批次。"""

    roots = build_synthetic_roots(tmp_path)
    manifest_path = build_manifest(roots, coverage_threshold=0.30, selection_seed=3407)
    dataset = CandidateDataset(
        roots=roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name=task_name,
        language_family="molformer" if task_name == "full_matcher" else None,
        pp_top_k=4,
        cache_bytes_per_worker=64 * 1024**2,
    )
    return collate_candidates([cast(CandidateItem, dataset[0])])


def test_candidate_encoder_shapes_and_pp_cross_attention_zero_initialization(tmp_path) -> None:
    """候选编码输出完整；强聚合 PP 分支初始化时对任意输入返回严格零增量。"""

    config = _small_config("candidate_cross_attention")
    model = CandidateEncoder(config).eval()
    batch = _batch(tmp_path, "candidate_pretrain")
    with torch.no_grad():
        output = model(batch)
    assert output["class_logits"].shape == (1, 6)
    assert output["candidate_embedding"].shape == (1, 16)
    assert output["ligand_area_logits"].shape == (1, 48, 48, 48)
    assert output["binding_area_logits"].shape == (1, 48, 48, 48)
    assert torch.isfinite(output["class_logits"]).all()

    aggregator = cast(CandidateConditionedAggregator, model.aggregator)
    query = torch.randn(2, 16)
    tokens = torch.randn(2, 3, 16)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    increment = aggregator.branches["pp"](query, tokens, mask)
    assert torch.count_nonzero(increment) == 0


def test_closed_set_matcher_forward_and_loss_are_finite(tmp_path) -> None:
    """图/语言/候选融合生成一个有效身份 logit，三种损失均可反向传播。"""

    batch = _batch(tmp_path, "full_matcher")
    model = ClosedSetMatcher(_small_config()).train()
    output = model(batch)
    assert output["identity_logits"].shape == (1, 1)
    assert output["identity_embeddings"].shape == (1, 1, 16)
    loss = compute_matcher_loss(
        output,
        batch,
        LossWeights(),
        identity_variant="oracle",
    )
    assert torch.isfinite(loss["total"])
    assert loss["identity_eligible_count"].item() == 1
    loss["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_identity_variants_gate_only_numerator_and_joint_reaches_class_probability() -> None:
    """三种身份损失都除以真实有效数，且只有 joint 把身份梯度传给 p_small。"""

    base_batch = {
        "class_target": torch.tensor([4, 4, 0]),
        "identity_target": torch.tensor([0, 1, 0]),
        "ligand_area_target": torch.zeros((3, 1, 1, 1), dtype=torch.bool),
        "binding_area_target": torch.zeros((3, 1, 1, 1), dtype=torch.bool),
    }
    weights = LossWeights(classification=0.0, ligand_area=0.0, binding_area=0.0, identity=1.0)
    identity_logits = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]],
        requires_grad=True,
    )

    class_gradient_norms: dict[str, float] = {}
    losses: dict[str, float] = {}
    for variant in ("oracle", "detached", "joint"):
        class_logits = torch.zeros((3, 6), requires_grad=True)
        outputs = {
            "class_logits": class_logits,
            "ligand_area_logits": torch.zeros((3, 1, 1, 1), requires_grad=True),
            "binding_area_logits": torch.zeros((3, 1, 1, 1), requires_grad=True),
            "identity_logits": identity_logits,
        }
        result = compute_matcher_loss(
            outputs,
            cast(CandidateBatch, base_batch),
            weights,
            identity_variant=variant,
        )
        gradient = torch.autograd.grad(result["total"], class_logits, retain_graph=True)[0]
        class_gradient_norms[variant] = float(gradient.abs().sum())
        losses[variant] = float(result["identity"].detach())
        assert result["identity_eligible_count"].item() == 2

    assert class_gradient_norms["oracle"] == 0.0
    assert class_gradient_norms["detached"] == 0.0
    assert class_gradient_norms["joint"] > 0.0
    # 初始 p_small=1/6，因此 detached 与 joint 数值相同，且都只是 oracle 分子的 1/6。
    assert abs(losses["detached"] - losses["joint"]) < 1e-7
    assert abs(losses["oracle"] / 6.0 - losses["joint"]) < 1e-7
