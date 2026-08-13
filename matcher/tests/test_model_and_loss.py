"""验证候选模型、完整身份匹配和三种监督门控的关键不变量。"""

from __future__ import annotations

from typing import cast

import pytest
import torch

from matcher.batching import collate_candidates
from matcher.candidate_data import CandidateDataset
from matcher.contracts import CandidateBatch, CandidateItem
from matcher.losses import IdentityLossVariant, LossWeights, MatcherLossBatch, compute_matcher_loss
from matcher.manifest import build_manifest
from matcher.model import CandidateEncoder, ClosedSetMatcher, MatcherModelConfig
from matcher.model.config import AggregationName
from matcher.model.fusion import (
    CandidateConditionedAggregator,
    gather_voxel_features,
    normalize_candidate_manual_features,
)
from matcher.model.ligand_matcher import (
    GraphMessageLayer,
    IdentityEncoder,
    IdentityRepresentations,
    IdentityScorer,
    SelfExcludingInventoryAttention,
)
from matcher.model.raunet import RaUnetBackbone

from .factories import build_synthetic_roots


def _small_config(aggregation: AggregationName = "mean_max") -> MatcherModelConfig:
    """返回适合 CPU 契约测试的窄 RAUNet，空间层级与正式模型相同。"""

    return MatcherModelConfig(
        aggregation=aggregation,
        use_manual_features=True,
        model_dim=16,
        attention_heads=4,
        dropout=0.0,
        voxel_feature_channels=8,
        raunet_planes=(4, 16, 16, 16, 16, 16, 8, 8, 4),
        recycle_iterations=1,
        gradient_checkpointing=False,
        bottleneck_attention_layers=1,
        graph_layers=1,
    )


def _batch(tmp_path, task_name: str) -> CandidateBatch:
    """从合成正式产物建立候选预训练或完整 Matcher 单样本批次。"""

    roots = build_synthetic_roots(tmp_path)
    manifest_path = build_manifest(
        roots,
        coverage_threshold=0.30,
        selection_seed=3407,
        split_names=("train", "validation"),
    )
    dataset = CandidateDataset(
        roots=roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name=task_name,
        language_family="molformer" if task_name == "full_matcher" else None,
        identity_scope="small_molecule" if task_name == "full_matcher" else None,
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
    increment.sum().backward()
    pp_output_gradient = aggregator.branches["pp"].output.weight.grad
    assert pp_output_gradient is not None
    assert torch.count_nonzero(pp_output_gradient) > 0


def test_closed_set_matcher_forward_and_loss_are_finite(tmp_path) -> None:
    """图/语言/候选融合生成一个有效身份 logit，三种损失均可反向传播。"""

    batch = _batch(tmp_path, "full_matcher")
    model = ClosedSetMatcher(_small_config()).train()
    output = model(batch)
    assert output["identity_logits"].shape == (1, 1)
    assert output["identity_native"].shape == (1, 1, 16)
    assert output["identity_graph"].shape == (1, 1, 16)
    assert output["identity_inventory"].shape == (1, 1, 16)
    loss = compute_matcher_loss(
        output,
        batch,
        LossWeights(classification=1.0, ligand_area=0.1, binding_area=0.1, identity=1.0),
        identity_variant="oracle",
        identity_scope="small_molecule",
        class_weights=None,
    )
    assert torch.isfinite(loss["total"])
    assert loss["identity_eligible_count"].item() == 1
    loss["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_graph_message_layer_accumulates_safely_under_bfloat16_autocast() -> None:
    """BF16 线性层输出必须先对齐 FP32 节点累加器，前向与反向均保持有限。"""

    layer = GraphMessageLayer(model_dim=16, edge_dim=10, dropout=0.0).train()
    nodes = torch.randn(4, 16, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_features = torch.randn(4, 10)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = layer(nodes, edge_index, edge_features)
        loss = output.square().mean()
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    loss.backward()
    assert nodes.grad is not None
    assert torch.isfinite(nodes.grad).all()


def test_closed_set_matcher_bfloat16_autocast_forward_and_backward(tmp_path) -> None:
    """整条完整 Matcher 在 BF16 自动混合精度下不得出现原地归约 dtype 冲突。"""

    batch = _batch(tmp_path, "full_matcher")
    model = ClosedSetMatcher(_small_config()).train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(batch)
        loss = output["class_logits"].float().square().mean()
        loss = loss + output["identity_logits"].float().square().mean()
        loss = loss + output["identity_graph"].float().square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_both_aggregators_accept_empty_modalities_without_nan(tmp_path) -> None:
    """真实数据常见 A/V 空集；弱聚合和强聚合都必须以零摘要稳定继续前向。"""

    original = _batch(tmp_path, "candidate_pretrain")
    aggregations = cast(tuple[AggregationName, ...], ("mean_max", "candidate_cross_attention"))
    for aggregation in aggregations:
        batch = cast(CandidateBatch, dict(original))
        batch["a_tokens"] = torch.empty((1, 0, 229), dtype=torch.float32)
        batch["a_mask"] = torch.empty((1, 0), dtype=torch.bool)
        batch["p_tokens"] = torch.empty((1, 0, 116), dtype=torch.float32)
        batch["p_mask"] = torch.empty((1, 0), dtype=torch.bool)
        batch["v_tokens"] = torch.empty((1, 0, 52), dtype=torch.float32)
        batch["v_mask"] = torch.empty((1, 0), dtype=torch.bool)
        batch["pp_tokens"] = torch.empty((1, 0, 4), dtype=torch.float32)
        batch["pp_mask"] = torch.empty((1, 0), dtype=torch.bool)
        batch["v_index_zyx"] = torch.empty((1, 0, 3), dtype=torch.long)
        batch["pp_index_zyx"] = torch.empty((1, 0, 3), dtype=torch.long)
        model = CandidateEncoder(_small_config(aggregation)).eval()
        with torch.no_grad():
            output = model(batch)
        assert torch.isfinite(output["candidate_embedding"]).all()
        assert torch.isfinite(output["class_logits"]).all()


def test_manual_feature_normalization_changes_only_token_counts() -> None:
    """四组三元组只对计数执行 log1p，固化概率与空模态零值逐项保持。"""

    raw = torch.tensor(
        [[0.0, 0.0, 0.0, 3.0, 0.25, 0.75, 1.0, 0.4, 0.9, 7.0, 0.1, 1.0]]
    )
    normalized = normalize_candidate_manual_features(raw)
    expected = raw.clone()
    expected[:, (0, 3, 6, 9)] = torch.log1p(expected[:, (0, 3, 6, 9)])
    assert torch.equal(normalized, expected)


def test_identity_variants_gate_only_numerator_and_joint_reaches_class_probability() -> None:
    """三种身份损失都除以真实有效数，且只有 joint 把身份梯度传给 p_small。"""

    base_batch = {
        "class_target": torch.tensor([4, 4, 0]),
        "identity_target": torch.tensor([0, 1, 0]),
        "identity_mask": torch.ones((3, 2), dtype=torch.bool),
        "identity_class_index": torch.full((3, 2), 4, dtype=torch.long),
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
    for variant in cast(tuple[IdentityLossVariant, ...], ("oracle", "detached", "joint")):
        class_logits = torch.zeros((3, 6), requires_grad=True)
        outputs = {
            "class_logits": class_logits,
            "ligand_area_logits": torch.zeros((3, 1, 1, 1), requires_grad=True),
            "binding_area_logits": torch.zeros((3, 1, 1, 1), requires_grad=True),
            "identity_logits": identity_logits,
        }
        result = compute_matcher_loss(
            outputs,
            cast(MatcherLossBatch, base_batch),
            weights,
            identity_variant=variant,
            identity_scope="small_molecule",
            class_weights=None,
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


def test_five_class_identity_loss_uses_each_target_class_probability() -> None:
    """五类身份门按真实身份类别取概率，背景与缺失目标均不进入身份分母。"""

    class_logits = torch.tensor(
        [
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    identity_logits = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]],
        requires_grad=True,
    )
    outputs = {
        "class_logits": class_logits,
        "identity_logits": identity_logits,
        "ligand_area_logits": torch.zeros((3, 1, 1, 1), requires_grad=True),
        "binding_area_logits": torch.zeros((3, 1, 1, 1), requires_grad=True),
    }
    batch = cast(
        MatcherLossBatch,
        {
            "class_target": torch.tensor([1, 5, 0]),
            "identity_target": torch.tensor([0, 1, -1]),
            "identity_mask": torch.ones((3, 2), dtype=torch.bool),
            "identity_class_index": torch.tensor([[1, 5], [1, 5], [1, 5]]),
            "ligand_area_target": torch.zeros((3, 1, 1, 1), dtype=torch.bool),
            "binding_area_target": torch.zeros((3, 1, 1, 1), dtype=torch.bool),
        },
    )
    result = compute_matcher_loss(
        outputs,
        batch,
        LossWeights(classification=0.0, ligand_area=0.0, binding_area=0.0, identity=1.0),
        identity_variant="detached",
        identity_scope="all_non_background",
        class_weights=None,
    )
    per_candidate = torch.nn.functional.cross_entropy(
        identity_logits[:2],
        torch.tensor([0, 1]),
        reduction="none",
    )
    probability = torch.softmax(class_logits, dim=-1)
    expected = (
        probability[0, 1].detach() * per_candidate[0]
        + probability[1, 5].detach() * per_candidate[1]
    ) / 2.0
    assert result["identity_eligible_count"].item() == 2
    assert torch.allclose(result["identity"], expected)
    class_gradient = torch.autograd.grad(result["total"], class_logits, allow_unused=True)[0]
    assert class_gradient is None or torch.count_nonzero(class_gradient) == 0


def test_identity_manual_features_are_learnable_but_class_probability_is_detached() -> None:
    """身份 MLP 读取手动统计与目标类概率，但身份路径不能反向更新六分类 logits。"""

    torch.manual_seed(3407)
    scorer = IdentityScorer(model_dim=16, dropout=0.0, use_manual_features=True).train()
    candidate = torch.randn(2, 16, requires_grad=True)
    representations = cast(
        IdentityRepresentations,
        {
            "native": torch.randn(2, 3, 16, requires_grad=True),
            "graph": torch.randn(2, 3, 16, requires_grad=True),
            "inventory": torch.randn(2, 3, 16, requires_grad=True),
        },
    )
    identity_mask = torch.ones((2, 3), dtype=torch.bool)
    identity_class_index = torch.tensor([[1, 4, 5], [1, 4, 5]])
    class_logits = torch.randn(2, 6, requires_grad=True)
    manual_features = torch.rand(2, 12, requires_grad=True)
    logits = scorer(
        candidate,
        representations,
        identity_mask,
        class_logits,
        identity_class_index,
        manual_features,
    )
    logits.sum().backward()
    assert class_logits.grad is None or torch.count_nonzero(class_logits.grad) == 0
    assert manual_features.grad is not None
    assert torch.count_nonzero(manual_features.grad) > 0
    assert candidate.grad is not None
    assert torch.count_nonzero(candidate.grad) > 0


def test_identity_loss_without_eligible_candidate_stays_finite_with_padded_logits() -> None:
    """没有身份监督时，补零身份的有限极小分数不得通过零乘法产生 NaN。"""

    identity_logits = torch.full((2, 3), torch.finfo(torch.float32).min, requires_grad=True)
    outputs = {
        "class_logits": torch.zeros((2, 6), requires_grad=True),
        "ligand_area_logits": torch.zeros((2, 1, 1, 1), requires_grad=True),
        "binding_area_logits": torch.zeros((2, 1, 1, 1), requires_grad=True),
        "identity_logits": identity_logits,
    }
    batch = cast(
        MatcherLossBatch,
        {
            "class_target": torch.tensor([0, 4]),
            "identity_target": torch.tensor([-1, -1]),
            "identity_mask": torch.ones((2, 3), dtype=torch.bool),
            "identity_class_index": torch.full((2, 3), 4, dtype=torch.long),
            "ligand_area_target": torch.zeros((2, 1, 1, 1), dtype=torch.bool),
            "binding_area_target": torch.zeros((2, 1, 1, 1), dtype=torch.bool),
        },
    )
    result = compute_matcher_loss(
        outputs,
        batch,
        LossWeights(classification=1.0, ligand_area=0.1, binding_area=0.1, identity=1.0),
        identity_variant="joint",
        identity_scope="small_molecule",
        class_weights=None,
    )
    assert torch.isfinite(result["identity"])
    assert result["identity"].item() == 0.0
    result["total"].backward()
    assert identity_logits.grad is not None
    assert torch.count_nonzero(identity_logits.grad) == 0


def test_raunet_three_recycles_have_no_dead_trainable_parameters() -> None:
    """三轮 recycle 直接回收 decoded0，最后一轮反向应覆盖每个可训练参数。"""

    backbone = RaUnetBackbone(
        input_channels=3,
        feature_channels=4,
        planes=(4, 16, 16, 16, 16, 16, 8, 8, 4),
        recycle_iterations=3,
        gradient_checkpointing=False,
        bottleneck_attention_layers=1,
    ).train()
    output = backbone(torch.randn(1, 3, 48, 48, 48, requires_grad=True))
    output.square().mean().backward()
    missing = [name for name, parameter in backbone.named_parameters() if parameter.grad is None]
    assert missing == []


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="需要支持 BF16 的 CUDA 设备。",
)
def test_checkpointed_raunet_recycles_preserve_gradients_under_cuda_bfloat16() -> None:
    """三轮 recycle 中每个无梯度前置轮清除缓存后，CUDA BF16 非重入激活重计算须为全部参数产生有限梯度。"""

    backbone = RaUnetBackbone(
        input_channels=3,
        feature_channels=4,
        planes=(4, 16, 16, 16, 16, 16, 8, 8, 4),
        recycle_iterations=3,
        gradient_checkpointing=True,
        bottleneck_attention_layers=1,
    ).cuda().train()
    voxel_input = torch.randn(1, 3, 48, 48, 48, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = backbone(voxel_input)
        loss = output.float().square().mean()
    loss.backward()
    missing = [name for name, parameter in backbone.named_parameters() if parameter.grad is None]
    assert missing == []
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in backbone.parameters()
        if parameter.grad is not None
    )


def test_single_identity_inventory_is_an_exact_zero_increment() -> None:
    """单身份清单没有其他身份可读，上下文分支必须返回严格零。"""

    inventory = SelfExcludingInventoryAttention(16, 4, 0.0).eval()
    identities = torch.randn(2, 1, 16)
    mask = torch.ones((2, 1), dtype=torch.bool)
    with torch.no_grad():
        enriched = inventory(identities, mask)
    assert torch.count_nonzero(enriched) == 0


def test_identity_encoder_rejects_duplicate_graph_for_one_identity(tmp_path) -> None:
    """同一样本的同一身份若映射两张图，必须拒绝而不是静默覆盖。"""

    original = _batch(tmp_path, "full_matcher")
    batch = cast(CandidateBatch, dict(original))
    node_count = int(original["graph_element"].shape[0])
    batch["graph_to_sample"] = original["graph_to_sample"].repeat(2)
    batch["graph_to_identity"] = original["graph_to_identity"].repeat(2)
    batch["graph_element"] = original["graph_element"].repeat(2)
    batch["graph_charge"] = original["graph_charge"].repeat(2)
    batch["graph_residue_id"] = original["graph_residue_id"].repeat(2)
    batch["graph_node_continuous"] = original["graph_node_continuous"].repeat(2, 1)
    batch["graph_node_graph_index"] = torch.cat(
        (original["graph_node_graph_index"], original["graph_node_graph_index"] + 1)
    )
    batch["graph_edge_index"] = torch.cat(
        (original["graph_edge_index"], original["graph_edge_index"] + node_count),
        dim=1,
    )
    batch["graph_edge_features"] = original["graph_edge_features"].repeat(2, 1)
    encoder = IdentityEncoder(_small_config()).eval()
    with pytest.raises(ValueError, match="恰好对应一张"):
        encoder(batch)


def test_identity_graph_and_inventory_scores_start_as_exact_zero_increments() -> None:
    """配体图和清单分支本身保持可学，但其打分出口门在初始化时严格为零。"""

    scorer = IdentityScorer(model_dim=16, dropout=0.0, use_manual_features=False).train()
    candidate = torch.randn(2, 16)
    representations = cast(
        IdentityRepresentations,
        {
            "native": torch.randn(2, 3, 16),
            "graph": torch.randn(2, 3, 16),
            "inventory": torch.randn(2, 3, 16),
        },
    )
    mask = torch.ones((2, 3), dtype=torch.bool)
    native_only = scorer.native(candidate, representations["native"], None, None)
    logits = scorer(
        candidate,
        representations,
        mask,
        torch.zeros((2, 6)),
        torch.full((2, 3), 4, dtype=torch.long),
        torch.zeros((2, 12)),
    )
    assert torch.equal(logits, native_only)
    logits.sum().backward()
    assert scorer.graph_gate.grad is not None
    assert scorer.inventory_gate.grad is not None
    assert torch.isfinite(scorer.graph_gate.grad)
    assert torch.isfinite(scorer.inventory_gate.grad)


def test_single_identity_inventory_score_stays_zero_after_gate_opens() -> None:
    """即使清单门已训练为非零，单身份样本仍没有可贡献的其他身份上下文。"""

    scorer = IdentityScorer(model_dim=16, dropout=0.0, use_manual_features=False).eval()
    scorer.inventory_gate.data.fill_(1.0)
    scorer.graph_gate.data.zero_()
    candidate = torch.randn(2, 16)
    native = torch.randn(2, 1, 16)
    representations = cast(
        IdentityRepresentations,
        {
            "native": native,
            "graph": torch.randn(2, 1, 16),
            "inventory": torch.zeros(2, 1, 16),
        },
    )
    mask = torch.ones((2, 1), dtype=torch.bool)
    with torch.no_grad():
        logits = scorer(
            candidate,
            representations,
            mask,
            torch.zeros((2, 6)),
            torch.full((2, 1), 4, dtype=torch.long),
            torch.zeros((2, 12)),
        )
        native_only = scorer.native(candidate, native, None, None)
    assert torch.equal(logits, native_only)


def test_model_config_and_voxel_gather_reject_invalid_contracts() -> None:
    """非正注意力头、非正 RAUNet 宽度和有效 token 越界索引都应尽早拒绝。"""

    values = _small_config().to_dict()
    values["attention_heads"] = 0
    with pytest.raises(ValueError, match="attention_heads"):
        MatcherModelConfig.from_dict(values)
    values = _small_config().to_dict()
    values["raunet_planes"] = [4, 16, 16, 16, 16, 16, 8, 8, 0]
    with pytest.raises(ValueError, match="raunet_planes"):
        MatcherModelConfig.from_dict(values)

    volume = torch.zeros((1, 2, 4, 4, 4))
    indices = torch.tensor([[[9, 0, 0]]])
    with pytest.raises(ValueError, match="超出"):
        gather_voxel_features(volume, indices, torch.tensor([[True]]))
    padded = gather_voxel_features(volume, indices, torch.tensor([[False]]))
    assert torch.count_nonzero(padded) == 0

    legacy_values = _small_config().to_dict()
    del legacy_values["use_manual_features"]
    assert MatcherModelConfig.from_dict(legacy_values).use_manual_features is False
