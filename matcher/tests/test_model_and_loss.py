"""验证候选模型、完整身份匹配和三种监督门控的关键不变量。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from matcher.data.manifest import build_manifest
from matcher.datasets.batching import collate_candidates
from matcher.datasets.candidate_data import CandidateDataset
from matcher.datasets.contracts import CandidateBatch, CandidateItem
from matcher.model import CandidateEncoder, ClosedSetMatcher, MatcherModelConfig
from matcher.model.config import AggregationName
from matcher.model.fusion import (
    CandidateConditionedAggregator,
    CandidateTokenAttentionBias,
    MeanMaxAggregator,
    MeanMaxCandidateConditionedResidualAggregator,
    candidate_token_attention_bias_features,
    gather_voxel_features,
    normalize_candidate_manual_features,
    normalize_candidate_sampling_features,
)
from matcher.model.ligand_matcher import (
    GraphMessageLayer,
    IdentityEncoder,
    IdentityRepresentations,
    IdentityScorer,
    SelfExcludingInventoryAttention,
)
from matcher.model.raunet import RaUnetBackbone
from matcher.training.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    load_candidate_encoder_checkpoint,
    save_checkpoint,
)
from matcher.training.losses import IdentityLossVariant, LossWeights, MatcherLossBatch, compute_matcher_loss

from .factories import build_synthetic_roots


def _small_config(
    aggregation: AggregationName = "mean_max",
    *,
    use_candidate_token_attention_bias: bool = False,
    initial_graph_gate: float = 0.0,
    initial_inventory_gate: float = 0.0,
    dropout: float = 0.0,
) -> MatcherModelConfig:
    """返回适合 CPU 契约测试的窄 RAUNet，空间层级与正式模型相同。"""

    return MatcherModelConfig(
        aggregation=aggregation,
        use_manual_features=True,
        use_sampling_features=True,
        use_candidate_token_attention_bias=use_candidate_token_attention_bias,
        initial_graph_gate=initial_graph_gate,
        initial_inventory_gate=initial_inventory_gate,
        model_dim=16,
        attention_heads=4,
        dropout=dropout,
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
        local_crop_strategy="fixed_center",
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
    increment = aggregator.branches["pp"](query, tokens, mask, None)
    assert torch.count_nonzero(increment) == 0
    increment.sum().backward()
    pp_output_gradient = aggregator.branches["pp"].output.weight.grad
    assert pp_output_gradient is not None
    assert torch.count_nonzero(pp_output_gradient) > 0


def test_enhanced_candidate_encoder_is_finite_under_bfloat16_autocast(tmp_path) -> None:
    """实体偏置增强的候选编码器应在 BF16 下完成整条前向与反向传播。"""

    model = CandidateEncoder(
        _small_config(
            "candidate_cross_attention",
            use_candidate_token_attention_bias=True,
        )
    ).train()
    batch = _batch(tmp_path, "candidate_pretrain")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(batch)
        loss = output["class_logits"].float().square().mean()
        loss = loss + output["candidate_embedding"].float().square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    aggregator = cast(CandidateConditionedAggregator, model.aggregator)
    assert aggregator.attention_biases is not None
    gradients = [
        parameter.grad
        for module in aggregator.attention_biases.values()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_closed_set_matcher_forward_and_loss_are_finite(tmp_path) -> None:
    """图/语言/候选融合生成一个有效身份 logit，三种损失均可反向传播。"""

    batch = _batch(tmp_path, "full_matcher")
    model = ClosedSetMatcher(_small_config()).train()
    output = model(batch)
    assert output["identity_logits"].shape == (1, 1)
    assert output["identity_score_native"].shape == (1, 1)
    assert output["identity_score_graph_increment"].shape == (1, 1)
    assert output["identity_score_inventory_increment"].shape == (1, 1)
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


def test_joint_identity_strings_do_not_change_model_forward(tmp_path) -> None:
    """联合身份字符串只解释身份轴；同一数值批次的正式前向必须逐张量相等。"""

    batch = _batch(tmp_path, "full_matcher")
    altered = cast(CandidateBatch, dict(batch))
    altered["identity_prepared_smiles"] = [["different-public-input"]]
    altered_metadata = [dict(item) for item in batch["metadata"]]
    altered_metadata[0]["matched_prepared_smiles"] = "different-public-input"
    altered["metadata"] = cast(Any, altered_metadata)
    model = ClosedSetMatcher(_small_config()).eval()
    with torch.inference_mode():
        expected = model(batch)
        actual = model(altered)
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name


def test_validation_identity_score_components_reuse_forward_without_syncing_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """评估诊断直接读取正式前向组件，不得重跑打分校验或改变 Torch 随机状态。"""

    batch = _batch(tmp_path, "full_matcher")
    model = ClosedSetMatcher(_small_config()).eval()
    with torch.inference_mode():
        output = model(batch)
        rng_before = torch.random.get_rng_state().clone()
        monkeypatch.setattr(
            type(model.identity_scorer),
            "score_components",
            lambda *_args, **_kwargs: pytest.fail("诊断不得重跑 score_components"),
        )
        components = model.identity_score_components(batch, output)
        rng_after = torch.random.get_rng_state().clone()
    assert components["native"].data_ptr() == output["identity_score_native"].data_ptr()
    assert components["graph_increment"].data_ptr() == output[
        "identity_score_graph_increment"
    ].data_ptr()
    assert components["inventory_increment"].data_ptr() == output[
        "identity_score_inventory_increment"
    ].data_ptr()
    expected = (
        components["native"]
        + components["graph_increment"]
        + components["inventory_increment"]
    ).masked_fill(
        ~batch["identity_mask"],
        torch.finfo(output["identity_logits"].dtype).min,
    )
    assert torch.equal(output["identity_logits"], expected)
    assert torch.equal(rng_before, rng_after)

    model.train()
    with pytest.raises(RuntimeError, match="只允许在 eval 模式"):
        model.identity_score_components(batch, output)


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
    model = ClosedSetMatcher(_small_config(initial_graph_gate=0.1)).train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(batch)
        loss = output["class_logits"].float().square().mean()
        loss = loss + output["identity_logits"].float().square().mean()
        loss = loss + output["identity_score_graph_increment"].float().square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_all_aggregators_accept_empty_modalities_without_nan(tmp_path) -> None:
    """真实数据常见 A/V 空集；三种聚合路径都必须以零摘要稳定继续前向。"""

    original = _batch(tmp_path, "candidate_pretrain")
    aggregations = cast(
        tuple[AggregationName, ...],
        (
            "mean_max",
            "candidate_cross_attention",
            "mean_max_candidate_cross_attention_concat",
        ),
    )
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


def test_sampling_features_use_fixed_geometric_normalization() -> None:
    """边界标志与保留率不变；L2/L∞ 分别按 24√3 和 24 归一化。"""

    raw = torch.tensor([[1.0, 24.0 * 3.0**0.5, 24.0, 0.75]])
    normalized = normalize_candidate_sampling_features(raw)
    assert torch.allclose(normalized, torch.tensor([[1.0, 1.0, 1.0, 0.75]]))


@pytest.mark.parametrize("feature_width", (229, 116, 52, 4))
def test_candidate_token_attention_bias_features_keep_xyz_probability_order(
    feature_width: int,
) -> None:
    """A/P/V/PP 四种正式宽度都把末尾概率、XYZ 重排为 XYZ、概率。"""

    tokens = torch.zeros((1, 2, feature_width))
    tokens[0, 0, -4:] = torch.tensor([0.75, -0.25, 0.50, 1.00])
    tokens[0, 1, -4:] = torch.tensor([0.20, 0.10, -0.30, 0.40])
    features = candidate_token_attention_bias_features(tokens)
    expected = torch.tensor(
        [[[-0.25, 0.50, 1.00, 0.75], [0.10, -0.30, 0.40, 0.20]]]
    )
    assert torch.equal(features, expected)


def _attention_aggregator_inputs() -> tuple[
    torch.Tensor,
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    """构造同时覆盖非空、补零和空模态的增强注意力窄测试输入。

    返回第 1 项是浮点 ``(2,16)`` 候选摘要。
    返回第 2 项是 A/P/V/PP 投影 token 与 bool 掩码字典，各模态 token 数独立。
    返回第 3 项是浮点 ``(2,12)`` 模态手动统计。
    返回第 4 项是浮点 ``(2,4)`` 候选采样统计。
    返回第 5 项是与每种模态 token 逐项对齐的浮点 ``(2,N_m,4)`` 注意力偏置输入字典。
    """

    candidate = torch.randn(2, 16)
    modalities = {
        "a": (torch.randn(2, 3, 16), torch.tensor([[True, True, False], [True, False, False]])),
        "p": (torch.randn(2, 2, 16), torch.tensor([[True, True], [False, False]])),
        "v": (torch.randn(2, 1, 16), torch.ones((2, 1), dtype=torch.bool)),
        "pp": (torch.empty((2, 0, 16)), torch.empty((2, 0), dtype=torch.bool)),
    }
    attention_features = {
        name: torch.randn(tokens.shape[0], tokens.shape[1], 4)
        for name, (tokens, _) in modalities.items()
    }
    return candidate, modalities, torch.randn(2, 12), torch.randn(2, 4), attention_features


def _concatenated_aggregator_inputs() -> tuple[
    torch.Tensor,
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    torch.Tensor,
    torch.Tensor,
    None,
]:
    """构造四个模态均含有效 token 的摘要拼接测试输入。

    - 返回第 1 项: 浮点 (2,16) 候选查询。
    - 返回第 2 项: 含 a、p、v、pp 四个键的模态映射。
    - 返回第 2 项[a]: 浮点 (2,3,16) 原子 token 与 bool (2,3) 掩码。
    - 返回第 2 项[p]: 浮点 (2,3,16) P 模态 token 与 bool (2,3) 掩码。
    - 返回第 2 项[v]: 浮点 (2,3,16) 体素 token 与 bool (2,3) 掩码。
    - 返回第 2 项[pp]: 浮点 (2,3,16) 概率图 token 与 bool (2,3) 掩码；每个样本至少有两个有效 token。
    - 返回第 3 项: 浮点 (2,12) 手动统计。
    - 返回第 4 项: 浮点 (2,4) 采样统计。
    - 返回第 5 项: None；摘要拼接结构不使用 token 注意力偏置。
    """

    candidate = torch.randn(2, 16)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    modalities = {
        name: (torch.randn(2, 3, 16), mask.clone())
        for name in ("a", "p", "v", "pp")
    }
    return candidate, modalities, torch.randn(2, 12), torch.randn(2, 4), None


@pytest.mark.parametrize("autocast_enabled", (False, True))
def test_concatenated_aggregator_zero_residual_exactly_matches_mean_max(
    autocast_enabled: bool,
) -> None:
    """零输出残差在 FP32/BF16 下逐值退化，并保持旧参数与随机状态。"""

    torch.manual_seed(12031)
    baseline = MeanMaxAggregator(16, 0.0, True, True).eval()
    baseline_rng_state = torch.get_rng_state().clone()
    torch.manual_seed(12031)
    concatenated = MeanMaxCandidateConditionedResidualAggregator(
        16,
        4,
        0.0,
        True,
        True,
    ).eval()
    concatenated_rng_state = torch.get_rng_state().clone()
    assert torch.equal(baseline_rng_state, concatenated_rng_state)
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, concatenated.mean_max.state_dict()[name]), name

    residual_final = cast(torch.nn.Linear, concatenated.residual[-1])
    assert torch.count_nonzero(residual_final.weight) == 0
    assert torch.count_nonzero(residual_final.bias) == 0
    inputs = _concatenated_aggregator_inputs()
    with torch.inference_mode():
        if autocast_enabled:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                expected = baseline(*inputs)
                actual = concatenated(*inputs)
        else:
            expected = baseline(*inputs)
            actual = concatenated(*inputs)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("autocast_enabled", (False, True))
def test_concatenated_full_matcher_preserves_common_initialization_and_outputs(
    tmp_path: Path,
    autocast_enabled: bool,
) -> None:
    """固定种子下旧公共参数、后续随机状态和全部正式前向输出保持逐值相等。"""

    batch = _batch(tmp_path, "full_matcher")
    torch.manual_seed(77221)
    baseline = ClosedSetMatcher(_small_config("mean_max")).eval()
    baseline_rng_state = torch.get_rng_state().clone()
    torch.manual_seed(77221)
    concatenated = ClosedSetMatcher(
        _small_config("mean_max_candidate_cross_attention_concat")
    ).eval()
    concatenated_rng_state = torch.get_rng_state().clone()
    assert torch.equal(baseline_rng_state, concatenated_rng_state)

    concatenated_state = concatenated.state_dict()
    for name, value in baseline.state_dict().items():
        target_name = name
        prefix = "candidate_encoder.aggregator."
        if name.startswith(prefix):
            target_name = f"{prefix}mean_max.{name[len(prefix):]}"
        assert torch.equal(value, concatenated_state[target_name]), name

    with torch.inference_mode():
        if autocast_enabled:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                expected = baseline(batch)
                actual = concatenated(batch)
        else:
            expected = baseline(batch)
            actual = concatenated(batch)
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name

    concatenated_aggregator = cast(
        MeanMaxCandidateConditionedResidualAggregator,
        concatenated.candidate_encoder.aggregator,
    )
    class_head_first = cast(torch.nn.Linear, concatenated.candidate_encoder.class_head[0])
    assert not torch.equal(
        concatenated_aggregator.conditioned.branches["a"].query.weight,
        class_head_first.weight,
    )


@pytest.mark.parametrize("autocast_enabled", (False, True))
def test_concatenated_training_forward_preserves_common_dropout_stream(
    tmp_path: Path,
    autocast_enabled: bool,
) -> None:
    """正式丢弃率下，零残差保持训练态输出与共同前向后的随机状态逐值相等。"""

    batch = _batch(tmp_path, "candidate_pretrain")
    torch.manual_seed(42031)
    baseline = CandidateEncoder(_small_config("mean_max", dropout=0.10)).train()
    baseline_construction_state = torch.get_rng_state().clone()
    torch.manual_seed(42031)
    concatenated = CandidateEncoder(
        _small_config(
            "mean_max_candidate_cross_attention_concat",
            dropout=0.10,
        )
    ).train()
    concatenated_construction_state = torch.get_rng_state().clone()
    assert torch.equal(baseline_construction_state, concatenated_construction_state)

    torch.manual_seed(77519)
    if autocast_enabled:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = baseline(batch)
    else:
        expected = baseline(batch)
    expected_rng_state = torch.get_rng_state().clone()

    torch.manual_seed(77519)
    if autocast_enabled:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = concatenated(batch)
    else:
        actual = concatenated(batch)
    actual_rng_state = torch.get_rng_state().clone()

    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name
    assert torch.equal(expected_rng_state, actual_rng_state)


def test_concatenated_aggregator_releases_gradients_in_three_steps() -> None:
    """三次反向传播与两次手工参数扰动依次释放残差、条件摘要和 PP 分支梯度。"""

    torch.manual_seed(99117)
    aggregator = MeanMaxCandidateConditionedResidualAggregator(
        16,
        4,
        0.0,
        True,
        True,
    ).train()
    inputs = _concatenated_aggregator_inputs()
    target = torch.randn(2, 16)
    residual_first = cast(torch.nn.Linear, aggregator.residual[1])
    residual_final = cast(torch.nn.Linear, aggregator.residual[-1])
    conditioned_final = cast(torch.nn.Linear, aggregator.conditioned.fusion[4])
    pp_branch = aggregator.conditioned.branches["pp"]

    (aggregator(*inputs) * target).sum().backward()
    assert residual_final.weight.grad is not None
    assert torch.count_nonzero(residual_final.weight.grad) > 0
    assert residual_first.weight.grad is not None
    assert torch.count_nonzero(residual_first.weight.grad) == 0
    assert conditioned_final.weight.grad is not None
    assert torch.count_nonzero(conditioned_final.weight.grad) == 0
    with torch.no_grad():
        residual_final.weight.add_(residual_final.weight.grad, alpha=-0.05)
        assert residual_final.bias.grad is not None
        residual_final.bias.add_(residual_final.bias.grad, alpha=-0.05)
    aggregator.zero_grad(set_to_none=True)

    (aggregator(*inputs) * target).sum().backward()
    assert residual_first.weight.grad is not None
    assert torch.count_nonzero(residual_first.weight.grad) > 0
    assert conditioned_final.weight.grad is not None
    assert torch.count_nonzero(conditioned_final.weight.grad) > 0
    assert pp_branch.output.weight.grad is not None
    assert torch.count_nonzero(pp_branch.output.weight.grad) > 0
    for name in ("a", "p", "v"):
        branch = aggregator.conditioned.branches[name]
        for projection in (branch.query, branch.key, branch.value, branch.output):
            assert projection.weight.grad is not None
            assert torch.count_nonzero(projection.weight.grad) > 0
    for projection in (pp_branch.query, pp_branch.key, pp_branch.value):
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad) == 0
    with torch.no_grad():
        pp_branch.output.weight.add_(pp_branch.output.weight.grad, alpha=-0.05)
        assert pp_branch.output.bias.grad is not None
        pp_branch.output.bias.add_(pp_branch.output.bias.grad, alpha=-0.05)
    aggregator.zero_grad(set_to_none=True)

    (aggregator(*inputs) * target).sum().backward()
    for projection in (pp_branch.query, pp_branch.key, pp_branch.value):
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad) > 0


def test_candidate_encoder_checkpoint_loader_separates_mean_max_and_concat(
    tmp_path: Path,
) -> None:
    """正式加载入口接受同构参数，并对 mean/max 与摘要拼接的双向错配失败关闭。"""

    baseline_source = CandidateEncoder(_small_config("mean_max"))
    concatenated_source = CandidateEncoder(
        _small_config("mean_max_candidate_cross_attention_concat")
    )
    baseline_path = tmp_path / "mean_max.pt"
    concatenated_path = tmp_path / "concat.pt"
    save_checkpoint(
        baseline_path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state": baseline_source.state_dict(),
        },
    )
    save_checkpoint(
        concatenated_path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state": concatenated_source.state_dict(),
        },
    )

    concatenated_target = CandidateEncoder(
        _small_config("mean_max_candidate_cross_attention_concat")
    )
    baseline_target = CandidateEncoder(_small_config("mean_max"))
    assert load_candidate_encoder_checkpoint(baseline_target, baseline_path) == {
        "missing_keys": [],
        "unexpected_keys": [],
    }
    assert load_candidate_encoder_checkpoint(concatenated_target, concatenated_path) == {
        "missing_keys": [],
        "unexpected_keys": [],
    }
    for name, value in concatenated_source.state_dict().items():
        assert torch.equal(value, concatenated_target.state_dict()[name]), name

    with pytest.raises(RuntimeError):
        load_candidate_encoder_checkpoint(concatenated_target, baseline_path)
    with pytest.raises(RuntimeError):
        load_candidate_encoder_checkpoint(baseline_target, concatenated_path)


@pytest.mark.parametrize("autocast_enabled", (False, True))
def test_zero_initialized_attention_bias_is_exactly_the_unenhanced_aggregator(
    autocast_enabled: bool,
) -> None:
    """零偏置逐值退化，并保留共同参数初值及后续模块可见的随机状态。"""

    torch.manual_seed(90210)
    baseline = CandidateConditionedAggregator(16, 4, 0.0, True, True, False).eval()
    baseline_rng_state = torch.get_rng_state()
    torch.manual_seed(90210)
    enhanced = CandidateConditionedAggregator(16, 4, 0.0, True, True, True).eval()
    enhanced_rng_state = torch.get_rng_state()
    assert torch.equal(baseline_rng_state, enhanced_rng_state)
    baseline_state = baseline.state_dict()
    enhanced_state = enhanced.state_dict()
    assert all(torch.equal(value, enhanced_state[name]) for name, value in baseline_state.items())
    assert enhanced.attention_biases is not None
    assert len({id(module) for module in enhanced.attention_biases.values()}) == 4
    for module in enhanced.attention_biases.values():
        final = cast(CandidateTokenAttentionBias, module).layers[-1]
        assert isinstance(final, torch.nn.Linear)
        assert torch.count_nonzero(final.weight) == 0
        assert torch.count_nonzero(final.bias) == 0

    inputs = _attention_aggregator_inputs()
    with torch.no_grad():
        if autocast_enabled:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                baseline_output = baseline(*inputs[:4], None)
                enhanced_output = enhanced(*inputs)
        else:
            baseline_output = baseline(*inputs[:4], None)
            enhanced_output = enhanced(*inputs)
    assert torch.equal(baseline_output, enhanced_output)


def test_attention_bias_ignores_padding_and_pp_wakes_after_output_projection_step() -> None:
    """无效 token 偏置不改变输出；PP 零出口更新一次后偏置末层权重获得梯度。"""

    aggregator = CandidateConditionedAggregator(16, 4, 0.0, False, False, True).train()
    assert aggregator.attention_biases is not None
    query = torch.randn(2, 16)
    tokens = torch.randn(2, 3, 16)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    features = torch.randn(2, 3, 4)

    a_bias = aggregator.attention_biases["a"]
    a_final = cast(CandidateTokenAttentionBias, a_bias).layers[-1]
    assert isinstance(a_final, torch.nn.Linear)
    with torch.no_grad():
        a_final.weight.normal_()
    with torch.no_grad():
        original = aggregator.branches["a"](query, tokens, mask, a_bias(features))
        altered_features = features.clone()
        altered_features[~mask] = 1.0e4
        altered = aggregator.branches["a"](
            query,
            tokens,
            mask,
            a_bias(altered_features),
        )
    assert torch.equal(original, altered)

    pp_bias = aggregator.attention_biases["pp"]
    pp_branch = aggregator.branches["pp"]
    first = pp_branch(query, tokens, mask, pp_bias(features))
    assert torch.count_nonzero(first) == 0
    first.sum().backward()
    assert pp_branch.output.weight.grad is not None
    assert torch.count_nonzero(pp_branch.output.weight.grad) > 0
    pp_final = cast(CandidateTokenAttentionBias, pp_bias).layers[-1]
    assert isinstance(pp_final, torch.nn.Linear)
    assert pp_final.weight.grad is not None
    assert torch.count_nonzero(pp_final.weight.grad) == 0
    with torch.no_grad():
        pp_branch.output.weight.add_(pp_branch.output.weight.grad, alpha=-0.1)
    pp_branch.zero_grad(set_to_none=True)
    pp_bias.zero_grad(set_to_none=True)
    second = pp_branch(query, tokens, mask, pp_bias(features))
    second.square().sum().backward()
    assert pp_final.weight.grad is not None
    assert torch.count_nonzero(pp_final.weight.grad) > 0


def test_nonzero_output_attention_branches_wake_bias_final_weights_on_first_backward() -> None:
    """A/P/V 含有效 token 且损失回传时，第一次反向传播唤醒各自偏置末层权重。"""

    aggregator = CandidateConditionedAggregator(16, 4, 0.0, False, False, True).train()
    assert aggregator.attention_biases is not None
    query = torch.randn(2, 16)
    tokens = torch.randn(2, 4, 16)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    features = torch.randn(2, 4, 4)
    target = torch.randn(2, 16)
    for name in ("a", "p", "v"):
        bias_module = aggregator.attention_biases[name]
        increment = aggregator.branches[name](query, tokens, mask, bias_module(features))
        (increment * target).sum().backward()
        final = cast(CandidateTokenAttentionBias, bias_module).layers[-1]
        assert isinstance(final, torch.nn.Linear)
        assert final.weight.grad is not None
        assert torch.count_nonzero(final.weight.grad) > 0


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


def test_identity_manual_and_sampling_features_are_learnable_but_class_probability_is_detached() -> None:
    """身份 MLP 可学习手动/采样统计，但身份类别概率不能反向更新六分类 logits。"""

    torch.manual_seed(3407)
    scorer = IdentityScorer(
        model_dim=16,
        dropout=0.0,
        use_manual_features=True,
        use_sampling_features=True,
        initial_graph_gate=0.0,
        initial_inventory_gate=0.0,
    ).train()
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
    sampling_features = torch.tensor(
        [[1.0, 12.0, 8.0, 0.75], [0.0, 0.5, 0.5, 1.0]],
        requires_grad=True,
    )
    scores = scorer(
        candidate,
        representations,
        identity_mask,
        class_logits,
        identity_class_index,
        manual_features,
        sampling_features,
    )
    scores["logits"].sum().backward()
    assert class_logits.grad is None or torch.count_nonzero(class_logits.grad) == 0
    assert manual_features.grad is not None
    assert torch.count_nonzero(manual_features.grad) > 0
    assert sampling_features.grad is not None
    assert torch.count_nonzero(sampling_features.grad) > 0
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

    scorer = IdentityScorer(
        model_dim=16,
        dropout=0.0,
        use_manual_features=False,
        use_sampling_features=False,
        initial_graph_gate=0.0,
        initial_inventory_gate=0.0,
    ).train()
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
    native_only = scorer.native(candidate, representations["native"], None, None, None)
    scores = scorer(
        candidate,
        representations,
        mask,
        torch.zeros((2, 6)),
        torch.full((2, 3), 4, dtype=torch.long),
        torch.zeros((2, 12)),
        torch.zeros((2, 4)),
    )
    assert torch.equal(scores["logits"], native_only)
    scores["logits"].sum().backward()
    assert scorer.graph_gate.grad is not None
    assert scorer.inventory_gate.grad is not None
    assert torch.isfinite(scorer.graph_gate.grad)
    assert torch.isfinite(scorer.inventory_gate.grad)


def test_identity_score_components_preserve_configured_gates_and_sum_to_logits() -> None:
    """非零初始门必须逐值进入两个增量，三组件之和必须等于正式身份分数。"""

    torch.manual_seed(3407)
    scorer = IdentityScorer(
        model_dim=16,
        dropout=0.0,
        use_manual_features=False,
        use_sampling_features=False,
        initial_graph_gate=0.2,
        initial_inventory_gate=0.3,
    ).eval()
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
    class_logits = torch.zeros((2, 6))
    identity_class_index = torch.full((2, 3), 4, dtype=torch.long)
    manual = torch.zeros((2, 12))
    sampling = torch.zeros((2, 4))
    with torch.inference_mode():
        scores = scorer(
            candidate,
            representations,
            mask,
            class_logits,
            identity_class_index,
            manual,
            sampling,
        )
        components = scores["components"]
        logits = scores["logits"]
        raw_graph = scorer.graph(candidate, representations["graph"], None, None, None)
        raw_inventory = scorer.inventory(
            candidate,
            representations["inventory"],
            None,
            None,
            None,
        )
    assert float(scorer.graph_gate) == pytest.approx(0.2)
    assert float(scorer.inventory_gate) == pytest.approx(0.3)
    assert torch.allclose(components["graph_increment"], 0.2 * raw_graph)
    assert torch.allclose(components["inventory_increment"], 0.3 * raw_inventory)
    assert torch.equal(
        logits,
        components["native"]
        + components["graph_increment"]
        + components["inventory_increment"],
    )


def test_single_identity_inventory_score_stays_zero_after_gate_opens() -> None:
    """即使清单门已训练为非零，单身份样本仍没有可贡献的其他身份上下文。"""

    scorer = IdentityScorer(
        model_dim=16,
        dropout=0.0,
        use_manual_features=False,
        use_sampling_features=False,
        initial_graph_gate=0.0,
        initial_inventory_gate=0.0,
    ).eval()
    with torch.no_grad():
        scorer.inventory_gate.fill_(1.0)
        scorer.graph_gate.zero_()
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
        scores = scorer(
            candidate,
            representations,
            mask,
            torch.zeros((2, 6)),
            torch.full((2, 1), 4, dtype=torch.long),
            torch.zeros((2, 12)),
            torch.zeros((2, 4)),
        )
        native_only = scorer.native(candidate, native, None, None, None)
    assert torch.equal(scores["logits"], native_only)


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
    values = _small_config().to_dict()
    values["use_candidate_token_attention_bias"] = True
    with pytest.raises(ValueError, match="candidate_cross_attention"):
        MatcherModelConfig.from_dict(values)
    concatenated = _small_config("mean_max_candidate_cross_attention_concat")
    assert MatcherModelConfig.from_dict(concatenated.to_dict()) == concatenated
    values = concatenated.to_dict()
    values["use_candidate_token_attention_bias"] = True
    with pytest.raises(ValueError, match="candidate_cross_attention"):
        MatcherModelConfig.from_dict(values)
    values = _small_config().to_dict()
    values["initial_graph_gate"] = -0.01
    with pytest.raises(ValueError, match="initial_graph_gate"):
        MatcherModelConfig.from_dict(values)
    values = _small_config().to_dict()
    values["initial_inventory_gate"] = 1.01
    with pytest.raises(ValueError, match="initial_inventory_gate"):
        MatcherModelConfig.from_dict(values)

    volume = torch.zeros((1, 2, 4, 4, 4))
    indices = torch.tensor([[[9, 0, 0]]])
    with pytest.raises(ValueError, match="超出"):
        gather_voxel_features(volume, indices, torch.tensor([[True]]))
    padded = gather_voxel_features(volume, indices, torch.tensor([[False]]))
    assert torch.count_nonzero(padded) == 0


def test_model_config_migrates_historical_feature_flags_and_identity_gates() -> None:
    """历史产物缺少的特征开关和身份门初值应按既有零增量结构迁移。"""

    legacy_values = _small_config().to_dict()
    del legacy_values["use_manual_features"]
    del legacy_values["use_sampling_features"]
    del legacy_values["use_candidate_token_attention_bias"]
    del legacy_values["initial_graph_gate"]
    del legacy_values["initial_inventory_gate"]
    migrated = MatcherModelConfig.from_dict(legacy_values)
    assert migrated.use_manual_features is False
    assert migrated.use_sampling_features is False
    assert migrated.use_candidate_token_attention_bias is False
    assert migrated.initial_graph_gate == 0.0
    assert migrated.initial_inventory_gate == 0.0

    batch1_values = _small_config().to_dict()
    del batch1_values["use_sampling_features"]
    del batch1_values["use_candidate_token_attention_bias"]
    del batch1_values["initial_graph_gate"]
    del batch1_values["initial_inventory_gate"]
    migrated_batch1 = MatcherModelConfig.from_dict(batch1_values)
    assert migrated_batch1.use_manual_features is True
    assert migrated_batch1.use_sampling_features is False
    assert migrated_batch1.use_candidate_token_attention_bias is False
    assert migrated_batch1.initial_graph_gate == 0.0
    assert migrated_batch1.initial_inventory_gate == 0.0

    batch2_values = _small_config("candidate_cross_attention").to_dict()
    del batch2_values["use_candidate_token_attention_bias"]
    del batch2_values["initial_graph_gate"]
    del batch2_values["initial_inventory_gate"]
    migrated_batch2 = MatcherModelConfig.from_dict(batch2_values)
    assert migrated_batch2.use_manual_features is True
    assert migrated_batch2.use_sampling_features is True
    assert migrated_batch2.use_candidate_token_attention_bias is False
    assert migrated_batch2.initial_graph_gate == 0.0
    assert migrated_batch2.initial_inventory_gate == 0.0
