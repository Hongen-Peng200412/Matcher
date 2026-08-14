"""编码 A/P/V/PP 原始 token，并以两种聚合器返回 ``(B,D_m)`` 候选表示。

符号 ``B/N/F/D_m/C_v/K`` 分别表示批次、token 数、原特征宽度、模型宽度、
体素特征通道和注意力头数；``Z/Y/X`` 表示三个体素空间轴。
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from ..schema import (
    A_TOKEN_DIM,
    CANDIDATE_MANUAL_FEATURE_DIM,
    CANDIDATE_SAMPLING_FEATURE_DIM,
    CANDIDATE_TOKEN_ATTENTION_BIAS_FEATURE_DIM,
    LOCAL_CONTEXT_HALF_EXTENT_VOXELS,
    P_TOKEN_DIM,
    PP_TOKEN_DIM,
    V_TOKEN_DIM,
)


MODALITY_INPUT_DIMS = {
    "a": A_TOKEN_DIM,
    "p": P_TOKEN_DIM,
    "v": V_TOKEN_DIM,
    "pp": PP_TOKEN_DIM,
}


def gather_voxel_features(
    volume: torch.Tensor,
    index_zyx: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """按整数 ZYX 索引从 ``(B,C_v,Z,Y,X)`` 读取 V/PP，返回 ``(B,N,C_v)``。

    ``index_zyx`` 是 int64 ``(B,N,3)``；``mask`` 必须是 bool ``(B,N)``，
    真值表示相应索引和 token 有效，假值位置返回严格零。
    """

    if volume.ndim != 5 or index_zyx.ndim != 3 or index_zyx.shape[-1] != 3:
        raise ValueError("volume 必须为 (B,C_v,Z,Y,X)，index_zyx 必须为 (B,N,3)。")
    if index_zyx.dtype != torch.long or mask.dtype != torch.bool:
        raise TypeError("index_zyx 必须是 int64，mask 必须是 bool。")
    if index_zyx.shape[:2] != mask.shape or index_zyx.shape[0] != volume.shape[0]:
        raise ValueError("体素索引、掩码和特征图的批次/token 轴不一致。")
    valid_indices = index_zyx[mask]
    if valid_indices.numel() and (
        torch.any(valid_indices < 0)
        or torch.any(valid_indices[:, 0] >= volume.shape[2])
        or torch.any(valid_indices[:, 1] >= volume.shape[3])
        or torch.any(valid_indices[:, 2] >= volume.shape[4])
    ):
        raise ValueError("V/PP 的有效 ZYX 索引超出 U-Net 特征图。")
    safe = index_zyx.clamp_min(0)
    safe_z = safe[..., 0].clamp_max(volume.shape[2] - 1)
    safe_y = safe[..., 1].clamp_max(volume.shape[3] - 1)
    safe_x = safe[..., 2].clamp_max(volume.shape[4] - 1)
    batch = torch.arange(volume.shape[0], device=volume.device)[:, None]
    channel_last = volume.permute(0, 2, 3, 4, 1)
    selected = channel_last[batch, safe_z, safe_y, safe_x]
    return selected * mask[..., None].to(dtype=selected.dtype)


def _masked_mean_max(tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对 ``(B,N,D_m)`` 变长 token 返回均值和最大值。

    ``mask`` 必须是 bool ``(B,N)``，真值表示有效 token；空集合返回两个
    ``(B,D_m)`` 零张量。
    """

    if tokens.ndim != 3 or tokens.shape[:2] != mask.shape:
        raise ValueError("tokens/mask 形状不一致。")
    if mask.dtype != torch.bool:
        raise TypeError("mask 必须是 bool。")
    if tokens.shape[1] == 0:
        empty = torch.zeros(
            (tokens.shape[0], tokens.shape[2]),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return empty, empty
    float_mask = mask[..., None].to(dtype=tokens.dtype)
    mean = (tokens * float_mask).sum(dim=1) / float_mask.sum(dim=1).clamp_min(1.0)
    maximum = tokens.masked_fill(~mask[..., None], torch.finfo(tokens.dtype).min).amax(dim=1)
    maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
    return mean, maximum


def normalize_candidate_manual_features(features: torch.Tensor) -> torch.Tensor:
    """对 ``(B,12)`` 手动特征仅执行 ``log1p(count)``，保留固化概率原值。"""

    if features.ndim != 2 or features.shape[1] != CANDIDATE_MANUAL_FEATURE_DIM:
        raise ValueError(
            "candidate_manual_features 必须为"
            f" (B,{CANDIDATE_MANUAL_FEATURE_DIM})。"
        )
    if not torch.isfinite(features).all() or torch.any(features < 0):
        raise ValueError("候选手动特征必须是有限非负数。")
    grouped = features.reshape(features.shape[0], -1, 3)
    normalized = torch.cat(
        (torch.log1p(grouped[..., :1]), grouped[..., 1:]),
        dim=-1,
    )
    return normalized.reshape_as(features)


def normalize_candidate_sampling_features(features: torch.Tensor) -> torch.Tensor:
    """把 ``(B,4)`` 采样特征归一化为稳定的无量纲输入。"""

    if features.ndim != 2 or features.shape[1] != CANDIDATE_SAMPLING_FEATURE_DIM:
        raise ValueError(
            "candidate_sampling_features 必须为"
            f" (B,{CANDIDATE_SAMPLING_FEATURE_DIM})。"
        )
    if not torch.isfinite(features).all():
        raise ValueError("候选采样特征必须全部有限。")
    if torch.any((features[:, 0] != 0.0) & (features[:, 0] != 1.0)):
        raise ValueError("crop_boundary_limited 必须为 0 或 1。")
    if torch.any(features[:, 1:3] < 0.0):
        raise ValueError("候选中心偏移距离不能为负。")
    if torch.any((features[:, 3] < 0.0) | (features[:, 3] > 1.0)):
        raise ValueError("v_retained_fraction 必须位于 [0,1]。")
    return torch.cat(
        (
            features[:, 0:1],
            features[:, 1:2]
            / (LOCAL_CONTEXT_HALF_EXTENT_VOXELS * math.sqrt(3.0)),
            features[:, 2:3] / LOCAL_CONTEXT_HALF_EXTENT_VOXELS,
            features[:, 3:4],
        ),
        dim=1,
    )


def candidate_token_attention_bias_features(tokens: torch.Tensor) -> torch.Tensor:
    """从原始模态 token 提取候选—实体注意力偏置输入。

    ``tokens`` 是浮点 ``(B,N,F)``，其中 ``B/N/F`` 分别是批次、token 数和原始特征宽度。
    ``tokens[..., -4:]`` 依次是有限 ``[0,1]`` Stage1 固化概率与无量纲相对 XYZ。
    A/P 相对 XYZ 以实际裁剪中心为原点，并按完整 ``80³`` 物理半边长归一化。
    V/PP 相对 XYZ 以实际裁剪中心为原点，并按实际 ``48³`` 物理半边长归一化。
    返回浮点 ``(B,N,4)``，与输入 token 轴逐项对齐，末轴顺序为 ``relative_x/relative_y/relative_z/stage1_probability``。
    """

    if tokens.ndim != 3 or tokens.shape[2] < CANDIDATE_TOKEN_ATTENTION_BIAS_FEATURE_DIM:
        raise ValueError("原始模态 token 必须为 (B,N,F)，且 F 不得小于 4。")
    return torch.cat((tokens[..., -3:], tokens[..., -4:-3]), dim=-1)


# ================================================================================================


class TokenProjector(nn.Module):
    """逐 token 投影一种原始模态。构造参数为输入宽度、模型宽度和 dropout；前向输入 ``(B,N,F)`` 与 ``(B,N)`` 布尔掩码，真值表示有效 token，输出 dtype 由投影算子决定的 ``(B,N,D_m)``。"""

    def __init__(self, input_dim: int, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """返回补零位置严格为零的 ``(B,N,D_m)`` 投影。"""

        if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
            raise ValueError("tokens 必须为 (B,N,F)，mask 必须为对应 (B,N)。")
        if mask.dtype != torch.bool:
            raise TypeError("mask 必须是 bool。")
        projected = self.projection(tokens)
        return projected * mask[..., None].to(dtype=projected.dtype)


class CandidateCrossAttention(nn.Module):
    """以候选摘要查询一种模态。

    ``model_dim`` 是模型宽度 ``D_m``。
    ``head_count`` 是注意力头数 ``K``，且必须整除 ``D_m``。
    ``dropout`` 是注意力权重的丢弃概率。
    ``zero_output`` 决定是否把输出投影权重与偏置初始化为零。
    ``candidate_query`` 是浮点 ``(B,D_m)`` 候选查询。
    ``tokens`` 是浮点 ``(B,N,D_m)`` 模态 token。
    ``mask`` 是 bool ``(B,N)`` 有效掩码，真值 token 参与 softmax。
    ``attention_bias`` 是可选无量纲 ``(B,K,N)`` softmax 前加性分数。
    返回浮点 ``(B,D_m)`` 增量；补零 token 在加性分数加入后由掩码排除。
    """

    def __init__(
        self,
        model_dim: int,
        head_count: int,
        dropout: float,
        zero_output: bool,
    ) -> None:
        super().__init__()
        if head_count <= 0 or model_dim <= 0 or model_dim % head_count != 0:
            raise ValueError("model_dim 必须能被交叉注意力头数整除。")
        self.model_dim = model_dim
        self.head_count = head_count
        self.head_dim = model_dim // head_count
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim, bias=False)
        self.value = nn.Linear(model_dim, model_dim, bias=False)
        self.output = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        if zero_output:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(
        self,
        candidate_query: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        attention_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """在每个 ``B/K`` 内计算内容分数与可选偏置，再沿 ``N`` 归一化。

        ``candidate_query`` 是浮点 ``(B,D_m)`` 候选查询。
        ``tokens`` 是浮点 ``(B,N,D_m)`` 模态 token。
        ``mask`` 是 bool ``(B,N)`` 有效掩码，向注意力头轴广播。
        ``attention_bias`` 是可选无量纲 ``(B,K,N)`` softmax 前加性分数。
        query 与 key 的 ``D_m/K`` 通道点积保留 ``B/K/N`` 轴。
        返回浮点 ``(B,D_m)``；一个样本没有有效 token 时对应增量严格为零。
        """

        if candidate_query.ndim != 2 or candidate_query.shape[1] != self.model_dim:
            raise ValueError("candidate_query 必须为 (B,model_dim)。")
        if tokens.ndim != 3 or tokens.shape[:2] != mask.shape:
            raise ValueError("tokens/mask 的批次和 token 轴不一致。")
        if mask.dtype != torch.bool:
            raise TypeError("mask 必须是 bool。")
        if tokens.shape[0] != candidate_query.shape[0] or tokens.shape[2] != self.model_dim:
            raise ValueError("交叉注意力的批次或通道宽度不一致。")
        if attention_bias is not None and attention_bias.shape != (
            tokens.shape[0],
            self.head_count,
            tokens.shape[1],
        ):
            raise ValueError("attention_bias 必须为与 B/K/N 轴对齐的 (B,K,N)。")
        if tokens.shape[1] == 0:
            return torch.zeros_like(candidate_query)

        batch_size, token_count, _ = tokens.shape
        query = self.query(candidate_query).reshape(
            batch_size, self.head_count, 1, self.head_dim
        )
        key = self.key(tokens).reshape(
            batch_size, token_count, self.head_count, self.head_dim
        ).transpose(1, 2)
        value = self.value(tokens).reshape(
            batch_size, token_count, self.head_count, self.head_dim
        ).transpose(1, 2)
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_bias is not None:
            score = score + attention_bias[:, :, None, :].to(dtype=score.dtype)
        valid = mask[:, None, None, :]
        score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
        weight = torch.softmax(score, dim=-1)
        weight = weight * valid.to(dtype=weight.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        context = torch.matmul(self.dropout(weight), value)
        context = context.reshape(batch_size, self.model_dim)
        increment = self.output(context)
        has_token = mask.any(dim=1, keepdim=True)
        return increment * has_token.to(dtype=increment.dtype)


class CandidateTokenAttentionBias(nn.Module):
    """把逐实体 Stage1 概率与相对坐标转换为逐注意力头分数偏置。

    ``model_dim`` 是隐藏宽度 ``D_m``。
    ``head_count`` 是输出注意力头数 ``K``。
    ``features`` 是浮点 ``(B,N,4)``，第二轴与一种模态的 token 逐项对齐。
    ``features[..., :3]`` 是无量纲相对 XYZ。
    ``features[..., 3]`` 是位于 ``[0,1]`` 的 Stage1 固化概率。
    返回无量纲 ``(B,K,N)`` softmax 前加性分数。
    末层权重与偏置采用零初始化，因此初始输出逐元素为零；训练后不承诺继续为零。
    末层 ``bias`` 对每个注意力头沿 token 轴只产生常数平移，数学上不改变 softmax。
    末层 ``weight`` 是该层有效的可学习路径。
    """

    def __init__(self, model_dim: int, head_count: int) -> None:
        super().__init__()
        if model_dim <= 0 or head_count <= 0:
            raise ValueError("注意力偏置的模型宽度和头数必须为正整数。")
        final = nn.Linear(model_dim, head_count)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.layers = nn.Sequential(
            nn.Linear(CANDIDATE_TOKEN_ATTENTION_BIAS_FEATURE_DIM, model_dim),
            nn.SiLU(),
            final,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """把 ``(B,N,4)`` 偏置输入转换为 ``(B,K,N)`` 分数。"""

        if (
            features.ndim != 3
            or features.shape[2] != CANDIDATE_TOKEN_ATTENTION_BIAS_FEATURE_DIM
        ):
            raise ValueError("候选实体注意力偏置输入必须为 (B,N,4)。")
        return self.layers(features).transpose(1, 2)


class MeanMaxAggregator(nn.Module):
    """对四模态取掩码均值/最大值，并按配置拼接模态统计与采样统计。

    ``model_dim`` 是模型宽度 ``D_m``。
    ``dropout`` 是融合 MLP 的丢弃概率。
    ``include_manual`` 决定是否拼接 12 项模态手动统计。
    ``include_sampling`` 决定是否拼接 4 项候选采样统计。
    ``candidate_summary`` 是浮点 ``(B,D_m)`` 候选摘要。
    ``modalities['a']`` 是浮点 ``(B,N_A,D_m)`` token 与 bool ``(B,N_A)`` 掩码。
    ``modalities['p']`` 是浮点 ``(B,N_P,D_m)`` token 与 bool ``(B,N_P)`` 掩码。
    ``modalities['v']`` 是浮点 ``(B,N_V,D_m)`` token 与 bool ``(B,N_V)`` 掩码。
    ``modalities['pp']`` 是浮点 ``(B,N_PP,D_m)`` token 与 bool ``(B,N_PP)`` 掩码。
    ``manual_features`` 是与开关一致的可选浮点 ``(B,12)`` 模态统计。
    ``sampling_features`` 是与开关一致的可选浮点 ``(B,4)`` 采样统计。
    ``attention_bias_features`` 必须为 ``None``。
    返回浮点 ``(B,D_m)`` 候选表示；各模态假值 token 不参与均值或最大值。
    """

    def __init__(
        self,
        model_dim: int,
        dropout: float,
        include_manual: bool,
        include_sampling: bool,
    ) -> None:
        super().__init__()
        self.include_manual = include_manual
        self.include_sampling = include_sampling
        # 一个候选摘要、可选四维采样统计、四种模态 mean/max 和可选四组三维统计。
        fusion_width = (
            model_dim * 9
            + (CANDIDATE_MANUAL_FEATURE_DIM if include_manual else 0)
            + (CANDIDATE_SAMPLING_FEATURE_DIM if include_sampling else 0)
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_width),
            nn.Linear(fusion_width, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(
        self,
        candidate_summary: torch.Tensor,
        modalities: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
        manual_features: torch.Tensor | None,
        sampling_features: torch.Tensor | None,
        attention_bias_features: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """按候选、采样统计及 A/P/V/PP 各自 mean/max/三项统计的顺序融合。

        ``candidate_summary`` 是浮点 ``(B,D_m)`` 候选摘要。
        ``modalities`` 的四个键、token 和掩码契约见类说明，四种 ``N_m`` 可以不同。
        ``manual_features`` 是与 ``include_manual`` 一致的可选浮点 ``(B,12)``。
        ``sampling_features`` 是与 ``include_sampling`` 一致的可选浮点 ``(B,4)``。
        ``attention_bias_features`` 必须为 ``None``。
        返回浮点 ``(B,D_m)`` 候选表示。
        """

        if attention_bias_features is not None:
            raise ValueError("mean_max 聚合不得接收候选实体注意力偏置输入。")
        summaries = [candidate_summary]
        if self.include_sampling:
            if sampling_features is None or sampling_features.shape != (
                candidate_summary.shape[0],
                CANDIDATE_SAMPLING_FEATURE_DIM,
            ):
                raise ValueError("启用采样特征时必须提供与候选轴对齐的 (B,4) 统计。")
            summaries.append(sampling_features)
        elif sampling_features is not None:
            raise ValueError("未启用采样特征时 sampling_features 必须为 None。")
        grouped_manual = None
        if self.include_manual:
            if manual_features is None or manual_features.shape != (
                candidate_summary.shape[0],
                CANDIDATE_MANUAL_FEATURE_DIM,
            ):
                raise ValueError("启用手动特征时必须提供与候选轴对齐的 (B,12) 统计。")
            grouped_manual = manual_features.reshape(candidate_summary.shape[0], 4, 3)
        elif manual_features is not None:
            raise ValueError("未启用手动特征时 manual_features 必须为 None。")
        for modality_index, name in enumerate(("a", "p", "v", "pp")):
            tokens, mask = modalities[name]
            mean, maximum = _masked_mean_max(tokens, mask)
            summaries.extend((mean, maximum))
            if grouped_manual is not None:
                summaries.append(grouped_manual[:, modality_index, :])
        return self.fusion(torch.cat(summaries, dim=-1))


class CandidateConditionedAggregator(nn.Module):
    """用候选摘要分别查询四模态，并按配置拼接模态统计与采样统计。

    ``model_dim`` 是模型宽度 ``D_m``。
    ``head_count`` 是每个模态分支的注意力头数 ``K``。
    ``dropout`` 是注意力权重和融合 MLP 的丢弃概率。
    ``include_manual`` 决定是否拼接 12 项模态手动统计。
    ``include_sampling`` 决定是否拼接 4 项候选采样统计。
    ``use_token_attention_bias`` 决定是否启用四套独立候选—实体偏置 MLP。
    ``candidate_summary`` 是浮点 ``(B,D_m)`` 候选查询。
    ``modalities['a']`` 是浮点 ``(B,N_A,D_m)`` token 与 bool ``(B,N_A)`` 掩码。
    ``modalities['p']`` 是浮点 ``(B,N_P,D_m)`` token 与 bool ``(B,N_P)`` 掩码。
    ``modalities['v']`` 是浮点 ``(B,N_V,D_m)`` token 与 bool ``(B,N_V)`` 掩码。
    ``modalities['pp']`` 是浮点 ``(B,N_PP,D_m)`` token 与 bool ``(B,N_PP)`` 掩码。
    ``manual_features`` 是与开关一致的可选浮点 ``(B,12)`` 模态统计。
    ``sampling_features`` 是与开关一致的可选浮点 ``(B,4)`` 采样统计。
    ``attention_bias_features['a']`` 是与 A token 逐项对齐的可选浮点 ``(B,N_A,4)``。
    ``attention_bias_features['p']`` 是与 P token 逐项对齐的可选浮点 ``(B,N_P,4)``。
    ``attention_bias_features['v']`` 是与 V token 逐项对齐的可选浮点 ``(B,N_V,4)``。
    ``attention_bias_features['pp']`` 是与 PP token 逐项对齐的可选浮点 ``(B,N_PP,4)``。
    每种偏置输入末轴依次为无量纲相对 XYZ 和位于 ``[0,1]`` 的 Stage1 固化概率。
    返回浮点 ``(B,D_m)`` 候选表示；各模态假值 token 不参与注意力。
    """

    def __init__(
        self,
        model_dim: int,
        head_count: int,
        dropout: float,
        include_manual: bool,
        include_sampling: bool,
        use_token_attention_bias: bool,
    ) -> None:
        super().__init__()
        self.include_manual = include_manual
        self.include_sampling = include_sampling
        self.branches = nn.ModuleDict(
            {
                name: CandidateCrossAttention(
                    model_dim,
                    head_count,
                    dropout,
                    zero_output=name == "pp",
                )
                for name in ("a", "p", "v", "pp")
            }
        )
        fusion_width = (
            model_dim * 5
            + (CANDIDATE_MANUAL_FEATURE_DIM if include_manual else 0)
            + (CANDIDATE_SAMPLING_FEATURE_DIM if include_sampling else 0)
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_width),
            nn.Linear(fusion_width, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.attention_biases: nn.ModuleDict | None = None
        if use_token_attention_bias:
            # 新分支使用隔离的初始化随机流，避免改变未增强模型的共同参数初值。
            with torch.random.fork_rng(devices=[]):
                self.attention_biases = nn.ModuleDict(
                    {
                        name: CandidateTokenAttentionBias(model_dim, head_count)
                        for name in ("a", "p", "v", "pp")
                    }
                )

    def forward(
        self,
        candidate_summary: torch.Tensor,
        modalities: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
        manual_features: torch.Tensor | None,
        sampling_features: torch.Tensor | None,
        attention_bias_features: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """按候选、采样统计及 A/P/V/PP 各自注意力增量/三项统计的顺序融合。

        ``candidate_summary`` 是浮点 ``(B,D_m)`` 候选查询。
        ``modalities`` 的四个键、独立 ``N_m``、token 和掩码契约见类说明。
        ``manual_features`` 是与 ``include_manual`` 一致的可选浮点 ``(B,12)``。
        ``sampling_features`` 是与 ``include_sampling`` 一致的可选浮点 ``(B,4)``。
        ``attention_bias_features`` 的四键与同模态 token 逐项对齐，存在性与 ``use_token_attention_bias`` 一致。
        返回浮点 ``(B,D_m)`` 候选表示。
        """

        if self.attention_biases is None:
            if attention_bias_features is not None:
                raise ValueError("未启用候选实体注意力偏置时对应输入必须为 None。")
        elif attention_bias_features is None or set(attention_bias_features) != {
            "a",
            "p",
            "v",
            "pp",
        }:
            raise ValueError("启用候选实体注意力偏置时必须提供完整 A/P/V/PP 输入。")
        grouped_manual = None
        if self.include_manual:
            if manual_features is None or manual_features.shape != (
                candidate_summary.shape[0],
                CANDIDATE_MANUAL_FEATURE_DIM,
            ):
                raise ValueError("启用手动特征时必须提供与候选轴对齐的 (B,12) 统计。")
            grouped_manual = manual_features.reshape(candidate_summary.shape[0], 4, 3)
        elif manual_features is not None:
            raise ValueError("未启用手动特征时 manual_features 必须为 None。")
        summaries = [candidate_summary]
        if self.include_sampling:
            if sampling_features is None or sampling_features.shape != (
                candidate_summary.shape[0],
                CANDIDATE_SAMPLING_FEATURE_DIM,
            ):
                raise ValueError("启用采样特征时必须提供与候选轴对齐的 (B,4) 统计。")
            summaries.append(sampling_features)
        elif sampling_features is not None:
            raise ValueError("未启用采样特征时 sampling_features 必须为 None。")
        for modality_index, name in enumerate(("a", "p", "v", "pp")):
            tokens, mask = modalities[name]
            attention_bias = None
            if self.attention_biases is not None:
                assert attention_bias_features is not None
                features = attention_bias_features[name]
                if features.shape[:2] != tokens.shape[:2]:
                    raise ValueError(f"{name.upper()} 偏置输入未与对应 token 的 B/N 轴对齐。")
                attention_bias = self.attention_biases[name](features)
            summaries.append(
                self.branches[name](candidate_summary, tokens, mask, attention_bias)
            )
            if grouped_manual is not None:
                summaries.append(grouped_manual[:, modality_index, :])
        return self.fusion(torch.cat(summaries, dim=-1))
