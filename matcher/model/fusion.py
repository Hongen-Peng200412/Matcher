"""编码 A/P/V/PP 原始 token，并以 ``MeanMaxAggregator`` 或 ``CandidateConditionedAggregator`` 返回 ``(B,D)`` 候选表示。符号 ``B/N/F/D/C/H`` 分别表示批次、token 数、原特征宽度、模型宽度、体素通道和注意力头数。"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ..schema import A_TOKEN_DIM, P_TOKEN_DIM, PP_TOKEN_DIM, V_TOKEN_DIM


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
    """按中心 ``48³`` 的整数 ZYX 索引读取 V/PP 特征，返回 ``(B,N,C)``。"""

    if volume.ndim != 5 or index_zyx.ndim != 3 or index_zyx.shape[-1] != 3:
        raise ValueError("volume 必须为 (B,C,D,H,W)，index_zyx 必须为 (B,N,3)。")
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
    """对 ``(B,N,D)`` 变长 token 返回均值和最大值，空集合返回两个 ``(B,D)`` 零张量。"""

    if tokens.ndim != 3 or tokens.shape[:2] != mask.shape:
        raise ValueError("tokens/mask 形状不一致。")
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


# ================================================================================================


class TokenProjector(nn.Module):
    """逐 token 投影一种原始模态。构造参数为输入宽度、模型宽度和 dropout；前向输入 ``(B,N,F)`` 与 ``(B,N)`` 布尔掩码，真值表示有效 token，输出 dtype 由投影算子决定的 ``(B,N,D)``。"""

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
        """返回补零位置严格为零的 ``(B,N,D)`` 投影。"""

        if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
            raise ValueError("tokens 必须为 (B,N,F)，mask 必须为对应 (B,N)。")
        projected = self.projection(tokens)
        return projected * mask[..., None].to(dtype=projected.dtype)


class CandidateCrossAttention(nn.Module):
    """以候选摘要查询一种模态。构造参数为模型宽度、头数、dropout 和零输出开关；前向输入 ``(B,D)`` query、``(B,N,D)`` token 与真值表示有效 token 的 ``(B,N)`` 掩码，输出 ``(B,D)`` 增量。"""

    def __init__(
        self,
        model_dim: int,
        head_count: int,
        dropout: float,
        zero_output: bool,
    ) -> None:
        super().__init__()
        if model_dim % head_count != 0:
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
    ) -> torch.Tensor:
        """在每个 ``B/H`` 内对 query 和 key 的 ``D/H`` 通道求点积，对 token 轴 ``N`` 做 softmax 并求 value 加权和；``(B,1,1,N)`` 有效掩码向 ``H`` 广播，无有效 token 时返回 ``(B,D)`` 严格零增量。"""

        if candidate_query.ndim != 2 or candidate_query.shape[1] != self.model_dim:
            raise ValueError("candidate_query 必须为 (B,model_dim)。")
        if tokens.ndim != 3 or tokens.shape[:2] != mask.shape:
            raise ValueError("tokens/mask 的批次和 token 轴不一致。")
        if tokens.shape[0] != candidate_query.shape[0] or tokens.shape[2] != self.model_dim:
            raise ValueError("交叉注意力的批次或通道宽度不一致。")
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


class MeanMaxAggregator(nn.Module):
    """对四模态取掩码均值/最大值。构造参数为模型宽度和 dropout；前向输入 ``(B,D)`` 候选摘要与 A/P/V/PP token/掩码映射，输出 ``(B,D)`` 候选表示。"""

    def __init__(self, model_dim: int, dropout: float) -> None:
        super().__init__()
        # 一个候选摘要加四种模态的 mean/max，共九个同宽向量。
        self.fusion = nn.Sequential(
            nn.LayerNorm(model_dim * 9),
            nn.Linear(model_dim * 9, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(
        self,
        candidate_summary: torch.Tensor,
        modalities: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """按 A、P、V、PP 固定顺序返回候选级融合表示。"""

        summaries = [candidate_summary]
        for name in ("a", "p", "v", "pp"):
            tokens, mask = modalities[name]
            mean, maximum = _masked_mean_max(tokens, mask)
            summaries.extend((mean, maximum))
        return self.fusion(torch.cat(summaries, dim=-1))


class CandidateConditionedAggregator(nn.Module):
    """用候选摘要分别查询四模态。构造参数为模型宽度、头数和 dropout；前向输入 ``(B,D)`` 候选摘要与 A/P/V/PP token/掩码映射，输出 ``(B,D)`` 候选表示。"""

    def __init__(self, model_dim: int, head_count: int, dropout: float) -> None:
        super().__init__()
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
        self.fusion = nn.Sequential(
            nn.LayerNorm(model_dim * 5),
            nn.Linear(model_dim * 5, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(
        self,
        candidate_summary: torch.Tensor,
        modalities: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """返回候选摘要和 A/P/V/PP 四个独立交叉注意力增量的融合。"""

        increments = []
        for name in ("a", "p", "v", "pp"):
            tokens, mask = modalities[name]
            increments.append(self.branches[name](candidate_summary, tokens, mask))
        return self.fusion(torch.cat((candidate_summary, *increments), dim=-1))
