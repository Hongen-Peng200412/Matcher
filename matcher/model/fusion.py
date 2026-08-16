"""投影 A/P/V/PP token，并把四种变长模态融合成候选表示。

gather_voxel_features 为 V 与 PP token 读取同位置 RAUNet 特征。两个归一化函数处理候选级手工统计与裁剪统计。MeanMaxAggregator 对每种模态计算掩码均值和最大值；CandidateConditionedAggregator 以候选体素摘要分别查询四种模态，并可把相对坐标与 Stage1 概率转换成 softmax 前的逐头加性偏置；MeanMaxCandidateConditionedResidualAggregator 拼接前两种完整候选表示，并用零输出残差保持初始化时与 mean/max 主表示逐值相等。本模块只生成候选表示，不读取语言身份、配体图或监督目标。
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

from ..data.schema import (
    A_TOKEN_DIM,
    CANDIDATE_MANUAL_FEATURE_DIM,
    CANDIDATE_SAMPLING_FEATURE_DIM,
    CANDIDATE_TOKEN_ATTENTION_BIAS_FEATURE_DIM,
    LOCAL_CONTEXT_HALF_EXTENT_VOXELS,
    P_TOKEN_DIM,
    PP_TOKEN_DIM,
    V_TOKEN_DIM,
)


# str 到 int 的固定映射；四键是模型内部模态顺序，数值是 CandidateBatch 对应原始 token 的最后一维宽度。
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
    """按局部 Z/Y/X 下标从五维体素特征读取 V 或 PP 位置。

    形状符号：
        - B: 当前物理批次的候选数。
        - C_v: 体素特征通道数。
        - N: 当前模态补零后的 token 数。
        - Z: 体素特征深度。
        - Y: 体素特征高度。
        - X: 体素特征宽度。

    输入参数：
        - volume: 浮点张量，(B,C_v,Z,Y,X)；通道优先的三维特征，本函数不限制具体空间大小。
        - index_zyx: int64 张量，(B,N,3)；每个 token 在 volume 三个空间轴中的 Z/Y/X 下标。
        - mask: bool 张量，(B,N)；True 表示同位置 index_zyx 和 token 有效，False 表示补零位置。

    返回值：
        - sampled: 浮点张量，(B,N,C_v)；与 index_zyx 的 B/N 轴逐项对齐，mask=False 的位置严格为零。

    只对 mask=True 的下标执行范围门禁。补零位置即使保存越界值也先逐轴夹紧到可索引范围，再由 mask 清零；正式 CandidateBatch 的补零下标为零，但本函数不依赖该具体值。
    """

    if volume.ndim != 5 or index_zyx.ndim != 3 or index_zyx.shape[-1] != 3:
        raise ValueError("volume 必须为 (B,C_v,Z,Y,X)，index_zyx 必须为 (B,N,3)。")
    if index_zyx.dtype != torch.long or mask.dtype != torch.bool:
        raise TypeError("index_zyx 必须是 int64，mask 必须是 bool。")
    if index_zyx.shape[:2] != mask.shape or index_zyx.shape[0] != volume.shape[0]:
        raise ValueError("体素索引、掩码和特征图的批次/token 轴不一致。")
    # int64，(N_valid,3)，只含 mask=True 的 Z/Y/X 下标；N_valid 是整个批次的有效 token 总数。
    valid_indices = index_zyx[mask]
    if valid_indices.numel() and (
        torch.any(valid_indices < 0)
        or torch.any(valid_indices[:, 0] >= volume.shape[2])
        or torch.any(valid_indices[:, 1] >= volume.shape[3])
        or torch.any(valid_indices[:, 2] >= volume.shape[4])
    ):
        raise ValueError("V/PP 的有效 ZYX 索引超出 U-Net 特征图。")
    # int64，(B,N,3)，把补零位置或待检查位置的负下标先夹到零；三个上界随后按各空间轴分别处理。
    safe = index_zyx.clamp_min(0)
    # int64，(B,N)，可安全索引 volume 深度轴的 Z 下标。
    safe_z = safe[..., 0].clamp_max(volume.shape[2] - 1)
    # int64，(B,N)，可安全索引 volume 高度轴的 Y 下标。
    safe_y = safe[..., 1].clamp_max(volume.shape[3] - 1)
    # int64，(B,N)，可安全索引 volume 宽度轴的 X 下标。
    safe_x = safe[..., 2].clamp_max(volume.shape[4] - 1)
    # int64，(B,1)，为每个候选的 N 个 token 广播同一个批次下标。
    batch = torch.arange(volume.shape[0], device=volume.device)[:, None]
    # 浮点，(B,Z,Y,X,C_v)，仅改排通道轴，便于一次高级索引得到每个 token 的完整通道向量。
    channel_last = volume.permute(0, 2, 3, 4, 1)
    # 浮点，(B,N,C_v)，安全下标位置的特征；补零位置尚未清零。
    selected = channel_last[batch, safe_z, safe_y, safe_x]
    return selected * mask[..., None].to(dtype=selected.dtype)


def _masked_mean_max(tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """沿一种模态的有效 token 轴计算每个候选的均值和最大值。

    输入参数：
        - tokens: 浮点张量，(B,N,D_m)；一种模态的投影 token。
        - mask: bool 张量，(B,N)；True token 参与归约，False token 被排除。

    返回值：
        - mean: 浮点张量，(B,D_m)；每个候选有效 token 的算术均值，没有有效 token 时为零。
        - maximum: 浮点张量，(B,D_m)；每个候选有效 token 的逐通道最大值，没有有效 token 时为零。

    N=0 与 N>0 但某个候选全为 False 都有显式零结果，不向聚合器传播 NaN 或 dtype 最小有限值。
    """

    if tokens.ndim != 3 or tokens.shape[:2] != mask.shape:
        raise ValueError("tokens/mask 形状不一致。")
    if mask.dtype != torch.bool:
        raise TypeError("mask 必须是 bool。")
    if tokens.shape[1] == 0:
        # 浮点，(B,D_m)，保持 tokens 的 device 和 dtype；两个返回项共享同一个零张量对象但后续只读。
        empty = torch.zeros(
            (tokens.shape[0], tokens.shape[2]),
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return empty, empty
    # 浮点，(B,N,1)，由 bool 有效掩码转换成可广播的 0/1 权重。
    float_mask = mask[..., None].to(dtype=tokens.dtype)
    # 浮点，(B,D_m)，沿 token 轴求和后除以有效数；clamp 只负责全空候选的零分母。
    mean = (tokens * float_mask).sum(dim=1) / float_mask.sum(dim=1).clamp_min(1.0)
    # 浮点，(B,D_m)，先把补零 token 替换成当前 dtype 最小有限值，再沿 token 轴取最大。
    maximum = tokens.masked_fill(~mask[..., None], torch.finfo(tokens.dtype).min).amax(dim=1)
    maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
    return mean, maximum


def normalize_candidate_manual_features(features: torch.Tensor) -> torch.Tensor:
    """压缩四种模态的 token 数，并保留两个 Stage1 概率统计。

    输入参数：
        - features: 浮点张量，(B,12)；按 A/P/V/PP 顺序排列四组三元组，每组依次为 token 数、Stage1 概率均值和 Stage1 概率最大值。

    返回值：
        - normalized: 浮点张量，(B,12)；每组三元组的 token 数替换为 log1p(count)，两个概率统计逐值保持。

    本函数只检查全部输入有限且非负，不检查概率两列是否不超过 1；正式 CandidateDataset 保证其来源是 [0,1] 概率统计。
    """

    if features.ndim != 2 or features.shape[1] != CANDIDATE_MANUAL_FEATURE_DIM:
        raise ValueError(
            "candidate_manual_features 必须为"
            f" (B,{CANDIDATE_MANUAL_FEATURE_DIM})。"
        )
    if not torch.isfinite(features).all() or torch.any(features < 0):
        raise ValueError("候选手动特征必须是有限非负数。")
    # 浮点，(B,4,3)，第二轴固定为 A/P/V/PP，第三轴依次为 token 数、概率均值、概率最大值。
    grouped = features.reshape(features.shape[0], -1, 3)
    # 浮点，(B,4,3)，只替换第三轴的计数分量，另外两个分量保持输入数值和 dtype。
    normalized = torch.cat(
        (torch.log1p(grouped[..., :1]), grouped[..., 1:]),
        dim=-1,
    )
    return normalized.reshape_as(features)


def normalize_candidate_sampling_features(features: torch.Tensor) -> torch.Tensor:
    """把候选裁剪诊断转换成固定尺度的四维无量纲输入。

    输入参数：
        - features: 浮点张量，(B,4)；四列依次为 crop_boundary_limited、center_offset_l2_voxel、center_offset_linf_voxel、v_retained_fraction。
        - features[:,0]: 0/1 浮点标志；1 表示请求的 V 质心裁剪起点至少一轴被限制到合法边界。
        - features[:,1]: 非负浮点数；实际裁剪中心相对完整 V token 质心的 L2 体素距离。
        - features[:,2]: 非负浮点数；同一中心偏移的 L∞ 体素距离。
        - features[:,3]: 范围 [0,1] 的浮点数；完整候选 V token 落入当前裁剪的比例。

    返回值：
        - normalized: 浮点张量，(B,4)；第一列与第四列保持原值，L2 除以 24×sqrt(3)，L∞ 除以 24。

    常数 24 是 LOCAL_CONTEXT_HALF_EXTENT_VOXELS，即固定 48 立方体每轴的半边长；本函数不从批次空间形状动态推导该值。
    """

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

    输入参数：
        - tokens: 浮点张量，(B,N,F)；A、P、V 或 PP 原始 token，F 至少为 4。
        - tokens[...,-4]: 浮点数；正式 CandidateBatch 契约中的 Stage1 固化概率。
        - tokens[...,-3]: 浮点数；以实际裁剪中心为原点的无量纲相对 X。
        - tokens[...,-2]: 浮点数；以实际裁剪中心为原点的无量纲相对 Y。
        - tokens[...,-1]: 浮点数；以实际裁剪中心为原点的无量纲相对 Z。

    返回值：
        - features: 浮点张量，(B,N,4)；与输入 B/N 轴逐项对齐，最后一轴依次为相对 X、相对 Y、相对 Z、Stage1 概率。

    A/P 相对坐标按完整 80 立方体物理半边长归一化，V/PP 按当前 48 立方体物理半边长归一化；这些是正式 token 来源契约，本函数只重排最后四个分量，不重新核验有限性、概率范围或坐标尺度。
    """

    if tokens.ndim != 3 or tokens.shape[2] < CANDIDATE_TOKEN_ATTENTION_BIAS_FEATURE_DIM:
        raise ValueError("原始模态 token 必须为 (B,N,F)，且 F 不得小于 4。")
    return torch.cat((tokens[..., -3:], tokens[..., -4:-3]), dim=-1)


# ================================================================================================


class TokenProjector(nn.Module):
    """独立归一化并投影一种原始 token 模态。

    构造参数：
        - input_dim: 整数 F；该模态原始特征宽度，本类不单独检查正值。
        - model_dim: 整数 D_m；投影后的共同模型宽度，本类不单独检查正值。
        - dropout: 数值；投影中间激活的丢弃概率，合法范围通常由 MatcherModelConfig 保证，本类直接构造时依赖 Dropout 检查。

    前向输入：
        - tokens: 浮点张量，(B,N,F)；一种模态补零后的原始 token。
        - mask: bool 张量，(B,N)；True 表示有效 token，False 表示补零。

    前向输出：
        - projected: 浮点张量，(B,N,D_m)；dtype 由线性层和自动混合精度上下文决定，补零位置严格为零。
    """

    def __init__(self, input_dim: int, model_dim: int, dropout: float) -> None:
        """建立 LayerNorm、线性投影、GELU、Dropout 与输出 LayerNorm。

        输入参数：
            - input_dim: 整数 F；首个 LayerNorm 与线性层读取的原始宽度。
            - model_dim: 整数 D_m；线性层输出及末层 LayerNorm 宽度。
            - dropout: 数值；GELU 后激活的丢弃概率。

        实例状态：
            - self.projection: Sequential；逐 token 作用，不改变 B/N 轴。
        """

        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """投影 token，并在末端重新清零补零位置。

        输入参数：
            - tokens: 浮点张量，(B,N,F)；最后一维必须与构造时 input_dim 一致，该关系由 LayerNorm 检查。
            - mask: bool 张量，(B,N)；True token 保留投影，False token 乘零。

        返回值：
            - projected: 浮点张量，(B,N,D_m)；与 mask 的 B/N 轴逐项对齐。
        """

        if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
            raise ValueError("tokens 必须为 (B,N,F)，mask 必须为对应 (B,N)。")
        if mask.dtype != torch.bool:
            raise TypeError("mask 必须是 bool。")
        # 浮点，(B,N,D_m)，补零 token 经带偏置线性层后可能非零，因此返回前必须再次乘 mask。
        projected = self.projection(tokens)
        return projected * mask[..., None].to(dtype=projected.dtype)


class CandidateCrossAttention(nn.Module):
    """以候选摘要查询一种模态。

    构造参数：
        - model_dim: 正整数 D_m；候选查询、token 和输出的共同宽度。
        - head_count: 正整数 K；注意力头数，必须整除 D_m。
        - dropout: 数值；归一化注意力权重的丢弃概率，本类直接构造时依赖 Dropout 检查范围。
        - zero_output: 布尔语义值；为真时把输出投影权重与偏置初始化为零，本类不要求运行时类型必须是精确 bool。

    前向输入：
        - candidate_query: 浮点张量，(B,D_m)；每个候选一个查询向量。
        - tokens: 浮点张量，(B,N,D_m)；同一种模态的投影 token。
        - mask: bool 张量，(B,N)；True token 参与 softmax，False token 在偏置相加后仍被排除。
        - attention_bias: 浮点张量或 None；非空时形状为 (B,K,N)，是在 softmax 前加入每个头、每个 token 内容分数的无量纲值。

    前向输出：
        - increment: 浮点张量，(B,D_m)；每个候选从该模态读取的增量，没有有效 token 的候选严格为零。

    训练模式下 Dropout 作用于已归一化的注意力权重，因此丢弃后不重新归一化。zero_output 只决定初始化状态，不在后续前向中持续强制输出为零。
    """

    def __init__(
        self,
        model_dim: int,
        head_count: int,
        dropout: float,
        zero_output: bool,
    ) -> None:
        """建立单查询多头注意力的查询、键、值和输出投影。

        输入参数：
            - model_dim: 正整数 D_m；必须能被 head_count 整除。
            - head_count: 正整数 K；每头宽度为 D_m/K。
            - dropout: 数值；注意力权重 Dropout 概率。
            - zero_output: 布尔语义值；为真时仅把 self.output 的初始参数清零。

        实例状态：
            - self.model_dim: 整数 D_m；前向形状门禁使用。
            - self.head_count: 整数 K；前向偏置形状和改排使用。
            - self.head_dim: 整数 D_m/K；内容分数除以其平方根。
            - self.query: Linear；把每个候选查询投影到 K 个头。
            - self.key: 无偏置 Linear；逐 token 产生 K 个键。
            - self.value: 无偏置 Linear；逐 token 产生 K 个值。
            - self.output: Linear；把 K 个头连接回 D_m，可能按 zero_output 零初始化。
            - self.dropout: Dropout；只作用于归一化权重。
        """

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
        """计算每个候选、每个头对一种模态 token 的注意力。

        输入参数：
            - candidate_query: 浮点张量，(B,D_m)；投影后改排为 (B,K,1,D_h)，D_h=D_m/K。
            - tokens: 浮点张量，(B,N,D_m)；键和值均改排为 (B,K,N,D_h)。
            - mask: bool 张量，(B,N)；广播成 (B,1,1,N)，同时遮盖所有头的补零 token。
            - attention_bias: 浮点张量或 None；非空时为 (B,K,N)，扩成 (B,K,1,N) 后加入内容分数。

        返回值：
            - increment: 浮点张量，(B,D_m)；沿 N 对值求加权和、连接 K 个头并经过输出投影，没有有效 token 的候选乘零。

        query 与 key 只沿 D_h 求点积，保留 B/K/单查询/N 轴；softmax 只沿 N 归一化。N=0 时直接返回与 candidate_query 同形的零张量。
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

        # 三个 int，当前批次大小 B、该模态补零宽度 N 和已核验的模型宽度 D_m。
        batch_size, token_count, _ = tokens.shape
        # 浮点，(B,K,1,D_h)，每个候选在每个头只有一个查询位置。
        query = self.query(candidate_query).reshape(
            batch_size, self.head_count, 1, self.head_dim
        )
        # 浮点，[B,N,D_m] -> [B,N,K,D_h] -> [B,K,N,D_h]，每个 token 的多头键。
        key = self.key(tokens).reshape(
            batch_size, token_count, self.head_count, self.head_dim
        ).transpose(1, 2)
        # 浮点，[B,N,D_m] -> [B,N,K,D_h] -> [B,K,N,D_h]，与 key 同 token 轴的多头值。
        value = self.value(tokens).reshape(
            batch_size, token_count, self.head_count, self.head_dim
        ).transpose(1, 2)
        # 浮点，(B,K,1,N)，沿 D_h 点积并按 sqrt(D_h) 缩放的内容分数。
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_bias is not None:
            score = score + attention_bias[:, :, None, :].to(dtype=score.dtype)
        # bool，(B,1,1,N)，同一 token 有效性向 K 个头广播。
        valid = mask[:, None, None, :]
        score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
        # 浮点，(B,K,1,N)，先沿 token 轴 softmax；全空候选随后乘零并用夹紧分母保持有限。
        weight = torch.softmax(score, dim=-1)
        weight = weight * valid.to(dtype=weight.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        # 浮点，(B,K,1,D_h)，沿 token 轴 N 对 value 求和；训练时使用丢弃后的权重。
        context = torch.matmul(self.dropout(weight), value)
        # 浮点，[B,K,1,D_h] -> [B,D_m]，连接头轴与每头通道轴。
        context = context.reshape(batch_size, self.model_dim)
        # 浮点，(B,D_m)，输出投影可含偏置，因此全空候选必须在末端再次乘零。
        increment = self.output(context)
        # bool，(B,1)，每个候选是否至少有一个有效 token。
        has_token = mask.any(dim=1, keepdim=True)
        return increment * has_token.to(dtype=increment.dtype)


class CandidateTokenAttentionBias(nn.Module):
    """把逐实体 Stage1 概率与相对坐标转换为逐注意力头分数偏置。

    构造参数：
        - model_dim: 正整数 D_m；四维输入投影后的隐藏宽度。
        - head_count: 正整数 K；每个 token 输出的注意力头数。

    前向输入：
        - features: 浮点张量，(B,N,4)；第二轴与一种模态 token 逐项对齐，最后一轴依次为无量纲相对 X/Y/Z 和 Stage1 概率。

    前向输出：
        - bias: 浮点张量，(B,K,N)；加入 CandidateCrossAttention softmax 前内容分数的逐头、逐 token 无量纲值。

    末层权重与偏置均零初始化，所以初始输出逐元素为零；训练后不承诺继续为零。末层 bias 对同一候选和注意力头的全部 token 产生相同平移，单独看数学上不改变 softmax；末层 weight 是让不同 token 获得不同偏置的有效学习路径。
    """

    def __init__(self, model_dim: int, head_count: int) -> None:
        """建立四维输入到逐头偏置的两层网络，并把末层清零。

        输入参数：
            - model_dim: 正整数 D_m；隐藏层宽度。
            - head_count: 正整数 K；末层输出宽度。

        实例状态：
            - self.layers: Sequential；依次执行 4 到 D_m 的线性层、SiLU 和 D_m 到 K 的零初始化线性层。
        """

        super().__init__()
        if model_dim <= 0 or head_count <= 0:
            raise ValueError("注意力偏置的模型宽度和头数必须为正整数。")
        # Linear，输入 D_m、输出 K；权重与偏置在加入 Sequential 前同时置零。
        final = nn.Linear(model_dim, head_count)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.layers = nn.Sequential(
            nn.Linear(CANDIDATE_TOKEN_ATTENTION_BIAS_FEATURE_DIM, model_dim),
            nn.SiLU(),
            final,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """把每个 token 的四维输入转换成逐注意力头偏置。

        输入参数：
            - features: 浮点张量，(B,N,4)；与当前模态 token 的 B/N 轴逐项对齐，本方法只检查形状，不检查数值范围。

        返回值：
            - bias: 浮点张量，[B,N,4] -> [B,N,K] -> [B,K,N]；最后一次转置使头轴与 CandidateCrossAttention 对齐。
        """

        if (
            features.ndim != 3
            or features.shape[2] != CANDIDATE_TOKEN_ATTENTION_BIAS_FEATURE_DIM
        ):
            raise ValueError("候选实体注意力偏置输入必须为 (B,N,4)。")
        return self.layers(features).transpose(1, 2)


class MeanMaxAggregator(nn.Module):
    """对四模态取掩码均值/最大值，并按配置拼接模态统计与采样统计。

    构造参数：
        - model_dim: 整数 D_m；候选摘要、四种模态 token 与输出的共同宽度，本类不单独检查正值。
        - dropout: 数值；融合 MLP 的丢弃概率，直接构造时依赖 Dropout 检查。
        - include_manual: 布尔语义值；为真时拼接 12 项 A/P/V/PP 固化统计，本类不要求运行时类型必须是精确 bool。
        - include_sampling: 布尔语义值；为真时拼接四项裁剪几何统计，本类不要求运行时类型必须是精确 bool。

    前向输入：
        - candidate_summary: 浮点张量，(B,D_m)；RAUNet 特征的空间均值/最大值投影摘要。
        - modalities: 四键映射；键必须至少含 a、p、v、pp，额外键不读取。
        - modalities.a: 二元组；浮点 (B,N_A,D_m) token 与 bool (B,N_A) 有效掩码。
        - modalities.p: 二元组；浮点 (B,N_P,D_m) token 与 bool (B,N_P) 有效掩码。
        - modalities.v: 二元组；浮点 (B,N_V,D_m) token 与 bool (B,N_V) 有效掩码。
        - modalities.pp: 二元组；浮点 (B,N_PP,D_m) token 与 bool (B,N_PP) 有效掩码。
        - manual_features: 浮点张量或 None；include_manual 为真时必须是与候选轴对齐的 (B,12)，否则必须为 None。
        - sampling_features: 浮点张量或 None；include_sampling 为真时必须是与候选轴对齐的 (B,4)，否则必须为 None。
        - attention_bias_features: 必须为 None；mean_max 不建立或消费 token 注意力偏置。

    前向输出：
        - candidate_embedding: 浮点张量，(B,D_m)；候选摘要、可选采样统计和 A/P/V/PP 各自有效 token 均值/最大值及可选三项手工统计的融合表示。

    某种模态没有有效 token 时，该模态的均值和最大值都是零；对应手工统计若启用，仍按输入值拼接。
    """

    def __init__(
        self,
        model_dim: int,
        dropout: float,
        include_manual: bool,
        include_sampling: bool,
    ) -> None:
        """建立固定输入宽度的 mean/max 融合 MLP。

        输入参数：
            - model_dim: 整数 D_m；基础输入含一个候选摘要和四种模态各两个 D_m 摘要，共 9D_m。
            - dropout: 数值；第一隐藏层 GELU 后的丢弃概率。
            - include_manual: 布尔语义值；为真时在 9D_m 后增加 12 维。
            - include_sampling: 布尔语义值；为真时在候选摘要后增加 4 维。

        实例状态：
            - self.include_manual: 布尔语义值；前向决定 manual_features 的存在性与三元组拼接。
            - self.include_sampling: 布尔语义值；前向决定 sampling_features 的存在性与拼接。
            - self.fusion: Sequential；把固定融合宽度投影为 D_m，并在输出使用 LayerNorm。
        """

        super().__init__()
        self.include_manual = include_manual
        self.include_sampling = include_sampling
        # int，拼接向量最后一维；基础为候选摘要 D_m 加四种模态的 mean/max 共 8D_m，再加配置启用的 12 维与 4 维统计。
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
        """按固定字段顺序连接候选摘要、统计和四模态归约结果。

        输入参数：
            - candidate_summary: 浮点张量，(B,D_m)；拼接向量第一段。
            - modalities: 四键映射；a、p、v、pp 的 token 数 N_m 可以彼此不同，字段契约见类说明。
            - manual_features: 浮点张量或 None；启用时为 (B,12)，按四种模态改排成 (B,4,3)。
            - sampling_features: 浮点张量或 None；启用时为 (B,4)，紧跟 candidate_summary 拼接。
            - attention_bias_features: None；非空立即失败。

        返回值：
            - candidate_embedding: 浮点张量，(B,D_m)；self.fusion 的输出。

        拼接顺序固定为 candidate_summary、可选 sampling、A mean、A max、可选 A 三项手工统计、P 对应三段、V 对应三段、PP 对应三段。额外 modalities 键被忽略，缺少任何固定键时由映射索引失败。
        """

        if attention_bias_features is not None:
            raise ValueError("mean_max 聚合不得接收候选实体注意力偏置输入。")
        # list[Tensor]，待沿最后一维拼接的 (B,F_part) 张量；首项固定为候选体素摘要。
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
        # None 或浮点 (B,4,3)；第二轴按 A/P/V/PP，第三轴为计数、概率均值、概率最大值。
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
            # 浮点 (B,N_m,D_m) 与 bool (B,N_m)，当前固定模态的投影 token 和有效掩码。
            tokens, mask = modalities[name]
            # 两个浮点 (B,D_m)，沿 N_m 归约的有效 token 均值和逐通道最大值。
            mean, maximum = _masked_mean_max(tokens, mask)
            summaries.extend((mean, maximum))
            if grouped_manual is not None:
                summaries.append(grouped_manual[:, modality_index, :])
        # torch.cat 产生浮点 (B,fusion_width)，按 Docstring 固定顺序连接后立即进入融合 MLP。
        return self.fusion(torch.cat(summaries, dim=-1))


class CandidateConditionedAggregator(nn.Module):
    """用候选摘要分别查询四模态，并按配置拼接模态统计与采样统计。

    构造参数：
        - model_dim: 整数 D_m；候选查询、四模态 token 和输出的共同宽度。
        - head_count: 正整数 K；每个独立模态分支的注意力头数，必须整除 D_m。
        - dropout: 数值；注意力权重和融合 MLP 的丢弃概率。
        - include_manual: 布尔语义值；为真时拼接 12 项模态手工统计。
        - include_sampling: 布尔语义值；为真时拼接四项候选裁剪统计。
        - use_token_attention_bias: 布尔语义值；为真时为 A/P/V/PP 建立四套互不共享参数的偏置 MLP。

    前向输入：
        - candidate_summary: 浮点张量，(B,D_m)；四个模态分支共用的候选查询。
        - modalities: 四键映射；键必须至少含 a、p、v、pp，额外键不读取。
        - modalities.a: 二元组；浮点 (B,N_A,D_m) token 与 bool (B,N_A) 有效掩码。
        - modalities.p: 二元组；浮点 (B,N_P,D_m) token 与 bool (B,N_P) 有效掩码。
        - modalities.v: 二元组；浮点 (B,N_V,D_m) token 与 bool (B,N_V) 有效掩码。
        - modalities.pp: 二元组；浮点 (B,N_PP,D_m) token 与 bool (B,N_PP) 有效掩码。
        - manual_features: 浮点张量或 None；include_manual 为真时必须是 (B,12)，否则必须为 None。
        - sampling_features: 浮点张量或 None；include_sampling 为真时必须是 (B,4)，否则必须为 None。
        - attention_bias_features: 四键映射或 None；存在性必须与 use_token_attention_bias 一致。
        - attention_bias_features.a: 浮点张量，(B,N_A,4)；与 A token 对齐，最后一轴为相对 X/Y/Z 和 Stage1 概率。
        - attention_bias_features.p: 浮点张量，(B,N_P,4)；与 P token 对齐，最后一轴顺序同上。
        - attention_bias_features.v: 浮点张量，(B,N_V,4)；与 V token 对齐，最后一轴顺序同上。
        - attention_bias_features.pp: 浮点张量，(B,N_PP,4)；与 PP token 对齐，最后一轴顺序同上。

    前向输出：
        - candidate_embedding: 浮点张量，(B,D_m)；候选摘要、四个模态注意力增量和配置启用统计的融合表示。

    每种模态没有有效 token 时对应注意力增量为零。PP 分支输出投影固定零初始化，所以初始 PP 增量为零；A/P/V 分支沿用常规初始化。PP 的零输出投影还会阻断第一次反向到达注意力偏置末层权重，输出投影更新后该路径才可唤醒；A/P/V 的非零输出投影允许偏置末层权重在第一次反向获得梯度。
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
        """建立四个候选查询分支、可选偏置网络和最终融合 MLP。

        输入参数：
            - model_dim: 整数 D_m；候选查询、模态 token、分支增量和最终输出宽度。
            - head_count: 正整数 K；传给每个 CandidateCrossAttention 分支。
            - dropout: 数值；传给注意力权重和融合 MLP。
            - include_manual: 布尔语义值；决定融合宽度是否增加 12。
            - include_sampling: 布尔语义值；决定融合宽度是否增加 4。
            - use_token_attention_bias: 布尔语义值；决定 self.attention_biases 是四键 ModuleDict 还是 None。

        实例状态：
            - self.include_manual: 布尔语义值；控制 manual_features 存在性和拼接。
            - self.include_sampling: 布尔语义值；控制 sampling_features 存在性和拼接。
            - self.branches: ModuleDict；a、p、v 使用常规输出投影初始化，pp 的输出投影权重与偏置初始化为零。
            - self.fusion: Sequential；把 5D_m 加可选统计的拼接向量映射到 D_m。
            - self.attention_biases: ModuleDict 或 None；启用时含 a、p、v、pp 四套独立 CandidateTokenAttentionBias。

        偏置模块在 torch.random.fork_rng(devices=[]) 中构造，使 CPU 随机状态在退出后恢复；这只隔离新增偏置模块的初始化随机消耗，不重置已经建立的共同分支参数。
        """

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
        # int，拼接向量最后一维；基础为候选摘要 D_m 加四个模态增量 4D_m，再加配置启用的 12 维与 4 维统计。
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
            # 新增四套偏置分支只消耗临时 CPU 随机流，退出上下文后恢复构造前状态，避免改变后续模块可见的 CPU RNG。
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
        """以候选摘要查询四种模态，再按固定顺序连接增量与统计。

        输入参数：
            - candidate_summary: 浮点张量，(B,D_m)；四个 CandidateCrossAttention 共用的查询，也是拼接向量第一段。
            - modalities: 四键映射；a、p、v、pp 的独立 N_m、token 与 mask 契约见类说明。
            - manual_features: 浮点张量或 None；启用时为 (B,12)，按 A/P/V/PP 改排成 (B,4,3)。
            - sampling_features: 浮点张量或 None；启用时为 (B,4)，紧跟 candidate_summary 拼接。
            - attention_bias_features: 四键映射或 None；启用偏置时键集合必须恰好是 a、p、v、pp，每个值的 B/N_m 轴必须与同模态 token 对齐。

        返回值：
            - candidate_embedding: 浮点张量，(B,D_m)；self.fusion 的输出。

        拼接顺序固定为 candidate_summary、可选 sampling、A 注意力增量、可选 A 三项手工统计、P 对应两段、V 对应两段、PP 对应两段。modalities 的额外键被忽略，缺少固定键时由映射索引失败。
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
        # None 或浮点 (B,4,3)；第二轴按 A/P/V/PP，第三轴为计数、概率均值、概率最大值。
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
        # list[Tensor]，待沿最后一维拼接的 (B,F_part) 张量；首项固定为候选摘要。
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
            # 浮点 (B,N_m,D_m) 与 bool (B,N_m)，当前固定模态的投影 token 和有效掩码。
            tokens, mask = modalities[name]
            # None 或浮点 (B,K,N_m)；非空时是当前模态逐头、逐 token 的 softmax 前偏置。
            attention_bias = None
            if self.attention_biases is not None:
                assert attention_bias_features is not None
                # 浮点，(B,N_m,4)，与当前模态 token 的 B/N_m 轴对齐的相对 X/Y/Z 和 Stage1 概率。
                features = attention_bias_features[name]
                if features.shape[:2] != tokens.shape[:2]:
                    raise ValueError(f"{name.upper()} 偏置输入未与对应 token 的 B/N 轴对齐。")
                attention_bias = self.attention_biases[name](features)
            # 分支返回浮点 (B,D_m) 当前模态注意力增量；全空模态严格为零，并按固定顺序直接追加。
            summaries.append(
                self.branches[name](candidate_summary, tokens, mask, attention_bias)
            )
            if grouped_manual is not None:
                summaries.append(grouped_manual[:, modality_index, :])
        # torch.cat 产生浮点 (B,fusion_width)，按 Docstring 固定顺序连接后立即进入融合 MLP。
        return self.fusion(torch.cat(summaries, dim=-1))


class MeanMaxCandidateConditionedResidualAggregator(nn.Module):
    """把两种完整候选摘要拼接为零输出残差，并保留 mean/max 主路径。

    - model_dim: 候选摘要宽度 D_m。
    - head_count: 候选条件注意力的头数 K。
    - dropout: 两个既有聚合器和残差多层感知机的丢弃概率。
    - include_manual: 是否让两个既有聚合器分别使用 12 项手动统计。
    - include_sampling: 是否让两个既有聚合器分别使用 4 项采样统计。
    - candidate_summary: 浮点 (B,D_m) U-Net 候选查询。
    - modalities: 含 a、p、v、pp 四个键；每个值是投影 token 与有效性掩码组成的二元组。
    - modalities[a]: 浮点 (B,N_A,D_m) 原子 token 与 bool (B,N_A) 掩码。
    - modalities[p]: 浮点 (B,N_P,D_m) P 模态 token 与 bool (B,N_P) 掩码。
    - modalities[v]: 浮点 (B,N_V,D_m) 体素 token 与 bool (B,N_V) 掩码。
    - modalities[pp]: 浮点 (B,N_PP,D_m) 概率图 token 与 bool (B,N_PP) 掩码。
    - manual_features: 与 include_manual 一致的可选浮点 (B,12) 统计。
    - sampling_features: 与 include_sampling 一致的可选浮点 (B,4) 统计。
    - attention_bias_features: 必须为 None；本结构不启用 token 注意力偏置。
    - self.mean_max: MeanMaxAggregator；生成保留为主路径的 D_m 维 mean/max 摘要。
    - self.conditioned: CandidateConditionedAggregator；独立生成 D_m 维条件注意力摘要，且关闭 token 注意力偏置。
    - self.residual: Sequential；把两个 D_m 维摘要拼成的 2D_m 维向量映射为 D_m 维学习残差，末层权重与偏置初始化为零。
    - 返回值: 浮点 (B,D_m) 候选表示。残差末层为零时逐值等于 mean/max 输出。

    mean/max 与条件注意力分支分别读取同一手动统计与采样统计。两个 D_m 维输出只在残差入口拼接一次，不会第三次追加原始统计。
    """

    def __init__(
        self,
        model_dim: int,
        head_count: int,
        dropout: float,
        include_manual: bool,
        include_sampling: bool,
    ) -> None:
        """建立两个完整摘要分支和零输出残差分支。

        - model_dim: 正整数 D_m；候选摘要、两个分支输出和最终残差的共同宽度。
        - head_count: 正整数 K；条件注意力分支的注意力头数，必须整除 D_m。
        - dropout: 范围为 [0,1) 的浮点数；传给两个完整聚合器和残差多层感知机。
        - include_manual: bool；是否让两个完整聚合器分别读取同一份 12 项手动统计。
        - include_sampling: bool；是否让两个完整聚合器分别读取同一份 4 项采样统计。
        - self.mean_max: MeanMaxAggregator；产生保留为主路径的浮点 (B,D_m) 摘要。
        - self.conditioned: CandidateConditionedAggregator；产生浮点 (B,D_m) 条件注意力摘要，并固定关闭 token 注意力偏置。
        - self.residual: Sequential；把固定顺序的浮点 (B,2D_m) 拼接摘要映射成浮点 (B,D_m) 残差，末层权重和偏置均初始化为零。
        - 返回值: None；子模块写入当前实例。条件注意力与残差分支在域分离的 CPU 随机数上下文中初始化，退出上下文后恢复外层 CPU 随机状态。
        """

        super().__init__()
        self.mean_max = MeanMaxAggregator(
            model_dim,
            dropout,
            include_manual,
            include_sampling,
        )
        # 新分支使用确定性的域分离 CPU 子流，且不推进共同模块的初始化随机状态。
        with torch.random.fork_rng(devices=[]):
            # CPU 随机数生成器；只为条件注意力分支与残差分支提供域分离的初始化状态。
            initialization_generator = torch.Generator(device="cpu")
            initialization_seed = (torch.initial_seed() & ((1 << 63) - 1)) ^ 0x4D4D4341434F4E43
            initialization_generator.manual_seed(initialization_seed)
            torch.set_rng_state(initialization_generator.get_state())
            self.conditioned = CandidateConditionedAggregator(
                model_dim,
                head_count,
                dropout,
                include_manual,
                include_sampling,
                False,
            )
            # Linear(2D_m,D_m)；作为残差多层感知机的零初始化末层。
            final = nn.Linear(model_dim * 2, model_dim)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            self.residual = nn.Sequential(
                nn.LayerNorm(model_dim * 2),
                nn.Linear(model_dim * 2, model_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                final,
            )

    def forward(
        self,
        candidate_summary: torch.Tensor,
        modalities: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
        manual_features: torch.Tensor | None,
        sampling_features: torch.Tensor | None,
        attention_bias_features: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """计算两个完整摘要，并把零初始化残差加到 mean/max 表示。

        - candidate_summary: 浮点 (B,D_m) U-Net 候选查询。
        - modalities: 含 a、p、v、pp 四个键；每个值是投影 token 与有效性掩码组成的二元组。
        - modalities[a]: 浮点 (B,N_A,D_m) 原子 token 与 bool (B,N_A) 掩码。
        - modalities[p]: 浮点 (B,N_P,D_m) P 模态 token 与 bool (B,N_P) 掩码。
        - modalities[v]: 浮点 (B,N_V,D_m) 体素 token 与 bool (B,N_V) 掩码。
        - modalities[pp]: 浮点 (B,N_PP,D_m) 概率图 token 与 bool (B,N_PP) 掩码。
        - manual_features: 可选浮点 (B,12) 手动统计。
        - sampling_features: 可选浮点 (B,4) 采样统计。
        - attention_bias_features: 必须为 None。
        - 返回值: 浮点 (B,D_m) mean/max 主表示与学习残差之和。
        """

        if attention_bias_features is not None:
            raise ValueError("摘要拼接结构不得接收候选实体注意力偏置输入。")
        # 浮点 (B,D_m)；保留为最终输出主路径的 mean/max 候选摘要。
        mean_max_summary = self.mean_max(
            candidate_summary,
            modalities,
            manual_features,
            sampling_features,
            None,
        )
        # list[int]；需要保存并恢复随机状态的 CUDA 设备编号，CPU 候选使用空列表。
        fork_devices: list[int] = []
        if candidate_summary.device.type == "cuda":
            device_index = candidate_summary.device.index
            fork_devices.append(torch.cuda.current_device() if device_index is None else device_index)
        # 新分支仍执行训练态 dropout，但其随机消耗不得改变后续共同分类头和辅助头的掩码。
        with torch.random.fork_rng(devices=fork_devices):
            # 浮点 (B,D_m)；由四模态条件注意力、手动统计与采样统计共同形成的摘要。
            conditioned_summary = self.conditioned(
                candidate_summary,
                modalities,
                manual_features,
                sampling_features,
                None,
            )
            # 浮点 (B,2D_m)；按 mean/max、条件注意力的固定顺序连接两个完整摘要。
            concatenated = torch.cat((mean_max_summary, conditioned_summary), dim=-1)
            # 浮点 (B,D_m)；零初始化残差分支对拼接摘要给出的增量。
            residual_increment = self.residual(concatenated)
        return mean_max_summary + residual_increment
