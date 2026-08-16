"""把一个物理批次的候选体素与 A/P/V/PP token 编码成候选级预测。

主要入口 CandidateEncoder.forward 先由 RaUnetBackbone 产生同分辨率体素特征，
再投影四种变长 token；V 与 PP token 额外读取各自 Z/Y/X 位置的体素特征。
MeanMaxAggregator、CandidateConditionedAggregator 或
MeanMaxCandidateConditionedResidualAggregator 把体素摘要、四模态和已启用的候选统计
融合成一个候选向量。六分类头和两个体素二分类头分别输出类别分数、
配体区域分数与结合区域分数。本模块不读取语言向量、身份字符串或 LigandObject 图。
"""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn

from ..data.schema import CLASS_NAMES
from ..datasets.contracts import CandidateBatch
from .config import MatcherModelConfig
from .fusion import (
    CandidateConditionedAggregator,
    MeanMaxAggregator,
    MeanMaxCandidateConditionedResidualAggregator,
    MODALITY_INPUT_DIMS,
    TokenProjector,
    candidate_token_attention_bias_features,
    gather_voxel_features,
    normalize_candidate_manual_features,
    normalize_candidate_sampling_features,
)
from .raunet import RaUnetBackbone


class CandidateEncoderOutput(TypedDict):
    """声明候选编码器返回的五个张量。

    形状符号：
        - B: 当前物理批次的候选数。
        - D_m: MatcherModelConfig.model_dim，候选与身份共同表示宽度。
        - C_v: MatcherModelConfig.voxel_feature_channels，RAUNet 输出通道数。

    字段：
        - candidate_embedding: 浮点张量，(B,D_m)；融合候选体素摘要、四模态 token 与配置启用统计后的候选表示，供六分类头与完整 Matcher 的身份打分器共同读取。
        - class_logits: 浮点张量，(B,6)；六分类未归一化分数，第二轴顺序为 background、ion、nucleotide_like、peptide_like、small_molecule、sugar。
        - ligand_area_logits: 浮点张量，(B,48,48,48)；当前候选裁剪内逐体素配体区域未归一化分数，空间轴依次为 Z/Y/X。
        - binding_area_logits: 浮点张量，(B,48,48,48)；当前候选裁剪内逐体素受体结合区域未归一化分数，空间轴依次为 Z/Y/X。
        - voxel_features: 浮点张量，(B,C_v,48,48,48)；RAUNet 最高分辨率特征，V/PP token 按其局部 Z/Y/X 下标从该张量采样。
    """

    candidate_embedding: torch.Tensor
    class_logits: torch.Tensor
    ligand_area_logits: torch.Tensor
    binding_area_logits: torch.Tensor
    voxel_features: torch.Tensor


class VoxelBinaryHead(nn.Module):
    """用两个逐体素一乘一乘一卷积产生单通道区域分数。

    构造参数：
        - channels: 整数 C_v；输入、隐藏通道数，调用位置使用 RAUNet 输出宽度，本类不单独检查正值。

    前向输入：
        - features: 浮点张量，(B,C_v,Z,Y,X)；任意三维空间大小的体素特征，通道数必须与构造参数一致。

    前向输出：
        - logits: 浮点张量，(B,Z,Y,X)；去掉卷积输出的单通道轴后，每个体素一个未归一化二分类分数。
    """

    def __init__(self, channels: int) -> None:
        """建立保持空间分辨率的两层体素分类头。

        输入参数：
            - channels: 整数 C_v；两个卷积的输入或隐藏通道数，本方法依赖 Conv3d 检查不能构造的宽度。

        实例状态：
            - self.layers: Sequential；依次执行 C_v 到 C_v 的一乘一乘一卷积、ReLU 和 C_v 到 1 的一乘一乘一卷积。
        """

        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(channels, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """返回去掉单通道轴的逐体素分数。

        输入参数：
            - features: 浮点张量，(B,C_v,Z,Y,X)；通道宽度与构造时 channels 一致，本方法不另做显式形状门禁。

        返回值：
            - logits: 浮点张量，(B,Z,Y,X)；self.layers 输出 (B,1,Z,Y,X) 的通道零切片。
        """

        return self.layers(features)[:, 0]


class CandidateEncoder(nn.Module):
    """组合体素主干、四模态聚合器、六分类头和两个区域头。

    构造参数：
        - config: MatcherModelConfig；提供 RAUNet、token 投影、聚合器、候选统计开关和可选实体注意力偏置的结构值。

    实例状态：
        - self.config: MatcherModelConfig；保留原配置供前向判断两个统计开关和注意力偏置开关。
        - self.voxel_backbone: RaUnetBackbone；把固定 105 通道 48 立方体编码为 C_v 通道同分辨率特征。
        - self.candidate_summary: Sequential；把体素特征的逐通道空间均值与最大值拼接向量从 2C_v 投影到 D_m。
        - self.token_projectors: ModuleDict；a、p、v、pp 四键分别把正式原始宽度投影到 D_m，并把补零 token 清零。
        - self.voxel_modulators: ModuleDict；v、pp 两键分别把采样的 C_v 维体素特征投影到 D_m，再加到同位置 token 表示。
        - self.aggregator: MeanMaxAggregator、CandidateConditionedAggregator 或 MeanMaxCandidateConditionedResidualAggregator；由 config.aggregation 唯一选择。
        - self.class_head: Sequential；把 (B,D_m) 候选表示映射成与 CLASS_NAMES 顺序一致的六分类 logits。
        - self.ligand_area_head: VoxelBinaryHead；从同一 RAUNet 特征产生配体区域 logits。
        - self.binding_area_head: VoxelBinaryHead；从同一 RAUNet 特征产生受体结合区域 logits。

    本类不读取 CandidateBatch 中的语言向量、身份字符串、身份类别、图字段或身份真值。
    """

    def __init__(self, config: MatcherModelConfig) -> None:
        """按冻结配置建立候选编码路径。

        输入参数：
            - config: MatcherModelConfig；通常已由 dataclass 门禁验证，本构造器不重复检查字段类型、范围或跨字段关系。

        聚合分支：
            - config.aggregation=mean_max: 建立掩码均值/最大值聚合器，且不建立 token 注意力偏置网络。
            - config.aggregation=candidate_cross_attention: 建立四个候选查询分支，并按 use_candidate_token_attention_bias 决定是否建立四套偏置网络。
            - config.aggregation=mean_max_candidate_cross_attention_concat: 分别建立完整 mean/max 与候选条件注意力聚合器，把两个 D_m 维摘要拼接后经零输出残差加回 mean/max 主路径；该分支不建立 token 注意力偏置网络。

        返回值：
            - return: None；全部子模块写入 self，参数初始化沿用 PyTorch 和子模块定义。
        """

        super().__init__()
        self.config = config
        self.voxel_backbone = RaUnetBackbone(
            input_channels=105,
            feature_channels=config.voxel_feature_channels,
            planes=config.raunet_planes,
            recycle_iterations=config.recycle_iterations,
            gradient_checkpointing=config.gradient_checkpointing,
            bottleneck_attention_layers=config.bottleneck_attention_layers,
        )
        self.candidate_summary = nn.Sequential(
            nn.LayerNorm(config.voxel_feature_channels * 2),
            nn.Linear(config.voxel_feature_channels * 2, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
        )
        self.token_projectors = nn.ModuleDict(
            {
                name: TokenProjector(input_dim, config.model_dim, config.dropout)
                for name, input_dim in MODALITY_INPUT_DIMS.items()
            }
        )
        self.voxel_modulators = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(config.voxel_feature_channels, config.model_dim),
                    nn.GELU(),
                    nn.Linear(config.model_dim, config.model_dim),
                )
                for name in ("v", "pp")
            }
        )
        if config.aggregation == "mean_max":
            self.aggregator: nn.Module = MeanMaxAggregator(
                config.model_dim,
                config.dropout,
                config.use_manual_features,
                config.use_sampling_features,
            )
        elif config.aggregation == "candidate_cross_attention":
            self.aggregator = CandidateConditionedAggregator(
                config.model_dim,
                config.attention_heads,
                config.dropout,
                config.use_manual_features,
                config.use_sampling_features,
                config.use_candidate_token_attention_bias,
            )
        elif config.aggregation == "mean_max_candidate_cross_attention_concat":
            self.aggregator = MeanMaxCandidateConditionedResidualAggregator(
                config.model_dim,
                config.attention_heads,
                config.dropout,
                config.use_manual_features,
                config.use_sampling_features,
            )
        else:
            raise RuntimeError(f"未实现 aggregation={config.aggregation!r}。")
        self.class_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim, len(CLASS_NAMES)),
        )
        self.ligand_area_head = VoxelBinaryHead(config.voxel_feature_channels)
        self.binding_area_head = VoxelBinaryHead(config.voxel_feature_channels)

    def forward(self, batch: CandidateBatch) -> CandidateEncoderOutput:
        """编码 CandidateBatch 中的候选体素、四模态 token 和已启用统计。

        形状符号：
            - B: 当前物理批次的候选数。
            - N_A: 批内补零后的最大 A token 数。
            - N_P: 批内补零后的最大 P token 数。
            - N_V: 批内补零后的最大 V token 数。
            - N_PP: 批内固定或补零后的 PP token 数。
            - D_m: config.model_dim。
            - C_v: config.voxel_feature_channels。

        输入字段：
            - batch: CandidateBatch；本函数只消费下列候选体素、四模态 token、位置索引和候选统计字段。
            - batch.voxel_input: 浮点张量，(B,105,48,48,48)；候选 48 立方体体素输入，空间轴依次为 Z/Y/X，进入主干前转换为 input_projection.weight 的 dtype。
            - batch.a_tokens: 浮点张量，(B,N_A,229)；完整 80 立方体 A token，末四个特征分量为 Stage1 概率与相对 X/Y/Z。
            - batch.a_mask: bool 张量，(B,N_A)；True 表示对应 A token 有效，False 表示补零。
            - batch.p_tokens: 浮点张量，(B,N_P,116)；完整 80 立方体 P token，末四个特征分量为 Stage1 概率与相对 X/Y/Z。
            - batch.p_mask: bool 张量，(B,N_P)；True 表示对应 P token 有效，False 表示补零。
            - batch.v_tokens: 浮点张量，(B,N_V,52)；当前 48 立方体内 V token，末四个特征分量为 Stage1 概率与相对 X/Y/Z。
            - batch.v_mask: bool 张量，(B,N_V)；True 表示对应 V token 与 v_index_zyx 有效，False 表示补零。
            - batch.v_index_zyx: int64 张量，(B,N_V,3)；每个有效 V token 在 voxel_features 空间轴中的 Z/Y/X 下标。
            - batch.pp_tokens: 浮点张量，(B,N_PP,4)；四个特征分量依次为 Stage1 概率与相对 X/Y/Z。
            - batch.pp_mask: bool 张量，(B,N_PP)；True 表示对应 PP token 与 pp_index_zyx 有效，False 表示补零。
            - batch.pp_index_zyx: int64 张量，(B,N_PP,3)；每个有效 PP token 在 voxel_features 空间轴中的 Z/Y/X 下标。
            - batch.candidate_manual_features: float32 张量，(B,12)；仅 use_manual_features 为真时读取，按 A/P/V/PP 顺序各含 token 数、Stage1 概率均值和最大值。
            - batch.candidate_sampling_features: float32 张量，(B,4)；仅 use_sampling_features 为真时读取，依次为边界受限标志、中心偏移 L2、中心偏移 L∞ 和 V 保留比例。

        返回字段：
            - output.candidate_embedding: 浮点张量，(B,D_m)；配置选择的聚合器输出。
            - output.class_logits: 浮点张量，(B,6)；CLASS_NAMES 顺序的六分类未归一化分数。
            - output.ligand_area_logits: 浮点张量，(B,48,48,48)；逐体素配体区域未归一化分数。
            - output.binding_area_logits: 浮点张量，(B,48,48,48)；逐体素受体结合区域未归一化分数。
            - output.voxel_features: 浮点张量，(B,C_v,48,48,48)；供区域头及 V/PP 采样共同使用的 RAUNet 特征。

        use_candidate_token_attention_bias 为真时，四种原始 token 的末四个特征分量从“概率、相对 X/Y/Z”重排成“相对 X/Y/Z、概率”；每个偏置输入仍与对应 token 和 mask 的 B/N_m 轴逐项对齐。
        """

        # 磁盘缓存是 float16；显式对齐主干参数 dtype 以兼容 CPU 测试，GPU autocast 仍可在算子内选择 BF16/FP16。
        voxel_input = batch["voxel_input"].to(
            dtype=self.voxel_backbone.input_projection.weight.dtype
        )
        # 浮点，(B,C_v,48,48,48)，最高分辨率 RAUNet 特征；dtype 由参数、设备和自动混合精度上下文共同决定。
        voxel_features = self.voxel_backbone(voxel_input)
        # 浮点，(B,2C_v)，先列逐通道空间均值，再列同通道空间最大值；三个空间轴 48 立方体全部归约。
        pooled = torch.cat(
            (voxel_features.mean(dim=(2, 3, 4)), voxel_features.amax(dim=(2, 3, 4))),
            dim=-1,
        )
        # 浮点，(B,D_m)，只含体素主干全局信息，随后作为聚合基底或四个交叉注意力分支的查询。
        candidate_summary = self.candidate_summary(pooled)

        # 浮点，(B,N_A,D_m)，A token 投影；a_mask=False 的补零位置严格为零。
        a_projected = self.token_projectors["a"](batch["a_tokens"], batch["a_mask"])
        # 浮点，(B,N_P,D_m)，P token 投影；p_mask=False 的补零位置严格为零。
        p_projected = self.token_projectors["p"](batch["p_tokens"], batch["p_mask"])
        # 浮点，(B,N_V,D_m)，尚未加入同位置 RAUNet 特征的 V token 投影。
        v_projected = self.token_projectors["v"](batch["v_tokens"], batch["v_mask"])
        # 浮点，(B,N_V,C_v)，按 v_index_zyx 从 voxel_features 读取；无效 V 位置严格为零。
        v_sampled = gather_voxel_features(voxel_features, batch["v_index_zyx"], batch["v_mask"])
        # 浮点，(B,N_V,D_m)，V 原始 token 投影与同位置体素调制向量之和。
        v_projected = v_projected + self.voxel_modulators["v"](v_sampled)
        # 调制 MLP 含偏置，补零位置的零采样也可能变为非零；再次乘 mask 才恢复严格零。
        v_projected = v_projected * batch["v_mask"][..., None].to(dtype=v_projected.dtype)
        # 浮点，(B,N_PP,D_m)，尚未加入同位置 RAUNet 特征的 PP token 投影。
        pp_projected = self.token_projectors["pp"](batch["pp_tokens"], batch["pp_mask"])
        # 浮点，(B,N_PP,C_v)，按 pp_index_zyx 从 voxel_features 读取；无效 PP 位置严格为零。
        pp_sampled = gather_voxel_features(voxel_features, batch["pp_index_zyx"], batch["pp_mask"])
        # 浮点，(B,N_PP,D_m)，PP 原始 token 投影与同位置体素调制向量之和。
        pp_projected = pp_projected + self.voxel_modulators["pp"](pp_sampled)
        # 调制 MLP 含偏置，补零位置的零采样也可能变为非零；再次乘 mask 才恢复严格零。
        pp_projected = pp_projected * batch["pp_mask"][..., None].to(dtype=pp_projected.dtype)
        # 四键映射，每个值是 (B,N_m,D_m) 投影 token 与同 B/N_m 轴 bool 有效掩码的二元组。
        modalities = {
            # a: A token 投影与 a_mask；A 不从 voxel_features 采样。
            "a": (a_projected, batch["a_mask"]),
            # p: P token 投影与 p_mask；P 不从 voxel_features 采样。
            "p": (p_projected, batch["p_mask"]),
            # v: 已加入同位置体素调制的 V token 与 v_mask。
            "v": (v_projected, batch["v_mask"]),
            # pp: 已加入同位置体素调制的 PP token 与 pp_mask。
            "pp": (pp_projected, batch["pp_mask"]),
        }
        # None 或四键映射；None 表示不向候选条件注意力加入相对坐标/概率偏置。
        attention_bias_features = None
        if self.config.use_candidate_token_attention_bias:
            # 四键映射，每个值是 (B,N_m,4) 的相对 X/Y/Z 与 Stage1 概率，并与同模态 token 逐项对齐。
            attention_bias_features = {
                # a: 从 a_tokens 末四个特征分量重排得到。
                "a": candidate_token_attention_bias_features(batch["a_tokens"]),
                # p: 从 p_tokens 末四个特征分量重排得到。
                "p": candidate_token_attention_bias_features(batch["p_tokens"]),
                # v: 从 v_tokens 末四个特征分量重排得到。
                "v": candidate_token_attention_bias_features(batch["v_tokens"]),
                # pp: 从 pp_tokens 全部四个特征分量重排得到。
                "pp": candidate_token_attention_bias_features(batch["pp_tokens"]),
            }

        # None 或浮点 (B,12)；四组三元组中仅 token 数执行 log1p，两个概率统计保持原值。
        normalized_manual = None
        if self.config.use_manual_features:
            normalized_manual = normalize_candidate_manual_features(
                batch["candidate_manual_features"]
            )
        # None 或浮点 (B,4)；L2 距离除以 24*sqrt(3)，Linf 距离除以 24，边界标志和保留率保持原值。
        normalized_sampling = None
        if self.config.use_sampling_features:
            normalized_sampling = normalize_candidate_sampling_features(
                batch["candidate_sampling_features"]
            )
        # 浮点，(B,D_m)，候选体素摘要、四模态和配置启用统计的最终融合表示。
        candidate_embedding = self.aggregator(
            candidate_summary,
            modalities,
            normalized_manual,
            normalized_sampling,
            attention_bias_features,
        )
        return {
            # (B,D_m)，同时进入六分类头和完整 Matcher 的三个身份配对打分器。
            "candidate_embedding": candidate_embedding,
            # (B,6)，CLASS_NAMES 固定顺序的未归一化类别分数。
            "class_logits": self.class_head(candidate_embedding),
            # (B,48,48,48)，配体区域未归一化体素分数。
            "ligand_area_logits": self.ligand_area_head(voxel_features),
            # (B,48,48,48)，受体结合区域未归一化体素分数。
            "binding_area_logits": self.binding_area_head(voxel_features),
            # (B,C_v,48,48,48)，与本次三个候选预测共享的 RAUNet 特征。
            "voxel_features": voxel_features,
        }
