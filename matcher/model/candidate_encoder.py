"""按四个具名变体把 Stage1 候选编码成候选表示和前景分数.

主要入口是 CandidateEncoder.forward.
strongest/trunk_only 使用完整 106 通道体素与四类 token;  density_blob_only 使用单实验密度通道与 V token;  pocket_only 只使用原始 A/L0.
本模块返回内存张量, 不读取清单、不计算损失且不写入磁盘.
"""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn

from ..data.schema import A_TOKEN_DIM, P_TOKEN_DIM, PP_TOKEN_DIM, V_TOKEN_DIM
from ..datasets.contracts import CandidateBatch
from .config import MatcherModelConfig
from .fusion import (
    MeanMaxAggregator,
    MeanMaxCandidateConditionedResidualAggregator,
    TokenProjector,
    gather_voxel_features,
    normalize_candidate_manual_features,
    normalize_candidate_sampling_features,
)
from .raunet import RaUnetBackbone


class CandidateEncoderOutput(TypedDict):
    """
    保存候选编码器的数值输出.

    字段:
        - candidate_embedding: 浮点张量, (B,D_m); 候选统一表示.
        - foreground_logit: 浮点张量, (B,); 真实五类前景相对背景的二分类 logit.
        - ligand_area_logits: 浮点张量或 None; 有区域主干时为 (B,48,48,48).
        - binding_area_logits: 浮点张量或 None; 有区域主干时为 (B,48,48,48).
        - voxel_features: 浮点张量或 None; 有 U-Net 时为 (B,C_v,48,48,48).
    """
    candidate_embedding: torch.Tensor
    foreground_logit: torch.Tensor
    ligand_area_logits: torch.Tensor | None
    binding_area_logits: torch.Tensor | None
    voxel_features: torch.Tensor | None


class VoxelBinaryHead(nn.Module):
    """
    从同分辨率三维特征生成区域 logit.

    - 输入: 浮点 (B,C_v,Z,Y,X) U-Net 特征.
    - 输出: 浮点 (B,Z,Y,X); 每个体素一个未归一化二分类分数.
    """
    def __init__(self, channels: int) -> None:
        """
        建立两个一乘一乘一卷积.

        - channels: 输入与隐藏通道数 C_v.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(channels, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        执行逐体素区域分类.

        - features: 浮点 (B,C_v,Z,Y,X); 空间分辨率保持不变.
        - return: 浮点 (B,Z,Y,X); 删除卷积输出的单通道轴.
        """
        return self.layers(features)[:, 0]


class PocketMeanMaxEncoder(nn.Module):
    def __init__(self, model_dim: int, dropout: float) -> None:
        """
        编码 pocket-only 的 A-L0 token.

        - 输入: 浮点 (B,N_A,50) 原始 L0 特征和 bool (B,N_A) 有效掩码.
        - 输出: 浮点 (B,D_m); 由有效 token 的逐通道均值与最大值融合得到.
        - 构造参数:
            - model_dim: 正整数 D_m; 逐 token 投影和候选输出宽度.
            - dropout: 融合多层感知机的丢弃概率.
            - return: None; 建立 50→D_m TokenProjector 和 2D_m→D_m 融合层.
        """
        super().__init__()
        self.projector = TokenProjector(50, model_dim, dropout)
        self.fusion = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        把浮点 (B,N_A,50) A-L0 token 归约为浮点 (B,D_m) 表示.

        - tokens: A token 原始 L0 特征, 不含概率、坐标或 L1/L2/L3.
        - mask: bool (B,N_A) 有效位置; 全空候选得到零 mean/max 输入.
        - return: 浮点 (B,D_m) pocket-only 候选表示.
        """
        projected = self.projector(tokens, mask)
        if projected.shape[1] == 0:
            mean = torch.zeros(
                (projected.shape[0], projected.shape[2]),
                dtype=projected.dtype,
                device=projected.device,
            )
            maximum = mean
        else:
            weights = mask[..., None].to(dtype=projected.dtype)
            mean = (projected * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            maximum = projected.masked_fill(
                ~mask[..., None],
                torch.finfo(projected.dtype).min,
            ).amax(dim=1)
            maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        return self.fusion(torch.cat((mean, maximum), dim=-1))


class CandidateEncoder(nn.Module):
    def __init__(self, config: MatcherModelConfig) -> None:
        """
        把一个 Stage1 候选编码为前景二分类与 PDB 内闭集身份打分共用的候选表示.

        形状符号:
            - B: 物理批次中的候选数.
            - D_m: `config.model_dim`, 候选表示与各投影 token 的共同宽度.
            - C_v: `config.voxel_feature_channels`, RAUNet 最高分辨率输出的通道数.
            - N_A、N_P、N_V、N_PP: 批内补齐后的受体原子、Stage1 锚点、source blob 体素与高概率体素 token 数.

        候选模态:
            - A: 受体原子 token. `strongest` 与 `trunk_only` 使用 L0-L3、Stage1 概率和相对 XYZ 共 230 维, `pocket_only` 只使用原始 50 维 L0.
            - P: Stage1 锚点 token. `strongest` 与 `trunk_only` 使用 L2、L3、Stage1 概率和相对 XYZ 共 116 维.
            - V: 当前 source blob 落入 supplied 48 立方体的体素 token. 非 `pocket_only` 变体使用 64 维 Stage1 特征、概率和相对 XYZ 共 68 维.
            - PP: supplied 48 立方体的完整 Stage1 概率图中按概率稳定选择的高概率体素 token. `strongest` 与 `trunk_only` 使用概率和相对 XYZ 共 4 维.

        变体路径:
            - `strongest` 与 `trunk_only`: 使用 106 通道体素、A/P/V/PP token、12 项模态统计和 4 项裁剪统计. 两者在本类中完全相同, `trunk_only` 的差异只发生在后续身份打分器.
            - `density_blob_only`: 使用单实验密度通道、V token、仅 V 组非零的 12 项模态统计和 4 项裁剪统计.
            - `pocket_only`: 只对 A token 的原始 L0 做掩码均值与最大值归约, 不实例化 RAUNet、体素调制器或区域辅助头.

        输入参数:
            - batch: CandidateBatch. B 轴由 `collate_candidates` 的输入候选顺序定义, 每种 token 的第二轴补齐到当前批内最大长度.
            - batch.voxel_input: 浮点张量, (B,C_in,48,48,48). `strongest` 与 `trunk_only` 的 C_in=106, `density_blob_only` 的 C_in=1, `pocket_only` 不读取该字段.
            - batch.a_tokens: 浮点张量. `strongest` 与 `trunk_only` 为 (B,N_A,230), `pocket_only` 为 (B,N_A,50), `density_blob_only` 不读取该字段.
            - batch.a_mask: bool 张量, (B,N_A). True 表示 `a_tokens` 第二维的真实受体原子, False 表示批内补零槽.
            - batch.p_tokens: 浮点张量, (B,N_P,116). 只由 `strongest` 与 `trunk_only` 读取.
            - batch.p_mask: bool 张量, (B,N_P). True 表示 `p_tokens` 第二维的真实 Stage1 锚点.
            - batch.v_tokens: 浮点张量, (B,N_V,68). 由三个非 `pocket_only` 变体读取.
            - batch.v_mask: bool 张量, (B,N_V). 同时遮盖 `v_tokens` 与 `v_index_zyx` 的第二维, False 表示批内补零槽.
            - batch.v_index_zyx: int64 张量, (B,N_V,3). 每个 V token 在 supplied 48 立方体中的 Z、Y、X 整数索引.
            - batch.pp_tokens: 浮点张量, (B,N_PP,4). 只由 `strongest` 与 `trunk_only` 读取.
            - batch.pp_mask: bool 张量, (B,N_PP). 同时遮盖 `pp_tokens` 与 `pp_index_zyx` 的第二维.
            - batch.pp_index_zyx: int64 张量, (B,N_PP,3). 每个 PP token 在 supplied 48 立方体中的 Z、Y、X 整数索引.
            - batch.candidate_manual_features: 浮点张量, (B,12). 按 A、P、V、PP 分为四组三元组, 每组依次为 token 数、Stage1 概率均值和最大值.
            - batch.candidate_sampling_features: 浮点张量, (B,4). 依次为裁剪边界限制标志、裁剪中心到 V 质心的 L2 距离、L-infinity 距离和 V 保留比例.

        返回值:
            - output: CandidateEncoderOutput. 五个字段与输入 B 轴逐候选对齐.
            - output.candidate_embedding: 浮点张量, (B,D_m). 后续闭集身份打分与前景二分类共用的候选表示.
            - output.foreground_logit: 浮点张量, (B,). 五类真实配体相对背景的未归一化二分类分数.
            - output.ligand_area_logits: 浮点张量 (B,48,48,48) 或 None. 非 `pocket_only` 变体返回配体区域辅助 logit.
            - output.binding_area_logits: 浮点张量 (B,48,48,48) 或 None. 非 `pocket_only` 变体返回结合受体区域辅助 logit.
            - output.voxel_features: 浮点张量 (B,C_v,48,48,48) 或 None. 非 `pocket_only` 变体返回共享 RAUNet 特征.

        实例状态:
            - self.config: MatcherModelConfig. 保留构造时的完整结构配置.
            - self.voxel_backbone: RaUnetBackbone 或 None. 非 `pocket_only` 变体把 (B,C_in,48,48,48) 体素输入编码为 (B,C_v,48,48,48), 其中 `density_blob_only` 的 C_in=1, 其余两个体素变体的 C_in=106.
            - self.candidate_summary: Sequential 或 None. 非 `pocket_only` 变体把 RAUNet 特征的空间均值与最大值拼接向量 (B,2C_v) 映射为 (B,D_m).
            - self.token_projectors: ModuleDict. `density_blob_only` 只含 `v`, `strongest` 与 `trunk_only` 含 `a`、`p`、`v`、`pp`, `pocket_only` 为空.
            - self.voxel_modulators: ModuleDict. `density_blob_only` 只含 `v`, `strongest` 与 `trunk_only` 含 `v`、`pp`, 每项把 token 同位置的 C_v 维 RAUNet 特征映射到 D_m.
            - self.aggregator: MeanMaxAggregator、MeanMaxCandidateConditionedResidualAggregator 或 None. 前者服务 `density_blob_only`, 后者服务 `strongest` 与 `trunk_only`.
            - self.pocket_encoder: PocketMeanMaxEncoder 或 None. 只在 `pocket_only` 中归约 50 维 A-L0 token.
            - self.ligand_area_head: VoxelBinaryHead 或 None. 非 `pocket_only` 变体从共享 RAUNet 特征产生配体区域 logit.
            - self.binding_area_head: VoxelBinaryHead 或 None. 非 `pocket_only` 变体从共享 RAUNet 特征产生结合受体区域 logit.
            - self.foreground_head: Sequential. 四个变体共用, 把 (B,D_m) 候选表示映射为 (B,1), `forward` 删除最后一个单例轴.

        本类不读取 SMILES 身份轴, 不计算闭集身份分数或训练损失, 也不对返回 logit 执行 sigmoid.
        """
        super().__init__()
        # MatcherModelConfig, 决定本实例的唯一候选编码路径, 并为 `_empty_modality` 提供 D_m.
        self.config = config
        # 下列可选模块先置空, 随后只为当前变体覆盖实际需要的成员, 使消融变体的参数树不包含禁用模块.
        self.voxel_backbone: RaUnetBackbone | None = None
        self.candidate_summary: nn.Module | None = None
        # ModuleDict[str,TokenProjector], 把当前变体启用的原始 token 从各自输入宽度投影到 D_m.
        self.token_projectors = nn.ModuleDict()
        # ModuleDict[str,Sequential], 把 V 或 PP 同位置采样的 C_v 维 RAUNet 特征映射到 D_m.
        self.voxel_modulators = nn.ModuleDict()
        self.aggregator: nn.Module | None = None
        self.pocket_encoder: PocketMeanMaxEncoder | None = None
        self.ligand_area_head: VoxelBinaryHead | None = None
        self.binding_area_head: VoxelBinaryHead | None = None

        if config.variant == "pocket_only":
            # pocket_only 不读取 voxel_input 或其它 token, 只归约 (B,N_A,50) A-L0.
            self.pocket_encoder = PocketMeanMaxEncoder(config.model_dim, config.dropout)
        else:
            # int, RAUNet 输入通道数 C_in. 单密度消融为 1, strongest/trunk_only 为 56 个密度通道加 50 个受体通道.
            input_channels = 1 if config.variant == "density_blob_only" else 106
            self.voxel_backbone = RaUnetBackbone(
                input_channels=input_channels,
                feature_channels=config.voxel_feature_channels,
                planes=config.raunet_planes,
                recycle_iterations=config.recycle_iterations,
                gradient_checkpointing=config.gradient_checkpointing,
                bottleneck_attention_layers=config.bottleneck_attention_layers,
            )
            # [B,C_v,48,48,48] -> [B,2C_v] -> [B,D_m], 先连接空间均值和逐通道最大值, 再形成候选体素摘要.
            self.candidate_summary = nn.Sequential(
                nn.LayerNorm(config.voxel_feature_channels * 2),
                nn.Linear(config.voxel_feature_channels * 2, config.model_dim),
                nn.GELU(),
                nn.LayerNorm(config.model_dim),
            )
            # dict[str,int], 当前变体实际读取的 token 名称及原始宽度. density_blob_only 不为禁用的 A/P/PP 建立投影参数.
            input_dims = (
                {"v": V_TOKEN_DIM}
                if config.variant == "density_blob_only"
                else {"a": A_TOKEN_DIM, "p": P_TOKEN_DIM, "v": V_TOKEN_DIM, "pp": PP_TOKEN_DIM}
            )
            self.token_projectors.update(
                {
                    name: TokenProjector(width, config.model_dim, config.dropout)
                    for name, width in input_dims.items()
                }
            )
            # tuple[str,...], 需要把同位置 RAUNet 特征加入 token 的模态. A/P 没有 supplied 48 立方体内的整数 ZYX 索引, 因而不做体素采样.
            modulated_names = ("v",) if config.variant == "density_blob_only" else ("v", "pp")
            self.voxel_modulators.update(
                {
                    name: nn.Sequential(
                        nn.Linear(config.voxel_feature_channels, config.model_dim),
                        nn.GELU(),
                        nn.Linear(config.model_dim, config.model_dim),
                    )
                    for name in modulated_names
                }
            )

            if config.variant == "density_blob_only":
                # density_blob_only 的 A/P/PP 在运行时保持空模态, MeanMaxAggregator 仍按固定四模态顺序接收它们的零摘要.
                self.aggregator = MeanMaxAggregator(
                    config.model_dim,
                    config.dropout,
                    include_manual=True,
                    include_sampling=True,
                )
            else:
                # strongest/trunk_only 使用同一候选侧聚合器, 把 mean/max 主表示与候选条件注意力表示送入零初始化残差分支.
                self.aggregator = MeanMaxCandidateConditionedResidualAggregator(
                    config.model_dim,
                    config.attention_heads,
                    config.dropout,
                    include_manual=True,
                    include_sampling=True,
                )
            # 两个区域头读取同一份 (B,C_v,48,48,48) RAUNet 特征, 各自输出一个无 sigmoid 的逐体素 logit 网格.
            self.ligand_area_head = VoxelBinaryHead(config.voxel_feature_channels)
            self.binding_area_head = VoxelBinaryHead(config.voxel_feature_channels)

        # [B,D_m] -> [B,1], 四个变体共用的前景二分类头; 最后一维在 `forward` 返回前删除.
        self.foreground_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim, 1),
        )

    def _empty_modality(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        为当前变体禁用的模态建立零长度投影 token 与有效掩码.

        输入参数:
            - batch_size: 整数 B. 当前物理批次中的候选数.
            - dtype: torch.dtype. 空 token 与 candidate_summary 共用的浮点类型.
            - device: torch.device. 空 token、掩码与 candidate_summary 共用的设备.

        返回值:
            - empty_tokens: 浮点张量, (B,0,D_m). 零长度 token 轴使固定四模态聚合器无需为禁用模态建立分支条件.
            - empty_mask: bool 张量, (B,0). 与 empty_tokens 的 B/token 轴对齐, 不含任何 True 位置.

        该返回值只作为只读占位传给聚合器, 不代表批次中存在零值 token.
        """
        return (
            torch.empty((batch_size, 0, self.config.model_dim), dtype=dtype, device=device),
            torch.empty((batch_size, 0), dtype=torch.bool, device=device),
        )

    def forward(self, batch: CandidateBatch) -> CandidateEncoderOutput:
        """
        按 `config.variant` 消费 CandidateBatch 的候选字段, 并返回候选级与逐体素预测.

        输入参数:
            - batch: CandidateBatch. B 轴由 `collate_candidates` 的输入候选顺序定义, 每种 token 的第二轴补齐到当前批内最大长度.
            - batch.voxel_input: 浮点张量, (B,C_in,48,48,48). `strongest` 与 `trunk_only` 的 C_in=106, `density_blob_only` 的 C_in=1, `pocket_only` 不读取该字段.
            - batch.a_tokens: 浮点张量. `strongest` 与 `trunk_only` 为 (B,N_A,230), `pocket_only` 为 (B,N_A,50), `density_blob_only` 不读取该字段.
            - batch.a_mask: bool 张量, (B,N_A). True 表示 `a_tokens` 第二维的真实受体原子, False 表示批内补零槽.
            - batch.p_tokens: 浮点张量, (B,N_P,116). 只由 `strongest` 与 `trunk_only` 读取.
            - batch.p_mask: bool 张量, (B,N_P). True 表示 `p_tokens` 第二维的真实 Stage1 锚点.
            - batch.v_tokens: 浮点张量, (B,N_V,68). 由三个非 `pocket_only` 变体读取.
            - batch.v_mask: bool 张量, (B,N_V). 同时遮盖 `v_tokens` 与 `v_index_zyx` 的第二维, False 表示批内补零槽.
            - batch.v_index_zyx: int64 张量, (B,N_V,3). 每个 V token 在 supplied 48 立方体中的 Z、Y、X 整数索引.
            - batch.pp_tokens: 浮点张量, (B,N_PP,4). 只由 `strongest` 与 `trunk_only` 读取.
            - batch.pp_mask: bool 张量, (B,N_PP). 同时遮盖 `pp_tokens` 与 `pp_index_zyx` 的第二维.
            - batch.pp_index_zyx: int64 张量, (B,N_PP,3). 每个 PP token 在 supplied 48 立方体中的 Z、Y、X 整数索引.
            - batch.candidate_manual_features: 浮点张量, (B,12). 按 A、P、V、PP 分为四组三元组, 每组依次为 token 数、Stage1 概率均值和最大值.
            - batch.candidate_sampling_features: 浮点张量, (B,4). 依次为裁剪边界限制标志、裁剪中心到 V 质心的 L2 距离、L-infinity 距离和 V 保留比例.

        返回值:
            - output: CandidateEncoderOutput. 五个字段与输入 B 轴逐候选对齐.
            - output.candidate_embedding: 浮点张量, (B,D_m). 后续闭集身份打分与前景二分类共用的候选表示.
            - output.foreground_logit: 浮点张量, (B,). 五类真实配体相对背景的未归一化二分类分数.
            - output.ligand_area_logits: 浮点张量 (B,48,48,48) 或 None. 非 `pocket_only` 变体返回配体区域辅助 logit.
            - output.binding_area_logits: 浮点张量 (B,48,48,48) 或 None. 非 `pocket_only` 变体返回结合受体区域辅助 logit.
            - output.voxel_features: 浮点张量 (B,C_v,48,48,48) 或 None. 非 `pocket_only` 变体返回共享 RAUNet 特征.

        `pocket_only` 不读取体素、P/V/PP 或候选统计. 其它三个变体始终使用归一化后的 12 项模态统计和 4 项裁剪统计. 本函数不消费 CandidateBatch 的身份字段或监督目标.
        """
        if self.config.variant == "pocket_only":
            if self.pocket_encoder is None:
                raise RuntimeError("pocket_only 编码器未建立。")
            # [B,N_A,50] + [B,N_A] -> [B,D_m], 只对真实 A-L0 token 做投影、均值和最大值归约.
            candidate_embedding = self.pocket_encoder(batch["a_tokens"], batch["a_mask"])
            # pocket_only 没有 RAUNet, 因而共享体素特征和两个区域辅助输出均显式返回 None.
            voxel_features = None
            ligand_area_logits = None
            binding_area_logits = None

        else:
            if self.voxel_backbone is None or self.candidate_summary is None or self.aggregator is None:
                raise RuntimeError("U-Net 候选路径未完整建立。")
            # 浮点张量, (B,C_in,48,48,48). Dataset 的 float16 缓存先对齐首层卷积参数 dtype, 自动混合精度仍由外层上下文决定实际算子精度.
            voxel_input = batch["voxel_input"].to(
                dtype=self.voxel_backbone.input_projection.weight.dtype
            )
            # 浮点张量, [B,C_in,48,48,48] -> [B,C_v,48,48,48], 最后一轮 RAUNet 回收得到的最高分辨率特征.
            voxel_features = self.voxel_backbone(voxel_input)
            # 浮点张量, [B,C_v,48,48,48] -> [B,2C_v]. 分别沿 Z/Y/X 三个空间轴求均值和逐通道最大值, 再按最后一维连接.
            pooled = torch.cat(
                (voxel_features.mean(dim=(2, 3, 4)), voxel_features.amax(dim=(2, 3, 4))),
                dim=-1,
            )
            # 浮点张量, [B,2C_v] -> [B,D_m], 为四模态聚合提供候选级体素摘要, 也是条件注意力的查询.
            candidate_summary = self.candidate_summary(pooled)
            # 二元组, 浮点 (B,0,D_m) token 与 bool (B,0) 掩码. 禁用模态共享该只读空占位.
            empty = self._empty_modality(
                candidate_summary.shape[0],
                candidate_summary.dtype,
                candidate_summary.device,
            )
            # dict[str,tuple[Tensor,Tensor]], 四键固定按 A/P/V/PP 对应投影 token 与有效掩码; 当前变体禁用的键暂时指向空占位.
            modalities: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
                "a": empty,
                "p": empty,
                "v": empty,
                "pp": empty,
            }
            # tuple[str,...], 本次实际从 CandidateBatch 读取并投影的模态名称. 顺序也是后续写回 modalities 的稳定顺序.
            names = ("v",) if self.config.variant == "density_blob_only" else ("a", "p", "v", "pp")
            for name in names:
                # 浮点 (B,N_m,F_m) 与 bool (B,N_m), 当前模态补齐后的原始 token 及真实 token 掩码.
                tokens = batch[f"{name}_tokens"]
                mask = batch[f"{name}_mask"]
                # 浮点张量, [B,N_m,F_m] -> [B,N_m,D_m]. TokenProjector 在末端把 mask=False 的补零位置重新清零.
                projected = self.token_projectors[name](tokens, mask)
                if name == "v":
                    if "v" not in self.voxel_modulators:
                        raise RuntimeError("V token 体素调制器未建立。")
                    # 浮点张量, [B,C_v,48,48,48] + [B,N_V,3] -> [B,N_V,C_v]. 按 v_index_zyx 为每个真实 V token 读取同位置 RAUNet 特征.
                    sampled = gather_voxel_features(voxel_features, batch["v_index_zyx"], mask)
                    # [B,N_V,D_m] + [B,N_V,C_v] -> [B,N_V,D_m], 把局部体素上下文作为加性调制写入 Stage1 V token 表示.
                    projected = projected + self.voxel_modulators["v"](sampled)
                    # 浮点张量, (B,N_V,D_m). 调制器含偏置, 因而相加后再次把补零 V token 严格清零.
                    projected = projected * mask[..., None].to(dtype=projected.dtype)
                elif name == "pp":
                    if "pp" not in self.voxel_modulators:
                        raise RuntimeError("PP token 体素调制器未建立。")
                    # 浮点张量, [B,C_v,48,48,48] + [B,N_PP,3] -> [B,N_PP,C_v]. 按 pp_index_zyx 读取每个高概率体素的 RAUNet 特征.
                    sampled = gather_voxel_features(voxel_features, batch["pp_index_zyx"], mask)
                    # [B,N_PP,D_m] + [B,N_PP,C_v] -> [B,N_PP,D_m], 把局部体素上下文加入概率与相对坐标投影.
                    projected = projected + self.voxel_modulators["pp"](sampled)
                    # 浮点张量, (B,N_PP,D_m). 调制器偏置可能激活补零槽, 因而返回聚合器前再次应用 PP 掩码.
                    projected = projected * mask[..., None].to(dtype=projected.dtype)
                # 当前键由空占位替换为浮点 (B,N_m,D_m) 投影 token 与 bool (B,N_m) 有效掩码.
                modalities[name] = (projected, mask)
            # 浮点张量, [B,12] -> [B,12]. 四组 token 数执行 log1p, 每组 Stage1 概率均值与最大值保持原值.
            manual = normalize_candidate_manual_features(batch["candidate_manual_features"])
            # 浮点张量, [B,4] -> [B,4]. 两项中心偏移分别按 24*sqrt(3) 与 24 缩放, 边界标志和 V 保留比例保持原值.
            sampling = normalize_candidate_sampling_features(batch["candidate_sampling_features"])
            # 浮点张量, [B,D_m] + 四模态投影 token/掩码 + [B,12] + [B,4] -> [B,D_m]. 具体归约方式由当前变体构造的聚合器固定.
            candidate_embedding = self.aggregator(
                candidate_summary,
                modalities,
                manual,
                sampling,
            )
            if self.ligand_area_head is None or self.binding_area_head is None:
                raise RuntimeError("区域辅助头未建立。")
            # 两个浮点张量, [B,C_v,48,48,48] -> [B,48,48,48]. 每个头独立产生一个未归一化的逐体素二分类分数.
            ligand_area_logits = self.ligand_area_head(voxel_features)
            binding_area_logits = self.binding_area_head(voxel_features)
        # CandidateEncoderOutput, 候选表示进入后续身份打分; foreground_head 输出的 (B,1) 删除最后一个单例轴得到 (B,).
        return {
            "candidate_embedding": candidate_embedding,
            "foreground_logit": self.foreground_head(candidate_embedding)[:, 0],
            "ligand_area_logits": ligand_area_logits,
            "binding_area_logits": binding_area_logits,
            "voxel_features": voxel_features,
        }
