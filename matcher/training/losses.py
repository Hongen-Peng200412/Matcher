"""计算候选六分类、两个体素区域和闭集联合身份损失。

compute_matcher_loss 是训练与验证共同入口。B 表示物理批量候选数，I 表示批内补零后的
最大身份槽数，Z、Y、X 表示三个体素空间轴。oracle、detached 和 joint 只改变身份交叉熵
前的候选真实类别概率乘数是否存在、是否阻断梯度；三者都使用同一联合身份目标和分母。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypedDict

import torch
from torch.nn import functional as F

from ..data.schema import CLASS_NAMES, IdentityScope, identity_scope_class_indices


# Literal类型，真实类别概率不参与、阻断梯度或保留梯度的三种身份损失门控方式。
IdentityLossVariant = Literal["oracle", "detached", "joint"]


@dataclass(frozen=True)
class LossWeights:
    """保存四项损失在 total 中使用的无量纲非负系数。

    - classification: 六分类交叉熵系数。
    - ligand_area: 配体区域平衡 BCE 与软 Dice 和的系数。
    - binding_area: 受体结合区域平衡 BCE 与软 Dice 和的系数。
    - identity: 闭集联合身份交叉熵系数；候选预训练的正式配置要求为 0。
    """

    classification: float
    ligand_area: float
    binding_area: float
    identity: float

    def __post_init__(self) -> None:
        """拒绝不能由 math.isfinite 处理的值、非有限值和负权重。

        本地门禁遍历实例四个字段，但不使用 type(...) is float；可与 math.isfinite 和数值比较
        兼容的其它实数类型可能被接受。dataclass frozen 只禁止字段重绑。
        """

        if any(not math.isfinite(value) or value < 0.0 for value in self.__dict__.values()):
            raise ValueError("损失权重必须是有限非负数。")


class MatcherLossBatch(TypedDict):
    """声明损失函数从 CandidateBatch 读取的六个监督字段。

    - class_target: int64 张量，形状 (B,)；每项为 CLASS_NAMES 的六分类下标，正式 Dataset 保证范围，本模块没有单独做完整范围门禁。
    - ligand_area_target: bool 张量，形状 (B,Z,Y,X)；True 表示当前 PDB 全部 occurrence 稀疏掩码并集落入候选裁剪的体素，本模块在计算时转换为对应 logits dtype。
    - binding_area_target: bool 张量，形状 (B,Z,Y,X)；True 表示位于结合区域，本模块在计算时转换为对应 logits dtype。
    - identity_target: int64 张量，形状 (B,)；非负值是该候选在身份轴 I 上的联合身份槽，-1 表示没有身份监督，低于 -1 的值也会被 eligibility 条件排除但不属于正式输入契约。
    - identity_mask: bool 张量，形状 (B,I)；True 表示真实身份槽，False 表示批次补零槽。
    - identity_class_index: int64 张量，形状 (B,I)；有效槽为 1 到 len(CLASS_NAMES)-1 的非背景类别，补零槽正式值为 -1。

    有效 identity_target 必须指向 identity_mask 的 True 槽，且对应 identity_class_index
    必须等于同一候选 class_target。只有身份路径显式检查这些关系；基础区域和分类路径依赖
    Dataset 与 PyTorch 运算共同暴露其它形状、dtype 或范围错误。
    """

    class_target: torch.Tensor
    ligand_area_target: torch.Tensor
    binding_area_target: torch.Tensor
    identity_target: torch.Tensor
    identity_mask: torch.Tensor
    identity_class_index: torch.Tensor


class LossOutput(TypedDict):
    """声明单个物理批次的可反向损失与聚合分母。

    - total: 浮点零维张量；四项损失按 LossWeights 相加，设备与计算图来自模型输出。
    - classification: 浮点零维张量；批内六分类交叉熵分子除以 max(classification_denominator,1)。
    - ligand_area: 浮点零维张量；逐候选配体区域损失的算术均值。
    - binding_area: 浮点零维张量；逐候选结合区域损失的算术均值。
    - identity: 浮点零维张量；身份路径启用且存在有效候选时为当前 identity_scope 内的身份损失均值；启用但有效候选数为零时由 identity_logits[eligible].sum() * 0.0 构成并连接身份 logits 计算图；identity_variant 为 None 时由 base.new_zeros(()) 构成，只继承 base 的设备和 dtype，不读取或连接 identity_logits。
    - classification_denominator: 浮点零维张量；未配置类别权重时为 B，配置时为各 class_target 对应权重之和，返回值本身不执行 clamp。
    - identity_eligible_count: int64 零维张量；当前身份范围内 identity_target 非负的候选数。
    """

    total: torch.Tensor
    classification: torch.Tensor
    ligand_area: torch.Tensor
    binding_area: torch.Tensor
    identity: torch.Tensor
    classification_denominator: torch.Tensor
    identity_eligible_count: torch.Tensor


def compute_matcher_loss(
    outputs: dict[str, torch.Tensor],
    batch: MatcherLossBatch,
    weights: LossWeights,
    identity_variant: IdentityLossVariant | None,
    identity_scope: IdentityScope | None,
    class_weights: torch.Tensor | None,
) -> LossOutput:
    """计算一个物理批次的四项损失、加权总损失和跨批次聚合分母。

    - outputs: CandidateEncoder 或 ClosedSetMatcher 的数值输出映射。
    - outputs.class_logits: 浮点张量，形状 (B,6)；未归一化六分类分数。
    - outputs.ligand_area_logits: 浮点张量，形状 (B,Z,Y,X)；未经过 sigmoid 的配体区域分数。
    - outputs.binding_area_logits: 浮点张量，形状 (B,Z,Y,X)；未经过 sigmoid 的结合区域分数。
    - outputs.identity_logits: 浮点张量，形状 (B,I)；仅 identity_variant 非空时必需，正式模型已把补零身份槽置为 dtype 最小有限值。
    - batch: 当前物理批次的六字段 MatcherLossBatch。
    - batch.class_target: int64 张量，形状 (B,)；每个候选的真实六分类下标。
    - batch.ligand_area_target: bool 张量，形状 (B,Z,Y,X)；True 为配体区域目标体素。
    - batch.binding_area_target: bool 张量，形状 (B,Z,Y,X)；True 为结合区域目标体素。
    - batch.identity_target: int64 张量，形状 (B,)；非负值指向真实身份槽，负值表示监督不可用。
    - batch.identity_mask: bool 张量，形状 (B,I)；True 为真实身份槽，False 为补零槽。
    - batch.identity_class_index: int64 张量，形状 (B,I)；有效身份槽所属的非背景六分类下标。
    - weights: 当前阶段的四项非负 LossWeights。
    - weights.classification: 六分类损失的非负系数。
    - weights.ligand_area: 配体区域损失的非负系数。
    - weights.binding_area: 结合区域损失的非负系数。
    - weights.identity: 身份损失的非负系数。
    - identity_variant: oracle、detached、joint 或 None；None 禁用身份损失且要求 identity_scope 也为 None。
    - identity_scope: small_molecule、all_non_background 或 None；非空时决定哪些真实类别进入身份分母。
    - class_weights: None，或 PyTorch 张量；正式 engine 从 stage.training.class_weights 构造 CUDA float32 (6,) 张量，唯一轴按 CLASS_NAMES 顺序对齐；本函数自身不验证长度、dtype、设备、有限性、非负范围或总和，错误由 cross_entropy、class_target 索引或底层张量运算暴露。

    - 返回值.total: 四项损失按 weights 加权后的零维张量。
    - 返回值.classification: 六分类损失零维张量。
    - 返回值.ligand_area: 配体区域损失零维张量。
    - 返回值.binding_area: 结合区域损失零维张量。
    - 返回值.identity: 身份损失零维张量；启用身份路径但有效候选数为零时连接 identity_logits 图，identity_variant 为 None 时仅与 base 同设备、同 dtype 且不连接 identity_logits。
    - 返回值.classification_denominator: 本物理批次六分类真实聚合分母。
    - 返回值.identity_eligible_count: 本物理批次身份真实聚合分母。

    oracle 对每个有效候选使用乘数 1；detached 和 joint 使用模型对真实类别的 softmax
    概率，前者对该概率 detach，后者保留梯度。三种变体的身份损失始终除以有效候选数，
    不除以概率乘数之和。
    """

    # 浮点张量，形状 (B,)；逐候选六分类交叉熵，尚未做批次分母归一。
    per_candidate_classification = F.cross_entropy(
        outputs["class_logits"],
        batch["class_target"],
        weight=class_weights,
        reduction="none",
    )
    if class_weights is None:
        # 浮点零维张量；无类别权重时等于当前物理批次候选数 B。
        classification_denominator = outputs["class_logits"].new_tensor(
            batch["class_target"].numel()
        )
    else:
        # 浮点零维张量；当前物理批次真实类别对应的六分类权重和。
        classification_denominator = class_weights[batch["class_target"]].sum()
    # 浮点零维张量；分母仅在本地除法时下限为1，原始分母另行返回供global batch精确聚合。
    classification = per_candidate_classification.sum() / classification_denominator.clamp_min(1.0)
    # 浮点零维张量；逐候选配体区域平衡BCE与软Dice和的批次均值。
    ligand_area = balanced_voxel_loss(
        outputs["ligand_area_logits"],
        batch["ligand_area_target"],
    )
    # 浮点零维张量；逐候选结合区域平衡BCE与软Dice和的批次均值。
    binding_area = balanced_voxel_loss(
        outputs["binding_area_logits"],
        batch["binding_area_target"],
    )
    # 浮点零维张量；不含身份项的候选基础损失。
    base = (
        weights.classification * classification
        + weights.ligand_area * ligand_area
        + weights.binding_area * binding_area
    )
    # 浮点零维张量；身份路径禁用时保持与base同设备、同dtype的零值。
    identity = base.new_zeros(())
    # int64零维张量；身份路径禁用时没有有效身份候选。
    eligible_count = torch.zeros((), device=base.device, dtype=torch.long)
    if identity_variant is not None:
        if identity_scope is None:
            raise ValueError("启用身份损失时必须指定 identity_scope。")
        if identity_variant not in ("oracle", "detached", "joint"):
            raise ValueError(f"不支持 identity_variant={identity_variant!r}。")
        if "identity_logits" not in outputs:
            raise KeyError("完整 Matcher 身份损失需要 identity_logits。")
        identity, eligible_count = _identity_loss(
            outputs["identity_logits"],
            outputs["class_logits"],
            batch,
            identity_variant,
            identity_scope,
        )
    elif identity_scope is not None:
        raise ValueError("未启用身份损失时 identity_scope 必须为 None。")
    total = base + weights.identity * identity
    return {
        "total": total,
        "classification": classification,
        "ligand_area": ligand_area,
        "binding_area": binding_area,
        "identity": identity,
        "classification_denominator": classification_denominator,
        "identity_eligible_count": eligible_count,
    }


def balanced_voxel_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算同形体素分数与目标的逐候选平衡 BCE 加软 Dice。

    - logits: 浮点张量，形状 (B,Z,Y,X)；未经过 sigmoid 的体素分数。
    - target: 与 logits 同形的张量；正式输入为 bool，本函数只转换 dtype，不显式验证二值范围。

    - 返回值: 浮点零维张量；先在每个候选内计算损失，再沿 B 取算术均值。

    对含正体素的候选，BCE 是正体素平均与负体素平均的 0.5 等权和；没有正体素时只用
    负体素平均。全正候选的负项分子为零且分母经 clamp_min(1) 后为零贡献。软 Dice 使用
    分子、分母各加 1 的平滑公式，sigmoid 在有限精度下可能得到 0 或 1 端点。
    """

    if logits.shape != target.shape or logits.ndim != 4:
        raise ValueError("区域 logits/target 必须具有相同的 (B,Z,Y,X) 形状。")
    # 浮点张量，形状 (B,Z,Y,X)；只对齐logits dtype，保留target设备和数值。
    target_float = target.to(dtype=logits.dtype)
    # 浮点张量，形状 (B,Z,Y,X)；逐体素未归约BCE。
    element = F.binary_cross_entropy_with_logits(logits, target_float, reduction="none")
    # 浮点张量，形状 (B,Z*Y*X)；每个候选按NumPy/PyTorch连续轴顺序展平空间。
    flat_loss = element.flatten(1)
    # 浮点张量，形状 (B,Z*Y*X)；与flat_loss逐体素对齐的目标。
    flat_target = target_float.flatten(1)
    # 浮点张量，形状 (B,)；每个候选目标值之和，正式二值输入下等于正体素数。
    positive_count = flat_target.sum(dim=1)
    # 浮点张量，形状 (B,)；正式二值输入下等于负体素数。
    negative_count = (1.0 - flat_target).sum(dim=1)
    # 浮点张量，形状 (B,)；正目标位置BCE之和除以至少1的正体素数。
    positive = (flat_loss * flat_target).sum(dim=1) / positive_count.clamp_min(1.0)
    # 浮点张量，形状 (B,)；负目标位置BCE之和除以至少1的负体素数。
    negative = (flat_loss * (1.0 - flat_target)).sum(dim=1) / negative_count.clamp_min(1.0)
    # 没有正体素的样本只使用负项，避免人为加入恒为零的“正类平均”。
    balanced_bce = torch.where(positive_count > 0, 0.5 * (positive + negative), negative)
    # 浮点张量，形状 (B,Z*Y*X)；sigmoid体素概率，与flat_target逐位置对齐。
    probability = torch.sigmoid(logits).flatten(1)
    # 浮点张量，形状 (B,)；预测概率与目标乘积沿空间轴求和。
    intersection = (probability * flat_target).sum(dim=1)
    # 浮点张量，形状 (B,)；逐候选平滑软Dice损失。
    dice = 1.0 - (2.0 * intersection + 1.0) / (
        probability.sum(dim=1) + positive_count + 1.0
    )
    return (balanced_bce + dice).mean()


def _identity_loss(
    identity_logits: torch.Tensor,
    class_logits: torch.Tensor,
    batch: MatcherLossBatch,
    variant: IdentityLossVariant,
    identity_scope: IdentityScope,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按真实类别范围选择候选，并计算三种变体之一的身份损失。

    - identity_logits: 浮点张量，形状 (B,I)；正式 ClosedSetMatcher 已屏蔽补零身份槽。
    - class_logits: 浮点张量，形状 (B,6)；detached 与 joint 从中读取真实类别概率。
    - batch: 当前物理批次的六字段 MatcherLossBatch。
    - batch.class_target: int64 张量，形状 (B,)；候选真实六分类下标。
    - batch.ligand_area_target: 本函数不读取，但 MatcherLossBatch 保留的 bool (B,Z,Y,X) 字段。
    - batch.binding_area_target: 本函数不读取，但 MatcherLossBatch 保留的 bool (B,Z,Y,X) 字段。
    - batch.identity_target: int64 张量，形状 (B,)；非负项为真实身份槽。
    - batch.identity_mask: bool 张量，形状 (B,I)；True 为有效身份槽。
    - batch.identity_class_index: int64 张量，形状 (B,I)；有效槽保存非背景六分类下标。
    - variant: oracle、detached 或 joint；调用方已经做枚举门禁，本函数按值选择乘数。
    - identity_scope: small_molecule 或 all_non_background；决定允许的 class_target 下标。

    - 返回值[0]: 浮点零维身份损失；以有效候选数为分母，无有效候选时为连接 identity_logits 计算图的零。
    - 返回值[1]: int64 零维张量；当前范围内 identity_target 非负的候选数。

    本函数验证身份 mask、类别和目标槽关系，但不会在交叉熵前再次把补零 logits 改成负大值；
    该屏蔽由正式 ClosedSetMatcher 输出契约提供。
    """

    # int64张量，形状 (B,)；每个候选的身份槽目标，负值不进入身份监督。
    target = batch["identity_target"]
    # bool张量，形状 (B,I)；真实身份槽掩码。
    identity_mask = batch["identity_mask"]
    if (
        identity_mask.dtype != torch.bool
        or identity_mask.ndim != 2
        or identity_mask.shape != identity_logits.shape
    ):
        raise ValueError("identity_mask 必须是与 identity_logits 同形的 bool (B,I) 张量。")
    # int64张量，形状 (B,I)；每个有效身份槽对应的非背景六分类下标。
    identity_class_index = batch["identity_class_index"]
    if identity_class_index.shape != identity_mask.shape or identity_class_index.dtype != torch.long:
        raise ValueError("identity_class_index 必须是与 identity_mask 同形的 int64 张量。")
    if torch.any(identity_class_index[identity_mask] <= 0) or torch.any(
        identity_class_index[identity_mask] >= len(CLASS_NAMES)
    ):
        raise ValueError("有效身份必须属于非背景类别。")
    # bool张量，形状 (B,)；真实类别属于identity_scope且identity_target非负的候选。
    eligible = identity_eligible_mask(batch, identity_scope)
    # int64零维张量；身份损失真实候选分母。
    count = eligible.sum()
    if not torch.any(eligible):
        # 保留身份分支计算图，使分布式或自动混合精度训练的零样本批次行为一致。
        return identity_logits[eligible].sum() * 0.0, count
    # 浮点张量，形状 (B_e,I)；B_e为有效候选数，保留完整补零后身份轴。
    selected_logits = identity_logits[eligible]
    # int64张量，形状 (B_e,)；有效候选在身份轴上的真实目标。
    selected_target = target[eligible]
    if selected_logits.shape[1] == 0:
        raise ValueError("存在身份监督目标时 identity_logits 的身份轴不能为零。")
    if torch.any(selected_target >= selected_logits.shape[1]):
        raise ValueError("identity_target 超出 identity_logits 的身份轴。")
    if not torch.all(identity_mask[eligible, selected_target]):
        raise ValueError("identity_target 必须指向 identity_mask 中的有效身份。")
    # int64张量，形状 (B_e,)；每个目标身份槽对应的六分类下标。
    selected_class = identity_class_index[eligible, selected_target]
    if not torch.equal(selected_class, batch["class_target"][eligible]):
        raise ValueError("identity_target 指向的身份类别必须等于候选真实六分类类别。")
    # 浮点张量，形状 (B_e,)；每个有效候选在完整I轴上的身份交叉熵。
    per_candidate = F.cross_entropy(selected_logits, selected_target, reduction="none")
    if variant == "oracle":
        # 浮点张量，形状 (B_e,)；oracle不使用模型六分类置信度，乘数恒为1。
        multiplier = torch.ones_like(per_candidate)
    else:
        # 浮点张量，形状 (B_e,)；模型对每个候选真实类别的softmax概率。
        target_class_probability = torch.softmax(class_logits, dim=-1)[
            eligible,
            selected_class,
        ]
        # 浮点张量，形状 (B_e,)；detached阻断概率路径，joint保留该路径的梯度。
        multiplier = (
            target_class_probability.detach()
            if variant == "detached"
            else target_class_probability
        )
    return (multiplier * per_candidate).sum() / count.to(per_candidate.dtype), count


def identity_eligible_mask(
    batch: MatcherLossBatch,
    identity_scope: IdentityScope,
) -> torch.Tensor:
    """根据真实类别范围与身份目标可用性返回候选级监督掩码。

    - batch: 当前物理批次的六字段 MatcherLossBatch；本函数只读取其中两个字段。
    - batch.class_target: 张量，正式类型为 int64，形状 (B,)；决定候选是否属于 identity_scope。
    - batch.ligand_area_target: 本函数不读取的 MatcherLossBatch 字段。
    - batch.binding_area_target: 本函数不读取的 MatcherLossBatch 字段。
    - batch.identity_target: 与 class_target 同形的张量；非负值表示存在身份目标。
    - batch.identity_mask: 本函数不读取的 MatcherLossBatch 字段。
    - batch.identity_class_index: 本函数不读取的 MatcherLossBatch 字段。
    - identity_scope: small_molecule 或 all_non_background；转换为允许的六分类下标集合。

    - 返回值: bool 张量，形状 (B,)；True 当且仅当 class_target 属于范围且 identity_target 非负。

    本函数只显式检查 class_target 与 identity_target 同形且为一维，不检查 dtype、类别范围、
    identity_target 上界或其指向的 identity_mask；这些关系由身份损失后续门禁处理。
    """

    # 张量，正式为int64 (B,)；每个候选真实六分类下标。
    class_target = batch["class_target"]
    # 张量，正式为int64 (B,)；每个候选身份槽目标，负值表示不参与监督。
    target = batch["identity_target"]
    if class_target.shape != target.shape or class_target.ndim != 1:
        raise ValueError("class_target 与 identity_target 必须是同形 (B,) 张量。")
    # bool张量，形状 (B,)；逐个允许类别下标并入的真实类别范围掩码。
    allowed = torch.zeros_like(class_target, dtype=torch.bool)
    for class_index in identity_scope_class_indices(identity_scope):
        allowed |= class_target == class_index
    return allowed & (target >= 0)
