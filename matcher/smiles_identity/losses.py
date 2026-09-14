"""计算候选六分类、两个体素区域和单一 SMILES 闭集身份损失. 

主要入口 compute_matcher_loss 返回可反向的总损失、四个损失分量和两个真实聚合分母.
本模块只执行张量计算, 不读取或写出文件. 
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypedDict

import torch
from torch.nn import functional as F

from .schema import CLASS_NAMES


IdentityLossVariant = Literal["oracle", "detached", "joint"]


@dataclass(frozen=True)
class LossWeights:
    """保存四项非负损失系数. 

    - classification: 六分类交叉熵系数. 
    - ligand_area: 配体区域平衡 BCE 与软 Dice 系数. 
    - binding_area: 结合区域平衡 BCE 与软 Dice 系数. 
    - identity: PDB 内精确 SMILES 交叉熵系数. 
    """

    classification: float
    ligand_area: float
    binding_area: float
    identity: float

    def __post_init__(self) -> None:
        """拒绝非有限、负数、布尔值或非数值损失系数. 

        - return: None; 四项权重合法时保留实例, 否则在训练前抛出 ValueError. 
        """

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in self.__dict__.values()
        ):
            raise ValueError("损失权重必须是有限非负数。")


class LossOutput(TypedDict):
    """保存可反向的四项损失、总损失和两个真实聚合分母. 

    - total: 浮点标量张量, 四项损失按 LossWeights 加权后的和. 
    - classification: 浮点标量张量, 当前物理批次的六分类交叉熵. 
    - ligand_area: 浮点标量张量, 当前物理批次的配体区域平衡 BCE 与软 Dice 均值. 
    - binding_area: 浮点标量张量, 当前物理批次的受体结合区域平衡 BCE 与软 Dice 均值. 
    - identity: 浮点标量张量, 合格前景候选的精确 SMILES 交叉熵; 关闭或没有合格候选时为连图零. 
    - classification_denominator: 浮点标量张量, 未加权时为候选数 B, 使用 class_weights 时为真实类别权重之和. 
    - identity_eligible_count: int64 标量张量, identity_target 非负并实际进入身份损失的候选数. 
    """

    total: torch.Tensor
    classification: torch.Tensor
    ligand_area: torch.Tensor
    binding_area: torch.Tensor
    identity: torch.Tensor
    classification_denominator: torch.Tensor
    identity_eligible_count: torch.Tensor


def balanced_voxel_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算逐候选正负平衡 BCE 与软 Dice 之和, 再沿批次取均值. 

    - logits: 浮点张量, (B, Z, Y, X), 逐体素未归一化分数. 
    - target: bool 或数值张量, 与 logits 同形; 非零位置是正类体素. 
    - return: 可反向标量; 每个候选的平衡 BCE 加软 Dice 后沿 B 取均值. 

    含正体素的候选把正、负 BCE 各赋一半权重; 全负候选只使用负类 BCE. Dice 的
    分子和分母都加一, 避免空目标产生除零. 
    """

    if logits.shape != target.shape or logits.ndim != 4:
        raise ValueError("区域 logits 与 target 必须共享 (B,Z,Y,X) 形状。")
    # 浮点张量, (B, Z, Y, X); target 转成 logits dtype 后的 0/1 体素目标.
    target_float = target.to(dtype=logits.dtype)
    # 浮点张量, (B, Z*Y*X); 逐体素 BCE 展平三个空间轴, 保留候选轴.
    element = F.binary_cross_entropy_with_logits(logits, target_float, reduction="none").flatten(1)
    # 浮点张量, (B, Z*Y*X); 与 element 逐体素对齐的展平 0/1 目标.
    flat_target = target_float.flatten(1)
    # 浮点张量, (B,); 每个候选的正类体素数.
    positive_count = flat_target.sum(dim=1)
    # 浮点张量, (B,); 每个候选的负类体素数.
    negative_count = (1 - flat_target).sum(dim=1)
    # 浮点张量, (B,); 每个候选在正类体素上的 BCE 均值, 无正类时分母下限为 1.
    positive = (element * flat_target).sum(dim=1) / positive_count.clamp_min(1)
    # 浮点张量, (B,); 每个候选在负类体素上的 BCE 均值.
    negative = (element * (1 - flat_target)).sum(dim=1) / negative_count.clamp_min(1)
    # 浮点张量, (B,); 有正类时正负 BCE 各占一半, 全负候选只使用负类 BCE.
    balanced_bce = torch.where(positive_count > 0, 0.5 * (positive + negative), negative)
    # 浮点张量, (B, Z*Y*X); Sigmoid 概率展平三个空间轴.
    probability = torch.sigmoid(logits).flatten(1)
    # 浮点张量, (B,); 每个候选预测概率与正类目标逐体素乘积之和.
    intersection = (probability * flat_target).sum(dim=1)
    # 浮点张量, (B,); 分子和分母加 1 的逐候选软 Dice 损失.
    dice = 1 - (2 * intersection + 1) / (probability.sum(dim=1) + positive_count + 1)
    return (balanced_bce + dice).mean()


def _identity_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    variant: IdentityLossVariant,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 identity_target 非负的全部前景候选计算 SMILES 身份交叉熵. 

    - outputs.identity_logits: 浮点 (B, I), 已屏蔽补零槽的闭集身份分数. 
    - outputs.class_logits: 浮点 (B, 6), detached 或 joint 模式的类别置信度来源. 
    - batch.identity_mask: bool (B, I), 真实身份槽掩码. 
    - batch.identity_class_index: int64 (B, I), 有效槽的五类前景下标. 
    - batch.identity_target: int64 (B,), 背景为 -1, 前景指向有效身份槽. 
    - batch.class_target: int64 (B,), 用于核对目标身份类别. 
    - variant: oracle、detached 或 joint; 决定是否以真实类别概率缩放逐候选交叉熵. 
    - return[0]: 可反向标量身份损失; 没有合格前景时为与 logits 连图的零. 
    - return[1]: int64 标量张量; 身份损失实际候选数. 
    """

    # 浮点张量, (B, I); 已在模型侧屏蔽补零槽的身份分数.
    identity_logits = outputs["identity_logits"]
    # bool 张量, (B, I); True 表示真实精确 SMILES 身份槽.
    identity_mask = batch["identity_mask"]
    # int64 张量, (B, I); 有效槽为五类前景下标, 补零槽为 -1.
    identity_class = batch["identity_class_index"]
    # int64 张量, (B,); 前景指向身份轴, 背景为 -1.
    target = batch["identity_target"]
    if identity_mask.dtype != torch.bool or identity_mask.shape != identity_logits.shape:
        raise ValueError("identity_mask 必须是与 identity_logits 同形的 bool 张量。")
    if identity_class.shape != identity_mask.shape or identity_class.dtype != torch.long:
        raise ValueError("identity_class_index 必须是与身份掩码同形的 int64 张量。")
    if torch.any(identity_class[identity_mask] <= 0) or torch.any(
        identity_class[identity_mask] >= len(CLASS_NAMES)
    ):
        raise ValueError("有效身份类别必须属于五类前景。")
    # bool 张量, (B,); True 表示当前候选具有精确 SMILES 身份监督.
    eligible = target >= 0
    # int64 标量张量, 当前批次实际进入身份交叉熵的候选数.
    count = eligible.sum()
    if not torch.any(eligible):
        return identity_logits[eligible].sum() * 0, count
    # int64 张量, (B_eligible,); 合格候选在各自 PDB 身份轴上的目标槽下标.
    selected_target = target[eligible]
    if torch.any(selected_target >= identity_logits.shape[1]):
        raise ValueError("identity_target 超出身份轴。")
    if not torch.all(identity_mask[eligible, selected_target]):
        raise ValueError("identity_target 必须指向有效身份槽。")
    if not torch.equal(identity_class[eligible, selected_target], batch["class_target"][eligible]):
        raise ValueError("目标 SMILES 身份类别必须等于候选真实类别。")
    # 浮点张量, (B_eligible,); 每个合格候选的闭集 SMILES 交叉熵.
    per_candidate = F.cross_entropy(identity_logits[eligible], selected_target, reduction="none")
    if variant == "oracle":
        multiplier = torch.ones_like(per_candidate)
    else:
        # 浮点张量, (B_eligible,); 每个合格候选对其真实前景类别的六分类概率.
        class_probability = torch.softmax(outputs["class_logits"][eligible], dim=-1).gather(
            1,
            batch["class_target"][eligible, None],
        )[:, 0]
        # 浮点张量, (B_eligible,); detached 阻断类别概率梯度, joint 保留该梯度.
        multiplier = class_probability.detach() if variant == "detached" else class_probability
    return (per_candidate * multiplier).sum() / count.to(per_candidate.dtype), count


# ================================================================================================


def compute_matcher_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: LossWeights,
    identity_variant: IdentityLossVariant | None,
    class_weights: torch.Tensor | None = None,
) -> LossOutput:
    """计算一个物理批次的候选和身份损失. 

    - outputs.class_logits: 浮点 (B, 6). 
    - outputs.ligand_area_logits: 浮点 (B, 48, 48, 48), 配体区域未归一化分数.
    - outputs.binding_area_logits: 浮点 (B, 48, 48, 48), 受体结合区域未归一化分数.
    - outputs.identity_logits: 浮点 (B, I); identity_variant 非 None 时必需. 
    - batch.class_target: int64 (B,). 
    - batch.ligand_area_target: bool (B, 48, 48, 48), 配体区域二值目标.
    - batch.binding_area_target: bool (B, 48, 48, 48), 受体结合区域二值目标.
    - batch.identity_target: int64 (B,); 所有五类前景均以非负值进入身份监督. 
    - weights: LossWeights, 四项非负系数分别缩放六分类、配体区域、受体结合区域和身份损失. 
    - weights.classification: float, 六分类损失系数. 
    - weights.ligand_area: float, 配体区域损失系数. 
    - weights.binding_area: float, 受体结合区域损失系数. 
    - weights.identity: float, 精确 SMILES 身份损失系数. 
    - identity_variant: None 关闭身份损失; oracle、detached、joint 控制类别置信度乘数. 
    - class_weights: None 或浮点张量, (6,); 六分类各真实类别的交叉熵权重. 
    - return: LossOutput, 七个标量字段的形状、聚合分母和梯度语义见 LossOutput. 
    """

    # 浮点张量, (B,); 每个候选的六分类交叉熵, 已应用可选真实类别权重.
    per_candidate = F.cross_entropy(
        outputs["class_logits"],
        batch["class_target"],
        weight=class_weights,
        reduction="none",
    )
    # 浮点标量张量, 未加权时为 B, 加权时为当前批次真实类别权重之和.
    denominator = (
        outputs["class_logits"].new_tensor(batch["class_target"].numel())
        if class_weights is None
        else class_weights[batch["class_target"]].sum()
    )
    # 浮点标量张量, 按真实分类分母归一化的六分类损失.
    classification = per_candidate.sum() / denominator.clamp_min(1.0)
    ligand_area = balanced_voxel_loss(outputs["ligand_area_logits"], batch["ligand_area_target"])
    binding_area = balanced_voxel_loss(outputs["binding_area_logits"], batch["binding_area_target"])
    # 浮点标量张量, 不含身份项的三项加权候选损失.
    base = (
        weights.classification * classification
        + weights.ligand_area * ligand_area
        + weights.binding_area * binding_area
    )
    # 浮点标量张量, 默认是与 base 同设备和 dtype 的连图前置零.
    identity = base.new_zeros(())
    # int64 标量张量, 默认没有候选进入身份监督.
    identity_count = torch.zeros((), device=base.device, dtype=torch.long)
    if identity_variant is not None:
        if identity_variant not in ("oracle", "detached", "joint"):
            raise ValueError(f"未知 identity_variant={identity_variant!r}。")
        if "identity_logits" not in outputs:
            raise KeyError("启用身份损失时模型必须返回 identity_logits。")
        identity, identity_count = _identity_loss(outputs, batch, identity_variant)
    total = base + weights.identity * identity
    return {
        "total": total,
        "classification": classification,
        "ligand_area": ligand_area,
        "binding_area": binding_area,
        "identity": identity,
        "classification_denominator": denominator,
        "identity_eligible_count": identity_count,
    }

