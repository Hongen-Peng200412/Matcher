"""计算前景二分类、两个区域辅助项和直接 SMILES 身份交叉熵.

主要入口是 compute_matcher_loss. balanced_voxel_loss 负责区域辅助监督,
smiles_identity_cross_entropy 只在受支持真前景候选上计算闭集身份分类. 本模块返回连图
标量与真实分母, 不修改模型、优化器或磁盘产物.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class LossWeights:
    """保存四项非负损失系数.

    - foreground: 候选前景二分类 BCE 系数.
    - ligand_area: 配体区域辅助损失系数.
    - binding_area: 受体结合区域辅助损失系数.
    - identity: 真实前景且身份受支持候选的直接 SMILES 交叉熵系数.
"""

    foreground: float
    ligand_area: float
    binding_area: float
    identity: float

    def __post_init__(self) -> None:
        """核对四项损失系数.

        - return: 四项均为有限非负数时返回 None; 布尔值、非数值、NaN、Inf 或负值失败.
"""

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in self.__dict__.values()
        ):
            raise ValueError("损失权重必须是有限非负数。")


class LossOutput(TypedDict):
    """保存一次物理批次的可反向损失和真实聚合分母.

    - total: 四项损失加权和.
    - foreground: 当前物理批次的前景二分类 BCE.
    - ligand_area: 配体区域辅助损失; 模型无该头时为连图零.
    - binding_area: 结合区域辅助损失; 模型无该头时为连图零.
    - identity: 受支持真前景上的直接 SMILES 交叉熵; 无合格候选时为连图零.
    - foreground_denominator: 当前物理批次候选数 B.
    - identity_eligible_count: 实际进入身份交叉熵的候选数.
"""

    total: torch.Tensor
    foreground: torch.Tensor
    ligand_area: torch.Tensor
    binding_area: torch.Tensor
    identity: torch.Tensor
    foreground_denominator: torch.Tensor
    identity_eligible_count: torch.Tensor


def balanced_voxel_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算区域辅助损失.

    - logits: 浮点 (B,Z,Y,X); 每个体素的未归一化二分类分数.
    - target: bool 或 0/1 数值 (B,Z,Y,X); 与 logits 逐体素对齐.
    - return: 标量张量; 每个候选的正负平衡 BCE 与软 Dice 之和, 再沿 B 取均值.

    没有正体素的候选只使用负类 BCE; Dice 的分子和分母加一, 保持空目标结果有限.
"""

    if logits.shape != target.shape or logits.ndim != 4:
        raise ValueError("区域 logits 与 target 必须共享 (B,Z,Y,X) 形状。")
    # 浮点 (B,Z,Y,X), 与 logits dtype 一致的 0/1 区域目标.
    target_float = target.to(dtype=logits.dtype)
    # 浮点 (B,Z*Y*X), 未归约的逐体素 BCE, 只展平空间轴.
    element = F.binary_cross_entropy_with_logits(logits, target_float, reduction="none").flatten(1)
    # 浮点 (B,Z*Y*X), 与 element 逐体素对齐的 0/1 目标.
    flat_target = target_float.flatten(1)
    # 浮点 (B,), 每个候选沿全部空间体素求和得到的正体素数.
    positive_count = flat_target.sum(dim=1)
    # 浮点 (B,), 每个候选沿全部空间体素求和得到的负体素数.
    negative_count = (1 - flat_target).sum(dim=1)
    # 浮点 (B,), 每个候选只在正体素上求均值的 BCE, 空正类分母夹到 1.
    positive = (element * flat_target).sum(dim=1) / positive_count.clamp_min(1)
    # 浮点 (B,), 每个候选只在负体素上求均值的 BCE.
    negative = (element * (1 - flat_target)).sum(dim=1) / negative_count.clamp_min(1)
    # 浮点 (B,), 有正类时正负 BCE 等权, 无正类时只保留负类 BCE.
    balanced_bce = torch.where(positive_count > 0, 0.5 * (positive + negative), negative)
    # 浮点 (B,Z*Y*X), sigmoid 后的逐体素前景概率.
    probability = torch.sigmoid(logits).flatten(1)
    # 浮点 (B,), 每个候选沿空间轴求和的软预测与真值交集.
    intersection = (probability * flat_target).sum(dim=1)
    # 浮点 (B,), 加一平滑后的软 Dice 损失.
    dice = 1 - (2 * intersection + 1) / (probability.sum(dim=1) + positive_count + 1)
    return (balanced_bce + dice).mean()


def smiles_identity_cross_entropy(
    identity_logits: torch.Tensor,
    identity_mask: torch.Tensor,
    identity_target: torch.Tensor,
    foreground_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在真实前景且身份受支持的候选上计算直接身份交叉熵.

    - identity_logits: 浮点张量, (B,I). 每个候选在所属 PDB 闭集身份轴上的未归一化分数.
    - identity_mask: bool 张量, (B,I). True 表示真实身份槽, False 表示批内补零槽.
    - identity_target: int64 张量, (B,). 受支持真前景指向 [0,I), 其它候选为 -1.
    - foreground_target: bool 张量, (B,). True 表示通过双向覆盖门槛的真实前景.
    - return[0]: 标量浮点张量. eligible 子集上的均值交叉熵, 无 eligible 候选时为连图零.
    - return[1]: int64 标量张量. 当前物理批次的 eligible 候选数.

    eligible = foreground_target & (identity_target >= 0). 损失不乘前景预测概率, 也不提供
    其它身份损失变体或历史兼容分支.
"""

    if identity_mask.dtype != torch.bool or identity_mask.shape != identity_logits.shape:
        raise ValueError("identity_mask 必须是与 identity_logits 同形的 bool 张量。")
    if identity_target.dtype != torch.long or identity_target.shape != foreground_target.shape:
        raise ValueError("identity_target 必须是与 foreground_target 对齐的 int64 张量。")
    # bool (B,), True 表示真实前景且目标精确 SMILES 具有图和语言资产.
    eligible = foreground_target.bool() & (identity_target >= 0)
    # int64 标量, 当前物理批次实际进入身份交叉熵的候选数.
    count = eligible.sum()
    if not torch.any(eligible):
        return identity_logits[eligible].sum() * 0, count
    # int64 (N_eligible,), eligible 候选在各自 PDB 身份轴上的目标槽下标.
    selected_target = identity_target[eligible]
    if torch.any(selected_target >= identity_logits.shape[1]):
        raise ValueError("identity_target 超出身份轴。")
    if not torch.all(identity_mask[eligible, selected_target]):
        raise ValueError("identity_target 必须指向有效身份槽。")
    return F.cross_entropy(identity_logits[eligible], selected_target), count


def compute_matcher_loss(
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, torch.Tensor],
    weights: LossWeights,
) -> LossOutput:
    """计算一个物理批次的四项正式训练损失.

    - outputs.foreground_logit: 浮点 (B,); 前景二分类 logit.
    - outputs.identity_logits: 浮点 (B,I); PDB 内闭集 SMILES 分数.
    - outputs.ligand_area_logits/binding_area_logits: 浮点区域张量或 None.
    - batch.foreground_target: bool (B,); 与身份资产可用性无关的几何前景真值.
    - batch.identity_target: int64 (B,); 受支持真前景指向目标身份, 其余为 -1.
    - weights: 四项损失系数; 区域头不存在时相应权重必须为零.
    - return: 可用于真实分母聚合的 LossOutput.
"""

    foreground_logit = outputs["foreground_logit"]
    identity_logits = outputs["identity_logits"]
    if not isinstance(foreground_logit, torch.Tensor) or not isinstance(identity_logits, torch.Tensor):
        raise TypeError("模型必须返回 foreground_logit 和 identity_logits 张量。")
    foreground_target = batch["foreground_target"]
    foreground = F.binary_cross_entropy_with_logits(
        foreground_logit, foreground_target.to(dtype=foreground_logit.dtype)
    )
    denominator = foreground_logit.new_tensor(foreground_target.numel())
    zero = foreground_logit.sum() * 0
    ligand_logits = outputs["ligand_area_logits"]
    binding_logits = outputs["binding_area_logits"]
    if ligand_logits is None:
        if weights.ligand_area != 0:
            raise ValueError("模型无配体区域头时 loss.ligand_area 必须为 0。")
        ligand_area = zero
    else:
        ligand_area = balanced_voxel_loss(ligand_logits, batch["ligand_area_target"])
    if binding_logits is None:
        if weights.binding_area != 0:
            raise ValueError("模型无结合区域头时 loss.binding_area 必须为 0。")
        binding_area = zero
    else:
        binding_area = balanced_voxel_loss(binding_logits, batch["binding_area_target"])
    identity, identity_count = smiles_identity_cross_entropy(
        identity_logits,
        batch["identity_mask"],
        batch["identity_target"],
        foreground_target,
    )
    total = (
        weights.foreground * foreground
        + weights.ligand_area * ligand_area
        + weights.binding_area * binding_area
        + weights.identity * identity
    )
    return {
        "total": total,
        "foreground": foreground,
        "ligand_area": ligand_area,
        "binding_area": binding_area,
        "identity": identity,
        "foreground_denominator": denominator,
        "identity_eligible_count": identity_count,
    }
