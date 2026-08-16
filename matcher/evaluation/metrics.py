"""跨物理批次累计 Matcher 的完整划分指标，并计算 validation 选模分数。

validation 是训练期间反复评估、用于选择 BEST checkpoint 的验证划分。本模块把每个物理
批次的六分类混淆矩阵、身份排名计数和区域体素计数累加为整数，最后一次性计算比率，因此
报告不依赖物理批次切分。身份指标同时区分忽略六分类结果的固有排名、要求六分类判对的
条件排名，以及经过背景或前景门控后的端到端判定。

本模块不读取文件，不生成 calibration 划分，也不负责把 NaN 转换为 JSON null。零分母在
内存报告中保留为 NaN；严格 JSON 写入器和 W&B 适配层分别在各自边界完成规范化。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Literal, TypedDict

import numpy as np
import torch

from ..data.schema import (
    CLASS_NAMES,
    FOREGROUND_CLASS_INDICES,
    FOREGROUND_CLASS_NAMES,
    SMALL_MOLECULE_INDEX,
    IdentityScope,
)
from ..datasets.contracts import CandidateMetadata


class EvaluationBatch(TypedDict):
    """声明评估器从一个物理批次读取的监督字段。

    B 是物理批次候选数，I 是本批次补齐后的最大身份槽数，Z、Y、X 是区域网格轴。本类型只
    描述评估器消费的 CandidateBatch 子集；EvaluationAccumulator.update 会核验这里列出的
    形状、dtype、目标范围和关键对齐关系，不重新验证候选 JSONL 的完整 schema。

    - class_target: Torch int64 张量，形状 (B,)，每项是 CLASS_NAMES 中 [0,6) 的真实类别下标。
    - ligand_area_target: Torch bool 张量，形状 (B,Z,Y,X)，True 表示候选配体区域真值体素。
    - binding_area_target: Torch bool 张量，形状 (B,Z,Y,X)，True 表示受体结合区域真值体素。
    - identity_target: Torch int64 张量，形状 (B,)，-1 表示当前候选没有完整可用的联合身份监督，否则是 [0,I) 内的真实身份槽下标。
    - identity_mask: Torch bool 张量，形状 (B,I)，True 表示该候选所在 PDB 的真实身份槽，False 表示批次补零槽。
    - identity_class_index: Torch int64 张量，形状 (B,I)，有效槽保存五类非背景下标 1 至 5，补零槽保存 -1。
    - metadata: 长度 B 的 CandidateMetadata 列表；身份历史拆分只读取其中 identity_seen_in_train。
    - metadata[*].split: 候选所属物化划分名称；本模块保留但不读取该字段。
    - metadata[*].pdb_id: 候选所属 PDB 标识；本模块保留但不读取该字段。
    - metadata[*].candidate_index: 当前 PDB 内的候选编号；本模块保留但不读取该字段。
    - metadata[*].matched_object_key: 命中 occurrence 的模板键或 None；本模块保留但不据此判定身份资格。
    - metadata[*].matched_prepared_smiles: 命中五类前景 occurrence 的顶层 PreparedLigand.smiles 或 None；本模块保留但不读取该字段。
    - metadata[*].matched_prepared_smiles_status: 顶层 PreparedLigand.smiles 的可用状态；本模块保留但不读取该字段。
    - metadata[*].identity_seen_in_train: 布尔值或 None；只有进入已配置条件身份分母的候选会被要求为严格 bool，并据此归入 seen 或 unseen。
    """

    class_target: torch.Tensor
    ligand_area_target: torch.Tensor
    binding_area_target: torch.Tensor
    identity_target: torch.Tensor
    identity_mask: torch.Tensor
    identity_class_index: torch.Tensor
    metadata: list[CandidateMetadata]


class PerClassMetrics(TypedDict):
    """描述六分类中一个类别的计数和比率。

    - precision: 浮点数，当前类别真阳性数除以 predicted_count；predicted_count 为零时是 NaN。
    - recall: 浮点数，当前类别真阳性数除以 support；support 为零时是 NaN。
    - f1: 浮点数，precision 与 recall 均有限且两者之和大于零时取调和平均，否则取 0。
    - support: 非负整数，真实类别等于当前类别的候选数。
    - predicted_count: 非负整数，六分类 argmax 结果等于当前类别的候选数。
    """

    precision: float
    recall: float
    f1: float
    support: int
    predicted_count: int


class SixClassMetrics(TypedDict):
    """描述全部候选的六分类报告。

    - accuracy: 浮点数，混淆矩阵对角线总和除以全部候选数；没有候选时是 NaN。
    - macro_f1: 浮点数，与既有 validation 选模兼容，等于 macro_f1_supported_classes。
    - macro_f1_supported_classes: 浮点数，只平均 support 大于零类别的 f1；没有支持类别时是 NaN。
    - macro_f1_all_classes: 浮点数，固定平均 CLASS_NAMES 中六个类别的 f1，零支持类别按 f1=0 参与平均。
    - per_class: 以 CLASS_NAMES 六个固定类别名为键的 PerClassMetrics 映射。
    - per_class[*].precision: 当前类别真阳性数除以预测数；零预测时是 NaN。
    - per_class[*].recall: 当前类别真阳性数除以真实支持数；零支持时是 NaN。
    - per_class[*].f1: 当前类别的有限调和平均；无法计算时取 0。
    - per_class[*].support: 当前类别真实候选数。
    - per_class[*].predicted_count: 当前类别预测候选数。
    - confusion_matrix: 形状 6×6 的整数嵌套列表，第一轴是真实类别，第二轴是 argmax 预测类别。
    """

    accuracy: float
    macro_f1: float
    macro_f1_supported_classes: float
    macro_f1_all_classes: float
    per_class: dict[str, PerClassMetrics]
    confusion_matrix: list[list[int]]


class IntrinsicIdentityMetrics(TypedDict):
    """描述一个身份范围内忽略六分类预测的固有身份排名。

    - top1: 浮点数，Top-1 正确数除以 eligible_count；分母为零时是 NaN。
    - topk: 浮点数，真实身份进入前 top_k 名的候选数除以 eligible_count；分母为零时是 NaN。
    - top_k: 正整数，报告采用的最大名次数；当有效槽少于该值时内部实际查询较少槽，但字段仍保留配置值。
    - eligible_count: 非负整数，真实类别属于当前范围且 identity_target 非负的候选数。
    """

    top1: float
    topk: float
    top_k: int
    eligible_count: int


class ConditionalIdentityMetrics(TypedDict):
    """描述六分类判对后的条件身份排名及训练可见性拆分。

    - top1: 浮点数，条件集合的身份 Top-1 正确率；eligible_count 为零时是 NaN。
    - topk: 浮点数，条件集合的身份 Top-K 正确率；eligible_count 为零时是 NaN。
    - top_k: 正整数，报告采用的最大名次数。
    - eligible_count: 非负整数，真实类别在配置身份范围、identity_target 非负且六分类预测等于真实类别的候选数。
    - seen_top1: 浮点数，identity_seen_in_train=True 子集的 Top-1 正确率；seen_count 为零时是 NaN。
    - seen_topk: 浮点数，identity_seen_in_train=True 子集的 Top-K 正确率；seen_count 为零时是 NaN。
    - seen_count: 非负整数，条件集合中联合身份曾作为训练正目标出现的候选数。
    - unseen_top1: 浮点数，identity_seen_in_train=False 子集的 Top-1 正确率；unseen_count 为零时是 NaN。
    - unseen_topk: 浮点数，identity_seen_in_train=False 子集的 Top-K 正确率；unseen_count 为零时是 NaN。
    - unseen_count: 非负整数，条件集合中联合身份未作为训练正目标出现的候选数。
    """

    top1: float
    topk: float
    top_k: int
    eligible_count: int
    seen_top1: float
    seen_topk: float
    seen_count: int
    unseen_top1: float
    unseen_topk: float
    unseen_count: int


class EndToEndIdentityMetrics(TypedDict):
    """描述背景或前景门控后最终身份 Top-1 的端到端计数和比率。

    - precision: 浮点数，true_positive 除以 predicted_positive；预测阳性为零时是 NaN。
    - recall: 浮点数，true_positive 除以 actual_positive；真实阳性为零时是 NaN。
    - f1: 浮点数，precision 与 recall 均有限且两者之和大于零时取调和平均，否则取 0。
    - true_positive: 非负整数，真实目标属于当前范围、预测通过相应前景门且 Top-1 身份正确的候选数。
    - predicted_positive: 非负整数，预测通过相应前景门、存在可评估身份槽且最终身份属于当前范围的候选数。
    - actual_positive: 非负整数，真实类别属于当前范围且 identity_target 非负的候选数。
    """

    precision: float
    recall: float
    f1: float
    true_positive: int
    predicted_positive: int
    actual_positive: int


class IdentityScopeMetrics(TypedDict):
    """组合一个身份范围的固有、条件和端到端指标。

    - intrinsic: 忽略六分类预测的 IntrinsicIdentityMetrics。
    - intrinsic.top1: 固有身份 Top-1 正确率。
    - intrinsic.topk: 固有身份 Top-K 正确率。
    - intrinsic.top_k: 报告采用的最大名次数。
    - intrinsic.eligible_count: 当前范围可监督身份候选数。
    - conditional: 仅在六分类预测等于真实类别时计算的 IntrinsicIdentityMetrics；该嵌套对象不含 seen 与 unseen 拆分。
    - conditional.top1: 条件身份 Top-1 正确率。
    - conditional.topk: 条件身份 Top-K 正确率。
    - conditional.top_k: 报告采用的最大名次数。
    - conditional.eligible_count: 当前范围六分类判对且可监督的候选数。
    - end_to_end: 经当前范围前景门计算的 EndToEndIdentityMetrics。
    - end_to_end.precision: 最终身份预测精确率。
    - end_to_end.recall: 最终身份预测召回率。
    - end_to_end.f1: 最终身份预测 F1。
    - end_to_end.true_positive: 最终身份真阳性数。
    - end_to_end.predicted_positive: 最终身份预测阳性数。
    - end_to_end.actual_positive: 最终身份真实阳性数。
    """

    intrinsic: IntrinsicIdentityMetrics
    conditional: IntrinsicIdentityMetrics
    end_to_end: EndToEndIdentityMetrics


class AreaMetrics(TypedDict):
    """描述 logit 0 阈值下跨候选、跨体素微平均的两项区域 Dice。

    - ligand_micro_dice: 浮点数，2×配体区域交集体素数除以 ligand_dice_denominator；分母为零时是 NaN。
    - binding_micro_dice: 浮点数，2×结合区域交集体素数除以 binding_dice_denominator；分母为零时是 NaN。
    - ligand_dice_denominator: 非负整数，配体区域预测为 True 的体素数与真实 True 体素数之和。
    - binding_dice_denominator: 非负整数，结合区域预测为 True 的体素数与真实 True 体素数之和。
    """

    ligand_micro_dice: float
    binding_micro_dice: float
    ligand_dice_denominator: int
    binding_dice_denominator: int


class EvaluationReport(TypedDict):
    """声明一次完整划分评估的顶层报告。

    所有比率均由跨批次整数计数统一计算。零分母保留为 Python NaN；write_json 等严格 JSON
    边界会把非有限浮点转换为 null，本类型自身不执行序列化。

    - six_class: 全部候选的 SixClassMetrics。
    - six_class.accuracy: 六分类总体准确率。
    - six_class.macro_f1: 与历史选模兼容的支持类 macro-F1。
    - six_class.macro_f1_supported_classes: 只平均真实支持类别的 macro-F1。
    - six_class.macro_f1_all_classes: 固定平均全部六类的 macro-F1。
    - six_class.per_class: 六个固定类别名到 PerClassMetrics 的映射。
    - six_class.per_class[*].precision: 当前类别精确率，零预测时为 NaN。
    - six_class.per_class[*].recall: 当前类别召回率，零支持时为 NaN。
    - six_class.per_class[*].f1: 当前类别 F1，无法计算时为 0。
    - six_class.per_class[*].support: 当前类别真实候选数。
    - six_class.per_class[*].predicted_count: 当前类别预测候选数。
    - six_class.confusion_matrix: 形状 6×6 的整数嵌套列表，第一轴真实、第二轴预测。
    - intrinsic_identity: 当前 identity_scope 配置范围的固有 IntrinsicIdentityMetrics；identity_scope=None 兼容解释为 small_molecule。
    - intrinsic_identity.top1: 配置范围固有 Top-1。
    - intrinsic_identity.topk: 配置范围固有 Top-K。
    - intrinsic_identity.top_k: 报告采用的最大名次数。
    - intrinsic_identity.eligible_count: 配置范围固有分母。
    - conditional_identity: 当前配置范围的 ConditionalIdentityMetrics，包含 seen 与 unseen 拆分。
    - conditional_identity.top1: 配置范围条件 Top-1。
    - conditional_identity.topk: 配置范围条件 Top-K。
    - conditional_identity.top_k: 报告采用的最大名次数。
    - conditional_identity.eligible_count: 配置范围条件分母。
    - conditional_identity.seen_top1: 训练可见联合身份子集 Top-1。
    - conditional_identity.seen_topk: 训练可见联合身份子集 Top-K。
    - conditional_identity.seen_count: 训练可见联合身份子集分母。
    - conditional_identity.unseen_top1: 训练未见联合身份子集 Top-1。
    - conditional_identity.unseen_topk: 训练未见联合身份子集 Top-K。
    - conditional_identity.unseen_count: 训练未见联合身份子集分母。
    - end_to_end_identity: 当前配置范围的 EndToEndIdentityMetrics。
    - end_to_end_identity.precision: 配置范围端到端精确率。
    - end_to_end_identity.recall: 配置范围端到端召回率。
    - end_to_end_identity.f1: 配置范围端到端 F1。
    - end_to_end_identity.true_positive: 配置范围端到端真阳性数。
    - end_to_end_identity.predicted_positive: 配置范围端到端预测阳性数。
    - end_to_end_identity.actual_positive: 配置范围端到端真实阳性数。
    - small_molecule_identity: 只允许 small_molecule 身份槽和严格 small_molecule 六分类门的 IdentityScopeMetrics。
    - small_molecule_identity.intrinsic: 小分子范围的固有排名报告。
    - small_molecule_identity.intrinsic.top1: 小分子范围固有 Top-1。
    - small_molecule_identity.intrinsic.topk: 小分子范围固有 Top-K。
    - small_molecule_identity.intrinsic.top_k: 报告采用的最大名次数。
    - small_molecule_identity.intrinsic.eligible_count: 小分子范围固有分母。
    - small_molecule_identity.conditional: 小分子范围且六分类判对后的排名报告。
    - small_molecule_identity.conditional.top1: 小分子范围条件 Top-1。
    - small_molecule_identity.conditional.topk: 小分子范围条件 Top-K。
    - small_molecule_identity.conditional.top_k: 报告采用的最大名次数。
    - small_molecule_identity.conditional.eligible_count: 小分子范围条件分母。
    - small_molecule_identity.end_to_end: 严格小分子分类门后的最终身份报告。
    - small_molecule_identity.end_to_end.precision: 小分子端到端精确率。
    - small_molecule_identity.end_to_end.recall: 小分子端到端召回率。
    - small_molecule_identity.end_to_end.f1: 小分子端到端 F1。
    - small_molecule_identity.end_to_end.true_positive: 小分子端到端真阳性数。
    - small_molecule_identity.end_to_end.predicted_positive: 小分子端到端预测阳性数。
    - small_molecule_identity.end_to_end.actual_positive: 小分子端到端真实阳性数。
    - all_non_background_identity: 在五类共享身份槽中排名、以任一非背景预测作为总体前景门的 IdentityScopeMetrics。
    - all_non_background_identity.intrinsic: 五类总体固有排名报告。
    - all_non_background_identity.intrinsic.top1: 五类总体固有 Top-1。
    - all_non_background_identity.intrinsic.topk: 五类总体固有 Top-K。
    - all_non_background_identity.intrinsic.top_k: 报告采用的最大名次数。
    - all_non_background_identity.intrinsic.eligible_count: 五类总体固有分母。
    - all_non_background_identity.conditional: 五类总体且六分类子类判对后的排名报告。
    - all_non_background_identity.conditional.top1: 五类总体条件 Top-1。
    - all_non_background_identity.conditional.topk: 五类总体条件 Top-K。
    - all_non_background_identity.conditional.top_k: 报告采用的最大名次数。
    - all_non_background_identity.conditional.eligible_count: 五类总体条件分母。
    - all_non_background_identity.end_to_end: 任一前景分类门后的最终身份报告。
    - all_non_background_identity.end_to_end.precision: 五类总体端到端精确率。
    - all_non_background_identity.end_to_end.recall: 五类总体端到端召回率。
    - all_non_background_identity.end_to_end.f1: 五类总体端到端 F1。
    - all_non_background_identity.end_to_end.true_positive: 五类总体端到端真阳性数。
    - all_non_background_identity.end_to_end.predicted_positive: 五类总体端到端预测阳性数。
    - all_non_background_identity.end_to_end.actual_positive: 五类总体端到端真实阳性数。
    - identity_per_class: 以五个 FOREGROUND_CLASS_NAMES 为键的 IdentityScopeMetrics；固有和条件分母按真实类别拆分，端到端预测阳性按最终 Top-1 身份槽所属类别归档。
    - identity_per_class[*].intrinsic: 当前真实前景类别的固有排名报告。
    - identity_per_class[*].intrinsic.top1: 当前类别固有 Top-1。
    - identity_per_class[*].intrinsic.topk: 当前类别固有 Top-K。
    - identity_per_class[*].intrinsic.top_k: 报告采用的最大名次数。
    - identity_per_class[*].intrinsic.eligible_count: 当前类别固有分母。
    - identity_per_class[*].conditional: 当前类别且六分类判对后的排名报告。
    - identity_per_class[*].conditional.top1: 当前类别条件 Top-1。
    - identity_per_class[*].conditional.topk: 当前类别条件 Top-K。
    - identity_per_class[*].conditional.top_k: 报告采用的最大名次数。
    - identity_per_class[*].conditional.eligible_count: 当前类别条件分母。
    - identity_per_class[*].end_to_end: 最终身份槽归入当前类别的端到端报告。
    - identity_per_class[*].end_to_end.precision: 当前类别端到端精确率。
    - identity_per_class[*].end_to_end.recall: 当前类别端到端召回率。
    - identity_per_class[*].end_to_end.f1: 当前类别端到端 F1。
    - identity_per_class[*].end_to_end.true_positive: 当前类别端到端真阳性数。
    - identity_per_class[*].end_to_end.predicted_positive: 当前类别端到端预测阳性数。
    - identity_per_class[*].end_to_end.actual_positive: 当前类别端到端真实阳性数。
    - area: 两个辅助区域的 AreaMetrics。
    - area.ligand_micro_dice: 配体区域微平均 Dice。
    - area.binding_micro_dice: 受体结合区域微平均 Dice。
    - area.ligand_dice_denominator: 配体区域预测体素数与真实体素数之和。
    - area.binding_dice_denominator: 结合区域预测体素数与真实体素数之和。
    """

    six_class: SixClassMetrics
    intrinsic_identity: IntrinsicIdentityMetrics
    conditional_identity: ConditionalIdentityMetrics
    end_to_end_identity: EndToEndIdentityMetrics
    small_molecule_identity: IdentityScopeMetrics
    all_non_background_identity: IdentityScopeMetrics
    identity_per_class: dict[str, IdentityScopeMetrics]
    area: AreaMetrics


def _safe_divide(numerator: float, denominator: float) -> float:
    """只在分母为正时执行除法，否则返回 NaN。

    - numerator: 正式调用传入的非负 Python int 或 float 计数；本函数不要求调用方预先转换成 float。
    - denominator: 正式调用传入的非负 Python int 或 float 计数；只有大于 0 时才作为分母。
    - 返回值: numerator/denominator 的结果，或 denominator 不为正时的 math.nan。

    本函数不检查 numerator 是否有限，也不把 NaN 转为 JSON null。
    """

    return numerator / denominator if denominator > 0 else math.nan


def _require_finite(values: torch.Tensor, name: str) -> None:
    """在 argmax 或阈值运算前拒绝含 NaN 或正负 Inf 的张量。

    - values: 任意形状的 Torch 张量；torch.isfinite 必须支持其 dtype。
    - name: 写入 FloatingPointError 的字段名称，不参与数值判断。
    - 返回值: None；全部元素有限时不修改 values。
    """

    if not torch.isfinite(values).all():
        raise FloatingPointError(f"{name} 含 NaN 或 Inf。")


@dataclass
class _IdentityCounts:
    """保存一个身份范围跨物理批次可加的排名与端到端整数计数。

    - intrinsic_correct_at_1: 固有分母中 Top-1 身份正确的累计候选数。
    - intrinsic_correct_at_k: 固有分母中真实身份进入前 top_k 名的累计候选数。
    - intrinsic_eligible: 真实类别属于当前范围且 identity_target 非负的累计候选数。
    - conditional_correct_at_1: 条件分母中 Top-1 身份正确的累计候选数。
    - conditional_correct_at_k: 条件分母中真实身份进入前 top_k 名的累计候选数。
    - conditional_eligible: 固有分母中六分类预测同时等于真实类别的累计候选数。
    - end_to_end_true_positive: 通过当前范围预测门且 Top-1 身份正确的累计候选数。
    - end_to_end_predicted_positive: 通过当前范围预测门并产生范围内最终身份的累计候选数。
    - end_to_end_actual_positive: 与 intrinsic_eligible 同口径的累计真实阳性数。

    本私有容器依赖调用者传入同形布尔数组，不单独验证 ndim、dtype 或数组长度。
    """

    intrinsic_correct_at_1: int = 0
    intrinsic_correct_at_k: int = 0
    intrinsic_eligible: int = 0
    conditional_correct_at_1: int = 0
    conditional_correct_at_k: int = 0
    conditional_eligible: int = 0
    end_to_end_true_positive: int = 0
    end_to_end_predicted_positive: int = 0
    end_to_end_actual_positive: int = 0

    def update(
        self,
        intrinsic_eligible: np.ndarray,
        conditional_eligible: np.ndarray,
        top1_correct: np.ndarray,
        top_k_correct: np.ndarray,
        end_to_end_correct: np.ndarray,
        predicted_positive: np.ndarray,
    ) -> None:
        """把一个物理批次的身份布尔判定累加到当前范围。

        - self: 当前身份范围的跨批次整数计数器。
        - intrinsic_eligible: NumPy bool 数组，形状 (B,)，True 表示候选进入固有身份分母。
        - conditional_eligible: NumPy bool 数组，形状 (B,)，True 表示候选同时进入条件身份分母。
        - top1_correct: NumPy bool 数组，形状 (B,)，True 表示范围内 Top-1 身份槽等于 identity_target。
        - top_k_correct: NumPy bool 数组，形状 (B,)，True 表示 identity_target 位于范围内前 top_k 名。
        - end_to_end_correct: NumPy bool 数组，形状 (B,)，True 表示端到端预测通过门且身份正确。
        - predicted_positive: NumPy bool 数组，形状 (B,)，True 表示端到端预测落入当前身份范围。
        - 返回值: None；九个 self 计数原地增加。

        本方法不验证六个数组同形或 dtype；正式调用由 EvaluationAccumulator._update_identity
        构造这些布尔数组。
        """

        self.intrinsic_eligible += int(intrinsic_eligible.sum())
        self.intrinsic_correct_at_1 += int((intrinsic_eligible & top1_correct).sum())
        self.intrinsic_correct_at_k += int((intrinsic_eligible & top_k_correct).sum())
        self.conditional_eligible += int(conditional_eligible.sum())
        self.conditional_correct_at_1 += int((conditional_eligible & top1_correct).sum())
        self.conditional_correct_at_k += int((conditional_eligible & top_k_correct).sum())
        self.end_to_end_true_positive += int(end_to_end_correct.sum())
        self.end_to_end_predicted_positive += int(predicted_positive.sum())
        self.end_to_end_actual_positive += int(intrinsic_eligible.sum())

    def compute(self, top_k: int) -> IdentityScopeMetrics:
        """从当前整数计数计算一个身份范围报告。

        - self: 已累计完整划分计数的 _IdentityCounts。
        - top_k: 写入 intrinsic.top_k 与 conditional.top_k 的配置名次数；本方法不再次验证正值。
        - 返回值.intrinsic: 忽略六分类结果的排名报告。
        - 返回值.intrinsic.top1: intrinsic_correct_at_1/intrinsic_eligible，零分母时为 NaN。
        - 返回值.intrinsic.topk: intrinsic_correct_at_k/intrinsic_eligible，零分母时为 NaN。
        - 返回值.intrinsic.top_k: 原样保存输入 top_k。
        - 返回值.intrinsic.eligible_count: intrinsic_eligible 的整数值。
        - 返回值.conditional: 六分类判对后的排名报告。
        - 返回值.conditional.top1: conditional_correct_at_1/conditional_eligible，零分母时为 NaN。
        - 返回值.conditional.topk: conditional_correct_at_k/conditional_eligible，零分母时为 NaN。
        - 返回值.conditional.top_k: 原样保存输入 top_k。
        - 返回值.conditional.eligible_count: conditional_eligible 的整数值。
        - 返回值.end_to_end: 当前范围的最终身份报告。
        - 返回值.end_to_end.precision: end_to_end_true_positive/end_to_end_predicted_positive，零分母时为 NaN。
        - 返回值.end_to_end.recall: end_to_end_true_positive/end_to_end_actual_positive，零分母时为 NaN。
        - 返回值.end_to_end.f1: 有限 precision 与 recall 的调和平均；未定义或两者之和为零时取 0。
        - 返回值.end_to_end.true_positive: 累计端到端真阳性数。
        - 返回值.end_to_end.predicted_positive: 累计端到端预测阳性数。
        - 返回值.end_to_end.actual_positive: 累计端到端真实阳性数。
        """

        # precision/recall 是当前范围端到端计数的最终比率；零分母保留为 NaN。
        precision = _safe_divide(
            self.end_to_end_true_positive,
            self.end_to_end_predicted_positive,
        )
        recall = _safe_divide(
            self.end_to_end_true_positive,
            self.end_to_end_actual_positive,
        )
        # F1 对未定义比率或全零比率显式回退为 0，避免 NaN 继续参与 selection score。
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if math.isfinite(precision)
            and math.isfinite(recall)
            and precision + recall > 0
            else 0.0
        )
        return {
            "intrinsic": {
                "top1": _safe_divide(self.intrinsic_correct_at_1, self.intrinsic_eligible),
                "topk": _safe_divide(self.intrinsic_correct_at_k, self.intrinsic_eligible),
                "top_k": top_k,
                "eligible_count": self.intrinsic_eligible,
            },
            "conditional": {
                "top1": _safe_divide(
                    self.conditional_correct_at_1,
                    self.conditional_eligible,
                ),
                "topk": _safe_divide(
                    self.conditional_correct_at_k,
                    self.conditional_eligible,
                ),
                "top_k": top_k,
                "eligible_count": self.conditional_eligible,
            },
            "end_to_end": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positive": self.end_to_end_true_positive,
                "predicted_positive": self.end_to_end_predicted_positive,
                "actual_positive": self.end_to_end_actual_positive,
            },
        }


def _masked_identity_prediction(
    scores: torch.Tensor,
    slot_mask: torch.Tensor,
    target: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """在指定身份槽子集内计算 Top-1 下标与真实槽的 Top-K 命中标志。

    - scores: 浮点 Torch 张量，形状 (B,I)，保存同一 PDB 身份槽的模型分数；调用者已核验有限。
    - slot_mask: Torch bool 张量，形状 (B,I)，True 表示本次排名允许使用的身份槽。
    - target: NumPy int64 数组，形状 (B,)，保存真实身份槽下标或 -1。
    - top_k: 正整数，期望查询的最大名次数；实际查询数为 min(top_k,I)。
    - 返回值[0]: NumPy int64 数组，形状 (B,)，保存允许槽中的 Top-1 原身份轴下标；没有允许槽时为 -1。
    - 返回值[1]: NumPy bool 数组，形状 (B,)，表示 target 是否进入实际查询的前若干名；没有允许槽时为 False。

    本函数不独立检查输入形状、dtype、target 范围或 top_k；正式调用由
    EvaluationAccumulator 的门禁和构造参数保证。identity 轴 I 为零时不会调用 argmax。
    """

    # 先建立 I=0 与无允许槽都可返回的哨兵数组，避免把补零槽解释为身份预测。
    prediction = np.full(target.shape, -1, dtype=np.int64)
    top_k_correct = np.zeros(target.shape, dtype=bool)
    if scores.shape[1] == 0:
        return prediction, top_k_correct
    # masked 保留形状 (B,I)；不允许槽被替换为当前浮点 dtype 的最小有限值。
    masked = scores.masked_fill(~slot_mask, torch.finfo(scores.dtype).min).detach().cpu()
    # prediction 保存全身份轴下标，不会把范围内槽重新编号。
    prediction = masked.argmax(dim=-1).numpy()
    # count 是模型身份轴与配置 top_k 的较小值；正式 top_k 为正，I>0 分支中 count 也为正。
    count = min(top_k, masked.shape[1])
    # top_indices 是 NumPy int64 (B,count)，每项仍引用原身份轴。
    top_indices = masked.topk(count, dim=-1).indices.numpy()
    top_k_correct = np.any(top_indices == target[:, None], axis=1)
    # has_slot 是 NumPy bool (B,)；全 False 的样本覆盖 argmax/topk 对最小值并列的任意选择。
    has_slot = slot_mask.any(dim=1).detach().cpu().numpy()
    prediction[~has_slot] = -1
    top_k_correct[~has_slot] = False
    return prediction, top_k_correct


def _scope_masks(
    class_prediction: np.ndarray,
    class_target: np.ndarray,
    target: np.ndarray,
    slot_options: np.ndarray,
    allowed_classes: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """为一个身份范围构造固有、条件和端到端布尔门。

    - class_prediction: NumPy整数数组，形状 (B,)，保存六分类 argmax 下标。
    - class_target: NumPy整数数组，形状 (B,)，保存六分类真实下标。
    - target: NumPy整数数组，形状 (B,)，保存真实身份槽下标或 -1。
    - slot_options: NumPy bool 数组，形状 (B,)，表示当前范围至少存在一个身份槽。
    - allowed_classes: 非空类别下标元组，定义当前身份范围的真实和预测类别集合。
    - 返回值[0]: NumPy bool (B,)，真实类别在范围内且 target 非负的固有分母。
    - 返回值[1]: NumPy bool (B,)，固有分母中六分类预测精确等于真实类别的条件分母。
    - 返回值[2]: NumPy bool (B,)，六分类预测属于 allowed_classes 的前景门。
    - 返回值[3]: NumPy bool (B,)，当前范围有槽，且不会把真实在范围内但 target=-1 的候选计为预测阳性。

    对真实背景候选，只要当前 PDB 存在范围内身份槽，返回值[3] 仍为 True，使错误前景预测
    可以进入端到端 predicted_positive。函数不独立验证数组形状或类别范围。
    """

    # true_in_scope/predicted_in_scope 是 NumPy bool (B,)，分别描述真实与预测类别范围。
    true_in_scope = np.isin(class_target, allowed_classes)
    predicted_in_scope = np.isin(class_prediction, allowed_classes)
    # intrinsic/conditional 是身份排名的两个逐步收紧分母。
    intrinsic = true_in_scope & (target >= 0)
    conditional = intrinsic & (class_prediction == class_target)
    # 当前统计范围内缺少联合身份监督的真实类别不作为预测阳性；范围外前景和背景仍可成为假阳性。
    end_to_end_scope = slot_options & ~(true_in_scope & (target < 0))
    return intrinsic, conditional, predicted_in_scope, end_to_end_scope


# ================================================================================================


@dataclass
class EvaluationAccumulator:
    """把多个物理批次合并为不依赖批次切分的 EvaluationReport 计数。

    - identity_scope: None、small_molecule 或 all_non_background；决定三个兼容顶层身份字段采用小分子范围还是五类总体范围，None 按历史兼容解释为 small_molecule。
    - class_confusion: NumPy int64 数组，形状 (6,6)，第一轴是真实类别，第二轴是六分类 argmax 预测类别。
    - small_identity: _IdentityCounts，只在 small_molecule 身份槽中排名并使用严格 small_molecule 分类门。
    - all_identity: _IdentityCounts，在五类共享身份槽中排名并使用任一非背景类别作为总体前景门。
    - per_class_identity: 整数类别下标到 _IdentityCounts 的映射；默认恰含五个 FOREGROUND_CLASS_INDICES。
    - seen_correct_at_1: 配置范围条件分母中 identity_seen_in_train=True 且 Top-1 正确的累计候选数。
    - seen_correct_at_k: 配置范围条件分母中 identity_seen_in_train=True 且命中 Top-K 的累计候选数。
    - seen_eligible: 配置范围条件分母中 identity_seen_in_train=True 的累计候选数。
    - unseen_correct_at_1: 配置范围条件分母中 identity_seen_in_train=False 且 Top-1 正确的累计候选数。
    - unseen_correct_at_k: 配置范围条件分母中 identity_seen_in_train=False 且命中 Top-K 的累计候选数。
    - unseen_eligible: 配置范围条件分母中 identity_seen_in_train=False 的累计候选数。
    - ligand_intersection: logit 大于等于 0 的配体区域预测与真值的累计交集体素数。
    - ligand_predicted: logit 大于等于 0 的累计配体预测体素数。
    - ligand_actual: 配体区域真值中的累计 True 体素数。
    - binding_intersection: logit 大于等于 0 的结合区域预测与真值的累计交集体素数。
    - binding_predicted: logit 大于等于 0 的累计结合区域预测体素数。
    - binding_actual: 结合区域真值中的累计 True 体素数。
    - top_k: API 契约为正整数，默认 5；内部排名实际取该值与身份轴宽度的较小值。

    构造后门禁只检查 top_k > 0、identity_scope 的允许值和 class_confusion 形状，不核验
    top_k 的精确类型，也不核验自定义混淆矩阵的 dtype、非负性或其它计数字段的一致性。
    """

    identity_scope: IdentityScope | None
    class_confusion: np.ndarray = field(
        default_factory=lambda: np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    )
    small_identity: _IdentityCounts = field(default_factory=_IdentityCounts)
    all_identity: _IdentityCounts = field(default_factory=_IdentityCounts)
    per_class_identity: dict[int, _IdentityCounts] = field(
        default_factory=lambda: {
            class_index: _IdentityCounts() for class_index in FOREGROUND_CLASS_INDICES
        }
    )
    seen_correct_at_1: int = 0
    seen_correct_at_k: int = 0
    seen_eligible: int = 0
    unseen_correct_at_1: int = 0
    unseen_correct_at_k: int = 0
    unseen_eligible: int = 0
    ligand_intersection: int = 0
    ligand_predicted: int = 0
    ligand_actual: int = 0
    binding_intersection: int = 0
    binding_predicted: int = 0
    binding_actual: int = 0
    top_k: int = 5

    def __post_init__(self) -> None:
        """检查构造时会直接破坏排名或混淆矩阵索引的边界。

        - self: 已由 dataclass 赋值但尚未用于累计的 EvaluationAccumulator。
        - 返回值: None；通过检查时不重建或复制任何字段。

        本方法对 top_k 只执行大于零比较，不执行 type(...) is int；正式配置和默认值提供整数。
        class_confusion 只检查形状，不检查 dtype、数值范围或是否由独立累加器专有。
        """

        if self.top_k <= 0:
            raise ValueError("top_k 必须为正整数。")
        if self.identity_scope not in (None, "small_molecule", "all_non_background"):
            raise ValueError("identity_scope 必须为 None、small_molecule 或 all_non_background。")
        # expected 是六分类混淆矩阵的固定形状 (6,6)。
        expected = (len(CLASS_NAMES), len(CLASS_NAMES))
        if self.class_confusion.shape != expected:
            raise ValueError(f"class_confusion 必须为 {expected}。")

    def update(
        self,
        outputs: Mapping[str, torch.Tensor],
        batch: EvaluationBatch,
    ) -> None:
        """从一个物理批次更新六分类、区域和可选身份计数。

        - self: 保存此前物理批次累计计数的 EvaluationAccumulator。
        - outputs: 模型输出张量映射；额外键被忽略。
        - outputs.class_logits: 正式模型产生的有限浮点张量，形状 (B,6)；本方法检查形状和有限性但不显式检查 dtype。
        - outputs.ligand_area_logits: 有限张量，形状 (B,Z,Y,X)，以大于等于 0 作为配体区域预测；必须存在。
        - outputs.binding_area_logits: 有限张量，形状 (B,Z,Y,X)，以大于等于 0 作为结合区域预测；必须存在。
        - outputs.identity_logits: 可选有限浮点张量，形状 (B,I)；存在时才核验并累计全部身份字段。
        - batch: 与 outputs 沿候选轴 B 对齐的 EvaluationBatch。
        - batch.class_target: Torch int64 (B,)，每项位于 [0,6)。
        - batch.ligand_area_target: Torch bool (B,Z,Y,X)，与 ligand_area_logits 同形。
        - batch.binding_area_target: Torch bool (B,Z,Y,X)，与 binding_area_logits 同形。
        - batch.identity_target: Torch int64 (B,)，每项为 -1 或 [0,I)；只在 outputs.identity_logits 存在时读取。
        - batch.identity_mask: Torch bool (B,I)，有效身份槽掩码；只在 outputs.identity_logits 存在时读取。
        - batch.identity_class_index: Torch int64 (B,I)，有效槽为五类前景下标、补零槽为 -1；只在 outputs.identity_logits 存在时读取。
        - batch.metadata: 长度 B 的 CandidateMetadata 列表；身份路径存在时按配置条件分母读取 identity_seen_in_train。
        - batch.metadata[*].split: 物化划分名；本方法不读取。
        - batch.metadata[*].pdb_id: PDB 标识；本方法不读取。
        - batch.metadata[*].candidate_index: PDB 内候选编号；本方法不读取。
        - batch.metadata[*].matched_object_key: 命中模板键或 None；本方法不读取。
        - batch.metadata[*].matched_prepared_smiles: 顶层 PreparedLigand.smiles 或 None；本方法不读取。
        - batch.metadata[*].matched_prepared_smiles_status: 顶层字符串状态；本方法不读取。
        - batch.metadata[*].identity_seen_in_train: 配置条件身份候选必须为严格 bool；其它候选可为 None。
        - 返回值: None；self 的混淆矩阵和相应计数原地增加。

        class_logits、两个区域键缺失时由映射索引抛出 KeyError。identity_logits 缺失表示候选
        预训练输出，身份 batch 字段不会在本次调用中验证。若一次调用在后续区域或身份门禁失败，
        此前已经增加的混淆矩阵或区域计数不会回滚，因此调用方必须传入正式 Dataset 契约。
        """

        if outputs["class_logits"].ndim != 2 or outputs["class_logits"].shape[1] != len(
            CLASS_NAMES
        ):
            raise ValueError("class_logits 必须为 (B,6)。")
        _require_finite(outputs["class_logits"], "class_logits")
        if batch["class_target"].shape != (outputs["class_logits"].shape[0],):
            raise ValueError("class_target 必须与 class_logits 的批次轴对齐。")
        if batch["class_target"].dtype != torch.long:
            raise TypeError("class_target 必须是 int64。")
        if torch.any(batch["class_target"] < 0) or torch.any(
            batch["class_target"] >= len(CLASS_NAMES)
        ):
            raise ValueError("class_target 超出六分类下标范围。")
        if len(batch["metadata"]) != outputs["class_logits"].shape[0]:
            raise ValueError("metadata 长度必须与批次轴 B 一致。")
        # batch_size 是物理批次候选数 B，后续两个区域输出必须复用该批次轴。
        batch_size = outputs["class_logits"].shape[0]
        # class_prediction/class_target 是 NumPy 整数数组 (B,)，供混淆矩阵与身份范围布尔运算。
        class_prediction = outputs["class_logits"].argmax(dim=-1).detach().cpu().numpy()
        class_target = batch["class_target"].detach().cpu().numpy()
        np.add.at(self.class_confusion, (class_target, class_prediction), 1)
        self._update_area(
            outputs["ligand_area_logits"],
            batch["ligand_area_target"],
            "ligand",
            batch_size,
        )
        self._update_area(
            outputs["binding_area_logits"],
            batch["binding_area_target"],
            "binding",
            batch_size,
        )
        if "identity_logits" in outputs:
            self._update_identity(outputs["identity_logits"], class_prediction, batch)

    def _update_identity(
        self,
        identity_logits: torch.Tensor,
        class_prediction: np.ndarray,
        batch: EvaluationBatch,
    ) -> None:
        """核验身份轴并更新固有、条件、端到端和 seen/unseen 计数。

        - self: 已累计此前批次的 EvaluationAccumulator。
        - identity_logits: 正式模型产生的有限浮点 Torch 张量，形状 (B,I)，I 可为零。
        - class_prediction: NumPy整数数组，形状 (B,)，由同批 class_logits.argmax 得到。
        - batch: 含身份目标、槽掩码、槽类别和 metadata 的 EvaluationBatch。
        - batch.class_target: Torch int64 (B,)，保存六分类真实类别。
        - batch.identity_target: Torch int64 (B,)，-1 表示无监督联合身份，否则指向 [0,I) 的有效槽。
        - batch.identity_mask: Torch bool (B,I)，True 为真实身份槽，False 为补零槽。
        - batch.identity_class_index: Torch int64 (B,I)，有效槽必须为 1 至 5，补零槽必须为 -1。
        - batch.metadata: 长度 B 的 CandidateMetadata；此前 update 已核验长度。
        - batch.metadata[*].identity_seen_in_train: 进入配置条件身份分母时必须为严格 bool，并与其它条件候选共同穷尽 seen/unseen。
        - 返回值: None；self 的三个范围计数器与 seen/unseen 计数原地增加。

        本方法核验目标槽有效且槽类别等于真实类别。它不重新读取联合身份字符串，也不按
        object_key 或 smiles 建槽；身份轴已经由 CandidateDataset 和 collate_candidates 冻结。
        若后段 metadata 门禁失败，本次调用此前增加的身份计数不会回滚。
        """

        if identity_logits.ndim != 2 or identity_logits.shape != batch["identity_mask"].shape:
            raise ValueError("identity_logits 和 identity_mask 必须是同形 (B,I) 张量。")
        if identity_logits.shape[0] != class_prediction.shape[0]:
            raise ValueError("identity_logits 的批次轴必须与 class_logits 对齐。")
        _require_finite(identity_logits, "identity_logits")
        if batch["identity_mask"].dtype != torch.bool:
            raise TypeError("identity_mask 必须是 bool。")
        if batch["identity_target"].dtype != torch.long or batch["identity_target"].shape != (
            identity_logits.shape[0],
        ):
            raise ValueError("identity_target 必须是与批次轴对齐的 int64 (B,) 张量。")
        # target_tensor 是 Torch int64 (B,)，-1 与非负身份槽下标在下方分开处理。
        target_tensor = batch["identity_target"]
        if torch.any(target_tensor < -1) or torch.any(target_tensor >= identity_logits.shape[1]):
            raise ValueError("identity_target 必须为 -1 或身份轴 [0,I) 的下标。")
        # target_available 是 Torch bool (B,)，只标记具备完整可用联合身份监督的候选。
        target_available = target_tensor >= 0
        if torch.any(target_available) and not torch.all(
            batch["identity_mask"][target_available, target_tensor[target_available]]
        ):
            raise ValueError("非负 identity_target 必须指向 identity_mask 的真值槽。")
        # identity_class_index 是 Torch int64 (B,I)，把每个有效身份槽映射到五类前景下标。
        identity_class_index = batch["identity_class_index"]
        if identity_class_index.dtype != torch.long or identity_class_index.shape != (
            identity_logits.shape
        ):
            raise ValueError("identity_class_index 必须是与身份轴对齐的 int64 (B,I) 张量。")
        if torch.any(identity_class_index[batch["identity_mask"]] <= 0) or torch.any(
            identity_class_index[batch["identity_mask"]] >= len(CLASS_NAMES)
        ):
            raise ValueError("有效身份必须属于五类非背景类别。")
        if torch.any(identity_class_index[~batch["identity_mask"]] != -1):
            raise ValueError("补零身份的 identity_class_index 必须为 -1。")
        if torch.any(target_available) and not torch.equal(
            identity_class_index[target_available, target_tensor[target_available]],
            batch["class_target"][target_available],
        ):
            raise ValueError("identity_target 的身份类别必须等于候选真实类别。")
        # target/class_target 是 NumPy整数数组 (B,)，供后续范围掩码与下标比较使用。
        target = batch["identity_target"].detach().cpu().numpy()
        class_target = batch["class_target"].detach().cpu().numpy()
        # full_prediction/full_top_k 在全部有效身份槽中排名，形状均为 (B,)。
        full_prediction, full_top_k = _masked_identity_prediction(
            identity_logits,
            batch["identity_mask"],
            target,
            self.top_k,
        )
        # full_top1 是 NumPy bool (B,)，无目标候选不会因 -1 预测哨兵而被纳入任何固有分母。
        full_top1 = full_prediction == target
        # small_slot_mask 是 Torch bool (B,I)，只保留 class_index=small_molecule 的有效槽。
        small_slot_mask = batch["identity_mask"] & (
            identity_class_index == SMALL_MOLECULE_INDEX
        )
        # small_prediction/small_top_k 在小分子子清单中排名，仍返回原身份轴下标。
        small_prediction, small_top_k = _masked_identity_prediction(
            identity_logits,
            small_slot_mask,
            target,
            self.top_k,
        )
        # small_top1 是 NumPy bool (B,)，供兼容小分子固有、条件和端到端报告共用。
        small_top1 = small_prediction == target

        # small_slot_options 是 NumPy bool (B,)，表示当前 PDB 身份轴至少含一个小分子槽。
        small_slot_options = small_slot_mask.any(dim=1).detach().cpu().numpy()
        # small_masks 依次是固有分母、精确分类条件分母、小分子预测门和可评估槽门。
        small_masks = _scope_masks(
            class_prediction,
            class_target,
            target,
            small_slot_options,
            (SMALL_MOLECULE_INDEX,),
        )
        # 小分子端到端要求六分类明确预测 small_molecule，不接受其它前景子类。
        small_predicted_positive = small_masks[2] & small_masks[3]
        small_end_to_end_correct = small_masks[0] & small_predicted_positive & small_top1
        self.small_identity.update(
            *small_masks[:2],
            small_top1,
            small_top_k,
            small_end_to_end_correct,
            small_predicted_positive,
        )

        # full_slot_options 是 NumPy bool (B,)，表示当前 PDB 身份轴至少含一个五类前景槽。
        full_slot_options = batch["identity_mask"].any(dim=1).detach().cpu().numpy()
        # all_masks 依次是五类固有分母、精确子类条件分母、任一前景预测门和可评估槽门。
        all_masks = _scope_masks(
            class_prediction,
            class_target,
            target,
            full_slot_options,
            FOREGROUND_CLASS_INDICES,
        )
        # 五类总体端到端只把六分类当作背景/前景门，不要求预测前景子类等于真实子类。
        all_predicted_positive = all_masks[2] & all_masks[3]
        all_end_to_end_correct = all_masks[0] & all_predicted_positive & full_top1
        self.all_identity.update(
            *all_masks[:2],
            full_top1,
            full_top_k,
            all_end_to_end_correct,
            all_predicted_positive,
        )

        # identity_class_values 是 NumPy int64 (B,I)，用于把最终 Top-1 身份归入其前景类别。
        identity_class_values = identity_class_index.detach().cpu().numpy()
        # predicted_identity_class 是 NumPy int64 (B,)；-1 表示没有任何有效身份槽。
        predicted_identity_class = np.full(target.shape, -1, dtype=np.int64)
        # valid_prediction 是 NumPy bool (B,)，只允许非负 Top-1 下标参与二维索引。
        valid_prediction = full_prediction >= 0
        if np.any(valid_prediction):
            # row_indices 与 valid_prediction 中的候选一一对齐，用于成对索引 (B,I) 槽类别。
            row_indices = np.arange(target.shape[0])[valid_prediction]
            predicted_identity_class[valid_prediction] = identity_class_values[
                row_indices,
                full_prediction[valid_prediction],
            ]
        for class_index in FOREGROUND_CLASS_INDICES:
            # class_slot_options 是 NumPy bool (B,)，表示当前 PDB 至少含一个该类别身份槽。
            class_slot_options = (
                batch["identity_mask"] & (identity_class_index == class_index)
            ).any(dim=1).detach().cpu().numpy()
            # class_masks 的固有和条件分母按真实子类收窄，端到端槽门仍保护缺监督前景。
            class_masks = _scope_masks(
                class_prediction,
                class_target,
                target,
                class_slot_options,
                (class_index,),
            )
            # 逐类预测阳性按最终身份槽所属类别归档；六分类只需通过任一前景门。
            class_predicted_positive = (
                all_masks[2]
                & all_masks[3]
                & (predicted_identity_class == class_index)
            )
            class_end_to_end_correct = (
                class_masks[0] & class_predicted_positive & full_top1
            )
            self.per_class_identity[class_index].update(
                *class_masks[:2],
                full_top1,
                full_top_k,
                class_end_to_end_correct,
                class_predicted_positive,
            )

        # configured_counts 选择兼容顶层身份报告的累计器；None 保持历史小分子口径。
        configured_counts = (
            self.small_identity
            if self.identity_scope in (None, "small_molecule")
            else self.all_identity
        )
        # 以下三个 NumPy bool (B,) 与 configured_counts 使用同一身份范围。
        configured_conditional = (
            small_masks[1]
            if self.identity_scope in (None, "small_molecule")
            else all_masks[1]
        )
        configured_top1 = (
            small_top1
            if self.identity_scope in (None, "small_molecule")
            else full_top1
        )
        configured_top_k = (
            small_top_k
            if self.identity_scope in (None, "small_molecule")
            else full_top_k
        )

        for index, metadata in enumerate(batch["metadata"]):
            if not configured_conditional[index]:
                continue
            if type(metadata["identity_seen_in_train"]) is not bool:
                raise ValueError("条件身份样本的 identity_seen_in_train 必须是布尔值。")
            if metadata["identity_seen_in_train"] is True:
                self.seen_eligible += 1
                self.seen_correct_at_1 += int(configured_top1[index])
                self.seen_correct_at_k += int(configured_top_k[index])
            elif metadata["identity_seen_in_train"] is False:
                self.unseen_eligible += 1
                self.unseen_correct_at_1 += int(configured_top1[index])
                self.unseen_correct_at_k += int(configured_top_k[index])
        # 跨批次累计的条件分母必须被 seen 与 unseen 两个严格布尔分支完整划分。
        if configured_counts.conditional_eligible != self.seen_eligible + self.unseen_eligible:
            raise ValueError("条件身份样本必须完整划分为 seen 或 unseen。")

    def _update_area(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        prefix: Literal["ligand", "binding"],
        expected_batch_size: int,
    ) -> None:
        """以 logit 0 为阈值累计一个区域头的微平均 Dice 整数计数。

        - self: 已累计此前物理批次区域计数的 EvaluationAccumulator。
        - logits: 有限 Torch 张量，形状 (B,Z,Y,X)；大于等于 0 的体素预测为 True。
        - target: Torch bool 张量，形状与 logits 相同；True 表示区域真值体素。
        - prefix: 正式调用为 ligand 或 binding，决定增加哪组三个计数。
        - expected_batch_size: class_logits 的批次候选数 B；必须等于 logits.shape[0]。
        - 返回值: None；对应 prefix 的交集、预测与真实体素计数原地增加。

        本方法只按 Literal 契约区分 prefix；运行时代码没有显式拒绝其它字符串，非 ligand 值
        会进入 binding 分支。它不按候选先算 Dice，而是连同全空真负候选一起汇总所有体素。
        """

        if logits.ndim != 4 or logits.shape != target.shape:
            raise ValueError(f"{prefix} 区域 logits/target 必须为同形 (B,Z,Y,X)。")
        if logits.shape[0] != expected_batch_size:
            raise ValueError(f"{prefix} 区域 logits 的批次轴必须与 class_logits 对齐。")
        _require_finite(logits, f"{prefix}_area_logits")
        if target.dtype != torch.bool:
            raise TypeError(f"{prefix} 区域 target 必须是 bool。")
        # prediction/target_bool 是 Torch bool (B,Z,Y,X)，只用于离线整数计数，不保留梯度。
        prediction = logits.detach() >= 0
        target_bool = target.detach().bool()
        # 三个 Python int 汇总本物理批次全部候选和体素，不受批次内样本顺序影响。
        intersection = int((prediction & target_bool).sum().item())
        predicted = int(prediction.sum().item())
        actual = int(target_bool.sum().item())
        if prefix == "ligand":
            self.ligand_intersection += intersection
            self.ligand_predicted += predicted
            self.ligand_actual += actual
        else:
            self.binding_intersection += intersection
            self.binding_predicted += predicted
            self.binding_actual += actual

    def compute(self) -> EvaluationReport:
        """从累计整数计数构造完整 EvaluationReport。

        - self: 已经消费零个或多个物理批次的 EvaluationAccumulator。
        - 返回值.six_class: 全候选六分类报告。
        - 返回值.six_class.accuracy: 对角线候选数除以全部候选数；空累计器为 NaN。
        - 返回值.six_class.macro_f1: 等于 macro_f1_supported_classes。
        - 返回值.six_class.macro_f1_supported_classes: 只平均 support>0 类别的 F1；无支持类别时为 NaN。
        - 返回值.six_class.macro_f1_all_classes: 固定平均六类 F1，零支持类别按 0 计入。
        - 返回值.six_class.per_class: 以 CLASS_NAMES 为键的 PerClassMetrics 映射。
        - 返回值.six_class.per_class[*].precision: 当前类真阳性除以预测数；零预测时为 NaN。
        - 返回值.six_class.per_class[*].recall: 当前类真阳性除以支持数；零支持时为 NaN。
        - 返回值.six_class.per_class[*].f1: 有限 precision/recall 的调和平均，否则为 0。
        - 返回值.six_class.per_class[*].support: 当前类真实候选数。
        - 返回值.six_class.per_class[*].predicted_count: 当前类预测候选数。
        - 返回值.six_class.confusion_matrix: 形状 6×6 的整数嵌套列表，第一轴真实、第二轴预测。
        - 返回值.intrinsic_identity: identity_scope 选择范围的固有身份排名；None 选择 small_molecule。
        - 返回值.intrinsic_identity.top1: 配置范围固有身份 Top-1 正确率。
        - 返回值.intrinsic_identity.topk: 配置范围固有身份 Top-K 正确率。
        - 返回值.intrinsic_identity.top_k: self.top_k。
        - 返回值.intrinsic_identity.eligible_count: 配置范围固有身份分母。
        - 返回值.conditional_identity: 配置范围六分类判对后的身份排名及 seen/unseen 拆分。
        - 返回值.conditional_identity.top1: 配置范围条件身份 Top-1 正确率。
        - 返回值.conditional_identity.topk: 配置范围条件身份 Top-K 正确率。
        - 返回值.conditional_identity.top_k: self.top_k。
        - 返回值.conditional_identity.eligible_count: 配置范围条件身份分母。
        - 返回值.conditional_identity.seen_top1: 训练可见联合身份子集 Top-1；零分母时为 NaN。
        - 返回值.conditional_identity.seen_topk: 训练可见联合身份子集 Top-K；零分母时为 NaN。
        - 返回值.conditional_identity.seen_count: 训练可见联合身份子集分母。
        - 返回值.conditional_identity.unseen_top1: 训练未见联合身份子集 Top-1；零分母时为 NaN。
        - 返回值.conditional_identity.unseen_topk: 训练未见联合身份子集 Top-K；零分母时为 NaN。
        - 返回值.conditional_identity.unseen_count: 训练未见联合身份子集分母。
        - 返回值.end_to_end_identity: 配置范围背景或前景门后的最终身份报告。
        - 返回值.end_to_end_identity.precision: 配置范围端到端身份精确率。
        - 返回值.end_to_end_identity.recall: 配置范围端到端身份召回率。
        - 返回值.end_to_end_identity.f1: 配置范围端到端身份 F1。
        - 返回值.end_to_end_identity.true_positive: 配置范围端到端真阳性数。
        - 返回值.end_to_end_identity.predicted_positive: 配置范围端到端预测阳性数。
        - 返回值.end_to_end_identity.actual_positive: 配置范围端到端真实阳性数。
        - 返回值.small_molecule_identity: 小分子子清单和严格小分子分类门的 IdentityScopeMetrics。
        - 返回值.small_molecule_identity.intrinsic: 小分子固有排名报告。
        - 返回值.small_molecule_identity.intrinsic.top1: 小分子固有 Top-1。
        - 返回值.small_molecule_identity.intrinsic.topk: 小分子固有 Top-K。
        - 返回值.small_molecule_identity.intrinsic.top_k: self.top_k。
        - 返回值.small_molecule_identity.intrinsic.eligible_count: 小分子固有分母。
        - 返回值.small_molecule_identity.conditional: 小分子条件排名报告。
        - 返回值.small_molecule_identity.conditional.top1: 小分子条件 Top-1。
        - 返回值.small_molecule_identity.conditional.topk: 小分子条件 Top-K。
        - 返回值.small_molecule_identity.conditional.top_k: self.top_k。
        - 返回值.small_molecule_identity.conditional.eligible_count: 小分子条件分母。
        - 返回值.small_molecule_identity.end_to_end: 小分子端到端报告。
        - 返回值.small_molecule_identity.end_to_end.precision: 小分子端到端精确率。
        - 返回值.small_molecule_identity.end_to_end.recall: 小分子端到端召回率。
        - 返回值.small_molecule_identity.end_to_end.f1: 小分子端到端 F1。
        - 返回值.small_molecule_identity.end_to_end.true_positive: 小分子端到端真阳性数。
        - 返回值.small_molecule_identity.end_to_end.predicted_positive: 小分子端到端预测阳性数。
        - 返回值.small_molecule_identity.end_to_end.actual_positive: 小分子端到端真实阳性数。
        - 返回值.all_non_background_identity: 五类共享清单和任一前景分类门的 IdentityScopeMetrics。
        - 返回值.all_non_background_identity.intrinsic: 五类总体固有排名报告。
        - 返回值.all_non_background_identity.intrinsic.top1: 五类总体固有 Top-1。
        - 返回值.all_non_background_identity.intrinsic.topk: 五类总体固有 Top-K。
        - 返回值.all_non_background_identity.intrinsic.top_k: self.top_k。
        - 返回值.all_non_background_identity.intrinsic.eligible_count: 五类总体固有分母。
        - 返回值.all_non_background_identity.conditional: 五类总体条件排名报告。
        - 返回值.all_non_background_identity.conditional.top1: 五类总体条件 Top-1。
        - 返回值.all_non_background_identity.conditional.topk: 五类总体条件 Top-K。
        - 返回值.all_non_background_identity.conditional.top_k: self.top_k。
        - 返回值.all_non_background_identity.conditional.eligible_count: 五类总体条件分母。
        - 返回值.all_non_background_identity.end_to_end: 五类总体端到端报告。
        - 返回值.all_non_background_identity.end_to_end.precision: 五类总体端到端精确率。
        - 返回值.all_non_background_identity.end_to_end.recall: 五类总体端到端召回率。
        - 返回值.all_non_background_identity.end_to_end.f1: 五类总体端到端 F1。
        - 返回值.all_non_background_identity.end_to_end.true_positive: 五类总体端到端真阳性数。
        - 返回值.all_non_background_identity.end_to_end.predicted_positive: 五类总体端到端预测阳性数。
        - 返回值.all_non_background_identity.end_to_end.actual_positive: 五类总体端到端真实阳性数。
        - 返回值.identity_per_class: 五个前景类别名到 IdentityScopeMetrics 的映射。
        - 返回值.identity_per_class[*].intrinsic: 当前真实子类的固有排名报告。
        - 返回值.identity_per_class[*].intrinsic.top1: 当前类别固有 Top-1。
        - 返回值.identity_per_class[*].intrinsic.topk: 当前类别固有 Top-K。
        - 返回值.identity_per_class[*].intrinsic.top_k: self.top_k。
        - 返回值.identity_per_class[*].intrinsic.eligible_count: 当前类别固有分母。
        - 返回值.identity_per_class[*].conditional: 当前真实子类且六分类判对的排名报告。
        - 返回值.identity_per_class[*].conditional.top1: 当前类别条件 Top-1。
        - 返回值.identity_per_class[*].conditional.topk: 当前类别条件 Top-K。
        - 返回值.identity_per_class[*].conditional.top_k: self.top_k。
        - 返回值.identity_per_class[*].conditional.eligible_count: 当前类别条件分母。
        - 返回值.identity_per_class[*].end_to_end: 按最终 Top-1 身份所属子类归档预测阳性的端到端报告。
        - 返回值.identity_per_class[*].end_to_end.precision: 当前类别端到端精确率。
        - 返回值.identity_per_class[*].end_to_end.recall: 当前类别端到端召回率。
        - 返回值.identity_per_class[*].end_to_end.f1: 当前类别端到端 F1。
        - 返回值.identity_per_class[*].end_to_end.true_positive: 当前类别端到端真阳性数。
        - 返回值.identity_per_class[*].end_to_end.predicted_positive: 当前类别端到端预测阳性数。
        - 返回值.identity_per_class[*].end_to_end.actual_positive: 当前类别端到端真实阳性数。
        - 返回值.area: 两个区域头的微平均 Dice 报告。
        - 返回值.area.ligand_micro_dice: 2×ligand_intersection/(ligand_predicted+ligand_actual)，零分母时为 NaN。
        - 返回值.area.binding_micro_dice: 2×binding_intersection/(binding_predicted+binding_actual)，零分母时为 NaN。
        - 返回值.area.ligand_dice_denominator: ligand_predicted+ligand_actual。
        - 返回值.area.binding_dice_denominator: binding_predicted+binding_actual。

        本方法不清空或修改累计器，可在没有新增 update 时重复调用。返回的 NaN 仍是内存
        浮点值；严格 JSON 写入器在后续边界才把它规范化为 null。
        """

        # 三个 NumPy (6,) 数组分别是支持数、预测数与真阳性；默认dtype为int64，自定义混淆矩阵沿用其归约dtype。
        support = self.class_confusion.sum(axis=1)
        predicted = self.class_confusion.sum(axis=0)
        true_positive = np.diag(self.class_confusion)
        # class_f1 只收集 support>0 类别的浮点 F1，用于历史兼容的支持类 macro-F1。
        class_f1 = []
        # class_metrics 以固定 CLASS_NAMES 为键，逐项保存比率和整数分母。
        class_metrics: dict[str, PerClassMetrics] = {}
        for index, name in enumerate(CLASS_NAMES):
            precision = _safe_divide(float(true_positive[index]), float(predicted[index]))
            recall = _safe_divide(float(true_positive[index]), float(support[index]))
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if math.isfinite(precision) and math.isfinite(recall) and precision + recall > 0
                else 0.0
            )
            if support[index] > 0:
                class_f1.append(f1)
            class_metrics[name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support[index]),
                "predicted_count": int(predicted[index]),
            }
        # 三个总体六分类标量分别对应准确率、支持类平均和固定六类平均。
        accuracy = _safe_divide(float(true_positive.sum()), float(self.class_confusion.sum()))
        supported_macro_f1 = float(np.mean(class_f1)) if class_f1 else math.nan
        all_classes_macro_f1 = float(
            np.mean([class_metrics[name]["f1"] for name in CLASS_NAMES])
        )
        # small_report/all_report 是两个并行口径的完整 IdentityScopeMetrics。
        small_report = self.small_identity.compute(self.top_k)
        all_report = self.all_identity.compute(self.top_k)
        # configured_report 只控制三个兼容顶层身份字段和 seen/unseen；两个详细报告始终返回。
        configured_report = (
            small_report
            if self.identity_scope in (None, "small_molecule")
            else all_report
        )
        configured_intrinsic = configured_report["intrinsic"]
        configured_conditional = configured_report["conditional"]
        configured_end_to_end = configured_report["end_to_end"]
        # conditional_with_history 在配置范围基本排名上补齐跨批次 seen/unseen 六个字段。
        conditional_with_history: ConditionalIdentityMetrics = {
            **configured_conditional,
            "seen_top1": _safe_divide(self.seen_correct_at_1, self.seen_eligible),
            "seen_topk": _safe_divide(self.seen_correct_at_k, self.seen_eligible),
            "seen_count": self.seen_eligible,
            "unseen_top1": _safe_divide(self.unseen_correct_at_1, self.unseen_eligible),
            "unseen_topk": _safe_divide(self.unseen_correct_at_k, self.unseen_eligible),
            "unseen_count": self.unseen_eligible,
        }
        # per_class_identity 固定把五个前景下标映射到同序 FOREGROUND_CLASS_NAMES。
        per_class_identity = {
            class_name: self.per_class_identity[class_index].compute(self.top_k)
            for class_index, class_name in zip(
                FOREGROUND_CLASS_INDICES,
                FOREGROUND_CLASS_NAMES,
            )
        }
        return {
            "six_class": {
                "accuracy": accuracy,
                "macro_f1": supported_macro_f1,
                "macro_f1_supported_classes": supported_macro_f1,
                "macro_f1_all_classes": all_classes_macro_f1,
                "per_class": class_metrics,
                "confusion_matrix": self.class_confusion.tolist(),
            },
            "intrinsic_identity": configured_intrinsic,
            "conditional_identity": conditional_with_history,
            "end_to_end_identity": configured_end_to_end,
            "small_molecule_identity": small_report,
            "all_non_background_identity": all_report,
            "identity_per_class": per_class_identity,
            "area": {
                "ligand_micro_dice": _safe_divide(
                    2 * self.ligand_intersection,
                    self.ligand_predicted + self.ligand_actual,
                ),
                "binding_micro_dice": _safe_divide(
                    2 * self.binding_intersection,
                    self.binding_predicted + self.binding_actual,
                ),
                "ligand_dice_denominator": self.ligand_predicted + self.ligand_actual,
                "binding_dice_denominator": self.binding_predicted + self.binding_actual,
            },
        }


def selection_score(
    task_name: str,
    six_class_macro_f1: float,
    conditional_identity_top1: float,
    end_to_end_identity_f1: float,
) -> float:
    """按任务类型把调用方提供的三个指标组合为 checkpoint 选择分数。

    - task_name: candidate_pretrain 或 full_matcher。
    - six_class_macro_f1: 调用方传入的支持类六分类 macro-F1；本函数不替换或拒绝非有限值。
    - conditional_identity_top1: 调用方传入的条件身份 Top-1；full_matcher 中非有限值按 0 计入，候选预训练忽略。
    - end_to_end_identity_f1: 调用方传入的端到端身份 F1；full_matcher 中非有限值按 0 计入，候选预训练忽略。
    - 返回值: candidate_pretrain 原样返回 six_class_macro_f1；full_matcher 等权平均 six_class_macro_f1 与两项有限化身份指标。

    训练引擎只用 validation 报告调用本函数，但函数签名不携带划分名，因此不会在本地识别
    或拒绝 calibration 来源。它不读取文件，也不检查最终平均值是否有限；训练引擎在选模前
    另行执行有限性门禁。
    """

    if task_name not in ("candidate_pretrain", "full_matcher"):
        raise ValueError(f"不支持 task_name={task_name!r}。")
    if task_name == "candidate_pretrain":
        return six_class_macro_f1
    # 两个身份项在零支持产生 NaN 时各自按 0 参与完整 Matcher 等权平均。
    finite_identity = (
        0.0 if not math.isfinite(end_to_end_identity_f1) else end_to_end_identity_f1
    )
    finite_conditional = (
        0.0 if not math.isfinite(conditional_identity_top1) else conditional_identity_top1
    )
    return (six_class_macro_f1 + finite_identity + finite_conditional) / 3.0
