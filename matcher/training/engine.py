"""执行单卡梯度累积训练、完整 validation、早停与阶段产物发布。

evaluate_model 汇总调用方指定的完整 DataLoader。run_training_stage 从零构造一个阶段，按轮次
训练、验证、调度学习率、更新 BEST.pt 与 LAST.pt，并通过 W&B online 写出曲线。resolved.json、
tracking.json、history.jsonl、两个 checkpoint 和 summary.json 全部位于显式 stage_directory，
本模块不读取 calibration。内存中的零支持指标可为 NaN；严格 JSON 写入层把这些值变成 null。
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypedDict, cast

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ..data.manifest import ensure_manifest
from ..data.schema import CLASS_NAMES, IdentityScope
from ..datasets.batching import (
    DATALOADER_MULTIPROCESSING_CONTEXT,
    ShuffledBatchSampler,
    build_candidate_loader,
)
from ..datasets.candidate_data import CandidateDataset
from ..datasets.contracts import CandidateBatch
from ..evaluation.metrics import (
    EvaluationAccumulator,
    EvaluationReport,
    IdentityScopeMetrics,
    selection_score,
)
from ..model import (
    CandidateEncoder,
    ClosedSetMatcher,
    FullMatcherOutput,
    IdentityScoreComponents,
)
from ..utils.json_artifacts import append_jsonl, write_json
from .checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    save_checkpoint,
)
from .config import DataConfig, Precision, StageConfig, TrackingConfig
from .initialization import (
    CandidateInitializationRecord,
    initialize_candidate_encoder_from_pretraining,
)
from .losses import (
    LossOutput,
    LossWeights,
    MatcherLossBatch,
    compute_matcher_loss,
    identity_eligible_mask,
)
from .numeric import configure_random_seeds_and_numeric_backend
from .tracking import MetricValue, WandbTracker, start_wandb_tracking


# tuple[float,float]，AdamW一阶与二阶矩指数衰减系数，写入resolved.json留证。
ADAM_BETAS = (0.9, 0.999)
# float，AdamW分母数值稳定项，写入resolved.json留证。
ADAM_EPSILON = 1e-8
# float，validation选择分数未改善时ReduceLROnPlateau的学习率乘数。
SCHEDULER_FACTOR = 0.5
# float，ReduceLROnPlateau允许下降到的最小学习率。
SCHEDULER_MINIMUM_LEARNING_RATE = 1e-7
# float，新旧学习率差不超过此值时调度器不执行更新。
SCHEDULER_EPSILON = 1e-8
# float，fp16 GradScaler初始缩放因子；bf16与fp32路径创建但禁用scaler。
GRAD_SCALER_INITIAL_SCALE = 65536.0
# float，连续有限更新达到增长间隔时的scaler乘数。
GRAD_SCALER_GROWTH_FACTOR = 2.0
# float，fp16非有限梯度时的scaler回退乘数；训练另有显式非有限门禁。
GRAD_SCALER_BACKOFF_FACTOR = 0.5
# int，GradScaler在连续有限更新后放大缩放因子的更新间隔。
GRAD_SCALER_GROWTH_INTERVAL = 2000
# tuple[str,...]，CandidateBatch中必须由CPU异步移动到训练device的全部张量字段顺序。
_TENSOR_BATCH_KEYS = (
    "voxel_input",
    "class_target",
    "ligand_area_target",
    "binding_area_target",
    "full_candidate_voxel_count",
    "candidate_manual_features",
    "candidate_sampling_features",
    "a_tokens",
    "a_mask",
    "p_tokens",
    "p_mask",
    "v_tokens",
    "v_mask",
    "v_index_zyx",
    "pp_tokens",
    "pp_mask",
    "pp_index_zyx",
    "language_embeddings",
    "identity_mask",
    "identity_class_index",
    "identity_target",
    "graph_to_sample",
    "graph_to_identity",
    "graph_element",
    "graph_charge",
    "graph_residue_id",
    "graph_node_continuous",
    "graph_node_graph_index",
    "graph_edge_index",
    "graph_edge_features",
)


class LossValues(TypedDict):
    """声明完整训练轮次或评估划分的五个全局归一损失。

    - total: Python float；四项分量按 LossWeights 加权之和。
    - classification: Python float；六分类交叉熵分子除以全局候选数或真实类别权重和。
    - ligand_area: Python float；逐候选配体区域损失的算术均值。
    - binding_area: Python float；逐候选结合区域损失的算术均值。
    - identity: Python float；当前身份范围内目标有效候选的身份损失均值，无有效候选时为 0。

    每个物理批次的损失张量在累计前经过有限性门禁；跨批次使用 Python float 乘加，仍可能
    溢出为 Inf，_LossStatistics.compute 不执行最终有限性检查，因此本类型不承诺五个值有限。
    """

    total: float
    classification: float
    ligand_area: float
    binding_area: float
    identity: float


class IdentityDiagnostics(TypedDict):
    """声明完整 Matcher 在训练 epoch 结束时的两个身份出口门值。

    - graph_gate: Python float；配体原子图身份分数增量的可训练标量门，不经过 sigmoid 或夹紧。
    - inventory_gate: Python float；自排除身份清单分数增量的可训练标量门，不经过 sigmoid 或夹紧。
    """

    graph_gate: float
    inventory_gate: float


class IdentityContributionDiagnostics(TypedDict):
    """声明 validation 身份分数增量相对语言基线排序尺度的诊断结果。

    - multi_identity_eligible_count: Python int；身份目标有效且 identity_mask 至少含两个 True 槽的候选数。
    - nonzero_native_count: Python int；上述候选中语言基线中心化 L2 范数大于 1e-12 的数量。
    - native_centered_l2_mean: Python float 或 None；语言基线在有效身份槽内去均值后的无量纲 L2 范数均值。
    - graph_increment_centered_l2_mean: Python float 或 None；已乘 graph_gate 的图分数增量中心化 L2 范数均值。
    - inventory_increment_centered_l2_mean: Python float 或 None；已乘 inventory_gate 的清单分数增量中心化 L2 范数均值。
    - graph_to_native_ratio_mean: Python float 或 None；图增量范数除以非零语言基线范数的逐候选均值。
    - inventory_to_native_ratio_mean: Python float 或 None；清单增量范数除以非零语言基线范数的逐候选均值。
    - combined_to_native_ratio_mean: Python float 或 None；两种门控增量和的中心化范数除以非零语言基线范数的逐候选均值。

    三个绝对范数均值以 multi_identity_eligible_count 为分母；三个相对比值均值只以
    nonzero_native_count 为分母。对应分母为零时字段为 None。
    """

    multi_identity_eligible_count: int
    nonzero_native_count: int
    native_centered_l2_mean: float | None
    graph_increment_centered_l2_mean: float | None
    inventory_increment_centered_l2_mean: float | None
    graph_to_native_ratio_mean: float | None
    inventory_to_native_ratio_mean: float | None
    combined_to_native_ratio_mean: float | None


class _IdentityContributionStatistics:
    """跨 validation 批次累积不参与损失、正式指标或选模的身份贡献尺度。

    - self.multi_identity_eligible_count: 身份目标有效且至少有两个真实身份槽的累计候选数。
    - self.nonzero_native_count: 上述候选中语言基线中心化范数大于 1e-12 的累计数量。
    - self.native_centered_l2_sum: 语言基线中心化 L2 范数累计和。
    - self.graph_increment_centered_l2_sum: 已门控图增量中心化 L2 范数累计和。
    - self.inventory_increment_centered_l2_sum: 已门控清单增量中心化 L2 范数累计和。
    - self.graph_to_native_ratio_sum: 非零语言基线候选的图增量范数比累计和。
    - self.inventory_to_native_ratio_sum: 非零语言基线候选的清单增量范数比累计和。
    - self.combined_to_native_ratio_sum: 非零语言基线候选的两种增量和范数比累计和。
    """

    def __init__(self) -> None:
        """把两个支持数与六个浮点分子初始化为零。"""

        self.multi_identity_eligible_count = 0
        self.nonzero_native_count = 0
        self.native_centered_l2_sum = 0.0
        self.graph_increment_centered_l2_sum = 0.0
        self.inventory_increment_centered_l2_sum = 0.0
        self.graph_to_native_ratio_sum = 0.0
        self.inventory_to_native_ratio_sum = 0.0
        self.combined_to_native_ratio_sum = 0.0

    @staticmethod
    def _centered_l2(values: torch.Tensor, identity_mask: torch.Tensor) -> torch.Tensor:
        """在每个候选的有效身份槽内去均值并计算 L2 范数。

        - values: 浮点张量，形状 (B,I)；身份分数或已门控分数增量。
        - identity_mask: bool 张量，形状 (B,I)；True 为真实身份槽，False 为补零槽。

        - 返回值: float64 张量，形状 (B,)；每个候选只在 True 槽内去均值并求 L2 范数。

        本函数不显式核验输入同形或 dtype。有效槽计数以 clamp_min(1) 防止空身份轴除零；
        正式调用只选择至少两个有效身份槽的候选。
        """

        # float64张量，形状 (B,I)；身份值提升精度后的副本，设备保持不变。
        values64 = values.to(dtype=torch.float64)
        # float64张量，形状 (B,I)；补零身份槽显式置零。
        valid_values = torch.where(identity_mask, values64, 0.0)
        # float64张量，形状 (B,)；每个候选有效身份数，下限1只用于避免除零。
        counts = identity_mask.sum(dim=1).to(dtype=torch.float64).clamp_min(1.0)
        # float64张量，形状 (B,)；每个候选有效身份分数均值。
        means = valid_values.sum(dim=1) / counts
        # float64张量，形状 (B,I)；有效槽减均值，补零槽保持0。
        centered = torch.where(identity_mask, values64 - means[:, None], 0.0)
        return centered.square().sum(dim=1).sqrt()

    def update(
        self,
        components: IdentityScoreComponents,
        batch: MatcherLossBatch,
        identity_scope: IdentityScope,
    ) -> None:
        """从一个 validation 物理批次累积三条身份分数分支的排序尺度。

        - components: 完整 Matcher 的三字段 IdentityScoreComponents。
        - components.native: 浮点张量，形状 (B,I)；语言身份原始配对分数。
        - components.graph_increment: 浮点张量，形状 (B,I)；已经乘 graph_gate 的图增量。
        - components.inventory_increment: 浮点张量，形状 (B,I)；已经乘 inventory_gate 的清单增量。
        - batch: 当前物理批次损失路径读取的 MatcherLossBatch。
        - batch.class_target: int64 张量，形状 (B,)；真实六分类下标。
        - batch.ligand_area_target: 本函数不读取。
        - batch.binding_area_target: 本函数不读取。
        - batch.identity_target: int64 张量，形状 (B,)；非负值表示身份监督可用。
        - batch.identity_mask: bool 张量，形状 (B,I)；True 为真实身份槽。
        - batch.identity_class_index: 本函数不读取。
        - identity_scope: small_molecule 或 all_non_background；决定 class_target 的监督范围。

        只选择身份目标有效且至少两个真实身份槽的候选。三个分支先各自在有效槽去均值；
        比值只对语言基线范数大于 1e-12 的候选计算。所有批次汇总值先在设备上堆叠，再一次
        转到 CPU；任一 NaN 或 Inf 都失败关闭。
        """

        # bool张量，形状 (B,I)；当前批次真实身份槽掩码。
        identity_mask = batch["identity_mask"]
        # bool张量，形状 (B,)；由真实类别范围和非负身份目标决定的候选监督掩码。
        eligible = identity_eligible_mask(
            batch,
            identity_scope,
        )
        # bool张量，形状 (B,)；进一步要求每个候选至少两个真实身份槽。
        selected = eligible & (identity_mask.sum(dim=1) >= 2)
        # bool张量，形状 (B_s,I)；B_s为本批选中候选数。
        selected_mask = identity_mask[selected]
        # float64张量，形状 (B_s,)；语言基线中心化L2范数。
        native = self._centered_l2(components["native"][selected], selected_mask)
        # float64张量，形状 (B_s,)；已门控图增量中心化L2范数。
        graph = self._centered_l2(
            components["graph_increment"][selected],
            selected_mask,
        )
        # float64张量，形状 (B_s,)；已门控清单增量中心化L2范数。
        inventory = self._centered_l2(
            components["inventory_increment"][selected],
            selected_mask,
        )
        # float64张量，形状 (B_s,)；两种已门控增量之和的中心化L2范数。
        combined = self._centered_l2(
            (
                components["graph_increment"][selected]
                + components["inventory_increment"][selected]
            ),
            selected_mask,
        )
        # bool张量，形状 (B_s,)；语言基线中心化范数具有稳定比值分母的候选。
        nonzero_native = native > 1e-12
        # float64张量，形状 (B_n,)；B_n为非零语言基线候选数。
        denominator = native[nonzero_native]
        # float64张量，形状 (8,)；两个计数、三个绝对范数和、三个相对比值和。
        summary = torch.stack(
            (
                selected.sum().to(dtype=torch.float64),
                nonzero_native.sum().to(dtype=torch.float64),
                native.sum(),
                graph.sum(),
                inventory.sum(),
                (graph[nonzero_native] / denominator).sum(),
                (inventory[nonzero_native] / denominator).sum(),
                (combined[nonzero_native] / denominator).sum(),
            )
        )
        # list[float]，一次设备同步后得到的八项批次汇总值。
        values = [float(value) for value in summary.detach().cpu().tolist()]
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError("validation 身份分数贡献诊断含 NaN 或 Inf。")
        self.multi_identity_eligible_count += int(values[0])
        self.nonzero_native_count += int(values[1])
        self.native_centered_l2_sum += values[2]
        self.graph_increment_centered_l2_sum += values[3]
        self.inventory_increment_centered_l2_sum += values[4]
        self.graph_to_native_ratio_sum += values[5]
        self.inventory_to_native_ratio_sum += values[6]
        self.combined_to_native_ratio_sum += values[7]

    def compute(self) -> IdentityContributionDiagnostics:
        """把累计分子和支持数转换为身份贡献诊断记录。

        - 返回值.multi_identity_eligible_count: 累计多身份有效候选数。
        - 返回值.nonzero_native_count: 累计非零语言基线候选数。
        - 返回值.native_centered_l2_mean: 语言基线绝对范数均值或 None。
        - 返回值.graph_increment_centered_l2_mean: 图增量绝对范数均值或 None。
        - 返回值.inventory_increment_centered_l2_mean: 清单增量绝对范数均值或 None。
        - 返回值.graph_to_native_ratio_mean: 图增量相对语言基线比值均值或 None。
        - 返回值.inventory_to_native_ratio_mean: 清单增量相对语言基线比值均值或 None。
        - 返回值.combined_to_native_ratio_mean: 两种增量和相对语言基线比值均值或 None。
        """

        # int，三个绝对范数均值共用的候选分母。
        candidate_count = self.multi_identity_eligible_count
        # int，三个相对比值均值共用的非零语言基线分母。
        native_count = self.nonzero_native_count
        return {
            "multi_identity_eligible_count": candidate_count,
            "nonzero_native_count": native_count,
            "native_centered_l2_mean": (
                self.native_centered_l2_sum / candidate_count
                if candidate_count > 0
                else None
            ),
            "graph_increment_centered_l2_mean": (
                self.graph_increment_centered_l2_sum / candidate_count
                if candidate_count > 0
                else None
            ),
            "inventory_increment_centered_l2_mean": (
                self.inventory_increment_centered_l2_sum / candidate_count
                if candidate_count > 0
                else None
            ),
            "graph_to_native_ratio_mean": (
                self.graph_to_native_ratio_sum / native_count
                if native_count > 0
                else None
            ),
            "inventory_to_native_ratio_mean": (
                self.inventory_to_native_ratio_sum / native_count
                if native_count > 0
                else None
            ),
            "combined_to_native_ratio_mean": (
                self.combined_to_native_ratio_sum / native_count
                if native_count > 0
                else None
            ),
        }


class TrainingEpochResult(TypedDict):
    """声明一个完整训练 epoch 的损失、梯度、性能与状态结果。

    - loss: 完整 epoch 的五项全局归一 LossValues。
    - loss.total: 四项损失加权和。
    - loss.classification: 六分类全局归一损失。
    - loss.ligand_area: 配体区域逐候选均值。
    - loss.binding_area: 结合区域逐候选均值。
    - loss.identity: 有效身份候选均值，无有效候选时为 0。
    - sample_count: 本轮实际参与训练的候选数。
    - identity_eligible_count: 当前身份范围内 identity_target 非负的真实候选数。
    - optimizer_steps: 本轮完成的 global batch 参数更新次数。
    - skipped_optimizer_steps: 保留统一产物字段；当前显式非有限失败策略下恒为 0。
    - mean_preclip_gradient_norm: 各更新在裁剪前的全模型 L2 梯度范数均值。
    - maximum_preclip_gradient_norm: 各更新在裁剪前的全模型 L2 梯度范数最大值。
    - mean_graph_branch_gradient_norm: 裁剪后图 PairScorer 参数梯度 L2 范数均值；没有现有梯度时为 None。
    - mean_inventory_branch_gradient_norm: 裁剪后清单 PairScorer 参数梯度 L2 范数均值；没有现有梯度时为 None。
    - identity_diagnostics: 完整 Matcher 的两个门值；CandidateEncoder 任务为 None。
    - identity_diagnostics.graph_gate: epoch 末图出口门 Python float。
    - identity_diagnostics.inventory_gate: epoch 末清单出口门 Python float。
    - elapsed_seconds: 从进入 _train_epoch 到结果构造的墙钟秒数。
    - data_wait_seconds: 等待 DataLoader 产生各完整累积组的墙钟秒数和。
    - compute_seconds: 各 global batch 从首个CPU批次搬运到CUDA同步完成的墙钟秒数和。
    - samples_per_second: sample_count / max(elapsed_seconds,1e-12)。
    - peak_cuda_allocated_bytes: reset_peak_memory_stats 后本轮 PyTorch 最大已分配显存 byte 数。
    - peak_cuda_reserved_bytes: reset_peak_memory_stats 后本轮 PyTorch 最大保留显存 byte 数。
    - global_step: 本轮结束后的累计参数更新序号。
    """

    loss: LossValues
    sample_count: int
    identity_eligible_count: int
    optimizer_steps: int
    skipped_optimizer_steps: int
    mean_preclip_gradient_norm: float
    maximum_preclip_gradient_norm: float
    mean_graph_branch_gradient_norm: float | None
    mean_inventory_branch_gradient_norm: float | None
    identity_diagnostics: IdentityDiagnostics | None
    elapsed_seconds: float
    data_wait_seconds: float
    compute_seconds: float
    samples_per_second: float
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    global_step: int


class EvaluationResult(TypedDict):
    """声明一个完整评估 DataLoader 的损失、指标与吞吐结果。

    - loss: 整个划分的五项全局归一 LossValues。
    - loss.total: 四项损失加权和。
    - loss.classification: 六分类全局归一损失。
    - loss.ligand_area: 配体区域逐候选均值。
    - loss.binding_area: 结合区域逐候选均值。
    - loss.identity: 有效身份候选均值，无有效候选时为 0。
    - metrics: EvaluationAccumulator.compute 返回的完整 EvaluationReport。
    - metrics.six_class: 全候选六分类报告。
    - metrics.six_class.accuracy: 全候选六分类准确率。
    - metrics.six_class.macro_f1: 与正式选模兼容的支持类 macro-F1。
    - metrics.six_class.macro_f1_supported_classes: 只平均真实支持数大于零类别的 macro-F1。
    - metrics.six_class.macro_f1_all_classes: 固定平均六类且把零支持类 F1 记为 0 的 macro-F1。
    - metrics.six_class.per_class: 以六个类别名为键的逐类指标映射。
    - metrics.six_class.per_class[*].precision: 当前类别 precision；零预测时为 NaN。
    - metrics.six_class.per_class[*].recall: 当前类别 recall；零支持时为 NaN。
    - metrics.six_class.per_class[*].f1: 当前类别 F1；无法计算时为 0。
    - metrics.six_class.per_class[*].support: 当前类别真实候选数。
    - metrics.six_class.per_class[*].predicted_count: 当前类别预测候选数。
    - metrics.six_class.confusion_matrix: 形状 6×6 的整数嵌套列表，第一轴为真实类，第二轴为预测类。
    - metrics.intrinsic_identity: 配置身份范围内忽略六分类预测的身份报告。
    - metrics.intrinsic_identity.top1: 固有身份 Top-1 正确率。
    - metrics.intrinsic_identity.topk: 固有身份 Top-K 正确率。
    - metrics.intrinsic_identity.top_k: 固有身份报告实际采用的最大名次 K。
    - metrics.intrinsic_identity.eligible_count: 固有身份可监督候选数。
    - metrics.conditional_identity: 只在六分类预测正确时统计的身份报告。
    - metrics.conditional_identity.top1: 条件身份 Top-1 正确率。
    - metrics.conditional_identity.topk: 条件身份 Top-K 正确率。
    - metrics.conditional_identity.top_k: 条件身份报告实际采用的最大名次 K。
    - metrics.conditional_identity.eligible_count: 条件身份候选数。
    - metrics.conditional_identity.seen_top1: 当前语言家族 train seen 子集 Top-1。
    - metrics.conditional_identity.seen_topk: 当前语言家族 train seen 子集 Top-K。
    - metrics.conditional_identity.seen_count: 当前语言家族 train seen 子集候选数。
    - metrics.conditional_identity.unseen_top1: 当前语言家族 train unseen 子集 Top-1。
    - metrics.conditional_identity.unseen_topk: 当前语言家族 train unseen 子集 Top-K。
    - metrics.conditional_identity.unseen_count: 当前语言家族 train unseen 子集候选数。
    - metrics.end_to_end_identity: 配置身份范围经前景门后的最终身份报告。
    - metrics.end_to_end_identity.precision: 端到端身份 precision。
    - metrics.end_to_end_identity.recall: 端到端身份 recall。
    - metrics.end_to_end_identity.f1: 端到端身份 F1。
    - metrics.end_to_end_identity.true_positive: 通过前景门且最终身份正确的候选数。
    - metrics.end_to_end_identity.predicted_positive: 通过前景门且最终身份属于配置范围的候选数。
    - metrics.end_to_end_identity.actual_positive: 配置范围内具备身份目标的真实候选数。
    - metrics.small_molecule_identity: 只含小分子身份且使用严格小分子分类门的范围报告。
    - metrics.small_molecule_identity.intrinsic: 小分子固有身份报告。
    - metrics.small_molecule_identity.intrinsic.top1: 小分子固有 Top-1。
    - metrics.small_molecule_identity.intrinsic.topk: 小分子固有 Top-K。
    - metrics.small_molecule_identity.intrinsic.top_k: 小分子报告实际采用的最大名次 K。
    - metrics.small_molecule_identity.intrinsic.eligible_count: 小分子固有身份候选数。
    - metrics.small_molecule_identity.conditional: 小分子条件身份报告。
    - metrics.small_molecule_identity.conditional.top1: 小分子条件 Top-1。
    - metrics.small_molecule_identity.conditional.topk: 小分子条件 Top-K。
    - metrics.small_molecule_identity.conditional.top_k: 小分子条件报告实际采用的最大名次 K。
    - metrics.small_molecule_identity.conditional.eligible_count: 小分子条件身份候选数。
    - metrics.small_molecule_identity.end_to_end: 小分子端到端报告。
    - metrics.small_molecule_identity.end_to_end.precision: 小分子端到端 precision。
    - metrics.small_molecule_identity.end_to_end.recall: 小分子端到端 recall。
    - metrics.small_molecule_identity.end_to_end.f1: 小分子端到端 F1。
    - metrics.small_molecule_identity.end_to_end.true_positive: 小分子端到端最终身份正确数。
    - metrics.small_molecule_identity.end_to_end.predicted_positive: 小分子端到端预测阳性数。
    - metrics.small_molecule_identity.end_to_end.actual_positive: 小分子端到端真实阳性数。
    - metrics.all_non_background_identity: 五类前景共享身份清单的范围报告。
    - metrics.all_non_background_identity.intrinsic: 五类前景固有身份报告。
    - metrics.all_non_background_identity.intrinsic.top1: 五类前景固有 Top-1。
    - metrics.all_non_background_identity.intrinsic.topk: 五类前景固有 Top-K。
    - metrics.all_non_background_identity.intrinsic.top_k: 五类前景报告实际采用的最大名次 K。
    - metrics.all_non_background_identity.intrinsic.eligible_count: 五类前景固有身份候选数。
    - metrics.all_non_background_identity.conditional: 五类前景条件身份报告。
    - metrics.all_non_background_identity.conditional.top1: 五类前景条件 Top-1。
    - metrics.all_non_background_identity.conditional.topk: 五类前景条件 Top-K。
    - metrics.all_non_background_identity.conditional.top_k: 五类前景条件报告实际采用的最大名次 K。
    - metrics.all_non_background_identity.conditional.eligible_count: 五类前景条件身份候选数。
    - metrics.all_non_background_identity.end_to_end: 五类前景端到端报告。
    - metrics.all_non_background_identity.end_to_end.precision: 五类前景端到端 precision。
    - metrics.all_non_background_identity.end_to_end.recall: 五类前景端到端 recall。
    - metrics.all_non_background_identity.end_to_end.f1: 五类前景端到端 F1。
    - metrics.all_non_background_identity.end_to_end.true_positive: 五类前景端到端最终身份正确数。
    - metrics.all_non_background_identity.end_to_end.predicted_positive: 五类前景端到端预测阳性数。
    - metrics.all_non_background_identity.end_to_end.actual_positive: 五类前景端到端真实阳性数。
    - metrics.identity_per_class: 以前景类别名为键的逐类身份范围报告。
    - metrics.identity_per_class[*].intrinsic: 当前前景类固有身份报告。
    - metrics.identity_per_class[*].intrinsic.top1: 当前前景类固有 Top-1。
    - metrics.identity_per_class[*].intrinsic.topk: 当前前景类固有 Top-K。
    - metrics.identity_per_class[*].intrinsic.top_k: 当前前景类报告实际采用的最大名次 K。
    - metrics.identity_per_class[*].intrinsic.eligible_count: 当前前景类固有身份候选数。
    - metrics.identity_per_class[*].conditional: 当前前景类条件身份报告。
    - metrics.identity_per_class[*].conditional.top1: 当前前景类条件 Top-1。
    - metrics.identity_per_class[*].conditional.topk: 当前前景类条件 Top-K。
    - metrics.identity_per_class[*].conditional.top_k: 当前前景类条件报告实际采用的最大名次 K。
    - metrics.identity_per_class[*].conditional.eligible_count: 当前前景类条件身份候选数。
    - metrics.identity_per_class[*].end_to_end: 当前前景类端到端报告。
    - metrics.identity_per_class[*].end_to_end.precision: 当前前景类端到端 precision。
    - metrics.identity_per_class[*].end_to_end.recall: 当前前景类端到端 recall。
    - metrics.identity_per_class[*].end_to_end.f1: 当前前景类端到端 F1。
    - metrics.identity_per_class[*].end_to_end.true_positive: 当前前景类端到端最终身份正确数。
    - metrics.identity_per_class[*].end_to_end.predicted_positive: 当前前景类端到端预测阳性数。
    - metrics.identity_per_class[*].end_to_end.actual_positive: 当前前景类端到端真实阳性数。
    - metrics.area: 两个体素区域的微平均 Dice 报告。
    - metrics.area.ligand_micro_dice: 配体区域跨候选与体素微平均 Dice。
    - metrics.area.binding_micro_dice: 结合区域跨候选与体素微平均 Dice。
    - metrics.area.ligand_dice_denominator: 配体区域预测体素数与真实体素数之和。
    - metrics.area.binding_dice_denominator: 结合区域预测体素数与真实体素数之和。
    - sample_count: 该 loader 实际纳入评估的候选数。
    - identity_eligible_count: 当前身份范围内 identity_target 非负的真实候选数。
    - elapsed_seconds: 从进入 evaluate_model 到完成指标汇总的墙钟秒数。
    - samples_per_second: sample_count / max(elapsed_seconds,1e-12)。
    """

    loss: LossValues
    metrics: EvaluationReport
    sample_count: int
    identity_eligible_count: int
    elapsed_seconds: float
    samples_per_second: float


class StageSummary(TypedDict):
    """声明训练循环完成后、调用 W&B finish(0) 前写入 summary.json 的十个字段。

    - stage: YAML 中的安全阶段名。
    - best_epoch: 最佳 validation 选择分数所在的非负零起始 epoch。
    - best_score: 有限最佳 validation 选择分数。
    - last_epoch: 实际完成的最后一个非负零起始 epoch。
    - global_step: 阶段结束时累计完成的参数更新数。
    - stopped_by_patience: 是否因 bad_validation_epochs 达到配置阈值而退出循环。
    - best_checkpoint: stage_directory / BEST.pt 的路径字符串。
    - last_checkpoint: stage_directory / LAST.pt 的路径字符串。
    - tracking_run_id: 本地 tracker 从 W&B SDK 取得的非空 run 标识；文件本身不证明 finish(0) 已返回或远端同步完成。
    - tracking_url: 本地 tracker 从 W&B SDK 取得的 HTTPS 地址；文件本身不查询远端 run 状态。

    run_training_stage 在写文件后调用 finish(0)，只有该调用正常返回才返回此映射；finish 失败时
    已写出的 summary.json 可以保留，且根计划不会把该阶段作为成功返回值继续收口。
    """

    stage: str
    best_epoch: int
    best_score: float
    last_epoch: int
    global_step: int
    stopped_by_patience: bool
    best_checkpoint: str
    last_checkpoint: str
    tracking_run_id: str
    tracking_url: str


class EpochRecord(TypedDict):
    """声明追加到 history.jsonl 并嵌入 checkpoint 的单轮记录。

    - epoch: 非负零起始轮次。
    - global_step: 本轮训练结束后的累计参数更新数。
    - learning_rate_used: 本轮训练开始前读取的 AdamW 学习率。
    - next_learning_rate: 本轮完整 validation 后调用调度器得到的下一轮学习率。
    - selection_score: 本轮完整 validation 的有限选模分数；candidate_pretrain 只取 six_class.macro_f1，full_matcher 才合成 six_class.macro_f1、conditional_identity.top1 与 end_to_end_identity.f1 三项。
    - improved: selection_score 是否严格大于此前 best_score。
    - bad_validation_epochs: 截至本轮连续未严格改善轮数；改善时重置为 0。
    - training: 本轮完整 TrainingEpochResult。
    - training.loss: 本轮五项 LossValues。
    - training.loss.total: 本轮四项加权总损失。
    - training.loss.classification: 本轮六分类损失。
    - training.loss.ligand_area: 本轮配体区域损失。
    - training.loss.binding_area: 本轮结合区域损失。
    - training.loss.identity: 本轮身份损失。
    - training.sample_count: 本轮训练候选数。
    - training.identity_eligible_count: 本轮有效身份候选数。
    - training.optimizer_steps: 本轮参数更新数。
    - training.skipped_optimizer_steps: 当前失败关闭策略下固定为 0。
    - training.mean_preclip_gradient_norm: 本轮裁剪前梯度范数均值。
    - training.maximum_preclip_gradient_norm: 本轮裁剪前梯度范数最大值。
    - training.mean_graph_branch_gradient_norm: 本轮裁剪后图分支梯度范数均值或 None。
    - training.mean_inventory_branch_gradient_norm: 本轮裁剪后清单分支梯度范数均值或 None。
    - training.identity_diagnostics: 两个身份门值或 None。
    - training.identity_diagnostics.graph_gate: epoch 末图门值。
    - training.identity_diagnostics.inventory_gate: epoch 末清单门值。
    - training.elapsed_seconds: 本轮墙钟秒数。
    - training.data_wait_seconds: 本轮 DataLoader 等待秒数和。
    - training.compute_seconds: 本轮各更新 CUDA 同步计算秒数和。
    - training.samples_per_second: 本轮候选吞吐。
    - training.peak_cuda_allocated_bytes: 本轮最大已分配显存 byte 数。
    - training.peak_cuda_reserved_bytes: 本轮最大保留显存 byte 数。
    - training.global_step: 与顶层 global_step 相同的本轮结束累计更新数。
    - validation: 本轮完整 EvaluationResult。
    - validation.loss: 整个 validation 划分的五项 LossValues。
    - validation.loss.total: validation 四项加权总损失。
    - validation.loss.classification: validation 六分类损失。
    - validation.loss.ligand_area: validation 配体区域损失。
    - validation.loss.binding_area: validation 结合区域损失。
    - validation.loss.identity: validation 身份损失。
    - validation.metrics: 完整 EvaluationReport。
    - validation.metrics.six_class: 全候选六分类报告。
    - validation.metrics.six_class.accuracy: 全候选六分类准确率。
    - validation.metrics.six_class.macro_f1: 与正式选模兼容的支持类 macro-F1。
    - validation.metrics.six_class.macro_f1_supported_classes: 只平均真实支持数大于零类别的 macro-F1。
    - validation.metrics.six_class.macro_f1_all_classes: 固定平均六类且把零支持类 F1 记为 0 的 macro-F1。
    - validation.metrics.six_class.per_class: 以六个类别名为键的逐类指标映射。
    - validation.metrics.six_class.per_class[*].precision: 当前类别 precision；零预测时为 NaN。
    - validation.metrics.six_class.per_class[*].recall: 当前类别 recall；零支持时为 NaN。
    - validation.metrics.six_class.per_class[*].f1: 当前类别 F1；无法计算时为 0。
    - validation.metrics.six_class.per_class[*].support: 当前类别真实候选数。
    - validation.metrics.six_class.per_class[*].predicted_count: 当前类别预测候选数。
    - validation.metrics.six_class.confusion_matrix: 形状 6×6 的整数嵌套列表，第一轴真实类、第二轴预测类。
    - validation.metrics.intrinsic_identity: 配置身份范围内忽略六分类预测的身份报告。
    - validation.metrics.intrinsic_identity.top1: 固有身份 Top-1 正确率。
    - validation.metrics.intrinsic_identity.topk: 固有身份 Top-K 正确率。
    - validation.metrics.intrinsic_identity.top_k: 固有身份报告采用的最大名次 K。
    - validation.metrics.intrinsic_identity.eligible_count: 固有身份可监督候选数。
    - validation.metrics.conditional_identity: 只在六分类预测正确时统计的身份报告。
    - validation.metrics.conditional_identity.top1: 条件身份 Top-1 正确率。
    - validation.metrics.conditional_identity.topk: 条件身份 Top-K 正确率。
    - validation.metrics.conditional_identity.top_k: 条件身份报告采用的最大名次 K。
    - validation.metrics.conditional_identity.eligible_count: 条件身份候选数。
    - validation.metrics.conditional_identity.seen_top1: train seen 子集 Top-1。
    - validation.metrics.conditional_identity.seen_topk: train seen 子集 Top-K。
    - validation.metrics.conditional_identity.seen_count: train seen 子集候选数。
    - validation.metrics.conditional_identity.unseen_top1: train unseen 子集 Top-1。
    - validation.metrics.conditional_identity.unseen_topk: train unseen 子集 Top-K。
    - validation.metrics.conditional_identity.unseen_count: train unseen 子集候选数。
    - validation.metrics.end_to_end_identity: 配置身份范围经前景门后的最终身份报告。
    - validation.metrics.end_to_end_identity.precision: 端到端身份 precision。
    - validation.metrics.end_to_end_identity.recall: 端到端身份 recall。
    - validation.metrics.end_to_end_identity.f1: 端到端身份 F1。
    - validation.metrics.end_to_end_identity.true_positive: 通过前景门且最终身份正确的候选数。
    - validation.metrics.end_to_end_identity.predicted_positive: 通过前景门且最终身份属于配置范围的候选数。
    - validation.metrics.end_to_end_identity.actual_positive: 配置范围内具备身份目标的真实候选数。
    - validation.metrics.small_molecule_identity: 只含小分子身份且使用严格小分子分类门的范围报告。
    - validation.metrics.small_molecule_identity.intrinsic: 小分子固有身份报告。
    - validation.metrics.small_molecule_identity.intrinsic.top1: 小分子固有 Top-1。
    - validation.metrics.small_molecule_identity.intrinsic.topk: 小分子固有 Top-K。
    - validation.metrics.small_molecule_identity.intrinsic.top_k: 小分子报告实际采用的最大名次 K。
    - validation.metrics.small_molecule_identity.intrinsic.eligible_count: 小分子固有身份候选数。
    - validation.metrics.small_molecule_identity.conditional: 小分子条件身份报告。
    - validation.metrics.small_molecule_identity.conditional.top1: 小分子条件 Top-1。
    - validation.metrics.small_molecule_identity.conditional.topk: 小分子条件 Top-K。
    - validation.metrics.small_molecule_identity.conditional.top_k: 小分子条件报告实际采用的最大名次 K。
    - validation.metrics.small_molecule_identity.conditional.eligible_count: 小分子条件身份候选数。
    - validation.metrics.small_molecule_identity.end_to_end: 小分子端到端报告。
    - validation.metrics.small_molecule_identity.end_to_end.precision: 小分子端到端 precision。
    - validation.metrics.small_molecule_identity.end_to_end.recall: 小分子端到端 recall。
    - validation.metrics.small_molecule_identity.end_to_end.f1: 小分子端到端 F1。
    - validation.metrics.small_molecule_identity.end_to_end.true_positive: 小分子端到端最终身份正确数。
    - validation.metrics.small_molecule_identity.end_to_end.predicted_positive: 小分子端到端预测阳性数。
    - validation.metrics.small_molecule_identity.end_to_end.actual_positive: 小分子端到端真实阳性数。
    - validation.metrics.all_non_background_identity: 五类前景共享身份清单的范围报告。
    - validation.metrics.all_non_background_identity.intrinsic: 五类前景固有身份报告。
    - validation.metrics.all_non_background_identity.intrinsic.top1: 五类前景固有 Top-1。
    - validation.metrics.all_non_background_identity.intrinsic.topk: 五类前景固有 Top-K。
    - validation.metrics.all_non_background_identity.intrinsic.top_k: 五类前景报告实际采用的最大名次 K。
    - validation.metrics.all_non_background_identity.intrinsic.eligible_count: 五类前景固有身份候选数。
    - validation.metrics.all_non_background_identity.conditional: 五类前景条件身份报告。
    - validation.metrics.all_non_background_identity.conditional.top1: 五类前景条件 Top-1。
    - validation.metrics.all_non_background_identity.conditional.topk: 五类前景条件 Top-K。
    - validation.metrics.all_non_background_identity.conditional.top_k: 五类前景条件报告实际采用的最大名次 K。
    - validation.metrics.all_non_background_identity.conditional.eligible_count: 五类前景条件身份候选数。
    - validation.metrics.all_non_background_identity.end_to_end: 五类前景端到端报告。
    - validation.metrics.all_non_background_identity.end_to_end.precision: 五类前景端到端 precision。
    - validation.metrics.all_non_background_identity.end_to_end.recall: 五类前景端到端 recall。
    - validation.metrics.all_non_background_identity.end_to_end.f1: 五类前景端到端 F1。
    - validation.metrics.all_non_background_identity.end_to_end.true_positive: 五类前景端到端最终身份正确数。
    - validation.metrics.all_non_background_identity.end_to_end.predicted_positive: 五类前景端到端预测阳性数。
    - validation.metrics.all_non_background_identity.end_to_end.actual_positive: 五类前景端到端真实阳性数。
    - validation.metrics.identity_per_class: 以前景类别名为键的逐类身份范围报告。
    - validation.metrics.identity_per_class[*].intrinsic: 当前前景类固有身份报告。
    - validation.metrics.identity_per_class[*].intrinsic.top1: 当前前景类固有 Top-1。
    - validation.metrics.identity_per_class[*].intrinsic.topk: 当前前景类固有 Top-K。
    - validation.metrics.identity_per_class[*].intrinsic.top_k: 当前前景类报告实际采用的最大名次 K。
    - validation.metrics.identity_per_class[*].intrinsic.eligible_count: 当前前景类固有身份候选数。
    - validation.metrics.identity_per_class[*].conditional: 当前前景类条件身份报告。
    - validation.metrics.identity_per_class[*].conditional.top1: 当前前景类条件 Top-1。
    - validation.metrics.identity_per_class[*].conditional.topk: 当前前景类条件 Top-K。
    - validation.metrics.identity_per_class[*].conditional.top_k: 当前前景类条件报告实际采用的最大名次 K。
    - validation.metrics.identity_per_class[*].conditional.eligible_count: 当前前景类条件身份候选数。
    - validation.metrics.identity_per_class[*].end_to_end: 当前前景类端到端报告。
    - validation.metrics.identity_per_class[*].end_to_end.precision: 当前前景类端到端 precision。
    - validation.metrics.identity_per_class[*].end_to_end.recall: 当前前景类端到端 recall。
    - validation.metrics.identity_per_class[*].end_to_end.f1: 当前前景类端到端 F1。
    - validation.metrics.identity_per_class[*].end_to_end.true_positive: 当前前景类端到端最终身份正确数。
    - validation.metrics.identity_per_class[*].end_to_end.predicted_positive: 当前前景类端到端预测阳性数。
    - validation.metrics.identity_per_class[*].end_to_end.actual_positive: 当前前景类端到端真实阳性数。
    - validation.metrics.area: 两个体素区域的微平均 Dice 报告。
    - validation.metrics.area.ligand_micro_dice: 配体区域微平均 Dice。
    - validation.metrics.area.binding_micro_dice: 结合区域微平均 Dice。
    - validation.metrics.area.ligand_dice_denominator: 配体区域预测与真实体素数和。
    - validation.metrics.area.binding_dice_denominator: 结合区域预测与真实体素数和。
    - validation.sample_count: validation 候选数。
    - validation.identity_eligible_count: validation 有效身份候选数。
    - validation.elapsed_seconds: 完整 validation 墙钟秒数。
    - validation.samples_per_second: validation 候选吞吐。
    - timestamp_unix: append_jsonl 前调用 time.time 得到的 Unix 墙钟秒数。
    """

    epoch: int
    global_step: int
    learning_rate_used: float
    next_learning_rate: float
    selection_score: float
    improved: bool
    bad_validation_epochs: int
    training: TrainingEpochResult
    validation: EvaluationResult
    timestamp_unix: float


@dataclass
class _LossStatistics:
    """按真实分母跨物理批次累积损失分子。

    - sample_count: 已累计的候选数。
    - classification_numerator: 各批 classification 乘回该批真实分类分母后的累计分子。
    - classification_denominator: 候选数或真实类别权重和的累计分类分母。
    - ligand_area_sum: 各批 ligand_area 乘物理批量后的累计和。
    - binding_area_sum: 各批 binding_area 乘物理批量后的累计和。
    - identity_numerator: 各批 identity 乘有效身份候选数后的累计分子。
    - identity_eligible_count: 当前身份范围内有效候选累计数。
    - self.sample_count: 运行期已累计的候选数，与 sample_count 字段同值。
    - self.classification_numerator: 运行期分类损失累计分子，与 classification_numerator 字段同值。
    - self.classification_denominator: 运行期分类损失累计分母，与 classification_denominator 字段同值。
    - self.ligand_area_sum: 运行期配体区域损失累计和，与 ligand_area_sum 字段同值。
    - self.binding_area_sum: 运行期结合区域损失累计和，与 binding_area_sum 字段同值。
    - self.identity_numerator: 运行期身份损失累计分子，与 identity_numerator 字段同值。
    - self.identity_eligible_count: 运行期身份损失累计分母，与 identity_eligible_count 字段同值。
    """

    sample_count: int = 0
    classification_numerator: float = 0.0
    classification_denominator: float = 0.0
    ligand_area_sum: float = 0.0
    binding_area_sum: float = 0.0
    identity_numerator: float = 0.0
    identity_eligible_count: int = 0

    def update(self, losses: LossOutput, batch_size: int) -> None:
        """检查并累加一个物理批次的损失与真实分母。

        - losses: 当前物理批次的七字段 LossOutput。
        - losses.total: 浮点零维张量；只做有限性检查，不直接累加。
        - losses.classification: 浮点零维张量；乘 classification_denominator 后加入分类分子。
        - losses.ligand_area: 浮点零维张量；乘 batch_size 后加入配体区域和。
        - losses.binding_area: 浮点零维张量；乘 batch_size 后加入结合区域和。
        - losses.identity: 浮点零维张量；乘 identity_eligible_count 后加入身份分子。
        - losses.classification_denominator: 浮点零维张量；加入全局分类分母。
        - losses.identity_eligible_count: int64 零维张量；必须非负并加入身份分母。
        - batch_size: 正整数语义的当前物理批量候选数；本地只检查大于 0。

        张量转 Python 数值会在 CUDA 路径产生设备同步。任一损失或分类分母含 NaN/Inf 时
        立即失败；本方法不验证 total 与四项分量的加权关系。
        """

        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数。")
        # tuple[tuple[str,Tensor],...]，必须逐项为有限值的六个浮点零维张量。
        finite_components = (
            ("total", losses["total"]),
            ("classification", losses["classification"]),
            ("ligand_area", losses["ligand_area"]),
            ("binding_area", losses["binding_area"]),
            ("identity", losses["identity"]),
            ("classification_denominator", losses["classification_denominator"]),
        )
        for name, value in finite_components:
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"损失分量 {name} 含 NaN 或 Inf。")
        # int，当前物理批次真实身份分母，从设备零维张量同步转换。
        eligible_count = int(losses["identity_eligible_count"].detach())
        if eligible_count < 0:
            raise ValueError("identity_eligible_count 不能为负数。")
        self.sample_count += batch_size
        # float，当前物理批次分类真实分母，从设备零维张量同步转换。
        classification_denominator = float(losses["classification_denominator"].detach())
        self.classification_numerator += (
            float(losses["classification"].detach()) * classification_denominator
        )
        self.classification_denominator += classification_denominator
        self.ligand_area_sum += float(losses["ligand_area"].detach()) * batch_size
        self.binding_area_sum += float(losses["binding_area"].detach()) * batch_size
        self.identity_numerator += float(losses["identity"].detach()) * eligible_count
        self.identity_eligible_count += eligible_count

    def compute(self, weights: LossWeights) -> LossValues:
        """用累计真实分母计算五个全局损失值。

        - weights: 当前阶段的四项非负 LossWeights。
        - weights.classification: 六分类损失系数。
        - weights.ligand_area: 配体区域损失系数。
        - weights.binding_area: 结合区域损失系数。
        - weights.identity: 身份损失系数。

        - 返回值.total: 四项全局损失按 weights 加权之和。
        - 返回值.classification: classification_numerator / classification_denominator。
        - 返回值.ligand_area: ligand_area_sum / sample_count。
        - 返回值.binding_area: binding_area_sum / sample_count。
        - 返回值.identity: identity_numerator / identity_eligible_count；分母为 0 时固定为 0。

        空统计或非正分类分母失败关闭。update 会拒绝每个物理批次的非有限损失张量，配置权重
        也由解析器保证有限；但有限 Python float 的跨批乘加仍可能溢出，本方法不执行最终有限性检查。
        """

        if self.sample_count == 0:
            raise RuntimeError("不能汇总空损失统计。")
        if self.classification_denominator <= 0:
            raise RuntimeError("六分类损失的全局归一分母必须为正数。")
        # float，跨全部候选或类别权重和归一的六分类损失。
        classification = self.classification_numerator / self.classification_denominator
        # float，跨全部候选算术平均的配体区域损失。
        ligand_area = self.ligand_area_sum / self.sample_count
        # float，跨全部候选算术平均的结合区域损失。
        binding_area = self.binding_area_sum / self.sample_count
        # float，跨有效身份候选平均的身份损失；无有效候选时为0。
        identity = (
            self.identity_numerator / self.identity_eligible_count
            if self.identity_eligible_count > 0
            else 0.0
        )
        # float，四项全局损失按配置权重合并的总损失。
        total = (
            weights.classification * classification
            + weights.ligand_area * ligand_area
            + weights.binding_area * binding_area
            + weights.identity * identity
        )
        return {
            "total": total,
            "classification": classification,
            "ligand_area": ligand_area,
            "binding_area": binding_area,
            "identity": identity,
        }


@dataclass
class _StageRuntime:
    """保存阶段构造完成后由所有 epoch 共用的运行对象。

    - model: 已移动到 CUDA 的 CandidateEncoder 或 ClosedSetMatcher。
    - train_loader: train DataLoader；sampler 只保留完整 global batch 所需候选，worker 每次完整迭代后退出。
    - validation_loader: validation DataLoader；不洗牌、不丢弃划分尾部，worker 每次完整迭代后退出。
    - train_sampler: 每轮由 set_epoch 更新洗牌序列的 ShuffledBatchSampler。
    - optimizer: 包含模型全部参数且语义显式冻结的 AdamW。
    - scheduler: 以完整 validation 选择分数为 max 指标的 ReduceLROnPlateau。
    - scaler: torch.amp.GradScaler；只在 precision=fp16 时启用。
    - class_weights: CUDA float32 张量，形状 (6,)，顺序与 CLASS_NAMES 对齐；未配置时为 None。
    - device: 正式运行的 CUDA torch.device。
    - training_config: StageConfig.training 经 asdict 得到且将写入 checkpoint 的普通 dict。
    - training_config.seed: 进程、sampler 与 DataLoader 派生随机种子的非负整数。
    - training_config.physical_batch_size: 每次前向的候选数。
    - training_config.gradient_accumulation: 每次参数更新累计的物理批次数。
    - training_config.global_batch_size: 每次参数更新的候选总数。
    - training_config.learning_rate: AdamW 当前阶段初始学习率。
    - training_config.weight_decay: AdamW 解耦权重衰减。
    - training_config.maximum_epochs: 正整数轮数上限，或无硬上限时为 None。
    - training_config.patience_bad_epochs: 连续未严格改善的早停轮数阈值。
    - training_config.precision: bf16、fp16 或 fp32。
    - training_config.gradient_clip_norm: 全模型 L2 梯度裁剪阈值。
    - training_config.class_weights: 按六分类顺序排列的六项权重列表，或未加权时为 None。
    - training_config.loss_weights: 四项损失系数映射。
    - training_config.loss_weights.classification: 六分类损失系数。
    - training_config.loss_weights.ligand_area: 配体区域损失系数。
    - training_config.loss_weights.binding_area: 结合区域损失系数。
    - training_config.loss_weights.identity: 身份损失系数。
    - resolved_config: 已写入 resolved.json 并浅复制发送给 W&B 的完整解析记录。
    - resolved_config.stage: 规范化阶段配置映射。
    - resolved_config.stage.name: 阶段安全名称。
    - resolved_config.stage.task_name: candidate_pretrain 或 full_matcher。
    - resolved_config.stage.language_family: 语言家族或 None。
    - resolved_config.stage.identity_variant: 身份损失变体或 None。
    - resolved_config.stage.identity_scope: 身份监督范围或 None。
    - resolved_config.stage.model: MatcherModelConfig.to_dict 产生的模型配置映射。
    - resolved_config.stage.model.aggregation: 候选摘要聚合方式。
    - resolved_config.stage.model.use_manual_features: 手工特征开关。
    - resolved_config.stage.model.use_sampling_features: 裁剪统计开关。
    - resolved_config.stage.model.use_candidate_token_attention_bias: 候选—token 注意力偏置开关。
    - resolved_config.stage.model.initial_graph_gate: 图增量门初值。
    - resolved_config.stage.model.initial_inventory_gate: 清单增量门初值。
    - resolved_config.stage.model.model_dim: 共同模型宽度。
    - resolved_config.stage.model.attention_heads: 候选、图与清单注意力共用头数。
    - resolved_config.stage.model.dropout: 模型丢弃概率。
    - resolved_config.stage.model.voxel_feature_channels: RAUNet 输出通道数。
    - resolved_config.stage.model.raunet_planes: 九层 RAUNet 通道列表。
    - resolved_config.stage.model.recycle_iterations: RAUNet 回收轮数。
    - resolved_config.stage.model.gradient_checkpointing: 激活重计算开关。
    - resolved_config.stage.model.bottleneck_attention_layers: RAUNet 瓶颈注意力层数。
    - resolved_config.stage.model.graph_layers: 配体图消息传递层数。
    - resolved_config.stage.training: 与 training_config 同源的训练配置映射。
    - resolved_config.stage.training.seed: 训练随机种子。
    - resolved_config.stage.training.physical_batch_size: 物理批量。
    - resolved_config.stage.training.gradient_accumulation: 梯度累计次数。
    - resolved_config.stage.training.global_batch_size: 全局批量。
    - resolved_config.stage.training.learning_rate: 初始学习率。
    - resolved_config.stage.training.weight_decay: 权重衰减。
    - resolved_config.stage.training.maximum_epochs: 轮数上限或 None。
    - resolved_config.stage.training.patience_bad_epochs: 早停坏轮数阈值。
    - resolved_config.stage.training.precision: 混合精度模式。
    - resolved_config.stage.training.gradient_clip_norm: 梯度裁剪阈值。
    - resolved_config.stage.training.class_weights: 六项类别权重或 None。
    - resolved_config.stage.training.loss_weights: 四项损失系数映射。
    - resolved_config.stage.training.loss_weights.classification: 六分类损失系数。
    - resolved_config.stage.training.loss_weights.ligand_area: 配体区域损失系数。
    - resolved_config.stage.training.loss_weights.binding_area: 结合区域损失系数。
    - resolved_config.stage.training.loss_weights.identity: 身份损失系数。
    - resolved_config.stage.initialize_candidate_from: 候选预训练 BEST.pt 路径字符串或 None。
    - resolved_config.data: 数据根、科学字段、worker 字段与派生进程事实映射。
    - resolved_config.data.data_root: AdaLigand 数据根路径字符串。
    - resolved_config.data.stage1_root: Stage1 Find_0 根路径字符串。
    - resolved_config.data.output_root: Matcher 清单与缓存根路径字符串。
    - resolved_config.data.coverage_threshold: 双向覆盖率阈值。
    - resolved_config.data.selection_seed: occurrence 选择种子。
    - resolved_config.data.pp_top_k: 每候选 PP token 上限。
    - resolved_config.data.local_crop_strategy: 局部裁剪策略。
    - resolved_config.data.train_num_workers: train worker 数。
    - resolved_config.data.validation_num_workers: validation worker 数。
    - resolved_config.data.prefetch_factor: 每 worker 预取批次数。
    - resolved_config.data.cache_gib_per_worker: 每 Dataset 实例缓存 GiB 预算。
    - resolved_config.data.multiprocessing_context: 固定为 spawn。
    - resolved_config.data.persistent_workers: 固定为 False。
    - resolved_config.data.worker_pool_lifecycle: 非持久 worker 生命周期说明。
    - resolved_config.data.train_worker_generator_seed: train loader 生成器初始种子。
    - resolved_config.data.validation_worker_generator_seed: validation loader 生成器初始种子。
    - resolved_config.data.worker_generator_lifecycle: 两个 loader 独立生成器的推进规则说明。
    - resolved_config.tracking: 七项 W&B 正式追踪配置映射。
    - resolved_config.tracking.provider: 固定为 wandb。
    - resolved_config.tracking.mode: 固定为 online。
    - resolved_config.tracking.entity: W&B 账户或团队名。
    - resolved_config.tracking.project: W&B 项目名。
    - resolved_config.tracking.group: W&B 比较组名。
    - resolved_config.tracking.log_every_global_steps: 参数更新曲线记录间隔。
    - resolved_config.tracking.init_timeout_seconds: W&B 初始化超时秒数。
    - resolved_config.device: CUDA 设备名称。
    - resolved_config.torch_version: PyTorch 版本字符串。
    - resolved_config.frozen_optimizer_semantics: 优化器、调度器与 GradScaler 的固定参数映射。
    - resolved_config.frozen_optimizer_semantics.adam_betas: AdamW 一阶与二阶矩衰减二元组。
    - resolved_config.frozen_optimizer_semantics.adam_epsilon: AdamW 分母稳定常数。
    - resolved_config.frozen_optimizer_semantics.adam_amsgrad: 固定为 False。
    - resolved_config.frozen_optimizer_semantics.adam_maximize: 固定为 False。
    - resolved_config.frozen_optimizer_semantics.adam_foreach: 固定为 True。
    - resolved_config.frozen_optimizer_semantics.adam_capturable: 固定为 False。
    - resolved_config.frozen_optimizer_semantics.adam_differentiable: 固定为 False。
    - resolved_config.frozen_optimizer_semantics.adam_fused: 固定为 False。
    - resolved_config.frozen_optimizer_semantics.scheduler: 固定为 ReduceLROnPlateau。
    - resolved_config.frozen_optimizer_semantics.scheduler_mode: 固定为 max。
    - resolved_config.frozen_optimizer_semantics.scheduler_factor: 未改善后学习率乘数。
    - resolved_config.frozen_optimizer_semantics.scheduler_patience: 调度器内部固定容忍轮数 0。
    - resolved_config.frozen_optimizer_semantics.scheduler_threshold: 固定绝对改善阈值 0。
    - resolved_config.frozen_optimizer_semantics.scheduler_threshold_mode: 固定为 abs。
    - resolved_config.frozen_optimizer_semantics.scheduler_cooldown: 固定冷却轮数 0。
    - resolved_config.frozen_optimizer_semantics.scheduler_minimum_learning_rate: 学习率下界。
    - resolved_config.frozen_optimizer_semantics.scheduler_epsilon: 忽略更小学习率变化的阈值。
    - resolved_config.frozen_optimizer_semantics.grad_scaler_initial_scale: fp16 损失缩放初值。
    - resolved_config.frozen_optimizer_semantics.grad_scaler_growth_factor: 无溢出后缩放增长乘数。
    - resolved_config.frozen_optimizer_semantics.grad_scaler_backoff_factor: 溢出后缩放回退乘数。
    - resolved_config.frozen_optimizer_semantics.grad_scaler_growth_interval: 连续无溢出多少次后增长缩放。
    - resolved_config.numeric_backend: 随机源和 PyTorch 数值后端事实记录。
    - resolved_config.numeric_backend.python_numpy_torch_seed: Python、NumPy、CPU 与全部 CUDA 生成器共用种子。
    - resolved_config.numeric_backend.cudnn_benchmark: cuDNN benchmark 开关，固定为 True。
    - resolved_config.numeric_backend.cudnn_deterministic: cuDNN deterministic 开关，固定为 False。
    - resolved_config.numeric_backend.deterministic_algorithms: PyTorch 全局确定性算法开关，固定为 False。
    - resolved_config.numeric_backend.float32_matmul_precision: float32 矩阵乘精度模式，固定为 high。
    - resolved_config.initialization: 候选初始化追溯记录或从零训练时的 None。
    - resolved_config.initialization.checkpoint_path: 来源 BEST.pt 解析后的绝对 POSIX 路径。
    - resolved_config.initialization.source_plan_name: 来源训练计划名。
    - resolved_config.initialization.source_stage_name: 来源候选预训练阶段名。
    - resolved_config.initialization.source_run_stamp: 来源运行目录标识。
    - resolved_config.initialization.source_slurm_job_id: 来源 Slurm Job ID 字符串。
    - resolved_config.initialization.source_epoch: 来源 BEST checkpoint 的零起始 epoch。
    - resolved_config.initialization.source_global_step: 来源 checkpoint 累计参数更新数。
    - resolved_config.initialization.source_best_score: 来源 validation 六分类支持类 macro-F1。
    - resolved_config.initialization.source_tracking_run_id: 来源 W&B run ID。
    - resolved_config.initialization.source_tracking_url: 来源 W&B HTTPS 地址。
    - resolved_config.initialization.source_candidate_model_config: 来源候选编码器科学结构映射。
    - resolved_config.initialization.source_candidate_model_config.aggregation: 来源候选摘要聚合方式。
    - resolved_config.initialization.source_candidate_model_config.use_manual_features: 来源手工特征开关。
    - resolved_config.initialization.source_candidate_model_config.use_sampling_features: 来源裁剪统计开关。
    - resolved_config.initialization.source_candidate_model_config.use_candidate_token_attention_bias: 来源候选—token 注意力偏置开关。
    - resolved_config.initialization.source_candidate_model_config.model_dim: 来源候选表示宽度。
    - resolved_config.initialization.source_candidate_model_config.attention_heads: 来源候选注意力头数。
    - resolved_config.initialization.source_candidate_model_config.dropout: 来源丢弃概率。
    - resolved_config.initialization.source_candidate_model_config.voxel_feature_channels: 来源 RAUNet 输出通道数。
    - resolved_config.initialization.source_candidate_model_config.raunet_planes: 来源九项 RAUNet 通道序列。
    - resolved_config.initialization.source_candidate_model_config.recycle_iterations: 来源回收轮数。
    - resolved_config.initialization.source_candidate_model_config.gradient_checkpointing: 来源激活重计算开关。
    - resolved_config.initialization.source_candidate_model_config.bottleneck_attention_layers: 来源瓶颈注意力层数。
    - resolved_config.initialization.source_scientific_data: 来源七项科学数据映射。
    - resolved_config.initialization.source_scientific_data.data_root: 来源 AdaLigand 数据根。
    - resolved_config.initialization.source_scientific_data.stage1_root: 来源 Stage1 Find_0 根。
    - resolved_config.initialization.source_scientific_data.output_root: 来源 Matcher 清单与缓存根。
    - resolved_config.initialization.source_scientific_data.coverage_threshold: 来源双向覆盖率阈值。
    - resolved_config.initialization.source_scientific_data.selection_seed: 来源 occurrence 选择种子。
    - resolved_config.initialization.source_scientific_data.pp_top_k: 来源 PP token 上限。
    - resolved_config.initialization.source_scientific_data.local_crop_strategy: 来源局部裁剪策略。
    - resolved_config.initialization.resolved_path: 来源阶段 resolved.json 绝对 POSIX 路径。
    - resolved_config.initialization.stage_summary_path: 来源阶段 summary.json 绝对 POSIX 路径。
    - resolved_config.initialization.run_record_path: 来源根 run.json 绝对 POSIX 路径。
    - resolved_config.initialization.run_summary_path: 来源根 summary.json 绝对 POSIX 路径。
    - resolved_config.initialization.training_plan_path: 来源训练 YAML 绝对 POSIX 路径。
    - resolved_config.initialization.missing_keys: 严格加载成功后的空缺失参数名列表。
    - resolved_config.initialization.unexpected_keys: 严格加载成功后的空额外参数名列表。
    - resolved_config.train_candidate_count: train 清单候选总数。
    - resolved_config.train_candidates_used_per_epoch: 每轮截断后实际训练候选数。
    - resolved_config.validation_candidate_count: validation 清单候选总数。
    """

    model: nn.Module
    train_loader: DataLoader[CandidateBatch]
    validation_loader: DataLoader[CandidateBatch]
    train_sampler: ShuffledBatchSampler
    optimizer: AdamW
    scheduler: ReduceLROnPlateau
    scaler: torch.amp.GradScaler
    class_weights: torch.Tensor | None
    device: torch.device
    training_config: dict[str, Any]
    resolved_config: dict[str, object]


def _build_stage_runtime(
    stage: StageConfig,
    data: DataConfig,
    tracking: TrackingConfig,
    stage_directory: Path,
) -> _StageRuntime:
    """构造数据、模型与优化状态，并发布阶段 resolved.json。

    - stage: 当前阶段的完整 StageConfig。
    - stage.name: 阶段安全名称，保存在解析记录中。
    - stage.task_name: candidate_pretrain 构造 CandidateEncoder，full_matcher 构造 ClosedSetMatcher。
    - stage.language_family: 传给两个 CandidateDataset；候选预训练为 None。
    - stage.identity_variant: 模型构造不读取，保存在解析记录并供损失函数使用。
    - stage.identity_scope: 传给两个 CandidateDataset并保存在解析记录。
    - stage.model: 构造 CandidateEncoder 或 ClosedSetMatcher 的完整 MatcherModelConfig。
    - stage.model.aggregation: mean_max 或 candidate_cross_attention 候选摘要聚合方式。
    - stage.model.use_manual_features: 12 维手工特征开关。
    - stage.model.use_sampling_features: 4 维裁剪统计开关。
    - stage.model.use_candidate_token_attention_bias: 候选—token 注意力偏置开关。
    - stage.model.initial_graph_gate: 图分数增量门初值。
    - stage.model.initial_inventory_gate: 同 PDB 身份清单增量门初值。
    - stage.model.model_dim: 候选、语言与图身份共同表示宽度。
    - stage.model.attention_heads: 候选、配体图和身份清单注意力共用头数。
    - stage.model.dropout: 模型共享丢弃概率。
    - stage.model.voxel_feature_channels: RAUNet 输出和两个体素头输入通道数。
    - stage.model.raunet_planes: 九项 RAUNet 通道序列。
    - stage.model.recycle_iterations: RAUNet 回收轮数。
    - stage.model.gradient_checkpointing: 高开销 RAUNet 子模块激活重计算开关。
    - stage.model.bottleneck_attention_layers: RAUNet 瓶颈注意力层数。
    - stage.model.graph_layers: 配体图消息传递层数。
    - stage.training: 构造优化器、scaler、sampler 与解析记录的完整 TrainingConfig。
    - stage.training.seed: 配置进程随机源、sampler 及 train/validation worker 生成器派生种子。
    - stage.training.physical_batch_size: 两个 sampler 的物理批量。
    - stage.training.gradient_accumulation: 本函数只保存；训练循环据此组合物理批次。
    - stage.training.global_batch_size: train sampler 的 sample_count_multiple。
    - stage.training.learning_rate: AdamW 初始学习率。
    - stage.training.weight_decay: AdamW 解耦权重衰减。
    - stage.training.maximum_epochs: 本函数只保存；阶段循环据此限制轮数。
    - stage.training.patience_bad_epochs: 本函数只保存；阶段循环据此早停。
    - stage.training.precision: 决定 GradScaler 是否启用，autocast 由训练与评估循环使用。
    - stage.training.gradient_clip_norm: 本函数只保存；训练循环用于全模型梯度裁剪。
    - stage.training.class_weights: None，或转换成 CUDA float32 (6,) 张量。
    - stage.training.loss_weights: 本函数只保存；损失与统计循环读取四项系数。
    - stage.training.loss_weights.classification: 六分类损失系数。
    - stage.training.loss_weights.ligand_area: 配体区域损失系数。
    - stage.training.loss_weights.binding_area: 结合区域损失系数。
    - stage.training.loss_weights.identity: 身份损失系数。
    - stage.initialize_candidate_from: None，或在 model.to(device) 与 AdamW 构造前完成来源核验和候选编码器加载。
    - data: 当前实验的完整 DataConfig。
    - data.roots: 传给 Dataset、Store 与 manifest 入口的三个 SourceRoots。
    - data.roots.data_root: Dataset 及其 Store 的 AdaLigand 来源根。
    - data.roots.stage1_root: Dataset 及其 Store 的 Stage1 Find_0 来源根。
    - data.roots.output_root: ensure_manifest、候选体素和 PP token 缓存根。
    - data.coverage_threshold: ensure_manifest 的双向覆盖率阈值。
    - data.selection_seed: ensure_manifest 的 occurrence 语言表示选择种子。
    - data.pp_top_k: 两个 Dataset 的每候选 PP token 上限。
    - data.local_crop_strategy: 两个 Dataset 与缓存共用的局部裁剪策略。
    - data.train_num_workers: train DataLoader worker 数。
    - data.validation_num_workers: validation DataLoader worker 数。
    - data.prefetch_factor: 两个 DataLoader 的每 worker 预取批次数。
    - data.cache_gib_per_worker: 转为 int(GiB*1024**3) 后传给每个 Dataset 的五类 LRU 总字节预算。
    - tracking: 当前实验的完整 TrackingConfig；本函数只写入解析记录，不启动 W&B。
    - tracking.provider: 原样写入 resolved.json。
    - tracking.mode: 原样写入 resolved.json。
    - tracking.entity: 原样写入 resolved.json。
    - tracking.project: 原样写入 resolved.json。
    - tracking.group: 原样写入 resolved.json。
    - tracking.log_every_global_steps: 原样写入 resolved.json。
    - tracking.init_timeout_seconds: 原样写入 resolved.json。
    - stage_directory: 已由调用方创建的新阶段目录；本函数发布 resolved.json，不创建或复用该目录。

    - 返回值.model: 已完成可选候选初始化并移动到 CUDA 的任务模型。
    - 返回值.train_loader: train CandidateBatch loader。
    - 返回值.validation_loader: validation CandidateBatch loader。
    - 返回值.train_sampler: 可在每轮调用 set_epoch 的训练 sampler。
    - 返回值.optimizer: 显式参数化 AdamW。
    - 返回值.scheduler: 监控 validation 选择分数的 ReduceLROnPlateau。
    - 返回值.scaler: fp16 启用、其它精度禁用的 GradScaler。
    - 返回值.class_weights: CUDA float32 (6,) 类别权重或 None。
    - 返回值.device: CUDA device。
    - 返回值.training_config: checkpoint 使用的训练配置普通 dict。
    - 返回值.training_config.seed: 训练随机种子。
    - 返回值.training_config.physical_batch_size: 物理批量。
    - 返回值.training_config.gradient_accumulation: 梯度累计次数。
    - 返回值.training_config.global_batch_size: 每次参数更新的候选数。
    - 返回值.training_config.learning_rate: AdamW 初始学习率。
    - 返回值.training_config.weight_decay: AdamW 权重衰减。
    - 返回值.training_config.maximum_epochs: 轮数上限或 None。
    - 返回值.training_config.patience_bad_epochs: 早停坏轮数阈值。
    - 返回值.training_config.precision: bf16、fp16 或 fp32。
    - 返回值.training_config.gradient_clip_norm: 全模型 L2 梯度裁剪阈值。
    - 返回值.training_config.class_weights: 六项分类权重列表或 None。
    - 返回值.training_config.loss_weights: 四项损失系数映射。
    - 返回值.training_config.loss_weights.classification: 六分类损失系数。
    - 返回值.training_config.loss_weights.ligand_area: 配体区域损失系数。
    - 返回值.training_config.loss_weights.binding_area: 结合区域损失系数。
    - 返回值.training_config.loss_weights.identity: 身份损失系数。
    - 返回值.resolved_config: 与 resolved.json 同源的完整普通 dict。

    - 返回值.resolved_config.stage: StageConfig 经 asdict 得到的映射，其中 model 由 MatcherModelConfig.to_dict 规范化。
    - 返回值.resolved_config.stage.name: 阶段安全名称。
    - 返回值.resolved_config.stage.task_name: candidate_pretrain 或 full_matcher。
    - 返回值.resolved_config.stage.language_family: 语言家族或 None。
    - 返回值.resolved_config.stage.identity_variant: 身份损失变体或 None。
    - 返回值.resolved_config.stage.identity_scope: 身份监督范围或 None。
    - 返回值.resolved_config.stage.model: 完整模型配置映射。
    - 返回值.resolved_config.stage.model.aggregation: 候选摘要聚合方式。
    - 返回值.resolved_config.stage.model.use_manual_features: 手工特征开关。
    - 返回值.resolved_config.stage.model.use_sampling_features: 裁剪统计开关。
    - 返回值.resolved_config.stage.model.use_candidate_token_attention_bias: 候选—token 注意力偏置开关。
    - 返回值.resolved_config.stage.model.initial_graph_gate: 图分数增量门初值。
    - 返回值.resolved_config.stage.model.initial_inventory_gate: 清单分数增量门初值。
    - 返回值.resolved_config.stage.model.model_dim: 共同模型宽度。
    - 返回值.resolved_config.stage.model.attention_heads: 候选、图与清单注意力共用头数。
    - 返回值.resolved_config.stage.model.dropout: 模型丢弃概率。
    - 返回值.resolved_config.stage.model.voxel_feature_channels: RAUNet 输出通道数。
    - 返回值.resolved_config.stage.model.raunet_planes: 九项 RAUNet 通道列表。
    - 返回值.resolved_config.stage.model.recycle_iterations: RAUNet 回收轮数。
    - 返回值.resolved_config.stage.model.gradient_checkpointing: 激活重计算开关。
    - 返回值.resolved_config.stage.model.bottleneck_attention_layers: RAUNet 瓶颈注意力层数。
    - 返回值.resolved_config.stage.model.graph_layers: 配体图消息传递层数。
    - 返回值.resolved_config.stage.training: 与返回值.training_config 同源的训练配置映射。
    - 返回值.resolved_config.stage.training.seed: 训练随机种子。
    - 返回值.resolved_config.stage.training.physical_batch_size: 物理批量。
    - 返回值.resolved_config.stage.training.gradient_accumulation: 梯度累计次数。
    - 返回值.resolved_config.stage.training.global_batch_size: 每次参数更新的候选数。
    - 返回值.resolved_config.stage.training.learning_rate: 初始学习率。
    - 返回值.resolved_config.stage.training.weight_decay: 权重衰减。
    - 返回值.resolved_config.stage.training.maximum_epochs: 轮数上限或 None。
    - 返回值.resolved_config.stage.training.patience_bad_epochs: 早停坏轮数阈值。
    - 返回值.resolved_config.stage.training.precision: bf16、fp16 或 fp32。
    - 返回值.resolved_config.stage.training.gradient_clip_norm: 梯度裁剪阈值。
    - 返回值.resolved_config.stage.training.class_weights: 六项分类权重或 None。
    - 返回值.resolved_config.stage.training.loss_weights: 四项损失系数映射。
    - 返回值.resolved_config.stage.training.loss_weights.classification: 六分类损失系数。
    - 返回值.resolved_config.stage.training.loss_weights.ligand_area: 配体区域损失系数。
    - 返回值.resolved_config.stage.training.loss_weights.binding_area: 结合区域损失系数。
    - 返回值.resolved_config.stage.training.loss_weights.identity: 身份损失系数。
    - 返回值.resolved_config.stage.initialize_candidate_from: 候选预训练 BEST.pt 路径字符串或 None。
    - 返回值.resolved_config.data: 十一项 DataConfig 字段加六项 DataLoader 派生事实。
    - 返回值.resolved_config.data.data_root: data.roots.data_root 路径字符串。
    - 返回值.resolved_config.data.stage1_root: data.roots.stage1_root 路径字符串。
    - 返回值.resolved_config.data.output_root: data.roots.output_root 路径字符串。
    - 返回值.resolved_config.data.coverage_threshold: 双向覆盖率阈值。
    - 返回值.resolved_config.data.selection_seed: occurrence 选择种子。
    - 返回值.resolved_config.data.pp_top_k: PP token 上限。
    - 返回值.resolved_config.data.local_crop_strategy: 局部裁剪策略。
    - 返回值.resolved_config.data.train_num_workers: train worker 数。
    - 返回值.resolved_config.data.validation_num_workers: validation worker 数。
    - 返回值.resolved_config.data.prefetch_factor: 每 worker 预取批次数。
    - 返回值.resolved_config.data.cache_gib_per_worker: 每 Dataset 实例缓存 GiB 预算。
    - 返回值.resolved_config.data.multiprocessing_context: 固定为 spawn。
    - 返回值.resolved_config.data.persistent_workers: 固定为 False。
    - 返回值.resolved_config.data.worker_pool_lifecycle: 固定为 per_complete_loader_iteration。
    - 返回值.resolved_config.data.train_worker_generator_seed: 2 * stage.training.seed。
    - 返回值.resolved_config.data.validation_worker_generator_seed: 2 * stage.training.seed + 1。
    - 返回值.resolved_config.data.worker_generator_lifecycle: 两个 loader 独立生成器的推进规则说明。
    - 返回值.resolved_config.tracking: TrackingConfig 经 asdict 得到的七字段映射。
    - 返回值.resolved_config.tracking.provider: 固定为 wandb。
    - 返回值.resolved_config.tracking.mode: 固定为 online。
    - 返回值.resolved_config.tracking.entity: W&B 账户或团队名。
    - 返回值.resolved_config.tracking.project: W&B 项目名。
    - 返回值.resolved_config.tracking.group: W&B 比较组名。
    - 返回值.resolved_config.tracking.log_every_global_steps: 参数更新曲线记录间隔。
    - 返回值.resolved_config.tracking.init_timeout_seconds: W&B 初始化超时秒数。
    - 返回值.resolved_config.device: torch.cuda.get_device_name 返回的设备名称字符串。
    - 返回值.resolved_config.torch_version: 当前 torch.__version__ 字符串。
    - 返回值.resolved_config.frozen_optimizer_semantics: AdamW、调度器与 GradScaler 固定参数映射。
    - 返回值.resolved_config.frozen_optimizer_semantics.adam_betas: AdamW 一阶与二阶矩衰减二元组。
    - 返回值.resolved_config.frozen_optimizer_semantics.adam_epsilon: AdamW 分母稳定常数。
    - 返回值.resolved_config.frozen_optimizer_semantics.adam_amsgrad: 固定为 False。
    - 返回值.resolved_config.frozen_optimizer_semantics.adam_maximize: 固定为 False。
    - 返回值.resolved_config.frozen_optimizer_semantics.adam_foreach: 固定为 True。
    - 返回值.resolved_config.frozen_optimizer_semantics.adam_capturable: 固定为 False。
    - 返回值.resolved_config.frozen_optimizer_semantics.adam_differentiable: 固定为 False。
    - 返回值.resolved_config.frozen_optimizer_semantics.adam_fused: 固定为 False。
    - 返回值.resolved_config.frozen_optimizer_semantics.scheduler: 固定为 ReduceLROnPlateau。
    - 返回值.resolved_config.frozen_optimizer_semantics.scheduler_mode: 固定为 max。
    - 返回值.resolved_config.frozen_optimizer_semantics.scheduler_factor: 未改善后的学习率乘数。
    - 返回值.resolved_config.frozen_optimizer_semantics.scheduler_patience: 固定为 0。
    - 返回值.resolved_config.frozen_optimizer_semantics.scheduler_threshold: 固定为 0。
    - 返回值.resolved_config.frozen_optimizer_semantics.scheduler_threshold_mode: 固定为 abs。
    - 返回值.resolved_config.frozen_optimizer_semantics.scheduler_cooldown: 固定为 0。
    - 返回值.resolved_config.frozen_optimizer_semantics.scheduler_minimum_learning_rate: 学习率下界。
    - 返回值.resolved_config.frozen_optimizer_semantics.scheduler_epsilon: 忽略更小学习率变化的阈值。
    - 返回值.resolved_config.frozen_optimizer_semantics.grad_scaler_initial_scale: fp16 损失缩放初值。
    - 返回值.resolved_config.frozen_optimizer_semantics.grad_scaler_growth_factor: 连续无溢出后的增长乘数。
    - 返回值.resolved_config.frozen_optimizer_semantics.grad_scaler_backoff_factor: 溢出后的回退乘数。
    - 返回值.resolved_config.frozen_optimizer_semantics.grad_scaler_growth_interval: 增长缩放前要求的连续无溢出次数。
    - 返回值.resolved_config.numeric_backend: NumericBackendRecord 五字段事实映射。
    - 返回值.resolved_config.numeric_backend.python_numpy_torch_seed: Python、NumPy、CPU 与 CUDA 共用种子。
    - 返回值.resolved_config.numeric_backend.cudnn_benchmark: 固定为 True。
    - 返回值.resolved_config.numeric_backend.cudnn_deterministic: 固定为 False。
    - 返回值.resolved_config.numeric_backend.deterministic_algorithms: 固定为 False。
    - 返回值.resolved_config.numeric_backend.float32_matmul_precision: 固定为 high。
    - 返回值.resolved_config.initialization: CandidateInitializationRecord，或从零训练时为 None。
    - 返回值.resolved_config.initialization.checkpoint_path: 来源 BEST.pt 的绝对 POSIX 路径。
    - 返回值.resolved_config.initialization.source_plan_name: 来源训练计划名。
    - 返回值.resolved_config.initialization.source_stage_name: 来源候选预训练阶段名。
    - 返回值.resolved_config.initialization.source_run_stamp: 来源运行目录标识。
    - 返回值.resolved_config.initialization.source_slurm_job_id: 来源 Slurm Job ID 字符串。
    - 返回值.resolved_config.initialization.source_epoch: 来源 BEST checkpoint 的零起始 epoch。
    - 返回值.resolved_config.initialization.source_global_step: 来源 checkpoint 累计参数更新数。
    - 返回值.resolved_config.initialization.source_best_score: 来源 validation 支持类 macro-F1。
    - 返回值.resolved_config.initialization.source_tracking_run_id: 来源 W&B run ID。
    - 返回值.resolved_config.initialization.source_tracking_url: 来源 W&B HTTPS 地址。
    - 返回值.resolved_config.initialization.source_candidate_model_config: 来源候选编码器结构映射。
    - 返回值.resolved_config.initialization.source_candidate_model_config.aggregation: 来源候选摘要聚合方式。
    - 返回值.resolved_config.initialization.source_candidate_model_config.use_manual_features: 来源手工特征开关。
    - 返回值.resolved_config.initialization.source_candidate_model_config.use_sampling_features: 来源裁剪统计开关。
    - 返回值.resolved_config.initialization.source_candidate_model_config.use_candidate_token_attention_bias: 来源候选—token 注意力偏置开关。
    - 返回值.resolved_config.initialization.source_candidate_model_config.model_dim: 来源候选表示宽度。
    - 返回值.resolved_config.initialization.source_candidate_model_config.attention_heads: 来源候选注意力头数。
    - 返回值.resolved_config.initialization.source_candidate_model_config.dropout: 来源模型丢弃概率。
    - 返回值.resolved_config.initialization.source_candidate_model_config.voxel_feature_channels: 来源 RAUNet 输出通道数。
    - 返回值.resolved_config.initialization.source_candidate_model_config.raunet_planes: 来源九项 RAUNet 通道列表。
    - 返回值.resolved_config.initialization.source_candidate_model_config.recycle_iterations: 来源 RAUNet 回收轮数。
    - 返回值.resolved_config.initialization.source_candidate_model_config.gradient_checkpointing: 来源激活重计算开关。
    - 返回值.resolved_config.initialization.source_candidate_model_config.bottleneck_attention_layers: 来源 RAUNet 瓶颈注意力层数。
    - 返回值.resolved_config.initialization.source_scientific_data: 来源七项科学数据映射。
    - 返回值.resolved_config.initialization.source_scientific_data.data_root: 来源 AdaLigand 数据根。
    - 返回值.resolved_config.initialization.source_scientific_data.stage1_root: 来源 Stage1 Find_0 根。
    - 返回值.resolved_config.initialization.source_scientific_data.output_root: 来源 Matcher 清单与缓存根。
    - 返回值.resolved_config.initialization.source_scientific_data.coverage_threshold: 来源双向覆盖率阈值。
    - 返回值.resolved_config.initialization.source_scientific_data.selection_seed: 来源 occurrence 选择种子。
    - 返回值.resolved_config.initialization.source_scientific_data.pp_top_k: 来源 PP token 上限。
    - 返回值.resolved_config.initialization.source_scientific_data.local_crop_strategy: 来源局部裁剪策略。
    - 返回值.resolved_config.initialization.resolved_path: 来源阶段 resolved.json 绝对 POSIX 路径。
    - 返回值.resolved_config.initialization.stage_summary_path: 来源阶段 summary.json 绝对 POSIX 路径。
    - 返回值.resolved_config.initialization.run_record_path: 来源根 run.json 绝对 POSIX 路径。
    - 返回值.resolved_config.initialization.run_summary_path: 来源根 summary.json 绝对 POSIX 路径。
    - 返回值.resolved_config.initialization.training_plan_path: 来源训练 YAML 绝对 POSIX 路径。
    - 返回值.resolved_config.initialization.missing_keys: 严格加载成功后的空缺失参数名列表。
    - 返回值.resolved_config.initialization.unexpected_keys: 严格加载成功后的空额外参数名列表。
    - 返回值.resolved_config.train_candidate_count: train Dataset 清单候选总数。
    - 返回值.resolved_config.train_candidates_used_per_epoch: 截断到 global_batch_size 整倍数的每轮训练候选数。
    - 返回值.resolved_config.validation_candidate_count: validation Dataset 清单候选总数。

    resolved.json 是返回值.resolved_config 在返回前经 write_json 发布的 JSON 表示；两者字段和值同源。

    ensure_manifest 可能构建或复用 train/validation 清单；Dataset 构造会读取清单，但本函数
    不迭代 Dataset，因而不触发候选体素或 PP 缓存构建，也不启动 DataLoader worker。W&B 初始化
    由调用方在本函数返回后执行。CUDA 不可用时失败，不回退 CPU，也不读取 calibration。
    """

    # NumericBackendRecord，同时完成进程随机源与PyTorch数值后端设置并保留五字段事实。
    numeric_backend = configure_random_seeds_and_numeric_backend(stage.training.seed)
    # torch.device，优先当前默认CUDA设备；无CUDA时构造cpu并立即失败关闭。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("正式 Matcher 训练必须在 CUDA GPU 上运行。")
    # Path，覆盖train与validation的schema v4 manifest.json；缺失时可由ensure_manifest构建。
    manifest_path = ensure_manifest(
        data.roots,
        data.coverage_threshold,
        data.selection_seed,
        ("train", "validation"),
    )
    # int，每个Dataset实例五类LRU共用的总字节预算；GiB换算后向零截断。
    cache_bytes_per_worker = int(data.cache_gib_per_worker * 1024**3)
    # CandidateDataset，冻结train候选与身份记录并持有按需数据Store；尚未访问候选或写缓存。
    train_dataset = CandidateDataset(
        roots=data.roots,
        manifest_path=manifest_path,
        split_name="train",
        task_name=stage.task_name,
        language_family=stage.language_family,
        identity_scope=stage.identity_scope,
        pp_top_k=data.pp_top_k,
        local_crop_strategy=data.local_crop_strategy,
        cache_bytes_per_worker=cache_bytes_per_worker,
    )
    # CandidateDataset，冻结validation候选与身份记录并持有独立按需数据Store。
    validation_dataset = CandidateDataset(
        roots=data.roots,
        manifest_path=manifest_path,
        split_name="validation",
        task_name=stage.task_name,
        language_family=stage.language_family,
        identity_scope=stage.identity_scope,
        pp_top_k=data.pp_top_k,
        local_crop_strategy=data.local_crop_strategy,
        cache_bytes_per_worker=cache_bytes_per_worker,
    )
    # ShuffledBatchSampler，每轮洗牌并把保留候选数截断到global_batch_size整数倍。
    train_sampler = ShuffledBatchSampler(
        train_dataset,
        batch_size=stage.training.physical_batch_size,
        shuffle=True,
        drop_last=True,
        seed=stage.training.seed,
        sample_count_multiple=stage.training.global_batch_size,
    )
    # ShuffledBatchSampler，保持validation清单顺序且保留尾部不满物理批量的候选。
    validation_sampler = ShuffledBatchSampler(
        validation_dataset,
        batch_size=stage.training.physical_batch_size,
        shuffle=False,
        drop_last=False,
        seed=stage.training.seed,
        sample_count_multiple=None,
    )
    # int，train DataLoader自有torch.Generator的初始种子，与validation生成器分离。
    train_worker_generator_seed = 2 * stage.training.seed
    # int，validation DataLoader自有torch.Generator的初始种子，与train生成器分离。
    validation_worker_generator_seed = 2 * stage.training.seed + 1
    # DataLoader[CandidateBatch]，spawn worker、非持久进程的train物理批次加载器。
    train_loader = build_candidate_loader(
        train_dataset,
        train_sampler,
        data.train_num_workers,
        data.prefetch_factor,
        train_worker_generator_seed,
    )
    # DataLoader[CandidateBatch]，spawn worker、非持久进程的validation物理批次加载器。
    validation_loader = build_candidate_loader(
        validation_dataset,
        validation_sampler,
        data.validation_num_workers,
        data.prefetch_factor,
        validation_worker_generator_seed,
    )

    # nn.Module，按任务构造的候选编码器或完整闭集Matcher，当前仍位于默认构造设备。
    model: nn.Module
    if stage.task_name == "candidate_pretrain":
        model = CandidateEncoder(stage.model)
    else:
        model = ClosedSetMatcher(stage.model)
    # CandidateInitializationRecord | None，候选预训练来源的完整交叉核对记录；从零训练为None。
    initialization: CandidateInitializationRecord | None = None
    if stage.initialize_candidate_from is not None:
        # Path，YAML字符串按POSIX绝对BEST.pt形式解析的来源checkpoint路径。
        checkpoint_path = _resolve_initial_checkpoint(
            stage.initialize_candidate_from,
        )
        initialization = initialize_candidate_encoder_from_pretraining(
            model,
            checkpoint_path,
            stage,
            data,
        )
    model.to(device)
    # AdamW，覆盖model.parameters()全部参数，固定实现开关与数值常量。
    optimizer = AdamW(
        model.parameters(),
        lr=stage.training.learning_rate,
        weight_decay=stage.training.weight_decay,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
        amsgrad=False,
        maximize=False,
        foreach=True,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    # ReduceLROnPlateau，以越大越好的完整validation选择分数逐轮更新学习率。
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=0,
        threshold=0.0,
        threshold_mode="abs",
        cooldown=0,
        min_lr=SCHEDULER_MINIMUM_LEARNING_RATE,
        eps=SCHEDULER_EPSILON,
    )
    # GradScaler，fp16时启用；bf16/fp32时对象存在但不执行缩放。
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=stage.training.precision == "fp16",
        init_scale=GRAD_SCALER_INITIAL_SCALE,
        growth_factor=GRAD_SCALER_GROWTH_FACTOR,
        backoff_factor=GRAD_SCALER_BACKOFF_FACTOR,
        growth_interval=GRAD_SCALER_GROWTH_INTERVAL,
    )
    # float32张量 (6,) | None，位于训练CUDA设备并与CLASS_NAMES顺序对齐。
    class_weights = (
        None
        if stage.training.class_weights is None
        else torch.tensor(stage.training.class_weights, device=device, dtype=torch.float32)
    )
    # dict[str,object]，StageConfig递归转成普通容器后的完整阶段记录。
    stage_record = asdict(stage)
    stage_record["model"] = stage.model.to_dict()
    # dict[str,object]，十一项数据配置与六项DataLoader进程派生事实。
    data_record = {
        "data_root": str(data.roots.data_root),
        "stage1_root": str(data.roots.stage1_root),
        "output_root": str(data.roots.output_root),
        "coverage_threshold": data.coverage_threshold,
        "selection_seed": data.selection_seed,
        "pp_top_k": data.pp_top_k,
        "local_crop_strategy": data.local_crop_strategy,
        "train_num_workers": data.train_num_workers,
        "validation_num_workers": data.validation_num_workers,
        "prefetch_factor": data.prefetch_factor,
        "cache_gib_per_worker": data.cache_gib_per_worker,
        "multiprocessing_context": DATALOADER_MULTIPROCESSING_CONTEXT,
        "persistent_workers": False,
        "worker_pool_lifecycle": "per_complete_loader_iteration",
        "train_worker_generator_seed": train_worker_generator_seed,
        "validation_worker_generator_seed": validation_worker_generator_seed,
        "worker_generator_lifecycle": (
            "one generator per loader; iterator creation advances only that generator"
        ),
    }
    # object，stage_record.training未验证节点；正常asdict路径应为普通dict。
    training_record = stage_record["training"]
    if not isinstance(training_record, dict):
        raise RuntimeError("阶段训练配置序列化后必须是字典。")
    # dict[str,object]，写入resolved.json并传给W&B config的完整阶段解析记录。
    resolved: dict[str, object] = {
        "stage": stage_record,
        "data": data_record,
        "tracking": asdict(tracking),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "frozen_optimizer_semantics": {
            "adam_betas": ADAM_BETAS,
            "adam_epsilon": ADAM_EPSILON,
            "adam_amsgrad": False,
            "adam_maximize": False,
            "adam_foreach": True,
            "adam_capturable": False,
            "adam_differentiable": False,
            "adam_fused": False,
            "scheduler": "ReduceLROnPlateau",
            "scheduler_mode": "max",
            "scheduler_factor": SCHEDULER_FACTOR,
            "scheduler_patience": 0,
            "scheduler_threshold": 0.0,
            "scheduler_threshold_mode": "abs",
            "scheduler_cooldown": 0,
            "scheduler_minimum_learning_rate": SCHEDULER_MINIMUM_LEARNING_RATE,
            "scheduler_epsilon": SCHEDULER_EPSILON,
            "grad_scaler_initial_scale": GRAD_SCALER_INITIAL_SCALE,
            "grad_scaler_growth_factor": GRAD_SCALER_GROWTH_FACTOR,
            "grad_scaler_backoff_factor": GRAD_SCALER_BACKOFF_FACTOR,
            "grad_scaler_growth_interval": GRAD_SCALER_GROWTH_INTERVAL,
        },
        "numeric_backend": numeric_backend,
        "initialization": initialization,
        "train_candidate_count": len(train_dataset),
        "train_candidates_used_per_epoch": len(train_sampler)
        * stage.training.physical_batch_size,
        "validation_candidate_count": len(validation_dataset),
    }
    write_json(stage_directory / "resolved.json", resolved)
    return _StageRuntime(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_sampler=train_sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        class_weights=class_weights,
        device=device,
        training_config=training_record,
        resolved_config=resolved,
    )


def _train_epoch(
    model: nn.Module,
    loader: DataLoader[CandidateBatch],
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    stage: StageConfig,
    class_weights: torch.Tensor | None,
    device: torch.device,
    initial_global_step: int,
    tracker: WandbTracker,
) -> TrainingEpochResult:
    """训练一个 epoch，并按完整 global batch 更新参数与记录在线指标。

    - model: CUDA CandidateEncoder 或 ClosedSetMatcher；本函数切换到 train 模式。
    - loader: train CandidateBatch DataLoader；批次数必须是 gradient_accumulation 的正整数倍。
    - optimizer: 覆盖 model 全部参数的 AdamW；进入与每次更新后都用 set_to_none=True 清梯度。
    - scaler: fp16 启用的 GradScaler，或 bf16/fp32 路径中的禁用对象。
    - stage: 当前阶段完整配置。
    - stage.name: 本轮首个参数更新事件使用的阶段名。
    - stage.task_name: 本函数不直接读取；model 与损失配置已体现任务类型。
    - stage.language_family: 本函数不直接读取；已冻结在 loader Dataset 中。
    - stage.identity_variant: 传给损失函数的身份变体或 None。
    - stage.identity_scope: 传给身份 eligibility 与损失函数的范围或 None。
    - stage.model: 本函数不直接读取；model 已按该配置构造。
    - stage.model.aggregation: 不在本函数读取的候选摘要聚合方式。
    - stage.model.use_manual_features: 不在本函数读取的手工特征开关。
    - stage.model.use_sampling_features: 不在本函数读取的裁剪统计开关。
    - stage.model.use_candidate_token_attention_bias: 不在本函数读取的候选—token 注意力偏置开关。
    - stage.model.initial_graph_gate: 不在本函数读取的图门初值。
    - stage.model.initial_inventory_gate: 不在本函数读取的清单门初值。
    - stage.model.model_dim: 不在本函数读取的共同模型宽度。
    - stage.model.attention_heads: 不在本函数读取的候选、图与清单注意力头数。
    - stage.model.dropout: 不在本函数读取的模型丢弃概率。
    - stage.model.voxel_feature_channels: 不在本函数读取的 RAUNet 输出通道数。
    - stage.model.raunet_planes: 不在本函数读取的九项 RAUNet 通道序列。
    - stage.model.recycle_iterations: 不在本函数读取的 RAUNet 回收轮数。
    - stage.model.gradient_checkpointing: 不在本函数读取的激活重计算开关。
    - stage.model.bottleneck_attention_layers: 不在本函数读取的 RAUNet 瓶颈注意力层数。
    - stage.model.graph_layers: 不在本函数读取的配体图消息传递层数。
    - stage.training: 当前训练循环直接读取的完整 TrainingConfig。
    - stage.training.seed: 本函数不直接读取的阶段随机种子。
    - stage.training.physical_batch_size: 本函数不直接读取；loader 已按该物理批量构造。
    - stage.training.gradient_accumulation: 每个 global batch 的物理批次数。
    - stage.training.global_batch_size: 每个参数更新必须精确包含的候选数。
    - stage.training.learning_rate: 本函数不直接读取；optimizer 已按该初值构造。
    - stage.training.weight_decay: 本函数不直接读取；optimizer 已按该值构造。
    - stage.training.maximum_epochs: 本函数不直接读取；外层阶段循环控制轮数。
    - stage.training.patience_bad_epochs: 本函数不直接读取；外层阶段循环控制早停。
    - stage.training.precision: CUDA autocast 模式。
    - stage.training.gradient_clip_norm: unscale 后全模型 L2 梯度裁剪阈值。
    - stage.training.loss_weights: 四项 global batch 损失系数。
    - stage.training.loss_weights.classification: 六分类损失系数。
    - stage.training.loss_weights.ligand_area: 配体区域损失系数。
    - stage.training.loss_weights.binding_area: 结合区域损失系数。
    - stage.training.loss_weights.identity: 身份损失系数。
    - stage.training.class_weights: CPU 端 global batch 分类分母使用的六项元组或 None。
    - stage.initialize_candidate_from: 本函数不直接读取；可选初始化已在 model 构造后完成。
    - class_weights: CUDA float32 (6,) 分类权重或 None，传给物理批次损失。
    - device: 正式 CUDA 设备；用于批次搬运、峰值显存与同步。
    - initial_global_step: 进入本轮前累计完成的非负参数更新数；本地没有单独范围门禁。
    - tracker: 已初始化的阶段 WandbTracker；每次更新后按间隔写曲线。
    - tracker.run_id: 本函数不直接读取。
    - tracker.run_url: 本函数不直接读取。
    - tracker.log_every_global_steps: 由 tracker.log_global_step 内部决定本步是否写入。
    - tracker._run: 本函数不直接读取的 W&B SDK run；由 tracker 方法封装写入。

    - 返回值.loss: 完整 epoch 的五项全局归一损失。
    - 返回值.loss.total: 四项全局损失的加权和。
    - 返回值.loss.classification: 完整 epoch 六分类全局归一损失。
    - 返回值.loss.ligand_area: 完整 epoch 配体区域逐候选均值。
    - 返回值.loss.binding_area: 完整 epoch 结合区域逐候选均值。
    - 返回值.loss.identity: 完整 epoch 有效身份候选均值，无有效候选时为 0。
    - 返回值.sample_count: 本轮实际训练候选数。
    - 返回值.identity_eligible_count: 本轮当前身份范围内有效候选数。
    - 返回值.optimizer_steps: 完整 global batch 数。
    - 返回值.skipped_optimizer_steps: 当前失败关闭策略下固定为 0。
    - 返回值.mean_preclip_gradient_norm: 各更新裁剪前全模型 L2 梯度范数均值。
    - 返回值.maximum_preclip_gradient_norm: 各更新裁剪前全模型 L2 梯度范数最大值。
    - 返回值.mean_graph_branch_gradient_norm: 各更新裁剪后图分支梯度范数均值或 None。
    - 返回值.mean_inventory_branch_gradient_norm: 各更新裁剪后清单分支梯度范数均值或 None。
    - 返回值.identity_diagnostics: epoch 末两个身份门值或 None。
    - 返回值.identity_diagnostics.graph_gate: ClosedSetMatcher 图门 Python float。
    - 返回值.identity_diagnostics.inventory_gate: ClosedSetMatcher 清单门 Python float。
    - 返回值.elapsed_seconds: 完整函数墙钟秒数。
    - 返回值.data_wait_seconds: 各累积组 DataLoader 等待秒数和。
    - 返回值.compute_seconds: 各 global batch 搬运、前向、反向、更新及 CUDA 同步秒数和。
    - 返回值.samples_per_second: sample_count / max(elapsed_seconds,1e-12)。
    - 返回值.peak_cuda_allocated_bytes: 本轮最大已分配显存 byte 数。
    - 返回值.peak_cuda_reserved_bytes: 本轮最大保留显存 byte 数。
    - 返回值.global_step: 本轮结束后的累计参数更新数。

    每个物理批次损失按整个 global batch 的真实分类分母、候选数和身份有效数缩放后反向。
    所有物理批次完成后先 unscale，再执行 error_if_nonfinite=True 的全模型梯度裁剪；返回的
    clip_grad_norm_ 数值是裁剪前范数。非有限反向损失或梯度直接失败，本函数不以跳步恢复。
    """

    # float，完整训练epoch墙钟计时起点。
    started_at = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # int，每个global batch包含的物理批次数。
    accumulation = stage.training.gradient_accumulation
    # _LossStatistics，跨本epoch全部物理批次累计真实分子与分母。
    statistics = _LossStatistics()
    # int，阶段累计参数更新序号的可变副本。
    global_step = initial_global_step
    # int，train sampler在本epoch将产生的物理批次数。
    batch_count = len(loader)
    if batch_count == 0:
        raise RuntimeError("训练 DataLoader 没有产生任何批次。")
    if batch_count % accumulation != 0:
        raise RuntimeError("训练 sampler 必须产生 gradient_accumulation 的整数倍物理批次。")
    # int，本epoch完整global batch数，也是预期optimizer.step次数。
    complete_group_count = batch_count // accumulation
    if complete_group_count == 0:
        raise RuntimeError("训练批次数不足以构成一个 global batch。")
    torch.cuda.reset_peak_memory_stats(device)
    data_wait_seconds = 0.0
    compute_seconds = 0.0
    gradient_norm_sum = 0.0
    maximum_gradient_norm = 0.0
    graph_gradient_norm_sum = 0.0
    inventory_gradient_norm_sum = 0.0
    identity_gradient_step_count = 0
    # 非有限反向损失与梯度在 optimizer.step 前直接报错，因此正式策略不跳过更新。
    skipped_optimizer_steps = 0
    # Iterator[CandidateBatch]，本轮唯一loader迭代器；销毁时终止非持久worker池。
    loader_iterator = iter(loader)
    for _ in range(complete_group_count):
        wait_started_at = time.perf_counter()
        # list[CandidateBatch]，当前global batch的完整CPU物理批次组。
        cpu_group = list(itertools.islice(loader_iterator, accumulation))
        group_data_wait_seconds = time.perf_counter() - wait_started_at
        data_wait_seconds += group_data_wait_seconds
        if len(cpu_group) != accumulation:
            raise RuntimeError("DataLoader 在完整 global batch 中途提前结束。")
        # int，当前累积组沿候选轴相加的真实global batch大小。
        group_sample_count = sum(int(batch["class_target"].shape[0]) for batch in cpu_group)
        if group_sample_count != stage.training.global_batch_size:
            raise RuntimeError(
                f"global batch 应含 {stage.training.global_batch_size} 个候选，"
                f"实际为 {group_sample_count}。"
            )
        # int，当前global batch在配置身份范围内identity_target非负的候选数；候选预训练为0。
        group_eligible_count = (
            sum(
                int(
                    identity_eligible_mask(
                        cast(MatcherLossBatch, batch),
                        stage.identity_scope,
                    ).sum()
                )
                for batch in cpu_group
            )
            if stage.identity_scope is not None
            else 0
        )
        # float，当前global batch六分类候选数或真实类别权重和。
        group_classification_denominator = _classification_denominator(
            cpu_group,
            stage.training.class_weights,
        )
        # CPU浮点张量，形状 (N,4)；仅用于当前global batch裁剪统计遥测。
        group_sampling_features = torch.cat(
            [batch["candidate_sampling_features"] for batch in cpu_group],
            dim=0,
        )
        if group_sampling_features.shape != (group_sample_count, 4):
            raise ValueError("global batch 的 candidate_sampling_features 必须为 (N,4)。")
        # _LossStatistics，只累计当前global batch，供逐更新W&B损失曲线使用。
        group_statistics = _LossStatistics()
        compute_started_at = time.perf_counter()
        for cpu_batch in cpu_group:
            # CandidateBatch，张量字段异步搬到CUDA、字符串与metadata保留CPU的物理批次。
            batch = _move_batch(cpu_batch, device)
            # int，当前物理批次候选数。
            batch_size = int(batch["class_target"].shape[0])
            with _autocast_context(stage.training.precision):
                # dict[str,Tensor]，任务模型当前物理批次输出；完整Matcher另含identity_logits。
                outputs = model(batch)
                # LossOutput，当前物理批次的四项损失与两个真实聚合分母。
                losses = compute_matcher_loss(
                    outputs,
                    batch,
                    stage.training.loss_weights,
                    stage.identity_variant,
                    stage.identity_scope,
                    class_weights,
                )
                # 浮点零维张量，按同一组局部损失与真实分母计算的当前物理批次代数比例贡献。
                scaled_loss = _accumulation_backward_loss(
                    losses,
                    stage.training.loss_weights,
                    sample_fraction=batch_size / group_sample_count,
                    group_classification_denominator=group_classification_denominator,
                    group_identity_eligible_count=group_eligible_count,
                )
                if not torch.isfinite(scaled_loss):
                    raise FloatingPointError("global batch 反向损失含 NaN 或 Inf。")
            scaler.scale(scaled_loss).backward()
            statistics.update(losses, batch_size)
            group_statistics.update(losses, batch_size)
        scaler.unscale_(optimizer)
        # float，clip_grad_norm_返回的裁剪前全模型L2梯度范数。
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                stage.training.gradient_clip_norm,
                error_if_nonfinite=True,
            ).detach()
        )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("裁剪前梯度范数含 NaN 或 Inf。")
        gradient_norm_sum += gradient_norm
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        # float | None，裁剪后图PairScorer具名参数中现有梯度的联合L2范数。
        graph_gradient_norm = _named_gradient_norm(model, "identity_scorer.graph.")
        # float | None，裁剪后清单PairScorer具名参数中现有梯度的联合L2范数。
        inventory_gradient_norm = _named_gradient_norm(model, "identity_scorer.inventory.")
        # str，完整Matcher中候选编码器参数名前缀；直属CandidateEncoder为空字符串。
        candidate_encoder_prefix = (
            "candidate_encoder." if isinstance(model, ClosedSetMatcher) else ""
        )
        if graph_gradient_norm is not None and inventory_gradient_norm is not None:
            graph_gradient_norm_sum += graph_gradient_norm
            inventory_gradient_norm_sum += inventory_gradient_norm
            identity_gradient_step_count += 1
        scaler.step(optimizer)
        scaler.update()
        global_step += 1
        is_first_stage_step = global_step == 1
        # CUDA 前向、反向与 AdamW 更新均可能异步；每步同步后才能得到可跨硬件比较的完整耗时。
        torch.cuda.synchronize(device)
        global_step_compute_seconds = time.perf_counter() - compute_started_at
        compute_seconds += global_step_compute_seconds
        if is_first_stage_step:
            print(
                json.dumps(
                    {
                        "event": "first_global_step_accepted",
                        "stage": stage.name,
                        "global_step": global_step,
                        "global_batch_size": group_sample_count,
                        "identity_eligible_count": group_eligible_count,
                        "preclip_gradient_norm": gradient_norm,
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                        "global_step_compute_seconds": global_step_compute_seconds,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        # LossValues，当前global batch按真实分母重新汇总的五项Python浮点损失。
        group_loss = group_statistics.compute(stage.training.loss_weights)
        if isinstance(model, ClosedSetMatcher):
            graph_gate: float | None = float(model.identity_scorer.graph_gate.detach())
            inventory_gate: float | None = float(model.identity_scorer.inventory_gate.detach())
        else:
            graph_gate = None
            inventory_gate = None
        # dict[str,MetricValue]，当前参数更新的损失、裁剪、门控、吞吐、显存与裁剪统计遥测。
        step_metrics: dict[str, MetricValue] = {
            "loss/train_step/total": group_loss["total"],
            "loss/train_step/classification": group_loss["classification"],
            "loss/train_step/ligand_area": group_loss["ligand_area"],
            "loss/train_step/binding_area": group_loss["binding_area"],
            "loss/train_step/identity": group_loss["identity"],
            "statistics/train_step/identity_eligible_count": group_eligible_count,
            "statistics/train_step/crop_boundary_fraction": float(
                group_sampling_features[:, 0].mean()
            ),
            "statistics/train_step/crop_offset_l2_mean_voxels": float(
                group_sampling_features[:, 1].mean()
            ),
            "statistics/train_step/crop_offset_linf_mean_voxels": float(
                group_sampling_features[:, 2].mean()
            ),
            "statistics/train_step/v_retained_fraction_mean": float(
                group_sampling_features[:, 3].mean()
            ),
            "statistics/train_step/v_retained_fraction_minimum": float(
                group_sampling_features[:, 3].amin()
            ),
            "gradient/train_step/preclip_norm": gradient_norm,
            "learning_rate/train_step/current": float(optimizer.param_groups[0]["lr"]),
            "performance/train_step/data_wait_seconds": group_data_wait_seconds,
            "performance/train_step/compute_seconds": global_step_compute_seconds,
            "performance/train_step/samples_per_second": group_sample_count
            / max(group_data_wait_seconds + global_step_compute_seconds, 1e-12),
            "memory/train_step/peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
            ),
            "memory/train_step/peak_cuda_reserved_bytes": (
                torch.cuda.max_memory_reserved(device)
            ),
        }
        if graph_gradient_norm is not None:
            step_metrics["gradient/train_step/graph_branch_norm"] = graph_gradient_norm
        if inventory_gradient_norm is not None:
            step_metrics[
                "gradient/train_step/inventory_branch_norm"
            ] = inventory_gradient_norm
        if graph_gate is not None and inventory_gate is not None:
            step_metrics["gate/train_step/graph"] = graph_gate
            step_metrics["gate/train_step/inventory"] = inventory_gate
        step_metrics.update(
            _attention_bias_weight_metrics(model, candidate_encoder_prefix)
        )
        tracker.log_global_step(global_step, step_metrics)
        optimizer.zero_grad(set_to_none=True)
    # 循环不会额外请求 StopIteration；显式释放迭代器，确保 validation 前回收训练 worker。
    del loader_iterator
    # float，完整训练epoch从入口到worker迭代器释放后的墙钟秒数。
    elapsed_seconds = time.perf_counter() - started_at
    identity_diagnostics: IdentityDiagnostics | None
    if isinstance(model, ClosedSetMatcher):
        identity_diagnostics = {
            "graph_gate": float(model.identity_scorer.graph_gate.detach()),
            "inventory_gate": float(model.identity_scorer.inventory_gate.detach()),
        }
    else:
        identity_diagnostics = None
    return {
        "loss": statistics.compute(stage.training.loss_weights),
        "sample_count": statistics.sample_count,
        "identity_eligible_count": statistics.identity_eligible_count,
        "optimizer_steps": complete_group_count,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "mean_preclip_gradient_norm": gradient_norm_sum / complete_group_count,
        "maximum_preclip_gradient_norm": maximum_gradient_norm,
        "mean_graph_branch_gradient_norm": (
            graph_gradient_norm_sum / identity_gradient_step_count
            if identity_gradient_step_count > 0
            else None
        ),
        "mean_inventory_branch_gradient_norm": (
            inventory_gradient_norm_sum / identity_gradient_step_count
            if identity_gradient_step_count > 0
            else None
        ),
        "identity_diagnostics": identity_diagnostics,
        "elapsed_seconds": elapsed_seconds,
        "data_wait_seconds": data_wait_seconds,
        "compute_seconds": compute_seconds,
        "samples_per_second": statistics.sample_count / max(elapsed_seconds, 1e-12),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "global_step": global_step,
    }


def _accumulation_backward_loss(
    losses: LossOutput,
    weights: LossWeights,
    sample_fraction: float,
    group_classification_denominator: float,
    group_identity_eligible_count: int,
) -> torch.Tensor:
    """按同一组局部损失与真实分母计算物理批次对 global batch 的代数贡献。

    - losses: 当前物理批次的七字段 LossOutput。
    - losses.total: 本函数不读取。
    - losses.classification: 当前物理批次六分类损失零维张量。
    - losses.ligand_area: 当前物理批次配体区域损失零维张量。
    - losses.binding_area: 当前物理批次结合区域损失零维张量。
    - losses.identity: 当前物理批次身份损失零维张量。
    - losses.classification_denominator: 当前物理批次分类真实分母零维张量。
    - losses.identity_eligible_count: 当前物理批次身份有效候选数零维张量。
    - weights: 当前阶段的四项 LossWeights。
    - weights.classification: 六分类损失系数。
    - weights.ligand_area: 配体区域损失系数。
    - weights.binding_area: 结合区域损失系数。
    - weights.identity: 身份损失系数。
    - sample_fraction: 当前物理批量候选数除以 global batch 候选数。
    - group_classification_denominator: 完整 global batch 的候选数或真实类别权重和。
    - group_identity_eligible_count: 完整 global batch 在当前身份范围内的有效候选数。

    - 返回值: 浮点零维张量；给定相同的局部损失张量与分母，在精确算术下沿累积组求和与相应完整 global batch 公式代数等价；不同前向划分及有限精度运算不保证逐位相同。

    身份分母为零时使用 max(count,1)，同时每个物理批次 local_eligible_count 都为零，因此身份
    贡献为零。调用方保证 sample_fraction 和分类分母有效，本函数没有额外范围门禁。
    """

    # 浮点零维张量，当前物理批次占global batch分类真实分母的比例。
    classification_fraction = (
        losses["classification_denominator"] / group_classification_denominator
    )
    # 浮点零维张量，当前物理批次对global batch三项候选基础损失的贡献。
    base = (
        weights.classification * losses["classification"] * classification_fraction
        + weights.ligand_area * losses["ligand_area"] * sample_fraction
        + weights.binding_area * losses["binding_area"]
        * sample_fraction
    )
    # 浮点零维张量，当前物理批次有效身份候选数，转换为base dtype。
    local_eligible_count = losses["identity_eligible_count"].to(dtype=base.dtype)
    # 浮点零维张量，当前物理批次占global batch身份真实分母的比例。
    identity_fraction = local_eligible_count / max(group_identity_eligible_count, 1)
    return base + weights.identity * losses["identity"] * identity_fraction


def _classification_denominator(
    batches: list[CandidateBatch],
    class_weights: tuple[float, ...] | None,
) -> float:
    """计算一组 CPU 物理批次的六分类真实归一分母。

    - batches: 构成同一个 global batch 的非空 CandidateBatch 列表；只读取每项 class_target。
    - batches[*].class_target: 正式为 CPU int64 张量，形状 (B_i,)；无权重时只读取 numel，有权重时用于索引。
    - class_weights: None，或与六个 CLASS_NAMES 顺序对齐的浮点元组。

    - 返回值: 无权重时为所有 B_i 之和；有权重时为每个真实类别对应权重之和的 Python float。

    有权重且分母不大于零时失败；无权重路径没有显式非空检查，正式调用已保证 global batch 非空。
    """

    if class_weights is None:
        return float(sum(int(batch["class_target"].numel()) for batch in batches))
    # CPU float64张量，形状 (6,)；与CLASS_NAMES顺序对齐的类别权重。
    weights = torch.tensor(class_weights, dtype=torch.float64)
    # float，所有物理批次真实类别权重和。
    denominator = sum(float(weights[batch["class_target"]].sum()) for batch in batches)
    if denominator <= 0:
        raise ValueError("一个 global batch 的六分类权重和必须为正数。")
    return denominator


def _named_gradient_norm(model: nn.Module, prefix: str) -> float | None:
    """计算具名前缀下全部现有参数梯度的联合 L2 范数。

    - model: 提供 named_parameters 的模型；不修改参数或梯度。
    - prefix: 参数全名必须以其开头的字符串；空字符串会匹配全部现有梯度。

    - 返回值: 各匹配梯度转 float32 后平方和开根的 Python float；没有匹配且 grad 非空的参数时为 None。

    本函数不执行有限性门禁；正式训练在全模型 clip_grad_norm_ 的非有限检查后调用。
    """

    squared_sum = 0.0
    found = False
    for name, parameter in model.named_parameters():
        if name.startswith(prefix) and parameter.grad is not None:
            squared_sum += float(parameter.grad.detach().float().square().sum())
            found = True
    return math.sqrt(squared_sum) if found else None


def _named_parameter_norm(model: nn.Module, prefix: str) -> float:
    """计算具名前缀下全部参数值的联合有限 L2 范数。

    - model: 提供 named_parameters 的模型；参数 detach 后只读。
    - prefix: 参数全名必须以其开头的字符串。

    - 返回值: 各匹配参数转 float32 后平方和开根的有限 Python float。

    没有任何匹配参数时抛出 RuntimeError，结果为 NaN 或 Inf 时抛出 FloatingPointError。
    """

    squared_sum = 0.0
    found = False
    for name, parameter in model.named_parameters():
        if name.startswith(prefix):
            squared_sum += float(parameter.detach().float().square().sum())
            found = True
    if not found:
        raise RuntimeError(f"没有找到参数前缀 {prefix!r}。")
    norm = math.sqrt(squared_sum)
    if not math.isfinite(norm):
        raise FloatingPointError(f"参数前缀 {prefix!r} 的 L2 范数含 NaN 或 Inf。")
    return norm


def _attention_bias_weight_metrics(
    model: nn.Module,
    candidate_encoder_prefix: str,
) -> dict[str, float]:
    """返回一次更新后四模态注意力偏置末层 weight 的梯度与参数范数。

    - model: CandidateEncoder 或 ClosedSetMatcher；通过 named_parameters 查找候选聚合器。
    - candidate_encoder_prefix: 直属 CandidateEncoder 为空字符串，完整 Matcher 为 candidate_encoder.。

    - 返回值.gradient/train_step/a_attention_bias_final_weight_postclip_norm: A 模态末层 weight 裁剪后梯度 L2 范数；无现有梯度时省略。
    - 返回值.gradient/train_step/p_attention_bias_final_weight_postclip_norm: P 模态对应值；无现有梯度时省略。
    - 返回值.gradient/train_step/v_attention_bias_final_weight_postclip_norm: V 模态对应值；无现有梯度时省略。
    - 返回值.gradient/train_step/pp_attention_bias_final_weight_postclip_norm: PP 模态对应值；无现有梯度时省略。
    - 返回值.attention_bias/train_step/a_final_weight_poststep_norm: A 模态本次 optimizer.step 后 weight L2 范数；对应梯度键存在时才写。
    - 返回值.attention_bias/train_step/p_final_weight_poststep_norm: P 模态对应值；对应梯度键存在时才写。
    - 返回值.attention_bias/train_step/v_final_weight_poststep_norm: V 模态对应值；对应梯度键存在时才写。
    - 返回值.attention_bias/train_step/pp_final_weight_poststep_norm: PP 模态对应值；对应梯度键存在时才写。

    调用时点必须在 unscale、全局梯度裁剪与 optimizer.step 之后、zero_grad 之前。只记录
    attention_biases.{a,p,v,pp}.layers.2.weight；末层 bias 对同一注意力头的所有 token 加同一
    常数，不在本诊断中记录。
    """

    # dict[str,float]，只为具有现有末层weight梯度的模态写入两项诊断。
    metrics: dict[str, float] = {}
    for name in ("a", "p", "v", "pp"):
        # str，当前模态注意力偏置MLP末层weight在named_parameters中的精确全名前缀。
        parameter_prefix = (
            f"{candidate_encoder_prefix}aggregator.attention_biases."
            f"{name}.layers.2.weight"
        )
        # float | None，当前模态末层weight裁剪后梯度范数。
        gradient_norm = _named_gradient_norm(model, parameter_prefix)
        if gradient_norm is None:
            continue
        metrics[
            f"gradient/train_step/{name}_attention_bias_final_weight_postclip_norm"
        ] = gradient_norm
        metrics[
            f"attention_bias/train_step/{name}_final_weight_poststep_norm"
        ] = _named_parameter_norm(model, parameter_prefix)
    return metrics


def _move_batch(batch: CandidateBatch, device: torch.device) -> CandidateBatch:
    """把 CandidateBatch 的固定张量字段移动到设备，并保留 Python 元数据。

    - batch: collate_candidates 产生的完整 CandidateBatch；本函数只检查固定张量键的运行类型。
    - batch.metadata: 长度 B 的 CandidateMetadata 列表；本函数不读取成员字段，原对象保留在 CPU。
    - batch.metadata[*].split: 当前候选所属物化划分名称。
    - batch.metadata[*].pdb_id: 当前候选所属 PDB 标识。
    - batch.metadata[*].candidate_index: 当前 PDB 内的 Stage1 F1 候选编号。
    - batch.metadata[*].matched_object_key: 命中 occurrence 的模板键或 None。
    - batch.metadata[*].matched_prepared_smiles: 命中前景联合身份的顶层语言字符串或 None。
    - batch.metadata[*].matched_prepared_smiles_status: 顶层语言字符串的可用性状态。
    - batch.metadata[*].identity_seen_in_train: 当前语言家族的联合身份 train 可见性布尔值或 None。
    - batch.voxel_input: float16 张量，形状 (B,105,48,48,48)；移动到 device。
    - batch.class_target: int64 张量，形状 (B,)；移动到 device。
    - batch.ligand_area_target: bool 张量，形状 (B,48,48,48)；移动到 device。
    - batch.binding_area_target: bool 张量，形状 (B,48,48,48)；移动到 device。
    - batch.full_candidate_voxel_count: int64 张量，形状 (B,)；移动到 device。
    - batch.candidate_manual_features: float32 张量，形状 (B,12)；移动到 device。
    - batch.candidate_sampling_features: float32 张量，形状 (B,4)；移动到 device。
    - batch.a_tokens: float32 张量，形状 (B,N_A_max,229)；移动到 device。
    - batch.a_mask: bool 张量，形状 (B,N_A_max)；移动到 device。
    - batch.p_tokens: float32 张量，形状 (B,N_P_max,116)；移动到 device。
    - batch.p_mask: bool 张量，形状 (B,N_P_max)；移动到 device。
    - batch.v_tokens: float32 张量，形状 (B,N_V_max,52)；移动到 device。
    - batch.v_mask: bool 张量，形状 (B,N_V_max)；移动到 device。
    - batch.v_index_zyx: int64 张量，形状 (B,N_V_max,3)；移动到 device。
    - batch.pp_tokens: float32 张量，形状 (B,K,4)；移动到 device。
    - batch.pp_mask: bool 张量，形状 (B,K)；移动到 device。
    - batch.pp_index_zyx: int64 张量，形状 (B,K,3)；移动到 device。
    - batch.language_embeddings: float32 张量，形状 (B,I_max,768)；移动到 device。
    - batch.identity_mask: bool 张量，形状 (B,I_max)；移动到 device。
    - batch.identity_class_index: int64 张量，形状 (B,I_max)；移动到 device。
    - batch.identity_target: int64 张量，形状 (B,)；移动到 device。
    - batch.identity_object_keys: 长度 B 的嵌套字符串列表；原对象保留在 CPU。
    - batch.identity_prepared_smiles: 长度 B 的嵌套字符串列表；原对象保留在 CPU。
    - batch.graph_to_sample: int64 张量，形状 (G,)；移动到 device。
    - batch.graph_to_identity: int64 张量，形状 (G,)；移动到 device。
    - batch.graph_element: int64 张量，形状 (N_node,)；移动到 device。
    - batch.graph_charge: int64 张量，形状 (N_node,)；移动到 device。
    - batch.graph_residue_id: int64 张量，形状 (N_node,)；移动到 device。
    - batch.graph_node_continuous: float32 张量，形状 (N_node,15)；移动到 device。
    - batch.graph_node_graph_index: int64 张量，形状 (N_node,)；移动到 device。
    - batch.graph_edge_index: int64 张量，形状 (2,N_directed_edge)；移动到 device。
    - batch.graph_edge_features: float32 张量，形状 (N_directed_edge,10)；移动到 device。
    - device: 目标 torch.device；正式训练为 CUDA。

    - 返回值.metadata: 与 batch.metadata 共享对象的长度 B CandidateMetadata 列表，留在 CPU。
    - 返回值.metadata[*].split: 与输入共享的物化划分名称。
    - 返回值.metadata[*].pdb_id: 与输入共享的 PDB 标识。
    - 返回值.metadata[*].candidate_index: 与输入共享的候选编号。
    - 返回值.metadata[*].matched_object_key: 与输入共享的模板键或 None。
    - 返回值.metadata[*].matched_prepared_smiles: 与输入共享的顶层语言字符串或 None。
    - 返回值.metadata[*].matched_prepared_smiles_status: 与输入共享的语言字符串可用性状态。
    - 返回值.metadata[*].identity_seen_in_train: 与输入共享的 train 可见性布尔值或 None。
    - 返回值.voxel_input: device 上的 float16 张量，形状 (B,105,48,48,48)。
    - 返回值.class_target: device 上的 int64 张量，形状 (B,)。
    - 返回值.ligand_area_target: device 上的 bool 张量，形状 (B,48,48,48)。
    - 返回值.binding_area_target: device 上的 bool 张量，形状 (B,48,48,48)。
    - 返回值.full_candidate_voxel_count: device 上的 int64 张量，形状 (B,)。
    - 返回值.candidate_manual_features: device 上的 float32 张量，形状 (B,12)。
    - 返回值.candidate_sampling_features: device 上的 float32 张量，形状 (B,4)。
    - 返回值.a_tokens: device 上的 float32 张量，形状 (B,N_A_max,229)。
    - 返回值.a_mask: device 上的 bool 张量，形状 (B,N_A_max)。
    - 返回值.p_tokens: device 上的 float32 张量，形状 (B,N_P_max,116)。
    - 返回值.p_mask: device 上的 bool 张量，形状 (B,N_P_max)。
    - 返回值.v_tokens: device 上的 float32 张量，形状 (B,N_V_max,52)。
    - 返回值.v_mask: device 上的 bool 张量，形状 (B,N_V_max)。
    - 返回值.v_index_zyx: device 上的 int64 张量，形状 (B,N_V_max,3)。
    - 返回值.pp_tokens: device 上的 float32 张量，形状 (B,K,4)。
    - 返回值.pp_mask: device 上的 bool 张量，形状 (B,K)。
    - 返回值.pp_index_zyx: device 上的 int64 张量，形状 (B,K,3)。
    - 返回值.language_embeddings: device 上的 float32 张量，形状 (B,I_max,768)。
    - 返回值.identity_mask: device 上的 bool 张量，形状 (B,I_max)。
    - 返回值.identity_class_index: device 上的 int64 张量，形状 (B,I_max)。
    - 返回值.identity_target: device 上的 int64 张量，形状 (B,)。
    - 返回值.identity_object_keys: 与 batch.identity_object_keys 共享对象的长度 B 嵌套字符串列表，留在 CPU。
    - 返回值.identity_prepared_smiles: 与 batch.identity_prepared_smiles 共享对象的长度 B 嵌套字符串列表，留在 CPU。
    - 返回值.graph_to_sample: device 上的 int64 张量，形状 (G,)。
    - 返回值.graph_to_identity: device 上的 int64 张量，形状 (G,)。
    - 返回值.graph_element: device 上的 int64 张量，形状 (N_node,)。
    - 返回值.graph_charge: device 上的 int64 张量，形状 (N_node,)。
    - 返回值.graph_residue_id: device 上的 int64 张量，形状 (N_node,)。
    - 返回值.graph_node_continuous: device 上的 float32 张量，形状 (N_node,15)。
    - 返回值.graph_node_graph_index: device 上的 int64 张量，形状 (N_node,)。
    - 返回值.graph_edge_index: device 上的 int64 张量，形状 (2,N_directed_edge)。
    - 返回值.graph_edge_features: device 上的 float32 张量，形状 (N_directed_edge,10)。

    non_blocking=True 只提出异步搬运请求；是否真正异步取决于来源内存是否固定及后端。函数
    要求 _TENSOR_BATCH_KEYS 每一键都存在且值为 torch.Tensor，但不在此重复验证 shape 或 dtype。
    """

    # dict[str,Any]，CandidateBatch的浅复制；张量值将逐键替换，嵌套Python对象保持共享。
    moved: dict[str, Any] = dict(batch)
    for key in _TENSOR_BATCH_KEYS:
        value = moved[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"CandidateBatch[{key!r}] 必须是 torch.Tensor。")
        moved[key] = value.to(device, non_blocking=True)
    return cast(CandidateBatch, moved)


def _autocast_context(precision: Precision) -> AbstractContextManager[None]:
    """按精度枚举构造 CUDA autocast 上下文管理器。

    - precision: bf16、fp16 或 fp32；正式配置已完成枚举门禁。

    - 返回值: bf16 使用 torch.bfloat16、fp16 使用 torch.float16，任何其它值返回 enabled=False 的 CUDA autocast 上下文。

    本函数只构造上下文，不立即改变线程局部 autocast 状态；调用方 with 进入后才生效。
    """

    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cuda", enabled=False)


def _resolve_initial_checkpoint(specification: str) -> Path:
    """把外部候选预训练路径文字解析为 POSIX 绝对 BEST.pt Path。

    - specification: 预期为 POSIX 绝对路径字符串，最后一个组件必须恰为 BEST.pt。

    - 返回值: 使用 PurePosixPath 词法门禁后转换的 Path；不检查文件存在性或内容。
    """

    # PurePosixPath，不访问文件系统的服务器checkpoint路径语法视图。
    path = PurePosixPath(specification)
    if not path.is_absolute() or path.name != "BEST.pt":
        raise ValueError("外部 checkpoint 必须使用以 BEST.pt 结尾的绝对路径。")
    return Path(path.as_posix())


def _append_identity_scope_wandb_metrics(
    destination: dict[str, MetricValue],
    prefix: str,
    report: IdentityScopeMetrics,
) -> None:
    """把一个身份范围的固有、条件和端到端报告写入目标扁平指标映射。

    - destination: 就地更新的 W&B 指标普通 dict；既有同名键会被覆盖。
    - prefix: 当前身份范围的非空性未核验路径前缀。
    - report: 一个固定身份范围的完整 IdentityScopeMetrics。
    - report.intrinsic: 忽略六分类预测的身份排名报告。
    - report.intrinsic.top1: 身份 Top-1 正确率。
    - report.intrinsic.topk: 身份 Top-K 正确率。
    - report.intrinsic.top_k: 本函数不写入 W&B。
    - report.intrinsic.eligible_count: 身份排名候选数。
    - report.conditional: 六分类预测正确条件下的身份排名报告。
    - report.conditional.top1: 条件身份 Top-1 正确率。
    - report.conditional.topk: 条件身份 Top-K 正确率。
    - report.conditional.top_k: 本函数不写入 W&B。
    - report.conditional.eligible_count: 条件身份候选数。
    - report.end_to_end: 最终身份预测的端到端报告。
    - report.end_to_end.precision: 端到端身份 precision。
    - report.end_to_end.recall: 端到端身份 recall。
    - report.end_to_end.f1: 端到端身份 F1。
    - report.end_to_end.true_positive: 最终身份正确候选数。
    - report.end_to_end.predicted_positive: 预测落入该身份范围的候选数。
    - report.end_to_end.actual_positive: 该身份范围内有真实身份目标的候选数。

    - destination[prefix/intrinsic/top1]: report.intrinsic.top1。
    - destination[prefix/intrinsic/topk]: report.intrinsic.topk。
    - destination[prefix/intrinsic/eligible_count]: report.intrinsic.eligible_count。
    - destination[prefix/conditional/top1]: report.conditional.top1。
    - destination[prefix/conditional/topk]: report.conditional.topk。
    - destination[prefix/conditional/eligible_count]: report.conditional.eligible_count。
    - destination[prefix/end_to_end/precision]: report.end_to_end.precision。
    - destination[prefix/end_to_end/recall]: report.end_to_end.recall。
    - destination[prefix/end_to_end/f1]: report.end_to_end.f1。
    - destination[prefix/end_to_end/true_positive_count]: report.end_to_end.true_positive。
    - destination[prefix/end_to_end/predicted_positive_count]: report.end_to_end.predicted_positive。
    - destination[prefix/end_to_end/actual_positive_count]: report.end_to_end.actual_positive。

    零支持 NaN 保留到 tracking 规范化边界处理。
    """

    # IntrinsicIdentityMetrics，当前范围忽略六分类门的身份排名指标。
    intrinsic = report["intrinsic"]
    # IntrinsicIdentityMetrics，当前范围经过六分类正确门的身份排名指标。
    conditional = report["conditional"]
    # EndToEndIdentityMetrics，当前范围最终身份决策的precision/recall/F1与三个计数。
    end_to_end = report["end_to_end"]
    destination.update(
        {
            f"{prefix}/intrinsic/top1": intrinsic["top1"],
            f"{prefix}/intrinsic/topk": intrinsic["topk"],
            f"{prefix}/intrinsic/eligible_count": intrinsic["eligible_count"],
            f"{prefix}/conditional/top1": conditional["top1"],
            f"{prefix}/conditional/topk": conditional["topk"],
            f"{prefix}/conditional/eligible_count": conditional["eligible_count"],
            f"{prefix}/end_to_end/precision": end_to_end["precision"],
            f"{prefix}/end_to_end/recall": end_to_end["recall"],
            f"{prefix}/end_to_end/f1": end_to_end["f1"],
            f"{prefix}/end_to_end/true_positive_count": end_to_end["true_positive"],
            f"{prefix}/end_to_end/predicted_positive_count": end_to_end[
                "predicted_positive"
            ],
            f"{prefix}/end_to_end/actual_positive_count": end_to_end[
                "actual_positive"
            ],
        }
    )


def _evaluation_wandb_metrics(
    evaluation: EvaluationResult,
    identity_scope: IdentityScope | None,
) -> dict[str, MetricValue]:
    """把完整 validation 结果展开为按职责命名的 W&B 扁平标量。

    - evaluation: 一个完整 validation DataLoader 的 EvaluationResult。
    - evaluation.loss: validation 五项全局归一损失。
    - evaluation.loss.total: validation 四项加权总损失。
    - evaluation.loss.classification: validation 六分类损失。
    - evaluation.loss.ligand_area: validation 配体区域损失。
    - evaluation.loss.binding_area: validation 结合区域损失。
    - evaluation.loss.identity: validation 身份损失。
    - evaluation.metrics: 完整 EvaluationReport。
    - evaluation.metrics.six_class.accuracy: 六分类准确率。
    - evaluation.metrics.six_class.macro_f1: 兼容口径支持类 macro-F1。
    - evaluation.metrics.six_class.macro_f1_supported_classes: 支持类 macro-F1。
    - evaluation.metrics.six_class.macro_f1_all_classes: 固定六类 macro-F1。
    - evaluation.metrics.six_class.per_class[*].precision: 每类 precision，零预测时为 NaN。
    - evaluation.metrics.six_class.per_class[*].recall: 每类 recall，零支持时为 NaN。
    - evaluation.metrics.six_class.per_class[*].f1: 每类 F1。
    - evaluation.metrics.six_class.per_class[*].support: 每类真实候选数。
    - evaluation.metrics.six_class.per_class[*].predicted_count: 每类预测候选数。
    - evaluation.metrics.six_class.confusion_matrix: 形状 (6,6) 的整数嵌套列表，第一轴真实类、第二轴预测类。
    - evaluation.metrics.intrinsic_identity: 配置身份范围的固有排名报告。
    - evaluation.metrics.intrinsic_identity.top1: 固有身份 Top-1。
    - evaluation.metrics.intrinsic_identity.topk: 固有身份 Top-K。
    - evaluation.metrics.intrinsic_identity.top_k: 本函数不写入 W&B。
    - evaluation.metrics.intrinsic_identity.eligible_count: 固有身份候选数。
    - evaluation.metrics.conditional_identity: 配置身份范围的六分类正确条件排名报告。
    - evaluation.metrics.conditional_identity.top1: 条件身份 Top-1。
    - evaluation.metrics.conditional_identity.topk: 条件身份 Top-K。
    - evaluation.metrics.conditional_identity.top_k: 本函数不写入 W&B。
    - evaluation.metrics.conditional_identity.eligible_count: 条件身份候选数。
    - evaluation.metrics.conditional_identity.seen_top1: train seen 子集 Top-1。
    - evaluation.metrics.conditional_identity.seen_topk: train seen 子集 Top-K。
    - evaluation.metrics.conditional_identity.seen_count: train seen 子集候选数。
    - evaluation.metrics.conditional_identity.unseen_top1: train unseen 子集 Top-1。
    - evaluation.metrics.conditional_identity.unseen_topk: train unseen 子集 Top-K。
    - evaluation.metrics.conditional_identity.unseen_count: train unseen 子集候选数。
    - evaluation.metrics.end_to_end_identity.precision: 配置范围端到端 precision。
    - evaluation.metrics.end_to_end_identity.recall: 配置范围端到端 recall。
    - evaluation.metrics.end_to_end_identity.f1: 配置范围端到端 F1。
    - evaluation.metrics.end_to_end_identity.true_positive: 配置范围最终身份正确数。
    - evaluation.metrics.end_to_end_identity.predicted_positive: 配置范围预测阳性数。
    - evaluation.metrics.end_to_end_identity.actual_positive: 配置范围真实阳性数。
    - evaluation.metrics.small_molecule_identity: 小分子固定范围的固有、条件与端到端报告。
    - evaluation.metrics.small_molecule_identity.intrinsic.top1: 小分子固有 Top-1。
    - evaluation.metrics.small_molecule_identity.intrinsic.topk: 小分子固有 Top-K。
    - evaluation.metrics.small_molecule_identity.intrinsic.top_k: 本函数不写入 W&B。
    - evaluation.metrics.small_molecule_identity.intrinsic.eligible_count: 小分子固有候选数。
    - evaluation.metrics.small_molecule_identity.conditional.top1: 小分子条件 Top-1。
    - evaluation.metrics.small_molecule_identity.conditional.topk: 小分子条件 Top-K。
    - evaluation.metrics.small_molecule_identity.conditional.top_k: 本函数不写入 W&B。
    - evaluation.metrics.small_molecule_identity.conditional.eligible_count: 小分子条件候选数。
    - evaluation.metrics.small_molecule_identity.end_to_end.precision: 小分子端到端 precision。
    - evaluation.metrics.small_molecule_identity.end_to_end.recall: 小分子端到端 recall。
    - evaluation.metrics.small_molecule_identity.end_to_end.f1: 小分子端到端 F1。
    - evaluation.metrics.small_molecule_identity.end_to_end.true_positive: 小分子端到端正确数。
    - evaluation.metrics.small_molecule_identity.end_to_end.predicted_positive: 小分子端到端预测阳性数。
    - evaluation.metrics.small_molecule_identity.end_to_end.actual_positive: 小分子端到端真实阳性数。
    - evaluation.metrics.all_non_background_identity: 五类前景固定范围的固有、条件与端到端报告。
    - evaluation.metrics.all_non_background_identity.intrinsic.top1: 五类前景固有 Top-1。
    - evaluation.metrics.all_non_background_identity.intrinsic.topk: 五类前景固有 Top-K。
    - evaluation.metrics.all_non_background_identity.intrinsic.top_k: 本函数不写入 W&B。
    - evaluation.metrics.all_non_background_identity.intrinsic.eligible_count: 五类前景固有候选数。
    - evaluation.metrics.all_non_background_identity.conditional.top1: 五类前景条件 Top-1。
    - evaluation.metrics.all_non_background_identity.conditional.topk: 五类前景条件 Top-K。
    - evaluation.metrics.all_non_background_identity.conditional.top_k: 本函数不写入 W&B。
    - evaluation.metrics.all_non_background_identity.conditional.eligible_count: 五类前景条件候选数。
    - evaluation.metrics.all_non_background_identity.end_to_end.precision: 五类前景端到端 precision。
    - evaluation.metrics.all_non_background_identity.end_to_end.recall: 五类前景端到端 recall。
    - evaluation.metrics.all_non_background_identity.end_to_end.f1: 五类前景端到端 F1。
    - evaluation.metrics.all_non_background_identity.end_to_end.true_positive: 五类前景端到端正确数。
    - evaluation.metrics.all_non_background_identity.end_to_end.predicted_positive: 五类前景端到端预测阳性数。
    - evaluation.metrics.all_non_background_identity.end_to_end.actual_positive: 五类前景端到端真实阳性数。
    - evaluation.metrics.identity_per_class[*]: 五个前景类别各自的固有、条件与端到端报告。
    - evaluation.metrics.identity_per_class[*].intrinsic.top1: 当前类别固有 Top-1。
    - evaluation.metrics.identity_per_class[*].intrinsic.topk: 当前类别固有 Top-K。
    - evaluation.metrics.identity_per_class[*].intrinsic.top_k: 本函数不写入 W&B。
    - evaluation.metrics.identity_per_class[*].intrinsic.eligible_count: 当前类别固有候选数。
    - evaluation.metrics.identity_per_class[*].conditional.top1: 当前类别条件 Top-1。
    - evaluation.metrics.identity_per_class[*].conditional.topk: 当前类别条件 Top-K。
    - evaluation.metrics.identity_per_class[*].conditional.top_k: 本函数不写入 W&B。
    - evaluation.metrics.identity_per_class[*].conditional.eligible_count: 当前类别条件候选数。
    - evaluation.metrics.identity_per_class[*].end_to_end.precision: 当前类别端到端 precision。
    - evaluation.metrics.identity_per_class[*].end_to_end.recall: 当前类别端到端 recall。
    - evaluation.metrics.identity_per_class[*].end_to_end.f1: 当前类别端到端 F1。
    - evaluation.metrics.identity_per_class[*].end_to_end.true_positive: 当前类别端到端正确数。
    - evaluation.metrics.identity_per_class[*].end_to_end.predicted_positive: 当前类别端到端预测阳性数。
    - evaluation.metrics.identity_per_class[*].end_to_end.actual_positive: 当前类别端到端真实阳性数。
    - evaluation.metrics.area.ligand_micro_dice: 配体区域微平均 Dice。
    - evaluation.metrics.area.binding_micro_dice: 结合区域微平均 Dice。
    - evaluation.metrics.area.ligand_dice_denominator: 配体区域预测与真实体素数之和。
    - evaluation.metrics.area.binding_dice_denominator: 结合区域预测与真实体素数之和。
    - evaluation.sample_count: validation 候选数。
    - evaluation.identity_eligible_count: validation 有效身份候选数。
    - evaluation.elapsed_seconds: validation 墙钟秒数。
    - evaluation.samples_per_second: validation 候选吞吐。
    - identity_scope: 当前阶段配置范围或 None；None 时不写任何身份指标子树。

    - 返回值.loss/validation/total: evaluation.loss.total。
    - 返回值.loss/validation/classification: evaluation.loss.classification。
    - 返回值.loss/validation/ligand_area: evaluation.loss.ligand_area。
    - 返回值.loss/validation/binding_area: evaluation.loss.binding_area。
    - 返回值.loss/validation/identity: evaluation.loss.identity。
    - 返回值.performance/validation/elapsed_seconds: evaluation.elapsed_seconds。
    - 返回值.performance/validation/samples_per_second: evaluation.samples_per_second。
    - 返回值.statistics/validation/sample_count: evaluation.sample_count。
    - 返回值.statistics/validation/identity_eligible_count: evaluation.identity_eligible_count。
    - 返回值.classification_summary/accuracy: evaluation.metrics.six_class.accuracy。
    - 返回值.classification_summary/macro_f1: evaluation.metrics.six_class.macro_f1。
    - 返回值.classification_summary/macro_f1_supported_classes: evaluation.metrics.six_class.macro_f1_supported_classes。
    - 返回值.classification_summary/macro_f1_all_classes: evaluation.metrics.six_class.macro_f1_all_classes。
    - 返回值.area/ligand/micro_dice: evaluation.metrics.area.ligand_micro_dice。
    - 返回值.area/ligand/dice_denominator: evaluation.metrics.area.ligand_dice_denominator。
    - 返回值.area/binding/micro_dice: evaluation.metrics.area.binding_micro_dice。
    - 返回值.area/binding/dice_denominator: evaluation.metrics.area.binding_dice_denominator。
    - 返回值.classification_per_class/{class_name}/precision: 当前六分类类别的 precision。
    - 返回值.classification_per_class/{class_name}/recall: 当前六分类类别的 recall。
    - 返回值.classification_per_class/{class_name}/f1: 当前六分类类别的 F1。
    - 返回值.classification_per_class/{class_name}/support: 当前六分类类别的真实候选数。
    - 返回值.classification_per_class/{class_name}/predicted_count: 当前六分类类别的预测候选数。
    - 返回值.classification_confusion/{true_name}/{predicted_name}: 当前真实类与预测类组合的候选计数。
    - 返回值.identity_configured/intrinsic/top1: 配置身份范围的固有 Top-1。
    - 返回值.identity_configured/intrinsic/topk: 配置身份范围的固有 Top-K。
    - 返回值.identity_configured/intrinsic/eligible_count: 配置身份范围的固有候选数。
    - 返回值.identity_configured/conditional/top1: 配置身份范围的条件 Top-1。
    - 返回值.identity_configured/conditional/topk: 配置身份范围的条件 Top-K。
    - 返回值.identity_configured/conditional/eligible_count: 配置身份范围的条件候选数。
    - 返回值.identity_configured/conditional/seen/top1: 配置身份范围 train seen 子集 Top-1。
    - 返回值.identity_configured/conditional/seen/topk: 配置身份范围 train seen 子集 Top-K。
    - 返回值.identity_configured/conditional/seen/count: 配置身份范围 train seen 子集候选数。
    - 返回值.identity_configured/conditional/unseen/top1: 配置身份范围 train unseen 子集 Top-1。
    - 返回值.identity_configured/conditional/unseen/topk: 配置身份范围 train unseen 子集 Top-K。
    - 返回值.identity_configured/conditional/unseen/count: 配置身份范围 train unseen 子集候选数。
    - 返回值.identity_configured/end_to_end/precision: 配置身份范围端到端 precision。
    - 返回值.identity_configured/end_to_end/recall: 配置身份范围端到端 recall。
    - 返回值.identity_configured/end_to_end/f1: 配置身份范围端到端 F1。
    - 返回值.identity_configured/end_to_end/true_positive_count: 配置身份范围最终身份正确数。
    - 返回值.identity_configured/end_to_end/predicted_positive_count: 配置身份范围预测阳性数。
    - 返回值.identity_configured/end_to_end/actual_positive_count: 配置身份范围真实阳性数。
    - 返回值.{scope_prefix}/intrinsic/top1: 当前固定身份范围的固有 Top-1。
    - 返回值.{scope_prefix}/intrinsic/topk: 当前固定身份范围的固有 Top-K。
    - 返回值.{scope_prefix}/intrinsic/eligible_count: 当前固定身份范围的固有候选数。
    - 返回值.{scope_prefix}/conditional/top1: 当前固定身份范围的条件 Top-1。
    - 返回值.{scope_prefix}/conditional/topk: 当前固定身份范围的条件 Top-K。
    - 返回值.{scope_prefix}/conditional/eligible_count: 当前固定身份范围的条件候选数。
    - 返回值.{scope_prefix}/end_to_end/precision: 当前固定身份范围的端到端 precision。
    - 返回值.{scope_prefix}/end_to_end/recall: 当前固定身份范围的端到端 recall。
    - 返回值.{scope_prefix}/end_to_end/f1: 当前固定身份范围的端到端 F1。
    - 返回值.{scope_prefix}/end_to_end/true_positive_count: 当前固定身份范围的最终身份正确数。
    - 返回值.{scope_prefix}/end_to_end/predicted_positive_count: 当前固定身份范围的预测阳性数。
    - 返回值.{scope_prefix}/end_to_end/actual_positive_count: 当前固定身份范围的真实阳性数。

    scope_prefix 依次取 identity_small_molecule、identity_all_non_background，以及五个
    identity_{class_name} 前景类别前缀。identity_scope 为 None 时不产生任何 identity_ 前缀键。

    本函数不归一化 NaN；WandbTracker 在同批支持数为零时转换为 None。CLASS_NAMES 决定
    per_class 与 confusion_matrix 的稳定输出顺序。
    """

    # EvaluationReport，完整validation的六分类、身份与区域指标树。
    report = evaluation["metrics"]
    # SixClassMetrics，六分类汇总、逐类指标与6×6混淆矩阵。
    six_class = report["six_class"]
    # AreaMetrics，两个区域微Dice与各自分母。
    area = report["area"]
    # dict[str,MetricValue]，基础validation损失、性能、统计、分类与区域指标。
    metrics: dict[str, MetricValue] = {
        "loss/validation/total": evaluation["loss"]["total"],
        "loss/validation/classification": evaluation["loss"]["classification"],
        "loss/validation/ligand_area": evaluation["loss"]["ligand_area"],
        "loss/validation/binding_area": evaluation["loss"]["binding_area"],
        "loss/validation/identity": evaluation["loss"]["identity"],
        "performance/validation/elapsed_seconds": evaluation["elapsed_seconds"],
        "performance/validation/samples_per_second": evaluation["samples_per_second"],
        "statistics/validation/sample_count": evaluation["sample_count"],
        "statistics/validation/identity_eligible_count": evaluation[
            "identity_eligible_count"
        ],
        "classification_summary/accuracy": six_class["accuracy"],
        "classification_summary/macro_f1": six_class["macro_f1"],
        "classification_summary/macro_f1_supported_classes": six_class[
            "macro_f1_supported_classes"
        ],
        "classification_summary/macro_f1_all_classes": six_class[
            "macro_f1_all_classes"
        ],
        "area/ligand/micro_dice": area["ligand_micro_dice"],
        "area/ligand/dice_denominator": area["ligand_dice_denominator"],
        "area/binding/micro_dice": area["binding_micro_dice"],
        "area/binding/dice_denominator": area["binding_dice_denominator"],
    }
    for class_name, class_metrics in six_class["per_class"].items():
        prefix = f"classification_per_class/{class_name}"
        metrics[f"{prefix}/precision"] = class_metrics["precision"]
        metrics[f"{prefix}/recall"] = class_metrics["recall"]
        metrics[f"{prefix}/f1"] = class_metrics["f1"]
        metrics[f"{prefix}/support"] = class_metrics["support"]
        metrics[f"{prefix}/predicted_count"] = class_metrics["predicted_count"]
    for true_index, true_name in enumerate(CLASS_NAMES):
        for predicted_index, predicted_name in enumerate(CLASS_NAMES):
            metrics[
                f"classification_confusion/{true_name}/{predicted_name}"
            ] = six_class["confusion_matrix"][true_index][predicted_index]
    if identity_scope is not None:
        # IntrinsicIdentityMetrics，当前配置身份范围忽略六分类门的排名指标。
        configured_intrinsic = report["intrinsic_identity"]
        # ConditionalIdentityMetrics，当前配置范围经六分类正确门并拆分seen/unseen的排名指标。
        configured_conditional = report["conditional_identity"]
        # EndToEndIdentityMetrics，当前配置范围最终身份决策指标。
        configured_end_to_end = report["end_to_end_identity"]
        metrics.update(
            {
                "identity_configured/intrinsic/top1": configured_intrinsic["top1"],
                "identity_configured/intrinsic/topk": configured_intrinsic["topk"],
                "identity_configured/intrinsic/eligible_count": configured_intrinsic[
                    "eligible_count"
                ],
                "identity_configured/conditional/top1": configured_conditional["top1"],
                "identity_configured/conditional/topk": configured_conditional["topk"],
                "identity_configured/conditional/eligible_count": configured_conditional[
                    "eligible_count"
                ],
                "identity_configured/conditional/seen/top1": configured_conditional[
                    "seen_top1"
                ],
                "identity_configured/conditional/seen/topk": configured_conditional[
                    "seen_topk"
                ],
                "identity_configured/conditional/seen/count": configured_conditional[
                    "seen_count"
                ],
                "identity_configured/conditional/unseen/top1": configured_conditional[
                    "unseen_top1"
                ],
                "identity_configured/conditional/unseen/topk": configured_conditional[
                    "unseen_topk"
                ],
                "identity_configured/conditional/unseen/count": configured_conditional[
                    "unseen_count"
                ],
                "identity_configured/end_to_end/precision": configured_end_to_end[
                    "precision"
                ],
                "identity_configured/end_to_end/recall": configured_end_to_end["recall"],
                "identity_configured/end_to_end/f1": configured_end_to_end["f1"],
                "identity_configured/end_to_end/true_positive_count": (
                    configured_end_to_end["true_positive"]
                ),
                "identity_configured/end_to_end/predicted_positive_count": (
                    configured_end_to_end["predicted_positive"]
                ),
                "identity_configured/end_to_end/actual_positive_count": (
                    configured_end_to_end["actual_positive"]
                ),
            }
        )
        _append_identity_scope_wandb_metrics(
            metrics,
            "identity_small_molecule",
            report["small_molecule_identity"],
        )
        _append_identity_scope_wandb_metrics(
            metrics,
            "identity_all_non_background",
            report["all_non_background_identity"],
        )
        for class_name, identity_report in report["identity_per_class"].items():
            _append_identity_scope_wandb_metrics(
                metrics,
                f"identity_{class_name}",
                identity_report,
            )
    return metrics


def _identity_contribution_wandb_metrics(
    diagnostics: IdentityContributionDiagnostics,
) -> dict[str, MetricValue]:
    """把不参与选模的身份分数贡献诊断展开成 W&B 扁平指标。

    - diagnostics: validation 身份分数贡献的八字段诊断记录。
    - diagnostics.multi_identity_eligible_count: 多身份有效候选数。
    - diagnostics.nonzero_native_count: 非零语言基线候选数。
    - diagnostics.native_centered_l2_mean: 语言基线中心化范数均值或 None。
    - diagnostics.graph_increment_centered_l2_mean: 图增量中心化范数均值或 None。
    - diagnostics.inventory_increment_centered_l2_mean: 清单增量中心化范数均值或 None。
    - diagnostics.graph_to_native_ratio_mean: 图增量相对语言基线比值均值或 None。
    - diagnostics.inventory_to_native_ratio_mean: 清单增量相对语言基线比值均值或 None。
    - diagnostics.combined_to_native_ratio_mean: 两种增量和相对语言基线比值均值或 None。

    - 返回值.identity_contribution/validation/multi_identity_eligible_count: 原样写入支持数。
    - 返回值.identity_contribution/validation/nonzero_native_count: 原样写入支持数。
    - 返回值.identity_contribution/validation/{诊断名}: 只为值非 None 的六个均值字段写入。
    """

    # dict[str,MetricValue]，先写两个始终存在的支持数，再按非空性追加六个均值。
    metrics: dict[str, MetricValue] = {
        "identity_contribution/validation/multi_identity_eligible_count": diagnostics[
            "multi_identity_eligible_count"
        ],
        "identity_contribution/validation/nonzero_native_count": diagnostics[
            "nonzero_native_count"
        ],
    }
    # dict[str,float|None]，不带W&B前缀的六个可空诊断均值。
    diagnostic_values = {
        "native_centered_l2_mean": diagnostics["native_centered_l2_mean"],
        "graph_increment_centered_l2_mean": diagnostics[
            "graph_increment_centered_l2_mean"
        ],
        "inventory_increment_centered_l2_mean": diagnostics[
            "inventory_increment_centered_l2_mean"
        ],
        "graph_to_native_ratio_mean": diagnostics["graph_to_native_ratio_mean"],
        "inventory_to_native_ratio_mean": diagnostics[
            "inventory_to_native_ratio_mean"
        ],
        "combined_to_native_ratio_mean": diagnostics[
            "combined_to_native_ratio_mean"
        ],
    }
    for name, value in diagnostic_values.items():
        if value is not None:
            metrics[f"identity_contribution/validation/{name}"] = value
    return metrics


# ================================================================================================


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[CandidateBatch],
    stage: StageConfig,
    class_weights: torch.Tensor | None,
    device: torch.device,
    *,
    identity_contribution_statistics: _IdentityContributionStatistics | None = None,
) -> EvaluationResult:
    """在调用方给定的完整 DataLoader 上执行推理并汇总损失与评估报告。

    - model: CandidateEncoder 或 ClosedSetMatcher；本函数切换到 eval 模式并在 inference_mode 下调用。
    - loader: 产生 CandidateBatch 的完整数据划分 DataLoader；不在本函数重排或截断候选。
    - stage: 当前评估阶段的完整 StageConfig。
    - stage.name: 本函数不读取的阶段安全名称。
    - stage.task_name: 传给后续选模的任务事实，但本函数只由损失与模型配置间接使用。
    - stage.language_family: 本函数不直接读取，已冻结在 loader Dataset 中。
    - stage.identity_variant: 传给 compute_matcher_loss。
    - stage.identity_scope: 传给损失、EvaluationAccumulator 和可选身份贡献统计器。
    - stage.model: 本函数不直接读取，model 已按该配置构造。
    - stage.model.aggregation: 不在本函数读取的候选摘要聚合方式。
    - stage.model.use_manual_features: 不在本函数读取的手工特征开关。
    - stage.model.use_sampling_features: 不在本函数读取的裁剪统计开关。
    - stage.model.use_candidate_token_attention_bias: 不在本函数读取的候选—token 注意力偏置开关。
    - stage.model.initial_graph_gate: 不在本函数读取的图门初值。
    - stage.model.initial_inventory_gate: 不在本函数读取的清单门初值。
    - stage.model.model_dim: 不在本函数读取的共同模型宽度。
    - stage.model.attention_heads: 不在本函数读取的候选、图与清单注意力头数。
    - stage.model.dropout: 不在本函数读取的丢弃概率。
    - stage.model.voxel_feature_channels: 不在本函数读取的 RAUNet 输出通道数。
    - stage.model.raunet_planes: 不在本函数读取的九项 RAUNet 通道序列。
    - stage.model.recycle_iterations: 不在本函数读取的 RAUNet 回收轮数。
    - stage.model.gradient_checkpointing: 不在本函数读取的激活重计算开关。
    - stage.model.bottleneck_attention_layers: 不在本函数读取的 RAUNet 瓶颈注意力层数。
    - stage.model.graph_layers: 不在本函数读取的配体图消息传递层数。
    - stage.training: 提供损失、精度与身份监督设置的完整 TrainingConfig。
    - stage.training.seed: 本函数不直接读取的训练随机种子。
    - stage.training.physical_batch_size: 本函数不直接读取的物理批量。
    - stage.training.gradient_accumulation: 本函数不直接读取的梯度累计次数。
    - stage.training.global_batch_size: 本函数不直接读取的每次参数更新候选数。
    - stage.training.learning_rate: 本函数不直接读取的初始学习率。
    - stage.training.weight_decay: 本函数不直接读取的权重衰减。
    - stage.training.maximum_epochs: 本函数不直接读取的轮数上限。
    - stage.training.patience_bad_epochs: 本函数不直接读取的早停阈值。
    - stage.training.precision: 决定 CUDA autocast 上下文。
    - stage.training.loss_weights: 传给每批损失和全划分汇总。
    - stage.training.loss_weights.classification: 六分类损失系数。
    - stage.training.loss_weights.ligand_area: 配体区域损失系数。
    - stage.training.loss_weights.binding_area: 结合区域损失系数。
    - stage.training.loss_weights.identity: 身份损失系数。
    - stage.training.gradient_clip_norm: 本函数不直接读取的全模型梯度裁剪阈值。
    - stage.training.class_weights: 本函数不直接读取；实际 CUDA 张量由独立 class_weights 参数传入。
    - stage.initialize_candidate_from: 本函数不直接读取的候选初始化来源路径或 None。
    - class_weights: CUDA float32 (6,) 分类权重或 None。
    - device: 正式 CUDA 设备；CandidateBatch 张量移到此设备。
    - identity_contribution_statistics: 可选可变累计器；只允许配置 identity_scope 的 ClosedSetMatcher，更新不改变正式返回值。

    - 返回值.loss: 整个 loader 的五项全局归一 LossValues。
    - 返回值.loss.total: 四项损失加权和。
    - 返回值.loss.classification: 六分类全局归一损失。
    - 返回值.loss.ligand_area: 配体区域逐候选均值。
    - 返回值.loss.binding_area: 结合区域逐候选均值。
    - 返回值.loss.identity: 有效身份候选均值或 0。
    - 返回值.metrics: EvaluationAccumulator.compute 返回的完整报告。
    - 返回值.metrics.six_class: 六分类报告。
    - 返回值.metrics.six_class.accuracy: 全候选六分类准确率。
    - 返回值.metrics.six_class.macro_f1: 支持类兼容 macro-F1。
    - 返回值.metrics.six_class.macro_f1_supported_classes: 支持类 macro-F1。
    - 返回值.metrics.six_class.macro_f1_all_classes: 固定六类 macro-F1。
    - 返回值.metrics.six_class.per_class[*].precision: 当前类 precision。
    - 返回值.metrics.six_class.per_class[*].recall: 当前类 recall。
    - 返回值.metrics.six_class.per_class[*].f1: 当前类 F1。
    - 返回值.metrics.six_class.per_class[*].support: 当前类真实候选数。
    - 返回值.metrics.six_class.per_class[*].predicted_count: 当前类预测候选数。
    - 返回值.metrics.six_class.confusion_matrix: 6×6 整数嵌套列表，第一轴真实类、第二轴预测类。
    - 返回值.metrics.intrinsic_identity.top1: 配置身份范围固有 Top-1。
    - 返回值.metrics.intrinsic_identity.topk: 配置身份范围固有 Top-K。
    - 返回值.metrics.intrinsic_identity.top_k: 实际 K 值。
    - 返回值.metrics.intrinsic_identity.eligible_count: 固有身份候选数。
    - 返回值.metrics.conditional_identity.top1: 配置身份范围条件 Top-1。
    - 返回值.metrics.conditional_identity.topk: 配置身份范围条件 Top-K。
    - 返回值.metrics.conditional_identity.top_k: 实际 K 值。
    - 返回值.metrics.conditional_identity.eligible_count: 条件身份候选数。
    - 返回值.metrics.conditional_identity.seen_top1: train seen 子集 Top-1。
    - 返回值.metrics.conditional_identity.seen_topk: train seen 子集 Top-K。
    - 返回值.metrics.conditional_identity.seen_count: train seen 子集候选数。
    - 返回值.metrics.conditional_identity.unseen_top1: train unseen 子集 Top-1。
    - 返回值.metrics.conditional_identity.unseen_topk: train unseen 子集 Top-K。
    - 返回值.metrics.conditional_identity.unseen_count: train unseen 子集候选数。
    - 返回值.metrics.end_to_end_identity.precision: 配置身份范围端到端 precision。
    - 返回值.metrics.end_to_end_identity.recall: 配置身份范围端到端 recall。
    - 返回值.metrics.end_to_end_identity.f1: 配置身份范围端到端 F1。
    - 返回值.metrics.end_to_end_identity.true_positive: 配置身份范围最终身份正确数。
    - 返回值.metrics.end_to_end_identity.predicted_positive: 配置身份范围预测阳性数。
    - 返回值.metrics.end_to_end_identity.actual_positive: 配置身份范围真实阳性数。
    - 返回值.metrics.small_molecule_identity.intrinsic.top1: 小分子固有 Top-1。
    - 返回值.metrics.small_molecule_identity.intrinsic.topk: 小分子固有 Top-K。
    - 返回值.metrics.small_molecule_identity.intrinsic.top_k: 小分子报告实际 K。
    - 返回值.metrics.small_molecule_identity.intrinsic.eligible_count: 小分子固有候选数。
    - 返回值.metrics.small_molecule_identity.conditional.top1: 小分子条件 Top-1。
    - 返回值.metrics.small_molecule_identity.conditional.topk: 小分子条件 Top-K。
    - 返回值.metrics.small_molecule_identity.conditional.top_k: 小分子报告实际 K。
    - 返回值.metrics.small_molecule_identity.conditional.eligible_count: 小分子条件候选数。
    - 返回值.metrics.small_molecule_identity.end_to_end.precision: 小分子端到端 precision。
    - 返回值.metrics.small_molecule_identity.end_to_end.recall: 小分子端到端 recall。
    - 返回值.metrics.small_molecule_identity.end_to_end.f1: 小分子端到端 F1。
    - 返回值.metrics.small_molecule_identity.end_to_end.true_positive: 小分子端到端正确数。
    - 返回值.metrics.small_molecule_identity.end_to_end.predicted_positive: 小分子端到端预测阳性数。
    - 返回值.metrics.small_molecule_identity.end_to_end.actual_positive: 小分子端到端真实阳性数。
    - 返回值.metrics.all_non_background_identity.intrinsic.top1: 五类前景固有 Top-1。
    - 返回值.metrics.all_non_background_identity.intrinsic.topk: 五类前景固有 Top-K。
    - 返回值.metrics.all_non_background_identity.intrinsic.top_k: 五类前景报告实际 K。
    - 返回值.metrics.all_non_background_identity.intrinsic.eligible_count: 五类前景固有候选数。
    - 返回值.metrics.all_non_background_identity.conditional.top1: 五类前景条件 Top-1。
    - 返回值.metrics.all_non_background_identity.conditional.topk: 五类前景条件 Top-K。
    - 返回值.metrics.all_non_background_identity.conditional.top_k: 五类前景报告实际 K。
    - 返回值.metrics.all_non_background_identity.conditional.eligible_count: 五类前景条件候选数。
    - 返回值.metrics.all_non_background_identity.end_to_end.precision: 五类前景端到端 precision。
    - 返回值.metrics.all_non_background_identity.end_to_end.recall: 五类前景端到端 recall。
    - 返回值.metrics.all_non_background_identity.end_to_end.f1: 五类前景端到端 F1。
    - 返回值.metrics.all_non_background_identity.end_to_end.true_positive: 五类前景端到端正确数。
    - 返回值.metrics.all_non_background_identity.end_to_end.predicted_positive: 五类前景端到端预测阳性数。
    - 返回值.metrics.all_non_background_identity.end_to_end.actual_positive: 五类前景端到端真实阳性数。
    - 返回值.metrics.identity_per_class[*].intrinsic.top1: 当前前景类别固有 Top-1。
    - 返回值.metrics.identity_per_class[*].intrinsic.topk: 当前前景类别固有 Top-K。
    - 返回值.metrics.identity_per_class[*].intrinsic.top_k: 当前前景类别报告实际 K。
    - 返回值.metrics.identity_per_class[*].intrinsic.eligible_count: 当前前景类别固有候选数。
    - 返回值.metrics.identity_per_class[*].conditional.top1: 当前前景类别条件 Top-1。
    - 返回值.metrics.identity_per_class[*].conditional.topk: 当前前景类别条件 Top-K。
    - 返回值.metrics.identity_per_class[*].conditional.top_k: 当前前景类别报告实际 K。
    - 返回值.metrics.identity_per_class[*].conditional.eligible_count: 当前前景类别条件候选数。
    - 返回值.metrics.identity_per_class[*].end_to_end.precision: 当前前景类别端到端 precision。
    - 返回值.metrics.identity_per_class[*].end_to_end.recall: 当前前景类别端到端 recall。
    - 返回值.metrics.identity_per_class[*].end_to_end.f1: 当前前景类别端到端 F1。
    - 返回值.metrics.identity_per_class[*].end_to_end.true_positive: 当前前景类别端到端正确数。
    - 返回值.metrics.identity_per_class[*].end_to_end.predicted_positive: 当前前景类别端到端预测阳性数。
    - 返回值.metrics.identity_per_class[*].end_to_end.actual_positive: 当前前景类别端到端真实阳性数。
    - 返回值.metrics.area.ligand_micro_dice: 配体区域微平均 Dice。
    - 返回值.metrics.area.binding_micro_dice: 结合区域微平均 Dice。
    - 返回值.metrics.area.ligand_dice_denominator: 配体区域预测与真实体素数和。
    - 返回值.metrics.area.binding_dice_denominator: 结合区域预测与真实体素数和。
    - 返回值.sample_count: loader 实际产生的候选数。
    - 返回值.identity_eligible_count: 当前身份范围内 identity_target 非负的候选数。
    - 返回值.elapsed_seconds: 完整函数墙钟秒数。
    - 返回值.samples_per_second: sample_count / max(elapsed_seconds,1e-12)。

    损失按整个划分真实分母汇总，指标由 EvaluationAccumulator 跨物理批次合并。空 loader、
    每个物理批次的非有限损失张量或不合约模型输出失败关闭；跨批 Python float 累计仍可能溢出，
    statistics.compute 没有最终有限门禁，直接调用本函数可返回含 Inf 的 LossValues。阶段路径随后
    把该值写入 history.jsonl（严格 JSON 层转为 null）和 checkpoint，再由 W&B 指标规范化拒绝
    未登记的 Inf。可选身份贡献统计只读取 ClosedSetMatcher 分数组件，不进入损失、正式指标、
    返回值或选模分数。
    """

    # float，完整评估墙钟计时起点。
    started_at = time.perf_counter()
    model.eval()
    # EvaluationAccumulator，跨物理批次累计六分类、身份与区域计数，Top-K固定为5。
    metrics = EvaluationAccumulator(identity_scope=stage.identity_scope, top_k=5)
    # _LossStatistics，跨完整loader累计损失真实分子与分母。
    statistics = _LossStatistics()
    with torch.inference_mode():
        for cpu_batch in loader:
            # CandidateBatch，张量字段位于评估device、Python元数据仍在CPU的物理批次。
            batch = _move_batch(cpu_batch, device)
            with _autocast_context(stage.training.precision):
                # dict[str,Tensor]，候选或完整Matcher当前物理批次输出。
                outputs = model(batch)
                # LossOutput，当前物理批次损失与两个真实聚合分母。
                losses = compute_matcher_loss(
                    outputs,
                    batch,
                    stage.training.loss_weights,
                    stage.identity_variant,
                    stage.identity_scope,
                    class_weights,
                )
                if identity_contribution_statistics is not None:
                    if not isinstance(model, ClosedSetMatcher) or stage.identity_scope is None:
                        raise TypeError("身份分数贡献诊断只适用于配置身份范围的完整 Matcher。")
                    # IdentityScoreComponents，语言基线及已乘门图/清单分数增量，形状均为 (B,I)。
                    components = model.identity_score_components(
                        batch,
                        cast(FullMatcherOutput, outputs),
                    )
                    identity_contribution_statistics.update(
                        components,
                        batch,
                        stage.identity_scope,
                    )
            # int，当前评估物理批次候选数。
            batch_size = int(batch["class_target"].shape[0])
            statistics.update(losses, batch_size)
            metrics.update(outputs, batch)
    if statistics.sample_count == 0:
        raise RuntimeError("评估 DataLoader 没有产生任何样本。")
    # float，完整loader迭代、损失与指标汇总后的墙钟秒数。
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "loss": statistics.compute(stage.training.loss_weights),
        "metrics": metrics.compute(),
        "sample_count": statistics.sample_count,
        "identity_eligible_count": statistics.identity_eligible_count,
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": statistics.sample_count / max(elapsed_seconds, 1e-12),
    }


def run_training_stage(
    stage: StageConfig,
    data: DataConfig,
    tracking: TrackingConfig,
    stage_directory: Path,
    plan_directory: Path,
) -> StageSummary:
    """从零执行一个正式阶段，发布全部训练产物并完成 W&B online 收口。

    - stage: 从正式 YAML 解析出的完整 StageConfig。
    - stage.name: 新建 stage_directory 的阶段名事实及所有产物 stage 字段。
    - stage.task_name: candidate_pretrain 或 full_matcher；决定模型、身份损失和选模公式。
    - stage.language_family: Dataset 使用的语言家族或 None。
    - stage.identity_variant: 身份损失变体或 None。
    - stage.identity_scope: 身份监督范围或 None。
    - stage.model: 构造任务模型并写入 resolved 与 checkpoint 的完整模型配置。
    - stage.model.aggregation: 候选摘要聚合方式。
    - stage.model.use_manual_features: 12 维手工特征开关。
    - stage.model.use_sampling_features: 4 维裁剪统计开关。
    - stage.model.use_candidate_token_attention_bias: 候选—token 注意力偏置开关。
    - stage.model.initial_graph_gate: 图分数增量门初值。
    - stage.model.initial_inventory_gate: 清单分数增量门初值。
    - stage.model.model_dim: 候选与身份共同表示宽度。
    - stage.model.attention_heads: 候选、图与清单注意力共用头数。
    - stage.model.dropout: 模型共享丢弃概率。
    - stage.model.voxel_feature_channels: RAUNet 输出通道数。
    - stage.model.raunet_planes: 九项 RAUNet 通道序列。
    - stage.model.recycle_iterations: RAUNet 回收轮数。
    - stage.model.gradient_checkpointing: 高开销 RAUNet 子模块激活重计算开关。
    - stage.model.bottleneck_attention_layers: RAUNet 瓶颈注意力层数。
    - stage.model.graph_layers: 配体图消息传递层数。
    - stage.training: 当前阶段的完整 TrainingConfig。
    - stage.training.seed: 数值后端、sampler 与 worker 生成器种子来源。
    - stage.training.physical_batch_size: 每次前向候选数。
    - stage.training.gradient_accumulation: 每次参数更新的物理批次数。
    - stage.training.global_batch_size: 每次参数更新的候选数。
    - stage.training.learning_rate: 初始 AdamW 学习率。
    - stage.training.weight_decay: AdamW 权重衰减。
    - stage.training.maximum_epochs: 正整数轮数上限或 None。
    - stage.training.patience_bad_epochs: 连续未改善早停阈值，正式配置为 2。
    - stage.training.precision: bf16、fp16 或 fp32。
    - stage.training.gradient_clip_norm: 全模型 L2 梯度裁剪阈值。
    - stage.training.class_weights: 六分类权重或 None。
    - stage.training.loss_weights: 四项损失权重。
    - stage.training.loss_weights.classification: 六分类损失系数。
    - stage.training.loss_weights.ligand_area: 配体区域损失系数。
    - stage.training.loss_weights.binding_area: 结合区域损失系数。
    - stage.training.loss_weights.identity: 身份损失系数。
    - stage.initialize_candidate_from: 候选编码器来源 BEST.pt 路径或 None。
    - data: 从正式 YAML 解析出的完整 DataConfig。
    - data.roots: train 与 validation 共享的三个 SourceRoots。
    - data.roots.data_root: train 与 validation 的 AdaLigand 数据根。
    - data.roots.stage1_root: train 与 validation 的 Stage1 Find_0 根。
    - data.roots.output_root: 清单、候选体素与 PP token 缓存根。
    - data.coverage_threshold: 清单双向覆盖率阈值。
    - data.selection_seed: 清单 occurrence 选择种子。
    - data.pp_top_k: 每候选 PP token 上限。
    - data.local_crop_strategy: fixed_center 或 v_centroid。
    - data.train_num_workers: train DataLoader worker 数。
    - data.validation_num_workers: validation DataLoader worker 数。
    - data.prefetch_factor: 每 worker 预取物理批次数。
    - data.cache_gib_per_worker: 每 Dataset 实例五类 LRU 总 GiB 预算。
    - tracking: 从正式 YAML 解析出的完整 TrackingConfig。
    - tracking.provider: 固定为 wandb。
    - tracking.mode: 固定为 online。
    - tracking.entity: W&B 账户或团队名。
    - tracking.project: W&B 项目名。
    - tracking.group: W&B 比较组名。
    - tracking.log_every_global_steps: 参数更新曲线写入间隔。
    - tracking.init_timeout_seconds: W&B 在线初始化超时秒数。
    - stage_directory: 必须尚不存在；本函数以 exist_ok=False 创建并写入所有阶段产物。
    - plan_directory: 计划单次运行根；只传给 W&B 组成 run_name。

    - 返回值.stage: stage.name。
    - 返回值.best_epoch: 最佳有限 selection_score 所在 epoch。
    - 返回值.best_score: 阶段最佳有限 validation 选择分数。
    - 返回值.last_epoch: 实际完成的最后 epoch。
    - 返回值.global_step: 阶段累计参数更新数。
    - 返回值.stopped_by_patience: 是否由连续坏轮数达到阈值停止。
    - 返回值.best_checkpoint: stage_directory / BEST.pt 路径字符串。
    - 返回值.last_checkpoint: stage_directory / LAST.pt 路径字符串。
    - 返回值.tracking_run_id: 本阶段 W&B online run ID。
    - 返回值.tracking_url: 本阶段 W&B online HTTPS URL。

    - resolved.json: _build_stage_runtime 生成的阶段、数据、追踪、环境、优化常量、数值后端、初始化来源与候选计数记录。
    - tracking.json: start_wandb_tracking 生成的十字段在线追踪记录。
    - epoch_record.epoch: 当前非负零起始轮次。
    - epoch_record.global_step: 当前轮训练结束后的累计参数更新数。
    - epoch_record.learning_rate_used: 当前轮训练开始前读取的 AdamW 学习率。
    - epoch_record.next_learning_rate: 当前轮完整 validation 后由调度器给出的下一轮学习率。
    - epoch_record.selection_score: 当前轮完整 validation 的有限选模分数。
    - epoch_record.improved: 当前轮 selection_score 是否严格大于此前 best_score。
    - epoch_record.bad_validation_epochs: 截至当前轮连续未严格改善轮数。
    - epoch_record.training: 当前轮完整训练结果。
    - epoch_record.training.loss: 当前轮五项全局归一损失。
    - epoch_record.training.loss.total: 当前轮四项损失加权和。
    - epoch_record.training.loss.classification: 当前轮六分类损失。
    - epoch_record.training.loss.ligand_area: 当前轮配体区域损失。
    - epoch_record.training.loss.binding_area: 当前轮结合区域损失。
    - epoch_record.training.loss.identity: 当前轮身份损失。
    - epoch_record.training.sample_count: 当前轮实际训练候选数。
    - epoch_record.training.identity_eligible_count: 当前轮有效身份候选数。
    - epoch_record.training.optimizer_steps: 当前轮完成的参数更新数。
    - epoch_record.training.skipped_optimizer_steps: 当前失败关闭策略下固定为 0。
    - epoch_record.training.mean_preclip_gradient_norm: 当前轮裁剪前全模型梯度范数均值。
    - epoch_record.training.maximum_preclip_gradient_norm: 当前轮裁剪前全模型梯度范数最大值。
    - epoch_record.training.mean_graph_branch_gradient_norm: 当前轮裁剪后图分支梯度范数均值或 None。
    - epoch_record.training.mean_inventory_branch_gradient_norm: 当前轮裁剪后清单分支梯度范数均值或 None。
    - epoch_record.training.identity_diagnostics: 完整 Matcher 的两个身份门值；候选预训练为 None。
    - epoch_record.training.identity_diagnostics.graph_gate: 当前轮结束时图分数增量门值。
    - epoch_record.training.identity_diagnostics.inventory_gate: 当前轮结束时清单分数增量门值。
    - epoch_record.training.elapsed_seconds: 当前轮完整训练墙钟秒数。
    - epoch_record.training.data_wait_seconds: 当前轮等待 DataLoader 的墙钟秒数和。
    - epoch_record.training.compute_seconds: 当前轮各参数更新 CUDA 同步计算秒数和。
    - epoch_record.training.samples_per_second: 当前轮候选吞吐。
    - epoch_record.training.peak_cuda_allocated_bytes: 当前轮 PyTorch 最大已分配显存 byte 数。
    - epoch_record.training.peak_cuda_reserved_bytes: 当前轮 PyTorch 最大保留显存 byte 数。
    - epoch_record.training.global_step: 与 record.global_step 相同的当前轮结束累计更新数。
    - epoch_record.validation: 当前轮完整 validation 结果。
    - epoch_record.validation.loss: 完整 validation 的五项全局归一损失。
    - epoch_record.validation.loss.total: validation 四项损失加权和。
    - epoch_record.validation.loss.classification: validation 六分类损失。
    - epoch_record.validation.loss.ligand_area: validation 配体区域损失。
    - epoch_record.validation.loss.binding_area: validation 结合区域损失。
    - epoch_record.validation.loss.identity: validation 身份损失。
    - epoch_record.validation.metrics: 完整 validation 指标树。
    - epoch_record.validation.metrics.six_class: 全候选六分类报告。
    - epoch_record.validation.metrics.six_class.accuracy: 六分类准确率。
    - epoch_record.validation.metrics.six_class.macro_f1: 与正式选模兼容的支持类 macro-F1。
    - epoch_record.validation.metrics.six_class.macro_f1_supported_classes: 只平均真实支持类的 macro-F1。
    - epoch_record.validation.metrics.six_class.macro_f1_all_classes: 固定平均六类且把零支持类记为 0 的 macro-F1。
    - epoch_record.validation.metrics.six_class.per_class: 以六个类别名为键的逐类指标映射。
    - epoch_record.validation.metrics.six_class.per_class[*].precision: 当前类别 precision；零预测时为 NaN。
    - epoch_record.validation.metrics.six_class.per_class[*].recall: 当前类别 recall；零支持时为 NaN。
    - epoch_record.validation.metrics.six_class.per_class[*].f1: 当前类别 F1；无法计算时为 0。
    - epoch_record.validation.metrics.six_class.per_class[*].support: 当前类别真实候选数。
    - epoch_record.validation.metrics.six_class.per_class[*].predicted_count: 当前类别预测候选数。
    - epoch_record.validation.metrics.six_class.confusion_matrix: 形状 6×6 的整数嵌套列表，第一轴真实类、第二轴预测类。
    - epoch_record.validation.metrics.intrinsic_identity: 配置身份范围内忽略六分类预测的身份报告。
    - epoch_record.validation.metrics.intrinsic_identity.top1: 固有身份 Top-1 正确率。
    - epoch_record.validation.metrics.intrinsic_identity.topk: 固有身份 Top-K 正确率。
    - epoch_record.validation.metrics.intrinsic_identity.top_k: 固有身份报告采用的最大名次 K。
    - epoch_record.validation.metrics.intrinsic_identity.eligible_count: 固有身份可监督候选数。
    - epoch_record.validation.metrics.conditional_identity: 只在六分类预测正确时统计的身份报告。
    - epoch_record.validation.metrics.conditional_identity.top1: 条件身份 Top-1 正确率。
    - epoch_record.validation.metrics.conditional_identity.topk: 条件身份 Top-K 正确率。
    - epoch_record.validation.metrics.conditional_identity.top_k: 条件身份报告采用的最大名次 K。
    - epoch_record.validation.metrics.conditional_identity.eligible_count: 条件身份候选数。
    - epoch_record.validation.metrics.conditional_identity.seen_top1: train seen 子集 Top-1。
    - epoch_record.validation.metrics.conditional_identity.seen_topk: train seen 子集 Top-K。
    - epoch_record.validation.metrics.conditional_identity.seen_count: train seen 子集候选数。
    - epoch_record.validation.metrics.conditional_identity.unseen_top1: train unseen 子集 Top-1。
    - epoch_record.validation.metrics.conditional_identity.unseen_topk: train unseen 子集 Top-K。
    - epoch_record.validation.metrics.conditional_identity.unseen_count: train unseen 子集候选数。
    - epoch_record.validation.metrics.end_to_end_identity: 配置身份范围经前景门后的最终身份报告。
    - epoch_record.validation.metrics.end_to_end_identity.precision: 端到端身份 precision。
    - epoch_record.validation.metrics.end_to_end_identity.recall: 端到端身份 recall。
    - epoch_record.validation.metrics.end_to_end_identity.f1: 端到端身份 F1。
    - epoch_record.validation.metrics.end_to_end_identity.true_positive: 通过前景门且最终身份正确的候选数。
    - epoch_record.validation.metrics.end_to_end_identity.predicted_positive: 通过前景门且最终身份属于配置范围的候选数。
    - epoch_record.validation.metrics.end_to_end_identity.actual_positive: 配置范围内具备身份目标的真实候选数。
    - epoch_record.validation.metrics.small_molecule_identity: 只含小分子身份且使用严格小分子分类门的范围报告。
    - epoch_record.validation.metrics.small_molecule_identity.intrinsic: 小分子固有身份报告。
    - epoch_record.validation.metrics.small_molecule_identity.intrinsic.top1: 小分子固有 Top-1。
    - epoch_record.validation.metrics.small_molecule_identity.intrinsic.topk: 小分子固有 Top-K。
    - epoch_record.validation.metrics.small_molecule_identity.intrinsic.top_k: 小分子报告采用的最大名次 K。
    - epoch_record.validation.metrics.small_molecule_identity.intrinsic.eligible_count: 小分子固有身份候选数。
    - epoch_record.validation.metrics.small_molecule_identity.conditional: 小分子条件身份报告。
    - epoch_record.validation.metrics.small_molecule_identity.conditional.top1: 小分子条件 Top-1。
    - epoch_record.validation.metrics.small_molecule_identity.conditional.topk: 小分子条件 Top-K。
    - epoch_record.validation.metrics.small_molecule_identity.conditional.top_k: 小分子报告采用的最大名次 K。
    - epoch_record.validation.metrics.small_molecule_identity.conditional.eligible_count: 小分子条件身份候选数。
    - epoch_record.validation.metrics.small_molecule_identity.end_to_end: 小分子端到端身份报告。
    - epoch_record.validation.metrics.small_molecule_identity.end_to_end.precision: 小分子端到端 precision。
    - epoch_record.validation.metrics.small_molecule_identity.end_to_end.recall: 小分子端到端 recall。
    - epoch_record.validation.metrics.small_molecule_identity.end_to_end.f1: 小分子端到端 F1。
    - epoch_record.validation.metrics.small_molecule_identity.end_to_end.true_positive: 小分子端到端最终身份正确数。
    - epoch_record.validation.metrics.small_molecule_identity.end_to_end.predicted_positive: 小分子端到端预测阳性数。
    - epoch_record.validation.metrics.small_molecule_identity.end_to_end.actual_positive: 小分子端到端真实阳性数。
    - epoch_record.validation.metrics.all_non_background_identity: 五类前景共享身份清单的范围报告。
    - epoch_record.validation.metrics.all_non_background_identity.intrinsic: 五类前景固有身份报告。
    - epoch_record.validation.metrics.all_non_background_identity.intrinsic.top1: 五类前景固有 Top-1。
    - epoch_record.validation.metrics.all_non_background_identity.intrinsic.topk: 五类前景固有 Top-K。
    - epoch_record.validation.metrics.all_non_background_identity.intrinsic.top_k: 五类前景报告采用的最大名次 K。
    - epoch_record.validation.metrics.all_non_background_identity.intrinsic.eligible_count: 五类前景固有身份候选数。
    - epoch_record.validation.metrics.all_non_background_identity.conditional: 五类前景条件身份报告。
    - epoch_record.validation.metrics.all_non_background_identity.conditional.top1: 五类前景条件 Top-1。
    - epoch_record.validation.metrics.all_non_background_identity.conditional.topk: 五类前景条件 Top-K。
    - epoch_record.validation.metrics.all_non_background_identity.conditional.top_k: 五类前景报告采用的最大名次 K。
    - epoch_record.validation.metrics.all_non_background_identity.conditional.eligible_count: 五类前景条件身份候选数。
    - epoch_record.validation.metrics.all_non_background_identity.end_to_end: 五类前景端到端身份报告。
    - epoch_record.validation.metrics.all_non_background_identity.end_to_end.precision: 五类前景端到端 precision。
    - epoch_record.validation.metrics.all_non_background_identity.end_to_end.recall: 五类前景端到端 recall。
    - epoch_record.validation.metrics.all_non_background_identity.end_to_end.f1: 五类前景端到端 F1。
    - epoch_record.validation.metrics.all_non_background_identity.end_to_end.true_positive: 五类前景端到端最终身份正确数。
    - epoch_record.validation.metrics.all_non_background_identity.end_to_end.predicted_positive: 五类前景端到端预测阳性数。
    - epoch_record.validation.metrics.all_non_background_identity.end_to_end.actual_positive: 五类前景端到端真实阳性数。
    - epoch_record.validation.metrics.identity_per_class: 以前景类别名为键的逐类身份范围报告。
    - epoch_record.validation.metrics.identity_per_class[*].intrinsic: 当前前景类固有身份报告。
    - epoch_record.validation.metrics.identity_per_class[*].intrinsic.top1: 当前前景类固有 Top-1。
    - epoch_record.validation.metrics.identity_per_class[*].intrinsic.topk: 当前前景类固有 Top-K。
    - epoch_record.validation.metrics.identity_per_class[*].intrinsic.top_k: 当前前景类报告采用的最大名次 K。
    - epoch_record.validation.metrics.identity_per_class[*].intrinsic.eligible_count: 当前前景类固有身份候选数。
    - epoch_record.validation.metrics.identity_per_class[*].conditional: 当前前景类条件身份报告。
    - epoch_record.validation.metrics.identity_per_class[*].conditional.top1: 当前前景类条件 Top-1。
    - epoch_record.validation.metrics.identity_per_class[*].conditional.topk: 当前前景类条件 Top-K。
    - epoch_record.validation.metrics.identity_per_class[*].conditional.top_k: 当前前景类报告采用的最大名次 K。
    - epoch_record.validation.metrics.identity_per_class[*].conditional.eligible_count: 当前前景类条件身份候选数。
    - epoch_record.validation.metrics.identity_per_class[*].end_to_end: 当前前景类端到端身份报告。
    - epoch_record.validation.metrics.identity_per_class[*].end_to_end.precision: 当前前景类端到端 precision。
    - epoch_record.validation.metrics.identity_per_class[*].end_to_end.recall: 当前前景类端到端 recall。
    - epoch_record.validation.metrics.identity_per_class[*].end_to_end.f1: 当前前景类端到端 F1。
    - epoch_record.validation.metrics.identity_per_class[*].end_to_end.true_positive: 当前前景类端到端最终身份正确数。
    - epoch_record.validation.metrics.identity_per_class[*].end_to_end.predicted_positive: 当前前景类端到端预测阳性数。
    - epoch_record.validation.metrics.identity_per_class[*].end_to_end.actual_positive: 当前前景类端到端真实阳性数。
    - epoch_record.validation.metrics.area: 两个体素区域的微平均 Dice 报告。
    - epoch_record.validation.metrics.area.ligand_micro_dice: 配体区域微平均 Dice。
    - epoch_record.validation.metrics.area.binding_micro_dice: 结合区域微平均 Dice。
    - epoch_record.validation.metrics.area.ligand_dice_denominator: 配体区域预测与真实体素数和。
    - epoch_record.validation.metrics.area.binding_dice_denominator: 结合区域预测与真实体素数和。
    - epoch_record.validation.sample_count: 完整 validation 候选数。
    - epoch_record.validation.identity_eligible_count: 完整 validation 有效身份候选数。
    - epoch_record.validation.elapsed_seconds: 完整 validation 墙钟秒数。
    - epoch_record.validation.samples_per_second: 完整 validation 候选吞吐。
    - epoch_record.timestamp_unix: 追加 history 前调用 time.time 得到的 Unix 墙钟秒数。
    - history.jsonl[*]: 每个完成 epoch 的 epoch_record，先于同轮 checkpoint 追加。
    - LAST.pt.schema_version: CHECKPOINT_SCHEMA_VERSION。
    - LAST.pt.stage_name: stage.name。
    - LAST.pt.task_name: stage.task_name。
    - LAST.pt.epoch: 当前零起始 epoch。
    - LAST.pt.global_step: 当前轮结束累计更新数。
    - LAST.pt.best_epoch: 截至当前轮最佳 epoch。
    - LAST.pt.best_score: 截至当前轮最佳 selection_score。
    - LAST.pt.bad_validation_epochs: 截至当前轮连续未改善轮数。
    - LAST.pt.model_config: stage.model.to_dict 的完整模型配置。
    - LAST.pt.model_config.aggregation: 候选摘要聚合方式。
    - LAST.pt.model_config.use_manual_features: 手工特征开关。
    - LAST.pt.model_config.use_sampling_features: 裁剪统计开关。
    - LAST.pt.model_config.use_candidate_token_attention_bias: 候选—token 注意力偏置开关。
    - LAST.pt.model_config.initial_graph_gate: 图分数增量门初值。
    - LAST.pt.model_config.initial_inventory_gate: 清单分数增量门初值。
    - LAST.pt.model_config.model_dim: 候选与身份共同表示宽度。
    - LAST.pt.model_config.attention_heads: 候选、图与清单注意力共用头数。
    - LAST.pt.model_config.dropout: 模型丢弃概率。
    - LAST.pt.model_config.voxel_feature_channels: RAUNet 输出通道数。
    - LAST.pt.model_config.raunet_planes: 九项 RAUNet 通道列表。
    - LAST.pt.model_config.recycle_iterations: RAUNet 回收轮数。
    - LAST.pt.model_config.gradient_checkpointing: 激活重计算开关。
    - LAST.pt.model_config.bottleneck_attention_layers: RAUNet 瓶颈注意力层数。
    - LAST.pt.model_config.graph_layers: 配体图消息传递层数。
    - LAST.pt.training_config: resolved stage.training 同源的普通 dict。
    - LAST.pt.training_config.seed: 训练随机种子。
    - LAST.pt.training_config.physical_batch_size: 物理批量。
    - LAST.pt.training_config.gradient_accumulation: 梯度累计次数。
    - LAST.pt.training_config.global_batch_size: 每次参数更新候选数。
    - LAST.pt.training_config.learning_rate: 阶段初始学习率。
    - LAST.pt.training_config.weight_decay: AdamW 权重衰减。
    - LAST.pt.training_config.maximum_epochs: 轮数上限或 None。
    - LAST.pt.training_config.patience_bad_epochs: 早停坏轮数阈值。
    - LAST.pt.training_config.precision: bf16、fp16 或 fp32。
    - LAST.pt.training_config.gradient_clip_norm: 全模型 L2 梯度裁剪阈值。
    - LAST.pt.training_config.class_weights: 六项分类权重列表或 None。
    - LAST.pt.training_config.loss_weights: 四项损失系数映射。
    - LAST.pt.training_config.loss_weights.classification: 六分类损失系数。
    - LAST.pt.training_config.loss_weights.ligand_area: 配体区域损失系数。
    - LAST.pt.training_config.loss_weights.binding_area: 结合区域损失系数。
    - LAST.pt.training_config.loss_weights.identity: 身份损失系数。
    - LAST.pt.model_state: 当前模型参数与缓冲区 state_dict。
    - LAST.pt.optimizer_state: 当前 AdamW state_dict。
    - LAST.pt.scheduler_state: 当前 ReduceLROnPlateau state_dict。
    - LAST.pt.scaler_state: 当前 GradScaler state_dict；禁用 scaler 时可为空映射。
    - LAST.pt.epoch_record: 与同轮 history.jsonl[*] 同源的 epoch_record 值。
    - BEST.pt: 只在 selection_score 严格改善时用与同轮 LAST.pt 相同的 checkpoint_state 发布。
    - summary.json: 与候选返回值同源的十字段本地阶段摘要；先于 finish(0) 写入，finish 失败时可保留但函数不返回成功结果。

    阶段目录先创建，再构造 runtime 和 W&B；任一失败不会回滚已发布文件。W&B 初始化发生在
    DataLoader 首次迭代前，因此 spawn worker 不继承已创建的客户端对象。训练、验证、写盘或
    在线记录失败时尝试 finish(exit_code=1) 后重抛；成功路径先发布 summary.json，再调用
    finish(exit_code=0)，同步失败仍会向上层传播。本入口不读取 calibration。
    """

    stage_directory.mkdir(parents=True, exist_ok=False)
    # _StageRuntime，持有CUDA模型、两个loader、优化状态及已发布resolved配置。
    runtime = _build_stage_runtime(
        stage,
        data,
        tracking,
        stage_directory,
    )
    # WandbTracker，已发布tracking.json并声明全部W&B横轴与指标模式的在线run封装。
    tracker = start_wandb_tracking(
        tracking,
        stage,
        stage_directory,
        plan_directory,
        runtime.resolved_config,
    )
    try:
        best_score = -math.inf
        best_epoch = -1
        bad_validation_epochs = 0
        global_step = 0
        # Path，每个完成epoch按顺序追加一条严格JSON记录的历史文件。
        history_path = stage_directory / "history.jsonl"
        epoch = 0
        last_epoch = -1
        while stage.training.maximum_epochs is None or epoch < stage.training.maximum_epochs:
            last_epoch = epoch
            runtime.train_sampler.set_epoch(epoch)
            learning_rate_used = float(runtime.optimizer.param_groups[0]["lr"])
            # TrainingEpochResult，当前完整训练epoch的损失、梯度、性能、显存与累计步数。
            training = _train_epoch(
                runtime.model,
                runtime.train_loader,
                runtime.optimizer,
                runtime.scaler,
                stage,
                runtime.class_weights,
                runtime.device,
                global_step,
                tracker,
            )
            global_step = int(training["global_step"])
            # _IdentityContributionStatistics | None，完整Matcher validation的非选模分数贡献诊断累计器。
            identity_contribution_statistics = (
                _IdentityContributionStatistics()
                if isinstance(runtime.model, ClosedSetMatcher)
                else None
            )
            # EvaluationResult，当前完整validation划分的损失、指标、支持数与吞吐。
            validation = evaluate_model(
                runtime.model,
                runtime.validation_loader,
                stage,
                runtime.class_weights,
                runtime.device,
                identity_contribution_statistics=identity_contribution_statistics,
            )
            # EvaluationReport，当前轮完整validation的六分类、身份和区域指标树。
            validation_metrics = validation["metrics"]
            # float，仅由任务规定的三个validation指标输入之一或组合得到的有限选模分数。
            score = selection_score(
                stage.task_name,
                validation_metrics["six_class"]["macro_f1"],
                validation_metrics["conditional_identity"]["top1"],
                validation_metrics["end_to_end_identity"]["f1"],
            )
            if not math.isfinite(score):
                raise FloatingPointError("validation 选择分数不是有限值。")
            runtime.scheduler.step(score)
            next_learning_rate = float(runtime.optimizer.param_groups[0]["lr"])
            improved = score > best_score
            if improved:
                best_score = score
                best_epoch = epoch
                bad_validation_epochs = 0
            else:
                bad_validation_epochs += 1
            # EpochRecord，追加history并嵌入本轮LAST/BEST checkpoint的完整轮次记录。
            epoch_record: EpochRecord = {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate_used": learning_rate_used,
                "next_learning_rate": next_learning_rate,
                "selection_score": score,
                "improved": improved,
                "bad_validation_epochs": bad_validation_epochs,
                "training": training,
                "validation": validation,
                "timestamp_unix": time.time(),
            }
            append_jsonl(history_path, epoch_record)
            # dict[str,object]，本轮LAST与严格改善时BEST共同使用的完整checkpoint顶层映射。
            checkpoint_state: dict[str, object] = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "stage_name": stage.name,
                "task_name": stage.task_name,
                "epoch": epoch,
                "global_step": global_step,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "bad_validation_epochs": bad_validation_epochs,
                "model_config": stage.model.to_dict(),
                "training_config": runtime.training_config,
                "model_state": runtime.model.state_dict(),
                "optimizer_state": runtime.optimizer.state_dict(),
                "scheduler_state": runtime.scheduler.state_dict(),
                "scaler_state": runtime.scaler.state_dict(),
                "epoch_record": epoch_record,
            }
            save_checkpoint(stage_directory / "LAST.pt", checkpoint_state)
            if improved:
                save_checkpoint(stage_directory / "BEST.pt", checkpoint_state)
            # dict[str,MetricValue]，本轮训练、validation、选模、门控与身份贡献W&B指标。
            epoch_metrics: dict[str, MetricValue] = {
                "loss/train_epoch/total": training["loss"]["total"],
                "loss/train_epoch/classification": training["loss"]["classification"],
                "loss/train_epoch/ligand_area": training["loss"]["ligand_area"],
                "loss/train_epoch/binding_area": training["loss"]["binding_area"],
                "loss/train_epoch/identity": training["loss"]["identity"],
                "statistics/train_epoch/sample_count": training["sample_count"],
                "statistics/train_epoch/identity_eligible_count": training[
                    "identity_eligible_count"
                ],
                "performance/train_epoch/elapsed_seconds": training["elapsed_seconds"],
                "performance/train_epoch/samples_per_second": training[
                    "samples_per_second"
                ],
                "performance/train_epoch/data_wait_seconds": training["data_wait_seconds"],
                "performance/train_epoch/compute_seconds": training["compute_seconds"],
                "memory/train_epoch/peak_cuda_allocated_bytes": training[
                    "peak_cuda_allocated_bytes"
                ],
                "memory/train_epoch/peak_cuda_reserved_bytes": training[
                    "peak_cuda_reserved_bytes"
                ],
                "gradient/train_epoch/preclip_mean_norm": training[
                    "mean_preclip_gradient_norm"
                ],
                "gradient/train_epoch/preclip_maximum_norm": training[
                    "maximum_preclip_gradient_norm"
                ],
                "learning_rate/train_epoch/used": learning_rate_used,
                "selection/score": score,
                "selection/best_score": best_score,
                "selection/improved": improved,
                "selection/bad_validation_epochs": bad_validation_epochs,
                "learning_rate/train_epoch/next": next_learning_rate,
            }
            if training["identity_diagnostics"] is not None:
                epoch_metrics["gate/train_epoch/graph"] = training[
                    "identity_diagnostics"
                ]["graph_gate"]
                epoch_metrics["gate/train_epoch/inventory"] = training[
                    "identity_diagnostics"
                ]["inventory_gate"]
            if training["mean_graph_branch_gradient_norm"] is not None:
                epoch_metrics["gradient/train_epoch/graph_branch_mean_norm"] = training[
                    "mean_graph_branch_gradient_norm"
                ]
            if training["mean_inventory_branch_gradient_norm"] is not None:
                epoch_metrics[
                    "gradient/train_epoch/inventory_branch_mean_norm"
                ] = training["mean_inventory_branch_gradient_norm"]
            epoch_metrics.update(
                _evaluation_wandb_metrics(validation, stage.identity_scope)
            )
            if identity_contribution_statistics is not None:
                epoch_metrics.update(
                    _identity_contribution_wandb_metrics(
                        identity_contribution_statistics.compute()
                    )
                )
            tracker.log_epoch(epoch, global_step, epoch_metrics)
            print(
                json.dumps(
                    {
                        "stage": stage.name,
                        "epoch": epoch,
                        "score": score,
                        "best_score": best_score,
                        "bad_validation_epochs": bad_validation_epochs,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if bad_validation_epochs >= stage.training.patience_bad_epochs:
                break
            epoch += 1
        if best_epoch < 0:
            raise RuntimeError(f"阶段 {stage.name} 没有产生有限 validation 选择分数。")
        # StageSummary，训练循环完成后的十字段本地摘要；先写文件，finish(0)正常返回后才交给根计划。
        summary: StageSummary = {
            "stage": stage.name,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "last_epoch": last_epoch,
            "global_step": global_step,
            "stopped_by_patience": (
                bad_validation_epochs >= stage.training.patience_bad_epochs
            ),
            "best_checkpoint": str(stage_directory / "BEST.pt"),
            "last_checkpoint": str(stage_directory / "LAST.pt"),
            "tracking_run_id": tracker.run_id,
            "tracking_url": tracker.run_url,
        }
        tracker.update_summary(
            {
                "best_epoch": best_epoch,
                "best_score": best_score,
                "last_epoch": last_epoch,
                "global_step": global_step,
                "stopped_by_patience": summary["stopped_by_patience"],
            }
        )
        write_json(stage_directory / "summary.json", summary)
    except BaseException as training_error:
        try:
            tracker.finish(exit_code=1)
        except BaseException as finish_error:
            print(
                json.dumps(
                    {
                        "event": "wandb_finish_failed_after_training_error",
                        "stage": stage.name,
                        "training_error_type": type(training_error).__name__,
                        "finish_error_type": type(finish_error).__name__,
                        "finish_error": str(finish_error),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
        raise
    tracker.finish(exit_code=0)
    return summary
