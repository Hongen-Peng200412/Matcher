"""在 validation 选定方案后，对一个 BEST checkpoint 执行一次 calibration 评估。

validation 是训练期间反复使用并选择 BEST checkpoint 的验证划分；calibration 是选择已经
冻结后只评估一次的保留划分。本入口要求 manifest 已经声明 calibration，不创建或扩充任何
数据划分，不训练模型，不更新 checkpoint，也不重新计算用于选模的 selection score。

一次调用会把 calibration 清单语义、候选 JSONL 和身份 JSONL 组合成数据指纹，并在 Matcher
产物根内用该指纹命名一次性 claim。claim 把数据指纹绑定到计划、阶段、BEST checkpoint 与
结果路径；claim 文件保留时，completed 状态阻止合作调用者第二次评估。锁文件只提供进程级
互斥，不是科学完成证据。
成功路径把完整 EvaluationResult、输入内容摘要和运行环境写入调用方指定的严格 JSON。

本模块中的 SHA-256 只属于一次性 calibration 留证和崩溃恢复契约，不参与模型前向、训练
配置或主数据解析。模块 Docstring 同时作为 argparse 帮助正文，改写说明文字不会改变参数
名称、默认值、解析或退出语义。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, NoReturn, TypedDict, cast

import torch

from ..data.manifest import require_manifest
from ..data.schema import LocalCropStrategy
from ..datasets.batching import (
    DATALOADER_MULTIPROCESSING_CONTEXT,
    ShuffledBatchSampler,
    build_candidate_loader,
)
from ..datasets.candidate_data import CandidateDataset
from ..model import CandidateEncoder, ClosedSetMatcher, MatcherModelConfig
from ..training.checkpointing import load_checkpoint
from ..training.config import (
    ExperimentPlan,
    StageConfig,
    load_experiment_plan,
    validate_matcher_output_path,
)
from ..training.engine import EvaluationResult, evaluate_model
from ..training.numeric import NumericBackendRecord, configure_random_seeds_and_numeric_backend
from ..utils.json_artifacts import write_json
from ..utils.locking import exclusive_file_lock


# artifact 与 claim 共同使用的整数格式版本；任一字段契约变化都应联动提升。
CALIBRATION_SCHEMA_VERSION = 2


class CalibrationCheckpointEvidence(TypedDict):
    """记录所选 BEST checkpoint 及与训练阶段相互核对的文件证据。

    - path: BEST.pt 的 POSIX 词法规范化路径文本；正式入口要求该文本位于 Matcher 服务器产物根且文件名恰为 BEST.pt，但不解析符号链接目标。
    - sha256: 评估前流式读取 BEST.pt 全部字节得到的 64 个小写十六进制字符 SHA-256。
    - epoch: 非负整数，checkpoint.epoch；必须等于 checkpoint.best_epoch 和 epoch_record.epoch。
    - global_step: 非负整数，保存该最佳轮次时累计完成的参数更新数；必须等于 epoch_record.global_step。
    - best_score: 有限浮点数，checkpoint.best_score；必须等于 epoch_record.selection_score 与阶段 summary.best_score。
    - resolved_path: 与 BEST.pt 同目录的 resolved.json 路径文本。
    - resolved_sha256: 评估时读取 resolved.json 全部字节得到的 SHA-256。
    - summary_path: 与 BEST.pt 同目录的阶段 summary.json 路径文本。
    - summary_sha256: 评估时读取阶段 summary.json 全部字节得到的 SHA-256。
    - run_path: BEST.pt 上两级目录中的 run.json 路径文本。
    - run_sha256: 评估时读取 run.json 全部字节得到的 SHA-256。
    - training_plan_path: run.json.plan_path 指向且评估时仍存在的训练 YAML 路径文本。
    - training_plan_sha256: 评估时重新读取 training_plan_path 全部字节得到的 SHA-256。
    - model_config: 由 checkpoint.model_config 经 MatcherModelConfig.from_dict 规范化、并与所选 YAML 阶段相等的字典。
    - model_config.aggregation: mean_max 或 candidate_cross_attention。
    - model_config.use_manual_features: 是否使用固化手工统计。
    - model_config.use_sampling_features: 是否使用裁剪几何统计。
    - model_config.use_candidate_token_attention_bias: 是否使用候选到 token 的注意力偏置。
    - model_config.initial_graph_gate: 图分数增量门的有限初值。
    - model_config.initial_inventory_gate: 身份清单分数增量门的有限初值。
    - model_config.model_dim: 候选与身份表示宽度。
    - model_config.attention_heads: 候选、图和清单注意力头数。
    - model_config.dropout: 模型共用丢弃概率。
    - model_config.voxel_feature_channels: RAUNet 最高分辨率输出通道数。
    - model_config.raunet_planes: 内存中为九项整数元组，写入 JSON 后编码为数组。
    - model_config.recycle_iterations: RAUNet 回收轮数。
    - model_config.gradient_checkpointing: 是否启用训练期激活重计算。
    - model_config.bottleneck_attention_layers: RAUNet 瓶颈注意力层数。
    - model_config.graph_layers: LigandObject 图消息传递层数。
    - training_config: checkpoint.training_config 的字符串键字典；必须与所选 YAML 阶段 TrainingConfig 的 asdict 结果相等。
    - training_config.seed: Python、NumPy、Torch 和采样使用的随机种子。
    - training_config.physical_batch_size: 单次前向候选数。
    - training_config.gradient_accumulation: 每次参数更新包含的物理批次数。
    - training_config.global_batch_size: physical_batch_size 与 gradient_accumulation 的乘积。
    - training_config.learning_rate: AdamW 初始学习率。
    - training_config.weight_decay: AdamW 解耦权重衰减系数。
    - training_config.maximum_epochs: 最大轮数或 None。
    - training_config.patience_bad_epochs: 早停允许的连续未改善轮数。
    - training_config.precision: bf16、fp16 或 fp32。
    - training_config.gradient_clip_norm: 全模型 L2 梯度裁剪阈值。
    - training_config.class_weights: 按 CLASS_NAMES 对齐的六项类别权重或 None。
    - training_config.loss_weights: 四项损失系数字典。
    - training_config.loss_weights.classification: 六分类损失系数。
    - training_config.loss_weights.ligand_area: 配体区域损失系数。
    - training_config.loss_weights.binding_area: 结合区域损失系数。
    - training_config.loss_weights.identity: 闭集身份损失系数。
    """

    path: str
    sha256: str
    epoch: int
    global_step: int
    best_score: float
    resolved_path: str
    resolved_sha256: str
    summary_path: str
    summary_sha256: str
    run_path: str
    run_sha256: str
    training_plan_path: str
    training_plan_sha256: str
    model_config: dict[str, Any]
    training_config: dict[str, Any]


class CalibrationDataEvidence(TypedDict):
    """记录一次 calibration 当前读取的 manifest 与两个划分记录文件证据。

    - manifest_path: 已存在 manifest.json 的路径文本。
    - manifest_sha256: 评估前读取 manifest.json 全部字节得到的 SHA-256；创建时间或其它划分变化也会改变它。
    - training_manifest_sha256: 预留给训练时 manifest 摘要的可空字段；当前实现固定写 None，没有从 checkpoint 或 run.json 恢复该值。
    - manifest_semantic_sha256: 对 calibration 相关稳定字段按键排序、紧凑严格 JSON 编码后得到的 SHA-256。
    - candidates_path: manifest.splits.calibration.candidates 在 manifest 同目录解析出的单级文件路径。
    - candidates_sha256: calibration 候选 JSONL 全部字节的 SHA-256。
    - identities_path: manifest.splits.calibration.identities 在 manifest 同目录解析出的单级文件路径。
    - identities_sha256: calibration 身份 JSONL 全部字节的 SHA-256。
    - fingerprint: 依次连接 manifest_semantic_sha256、candidates_sha256、identities_sha256 与换行符后的 ASCII 字节 SHA-256。

    manifest_semantic_sha256 还把下列三个来源对象按完整 JSON 值编码；已知字段逐项列出，
    未识别的附加键也随父对象进入摘要，但本函数不解析或验证其含义：
    - manifest.splits.calibration: 完整 calibration 文件对象。
    - manifest.splits.calibration.candidates: 候选 JSONL 的单级文件名字符串。
    - manifest.splits.calibration.identities: 身份 JSONL 的单级文件名字符串。
    - manifest.split_pdb_counts.calibration: 完整 calibration PDB 计数 JSON 值；父对象不是映射时以 null 编码，当前函数不要求该值本身是对象。
    - manifest.split_pdb_counts.calibration.discovered: 已知字段，发现的 PDB 目录数。
    - manifest.split_pdb_counts.calibration.included: 已知字段，成功纳入的 PDB 数。
    - manifest.split_pdb_counts.calibration.excluded: 已知字段，整 PDB 排除数。
    - manifest.summary.splits.calibration: 完整 calibration 汇总 JSON 值；任一父对象不是映射时以 null 编码，当前函数不要求该值本身是对象。
    - manifest.summary.splits.calibration.candidate_count: 已知字段，候选记录数。
    - manifest.summary.splits.calibration.class_counts: 已知字段，实际出现类别到候选数的映射。
    - manifest.summary.splits.calibration.class_counts.background: 条件字段，背景候选数。
    - manifest.summary.splits.calibration.class_counts.ion: 条件字段，离子候选数。
    - manifest.summary.splits.calibration.class_counts.nucleotide_like: 条件字段，核苷酸样候选数。
    - manifest.summary.splits.calibration.class_counts.peptide_like: 条件字段，肽样候选数。
    - manifest.summary.splits.calibration.class_counts.small_molecule: 条件字段，小分子候选数。
    - manifest.summary.splits.calibration.class_counts.sugar: 条件字段，糖候选数。
    - manifest.summary.splits.calibration.identity_count: 已知字段，联合身份记录数。
    - manifest.summary.splits.calibration.unique_object_key_count: 已知字段，不同模板键数。
    - manifest.summary.splits.calibration.same_pdb_object_key_multi_prepared_smiles_group_count: 已知字段，同 PDB 同模板键多字符串分组数。
    - manifest.summary.splits.calibration.matched_foreground_without_prepared_smiles_candidate_count: 已知字段，前景命中但顶层字符串不可用的候选数。
    - manifest.summary.splits.calibration.matched_foreground_explicit_null_candidate_count: 已知字段，顶层字符串显式为 null 的前景候选数。
    - manifest.summary.splits.calibration.matched_foreground_missing_prepared_file_candidate_count: 已知字段，PreparedLigand 文件缺失的前景候选数。
    - manifest.summary.splits.calibration.language_available_identity_count: 已知字段，两个语言家族到可用身份数的映射。
    - manifest.summary.splits.calibration.language_available_identity_count.molformer: MoLFormer 可用身份数。
    - manifest.summary.splits.calibration.language_available_identity_count.smi_ted_289m: SMI-TED 可用身份数。
    - manifest.summary.splits.calibration.language_multi_model_smiles_identity_count: 已知字段，两个语言家族到多 tokenizer 字符串身份数的映射。
    - manifest.summary.splits.calibration.language_multi_model_smiles_identity_count.molformer: MoLFormer 多字符串身份数。
    - manifest.summary.splits.calibration.language_multi_model_smiles_identity_count.smi_ted_289m: SMI-TED 多字符串身份数。
    - manifest.summary.splits.calibration.seen_positive_identity_count: 已知字段，两个语言家族到训练可见身份数的映射。
    - manifest.summary.splits.calibration.seen_positive_identity_count.molformer: MoLFormer 口径训练可见身份数。
    - manifest.summary.splits.calibration.seen_positive_identity_count.smi_ted_289m: SMI-TED 口径训练可见身份数。

    其它语义输入是 schema_version、三个数据根、coverage_threshold、selection_seed、
    identity_key_fields、prepared_smiles_rule、identity_class_rule、split_pdb_rule、
    language_selection_rule、seen_identity_rule 和 class_names。不覆盖 created_at_utc 或其它划分
    summary。候选与身份 JSONL 不在此解析字段，只按完整字节绑定。
    """

    manifest_path: str
    manifest_sha256: str
    training_manifest_sha256: str | None
    manifest_semantic_sha256: str
    candidates_path: str
    candidates_sha256: str
    identities_path: str
    identities_sha256: str
    fingerprint: str


class CalibrationRuntimeEvidence(TypedDict):
    """记录 calibration 数据加载、随机源、数值后端和 CUDA 环境。

    - candidate_count: 非负整数，构造 calibration CandidateDataset 后调用 len 得到的当时候选数；Dataset 和输入文件不被快照或锁定。
    - physical_batch_size: 正整数，沿用所选阶段的单次模型前向候选数。
    - num_workers: 非负整数，沿用 DataConfig.validation_num_workers。
    - prefetch_factor: 正整数，每个 DataLoader 工作进程预取的物理批次数。
    - cache_gib_per_worker: 非负浮点数，每个 CandidateDataset 实例分给五类 ByteBudgetLru 的总预算配置；DataLoader 场景通常每个 worker 副本各有实例，预算为零或单项超预算时不保留，且不是进程 RSS 硬上限。
    - local_crop_strategy: fixed_center 或 v_centroid，沿用训练计划冻结的候选局部裁剪策略。
    - multiprocessing_context: 字符串 spawn，来自 DATALOADER_MULTIPROCESSING_CONTEXT。
    - persistent_workers: 固定 False，完整 loader 迭代结束后不保留工作进程池。
    - worker_generator_seed: 整数 2*training.seed+2，只用于 calibration DataLoader worker 生成器。
    - numeric_backend: configure_random_seeds_and_numeric_backend 返回的进程级设置记录。
    - numeric_backend.python_numpy_torch_seed: 与所选训练阶段 seed 相同的随机种子。
    - numeric_backend.cudnn_benchmark: cuDNN 是否按输入形状试选算法。
    - numeric_backend.cudnn_deterministic: 是否强制 cuDNN 使用确定性实现。
    - numeric_backend.deterministic_algorithms: 是否要求 Torch 算子采用确定性实现。
    - numeric_backend.float32_matmul_precision: Torch float32 矩阵乘法内部精度策略。
    - device: torch.cuda.get_device_name 返回的实际 CUDA 设备名称。
    - torch_version: 执行一次性评估的 torch.__version__ 字符串。
    """

    candidate_count: int
    physical_batch_size: int
    num_workers: int
    prefetch_factor: int
    cache_gib_per_worker: float
    local_crop_strategy: LocalCropStrategy
    multiprocessing_context: str
    persistent_workers: bool
    worker_generator_seed: int
    numeric_backend: NumericBackendRecord
    device: str
    torch_version: str


class CalibrationArtifact(TypedDict):
    """声明一次成功 calibration 输出 JSON 的完整字段。

    - schema_version: 整数 CALIBRATION_SCHEMA_VERSION，同时约束 artifact 与 claim 格式。
    - split: 固定字符串 calibration。
    - plan_path: 本次入口实际读取且 resolve(strict=True) 成功的实验 YAML 路径文本。
    - plan_sha256: 评估前读取 plan_path 全部字节得到的 SHA-256。
    - plan_name: 实验 YAML 经正式 loader 核验的计划名称。
    - stage_name: validation 已选且在计划中恰好出现一次的阶段名称。
    - task_name: candidate_pretrain 或 full_matcher。
    - language_family: full_matcher 的 molformer 或 smi_ted_289m；candidate_pretrain 为 None。
    - identity_variant: full_matcher 的 oracle、detached 或 joint；candidate_pretrain 为 None。
    - checkpoint: CalibrationCheckpointEvidence。
    - checkpoint.path: BEST.pt 的 POSIX 词法规范化路径文本；不表示符号链接目标已经解析。
    - checkpoint.sha256: BEST.pt 全部字节的 SHA-256。
    - checkpoint.epoch: checkpoint 最佳轮次。
    - checkpoint.global_step: 最佳轮次累计参数更新数。
    - checkpoint.best_score: validation 最佳 selection score。
    - checkpoint.resolved_path: 阶段 resolved.json 路径文本。
    - checkpoint.resolved_sha256: resolved.json 全部字节的 SHA-256。
    - checkpoint.summary_path: 阶段 summary.json 路径文本。
    - checkpoint.summary_sha256: 阶段 summary.json 全部字节的 SHA-256。
    - checkpoint.run_path: 训练 run.json 路径文本。
    - checkpoint.run_sha256: 训练 run.json 全部字节的 SHA-256。
    - checkpoint.training_plan_path: run.json 记录的训练 YAML 路径文本。
    - checkpoint.training_plan_sha256: 训练 YAML 当前全部字节的 SHA-256。
    - checkpoint.model_config: 与所选阶段相等的规范化 MatcherModelConfig 字典。
    - checkpoint.model_config.aggregation: mean_max 或 candidate_cross_attention。
    - checkpoint.model_config.use_manual_features: 是否使用固化手工统计。
    - checkpoint.model_config.use_sampling_features: 是否使用裁剪几何统计。
    - checkpoint.model_config.use_candidate_token_attention_bias: 是否使用候选到 token 的注意力偏置。
    - checkpoint.model_config.initial_graph_gate: 图分数增量门初值。
    - checkpoint.model_config.initial_inventory_gate: 身份清单分数增量门初值。
    - checkpoint.model_config.model_dim: 候选与身份表示宽度。
    - checkpoint.model_config.attention_heads: 候选、图和清单注意力头数。
    - checkpoint.model_config.dropout: 模型共用丢弃概率。
    - checkpoint.model_config.voxel_feature_channels: RAUNet 最高分辨率输出通道数。
    - checkpoint.model_config.raunet_planes: 内存中为九项整数元组，严格 JSON 写出后编码为数组。
    - checkpoint.model_config.recycle_iterations: RAUNet 回收轮数。
    - checkpoint.model_config.gradient_checkpointing: 是否启用训练期激活重计算。
    - checkpoint.model_config.bottleneck_attention_layers: RAUNet 瓶颈注意力层数。
    - checkpoint.model_config.graph_layers: LigandObject 图消息传递层数。
    - checkpoint.training_config: 与所选阶段相等的 TrainingConfig 字典。
    - checkpoint.training_config.seed: 训练和评估随机种子。
    - checkpoint.training_config.physical_batch_size: 单次前向候选数。
    - checkpoint.training_config.gradient_accumulation: 每次参数更新包含的物理批次数。
    - checkpoint.training_config.global_batch_size: 每次参数更新对应的候选数。
    - checkpoint.training_config.learning_rate: AdamW 初始学习率。
    - checkpoint.training_config.weight_decay: AdamW 解耦权重衰减系数。
    - checkpoint.training_config.maximum_epochs: 最大轮数或 None。
    - checkpoint.training_config.patience_bad_epochs: 早停允许的连续未改善轮数。
    - checkpoint.training_config.precision: bf16、fp16 或 fp32。
    - checkpoint.training_config.gradient_clip_norm: 全模型 L2 梯度裁剪阈值。
    - checkpoint.training_config.class_weights: 六项类别权重或 None。
    - checkpoint.training_config.loss_weights: 四项损失系数字典。
    - checkpoint.training_config.loss_weights.classification: 六分类损失系数。
    - checkpoint.training_config.loss_weights.ligand_area: 配体区域损失系数。
    - checkpoint.training_config.loss_weights.binding_area: 结合区域损失系数。
    - checkpoint.training_config.loss_weights.identity: 闭集身份损失系数。
    - data: CalibrationDataEvidence。
    - data.manifest_path: manifest.json 路径文本。
    - data.manifest_sha256: manifest.json 全部字节的 SHA-256。
    - data.training_manifest_sha256: 当前实现固定为 None 的预留字段。
    - data.manifest_semantic_sha256: calibration 相关稳定 manifest 子对象的 SHA-256。
    - data.candidates_path: calibration 候选 JSONL 路径文本。
    - data.candidates_sha256: calibration 候选 JSONL 全部字节的 SHA-256。
    - data.identities_path: calibration 身份 JSONL 路径文本。
    - data.identities_sha256: calibration 身份 JSONL 全部字节的 SHA-256。
    - data.fingerprint: manifest 语义、候选字节与身份字节三个摘要的组合 SHA-256。
    - claim_path: data.fingerprint 对应一次性 claim JSON 的路径文本。
    - runtime: CalibrationRuntimeEvidence。
    - runtime.candidate_count: calibration Dataset 候选数。
    - runtime.physical_batch_size: 单次前向候选数。
    - runtime.num_workers: calibration DataLoader 工作进程数。
    - runtime.prefetch_factor: 每个工作进程预取批次数。
    - runtime.cache_gib_per_worker: 每个 CandidateDataset 实例的五类 LRU 总预算配置；预算不足时可不保留载荷。
    - runtime.local_crop_strategy: fixed_center 或 v_centroid。
    - runtime.multiprocessing_context: 固定 spawn。
    - runtime.persistent_workers: 固定 False。
    - runtime.worker_generator_seed: calibration worker 独立生成器种子。
    - runtime.numeric_backend: 随机源与数值后端设置字典。
    - runtime.numeric_backend.python_numpy_torch_seed: 训练阶段随机种子。
    - runtime.numeric_backend.cudnn_benchmark: cuDNN benchmark 开关。
    - runtime.numeric_backend.cudnn_deterministic: cuDNN 确定性开关。
    - runtime.numeric_backend.deterministic_algorithms: Torch 确定性算法开关。
    - runtime.numeric_backend.float32_matmul_precision: float32 矩阵乘法内部精度策略。
    - runtime.device: 实际 CUDA 设备名称。
    - runtime.torch_version: 实际 PyTorch 版本字符串。
    - evaluation: evaluate_model 返回的完整 EvaluationResult。
    - evaluation.loss: 整个 calibration 划分全局归一的五项损失字典。
    - evaluation.loss.total: 四项加权损失总和。
    - evaluation.loss.classification: 六分类全局归一损失。
    - evaluation.loss.ligand_area: 配体区域逐候选均值损失。
    - evaluation.loss.binding_area: 结合区域逐候选均值损失。
    - evaluation.loss.identity: 有效身份候选均值损失；没有有效身份时为 0。
    - evaluation.metrics: EvaluationAccumulator.compute 返回的 EvaluationReport。
    - evaluation.metrics.six_class: 六分类报告。
    - evaluation.metrics.six_class.accuracy: 全候选六分类准确率。
    - evaluation.metrics.six_class.macro_f1: 与历史选模兼容的支持类 macro-F1。
    - evaluation.metrics.six_class.macro_f1_supported_classes: 只平均真实支持类别的 macro-F1。
    - evaluation.metrics.six_class.macro_f1_all_classes: 固定平均六类的 macro-F1。
    - evaluation.metrics.six_class.per_class: 六个类别名到逐类指标的映射。
    - evaluation.metrics.six_class.per_class[*].precision: 当前类别精确率，零预测时在内存中为 NaN。
    - evaluation.metrics.six_class.per_class[*].recall: 当前类别召回率，零支持时在内存中为 NaN。
    - evaluation.metrics.six_class.per_class[*].f1: 当前类别 F1，未定义时为 0。
    - evaluation.metrics.six_class.per_class[*].support: 当前类别真实候选数。
    - evaluation.metrics.six_class.per_class[*].predicted_count: 当前类别预测候选数。
    - evaluation.metrics.six_class.confusion_matrix: 形状 6×6 的整数嵌套列表，第一轴真实、第二轴预测。
    - evaluation.metrics.intrinsic_identity: 配置身份范围的固有排名报告。
    - evaluation.metrics.intrinsic_identity.top1: 固有身份 Top-1 正确率。
    - evaluation.metrics.intrinsic_identity.topk: 固有身份 Top-K 正确率。
    - evaluation.metrics.intrinsic_identity.top_k: 报告采用的最大名次数。
    - evaluation.metrics.intrinsic_identity.eligible_count: 固有身份可监督候选数。
    - evaluation.metrics.conditional_identity: 配置身份范围的条件排名报告。
    - evaluation.metrics.conditional_identity.top1: 条件身份 Top-1 正确率。
    - evaluation.metrics.conditional_identity.topk: 条件身份 Top-K 正确率。
    - evaluation.metrics.conditional_identity.top_k: 报告采用的最大名次数。
    - evaluation.metrics.conditional_identity.eligible_count: 条件身份可监督候选数。
    - evaluation.metrics.conditional_identity.seen_top1: 训练可见联合身份子集 Top-1。
    - evaluation.metrics.conditional_identity.seen_topk: 训练可见联合身份子集 Top-K。
    - evaluation.metrics.conditional_identity.seen_count: 训练可见联合身份子集候选数。
    - evaluation.metrics.conditional_identity.unseen_top1: 训练未见联合身份子集 Top-1。
    - evaluation.metrics.conditional_identity.unseen_topk: 训练未见联合身份子集 Top-K。
    - evaluation.metrics.conditional_identity.unseen_count: 训练未见联合身份子集候选数。
    - evaluation.metrics.end_to_end_identity: 配置身份范围的端到端报告。
    - evaluation.metrics.end_to_end_identity.precision: 端到端身份精确率。
    - evaluation.metrics.end_to_end_identity.recall: 端到端身份召回率。
    - evaluation.metrics.end_to_end_identity.f1: 端到端身份 F1。
    - evaluation.metrics.end_to_end_identity.true_positive: 端到端身份真阳性数。
    - evaluation.metrics.end_to_end_identity.predicted_positive: 端到端身份预测阳性数。
    - evaluation.metrics.end_to_end_identity.actual_positive: 端到端身份真实阳性数。
    - evaluation.metrics.small_molecule_identity: 小分子兼容口径报告。
    - evaluation.metrics.small_molecule_identity.intrinsic: 小分子固有排名对象。
    - evaluation.metrics.small_molecule_identity.intrinsic.top1: 小分子固有 Top-1。
    - evaluation.metrics.small_molecule_identity.intrinsic.topk: 小分子固有 Top-K。
    - evaluation.metrics.small_molecule_identity.intrinsic.top_k: 报告采用的最大名次数。
    - evaluation.metrics.small_molecule_identity.intrinsic.eligible_count: 小分子固有分母。
    - evaluation.metrics.small_molecule_identity.conditional: 小分子条件排名对象。
    - evaluation.metrics.small_molecule_identity.conditional.top1: 小分子条件 Top-1。
    - evaluation.metrics.small_molecule_identity.conditional.topk: 小分子条件 Top-K。
    - evaluation.metrics.small_molecule_identity.conditional.top_k: 报告采用的最大名次数。
    - evaluation.metrics.small_molecule_identity.conditional.eligible_count: 小分子条件分母。
    - evaluation.metrics.small_molecule_identity.end_to_end: 小分子端到端对象。
    - evaluation.metrics.small_molecule_identity.end_to_end.precision: 小分子端到端精确率。
    - evaluation.metrics.small_molecule_identity.end_to_end.recall: 小分子端到端召回率。
    - evaluation.metrics.small_molecule_identity.end_to_end.f1: 小分子端到端 F1。
    - evaluation.metrics.small_molecule_identity.end_to_end.true_positive: 小分子端到端真阳性数。
    - evaluation.metrics.small_molecule_identity.end_to_end.predicted_positive: 小分子端到端预测阳性数。
    - evaluation.metrics.small_molecule_identity.end_to_end.actual_positive: 小分子端到端真实阳性数。
    - evaluation.metrics.all_non_background_identity: 五类总体口径报告。
    - evaluation.metrics.all_non_background_identity.intrinsic: 五类总体固有排名对象。
    - evaluation.metrics.all_non_background_identity.intrinsic.top1: 五类总体固有 Top-1。
    - evaluation.metrics.all_non_background_identity.intrinsic.topk: 五类总体固有 Top-K。
    - evaluation.metrics.all_non_background_identity.intrinsic.top_k: 报告采用的最大名次数。
    - evaluation.metrics.all_non_background_identity.intrinsic.eligible_count: 五类总体固有分母。
    - evaluation.metrics.all_non_background_identity.conditional: 五类总体条件排名对象。
    - evaluation.metrics.all_non_background_identity.conditional.top1: 五类总体条件 Top-1。
    - evaluation.metrics.all_non_background_identity.conditional.topk: 五类总体条件 Top-K。
    - evaluation.metrics.all_non_background_identity.conditional.top_k: 报告采用的最大名次数。
    - evaluation.metrics.all_non_background_identity.conditional.eligible_count: 五类总体条件分母。
    - evaluation.metrics.all_non_background_identity.end_to_end: 五类总体端到端对象。
    - evaluation.metrics.all_non_background_identity.end_to_end.precision: 五类总体端到端精确率。
    - evaluation.metrics.all_non_background_identity.end_to_end.recall: 五类总体端到端召回率。
    - evaluation.metrics.all_non_background_identity.end_to_end.f1: 五类总体端到端 F1。
    - evaluation.metrics.all_non_background_identity.end_to_end.true_positive: 五类总体端到端真阳性数。
    - evaluation.metrics.all_non_background_identity.end_to_end.predicted_positive: 五类总体端到端预测阳性数。
    - evaluation.metrics.all_non_background_identity.end_to_end.actual_positive: 五类总体端到端真实阳性数。
    - evaluation.metrics.identity_per_class: 五个前景类别名到范围报告的映射。
    - evaluation.metrics.identity_per_class[*].intrinsic: 当前类别固有排名对象。
    - evaluation.metrics.identity_per_class[*].intrinsic.top1: 当前类别固有 Top-1。
    - evaluation.metrics.identity_per_class[*].intrinsic.topk: 当前类别固有 Top-K。
    - evaluation.metrics.identity_per_class[*].intrinsic.top_k: 报告采用的最大名次数。
    - evaluation.metrics.identity_per_class[*].intrinsic.eligible_count: 当前类别固有分母。
    - evaluation.metrics.identity_per_class[*].conditional: 当前类别条件排名对象。
    - evaluation.metrics.identity_per_class[*].conditional.top1: 当前类别条件 Top-1。
    - evaluation.metrics.identity_per_class[*].conditional.topk: 当前类别条件 Top-K。
    - evaluation.metrics.identity_per_class[*].conditional.top_k: 报告采用的最大名次数。
    - evaluation.metrics.identity_per_class[*].conditional.eligible_count: 当前类别条件分母。
    - evaluation.metrics.identity_per_class[*].end_to_end: 当前类别端到端对象。
    - evaluation.metrics.identity_per_class[*].end_to_end.precision: 当前类别端到端精确率。
    - evaluation.metrics.identity_per_class[*].end_to_end.recall: 当前类别端到端召回率。
    - evaluation.metrics.identity_per_class[*].end_to_end.f1: 当前类别端到端 F1。
    - evaluation.metrics.identity_per_class[*].end_to_end.true_positive: 当前类别端到端真阳性数。
    - evaluation.metrics.identity_per_class[*].end_to_end.predicted_positive: 当前类别端到端预测阳性数。
    - evaluation.metrics.identity_per_class[*].end_to_end.actual_positive: 当前类别端到端真实阳性数。
    - evaluation.metrics.area: 两个区域头的微平均 Dice 报告。
    - evaluation.metrics.area.ligand_micro_dice: 配体区域微平均 Dice。
    - evaluation.metrics.area.binding_micro_dice: 结合区域微平均 Dice。
    - evaluation.metrics.area.ligand_dice_denominator: 配体区域预测与真实体素数之和。
    - evaluation.metrics.area.binding_dice_denominator: 结合区域预测与真实体素数之和。
    - evaluation.sample_count: 实际遍历的 calibration 候选数，成功前必须等于 runtime.candidate_count。
    - evaluation.identity_eligible_count: compute_matcher_loss 的 identity_eligible_mask 计数；除 identity_target 非负外还受当前 stage.identity_scope 限定，candidate_pretrain 或 identity_scope=None 时为零。
    - evaluation.elapsed_seconds: evaluate_model 测得的墙钟秒数。
    - evaluation.samples_per_second: sample_count/elapsed_seconds 的吞吐率。
    - hostname: socket.gethostname 返回的执行节点名称。
    - slurm_job_id: 环境变量 SLURM_JOB_ID 的字符串值；未设置时为 None，本入口不验证其格式。
    - task_launch_dir: 环境变量 TASK_LAUNCH_DIR 的字符串值；未设置时为 None，本入口不验证路径存在或内容。
    - started_at_utc: claim 创建或复用前生成的带时区 ISO 8601 时间字符串。
    - finished_at_utc: evaluate_model 与样本数核验完成后生成的带时区 ISO 8601 时间字符串。

    本类型没有独立 identity_scope 字段，identity_eligible_count 的范围只能结合本次绑定的计划、
    resolved 与 checkpoint 配置解释。它也不证明远端调度状态或 launch 目录内容；相关两个环境
    字段只是运行时文本。严格 JSON 写入器会把 EvaluationReport 中的 NaN 以及跨批 Python float
    损失聚合理论上可能产生的正负 Inf 都规范化为 null；内存 TypedDict 中对应字段仍声明为 float。
    """

    schema_version: int
    split: Literal["calibration"]
    plan_path: str
    plan_sha256: str
    plan_name: str
    stage_name: str
    task_name: str
    language_family: str | None
    identity_variant: str | None
    checkpoint: CalibrationCheckpointEvidence
    data: CalibrationDataEvidence
    claim_path: str
    runtime: CalibrationRuntimeEvidence
    evaluation: EvaluationResult
    hostname: str
    slurm_job_id: str | None
    task_launch_dir: str | None
    started_at_utc: str
    finished_at_utc: str


class CalibrationClaim(TypedDict):
    """在 claim 文件留存期间，把 calibration 数据指纹绑定到一个模型选择和结果路径。

    - schema_version: 整数 CALIBRATION_SCHEMA_VERSION，必须与恢复时输出 artifact 版本相等。
    - status: claimed 或 completed；claimed 允许同一不可变选择重试，claim 文件保留时 completed 阻止合作调用者再次评估。
    - data_fingerprint: manifest 语义、calibration 候选字节和身份字节的组合 SHA-256。
    - plan_sha256: 首次 claim 时选定实验 YAML 全部字节的 SHA-256。
    - stage_name: 首次 claim 时选定的阶段名称。
    - checkpoint_sha256: 首次 claim 时选定 BEST.pt 全部字节的 SHA-256。
    - checkpoint_path: 首次 claim 时选定 BEST.pt 的路径文本。
    - output_path: 首次 claim 时绑定的结果 JSON 路径文本；重试不能改用其它路径。
    - output_sha256: completed 时为已发布结果全部字节的 SHA-256；claimed 时必须为 None。
    - claimed_at_utc: 首次创建 claim 时的带时区 ISO 8601 时间文本；同选择重试保留旧值。
    - completed_at_utc: completed 时为结果 artifact.finished_at_utc；claimed 时必须为 None。

    claim 文件保留期间，completed 是唯一禁止合作调用者重评的本地状态；删除 claim 后仍可按
    新输出路径建立新 claim。互斥锁文件是否存在不代表 claimed 或 completed；锁只串行同一数据
    fingerprint 且争用该锁的调用者，output_path 没有独立全局锁，外部写者或不同 fingerprint
    调用可与 exists 检查及 write_json 原子替换竞争。
    """

    schema_version: int
    status: Literal["claimed", "completed"]
    data_fingerprint: str
    plan_sha256: str
    stage_name: str
    checkpoint_sha256: str
    checkpoint_path: str
    output_path: str
    output_sha256: str | None
    claimed_at_utc: str
    completed_at_utc: str | None


def _select_stage(plan: ExperimentPlan, stage_name: str) -> StageConfig:
    """按完整名称从实验计划中选择恰好一个阶段。

    - plan: 已由 load_experiment_plan 核验的 ExperimentPlan。
    - plan.plan_name: 本函数不读取的计划名称。
    - plan.data: 本函数不读取的数据配置。
    - plan.tracking: 本函数不读取的 W&B 配置。
    - plan.stages: 保持 YAML 顺序的 StageConfig 元组；本函数只读取每项 name。
    - stage_name: 调用方要求的完整阶段名称，按 Python 字符串精确相等比较。
    - 返回值.name: 与 stage_name 相等的唯一阶段名称。
    - 返回值.task_name: candidate_pretrain 或 full_matcher。
    - 返回值.language_family: full_matcher 的语言家族，候选预训练为 None。
    - 返回值.identity_variant: full_matcher 的身份损失变体，候选预训练为 None。
    - 返回值.identity_scope: full_matcher 的身份评估范围，候选预训练为 None。
    - 返回值.model: 所选阶段的 MatcherModelConfig。
    - 返回值.model.aggregation: mean_max 或 candidate_cross_attention。
    - 返回值.model.use_manual_features: 是否使用固化手工统计。
    - 返回值.model.use_sampling_features: 是否使用裁剪几何统计。
    - 返回值.model.use_candidate_token_attention_bias: 是否使用候选到 token 的注意力偏置。
    - 返回值.model.initial_graph_gate: 图分数增量门初值。
    - 返回值.model.initial_inventory_gate: 身份清单分数增量门初值。
    - 返回值.model.model_dim: 候选与身份表示宽度。
    - 返回值.model.attention_heads: 候选、图和清单注意力头数。
    - 返回值.model.dropout: 模型共用丢弃概率。
    - 返回值.model.voxel_feature_channels: RAUNet 最高分辨率输出通道数。
    - 返回值.model.raunet_planes: 九项 RAUNet 通道元组。
    - 返回值.model.recycle_iterations: RAUNet 回收轮数。
    - 返回值.model.gradient_checkpointing: 是否启用训练期激活重计算。
    - 返回值.model.bottleneck_attention_layers: RAUNet 瓶颈注意力层数。
    - 返回值.model.graph_layers: LigandObject 图消息传递层数。
    - 返回值.training: 所选阶段的 TrainingConfig。
    - 返回值.training.seed: 训练和评估随机种子。
    - 返回值.training.physical_batch_size: 单次前向候选数。
    - 返回值.training.gradient_accumulation: 每次参数更新的物理批次数。
    - 返回值.training.global_batch_size: 每次参数更新的候选数。
    - 返回值.training.learning_rate: AdamW 初始学习率。
    - 返回值.training.weight_decay: AdamW 解耦权重衰减。
    - 返回值.training.maximum_epochs: 最大轮数或 None。
    - 返回值.training.patience_bad_epochs: 早停允许的连续未改善轮数。
    - 返回值.training.precision: bf16、fp16 或 fp32。
    - 返回值.training.gradient_clip_norm: 全模型 L2 梯度裁剪阈值。
    - 返回值.training.class_weights: 六项类别权重元组或 None。
    - 返回值.training.loss_weights: 四项 LossWeights。
    - 返回值.training.loss_weights.classification: 六分类损失系数。
    - 返回值.training.loss_weights.ligand_area: 配体区域损失系数。
    - 返回值.training.loss_weights.binding_area: 结合区域损失系数。
    - 返回值.training.loss_weights.identity: 闭集身份损失系数。
    - 返回值.initialize_candidate_from: 候选编码器初始化 BEST.pt 的 POSIX 绝对路径文本或 None。

    名称不存在或重复都会抛出 ValueError；函数不按位置回退，也不复制 frozen StageConfig。
    """

    # matches 保持 plan.stages 原顺序，且只收集名称精确相等的 StageConfig。
    matches = [stage for stage in plan.stages if stage.name == stage_name]
    if len(matches) != 1:
        raise ValueError(f"计划中必须恰好存在一个 stage={stage_name!r}，实际为 {len(matches)}。")
    return matches[0]


def _sha256_file(path: Path) -> str:
    """按固定块大小流式读取一个文件并返回内容 SHA-256。

    - path: 待读取文件路径；本函数不预先调用 is_file 或 resolve。
    - 返回值: 64 个小写十六进制字符组成的摘要字符串。

    文件打开或读取异常直接传播。摘要绑定调用期间实际读取的字节，但本函数不提供文件锁，
    也不检测读取期间文件是否被另一个进程替换。
    """

    # digest 是空 SHA-256 状态，按文件顺序接收最多 8 MiB 的字节块。
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_json_constant(token: str) -> NoReturn:
    """作为 json.loads 回调拒绝非标准的非有限数值常量。

    - token: 解码器遇到的 NaN、Infinity 或 -Infinity 文本。
    - 返回值: 不返回；始终抛出 ValueError。
    """

    raise ValueError(f"严格 JSON 不得包含非有限常量 {token}。")


def _read_json_mapping(path: Path, name: str) -> dict[str, Any]:
    """读取 UTF-8 JSON，并要求顶层是字符串键对象。

    - path: 输入 JSON 文件路径；必须由 is_file 判定为普通文件或指向普通文件的有效路径。
    - name: 仅用于错误消息的实体名称。
    - 返回值: 可变 dict[str,Any]；保留解码后的全部顶层键值。

    parse_constant 回调拒绝 NaN 与正负 Infinity。函数不检测重复 JSON 键，Python 解码器会
    保留后出现的同名值；也不递归验证嵌套字段 schema。文件缺失时显式抛出 FileNotFoundError，
    UTF-8、JSON 语法或顶层类型错误直接失败。
    """

    if not path.is_file():
        raise FileNotFoundError(path)
    # value 是严格常量解码后的任意 JSON 顶层实体，随后收窄到字符串键 dict。
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} 必须是字符串键 JSON 对象。")
    return cast(dict[str, Any], value)


def _calibration_data_evidence(manifest_path: Path) -> CalibrationDataEvidence:
    """计算 manifest、calibration 候选与身份记录的内容证据。

    - manifest_path: 现存 manifest.json 路径；本函数不根据 SourceRoots 推导它。
    - manifest_path.splits: 必须是 JSON 对象。
    - manifest_path.splits.calibration: 必须是 JSON 对象。
    - manifest_path.splits.calibration.candidates: manifest 同目录内的单级文件名字符串。
    - manifest_path.splits.calibration.identities: manifest 同目录内的单级文件名字符串。
    - manifest_path.splits.calibration 的附加键: 不解析或验证，但随完整 calibration 对象进入语义摘要。
    - manifest_path.schema_version: 原样纳入语义摘要，可缺失并以 null 编码。
    - manifest_path.data_root: 原样纳入语义摘要，可缺失并以 null 编码。
    - manifest_path.stage1_root: 原样纳入语义摘要，可缺失并以 null 编码。
    - manifest_path.output_root: 原样纳入语义摘要，可缺失并以 null 编码。
    - manifest_path.coverage_threshold: 原样纳入语义摘要，可缺失并以 null 编码。
    - manifest_path.selection_seed: 原样纳入语义摘要，可缺失并以 null 编码。
    - manifest_path.identity_key_fields: 原样纳入语义摘要，可缺失并以 null 编码。
    - manifest_path.prepared_smiles_rule: 原样纳入语义摘要，可缺失并以 null 编码。
    - manifest_path.identity_class_rule: 原样纳入语义摘要，可缺失并以 null 编码。
    - manifest_path.split_pdb_rule: 原样纳入语义摘要，可缺失并以 null 编码。
    - manifest_path.language_selection_rule: 原样纳入语义摘要，可缺失并以 null 编码。
    - manifest_path.seen_identity_rule: 原样纳入语义摘要，可缺失并以 null 编码。
    - manifest_path.class_names: 原样纳入语义摘要，可缺失并以 null 编码。
    - manifest_path.split_pdb_counts.calibration: 若 split_pdb_counts 是映射则把其 calibration 键的完整 JSON 值纳入语义摘要，否则以 null 编码；该值不要求是对象。
    - manifest_path.split_pdb_counts.calibration.discovered: 正式 manifest 的已知 PDB 目录发现数；本函数不单独验证。
    - manifest_path.split_pdb_counts.calibration.included: 正式 manifest 的已知纳入 PDB 数；本函数不单独验证。
    - manifest_path.split_pdb_counts.calibration.excluded: 正式 manifest 的已知排除 PDB 数；本函数不单独验证。
    - manifest_path.split_pdb_counts.calibration 的附加键: 不解析或验证，但随完整对象进入语义摘要。
    - manifest_path.summary.splits.calibration: 若两层父对象均为映射则把其 calibration 键的完整 JSON 值纳入语义摘要，否则以 null 编码；该值不要求是对象。
    - manifest_path.summary.splits.calibration.candidate_count: 正式 manifest 的已知候选数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.class_counts: 正式 manifest 的已知类别计数映射；本函数不单独验证其键值。
    - manifest_path.summary.splits.calibration.class_counts.background: 条件字段，正式 manifest 的背景候选数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.class_counts.ion: 条件字段，正式 manifest 的离子候选数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.class_counts.nucleotide_like: 条件字段，正式 manifest 的核苷酸样候选数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.class_counts.peptide_like: 条件字段，正式 manifest 的肽样候选数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.class_counts.small_molecule: 条件字段，正式 manifest 的小分子候选数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.class_counts.sugar: 条件字段，正式 manifest 的糖候选数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.identity_count: 正式 manifest 的已知联合身份数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.unique_object_key_count: 正式 manifest 的已知不同模板键数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.same_pdb_object_key_multi_prepared_smiles_group_count: 正式 manifest 的已知同 PDB 多字符串分组数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.matched_foreground_without_prepared_smiles_candidate_count: 正式 manifest 的已知无顶层字符串前景候选数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.matched_foreground_explicit_null_candidate_count: 正式 manifest 的已知显式 null 前景候选数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.matched_foreground_missing_prepared_file_candidate_count: 正式 manifest 的已知缺 PreparedLigand 文件前景候选数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.language_available_identity_count: 正式 manifest 的已知逐语言可用身份映射；本函数不单独验证。
    - manifest_path.summary.splits.calibration.language_available_identity_count.molformer: 正式 manifest 的 MoLFormer 可用身份数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.language_available_identity_count.smi_ted_289m: 正式 manifest 的 SMI-TED 可用身份数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.language_multi_model_smiles_identity_count: 正式 manifest 的已知逐语言多字符串身份映射；本函数不单独验证。
    - manifest_path.summary.splits.calibration.language_multi_model_smiles_identity_count.molformer: 正式 manifest 的 MoLFormer 多字符串身份数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.language_multi_model_smiles_identity_count.smi_ted_289m: 正式 manifest 的 SMI-TED 多字符串身份数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.seen_positive_identity_count: 正式 manifest 的已知逐语言训练可见身份映射；本函数不单独验证。
    - manifest_path.summary.splits.calibration.seen_positive_identity_count.molformer: 正式 manifest 的 MoLFormer 口径可见身份数；本函数不单独验证。
    - manifest_path.summary.splits.calibration.seen_positive_identity_count.smi_ted_289m: 正式 manifest 的 SMI-TED 口径可见身份数；本函数不单独验证。
    - manifest_path.summary.splits.calibration 的附加键: 不解析或验证，但随完整对象进入语义摘要。
    - 返回值.manifest_path: 输入路径的 POSIX 文本。
    - 返回值.manifest_sha256: manifest 文件全部字节的 SHA-256。
    - 返回值.training_manifest_sha256: 当前实现固定为 None。
    - 返回值.manifest_semantic_sha256: calibration 相关稳定子对象的规范 JSON SHA-256。
    - 返回值.candidates_path: manifest 同目录内候选 JSONL 的 POSIX 路径文本。
    - 返回值.candidates_sha256: 候选 JSONL 全部字节的 SHA-256。
    - 返回值.identities_path: manifest 同目录内身份 JSONL 的 POSIX 路径文本。
    - 返回值.identities_sha256: 身份 JSONL 全部字节的 SHA-256。
    - 返回值.fingerprint: 三个内容摘要按固定换行公式连接后的 SHA-256。

    本函数只解析 manifest，不解析两个 JSONL 内部记录。manifest.created_at_utc 和其它划分
    summary 不进入语义摘要，但仍会改变 manifest_sha256；任一 calibration JSONL 字节变化会
    改变对应摘要和 fingerprint。manifest 的 JSON 解析与完整字节摘要是两次独立文件读取，
    不共享同一内存字节缓冲；函数依赖正式输入在 calibration 期间保持不变，不提供快照锁。
    """

    # manifest 是严格常量解码后的顶层字符串键对象；嵌套 schema 在下方按消费位置核验。
    manifest = _read_json_mapping(manifest_path, "manifest.json")
    # splits/calibration 是两层 JSON 对象，后者至少要提供两个单级文件名。
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("manifest.json.splits 必须是 JSON 对象。")
    calibration = splits.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("manifest.json.splits.calibration 必须是 JSON 对象。")
    # candidate_name/identity_name 是字符串文件名；禁止绝对路径或任何目录层级。
    candidate_name = calibration.get("candidates")
    identity_name = calibration.get("identities")
    if not isinstance(candidate_name, str) or Path(candidate_name).name != candidate_name:
        raise ValueError("calibration candidates 必须是清单目录内的单级文件名。")
    if not isinstance(identity_name, str) or Path(identity_name).name != identity_name:
        raise ValueError("calibration identities 必须是清单目录内的单级文件名。")
    # 两个路径都锚定 manifest 目录，不调用 resolve，随后由文件读取失败关闭。
    candidates_path = manifest_path.parent / candidate_name
    identities_path = manifest_path.parent / identity_name
    # manifest_sha256 绑定完整 manifest 字节，与下方只绑定稳定子对象的语义摘要并行保存。
    manifest_sha256 = _sha256_file(manifest_path)
    # split_pdb_counts 和 summary_splits 保留任意 Mapping；这里只抽取 calibration 键的完整 JSON 值。
    split_pdb_counts = manifest.get("split_pdb_counts")
    summary = manifest.get("summary")
    summary_splits = summary.get("splits") if isinstance(summary, Mapping) else None
    # semantic_manifest 只含身份与划分规则、三根路径和 calibration 文件/计数/summary。
    semantic_manifest = {
        "schema_version": manifest.get("schema_version"),
        "data_root": manifest.get("data_root"),
        "stage1_root": manifest.get("stage1_root"),
        "output_root": manifest.get("output_root"),
        "coverage_threshold": manifest.get("coverage_threshold"),
        "selection_seed": manifest.get("selection_seed"),
        "identity_key_fields": manifest.get("identity_key_fields"),
        "prepared_smiles_rule": manifest.get("prepared_smiles_rule"),
        "identity_class_rule": manifest.get("identity_class_rule"),
        "split_pdb_rule": manifest.get("split_pdb_rule"),
        "language_selection_rule": manifest.get("language_selection_rule"),
        "seen_identity_rule": manifest.get("seen_identity_rule"),
        "class_names": manifest.get("class_names"),
        "calibration_files": calibration,
        "calibration_pdb_counts": (
            split_pdb_counts.get("calibration")
            if isinstance(split_pdb_counts, Mapping)
            else None
        ),
        "calibration_summary": (
            summary_splits.get("calibration") if isinstance(summary_splits, Mapping) else None
        ),
    }
    # semantic_bytes 是 UTF-8、按键排序、无空白且禁止 NaN/Inf 的稳定 JSON 字节。
    semantic_bytes = json.dumps(
        semantic_manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    # 三个叶文件摘要均为 64 个小写十六进制字符组成的字符串。
    manifest_semantic_sha256 = hashlib.sha256(semantic_bytes).hexdigest()
    candidates_sha256 = _sha256_file(candidates_path)
    identities_sha256 = _sha256_file(identities_path)
    # fingerprint 绑定三个摘要的固定顺序和每项后的单个换行符。
    fingerprint = hashlib.sha256(
        (
            f"{manifest_semantic_sha256}\n"
            f"{candidates_sha256}\n"
            f"{identities_sha256}\n"
        ).encode("ascii")
    ).hexdigest()
    return {
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": manifest_sha256,
        "training_manifest_sha256": None,
        "manifest_semantic_sha256": manifest_semantic_sha256,
        "candidates_path": candidates_path.as_posix(),
        "candidates_sha256": candidates_sha256,
        "identities_path": identities_path.as_posix(),
        "identities_sha256": identities_sha256,
        "fingerprint": fingerprint,
    }


def _validate_checkpoint_for_stage(
    checkpoint_path: Path,
    checkpoint: Mapping[str, object],
    plan: ExperimentPlan,
    stage: StageConfig,
) -> CalibrationCheckpointEvidence:
    """核对所选阶段、BEST checkpoint 和相邻训练产物的自洽字段。

    - checkpoint_path: 已存在 BEST.pt 的路径；本函数用其目录定位 resolved.json、summary.json 和 run.json，但不再次核验服务器产物根或文件名。
    - checkpoint: load_checkpoint 返回的字符串键映射；本函数不核验 schema_version 或 model_state，它们由 checkpoint loader 和后续模型加载消费。
    - checkpoint.stage_name: 必须精确等于 stage.name。
    - checkpoint.task_name: 必须精确等于 stage.task_name。
    - checkpoint.epoch: type(...) is int 的非负最佳轮次，必须等于 best_epoch 与 epoch_record.epoch。
    - checkpoint.best_epoch: type(...) is int 的非负最佳轮次，必须等于 epoch。
    - checkpoint.global_step: type(...) is int 的非负参数更新数，必须等于 epoch_record.global_step。
    - checkpoint.best_score: 非 bool 的 int 或 float 实例，转成 float 后必须有限，并等于 epoch_record.selection_score。
    - checkpoint.model_config: 字符串键映射，经 MatcherModelConfig.from_dict 规范化后必须等于 stage.model.to_dict。
    - checkpoint.model_config.aggregation: 必须等于 stage.model.aggregation。
    - checkpoint.model_config.use_manual_features: 必须等于 stage.model.use_manual_features。
    - checkpoint.model_config.use_sampling_features: 必须等于 stage.model.use_sampling_features。
    - checkpoint.model_config.use_candidate_token_attention_bias: 必须等于 stage.model.use_candidate_token_attention_bias。
    - checkpoint.model_config.initial_graph_gate: 缺失时由正式配置迁移补 0.0，规范化后必须等于 stage.model.initial_graph_gate。
    - checkpoint.model_config.initial_inventory_gate: 缺失时由正式配置迁移补 0.0，规范化后必须等于 stage.model.initial_inventory_gate。
    - checkpoint.model_config.model_dim: 必须等于 stage.model.model_dim。
    - checkpoint.model_config.attention_heads: 必须等于 stage.model.attention_heads。
    - checkpoint.model_config.dropout: 必须等于 stage.model.dropout。
    - checkpoint.model_config.voxel_feature_channels: 必须等于 stage.model.voxel_feature_channels。
    - checkpoint.model_config.raunet_planes: 元组或列表输入经正式配置规范化后必须等于 stage.model.raunet_planes。
    - checkpoint.model_config.recycle_iterations: 必须等于 stage.model.recycle_iterations。
    - checkpoint.model_config.gradient_checkpointing: 必须等于 stage.model.gradient_checkpointing。
    - checkpoint.model_config.bottleneck_attention_layers: 必须等于 stage.model.bottleneck_attention_layers。
    - checkpoint.model_config.graph_layers: 必须等于 stage.model.graph_layers。
    - checkpoint.training_config: 字符串键映射，必须等于 asdict(stage.training)。
    - checkpoint.training_config.seed: 必须等于 stage.training.seed。
    - checkpoint.training_config.physical_batch_size: 必须等于 stage.training.physical_batch_size。
    - checkpoint.training_config.gradient_accumulation: 必须等于 stage.training.gradient_accumulation。
    - checkpoint.training_config.global_batch_size: 必须等于 stage.training.global_batch_size。
    - checkpoint.training_config.learning_rate: 必须等于 stage.training.learning_rate。
    - checkpoint.training_config.weight_decay: 必须等于 stage.training.weight_decay。
    - checkpoint.training_config.maximum_epochs: 必须等于 stage.training.maximum_epochs。
    - checkpoint.training_config.patience_bad_epochs: 必须等于 stage.training.patience_bad_epochs。
    - checkpoint.training_config.precision: 必须等于 stage.training.precision。
    - checkpoint.training_config.gradient_clip_norm: 必须等于 stage.training.gradient_clip_norm。
    - checkpoint.training_config.class_weights: 必须等于 stage.training.class_weights。
    - checkpoint.training_config.loss_weights: 必须等于 asdict(stage.training.loss_weights)。
    - checkpoint.training_config.loss_weights.classification: 必须等于 stage.training.loss_weights.classification。
    - checkpoint.training_config.loss_weights.ligand_area: 必须等于 stage.training.loss_weights.ligand_area。
    - checkpoint.training_config.loss_weights.binding_area: 必须等于 stage.training.loss_weights.binding_area。
    - checkpoint.training_config.loss_weights.identity: 必须等于 stage.training.loss_weights.identity。
    - checkpoint.epoch_record: 字符串键映射，允许含本函数不读取的其它 EpochRecord 字段。
    - checkpoint.epoch_record.epoch: 必须等于 checkpoint.epoch。
    - checkpoint.epoch_record.global_step: 必须等于 checkpoint.global_step。
    - checkpoint.epoch_record.selection_score: 必须等于 checkpoint.best_score。
    - checkpoint.epoch_record.improved: 必须为 True。
    - plan: 当前 calibration YAML 经 load_experiment_plan 得到的 ExperimentPlan。
    - plan.plan_name: 必须等于训练 run.json.plan_name。
    - plan.data.roots: data_root、stage1_root、output_root 三个路径必须与 resolved.json.data 文本相等。
    - plan.data.coverage_threshold: 必须与 resolved.json.data.coverage_threshold 相等。
    - plan.data.selection_seed: 必须与 resolved.json.data.selection_seed 相等。
    - plan.data.pp_top_k: 必须与 resolved.json.data.pp_top_k 相等。
    - plan.data.local_crop_strategy: 必须与 resolved.json.data.local_crop_strategy 相等。
    - stage: 当前计划中已唯一选择的 StageConfig。
    - stage.name: 与 checkpoint、resolved stage 和 summary stage 交叉核对。
    - stage.task_name: 与 checkpoint 和 resolved stage 交叉核对。
    - stage.language_family: 作为完整 expected_stage 的一部分核对。
    - stage.identity_variant: 作为完整 expected_stage 的一部分核对。
    - stage.identity_scope: 作为完整 expected_stage 的一部分核对；历史 resolved 缺失时按任务补入兼容值。
    - stage.model: 规范化为 JSON 值后与 resolved.json.stage.model、checkpoint.model_config 比较。
    - stage.model.aggregation: 与两份模型配置中的同名字段比较。
    - stage.model.use_manual_features: 与两份模型配置中的同名字段比较。
    - stage.model.use_sampling_features: 与两份模型配置中的同名字段比较。
    - stage.model.use_candidate_token_attention_bias: 与两份模型配置中的同名字段比较。
    - stage.model.initial_graph_gate: 与两份模型配置中的同名字段比较。
    - stage.model.initial_inventory_gate: 与两份模型配置中的同名字段比较。
    - stage.model.model_dim: 与两份模型配置中的同名字段比较。
    - stage.model.attention_heads: 与两份模型配置中的同名字段比较。
    - stage.model.dropout: 与两份模型配置中的同名字段比较。
    - stage.model.voxel_feature_channels: 与两份模型配置中的同名字段比较。
    - stage.model.raunet_planes: JSON 比较时编码为数组。
    - stage.model.recycle_iterations: 与两份模型配置中的同名字段比较。
    - stage.model.gradient_checkpointing: 与两份模型配置中的同名字段比较。
    - stage.model.bottleneck_attention_layers: 与两份模型配置中的同名字段比较。
    - stage.model.graph_layers: 与两份模型配置中的同名字段比较。
    - stage.training: asdict 后与 resolved.json.stage.training、checkpoint.training_config 比较。
    - stage.training.seed: 与两份训练配置中的同名字段比较。
    - stage.training.physical_batch_size: 与两份训练配置中的同名字段比较。
    - stage.training.gradient_accumulation: 与两份训练配置中的同名字段比较。
    - stage.training.global_batch_size: 与两份训练配置中的同名字段比较。
    - stage.training.learning_rate: 与两份训练配置中的同名字段比较。
    - stage.training.weight_decay: 与两份训练配置中的同名字段比较。
    - stage.training.maximum_epochs: 与两份训练配置中的同名字段比较。
    - stage.training.patience_bad_epochs: 与两份训练配置中的同名字段比较。
    - stage.training.precision: 与两份训练配置中的同名字段比较。
    - stage.training.gradient_clip_norm: 与两份训练配置中的同名字段比较。
    - stage.training.class_weights: 与两份训练配置中的同名字段比较。
    - stage.training.loss_weights: 与两份训练配置中的同名对象比较。
    - stage.training.loss_weights.classification: 与两份训练配置中的同名字段比较。
    - stage.training.loss_weights.ligand_area: 与两份训练配置中的同名字段比较。
    - stage.training.loss_weights.binding_area: 与两份训练配置中的同名字段比较。
    - stage.training.loss_weights.identity: 与两份训练配置中的同名字段比较。
    - stage.initialize_candidate_from: 作为完整 expected_stage 的一部分与 resolved.json.stage 比较。
    - resolved.json.stage: 必须是映射，补历史默认值后必须与当前 stage 的完整 JSON 值相等。
    - resolved.json.stage.model: 必须是字符串键映射，经 MatcherModelConfig.from_dict 规范化后比较。
    - resolved.json.stage.identity_scope: 缺失时 full_matcher 补 small_molecule、candidate_pretrain 补 None。
    - resolved.json.data: 必须是映射；只消费下列七个科学字段。
    - resolved.json.data.data_root: 必须等于 plan.data.roots.data_root 的字符串。
    - resolved.json.data.stage1_root: 必须等于 plan.data.roots.stage1_root 的字符串。
    - resolved.json.data.output_root: 必须等于 plan.data.roots.output_root 的字符串。
    - resolved.json.data.coverage_threshold: 必须等于 plan.data.coverage_threshold。
    - resolved.json.data.selection_seed: 必须等于 plan.data.selection_seed。
    - resolved.json.data.pp_top_k: 必须等于 plan.data.pp_top_k。
    - resolved.json.data.local_crop_strategy: 缺失时补 fixed_center，随后必须等于 plan.data.local_crop_strategy。
    - summary.json.stage: 必须等于 stage.name。
    - summary.json.best_epoch: 必须等于 checkpoint.epoch。
    - summary.json.best_score: 必须等于 checkpoint.best_score。
    - summary.json.last_epoch: 必须是 Python int 且不小于 checkpoint.epoch。
    - summary.json.best_checkpoint: 必须是字符串路径，POSIX 文本等于 checkpoint_path。
    - run.json.plan_name: 必须等于 plan.plan_name。
    - run.json.plan_path: 必须是字符串，构造的 Path 必须在当前工作目录语义下指向现存文件。
    - 返回值.path: checkpoint_path 的 POSIX 文本。
    - 返回值.sha256: BEST.pt 当前全部字节的 SHA-256。
    - 返回值.epoch: 已核对的最佳轮次。
    - 返回值.global_step: 已核对的参数更新数。
    - 返回值.best_score: 已核对的有限 validation selection score。
    - 返回值.resolved_path: resolved.json 路径文本。
    - 返回值.resolved_sha256: resolved.json 当前全部字节的 SHA-256。
    - 返回值.summary_path: 阶段 summary.json 路径文本。
    - 返回值.summary_sha256: 阶段 summary.json 当前全部字节的 SHA-256。
    - 返回值.run_path: 训练 run.json 路径文本。
    - 返回值.run_sha256: 训练 run.json 当前全部字节的 SHA-256。
    - 返回值.training_plan_path: run.json.plan_path 指向的现存训练 YAML 路径文本。
    - 返回值.training_plan_sha256: 训练 YAML 当前全部字节的 SHA-256。
    - 返回值.model_config: 已规范化且与 stage.model 相等的完整模型配置字典。
    - 返回值.model_config.aggregation: 规范化聚合方式。
    - 返回值.model_config.use_manual_features: 规范化手工特征开关。
    - 返回值.model_config.use_sampling_features: 规范化采样特征开关。
    - 返回值.model_config.use_candidate_token_attention_bias: 规范化注意力偏置开关。
    - 返回值.model_config.initial_graph_gate: 规范化图门初值。
    - 返回值.model_config.initial_inventory_gate: 规范化清单门初值。
    - 返回值.model_config.model_dim: 规范化模型宽度。
    - 返回值.model_config.attention_heads: 规范化注意力头数。
    - 返回值.model_config.dropout: 规范化丢弃概率。
    - 返回值.model_config.voxel_feature_channels: 规范化体素特征宽度。
    - 返回值.model_config.raunet_planes: 九项整数元组，artifact JSON 中编码为数组。
    - 返回值.model_config.recycle_iterations: 规范化回收轮数。
    - 返回值.model_config.gradient_checkpointing: 规范化激活重计算开关。
    - 返回值.model_config.bottleneck_attention_layers: 规范化瓶颈注意力层数。
    - 返回值.model_config.graph_layers: 规范化图消息层数。
    - 返回值.training_config: 已与 stage.training 相等的完整训练配置字典。
    - 返回值.training_config.seed: 训练随机种子。
    - 返回值.training_config.physical_batch_size: 物理批量。
    - 返回值.training_config.gradient_accumulation: 梯度累积次数。
    - 返回值.training_config.global_batch_size: 全局批量。
    - 返回值.training_config.learning_rate: 初始学习率。
    - 返回值.training_config.weight_decay: 权重衰减。
    - 返回值.training_config.maximum_epochs: 最大轮数或 None。
    - 返回值.training_config.patience_bad_epochs: 早停坏轮数。
    - 返回值.training_config.precision: 混合精度模式。
    - 返回值.training_config.gradient_clip_norm: 梯度裁剪阈值。
    - 返回值.training_config.class_weights: 六类权重或 None。
    - 返回值.training_config.loss_weights: 四项损失系数字典。
    - 返回值.training_config.loss_weights.classification: 六分类损失系数。
    - 返回值.training_config.loss_weights.ligand_area: 配体区域损失系数。
    - 返回值.training_config.loss_weights.binding_area: 结合区域损失系数。
    - 返回值.training_config.loss_weights.identity: 身份损失系数。

    resolved.json.stage 会按正式模型 loader 补入历史缺失的两个门初值，并按任务补入历史缺失的
    identity_scope；resolved.json.data 会补入历史缺失的 fixed_center。这里只核验七个科学
    data 字段，不核验 worker、缓存或其它 data 键。summary 只核验这里列出的最佳路径、轮次、
    分数及 last_epoch；训练引擎在 tracker.finish(0) 之前就写入阶段 summary，本函数也不读取根
    summary，因此阶段 summary 不能证明 tracker.finish 成功、完整本地阶段收口或远端 W&B 状态。
    run.json.plan_path 只要求指向现存文件并留存其摘要，不要求其路径或字节等于当前 plan_path。
    checkpoint 的反序列化与 SHA-256 也是两次独立文件读取，依赖正式训练产物在 calibration
    期间保持不变。
    """

    if checkpoint.get("stage_name") != stage.name:
        raise ValueError("checkpoint.stage_name 与所选 YAML 阶段不一致。")
    if checkpoint.get("task_name") != stage.task_name:
        raise ValueError("checkpoint.task_name 与所选 YAML 阶段不一致。")

    # integers 收集三个通过严格 Python int 与非负门禁的 checkpoint 标量。
    integers: dict[str, int] = {}
    for name in ("epoch", "best_epoch", "global_step"):
        value = checkpoint.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"checkpoint.{name} 必须是非负整数。")
        integers[name] = value
    if integers["epoch"] != integers["best_epoch"]:
        raise ValueError("BEST.pt 必须满足 checkpoint.epoch == checkpoint.best_epoch。")
    # score_value 保留原映射实体用于排除 bool，再转换为 Python float best_score。
    score_value = checkpoint.get("best_score")
    if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
        raise ValueError("checkpoint.best_score 必须是有限数值。")
    best_score = float(score_value)
    if not math.isfinite(best_score):
        raise ValueError("checkpoint.best_score 必须是有限数值。")
    # checkpoint_mappings 把三个字符串键 Mapping 复制为普通 dict，后续不会修改 checkpoint。
    checkpoint_mappings: dict[str, dict[str, Any]] = {}
    for name in ("model_config", "training_config", "epoch_record"):
        value = checkpoint.get(name)
        if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
            raise ValueError(f"checkpoint.{name} 必须是字符串键映射。")
        checkpoint_mappings[name] = {str(key): item for key, item in value.items()}
    # epoch_record 是保存 BEST.pt 时同轮 EpochRecord 的部分或完整字典。
    epoch_record = checkpoint_mappings["epoch_record"]
    if (
        epoch_record.get("epoch") != integers["epoch"]
        or epoch_record.get("global_step") != integers["global_step"]
        or epoch_record.get("selection_score") != best_score
        or epoch_record.get("improved") is not True
    ):
        raise ValueError("BEST.pt 的 epoch_record 必须对应同一最佳轮次、更新数和选择分数。")

    # resolved_path/resolved 对应训练阶段开始前发布的解析配置及其顶层 JSON 对象。
    resolved_path = checkpoint_path.parent / "resolved.json"
    resolved = _read_json_mapping(resolved_path, "resolved.json")
    # expected_stage 是当前 YAML StageConfig 的 JSON 兼容深拷贝，模型元组会转成列表。
    expected_stage = asdict(stage)
    expected_stage["model"] = stage.model.to_dict()
    expected_stage_json = json.loads(
        json.dumps(expected_stage, ensure_ascii=False, allow_nan=False)
    )
    # resolved_stage_value 必须是映射；复制后允许为历史字段补入当前默认值。
    resolved_stage_value = resolved.get("stage")
    if not isinstance(resolved_stage_value, Mapping):
        raise ValueError("resolved.json.stage 必须是映射。")
    resolved_stage = {str(key): value for key, value in resolved_stage_value.items()}
    # resolved_model_value 是历史或当前模型配置映射，经正式配置类补默认值并重新核验。
    resolved_model_value = resolved_stage.get("model")
    if not isinstance(resolved_model_value, Mapping) or not all(
        isinstance(key, str) for key in resolved_model_value
    ):
        raise ValueError("resolved.json.stage.model 必须是字符串键映射。")
    # normalized_resolved_model 是完整模型配置字典，raunet_planes 稍后经 JSON 转为列表。
    normalized_resolved_model = MatcherModelConfig.from_dict(
        {str(key): value for key, value in resolved_model_value.items()}
    ).to_dict()
    resolved_stage["model"] = json.loads(
        json.dumps(normalized_resolved_model, ensure_ascii=False, allow_nan=False)
    )
    if "identity_scope" not in resolved_stage:
        task_name = resolved_stage.get("task_name")
        if task_name == "full_matcher":
            resolved_stage["identity_scope"] = "small_molecule"
        elif task_name == "candidate_pretrain":
            resolved_stage["identity_scope"] = None
    if resolved_stage != expected_stage_json:
        raise ValueError("resolved.json.stage 与所选 YAML 阶段不一致。")
    # resolved_data_value/resolved_data 是训练时解析的数据配置；只比较下方七个科学字段。
    resolved_data_value = resolved.get("data")
    if not isinstance(resolved_data_value, Mapping):
        raise ValueError("resolved.json.data 必须是映射。")
    resolved_data = {str(key): value for key, value in resolved_data_value.items()}
    if "local_crop_strategy" not in resolved_data:
        resolved_data["local_crop_strategy"] = "fixed_center"
    # expected_scientific_data 由当前 calibration 计划生成，三个 Path 用 str 保持训练记录口径。
    expected_scientific_data = {
        "data_root": str(plan.data.roots.data_root),
        "stage1_root": str(plan.data.roots.stage1_root),
        "output_root": str(plan.data.roots.output_root),
        "coverage_threshold": plan.data.coverage_threshold,
        "selection_seed": plan.data.selection_seed,
        "pp_top_k": plan.data.pp_top_k,
        "local_crop_strategy": plan.data.local_crop_strategy,
    }
    # data_mismatches 只保存实际不等字段及其 actual/expected，空映射表示七项相等。
    data_mismatches = {
        name: {"actual": resolved_data.get(name), "expected": expected}
        for name, expected in expected_scientific_data.items()
        if resolved_data.get(name) != expected
    }
    if data_mismatches:
        raise ValueError(f"resolved.json.data 科学字段与 calibration 计划不一致：{data_mismatches}")
    # model_config/training_config 是 checkpoint 内的科学配置，分别与当前 StageConfig 比较。
    model_config = MatcherModelConfig.from_dict(
        checkpoint_mappings["model_config"]
    ).to_dict()
    training_config = checkpoint_mappings["training_config"]
    if model_config != stage.model.to_dict():
        raise ValueError("checkpoint.model_config 与所选 YAML 阶段不一致。")
    if training_config != asdict(stage.training):
        raise ValueError("checkpoint.training_config 与所选 YAML 阶段不一致。")

    # summary_path/summary 是tracker.finish(0)前写出的阶段摘要；这里只核对列出的最佳轮次与路径关系。
    summary_path = checkpoint_path.parent / "summary.json"
    summary = _read_json_mapping(summary_path, "stage summary.json")
    # summary_best_path 必须是路径文本，随后以 POSIX 文本与 checkpoint_path 精确比较。
    summary_best_path = summary.get("best_checkpoint")
    if not isinstance(summary_best_path, str):
        raise ValueError("stage summary.best_checkpoint 必须是字符串路径。")
    # summary_score 保留 JSON 数值实体，按 Python 相等关系与 best_score 比较。
    summary_score = summary.get("best_score")
    if (
        summary.get("stage") != stage.name
        or summary.get("best_epoch") != integers["epoch"]
        or summary.get("last_epoch") is None
        or type(summary.get("last_epoch")) is not int
        or cast(int, summary["last_epoch"]) < integers["epoch"]
        or summary_score != best_score
        or Path(summary_best_path).as_posix() != checkpoint_path.as_posix()
    ):
        raise ValueError("阶段 summary.json 未证明该路径是已完成阶段的最佳 checkpoint。")

    # run_path/run_record 是训练根运行记录；只读取 plan_name 与训练时 plan_path。
    run_path = checkpoint_path.parent.parent / "run.json"
    run_record = _read_json_mapping(run_path, "run.json")
    if run_record.get("plan_name") != plan.plan_name:
        raise ValueError("run.json.plan_name 与 calibration 计划不一致。")
    # training_plan_value 必须是字符串，training_plan_path 必须在 calibration 时仍可读取。
    training_plan_value = run_record.get("plan_path")
    if not isinstance(training_plan_value, str):
        raise ValueError("run.json.plan_path 必须是字符串路径。")
    training_plan_path = Path(training_plan_value)
    if not training_plan_path.is_file():
        raise FileNotFoundError(f"训练时 YAML 已不存在，无法完成留证：{training_plan_path}")
    return {
        "path": checkpoint_path.as_posix(),
        "sha256": _sha256_file(checkpoint_path),
        "epoch": integers["epoch"],
        "global_step": integers["global_step"],
        "best_score": best_score,
        "resolved_path": resolved_path.as_posix(),
        "resolved_sha256": _sha256_file(resolved_path),
        "summary_path": summary_path.as_posix(),
        "summary_sha256": _sha256_file(summary_path),
        "run_path": run_path.as_posix(),
        "run_sha256": _sha256_file(run_path),
        "training_plan_path": training_plan_path.as_posix(),
        "training_plan_sha256": _sha256_file(training_plan_path),
        "model_config": model_config,
        "training_config": training_config,
    }


def _claim_calibration(
    claim_path: Path,
    data_fingerprint: str,
    plan_sha256: str,
    stage_name: str,
    checkpoint: CalibrationCheckpointEvidence,
    output_path: Path,
    claimed_at_utc: str,
) -> CalibrationClaim:
    """创建一次性 claim，或复用同一不可变选择的未完成 claim。

    - claim_path: 数据指纹对应的 claim JSON 路径；新 claim 会由 write_json 原子发布。
    - data_fingerprint: 当前 CalibrationDataEvidence.fingerprint。
    - plan_sha256: 当前 calibration 实验 YAML 全部字节的 SHA-256。
    - stage_name: validation 已选阶段名称。
    - checkpoint: 已经通过阶段自洽门禁的 CalibrationCheckpointEvidence。
    - checkpoint.path: 绑定到 claim 的 BEST.pt 路径文本。
    - checkpoint.sha256: 绑定到 claim 的 BEST.pt 内容摘要。
    - output_path: 首次选择的结果 JSON 路径；同选择重试必须保持相同 POSIX 文本。
    - claimed_at_utc: 新 claim 的带时区 ISO 8601 时间文本；复用既有 claim 时不会覆盖旧值。
    - 返回值.schema_version: CALIBRATION_SCHEMA_VERSION。
    - 返回值.status: 新建或可重试时为 claimed。
    - 返回值.data_fingerprint: 输入 data_fingerprint。
    - 返回值.plan_sha256: 输入 plan_sha256。
    - 返回值.stage_name: 输入 stage_name。
    - 返回值.checkpoint_sha256: checkpoint.sha256。
    - 返回值.checkpoint_path: checkpoint.path。
    - 返回值.output_path: output_path 的 POSIX 文本。
    - 返回值.output_sha256: claimed 状态为 None。
    - 返回值.claimed_at_utc: 新建时为输入时间，复用时保留首次 claim 时间。
    - 返回值.completed_at_utc: claimed 状态为 None。

    既有文件会先逐项比较 schema_version、数据指纹、计划摘要、阶段、checkpoint 摘要与路径、
    output 路径。任一变化都拒绝；完全相同且 status=claimed 时只允许 output_sha256 与
    completed_at_utc 都为 None。claim 文件存在且状态为 completed 时拒绝重评；删除 claim 后本函数
    看不到既有完成状态，仍可按新输出路径建立 claim。函数不验证时间字符串格式、摘要字符格式
    或未知附加键。
    """

    # immutable 是 claim 生命周期内禁止变化的七个 JSON 标量。
    immutable = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "data_fingerprint": data_fingerprint,
        "plan_sha256": plan_sha256,
        "stage_name": stage_name,
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_path": checkpoint["path"],
        "output_path": output_path.as_posix(),
    }
    if claim_path.exists():
        # existing 是磁盘 claim 的顶层对象；只核验本函数消费的字段和状态关系。
        existing = _read_json_mapping(claim_path, "calibration claim")
        # mismatches 逐字段保存既有值与本次期望值，禁止更换选择或结果路径。
        mismatches = {
            name: {"actual": existing.get(name), "expected": expected}
            for name, expected in immutable.items()
            if existing.get(name) != expected
        }
        if mismatches:
            raise RuntimeError(f"该 calibration 数据已绑定另一项选择：{mismatches}")
        if existing.get("status") == "completed":
            raise RuntimeError("该 calibration 数据已经完成一次评估，禁止再次运行。")
        if existing.get("status") != "claimed" or not isinstance(
            existing.get("claimed_at_utc"), str
        ):
            raise ValueError("既有 calibration claim 的状态或时间字段无效。")
        if existing.get("completed_at_utc") is not None:
            raise ValueError("claimed 状态的 completed_at_utc 必须为 null。")
        if existing.get("output_sha256") is not None:
            raise ValueError("claimed 状态的 output_sha256 必须为 null。")
        return cast(CalibrationClaim, existing)
    # claim 是新数据指纹的 claimed 状态；完成字段必须以 None 起始。
    claim: CalibrationClaim = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": "claimed",
        "data_fingerprint": data_fingerprint,
        "plan_sha256": plan_sha256,
        "stage_name": stage_name,
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_path": checkpoint["path"],
        "output_path": output_path.as_posix(),
        "output_sha256": None,
        "claimed_at_utc": claimed_at_utc,
        "completed_at_utc": None,
    }
    write_json(claim_path, claim)
    return claim


def _recover_claim_from_published_output(
    output_path: Path,
    claim_path: Path,
    claim: CalibrationClaim,
) -> bool:
    """用已发布结果收口“结果已写、claim 尚未完成”的崩溃窗口。

    - output_path: claim.output_path 对应的结果 JSON；不存在时立即返回 False。
    - claim_path: 待更新为 completed 的 claim JSON 路径，也是 artifact.claim_path 的期望值。
    - claim: 已由 _claim_calibration 核验为同一不可变选择的 CalibrationClaim。
    - claim.schema_version: 必须等于 artifact.schema_version。
    - claim.data_fingerprint: 必须等于 artifact.data.fingerprint。
    - claim.plan_sha256: 必须等于 artifact.plan_sha256。
    - claim.stage_name: 必须等于 artifact.stage_name。
    - claim.checkpoint_sha256: 必须等于 artifact.checkpoint.sha256。
    - claim.status: 前置 _claim_calibration 已核验为 claimed；完成时覆盖为 completed。
    - claim.checkpoint_path: 前置 claim 已绑定的 BEST.pt 路径；恢复不从 artifact 再核对此字段，完成时原样保留。
    - claim.output_path: 前置 claim 已绑定且由调用方传入的 output_path 文本；完成时原样保留。
    - claim.output_sha256: claimed 前置状态必须为 None；完成时覆盖为当前 output_path 全部字节的 SHA-256。
    - claim.claimed_at_utc: 首次 claim 时间；完成时原样保留。
    - claim.completed_at_utc: claimed 前置状态必须为 None；完成时覆盖为 artifact.finished_at_utc。
    - 返回值: output_path 不存在时为 False；结果通过绑定门禁且 claim 已写为 completed 时为 True。

    恢复只核验 schema_version、split、plan_sha256、stage_name、claim_path、checkpoint.sha256、
    data.fingerprint 和 finished_at_utc 类型，不重新验证完整 EvaluationResult、运行环境或其它
    artifact 字段。通过后重新计算 output SHA-256，并把 finished_at_utc 复制到 claim 的
    completed_at_utc；不会再次调用模型评估。
    """

    if not output_path.exists():
        return False
    # artifact 是已发布输出的顶层严格 JSON 对象；checkpoint/data 必须至少是 Mapping。
    artifact = _read_json_mapping(output_path, "calibration output")
    # checkpoint/data 是 artifact 中两个嵌套 Mapping，只读取各自的绑定摘要字段。
    checkpoint = artifact.get("checkpoint")
    data = artifact.get("data")
    if not isinstance(checkpoint, Mapping) or not isinstance(data, Mapping):
        raise ValueError("既有 calibration output 缺少 checkpoint 或 data 对象。")
    # expected 保存 artifact 顶层必须与 claim 相等的五个标量。
    expected = {
        "schema_version": claim["schema_version"],
        "split": "calibration",
        "plan_sha256": claim["plan_sha256"],
        "stage_name": claim["stage_name"],
        "claim_path": claim_path.as_posix(),
    }
    # mismatches 还会在下方加入两个嵌套摘要不一致项。
    mismatches = {
        name: {"actual": artifact.get(name), "expected": value}
        for name, value in expected.items()
        if artifact.get(name) != value
    }
    if checkpoint.get("sha256") != claim["checkpoint_sha256"]:
        mismatches["checkpoint.sha256"] = {
            "actual": checkpoint.get("sha256"),
            "expected": claim["checkpoint_sha256"],
        }
    if data.get("fingerprint") != claim["data_fingerprint"]:
        mismatches["data.fingerprint"] = {
            "actual": data.get("fingerprint"),
            "expected": claim["data_fingerprint"],
        }
    if mismatches:
        raise RuntimeError(f"既有 calibration output 与 claimed 选择不一致：{mismatches}")
    # finished_at_utc 只核验字符串类型，作为本地完成时间原样写入 completed claim。
    finished_at_utc = artifact.get("finished_at_utc")
    if not isinstance(finished_at_utc, str):
        raise ValueError("calibration output.finished_at_utc 必须是字符串。")
    # completed_claim 保留所有不可变字段，只补状态、输出摘要和完成时间。
    completed_claim: CalibrationClaim = {
        **claim,
        "status": "completed",
        "output_sha256": _sha256_file(output_path),
        "completed_at_utc": finished_at_utc,
    }
    write_json(claim_path, completed_claim)
    return True


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    """解析一次性 calibration 命令的四个必填参数。

    - argv: 不含解释器和模块名的命令参数字符串列表。
    - 返回值.plan: argparse 用 Path 转换后的 --plan 值。
    - 返回值.stage: --stage 的原始字符串值。
    - 返回值.checkpoint: argparse 用 Path 转换后的 --checkpoint 值。
    - 返回值.output: argparse 用 Path 转换后的 --output 值。

    四项均为 required，没有默认值。argparse 负责缺参、未知参数和帮助退出；本函数不检查
    路径存在、后缀、服务器根或阶段名称，这些门禁在 run_calibration 中执行。
    """

    # parser 的 description 取模块 Docstring；这里只影响帮助正文，不参与参数值解析。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="包含所选阶段的实验 YAML。")
    parser.add_argument("--stage", required=True, help="YAML 中已完成训练阶段的完整名称。")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="同一已完成阶段在 Matcher 产物根下的 BEST.pt。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Matcher 产物根下尚不存在的 calibration 结果 .json。",
    )
    return parser.parse_args(argv)


# ================================================================================================


def run_calibration(
    plan_path: Path,
    stage_name: str,
    checkpoint_path: Path,
    output_path: Path,
) -> Path:
    """用所选 BEST checkpoint 评估一次 calibration，并写出一次性结果 JSON。

    - plan_path: UTF-8 实验 YAML 路径；必须存在并通过 load_experiment_plan，随后 resolve(strict=True) 用于 artifact。
    - stage_name: validation 已选阶段的完整名称；计划中必须恰好出现一次。
    - checkpoint_path: BEST.pt 路径；经 POSIX 词法规范化后的文本必须位于 /storage/penghongen/tmp/Matcher，文件名恰为 BEST.pt 且存在，但不解析符号链接目标。
    - output_path: 尚待发布的 .json 路径；经 POSIX 词法规范化后的文本必须位于 Matcher 服务器产物根，且不得等于 claim 或锁路径；不解析符号链接目标。
    - 返回值: 规范化后的 output_path；新评估成功发布，或已发布结果通过崩溃恢复门禁时返回。

    输出 JSON 字段：
    - artifact.schema_version: CALIBRATION_SCHEMA_VERSION。
    - artifact.split: 固定 calibration。
    - artifact.plan_path: resolved plan_path 的 POSIX 文本。
    - artifact.plan_sha256: 当前 calibration YAML 字节摘要。
    - artifact.plan_name: 正式 loader 核验后的计划名称。
    - artifact.stage_name: 所选阶段名称。
    - artifact.task_name: candidate_pretrain 或 full_matcher。
    - artifact.language_family: 完整 Matcher 语言家族或 None。
    - artifact.identity_variant: 完整 Matcher 身份损失变体或 None。
    - artifact.checkpoint: BEST.pt、resolved.json、summary.json、run.json、训练 YAML 及两组配置的 CalibrationCheckpointEvidence。
    - artifact.checkpoint.path: BEST.pt 的 POSIX 词法规范化路径文本；不表示符号链接目标已经解析。
    - artifact.checkpoint.sha256: BEST.pt 字节摘要。
    - artifact.checkpoint.epoch: 最佳轮次。
    - artifact.checkpoint.global_step: 最佳轮次参数更新数。
    - artifact.checkpoint.best_score: validation 最佳 selection score。
    - artifact.checkpoint.resolved_path: 阶段 resolved.json 路径。
    - artifact.checkpoint.resolved_sha256: resolved.json 字节摘要。
    - artifact.checkpoint.summary_path: 阶段 summary.json 路径。
    - artifact.checkpoint.summary_sha256: summary.json 字节摘要。
    - artifact.checkpoint.run_path: 训练 run.json 路径。
    - artifact.checkpoint.run_sha256: run.json 字节摘要。
    - artifact.checkpoint.training_plan_path: 训练 YAML 路径。
    - artifact.checkpoint.training_plan_sha256: 训练 YAML 当前字节摘要。
    - artifact.checkpoint.model_config: 与所选阶段相等的完整模型配置字典。
    - artifact.checkpoint.model_config.aggregation: mean_max 或 candidate_cross_attention。
    - artifact.checkpoint.model_config.use_manual_features: 是否使用固化手工统计。
    - artifact.checkpoint.model_config.use_sampling_features: 是否使用裁剪几何统计。
    - artifact.checkpoint.model_config.use_candidate_token_attention_bias: 是否使用候选到 token 的注意力偏置。
    - artifact.checkpoint.model_config.initial_graph_gate: 图分数增量门初值。
    - artifact.checkpoint.model_config.initial_inventory_gate: 身份清单分数增量门初值。
    - artifact.checkpoint.model_config.model_dim: 候选与身份表示宽度。
    - artifact.checkpoint.model_config.attention_heads: 候选、图和清单注意力头数。
    - artifact.checkpoint.model_config.dropout: 模型共用丢弃概率。
    - artifact.checkpoint.model_config.voxel_feature_channels: RAUNet 最高分辨率输出通道数。
    - artifact.checkpoint.model_config.raunet_planes: 九项整数数组；内存配置中的元组经严格 JSON 转换而来。
    - artifact.checkpoint.model_config.recycle_iterations: RAUNet 回收轮数。
    - artifact.checkpoint.model_config.gradient_checkpointing: 是否启用训练期激活重计算。
    - artifact.checkpoint.model_config.bottleneck_attention_layers: RAUNet 瓶颈注意力层数。
    - artifact.checkpoint.model_config.graph_layers: LigandObject 图消息传递层数。
    - artifact.checkpoint.training_config: 与所选阶段相等的完整训练配置字典。
    - artifact.checkpoint.training_config.seed: 训练和评估随机种子。
    - artifact.checkpoint.training_config.physical_batch_size: 单次前向候选数。
    - artifact.checkpoint.training_config.gradient_accumulation: 每次参数更新包含的物理批次数。
    - artifact.checkpoint.training_config.global_batch_size: 每次参数更新对应的候选数。
    - artifact.checkpoint.training_config.learning_rate: AdamW 初始学习率。
    - artifact.checkpoint.training_config.weight_decay: AdamW 解耦权重衰减系数。
    - artifact.checkpoint.training_config.maximum_epochs: 最大轮数或 None。
    - artifact.checkpoint.training_config.patience_bad_epochs: 早停允许的连续未改善轮数。
    - artifact.checkpoint.training_config.precision: bf16、fp16 或 fp32。
    - artifact.checkpoint.training_config.gradient_clip_norm: 全模型 L2 梯度裁剪阈值。
    - artifact.checkpoint.training_config.class_weights: 按六分类顺序排列的权重或 None。
    - artifact.checkpoint.training_config.loss_weights: 四项损失系数字典。
    - artifact.checkpoint.training_config.loss_weights.classification: 六分类损失系数。
    - artifact.checkpoint.training_config.loss_weights.ligand_area: 配体区域损失系数。
    - artifact.checkpoint.training_config.loss_weights.binding_area: 结合区域损失系数。
    - artifact.checkpoint.training_config.loss_weights.identity: 闭集身份损失系数。
    - artifact.data: manifest、calibration 候选与身份记录的 CalibrationDataEvidence。
    - artifact.data.manifest_path: manifest 路径。
    - artifact.data.manifest_sha256: manifest 完整字节摘要。
    - artifact.data.training_manifest_sha256: 当前固定为 None 的预留字段。
    - artifact.data.manifest_semantic_sha256: calibration 相关稳定 manifest 语义摘要。
    - artifact.data.candidates_path: calibration 候选 JSONL 路径。
    - artifact.data.candidates_sha256: calibration 候选 JSONL 字节摘要。
    - artifact.data.identities_path: calibration 身份 JSONL 路径。
    - artifact.data.identities_sha256: calibration 身份 JSONL 字节摘要。
    - artifact.data.fingerprint: 三个稳定内容摘要的组合摘要。
    - artifact.claim_path: 按 data.fingerprint 命名的 claim JSON 路径。
    - artifact.runtime: Dataset 数量、loader 配置、随机源、数值后端、CUDA 设备与 Torch 版本。
    - artifact.runtime.candidate_count: calibration Dataset 候选数。
    - artifact.runtime.physical_batch_size: 物理批量。
    - artifact.runtime.num_workers: DataLoader worker 数。
    - artifact.runtime.prefetch_factor: 每 worker 预取批次数。
    - artifact.runtime.cache_gib_per_worker: 每个 CandidateDataset 实例的五类 LRU 总预算配置；预算为零或单项超预算时可不保留。
    - artifact.runtime.local_crop_strategy: fixed_center 或 v_centroid。
    - artifact.runtime.multiprocessing_context: 固定 spawn。
    - artifact.runtime.persistent_workers: 固定 False。
    - artifact.runtime.worker_generator_seed: calibration worker 独立生成器种子。
    - artifact.runtime.numeric_backend: 随机源与数值后端设置。
    - artifact.runtime.numeric_backend.python_numpy_torch_seed: 阶段随机种子。
    - artifact.runtime.numeric_backend.cudnn_benchmark: cuDNN benchmark 开关。
    - artifact.runtime.numeric_backend.cudnn_deterministic: cuDNN 确定性开关。
    - artifact.runtime.numeric_backend.deterministic_algorithms: Torch 确定性算法开关。
    - artifact.runtime.numeric_backend.float32_matmul_precision: float32 矩阵乘法内部精度策略。
    - artifact.runtime.device: 实际 CUDA 设备名。
    - artifact.runtime.torch_version: 实际 PyTorch 版本。
    - artifact.evaluation: evaluate_model 返回的损失、EvaluationReport、两个样本分母、耗时与吞吐。
    - artifact.evaluation.loss: 五项全局归一损失字典。
    - artifact.evaluation.loss.total: 四项加权损失总和。
    - artifact.evaluation.loss.classification: 六分类损失。
    - artifact.evaluation.loss.ligand_area: 配体区域损失。
    - artifact.evaluation.loss.binding_area: 结合区域损失。
    - artifact.evaluation.loss.identity: 有效身份候选损失。
    - artifact.evaluation.metrics: 完整 EvaluationReport。
    - artifact.evaluation.metrics.six_class: 六分类报告。
    - artifact.evaluation.metrics.six_class.accuracy: 全候选六分类准确率。
    - artifact.evaluation.metrics.six_class.macro_f1: 支持类 macro-F1 的兼容别名。
    - artifact.evaluation.metrics.six_class.macro_f1_supported_classes: 只平均真实支持类别的 macro-F1。
    - artifact.evaluation.metrics.six_class.macro_f1_all_classes: 固定平均六类的 macro-F1。
    - artifact.evaluation.metrics.six_class.per_class: 六个固定类别名到逐类指标对象的映射。
    - artifact.evaluation.metrics.six_class.per_class.background: 背景类别逐类指标对象。
    - artifact.evaluation.metrics.six_class.per_class.background.precision: 背景精确率。
    - artifact.evaluation.metrics.six_class.per_class.background.recall: 背景召回率。
    - artifact.evaluation.metrics.six_class.per_class.background.f1: 背景 F1。
    - artifact.evaluation.metrics.six_class.per_class.background.support: 背景真实候选数。
    - artifact.evaluation.metrics.six_class.per_class.background.predicted_count: 背景预测候选数。
    - artifact.evaluation.metrics.six_class.per_class.ion: 离子类别逐类指标对象。
    - artifact.evaluation.metrics.six_class.per_class.ion.precision: 离子精确率。
    - artifact.evaluation.metrics.six_class.per_class.ion.recall: 离子召回率。
    - artifact.evaluation.metrics.six_class.per_class.ion.f1: 离子 F1。
    - artifact.evaluation.metrics.six_class.per_class.ion.support: 离子真实候选数。
    - artifact.evaluation.metrics.six_class.per_class.ion.predicted_count: 离子预测候选数。
    - artifact.evaluation.metrics.six_class.per_class.nucleotide_like: 核苷酸样类别逐类指标对象。
    - artifact.evaluation.metrics.six_class.per_class.nucleotide_like.precision: 核苷酸样精确率。
    - artifact.evaluation.metrics.six_class.per_class.nucleotide_like.recall: 核苷酸样召回率。
    - artifact.evaluation.metrics.six_class.per_class.nucleotide_like.f1: 核苷酸样 F1。
    - artifact.evaluation.metrics.six_class.per_class.nucleotide_like.support: 核苷酸样真实候选数。
    - artifact.evaluation.metrics.six_class.per_class.nucleotide_like.predicted_count: 核苷酸样预测候选数。
    - artifact.evaluation.metrics.six_class.per_class.peptide_like: 肽样类别逐类指标对象。
    - artifact.evaluation.metrics.six_class.per_class.peptide_like.precision: 肽样精确率。
    - artifact.evaluation.metrics.six_class.per_class.peptide_like.recall: 肽样召回率。
    - artifact.evaluation.metrics.six_class.per_class.peptide_like.f1: 肽样 F1。
    - artifact.evaluation.metrics.six_class.per_class.peptide_like.support: 肽样真实候选数。
    - artifact.evaluation.metrics.six_class.per_class.peptide_like.predicted_count: 肽样预测候选数。
    - artifact.evaluation.metrics.six_class.per_class.small_molecule: 小分子类别逐类指标对象。
    - artifact.evaluation.metrics.six_class.per_class.small_molecule.precision: 小分子精确率。
    - artifact.evaluation.metrics.six_class.per_class.small_molecule.recall: 小分子召回率。
    - artifact.evaluation.metrics.six_class.per_class.small_molecule.f1: 小分子 F1。
    - artifact.evaluation.metrics.six_class.per_class.small_molecule.support: 小分子真实候选数。
    - artifact.evaluation.metrics.six_class.per_class.small_molecule.predicted_count: 小分子预测候选数。
    - artifact.evaluation.metrics.six_class.per_class.sugar: 糖类别逐类指标对象。
    - artifact.evaluation.metrics.six_class.per_class.sugar.precision: 糖精确率。
    - artifact.evaluation.metrics.six_class.per_class.sugar.recall: 糖召回率。
    - artifact.evaluation.metrics.six_class.per_class.sugar.f1: 糖 F1。
    - artifact.evaluation.metrics.six_class.per_class.sugar.support: 糖真实候选数。
    - artifact.evaluation.metrics.six_class.per_class.sugar.predicted_count: 糖预测候选数。
    - artifact.evaluation.metrics.six_class.confusion_matrix: 六行六列整数嵌套列表，第一轴为真实类别、第二轴为预测类别。
    - artifact.evaluation.metrics.intrinsic_identity: 配置范围固有身份报告。
    - artifact.evaluation.metrics.intrinsic_identity.top1: 固有身份 Top-1 正确率。
    - artifact.evaluation.metrics.intrinsic_identity.topk: 固有身份 Top-K 正确率。
    - artifact.evaluation.metrics.intrinsic_identity.top_k: 实际报告采用的最大名次数。
    - artifact.evaluation.metrics.intrinsic_identity.eligible_count: 固有身份可监督候选数。
    - artifact.evaluation.metrics.conditional_identity: 配置范围条件身份与 seen/unseen 报告。
    - artifact.evaluation.metrics.conditional_identity.top1: 条件身份 Top-1 正确率。
    - artifact.evaluation.metrics.conditional_identity.topk: 条件身份 Top-K 正确率。
    - artifact.evaluation.metrics.conditional_identity.top_k: 实际报告采用的最大名次数。
    - artifact.evaluation.metrics.conditional_identity.eligible_count: 条件身份可监督候选数。
    - artifact.evaluation.metrics.conditional_identity.seen_top1: 训练可见联合身份子集 Top-1。
    - artifact.evaluation.metrics.conditional_identity.seen_topk: 训练可见联合身份子集 Top-K。
    - artifact.evaluation.metrics.conditional_identity.seen_count: 训练可见联合身份子集候选数。
    - artifact.evaluation.metrics.conditional_identity.unseen_top1: 训练未见联合身份子集 Top-1。
    - artifact.evaluation.metrics.conditional_identity.unseen_topk: 训练未见联合身份子集 Top-K。
    - artifact.evaluation.metrics.conditional_identity.unseen_count: 训练未见联合身份子集候选数。
    - artifact.evaluation.metrics.end_to_end_identity: 配置范围端到端身份报告。
    - artifact.evaluation.metrics.end_to_end_identity.precision: 端到端身份精确率。
    - artifact.evaluation.metrics.end_to_end_identity.recall: 端到端身份召回率。
    - artifact.evaluation.metrics.end_to_end_identity.f1: 端到端身份 F1。
    - artifact.evaluation.metrics.end_to_end_identity.true_positive: 端到端身份真阳性数。
    - artifact.evaluation.metrics.end_to_end_identity.predicted_positive: 端到端身份预测阳性数。
    - artifact.evaluation.metrics.end_to_end_identity.actual_positive: 端到端身份真实阳性数。
    - artifact.evaluation.metrics.small_molecule_identity: 小分子兼容范围报告。
    - artifact.evaluation.metrics.small_molecule_identity.intrinsic: 小分子固有排名对象。
    - artifact.evaluation.metrics.small_molecule_identity.intrinsic.top1: 小分子固有 Top-1。
    - artifact.evaluation.metrics.small_molecule_identity.intrinsic.topk: 小分子固有 Top-K。
    - artifact.evaluation.metrics.small_molecule_identity.intrinsic.top_k: 小分子固有报告最大名次数。
    - artifact.evaluation.metrics.small_molecule_identity.intrinsic.eligible_count: 小分子固有分母。
    - artifact.evaluation.metrics.small_molecule_identity.conditional: 小分子条件排名对象。
    - artifact.evaluation.metrics.small_molecule_identity.conditional.top1: 小分子条件 Top-1。
    - artifact.evaluation.metrics.small_molecule_identity.conditional.topk: 小分子条件 Top-K。
    - artifact.evaluation.metrics.small_molecule_identity.conditional.top_k: 小分子条件报告最大名次数。
    - artifact.evaluation.metrics.small_molecule_identity.conditional.eligible_count: 小分子条件分母。
    - artifact.evaluation.metrics.small_molecule_identity.end_to_end: 小分子端到端对象。
    - artifact.evaluation.metrics.small_molecule_identity.end_to_end.precision: 小分子端到端精确率。
    - artifact.evaluation.metrics.small_molecule_identity.end_to_end.recall: 小分子端到端召回率。
    - artifact.evaluation.metrics.small_molecule_identity.end_to_end.f1: 小分子端到端 F1。
    - artifact.evaluation.metrics.small_molecule_identity.end_to_end.true_positive: 小分子端到端真阳性数。
    - artifact.evaluation.metrics.small_molecule_identity.end_to_end.predicted_positive: 小分子端到端预测阳性数。
    - artifact.evaluation.metrics.small_molecule_identity.end_to_end.actual_positive: 小分子端到端真实阳性数。
    - artifact.evaluation.metrics.all_non_background_identity: 五类总体范围报告。
    - artifact.evaluation.metrics.all_non_background_identity.intrinsic: 五类总体固有排名对象。
    - artifact.evaluation.metrics.all_non_background_identity.intrinsic.top1: 五类总体固有 Top-1。
    - artifact.evaluation.metrics.all_non_background_identity.intrinsic.topk: 五类总体固有 Top-K。
    - artifact.evaluation.metrics.all_non_background_identity.intrinsic.top_k: 五类总体固有报告最大名次数。
    - artifact.evaluation.metrics.all_non_background_identity.intrinsic.eligible_count: 五类总体固有分母。
    - artifact.evaluation.metrics.all_non_background_identity.conditional: 五类总体条件排名对象。
    - artifact.evaluation.metrics.all_non_background_identity.conditional.top1: 五类总体条件 Top-1。
    - artifact.evaluation.metrics.all_non_background_identity.conditional.topk: 五类总体条件 Top-K。
    - artifact.evaluation.metrics.all_non_background_identity.conditional.top_k: 五类总体条件报告最大名次数。
    - artifact.evaluation.metrics.all_non_background_identity.conditional.eligible_count: 五类总体条件分母。
    - artifact.evaluation.metrics.all_non_background_identity.end_to_end: 五类总体端到端对象。
    - artifact.evaluation.metrics.all_non_background_identity.end_to_end.precision: 五类总体端到端精确率。
    - artifact.evaluation.metrics.all_non_background_identity.end_to_end.recall: 五类总体端到端召回率。
    - artifact.evaluation.metrics.all_non_background_identity.end_to_end.f1: 五类总体端到端 F1。
    - artifact.evaluation.metrics.all_non_background_identity.end_to_end.true_positive: 五类总体端到端真阳性数。
    - artifact.evaluation.metrics.all_non_background_identity.end_to_end.predicted_positive: 五类总体端到端预测阳性数。
    - artifact.evaluation.metrics.all_non_background_identity.end_to_end.actual_positive: 五类总体端到端真实阳性数。
    - artifact.evaluation.metrics.identity_per_class: 五个前景类别的身份报告映射。
    - artifact.evaluation.metrics.identity_per_class.ion: 离子范围报告。
    - artifact.evaluation.metrics.identity_per_class.ion.intrinsic: 离子固有排名对象。
    - artifact.evaluation.metrics.identity_per_class.ion.intrinsic.top1: 离子固有 Top-1。
    - artifact.evaluation.metrics.identity_per_class.ion.intrinsic.topk: 离子固有 Top-K。
    - artifact.evaluation.metrics.identity_per_class.ion.intrinsic.top_k: 离子固有报告最大名次数。
    - artifact.evaluation.metrics.identity_per_class.ion.intrinsic.eligible_count: 离子固有分母。
    - artifact.evaluation.metrics.identity_per_class.ion.conditional: 离子条件排名对象。
    - artifact.evaluation.metrics.identity_per_class.ion.conditional.top1: 离子条件 Top-1。
    - artifact.evaluation.metrics.identity_per_class.ion.conditional.topk: 离子条件 Top-K。
    - artifact.evaluation.metrics.identity_per_class.ion.conditional.top_k: 离子条件报告最大名次数。
    - artifact.evaluation.metrics.identity_per_class.ion.conditional.eligible_count: 离子条件分母。
    - artifact.evaluation.metrics.identity_per_class.ion.end_to_end: 离子端到端对象。
    - artifact.evaluation.metrics.identity_per_class.ion.end_to_end.precision: 离子端到端精确率。
    - artifact.evaluation.metrics.identity_per_class.ion.end_to_end.recall: 离子端到端召回率。
    - artifact.evaluation.metrics.identity_per_class.ion.end_to_end.f1: 离子端到端 F1。
    - artifact.evaluation.metrics.identity_per_class.ion.end_to_end.true_positive: 离子端到端真阳性数。
    - artifact.evaluation.metrics.identity_per_class.ion.end_to_end.predicted_positive: 离子端到端预测阳性数。
    - artifact.evaluation.metrics.identity_per_class.ion.end_to_end.actual_positive: 离子端到端真实阳性数。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like: 核苷酸样范围报告。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like.intrinsic: 核苷酸样固有排名对象。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like.intrinsic.top1: 核苷酸样固有 Top-1。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like.intrinsic.topk: 核苷酸样固有 Top-K。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like.intrinsic.top_k: 核苷酸样固有报告最大名次数。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like.intrinsic.eligible_count: 核苷酸样固有分母。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like.conditional: 核苷酸样条件排名对象。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like.conditional.top1: 核苷酸样条件 Top-1。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like.conditional.topk: 核苷酸样条件 Top-K。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like.conditional.top_k: 核苷酸样条件报告最大名次数。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like.conditional.eligible_count: 核苷酸样条件分母。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like.end_to_end: 核苷酸样端到端对象。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like.end_to_end.precision: 核苷酸样端到端精确率。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like.end_to_end.recall: 核苷酸样端到端召回率。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like.end_to_end.f1: 核苷酸样端到端 F1。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like.end_to_end.true_positive: 核苷酸样端到端真阳性数。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like.end_to_end.predicted_positive: 核苷酸样端到端预测阳性数。
    - artifact.evaluation.metrics.identity_per_class.nucleotide_like.end_to_end.actual_positive: 核苷酸样端到端真实阳性数。
    - artifact.evaluation.metrics.identity_per_class.peptide_like: 肽样范围报告。
    - artifact.evaluation.metrics.identity_per_class.peptide_like.intrinsic: 肽样固有排名对象。
    - artifact.evaluation.metrics.identity_per_class.peptide_like.intrinsic.top1: 肽样固有 Top-1。
    - artifact.evaluation.metrics.identity_per_class.peptide_like.intrinsic.topk: 肽样固有 Top-K。
    - artifact.evaluation.metrics.identity_per_class.peptide_like.intrinsic.top_k: 肽样固有报告最大名次数。
    - artifact.evaluation.metrics.identity_per_class.peptide_like.intrinsic.eligible_count: 肽样固有分母。
    - artifact.evaluation.metrics.identity_per_class.peptide_like.conditional: 肽样条件排名对象。
    - artifact.evaluation.metrics.identity_per_class.peptide_like.conditional.top1: 肽样条件 Top-1。
    - artifact.evaluation.metrics.identity_per_class.peptide_like.conditional.topk: 肽样条件 Top-K。
    - artifact.evaluation.metrics.identity_per_class.peptide_like.conditional.top_k: 肽样条件报告最大名次数。
    - artifact.evaluation.metrics.identity_per_class.peptide_like.conditional.eligible_count: 肽样条件分母。
    - artifact.evaluation.metrics.identity_per_class.peptide_like.end_to_end: 肽样端到端对象。
    - artifact.evaluation.metrics.identity_per_class.peptide_like.end_to_end.precision: 肽样端到端精确率。
    - artifact.evaluation.metrics.identity_per_class.peptide_like.end_to_end.recall: 肽样端到端召回率。
    - artifact.evaluation.metrics.identity_per_class.peptide_like.end_to_end.f1: 肽样端到端 F1。
    - artifact.evaluation.metrics.identity_per_class.peptide_like.end_to_end.true_positive: 肽样端到端真阳性数。
    - artifact.evaluation.metrics.identity_per_class.peptide_like.end_to_end.predicted_positive: 肽样端到端预测阳性数。
    - artifact.evaluation.metrics.identity_per_class.peptide_like.end_to_end.actual_positive: 肽样端到端真实阳性数。
    - artifact.evaluation.metrics.identity_per_class.small_molecule: 小分子单类范围报告。
    - artifact.evaluation.metrics.identity_per_class.small_molecule.intrinsic: 小分子单类固有排名对象。
    - artifact.evaluation.metrics.identity_per_class.small_molecule.intrinsic.top1: 小分子单类固有 Top-1。
    - artifact.evaluation.metrics.identity_per_class.small_molecule.intrinsic.topk: 小分子单类固有 Top-K。
    - artifact.evaluation.metrics.identity_per_class.small_molecule.intrinsic.top_k: 小分子单类固有报告最大名次数。
    - artifact.evaluation.metrics.identity_per_class.small_molecule.intrinsic.eligible_count: 小分子单类固有分母。
    - artifact.evaluation.metrics.identity_per_class.small_molecule.conditional: 小分子单类条件排名对象。
    - artifact.evaluation.metrics.identity_per_class.small_molecule.conditional.top1: 小分子单类条件 Top-1。
    - artifact.evaluation.metrics.identity_per_class.small_molecule.conditional.topk: 小分子单类条件 Top-K。
    - artifact.evaluation.metrics.identity_per_class.small_molecule.conditional.top_k: 小分子单类条件报告最大名次数。
    - artifact.evaluation.metrics.identity_per_class.small_molecule.conditional.eligible_count: 小分子单类条件分母。
    - artifact.evaluation.metrics.identity_per_class.small_molecule.end_to_end: 小分子单类端到端对象。
    - artifact.evaluation.metrics.identity_per_class.small_molecule.end_to_end.precision: 小分子单类端到端精确率。
    - artifact.evaluation.metrics.identity_per_class.small_molecule.end_to_end.recall: 小分子单类端到端召回率。
    - artifact.evaluation.metrics.identity_per_class.small_molecule.end_to_end.f1: 小分子单类端到端 F1。
    - artifact.evaluation.metrics.identity_per_class.small_molecule.end_to_end.true_positive: 小分子单类端到端真阳性数。
    - artifact.evaluation.metrics.identity_per_class.small_molecule.end_to_end.predicted_positive: 小分子单类端到端预测阳性数。
    - artifact.evaluation.metrics.identity_per_class.small_molecule.end_to_end.actual_positive: 小分子单类端到端真实阳性数。
    - artifact.evaluation.metrics.identity_per_class.sugar: 糖范围报告。
    - artifact.evaluation.metrics.identity_per_class.sugar.intrinsic: 糖固有排名对象。
    - artifact.evaluation.metrics.identity_per_class.sugar.intrinsic.top1: 糖固有 Top-1。
    - artifact.evaluation.metrics.identity_per_class.sugar.intrinsic.topk: 糖固有 Top-K。
    - artifact.evaluation.metrics.identity_per_class.sugar.intrinsic.top_k: 糖固有报告最大名次数。
    - artifact.evaluation.metrics.identity_per_class.sugar.intrinsic.eligible_count: 糖固有分母。
    - artifact.evaluation.metrics.identity_per_class.sugar.conditional: 糖条件排名对象。
    - artifact.evaluation.metrics.identity_per_class.sugar.conditional.top1: 糖条件 Top-1。
    - artifact.evaluation.metrics.identity_per_class.sugar.conditional.topk: 糖条件 Top-K。
    - artifact.evaluation.metrics.identity_per_class.sugar.conditional.top_k: 糖条件报告最大名次数。
    - artifact.evaluation.metrics.identity_per_class.sugar.conditional.eligible_count: 糖条件分母。
    - artifact.evaluation.metrics.identity_per_class.sugar.end_to_end: 糖端到端对象。
    - artifact.evaluation.metrics.identity_per_class.sugar.end_to_end.precision: 糖端到端精确率。
    - artifact.evaluation.metrics.identity_per_class.sugar.end_to_end.recall: 糖端到端召回率。
    - artifact.evaluation.metrics.identity_per_class.sugar.end_to_end.f1: 糖端到端 F1。
    - artifact.evaluation.metrics.identity_per_class.sugar.end_to_end.true_positive: 糖端到端真阳性数。
    - artifact.evaluation.metrics.identity_per_class.sugar.end_to_end.predicted_positive: 糖端到端预测阳性数。
    - artifact.evaluation.metrics.identity_per_class.sugar.end_to_end.actual_positive: 糖端到端真实阳性数。
    - artifact.evaluation.metrics.area: 两个区域头的微平均 Dice 报告。
    - artifact.evaluation.metrics.area.ligand_micro_dice: 配体区域微平均 Dice。
    - artifact.evaluation.metrics.area.binding_micro_dice: 结合区域微平均 Dice。
    - artifact.evaluation.metrics.area.ligand_dice_denominator: 配体预测与真实体素数之和。
    - artifact.evaluation.metrics.area.binding_dice_denominator: 结合区域预测与真实体素数之和。
    - artifact.evaluation.sample_count: 实际遍历候选数。
    - artifact.evaluation.identity_eligible_count: compute_matcher_loss 的 identity_eligible_mask 计数；还受当前 stage.identity_scope 限定，artifact 顶层没有独立 identity_scope 字段。
    - artifact.evaluation.elapsed_seconds: 完整评估墙钟秒数。
    - artifact.evaluation.samples_per_second: 候选吞吐。
    - artifact.hostname: 当前节点名称。
    - artifact.slurm_job_id: SLURM_JOB_ID 环境文本或 None。
    - artifact.task_launch_dir: TASK_LAUNCH_DIR 环境文本或 None。
    - artifact.started_at_utc: 本次新评估尝试的开始时间。
    - artifact.finished_at_utc: 完成评估与样本数核验后的时间。

    状态与副作用：
    - claim_path: 位于 plan.data.roots.output_root/calibration，文件名由数据 fingerprint 唯一确定。
    - lock_path: claim_path 后追加 .lock 后缀的进程互斥文件；文件内容和是否留存没有科学状态语义。
    - claim.status: claim 文件保留时，claimed 允许相同选择重试，completed 阻止合作调用者重评；删除 claim 后可另建新 claim。
    - output_path: 先发布完整 artifact，再把 claim 改写为 completed；两次写入之间崩溃可由同选择调用恢复，但路径没有独立全局锁。
    - Dataset 缓存: CandidateDataset 首次读取时可按 data 与 datasets 模块契约，在 output_root 下加锁构建缺失的体素或 PP 版本化缓存。

    本函数只要求既有 manifest 声明 calibration，不创建、扩充或重排清单。采样器固定
    shuffle=False、drop_last=False、sample_count_multiple=None，因此完整遍历 Dataset。
    正式评估必须有 CUDA。函数不构造优化器、调度器或 W&B，不修改 checkpoint，不调用
    selection_score。claim 创建后的异常会保留 claimed 状态供相同选择重试；函数不删除
    claim、锁文件、输出或 Dataset 缓存。两次 exists 检查只拒绝当时观察到的既有或新出现输出；
    不争用同一 fingerprint 锁的外部写者或不同 fingerprint 调用仍可在检查与 write_json 原子
    替换之间竞争，因此这里不承诺并发绝对防覆盖。
    """

    # plan/stage 是当前 calibration YAML 经正式门禁后的不可变配置与唯一所选阶段。
    plan = load_experiment_plan(plan_path)
    stage = _select_stage(plan, stage_name)
    # 两个服务器产物路径只做 POSIX 词法规范化与根目录限制，不解析符号链接。
    checkpoint_path = validate_matcher_output_path(checkpoint_path, "checkpoint")
    output_path = validate_matcher_output_path(output_path, "output")
    if checkpoint_path.name != "BEST.pt":
        raise ValueError("calibration 只接受文件名为 BEST.pt 的所选最佳 checkpoint。")
    if output_path.suffix != ".json":
        raise ValueError("calibration 输出必须是 .json 文件。")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    # resolved_plan_path 绑定当前文件系统解析出的现存 YAML 绝对路径，plan_sha256 绑定其字节。
    resolved_plan_path = plan_path.resolve(strict=True)
    plan_sha256 = _sha256_file(resolved_plan_path)
    # checkpoint 是 load_checkpoint 通过基础 schema 门禁后的映射；evidence 再做阶段自洽检查。
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_evidence = _validate_checkpoint_for_stage(
        checkpoint_path,
        checkpoint,
        plan,
        stage,
    )
    # manifest_path 必须来自既有且关键字段相容、声明 calibration 的 manifest；本入口不物化划分。
    manifest_path = require_manifest(
        plan.data.roots,
        plan.data.coverage_threshold,
        plan.data.selection_seed,
        ("calibration",),
    )
    # data_evidence 同时绑定完整 manifest 字节、稳定语义子对象与两个 calibration JSONL 字节。
    data_evidence = _calibration_data_evidence(manifest_path)
    # claim_path 在数据 output_root 下按 fingerprint 固定，且再次受服务器产物根门禁。
    claim_path = validate_matcher_output_path(
        plan.data.roots.output_root
        / "calibration"
        / f"claim_{data_evidence['fingerprint']}.json",
        "calibration claim",
    )
    # lock_path 与 claim 一一对应，只序列化争用该文件锁的同一数据 fingerprint 状态转换。
    lock_path = claim_path.with_suffix(f"{claim_path.suffix}.lock")
    if output_path in (claim_path, lock_path):
        raise ValueError("calibration 输出不得与一次性 claim 或其互斥锁使用同一路径。")
    with exclusive_file_lock(lock_path):
        # claim_existed_before 区分合法崩溃恢复与没有任何绑定来源的孤立 output。
        claim_existed_before = claim_path.exists()
        if output_path.exists() and not claim_existed_before:
            raise FileExistsError(
                f"calibration 输出已存在但没有对应 claim，禁止把它当作崩溃恢复：{output_path}"
            )
        # started_at_utc 是本次调用进入锁后的 UTC 时间；复用 claim 不会改写 claimed_at_utc。
        started_at_utc = datetime.now(timezone.utc).isoformat()
        # claim 是新建或已核验可重试的 claimed 状态，全部不可变选择字段已绑定。
        claim = _claim_calibration(
            claim_path,
            data_evidence["fingerprint"],
            plan_sha256,
            stage.name,
            checkpoint_evidence,
            output_path,
            started_at_utc,
        )
        if _recover_claim_from_published_output(output_path, claim_path, claim):
            print(
                json.dumps(
                    {
                        "event": "calibration_claim_recovered",
                        "stage": stage.name,
                        "checkpoint_sha256": checkpoint_evidence["sha256"],
                        "output": output_path.as_posix(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            return output_path
        # numeric_backend 记录重置随机源后的进程级数值设置，与训练入口使用同一函数。
        numeric_backend = configure_random_seeds_and_numeric_backend(stage.training.seed)
        # device 必须解析为 CUDA；CPU 只用于构造错误消息前的分支，不执行正式 calibration。
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("正式 calibration 必须在 CUDA GPU 上运行。")

        # model 的结构由 task_name 唯一决定，并使用 checkpoint 中严格匹配的完整状态字典。
        model: torch.nn.Module
        if stage.task_name == "candidate_pretrain":
            model = CandidateEncoder(stage.model)
        else:
            model = ClosedSetMatcher(stage.model)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device)

        # calibration_dataset 按既有 manifest 的 calibration 两份 JSONL 建立候选与身份轴。
        calibration_dataset = CandidateDataset(
            roots=plan.data.roots,
            manifest_path=manifest_path,
            split_name="calibration",
            task_name=stage.task_name,
            language_family=stage.language_family,
            identity_scope=stage.identity_scope,
            pp_top_k=plan.data.pp_top_k,
            local_crop_strategy=plan.data.local_crop_strategy,
            cache_bytes_per_worker=int(plan.data.cache_gib_per_worker * 1024**3),
        )
        # calibration_sampler 不洗牌、不丢末批且不截断，索引序列覆盖 [0,len(dataset))。
        calibration_sampler = ShuffledBatchSampler(
            calibration_dataset,
            batch_size=stage.training.physical_batch_size,
            shuffle=False,
            drop_last=False,
            seed=stage.training.seed,
            sample_count_multiple=None,
        )
        # worker_generator_seed 与训练/validation loader 的派生种子错开，只播种 calibration worker。
        worker_generator_seed = 2 * stage.training.seed + 2
        # calibration_loader 使用 spawn，persistent_workers=False 由 build_candidate_loader 固定。
        calibration_loader = build_candidate_loader(
            calibration_dataset,
            calibration_sampler,
            plan.data.validation_num_workers,
            plan.data.prefetch_factor,
            worker_generator_seed,
        )
        # class_weights 为 None 或 CUDA float32 (6,)，顺序与 CLASS_NAMES 相同。
        class_weights = (
            None
            if stage.training.class_weights is None
            else torch.tensor(stage.training.class_weights, device=device, dtype=torch.float32)
        )
        # evaluation 是完整划分 EvaluationResult；evaluate_model 内部切换 eval/no_grad 并全局归一损失。
        evaluation = evaluate_model(model, calibration_loader, stage, class_weights, device)
        if evaluation["sample_count"] != len(calibration_dataset):
            raise RuntimeError(
                "calibration EvaluationResult.sample_count 与冻结 Dataset 候选数不一致。"
            )
        # finished_at_utc 只在完整评估和样本数一致性门禁通过后生成。
        finished_at_utc = datetime.now(timezone.utc).isoformat()
        # artifact 是待由严格 JSON 写入器原子发布的完整一次性评估记录。
        artifact: CalibrationArtifact = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "split": "calibration",
            "plan_path": resolved_plan_path.as_posix(),
            "plan_sha256": plan_sha256,
            "plan_name": plan.plan_name,
            "stage_name": stage.name,
            "task_name": stage.task_name,
            "language_family": stage.language_family,
            "identity_variant": stage.identity_variant,
            "checkpoint": checkpoint_evidence,
            "data": data_evidence,
            "claim_path": claim_path.as_posix(),
            "runtime": {
                "candidate_count": len(calibration_dataset),
                "physical_batch_size": stage.training.physical_batch_size,
                "num_workers": plan.data.validation_num_workers,
                "prefetch_factor": plan.data.prefetch_factor,
                "cache_gib_per_worker": plan.data.cache_gib_per_worker,
                "local_crop_strategy": plan.data.local_crop_strategy,
                "multiprocessing_context": DATALOADER_MULTIPROCESSING_CONTEXT,
                "persistent_workers": False,
                "worker_generator_seed": worker_generator_seed,
                "numeric_backend": numeric_backend,
                "device": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__,
            },
            "evaluation": evaluation,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "task_launch_dir": os.environ.get("TASK_LAUNCH_DIR"),
            "started_at_utc": started_at_utc,
            "finished_at_utc": finished_at_utc,
        }
        if output_path.exists():
            raise FileExistsError(f"calibration 输出在评估期间出现，禁止覆盖：{output_path}")
        write_json(output_path, artifact)
        # completed_claim 在 output 成功发布后绑定其当前摘要，并复用 artifact 的完成时间。
        completed_claim: CalibrationClaim = {
            **claim,
            "status": "completed",
            "output_sha256": _sha256_file(output_path),
            "completed_at_utc": finished_at_utc,
        }
        write_json(claim_path, completed_claim)
        print(
            json.dumps(
                {
                    "event": "calibration_completed",
                    "stage": stage.name,
                    "checkpoint_sha256": checkpoint_evidence["sha256"],
                    "sample_count": evaluation["sample_count"],
                    "output": output_path.as_posix(),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    return output_path


def main(argv: list[str] | None = None) -> int:
    """解析命令参数、执行一次 calibration，并把结果路径写到标准输出。

    - argv: 显式命令参数字符串列表，或 None；None 时读取 sys.argv[1:]。
    - arguments.plan: _parse_arguments 返回的实验 YAML Path。
    - arguments.stage: _parse_arguments 返回的阶段名称字符串。
    - arguments.checkpoint: _parse_arguments 返回的 BEST.pt Path。
    - arguments.output: _parse_arguments 返回的结果 JSON Path。
    - 返回值: 成功时为整数 0；解析错误和运行异常不在本函数内吞掉。

    run_calibration 自身会输出 calibration_completed 或 calibration_claim_recovered JSON 事件；
    本函数随后另打印返回 Path 的普通字符串表示。
    """

    # arguments 是含 plan、stage、checkpoint、output 四个属性的 argparse.Namespace。
    arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
    # output_path 是已成功发布或成功恢复的结果路径。
    output_path = run_calibration(
        arguments.plan,
        arguments.stage,
        arguments.checkpoint,
        arguments.output,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
