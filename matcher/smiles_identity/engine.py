"""运行单一 SMILES 身份 Matcher 的单卡训练与完整验证.

主要入口 run_smiles_training 创建唯一运行目录, 写出 run.json、history.jsonl 和正常结束时
的 summary.json, 并按 TrackingConfig 记录 W&B 标量与验证图像. 本模块不构建数据资产或清单.
"""

from __future__ import annotations

import json
import os
import platform
import random
import socket
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from matcher.utils.json_artifacts import append_jsonl, write_json

from .batching import ShuffledBatchSampler, build_candidate_loader
from .config import SmilesExperimentConfig
from .contracts import CandidateBatch
from .dataset import CandidateDataset
from .losses import compute_matcher_loss
from .model import ClosedSetMatcher


def _validation_images(
    outputs: dict[str, torch.Tensor],
    batch: CandidateBatch,
    tracker: Any,
) -> dict[str, Any]:
    """把首个验证候选的预测与目标沿 Z 轴最大投影为四张 W&B 图像.

    - outputs: 当前验证批次的模型输出, 读取两类区域 logits 的候选零位置.
    - batch: 与 outputs 对齐的 CandidateBatch, 读取两类区域目标的候选零位置.
    - tracker: W&B 运行或 None; None 时不导入 wandb 并返回空字典.
    - return: dict[str, Any], 四个 W&B Image 的映射; tracker=None 时返回空字典.
        - return.validation_image/ligand_probability_max_z: wandb.Image, 配体区域预测概率的 Y/X 最大投影.
        - return.validation_image/ligand_target_max_z: wandb.Image, 配体区域二值目标的 Y/X 最大投影.
        - return.validation_image/binding_probability_max_z: wandb.Image, 受体结合区域预测概率的 Y/X 最大投影.
        - return.validation_image/binding_target_max_z: wandb.Image, 受体结合区域二值目标的 Y/X 最大投影.
    """

    if tracker is None:
        return {}
    import wandb

    # ndarray[float32, (Y, X)], 首个候选配体区域 sigmoid 概率沿 Z 轴的最大投影.
    ligand_probability = torch.sigmoid(outputs["ligand_area_logits"][0]).amax(dim=0).float().cpu().numpy()
    # ndarray[float32, (Y, X)], 首个候选受体结合区域 sigmoid 概率沿 Z 轴的最大投影.
    binding_probability = torch.sigmoid(outputs["binding_area_logits"][0]).amax(dim=0).float().cpu().numpy()
    # ndarray[float32, (Y, X)], 首个候选配体区域二值目标沿 Z 轴的最大投影.
    ligand_target = batch["ligand_area_target"][0].amax(dim=0).float().cpu().numpy()
    # ndarray[float32, (Y, X)], 首个候选受体结合区域二值目标沿 Z 轴的最大投影.
    binding_target = batch["binding_area_target"][0].amax(dim=0).float().cpu().numpy()
    return {
        "validation_image/ligand_probability_max_z": wandb.Image(ligand_probability),
        "validation_image/ligand_target_max_z": wandb.Image(ligand_target),
        "validation_image/binding_probability_max_z": wandb.Image(binding_probability),
        "validation_image/binding_target_max_z": wandb.Image(binding_target),
    }


def _move_batch(batch: CandidateBatch, device: torch.device) -> CandidateBatch:
    """把批次中的全部张量异步移到 GPU, 字符串与 metadata 保留在 CPU.

    - batch: CandidateBatch, DataLoader 在固定 CPU 内存中生成的候选批次.
    - device: torch.device, 本次单卡训练使用的 CUDA 设备.
    - return: CandidateBatch, 张量叶节点已迁移到 device, 非张量叶节点保持原对象.
    """

    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }  # type: ignore[return-value]


def _start_tracking(config: SmilesExperimentConfig, metadata: dict[str, Any]) -> Any:
    """按配置启动 W&B; disabled 返回 None.

    - config: 运行名称、项目、分组、实体、模式和完整实验配置的来源.
    - metadata: 已落盘的 run.json 来源记录; 同一对象作为 provenance 上传.
    - return: wandb Run; tracking.mode=disabled 时返回 None.

    Path 等非 JSON 原生配置叶节点先转为字符串, 不依赖 W&B 的隐式序列化行为.
    """

    if config.tracking.mode == "disabled":
        return None
    import wandb

    return wandb.init(
        entity=config.tracking.entity,
        project=config.tracking.project,
        group=config.tracking.group,
        name=f"{config.name}-{metadata['run_stamp']}",
        mode=config.tracking.mode,
        config={
            "experiment": json.loads(json.dumps(asdict(config), default=str)),
            "provenance": metadata,
        },
    )


def _track(tracker: Any, values: dict[str, Any], global_step: int) -> None:
    """向已启用的 W&B 运行写一次同 global_step 指标.

    - tracker: W&B Run 或 None; None 时不产生外部记录.
    - values: dict[str, Any], 本次参数更新或完整验证对应的标量与图像.
    - global_step: int, values 绑定的已完成 AdamW 参数更新数.
    - return: None.
    """

    if tracker is not None:
        tracker.log(values, step=global_step)


def _summarize_training_window(records: list[dict[str, float]]) -> dict[str, float]:
    """汇总一次参数更新实际使用的全部物理批次损失.

    - records: list[dict[str, float]], 长度等于梯度累积步数; 每项对应一次物理批次前向.
        - records[i].total: float, 第 i 个物理批次的加权总损失.
        - records[i].classification: float, 第 i 个物理批次的六分类损失.
        - records[i].ligand_area: float, 第 i 个物理批次的配体区域损失.
        - records[i].binding_area: float, 第 i 个物理批次的受体结合区域损失.
        - records[i].identity: float, 第 i 个物理批次的精确 SMILES 身份损失.
        - records[i].identity_eligible_count: float, 第 i 个物理批次实际进入身份损失的候选数.
    - return: dict[str, float], 与梯度反向缩放一致的参数更新窗口摘要.
        - return.total: float, 窗口内 total 的物理批次算术均值.
        - return.classification: float, 窗口内 classification 的物理批次算术均值.
        - return.ligand_area: float, 窗口内 ligand_area 的物理批次算术均值.
        - return.binding_area: float, 窗口内 binding_area 的物理批次算术均值.
        - return.identity: float, 窗口内 identity 的物理批次算术均值, 不按各批身份监督数重新加权.
        - return.identity_eligible_count: float, 窗口内各物理批次身份监督候选数之和.
    """

    if not records:
        raise ValueError("训练记录窗口不能为空。")
    names = ("total", "classification", "ligand_area", "binding_area", "identity")
    result = {
        name: sum(record[name] for record in records) / len(records)
        for name in names
    }
    result["identity_eligible_count"] = sum(record["identity_eligible_count"] for record in records)
    return result


def _set_seed(seed: int) -> None:
    """固定 Python、NumPy、CPU 和当前主机全部 CUDA 随机源.

    - seed: int, Python、NumPy、PyTorch CPU 和全部可见 CUDA 设备共用的基准种子.
    - return: None.

    同时把 float32 矩阵乘精度设为 high; 该设置属于本训练入口的数值后端契约.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


# ================================================================================================


@torch.no_grad()
def _validate(
    model: ClosedSetMatcher,
    loader: Any,
    config: SmilesExperimentConfig,
    device: torch.device,
    global_step: int,
    tracker: Any,
) -> dict[str, float | int]:
    """遍历完整 validation 清单并记录损失、分类准确率和条件身份准确率.

    - model: 当前 ClosedSetMatcher; 进入函数后切换到评估模式, 返回前恢复训练模式.
    - loader: 完整 validation DataLoader; 不得只取前缀或随机子集.
    - config: 损失权重和身份损失模式的来源.
    - device: 本次单卡 CUDA 设备.
    - global_step: 当前已完成的参数更新数; 初始验证为零.
    - tracker: W&B 运行或 None; 只影响记录, 不改变指标计算.
    - return: dict[str, float | int], 本次完整验证的计数、损失和准确率.
        - return.global_step: int, 启动本次验证前已完成的 AdamW 参数更新数.
        - return.candidate_count: int, validation DataLoader 实际遍历的候选总数.
        - return.identity_eligible_count: int, identity_target 非负且参与身份准确率的前景候选数.
        - return.classification_loss: float, 以各物理批次 classification_denominator 聚合的六分类损失.
        - return.ligand_area_loss: float, 以各物理批次候选数聚合的配体区域损失.
        - return.binding_area_loss: float, 以各物理批次候选数聚合的受体结合区域损失.
        - return.identity_loss: float, 以各物理批次身份监督候选数聚合的精确 SMILES 身份损失; 分母为零时为 0.
        - return.total_loss: float, 四个完整验证分量分别聚合后按 LossWeights 重新加权求和的总损失.
        - return.class_accuracy: float, 全部候选的六分类 Top-1 准确率.
        - return.identity_top1: float, 有身份监督候选的精确 SMILES 闭集 Top-1 准确率; 分母为零时为 NaN.

    六分类分母由 LossOutput.classification_denominator 给出; 两个区域损失各以 candidate_count 为分母; 身份损失与身份 Top-1 各以 identity_eligible_count 为分母.
    """

    model.eval()
    # float, 六分类损失按 classification_denominator 还原后的分子累计和.
    classification_sum = 0.0
    # float, 配体区域损失按物理批次候选数还原后的分子累计和.
    ligand_area_sum = 0.0
    # float, 受体结合区域损失按物理批次候选数还原后的分子累计和.
    binding_area_sum = 0.0
    # float, 精确 SMILES 身份损失按身份监督候选数还原后的分子累计和.
    identity_sum = 0.0
    # float, 各物理批次 LossOutput.classification_denominator 的累计和.
    classification_denominator = 0.0
    # int, 已遍历的验证候选总数.
    candidate_count = 0
    # int, 六分类 Top-1 预测正确的候选累计数.
    class_correct = 0
    # int, 精确 SMILES 闭集 Top-1 预测正确的有身份监督候选累计数.
    identity_correct = 0
    # int, identity_target 非负的前景候选累计数.
    identity_count = 0
    # dict[str, Any]|None, 首个验证批次候选零的四张 W&B 区域投影图; 未启用 W&B 时为空字典.
    image_payload: dict[str, Any] | None = None
    for cpu_batch in loader:
        batch = _move_batch(cpu_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs = model(batch)
            losses = compute_matcher_loss(
                outputs,
                batch,
                config.loss_weights,
                config.identity_variant,
            )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError("validation 总损失出现 NaN 或 Inf。")
        count = int(batch["class_target"].numel())
        # float, 当前物理批次六分类损失的真实聚合分母; 未加类别权重时等于 count.
        batch_classification_denominator = float(losses["classification_denominator"])
        # int, 当前物理批次实际进入精确 SMILES 身份损失的候选数.
        batch_identity_count = int(losses["identity_eligible_count"])
        candidate_count += count
        classification_denominator += batch_classification_denominator
        classification_sum += float(losses["classification"]) * batch_classification_denominator
        ligand_area_sum += float(losses["ligand_area"]) * count
        binding_area_sum += float(losses["binding_area"]) * count
        identity_sum += float(losses["identity"]) * batch_identity_count
        class_correct += int((outputs["class_logits"].argmax(dim=1) == batch["class_target"]).sum())
        # BoolTensor[B], 当前批次中参加精确 SMILES 身份准确率计算的候选掩码.
        eligible = batch["identity_target"] >= 0
        identity_count += batch_identity_count
        if torch.any(eligible):
            identity_correct += int(
                (
                    outputs["identity_logits"][eligible].argmax(dim=1)
                    == batch["identity_target"][eligible]
                ).sum()
            )
        if image_payload is None:
            image_payload = _validation_images(outputs, batch, tracker)
    if candidate_count == 0:
        raise ValueError("validation 清单不能为空。")
    # float, 以全部物理批次真实六分类分母聚合的完整 validation 六分类损失.
    classification_loss = classification_sum / classification_denominator
    # float, 以全部验证候选数聚合的完整 validation 配体区域损失.
    ligand_area_loss = ligand_area_sum / candidate_count
    # float, 以全部验证候选数聚合的完整 validation 受体结合区域损失.
    binding_area_loss = binding_area_sum / candidate_count
    # float, 以全部身份监督候选数聚合的完整 validation 身份损失; 没有身份监督时为 0.
    identity_loss = identity_sum / identity_count if identity_count else 0.0
    # float, 四个完整 validation 分量按正式配置权重重新组合的总损失.
    total_loss = (
        config.loss_weights.classification * classification_loss
        + config.loss_weights.ligand_area * ligand_area_loss
        + config.loss_weights.binding_area * binding_area_loss
        + config.loss_weights.identity * identity_loss
    )
    # dict[str, float | int], 本次完整验证写入 history.jsonl、W&B 和最终 summary.json 的指标.
    result: dict[str, float | int] = {
        "global_step": global_step,
        "candidate_count": candidate_count,
        "identity_eligible_count": identity_count,
        "classification_loss": classification_loss,
        "ligand_area_loss": ligand_area_loss,
        "binding_area_loss": binding_area_loss,
        "identity_loss": identity_loss,
        "total_loss": total_loss,
        "class_accuracy": class_correct / candidate_count,
        "identity_top1": identity_correct / identity_count if identity_count else float("nan"),
    }
    _track(
        tracker,
        {
            "validation/total_loss": result["total_loss"],
            "validation/classification_loss": result["classification_loss"],
            "validation/ligand_area_loss": result["ligand_area_loss"],
            "validation/binding_area_loss": result["binding_area_loss"],
            "validation/identity_loss": result["identity_loss"],
            "validation/class_accuracy": result["class_accuracy"],
            "validation/identity_top1": result["identity_top1"],
            **(image_payload or {}),
        },
        global_step,
    )
    model.train()
    return result


def run_smiles_training(config: SmilesExperimentConfig, config_path: Path) -> Path:
    """创建唯一运行目录, 执行初始验证和限时训练, 并记录 W&B 曲线.

    - config: 已完成字段门禁的训练配置.
    - config_path: 正式 YAML 路径; 解析后的绝对路径写入 run.json.
    - return: Path, 正常达到 epoch 或墙钟停止条件后的唯一运行目录.

    文件产物:
        - <run_directory>/run.json: JSON object, 先在数据集和模型构造前发布来源记录, 再在 W&B 启动后覆盖补入 tracking.
            - schema: str, 固定为 smiles_training_v1.
            - name: str, config.name.
            - run_stamp: str, TASK_RUN_STAMP 或当前 UTC 时间戳, 同时定义本次运行目录名.
            - hostname: str, 当前训练主机名.
            - slurm_job_id: str|None, SLURM_JOB_ID 环境变量.
            - task_launch_dir: str|None, TASK_LAUNCH_DIR 环境变量.
            - config_path: str, 正式 YAML 的解析后绝对路径.
            - manifest_path: str, manifest.json 的解析后绝对路径.
            - started_at_utc: str, ISO 8601 UTC 启动时间.
            - save_checkpoints: bool, 本次运行是否允许写 LAST.pt.
            - tracking: dict[str, Any], W&B 启动后补入的记录模式和运行定位字段; 启动前失败时该字段不存在.
                - tracking.mode: str, online、offline 或 disabled, 与正式 YAML 一致.
                - tracking.run_id: str|None, W&B 运行标识; disabled 时为 None.
                - tracking.url: str|None, W&B 运行页面地址; disabled 时为 None.
            - worker_provenance: dict[str, Any], 项目根、Python、PyTorch、CUDA、GPU 和 DataLoader worker 来源.
                - worker_provenance.task_project_root: str|None, TASK_PROJECT_ROOT 环境变量.
                - worker_provenance.python: str, Python 版本.
                - worker_provenance.torch: str, PyTorch 版本.
                - worker_provenance.cuda_runtime: str|None, PyTorch 绑定的 CUDA runtime 版本.
                - worker_provenance.cuda_available: bool, 构建训练对象前 CUDA 是否可用.
                - worker_provenance.gpu_name: str|None, 当前 CUDA 设备名称.
                - worker_provenance.data_loader_workers: int, 每个 DataLoader 的工作进程数.
                - worker_provenance.prefetch_factor: int, 每个工作进程预取的物理批次数.
        - <run_directory>/history.jsonl: JSONL, 每项是一次完整验证事件; initial_validation 含 event 与 _validate 返回字段, 后续 validation 另含 epoch.
        - <run_directory>/LAST.pt: PyTorch checkpoint, 仅 save_checkpoints=True 且完成周期验证时覆盖写入; 含 model、optimizer 和 global_step.
        - <run_directory>/summary.json: JSON object, 仅正常停止时写出; 复制最终 run.json 字段并追加 finished_at_utc、global_step、stopped_by_time、last_validation 和 tracking_url.

    驻留任务可在外部 kill_lock 到达时由 allocation 终止; 这种情况下不会伪造正常
    summary.json. save_checkpoints=false 时整个运行不写模型参数文件.
    """

    # str, TASK_RUN_STAMP 或当前 UTC 时间戳; 同一 output_root/name 下必须唯一.
    run_stamp = os.environ.get("TASK_RUN_STAMP") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Path, 本次运行的唯一落盘目录.
    run_directory = config.output_root / config.name / run_stamp
    run_directory.mkdir(parents=True, exist_ok=False)
    # dict[str, Any], run.json 与 W&B provenance 共用的来源绑定字段.
    metadata = {
        "schema": "smiles_training_v1",
        "name": config.name,
        "run_stamp": run_stamp,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "task_launch_dir": os.environ.get("TASK_LAUNCH_DIR"),
        "config_path": str(Path(config_path).resolve()),
        "manifest_path": str(config.data.manifest_path.resolve()),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "save_checkpoints": config.optimization.save_checkpoints,
        "worker_provenance": {
            "task_project_root": os.environ.get("TASK_PROJECT_ROOT"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "data_loader_workers": config.data.num_workers,
            "prefetch_factor": config.data.prefetch_factor,
        },
    }
    write_json(run_directory / "run.json", metadata)
    _set_seed(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("正式训练要求可用 CUDA GPU。")
    device = torch.device("cuda")
    # CandidateDataset, 按 train_split 清单顺序物化训练候选.
    train_dataset = CandidateDataset(
        config.data.manifest_path,
        config.data.train_split,
        config.data.pp_top_k,
        config.data.cache_gib_per_worker,
    )
    # CandidateDataset, 按 validation_split 清单顺序物化完整验证候选.
    validation_dataset = CandidateDataset(
        config.data.manifest_path,
        config.data.validation_split,
        config.data.pp_top_k,
        config.data.cache_gib_per_worker,
    )
    # ShuffledBatchSampler, 每个 epoch 可复现洗牌并截短到完整梯度累积边界.
    train_sampler = ShuffledBatchSampler(
        train_dataset,
        config.data.batch_size,
        True,
        True,
        config.seed,
        config.data.batch_size * config.data.accumulation_steps,
    )
    # ShuffledBatchSampler, 保持清单顺序且保留不足物理批量的验证末批.
    validation_sampler = ShuffledBatchSampler(
        validation_dataset,
        config.data.batch_size,
        False,
        False,
        config.seed,
        None,
    )
    train_loader = build_candidate_loader(
        train_dataset,
        train_sampler,
        config.data.num_workers,
        config.data.prefetch_factor,
        config.seed + 1,
    )
    validation_loader = build_candidate_loader(
        validation_dataset,
        validation_sampler,
        config.data.num_workers,
        config.data.prefetch_factor,
        config.seed + 2,
    )
    model = ClosedSetMatcher(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    tracker = _start_tracking(config, metadata)
    # dict[str, str|None], 最终 run.json 绑定的 W&B 模式、运行标识和页面地址.
    metadata["tracking"] = {
        "mode": config.tracking.mode,
        "run_id": getattr(tracker, "id", None),
        "url": getattr(tracker, "url", None),
    }
    write_json(run_directory / "run.json", metadata)
    # float, 训练与验证共同墙钟计时的单调时钟起点.
    started = time.monotonic()
    # int, 已完成 AdamW 参数更新数; 梯度累积中的前向不增加该值.
    global_step = 0
    try:
        validation = _validate(model, validation_loader, config, device, global_step, tracker)
        append_jsonl(run_directory / "history.jsonl", {"event": "initial_validation", **validation})
        optimizer.zero_grad(set_to_none=True)
        stopped_by_time = False
        # list[dict[str, float]], 当前 AdamW 更新尚未消费的物理批次损失和身份监督计数; 完成更新后清空.
        training_window: list[dict[str, float]] = []
        for epoch in range(config.optimization.max_epochs):
            train_sampler.set_epoch(epoch)
            for micro_step, cpu_batch in enumerate(train_loader, start=1):
                batch = _move_batch(cpu_batch, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(batch)
                    losses = compute_matcher_loss(
                        outputs,
                        batch,
                        config.loss_weights,
                        config.identity_variant,
                    )
                    scaled_loss = losses["total"] / config.data.accumulation_steps
                if not torch.isfinite(scaled_loss):
                    raise FloatingPointError("训练总损失出现 NaN 或 Inf。")
                scaled_loss.backward()
                training_window.append(
                    {
                        name: float(losses[name].detach())
                        for name in ("total", "classification", "ligand_area", "binding_area", "identity")
                    }
                )
                training_window[-1]["identity_eligible_count"] = float(losses["identity_eligible_count"])
                if micro_step % config.data.accumulation_steps != 0:
                    continue
                # 浮点标量张量, 本次更新裁剪前全部模型参数梯度的全局 L2 范数.
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.optimization.gradient_clip_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % config.tracking.log_every_updates == 0:
                    # dict[str, float], 当前梯度累积窗口的五项批次算术均值和身份监督候选总数.
                    window_values = _summarize_training_window(training_window)
                    # dict[str, float], 当前参数更新写入 W&B 的损失、裁剪和墙钟标量.
                    values = {
                        "train/total_loss": window_values["total"],
                        "train/classification_loss": window_values["classification"],
                        "train/ligand_area_loss": window_values["ligand_area"],
                        "train/binding_area_loss": window_values["binding_area"],
                        "train/identity_loss": window_values["identity"],
                        "train/identity_eligible_count": window_values["identity_eligible_count"],
                        "train/gradient_norm_before_clip": float(gradient_norm),
                        "train/clip_triggered": float(gradient_norm > config.optimization.gradient_clip_norm),
                        "runtime/elapsed_hours": (time.monotonic() - started) / 3600,
                    }
                    _track(tracker, values, global_step)
                training_window.clear()
                if global_step % config.optimization.validation_every_updates == 0:
                    validation = _validate(model, validation_loader, config, device, global_step, tracker)
                    append_jsonl(
                        run_directory / "history.jsonl",
                        {"event": "validation", "epoch": epoch, **validation},
                    )
                    if config.optimization.save_checkpoints:
                        torch.save(
                            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "global_step": global_step},
                            run_directory / "LAST.pt",
                        )
                if (time.monotonic() - started) / 3600 >= config.optimization.max_hours:
                    stopped_by_time = True
                    break
            if stopped_by_time:
                break
        # dict[str, Any], 正常停止时写入 summary.json 的运行来源与最终状态.
        summary = {
            **metadata,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "global_step": global_step,
            "stopped_by_time": stopped_by_time,
            "last_validation": validation,
            "tracking_url": getattr(tracker, "url", None),
        }
        write_json(run_directory / "summary.json", summary)
        return run_directory
    finally:
        if tracker is not None:
            tracker.finish()
