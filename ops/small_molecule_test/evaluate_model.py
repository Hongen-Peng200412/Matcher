"""用训练 source release 和冻结 BEST 评估 test_smallmolecule 身份匹配.

主要入口是 evaluate_model.当前工作树只负责选择 Stage3 目标候选、组织 Dataset、保存
两种最终 argmax 和汇总 top-k; 模型配置、模型类与 checkpoint 解释始终来自训练 run 指向的
source release.本模块不读取或使用前景阈值, 也不生成图像.
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from matcher.datasets.batching import ShuffledBatchSampler, build_candidate_loader
from matcher.datasets.candidate_data import load_manifest_dataset
from matcher.evaluation.checkpoint import load_matcher_from_training_release

from .artifacts import load_config, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from .ranking import build_topk_curve


def _evaluate_condition(
    model: Any,
    model_config: Any,
    source_manifest_path: Path,
    split: str,
    small_molecule_by_identity: dict[tuple[str, str], bool],
    stage3_candidates: list[dict[str, Any]],
    runtime: dict[str, Any],
    device: Any,
) -> list[dict[str, Any]]:
    """对一种受体条件的全部 Stage3 目标候选执行一次完整身份前向.

    输入参数:
        - model: 从训练 source release 恢复且处于 eval 模式的 Matcher.
        - model_config: 训练 checkpoint 内恢复的模型配置; variant 决定 Dataset 实际读取字段.
        - source_manifest_path: Path, 当前受体条件的完整 179-PDB test_0 manifest.
        - split: str, 固定测试划分名称 held_out_test_0.
        - small_molecule_by_identity: dict[(str, str), bool].
          - 键: pdb_id 与精确 PreparedLigand.smiles.
          - 值: True 表示该身份原始 type_tag 为 small_molecule, 可参与第二档 argmax.
        - stage3_candidates: list[dict[str, Any]], 唯一目标属于 Stage3 的 Find_1 候选.
          - pdb_id: str, 小写 PDB 标识.
          - centered_box_index: int, centered 候选稳定编号.
          - source_blob_index: int, 完整来源 blob 编号.
          - matched_occurrence_id: int, 唯一 Stage3 目标 occurrence 编号.
          - target_smiles: str, 唯一目标精确顶层 SMILES.
          - blob_center_world_xyz_A: list[float], 长度 3 的世界 XYZ 中心, 单位 Å.
        - runtime: dict[str, Any].
          - batch_size: int, DataLoader 物理批量.
          - num_workers: int, 当前受体条件的工作进程数.
          - prefetch_factor: int, 每个 worker 的预取批次数.
          - worker_generator_seed: int, DataLoader worker 种子.
          - sampler_seed: int, 不洗牌采样器的显式种子.
          - torch_seed: int, 当前评测进程与 CUDA 种子.
          - pp_top_k: int, strongest 的 PP token 上限.
          - cache_gib_per_worker: float, 每个 worker 的 PDB 载荷缓存预算.
        - device: torch.device, 实际 A800 CUDA 设备.

    返回值:
        - predictions: list[dict[str, Any]], 每个 Stage3 目标候选一条.
          - pdb_id: str, 小写 PDB 标识.
          - centered_box_index: int, centered 候选稳定编号.
          - source_blob_index: int, 完整来源 blob 编号.
          - candidate_selected: bool, 候选属于排序前段还是追加段.
          - matched_occurrence_id: int, 唯一 Stage3 目标 occurrence 编号.
          - target_smiles: str, 唯一目标精确顶层 SMILES.
          - identity_smiles: list[str], 当前 PDB 的完整身份轴, 长度 I.
          - identity_logits: list[float], 与 identity_smiles 逐槽对齐的原始分数, 长度 I.
          - small_molecule_answer_mask: list[bool], 与身份轴逐槽对齐的最终答案掩码.
          - predicted_smiles_all_ligand: str, 完整身份轴上的 argmax SMILES.
          - predicted_smiles_small_molecule: str, 最终类别掩码后的 argmax SMILES.
          - blob_center_world_xyz_A: list[float], 长度 3 的世界 XYZ 中心, 单位 Å.

    形状符号:
        - B: 当前物理批量中的候选数.
        - I_max: 当前批次内最大的 PDB 身份数; 三类身份张量均为 (B, I_max).
        - identity_mask=False 表示批次补齐槽; small_molecule_mask=True 只可能出现在有效身份槽.
    """
    import torch

    dataset = load_manifest_dataset(
        source_manifest_path,
        split=split,
        selected_only=False,
        mode="test",
        variant=model_config.variant,
        pp_top_k=int(runtime["pp_top_k"]),
        cache_gib_per_worker=float(runtime["cache_gib_per_worker"]),
    )
    stage3_by_key = {
        (
            str(candidate["pdb_id"]),
            int(candidate["centered_box_index"]),
            int(candidate["source_blob_index"]),
        ): candidate
        for candidate in stage3_candidates
    }
    dataset.candidates = [
        candidate
        for candidate in dataset.candidates
        if (
            str(candidate["pdb_id"]),
            int(candidate["centered_box_index"]),
            int(candidate["source_blob_index"]),
        )
        in stage3_by_key
    ]
    if len(dataset.candidates) != len(stage3_by_key):
        raise RuntimeError("原 test_0 Dataset 没有逐项覆盖冻结 Stage3 目标候选.")
    sampler = ShuffledBatchSampler(
        dataset,
        int(runtime["batch_size"]),
        False,
        False,
        int(runtime["sampler_seed"]),
        None,
    )
    loader = build_candidate_loader(
        dataset,
        sampler,
        int(runtime["num_workers"]),
        int(runtime["prefetch_factor"]),
        int(runtime["worker_generator_seed"]),
    )
    predictions: list[dict[str, Any]] = []
    iterator = iter(loader)
    try:
        with torch.no_grad():
            for cpu_batch in iterator:
                batch = {
                    key: value.to(device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in cpu_batch.items()
                }
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(batch)
                # float32, (B, I_max), 每个候选对完整 PDB 身份轴的未归一化匹配分数.
                identity_logits = output["identity_logits"].float()
                # bool, (B, I_max), True 表示当前候选所属 PDB 的有效身份槽; 遮盖批次补齐槽.
                identity_mask = batch["identity_mask"].bool()
                # bool, (B, I_max), True 表示有效身份槽的原始类别为 small_molecule.
                small_molecule_mask = torch.zeros_like(identity_mask)
                for sample_index, smiles_axis in enumerate(cpu_batch["identity_smiles"]):
                    pdb_id = str(cpu_batch["metadata"][sample_index]["pdb_id"])
                    allowed = [
                        small_molecule_by_identity[(pdb_id, str(smiles))]
                        for smiles in smiles_axis
                    ]
                    small_molecule_mask[sample_index, : len(allowed)] = torch.tensor(
                        allowed, dtype=torch.bool, device=device
                    )
                if not bool(torch.all(torch.any(small_molecule_mask, dim=1))):
                    raise RuntimeError("批次中存在没有可选 small_molecule 身份的 PDB.")
                full_logits = identity_logits.masked_fill(~identity_mask, float("-inf"))
                small_logits = identity_logits.masked_fill(~small_molecule_mask, float("-inf"))
                full_index = full_logits.argmax(dim=1).cpu().tolist()
                small_index = small_logits.argmax(dim=1).cpu().tolist()
                raw_logits = identity_logits.cpu().tolist()
                for sample_index, metadata in enumerate(cpu_batch["metadata"]):
                    pdb_id = str(metadata["pdb_id"])
                    centered_box_index = int(metadata["centered_box_index"])
                    source_blob_index = int(metadata["source_blob_index"])
                    key = (pdb_id, centered_box_index, source_blob_index)
                    frozen = stage3_by_key[key]
                    if (
                        metadata["matched_occurrence_id"] != frozen["matched_occurrence_id"]
                        or metadata["target_smiles"] != frozen["target_smiles"]
                    ):
                        raise RuntimeError(f"{key}: test_0 Dataset 与小分子 manifest 目标不一致.")
                    smiles_axis = [str(smiles) for smiles in cpu_batch["identity_smiles"][sample_index]]
                    allowed = [
                        bool(small_molecule_by_identity[(pdb_id, smiles)])
                        for smiles in smiles_axis
                    ]
                    predictions.append(
                        {
                            "pdb_id": pdb_id,
                            "centered_box_index": centered_box_index,
                            "source_blob_index": source_blob_index,
                            "candidate_selected": bool(metadata["candidate_selected"]),
                            "matched_occurrence_id": int(metadata["matched_occurrence_id"]),
                            "target_smiles": str(metadata["target_smiles"]),
                            "blob_center_world_xyz_A": frozen["blob_center_world_xyz_A"],
                            "identity_smiles": smiles_axis,
                            "identity_logits": raw_logits[sample_index][: len(smiles_axis)],
                            "small_molecule_answer_mask": allowed,
                            "predicted_smiles_all_ligand": smiles_axis[int(full_index[sample_index])],
                            "predicted_smiles_small_molecule": smiles_axis[
                                int(small_index[sample_index])
                            ],
                        }
                    )
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
        del iterator
        del loader
        del dataset
        gc.collect()
    return predictions


# ================================================================================================


def evaluate_model(config_path: Path, model_name: str) -> Path:
    """评估一个冻结 BEST 并发布两档身份曲线与 docking handoff.

    输入参数:
        - config_path: Path, test_smallmolecule.yaml 的现存路径.
        - model_name: str, density_blob_only 或 strongest; 决定 checkpoint 与版本化结果目录.

    返回值:
        - result_path: Path, 已发布的 evaluation_root/model/training_run/evaluation.json.

    输出文件:
        - evaluation.json: object.
          - schema: str, 模型评测产物契约版本.
          - created_at_utc: str, UTC ISO 8601 发布时间.
          - model_name: str, density_blob_only 或 strongest.
          - checkpoint.path: str, 冻结 BEST 绝对路径.
          - checkpoint.sha256: str, 冻结 BEST 内容摘要.
          - checkpoint.best_epoch: int, BEST 所属训练 epoch.
          - checkpoint.best_validation_check: int, BEST 所属 validation 次数.
          - checkpoint.best_score: float, 冻结 selection.
          - checkpoint.global_step: int, BEST 所属优化器更新数.
          - model_code_source.model_code_source: str, 固定为 training_release.
          - model_code_source.training_run_json_path: str, 训练 run.json 绝对路径.
          - model_code_source.training_project_root: str, 训练 source release 项目根.
          - model_code_source.training_release_manifest_path: str, 训练 release manifest 路径.
          - training_provenance.run_json_path: str, 训练 run.json 绝对路径.
          - training_provenance.run_json_sha256: str, 训练 run.json 内容摘要.
          - training_provenance.config_path: str, 训练 YAML 绝对路径.
          - training_provenance.config_path_sha256: str, 训练 YAML 内容摘要.
          - training_provenance.resolved_config_path: str, 训练 resolved config 路径.
          - training_provenance.resolved_config_path_sha256: str, resolved config 内容摘要.
          - training_provenance.manifest_path: str, 训练 manifest 原入口路径.
          - training_provenance.manifest_path_sha256: str, 训练 manifest 原入口摘要.
          - training_provenance.manifest_snapshot_path: str, run 内 manifest 快照路径.
          - training_provenance.manifest_snapshot_path_sha256: str, run 内 manifest 快照摘要.
          - training_provenance.source_project_root: str, 训练 source release 项目根.
          - training_provenance.source_release_manifest_path: str, 训练 release manifest 路径.
          - training_provenance.source_release_manifest_path_sha256: str, 训练 release manifest 摘要.
          - evaluation_code.project_root: str, 当前评测 release 项目根.
          - evaluation_code.release_manifest_path: str, 当前评测 release manifest 路径.
          - evaluation_code.release_manifest_sha256: str, 当前评测 release manifest 摘要.
          - evaluation_code.launch_dir: str, 当前正式 launch 目录.
          - evaluation_code.launch_json_path: str, 当前 launch.json 路径.
          - evaluation_code.launch_json_sha256: str, 当前 launch.json 内容摘要.
          - evaluation_code.run_cmd_path: str, 当前短正式动态命令路径.
          - evaluation_code.run_cmd_sha256: str, 当前动态命令内容摘要.
          - evaluation_code.config_path: str, 当前 test_smallmolecule YAML 路径.
          - evaluation_code.config_sha256: str, 当前 YAML 内容摘要.
          - manifest.path: str, test_smallmolecule manifest 绝对路径.
          - manifest.sha256: str, test_smallmolecule manifest 内容摘要.
          - stage1_only.path: str, Stage1-only 正式 JSON 绝对路径.
          - stage1_only.sha256: str, Stage1-only 正式 JSON 内容摘要.
          - source_test_manifests.real_receptor.path: str, 真实受体完整 test_0 manifest 路径.
          - source_test_manifests.real_receptor.sha256: str, 真实受体 test_0 manifest 摘要.
          - source_test_manifests.cryoatom2_receptor.path: str, CryoAtom2 完整 test_0 manifest 路径.
          - source_test_manifests.cryoatom2_receptor.sha256: str, CryoAtom2 test_0 manifest 摘要.
          - runtime.hostname: str, 实际计算节点名称.
          - runtime.slurm_job_id: str, 实际 Slurm Job ID.
          - runtime.task_project_root: str, 当前评测 release 项目根.
          - runtime.task_launch_dir: str, 当前正式 launch 目录.
          - runtime.device: str, CUDA 设备名称.
          - runtime.torch_version: str, PyTorch 版本.
          - runtime.cuda_runtime: str, CUDA runtime 版本.
          - runtime.batch_size: int, 身份前向物理批量.
          - runtime.num_workers: int, 单一活动 DataLoader worker 数.
          - runtime.prefetch_factor: int, 每个 worker 预取批次数.
          - runtime.python: str, Python 版本.
          - runtime.foreground_head_usage: str, 固定为 ignored.
          - predictions: object, 受体条件到逐候选 predictions 的映射; 单条字段由 _evaluate_condition 定义.
          - curves.gaussian: object, 受体条件和答案范围到 Gaussian top-k 曲线的映射.
          - curves.source_probability_mean: object, 受体条件和答案范围到概率均值 top-k 曲线的映射.
          - traces.gaussian: object, 与 Gaussian 曲线逐键对齐的逐 PDB 轨迹.
          - traces.source_probability_mean: object, 与概率均值曲线逐键对齐的逐 PDB 轨迹.
          - handoff.path: str, docking 投影 JSONL 绝对路径.
          - handoff.record_count: int, docking 投影记录数.
          - handoff.selection_scope: str, 固定为 evaluation_oracle_filtered.
        - docking_handoff_candidates.jsonl: 每条对应一个身份预测正确的 Stage3 候选.
          - pdb_id: str, 小写 PDB 标识.
          - receptor_condition: str, 真实受体或 CryoAtom2.
          - sorting: str, Gaussian 或来源概率均值.
          - answer_scope: str, 全配体或 small-molecule 答案范围.
          - source_blob_index: int, 完整来源 blob 编号.
          - centered_box_index: int, Find_1 centered 候选编号.
          - raw_rank: int, 完整冻结顺序中的一基位置.
          - attempt_index: int, 跳过非 Stage3 前景后的计费尝试编号.
          - blob_center_world_xyz_A: list[float], 长度 3 的世界 XYZ 中心, 单位 Å.
          - predicted_smiles: str, Matcher 预测的精确 SMILES.
          - checkpoint_path: str, 冻结 BEST 绝对路径.
          - checkpoint_sha256: str, 冻结 BEST 内容摘要.
          - manifest_path: str, test_smallmolecule manifest 绝对路径.
          - manifest_sha256: str, test_smallmolecule manifest 内容摘要.
          - selection_scope: str, 固定为 evaluation_oracle_filtered, 声明真值适用范围门禁.
    """
    import torch

    config = load_config(config_path)
    if model_name not in config["models"]:
        raise KeyError(f"配置不含模型 {model_name!r}.")
    model_record = config["models"][model_name]
    checkpoint_path = Path(model_record["checkpoint"])
    manifest_path = Path(config["outputs"]["manifest_root"]) / "manifest.json"
    stage1_path = Path(config["outputs"]["evaluation_root"]) / "stage1_only/stage1_only.json"
    manifest = read_json(manifest_path)
    stage1 = read_json(stage1_path)
    manifest_sha256 = sha256_file(manifest_path)
    if stage1.get("schema") != "matcher_test_smallmolecule_stage1_v1":
        raise ValueError("Stage1-only 结果 schema 与 test_smallmolecule 不一致.")
    if stage1.get("manifest") != {
        "path": str(manifest_path),
        "sha256": manifest_sha256,
    }:
        raise ValueError("Stage1-only 结果没有精确绑定当前 manifest 路径与 SHA-256.")
    root = manifest_path.parent
    identities = read_jsonl(root / manifest["files"]["identities"])
    small_molecule_by_identity = {
        (str(identity["pdb_id"]), str(identity["smiles"])): bool(
            identity["allowed_as_small_molecule_answer"]
        )
        for identity in identities
    }
    candidates_by_condition = {
        "real_receptor": read_jsonl(root / manifest["files"]["real_candidates"]),
        "cryoatom2_receptor": read_jsonl(
            root / manifest["files"]["cryoatom2_candidates"]
        ),
    }
    source_manifest_by_condition = {
        "real_receptor": Path(config["sources"]["real_manifest"]),
        "cryoatom2_receptor": Path(config["sources"]["cryoatom2_manifest"]),
    }

    if not torch.cuda.is_available():
        raise RuntimeError("正式 test_smallmolecule Matcher 评估要求 CUDA GPU.")
    torch.manual_seed(int(config["runtime"]["torch_seed"]))
    torch.cuda.manual_seed_all(int(config["runtime"]["torch_seed"]))
    device = torch.device("cuda")
    model, checkpoint, model_config, model_code_source = load_matcher_from_training_release(
        checkpoint_path, device
    )
    if str(model_config.variant) != model_name:
        raise ValueError(
            f"checkpoint 变体 {model_config.variant!r} 与请求模型 {model_name!r} 不一致."
        )

    checkpoint_sha256 = sha256_file(checkpoint_path)
    stage1_sha256 = sha256_file(stage1_path)
    training_run_json_path = Path(model_code_source["training_run_json_path"])
    training_run = read_json(training_run_json_path)
    training_provenance = {
        "run_json_path": str(training_run_json_path),
        "run_json_sha256": sha256_file(training_run_json_path),
        "config_path": training_run.get("config_path"),
        "resolved_config_path": training_run.get("resolved_config_path"),
        "manifest_path": training_run.get("manifest_path"),
        "manifest_snapshot_path": training_run.get("manifest_snapshot_path"),
        "source_project_root": training_run.get("source_project_root"),
        "source_release_manifest_path": training_run.get("source_release_manifest_path"),
    }
    for field in (
        "config_path",
        "resolved_config_path",
        "manifest_path",
        "manifest_snapshot_path",
        "source_release_manifest_path",
    ):
        value = training_provenance[field]
        if not isinstance(value, str) or not Path(value).is_file():
            raise FileNotFoundError(f"训练来源字段 {field} 没有指向现存普通文件: {value!r}.")
        training_provenance[f"{field}_sha256"] = sha256_file(Path(value))
    training_project_root = training_provenance["source_project_root"]
    if not isinstance(training_project_root, str) or not Path(training_project_root).is_dir():
        raise FileNotFoundError("训练 source project root 不存在.")

    evaluation_project_root_value = os.environ.get("TASK_PROJECT_ROOT")
    evaluation_launch_dir_value = os.environ.get("TASK_LAUNCH_DIR")
    if not evaluation_project_root_value or not evaluation_launch_dir_value:
        raise RuntimeError("正式评估缺少 TASK_PROJECT_ROOT 或 TASK_LAUNCH_DIR.")
    evaluation_project_root = Path(evaluation_project_root_value).resolve()
    evaluation_launch_dir = Path(evaluation_launch_dir_value).resolve()
    evaluation_release_manifest = evaluation_project_root.parent / "manifest.json"
    evaluation_launch_json = evaluation_launch_dir / "launch.json"
    evaluation_run_cmd = evaluation_launch_dir / "run_cmd.sh"
    for path in (
        evaluation_release_manifest,
        evaluation_launch_json,
        evaluation_run_cmd,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    evaluation_code = {
        "project_root": str(evaluation_project_root),
        "release_manifest_path": str(evaluation_release_manifest),
        "release_manifest_sha256": sha256_file(evaluation_release_manifest),
        "launch_dir": str(evaluation_launch_dir),
        "launch_json_path": str(evaluation_launch_json),
        "launch_json_sha256": sha256_file(evaluation_launch_json),
        "run_cmd_path": str(evaluation_run_cmd),
        "run_cmd_sha256": sha256_file(evaluation_run_cmd),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path.resolve()),
    }

    predictions_by_condition: dict[str, list[dict[str, Any]]] = {}
    for condition, candidates in candidates_by_condition.items():
        stage3_candidates = [
            candidate
            for candidate in candidates
            if candidate["target_scope"] == "stage3_foreground"
        ]
        predictions_by_condition[condition] = _evaluate_condition(
            model,
            model_config,
            source_manifest_by_condition[condition],
            str(config["runtime"]["split"]),
            small_molecule_by_identity,
            stage3_candidates,
            config["runtime"],
            device,
        )

    curves: dict[str, Any] = {"gaussian": {}, "source_probability_mean": {}}
    traces: dict[str, Any] = {"gaussian": {}, "source_probability_mean": {}}
    handoff_records: list[dict[str, Any]] = []
    for condition, candidates in candidates_by_condition.items():
        prediction_records = predictions_by_condition[condition]
        prediction_keys = [
            (str(prediction["pdb_id"]), int(prediction["source_blob_index"]))
            for prediction in prediction_records
        ]
        expected_keys = {
            (str(candidate["pdb_id"]), int(candidate["source_blob_index"]))
            for candidate in candidates
            if candidate["target_scope"] == "stage3_foreground"
        }
        if len(prediction_keys) != len(set(prediction_keys)) or set(prediction_keys) != expected_keys:
            raise RuntimeError(f"{condition}: Matcher 预测没有精确覆盖全部 Stage3 候选键.")
        predictions = {
            (str(prediction["pdb_id"]), int(prediction["source_blob_index"])): prediction
            for prediction in prediction_records
        }
        for sorting_field in ("gaussian_score", "source_probability_mean"):
            sorting_key = "gaussian" if sorting_field == "gaussian_score" else sorting_field
            for answer_scope, prediction_field in (
                ("all_ligand", "predicted_smiles_all_ligand"),
                ("small_molecule", "predicted_smiles_small_molecule"),
            ):
                curve, candidate_traces = build_topk_curve(
                    f"{model_name} {condition} {answer_scope}",
                    checkpoint_path,
                    [str(pdb_id) for pdb_id in manifest["pdb_ids"]],
                    candidates,
                    sorting_field,
                    lambda candidate, field=prediction_field: (
                        predictions[(
                            str(candidate["pdb_id"]),
                            int(candidate["source_blob_index"]),
                        )][field]
                        == candidate["target_smiles"]
                    ),
                )
                stage1_curve = stage1["curves"][sorting_key][condition]
                # 两阶段曲线必须逐候选复用 Stage1-only 排序, Matcher 不得重排 Stage1.
                if curve["candidate_order_sha256"] != stage1_curve["candidate_order_sha256"]:
                    raise RuntimeError("Stage1-only 与 Stage1+Stage2 候选顺序不一致.")
                curves[sorting_key].setdefault(condition, {})[answer_scope] = curve
                traces[sorting_key].setdefault(condition, {})[answer_scope] = candidate_traces
                for pdb_id, pdb_trace in candidate_traces.items():
                    for trace in pdb_trace:
                        if trace["target_scope"] != "stage3_foreground":
                            continue
                        prediction = predictions[(pdb_id, int(trace["source_blob_index"]))]
                        if prediction[prediction_field] != trace["target_smiles"]:
                            continue
                        handoff_records.append(
                            {
                                "pdb_id": pdb_id,
                                "receptor_condition": condition,
                                "sorting": sorting_key,
                                "answer_scope": answer_scope,
                                "source_blob_index": int(trace["source_blob_index"]),
                                "centered_box_index": int(trace["centered_box_index"]),
                                "raw_rank": int(trace["raw_rank"]),
                                "attempt_index": int(trace["attempt_index"]),
                                "blob_center_world_xyz_A": prediction[
                                    "blob_center_world_xyz_A"
                                ],
                                "predicted_smiles": prediction[prediction_field],
                                "checkpoint_path": str(checkpoint_path),
                                "checkpoint_sha256": checkpoint_sha256,
                                "manifest_path": str(manifest_path),
                                "manifest_sha256": manifest_sha256,
                                "selection_scope": "evaluation_oracle_filtered",
                            }
                        )

    output_root = (
        Path(config["outputs"]["evaluation_root"])
        / model_name
        / str(model_record["training_run_name"])
    )
    result_path = output_root / "evaluation.json"
    handoff_path = output_root / "docking_handoff_candidates.jsonl"
    if output_root.exists():
        raise FileExistsError(f"{model_name} 正式小分子结果已经存在, 禁止覆盖.")
    artifact = {
        "schema": "matcher_test_smallmolecule_model_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "best_epoch": checkpoint.get("best_epoch"),
            "best_validation_check": checkpoint.get("best_validation_check"),
            "best_score": checkpoint.get("best_score"),
            "global_step": checkpoint.get("global_step"),
        },
        "model_code_source": model_code_source,
        "training_provenance": training_provenance,
        "evaluation_code": evaluation_code,
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
        "stage1_only": {"path": str(stage1_path), "sha256": stage1_sha256},
        "source_test_manifests": {
            condition: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for condition, path in source_manifest_by_condition.items()
        },
        "runtime": {
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "task_project_root": os.environ.get("TASK_PROJECT_ROOT"),
            "task_launch_dir": os.environ.get("TASK_LAUNCH_DIR"),
            "device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "batch_size": int(config["runtime"]["batch_size"]),
            "num_workers": int(config["runtime"]["num_workers"]),
            "prefetch_factor": int(config["runtime"]["prefetch_factor"]),
            "python": platform.python_version(),
            "foreground_head_usage": "ignored",
        },
        "predictions": predictions_by_condition,
        "curves": curves,
        "traces": traces,
        "handoff": {
            "path": str(handoff_path),
            "record_count": len(handoff_records),
            "selection_scope": "evaluation_oracle_filtered",
        },
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building.", dir=output_root.parent)
    )
    try:
        write_json(temporary_root / "evaluation.json", artifact)
        write_jsonl(temporary_root / "docking_handoff_candidates.jsonl", handoff_records)
        os.replace(temporary_root, output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return result_path


def main(argv: list[str] | None = None) -> int:
    """解析命令行并执行一次正式小分子身份评估.

    输入参数:
        - argv: list[str]|None, 可测试注入的配置与模型参数; None 时读取当前进程参数.

    返回值:
        - exit_code: int, 评测目录原子发布后固定为 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--model", choices=("density_blob_only", "strongest"), required=True
    )
    arguments = parser.parse_args(argv)
    print(evaluate_model(arguments.config, arguments.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
