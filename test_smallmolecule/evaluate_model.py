"""用冻结 BEST 评估 Stage3 小分子测试集的 SMILES 身份匹配.

主要入口是 evaluate_model. 模型配置、模型类和 checkpoint 解释来自训练 run 指向的
代码快照. 当前代码只选择 Stage3 目标候选、执行两种身份 argmax、汇总 top-k, 并写出
PocketXMol 使用的候选交接文件. 本模块不读取前景阈值, 也不生成图像.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

from matcher.datasets.batching import ShuffledBatchSampler, build_candidate_loader
from matcher.datasets.candidate_data import load_manifest_dataset
from matcher.evaluation.checkpoint import load_matcher_from_training_release

from .artifacts import load_config, read_json, read_jsonl, write_json, write_jsonl
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
    """根据 stage3_candidates, 用 model 跑 “split 内唯一匹配 stage_3 candidates 的全部候选(candidates)”, 产生原始 logits 、使用/不使用 small_molecule_mask 的SMILES身份选择结果。
        不跑假阳性 candidates、匹配的不是 stage3_small_molecule 的candidates(前者消耗次数, 后者不消耗)。

    输入参数:
        - model: 从训练代码快照恢复且处于评估模式的 Matcher.
        - model_config: checkpoint 中的模型配置, variant 决定 Dataset 实际读取字段.
        - source_manifest_path: Path, 当前受体条件的完整 test_0 manifest.
        - split: str, 测试划分名称, 如 held_out_test_0.
        - small_molecule_by_identity: dict[tuple[str,str], bool], PDB 与精确 SMILES 到 small_molecule 答案资格的映射.
        - stage3_candidates: list[dict[str, Any]], 唯一目标属于 Stage3 的 Find_1 候选.
        - runtime: dict[str, Any], DataLoader 批量、worker、随机种子、PP 上限和缓存预算.
        - device: torch.device, 当前 CUDA 设备.

    返回值:
        - predictions: list[dict[str, Any]], 每个 Stage3 目标候选一条, 保存候选标识、唯一目标、完整身份分数、两种答案范围的预测 SMILES 和 blob 世界中心.
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
                    key: (value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value)
                    for key, value in cpu_batch.items()
                }
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(batch)
                # float32, (B, I_max), 每个候选对当前 PDB 完整身份轴的未归一化匹配分数.
                identity_logits = output["identity_logits"].float()
                # bool, (B, I_max), True 表示当前候选所属 PDB 的有效身份槽, False 表示批次补齐槽.
                identity_mask = batch["identity_mask"].bool()
                # bool, (B, I_max), True 表示有效身份槽可作为 small_molecule 答案.
                small_molecule_mask = torch.zeros_like(identity_mask)
                for sample_index, smiles_axis in enumerate(cpu_batch["identity_smiles"]):
                    pdb_id = str(cpu_batch["metadata"][sample_index]["pdb_id"])
                    allowed = [
                        small_molecule_by_identity[(pdb_id, str(smiles))]
                        for smiles in smiles_axis
                    ]
                    small_molecule_mask[sample_index, : len(allowed)] = torch.tensor(allowed, dtype=torch.bool, device=device)

                # (B,): 第 b 个样本选中的 SMILES-index
                full_index = (
                    identity_logits.masked_fill(~identity_mask, float("-inf"))
                    .argmax(dim=1)
                    .cpu()
                    .tolist()
                )
                small_molecule_index = (
                    identity_logits.masked_fill(~small_molecule_mask, float("-inf"))
                    .argmax(dim=1)
                    .cpu()
                    .tolist()
                )
                raw_logits = identity_logits.cpu().tolist()

                for sample_index, metadata in enumerate(cpu_batch["metadata"]):
                    pdb_id = str(metadata["pdb_id"])
                    centered_box_index = int(metadata["centered_box_index"])
                    source_blob_index = int(metadata["source_blob_index"])
                    frozen_candidate = stage3_by_key[(pdb_id, centered_box_index, source_blob_index)]
                    # (I, ), list[str], 这个 PDB 的所有可用 SMILES
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
                            "blob_center_world_xyz_A": frozen_candidate["blob_center_world_xyz_A"],
                            "identity_smiles": smiles_axis,
                            "identity_logits": raw_logits[sample_index][ : len(smiles_axis)],
                            "small_molecule_answer_mask": allowed,
                            "predicted_smiles_all_ligand": smiles_axis[int(full_index[sample_index])],
                            "predicted_smiles_small_molecule": smiles_axis[int(small_molecule_index[sample_index])],
                        }
                    )
    finally:
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()
        del iterator
        del loader
        del dataset
        gc.collect()
    return predictions


def evaluate_model(config_path: Path, model_name: str) -> Path:
    """评估一个冻结 Matcher BEST, 并写出 top-k 结果与 PocketXMol handoff.

    输入参数:
        - config_path: Path, 小分子测试 YAML.
        - model_name: str, config.models 中的稳定实验标识, 如 strongest-1.

    返回值:
        - result_path: Path, 写出的 evaluation.json.

    目录产物:
        - evaluation.json: object, 顶层只含 predictions、curves 和 traces.
        - docking_handoff_candidates.jsonl: 每条对应一个身份预测正确的 Stage3 候选, 保存受体条件、排序、答案范围、候选编号、排名、世界中心和预测 SMILES.
    """
    import torch

    config = load_config(config_path)
    model_record = config["models"][model_name]
    checkpoint_path = Path(model_record["checkpoint"])
    manifest_path = Path(config["outputs"]["manifest_root"]) / "manifest.json"
    manifest = read_json(manifest_path)
    root = manifest_path.parent
    identities = read_jsonl(root / manifest["files"]["identities"])
    small_molecule_by_identity = {
        (str(identity["pdb_id"]), str(identity["smiles"])):  bool(identity["allowed_as_small_molecule_answer"])
        for identity in identities
    }
    candidates_by_condition = {
        "real_receptor": read_jsonl(root / manifest["files"]["real_candidates"]),
        "cryoatom2_receptor": read_jsonl( root / manifest["files"]["cryoatom2_candidates"]),
    }
    source_manifest_by_condition = {
        "real_receptor": Path(config["sources"]["real_manifest"]),
        "cryoatom2_receptor": Path(config["sources"]["cryoatom2_manifest"]),
    }

    torch.manual_seed(int(config["runtime"]["torch_seed"]))
    torch.cuda.manual_seed_all(int(config["runtime"]["torch_seed"]))
    device = torch.device("cuda")
    model, _, model_config = load_matcher_from_training_release(checkpoint_path, device)
    predictions_by_condition: dict[str, list[dict[str, Any]]] = {}
    for condition, candidates in candidates_by_condition.items():
        predictions_by_condition[condition] = _evaluate_condition(
            model,
            model_config,
            source_manifest_by_condition[condition],
            str(config["runtime"]["split"]),
            small_molecule_by_identity,
            [candidate for candidate in candidates if candidate["target_scope"] == "stage3_foreground"],
            config["runtime"],
            device,
        )

    curves: dict[str, Any] = {
        "gaussian": {},
        "source_probability_mean": {},
    }
    traces: dict[str, Any] = {
        "gaussian": {},
        "source_probability_mean": {},
    }
    handoff_records: list[dict[str, Any]] = []

    for condition, candidates in candidates_by_condition.items():
        predictions = {
            (
                str(prediction["pdb_id"]),
                int(prediction["source_blob_index"]),
            ): prediction
            for prediction in predictions_by_condition[condition]
        }
        for sorting_field in ("gaussian_score", "source_probability_mean"):
            sorting_key = ("gaussian" if sorting_field == "gaussian_score" else sorting_field)
            for answer_scope, prediction_field in (
                ("all_ligand", "predicted_smiles_all_ligand"),
                ("small_molecule", "predicted_smiles_small_molecule"),
            ):
                curve, candidate_traces = build_topk_curve(
                    [str(pdb_id) for pdb_id in manifest["pdb_ids"]],
                    candidates,
                    sorting_field,
                    lambda candidate, field=prediction_field:   predictions[(str(candidate["pdb_id"]), int(candidate["source_blob_index"]))][field] == candidate["target_smiles"],
                )
                curves[sorting_key].setdefault(condition, {})[answer_scope] = curve
                traces[sorting_key].setdefault(condition, {})[answer_scope] = candidate_traces

                # 再额外写 docking_handoff_candidates.jsonl, traces 仍将完整保存
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
                                "blob_center_world_xyz_A": prediction["blob_center_world_xyz_A"],
                                "predicted_smiles": prediction[prediction_field],
                            }
                        )

    output_root = (
        Path(config["outputs"]["evaluation_root"])
        / model_name
        / str(model_record["training_run_name"])
    )
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "evaluation.json"
    write_json(
        result_path,
        {
            "predictions": predictions_by_condition,
            "curves": curves,
            "traces": traces,
        },
    )
    write_jsonl(
        output_root / "docking_handoff_candidates.jsonl",
        handoff_records,
    )
    return result_path


def main(argv: list[str] | None = None) -> int:
    """解析配置和实验标识, 并执行一次小分子身份评估."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    arguments = parser.parse_args(argv)
    print(evaluate_model(arguments.config, arguments.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
