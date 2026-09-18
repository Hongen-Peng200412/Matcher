"""连接冻结 test_0 评估事实, 并生成可复用的 Stage1-only top-k 元数据.

主要入口是 build_stage1_only_preview_metadata 和 build_final_test_artifact. 前者只读取三套
Stage1 正式来源并写预览 JSON, 后者在阶段 E 加载真实 BEST 和两份 test_0 manifest 后写
候选预测、测试指标与 Stage1+Matcher 曲线. 本模块不拟合测试阈值且不修改来源产物.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.json_artifacts import write_json


COVERAGE_THRESHOLD = 0.30
TOP_K_VALUES = tuple(range(1, 21))
EXPECTED_PDB_COUNT = 179


def _candidate_hit_mask(
    intersections: np.ndarray, pred_sizes: np.ndarray, gt_sizes: np.ndarray
) -> np.ndarray:
    """
    判断每个候选是否命中任一真实 occurrence.

    - intersections: 整数 (N_candidate,N_occurrence); 候选与真实 occurrence 的交集体素数.
    - pred_sizes: 整数 (N_candidate,); 每个候选的体素数.
    - gt_sizes: 整数 (N_occurrence,); 每个真实 occurrence 的体素数.
    - return: bool (N_candidate,); 至少一个 occurrence 的候选覆盖率与 occurrence 覆盖率 同时不低于 0.30 时为 True.
    """
    overlap = np.asarray(intersections, dtype=np.int64)
    predicted = np.asarray(pred_sizes, dtype=np.int64)
    truth = np.asarray(gt_sizes, dtype=np.int64)
    if overlap.shape != (len(predicted), len(truth)):
        raise ValueError("intersections 必须与候选和 occurrence 轴对齐。")
    return np.any(
        (overlap >= COVERAGE_THRESHOLD * predicted[:, None])
        & (overlap >= COVERAGE_THRESHOLD * truth[None, :]),
        axis=1,
    )


def _ranked_hit_curve(
    score: np.ndarray,
    source_blob_index: np.ndarray,
    selected: np.ndarray,
    candidate_hit: np.ndarray,
) -> list[bool]:
    """
    按冻结的 Stage1 顺序返回一个 PDB 的 top-k 命中状态.

    - score: 浮点 (N_candidate,); 当前排序采用的 Gaussian 分数或来源概率均值.
    - source_blob_index: 整数 (N_candidate,); 同分候选的升序并列键.
    - selected: bool (N_candidate,); True 候选整段排在 False 候选之前.
    - candidate_hit: bool (N_candidate,); 由双向 0.30 覆盖门槛得到的 Stage1 命中标志.
    - return: 长度 20 的布尔列表; 第 k-1 项表示扩展顺序前 k 个候选中是否至少一次命中.
    """
    score = np.asarray(score, dtype=np.float64)
    blob = np.asarray(source_blob_index, dtype=np.int64)
    selected = np.asarray(selected, dtype=np.bool_)
    hit = np.asarray(candidate_hit, dtype=np.bool_)
    if not (score.shape == blob.shape == selected.shape == hit.shape):
        raise ValueError("候选分数、编号、选择掩码和命中掩码必须同形。")
    order = _selected_priority_order(score, blob, selected)
    ranked_hit = hit[order]
    return [bool(np.any(ranked_hit[:k])) for k in TOP_K_VALUES]


def _selected_priority_order(
    score: np.ndarray,
    source_blob_index: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    """
    生成 selected 优先的唯一扩展候选顺序.

    - score: 浮点 (N_candidate,), 当前排序使用的 Stage1 分数.
    - source_blob_index: 整数 (N_candidate,), 分数并列时较小者优先.
    - selected: bool (N_candidate,), 决定候选进入前段或追加段.
    - return: 整数 (N_candidate,), 先含全部 selected 候选, 再含全部 unselected 候选.

    两段内部均按分数降序、source_blob_index 升序排列. 当两项键都相同时,
    np.lexsort 保留来源数组中的确定位置, 因此不会让 unselected 候选挤占 selected 前缀.
    """
    score = np.asarray(score, dtype=np.float64)
    blob = np.asarray(source_blob_index, dtype=np.int64)
    selected = np.asarray(selected, dtype=np.bool_)
    if not (score.shape == blob.shape == selected.shape) or score.ndim != 1:
        raise ValueError("候选分数、编号和选择掩码必须是一维同形数组。")
    segments: list[np.ndarray] = []
    for flag in (True, False):
        indices = np.flatnonzero(selected == flag)
        segments.append(indices[np.lexsort((blob[indices], -score[indices]))])
    return np.concatenate(segments)


def _assert_selected_prefix_unchanged(
    score: np.ndarray,
    source_blob_index: np.ndarray,
    selected: np.ndarray,
) -> None:
    """
    证明扩展顺序的 selected 前缀与原 selected-only 顺序逐候选相同.

    - score / source_blob_index / selected: 与 _selected_priority_order 相同的三个候选轴字段.
    - return: None; 前缀发生挤占、遗漏或重排时抛出 RuntimeError.
    """
    score = np.asarray(score, dtype=np.float64)
    blob = np.asarray(source_blob_index, dtype=np.int64)
    selected = np.asarray(selected, dtype=np.bool_)
    selected_indices = np.flatnonzero(selected)
    original_order = selected_indices[np.lexsort((blob[selected_indices], -score[selected_indices]))]
    expanded_order = _selected_priority_order(score, blob, selected)
    if not np.array_equal(expanded_order[: len(original_order)], original_order):
        raise RuntimeError("扩展 top-k 改变了原 selected-only 候选前缀。")


def _find_probability_mean(pdb_root: Path) -> dict[int, float]:
    """
    读取 Find_1 F2 blob 的原生来源概率均值.

    - pdb_root: 一个 PDB 的 Find_1 产物目录.
    - return: source_blob_index 到 source_probability_mean 的一一映射; 重复编号或字段轴 不一致时立即失败.
    """
    path = pdb_root / "blobs/F2_blobs.npz"
    with np.load(path, allow_pickle=False) as archive:
        blob = np.asarray(archive["blob_index"], dtype=np.int64)
        mean = np.asarray(archive["source_probability_mean"], dtype=np.float64)
    if blob.shape != mean.shape or len(set(blob.tolist())) != len(blob):
        raise ValueError(f"{path}: blob_index 与 source_probability_mean 未形成唯一对齐。")
    return {int(blob[index]): float(mean[index]) for index in range(len(blob))}


def _curve_record(
    name: str,
    source: Path,
    pdb_hits: dict[str, list[bool]],
) -> dict[str, Any]:
    """
    把逐 PDB 命中状态汇总为固定 179 分母的曲线记录.

    - name: 图例使用的完整曲线名称.
    - source: 生成该曲线的只读目录或 checkpoint 路径.
    - pdb_hits: PDB 标识到长度 20 布尔命中序列的映射; 必须恰有 179 个键.
    - return: 同时保存命中数、命中比例、PDB 列表和逐 PDB 命中的可复用记录.
    """
    pdb_ids = sorted(pdb_hits)
    if len(pdb_ids) != EXPECTED_PDB_COUNT:
        raise ValueError(f"{name} 必须覆盖 {EXPECTED_PDB_COUNT} 个 PDB，实际为 {len(pdb_ids)}。")
    counts = [sum(int(pdb_hits[pdb_id][index]) for pdb_id in pdb_ids) for index in range(20)]
    return {
        "name": name,
        "source": str(source),
        "pdb_count": EXPECTED_PDB_COUNT,
        "pdb_ids": pdb_ids,
        "top_k": list(TOP_K_VALUES),
        "hit_count": counts,
        "hit_ratio": [count / EXPECTED_PDB_COUNT for count in counts],
        "pdb_hit": {pdb_id: pdb_hits[pdb_id] for pdb_id in pdb_ids},
    }


# ================================================================================================


def build_find_stage1_curves(stage1_test_root: Path, label: str) -> dict[str, dict[str, Any]]:
    """
    从一套 Find_1 held_out_test_0 重算两条 Stage1-only 曲线.

    - stage1_test_root: 含 179 个 PDB 子目录的 held_out_test_0 根目录.
    - label: 区分真实受体或 CryoAtom2 受体的图例前缀.
    - return.gaussian: 按 candidate_score 降序得到的固定分母曲线.
    - return.source_probability_mean: 按 F2 blob 原生来源概率均值降序得到的同候选集合曲线.
    """
    root = Path(stage1_test_root)
    gaussian_hits: dict[str, list[bool]] = {}
    probability_hits: dict[str, list[bool]] = {}
    pdb_roots = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "evaluation/f2_centered_gaussian_macro_selected.npz").is_file()
    )
    for pdb_root in pdb_roots:
        path = pdb_root / "evaluation/f2_centered_gaussian_macro_selected.npz"
        with np.load(path, allow_pickle=False) as archive:
            blob = np.asarray(archive["source_blob_index"], dtype=np.int64)
            gaussian = np.asarray(archive["candidate_score"], dtype=np.float64)
            selected = np.asarray(archive["candidate_selected"], dtype=np.bool_)
            candidate_hit = _candidate_hit_mask(
                archive["intersections"], archive["pred_sizes"], archive["gt_sizes"]
            )
        probability_by_blob = _find_probability_mean(pdb_root)
        probability = np.asarray([probability_by_blob[int(value)] for value in blob], dtype=np.float64)
        _assert_selected_prefix_unchanged(gaussian, blob, selected)
        _assert_selected_prefix_unchanged(probability, blob, selected)
        gaussian_hits[pdb_root.name.lower()] = _ranked_hit_curve(
            gaussian, blob, selected, candidate_hit
        )
        probability_hits[pdb_root.name.lower()] = _ranked_hit_curve(
            probability, blob, selected, candidate_hit
        )
    return {
        "gaussian": _curve_record(f"{label} Find_1 Stage1-only", root, gaussian_hits),
        "source_probability_mean": _curve_record(
            f"{label} Find_1 Stage1-only", root, probability_hits
        ),
    }


def build_emap2lig_stage1_curve(evaluation_root: Path) -> dict[str, Any]:
    """
    按 Emap2lig 原生 ligand probability 均值重算单条曲线.

    - evaluation_root: 正式 Emap2lig evaluation 目录; 逐 PDB 文件位于 per_pdb.
    - return: 固定 179 分母的 Stage1-only 曲线; top-3/4/5 必须严格复现 87/89/90.

    Emap2lig 不读取受体模型, 因此本函数只返回一条 density-only、receptor-independent
    曲线, 不复制为真实受体与 CryoAtom2 两条曲线.
    """
    root = Path(evaluation_root)
    pdb_hits: dict[str, list[bool]] = {}
    for path in sorted((root / "per_pdb").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            blob = np.asarray(archive["source_blob_index"], dtype=np.int64)
            score = np.asarray(archive["candidate_score"], dtype=np.float64)
            selected = np.asarray(archive["candidate_selected"], dtype=np.bool_)
            candidate_hit = _candidate_hit_mask(
                archive["intersections"], archive["pred_sizes"], archive["gt_sizes"]
            )
        _assert_selected_prefix_unchanged(score, blob, selected)
        pdb_hits[path.stem.lower()] = _ranked_hit_curve(score, blob, selected, candidate_hit)
    curve = _curve_record(
        "Emap2lig Find Stage1-only (density-only, receptor-independent)",
        root,
        pdb_hits,
    )
    return curve


def build_stage1_only_preview_metadata(
    real_test_root: Path,
    cryoatom2_test_root: Path,
    emap2lig_evaluation_root: Path,
    output_path: Path,
) -> Path:
    """
    冻结三套来源的 Stage1-only 事实.

    - real_test_root: Find_1 真实受体 held_out_test_0 根目录.
    - cryoatom2_test_root: Find_1 CryoAtom2 受体 held_out_test_0 根目录.
    - emap2lig_evaluation_root: Emap2lig 正式 evaluation 根目录.
    - output_path: Matcher 评估产物中的 JSON 输出路径.
    - return: 已原子写入的 output_path.
    - output.schema: str. 固定为 matcher_stage1_only_preview_v1.
    - output.status: str. 固定为 stage1_only_preview, 明确不含 Matcher 数值.
    - output.coverage_threshold: float. 固定为 0.30.
    - output.denominator_pdb_count: int. 固定为 179.
    - output.stage1_only_hit_definition: str. top-k 内任一候选满足双向覆盖门槛即命中的定义.
    - output.candidate_set_note: str. Find_1 与 Emap2lig 候选集合不逐候选配对的边界.
    - output.curves.gaussian: dict. 含 real_receptor、cryoatom2_receptor 和唯一 emap2lig 曲线.
    - output.curves.source_probability_mean: dict. 含同三条曲线, 其中 emap2lig 与 Gaussian 记录逐点相同.
    - output.curves.*.*.name: str. 图例使用的完整曲线名称.
    - output.curves.*.*.source: str. 只读来源目录.
    - output.curves.*.*.pdb_count: int. 固定分母 179.
    - output.curves.*.*.pdb_ids: list[str]. 曲线分母中的 179 个有序唯一 PDB 标识.
    - output.curves.*.*.top_k: list[int]. 固定为 1 到 20.
    - output.curves.*.*.hit_count: list[int]. 与 top_k 对齐的 PDB 命中数.
    - output.curves.*.*.hit_ratio: list[float]. hit_count 除以 179.
    - output.curves.*.*.pdb_hit: dict[str, list[bool]]. 每个 PDB 在 k=1..20 的累计命中状态.

    产物状态固定为 stage1_only_preview, 不创建 Matcher 概率、预测 SMILES 或 Stage1+Matcher
    曲线. 两种排序共享同一条 Emap2lig 曲线对象.
    """
    real = build_find_stage1_curves(real_test_root, "Find_1 real-receptor")
    cryo = build_find_stage1_curves(cryoatom2_test_root, "Find_1 CryoAtom2-receptor")
    emap = build_emap2lig_stage1_curve(emap2lig_evaluation_root)
    if not (
        set(real["gaussian"]["pdb_ids"])
        == set(cryo["gaussian"]["pdb_ids"])
        == set(emap["pdb_ids"])
    ):
        raise RuntimeError("三套 Stage1-only 来源的 179 个 PDB 集合不一致。")
    artifact = {
        "schema": "matcher_stage1_only_preview_v1",
        "status": "stage1_only_preview",
        "coverage_threshold": COVERAGE_THRESHOLD,
        "denominator_pdb_count": EXPECTED_PDB_COUNT,
        "candidate_order": (
            "candidate_selected=true 按指定分数排序后，追加 candidate_selected=false 的同分数排序"
        ),
        "stage1_only_hit_definition": (
            "top-k 内至少一个候选与任一真实 occurrence 的 candidate_coverage 和 "
            "occurrence_coverage 均不低于 0.30"
        ),
        "candidate_set_note": (
            "Find_1 与 Emap2lig 保留各自候选生成、选择和排序契约，不作逐候选配对。"
        ),
        "curves": {
            "gaussian": {"real_receptor": real["gaussian"], "cryoatom2_receptor": cryo["gaussian"], "emap2lig": emap},
            "source_probability_mean": {"real_receptor": real["source_probability_mean"], "cryoatom2_receptor": cryo["source_probability_mean"], "emap2lig": emap},
        },
    }
    write_json(Path(output_path), artifact)
    return Path(output_path)


def _run_matcher_condition(
    checkpoint_path: Path,
    manifest_path: Path,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    pp_top_k: int,
    cache_gib_per_worker: float,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[str]]:
    """
    对一个受体条件执行同一冻结 BEST, 并保留候选预测.

    - checkpoint_path: 阶段 E 产生的真实 BEST checkpoint; 必须含模型配置和 validation 拟合的前景阈值.
    - manifest_path / split: 当前受体条件的冻结测试清单及划分键.
    - batch_size / num_workers: 冻结推理的物理批量和持久工作进程数.
    - pp_top_k / cache_gib_per_worker: Dataset 的 PP token 上限与每 worker 缓存预算.
    - return[0]: 在原 selected 候选口径上使用 checkpoint 前景阈值计算的二分类和 SMILES 指标.
    - return[1]: PDB 到全部 centered 候选预测记录列表的映射; 两种扩展排序共同复用.
    - return[2]: manifest summary 冻结的 179 个唯一 PDB 标识, 保留原有顺序.

    manifest 中存在但没有 centered 候选的 PDB 保留在指标 PDB 计数中, 候选记录映射不为其
    虚构空记录.
    """
    import torch

    from ..data.manifest import load_manifest
    from ..datasets.batching import ShuffledBatchSampler, build_candidate_loader
    from ..datasets.candidate_data import load_manifest_dataset
    from ..model.config import MatcherModelConfig
    from ..model.ligand_matcher import ClosedSetMatcher
    from ..training.checkpointing import load_checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError("正式 Matcher test 要求 CUDA GPU。")
    device = torch.device("cuda")
    checkpoint = dict(load_checkpoint(checkpoint_path, map_location=device))
    threshold = checkpoint.get("foreground_threshold")
    model_config = checkpoint.get("model_config")
    if not isinstance(threshold, (int, float)) or not np.isfinite(float(threshold)):
        raise ValueError("BEST checkpoint 缺少有限 foreground_threshold。")
    if not isinstance(model_config, dict):
        raise ValueError("BEST checkpoint 缺少 model_config。")
    config = MatcherModelConfig.from_dict(model_config)
    dataset = load_manifest_dataset(
        manifest_path,
        split=split,
        selected_only=False,
        mode="test",
        variant=config.variant,
        pp_top_k=pp_top_k,
        cache_gib_per_worker=cache_gib_per_worker,
    )
    sampler = ShuffledBatchSampler(dataset, batch_size, False, False, 0, None)
    loader = build_candidate_loader(dataset, sampler, num_workers, 2, 1)
    model = ClosedSetMatcher(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    rows_by_pdb: dict[str, list[dict[str, Any]]] = {}
    with torch.no_grad():
        for cpu_batch in loader:
            batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in cpu_batch.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch)  # type: ignore[arg-type]
            foreground_probability = torch.sigmoid(output["foreground_logit"]).float().cpu().numpy()
            predicted_index = output["identity_logits"].argmax(dim=1).cpu().tolist() if output["identity_logits"].shape[1] else [-1] * len(cpu_batch["metadata"])
            for index, metadata in enumerate(cpu_batch["metadata"]):
                smiles_axis = cpu_batch["identity_smiles"][index]
                slot = int(predicted_index[index])
                predicted_smiles = smiles_axis[slot] if 0 <= slot < len(smiles_axis) else None
                row = {
                    "centered_box_index": metadata["centered_box_index"],
                    "source_blob_index": metadata["source_blob_index"],
                    "gaussian_score": metadata["gaussian_score"],
                    "source_probability_mean": metadata["source_probability_mean"],
                    "candidate_selected": metadata["candidate_selected"],
                    "foreground_target": metadata["foreground_target"],
                    "target_category": metadata["target_category"],
                    "matched_occurrence_id": metadata["matched_occurrence_id"],
                    "target_smiles": metadata["target_smiles"],
                    "identity_supported": metadata["identity_supported"],
                    "predicted_smiles": predicted_smiles,
                    "foreground_probability": float(foreground_probability[index]),
                }
                rows_by_pdb.setdefault(metadata["pdb_id"], []).append(row)
    manifest = load_manifest(manifest_path)
    summary = manifest["summary"].get(split)
    if not isinstance(summary, dict) or not isinstance(summary.get("pdb_ids"), list):
        raise ValueError("测试 manifest 摘要缺少冻结的 pdb_ids。")
    manifest_pdb_ids = [str(pdb_id).lower() for pdb_id in summary["pdb_ids"]]
    if len(manifest_pdb_ids) != EXPECTED_PDB_COUNT or len(set(manifest_pdb_ids)) != EXPECTED_PDB_COUNT:
        raise ValueError("正式 test_0 manifest 必须冻结 179 个唯一 PDB 标识。")
    expected_pdb_ids = set(manifest_pdb_ids)
    extra_prediction_ids = sorted(set(rows_by_pdb) - expected_pdb_ids)
    if extra_prediction_ids:
        raise ValueError(f"Matcher 预测含 manifest 集合外 PDB: {extra_prediction_ids[:5]}。")
    metrics = _metrics_from_rows(rows_by_pdb, float(threshold), expected_pdb_ids)
    return metrics, rows_by_pdb, manifest_pdb_ids


def _metrics_from_rows(
    rows_by_pdb: dict[str, list[dict[str, Any]]],
    threshold: float,
    pdb_ids: set[str] | None = None,
) -> dict[str, Any]:
    """
    从一次 test_0 冻结推理重算全量或指定 PDB 子集指标.

    - rows_by_pdb: 一次冻结前向生成的全部候选预测; 每个记录同时保存目标、预测和选择标志.
    - threshold: BEST checkpoint 中保存的 validation 前景阈值; 本函数不重新拟合.
    - pdb_ids: 可选 PDB 集合; None 使用所有有候选的 PDB, 显式集合允许包含零候选 PDB.
    - return: 原 selected 候选口径的前景、SMILES 和不受支持身份明细; pdb_count 使用显式集合分母.
    """
    from .metrics import ValidationAccumulator

    selected_pdb_ids = set(rows_by_pdb) if pdb_ids is None else pdb_ids
    accumulator = ValidationAccumulator.empty()
    unsupported: list[dict[str, Any]] = []
    for pdb_id in sorted(selected_pdb_ids):
        for row in rows_by_pdb.get(pdb_id, []):
            if not bool(row["candidate_selected"]):
                continue
            accumulator.foreground_target.append(bool(row["foreground_target"]))
            accumulator.foreground_probability.append(float(row["foreground_probability"]))
            accumulator.predicted_smiles.append(row["predicted_smiles"])
            accumulator.target_smiles.append(row["target_smiles"])
            accumulator.target_category.append(row["target_category"])
            accumulator.identity_supported.append(bool(row["identity_supported"]))
            if row["foreground_target"] and not row["identity_supported"]:
                unsupported.append(
                    {
                        "pdb_id": pdb_id,
                        "centered_box_index": row["centered_box_index"],
                        "source_blob_index": row["source_blob_index"],
                        "matched_occurrence_id": row["matched_occurrence_id"],
                        "target_smiles": row["target_smiles"],
                    }
                )
    metrics = accumulator.finalize(threshold)
    metrics.pop("selection", None)
    metrics["pdb_count"] = len(selected_pdb_ids)
    metrics["unsupported_identity_target_candidate_count"] = len(unsupported)
    metrics["unsupported_identity_targets"] = unsupported
    return metrics


def _read_pdb_subset(path: Path) -> set[str]:
    """
    读取官方 test_1 补充子集列表.

    - path: UTF-8 文本路径, 每个非空行保存一个 PDB 标识.
    - return: 统一小写后的 149 个唯一 PDB 标识.
    """
    pdb_ids = {
        line.strip().lower()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(pdb_ids) != 149:
        raise ValueError(f"test_1 官方 PDB 列表必须含 149 个唯一标识，实际为 {len(pdb_ids)}。")
    return pdb_ids


def _preview_pdb_axis(curve: dict[str, Any], label: str) -> list[str]:
    """
    读取并验证一条 Stage1-only 预览曲线的固定 PDB 轴.

    - curve: stage1_only_preview 中的一条曲线记录.
    - label: 错误消息使用的受体条件与排序名称.
    - return: 179 个小写唯一 PDB 标识, 顺序与曲线命中分母一致.
    """
    if not isinstance(curve, dict):
        raise ValueError(f"{label} 曲线必须是对象映射。")
    values = curve.get("pdb_ids")
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{label} 缺少字符串 pdb_ids 列表。")
    pdb_ids = [value.lower() for value in values]
    if len(pdb_ids) != EXPECTED_PDB_COUNT or len(set(pdb_ids)) != EXPECTED_PDB_COUNT:
        raise ValueError(f"{label} 必须含 179 个唯一 PDB 标识。")
    return pdb_ids


def _matcher_curve(
    name: str,
    rows_by_pdb: dict[str, list[dict[str, Any]]],
    pdb_ids: list[str],
    sorting: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """
    按冻结 Stage1 顺序字段汇总 Stage1+Matcher top-k 曲线.

    - name: 图例使用的完整曲线名称.
    - rows_by_pdb: 同一次 Matcher 冻结推理的候选预测.
    - pdb_ids: Stage1-only 元数据给出的有序 179-PDB 列表.
    - sorting: gaussian_score 或 source_probability_mean; 只改变 selected 段和追加段各自的内部顺序.
    - checkpoint_path: 曲线来源记录使用的 BEST checkpoint 路径.
    - return: 固定 179 分母的 k=1..20 曲线; 零候选 PDB 的 20 个命中值均为 False.

    每个 PDB 先排列全部 selected 候选, 再追加全部 unselected 候选. 不受支持身份、背景候选或
    预测 SMILES 不等于确定目标 SMILES 的候选均不能成为命中; 它们保留原排名, 不删除后补位.
    """
    expected_pdb_ids = set(pdb_ids)
    extra = sorted(set(rows_by_pdb) - expected_pdb_ids)
    if extra:
        raise ValueError(f"Matcher 冻结预测含 179-PDB 集合外标识: {extra[:5]}。")
    pdb_hit: dict[str, list[bool]] = {}
    for pdb_id in pdb_ids:
        rows = rows_by_pdb.get(pdb_id, [])
        score = np.asarray([float(row[sorting]) for row in rows], dtype=np.float64)
        blob = np.asarray([int(row["source_blob_index"]) for row in rows], dtype=np.int64)
        selected = np.asarray([bool(row["candidate_selected"]) for row in rows], dtype=np.bool_)
        order = _selected_priority_order(score, blob, selected)
        ordered = [rows[int(index)] for index in order]
        correct = [
            bool(row["foreground_target"])
            and bool(row["identity_supported"])
            and row["predicted_smiles"] == row["target_smiles"]
            for row in ordered
        ]
        pdb_hit[pdb_id] = [any(correct[:k]) for k in TOP_K_VALUES]
    return _curve_record(name, checkpoint_path, pdb_hit)


def build_final_test_artifact(
    checkpoint_path: Path,
    real_manifest_path: Path,
    cryoatom2_manifest_path: Path,
    split: str,
    stage1_preview_path: Path,
    output_path: Path,
    *,
    test1_pdb_list_path: Path | None,
    batch_size: int,
    num_workers: int,
    pp_top_k: int,
    cache_gib_per_worker: float,
) -> Path:
    """
    阶段 E 使用同一 BEST 生成正式测试产物.

    - checkpoint_path: 四项训练完成后选定的真实 BEST checkpoint.
    - real_manifest_path / cryoatom2_manifest_path: 两种受体条件的冻结 test_0 manifest.
    - split: 两份 manifest 共同的测试划分键.
    - stage1_preview_path: 本轮已验收的 Stage1-only 元数据.
    - output_path: 最终冻结测试 JSON 路径.
    - test1_pdb_list_path: 可选官方 149-PDB 列表; 只筛选 test_0 预测, 不重新前向.
    - batch_size / num_workers / pp_top_k / cache_gib_per_worker: 两种受体条件共用的推理参数.
    - return: 已原子写入的 output_path.
    - output.schema: str. 固定为 matcher_frozen_test_v1.
    - output.status: str. 固定为 final_with_matcher.
    - output.checkpoint_path: str. 真实 BEST checkpoint 路径.
    - output.real_manifest_path: str. 真实受体 test_0 manifest 路径.
    - output.cryoatom2_manifest_path: str. CryoAtom2 test_0 manifest 路径.
    - output.stage1_preview_path: str. 已验收 Stage1-only 预览 JSON 路径.
    - output.foreground_threshold_source: str. 固定说明阈值来自 BEST validation 拟合.
    - output.metrics.real_receptor: dict. 真实受体候选的前景 PRAUC/F1、固定阈值、SMILES 分层准确率、支持数和 unsupported 明细, 不含 selection.
    - output.metrics.cryoatom2_receptor: dict. CryoAtom2 候选的同字段测试指标, 不含 selection.
    - output.predictions.real_receptor: dict[str, list[dict]]. PDB 到真实受体逐候选冻结预测的映射.
    - output.predictions.cryoatom2_receptor: dict[str, list[dict]]. PDB 到 CryoAtom2 逐候选冻结预测的映射.
    - output.predictions.*[].centered_box_index: int. 当前候选稳定编号.
    - output.predictions.*[].source_blob_index: int. 当前候选源 blob 编号.
    - output.predictions.*[].gaussian_score: float. Stage1 Gaussian 排序分数.
    - output.predictions.*[].source_probability_mean: float. Stage1 原生概率均值.
    - output.predictions.*[].foreground_target: bool. 几何前景真值.
    - output.predictions.*[].target_category: str 或 None. 五类分层标签.
    - output.predictions.*[].matched_occurrence_id: int 或 None. 最大覆盖 occurrence 编号.
    - output.predictions.*[].target_smiles: str 或 None. 唯一正确身份.
    - output.predictions.*[].identity_supported: bool. 目标图和语言资产是否均可用.
    - output.predictions.*[].predicted_smiles: str 或 None. PDB 内闭集 argmax 精确 SMILES.
    - output.predictions.*[].foreground_probability: float. sigmoid 后前景概率.
    - output.curves.gaussian: dict. 真实受体/CryoAtom2 的 Stage1-only 与 Stage1+Matcher 曲线及唯一 Emap2lig 曲线.
    - output.curves.source_probability_mean: dict. 同一冻结预测按概率均值排序的五条曲线.
    - output.test_1: 可选 dict. 仅复用 test_0 预测筛选官方 149-PDB 严格子集, 保存来源、定义和两种受体指标.

    本函数分别对真实受体和 CryoAtom2 运行一次 Matcher; 每次产生的候选预测同时支持
    Gaussian 与来源概率均值排序, 不为第二种排序重复前向.
    """
    preview = json.loads(Path(stage1_preview_path).read_text(encoding="utf-8"))
    if preview.get("status") != "stage1_only_preview":
        raise ValueError("最终评估必须连接已验收的 Stage1-only 预览元数据。")
    real_metrics, real_rows, real_manifest_pdb_ids = _run_matcher_condition(
        checkpoint_path,
        real_manifest_path,
        split,
        batch_size=batch_size,
        num_workers=num_workers,
        pp_top_k=pp_top_k,
        cache_gib_per_worker=cache_gib_per_worker,
    )
    cryo_metrics, cryo_rows, cryo_manifest_pdb_ids = _run_matcher_condition(
        checkpoint_path,
        cryoatom2_manifest_path,
        split,
        batch_size=batch_size,
        num_workers=num_workers,
        pp_top_k=pp_top_k,
        cache_gib_per_worker=cache_gib_per_worker,
    )
    preview_axes: dict[str, dict[str, list[str]]] = {}
    for sorting in ("gaussian", "source_probability_mean"):
        preview_curves = preview.get("curves", {}).get(sorting)
        if not isinstance(preview_curves, dict):
            raise ValueError(f"Stage1-only 预览缺少 {sorting} 曲线映射。")
        preview_axes[sorting] = {
            name: _preview_pdb_axis(preview_curves.get(name, {}), f"{sorting}/{name}")
            for name in ("real_receptor", "cryoatom2_receptor", "emap2lig")
        }
    reference_set = set(preview_axes["gaussian"]["real_receptor"])
    if any(
        set(pdb_ids) != reference_set
        for sorting_axes in preview_axes.values()
        for pdb_ids in sorting_axes.values()
    ):
        raise ValueError("Stage1-only 两种排序和三套来源的 179-PDB 集合不一致。")
    if preview["curves"]["gaussian"]["emap2lig"] != preview["curves"][
        "source_probability_mean"
    ]["emap2lig"]:
        raise ValueError("Stage1-only 两种排序必须引用逐字段相同的 Emap2lig 单曲线。")
    if set(real_manifest_pdb_ids) != reference_set:
        raise ValueError("真实受体 test_0 manifest 与 Stage1-only 预览的 PDB 集合不一致。")
    if set(cryo_manifest_pdb_ids) != reference_set:
        raise ValueError("CryoAtom2 test_0 manifest 与 Stage1-only 预览的 PDB 集合不一致。")

    curves: dict[str, Any] = {}
    for sorting in ("gaussian", "source_probability_mean"):
        preview_curves = preview["curves"][sorting]
        curves[sorting] = {
            "real_stage1_only": preview_curves["real_receptor"],
            "real_stage1_matcher": _matcher_curve(
                "Find_1 real-receptor Stage1+Matcher",
                real_rows,
                preview_axes[sorting]["real_receptor"],
                "gaussian_score" if sorting == "gaussian" else sorting,
                checkpoint_path,
            ),
            "cryo_stage1_only": preview_curves["cryoatom2_receptor"],
            "cryo_stage1_matcher": _matcher_curve(
                "Find_1 CryoAtom2-receptor Stage1+Matcher",
                cryo_rows,
                preview_axes[sorting]["cryoatom2_receptor"],
                "gaussian_score" if sorting == "gaussian" else sorting,
                checkpoint_path,
            ),
            "emap2lig": preview_curves["emap2lig"],
        }
    artifact = {
        "schema": "matcher_frozen_test_v1",
        "status": "final_with_matcher",
        "checkpoint_path": str(checkpoint_path),
        "real_manifest_path": str(real_manifest_path),
        "cryoatom2_manifest_path": str(cryoatom2_manifest_path),
        "stage1_preview_path": str(stage1_preview_path),
        "candidate_order": (
            "candidate_selected=true 按指定分数排序后，追加 candidate_selected=false 的同分数排序"
        ),
        "foreground_threshold_source": "BEST checkpoint validation fit",
        "metrics": {
            "real_receptor": real_metrics,
            "cryoatom2_receptor": cryo_metrics,
        },
        "predictions": {
            "real_receptor": real_rows,
            "cryoatom2_receptor": cryo_rows,
        },
        "curves": curves,
    }
    if test1_pdb_list_path is not None:
        test1_pdb_ids = _read_pdb_subset(test1_pdb_list_path)
        if not test1_pdb_ids < reference_set:
            raise ValueError("test_1 的 149 个 PDB 必须是正式 test_0 集合的严格子集。")
        threshold = float(real_metrics["foreground_threshold"])
        artifact["test_1"] = {
            "source": str(test1_pdb_list_path),
            "definition": "reuse test_0 forward and filter by the official 149-PDB list",
            "metrics": {
                "real_receptor": _metrics_from_rows(real_rows, threshold, test1_pdb_ids),
                "cryoatom2_receptor": _metrics_from_rows(cryo_rows, threshold, test1_pdb_ids),
            },
        }
    write_json(Path(output_path), artifact)
    return Path(output_path)


def main(argv: list[str] | None = None) -> int:
    """
    解析 Stage1-only 元数据构建命令.

    - argv: 可选命令行参数; None 表示读取当前进程命令行.
    - return: 元数据成功写入时返回 0, 契约不一致时抛出异常并以非零状态退出.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-test-root", type=Path, required=True)
    parser.add_argument("--cryoatom2-test-root", type=Path, required=True)
    parser.add_argument("--emap2lig-evaluation-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(
        build_stage1_only_preview_metadata(
            arguments.real_test_root,
            arguments.cryoatom2_test_root,
            arguments.emap2lig_evaluation_root,
            arguments.output_path,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
