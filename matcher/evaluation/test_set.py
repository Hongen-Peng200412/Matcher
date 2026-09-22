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


def build_find_stage1_curves(
    stage1_test_root: Path,
    label: str,
) -> dict[str, dict[str, Any]]:
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
        probability = np.asarray(
            [probability_by_blob[int(value)] for value in blob],
            dtype=np.float64,
        )
        gaussian_hits[pdb_root.name.lower()] = _ranked_hit_curve(
            gaussian, blob, selected, candidate_hit
        )
        probability_hits[pdb_root.name.lower()] = _ranked_hit_curve(
            probability, blob, selected, candidate_hit
        )
    return {
        "gaussian": _curve_record(f"{label} Find_1 Stage1-only", gaussian_hits),
        "source_probability_mean": _curve_record(
            f"{label} Find_1 Stage1-only", probability_hits
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
        pdb_hits[path.stem.lower()] = _ranked_hit_curve(
            score,
            blob,
            selected,
            candidate_hit,
        )
    curve = _curve_record(
        "Emap2lig Find Stage1-only (density-only, receptor-independent)",
        pdb_hits,
    )
    return curve


# ====================================================================================================================================
# ====================================================================================================================================


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
    return np.any(
        (overlap >= COVERAGE_THRESHOLD * predicted[:, None])
        & (overlap >= COVERAGE_THRESHOLD * truth[None, :]),
        axis=1,
    )


def selected_priority_order(
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
    segments: list[np.ndarray] = []
    for flag in (True, False):
        indices = np.flatnonzero(selected == flag)
        segments.append(indices[np.lexsort((blob[indices], -score[indices]))])
    return np.concatenate(segments)


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
    order = selected_priority_order(score, blob, selected)
    ranked_hit = hit[order]
    return [bool(np.any(ranked_hit[:k])) for k in TOP_K_VALUES]


def _find_probability_mean(pdb_root: Path) -> dict[int, float]:
    """
    读取 Find_1 F2 blob 的原生来源概率均值.

    - pdb_root: 一个 PDB 的 Find_1 产物目录.
    - return: source_blob_index 到 source_probability_mean 的映射.
    """
    path = pdb_root / "blobs/F2_blobs.npz"
    with np.load(path, allow_pickle=False) as archive:
        blob = np.asarray(archive["blob_index"], dtype=np.int64)
        mean = np.asarray(archive["source_probability_mean"], dtype=np.float64)
    return {int(blob[index]): float(mean[index]) for index in range(len(blob))}


def _curve_record(
    name: str,
    pdb_hits: dict[str, list[bool]],
) -> dict[str, Any]:
    """
    把逐 PDB 命中状态汇总为固定 179 分母的曲线记录.

    - name: 图例使用的完整曲线名称.
    - pdb_hits: PDB 标识到长度 20 布尔命中序列的映射.
    - return: 同时保存命中数、命中比例、PDB 列表和逐 PDB 命中的可复用记录.
    """
    pdb_ids = sorted(pdb_hits)
    counts = [
        sum(int(pdb_hits[pdb_id][index]) for pdb_id in pdb_ids) for index in range(20)
    ]
    return {
        "name": name,
        "pdb_count": EXPECTED_PDB_COUNT,
        "pdb_ids": pdb_ids,
        "top_k": list(TOP_K_VALUES),
        "hit_count": counts,
        "hit_ratio": [count / EXPECTED_PDB_COUNT for count in counts],
        "pdb_hit": {pdb_id: pdb_hits[pdb_id] for pdb_id in pdb_ids},
    }


# ====================================================================================================================================
# ====================================================================================================================================


def build_stage1_only_preview_metadata(
    real_test_root: Path,
    cryoatom2_test_root: Path,
    emap2lig_evaluation_root: Path,
    output_path: Path,
) -> Path:
    """冻结三套来源的 Stage1-only 事实.

    - real_test_root: Find_1 真实受体 held_out_test_0 根目录.
    - cryoatom2_test_root: Find_1 CryoAtom2 受体 held_out_test_0 根目录.
    - emap2lig_evaluation_root: Emap2lig 正式 evaluation 根目录.
    - output_path: Matcher 评估产物中的 JSON 输出路径.
    - return: 已原子写入的 output_path.
    - output.coverage_threshold: float. 固定为 0.30.
    - output.denominator_pdb_count: int. 固定为 179.
    - output.curves.gaussian: dict. 含 real_receptor、cryoatom2_receptor 和唯一 emap2lig 曲线.
    - output.curves.source_probability_mean: dict. 含同三条曲线, 其中 emap2lig 与 Gaussian 记录逐点相同.
    - output.curves.*.*.name: str. 图例使用的完整曲线名称.
    - output.curves.*.*.pdb_count: int. 固定分母 179.
    - output.curves.*.*.pdb_ids: list[str]. 曲线分母中的 179 个有序唯一 PDB 标识.
    - output.curves.*.*.top_k: list[int]. 固定为 1 到 20.
    - output.curves.*.*.hit_count: list[int]. 与 top_k 对齐的 PDB 命中数.
    - output.curves.*.*.hit_ratio: list[float]. hit_count 除以 179.
    - output.curves.*.*.pdb_hit: dict[str, list[bool]]. 每个 PDB 在 k=1..20 的累计命中状态.

    本产物不创建 Matcher 概率、预测 SMILES 或 Stage1+Matcher 曲线. 两种排序共享同一条
    Emap2lig 曲线对象.
    """
    real = build_find_stage1_curves(real_test_root, "Find_1 real-receptor")
    cryo = build_find_stage1_curves(cryoatom2_test_root, "Find_1 CryoAtom2-receptor")
    emap = build_emap2lig_stage1_curve(emap2lig_evaluation_root)
    artifact = {
        "coverage_threshold": COVERAGE_THRESHOLD,
        "denominator_pdb_count": EXPECTED_PDB_COUNT,
        "curves": {
            "gaussian": {
                "real_receptor": real["gaussian"],
                "cryoatom2_receptor": cryo["gaussian"],
                "emap2lig": emap,
            },
            "source_probability_mean": {
                "real_receptor": real["source_probability_mean"],
                "cryoatom2_receptor": cryo["source_probability_mean"],
                "emap2lig": emap,
            },
        },
    }
    write_json(Path(output_path), artifact)
    return Path(output_path)


# ------------------------------------------------ 主要函数 ---------------------------------------------------
def _run_matcher_condition(
    checkpoint_path: Path,
    manifest_path: Path,
    split: str,
    *,
    batch_size: int,
    num_workers: int,
    pp_top_k: int,
    cache_gib_per_worker: float,
) -> tuple[
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    list[str],
]:
    """
    使用训练 release 中的模型代码对一种受体条件的全部 centered 候选(即便 selected=False, 即便超过20)执行冻结推理. rows_by_pdb 会保存全部 centered 候选原始结果, metrics 则只从中汇总 selected=True 的结果.

    输入参数:
        - checkpoint_path: Path; 阶段 E 选定的 BEST checkpoint, 必须含模型配置、模型参数和 validation 拟合的有限前景阈值.
        - manifest_path: Path; 当前受体条件的冻结 manifest.json, 如真实受体或 CryoAtom2 受体的 test_0 manifest.
        - split: str; manifest 中待评估的划分键, 如 test.
        - batch_size: int; 单次 CUDA 前向包含的候选数.
        - num_workers: int; 当前 Dataset 遍历使用的工作进程数.
        - pp_top_k: int; 每个候选按 Stage1 概率稳定保留的 PP token 数量上限, 如 2048.
        - cache_gib_per_worker: float; 每个 Dataset 副本的 PDB 载荷最近使用缓存预算, 单位 GiB; 0 表示不缓存.

    返回值:
        - metrics: dict[str, Any]; 只统计 candidate_selected=True 候选的测试指标, 使用 checkpoint 保存的前景阈值且不重新拟合, 不保留 validation 选模分数 selection.
            - metrics.candidate_count: int; 进入指标计算的 selected 候选数.
            - metrics.pdb_count: int; manifest summary 冻结的 PDB 分母, 当前正式 test_0 固定为 179, 包含零 centered 候选的 PDB.
            - metrics.foreground_prauc: float; 五类真实配体合并为前景后的 average precision, 没有真实前景时为 NaN.
            - metrics.foreground_threshold: float; checkpoint 中由 validation 拟合并在本次测试直接应用的前景概率阈值.
            - metrics.foreground: dict[str, int | float]; 固定阈值上的前景二分类混淆计数与 F1.
            - metrics.foreground.true_positive: int; 真实前景且前景概率达到阈值的 selected 候选数.
            - metrics.foreground.false_positive: int; 真实背景但前景概率达到阈值的 selected 候选数.
            - metrics.foreground.false_negative: int; 真实前景但前景概率低于阈值的 selected 候选数.
            - metrics.foreground.true_negative: int; 真实背景且前景概率低于阈值的 selected 候选数.
            - metrics.foreground.f1: float; 前景二分类 F1, 分母为零时为 NaN.

            - metrics.smiles_match: dict[str, dict[str, int | float | None]]; 键为 overall、ion、nucleotide_like、peptide_like、small_molecule 和 sugar, 每组只统计 identity_supported=True 的真实前景 selected 候选.
            - metrics.smiles_match.<group>.correct_count: int; 当前组中 predicted_smiles == target_smiles 的候选数.
            - metrics.smiles_match.<group>.eligible_count: int; 当前组中进入 SMILES 准确率分母的受支持真前景候选数.
            - metrics.smiles_match.<group>.accuracy: float | None; correct_count / eligible_count, 当前组没有合格候选时为 None.

            - metrics.unsupported_identity_target_candidate_count: int; 真实前景但目标身份缺少合法 SMILES 图或 SMI-TED 表征的 selected 候选数.
            - metrics.unsupported_identity_targets: list[dict[str, Any]]; 上述不受支持候选的定位记录列表.
            - metrics.unsupported_identity_targets[*].pdb_id: str; 候选所属的小写 PDB 标识, 如 1abc.
            - metrics.unsupported_identity_targets[*].centered_box_index: int; 候选在当前 PDB centered 候选轴上的编号.
            - metrics.unsupported_identity_targets[*].source_blob_index: int; 候选对应的原始 Stage1 blob 编号.
            - metrics.unsupported_identity_targets[*].matched_occurrence_id: int; 与候选双向覆盖率达到阈值的真实 occurrence 编号.
            - metrics.unsupported_identity_targets[*].target_smiles: str | None; 最大覆盖 occurrence 的精确 PreparedLigand SMILES, 来源缺失时可为 None.

        - rows_by_pdb: dict[str, list[dict[str, Any]]]; PDB 标识到全部 centered 候选预测列表的映射, 同一次前向结果供 Gaussian 和来源概率均值两种排序复用; 零候选 PDB 不建立空列表.
            - rows_by_pdb.<pdb_id>[*].centered_box_index: int; 候选在当前 PDB centered 候选轴上的编号.
            - rows_by_pdb.<pdb_id>[*].source_blob_index: int; 候选对应的原始 Stage1 blob 编号.
            - rows_by_pdb.<pdb_id>[*].gaussian_score: float; Stage1 Gaussian 候选分数.
            - rows_by_pdb.<pdb_id>[*].source_probability_mean: float; 原始 Stage1 blob 内来源概率的算术平均值.
            - rows_by_pdb.<pdb_id>[*].candidate_selected: bool; True 表示候选属于 Stage1 原 selected 集合并进入 metrics 指标.
            - rows_by_pdb.<pdb_id>[*].foreground_target: bool; True 表示候选与一个真实 occurrence 的双向覆盖率均达到 manifest 阈值.
            - rows_by_pdb.<pdb_id>[*].target_category: str | None; 真实前景类别, 如 small_molecule; 背景为 None.
            - rows_by_pdb.<pdb_id>[*].matched_occurrence_id: int | None; 最大覆盖真实 occurrence 编号, 背景为 None.
            - rows_by_pdb.<pdb_id>[*].target_smiles: str | None; 最大覆盖 occurrence 的精确 PreparedLigand SMILES, 背景或来源缺失时为 None.
            - rows_by_pdb.<pdb_id>[*].identity_supported: bool; True 表示目标身份同时具有合法 SMILES 图和 SMI-TED 表征.
            - rows_by_pdb.<pdb_id>[*].predicted_smiles: str | None; 当前 PDB 闭集身份 logits 最大槽对应的精确 SMILES, 身份轴为空时为 None.
            - rows_by_pdb.<pdb_id>[*].foreground_probability: float; 前景 logit 经 sigmoid 得到的概率, 数值位于 [0, 1].

        - manifest_pdb_ids: list[str]; manifest summary 冻结的 179 个唯一小写 PDB 标识, 保留原有顺序并包含零 centered 候选的 PDB.

    本函数要求 CUDA, 在 torch.no_grad() 和 bfloat16 autocast 下运行模型, 只返回内存对象且不写入测试产物.
    """
    import torch

    from ..data.manifest import load_manifest
    from ..datasets.batching import ShuffledBatchSampler, build_candidate_loader
    from ..datasets.candidate_data import load_manifest_dataset
    from .checkpoint import load_matcher_from_training_release

    device = torch.device("cuda")
    model, checkpoint, config = load_matcher_from_training_release(
        checkpoint_path,
        device,
    )
    threshold = float(checkpoint["foreground_threshold"])
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
    rows_by_pdb: dict[str, list[dict[str, Any]]] = {}
    with torch.no_grad():
        for cpu_batch in loader:
            batch = {
                key: (
                    value.to(device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in cpu_batch.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch)  # type: ignore[arg-type]
            # 形状符号: B 是当前物理批次的候选数, I_max 是当前批次任一候选所属 PDB 的最大有效身份数.
            # float32, (B,), 每个候选的前景 logit 经 sigmoid 转成 [0, 1] 概率并迁回 CPU.
            foreground_probability = (
                torch.sigmoid(output["foreground_logit"]).float().cpu().numpy()
            )
            # 长度 B 的 int 列表, 沿 (B, I_max) 身份轴取最大 logit; 有效身份非空时第 b 项索引 cpu_batch["identity_smiles"][b], 整批身份轴为空时用 -1 表示无法预测.
            predicted_index = (
                output["identity_logits"].argmax(dim=1).cpu().tolist()
                if output["identity_logits"].shape[1]
                else [-1] * len(cpu_batch["metadata"])
            )
            # metadata 外层长度为 B, index 同时索引批次张量第一维、identity_smiles 和 foreground_probability.
            for index, metadata in enumerate(cpu_batch["metadata"]):
                # 字符串列表, 当前候选所属 PDB 按字符串升序排列的有效精确 SMILES 身份轴; 槽位与 identity_logits[index, :len(smiles_axis)] 逐项对齐.
                smiles_axis = cpu_batch["identity_smiles"][index]
                # int 标量, 当前候选在 smiles_axis 中的预测身份槽; -1 表示当前批次没有任何有效身份槽.
                slot = int(predicted_index[index])
                # str 或 None, 预测槽对应的精确 PreparedLigand SMILES; 空身份轴或越界保护分支返回 None.
                predicted_smiles = (
                    smiles_axis[slot] if 0 <= slot < len(smiles_axis) else None
                )
                # 普通字典, 同时保存一个 centered 候选的 Stage1 排序字段、真值身份字段和 Matcher 预测字段.
                row = {
                    # int 标量, 当前候选在所属 PDB centered 候选轴上的编号, 不等于当前物理批次的 index.
                    "centered_box_index": metadata["centered_box_index"],
                    # int 标量, 当前 centered 候选对应的原始 Stage1 blob 编号.
                    "source_blob_index": metadata["source_blob_index"],
                    # float 标量, 当前候选用于 Gaussian 曲线排序的 Stage1 分数.
                    "gaussian_score": metadata["gaussian_score"],
                    # float 标量, 原始 Stage1 blob 内来源概率的算术平均值, 用于另一条曲线排序.
                    "source_probability_mean": metadata["source_probability_mean"],
                    # bool 标量, True 表示候选属于 Stage1 原 selected 集合并进入正式测试指标.
                    "candidate_selected": metadata["candidate_selected"],
                    # bool 标量, True 表示候选与一个真实 occurrence 的双向覆盖率均达到 manifest 阈值.
                    "foreground_target": metadata["foreground_target"],
                    # str 或 None, 真实前景的五类配体类别, 如 small_molecule; 背景为 None.
                    "target_category": metadata["target_category"],
                    # int 或 None, 最大覆盖真实 occurrence 的 PDB 内编号; 背景为 None.
                    "matched_occurrence_id": metadata["matched_occurrence_id"],
                    # str 或 None, 最大覆盖 occurrence 的精确 PreparedLigand SMILES; 背景或来源缺失时为 None.
                    "target_smiles": metadata["target_smiles"],
                    # bool 标量, True 表示 target_smiles 存在于当前 PDB 的有效身份轴, 即对应身份同时具有合法 SMILES 图和 SMI-TED 表征.
                    "identity_supported": metadata["identity_supported"],
                    # str 或 None, identity_logits 最大有效槽对应的预测 SMILES.
                    "predicted_smiles": predicted_smiles,
                    # float 标量, 当前候选的前景概率, 与 foreground_probability[index] 数值一致.
                    "foreground_probability": float(foreground_probability[index]),
                }
                # 按小写 PDB 标识聚合全部 centered 候选, 包括 candidate_selected=False 的候选.
                rows_by_pdb.setdefault(metadata["pdb_id"], []).append(row)
    manifest = load_manifest(manifest_path)
    manifest_pdb_ids = [
        str(pdb_id).lower() for pdb_id in manifest["summary"][split]["pdb_ids"]
    ]
    expected_pdb_ids = set(manifest_pdb_ids)
    metrics = _metrics_from_rows(rows_by_pdb, threshold, expected_pdb_ids)
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
            accumulator.foreground_probability.append(
                float(row["foreground_probability"])
            )
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

    - path: 官方 test_1 JSON; 顶层 pdb_ids 按 test_0 顺序保存 PDB 标识.
    - return: 统一小写后的 149 个唯一 PDB 标识.
    """
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(value).lower() for value in record["pdb_ids"]}


def _matcher_curve(
    name: str,
    rows_by_pdb: dict[str, list[dict[str, Any]]],
    pdb_ids: list[str],
    sorting: str,
) -> dict[str, Any]:
    """
    按冻结 Stage1 顺序字段汇总 Stage1+Matcher top-k 曲线.

    - name: 图例使用的完整曲线名称.
    - rows_by_pdb: 同一次 Matcher 冻结推理的候选预测.
    - pdb_ids: Stage1-only 元数据给出的有序 179-PDB 列表.
    - sorting: gaussian_score 或 source_probability_mean; 只改变 selected 段和追加段各自的内部顺序.
    - return: 固定 179 分母的 k=1..20 曲线; 零候选 PDB 的 20 个命中值均为 False.

    每个 PDB 先排列全部 selected 候选, 再追加全部 unselected 候选. 不受支持身份、背景候选或
    预测 SMILES 不等于确定目标 SMILES 的候选均不能成为命中; 它们保留原排名, 不删除后补位.
    """
    pdb_hit: dict[str, list[bool]] = {}
    for pdb_id in pdb_ids:
        rows = rows_by_pdb.get(pdb_id, [])
        score = np.asarray([float(row[sorting]) for row in rows], dtype=np.float64)
        blob = np.asarray(
            [int(row["source_blob_index"]) for row in rows], dtype=np.int64
        )
        selected = np.asarray(
            [bool(row["candidate_selected"]) for row in rows], dtype=np.bool_
        )
        order = selected_priority_order(score, blob, selected)
        ordered = [rows[int(index)] for index in order]
        correct = [
            bool(row["foreground_target"])
            and bool(row["identity_supported"])
            and row["predicted_smiles"] == row["target_smiles"]
            for row in ordered
        ]
        pdb_hit[pdb_id] = [any(correct[:k]) for k in TOP_K_VALUES]
    return _curve_record(name, pdb_hit)


# ----------------------- 最终运行包装器 ------------------------
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

    输入:
        - checkpoint_path: 四项训练完成后选定的真实 BEST checkpoint.
        - real_manifest_path / cryoatom2_manifest_path: 两种受体条件的冻结 test_0 manifest.
        - split: 两份 manifest 共同的测试划分键.
        - stage1_preview_path: 本轮已验收的 Stage1-only 元数据.
        - output_path: 最终冻结测试 JSON 路径.
        - test1_pdb_list_path: 可选官方 149-PDB 列表; 只筛选 test_0 预测, 不重新前向.
        - batch_size / num_workers / pp_top_k / cache_gib_per_worker: 两种受体条件共用的推理参数.

    写入值:
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
        - output.test_1: 可选 dict. 仅复用 test_0 预测筛选官方 149-PDB 子集, 保存两种受体指标.

    本函数分别对真实受体和 CryoAtom2 运行一次 Matcher; 每次产生的候选预测同时支持
    Gaussian 与来源概率均值排序, 不为第二种排序重复前向.
    """
    preview = json.loads(Path(stage1_preview_path).read_text(encoding="utf-8"))
    (
        real_metrics,
        real_rows,
        _,
    ) = _run_matcher_condition(
        checkpoint_path,
        real_manifest_path,
        split,
        batch_size=batch_size,
        num_workers=num_workers,
        pp_top_k=pp_top_k,
        cache_gib_per_worker=cache_gib_per_worker,
    )
    (
        cryo_metrics,
        cryo_rows,
        _,
    ) = _run_matcher_condition(
        checkpoint_path,
        cryoatom2_manifest_path,
        split,
        batch_size=batch_size,
        num_workers=num_workers,
        pp_top_k=pp_top_k,
        cache_gib_per_worker=cache_gib_per_worker,
    )
    curves: dict[str, Any] = {}
    for sorting in ("gaussian", "source_probability_mean"):
        preview_curves = preview["curves"][sorting]
        curves[sorting] = {
            "real_stage1_only": preview_curves["real_receptor"],
            "real_stage1_matcher": _matcher_curve(
                "Find_1 real-receptor Stage1+Matcher",
                real_rows,
                [str(pdb_id) for pdb_id in preview_curves["real_receptor"]["pdb_ids"]],
                "gaussian_score" if sorting == "gaussian" else sorting,
            ),
            "cryo_stage1_only": preview_curves["cryoatom2_receptor"],
            "cryo_stage1_matcher": _matcher_curve(
                "Find_1 CryoAtom2-receptor Stage1+Matcher",
                cryo_rows,
                [
                    str(pdb_id)
                    for pdb_id in preview_curves["cryoatom2_receptor"]["pdb_ids"]
                ],
                "gaussian_score" if sorting == "gaussian" else sorting,
            ),
            "emap2lig": preview_curves["emap2lig"],
        }
    artifact = {
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
        threshold = float(real_metrics["foreground_threshold"])
        artifact["test_1"] = {
            "metrics": {
                "real_receptor": _metrics_from_rows(
                    real_rows, threshold, test1_pdb_ids
                ),
                "cryoatom2_receptor": _metrics_from_rows(
                    cryo_rows, threshold, test1_pdb_ids
                ),
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
