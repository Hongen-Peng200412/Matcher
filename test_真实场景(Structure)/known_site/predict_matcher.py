"""在冻结的真实配体质心上执行两个既有 Matcher 密度模型的身份前向。

每个查询点独立保存预测。模型、密度构造、图读取与批处理均来自该模型训练时的
release 快照；本文件只把给定世界坐标裁成模型已经使用的 48³ 输入。
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from matcher.evaluation.checkpoint import load_matcher_from_training_release


def read_jsonl(path: Path) -> list[dict]:
    """读取一份冻结的逐行 JSON 清单。

    - path: 待读取的 PDB、查询点或身份清单路径。
    - return: 文件顺序不变的记录列表。
    """
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def crop_density(grid: np.ndarray, start_zyx: np.ndarray) -> np.ndarray:
    """按全局 ZYX 起点提取 48³，超出完整密度网格处补零。

    - grid: float32，(Z,Y,X)，AdaLigand 的重采样完整密度。
    - start_zyx: int64，(3,)，裁块首体素在完整网格的下标。
    - return: float32，(48,48,48)，不更改给定质心。
    """
    cropped = np.zeros((48, 48, 48), dtype=np.float32)
    low = np.maximum(start_zyx, 0)
    high = np.minimum(start_zyx + 48, grid.shape)
    if np.all(high > low):
        target_low = low - start_zyx
        target_high = target_low + high - low
        cropped[
            target_low[0]:target_high[0],
            target_low[1]:target_high[1],
            target_low[2]:target_high[2],
        ] = grid[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    return cropped


def candidate_item(
    query: dict,
    identity_smiles: list[str],
    embeddings: dict[str, np.ndarray],
    graph_store: object,
    voxel_input: np.ndarray,
) -> dict:
    """把一个已知位置和当前 PDB 的受支持身份轴交给训练快照的批处理器。

    - query: 当前任务清单的一条 occurrence，含 PDB、编号、类别和真值 SMILES。
    - identity_smiles: 当前任务该 PDB 内可由模型编码的精确 SMILES，字典序排列。
    - embeddings: 精确 SMILES 到原训练 SMI-TED float32 (768,) 向量的映射。
    - graph_store: 训练快照 SmilesGraphStore，已完整载入图包。
    - voxel_input: float16，(1,48,48,48) 或 (106,48,48,48)。
    - return: 单样本 CandidateItem；禁用的 token、区域目标和人工统计均为空或零。
    """
    zeros = np.zeros((48, 48, 48), dtype=np.bool_)
    empty_index = np.empty((0, 3), dtype=np.int64)
    target = query["target_smiles"]
    return {
        "metadata": {
            "split": "held_out_test_0",
            "pdb_id": query["pdb_id"],
            "centered_box_index": -1,
            "source_blob_index": -1,
            "candidate_selected": True,
            "gaussian_score": 0.0,
            "source_probability_mean": 0.0,
            "foreground_target": True,
            "target_category": query["type_tag"],
            "matched_occurrence_id": query["occurrence_id"],
            "target_smiles": target,
            "identity_supported": target in identity_smiles,
        },
        "voxel_input": voxel_input,
        "a_tokens": np.empty((0, 230), dtype=np.float32),
        "p_tokens": np.empty((0, 116), dtype=np.float32),
        "v_tokens": np.empty((0, 68), dtype=np.float32),
        "v_index_zyx": empty_index,
        "pp_tokens": np.empty((0, 4), dtype=np.float32),
        "pp_index_zyx": empty_index.copy(),
        "foreground_target": True,
        "ligand_area_target": zeros,
        "binding_area_target": zeros.copy(),
        "full_candidate_voxel_count": 0,
        "candidate_manual_features": np.zeros(12, dtype=np.float32),
        "candidate_sampling_features": np.zeros(4, dtype=np.float32),
        "identity_smiles": identity_smiles,
        "language_embeddings": np.stack([embeddings[smiles] for smiles in identity_smiles]),
        "smiles_graphs": [graph_store.get(smiles) for smiles in identity_smiles],
        "identity_target": identity_smiles.index(target) if target in identity_smiles else -1,
    }


def main() -> None:
    """读取冻结清单与训练快照，在 NxU、nxu 各点分别落盘前向结果。

    - --manifest-root: build_manifest.py 生成的两项任务清单。
    - --identity-manifest: 可选的全类别 PDB 身份轴文件；不改变 --manifest-root 的查询点。
    - --source-manifest: 已冻结 held_out_test_0 身份与语言向量清单。
    - --checkpoint: 当前模型的 BEST.pt；run.json 定位训练 release。
    - --model-name: 稳定实验标识，仅用来隔离输出目录。
    - --output-root: 三方法共享的正式测试产物根。
    - --data-root: AdaLigand 原始 Ori_Data；读取实验密度与身份语言资产。
    - --receptor-root: 106 通道所用模拟密度和受体 token 的 Ori_Data；省略时使用 --data-root。
    - --task-names: 本次要前向的冻结任务；CryoAtom2 小分子运行只选择 nxu。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--identity-manifest", type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--receptor-root", type=Path)
    parser.add_argument("--task-names", nargs="+", choices=("NxU", "nxu"), default=("NxU", "nxu"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    receptor_root = args.receptor_root if args.receptor_root is not None else args.data_root

    device = torch.device("cuda")
    model, _, config = load_matcher_from_training_release(args.checkpoint, device)
    package = model.__module__.split(".model.")[0]
    density = importlib.import_module(f"{package}.data.density")
    batching = importlib.import_module(f"{package}.datasets.batching")
    candidate_data = importlib.import_module(f"{package}.datasets.candidate_data")
    graphs = importlib.import_module(f"{package}.data.smiles_graph")
    if config.variant not in ("density_only", "density_only_106"):
        raise ValueError(f"此已知位点入口不支持 {config.variant}")

    source = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    identity_path = args.source_manifest.parent / source["splits"]["held_out_test_0"]["identities"]
    identity_assets = {
        (record["pdb_id"], record["smiles"]): args.data_root / record["language_relative_path"]
        for record in read_jsonl(identity_path)
    }
    graph_store = graphs.SmilesGraphStore(Path(source["graph_asset_path"]))
    graph_store.initialize()

    # 每个任务独立定义 PDB 内候选集合；同一 PDB 的 NxU 与 nxu 不共用身份前向。
    for task_name in args.task_names:
        task_root = args.manifest_root / task_name
        identity_path = (args.identity_manifest if args.identity_manifest is not None
                         else task_root / "pdbs.jsonl")
        pdbs = {record["pdb_id"]: record for record in read_jsonl(identity_path)}
        queries_by_pdb: dict[str, list[dict]] = {}
        for query in read_jsonl(task_root / "queries.jsonl"):
            queries_by_pdb.setdefault(query["pdb_id"], []).append(query)
        for pdb_id, queries in queries_by_pdb.items():
            output_dir = args.output_root / args.model_name / task_name / pdb_id
            output_dir.mkdir(parents=True, exist_ok=True)
            small_smiles = set(pdbs[pdb_id]["small_molecule_identity_smiles"]) if args.identity_manifest else None
            identity_smiles = [
                smiles for smiles in pdbs[pdb_id]["identity_smiles"]
                if (pdb_id, smiles) in identity_assets and smiles in graph_store
            ]
            if not identity_smiles:
                for query in queries:
                    result = {"pdb_id": pdb_id, "occurrence_id": query["occurrence_id"],
                              "status": "no_supported_identity", "predicted_smiles": None,
                              "scored_smiles": []}
                    if small_smiles is not None:
                        result["predicted_small_molecule_smiles"] = None
                        result["identity_logits"] = []
                    (output_dir / f"{query['occurrence_id']}.json").write_text(
                        json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8"
                    )
                continue
            embeddings = {}
            for smiles in identity_smiles:
                with np.load(identity_assets[(pdb_id, smiles)], allow_pickle=False) as asset:
                    embeddings[smiles] = np.asarray(asset["embedding"], dtype=np.float32).copy()
            density_dir = args.data_root / "density" / pdb_id
            experimental = np.load(density_dir / "exp.npy", mmap_mode="r")[0]
            with np.load(density_dir / "exp.npz", allow_pickle=False) as metadata:
                origin = np.asarray(metadata["origin"], dtype=np.float64)
                voxel_size = np.asarray(metadata["voxel_size"], dtype=np.float64)
            if config.variant == "density_only_106":
                simulated = np.load(receptor_root / "density" / pdb_id / "sim.npy", mmap_mode="r")[0]
                with np.load(receptor_root / "parse" / pdb_id / "receptor_tokens.npz", allow_pickle=False) as receptor:
                    receptor_coords = np.asarray(receptor["coords"], dtype=np.float32)
                    receptor_features = np.concatenate(
                        (np.asarray(receptor["feat"], dtype=np.float32),
                         np.asarray(receptor["is_backbone"], dtype=np.float32)[:, None]), axis=1
                    )
            for batch_start in range(0, len(queries), args.batch_size):
                selected_queries = queries[batch_start:batch_start + args.batch_size]
                items = []
                for query in selected_queries:
                    # 世界 XYZ 到全局 ZYX 下标；首体素为查询所在体素减去局部半边长 24。
                    center = np.asarray(query["center_xyz_angstrom"], dtype=np.float64)
                    start_zyx = np.floor((center - origin) / voxel_size).astype(np.int64)[::-1] - 24
                    crop_exp = crop_density(experimental, start_zyx)
                    if config.variant == "density_only":
                        voxel_input = density.build_experimental_density_channel(crop_exp).astype(np.float16)
                    else:
                        crop_sim = crop_density(simulated, start_zyx)
                        crop_origin = (origin + start_zyx[::-1] * voxel_size).astype(np.float32)
                        receptor_grid, hardmask = candidate_data._scatter_receptor(
                            receptor_coords, receptor_features, crop_origin,
                            voxel_size.astype(np.float32), np.zeros(3, dtype=np.int64)
                        )
                        bank = density.build_density_channels(crop_exp, crop_sim, hardmask)
                        voxel_input = np.concatenate((bank, receptor_grid), axis=0).astype(np.float16)
                    items.append(candidate_item(query, identity_smiles, embeddings, graph_store, voxel_input))
                cpu_batch = batching.collate_candidates(items)
                gpu_batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in cpu_batch.items()
                }
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(gpu_batch)
                logits = output["identity_logits"].float().cpu().numpy()
                for index, query in enumerate(selected_queries):
                    scores = logits[index, :len(identity_smiles)]
                    result = {
                        "pdb_id": pdb_id,
                        "occurrence_id": query["occurrence_id"],
                        "status": "scored",
                        "predicted_smiles": identity_smiles[int(np.argmax(scores))],
                        "scored_smiles": identity_smiles,
                        "identity_logits": scores.tolist(),
                    }
                    if small_smiles is not None:
                        # 掩码只在已经计算的完整身份 logits 上读答案，不再次调用模型。
                        small_indices = [slot for slot, smiles in enumerate(identity_smiles)
                                         if smiles in small_smiles]
                        result["predicted_small_molecule_smiles"] = (
                            identity_smiles[max(small_indices, key=lambda slot: scores[slot])]
                            if small_indices else None
                        )
                    (output_dir / f"{query['occurrence_id']}.json").write_text(
                        json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8"
                    )


if __name__ == "__main__":
    main()
