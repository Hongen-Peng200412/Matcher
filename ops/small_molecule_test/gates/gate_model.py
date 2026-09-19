from __future__ import annotations

import argparse
from pathlib import Path

from ops.small_molecule_test.artifacts import load_config, read_json, read_jsonl
from ops.small_molecule_test.evaluate_model import _evaluate_condition


def gate_model(config_path: Path, model_name: str) -> None:
    """在实际 A800 上门禁一个冻结 BEST 的训练快照身份前向.

    输入参数:
        - config_path: Path, 正式评测使用的 test_smallmolecule YAML.
        - model_name: str, density_blob_only 或 strongest.

    返回值:
        - None: 一个真实 Stage3 候选成功产生两档身份答案后打印摘要.
    """
    import torch

    from matcher.evaluation.checkpoint import load_matcher_from_training_release

    config = load_config(config_path)
    gpu_name = torch.cuda.get_device_name(0)
    if "A800" not in gpu_name:
        raise RuntimeError(f"门禁要求 A800, 实际为 {gpu_name}.")
    model_record = config["models"][model_name]
    checkpoint_path = Path(model_record["checkpoint"])
    model, _, model_config, source = load_matcher_from_training_release(
        checkpoint_path, torch.device("cuda")
    )
    if model_config.variant != model_name:
        raise RuntimeError("checkpoint 变体与门禁模型名称不一致.")
    manifest_path = Path(config["outputs"]["manifest_root"]) / "manifest.json"
    manifest = read_json(manifest_path)
    root = manifest_path.parent
    identities = read_jsonl(root / manifest["files"]["identities"])
    small_molecule_by_identity = {
        (record["pdb_id"], record["smiles"]): bool(
            record["allowed_as_small_molecule_answer"]
        )
        for record in identities
    }
    candidates = read_jsonl(root / manifest["files"]["real_candidates"])
    first = next(
        record for record in candidates if record["target_scope"] == "stage3_foreground"
    )
    predictions = _evaluate_condition(
        model,
        model_config,
        Path(config["sources"]["real_manifest"]),
        str(config["runtime"]["split"]),
        small_molecule_by_identity,
        [first],
        {**config["runtime"], "num_workers": 0},
        torch.device("cuda"),
    )
    prediction = predictions[0]
    if len(prediction["identity_logits"]) != len(prediction["identity_smiles"]):
        raise RuntimeError("原始身份 logits 与完整 PDB 身份轴未对齐.")
    if "foreground_probability" in prediction or "foreground_accepted" in prediction:
        raise RuntimeError("小分子正式身份决策不得保存或使用前景阈值结果.")
    print(
        {
            "model": model_name,
            "gpu": gpu_name,
            "training_snapshot": source,
            "pdb_id": prediction["pdb_id"],
            "identity_count": len(prediction["identity_smiles"]),
            "all_ligand": prediction["predicted_smiles_all_ligand"],
            "small_molecule": prediction["predicted_smiles_small_molecule"],
        }
    )


def main() -> int:
    """解析配置与模型名称并执行一次真实 GPU 门禁."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=("density_blob_only", "strongest"), required=True)
    arguments = parser.parse_args()
    gate_model(arguments.config, arguments.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
