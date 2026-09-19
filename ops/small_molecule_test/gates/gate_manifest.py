from __future__ import annotations

import argparse
from pathlib import Path

from ops.small_molecule_test.artifacts import load_config, read_json, read_jsonl, sha256_file


def gate_manifest(config_path: Path) -> None:
    """独立复核正式 manifest 文件身份与 9rd8 特例.

    输入参数:
        - config_path: Path, 与正式构建入口相同的 test_smallmolecule YAML.

    返回值:
        - None: 全部门禁通过后打印摘要; 任一内容身份或科学特例不符时失败关闭.
    """
    config = load_config(config_path)
    manifest_path = Path(config["outputs"]["manifest_root"]) / "manifest.json"
    manifest = read_json(manifest_path)
    root = manifest_path.parent
    for key, relative_path in manifest["files"].items():
        path = root / relative_path
        if sha256_file(path) != manifest["file_sha256"][key]:
            raise RuntimeError(f"{key}: 文件内容与 manifest SHA-256 不一致.")
    real = read_jsonl(root / manifest["files"]["real_candidates"])
    special = [
        record
        for record in real
        if record["pdb_id"] == "9rd8" and int(record["source_blob_index"]) == 0
    ]
    if len(special) != 1 or special[0]["target_scope"] != "non_stage3_foreground":
        raise RuntimeError("9rd8/source_blob_index=0 没有稳定保留为非 Stage3 前景.")
    print(
        {
            "manifest": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "summary": manifest["summary"],
            "special_case": special[0]["target_scope"],
        }
    )


def main() -> int:
    """解析唯一配置路径并执行 manifest 门禁."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    gate_manifest(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
