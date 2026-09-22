"""读取小分子测试配置, 并写出 JSON 与 JSONL 科学产物.

本模块只处理稳定文件格式. 候选目标、排序、命中和模型推理由同目录中的
build_manifest、ranking、evaluate_stage1 和 evaluate_model 定义.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """读取小分子测试 YAML.

    输入参数:
        - path: Path, YAML 文件路径.

    返回值:
        - config: dict[str, Any], 含 sources、outputs、runtime 和 models 四组配置.
    """
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def read_json(path: Path) -> dict[str, Any]:
    """读取顶层为对象映射的 UTF-8 JSON 文件."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """按文件顺序读取 UTF-8 JSONL 中的非空记录."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> Path:
    """把一个对象映射写成 UTF-8 JSON 文件."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> Path:
    """按输入顺序写出 UTF-8 JSONL 记录."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
            )
            handle.write("\n")
    return path
