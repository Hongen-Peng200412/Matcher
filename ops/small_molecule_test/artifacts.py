"""读取 test_smallmolecule 配置并原子写出 JSON、JSONL 产物.

主要入口是 load_config、read_jsonl、write_json 和 write_jsonl.本模块只处理
文件格式与 SHA-256 留证, 不定义候选排名、命中或身份选择等科学规则.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

def sha256_file(path: Path) -> str:
    """计算一个现存文件的 SHA-256 十六进制摘要.

    输入参数:
        - path: Path, 需要绑定内容的现存普通文件.

    返回值:
        - sha256: str, 64 个小写十六进制字符组成的文件内容摘要.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    """读取小分子评测 YAML, 并要求所有顶层职责组显式存在.

    输入参数:
        - path: Path, test_smallmolecule.yaml 的现存路径.

    返回值:
        - config: dict[str, Any], 含 schema、sources、outputs、runtime、models 和 expected 六组显式配置.
    """
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"schema", "sources", "outputs", "runtime", "models", "expected"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"配置顶层字段必须精确为 {sorted(required)}.")
    if config["schema"] != "matcher_test_smallmolecule_config_v1":
        raise ValueError("配置 schema 不是 matcher_test_smallmolecule_config_v1.")
    return config


def read_json(path: Path) -> dict[str, Any]:
    """读取顶层为对象映射的 UTF-8 JSON 文件.

    输入参数:
        - path: Path, 需要读取的现存 JSON 文件.

    返回值:
        - value: dict[str, Any], JSON 顶层对象; 其它顶层类型失败关闭.
    """
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON 顶层必须是对象映射.")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """按文件顺序读取 UTF-8 JSONL 中的对象映射.

    输入参数:
        - path: Path, 需要读取的现存 JSONL 文件.

    返回值:
        - records: list[dict[str, Any]], 跳过空白行后按文件顺序保存的对象记录.
    """
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL 记录必须是对象映射.")
            records.append(value)
    return records


def write_json(path: Path, value: dict[str, Any]) -> Path:
    """原子写出一个可复算的 UTF-8 JSON 对象.

    输入参数:
        - path: Path, 目标 JSON 文件; 父目录不存在时创建.
        - value: dict[str, Any], 需要保存的完整顶层对象.

    返回值:
        - path: Path, 已发布的目标文件路径.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)
    return path


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> Path:
    """原子写出顺序有科学含义的 UTF-8 JSONL 记录序列.

    输入参数:
        - path: Path, 目标 JSONL 文件; 父目录不存在时创建.
        - records: iterable[dict[str, Any]], 按冻结顺序保存的对象映射.

    返回值:
        - path: Path, 已发布的目标文件路径.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False))
            handle.write("\n")
    os.replace(temporary, path)
    return path
