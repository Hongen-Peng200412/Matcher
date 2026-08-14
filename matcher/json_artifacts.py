"""把训练与实验记录写成严格 UTF-8 JSON 或 JSONL。

``write_json`` 先同步临时文件再原子发布完整 JSON；``append_jsonl`` 同步追加一条记录。
两条入口都会把内存中的 NaN/Inf 转为 JSON ``null``，并拒绝非字符串映射键。
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Mapping[str, object]) -> None:
    """把结构化记录写到临时文件，同步后原子替换目标路径。

    内存指标用 NaN 表示零支持分母；严格 JSON 不允许 NaN/Inf，因此落盘边界
    递归转换为 ``null``。元组转换为 JSON 数组，其余值保持原有 JSON 类型。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    """追加并同步一条严格 JSON 记录；非有限浮点数按 ``write_json`` 规则写为 null。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _json_safe(value: Any) -> Any:
    """递归转换非有限浮点数和元组，使返回值可由标准 JSON 编码器接受。"""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON 记录的所有映射键必须是字符串。")
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
