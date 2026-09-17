"""原子保存并最小验证当前 Matcher 的 BEST/LAST checkpoint.

主要入口是 save_checkpoint 与 load_checkpoint. 保存入口在目标同目录写临时文件并原子
替换, 加载入口只接受当前 schema_version 且要求 model_state 映射存在. 本模块不迁移历史
checkpoint, 也不计算内容摘要.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, cast

import torch


CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointEnvelope(TypedDict):
    """声明加载器保证存在的两个 checkpoint 字段.

    - schema_version: 当前必须为 1.
    - model_state: 参数或缓冲区名称到张量的映射; 设备由 map_location 决定.
"""

    schema_version: int
    model_state: dict[str, torch.Tensor]


def save_checkpoint(path: Path, state: Mapping[str, object]) -> None:
    """原子保存一个 checkpoint.

    - path: BEST.pt 或 LAST.pt 目标路径.
    - state: 可由 torch.save 序列化的 checkpoint 字段映射.
    - return: None; 先写同目录进程专属临时文件、刷新并 fsync, 再原子替换目标.

    finally 始终删除遗留临时文件; 现有目标只会在完整临时文件写入成功后被替换.
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path, map_location: str | torch.device = "cpu"
) -> CheckpointEnvelope:
    """加载受信任 checkpoint, 并验证 schema_version 与 model_state.

    - path: 受信任的本项目 checkpoint 路径.
    - map_location: torch.load 使用的目标设备, 默认先加载到 CPU.
    - return: 至少含当前 schema_version 和字符串键张量 model_state 的映射.

    训练状态、BEST validation、前景阈值和来源绑定等附加字段保持原值交给调用方验证.
"""

    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(state, dict) or state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"{path} 不是 schema_version={CHECKPOINT_SCHEMA_VERSION} 的 Matcher checkpoint。")
    model_state = state.get("model_state")
    if not isinstance(model_state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in model_state.items()
    ):
        raise ValueError(f"{path}.model_state 必须是字符串键张量映射。")
    state["model_state"] = dict(model_state)
    return cast(CheckpointEnvelope, state)
