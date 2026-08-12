"""保存、验证并加载 Matcher 训练 checkpoint。

``save_checkpoint`` 原子发布 BEST.pt 与 LAST.pt；``load_checkpoint`` 验证项目 schema；
``load_candidate_encoder_checkpoint`` 从候选预训练或完整 Matcher 中提取候选编码器参数。
正式 checkpoint 顶层至少包含 ``schema_version``、``model_state``，训练产物另保存优化器、
调度器、混合精度缩放器、epoch、早停状态和该轮结构化记录。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, cast

import torch
from torch import nn


CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointEnvelope(TypedDict):
    """所有可加载 Matcher checkpoint 必须提供的最小顶层结构。

    ``schema_version`` 是与 CHECKPOINT_SCHEMA_VERSION 严格相等的整数。
    ``model_state`` 是字符串参数名到张量的完整模型状态映射。
    """

    schema_version: int
    model_state: dict[str, torch.Tensor]


def save_checkpoint(path: Path, state: Mapping[str, object]) -> None:
    """把 checkpoint 写到同目录临时文件，落盘同步后原子替换目标。"""

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
    path: Path,
    map_location: str | torch.device = "cpu",
) -> CheckpointEnvelope:
    """验证并返回项目 checkpoint；默认加载到 CPU，不提前占用 GPU 显存。"""

    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"{path} 不是 Matcher checkpoint。")
    if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"{path} 的 checkpoint schema_version 必须为 {CHECKPOINT_SCHEMA_VERSION}。"
        )
    model_state = state.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError(f"{path} 的 model_state 必须是字符串键张量映射。")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in model_state.items()
    ):
        raise ValueError(f"{path} 的 model_state 必须只含字符串键和张量值。")
    state["model_state"] = dict(model_state)
    return cast(CheckpointEnvelope, state)


def load_candidate_encoder_checkpoint(
    model: nn.Module,
    path: Path,
) -> dict[str, list[str]]:
    """把候选预训练 checkpoint 严格加载到候选编码器同名参数。

    ``model`` 可以是 ``CandidateEncoder``，也可以是含 ``candidate_encoder`` 子模块
    的完整 Matcher。完整模型的配体身份分支保持自身初始化。
    """

    state = load_checkpoint(path)
    target = getattr(model, "candidate_encoder", model)
    if not isinstance(target, nn.Module):
        raise TypeError("model.candidate_encoder 必须是 torch.nn.Module。")
    source_state = state["model_state"]
    if any(key.startswith("candidate_encoder.") for key in source_state):
        source_state = {
            key.removeprefix("candidate_encoder."): value
            for key, value in source_state.items()
            if key.startswith("candidate_encoder.")
        }
    result = target.load_state_dict(source_state, strict=True)
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }
