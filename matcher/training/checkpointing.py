"""保存、验证并加载 Matcher 训练 checkpoint。

save_checkpoint 只负责把调用方映射发布成 PyTorch checkpoint，不解释训练字段；
run_training_stage 用它覆盖 LAST.pt，并在验证分数严格改善时覆盖 BEST.pt。
load_checkpoint 只验证 schema_version 和 model_state 两个最小字段，其余训练状态保持原值。
load_candidate_encoder_checkpoint 是兼容性入口，可从候选编码器直属状态或完整 Matcher
中带 candidate_encoder. 前缀的状态提取候选编码器参数。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, cast

import torch
from torch import nn


# int，load_checkpoint接受且run_training_stage写入的最小checkpoint schema版本。
CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointEnvelope(TypedDict):
    """声明 load_checkpoint 实际核验并返回的最小顶层结构。

    - schema_version: Python 整数语义的版本标记；当前必须等于 CHECKPOINT_SCHEMA_VERSION，但加载器使用相等比较，未额外拒绝与整数相等的其它标量类型。
    - model_state: 参数或缓冲区名称到 torch.Tensor 的映射；加载时复制为普通 dict，张量设备由 map_location 决定。

    TypedDict 只描述两个必需键。正式训练 checkpoint 还包含 stage_name、task_name、epoch、
    global_step、best_epoch、best_score、bad_validation_epochs、model_config、training_config、
    optimizer_state、scheduler_state、scaler_state 和 epoch_record；load_checkpoint 不验证或删除这些附加键。
    """

    schema_version: int
    model_state: dict[str, torch.Tensor]


def save_checkpoint(path: Path, state: Mapping[str, object]) -> None:
    """把映射写入同目录临时文件，文件同步后原子替换目标路径。

    - path: 目标 checkpoint 路径；父目录不存在时递归创建，既有目标由 os.replace 覆盖。
    - state: 交给 torch.save 的字符串键或其它键映射；本函数不验证字段、键类型、张量设备或可序列化性。

    临时文件名为目标文件名前加点并附当前 PID；同一进程对同一目标并发调用会使用同一路径，
    本函数不提供进程内互斥。写入、flush、fsync 或替换失败后 finally 会尝试删除临时文件；
    删除自身也可能失败并遮蔽原异常，且失败时临时文件可能保留。目录本身不执行 fsync。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # Path，与目标同目录且按PID命名的临时文件；同进程并发不隔离，同文件系统os.replace负责目标替换。
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
    """加载 checkpoint，并核验最小 schema 后返回包含附加顶层键的普通字典。

    - path: 必须存在的 checkpoint 文件路径；不存在时抛出 FileNotFoundError。
    - map_location: 逐张量重定位规则，直接传给 torch.load；默认字符串 cpu，调用方也可传 torch.device。

    - 返回值.schema_version: 必须与 CHECKPOINT_SCHEMA_VERSION 相等；本函数没有使用 type(...) is int 的严格类型门禁。
    - 返回值.model_state: 字符串名称到 torch.Tensor 的新普通 dict；只复制映射容器，不复制张量。
    - 返回值[附加顶层键]: torch.load 得到的原值；本函数保留任意附加键，不验证训练状态的字段、类型或相互关系。

    本入口显式使用 weights_only=False，因此只应读取受信任的项目 checkpoint。默认 CPU
    重定位可避免在只读核验阶段提前占用 GPU，但调用方覆盖 map_location 后不作此保证。
    """

    if not path.is_file():
        raise FileNotFoundError(path)
    # object，反序列化后的完整 checkpoint 根对象；最小 schema 门禁前暂不假设为映射。
    state = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"{path} 不是 Matcher checkpoint。")
    if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"{path} 的 checkpoint schema_version 必须为 {CHECKPOINT_SCHEMA_VERSION}。"
        )
    # object，checkpoint.model_state 的未验证值；随后要求为字符串键到张量的映射。
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
    """从兼容 checkpoint 中提取候选编码器状态并严格加载到目标模块。

    - model: CandidateEncoder，或提供 candidate_encoder 属性的完整 Matcher；若该属性存在但不是 nn.Module，则拒绝。
    - path: 由 load_checkpoint 读取的受信任项目 checkpoint 路径；本函数固定按默认规则加载到 CPU。

    - 返回值.missing_keys: strict=True 加载结果中的缺失参数名列表；成功返回时通常为空。
    - 返回值.unexpected_keys: strict=True 加载结果中的额外参数名列表；成功返回时通常为空。

    若 model_state 至少有一个 candidate_encoder. 前缀键，本函数只保留此类键并删除前缀；
    否则把完整 model_state 当作候选编码器直属状态。strict=True 会为键或形状不相容抛出
    异常，因此不相容结果不会作为非空列表正常返回；但该兼容入口没有预先逐张量核验，异常前
    已匹配的目标张量可能已经被复制，加载操作不具备事务原子性。完整 Matcher 的其它子模块不修改。
    """

    state = load_checkpoint(path)
    # nn.Module，实际接收状态的候选编码器；完整 Matcher 取其 candidate_encoder，直属编码器取自身。
    target = getattr(model, "candidate_encoder", model)
    if not isinstance(target, nn.Module):
        raise TypeError("model.candidate_encoder 必须是 torch.nn.Module。")
    # dict[str,Tensor]，来源最小 schema 中的模型状态；可能是直属候选状态，也可能带完整模型前缀。
    source_state = state["model_state"]
    if any(key.startswith("candidate_encoder.") for key in source_state):
        source_state = {
            key.removeprefix("candidate_encoder."): value
            for key, value in source_state.items()
            if key.startswith("candidate_encoder.")
        }
    # _IncompatibleKeys，严格加载成功后的兼容性报告；严格失败抛出 RuntimeError，不返回此对象。
    result = target.load_state_dict(source_state, strict=True)
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }
