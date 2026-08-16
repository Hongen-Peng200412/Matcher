"""
把运行记录写成不含非标准数值字面量的 UTF-8 JSON 或 JSONL。

主要入口:
    - write_json: 先序列化完整对象，再同步同目录 PID 命名临时文件并用 os.replace 发布目标文件。
    - append_jsonl: 以追加模式写入并同步一个 JSON 对象和一个 LF 行结束符。
    - _json_safe: 递归把 isinstance(value, float) 且非有限的值转换为 None，并把元组转换为列表。

两条公开入口都要求所有嵌套映射键为字符串，并把满足 isinstance(value, float) 的非有限值
落盘为 JSON null。
它们不会把任意 Python 对象转换成 JSON：未被 _json_safe 处理的类型仍交给标准 json 编码器拒绝。
write_json 的同目录替换和 append_jsonl 的追加没有内置文件锁；需要并发串行化时，调用者必须在
更高层使用同一个合作锁协议。
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Mapping[str, object]) -> None:
    """
    同步完整临时文件，并用同目录替换发布一个严格 JSON 对象。

    输入参数:
        - path: Path；最终 JSON 文件路径；父目录不存在时递归创建，已有目标在成功的 os.replace 调用中被替换。
        - value: Mapping[str, object]；顶层 JSON 对象来源，本函数不修改调用者映射。
        - value.keys: str；顶层和任意嵌套 Mapping 的每个键都必须是 str 实例。
        - value.values: object；Python float 非有限值递归转为 None，list、tuple 和 Mapping 递归处理，其它对象保持原值交给 json.dumps 验证。

    返回值:
        - result: None；成功时 path 指向本次发布的完整 JSON 文件；父目录创建和既有目标替换也属于文件系统副作用。

    输出格式:
        - output_text: str；UTF-8、ensure_ascii=False、两空格缩进、所有映射键排序、末尾一个 LF，并由 allow_nan=False 拒绝剩余非标准数值。
        - temporary: Path；与 path 同目录，文件名为点号、目标文件名、.tmp. 和当前 PID；该名称不区分同一进程内对同一路径的并发调用。

    发布与失败边界:
        - file_sync: 临时文件在替换前执行 flush 和 fsync；本函数不 fsync 父目录，因此不承诺断电后的目录项持久性。
        - atomic_replace: 临时文件与目标位于同一目录；是否提供原子目录项替换服从 os.replace 和当前文件系统语义。
        - concurrency: 本函数没有内部锁，同一进程对同一 path 的并发调用会争用同一临时路径，对同一 path 的不同调用也可能互相替换目标或清理对方临时文件。
        - cleanup: finally 尝试删除仍存在的临时文件；删除本身失败时可能遮蔽此前异常并留下临时文件。
        - serialization: json.dumps 在临时文件建立前运行；序列化失败时父目录可能已创建，但不会由本调用新建临时文件。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # str，已经完成递归数值转换和严格标准编码的完整 JSON 文本，末尾固定一个 LF。
    text = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    # Path，与目标同目录且由当前 PID 命名；不是同一进程并发调用的唯一临时文件。
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
    """
    以追加模式同步一个严格 JSON 对象记录。

    输入参数:
        - path: Path；最终 JSONL 文件路径；父目录不存在时递归创建，文件不存在时由追加模式建立。
        - value: Mapping[str, object]；本次单行 JSON 对象来源，本函数不修改调用者映射。
        - value.keys: str；顶层和任意嵌套 Mapping 的每个键都必须是 str 实例。
        - value.values: object；递归转换规则与 write_json 相同，未支持类型仍由 json.dumps 拒绝。

    返回值:
        - result: None；成功后 path 末尾新增一个完整记录。

    输出格式:
        - output_line: str；UTF-8、ensure_ascii=False、键排序、无缩进、末尾一个 LF，并由 allow_nan=False 拒绝剩余非标准数值。

    追加与失败边界:
        - file_sync: 写入后执行 flush 和 fsync；本函数不 fsync 父目录。
        - concurrency: 本函数没有内部锁，不承诺多个进程或线程追加时整条记录不可交错；合作调用者需要在外层共用锁。
        - partial_write: 写入或同步失败时没有回滚，文件尾可能保留不完整记录。
        - existing_content: 本函数不读取、不验证也不重写已有行；已有畸形内容不会在本次追加前被发现。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # str，单个无缩进 JSON 对象和一个 LF；字符串内部换行由标准编码器转义。
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
    """
    递归转换本模块明确支持的严格 JSON 边界值。

    输入参数:
        - value: Any；当前递归节点，可能是 float、Mapping、list、tuple 或其它对象。
        - value.mapping_keys: object；Mapping 分支要求每个键都是 str 实例，不把整数键等对象隐式转成字符串。
        - value.mapping_values: object；Mapping 分支按原迭代关系递归每个值，后续 json.dumps 负责键排序。
        - value.sequence_items: object；list 和 tuple 分支按位置递归，并统一返回新的 list。

    返回值:
        - converted: Any；有限 Python float 原值返回，非有限 Python float 返回 None，Mapping 返回新 dict，list 或 tuple 返回新 list，其它对象原样返回。

    边界:
        - json_compatibility: 返回值不保证已经可 JSON 编码；Path、NumPy 对象、Torch 张量、dataclass、集合和其它未处理对象仍由 json.dumps 决定是否报错。
        - recursion: 本函数不检测循环引用；循环 Mapping、list 或 tuple 会递归直至 Python 报错。
        - numeric_types: 只有满足 isinstance(value, float) 的对象进入有限性转换；其它数值类型不会由本函数主动规范化。
    """

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON 记录的所有映射键必须是字符串。")
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
