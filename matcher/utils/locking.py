"""
提供基于同一路径锁文件的跨平台合作进程互斥上下文.

主要入口:
- exclusive_file_lock: 建立或打开锁文件, 阻塞请求平台独占锁, 取得成功后执行调用者代码并最终尝试解锁和关闭句柄.
- _lock: Windows 锁定文件首字节, 非 Windows 平台通过 fcntl.flock 锁定打开文件.
- _unlock: 按与 _lock 相同的平台分支释放锁.

该锁只串行所有使用同一路径和兼容锁协议的参与者, 不锁定受保护的其它文件, 也不能阻止
外部写者绕过协议. 锁文件会保留在磁盘, 其存在不表示当前有人持锁; 网络或特殊文件系统
能否在不同节点间提供预期互斥, 取决于操作系统和挂载实现. 本模块没有调用者可配置的超时、
非阻塞入口、所有者记录或 Slurm try_lock、pre_lock、after_lock、kill_lock 的资源状态机语义;
阻塞锁最终等待还是报错服从对应平台运行库.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """
    阻塞请求 path 对应的合作独占锁, 取得成功后在上下文退出时尝试释放.

    输入参数:
    - path: Path; 锁文件路径; 父目录不存在时递归创建, 文件以 UTF-8 a+ 模式打开且退出后保留.

    生成值:
    - value: None; 调用者只使用持锁区间, 不从上下文接收文件句柄或所有者信息.

    生命周期:
    - handle: TextIO; 打开后先传给 _lock, 取得锁成功才进入调用者代码; 最终由外层 finally 调用 close.
    - acquisition: _lock 失败时不调用 _unlock, 也不进入上下文正文, 但仍尝试关闭 handle.
    - release: 正文正常或异常退出都会调用 _unlock; 没有调用者可配置的超时、取消标志或非阻塞分支.
    - lock_file: 文件内容和路径在退出后保留, 不能以 exists 判断锁是否正被持有.

    异常优先级:
    - body_error: 正文异常优先于 _unlock 异常; 若异常对象支持 add_note, 则把解锁异常文本附加为说明.
    - unlock_error: 正文没有异常时, _unlock 异常向调用者传播.
    - close_error: handle.close 位于最外层 finally; 关闭失败可能遮蔽加锁、正文或解锁阶段正在传播的异常.
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    # TextIO, 以追加读写模式打开的锁文件句柄; 文件不会在上下文结束时删除.
    handle = path.open("a+", encoding="utf-8")
    try:
        _lock(handle)
        try:
            yield
        finally:
            # BaseException 或 None; 正文正在传播的异常用于决定解锁异常的优先级.
            pending_exception = sys.exc_info()[1]
            try:
                _unlock(handle)
            except BaseException as unlock_error:
                if pending_exception is None:
                    raise
                if hasattr(pending_exception, "add_note"):
                    pending_exception.add_note(f"释放文件锁时另有异常：{unlock_error!r}")
    finally:
        handle.close()


def _lock(handle: TextIO) -> None:
    """
    按当前操作系统对已打开句柄取得阻塞式独占锁.

    输入参数:
    - handle: TextIO; exclusive_file_lock 以 a+ 模式打开的锁文件句柄, 本函数会改变其当前位置.

    返回值:
    - result: None; 返回即表示本平台锁调用成功, 锁由 handle 保持.

    平台行为:
    - windows: os.name 等于 nt 时, 本句柄观察到空文件便写入字符 0 并 flush, 再从偏移 0 通过 msvcrt.LK_LOCK 请求一个字节锁; 重试与最终 OSError 语义由 Windows C runtime 提供, 该初始化不执行 fsync.
    - non_windows: 其它平台通过 fcntl.flock 和 LOCK_EX 阻塞请求打开文件的锁; 等待、中断和文件系统语义由平台提供.

    异常:
    - platform_error: 导入平台模块、句柄定位、读写、flush 或系统锁失败时, 底层异常直接传播.
"""

    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        if handle.read(1) == "":
            handle.write("0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]


def _unlock(handle: TextIO) -> None:
    """
    按当前操作系统释放 handle 对应的独占锁.

    输入参数:
    - handle: TextIO; 已经由 _lock 成功加锁且尚未关闭的文件句柄, 本函数会改变其当前位置.

    返回值:
    - result: None; 返回即表示平台解锁调用正常完成, 文件内容和路径仍保留.

    平台行为:
    - windows: os.name 等于 nt 时定位到偏移 0, 并通过 msvcrt.LK_UNLCK 释放一个字节.
    - non_windows: 其它平台通过 fcntl.flock 和 LOCK_UN 释放打开文件上的锁.

    异常:
    - platform_error: 导入平台模块、句柄定位或系统解锁失败时, 底层异常直接传播; 调用者决定其与正文异常的优先级.
"""

    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
