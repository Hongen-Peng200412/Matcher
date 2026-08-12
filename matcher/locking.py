"""提供服务器并发任务共享正式清单和缓存时使用的进程级文件锁。"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """无限期等待并独占一个小文件，进程退出时由操作系统释放锁。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        _lock(handle)
        yield
    finally:
        _unlock(handle)
        handle.close()


def _lock(handle: TextIO) -> None:
    """按当前操作系统取得阻塞式独占锁。"""

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

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock(handle: TextIO) -> None:
    """释放当前进程持有的文件锁。"""

    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
