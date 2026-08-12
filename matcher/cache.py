"""提供按数组实际字节数约束的 DataLoader 工作进程内 LRU 缓存。"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import fields, is_dataclass
from typing import Generic, TypeVar

import numpy as np


Key = TypeVar("Key")
Value = TypeVar("Value")


class ByteBudgetLru(Generic[Key, Value]):
    """缓存最近访问的对象，并在总字节数越界时淘汰最久未使用项。"""

    def __init__(self, maximum_bytes: int) -> None:
        if maximum_bytes < 0:
            raise ValueError("maximum_bytes 不能为负数。")
        self.maximum_bytes = int(maximum_bytes)
        self.current_bytes = 0
        self._values: OrderedDict[Key, tuple[Value, int]] = OrderedDict()

    def get(self, key: Key) -> Value | None:
        """返回命中值并把它移到最近使用端；未命中时返回 ``None``。"""

        entry = self._values.get(key)
        if entry is None:
            return None
        self._values.move_to_end(key)
        return entry[0]

    def put(self, key: Key, value: Value) -> None:
        """加入一个对象，并按字节预算淘汰旧对象。单个超大对象不进入缓存。"""

        old = self._values.pop(key, None)
        if old is not None:
            self.current_bytes -= old[1]
        size = array_nbytes(value)
        if self.maximum_bytes == 0 or size > self.maximum_bytes:
            return
        self._values[key] = (value, size)
        self.current_bytes += size
        while self.current_bytes > self.maximum_bytes:
            _, (_, removed_size) = self._values.popitem(last=False)
            self.current_bytes -= removed_size

    def __len__(self) -> int:
        """返回当前缓存对象数量。"""

        return len(self._values)

    def keys(self) -> Iterator[Key]:
        """按从最久未使用到最近使用的顺序迭代键。"""

        return iter(self._values.keys())


def array_nbytes(value: object) -> int:
    """递归统计 NumPy 数组、映射和序列持有的有效数据字节数。"""

    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(array_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(array_nbytes(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return sum(array_nbytes(getattr(value, field.name)) for field in fields(value))
    return 0
