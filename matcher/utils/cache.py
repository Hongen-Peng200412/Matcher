"""
提供按选定 NumPy 载荷字节数记账的进程内最近使用缓存.

主要入口:
- ByteBudgetLru: 用 OrderedDict 保存键、对象和插入时计算的字节数, 并在超出预算时从最久未使用端淘汰.
- array_nbytes: 递归累计 NumPy 数组、映射值、列表、元组和 dataclass 字段中的数组 nbytes.

这里的字节数不是进程常驻内存、Python 容器总开销或底层共享存储占用. 对象写入缓存后若
被原地改变, 缓存不会重新计量; 同一数组经多个引用出现时也会重复累计. 该工具只提供
单个 Python 对象内部的普通内存状态, 不提供线程锁、进程共享或磁盘缓存.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import fields, is_dataclass
from typing import Generic, TypeVar

import numpy as np


# TypeVar, 缓存键类型; 实际键必须可供 OrderedDict 哈希.
Key = TypeVar("Key")
# TypeVar, 缓存值类型; 运行时不限制为 NumPy 数组.
Value = TypeVar("Value")


class ByteBudgetLru(Generic[Key, Value]):
    """
    按 array_nbytes 的插入时估算值淘汰最久未使用对象.

    实例状态:
    - self.maximum_bytes: int; 非负字节预算, put 不会让已记录的 current_bytes 超过该值.
    - self.current_bytes: int; 当前各缓存项插入时 size 的总和, 不含键、OrderedDict、元组和未被 array_nbytes 统计的对象开销.
    - self._values: OrderedDict[Key, tuple[Value, int]]; 从最久未使用到最近使用排序, 每个值同时保存对象引用和插入时字节数.

    状态边界:
    - mutation: 缓存不复制 value, 也不把内部数组设为只读; 写入后的原地变化不会更新已记录字节数.
    - concurrency: 所有状态都属于当前 Python 实例; 本类没有线程互斥或跨进程共享.
    - zero_size: maximum_bytes 大于零时, array_nbytes 为零的对象可进入缓存, 因此字节预算不限制这类对象的数量.
"""

    def __init__(self, maximum_bytes: int) -> None:
        """
        建立一个空缓存并保存非负字节预算.

        输入参数:
        - maximum_bytes: int; 正式调用传入非负整数; 本函数先执行与零比较, 再调用 int 转换, 不单独拒绝 bool 或所有非 int 直接调用值.

        初始化结果:
        - self.maximum_bytes: int; maximum_bytes 转换后的字节预算.
        - self.current_bytes: int; 初值为 0.
        - self._values: OrderedDict[Key, tuple[Value, int]]; 初始为空.

        异常:
        - ValueError: maximum_bytes 与零比较后小于零.

        不支持与零比较或不能转换为 int 的直接调用值由 Python 原样报错.
"""

        if maximum_bytes < 0:
            raise ValueError("maximum_bytes 不能为负数。")
        self.maximum_bytes = int(maximum_bytes)
        self.current_bytes = 0
        self._values: OrderedDict[Key, tuple[Value, int]] = OrderedDict()

    def get(self, key: Key) -> Value | None:
        """
        查询键, 并把命中项移动到最近使用端.

        输入参数:
        - key: Key; 供 OrderedDict 查询的可哈希键, 本函数不做类型或可哈希性预检.

        返回值:
        - value: Value 或 None; 命中时返回缓存对象的同一引用, 未命中时返回 None; 若调用者主动缓存了值 None, 则返回值不能区分该命中与未命中.

        状态变化:
        - self._values: 命中项移动到有序映射末端; 未命中时顺序不变.
        - self.current_bytes: 查询不改变已记录字节数.
"""

        # tuple[Value, int] 或 None; 第二项是该对象 put 时由 array_nbytes 得到的字节数.
        entry = self._values.get(key)
        if entry is None:
            return None
        self._values.move_to_end(key)
        return entry[0]

    def put(self, key: Key, value: Value) -> None:
        """
        写入或替换一个键, 并从最久未使用端淘汰到预算内.

        输入参数:
        - key: Key; 供 OrderedDict 保存的可哈希键, 本函数不做类型或可哈希性预检.
        - value: Value; 缓存保存原对象引用, 字节数由 array_nbytes 在本次调用中计算一次.

        返回值:
        - result: None; 调用者通过 get、len 和 keys 观察缓存结果.

        状态变化:
        - replacement: 同名旧项先被删除; 即使新值因零预算或单项超预算而被拒收, 旧项也不会恢复.
        - measurement_failure: 替换旧项后 array_nbytes 若因循环结构等输入错误而失败, 异常直接传播且已删除的旧项不会恢复.
        - admission: maximum_bytes 为零或 value 的记录字节数大于预算时不保存新值; 等于预算时允许保存.
        - eviction: 接纳后从最久未使用端反复删除, 直到 current_bytes 不大于 maximum_bytes.
        - ordering: 新接纳项位于最近使用端; 淘汰不会读取或修改 value 内部内容.
"""

        # tuple[Value, int] 或 None; 替换时先撤销旧项及其插入时字节数.
        old = self._values.pop(key, None)
        if old is not None:
            self.current_bytes -= old[1]
        # int, value 当前由 array_nbytes 可见的载荷字节数; 后续原地变化不会重算.
        size = array_nbytes(value)
        if self.maximum_bytes == 0 or size > self.maximum_bytes:
            return
        self._values[key] = (value, size)
        self.current_bytes += size
        while self.current_bytes > self.maximum_bytes:
            # removed_size 是最久未使用项写入时记录的非负字节数.
            _, (_, removed_size) = self._values.popitem(last=False)
            self.current_bytes -= removed_size

    def __len__(self) -> int:
        """
        返回缓存当前保存的键数量.

        返回值:
        - count: int; self._values 的条目数, 包括 array_nbytes 为零的已接纳值.

        状态变化:
        - cache_state: 本查询不改变最近使用顺序或字节计数.
"""

        return len(self._values)

    def keys(self) -> Iterator[Key]:
        """
        返回按最近使用顺序遍历键的迭代器.

        返回值:
        - iterator: Iterator[Key]; 从最久未使用到最近使用遍历 self._values 的动态键视图, 不复制键列表.

        迭代边界:
        - mutation: 取得迭代器后修改缓存可能使迭代器反映变化或由 OrderedDict 报出迭代期间变更错误; 调用者需要自行避免并发修改.
"""

        return iter(self._values.keys())


def array_nbytes(value: object) -> int:
    """
    递归累计本缓存选定载荷中的 NumPy 数组 nbytes.

    输入参数:
    - value: object; 可为 NumPy 数组、Mapping、list、tuple、dataclass 实例或其它对象.
    - value.mapping_values: Mapping 分支只递归值, 不统计键、映射容器或映射元数据.
    - value.sequence_items: list 与 tuple 分支按位置递归元素, 不统计序列容器本身.
    - value.dataclass_fields: dataclass 实例分支按声明字段递归当前字段值; dataclass 类对象不进入该分支.

    返回值:
    - byte_count: int; 每个遇到的 NumPy 数组贡献 int(array.nbytes), 其它未支持对象直接贡献 0.

    计量边界:
    - ndarray_view: 数组视图按自身 nbytes 计量, 不追踪其底层存储是否与其它数组共享.
    - object_dtype: object 数组只计数组缓冲区报告的 nbytes, 不递归统计其 Python 对象载荷.
    - repeated_reference: 不记录对象身份, 同一数组经多个位置引用会重复累计.
    - cycles: 不检测容器或 dataclass 引用环; 循环结构会继续递归并最终由 Python 报错.
    - excluded_objects: NumPy 标量、字符串、Torch 张量、Path、集合、普通对象和容器自身开销均贡献 0.
"""

    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(array_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(array_nbytes(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return sum(array_nbytes(getattr(value, field.name)) for field in fields(value))
    return 0
