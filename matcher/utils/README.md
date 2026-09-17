# utils：无科学含义的通用基础设施

本目录提供按字节预算的最近使用缓存、严格 JSON/JSONL 写入和合作文件锁。它不定义 SMILES 身份、候选标签、模型、损失、选模或评估口径。

## 推荐阅读顺序

1. `cache.py`：理解 Dataset 的进程内字节预算缓存。
2. `json_artifacts.py`：理解严格 JSON、原子替换和 JSONL 追加边界。
3. `locking.py`：理解跨平台合作独占锁及其不保证的事项。
4. `__init__.py`：查看目录公开入口。

## 文件职责

| 文件 | 公开能力 | 不负责的事项 |
| --- | --- | --- |
| `cache.py` | `ByteBudgetLru` 按 NumPy 数组实际字节数淘汰最久未使用的 PDB 载荷。 | 不改变候选顺序和数组内容；预算 0 表示不缓存。 |
| `json_artifacts.py` | `write_json()` 原子发布完整严格 JSON；`append_jsonl()` 同步追加一个 JSON 对象；非有限 Python `float` 写为 `null`。 | 不提供并发锁；不自动转换 Path、NumPy 数组或张量。 |
| `locking.py` | `exclusive_file_lock()` 让使用同一路径和兼容协议的进程串行进入临界区。 | 不实现 Slurm `try_lock`、`pre_lock`、`after_lock` 或 `kill_lock`，也不能阻止绕过协议的写者。 |

## 缓存契约

`ByteBudgetLru[str, dict]` 的键是 PDB 标识，值是当前 Dataset worker 已解压的 PDB 载荷。预算按值中 NumPy 数组的实际字节数计算；插入新值后从最久未使用项开始淘汰。缓存命中只减少重复文件读取，不改变 manifest 冻结顺序、SMILES 身份轴或张量数值。

## JSON 写入边界

`write_json(path, value)` 先完成严格序列化，再写同目录 PID 临时文件、`flush`、`fsync` 并 `os.replace` 目标。成功后目标是一个 UTF-8、两空格缩进、键排序且末尾含换行的 JSON 对象。`append_jsonl()` 每次追加一个无缩进对象和换行并同步文件，但写入失败时不回滚文件尾。

两个入口都要求所有嵌套映射键是字符串。Python 非有限浮点数被递归转换为 `null`；元组转为列表；未支持类型交给标准 JSON 编码器拒绝。它们均不内置文件锁，同一路径并发写入必须由调用者使用 `exclusive_file_lock()` 或更高层单写者协议串行化。

## 文件锁边界

`exclusive_file_lock(path)` 会保留锁文件本身；文件存在不表示当前有人持锁。Windows 锁定首字节，非 Windows 使用 `fcntl.flock(LOCK_EX)`。接口没有超时、非阻塞模式、所有者记录或集群资源状态机；跨节点网络文件系统是否提供预期互斥取决于挂载和操作系统实现。
