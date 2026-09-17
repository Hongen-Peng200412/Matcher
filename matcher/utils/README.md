# utils：无科学含义的通用基础设施

本目录提供按字节预算的最近使用缓存、严格 JSON 写入和互斥文件锁。它不定义 SMILES 身份、标签、模型、损失、选模或评估口径。

| 文件 | 公开能力 |
|---|---|
| `cache.py` | `ByteBudgetLru`：按 NumPy 数组实际字节数约束 PDB 载荷缓存 |
| `json_artifacts.py` | 原子 JSON、追加 JSONL、有限数值转换和独占文件锁 |

`ByteBudgetLru` 只影响重复 I/O，不改变候选顺序或张量内容。预算为 0 时不缓存。文件锁只协调遵守同一路径协议的进程，不替代 Slurm allocation 的锁治理。
