"""读取由精确 PreparedLigand.smiles 唯一决定的全局重原子图包.

主要入口:
- SmilesGraph: 保存一个 SMILES 的原子和有向化学键数组.
- SmilesGraphStore: 每个 DataLoader 工作进程完整加载一次图包, 并按精确字符串取图.

图包不保存坐标、模板键或另造的图标识. 运行时字典的整数值只是数组切片位置,
不会写回清单, 也不构成新的科学身份.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SmilesGraph:
    """保存一个精确 SMILES 对应的重原子化学图.

    字段:
    - element: int64 数组, (N_atom,); 按 RDKit 重原子顺序保存原子序数.
    - charge: int64 数组, (N_atom,); 与 element 逐原子对齐的形式电荷.
    - atom_in_ring: float32 数组, (N_atom, 4); 与 element 逐原子对齐, 四列依次标记 3、4、5、6 元环.
    - edge_index: int64 数组, (2, 2*N_bond); 第一轴为源端点和目标端点, 每条无向键按图包方向、反向各保存一次.
    - edge_features: float32 数组, (2*N_bond, 9); 与 edge_index 第二轴逐边对齐, 前五列为键型独热编码, 后四列依次标记 3、4、5、6 元环.
"""

    element: np.ndarray
    charge: np.ndarray
    atom_in_ring: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray


class SmilesGraphStore:
    """在当前进程内完整载入一个版本化 SMILES 图包.

    构造参数:
    - path: smiles_graphs_v1.npz 的绝对或相对路径.

    生命周期:
    - initialize: 打开图包一次, 把全部数组复制到内存并关闭文件.
    - get: 按精确顶层 SMILES 返回由内存切片和有向边派生数组构成的 SmilesGraph.

    同一 Dataset 副本只持有一个 Store. DataLoader 使用持久工作进程时, worker_init_fn
    在每个工作进程启动阶段调用 initialize, 样本读取阶段不再打开图包.
"""

    def __init__(self, path: Path) -> None:
        """记录图包路径; 此时不执行文件 I/O.

        - path: 图包路径; initialize 会检查文件和字段.
        - self.path: 规范化为 Path 的图包位置.
        - self._arrays: 初始化前为 None; 初始化后保存内存数组映射.
        - self._index_by_smiles: 初始化前为空; 初始化后把精确字符串映射到数组位置.
"""

        self.path = Path(path)
        self._arrays: dict[str, np.ndarray] | None = None
        self._index_by_smiles: dict[str, int] = {}

    def initialize(self) -> None:
        """完整读取并验证图包, 重复调用时直接复用已有内存.

        图包字段:
        - schema_version: int64 标量, 固定为 1.
        - smiles: Unicode 数组, (K,); K 个精确 PreparedLigand.smiles 按字符串严格递增且不重复.
        - atom_offsets: int64 数组, (K + 1,); atom_offsets[i]:atom_offsets[i + 1] 同步切分 element、charge 和 atom_in_ring, 首值为 0, 末值为总原子数 N_atom_total.
        - element: int16 数组, (N_atom_total,); K 张图连接后的原子序数.
        - charge: int8 数组, (N_atom_total,); 与 element 逐原子对齐的形式电荷.
        - atom_in_ring: bool 数组, (N_atom_total, 4); 与 element 逐原子对齐的 3、4、5、6 元环标志.
        - bond_offsets: int64 数组, (K + 1,); bond_offsets[i]:bond_offsets[i + 1] 同步切分 bond_index、bond_type 和 bond_in_ring, 首值为 0, 末值为无向键总数 N_bond_total.
        - bond_index: int32 数组, (2, N_bond_total); 每张图内部的无向键局部端点, 两行依次为 RDKit 起始原子与终止原子下标.
        - bond_type: uint8 数组, (N_bond_total,); 与 bond_index 第二轴逐键对齐, 0:5 依次为 single、double、triple、dative、aromatic.
        - bond_in_ring: bool 数组, (N_bond_total, 4); 与 bond_index 第二轴逐键对齐的 3、4、5、6 元环标志.

        返回状态:
        - self._arrays: dict[str, np.ndarray], 全部字段的独立内存副本; np.load 上下文在赋值前关闭.
        - self._index_by_smiles: dict[str, int], 精确 SMILES 到 smiles 第一轴位置的映射.
        - return: None; 已初始化时不重复打开文件.
"""

        if self._arrays is not None:
            return
        required = {
            "schema_version",
            "smiles",
            "atom_offsets",
            "element",
            "charge",
            "atom_in_ring",
            "bond_offsets",
            "bond_index",
            "bond_type",
            "bond_in_ring",
        }
        with np.load(self.path, allow_pickle=False) as archive:
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{self.path} 缺少字段: {sorted(missing)}")
            arrays = {name: np.asarray(archive[name]).copy() for name in required}
        if arrays["schema_version"].shape != () or int(arrays["schema_version"]) != 1:
            raise ValueError(f"{self.path}: schema_version 必须是标量 1。")
        smiles_values = [str(value) for value in arrays["smiles"].tolist()]
        if not smiles_values or smiles_values != sorted(set(smiles_values)):
            raise ValueError(f"{self.path}: smiles 必须非空、唯一并按精确字符串递增。")
        atom_offsets = arrays["atom_offsets"]
        bond_offsets = arrays["bond_offsets"]
        if atom_offsets.dtype != np.int64 or atom_offsets.shape != (len(smiles_values) + 1,):
            raise ValueError("atom_offsets 必须是长度 K+1 的 int64 数组。")
        if bond_offsets.dtype != np.int64 or bond_offsets.shape != (len(smiles_values) + 1,):
            raise ValueError("bond_offsets 必须是长度 K+1 的 int64 数组。")
        if int(atom_offsets[0]) != 0 or int(bond_offsets[0]) != 0:
            raise ValueError("atom_offsets 和 bond_offsets 必须从零开始。")
        if np.any(np.diff(atom_offsets) <= 0) or np.any(np.diff(bond_offsets) < 0):
            raise ValueError("每个 SMILES 必须至少含一个重原子，键数量允许为零。")
        atom_count = int(atom_offsets[-1])
        bond_count = int(bond_offsets[-1])
        expected = {
            "element": (np.dtype(np.int16), (atom_count,)),
            "charge": (np.dtype(np.int8), (atom_count,)),
            "atom_in_ring": (np.dtype(np.bool_), (atom_count, 4)),
            "bond_index": (np.dtype(np.int32), (2, bond_count)),
            "bond_type": (np.dtype(np.uint8), (bond_count,)),
            "bond_in_ring": (np.dtype(np.bool_), (bond_count, 4)),
        }
        for name, (dtype, shape) in expected.items():
            if arrays[name].dtype != dtype or arrays[name].shape != shape:
                raise ValueError(f"{name} 必须是 {dtype} {shape}，实际为 {arrays[name].dtype} {arrays[name].shape}。")
        if np.any(arrays["bond_type"] > 4):
            raise ValueError("bond_type 只能取 0 到 4。")
        for graph_index in range(len(smiles_values)):
            atom_count_in_graph = int(atom_offsets[graph_index + 1] - atom_offsets[graph_index])
            bond_start, bond_end = bond_offsets[graph_index : graph_index + 2]
            bond_index = arrays["bond_index"][:, bond_start:bond_end]
            if np.any(bond_index < 0) or np.any(bond_index >= atom_count_in_graph):
                raise ValueError(f"第 {graph_index} 张图的 bond_index 超出其局部原子轴。")
        self._arrays = arrays
        self._index_by_smiles = {smiles: index for index, smiles in enumerate(smiles_values)}

    def __contains__(self, smiles: str) -> bool:
        """返回精确 SMILES 是否存在.

        - smiles: str, PreparedLigand 顶层 smiles 原始字符串, 不做 canonicalize 或空白修剪.
        - return: bool, 字符串是否精确存在于图包 smiles 第一轴; 首次查询会触发一次完整加载.
"""

        self.initialize()
        return smiles in self._index_by_smiles

    def get(self, smiles: str) -> SmilesGraph:
        """按精确字符串返回一张图; 不存在时抛出 KeyError.

        - smiles: PreparedLigand 顶层 smiles 的原始字符串, 不做 canonicalize 或空白修剪.
        - return: SmilesGraph; 数组顺序沿用离线 RDKit 解析后的重原子顺序.
"""

        self.initialize()
        assert self._arrays is not None
        try:
            graph_index = self._index_by_smiles[smiles]
        except KeyError as error:
            raise KeyError(f"SMILES 图包中不存在精确字符串: {smiles!r}") from error
        atom_start, atom_end = self._arrays["atom_offsets"][graph_index : graph_index + 2]
        bond_start, bond_end = self._arrays["bond_offsets"][graph_index : graph_index + 2]
        # int64 (2, N_bond), 图包保存图内局部原子下标; 连接批次时才增加节点偏移.
        undirected = self._arrays["bond_index"][:, bond_start:bond_end].astype(np.int64, copy=False)
        # float32 (N_bond, 5), 把 0 到 4 的键型编号转为独热编码.
        bond_type = np.eye(5, dtype=np.float32)[self._arrays["bond_type"][bond_start:bond_end]]
        # float32 (N_bond, 4), 四列依次对应 3、4、5、6 元环.
        bond_ring = self._arrays["bond_in_ring"][bond_start:bond_end].astype(np.float32)
        # int64 (2, 2*N_bond), 先保留图包方向, 再追加反向; 无键图得到 (2, 0).
        edge_index = np.concatenate((undirected, undirected[::-1]), axis=1)
        # float32 (2*N_bond, 9), 正反两向共享相同的键型和环标志.
        one_direction = np.concatenate((bond_type, bond_ring), axis=1)
        edge_features = np.concatenate((one_direction, one_direction), axis=0)
        return SmilesGraph(
            element=self._arrays["element"][atom_start:atom_end].astype(np.int64),
            charge=self._arrays["charge"][atom_start:atom_end].astype(np.int64),
            atom_in_ring=self._arrays["atom_in_ring"][atom_start:atom_end].astype(np.float32),
            edge_index=edge_index,
            edge_features=edge_features,
        )

    @property
    def smiles_count(self) -> int:
        """返回图包内精确 SMILES 数量.

        - return: int, 图包 smiles 第一轴长度; 首次读取会完成初始化.
"""

        self.initialize()
        return len(self._index_by_smiles)
