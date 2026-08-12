"""读取 A–G LigandObject，并转换为完整 Matcher 使用的配体图张量契约。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cache import ByteBudgetLru
from .schema import safe_object_filename


@dataclass(frozen=True)
class LigandGraph:
    """保存一个去重配体身份的节点类别、连续属性和双向化学键。

    ``element/charge/residue_id`` 是对齐到 N 个原子的整数类别。15 维节点连续
    特征依次为 chirality(7)、in_ring(4)、is_present(1) 和中心化后除以
    10 Å 的参考 XYZ(3)。``edge_index`` 为 ``(2,E)`` 双向端点；10 维边特征
    依次为 bond type(5)、in_ring(4) 和原子间距离除以 10 Å(1)。
    """

    element: np.ndarray
    charge: np.ndarray
    residue_id: np.ndarray
    node_continuous: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray


class LigandGraphStore:
    """按 object_key 读取配体图，并在单 worker 的独立字节预算内缓存。"""

    def __init__(self, data_root: Path, cache_bytes: int = 512 * 1024**2) -> None:
        if cache_bytes < 0:
            raise ValueError("cache_bytes 不能为负数。")
        self.data_root = data_root
        self.cache_bytes = int(cache_bytes)
        self._cache: ByteBudgetLru[str, LigandGraph] = ByteBudgetLru(self.cache_bytes)

    def load(self, object_key: str) -> LigandGraph:
        """返回经过字段和端点验证的双向 LigandObject 图。"""

        cached = self._cache.get(object_key)
        if cached is not None:
            return cached
        path = self.data_root / "ligand_objects" / f"{safe_object_filename(object_key)}.npz"
        graph = _load_ligand_graph(path, object_key)
        self._cache.put(object_key, graph)
        return graph

    def __getstate__(self) -> dict[str, object]:
        """DataLoader 派生 worker 时不复制父进程图缓存。"""

        state = self.__dict__.copy()
        state["_cache"] = ByteBudgetLru(self.cache_bytes)
        return state


def _load_ligand_graph(path: Path, object_key: str) -> LigandGraph:
    """从 LigandObject 结构化数组建立节点字段和双向键字段。"""

    with np.load(path, allow_pickle=False) as archive:
        required = {"atoms", "bonds", "name"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"{path} 缺少 LigandObject 字段 {missing}。")
        atoms = archive["atoms"].copy()
        bonds = archive["bonds"].copy()
        actual_name = str(archive["name"].item())
    if actual_name != object_key:
        raise ValueError(f"{path}:name={actual_name!r}，期望 {object_key!r}。")
    atom_fields = {"element", "charge", "ref_pos", "is_present", "chirality", "in_ring", "residue_id"}
    bond_fields = {"atom_1", "atom_2", "type", "in_ring"}
    if atoms.dtype.names is None or not atom_fields.issubset(atoms.dtype.names):
        raise ValueError(f"{path}:atoms 缺少字段 {sorted(atom_fields)}。")
    if bonds.dtype.names is None or not bond_fields.issubset(bonds.dtype.names):
        raise ValueError(f"{path}:bonds 缺少字段 {sorted(bond_fields)}。")
    atom_count = int(atoms.shape[0])
    if atoms.ndim != 1 or atom_count == 0 or bonds.ndim != 1:
        raise ValueError(f"{path}:atoms 必须为非空一维数组，bonds 必须为一维数组。")

    reference_position = np.asarray(atoms["ref_pos"], dtype=np.float32)
    if reference_position.shape != (atom_count, 3) or not np.isfinite(reference_position).all():
        raise ValueError(f"{path}:atoms.ref_pos 必须为有限 (N,3)。")
    centered_position = (reference_position - reference_position.mean(axis=0, keepdims=True)) / 10.0
    node_continuous = np.concatenate(
        (
            np.asarray(atoms["chirality"], dtype=np.float32),
            np.asarray(atoms["in_ring"], dtype=np.float32),
            np.asarray(atoms["is_present"], dtype=np.float32)[:, None],
            centered_position,
        ),
        axis=1,
    )

    if bonds.size:
        source = np.asarray(bonds["atom_1"], dtype=np.int64)
        target = np.asarray(bonds["atom_2"], dtype=np.int64)
        if np.any(source < 0) or np.any(target < 0) or np.any(source >= atom_count) or np.any(target >= atom_count):
            raise ValueError(f"{path}:bonds 含越界原子端点。")
        undirected = np.stack((source, target), axis=0)
        edge_index = np.concatenate((undirected, undirected[::-1]), axis=1)
        bond_type = np.asarray(bonds["type"], dtype=np.float32)
        bond_ring = np.asarray(bonds["in_ring"], dtype=np.float32)
        distance = np.linalg.norm(reference_position[source] - reference_position[target], axis=1)
        one_direction = np.concatenate((bond_type, bond_ring, (distance / 10.0)[:, None]), axis=1)
        edge_features = np.concatenate((one_direction, one_direction), axis=0)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_features = np.empty((0, 10), dtype=np.float32)

    return LigandGraph(
        element=np.asarray(atoms["element"], dtype=np.int64),
        charge=np.asarray(atoms["charge"], dtype=np.int64),
        residue_id=np.asarray(atoms["residue_id"], dtype=np.int64),
        node_continuous=node_continuous.astype(np.float32, copy=False),
        edge_index=edge_index,
        edge_features=edge_features.astype(np.float32, copy=False),
    )
