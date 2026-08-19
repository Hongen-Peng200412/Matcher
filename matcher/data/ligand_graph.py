"""
按 object_key 读取 A–G LigandObject NPZ，并转换为完整 Matcher 使用的配体模板图。

主要入口:
    - LigandGraphStore.load: 定位 ligand_objects/{safe_object_filename(object_key)}.npz，校验图字段并在当前 DataLoader worker 内按字节预算缓存。

返回对象:
    - LigandGraph: 保存逐模板原子的离散字段、15 维连续字段、双向化学键端点和 10 维边特征。

LigandObject 图只由 object_key 定位和复用；PreparedLigand.smiles、occurrence 沉积坐标与 present_{candidate_id} 掩码不参与本模块的模板图读取，也不由 atoms.is_present 定义 v4 联合身份。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils.cache import ByteBudgetLru
from .schema import safe_object_filename


@dataclass(frozen=True)
class LigandGraph:
    """
    保存一个 LigandObject 模板的节点类别、连续属性和双向化学键。

    形状符号:
        - N_atom: LigandObject.atoms 的模板重原子数量，必须大于 0。
        - N_bond: LigandObject.bonds 的无向化学键数量，必须大于或等于 0。
        - N_directed_edge: 双向化后的边数量，等于 2 * N_bond；N_bond 为 0 时 edge_index 为 (2, 0)，edge_features 为 (0, 10)。

    字段:
        - element: int64 数组，(N_atom,)；逐模板原子的原子序数，与 LigandObject.atoms 顺序一致。
        - charge: int64 数组，(N_atom,)；逐模板原子的形式电荷，与 element 逐位置对齐。
        - residue_id: int64 数组，(N_atom,)；逐模板原子所属 components.index，数值位于 [0, 512)，用于模型残基嵌入和分组参考坐标。
        - node_continuous: float32 数组，(N_atom, 15)；依次拼接 chirality 7 维、3/4/5/6 元环 4 维标志、LigandObject 模板 is_present 1 维和组内中心化参考 XYZ 3 维；来源 schema 定义 chirality 为独热、is_present 固定为 False，Matcher 读取器只转换 dtype，不额外核验这两项来源约束。
        - node_continuous[:, 0:7]: float32 数组，(N_atom, 7)；逐值转换 LigandObject.atoms.chirality；来源 schema 定义为七种手性类别独热，Matcher 读取器不额外核验每个原子恰有一列为 True。
        - node_continuous[:, 7:11]: float32 数组，(N_atom, 4)；逐值转换 LigandObject.atoms.in_ring 的 3/4/5/6 元环标志。
        - node_continuous[:, 11]: float32 数组，(N_atom,)；把 LigandObject.atoms.is_present 转成 float32；来源 schema 定义该模板字段固定为 False，Matcher 读取器不额外核验全为 False；它不是 occurrence 的 present_{candidate_id} 掩码，也不定义联合身份。
        # NOTE: 在图网络中，似乎很少有人把中心化坐标当作特征？至少本地的PocketXmol仓库似乎并非如此。
        - node_continuous[:, 12:15]: float32 数组，(N_atom, 3)；每个 residue_id 原子组分别用 ref_pos 减组内质心再除以 10 Å，最后一维依次为 X、Y、Z。
        - edge_index: int64 数组，(2, N_directed_edge)；第一轴依次为源端点和目标端点，数值索引 element、charge、residue_id 与 node_continuous 的第一维；先保存全部原始化学键的 atom_1→atom_2，再保存全部反向边，因此第 b 条原始键的反向边位于 b + N_bond。
        - edge_features: float32 数组，(N_directed_edge, 10)；依次拼接键型 5 维、3/4/5/6 元环 4 维标志和参考距离除以 10 Å 的 1 维数值；来源 schema 定义键型为独热，Matcher 读取器只转换 dtype，不额外核验每条键恰有一列为 True。
        - edge_features[:, 0:5]: float32 数组，(N_directed_edge, 5)；逐值转换单键、双键、三键、配位键和芳香键字段，正反边共享同一值；来源 schema 定义该字段为独热，Matcher 读取器不额外核验独热约束。
        - edge_features[:, 5:9]: float32 数组，(N_directed_edge, 4)；化学键是否位于 3/4/5/6 元环的标志，正反边共享同一值。
        - edge_features[:, 9]: float32 数组，(N_directed_edge,)；同一 residue_id 内使用 ref_pos 欧氏距离除以 10 Å，不同 residue_id 间保留边拓扑和键型但该距离通道为 0，因为各 CCD 组件参考坐标系彼此独立。
    """

    element: np.ndarray
    charge: np.ndarray
    residue_id: np.ndarray
    node_continuous: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray


class LigandGraphStore:
    """
    按 object_key 读取配体模板图，并在单个 DataLoader worker 的独立字节预算内缓存。

    构造参数:
        - data_root: Path；AdaLigand Ori_Data 根目录，配体模板文件位于 data_root/ligand_objects。
        - cache_bytes: int；当前进程图缓存的最大字节数，默认 512 MiB，0 表示每次 put 后都不能长期保留图。

    缓存字段:
        - _cache: ByteBudgetLru[str, LigandGraph]；键是未经文件名转换的 object_key，值是已验证 LigandGraph，字节统计递归包含全部 NumPy 字段。
        - _cache.value.element: int64 数组，(N_atom,)；模板原子序数。
        - _cache.value.charge: int64 数组，(N_atom,)；模板原子形式电荷。
        - _cache.value.residue_id: int64 数组，(N_atom,)；模板内部残基编号。
        - _cache.value.node_continuous: float32 数组，(N_atom, 15)；节点连续特征。
        - _cache.value.edge_index: int64 数组，(2, 2*N_bond)；双向边端点。
        - _cache.value.edge_features: float32 数组，(2*N_bond, 10)；与边端点逐边对齐的边特征。

    本类读取 LigandObject NPZ；不读取 occurrence 语言表示、沉积坐标或 present 掩码，不写入来源文件。
    """

    def __init__(self, data_root: Path, cache_bytes: int = 512 * 1024**2) -> None:
        """
        建立按 object_key 读取 LigandObject 模板图的字节预算缓存。

        输入参数:
            - data_root: Path；冻结 A–G 数据根，模板图位于 data_root/ligand_objects/。
            - cache_bytes: int；当前进程 LigandGraph LRU 的最大估算字节数，默认 512 MiB，必须非负；0 表示 put 后不能长期保留条目。

        实例字段:
            - data_root: Path；原样保存的冻结数据根。
            - cache_bytes: int；规范化为内建 int 的缓存预算。
            - _cache: ByteBudgetLru[str, LigandGraph]；键为原始 object_key，值为 _load_ligand_graph 按现有字段、名称、坐标、残基和端点门禁读取并转换的模板图。
            - _cache.value.element: int64 数组，(N_atom,)；模板原子序数。
            - _cache.value.charge: int64 数组，(N_atom,)；模板原子形式电荷。
            - _cache.value.residue_id: int64 数组，(N_atom,)；模板内部残基编号。
            - _cache.value.node_continuous: float32 数组，(N_atom, 15)；模板节点连续特征。
            - _cache.value.edge_index: int64 数组，(2, 2*N_bond)；模板双向边端点。
            - _cache.value.edge_features: float32 数组，(2*N_bond, 10)；模板双向边特征。
        """

        if cache_bytes < 0:
            raise ValueError("cache_bytes 不能为负数。")
        self.data_root = data_root
        self.cache_bytes = int(cache_bytes)
        self._cache: ByteBudgetLru[str, LigandGraph] = ByteBudgetLru(self.cache_bytes)

    def load(self, object_key: str) -> LigandGraph:
        """
        返回经过字段、名称和边端点验证的双向 LigandObject 图。

        输入参数:
            - object_key: str；清单联合身份中的模板图键，用于缓存键和 LigandObject 文件名转换。

        返回值:
            - graph: LigandGraph；相同 object_key 在当前 worker 缓存命中时返回同一个 dataclass 实例；frozen 仅禁止字段重绑，内部 NumPy 数组未设置写保护。
            - graph.element: int64 数组，(N_atom,)；模板原子序数。
            - graph.charge: int64 数组，(N_atom,)；与 element 逐原子对齐的形式电荷。
            - graph.residue_id: int64 数组，(N_atom,)；数值位于 [0, 512) 的模板内部残基编号。
            - graph.node_continuous: float32 数组，(N_atom, 15)；依次为手性 7 维、环标志 4 维、模板 is_present 1 维和组内中心化参考 XYZ 3 维；来源 schema 的手性独热和 is_present 固定 False 约束不会由本读取器额外核验。
            - graph.node_continuous[:, 0:7]: float32 数组，(N_atom, 7)；由来源手性字段逐值转换；来源 schema 定义为七种类别独热，本读取器不额外核验恰有一列为 True。
            - graph.node_continuous[:, 7:11]: float32 数组，(N_atom, 4)；3/4/5/6 元环标志。
            - graph.node_continuous[:, 11]: float32 数组，(N_atom,)；由模板 atoms.is_present 转成 float32；来源 schema 定义其固定为 False，本读取器不额外核验全为 False；它不是 occurrence 的 present 掩码。
            - graph.node_continuous[:, 12:15]: float32 数组，(N_atom, 3)；每个 residue_id 组内中心化并除以 10 Å 的参考 X/Y/Z。
            - graph.edge_index: int64 数组，(2, 2*N_bond)；先保存全部 atom_1→atom_2，再保存全部反向边。
            - graph.edge_features: float32 数组，(2*N_bond, 10)；依次为键型 5 维、环标志 4 维和参考距离除以 10 Å；来源 schema 的键型独热约束不会由本读取器额外核验。
            - graph.edge_features[:, 0:5]: float32 数组，(2*N_bond, 5)；由单/双/三/配位/芳香键字段逐值转换；来源 schema 定义为独热，本读取器不额外核验恰有一列为 True。
            - graph.edge_features[:, 5:9]: float32 数组，(2*N_bond, 4)；3/4/5/6 元环标志。
            - graph.edge_features[:, 9]: float32 数组，(2*N_bond,)；同 residue_id 边的参考距离除以 10 Å，跨 residue_id 边为 0。

        文件输入:
            - ligand_objects/{safe_object_filename(object_key)}.npz: NPZ；当前 object_key 对应的只读 LigandObject 模板图。
            - ligand_objects/{safe_object_filename(object_key)}.npz/name: Unicode 标量；必须逐字等于 object_key。
            - ligand_objects/{safe_object_filename(object_key)}.npz/atoms: 一维结构化数组，(N_atom,)；必须非空。
            - ligand_objects/{safe_object_filename(object_key)}.npz/atoms.element: 整数标量字段，(N_atom,)；模板原子序数。
            - ligand_objects/{safe_object_filename(object_key)}.npz/atoms.charge: 整数标量字段，(N_atom,)；模板形式电荷。
            - ligand_objects/{safe_object_filename(object_key)}.npz/atoms.ref_pos: float32 子数组字段，(N_atom, 3)；模板参考 XYZ 坐标，单位 Å。
            - ligand_objects/{safe_object_filename(object_key)}.npz/atoms.is_present: bool 标量字段，(N_atom,)；来源 schema 定义为固定 False，本读取器只转成 float32 而不核验全为 False；它不是 occurrence 掩码。
            - ligand_objects/{safe_object_filename(object_key)}.npz/atoms.chirality: bool 子数组字段，(N_atom, 7)；来源 schema 定义为手性类别独热，本读取器只转换 dtype 而不核验每个原子恰有一列为 True。
            - ligand_objects/{safe_object_filename(object_key)}.npz/atoms.in_ring: bool 子数组字段，(N_atom, 4)；3/4/5/6 元环标志。
            - ligand_objects/{safe_object_filename(object_key)}.npz/atoms.residue_id: int32 标量字段，(N_atom,)；模板内部残基编号。
            - ligand_objects/{safe_object_filename(object_key)}.npz/bonds: 一维结构化数组，(N_bond,)；允许为空。
            - ligand_objects/{safe_object_filename(object_key)}.npz/bonds.atom_1: 整数标量字段，(N_bond,)；第一端点索引。
            - ligand_objects/{safe_object_filename(object_key)}.npz/bonds.atom_2: 整数标量字段，(N_bond,)；第二端点索引。
            - ligand_objects/{safe_object_filename(object_key)}.npz/bonds.type: bool 子数组字段，(N_bond, 5)；来源 schema 定义为键型独热，本读取器只转换 dtype 而不核验每条键恰有一列为 True。
            - ligand_objects/{safe_object_filename(object_key)}.npz/bonds.in_ring: bool 子数组字段，(N_bond, 4)；化学键环标志。
        """

        # LigandGraph 或 None；缓存键保留原始 object_key，不使用安全文件名，避免文件名转换掩盖键语义。
        cached = self._cache.get(object_key)
        if cached is not None:
            return cached
        # Path, 当前 object_key 对应的只读 LigandObject NPZ 路径。
        path = self.data_root / "ligand_objects" / f"{safe_object_filename(object_key)}.npz"
        graph = _load_ligand_graph(path, object_key)
        self._cache.put(object_key, graph)
        return graph

    def __getstate__(self) -> dict[str, object]:
        """
        为 DataLoader spawn 序列化实例状态，并把父进程缓存替换为空缓存。

        返回值:
            - state: dict[str, object]；可由 DataLoader spawn 序列化的实例状态。
            - state.data_root: Path；与实例 data_root 相同的冻结数据根。
            - state.cache_bytes: int；与实例 cache_bytes 相同的字节预算。
            - state._cache: ByteBudgetLru[str, LigandGraph]；使用 state.cache_bytes 新建的空缓存，不包含父进程图数组。

        该方法防止父进程已加载的图数组被复制进每个 worker；各 worker 随后独立填充缓存。
        """

        # dict[str, object], 当前实例属性的浅拷贝；替换 _cache 不修改父进程实例。
        state = self.__dict__.copy()
        state["_cache"] = ByteBudgetLru(self.cache_bytes)
        return state


def _load_ligand_graph(path: Path, object_key: str) -> LigandGraph:
    """
    从一个 LigandObject NPZ 的结构化数组建立节点字段和双向化学键字段。

    输入参数:
        - path: Path；只读 LigandObject NPZ 文件路径。
        - object_key: str；期望的模板键，必须与 NPZ 顶层 name 标量字符串完全相同。

    文件字段:
        - name: 标量字符串；必须等于 object_key。
        - atoms: 一维结构化数组，(N_atom,)；N_atom 必须大于 0，至少包含 element、charge、ref_pos、is_present、chirality、in_ring 和 residue_id。
        - atoms.element: 整数标量字段，(N_atom,)；逐模板原子的原子序数。
        - atoms.charge: 整数标量字段，(N_atom,)；逐模板原子的形式电荷。
        - atoms.ref_pos: float32 子数组字段，(N_atom, 3)；CCD 或 RDKit 参考构象 XYZ，单位 Å，不是 occurrence 沉积坐标。
        - atoms.is_present: bool 标量字段，(N_atom,)；来源 schema 定义为固定 False，本模块用 np.asarray 转成 float32 连续特征但不核验全为 False；它不读取 ligand_coords.npz 的 occurrence 掩码。
        - atoms.chirality: bool 子数组字段，(N_atom, 7)；来源 schema 定义为七种手性类别独热，本模块只转换 dtype，不核验每个原子恰有一列为 True。
        - atoms.in_ring: bool 子数组字段，(N_atom, 4)；依次表示原子是否位于 3、4、5、6 元环。
        - atoms.residue_id: int32 标量字段，(N_atom,)；原子所属 components.index，本模块要求数值位于 [0, 512)。
        - bonds: 一维结构化数组，(N_bond,)；允许 N_bond 为 0，至少包含 atom_1、atom_2、type 和 in_ring。
        - bonds.atom_1: 整数标量字段，(N_bond,)；第一端点索引，指向 atoms 第一维。
        - bonds.atom_2: 整数标量字段，(N_bond,)；第二端点索引，指向 atoms 第一维。
        - bonds.type: bool 子数组字段，(N_bond, 5)；依次表示单键、双键、三键、配位键和芳香键；来源 schema 定义为独热，本模块只转换 dtype，不核验每条键恰有一列为 True。
        - bonds.in_ring: bool 子数组字段，(N_bond, 4)；依次表示化学键是否位于 3、4、5、6 元环。

    返回值: # NOTE: 点与边的特征构造的有问题，存在缺失（如残基类型、主链标志等都没拼全）且似乎多拼接了坐标，需仔细订正。
        - graph: LigandGraph；节点顺序保持 atoms 顺序，每条 bonds 记录扩展为正向和反向两条边。
        - graph.element: int64 数组，(N_atom,)；由 atoms.element 逐值转换的原子序数。
        - graph.charge: int64 数组，(N_atom,)；由 atoms.charge 逐值转换的形式电荷。
        - graph.residue_id: int64 数组，(N_atom,)；由 atoms.residue_id 逐值转换，数值位于 [0, 512)。
        - graph.node_continuous: float32 数组，(N_atom, 15)；手性 7 维、环标志 4 维、模板 is_present 1 维和组内中心化参考 XYZ 3 维；来源 schema 的手性独热和 is_present 固定 False 约束不会由本读取器额外核验。
        - graph.node_continuous[:, 0:7]: float32 数组，(N_atom, 7)；由 atoms.chirality 逐值转换；来源 schema 定义为手性独热，本读取器不额外核验独热约束。
        - graph.node_continuous[:, 7:11]: float32 数组，(N_atom, 4)；由 atoms.in_ring 转换的 3/4/5/6 元环标志。
        - graph.node_continuous[:, 11]: float32 数组，(N_atom,)；由模板 atoms.is_present 转成 float32；来源 schema 定义其固定为 False，本读取器不额外核验全为 False；它不是 occurrence 掩码。
        - graph.node_continuous[:, 12:15]: float32 数组，(N_atom, 3)；每个 residue_id 组内中心化并除以 10 Å 的参考 X/Y/Z。
        - graph.edge_index: int64 数组，(2, 2*N_bond)；先保存全部 atom_1→atom_2，再保存全部 atom_2→atom_1。
        - graph.edge_features: float32 数组，(2*N_bond, 10)；键型 5 维、环标志 4 维和参考距离除以 10 Å，正反边共享对应原始键特征；来源 schema 的键型独热约束不会由本读取器额外核验。
        - graph.edge_features[:, 0:5]: float32 数组，(2*N_bond, 5)；由单/双/三/配位/芳香键字段逐值转换；来源 schema 定义为独热，本读取器不额外核验独热约束。
        - graph.edge_features[:, 5:9]: float32 数组，(2*N_bond, 4)；3/4/5/6 元环标志。
        - graph.edge_features[:, 9]: float32 数组，(2*N_bond,)；同 residue_id 边的参考距离除以 10 Å，跨 residue_id 边为 0。
    """

    with np.load(path, allow_pickle=False) as archive:
        # set[str]，LigandObject 顶层必须包含的模板名称、原子表和化学键表字段。
        required = {"atoms", "bonds", "name"}
        # list[str]，当前 NPZ 缺少的顶层必要字段，按字典序报告。
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"{path} 缺少 LigandObject 字段 {missing}。")
        # 结构化数组，(N_atom,), 拷贝后可在关闭 NPZ archive 后继续读取全部原子子字段。
        atoms = archive["atoms"].copy()
        # 结构化数组，(N_bond,), 拷贝后可在关闭 NPZ archive 后继续读取全部化学键子字段。
        bonds = archive["bonds"].copy()
        # str，NPZ 顶层 Unicode 标量保存的模板名称，必须与请求 object_key 逐字一致。
        actual_name = str(archive["name"].item())
    if actual_name != object_key:
        raise ValueError(f"{path}:name={actual_name!r}，期望 {object_key!r}。")
    # set[str], Matcher 节点离散字段和 15 维连续字段依赖的全部 atoms 子字段。
    atom_fields = {"element", "charge", "ref_pos", "is_present", "chirality", "in_ring", "residue_id"}
    # set[str], 双向边端点和 10 维边特征依赖的全部 bonds 子字段。
    bond_fields = {"atom_1", "atom_2", "type", "in_ring"}
    if atoms.dtype.names is None or not atom_fields.issubset(atoms.dtype.names):
        raise ValueError(f"{path}:atoms 缺少字段 {sorted(atom_fields)}。")
    if bonds.dtype.names is None or not bond_fields.issubset(bonds.dtype.names):
        raise ValueError(f"{path}:bonds 缺少字段 {sorted(bond_fields)}。")
    atom_count = int(atoms.shape[0])
    if atoms.ndim != 1 or atom_count == 0 or bonds.ndim != 1:
        raise ValueError(f"{path}:atoms 必须为非空一维数组，bonds 必须为一维数组。")

    # float32, (N_atom, 3), CCD 或 RDKit 模板参考 XYZ 坐标，单位 Å；不代表当前 PDB occurrence 的沉积位置。
    reference_position = np.asarray(atoms["ref_pos"], dtype=np.float32)
    if reference_position.shape != (atom_count, 3) or not np.isfinite(reference_position).all():
        raise ValueError(f"{path}:atoms.ref_pos 必须为有限 (N,3)。")
    # int64, (N_atom,), 与原子字段逐位置对齐的 components.index；后续分别中心化每个 residue_id 原子组。
    residue_id = np.asarray(atoms["residue_id"], dtype=np.int64)
    if np.any(residue_id < 0) or np.any(residue_id >= 512):
        raise ValueError(f"{path}:atoms.residue_id 必须位于 [0,512)。")
    # float32, (N_atom, 3), 每个 residue_id 原子组分别减去组内 XYZ 质心并除以 10 Å。
    centered_position = np.empty_like(reference_position)
    for current_residue_id in np.unique(residue_id):
        # bool, (N_atom,), True 表示模板原子属于当前 residue_id 原子组。
        residue_mask = residue_id == current_residue_id
        centered_position[residue_mask] = (
            reference_position[residue_mask]
            - reference_position[residue_mask].mean(axis=0, keepdims=True)
        ) / 10.0
    # float32, (N_atom, 15), [chirality 7, in_ring 4, template is_present 1, grouped reference XYZ 3]；只转换来源值，不核验 chirality 独热或 is_present 固定 False。
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
        # int64, (N_bond,), 原始无向化学键第一端点；数值索引 atoms 第一维。
        source = np.asarray(bonds["atom_1"], dtype=np.int64)
        # int64, (N_bond,), 原始无向化学键第二端点；与 source 逐键对齐并索引 atoms 第一维。
        target = np.asarray(bonds["atom_2"], dtype=np.int64)
        if np.any(source < 0) or np.any(target < 0) or np.any(source >= atom_count) or np.any(target >= atom_count):
            raise ValueError(f"{path}:bonds 含越界原子端点。")
        # int64, (2, N_bond), bonds 中每条原始边的第一、第二端点。
        undirected = np.stack((source, target), axis=0)
        # int64, (2, 2*N_bond), 先保存全部 source→target，再保存全部 target→source。
        edge_index = np.concatenate((undirected, undirected[::-1]), axis=1)
        # float32, (N_bond, 5), 单/双/三/配位/芳香键字段；来源 schema 定义为独热，本读取器只转换 dtype。
        bond_type = np.asarray(bonds["type"], dtype=np.float32)
        # float32, (N_bond, 4), 3/4/5/6 元环标志，与 bond_type 第一维逐键对齐。
        bond_ring = np.asarray(bonds["in_ring"], dtype=np.float32)
        # float32, (N_bond,), 每条原始边的参考距离，跨 residue_id 的边保持为 0。
        distance = np.zeros(source.shape[0], dtype=np.float32)
        # bool, (N_bond,), True 表示原始化学键两端属于同一 residue_id 原子组，其 ref_pos 距离可比较。
        distance_valid = residue_id[source] == residue_id[target]
        distance[distance_valid] = np.linalg.norm(
            reference_position[source[distance_valid]]
            - reference_position[target[distance_valid]],
            axis=1,
        )
        # float32, (N_bond, 10), 单方向边特征；最后一维参考距离已经除以 10 Å。
        one_direction = np.concatenate((bond_type, bond_ring, (distance / 10.0)[:, None]), axis=1)
        # float32, (2*N_bond, 10), 与 edge_index 第二维对齐，反向边复制对应原始边特征。
        edge_features = np.concatenate((one_direction, one_direction), axis=0)
    else:
        # int64, (2, 0), 无键 LigandObject 的空双向端点数组。
        edge_index = np.empty((2, 0), dtype=np.int64)
        # float32, (0, 10), 与空 edge_index 对齐的边特征数组。
        edge_features = np.empty((0, 10), dtype=np.float32)

    return LigandGraph(
        element=np.asarray(atoms["element"], dtype=np.int64),
        charge=np.asarray(atoms["charge"], dtype=np.int64),
        residue_id=residue_id,
        node_continuous=node_continuous.astype(np.float32, copy=False),
        edge_index=edge_index,
        edge_features=edge_features.astype(np.float32, copy=False),
    )
