"""定义两个 Matcher Dataset 与模型共同使用的内存契约。

`PocketInput` 是 Matcher 与未来 Stage3 共用的候选受体环境；A/B/O 标签、配体
slot 和匈牙利匹配只存在于 `MatcherSample`，不会进入 Stage3 需要读取的接口。
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from matcher.graph import MolecularGraph


@dataclass(frozen=True)
class RawGraph:
    """一个分子或受体区域的固定图输入。

    `node_input` 为 `[N,F]`；`coordinates` 为世界或参考构象 XYZ `[N,3]`，单位 Å；
    `element` 为原子序数 `[N]`；`binding_atom` 是只供受体辅助监督使用的可选布尔标签。
    """

    node_input: Tensor
    coordinates: Tensor
    graph: MolecularGraph
    element: Tensor
    binding_atom: Tensor | None = None


@dataclass(frozen=True)
class LigandInput:
    """一个 PDB 内按 `object_key` 去重的配体模板图。"""

    object_key: str
    graph: RawGraph


@dataclass(frozen=True)
class Stage1Context:
    """一个预测候选的 Stage1 专属 V、P 和概率派生 PP。

    V/P/PP 坐标均为当前 48³ BOX 的局部 XYZ，单位 Å。`V_feature`、`P_feature`
    的末维由 Stage1 模型来源决定；`PP_probability` 只决定采样身份，PP 的模型
    初始表示由密度 U-Net 在 `PP_coord_local_xyz` 处采样得到。
    """

    V_coord_local_xyz: Tensor
    V_probability: Tensor
    V_feature: Tensor
    P_coord_local_xyz: Tensor
    P_probability: Tensor
    P_feature: Tensor
    PP_coord_local_xyz: Tensor
    PP_probability: Tensor
    A_probability: Tensor
    A_feature: Tensor


@dataclass(frozen=True)
class PocketInput:
    """一个候选受体环境，也是 Stage3 可以直接消费的模型无关边界。

    `density` 为 `[C,48,48,48]`，空间轴依次为 ZYX；`map_start_zyx` 是该 48³
    裁剪在完整实验图中的整数起点；`candidate_mask_zyx` 是完整图 ZYX 稀疏索引。
    训练旋转只改变 `density`，`rotation_axes_zyx/rotation_k` 记录怎样由原裁剪得到
    当前张量，供以后需要从 U-Net 特征恢复空间点时使用。
    """

    center_xyz: Tensor
    density: Tensor
    map_start_zyx: Tensor
    voxel_size_xyz: Tensor
    origin_xyz: Tensor
    A_graph: RawGraph
    A_global_index: Tensor
    candidate_mask_zyx: Tensor
    rotation_axes_zyx: tuple[int, int] | None
    rotation_k: int
    stage1: Stage1Context | None = None


@dataclass(frozen=True)
class MatcherSample:
    """一个完整 PDB 的配体 slot、候选环境和六个监督矩阵。

    六个标签均为 `[N_candidate,N_occurrence]`。`occurrence_coordinates` 与
    `occurrence_present` 按 occurrence 顺序保存模板原子的真实坐标和存在掩码，
    只供细粒度辅助监督使用。
    """

    pdb_id: str
    manifest_index: int
    ligands: tuple[LigandInput, ...]
    pockets: tuple[PocketInput, ...]
    occurrence_to_ligand: Tensor
    occurrence_candidate_id: Tensor
    occurrence_centroid_xyz: Tensor
    occurrence_coordinates: tuple[Tensor, ...]
    occurrence_present: tuple[Tensor, ...]
    A_target: Tensor
    B_target: Tensor
    O_target: Tensor
    A_prime_target: Tensor
    B_prime_target: Tensor
    O_prime_target: Tensor

    @property
    def num_occurrences(self) -> int:
        return int(self.occurrence_candidate_id.numel())

    @property
    def num_candidates(self) -> int:
        return len(self.pockets)

    @property
    def candidates(self) -> tuple[PocketInput, ...]:
        """兼容已验证的 Matcher 算子使用的候选命名。"""

        return self.pockets

    @property
    def ligand_gt_coordinates(self) -> tuple[Tensor, ...]:
        """返回细分支监督所需的真实配体坐标。"""

        return self.occurrence_coordinates

    @property
    def ligand_gt_present(self) -> tuple[Tensor, ...]:
        """返回细分支监督所需的真实原子存在掩码。"""

        return self.occurrence_present


@dataclass(frozen=True)
class MatcherBatch:
    """一个 optimizer step 的 PDB 样本及可批量执行的共同输入。

    `density` 沿所有 PDB 的候选轴堆叠；`candidate_ptr`、`ligand_identity_ptr`
    分别恢复每个 PDB 的候选和去重配体范围。图仍以 Python tuple 保存，模型只在
    一次前向开始时把它们拼成互不连边的大图。
    """

    samples: tuple[MatcherSample, ...]
    density: Tensor
    ligand_graphs: tuple[RawGraph, ...]
    A_graphs: tuple[RawGraph, ...]
    candidate_ptr: tuple[int, ...]
    ligand_identity_ptr: tuple[int, ...]
