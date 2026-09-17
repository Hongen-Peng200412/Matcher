"""定义 Matcher 的固定类别、输入维度和正式数据根目录.

主要入口:
- SourceRoots: 连接 AdaLigand 原始产物与 Find_1 centered 产物的只读根目录.

本模块不读写文件. 新正式管线只有一种身份、语言表示和裁剪契约, 因此不再暴露
schema 数字、语言家族、身份范围或局部裁剪策略开关.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# str tuple, 长度 6; 只用于真值分层, 模型前景头不预测这六类.
CLASS_NAMES = (
    "background",
    "ion",
    "nucleotide_like",
    "peptide_like",
    "small_molecule",
    "sugar",
)
# str tuple, 长度 5; 真实前景候选报告使用的固定类别顺序.
FOREGROUND_CLASS_NAMES = CLASS_NAMES[1:]
# str, manifest.json 顶层 schema 的唯一允许值.
MANIFEST_SCHEMA = "matcher_stage1_manifest_v1"

# str, 每份身份语言归档的 model_name 必须匹配该公开模型名称.
SMI_TED_MODEL_NAME = "SMI-TED Light 289M"
# int, 每个精确 SMILES 的 SMI-TED 浮点表示宽度.
LANGUAGE_EMBEDDING_DIM = 768

# int, Find_1 centered 中每个候选完整体素框的单轴体素数.
STAGE1_BOX_SIZE = 80
# int, supplied crop 和 Matcher 区域预测的单轴体素数.
LOCAL_BOX_SIZE = 48
# int, 体素输入通道数; 56 个密度通道与 50 个受体通道连接得到 106.
VOXEL_INPUT_CHANNELS = 106
# int, A token 宽度; 50、64、64、48 维层级特征与概率、相对 XYZ 连接得到 230.
A_TOKEN_DIM = 230
# int, P token 宽度; 64、48 维层级特征与概率、相对 XYZ 连接得到 116.
P_TOKEN_DIM = 116
# int, V token 宽度; 64 维体素特征与概率、相对 XYZ 连接得到 68.
V_TOKEN_DIM = 68
# int, PP token 宽度; 依次为 Stage1 概率与相对 X、Y、Z.
PP_TOKEN_DIM = 4

# str tuple, 长度 12; 按 A、P、V、PP 分组, 每组依次为 token 数、概率均值和概率最大值.
CANDIDATE_MANUAL_FEATURE_NAMES = tuple(
    f"{modality}_{statistic}"
    for modality in ("A", "P", "V", "PP")
    for statistic in ("count", "probability_mean", "probability_max")
)
# int, candidate_manual_features 的固定最后一轴宽度.
CANDIDATE_MANUAL_FEATURE_DIM = len(CANDIDATE_MANUAL_FEATURE_NAMES)
# str tuple, 长度 4; candidate_sampling_features 四个分量的固定顺序.
CANDIDATE_SAMPLING_FEATURE_NAMES = (
    "crop_boundary_limited",
    "crop_center_to_v_centroid_l2_voxels",
    "crop_center_to_v_centroid_linf_voxels",
    "v_retained_fraction",
)
# int, candidate_sampling_features 的固定最后一轴宽度.
CANDIDATE_SAMPLING_FEATURE_DIM = len(CANDIDATE_SAMPLING_FEATURE_NAMES)
# float, 完整 80 立方体的半边长, 单位为体素; 用于归一化 A/P 相对坐标.
FULL_CONTEXT_HALF_EXTENT_VOXELS = STAGE1_BOX_SIZE / 2.0
# float, supplied 48 立方体的半边长, 单位为体素; 用于归一化 V/PP 相对坐标.
LOCAL_CONTEXT_HALF_EXTENT_VOXELS = LOCAL_BOX_SIZE / 2.0


@dataclass(frozen=True)
class SourceRoots:
    """
    保存新 Matcher 扫描清单时读取的两个显式根目录.

    构造参数:
    - data_root: AdaLigand Ori_Data 根目录, 如 `/storage/penghongen/AdaLigand/Ori_Data`.
    - stage1_root: Find_1 的 `artifacts/Find_1` 根目录, 其下按 validation 等划分保存 PDB.

    派生路径:
    - prepared_root: PreparedLigand 顶层 SMILES 清单的共同父目录.
    - language_root: SMI-TED 289M occurrence 向量的共同父目录.
    """
    data_root: Path
    stage1_root: Path

    @property
    def prepared_root(self) -> Path:
        """
        返回按 PDB 保存 PreparedLigand 顶层字段的目录.

        - return: data_root 下固定的 ligand_language_models/prepared 路径.
        """
        return self.data_root / "stage1_preparation_box_pool_2/ligand_language_models/prepared"

    @property
    def language_root(self) -> Path:
        """
        返回按 PDB 和 occurrence 保存 SMI-TED 289M 向量的目录.

        - return: data_root 下固定的 ligand_language_models/smi_ted_289m 路径.
        """
        return self.data_root / "stage1_preparation_box_pool_2/ligand_language_models/smi_ted_289m"
