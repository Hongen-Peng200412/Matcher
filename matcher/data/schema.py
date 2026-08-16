"""
集中定义 Matcher 数据适配层共享的稳定名称、维度和根目录派生规则。

主要入口:
    - SourceRoots: 保存三个来源根目录，并派生语言表示、清单、体素缓存和 PP token 缓存目录。
    - identity_scope_class_indices: 把身份监督范围转换成 CLASS_NAMES 的类别下标。
    - safe_object_filename: 把 object_key 转成 LigandObject NPZ 的安全文件名主体。

本模块只返回常量、类别下标和 Path，不读取或写入文件。MANIFEST_SCHEMA_VERSION、VOXEL_CACHE_SCHEMA_VERSION 与 PP_CACHE_SCHEMA_VERSION 分别进入清单和缓存目录名，防止不同字段契约共用同一路径。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


# int, 监督清单 schema 版本；v4 以 object_key 与源自 PreparedLigand 顶层 smiles 的 prepared_smiles 共同定义身份。
MANIFEST_SCHEMA_VERSION = 4
# int, 候选级 105 通道体素缓存格式版本；目录名由 SourceRoots.voxel_cache_root 生成。
VOXEL_CACHE_SCHEMA_VERSION = 2
# int, 候选级 PP token 缓存格式版本；目录名还绑定 top_k 与局部裁剪策略。
PP_CACHE_SCHEMA_VERSION = 2

# tuple[str, ...], 长度 6；下标就是六分类标签，0 固定为背景类，其余顺序不得脱离 checkpoint 和指标单独修改。
CLASS_NAMES = (
    "background",
    "ion",
    "nucleotide_like",
    "peptide_like",
    "small_molecule",
    "sugar",
)
# dict[str, int], 把 CLASS_NAMES 中的类别名称映射为同一位置的六分类下标。
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
# int, small_molecule 在 CLASS_NAMES 中的固定下标。
SMALL_MOLECULE_INDEX = CLASS_TO_INDEX["small_molecule"]
# tuple[str, ...], 长度 5；依次保留 CLASS_NAMES 中除 background 外的全部类别名称。
FOREGROUND_CLASS_NAMES = CLASS_NAMES[1:]
# tuple[int, ...], 长度 5；与 FOREGROUND_CLASS_NAMES 逐位置对齐的六分类下标。
FOREGROUND_CLASS_INDICES = tuple(CLASS_TO_INDEX[name] for name in FOREGROUND_CLASS_NAMES)
# 身份监督范围类型；small_molecule 只监督小分子，all_non_background 监督五个非背景类别。
IdentityScope = Literal["small_molecule", "all_non_background"]
# tuple[str, ...], load_experiment_plan 接受的全部 identity_scope 枚举值。
IDENTITY_SCOPES = ("small_molecule", "all_non_background")
# tuple[str, str], schema v4 联合身份字段顺序；两个字段都相同的 occurrence 才共享一个身份槽。
IDENTITY_KEY_FIELDS = ("object_key", "prepared_smiles")
# PreparedLigand 顶层 smiles 的可用状态；状态与 prepared_smiles 值共同写入候选清单和 metadata。
PreparedSmilesStatus = Literal[
    "available",
    "explicit_null",
    "missing_file",
    "not_applicable",
]
# tuple[str, ...], CandidateDataset 接受的全部 prepared_smiles_status 枚举值。
PREPARED_SMILES_STATUSES = (
    "available",
    "explicit_null",
    "missing_file",
    "not_applicable",
)
# 48³ 局部裁剪策略；fixed_center 使用 80³ 中心，v_centroid 以候选 V 体素质心请求裁剪中心。
LocalCropStrategy = Literal["fixed_center", "v_centroid"]
# tuple[str, str], 配置和缓存路径接受的全部局部裁剪策略。
LOCAL_CROP_STRATEGIES = ("fixed_center", "v_centroid")
# dict[str, str], occurrence type_tag 到六分类名称的映射；other 明确归入 background。
OCCURRENCE_TYPE_TO_CLASS = {
    "ion": "ion",
    "nucleotide_like": "nucleotide_like",
    "peptide_like": "peptide_like",
    "small_molecule": "small_molecule",
    "sugar": "sugar",
    "other": "background",
}

# tuple[str, str], 正式实验支持的两种冻结配体语言表示家族。
LANGUAGE_FAMILIES = ("molformer", "smi_ted_289m")
# dict[str, str], 语言表示家族到来源模型公开名称的映射；键与 LANGUAGE_FAMILIES 一致。
LANGUAGE_MODEL_NAMES = {
    "molformer": "ibm-research/MoLFormer-XL-both-10pct",
    "smi_ted_289m": "SMI-TED Light 289M",
}

# int, Stage1 每个候选来源立方体的单轴体素数，三个空间轴均为 80。
STAGE1_BOX_SIZE = 80
# int, Matcher 从 Stage1 立方体读取的局部裁剪单轴体素数，三个空间轴均为 48。
LOCAL_BOX_SIZE = 48
# int, fixed_center 时从 80³ 每个低坐标面去除的体素数；(80 - 48) / 2 = 16。
LOCAL_BOX_MARGIN = (STAGE1_BOX_SIZE - LOCAL_BOX_SIZE) // 2
# int, MoLFormer 与 SMI-TED occurrence 级冻结语言向量的共同宽度。
LANGUAGE_EMBEDDING_DIM = 768
# float, 世界坐标单位 Å；允许 float32 重算的候选框原点与来源原点之间最多相差 5e-5 Å。
BOX_ORIGIN_ABSOLUTE_TOLERANCE_ANGSTROM = 5e-5

# A/P/V/PP token 的最后四个特征分量依次是 Stage1 固化概率、relative_x、relative_y、relative_z；概率必须有限且位于 [0, 1]。
# A/P 相对 XYZ 以候选实际裁剪中心为原点，并按完整 80³ 物理半边长归一化。
# V/PP 相对 XYZ 以候选实际裁剪中心为原点，并按候选级 48³ 物理半边长归一化。
# int, A token 宽度；前 225 个特征分量由受体原子来源字段构成，最后 4 个特征分量遵守上面的共同顺序。
A_TOKEN_DIM = 229
# int, P token 宽度；前 112 个特征分量由 Stage1 P token 来源字段构成，最后 4 个特征分量遵守上面的共同顺序。
P_TOKEN_DIM = 116
# int, V token 宽度；前 48 个特征分量由 Stage1 V token 来源字段构成，最后 4 个特征分量遵守上面的共同顺序。
V_TOKEN_DIM = 52
# int, PP token 宽度；只包含上面定义的概率和三个归一化相对坐标。
PP_TOKEN_DIM = 4
# tuple[str, ...], 长度 12；按 A、P、V、PP 模态排列，每个模态依次记录 token 数、概率均值和概率最大值。
CANDIDATE_MANUAL_FEATURE_NAMES = tuple(
    f"{modality}_{statistic}"
    for modality in ("a", "p", "v", "pp")
    for statistic in ("count", "probability_mean", "probability_max")
)
# int, candidate_manual_features 的最后一维宽度，与 CANDIDATE_MANUAL_FEATURE_NAMES 长度一致。
CANDIDATE_MANUAL_FEATURE_DIM = len(CANDIDATE_MANUAL_FEATURE_NAMES)
# tuple[str, ...], 长度 4；与 candidate_sampling_features 的最后一维逐位置对齐。
CANDIDATE_SAMPLING_FEATURE_NAMES = (
    "crop_boundary_limited",
    "crop_center_to_v_centroid_l2_voxels",
    "crop_center_to_v_centroid_linf_voxels",
    "v_retained_fraction",
)
# int, candidate_sampling_features 的最后一维宽度。
CANDIDATE_SAMPLING_FEATURE_DIM = len(CANDIDATE_SAMPLING_FEATURE_NAMES)
# tuple[str, ...], 长度 4；候选—token 注意力偏置输入依次取 token 的 XYZ 相对坐标和 Stage1 概率。
CANDIDATE_TOKEN_ATTENTION_BIAS_FEATURE_NAMES = (
    "relative_x",
    "relative_y",
    "relative_z",
    "stage1_probability",
)
# int, 候选—token 注意力偏置特征的最后一维宽度。
CANDIDATE_TOKEN_ATTENTION_BIAS_FEATURE_DIM = len(
    CANDIDATE_TOKEN_ATTENTION_BIAS_FEATURE_NAMES
)
# float, 40 个体素；A/P 相对坐标的归一化物理半边长需要再乘各轴 voxel_size_xyz。
FULL_CONTEXT_HALF_EXTENT_VOXELS = STAGE1_BOX_SIZE / 2.0
# float, 24 个体素；V/PP 相对坐标的归一化物理半边长需要再乘各轴 voxel_size_xyz。
LOCAL_CONTEXT_HALF_EXTENT_VOXELS = LOCAL_BOX_SIZE / 2.0


def identity_scope_class_indices(scope: IdentityScope) -> tuple[int, ...]:
    """
    返回指定身份监督范围包含的六分类下标。

    输入参数:
        - scope: IdentityScope；small_molecule 只选择 SMALL_MOLECULE_INDEX，all_non_background 选择 FOREGROUND_CLASS_INDICES。

    返回值:
        - class_indices: tuple[int, ...]；每个整数索引 CLASS_NAMES，顺序与身份监督允许的类别顺序一致。
    """

    if scope == "small_molecule":
        return (SMALL_MOLECULE_INDEX,)
    if scope == "all_non_background":
        return FOREGROUND_CLASS_INDICES
    raise ValueError(f"不支持 identity_scope={scope!r}。")


@dataclass(frozen=True)
class SourceRoots:
    """
    描述只读科学输入与 Matcher 可写产物的三个根目录。

    构造参数:
        - data_root: Path；AdaLigand Ori_Data 根目录，提供 occurrence、LigandObject、受体、密度和冻结语言表示。
        - stage1_root: Path；Stage1 Find_0 产物根目录，提供 F1 候选、centered 几何、A/P/V token 和概率图。
        - output_root: Path；Matcher 可写根目录，承载版本化监督清单、体素缓存和 PP token 缓存。

    派生路径:
        - language_root: Path；data_root 下两套冻结配体语言表示的共同父目录。
        - manifest_root: Path；output_root 下与 MANIFEST_SCHEMA_VERSION 绑定的闭集监督清单目录。
        - voxel_cache_root: Path；output_root 下与体素缓存 schema 和局部裁剪策略绑定的目录。
        - pp_cache_root: Path；output_root 下与 PP 缓存 schema、局部裁剪策略和 top_k 绑定的目录。

    本类只计算路径，不创建目录，也不读取或写入文件。
    """

    data_root: Path
    stage1_root: Path
    output_root: Path

    @property
    def language_root(self) -> Path:
        """
        返回两套冻结配体语言表示的共同父目录。

        返回值:
            - language_root: Path；data_root/stage1_preparation_box_pool_2/ligand_language_models。
        """

        return self.data_root / "stage1_preparation_box_pool_2/ligand_language_models"

    @property
    def manifest_root(self) -> Path:
        """
        返回版本化闭集监督清单目录。

        返回值:
            - manifest_root: Path；目录名 closed_set_v{MANIFEST_SCHEMA_VERSION} 防止不同联合身份契约共用清单文件。
        """

        return self.output_root / f"manifests/closed_set_v{MANIFEST_SCHEMA_VERSION}"

    def voxel_cache_root(self, crop_strategy: LocalCropStrategy) -> Path:
        """
        返回由缓存 schema 与局部裁剪策略共同隔离的 105 通道体素缓存目录。

        输入参数:
            - crop_strategy: LocalCropStrategy；决定每个候选的 48³ 裁剪起点，也是缓存目录的一部分。

        返回值:
            - voxel_cache_root: Path；output_root/cache/voxel_input_v{VOXEL_CACHE_SCHEMA_VERSION}/{crop_strategy}。
        """

        if crop_strategy not in LOCAL_CROP_STRATEGIES:
            raise ValueError(f"不支持 local_crop_strategy={crop_strategy!r}。")
        return (
            self.output_root
            / f"cache/voxel_input_v{VOXEL_CACHE_SCHEMA_VERSION}/{crop_strategy}"
        )

    def pp_cache_root(self, top_k: int, crop_strategy: LocalCropStrategy) -> Path:
        """
        返回由缓存 schema、局部裁剪策略与 top_k 共同隔离的 PP token 缓存目录。

        输入参数:
            - top_k: int；每个候选最多保留的 PP token 数，必须为正整数。
            - crop_strategy: LocalCropStrategy；决定每个候选的 48³ 裁剪起点，也是缓存目录的一部分。

        返回值:
            - pp_cache_root: Path；output_root/cache/pp_input_v{PP_CACHE_SCHEMA_VERSION}/{crop_strategy}/top_{top_k}。
        """

        if top_k <= 0:
            raise ValueError("PP token 缓存的 top_k 必须为正整数。")
        if crop_strategy not in LOCAL_CROP_STRATEGIES:
            raise ValueError(f"不支持 local_crop_strategy={crop_strategy!r}。")
        return (
            self.output_root
            / f"cache/pp_input_v{PP_CACHE_SCHEMA_VERSION}/{crop_strategy}/top_{top_k}"
        )


def safe_object_filename(object_key: str) -> str:
    """
    按 A–G 文件规则把 object_key 转换成 LigandObject 文件名主体。

    输入参数:
        - object_key: str；联合身份中的模板图键，原值仍保存在清单和 Dataset 字段中。

    返回值:
        - filename_stem: str；字母、数字、下划线、点和连字符保持不变，其余字符逐字符替换为下划线。
    """

    return "".join(character if character.isalnum() or character in "_.-" else "_" for character in object_key)
