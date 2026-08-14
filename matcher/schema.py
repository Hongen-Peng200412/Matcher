"""集中定义新 Matcher 数据与实验产物的稳定名称和路径契约。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


MANIFEST_SCHEMA_VERSION = 3
VOXEL_CACHE_SCHEMA_VERSION = 2
PP_CACHE_SCHEMA_VERSION = 2

CLASS_NAMES = (
    "background",
    "ion",
    "nucleotide_like",
    "peptide_like",
    "small_molecule",
    "sugar",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
SMALL_MOLECULE_INDEX = CLASS_TO_INDEX["small_molecule"]
FOREGROUND_CLASS_NAMES = CLASS_NAMES[1:]
FOREGROUND_CLASS_INDICES = tuple(CLASS_TO_INDEX[name] for name in FOREGROUND_CLASS_NAMES)
IdentityScope = Literal["small_molecule", "all_non_background"]
IDENTITY_SCOPES = ("small_molecule", "all_non_background")
LocalCropStrategy = Literal["fixed_center", "v_centroid"]
LOCAL_CROP_STRATEGIES = ("fixed_center", "v_centroid")
OCCURRENCE_TYPE_TO_CLASS = {
    "ion": "ion",
    "nucleotide_like": "nucleotide_like",
    "peptide_like": "peptide_like",
    "small_molecule": "small_molecule",
    "sugar": "sugar",
    "other": "background",
}

LANGUAGE_FAMILIES = ("molformer", "smi_ted_289m")
LANGUAGE_MODEL_NAMES = {
    "molformer": "ibm-research/MoLFormer-XL-both-10pct",
    "smi_ted_289m": "SMI-TED Light 289M",
}

STAGE1_BOX_SIZE = 80
LOCAL_BOX_SIZE = 48
LOCAL_BOX_MARGIN = (STAGE1_BOX_SIZE - LOCAL_BOX_SIZE) // 2
LANGUAGE_EMBEDDING_DIM = 768
BOX_ORIGIN_ABSOLUTE_TOLERANCE_ANGSTROM = 5e-5

# A/P/V/PP token 的共同末四维依次是 Stage1 固化概率、relative_x、relative_y、relative_z。
# Stage1 固化概率有限且位于 [0,1]。
# A/P 相对 XYZ 以候选实际裁剪中心为原点，并按完整 80³ 物理半边长归一化。
# V/PP 相对 XYZ 以候选实际裁剪中心为原点，并按候选级 48³ 物理半边长归一化。
A_TOKEN_DIM = 229
P_TOKEN_DIM = 116
V_TOKEN_DIM = 52
PP_TOKEN_DIM = 4
CANDIDATE_MANUAL_FEATURE_NAMES = tuple(
    f"{modality}_{statistic}"
    for modality in ("a", "p", "v", "pp")
    for statistic in ("count", "probability_mean", "probability_max")
)
CANDIDATE_MANUAL_FEATURE_DIM = len(CANDIDATE_MANUAL_FEATURE_NAMES)
CANDIDATE_SAMPLING_FEATURE_NAMES = (
    "crop_boundary_limited",
    "crop_center_to_v_centroid_l2_voxels",
    "crop_center_to_v_centroid_linf_voxels",
    "v_retained_fraction",
)
CANDIDATE_SAMPLING_FEATURE_DIM = len(CANDIDATE_SAMPLING_FEATURE_NAMES)
CANDIDATE_TOKEN_ATTENTION_BIAS_FEATURE_NAMES = (
    "relative_x",
    "relative_y",
    "relative_z",
    "stage1_probability",
)
CANDIDATE_TOKEN_ATTENTION_BIAS_FEATURE_DIM = len(
    CANDIDATE_TOKEN_ATTENTION_BIAS_FEATURE_NAMES
)
FULL_CONTEXT_HALF_EXTENT_VOXELS = STAGE1_BOX_SIZE / 2.0
LOCAL_CONTEXT_HALF_EXTENT_VOXELS = LOCAL_BOX_SIZE / 2.0


def identity_scope_class_indices(scope: IdentityScope) -> tuple[int, ...]:
    """返回身份监督范围包含的六分类下标。"""

    if scope == "small_molecule":
        return (SMALL_MOLECULE_INDEX,)
    if scope == "all_non_background":
        return FOREGROUND_CLASS_INDICES
    raise ValueError(f"不支持 identity_scope={scope!r}。")


@dataclass(frozen=True)
class SourceRoots:
    """描述只读科学输入和 Matcher 可写产物根目录。"""

    data_root: Path
    stage1_root: Path
    output_root: Path

    @property
    def language_root(self) -> Path:
        """返回两套冻结配体语言表示的共同父目录。"""

        return self.data_root / "stage1_preparation_box_pool_2/ligand_language_models"

    @property
    def manifest_root(self) -> Path:
        """返回版本化清单目录。"""

        return self.output_root / f"manifests/closed_set_v{MANIFEST_SCHEMA_VERSION}"

    def voxel_cache_root(self, crop_strategy: LocalCropStrategy) -> Path:
        """返回由 schema 与局部裁剪策略共同隔离的 105 通道体素缓存目录。"""

        if crop_strategy not in LOCAL_CROP_STRATEGIES:
            raise ValueError(f"不支持 local_crop_strategy={crop_strategy!r}。")
        return (
            self.output_root
            / f"cache/voxel_input_v{VOXEL_CACHE_SCHEMA_VERSION}/{crop_strategy}"
        )

    def pp_cache_root(self, top_k: int, crop_strategy: LocalCropStrategy) -> Path:
        """返回由 schema、裁剪策略与 ``top_k`` 共同隔离的 PP token 缓存目录。"""

        if top_k <= 0:
            raise ValueError("PP token 缓存的 top_k 必须为正整数。")
        if crop_strategy not in LOCAL_CROP_STRATEGIES:
            raise ValueError(f"不支持 local_crop_strategy={crop_strategy!r}。")
        return (
            self.output_root
            / f"cache/pp_input_v{PP_CACHE_SCHEMA_VERSION}/{crop_strategy}/top_{top_k}"
        )


def safe_object_filename(object_key: str) -> str:
    """按 A–G 规则把 ``object_key`` 转换成 LigandObject 文件名主体。"""

    return "".join(character if character.isalnum() or character in "_.-" else "_" for character in object_key)
