"""集中定义新 Matcher 数据与实验产物的稳定名称和路径契约。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MANIFEST_SCHEMA_VERSION = 1
VOXEL_CACHE_SCHEMA_VERSION = 1

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

    @property
    def voxel_cache_root(self) -> Path:
        """返回版本化 105 通道候选体素缓存目录。"""

        return self.output_root / f"cache/voxel_input_v{VOXEL_CACHE_SCHEMA_VERSION}"


def safe_object_filename(object_key: str) -> str:
    """按 A–G 规则把 ``object_key`` 转换成 LigandObject 文件名主体。"""

    return "".join(character if character.isalnum() or character in "_.-" else "_" for character in object_key)
