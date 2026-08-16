"""
只转发 Matcher 包级稳定名称，其余数据、模型、训练和评估入口仍从所属子模块导入。

公开名称:
    - CLASS_NAMES: tuple[str, ...]，长度 6；六分类下标顺序依次为 background、ion、nucleotide_like、peptide_like、small_molecule、sugar，checkpoint、损失和指标共同依赖该顺序。
    - MANIFEST_SCHEMA_VERSION: int；当前值为 4，表示清单按 object_key 与 PreparedLigand 顶层 smiles 的联合身份组织。

导入 matcher 会加载 matcher.data.schema 并把上述两个对象绑定到包命名空间；本模块不读取
数据文件、不构造 Dataset，也不启动模型、训练或评估。其余入口仍从各自子模块显式导入，
避免把内部实现误当作稳定包级接口。
"""

from .data.schema import CLASS_NAMES, MANIFEST_SCHEMA_VERSION

__all__ = ["CLASS_NAMES", "MANIFEST_SCHEMA_VERSION"]
