"""
适配 Stage1 与 A–G 冻结来源，并构造 Matcher 监督清单和候选级输入缓存。

子模块职责:
    - schema: 集中定义类别、联合身份、语言家族、体素尺寸、token 维度和缓存 schema 常量。
    - manifest: 扫描候选覆盖监督，按 object_key 与顶层 PreparedLigand.smiles 构造 PDB 内联合身份。
    - labels: 用候选与 occurrence 稀疏体素的双向覆盖率分配六分类目标。
    - geometry: 解析 48³ 起点、转换坐标并把受体原子硬散射到 80³ 网格。
    - density: 从实验密度、模拟密度和受体占据掩码生成 56 个派生通道。
    - voxel_input: 拼接密度 56 通道与受体 49 通道，发布候选级 float16 体素缓存。
    - pp_input: 从完整 Stage1 probability_map 构造 PP 模态 top-k 概率 token，发布 PDB 级 NPZ 缓存。
    - ligand_graph: 把 LigandObject NPZ 转为模板图节点与边特征。
    - precompute: 按 PDB 多进程物化显式选择的 voxel 与 pp 缓存。

本包只负责冻结来源适配与缓存契约；Dataset 拼装、训练循环和评估逻辑分别位于 matcher.datasets、matcher.training 和 matcher.evaluation。
"""
