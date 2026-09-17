"""
定义 Matcher 的固定数据契约、最大覆盖标签和精确 SMILES 图读取。

子模块职责:
    - schema: 固定报告类别、精确 SMILES 身份、张量宽度和来源目录。
    - manifest: 读取并核验唯一活动 Stage1 manifest。
    - labels: 用候选与 occurrence 稀疏体素的双向覆盖率建立唯一前景目标。
    - density: 构造完整 56 通道密度或直接构造单实验密度通道。
    - smiles_graph: 每个工作进程一次载入全局图包，并按精确 SMILES 返回化学图。

目录扫描和产物发布位于 ops.matcher；Dataset、训练和评估分别位于 matcher.datasets、
matcher.training 和 matcher.evaluation。
"""
