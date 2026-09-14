"""实现以精确 PreparedLigand.smiles 为唯一化学身份的新 Matcher 正式管线. 

本包集中保存图资产构建、Find_1 数据适配、模型、损失、训练和 I/O 验收入口. 
旧 object_key/LigandObject 管线保留在原有 matcher.data、matcher.datasets、
matcher.model、matcher.training 和 matcher.evaluation 模块中. 本包只复用其中不消费
LigandObject 的密度通道、字节预算缓存、特征融合与 RAUNet 数值模块, 不复用旧图读取器、
旧清单或旧身份打分器. 

本模块不聚合子模块对象, 避免资产构建命令在化学环境中提前导入训练依赖. 
"""
