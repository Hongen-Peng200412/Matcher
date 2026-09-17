"""把冻结候选记录物化为模型样本, 并把变长样本整理成物理批次.

candidate_data 负责单候选的体素、四模态 token、区域监督和精确 SMILES 身份轴; contracts
集中声明跨 Dataset、批处理和模型边界传递的结构化字段; batching 负责候选顺序、
补零掩码、配体图连接和采用 spawn 工作进程的 DataLoader. 该包只消费 data 包已经
定义的数据产物, 不重新定义 SMILES 图或 Stage1 概率的科学语义.
"""
