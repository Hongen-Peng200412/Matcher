"""集中放置 validation 指标、冻结测试连接和只读绘图函数。

validation 是训练期间反复使用、用于选择 BEST checkpoint 的验证划分。metrics 模块只在
内存中累计指标；test_set 连接冻结的 Stage1 测试事实；plotting 只读取冻结评估产物。
包入口不重导出子模块符号，调用方应从具体模块导入所需实体。
"""
