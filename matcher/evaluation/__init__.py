"""集中放置完整划分指标、validation 选模分数和一次性 calibration 编排。

validation 是训练期间反复使用、用于选择 BEST checkpoint 的验证划分；calibration 是方案
已经由 validation 选定后只评估一次的保留划分。本包的 metrics 模块只在内存中累计指标，
calibration 模块另行记录所选 checkpoint、当前数据证据与一次性 claim 后写出严格 JSON；
输入文件不会被快照或锁定。
包入口不重导出子模块符号，调用方应从具体模块导入所需实体。
"""
