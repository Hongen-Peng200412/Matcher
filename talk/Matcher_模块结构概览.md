# Matcher 模块结构概览

本文说明当前单候选闭集 Matcher 的正式 Python 包。历史双模式设计已经退出活动代码；需要追溯历史时读取 Git 历史与 `文档/mapping/计划执行映射.md`，不要把旧模块名解释为当前入口。

## 冷读顺序

1. `matcher/data/schema.py`：类别顺序、语言家族、局部裁剪策略、缓存版本和来源根目录。
2. `matcher/datasets/contracts.py`：单候选样本、批次张量和候选 metadata 的内存字段。
3. `matcher/data/`：Stage1 与 A–G 来源适配、标签、候选几何、冻结清单以及 voxel/PP 缓存。
4. `matcher/datasets/candidate_data.py` 与 `matcher/datasets/batching.py`：样本装配、全局随机批次、批量合并和工作进程生命周期。
5. `matcher/model/`：105 通道 RAUNet、A/P/V/PP 融合、候选辅助头和三条闭集身份分支。
6. `matcher/training/losses.py` 与 `matcher/evaluation/metrics.py`：训练目标、validation 指标和选模分数。
7. `matcher/training/engine.py`：阶段运行环境、训练 epoch、完整划分评估、BEST/LAST 和早停。
8. `matcher/training/experiment.py`：按 YAML 顺序运行训练计划；`matcher/evaluation/calibration.py` 只负责最终确认后的一次性评估。

## 依赖方向

`matcher/utils` 不读取科学模块；`matcher/data` 可以读取 `utils`；`matcher/datasets` 可以读取 `data`；`matcher/model` 只消费数据 schema 和 Dataset 批次契约；`matcher/training` 连接数据、模型、损失、指标、checkpoint 与 W&B；`matcher/evaluation/calibration.py` 可以复用训练引擎的完整划分评估，但 `matcher/evaluation/__init__.py` 不反向导出 calibration。

各子包的 `__init__.py` 只说明职责。稳定模型入口继续由 `matcher.model` 导出；其余正式入口使用完整模块路径，不建立旧顶层模块的兼容转发层。

## 正式命令入口

- `python -m matcher.data.manifest`：构建 train/validation 等明确允许划分的冻结清单。
- `python -m matcher.data.precompute`：按 PDB 预计算 voxel 与 PP 候选缓存。
- `python -m matcher.training.experiment`：执行一份正式实验 YAML。
- `python -m matcher.evaluation.calibration`：仅在用户最终确认后执行一次 calibration。

## 可读性约束

- 数据来源适配不导入模型或训练状态；Dataset 不扫描配置目录。
- 训练主链保持可直读：YAML → Dataset/loader → model → loss → validation → checkpoint。
- validation 指标与 calibration 复用同一累计器，但 calibration 结果不回流训练或选模。
- 发布字节摘要、launch/release 路径取证和一次性诊断留在 `tmp/` 或具名 `ops/`，不进入正式 `matcher` 包。
- 临时分析、smoke 和画像产物进入任务专属临时目录，不能混入正式清单、配置或运行目录。
