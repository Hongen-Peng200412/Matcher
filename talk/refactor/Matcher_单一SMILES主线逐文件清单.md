# Matcher 单一 SMILES 主线逐文件清单

## 清单用途

本清单覆盖 `Learn/CUMULATIVE@75ac5eaa5f4ec682724b72110605c74ec9b846a2` 中参与 Matcher 活动科学主线的 Python、配置、训练入口和测试。它在删除旧实现前冻结每个文件的当前职责、主要调用者、生命周期和目标位置。历史结果文档不属于活动运行时，不因本轮重构而删除。

本轮只建立一条运行链：Stage1 产物经 `ops/matcher/` 构建 manifest，`matcher/data/` 与 `matcher/datasets/` 读取该 manifest，`matcher/model/` 完成四种具名模型前向，`matcher/training/` 训练并保存 BEST/LAST，`matcher/evaluation/` 消费冻结预测。`matcher/smiles_identity/` 只作为迁移来源，不保留为平行入口。

## 数据层

| 现有文件 | 当前职责与主要调用者 | 生命周期 | 目标与理由 |
| --- | --- | --- | --- |
| `matcher/data/__init__.py` | 数据包标识；由测试和 Dataset 间接导入 | 保留 | 更新为单一 SMILES 数据层的最小公开入口 |
| `matcher/data/schema.py` | 旧 v4 联合身份、语言家族、类别和路径契约；被旧 manifest、Dataset、模型与测试调用 | 替换 | 只保留二分类、精确 SMILES、输入宽度和正式根目录契约 |
| `matcher/data/density.py` | 构造历史 56 个密度通道并实现截断归一化；被 Dataset 调用 | 保留并扩展 | 保留 strongest 的 56 通道函数；增加直接构造单实验密度通道的公开函数 |
| `matcher/data/labels.py` | 旧候选与 occurrence 覆盖目标选择；被旧 manifest 和测试调用 | 重写 | 冻结双向覆盖阈值、几何平均最大和较小 `occurrence_id` 并列规则；供 manifest 构建和测试评估共用 |
| `matcher/data/ligand_graph.py` | 由 `object_key` 读取 LigandObject 图；被旧 Dataset 调用 | 删除 | 由 `matcher/data/smiles_graph.py` 的精确 SMILES 图包读取替代 |
| `matcher/data/manifest.py` | 扫描旧 Stage1 目录、构造 v4 联合身份清单并混合运行时读取 | 重写 | 只保留版本化 manifest schema、读取和必要失败关闭检查；目录扫描移到 `ops/matcher/` |
| `matcher/data/geometry.py` | 旧裁剪、坐标换算和候选几何帮助函数 | 迁移后删除 | supplied crop 仍需要的短小换算并入唯一 Dataset；不保留平行几何入口 |
| `matcher/data/pp_input.py` | 旧 PP token 构造和缓存 | 迁移后删除 | 现场 PP 构造并入唯一 Dataset |
| `matcher/data/precompute.py` | 旧预计算命令入口 | 删除 | 新主线不预先落盘高维体素输入 |
| `matcher/data/voxel_input.py` | 旧 105/106 通道体素缓存读取与构造 | 迁移后删除 | supplied crop 和现场构造进入唯一 Dataset；不保留体素缓存旁路 |
| `matcher/smiles_identity/schema.py` | 新精确 SMILES 常量和 SourceRoots | 迁移来源 | 合并到 `matcher/data/schema.py` 后删除 |
| `matcher/smiles_identity/graph.py` | 每个 worker 一次完整载入全局 SMILES 图包 | 迁移来源 | 移到 `matcher/data/smiles_graph.py` |
| `matcher/smiles_identity/labels.py` | 确定性最大覆盖目标纯函数 | 迁移来源 | 合并到 `matcher/data/labels.py` |
| `matcher/smiles_identity/manifest.py` | 读取精确 SMILES manifest，并依赖标签旁路文件构建 validation 清单 | 迁移后删除 | 运行时读取迁入 `matcher/data/manifest.py`；扫描构建改到 `ops/matcher/` 且不依赖标签旁路 |
| `matcher/smiles_identity/build_assets.py` | 从 PreparedLigand 顶层 SMILES 构建图包和自同构包 | 迁移来源 | 移到 `ops/matcher/build_smiles_assets.py`；不被训练运行时导入 |
| `matcher/smiles_identity/write_labels.py` | 单独发布 `F2_centered_label.npz` | 删除 | 新 manifest 直接保存候选几何监督，不再设置标签旁路阶段 |

## Dataset 与批次层

| 现有文件 | 当前职责与主要调用者 | 生命周期 | 目标与理由 |
| --- | --- | --- | --- |
| `matcher/datasets/__init__.py` | Dataset 包标识 | 保留 | 导出唯一 `load_manifest_dataset()` 和批次入口 |
| `matcher/datasets/contracts.py` | 旧联合身份 CandidateItem/CandidateBatch | 重写 | 删除六分类、`object_key`、seen/unseen 和语言家族字段；逐字段冻结新数组形状 |
| `matcher/datasets/candidate_data.py` | 旧 v4 Dataset、PDB 缓存、LigandObject 图和预计算体素读取 | 重写 | 迁入 supplied crop、现场密度/受体/PP 构造、SMILES 身份轴和 `selected_only` 过滤；公开 `load_manifest_dataset()` |
| `matcher/datasets/batching.py` | 旧候选 collate、图连接、采样器和 worker 初始化 | 重写 | 迁入 SMILES 图连接；保证每个持久 worker 一次初始化图包 |
| `matcher/smiles_identity/contracts.py` | 精确 SMILES CandidateItem/CandidateBatch | 迁移来源 | 合并到 `matcher/datasets/contracts.py` 后删除 |
| `matcher/smiles_identity/dataset.py` | validation supplied crop Dataset 和 PDB 内缓存 | 迁移来源 | 合并到 `matcher/datasets/candidate_data.py`，补全四变体所需输入与 manifest 过滤 |
| `matcher/smiles_identity/batching.py` | 精确 SMILES collate、采样器与 worker 初始化 | 迁移来源 | 合并到 `matcher/datasets/batching.py` 后删除 |

## 模型层

| 现有文件 | 当前职责与主要调用者 | 生命周期 | 目标与理由 |
| --- | --- | --- | --- |
| `matcher/model/__init__.py` | 旧模型公开符号 | 保留 | 只导出四变体统一模型与配置 |
| `matcher/model/config.py` | 历史任意组合开关矩阵 | 重写 | `variant` 只允许 `strongest`、`density_blob_only`、`pocket_only`、`trunk_only`；共同超参数显式固定 |
| `matcher/model/raunet.py` | 三维 U-Net 主干 | 保留 | 只调整输入通道由具名变体决定，不改变主干数学实现 |
| `matcher/model/fusion.py` | 候选 token 聚合、统计归一化和摘要拼接 | 精简保留 | 保留 mean/max、候选条件注意力与摘要拼接；删除不再暴露的历史组合 |
| `matcher/model/candidate_encoder.py` | 旧六分类候选编码器 | 重写 | 输出单个前景 logit；按四变体严格禁用输入和模块 |
| `matcher/model/ligand_matcher.py` | 旧联合身份、类别概率耦合与三分支打分 | 重写 | 身份键只用 SMILES；删除类别概率输入；trunk-only 不实例化图、清单和门参数 |
| `matcher/smiles_identity/model_config.py` | 精确 SMILES 模型超参数，但只有一套结构 | 迁移来源 | 有效共同字段并入 `matcher/model/config.py` |
| `matcher/smiles_identity/candidate_encoder.py` | 单前景 logit、RAUNet 与 A/P/V/PP 编码 | 迁移来源 | 合并到正式候选编码器并增加三项消融结构 |
| `matcher/smiles_identity/model.py` | 语言、SMILES 图、清单三分支和 PairScorer | 迁移来源 | 合并到正式 matcher，删除类别耦合并实现 trunk-only 物理裁剪 |

## 训练与评估层

| 现有文件 | 当前职责与主要调用者 | 生命周期 | 目标与理由 |
| --- | --- | --- | --- |
| `matcher/training/__init__.py` | 训练包标识 | 保留 | 导出唯一训练入口 |
| `matcher/training/config.py` | 历史多阶段、多语言、多损失变体配置 | 重写 | 压缩为数据、四变体、优化、辅助损失和 W&B 的单阶段配置 |
| `matcher/training/losses.py` | 六分类、区域和 oracle/detached/joint 身份损失 | 重写 | 前景 BCE、直接 SMILES 身份交叉熵和可选区域辅助损失 |
| `matcher/training/engine.py` | 完整训练、validation、早停、恢复和大量旧指标 | 重写 | 保留 epoch/BEST/LAST/早停/BF16/累积/裁剪能力，删除旧类别与兼容分支 |
| `matcher/training/checkpointing.py` | checkpoint 原子保存与恢复 | 保留并收紧 | 保存前景阈值、selection 分项、配置和输入证据 |
| `matcher/training/tracking.py` | W&B 和训练诊断 | 精简保留 | 只记录损失、PRAUC、F1、SMILES 匹配准确率、学习率、梯度与资源诊断 |
| `matcher/training/numeric.py` | 数值后端快照 | 保留 | 继续进入 checkpoint 与 resolved 证据 |
| `matcher/training/initialization.py` | 候选预训练 checkpoint 迁移 | 删除 | 四项正式实验都从零训练 |
| `matcher/training/experiment.py` | 旧计划式训练入口 | 重写 | 唯一薄入口，读取单个 YAML 并调用 engine |
| `matcher/smiles_identity/losses.py` | 单前景、区域和直接身份损失 | 迁移来源 | 合并到 `matcher/training/losses.py` |
| `matcher/smiles_identity/config.py` | 驻留健康任务配置 | 迁移来源 | 仅复用可证明必要的简单字段；不保留平行 loader |
| `matcher/smiles_identity/engine.py` | 简化训练与同 split validation | 迁移来源 | 与旧正式 engine 的 BEST/LAST、恢复和阈值能力合并 |
| `matcher/smiles_identity/experiment.py` | 平行训练入口 | 删除 | 正式入口迁入 `matcher/training/experiment.py` |
| `matcher/evaluation/metrics.py` | 旧六分类、身份范围、seen/unseen 和端到端指标 | 重写 | 只保留二分类 PRAUC/F1、阈值选择和真实前景候选上的 SMILES 匹配准确率 |
| `matcher/evaluation/calibration.py` | 历史 calibration 评估入口 | 删除 | 本轮不读取 calibration，也不保留活动 split |
| `matcher/evaluation/test_set.py` | 尚不存在 | 新增 | 连接 real/Cryo 测试事实，生成冻结预测和两种 Find_1 top-k 汇总 |
| `matcher/evaluation/plotting.py` | 尚不存在 | 新增 | 只读取冻结元数据，生成七栏图和两张 top-k 图；当前只生成 Stage1-only 预览 |

## 运维、配置和测试

| 现有位置 | 当前职责 | 生命周期 | 目标与理由 |
| --- | --- | --- | --- |
| `ops/` | 当前没有正式 Matcher 工具 | 新建 | 增加图包构建、manifest 构建、Stage1-only 元数据冻结和严格离线审计；正式科学规则仍位于 `matcher/` |
| `configs/experiments/smiles_identity/` | validation 驻留健康任务 | 删除 | 被四个正式训练 YAML 替代 |
| `configs/experiments/initial/`、`batch1/`—`batch5/`、`joint_identity_v4/` | 历史实验活动 YAML 与说明 | 删除活动入口 | 历史结果由 Git 与结果文档保留，不让冷启动读者误入旧契约 |
| `训练与运行/sh/build_smiles_identity.sh` | 图、标签旁路和 validation manifest 三阶段构建 | 替换 | 改为简短的图包构建入口；manifest 使用独立短入口且不生成标签旁路 |
| `训练与运行/sh/train_smiles_identity.sh` | validation 驻留训练 | 删除 | 被唯一正式训练入口替代 |
| `训练与运行/sh/train_experiment.sh` | 旧计划式训练 | 重写 | 接收一个正式 YAML |
| 历史批次和最优变体推荐 Shell | 指向旧 YAML 的历史训练入口 | 删除活动入口 | 历史命令保留在 Git 和执行记录，不作为新版冷启动入口 |
| `matcher/tests_smiles/` | 平行 SMILES 包回归 | 删除目录 | 仍适用测试迁入 `matcher/tests/`，测试不做学习型细粒度注释 |
| `matcher/tests/` | 旧正式管线回归 | 重写 | 覆盖唯一数据链、四变体、训练、Stage1-only 与冻结评估；测试文件最后进入学习历史 |

## 明确不纳入

- `文档/记录/` 中的历史实验结果、Job、W&B 和科学结论不删除、不改写成新版结果。
- `RE-talk/关于ligand_object—SMILES/` 中的观察、转换建议和既有验收记录保留为决策证据。
- 通用 `训练与运行/runtime/`、`训练与运行/sbatch/`、`submit_task.sh` 与 `与服务器交互/` 基础设施继续保留。
- 当前第 6 片 Stage1 推理由独立任务 `codex://threads/01a099cd-293c-7200-9df3-31ebde48b468` 负责；本轮不接管、不写入正式训练 manifest。

## 实施后核对

本节记录计划清单与最终训练前代码的有限差异，避免把实施前判断误读为当前文件状态。

- `matcher/training/tracking.py` 未保留。W&B 初始化与少量正式指标直接位于 `matcher/training/engine.py`，单独模块只会增加一次跳转。
- `matcher/training/numeric.py` 未保留。新版 checkpoint 直接记录 Python、PyTorch、CUDA、GPU、manifest 与 YAML 来源；没有第二套数值配置入口。
- `ops/matcher/` 只保存图包、manifest、Stage1-only 元数据、冻结测试和绘图的薄命令入口。一次性哈希与机械审计脚本留在 ignored `tmp/`，不转成活动工具。
- `train_experiment.sh` 没有继续作为泛化历史入口；四个 `训练与运行/sh/final_smiles/train_*.sh` 分别绑定四份正式 YAML，使冷启动读者不需要猜测参数。
- `matcher/model/fusion.py` 不保留候选实体注意力偏置开关或输入。候选条件注意力只由候选摘要和 token 内容确定，与四变体冻结边界一致。
- `pytest.ini` 把正式测试收集范围固定为 `matcher/tests`；ignored `tmp/` 中的历史审计文件不再被完整 pytest 误收集。
