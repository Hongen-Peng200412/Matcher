# Matcher

本仓库独立承载 AdaLigand 的单候选闭集配体身份分类阶段。AdaLigand 生产 A–G 解析产物、密度图和 Stage1 Find_1 候选；Matcher 对每个候选执行六分类，并从同一 PDB 的精确 `PreparedLigand.smiles` 身份清单中选择身份。当前从头训练入口把交给 SMI-TED 公开接口的顶层字符串同时作为监督身份和全局重原子图查询键；它不读取 `object_key` 或 LigandObject。无法取得合法 SMILES 图，或其 SMILES 身份没有任何有效语言向量的前景候选整条跳过；背景候选继续保留。

## 当前科学入口

- `matcher.smiles_identity`：当前正式管线。它读取 Find_1 已提供的 `48³` supplied crop，在内存连接 56 个密度通道和 50 个受体通道，并构造 A/P/V/PP token、六分类监督、两个区域监督和五类前景 SMILES 身份监督。
- 候选与真实 occurrence 的全局组件覆盖率和 occurrence 覆盖率都至少为 `0.30` 才命中。多个命中使用两侧覆盖率几何平均的最大值，并采用稳定并列规则。
- 一个 PDB 内的相同精确顶层 SMILES 共享一个身份槽；不同 PDB 不共享同一次闭集归一化。新清单不保存 `object_key`、`model_smiles`、seen/unseen、语言家族或另造的图编号。
- `matcher/data`、`matcher/datasets`、`matcher/model`、`matcher/training` 和 `matcher/evaluation` 保留联合身份 v4 历史管线，供既有实验复现。新管线只复用其中不读取 LigandObject 的密度通道、字节预算缓存、特征融合和 RAUNet 数值模块。
- 当前驻留任务只在 Find_1 validation 上训练并在同一 validation 上监测运行健康；它不读取 calibration，也不能解释为独立泛化评估。

## 目录

- `matcher/smiles_identity/`：精确 SMILES 图资产、Find_1 标签和清单、Dataset、批次、模型、损失、训练与验收入口；详细契约见该目录 README。
- `matcher/data/`：联合身份 v4 的 Stage1 与 A–G 来源适配、标签、候选几何、冻结清单、105 通道体素缓存和 PP token 缓存。
- `matcher/datasets/`：`CandidateDataset`、样本与批次字段、全局随机采样器、批量合并和 `DataLoader` 构造。
- `matcher/model/`：RAUNet、A/P/V/PP 四模态候选聚合、候选辅助头和闭集身份模型。
- `matcher/training/`：实验配置、损失、候选预训练初始化、checkpoint、W&B online 追踪、阶段训练和实验计划入口。
- `matcher/evaluation/`：validation/calibration 共用指标、validation 选模分数和一次性 calibration 编排。
- `matcher/utils/`：不决定科学语义的进程内缓存、严格 JSON 写入和跨进程文件锁。
- `matcher/tests/`：数据、空模态、模型与监督、global-batch 等价、指标、配置、checkpoint 和产物边界回归测试。
- `configs/experiments/initial/`：第 0 批七个自包含 YAML；一项 A100 顺序运行两个候选六分类阶段，另外六项定义两种语言表示和三种身份损失。
- `configs/experiments/batch1/`：第 1 批三个自包含 YAML；固定 SMI-TED、mean/max、五类非背景身份与手动特征，只比较 oracle/detached/joint。
- `configs/experiments/batch2/`：第 2 批五个自包含 YAML；统一 V 质心 `48³` 与采样统计，比较三种完整 Matcher 损失和两种候选聚合。
- `configs/experiments/batch3/`：第 3 批四个自包含 YAML；为候选条件注意力加入按模态独立、末层零初始化的实体分数偏置。
- `训练与运行/`：通用 Slurm、release、launch、四锁控制，以及 Matcher 的预计算、训练和一次性 calibration shell 入口。
- `configs/experiments/smiles_identity/`：精确 SMILES 身份管线的完整正式训练配置。
- `与服务器交互/`：只面向 `/home/penghongen/My_Project/Matcher` 的安全同步入口。
- `文档/exec_plan/`：逐项命令、Job ID、审查发现、临时决定和结果证据。
- `文档/记录/`：按单任务、批次和全局三个层次维护 validation 结果、对照结论与后续决策。
- `文档/mapping/`：规划、活动实现和历史实现之间的映射。

## 本地验证

```powershell
python -m compileall -q matcher
python -m mypy matcher
python -m pytest matcher\tests -q
python -m pytest matcher\tests_smiles -q
git diff --check
```

本地测试不代替服务器真实数据、CUDA 显存和吞吐验收。

## 服务器边界

- 正式源码：`/home/penghongen/My_Project/Matcher`
- release、launch、Slurm 日志和锁：`/home/penghongen/Feedback/Matcher`
- Matcher 清单、缓存、checkpoint、指标与评估结果：`/storage/penghongen/tmp/Matcher/`
- 只读 A–G 与 Stage1 输入：`/storage/penghongen/AdaLigand/...` 和 `/storage/penghongen/AdaLigand_stage1_inference/...`

安全同步不会删除远端 Matcher 目录；删除式同步只允许人类手动运行。正式任务通过 `训练与运行/submit_task.sh` 申请资源，并由 `pre_lock/try_lock/kill_lock/after_lock` 控制开始、复用、终止当前命令和释放 allocation。
