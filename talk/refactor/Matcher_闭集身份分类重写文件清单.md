# Matcher 闭集身份分类重写文件清单

本文冻结提交 `6220ece3595e48dae9ee1ef749146efa1048305c` 之后、正式批量删除旧实现之前的文件处置边界。当前重写以 Stage1 F1 候选的六分类和 PDB 内小分子身份分类为目标，不保留旧 `ground_truth_context`、A/B/O、A′/B′/O′、Hungarian 匹配或粗细两级槽位的活动实现。

“历史记录”表示文件继续由 Git 跟踪，但不代表当前规格；“项目基础设施”表示不决定科学任务的数据、模型或监督语义。测试文件在实现分支可以随旧实现一同退出；学习分支按双线历史规则把本轮全部测试变更集中到最后一个提交。

## 仓库入口、许可证与历史材料

| 当前文件 | 当前职责与调用者 | 生命周期与现有测试 | 目标位置与动作 | 具体理由 |
|---|---|---|---|---|
| `.gitignore` | 排除本地缓存和运行产物；Git 读取 | 项目基础设施；无需单元测试 | 原位保留并按新产物补充 | 不包含旧科学语义 |
| `AGENTS.md` | 定义 Matcher 的双线 Git 和文字入口规则；所有 agent 读取 | 项目基础设施 | 原位保留 | 当前项目治理规则继续有效 |
| `CLAUDE/memory/handoffs/2026-08-06-matcher-repository-extraction.md` | 记录独立仓库迁移历史；无运行调用者 | 历史记录 | 原位保留 | 保留仓库来源证据，不作为当前规格 |
| `README.md` | 介绍旧双模式 Matcher；读者入口 | 当前契约，但内容失效 | 原位重写 | 新系统的任务、产物和稳定入口均已改变 |
| `grill_with_memory/07-28-17-50.md` | 保存旧 O/O′ 设计问答；无运行调用者 | 历史记录 | 原位保留 | 保留决定演变，不让旧问答控制新实现 |
| `licenses/PocketXMol.LICENSE` | 保存 PocketXMol 许可证原文 | 法律记录 | 原位保留 | 重写不改变第三方许可证义务 |
| `licenses/PocketXMol.md` | 解释许可证来源 | 法律记录 | 原位保留 | 与旧 Dataset 和 Matcher 监督无关 |

## 旧配置

| 当前文件 | 当前职责与调用者 | 生命周期与现有测试 | 目标位置与动作 | 具体理由 |
|---|---|---|---|---|
| `configs/mode1/phase1.yaml` | 配置旧 Stage1 推理模式第一阶段训练；由 `matcher.train` 读取 | 旧正式配置；训练测试间接覆盖 | 删除；以 `configs/experiments/initial/` 的新配置替代 | 字段绑定 A/B/O、Hungarian 和旧两阶段模型 |
| `configs/mode1/phase2.yaml` | 配置旧 Stage1 推理模式第二阶段训练；由 `matcher.train` 读取 | 旧正式配置；训练测试间接覆盖 | 删除；以新的六分类和完整 Matcher 配置替代 | 旧阶段含义与本轮两个科学任务不一致 |

## 旧 Matcher 活动代码

| 当前文件 | 当前职责与具体调用者 | 生命周期与现有测试 | 目标位置与动作 | 具体理由 |
|---|---|---|---|---|
| `matcher/README.md` | 说明旧单包导出和 Stage3 叶子；读者入口 | 旧当前契约 | 删除后按新包结构重写 | 旧模块、字段和稳定入口整体退出 |
| `matcher/__init__.py` | 标记旧 Python 包 | 正式代码；导入测试间接覆盖 | 删除后建立新的最小包入口 | 避免旧导出语义残留 |
| `matcher/batching.py` | 按 occurrence 数量采样并拼装旧 `MatcherBatch`；由模型测试和训练读取 | 正式代码；`test_model.py` 覆盖 | 删除；新建候选批次模块 | 旧批次围绕 occurrence 矩阵与 Hungarian 组织 |
| `matcher/context.py` | 从 Map 读取 V/PP 并形成旧候选上下文；由 `model_core.py` 使用 | 正式代码；`test_context.py` 覆盖 | 删除；在新多模态模型中重写 V/PP 读取 | Map 只调制 V/PP、48³ 截断和新聚合方式需要新契约 |
| `matcher/contracts.py` | 定义旧图、配体、PDB 级样本和批次对象；被 Dataset、模型和测试读取 | 正式代码；`test_contracts.py` 覆盖 | 删除；新建候选样本、身份清单和批次契约 | 旧对象直接承载 A/B/O 与出现实例槽位 |
| `matcher/data_common.py` | 读取旧 JSONL、构图、裁密度并计算 A/B/O 覆盖监督；被两个 Dataset 使用 | 正式代码；`test_contracts.py` 间接覆盖 | 删除；按冻结清单和六分类规则重写 | 用户明确要求删除旧 Dataset 与 loader 逻辑 |
| `matcher/data_ground_truth.py` | 以真实配体出现实例构造候选；由旧训练模式选择 | 正式代码；训练测试间接覆盖 | 删除且不提供替代模式 | 本轮明确不实现 `ground_truth_context` |
| `matcher/data_stage1.py` | 读取旧 F1-centered Stage1 候选；由旧训练入口使用 | 正式代码；训练测试间接覆盖 | 删除；新建只接受已完成 F1 产物的 Dataset | 需要跳过 `_BLOB_EXCEED`、新标签和新的空间边界 |
| `matcher/decode_ground_truth.py` | 对旧预测执行 Hungarian 解码与指标；由推理入口使用 | 正式代码；`test_decode.py` 覆盖 | 删除；新建六分类、条件身份和端到端指标 | 本轮评估明确不使用 Hungarian |
| `matcher/element_properties.py` | 保存旧 149 维配体图中的元素属性常量；由 `data_common.py` 使用 | 正式代码；图测试间接覆盖 | 删除；需要时依据新图契约重新实现 | 不让旧配体节点宽度成为隐式兼容约束 |
| `matcher/graph.py` | 构造并编码旧配体/受体图；被契约、Dataset 和图测试读取 | 正式代码；`test_graph.py` 覆盖 | 删除；在新配体表示模块中按需重写 | 可参考 Git 历史，但不原样保留旧接口 |
| `matcher/infer_ground_truth.py` | 执行旧真实候选推理；由旧训练脚本调用 | 正式入口；解码测试间接覆盖 | 删除且不建立对应入口 | `ground_truth_context` 不在新范围内 |
| `matcher/manifest.py` | 生成旧真实候选清单和 Stage1 指针；由旧 CPU 脚本调用 | 正式代码；`test_manifest.py` 覆盖 | 删除；新建版本化冻结清单生成器 | 新清单必须记录 `_BLOB_EXCEED` 跳过、身份和语言表示选择 |
| `matcher/map.py` | 实现旧轻量 U-Net 和摘要头；由 `context.py`、`model_core.py` 使用 | 正式代码；`test_map.py` 覆盖 | 删除；以核实过的 RAUNet 血统主干重写 | 新输入为 48³ 的 105 通道，并有 V/PP 两个调制头和候选摘要 |
| `matcher/model.py` | 包装旧六头 Matcher、参数组和阶段接口；由训练与测试读取 | 正式代码；`test_model.py` 覆盖 | 删除；按候选预训练和完整 Matcher 重建 | 六头与阶段接口均不再成立 |
| `matcher/model_core.py` | 实现旧 4+4 粗细槽位、A/B/O 注意力和 Hungarian 相关输出 | 正式代码；`test_model.py` 覆盖 | 删除；按 U-Net、多模态候选编码器和身份清单模块拆分重写 | 这是上一轮失败科学设计的主要活动实现 |
| `matcher/objectives.py` | 实现旧身份 Hungarian 指派和损失；由训练与测试读取 | 正式代码；`test_model.py` 覆盖 | 删除；实现六分类及 oracle/detached/joint 身份损失 | 监督定义已经根本改变 |
| `matcher/schedule.py` | 控制旧两阶段解冻和梯度门控；由训练入口使用 | 正式代码；训练测试间接覆盖 | 删除；若需要则在新训练引擎中重写 | 新主线是预训练后完整模型与完整模型从零训练 |
| `matcher/stage3.py` | 暴露旧 Stage3 可复用叶子 | 正式兼容入口；导入检查覆盖 | 删除且不提供兼容层 | Stage3 不属于本轮 Matcher 契约 |
| `matcher/train.py` | 选择两个旧 Dataset、构建 loader、训练、验证和 checkpoint；由旧 shell 调用 | 正式入口；`test_training.py`、`test_model.py` 覆盖 | 删除；新建统一实验入口与训练引擎 | 数据、模型、损失、指标、早停和实验账本全部改变 |

## 旧测试

| 当前文件 | 当前职责与被测代码 | 生命周期 | 目标位置与动作 | 具体理由 |
|---|---|---|---|---|
| `matcher/tests/__init__.py` | 标记旧测试包 | 测试代码 | 随旧测试删除；最终新建测试包 | 新测试不继承旧夹具 |
| `matcher/tests/sample_factory.py` | 构造旧 `RawGraph` 测试样本 | 测试代码 | 删除；按新契约重建夹具 | 旧图和样本字段退出 |
| `matcher/tests/test_context.py` | 测试旧 Map/V/PP 读取 | 测试代码 | 删除；新测试覆盖 48³ 截断和调制 | 被测实现退出 |
| `matcher/tests/test_contracts.py` | 测试旧样本与 A/B/O 覆盖 | 测试代码 | 删除；新测试覆盖六分类与身份标签 | 被测契约退出 |
| `matcher/tests/test_decode.py` | 测试旧 Hungarian 解码与评估 | 测试代码 | 删除；新测试覆盖条件身份和端到端指标 | 禁止在新评估中保留 Hungarian |
| `matcher/tests/test_graph.py` | 测试旧图层和构图 | 测试代码 | 删除；新配体图实现存在时重新建立测试 | 不把旧接口当作兼容标准 |
| `matcher/tests/test_manifest.py` | 测试旧真实候选与 Stage1 指针清单 | 测试代码 | 删除；新测试覆盖冻结选择与缺失策略 | 新清单字段和样本粒度不同 |
| `matcher/tests/test_map.py` | 测试旧小型 U-Net | 测试代码 | 删除；新测试覆盖 RAUNet、摘要与 V/PP 头 | 被测主干退出 |
| `matcher/tests/test_model.py` | 测试旧 Matcher、阶段加载和 Hungarian | 测试代码 | 删除；按两种聚合和三种损失重建 | 被测模型整体退出 |
| `matcher/tests/test_training.py` | 测试旧训练停止和 checkpoint | 测试代码 | 删除；新测试覆盖早停、恢复和账本 | 训练状态机改变 |

## 讨论、规划、执行记录与映射

| 当前文件 | 当前职责 | 生命周期 | 目标位置与动作 | 具体理由 |
|---|---|---|---|---|
| `talk/Matcher_模块结构概览.md` | 解释旧模块调用关系 | 历史记录 | 原位保留并标明历史身份 | 供 Git 冷读，不代表新结构 |
| `talk/Matcher_输入与产物概览.md` | 解释旧输入和产物 | 历史记录 | 原位保留并标明历史身份 | 保存旧实现事实 |
| `talk/Matcher仓库迁移文件清单.md` | 记录独立仓库迁移时的文件处置 | 历史记录 | 原位保留 | 与本轮重写清单承担不同时间点 |
| `文档/exec_plan/Matcher_Anchor_OOPrime端到端实施.md` | 记录旧 O/O′ 实施过程 | 历史执行记录 | 原位保留 | 不篡改既有执行历史 |
| `文档/exec_plan/Matcher_双模式ABO端到端实施.md` | 记录上一轮失败实现和训练 | 历史执行记录 | 原位保留 | 保存失败证据和指标 |
| `文档/exec_plan/Matcher独立仓库迁移与单包整合.md` | 记录仓库迁移过程 | 历史执行记录 | 原位保留 | 迁移事实仍有效 |
| `文档/mapping/计划执行映射.md` | 连接旧计划、执行记录和代码 | 活动索引 | 原位更新 | 需要把旧计划标记为已被取代并连接新实现 |
| `文档/规划文档/Matcher_Anchor_OOPrime实施计划.md` | 规定旧 O/O′ 模型 | 历史规划 | 原位保留并在映射中标记已被取代 | 保留科学设计演变 |
| `文档/规划文档/Matcher_Anchor_OOPrime数据契约.md` | 规定旧 O/O′ 数据字段 | 历史规划 | 原位保留并在映射中标记已被取代 | 不静默改写旧契约 |
| `文档/规划文档/Matcher_双模式ABO实施计划.md` | 规定上一轮双模式模型 | 历史规划 | 原位保留并在映射中标记已被取代 | 不把新契约混写进旧计划 |
| `文档/规划文档/Matcher_双模式数据契约.md` | 规定上一轮 Dataset 与批次 | 历史规划 | 原位保留并在映射中标记已被取代 | 旧字段仍需可追溯 |
| `文档/讨论/matcher/事实与索引.md` | 汇总旧 Matcher 调研和问答索引 | 历史讨论 | 原位保留并注明新计划优先级 | 事实来源仍可查，但不控制当前实现 |

## 服务器交互与任务提交

| 当前文件 | 当前职责与调用者 | 生命周期与现有验证 | 目标位置与动作 | 具体理由 |
|---|---|---|---|---|
| `与服务器交互/README.md` | 说明 Matcher 安全同步和服务器位置 | 项目基础设施；人工冷读 | 原位更新业务产物根 | 新实验只写 `/storage/penghongen/tmp/Matcher/` |
| `与服务器交互/other/Invoke-PasswordSsh.ps1` | 委托用户级 SSH helper | 项目基础设施 | 原位保留 | 不决定科学任务 |
| `与服务器交互/run_sync.bat` | 启动安全同步 | 项目基础设施 | 原位保留 | 已用于本轮安全同步 |
| `与服务器交互/run_syncWithClean.bat` | 人工启动删除式同步 | 项目基础设施 | 原位保留 | 本轮不调用；保留既有人工工具 |
| `与服务器交互/sync_code.ps1` | 保留远端已有文件并上传本地 Matcher | 项目基础设施 | 原位保留 | 正式安全同步入口 |
| `与服务器交互/sync_codeWithClean.ps1` | 删除后重建远端 Matcher | 项目基础设施 | 原位保留 | 未获删除式同步授权，不调用 |
| `与服务器交互/sync_wandb_remote.bat` | 启动 W&B 日志同步 | 可复用运维工具 | 原位保留 | 与旧 Dataset 无关 |
| `与服务器交互/sync_wandb_remote.ps1` | 同步 W&B 远端日志 | 可复用运维工具 | 原位保留并按需核对项目名 | 不决定模型与监督 |
| `训练与运行/README.md` | 说明通用提交器和旧 Matcher 结果根 | 当前运维契约 | 原位更新任务清单、命令和产物根 | 通用机制保留，项目示例与业务路径变化 |
| `训练与运行/runtime/allocation_runner.sh` | 管理 pre/try/kill/after 四类锁和动态命令 | 项目基础设施；已有模拟测试历史 | 原位保留 | 七个排队 Job 已依赖该文件 |
| `训练与运行/runtime/create_launch.sh` | 写 launch 证据 | 项目基础设施；已有模拟测试历史 | 原位保留 | 逐次执行留证仍是硬要求 |
| `训练与运行/runtime/create_release.sh` | 冻结即将执行的项目副本 | 项目基础设施；已有模拟测试历史 | 原位保留 | 保证排队 Job 执行最新已同步提交 |
| `训练与运行/sbatch/task.sbatch` | 把 Slurm allocation 交给四锁执行器 | 项目基础设施；`bash -n` 与模拟测试历史 | 原位保留 | 七个排队 Job 已保存该脚本 |
| `训练与运行/sh/README.md` | 列出旧 mode1 任务入口 | 当前运维索引 | 原位重写 | 新系统改为实验计划入口 |
| `训练与运行/sh/mode1/README.md` | 解释旧双模式脚本 | 旧运维说明 | 删除 | mode1 活动入口整体退出 |
| `训练与运行/sh/mode1/prepare_manifest_cpu.sh` | 调用旧 `matcher.manifest ground-truth` | 旧任务脚本 | 删除；新建清单审计入口 | 命令与数据定义均已失效 |
| `训练与运行/sh/mode1/train.sh` | 调用旧训练和 GT 推理 | 旧任务脚本 | 删除 | 七个新实验不应经过旧模式 |
| `训练与运行/sh/train_experiment.sh` | 把冻结任务参数交给 `matcher.experiment`；七个排队 Job 已引用 | 新正式薄入口；已通过 `bash -n` | 原位保留，随新 Python 入口验证 | 提前排队所需，且不包含科学逻辑 |
| `训练与运行/submit_task.sh` | 解析资源、提交 Slurm 并传入 release/launch 系统 | 项目基础设施；已有模拟测试历史 | 原位保留 | 七个 Job `340893`–`340899` 已由它提交 |

## 执行顺序约束

1. 在删除旧代码前先提交本文，使每个现有文件的去向可追溯。
2. 只读审计 Stage1、A–G 和语言模型产物，形成冻结清单；`_BLOB_EXCEED` 或缺少完整 F1 产物的 PDB 不进入任何指标分母。
3. SMI-TED Light 289M 缺少某个 occurrence 时只删除该表示候选；只要同一 PDB 仍有可用身份表示，就不因该 occurrence 排除整个 PDB。
4. 基本删除旧活动代码、配置和 mode1 脚本，再按新依赖顺序实现数据契约、模型、监督、训练、评估和实验配置。
5. 实现线保留真实调试提交；最终学习线第一个提交删除全部旧非测试文件，文档随后提交，全部测试变更集中到最后一个学习提交。
