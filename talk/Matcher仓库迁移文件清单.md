# Matcher 仓库迁移文件清单

本文在迁移和合并大量文件之前，逐项记录 AdaLigand 当前 Matcher 文件的职责、调用关系、生命周期、测试覆盖、目标位置和处理理由。源快照为 AdaLigand `Learn/CUMULATIVE@28eb6c3`；最终目标是在独立仓库中只保留一个 `matcher` Python 包。

| 源文件 | 当前职责与调用关系 | 生命周期与验证 | 目标动作与理由 |
| --- | --- | --- | --- |
| `matcher/LICENSE.PocketXMol` | PocketXMol 派生实现许可证；随图与细配对算子分发 | 长期法律文件；人工核对 | 保留到根目录 `licenses/`，避免与运行包混写 |
| `matcher/THIRD_PARTY.md` | 说明 PocketXMol 来源与迁移范围 | 长期归属说明；人工核对 | 保留并改写仓库相对位置 |
| `matcher/README.md` | 旧 Anchor 输入、模块与训练入口契约 | 历史代码 README | 内容归入历史执行说明；不作为最终当前契约 |
| `matcher/__init__.py` | 旧 Anchor 包入口 | 被旧训练与测试导入 | 由最终单一 `matcher/__init__.py` 取代 |
| `matcher/anchor_data.py` | 模拟 bias/context 的旧 Anchor Dataset | 旧 `train.py`、`infer_anchor.py` 与 Dataset 测试 | 退出活跃树；真实过程由 Git 保存 |
| `matcher/anchor_manifest.py` | 旧 Anchor 清单生成 | 旧正式清单脚本 | 退出活跃树 |
| `matcher/batching.py` | 按 occurrence 预算装箱与旧 collate | 旧训练；`test_batching.py` | 把仍适用的装箱逻辑并入正式批次模块 |
| `matcher/consistency.py` | O/O′ 无梯度一致性指标 | 旧训练与 `test_oo_prime.py` | 若正式六头训练仍调用则合并，否则退出 |
| `matcher/element_properties.py` | 元素属性常量 | 双模式 `data_common.py` | 保留到最终 `matcher/element_properties.py` |
| `matcher/graph.py` | 同构分子图、节点边表示与 GatedGCN | 双模式契约/模型；图测试 | 保留为正式基础算子 |
| `matcher/infer_anchor.py` | 旧 Anchor 推理入口 | 旧正式 shell | 退出活跃树 |
| `matcher/map.py` | 轻量三维 MapBackbone 与摘要头 | 双模式 `context.py`；Map 测试 | 保留并由正式 context 模块调用 |
| `matcher/model.py` | 旧粗分支、FinePair、FiLM_plus 及旧整模 | 双模式 `model.py` 复用基础算子；模型测试 | 拆出仍用基础算子并合入单包，删除旧整模入口 |
| `matcher/objectives.py` | 旧 O/O′ 与辅助目标 | 旧训练测试 | 由双模式六头目标取代 |
| `matcher/oo_prime.py` | 旧 O/O′ 解码与指标 | 旧推理测试 | 由双模式解码取代 |
| `matcher/train.py` | 旧 Anchor 两阶段训练和 WarmupPlateau | 双模式只复用调度器 | 保留调度器，删除旧训练入口 |
| `matcher/tests/__init__.py` | 旧测试包标记 | pytest | 随旧测试退出 |
| `matcher/tests/sample_factory.py` | 旧图与 Anchor 测试夹具 | 旧测试及双模式模型测试 | 仅保留双模式仍需的最小图夹具 |
| `matcher/tests/test_anchor_data.py` | 旧 Anchor Dataset 测试 | pytest | 退出 |
| `matcher/tests/test_batching.py` | 旧装箱测试 | pytest | 把仍适用断言迁入正式批次测试 |
| `matcher/tests/test_graph.py` | 图与 GatedGCN 测试 | pytest | 迁入正式测试 |
| `matcher/tests/test_map.py` | MapBackbone 测试 | pytest | 迁入正式测试 |
| `matcher/tests/test_model.py` | 旧整模与 FinePair 测试 | pytest | 保留基础算子断言，删除旧整模断言 |
| `matcher/tests/test_objectives.py` | 旧目标测试 | pytest | 退出 |
| `matcher/tests/test_oo_prime.py` | 旧解码/一致性测试 | pytest | 仅保留仍适用一致性断言 |
| `matcher/tests/test_training.py` | 旧训练恢复与 W&B 测试 | pytest | 由双模式训练测试取代 |
| `matcher_v2/README.md` | 双模式当前代码边界和 Stage3 复用叶子 | 当前代码 README | 迁为最终 `matcher/README.md` 并按新路径核对 |
| `matcher_v2/__init__.py` | 双模式公开入口 | 训练与外部导入 | 迁为最终单包入口 |
| `matcher_v2/batching.py` | 双模式 collate | 正式训练 | 保留 |
| `matcher_v2/context.py` | 密度裁剪、多尺度特征和 PP 构造 | 双模式模型与 Stage3 复用 | 保留 |
| `matcher_v2/contracts.py` | Ligand/Pocket/Stage1Context/MatcherSample 内存契约 | 两个 Dataset、模型与 Stage3 复用 | 保留 |
| `matcher_v2/data_common.py` | 配体读取、建图、中心裁剪与覆盖标签 | 两个 Dataset | 保留 |
| `matcher_v2/data_ground_truth.py` | 不依赖 Stage1 的 ground_truth_context Dataset | 模式一训练/推理 | 保留 |
| `matcher_v2/data_stage1.py` | 读取 F1-centered 指针的 stage1_context Dataset | 模式二 smoke/未来训练 | 保留 |
| `matcher_v2/decode_ground_truth.py` | 模式一六头解码、阈值扫描与指标 | 验证和推理 | 保留并明确模式隔离 |
| `matcher_v2/infer_ground_truth.py` | 模式一推理入口 | 正式 shell | 保留 |
| `matcher_v2/manifest.py` | 两种模式正式清单/指针生成 | CPU 清单入口 | 保留 |
| `matcher_v2/model.py` | 双模式六头粗细 Matcher 与严格零增量条件 | 正式训练 | 保留，改为包内基础算子导入 |
| `matcher_v2/objectives.py` | 六头 Hungarian、硬主损失和连续辅助损失 | 正式训练 | 保留 |
| `matcher_v2/train.py` | 双模式 Phase1/Phase2 训练与验证 | 正式训练 shell | 保留，改为单包入口 |
| `matcher_v2/tests/__init__.py` | 双模式测试包标记 | pytest | 保留 |
| `matcher_v2/tests/test_context.py` | Map/PP 构造测试 | pytest | 保留 |
| `matcher_v2/tests/test_contracts.py` | 双模式契约与零增量测试 | pytest | 保留 |
| `matcher_v2/tests/test_decode.py` | 模式一解码指标测试 | pytest | 保留 |
| `matcher_v2/tests/test_manifest.py` | 清单/指针生成测试 | pytest | 保留 |
| `matcher_v2/tests/test_model.py` | 双模式模型、两阶段恢复与 Hungarian 测试 | pytest | 保留并移除旧包夹具依赖 |
| `matcher_v2/tests/test_training.py` | 训练批次和加载测试 | pytest | 保留 |
| `configs/matcher/anchor_O_O_prime_phase1.yaml` | 旧 Anchor Phase1 正式配置 | 旧训练入口 | 移入历史证据，不作为可运行配置 |
| `configs/matcher/anchor_O_O_prime_phase1_profile.yaml` | 旧 Anchor Phase1 画像配置 | 旧画像入口 | 退出活跃配置 |
| `configs/matcher/anchor_O_O_prime_phase2.yaml` | 旧 Anchor Phase2 正式配置 | 旧训练入口 | 移入历史证据 |
| `configs/matcher/anchor_O_O_prime_phase2_profile.yaml` | 旧 Anchor Phase2 画像配置 | 旧画像入口 | 退出活跃配置 |
| `configs/matcher_v2/mode1/phase1.yaml` | 双模式模式一 Phase1 正式配置 | 正式训练 | 迁为 `configs/mode1/phase1.yaml` |
| `configs/matcher_v2/mode1/phase2.yaml` | 双模式模式一 Phase2 正式配置 | 正式训练 | 迁为 `configs/mode1/phase2.yaml` |
| `ops/profile_matcher_memory.py` | 旧 Anchor 单步显存画像 | 旧模型内部私有函数 | 退出；新模型需要时另写正式薄画像入口 |
| `ops/profile_matcher_throughput.py` | 旧 Anchor 限时吞吐画像 | 旧模型内部私有函数 | 退出 |
| `训练与运行/README.md` | 通用 release/launch/锁提交系统契约 | `submit_task.sh` 使用者 | 保留并改写 Matcher 路径 |
| `训练与运行/submit_task.sh` | 通用 Slurm 提交入口 | 正式人工提交 | 保留 |
| `训练与运行/sbatch/task.sbatch` | 通用 Slurm 资源模板 | `submit_task.sh` | 保留 |
| `训练与运行/runtime/allocation_runner.sh` | pre/after hold 与动态执行控制器 | `task.sbatch` | 保留 |
| `训练与运行/runtime/create_release.sh` | 内容哈希 release 生成 | allocation runner | 保留 |
| `训练与运行/runtime/create_launch.sh` | launch 审计记录生成 | allocation runner | 保留 |
| `训练与运行/sh/README.md` | 正式任务脚本目录说明 | 人类入口 | 保留并改写 Matcher 范围 |
| `训练与运行/sh/matcher/README.md` | 旧 Anchor 正式运行说明 | 旧脚本 | 移入历史说明或退出 |
| `训练与运行/sh/matcher/train_anchor_O_O_prime.sh` | 旧 Anchor 训练脚本 | 旧正式作业 | 退出活跃树 |
| `训练与运行/sh/matcher_v2/mode1/README.md` | 双模式模式一运行说明 | 当前正式入口 | 迁为 `训练与运行/sh/mode1/README.md` |
| `训练与运行/sh/matcher_v2/mode1/prepare_manifest_cpu.sh` | 模式一正式清单生成 | CPU Slurm | 迁为 `训练与运行/sh/mode1/prepare_manifest_cpu.sh` |
| `训练与运行/sh/matcher_v2/mode1/train.sh` | 模式一 Phase1→Phase2→评估 | GPU Slurm | 迁为 `训练与运行/sh/mode1/train.sh` |
| `与服务器交互/run_sync.bat` | 安全同步 Windows 入口 | 调用 `sync_code.ps1` | 保留 |
| `与服务器交互/run_syncWithClean.bat` | 删除式同步入口 | 只在明确授权时调用 | 保留但本次不运行 |
| `与服务器交互/sync_code.ps1` | 项目安全同步 | 本地人工入口 | 保留并固定远端 `/home/penghongen/My_Project/Matcher` |
| `与服务器交互/sync_codeWithClean.ps1` | 项目删除式同步 | 本地人工入口 | 保留并固定 Matcher 精确目录 |
| `与服务器交互/sync_wandb_remote.bat` | 通用 W&B 下载入口 | 调用 PowerShell 实现 | 保留 |
| `与服务器交互/sync_wandb_remote.ps1` | 通用远端 W&B 快照同步 | 人工入口 | 保留 |
| `与服务器交互/other/Invoke-PasswordSsh.ps1` | 统一 SSH 用户级入口的项目薄包装 | AI 轻量命令 | 保留 |
| `与服务器交互/other/readme.md` | AdaLigand 服务器说明与项目专用工具混合文档 | 人类/AI | 不原样迁移；重写为 Matcher 独立说明 |
| `文档/规划文档/Matcher_Anchor_OOPrime实施计划.md` | 旧 Anchor 科学与工程计划 | 历史依据 | 保留为历史计划 |
| `文档/规划文档/Matcher_Anchor_OOPrime数据契约.md` | 旧 Anchor 数据契约 | 历史依据 | 保留为历史契约 |
| `文档/规划文档/Matcher_双模式ABO实施计划.md` | 双模式当前规格 | 当前实现依据 | 保留并更新单仓库路径 |
| `文档/规划文档/Matcher_双模式数据契约.md` | 双模式字段与两种 Dataset 契约 | 当前实现依据 | 保留并更新单包路径 |
| `文档/exec_plan/Matcher_Anchor_OOPrime端到端实施.md` | 旧 Anchor 实施和运行证据 | 历史记录 | 保留为历史执行记录 |
| `文档/exec_plan/Matcher_双模式ABO端到端实施.md` | 双模式实现与旧训练证据 | 当前历史记录 | 保留并追加迁移事实 |
| `文档/讨论/matcher/事实与索引.md` | 早期事实来源索引 | 历史查阅 | 保留为历史索引并标记权重 |
| `talk/Matcher_输入与产物概览.md` | 冷读输入与产物概览 | 人类阅读 | 更新后保留 |
| `talk/Matcher_模块结构概览.md` | 冷读模块调用概览 | 人类阅读 | 更新后保留 |
| `grill_with_memory/07-28-17-50.md` | Question 1–105 决策账本 | 科学决定来源 | 复制为只读历史依据，不在迁移中改写 |

## 行为冻结与验收

迁移前基线是 AdaLigand 实现端点 `7418cc5` 与学习端点 `a4b4570` 的等价 Matcher tree；工作区事实已进一步收口到 `28eb6c3`。迁移过程中不改变 A/B/O 与 A′/B′/O′ 标签、两种 Dataset 的输入语义、4+4 粗细课程、GatedGCN/FinePair/FiLM_plus 数学或模式一解码指标。

验收至少包括路径级清单、Python 可执行语句比较、全部 Matcher 单元测试、两种模式真实小样本、YAML 解析、shell/PowerShell 语法、服务器同步目标与 release/launch 根隔离。无法证明等价的科学变化必须停止并请求用户决定。

