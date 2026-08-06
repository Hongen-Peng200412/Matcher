# Matcher 双模式 A/B/O 端到端实施

本文实时记录 `文档/规划文档/Matcher_双模式ABO实施计划.md` 与 `文档/规划文档/Matcher_双模式数据契约.md` 的实现、验证和运行事实。旧 Anchor O/O′ Matcher 的事故与运行历史保留在 `文档/exec_plan/Matcher_Anchor_OOPrime端到端实施.md`，不复制到本文。

## 当前状态

- 任务状态：实现中。
- 实现分支：`codex/matcher-v2-dual-context`。
- 共同基点：`Learn/CUMULATIVE@a4dde27b1ba1d9b199a7d5eb2cca1ac306069658`。
- 当前主目标：实现两个显式数据模式与共享模型，完整训练、推理和评估 `ground_truth_context`。
- 当前不包含：Stage3 完整 Builder、模式二全量训练、Stage1 新模型效果调优。

## 进度

- [x] 2026-08-05：重新读取 `Pocket_Plus/talk/global.md`、A–G/BOX 契约、Matcher 旧计划与决定账本，确认 P 与 PP 是不同实体；P 是 Find 已落盘伪原子，PP 由完整图概率 top-k 派生。
- [x] 2026-08-05：用户确认 A/B/O 连续监督与 A′/B′/O′ 硬监督；O 使用以 Å 为数值单位的 `1/(1+d)`，O′ 使用 `d<10 Å`。
- [x] 2026-08-05：用户明确授权端到端实现、服务器测试、抢占旧 Matcher Job 335493 的 A800 allocation、正式模式一训练与当前对话 heartbeat。
- [x] 2026-08-05：双线 Git 起点核验通过；`Learn/CUMULATIVE@a4dde27` 是按提交者时间形成的唯一最新提交。既有 Stage1 日志、调度器和记忆修改保留在工作区，不纳入 Matcher v2 的路径级提交。
- [x] 2026-08-05：建立实现分支 `codex/matcher-v2-dual-context`，开始两项只读审查：契约/Stage3 适配性与代码可读性。
- [x] 2026-08-05：建立新版实施计划、数据契约和本文执行记录。
- [x] 2026-08-05：完成 `ground_truth_context` Dataset、正式清单生成入口和 A/B/O、A′/B′/O′ 标签测试。
- [x] 2026-08-05：完成 `stage1_context` 指针、Dataset 和真实 F1-centered smoke；`10ad` 同时完成两种模式的真实读取，模式二得到 20 个候选、16 个 occurrence，以及宽度为 48/112/176 的 V/P/A 特征。
- [x] 2026-08-05：完成共享 Matcher、严格零增量、Phase1/Phase2、六项输出、身份内 Hungarian、连续 SmoothL1、硬标签 focal gamma=2 与粗细辅助损失。
- [x] 2026-08-05：完成模式一推理、阈值评估、正式 Phase1/Phase2 YAML 和 `训练与运行/sh/matcher_v2/mode1/`。
- [x] 完成本地与服务器 smoke、显存/吞吐画像、两类独立审查和修正；本地与服务器回归、双模式真实前向、两类复审和 A800 完整优化步画像均已通过。
- [x] 完成实现线与学习线等价收口；停止旧任务 heartbeat，使用 Job 335493 的锁协议启动新版模式一训练，并为当前对话建立 heartbeat。
- [ ] 记录正式运行身份、W&B、checkpoint 和初始指标；整理 handoff 与 Stage3 无上下文 Agent 提示词。

## 已确认决定

- 新版使用独立 `matcher_v2/`，旧 `matcher/` 不承担兼容层职责。
- 两种数据入口保持显式隔离，不建立通用多模态 Dataset、后处理或推理插件系统。
- 模式二从模式一权重继续训练；Stage1 专属条件缺失时新增分支显式产生逐元素零增量。基础参数可以继续更新，所以退化只保证当前网络执行基础路径，不保证与历史模式一 checkpoint 数值相同。
- 模式一候选与 occurrence 一一对应，候选 mask 使用真实 GT ligand-area；不模拟 Stage1 的 miss、split 或假阳。
- 模式一为未来 U-Net 采样点保留生态位，但当前正式版本只使用 `MapSummaryHead`；模式二 PP 由完整图概率 top-k 定位，并从同一次 U-Net 的四尺度特征取得初始表示。
- Stage3 只复用密度、采样点与条件表示模块，不依赖 Matcher Dataset、训练器、损失、匈牙利或解码器。
- 模式一正式脚本放在 `训练与运行/sh/matcher_v2/mode1/`，在线同步 W&B。正式文件与临时画像严格分目录。
- 新版训练正式启动时不重新申请 GPU；按用户授权经 `kill_lock_335493` 结束旧训练并复用其 A800 allocation。旧任务 heartbeat 随后停止，监控只绑定当前对话 `codex://threads/019fcd2f-281f-73a1-9270-2612fa4830b6`。

## 发现与证据

- 旧 `matcher/model.py` 直接导入 `AnchorSample` 和 `RawGraph`，`Matcher.forward()` 接收 Anchor 专属 `MatcherBatch`；只替换 Dataset 无法形成双模式模型。
- 旧 `matcher/` 已具有可复用的同构分子图、GatedGCN、48³ 小型 U-Net、typed attention、细配对批量计算、身份内匈牙利、checkpoint、W&B 和性能画像经验。
- `Data_Preprocessing/Ori_Data/README.md` 已提供 occurrence、配体模板、真实坐标、受体原子、实验密度和稀疏 ligand-area；模式一不需要生成新的重型科学标签。
- `文档/讨论/BOX-level数据契约.md` 已明确 Find 的 V、P、A 与 centered 几何；PP 不属于该盘上契约。
- 首轮契约审查指出模式一 A 应按 `present=True` 的真实配体原子 10 Å 包络计算，GT ligand-area mask 只承担覆盖监督；代码与数据契约已在写入首轮 Dataset 时同步纠正。
- 模式二零候选 PDB 在训练 Dataset 边界跳过，但未来端到端评估必须把对应 occurrence 计为漏检；该状态不进入模型或 batch。
- 当前 AdaLigand 工作区还包含其他任务的未提交记录与已经暂存的调度器改动。本任务使用精确路径暂存，禁止把它们误混入 Matcher v2 提交。
- 两项首次只读审查分别发现科学契约与可读性缺口。当前工作树已修正：Phase2 六个最终头全部读取 FinePair 表示；A 的 Stage1 特征形成逐原子零初始化增量，V/P/PP 形成 BOX 候选增量；模式二指针冻结候选数和 occurrence 数；验证与推理使用 BF16；阈值扫描不再缓存图和密度；训练支持精确五次验证、事件日志、在线 W&B 和断点续训。
- 新版顶层模型有意复用已经通过旧正式训练 smoke 的主干科学算子，而不复制一份一千余行实现。公开输入已经改为 `matcher_v2/contracts.py` 的中性契约；该复用关系在 `matcher_v2/README.md` 明示，仍需在收口复审中确认不会给人类阅读造成不可接受的跳转成本。

## 验证记录

- 2026-08-05：`python -m pytest -q matcher_v2/tests matcher/tests/test_graph.py matcher/tests/test_map.py` 最新通过，结果为 `22 passed`。新增覆盖完整模型 Stage1 严格零增量、Phase1→Phase2 装载、Phase2 resume 冻结、六头细分支反向传播、同身份 occurrence 交换、PP 四尺度取样梯度和精确五次验证位置。
- 2026-08-05：纯内存 Phase1 完整前向、六项损失和反向传播通过；纯内存 Phase1 checkpoint→Phase2 独立 stem/FiLM_plus/FinePair 前向与反向传播通过。
- 2026-08-05：服务器只读核对 `10ad`、`10rk` 的 A–G 字段。`exp.npz` 使用 `grid/voxel_size/origin`；真实 Find F1-centered 的宽度为 `V=48`、`P=64+48=112`、`A=64+64+48=176`，已写入正式 YAML。
- 2026-08-05：修正后代码安全同步至服务器；服务器同样通过 20 项当时版本回归。真实 `10ad` 以两个候选执行共同模型前向：模式一与模式二均得到有限的六头 `[2,16]` 输出。模式二临时指针由新版生成器冻结 32 个现有 train PDB，只位于 `/home/penghongen/My_Project/tmp/matcher_v2_smoke_20260805/`。
- 2026-08-05：正式清单 CPU Job `336660` 暴露 Conda 在 nounset 下激活失败，未生成产物；两个正式 shell 已统一为只在 Conda 激活期间暂时关闭 nounset。重提 Job `336669`，当前在 `cnode01` 运行且尚未发布正式 manifest。
- 2026-08-05：第二轮契约/Stage3 复审确认没有 mode1 科学阻断。复审同时发现并已修正 manifest CLI 默认参数提前求值，以及 Phase2 resume 未先恢复冻结边界两项工程阻断；后者已经增加回归测试。
- 2026-08-05：第二轮可读性复审确认正式 mode1 YAML、shell 与训练主流程可以进入正式训练；旧 `matcher.model` 主干复用作为首轮过渡边界可接受，首轮完成后再抽取中性共享核心，不在提交训练前复制或重写一千余行代码。审查建议的测试包边界、原子 BEST/COMPLETED JSON 和推理中断安全重跑均已修正。
- 2026-08-05：合并执行 `python -m pytest -q matcher/tests matcher_v2/tests` 通过，结果为 `55 passed, 1 skipped`；两个测试目录可以在同一次 pytest 中运行，不再发生同名模块导入冲突。
- 2026-08-05：按授权创建 `kill_lock_335493`，旧 Anchor 训练在 Phase1 step 5261 后被精确终止；runner 自动创建 `try_lock_335493`，`after_lock_335493` 保留，A800 空闲时仅占 2 MiB，allocation 未释放。旧对话 heartbeat 已删除。
- 2026-08-05：Job `335493` attempt 3 使用临时画像 release `AdaLigand_b84a468ee301`。首个真实 CUDA 优化步在损失阶段发现 assignment 位于 CUDA、标签仍在 CPU 的设备不一致；尚未形成显存结论。代码已改为先把六个标签移动到预测设备再按 assignment 索引，本地 22 项定向回归通过，allocation 再次由 `try_lock` 保留。
- 2026-08-05：正式模式一清单 CPU Job `336690` 成功结束。正式清单包含 train 12,881 个 PDB / 180,382 个 occurrence、validation 188 / 2,914、calibration 89 / 1,386；产物位于 `/storage/penghongen/AdaLigand/Ori_Data/matcher_v2/ground_truth_context/manifest.json`。
- 2026-08-05：Job `335493` attempt 4 使用 release `AdaLigand_e0072b2df641` 完成修正后的 BF16 画像。46 个 occurrence、46 个候选、3 个 PDB 的普通 batch：数据组装 2.85 秒，完整前向、六头损失、反向传播与 AdamW 更新 7.78 秒，峰值已分配/保留显存 4.61/6.04 GiB。最大样本 `6v22` 含 100 个 occurrence 和候选：数据组装 2.54 秒，优化步 6.00 秒，峰值已分配/保留显存 7.49/8.83 GiB。两项均无 OOM，低于人工设定的 71.397 GiB 上限；无需修改模型通道数或正式运行参数。
- 2026-08-05：实现端点 `codex/matcher-v2-dual-context@acdaba9` 与学习端点 `Learn/matcher-v2-dual-context@defbb26` 的 Git tree 完全一致；学习历史按“契约与边界→两种数据输入→共享粗细模型→训练推理评估→集中测试”组织。学习端点再次通过 `55 passed, 1 skipped`，`Learn/CUMULATIVE` 已快进到 `defbb26`。
- 2026-08-05：Job `335493` attempt 5 已使用 release `AdaLigand_aa34cc600a27`、launch `train_anchor_O_O_prime_job335493_20260805T041302_a5` 启动正式模式一训练。正式输出根为 `/storage/penghongen/AdaLigand/Results/matcher_v2/ground_truth_context/seed_3407_occ48`；Phase1 W&B online run 为项目 `AdaLigand_Matcher` 下的 `matcher-v2-gt-3407-p1`。训练进程与 W&B service 存活，GPU 初始占用约 8.4 GiB；当前仍处于 epoch 0 候选准备期，尚未把它计为一次健康优化检查。监控 automation `matcher-v2` 只绑定当前对话，并按 30 分钟频率运行。
- 2026-08-05：attempt 5 已结束 epoch 0 候选准备并完成首次正式验证。`global_step=879` 时，precision=0.15913688、recall=0.08098833、F1=0.10734592、PR-AUC=0.03771764，冻结候选阈值为 0.135；预测 1,483 个匹配，其中 236 个正确，验证集含 2,914 个 occurrence。`latest.pt`、`best.pt` 与 `BEST.json` 已原子落盘；无新 NaN、OOM 或 Traceback，A800 检查时占用约 15.0 GiB。本次计为实际优化健康检查 1/3。
- 2026-08-05：attempt 5 已连续通过三次实际优化健康检查；检查覆盖至 `global_step=2637`，期间无新 NaN、OOM 或 Traceback，A800 峰值观察值约 15.7 GiB。监控 automation `matcher-v2` 已从每 30 分钟改为每 5 小时。随后 `global_step=3516` 的验证 F1 为 0.09458749，未超过当前 BEST；BEST 仍是 step 879 的 F1=0.10734592。
- 2026-08-05：5 小时检查时 Phase1 已进入 epoch 2、`global_step=10547`。当前 BEST 更新到 step 6151：precision=0.13731959、recall=0.11427591、F1=0.12474246、PR-AUC=0.03478840、阈值=0.06；预测 2,425 个匹配，其中 333 个正确。训练进程、W&B online、`after_lock` 和 A800 均正常，无新 NaN、OOM 或 Traceback；检查瞬间显存约 15.7 GiB。
- 2026-08-05：Phase1 在 `global_step=11428` 触发 `scheduler_stop` 并正常写入 `COMPLETED.json`，最终 BEST 保持 step 6151 的 F1=0.12474246。正式脚本随后自动加载 Phase1 BEST 并进入 Phase2。5 小时检查时 Phase2 已到 epoch 1、`global_step=7028`；当前 BEST 为 precision=0.19331586、recall=0.10123542、F1=0.13288288、PR-AUC=0.05658531、阈值=0.132。Phase2 训练与 W&B 进程存活，A800 检查时占用约 24.0 GiB；无新 NaN、OOM 或 Traceback，`after_lock` 保留且无 `try_lock`。
- 2026-08-05：后续 5 小时检查时 Phase2 已进入 epoch 3、`global_step=14076`。当前 BEST 更新到 step 12310：precision=0.22332671、recall=0.11564859、F1=0.15238526、PR-AUC=0.07005863、阈值=0.12；预测 1,509 个匹配，其中 337 个正确。训练与 W&B 进程正常，A800 检查时占用约 24.7 GiB；无新 NaN、OOM 或 Traceback。

## 计划与实现差异

- 有益差异：验证期立即压缩为候选—slot 得分和硬正确矩阵；PR-AUC 改为解码后 precision–recall 曲线；长训练增加最小断点恢复与事件留证。这些变化不改变训练标签。
- 中性差异：当前正式模式一不实现尚未敲定点身份监督的 U-Net 自生成点，只保留 Stage3/模式二需要的采样叶子；与旧计划中的“允许”能力相比是显式延后，不是删除接口。
- 有害差异：当前未发现。
- 未完成范围：双线 Git 收口、正式模式一训练启动与监控、首个 checkpoint 后的正式推理与评估。
