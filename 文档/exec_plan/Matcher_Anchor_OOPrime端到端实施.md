# Matcher Anchor O/O′ 端到端实施记录

## 职责与权威来源

本文按过去时持续记录本轮 Matcher 实施中的代码落点、验证证据、实际偏差和服务器运行状态。它不重新定义科学契约。最新决定以 `grill_with_memory/07-28-17-50.md` 的 Question 1–105 为最高优先级，执行规格见 `文档/规划文档/Matcher_Anchor_OOPrime实施计划.md`。

## 当前状态

- 状态：实施中。
- Git 基点：`Learn/CUMULATIVE@35d3e0e`。
- 实现分支：`codex/matcher-server-smoke`。
- 正式服务器提交：尚未授权；只允许完成实现、测试、隔离 smoke、显存画像和正式脚本准备。

## 2026-08-03：实施启动

已完成以下启动检查：

- 确认 `Learn/CUMULATIVE` 位于仓库按提交者时间形成的唯一最新提交，工作区清洁后创建实现分支。
- 重新核对五份事实/计划文档和 Question 1–105。最新账本覆盖旧计划中关于候选采样、粗分支微观结构、Phase2 接力、细分支、辅助监督和 O/O′ 解码的冲突描述。
- 确认 `matcher/` 在 HEAD 中没有正式源文件，因此本轮从清楚的职责边界开始，不兼容历史 Matcher 代码。
- 确认当前正式数据清单、BOX pool、配体对象、真实配体坐标、受体图和实验密度均已存在，可支持 Anchor 路线训练。
- 冻结可读性边界：当前路线使用显式 `anchor_data.py` 与 `oo_prime.py`；不创建 Dataset/decoder 工厂、模态注册器、Trainer/Callback 层级或 `utils.py`。

## 计划中的验证证据

后续每完成一个阶段，在本节追加实际命令、通过数、真实样本、GPU 型号、峰值显存、checkpoint 接力结果和发现的偏差。临时 smoke 产物只进入 `tmp/`；正式清单、YAML、shell、checkpoint 和评估结果使用各自正式目录。

## 开放门槛

1. 完成本地实现与 CPU 测试。
2. 完成真实数据读取和单步 forward/backward。
3. 在空闲 A100/A800 上完成隔离 smoke。
4. 使用完整 Phase2 训练步和用户指定的 90% 显存上限完成 U-Net 通道画像。
5. 准备正式 YAML 和 `.sh`，向用户报告后请求正式提交授权。

## 2026-08-03：本地实现第一轮闭环

已经建立当前 Anchor 路线的正式数据、模型、目标函数、训练和 O/O′ 推理骨架：

- `matcher/anchor_manifest.py` 一次性生成版本化实验清单；`AnchorPocketDataset` 只读取该清单和完整 BOX 池。训练期先固定本 epoch 的 synthetic-anchor 候选、跳过零候选 PDB，再交给 occurrence 预算装箱。
- `MatcherBatch` 沿候选轴打包 Map，并分别打包配体图与 A 图；集合注意力仍在模型主前向中按 PDB 显式隔离。
- 模型包含可配置的 4+4 个 Matcher Block、小型 48³ U-Net 与四尺度 `MapSummaryHead`、PocketXMol 式节点—边联合更新、八个独立 O/O′ 预测头，以及 Phase2 独立 stem、可选零初始化 `FiLM_plus`、FinePair 分支和两类辅助头。
- FinePair 原子辅助监督保存为 `[C,S_pred,S_gt]`，先完成最终 O/O′ Hungarian，再选择预测槽位实际分配到的真实 occurrence，避免同身份槽位编号泄漏。
- 训练入口支持每 epoch 五次完整验证、同构 O/O′ decoder/evaluator、全状态断点恢复、原子发布 checkpoint 与 BEST、BF16、AdamW、warmup-plateau 和两阶段 model-only 接力。
- 当前 profile YAML 只是通道画像起点，正式训练 YAML 尚未生成；正式服务器训练仍未获授权。

第一轮契约审查发现的七项阻断偏差均已修正：FinePair 对齐、正式 O 的 focal+Dice 权重、PocketXMol 节点消息、Phase2 冻结模块 `eval`、零候选过滤后装箱、Block 数可配置、Map 显式配置和立即失败。随后又补齐了 batch 打包、图边更新消融开关、训练历史最佳值恢复和来源清单身份记录。

本地验证命令：

```text
D:\Anaconda\envs\Pocket_Plus_windows\python.exe -m pytest matcher/tests -q
```

第一轮结果为 26 项通过；覆盖 Map 与 FinePair activation checkpoint 等价、FinePair 普通分块与 checkpoint 分块的输出和梯度等价、Phase2 冻结后 Map 梯度穿透、重复身份槽位交换、正式 O 损失、严格 Phase1→Phase2 加载和 occurrence 装箱。

下一道门槛是第二轮双审查、服务器环境与 18 Å 数据统计核对、真实样本 CPU 预检、隔离 GPU smoke 和显存画像。所有 smoke 与画像产物继续进入 `tmp/`；正式清单不得进入临时目录。

## 2026-08-03：数据边界与运行依赖收紧

第二轮审查后完成以下收紧：

- 图编码改编已逐项对齐 PocketXMol 的 gated edge-message 公式；补充上游 MIT 许可全文和版权声明。
- 图编码改为先打包 batch 内互不连边的图，再在每个 Block 后按 PDB 切回粗分支；打包与逐 PDB 前向的数值等价测试通过。
- 零候选 PDB 现在只存在于 `prepare_epoch` 的样本收集边界。sampler 只接收 `available_indices`；`AnchorSample`、collate、训练、验证、推理和显存画像都不再接受 `None`。
- 新增字段级自包含契约 `文档/规划文档/Matcher_Anchor_OOPrime数据契约.md`。Dataset 会立即核对实验清单的 `schema_version`、`route` 和六项候选抽样参数；清单同时记录来源 BOX manifest 与 config 的 SHA-256。
- 服务器 Torch 环境缺少 `gemmi`。为避免修改共享环境，使用既有 `AdaLigand_stage1_py310` 环境只读导出 0–127 号元素的四项属性，并固化到 `matcher/element_properties.py`。运行时不再依赖 Gemmi；后续真实数据 smoke 仍需逐字段对照原实现。
- O/O′ 独立评估使用 checkpoint 冻结阈值，不在测评时重新扫描。当前正式路线是单卡，因此没有提前保留尚无调用者的分布式汇总边界。
- 断点恢复保存历史最佳 F1/阈值，并在恢复到已经完成第三次降学习率的 checkpoint 时立即正常结束。DataLoader 使用独立随机生成器，不消费模型的全局随机流。

本地验证命令保持不变，最新结果为 33 项通过，`python -m compileall -q matcher ops` 与 `git diff --check` 同时通过。第三轮契约与可读性审查已经闭合；审查确认数据、模型、训练和可读性没有剩余提交阻断项。

## 2026-08-03：本地历史闭合与服务器预检

- 真实实现提交为 `e699a00`。学习历史从同一累计基点按“图与 Map 基础 → 数据与清单 → 两阶段模型与目标 → 训练、推理和运行”重建为 `d42ecb4`、`e89f061`、`eb08bea`、`12206cd`。
- 实现端点与学习端点的 Git tree 均为 `a79f3b2cd7296d6a9574f095ac0d7bfa2afe41d6`；`Learn/CUMULATIVE` 已快进到 `12206cd`。
- 项目安全同步的上传主体完成后，本地等待在整仓权限整理阶段超时。没有盲目重跑：远端关键 Matcher 文件与本地 SHA-256 一致，远端无本次遗留 rsync，新增 Python 文件权限为 755、配置和运维脚本为 644。
- 服务器 `Pocket_Plus_centos7_cu121_allgpu` 环境独立执行全部 Matcher 测试，结果为 33 项通过、13 条已知 PyTorch 性能/弃用提示；未发现环境差异失败。
- 正式 manifest 生成任务为 CPU Slurm Job `334806`，唯一正式目标为 `/storage/penghongen/AdaLigand/Ori_Data/matcher/anchor_O_O_prime_v1/manifest.json`，日志位于 `/home/penghongen/My_Project/tmp/matcher_manifest_20260803/`。当前任务仍在运行，正式文件只会在完整构造后原子出现。
- A800 分区是 `nvlink`，节点 `gnode09/gnode10` 当前可提供隔离 smoke 资源。一次性 smoke 脚本已写入服务器项目的 `tmp/matcher_anchor_smoke_20260803/`，不属于正式入口；它会使用小型临时 manifest 完成 Phase1、精确 BEST 接力、Phase2、冻结阈值推理与 validation 评估。

## 2026-08-03：正式清单与 A100 隔离 smoke

- 正式清单生成 Job `334806` 以 `COMPLETED 0:0` 结束。正式文件为 `/storage/penghongen/AdaLigand/Ori_Data/matcher/anchor_O_O_prime_v1/manifest.json`，SHA-256 为 `9978178c26952ef2a6f15e373fa8cb1428368132093eb503ca11491e2909b795`。清单包含 12881 个训练 PDB、180382 个训练 occurrence；验证集原有 188 个 PDB、2914 个 occurrence，其中 6 个零候选 PDB 只在样本收集边界排除，最终保留 182 个 PDB、2907 个 occurrence。
- A800 任务因账号当时达到 `QOSGrpSubmitJobsLimit` 而在生成 Job ID 前被调度器拒绝，没有留下 A800 锁或作业。随后在 `gnode08` 的一张 NVIDIA A100-PCIE-40GB 上以 simple 模式运行隔离 smoke，Slurm Job 为 `334808`，临时产物位于 `/home/penghongen/My_Project/tmp/matcher_anchor_smoke_20260803/`。
- smoke 的第一次执行发现临时 shell 的 `set -u` 与 Conda 激活脚本冲突；只修改临时脚本为 `set -eo pipefail`。第二次执行在真实配体结构化数组上发现 NumPy 字段步长不满足 `torch.from_numpy` 要求；正式 Dataset 在 NumPy→PyTorch 边界显式复制三个相关字段，并增加非对齐结构化 dtype 回归测试。第三次执行进入 BF16 前向后发现细配对承载张量错误继承 FP32 dtype；承载张量改为分别继承 A 与配体节点状态的实际 dtype/device，并增加 CPU BF16 autocast 前向与反向回归测试。
- 修复后的本地 Matcher 回归为 35 项通过；服务器环境的两个专项测试分别为 5 项和 6 项通过。第四次 smoke 完成 Phase1 三个训练步和完整验证，从 Phase1 `BEST.json` 指向的精确 checkpoint 接入 Phase2，再完成 Phase2 三个训练步、完整验证和冻结阈值推理评估。最终 `summary.json` 状态为 `passed`，短跑验证集含 3 个 PDB、8 个 occurrence，阈值 0.01，precision/recall/F1 均为 0.125。该数值只证明端到端管线闭环，不承担科学结论。
- Phase1 checkpoint 为 928622646 字节，Phase2 checkpoint 为 681062731 字节；后续正式运行除 GPU 峰值外还需考虑 checkpoint 容量。Job `334808` 最终为 `COMPLETED 0:0`，耗时 14 分 42 秒，MaxRSS 2068512K；产物核对后释放 `after_lock_334808`，没有再次执行。

下一道门槛是完整 Phase2 训练步的 A800 80GB 显存画像。画像以 72 GiB 为首轮人工上限，记录实际 batch、峰值 allocated/reserved 和 OOM；画像确定 U-Net 通道后才能生成正式 YAML 与 `.sh` 并请求正式长训练授权。

## 2026-08-03：A800 初步画像、重负载补测与正式入口

- A800 QOS 组释放提交位后，以 simple 模式启动 Job `334813`，节点为 `gnode10`，GPU 为 NVIDIA A800-SXM4-80GB，实际可见显存 79.325 GiB。初步画像运行一个 Phase2 forward、正式 loss、backward、梯度裁剪和 AdamW step；Phase1 权重数值不影响显存，因此画像允许省略 checkpoint，并仍按正式 Phase2 规则冻结前四个 Block。
- 初步画像使用只保留正式训练清单开头 128 个 PDB 的临时清单，实际 batch 为 9 个 PDB、55 个 occurrence、103 个候选、52556 个 A 原子。`map_channels=[16,32,48,64]`、`[24,48,72,96]`、`[40,80,120,160]` 的 allocated/reserved 峰值分别为 44.042/46.928、49.614/52.328、60.756/68.229 GiB。Job `334813` 为 `COMPLETED 0:0`，总 allocation 用时 8 分 33 秒。
- 阶段审查确认上述结果不能冻结正式通道：临时清单改变了 synthetic-anchor 的训练清单长度，因而改变候选抽样种子；首个 55-occurrence batch 也不能代表已约定的两类验收负载。`[40,80,120,160]` 已撤回为待测值，不作为正式结论。
- `ops/profile_matcher_memory.py` 随后改为先预热一个完整训练步，再重置峰值并测量第二个完整训练步；CUDA OOM 会区分 `model_setup`、`warmup` 与 `measurement` 落盘。重负载验收使用临时冻结候选快照，正式工具只调用 Dataset 公共接口。
- A100 全清单基线 Job `334810` 因逐 PDB 准备 epoch 0 的串行 I/O 达到 1 小时时限，最终 `TIMEOUT`；它未进入 GPU 测量，也没有产生可用于选通道的结果。
- CPU Job `334816` 在 45 分 54 秒内从正式训练清单的 epoch 0 选出两类精确候选并以 `COMPLETED 0:0` 结束。正常负载为 manifest 索引 9286、PDB `9cpk`：64 个 occurrence、177 个候选、106044 个 A 原子；65–100 occurrence 的最重单 PDB 为索引 10748、PDB `9kdv`：100 个 occurrence、235 个候选、156743 个 A 原子。两者的候选起点已固化到 `tmp/matcher_memory_profile_20260803/` 的临时快照清单。
- A800 simple Job `334818` 在 `gnode10` 依次定位峰值来源。正常负载在 `[40,80,120,160]` 时于预热步 OOM，allocated/reserved 为 78.557/78.701 GiB；下调到 `[24,48,72,96]` 仍为 77.932/78.318 GiB；把 `fine_pair_chunk_size` 从 2048 改为 1024 得到与首轮相同的 78.557/78.701 GiB；开启 Map activation checkpoint 仍为 78.216/78.805 GiB。上述对照证明主要峰值不是 U-Net 通道、FinePair chunk 或 Map 激活。
- 代码检查定位到 106044–156743 个 A 原子及其 radius 边在四个 Phase2 GatedGCN Block 中保留的反向激活。新增单一 `graph_activation_checkpoint` 开关，通过非重入 checkpoint 重算完全相同的节点—边更新；没有改变图、特征、监督或注意力拓扑。输出与参数梯度等价回归通过。可读性终审随后删除未被单卡正式路线调用的分布式评估汇总及其 noop 测试，并删除 AdamW 从同一参数列表构造后重复执行的恒真校验；最终本地回归为 35 项通过。
- 开启图 checkpoint 后，正常负载完成预热和正式测量，allocated 峰值降至 68.072 GiB；但 CUDA 分配器在预热后保留缓存，reserved 仍为 76.545 GiB，正式测量步耗时约 640 秒。Job `334818` 随后达到 1 小时时限，未开始有效的超大 PDB 测量。
- A800 Job `334836` 对同一正常负载启用 `expandable_segments:True` 后完成两步画像，allocated/reserved 为 68.037/76.025 GiB，仍未达到 reserved 不超过 72 GiB 的人工门槛。该结果证明分配器设置只能缓解约 0.52 GiB 的缓存峰值，不能使 `[40,80,120,160]` 成为正式通道。
- 不改变图半径、邻居数或其他科学契约，Job `334837` 使用 `[24,48,72,96]`、图 activation checkpoint 和同一分配器完成 `9cpk` 两步画像。64 个 occurrence、177 个候选、106044 个 A 原子对应 allocated/reserved 48.870/56.670 GiB，低于 72 GiB 门槛；正式测量步耗时约 629 秒。JSON 同时记录了 `cuda_allocator_config=expandable_segments:True`。
- Job `334841` 以完全相同的模型与显存设置验收 `9kdv`：100 个 occurrence、235 个候选、156743 个 A 原子。作业持续有效 GPU 计算且未报告 OOM，但在 90 分 06 秒达到 Slurm 时限，没有完成第二个优化步，也没有生成结果 JSON；因此该次结果是 `TIMEOUT`，不能用于通过或否决显存门槛。
- Job `334868` 只把同一重负载画像的隔离时限放宽到 2 小时 30 分，模型、冻结候选、BF16、通道、chunk、图 checkpoint、分配器和 72 GiB 门槛均不改变。只有它完整写出成功 JSON，才会把 24 通道组回填为正式值。
- 新增正式配置 `configs/matcher/anchor_O_O_prime_phase1.yaml`、`configs/matcher/anchor_O_O_prime_phase2.yaml`，两者使用相同的固定通道与正式 manifest。新增正式入口 `训练与运行/sh/matcher/train_anchor_O_O_prime.sh`：它只接受全新正式输出根，依次运行 Phase1、从 Phase1 `BEST.json` 精确接力 Phase2、从 Phase2 `BEST.json` 使用冻结阈值完成 validation 推理评估。正式输出根为 `/storage/penghongen/AdaLigand/Results/matcher/anchor_O_O_prime_v1/seed_3407/`，不依赖任何 smoke 或画像临时文件。
- 两份正式配置与 shell 的结构已经生成，但 `map_channels` 仍等待两类重负载的双步画像后最终固定。服务器环境已通过正式 shell 语法、训练模块和推理模块的静态入口检查；没有提交正式长训练。

当前门槛包括两类重负载画像、正式通道回填、阶段性双审查、双线 Git 再闭合和用户人工授权。获得明确授权前不得运行 `训练与运行/submit_task.sh` 的正式 A800 命令。

## 2026-08-03：FinePair 等价加速与无限时 A800 工作台

- Job `334868` 在运行 2 小时 30 分 21 秒后达到 Slurm 时限，没有生成结果 JSON。它与 `334841` 一样只证明旧实现持续计算，不能作为超大 PDB 的显存或吞吐验收结果。后续画像任务不再设置 Slurm 时限。
- `9cpk` 单个正式测量优化步约 629 秒，不能接受。代码审计确认主要问题位于 FinePair 执行方式：旧实现虽然配置了 chunk，chunk 内仍按候选框—occurrence 配对逐次执行 Python 循环和独立 attention；带 padding 的 PyTorch scaled-dot-product attention 也无法利用真实变长序列减少计算。
- 性能修复没有改变完整 `C×S` 配对网格、候选、标签、Hungarian、损失或归一化。当前实现把 chunk 内全部候选框—occurrence 配对作为一个批量执行；配体和 A 先补齐并携带真实原子掩码，每个 chunk 再裁去不需要的尾部 padding。接触标签在 activation checkpoint 外按候选框—真实 occurrence 批量构造并复用，辅助损失仍返回 `[C,S_pred,S_gt]`。
- CUDA BF16/FP16、单上下文且无 attention bias 时，`TypedAttention` 会先移除 padding，再调用 `flash_attn_varlen_func`；其它情况仍使用 PyTorch scaled-dot-product attention。显式 Flash 是执行后端优化，不改变 attention 数学定义。空 A、`present=False`、可选 FFN 和 attention bias fallback 均有独立回归。
- 本地完整 Matcher 测试为 40 项通过、1 项 CUDA/Flash 专项因 Windows 无 CUDA 跳过。契约审查确认同身份多真实 occurrence 形成完整 `[S_pred,S_gt]` 笛卡尔关系，并与已有 Hungarian 槽位交换测试共同覆盖最终选择；可读性审查确认生产代码只新增一个补齐函数，没有新增后端类或通用框架，chunk 是 `_run_fine_pairs` 中唯一保留的性能循环。
- 新增 `ops/profile_matcher_throughput.py`。该入口强制 Phase2，复用正式 Dataset、occurrence 装箱、loss、BF16、反向传播、梯度裁剪与 AdamW，在用户指定时长内记录平均 batch、PDB、occurrence 时间、数据等待、峰值显存及 OOM 阶段。它不执行周期性验证或保存 checkpoint，因此只回答训练步吞吐。
- 无限时 A800 工作台为 Job `335261`，节点 `gnode10`，16 CPU，QOS `cpu96`，Slurm `TimeLimit=UNLIMITED`。当前仍保留 `pre_lock_335261`，尚未执行旧画像脚本。加速代码已通过项目安全同步上传；随后登录节点在 SSH 密钥交换前主动断开连接，因此没有删除 pre-lock，也没有误启旧命令。连接恢复后首先执行服务器 CUDA Flash↔SDPA 输出与梯度等价测试，再分别测量 `9cpk` 极端样本和普通 Dataset 一小时吞吐。

## 2026-08-03：FinePair、数据建图与分阶段性能画像

- 服务器连接恢复后继续使用无限时 A800 Job `335261`。CUDA 专项测试实际进入显式 varlen Flash 后端，并与强制 PyTorch scaled-dot-product attention 的输出和参数梯度一致。BF16 smoke 先后暴露两个承载张量 dtype 问题：FinePair attention 增量必须转回残差张量 dtype，辅助 focal loss 网格必须显式保持 FP32。两项修复均增加 BF16 autocast 回归；本地 Matcher 测试当时为 40 项通过、1 项 CUDA 专项跳过。
- 当前 `heavy_snapshot_manifest.json` 实际只包含更重的 `9kdv`，不能再按旧临时文件名误称 `9cpk`。冻结样本含 100 个 occurrence、235 个候选框和 156743 个 A 原子。`map_channels=[24,48,72,96]`、`occurrence_budget_per_batch=64`、`map_candidate_chunk_size=64`、`fine_pair_chunk_size=2048`、图 activation checkpoint、BF16 和 `expandable_segments:True` 下，无探针完整优化步为 13.007 秒；allocated/reserved 峰值为 69.942/71.383 GiB，满足人工指定的 72 GiB 门槛。
- 同一完整步的阶段计时为：forward 4.438 秒、正式 loss 与 Hungarian 0.293 秒、backward 8.188 秒、梯度裁剪 0.052 秒、AdamW 0.028 秒。forward 的主要组件为八层粗分支与回吸约 1.66 秒、八层 A 图网络约 0.92 秒、FinePair 0.64 秒、四个粗预测头约 0.30 秒、Map Backbone 0.27 秒。组件 CUDA Event 存在父子嵌套，只用于定位热点，不能直接求和重建阶段 wall time。
- 数据函数画像证明 170.57 秒样本组装中的 165.48 秒位于 `build_molecular_graph`；磁盘读取和 `torch.cdist` 均不是主因。旧实现对边字段逐条创建小张量，调用 `_distance_rbf` 2097548 次。提交 `ee0af56` 把 radius/chemical 字段和 RBF 改为每张图批量构造；提交 `9f9d76a` 再把逐 target 邻居循环改为距离矩阵按 target 列稳定排序。半径、同几何组、每 target 最多 48 邻居、等距来源索引优先、化学边覆盖、跨组距离、最终有向边排序和 6 类边标签均保持不变。
- 数据优化通过 50 组随机图逐元素等价、重复反向化学边最后覆盖、等距邻居稳定截断和完整 Matcher 回归；最新本地结果为 42 项通过、1 项 CUDA 专项跳过。服务器同一 `9kdv` 的 cProfile 样本组装从 170.57 秒降为 6.01 秒，普通无探针组装为 5.35 秒。新建图临时矩阵约占 `18*N^2` 字节；当前 18 Å 候选区域与 4 个 DataLoader worker 可接受，未来单候选扩展到数千原子或完整受体时必须重新评估。
- backward 算子画像的自身 CUDA 时间中，3D convolution backward 约 1.57 秒，LayerNorm backward 约 1.05 秒，copy 约 0.53 秒，矩阵乘约 0.38 秒，index/index_put 合计约 0.67 秒，显式 varlen Flash backward 约 0.29 秒。该结果说明 FinePair 已不再是首要热点；下一种可能的等价优化是联合批处理候选级粗分支和回吸，但它会显著增加模型代码复杂度，必须先看普通 Dataset 平均吞吐再决定。算子表保留为临时文本，483 MB 的一次性 Chrome trace 已删除。
- 普通训练 Dataset 的首次 60 分钟吞吐执行没有生成 JSON。四个 DataLoader worker 长时间向主进程传递含大量独立张量的 `AnchorBatch`，默认 `file_descriptor` 共享策略最终触发 `OSError: [Errno 24] Too many open files`。修复仅在当前 Anchor 的训练、吞吐画像和独立推理三个入口中，当 `num_workers>0` 时显式设置 `torch.multiprocessing` 的 `file_system` 共享策略。它不位于 Dataset、collate 或公共数据层，不改变候选、batch 字段与数值、随机数、模型或损失，也不与未来 Stage1 数据入口耦合。
- 修复后的完整训练集 epoch-0 候选准备约用 59 分钟；该时间发生在 60 分钟正式测量窗口之前，不得被平均 DataLoader 等待时间掩盖。后续 3602.322 秒测量完成 708 个正式 Phase2 优化步、2,755 个 PDB 和 37,925 个 occurrence；平均 batch wall time 为 5.0880 秒，其中优化步 5.0606 秒、数据等待 0.02744 秒，折合 1.3076 秒/PDB 和 0.09499 秒/occurrence。结果为 `status=ok`、无 OOM，allocated/reserved 峰值为 71.150/72.910 GiB；物理显存仍有约 6.4 GiB 余量，但严格的 72 GiB 人工门槛超出 0.910 GiB，因此必须如实记为 `within_memory_limit=false`。
- 真正的 `9cpk` 冻结样本含 64 个 occurrence、177 个候选框和 106044 个 A 原子。数据组装为 3.137 秒，完整 Phase2 优化步为 9.476 秒，allocated/reserved 峰值为 48.873/49.775 GiB，无 OOM 且通过 72 GiB 门槛。更重的 `9kdv` 既有结果仍为 13.007 秒和 69.942/71.383 GiB。
- Job `335261` 的所有画像结果核对后，在 `try_lock` 空闲态删除精确 `after_lock_335261`；作业以 `COMPLETED 0:0` 结束，总 allocation 用时 6 小时 32 分 57 秒。该 simple 工作台不会被复用为正式训练；正式训练仍未启动，必须先展示 YAML、shell 和无 `--time` 的 A800 提交命令并获得用户明确授权。

## 2026-08-04：正式 Phase1/Phase2 训练已授权提交

- 用户在审阅正式 Phase1/Phase2 YAML、`train_anchor_O_O_prime.sh` 和无 `--time` 的 A800 命令后，明确授权启动正式训练。用户同时确认 72.910 GiB reserved 相对 72 GiB 的小偏差不构成启动阻断。
- 提交前门禁确认正式输出根 `/storage/penghongen/AdaLigand/Results/matcher/anchor_O_O_prime_v1/seed_3407` 不存在，且没有同名 Matcher Job。实际命令使用 `a800`、1 GPU、16 CPU、QOS `cpu96`、完整 release/launch 模式，没有 `--simple`、`--hold` 或 `--time`。已核对该 Job 实际读取的服务器旧四锁 runner：任务成功、失败或 `kill_lock` 中断后都会创建 `try_lock_335493` 并保留 allocation。作业启动前不得同步那套改为“默认自动释放”的未提交调度器改动。
- Slurm 返回正式 Job `335493`，作业名 `matcher_anchor_OOprime_v1`。提交后首次核对为 `PENDING (Priority)`，`Partition=nvlink`、`QOS=cpu96`、`TresPerNode=gres:gpu:a800:1`、`NumCPUs=16`、`TimeLimit=UNLIMITED`；正式输出根尚未创建。release 与 launch 依项目调度契约在作业实际执行前才冻结，不在排队时伪造。
- 已创建只绑定当前对话 `codex://threads/019fc69d-c101-7d20-9ab2-4d16e4433419` 的 heartbeat `matcher`。排队阶段每 3 小时检查一次，不计入训练健康次数；检测到首次 `RUNNING` 后自动改为每 30 分钟一次。实际训练连续三次健康后再改为每 5 小时一次；监控不得修改、取消或重提任务，也不得删除 `try_lock_335493` 或对应 `after_lock`。

## 2026-08-04：正式 Job 进入运行与首次启动核对

- Job `335493` 于 `2026-08-04 05:09:07 +0800` 在 `gnode10` 进入 `RUNNING`。heartbeat 已按约定从排队期每 3 小时改为每 30 分钟；本次只是首次运行状态核对，尚未计为一次完成优化步后的训练健康检查。
- release 已冻结为 `AdaLigand_edd59b6263cb`，launch 已冻结为 `train_anchor_O_O_prime_job335493_20260804T051115_a1`。`after_lock_335493` 存在，正式 Phase1 配置已写入输出根的 `phase1/resolved_config.yaml`；未发现 NaN、OOM、Traceback 或其他错误。
- 首次核对时训练进程位于 `AnchorPocketDataset.set_epoch(0) -> prepare_epoch -> sample_candidate_starts -> _A_indices`，正在逐 PDB 读取 `exp.npz` 并准备 epoch 0 候选。10 秒内底层读取量从约 83.85 GiB 增至 84.43 GiB，说明进程持续前进而非挂死；此阶段尚未进入 GPU 优化步，因此 A800 暂时空闲。此前同一完整训练清单的候选准备实测约 59 分钟，本次行为与该已知启动成本一致。
- `06:03` 后 epoch 0 候选准备完成：保留 12,678 个 PDB，按既定样本收集边界跳过 203 个零候选 PDB；得到 161,759 个 bias 候选和 179,602 个被接受的 context 候选。随后 Phase1 已连续写出前三个正式优化步，`loss_total` 分别为 1.9133、1.3785、1.4448，所有分项有限；A800 已由主训练进程占用约 25.8 GiB。未发现 NaN、OOM、Traceback 或异常退出，本次记为连续训练健康核对 `1/3`。
- 连续三次训练健康核对已完成，heartbeat 按约定改为每 5 小时。第三次核对时 Phase1 已到 `global_step=753`，A800 约占用 75.5 GiB 且持续计算；未发现 NaN、OOM、Traceback 或错误。
- `global_step=684` 完成首次验证并保存约 941 MB 的 checkpoint。当前 `BEST.json` 指向 `checkpoint_step_00000684.pt`：`F1_O_O_prime=0.08603167`、precision `0.13746224`、recall `0.06260750`、`O_threshold=0.04`。这是训练早期的首个验证坐标，不作为最终效果结论。
- 后续四次验证的 F1 依次为：step 1368 的 `0.11188`、step 2052 的 `0.13414`、step 2736 的 `0.13656`、step 3419 的 `0.14742`，每次都产生新的 BEST。当前 BEST 为 step 3419：precision `0.21019`、recall `0.11352`、`O_threshold=0.01`；验证损失相对首次从 `49.74` 降至 `46.33`。Phase1 已进入 epoch 1 并推进至约 step 3682。
- 本次 GPU 快照显示训练主进程约占用 80,122 MiB，接近 A800 80 GiB 容量，但作业仍持续计算且没有 OOM。该值是 `nvidia-smi` 的进程占用快照，不等同于 PyTorch 的 allocated/reserved 峰值；当前不据此改动配置，只在后续 heartbeat 继续观察。
- Phase1 在 epoch 1 完成 `global_step=4069` 后，于下一批次的 backward 发生 CUDA OOM。异常位于 activation checkpoint 重算 A 图网络的边更新 `LayerNorm`；当时 PyTorch 已分配 78.11 GiB、保留但未分配 643.63 MiB，只剩 63.81 MiB 可用，而该算子还需 494 MiB。该证据说明本次失败主要是实际 batch 图规模超过显存容量，不是大量未使用 reserved 内存造成的碎片。
- 训练进程已经退出，Phase2 尚未开始；最后可用 BEST 仍为 step 3419、F1 `0.14742`。旧四锁 runner 已创建 `try_lock_335493`，并保留原有 `after_lock_335493`，所以 Slurm Job 仍为 `RUNNING`、A800 allocation 未释放且 GPU 当前空闲。监控没有删除任何锁、重提任务或修改配置，等待用户决定诊断与续跑边界。

## 2026-08-04：显存安全协议与在线 W&B 的第二次正式执行

- 用户授权从头重新训练，不续接第一次执行的 checkpoint，并要求启用 W&B 在线同步。第二次实验保留 `map_channels=[24,48,72,96]`、图结构、模型参数、标签、O/O′/辅助损失公式和完整候选集合；显存安全训练协议把 `occurrence_budget_per_batch` 从 64 改为 48、`map_candidate_chunk_size` 从 64 改为 32、开启 Map activation checkpoint，并把 Phase2 的 `fine_pair_chunk_size` 从 2048 改为 1024。
- `occurrence_budget_per_batch=48` 不只是纯执行分块：它同时改变 PDB 装箱、损失固定除数、每 epoch 优化步数和 warmup 步数，因此使用新实验身份与输出根 `/storage/penghongen/AdaLigand/Results/matcher/anchor_O_O_prime_v1/seed_3407_occ48/`。旧失败根 `seed_3407/` 保持不变。
- W&B 只在正式配置 `wandb.enabled=true` 时延迟导入并以 `mode=online` 初始化；Phase1/Phase2 分别使用 run name `anchor_O_O_prime_v1_seed_3407_occ48_phase1` 和 `..._phase2`，共同归入项目 `AdaLigand_Matcher`、group `anchor_O_O_prime_v1_seed_3407_occ48`。候选统计、训练 loss 与验证指标都携带 `global_step`，W&B 内部提交顺序不替代 `events.jsonl`。W&B 初始化后重新设置 seed，避免第三方初始化影响 Dataset、模型和优化器随机状态。
- 本地完整 Matcher 回归为 43 项通过、1 项 CUDA 专项跳过；服务器定向训练测试为 5 项通过。契约与可读性双审查最终均无阻断。实现端点为 `codex/matcher-oom-recovery@1eba879`，学习端点与累计学习分支为 `Learn/CUMULATIVE@a4dde27`，两端 tree 均为 `e57762b7819dc80fe99a99cdd946e16a0af42db9`。
- 保留 allocation 内确认 W&B `0.17.3`、API key 已配置、`api.wandb.ai:443` 可连接。安全同步虽然因本地 120 秒命令上限退出，但服务器五个目标文件的 SHA-256 与学习端点逐一相同，随后服务器定向测试通过，因此同步证据完整。
- 在新输出根不存在、训练进程为空、`try_lock_335493` 与 `after_lock_335493` 身份正确时，只删除精确的 `try_lock_335493`。第二次执行冻结 release `AdaLigand_1126746ccd75`，launch 为 `train_anchor_O_O_prime_job335493_20260804T194729_a2`；Phase1 的 `resolved_config.yaml` 已写出，W&B online run ID 为 `93omhb5u`。当前处于 epoch 0 候选准备期，heartbeat 已重新设为每 30 分钟，实际优化连续三次健康后再改为每 5 小时。
- 第二次执行的 epoch 0 候选准备已经完成：保留 12,678 个 PDB，按既定边界跳过 203 个零候选 PDB；得到 161,759 个 bias 候选，180,268 个 context 请求中接受 179,602 个。Phase1 随后推进到 `global_step=177`，最近 `loss_total=1.30499`，O、O′ 与辅助损失均为有限值；未发现 NaN、OOM 或 Traceback。最近 GPU 快照约占用 35.4 GiB，本次记为第二次执行的连续训练健康核对 `1/3`。
