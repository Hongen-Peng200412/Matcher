# Matcher 单一 SMILES 主线重构执行记录

## 记录边界

本记录对应 `talk_new/反馈/re-9-16.md` 已获批准的阶段 A—D，以及阶段 C 中不依赖新版真实 BEST checkpoint 的实现、机械门禁和 Stage1-only 结果。实现共同基点是按提交者时间唯一最新的 `Learn/CUMULATIVE@75ac5eaa5f4ec682724b72110605c74ec9b846a2`，实现分支是 `codex/matcher-single-pipeline-refactor`。

本轮不执行四项正式训练，不生成 Matcher 正式测试数值，不生成第一张七栏图或含 Stage1+Matcher 曲线的最终图。第 6 片 Stage1 推理由独立任务负责，本轮不接管，也不把未完成分片写入训练 manifest。

## 输入与只读来源

| 来源 | 用途 | 写入边界 |
| --- | --- | --- |
| `talk_new/反馈/re-9-16.md` | 当前代码、训练和评估契约 | 保留在用户工作区，本轮不擅自暂存或提交 |
| `/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1/` | validation、训练分片和真实受体测试的 Stage1 产物 | 只读 |
| `/storage/penghongen/AdaLigand_stage1_inference/Find_1/CryoAtom2受体/artifacts/Find_1/held_out_test_0/` | CryoAtom2 主测试 Stage1 产物 | 只读 |
| `/storage/penghongen/AdaLigand_stage1_inference/Emap2lig/official_find_li/held_out_test_0/evaluation/` | Emap2lig Stage1-only 外部基线 | 只读；派生元数据只写 Matcher 评估产物根 |
| `/storage/penghongen/AdaLigand/Ori_Data/smiles_assets/` | 全局精确 SMILES 图包 | 只读 |

## 实施路标

| 路标 | 交付物 | 状态 |
| --- | --- | --- |
| 计划补充 | Emap2lig 来源、排序、单曲线语义、命中定义与候选集合差异写入新版计划 | 已完成，未暂存 |
| 阶段 A 前置 | 逐文件保留、迁移、删除清单 | 已完成 |
| 阶段 A | 单一 schema、SMILES 图、目标、manifest、Dataset 和 batch | 已完成机械门禁 |
| 阶段 B | 四变体、二分类、直接身份交叉熵、训练、BEST/LAST 与阈值 | 已完成训练前实现与 CPU 门禁 |
| 阶段 C 本轮范围 | 测试连接、Stage1-only 元数据、Emap2lig 门禁与两张预览图 | 已完成 checkpoint-independent 范围 |
| 独立审查 | 三个独立方向各一轮全面审查，修复后只复核报告问题 | 科学逻辑、函数布局/Git、注释与 YAML 均已独立 PASS |
| 阶段 D | 删除平行和过时活动主线，刷新 README、mapping 与 handoff | 已完成 |
| 双线 Git | 实现历史、学习历史、端点等价与唯一最新累计分支 | 已完成 |

## 正式运行命令

本轮没有正式训练或 Matcher 正式测试命令。已经执行的正式数据派生命令如下；完整参数及来源同时保存在对应 shell 和派生产物中。

```bash
/home/penghongen/anaconda3/bin/python -m ops.matcher.build_stage1_manifest --stage1-root /storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1 --split validation --output-path /storage/penghongen/Matcher/manifests/stage1_smiles_v1/manifest.json --data-root /storage/penghongen/AdaLigand/Ori_Data --graph-asset-path /storage/penghongen/AdaLigand/Ori_Data/smiles_assets/smiles_graphs_v1.npz
```

```bash
/home/penghongen/anaconda3/envs/AdaLigand_stage1_py310/bin/python -m matcher.evaluation.test_set --real-test-root /storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1/held_out_test_0 --cryoatom2-test-root /storage/penghongen/AdaLigand_stage1_inference/Find_1/CryoAtom2受体/artifacts/Find_1/held_out_test_0 --emap2lig-evaluation-root /storage/penghongen/AdaLigand_stage1_inference/Emap2lig/official_find_li/held_out_test_0/evaluation --output-path /storage/penghongen/Matcher/evaluation/stage1_only_preview/stage1_only_preview.json
```

真实受体和 CryoAtom2 的 test_0 manifest 使用同一 `ops.matcher.build_stage1_manifest` 入口，分别写入 `test0_real_smiles_v1/manifest.json` 与 `test0_cryoatom2_smiles_v1/manifest.json`。扩展 top-k 契约冻结后，两张预览图使用以下正式命令从同一份 Stage1-only JSON 重新生成：

```bash
/home/penghongen/anaconda3/envs/AdaLigand_stage1_py310/bin/python -m ops.matcher.plot_stage1_preview --metadata-path /storage/penghongen/Matcher/evaluation/stage1_only_preview/stage1_only_preview.json --output-directory /storage/penghongen/Matcher/evaluation/stage1_only_preview
```

## 机械门禁命令

机械门禁与正式运行命令严格分开。已执行：

```powershell
D:\Anaconda\python.exe -m pytest -q
D:\Anaconda\python.exe -m compileall -q matcher ops
git diff --check
```

审查修复后的本地结果为 `23 passed`，30 个非测试 Matcher 模块全部可导入，语法编译、注释标点机械门禁与差异空白检查通过。`pytest.ini` 把正式测试范围固定为 `matcher/tests`，不再误收集 ignored `tmp/` 中保留的历史审计脚本。

一次性 GPU 门禁原拟通过以下机械命令提交；它不属于正式训练命令：

```bash
bash 训练与运行/submit_task.sh --sh tmp/matcher_gpu_preflight.sh --resource a100 --gpus 1 --cpus 16 --time 06:00:00 --job-name MatchGate
```

该命令产生待调度的 Job 385011，但作业始终处于 `PENDING (Priority)`，没有获得节点、没有创建 release、没有执行 Python 门禁。用户随后明确要求把 GPU 门禁延后到阶段 E 正式运行之前；Job 385011 已用精确 `scancel 385011` 取消并确认从 `squeue` 消失。阶段 E 应基于届时最终代码重新生成一次性门禁，门禁通过后使用新的正式启动命令，不复用本次未执行的 Job 或脚本快照。

服务器真实 validation batch 门禁在 gnode10 使用 PocketXMol 可运行的 PyTorch 2.6.0 环境执行，只读取 CPU 与共享存储，不使用或干扰 Job 378693 的 GPU 进程。四个变体的 Dataset 长度均为 5,544：

- strongest 与 trunk_only 的体素输入为 `(2,106,48,48,48)`；
- density_blob_only 的体素输入为 `(2,1,48,48,48)`，A/P/PP token 轴长度严格为零；
- pocket_only 的体素输入为 `(2,0,48,48,48)`，只保留 `(2,489,50)` 的 A-L0 token；
- strongest、density_blob_only、pocket_only 读取同一 SMILES 图，trunk_only 的图节点和边轴严格为空；
- 四个变体的 PDB 内身份轴、SMI-TED `(B,I,768)`、身份目标和候选顺序一致。

## 变更记录

### 2026-09-17：实施启动

- 只读确认单一工作树，`Learn/CUMULATIVE@75ac5eaa...` 是按提交者时间唯一最新提交。
- 启动时没有暂存改动；唯一工作区改动是用户修订的 `talk_new/反馈/re-9-16.md`。
- 从累计端点建立 `codex/matcher-single-pipeline-refactor`，没有重置到历史 `9e6e974...`。
- 先把 Emap2lig 契约和本轮 checkpoint-independent 边界回填到计划，再建立本执行记录与逐文件清单。

### 2026-09-17：阶段 A—D 训练前实现和真实数据门禁

- 单一活动 manifest 使用精确 `PreparedLigand.smiles`；validation 原始候选 7,413、已选候选 5,554，跳过 10 个不受支持真前景后分别得到 7,403 和 5,544 个训练/验证候选。PDB 内身份记录 603 个、唯一精确 SMILES 208 个。
- 全局 `smiles_graphs_v1.npz` 含 3,497 个精确 SMILES、133,856 个重原子和 140,934 条有向键，SHA-256 为 `e8aad7336ccce54b01e606dfc79a82297f5bf7b86897a393bcfccf0c8d820625`。
- 完整 test_0 的 240 个重复精确 SMILES 组共比较 2,150 对 SMI-TED 向量，最大绝对差为 `5.9604644775390625e-06`。清单构建器据此使用十进制绝对容差 `1e-5`，超过该范围仍失败关闭。
- 真实受体 test_0 manifest SHA-256 为 `ed9ab3212d4a9aeb1be8730f35bd86bc434b3ecca4c72732942b7397ad5f90ce`，含 2,251 个候选、1,653 个已选候选、1,419 个真前景和两个不受支持真前景。
- CryoAtom2 test_0 manifest SHA-256 为 `14f8d2316d0555fb5b794f034ff96a577597149909deb8b2c340164faede7fda`，含 2,407 个候选、1,651 个已选候选、1,403 个真前景和两个不受支持真前景。
- 对两份 test_0 清单逐候选重算完整交集矩阵，最大几何平均、双向 0.30 覆盖与较小 occurrence_id 并列规则的偏差数均为 0。`9ian` occurrence 0 的真实受体/CryoAtom2 已选排名为 3/5；`9oq7` occurrence 10 的排名为 2/1。
- 两种受体条件各有 174 个 PDB 含 centered 候选，另外五个 PDB 为空候选。最终报告器把 manifest 摘要中的全部 179 个 PDB 作为固定 top-k 分母，空候选 PDB 的 k=1…20 均为未命中，不虚构候选。
- 扩展 top-k Stage1-only JSON SHA-256 为 `cd554ebfac2865dfd3c2a67fcef93eb707690c058a36b9e9f1da26722a6a0b62`。Emap2lig top-3/4/5 严格复现 87/89/90，且两种 Find_1 排序引用的 Emap2lig 曲线逐点相同。
- 两张扩展 top-k 预览图分别为 `topk_gaussian_stage1_only_preview.png` 和 `topk_source_probability_mean_stage1_only_preview.png`；SHA-256 分别为 `a05369f1060c59c6ea3a0b870f362eb9330364c811ddc1fa306380ecf8de79a0` 与 `53568cec2f27b936df2f9bfa497e49b63f814deb0dcde4466edab4de47aa6177`。两图没有 Matcher 数值，也未被称为最终图。
- 删除六分类、类别概率到 PairScorer 的连接、旧身份损失变体、seen/unseen、calibration、历史活动 YAML 和平行 `matcher/smiles_identity/` 包。旧实现仍可由 Git 恢复。
- Job 378693 当前由 PocketXMol 真实训练占用，GPU0 显存约 73 GiB 且持续计算；本轮没有触碰其锁或进程。Matcher BF16 前后向和完整 validation GPU 门禁按用户最新决定延后到阶段 E 正式运行之前，不借用该 GPU。

### 2026-09-17：一轮独立全面审查与定向修复

- Git/函数布局审查、代码注释审查和科学逻辑审查各完成且仅完成一轮全面核查，三者均先返回 `FAIL`。后续只允许对本节问题进行窄口径复核，不再扩大范围。
- 科学审查发现四个变体在同一 seed 下会因可选模块推进全局随机状态而改变共享参数初值。修复后，每个叶模块按完整模块路径使用独立初始化子流，构造过程恢复调用方随机状态；机械测试逐值比较四变体中同名、同结构模块的共同参数。
- 两条审查线共同发现旧 manifest 更新会先覆盖固定名 JSONL，再核对旧入口。修复后先核对既有固定契约，两份 JSONL 使用不可变的唯一生成文件名，最后只原子替换 `manifest.json` 作为发布点；来源不一致测试证明旧入口引用文件不变。
- 科学审查发现阶段 E 评估器没有失败关闭真实受体、CryoAtom2、Stage1-only 预览和 Emap2lig 的 179-PDB 集合。修复后两份 manifest 各自要求 179 个唯一 PDB，并与两种排序的三套预览来源逐集合相等；`test_1` 必须是 test_0 的 149-PDB 严格子集，五个空候选 PDB 仍作为未命中保留。
- 测试结果不再写 validation 专用 `selection`。正式训练与测试代码也不再计算或持久化 SHA-256；运行目录改为保存 `resolved_config.json` 和 `manifest.snapshot.json`，checkpoint 保存解析配置与 manifest 快照路径。一次性 SHA 证据仍只保留在本执行记录。
- Dataset/Batch、坐标变换、损失、梯度累积、指标和冻结 JSON 的注释已按真实字段、形状、掩码、索引和分母补全；正式代码注释与 Docstring 已统一使用技能要求的英文标点，结构化字段保持一项一行。

### 2026-09-18：阶段 E 启动

- Find_1 训练第 6 片已经完成并通过正式验收；完整 train 根现含互不重复的 1,650 个 PDB。阶段 E 禁止继续使用不含第 6 片的旧训练 manifest，新入口固定为 `/storage/penghongen/Matcher/manifests/final_train_validation_shards01_06_v1/manifest.json`。
- 用户确认 selection 的唯一有效改善量为 `0.001`。每个 epoch 的优化器更新确定性地划为三个连续区段，同一 DataLoader 只遍历一次；每段后执行完整 validation。连续第三次未改善触发平台动作，前两次平台把学习率乘以 `0.8`，第三次平台直接早停；最多 15 个 epoch、45 次完整 validation。
- 每次完整 validation 保存独立 checkpoint 并更新 LAST；只有达到 `0.001` 改善量才更新 BEST。训练结束后复用 BEST 检查时保存的预测拟合一次前景阈值，不额外执行完整 validation。
- 当前实现从干净的 `Learn/CUMULATIVE@d275bc6ece36c742f3573c1112d470403d4a9ecb` 建立 `codex/final-train-control`。本地语法编译、25 项测试和差异空白检查通过；服务器 manifest、GPU 门禁和正式训练状态由 `文档/记录/正式单一SMILES训练/` 持续记录。
- 修复后只对三份原报告中的明确问题做窄口径复核，再对窄复核指出的具体行做最后一次小范围确认。科学逻辑、函数布局与 Git 边界、代码注释与 YAML 三条审查线均返回最终 `PASS`，没有继续扩大审查范围。

### 2026-09-17：阶段 D 与训练前停止点

- 活动包只保留 `matcher/data`、`datasets`、`model`、`training`、`evaluation` 和 `utils` 的单一管线；平行 `matcher/smiles_identity`、旧 calibration、六分类、历史活动 YAML 与旧 shell 均已从当前树删除，仍可由 Git 历史恢复。
- 四份正式 YAML、四个从头训练 shell、数据构建入口、冻结测试入口和各目录 README 已连接到同一精确 SMILES 契约。阶段 C 的 Stage1-only 元数据与两张预览图已冻结，未生成任何虚构 Matcher 数值。
- 阶段 E 的冻结测试从训练 run 的 `run.json` 解析并加载原训练 release 中的模型配置、模型类与 checkpoint 规则；当前评估 release 只提供冻结清单读取、扩展 top-k 汇总和绘图，因此后续覆盖同名变体定义不会改变历史 checkpoint 的解释。
- 三条独立审查均已 PASS，本机完整回归为 23 passed。尚未执行的 GPU 门禁、四项正式训练、冻结 Matcher 测试、Stage1+Matcher 曲线、第一张七栏图和三张最终图全部属于阶段 E。
- 阶段 E 启动还必须等待用户对本轮代码的修改意见以及独立任务完成第 6 片推理的通知；未完成分片没有写入任何正式训练 manifest。

### 2026-09-17：双线 Git 收口

- 实现分支 `codex/matcher-single-pipeline-refactor` 保存真实实施、审查修复和治理回填过程。
- 学习历史从共同基点 `75ac5eaa5f4ec682724b72110605c74ec9b846a2` 开始，按删除旧主线、建立数据契约、连接 Dataset/Batch、定义四变体、实现训练评估、连接文档配置、锁定测试的顺序重建。
- 实现端点与学习端点通过完整树比较，结果无差异；学习端点重新执行完整测试，结果仍为 23 passed。
- `Learn/CUMULATIVE` 已推进到学习端点，隔离学习工作树和临时学习分支已删除。最终只保留 `C:/Users/15919/Desktop/Matcher` 一个工作树，用户未暂存的 `talk_new/反馈/re-9-16.md` 在分支切换前后保持同一字节哈希。
