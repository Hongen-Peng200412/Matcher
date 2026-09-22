# Handoff: Matcher Stage3 小分子测试实现重整

Date: 2026-09-22

## Current State

Matcher 的 Stage3 小分子专用测试已经从一次性运维目录 `ops/small_molecule_test/` 收拢到正式代码目录 `test_smallmolecule/`。本轮只整理本地代码、配置、README、测试和运行入口，没有重新训练 Matcher、没有重新执行 Matcher 推理，也没有读取或改写服务器上的正式 manifest、`evaluation.json`、`docking_handoff_candidates.jsonl` 或图片。

本文件所在提交的直接父提交是 `Learn/CUMULATIVE@75cda7a79ff594ae8e5582cca90c244281c2fb7b`。该提交是本轮逐文件比较使用的“旧逻辑精确快照”：它包含重整前仍然活动的 `ops/small_molecule_test/`、`tmp/test_smallmolecule/` 和相应 shell 入口。旧的正式小分子实现最初在当前学习历史的 `b3d40175d2aae84b18aeb714f49b2c797cde7a37` 中进入仓库；后续第二轮训练与评估提交继续扩充模型入口，但本轮等价检查始终以紧邻当前改动的 `75cda7a...` 为准。

历史冻结服务器产物继续位于：

- manifest：`/storage/penghongen/Matcher/manifests/test_smallmolecule_v1/`
- Stage1-only 与 Matcher 评估：`/storage/penghongen/Matcher/evaluation/test_smallmolecule_v1/`

这些历史产物仍保留旧代码写入的追溯字段。它们没有因本轮代码重整而改变，正在使用这些产物的 PocketXMol 任务 `codex://threads/01a0a4da-d585-7470-9c06-b6b213581831` 不需要迁移文件或重新运行 Matcher。

## Completed

### 旧实现与新实现的位置

重整前的完整实现分散在以下位置：

- `ops/small_molecule_test/`：manifest 构建、Stage1-only、Matcher 前向、排序、绘图、I/O、门禁和 YAML。
- `tmp/test_smallmolecule/`：一次性契约测试及说明。
- `matcher/evaluation/`：完整测试集与小分子测试共同使用的一部分排序、checkpoint 加载和绘图逻辑。
- `训练与运行/sh/small_molecule_test/`：正式短入口。

重整后的活动实现按照科学职责组织如下：

| 当前文件 | 当前职责 |
| --- | --- |
| `test_smallmolecule/build_manifest.py` | 连接 77 个 Stage3 测试 PDB、446 个 small-molecule occurrence、两套 Find_1 候选、Emap2lig 候选和 PDB 内受支持 SMILES 身份，写出模型无关清单。 |
| `test_smallmolecule/ranking.py` | 实现 selected 候选优先、unselected 候选追加的排序，以及 Stage3 专用 attempt 计费和 top-k 曲线。 |
| `test_smallmolecule/evaluate_stage1.py` | 从冻结 manifest 生成真实受体 Find_1、CryoAtom2 Find_1 和 Emap2lig 的 Stage1-only 曲线与逐候选轨迹。 |
| `test_smallmolecule/evaluate_model.py` | 从训练 release 快照恢复冻结 BEST，执行一次身份前向，生成两档答案范围的预测、曲线和 PocketXMol 交接记录。 |
| `test_smallmolecule/plot_results.py` | 只读取冻结 JSON，绘制 Gaussian 与 `source_probability_mean` 两种排序的图片。 |
| `test_smallmolecule/artifacts.py` | 读取 YAML、JSON、JSONL，并写出 JSON 与 JSONL 科学产物；不再承担追溯摘要或运行核验。 |
| `test_smallmolecule/test_smallmolecule.yaml` | 集中保存只读来源、产物根、前向参数和各稳定实验标识的冻结 checkpoint 输入。 |
| `test_smallmolecule/README.md` | 解释处理顺序、命中规则、正式产物树和 `stage1_only.json`、`evaluation.json`、`docking_handoff_candidates.jsonl` 的逐字段契约。 |
| `test_smallmolecule/MANIFEST_README.md` | 解释 manifest 目录六种科学文件、逐字段类型、单位、空值和跨文件编号关系。 |

共享能力保留在正式 Matcher 包：

- `matcher/evaluation/test_set.py` 公开 `selected_priority_order()`，完整 179-PDB 测试和 77-PDB 小分子测试共用同一 selected 优先顺序。
- `matcher/evaluation/checkpoint.py` 只负责从训练 release 快照恢复模型类、配置和 checkpoint，不再把 release、manifest 或文件摘要写进科学结果。
- `matcher/evaluation/plotting.py` 只读取已经冻结的科学字段绘图，不依赖已经删除的追溯字段。
- `matcher/tests/test_small_molecule_evaluation.py` 固定唯一目标、attempt 计费、selected 前缀和 Stage1-only 科学产物边界。
- `matcher/tests/test_new_pipeline.py` 随公共排序函数改名更新导入，不改变测试口径。

正式入口继续位于 `训练与运行/sh/small_molecule_test/`。所有脚本现在调用 `test_smallmolecule.build_manifest`、`test_smallmolecule.evaluate_stage1`、`test_smallmolecule.evaluate_model` 或 `test_smallmolecule.plot_results`；脚本与稳定实验标识的对应关系见同目录 `README.md`。

重整后退出的路径是：

- `ops/small_molecule_test/` 整个旧实现及其中的 `gates/`。
- `tmp/test_smallmolecule/` 一次性契约测试。

Git 历史仍可从 `75cda7a...` 恢复这些文件；它们不再是当前运行入口。

### 保留的科学契约

本轮没有改变以下计算：

1. 候选和 occurrence 的覆盖率都必须不低于 `0.30`。
2. 同一候选覆盖多个 occurrence 时，先最大化两项覆盖率的几何平均；仍并列时选择较小的 `occurrence_id`。
3. 每个 PDB 先排列 `candidate_selected=true` 候选，再追加 `candidate_selected=false` 候选；每一段内部按指定分数降序、`source_blob_index` 升序排列。
4. `background` 消耗一次 attempt，`non_stage3_foreground` 免费跳过，`stage3_foreground` 消耗一次 attempt 并进入命中判断。
5. Gaussian 曲线使用 `gaussian_score`；概率均值曲线使用 `source_probability_mean`。Matcher 不改变 Stage1 候选顺序。
6. `all_ligand` 在当前 PDB 全部受支持身份中选择最大 `identity_logits`；`small_molecule` 只在原始类别为 `small_molecule` 的身份槽中选择最大分数。两档结果复用同一次完整身份前向。
7. 前景头和 validation 前景阈值不参与 Stage3 小分子测试。
8. `blob_center_world_xyz_A` 仍由完整来源 blob 的全局 ZYX 体素中心转换为世界 XYZ 坐标，单位为 Å；它不是 centered 框中心。
9. top-k 固定计算 `k=1,...,20`，分母固定为 Stage3 测试集的 77 个 PDB。

### 删除的非科学内容

新产物不再写入以下内容：

- release、launch、checkpoint、manifest 或来源文件的 SHA-256；
- 绝对追溯路径、正式命令、服务器 Job、创建时间和 Python 运行时间；
- 手写的 schema/status/selection_scope 等追溯标签；
- `source_snapshot.json`、候选顺序摘要和一次性 gate 报告；
- 仅用于重复证明输入形状、键集合或已知前缀性质的大量失败关闭校验。

这些信息应由实验的“追溯与说明”和执行日志保存。成功完成一次运行时，新旧实现的科学数值相同；完整 JSON 不会逐字节相同，因为上述追溯字段被有意删除，而且新 I/O 不再建立临时发布目录。未来读者不能用“完整文件哈希相同”描述此次等价关系。

### PocketXMol 交接兼容性

旧 `docking_handoff_candidates.jsonl` 每条记录包含十个科学字段和五个追溯字段。新实现逐项保留 PocketXMol 实际需要的十个科学字段：

- `pdb_id`
- `receptor_condition`
- `sorting`
- `answer_scope`
- `source_blob_index`
- `centered_box_index`
- `raw_rank`
- `attempt_index`
- `blob_center_world_xyz_A`
- `predicted_smiles`

新实现只删除 `checkpoint_path`、`checkpoint_sha256`、`manifest_path`、`manifest_sha256` 和 `selection_scope`。因此，如果以后使用新代码重新生成结果，PocketXMol 仍可仅凭十个科学字段完成候选定位、世界坐标定位和预测 SMILES 读取；追溯关系改由实验文档提供。现有服务器 JSONL 没有被回写，所以依赖旧十五字段文件的正在运行任务也不会受到影响。

## Decisions

- `test_smallmolecule/` 是正式科学实现，不再把 Stage3 小分子测试称为一次性 ops。
- `matcher/evaluation/` 只保留完整测试集与小分子测试真正共用的排序、模型恢复和绘图能力。
- 科学结果文件只保存计算结果及解释这些结果必需的字段；运行来源与留证信息只写文档。
- 删除冗余校验不代表放宽科学定义。唯一目标、排序、计费、身份选择和输出字段由更小的直接实现与回归测试固定。
- 旧服务器产物继续作为历史正式结果；本轮不以新代码覆盖它们，也不要求下游重新消费。

## Equivalence Evidence

提交前在 `Learn/CUMULATIVE@75cda7a...` 的未提交重整快照上完成以下检查：

1. `D:\Anaconda\python.exe -m pytest -q`：`48 passed in 19.04s`。
2. `D:\Anaconda\python.exe -m compileall -q matcher test_smallmolecule`：通过。
3. `C:\Program Files\Git\bin\bash.exe -n 训练与运行/sh/small_molecule_test/*.sh`：通过。
4. `git diff --check`：通过；仅出现 Windows 工作区未来可能进行 LF/CRLF 转换的提示，没有空白错误。
5. 直接从 `75cda7a...` 读取旧函数，与当前函数执行随机化逐值比较：
   - selected 优先候选顺序：1,000 组，索引数组逐元素相同；
   - 双向覆盖唯一目标：1,000 组，occurrence 编号、两项覆盖率和几何平均逐值相同；
   - Gaussian 与概率均值 top-k：500 组候选集合乘两种排序，共 1,000 组；投影到 `pdb_ids`、`top_k`、`hit_count`、`hit_ratio`、`pdb_hit` 后完全相同，逐候选 trace 也完全相同。
6. 四份活动 README 的相对链接、11 个 JSON 示例和 11 组代码—字段契约覆盖检查均通过。

本轮没有重新执行真实 Matcher 前向，因此不声称重新生成了一份数值产物并与服务器文件逐字节比较。模型前向保留相同 Dataset、训练 release 恢复方式、`identity_logits`、两档 argmax 和候选键连接逻辑；冻结服务器结果本身保持不变。

## Open Questions

当前没有待决定的科学问题。若未来要在服务器重新运行本代码，应先选择新的输出目录或取得覆盖既有冻结目录的明确授权；不要把本轮本地代码提交理解为允许改写 `/storage/penghongen/Matcher/manifests/test_smallmolecule_v1/` 或 `/storage/penghongen/Matcher/evaluation/test_smallmolecule_v1/`。

## Next Actions

本轮提交后不需要训练、推理、绘图或服务器同步。未来需要重新生成 Stage3 小分子测试时，按以下顺序执行：

1. 阅读 `test_smallmolecule/README.md` 和 `test_smallmolecule/MANIFEST_README.md`。
2. 在新的 release 快照中，从仓库根目录运行 `训练与运行/sh/small_molecule_test/prepare_stage1.sh`。
3. 运行目标实验对应的 `evaluate_<variant>.sh`，再运行同名 `plot_<variant>.sh`。
4. 在实验“追溯与说明”中记录 release、launch、checkpoint、manifest、正式命令和文件摘要，不把它们重新写回科学 JSON 或 JSONL。
5. PocketXMol 只读取 `docking_handoff_candidates.jsonl` 的十个科学字段，不依赖旧的五个追溯字段。

## Files To Reopen

- `test_smallmolecule/README.md`
- `test_smallmolecule/MANIFEST_README.md`
- `test_smallmolecule/build_manifest.py`
- `test_smallmolecule/ranking.py`
- `test_smallmolecule/evaluate_stage1.py`
- `test_smallmolecule/evaluate_model.py`
- `test_smallmolecule/plot_results.py`
- `matcher/evaluation/README.md`
- `matcher/evaluation/test_set.py`
- `matcher/tests/test_small_molecule_evaluation.py`
- `训练与运行/sh/small_molecule_test/README.md`
- `文档/mapping/计划执行映射.md`
