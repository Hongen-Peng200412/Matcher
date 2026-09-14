# Handoff: Matcher 单一精确 SMILES 身份转换

Date: 2026-09-14

## Current State

Matcher 已新增一条不读取 LigandObject、`object_key`、`model_smiles` 或 seen/unseen 的
正式活动路径。该路径以同一 PDB 内逐字相同的顶层 `PreparedLigand.smiles` 建立监督
身份槽，并以同一精确字符串读取全局 SMILES 重原子图。旧代码和旧产物仍保留，只服务
历史复现，不进入新路径。

Job 378693 的第 20 次执行正在 gnode10 的 NVIDIA A800-SXM4-80GB GPU 0 上运行 Find_1
validation 同集驻留训练。最近一次核验时 W&B 已记录到第 82 次参数更新，GPU 占用约
19.6 GiB、利用率 100%，模型进程已连续存活约 20 分钟。精确 `try_lock` 不存在，
`after_lock` 存在，`pre_lock` 与 `kill_lock` 不存在；不得删除 `after_lock`，等待用户以后
用 `kill_lock` 终止。

双线 Git 的核心实现端点 `c4de3cec2acbf88664abc0f1f35baf192e582619` 与学习端点
`64696be` 共享树对象 `55a2d50138f114568a3b91887b1db7cae9c8eb00`。本 handoff 作为
最终治理记录加入两条线后，`Learn/CUMULATIVE` 指向学习端点；实现分支继续保留真实
开发过程，工作区只保留 `C:/Users/15919/Desktop/Matcher` 一个 worktree。

## Completed

- 新包 `matcher/smiles_identity/` 覆盖图资产、Find_1 标签、单一 SMILES 清单、Dataset、
  Batch、模型、损失、完整 validation、驻留训练和在线 W&B 记录。
- 全局 `smiles_graphs_v1.npz` 与 `smiles_symmetries_v1.npz` 以精确 SMILES 为唯一逻辑键；
  坐标不混入图包，也没有新增 `graph_id`。
- 全部 PreparedLigand 输入产生 3,588 个唯一非空精确 SMILES；3,497 个通过 RDKit
  2026.03.3 严格门禁，91 个统一排除。
- Find_1 validation 发布 200 个 PDB、7,403 个候选、3,619 个前景候选、603 个同 PDB
  身份槽和 208 个跨 PDB 唯一 SMILES；10 个缺少合格图或身份级 SMI-TED 表示的前景
  候选整体跳过，背景候选保留。
- Stage3 validation 的 118 个 PDB、781 个 occurrence 和 125 个
  `(精确 SMILES, object_key)` 对全部通过图包字段核验和忽略原子编号的化学图同构检查，
  失败数为 0。
- I/O 对照确认全局图包每进程只执行一次 `numpy.load`，预热后的样本读取不重复打开图包；
  8 GiB/worker 缓存方案取得本轮最高预热吞吐，正式驻留配置采用该方案。
- 历史最佳联合身份 v4 checkpoint 的 128 候选参考前向中，36 个身份监督候选的 top-1
  改变数为 0；该结果只作非阻断参考，不把旧 checkpoint 兼容带入新模型。
- 输入快照门禁把当前 YAML、manifest、图包、两个清单 JSONL 和 validation 的逐 PDB
  输入共 2,208 份文件绑定到 Job、release、launch、选择口径和 worker 来源；两次快照
  都得到聚合 SHA-256
  `804de80be084766f2c8b71d0fa6cc301d5b1ab182d02c880661c623924c035f9`。
- 驻留训练的初始完整 validation 得到总损失 3.231579231601014、六分类准确率
  0.3121707415912468、身份 top-1 0.25227963525835867；随后参数更新持续进行，没有
  NaN、OOM、FloatingPointError 或回溯，也没有保存 checkpoint。
- 本机完整回归为 236 passed；最终机械空白清理后，专项回归为 12 passed，compileall、
  两个正式 shell 入口语法和基线到端点的 `git diff --check` 均通过。代码布局、中文注释、
  科学逻辑和实际输入快照均取得相互独立的 PASS。

## Decisions

- 身份源只能是 PreparedLigand 顶层 canonical non-isomeric SMILES；同一 PDB 内精确字符串
  相同才共享身份，不跨 PDB 合并监督槽。
- SMILES 图只保留原子序数、形式电荷、3/4/5/6 元环标志、化学键端点、键型和化学键
  3/4/5/6 元环标志。参考坐标、参考键长、`residue_id`、模板 `is_present` 和模板手性
  不进入新模型。
- 严格 RDKit 解析、去氢或 sanitize 失败时统一排除；合法的断开组分保留，不增加异常
  回退、手工修补表或 LigandObject 备用路径。
- 自同构只供未来 PocketXMol 的坐标重分配与 RMSD 使用，不改变 Matcher 身份，也不要求
  Matcher 选择某个自同构。
- 语言向量属于同一 PDB 的精确 SMILES 身份；同组 occurrence 固定选择编号最小的有效
  SMI-TED 文件并核对其余可用向量，整组无有效向量时不发布该身份。
- 正式运行命令只有 `bash 训练与运行/sh/build_smiles_identity.sh` 和
  `bash 训练与运行/sh/train_smiles_identity.sh`。SHA-256、release/launch 取证和锁控制
  只保存在 ignored `tmp/ops`、服务器 Feedback 运维目录和执行记录中。
- calibration 在本轮始终禁止读取。驻留任务使用 validation 训练并在 validation 自验证，
  不把其指标解释成独立泛化性能。

## Next Actions

- Matcher 本轮代码、科学门禁和 Git 收口完成后不再追加功能或实验。
- 用户准备结束驻留任务时，在既有控制器边界内使用 Job 378693 的 `kill_lock`；不要删除
  `after_lock`，不要用 `scancel` 代替锁协议。
- 未来负责 PocketXMol 的代理可按本目录的观察与建议文档，把运行时图入口和坐标排列从
  LigandObject 改为同一精确 SMILES 与 RDKit 重原子顺序；本轮没有修改 PocketXMol。

## Files To Reopen

- `RE-talk/关于ligand_object—SMILES/信息.md`
- `RE-talk/关于ligand_object—SMILES/Matcher实施计划.md`
- `RE-talk/关于ligand_object—SMILES/执行记录.md`
- `matcher/smiles_identity/README.md`
- `configs/experiments/smiles_identity/smiles_identity_find1_validation_resident_a800.yaml`
- `文档/mapping/计划执行映射.md`
- `文档/记录/目标草案.md`
