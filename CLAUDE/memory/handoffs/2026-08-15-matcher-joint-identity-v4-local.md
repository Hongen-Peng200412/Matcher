# Handoff: Matcher 联合身份 v4 本地候选与真实等价门禁

Date: 2026-08-15

## Current State

联合身份 v4 初版已在 `codex/matcher-joint-identity-v4` 与 `Learn/matcher-joint-identity-v4` 完成双线 Git。Job `342285` 暴露跨 PDB 类别边界后，PDB 本地规则修订与零门配置又从 `Learn/CUMULATIVE@f3aa704453dca61a2d96d6cf3cf061618a59e28b` 收口：实现端点为 `codex/matcher-v4-zero-gate-baseline@18fe8cb7f2e24cb1552f307517dfa6fb93a88e27`，学习端点为 `Learn/matcher-v4-zero-gate-baseline@011b289dd26422a98573508bafe9fc20520a9bc5`，两端树摘要均为 `ba7d910b617a3cc87b8ed7ae1531e886a1d29403`。`Learn/CUMULATIVE` 已快进到该学习端点。Git 忽略目录中另有一次性 train/validation 清单比较工具 `tmp/joint_identity_v4_audit/audit_manifest_equivalence.py`。

第 5 批继续冻结 `closed_set_v3`。截至 2026-08-15 08:29，双门 `0.02` 与 `clip5` 已自然早停，另外四项继续训练；第七项学习率 `2e-4` 已在分支 `codex/matcher-batch5-lr-2e4`、隔离工作树 `C:/Users/15919/Desktop/Matcher_batch5_lr_2e4` 完成配置和本地机械门禁。该工作树基于 v4 累计端点，直接发布会读取 `closed_set_v4`，独立科学复审因此判为 FAIL；它只能保留配置候选，不能直接接续空闲 A100 Job `340899`。Job `340894/340899` 的共享 `try_lock` 均已自然存在，六个 Slurm allocation 均为 `RUNNING`，六个 `after_lock` 全部保留。不得为 v4 开发中断、释放或热替换这些任务。

calibration 模型推理和正式指标仍需用户在全部迭代结束后单独确认。v4 开发与审计只允许读取 train/validation；没有 heartbeat。

截至 2026-08-15 09:57，v3→v4 Dataset/Batch CPU 门禁已经完成。首次 Job `342324` 因每类恰好四个未受影响 PDB 的过严抽样规则在 Dataset 实例化前失败；修订后 Job `342325` 以 `COMPLETED/0:0` 收口，train/validation 分别比较 92/72 个候选和 12/9 个 collated batch，报告确认科学输入、监督与 batch 张量精确一致。当前下一步是冻结并双审同一 A800 的固定权重前向与完整 validation 门禁；Job `340894` 继续由 `try_lock` 保护。

## Completed

- 清单 schema 从 3 提升到 4，正式身份键顺序为 `object_key,prepared_smiles`。
- `PreparedLigand.smiles` 逐字取自 `language_root/prepared/<pdb>/prepared_smiles.jsonl` 顶层字段；不使用 `model_smiles` 或额外规范化。
- 同模板不同字符串拆槽，同字符串不同模板不合并；LigandObject 图继续按模板键复用。
- 显式 `null`、整个 PreparedLigand 文件缺失、已有文件缺记录采用不同边界；前两者保留候选，后者失败关闭。
- 语言归档必须与联合身份的 `prepared_smiles` 完全相同；没有拆分的组继续使用 v3 selector salt。
- seen 标记按联合键和语言家族计算；同一 PDB 的本地身份槽类别冲突或候选记录跨划分重叠时发布前失败，整 PDB 排除记录不计入重叠；不同 PDB 的类别差异只作结构化诊断。
- Dataset、collate、metadata 和 batch 已携带与身份轴对齐的 `identity_prepared_smiles`；模型不读取字符串。
- 两轮独立审查发现并已修正任意非空 `PreparedLigand.kind`、持久字段隐式转换、扩展路径类型漂移，以及一次性审计器的源真值和跨类别覆盖等边界。
- 第二轮修订后正式全套为 `162 passed in 152.31s`；mypy 检查 45 个 Python 文件零问题；compileall、pyflakes、diff-check 通过。
- 一次性清单比较工具另有 8 项回归全部通过，并通过 py_compile 与 pyflakes；它不进入 Git，也不参与正式训练。
- 第二轮修订快照的科学正确性与严格技术表达两项独立复审均为 PASS，阻断项为零。
- 实现线与学习线的树摘要完全相同；学习端点另行通过 `162 passed`、mypy、compileall、pyflakes 和 diff-check。
- 首次真实 CPU Job `342285` 以 `FAILED/1` 结束且未发布清单；它发现 `CCD:SAH` 在 `train/7fgg` 为 `peptide_like`、在 `train/7pnx` 与 `validation/7pnz` 为 `small_molecule`，但各 PDB 内部一致。
- 本次规则修订把类别唯一性限定到同一 PDB 的身份轴，新增候选记录跨划分重叠门禁并明确整 PDB 排除记录不计入重叠，同时记录不同 PDB 的类别、位置和 occurrence 数诊断。学习端点的版本化正式测试为 `160 passed in 155.57s`；主工作区根目录合并运行时自动发现 ignored 审计器的 12 项测试，因此为 `172 passed in 150.01s`。mypy 检查 45 个 Python 文件零问题，29 份 YAML、compileall、pyflakes、diff-check 通过；一次性审计器单独运行仍为 `12 passed`。
- 本次规则修订的科学正确性与严格技术表达两项独立终审均为 PASS，阻断项为零。
- v4 零门 oracle 配置已经准备且只在任务命名与 W&B 隔离字段上不同于第 2 批 oracle；真实清单和同卡数值门禁完成前不得触发。
- 干净学习树已安全同步并创建 release `Matcher_28e4bacd0dbc`。第二次 CPU-only 构建 Job `342303` 以 `COMPLETED/0:0` 结束并发布 v4 train/validation 清单。一次性审计 Job `342318` 完整发布报告后因真实 seen 差异按零差异门禁退出非零；输入前后快照一致。
- 审计确认两个划分的候选 `14,312/6,495`、身份 `1,477/536`、候选集合与顺序、固定 occurrence、可监督候选数和排除记录均与 v3 相同，没有同 PDB 身份拆槽。唯一变化是 `train/7pnx` 与 `validation/7pnz` 的 `CCD:SAH` seen 身份数在两种语言中各增加 1；监督数不变。
- 第 5 批双门 `0.02` 的 BEST/last epoch 为 `7/9`，selection score 为 `0.6957138818`；相对第 2 批零门 oracle 没有改善。学习率 `2e-4` 单轴探针已建立配置、专项测试与单任务记录，但当前 v4 运行时不满足其 v3 契约，尚未触发。
- 第 5 批 `clip5` 的 BEST/last epoch 为 `8/10`，selection score 为 `0.6908370762`；相对 A100 `clip1` 次级基线未取得综合改善，同一 A800 的 `clip200` 主对照仍在训练。Job `340894` 已自然恢复 `try_lock`，可在 v4 真实等价门禁通过后接续 v4 零门基线。
- Dataset 审计 Job `342324` 在 Dataset 实例化前发现 validation/nucleotide_like 低基数并以 `FAILED/1:0` 结束；旧目录保留，没有发布正式结果。修订后的 Job `342325` 以 `COMPLETED/0:0` 结束，运行 `00:03:08`，正式比较 train/validation 的 92/72 个候选和 12/9 个 collated batch。
- Job `342325` 的 `model_input_and_supervision_exact` 与 `collated_tensor_fields_exact` 均为 true，允许变化列表为空；清单审计相关 PDB 的全部候选和未受影响分层样本均已覆盖。该结果没有读取 calibration、创建缓存或触碰 GPU 锁。

## Decisions

- 当前 v3 Batch5 结果继续只和 v3 基线比较。v4 必须独立发布清单和 release，并先建立零门 oracle 基线。
- 固定语言 occurrence 选择不把 `prepared_smiles` 加入 selector salt；拆分只缩小合法 occurrence 集合，未拆分组必须保持旧选择。
- `smiles=null` 不创建联合身份，也不能借用同 object_key 的非空 occurrence。
- 整个 PreparedLigand 文件缺失可保留 PDB、候选、六分类和区域监督，但必须用 `missing_file` 与显式空值区分；已有文件不完整不允许静默降级。
- 同一 PDB 的本地身份轴中，`(object_key,prepared_smiles)` 只能对应一个类别；同一小写 `pdb_id` 的候选记录不得同时发布到多个划分，仅有整 PDB 排除记录时不算候选重叠。类别不属于身份键，不同 PDB 的差异不改变 seen，只进入诊断。
- 一次性清单、release、launch 或 SHA-256 留证只能在 ignored `tmp`/diagnostics 中存在，不能进入正式 `matcher`、YAML 或 `resolved.json`。

## Open Questions

- 同一 Batch2 oracle checkpoint 在确定性后端下，v3/v4 的固定权重前向与完整 validation 科学评估字段能否精确一致？
- validation 的 seen/unseen 子统计是否只发生清单已声明的 `7pnz/CCD:SAH` 重新分层，且总体指标与 selection score 不变？
- v4 零门 oracle 应等待同卡数值门禁；Job `340899` 已预定给更低成本的 v3 学习率接续，不在该门禁完成前抢跑 v4。

## Next Actions

1. 在 ignored `tmp/joint_identity_v4_audit/` 中准备独立 GPU 门禁：同一空闲 A800、同一 Batch2 oracle `BEST.pt`、相同样本顺序与确定性数值后端，依次运行 v3/v4 固定权重前向和完整 validation。
2. 对 GPU 门禁脚本执行本地测试、科学正确性与严格技术表达两项独立复审；两项均 PASS 后，才允许受保护复用 Job `340894`，且不得删除 `after_lock`。
3. 先取得用户对运行分支边界的确认：推荐从已验收 v3 第 5 批端点建立独立历史契约运行分支，只携带学习率 YAML、配置说明和回归，不把 v4 代码混入；确认后再处理 Job `340899`，不抢跑。
4. v4 零门配置已经准备，但同卡门禁通过前不触发训练。
5. 第 5 批只在完整 validation、BEST 刷新、自然早停或失败等明确事件回填；不建立 heartbeat。

## Files To Reopen

- `文档/记录/目标草案.md`
- `文档/记录/批次记录/联合身份v4/联合身份v4开发与等价审计.md`
- `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`
- `文档/mapping/计划执行映射.md`
- `matcher/data/schema.py`
- `matcher/data/manifest.py`
- `matcher/datasets/contracts.py`
- `matcher/datasets/candidate_data.py`
- `matcher/datasets/batching.py`
- `matcher/tests/test_data_pipeline.py`
- `matcher/tests/test_model_and_loss.py`
- `训练与运行/README.md`
- `tmp/joint_identity_v4_audit/audit_dataset_equivalence.py`（仅当本地 ignored 文件仍存在时只读；不是版本化恢复输入）
- `configs/experiments/batch5/README.md`
- `configs/experiments/batch5/batch5_smi_ted_v_centroid_lr_2e4_oracle_a100.yaml`
- `matcher/tests/test_metrics_and_config.py`
- `文档/记录/批次记录/第5批次/第5批证据驱动优化.md`
- `文档/记录/单任务记录/第五批次/SMI-TED_V质心_mean_max_oracle_学习率2e-4.md`
