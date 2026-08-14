# Handoff: Matcher 联合身份 v4 本地候选与真实等价门禁

Date: 2026-08-15

## Current State

联合身份 v4 已从共同基点 `Learn/CUMULATIVE@a25f2e71adb35c0edf6893142dc3311077bfe0ca` 完成双线 Git 收口。实现分支 `codex/matcher-joint-identity-v4` 保留真实实现提交；学习分支 `Learn/matcher-joint-identity-v4` 按文档契约、schema、清单构造、Dataset 和测试的顺序重建，两端最终树逐文件相同，`Learn/CUMULATIVE` 已快进到学习端点。Git 忽略目录中另有一次性 train/validation 清单比较工具 `tmp/joint_identity_v4_audit/audit_manifest_equivalence.py`。

第 5 批六项继续冻结 `closed_set_v3` 并占用六张受管 GPU。本 handoff 没有刷新服务器状态；最近留证仍是 2026-08-15 03:28 的六个 Slurm allocation `RUNNING`、六个 W&B run `running`、六个 `after_lock` 保留且无 `pre_lock/kill_lock`。不得为 v4 开发中断、释放或热替换这些任务。

calibration 模型推理和正式指标仍需用户在全部迭代结束后单独确认。v4 开发与审计只允许读取 train/validation；没有 heartbeat。

## Completed

- 清单 schema 从 3 提升到 4，正式身份键顺序为 `object_key,prepared_smiles`。
- `PreparedLigand.smiles` 逐字取自 `language_root/prepared/<pdb>/prepared_smiles.jsonl` 顶层字段；不使用 `model_smiles` 或额外规范化。
- 同模板不同字符串拆槽，同字符串不同模板不合并；LigandObject 图继续按模板键复用。
- 显式 `null`、整个 PreparedLigand 文件缺失、已有文件缺记录采用不同边界；前两者保留候选，后者失败关闭。
- 语言归档必须与联合身份的 `prepared_smiles` 完全相同；没有拆分的组继续使用 v3 selector salt。
- seen 标记改按联合键和语言家族计算；同一联合键跨 PDB/已物化划分类别漂移时在发布前失败。
- Dataset、collate、metadata 和 batch 已携带与身份轴对齐的 `identity_prepared_smiles`；模型不读取字符串。
- 两轮独立审查发现并已修正任意非空 `PreparedLigand.kind`、持久字段隐式转换、扩展路径类型漂移，以及一次性审计器的源真值和跨类别覆盖等边界。
- 第二轮修订后正式全套为 `162 passed in 152.31s`；mypy 检查 45 个 Python 文件零问题；compileall、pyflakes、diff-check 通过。
- 一次性清单比较工具另有 8 项回归全部通过，并通过 py_compile 与 pyflakes；它不进入 Git，也不参与正式训练。
- 第二轮修订快照的科学正确性与严格技术表达两项独立复审均为 PASS，阻断项为零。
- 实现线与学习线的树摘要完全相同；学习端点另行通过 `162 passed`、mypy、compileall、pyflakes 和 diff-check。

## Decisions

- 当前 v3 Batch5 结果继续只和 v3 基线比较。v4 必须独立发布清单和 release，并先建立零门 oracle 基线。
- 固定语言 occurrence 选择不把 `prepared_smiles` 加入 selector salt；拆分只缩小合法 occurrence 集合，未拆分组必须保持旧选择。
- `smiles=null` 不创建联合身份，也不能借用同 object_key 的非空 occurrence。
- 整个 PreparedLigand 文件缺失可保留 PDB、候选、六分类和区域监督，但必须用 `missing_file` 与显式空值区分；已有文件不完整不允许静默降级。
- 同一 `(object_key,prepared_smiles)` 的类别在 train/validation 中必须唯一；类别不属于身份键的一部分。
- 一次性清单、release、launch 或 SHA-256 留证只能在 ignored `tmp`/diagnostics 中存在，不能进入正式 `matcher`、YAML 或 `resolved.json`。

## Open Questions

- 真实 train/validation 中有多少同 PDB `(class_name,object_key)` 组会按顶层字符串拆分？用户给出的全量 17 组不能直接当作本次结果。
- 是否存在显式 `null`、PreparedLigand 文件缺失、跨 split seen 变化或同联合键类别漂移？
- 未受影响 PDB 的全部 Dataset 科学张量、固定权重前向和科学评估字段能否逐值相同？
- v4 零门 oracle 应在第 5 批哪个自然空闲事件后使用哪张受管卡？真实审计和双审前不得决定或触发。

## Next Actions

1. 安全同步等价学习树并创建独立 v4 release；只构建 `closed_set_v4` 的 train/validation。
2. 运行一次性清单审计并回填受影响 PDB、身份、occurrence、候选、seen 和有效监督数量；不读取 calibration。
3. 根据清单结果补齐 Dataset、固定权重前向和科学评估精确比较。真实门禁通过后，才准备 v4 零门 oracle 配置和单任务记录。
4. 第 5 批只在完整 validation、BEST 刷新、自然早停或失败等明确事件回填；不建立 heartbeat。

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
