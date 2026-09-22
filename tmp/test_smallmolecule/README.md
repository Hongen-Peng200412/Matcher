# `test_smallmolecule` 本地契约测试

本目录不属于正式运行入口。`test_contract.py` 只在本机验证 `ops/small_molecule_test` 的纯 CPU 契约，不读取服务器 manifest、不加载 checkpoint，也不生成正式评测产物。需要进入服务器 release 的 manifest 与模型门禁位于 [`../../ops/small_molecule_test/gates/`](../../ops/small_molecule_test/gates/)。

## 覆盖的规则

| 测试 | 核对内容 |
| --- | --- |
| `test_duplicate_candidate_key_fails_closed` | 同一来源内重复的 `(pdb_id, source_blob_index)` 必须失败关闭 |
| `test_unique_target_uses_overlap_then_occurrence_id` | 双向覆盖达到 0.30 后先最大化几何平均；分数并列时选择较小 occurrence 编号 |
| `test_non_stage3_foreground_is_free_but_background_spends_attempt` | 非 Stage3 前景免费跳过，背景和 Stage3 目标消耗尝试次数；`raw_rank` 仍保留完整候选位置 |
| `test_selected_segment_always_precedes_unselected_segment` | `candidate_selected=true` 段必须整体位于 False 段之前，即使 unselected 候选分数更高 |

## 本地机械测试

```powershell
python -m unittest tmp.test_smallmolecule.test_contract -v
python -m compileall -q ops/small_molecule_test tmp/test_smallmolecule
```

这些命令只验证 Python 语法和上述状态机，不证明正式来源文件、GPU、训练 release 或 checkpoint 可用。服务器门禁的具体命令见 [`../../ops/small_molecule_test/gates/README.md`](../../ops/small_molecule_test/gates/README.md)。门禁命令只进入执行记录的“测试与门禁命令”部分，不得复制到正式 shell 脚本或正式命令栏。
