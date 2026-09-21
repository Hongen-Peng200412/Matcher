# Handoff: Matcher 第二轮训练与评估收口

Date: 2026-09-21

## Current State

第二轮 `strongest-1`、`blob_only-0`、`density_only-0`、`density_only-106` 与 `pocket_only-0` 均已完成训练、完整 held_out_test_0、同次前向筛选的 test_1，以及 77-PDB Stage3 小分子测试。正式结果位于 `测试结果与追溯/experiment-centric/`，两个评估口径总览位于 `测试结果与追溯/测试结果总览/`。

实现线端点为 `codex/second-round-training@59b7b19d133b6292fc1b5be9eaaff1fbc1595b6e`。学习线从共同基点 `a7947478200568fbd36c0fd2b91318a8d6d6a1a7` 重建为五个概念提交；加入本交接文档前，学习端点 `afc6f899fb7ebdb1a2e1a93219109156c3a293a5` 与实现端点具有相同 Git tree `7ef43214154904e68fad778d5951166d5e43c95a`。`Learn/CUMULATIVE` 已推进到本轮学习线，隔离工作树已删除，最终只保留 `C:\Users\15919\Desktop\Matcher` 主工作树。

## Completed

- `strongest-1` 正式使用 `0.30/0.00/0.00/1.00` 损失权重；旧区域损失运行作为被取代的中断证据保留。
- `density_only-106` 只使用 56 通道完整密度 bank 与 50 通道受体体素特征；没有覆盖单通道 `density_only` 定义。
- 五项实验均从各自训练 release 快照恢复模型类、配置和 checkpoint 完成冻结评估。
- 全测试集与小分子总览均已补齐 strongest-1、density_only-0、density_only-106 和 pocket_only-0。
- 正式图片直接复制自服务器评估目录，并逐项核对 SHA-256。
- 学习线执行 `45 passed`、`compileall`、活动 YAML 解析和 shell 语法检查，全部通过。

## Decisions

- BEST 公式固定为 `(0.3 × 前景 PRAUC + 受支持真实前景 SMILES 准确率) / 1.3`。
- 完整测试只使用 BEST validation 冻结阈值；不读取 calibration，不在测试集重新拟合。
- Stage3 小分子测试忽略前景头和阈值，只评价冻结候选顺序下的精确 SMILES 身份命中。
- `density_only-0` 使用 A800，`density_only-106` 使用 H100；两者只作描述性比较，不能把差异严格归因于 1→106 通道。

## Open Questions

无科学实现问题。2026-09-21 16:12 左右只读检查时，Job 378587、386151 与数组任务 379403_1 均已恢复 try_lock，pre_lock/kill_lock 不存在，节点上没有 Matcher 训练或评估残留进程；after_lock 当时已不存在，但三个 allocation 仍显示 RUNNING。此次收口没有删除、创建或恢复 after_lock，也没有使用 scancel。

## Next Actions

本轮没有遗留实现、训练或评估动作。后续若需要释放三个仍显示 RUNNING 的 allocation，应先重新只读核对锁与进程，再按用户届时给出的锁协议操作；不要把本交接文档视为删除锁或调用 `scancel` 的授权。

## Files To Reopen

- `文档/记录/第二轮训练/总日志.md`
- `文档/mapping/计划执行映射.md`
- `测试结果与追溯/测试结果总览/全测试集.md`
- `测试结果与追溯/测试结果总览/小分子stage3专用测试集.md`
- `测试结果与追溯/experiment-centric/strongest-1/追溯与说明.md`
- `测试结果与追溯/experiment-centric/density_only-106/追溯与说明.md`
