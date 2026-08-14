# Handoff: Matcher 第 4 批发布前冻结

Date: 2026-08-14

## Current State

活动 Goal 是 Matcher 第 1–5 批开发、validation 头对头比较和六张受管 GPU 调度。当前实现分支为 `codex/matcher-batch4-pretrained-full`，共同基点是 `Learn/CUMULATIVE@4db3326`。恢复时先读取 `文档/记录/目标草案.md`，再用 `git status` 和本文件核对工作树是否仍处于第 4 批发布前状态。

六个受管 allocation 仅为 Job `340893/340894/340895/340897/340898/340899`。2026-08-14 18:51 的只读状态为：六项均为 `RUNNING` 且全部保留 `after_lock`；Job `340894/340895/340897` 具有 `try_lock`，Job `340893/340898/340899` 继续第 3 批完整 Matcher。不得删除任何 `after_lock`、释放 allocation、触碰 `kill_lock` 或修改用户另行提交的任务。

calibration 清单与缓存可以准备，但在全部批次结束且用户再次明确确认前，不得执行 calibration 模型推理、读取正式指标或用于方案选择。

## Completed

- 正式 Python 包已整理为 `matcher/data`、`matcher/datasets`、`matcher/model`、`matcher/training`、`matcher/evaluation` 和 `matcher/utils`，实现端点与学习端点逐文件等价并完成服务器可变项目根清理；运行中的旧 release 未被热替换。
- 第 3 批增强六分类在 last epoch 6 自然早停，BEST epoch 4 macro-F1 为 `0.5276366341`；增强相对未增强注意力约 `+0.07%`，相对第 2 批 mean/max 约 `-0.51%`。
- 第 4 批来源选择为 Job `340894` 的第 2 批 V 质心 mean/max 候选预训练，运行 `train_experiment_job340894_20260814T093012_a7`，BEST epoch 1，macro-F1 `0.5303632887`。
- 一次性最终来源审计报告为 `/storage/penghongen/tmp/Matcher/diagnostics/batch4_source_audit_20260814/batch2_meanmax_job340894_a7_final.json`；完整命令在第 4 批批次记录中。脚本与文件摘要不进入正式包、YAML 解析值或 `resolved.json`。
- `configs/experiments/batch4/` 已建立 oracle、detached、joint 三份单阶段计划；`文档/记录/单任务记录/第四批次/` 已建立三份独立任务记录。当前 3 项已决定、0 项已提交。
- 当前快照完整 pytest `119 passed`；mypy 检查 45 个 Python 文件零问题；pyflakes、compileall、22 份正式 YAML 解析和 diff-check 通过。科学正确性与严格技术表达两项独立终审均为 PASS，阻断项为零。

## Decisions

- 三项只把对应第 2 批完整 Matcher 的 `candidate_encoder` 从随机初始化改成同一来源 BEST。模型、数据、损失、优化、SMI-TED 289M、五类非背景身份和 global batch 32 保持一致。
- oracle 使用 Job `340897` 的 A100、物理批量 8×梯度累积 4、每工作进程缓存 16 GiB；detached/joint 使用 Job `340895/340894` 的 A800、物理批量 16×梯度累积 2、每工作进程缓存 32 GiB。硬件适配不改变科学配置。
- W&B 使用 online project `AdaLigand-Matcher-Batch4` 和 group `matcher-batch4-pretrained-mean-max`；三项必须分别记录 run ID、URL、launch、release、完整控制命令和动态训练命令。
- 触发顺序固定为 oracle→detached→joint。相邻批次可以并行，因为第 3 批四个非重复任务已经全部提交；不得重复第 3 批或跳过本批未提交任务。
- 正式代码禁用来源发布哈希；发布关系和文件字节只由一次性 `tmp/batch4_source_audit/` 脚本留证。

## Open Questions

- 实现提交、学习端点、内容寻址 release、三项 launch 与 W&B run 尚未产生，不能提前填入任务记录。
- 第 3 批三个增强完整 Matcher 仍在运行；它们的最终 validation 只用于后续综合结论，不阻止已经由增强六分类收口确定的第 4 批候选来源。

## Next Actions

1. 在实现分支保存真实提交，从共同基点建立 `Learn/matcher-batch4-pretrained-full`，按文档与配置、测试的阅读顺序重建，并核验两个端点逐文件等价。
2. 快进 `Learn/CUMULATIVE` 后安全同步可变项目根；执行本地与服务器正式解析、导入和候选编码器初始化门禁，再创建内容寻址 release。
3. 每项触发前重新核对 Job、GPU、16 CPU、动态命令、`try_lock/after_lock`、目标产物目录和来源 BEST；只删除对应 `try_lock`，永不删除 `after_lock`。
4. 按 oracle→detached→joint 触发 Job `340897/340895/340894`，立即回填三份单任务记录、批次记录、总记录、ExecPlan 和新 handoff。

## Files To Reopen

- `文档/记录/目标草案.md`
- `configs/experiments/batch4/README.md`
- `configs/experiments/batch4/full_smi_ted_v_centroid_pretrained_mean_max_oracle_a100.yaml`
- `configs/experiments/batch4/full_smi_ted_v_centroid_pretrained_mean_max_detached_a800.yaml`
- `configs/experiments/batch4/full_smi_ted_v_centroid_pretrained_mean_max_joint_a800.yaml`
- `matcher/tests/test_metrics_and_config.py`
- `matcher/training/initialization.py`
- `matcher/training/engine.py`
- `文档/记录/批次记录/第4批次/第4批候选预训练初始化.md`
- `文档/记录/单任务记录/第四批次/`
- `文档/记录/总记录.md`
- `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`
- `文档/mapping/计划执行映射.md`
- `tmp/batch4_source_audit/audit_candidate_pretrain_source.py`（仅当本地仍存在时只读核对；它不是版本化恢复输入或未来来源的已验收工具）
