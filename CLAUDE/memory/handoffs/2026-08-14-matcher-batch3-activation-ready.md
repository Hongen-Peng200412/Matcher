# Handoff: Matcher 第 3 批受保护接续已就绪

Date: 2026-08-14

## Current State

活动 Goal 是 Matcher 第 1–5 批开发、validation 头对头比较与六张受管 GPU 调度。当前分支为 `codex/matcher-closed-set-rewrite`，本地 HEAD 为 `4fb6c7b832c112961a2394cb21d69742266c6bb7`；第 3 批科学实现端点是 `495e7de54dca88ab3464fe0a667dcce5a540ef72`。冻结 release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_d15132948a18/Matcher`，内容摘要为 `d15132948a1885bae9a2019ac55175503c5c15459f043145147fe4a6d3598f89`。

2026-08-14 10:44:47 的只读状态为：Job `340893/340894/340895/340897/340898/340899` 全部 `RUNNING`，每项 `after_lock=1`，`try_lock/pre_lock/kill_lock=0`。第 1 批 joint 与第 2 批五项继续运行；没有执行第 3 批控制器，没有安装新动态命令，也没有删除任何锁。

calibration 清单或缓存可以作为不参与判断的准备产物；在全部批次结束且用户再次明确确认前，不得运行 calibration 模型推理、读取其指标或用于方案选择。

## Completed

- 第 3 批实现、配置、测试、服务器门禁与 release 已完成；科学正确性和严格技术表达两项独立审查均为 PASS。
- A100 控制器为 `tmp/batch3_launch/activate_next_a100.sh`，SHA-256 `6938a27fe13a30a0121b2f790e40b195b44b9edc8b5988f5ee165b9817c80731`。
- A800 控制器为 `tmp/batch3_launch/activate_candidate_a800.sh`，SHA-256 `71eceb2ccc71c64502c72da17e8e344fcbd748777643812608c8ec892fba2bd7`。
- 两个控制器共享 `batch3_activation.lock`，覆盖全局目标选择、动态命令安装和 `try_lock` 删除；A100 维持 oracle→detached→joint，A800 只允许增强六分类一次。
- 控制器验证当前 allocation、精确资源字段、锁、冻结 release、前序不可变 launch、运行根与阶段摘要、BEST/LAST、W&B 正常收口、目标输出防重复和本批配置；错误全部失败关闭。
- 已安装但尚未触发的命令可以在完整命令逐字节一致时恢复。其它 Job 若仍有半触发命令，A100 不会越过它选择下一项。
- 生成的动态命令在训练前断言 `TASK_PROJECT_ROOT` 精确等于冻结 release；控制器只删除当前 Job 的 `try_lock`，永不删除 `after_lock`。
- 两个控制器分别通过科学部署终审与严格 Bash/技术表达终审；`bash -n`、内嵌 Python 编译、编码与换行、哈希、正常安装、冲突、空洞、重复和半触发恢复探针全部通过。

## Decisions

- 不为第 3 批预占或中断正在运行的 GPU；只有受管 allocation 自然出现 `try_lock` 时才运行对应控制器。
- A100 任意一张自然空闲时都按全局前缀选择下一项，而不是把 oracle/detached/joint 永久绑定到特定 Job。
- A800 两张卡中的任意一张自然空闲时可以承接唯一的增强六分类；另一张不得重复提交。
- 运行根 `summary.json` 是前序命令和 W&B `finish(0)` 正常收口的必要证据；只有阶段摘要不足以放行下一项。
- 若动态命令已经安装而 `try_lock` 仍在，必须在同一 Job 恢复；不得补建 `try_lock`，也不得把该目标视为已触发后越序推进。
- 本地后续文档提交不得同步到服务器，直到剩余第 3 批任务都已从 `Matcher_d15132948a18` 启动；否则可变项目根会生成新 release。动态命令的固定 release 断言会阻止误版本运行，但不应主动制造这种失败。

## Open Questions

- 哪个受管 Job 会先自然出现 `try_lock` 尚未知；不轮询式高频记录，只在明确事件发生时复核。
- 第 4/5 批任务组合仍等待前三批 validation 完整结果。
- 共享文件系统的跨进程 `flock` 语义按服务器既有行为使用，没有为了验证而操作真实控制锁；这是剩余的运行环境假设。

## Next Actions

1. 低频或事件式运行 `tmp/batch3_launch/read_managed_status.sh`；只有目标 Job 出现 `try_lock=1` 且 `after_lock=1` 时进入下一步。
2. A100 自然空闲时，执行：

   ```powershell
   & "$env:USERPROFILE\.codex\tools\Invoke-ProjectSsh.ps1" `
     -Command "bash -s -- '<JOB_ID>'" `
     -InputFile 'C:\Users\15919\Desktop\Matcher\tmp\batch3_launch\activate_next_a100.sh'
   ```

3. A800 自然空闲时，执行：

   ```powershell
   & "$env:USERPROFILE\.codex\tools\Invoke-ProjectSsh.ps1" `
     -Command "bash -s -- '<JOB_ID>'" `
     -InputFile 'C:\Users\15919\Desktop\Matcher\tmp\batch3_launch\activate_candidate_a800.sh'
   ```

4. 控制器成功后立即只读核对 launch、release、W&B run、首个 global batch 32 更新和 `after_lock`；再把完整命令、Job、launch、W&B、首步与个性化信息回填到单任务、批次、总记录和 ExecPlan。
5. 不得执行或恢复 calibration 推理，不得删除任何 `after_lock`，不得释放受管 GPU，不得触碰用户另行提交的任务。

## Files To Reopen

- `文档/记录/目标草案.md`
- `CLAUDE/memory/handoffs/2026-08-14-matcher-batch3-activation-ready.md`
- `tmp/batch3_launch/read_managed_status.sh`
- `tmp/batch3_launch/activate_next_a100.sh`
- `tmp/batch3_launch/activate_candidate_a800.sh`
- `configs/experiments/batch3/README.md`
- `configs/experiments/batch3/*.yaml`
- `文档/记录/总记录.md`
- `文档/记录/批次记录/第3批次/第3批增强候选条件注意力.md`
- `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`
- `文档/mapping/计划执行映射.md`
