# 第 5 批 SMI-TED V 质心 mean/max oracle 双门 0.02

- 科学任务：五类非背景完整 Matcher；SMI-TED Light 289M；V 质心 48³；mean/max；oracle。
- 配置：`configs/experiments/batch5/batch5_smi_ted_v_centroid_gate_002_oracle_a100.yaml`。
- 唯一科学变量：`initial_graph_gate=0.02`、`initial_inventory_gate=0.02`；第 2 批直接基线均为 `0.0`。
- 直接对照：第 2 批 `batch2_smi_ted_v_centroid_oracle_a100`，BEST selection score `0.6981048482`。
- 动机：第 1–3 批门值自然学习到约 `0.01–0.03`，首项当时选择同量级 `0.02`，没有把 `0.1` 当作默认或首个探针；后续独立的双门 `0.1` 任务只检验初始化尺度敏感性。
- 判断方式：优先比较 validation selection score、六分类、五类身份、small_molecule 兼容身份和区域指标；结合 `identity_contribution/validation/*` 判断门控增量相对语言基线的尺度。单个指标波动不构成正负结论。
- 预定资源：受管 A100 Job `340899`；最多 16 CPU；物理批量 8、梯度累积 4、global batch 32；不设置训练或 Slurm 时间上限。
- W&B：online project `AdaLigand-Matcher-Batch5`；group `matcher-batch5-gate-initialization`；run [`yd20vpcv`](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch5/runs/yd20vpcv)。
- 状态：2026-08-14 23:20（Asia/Shanghai）已在受管 A100 Job `340899` 启动并完成首个 global batch；训练继续运行。

## 提交留证

- 实现提交：`5098f382076795033e3461a1685bd820a928c3f9`；等价学习端点：`ec012345153ad3d758a47dc88b4d0e81e5e5b0a8`。
- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_1e75c58542e5/Matcher`。远端门禁正式解析 23 份 YAML 的 24 个阶段、内存编译全部 Matcher Python 源码，并从 train/validation 各读取 3 个真实候选；没有读取 calibration。
- 受保护控制器：`tmp/batch5_launch/activate_batch5_gate.sh`。该 Git 忽略的一次性脚本经科学正确性和严格技术表达两项独立审查后均为 PASS；实际执行字节的 SHA-256 为 `b6bbc2ab46e4e92713d442a1aa5b63b55e9edfe954c838670857dfb1dfa8bc45`，且它只管理 Job `340899`。
- 控制端完整命令：

```powershell
& "$env:USERPROFILE\.codex\tools\Invoke-ProjectSsh.ps1" `
  -Command 'bash -s' `
  -InputFile 'C:\Users\15919\Desktop\Matcher\tmp\batch5_launch\activate_batch5_gate.sh'
```

- 动态训练命令：`exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/train_experiment.sh" --plan 'configs/experiments/batch5/batch5_smi_ted_v_centroid_gate_002_oracle_a100.yaml'`。
- Slurm/launch：Job `340899`；launch `train_experiment_job340899_20260814T232001_a4`；allocation 为 1×A100、16 CPU、`TimeLimit=UNLIMITED`。
- 运行根：`/storage/penghongen/tmp/Matcher/experiments/batch5_smi_ted_v_centroid_gate_002_oracle_a100/train_experiment_job340899_20260814T232001_a4`。
- 锁边界：控制器只删除了 `try_lock_340899`；`after_lock_340899` 继续存在，`pre_lock_340899/kill_lock_340899` 均不存在。没有取消、重启或释放 allocation。

## 验收与结果

- 首个 global batch：global step 1、global batch 32、总损失 `3.6084863407`；分类/ligand 区域/binding 区域/身份损失依次为 `1.7836914062/1.7509000003/1.6756209433/1.4821428401`。裁剪前梯度范数 `42.3180923462`，数据等待 `6.7542922919 s`，计算 `24.3319061401 s`，吞吐 `1.0293957323 candidate/s`，身份有效候选数 7；显存峰值 allocated/reserved 为 `14,195,046,400/19,667,091,456` bytes。第一次更新后的 graph/inventory 门值为 `0.0200999994/0.0200999975`。上述值均为有限数，W&B online 已完成写入。
- 完整 validation 时间线：epoch 0–3 的 selection score 依次为 `0.6388081598/0.6435844626/0.6483033074/0.6773211521`，四轮均严格改善。epoch 3 的六分类 macro-F1、五类固有 Top-1、精确子类条件 Top-1、宽松五类端到端 F1 与 small_molecule 兼容端到端 F1 依次为 `0.5293661687/0.8451568895/0.8536398467/0.6489574407/0.6090828139`；ligand/binding micro Dice 为 `0.2048187709/0.0201932732`。
- epoch 3 的 graph/inventory/合计增量相对语言基线中心化 L2 比例为 `0.2174344/0.0457968/0.2068664`，有效多身份候选与语言基线非零候选均为 `1313`。该尺度支持拆分门控因子矩阵，但不是有符号收益。
- 当前 BEST 比第 2 批最终 BEST 低约 `2.98%`，同时比第 2 批同 epoch 3 高约 `2.24%`；任务继续到自然早停，不用当前轮次提前判正负。
- 最终结果：待自然早停后回填 BEST/last、W&B、运行产物和与第 2 批 oracle 的头对头判断。
