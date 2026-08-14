# Handoff: Matcher 第 3 批增强注意力开发

Date: 2026-08-14

## Current State

活动 Goal 仍是 Matcher 第 1–5 批开发与六卡调度。当前分支为 `codex/matcher-closed-set-rewrite`；第 3 批实现端点是 `495e7de54dca88ab3464fe0a667dcce5a540ef72`，已经安全同步并预冻结为 release `Matcher_d15132948a18`。

六个受管 allocation `340893/340894/340895/340897/340898/340899` 都在运行。Job `340893` 运行第 1 批 joint；Job `340897/340898/340899` 运行第 2 批 oracle/detached/joint；Job `340894/340895` 已按顺序接续第 2 批 mean/max 与未增强候选条件注意力候选预训练。不得删除任何 `after_lock`、释放 allocation 或触碰用户另行提交的任务。第 2 批五个非重复任务已经全部提交，后续自然空闲 GPU 才能进入第 3 批。

calibration 可以有不参与判断的准备产物，但全部批次结束且用户明确确认前，不得运行 calibration 模型推理、读取其指标或形成选择结论。

## Completed

- 第 2 批三个完整 Matcher 已使用 release `Matcher_3cc75993d69e` 启动并完成首步；W&B run 为 oracle `6soiol5e`、detached `i0j1i1hc`、joint `s3lj6298`。
- 第 2 批两个候选预训练也已使用同一 release 启动并完成首步；mean/max 的 Job/W&B 为 `340894/79immary`，未增强候选条件注意力为 `340895/12oeeshn`。
- 2026-08-14 09:57 的最近完整 validation 快照：第 1 批 joint 在 epoch 9 刷新 selection score `0.6928435444`；第 2 批 oracle/detached/joint 在 epoch 3 分别为 `0.6624755578/0.6739567888/0.6543584471`，均仍改善；第 2 批 mean/max 与未增强交叉注意力候选任务在 epoch 0 分别为 `0.5234197383/0.4935270492`。六项都继续运行，不能按首轮或单指标提前判停。
- 第 3 批一次性结构审计完成：四模态偏置输入固定为每个实体的归一化相对 XYZ 与 Stage1 固化概率，不移植 Matcher 不具备的 selector 树谱系字段。
- `CandidateTokenAttentionBias` 已实现为每模态独立 `Linear(4,D_m) → SiLU → Linear(D_m,K)`；末层权重和偏置采用零初始化，偏置加在 softmax 前。
- 新分支构造隔离 CPU 随机流，未增强/增强模型的共同参数初值逐值一致；PP 原有零输出投影未改变。
- `MatcherModelConfig` 新增正式必填开关 `use_candidate_token_attention_bias`。第 0–2 批 YAML 显式为 false，第 3 批为 true；历史恢复入口把缺字段迁移为 false。
- W&B 在 `gradient/train_step/*postclip_norm` 记录全局梯度裁剪后的偏置末层权重梯度范数，在 `attention_bias/train_step/*poststep_norm` 记录同次更新后的末层权重范数；整批无有效 token 的模态省略两项，末层 `bias` 不记录。
- 四份第 3 批 YAML、四份单任务记录与批次记录已经建立。首个冻结候选为 pytest 86 项通过，科学终审为 PASS，严格技术表达终审发现五组阻断。集中修订后的最终快照为 pytest 93 项通过、mypy 38 个源文件零问题、compileall、pyflakes、19 份 YAML 正式解析、49 个 shell 的 `bash -n` 和 diff-check 通过；科学正确性与严格技术表达两项独立复审最终均为 PASS。
- 第 3 批实现与账本已提交为 `495e7de54dca88ab3464fe0a667dcce5a540ef72`。安全同步后，服务器项目根与 release `Matcher_d15132948a18` 的热点文件逐项 SHA-256 均与本地一致，四份配置在正式 Conda 环境解析通过；release 内容摘要为 `d15132948a1885bae9a2019ac55175503c5c15459f043145147fe4a6d3598f89`。

## Decisions

- A/P 偏置相对 XYZ 来源于完整 `80³` token，但原点是候选实际 `48³` 中心，按 `40 × voxel_size` 归一化；V/PP 来源于候选级 `48³`，按 `24 × voxel_size` 归一化。偏置直接读取已经冻结在原始 token 末尾的数值。
- 四模态必须各有独立参数。最后一层权重与偏置采用零初始化，训练后不承诺继续为零；不增加门控、Fourier 特征、LayerNorm、dropout 或额外概率先验。
- 当相应模态含有效 token 且损失沿该分支回传时，A/P/V 偏置末层权重应在第一次反向传播获得非零梯度。PP 第一次先让零输出投影权重获得非零梯度，更新后下一次反向才让偏置末层权重获得非零梯度。末层 `bias` 只产生 softmax 常数平移，数学上不可辨识，不作为学习路径或监控对象。
- 第 3 批完整 Matcher 不增加未增强交叉注意力对照，按用户预算用第 2 批 mean/max 作基线，并结合第 2/3 批六分类直接注意力对照综合判断。
- 第 3 批使用 W&B project `AdaLigand-Matcher-Batch3`；完整 Matcher 与候选预训练使用两个独立 group。

## Open Questions

- 第 4/5 批任务矩阵等待前三批 validation 结果。
- 第 3 批已达到发布条件；尚未出现本批 Job、launch 或 W&B run，需等待受管 GPU 自然空闲后再按优先级绑定。

## Next Actions

1. 在自然空闲 GPU 出现时，以 release `Matcher_d15132948a18` 按第 3 批 oracle/detached/joint/增强六分类的既定优先顺序接续；每次先核对 allocation、锁和目标目录。
2. 继续低频监控第 1 批 joint 与第 2 批五项；只在完整 validation、失败或自然收口时更新账本，不中断任务。
3. 实际接续后把完整动态命令、Job、release、launch、W&B run 和首步验收回填到单任务、批次、总记录与 ExecPlan。
4. 不得读取 calibration 做模型推理或指标计算；最终一次评估仍需用户另行明确确认。

## Files To Reopen

- `文档/记录/目标草案.md`
- `matcher/model/config.py`
- `matcher/model/fusion.py`
- `matcher/model/candidate_encoder.py`
- `matcher/training.py`
- `matcher/tracking.py`
- `matcher/tests/test_model_and_loss.py`
- `matcher/tests/test_metrics_and_config.py`
- `configs/experiments/batch3/README.md`
- `configs/experiments/batch3/*.yaml`
- `文档/记录/总记录.md`
- `文档/记录/批次记录/第3批次/第3批增强候选条件注意力.md`
- `文档/exec_plan/Matcher_单候选闭集身份分类重写执行记录.md`
