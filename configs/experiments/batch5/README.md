# 第 5 批证据驱动优化配置

本目录只保存已经具备 validation 证据、直接对照和独立记录的第 5 批任务。首项任务不同时改变聚合、视野、监督、优化器或初始化来源；它只把完整 Matcher 的图分支和清单分支身份分数增量初始门从 `0.0/0.0` 改为 `0.02/0.02`。

选择 `0.02` 的依据是第 1–3 批从零训练中两个门自然学习到约 `0.01–0.03`，而不是把用户早期提出的 `0.1` 直接作为默认。首项使用第 2 批 V 质心 mean/max oracle 作为逐字段基线，保持 SMI-TED Light 289M、五类非背景身份监督、global batch 32、学习率、梯度裁剪和全部损失权重不变。第 3 批增强注意力六分类与已收口 joint 均未显示稳定优势，因此本任务不混入增强注意力。

W&B online project 固定为 `AdaLigand-Matcher-Batch5`，首个 group 为 `matcher-batch5-gate-initialization`。validation 继续记录既有 selection score、六分类、身份和区域指标，并额外读取 `identity_contribution/validation/*` 诊断门控增量相对语言基线的排序尺度；该诊断不参与损失或选模。

当前任务：

- `batch5_smi_ted_v_centroid_gate_002_oracle_a100.yaml`：从零训练 oracle，图门与清单门都从 `0.02` 开始；直接对照第 2 批 oracle。

新增任务不得仅为填卡而复制。每项必须在 `文档/记录/单任务记录/第五批次/` 建立独立记录，并在本目录 README、批次记录和总记录中写明动机、直接对照与预期判断方式。全部开发与选择只使用 validation；正式 calibration 推理仍等待用户最终确认。
