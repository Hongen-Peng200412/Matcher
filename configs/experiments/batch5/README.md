# 第 5 批证据驱动优化配置

本目录只保存已经具备 validation 证据、直接对照和独立记录的第 5 批任务。六项均使用第 2 批 V 质心 mean/max oracle 的冻结 `closed_set_v3` 身份契约，不改变聚合、视野、语言、监督、优化器、损失、随机种子或初始化来源。联合身份 `(object_key, PreparedLigand.smiles)` 将另升清单 schema 并先建立新契约零门基线，不混入本目录的直接对照。

选择 `0.02` 的依据是第 1–3 批从零训练中两个门自然学习到约 `0.01–0.03`，而不是把用户早期提出的 `0.1` 直接作为默认。首项使用第 2 批 V 质心 mean/max oracle 作为逐字段基线，保持 SMI-TED Light 289M、五类非背景身份监督、global batch 32、学习率、梯度裁剪和全部损失权重不变。第 3 批四项均已收口且未显示增强注意力的跨任务稳定优势，因此本批不混入增强注意力。

W&B online project 固定为 `AdaLigand-Matcher-Batch5`。门初值矩阵使用 group `matcher-batch5-gate-initialization`；梯度裁剪对照使用 `matcher-batch5-gradient-clipping`。validation 继续记录既有 selection score、六分类、身份和区域指标，并额外读取 `identity_contribution/validation/*` 诊断门控增量相对语言基线的排序尺度；该诊断不参与损失或选模。

当前任务：

- `batch5_smi_ted_v_centroid_gate_002_oracle_a100.yaml`：从零训练 oracle，图门与清单门都从 `0.02` 开始；直接对照第 2 批 oracle。
- `batch5_smi_ted_v_centroid_gate_010_oracle_a100.yaml`：图门与清单门都从 `0.1` 开始，检验较高初始化尺度。
- `batch5_smi_ted_v_centroid_graph_gate_002_oracle_a100.yaml`：只把图门设为 `0.02`，清单门为零。
- `batch5_smi_ted_v_centroid_inventory_gate_002_oracle_a100.yaml`：只把清单门设为 `0.02`，图门为零。
- `batch5_smi_ted_v_centroid_clip_5_oracle_a800.yaml`：双门保持零，把裁剪阈值从 `1.0` 放宽到 `5.0`。
- `batch5_smi_ted_v_centroid_clip_200_oracle_a800.yaml`：双门保持零，以阈值 `200.0` 建立近似不裁剪对照。

第 2 批零门 `00`、本目录的 graph-only `10`、inventory-only `01` 与双门 `11` 组成 `0.02` 初始化下的完整 2×2 因子矩阵；双门 `0.1` 另与同硬件的零门和双门 `0.02` 检验初始化尺度敏感性。`clip5` 与 `clip200` 都使用 A800 物理批量 16、梯度累积 2，二者是同硬件直接对照；对第 2 批 A100 `clip1` 的比较只作次级参考。只有实际 `preclip_norm > 200` 的比例不超过 `0.1%` 时，`clip200` 才称“近似不裁剪”；最终必须同时报告总更新数、触发数、最大裁剪前范数与最小缩放系数。

新增任务不得仅为填卡而复制。每项必须在 `文档/记录/单任务记录/第五批次/` 建立独立记录，并在本目录 README、批次记录和总记录中写明动机、直接对照与预期判断方式。全部开发与选择只使用 validation；正式 calibration 推理仍等待用户最终确认。
