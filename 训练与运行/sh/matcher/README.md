# Matcher Anchor O/O′ 正式训练入口

本目录只承载“不依赖 Stage1 推理产物、受体侧只有 A 原子”的 Anchor O/O′ 路线。未来含 Stage1 产物或 A/B 后处理的路线必须新增独立入口，不修改本脚本的数据与推理语义。

正式入口是 `train_anchor_O_O_prime.sh`。它依次完成：

1. 使用 `configs/matcher/anchor_O_O_prime_phase1.yaml` 训练 Phase1。
2. 从 Phase1 `BEST.json` 读取精确 checkpoint，注入 `configs/matcher/anchor_O_O_prime_phase2.yaml` 后训练 Phase2。
3. 从 Phase2 `BEST.json` 读取精确 checkpoint，使用其中冻结的 O 概率阈值运行完整 validation 推理与评估。

正式输出根目录固定为：

```text
/storage/penghongen/AdaLigand/Results/matcher/anchor_O_O_prime_v1/seed_3407/
├── phase1/
├── phase2/
└── validation_O_O_prime/
```

脚本只接受全新的空输出位置，发现正式输出根目录已存在会立即停止，避免覆盖或混合实验。断点恢复由 `matcher.train` 的 `run.resume_checkpoint` 明确指定，不通过删除正式产物重新开始。

经人工审阅并明确授权后，从服务器 AdaLigand 项目根目录提交：

```bash
bash 训练与运行/submit_task.sh \
  --sh 训练与运行/sh/matcher/train_anchor_O_O_prime.sh \
  --resource a800 \
  --gpus 1 \
  --cpus 16 \
  --qos cpu96 \
  --job-name matcher_anchor_OOprime_v1
```

该命令使用完整 release/launch 留证模式，不使用 `--simple`。首轮没有独立 heldout BOX 池，最终结果只能称为 train/validation 闭环，不能称为独立 heldout 测评。

当前服务器 `nvlink` 分区已经只读核实为 `DefaultTime=NONE`、`MaxTime=UNLIMITED`，因此正式命令不填写 `--time`，不会继承未说明的短时限。训练仍由每阶段最多 20 epoch 和累计三次实际降学习率的模型侧停止条件收口。

正式 shell 显式设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，与 A800 显存画像的 CUDA 分配器环境保持一致。两份正式 YAML 当前使用 `map_channels=[24,48,72,96]`。真 `9cpk` 与更重 `9kdv` 的 reserved 峰值分别为 49.775 和 71.383 GiB；普通 Dataset 连续 708 个优化步无 OOM，但 reserved 峰值 72.910 GiB 超出 72 GiB 人工门槛 0.910 GiB。完整证据记录在 `文档/exec_plan/Matcher_Anchor_OOPrime端到端实施.md`。
