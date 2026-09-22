# `test_smallmolecule` 服务器门禁

本目录保存需要随评测代码进入服务器 release 的一次性机械门禁。门禁只证明正式输入、训练快照和一个真实候选前向满足运行前提，不生成 `evaluation.json`、`docking_handoff_candidates.jsonl` 或正式图片，也不是正式评测入口。

## 两项门禁

### `gate_manifest.py`

该入口读取 `test_smallmolecule.yaml` 指向的已发布 `manifest.json`，完成两项检查：

1. 遍历 `manifest.files`，重新计算每个派生文件的 SHA-256，并要求与 `manifest.file_sha256` 的同名键一致。
2. 在 `real_candidates.jsonl` 中要求 `(pdb_id="9rd8", source_blob_index=0)` 恰好出现一次，且 `target_scope="non_stage3_foreground"`。

通过后只向标准输出打印 manifest 路径、manifest SHA-256、冻结 `summary` 和 `9rd8` 特例状态；不写文件。

### `gate_model.py`

该入口要求实际 GPU 名称包含 `A800`，并按配置中的稳定实验标识完成下列检查：

1. 从冻结 BEST 所属的训练 source release 恢复模型类、模型配置和 checkpoint。
2. 要求 checkpoint 内 `model_config.variant` 与 `test_smallmolecule.yaml.models.<experiment_id>.variant` 精确一致。
3. 选择真实受体 `real_candidates.jsonl` 中第一个 `target_scope="stage3_foreground"` 的候选，以 `num_workers=0` 执行一次真实 CUDA 身份前向。
4. 要求 `identity_logits` 与 `identity_smiles` 长度相同，并要求预测记录不含 `foreground_probability` 或 `foreground_accepted`。

通过后只向标准输出打印稳定实验标识、实际 GPU、训练快照来源、PDB、身份数和两档预测 SMILES；不写正式评测目录。

## 调用命令

调用命令只写入执行记录的“测试与门禁命令”部分：

```bash
python -m ops.small_molecule_test.gates.gate_manifest \
  --config ops/small_molecule_test/test_smallmolecule.yaml

python -m ops.small_molecule_test.gates.gate_model \
  --config ops/small_molecule_test/test_smallmolecule.yaml \
  --model density_blob_only
```

第二条命令的 `--model` 接收配置 `models` 下的稳定实验标识，如 `strongest-1`，不是模型结构变体名 `strongest`。门禁使用该记录的 `variant` 核对 checkpoint 结构。

门禁命令只进入执行记录的“测试与门禁命令”部分。正式命令只调用 [`../../../训练与运行/sh/small_molecule_test/`](../../../训练与运行/sh/small_molecule_test/) 中的短 shell；两类命令不得混用。
