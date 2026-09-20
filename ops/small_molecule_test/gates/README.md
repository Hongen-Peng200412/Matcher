# test_smallmolecule 服务器门禁

本目录保存需要进入服务器 release 的一次性门禁，不是正式运行入口。

- `gate_manifest.py`：独立核对正式 manifest 内全部派生文件的 SHA-256，并检查 `9rd8` 特例。
- `gate_model.py`：在实际 A800 上从训练 source release 加载一个冻结 BEST，并执行一个真实 Stage3 目标候选的身份前向。

调用命令只写入执行记录的“测试与门禁命令”部分：

```bash
python -m ops.small_molecule_test.gates.gate_manifest \
  --config ops/small_molecule_test/test_smallmolecule.yaml

python -m ops.small_molecule_test.gates.gate_model \
  --config ops/small_molecule_test/test_smallmolecule.yaml \
  --model density_blob_only
```

第二条命令的模型名也可为配置 `models` 下的其它稳定实验标识；门禁使用该记录的 `variant` 核对 checkpoint 结构。正式命令仍只调用
`训练与运行/sh/small_molecule_test/` 中的短 shell，不得混用。
