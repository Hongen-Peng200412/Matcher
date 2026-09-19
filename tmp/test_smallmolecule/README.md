# test_smallmolecule 一次性测试与门禁

本目录不属于正式运行入口。其文件只在本机验证 `ops/small_molecule_test` 的排序状态机；
需要进入服务器 release 的 manifest 与模型门禁位于 `ops/small_molecule_test/gates/`。

## 本地机械测试

```powershell
python -m unittest tmp.test_smallmolecule.test_contract -v
python -m compileall -q ops/small_molecule_test tmp/test_smallmolecule
```

服务器门禁的具体命令见 `ops/small_molecule_test/gates/README.md`。门禁命令只进入执行
记录的“测试与门禁命令”部分，不得复制到正式 shell 脚本或正式命令栏。
