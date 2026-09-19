# Stage3 小分子子集评测入口

本目录只保存 `test_smallmolecule` 的短正式入口。科学规则、长路径与冻结计数统一位于
`ops/small_molecule_test/test_smallmolecule.yaml`；脚本不承担测试或门禁职责。

| 顺序 | 脚本 | 正式职责 |
|---:|---|---|
| 1 | `prepare_stage1.sh` | 原子发布独立 manifest，并生成 Stage1-only JSON |
| 2 | `plot_stage1.sh` | 只从 Stage1-only JSON 生成两张图 |
| 3 | `evaluate_density_blob_only.sh` | 用训练快照解释冻结 BEST，生成两档身份结果 |
| 4 | `plot_density_blob_only.sh` | 只从冻结 JSON 绘图 |
| 5 | `evaluate_strongest.sh` | 用 strongest 训练快照解释冻结 BEST，生成两档身份结果 |
| 6 | `plot_strongest.sh` | 只从冻结 JSON 绘图 |

每个脚本文件头记录独立提交命令。复用现有 A800 allocation 时，控制器把对应脚本作为
极短 `run_cmd` 写入新的 launch；SHA-256、锁路径和一次性门禁命令只进入执行记录，不进入
这些正式脚本。

正式顺序不能把绘图合并到模型前向。绘图失败不会使已原子发布的评测 JSON 失效，也不能
触发第二次模型前向。
