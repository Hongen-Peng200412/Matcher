# Stage3 小分子子集评测入口

本目录只保存 `test_smallmolecule_v1` 的短正式入口。科学规则、长路径、冻结 checkpoint 与机械计数统一位于 [`../../../ops/small_molecule_test/test_smallmolecule.yaml`](../../../ops/small_molecule_test/test_smallmolecule.yaml)；脚本不承担测试或门禁职责。科学规则和产物字段见 [`../../../ops/small_molecule_test/README.md`](../../../ops/small_molecule_test/README.md)。

## 脚本与产物

| 顺序 | 脚本 | 正式职责 | 主要产物 |
| ---: | --- | --- | --- |
| 1 | `prepare_stage1.sh` | 原子发布独立 manifest，并生成 Stage1-only JSON | `manifests/test_smallmolecule_v1/`、`evaluation/test_smallmolecule_v1/stage1_only/stage1_only.json` |
| 2 | `plot_stage1.sh` | 只从 Stage1-only JSON 绘图 | `stage1_only/figures/` 下两张 PNG |
| 3 | `evaluate_density_blob_only.sh` | 用 `density_blob_only` 训练快照解释冻结 BEST | 模型目录下 `evaluation.json` 与 `docking_handoff_candidates.jsonl` |
| 4 | `plot_density_blob_only.sh` | 只从 `density_blob_only` 冻结 JSON 绘图 | 模型目录 `figures/` 下两张 PNG |
| 5 | `evaluate_strongest.sh` | 用 `strongest` 训练快照解释冻结 BEST | 模型目录下两个 JSON/JSONL 科学产物 |
| 6 | `plot_strongest.sh` | 只从 `strongest` 冻结 JSON 绘图 | 模型目录下两张 PNG |
| 7 | `evaluate_blob_only.sh` | 用 `blob_only-0` 训练快照解释冻结 BEST | 模型目录下两个 JSON/JSONL 科学产物 |
| 8 | `plot_blob_only.sh` | 只从 `blob_only-0` 冻结 JSON 绘图 | 模型目录下两张 PNG |
| 9 | `evaluate_strongest_1.sh` | 用 `strongest-1` 训练快照解释冻结 BEST | 模型目录下两个 JSON/JSONL 科学产物 |
| 10 | `plot_strongest_1.sh` | 只从 `strongest-1` 冻结 JSON 绘图 | 模型目录下两张 PNG |
| 11 | `evaluate_density_only.sh` | 用 `density_only-0` 训练快照解释冻结 BEST | 模型目录下两个 JSON/JSONL 科学产物 |
| 12 | `plot_density_only.sh` | 只从 `density_only-0` 冻结 JSON 绘图 | 模型目录下两张 PNG |
| 13 | `evaluate_density_only_106.sh` | 用 `density_only-106` 训练快照解释冻结 BEST | 模型目录下两个 JSON/JSONL 科学产物 |
| 14 | `plot_density_only_106.sh` | 只从 `density_only-106` 冻结 JSON 绘图 | 模型目录下两张 PNG |
| 15 | `evaluate_pocket_only.sh` | 用 `pocket_only-0` 训练快照解释冻结 BEST | 模型目录下两个 JSON/JSONL 科学产物 |
| 16 | `plot_pocket_only.sh` | 只从 `pocket_only-0` 冻结 JSON 绘图 | 模型目录下两张 PNG |

每个脚本文件头记录独立提交命令。复用现有 allocation 时，控制器把对应脚本作为极短 `run_cmd` 写入新的 launch；SHA-256、锁路径和一次性门禁命令只进入执行记录，不进入这些正式脚本。

正式顺序不能把绘图合并到模型前向。绘图失败不会使已原子发布的 `evaluation.json` 与 `docking_handoff_candidates.jsonl` 失效，也不能触发第二次模型前向。模型评测目标目录已经存在时入口失败关闭，不覆盖既有结果。
