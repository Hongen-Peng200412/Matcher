# Stage3 小分子子集评测入口

本目录只保存 `test_smallmolecule_v1` 的短正式入口。只读来源、产物根、前向参数和冻结 checkpoint 输入位于 [`../../../test_smallmolecule/test_smallmolecule.yaml`](../../../test_smallmolecule/test_smallmolecule.yaml)；科学规则和产物字段见 [`../../../test_smallmolecule/README.md`](../../../test_smallmolecule/README.md)。

所有脚本都从 Matcher 仓库根目录运行，不接收位置参数。脚本只选择一个明确的 Python 模块、统一 YAML 和稳定实验标识；长路径和模型前向参数不在多个 shell 中重复维护。

## 脚本与产物

| 顺序 | 脚本 | 正式职责 | 主要产物 |
| ---: | --- | --- | --- |
| 1 | `prepare_stage1.sh` | 构建独立 manifest，并生成 Stage1-only JSON | `manifests/test_smallmolecule_v1/`、`evaluation/test_smallmolecule_v1/stage1_only/stage1_only.json` |
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

每个脚本文件头记录独立提交命令。release、launch、锁、文件摘要和运行状态只进入实验追溯文档与执行记录，不写入科学评估产物。

正式顺序不把绘图合并到模型前向。绘图只读取已经写出的 `evaluation.json`，因此图片重建不触发第二次模型前向。

## 入口之间的依赖关系

```text
prepare_stage1.sh
├── test_smallmolecule.build_manifest   → <manifest_root>/manifest.json 与 5 份 JSONL
└── test_smallmolecule.evaluate_stage1  → <evaluation_root>/stage1_only/stage1_only.json
    ↓
plot_stage1.sh                           → stage1_only/figures/ 下两张 Stage1-only PNG

evaluate_<variant>.sh
└── test_smallmolecule.evaluate_model    → <evaluation_root>/<experiment_id>/<training_run_name>/
                                          ├── evaluation.json
                                          └── docking_handoff_candidates.jsonl
    ↓
plot_<variant>.sh                        → 同一实验目录 figures/ 下两张 Stage1+Matcher PNG
```

- `prepare_stage1.sh` 是唯一会构建模型无关清单的正式入口。模型评估脚本只读取已经存在的清单，不重复计算 occurrence 目标或 Stage1-only 曲线。
- `plot_stage1.sh` 依赖 `stage1_only.json`；任一 `plot_<variant>.sh` 依赖同一实验的 `evaluation.json`。绘图失败不影响已写出的 JSON 或 JSONL 科学产物。
- 每个 `evaluate_<variant>.sh` 只改变传给 `--model` 的稳定实验标识。该标识在 YAML 的 `models` 映射中定位冻结 BEST 和 `training_run_name`，不直接充当代码模型类名。
- `density_blob_only` 和 `strongest` 是历史稳定标识；`blob_only-0`、`strongest-1`、`density_only-0`、`density_only-106` 与 `pocket_only-0` 是第二轮稳定标识。输出目录使用这些标识，不因同名模型代码以后发生变化而覆盖旧实验。

## YAML 字段分工

| YAML 路径 | 被哪些入口读取 | 具体职责 | 示例值 |
| --- | --- | --- | --- |
| `sources.stage3_occurrences` | `prepare_stage1.sh` | PocketXMol Stage3 冻结测试 occurrence 清单 | `/storage/penghongen/PocketXMol/data/smiles-v1/frozen/test.jsonl` |
| `sources.data_root` | `prepare_stage1.sh` | AdaLigand occurrence 类别与原始编号的共同数据根 | `/storage/penghongen/AdaLigand/Ori_Data` |
| `sources.prepared_root` | `prepare_stage1.sh` | 按 PDB 保存顶层 `PreparedLigand.smiles` 的目录根 | `/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_2/ligand_language_models/prepared` |
| `sources.real_manifest`、`sources.cryoatom2_manifest` | 清单构建与模型评估 | 两种受体条件的完整 Matcher test_0 候选和身份入口 | `/storage/penghongen/Matcher/manifests/test0_real_smiles_v1/manifest.json` |
| `sources.find_real_root`、`sources.find_cryoatom2_root` | `prepare_stage1.sh` | Find_1 完整 blob、centered 几何和排序字段 | `/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1/held_out_test_0` |
| `sources.emap2lig_root` | `prepare_stage1.sh` | Emap2lig 覆盖数组、原生概率均值和完整 blob | `/storage/penghongen/AdaLigand_stage1_inference/Emap2lig/official_find_li/held_out_test_0` |
| `outputs.manifest_root` | 清单构建及全部评估入口 | 模型无关清单目录 | `/storage/penghongen/Matcher/manifests/test_smallmolecule_v1` |
| `outputs.evaluation_root` | Stage1、模型评估与绘图入口 | 冻结 JSON、handoff 和图片共同根目录 | `/storage/penghongen/Matcher/evaluation/test_smallmolecule_v1` |
| `runtime.split` | 模型评估入口 | 两份来源 manifest 中读取的测试划分键 | `held_out_test_0` |
| `runtime.batch_size` | 模型评估入口 | 一次 CUDA 前向包含的候选数量 | `32` |
| `runtime.num_workers` | 模型评估入口 | 当前受体条件唯一活动 DataLoader 的工作进程数 | `8` |
| `runtime.prefetch_factor` | 模型评估入口 | 每个 DataLoader 工作进程预取的物理批次数 | `2` |
| `runtime.worker_generator_seed` | 模型评估入口 | DataLoader 工作进程随机种子 | `3407` |
| `runtime.sampler_seed` | 模型评估入口 | 不洗牌批采样器使用的确定性种子 | `3407` |
| `runtime.torch_seed` | 模型评估入口 | 当前评估进程与 CUDA 的随机种子 | `3407` |
| `runtime.pp_top_k` | 模型评估入口 | 每个候选保留的 PP token 数量上限 | `2048` |
| `runtime.cache_gib_per_worker` | 模型评估入口 | 每个 Dataset 工作进程的 PDB 载荷最近使用缓存预算，单位 GiB；`0` 表示禁用缓存 | `8.0` |
| `models.<experiment_id>.checkpoint` | 对应 `evaluate_<variant>.sh` | 待恢复的冻结 BEST；模型类和配置继续由其训练代码快照解释 | `/storage/penghongen/Matcher/runs/strongest-1/<run>/BEST.pt` 是格式示例 |
| `models.<experiment_id>.training_run_name` | 对应模型评估与绘图入口 | 当前实验在 `evaluation_root` 下的结果子目录名 | `train_strongest1_revised_job378587_20260920T230727` |

YAML 中的 checkpoint 和 manifest 路径是运行所需输入，不会被复制进 `evaluation.json` 或 `docking_handoff_candidates.jsonl`。这些输入对应哪次 release、launch 和服务器作业，由实验的“追溯与说明”及执行日志记录。
