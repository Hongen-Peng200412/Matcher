# Stage3 小分子独立评测工具

本目录实现 `test_smallmolecule_v1`：在 PocketXMol Stage3 冻结的 77 个测试 PDB 上，汇总 Find_1、Emap2lig 和冻结 Matcher BEST checkpoint 的 top-k 表现。它独立于 Matcher 的训练与完整 179-PDB `test_0` 评测，不修改训练 Dataset、模型、损失、checkpoint 或既有测试产物。

本 README 说明代码职责、科学规则和评测产物。manifest 的逐字段契约另见 [`MANIFEST_README.md`](MANIFEST_README.md)；正式 shell 入口另见 [`../../训练与运行/sh/small_molecule_test/README.md`](../../训练与运行/sh/small_molecule_test/README.md)。

## 输入、处理与输出概览

`test_smallmolecule.yaml` 集中指定下列只读输入：

- PocketXMol Stage3 的 446 个测试 occurrence；occurrence 是一个 PDB 中具有独立 `candidate_id` 的沉积配体实例。
- 真实受体与 CryoAtom2 两套 Find_1 `held_out_test_0` 候选和完整 blob 几何。
- Emap2lig 的原生候选、概率均值和候选与真实 occurrence 的体素交集。
- 两套 Matcher `test_0` manifest 中的完整 PDB 内 SMILES 身份轴。
- 每个正式实验的冻结 BEST checkpoint 与训练运行目录名。

代码先把上述来源冻结成模型无关 manifest，再生成 Stage1-only top-k，随后按模型运行一次真实身份前向并同时得到“全配体答案范围”和“small-molecule 答案范围”，最后只从冻结 JSON 绘图。

```text
冻结来源
├── PocketXMol Stage3 occurrence
├── Find_1 真实受体与 CryoAtom2 候选
├── Emap2lig 原生候选
└── Matcher 完整身份轴
    ↓ build_manifest.py
test_smallmolecule_v1 manifest
    ↓ evaluate_stage1.py
Stage1-only top-k 与逐候选轨迹
    ↓ evaluate_model.py + 冻结 BEST
Stage1+Stage2 top-k、逐候选身份预测与 docking 评测投影
    ↓ plot_results.py
冻结 JSON 驱动的 PNG 图
```

## 产物目录

以下两棵树分别列出稳定科学产物和由稳定 JSON 重建的图片。`<experiment_id>` 是配置 `models` 下的稳定实验标识，如 `strongest-1`；`<training_run_name>` 是该实验冻结训练运行的目录名。

```text
## <科学产物>
/storage/penghongen/Matcher/manifests/test_smallmolecule_v1/
├── manifest.json                                      # 全部派生文件、科学规则、冻结计数和 SHA-256 的唯一入口
├── stage3_occurrences.jsonl                           # 446 个 PocketXMol Stage3 测试 occurrence
├── identities.jsonl                                  # 77 个 PDB 内 180 个受支持 SMILES 身份
├── real_candidates.jsonl                             # 真实受体 Find_1 的 598 个 centered 候选
├── cryoatom2_candidates.jsonl                        # CryoAtom2 Find_1 的 632 个 centered 候选
└── emap2lig_candidates.jsonl                         # Emap2lig 的 30,561 个原生候选

/storage/penghongen/Matcher/evaluation/test_smallmolecule_v1/
├── stage1_only/
│   └── stage1_only.json                              # 三类 Stage1 来源的 top-k 曲线与逐 PDB 候选轨迹
└── <experiment_id>/<training_run_name>/
    ├── evaluation.json                               # checkpoint、代码、manifest、预测、曲线和追溯信息
    └── docking_handoff_candidates.jsonl              # 身份预测正确的 Stage3 候选评测投影

## <其他文件>
/storage/penghongen/Matcher/manifests/test_smallmolecule_v1/
├── source_snapshot.json                              # 全部实际读取来源的路径、字节数和 SHA-256（来源留证）
└── README.md                                         # manifest 字段与跨文件关系（可再生说明）

/storage/penghongen/Matcher/evaluation/test_smallmolecule_v1/
├── stage1_only/figures/
│   ├── topk_gaussian_stage1_only.png                 # Gaussian 排序 Stage1-only 曲线（可再生辅助导出）
│   └── topk_probability_mean_stage1_only.png         # 概率均值排序 Stage1-only 曲线（可再生辅助导出）
└── <experiment_id>/<training_run_name>/figures/
    ├── topk_gaussian_stage1_stage2.png               # Gaussian 排序 Stage1 与两档 Stage2 曲线（可再生辅助导出）
    └── topk_probability_mean_stage1_stage2.png       # 概率均值排序 Stage1 与两档 Stage2 曲线（可再生辅助导出）
```

`figures/` 只保存由冻结 JSON 重建的图片，不能替代 JSON 科学产物。模型评测目录通过临时目录一次发布；`evaluation.json` 与 `docking_handoff_candidates.jsonl` 不会只发布其中一个。

## 科学规则

### 唯一目标与候选三态

一个候选与一个 occurrence 同时满足候选覆盖率和 occurrence 覆盖率不低于 0.30 时，才形成有效覆盖。候选有多个有效覆盖时，先最大化两项覆盖率的几何平均；并列时选择较小的 `occurrence_id`。由此每个候选只有一个确定目标，并被划分为：

| `target_scope` | 唯一目标状态 | top-k 尝试次数规则 |
| --- | --- | --- |
| `background` | 没有 occurrence 通过双向覆盖门槛 | 消耗一次尝试，不能命中 |
| `non_stage3_foreground` | 唯一目标存在，但不属于 PocketXMol Stage3 测试 occurrence | 免费跳过，不增加尝试次数，不调用 Matcher |
| `stage3_foreground` | 唯一目标属于 PocketXMol Stage3 测试 occurrence | 消耗一次尝试；Stage1 在此直接命中，Stage2 还要求 SMILES 预测正确 |

固定分母为 77 个 PDB。`raw_rank` 表示候选在完整冻结顺序中的一基位置；`attempt_index` 只为消耗尝试的候选编号，因此 `non_stage3_foreground` 的 `attempt_index` 为 `null`。

### 候选顺序

每个 PDB 先排列 `candidate_selected=true` 候选，再追加 `candidate_selected=false` 候选。两段分别按当前分数降序排列，并以较小 `source_blob_index` 作为同分次级键：

- Gaussian 曲线使用 `gaussian_score`，只适用于 Find_1。
- 概率均值曲线使用 `source_probability_mean`。
- Emap2lig 只有一条不依赖受体模型的概率均值曲线；它不复制成真实受体和 CryoAtom2 两条曲线。

Stage1-only 与 Stage1+Stage2 必须具有相同 `candidate_order_sha256`。Matcher 只判断 Stage3 前景候选的 SMILES 是否匹配，不删除、插入或重排候选。

### 两档 SMILES 答案范围

Matcher 对一个候选只执行一次完整 PDB 内身份 forward，并保存全部有效身份槽的原始 `identity_logits`：

- `all_ligand` 在完整受支持身份轴上取最大 logit 对应的精确 SMILES。
- `small_molecule` 在相同 logits 上屏蔽原始 `type_tag` 不是 `small_molecule` 的身份槽，再取最大值。

第二档不是只向模型输入小分子身份，也不是使用 104 个 Stage3 真值身份作为答案集合。当前完整身份轴有 180 个身份，其中 127 个允许成为 small-molecule 答案；非小分类别身份仍参与同一次身份编码和身份清单注意力。

Matcher 前景头只属于训练辅助输出。本评测不读取 validation 前景阈值，不按前景概率筛选候选，也不把前景概率写入正式结果或 docking 评测投影。

### blob 世界中心

Find_1 与 Emap2lig 都使用完整来源 blob 的全局 ZYX 体素索引计算体素中心算术平均，再转换成世界 XYZ 坐标，单位 Å。中心不是 centered 80³ 框中心；即使完整 Find_1 blob 不能完全装入 centered 框，仍按完整 blob 计算中心。

## 主要产物字段

### Stage1-only 统计

#### `stage1_only/stage1_only.json`

该文件由 `evaluate_stage1.py` 写出，不加载 Matcher checkpoint。它保存两种 Find_1 排序和一条 Emap2lig 概率均值曲线。

| 顶层字段 | 数据类型或结构 | 具体含义 | 示例值 |
| --- | --- | --- | --- |
| `schema` | 字符串 | 固定契约版本 | `"matcher_test_smallmolecule_stage1_v1"` |
| `created_at_utc` | 字符串 | JSON 原子发布时的 UTC ISO 8601 时间 | `"2026-09-20T12:34:56+00:00"` 是格式示例 |
| `manifest` | 对象 | 当前 `manifest.json` 的绝对路径与 SHA-256 | `manifest.path="/storage/penghongen/Matcher/manifests/test_smallmolecule_v1/manifest.json"` |
| `runtime` | 对象 | 计算节点、Slurm Job、release 与 launch 环境来源 | `runtime.hostname="hnode02"` 是格式示例 |
| `coverage_threshold` | 浮点数 | 候选与 occurrence 共用的双向覆盖门槛 | `0.3` |
| `denominator_pdb_count` | 整数 | 所有 top-k 点的固定 PDB 分母 | `77` |
| `candidate_order` | 字符串 | selected 优先、段内分数降序和 blob 编号升序规则 | `"candidate_selected=true 候选按分数降序和 source_blob_index 升序排列, 随后追加按相同规则排列的 candidate_selected=false 候选"` |
| `attempt_rule` | 字符串 | 三态候选如何消耗尝试次数 | `"背景与 Stage3 目标候选消耗一次尝试; 非 Stage3 前景候选免费跳过"` |
| `curves.gaussian` | 对象 | `real_receptor` 与 `cryoatom2_receptor` 的 Gaussian 曲线 | 两个条件键 |
| `curves.source_probability_mean` | 对象 | 两种 Find_1 受体条件的概率均值曲线 | 两个条件键 |
| `emap2lig` | 曲线对象 | 唯一的 Emap2lig 原生概率均值曲线 | `sorting="source_probability_mean"` |
| `traces` | 嵌套对象 | 与每条曲线对应的逐 PDB 完整候选轨迹 | `traces.gaussian.real_receptor["11jb"]` |
| `checkpoint_hit_counts` | 对象 | k=1、3、5、10、20 的机械复现计数 | `real_gaussian` 等五个键 |

每条曲线包含 `name`、`source`、`sorting`、`pdb_count`、`pdb_ids`、`top_k`、`hit_count`、`hit_ratio`、`pdb_hit` 和 `candidate_order_sha256`。`top_k` 固定为 `[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]`；`hit_count[j]` 是在 `top_k[j]` 次计费尝试内命中的 PDB 数；`hit_ratio[j] = hit_count[j] / 77`。

每条候选轨迹包含 `source_blob_index`、`centered_box_index`、`raw_rank`、`attempt_index`、`action`、`target_scope`、`matched_occurrence_id`、`target_smiles` 和 `success`。`action` 取 `skip_non_stage3_foreground`、`spend_background` 或 `spend_stage3_target`，分别对应上一节的三态规则。

`runtime` 包含 `hostname`、`slurm_job_id`、`task_project_root` 和 `task_launch_dir`。后三项在对应 Slurm 或任务环境变量不存在时可为 `null`；正式服务器运行应保存实际 Job、release 和 launch 来源。

### Matcher 模型评测

#### `<experiment_id>/<training_run_name>/evaluation.json`

该文件由 `evaluate_model.py` 写出。模型类、模型配置和 checkpoint 解释从训练运行的 source release 恢复；当前评测 release 只负责读取冻结 manifest、运行候选前向和汇总结果。

| 顶层字段 | 数据类型或结构 | 具体含义 | 示例值 |
| --- | --- | --- | --- |
| `schema` | 字符串 | 固定模型评测契约版本 | `"matcher_test_smallmolecule_model_v1"` |
| `created_at_utc` | 字符串 | 模型评测目录原子发布时的 UTC ISO 8601 时间 | `"2026-09-20T12:34:56+00:00"` 是格式示例 |
| `model_name` | 字符串 | 配置 `models` 下的稳定实验标识 | `"strongest-1"` |
| `checkpoint` | 对象 | BEST 路径、SHA-256、epoch、validation check、selection 和 global step | `checkpoint.best_epoch=2` 是格式示例 |
| `model_code_source` | 对象 | 训练 release 的模型代码来源 | `model_code_source.model_code_source="training_release"` |
| `training_provenance` | 对象 | 训练 run、YAML、resolved config、manifest 快照与 release manifest 的路径和 SHA-256 | 每个文件字段都有对应 `_sha256` |
| `evaluation_code` | 对象 | 当前评测 release、launch、`run_cmd.sh` 和 YAML 的路径与 SHA-256 | `evaluation_code.run_cmd_path="/storage/penghongen/Matcher/launch/example/run_cmd.sh"` 是构造格式示例 |
| `manifest` | 对象 | 小分子 manifest 路径与 SHA-256 | `manifest.path="/storage/penghongen/Matcher/manifests/test_smallmolecule_v1/manifest.json"` |
| `stage1_only` | 对象 | 被复用的 Stage1-only JSON 路径与 SHA-256 | `stage1_only.path="/storage/penghongen/Matcher/evaluation/test_smallmolecule_v1/stage1_only/stage1_only.json"` |
| `source_test_manifests` | 条件映射 | 两种受体条件各自完整 `test_0` manifest 的路径与 SHA-256 | `real_receptor`、`cryoatom2_receptor` |
| `runtime` | 对象 | GPU、PyTorch、CUDA、Python、批量和 worker 等实际执行信息 | `foreground_head_usage="ignored"` |
| `predictions` | 条件映射 | 两种受体条件下每个 Stage3 目标候选的完整身份输出 | 详见下一节 |
| `curves` | 排序与条件映射 | 两种排序、两种受体条件、两档答案范围的 top-k 曲线 | `curves.gaussian.real_receptor.all_ligand` |
| `traces` | 排序与条件映射 | 与 `curves` 逐键对应的完整候选轨迹 | `traces.source_probability_mean.cryoatom2_receptor.small_molecule` |
| `handoff` | 对象 | docking 评测投影路径、记录数和筛选范围 | `selection_scope="evaluation_oracle_filtered"` |

追溯对象继续逐文件展开：

| 对象 | 必需字段 | 字段关系 |
| --- | --- | --- |
| `checkpoint` | `path`、`sha256`、`best_epoch`、`best_validation_check`、`best_score`、`global_step` | 六项共同定位并解释当前冻结 BEST；训练未保存的元数据可为 `null` |
| `model_code_source` | `model_code_source`、`training_run_json_path`、`training_project_root`、`training_release_manifest_path` | 第一项固定为 `training_release`；其它三项定位模型类与配置的训练时代码快照 |
| `training_provenance` | `run_json_path`、`config_path`、`resolved_config_path`、`manifest_path`、`manifest_snapshot_path`、`source_project_root`、`source_release_manifest_path` | 除目录字段 `source_project_root` 外，每个文件路径都有同名加 `_sha256` 的摘要字段 |
| `evaluation_code` | `project_root`、`release_manifest_path`、`launch_dir`、`launch_json_path`、`run_cmd_path`、`config_path` | 除目录字段 `project_root` 与 `launch_dir` 外，每个文件路径都有同名加 `_sha256` 的摘要字段 |
| `runtime` | `hostname`、`slurm_job_id`、`task_project_root`、`task_launch_dir`、`device`、`torch_version`、`cuda_runtime`、`batch_size`、`num_workers`、`prefetch_factor`、`python`、`foreground_head_usage` | 保存实际执行环境；`foreground_head_usage` 固定为 `ignored` |
| `handoff` | `path`、`record_count`、`selection_scope` | `path` 指向同目录 JSONL；`record_count` 等于其非空记录数；筛选范围固定为 `evaluation_oracle_filtered` |

`predictions.<condition>[]` 的每项对应一个 `target_scope=stage3_foreground` 的 Find_1 候选：

| 字段 | 数据类型与形状 | 具体含义 | 示例值 |
| --- | --- | --- | --- |
| `pdb_id` | 字符串 | 小写 PDB 标识 | `"11jb"` |
| `centered_box_index` | 整数 | `F2_centered.npz` 的候选编号 | `0` |
| `source_blob_index` | 整数 | 完整来源 blob 编号 | `12` |
| `candidate_selected` | 布尔值 | 候选位于排序前段还是追加段 | `true` |
| `matched_occurrence_id` | 整数 | 双向覆盖规则选出的唯一 Stage3 occurrence | `0` |
| `target_smiles` | 字符串 | 唯一目标的精确顶层 `PreparedLigand.smiles` | `"CCO"` |
| `blob_center_world_xyz_A` | 浮点列表，长度 3 | 完整 blob 的世界 XYZ 中心，单位 Å | `[12.5,8.0,-3.5]` 是格式示例 |
| `identity_smiles` | 字符串列表，长度 I | 当前 PDB 完整受支持身份轴 | `["CCO","CCN"]` 是格式示例 |
| `identity_logits` | 浮点列表，长度 I | 与 `identity_smiles` 逐槽对齐的原始匹配分数 | `[1.2,-0.4]` 是格式示例 |
| `small_molecule_answer_mask` | 布尔列表，长度 I | 与身份轴逐槽对齐；True 表示允许成为第二档答案 | `[true,false]` 是格式示例 |
| `predicted_smiles_all_ligand` | 字符串 | 完整身份轴 argmax 的精确 SMILES | `"CCO"` |
| `predicted_smiles_small_molecule` | 字符串 | 类别掩码后 argmax 的精确 SMILES | `"CCO"` |

下面是说明字段关系的构造记录，不代表某次正式评测的真实数值：

```json
{"blob_center_world_xyz_A":[12.5,8.0,-3.5],"candidate_selected":true,"centered_box_index":0,"identity_logits":[1.2,-0.4],"identity_smiles":["CCO","CCN"],"matched_occurrence_id":0,"pdb_id":"11jb","predicted_smiles_all_ligand":"CCO","predicted_smiles_small_molecule":"CCO","small_molecule_answer_mask":[true,true],"source_blob_index":12,"target_smiles":"CCO"}
```

#### `<experiment_id>/<training_run_name>/docking_handoff_candidates.jsonl`

该文件是后续 docking 接线的窄评测投影，但不是无需真值即可在线生成的生产候选列表。代码先使用 `target_scope=stage3_foreground` 和精确目标 SMILES 判断预测正确性，再写入记录；同一个物理候选还会因两种排序和两档答案范围形成多条记录。因此，`selection_scope` 固定为 `evaluation_oracle_filtered`，不能把该文件解释为部署时 Matcher 可自行选择的无真值 docking 输入。

| 字段 | 数据类型 | 具体含义 | 示例值 |
| --- | --- | --- | --- |
| `pdb_id` | 字符串 | 候选所属小写 PDB | `"11jb"` |
| `receptor_condition` | 字符串 | Find_1 候选条件 | `"real_receptor"` 或 `"cryoatom2_receptor"` |
| `sorting` | 字符串 | 产生当前排名的分数字段 | `"gaussian"` 或 `"source_probability_mean"` |
| `answer_scope` | 字符串 | Matcher 最终 argmax 的答案范围 | `"all_ligand"` 或 `"small_molecule"` |
| `source_blob_index` | 整数 | 完整来源 blob 编号 | `12` |
| `centered_box_index` | 整数 | Find_1 centered 候选编号 | `0` |
| `raw_rank` | 整数 | selected 优先完整顺序中的一基位置 | `3` |
| `attempt_index` | 整数 | 免费跳过非 Stage3 前景后的计费尝试编号 | `2` |
| `blob_center_world_xyz_A` | 浮点列表，长度 3 | 完整 blob 的世界 XYZ 中心，单位 Å | `[12.5,8.0,-3.5]` 是格式示例 |
| `predicted_smiles` | 字符串 | 与唯一目标精确相同的 Matcher 预测 SMILES | `"CCO"` |
| `checkpoint_path` | 字符串 | 冻结 BEST 的绝对路径 | `"/storage/penghongen/Matcher/runs/strongest-1/train_strongest1_revised_job378587_20260920T230727/BEST.pt"` |
| `checkpoint_sha256` | 字符串 | BEST 文件的 64 位十六进制摘要 | `"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"` 是构造格式示例 |
| `manifest_path` | 字符串 | 小分子 `manifest.json` 绝对路径 | `"/storage/penghongen/Matcher/manifests/test_smallmolecule_v1/manifest.json"` |
| `manifest_sha256` | 字符串 | manifest 文件的 64 位十六进制摘要 | `"abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"` 是构造格式示例 |
| `selection_scope` | 字符串 | 声明记录经过评测真值筛选 | `"evaluation_oracle_filtered"` |

下面是一条字段齐全的构造 JSONL 记录，不代表正式运行观测：

```json
{"answer_scope":"small_molecule","attempt_index":2,"blob_center_world_xyz_A":[12.5,8.0,-3.5],"centered_box_index":0,"checkpoint_path":"/storage/project/run/BEST.pt","checkpoint_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","manifest_path":"/storage/project/manifests/test_smallmolecule_v1/manifest.json","manifest_sha256":"abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789","pdb_id":"11jb","predicted_smiles":"CCO","raw_rank":3,"receptor_condition":"real_receptor","selection_scope":"evaluation_oracle_filtered","sorting":"gaussian","source_blob_index":12}
```

### 正式图片

`plot_results.py` 只读取 `stage1_only.json` 或模型 `evaluation.json` 中的 `top_k` 与 `hit_ratio`，不读取来源 NPZ、不加载 checkpoint，也不重算候选顺序。Stage1-only 图片各含真实受体 Find_1、CryoAtom2 Find_1 和一条 Emap2lig 曲线；模型图片各含两种受体条件下的 Stage1-only、全配体答案和 small-molecule 答案曲线。

## 代码组织与推荐阅读顺序

1. `test_smallmolecule.yaml`：核对全部只读来源、输出根、冻结 checkpoint、运行参数和机械计数。
2. `build_manifest.py`：连接 Stage3 occurrence、完整 Matcher 身份轴、Find_1 与 Emap2lig 候选，并原子发布 manifest。
3. `ranking.py`：定义 selected 优先候选顺序、三态尝试次数和曲线/轨迹字段。
4. `evaluate_stage1.py`：生成不依赖 Matcher 的 Stage1-only JSON。
5. `evaluate_model.py`：从训练 source release 恢复模型，执行一次完整身份 forward，并发布模型 JSON 与 docking 评测投影。
6. `plot_results.py`：只读取冻结 JSON 绘图。
7. `artifacts.py`：提供 YAML、JSON、JSONL 和 SHA-256 的读取与原子写盘。
8. `gates/`：保存进入服务器 release 的一次性机械门禁，不是正式运行入口。

`MANIFEST_README.md` 会作为 `README.md` 随 manifest 一起发布。`test_smallmolecule.yaml` 的 `models` 键是稳定实验标识；每项 `variant` 是 checkpoint 内必须精确匹配的模型结构变体。例如，`strongest-1` 可以绑定 `strongest`，而稳定实验目录不需要随模型类名变化。

## 稳定入口与验证边界

正式任务只调用 [`../../训练与运行/sh/small_molecule_test/`](../../训练与运行/sh/small_molecule_test/) 中的薄 shell。服务器门禁调用 [`gates/`](gates/)，本地排序契约测试调用 [`../../tmp/test_smallmolecule/`](../../tmp/test_smallmolecule/)；门禁与测试命令不得抄入正式运行命令。

本目录不会读取 calibration，不在测试集拟合阈值，不修改任何来源文件，也不把 `docking_handoff_candidates.jsonl` 声明为无真值生产推理接口。
