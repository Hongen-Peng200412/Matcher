# Matcher 训练与运行入口

本目录提供一套可以随项目复制的 Slurm 任务提交系统。读者只需要准备一个 Bash 任务脚本，就可以申请计算资源、冻结本次执行所见的完整项目、保留启动证据，并按需通过锁文件控制开始时间、重复执行和退出。

“任务根目录”指需要冻结并执行的项目目录。默认任务根目录是本目录的上一层，即 Matcher 项目根目录；通用脚本内部不写死项目名称。

## 1. 直接启动正式任务

先把任务脚本放入 `训练与运行/sh/`，然后在服务器的 Matcher 项目根目录运行：

```bash
bash 训练与运行/submit_task.sh \
  --sh 训练与运行/sh/precompute_data.sh \
  --resource cpu \
  --cpus 16 \
  -- \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --stage1-root /storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477/artifacts/Find_0 \
  --output-root /storage/penghongen/tmp/Matcher \
  --coverage-threshold 0.30 \
  --selection-seed 3407 \
  --splits train validation \
  --workers 16 \
  --products pp \
  --pp-top-k 1024
```

GPU 示例：

```bash
bash 训练与运行/submit_task.sh \
  --sh 训练与运行/sh/train_experiment.sh \
  --resource a800 \
  --gpus 1 \
  --cpus 16 \
  -- --plan configs/experiments/initial/full_scratch_smi_ted_light_289m_oracle.yaml
```

validation 选定唯一方案后，保留集只评估一次。下例中的计划、阶段和 BEST 路径必须属于
同一个已完成训练阶段。入口把现存 manifest 的稳定科学内容、calibration 候选与身份记录的
组合指纹绑定到该 checkpoint 和输出路径；对同一稳定数据指纹不能再次运行评估：

```bash
bash 训练与运行/submit_task.sh \
  --sh 训练与运行/sh/evaluate_calibration.sh \
  --resource a100 \
  --gpus 1 \
  --cpus 16 \
  -- \
  --plan configs/experiments/initial/full_scratch_molformer_joint.yaml \
  --stage full_scratch_molformer_joint \
  --checkpoint /storage/penghongen/tmp/Matcher/experiments/<plan>/<run>/<stage>/BEST.pt \
  --output /storage/penghongen/tmp/Matcher/calibration/<selected-run>.json
```

`--sh` 接受三种等价写法：只有文件名、相对于任务根目录的路径、任务根目录内的绝对路径。三种写法都会记录为同一个项目内相对路径，并从同一份 release 执行。

默认行为是获得资源后立即执行一次任务，随后自动退出并释放资源。`--pre_hold` 会在第一次
执行前创建 `pre_lock`，等待人工删除；`--after_hold` 会在每次任务结束后创建 `try_lock`，
继续保留 allocation 和资源。两个开关相互独立，可以只用一个或同时使用。

## 2. 一次提交到底做了什么

`submit_task.sh` 完成以下工作：

1. 解析任务脚本、硬件类型、GPU 数、CPU 核数和其他 Slurm 参数。
2. 确认任务脚本确实位于任务根目录内。
3. 调用 `sbatch/task.sbatch` 申请计算资源。
4. 计算资源启动后，由 `runtime/allocation_runner.sh` 建立锁文件和动态命令。
5. 每次真正开始执行动态命令前，由 `runtime/create_release.sh` 冻结当时的项目内容。
6. 由 `runtime/create_launch.sh` 保存本次执行所用 release、任务脚本和资源参数。

## 3. release 为什么在执行前创建

Slurm 作业可能排队数小时。如果在提交命令运行时就复制代码，作业真正开始时可能仍执行旧代码。本系统在每次即将执行任务时才计算项目内容哈希并创建 release；启用 `--after_hold` 后，删除 `try_lock` 触发的再次执行也会建立对应 release，因此每次执行与 release 一一对应。

release 是只读运行副本，不是训练结果目录。任务产生的 checkpoint、指标、概率图或其他业务产物，仍由任务脚本和项目代码决定位置。

## 4. 服务器目录分别保存什么

完整模式默认把控制记录放在：

```text
$HOME/Feedback/Matcher/
├── allocations/<job-id>/
│   ├── out
│   ├── err
│   ├── run_cmd_<job-id>.sh
│   ├── after_lock_<job-id>
│   └── kill_lock_<job-id>
├── launches/<job-id>/<launch-id>/
│   ├── launch.json
│   └── run_cmd.sh
└── releases/Matcher_<内容哈希>/
    ├── manifest.json
    └── Matcher/
```

- `allocations/` 保存 Slurm 资源内当前可控制的执行状态和合并日志。
- `launches/` 保存每次动态命令执行时的证据，不保存模型结果。
- `releases/` 保存按内容哈希冻结的项目副本；内容相同的副本会复用。
- `manifest.json` 保存来源目录、Git 提交、分支、工作区是否有未提交修改和完整内容哈希。
- `launch.json` 保存作业编号、release 路径、任务脚本、资源参数和 `TASK_RUN_STAMP`。

可用 `--feedback-root /指定目录` 修改这三类控制记录的共同根目录。

## 5. 任务脚本应怎样编写

任务脚本放在 `训练与运行/sh/`，或者放在任务根目录内的其他明确位置。最小示例：

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "${TASK_PROJECT_ROOT}"
python -m your_package.your_entry
```

完整模式下，`TASK_PROJECT_ROOT` 指向该次 release 中的项目根目录；`TASK_RUN_STAMP` 是本次执行的唯一名称；`TASK_LAUNCH_DIR` 指向启动证据目录。任务脚本不得假定调用者当前位于某个固定目录。

## 6. 参数怎样传给任务脚本

在提交系统参数之后写 `--`，后面的内容会按原始参数边界传给任务脚本：

```bash
bash 训练与运行/submit_task.sh \
  --sh prepare_labels.sh \
  --resource cpu \
  --cpus 16 \
  -- --split validation --workers 16
```

动态命令通过 Bash 的安全转义保存这些参数，包含空格的单个参数不会被错误拆分。

## 7. 资源参数与默认值

`--resource` 支持 `cpu`、`a100`、`a800`、`h100` 和 `h200`。GPU 任务必须填写正整数 `--gpus`。`--cpus` 未填写时采用以下默认值：CPU 为 16；A100 为每张卡 16；A800、H100、H200 为每张卡 24。

可以显式提供 `--partition`、`--qos`、`--nodelist`、`--mem`、`--time`、`--nodes` 和 `--array`。显式值优先于硬件类型默认值。`--pre_hold` 控制第一次执行前是否等待，`--after_hold` 控制任务结束后是否保留资源；两者默认都关闭。

## 8. 四种锁文件如何控制任务

- `pre_lock_<job-id>`：仅使用 `--pre_hold` 时创建。删除后才开始第一次执行。
- `try_lock_<job-id>`：仅使用 `--after_hold` 时在执行结束后创建。可以先编辑 `run_cmd_<job-id>.sh`，再删除它以复用当前资源执行下一次命令。
- `kill_lock_<job-id>`：人工创建后，监视进程会终止当前命令的进程组，但不释放 Slurm 资源。
- `after_lock_<job-id>`：删除后不再执行新命令，并退出 Slurm 作业、释放资源。

不要用 `scancel` 代替 `kill_lock` 或 `after_lock`，除非用户明确要求释放整个作业。

## 9. simple 模式适合什么任务

`--simple` 保留同一套锁控制，但不创建 release 和 launch，任务直接读取当前项目目录。控制文件和 Slurm 日志统一放在 `$HOME/SIMPLE_RUN`。没有 `--pre_hold` 或 `--after_hold` 时，simple 任务同样只执行一次并自动释放资源。

它适合严格的一次性检查，不适合需要长期追溯的正式训练、数据生产或推理。正式任务默认使用完整模式。

## 10. 怎样选择另一个任务根目录

复制完整的 `训练与运行/` 到另一个项目后，直接运行新项目中的 `submit_task.sh`，默认任务根目录就是新项目根目录。也可以从现有入口显式选择：

```bash
bash 训练与运行/submit_task.sh \
  --task-root /home/user/AnotherProject \
  --sh /home/user/AnotherProject/训练与运行/sh/task.sh \
  --resource cpu \
  --cpus 16
```

真正位于任务根目录之外的脚本会被拒绝。需要提交另一个项目时，应选择包含该脚本的项目作为 `--task-root`，而不是绕过冻结机制直接执行外部脚本。

可复制的完整目录为：

```text
训练与运行/
├── README.md
├── submit_task.sh
├── sbatch/task.sbatch
├── runtime/allocation_runner.sh
├── runtime/create_release.sh
├── runtime/create_launch.sh
└── sh/                     # 项目自己的任务脚本
```

## 11. 怎样修改或新增任务

通常只需新增或修改 `训练与运行/sh/` 中的人类入口脚本。资源申请留在 `submit_task.sh` 参数中；项目环境、Python 入口、配置文件和结果目录写在任务脚本中。不要把某次作业编号、某台节点或某个临时目录写进通用运行组件。

修改 `submit_task.sh`、`sbatch/` 或 `runtime/` 时，应同时验证相对路径、绝对路径、显式 `--task-root`、完整模式和 simple 模式。

## 12. 无卡验证边界

不申请计算资源也能验证：

- 所有 Bash 文件通过 `bash -n`。
- 用假的 `sbatch` 检查资源参数和项目内任务相对路径。
- 在临时目录模拟 `SLURM_JOB_ID`，验证 release、launch、四锁和动态命令。
- 比较 release 的内容哈希与复制后内容。

无卡验证不能证明服务器分区、QOS、GPU 驱动、训练环境和真实数据可用。正式提交前仍需检查服务器资源和任务脚本自己的运行条件。

## 13. 项目边界

Matcher 的正式任务只从 `/home/penghongen/My_Project/Matcher` 冻结 release。A–G 与 Stage1 产物仍由 AdaLigand 生产并从其正式存储目录只读加载；Matcher 的版本化清单、105 通道缓存、PP token 缓存、checkpoint、指标和评估结果只能写入 `/storage/penghongen/tmp/Matcher/`，不写回 AdaLigand 代码目录。release、launch、Slurm 日志和四锁写入 `/home/penghongen/Feedback/Matcher/`，不属于业务产物根限制。

## 14. 当前 Matcher 任务脚本

- `sh/build_manifest.sh`：只构建或核验版本化监督清单，不遍历候选体素缓存。
- `sh/precompute_data.sh`：先构建或复用清单，再按 `--products voxel pp` 显式选择 105 通道体素缓存、PP token 缓存或两者；选择 `pp` 时必须同时给出 `--pp-top-k`。
- `sh/train_experiment.sh`：读取调用方通过 `--plan` 指定的正式 YAML，顺序执行其中一个或多个训练阶段。
- `sh/evaluate_calibration.sh`：核对已选定阶段的 `BEST.pt`、`resolved.json`、阶段 `summary.json` 与运行 `run.json`，要求现存冻结清单且不重建，只评估 calibration。

`build_manifest.sh` 与 `precompute_data.sh` 的清单参数必须显式提供 `data-root/stage1-root/output-root/coverage-threshold/selection-seed`。`precompute_data.sh` 还必须显式提供 `--products`；仅当选择 `pp` 时提供正整数 `--pp-top-k`。`train_experiment.sh` 只接收 `--plan`，其它科学与训练参数都由 YAML 唯一声明。`evaluate_calibration.sh` 必须显式提供 `--plan/--stage/--checkpoint/--output`。它核对 checkpoint 最佳轮次、阶段完成摘要、模型与训练配置，以及训练时 `data_root/stage1_root/output_root/coverage_threshold/selection_seed/pp_top_k`；数据清单缺失时直接失败，不在保留评估中重建。

### 版本化监督清单字段

清单位于 `data.output_root/manifests/closed_set_v<schema>/`。`manifest.json` 的顶层字段如下：

| 字段 | 类型与语义 |
| --- | --- |
| `schema_version` | 整数；任何字段或监督语义变化都必须提升版本。 |
| `created_at_utc` | 首次构建清单的带时区 ISO 8601 时间；扩展划分时保持不变。 |
| `updated_at_utc` | 最近一次原子发布的带时区 ISO 8601 时间。 |
| `data_root` | A–G、语言表示和配体图的只读根。 |
| `stage1_root` | Stage1 Find_0 产物的只读根。 |
| `output_root` | Matcher 清单和缓存的可写根。 |
| `coverage_threshold` | 候选与 occurrence 双向覆盖率共同采用的阈值。 |
| `selection_seed` | 同一身份存在多个有效 occurrence 时的确定性选择种子。 |
| `language_selection_rule` | occurrence 排序、循环起点和固定选择规则的中文说明。 |
| `seen_identity_rule` | 按语言模型与 `(class_name,object_key)` 判定 train seen 的中文说明。 |
| `class_names` | 固定六分类顺序，必须是 `background/ion/nucleotide_like/peptide_like/small_molecule/sugar`。 |
| `available_splits` | 已原子物化的划分名称列表。 |
| `deferred_splits` | 尚未物化的划分名称列表。 |
| `splits` | 每个已物化划分到 `candidates` 和 `identities` JSONL 文件名的映射。 |
| `excluded_pdb` | 整 PDB 排除记录 JSONL 的文件名。 |
| `split_pdb_counts` | 每个划分的 `discovered/included/excluded` PDB 数。 |
| `summary` | 每划分候选类别数、身份数、语言可用数、多 SMILES 数、seen 数及排除原因计数。 |

每个 `<split>_candidates.jsonl` 对应一个 Stage1 候选，字段如下：

| 字段 | 类型与语义 |
| --- | --- |
| `split` | 候选所属划分。 |
| `pdb_id` | 小写 PDB 标识。 |
| `candidate_index` | 该 PDB 的从 0 开始连续候选编号。 |
| `source_tree_id` | Stage1 固化候选来源树编号。 |
| `source_node_id` | Stage1 固化候选来源节点编号。 |
| `source_threshold_grid_index` | Stage1 固化阈值网格编号。 |
| `full_candidate_voxel_count` | 未裁剪完整全局 V 组件的唯一体素数。 |
| `class_name` | 六分类字符串。 |
| `class_index` | `class_names` 中的整数下标。 |
| `matched_candidate_id` | 命中的真实 occurrence 编号；背景未命中时为 `null`。 |
| `matched_object_key` | 命中身份的 object_key；背景未命中时为 `null`。 |
| `candidate_coverage` | 交集占完整候选组件的比例；未命中时为 0。 |
| `occurrence_coverage` | 交集占真实 occurrence 的比例；未命中时为 0。 |
| `overlap_score` | 两侧覆盖率几何平均；未命中时为 0。 |
| `language_target_available` | 按语言家族记录“身份图和所选语言表示同时可用”的布尔值。 |
| `identity_seen_in_train_by_language` | 按语言家族记录前景正身份是否曾作为 train 可监督目标；背景或没有前景身份时为 `null`。 |

每个 `<split>_identities.jsonl` 对应一个 PDB 内 `(class_name,object_key)` 五类前景身份，字段如下：

| 字段 | 类型与语义 |
| --- | --- |
| `split` | 身份所属划分。 |
| `pdb_id` | 小写 PDB 标识。 |
| `class_name` | 五类非背景类别字符串。 |
| `class_index` | 六分类中的整数下标，取值为 1–5。 |
| `object_key` | PDB 内与类别共同定义去重身份的非空字符串。 |
| `occurrence_candidate_ids` | 聚合到该身份的升序 occurrence 编号列表。 |
| `graph_available` | 对应配体图归档是否存在。 |
| `graph_relative_path` | 相对 `data_root` 的配体图归档路径。 |
| `language` | `molformer` 与 `smi_ted_289m` 两个语言家族的选择证据。 |
| `seen_in_train_by_language` | 按语言家族记录该 `(class_name,object_key)` 是否在 train 有可监督正目标。 |

`language.<family>` 的 `available` 表示是否至少有一个有效归档；`selected_candidate_id/selected_relative_path` 在不可用时为 `null`；`valid_candidate_ids` 保存全部有效 occurrence；`failed_occurrences` 逐项保存缺文件或归档无效原因；`distinct_model_smiles` 保存有效归档中去重后的模型输入字符串。`excluded_pdb.jsonl` 至少保存 `split/pdb_id/reason`；`reason=missing_required_product` 时另含 `missing_paths`。身份范围或语言缺失只令候选身份目标不可用，不写入整 PDB 排除记录。

校准入口分别记录本次 YAML 与训练 release 中原 YAML 的路径和 SHA-256，但不要求两个 YAML 字节相同：训练完成后可以显式调整 validation 工作进程数、预取量和单工作进程缓存等 I/O 资源。模型、训练超参数和上段六个科学数据字段仍必须逐项相同；因此 I/O 调整不会被误报成科学方案变化，也不能借 I/O 调整更换监督清单。

支持类 macro-F1 只平均真实候选数大于零的类别；固定六类 macro-F1 把零支持类 F1 记为 0 后仍平均 `background/ion/nucleotide_like/peptide_like/small_molecule/sugar` 六类。兼容字段 `evaluation.metrics.six_class.macro_f1` 等于 `macro_f1_supported_classes`，validation 选模继续使用该兼容口径。

一次评估把结果 JSON 写到调用方指定的 Matcher 业务根子路径；`data.output_root/calibration/` 保存 claim JSON 和空互斥锁文件。结果 JSON 顶层字段如下：

| 字段 | 类型与语义 |
| --- | --- |
| `schema_version` | 整数；结果与 claim 共用且联动提升的格式版本。 |
| `split` | 字符串，固定为 `calibration`。 |
| `plan_path` | 本次 calibration 读取的 YAML 绝对路径。 |
| `plan_sha256` | 本次 calibration YAML 的 SHA-256。 |
| `plan_name` | YAML 声明的安全计划名。 |
| `stage_name` | validation 选定并已完成的阶段名。 |
| `task_name` | `candidate_pretrain` 或 `full_matcher`。 |
| `language_family` | 完整 Matcher 的语言表示；候选预训练为 `null`。 |
| `identity_variant` | 完整 Matcher 的 `oracle/detached/joint`；候选预训练为 `null`。 |
| `checkpoint` | 下表定义的 BEST 与训练证据对象。 |
| `data` | 下表定义的现存清单与 calibration 记录证据对象。 |
| `claim_path` | 绑定本次选择的一次性 claim JSON 绝对路径。 |
| `runtime` | 下表定义的数据加载、数值后端与设备对象。 |
| `evaluation` | 下表定义的完整损失、指标、计数、耗时与吞吐对象。 |
| `hostname` | 实际执行节点名。 |
| `slurm_job_id` | Slurm Job ID；非 Slurm 运行时为 `null`。 |
| `task_launch_dir` | launch 留证目录；环境变量未提供时为 `null`。 |
| `started_at_utc` | 开始评估的带时区 ISO 8601 时间。 |
| `finished_at_utc` | 完成评估的带时区 ISO 8601 时间。 |

`checkpoint` 字段如下：

| 字段 | 类型与语义 |
| --- | --- |
| `path` | 已完成阶段 `BEST.pt` 的绝对路径。 |
| `sha256` | `BEST.pt` 的 SHA-256。 |
| `epoch` | 从 0 开始的最佳 epoch，必须等于 checkpoint 的 `best_epoch`。 |
| `global_step` | 保存最佳模型时累计完成的参数更新数。 |
| `best_score` | 该最佳 epoch 的有限 validation 选择分数。 |
| `resolved_path` | 同阶段 `resolved.json` 的绝对路径。 |
| `resolved_sha256` | `resolved.json` 的 SHA-256。 |
| `summary_path` | 只在阶段收口时发布的阶段 `summary.json` 路径。 |
| `summary_sha256` | 阶段 `summary.json` 的 SHA-256。 |
| `run_path` | 阶段上一级 `run.json` 的绝对路径。 |
| `run_sha256` | `run.json` 的 SHA-256。 |
| `training_plan_path` | `run.json` 记录的训练 release YAML 路径。 |
| `training_plan_sha256` | 评估时读取上述训练 YAML 得到的 SHA-256。 |
| `model_config` | 已与本次 YAML 逐字段相等的模型配置对象。 |
| `training_config` | 已与本次 YAML 逐字段相等的训练配置对象。 |

`data` 字段如下：

| 字段 | 类型与语义 |
| --- | --- |
| `manifest_path` | 现存 `manifest.json` 的绝对路径；缺失时禁止评估。 |
| `manifest_sha256` | 现存 `manifest.json` 全文件的 SHA-256。 |
| `training_manifest_sha256` | 训练开始时留存的清单摘要；既有首轮运行未记录，因此为 `null`。 |
| `manifest_semantic_sha256` | 下文所定义 calibration 稳定清单字段的规范 JSON 摘要。 |
| `candidates_path` | `calibration_candidates.jsonl` 的绝对路径。 |
| `candidates_sha256` | calibration 候选 JSONL 的 SHA-256。 |
| `identities_path` | `calibration_identities.jsonl` 的绝对路径。 |
| `identities_sha256` | calibration 身份 JSONL 的 SHA-256。 |
| `fingerprint` | 下文公式得到的稳定数据指纹，也是 claim 文件名的一部分。 |

`runtime` 字段如下：

| 字段 | 类型与语义 |
| --- | --- |
| `candidate_count` | 冻结 calibration Dataset 的候选数，必须等于评估样本数。 |
| `physical_batch_size` | 单次前向候选数，沿用所选阶段。 |
| `num_workers` | calibration DataLoader 工作进程数，沿用本次 YAML 的 validation 设置。 |
| `prefetch_factor` | 每个工作进程预取的物理批次数。 |
| `cache_gib_per_worker` | 单个工作进程五类最近最少使用缓存的 GiB 上限。 |
| `multiprocessing_context` | 固定为 `spawn`。 |
| `persistent_workers` | 固定为 `false`。 |
| `worker_generator_seed` | 只供 calibration 工作进程播种的独立 Torch 种子。 |
| `numeric_backend` | 含 `python_numpy_torch_seed/cudnn_benchmark/cudnn_deterministic/deterministic_algorithms/float32_matmul_precision`。 |
| `device` | 实际 CUDA 设备名称。 |
| `torch_version` | 执行评估的 PyTorch 版本。 |

`evaluation` 字段如下；任一零分母浮点指标在内存中为 NaN，严格 JSON 落盘时为 `null`：

| 字段 | 类型与语义 |
| --- | --- |
| `loss.total` | 按完整划分真实分母汇总的加权总损失。 |
| `loss.classification` | 六分类损失。 |
| `loss.ligand_area` | 候选配体区域损失。 |
| `loss.binding_area` | 受体结合区域损失。 |
| `loss.identity` | 配置身份范围内有效身份目标上的闭集身份损失。 |
| `metrics.six_class.accuracy` | 全部候选的六分类正确率。 |
| `metrics.six_class.macro_f1` | 兼容字段，等于 `macro_f1_supported_classes`。 |
| `metrics.six_class.macro_f1_supported_classes` | 只平均真实候选数大于零类别的 F1。 |
| `metrics.six_class.macro_f1_all_classes` | 零支持类 F1 记为 0 后固定平均六类。 |
| `metrics.six_class.per_class` | 六个类别分别保存 `precision/recall/f1/support`。 |
| `metrics.six_class.confusion_matrix` | 整数 `(6,6)`；第一轴为真实类，第二轴为预测类。 |
| `metrics.intrinsic_identity.top1` | 忽略六分类预测后的身份 Top-1。 |
| `metrics.intrinsic_identity.topk` | 忽略六分类预测后的身份 Top-K。 |
| `metrics.intrinsic_identity.top_k` | 报告采用的最大名次 K。 |
| `metrics.intrinsic_identity.eligible_count` | 真实类别属于配置身份范围且身份目标有效的候选数。 |
| `metrics.conditional_identity.top1` | 六分类精确判对真实子类条件下的身份 Top-1。 |
| `metrics.conditional_identity.topk` | 同一条件集合上的身份 Top-K。 |
| `metrics.conditional_identity.top_k` | 报告采用的最大名次 K。 |
| `metrics.conditional_identity.eligible_count` | 六分类精确判对且身份目标有效的配置范围候选数。 |
| `metrics.conditional_identity.seen_top1` | 当前语言模型 seen 子集 Top-1。 |
| `metrics.conditional_identity.seen_topk` | 当前语言模型 seen 子集 Top-K。 |
| `metrics.conditional_identity.seen_count` | 当前语言模型 seen 子集候选数。 |
| `metrics.conditional_identity.unseen_top1` | 当前语言模型 unseen 子集 Top-1。 |
| `metrics.conditional_identity.unseen_topk` | 当前语言模型 unseen 子集 Top-K。 |
| `metrics.conditional_identity.unseen_count` | 当前语言模型 unseen 子集候选数。 |
| `metrics.end_to_end_identity.precision` | 通过配置范围前景门且身份正确数除以预测阳性数。 |
| `metrics.end_to_end_identity.recall` | 通过配置范围前景门且身份正确数除以有身份目标的真实阳性数。 |
| `metrics.end_to_end_identity.f1` | 上述 precision 与 recall 的调和平均。 |
| `metrics.end_to_end_identity.true_positive` | 通过配置范围前景门且最终身份正确的候选数。 |
| `metrics.end_to_end_identity.predicted_positive` | 通过配置范围前景门且最终身份落入该范围的候选数。 |
| `metrics.end_to_end_identity.actual_positive` | 配置范围内有当前语言身份目标的真实候选数。 |
| `metrics.small_molecule_identity` | 小分子子清单的 intrinsic/conditional/end_to_end；端到端使用严格 `small_molecule` 六分类门。 |
| `metrics.all_non_background_identity` | 五类共享清单的 intrinsic/conditional/end_to_end；端到端只要求预测为任一前景类。 |
| `metrics.identity_per_class` | 五类逐类报告；端到端预测阳性按最终 Top-1 身份类别归桶。 |
| `metrics.area.ligand_micro_dice` | 跨候选、跨体素汇总的配体区域 Dice。 |
| `metrics.area.binding_micro_dice` | 跨候选、跨体素汇总的结合区域 Dice。 |
| `metrics.area.ligand_dice_denominator` | 配体区域预测与真实体素数之和。 |
| `metrics.area.binding_dice_denominator` | 结合区域预测与真实体素数之和。 |
| `sample_count` | 实际评估候选数，必须等于 `runtime.candidate_count`。 |
| `identity_eligible_count` | 配置身份范围内有当前语言身份目标的真实候选数。 |
| `elapsed_seconds` | 完整评估墙钟秒数。 |
| `samples_per_second` | `sample_count / elapsed_seconds`。 |

`claim_<data_fingerprint>.json` 字段如下：

| 字段 | 类型与状态语义 |
| --- | --- |
| `schema_version` | 与结果 JSON 共用的整数格式版本。 |
| `status` | `claimed` 表示选择已绑定且允许同参数失败重试；`completed` 表示结果已发布并禁止再次运行。 |
| `data_fingerprint` | 与结果 `data.fingerprint` 相等。 |
| `plan_sha256` | 与结果 `plan_sha256` 相等。 |
| `stage_name` | 已绑定阶段名。 |
| `checkpoint_sha256` | 已绑定 `BEST.pt` 的 SHA-256。 |
| `checkpoint_path` | 已绑定 `BEST.pt` 的绝对路径。 |
| `output_path` | 首次 claim 时绑定的结果 JSON 路径。 |
| `output_sha256` | `claimed` 时为 `null`；`completed` 时为结果 JSON 的 SHA-256。 |
| `claimed_at_utc` | 首次绑定时带时区的 ISO 8601 时间。 |
| `completed_at_utc` | `claimed` 时为 `null`；`completed` 时为结果记录的评估完成时间。 |

结果已发布而 claim 尚未收口时，同参数重试先核验结果中的计划、阶段、checkpoint 和数据指纹，再只把 claim 改为 `completed`，不再次执行模型。`claim_<data_fingerprint>.json.lock` 是入口不会主动删除的空互斥锁文件，只用于并发串行化，没有科学或完成状态语义；是否完成只能读取 claim 的 `status`。

组合指纹先对 manifest 中来源根、监督参数、类别顺序、语言/seen 规则、calibration 文件名、PDB 计数和 calibration 摘要组成的规范 JSON 计算 SHA-256；创建时间和 train/validation 摘要不参与。再把该摘要、calibration 候选 JSONL 摘要和身份 JSONL 摘要按 `semantic + "\n" + candidates + "\n" + identities + "\n"` 的 ASCII 字节再次计算 SHA-256。SHA-256 在此只用于文件内容留证和 claim 定位，不参与样本、监督或模型计算。

PP 缓存写入 `cache/pp_input_v1/top_<k>/<split>/<pdb>.npz`，并与 `F1_centered.npz` 的候选编号逐项对齐。令 `C` 为该 PDB 的候选数，`K=min(top_k,48³)`：

- `schema_version` 是 int64 标量，当前必须为 1。
- `top_k` 是 int64 标量，必须与路径中的 `<k>` 一致。
- `candidate_indices` 是 int64 `(C,)`，必须为从 0 开始的连续候选编号。
- `tokens` 是 float32 `(C,K,4)`；最后一轴依次为无量纲的 Stage1 原始概率和按局部空间物理半边长归一化的无量纲相对 XYZ。
- `index_zyx` 是 int16 `(C,K,3)`；最后一轴是中心 `48³` 内取值范围为 `[0,48)` 的 ZYX 索引。
- `probability_origin` 是 float32 `(3,)`，表示完整 Stage1 概率图的 XYZ 角点原点，单位 Å。
- `probability_voxel_size` 是 float32 `(3,)`，表示完整 Stage1 概率图的 XYZ 体素尺寸，单位 Å/voxel。

PP cache schema v1 使用 NumPy `argpartition` 决定入选子集；第 K 名概率并列时不另行改写选择语义。改变该规则必须提升 `PP_CACHE_SCHEMA_VERSION`，防止复用语义不同的缓存。缓存只消除重复解压和重复构造，不改变训练样本与 PP token。
