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
  --pp-top-k 1024 \
  --local-crop-strategy v_centroid
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
组合指纹绑定到该 checkpoint 和输出路径。claim 为 ``claimed`` 时只允许同一选择在
原路径失败重试；状态成为 ``completed`` 后，同一稳定数据指纹禁止再次运行评估：

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

`build_manifest.sh` 与 `precompute_data.sh` 的清单参数必须显式提供 `data-root/stage1-root/output-root/coverage-threshold/selection-seed`。`precompute_data.sh` 还必须显式提供 `--products` 和 `--local-crop-strategy`；仅当选择 `pp` 时提供正整数 `--pp-top-k`。`fixed_center` 表示固定 `[16,64)³`，`v_centroid` 表示按完整权威 V 质心解析候选起点。`train_experiment.sh` 只接收 `--plan`，其它科学与训练参数都由 YAML 唯一声明。`evaluate_calibration.sh` 必须显式提供 `--plan/--stage/--checkpoint/--output`。它核对 checkpoint 最佳轮次、阶段完成摘要、模型与训练配置，以及训练时 `data_root/stage1_root/output_root/coverage_threshold/selection_seed/pp_top_k/local_crop_strategy`；数据清单缺失时直接失败，不在保留评估中重建。

### 训练阶段产物与候选预训练初始化

每个阶段目录固定保存 `resolved.json`、`tracking.json`、`history.jsonl`、`BEST.pt`、`LAST.pt` 和阶段 `summary.json`。运行目录另保存开始前的 `run.json`；全部阶段及其 W&B online run 正常收口后才发布运行根 `summary.json`。

`resolved.json` 是严格 JSON 对象，顶层字段如下：

| 字段 | 类型与语义 |
| --- | --- |
| `stage` | JSON 对象；保存当前 YAML 阶段的名称、任务、语言家族、身份损失、身份范围、模型、训练和初始化配置。 |
| `data` | JSON 对象；字段见下表。 |
| `tracking` | JSON 对象；逐字段保存 `provider/mode/entity/project/group/log_every_global_steps/init_timeout_seconds`。 |
| `device` | 字符串；实际 CUDA 设备名称。 |
| `torch_version` | 字符串；实际 PyTorch 版本。 |
| `frozen_optimizer_semantics` | JSON 对象；逐字段保存 AdamW、ReduceLROnPlateau 与 FP16 GradScaler 的显式参数。 |
| `numeric_backend` | JSON 对象；逐字段保存 `python_numpy_torch_seed/cudnn_benchmark/cudnn_deterministic/deterministic_algorithms/float32_matmul_precision`。 |
| `initialization` | JSON 对象或 `null`；从零训练时为 `null`，加载候选预训练 BEST 时见后表。 |
| `train_candidate_count` | 非负整数；train 清单候选总数。 |
| `train_candidates_used_per_epoch` | 非负整数；每轮截成完整 global batch 后实际训练的候选数。 |
| `validation_candidate_count` | 非负整数；validation 清单候选总数。 |

`data` 字段如下：

| 字段 | 类型与语义 |
| --- | --- |
| `data_root` | 字符串；AdaLigand 原始数据只读根。 |
| `stage1_root` | 字符串；Stage1 固化推理产物只读根。 |
| `output_root` | 字符串；Matcher 清单与缓存可写根。 |
| `coverage_threshold` | 浮点数；候选与 occurrence 双向覆盖率阈值。 |
| `selection_seed` | 非负整数；身份 occurrence 的确定性选择种子。 |
| `pp_top_k` | 正整数；每个候选的 PP token 上限。 |
| `local_crop_strategy` | 字符串；`fixed_center` 或 `v_centroid`。 |
| `train_num_workers` | 正整数；一次 train 迭代建立的 DataLoader 工作进程数。 |
| `validation_num_workers` | 正整数；一次 validation 迭代建立的 DataLoader 工作进程数。 |
| `prefetch_factor` | 正整数；每个工作进程预取的物理批次数。 |
| `cache_gib_per_worker` | 正浮点数；单个工作进程五类 LRU 缓存的 GiB 总上限。 |
| `multiprocessing_context` | 字符串；当前固定为 `spawn`。 |
| `persistent_workers` | 布尔值；当前固定为 `false`。 |
| `worker_pool_lifecycle` | 字符串；当前为 `per_complete_loader_iteration`，表示每次完整 train 或 validation 迭代单独建立并回收进程池。 |
| `train_worker_generator_seed` | 非负整数；train DataLoader 专用生成器种子。 |
| `validation_worker_generator_seed` | 非负整数；validation DataLoader 专用生成器种子。 |
| `worker_generator_lifecycle` | 字符串；说明两个 DataLoader 各自持有生成器，建立迭代器不会推进模型的全局 Torch 随机状态。 |

`stage.model` 中，`initial_graph_gate` 与 `initial_inventory_gate` 是完整 Matcher 图分支和清单分支身份分数增量的初始标量门。二者必须是 `[0,1]` 内有限 JSON 数值，可以写成整数或浮点数；训练开始后仍是无边界可训练参数。候选预训练不构造身份打分分支，因此两项必须显式为 `0.0`。第 0–4 批正式 YAML 也都显式为 `0.0`，保持原有严格零增量初始化；对应历史 checkpoint 或 `resolved.json` 缺少字段时，恢复入口迁移为 `0.0`。正式 YAML 缺少任一字段立即失败，不使用迁移默认值。

`frozen_optimizer_semantics` 中，`adam_betas/adam_epsilon/adam_amsgrad/adam_maximize/adam_foreach/adam_capturable/adam_differentiable/adam_fused` 冻结 AdamW 行为；`scheduler/scheduler_mode/scheduler_factor/scheduler_patience/scheduler_threshold/scheduler_threshold_mode/scheduler_cooldown/scheduler_minimum_learning_rate/scheduler_epsilon` 冻结验证分数调度行为；`grad_scaler_initial_scale/grad_scaler_growth_factor/grad_scaler_backoff_factor/grad_scaler_growth_interval` 冻结可选 FP16 缩放器行为。数值类型分别沿用名称对应的 JSON 数值、布尔值或字符串，不存在隐藏库默认值。

`stage.initialize_candidate_from` 只接受另一项已完成运行中以 `BEST.pt` 结尾的服务器绝对路径，只允许用于 `full_matcher`。它不接受 `LAST.pt`、相对路径或候选预训练阶段继续继承参数。目标优化器构造前，入口要求来源 `task_name=candidate_pretrain`、checkpoint 最佳轮次自洽、阶段摘要与运行根摘要已经发布、W&B run ID/HTTPS 地址存在，并核对来源训练 YAML、`resolved.json`、checkpoint 与运行记录中的阶段、模型、训练、数据和追踪配置。来源与目标还必须具有相容的候选编码器结构，以及相同的 `data_root/stage1_root/output_root/coverage_threshold/selection_seed/pp_top_k/local_crop_strategy`。工作进程数、预取量和单工作进程缓存只影响 I/O，不参与来源—目标科学配置比较，但来源 YAML 与来源 `resolved.json` 仍须自洽。

训练主线不读取 `launch.json`、`run_cmd.sh` 或 release manifest，也不计算文件 SHA-256。选定第 4 批来源 BEST 后，提交者在该次任务专用的 `tmp/<task-id>/` 工作区生成一次性审计脚本，用它记录发布路径、release manifest 声明的内容摘要和关键文件当时的字节摘要，并把报告写入 Matcher diagnostics 和批次记录。该脚本只由实现历史与任务记录保存，不进入累计学习主线；它不独立重算或证明整棵 release 目录的内容摘要，也不进入训练包、模型输入、损失、选模或 `resolved.json`。

非 `null` 的 `initialization` 字段如下：

| 字段 | 类型与语义 |
| --- | --- |
| `checkpoint_path` | 字符串；已解析的来源 `BEST.pt` 绝对路径。 |
| `source_plan_name` | 字符串；来源候选预训练运行的计划名称。 |
| `source_stage_name` | 字符串；来源候选预训练阶段名称。 |
| `source_run_stamp` | 字符串；来源运行目录名称。 |
| `source_slurm_job_id` | 字符串；来源运行记录中的 Slurm Job ID。 |
| `source_epoch` | 非负整数；来源 BEST 的零起始 validation 最佳轮次。 |
| `source_global_step` | 非负整数；来源 BEST 保存时累计完成的参数更新数。 |
| `source_best_score` | 有限浮点数；来源候选预训练的 validation `evaluation.metrics.six_class.macro_f1`；该兼容字段等于只平均真实支持数大于 0 类别的 `macro_f1_supported_classes`。 |
| `source_tracking_run_id` | 非空字符串；来源 W&B online run ID。 |
| `source_tracking_url` | HTTPS 字符串；来源 W&B online 曲线地址。 |
| `source_candidate_model_config` | JSON 对象；含 `aggregation/use_manual_features/use_sampling_features/use_candidate_token_attention_bias/model_dim/attention_heads/dropout/voxel_feature_channels/raunet_planes/recycle_iterations/gradient_checkpointing/bottleneck_attention_layers`，不含只影响身份分支的 `graph_layers`。 |
| `source_scientific_data` | JSON 对象；含来源与目标逐字段相等的 `data_root/stage1_root/output_root/coverage_threshold/selection_seed/pp_top_k/local_crop_strategy`。 |
| `resolved_path` | 字符串；来源阶段 `resolved.json` 的绝对路径。 |
| `stage_summary_path` | 字符串；来源阶段 `summary.json` 的绝对路径。 |
| `run_record_path` | 字符串；来源运行 `run.json` 的绝对路径。 |
| `run_summary_path` | 字符串；来源全部阶段及 W&B 收口后发布的根 `summary.json` 绝对路径。 |
| `training_plan_path` | 字符串；来源运行记录中的训练 YAML 绝对路径。 |
| `missing_keys` | 字符串数组；严格加载候选编码器后缺失的参数名，成功时为空。 |
| `unexpected_keys` | 字符串数组；严格加载候选编码器后多出的参数名，成功时为空。 |

`source_candidate_model_config` 的字段如下；元组写入 JSON 后使用数组表示。

| 字段 | JSON 类型与语义 |
| --- | --- |
| `aggregation` | 字符串；`mean_max` 或 `candidate_cross_attention`。 |
| `use_manual_features` | 布尔值；是否使用 12 维候选统计及身份类别概率特征。 |
| `use_sampling_features` | 布尔值；是否使用 4 维候选裁剪统计。 |
| `use_candidate_token_attention_bias` | 布尔值；是否启用四模态实体注意力偏置。 |
| `model_dim` | 正整数；候选表示宽度。 |
| `attention_heads` | 正整数；注意力头数，且必须整除 `model_dim`。 |
| `dropout` | 浮点数；位于 `[0,1)` 的无量纲丢弃概率。 |
| `voxel_feature_channels` | 正整数；RAUNet 输出通道数。 |
| `raunet_planes` | 九个正整数组成的 JSON 数组；顺序为 `enc0/enc1/enc2/enc3/bottleneck/dec3/dec2/dec1/dec0`。 |
| `recycle_iterations` | 正整数；RAUNet 回收轮数。 |
| `gradient_checkpointing` | 布尔值；训练时是否以重计算节省显存。 |
| `bottleneck_attention_layers` | 正整数；RAUNet 瓶颈注意力层数。 |

来源模型配置缺少 `use_manual_features/use_sampling_features/use_candidate_token_attention_bias` 时，对应开关在历史产物恢复入口迁移为 `false`；缺少 `initial_graph_gate/initial_inventory_gate` 时迁移为 `0.0`。来源数据配置缺少 `local_crop_strategy` 时迁移为 `fixed_center`；来源候选阶段缺少 `identity_scope` 时迁移为 `null`。正式 YAML 解析仍要求这些字段显式存在。该过程只构造 train/validation Dataset 并读取训练产物，不读取 calibration 候选、特征或指标。

### Validation 身份分数贡献诊断

完整 Matcher 可把只读诊断写入 W&B 的 `identity_contribution/validation/*` 指标树。该诊断只统计真实身份目标有效且当前 PDB 身份清单至少有两个有效槽的候选。对语言基线分数、已乘图门的分数增量、已乘清单门的分数增量，以及两个增量之和，均先在每个候选的有效身份槽内减去均值，再计算 L2 范数；补零身份槽不参与计算。

`native_centered_l2_mean`、`graph_increment_centered_l2_mean` 与 `inventory_increment_centered_l2_mean` 是上述候选内范数的跨候选均值。三个 `*_to_native_ratio_mean` 先对语言基线范数大于 `1e-12` 的候选逐一求比值，再跨候选平均；语言基线没有可辨识排序尺度时不发送比值。`multi_identity_eligible_count` 与 `nonzero_native_count` 分别保存两个分母。

正式 validation 前向在完成全部输入校验和三个配对打分器计算时，同时把语言基线与两个门控增量保存在内存输出中；诊断直接读取这些同批组件，不重跑打分器，也不再次执行由 `torch.any(...)` 触发的 device→host 布尔判定。每个物理批次新增的唯一 device→host 传输是一组聚合标量；该过程不修改模型、优化器或随机状态。诊断不写入损失、`EvaluationResult`、`history.jsonl`、checkpoint、validation 选择分数或 calibration 产物，也不能单独作为方案优劣结论。

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
| `identity_key_fields` | 固定为字符串数组 `["object_key","prepared_smiles"]`；两项共同定义身份。 |
| `prepared_smiles_rule` | 顶层 PreparedLigand.smiles 的逐字读取、显式 `null` 与缺失来源处理说明。 |
| `identity_class_rule` | 同一联合身份在全部已物化划分中只能映射到一个六分类类别。 |
| `language_selection_rule` | 在联合身份内按 occurrence 编号排序，并沿用 v3 的 `(pdb_id,object_key,language_family)` 选择盐值。 |
| `seen_identity_rule` | 按语言模型与 `(object_key,prepared_smiles)` 判定 train seen 的中文说明。 |
| `class_names` | 固定六分类顺序，必须是 `background/ion/nucleotide_like/peptide_like/small_molecule/sugar`。 |
| `available_splits` | 已原子物化的划分名称列表。 |
| `deferred_splits` | 尚未物化的划分名称列表。 |
| `splits` | 每个已物化划分到 `candidates` 和 `identities` JSONL 文件名的映射。 |
| `excluded_pdb` | 整 PDB 排除记录 JSONL 的文件名。 |
| `split_pdb_counts` | 每个划分的 `discovered/included/excluded` PDB 数。 |
| `summary` | 下表定义的逐划分计数与整 PDB 排除原因计数。 |

`summary.splits.<split>` 的字段如下；`summary.excluded_pdb_reason_counts` 是整 PDB 排除原因到数量的映射。

| 字段 | 类型与语义 |
| --- | --- |
| `candidate_count` | 当前划分的候选总数。 |
| `class_counts` | 六分类名称到候选数的映射，只含实际出现的类别。 |
| `identity_count` | 非空联合身份总数。 |
| `unique_object_key_count` | 联合身份涉及的不同 LigandObject 模板键数量。 |
| `same_pdb_object_key_multi_prepared_smiles_group_count` | 同一 PDB 内被多个非空顶层 SMILES 拆分的 object_key 组数。 |
| `matched_foreground_without_prepared_smiles_candidate_count` | 命中五类前景但顶层 SMILES 不可用的候选数，包含下列两种原因。 |
| `matched_foreground_explicit_null_candidate_count` | 命中 occurrence 的 PreparedLigand 记录存在且顶层 smiles 明确为 `null` 的候选数。 |
| `matched_foreground_missing_prepared_file_candidate_count` | 当前 PDB 的 `prepared_smiles.jsonl` 不存在，因而无法建立联合身份的候选数。 |
| `language_available_identity_count` | 每个语言家族下至少有一个有效语言归档的身份数；不要求 LigandObject 图可用。 |
| `language_multi_model_smiles_identity_count` | 每个语言家族下有效归档含多个 `model_smiles` 诊断字符串的联合身份数。 |
| `seen_positive_identity_count` | 每个语言家族下曾作为 train 可监督正目标的联合身份数。 |

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
| `matched_prepared_smiles` | 命中 occurrence 的顶层 PreparedLigand.smiles；显式 `null`、来源文件缺失或背景不适用时为 `null`。 |
| `matched_prepared_smiles_status` | `available/explicit_null/missing_file/not_applicable`；分别表示非空字符串、明确空值、PreparedLigand 文件缺失和背景不适用。 |
| `candidate_coverage` | 交集占完整候选组件的比例；未命中时为 0。 |
| `occurrence_coverage` | 交集占真实 occurrence 的比例；未命中时为 0。 |
| `overlap_score` | 两侧覆盖率几何平均；未命中时为 0。 |
| `language_target_available` | 按语言家族记录“身份图和所选语言表示同时可用”的布尔值。 |
| `identity_seen_in_train_by_language` | 按语言家族记录前景正身份是否曾作为 train 可监督目标；背景或没有前景身份时为 `null`。 |

每个 `<split>_identities.jsonl` 对应一个 PDB 内 `(object_key,prepared_smiles)` 五类前景身份。`class_name/class_index` 是联合身份属性；同一联合键在全部已物化划分中只能映射到一个类别。字段如下：

| 字段 | 类型与语义 |
| --- | --- |
| `split` | 身份所属划分。 |
| `pdb_id` | 小写 PDB 标识。 |
| `class_name` | 五类非背景类别字符串。 |
| `class_index` | 六分类中的整数下标，取值为 1–5。 |
| `object_key` | 联合身份中的 LigandObject 模板键；相同 SMILES 但不同模板键不会合并。 |
| `prepared_smiles` | 联合身份中的非空顶层 PreparedLigand.smiles；逐字采用公开模型接口输入，不使用 `model_smiles`。 |
| `occurrence_candidate_ids` | 聚合到该身份的升序 occurrence 编号列表。 |
| `graph_available` | 对应配体图归档是否存在。 |
| `graph_relative_path` | 相对 `data_root` 的配体图归档路径。 |
| `language` | `molformer` 与 `smi_ted_289m` 两个语言家族的选择证据。 |
| `seen_in_train_by_language` | 按语言家族记录该 `(object_key,prepared_smiles)` 是否在 train 有可监督正目标。 |

`language.<family>` 的 `available` 表示是否至少有一个 object_key 和 `prepared_smiles` 都与联合身份相等的有效归档；`selected_candidate_id/selected_relative_path` 在不可用时为 `null`；`valid_candidate_ids` 保存全部有效 occurrence；`failed_occurrences` 逐项保存缺文件或归档无效原因；`distinct_model_smiles` 只保存 tokenizer 输入诊断，不参与身份定义。`excluded_pdb.jsonl` 至少保存 `split/pdb_id/reason`；`reason=missing_required_product` 时另含 `missing_paths`。显式 `smiles=null` 或整个 PreparedLigand 文件缺失都保留 PDB、候选、六分类与区域监督，并令相应身份目标不可用；已有 PreparedLigand 文件缺记录或字段损坏时失败关闭。

校准入口分别记录本次 YAML 与训练 release 中原 YAML 的路径和 SHA-256，但不要求两个 YAML 字节相同：训练完成后可以显式调整 validation 工作进程数、预取量和单工作进程缓存等 I/O 资源。模型、训练超参数和 `source_scientific_data` 表列出的七个科学数据字段仍必须逐项相同；因此 I/O 调整不会被误报成科学方案变化，也不能借 I/O 调整更换监督清单或候选视野。

支持类 macro-F1 只平均真实候选数大于零的类别；固定六类 macro-F1 把零支持类 F1 记为 0 后仍平均 `background/ion/nucleotide_like/peptide_like/small_molecule/sugar` 六类。兼容字段 `evaluation.metrics.six_class.macro_f1` 等于 `macro_f1_supported_classes`，validation 选模继续使用该兼容口径。

一次评估把结果 JSON 写到调用方指定的 Matcher 业务根子路径；`data.output_root/calibration/` 保存 claim JSON 和空互斥锁文件。结果 JSON 顶层字段如下：

| 字段 | 类型与语义 |
| --- | --- |
| `schema_version` | 整数；结果与 claim 共用且联动提升的格式版本，当前为 2。 |
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
| `local_crop_strategy` | checkpoint 训练时冻结的 `fixed_center` 或 `v_centroid` 候选 `48³` 裁剪策略。 |
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
| `schema_version` | 与结果 JSON 共用的整数格式版本，当前为 2。 |
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

组合指纹先对 manifest 中来源根、监督参数、身份键字段、PreparedLigand.smiles 规则、联合身份类别一致性规则、类别顺序、语言/seen 规则、calibration 文件名、PDB 计数和 calibration 摘要组成的规范 JSON 计算 SHA-256；创建时间和 train/validation 摘要不参与。再把该摘要、calibration 候选 JSONL 摘要和身份 JSONL 摘要按 `semantic + "\n" + candidates + "\n" + identities + "\n"` 的 ASCII 字节再次计算 SHA-256。SHA-256 在此只用于文件内容留证和 claim 定位，不参与样本、监督或模型计算。

105 通道缓存写入 `cache/voxel_input_v2/<strategy>/<split>/<pdb>/candidate_<index>_start_<z>_<y>_<x>.npy`。每个文件是有限且无缺失值的 float16 `(C,Z,Y,X)=(105,48,48,48)`。通道 0–55 继承完整实验/模拟密度的数值语义，并按 `operation → normalization → postprocess` 的嵌套顺序排列：`operation` 依次为 `exp/sim/diff/posdiff`，`normalization` 依次为 `nonorm/clipnorm`，`postprocess` 依次为 `nopost/gauss1/gauss2/DoG1/DoG2/smooth1/smooth2`。通道 56–104 原样保持 `receptor_tokens.npz:feat[:,0:49]` 的第二轴顺序；受体世界坐标换算后向下取整得到所属体素，同一体素内的特征求和。这些通道继承来源特征的数值语义。两组通道均先在完整 `80³` 上构造，再裁剪候选级 `48³`。`<strategy>` 为 `fixed_center` 或 `v_centroid`；文件名还保存实际 ZYX 起点，策略或起点不同均不复用数组。

缺失体素缓存首次构造时，入口先要求 `exp.npz:grid` 与 `sim.npz:grid` 共享 `(1,Z,Y,X)` 形状、XYZ 角点原点和 XYZ 体素尺寸，再核对全部 F1 候选的 `(C,3)` `box_start_zyx`、`box_origin_world` 和 `voxel_size_world`。F1 体素尺寸必须与完整密度图逐元素相等，候选框原点必须满足 `density_origin_xyz + box_start_zyx[:,[2,1,0]] * density_voxel_size_xyz`；float32 重算允许的最大绝对误差为 `5e-5 Å`。任一检查失败都发生在 NPY 发布前。结合 Dataset 已有的 F1 与 probability-map 几何校验，这条链防止 105 通道、V/PP 和区域目标在形状合法时仍发生静默空间错位。缓存命中后只读取已验证并原子发布的 NPY，不重新打开完整密度图。

PP 缓存写入 `cache/pp_input_v2/<strategy>/top_<k>/<split>/<pdb>.npz`，并与 `F1_centered.npz` 的候选编号逐项对齐。令 `C` 为该 PDB 的候选数，`K=min(top_k,48³)`：

- `schema_version` 是 int64 标量，当前必须为 2。
- `top_k` 是 int64 标量，必须与路径中的 `<k>` 一致。
- `candidate_indices` 是 int64 `(C,)`，必须为从 0 开始的连续候选编号。
- `tokens` 是有限且无缺失值的 float32 `(C,K,4)`；最后一轴依次为无量纲的 `probability,x,y,z`，其中概率是 Stage1 原始值，XYZ 按局部空间物理半边长归一化。
- `index_zyx` 是 int16 `(C,K,3)`；`index_zyx[c,k]` 与 `tokens[c,k]` 对齐，最后一轴是候选实际裁剪 `48³` 内逐轴位于 `[0,48)` 的 ZYX 索引。
- `crop_start_zyx` 是 int16 `(C,3)`；`crop_start_zyx[c]` 与 `candidate_indices[c]` 对齐，保存该候选在 F1 `80³` 内逐轴位于 `[0,32]` 的实际裁剪起点。
- `probability_origin` 是有限且无缺失值的 float32 `(3,)`，表示完整 Stage1 概率图的 XYZ 角点原点，单位 Å。
- `probability_voxel_size` 是有限、无缺失且逐轴为正的 float32 `(3,)`，表示完整 Stage1 概率图的 XYZ 体素尺寸，单位 Å/voxel。

PP cache schema v2 使用 NumPy `argpartition` 决定入选子集；第 K 名概率并列时不另行改写选择语义。改变并列规则、裁剪公式或 PP 坐标语义必须提升 `PP_CACHE_SCHEMA_VERSION`，防止复用语义不同的缓存。缓存只消除重复解压和重复构造，不改变训练样本与监督。
