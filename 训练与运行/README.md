# AdaLigand 训练与运行入口

本目录提供一套可以随项目复制的 Slurm 任务提交系统。读者只需要准备一个 Bash 任务脚本，就可以申请计算资源、冻结本次执行所见的完整项目、保留启动证据，并按需通过锁文件控制开始时间、重复执行和退出。

“任务根目录”指需要冻结并执行的项目目录。默认任务根目录是本目录的上一层，即 AdaLigand 项目根目录；脚本内部没有写死 `AdaLigand` 或 `Pocket_Plus`。

## 1. 直接启动正式任务

先把任务脚本放入 `训练与运行/sh/`，例如 `训练与运行/sh/prepare_labels.sh`，然后在服务器的 AdaLigand 项目根目录运行：

```bash
bash 训练与运行/submit_task.sh \
  --sh prepare_labels.sh \
  --resource cpu \
  --cpus 16
```

GPU 示例：

```bash
bash 训练与运行/submit_task.sh \
  --sh 训练与运行/sh/train_stage1.sh \
  --resource h100 \
  --gpus 2 \
  --cpus 48
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
$HOME/Feedback/AdaLigand/
├── allocations/<job-id>/
│   ├── out
│   ├── err
│   ├── run_cmd_<job-id>.sh
│   ├── after_lock_<job-id>
│   └── kill_lock_<job-id>
├── launches/<job-id>/<launch-id>/
│   ├── launch.json
│   └── run_cmd.sh
└── releases/AdaLigand_<内容哈希>/
    ├── manifest.json
    └── AdaLigand/
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

## 13. 与既有提交系统的关系

AdaLigand 原有的 `与服务器交互/sbatch/` 保存旧任务脚本和旧提交方式。新正式任务优先使用本目录；正在运行的旧作业继续使用它们启动时已经冻结的代码，不应为了迁移而修改或重启。

迁移不会自动改写旧脚本，也不会删除 CPC、数据处理或推理任务的既有能力。需要复用旧任务时，应先把任务主体整理为 `训练与运行/sh/` 中可冷读的项目入口，再通过本目录提交。
