# Matcher 单候选闭集身份分类重写执行记录

## 记录边界

本轮以 Matcher 提交 `6220ece3595e48dae9ee1ef749146efa1048305c` 为唯一实现基点，保留 Git 历史、通用 Slurm/release/launch/四锁、安全同步工具和旧文档证据，删除上轮活动 Dataset/Loader/模型/监督与 mode1 入口。旧科学实现不再映射到活动 Python 入口。

当前规划来源是 `C:/Users/15919/Desktop/AdaLigand/talk/talking/Matcher_单候选闭集配体身份分类新方案.md` 及本轮对话中已确认的修订。该规划会在代码、测试和首轮运行入口稳定后按实现回填，不用本记录反向篡改旧实验结果。

## 已冻结的科学契约

- 候选六分类顺序是 `background/ion/nucleotide_like/peptide_like/small_molecule/sugar`，所有 Find_0 F1 候选都参与，不使用配体语言表示或配体图。
- 候选全局组件与 occurrence 的双向覆盖率都至少为 `0.30` 才命中；多个命中按两侧覆盖率几何平均最大者选择，并使用稳定并列规则。监督使用完整全局候选，不使用裁剪后 V。
- U-Net 输入是候选中心 `48³` 的 105 通道：56 个 Find_0 密度通道，加 49 个把完整受体原子世界坐标换算后向下取整到所属体素、并对同体素特征求和得到的通道。A/P 保留 `80³` 完整上下文且不受 U-Net 调制；V/PP 截断到中心 `48³` 并只有这两个模态读取 U-Net 特征增量。
- 候选聚合有两种：弱聚合对 A/P/V/PP 分别做 mean/max；强聚合用候选摘要分别查询四种未压缩 token。强聚合 PP 分支的输出层权重与偏置初始为严格零，但输出层自身在首次反向可获得梯度。
- 完整 Matcher 以 native 冻结语言表示为基线；native 分别查询自身未池化配体原子 token 和同一 PDB 内排除自身的其他 native 身份。native/graph/inventory 三条候选—身份打分保持独立，graph/inventory 出口门初始为严格零；单身份 inventory 上下文严格为零。
- `object_key` 在每个 PDB 内作为身份键。同一键的多个 occurrence 可由不同 SMILES 产生不同语言表示；首轮分别对 MoLFormer 和 SMI-TED Light 289M 从有效 occurrence 中按清单种子固定选择一个并记录证据。
- SMI-TED 某个 occurrence 缺失只移除该表示，不排除整个 PDB。真实小分子若没有当前语言模型的可用目标，保留六分类且使用 `identity_target=-1`，不进入身份损失和身份指标。
- `oracle/detached/joint` 都只以真实 small-molecule 且当前身份目标有效的候选作监督。它们的身份损失分母是同一 global batch 中的真实有效候选数，不是 `p_small` 权重和；`detached` 不把身份梯度传给 `p_small`，`joint` 传递该梯度。
- 候选不按 PDB 分批。每个 epoch 在全候选轴上使用 `seed+epoch` 洗牌，先截断到 32 的整倍，再切成 A100 的 `8×4` 或 A800 的 `16×2`，因此两种硬件每个 epoch 使用同一候选序列。
- train/validation/calibration 按 PDB 互斥。训练和模型选择只构造 train 与 validation Dataset；calibration 虽可在冻结清单中物化，但只能在选定最终方案后做一次保留评估。

## 实现历史

| 提交 | 作用 |
|---|---|
| `0f2967c` | 增加已排队实验的通用训练入口占位 |
| `43ac06b` | 盘点重写边界和保留基础设施 |
| `5825760` | 删除被新方案替代的活动 Matcher 科学实现 |
| `8f6e893` | 重建单候选数据清单、Dataset、collate 与体素缓存 |
| `7f8580b` | 使服务器任务激活 Matcher 训练环境 |
| `2fcea8f` | 改为全候选洗牌，并使用完整受体原子修复 binding-area 监督 |
| `8dc51fd` | 实现 RAUNet、两种候选聚合、完整 Matcher 和三种身份门控损失 |
| `a8fd3ef` | 覆盖真实数据中常见的空 A/P/V/PP 模态 |
| `a908bb5` | 修复空监督 NaN、RAUNet recycle 死参数、单身份退化、重复图映射与配体侧结构偏差 |
| `150f575` | 收紧模型张量、掩码、坐标、dtype 和 Python 类型契约 |
| `c03a693` | 完成模型与监督阶段的严格表达修订和最终门禁 |
| `aa19aa9` | 将 seen/unseen 冻结为语言模型特异的训练正目标契约，并收紧配体图 residue_id 范围 |
| `aec0388` | 增加正式训练、评估、checkpoint、严格 JSON、七项 YAML 与边界回归测试 |
| `b756ab6` | 接通 v2 清单与首轮实验脚本，并排除安全同步中的本地静态检查缓存 |
| `85a5fd0` | 在每个阶段首个真实 global optimizer step 后记录显存、梯度与耗时验收事件 |
| `ab3867e` | 消除 BRANCHED 独立 CCD 参考坐标系产生的跨组件伪几何，并记录 v2 清单审计 |

## 阶段审查与修复证据

数据科学审查在真实 train/validation/calibration 产物中发现，旧的 binding-area 目标从筛选后 A token 散射，会漏掉局部结合原子的约 60%--75%。修复后目标独立使用完整 `receptor_tokens.coords` 和 `atom_labels.binding_atom`，把世界 XYZ 按候选几何换算后向下取整到所属体素，再散射到中心 `48³`。实际纳入 PDB 的坐标/标签长度全部对齐：train 493、validation 183、calibration 86 个 PDB 都为零异常。

数据审查还确认空模态是数据几何而非加载错误：A 为空的候选约为 train `6529/14312`、validation `3334/6495`、calibration `1815/3363`；V 裁剪后为空也很常见。calibration `7z2c/candidate_1` 是已知少量正例中的一个：完整 G2P 候选在中心 `48³` 之外。这是已确认的 V 硬截断边界，暂不改变数据或监督。

区域辅助目标大量为空：ligand-area 空目标为 train `7007/14312`、validation `3704/6495`、calibration `2025/3363`；binding-area 对应计数为 `6997/3691/2024`。损失对空目标保留负体素 BCE 和软 Dice，指标保留这些真负样本，不只按含正体素的候选归一。

105 通道在真实候选上与 Pocket Plus Find_0 参考构造逐值相等：形状 `(56,80,80,80)`、最大绝对差 `0`、平均绝对差 `0`。最终缓存为 `(105,48,48,48)` float16；float16 量化误差仍需在首轮预计算摘要中报告。

模型科学审查发现并修复四类测试之外问题：没有有效身份候选时补零分数求和乘零可产生 NaN；RAUNet 原 recycle 投影是永久无梯度的死参数且回收张量与 Pocket Plus 参考 `c0` 不一致；单身份 inventory 在零注意力增量后再过 LayerNorm，不是严格零分支；同一身份对应两张配体图时布尔 occupancy 校验会静默通过。修复均已加入回归测试。

## 首轮 Slurm 任务

七个 GPU 任务都使用 `--pre_hold --after_hold`，每个任务 16 个 CPU 核，不配置 `--time`，Slurm `TimeLimit=UNLIMITED`。所有任务在删除 `pre_lock` 前只占位排队/资源，不执行训练；除非用户再次明确允许，任务结束后不删除 `after_lock`。

```bash
bash 训练与运行/submit_task.sh --sh 训练与运行/sh/train_experiment.sh --resource a100 --gpus 1 --cpus 16 --partition a100 --qos a100g4 --pre_hold --after_hold --job-name matcher-pretrain-pair -- --plan configs/experiments/initial/candidate_pretrain_pair.yaml
bash 训练与运行/submit_task.sh --sh 训练与运行/sh/train_experiment.sh --resource a800 --gpus 1 --cpus 16 --partition nvlink --qos cpu96 --pre_hold --after_hold --job-name matcher-smi-oracle -- --plan configs/experiments/initial/full_scratch_smi_ted_light_289m_oracle.yaml
bash 训练与运行/submit_task.sh --sh 训练与运行/sh/train_experiment.sh --resource a800 --gpus 1 --cpus 16 --partition nvlink --qos cpu96 --pre_hold --after_hold --job-name matcher-smi-detached -- --plan configs/experiments/initial/full_scratch_smi_ted_light_289m_detached.yaml
bash 训练与运行/submit_task.sh --sh 训练与运行/sh/train_experiment.sh --resource a800 --gpus 1 --cpus 16 --partition nvlink --qos cpu96 --pre_hold --after_hold --job-name matcher-smi-joint -- --plan configs/experiments/initial/full_scratch_smi_ted_light_289m_joint.yaml
bash 训练与运行/submit_task.sh --sh 训练与运行/sh/train_experiment.sh --resource a100 --gpus 1 --cpus 16 --partition a100 --qos a100g4 --pre_hold --after_hold --job-name matcher-molf-oracle -- --plan configs/experiments/initial/full_scratch_molformer_oracle.yaml
bash 训练与运行/submit_task.sh --sh 训练与运行/sh/train_experiment.sh --resource a100 --gpus 1 --cpus 16 --partition a100 --qos a100g4 --pre_hold --after_hold --job-name matcher-molf-detached -- --plan configs/experiments/initial/full_scratch_molformer_detached.yaml
bash 训练与运行/submit_task.sh --sh 训练与运行/sh/train_experiment.sh --resource a100 --gpus 1 --cpus 16 --partition a100 --qos a100g4 --pre_hold --after_hold --job-name matcher-molf-joint -- --plan configs/experiments/initial/full_scratch_molformer_joint.yaml
```

| Job ID | 任务 | 硬件 | 顺序/损失 | 2026-08-13 状态 |
|---:|---|---|---|---|
| `340893` | `matcher-pretrain-pair` | 1×A100 | mean/max 六分类→candidate-conditioned cross-attention 六分类 | PENDING/Priority |
| `340894` | `matcher-smi-oracle` | 1×A800 | SMI-TED Light 289M/oracle | PENDING/Priority |
| `340895` | `matcher-smi-detached` | 1×A800 | SMI-TED Light 289M/detached | PENDING/Priority |
| `340896` | `matcher-smi-joint` | 1×A800 | SMI-TED Light 289M/joint | CANCELLED before allocation；用户要求暂时跳过 |
| `340897` | `matcher-molf-oracle` | 1×A100 | MoLFormer/oracle | PENDING/Priority |
| `340898` | `matcher-molf-detached` | 1×A100 | MoLFormer/detached | PENDING/Priority |
| `340899` | `matcher-molf-joint` | 1×A100 | MoLFormer/joint | PENDING/Priority |

2026-08-13 06:47（Asia/Shanghai）只读查询 `squeue --start` 时，调度器给出的暂定开始时间是：四个 A100 任务均为 `2026-08-16 06:32:11`，三个 A800 任务均为 `2026-08-18 14:53:47`。这些时间是 Slurm 根据当前优先级和已知任务计算的预测，不是资源承诺；七个任务仍为 `PENDING (Priority)`，因此尚未创建可操作的 `pre_lock`。

同一时点核对了用户提示的其它 A100 专用分区。`a100_2` 只包含 `gnode02`、`gnode04` 和不提供 GPU 的 `master`；其中两台 GPU 节点已经包含在当前 `a100` 分区的 `gnode01–08` 中。把单卡任务从 `a100` 扩展到 `a100,a100_2` 不会增加可用 GPU 节点，反而不能改变同一节点上的资源竞争，因此保留当前更宽的 `a100` 分区。账户允许 `a100g1/a100g2/a100g4/a100g6/cpu96/nvlinkg8`；当前四个 A100 任务使用 `a100g4`，与四张 A100 的并行上限一致，三个 A800 任务已经由调度器正常接受。没有取消、重提或修改任何任务。

## 资源、输出与监控

正式源码同步目标是 `/home/penghongen/My_Project/Matcher`。业务产物只能位于 `/storage/penghongen/tmp/Matcher/`；release、launch、Slurm 日志和四锁位于 `/home/penghongen/Feedback/Matcher/`，不属于业务产物根的限制对象。

候选不按 PDB 排列。初次提交时 DataLoader 使用 16 个工作进程、`prefetch_factor=2`；A100 每个工作进程最多使用 16 GiB LRU，A800 每个工作进程最多使用 32 GiB LRU。这是每个工作进程的上限，不是整个训练进程的总预算；LRU 不预分配内存。I/O 审计后的新 release 已改为 train/validation 各 4 个非持久工作进程，完整原因、对照和不变边界见本记录后文；本段保留最初提交契约，不表示当前 release 仍固定使用 16。

任务稳定训练后不设 heartbeat。有明确时间点或状态变化时以 30 分钟间隔检查；预计长时间无新事件时可改为 1 小时。没有状态变化不追加重复日志。

## 训练、测评与首轮 CPU 预计算

- 训练入口实现完整 global batch 32 的严格分母：A100 使用 `8 × 4`，A800 使用 `16 × 2`；候选在整个划分上确定性随机打乱，不按 PDB 分组。
- validation 每轮完整运行一次；严格改善才更新 `BEST.pt`，每轮更新 `LAST.pt`；连续两轮没有改善时停止，不设置墙钟或 epoch 上限。
- 评估同时记录六分类、忽略六分类的固有身份 Top-K、仅六分类判对的小分子条件身份 Top-K、端到端身份精确率/召回率/F1、seen/unseen 和两个区域微平均 Dice。
- MoLFormer 与 SMI-TED Light 289M 的 seen/unseen 分别按“训练划分中实际可监督的小分子正目标”冻结，清单 schema 从 v1 升为 v2；新训练入口写入 `manifests/closed_set_v2/`，不会误用 CPU 预计算阶段生成的 v1 清单。体素语义没有改变，仍复用独立版本的 `cache/voxel_input_v1/`。
- 数值边界采用 fail-fast：模型输出、损失、反向损失或裁剪前梯度含 NaN/Inf 时立即停止，不允许保存新的最佳模型。
- `resolved.json` 明确记录 AdamW、ReduceLROnPlateau、GradScaler、随机种子、cuDNN 与 float32 矩阵乘法语义；严格 JSON 落盘把零支持指标的 NaN 转成 `null`。
- 第一次 CPU 提交在形成 Slurm Job 前失败：远端 `训练与运行/submit_task.sh` 的 CRLF 使 `set -o pipefail` 无法解析。只在 `/home/penghongen/My_Project/Matcher` 内把 shell 脚本转换为 LF 后，按下面的同一命令成功提交。

```bash
cd /home/penghongen/My_Project/Matcher
bash 训练与运行/submit_task.sh \
  --sh 训练与运行/sh/precompute_data.sh \
  --resource cpu \
  --cpus 16 \
  --job-name matcher-precompute \
  -- \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --stage1-root /storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477/artifacts/Find_0 \
  --output-root /storage/penghongen/tmp/Matcher \
  --coverage-threshold 0.30 \
  --selection-seed 3407 \
  --splits train validation \
  --workers 16
```

- Job `340926` 使用 16 核 CPU，在 `cpu/cpu96` 上运行 `00:57:13` 后成功完成；allocation 自动释放，不涉及七个 GPU 任务的 `after_hold`。
- 任务生成 `20,807` 个 `(105,48,48,48)` float16 NPY，等于 train `14,312` 加 validation `6,495` 个候选，全部位于 `/storage/penghongen/tmp/Matcher/cache/voxel_input_v1/`。每个 NPY 旁的同名 `.lock` 是持久并发互斥文件，不是失败样本。
- GPU Job `340893–340899` 此时仍全部为 `PENDING (Priority)`，没有重复提交，`pre_hold=1` 与 `after_hold=1` 保持不变。
- 本地训练与测评科学审查通过；随后补齐严格配置说明、checkpoint schema、路径穿越、严格 JSON、阶段顺序与批次轴错位等边界测试。最终冻结快照由独立科学审查与严格表达审查分别判定为通过：Matcher 全部 33 项测试通过，mypy 检查 32 个源文件零问题，compileall、7 个 shell 的 `bash -n`、七个 YAML 正式解析和 `git diff --check` 均通过。
- 安全同步到 `/home/penghongen/My_Project/Matcher` 后逐文件核对正式训练入口、指标、配置、JSON、首轮 YAML 和清单脚本的 SHA-256。同步器新增 `.mypy_cache/` 排除项，并在确认绝对路径后删除本次误传的远端 `/home/penghongen/My_Project/Matcher/.mypy_cache/`；删除对象仅为 18 MiB 静态类型检查缓存，可由本地检查重新生成。

v2 清单使用下面的完整命令单独提交，避免重新遍历已经生成的 20,807 个体素缓存：

```bash
cd /home/penghongen/My_Project/Matcher
bash 训练与运行/submit_task.sh \
  --sh 训练与运行/sh/build_manifest.sh \
  --resource cpu \
  --cpus 16 \
  --job-name matcher-manifest-v2 \
  -- \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --stage1-root /storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477/artifacts/Find_0 \
  --output-root /storage/penghongen/tmp/Matcher \
  --coverage-threshold 0.30 \
  --selection-seed 3407
```

- Job `340942` 使用 `cpu/cpu96`、16 核 CPU，运行 `00:11:20` 后以退出码 0 完成。该任务没有 `pre_hold` 或 `after_hold`，CPU allocation 已自动释放。

## v2 清单与只读数据审计

`closed_set_v2` 重新物化的候选数严格等于体素缓存口径：train `14,312`、validation `6,495`、calibration `3,363`。三个划分的 PDB 交集均为 0；发现/纳入/排除 PDB 分别是 train `550/493/57`、validation `200/183/17`、calibration `100/86/14`。全部 88 个排除项都是用户允许的 `blob_exceed`，没有 `missing_required_product` 被静默纳入排除。

| 划分 | background | ion | nucleotide_like | peptide_like | small_molecule | sugar |
|---|---:|---:|---:|---:|---:|---:|
| train | 10,120 | 1,371 | 17 | 9 | 1,561 | 1,234 |
| validation | 5,023 | 576 | 3 | 8 | 551 | 334 |
| calibration | 2,489 | 307 | 3 | 0 | 320 | 244 |

validation 含全部六类，因此首轮宏 F1 不会因缺类改变分母。calibration 不含 `peptide_like`；该事实不影响训练或模型选择，但最终一次保留评估必须同时报告固定六类口径与真实支持数，不能把缺类解释为模型能力。

清单含 train/validation/calibration 身份记录 `724/297/111`。MoLFormer 三个划分全部有表示；SMI-TED Light 289M 在 validation 与 calibration 各缺 2 个身份表示，缺失只影响相应 occurrence/身份监督，不排除 PDB。两种语言模型的 `language_multi_smiles_identity_count` 都为 0，即当前纳入数据没有出现同一 `object_key` 的多个有效模型 SMILES；实现仍保留确定性固定选择契约。按训练可监督小分子正目标冻结的 seen 身份数，MoLFormer 与 SMI-TED 在 train/validation/calibration 都分别为 `583/178/69`。

16 个跨 train/validation 索引的确定性候选全部与正式 float16 NPY 逐元素一致。量化前后最大绝对误差为 `0.0078086853`，平均绝对误差为 `2.0170709e-05`，均方根误差为 `8.5542191e-05`；首轮接受该磁盘量化，不改变体素 schema。

386 个去重 LigandObject 都有图文件，`residue_id` 范围为 `[1,2]`，不超出 512 类嵌入表。审计发现 7 个由多个 Chemical Component Dictionary（CCD，化学组件字典）组件组成的 BRANCHED 图，各有一条跨 `residue_id` 原子组化学键；因为两套 CCD `ref_pos` 是独立参考坐标系，旧距离落在 `5.3021–70.4320 Å`，中位数 `12.8629 Å`，属于确定的伪几何。为不让服务器排卡等待用户即时回复，先按已授权的临时决策处理：节点参考坐标按 `residue_id` 原子组分别中心化；同一组内的键保留距离；不同组之间的化学键保留拓扑与键型，但距离通道置零。该修复不改变节点/边维度，新增回归测试，并在本记录中等待用户事后复核。

## 放卡前远端源码门禁

安全同步只覆盖同名文件，不删除上一轮已经被 Git 移除的文件。首次远端全套 pytest 因此收集到旧 `test_contracts.py`、`test_decode.py`、`test_manifest.py`、`test_model.py` 和 `test_training.py`，旧测试又导入已经删除的 `MatcherSample`、`RawGraph`、`build_ground_truth_manifest` 等符号。该失败是远端源码树残留，不是新实现测试失败；若直接创建 release，旧文件也会被复制进去，因此在释放 GPU `pre_lock` 前必须闭合。

只读差集使用：

```powershell
git diff --name-status 6220ece3595e48dae9ee1ef749146efa1048305c..HEAD
rg --files matcher
& "$env:USERPROFILE/.codex/tools/Invoke-ProjectSsh.ps1" `
  -Command "cd /home/penghongen/My_Project/Matcher && find matcher -type f | sort"
```

删除操作只接受固定根 `/home/penghongen/My_Project/Matcher`，逐个把目标规范化后检查仍位于该根内。实际删除 30 个旧文件：`configs/mode1/` 两个 YAML、`训练与运行/sh/mode1/` 三个脚本/说明、旧 `matcher/README.md`、16 个旧科学模块，以及 9 个旧测试/工厂文件；同时删除 `matcher/` 下三个 `__pycache__` 和项目 `.pytest_cache`。这些删除与 `6220…HEAD` 的 Git 删除集合一一对应，未使用整目录 clean sync。缓存和旧科学文件不可从本轮工作恢复为活动入口；Git 历史仍可恢复旧文件。release、launch、四锁和 `/storage/penghongen/tmp/Matcher/` 均未触碰。

清理后在服务器正式 `Pocket_Plus_centos7_cu121_allgpu` 环境运行：

```bash
cd /home/penghongen/My_Project/Matcher
python -m pytest -q matcher/tests
python -m matcher.experiment --help
```

结果为 `34 passed in 16.99s`，实验入口帮助命令退出码为 0。正式训练、图、清单、指标、首轮 YAML 与训练 shell 的本地/远端 SHA-256 逐项一致。

安全同步器继续采用非删除式 rsync；本轮只增加两项收口行为：排除 `.mypy_cache/`，并在上传与行尾修复后使用 `rmdir` 尝试移除 `configs/mode1` 与 `训练与运行/sh/mode1`。`rmdir` 只会删除空目录，目录内若重新出现任何文件就会拒绝，因此不会把同步器扩大为隐式删除工具。

## 首个 A800 allocation 与 BF16 图消息阻断

Job `340894` 于 2026-08-13 07:35 左右取得 `gnode10` 的一张 80 GiB A800；08:28 的一小时检查首次观察到该状态。放卡前核对结果为：Slurm 仍记录 `nvlink/cpu96`、16 核 CPU、单张 A800、`pre_hold=1`、`after_hold=1`；`pre_lock_340894`、`after_lock_340894` 和动态命令均存在；动态命令只运行 `full_scratch_smi_ted_light_289m_oracle.yaml`；该 YAML 的本地与远端 SHA-256 均为 `8fa9f11da5195137bdd4d4da7247fbb588520986b6d0b2c1454a659e42478bd0`；节点 GPU 使用量为 2 MiB、利用率 0%。

第一次控制端保护命令因 PowerShell 在本地错误解释 shell 的 `$()` 表达式而在远端删除步骤前失败；随即确认 `pre_lock` 仍存在，改用不插值的字面量脚本重试。登录节点在 08:30:08 报告 `pre_lock_340894` 已删除且 `after_lock_340894` 保留。计算节点建立：

- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_f40d937354b4/Matcher`；
- launch：`/home/penghongen/Feedback/Matcher/launches/340894/train_experiment_job340894_20260813T082926_a1`。

第一次正式前向在任何 loss、反向传播或优化器更新之前失败；因此没有打印 `first_global_step_accepted`，也没有产生可能被误认成成功首步的 checkpoint。确定异常是：

```text
RuntimeError: index_add_(): self (Float) and source (BFloat16) must have the same scalar type
```

调用位置是 `GraphMessageLayer.forward`。配体元素、电荷和残基嵌入使节点残差与 `aggregated` 累加器保持 FP32；BF16 自动混合精度使两个线性消息投影返回 BF16，而 `index_add_` 不自动提升源张量类型。临时且可逆的修复决定是不改变模型结构、损失或批量：把边消息显式转换为节点累加器 dtype 后再按目标原子求和，使图消息以 FP32 累加并保留转换算子的反向梯度。

新增 CPU BF16 自动混合精度回归，在四节点双向消息层上覆盖前向 dtype、有限输出和有限节点梯度；旧实现可由该测试稳定复现同一异常。进一步运行完整 `ClosedSetMatcher` 的本地 CUDA BF16 前后向时，又在 `GraphIdentityAttention` 发现第二处同类问题：归一化权重经过归约后是 FP32，与 BF16 `value` 相乘得到 FP32 `weighted_value`，旧代码却按 `value` 创建 BF16 `context` 累加器。最终把该累加器改为继承实际乘积 dtype，并新增完整 Matcher 的 CPU BF16 前后向回归。

最新本地设备探针使用 NVIDIA GeForce RTX 4060 Laptop GPU：候选、六分类、两个区域、三种身份表示和身份分数全部有限，loss 有限，422 个实际得到梯度的参数张量全部具有有限梯度。修复后的阶段证据为模型专项 `13 passed`、全套 `36 passed`、mypy 32 个源文件零问题、compileall 与 `git diff --check` 通过。08:32 后 `try_lock_340894` 与 `after_lock_340894` 同时存在，A800 回到 2 MiB、0% 使用；在独立科学和严格表达审查对最终两处修复通过并同步新源码后，计划复用同一 allocation 进行第二次正式执行，不删除 `after_lock`，也不先缩减 RAUNet 或物理批量。

两项独立审查随后均判定两处 dtype 修复通过。本地提交 `c1fbd92` 同步后，远端 `ligand_matcher.py` SHA-256 为 `1fb991173e99047ce10613ca2e2e517e5c73a0abb6f5913efd64e2953ca80a59`；08:49:47 删除 `try_lock_340894`，仍保留 `after_lock_340894`。第二次执行建立 release `Matcher_3914f11a8153` 与 launch `train_experiment_job340894_20260813T084857_a2`，成功越过图消息和图身份注意力前向，但在第一次 global batch 的反向传播中由 PyTorch 2.4.1 抛出 `torch.utils.checkpoint.CheckpointError`：非重入式激活重计算保存的张量元数据与回算元数据不一致。仍未执行优化器更新，也未打印首步验收事件。

A800 设备探针先把四层瓶颈注意力单独放入 checkpoint；非重入带缓存、非重入禁用缓存和重入带缓存三种方式都可有限反向，说明注意力算子本身不是根因。把正式宽度、三轮 recycle 的 RAUNet 单独运行后得到决定性证据：

- 非重入 checkpoint、默认 autocast 权重缓存：复现同一个 `CheckpointError`，冻结探针峰值为 `3.3944 GiB`；
- 重入 checkpoint、默认缓存：虽然不抛异常，但 `input_projection`、`stem`、三个多尺度卷积与 `feature_projection` 共 10 个参数无梯度，不能采用；
- 非重入 checkpoint、禁用全部 autocast 权重缓存：反向通过、无缺失或非有限梯度，冻结探针峰值为 `3.4782 GiB`；
- 非重入 checkpoint、仅在每个 `no_grad` recycle 后清空 autocast 权重缓存：同样反向通过、无缺失或非有限梯度，冻结探针峰值为 `3.4818 GiB`。

根因是前三轮 recycle 的前两轮处于 `no_grad`，但和最终可训练轮共享一个外层 autocast 上下文。权重转换缓存会把前两轮不带梯度的转换留给最终轮复用，并使 checkpoint 回算重新保存不同的张量序列。最终采用更窄的修复：每个前置 `no_grad` recycle 形成 `decoded0.detach()` 后调用 `torch.clear_autocast_cache()`；不改变 BF16 算子、模型结构、激活重计算方式、损失或批量，只隔离无梯度轮与最终可训练轮的权重转换缓存。新增可选 CUDA BF16 回归，要求三轮 recycle、非重入 checkpoint 的全部参数都有有限梯度。修复后模型专项 `14 passed`，mypy 32 个源文件零问题，compileall 与 `git diff --check` 通过。

四组对照已经冻结为正式包之外的服务器诊断脚本和日志：

- 脚本：`/storage/penghongen/tmp/Matcher/diagnostics/job340894_checkpoint_20260813/probe_checkpoint_recycle.py`；
- 脚本 SHA-256：`8169584e19774df9132127be933759a4ed941c637be9c8163d7683f6c6f38fe5`；
- 日志：`/storage/penghongen/tmp/Matcher/diagnostics/job340894_checkpoint_20260813/probe_checkpoint_recycle.log`。

脚本以相同随机种子、正式九级通道、三轮 recycle、四层瓶颈注意力和物理批量 2 依次执行上述四组；完整运行命令为：

```bash
srun --jobid=340894 --overlap --ntasks=1 --cpus-per-task=2 bash -lc '
source /home/penghongen/anaconda3/etc/profile.d/conda.sh &&
conda activate Pocket_Plus_centos7_cu121_allgpu &&
export PYTHONPATH=/home/penghongen/My_Project/Matcher &&
python /storage/penghongen/tmp/Matcher/diagnostics/job340894_checkpoint_20260813/probe_checkpoint_recycle.py 2>&1 |
tee /storage/penghongen/tmp/Matcher/diagnostics/job340894_checkpoint_20260813/probe_checkpoint_recycle.log'
```

第一次冻结脚本运行遗漏 `PYTHONPATH`，在导入 `matcher` 时立即以 `ModuleNotFoundError` 退出，未进入 GPU 探针；上面的修正命令显式绑定远端正式源码并成功生成最终日志。

## 第三次 A800 执行与首个优化器更新

自动混合精度缓存修复及其契约说明由独立科学审查和严格表达审查分别判定通过；CUDA BF16 定向回归实际执行后得到 `14 passed`，mypy 检查 32 个源文件零问题，compileall 与 `git diff --check` 通过。实现提交为 `f0c39cc531abb076fd444eb2f5077e8811f2194b`。安全同步完成后，本地与远端的 `matcher/model/raunet.py`、`matcher/tests/test_model_and_loss.py` 和本执行记录 SHA-256 分别一致为 `017767d0f0b9bde417702de701d737a883918689df541b5265a6ddeac87a5a25`、`a38ed1dff51389421328fbde5cca9f43fe741d42a784b6a1bcede9ce930e10fc` 和 `6ef86090ee3f88926878e1cd4d34a6eb8ee7fc3ee1aaa413675c5a8567ab4e19`。

09:25 前再次核对 A800 空闲为 `2 MiB/0%`，并确认待删除目标是 `/home/penghongen/Feedback/Matcher/allocations/try_lock_340894`，必须保留的资源锁是 `/home/penghongen/Feedback/Matcher/allocations/340894/after_lock_340894`。只删除前者后，allocation 建立第三次不可变执行快照：

- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_966f0535491e/Matcher`；
- launch：`/home/penghongen/Feedback/Matcher/launches/340894/train_experiment_job340894_20260813T092542_a3`；
- 业务产物根：`/storage/penghongen/tmp/Matcher/experiments/full_scratch_smi_ted_light_289m_oracle/train_experiment_job340894_20260813T092542_a3/`。

第三次执行在 09:27:45 打印如下验收事件；该事件只会在两个物理批量完成有限损失检查、反向传播、有限梯度检查、参数更新与 CUDA 同步后产生：

```json
{"event":"first_global_step_accepted","stage":"full_scratch_smi_ted_light_289m_oracle","global_step":1,"global_batch_size":32,"identity_eligible_count":5,"preclip_gradient_norm":39.981868743896484,"peak_cuda_allocated_bytes":28140193792,"peak_cuda_reserved_bytes":40080769024,"global_step_compute_seconds":63.555907065980136}
```

因此 A800 正式宽度、BF16、物理批量 16、梯度累积 2 和 global batch 32 已在真实训练数据上完成首个优化器更新。首步后的只读采样显示 A800 为 `38847 MiB/100%`，训练进程和 16 个 DataLoader worker 均继续运行；`after_lock_340894` 未删除。此前两次失败保留在同一 allocation 的历史日志中，但第三次 launch 使用独立产物目录，不会把旧异常误作当前训练状态。

第三次执行随后于 09:33:48 在 DataLoader worker 6 加载后续候选时退出，`try_lock_340894` 自动重建，A800 回到 `2 MiB/0%`，`after_lock_340894` 继续保留。失败样本的 F1-centered `box_origin=[333.54345703125,25.145553588867188,279.2139892578125]`，而 probability map 的 float32 原点、体素尺寸与整数起点重算得到 `[333.5434875488281,25.145553588867188,279.2139892578125]`；最大绝对差为 `3.0517578125e-05 Å`，超过旧的 `2e-5 Å` 固定容差。该次执行至少完成了已记录的首个更新；后续精确更新数没有 checkpoint 或逐步记录可恢复，因此不作推断。

为区分 float32 重算误差与真实几何错位，使用下面的冻结探针审计 train、validation 和 calibration 全部 `24,170` 个候选：

- 脚本：`/storage/penghongen/tmp/Matcher/diagnostics/job340894_box_origin_20260813/audit_box_origin_tolerance.py`；
- 脚本 SHA-256：`7665eedafb36cf0d06b6309a158bbb0796afa0d004ab795a68ffcf5d46752287`；
- 日志：`/storage/penghongen/tmp/Matcher/diagnostics/job340894_box_origin_20260813/audit_box_origin_tolerance.log`。

```bash
srun --jobid=340894 --overlap --ntasks=1 --cpus-per-task=4 bash -lc '
source /home/penghongen/anaconda3/etc/profile.d/conda.sh &&
conda activate Pocket_Plus_centos7_cu121_allgpu &&
python /storage/penghongen/tmp/Matcher/diagnostics/job340894_box_origin_20260813/audit_box_origin_tolerance.py 2>&1 |
tee /storage/penghongen/tmp/Matcher/diagnostics/job340894_box_origin_20260813/audit_box_origin_tolerance.log'
```

审计结果是：体素尺寸逐元素不一致数为 0；超过 `2e-5 Å` 的候选为 20 个；全体最大误差仍为 `3.0517578125e-05 Å`；超过 `5e-5 Å` 的候选为 0。临时修复据此只把绝对容差从 `2e-5 Å` 调到 `5e-5 Å`，保持 `rtol=0` 与体素尺寸逐元素相等检查，并在错误中报告最大差；新增回归接受真实的一格 float32 间隔，同时拒绝约 `1e-3 Å` 的额外错位。该容差仍比一个 1.5 Å 体素小 30,000 倍，不改变 A/P/V/PP 的轴、原点或裁剪边界。

该修复由独立科学审查与严格表达审查分别判定通过；完整 Matcher 测试为 `38 passed`，服务器正式环境的数据专项为 `12 passed`。提交 `1257e7d` 安全同步后，本地与远端的 `matcher/candidate_data.py`、`matcher/tests/test_data_pipeline.py` 和本执行记录 SHA-256 分别一致为 `f00b9aadfef3148d0c8d541a52d9be849ba935c36df969c73b8fcc14faca8d2b`、`9a10ca97c8d8d78cb319a329de5646caf418475ea1c780ab769113568cf1ba26` 和 `d5467400c45815c7effb8d74303706105fa0ca778ed1edaa97e93c73f07e9b6b`。只删除 `try_lock_340894` 后建立第四次快照：release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_7e0270aab6a1/Matcher`，launch 为 `/home/penghongen/Feedback/Matcher/launches/340894/train_experiment_job340894_20260813T102050_a4`。

第四次执行在 10:22:47 完成自己的首个 global batch 32 优化器更新：计算耗时 `62.4596 s`，有效身份监督数 5，裁剪前梯度范数 `39.9936`，峰值已分配/保留显存仍为 `28140193792/40080769024` bytes。10:32 的复核已经超过第三次执行首步后约 6 分钟的确定性失败窗口；此时训练进程及 workers 已运行约 11 分钟，A800 为 `38849 MiB/100%`，日志没有新异常，`try_lock` 未重建，`after_lock` 保留。因此把第四次执行判定为稳定训练，而不是只通过首步验收；后续按 30 分钟低消耗间隔检查明确状态变化，不为无变化状态追加日志。

## 后续任务的 W&B online 追踪

用户追加要求：已经运行的任务可以继续，新的任务优先使用 W&B online，以便直接查看曲线。由此作出以下不追溯决定：Job `340894` 继续运行其不可变 release `Matcher_7e0270aab6a1`，不为追踪功能重启；尚未启动的 Job `340893`、`340895`—`340899` 以及 Job `340894` 未来可能发生的新 launch 才使用在线追踪。2026-08-13 的只读队列复核仍显示 Job `340894` 在 `gnode10` 运行，其余六个任务全部为 `PENDING (Priority)`；所有 `after_lock` 继续保留。

首轮在线位置固定为 W&B entity `pencounkdual-111`、project `AdaLigand-Matcher`、group `matcher-initial-round1`。每个 stage 建立独立 run；每个含 32 个候选的 global batch 完成一次参数更新后记录总损失和四个分项、裁剪前全模型梯度范数、图与清单分支梯度范数、两个零初始化门值、实际学习率、数据等待、GPU 计算、吞吐与峰值显存。每个完整 validation epoch 记录六分类、固有身份、以六分类判对为条件的身份、seen/unseen、端到端身份、两个区域 Dice、选模分数及其支持数或分母。逐步损失、梯度、门值中的 NaN/Inf 直接失败；只有同次记录明确提供零分母的评估 NaN 才转换为 W&B 空值。

在线运行不会在 YAML、`resolved.json`、`tracking.json` 或日志中保存 API key。认证读取服务器进程环境或现有 `~/.netrc`；服务器探针确认正式环境包含 W&B `0.17.3`、当前凭据可解析、`wandb.Api().default_entity` 为 `pencounkdual-111`。SDK 的 `wandb.init` 签名确认存在 `force` 参数；正式适配器显式使用 `mode="online"`、`force=True`、`resume="never"`，在线初始化或 run ID、HTTPS URL、横轴定义、元数据写入失败时不降级为 offline。`init_timeout_seconds=120` 只限制 W&B 在线握手等待，不限制训练、epoch 或 Slurm allocation 的运行时间。

服务器环境和身份探针使用以下命令；两个脚本只输出布尔状态、SDK 版本和 entity，不输出认证值：

```powershell
& '.\与服务器交互\other\Invoke-PasswordSsh.ps1' `
  -Command 'bash -s' `
  -InputFile '.\tmp\probe_wandb_environment.sh'

& '.\与服务器交互\other\Invoke-PasswordSsh.ps1' `
  -Command 'bash -s' `
  -InputFile '.\tmp\probe_wandb_identity.sh'

& '.\与服务器交互\other\Invoke-PasswordSsh.ps1' `
  -Command 'source /home/penghongen/anaconda3/etc/profile.d/conda.sh && conda activate Pocket_Plus_centos7_cu121_allgpu && python -c "import inspect,wandb; print(wandb.__version__); print(inspect.signature(wandb.init))"'
```

本地追踪增量的冻结门禁结果为 mypy 34 个源文件零问题、pytest `42 passed`、compileall、7 个 shell 的 `bash -n`、七个 YAML 正式解析与 `git diff --check` 通过。服务器 W&B `0.17.3` 的实际 `wandb.init` 签名同样接受上述显式参数。独立科学审查确认 global-batch 损失、A100/A800 更新等价、CUDA 同步计时、支持数、异常收口与 checkpoint 时序通过；严格表达审查确认 W&B 最小接口、十字段元数据、六条自定义横轴、七份 YAML 和 README 契约通过。实现提交为 `d9717c2`。

安全同步命令为：

```powershell
& '.\与服务器交互\sync_code.ps1'
```

同步只覆盖 `/home/penghongen/My_Project/Matcher` 的同名源码，不改变正在运行的 release。下列四个代表文件的本地与远端 SHA-256 逐项一致：

- `matcher/tracking.py`：`4ec0da1eef41acf38a9df04a804a6f7bb61bf760c28a8f27b823249448185e37`；
- `matcher/training.py`：`ccfedd01c8632f776976a96d449e310a30c78fc63f20b55427f2543f6a8afbd4`；
- `configs/experiments/initial/candidate_pretrain_pair.yaml`：`4c87430757c643bbaafed8c6acc788f7f23654faeb7f5708e4860344a2044518`；
- `configs/experiments/initial/full_scratch_molformer_oracle.yaml`：`0f7ff427b4d28a9d08e9e3ca8dd7948d19b47b572823c279d6205609f484defd`。

正式环境验证通过统一 SSH 入口把本地临时脚本 `tmp/validate_wandb_remote.sh` 作为标准输入发送，不把探针脚本混入正式源码：

```powershell
& '.\与服务器交互\other\Invoke-PasswordSsh.ps1' `
  -Command 'bash -s' `
  -InputFile '.\tmp\validate_wandb_remote.sh'
```

第一次尝试在 conda 激活前启用 `set -u`，被环境激活脚本的未定义 `ADDR2LINE` 立即拒绝；第二次先激活环境，但试图运行正式环境未安装的 mypy，同样在测试与 W&B 初始化前结束。最终脚本先以 `set -eo pipefail` 激活 `Pocket_Plus_centos7_cu121_allgpu`，再启用 `set -u`；mypy 只使用已经通过的本地门禁，服务器执行 compileall、pytest 和真实 W&B 生命周期。最终正式环境结果为 pytest `41 passed, 1 skipped`，其中一项只因登录节点没有 CUDA 而按测试设计跳过。

真实 online 探针创建 run `wimmkf3s`，先记录一个 global batch 32 的逐步点，再记录一个 validation/selection epoch 点和最终摘要，以退出码 0 完成同步。曲线地址是 `https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/wimmkf3s`；全部本地诊断产物位于 `/storage/penghongen/tmp/Matcher/diagnostics/wandb_online_20260813T034138Z/`。该探针明确命名为 `wandb_adapter_smoke`，不会与七个正式 stage 混淆。至此可以让后续新 launch 使用 W&B online；仍不删除任何待启动任务的 `pre_lock`，只有任务实际取得资源且放卡前检查通过时才启动。

## Job 340895：首个带 W&B 曲线的正式任务

2026-08-13 11:57:27（Asia/Shanghai），Job `340895` 获得 `gnode09` 的一张 A800；12:19 的检查首次发现这一状态。放卡前逐项确认：Slurm 资源为 `nvlink/cpu96`、1×A800、16 CPU，提交命令没有显式 `--time`；任务使用 `--pre_hold --after_hold`；`/home/penghongen/Feedback/Matcher/allocations/pre_lock_340895` 和 allocation 内的 `after_lock_340895` 均存在；动态命令只运行 `configs/experiments/initial/full_scratch_smi_ted_light_289m_detached.yaml`；该配置的本地与远端 SHA-256 都是 `33578f58a10300781dfcd0eabc20272c2a11a7c17d38739b65974a6fc07ca466`。配置保持 A800 物理批量 16、梯度累积 2、global batch 32、16 个 worker 和每 worker 32 GiB 缓存上限。

第一次放锁命令把远端 `$(sha256sum ...)` 放在 PowerShell 可插值字符串中，本地 PowerShell 在连接前误解释并拒绝命令；随后确认 `pre_lock_340895` 仍存在，因此没有发生远端删除。正式放锁改用本地临时审核脚本 `tmp/release_prelock_340895.sh`，脚本同时断言 pre-lock、after-lock 和远端配置哈希，仅在三项均正确时删除指定 pre-lock：

```powershell
& '.\与服务器交互\other\Invoke-PasswordSsh.ps1' `
  -Command 'bash -s' `
  -InputFile '.\tmp\release_prelock_340895.sh'
```

12:21:41 脚本成功删除且只删除 `/home/penghongen/Feedback/Matcher/allocations/pre_lock_340895`，allocation 随即冻结：

- release：`/home/penghongen/Feedback/Matcher/releases/Matcher_16b0873c6884/Matcher`；
- launch：`/home/penghongen/Feedback/Matcher/launches/340895/train_experiment_job340895_20260813T121932_a1`；
- 业务产物根：`/storage/penghongen/tmp/Matcher/experiments/full_scratch_smi_ted_light_289m_detached/train_experiment_job340895_20260813T121932_a1/`；
- W&B run ID：`pki0o1zr`；
- W&B 曲线：`https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/pki0o1zr`。

W&B online 初始化成功后，任务在 12:24 左右完成首个真实 global optimizer step，并同时把逐步指标写入在线 run：

```json
{"event":"first_global_step_accepted","stage":"full_scratch_smi_ted_light_289m_detached","global_step":1,"global_batch_size":32,"identity_eligible_count":5,"preclip_gradient_norm":29.054296493530273,"peak_cuda_allocated_bytes":28140193792,"peak_cuda_reserved_bytes":40076574720,"global_step_compute_seconds":65.70703677739948}
```

该事件发生在损失与梯度有限性检查、参数更新和无条件 CUDA 同步之后，证明最新 W&B release 没有改变 A800 正式训练语义。此时 Job `340895` 仍为 RUNNING，`after_lock_340895` 保留，`try_lock_340895` 不存在。后续逐步损失、梯度、学习率、数据等待、计算吞吐、门值和显存可以由用户直接在线查看；完整 validation 指标在 epoch 结束后出现。

## W&B 网络异常与 DataLoader 进程隔离

Job `340895` 完成第一个训练 epoch 后进入首次完整 validation。W&B API 随后把 run `pki0o1zr` 显示为 `crashed`，但 2026-08-13 15:00--15:10 的只读交叉检查同时确认：Slurm 任务仍为 RUNNING；主 Python 进程、W&B 服务进程、16 个训练 worker 和 16 个 validation worker 均存活；`after_lock_340895` 存在而 `try_lock_340895` 不存在；worker 的累计读取字节仍增加。因此没有依据重启或释放该任务。

本地 W&B `debug-internal.log` 给出了状态冲突的直接原因：gnode09 从 13:22 起反复收到 `api.wandb.ai` 的 `SSLEOFError` 或 `_ssl.c` 系统错误，在线 file stream 无法续传；日志中没有 `finish`、训练异常或服务关闭记录，15:08 仍持续产生 keepalive 和状态报告。网页的 `crashed` 是在线心跳中断后的远端状态，不代表 Matcher 进程已经退出。检查命令为：

```powershell
& '.\与服务器交互\other\Invoke-PasswordSsh.ps1' `
  -Command 'bash -s' `
  -InputFile '.\tmp\inspect_wandb_340895.sh'
```

同一日志还显示 W&B 初始化早于 Linux DataLoader workers，旧 release 使用默认 `fork`；同一毫秒出现成组的 keepalive/status 请求，说明子进程继承追踪客户端状态会放大不必要的进程间通信。提交 `184e030` 因而把正式 train 与 validation DataLoader 的 `multiprocessing_context` 显式冻结为 `spawn`，并写入 `resolved.json:data.multiprocessing_context`。该改变不重启 Job `340894/340895`，只进入尚未冻结的新 release；它不改变 sampler、候选集合、物理批量、梯度累积或损失分母。

本地门禁为 pytest `42 passed`、mypy 34 个源文件零问题和 `git diff --check` 通过。安全同步仍使用：

```powershell
& '.\与服务器交互\sync_code.ps1'
```

服务器真实生命周期探针先把临时代码写入允许的诊断目录，再在正式 Conda 环境中执行；没有把探针混入 release 源码：

```powershell
& '.\与服务器交互\other\Invoke-PasswordSsh.ps1' `
  -Command 'mkdir -p /storage/penghongen/tmp/Matcher/diagnostics/wandb_spawn_20260813T151807+0800 && cat > /storage/penghongen/tmp/Matcher/diagnostics/wandb_spawn_20260813T151807+0800/probe.py' `
  -InputFile '.\tmp\probe_wandb_spawn.py'

& '.\与服务器交互\other\Invoke-PasswordSsh.ps1' `
  -Command 'source /home/penghongen/anaconda3/etc/profile.d/conda.sh && conda activate Pocket_Plus_centos7_cu121_allgpu && cd /home/penghongen/My_Project/Matcher && MATCHER_DIAGNOSTIC_ROOT=/storage/penghongen/tmp/Matcher/diagnostics/wandb_spawn_20260813T151807+0800 PYTHONPATH=. python /storage/penghongen/tmp/Matcher/diagnostics/wandb_spawn_20260813T151807+0800/probe.py'
```

探针创建 online run `tf8cphpo`，四个 `spawn` worker 的 PID 为 `71468/71471/71474/71478`，`wandb_module_inherited_count=0`；两次逐步曲线、最终摘要和退出码 0 全部同步成功。曲线地址是 `https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/tf8cphpo`，本地报告位于 `/storage/penghongen/tmp/Matcher/diagnostics/wandb_spawn_20260813T151807+0800/report.json`。

用户随后解除“候选全局随机、16 workers、单 worker 16/32 GiB”作为吞吐冻结值的约束，允许按实测调整以提高速度。已运行 release 保持不变；尚未启动任务只采用有对照证据的 I/O 调整，不凭单次 GPU 空闲现象立即改变科学批次组成。

## PP token 缓存与 15:57 队列复核

首轮 A800 的训练计算可以持续占满 GPU，但第一次 validation 长时间处于 CPU/I/O 阶段。对真实清单和源文件大小的只读统计表明，train 的完整 probability map 压缩文件约占 PDB 级输入字节的 90%；在候选全局随机、16 个 worker 各自缓存的旧路径中，同一 PDB 的 probability map 会跨 worker 重复解压。按 worker/PDB 配对做的静态估算为 train 约读取 895 GB PDB 级压缩产物，而每个 PDB 只读取一次约为 76 GB；该估算用于确定优化优先级，不冒充真实训练吞吐测量。统计报告位于 `/storage/penghongen/tmp/Matcher/diagnostics/dataloader_locality_20260813/report.json`。

因此新增版本化 PP 缓存 `cache/pp_input_v1/top_1024/<split>/<pdb>.npz`。它一次读取某个 PDB 的完整 probability map，按候选编号保存全部中心 `48³` PP token；训练命中缓存后只读取小型产物，缺失时由第一个取得 Linux 阻塞文件锁的进程构建，其余进程复用。该改变不修改 sampler、候选集合、物理批量、梯度累积、监督字段或 global batch。单 worker 缓存预算改为五个互不重叠的 LRU 分区，只决定可保留的已读数组，不预分配内存。

科学审查发现第 K 名概率并列时，如果把旧 NumPy `argpartition` 改为“完整排序后取展平索引最小者”，会改变少量 PP 的 XYZ/索引，从而破坏已经运行任务与后续损失对照的输入可比性。本轮缓存因此逐值保留旧 Dataset 的 `argpartition` 入选子集，并只对入选项按概率降序、展平索引升序排序；更改并列规则必须使用新的缓存 schema 和新一轮实验。独立科学审查用 unique、重度并列、全零和裁剪边界概率图，组合 `top_k={1,4,1024,48³,>48³}` 共 20 个探针与旧函数逐数组比较，token 与索引全部相等；真实清单 train `480 PDB/14,312 candidates`、validation `181 PDB/6,495 candidates` 的候选编号也全部从 0 连续对齐。冻结门禁为 pytest `46 passed`、mypy 35 个源文件零问题、compileall、Git Bash `bash -n` 与 `git diff --check` 通过。

正式 CPU 预计算将使用下面的完整命令，不设置 Slurm 或程序时间上限；业务产物全部位于允许的 `/storage/penghongen/tmp/Matcher/`：

```bash
bash 训练与运行/submit_task.sh \
  --sh 训练与运行/sh/precompute_data.sh \
  --resource cpu \
  --cpus 16 \
  --partition cpu \
  --qos Cpu96 \
  --job-name matcher-pp-precompute \
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

实现提交为 `c0ac5fc`。安全同步后，本地与远端 `matcher/pp_input.py` 的 SHA-256 均为 `8255d03fb76863ea5bb7adfd011b060ef30bd267ee2c4803a99b8ebe47c6166e`，`matcher/precompute.py` 均为 `5c62a481bd0e31960852ab3fca5f004316372f4240a9c6b853ad29035b54149a`；服务器正式 Conda 环境的数据专项为 `15 passed`。上述命令随后原样提交为 Job `341221`。提交器报告 `resource=cpu`、`cpus_per_task=16`、`pre_hold=0`、`after_hold=0`，未出现 `--time`。Job 在 `cpu/cpu96` 的 `cnode04` 运行 `00:01:40` 后以 ExitCode `0:0` 完成并自动释放；没有创建或删除任何 GPU 任务的 `after_lock`。

正式缓存结果为 train `480 PDB/14,312 candidates/323,431,872 bytes`，validation `181 PDB/6,495 candidates/146,709,428 bytes`，合计 `661 PDB/20,807 candidates/470,141,300 bytes`。独立只读验收逐个 PDB 检查缓存候选数与 `closed_set_v2` 清单从 0 连续对齐；再从两个划分各取排序后首、中、末 PDB 的中间候选，共 6 个真实候选，从原 `probability_map.npz` 重新构造 PP，缓存 token 和 ZYX 索引均逐值相等。由此，后续新 release 的正式 train/validation PP 热路径可以确定命中，不再读取完整 probability map。

2026-08-13 15:57:21（Asia/Shanghai）的轻量只读检查显示 Job `340894` 在 `gnode10` 已运行 `8:21:41`，Job `340895` 在 `gnode09` 已运行 `3:59:54`；两者 `after_lock` 均保留、`try_lock` 均不存在，且没有新 checkpoint。Job `340893`、`340897`、`340898`、`340899` 仍为 `PENDING (Priority)`。Job `340896` 已从 `squeue` 消失，随后用 `sacct -X` 核实为 `CANCELLED by 0`、elapsed `00:00:00`、未分配节点、未建立锁或业务产物；这不是训练异常。用户随后明确要求暂时跳过该 SMI-TED Light 289M/joint 任务，因此不提交替代 Job、不创建新锁，也不把它作为当前批次完成的前置条件；只有用户再次明确要求时才恢复。

2026-08-13 16:24:29，Job `340895` 完成 epoch 0 的完整 validation，并写出 `history.jsonl`、`LAST.pt` 和 `BEST.pt`；选择分数为 `0.5529791207038862`。训练使用 14,304 个候选，447 次参数更新，纯计算 `1263.75 s`、DataLoader 等待 `51.18 s`、总计 `1317.12 s`，吞吐 `10.8600 candidates/s`。相同模型的 validation 读取 6,495 个候选却耗时 `13343.99 s`，吞吐只有 `0.486736 candidates/s`；六分类 macro-F1 为 `0.5220903`，以六分类判对为条件的身份 top-1 为 `0.6264501`，端到端身份 F1 为 `0.5103970`。16:34 的 Slurm 统计显示 Job `340894/340895` 累计读字节约为 `2.979/3.034 TB`，MaxRSS 约为 `548/673 GB`。这组同一真实运行的训练/验证耗时、累计读取和内存证据证明旧 release 的 validation 是 I/O/内存系统瓶颈，而不是三维模型计算瓶颈；Job `340895` 随后继续 epoch 1，现有 checkpoint 与指标保留。

16:45 的 allocation 内 GPU 探针显示 Job `340894` 占用约 `38.8 GiB` 显存但 GPU 与显存带宽利用率均为 0%；它仍没有 `BEST.pt/LAST.pt`，16:34--16:41 累计读取只增加约 `0.8 GB`。同期 Job `340895` 为 GPU 100%，继续 epoch 1，因此只重启已经长期停滞的 oracle，不中断 detached。该决定使用的成熟方案不是尚未收口的参数搜索，而是已经完成全量预计算、双审和逐值验收的 PP cache schema v1；它确定消除旧 release 中完整 probability map 的跨 worker 重复解压。

先再次安全同步当前项目，再使用以下控制端命令；本地临时脚本只保存精确断言和锁操作，不进入 release，完整前因后果保留在本记录：

```powershell
& '.\与服务器交互\other\Invoke-PasswordSsh.ps1' `
  -Command 'bash -s' `
  -InputFile '.\tmp\kill_and_hold_340894.sh'

& '.\与服务器交互\other\Invoke-PasswordSsh.ps1' `
  -Command 'bash -s' `
  -InputFile '.\tmp\release_try_340894_pp.sh'
```

`kill_and_hold_340894.sh` 在写锁前断言 Job 仍 RUNNING、只有 `after_lock_340894`、动态命令仍为 oracle、旧业务目录没有 checkpoint、本地/远端四个代表文件哈希一致且 PP 文件数为 `480/181`；然后只创建 `/home/penghongen/Feedback/Matcher/allocations/340894/kill_lock_340894`。监视器于 16:49:09--16:49:19 终止旧进程组、自行删除 `kill_lock` 并建立 `try_lock_340894`，Slurm allocation 和 `after_lock_340894` 均保留。空闲探针确认 GPU 为 `2 MiB/0%` 后，第二个脚本再次核对 W&B online、PP 文件数、命令和三锁，只删除精确的 `try_lock_340894`。

第五次 oracle 执行随后冻结 release `/home/penghongen/Feedback/Matcher/releases/Matcher_67c4e42bd46a/Matcher`，launch 为 `/home/penghongen/Feedback/Matcher/launches/340894/train_experiment_job340894_20260813T165002_a5`，业务目录为 `/storage/penghongen/tmp/Matcher/experiments/full_scratch_smi_ted_light_289m_oracle/train_experiment_job340894_20260813T165002_a5/`。W&B run ID 为 `aiqbj6b5`，曲线地址为 `https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/aiqbj6b5`。新 release 在 16:53 前完成首个真实 global batch 32 参数更新：计算 `62.2150 s`、有效身份监督数 5、裁剪前梯度范数 `39.9681`、峰值已分配/保留显存 `28140193792/39564869632 bytes`；`after_lock_340894` 继续保留，`try_lock/kill_lock` 均不存在。

## I/O 系统审计与 DataLoader 生命周期收口

用户允许独立使用 CPU 诊断 I/O，并要求每项预设最长测试时间、最迟两小时收口。三个诊断 Job 的脚本、标准输出、标准错误和 JSON 只写入 `/storage/penghongen/tmp/Matcher/diagnostics/`，没有申请 GPU，也没有使用或改变任何 GPU allocation 的四锁。实际提交命令如下：

```bash
sbatch --parsable \
  --job-name=matcher-io-audit \
  --partition=cpu --qos=cpu96 \
  --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=1000G \
  --time=01:50:00 \
  --output=/storage/penghongen/tmp/Matcher/diagnostics/io_system_audit_20260813/slurm-%j.out \
  --error=/storage/penghongen/tmp/Matcher/diagnostics/io_system_audit_20260813/slurm-%j.err \
  --wrap='exec bash /storage/penghongen/tmp/Matcher/diagnostics/io_system_audit_20260813/run_cpu_audit.sh'

sbatch --parsable \
  --job-name=matcher-io-dataloader \
  --partition=cpu --qos=cpu96 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=256G \
  --time=01:20:00 \
  --output=/storage/penghongen/tmp/Matcher/diagnostics/io_dataloader_20260813/slurm-%j.out \
  --error=/storage/penghongen/tmp/Matcher/diagnostics/io_dataloader_20260813/slurm-%j.err \
  --wrap='exec bash /storage/penghongen/tmp/Matcher/diagnostics/io_dataloader_20260813/run_dataloader_audit.sh'

sbatch --parsable \
  --job-name=matcher-io-full-val \
  --partition=cpu --qos=cpu96 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=256G \
  --time=00:50:00 \
  --output=/storage/penghongen/tmp/Matcher/diagnostics/io_full_validation_20260813/slurm-%j.out \
  --error=/storage/penghongen/tmp/Matcher/diagnostics/io_full_validation_20260813/slurm-%j.err \
  --wrap='exec bash /storage/penghongen/tmp/Matcher/diagnostics/io_full_validation_20260813/run_full_validation_audit.sh'
```

Job `341231` 的四个单进程动作依次设为 20、25、25、20 分钟。原请求为 32 CPU/1000 G；确认负载不使用额外核后，在尚未分配时通过 `scontrol update` 逐步降为 1 CPU/125 G，并清除临时节点约束，最终在 cnode01 运行 `1:21`，`COMPLETED/0:0`。Job `341236` 的准备动作限 20 分钟，四个对照各限 12 分钟，最终在 cnode02 运行 `6:22`，`COMPLETED/0:0`。Job `341238` 只运行完整 6,495 候选验证集读取，动作限 45 分钟，最终在 cnode02 运行 `10:02`，`COMPLETED/0:0`。所有动作都使用 `timeout --signal=TERM --kill-after=30s`；这些上限仅属于用户要求的 CPU 诊断，不是训练或 GPU Slurm 时限。

审计确认以下热路径事实：PP schema v1 命中后不再读取完整 probability map；105 通道体素缓存命中后不读取原始实验或模拟密度图。旧 Job `340895` 的 16 个 train worker 与 16 个 validation worker 曾同时常驻，进程树 MaxRSS 约 `677.5 GB`，首次 validation 为 `13343.99 s`；epoch 1 在操作系统页缓存变热后缩短为 `289.72 s`，所以不能把旧 release 的每轮 validation 都描述为数小时。当前 HEAD 使用完整 PP 缓存、未压缩体素和 4 workers 读取完整 validation 的结果为 `568.216 s`、`11.4305 candidates/s`、进程树峰值 `13.734 GB`、累计读取 `151.785 GB`。固定 256 候选对照中，候选派生量预物化把吞吐从 `9.51` 提高到 `11.48 candidates/s`，但样本和收益不足以支持立即增加正式 schema；Deflate level 1 虽降低读取量，却把吞吐降到 `8.77 candidates/s`，因此不采用。完整证据分别位于 `io_system_audit_20260813`、`io_dataloader_20260813` 和 `io_full_validation_20260813` 三个诊断目录。

源代码复核同时纠正一项审计过程中的误读：`matcher/candidate_data.py::_CENTERED_FIELDS` 只有一个 `voxel_final`。早期终端输出的相邻分片发生重叠，曾被误读成重复字段；这不是正式实现缺陷，也没有产生删除改动。

据此，提交 `e8241dd` 将 train 与 validation 的工作进程数拆成两个必填配置，七份首轮 YAML 均显式采用 `4/4`；两个 DataLoader 使用 `spawn`、`persistent_workers=False`，训练精确消费结束后显式释放迭代器，因此 validation 前不会保留训练 worker。train 与 validation 各自使用独立 `torch.Generator`，种子为 `2 × training.seed` 与 `2 × training.seed + 1`；这避免每轮重建迭代器推进模型和 dropout 使用的全局 Torch 随机状态。`resolved.json` 保存两套 worker 数、spawn、非持久生命周期、两个生成器种子及其推进规则。候选集合、`ShuffledBatchSampler(seed+epoch)` 的全局随机顺序、监督、模型、损失、物理批量、梯度累积和 global batch 32 均未改变。

独立科学审查和严格表达审查均给出 PASS。冻结门禁为 pytest `47 passed`、mypy 35 个源文件零问题、compileall、七份 YAML 解析、Git Bash `bash -n` 与 `git diff --check` 通过。服务器登录节点的 CPU-only 生命周期测试最初因测试模拟 CUDA 可用后真实启动 `pin_memory=True`，而节点没有 NVIDIA 驱动，触发 PyTorch 2.4.1 初始化失败；提交 `a433715` 先断言生产 loader 的 `pin_memory=True`，再只对 CPU 测试对象关闭页锁定线程。科学与表达复核确认该测试仍真实覆盖 spawn、独立生成器、全局随机状态隔离和 worker 退出；服务器正式 Conda 环境随后为 `14 passed`。

## A100 放卡与在线曲线

2026-08-13 17:28 左右，Job `340893` 和 `340897` 分别取得 gnode02 与 gnode07 的一张 A100；两者均为 `a100/a100g4`、16 CPU、`TIME_LIMIT=UNLIMITED`，并由 `pre_lock` 与 `after_lock` 安全保留。完成上述实现、双审、本地门禁、服务器门禁和同步后，放卡脚本逐项断言：两个 Job 仍 RUNNING；只有各自的 pre/after 锁；动态命令分别指向 `candidate_pretrain_pair.yaml` 与 `full_scratch_molformer_oracle.yaml`；四个代表文件的本地/远端 SHA-256 一致；配置为 train/validation 4/4 workers、W&B online、A100 `8 × 4 = global 32`；PP 缓存文件数为 train 480、validation 181。脚本随后且只删除 `pre_lock_340893` 与 `pre_lock_340897`，两个 `after_lock` 均保留。

两个任务冻结同一 release `/home/penghongen/Feedback/Matcher/releases/Matcher_95a9659383c2/Matcher`：

- Job `340893` launch 为 `/home/penghongen/Feedback/Matcher/launches/340893/train_experiment_job340893_20260813T175925_a1`。第一阶段 W&B run ID 为 `mxornctd`，曲线为 `https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/mxornctd`。首个 global batch 32 于 18:01 前完成：计算 `23.8973 s`、身份监督数 0、裁剪前梯度范数 `28.1286`、峰值已分配/保留显存 `14154330112/19363004416 bytes`。
- Job `340897` launch 为 `/home/penghongen/Feedback/Matcher/launches/340897/train_experiment_job340897_20260813T175915_a1`。W&B run ID 为 `mkaiyufz`，曲线为 `https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/mkaiyufz`。首个 global batch 32 于 18:01 前完成：计算 `25.0321 s`、有效身份监督数 5、裁剪前梯度范数 `53.1809`、峰值已分配/保留显存 `14190904832/19526582272 bytes`。

18:01 的轻量统计中，两个新作业进程树 RSS 约为 `15.4/15.1 GB`，两个首步事件均发生在有限损失与梯度检查、参数更新和 CUDA 同步之后。由此，新 DataLoader 生命周期不仅通过合成测试，也已在真实 A100 正式任务中稳定启动。

同一次状态检查发现 Job `340898/340899` 也已分别在 gnode04 与 gnode02 取得一张 A100，并由 pre/after 锁保留。放卡脚本再次逐项核对 Job 为 RUNNING、动态命令、锁、四个代表文件哈希、W&B online、MolFormer 身份损失变体、`4/4` workers、A100 `8 × 4 = global 32` 与 PP 缓存文件数；随后且只删除这两个 pre-lock，after-lock 继续保留。两者同样复用 release `Matcher_95a9659383c2`：

- Job `340898` launch 为 `/home/penghongen/Feedback/Matcher/launches/340898/train_experiment_job340898_20260813T180543_a1`。W&B run ID 为 `0lam2vpk`，曲线为 `https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/0lam2vpk`。detached 首个 global batch 32 的计算时间为 `24.8004 s`、有效身份监督数 5、裁剪前梯度范数 `29.4950`、峰值已分配/保留显存 `14192313856/19457376256 bytes`。
- Job `340899` launch 为 `/home/penghongen/Feedback/Matcher/launches/340899/train_experiment_job340899_20260813T180559_a1`。W&B run ID 为 `1z2vvfxj`，曲线为 `https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/1z2vvfxj`。joint 首个 global batch 32 的计算时间为 `22.2045 s`、有效身份监督数 5、裁剪前梯度范数 `29.3475`、峰值已分配/保留显存 `14192313856/19457376256 bytes`。

截至 18:07，四个 A100 任务均已稳定开始正式训练，没有剩余 pre-lock；Job `340896` 仍按用户要求跳过。同期，SMI-TED Light 289M/oracle 已完成 epoch 1，选择分数从 `0.5607736` 改善为 `0.5727896`；旧 release 的 SMI-TED Light 289M/detached 已完成 epoch 3，选择分数从 epoch 2 的 `0.6022308` 改善为 `0.6373626`。两个 A800 任务均继续训练，after-lock 保留。

## 计划与实现差异：运行中快照

以下差异只记录截至 2026-08-13 18:13 已经发生的事实；计划书冻结文本尚未据此改写，最终收口时还要结合完整实验结果复核。

### 有利调整

- 计划书第 8 节的首轮起点是每次 DataLoader 迭代使用 16 个工作进程。I/O 审计后，尚未冻结的新 release 改为 train 与 validation 各 4 个非持久工作进程，并为两者使用独立随机数生成器；完整 validation 的 CPU 对照、真实 spawn 生命周期测试和 A100 首步共同证明该值能显著降低进程树内存，同时保持候选集合、顺序、global batch、监督和模型随机状态语义。已经运行的 Job `340895` 保持原 release，不为追溯该调整而重启。
- PP token 从每个 worker 命中概率图时临时派生，改为版本化 PDB 级预计算缓存。逐值核验确认缓存保持旧 `argpartition` 入选规则，训练命中后不再读取完整 probability map；因此这是输入等价的 I/O 实现调整，不是科学样本或特征定义变化。

### 中性或待用户最终确认

- BRANCHED 多残基配体图已按计划书中的暂定规则实现，并由独立科学审查确认不会把独立 CCD 参考坐标系误作共同三维构象；该规则仍保留为待用户最终复核的科学边界。
- 计划书仍列出 SMI-TED Light 289M/joint。原 Job `340896` 在未分配节点、未运行和未建立锁时被取消；用户明确要求当前先跳过，因此不补交替代任务，也不把缺失结果记为零分。是否在本轮最终结论前恢复，仍取决于用户后续指令。

### 尚未完成

- 候选预训练的两个阶段、五个正在运行的完整 Matcher 实验尚未全部达到早停；当前不能形成聚合方式、语言表示或身份损失的最终排序。
- calibration 仍严格隔离，尚未被任何训练、早停或选模入口读取。只有首轮方案选定后才能做一次保留评估。
- 计划书第 10 节仍以七个任务和六个完整 Matcher 结果作为原完成判据；由于 Job `340896` 被用户暂停，最终收口前需要明确采用“六个已授权运行结果”还是恢复第七个任务，不能通过改写完成判据掩盖该缺口。

### 已知有害偏差

截至本快照，没有发现仍留在活动 release 中且会改变样本、监督、global batch 或模型定义的有害偏差。若后续训练或评估出现此类证据，将在本节单独记录，而不会归入性能调整。

## 18:38 首轮 epoch 进展

2026-08-13 18:37:56（Asia/Shanghai）只读检查确认六个已授权 Job 仍为 RUNNING，且每个任务都只有对应 `after_lock`；没有 `pre_lock`、`try_lock` 或 `kill_lock`，标准错误末尾只有 PyTorch 的未来弃用警告，没有训练异常。Job `340896` 继续按用户要求跳过。

候选 mean/max 阶段完成 epoch 0，validation macro-F1 为 `0.5268334`。训练 14,304 个候选耗时 `1461.65 s`，吞吐 `9.7862 candidates/s`，其中 DataLoader 等待仅 `4.68 s`；完整 validation 的 `6495` 个候选耗时 `748.75 s`，吞吐 `8.6745 candidates/s`。这组真实 GPU 结果说明 4 个 train worker 没有形成候选预训练的主要瓶颈，因此不为增加 worker 数中断任务。

SMI-TED Light 289M/oracle 从 epoch 2 的 `0.5787330` 继续改善到 epoch 3 的 `0.6130376`。epoch 3 的三项选模分量分别是六分类 macro-F1 `0.5025572`、条件身份 Top-1 `0.7687861` 和端到端身份 F1 `0.5677695`；训练/validation 分别耗时 `1174.49/291.47 s`，训练 DataLoader 等待 `8.76 s`。

SMI-TED Light 289M/detached 在 epoch 4 首次未改善：选择分数 `0.6279913`，坏轮数为 1，调度器把下一轮学习率从 `1e-4` 降为 `5e-5`。epoch 5 使用 `5e-5` 后重新改善到 `0.6455865`，坏轮数归零；三项分量是 `0.5301863/0.7588358/0.6477374`。该轮训练/validation 分别耗时 `1304.36/284.72 s`，所以没有早停或重启依据。

MoLFormer/oracle 随后完成 epoch 0：选择分数 `0.5346026`，三项分量是 `0.5021218/0.6161369/0.4855491`。训练耗时 `1927.36 s`，其中 DataLoader 等待 `352.97 s`；validation 耗时 `376.50 s`。第一轮的数据等待占训练总时长约 18%，但仍远低于模型计算且任务已经产生 BEST/LAST；先观察缓存变热后的下一轮，不用单个冷轮结果重启。MoLFormer/detached 与 joint 尚未写出 epoch 记录，仍在各自 epoch 0。

新 A100 allocation 的当前进程树 MaxRSS 为约 `28–29 GB`，与旧 A800 detached allocation 的历史峰值 `677.5 GB` 不在同一数量级；后者包含旧 release 同时常驻两套 16-worker 进程池的累计历史，不能用来描述新 4/4 非持久方案。六个任务的 `after_lock` 全部继续保留。

## 一次性 calibration 入口：尚未读取保留集

在首轮方案仍由 validation 比较期间，新增 `python -m matcher.calibration` 和 `训练与运行/sh/evaluate_calibration.sh`，但没有执行该入口，也没有读取 calibration 候选。入口只接受已完成阶段的 `BEST.pt`：它核对 checkpoint 的 `epoch == best_epoch`、最佳轮次 `epoch_record`、阶段 `summary.json`、运行 `run.json`、相邻 `resolved.json`、模型与训练配置，并逐项比较训练记录和本次计划中的 `data_root/stage1_root/output_root/coverage_threshold/selection_seed/pp_top_k`。

训练完成后允许显式调整 validation 工作进程数、预取量和单工作进程缓存，因此入口分别记录训练 release 中 YAML 与本次 calibration YAML 的路径和 SHA-256，但不要求两份 YAML 字节相同；模型、训练超参数和六个科学数据字段仍必须相同。这一临时决定遵循用户允许基于 I/O 证据改变加载参数的边界，避免旧 release 的 16-worker 任务仅因后续 4/4-worker 优化而无法做同口径评估。

入口只调用 `require_manifest`：冻结清单不存在时立即失败，不允许在保留评估中静默重建。已经运行的 release 没有在 `resolved.json` 保存训练当时的 manifest 内容哈希，所以当前入口不能密码学证明训练时与评估时的清单文件逐字节相同；临时采用的治理边界是版本化 `closed_set_v2` 路径、来源根与监督参数必须一致，现存 manifest、calibration candidates 和 identities 在评估前重新计算并落盘 SHA-256。该限制不会被隐去，最终 calibration 报告必须保留为溯源限制；以后新训练若需要更强证据，应在训练开始时把清单内容摘要写进 resolved/checkpoint，而不是回填旧运行。

同一 calibration 稳定数据指纹在 claim 文件保留期间只能绑定一个阶段、checkpoint 和输出路径。独占锁覆盖整次评估；若结果 JSON 已原子发布但 claim 尚未从 `claimed` 收口到 `completed`，相同参数重试会先核验结果内的计划、阶段、checkpoint 和数据指纹，再只完成 claim，不再次运行模型。没有既有 claim 的任意同名输出不得冒充崩溃恢复。固定六类 macro-F1 把零支持类 F1 记为 0 并平均六类；支持类 macro-F1 只平均真实支持数大于零的类别；兼容字段 `macro_f1` 继续等于支持类口径，所以 validation 选模定义不变。

表达审查代理明确批准 `_sha256_file` 以及 `_calibration_data_evidence` 内稳定语义清单摘要与三摘要组合计算的具名硬限制例外；这些哈希只用于多类正式文件的内容留证和 claim 定位，不参与样本、监督、模型或选模计算。代理同时批准保留低复用但具有独立协议或事务语义的 `require_manifest`、`configure_random_seeds_and_numeric_backend`、`_select_stage`、`_calibration_data_evidence`、`_validate_checkpoint_for_stage`、`_claim_calibration`、`_recover_claim_from_published_output`、`_parse_arguments`、`run_calibration` 和 `main`。首轮终审发现并要求修复的两项阻断是数据/BEST 完成证据未闭环，以及 output 与 completed claim 两次发布之间的崩溃恢复窗口；修订后新增数据配置错配、非最佳 checkpoint、阶段完成、稳定指纹、真实非空指标、claim 恢复、无 claim 同名输出和只读清单缺失回归测试。

第二轮独立科学终审与严格表达终审均为 PASS，阻断项 0。科学审查独立核对错 selection seed、非最佳 checkpoint、epoch record 分数错配、仅 I/O 字段变化、claim 身份错配与恢复路径；表达审查逐项核对 TypedDict、CLI help、README 的结果/claim/runtime/evaluation 字段表、模块布局、中文 Docstring 与测试替身诚实性。冻结门禁为 pytest `51 passed`、mypy 38 个源文件零问题，compileall、七份 YAML、全部正式 shell 的 `bash -n`、calibration CLI help 和 `git diff --check` 全部通过。该入口截至此处仍未在服务器执行，也没有读取 calibration。

实现提交为 `f56a74b feat(evaluation): add one-time calibration entrypoint`。随后从本机 Matcher 根执行 `与服务器交互/run_sync.bat`，按保留远端既有文件的安全同步规则更新 `/home/penghongen/My_Project/Matcher`；正在运行任务继续使用各自冻结 release，不会热加载项目根。服务器 CPU 验收执行的核心命令是：

```bash
cd /home/penghongen/My_Project/Matcher
sha256sum matcher/calibration.py matcher/metrics.py matcher/training.py \
  matcher/tests/test_calibration.py 训练与运行/sh/evaluate_calibration.sh
set +u
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate Pocket_Plus_centos7_cu121_allgpu
set -u
python -m matcher.calibration --help >/dev/null
python -m pytest matcher/tests/test_calibration.py -q
```

第一次验收在 `set -u` 状态直接激活 Conda，环境自带 `activate-binutils_linux-64.sh` 因 `ADDR2LINE` 未定义而退出；pytest 尚未开始，不是代码或数据失败。改为与正式 shell 相同的 `set +u / activate / set -u` 后通过。五个文件本地/远端 SHA-256 依次完全相同：`88a8f0a1…d308bd`、`d1fb1e99…dc80fd`、`c118935c…b18cb4`、`2d86b91d…cbae3f`、`54ad9042…4b5b1b`；服务器专项结果为 `1 passed in 6.72s`。该验收只导入入口并运行合成编排测试，没有构造真实 calibration Dataset，没有申请 GPU，也没有创建 calibration claim 或结果 JSON。

## 首轮 validation 收口与一次性方案选择

2026-08-14 01:47:50（Asia/Shanghai）的只读检查确认，六个已授权运行均已自然早停并写出阶段 `summary.json`；对应 Slurm allocation 继续由 `try_lock + after_lock` 保留，没有删除任何 `after_lock`。Job `340896` 仍按用户明确要求跳过，没有补交、没有分配节点，也没有计入零分。候选六分类两阶段的 BEST 如下：

| 阶段 | BEST epoch | last epoch | validation macro-F1 | W&B |
| --- | ---: | ---: | ---: | --- |
| mean/max | 1 | 3 | 0.5304244533 | [mxornctd](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/mxornctd) |
| candidate-conditioned cross-attention | 4 | 6 | 0.5269085347 | [762wc627](https://wandb.ai/pencounkdual-111/AdaLigand-Matcher/runs/762wc627) |

两者差值为 `0.0035159187`，首轮六分类结果支持以 mean/max 作为较弱但更稳的聚合起点；cross-attention 的结合区域 micro-Dice 与配体区域 micro-Dice 更高，但不改变本轮预先规定的六分类 macro-F1 选模口径。完整 Matcher 的 BEST validation 结果按选择分数降序如下；选择分数始终是六分类 macro-F1、以六分类判对为条件的身份 Top-1 和端到端身份 F1 的等权平均：

| 排名 | Job / 语言表示 / 损失 | BEST / last epoch | 选择分数 | 六分类 macro-F1 | 条件身份 Top-1 | 端到端身份 F1 | 内在身份 Top-1 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 340895 / SMI-TED Light 289M / detached | 9 / 11 | 0.6696320918 | 0.5283099092 | 0.8034934498 | 0.6770929163 | 0.7743119266 |
| 2 | 340894 / SMI-TED Light 289M / oracle | 7 / 9 | 0.6654351100 | 0.5254060463 | 0.8093023256 | 0.6615969582 | 0.7504587156 |
| 3 | 340899 / MolFormer / joint | 7 / 9 | 0.6618542906 | 0.5129085101 | 0.8405405405 | 0.6321138211 | 0.7114337568 |
| 4 | 340897 / MolFormer / oracle | 7 / 9 | 0.6539754941 | 0.5239893710 | 0.7850678733 | 0.6528692380 | 0.7205081670 |
| 5 | 340898 / MolFormer / detached | 7 / 9 | 0.6398127499 | 0.5243064395 | 0.7648401826 | 0.6302916275 | 0.7059891107 |

因此，本轮在读取 calibration 前唯一选定 Job `340895` 的 SMI-TED Light 289M detached BEST：`/storage/penghongen/tmp/Matcher/experiments/full_scratch_smi_ted_light_289m_detached/train_experiment_job340895_20260813T121932_a1/full_scratch_smi_ted_light_289m_detached/BEST.pt`。它比第二名高 `0.0041969818`；该差距只来自单随机种子 validation 比较，且缺少用户暂时跳过的 SMI-TED Light 289M joint，因此最终报告必须保留“首轮方案选择、非方差估计”的限制，不能把小差距解释成稳定的损失优劣结论。一次性 calibration 只用于报告这个预先选定 checkpoint，不再参与换模型、调参或选模。

## Calibration 禁读边界更新与竞态处置

在后续批次尚未提出前，本代理曾依据“六个已授权运行均已结束且 validation 唯一选出第一名”的旧边界，准备复用 Job `340895` 做一次性评估。放行前核对该 allocation 为 `RUNNING`、`16 CPU + 1 A800`，只有 `try_lock_340895 + after_lock_340895`；checkpoint SHA-256 为 `c4ad33ce5b91e83c5e408b4d214114fc3abdffa66b49401264860649149d73cb`，目标结果与 claim 目录均为空。动态命令 SHA-256 为 `0fce31f463a03121d5e9d0fb1de79a42faf2bbee2db77bbfdf2976b775785255`，完整内容如下：

```bash
plan_relative=configs/experiments/initial/full_scratch_smi_ted_light_289m_detached.yaml
stage_name=full_scratch_smi_ted_light_289m_detached
checkpoint=/storage/penghongen/tmp/Matcher/experiments/full_scratch_smi_ted_light_289m_detached/train_experiment_job340895_20260813T121932_a1/full_scratch_smi_ted_light_289m_detached/BEST.pt
output=/storage/penghongen/tmp/Matcher/calibration/round1_smi_ted_light_289m_detached_job340895_epoch9.json

bash "${TASK_PROJECT_ROOT}/训练与运行/sh/precompute_data.sh" \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --stage1-root /storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477/artifacts/Find_0 \
  --output-root /storage/penghongen/tmp/Matcher \
  --coverage-threshold 0.30 \
  --selection-seed 3407 \
  --splits calibration \
  --workers 16 \
  --products voxel pp \
  --pp-top-k 1024

exec bash "${TASK_PROJECT_ROOT}/训练与运行/sh/evaluate_calibration.sh" \
  --plan "${plan_relative}" \
  --stage "${stage_name}" \
  --checkpoint "${checkpoint}" \
  --output "${output}"
```

01:54:38（登录节点时钟）只删除 `try_lock_340895`；allocation runner 建立 release `/home/penghongen/Feedback/Matcher/releases/Matcher_5952c2c41169/Matcher` 与 launch `/home/penghongen/Feedback/Matcher/launches/340895/train_experiment_job340895_20260814T015240_a2` 后开始特征预计算。用户随即建立新的严格边界：第 0 批和后续全部开发、选择、头对头比较只使用 validation；后续批次全部结束且用户明确确认最终方案前，不得实际读取 calibration，是否执行最终一次评估也由用户另行确认。该新边界覆盖前述“已选出第一名后即可评估”的旧时序。

收到指令后立即停止。02:02:04 创建且只创建 `/home/penghongen/Feedback/Matcher/allocations/340895/kill_lock_340895`；监控器终止本次预计算进程组，attempt 2 以退出码 `137` 结束，并在 02:02:39 前重新建立 `try_lock_340895`。`after_lock_340895`、第 0 批 checkpoint 与训练日志均未改变。终止发生在 `CALIBRATION_PRECOMPUTE_COMPLETED` 和 `CALIBRATION_EVALUATION_STARTED` 之前，因此模型评估没有启动，calibration claim 和结果 JSON 从未创建。

并发预计算被终止前已经读取并物化了部分 calibration 派生输入，这是发生在新指令到达前、但仍必须如实记录的边界触碰。最终盘上有 `3294/3363` 个 schema v1 体素 `.npy` 和 `83/86` 个 PP `.npz`，大小分别为 `76,502,003,456` 与 `72,552,964` bytes；另有 `3295/83` 个空锁文件，没有遗留临时原子写入文件。这些派生缓存没有指标、预测或 checkpoint 选择信息。本代理不擅自删除它们；从此刻起不再打开 calibration 清单、派生缓存或底层特征，直到用户在全部批次结束后明确授权。

## 验证命令

```powershell
python -m compileall -q matcher
python -m mypy matcher
python -m pytest matcher/tests -q
git diff --check
```

模型审查修复时的阶段证据是 mypy 31 个源文件零问题、pytest `22 passed`、compileall 与 `git diff --check` 通过。训练、测评、配置与产物边界完成后的冻结结果是 mypy 32 个源文件零问题、pytest `33 passed`、compileall、7 个 shell 的 `bash -n`、七个 YAML 正式解析与 `git diff --check` 通过。BRANCHED 伪几何修复后的当前结果是 pytest `34 passed`、mypy 32 个源文件零问题、compileall 与 `git diff --check` 通过；三组结果都保留，不用新结果覆盖旧阶段证据。

## 待执行与待复核项

- v2 清单、105 通道体素、PP token 缓存、BRANCHED 多残基伪几何修复和 DataLoader 生命周期均已完成双审、安全同步与服务器门禁；后续 release 必须继续在放卡前核对源码和配置哈希。
- `pp_top_k=1024`、六分类 `class_weights=null`、区域辅助损失权重 `0.10/0.10` 和完整 Matcher validation 三指标等权选模是首轮显式起点，需根据清单摘要和首个稳定实验记录复核。
- Job `340893/340894/340895/340897/340898/340899` 均在正式训练。仅在新 epoch、训练阶段切换、失败或任务完成时追加执行记录，不按固定心跳重复写相同状态。Job `340896` 保持暂停，不自动补交。
- GPU 资源实际分配后，删除 `pre_lock` 会直接启动正式命令；每个阶段的首个真实 global optimizer step 同时作为连续验收点。只有完成有限损失与梯度检查、参数更新和 CUDA 同步后，程序才打印一次 `first_global_step_accepted`，其中含 global batch、有效身份数、裁剪前梯度范数、峰值显存和首步计算耗时。这样不在验收与正式训练之间闲置 GPU；若 OOM 或非有限数值失败，程序不会打印成功事件，`after_hold` 仍会保留 allocation 供修订后重试。若 batch size 超出显存，按用户确认的顺序先将 RAUNet 通道降到原宽度的 3/4，再降低物理批量并增加梯度累积，始终保持 global batch 32。
- 首个完整训练若超过 16 小时，只暂停后续随机种子；不中止当前训练，不设任何时间上限。

## 2026-08-14：第 0 批收口与第 1–5 批接续

第 0 批 Goal 已按“重建与基线实验”完成，新的活动 Goal 覆盖第 1–5 批开发、测试、发布、六张受管 GPU 的批次优先调度、W&B 分批管理、validation 头对头比较和三层实验记录。Goal 切换没有修改 Slurm 任务、GPU 进程或锁。02:40 的只读状态显示 Job `340893/340894/340895/340897/340898/340899` 均保留 `after_lock`；没有取消、释放或删除锁。

新增 `文档/记录/总记录.md`、批次记录和单任务记录。第 0 批七个实际实验分别保存提交命令、Job、release、launch、W&B、首步或最终结果；第 1 批三个待运行实验先保存配置、预定受管 allocation 和直接对照。`CLAUDE/memory/handoffs/2026-08-14-matcher-batches-1-5.md` 保存跨会话状态、决定、开放问题和下一步，不替代本执行记录。

第 1 批覆盖率审计使用 CPU-only Job `341588`，精确提交命令为：

```bash
sbatch /storage/penghongen/tmp/Matcher/diagnostics/identity_coverage_train_validation_20260813T181421Z/run.sbatch
```

该 Job 使用 `cpu/cpu96`、2 CPU、16 GiB 内存，20 分 44 秒后 `COMPLETED/0:0`；脚本硬编码只读 train/validation，没有 GRES，没有打开 calibration 或 Stage1 特征数组。train 五类共 1,477 个 PDB 内身份、10,681 个 occurrence，SMI-TED 与 MoLFormer 均为图+语言 100% 可用。validation 共 536 个身份、4,069 个 occurrence；MoLFormer 100%，SMI-TED 为 534/536 个身份、4,063/4,069 个 occurrence。唯一缺口是 `6rkd` 的两个小分子身份和六个 occurrence，对应六个 validation 候选。

按候选监督口径，SMI-TED 在 train 覆盖 `4,192/4,192` 个前景候选，在 validation 覆盖 `1,466/1,472`。缺失候选继续参与六分类和区域监督，只令 `identity_target=-1`；不混用 MoLFormer 回退，也不排除 `6rkd`。审计未发现同一 PDB、类别、`object_key` 对应多个模型 SMILES、跨类别身份冲突或配体图安全文件名碰撞。证据根是 `/storage/penghongen/tmp/Matcher/diagnostics/identity_coverage_train_validation_20260813T181421Z/`，其中 `SHA256SUMS` 保存完整摘要。该证据支持第 1 批统一使用 SMI-TED，但不单独证明其表征质量优于 MoLFormer。

当前工作树正在实现 manifest schema v3、显式划分物化、五类身份清单、`identity_scope` 兼容开关、12 维固化候选统计、目标身份类别概率特征、多层身份指标和分层 W&B 命名。第 1 批已建立三个 YAML：A800 oracle、A800 detached、A100 joint，global batch 均为 32。calibration 可以顺路准备清单或缓存，但在全部批次结束且用户再次确认前，不调用模型、不产生 calibration 指标、不把任何准备产物用于选模。

2026-08-14 03:05（Asia/Shanghai），第 1 批提交前本地快照完成门禁：Matcher 全量 pytest 为 `56 passed`，mypy 检查 38 个源文件零问题，compileall 通过，七份第 0 批和三份第 1 批 YAML 均由 `load_experiment_plan` 成功解析，`git diff --check` 通过。本轮新增 W&B 回归明确覆盖 `train/{step,epoch}` 的损失、优化、性能、统计四棵子树，以及 validation 六分类、混淆矩阵、配置身份范围、小分子兼容子集、五类总体、逐类身份、区域、分母和选模子树；有真实零支持分母的 NaN 只在在线边界转为 null，非有限训练值仍立即失败。该快照接下来只交由阶段性科学正确性审查和严格技术表达审查；双审完成前不生成 release、不删除任何受管 allocation 的锁，也不启动 calibration 推理。

### 第 1 批科学语义与在线曲线树修订

阶段性科学审查发现，最初把 A/P/V/PP 的 12 项手动统计先经单个 `12→D` 编码器混合，再只在最终融合层追加，不能证明各模态统计与对应模态摘要逐一绑定。活动实现改为把标准化后的四组三维统计分别紧邻 A/P/V/PP 的 mean/max 摘要；候选条件注意力路径则紧邻相应模态的注意力增量。身份匹配得分继续读取相同 12 项统计，并拼接 `stopgrad(softmax(class_logits)[identity_class])`。这项修订不改变候选、监督或 global batch，只把用户指定的模态归属落实为直接张量契约。

用户随后明确五类端到端指标中的六分类子类只是一项中间辅助量。由此冻结三种口径：`conditional` 仍要求六分类预测精确等于真实子类；五类总体和逐类 `end_to_end` 只要求六分类预测为任一非背景类，再由 Matcher Top-1 是否命中真实身份决定成功；第 0 批兼容的小分子端到端仍要求六分类预测为 `small_molecule`。逐类端到端预测阳性按最终 Top-1 身份类别归桶，真实前景但 `identity_target=-1` 的候选排除在身份指标外，背景误报为前景且存在身份槽时计为假阳性。独立八样本探针覆盖跨子类正确身份、背景误报、缺失目标和逐类加和不变量。

W&B 指标同时从笼统的 `train/*`、`validation/*` 首层改为职责优先路径。第一层为 `loss/gradient/gate/learning_rate/performance/memory/statistics/classification_summary/classification_per_class/classification_confusion/identity_*/area/selection`，第二层再区分 train step、train epoch 或 validation。这样符合 W&B 默认工作区按第一个 `/` 前缀建立区块的实际行为；候选预训练不写身份分支、身份指标和门值的空数据。完整嵌套报告仍逐 epoch 保存在 `history.jsonl`，在线曲线不替代落盘证据。

兼容性方面，calibration 在读取旧第 0 批 `resolved.json` 时，会把缺失的 `model.use_manual_features/use_sampling_features` 迁移为 `false/false`，并按旧任务语义把缺失的 `stage.identity_scope` 迁移为 `small_molecule` 或 `None`。第 1 批 checkpoint 已有 `use_manual_features=true` 而只缺采样开关时，保留手动统计并仅迁移 `use_sampling_features=false`。历史 YAML 原文件仍不接受缺字段的静默解析；复现旧 checkpoint 时使用当前已显式补全字段的第 0 批 YAML。这一边界保持正式配置自包含，同时避免旧 BEST 因新增字段失去可核验性。

本轮修订期间的定向回归为 `39 passed`，mypy 检查 38 个源文件零问题。最终全量门禁与两名阶段审查代理的冻结结论在本节后续追加；在其通过前不建立 release、不启动第 1 批任务。第 0 批六个受管 allocation 与所有 `after_lock` 保持原样，calibration 模型推理仍未执行。

03:54 的最终冻结门禁为 pytest `59 passed`、mypy 38 个源文件零问题、pyflakes、compileall、10 份正式 YAML 解析、全部 shell 的 `bash -n` 和 `git diff --check` 通过。独立科学终审结论为 PASS：五类清单与缺失表示跳过语义、三种损失的类别概率梯度、global-batch 分母、四模态手动特征拼接、宽松端到端逐类归桶、PP 零增量和旧 artifact 迁移均无阻断。严格表达终审结论为 PASS：持久 manifest v3 字段、候选与批次张量、W&B 路径、三份 YAML、三层账本和测试替身均可冷读。

严格表达代理代用户批准以下具名结构例外：旧配置经 `MatcherModelConfig.from_dict` 缺失 `use_manual_features` 时只迁移为 `false`；`CandidateDataset.cache_bytes_per_worker` 与 `EvaluationAccumulator.top_k` 的程序化默认仅服务独立构造，正式 YAML 仍显式配置；低复用的 `_candidate_manual_features/_identity_target_index/_extend_manifest/_annotate_seen/_publish_manifest/identity_eligible_mask/normalize_candidate_manual_features/identity_scope_class_indices/_evaluation_wandb_metrics` 各自集中单一科学或事务契约；`matcher/manifest.py` 本批保留部分 `dict[str,Any]`，由持久字段文档和回归测试冻结。`文档/记录/目标草案.md` 保留用户原始讨论文本，不改写成正式规划文体。至此第 1 批可以进入实现提交、服务器同步和清单预计算；calibration 模型推理仍被禁止。

### 第 1 批实现提交、同步与清单 v3 物化

第 1 批实现与测试提交为 `2467b39 feat(batch1): expand identity matching to five ligand classes`，冻结门禁和三层账本提交为 `8f7f56c docs(batch1): record five-class release gate`。随后执行 `powershell -File 与服务器交互/sync_code.ps1`，只做安全同步，不删除服务器项目、release、launch、锁或实验产物。远端正式环境的 Matcher 全量 pytest 为 59 项通过；compileall 和 10 份 YAML 正式解析通过。远端 Conda 环境没有安装 mypy，因此服务器 mypy 命令以 `No module named mypy` 退出；这不是代码失败，本地冻结门禁已经对 38 个源文件完成 mypy 零问题检查。

同步后，本地与 `/home/penghongen/My_Project/Matcher` 的关键文件 SHA-256 一致：`manifest.py=2b45a16c…71525`、`candidate_data.py=e65efd3e…b7d17`、`losses.py=956dde82…8fec`、`metrics.py=9adb8dc8…3a40`、`tracking.py=c4b1e3ae…03b6`；三份第 1 批配置分别为 oracle `de15aa54…615a`、detached `56d2ad00…93d3`、joint `af591933…606a`。完整摘要由本地与远端 `sha256sum` 输出留证；省略号仅用于本执行记录的紧凑索引。

03:53:45（Asia/Shanghai）提交 CPU-only Job `341600`，精确命令为：

```bash
cd /home/penghongen/My_Project/Matcher
bash 训练与运行/submit_task.sh \
  --sh 训练与运行/sh/build_manifest.sh \
  --resource cpu --partition cpu --qos cpu96 --cpus 2 --mem 16G \
  --job-name matcher-batch1-manifest-v3 -- \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --stage1-root /storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477/artifacts/Find_0 \
  --output-root /storage/penghongen/tmp/Matcher \
  --coverage-threshold 0.30 --selection-seed 3407 \
  --splits train validation
```

该 Job 使用 `cpu/cpu96`、2 CPU、16 GiB，不申请 GPU，也不设置时间上限；release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_bb899c6e37ed/Matcher`，launch 为 `/home/penghongen/Feedback/Matcher/launches/341600/build_manifest_job341600_20260814T035348_a1`。04:03 前该 Job 仍在构建，清单目录只有原子构建锁 `.build.lock`，尚未发布不完整 JSONL。六个受管 GPU allocation 同时核验为 RUNNING 且空闲；共享 allocation 根中的六个 `try_lock_<job>` 与各 Job 目录中的六个 `after_lock_<job>` 均存在，没有 `pre_lock` 或 `kill_lock`。在清单候选数、身份数、类别计数、SMI-TED 目标可用数和 calibration deferred 状态全部通过前，不替换动态命令、不删除任何 `try_lock`。

Job `341600` 于 04:17:54 以 `COMPLETED/0:0` 收口，耗时 24 分 09 秒。放行脚本在任何锁修改前验证本地与远端五个关键源码、三份配置的完整 SHA-256，并逐行读取新清单确认：schema 3；仅有 train/validation；calibration deferred 且不存在对应候选或身份 JSONL；train 为 14,312 个候选、1,477 个身份、4,192 个 SMI-TED 可监督前景候选；validation 为 6,495 个候选、536 个身份、1,466 个 SMI-TED 可监督前景候选；六类计数与覆盖率审计一致；背景目标均不可用。

04:18:48（控制端时钟）依次为 Job `340894/340895/340893` 原子安装 oracle/detached/joint 动态命令，命令 SHA-256 分别为 `91b8825f927a177dcc4416782a2ec28f58384e01f856d137bfd8f06ca1096588`、`87fd62a9318b765f8fd291ad5e6e297dbe858d303791c9dc7bbe972cd039edc6`、`483412ff7af602feb3f8d47e480a36afaeea978d4c2f7ecf68ef1fd33a7b32db`。三次触发只删除共享 allocation 根下各自的 `try_lock`；三个 `after_lock` 在删除前后均存在。Job `340895` 的旧动态命令是已经终止的 calibration 准备入口，本次覆盖后未恢复、未继续或执行 calibration。

三个 attempt 均复用内容寻址 release `/home/penghongen/Feedback/Matcher/releases/Matcher_4cf3ef45e8a8/Matcher`：

- oracle：Job `340894`，launch `/home/penghongen/Feedback/Matcher/launches/340894/train_experiment_job340894_20260814T041802_a6`，W&B run `h41na5ea`，地址 `https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch1/runs/h41na5ea`；
- detached：Job `340895`，launch `/home/penghongen/Feedback/Matcher/launches/340895/train_experiment_job340895_20260814T041633_a3`，W&B run `vry3gxt2`，地址 `https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch1/runs/vry3gxt2`；
- joint：Job `340893`，launch `/home/penghongen/Feedback/Matcher/launches/340893/train_experiment_job340893_20260814T041902_a2`，W&B run `amdxqwjc`，地址 `https://wandb.ai/pencounkdual-111/AdaLigand-Matcher-Batch1/runs/amdxqwjc`。

三项 tracking 产物都确认 `mode=online`、project `AdaLigand-Matcher-Batch1`、group `matcher-batch1-five-class-manual-features`。节点时钟并未完全同步，所以 detached 的 launch 名时间早于控制端触发时间；attempt 序号、release、命令哈希、业务目录和 W&B 身份一致，不把名称时间当作执行排序依据。

三项都完成首个真实 global batch 32 更新。oracle 的首步为 `61.7351 s`、有效身份 7、裁剪前梯度范数 `49.9738`、峰值已分配/保留显存 `28,143,265,792 / 39,747,321,856 bytes`；detached 为 `62.6836 s`、7、`32.3384`、`28,143,814,144 / 39,522,926,592 bytes`；joint 为 `24.2634 s`、7、`30.3424`、`14,195,499,008 / 19,511,902,208 bytes`。成功事件均发生在有限性检查、优化器更新与 CUDA 同步之后；日志只有既有 PyTorch 弃用警告，没有 OOM、NaN、非有限梯度或 Traceback。三个任务继续运行，所有 `after_lock` 保留；后续改为按明确事件低频记录。

### 第 2 批 V 质心视野实现与发布前门禁

第 2 批只改变局部视野并新增四项采样统计，不改变候选集合、全局双向覆盖率、六分类/身份监督、global batch 或候选全局随机排列；第 1 批已有的 12 项模态手动统计保持原定义。活动实现新增 `CandidateCropGeometry`，以完整权威 V 的 float64 下标均值计算 `np.rint(mean(V_zyx)+0.5-24)`，再逐轴限制到 `[0,32]`。Dataset 为每个候选解析一次实际起点，并把它用于 105 通道、V、A/P 相对坐标以及配体/结合区域；PP 预计算按 PDB 独立调用同一几何函数，Dataset 读取后再逐项核对其缓存起点。105 通道仍先在完整 `80³` 上完成 Find_0 密度构造和完整受体散射，再裁取候选 `48³`，避免改变归一化与 Gaussian 边界语义。

voxel 与 PP 缓存均提升为 schema v2，路径分别是 `cache/voxel_input_v2/<strategy>/` 和 `cache/pp_input_v2/<strategy>/top_<k>/`。voxel 文件名同时绑定 `candidate_index` 与实际 `crop_start_zyx`；PP v2 在 NPZ 内保存每个候选的 `crop_start_zyx`，Dataset 命中后逐项核对。`fixed_center` 与 `v_centroid` 不复用缓存。旧第 0/1 批任务继续从各自内容寻址 release 读取原 schema，不受活动工作区影响。

新增 `candidate_sampling_features` 四项原始 float32 统计：起点边界限制标志、实际裁剪中心到完整 V 体素中心质心的 L2/L∞ 距离和 V 保留比例。候选融合以 `24√3/24` 归一化距离后紧邻 U-Net 摘要拼接；完整 Matcher 的 native、graph、inventory 三个最终打分 MLP 读取同一统计。12 项 A/P/V/PP 计数与固化概率仍保持第 1 批顺序和模态归属。

合成测试覆盖内部候选、六个边界面、NumPy 最近偶数取整、宽 V 部分保留和高 X 边界端到端同源性；后者在一个样本中同时核对 105 通道、V/PP 索引、配体/结合目标以及 A/P 新相对原点。阶段门禁为 pytest `68 passed`、mypy 38 个源码文件零问题、pyflakes、compileall 与 `git diff --check` 通过。该结果早于五份 Batch2 YAML、README 和三层账本的最终落盘，发布前必须重新执行完整门禁和独立双审。

独立科学审查随后复现两项会绕过张量形状检查的缓存风险。第一，早期 v2 NPY 仅按候选编号寻址，同一候选换起点会静默复用旧数组；现已把起点编码进文件名，并用双起点逐值回归锁定。第二，105 通道使用 exp 的原点/尺寸和 F1 的 `box_start_zyx`，但原实现没有 exp/sim↔F1 桥接；现于缓存写入前核对全部候选的 `(C,3)` 起点、原点和尺寸，要求 F1 spacing 与密度图逐元素相等，并以 `5e-5 Å` 绝对容差核对 `F1 box_origin = density origin + ZYX起点转XYZ × spacing`。各向异性 spacing、非零原点与非对称起点正例，以及 exp origin/spacing 漂移反例均通过；数据专项为 `31 passed`，相关 mypy 四文件零问题。结合既有 F1↔probability 校验，exp/sim↔F1↔probability/PP 的空间链已闭合。

05:19 的只读 Slurm 核验显示六个受管 allocation 均为 `RUNNING`；Job `340897/340898/340899` 的第 0 批 MoLFormer 已分别在 epoch 9 自然早停，当前只有共享 allocation 根中的 `try_lock` 和各 Job 目录中的 `after_lock`。没有修改这些锁或动态命令。第 2 批前三个完整 Matcher 预定依次复用这三张 A100，物理批量 8、累积 4；两个候选预训练预定在受管 A800 自然空闲后以 16×2 接续。五项均为 global batch 32，并写入独立 W&B project `AdaLigand-Matcher-Batch2`。

真实 `7z2c/candidate_1` 回归只作为 calibration 准备性几何核验。诊断产物根为 `/storage/penghongen/tmp/Matcher/diagnostics/batch2_7z2c_geometry_20260814T0524/`。第一次 CPU-only Job `341614` 因 `sbatch --wrap` 嵌套引号丢失，在 1 秒内以 Python `SyntaxError` 退出，未成功解释脚本、未打开 NPZ；该失败是命令封装问题，不是几何失败。修订为通过 Slurm 环境变量传递 base64 脚本后，成功提交时执行的控制端 PowerShell 命令如下；临时脚本 `tmp/batch2_7z2c_geometry_probe.py` 的 SHA-256 为 `E14DB0852B18088533FA82EB60968FE1F3F9FE219FB224B3B2FC2C729614EDD3`：

```powershell
$encoded = [Convert]::ToBase64String(
  [IO.File]::ReadAllBytes('C:\Users\15919\Desktop\Matcher\tmp\batch2_7z2c_geometry_probe.py')
)
$envName = 'chr(80)+chr(82)+chr(79)+chr(66)+chr(69)+chr(95)+chr(66)+chr(54)+chr(52)'
$remote = "sbatch --parsable --job-name=matcher-batch2-7z2c-geometry " +
  "--partition=cpu --qos=cpu96 --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G " +
  "--export=ALL,PROBE_B64=$encoded " +
  "--output=/storage/penghongen/tmp/Matcher/diagnostics/batch2_7z2c_geometry_20260814T0524/slurm-%j.out " +
  "--error=/storage/penghongen/tmp/Matcher/diagnostics/batch2_7z2c_geometry_20260814T0524/slurm-%j.err " +
  "--wrap='exec /home/penghongen/anaconda3/envs/Pocket_Plus_centos7_cu121_allgpu/bin/python " +
  "-c `"import base64,os;exec(base64.b64decode(os.environ[$envName]))`"'"
& "$env:USERPROFILE\.codex\tools\Invoke-ProjectSsh.ps1" -Command $remote
```

Job `341618` 为 `COMPLETED/0:0`，运行 11 秒，stderr 为空。输出证实完整 V `170`，质心 ZYX `[39.6117647059,39.8352941176,70.5882352941]`，请求起点 `[16,16,47]`，限制后 `[16,16,32]`；旧 `[16,64)³` 保留 0，新视野保留 170。脚本只读取一个 `F1_centered.npz` 的 V offsets/indices，没有导入 Matcher 模型、加载 checkpoint、创建 DataLoader、运行 W&B、计算指标或创建 calibration claim。

06:28 的第 2 批冻结门禁为 pytest `77 passed`、mypy 38 个源码文件零问题、pyflakes、compileall、15 份正式 YAML 解析、8 个 shell 的 `bash -n` 与 `git diff --check` 全部通过。独立科学终审确认候选集合、监督、global batch、统一裁剪几何、缓存起点绑定、exp/sim↔F1↔probability 空间链、采样统计及旧产物迁移均无阻断；严格表达终审确认跨模块函数与结构化字段、五份 YAML、持久缓存契约、三层账本和 calibration 状态说明均可冷读。两项终审结论均为 PASS，阻断项为零；下一步只做实现提交、安全同步、内容寻址 release 与 train/validation v2 缓存预计算，不执行 calibration 模型推理。

第 2 批实现、五份配置、测试和三层账本提交为 `852c56954e0d74a6344cf75fedd6e6962d6a5ae6 feat(batch2): center local views on candidate V`。随后执行 `与服务器交互/sync_code.ps1` 安全同步；它没有删除服务器项目、release、launch、锁或实验产物。服务器 `geometry.py/voxel_input.py/pp_input.py/precompute.py/oracle YAML` 的 SHA-256 与本地逐项一致，正式环境全套 pytest 为 `76 passed, 1 skipped`；跳过项只受服务器设备条件影响，本地同一提交为 77 项全过。

同步后先确认 `cache/voxel_input_v2/v_centroid` 与 `cache/pp_input_v2/v_centroid/top_1024` 均不存在，再提交纯 CPU 预计算 Job `341634`。完整控制端执行命令如下；命令不含 `--time`、不申请 GPU、只请求 train/validation，并同时显式选择 voxel/PP 和 `v_centroid`：

```bash
cd /home/penghongen/My_Project/Matcher && bash 训练与运行/submit_task.sh \
  --sh 训练与运行/sh/precompute_data.sh \
  --resource cpu --partition cpu --qos cpu96 --cpus 16 --mem 512G \
  --job-name matcher-batch2-v2-precompute \
  -- \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --stage1-root /storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477/artifacts/Find_0 \
  --output-root /storage/penghongen/tmp/Matcher \
  --coverage-threshold 0.30 --selection-seed 3407 \
  --splits train validation --workers 16 \
  --products voxel pp --pp-top-k 1024 \
  --local-crop-strategy v_centroid
```

Job `341634` 于 06:40 在 `cnode01` 开始运行，资源为 `cpu/cpu96`、16 CPU、512 GiB，release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_3cc75993d69e/Matcher`，launch 为 `/home/penghongen/Feedback/Matcher/launches/341634/precompute_data_job341634_20260814T064021_a1`。本次没有 `--pre_hold/--after_hold`；程序自然结束后 CPU allocation 自动释放。06:49 的明确进度为程序已报告完成 train 候选 `5,114/14,312`，已原子发布 train voxel `5,761/14,312` 和 PP PDB `137/480`，validation 尚未开始；文件数可因多个工作进程正在并发发布而暂时领先主进程完成计数。stderr 只有 release 创建记录，没有异常。GPU Job 和六个 `after_lock` 均未修改；前三个第 2 批 A100 命令只在缓存自然完成并通过计数与随机样本读取后安装。

06:54 对第 1 批三份 `history.jsonl` 做事件式复核：oracle 的最佳 selection score 为 epoch 3 的 `0.6812668732`，epoch 4 为 `0.6786411725`、坏验证轮数 1、下一轮学习率 `5e-5`；detached 在 epoch 4 改善到 `0.6756995547`；joint 在 epoch 3 改善到 `0.6452062205`。三项继续运行，所有 `after_lock` 保留；这次读取只使用 validation 记录，没有访问 calibration。

Job `341634` 随后完成全部 `20,807` 个 train/validation 候选，于 07:31 记账为 `COMPLETED/0:0`，耗时 `00:50:45`。最终 v2 voxel 文件精确为 train `14,312`、validation `6,495`，PP 文件精确为 train `480`、validation `181`；没有生成 calibration 缓存。放卡脚本 `tmp/batch2_launch/install_a100_tasks.sh` 的 SHA-256 为 `313774F25A09D1953B530CFAE7B128329DBDF4BA3C1E80CF818450D705F1F022`；它在首次删除锁前完成四个热点源码与三份 YAML 的 SHA-256 核对、三项 allocation 旧任务身份与资源核对、三类实验输出防重复、正式 YAML 解析，以及固定 seed `3407` 下每个划分 8 个候选的 Dataset 实际读取。该脚本经独立只读终审判定 PASS；三份动态命令全部安装完成后才开始删除锁，且唯一删除对象是 `try_lock_340897/340898/340899`。三份 `run_cmd` 的 SHA-256 依次为 `9bf0f8ac2add452b39e8e5c8083482544cf7e4f70a467f1d34268861acabb0f4`、`0a7f8a4c4fd6fa2fbfae74cbf772cfb960791c7e6e16d6c0c6d4001c3fc17669`、`97a01ac91fc6549043bbe267f2f24321fa6f49b83b7f02e9859fa356d69738c0`。

07:33:00 控制端严格按 oracle→detached→joint 顺序删除上述三把 `try_lock`；allocation runner 各自独立轮询，因此实际 launch 墙钟顺序不作科学含义解释。三项均使用 release `/home/penghongen/Feedback/Matcher/releases/Matcher_3cc75993d69e/Matcher`，完整身份如下：

| 变体 | Job / launch | W&B run | 首步计算 / 裁剪前梯度 / 峰值保留显存 |
|---|---|---|---|
| oracle | `340897` / `train_experiment_job340897_20260814T073305_a2` | `6soiol5e` | `25.8580 s / 42.1343 / 19,610,468,352 bytes` |
| detached | `340898` / `train_experiment_job340898_20260814T073316_a2` | `i0j1i1hc` | `25.0679 s / 26.4270 / 19,440,599,040 bytes` |
| joint | `340899` / `train_experiment_job340899_20260814T073303_a2` | `s3lj6298` | `24.4089 s / 25.6375 / 19,442,696,192 bytes` |

三项均连接 W&B project `AdaLigand-Matcher-Batch2`、group `matcher-batch2-full-v-centroid-sampling`，首个参数更新的 global batch 为 32、有效身份目标为 7；事件发生在有限性检查、参数更新和 CUDA 同步之后。未见 OOM、NaN、Traceback 或缓存回退。07:35 复核显示六个受管 allocation 均为 `RUNNING` 且只保留各自 `after_lock`；没有删除或释放任何 `after_lock`。第 2 批两个候选任务继续等待第 1 批 A800 自然产生 `try_lock`，不抢占或中断正在运行的 Job `340894/340895`。

### 第 3 批增强候选条件注意力实现

第 3 批只改变候选条件注意力的实体分数，不改变第 2 批 V 质心视野、12 项模态统计、4 项采样统计、候选集合、监督、global batch 或优化器。一次性结构审计先核对 `Stage1训练与多阈值推理.md`、Pocket_Plus selector 和当前 Matcher token 契约，确定四种模态都已在原始 token 末尾保存 Stage1 固化概率与相对候选实际裁剪中心的归一化 XYZ。因此每模态偏置输入固定为 `(relative_x, relative_y, relative_z, stage1_probability)`，不移植 selector 中 Matcher 不具备的树谱系、最旧候选或框层级字段。

活动实现为 A/P/V/PP 分别建立 `Linear(4,D_m) → SiLU → Linear(D_m,K)`，其中 `K` 是注意力头数。末层权重和偏置采用零初始化，输出的 `(B,K,N)` 偏置在有效 token 掩码生效前加到 softmax 分数；初始化时逐元素退化为原有内容注意力，不增加额外门控。新增分支用隔离的 CPU 随机初始化上下文构造，使后续 class/区域头与未增强模型的共同参数初值不被新增参数的随机采样推进。PP 的注意力输出投影仍采用零初始化：当 PP 含有效 token 且损失沿该分支回传时，第一次反向传播先让该输出投影权重获得非零梯度，更新后下一次反向才把非零梯度传入 PP 偏置末层权重。偏置 MLP 末层的 `bias` 对一个注意力头沿 token 轴只产生常数平移，数学上不改变 softmax，因此不作为可学习路径或监控对象。

正式模型配置新增必填布尔字段 `use_candidate_token_attention_bias`。第 0–2 批全部显式为 `false`，第 3 批四份 YAML 为 `true`；正式 YAML 缺字段立即失败。历史第 0–2 批 checkpoint/resolved 只在 `MatcherModelConfig.from_dict` 恢复入口把缺失字段迁移为 `false`，从而保持严格 state dict 兼容。`true` 只允许与 `aggregation=candidate_cross_attention` 同时使用。

W&B online 在 `gradient/train_step/{a|p|v|pp}_attention_bias_final_weight_postclip_norm` 记录全局梯度裁剪后的偏置末层权重梯度范数，在 `attention_bias/train_step/{a|p|v|pp}_final_weight_poststep_norm` 记录同次优化器更新后的末层权重范数。某模态在整个 global batch 中没有有效 token、末层权重没有梯度时，两项都省略；末层 `bias` 不记录。四项实验使用 project `AdaLigand-Matcher-Batch3`；三个完整 Matcher group 为 `matcher-batch3-full-enhanced-cross-attention`，六分类 group 为 `matcher-batch3-candidate-enhanced-cross-attention`。三份完整 Matcher 以第 2 批相同损失的 mean/max 任务作为预算内基线；增强六分类与第 2 批未增强候选条件注意力直接比较。

首个冻结候选的机械检查为 pytest `86 passed`、mypy 检查 38 个源文件零问题、compileall、pyflakes、19 份正式 YAML 解析与 `git diff --check` 通过。独立科学终审为 PASS；严格技术表达终审发现 W&B 诊断时序、模型张量契约、四份 YAML 自包含性、随机流测试诚实性和三层账本状态五组阻断。集中修订后的当前快照为 pytest `93 passed`、mypy 38 个源文件零问题、compileall、pyflakes、19 份正式 YAML 解析、49 个 shell 的 `bash -n` 与 `git diff --check` 全部通过；现等待科学正确性与严格技术表达两项独立复审。当前没有发布 release、安装动态命令或提交第 3 批 GPU 任务；第 2 批两个候选任务仍拥有更高优先级。

最终复审对集中修订后的冻结快照均给出 PASS。科学复审独立核对四模态概率、FP32/BF16 零退化、PP 两步唤醒、完整模型共同参数与随机流、AdamW 连续两步更新以及 W&B 裁剪后梯度/更新后参数时序；严格技术表达复审核对逐字段张量契约、四份自包含 YAML、诊断键、条件梯度表述和三层账本。最后一处 `CandidateMetadata` 字段说明修正后再次通过窄审。第 3 批由此达到提交条件，但在形成提交并安全同步前仍不作为服务器发布源。

2026-08-14 09:20 后，Job `340894/340895` 的第 1 批 oracle/detached 已分别在 epoch 9/10 自然早停，allocation runner 创建共享根 `try_lock` 并保留各 Job 目录的 `after_lock`。控制端先确认两项 Job 仍为 `RUNNING`、16 CPU、单张 A800，目标实验根不存在，远端 9 个第 2 批源码/配置 SHA-256 与冻结端点一致，正式配置为 W&B online、V 质心视野、4/4 非持久 worker、A800 `16×2=global 32`，且 v2 voxel/PP 缓存计数为 `14,312/6,495` 与 `480/181`。受保护接续脚本为 `tmp/batch2_candidate_handoff_20260814/activate_a800_candidates.sh`，SHA-256 `092a61bdd10e301bd9f23c05cbe269340cdddfeb7358156d37692e23d00093c4`；完整控制端命令为：

```powershell
& "$env:USERPROFILE\.codex\tools\Invoke-ProjectSsh.ps1" -Command 'bash -s' -InputFile 'C:\Users\15919\Desktop\Matcher\tmp\batch2_candidate_handoff_20260814\activate_a800_candidates.sh'
```

脚本先原子安装 Job `340894` 的 mean/max 与 Job `340895` 的未增强候选条件注意力命令，动态命令 SHA-256 分别为 `590e4af8133826d0aac0088814dfcc42c6936e00c21763743b5843f28138d1a2` 与 `295046288c5f67fa12aa289073bf8b1fcde0bde76db706d`，再于 09:31:09 一次删除两把 `try_lock`；两个 `after_lock` 未删除。两项均复用 release `Matcher_3cc75993d69e`。mean/max launch 为 `train_experiment_job340894_20260814T093012_a7`、W&B run `79immary`；未增强注意力 launch 为 `train_experiment_job340895_20260814T092853_a4`、W&B run `12oeeshn`。二者首个 global batch 32 更新分别耗时 `61.3182/61.9103 s`，裁剪前梯度范数 `27.0919/81.2901`，峰值保留显存 `39,439,040,512/39,634,075,648 bytes`；候选任务的身份监督数按契约为零。成功事件都在有限性检查、参数更新与 CUDA 同步后出现，未见 OOM、NaN、Traceback 或缓存回退。第 2 批五个非重复任务至此全部提交。

第 3 批实现、四份配置、测试与当时三层账本提交为 `495e7de54dca88ab3464fe0a667dcce5a540ef72 feat(batch3): add zero-initialized attention biases`。提交前 `git diff --cached --check` 通过，范围为 52 个预期文件；忽略目录中的临时部署脚本未进入提交。随后从该干净提交执行安全同步：

```powershell
& 'C:\Users\15919\Desktop\Matcher\与服务器交互\sync_code.ps1'
```

同步只增量更新 `/home/penghongen/My_Project/Matcher`，不使用 `rsync --delete`，没有修改任何 release、launch、实验产物或 allocation 锁。远端门禁脚本 `tmp/verify_batch3_remote_20260814.sh` 的最终 SHA-256 为 `ca956957c12434065fe922763ff07fe9ef7fe01439eab704dd4d91426cfa43a6`。第一次远端尝试在文件哈希通过后因 Conda 激活脚本与 `set -u` 的 `ADDR2LINE` 未定义变量冲突而停止；该尝试没有创建 release 或触发任务。脚本仅在 Conda 激活期间暂时关闭未定义变量检查后，以以下控制端命令通过：

```powershell
& "$env:USERPROFILE\.codex\tools\Invoke-ProjectSsh.ps1" `
  -Command 'bash -s' `
  -InputFile 'C:\Users\15919\Desktop\Matcher\tmp\verify_batch3_remote_20260814.sh'
```

门禁逐项核对 `contracts/experiment_config/tracking/training/candidate_encoder/config/fusion/schema` 八个源码文件和四份 Batch3 YAML 的 SHA-256；正式 Conda 环境还确认四项均使用 W&B online、project `AdaLigand-Matcher-Batch3`、`v_centroid`、4/4 workers、增强候选条件注意力和 global batch 32，候选任务为 A800 `16×2` 且身份损失权重为零，三项完整 Matcher 为 A100 `8×4`、SMI-TED 289M、五类非背景身份和 oracle/detached/joint。

通过门禁后执行：

```bash
bash '/home/penghongen/My_Project/Matcher/训练与运行/runtime/create_release.sh' \
  '/home/penghongen/My_Project/Matcher' \
  '/home/penghongen/Feedback/Matcher/releases'
```

得到预冻结 release `/home/penghongen/Feedback/Matcher/releases/Matcher_d15132948a18/Matcher`，manifest 内容摘要为 `d15132948a1885bae9a2019ac55175503c5c15459f043145147fe4a6d3598f89`，创建时间为 `2026-08-14T09:50:02+0800`。同一远端门禁随后以该 release 为项目根再次完整通过。此时只完成第 3 批发布准备，没有安装动态运行命令、删除 `try_lock`、删除 `after_lock` 或提交 GPU；四项仍按自然空闲资源与批次优先级等待。

09:57 进行一次事件式只读核验。六个受管 allocation 均为 `RUNNING`，每项 `after_lock` 存在，`try_lock/kill_lock` 均不存在；因此没有放行第 3 批。第 1 批 joint 的 epoch 9 再次改善，selection score 为 `0.6928435444`，六分类 macro-F1、五类固有 Top-1、精确子类条件 Top-1、宽松五类端到端 F1 和小分子兼容端到端 F1 分别为 `0.5193692059/0.8526603001/0.8971229294/0.6620384979/0.6646646647`。第 2 批三个完整 Matcher 在 epoch 3 的 selection score 为 oracle `0.6624755578`、detached `0.6739567888`、joint `0.6543584471`，三轮均标记为改善；五类固有 Top-1 分别为 `0.8253751705/0.8431105048/0.8437926330`，宽松端到端 F1 分别为 `0.6284892086/0.6444572085/0.5985090522`。第 2 批 mean/max 与未增强候选条件注意力完成 epoch 0 validation，macro-F1 分别为 `0.5234197383/0.4935270492`；二者首轮 nucleotide_like/peptide_like F1 均为零。全部任务继续运行，本次快照只作为曲线与吞吐证据，不据首轮或单指标提前判定改动方向。

### 第 3 批受保护接续控制器

为了在任一受管 allocation 自然出现 `try_lock` 时立即按批次优先级接续，同时不把“检查完成”和“删除锁”之间的半触发窗口变成重复提交，建立两个一次性控制器：A100 为 `tmp/batch3_launch/activate_next_a100.sh`，SHA-256 `6938a27fe13a30a0121b2f790e40b195b44b9edc8b5988f5ee165b9817c80731`；A800 为 `tmp/batch3_launch/activate_candidate_a800.sh`，SHA-256 `71eceb2ccc71c64502c72da17e8e344fcbd748777643812608c8ec892fba2bd7`。前者只接受 Job `340893/340897/340898/340899` 并维持 oracle→detached→joint 的全局已提交前缀，后者只接受 Job `340894/340895` 且只允许增强六分类一次。

第一轮只读审查发现四类阻断：不同 Job 并发可能重复选择同一目标；原子替换 `run_cmd` 后、删除 `try_lock` 前退出时不能恢复；只凭阶段 `summary.json` 存在不能证明 W&B 与整条运行命令正常收口；allocation runner 在解锁后重新选择 release，可能使实际执行版本偏离审查版本。严格 Bash 审查还发现命令替换 stderr 污染、Python `assert` 可被优化模式跳过、grep 读取错误失败开放和资源字段子串匹配。控制器随后集中修订，而不是在 GPU 锁上边试边改。

最终状态机在全部全局读取、动态命令替换和锁删除之前，共同持有 `/home/penghongen/Feedback/Matcher/batch3_activation.lock`。它从当前 Job 最新不可变 `launch.json` 的冻结 release 读取前序配置，要求运行根 `summary.json` 与内嵌阶段摘要一致，BEST/LAST 文件存在，epoch、global step、selection score、W&B run ID 与 URL 有效；A100 还会检查其它受管 Job 是否留有已安装但未触发的第 3 批命令。若当前 Job 已安装目标命令但 `try_lock` 仍存在，控制器只在重新生成的完整命令逐字节一致时恢复删除该锁；否则失败关闭。

生成的动态命令在进入训练前断言 `TASK_PROJECT_ROOT=/home/penghongen/Feedback/Matcher/releases/Matcher_d15132948a18/Matcher`，从而使 allocation runner 即使在解锁后看到可变项目根的新内容，也只会安全失败而不会用错 release。控制器唯一持久 Job 控制文件替换是当前 Job 的 `run_cmd`，唯一受管锁删除是当前 Job 的 `try_lock`；`after_lock/pre_lock/kill_lock` 和其它 Job 均不修改，calibration 入口与数据均未引用。

最终两份脚本分别通过独立科学部署终审和严格 Bash/技术表达终审。验证覆盖 `bash -n`、UTF-8/LF、内嵌 Python 编译、冻结源码与四份 YAML 的 SHA-256，以及不执行真实脚本的合成正常安装、A100 三种前缀、A800 唯一提交、跨 Job 半触发、空洞、重复和恢复探针；两项结论均为 PASS，阻断项为零。共享文件系统的跨进程 `flock` 语义没有用真实服务器控制锁做破坏性探针，作为运行环境假设保留；W&B 完成证据是 `finish(0)` 返回后才发布的本地运行根摘要，不额外查询远端 API。

2026-08-14 10:44:47 再次只读核验六个受管 Job：全部为 `RUNNING`，每项 `after_lock=1`，`try_lock/pre_lock/kill_lock=0`。因此两个控制器均未执行，没有安装第 3 批命令、删除锁、同步服务器或读取 calibration。修订说明：本节只记录未来自然空闲时可恢复且失败关闭的接续协议，不把第 3 批四项记作已提交。

为落实低频监控而不让当前任务依赖人工轮询，随后创建 30 分钟 heartbeat `matcher-1-5`，目标仍是本 active Goal。其持久提示要求每次先重读目标草案与本批 handoff，只读检查六个白名单 Job；状态无变化时不写日志或重复通知，只有自然出现 `try_lock`、完整 validation、自然收口或失败时才执行相称动作和回填记录。该 heartbeat 不是 Slurm 或训练时间上限，不允许删除 `after_lock`、释放 GPU、触碰用户任务或执行 calibration 模型推理。

10:50 的第一次后续事件检查没有出现 `try_lock`，但六项都产生了新的完整 validation。第 1 批 joint 的 epoch 10 selection score 为 `0.6841031473`，没有刷新 epoch 9 BEST `0.6928435444`；坏验证轮数为 1，任务继续。第 2 批 oracle/joint 在 epoch 4 改善至 `0.6699438211/0.6840046283`，detached 的 epoch 4 回落至 `0.6594240440`、BEST 仍是 epoch 3 的 `0.6739567888`。第 2 批 mean/max 六分类 BEST 为 epoch 1 的 `0.5303632887`，未增强交叉注意力在 epoch 2 改善至 `0.5146977235`。

当前第 2 批三个完整 Matcher BEST 相对第 1 批同损失约低 `3.29%/2.82%/1.28%`；mean/max 与第 0 批同聚合几乎相同，未增强交叉注意力低约 `2.32%`。没有多个核心指标共同约 10% 下降，不提前判负或改任务。

第 5 批证据审计直接读取现有 W&B 训练历史，不启动新 run。`tmp/batch5_gradient_audit.py` 的 SHA-256 为 `70b1c3e4b85586d11f24df3cbf3391188beca1d97b05a32d48e0043ea094ff4b`；六项分别得到 1,575–5,369 个 `gradient/train_step/preclip_norm`，其中 `97.52%–99.93%` 严格大于阈值 `1.0`，P50 为 `2.44–6.56`，P95 为 `5.72–16.31`。裁剪几乎每步生效已经确认，但当前结果没有证明其有害；后续最多从完整 Matcher 分布选一个替代阈值。

`tmp/batch5_branch_audit.py` 的 SHA-256 为 `39fc36660b1e3b2ab4031f9209aba63f7c856a0b6bf3e6d1f3cdb60ddaba040a`。四个完整 Matcher 最新 graph/inventory 门值为 `0.0166–0.0332/0.0136–0.0225`，裁剪后分支梯度 P50 为 graph `0.00203–0.00757`、inventory `0.000928–0.00204`。这证明严格零门没有锁死分支；但缺少门控后分数增量相对语言表示基线身份分数的贡献比例，所以第 5 批先补贡献诊断，非零门值优先考虑真实学习量级附近的 `0.02`，不把 `0.1` 当作无条件默认。详细表格与只读命令写入 `文档/记录/批次记录/第5批次/第5批证据驱动优化.md`；没有提交第 5 批任务。

### 第 4 批候选预训练 BEST 初始化

第 4 批的具体实验组合仍必须等待第 1–3 批 validation 全部自然收口。为避免资源空闲后才临时拼接 checkpoint，本轮先实现不绑定具体任务的初始化科学契约；没有创建第 4 批 YAML，没有同步服务器、发布 release、提交 GPU 或运行 calibration。

`matcher/initialization.py` 要求来源为绝对路径 `BEST.pt`，并在语义层交叉核对 checkpoint、阶段 `resolved.json`、阶段 `summary.json`、运行 `run.json`、运行根 `summary.json` 和训练 YAML。校验覆盖来源任务类型、epoch/best_epoch、global step、selection score、W&B online 收口、科学数据字段和候选编码器结构；完整 Matcher 独有的图网络层数以及 worker/prefetch/cache 等性能字段不作为来源—目标候选编码器兼容条件。来源 YAML 与 resolved 都必须显式证明 `initialize_candidate_from=null`。参数名、形状与 dtype 全部核对通过后，只对目标 `model.candidate_encoder` 执行严格 state-dict 加载；该步骤发生在模型迁移到设备和 optimizer 构造之前。

最初审查建议把来源 `launch.json`、`run_cmd.sh`、release manifest、训练文件 SHA-256 和同字节解析/摘要写入正式初始化记录。用户随后明确引用 `python-writing-style-cn` 裁定：这些属于一次性运维取证，不得进入训练包或主线。本轮因此删除新建的正式摘要模块和训练清单指纹链；`resolved.json.initialization` 只保留来源语义字段与加载结果。选定实际来源 BEST 后，才运行 `tmp/batch4_source_audit/audit_candidate_pretrain_source.py`，把 checkpoint、训练记录、launch、release 和训练 YAML 的路径关系及 SHA-256 报告写入 `/storage/penghongen/tmp/Matcher/diagnostics/`，并在单任务与批次记录中留证。临时脚本不被正式包导入，不读取 calibration，也不参与模型、损失或选模。

历史产物恢复只按字段缺失触发：三个模型特征开关缺失时迁移为 `false`，`local_crop_strategy` 缺失时迁移为 `fixed_center`，候选阶段 `identity_scope` 缺失时迁移为 `null`。正式 YAML 解析仍要求全部字段显式存在。测试使用候选子模块和非候选哨兵，证明成功路径只改变候选编码器且不推进 Torch 随机状态；失败路径在参数变更前拒绝非 BEST、数据/结构错配、未完成运行、训练 YAML 漂移、非从零来源和不兼容 state-dict。

职责收缩后的初始化与配置专项为 `34 passed`。最终候选通过 pytest `103 passed`、mypy 40 个正式源码文件零问题、pyflakes、compileall、19 份正式 YAML 解析、8 个 shell 的 `bash -n`、临时运维脚本编译和 `git diff --check`；科学正确性与严格技术表达两项独立复审正在以该快照为准重新执行。结论出来前不提交、同步或发布这一实现。

### 第 3 批实际接续与第 4 批初始化收口

2026-08-14 15:17:17–15:23:21，四个第 3 批非重复任务按 oracle→detached→joint→增强六分类的既定顺序全部接续。完整身份如下：

| 任务 | Job | launch | W&B run ID | 冻结 release |
| --- | --- | --- | --- | --- |
| SMI-TED 增强注意力 oracle | `340893` | `train_experiment_job340893_20260814T151725_a3` | `4iqynr6i` | `Matcher_d15132948a18` |
| SMI-TED 增强注意力 detached | `340898` | `train_experiment_job340898_20260814T151821_a3` | `b3vua4zk` | `Matcher_d15132948a18` |
| SMI-TED 增强注意力 joint | `340899` | `train_experiment_job340899_20260814T151852_a3` | `z41ltqiv` | `Matcher_d15132948a18` |
| 六分类增强候选条件注意力 | `340894` | `train_experiment_job340894_20260814T152229_a8` | `7k0bcaek` | `Matcher_d15132948a18` |

A100 与 A800 的第一次控制尝试分别因为前序旧 YAML 缺少后续版本新增的 `local_crop_strategy` 与 `use_candidate_token_attention_bias` 而失败关闭；两次失败都发生在替换动态命令和删除 `try_lock` 之前。实际成功触发时只为前序旧 YAML 使用轻量定位读取，当前第 3 批目标 YAML 仍由正式解析器严格校验。实际执行版本摘要为 A100 `0f2713e2207e1948d96efd0133914b2ad6a035a09d2fe58b999a131b3b4df04d`、A800 `43b654aceb0d38e0b38388017d4fa270d1e509f13543281325ac1cb0caeb0d64`；触发后本地继续加固安全单级名称与固定实验根规范化，但没有热替换已经启动的 release。四个 Job 的 `after_lock` 全部保留，未创建或触碰 `kill_lock`，未读取 calibration。

自动化 `matcher-1-5` 曾按 30 分钟规则创建，但随后被暂停。2026-08-14 本地持久配置明确为 `status = "PAUSED"`；当前没有活动 heartbeat，后续只在首步、完整 validation、自然收口或失败等明确事件发生时人工检查。

第 4 批初始化随后按用户裁定继续收缩职责：正式包只保存来源运行的语义自洽、候选结构和严格加载，不保存文件哈希或 launch/release 字节证据。一次性取证只留在 `tmp/batch4_source_audit/audit_candidate_pretrain_source.py`，报告只能写入 Matcher diagnostics；脚本不被正式包导入，也不参与训练或选模。正式初始化要求来源 YAML 与 `resolved.json` 显式保存候选任务的 `language_family=null`、`identity_variant=null` 和 `initialize_candidate_from=null`；来源 BEST/LAST、训练 YAML 与阶段/运行摘要必须是语义自洽的绝对路径。共享 checkpoint loader 还在目标参数变化前拒绝非字符串参数名或非张量参数值。

第四批单任务记录入口已建立在 `文档/记录/单任务记录/第四批次/README.md`。当前明确记录 0 项任务已决定、0 项已提交；后续每项任务必须在触发 GPU 前单独保存科学变量、YAML、实现提交、release、一次性来源审计、Slurm、W&B、首步、validation 和对照结论。

用户另行要求整理 `matcher/` 的目录布局。该工作不与第 4 批初始化混入同一提交：初始化基础先完成独立门禁和提交；随后在 `talk/refactor/` 建立全部正式文件的职责、调用者、生命周期、测试与目标位置清单，再以行为等价为唯一边界移动数据、加载、训练和评估模块。该结构整理不改变活动 Goal；第 1–5 批实验和六卡优先级仍是主线。

当前第 4 批冻结候选的初始化与配置定向测试为 `49 passed`，全套 pytest 为 `118 passed in 229.43s`。mypy 检查 40 个正式源码文件零问题；pyflakes、compileall、19 份正式 YAML 解析、8 个 shell 的 `bash -n`、临时审计脚本编译和 `git diff --check` 全部通过。科学正确性与严格技术表达两项独立复审均为 PASS，阻断项为零。

### Matcher 正式包职责整理

第 4 批初始化基础完成双线 Git 收口后，`Learn/CUMULATIVE` 位于 `a74deca9b7a3f6a58909410623ba1c8b99eb9625`。用户确认整理方案后，从该端点建立 `codex/matcher-package-layout`，并先提交 `talk/refactor/Matcher_包目录职责整理.md` 的全部正式文件清单。整理只改变模块位置、导入路径、四个 `python -m` 入口、十九份 YAML 的读取入口说明和当前结构文档。

新目录按 `data/datasets/model/training/evaluation/utils` 划分来源适配、Dataset 与 DataLoader、模型、训练、评估和非科学通用支撑。`evaluation.calibration` 可以复用 `training.engine.evaluate_model`；`training.engine` 只读取 `evaluation.metrics`，各子包入口不建立批量转发，从而不引入循环导入。旧顶层模块全部移除，不建立兼容包装；运行中的任务继续使用既有 release，不会读取本地移动结果。

以整理前累计端点为基准，对 36 个既有 Python 文件做语法树归一比较，移除导入与模块说明后全部等价。测试按文件串行得到 `test_data_pipeline=35`、`test_model_and_loss=28`、`test_metrics_and_config=24`、calibration/initialization/tracking 合计 `31`，总数精确为 `118 passed`；重型模型测试在当前本机负载下耗时约 10 分 34 秒。mypy 45 个源码文件零问题，pyflakes、compileall、19 份正式 YAML、8 个 Shell、四个新模块帮助入口、`spawn` 工作进程生命周期和 diff-check 均通过。第一次总套件与一次并行尝试因外层 5 分钟限制被工具终止，没有报告断言失败；它们不计为通过证据，最终只采用上述干净串行结果。

本轮没有访问 calibration 数据或运行模型评估，没有同步服务器、创建 release、安装动态命令、改动 Slurm 作业或删除任何锁。独立科学正确性与严格技术表达复核均为 PASS，阻断项为零。

实现端点提交为 `codex/matcher-package-layout@6176122`。学习历史从整理前累计端点开始，依次形成职责清单 `67b3a14`、运行模块与入口 `4aabd64`、测试导入 `1c5e99e` 和文档证据 `8d6ba38`；实现端点与学习端点的树摘要均为 `dd86b658cab8d0401248726563ff504b662a2c56`。逐文件内容完全一致后，`Learn/CUMULATIVE` 快进到 `8d6ba38`，临时学习工作树已删除。

### 第 1–2 批最终回填与第 3 批等待边界

2026-08-14 18:00，补读已经发布的 train/validation 运行摘要，不读取 calibration。第 1 批 joint 已于 12:14:17 自然早停，BEST/last epoch 为 `11/13`，selection score `0.6945667425`；第 1 批 oracle/detached/joint 三项最终分数最大差约 `0.26%`，各指标互有胜负，不能从单随机种子确定损失排序。

第 2 批五项均已自然早停：完整 Matcher oracle/detached/joint 的 BEST/last epoch 为 `11/13`、`5/7`、`6/8`，selection score 为 `0.6981048482/0.6864829756/0.6887556119`；相对第 1 批同损失分别变化 `+0.77%/-1.01%/-0.84%`。mean/max 与未增强交叉注意力候选预训练的最终 macro-F1 为 `0.5303632887/0.5272674639`，相对各自第 0 批约为 `-0.01%/+0.07%`。没有多个核心指标约 10% 的共同下降，且 oracle 改善分类、身份和区域，因此后续继续使用 V 质心单视野。逐类身份有效数、六分类逐类指标和混淆矩阵已回填对应单任务记录。

同一时点六个受管 Slurm allocation 均为 `RUNNING` 且保留 `after_lock`。Job `340895/340897` 已完成当前训练命令并由 `try_lock` 保留；Job `340893/340894/340898/340899` 继续第 3 批。增强六分类当前 BEST macro-F1 `0.5276366341`，与第 2 批未增强交叉注意力 `0.5272674639` 近似持平，但尚未自然收口；三个增强完整 Matcher 也仍在训练。因此候选预训练 BEST 来源和第 4 批任务矩阵继续等待第 3 批完整 validation，不为填卡跳过该比较或重复早期任务。
