# Matcher 单候选闭集身份分类重写执行记录

## 记录边界

本轮以 Matcher 提交 `6220ece3595e48dae9ee1ef749146efa1048305c` 为唯一实现基点，保留 Git 历史、通用 Slurm/release/launch/四锁、安全同步工具和旧文档证据，删除上轮活动 Dataset/Loader/模型/监督与 mode1 入口。旧科学实现不再映射到活动 Python 入口。

当前规划来源是 `C:/Users/15919/Desktop/AdaLigand/talk/talking/Matcher_单候选闭集配体身份分类新方案.md` 及本轮对话中已确认的修订。该规划会在代码、测试和首轮运行入口稳定后按实现回填，不用本记录反向篡改旧实验结果。

## 已冻结的科学契约

- 候选六分类顺序是 `background/ion/nucleotide_like/peptide_like/small_molecule/sugar`，所有 Find_0 F1 候选都参与，不使用配体语言表示或配体图。
- 候选全局组件与 occurrence 的双向覆盖率都至少为 `0.30` 才命中；多个命中按两侧覆盖率几何平均最大者选择，并使用稳定并列规则。监督使用完整全局候选，不使用裁剪后 V。
- U-Net 输入是候选中心 `48³` 的 105 通道：56 个 Find_0 密度通道加 49 个完整受体原子 hard-floor/sum 通道。A/P 保留 `80³` 完整上下文且不受 U-Net 调制；V/PP 截断到中心 `48³` 并只有这两个模态读取 U-Net 特征增量。
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
| `354b489` | 消除 BRANCHED 独立 CCD 参考坐标系产生的跨组件伪几何，并记录 v2 清单审计 |

## 阶段审查与修复证据

数据科学审查在真实 train/validation/calibration 产物中发现，旧的 binding-area 目标从筛选后 A token 散射，会漏掉局部结合原子的约 60%--75%。修复后目标独立使用完整 `receptor_tokens.coords` 和 `atom_labels.binding_atom`，世界 XYZ 按几何换算后使用 home-voxel floor 散射到中心 `48³`。实际纳入 PDB 的坐标/标签长度全部对齐：train 493、validation 183、calibration 86 个 PDB 都为零异常。

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
| `340896` | `matcher-smi-joint` | 1×A800 | SMI-TED Light 289M/joint | PENDING/Priority |
| `340897` | `matcher-molf-oracle` | 1×A100 | MoLFormer/oracle | PENDING/Priority |
| `340898` | `matcher-molf-detached` | 1×A100 | MoLFormer/detached | PENDING/Priority |
| `340899` | `matcher-molf-joint` | 1×A100 | MoLFormer/joint | PENDING/Priority |

## 资源、输出与监控

正式源码同步目标是 `/home/penghongen/My_Project/Matcher`。业务产物只能位于 `/storage/penghongen/tmp/Matcher/`；release、launch、Slurm 日志和四锁位于 `/home/penghongen/Feedback/Matcher/`，不属于业务产物根的限制对象。

候选不按 PDB 排列。DataLoader 固定 16 worker、`prefetch_factor=2`；A100 每个 worker 最多使用 16 GiB LRU，A800 每个 worker 最多使用 32 GiB LRU。这是每个 worker 的上限，不是整个训练进程的总预算；LRU 不预分配内存。

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

## 验证命令

```powershell
python -m compileall -q matcher
python -m mypy matcher
python -m pytest matcher/tests -q
git diff --check
```

模型审查修复时的阶段证据是 mypy 31 个源文件零问题、pytest `22 passed`、compileall 与 `git diff --check` 通过。训练、测评、配置与产物边界完成后的冻结结果是 mypy 32 个源文件零问题、pytest `33 passed`、compileall、7 个 shell 的 `bash -n`、七个 YAML 正式解析与 `git diff --check` 通过。BRANCHED 伪几何修复后的当前结果是 pytest `34 passed`、mypy 32 个源文件零问题、compileall 与 `git diff --check` 通过；三组结果都保留，不用新结果覆盖旧阶段证据。

## 待执行与待复核项

- v2 清单与首轮数据审计已经完成；BRANCHED 伪几何修复已通过独立科学与严格表达审查，待重新安全同步并冻结放卡前源码哈希。
- `pp_top_k=1024`、六分类 `class_weights=null`、区域辅助损失权重 `0.10/0.10` 和完整 Matcher validation 三指标等权选模是首轮显式起点，需根据清单摘要和首个稳定实验记录复核。
- GPU 资源实际分配后，删除 `pre_lock` 会直接启动正式命令；每个阶段的首个真实 global optimizer step 同时作为连续验收点。只有完成有限损失与梯度检查、参数更新和 CUDA 同步后，程序才打印一次 `first_global_step_accepted`，其中含 global batch、有效身份数、裁剪前梯度范数、峰值显存和首步计算耗时。这样不在验收与正式训练之间闲置 GPU；若 OOM 或非有限数值失败，程序不会打印成功事件，`after_hold` 仍会保留 allocation 供修订后重试。若 batch size 超出显存，按用户确认的顺序先将 RAUNet 通道降到原宽度的 3/4，再降低物理批量并增加梯度累积，始终保持 global batch 32。
- 首个完整训练若超过 16 小时，只暂停后续随机种子；不中止当前训练，不设任何时间上限。
