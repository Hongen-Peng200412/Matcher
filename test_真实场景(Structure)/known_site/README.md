# 已知真实位点的身份比较

本目录只负责三种方法共同使用的已知位点清单，以及两个已训练 Matcher 密度变体和官方 EMERALD-ID 的逐点身份预测。真实配体坐标只确定查询位置与评估答案，不提供候选配体构象。NxU 全配体和 nxu 小分子是两项独立任务：前者在当前 PDB 的全部目标 SMILES 中选择，后者只在小分子 occurrence 对应的 SMILES 中选择。受体条件也独立：原始真实受体与 CryoAtom2 最终预测受体在相同 nxu 清单上分别评估，不能互相覆盖结果。

## 输入和冻结清单

`build_manifest.py` 从 `test0_real_smiles_v1/manifest.json` 固定的 held_out_test_0 PDB 集合出发，读取 AdaLigand 的 `parse/{pdb}/occurrences.jsonl`、`parse/{pdb}/ligand_coords.npz`、`stage1_preparation_box_pool_2/ligand_language_models/prepared/{pdb}/prepared_smiles.jsonl` 和 `raw/pair_list.jsonl`。身份只采用 prepared 文件的顶层精确 SMILES；同一个 PDB 内重复字符串只占一个候选身份。`present` 只用于确定沉积原子的世界坐标质心，不定义身份。

每项任务各自按 PDB 计算 `occurrence 数 × 精确 SMILES 数`，成本降序、同成本 PDB ID 升序截去最高 `ceil(K% × 有查询点 PDB 数)` 个完整 PDB。正式 K=10。`manifest.json` 给出来源和两项任务汇总；每项下有三个逐行 JSON 文件：

| 文件 | 每条记录的含义 |
| --- | --- |
| `NxU/pdbs.jsonl`、`nxu/pdbs.jsonl` | 一个正式保留的 PDB。`identity_smiles` 是该任务此 PDB 的完整、排序后的精确候选字符串；`occurrence_count` 是查询点数；`docking_count` 是两者乘积；`emdb_id` 与 `resolution_angstrom` 分别定位 EMERALD 原始图和官方评分使用的全局分辨率。 |
| `NxU/queries.jsonl`、`nxu/queries.jsonl` | 一个正式查询点。`pdb_id` 和 `occurrence_id` 共同定位原始 occurrence；`center_xyz_angstrom` 是沉积 present 原子质心；`type_tag` 用于类别结果；`target_smiles` 只用于评估，不进入预测输入。 |
| `NxU/excluded_pdbs.jsonl`、`nxu/excluded_pdbs.jsonl` | 一个按成本截去的完整 PDB，记录 occurrence 数、候选身份数及两者乘积，便于复核截尾边界。 |

全部保留 PDB 是每项任务的主视图。多身份视图仅从同一批预测筛选 `identity_smiles` 长度大于一的 PDB；不重建候选集合、不再次推理、不再次截尾。两项任务的 PDB 集合不同，不可将两项准确率差异归因于单一候选范围变化。某身份无法准备或评分时，查询点和原始候选仍保留在清单和固定分母中。

## Matcher 前向

`predict_matcher.py` 从指定 BEST 的训练 `run.json` 找到训练 release，并从该快照恢复模型、密度构造、SMILES 图读取和批处理。程序读取完整重采样密度及每个查询点的世界 XYZ 质心，取查询所在体素为局部中心；48³ 超出完整网格处补零，不移动真实查询坐标。`density_only-0` 只对实验密度构造单个 exp_clipnorm_nopost 通道。`density_only-106` 同位置读取实验与模拟密度及受体坐标、49 维 L0 特征、backbone 标志，依训练函数生成 56 通道密度 bank 与 50 通道受体特征网格。两者均不读取 Stage1 blob、token 或区域标签。

`--data-root` 始终指向原始 AdaLigand Ori_Data，提供实验密度、网格几何和语言身份资产。`--receptor-root` 只改变 106 通道模型读取的 `density/{pdb_id}/sim.npy` 和 `parse/{pdb_id}/receptor_tokens.npz`；省略时两者仍从原始 Ori_Data 读取。CryoAtom2 条件取 `/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/test_0_chain06/Ori_Data`，其中 84 个 nxu PDB 的受体 token 与模拟密度均由最终预测 CIF 生成。`--task-names nxu` 使这次只读取 84 PDB／357 点，不前向 NxU。CryoAtom2 适配根的 `exp.npy`、`exp.npz` 指向原始文件；单通道模型不读取受体或模拟密度，因此在核对这两个链接、相同清单、身份资产与 checkpoint 后，可直接复用真实受体 nxu 的 357 个冻结预测，不生成第二套数值。

### nxu 查询点上的全部 SMILES 输入

`build_all_smiles_manifest.py` 只读取原 `manifest/nxu/pdbs.jsonl` 所列 84 个 PDB，保持原 `manifest/nxu/queries.jsonl` 的 357 个查询点及其答案。它从这 84 个 PDB 的全部 `occurrences.jsonl` 和顶层 `prepared_smiles.jsonl` 得到全类别身份轴；若同一 PDB 的同一精确字符串对应不同 `type_tag`，会停止冻结，不能静默决定小分子掩码。派生目录只有 `pdbs.jsonl`，每条记录对应一个原保留 PDB：

| 字段 | 类型与含义 |
| --- | --- |
| `pdb_id` | 字符串；与原 nxu 清单及 357 个查询点中的 PDB 编号相同。 |
| `identity_smiles` | 字符串列表，长度为该 PDB 全部 occurrence 的精确顶层 SMILES 去重数；按字符串排序，定义一次模型前向的完整候选身份轴。 |
| `identity_categories` | 字符串列表，与 `identity_smiles` 逐槽对齐；每个值是该精确 SMILES 所属的唯一 occurrence 类别。 |
| `small_molecule_identity_smiles` | 字符串列表，是完整身份轴中的小分子子集；与原 nxu 的 `identity_smiles` 逐字相同，作为前向之后的答案读取掩码。 |

例如同一 PDB 的完整身份轴为 `["C", "O"]`，类别为 `["small_molecule", "ion"]` 时，小分子身份轴为 `["C"]`。模型在 `C` 与 `O` 上只前向一次；不掩码读取两者最高 logit，掩码读取 `C` 的 logit，不把类别信息送入模型。正式预核算为 84 PDB、114 个小分子身份槽、164 个全类别身份槽；这些槽数是按 PDB 分别计数，同一字符串在不同 PDB 中分别占槽。

对 `predict_matcher.py` 提供 `--identity-manifest <派生 pdbs.jsonl>` 时，原 `--manifest-root` 仍提供冻结查询点，新文件只替换每个 PDB 的候选身份轴。输出根与旧小分子输入结果隔离。每个 `<输出根>/<模型标识>/nxu/<pdb_id>/<occurrence_id>.json` 对应原清单的一个查询点：`scored_smiles` 是完整身份轴中同时具有训练图与 SMI-TED 向量的精确字符串子集，`identity_logits` 与其逐槽对齐；`predicted_smiles` 是全部可评分身份中的最高 logit 字符串，`predicted_small_molecule_smiles` 只在其中小分子身份上读取最高 logit。若没有任何可评分身份，两个预测均为 null、`scored_smiles` 与 `identity_logits` 均为空；若仅小分子身份不可评分，后一个预测为 null。原小分子输入模式不提供 `--identity-manifest`，其既有逐点文件及字段不改变。两种答案均与查询真值精确字符串比较，身份资产缺失不能删去查询点。

每个点的结果保存在 `<输出根>/<实验标识>/<任务>/<pdb_id>/<occurrence_id>.json`。`predicted_smiles` 是模型最高身份 logit 对应的精确字符串；`scored_smiles` 是本点具有训练图和 SMI-TED 向量的候选身份，仍按清单字符串顺序；`identity_logits` 与其逐槽对齐。无可用身份时 `status=no_supported_identity`，预测为空，不能从统计分母删除该点。

## 官方 EMERALD-ID

`run_emerald_shard.py` 一次只接收 NxU 或 nxu 一项任务，在该项的 Slurm 数组中按理论身份评分数分配查询点；每个 8-CPU 数组元素最多同时启动 8 个单核进程。真实受体的 `run_emerald_small.sh` 与 `run_emerald_all.sh` 分别选择 nxu 和 NxU，不在同一次执行中混合查询点。`run_emerald_query.py` 为每个点独立准备完整的 PDB 内候选 SMILES、受体 ATOM 与质心虚拟原子、原始 EMDB 图、Rosetta 参数，然后一次调用官方 `multiple_ligands` 协议。官方 `rank_ligand_identity.py` 只读取该点的有效评分文件；不按裸 zscore 自行排名，不拼接不同点或拆分同一点的候选身份。固定 Rosetta 随机种子由官方 `-constant_seed -jran 3409` 参数提供，官方 XML 不改。官方参数生成器没有写出 `.params`、写出重复原子名或 MOL2 回读为空时，该精确身份记为准备失败；PDB 内原始候选集合和查询点分母不变，其余可准备身份仍可进入同点官方评分。

CryoAtom2 nxu 使用同一冻结清单和精确候选集合，`--data-root` 仍从原始 AdaLigand Ori_Data 读取未重采样 EMDB 实验图；`--receptor-root` 仅把 `raw/rcsb_mmcif/{pdb_id}.cif` 换成 CryoAtom2 适配根中指向 `latest.json::final_cif` 的最终预测受体。一个查询点的全部候选仍进入同一次官方评分。

每个点位于 `<输出根>/EMERALD-ID/<任务>/<pdb_id>/<occurrence_id>/`：`result.json` 保存评分状态、成功评分的精确 SMILES、准备失败原因及预测；`identity_map.json` 保存短配体名称到原始精确 SMILES 的映射；`official_rank.txt`、`.sc` 与 Rosetta 标准输出/错误保留原始评分证据。退出码为零并不代表有效评分；没有有效评分时 `predicted_smiles=null`。MOL2 保持原始重原子图、键级、形式电荷和组分；Rosetta 官方 `.params` 对已核实的共轭 C–N 键存在内部编码差异，解释见 [比较计划书](../参考材料%20%26%20想法/Matcher_EMERALD-ID%20比较计划书.md)。

## 命令与结果

`summarize.py` 只读取三方法逐点预测，不再次运行模型或对接。每次指定一项任务，为该任务生成 `all_retained` 和 `multi_identity` 两个视图。每个视图的 `summary.json` 含 PDB 数和查询点固定分母；每种方法下的 `correct`、`micro_top1` 是逐点正确数与比率，`pdb_macro_top1` 是先逐 PDB 计算正确率再对 PDB 等权平均。`categories` 按整体及五种来源类别分别给出正确数、支持点数和准确率；无支持类别的准确率为 null。`scored_identity_pairs / total_identity_pairs` 衡量候选身份评分覆盖，`scored_query_count` 统计至少一个身份得到有效分数的点数，`fully_scored_query_count` 统计该点全部原始候选均得到有效分数的点数。任一正式点缺少结果文件时，汇总停止，而不是悄悄缩小分母。每个视图的 `identity_metrics.png` 与同目录 JSON 一起保存。

CryoAtom2 nxu 汇总时，`--prediction-root` 指向独立的 CryoAtom2 输出根，读取其 density_only-106 与 EMERALD-ID 逐点文件；`--density0-root` 明确指向真实受体冻结输出根，仅让单通道 density_only-0 读取同输入的原预测。两种根中的三个方法仍按相同的 84 PDB／357 点和同一候选 SMILES 集合计算；输出写入 `CryoAtom2/summary/nxu/`，不覆盖真实受体 `summary/nxu/`。

`summarize_all_smiles.py` 只读取上述原 nxu 与新完整身份逐点文件，不加载模型或 Rosetta。它为每种受体条件生成一个独立 `summary.json` 和四张 PNG；两种条件共八张图。`summary.json` 的 `condition` 是 real 或 CryoAtom2，`query_count` 固定 357；`methods` 包含两个密度模型各自的小分子输入、不掩码完整输入、掩码完整输入及 EMERALD-ID，共七种模式。每种模式的 `fixed`、`own_complete`、`common` 各有 `correct`、`support`、`accuracy`：分别使用全部 357 点、该方法全部候选可评分点，以及七种模式共同完整可用的相同查询点。`own_complete_query_keys` 和顶层 `common_query_keys` 以 `pdb_id/occurrence_id` 保存实际纳入子集的冻结查询键；不是重新抽样的点。`grouped` 按原 PDB 的小分子身份去重数分成 `1、2、3、4、5、>5`，在每组重复上述三种分母的计数，空组准确率为 null。

每种受体条件的 `unmasked_overall.png` 和 `small_masked_overall.png` 各画 106 通道 Matcher 与官方 EMERALD-ID 的固定分母和各自完整子集四条曲线；`unmasked_paired.png` 和 `small_masked_paired.png` 各在完全相同的共同查询键上画两条曲线。横轴只来自原 nxu 的 `identity_smiles` 长度；可评分身份数变化不改组别。图中的 EMERALD-ID 基线始终是同一份原正式逐点结果，掩码不使官方方法再次运行。图旁所有完整模型与模式的数值应从同一 `summary.json` 读取，不从 PNG 读数。

`训练与运行/sh/known_site/` 内的各汇总脚本只读对应逐点结果，不再次评分。若 CPU 数组到达时间上限而仍有查询点没有 `result.json`，先核实所有工作进程已结束，再运行 `record_unfinished_small.sh`：它只为缺失点写入 `allocation_ended_before_result`，不覆盖已完成结果，也不从部分 `.sc` 文件拼凑身份排名。这个状态计入原固定分母，其评分覆盖数为零。简短正式提交命令写在该脚本目录各文件头；门禁命令和只读核查命令分别写在执行日志，不把一次性来源哈希塞入科学 JSON。总日志是 [`../执行记录/总日志.md`](../执行记录/总日志.md)，每个方法有同目录的独立单实验日志。
