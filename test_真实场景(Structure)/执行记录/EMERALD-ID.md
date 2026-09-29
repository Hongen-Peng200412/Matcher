# EMERALD-ID：已知位点身份比较日志

本日志记录官方 EMERALD-ID 在 NxU/nxu 给定真实配体位置、给定精确 SMILES 集合下的试运行与最终结果。依据是 [比较计划书](../参考材料%20%26%20想法/Matcher_EMERALD-ID%20比较计划书.md)；清单和固定种子门禁均已完成，当前先运行 nxu 小分子测试。

2026-09-25 仅整理本地位置：下述门禁代码现位于 `test_真实场景(Structure)/smoke/`；本日志中历史 Job 使用的服务器 `.../code/` 路径仍是真实来源，不随本地目录改名。服务器试运行输入、评分、排名及其它科学产物均未改动。

## 实验结果与产物

2026-09-29 新的 nxu 全部 SMILES 输入比较仅扩展两个 Matcher 的候选身份。EMERALD-ID **没有**重新准备、对接、评分或排名，继续复用真实受体与 CryoAtom2 的原 357 点正式小分子输入结果；其 PDB 内候选仍为原精确小分子 SMILES 集合。固定正确数分别为真实受体 233/357、CryoAtom2 263/357；全部小分子身份取得有效官方评分的自身完整子集分别为 182/190、214/222，恰与相应条件的七模式共同可用查询键相同。这里的“自身完整”是官方化学准备及有效评分，不与 Matcher 的图／语言资产可用性混称。八张新图中的不掩码与小分子掩码条件均画**同一份冻结 EMERALD-ID 基线**，不是两次官方运行。完整表见[扩展结果页](../../测试结果与追溯/测试结果总览/nxu全部SMILES输入比较.md)，只读复用与计算命令见[扩展任务日志](nxu全部SMILES输入.md)。原官方逐点目录未改动，不运行 NxU。

2026-09-29 CryoAtom2 nxu 官方对接与两视图正式汇总已收口。全部保留 84 PDB／357 点正确 263、micro Top-1 为 73.67%、PDB 平均为 82.69%；多身份 19 PDB／143 点正确 69、micro Top-1 为 48.25%、PDB 平均为 49.80%。全部点都有正式状态：313 scored、42 no_prepared_identity、2 no_valid_score；多身份视图为 119、22、2。旧受体链名格式错误的 22 点目录均保存在 `attempts/chain_name_pre_rosetta/nxu/`，从原冻结查询和完整候选身份补跑，最终没有 query_failed。正式命令分别是 `bash 训练与运行/sh/known_site/run_emerald_cryo_small.sh`、`bash 训练与运行/sh/known_site/retry_cryo_chain_names.sh`、`bash 训练与运行/sh/known_site/retry_cryo_chain_names_last.sh`；门禁命令及所有 release、launch、逐点根、汇总 SHA-256 另见[CryoAtom2 独立日志](CryoAtom2-nxu.md)，结果与图片见[结果页](../../测试结果与追溯/测试结果总览/nxu小分子CryoAtom2测试.md)。下文“正在运行”均为历史进度，不代表当前状态。两个受体条件的 NxU 都保持暂停。

CryoAtom2 受体条件的 nxu 小分子测试已通过官方单点门禁，并在 11 个旧 CPU allocation 中正式启动。固定清单仍为 84 PDB／357 点；每个查询点始终以完整 PDB 内候选 SMILES 集合调用官方评分和排名。当前结果尚未收齐，不计算正式准确率；`result.json` 不存在的查询点不算评分失败。Job `390207` 的首次 Cryo 门禁因替代 mmCIF 缺少 Gemmi 的 `entity_type` 而在 Rosetta 前失败，保留原 scratch 与 release/launch；仅针对替代受体跳过 `remove_ligands_and_waters()` 且保留 PDB `ATOM` 行过滤后，同一 Job 的重跑门禁得到两个身份的有效 `.sc` 评分并 PASS。正式代码 release 是 `/home/penghongen/Feedback/Matcher/releases/Matcher_6e7096a3a92c/Matcher`，独立任务输出根为 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/CryoAtom2/EMERALD-ID/nxu/`，实际 Job、完整 launch、正式命令与门禁命令分别见 [CryoAtom2 任务日志](CryoAtom2-nxu.md)。下述真实受体 NxU 的中断状态不因此改变。

上述首次 CryoAtom2 CPU 批次在三个预测受体 PDB 的三字符链名上发现代码输入格式失败，绝不把旧 `query_failed` 当作方法的有效评分失败。实际官方 `9nc9/0` 门禁表明：仅将超长预测链名重编号为唯一两字符还不够，必须保留 PDB 虚拟查询链 `X`，否则 Rosetta 把两者混合。最终修订在隔离目录取得 `scored` 后，21 个旧错误点目录已原样移入 `attempts/chain_name_pre_rosetta/nxu/`，并在新 release 的 Job `390383` 中从同一冻结清单派生队列正式补跑。仍未完成的旧批次点不计评分失败；补跑的暂时进度和两次门禁失败链均见 [独立日志](CryoAtom2-nxu.md)。

2026-09-28 21:36 真实受体 NxU 按用户新顺序安全中断，等待未来继续。数组 `390206` 的 12 个实际 Job、launch 和 release 见下方正式启动记录；中断前它们均为 RUNNING，每个 Job 的精确 `after_lock` 存在，而 `try_lock`、`pre_lock`、`kill_lock` 不存在。逐一创建对应 Job 目录中的精确 `kill_lock` 后，12 个 Job 全部恢复 `try_lock`、清除 `kill_lock` 且保留 `after_lock`；gnode08 与 gnode11 无残留 `run_emerald_shard.py` 或 `rosetta_scripts_430` 进程。正式逐点根保留 317/995 个完整 `result.json`，其中 122 有有效评分、101 无准备成功身份、94 无有效评分；其余 678 个查询点不是失败记录，未来续跑须按清单逐点核对，不能从本次部分结果计算正式 NxU 准确率。此中断尝试仍归属同一个 EMERALD-ID 实验日志，不覆盖 nxu 小分子结果。下一步只测试 CryoAtom2 受体条件的 nxu；NxU 两条件均待用户后续通知。

2026-09-28 18:26 NxU 全配体阶段已在小分子收口后启动。旧数组 `388458` 的 12 个实际 Job 在 try_lock 状态核对完成后，仅删除各自精确 after_lock 并正常退出，旧 NxU 正式目录没有逐点结果。新数组基础 Job `390206` 已直接进入 `debug/Cpu96`；12 元素各 8 CPU、72 小时、`after_hold`，实际 Job 下标 0–10 依次是 `390207`–`390217`，下标 11 是 `390206`。共用正式 release `/home/penghongen/Feedback/Matcher/releases/Matcher_09b9eb624d78/Matcher`，正式运行短命令为 `bash 训练与运行/sh/known_site/run_emerald_all.sh`；正式提交命令为 `bash 训练与运行/submit_task.sh --sh 训练与运行/sh/known_site/run_emerald_all.sh --resource cpu --cpus 8 --array 0-11%12 --partition debug --qos Cpu96 --time 72:00:00 --after_hold --job-name known_emerald_NxU`。逐点根为 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/EMERALD-ID/NxU/`，清单 SHA-256 和输入快照与 nxu 相同。12 个实际 Job 均已在自己的 allocation 内输出 shard 计划，总计 995 点、1803 个理论身份评分；这仅确认任务启动，未产生 NxU 正式准确率。

| NxU 数组下标／实际 Job | 正式 launch，公共前缀为 `/home/penghongen/Feedback/Matcher/launches/` | 查询点／理论身份评分 |
| --- | --- | ---: |
| 0／390207 | `390207/run_emerald_all_job390207_20260928T182134_a1` | 83／151 |
| 1／390208 | `390208/run_emerald_all_job390208_20260928T182134_a1` | 83／151 |
| 2／390209 | `390209/run_emerald_all_job390209_20260928T182134_a1` | 83／151 |
| 3／390210 | `390210/run_emerald_all_job390210_20260928T182134_a1` | 82／150 |
| 4／390211 | `390211/run_emerald_all_job390211_20260928T182134_a1` | 83／150 |
| 5／390212 | `390212/run_emerald_all_job390212_20260928T182134_a1` | 83／150 |
| 6／390213 | `390213/run_emerald_all_job390213_20260928T182134_a1` | 83／150 |
| 7／390214 | `390214/run_emerald_all_job390214_20260928T182134_a1` | 83／150 |
| 8／390215 | `390215/run_emerald_all_job390215_20260928T182134_a1` | 83／150 |
| 9／390216 | `390216/run_emerald_all_job390216_20260928T182534_a1` | 83／150 |
| 10／390217 | `390217/run_emerald_all_job390217_20260928T182534_a1` | 83／150 |
| 11／390206 | `390206/run_emerald_all_job390206_20260928T182534_a1` | 83／150 |

NxU 首次执行的 release、launch、GPU=无、worker 为每元素 8 个单核进程，均由标准入口的 launch.json 和 allocation 标准输出留证。测试和门禁命令仍在本日志后部单独列出，不混入正式命令。下次只在逐点覆盖发生明确变化或出现故障时更新状态快照，不按每点记流水账。

nxu 小分子阶段正式收口：84 PDB／357 点中 233 点预测身份正确，逐点 micro Top-1 为 65.27%，PDB 等权平均为 75.79%；其中多身份的 19 PDB／143 点正确 70 点，micro Top-1 为 48.95%，PDB 等权平均为 48.22%。正式 357 个 `result.json` 中 282 个 `scored`、42 个 `no_prepared_identity`、33 个 `no_valid_score`，没有无记录的查询点；后两类仍在固定分母。至少一个身份有分数的点数为 282/357，全部身份有分数的点数为 190/357，身份评分对为 365/588；多身份视图分别为 119/143、27/143、202/374。正式汇总 Job `388458` 成功，release `/home/penghongen/Feedback/Matcher/releases/Matcher_09b9eb624d78/Matcher`、launch `/home/penghongen/Feedback/Matcher/launches/388458/run_emerald_job388458_20260928T180709_a4`；正式汇总命令 `bash 训练与运行/sh/known_site/summarize_small.sh`。两视图的 JSON、原样 PNG 和 SHA-256 见 [小分子结果页](../../测试结果与追溯/测试结果总览/nxu小分子真实测试.md)及[共享总日志](总日志.md)。先前段落中的“仍在运行”“尚无正式准确率”是当时状态，不代表当前。NxU 尚未开始官方对接。

2026-09-27 22:44 长尾状态：nxu 的 357 个固定查询中 341 个有 `result.json`，16 个仍在实际 Rosetta 计算中；数组 walltime 已接近 48/72 小时。部分长尾标准输出仍在更新，不能把“尚未完成”写作评分失败。未改变任务科学参数、未删除 `after_lock`，也未运行 NxU；关于是否提前设明确超时已向用户提出非阻塞问题，当前仍按已提交的 72 小时上限执行。

2026-09-27 09:51 连接恢复后的只读状态：nxu 已有 338/357 个逐点 `result.json`；状态分别为 `scored` 263、`no_prepared_identity` 42、`no_valid_score` 33。余下 19 个点仍无结果文件，其目录中的 Rosetta 标准输出表明至少部分仍在继续写入，不能把暂时缺文件当作最终评分失败。5 个数组元素已经回到 `try_lock`，其余 7 个继续执行；所有 `after_lock` 保留。此前连接中断没有停止或修改这些任务。后续须等所有点完成，或对确实终止而未写出结果的点建立明确失败记录，才可运行 `summarize_small.sh` 并以固定 357 点收口。

2026-09-26 16:41 连接间断：最后一次成功只读检查显示 nxu 有 192/357 个 `result.json`，所有 12 个正式数组元素仍在运行，NxU 目录尚不存在。此后本机连接服务器的 TCP 在 SSH 认证前超时，无法取得更新的逐点状态；没有因此改动进程、锁或产物。恢复连接后须先区分仍在计算的点、已完成且有效评分的点，以及有明确失败记录的点，不能把暂时无法访问误记为评分失败。

2026-09-26 09:52 状态：12 个 CPU 数组元素已触发正式 nxu，固定分母 84 PDB／357 点，多身份视图 19 PDB／143 点。每个查询点保留完整 PDB 内精确 SMILES 候选集合；12 个 shard 的查询点合计 357、理论点—身份评分合计 588。共同代码 release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_491e58a92021/Matcher`，正式逐点根为 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/EMERALD-ID/nxu/`，正式运行命令单独记录为 `bash 训练与运行/sh/known_site/run_emerald_small.sh`。NxU 的正式逐点目录尚不存在；小分子的逐点结果、两个视图和文档收口前不会启动 NxU。当前逐点任务在运行，不把未完成点算作评分失败，不提前生成三方法总准确率。

| 数组下标／实际 Job | 本次 nxu launch，公共前缀为 `/home/penghongen/Feedback/Matcher/launches/` |
| --- | --- |
| 0／388530 | `388530/run_emerald_job388530_20260926T094808_a4` |
| 1／388531 | `388531/run_emerald_job388531_20260926T094808_a4` |
| 2／388532 | `388532/run_emerald_job388532_20260926T094810_a3` |
| 3／388533 | `388533/run_emerald_job388533_20260926T094808_a3` |
| 4／388534 | `388534/run_emerald_job388534_20260926T094808_a3` |
| 5／388535 | `388535/run_emerald_job388535_20260926T094808_a3` |
| 6／388536 | `388536/run_emerald_job388536_20260926T094808_a3` |
| 7／388537 | `388537/run_emerald_job388537_20260926T094808_a3` |
| 8／388538 | `388538/run_emerald_job388538_20260926T094808_a3` |
| 9／388539 | `388539/run_emerald_job388539_20260926T095209_a4` |
| 10／388540 | `388540/run_emerald_job388540_20260926T095216_a5` |
| 11／388458 | `388458/run_emerald_job388458_20260926T095228_a3` |

旧的两次混跑尝试完整保存于同一评估根的 `attempts/formal_attempt_1/` 和 `attempts/formal_attempt_2_mixed/`，没有并入本次 nxu 正式输出。第一次发现官方参数生成返回零但缺失 `.params`、Open Babel 回读空 MOL2；第二次在 NxU 的 `9uat` 发现官方 `.params` 重复原子名导致 Rosetta 无有效评分。仅针对这些已观察到的错误做最小输入适配，随后在实际 CPU 上分别通过缺失参数、空 MOL2 和重复原子名门禁。重复原子名门禁使用 Job `388540` 第 4 次执行，release `/home/penghongen/Feedback/Matcher/releases/Matcher_491e58a92021/Matcher`，launch `/home/penghongen/Feedback/Matcher/launches/388540/run_emerald_job388540_20260926T094808_a4`；它是门禁命令，不是正式评分。两次混跑均通过精确 `kill_lock` 结束，12 个 allocation 的 `after_lock` 始终保留。第二次混跑中断时有 101 个点目录、5 个 `no_valid_score` 结果文件及 96 个未完成目录，故不能当成正式评分覆盖率。

2026-09-26 当前状态：正式 K=10 清单由 Job `388464` 在 gnode11 建立并完成来源复核，`manifest/manifest.json` 的 SHA-256 为 `095f1bdc6b76a98374eca197e7a179bccc1dfe244d66ffcdefa941d112cde211`；七个清单文件与先前独立门禁逐字节相同。NxU 161 PDB／995 点、多身份 72／540；nxu 84 PDB／357 点、多身份 19／143。清单 release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_42d32da3a160/Matcher`，launch 为 `/home/penghongen/Feedback/Matcher/launches/388464/build_manifest_job388464_20260925T214201_a1`。CPU 对接数组 `388458` 的 12 个实际 Job 各占 8 CPU，均已通过标准 Matcher 入口留存 release/launch。固定种子复现门禁 Job `388465` 在同一 allocation 的第 2 次执行正式 PASS：两次 `9nal/0` 均对两个身份完成评分，官方排名文件 SHA-256 同为 `594d6573be401dd128e2821d648ff4d4dcf28e2a55854336df040b8d9a339de0`；该门禁 allocation 已通过精确 `after_lock` 释放。数组首次执行锁随后才逐个解除。正式对接第一轮发现官方参数生成器对部分单原子离子返回零退出码但没有写出 `.params`，原代码误将不存在的参数交给 Rosetta；另一个复杂铁硫身份的 Open Babel 回读为空，原代码将其变成整点 `IndexError`。这两项是输入适配错误，不得记为方法科学失败。12 个 Job 均通过精确 `kill_lock` 停止第一轮命令并恢复 `try_lock`，没有释放 allocation；全部第一轮目录已完整移动保存在同一正式根的 `attempts/formal_attempt_1/`，其中 41 个点目录、17 个结果文件（16 个 `no_valid_score`、1 个 `query_failed`）和 24 个未结束目录均未覆盖，第一轮没有任何成功评分。正式逐点根已重新留空。最小修正只要求 `.params` 实际存在，并把空 MOL2 回读解释为该身份准备失败；在 Job `388530`、`388531` 各自隔离的实际 CPU 门禁完成前，不再次放行正式对接。输入快照 Job `388503` 已完成，1902 个文件的快照记录为同根下 `provenance/input_files.jsonl`，SHA-256 为 `5da73e14a6fed0ecb17134953c68eaebca2b4e2ce6ffe18d9d8e8846d89313cc`；其 release 和 launch分别为 `/home/penghongen/Feedback/Matcher/releases/Matcher_876ce2f79143/Matcher` 与 `/home/penghongen/Feedback/Matcher/launches/388503/snapshot_known_site_inputs_job388503_20260925T220535_a1`。单身份 PDB 参与统一成本截尾，不预先删除；高成本单身份 PDB 仍可能被截去。

当前没有正式 NxU/nxu 正确数、逐配体 micro 准确率或可发表图。三个历史 CPU smoke 均已结束。官方教程 Job `387990` 的 ADP、ATP、CLR、Y01 各有一条本次运行的有效评分；`9nal` 重试 Job `387998_0` 的 L00、L01 也各有一条，首位 L01 与该单点真值精确 SMILES 一致；`9nnz` 独立复核 Job `388007_1` 则重现环构象错误且没有 `.sc`，即使 Slurm 标为 `COMPLETED` 也不算评分成功。任何单点结果都不能填作完整 NxU/nxu 准确率；K=10 的预期正式分母为 NxU 995 点、nxu 357 点，最终以冻结清单为准。

本次 7bxu 教程计算历时 4 小时 14 分 12 秒，单核 CPU 时间 4 小时 6 分 28 秒，峰值驻留内存约 2.64 GiB。新生成的单个 `emerald-id_score.sc` 含四个身份各一条有效评分，官方排名为 ATP、ADP、CLR、Y01，概率分别为 0.457124、0.451823、0.084574、0.006479，缺失身份为空。作者教程目录随附的旧参考 `.sc` 也能通过同一排名脚本，但其首两名次序是 ADP、ATP；两组文件从始至终分目录，不能把它们当作同一次运行或要求随机对接结果逐位相同。这份历史 smoke 当时尚未固定正式批量随机种子；当前方案使用官方 `-constant_seed -jran 3409`，复现门禁仍在进行。不从接近的前两名概率推断方法稳定性。

资源量级只作历史试点估算：`9nal` 的 2 个身份约 101.7 分钟，教程的 4 个身份约 254.2 分钟，平均每个成功身份约 51–64 分钟。当时机械乘以未截尾 NxU/nxu 共 12,286 个位置—身份组合，得到约 10,400–13,000 单核 CPU 小时；若最多 96 核始终满载，理想下界约 4.5–5.6 天。正式 K=10 清单的理论组合数已经降低为 2391，此旧估算不能冒充新预算。两种试点的分子、受体和运行路径不同，且 `9nnz` 无有效评分，所以外推不是预算承诺。完整 K=10 运行此后已获授权，但不能因糖等失败删减固定分母。

`9nal` 的这次 Rosetta 计算历时 1 小时 41 分 41 秒，单核 CPU 时间 1 小时 41 分 5 秒，峰值驻留内存约 2.26 GiB。当前查询点有 L00、L01 两个原始精确 SMILES；单个 `emerald-id_score.sc` 有一行表头和各身份一条完整分数，官方 `rank_ligand_identity.py` 的输出为 L01 概率 0.707079、L00 概率 0.292921，缺失身份为空。L01 的原始精确字符串与 `truth_for_gate.json` 完全相同。该结果证明此多身份小分子案例可以使用官方评分与排名链，但不能外推到其他结构，也不能替代严格化学输入门禁。

`check_params_graph.py` 从本地通过 SSH 标准输入作为一次性门禁执行，不改动服务器文件。L00 的 MOL2 与 `.params` 在元素、原子数、连接、键级和组分上同构；L01 均有 42 个原子、44 条键且元素与连接同构，但 MOL2 中芳香环外的 C–N 为单键，`.params` 中对应 `BOND_TYPE C5 N3 2`，同时 N3 仍连两个显式氢。已查服务器官方源码 `BasicClasses.py` 中 `BondClass.order_in_params`：被标记共轭的单键输出为 2；`SetupTopology.py` 的 `assign_bond_conjugation` 会作此标记。因此这是官方参数生成器的内部共轭表示，不是本项目在 SMILES→MOL2 时改变了候选，也不是名称映射丢失。用户已明确接受并要求披露这一已核实差异；不修改官方方法。严格的 `.params` 字面逐键级比较仍是 FAIL，原始 SMILES→MOL2 化学图门禁及 L01 的原子/连接核对是 PASS；不能把二者合称为“所有键级严格一致”。其他类型的未来差异不在本次许可范围内。

| 追溯项目 | 已核实位置或状态 |
| --- | --- |
| 官方教程源 | 本地 `test_真实场景(Structure)/参考材料 & 想法/tutorials/tutorials/EMERALD-ID/`；含 `7bxu.pdb`、`emd_30238.map`、4 个 `.params`、XML 和官方排名脚本 |
| Rosetta 执行入口 | `/home/penghongen/software/rosetta/bin/rosetta_scripts_430`；源码报告版本 `2026.20+release.b2a0ff3` |
| Rosetta 源码与数据库 | `/home/penghongen/software/rosetta/source_build/rosetta.source.release-430/main/`；其 `main/database` 是实际数据库目录 |
| 候选来源 | `/storage/penghongen/AdaLigand/Ori_Data/stage1_preparation_box_pool_2/ligand_language_models/prepared/{pdb}/prepared_smiles.jsonl` 的顶层 `smiles` |
| 原始图与受体来源 | `/storage/penghongen/AdaLigand/Ori_Data/raw/emdb_maps/`、`raw/rcsb_mmcif/` 与 `raw/pair_list.jsonl` |
| 教程输入副本 | `/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/tutorial_source/`；其中 `inputs/emd_30238.map` 的本地与服务器 SHA-256 均为 `50bde258ba42207fd7688bf8828c906f111320a959c40b09bc077bafe039481e` |
| 教程脚本副本 | `/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/smoke_tutorial.sh`；本地与服务器 SHA-256 均为 `718757ec0c922f8e052ba78a6db03962a51a3ff2d962c573ec40e450c1e61df3`；服务器 `bash -n` 已通过 |
| 教程 smoke 输出根 | `/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/tutorial_7bxu/`；提交时为空，后续由 Job 写入 |
| 项目门禁代码副本 | `/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/code/`；首次 `prepare_project_smoke.py`/`smoke_project_job.sh` SHA-256 为 `d191c7d58ff0e4678d80d07a69a979f474cb044a64932db2d7e872309a1c1473` / `78b5a567ece8d4da2c2439828cd057d1b619bd96a761f2471ed664fd0be10a3f`；重试版为 `5b9476d0bc5292874b168547aaccca1d113558f2bce4f199776c025102636285` / `a6ed7958415f8bc2fe8484db024900a1b9649d5d4bc149bffb8e8d30ea2f7d0d`；不变的 `smoke_project.sh` 为 `539218f2900e7fb1c30d9c685dc56e01994d4ee6dcb9a38b46385a53c950b69e` |
| 项目 smoke 输出根 | `/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/{9nal,9nnz}/occurrence_0/{small,all}/`；首次输入与运行在其 `input/`、`run/`，两个复核分别在各自 `attempt_2/input/`、`attempt_2/run/`；第一次失败目录保留 |
| CPU Job、launch、W&B | 教程 `387990`：1 CPU、8 GiB、最长 6 小时；项目数组 `387992_0`、`387992_1` 与两个复核 `387998_0`、`388007_1`：各 1 CPU、16 GiB、最长 12 小时。初始提交 `cpu/Cpu96` 排队，运行前改到 `debug/Cpu96`；复核直接提交 `debug/Cpu96`。直接 Slurm 门禁没有 release/launch，也不启动 W&B |

历史提交命令：Job `388458` 最初以 `bash 训练与运行/submit_task.sh --sh 训练与运行/sh/known_site/run_emerald.sh --resource cpu --cpus 8 --array 0-11%12 --time 72:00:00 --pre_hold --after_hold --job-name known_emerald` 申请；这是已经结束的混跑尝试所用入口，不是当前正式 nxu 命令。固定种子与清单门禁通过后，首次 `pre_lock` 才解除；本轮小分子正式运行改为本节前部记录的 `run_emerald_small.sh`，其 12 个实际 Job 各有独立 release 与 launch。

2026-09-25 排队调整：原 `cpu` 分区的 384 CPU 全部占用，故在尚未运行任何数组元素时仅执行 `scontrol update JobId=388458 Partition=debug`；未改变代码、清单、命令或评分契约。12 个数组元素随后全部在 `debug` 获得 8 CPU：下标 0–10 的实际 Slurm Job ID 依次为 `388530`–`388540`，下标 11 的实际 Job ID 为 `388458`。每个实际 ID 的首次执行锁位于 `/home/penghongen/Feedback/Matcher/allocations/pre_lock_<实际ID>`，退出锁位于 `/home/penghongen/Feedback/Matcher/allocations/<实际ID>/after_lock_<实际ID>`；已逐一核实 12 对锁均存在，`try_lock` 与 `kill_lock` 不存在。固定种子门禁结束前不删除任何首次执行锁，因此当前占用的 CPU 不是已经开始的对接。

第一轮正式尝试的共同代码 release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_876ce2f79143/Matcher`。每个 launch 均在 `/home/penghongen/Feedback/Matcher/launches/<实际 Job ID>/` 内：下标 0–1、3–7 的目录名为 `run_emerald_job<实际ID>_20260926T081504_a1`，下标 2 为 `run_emerald_job388532_20260926T081507_a1`，下标 8 为 `run_emerald_job388538_20260926T081505_a1`，下标 9–11 为 `run_emerald_job<实际ID>_20260926T081912_a1`。开始后发现并保存了本日志前述输入适配错误，12 个命令通过各自精确 `kill_lock` 终止并恢复 `try_lock`，没有删除任何 `after_lock`。第一轮的 41 个查询点目录完整保存在 `attempts/formal_attempt_1/`，不是正式方法覆盖率。新的缺失 `.params` 门禁使用 Job `388530` 第 2 次执行与临时根 `/storage/penghongen/tmp/Matcher/known_site_missing_params_gate_20260926/`；空 MOL2 门禁使用 Job `388531` 第 2 次执行与独立临时根 `/storage/penghongen/tmp/Matcher/known_site_empty_mol2_gate_20260926/`，两者均不是正式生产命令。修复后再次正式运行仍只使用上述短命令，并保留独立的新 release/launch。

测试与门禁命令：教程实际提交命令为下方代码块；Job 内只执行官方教程的 Rosetta 调用，不属于正式 NxU/nxu 命令。项目样本后续另查化学图、PDB 受体排除真实配体、每个查询点隔离 `.sc`、每成功身份一条有效分数，以及 `ligandname` 到原始 SMILES 的单值映射。

```bash
sbatch --parsable --job-name=emerald_id_tutorial_smoke --partition=cpu --qos=Cpu96 --cpus-per-task=1 --mem=8G --time=06:00:00 --output=/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/tutorial_7bxu/slurm-%j.out --error=/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/tutorial_7bxu/slurm-%j.err --wrap="bash /storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/smoke_tutorial.sh /storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/tutorial_source /storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/tutorial_7bxu"
```

项目样本数组元素 `0` 为 `9nal/candidate_id=0/nxu`，元素 `1` 为 `9nnz/candidate_id=0/NxU`。其门禁提交命令为：

```bash
sbatch --parsable --job-name=emerald_id_project_smoke --partition=cpu --qos=Cpu96 --array=0-1 --cpus-per-task=1 --mem=16G --time=12:00:00 --output=/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/project-%A_%a.out --error=/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/project-%A_%a.err --wrap="bash /storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/code/smoke_project_job.sh"
```

提交后改排的命令仅改变尚未运行的 CPU 分区，不改变输入、脚本、科学参数或 Job ID：

```bash
scontrol update JobId=387990 Partition=debug
scontrol update JobId=387992 Partition=debug
```

同一 `9nal` 实验的第二次门禁提交命令为：

```bash
sbatch --parsable --job-name=emerald_id_9nal_retry2 --partition=debug --qos=Cpu96 --array=0 --cpus-per-task=1 --mem=16G --time=12:00:00 --export=ALL,SMOKE_ATTEMPT=2 --output=/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/project-retry2-%A_%a.out --error=/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/project-retry2-%A_%a.err --wrap="bash /storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/code/smoke_project_job.sh"
```

同一 `9nnz` 实验的第二次门禁提交命令为：

```bash
sbatch --parsable --job-name=emerald_id_9nnz_retry2 --partition=debug --qos=Cpu96 --array=1 --cpus-per-task=1 --mem=16G --time=12:00:00 --export=ALL,SMOKE_ATTEMPT=2 --output=/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/project-9nnz-retry2-%A_%a.out --error=/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/project-9nnz-retry2-%A_%a.err --wrap='bash /storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/code/smoke_project_job.sh'
```

`9nal` 在 Rosetta 自然结束后另行执行的评分与官方排名门禁命令如下；它不重新对接，也不属于正式 NxU/nxu 命令：

```bash
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python /storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/code/check_smoke_scores.py /storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/9nal/occurrence_0/small/attempt_2/run/outputs /storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/tutorial_source/rank_ligand_identity.py /storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/9nal/occurrence_0/small/attempt_2/gate --mapping /storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/9nal/occurrence_0/small/attempt_2/input/identity_map.json
```

教程自然结束后的本次新输出门禁同样单独执行；`--expected-names` 是教程四个配体名称，不是 Matcher 的精确 SMILES：

```bash
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python /storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/code/check_smoke_scores.py /storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/tutorial_7bxu/outputs /storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/tutorial_source/rank_ligand_identity.py /storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/tutorial_7bxu/gate --expected-names ADP,ATP,CLR,Y01
```

MOL2 到 `.params` 的严格化学图门禁使用本地 `check_params_graph.py` 通过 SSH 标准输入执行，没有写入服务器；L00 通过，L01 因官方共轭键级编码差异未通过：

```bash
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python - /storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/9nal/occurrence_0/small/attempt_2/input/ligands
```

只读源码核查除前述教程和评分文件外，还检查了服务器 `generic_potential/BasicClasses.py` 的 `BondClass.order_in_params`、`SetupTopology.py` 的共轭键标记，以及实际 L01 的 MOL2 与 `.params` 键条目；没有修改 Rosetta 源码、参数文件或已有评分。

只读核查命令：已读取本地教程的 `run_EMERALD-ID.sh`、`ligand_screen.xml`、`rank_ligand_identity.py`，并确认服务器 Rosetta 包装器可执行、RDKit 2023.09.3 与 Open Babel 3.1.0 可用。`9nal/attempt_2/input/input.pdb` 有 9,367 条受体 `ATOM` 和恰好 1 条最后的查询点 `HETATM`；`9nnz` 对应 2,417 条 `ATOM` 和 1 条 `HETATM`。这些是输入准备事实，不是实际对接成功证据。

作者教程源目录自带四个参考 `.sc` 文件；它们与 Job `387990` 的新输出目录分开。门禁代码 `check_smoke_scores.py` 在这些**作者参考文件**上核对到 ADP、ATP、CLR、Y01 各一条有效评分，缺评分名称为空，并原样调用官方排名脚本得到 ADP、ATP、CLR、Y01 的顺序。报告位于 `/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/tutorial_reference_gate/`；这只证明读取与排名门禁可用于官方格式，不代表当前 Job 已完成，更不进入正式测试指标。

本次项目样本 `9nal` 的评分及官方排名分别在 `/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/9nal/occurrence_0/small/attempt_2/run/outputs/emerald-id_score.sc` 和同级 `attempt_2/gate/official_rank.txt`；前者 SHA-256 为 `abbd232d92370fe5e025e847bb7300327fd0ba8fcb1f4ad245dfa284e301b95b`，后者为 `06c547ab7149c2860e747316083ab7a4fb649a4b4cfd597b6f713c3435d6d019`。`gate/score_gate.json` 保存原始 zscore、lig_dens、官方概率和 L00/L01 到精确 SMILES 的映射；`input/identity_map.json` 的 SHA-256 为 `8ab88952c329baf1d016fce3d27a90b753756df2f14c4ff4522b00d892a450cd`。只读检查时，这些文件均属于当前单一查询点，未合并其他点的 `.sc`。

本次教程的评分及官方排名分别在 `/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924/tutorial_7bxu/outputs/emerald-id_score.sc` 和同级 `gate/official_rank.txt`；SHA-256 分别为 `7e04a387f0401ff025ea29c8576aabd1c982a5fe12a32dd98b667bb66934546f`、`0f8e22ce23e05b191a68387428c7357a72fc085095d8ea0403edb8a800210621`。新运行输出和作者自带参考输出是不同来源，均保留，不互相覆盖。

## 先前尝试与限制

首次 `9nal` Job `387992_0` 在约 6 分 26 秒后以退出码 1 停止。失败发生在第二个候选 `L01.mol2` 的门禁：RDKit 的 MOL2 读取器无法对一个芳香杂环完成 kekulize，因此旧门禁拒绝该候选。只读使用 Open Babel 将**同一份已生成 MOL2** 回读为 SMILES，再以 RDKit 的非立体、去氢重原子 SMILES 对照原始字符串，两边均为 `Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OP(=O)(O)O)C(O)C1O`，重原子数均为 27；没有观察到化学图改变。故第二次门禁只替换回读工具，仍严格比较重原子、键级、形式电荷与组分；第一次的输入、标准输出及错误保留在原目录和 `project-387992_0.{out,err}`。这不是把准备失败候选从原始身份集合移走。

首次 `9nnz` Job `387992_1` 在 16 分 9 秒后以退出码 2 停止。它已完成两个糖候选的输入准备并进入 Rosetta；Rosetta 的 `RingConformerSet.cc:325` 随后报告 `No conformer with given nu angles found in this set`，本次运行目录没有新 `.sc`，因此不能计算身份排名。此 Job 执行期间，服务器上的 `smoke_project_job.sh` 被更新以支持另一个样本的重试，错误输出末尾又出现 shell 的 `unexpected EOF while looking for matching '"'`；这是独立的操作性瑕疵，不能把环构象异常归因于它，也不能声称该次运行具备干净的冻结脚本证据。原输入、运行目录和 `project-387992_1.{out,err}` 均保留；后续不再覆盖正在运行的脚本。

复核 `9nnz` Job `388007_1` 使用与 `9nal` 重试相同的准备与 Rosetta 脚本版本、原始 SMILES、官方 XML 和命令参数，但写入独立的 `attempt_2` 目录；运行期间未再更新上述脚本。18 分 9 秒后，Rosetta 再次在 `RingConformerSet.cc:325` 报同一异常，只留下 `badstructure.pdb`，没有任何新 `.sc`。Slurm 的 `COMPLETED`/退出码 0 来自 Rosetta 捕获异常后仍以 0 结束，不能视为身份评分门禁通过。该 Job 的 CPU 时间为 17 分 44 秒，峰值驻留内存约 2.0 GiB；标准输出、错误分别位于 `project-9nnz-retry2-388007_1.{out,err}`。这次独立复现把首试的脚本热更新瑕疵与 Rosetta 环构象失败区分开，但仍不能说明所有非小分子都会失败，也不允许从固定 NxU 测试集合中删除糖或其他类别。

官方教程文字称顺序模式会写出 `probability`，但已检查的服务器 Rosetta 源码没有启用该写回语句；官方 `rank_ligand_identity.py` 从 `zscore` 与 `lig_dens` 计算概率。因此不得仅用裸 `zscore` 排名，亦不得把不同查询点的 `.sc` 放在同一排名目录。
