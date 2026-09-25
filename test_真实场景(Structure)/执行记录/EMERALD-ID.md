# EMERALD-ID：已知位点身份比较日志

本日志记录官方 EMERALD-ID 在 NxU/nxu 给定真实配体位置、给定精确 SMILES 集合下的试运行与最终结果。依据是 [比较计划书](../参考材料%20%26%20想法/Matcher_EMERALD-ID%20比较计划书.md)；当前只获准 CPU smoke/门禁，完整批量对接尚未开始。

2026-09-25 仅整理本地位置：下述门禁代码现位于 `test_真实场景(Structure)/smoke/`；本日志中历史 Job 使用的服务器 `.../code/` 路径仍是真实来源，不随本地目录改名。服务器试运行输入、评分、排名及其它科学产物均未改动。

## 实验结果与产物

当前没有正式 NxU/nxu 正确数、逐配体 micro 准确率或可发表图。三个案例的 CPU smoke 均已结束。官方教程 Job `387990` 的 ADP、ATP、CLR、Y01 各有一条本次运行的有效评分；`9nal` 重试 Job `387998_0` 的 L00、L01 也各有一条，首位 L01 与该单点真值精确 SMILES 一致；`9nnz` 独立复核 Job `388007_1` 则重现环构象错误且没有 `.sc`，即使 Slurm 标为 `COMPLETED` 也不算评分成功。任何单点结果都不能填作完整 NxU/nxu 准确率；最终分母预先固定为 NxU 2,502 点、nxu 821 点。

本次 7bxu 教程计算历时 4 小时 14 分 12 秒，单核 CPU 时间 4 小时 6 分 28 秒，峰值驻留内存约 2.64 GiB。新生成的单个 `emerald-id_score.sc` 含四个身份各一条有效评分，官方排名为 ATP、ADP、CLR、Y01，概率分别为 0.457124、0.451823、0.084574、0.006479，缺失身份为空。作者教程目录随附的旧参考 `.sc` 也能通过同一排名脚本，但其首两名次序是 ADP、ATP；两组文件从始至终分目录，不能把它们当作同一次运行或要求随机对接结果逐位相同。正式批量运行的随机种子与重复次数尚未定稿，当前不从此接近的前两名概率推断方法稳定性。

资源量级只作试点估算：`9nal` 的 2 个身份约 101.7 分钟，教程的 4 个身份约 254.2 分钟，平均每个成功身份约 51–64 分钟。机械乘以 NxU/nxu 共 12,286 个位置—身份组合约需 10,400–13,000 单核 CPU 小时；若最多 96 核始终满载，理想下界约 4.5–5.6 天。两种试点的分子、受体和运行路径不同，且 `9nnz` 无有效评分，所以该外推不是正式预算承诺。完整运行尚未获本轮授权，也不能因糖等失败删减固定分母。

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

正式生产命令：尚未定义或执行。未来全量运行的短命令须在通过试运行后另列，不与 smoke 命令混写。

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
