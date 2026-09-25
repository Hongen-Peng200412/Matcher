# 已知位点身份比较的输入接入与小规模门禁

本目录只保存 Matcher 对真实 occurrence 质心的输入接入，以及官方 EMERALD-ID 的任务准备、试运行和独立核查代码。它不修改 Matcher 模型、已有 checkpoint、AdaLigand 科学产物或官方 Rosetta 实现。正式科学定义、固定分母和停止条件见 [比较计划书](../参考材料%20%26%20想法/Matcher_EMERALD-ID%20比较计划书.md)；真实 Job、命令和观察值只写入 [执行记录](../执行记录/总日志.md)。

当前目录中的 `smoke_tutorial.sh` 是**门禁任务**，不是 NxU/nxu 正式生产命令。它按官方教程的 XML、参数文件、原始实验图和示例受体 PDB 调用服务器 Rosetta，仅将本机不适用的教程相对路径与数据库路径换成明确绝对路径。程序结束后，须在单独的查询目录检查评分文件与官方排名脚本；不能凭进程退出码宣称化学身份评分成功。

```text
test_真实场景(Structure)/smoke/
├── README.md                  # 本目录任务边界和门禁入口
├── smoke_tutorial.sh          # 官方 7bxu 教程的一次 CPU 试运行，不代表正式测试
├── prepare_project_smoke.py   # 一个真实 occurrence 质心、原始图、受体 PDB 与候选化学输入
├── smoke_project.sh           # 对一个准备好的项目查询点调用官方 Rosetta 协议
├── smoke_project_job.sh       # 两个独立 CPU 数组元素分别试运行 9nal 与 9nnz
├── check_smoke_scores.py      # 一个查询目录的评分完整性门禁和官方排名调用
└── check_params_graph.py      # MOL2 与 Rosetta params 的元素、连接和键级诊断
```

`smoke_tutorial.sh` 接收两个位置参数：教程根目录，例如本地资料中的 `.../tutorials/EMERALD-ID/`，以及一次运行的全新输出目录，例如服务器临时根下的 `tutorial_7bxu/`。输出目录中的 `outputs/` 由 Rosetta 写入 scorefile 和构象；标准输出、标准错误及 Slurm Job 状态由提交方保存。输入必须是教程自带的 `inputs/7bxu.pdb`、`inputs/emd_30238.map`、`library/{ADP,ATP,CLR,Y01}.params` 与 `ligand_screen.xml`；脚本不读取项目测试真值。

门禁和正式运行严格分开：教程任务只证明官方环境可执行；项目 `9nal`、`9nnz` 要另外核查原始 SMILES 到生成配体的重原子连接、键级、形式电荷和分子组分不变，并在每个查询点独立汇总评分。只有这些门禁通过后，才允许定义短小的正式批量入口。本目录后续新增文件时，应按实际代码与产物更新本 README，而非提前声明不存在的文件或成功状态。

`prepare_project_smoke.py` 对每个原始顶层 SMILES 写出一个确定名称，如 `L00`。它在将 SDF 转为带 MMFF94 部分电荷的 MOL2 后，使用 Open Babel 回读该 MOL2 为 SMILES，再由 RDKit 比较原始与回读的去氢化学图；不同即停止该查询的准备，不把改写后的分子当作原身份。这样可以避开 RDKit 自身 MOL2 读取器对部分芳香环的解析限制，但比较仍保留重原子、键级、形式电荷和组分。`identity_map.json` 保留名称、精确 SMILES 和全局分辨率；`truth_for_gate.json` 单独保留真实身份和查询中心，只供门禁与事后评价，Rosetta 命令不读取。`input.pdb` 只有受体 ATOM 重原子和最后一条位于查询质心的 `C3/LG2` 虚拟 HETATM；`map.map` 是原始 EMDB 压缩图的无损解压。`smoke_project.sh` 对 `identity_map.json` 中的同一查询候选设置官方 `multiple_ligands`，并用全局分辨率填入教程所需的 `local_res` 参数；没有伪造局部测量值。

`smoke_project_job.sh` 的数组元素 `0` 对应 `9nal` 的 `candidate_id=0`、仅小分子闭集；元素 `1` 对应 `9nnz` 的 `candidate_id=0`、全量闭集。两个元素各只使用 1 CPU 和 1 线程，各自输出到不同 PDB/occurrence/task 目录。当前脚本用环境变量 `SMOKE_ATTEMPT` 给重试选择新目录，如 `attempt_2/`；未设时使用 `attempt_1/`。最初提交的旧版脚本没有尝试号，其首次 `input/`、`run/` 直接位于 PDB/occurrence/task 目录；旧输出保持原位置。其他查询点和正式结果不由该脚本产生。

`check_smoke_scores.py` 只扫描传入的单查询 `outputs/*.sc`，要求每个有评分的 `ligandname` 在当前身份映射中且只出现一次，`zscore` 与 `lig_dens` 均为有限值；缺评分身份被保留在 `missing_names`。它随后调用原样的 `rank_ligand_identity.py` 生成 `official_rank.txt`，并在 `score_gate.json` 中逐身份记录官方概率及原始精确 SMILES。教程没有项目 SMILES，只用 `--expected-names ADP,ATP,CLR,Y01`；项目查询传 `--mapping input/identity_map.json`。此脚本是试运行门禁，不是正式科学打分器，不能把它写出的概率冒充完整测试结果。

`check_params_graph.py` 对一个查询的每份 MOL2 与对应 `.params` 比较原子元素、带键级的连接和分子组分。Rosetta 会给重复原子名称重新编号并增加虚拟环原子，因此程序用分子图同构，而不是按原子名称逐行比较；虚拟原子不算原始化学图。它保持严格的字面键级检查：`9nal` 的 L00 通过，L01 因官方参数生成器把一条共轭 C–N 单键写作 `BOND_TYPE ... 2` 而报告差异。用户已允许这条有源码证据的官方内部编码并要求披露；本程序没有因此放宽其它键级、原子或连接变化，也不负责检查 Rosetta 文件不能直接表示的逐原子形式电荷。原始 SMILES 到 MOL2 的化学图检查仍由 `prepare_project_smoke.py` 独立执行。
