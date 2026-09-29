# nxu 固定查询点的全部 SMILES 输入比较：独立任务日志

本任务扩展[已知位点比较计划](../参考材料%20%26%20想法/Matcher_EMERALD-ID%20比较计划书.md)中的 nxu 小分子位点身份实验。查询仍是原冻结清单的 84 PDB／357 个小分子 occurrence；只扩展两个 Matcher 模型的 PDB 内身份候选，不重新按成本截尾。真实受体和 CryoAtom2 的既有小分子输入 Matcher 与 EMERALD-ID 逐点结果均只读复用。本任务不运行全配体 NxU、不训练、不读取 calibration。

## 已收口状态与结果入口

2026-09-29 三次正式 GPU 前向、两种受体条件的完整身份读出和八张折线图均已完成。固定 357 点中，真实受体 density_only-106 的小分子输入／全部输入不掩码／全部输入小分子掩码分别正确 301／290／301 点，CryoAtom2 为 303／291／303 点；density_only-0 的两条件均为 286／259／284 点。EMERALD-ID 原冻结小分子输入分别为 233／263 点。本任务的完整七模式、固定分母、自身完整和共同可用点表见[结果页](../../测试结果与追溯/测试结果总览/nxu全部SMILES输入比较.md)。真实受体共同可用 190 点，CryoAtom2 共同可用 222 点。服务器汇总原件在 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/all_smiles_comparison/{real,CryoAtom2}/`，本地逐文件复制的两份 summary.json 与八张 PNG 的 SHA-256 和服务器一致。下面的“待门禁”“尚待”等字样均保留为当时状态，不表示本任务仍在运行。

本轮唯一复用 A100 allocation Job `390871` 已在第 10 次绘图成功、精确 `try_lock` 恢复、没有预测或汇总子进程后，仅删除它自己的 `after_lock_390871`。`sacct` 确认 Job 与 batch 均 `COMPLETED 0:0`；未使用 `scancel`，没有改变旧真实受体、CryoAtom2、NxU 或 calibration 产物。

## 当前状态与预核算

2026-09-29 只读检查服务器正式 `manifest/nxu/`、84 个 PDB 的完整 occurrence、顶层 PreparedLigand.smiles、`test0_real_smiles_v1` 身份清单和 `smiles_graphs_v1.npz`：原小分子身份槽总数为 114，全类别身份槽为 164，新增 50 个按 PDB 计数的身份槽；357 点与其完整身份轴相乘共 771 个查询点—身份配对。同一 PDB 内没有由不同类别共用精确 SMILES 的反例。全类别身份数的 PDB 分布为 1 个身份 34 PDB、2 个 28 PDB、3 个 15 PDB、4 个 6 PDB、5 个 1 PDB。图与语言资产的缺失仍各对应 `9ian`、`9oq7` 的同两个精确字符串；不能因缺失删查询点。此处是输入规模与可用性预核算，不是新模式的准确率。

原固定 nxu 的 EMERALD-ID 全小分子身份评分完整点，真实受体为 190/357、CryoAtom2 为 222/357；其中 PDB 不含额外非小分子身份的完整点分别为 89 和 110。这是预核算，不是将来的共同可用子集：正式配对仍须逐点读取两个 Matcher 全类别前向与小分子输入结果后再冻结。

2026-09-29 清单正式构建已由 CPU Job `390862` 执行成功，命令为 `bash 训练与运行/sh/known_site/build_all_smiles_manifest.sh`；release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_076e62e43657/Matcher`，标准 launch 为 `/home/penghongen/Feedback/Matcher/launches/390862/build_all_smiles_manifest_job390862_20260929T124856_a1`。派生文件 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest_all_smiles/nxu/pdbs.jsonl` 的 SHA-256 是 `4442864ff95d5b62e1e9464f6811de49fbe7fb98ea8bbd18de3097c201e27a65`，逐字段复核为 84 PDB、164 全类别槽、114 小分子槽，类别轴长度与完整身份轴逐 PDB 相同。Job 自动释放。

输入快照最初以 8 CPU Job `390866` 提交，但 Slurm 只读预计启动时间约为 2026-10-01 06:03，且核对其仍为 PENDING、没有 launch 或输入快照产物后，仅取消了这个尚未执行的 Job，没有删除科学文件。A100 Job `390871` 已按标准入口申请 1 GPU＋4 CPU、`pre_hold` 与 `after_hold`；在其精确 `pre_lock`、`after_lock` 与无 `try_lock`／`kill_lock` 状态下保存初始门禁动态命令到 `/storage/penghongen/tmp/Matcher/ops/all_smiles_20260929/job390871_gate_original.sh`，把输入快照脚本复制到该 Job 的精确动态命令，双方 SHA-256 均为 `076cb0765f8199d0236516e4a53c0b885086c9a50f5eb8912984f6bd1c49d96c`。只删除 `pre_lock_390871` 触发这次快照。待它回到 `try_lock` 且输入文件哈希快照成功，才恢复门禁命令并触发实际 GPU 检查；门禁成功前不运行正式前向。

上述输入快照现已在 Job `390871` 第一次执行成功，并回到精确 `try_lock`。这次是追溯命令 `bash 'test_真实场景(Structure)/smoke/snapshot_all_smiles_inputs.sh'`，不是模型正式前向；release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_7408396614c3/Matcher`，launch 为 `/home/penghongen/Feedback/Matcher/launches/390871/gate_all_smiles_gpu_job390871_20260929T125621_a1`。文件 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/provenance/all_smiles_input/input_files.jsonl` 保存 842 个实际输入文件的路径、字节数和 SHA-256，自身 SHA-256 为 `1c8b8446f80e9fed012d4fd150e61a57a257f0fc497f4f2269e4abb50418f328`。launch 名沿用原首次门禁脚本名，但其冻结动态命令及输出明确为输入快照，不可把它当作 GPU 门禁 PASS。

## 正式产物与命令

正式构建命令见上。三个正式 GPU 前向命令分别为 `bash 训练与运行/sh/known_site/predict_all_smiles_density0.sh`、`bash 训练与运行/sh/known_site/predict_all_smiles_density106_real.sh` 和 `bash 训练与运行/sh/known_site/predict_all_smiles_density106_cryo.sh`；它们与门禁、画图及只读核查命令分开。三次正式前向现均完成，逐次 release、launch 与产物位置见后文。画图命令另记，不归入正式 GPU 前向。

两个既有训练 BEST 均只读：density_only-0 为 `/storage/penghongen/Matcher/runs/density_only-0/train_density_only0_job379403_20260920T205216/BEST.pt`，SHA-256 `7667d645dd871651af85283fcbae147344b3834c547ee4048c52ec3684015376`，其 `run.json` 的训练代码 release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_8c8f126c6d8b/Matcher`；density_only-106 为 `/storage/penghongen/Matcher/runs/density_only-106/train_density_only106_job386151_20260920T230728/BEST.pt`，SHA-256 `f779f96594dc49f91cc9a4eadb0d7b5440cc65e582d87b5d74354186ad35a258`，训练 release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_a67eb7f830e0/Matcher`。正式前向通过 `load_matcher_from_training_release` 读取各自 `run.json`、模型类与配置，不从当前工作区直接解释历史 BEST。

## 门禁与只读核查

Job `390871` 第 2 次执行的实际 A100 GPU 门禁 launch 为 `/home/penghongen/Feedback/Matcher/launches/390871/gate_all_smiles_gpu_job390871_20260929T130231_a2`、release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_3d24f37880b6/Matcher`。它在 Bash 展开空的 `receptor_args` 数组时因 `set -u` 报 `unbound variable`，尚未调用 Python、模型或读入任何查询点；退出码 1 后恢复精确 `try_lock`。这只是门禁脚本参数传递缺陷，不能算科学门禁通过。最小修订改为始终显式传入受体根：真实受体为原 Ori_Data，CryoAtom2 为其预测受体适配根；不改密度或身份计算。下一次门禁仍需在实际 A100 上取得模型前向和同一 logits 的两种读出一致性。

第 3 次执行是修订后的 density_only-0／真实受体实际 A100 GPU 门禁，release `/home/penghongen/Feedback/Matcher/releases/Matcher_1b3417accbc5/Matcher`，launch `/home/penghongen/Feedback/Matcher/launches/390871/gate_all_smiles_gpu_job390871_20260929T130635_a3`。`21kr` 的隔离查询实际编码了完整身份轴中的非小分子候选，门禁从同一 `identity_logits` 复算不掩码与小分子掩码预测均一致，控制器报告 PASS 且恢复 `try_lock`。此门禁不是正式 357 点结果。

确认相同 Job 的精确 `try_lock`、`after_lock` 存在，`pre_lock`、`kill_lock` 不存在后，将 `predict_all_smiles_density0.sh` 原样复制到该 Job 的动态命令；双方 SHA-256 均为 `761bccda94744f326b0229be4e95b54548680695f5be9512f57b3c416b0741fb`。仅删除 `try_lock_390871` 启动 density_only-0 的 357 点完整身份正式前向。动态脚本只含这一项正式任务；原始提交任务名仍为门禁名，必须结合此动态命令及即将生成的 launch 判断实际运行内容。正式输出指向全新 `all_smiles_input/real/density_only-0/nxu/`，不覆盖旧小分子输入产物。

density_only-0 正式前向已结束并恢复 `try_lock`。实际 A100 release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_1b3417accbc5/Matcher`，launch 为 `/home/penghongen/Feedback/Matcher/launches/390871/gate_all_smiles_gpu_job390871_20260929T130824_a4`，正式命令如上。输出 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/all_smiles_input/real/density_only-0/nxu/` 恰有 357 份 `scored` JSON，与原查询键一一对应；每点可评分身份与 `identity_logits` 逐槽同长且包含于完整冻结身份轴，逐点复算不掩码和掩码 argmax 全部一致。两种读出有 25 个点的预测字符串不同，这是模型输出事实，不是准确率结论。单通道模型读取同一原始实验密度、同一查询清单、同一全类别身份轴、相同 BEST 和训练 release；`density_only` 分支不读取 `--receptor-root`，故 CryoAtom2 条件复用本次完整身份预测，不再前向一次。

随后在 Job `390871` 的精确四锁状态下，从保留的初始动态门禁命令恢复门禁包装，只把参数改为 `density_only-106 real`；动态命令 SHA-256 为 `2c9acb5592b2ec4699db8da667449577588333c899e2321b337cec89b2d88f12`。仅删除精确 `try_lock_390871` 触发第二种模型的真实受体 GPU 门禁。该门禁只在隔离点前向，不写 357 点正式结果；门禁成功且 `try_lock` 恢复后才安装 `predict_all_smiles_density106_real.sh`。

density_only-106／真实受体的实际 A100 门禁在 release `/home/penghongen/Feedback/Matcher/releases/Matcher_f067782cadf7/Matcher`、launch `/home/penghongen/Feedback/Matcher/launches/390871/gate_all_smiles_gpu_job390871_20260929T131342_a5` 报 PASS，非小分子候选已编码，同一 logits 的两种读出可复算。返回 `try_lock` 后把 `predict_all_smiles_density106_real.sh` 原样安装为动态命令，双方 SHA-256 `bc2ccbb9731dc9c59566306b144b57ed8ff2db51b7175f1f8f089aec0315edde`；仅删除精确 `try_lock_390871` 触发其正式 357 点前向。正式结果位于新的 `all_smiles_input/real/density_only-106/nxu/`，尚待完整性验收。

density_only-106／真实受体的正式前向已在 Job `390871` 第 6 次执行结束，release `/home/penghongen/Feedback/Matcher/releases/Matcher_f067782cadf7/Matcher`，launch `/home/penghongen/Feedback/Matcher/launches/390871/gate_all_smiles_gpu_job390871_20260929T131514_a6`，正式命令如上。357 个冻结查询键各有一份 `scored` JSON；逐点检查 `scored_smiles` 与 `identity_logits` 同长、为完整身份轴子集，两种读出逐点等于同一 logits 的对应 argmax；0 项不一致。两个读出的预测字符串在 12 个点不同，尚不是准确率汇总。Job 回到 `try_lock`，可在同一 A100 上进行 CryoAtom2 条件的独立门禁。

第 7 次执行在同一 A100 上对 density_only-106／CryoAtom2 做隔离 GPU 门禁，动态命令 SHA-256 为 `4161f0e0752ba1b290b1415f3b97a50f128928556ca3a27db8268c247a681dc7`；release `/home/penghongen/Feedback/Matcher/releases/Matcher_f067782cadf7/Matcher`，launch `/home/penghongen/Feedback/Matcher/launches/390871/gate_all_smiles_gpu_job390871_20260929T131953_a7`。门禁实际编码了非小分子身份，并从同一组 logits 复算两种答案，报告 PASS；`try_lock` 恢复后才替换正式命令。

第 8 次执行是 density_only-106／CryoAtom2 的正式前向，动态短命令为 `bash 训练与运行/sh/known_site/predict_all_smiles_density106_cryo.sh`，动态脚本 SHA-256 `8e7b60954ec602358d3dec952320a6cbe97f262c836a1f7fa047952de665492f`。release `/home/penghongen/Feedback/Matcher/releases/Matcher_f067782cadf7/Matcher`，launch `/home/penghongen/Feedback/Matcher/launches/390871/gate_all_smiles_gpu_job390871_20260929T132247_a8`；新的输出根为 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/all_smiles_input/CryoAtom2/density_only-106/nxu/`。357 个冻结查询键各有一份 `scored` JSON；逐点检查 logits 长度、冻结身份轴子集以及两种答案的 argmax，三项不一致数均为 0。两种读出的预测字符串在 15 点不同。第 8 次运行成功并返回 `try_lock`；没有覆盖真实受体或原小分子输入结果。

## 绘图命令与汇总验收

绘图与表格计算只读取五种已有逐点根，不再调用模型或 EMERALD-ID。真实受体使用独立绘图命令 `bash 训练与运行/sh/known_site/plot_all_smiles_real.sh`，Job `390871` 第 9 次执行；动态脚本 SHA-256 `0c855bb1f6c11def156692b0cd893f35407a0253a0dc0b6d09c79dfed330aab6`，release `/home/penghongen/Feedback/Matcher/releases/Matcher_f067782cadf7/Matcher`，launch `/home/penghongen/Feedback/Matcher/launches/390871/gate_all_smiles_gpu_job390871_20260929T132646_a9`。CryoAtom2 使用另一个绘图命令 `bash 训练与运行/sh/known_site/plot_all_smiles_cryo.sh`，第 10 次执行；动态脚本 SHA-256 `16579aaf72e98567236c2ad0706c57da81a30a7fb2a8f8b7e14811ad67ee90b0`，同一 release，launch `/home/penghongen/Feedback/Matcher/launches/390871/gate_all_smiles_gpu_job390871_20260929T132842_a10`。两次虽借用 A100 allocation 附属 CPU，但没有执行 GPU 前向；因此它们归类为绘图命令，不混入正式模型运行命令。

只读汇总核查：每条件 summary.json 都含原冻结 357 个键，七种方法的固定分母均为 357；真实受体共同键 190 个，CryoAtom2 共同键 222 个，均逐点包含于七方法自身完整键。六个横轴组别的固定支持数相加均为 357；每条件恰有四张 PNG。八图和两份 JSON 原件 SHA-256 逐文件记于本地对应副本的文件哈希，结果页直接链接这些副本；本机复制后也已逐文件与服务器哈希对照相等。小分子身份数分组的固定支持数为 1→214、2→75、3→48、4→20、5→0、>5→0，空组按 N/A 处理。所有准备和评分失败点仍留在固定 357 点主分母。绘图原件不覆盖原逐点结果。

| 受体条件 | 正式汇总文件 | SHA-256 |
| --- | --- | --- |
| 真实受体 | `summary.json` | `10f1666d592a5b2b0daf00b636d7401a9ff05f7a234c7c43aea6502ca3c7cf3e` |
| 真实受体 | `unmasked_overall.png` | `ca1653165502698dcf83beb4f66f372dfb6a9bef1bf3d5c186295fedb6ef9c29` |
| 真实受体 | `unmasked_paired.png` | `04cd1b09c711579dea014d94c6353275adae4bdf052e9f7a8805e69913528dcd` |
| 真实受体 | `small_masked_overall.png` | `9ca84381877ad030b200adcf0c6b1853cde63b75029f41d6a2ad738cfb767636` |
| 真实受体 | `small_masked_paired.png` | `a17b9d4688f59566540c9a75f2f0d4e4a407859962e9066230d4467e1aa2e05b` |
| CryoAtom2 | `summary.json` | `71c670b88e886bebacbf57b9104a6f2c432914c44bb51844abbf6a26e1208f15` |
| CryoAtom2 | `unmasked_overall.png` | `c49dc7edd81650659511875dc9e0b5b994d3a63d40b126762bed4aab45e51cc3` |
| CryoAtom2 | `unmasked_paired.png` | `93ca5c968f0eb0f5ded3f82732e2036f96a3b252f484c5fa479e8fdaddea77f0` |
| CryoAtom2 | `small_masked_overall.png` | `1479280de6dd130c8fd3bda831e68e59529dc6c16bd9aa441f2c1842eeaaa26c` |
| CryoAtom2 | `small_masked_paired.png` | `c447e059494d96ffe324f4e50a655ec58a002e9f94a5fbf48d5768b028670426` |

本机 `py_compile` 已通过新增清单、预测入口与 GPU 门禁入口；新增或修改的 Bash 脚本已通过 MSYS2 `bash -n`。四项实际 GPU 门禁中的首项因 Bash 空数组展开失败，修订后 density_only-0／真实受体、density_only-106／真实受体、density_only-106／CryoAtom2 均 PASS。门禁命令使用 `bash 'test_真实场景(Structure)/smoke/gate_all_smiles_gpu.sh' <模型> <受体条件>`，与三项正式前向命令隔离。只读预核算逐个读取 84 个原始 occurrence、prepared 顶层 SMILES、图包字符串轴与 held_out_test_0 身份资产索引，未修改服务器文件。

## 计划与实现差异

当前未发现改变用户新增科学契约的差异。NxU 全配体属于明确不执行范围。单通道模型跨受体复用有代码和输入依据：两条件均采用原始实验密度、同一冻结查询中心、同一完整身份清单及相同训练 release／BEST；`density_only` 分支根本不读取受体文件。新图中的分母和结果待正式汇总后回填。
