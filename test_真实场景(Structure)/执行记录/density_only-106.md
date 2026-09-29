# density_only-106：已知位点身份比较日志

本日志只记录既有 `density_only-106` 模型在 NxU/nxu 已知位点任务的复用与结果，不覆盖原训练和通用测试档案。依据是 [比较计划书](../参考材料%20%26%20想法/Matcher_EMERALD-ID%20比较计划书.md)；本轮不重新训练。

## 实验结果与产物

2026-09-29 同一 nxu 84 PDB／357 点新增全部类别 SMILES 输入。训练 release `/home/penghongen/Feedback/Matcher/releases/Matcher_a67eb7f830e0/Matcher` 与既有 BEST 不变。真实受体实际 A100 门禁 release／launch 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_f067782cadf7/Matcher`、`/home/penghongen/Feedback/Matcher/launches/390871/gate_all_smiles_gpu_job390871_20260929T131342_a5`，正式前向同一 release、launch `/home/penghongen/Feedback/Matcher/launches/390871/gate_all_smiles_gpu_job390871_20260929T131514_a6`，短命令 `bash 训练与运行/sh/known_site/predict_all_smiles_density106_real.sh`。CryoAtom2 独立 GPU 门禁同一 release、launch `/home/penghongen/Feedback/Matcher/launches/390871/gate_all_smiles_gpu_job390871_20260929T131953_a7`，正式前向 launch `/home/penghongen/Feedback/Matcher/launches/390871/gate_all_smiles_gpu_job390871_20260929T132247_a8`，短命令 `bash 训练与运行/sh/known_site/predict_all_smiles_density106_cryo.sh`。两个新输出根分别为 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/all_smiles_input/{real,CryoAtom2}/density_only-106/nxu/`，每条件 357 个状态为 `scored` 的独立 JSON。小分子输入／全部输入不掩码／全部输入小分子掩码的固定正确数：真实受体 301／290／301，CryoAtom2 303／291／303；全部输入两读出各有 341 个完整候选支持点。完整七模式、共同可用点及八图见[扩展结果页](../../测试结果与追溯/测试结果总览/nxu全部SMILES输入比较.md)，输入快照、门禁与命令追溯见[独立扩展任务日志](nxu全部SMILES输入.md)。掩码与小分子输入的总正确数相同，不等于逐点预测相同；没有覆盖旧逐点结果或启动 NxU。

2026-09-29 CryoAtom2 nxu 正式两视图已经收口。全部保留 84 PDB／357 点正确 303、micro Top-1 为 84.87%、PDB 平均为 91.25%；多身份 19 PDB／143 点正确 89、micro Top-1 为 62.24%、PDB 平均为 61.29%。全部 357 点有身份预测，16 个缺少有效身份候选的点按固定分母计入。正式 GPU 前向沿用下文记载的训练快照 release 与 launch；正式两视图图、汇总 release／launch 和 SHA-256 见[结果页](../../测试结果与追溯/测试结果总览/nxu小分子CryoAtom2测试.md)及[CryoAtom2 独立日志](CryoAtom2-nxu.md)。下段“尚待汇总”是先前时点的记录。

CryoAtom2 小分子已知位点的正式前向已在 H200 Job `390383` 完成：冻结 nxu 的 357 个查询点全部有状态为 `scored` 的独立 JSON，查询键没有遗漏或额外项，PDB 内候选身份轴与真实受体正式前向逐点相同。该条件使用 CryoAtom2 最终受体的模拟密度和受体 token，因而有 5 个点的 Top-1 身份预测不同。正式结果尚待 EMERALD-ID 对接完成后共同汇总，不能提前填三方法比较准确率。此次 GPU 门禁 launch 为 `/home/penghongen/Feedback/Matcher/launches/390383/gate_cryo_density106_nxu_job390383_20260928T223004_a1`，正式 launch 为 `/home/penghongen/Feedback/Matcher/launches/390383/gate_cryo_density106_nxu_job390383_20260928T223343_a2`，共同执行 release 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_6e7096a3a92c/Matcher`；正式命令是 `bash 训练与运行/sh/known_site/predict_cryo_density_only106_nxu.sh`，门禁命令是 `bash test_真实场景(Structure)/smoke/gate_cryo_density106_nxu.sh`。详情见 [独立 CryoAtom2 任务日志](CryoAtom2-nxu.md)。

nxu 小分子阶段正式收口：84 PDB／357 点中身份正确 301 点，逐点 micro Top-1 为 84.31%，PDB 等权平均为 91.18%；其中多身份的 19 PDB／143 点正确 87 点，逐点 micro Top-1 为 60.84%，PDB 等权平均为 61.03%。本方法至少一个身份得分的点数为 357/357；全部身份有分数的点数为 341/357，身份评分对为 572/588；多身份视图分别为 143/143、127/143、358/374。正式两视图 JSON 与图见 [小分子结果页](../../测试结果与追溯/测试结果总览/nxu小分子真实测试.md)；汇总的正式命令为 `bash 训练与运行/sh/known_site/summarize_small.sh`，release 与 launch 见 [共享总日志](总日志.md)。以下“仍待汇总”文字是正式汇总前保留的历史状态，NxU 官方对接仍待运行。

正式已知位点前向已结束：NxU 的 995 个、nxu 的 357 个冻结查询点均有独立 JSON，状态均为 `scored`。按冻结真值对正式逐点文件进行只读核对，NxU 正确 820/995，nxu 正确 301/357；真值身份分别有 993/995、352/357 个点受模型身份资产支持，未支持点仍在分母中。候选身份、预测身份和 logit 数组逐点对齐检查无错误。PDB 平均、多身份分层与类别图仍待三方法共同汇总，不以评分成功数替换固定分母。

| 追溯项目 | 已核实位置或状态 |
| --- | --- |
| 正式变体 | `density_only_106`；位点侧读取 56 个密度派生通道及 50 个受体原子栅格通道 |
| 训练 release | `/home/penghongen/Feedback/Matcher/releases/Matcher_a67eb7f830e0/Matcher` |
| 冻结 BEST | `/storage/penghongen/Matcher/runs/density_only-106/train_density_only106_job386151_20260920T230728/BEST.pt` |
| 原训练与完整测试追溯 | `测试结果与追溯/experiment-centric/density_only-106/追溯与说明.md` |
| 本任务 known-site 输出 | `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/density_only-106/{NxU,nxu}/{pdb_id}/{occurrence_id}.json`；分别已有 995 和 357 个状态为 `scored` 的逐点结果 |
| 当前 allocation | Job `388457`，gnode09 A800、4 CPU；第 4 次正式前向成功后仅删除精确 `after_lock_388457` 正常退出，资源已释放；`allocations/388457/out` 和 `err` 保留 |
| 冻结比较清单 | `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest/manifest.json`，SHA-256 `095f1bdc6b76a98374eca197e7a179bccc1dfe244d66ffcdefa941d112cde211`；held_out_test_0 来源 manifest SHA-256 `ed9ab3212d4a9aeb1be8730f35bd86bc434b3ecca4c72732942b7397ad5f90ce` |
| 正式前向 release／launch | `/home/penghongen/Feedback/Matcher/releases/Matcher_0666fa2b8713/Matcher`；`/home/penghongen/Feedback/Matcher/launches/388457/predict_density_only106_job388457_20260925T214638_a4` |

正式生产命令：Job `388457` 的第 4 次执行为 `bash 训练与运行/sh/known_site/predict_density_only106.sh`。最初提交采用 `bash 训练与运行/submit_task.sh --sh 训练与运行/sh/known_site/predict_density_only106.sh --resource a800 --gpus 1 --cpus 4 --qos h200g4 --time 24:00:00 --pre_hold --after_hold --job-name known_d106`；运行时只编辑该 Job 的动态命令并删除精确 try_lock，不删除 after_lock。当前 NxU/nxu 逐点输出中，结果尚未汇总。

测试与门禁命令：Job `388457` 的第 1–3 次执行均为 `bash test_真实场景(Structure)/smoke/gate_density_only106.sh`，与正式命令分开。第 1 次因默认 Python 无 SciPy 失败，release `/home/penghongen/Feedback/Matcher/releases/Matcher_ef197c9b267a/Matcher`、launch `/home/penghongen/Feedback/Matcher/launches/388457/predict_density_only106_job388457_20260925T211221_a1` 保留。第 2 次项目 Python 环境修正后在临时清单 PASS。第 3 次绑定最终清单、实际 A800 和 release `Matcher_0666fa2b8713` PASS，launch 为 `/home/penghongen/Feedback/Matcher/launches/388457/predict_density_only106_job388457_20260925T214357_a3`，门禁产物在 `/storage/penghongen/tmp/Matcher/gate_density_only-106_fdbyn31m/`。独立数值门禁还证明同一既有 centered 候选上的实验与模拟裁块、50 通道受体栅格及完整 56 通道 bank 与训练读取结果逐元素完全一致。

只读核查命令：已检查 `matcher/evaluation/checkpoint.py` 的训练快照加载逻辑，以及 `matcher/datasets/candidate_data.py` 中 56+50 通道构造所需的 centered 字段和受体原子字段。

## 先前尝试与限制

首轮 GPU 门禁是运行环境失败尝试，不是模型科学结果；服务器 `allocations/388457/err` 与上述 launch 保留完整错误。`density_only-106` 并非纯实验密度模型；真实点裁块中的模拟密度和受体栅格必须与训练时的空间约定一致。`9ian` 和 `9oq7` 的不受支持真实身份对应点仍留在固定分母。
