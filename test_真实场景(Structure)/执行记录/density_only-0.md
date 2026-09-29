# density_only-0：已知位点身份比较日志

本日志只记录既有 `density_only-0` 模型在 NxU/nxu 已知位点任务的复用与结果，不覆盖原训练和通用测试档案。依据是 [比较计划书](../参考材料%20%26%20想法/Matcher_EMERALD-ID%20比较计划书.md)；本轮不重新训练，不把旧 Stage1 候选结果冒充给定真实点结果。

## 实验结果与产物

2026-09-29 同一 nxu 84 PDB／357 点新增全部类别 SMILES 输入。训练 release `/home/penghongen/Feedback/Matcher/releases/Matcher_8c8f126c6d8b/Matcher` 与既有 BEST 不变；实际 A100 门禁 release／launch 为 `/home/penghongen/Feedback/Matcher/releases/Matcher_1b3417accbc5/Matcher`、`/home/penghongen/Feedback/Matcher/launches/390871/gate_all_smiles_gpu_job390871_20260929T130635_a3`，正式前向使用同一 release、launch `/home/penghongen/Feedback/Matcher/launches/390871/gate_all_smiles_gpu_job390871_20260929T130824_a4`，短命令 `bash 训练与运行/sh/known_site/predict_all_smiles_density0.sh`。新输出 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/all_smiles_input/real/density_only-0/nxu/` 有 357 个状态为 `scored` 的独立 JSON。原小分子输入、全部输入不掩码、全部输入小分子掩码的固定正确数依次为 286／259／284；全部输入两读出各有 341 个完整候选支持点。CryoAtom2 不另做前向：查询中心、原实验密度、完整身份轴、BEST 和训练快照相同，本变体不读取受体文件。共同可用分母随 EMERALD-ID 条件变化，真实受体 190 点、CryoAtom2 222 点；完整表、八图和结果解释见[扩展结果页](../../测试结果与追溯/测试结果总览/nxu全部SMILES输入比较.md)，输入快照、GPU 门禁、命令及失败尝试见[独立扩展任务日志](nxu全部SMILES输入.md)。未覆盖原小分子逐点结果，不运行 NxU。

2026-09-29 CryoAtom2 nxu 正式两视图已经收口。本模型没有第二次前向：同一原始实验密度、查询点和 PDB 内候选 SMILES 经逐点核对后，汇总直接读取真实受体条件的冻结预测。全部保留 84 PDB／357 点正确 286、micro Top-1 为 80.11%、PDB 平均为 89.61%；多身份 19 PDB／143 点正确 72、micro Top-1 为 50.35%、PDB 平均为 54.07%。全部查询点有身份预测；缺少有效身份候选的 16 个点仍按固定分母计入。正式图和三方法比较见[结果页](../../测试结果与追溯/测试结果总览/nxu小分子CryoAtom2测试.md)，正式汇总 release／launch 和结果 SHA-256 见[CryoAtom2 独立日志](CryoAtom2-nxu.md)。下段“将汇总”是先前时点的记录。

CryoAtom2 nxu 条件不重新运行本模型：其候选侧唯一输入是原始实验密度的单通道 `(1,48,48,48)` 裁块，冻结查询点、原始实验密度文件、身份资产和 PDB 内候选集合均与真实受体 nxu 逐点相同。只读核对 84 个 PDB 的原始实验密度路径、357 个查询键与已有 357 份 `scored` 结果后，CryoAtom2 两视图汇总将通过 `--density0-root` 直接读取 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/density_only-0/nxu/`；不复制结果、不声称进行了第二次 GPU 前向。详见 [CryoAtom2 独立任务日志](CryoAtom2-nxu.md)。

nxu 小分子阶段正式收口：84 PDB／357 点中身份正确 286 点，逐点 micro Top-1 为 80.11%，PDB 等权平均为 89.61%；其中多身份的 19 PDB／143 点正确 72 点，逐点 micro Top-1 为 50.35%，PDB 等权平均为 54.07%。本方法至少一个身份得分的点数为 357/357；全部身份有分数的点数为 341/357，身份评分对为 572/588；多身份视图分别为 143/143、127/143、358/374。正式两视图 JSON 与图见 [小分子结果页](../../测试结果与追溯/测试结果总览/nxu小分子真实测试.md)；汇总的正式命令为 `bash 训练与运行/sh/known_site/summarize_small.sh`，release 与 launch 见 [共享总日志](总日志.md)。以下“仍待汇总”文字是正式汇总前保留的历史状态，NxU 官方对接仍待运行。

正式已知位点前向已结束：NxU 的 995 个、nxu 的 357 个冻结查询点均有独立 JSON，状态均为 `scored`。按冻结真值对正式逐点文件进行只读核对，NxU 正确 754/995，nxu 正确 286/357；真值身份分别有 993/995、352/357 个点受模型身份资产支持，未支持点仍在分母中。候选身份、预测身份和 logit 数组逐点对齐检查无错误。PDB 平均、多身份分层与类别图仍待三方法共同汇总，不以评分成功数替换固定分母。

| 追溯项目 | 已核实位置或状态 |
| --- | --- |
| 正式变体 | `density_only`；位点侧仅一个 `(1,48,48,48)` 实验密度通道 |
| 训练 release | `/home/penghongen/Feedback/Matcher/releases/Matcher_8c8f126c6d8b/Matcher` |
| 冻结 BEST | `/storage/penghongen/Matcher/runs/density_only-0/train_density_only0_job379403_20260920T205216/BEST.pt` |
| 原训练与完整测试追溯 | `测试结果与追溯/experiment-centric/density_only-0/追溯与说明.md` |
| 本任务 known-site 输出 | `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/density_only-0/{NxU,nxu}/{pdb_id}/{occurrence_id}.json`；分别已有 995 和 357 个状态为 `scored` 的逐点结果 |
| 当前 allocation | Job `388456`，gnode09 A800、4 CPU；第 4 次正式前向成功后仅删除精确 `after_lock_388456` 正常退出，Slurm `COMPLETED 0:0`，资源已释放；`allocations/388456/out` 和 `err` 保留 |
| 冻结比较清单 | `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest/manifest.json`，SHA-256 `095f1bdc6b76a98374eca197e7a179bccc1dfe244d66ffcdefa941d112cde211`；held_out_test_0 来源 manifest SHA-256 `ed9ab3212d4a9aeb1be8730f35bd86bc434b3ecca4c72732942b7397ad5f90ce` |
| 正式前向 release／launch | `/home/penghongen/Feedback/Matcher/releases/Matcher_0666fa2b8713/Matcher`；`/home/penghongen/Feedback/Matcher/launches/388456/predict_density_only0_job388456_20260925T214630_a4` |

正式生产命令：Job `388456` 的第 4 次执行为 `bash 训练与运行/sh/known_site/predict_density_only0.sh`。最初提交采用 `bash 训练与运行/submit_task.sh --sh 训练与运行/sh/known_site/predict_density_only0.sh --resource a800 --gpus 1 --cpus 4 --qos h200g4 --time 24:00:00 --pre_hold --after_hold --job-name known_d0`；运行时只编辑该 Job 的动态命令并删除精确 try_lock，不删除 after_lock。当前 NxU/nxu 逐点输出中，结果尚未汇总。

测试与门禁命令：Job `388456` 的第 1–3 次执行均为 `bash test_真实场景(Structure)/smoke/gate_density_only0.sh`，与正式命令分开。第 1 次因默认 Python 无 SciPy 失败，release `/home/penghongen/Feedback/Matcher/releases/Matcher_ef197c9b267a/Matcher`、launch `/home/penghongen/Feedback/Matcher/launches/388456/predict_density_only0_job388456_20260925T211224_a1` 保留。第 2 次项目 Python 环境修正后在临时清单 PASS。第 3 次绑定最终清单、实际 A800 和 release `Matcher_0666fa2b8713` PASS，launch 为 `/home/penghongen/Feedback/Matcher/launches/388456/predict_density_only0_job388456_20260925T214400_a3`，门禁产物在 `/storage/penghongen/tmp/Matcher/gate_density_only-0_pdg5bx3i/`。独立数值门禁还证明同一既有 centered 候选上的裁块、单密度通道与训练读取结果逐元素完全一致。

只读核查命令：已检查 `matcher/evaluation/checkpoint.py` 的训练快照加载逻辑，以及 `matcher/datasets/candidate_data.py` 中 `density_only` 的单通道构造。

## 先前尝试与限制

首轮 GPU 门禁是运行环境失败尝试，不是模型科学结果；服务器 `allocations/388456/err` 与上述 launch 保留完整错误。既有 Dataset 以 Stage1 `centered_box_index` 定位候选，新任务必须按真实 occurrence 质心构造输入，且在 NxU/nxu 各自完整身份集合下单独前向。`9ian` 和 `9oq7` 的不受支持真实身份对应点仍留在固定分母。
