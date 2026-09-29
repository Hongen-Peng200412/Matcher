# 已知位点任务的正式短命令

本目录的 `.sh` 文件由 `训练与运行/submit_task.sh` 在 Slurm allocation 中执行；提交时自动冻结 Matcher release，并为每次运行建立 launch。脚本只写稳定参数与输入、输出根，不负责保存 SHA-256 或一次性门禁证据。实际 Job、release、launch 和动态命令记录在 `test_真实场景(Structure)/执行记录/`，不得从脚本文件名推断某次运行已完成。

`build_manifest.sh` 从 held_out_test_0 冻结 NxU 全配体和 nxu 小分子两份清单。`predict_density_only0.sh` 与 `predict_density_only106.sh` 使用原始真实受体条件，在两份清单的给定查询点上前向；`run_emerald_small.sh` 与 `run_emerald_all.sh` 分别对 nxu 和 NxU 调用官方 EMERALD-ID；`summarize_small.sh` 与 `summarize_all.sh` 只从逐点预测生成每项任务的全部保留和多身份两视图。原始受体的正式结果根是 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/`。

CryoAtom2 小分子任务使用三份独立脚本：`predict_cryo_density_only106_nxu.sh` 只对 nxu 的 357 点读取预测受体模拟密度及原子 token 并使用训练 release 前向；`run_emerald_cryo_small.sh` 保持原始实验图和 PDB 内全部候选身份，仅把受体 mmCIF 换成 CryoAtom2 最终预测结构；`summarize_cryo_small.sh` 读取这两份新结果，并通过 `--density0-root` 复用已核实输入完全相同的真实受体单通道预测。三者的科学结果根为 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/CryoAtom2/`。门禁位于 `test_真实场景(Structure)/smoke/`，与正式短命令分开。

`build_all_smiles_manifest.sh` 只为原 nxu 的 84 个 PDB 构建全类别身份轴，不重建或截尾 357 个查询点。`predict_all_smiles_density0.sh` 在真实受体条件下为单通道模型前向一次；`predict_all_smiles_density106_real.sh` 与 `predict_all_smiles_density106_cryo.sh` 分别在两种受体条件下为 106 通道模型前向一次。每次完整身份前向同时落盘不掩码和小分子掩码两种读出，不为掩码再运行模型。前向根为 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/all_smiles_input/{real,CryoAtom2}/`；单通道模型的 CryoAtom2 条件在输入等价核对后读取真实受体结果，不运行第四次前向。正式脚本文件头是提交命令；`smoke/gate_all_smiles_gpu.sh` 是隔离的 GPU 门禁，不是正式前向或画图命令。

`plot_all_smiles_real.sh` 与 `plot_all_smiles_cryo.sh` 是独立的只读汇总及画图命令，只从已有逐点结果产生两个受体条件的各一份 JSON、各四张 PNG。它们与三项正式 GPU 前向命令、实际 GPU 门禁命令分别记录在执行日志；输出根为 `/storage/penghongen/Matcher/evaluation/test_EMERALD-ID/all_smiles_comparison/{real,CryoAtom2}/`。字段与每张图的分母见 `test_真实场景(Structure)/known_site/README.md`。

`retry_cryo_chain_names.sh` 与 `retry_cryo_chain_names_last.sh` 仅用于同一次 CryoAtom2 nxu 实验的输入格式失败补跑：它们分别读取从原冻结清单派生的 21 点和 1 点队列，使用相同官方单点评分入口。旧目录先完整保存在 `CryoAtom2/EMERALD-ID/attempts/chain_name_pre_rosetta/nxu/`，再在原正式逐点位置生成新结果；这两个补跑清单不是新的测试分母。确切失败原因、Job、release、launch 和最终覆盖率见 `test_真实场景(Structure)/执行记录/CryoAtom2-nxu.md`。

每个 `.sh` 文件头给出其提交方式。CryoAtom2 小分子任务的清单、逐点字段和结果视图应先读 `test_真实场景(Structure)/known_site/README.md`，再读执行记录中的对应 Job 和产物位置。
