# blob_only-0 实验日志

本记录保存 `blob_only-0` 的单实验追溯。正式模型的候选侧只使用 V/blob token、V 的 count/mean/max 和四项裁剪统计; 身份侧仍使用 SMI-TED、SMILES 图和同 PDB 身份清单。

## 当前状态

- 阶段: 正式训练已按 patience 自然早停；全测试集与 Stage3 小分子正式评估及绘图均已完成并归档。
- 正式资源: A800 数组任务 `379403_1`，锁与反馈目录使用基础编号 `379403`，15 个阶段互斥 worker；`after_lock_379403` 保留。
- 正式实验目录: `测试结果与追溯/experiment-centric/blob_only-0/`。
- release: `/home/penghongen/Feedback/Matcher/releases/Matcher_6701e7a7ca8d/Matcher`。
- 门禁 launch/report: `gate_blob_only0_job379403_20260920T142200`；`/home/penghongen/Feedback/Matcher/gates/second_round/blob_only-0/gate_job379403_20260920T142200.json`。
- 训练 launch: `train_blob_only0_job379403_20260920T142850`。
- 运行根: `/storage/penghongen/Matcher/runs/blob_only-0/train_blob_only0_job379403_20260920T142850`。
- W&B: `https://wandb.ai/pencounkdual-111/Matcher/runs/6zqysvlb`。
- 首个训练区段与首次完整 validation: global step `465`；训练总损失 `0.5835359650`；validation 前景 PRAUC `0.7695570682`、SMILES 匹配准确率 `0.8716594158`、selection `0.8480973356`，已写入首个 BEST。
- 冻结 BEST: `/storage/penghongen/Matcher/runs/blob_only-0/train_blob_only0_job379403_20260920T142850/BEST.pt`；SHA-256 `0af958d61aeefd7d0dc723a7962fd88e83d47d7c6659e7343464ee0f88381431`。
- BEST 位于 epoch `6`、第 `20` 次 validation、global step `9300`：前景 PRAUC `0.7832593992`、SMILES 匹配准确率 `0.9111249223`、selection `0.8816174939`。
- 最后状态: epoch `9`、global step `13020`、共 `28` 次完整 validation 与 `28` 份逐检查 checkpoint；累计实际降学习率两次后按 patience 自然早停。
- 冻结 validation 前景阈值 `0.30859375`；阈值只由最终 BEST 的 validation 预测拟合。
- 全测试集冻结结果: `/storage/penghongen/Matcher/evaluation/final_smiles/blob_only-0/frozen_test.json`；SHA-256 `7c9a38896c3178565e71c5ca0c0039a0b617a0425a950df655afa1490ffded1c`。
- Stage3 小分子冻结结果: `/storage/penghongen/Matcher/evaluation/test_smallmolecule_v1/blob_only-0/train_blob_only0_job379403_20260920T142850/evaluation.json`；SHA-256 `28eb42ced75d79fd20aeca60699341197b160ea52d0595c913a3ad70761ace5b`。
- 本地归档: `测试结果与追溯/experiment-centric/blob_only-0/`；包含两份文档、三张全测试集图和四张小分子图，图片哈希已与服务器正式产物逐项一致。
- A800 已恢复 `try_lock_379403`，`after_lock_379403` 继续保留；该 allocation 将用于稍后并行执行 `strongest-1` 的小分子冻结评估，不提前释放。
- 正式 YAML: `configs/experiments/final_smiles/blob_only.yaml`；实验标识 `blob_only-0`，15 个阶段互斥 worker，损失权重 `0.30/0/0/1.00`。

## 正式命令

`bash 训练与运行/sh/final_smiles/train_blob_only.sh`

`bash 训练与运行/sh/final_smiles/evaluate_blob_only.sh`

`bash 训练与运行/sh/small_molecule_test/evaluate_blob_only.sh`

`bash 训练与运行/sh/small_molecule_test/plot_blob_only.sh`

## 失败或中断尝试

- 首轮 A800 门禁 launch `gate_blob_only0_job379403_20260920T141757` 在加载 Dataset 前退出：绝对路径执行的临时门禁脚本没有把 Matcher release 根加入模块搜索路径，报 `ModuleNotFoundError: matcher`。数组任务自然恢复基础锁 `try_lock_379403`，未启动正式训练。第二次门禁只补充 `PYTHONPATH`，科学配置、release、manifest、GPU 和 worker 数均未改变。
