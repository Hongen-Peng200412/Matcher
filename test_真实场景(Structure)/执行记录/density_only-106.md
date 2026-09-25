# density_only-106：已知位点身份比较日志

本日志只记录既有 `density_only-106` 模型在 NxU/nxu 已知位点任务的复用与结果，不覆盖原训练和通用测试档案。依据是 [比较计划书](../参考材料%20%26%20想法/Matcher_EMERALD-ID%20比较计划书.md)；本轮仅准备接入层和门禁，不重新训练。

## 实验结果与产物

当前没有本任务的已知位点推理、正确数、准确率或结果图片。正式 NxU 和 nxu 的逐配体 micro、PDB 平均、多身份分层与类别图均待实际前向后填写；原始分母分别固定为 2,502 与 821。

| 追溯项目 | 已核实位置或状态 |
| --- | --- |
| 正式变体 | `density_only_106`；位点侧读取 56 个密度派生通道及 50 个受体原子栅格通道 |
| 训练 release | `/home/penghongen/Feedback/Matcher/releases/Matcher_a67eb7f830e0/Matcher` |
| 冻结 BEST | `/storage/penghongen/Matcher/runs/density_only-106/train_density_only106_job386151_20260920T230728/BEST.pt` |
| 原训练与完整测试追溯 | `测试结果与追溯/experiment-centric/density_only-106/追溯与说明.md` |
| 本任务 known-site 输出 | 尚未生成；不得借用原通用测试数值 |

正式生产命令：尚未执行。

测试与门禁命令：尚未执行。

只读核查命令：已检查 `matcher/evaluation/checkpoint.py` 的训练快照加载逻辑，以及 `matcher/datasets/candidate_data.py` 中 56+50 通道构造所需的 centered 字段和受体原子字段。

## 先前尝试与限制

当前没有本任务的失败或中断尝试。`density_only-106` 并非纯实验密度模型；真实点裁块中的模拟密度和受体栅格必须与训练时的空间约定一致。`9ian` 和 `9oq7` 的不受支持真实身份对应点仍留在固定分母。
