# strongest-1 实验日志

本记录保存 `strongest-1` 的单实验追溯。正式模型使用 106 通道体素输入、A/P/V/PP token、末端无多尺度卷积的 RAUNet, 以及初值为 `0.01` 的 A/P/PP 独立门。

## 当前状态

- 阶段: H100 门禁已通过，正式训练已完成首个训练区段和首次完整 validation，继续运行。
- 正式资源: H100 Job `378587`，30 个阶段互斥 worker；`after_lock_378587` 保留。
- 正式实验目录: `测试结果与追溯/experiment-centric/strongest-1/`。
- release: `/home/penghongen/Feedback/Matcher/releases/Matcher_6701e7a7ca8d/Matcher`。
- 门禁 launch/report: `gate_strongest1_job378587_20260920T142200`；`/home/penghongen/Feedback/Matcher/gates/second_round/strongest-1/gate_job378587_20260920T142200.json`。
- 训练 launch: `train_strongest1_job378587_20260920T142850`。
- 运行根: `/storage/penghongen/Matcher/runs/strongest-1/train_strongest1_job378587_20260920T142850`。
- W&B: `https://wandb.ai/pencounkdual-111/Matcher/runs/qm0n26ai`。
- 首个训练区段与首次完整 validation: global step `465`；训练总损失 `0.7534053509`；validation 前景 PRAUC `0.7758368492`、SMILES 匹配准确率 `0.8433809820`、selection `0.8277938744`，已写入首个 BEST。
- BEST checkpoint 与两套评测产物: 待训练自然停止后填写。
- 正式 YAML: `configs/experiments/final_smiles/strongest.yaml`；实验标识 `strongest-1`，30 个阶段互斥 worker，损失权重 `0.30/0.05/0.05/1.00`。

## 正式命令

`bash 训练与运行/sh/final_smiles/train_strongest.sh`

## 失败或中断尝试

- 首轮 H100 门禁 launch `gate_strongest1_job378587_20260920T141757` 在加载 Dataset 前退出：绝对路径执行的临时门禁脚本没有把 Matcher release 根加入模块搜索路径，报 `ModuleNotFoundError: matcher`。Job 自然恢复 `try_lock`，未启动正式训练。第二次门禁只补充 `PYTHONPATH`，科学配置、release、manifest、GPU 和 worker 数均未改变。
