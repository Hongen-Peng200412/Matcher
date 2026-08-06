# Matcher v2 模式一运行入口

`prepare_manifest_cpu.sh` 从新版 Stage1-Find BOX 池继承 train/validation PDB 成员，并加入既有 calibration 清单；它只生成模式一正式清单，不运行 GPU 模型。

`train.sh` 依次完成 Phase1、Phase2、validation 推理评估和 calibration 推理评估。两阶段均使用在线 W&B；若阶段目录已有 `latest.pt`，脚本从该 checkpoint 续训。GPU 资源和锁由项目通用提交器管理，本目录脚本不写 Slurm 参数，也不设置运行时限。
