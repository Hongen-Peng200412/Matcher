# 最优变体从头训练入口

本目录保存综合全部 validation 实验后推荐的三个零参数 Shell 入口。每个脚本只核验当前任务项目根、选择唯一 YAML，并把执行权交给 [`train_experiment.sh`](../train_experiment.sh)；脚本不创建 release、不管理 Slurm 锁、不查询服务器，也不保存一次性运行取证。

## 推荐清单

| 排名 | 推荐用途 | Shell 入口 | 唯一 YAML | 关键科学字段 | 已观测 BEST selection |
| ---: | --- | --- | --- | --- | ---: |
| 1 | 精度优先 | [`joint_identity_v4_concat_zero_gate_clip200_seed3409_lr1e4_a100.sh`](joint_identity_v4_concat_zero_gate_clip200_seed3409_lr1e4_a100.sh) | `joint_identity_v4_concat_zero_gate_clip_200_seed_3409_lr_1e4_oracle_a100.yaml` | 摘要拼接、零门、clip200、LR `1e-4`、seed `3409` | `0.7057373298` |
| 2 | 摘要拼接高分复现 | [`joint_identity_v4_concat_zero_gate_clip1_seed3408_lr1e4_a100.sh`](joint_identity_v4_concat_zero_gate_clip1_seed3408_lr1e4_a100.sh) | `joint_identity_v4_concat_zero_gate_seed_3408_lr_1e4_oracle_a100.yaml` | 摘要拼接、零门、clip1、LR `1e-4`、seed `3408` | `0.7009100312` |
| 3 | 结构简单的稳健候选 | [`joint_identity_v4_mean_max_zero_gate_clip1_seed3408_lr2e4_a100.sh`](joint_identity_v4_mean_max_zero_gate_clip1_seed3408_lr2e4_a100.sh) | `joint_identity_v4_seed_3408_lr_2e4_oracle_a100.yaml` | mean/max、零门、clip1、LR `2e-4`、seed `3408` | `0.6985805290` |

三项都使用联合身份 v4、SMI-TED、oracle、`all_non_background`、BF16、物理批量 `8`、梯度累积 `4`、global batch `32` 和从零初始化。`training.seed` 固定为对应实测 seed，用于复现实验轨迹；它不表示已经证明该 seed 在未来训练中必然最优。

排名 1 是当前最高 validation 结果。排名 2 保留 clip1，用于复核摘要拼接的高分轨迹。排名 3 不使用摘要拼接，适合优先控制模型结构复杂度的训练。没有提供“摘要拼接 + LR `2e-4`”脚本，因为该组合没有被实际实验验证。

## 提交方式

在 Matcher 项目根目录复制目标脚本头部的 `submit_task.sh` 命令。三个脚本都不接受额外参数；提交工具负责创建任务项目根并设置 `TASK_PROJECT_ROOT`。

每份 YAML 中：

- `data.data_root` 指向 AdaLigand PDB、occurrence 与 PreparedLigand 输入根目录。
- `data.stage1_root` 指向只读 Stage1 Find_0 产物根目录。
- `data.output_root` 指向版本化 manifest 与候选缓存根目录。
- 顶层 `output_root` 指向训练运行、checkpoint 与指标根目录。
- `initialize_candidate_from: null` 表示不加载候选预训练 checkpoint，从头训练完整模型。
- `tracking.mode: online` 要求 W&B online 初始化成功；失败时训练入口关闭，不静默降级。

脚本使用当前代码与当前 YAML 解释这些字段，不封装历史 release、launch 或依赖环境。配置格式、清单 schema、依赖或缓存契约变化后，应先通过正式 YAML 解析和相称测试，再提交训练。

推荐依据、指标分解和不可外推边界见[结果分析与最优结果](../../../文档/记录/结果分析与最优结果.md)。
