# Matcher 正式 shell 入口

本目录只保存简短、自包含的人类入口。每个脚本在文件头给出 `训练与运行/submit_task.sh` 提交命令；Job、release、launch、SHA-256、锁操作和一次性取证只进入任务执行记录。

## 数据与评估

| 脚本 | 作用 |
|---|---|
| `build_smiles_assets.sh` | 发布版本一全局 SMILES 图包和 Stage3 自同构包 |
| `build_stage1_manifest.sh` | 扫描一个已经闭合的 Stage1 划分并写入同一 manifest |
| `build_final_train_validation_manifest.sh` | 顺序扫描完整 train 与 validation，发布本轮冻结训练 manifest |
| `build_stage1_only_preview.sh` | 冻结 Find_1 与 Emap2lig 的 Stage1-only top-k 元数据 |
| `plot_stage1_only_preview.sh` | 从冻结元数据绘制一张明确标记的预览图 |
| `evaluate_frozen_test.sh` | 阶段 E 用同一 BEST 评估真实受体、CryoAtom2 和可选 test_1 子集 |

`build_stage1_manifest.sh` 不应对仍在写入的训练分片运行。当前负责第 6 片推理的独立任务不由这些脚本接管。

## 四项正式训练

[`final_smiles/`](final_smiles/README.md) 只有 `strongest`、`density_blob_only`、`pocket_only` 和 `trunk_only` 四个从零训练入口。它们在阶段 E 读取同一份冻结 train/validation manifest。

## 历史目录

旧批次 YAML 与启动脚本已经退出活动科学主线，由 Git 历史保持可恢复。`批次训练/` 和 `最优变体推荐/` 中若只剩空目录或旧 README，不是当前训练入口。
