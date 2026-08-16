#!/usr/bin/env bash
# 提交命令：bash 训练与运行/submit_task.sh --sh 训练与运行/sh/最优变体推荐/joint_identity_v4_concat_zero_gate_clip200_seed3409_lr1e4_a100.sh --resource a100 --gpus 1 --cpus 16
# 唯一计划 YAML：configs/experiments/joint_identity_v4/joint_identity_v4_concat_zero_gate_clip_200_seed_3409_lr_1e4_oracle_a100.yaml。
# 科学设置：联合身份 v4、SMI-TED、oracle、摘要拼接、零门、clip200、LR 1e-4、training seed 3409、从零训练。
# data.data_root：/storage/penghongen/AdaLigand/Ori_Data；只读 AdaLigand PDB、occurrence 与 PreparedLigand 输入根目录。
# data.stage1_root：/storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477/artifacts/Find_0；只读 Stage1 Find_0 产物根目录。
# data.output_root：/storage/penghongen/tmp/Matcher；可写版本化 manifest 与候选缓存根目录。
# 顶层 output_root：/storage/penghongen/tmp/Matcher/experiments；训练运行、checkpoint 与指标根目录。
# TASK_PROJECT_ROOT：由 训练与运行/submit_task.sh 设置为本次任务的项目根目录。
# Conda 可选覆盖：训练与运行/sh/train_experiment.sh 定义 CONDA_BASE 与 MATCHER_CONDA_ENV。
# 兼容边界：当前代码与依赖解释当前 YAML；配置或依赖变化后必须重新通过正式解析和训练门禁。
set -euo pipefail

[[ "$#" -eq 0 ]] || { printf '错误：推荐变体脚本不接受参数。\n' >&2; exit 2; }
[[ -n "${TASK_PROJECT_ROOT:-}" && -d "${TASK_PROJECT_ROOT}" ]] || { printf '错误：TASK_PROJECT_ROOT 必须指向当前任务项目根目录。\n' >&2; exit 2; }

cd -- "${TASK_PROJECT_ROOT}"
plan='configs/experiments/joint_identity_v4/joint_identity_v4_concat_zero_gate_clip_200_seed_3409_lr_1e4_oracle_a100.yaml'
[[ -f "${plan}" && -f '训练与运行/sh/train_experiment.sh' ]] || { printf '错误：当前项目缺少训练入口或推荐计划 %s。\n' "${plan}" >&2; exit 2; }
exec bash '训练与运行/sh/train_experiment.sh' --plan "${plan}"
