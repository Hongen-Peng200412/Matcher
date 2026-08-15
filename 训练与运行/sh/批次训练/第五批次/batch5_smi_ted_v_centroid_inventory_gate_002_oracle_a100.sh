#!/usr/bin/env bash
# 提交命令：bash 训练与运行/submit_task.sh --sh 训练与运行/sh/批次训练/第五批次/batch5_smi_ted_v_centroid_inventory_gate_002_oracle_a100.sh --resource a100 --gpus 1 --cpus 16
# 唯一计划 YAML：configs/experiments/batch5/batch5_smi_ted_v_centroid_inventory_gate_002_oracle_a100.yaml。
# data.roots.data_root：/storage/penghongen/AdaLigand/Ori_Data；只读 AdaLigand 输入根目录。
# data.roots.stage1_root：/storage/penghongen/AdaLigand_stage1_inference/Find_0-CPC1-ligand_PRAUC_0.675477/artifacts/Find_0；只读 Stage1 Find_0 输入根目录。
# data.roots.output_root：/storage/penghongen/tmp/Matcher；可写清单与缓存根目录。
# 顶层 output_root：/storage/penghongen/tmp/Matcher/experiments；训练运行与指标根目录。
# TASK_PROJECT_ROOT：由 训练与运行/submit_task.sh 设置为冻结任务项目根目录。
# Conda 可选覆盖：训练与运行/sh/train_experiment.sh 定义 CONDA_BASE 与 MATCHER_CONDA_ENV。
# 历史兼容性：当前代码和依赖解释历史 YAML，不保证复现最初实验环境。
set -euo pipefail

[[ "$#" -eq 0 ]] || { printf '错误：固定计划脚本不接受参数。\n' >&2; exit 2; }
[[ -n "${TASK_PROJECT_ROOT:-}" && -d "${TASK_PROJECT_ROOT}" ]] || { printf '错误：TASK_PROJECT_ROOT 必须指向当前任务项目根目录。\n' >&2; exit 2; }

cd -- "${TASK_PROJECT_ROOT}"
plan='configs/experiments/batch5/batch5_smi_ted_v_centroid_inventory_gate_002_oracle_a100.yaml'
[[ -f "${plan}" && -f '训练与运行/sh/train_experiment.sh' ]] || { printf '错误：当前项目缺少训练入口或固定计划 %s。\n' "${plan}" >&2; exit 2; }
exec bash '训练与运行/sh/train_experiment.sh' --plan "${plan}"
