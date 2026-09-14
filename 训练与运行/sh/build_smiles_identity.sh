#!/usr/bin/env bash
set -euo pipefail

# 提交命令: bash 训练与运行/submit_task.sh --sh 训练与运行/sh/build_smiles_identity.sh --resource cpu --cpus 16
# 构建全局 SMILES 图、自同构、Find_1 validation 标签旁路和正式清单.
# TASK_PROJECT_ROOT 由 submit_task.sh 绑定到本次 release; 本脚本不回退到本地检出目录.
# CONDA_BASE 是 Conda 安装根目录; allocation 环境可显式覆盖.
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
# CHEMISTRY_ENV_NAME 是含 RDKit 的资产构建环境; MATCHER_CHEMISTRY_ENV 可显式覆盖.
CHEMISTRY_ENV_NAME="${MATCHER_CHEMISTRY_ENV:-AdaLigand_stage1_py310}"
# MATCHER_ENV_NAME 是标签和清单构建环境; MATCHER_CONDA_ENV 可显式覆盖.
MATCHER_ENV_NAME="${MATCHER_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"
# DATA_ROOT 是顶层 PreparedLigand.smiles 与旧迁移产物的只读来源.
DATA_ROOT="/storage/penghongen/AdaLigand/Ori_Data"
# STAGE1_ROOT 提供 Find_1 centered validation 候选、source_blob_index 和预测数组, 并接收或覆盖各 PDB centered 目录内的 F2_centered_label.npz.
STAGE1_ROOT="/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1"
# OUTPUT_ROOT 只接收正式 manifest 目录; 全局图包与自同构包写回 DATA_ROOT 的固定派生目录.
OUTPUT_ROOT="/storage/penghongen/tmp/Matcher"

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CHEMISTRY_ENV_NAME}"
set -u

# PYTHONPATH 优先解析本次 release 内的 matcher 包.
export PYTHONPATH="${TASK_PROJECT_ROOT}:${PYTHONPATH:-}"
# PYTHONUNBUFFERED 让正式执行记录及时接收 Python 标准输出和错误输出.
export PYTHONUNBUFFERED=1
# OMP_NUM_THREADS 限制 OpenMP 在线程外层并行中不再扩张.
export OMP_NUM_THREADS=1
# MKL_NUM_THREADS 限制 MKL 在线程外层并行中不再扩张.
export MKL_NUM_THREADS=1

cd "${TASK_PROJECT_ROOT}"
# 第一阶段在含 RDKit 的化学环境中按精确顶层 SMILES 构建全局图包和自同构包.
python -m matcher.smiles_identity.build_assets --data-root "${DATA_ROOT}"
set +u
conda activate "${MATCHER_ENV_NAME}"
set -u
# 第二阶段在 Matcher 环境中按 PDB 并行写 validation 标签旁路; worker 数优先取 Slurm 分配值.
python -m matcher.smiles_identity.write_labels \
    --data-root "${DATA_ROOT}" \
    --stage1-root "${STAGE1_ROOT}" \
    --split validation \
    --coverage-threshold 0.30 \
    --mode overwrite \
    --workers "${SLURM_CPUS_PER_TASK:-16}"
# 第三阶段只引用已完成资产和旁路, 写出正式 validation 清单.
python -m matcher.smiles_identity.manifest \
    --data-root "${DATA_ROOT}" \
    --stage1-root "${STAGE1_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --split validation
