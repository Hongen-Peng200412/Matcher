#!/usr/bin/env bash
set -euo pipefail

# 提交命令: bash 训练与运行/submit_task.sh --sh 训练与运行/sh/train_smiles_identity.sh --resource a800 --gpus 1 --cpus 16 --time 3-00:00:00 --after_hold
# 启动 Find_1 validation 自训练与自验证驻留任务; 配置明确关闭 checkpoint 并启用 W&B online.
# TASK_PROJECT_ROOT 由 submit_task.sh 绑定到本次 release; 本脚本不回退到本地检出目录.
# CONDA_BASE 是 Conda 安装根目录; allocation 环境可显式覆盖.
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
# CONDA_ENV_NAME 是正式 Matcher 训练环境; MATCHER_CONDA_ENV 可显式覆盖.
CONDA_ENV_NAME="${MATCHER_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"
# CONFIG_PATH 是唯一正式训练入口; 数据、模型、损失、优化和 W&B 字段均由该 YAML 显式给出.
CONFIG_PATH="configs/experiments/smiles_identity/smiles_identity_find1_validation_resident_a800.yaml"

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

# PYTHONPATH 优先解析本次 release 内的 matcher 包.
export PYTHONPATH="${TASK_PROJECT_ROOT}:${PYTHONPATH:-}"
# PYTHONUNBUFFERED 让正式执行记录及时接收 Python 标准输出和错误输出.
export PYTHONUNBUFFERED=1
# OMP_NUM_THREADS 限制 OpenMP 在 DataLoader 多进程内不再扩张.
export OMP_NUM_THREADS=1
# MKL_NUM_THREADS 限制 MKL 在 DataLoader 多进程内不再扩张.
export MKL_NUM_THREADS=1
# OPENBLAS_NUM_THREADS 限制 OpenBLAS 在 DataLoader 多进程内不再扩张.
export OPENBLAS_NUM_THREADS=1
# 仅设置 CUDA 缓存分配器的最大拆分块; 不改变模型、损失或数据科学契约.
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"

cd "${TASK_PROJECT_ROOT}"
# exec 让 Python 进程继承 allocation 的锁控制和终止信号; 正常收口目录由训练入口打印.
exec python -m matcher.smiles_identity.experiment --config "${CONFIG_PATH}"
