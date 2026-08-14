#!/usr/bin/env bash
set -euo pipefail

# 构建正式清单，并按 --products 用申请到的 CPU 核并行生成体素或 PP token 缓存。
CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
CONDA_ENV_NAME="${MATCHER_CONDA_ENV:-Pocket_Plus_centos7_cu121_allgpu}"

set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
set -u

export PYTHONPATH="${TASK_PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${TASK_PROJECT_ROOT}"
exec python -m matcher.data.precompute "$@"
