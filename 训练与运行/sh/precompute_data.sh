#!/usr/bin/env bash
set -euo pipefail

# 构建正式清单，并用申请到的 CPU 核并行生成候选 105 通道体素缓存。
cd "${TASK_PROJECT_ROOT}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

exec python -m matcher.precompute "$@"
