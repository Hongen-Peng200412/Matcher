#!/usr/bin/env bash
# 提交命令: bash 训练与运行/submit_task.sh --sh 训练与运行/sh/known_site/run_emerald_small.sh --resource cpu --cpus 8 --array '0-11%12' --time 72:00:00 --after_hold
set -euo pipefail
cd "${TASK_PROJECT_ROOT}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python 'test_真实场景(Structure)/known_site/run_emerald_shard.py' \
  --manifest-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest \
  --task-name nxu \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --output-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/EMERALD-ID \
  --tutorial-root 'test_真实场景(Structure)/参考材料 & 想法/tutorials/tutorials/EMERALD-ID' \
  --shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count 12 --workers 8
