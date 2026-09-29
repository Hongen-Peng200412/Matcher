#!/usr/bin/env bash
# 提交命令: bash 训练与运行/submit_task.sh --sh 训练与运行/sh/known_site/summarize_all.sh --resource cpu --cpus 8 --time 02:00:00
set -euo pipefail
cd "${TASK_PROJECT_ROOT}"
export MPLBACKEND=Agg
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python 'test_真实场景(Structure)/known_site/summarize.py' \
  --manifest-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest \
  --task-name NxU \
  --prediction-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID \
  --output-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/summary
