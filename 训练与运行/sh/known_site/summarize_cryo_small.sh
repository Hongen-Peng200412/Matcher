#!/usr/bin/env bash
# 提交命令: bash 训练与运行/submit_task.sh --sh 训练与运行/sh/known_site/summarize_cryo_small.sh --resource cpu --cpus 8
set -euo pipefail
cd "${TASK_PROJECT_ROOT}"
export MPLBACKEND=Agg
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python 'test_真实场景(Structure)/known_site/summarize.py' \
  --manifest-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest \
  --task-name nxu \
  --prediction-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/CryoAtom2 \
  --density0-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID \
  --output-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/CryoAtom2/summary
