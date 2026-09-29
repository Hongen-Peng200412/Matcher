#!/usr/bin/env bash
# 提交命令: bash 训练与运行/submit_task.sh --sh 训练与运行/sh/known_site/record_unfinished_small.sh --resource cpu --cpus 1 --time 01:00:00
set -euo pipefail
cd "${TASK_PROJECT_ROOT}"
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python 'test_真实场景(Structure)/known_site/record_unfinished.py' \
  --manifest-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest \
  --task-name nxu \
  --output-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/EMERALD-ID
