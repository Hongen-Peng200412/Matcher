#!/usr/bin/env bash
# 门禁命令：bash test_真实场景(Structure)/smoke/gate_emerald_missing_params.sh
set -euo pipefail
cd "${TASK_PROJECT_ROOT}"
export PYTHONPATH="${TASK_PROJECT_ROOT}/test_真实场景(Structure)/known_site:${TASK_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python 'test_真实场景(Structure)/smoke/gate_emerald_missing_params.py' \
  --manifest-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --output-root /storage/penghongen/tmp/Matcher/known_site_missing_params_gate_20260926_gnode11 \
  --tutorial-root 'test_真实场景(Structure)/参考材料 & 想法/tutorials/tutorials/EMERALD-ID'
