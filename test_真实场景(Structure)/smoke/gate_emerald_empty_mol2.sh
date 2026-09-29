#!/usr/bin/env bash
# 门禁命令：bash test_真实场景(Structure)/smoke/gate_emerald_empty_mol2.sh
set -euo pipefail
cd "${TASK_PROJECT_ROOT}"
export PYTHONPATH="${TASK_PROJECT_ROOT}/test_真实场景(Structure)/known_site:${TASK_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python 'test_真实场景(Structure)/smoke/gate_emerald_empty_mol2.py' \
  --manifest-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest \
  --output-root /storage/penghongen/tmp/Matcher/known_site_empty_mol2_gate_20260926_gnode11
