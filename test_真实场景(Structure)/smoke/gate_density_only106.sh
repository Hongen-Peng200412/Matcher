#!/usr/bin/env bash
# 门禁命令：bash test_真实场景(Structure)/smoke/gate_density_only106.sh
set -euo pipefail
cd "${TASK_PROJECT_ROOT}"
export PYTHONPATH="${TASK_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python 'test_真实场景(Structure)/smoke/gate_known_site_gpu.py' \
  --manifest-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest \
  --source-manifest /storage/penghongen/Matcher/manifests/test0_real_smiles_v1/manifest.json \
  --checkpoint /storage/penghongen/Matcher/runs/density_only-106/train_density_only106_job386151_20260920T230728/BEST.pt \
  --model-name density_only-106 \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --scratch-root /storage/penghongen/tmp/Matcher
