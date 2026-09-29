#!/usr/bin/env bash
# 提交命令: bash 训练与运行/submit_task.sh --sh 训练与运行/sh/known_site/predict_density_only106.sh --resource a800 --gpus 1 --cpus 4 --time 24:00:00 --after_hold
set -euo pipefail
cd "${TASK_PROJECT_ROOT}"
export PYTHONPATH="${TASK_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python 'test_真实场景(Structure)/known_site/predict_matcher.py' \
  --manifest-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest \
  --source-manifest /storage/penghongen/Matcher/manifests/test0_real_smiles_v1/manifest.json \
  --checkpoint /storage/penghongen/Matcher/runs/density_only-106/train_density_only106_job386151_20260920T230728/BEST.pt \
  --model-name density_only-106 \
  --output-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID \
  --data-root /storage/penghongen/AdaLigand/Ori_Data
