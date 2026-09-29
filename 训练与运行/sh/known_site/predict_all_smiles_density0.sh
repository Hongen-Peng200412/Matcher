#!/usr/bin/env bash
# 提交命令: bash 训练与运行/submit_task.sh --sh 训练与运行/sh/known_site/predict_all_smiles_density0.sh --resource a100 --gpus 1 --cpus 4 --after_hold
set -euo pipefail
cd "${TASK_PROJECT_ROOT}"
export PYTHONPATH="${TASK_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python 'test_真实场景(Structure)/known_site/predict_matcher.py' \
  --manifest-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest \
  --identity-manifest /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest_all_smiles/nxu/pdbs.jsonl \
  --source-manifest /storage/penghongen/Matcher/manifests/test0_real_smiles_v1/manifest.json \
  --checkpoint /storage/penghongen/Matcher/runs/density_only-0/train_density_only0_job379403_20260920T205216/BEST.pt \
  --model-name density_only-0 \
  --output-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/all_smiles_input/real \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --task-names nxu
