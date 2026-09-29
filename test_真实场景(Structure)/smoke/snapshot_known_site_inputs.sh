#!/usr/bin/env bash
# 一次性追溯命令：bash 训练与运行/submit_task.sh --sh 'test_真实场景(Structure)/smoke/snapshot_known_site_inputs.sh' --resource cpu --cpus 8 --partition debug --time 06:00:00
set -euo pipefail
cd "${TASK_PROJECT_ROOT}"
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python 'test_真实场景(Structure)/smoke/snapshot_known_site_inputs.py' \
  --manifest-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest \
  --source-manifest /storage/penghongen/Matcher/manifests/test0_real_smiles_v1/manifest.json \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --checkpoint /storage/penghongen/Matcher/runs/density_only-0/train_density_only0_job379403_20260920T205216/BEST.pt \
  --checkpoint /storage/penghongen/Matcher/runs/density_only-106/train_density_only106_job386151_20260920T230728/BEST.pt \
  --tutorial-root "${TASK_PROJECT_ROOT}/test_真实场景(Structure)/参考材料 & 想法/tutorials/tutorials/EMERALD-ID" \
  --output-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/provenance
