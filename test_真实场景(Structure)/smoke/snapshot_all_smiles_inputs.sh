#!/usr/bin/env bash
# 追溯命令: bash 训练与运行/submit_task.sh --sh 'test_真实场景(Structure)/smoke/snapshot_all_smiles_inputs.sh' --resource cpu --cpus 8
set -euo pipefail
cd "${TASK_PROJECT_ROOT}"
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python 'test_真实场景(Structure)/smoke/snapshot_all_smiles_inputs.py' \
  --manifest-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest \
  --all-identity-manifest /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest_all_smiles/nxu/pdbs.jsonl \
  --source-manifest /storage/penghongen/Matcher/manifests/test0_real_smiles_v1/manifest.json \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --cryo-root /storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/test_0_chain06/Ori_Data \
  --checkpoint /storage/penghongen/Matcher/runs/density_only-0/train_density_only0_job379403_20260920T205216/BEST.pt \
  --checkpoint /storage/penghongen/Matcher/runs/density_only-106/train_density_only106_job386151_20260920T230728/BEST.pt \
  --output-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/provenance/all_smiles_input
