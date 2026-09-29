#!/usr/bin/env bash
# 门禁命令: bash 'test_真实场景(Structure)/smoke/gate_all_smiles_gpu.sh' density_only-0 real
set -euo pipefail
cd "${TASK_PROJECT_ROOT}"
export PYTHONPATH="${TASK_PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
model="$1"
receptor="$2"
if [[ "${model}" == density_only-0 ]]; then
  checkpoint=/storage/penghongen/Matcher/runs/density_only-0/train_density_only0_job379403_20260920T205216/BEST.pt
else
  checkpoint=/storage/penghongen/Matcher/runs/density_only-106/train_density_only106_job386151_20260920T230728/BEST.pt
fi
receptor_root=/storage/penghongen/AdaLigand/Ori_Data
if [[ "${receptor}" == CryoAtom2 ]]; then
  receptor_root=/storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/test_0_chain06/Ori_Data
fi
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python 'test_真实场景(Structure)/smoke/gate_known_site_gpu.py' \
  --manifest-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest \
  --identity-manifest /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest_all_smiles/nxu/pdbs.jsonl \
  --source-manifest /storage/penghongen/Matcher/manifests/test0_real_smiles_v1/manifest.json \
  --checkpoint "${checkpoint}" \
  --model-name "${model}" \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --receptor-root "${receptor_root}" \
  --task-names nxu \
  --scratch-root /storage/penghongen/tmp/Matcher
