#!/usr/bin/env bash
# 门禁命令: bash 'test_真实场景(Structure)/smoke/gate_cryo_chain_names.sh'
set -euo pipefail
cd "${TASK_PROJECT_ROOT}"
export PYTHONPATH="${TASK_PROJECT_ROOT}/test_真实场景(Structure)/known_site${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python 'test_真实场景(Structure)/smoke/gate_cryo_emerald_nxu.py' \
  --manifest-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --receptor-root /storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/test_0_chain06/Ori_Data \
  --tutorial-root 'test_真实场景(Structure)/参考材料 & 想法/tutorials/tutorials/EMERALD-ID' \
  --scratch-root /storage/penghongen/tmp/Matcher \
  --pdb-id 9nc9 --occurrence-id 0
