#!/usr/bin/env bash
# 提交命令: bash 训练与运行/submit_task.sh --sh 训练与运行/sh/known_site/retry_cryo_chain_names.sh --resource h200 --gpus 1 --cpus 4 --after_hold
set -euo pipefail
cd "${TASK_PROJECT_ROOT}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python 'test_真实场景(Structure)/known_site/run_emerald_shard.py' \
  --manifest-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/CryoAtom2/retry_chain_names_manifest \
  --task-name nxu \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --receptor-root /storage/penghongen/Adaligand_infered_receptor_data/cryoatom2/test_0_chain06/Ori_Data \
  --output-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/CryoAtom2/EMERALD-ID \
  --tutorial-root 'test_真实场景(Structure)/参考材料 & 想法/tutorials/tutorials/EMERALD-ID' \
  --shard-index 0 --shard-count 1 --workers 4
