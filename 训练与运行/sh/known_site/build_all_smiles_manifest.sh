#!/usr/bin/env bash
# 提交命令: bash 训练与运行/submit_task.sh --sh 训练与运行/sh/known_site/build_all_smiles_manifest.sh --resource cpu --cpus 4
set -euo pipefail
cd "${TASK_PROJECT_ROOT}"
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python 'test_真实场景(Structure)/known_site/build_all_smiles_manifest.py' \
  --query-manifest-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest \
  --data-root /storage/penghongen/AdaLigand/Ori_Data \
  --output-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest_all_smiles/nxu
