#!/usr/bin/env bash
# 画图与汇总命令: bash 训练与运行/submit_task.sh --sh 训练与运行/sh/known_site/plot_all_smiles_real.sh --resource cpu --cpus 4
set -euo pipefail
cd "${TASK_PROJECT_ROOT}"
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python 'test_真实场景(Structure)/known_site/summarize_all_smiles.py' \
  --manifest-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest \
  --all-identity-manifest /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/manifest_all_smiles/nxu/pdbs.jsonl \
  --prediction-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID \
  --cryo-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/CryoAtom2 \
  --all-prediction-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/all_smiles_input \
  --condition real \
  --output-root /storage/penghongen/Matcher/evaluation/test_EMERALD-ID/all_smiles_comparison/real
