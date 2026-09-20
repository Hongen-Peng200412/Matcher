#!/usr/bin/env bash
# 提交命令：bash 训练与运行/submit_task.sh --sh 训练与运行/sh/final_smiles/evaluate_strongest_1.sh --resource h100 --gpus 1 --cpus 32
set -euo pipefail

output_root=/storage/penghongen/Matcher/evaluation/final_smiles/strongest-1
result_path="${output_root}/frozen_test.json"
mkdir -p "${output_root}"

python -m ops.matcher.evaluate_frozen_test \
    --checkpoint /storage/penghongen/Matcher/runs/strongest-1/train_strongest1_job378587_20260920T142850/BEST.pt \
    --real-manifest /storage/penghongen/Matcher/manifests/test0_real_smiles_v1/manifest.json \
    --cryoatom2-manifest /storage/penghongen/Matcher/manifests/test0_cryoatom2_smiles_v1/manifest.json \
    --split held_out_test_0 \
    --stage1-preview /storage/penghongen/Matcher/evaluation/stage1_only_preview/stage1_only_preview.json \
    --output-path "${result_path}" \
    --test1-pdb-list /storage/penghongen/AdaLigand/held_out/split/held_out_06_chain/test_1.json \
    --batch-size 32 \
    --num-workers 30 \
    --pp-top-k 2048 \
    --cache-gib-per-worker 8.0

python -m ops.matcher.plot_frozen_test \
    --result-path "${result_path}" \
    --output-directory "${output_root}"
