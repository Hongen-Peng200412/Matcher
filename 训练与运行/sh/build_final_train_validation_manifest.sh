#!/usr/bin/env bash
# 提交命令：bash 训练与运行/submit_task.sh --sh 训练与运行/sh/build_final_train_validation_manifest.sh --resource cpu --cpus 16
set -euo pipefail

stage1_root=/storage/penghongen/AdaLigand_stage1_inference/Find_1/真实受体/artifacts/Find_1
data_root=/storage/penghongen/AdaLigand/Ori_Data
graph_asset=/storage/penghongen/AdaLigand/Ori_Data/smiles_assets/smiles_graphs_v1.npz
manifest=/storage/penghongen/Matcher/manifests/final_train_validation_shards01_06_v1/manifest.json

python -m ops.matcher.build_stage1_manifest \
  --stage1-root "${stage1_root}" \
  --split train \
  --output-path "${manifest}" \
  --data-root "${data_root}" \
  --graph-asset-path "${graph_asset}"

python -m ops.matcher.build_stage1_manifest \
  --stage1-root "${stage1_root}" \
  --split validation \
  --output-path "${manifest}" \
  --data-root "${data_root}" \
  --graph-asset-path "${graph_asset}"
