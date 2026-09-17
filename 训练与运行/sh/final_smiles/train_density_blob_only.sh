#!/usr/bin/env bash
# 提交命令：bash 训练与运行/submit_task.sh --sh 训练与运行/sh/final_smiles/train_density_blob_only.sh --resource a800 --gpus 1 --cpus 24
set -euo pipefail

python -m matcher.training.experiment \
  --config configs/experiments/final_smiles/density_blob_only.yaml
