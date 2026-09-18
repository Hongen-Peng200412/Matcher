#!/usr/bin/env bash
# 提交命令：bash 训练与运行/submit_task.sh --sh 训练与运行/sh/final_smiles/train_strongest.sh --resource h100 --gpus 1 --cpus 32
set -euo pipefail

python -m matcher.training.experiment \
  --config configs/experiments/final_smiles/strongest.yaml
