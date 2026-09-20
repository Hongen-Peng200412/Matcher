#!/usr/bin/env bash
# 提交命令：bash 训练与运行/submit_task.sh --sh 训练与运行/sh/small_molecule_test/evaluate_density_only_106.sh --resource h100 --gpus 1 --cpus 32
set -euo pipefail

python -m ops.small_molecule_test.evaluate_model \
    --config ops/small_molecule_test/test_smallmolecule.yaml \
    --model density_only-106
