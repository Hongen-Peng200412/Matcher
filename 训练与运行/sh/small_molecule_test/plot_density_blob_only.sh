#!/usr/bin/env bash
# 提交命令：bash 训练与运行/submit_task.sh --sh 训练与运行/sh/small_molecule_test/plot_density_blob_only.sh --resource a800 --gpus 1 --cpus 16 --after_hold
set -euo pipefail

python -m ops.small_molecule_test.plot_results \
    --config ops/small_molecule_test/test_smallmolecule.yaml \
    --target density_blob_only
