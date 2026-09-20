#!/usr/bin/env bash
# 提交命令：bash 训练与运行/submit_task.sh --sh 训练与运行/sh/small_molecule_test/plot_strongest_1.sh --resource cpu --cpus 2
set -euo pipefail

python -m ops.small_molecule_test.plot_results \
    --config ops/small_molecule_test/test_smallmolecule.yaml \
    --target strongest-1
