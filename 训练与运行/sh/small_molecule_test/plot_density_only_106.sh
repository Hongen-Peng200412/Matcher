#!/usr/bin/env bash
# 提交命令：bash 训练与运行/submit_task.sh --sh 训练与运行/sh/small_molecule_test/plot_density_only_106.sh --resource cpu --cpus 2
set -euo pipefail

python -m test_smallmolecule.plot_results \
    --config test_smallmolecule/test_smallmolecule.yaml \
    --target density_only-106
