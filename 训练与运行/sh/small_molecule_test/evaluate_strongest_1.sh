#!/usr/bin/env bash
# 提交命令：bash 训练与运行/submit_task.sh --sh 训练与运行/sh/small_molecule_test/evaluate_strongest_1.sh --resource a800 --gpus 1 --cpus 16
set -euo pipefail

python -m test_smallmolecule.evaluate_model \
    --config test_smallmolecule/test_smallmolecule.yaml \
    --model strongest-1
