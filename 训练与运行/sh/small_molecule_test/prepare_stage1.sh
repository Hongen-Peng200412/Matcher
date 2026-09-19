#!/usr/bin/env bash
# 提交命令：bash 训练与运行/submit_task.sh --sh 训练与运行/sh/small_molecule_test/prepare_stage1.sh --resource a800 --gpus 1 --cpus 16 --after_hold
set -euo pipefail

config=ops/small_molecule_test/test_smallmolecule.yaml
python -m ops.small_molecule_test.build_manifest --config "${config}"
python -m ops.small_molecule_test.evaluate_stage1 --config "${config}"
