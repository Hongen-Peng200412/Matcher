#!/usr/bin/env bash
# 提交命令：bash 训练与运行/submit_task.sh --sh 训练与运行/sh/small_molecule_test/prepare_stage1.sh --resource a800 --gpus 1 --cpus 16 --after_hold
set -euo pipefail

config=test_smallmolecule/test_smallmolecule.yaml
python -m test_smallmolecule.build_manifest --config "${config}"
python -m test_smallmolecule.evaluate_stage1 --config "${config}"
