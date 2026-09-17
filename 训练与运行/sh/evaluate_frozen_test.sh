#!/usr/bin/env bash
# 提交命令：bash 训练与运行/submit_task.sh --sh 训练与运行/sh/evaluate_frozen_test.sh --resource a800 --gpus 1 --cpus 24 -- <BEST 与两份测试 manifest 参数>
set -euo pipefail

python -m ops.matcher.evaluate_frozen_test "$@"
