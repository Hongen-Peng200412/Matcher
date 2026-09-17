#!/usr/bin/env bash
# 提交命令：bash 训练与运行/submit_task.sh --sh 训练与运行/sh/build_stage1_only_preview.sh --resource cpu --cpus 8 -- <评估来源与输出参数>
set -euo pipefail

python -m matcher.evaluation.test_set "$@"
