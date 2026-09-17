#!/usr/bin/env bash
# 提交命令：bash 训练与运行/submit_task.sh --sh 训练与运行/sh/build_stage1_manifest.sh --resource cpu --cpus 16 -- <构建参数>
set -euo pipefail

python -m ops.matcher.build_stage1_manifest "$@"
