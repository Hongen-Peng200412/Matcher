#!/usr/bin/env bash
# 提交命令：bash 训练与运行/submit_task.sh --sh 训练与运行/sh/plot_frozen_test.sh --resource cpu --cpus 2 -- <冻结测试 JSON 与输出目录参数>
set -euo pipefail

python -m ops.matcher.plot_frozen_test "$@"
