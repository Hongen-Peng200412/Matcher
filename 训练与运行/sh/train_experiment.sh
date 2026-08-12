#!/usr/bin/env bash
set -euo pipefail

# 正式 Matcher 实验入口；任务参数原样交给 Python 训练入口解释。
cd "${TASK_PROJECT_ROOT}"
exec python -m matcher.experiment "$@"
