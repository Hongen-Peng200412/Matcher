#!/usr/bin/env bash
set -euo pipefail

# 两个数组元素各使用一个独立真实点, 各自准备输入后调用官方 Rosetta 协议.
case "${SLURM_ARRAY_TASK_ID}" in
    0) pdb_id=9nal; occurrence_id=0; task=small ;;
    1) pdb_id=9nnz; occurrence_id=0; task=all ;;
    *) exit 2 ;;
esac

smoke_root=/storage/penghongen/tmp/Matcher/emerald_id_known_site_smoke_20260924
attempt="${SMOKE_ATTEMPT:-1}"
case_root="$smoke_root/$pdb_id/occurrence_$occurrence_id/$task/attempt_$attempt/input"
run_root="$smoke_root/$pdb_id/occurrence_$occurrence_id/$task/attempt_$attempt/run"
code_root="$smoke_root/code"
data_root=/storage/penghongen/AdaLigand/Ori_Data

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$case_root" "$run_root"
/home/penghongen/anaconda3/envs/pxm_phase1/bin/python \
    "$code_root/prepare_project_smoke.py" \
    "$data_root" "$pdb_id" "$occurrence_id" "$task" "$case_root"
bash "$code_root/smoke_project.sh" "$case_root" "$run_root" "$smoke_root/tutorial_source"
