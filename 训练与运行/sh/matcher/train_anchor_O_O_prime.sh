#!/usr/bin/env bash
set -eo pipefail

# 当前入口只运行不依赖 Stage1 推理产物的 Anchor/A-only O/O′ 路线。
# 它执行 Phase1、精确 BEST 接力的 Phase2，以及使用冻结阈值的完整 validation 评估。
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="${TASK_PROJECT_ROOT:-$(cd "${script_dir}/../../.." && pwd -P)}"
phase1_config="${project_root}/configs/matcher/anchor_O_O_prime_phase1.yaml"
phase2_config="${project_root}/configs/matcher/anchor_O_O_prime_phase2.yaml"
output_root=/storage/penghongen/AdaLigand/Results/matcher/anchor_O_O_prime_v1/seed_3407_occ48
phase1_output="${output_root}/phase1"
phase2_output="${output_root}/phase2"
evaluation_output="${output_root}/validation_O_O_prime"

if [[ -e "${output_root}" ]]; then
  echo "[matcher][错误] 正式输出目录已存在：${output_root}" >&2
  echo "[matcher][错误] 本入口只启动全新实验；恢复任务必须显式指定 checkpoint。" >&2
  exit 2
fi

source /home/penghongen/anaconda3/etc/profile.d/conda.sh
conda activate Pocket_Plus_centos7_cu121_allgpu
cd "${project_root}"
export PYTHONPATH="${project_root}"
# 正式训练采用与 A800 显存画像相同的 CUDA 分配器设置。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m matcher.train --config "${phase1_config}"
phase1_checkpoint="$(
  python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["checkpoint"])' \
    "${phase1_output}/BEST.json"
)"
echo "[matcher] Phase2 接入 Phase1 BEST：${phase1_checkpoint}"

python -m matcher.train \
  --config "${phase2_config}" \
  "run.phase1_checkpoint=${phase1_checkpoint}"
phase2_checkpoint="$(
  python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["checkpoint"])' \
    "${phase2_output}/BEST.json"
)"
echo "[matcher] validation 评估使用 Phase2 BEST：${phase2_checkpoint}"

python -m matcher.infer_anchor \
  --config "${phase2_config}" \
  --checkpoint "${phase2_checkpoint}" \
  --output-dir "${evaluation_output}" \
  --split validation \
  --evaluate

echo "[matcher] Anchor O/O′ 正式流程完成：${output_root}"
