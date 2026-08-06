#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="${TASK_PROJECT_ROOT:-$(cd "${script_dir}/../../../.." && pwd -P)}"
config_root="${project_root}/configs/matcher_v2/mode1"
output_root=/storage/penghongen/AdaLigand/Results/matcher_v2/ground_truth_context/seed_3407_occ48

set +u
source /home/penghongen/anaconda3/etc/profile.d/conda.sh
conda activate Pocket_Plus_centos7_cu121_allgpu
set -u
cd "${project_root}"
export PYTHONPATH="${project_root}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_phase() {
  local phase="$1"
  local completed="${output_root}/${phase}/COMPLETED.json"
  local latest="${output_root}/${phase}/latest.pt"
  if [[ -f "${completed}" ]]; then
    return
  fi
  local arguments=(--config "${config_root}/${phase}.yaml")
  if [[ -f "${latest}" ]]; then
    arguments+=(--resume "${latest}")
  fi
  python -m matcher_v2.train "${arguments[@]}"
}

run_phase phase1
run_phase phase2

run_inference() {
  local split="$1"
  local output="${output_root}/${split}"
  if [[ -f "${output}/metrics.json" ]]; then
    return
  fi
  python -m matcher_v2.infer_ground_truth \
    --config "${config_root}/phase2.yaml" \
    --checkpoint "${output_root}/phase2/best.pt" \
    --split "${split}" \
    --output-dir "${output}"
}

run_inference validation
run_inference calibration
