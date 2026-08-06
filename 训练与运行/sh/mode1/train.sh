#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="${TASK_PROJECT_ROOT:-$(cd "${script_dir}/../../.." && pwd -P)}"
config_root="${project_root}/configs/mode1"

set +u
source /home/penghongen/anaconda3/etc/profile.d/conda.sh
conda activate Pocket_Plus_centos7_cu121_allgpu
set -u
cd "${project_root}"
export PYTHONPATH="${project_root}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 结果根只由 Phase1 YAML 的 output_dir 决定，避免 shell 与配置写入不同实验目录。
output_root="$(python -c 'from pathlib import Path; from omegaconf import OmegaConf; import sys; print(Path(OmegaConf.load(sys.argv[1]).run.output_dir).parent)' "${config_root}/phase1.yaml")"

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
  python -m matcher.train "${arguments[@]}"
}

run_phase phase1
run_phase phase2

run_inference() {
  local split="$1"
  local output="${output_root}/${split}"
  if [[ -f "${output}/metrics.json" ]]; then
    return
  fi
  python -m matcher.infer_ground_truth \
    --config "${config_root}/phase2.yaml" \
    --checkpoint "${output_root}/phase2/best.pt" \
    --split "${split}" \
    --output-dir "${output}"
}

run_inference validation
run_inference calibration
