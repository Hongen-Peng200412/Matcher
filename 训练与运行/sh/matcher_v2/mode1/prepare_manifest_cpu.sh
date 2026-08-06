#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="${TASK_PROJECT_ROOT:-$(cd "${script_dir}/../../../.." && pwd -P)}"
data_root=/storage/penghongen/AdaLigand/Ori_Data
output=${data_root}/matcher_v2/ground_truth_context/manifest.json

set +u
source /home/penghongen/anaconda3/etc/profile.d/conda.sh
conda activate Pocket_Plus_centos7_cu121_allgpu
set -u
cd "${project_root}"
export PYTHONPATH="${project_root}"

python -m matcher_v2.manifest ground-truth \
  --data-root "${data_root}" \
  --box-manifest "${data_root}/stage1_preparation_box_pool_2/box_pool/manifest.json" \
  --calibration /storage/penghongen/AdaLigand_stage1_inference/calibration_pdb_ids.json \
  --output "${output}"
