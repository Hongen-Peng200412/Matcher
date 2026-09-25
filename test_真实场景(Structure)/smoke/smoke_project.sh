#!/usr/bin/env bash
set -euo pipefail

# 当前脚本只在一个给定真实点上调用教程同款 EMERALD-ID, 不汇总正式准确率.
case_root="$(realpath "$1")"
output_root="$(realpath -m "$2")"
tutorial_root="$(realpath "$3")"
rosetta_root=/home/penghongen/software/rosetta/source_build/rosetta.source.release-430/main
rosetta_executable=/home/penghongen/software/rosetta/bin/rosetta_scripts_430

mkdir -p "$output_root/outputs"
cd "$output_root"
mapfile -t params < <(find "$case_root/ligands" -maxdepth 1 -name '*.params' -type f | sort)
ligands="$(printf '%s,' "${params[@]##*/}")"
ligands="${ligands%,}"
ligands="${ligands//.params/}"
resolution="$(/home/penghongen/anaconda3/envs/pxm_phase1/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["resolution_angstrom"])' "$case_root/identity_map.json")"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$rosetta_executable" \
    -in:file:extra_res_fa "${params[@]}" \
    -in:file:override_database_params \
    -out:path:all "$output_root/outputs" \
    -out:prefix emerald-id_ \
    -gen_potential \
    -database "$rosetta_root/database" \
    -s "$case_root/input.pdb" \
    -multi_cool_annealer 10 \
    -parser:protocol "$tutorial_root/ligand_screen.xml" \
    -no_autogen_cart_improper \
    -atom_mask 2 \
    -parser:script_vars "map=$case_root/map.map" \
    -parser:script_vars "ligands=$ligands" \
    -parser:script_vars "local_res=$resolution" \
    -sliding_window 1 \
    -edensity::score_sliding_window_context \
    -edensity::mapreso "$resolution" \
    -edensity::grid_spacing 1.0
