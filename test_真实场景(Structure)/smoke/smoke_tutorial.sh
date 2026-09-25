#!/usr/bin/env bash
set -euo pipefail

# 只运行作者提供的 7bxu 教程, 不承载 NxU/nxu 正式评估.
tutorial_root="$(realpath "$1")"
output_root="$(realpath -m "$2")"
rosetta_root=/home/penghongen/software/rosetta/source_build/rosetta.source.release-430/main
rosetta_executable=/home/penghongen/software/rosetta/bin/rosetta_scripts_430

mkdir -p "$output_root/outputs"
cd "$output_root"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$rosetta_executable" \
    -in:file:extra_res_fa \
    "$tutorial_root/library/ADP.params" \
    "$tutorial_root/library/ATP.params" \
    "$tutorial_root/library/CLR.params" \
    "$tutorial_root/library/Y01.params" \
    -in:file:override_database_params \
    -out:path:all "$output_root/outputs" \
    -out:prefix emerald-id_ \
    -gen_potential \
    -database "$rosetta_root/database" \
    -s "$tutorial_root/inputs/7bxu.pdb" \
    -multi_cool_annealer 10 \
    -parser:protocol "$tutorial_root/ligand_screen.xml" \
    -no_autogen_cart_improper \
    -atom_mask 2 \
    -parser:script_vars "map=$tutorial_root/inputs/emd_30238.map" \
    -parser:script_vars ligands=ADP,ATP,CLR,Y01 \
    -parser:script_vars local_res=3.97 \
    -sliding_window 1 \
    -edensity::score_sliding_window_context \
    -edensity::mapreso 3.70 \
    -edensity::grid_spacing 1.0
