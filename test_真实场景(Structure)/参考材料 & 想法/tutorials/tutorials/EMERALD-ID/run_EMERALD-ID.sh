#!/bin/bash

d=$1
if [ $# -eq 0 ]; then
	echo "No params directory or file provided"
	echo "Proper use: ./run_EMERALD-ID.sh library/"
	exit 1
fi

ligand_list=""
params_list=""
#get all ligands in library
for params in $(find $d -maxdepth 1 -type f -name "*.params"); do 
	ligand_name=$(head -n1 $params | cut -d' ' -f2)
	ligand_list="${ligand_list}${ligand_name},"
	params_list="${params_list}${params} "
done
ligand_list=${ligand_list::-1}
params_list=${params_list::-1}

prefix=""
if [ $(awk -F, '{print NF-1}' <<< "$ligand_list") == 0 ]; then
	prefix="emerald-id_${ligand_list}_"
else
	prefix="emerald-id_"
fi

#runs ligand dock
$ROSETTA/source/bin/rosetta_scripts.linuxgccrelease \
    -in:file:extra_res_fa $params_list \
    -in:file:override_database_params \
    -out:path:all outputs \
    -out:prefix $prefix \
    -gen_potential \
    -database ~/git/main/database \
    -s inputs/7bxu.pdb \
    -multi_cool_annealer 10 \
    -parser:protocol ligand_screen.xml \
    -gen_potential \
    -no_autogen_cart_improper \
    -atom_mask 2 \
    -parser:script_vars map=inputs/emd_30238.map \
    -parser:script_vars ligands=$ligand_list \
    -parser:script_vars local_res=3.97 \
    -sliding_window 1 \
    -edensity::score_sliding_window_context \
    -edensity::mapreso 3.70 \
    -edensity::grid_spacing 1.0
