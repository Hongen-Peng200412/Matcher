#!/bin/bash
cat $1 | tr ' ' ',' | while read line; do 
	buried=$(echo $line | cut -c 56-60)
	volume=$(echo $line | cut -c 61-66)
	volume=${volume//,}
	if (( $(echo "$buried > 0.90" | bc -l) )) && (( $(echo "$volume > 70.0" | bc -l) )); then
		coords=$(echo $line | cut -c 31-54)
	       	if ! python check_terminal_distance.py inputs/7bxu.pdb $coords | grep -q "Blob"; then
			tr ',' ' ' <<< $line
	       	fi
	fi
done
