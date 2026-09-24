#!/usr/bin/env python

import os
import pandas
import numpy
import argparse
import sys

def softmax(zscores, lig_dens_ccs):
    max_z = numpy.max( zscores )
    array_scaled = (( numpy.exp( 0.1*max_z ) ) + ( numpy.exp( 0.3*( lig_dens_ccs-0.6 ) ) )) * zscores
    array_e = numpy.exp(array_scaled)
    return array_e / array_e.sum()

def main(args):
    
    if not os.path.isdir(args.directory):
        sys.exit("Directory does not exist. Exiting.")

    if ( len( [filename for filename in os.listdir(args.directory) if filename.endswith(".sc")] ) == 0):
        sys.exit("No Rosetta scorefiles exist in the directory. Exiting.")

    #loop through directory that contains scorefiles and add to dataframe
    data = []
    cols = []
    for scorefile in os.listdir(args.directory):
        if scorefile.endswith(".sc"):
            with open(args.directory+'/'+scorefile, 'r') as scores:
                nline = 0
                for row in scores:
                    nline += 1
                    if nline == 1:
                        continue

                    if nline == 2:
                        if "zscore" not in row:
                            print("File %s does not contain Z-scores for ligand identification. Skipping."%scorefile)
                            break
                        if len( cols ) == 0:
                            cols = row.split()
                        else:
                            if row.split() != cols:
                                print("File columns in %s do not match first entry. Skipping."%scorefile)
                                break

                    if nline > 2:
                        data.append(row.split())

    df = pandas.DataFrame(data, columns=cols)
    df["zscore"] = df["zscore"].astype(float)
    df["lig_dens"] = df["lig_dens"].astype(float)

    #modify Z-scores and apply softmax
    probability = softmax( df["zscore"], df["lig_dens"] )
    df["probability"] = probability

    #sort dataframe and output lig_ID, dG, dG_Z, dens, dens_Z, Z-score, probability
    df_sorted = df.sort_values( by = ["probability"], ascending=False )
    with open(args.output, 'w') as output:
        df_string = df_sorted.to_string(columns = ["ligandname","dG","lig_dens","dG_Z","dens_Z","zscore","probability"], index=False)
        for line in df_string:
            output.write(line)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing Rosetta scoresfiles to parse')
    parser.add_argument('-o', '--output', type=str, default='id_output.txt', help='Name of output file. Default: id_output.txt')

    args = parser.parse_args()

    main(args)
