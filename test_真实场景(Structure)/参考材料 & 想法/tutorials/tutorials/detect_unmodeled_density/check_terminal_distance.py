import sys
import math

coords = [x for x in sys.argv[2].split(',') if x.strip()]
blob_x = float(coords[0])
blob_y = float(coords[1])
blob_z = float(coords[2])

term_atoms = []
prev_res = -1
prev_line = ""

with open(sys.argv[1], 'r') as pdb:
    for line in pdb:
        if line.startswith("ATOM"):
            res = int(line[22:26].strip())
            if res != prev_res:
                if res != prev_res + 1:
                    term_atoms.append(line[21:26])
                    if prev_res != -1:
                        term_atoms.append(prev_line[21:26])
            prev_res = res
            prev_line = line
        if line.startswith("TER"):
            if line[21:26] not in term_atoms:
                term_atoms.append(line[21:26])

with open(sys.argv[1], 'r') as pdb:
    for line in pdb:
        if line.startswith("ATOM"):
            if line[21:26] in term_atoms:
                if line[77] != 'H':
                    #atom_x += float(line[30:38])
                    #atom_y += float(line[38:46])
                    #atom_z += float(line[46:54])
                    distance = math.sqrt((blob_x-float(line[30:38]))**2 + (blob_y-float(line[38:46]))**2 + (blob_z-float(line[46:54]))**2)
                    if distance < 5.0:
                        print("Blob is near terminus")

#for atom in term_atoms:
#    print(atom)
