"""This script creates a new python script in which a diction containing the
parameters for the ZJW-2004 potential are contained. These are the same
parameters that were published in the 2004 paper. They are read in from the
text file  'ZJW-2004-parameters.txt'

"""
import gamdpy as gp
import pandas as pd
import numpy as np
import math
import h5py
import os

# to allow this script to be run by testing need to specify that the location
# of theinput file is the same as that of the script itself

cwd = os.getcwd()

script_dir = os.path.dirname(os.path.abspath(__file__))

filename = os.path.join(script_dir, 'ZJW-2004-parameters.txt')

with open(filename, 'r') as f:
    header_line = f.readline().strip()

column_names = header_line.lstrip('#').split()

df = pd.read_csv(filename, sep=r'\s+', skiprows=1, names=column_names)


EAM_ZJW_2004_params = {}

for element in column_names:

    EAM_ZJW_2004_params[element] = df[element].to_numpy()


EAM_ZJW_2004_params['comment'] = {'Embedded-atom method (EAM) potential parameters for the Zhou-Johnson-Wadley formulation of the EAM. Parameters for 16 elements as provided in the 2004 paper are given These are: Cu, Ag, Au, Ni, Pd, Pt, Al, Pb, Fe, Mo, Ta, W, Mg, Co, Ti, and Zr. Units are eV and Å.'}

EAM_ZJW_2004_params['reference'] = {
            'title': 'Misfit-energy-increasing dislocations in vapor-deposited CoFe/NiFe multilayers',
            'volume': 69,
            'DOI': '10.1103/PhysRevB.69.144113',
            'url': 'http://dx.doi.org/10.1103/PhysRevB.69.144113',
            'number': 14,
            'journal': 'Physical Review B',
            'shortjournal': 'Phys. Rev. B',
            'author': ['X. W. Zhou', 'R. A. Johnson', 'H. N. G. Wadley'],
            'year': 2004,
            'month': 'apr'
            }




outfilename = "EAM_ZJW_2004_params.py.tmp"

if script_dir == cwd:

    with open(outfilename, "w") as f:
        f.write("from numpy import array\n")
        print("EAM_ZJW_2004_params = ", EAM_ZJW_2004_params, file=f)
        
    print("Wrote parameters as a dictionary to", outfilename)
    print("Rename to EAM_ZJW_2004_params.py if necessary to replace")
