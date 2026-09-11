""" Using gradient descent followed by conjugate gradient to 'quench' configurations saved as restarts in a trajectory h5 file

The model is binary Kob & Andersen with shifted force cut-off, as found eg. in Data/KABLJ_Rho1.200_T0.800_toread.h5

Usage:
    python3 quench_restarts.py filename
"""

import gamdpy as gp
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import pickle

num_restarts = 32 # Number of restarts to quench 

gp.select_gpu()

argv = sys.argv.copy()
argv.pop(0)  # remove scriptname
if __name__ == "__main__":
    if argv:
        filename = argv.pop(0) # get filename
    else:
        filename = 'Data/KABLJ_Rho1.200_T0.400_toread' # Used in testing
else:
    filename = 'Data/KABLJ_Rho1.200_T0.400_toread' # Used in testing


# Load existing configuration, twice for convinience
with h5py.File(filename+'.h5', 'r') as f:
    configuration1 = gp.Configuration.from_h5(f, "restarts/restart0000", )
    configuration2 = gp.Configuration.from_h5(f, "restarts/restart0000", )
   
N = configuration1.N
D = configuration1.D

configuration1.copy_to_device()
calc_rdf = gp.CalculatorRadialDistribution(configuration1, bins=200)

for restart in range(num_restarts):
    with h5py.File(filename+'.h5', 'r') as f:
        configuration2 = gp.Configuration.from_h5(f, f"restarts/restart{restart:04d}",)

    configuration1['r'] = configuration2['r']
    #print(np.array(configuration2.ptype)[:35])
    configuration1.ptype = configuration2.ptype
    configuration1.copy_to_device()
    calc_rdf.update()

rdf_data = calc_rdf.read()

with open(filename+'_rdf.pkl', 'wb') as f:     
    pickle.dump(rdf_data, f)
print(f"Wrote: {filename+'_rdf.pkl'}")

num_types = rdf_data['rdf'].shape[1]
plt.figure(figsize=(8, 4))
for i in range(num_types):
    for j in range(i, num_types):
        plt.plot(rdf_data['distances'], rdf_data['rdf'][:,i,j], label=f'{i}-{j}')
if num_types > 1:
    plt.legend()
plt.title(filename)
plt.xlabel('Distance')
plt.ylabel('Radial Distribution Function')
plt.grid(linestyle='--', alpha=0.5)
plt.savefig(filename+'_rdf.pdf')
print(f"Wrote: {filename+'_rdf.pdf'}")

if __name__ == "__main__":
    plt.show()
