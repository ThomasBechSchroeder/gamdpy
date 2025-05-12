""" Minimal example for calculing rdf from existing data 

    Usage:

    analyze_structure filename
"""

import gamdpy as gp
import numpy as np
import numba
import matplotlib.pyplot as plt
import sys
import pickle

gp.select_gpu()

argv = sys.argv.copy()
argv.pop(0)  # remove scriptname
if __name__ == "__main__":
    if argv:
        filename = argv.pop(0) # get filename (.h5 added by script)
    else:
        filename = 'Data/LJ_r0.973_T0.70_toread' # Used in testing
else:
    filename = 'Data/LJ_r0.973_T0.70_toread' # Used in testing

# Load existing data
output = gp.tools.TrajectoryIO(filename+'.h5').get_h5()
# Read number of particles N and dimensions from data
nblocks, nconfs, N, D = output['block/positions'].shape

# Create configuration object
configuration = gp.Configuration(D=D, N=N)
configuration.simbox = gp.Orthorhombic(D, output.attrs['simbox_initial'])
configuration.ptype = output['ptype']
configuration.copy_to_device()
# Call the rdf calculator
calc_rdf = gp.CalculatorRadialDistribution(configuration, bins=300)

# NOTE: the structure of the block is (outer_block, inner_steps, pos&img, npart, dimensions)
#       the zero is to select the position array and discard images
positions = output['block/positions'][:,:,:,:]
positions = positions.reshape(nblocks*nconfs,N,D)
# Loop over saved configurations
for pos in positions[nconfs-1::nconfs]:
    configuration['r'] = pos
    configuration.copy_to_device()
    calc_rdf.update()

rdf_data = calc_rdf.read()
with open(filename+'_rdf.pkl', 'wb') as f:     
    pickle.dump(rdf_data, f)
print(f"Wrote: {filename+'_rdf.pkl'}")

num_types = rdf_data['rdf_ptype'].shape[1]
plt.figure(figsize=(8, 4))
for i in range(num_types):
    for j in range(i, num_types):
        rdf_ij = np.mean(rdf_data['rdf_ptype'][:,i,j,:], axis=0)
        plt.plot(rdf_data['distances'], rdf_ij, label=f'{i}-{j}')
if num_types > 1:
    plt.legend()
plt.title(filename)
plt.xlabel('Distance')
plt.ylabel('Radial Distribution Function')
plt.savefig(filename+'_rdf.pdf')
print(f"Wrote: {filename+'_rdf.pdf'}")
plt.show()

