""" Minimal example for calculing rdf from existing data """

import os

import gamdpy as gp

file_to_read = "LJ_T0.70.h5" 

if not os.path.isfile(file_to_read):
    print("This example needs a the file LJ_T0.70.h5 to be present")
    print(f"{file_to_read} can be generated using minimal.py example")
    exit()

# Load existing data
output = gp.tools.TrajectoryIO(file_to_read).get_h5()
# Read number of particles N and dimensions from data
nblocks, nconfs, N, D = output['block/positions'].shape

# Create configuration object
configuration = gp.Configuration(D=D, N=N)
configuration.simbox = gp.Orthorhombic(D, output.attrs['simbox_initial'])
configuration.copy_to_device()
# Call the rdf calculator
calc_rdf = gp.CalculatorRadialDistribution(configuration, bins=1000)

# NOTE: the structure of the block is (outer_block, inner_steps, pos&img, npart, dimensions)
#       the zero is to select the position array and discard images
positions = output['block/positions'][:,:,:,:]
positions = positions.reshape(nblocks*nconfs,N,D)
# Loop over saved configurations
for pos in positions[nconfs-1::nconfs]:
    configuration['r'] = pos
    configuration.copy_to_device()
    calc_rdf.update()

# Save rdf
calc_rdf.save_average()

