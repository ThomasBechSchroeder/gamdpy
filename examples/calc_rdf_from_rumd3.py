""" Minimal example for calculing rdf from existing data """

import os

import gamdpy as gp

file_to_read = "Data/NVT_N4000_T2.0_rho1.2_KABLJ_rumd3/TrajectoryFiles"

if not os.path.isdir(file_to_read):
    print(f"This example needs {file_to_read} to be present")
    exit()

# Load existing data
output = gp.tools.TrajectoryIO(file_to_read).get_h5()
# Read number of particles N and dimensions from data
nblocks, nconfs, N, D = output['trajectory_saver/positions'].shape
# Set up the configuration object
configuration = gp.Configuration(D=D, N=N)
configuration.simbox = gp.Orthorhombic(D, output['initial_configuration'].attrs['simbox_data'])
configuration.ptype = output['initial_configuration/ptype']
configuration.copy_to_device()
# Call the rdf calculator
calc_rdf = gp.CalculatorRadialDistribution(configuration, bins=1000, ptype=output['initial_configuration/ptype'])

# NOTE: the structure of the block is (outer_block, inner_steps, pos&img, npart, dimensions)
#       the zero is to select the position array and discard images
positions = output['trajectory_saver/positions'][:,:,:,:]
positions = positions.reshape(nblocks*nconfs,N,D)
# Loop over saved configurations
for pos in positions[nconfs-1::int(nconfs/8)]:
    configuration['r'] = pos
    configuration.copy_to_device()
    calc_rdf.update()

# Save rdf
calc_rdf.save_average("rdf_rumd3.dat")

