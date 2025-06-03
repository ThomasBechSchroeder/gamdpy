import gamdpy as gp
import h5py
import numpy as np
import matplotlib.pyplot as plt

filename = '../Data/LJ_r0.973_T0.70_toread.h5'

h5file = h5py.File(filename, 'r')

# Print available data columns
print('Available data columns:', gp.ScalarSaver.columns(h5file))

print("More detailed information:")
print(gp.ScalarSaver.info(h5file))
                             
# Data is extracted as a list of numpy arrays, which we here unpack to individual arrays, 
# while ignoring data from first timeblock (0) of the simulation
U, W = gp.ScalarSaver.extract(h5file, columns=['U', 'W'], first_block=1)
print(f'{np.mean(U)=}, {np.mean(W)=}')

# Taking eg the mean directly, not storing full numpy-arrays:
mU, mW = gp.ScalarSaver.extract(h5file, columns=['U', 'W'], first_block=1, function=np.mean)
print(f'{mU=}, {mW=}')

# By default data is given divided by the number of particles. This can be changed:
mU, mW = gp.ScalarSaver.extract(h5file, columns=['U', 'W'], per_particle=False, 
                                first_block=1, 
                                function=np.mean)
print(f'{mU=}, {mW=}')

# Get the times associated with the scalar data (eg. for plotting). 
# Use subsamble > 1 (integer) to reduce the amount of data
times = gp.ScalarSaver.get_times(h5file, first_block=0, subsample=2)
U, = gp.ScalarSaver.extract(h5file, columns=['U'], first_block=0, subsample=2)
print(f'{U.shape=}, {times.shape=}')

plt.plot(times, U, '.-')
plt.show(block=False)

h5file.close()


# Works also on-the-fly:
sim = gp.get_default_sim()
for block in sim.run_timeblocks():
    mU, mW = gp.ScalarSaver.extract(sim.output, columns=['U','W'], 
                                    first_block=block, last_block=block+1, 
                                    function=np.mean)
    print(f'{block=:5}, {mU=:.8}, {mW=:.8}')

# ... and after-the-fly
mU, mW = gp.ScalarSaver.extract(sim.output, columns=['U', 'W'], first_block=1, function=np.mean)
print(f'{mU=}, {mW=}')
