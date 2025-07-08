""" Using gradient descent followed by conjugate gradient to 'quench' trajectory in a h5 file

The model is binary Kob & Andersen with shifted force cut-off, as found eg. in Data/KABLJ_Rho1.200_T0.800_toread.h5

Usage:
    python3 quench_trajectory.py filename
"""

import gamdpy as gp
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
from scipy.optimize import minimize

Tconf_switch = 1e-4 # Do gradient descent until Tconf_switch is reached
include_cg = True   # ... and then do conjugate gradient if this flag is True
steps_between_output=32 # For gd integrator
num_timeblocks = 1     # Number of timeblocks to quench, 0 to do all 
num_configurations = 6 # Numner of configurations in each timeblock, 0 to do all
gp.select_gpu()

argv = sys.argv.copy()
argv.pop(0)  # remove scriptname
if __name__ == "__main__":
    if argv:
        filename = argv.pop(0) # get filename
    else:
        filename = 'Data/KABLJ_Rho1.200_T0.800_toread.h5' # Used in testing
else:
    filename = 'Data/KABLJ_Rho1.200_T0.800_toread.h5' # Used in testing

# function to interface with minimize function from scipy
def calc_u(Rflat):
        configuration2['r'] = Rflat.reshape(N,D).astype('float32')
        evaluator.evaluate(configuration2)
        return np.sum(configuration2['U'].astype('float64'))
def calc_du(Rflat):
        configuration2['r'] = Rflat.reshape(N,D).astype('float32')
        evaluator.evaluate(configuration2)
        return -configuration2['f'].astype('float64').flatten()

# Load existing configuration, twice for convenience 
with h5py.File(filename, 'r') as f:
    configuration1 = gp.Configuration.from_h5(f, "restarts/restart0000", compute_flags={'lapU':True, 'Fsq':True})
    configuration2 = gp.Configuration.from_h5(f, "restarts/restart0000", compute_flags={'lapU':True, 'Fsq':True})
print(configuration1)
N = configuration1.N
D = configuration1.D

# Setup pair potential: Kob & Andersen Binary Lennard-Jones Mixture
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig = [[1.00, 0.80],
       [0.80, 0.88]]
eps = [[1.00, 1.50],
       [1.50, 0.50]]
cut = np.array(sig)*2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

evaluator = gp.Evaluator(configuration2, pair_pot)
evaluator.evaluate(configuration2)
print(configuration2)

# Check that we are using the same model
assert(np.allclose(configuration1['U'], configuration2['U']))
assert(np.allclose(configuration1['W'], configuration2['W'], atol=0.00001)), f"({configuration1['W']} {configuration2['W']}"
assert(np.allclose(configuration1['f'], configuration2['f'], atol=0.0001)), f"({configuration1['f']} {configuration2['f']}"

# Setup integrator: NVT
integrator = gp.integrators.GradientDescent(dt=0.00001) # v = f*dt

# Setup runtime actions, i.e., actions performed during simulation of timeblocks
runtime_actions = [gp.ScalarSaver(steps_between_output=32, compute_flags={'lapU':True, 'Fsq':True}),]

# Setup Simulation. 
sim = gp.Simulation(configuration1, [pair_pot], integrator, runtime_actions,
                    num_timeblocks=32, steps_per_timeblock=2*1024,
                    storage='memory')

output = gp.tools.TrajectoryIO(filename).get_h5()
nblocks, nconfs, N, D = output['trajectory_saver/positions'].shape

if num_timeblocks==0:
    num_timeblocks=nblocks

if num_configurations==0:
    num_configurations=nconfs

for saved_timeblock in range(num_timeblocks):
    #ptype = trajectory['initial_configuration/ptype'][:].copy()
    #attributes = trajectory.attrs
    #simbox = trajectory['initial_configuration'].attrs['simbox_data'].copy()
    #num_types = np.max(ptype) + 1
    #if isinstance(qvalues, float):
    #    qvalues = np.ones(num_types)*qvalues
    #num_blocks, conf_per_block, N, D = trajectory['trajectory_saver/positions'].shape
    #blocks = trajectory['trajectory_saver/positions']  # If picking out dataset in inner loop: Very slow!
    #images = trajectory['trajectory_saver/images']

    for saved_conf in range(num_configurations):
        configuration1['r'] = output['trajectory_saver/positions'][saved_timeblock,saved_conf,:,:]
        configuration2['r'] = output['trajectory_saver/positions'][saved_timeblock,saved_conf,:,:]

        # Run simulation
        for timeblock in sim.run_timeblocks():
            Fsq_ = np.sum(configuration1['f'].astype(np.float64)**2)/configuration1.N
            lapU_ = np.sum(configuration1['lapU'].astype(np.float64))/configuration1.N
            Tconf_ = Fsq_ / lapU_
            if Tconf_ < Tconf_switch:
                break
            
        R0flat = configuration1['r'].astype('float64').flatten()
        res = minimize(calc_u, R0flat, method='CG', jac=calc_du, options={'gtol': 1e-9, 'disp': False, 'return_all':False})
        configuration2['r'] = res.x.reshape(N,D).astype('float32')
        evaluator.evaluate(configuration2)
        
        u_min = calc_u(res.x)/N
        Fsq_min = np.sum(calc_du(res.x)**2)/N
        Tconf_min =  Fsq_min / np.sum(configuration2['lapU'].astype(np.float64)/N)

        print(f'{saved_timeblock=}, {saved_conf=}, {u_min=}, {Fsq_min=}, {Tconf_min=}') 