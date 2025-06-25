""" Using gradient descent to 'quench' configurations saved as restarts in a trajectory h5 file

For now the model is single component LJ, as found eg. in Data/LJ_r0.973_T0.70_toread.h5
- will be changed to Kob&Andersen

Usage:
    python3 quench_restarts_gradient_descent.py filename
"""

import gamdpy as gp
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py

gp.select_gpu()

argv = sys.argv.copy()
argv.pop(0)  # remove scriptname
if __name__ == "__main__":
    if argv:
        filename = argv.pop(0) # get filename (.h5 added by script)
    else:
        filename = 'Data/LJ_r0.973_T0.70_toread.h5' # Used in testing
else:
    filename = 'Data/LJ_r0.973_T0.70_toread.h5' # Used in testing

# Load existing configuration, twice for convenience 
with h5py.File(filename, 'r') as f:
    configuration1 = gp.Configuration.from_h5(f, "restarts/restart0000", compute_flags={'lapU':True, 'Fsq':True})
    configuration2 = gp.Configuration.from_h5(f, "restarts/restart0000", compute_flags={'lapU':True, 'Fsq':True})
print(configuration1)
N = configuration1.N
D = configuration1.D

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
#pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
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
runtime_actions = [gp.ScalarSaver(compute_flags={'lapU':True, 'Fsq':True}),]

# Setup Simulation. 
sim = gp.Simulation(configuration1, [pair_pot], integrator, runtime_actions,
                    num_timeblocks=32, steps_per_timeblock=2*1024,
                    storage='memory')

fig = plt.figure(figsize=(8, 14))
axs = fig.subplot_mosaic([["u", "u"],
                          ["lu", "lu"],
                          ["du", "du"],
                          ["Tc", "Tc"],
                          ], sharex=True)
fig.subplots_adjust(hspace=0.00)
axs['u'].set_ylabel('U/N')
axs['u'].grid(linestyle='--', alpha=0.5)
axs['lu'].set_ylabel('U/N - Umin/N')
axs['lu'].grid(linestyle='--', alpha=0.5)
axs['du'].set_ylabel('F**2/N')
axs['du'].grid(linestyle='--', alpha=0.5)
axs['Tc'].set_ylabel('Tconf = F**2 / lapU')
axs['Tc'].grid(linestyle='--', alpha=0.5)
axs['Tc'].set_xlabel('Iteration')

for restart in range(3):
    with h5py.File(filename, 'r') as f:
        configuration2 = gp.Configuration.from_h5(f, f"restarts/restart{restart:04d}", compute_flags={'lapU':True})

    configuration1['r'] = configuration2['r']

    # Run simulation
    for timeblock in sim.run_timeblocks():
        print(sim.status(per_particle=True))
    print(sim.summary())
   
    U, Fsq, lapU = gp.ScalarSaver.extract(sim.output, columns=['U', 'Fsq', 'lapU'], first_block=0)
    iteration = np.arange(len(U))*sim.output['scalar_saver'].attrs['steps_between_output']
    Tconf = Fsq / lapU

    axs['u'].plot(iteration, U, '-', label=f'Umin/N = {U[-1]:.10f}')
    axs['lu'].semilogy(iteration, U-U[-1], '-')
    axs['du'].semilogy(iteration, Fsq, '-')
    axs['Tc'].semilogy(iteration, Tconf, '-', label=f'{Tconf[0]:.2e} -> {Tconf[-1]:.3e}')

axs['u'].legend()
axs['Tc'].legend()

if __name__ == "__main__":
    plt.show(block=True)