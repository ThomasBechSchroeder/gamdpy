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
from scipy.optimize import minimize

Tconf_switch = 1e-4 # Do gradient descent until Tconf_switch is reached
include_cg = True   # ... and then do conjugate gradient if this flaf is True
steps_between_output=32 # For gd integrator

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

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
#pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup pair potential: Kob & Andersen Binary Lennard-Jones Mixture
#pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
#sig = [[1.00, 0.80],
#       [0.80, 0.88]]
#eps = [[1.00, 1.50],
#       [1.50, 0.50]]
#cut = np.array(sig)*2.5
#pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)


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
axs['Tc'].set_xlabel(f'Iteration (1 iteration = {steps_between_output} gradient descent steps)')

for restart in range(5):
    with h5py.File(filename, 'r') as f:
        configuration2 = gp.Configuration.from_h5(f, f"restarts/restart{restart:04d}", compute_flags={'lapU':True})

    configuration1['r'] = configuration2['r']

    # Run simulation
    print(f'{restart=}')
    for timeblock in sim.run_timeblocks():
        Fsq_ = np.sum(configuration1['f'].astype(np.float64)**2)/configuration1.N
        lapU_ = np.sum(configuration1['lapU'].astype(np.float64))/configuration1.N
        Tconf_ = Fsq_ / lapU_
        print(sim.status(per_particle=True), f'{Tconf_=:.3e}')
        if Tconf_ < Tconf_switch:
            break
    #print(sim.summary()) # Does not work when using 'break' ...
   
    U_gd, Fsq_gd, lapU_gd = gp.ScalarSaver.extract(sim.output, columns=['U', 'Fsq', 'lapU'], first_block=0, last_block=timeblock+1)
    iteration_gd = np.arange(len(U_gd))
    Tconf_gd = Fsq_gd / lapU_gd

    u_min = U_gd[-1]
    Tconf_min = Tconf_gd[-1]

    if include_cg:
        R0flat = configuration1['r'].astype('float64').flatten()
        res = minimize(calc_u, R0flat, method='CG', jac=calc_du, options={'gtol': 1e-9, 'disp': True, 'return_all':True})
        configuration2['r'] = res.x.reshape(N,D).astype('float32')
        evaluator.evaluate(configuration2)

        U_cg = []
        Fsq_cg = []
        Tconf_cg = []

        for x in res.allvecs:
            U_cg.append(calc_u(x)/N)
            Fsq_cg.append(np.sum(calc_du(x)**2)/N)
            Tconf_cg.append( Fsq_cg[-1] / np.sum(configuration2['lapU']/N) )
        
        u_min = U_cg[-1]
        Tconf_min = Tconf_cg[-1]

    axs['u'].plot(iteration_gd, U_gd, '-', label=f'Umin/N = {u_min:.10f}')
    axs['lu'].semilogy(iteration_gd, U_gd-u_min, '-')
    axs['du'].semilogy(iteration_gd, Fsq_gd, '-')
    axs['Tc'].semilogy(iteration_gd, Tconf_gd, '-', label=f'{Tconf_gd[0]:.2e} -> {Tconf_min:.3e}')

    if include_cg:
        iteration_cg = np.arange(len(U_cg)) + iteration_gd[-1]
        axs['u'].plot(iteration_cg, U_cg, '.-')
        axs['lu'].semilogy(iteration_cg, U_cg-u_min, '.-')
        axs['du'].semilogy(iteration_cg, Fsq_cg, '.-')
        axs['Tc'].semilogy(iteration_cg, Tconf_cg, '.-')

axs['u'].legend()
axs['Tc'].legend()

if __name__ == "__main__":
    plt.show(block=True)