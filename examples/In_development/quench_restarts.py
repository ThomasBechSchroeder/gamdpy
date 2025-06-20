""" Using scipy to 'quench' configurations saved as restarts in a trajectory h5 file

For now the model is single component LJ, as found eg. in Data/LJ_r0.973_T0.70_toread.h5
- will be changed to Kob&Andersen

Usage:
    python3 quench_restarts.py filename
"""

import gamdpy as gp
import numpy as np
import numba
import matplotlib.pyplot as plt
import sys
import pickle
import h5py
from scipy.optimize import minimize

gp.select_gpu()

# Choose minimization method, see:
#  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
method = 'CG' # Conjugate Gradient

argv = sys.argv.copy()
argv.pop(0)  # remove scriptname
if __name__ == "__main__":
    if argv:
        filename = argv.pop(0) # get filename (.h5 added by script)
    else:
        filename = 'Data/LJ_r0.973_T0.70_toread' # Used in testing
else:
    filename = 'Data/LJ_r0.973_T0.70_toread' # Used in testing

# Load existing configuration, twice for convenience 
with h5py.File(filename, 'r') as f:
    configuration1 = gp.Configuration.from_h5(f, "restarts/restart0000", compute_flags={'lapU':True})
    configuration2 = gp.Configuration.from_h5(f, "restarts/restart0000", compute_flags={'lapU':True})
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

# function to interface with minimize function from 
def calc_u(Rflat):
        configuration2['r'] = Rflat.reshape(N,D).astype('float32')
        evaluator.evaluate(configuration2)
        return np.sum(configuration2['U'].astype('float64'))

def calc_du(Rflat):
        configuration2['r'] = Rflat.reshape(N,D).astype('float32')
        evaluator.evaluate(configuration2)
        return -configuration2['f'].astype('float64').flatten()

fig = plt.figure(figsize=(8, 14))
axs = fig.subplot_mosaic([["u", "u"],
                          ["lu", "lu"],
                          ["du", "du"],
                          ["Tc", "Tc"],
                          ["dr", "dr"],
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
axs['dr'].set_ylabel('(R - Rmin)**2 / N')
axs['dr'].set_xlabel('Iteration')
axs['dr'].grid(linestyle='--', alpha=0.5)

for restart in range(5):
    with h5py.File(filename, 'r') as f:
        configuration2 = gp.Configuration.from_h5(f, f"restarts/restart{restart:04d}", compute_flags={'lapU':True})


    R0flat = configuration2['r'].astype('float64').flatten()
    res = minimize(calc_u, R0flat, method=method, jac=calc_du, options={'gtol': 1e-9, 'disp': True, 'return_all':True})

    configuration2['r'] = res.x.reshape(N,D).astype('float32')
    evaluator.evaluate(configuration2)
    print(np.sum(configuration2['f']**2)/N)

    u = []
    du = []
    dr = []
    Tconf = []

    for x in res.allvecs:
        u.append(calc_u(x)/N)
        Fsq = np.sum(calc_du(x)**2)
        du.append(Fsq/N)
        dr.append(np.sum( (x - res.x)**2/N ))
        Tconf.append( Fsq / np.sum(configuration2['lapU']) )

    u = np.array(u)
    du = np.array(du)
    Tconf = np.array(Tconf)
    dr = np.array(dr) 

    axs['u'].plot(u, '.-', label=f'Umin/N = {u[-1]:.10f}')
    axs['lu'].semilogy(u-u[-1], '.-')
    axs['du'].semilogy(du, '.-')
    axs['Tc'].semilogy(Tconf, '.-', label=f'{Tconf[0]:.2e} -> {Tconf[-1]:.3e}')
    axs['dr'].semilogy(dr, '.-', label=f'{dr[0]:.2e} -> {dr[-2]:.3e}')

axs['u'].legend()
axs['Tc'].legend()
axs['dr'].legend()

if __name__ == "__main__":
    plt.show(block=True)