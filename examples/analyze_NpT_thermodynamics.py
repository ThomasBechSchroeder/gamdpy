""" Compute thermodynamic response functions from an NpT simulation.

Usage:

    python analyse_NpT_thermodynamics.py <filename>

"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

import gamdpy as gp

max_plot_points = 100_000

argv = sys.argv.copy()
argv.pop(0)  # remove script name
if __name__ == "__main__":
    if argv:
        filename_h5 = argv.pop(0) # get h5 filename as the first argument of the command line
    else:
        filename_h5 = 'Data/LJ_p4.70_T2.0_toread.h5' # Used in testing
else:
    filename_h5 = 'Data/LJ_p4.70_T2.0_toread.h5'

fname_root, fname_suffix = os.path.splitext(filename_h5)

if fname_suffix != '.h5':  # Try to add the .h5 extension
    fname_root = filename_h5
    fname_suffix = '.h5'
    filename_h5 = fname_root + fname_suffix

# Read thermodynamic data and compute response functions
output = gp.read_h5(filename_h5)
*_, N, D = output['trajectory/positions'].shape
dof = D * N - D
fluctuations = gp.ScalarSaver.extract_as_dict(output)  # Read fluctuation data
response_functions = gp.tools.thermodynamics_NpT(N, dof, **fluctuations, k_B=1.0)

# Print and write data
to_toml_file = ""
for key in response_functions:
    print(f'{key:>42} = {response_functions[key]:10.5f}')
    to_toml_file += f'{key} = {response_functions[key]}' + '\n'
print(to_toml_file, file=open(fname_root + '_NpT_thermodynamics.toml', 'w'))
print('Wrote:', fname_root+'_NpT_thermodynamics.toml')

# Plot fluctuations
U = fluctuations['U']
Vol = fluctuations['Vol']
K = fluctuations['K']
times = gp.ScalarSaver.get_times(output)

plotindex = range(len(U))
if len(U)>max_plot_points:
    step = int(len(U)/max_plot_points+1)
    plotindex = plotindex[::step]

fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
fig.subplots_adjust(hspace=0.00)  # Remove vertical space between axes
gamma = response_functions['configurational_adiabatic_scaling_exponent']
c_p = response_functions['isobaric_heat_capacity']
axs[0].set_title(r'$N=$' f'{N},  ' r'$\rho=$' f'{response_functions['density']:.3f},  ' r'$T_\text{kin}=$' f'{response_functions['kinetic_temperature']:.3f},  p={response_functions['pressure']:.3f}, ' r'$c_p=$' f'{c_p:.2f}, ' r'$\gamma=$' f'{gamma:.2f}')
axs[0].set_ylabel('U/N')
axs[1].set_ylabel('V/N')
axs[2].set_ylabel('K/N')
axs[2].set_xlabel('Time')
axs[0].grid(linestyle='--', alpha=0.5)
axs[1].grid(linestyle='--', alpha=0.5)
axs[2].grid(linestyle='--', alpha=0.5)

label  = f'mean: {np.mean(U):.3f}   std: {np.std(U):.3f}'
axs[0].plot(times[plotindex], U[plotindex], label=label)
axs[0].axhline(np.mean(U), color='k', linestyle='--')
axs[0].legend(loc='upper right')

label  = f'mean: {np.mean(Vol):.3f}   std: {np.std(Vol):.3f}'
axs[1].plot(times[plotindex], Vol[plotindex] , label=label)
axs[1].axhline(np.mean(Vol) , color='k', linestyle='--')
axs[1].legend(loc='upper right')

label  = f'mean: {np.mean(K):.3f}   std: {np.std(K):.3f}'
axs[2].plot(times[plotindex], K[plotindex] , label=label)
axs[2].axhline(np.mean(K) , color='k', linestyle='--')
axs[2].legend(loc='upper right')

fig.savefig(fname_root+'_NpT_thermodynamics.pdf')
print('Wrote:', fname_root+'_NpT_thermodynamics.pdf')
if __name__ == "__main__":
    plt.show(block=True)
