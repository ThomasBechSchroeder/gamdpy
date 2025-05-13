""" Analyze and plot dynamics computed from a .h5 file 
    Usage:

    analyze_dynamics filename
"""

import matplotlib.pyplot as plt
import gamdpy as gp
import numpy as np
import sys
import pickle

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

dynamics = gp.tools.calc_dynamics(output, 0, qvalues=7.5)
with open(filename+'_dynamics.pkl', 'wb') as f:     
    pickle.dump(dynamics, f)
print(f"Wrote: {filename+'_dynamics.pkl'}")

fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
fig.subplots_adjust(hspace=0.00)  # Remove vertical space between axes
axs[0].set_ylabel('MSD')
axs[1].set_ylabel('Non Gaussian parameter')
axs[2].set_ylabel('Intermediate scattering function')
axs[2].set_xlabel('Time')
axs[0].grid(linestyle='--', alpha=0.5)
axs[1].grid(linestyle='--', alpha=0.5)
axs[2].grid(linestyle='--', alpha=0.5)

num_types = dynamics['msd'].shape[1]
for i in range(num_types):
    axs[0].loglog(dynamics['times'], dynamics['msd'][:,i], 'o--', label=f'{i}')
    axs[1].semilogx(dynamics['times'], dynamics['alpha2'][:,i], 'o--')
    axs[2].semilogx(dynamics['times'], dynamics['Fs'][:,i], 'o--', label=f'{i}, q={dynamics["qvalues"][i]}')
factor = np.array([1, 30])
axs[0].loglog(dynamics['times'][0]*factor, np.max(dynamics['msd'][0,:])*factor**2, 'k--', alpha=0.5, label='Slope 2')
axs[0].loglog(dynamics['times'][-1]/factor, np.min(dynamics['msd'][-1,:])/factor, 'k-.', alpha=0.5, label='Slope 1')

axs[0].set_title(filename)
axs[0].legend()
axs[2].legend()
fig.savefig(filename+'_dynamics.pdf')
print(f"Wrote: {filename+'_dynamics.pdf'}")

if __name__ == "__main__":
    plt.show(block=True)

