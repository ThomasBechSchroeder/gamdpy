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
        filename = 'Data/KABLJ_Rho1.200_T0.400_toread' # Used in testing
else:
    filename = 'Data/KABLJ_Rho1.200_T0.400_toread' # Used in testing

# Load existing data
output = gp.tools.TrajectoryIO(filename+'.h5').get_h5()
ptype = output['initial_configuration/ptype'][:].copy()

qA = 0.5
qB = -2*qA
charges = (ptype==0)*qA + (ptype==1)*qB
print(f'{np.sum(charges)=}')

conductivity = gp.tools.calc_conductivity(output, 0, charges)
with open(filename+'_conductivity.pkl', 'wb') as f:     
    pickle.dump(conductivity, f)
print(f"Wrote: {filename+'_conductivity.pkl'}")

fig, axs = plt.subplots(2, 1, figsize=(9, 8))
axs[0].set_ylabel('Conductivity')
axs[0].set_xlabel('Time')
axs[0].grid(linestyle='--', alpha=0.5)
plotstop = -2

axs[0].loglog(conductivity['times'], conductivity['nernst_einstein'], 'o--', label='nernst_einstein')
axs[0].loglog(conductivity['times'], conductivity['einstein'], 'o--', label='einstein')
axs[0].loglog(conductivity['times'], abs(conductivity['crossterm']), '--', label='abs(crossterm)')
axs[0].loglog(conductivity['times'], abs(conductivity['einstein'] - conductivity['nernst_einstein']), 'o', label='abs(einstein - nernst_einstein)')
axs[0].set_title(filename)
axs[0].legend()

axs[1].set_ylabel('Conductivity')
axs[1].set_xlabel('Time')
axs[1].grid(linestyle='--', alpha=0.5)

axs[1].plot(conductivity['times'][:plotstop], conductivity['nernst_einstein'][:plotstop], 'o--', label='nernst_einstein')
axs[1].plot(conductivity['times'][:plotstop], conductivity['einstein'][:plotstop], 'o--', label='einstein')
axs[1].plot(conductivity['times'][:plotstop], abs(conductivity['crossterm'][:plotstop]), '--', label='abs(crossterm)')
axs[1].plot(conductivity['times'][:plotstop], abs(conductivity['einstein'][:plotstop] - conductivity['nernst_einstein'][:plotstop]), 'o', label='abs(einstein - nernst_einstein)')
axs[1].set_title(filename)
axs[1].legend()


fig.savefig(filename+'_conductivity.pdf')
print(f"Wrote: {filename+'_conductivity.pdf'}")

if __name__ == "__main__":
    plt.show(block=True)

