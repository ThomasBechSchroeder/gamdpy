""" Example of a binary LJ simulation using gamdpy.

NVT simulation of the Kob-Andersen mixture, and compare results with Rumd3 (rumd.org)
"""

import gamdpy as gp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path

# Specify statepoint
num_part = 2000
rho = 1.200
temperature = 0.80

# Setup configuration: 
configuration = gp.Configuration(D=3, compute_flags={'Fsq':True, 'lapU':True, 'Vol':True})
configuration.make_positions(N=num_part, rho=rho)
configuration['m'] = 1.0 # Specify all masses to unity 
configuration.randomize_velocities(temperature=2.0) # Initial high temperature for randomizing
configuration.ptype[::5] = 1 # Every fifth particle set to type 1 (4:1 mixture)

# Setup pair potential: Binary Kob-Andersen LJ mixture.
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig = [[1.00, 0.80],
       [0.80, 0.88]]
eps = [[1.00, 1.50],
       [1.50, 0.50]]
cut = np.array(sig)*2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator. 
# Increase 'num_blocks' for longer runs, better statistics, AND bigger storage consumption
# Increase 'steps_per_block' for longer runs
dt = 0.004  # timestep
num_timeblocks = 32           # Do simulation in this many 'blocks'. 
steps_per_timeblock = 2*1024  # ... each of this many steps
running_time = dt*num_timeblocks*steps_per_timeblock
filename = f'Data/KABLJ_Rho{rho:.3f}_T{temperature:.3f}.h5'

print('High Temperature followed by cooling and equilibration:')
Ttarget_function = gp.make_function_ramp(value0=2.000,       x0=running_time*(1/8),
                                         value1=temperature, x1=running_time*(1/4))
integrator = gp.integrators.NVT(temperature=Ttarget_function, tau=0.2, dt=dt)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.MomentumReset(100),]

sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions, 
                    num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock,
                    storage="memory") 

for block in sim.run_timeblocks():
    print(f'{sim.status(per_particle=True)}')
print(sim.summary())

# Print current status of configuration
print(configuration)

print('\nProduction:')
integrator = gp.integrators.NVT(temperature=temperature, tau=0.2, dt=dt)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
#runtime_actions = [gp.ConfigurationSaver(),
#                   gp.ScalarSaver(16, {'Fsq':True, 'lapU':True}),
#                   gp.MomentumReset(100)]

runtime_actions = [gp.MomentumReset(100),
                   gp.ConfigurationSaver(),
                   gp.ScalarSaver(16, {'Fsq':True, 'lapU':True}), ]


sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock,
                    storage=filename)
for block in sim.run_timeblocks():
    print(f'{sim.status(per_particle=True)}')
print(sim.summary())

# Print current status of configuration
print(configuration)



columns = ['U', 'W', 'K', 'Fsq', 'lapU', 'Vol']
data = np.array(gp.extract_scalars(sim.output, columns, first_block=0))
df = pd.DataFrame(data.T, columns=columns)
df = pd.DataFrame(data.T, columns=columns)
df['t'] = np.arange(len(df['U'])) * dt * sim.output['scalar_saver'].attrs["steps_between_output"]

mu = np.mean(df['U'])/configuration.N
mw = np.mean(df['W'])/configuration.N
cvex = np.var(df['U'])/temperature**2/configuration.N

print('\ngamdpy:')
print(f'Potential energy:     {mu:.4f}')
print(f'Excess heat capacity: {cvex:.3f}')
print(f'Virial                {mw:.4f}')

if rho==1.200 and temperature==0.800:
       mu3 = -6.346
       mw3 =  5.534
       cvex3 = 0.0001089086505*10000/0.8**2
       print('\nRumd3:')
       print(f'Potential energy:     {mu3:.4f}')
       print(f'Excess heat capacity: {cvex3:.3f}')
       print(f'Virial                {mw3:.4f}')

if __name__ == "__main__":
    gp.plot_scalars(df, configuration.N,  configuration.D, figsize=(10,8), block=False)

dyn = gp.tools.calc_dynamics(sim.output, first_block=0, qvalues=[7.5, 5.5])
fig, axs = plt.subplots(1, 1, figsize=(6,4))
axs.loglog(dyn['times'], dyn['msd'], '.-', label=['A (gamdpy)', 'B (gamdpy)'])
axs.set_xlabel('Time')
axs.set_ylabel('MSD')

rumd3filename = f'Data/KABLJ_msd_R{rho:.3f}_T{temperature:.3f}_rumd3.dat'
if os.path.isfile(rumd3filename):
     msd3 = np.loadtxt(rumd3filename)
     axs.loglog(msd3[:21,0], msd3[:21,1:], '--', label=['A (rumd3)', 'B (rumd3)'])

axs.legend()
if __name__ == "__main__":
    plt.show(block=True)

if rho==1.200 and temperature==0.800:
     print('\nTesting complience with Rumd3:')
     assert abs(mu - mu3)     < 0.01, f"{mu=} but in rumd3 is {mu3=}"
     assert abs(cvex - cvex3) < 0.2 , f"{cvex=} but in rumd3 is {cvex3=}"
     assert abs(mw - mw3)     < 0.03, f"{mw=} but in rumd3 is {mw3=}"
     print('Passed')
