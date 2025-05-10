import sys

import numba
import numpy as np
import pandas as pd

import gamdpy as gp

integrator_name = 'NVE'
if 'NVT' in sys.argv:
    integrator_name = 'NVT'
if 'NVT_Langevin' in sys.argv:
    integrator_name = 'NVT_Langevin'
if 'NPT_Langevin' in sys.argv:            # use with NoRDF since box size is varying
    integrator_name = 'NPT_Langevin'

# Generate configuration with a FCC lattice
configuration = gp.Configuration(D=3, compute_flags={'Fsq':True, 'lapU':True, 'Vol':True})
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.8442)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=1.44)

# Make pair potential
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Make integrator
dt = 0.005  # timestep
num_blocks = 64              # Do simulation in this many 'blocks'
steps_per_block = 2*1024     # ... each of this many steps
running_time = dt*num_blocks*steps_per_block
temperature = 0.7  # Not used for NVE
pressure = 1.2  # Not used for NV*

# Parameters, temperature and pressure, can be functions of time:
#temperature = gp.make_function_ramp(value0=0.7, x0=0.5*running_time, value1=1.7, x1=0.8*running_time)
#pressure = gp.make_function_ramp(value0=1.2, x0=0.3*running_time, value1=3.2, x1=0.6*running_time)

if integrator_name=='NVE':
    integrator = gp.integrators.NVE(dt=dt)
if integrator_name=='NVT':
    integrator = gp.integrators.NVT(temperature=temperature, tau=0.2, dt=dt)
if integrator_name=='NVT_Langevin':
    integrator = gp.integrators.NVT_Langevin(temperature=temperature, alpha=0.2, dt=dt, seed=2023)
if integrator_name=='NPT_Langevin':
    integrator = gp.integrators.NPT_Langevin(temperature=temperature, pressure=pressure,
                                             alpha=0.1, alpha_baro=0.0001, mass_baro=0.0001,
                                             volume_velocity=0.0, barostatModeISO = True , boxFlucCoord = 2,
                                             dt=dt, seed=2023)

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
compute_plan = gp.get_default_compute_plan(configuration)
print(compute_plan)

runtime_actions = [gp.MomentumReset(100),
                   gp.ConfigurationSaver(),
                   gp.ScalarSaver(32, {'Fsq':True, 'lapU':True}), ]


sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                    compute_plan=compute_plan, storage='memory')

for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))
print(sim.summary())

columns = ['U', 'W', 'K', 'Fsq','lapU', 'Vol']
data = np.array(gp.extract_scalars(sim.output, columns, first_block=1))
df = pd.DataFrame(data.T, columns=columns)
df['t'] = np.arange(len(df['U']))* dt * sim.output.attrs["scalar_saver/steps_between_output"]
if integrator_name!='NVE' and callable(temperature):
    df['Ttarget'] = numba.vectorize(temperature)(np.array(df['t']))
if integrator_name=='NPT_Langevin' and callable(pressure):
    df['Ptarget'] = numba.vectorize(pressure)(np.array(df['t']))

gp.plot_scalars(df, configuration.N,  configuration.D, figsize=(10,8), block=True)
