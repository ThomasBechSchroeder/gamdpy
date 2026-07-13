"""  Lennard-Jones conversion from Argon units to other units

The simulation is done in reduced units where σ=ε=m=1.
This script contains a dictionary (cf) that contain conversion factors,
from simulation units to other units. The script show how to convert between reduced Argon units and other units.

"""
from pprint import pprint

import numpy as np

import gamdpy as gp

print('Possible input keywords for gamdpy.conversion_factors():')
print(' name: factor_to_SI_units')
pprint(gp.conversion_factors(get_possible_inputs=True))

print('\nDictionary with conversion factors from simulation units to other units:')
cf = gp.conversion_factors(unit_length_in_Angstrom=3.4, unit_energy_in_Kelvin=120, unit_mass_in_u=39.948)
pprint(cf)

# Set up simulation in reduced units
configuration = gp.Configuration(D=3)
density_in_g_per_ml = 1.6  # g/ml (aka g/cm^3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=density_in_g_per_ml/cf['g/ml'])
configuration['m'] = 1.0
temperature_in_K = 84  # K
configuration.randomize_velocities(temperature=temperature_in_K/cf['K'])
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5  # in reduced units
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)
dt_in_fs = 5.0  # fs
tau_in_ps = 0.5  # ps
integrator = gp.integrators.NVT(temperature=temperature_in_K/cf['K'], tau=tau_in_ps/cf['ps'], dt=dt_in_fs/cf['fs'])
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(16),
                   gp.RestartSaver(),
                   gp.MomentumReset(100)]
sim = gp.Simulation(configuration, [pair_pot], integrator, runtime_actions,
                    num_timeblocks=8, steps_per_timeblock=1*1024,
                    storage='memory')

# Run simulation and print information in SI units
for timeblock in sim.run_timeblocks():
    time = (timeblock+1) * sim.steps_per_block * sim.dt
    time_in_ps = (timeblock+1) * sim.steps_per_block * sim.dt * cf['ps']
    energy =  np.mean(configuration['U'])
    energy_in_kJ_per_mol = np.mean(configuration['U']) * cf['kJ/mol']
    N, D = configuration.N, configuration.D
    dof = D * N - D
    T_kin = 2.0 * np.sum(configuration['K']) / dof
    T_kin_in_K = T_kin * cf['Kelvin']
    volume = configuration.get_volume()
    rho = N/volume
    P = rho * T_kin + np.sum(configuration['W']) / volume  # Pressure in reduced units
    pressure_in_MPa = P * cf['MPa']
    print(f'Reduce units: t = {time:>6.1f},     u = {energy:>6.2f},         T_kin = {T_kin:>6.1f},    P = {P:>6.1f}')
    print(f'Real units:   t = {time_in_ps:>6.1f} ps,  u = {energy_in_kJ_per_mol:>6.2f} kJ/mol,  T_kin = {T_kin_in_K:>6.1f} K,  P = {pressure_in_MPa:>6.1f} MPa')
