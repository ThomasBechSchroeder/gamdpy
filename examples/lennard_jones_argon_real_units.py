"""  Lennard-Jones simulation of Argon using real units

The simulation is done in units similar to LAMMPS real units (https://docs.lammps.org/units.html),
where units are derived from

 - energy: kcal/mol,
 - length: Å
 - mass: u

In these units, the Argon parameters of the Lennard-Jones potential
is ε = 0.2384 kcal/mol, σ = 3.4 Å and m = 39.948 u.

This script contains a dictionary (cf) that contain conversion factors,
that converts from simulation units to other units.

"""

from pprint import pprint

import numpy as np

import gamdpy as gp

print('\nDictionary with conversion factors from simulation units to other units:')
cf = gp.conversion_factors(unit_length_in_Angstrom=1.0, unit_energy_in_kcal_per_mol=1.0, unit_mass_in_u=1.0)
pprint(cf)

# Set up simulation in reduced units
configuration = gp.Configuration(D=3)
rho_in_g_per_ml = 1.6  # g/ml (aka g/cm^3 or g/cc)
print(f"Mass density: {rho_in_g_per_ml:.2} g/ml = {rho_in_g_per_ml/cf['g/ml']:.3f} u/Å³ = {rho_in_g_per_ml*cf['u/nm3']/cf['g/ml']:.1f} u/nm³")
mass_of_an_atom = 39.948  # u
rho = rho_in_g_per_ml / cf['g/ml'] / mass_of_an_atom  # Number density in 1/Å³
print(f"Number density: rho = {rho:.4f} atoms/Å³ = {rho/cf['cubic_nanometer']:.1f} atoms/nm³")
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=rho)
configuration['m'] = mass_of_an_atom  # u
temperature_in_K = 84  # K
configuration.randomize_velocities(temperature=temperature_in_K/cf['K'])
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 3.4, 0.2384, 8.5
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
    N, D = configuration.N, configuration.D
    dof = D * N - D
    T_kin = 2.0 * np.sum(configuration['K']) / dof
    T_kin_in_K = T_kin * cf['Kelvin']
    volume = configuration.get_volume()
    rho = N/volume
    P = rho * T_kin + np.sum(configuration['W']) / volume  # Pressure in simulation units
    pressure_in_atm = P * cf['atm']
    print(f'Simulation units: t = {time:>6.1f},    u = {energy:>6.2f},         T_kin = {T_kin:>6.3f},    P = {P:>6.4f}')
    print(f'Real units:       t = {time_in_ps:>6.1f} ps, u = {energy:>6.2f} kcal/mol,  T_kin = {T_kin_in_K:>6.1f} K,  P = {pressure_in_atm:>6.1f} atm')
