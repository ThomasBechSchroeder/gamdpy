""" Simulation of water with real units.
    Starts from a dilute setup which is compressed. The production run.
    https://doi.org/10.1063/1.2136877
    Model: SPC/Fw [Wu et al. J. Chem. Phys. 124:024503 (2006)]
"""

import gamdpy as gp
import numpy as np
from math import exp

# Conversion factor dictionary
cf = gp.conversion_factors(
    unit_length_in_Angstrom=3.165492,       # Oxygen Lennard-Jones sigma
    unit_energy_in_kcal_per_mol=0.1554253,  # Oxygen Lennard-Jones epsilon
    unit_mass_in_u=15.999,                # Oxygen mass
    unit_charge_in_e=1.0,                   # Use unit charge
)

number_of_molecules = 500

rho_in_g_per_cm3 = 0.9970
molecule_mass = (16 + 2*1)/cf['u']
atoms_per_molecule = 3
g_per_cm3 = cf['g/cm3'] * molecule_mass / atoms_per_molecule
rho_atoms = rho_in_g_per_cm3 / g_per_cm3
rho_desired = rho_atoms
rho_initial = rho_desired / 8  # used for inflated initial box

temperature_in_Celcius = 25.0
temperature_in_K = temperature_in_Celcius + 273.15
temperature = temperature_in_K/cf['K']

qH = 0.41  # e
qO = -2 * qH  # e
qH_natural = qH * cf['charge_coulomb_natural_units']  # Natural units where Columbs constant is k_e = 1.0
qO_natural = qO * cf['charge_coulomb_natural_units']

mH = 1.008/cf['u']  # 1.0
mO = 1.0

bond_length_in_Angstrom = 1.012  # Å
bond_length = bond_length_in_Angstrom / cf['Angstrom']
bond_spring_constant_real =  1059.162  # kcal/mol/Å²
bond_spring_constant = bond_spring_constant_real / cf['kcal/mol'] * cf['Angstrom'] ** 2

angle_in_degrees = 113.24
angle = angle_in_degrees / cf['degrees']
angle_spring_constant_real = 75.90  # kcal/mol/rad**2
angle_spring_constant = angle_spring_constant_real / cf['kcal/mol']

# Timestep
dt_in_fs = 1.0
dt = dt_in_fs/cf['fs']
print(f'Timestep: {dt_in_fs} fs   ({dt = } in reduced units)')

# Thermostat relaxation time
tau_in_fs = 150
tau = tau_in_fs/cf['fs']

# Compression factor for equilibration
alpha = 10.0

# Atom positions; H-O-H
r0 = [[0.00, 0.184, 0.0],  # H
      [0.26, 0.0, 0.0],    # O
      [0.53, 0.184, 0.0]]  # H
mass = [mH, mO, mH]
types = [0, 1, 0]

top = gp.Topology(['water', ])
top.bonds = gp.bonds_from_positions(r0, cut_off=0.5, bond_type=0)
top.angles = gp.angles_from_bonds(top.bonds, angle_type=0)
top.molecules['water'] = gp.molecules_from_bonds(top.bonds)

dict_this_mol = {"positions": r0,
                 "particle_types": types,
                 "masses": mass,
                 "topology": top}

configuration = gp.replicate_molecules([dict_this_mol], [number_of_molecules], safety_distance=2.0)
configuration.randomize_velocities(temperature=temperature)

# Make bonds
bond_potential = gp.harmonic_bond_function
bond_params = [[bond_length, bond_spring_constant], ]
bonds = gp.Bonds(bond_potential, configuration.topology.bonds, bond_params)

# Angles
angle_potential = gp.cos_angle_function
angle_params = [[angle, angle_spring_constant], ]
angles = gp.Angles(angle_potential, configuration.topology.angles, angle_params)

# Angle exclusions
exclusion = angles.get_exclusions(configuration)

# Make pair potential
pair_func = gp.LJ_coulomb_sf
sig = [
    [0.0, 0.0],
    [0.0, 1.0]
]
eps = [
    [0.0, 0.0],
    [0.0, 1.0]
]
charge = [
    [qH_natural * qH_natural, qH_natural * qO_natural],
    [qO_natural * qH_natural, qO_natural * qO_natural]
]
cut_lj = [
    [3.0, 3.0],
    [3.0, 3.0]
]
cut_coulomb = [
    [3.5, 3.5],
    [3.5, 3.5]
]

pair_pot = gp.PairPotential(
    pair_func,
    params=[sig, eps, charge, cut_lj, cut_coulomb],
    exclusions=exclusion,
    max_num_nbs=1000
)

# Make integrator
integrator = gp.integrators.NVT(temperature=temperature, tau=tau, dt=dt)

# Set up runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.MomentumReset(100), ]

# Eq. setup simulation
sim = gp.Simulation(configuration, [pair_pot, bonds, angles], integrator, runtime_actions,
                    num_timeblocks=3000, steps_per_timeblock=32, storage='memory')

npart = configuration.N
for block in sim.run_timeblocks():
    rho = npart / configuration.simbox.get_volume()
    prefac = exp(-alpha * rho / rho_desired) + 1.0
    if prefac > 1.02:
        prefac = 1.02

    if rho < rho_desired:
        rho = prefac * rho
        configuration.atomic_scale(density=rho)
        configuration.copy_to_device()
    if block % 100 == 0:
        print(f'Equbriliation {block}/{sim.num_blocks} at {rho*g_per_cm3:4f} g/cm3')

print(f'Done with equbriliation, rho = {rho*g_per_cm3:4f} g/cm3  ({rho = :4f} in reduced units)')
print(sim.status(per_particle=True))

runtime_actions = [gp.MomentumReset(100), gp.TrajectorySaver()]

sim = gp.Simulation(configuration, [pair_pot, bonds, angles], integrator, runtime_actions,
                    num_timeblocks=10, steps_per_timeblock=256, storage='Data/water.h5')

for timeblock in sim.run_timeblocks():
    time = (timeblock + 1) * sim.steps_per_block * sim.dt
    time_in_ps = time * cf['ps']
    N, D = configuration.N, configuration.D
    dof = D * N - D
    T_kin = 2.0 * np.sum(configuration['K']) / dof
    T_kin_in_K = T_kin * cf['Kelvin']
    volume = configuration.get_volume()
    rho = N/volume
    P = rho * T_kin + np.sum(configuration['W']) / volume  # Pressure in simulation units
    pressure_in_atm = P * cf['atm']
    print(f"t ={time_in_ps:.2f} ps   T = {T_kin_in_K:.2f} K   p = {pressure_in_atm:.2f} atm")

print(sim.status(per_particle=True))


