#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 13:33:58 2025

@author: nbailey
"""

import numpy as np
import gamdpy as gp

gp.select_gpu()

# Simulation params 
rho, temperature = 0.85, 1.5
N_nominal = 2000
chain_lengths = [5, 10]
composition = [1, 2] # non-negative integers. Should be relatively prime (no common factors)

#  size_base is the number of atoms in the smallest repeating unit
size_base = 0
for cl, frac in zip(chain_lengths, composition):
    size_base += cl*frac

num_base_units = N_nominal // size_base
num_mols_each_type = []
N_mol = 0
N = 0

for cl, frac in zip(chain_lengths, composition):
    num_mols_each_type.append(frac*num_base_units)
    N_mol += num_mols_each_type[-1]
    N += num_mols_each_type[-1] * cl

#N_mol = N_nominal//chain_length
#N = N_mol * chain_length


print(f"N={N}; N_mol={N_mol};num_base_units={num_base_units}")
print("num_mols_each_type")
print(num_mols_each_type)

filename = 'Data/chains_poly'
num_timeblocks_equilibration = 64
num_timeblocks_production = 16
steps_per_timeblock = 1 * 1024

molecule_dicts = []


for index, cl in enumerate(chain_lengths):
    pos_this_mol = []
    types_this_mol = []
    masses_this_mol = []
    for i in range(cl):
        pos_this_mol.append( [ i*1.0, (i%2)*.1, 0. ] ) # x, y, z for this particle
        types_this_mol.append( index ) # all particles in a molecule have the same type, but this differs from molecule to molecule
        masses_this_mol.append( 1.0 )  

    # Setup configuration: Single molecule first, then duplicate
    top_this_mol = gp.Topology([f'MyMolecule{cl}', ])
    top_this_mol.bonds = gp.bonds_from_positions(pos_this_mol, cut_off=1.1, bond_type=0)
    top_this_mol.angles = gp.angles_from_bonds(top_this_mol.bonds, angle_type=0)
    top_this_mol.dihedrals = gp.dihedrals_from_angles(top_this_mol.angles, dihedral_type=0)
    top_this_mol.molecules[f'MyMolecule{cl}'] = gp.molecules_from_bonds(top_this_mol.bonds)


    dict_this_mol = {"positions" : pos_this_mol,
                     "particle_types" : types_this_mol,
                     "masses" : masses_this_mol,
                     "topology" : top_this_mol}
    molecule_dicts.append(dict_this_mol)

    print(f'Initial Positions for molecule with chain length: {cl}')
    for position in pos_this_mol:
        print('\t\t', position)
    print('Particle types:\t', types_this_mol)
    print('Bonds:         \t', top_this_mol.bonds)
    print('Angles:        \t', top_this_mol.angles)
    print('Dihedrals:     \t', top_this_mol.dihedrals)
    print()


    # This call creates the pdf "molecule.pdf" with a drawing of the molecule 
    # Use block=True to visualize the molecule before running the simulation
    gp.plot_molecule(top_this_mol, pos_this_mol, types_this_mol, filename=f"chain{cl}.pdf", block=False)


configuration = gp.replicate_molecules(molecule_dicts, num_mols_each_type, safety_distance=3.0)
configuration.randomize_velocities(temperature=temperature)

print(f'Number of molecules: {len(configuration.topology.molecules[f"MyMolecule{chain_lengths[0]}"])}, {len(configuration.topology.molecules[f"MyMolecule{chain_lengths[1]}"])}')
print(f'Number of particles: {configuration.N}\n')

# Make bond interactions
bond_potential = gp.harmonic_bond_function
bond_params = [[0.8, 1000.], ]
bonds = gp.Bonds(bond_potential, bond_params, configuration.topology.bonds)

# Make angle interactions
angle0, k = 2.0, 500.0
angle_potential = gp.cos_angle_function
angles = gp.Angles(angle_potential, configuration.topology.angles, parameters=[[k, angle0],])

# Make dihedral interactions
rbcoef=[.0, 5.0, .0, .0, .0, .0]    
dihedral_potential = gp.ryckbell_dihedral
dihedrals = gp.Dihedrals(dihedral_potential, configuration.topology.dihedrals, parameters=[rbcoef, ])

# Exlusion list
exclusions = dihedrals.get_exclusions(configuration)
#exclusions = bonds.get_exclusions(configuration)

# Make pair potential
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig = [[1.00, 0.95],
       [0.95, 0.9]]
eps = [[1.00, 0.95],
       [0.95, 0.90]]
cut = np.array(sig)*2.5



pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], exclusions=exclusions, max_num_nbs=1000)

# Make integrator
integrator = gp.integrators.NVT(temperature=temperature, tau=0.1, dt=0.004)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(),
                   gp.MomentumReset(100)]

# Setup simulation
sim = gp.Simulation(configuration, [pair_pot, bonds, angles, dihedrals], integrator, runtime_actions,
                    num_timeblocks=num_timeblocks_equilibration, steps_per_timeblock=steps_per_timeblock,
                    storage='memory')

print('\nCompression and equilibration: ')
dump_filename = 'Data/dump_compress.lammps'
with open(dump_filename, 'w') as f:
    print(gp.configuration_to_lammps(sim.configuration, timestep=0), file=f)


initial_rho = configuration.N / configuration.get_volume()
for block in sim.run_timeblocks():
    volume = configuration.get_volume()
    N = configuration.N
    current_rho = N/volume
    print(sim.status(per_particle=True), f'rho= {current_rho:.3}', end='\t')
    print(f'P= {(N*temperature + np.sum(configuration["W"]))/volume:.3}') # pV = NkT + W
    with open(dump_filename, 'a') as f:
        print(gp.configuration_to_lammps(sim.configuration, timestep=sim.steps_per_block*(block+1)), file=f)

    # Scale configuration to get closer to final density, rho
    if block<sim.num_blocks/2:
        desired_rho = (block+1)/(sim.num_blocks/2)*(rho - initial_rho) + initial_rho
        if desired_rho > 1.2*current_rho:
            desired_rho = 1.2*current_rho 
        configuration.atomic_scale(density=desired_rho)
        configuration.copy_to_device() # Since we altered configuration, we need to copy it back to device
print(sim.summary()) 
print(configuration)

sim = gp.Simulation(configuration, [pair_pot, bonds, angles, dihedrals], integrator, runtime_actions,
                    num_timeblocks=num_timeblocks_production, steps_per_timeblock=steps_per_timeblock,
                    compute_plan=sim.compute_plan, storage=filename+'.h5')

print('\nProduction: ')
dump_filename = 'Data/dump.lammps'
with open(dump_filename, 'w') as f:
    print(gp.configuration_to_lammps(sim.configuration, timestep=0), file=f)


for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))
    with open(dump_filename, 'a') as f:
        print(gp.configuration_to_lammps(sim.configuration, timestep=sim.steps_per_block*(block+1)), file=f)

print(sim.summary()) 
print(configuration)

