""" Example of a simulation of a Lennard-Jones chain with 10 beads per chain """

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gamdpy as gp

# Generate configuration with a FCC lattice
rho = 1.0
temperature = 0.7
Nchain = 10

# Setup configuration: Single molecule first, then replicate
mol_dict = {"positions" : [ [ i*1.0, (i%2)*.1, 0.] for i in range(Nchain) ],
            "particle_types" : [ 0 for i in range(Nchain) ], 
            "masses"  : [ 1.0 for i in range(Nchain) ]}

top = gp.Topology(['Chain', ])
top.bonds = gp.bonds_from_positions(mol_dict['positions'], cut_off=1.1, bond_type=0)
top.molecules['Chain'] = gp.molecules_from_bonds(top.bonds)
mol_dict['topology'] = top

print(mol_dict)
print(mol_dict['topology'])

configuration = gp.replicate_molecules([mol_dict,], [200,], safety_distance=2.0, 
                                       compute_flags={'Fsq':True, 'lapU':True, 'Vol':True})
configuration.randomize_velocities(temperature=10.0)

# Make bond interactions
bond_potential = gp.harmonic_bond_function
bond_params = [[1.00, 3000.], ]
bonds = gp.Bonds(bond_potential, configuration.topology.bonds, bond_params)

# Make pair potential
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
exclusions = bonds.get_exclusions(configuration)
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], exclusions=exclusions, max_num_nbs=1000)

# Make integrator
dt = 0.002  # timestep
num_blocks = 32  # Do simulation in this many 'blocks'
steps_per_block = 1 * 1024  # ... each of this many steps
running_time = dt * num_blocks * steps_per_block

filename = 'Data/LJchain10_Rho1.00_T0.700'

integrator = gp.integrators.NVT(temperature=temperature, tau=0.2, dt=dt)

runtime_actions = [gp.MomentumReset(100),
                   gp.RestartSaver(),
                   gp.TrajectorySaver(),
                   gp.ScalarSaver(32, {'Fsq':True, 'lapU':True}), ]

sim = gp.Simulation(configuration, [pair_pot, bonds], integrator, runtime_actions,
                    num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                    storage=filename+'_compress.h5')

print('\nCompression and equilibration: ')
initial_rho = configuration.N / configuration.get_volume()
for block in sim.run_timeblocks():
    volume = configuration.get_volume()
    N = configuration.N
    current_rho = N/volume
    print(sim.status(per_particle=True), f'rho= {current_rho:.3}', end='\t')
    print(f'P= {(N*temperature + np.sum(configuration["W"]))/volume:.3}') # pV = NkT + W
    
    # Scale configuration to get closer to final density, rho
    if block<sim.num_blocks/2:
        desired_rho = (block+1)/(sim.num_blocks/2)*(rho - initial_rho) + initial_rho
        if desired_rho > 1.4*current_rho:
            desired_rho = 1.4*current_rho 
        configuration.atomic_scale(density=desired_rho)
        configuration.copy_to_device() # Since we altered configuration, we need to copy it back to device
print(sim.summary()) 
print(configuration)

sim = gp.Simulation(configuration, [pair_pot, bonds], integrator, runtime_actions,
                    num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                    storage=filename+'.h5')

print('Production:')
for block in sim.run_timeblocks():
    if block % 10 == 0:
        print(f'{block=:4}  {sim.status(per_particle=True)}')
print(sim.summary())
