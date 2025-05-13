""" Example using tethered LJ particles 

    Particles with label 1 and 2 are tethered with different Hooke springs.  
    All particles are integrated forward in time with the NVT integrator.
    Density of fluid/free particles and tethered particles is the same.
    Initial and final configurations are saved in xyz format for easy inspection in vmd 
    Both ways of defining the anchor points (from particle indices and types)
    are shown.
"""
import os
import numpy as np
import gamdpy as gp

# Setup a default fcc configuration
use_list = True

nx, ny, nz, rho = 6, 6, 10, 1.0
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[nx, ny, nz], rho=rho)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=2.0)

# Fluid/free particles have type '0', tethered particles '1' and '2'
indices = []; kspring = [];
for n in range(configuration.N):
    if  -3 < configuration['r'][n][2] < -2:
        indices.append(n)
        kspring.append(300)
        configuration.ptype[n] = 1
    elif -1 < configuration['r'][n][1] < 0:
        indices.append(n)
        kspring.append(500)
        configuration.ptype[n] = 2

gp.tools.save_configuration(configuration, "initial.xyz")

# Tether specifications. 
tether = gp.Tether()

if use_list:
    tether.set_anchor_points_from_lists(indices, kspring, configuration)
else:
    tether.set_anchor_points_from_types(particle_types=[1, 2], spring_constants=[300, 500], configuration=configuration)

# Set the pair interactions
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
eps = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
cut = np.array(sig)*2.5 
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator: NVT
integrator = gp.integrators.NVT(temperature=2.0, tau=0.2, dt=0.005)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver()]

# Setup Simulation. Total number of time steps: num_blocks * steps_per_block
sim = gp.Simulation(configuration, [pair_pot, tether], integrator, runtime_actions,
                    num_timeblocks=16, steps_per_timeblock=1024,
                    storage='memory')

# Run simulation one block at a time
for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))

print(sim.summary())

gp.tools.save_configuration(configuration, "final.xyz")

