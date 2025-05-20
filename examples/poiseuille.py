""" Example of a nanoslit pore simulation using tethered LJ particles 

    The particles are tethered with a Hooke spring. The wall particles interact with a relaxation device.   
    All particles are integrated forward in time with the NVE integrator

    Wall density is set to 1.0 and fluid density to 0.8 - this is achieved by
    inclusion of a dummy particle type 

    Initial and final configurations are saved in xyz format for easy inspection in vmd
"""

import random
import os

import numpy as np

import gamdpy as gp

# Some system parameters
nx, ny, nz = 6, 6, 10
rhoWall = 1.0
rhoFluid = 0.7

# Setup a default fcc configuration
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[nx, ny, nz], rho=rhoWall)
configuration['m'] = 1.0


# Fluid particles have type '0', wall particles '1', dummy particles '2'
nwall, npart = 0, configuration.N
hlz = 0.5 * configuration.simbox.get_lengths()[2]
for n in range(npart):
    if configuration['r'][n][2] + hlz < 3.0:
        configuration.ptype[n] = 1
        nwall = nwall + 1

nfluid = np.sum(configuration.ptype == 0)
nfluidWanted = int(nfluid * rhoFluid / rhoWall)

while nfluid > nfluidWanted:
    idx = random.randint(0, npart - 1)
    if configuration.ptype[idx] == 0:
        configuration.ptype[idx] = 2
        nfluid = nfluid - 1

gp.tools.save_configuration(configuration, "initial.xyz")

# Tether specifications. 
tether = gp.Tether()
tether.set_anchor_points_from_types(particle_types=[1], spring_constants=[300.0], configuration=configuration)

# Add gravity force 
grav = gp.Gravity()
grav.set_gravity_from_types(particle_types=[0], forces=[0.01], configuration=configuration)

# Temp relaxation for wall particles
relax = gp.Relaxtemp()
relax.set_relaxation_from_types(particle_types=[1], temperature=[2.],
                                relax_times=[0.01],configuration=configuration); 

# Set the pair interactions
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
eps = [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
cut = np.array(sig) * 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Temperature
configuration.randomize_velocities(temperature=2.0)

# Setup integrator: NVT
integrator = gp.integrators.NVE(dt=0.005)

# Compute plan
compute_plan = gp.get_default_compute_plan(configuration)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver()]

# Setup Simulation. Total number of time steps: num_blocks * steps_per_block
sim = gp.Simulation(configuration, [pair_pot, tether, grav, relax], integrator, runtime_actions,
                    num_timeblocks=100, steps_per_timeblock=64,
                    storage='memory', compute_plan=compute_plan)

prof = gp.CalculatorHydrodynamicProfile(configuration, 0)

# Run simulation one block at a time
for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))
    prof.update()

print(sim.summary())

prof.read()

gp.tools.save_configuration(configuration, "final.xyz")

