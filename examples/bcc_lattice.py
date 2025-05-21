""" Example of a BCC lattice simulation with Lennard-Jones potential.

This example demonstrates how to set up a different lattice than the default FCC lattice.
Note more lattices are available, or you can define your own lattice.

"""

import gamdpy as gp

# Setup configuration. BCC Lattice
configuration = gp.Configuration(D=3)
configuration.make_lattice(unit_cell=gp.unit_cells.BCC, cells=[8, 8, 8], rho=1.0)

# Setup masses and velocities
configuration['m'] = 1.0  # Set all masses to 1.0
configuration.randomize_velocities(temperature=0.7)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator
integrator = gp.integrators.NVT(temperature=0.7, tau=0.2, dt=0.005)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(),
                   gp.MomentumReset(100)]

# Setup Simulation.
sim = gp.Simulation(configuration, [pair_pot, ], integrator, runtime_actions,
                    num_timeblocks=32, steps_per_timeblock=1024, 
                    storage='memory')

# Run simulation
sim.run()
