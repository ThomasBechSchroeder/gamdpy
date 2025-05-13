""" Simulate Lennard-Jones system and evaluate the inverse power law potential. 

In this example, we simulate a Lennard-Jones system.
For the last configuration after each timeblock,
we evaluate the r**-12 inverse power law potential (IPL),
and compute the mean.

""" 

import numpy as np

import gamdpy as gp

# Setup configuration: FCC Lattice
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=0.7)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
pair_pot = gp.PairPotential(pair_func, params=[1.0, 1.0, 2.5], max_num_nbs=1000)

# Setup integrator: NVT
integrator = gp.integrators.NVT(temperature=0.7, tau=0.2, dt=0.005)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(16),
                   gp.MomentumReset(100)]


# Setup Simulation.
sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=32,
                    steps_per_timeblock=2048,
                    storage='memory')

# Create evaluator for the inverse power law potential (IPL)
#     (replace with your potential of interest)
pair_func_ref = gp.apply_shifted_potential_cutoff(gp.LJ_12_6)
ipl12 = gp.PairPotential(pair_func_ref, params=[4.0, 0.0, 2.5], max_num_nbs=1000)
evaluator = gp.Evaluator(sim.configuration, ipl12)

# Run simulation
u_ipl = []
for block in sim.run_timeblocks():
    evaluator.evaluate(sim.configuration)  # Evaluate IPL for final configuration of timeblock
    u_ipl.append(np.sum(evaluator.configuration['U']))

print(f'Mean IPL potential energy: {np.mean(u_ipl)}')

