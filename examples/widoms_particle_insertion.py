""" Widom's particle insertion method for calculating the chemical potential

In this example we use the Widom's particle insertion method
to calculate the chemical potential of a Lennard-Jones fluid.

The excess chemical potential, μᵉˣ, of a not to dense fluid can be computed as:

  μᵉˣ = -kT ln 〈exp(-ΔU/kT)〉

where ΔU is the energy difference between the system with and without a ghost particle.
And 〈...〉 is an average for all possible positions of the ghost particle 
(sometimes writte as an integral over space).
"""

import numpy as np

import gamdpy as gp

# Setup configuration: FCC Lattice
rho = 0.4  # Number density
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=rho)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=2.0)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator: NVT
temperature = 1.0
integrator = gp.integrators.NVT(temperature=temperature, tau=0.2, dt=0.005)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(),
                   gp.MomentumReset(100)]

# Setup Simulation
sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=20, steps_per_timeblock=1024,
                    storage='memory')

# Equilibrate the system
sim.run()

# Setup the Widom's particle insertion calculator
num_ghost_particles = 500_000
ghost_positions = np.random.rand(num_ghost_particles, configuration.D) * configuration.simbox.lengths
calc_widom = gp.CalculatorWidomInsertion(sim.configuration, pair_pot, temperature, ghost_positions)
print('Production run:')
for block in sim.run_timeblocks():
    calc_widom.update()
    print('.', end='', flush=True)

calc_widom_data = calc_widom.read()
print(f"\nExcess chemical potential: {calc_widom_data['chemical_potential']}")

# Error estimation assuming that timeblocks are statistically independent
mu = calc_widom_data['chemical_potential']
sigma = np.std(calc_widom_data['chemical_potentials']) / np.sqrt(len(calc_widom_data['chemical_potentials']))
print(f"95 % confidence interval: [{mu - 1.96*sigma}, {mu + 1.96*sigma}]")

