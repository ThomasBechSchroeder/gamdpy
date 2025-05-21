import matplotlib.pyplot as plt

import gamdpy as gp

# Setup configuration. BCC Lattice
configuration = gp.Configuration(D=2)
configuration.make_lattice(unit_cell=gp.unit_cells.HEXAGONAL, cells=[16, 10], rho=1.0)

# Setup masses and velocities
configuration['m'] = 1.0  # Set all masses to 1.0
configuration.randomize_velocities(temperature=0.7 * 2)

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
sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=32, steps_per_timeblock=32*1024, 
                    storage='memory')

# Plot initial configuration
plt.figure()

# Run simulation
sim.run()

# Plot final configuration and box
plt.plot(configuration['r'][:, 0], configuration['r'][:, 1], 'o',
         color='blue', label='Final configuration')
box_lengths = configuration.simbox.get_lengths()
x_min, x_max = -box_lengths[0] / 2, box_lengths[0] / 2
y_min, y_max = -box_lengths[1] / 2, box_lengths[1] / 2
x_vals = [x_min, x_max, x_max, x_min, x_min]
y_vals = [y_min, y_min, y_max, y_max, y_min]
plt.plot(x_vals, y_vals, 'k--', label='Box')
plt.axis('equal')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
if __name__ == "__main__":
    plt.show()
