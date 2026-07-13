""" Purely repulsive active particles forming dense clusters (MIPS) """

import gamdpy as gp

configuration = gp.Configuration(D=2)
configuration.make_lattice(unit_cell=gp.unit_cells.HEXAGONAL, cells=[16, 10], rho=0.6)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=0.7 * 2)

pair_func = gp.harmonic_repulsion
eps, sig = 10.0, 1.0
pair_pot = gp.PairPotential(pair_func, params=[eps, sig], max_num_nbs=1000)

# Set DA=0.0 for passive particles (for comparison)
integrator = gp.integrators.ActiveOUP(DT=0.25, DA=40.0, mu=1.0, tau=5.0, dt=0.005, seed=2025)

runtime_actions = [gp.RestartSaver(),
                   gp.TrajectorySaver(),
                   gp.ScalarSaver(16)]

sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=8, steps_per_timeblock=8*1024,
                    storage='Data/ActiveOUP.h5')

for block in sim.run_timeblocks():
    print(f'{sim.status(per_particle=True)}')
print(sim.summary())

import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))
plt.plot(configuration['r'][:, 0], configuration['r'][:, 1], 'o', color='blue')
plt.axis('equal')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()
