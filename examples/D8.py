""" A simulation in eight dimensional space """

import numba

import gamdpy as gp

N = 1024  # Number of particles (65536)
temperature = 1.0

configuration = gp.Configuration(D=8)
configuration.make_positions(N, rho=1.0)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature)

pair_func = numba.njit(gp.harmonic_repulsion)
pair_potential = gp.PairPotential(pair_func, params=[2.0, 1.0], max_num_nbs=8192)

integrator = gp.integrators.NVT(temperature=temperature, tau=0.08, dt=0.001)

runtime_actions = [gp.MomentumReset(100),
                   gp.TrajectorySaver(),
                   gp.ScalarSaver(32)]

sim = gp.Simulation(configuration, pair_potential, integrator, runtime_actions,
                    num_timeblocks=16, steps_per_timeblock=128,
                    storage='memory')

for _ in sim.run_timeblocks():
        print(sim.status(per_particle=True))
print(sim.summary())
