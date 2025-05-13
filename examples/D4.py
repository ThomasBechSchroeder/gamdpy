import matplotlib.pyplot as plt
import numpy as np
import gamdpy as gp
import math

temperature, density, npart, D = 3.40410425853, 0.34, 2401, 4

configuration = gp.Configuration(D=D)
# Setup configuration. SC Lattice in 4D
configuration.make_positions(N=npart, rho=density)
# Setup masses and velocities
configuration['m'] = 1.0  # Set all masses to 1.0
configuration.randomize_velocities(temperature=temperature)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator
integrator = gp.integrators.NVT(temperature=temperature, tau=0.08, dt=0.001)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(),
                   gp.MomentumReset(100)]


# Setup Simulation
sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=16, steps_per_timeblock=4096,
                    storage='memory')

# Run simulation
sim.run(verbose=False)
sim.run(verbose=False)

U, W, K = gp.extract_scalars(sim.output, ['U', 'W', 'K'], first_block=1)
dU = U - np.mean(U)
dW = W - np.mean(W)
gamma = np.dot(dW,dU)/np.dot(dU,dU)
R = np.dot(dW,dU)/(np.dot(dW,dW)*np.dot(dU,dU))**0.5
print(f'{density=} {temperature=} {R=:.3f} {gamma=:.3f}')
# Reference results are from http://glass.ruc.dk/pdf/articles/2016_JChemPhys_144_231101.pdf
assert math.isclose(0.80, R    , rel_tol=0.1), f"{R=} but should be 0.80"
assert math.isclose(3.33, gamma, rel_tol=0.1), f"{gamma=} but should be 3.33"

