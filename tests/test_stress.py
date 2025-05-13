""" Example of a Simulation using gamdpy, including the stress tensor (configurational part only).

"""

import gamdpy as gp
import numpy as np

# Setup configuration: FCC Lattice
rho = 0.973
configuration = gp.Configuration(D=3, compute_flags={'stresses':True})
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=rho)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=0.8 * 2)

# Setup pair potential: Single component 12-6 Lennard-Jones
pairfunc = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pairpot = gp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator: NVT
integrator = gp.integrators.NVT(temperature=0.70, tau=0.2, dt=0.005)

num_timeblocks = 4

runtime_actions = [gp.MomentumReset(100), 
                   gp.TrajectorySaver(), 
                   gp.ScalarSaver(32, {'stresses':True}), ]

# Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
sim = gp.Simulation(configuration, pairpot, integrator, runtime_actions,
                    num_timeblocks=num_timeblocks, steps_per_timeblock=128,
                    storage='memory',)

p_conf_array = np.zeros(num_timeblocks)
scalar_stress_array = np.zeros(num_timeblocks)

# Run simulation one block at a time
for block in sim.run_timeblocks():
    print(sim.status(per_particle=True))
    vol = configuration.get_volume()
    
    sts_x_row = (np.sum(sim.configuration['sx'], axis=0)/vol)
    sts_y_row = (np.sum(sim.configuration['sy'], axis=0)/vol)
    sts_z_row = (np.sum(sim.configuration['sz'], axis=0)/vol)
    #print('Stress tensor (configurational part)')
    mean_diagonal = (sts_x_row[0] + sts_y_row[1] + sts_z_row[2])/3
    virial = np.mean(configuration['W'])
    scalar_stress_array[block] = mean_diagonal
    p_conf_array[block] = virial * rho

assert np.allclose(p_conf_array, -scalar_stress_array)
