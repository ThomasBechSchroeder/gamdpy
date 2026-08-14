""" Simulation of Active Browinian Partivles """

import gamdpy as gp


#Set up the initial configuration with 2 particle types 
configuration = gp.Configuration(D=3, compute_flags={'stresses': True, 'Vol': True, 'orientations':True})
configuration.make_lattice(unit_cell=gp.unit_cells.FCC, cells=[10, 5, 5], rho=0.6)


configuration['m'] = 1.0

configuration.ptype[configuration['r'][:, 2]< 0] = 0   # species A
configuration.ptype[configuration['r'][:, 2]> 0] = 1    # species B


configuration.randomize_velocities(temperature= 2.0)
configuration.randomize_orientations()

dt = 0.005
sig = [[1.00, 1.00],
       [1.00, 1.00]]
eps = [[1.00, 0.5],
       [0.5, 1.00]]
cut_mult = 2.5
cut = [[sig[i][j] * cut_mult for j in range(2)] for i in range(2)]

pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
pair_pot  = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)



# Set DA=0.0 for passive particles (for comparison)
integrator = gp.integrators.ActiveBP(DT=[0.05, 0.05], DR=[1.0, 1.0], mu=[0.05, 0.05], v0=[0.5, 0.0] , dt=dt, seed=2028)

runtime_actions = [gp.RestartSaver(),
                   gp.TrajectorySaver(),
                   gp.ScalarSaver(16)]

sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=8, steps_per_timeblock=8*1024,
                    storage='Data/ActiveABP.h5')

for block in sim.run_timeblocks():
    print(f'{sim.status(per_particle=True)}')
print(sim.summary())
