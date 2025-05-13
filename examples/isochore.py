""" Simple example of performing several simulation in one go using gamdpy.

Simulation of heating a Lennard-Jones crystal on an isochore in the NVT ensemble.
For an even simpler script, see minimal.py

"""

import gamdpy as gp

# Setup fcc configuration
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[6, 6, 6], rho=0.973)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=1.6)

# Setup pair potential.
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Specify duration of simulations. 
# Increase 'num_timelocks' for longer runs, better statistics, AND larger storage consumption
# Increase 'steps_per_block' for longer runs
dt = 0.004  # timestep
num_timeblocks = 8            # Do simulation in this many 'timeblocks'. 
steps_per_timeblock = 1*1024  # ... each of this many steps



for temperature in ['0.70', '1.10', '1.50']:
    print('\n\nTemperature: ' + temperature)
    
    # Setup integrator
    integrator = gp.integrators.NVT(temperature=temperature, tau=0.2, dt=0.005)

    # Setup runtime actions, i.e. actions performed during simulation of timeblocks
    runtime_actions = [gp.TrajectorySaver(),
                    gp.ScalarSaver(),
                    gp.MomentumReset(100)]

    # Setup Simulation
    sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions, 
                        num_timeblocks=num_timeblocks, steps_per_timeblock=steps_per_timeblock,
                        storage='Data/LJ_r0.973_T'+temperature+'.h5')

    print('Equilibration:')
    for block in sim.run_timeblocks():
        print(sim.status(per_particle=True))
    print(sim.summary())
    
    print('Production:')
    for block in sim.run_timeblocks():
        print(sim.status(per_particle=True))
    print(sim.summary())

# To get a plot of the MSD do something like this:
# python3 -m rumdpy.tools.calc_dynamics -o Data/msd_r0.973.pdf Data/LJ_r0.973_T*.h5
