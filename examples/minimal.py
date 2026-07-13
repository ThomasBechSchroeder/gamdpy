""" Minimal example of running a simulation using gamdpy

Simulation of a Lennard-Jones crystal in the NVT ensemble.

"""

import gamdpy as gp

# Setup configuration: FCC Lattice
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=0.7)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator: NVT
integrator = gp.integrators.NVT(temperature=0.7, tau=0.2, dt=0.005)

# Setup runtime actions, i.e., actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(16),
                   gp.RestartSaver(),
                   gp.MomentumReset(100)]

# Setup Simulation. 
sim = gp.Simulation(configuration, [pair_pot], integrator, runtime_actions,
                    num_timeblocks=32, steps_per_timeblock=1*1024,
                    storage='LJ_T0.70.h5')

# Run simulation
for timeblock in sim.run_timeblocks():
        print(sim.status(per_particle=True))
print(sim.summary())

# Print current status of configuration
print(configuration)

print('\nAnalyse the saved simulation with scripts found in "examples"')
print('(visualize requires that ovito is installed):')
print('   python3 analyze_structure.py LJ_T0.70')
print('   python3 analyze_dynamics.py LJ_T0.70')
print('   python3 analyze_thermodynamics.py LJ_T0.70')
print('   python3 visualize.py LJ_T0.70.h5')
