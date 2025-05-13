""" Simulate a system of asymmetric dumbbells (ASD) with a Lennard-Jones potential and a harmonic bond potential. """

import numpy as np
import gamdpy as gp

# Specify state point
rho = 1.863  # Atomic density = Molecular density * 2
temperature = 0.465
filename = f'Data/ASD_rho{rho:.3f}_T{temperature:.3f}.h5'

# Generate two-component configuration with a FCC lattice 
configuration = gp.Configuration(D=3, compute_flags={'Fsq':True, 'lapU':True, 'Ptot':True, 'Vol':True})
configuration.make_lattice(gp.unit_cells.FCC, cells=[6, 6, 6], rho=rho)
configuration['m'] = 1.0
B_particles = range(1, configuration.N, 2)
configuration.ptype[B_particles] = 1  # Setting particle type of B particles
configuration['m'][B_particles] = 0.195  # Setting masses of B particles
configuration.randomize_velocities(temperature=1.44)

# Make bonds
bond_potential = gp.harmonic_bond_function
bond_params = [[0.584, 3000.], ]  # Parameters for bond type 0, 1, 2 etc (here only 0)
bond_indices = [[i, i + 1, 0] for i in range(0, configuration.N - 1, 2)]  # dumbells: i(even) and i+1 bonded with type 0
bonds = gp.Bonds(bond_potential, bond_params, bond_indices)

# Make pair potential
# pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig = [[1.000, 0.894],
       [0.894, 0.788]]
eps = [[1.000, 0.342],
       [0.342, 0.117]]
cut = np.array(sig) * 2.5
exclusions = bonds.get_exclusions(configuration)
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], exclusions=exclusions, max_num_nbs=1000)

# Make integrator
dt = 0.002  # timestep
num_blocks = 64  # Do simulation in this many 'blocks'
steps_per_block = 1*1024  # ... each of this many steps (increase for better statistics)
running_time = dt * num_blocks * steps_per_block

Ttarget_function = gp.make_function_ramp(value0=10.000, x0=running_time * (1 / 8),
                                         value1=temperature, x1=running_time * (1 / 4))
integrator0 = gp.integrators.NVT(Ttarget_function, tau=0.2, dt=dt)

sim = gp.Simulation(configuration, [pair_pot, bonds], integrator0,
                    runtime_actions=[gp.MomentumReset(100)],
                    num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                    storage='memory')

print('High Temperature followed by cooling and equilibration:')
for block in sim.run_timeblocks():
    if block % 10 == 0:
        print(sim.status(per_particle=True))
print(sim.summary())

print('Production:')
integrator = gp.integrators.NVT(temperature=temperature, tau=0.2, dt=dt)

runtime_actions = [gp.MomentumReset(100), 
                   gp.TrajectorySaver(), 
                   gp.ScalarSaver(32, {'Fsq':True, 'lapU':True, 'Ptot':True}), ]

sim = gp.Simulation(configuration, [pair_pot, bonds], integrator, runtime_actions, 
                    num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                    storage=filename)

for block in sim.run_timeblocks():
    if block % 10 == 0:
        print(f'{block=:4}  {sim.status(per_particle=True)}')
print(sim.summary())
