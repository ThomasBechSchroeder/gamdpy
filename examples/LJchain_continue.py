""" Continue a simulation of LJ chains from a previous run. """

import h5py

import gamdpy as gp

# Parameters
h5file = h5py.File('Data/LJchain_toread.h5')
restart_index = 31
temperature = 0.7

# Load configuration from the HDF5 file
group_name = f'restarts/restart{restart_index:04d}'
configuration = gp.Configuration.from_h5(
    h5file,
    group_name,
    compute_flags={'Fsq':True, 'lapU':True},
    reset_images=True,
    include_topology=True
)

# Make bond interactions
bond_potential = gp.harmonic_bond_function
bond_params = [[1.00, 3000.], ]
bonds = gp.Bonds(bond_potential, configuration.topology.bonds, bond_params)

# Make pair potential
pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
exclusions = bonds.get_exclusions(configuration)
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], exclusions=exclusions, max_num_nbs=1000)

# Make integrator
dt = 0.002
num_blocks = 32
steps_per_block = 1 * 1024
running_time = dt * num_blocks * steps_per_block
integrator = gp.integrators.NVT(temperature=temperature, tau=0.2, dt=dt)

# Make a simulation object
runtime_actions = [gp.MomentumReset(steps_between_reset=100),
                   gp.RestartSaver(),
                   gp.TrajectorySaver(),
                   gp.ScalarSaver(steps_between_output=32, compute_flags={'Fsq':True, 'lapU':True}), ]
output_file = 'Data/LJchain10_Rho1.00_T0.700_c1.h5'
sim = gp.Simulation(configuration, [pair_pot, bonds], integrator, runtime_actions,
                    num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                    storage=output_file)

# Run simulation
for block in sim.run_timeblocks():
    if block % 10 == 0:
        print(f'{block=:4}  {sim.status(per_particle=True)}')
print(sim.summary())
