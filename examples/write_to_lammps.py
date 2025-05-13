""" Example of writing a configuration to a LAMMPS dump file. """
import os

import gamdpy as gp

# Setup configuration: FCC Lattice
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=0.7)

# Write initial configuration to LAMMPS dump file
lmp_dump = gp.configuration_to_lammps(configuration)
print(lmp_dump, file=open('dump.initial', 'w'))

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator: NVT
integrator = gp.integrators.NVT(temperature=0.7, tau=0.2, dt=0.005)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(),
                   gp.MomentumReset(100)]


# Setup Simulation.
sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=100, steps_per_timeblock=1000,
                    storage='memory')

# Delete old dump file if it exists
dump_filename = 'dump.lammps'
if os.path.exists(dump_filename):
    os.remove(dump_filename)

# Run simulation and write configuration to LAMMPS dump file on the fly
for block in sim.run_timeblocks():
    lmp_dump = gp.configuration_to_lammps(sim.configuration, timestep=sim.steps_per_block*block)
    print(lmp_dump, file=open(dump_filename, 'a'))

# Open dump file in ovito with
#   ovito dump.lammps
#
# Open in VMD with
#   vmd -lammpstrj dump.lammps
