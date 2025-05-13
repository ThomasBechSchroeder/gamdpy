""" Minimal example of a Simulation using gamdpy on CPU.

Simulation of a Lennard-Jones crystal in the NVT ensemble.
This simulation uses the numba CUDA simulator to run on CPU:
https://numba.pydata.org/numba-doc/dev/cuda/simulator.html

"""
import os
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["NUMBA_CUDA_DEBUGINFO"] = "1"

import gamdpy as gp

# Setup fcc configuration
configuration = gp.Configuration(D=3)
configuration.make_lattice(gp.unit_cells.FCC, cells=[4, 4, 4], rho=0.973)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=0.7)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator: NVT
integrator = gp.integrators.NVT(temperature=0.7, tau=0.2, dt=0.005)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(2),
                   gp.MomentumReset(100)]

# Setup Simulation.
sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=2, steps_per_timeblock=4,
                    storage='LJ_T0.70.h5', timing=False)

# Run simulation
sim.run()

# To get a plot of the MSD do something like this:
# python -m gamdpy.tools.calc_dynamics -f 4 -o msd.pdf LJ_T*.h5

# Disable settings for the CUDA simulator
os.environ["NUMBA_ENABLE_CUDASIM"] = "0"
os.environ["NUMBA_DISABLE_JIT"] = "0"
os.environ["NUMBA_CUDA_DEBUGINFO"] = "0"

