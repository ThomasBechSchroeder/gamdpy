""" Example of how to calculate the two hydrodynamic correlation functions 
       
       1: the transverse current correlation function (jacf), and 
       2: the longitudinal density correlation function (dacf)

    The system is a simple Lennard-Jones system at the liquid state point.

    The correlation functions are stored in the files jacf.dat and dacf.dat when calling the calculator read method
    1st column is time, remaining columns are correlation function values at the different wavevectors k = 2*pi*n/L 
    where n is the wave number and L the system box length in the x-direction. 

    It is recommended to use a simple cubic box for these calculations.

    """

import gamdpy as gp
import os

# Setup configuration: FCC Lattice
configuration = gp.Configuration(D=3, compute_flags={'W':True})
configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.8)
configuration['m'] = 1.0
configuration.randomize_velocities(temperature=0.9)

# Setup pair potential: Single component 12-6 Lennard-Jones
pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
sig, eps, cut = 1.0, 1.0, 2.5
pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

# Setup integrator: NVT
integrator = gp.integrators.NVT(temperature=0.9, tau=0.2, dt=0.005)

# Setup runtime actions, i.e. actions performed during simulation of timeblocks
runtime_actions = [gp.TrajectorySaver(),
                   gp.ScalarSaver(16, {'W':True}),
                   gp.MomentumReset(100)]

# Setup Simulation. 
sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                    num_timeblocks=2048, steps_per_timeblock=32,
                    storage='memory')

print('Simulation created')

hydrocorr = gp.CalculatorHydrodynamicCorrelations(configuration, dtsample=32*0.005, nwaves=10)

# Run simulation
for _ in sim.run_timeblocks():
        hydrocorr.update()

print(sim.summary())

hydrocorr.read()

