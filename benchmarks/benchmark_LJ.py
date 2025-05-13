""" 
Benchmark rumdpy using the lammps LJ benchmark
Command line options:

- Nsquared : Use O(N^2) for neighbor-list update (Default)
- LinkedLists : Use O(N) linked lists for neighbor-list update

- NVE : Use NVE integrator (default)
- NVT : Use NVT integrator
- NVT_Langevin : Use NVT_Langevin integrator

"""

import glob
import sys
# from rumdpy.integrators import nve, nvt_nh, nvt_langevin  # OLD CODE

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numba
from numba import cuda, config

import rumdpy as gp
import pickle


def setup_lennard_jones_system(nx, ny, nz, rho=0.8442, cut=2.5, verbose=False):
    """
    Setup and return configuration, potential function, and potential parameters for the LJ benchmark
    """

    # Generate configuration with a FCC lattice
    # Setup configuration: FCC Lattice
    c1 = gp.Configuration(D=3)
    c1.make_lattice(rp.unit_cells.FCC, cells=[nx, ny, nz], rho=rho)
    c1['m'] = 1.0
    c1.randomize_velocities(temperature=1.44)
    #  c1 = gp.make_configuration_fcc(nx=nx,  ny=ny,  nz=nz,  rho=rho, T=1.44)

    # Setup pair potential.
    #pair_func = gp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
    pair_func = gp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
    sig, eps, cut = 1.0, 1.0, 2.5
    pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=500)

    return c1, pair_pot


def run_benchmark(c1, pair_pot, compute_plan, steps, integrator='NVE', autotune=False, verbose=False):
    """
    Run LJ benchmark
    Could be run with other potential and/or parameters, but asserts would need to be updated
    """

    # Set up the integrator
    dt = 0.005

    if integrator == 'NVE':
        integrator = gp.integrators.NVE(dt=dt)
    if integrator == 'NVT':
        integrator = gp.integrators.NVT(temperature=0.70, tau=0.2, dt=dt)
    if integrator == 'NVT_Langevin':
        integrator = gp.integrators.NVT_Langevin(temperature=0.70, alpha=0.2, dt=dt, seed=213)
    
    # Setup Simulation. Total number of timesteps: num_blocks * steps_per_block
    sim = gp.Simulation(c1, pair_pot, integrator, [rp.MomentumReset(100), ],
                        num_timeblocks=1, steps_per_timeblock=steps,
                        compute_plan=compute_plan, storage='memory', verbose=False)
    
    # Run simulation one block at a time
    for block in sim.run_timeblocks():
        pass
    
    if autotune:
        #sim.autotune_bruteforce(verbose=False)
        sim.autotune()

    nbflag0 = pair_pot.nblist.d_nbflag.copy_to_host()
    for block in sim.run_timeblocks():
        pass

    #print(sim.summary())

    #assert 0.55 < Tkin < 0.85, f'{Tkin=}'
    #assert 0.55 < Tconf < 0.85, f'{Tconf=}'
    #if integrator == 'NVE':  # Only expect conservation of energy if we are running NVE
    #    assert -0.01 < de < 0.01
    
    tps = sim.last_num_blocks * sim.steps_per_block / np.sum(sim.timing_numba_blocks) * 1000
    time_in_sec = sim.timing_numba / 1000

    nbflag = pair_pot.nblist.d_nbflag.copy_to_host()
    nbupdates = nbflag[2] - nbflag0[2]
    print(f"{c1.N:7} {tps:.2e} {steps:.1e} {time_in_sec:.1e}  {nbupdates:6} {steps/nbupdates:.1f} {sim.compute_plan}")
    assert nbflag[0] == 0
    assert nbflag[1] == 0
    return tps, time_in_sec, steps, sim.compute_plan.copy()


def main(integrator, nblist, identifier, autotune):
    config.CUDA_LOW_OCCUPANCY_WARNINGS = False
    print(f'Benchmarking LJ with {integrator} integrator:')
    nxyzs = (
        (8, 8, 8), (8, 8, 16), (8, 16, 16), (16, 16, 16), (16, 16, 32), (16, 32, 32), (32, 32, 32) )
    if nblist == 'Nsquared':
        nxyzs = ((4, 4, 8), (4, 8, 8),) + nxyzs
    if nblist == 'LinkedLists':
        nxyzs += (32, 32, 64), (32, 64, 64), (64, 64, 64)
    if nblist == 'default':
        nxyzs = ((4, 4, 8), (4, 8, 8),) + nxyzs
        nxyzs += (32, 32, 64), (32, 64, 64), (64, 64, 64)
    #nxyzs = ( (4, 4, 8), (4, 8, 8) ) # For quick debuging
    #nxyzs = ( (32, 32, 32), ) # For quick debuging
    Ns = []
    tpss = []
    tpss_at = []
    compute_plans = []
    compute_plans_at = []
    target_time_in_sec = 0.50 # At least this time to get reliable timing
    magic_number = 1e7
    print('    N     TPS     Steps   Time     NbUpd Steps/NbUpd')
    for nxyz in nxyzs:
        c1, LJ_func = setup_lennard_jones_system(*nxyz, cut=2.5, verbose=False)
        time_in_sec = 0
        while time_in_sec < target_time_in_sec:  
            steps = int(magic_number / c1.N)
            compute_plan = gp.get_default_compute_plan(c1)
            tps, time_in_sec, steps, compute_plan = run_benchmark(c1, LJ_func, compute_plan, steps, integrator=integrator, verbose=False)
            magic_number *= (2*target_time_in_sec) / time_in_sec  # Aim for 2*target_time (Assuming O(N) scaling)
        Ns.append(c1.N)
        tpss.append(tps)
        compute_plans.append(compute_plan)
        
        if autotune:
            tps_at, time_in_sec_at, steps_at, compute_plan_at = run_benchmark(c1, LJ_func, compute_plan, steps, integrator=integrator, autotune=autotune, verbose=False)
            tpss_at.append(tps_at)
            compute_plans_at.append(compute_plan_at)
    
    # Save this run to csv file
    if autotune:
        df = pd.DataFrame({'N': Ns, 'TPS': tpss, 'TPS_AT':tpss_at})
        data = {'N': Ns, 'TPS': tpss, 'TPS_AT':tpss_at, 'compute_plans':compute_plans, 'compute_plans_at':compute_plans_at}
    else:
        df = pd.DataFrame({'N': Ns, 'TPS': tpss}, )
        data = {'N': Ns, 'TPS': tpss, 'compute_plans':compute_plans,}
    
    df.to_csv(f'Data/benchmark_LJ_{identifier}.csv', index=False)
    with open(f'Data/benchmark_LJ_{identifier}.pkl', 'wb') as file:  
        pickle.dump(data, file) 

if __name__ == "__main__":
    integrator = 'NVE'
    
    identifier = sys.argv[1]

    if 'NVT' in sys.argv:
        integrator = 'NVT'
    if 'NVT_Langevin' in sys.argv:
        integrator = 'NVT_Langevin'

    nblist = 'default'    
    if 'LinkedLists' in sys.argv:
        nblist = 'LinkedLists'
    if 'NSquared' in sys.argv:
        nblist = 'NSquared'

    autotune = 'autotune' in sys.argv

    main(integrator=integrator, nblist=nblist, identifier=identifier, autotune=autotune)
