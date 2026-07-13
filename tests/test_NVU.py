"""
"""

import gamdpy as gp
import pandas as pd
import numpy as np
import pandas as pd
import pytest
from numba import config
import time
import sys



def test_NVU():
    # Setup pair potential: Kob-Andersen Binary Lennard-Jones System with shifted potential.
    pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
    sig = [[1.00, 0.80],
           [0.80, 0.88]]
    eps = [[1.00, 1.50],
           [1.50, 0.50]]
    cut = np.array(sig)*2.5
    pairpot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

    # Setup configuration
    configuration = gp.Configuration(D=3, compute_flags={'Fsq':True, 'lapU':True, 'Vol':True})
    configuration.make_positions(N=8*8*9*4, rho=1.2)
    configuration['m'] = 1.0 # Specify all masses to unity
    configuration.randomize_velocities(temperature=1.6) # Initial high temperature for randomizing
    configuration.ptype[::5] = 1 # Every fifth particle set to type 1 (4:1 mixture)
    
    # Setup of NVT integrator and simulation, in order to find the average value
    # of the potential energy to be used by the NVU integrator.
    NVT_integrator = gp.integrators.NVT(temperature = 0.7, tau = 0.2, dt = 0.004)
    runtime_actions = [gp.MomentumReset(100),
                       gp.ScalarSaver(2, {'Fsq':True, 'lapU':True}), ]
    NVT_sim = gp.Simulation(configuration, pairpot, NVT_integrator, runtime_actions,
                            num_timeblocks = 8, steps_per_timeblock = 1024,
                            storage = 'memory')

    #Running the NVT simulation
    for timeblock in NVT_sim.run_timeblocks():
        pass 

    #Finding the average potential energy (= U_0) of the run.
    columns = ['U']
    data = np.array(gp.ScalarSaver.extract(NVT_sim.output, columns, per_particle=False, first_block=4))
    df = pd.DataFrame(data.T, columns=columns)
    U_0 = np.mean(df['U'])/configuration.N

    #Setting up the NVU integrator and simulation.
    NVU_integrator = gp.integrators.NVU(U_0 = U_0, dl = 0.03)
    runtime_actions = [gp.MomentumReset(100),
                       gp.ScalarSaver(2, {'Fsq':True, 'lapU':True}), ]
    NVU_sim = gp.Simulation(configuration, pairpot, NVU_integrator, runtime_actions,
                            num_timeblocks = 8, steps_per_timeblock = 1024,
                            storage = 'memory')

    #Running the NVU simulation
    for timeblock in NVU_sim.run_timeblocks():
        pass 

    
    #Calculating the configurational temperature
    columns = ['U', 'lapU', 'Fsq']
    data = np.array(gp.ScalarSaver.extract(NVU_sim.output, columns, per_particle=False, first_block=4))
    df = pd.DataFrame(data.T, columns=columns)
    df['Tconf'] = df['Fsq']/df['lapU']
    Tconf = np.mean(df['Tconf'],axis=0)
    assert 0.67 < Tconf < 0.73, print("ATconf should be around 0.7, but is",
                                      f"{Tconf}. For this test, assertionError",
                                      "arises if Tconf is not in the interval",
                                      "[0.68; 0.72]. Try the test again if Tconf",
                                      "is close to the interval")
    assert np.allclose(np.mean(df['U'])/configuration.N,U_0), print("For this test,",
                                                    "assertionError arises if",
                                                    "the average potential",
                                                    "energy <U> =", 
                                                    f"{np.mean(df['U'])}",
                                                    "is not close enough to the",
                                                    f"set energy: U_0 = {U_0}.\n",
                                                    "The closeness is defined by",
                                                    "np.allclose(<U>,U_0).")



    # Setup configuration:
    configuration = gp.Configuration(D=3, compute_flags={'Fsq':True, 'lapU':True, 'Vol':True})
    configuration.make_positions(N=3*3*3*4, rho=1.2)
    configuration['m'] = 1.0 # Specify all masses to unity
    configuration.randomize_velocities(temperature=1.6) # Initial high temperature for randomizing
    configuration.ptype[::5] = 1 # Every fifth particle set to type 1 (4:1 mixture)
    
    NVT_integrator = gp.integrators.NVT(temperature = 0.7, tau = 0.2, dt = 0.004)
    runtime_actions = [gp.MomentumReset(100),
                        gp.ScalarSaver(2, {'Fsq':True, 'lapU':True}), ]
    
    NVT_sim = gp.Simulation(configuration, pairpot, NVT_integrator, runtime_actions,
                            num_timeblocks = 8, steps_per_timeblock = 4*1024,
                            storage = 'memory')

    #Running the NVT simulation
    for timeblock in NVT_sim.run_timeblocks():
        pass 

    # Finding the average potential energy (= U_0) of the run.
    columns = ['U']
    data = np.array(gp.ScalarSaver.extract(NVT_sim.output, columns, per_particle=False, first_block=4))
    df = pd.DataFrame(data.T, columns=columns)
    U_0 = np.mean(df['U'])/configuration.N

    #Setting up the NVU integrator and simulation.
    NVU_integrator = gp.integrators.NVU(U_0 = U_0, dl = 0.03)
    runtime_actions = [gp.MomentumReset(100),
                        gp.ScalarSaver(2, {'Fsq':True, 'lapU':True}), ]
    NVU_sim = gp.Simulation(configuration, pairpot, NVU_integrator, runtime_actions,
                            num_timeblocks = 8, steps_per_timeblock = 32*1024,
                            storage = 'memory')

    #Running the NVU simulation
    for timeblock in NVU_sim.run_timeblocks():
        pass


    #Calculating the configurational temperature
    columns = ['U', 'lapU', 'Fsq']
    data = np.array(gp.ScalarSaver.extract(NVU_sim.output, columns, per_particle=False, first_block=4))
    df = pd.DataFrame(data.T, columns=columns)
    df['Tconf'] = df['Fsq']/df['lapU']
    Tconf = np.mean(df['Tconf'],axis=0)
    assert 0.67 < Tconf < 0.74, print("BTconf should be around 0.7, but is",
                                      f"{Tconf}. For this test, assertionError",
                                      "arises if Tconf is not in the interval",
                                      "[0.68; 0.72].Try the test again if Tconf",
                                      "is close to the interval")
    assert np.allclose(np.mean(df['U'])/configuration.N,U_0), print("For this test,",
                                                    "assertionError arises if",
                                                    "the average potential",
                                                    "energy <U> =",
                                                    f"{np.mean(df['U'])}",
                                                    "is not close enough to the",
                                                    f"set energy: U_0 = {U_0}.\n",
                                                    "The closeness is defined by",
                                                    "np.allclose(<U>,U_0).")




if __name__ == '__main__':
    config.CUDA_LOW_OCCUPANCY_WARNINGS = False
    test_NVU()
