""" Test Active Ornstein-Uhlenbeck Particle
The integrator has been tested with an analytical result for the simple one-dimensional case without interactions, source:https://doi.org/10.1103/PhysRevLett.113.238303.
Further test could be done with some published benchmark result"""

import numpy as np
import gamdpy as gp

def test_AOUP_interface():
    # Test positional arguments
    integrator = gp.ActiveOUP(0.25, 0.5, 1.0, 0.75, 0.001, 2025)
    assert integrator.DT == 0.25
    assert integrator.DA == 0.5
    assert integrator.mu == 1.0
    assert integrator.tau == 0.75
    assert integrator.dt == 0.001
    assert integrator.seed == 2025

    # Test keyword arguments
    DT=0.25
    DA=0.5
    mu=1.0
    tau=0.75
    dt=0.001
    seed=2025
    integrator = gp.ActiveOUP(DT=DT, DA=DA, mu=mu, tau=tau, dt=dt, seed=seed)
    assert integrator.DT == DT
    assert integrator.DA == DA
    assert integrator.mu == mu
    assert integrator.tau == tau
    assert integrator.dt == dt
    assert integrator.seed == seed

    # Test other interface
    integrator = gp.integrators.ActiveOUP(DT=0.25, DA=0.5, mu=1.0, tau=0.75, dt=0.001, seed=2025)
    assert integrator.DT == 0.25
    assert integrator.DA == 0.5
    assert integrator.mu == 1.0
    assert integrator.tau == 0.75
    assert integrator.dt == 0.001
    assert integrator.seed == 2025

def test_AOUP_simulation(verbose=False, plot=False):
    # State-point
    density = 0.01

    # Setup configuration 
    configuration = gp.Configuration(D=1)
    configuration.make_positions(N=2048, rho=density)

    # Setup pair potential.
    pairfunc = gp.harmonic_repulsion
    eps, sig = 0.00001, 1.0
    pairpot = gp.PairPotential(pairfunc, params=[eps, sig], max_num_nbs=1000)
    interactions = [pairpot, ]

    # Setup integrator
    DT=0.25
    DA=0.5
    mu=1.0
    tau=0.75
    dt=0.001

    integrator = gp.integrators.ActiveOUP(DT=0.25, DA=0.5, mu=1.0, tau=0.75, dt=0.001, seed=25)
    runtime_actions = [
        gp.RestartSaver(),
        gp.TrajectorySaver(),
        gp.MomentumReset(steps_between_reset=100),
        gp.ScalarSaver(16, {'W' :True, 'K' :True}),
    ]

    sim = gp.Simulation(configuration, interactions, integrator, runtime_actions,
                        num_timeblocks=16,
                        steps_per_timeblock=512,
                        storage='memory')
    
    #equilibration
    for block in sim.run_timeblocks():
        if verbose:
            print(sim.status(per_particle=True))
    if verbose:
        print(sim.summary())

    #production
    for _ in sim.run_timeblocks():
        if verbose:
            print(sim.status(per_particle=True))
    if verbose:
        print(sim.summary())
    
    dynamics = gp.tools.calc_dynamics(sim.output, first_block=0,qvalues=1.0)
    
    #expected values of msd
    t = dynamics['times']
    expected_msd = 2*DT*t+2*DA*(t-tau*(1-np.exp(-t/tau)))

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 9))
        plt.plot(np.log(dynamics['times']), np.log(dynamics['msd']))
        plt.plot(np.log(t), np.log(expected_msd), linestyle='dashed')
        plt.xlabel('t')
        plt.ylabel('MSD')
        plt.title('MSD 1D AOUP dynamics')
        plt.grid(True)
        plt.tight_layout()     
        plt.show()

    # print(np.log(dynamics['msd']).flatten())
    # print(np.log(expected_msd).flatten())
    # diff = np.log(dynamics['msd']).flatten() - np.log(expected_msd).flatten()
    # print(diff)
    assert np.allclose(np.log(dynamics['msd']).flatten(), np.log(expected_msd).flatten(), atol=0.1), f"simulated msd is not close to the analytical result"

if __name__ == '__main__':
    test_AOUP_interface()
    test_AOUP_simulation(verbose=True, plot=True)
