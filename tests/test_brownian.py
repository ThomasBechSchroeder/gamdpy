""" Test Brownian thermostat
Investigate temperature T=1.2 (r_c=2.5) fcc-liquid coexistence state-point in https://doi.org/10.1063/1.4818747 """

import numpy as np
import gamdpy as gp

def test_brownain_interface():
    # Test positional arguments
    integrator = gp.Brownian(1.0, 0.1, 0.005, 2025)
    assert integrator.temperature == 1.0
    assert integrator.tau == 0.1
    assert integrator.dt == 0.005
    assert integrator.seed == 2025

    # Test keyword arguments
    temperature = 1.0
    tau = 0.1
    dt = 0.005
    seed = 2025
    integrator = gp.Brownian(temperature=temperature, tau=tau, dt=dt, seed=seed)
    assert integrator.temperature == temperature
    assert integrator.tau == tau
    assert integrator.dt == dt
    assert integrator.seed == seed

    # Test other inteface
    integrator = gp.integrators.Brownian(1.0, 0.1, 0.005, 2025)
    assert integrator.temperature == 1.0
    assert integrator.tau == 0.1
    assert integrator.dt == 0.005
    assert integrator.seed == 2025

def test_brownain_simulation(verbose=False, plot=False):
    # State-point
    temperature = 1.2
    density = 1 / 0.9672

    # Expected values
    expected_kinetic_energy = 3 / 2 * temperature
    expected_total_energy = -4.020
    expected_potential_energy = expected_total_energy - expected_kinetic_energy
    if verbose:
        print(f'Expected potential energy: {expected_potential_energy}')

    # Setup configuration (give temperature kick to particles to get closer to equilibrium)
    configuration = gp.Configuration(D=3, compute_flags={'W':True, 'K':True})
    configuration.make_lattice(gp.unit_cells.FCC, cells=[7, 7, 7], rho=density)
    configuration['m'] = 1.0

    # Setup pair potential.
    pairfunc = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
    sig, eps, cut = 1.0, 1.0, 2.5
    pairpot = gp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

    # Setup integrator
    dt = 0.005
    tau = 0.1
    integrator = gp.integrators.Brownian(temperature=temperature, tau=tau, dt=dt, seed=2025)
    runtime_actions = [gp.MomentumReset(100),
                       gp.ScalarSaver(16, {'W':True, 'K':True}), ]
    sim = gp.Simulation(configuration, [pairpot, ], integrator, runtime_actions,
                        num_timeblocks=16, steps_per_timeblock=512,
                        storage='memory')

    for _ in sim.run_timeblocks():
        if verbose:
            print(sim.status(per_particle=True))
    if verbose:
        print(sim.summary())

    N = configuration.N
    U, W, K = gp.extract_scalars(sim.output, ['U', 'W', 'K'], first_block=1)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(U/N)
        plt.hlines(expected_potential_energy, 0, len(U), colors='r')
        plt.show()

    assert np.allclose(np.mean(U/N), expected_potential_energy, atol=0.01), f"{np.mean(U/N):.3f} expected {expected_potential_energy:.3f}"

if __name__ == '__main__':
    test_brownain_interface()
    test_brownain_simulation(verbose=True, plot=True)
