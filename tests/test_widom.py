""" Test Widom's particle insertion method """

import numpy as np
import matplotlib.pyplot as plt

import gamdpy as gp

def test_widom_insertion():
    np.random.seed(0)

    # Setup configuration: FCC Lattice
    configuration = gp.Configuration(D=3)
    configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.4)
    configuration['m'] = 1.0
    configuration.randomize_velocities(temperature=2.0, seed=0)

    # Setup pair potential: Single component 12-6 Lennard-Jones
    pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
    sig, eps, cut = 1.0, 1.0, 2.5
    pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

    # Setup integrator: NVT
    temperature = 1.0
    integrator = gp.integrators.NVT(temperature=1.0, tau=0.2, dt=0.0)  # dummy dt

    # Setup runtime actions, i.e. actions performed during simulation of timeblocks
    runtime_actions = [gp.TrajectorySaver(), 
                       gp.ScalarSaver(), 
                       gp.MomentumReset(16)]

    # Setup Simulation
    sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                        num_timeblocks=2, steps_per_timeblock=32,
                        storage='memory')

    # Setup the Widom's particle insertion calculator
    num_ghost_particles = 500_000
    ghost_positions = np.random.rand(num_ghost_particles, configuration.D) * configuration.simbox.lengths
    calc_widom = gp.CalculatorWidomInsertion(sim.configuration, pair_pot, temperature, ghost_positions)
    print('Production run')
    for block in sim.run_timeblocks():
        calc_widom.update()

    calc_widom_data = calc_widom.read()

    # Test if the excess chemical potential in in expected range
    mu = calc_widom_data['chemical_potential']
    print(f"Excess chemical potential: {mu}")
    mu_expected = 0.4469328390273799
    print(f"Expected excess chemical potential: {mu_expected}")
    print(f"Error: {mu - mu_expected}")
    my_tol = 1e-4
    assert np.isclose(mu, mu_expected, rtol=my_tol), f"mu = {mu}, mu_expected = {mu_expected}"

if __name__ == '__main__':  # pragma: no cover
    test_widom_insertion()
