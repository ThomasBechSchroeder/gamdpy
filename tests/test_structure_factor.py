import numpy as np
import matplotlib.pyplot as plt

import gamdpy as gp
from object_lib import pairpot_LJ as pair_potential

def test_structure_factor():
    # verbose = False
    # plot = False

    # Setup simulation of single-component Lennard-Jones liquid
    temperature = 2.0
    configuration = gp.Configuration(D=3)
    configuration.make_lattice(gp.unit_cells.FCC, [8, 8, 8], rho=0.973)
    configuration['m'] = 1.0
    configuration.randomize_velocities(temperature=temperature * 2)

    integrator = gp.integrators.NVT(temperature=temperature, tau=0.2, dt=0.005)

    runtime_actions = [ gp.TrajectorySaver(),
                        gp.ScalarSaver(), 
                        gp.MomentumReset(100)]

    sim = gp.Simulation(configuration, pair_potential, integrator, runtime_actions,
                        steps_per_timeblock=1024, num_timeblocks=16, 
                        storage='memory')

    # if verbose:
    #     print('Equilibrating...')
    # for _ in sim.run_timeblocks():
    #     if verbose:
    #         print(sim.status(per_particle=True))
    #
    # if verbose:
    #     print('Calculating structure factor in production run ...')

    q_max: float = 16.0
    calc_struct_fact = gp.CalculatorStructureFactor(configuration)
    calc_struct_fact.generate_q_vectors(q_max=q_max)
    for _ in sim.run_timeblocks():
        calc_struct_fact.update()
        # if verbose:
        #     print(sim.status(per_particle=True))

    # Read the (binned) structure factor
    # if verbose:
    #     print('Testing binned data output ...')
    struc_fact = calc_struct_fact.read(bins=128)

    # Assert that the structure factor is a dictionary
    assert isinstance(struc_fact, dict)
    # Assert that the dictionary contains the keys '|q|' and 'S(|q|)'
    assert '|q|' in struc_fact, f"struc_fact.keys() = {struc_fact.keys()}"
    assert 'S(|q|)' in struc_fact, f"struc_fact.keys() = {struc_fact.keys()}"
    # Assert that the values of the dictionary are numpy arrays
    assert isinstance(struc_fact['|q|'], type(np.array([])))
    assert isinstance(struc_fact['S(|q|)'], type(np.array([])))
    # Assert that the length of the arrays are the same
    assert len(struc_fact['|q|']) == len(struc_fact['S(|q|)'])
    # Assert that q values are between 0 and max_q
    assert np.all(struc_fact['|q|'] >= 0)
    assert np.all(struc_fact['|q|'] <= q_max)
    # Assert that max S_q is between 2 and 3
    assert np.max(struc_fact['S(|q|)']) < 3, f"max(S(q)) = {np.max(struc_fact['S(|q|)'])}"
    assert np.max(struc_fact['S(|q|)']) > 2, f"max(S(q)) = {np.max(struc_fact['S(|q|)'])}"
    # Assert that min S_q is between 0 and 1
    assert np.min(struc_fact['S(|q|)']) < 1, f"min(S(q)) = {np.min(struc_fact['S(|q|)'])}"
    assert np.min(struc_fact['S(|q|)']) >= 0, f"min(S(q)) = {np.min(struc_fact['S(|q|)'])}"

    # if verbose:
    #     # Print shape of the arrays
    #     for key in struc_fact.keys():
    #         print(f"Shape of {key}: {struc_fact[key].shape}")
    #     print(f"Max of S(q): {np.max(struc_fact['S(|q|)'])}")
    #     print(f"Length of q at max of S(q): {struc_fact['|q|'][np.argmax(struc_fact['S(|q|)'])]}")
    #
    # if plot:
    #     plt.figure()
    #     plt.title('Structure factor')
    #     plt.plot(struc_fact['|q|'], struc_fact['S(|q|)'], 'o--')
    #     plt.xlabel('|q|')
    #     plt.ylabel('S(|q|)')
    #     plt.xlim(0, None)
    #     plt.ylim(0, None)
    #     plt.show()
    #
    # if verbose:
    #     print('Testing raw data output ...')

    struc_fact_raw = calc_struct_fact.read(bins=None)
    # Assert that the output is a dictionary
    assert isinstance(struc_fact_raw, dict)
    # Assert that the dictionary contains the keys 'q', '|q|', 'S(|q|)', 'rho_q', 'n_vectors'
    assert 'q' in struc_fact_raw, f"struc_fact_raw.keys() = {struc_fact_raw.keys()}"
    assert '|q|' in struc_fact_raw, f"struc_fact_raw.keys() = {struc_fact_raw.keys()}"
    assert 'S(q)' in struc_fact_raw, f"struc_fact_raw.keys() = {struc_fact_raw.keys()}"
    assert 'rho_q' in struc_fact_raw, f"struc_fact_raw.keys() = {struc_fact_raw.keys()}"
    assert 'n_vectors' in struc_fact_raw, f"struc_fact_raw.keys() = {struc_fact_raw.keys()}"
    # Assert that the values of the dictionary are numpy arrays
    assert isinstance(struc_fact_raw['q'], type(np.array([])))
    assert isinstance(struc_fact_raw['|q|'], type(np.array([])))
    assert isinstance(struc_fact_raw['S(q)'], type(np.array([])))
    assert isinstance(struc_fact_raw['rho_q'], type(np.array([])))
    assert isinstance(struc_fact_raw['n_vectors'], type(np.array([])))
    # Assert that the length of the arrays are the same

    # if verbose:
    #     # Print dimensions of the arrays
    #     for key in struc_fact_raw.keys():
    #         print(f"Shape of {key}: {struc_fact_raw[key].shape}")
    #
    # if plot:
    #     # Plot image of 2D S(q) for q_z = 0
    #     q: np.ndarray = struc_fact_raw['q']
    #     S_q: np.ndarray = struc_fact_raw['S(q)']
    #     # Select q_vectors where q_z = 0
    #     mask = np.isclose(q[:, 2], 0)
    #     q = q[mask]
    #     q_x = q[:, 0]
    #     q_y = q[:, 1]
    #     S_q = S_q[mask]
    #     # Plot image of 2D S(q)
    #     plt.figure()
    #     plt.title('Structure factor')
    #     plt.scatter(q_x, q_y, c=S_q, s=50, alpha=0.8, cmap='viridis')
    #     plt.xlabel('q_x')
    #     plt.ylabel('q_y')
    #     plt.colorbar()
    #     plt.show()
    #
    # if verbose:
    #     print('All tests passed successfully')

def test_structure_factor_backends():
    # Test structure factor calculation with different backends
    backends = ['CPU multi core', 'CPU single core']
    sim = gp.get_default_sim()
    for backend in backends:
        print(f'Testing backend: {backend}')
        
        # Test direct 
        n_vectors = [[5,6,7], [0, 0, 1], [2, -2, -5]]
        calc_struct_fact = gp.CalculatorStructureFactor(sim.configuration, backend=backend, n_vectors=n_vectors)
        calc_struct_fact.update()
        struc_fact = calc_struct_fact.read(bins=8)

        # Test generated q_vectors
        calc_struct_fact = gp.CalculatorStructureFactor(sim.configuration, backend=backend)
        calc_struct_fact.generate_q_vectors(q_max=10.0)
        calc_struct_fact.update()
        struc_fact = calc_struct_fact.read(bins=128)

def test_atomic_form_factors():
        D = 3
        rho = 1.0
        number_of_particles = 10_000
        conf = gp.Configuration(D=D)
        conf.make_positions(N=number_of_particles, rho=rho)
        conf['m'] = 1.0

        n_vectors = np.array([[5,6,7], [0, 0, 1], [2, -2, -5]])
        atomic_form_factors = np.random.random(number_of_particles)

        calc_single = gp.CalculatorStructureFactor(conf, n_vectors, atomic_form_factors, 'CPU single core')
        number_of_updates = 4
        for _ in range(number_of_updates):
            conf['r'] = (np.random.rand(number_of_particles, D)-0.5) * conf.simbox.lengths  # Ideal gas configuration
            conf.copy_to_device()
            calc_single.update()
        calc_multi = gp.CalculatorStructureFactor(conf, n_vectors, atomic_form_factors, 'CPU multi core')
        for _ in range(number_of_updates):
            conf['r'] = (np.random.rand(number_of_particles, D)-0.5) * conf.simbox.lengths  # Ideal gas configuration
            conf.copy_to_device()
            calc_multi.update()

def test_structure_factor_gpu():
    D = 3
    rho = 1.0
    number_of_particles = 10_000
    conf = gp.Configuration(D=D)
    conf.make_positions(N=number_of_particles, rho=rho)
    conf['m'] = 1.0

    # Ideal gas
    conf['r'] = (np.random.rand(number_of_particles, D)-0.5) * conf.simbox.lengths

    # Test direct
    calc_struct_fact = gp.CalculatorStructureFactor(conf, backend='GPU')
    calc_struct_fact.generate_q_vectors(q_max=10.0)
    print(f'{calc_struct_fact.q_vectors.shape = }')
    calc_struct_fact.update()
    struc_fact = calc_struct_fact.read(bins=32)

    assert np.isclose(sum(struc_fact['S(|q|)'])/len(struc_fact['S(|q|)']), 1.0, atol=0.2)


if __name__ == '__main__':  # pragma: no cover
    test_structure_factor()
    test_structure_factor_backends()
    test_structure_factor_gpu()

