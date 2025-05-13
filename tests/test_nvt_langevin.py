def test_step_langevin(verbose=False, plot_figures=False) -> None:
    """ Test NVT langevin thermostat
    Test temperature T=1.2 (r_c=2.5) fcc-liquid coexistence state-point in https://doi.org/10.1063/1.4818747 """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import time


    import gamdpy as gp

    # State-point
    temperature = 1.2
    density = 1 / 0.9672

    # Expected values
    expected_kinetic_energy = 3 / 2 * temperature
    expected_total_energy = -4.020
    expected_potential_energy = expected_total_energy - expected_kinetic_energy

    # Setup configuration (give temperature kick to particles to get closer to equilibrium)
    configuration = gp.Configuration(D=3, compute_flags={'W':True, 'K':True})
    configuration.make_lattice(gp.unit_cells.FCC, cells=[7, 7, 7], rho=density)
    configuration['m'] = 1.0
    configuration.randomize_velocities(temperature=2 * temperature, seed=0)

    # Setup pair potential.
    pairfunc = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
    sig, eps, cut = 1.0, 1.0, 2.5
    pairpot = gp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

    # Setup integrator
    dt = 0.005
    alpha = 0.1
    integrator = gp.integrators.NVT_Langevin(temperature, alpha=alpha, dt=dt, seed=0)

    runtime_actions = [gp.MomentumReset(100), 
                   gp.ScalarSaver(32, {'W':True, 'K':True}), ]
    
    # Setup the Simulation
    num_blocks = 32
    steps_per_block = 512
    sim = gp.Simulation(configuration, pairpot, integrator, runtime_actions,
                        num_timeblocks=num_blocks, steps_per_timeblock=steps_per_block,
                        storage='memory')

    # Run simulation one block at a time
    for block in sim.run_timeblocks():
        pass 
    print(sim.summary())

    # Convert scalars to dataframe
    columns = ['U', 'W', 'K']
    data = np.array(gp.extract_scalars(sim.output, columns, first_block=1))
    df = pd.DataFrame(data.T/configuration.N, columns=columns) 

    # Compute summary statistics
    summary_statistics = df.describe()
    potential_energy = summary_statistics['U']['mean']   # per particle
    kinetic_energy = summary_statistics['K']['mean']  # per particle
    if verbose:
        print(
            f'Potential energy (per particle): {potential_energy: 8.4f} (expected: {expected_potential_energy: 8.4f})')
        print(f'Kinetic energy (per particle):   {kinetic_energy: 8.4f} (expected: {expected_kinetic_energy: 8.4f})')

    # Assert that the energies are close to the expected values
    assert abs(potential_energy - expected_potential_energy) < 0.05
    assert abs(kinetic_energy - expected_kinetic_energy) < 0.05

    if plot_figures:
        plt.figure(figsize=(6, 8))
        plt.subplot(2, 1, 1)
        plt.plot(df['U'] , label='u')
        plt.subplot(2, 1, 2)
        plt.plot(df['K'] , label='k')
    df = df.iloc[len(df) // 2:, :]  # last half
    if plot_figures:
        plt.subplot(2, 1, 1)
        plt.plot(df['U'] , label='u (last half)')
        plt.plot([0, 2*len(df)], [expected_potential_energy, expected_potential_energy], 'k--', label='expected')
        plt.ylabel(r'Potential energy, $u$')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(df['K'] , label='k (last half)')
        plt.plot([0, 2*len(df)], [expected_kinetic_energy, expected_kinetic_energy], 'k--', label='expected')
        plt.ylabel(r'Kinetic energy, $k$')
        plt.xlabel('Outer loop step')
        plt.legend()
        plt.show()
        
    return

if __name__ == '__main__':
    test_step_langevin(verbose=True, plot_figures=True)
