
def test_NPT_Langevin_isotropic(verbose=False, plot=False):
    # Investigate a state-point in Table I of https://doi.org/10.1063/1.4818747
    import numpy as np
    import gamdpy as gp

    # Setup configuration: FCC Lattice
    configuration = gp.Configuration(D=3, compute_flags={'Vol':True})
    expected_volume_per_particle = 0.8835
    configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=1/expected_volume_per_particle)
    configuration['m'] = 1.0
    configuration.randomize_velocities(temperature=2.0, seed=2025)

    # Setup pair potential: Single component 12-6 Lennard-Jones
    pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
    sig, eps, cut = 1.0, 1.0, 2.5
    pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)
    interactions = [pair_pot, ]

    # Setup integrator
    integrator = gp.integrators.NPT_Langevin(
        temperature=2.0,
        pressure=22.007,
        alpha=1.0,
        alpha_baro=0.01,
        mass_baro=1.0,
        volume_velocity=0.0,
        dt=0.004,
        seed=2025,
    )

    runtime_actions = [
        gp.ScalarSaver(steps_between_output=16),
        gp.MomentumReset(steps_between_reset=100),
    ]

    sim = gp.Simulation(configuration, interactions, integrator, runtime_actions,
                        num_timeblocks=32, steps_per_timeblock=2048,
                        storage='memory')

    initial_box_lengths = configuration.simbox.get_lengths()
    if verbose:
        print(f"Initial box lengths: {initial_box_lengths}")
        print(sim.configuration['r'])
        print(f'Number of particles: {configuration.N}')

    # Production run
    for _ in sim.run_timeblocks():
        if verbose:
            print(sim.status(per_particle=True), configuration.simbox.get_lengths())

    final_box_lengths = configuration.simbox.get_lengths()
    if verbose:
        print(f"Final box lengths:   {final_box_lengths}")
        print(sim.configuration['r'])

    # Assert that the box is still cubic by testing that lengths[1]/lengths[0]=lengths[2]/lengths[0]=1
    assert np.isclose(final_box_lengths[1]/final_box_lengths[0], 1.0, rtol=0.01), f"Box lengths are not cubic: {final_box_lengths}"
    assert np.isclose(final_box_lengths[2]/final_box_lengths[0], 1.0, rtol=0.01), f"Box lengths are not cubic: {final_box_lengths}"

    U, W, K, Vol = gp.extract_scalars(sim.output, ['U', 'W', 'K', 'Vol'], first_block=0)

    if plot:
        # Plot potential energy per particle as a function of time
        # Get times
        dt = sim.output.attrs['dt']  # Timestep
        time = np.arange(len(U)) * dt * sim.output['scalar_saver'].attrs['steps_between_output']

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(time, Vol/configuration.N)
        # Horisontal line at expected value
        plt.axhline(expected_volume_per_particle, color='k', linestyle='--')
        tolerence_for_mean = 0.005
        plt.axhline(expected_volume_per_particle + tolerence_for_mean, color='r', linestyle=':')
        plt.axhline(expected_volume_per_particle - tolerence_for_mean, color='r', linestyle=':')
        plt.xlabel(r'Time, $t$')
        plt.ylabel('Volume per particle, $v = V/N$')
        plt.show()

    volume_per_particle_mean = float(np.mean(Vol/configuration.N))
    volume_per_particle_std = float(np.std(Vol/configuration.N))
    expected_std = 0.005

    if verbose:
        print(f"Volume per particle mean: {volume_per_particle_mean = }")
        print(f"Expected volume per particle: {expected_volume_per_particle = }")
        print(f"Standard deviation of volume per particle: {volume_per_particle_std = }")
        print(f"Expected standard deviation of volume per particle: {expected_std = }")

    assert np.isclose(volume_per_particle_mean, expected_volume_per_particle, atol=0.05), "Wrong volume per particle"
    assert np.isclose(volume_per_particle_std, expected_std, atol=0.002), "Wrong standard deviation of volume per particle"

if __name__ == '__main__':
    test_NPT_Langevin_isotropic(verbose=True, plot=True)
