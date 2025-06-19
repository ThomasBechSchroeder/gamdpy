
def test_NPT_Langevin_isotropic(verbose=False, plot=False):
    # Investigate state-point in Table I of https://doi.org/10.1063/1.4818747
    import numpy as np
    import gamdpy as gp

    # Setup configuration: FCC Lattice
    configuration = gp.Configuration(D=3, compute_flags={'Vol':True})
    configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=1/0.8835)
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
        alpha_baro=0.1,
        mass_baro=0.1,
        volume_velocity=0.0,
        barostatModeISO=True,  # !!! If this is change to False, then barostat is NOT anisotropic
        boxFlucCoord=2,
        dt=0.01,
        seed=2025,
    )

    runtime_actions = [
        gp.ScalarSaver(steps_between_output=16),
        gp.MomentumReset(steps_between_reset=100),
    ]

    sim = gp.Simulation(configuration, interactions, integrator, runtime_actions,
                        num_timeblocks=16, steps_per_timeblock=1024*8,
                        storage='memory')

    initial_box_lengths = configuration.simbox.get_lengths()
    if verbose:
        print(f"Initial box lengths: {initial_box_lengths}")
        print(sim.configuration['r'])

    # Equilibration run
    for _ in sim.run_timeblocks():
        pass
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

if __name__ == '__main__':
    test_NPT_Langevin_isotropic(verbose=True)
