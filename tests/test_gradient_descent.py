def test_gradient_descent(verbose=False):
    import gamdpy as gp
    import numpy as np
    gp.select_gpu()

    # Setup configuration: FCC Lattice
    configuration = gp.Configuration(D=3, compute_flags={'lapU':True})
    configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
    configuration['m'] = 1.0

    # Setup pair potential: Single component 12-6 Lennard-Jones
    pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
    sig, eps, cut = 1.0, 1.0, 2.5
    pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

    evaluator = gp.Evaluator(configuration, pair_pot)
    evaluator.evaluate(configuration)
    U_ref = np.sum(configuration['U'].astype(np.float64))/configuration.N
    Fsq_ref = np.sum(configuration['f'].astype(np.float64)**2)/configuration.N
    if verbose:
        print(configuration)

    # move particles away from crystal positions.
    np.random.seed(1234)
    configuration['r'] += np.random.uniform(-.2, +.2, configuration['r'].shape)
    configuration.copy_to_device()
    evaluator.evaluate(configuration)
    U_init = np.sum(configuration['U'].astype(np.float64))/configuration.N
    Fsq_init = np.sum(configuration['f'].astype(np.float64)**2)/configuration.N
    if verbose:
        print(configuration)

    # gradient descent as an integrator: 
    integrator = gp.integrators.GradientDescent(dt=0.00001) # v = -f*dt

    # Setup runtime actions, i.e., actions performed during simulation of timeblocks
    runtime_actions = [gp.ScalarSaver(compute_flags={'lapU':True}), ]

    # Setup Simulation. 
    sim = gp.Simulation(configuration, [pair_pot,], integrator, runtime_actions,
                        num_timeblocks=128, steps_per_timeblock=256,
                        storage='memory')

    # Run simulation
    for timeblock in sim.run_timeblocks():
        U = np.sum(configuration['U'].astype(np.float64))/configuration.N
        Fsq = np.sum(configuration['f'].astype(np.float64)**2)/configuration.N
        if verbose:
            print(f'{timeblock=} {U=} {Fsq=}')
    print(sim.summary())

    # U, Fsq contains the final results
    print(f'{U_ref=} {Fsq_ref=}') 
    print(f'{U_init=} {Fsq_init=}') 
    print(f'{U=} {Fsq=}')

    assert(abs(U - U_ref)<0.0001), f'{U=}, {U_ref=}, {U-U_ref=} (>=0.0001)'
    assert(Fsq<0.001), f'{Fsq=}, (>=0.001)'

if __name__ == '__main__':
    test_gradient_descent(verbose=True)