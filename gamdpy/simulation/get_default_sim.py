"Return a sim object of the single component LJ crystal in the NVT ensemble."

def get_default_sim():
    """ Return a sim object of the single component LJ crystal in the NVT ensemble.
    The purpose of this function is to provide a default simulation for testing and simplifying examples.

    Example
    -------

    >>> import os
    >>> os.environ['NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS'] = '0'   # Removes warnings from low occupacy (optional)
    >>> import gamdpy as gp
    >>> sim = gp.get_default_sim()

    """
    import gamdpy as gp

    # Setup configuration: FCC Lattice
    configuration = gp.Configuration(D=3)
    configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.973)
    configuration['m'] = 1.0
    temperature = 0.7
    configuration.randomize_velocities(temperature=temperature)

    # Setup pair potential: Single component 12-6 Lennard-Jones
    pair_func = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
    sig, eps, cut = 1.0, 1.0, 2.5
    pair_pot = gp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

    # Setup integrator: NVT
    integrator = gp.integrators.NVT(temperature=temperature, tau=0.2, dt=0.005)

    # Setup runtime actions, i.e. actions performed during simulation of timeblocks
    runtime_actions = [gp.TrajectorySaver(), 
                   gp.ScalarSaver(), 
                   gp.MomentumReset(100)]


    # Setup Simulation
    sim = gp.Simulation(configuration, pair_pot, integrator, runtime_actions, 
                        num_timeblocks=8, steps_per_timeblock=1024,
                        storage='memory')
    return sim
