""" Test the class TrajectoryIO. """

def test_TrajectoryIO():
    import os
    import pytest
    import h5py
    import gamdpy as rp

    ## Perform a short simulation
    # Create a starting configuration
    temperature, density, npart, D = 0.7, 1.0, 1987, 3
    configuration = rp.Configuration(D=D)
    configuration.make_positions(N=npart, rho=density)
    configuration['m'] = 1.0
    configuration.randomize_velocities(temperature=temperature)

    # Setup pair potential: Single component 12-6 Lennard-Jones
    pair_func = rp.apply_shifted_force_cutoff(rp.LJ_12_6_sigma_epsilon)
    pair_func = rp.apply_shifted_potential_cutoff(rp.LJ_12_6_sigma_epsilon)
    sig, eps, cut = 1.0, 1.0, 2.5
    pair_pot = rp.PairPotential(pair_func, params=[sig, eps, cut], max_num_nbs=1000)

    # Setup integrator
    integrator = rp.integrators.NVT(temperature=temperature, tau=0.2, dt=0.001)

    # Setup runtime actions, i.e. actions performed during simulation of timeblocks
    runtime_actions = [rp.ConfigurationSaver(), 
                   rp.ScalarSaver(), 
                   rp.MomentumReset(100)]

    # Setup Simulation      # Note: useless reducing steps_per_timeblock, the time is spent by the jit not by the run
    sim = rp.Simulation(configuration, pair_pot, integrator, runtime_actions,
                        num_timeblocks=1, steps_per_timeblock=128, 
                        storage='memory')

    # Run simulation
    sim.run(verbose=False)
    sim.run(verbose=False)

    ## Save output to -h5 file
    output = rp.tools.TrajectoryIO()
    assert output.h5 == None, "Error with no input initialization"
    output.h5 = sim.output
    output.save_h5(f"LJ_r{density}_T{temperature}.h5")

    ## Test read from h5
    output = rp.tools.TrajectoryIO(f"LJ_r{density}_T{temperature}.h5").get_h5()
    isinstance(output.file, h5py.File)
    nblocks, nconfs, N, D = output['block/positions'].shape
    assert (N, D) == (1987, 3), "Error reading N and D in TrajectoryIO while reading from .h5"
    os.remove(f"LJ_r{density}_T{temperature}.h5")

    ## Test read from rumd3
    # Test read from rumd3 TrajectoryFiles
    output = rp.tools.TrajectoryIO("examples/Data/NVT_N4000_T2.0_rho1.2_KABLJ_rumd3/TrajectoryFiles").get_h5()
    nblocks, nconfs, N, D = output['block/positions'].shape
    assert (N, D) == (4000, 3), "Error reading N and D in TrajectoryIO while reading from  examples/Data/NVT_N4000_T2.0_rho1.2_KABLJ_rumd3/TrajectoryFiles"

    # Test read from rumd3 TrajectoryFiles, trajectory only
    output = rp.tools.TrajectoryIO("examples/Data/NVT_N4000_T2.0_rho1.2_KABLJ_rumd3/TrajectoryFiles_trajonly").get_h5()
    assert isinstance(output.file, h5py.File), "Error in TrajectoryIO while reading from rumd3 trajectory only"

    # Test read from rumd3 TrajectoryFiles, energies only
    output = rp.tools.TrajectoryIO("examples/Data/NVT_N4000_T2.0_rho1.2_KABLJ_rumd3/TrajectoryFiles_eneronly").get_h5()
    assert isinstance(output.file, h5py.File), "Error in TrajectoryIO while reading from rumd3 energy only"

    # Test read from rumd3 TrajectoryFiles but is empty
    output = rp.tools.TrajectoryIO("examples/Data/NVT_N4000_T2.0_rho1.2_KABLJ_rumd3/TrajectoryFiles_empty").get_h5()

    # Test read from rumd3 TrajectoryFiles but folder is not there
    with pytest.raises(Exception):
        output = rp.tools.TrajectoryIO("examples/Data/NVT_N4000_T2.0_rho1.2_KABLJ_rumd3/TrajectoryFiles_not_here")
 
    ## Test read from unsupported format
    output = rp.tools.TrajectoryIO("file.abc").get_h5()
    assert output == None, "Error in TrajectoryIO with not recognized input/unsupported format"

if __name__ == '__main__':
    test_TrajectoryIO()
