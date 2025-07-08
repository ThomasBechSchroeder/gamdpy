def test_npt_atomic() -> None:
    import numpy as np
    import gamdpy as gp

    from object_lib import configuration_SC as configuration
    
    # Test init
    itg = gp.integrators.NPT_Atomic(2.0, 0.4, 4.7, 20, 0.001)
    assert itg.temperature==2.0 , "Integrator NPT_Atomic: error with temperature input"
    assert itg.tau==0.4         , "Integrator NPT_Atomic: error with temperature relax time tau input"
    assert itg.pressure==4.7    , "Integrator NPT_Atomic: error with pressure input"
    assert itg.tau_p==20        , "Integrator NPT_Atomic: error with pressure relax time tau_p input"
    assert itg.dt==0.001        , "Integrator NPT_Atomic: error with timestep dt input"
    # Check initialization of barostat and thermostat state
    assert np.all(itg.thermostat_state == np.array([0,0])), "Integrator NPT_Atomic: error with thermostat_state initialization"
    assert np.all(itg.barostat_state == np.array([0,0,0])), "Integrator NPT_Atomic: error with barostat_state initialization"
    assert isinstance(itg.get_params(configuration, ()), tuple), "Integrator NPT_Atomic: error with get_params"

    # Test get_kernel
    itg.get_kernel(configuration=configuration,
            compute_plan = gp.get_default_compute_plan(configuration), 
            compute_flags = gp.get_default_compute_flags(),
            interactions_kernel = None,
            verbose=True)

    # Test init for callable temperatures and pressures
    temperature = gp.make_function_ramp(2.0, 100, 3.0, 400)
    pressure    = gp.make_function_ramp(4.0, 100, 5.0, 400)
    itg = gp.integrators.NPT_Atomic(temperature=temperature, tau=0.4, pressure=pressure, tau_p=20, dt=0.001)

    # Test get_kernel
    compute_flags = gp.get_default_compute_flags()
    compute_flags['Fsq'] = True
    itg.get_kernel(configuration=configuration,
            compute_plan = gp.get_default_compute_plan(configuration), 
            compute_flags = compute_flags,
            interactions_kernel = None,
            verbose=True)
    return

def test_npt_atomic_no_virial():
    # Test that code raise an error if configuration object has no virial flag set True
    import gamdpy as gp
    import pytest

    configuration = gp.Configuration(D=3, compute_flags={'W':False})
    configuration.make_positions(N=1000, rho=1.0)
    from object_lib import pairpot_LJ
    itg = gp.integrators.NPT_Atomic(temperature=2.0, tau=0.4, pressure=1.0, tau_p=20, dt=0.001)
    with pytest.raises(ValueError,
                       match="The NPT_Atomic requires virial flag to be True in the configuration object."):
        itg.get_kernel(configuration=configuration,
                compute_plan = gp.get_default_compute_plan(configuration),
                compute_flags = gp.get_default_compute_flags(),
                interactions_kernel = None,
                verbose=True)

def test_npt_atomic_not_Orthorhombic():
    # Test that code raise an error for Lees Edwards Simulation cell
    import gamdpy as gp
    import pytest

    configuration = gp.Configuration(D=3)
    configuration.make_positions(N=1000, rho=1.0)
    configuration.simbox = gp.LeesEdwards(configuration.D, configuration.simbox.get_lengths())
    interactions = [gp.PairPotential(gp.harmonic_repulsion, params=[1.0, 1.0], max_num_nbs=1000), ]
    integrator = gp.integrators.NPT_Atomic(temperature=2.0, tau=0.4, pressure=1.0, tau_p=20, dt=0.001)
    runtime_actions = []

    with pytest.raises(TypeError,
                       match="The NPT Atomic integrator expected Orthorhombic simulation box but got .*LeesEdwards.*."):
        sim = gp.Simulation(configuration, interactions, integrator, runtime_actions,
                            num_timeblocks=3, steps_per_timeblock=128, storage='memory')

def test_npt_atomic_not_3d():
    # Test that an error is raised if the spatial dimension in not D=3
    import gamdpy as gp
    import pytest

    configuration = gp.Configuration(D=2)
    configuration.make_positions(N=1000, rho=1.0)
    interactions = [gp.PairPotential(gp.harmonic_repulsion, params=[1.0, 1.0], max_num_nbs=1000), ]
    integrator = gp.integrators.NPT_Atomic(temperature=2.0, tau=0.4, pressure=1.0, tau_p=20, dt=0.001)
    runtime_actions = []

    with pytest.raises(ValueError,
                       match="This integrator expected a simulation box with D=3 but got 2."):
        sim = gp.Simulation(configuration, interactions, integrator, runtime_actions,
                            num_timeblocks=3, steps_per_timeblock=128, storage='memory')

if __name__ == '__main__':
    test_npt_atomic()
    test_npt_atomic_no_virial()
    test_npt_atomic_not_Orthorhombic()
    test_npt_atomic_not_3d()
