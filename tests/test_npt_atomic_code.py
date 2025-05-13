def test_npt_atomic() -> None:
    import numpy as np
    import gamdpy as gp

    from object_lib import configuration_SC as configuration
    
    # Test init
    itg = gp.integrators.NPT_Atomic(temperature=2.0, tau=0.4, pressure=4.7, tau_p=20, dt=0.001)
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
    itg.get_kernel(configuration=configuration,
            compute_plan = gp.get_default_compute_plan(configuration), 
            compute_flags = gp.get_default_compute_flags(),
            interactions_kernel = None,
            verbose=True)
    return

if __name__ == '__main__':
    test_npt_atomic()
