""" Test the get_default_sim function. """

def test_get_default_sim():
    import gamdpy as gp
    sim = gp.get_default_sim()
    assert isinstance(sim, gp.Simulation)
    assert isinstance(sim.configuration, gp.Configuration)
    assert sim.configuration['r'] is not None
    assert sim.configuration['m'] is not None
    assert sim.configuration['v'] is not None
    assert isinstance(sim.integrator, gp.integrators.NVT)
    assert isinstance(sim.interactions[0], gp.PairPotential)

if __name__ == '__main__':  # pragma: no cover
    test_get_default_sim()
