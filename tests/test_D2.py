import pytest
import gamdpy as gp
import numpy as np

@pytest.mark.parametrize("grid_size", [
    [10, 10],  #     200 particles
    [30, 30],   #   1800 particles
    [100, 100], #  20000 particles - triggers 'nblist': 'linked lists'
    [300, 300]  # 180000 particles - triggers 'nblist': 'linked lists'
])
def test_system_scaling(grid_size):
    """
    Verifies that the simulation runs and maintains physical sanity 
    across different system sizes.
    """
    # 1. Setup configuration based on parameterized grid_size
    conf = gp.Configuration(D=2)
    conf.make_lattice(unit_cell=gp.unit_cells.HEXAGONAL, cells=grid_size, rho=1.0)
    conf['m'] = 1.0
    conf.randomize_velocities(temperature=0.7)

    # 2. Potential setup
    pair_func = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
    pair_pot = gp.PairPotential(pair_func, params=[1.0, 1.0, 2.5], max_num_nbs=100)

    # 3. Integrator (NVT)
    integrator = gp.integrators.NVT(temperature=0.7, tau=0.2, dt=0.005)

    # 4. Short run
    sim = gp.Simulation(conf, pair_pot, integrator, runtime_actions=[], 
        num_timeblocks=1, steps_per_timeblock=1024,
        storage='memory'
    )
    
    for _ in sim.run_timeblocks():
        pass
    print(sim.summary())
    print(sim.compute_plan)
    print(conf)

    # 5. Thermodynamic Checks
    k = np.mean(conf['K'])  # Kinetic Energy
    u = np.mean(conf['U'])  # Potential Energy
    w = np.mean(conf['W'])  # Virial

    # The Equipartition Theorem in 2D suggests:
    # <K> = (degrees of freedom / 2) * k_B * T
    # For a per-particle basis in 2D: <k> \approx T
    
    assert 0.6 < k < 0.8, f"Kinetic energy {k} unstable for grid {grid_size}"
    assert -2.2 < u < -1.8, f"Potential energy {u} unrealistic for grid {grid_size}"
    assert 10.0 < w < 12.0, f"Virial {w} suggests system instability for grid {grid_size}"

if __name__ == "__main__":
    pytest.main([__file__])
