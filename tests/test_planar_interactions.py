import numpy as np


def test_planar_interactions_old_interface() -> None:
    import math
    import numpy as np
    import gamdpy as gp

    # Create a configuration object
    rho, T = 1.5, 1.44
    configuration = gp.Configuration(D=3, compute_flags={'lapU':True})
    configuration.make_positions(N=768, rho=rho)
    configuration['m'] = 1.0  
    configuration.randomize_velocities(temperature=T)

    compute_plan = gp.get_default_compute_plan(configuration)

    # Set up a wall
    wall_dist = 6.31 # Ingebrigtsen & Dyre (2014)
    A = 4.0*math.pi/3*rho
    wall_potential = gp.apply_shifted_force_cutoff(gp.make_LJ_m_n(9,3))
    potential_params_list   = [[A/15.0, -A/2.0, 3.0], [A/15.0, -A/2.0, 3.0]]              # Ingebrigtsen & Dyre (2014)
    particles_list          = [np.arange(configuration.N), np.arange(configuration.N)]    # All particles feel the walls
    wall_point_list         = [[0, 0, wall_dist/2.0], [0, 0, -wall_dist/2.0] ]
    normal_vector_list      = [[0,0,1],               [0,0,-1]]                            
    walls = gp.setup_planar_interactions(configuration, wall_potential, potential_params_list,
                                        particles_list, wall_point_list, normal_vector_list, compute_plan, verbose=True)

    assert isinstance(walls, dict), "gp.setup_planar_interactions should return a dictionary but has not"
    try: value = walls['interactions']
    except KeyError: print("gp.setup_planar_interactions should have 'interactions' key but it hasn't")
    try: value = walls['interaction_params']
    except KeyError: print("gp.setup_planar_interactions should have 'interaction_params' key but it hasn't")

def test_planar_interactions() -> None:
    import gamdpy as gp
    # Make planar interaction
    N = 16
    potential = gp.harmonic_repulsion
    eps, sig = 1.0, 1.0
    params = [[eps, sig], [eps, sig]]
    indices =  [[i, 0] for i in range(N)]
    indices += [[i, 1] for i in range(N)]
    normal_vectors = [[1,0,0], [-1,0,0]]
    points = [[-10,0,0], [10,0,0]]
    planar = gp.interactions.Planar(potential, params, indices, normal_vectors, points)

    # Test that it can be passed to sim object
    rho, T = 1.5, 1.44
    configuration = gp.Configuration(D=3, compute_flags={'lapU':True})
    configuration.make_positions(N=768, rho=rho)
    configuration['m'] = 1.0
    configuration.randomize_velocities(temperature=T)
    compute_plan = gp.get_default_compute_plan(configuration)
    integrator = gp.integrators.NVE(dt=0.01)
    runtime_actions = [gp.ConfigurationSaver(),
                       gp.ScalarSaver(steps_between_output=1)]
    sim = gp.Simulation(configuration, [planar, ], integrator, runtime_actions,
                        num_timeblocks=2, steps_per_timeblock=16,
                        storage='memory')

if __name__ == '__main__':
    test_planar_interactions_old_interface()
    test_planar_interactions()
