import unittest

class Test(unittest.TestCase):

    def setUp(self):
       pass 

    def test_(self):
        import gamdpy as gp
        import numpy as np
        
        # Generate configurations with a FCC lattice
        configuration1 = gp.Configuration(D=3)
        configuration1.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=0.8442)
        configuration1['m'] = 1.0
        configuration1.randomize_velocities(temperature=1.44)
        configuration2 = gp.Configuration(D=3)
        configuration2.make_lattice(gp.unit_cells.FCC, cells=[5, 5, 13], rho=1.2000)
        configuration2['m'] = 1.0
        configuration2.randomize_velocities(temperature=0.44)
        configuration3 = gp.Configuration(D=3)
        configuration3.make_lattice(gp.unit_cells.FCC, cells=[16, 16, 32], rho=0.8442)
        configuration3['m'] = 1.0
        configuration3.randomize_velocities(temperature=2.44)

        # Make pair potentials
        pairfunc = gp.apply_shifted_force_cutoff(gp.LJ_12_6_sigma_epsilon)
        sig, eps, cut = 1.0, 1.0, 2.5
        pairpot1 = gp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

        pairfunc = gp.apply_shifted_potential_cutoff(gp.LJ_12_6_sigma_epsilon)
        sig = [[1.00, 0.80],
               [0.80, 0.88]]
        eps = [[1.00, 1.50],
               [1.50, 0.50]]
        cut = np.array(sig)*2.5
        pairpot2 = gp.PairPotential(pairfunc, params=[sig, eps, cut], max_num_nbs=1000)

        # Make integrators
        #dt = 0.005 # timestep 
        #temperature = 0.7 # Not used for NVE
        #pressure = 1.2 # Not used for NV*

        #integrator1 = gp.integrators.NVE(dt=dt)
        #integrator2 = gp.integrators.NVT(temperature=temperature, tau=0.2, dt=dt)
        #integrator3 = gp.integrators.NVT_Langevin(temperature=temperature, alpha=0.2, dt=dt, seed=2023)
        #integrator4 = gp.integrators.NPT_Langevin(temperature=temperature, pressure=pressure, 
        #                                        alpha=0.1, alpha_baro=0.0001, mass_baro=0.0001, 
        #                                        volume_velocity=0.0, barostatModeISO = True , boxFlucCoord = 2,
        #                                        dt=dt, seed=2023)

        #for configuration in [configuration1, configuration2, configuration3]:
        #    for pairpot in [pairpot1, pairpot2]:
        #        ev = gp.Evaluater(configuration, pairpot)
        #        for integrator in [integrator1, integrator2, integrator3, integrator4]:
        #            sim = gp.Simulation(configuration, pairpot, integrator,
        #                                steps_between_momentum_reset=100,
        #                                num_blocks=64, steps_per_block=1024, storage='memory')

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
