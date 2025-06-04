""" Test code for computing radial distribution function """

from itertools import product

import numpy as np

import gamdpy as gp

def test_radial_distribution():
    spatial_dimensions = 1, 2, 3, 4
    densities = 1.0, 0.32
    for D, rho in product(spatial_dimensions, densities):
        # print(D, rho)
        number_of_particles = 10_000
        conf = gp.Configuration(D=D)
        conf.make_positions(N=number_of_particles, rho=rho)
        conf['m'] = 1.0

        bins=16
        calc_rdf = gp.CalculatorRadialDistribution(conf, bins=bins)
        number_of_updates = 4
        for _ in range(number_of_updates):
            conf['r'] = (np.random.rand(number_of_particles, D)-0.5) * conf.simbox.get_lengths()  # Ideal gas configuration
            # print(conf['r'][0])
            conf.copy_to_device()
            calc_rdf.update()

        rdf_data = calc_rdf.read()
        r = rdf_data['distances']
        assert len(r) == bins, "Problem with (D, rho) = " + str((D, rho))
        rdfs = rdf_data['rdf']
        # print(rdfs)
        assert rdfs.shape == (number_of_updates, bins), "Problem with (D, rho) = " + str((D, rho))
        mean_rdf = np.mean(rdfs)
        assert abs(mean_rdf - 1.0) < 0.01, "Problem with (D, rho) = " + str((D, rho))
        assert abs(np.max(rdfs) - 1.0) < 0.8, "Problem with (D, rho) = " + str((D, rho))
        assert abs(np.min(rdfs) - 1.0) < 0.8, "Problem with (D, rho) = " + str((D, rho))


def test_radial_distribution_lees_edwards():
    spatial_dimensions = 2, 3, 4
    densities = 1.0, 0.32
    for D, rho in product(spatial_dimensions, densities):
        number_of_particles = 10_000
        conf = gp.Configuration(D=D)
        conf.make_positions(N=number_of_particles, rho=rho)
        conf['m'] = 1.0

        conf.simbox = gp.LeesEdwards(conf.D, conf.simbox.get_lengths(), box_shift=0.1)

        bins=16
        calc_rdf = gp.CalculatorRadialDistribution(conf, bins=bins)

        number_of_updates = 4
        for _ in range(number_of_updates):
            conf['r'] = (np.random.rand(number_of_particles, D)-0.5) * conf.simbox.get_lengths()  # Ideal gas configuration
            # print(conf['r'][0])
            conf.copy_to_device()
            calc_rdf.update()

        rdf_data = calc_rdf.read()
        r = rdf_data['distances']
        assert len(r) == bins, "Problem with (D, rho) = " + str((D, rho))
        rdfs = rdf_data['rdf']
        # print(rdfs)
        assert rdfs.shape == (number_of_updates, bins), "Problem with (D, rho) = " + str((D, rho))
        mean_rdf = np.mean(rdfs)
        assert abs(mean_rdf - 1.0) < 0.01, "Problem with (D, rho) = " + str((D, rho))
        assert abs(np.max(rdfs) - 1.0) < 0.8, "Problem with (D, rho) = " + str((D, rho))
        assert abs(np.min(rdfs) - 1.0) < 0.8, "Problem with (D, rho) = " + str((D, rho))

if __name__ == "__main__":
    test_radial_distribution()
    test_radial_distribution_lees_edwards()
