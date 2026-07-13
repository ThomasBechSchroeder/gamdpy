""" Test Ewald summations (Electrostatics interface).
Compute the Madelung constant associated to rocksalt NaCl (https://en.wikipedia.org/wiki/Madelung_constant).
"""

import numpy as np
import gamdpy as gp

def test_charges():
    # Creating a neutral mixture
    num_part = 600
    rho = 1.0
    qA = -1
    qB = 1
    qs = [qA, qB]

    conf = gp.Configuration(D=3)
    conf.make_positions(N=num_part, rho=rho)
    conf.ptype[::2] = 1

    # Settings charges based on types
    conf.set_charges_from_types(qs)

    assert np.sum(conf["q"]) == 0
    assert np.sum(conf["q"]**2) == conf.N
    assert np.all(np.isin(conf["q"], [-1, 1]))

    # Creating a mixture of atoms/ions
    conf = gp.Configuration(D=3)
    conf.make_positions(N=num_part, rho=rho)

    conf["q"][:] = qA
    conf["q"][::6] = qB
    conf["q"][1::6] = 0

    num_positive = np.sum(conf["q"] == qA)
    num_negative = np.sum(conf["q"] == qB)
    num_neutral  = np.sum(conf["q"] == 0.0)

    assert num_positive == 4 * num_negative
    assert num_positive + num_negative + num_neutral == conf.N
    assert np.isclose(np.sum(conf["q"]), 0.0)

    # Test helper function to get charged particles
    charges, charged_idx = conf.get_charged_particles()

    assert len(charges) == num_positive + num_negative
    assert len(charged_idx) == num_positive + num_negative
    assert np.all(charges != 0.0)
    assert np.all(charges == conf["q"][charged_idx])

def test_Ewald():
    # Create NaCl
    ncells = 6
    NaCl = {
        "fractional_coordinates": [
            # Na
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],

            # Cl
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.5],
        ],
        "lattice_constants": [1.0, 1.0, 1.0]
    }
    conf = gp.Configuration(D=3)

    nx = ny = nz = ncells
    conf.make_lattice(NaCl, cells=[nx, ny, nz])

    conf["q"] = np.tile([1, 1, 1, 1, -1, -1, -1, -1], nx * ny * nz)

    # Setting up Electrostatics parameters
    damping = 1
    rc_real = 5
    cut_real = np.full((2,2), rc_real)
    nk = 20

    # Test interface
    electro_pot = gp.Electrostatics(damping=damping, cutoff=cut_real, ncut=nk, max_num_nbs=2000)

    assert electro_pot.damping == damping
    assert np.all(electro_pot.cutoff == cut_real)
    assert electro_pot.ncut == nk

    # Evaluate energy
    evaluator = gp.Evaluator(conf, [electro_pot])
    evaluator.evaluate(conf)

    expected_self_energy = 0.564190
    assert np.isclose(electro_pot.self_energy, expected_self_energy)

    expected_madelung = -1.747565
    u = np.mean(evaluator.configuration['U'] - electro_pot.self_energy)
    assert np.isclose(u, expected_madelung)

if __name__ == '__main__':
    test_charges()
    test_Ewald()