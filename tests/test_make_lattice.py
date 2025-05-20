import numpy as np

import gamdpy as gp

# cells = [2, 2, 2]
EXPECTED_FCC_POSITIONS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [-0.5, -0.5, -1.0],
        [-0.5, -1.0, -0.5],
        [-1.0, -0.5, -0.5],
        [-1.0, -1.0,  0.0],
        [-0.5, -0.5,  0.0],
        [-0.5, -1.0,  0.5],
        [-1.0, -0.5,  0.5],
        [-1.0,  0.0, -1.0],
        [-0.5,  0.5, -1.0],
        [-0.5,  0.0, -0.5],
        [-1.0,  0.5, -0.5],
        [-1.0,  0.0,  0.0],
        [-0.5,  0.5,  0.0],
        [-0.5,  0.0,  0.5],
        [-1.0,  0.5,  0.5],
        [ 0.0, -1.0, -1.0],
        [ 0.5, -0.5, -1.0],
        [ 0.5, -1.0, -0.5],
        [ 0.0, -0.5, -0.5],
        [ 0.0, -1.0,  0.0],
        [ 0.5, -0.5,  0.0],
        [ 0.5, -1.0,  0.5],
        [ 0.0, -0.5,  0.5],
        [ 0.0,  0.0, -1.0],
        [ 0.5,  0.5, -1.0],
        [ 0.5,  0.0, -0.5],
        [ 0.0,  0.5, -0.5],
        [ 0.0,  0.0,  0.0],
        [ 0.5,  0.5,  0.0],
        [ 0.5,  0.0,  0.5],
        [ 0.0,  0.5,  0.5]
    ]
)

# cells = [2, 2, 2]
EXPECTED_BCC_POSITIONS = np.array([
    [-1.0, -1.0, -1.0],
    [-0.5, -0.5, -0.5],
    [-1.0, -1.0,  0.0],
    [-0.5, -0.5,  0.5],
    [-1.0,  0.0, -1.0],
    [-0.5,  0.5, -0.5],
    [-1.0,  0.0,  0.0],
    [-0.5,  0.5,  0.5],
    [ 0.0, -1.0, -1.0],
    [ 0.5, -0.5, -0.5],
    [ 0.0, -1.0,  0.0],
    [ 0.5, -0.5,  0.5],
    [ 0.0,  0.0, -1.0],
    [ 0.5,  0.5, -0.5],
    [ 0.0,  0.0,  0.0],
    [ 0.5,  0.5,  0.5]
])

# cells = [4, 2]
EXPECTED_HEXAGONAL_POSITIONS = np.array([
    [-2.0, -1.7320508],
    [-1.5, -0.8660254],
    [-2.0,  0.0],
    [-1.5,  0.8660254],
    [-1.0, -1.7320508],
    [-0.5, -0.8660254],
    [-1.0,  0.0],
    [-0.5,  0.8660254],
    [ 0.0, -1.7320508],
    [ 0.5, -0.8660254],
    [ 0.0,  0.0],
    [ 0.5,  0.8660254],
    [ 1.0, -1.7320508],
    [ 1.5, -0.8660254],
    [ 1.0,  0.0],
    [ 1.5,  0.8660254]
])


def test_fcc_lattice():
    # verbose = False
    # plot = False

    cells = [2, 2, 2]
    positions, box_vector = gp.configuration.make_lattice(gp.unit_cells.FCC, cells)
    configuration = gp.Configuration(D=3)
    configuration['r'] = positions
    configuration.simbox = gp.Orthorhombic(configuration.D, box_vector)
    expected_box_vector = np.array([2.0, 2.0, 2.0])
    assert np.allclose(configuration.simbox.get_lengths(), expected_box_vector)
    expected_number_of_particles = 32
    assert configuration['r'].shape[0] == expected_number_of_particles
    # if verbose:
    #     print('    FCC lattice')
    #     print("positions:", configuration['r'])
    #     print("box_vector:", configuration.simbox.get_lengths())
    # if plot:
    #     import matplotlib.pyplot as plt
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(configuration['r'][:, 0], configuration['r'][:, 1], configuration['r'][:, 2])
    #     plt.show()

def test_fcc_lattice_method():
    print("    FCC lattice usign conf.make_lattice")
    conf = gp.Configuration(D=3)
    conf.make_lattice(gp.unit_cells.FCC, [2, 2, 2])
    positions = conf['r']
    print("positions:", positions)
    box_vector = conf.simbox.get_lengths()
    expected_box_vector = np.array([2.0, 2.0, 2.0])
    assert np.allclose(positions, EXPECTED_FCC_POSITIONS, rtol=1e-4)
    assert np.allclose(conf.simbox.get_lengths(), expected_box_vector)
    print("positions:", positions)


def test_bcc_lattice():
    cells = [2, 2, 2]
    positions, box_vector = gp.configuration.make_lattice(gp.unit_cells.BCC, cells)
    configuration = gp.Configuration(D=3)
    configuration['r'] = positions
    configuration.simbox = gp.Orthorhombic(configuration.D, box_vector)
    expected_number_of_particles = 16
    assert configuration['r'].shape[0] == expected_number_of_particles
    assert np.allclose(configuration['r'], EXPECTED_BCC_POSITIONS)
    expected_box_vector = np.array([2.0, 2.0, 2.0])
    assert np.allclose(configuration.simbox.get_lengths(), expected_box_vector)
    # if verbose:
    #     print("    BCC lattice")
    #     print("positions:", configuration['r'])
    #     print("box_vector:", configuration.simbox.get_lengths())
    # if plot:
    #     import matplotlib.pyplot as plt
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(configuration['r'][:, 0], configuration['r'][:, 1], configuration['r'][:, 2])
    #     plt.show()


def test_hexagonal():
    # verbose = False
    # plot = False

    cells = [4, 2]
    positions, box_vector = gp.configuration.make_lattice(gp.unit_cells.HEXAGONAL, cells=cells)
    configuration = gp.Configuration(D=2)
    configuration['r'] = positions
    configuration.simbox = gp.Orthorhombic(configuration.D, box_vector)
    expected_dimensions_of_space = 2
    assert configuration['r'].shape[1] == expected_dimensions_of_space
    expected_number_of_particles = 16
    assert configuration['r'].shape[0] == expected_number_of_particles

    assert np.allclose(configuration['r'], EXPECTED_HEXAGONAL_POSITIONS, rtol=1e-4)

    # if verbose:
    #     print('  Hexagonal lattice')
    #     print("positions:", configuration['r'])
    #     print("box_vector:", configuration.simbox.get_lengths())
    # if plot:
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.title("Hexagonal lattice")
    #     plt.scatter(configuration['r'][:, 0], configuration['r'][:, 1])
    #     # Plot the box
    #     L_x, L_y = box_vector
    #     box = plt.Rectangle([-L_x/2, -L_y/2], L_x, L_y, fill=False)
    #     plt.gca().add_patch(box)
    #     plt.axis('equal')
    #     plt.show()


if __name__ == "__main__":  # pragma: no cover
    test_hexagonal()
    test_fcc_lattice()
    test_fcc_lattice_method()
    test_bcc_lattice()
