def make_lattice(unit_cell: dict, cells: list = None, rho=None, ptype_unit_cell: list = None) -> tuple:
    """ Returns a configuration of a crystal lattice.
    The lattice is constructed by replicating the unit cell in all directions.
    The `unit_cell` is a dictonary with `fractional_coordinates` for particles, and
    the `lattice_constants` as a list of lengths of the unit cell in all directions.
    The `cells` are the number of unit cells in each direction.
    `ptype_unit_cell` is a list of particle types whose size is the same as that of the fractional_coordinates.
    Returns a dictionary containing the list of positions of the atoms in the lattice (key:"positions"),
    the box vector of the lattice (key: "box_vector"),and if ptype_unit_cell was given, the full list of particle types (key: "ptype").
    The returned positions are in a box of size -L/2 to L/2 for each direction.

    Example
    -------

    >>> import gamdpy as gp
    >>> lattice = gp.configuration.make_lattice(gp.unit_cells.FCC, cells=[8, 8, 8], rho=1.0)
    >>> positions, box_vector = lattice["positions"], lattice["box_vector"]
    >>> configuration = gp.Configuration(D=3)
    >>> configuration['r'] = positions
    >>> configuration.simbox = gp.Orthorhombic(configuration.D, box_vector)

    See also
    --------

    :meth:`gamdpy.Configuration.make_lattice`

    """
    import numpy as np

    # Confirm that input data agrees on the dimension of space
    D_pos = len(unit_cell["fractional_coordinates"][0])  #
    D_lat = len(unit_cell["lattice_constants"])
    D_cells = len(cells)
    if D_pos != D_lat:
        raise ValueError("Dimension of lattice constants must equal dimension of positions")
    if D_pos != D_cells:
        raise ValueError("Dimension of cells must equal dimension of positions")

    # Initialize data structures
    pos = unit_cell["fractional_coordinates"]
    lat = unit_cell["lattice_constants"]
    particles_in_unit_cell = len(pos)
    spatial_dimension = len(pos[0])
    if cells is None:
        cells = [1] * spatial_dimension
    number_of_cells = np.prod(cells)
    positions = np.zeros(
        shape=(particles_in_unit_cell * number_of_cells, spatial_dimension),
        dtype=np.float64,
    )
    if ptype_unit_cell is not None:
        if len(ptype_unit_cell) != particles_in_unit_cell:
            raise ValueError('In make_lattice, length of types is inconsistent with number of particles in unit cell')
        ptype = np.zeros(len(positions), dtype=np.int32)

    for cell_index in range(number_of_cells):
        cell_coordinates = np.array(
            np.unravel_index(cell_index, cells), dtype=np.float64
        )
        for particle_index in range(particles_in_unit_cell):
            positions[cell_index * particles_in_unit_cell + particle_index] = (
                pos[particle_index] + cell_coordinates
            )
            if ptype_unit_cell is not None:
                ptype[cell_index * particles_in_unit_cell + particle_index] = ptype_unit_cell[particle_index]
    positions *= lat
    box_vector = np.array(lat) * np.array(cells)

    if rho is not None:
        box_volume = np.prod(box_vector)
        number_of_particles = len(positions)
        volume_per_particle = box_volume / number_of_particles
        target_volume_per_particle = 1.0 / rho
        scale_factor = (target_volume_per_particle / volume_per_particle) ** (1.0 / spatial_dimension)
        positions *= scale_factor
        box_vector *= scale_factor

    # Center the box (-L/2 to L/2)
    positions -= box_vector/2.0

    lattice =  {"positions": positions,
                "box_vector": box_vector}
    if ptype_unit_cell is not None:
        lattice["ptype"] = ptype

    return lattice
