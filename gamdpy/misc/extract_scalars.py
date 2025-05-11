def extract_scalars(data, column_list, first_block=0, D=3):
    """ Extracts scalar data from simulation output.

    Parameters
    ----------

    data : dict
        Output from a Simulation object.

    column_list : list of str

    first_block : int
        Index of the first timeblock to extract data from.

    D : int
        Dimension of the simulation.

    Returns
    -------

    tuple
        Tuple of 1D numpy arrays containing the extracted scalar data.


    Example
    -------

    >>> import numpy as np
    >>> import gamdpy as gp
    >>> sim = gp.get_default_sim()  # Replace with your simulation object
    >>> for block in sim.run_timeblocks(): pass
    >>> U, W = gp.extract_scalars(sim.output, ['U', 'W'], first_block=1)
    """

    # Indices hardcoded for now (see scalar_calculator above)
    column_indices = {}
    try:
        scalar_names = data['scalar_saver'].attrs['scalar_names']
    except KeyError:
        # try the old label
        print("Data file uses old format (meta data labelled 'scalars_names' rather than 'scalar_names'); at some point suport for this format will be removed.")
        scalar_names = data.attrs['scalars_names']

    for index, name in enumerate(scalar_names):
        column_indices[name] = index

    output_list = []
    for column in column_list:
        output_list.append(data['scalar_saver/scalars'][first_block:,:,column_indices[column]].flatten())
    return tuple(output_list)

