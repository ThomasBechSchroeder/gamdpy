import h5py
import gzip
import numpy as np
import numba
import math
from numba import cuda

from .colarray import colarray
from ..simulation_boxes import Orthorhombic, LeesEdwards
from .topology import Topology, duplicate_topology, replicate_topologies
from ..simulation.get_default_compute_flags import get_default_compute_flags
from .Configuration import Configuration

'''
def configuration_to_hdf5(configuration: Configuration, filename: str, meta_data=None) -> None:
        """ Write a configuration to a HDF5 file

        Parameters
        ----------

        configuration : gamdpy.Configuration
            a gamdpy configuration object

        filename : str
            filename of the output file .h5

        meta_data : str
            not used in the function so far (default None)

        Example
        -------

        >>> import os
        >>> import gamdpy as gp
        >>> conf = gp.Configuration(D=3)
        >>> conf.make_positions(N=10, rho=1.0)
        >>> gp.configuration_to_hdf5(configuration=conf, filename="final.h5")
        >>> os.remove("final.h5")       # Removes file (for doctests)

        """

        if not filename.endswith('.h5'):
            filename += '.h5'
        with h5py.File(filename, "w") as f:
            f.attrs['simbox'] = configuration.simbox.get_lengths()
            if meta_data is not None:
                for item in meta_data:
                    f.attrs[item] = meta_data[item]

            ds_r = f.create_dataset('r', shape=(configuration.N, configuration.D), dtype=np.float32)
            ds_v = f.create_dataset('v', shape=(configuration.N, configuration.D), dtype=np.float32)
            ds_p = f.create_dataset('ptype', shape=(configuration.N), dtype=np.int32)
            ds_m = f.create_dataset('m', shape=(configuration.N), dtype=np.float32)
            ds_r_im = f.create_dataset('r_im', shape=(configuration.N, configuration.D), dtype=np.int32)
            ds_r[:] = configuration['r']
            ds_v[:] = configuration['v']
            ds_p[:] = configuration.ptype
            ds_m[:] = configuration['m']
            ds_r_im[:] = configuration.r_im
'''


def configuration_from_hdf5(filename: str, reset_images=False, compute_flags=None) -> Configuration:
    """ Read a configuration from a HDF5 file

    Parameters
    ----------

    filename : str
        filename of the input file .h5

    reset_images : bool
        if True set the images to zero (default False)

    Returns
    -------

    configuration : gamdpy.Configuration
        a gamdpy configuration object

    Example
    -------

    >>> import gamdpy as gp
    >>> conf = gp.configuration_from_hdf5("examples/Data/final.h5")
    >>> print(conf.D, conf.N, conf['r'][0])     # Print number of dimensions D, number of particles N and position of first particle
    3 10 [-0.7181449 -1.3644753 -1.5799187]

    """

    if not filename.endswith('.h5'):
        raise ValueError('Filename not in HDF5 format')
    with h5py.File(filename, "r") as f:
        lengths = f.attrs['simbox']
        r = f['r'][:]
        v = f['v'][:]
        ptype = f['ptype'][:]
        m = f['m'][:]
        r_im = f['r_im'][:]
    N, D = r.shape
    configuration = Configuration(D=D, compute_flags=compute_flags)
    configuration.simbox = Orthorhombic(D, lengths)
    configuration['r'] = r
    configuration['v'] = v
    configuration.ptype = ptype
    configuration['m'] = m
    if reset_images:
        configuration.r_im = np.zeros((N, D), dtype=np.int32)
    else:
        configuration.r_im = r_im
    return configuration


'''
def configuration_from_hdf5_group(f, group_name, reset_images=False, compute_flags=None) -> Configuration:
    """ Read a configuration from an open HDF5 file identified by group-name

    Parameters
    ----------
    f : HDF5 File
        open HDF5 open, as returned by h5py.File()

    reset_images : bool
        if True set the images to zero (default False)

    Returns
    -------

    configuration : gamdpy.Configuration
        a gamdpy configuration object


    Example:
    --------

    >>> import gamdpy as gp
    >>> output_file = h5py.File('examples/Data/LJ_r0.973_T0.70_toread.h5')
    >>> conf = gp.configuration_from_hdf5_group(output_file, 'restarts/restart0000')
    >>> print(conf.D, conf.N, conf['r'][0])     # Print number of dimensions D, number of particles N and position of first particle
    3 2048 [-6.384221  -6.3622074 -6.3125153]

    """


    vectors_array = f[group_name]['vectors'][:]
    _, N, D = vectors_array.shape
    configuration = Configuration(D=D, N=N, compute_flags=compute_flags)


    configuration.vector_columns = f[group_name]['vectors'].attrs['vector_columns']
    configuration.scalar_columns = f[group_name]['scalars'].attrs['scalar_columns']

    configuration.ptype = f[group_name]['ptype'][:]


    scalars_array = f[group_name]['scalars'][:]
    configuration.scalars = scalars_array
    configuration.vectors.array = vectors_array


    simbox_name = f[group_name].attrs['simbox_name']
    simbox_data = f[group_name].attrs['simbox_data']

    if simbox_name == 'Orthorhombic':
        configuration.simbox = Orthorhombic(D, simbox_data)
    elif simbox_name == 'LeesEdwards':
        box_shift_image = {True:0, False: int(simbox_data[D+1])} [reset_images]
        configuration.simbox = LeesEdwards(D, simbox_data[:D], simbox_data[D], box_shift_image)
    else:
        raise ValueError('simbox_name %s not recognized in group %s' % (simbox_name, group_name))

    if reset_images:
        configuration.r_im = np.zeros((N, D), dtype=np.int32)
    else:
        configuration.r_im = f[group_name]['r_im'][:]
    return configuration
'''

def configuration_to_rumd3(configuration: Configuration, filename: str) -> None:
    """ Write a configuration to a RUMD3 file 

    Parameters
    ----------

    configuration : gamdpy.Configuration
        a gamdpy configuration object

    filename : str
        filename of the output file .xyz.gz

    Example
    -------

    >>> import os
    >>> import gamdpy as gp
    >>> conf = gp.Configuration(D=3)
    >>> conf.make_positions(N=10, rho=1.0)
    >>> gp.configuration_to_rumd3(configuration=conf, filename="restart.xyz.gz")
    >>> os.remove("restart.xyz.gz")       # Removes file (for doctests)

    """
    N = configuration.N
    if configuration.D != 3:
        raise ValueError("Only D==3 is compatibale with RUMD-3")

    r = configuration['r']
    v = configuration['v']
    ptype = configuration.ptype
    m = configuration['m']
    r_im = configuration.r_im

    num_types = max(ptype) + 1  # assumes consecutive types  starting from zero
    # find corresponding masses assuming unique mass for each type as required by RUMD-3
    masses = np.ones(num_types, dtype=np.float32)
    for type in range(num_types):
        type_first_idx = np.where(ptype == type)[0][0]
        masses[type] = m[type_first_idx]

    sim_box = configuration.simbox.get_lengths()
    if not filename.endswith('.gz'):
        filename += '.gz'

    with gzip.open(filename, 'wt') as f:
        f.write('%d\n' % N)
        comment_line = 'ioformat=2 numTypes=%d' % (num_types)
        comment_line += ' sim_box=RectangularSimulationBox,%f,%f,%f' % (sim_box[0], sim_box[1], sim_box[2])
        comment_line += ' mass=%f' % (masses[0])
        for mass in masses[1:]:
            comment_line += ',%f' % mass
        comment_line += ' columns=type,x,y,z,imx,imy,imz,vx,vy,vz'
        comment_line += '\n'
        f.write(comment_line)
        for idx in range(N):
            line_out = '%d %.9f %.9f %.9f %d %d %d %f %f %f\n' % (
                ptype[idx], r[idx, 0], r[idx, 1], r[idx, 2], r_im[idx, 0], r_im[idx, 1], r_im[idx, 2], v[idx, 0],
                v[idx, 1],
                v[idx, 2])
            f.write(line_out)


def configuration_from_rumd3(filename: str, reset_images=False, compute_flags=None) -> Configuration:
    """ Read a configuration from a RUMD3 file 

    Parameters
    ----------

    filename : str
        filename of the output file .xyz.gz

    Returns
    -------

    configuration : gamdpy.Configuration
        a gamdpy configuration object

    Example
    -------

    >>> import gamdpy as gp
    >>> conf = gp.configuration_from_rumd3("examples/Data/NVT_N4000_T2.0_rho1.2_KABLJ_rumd3/TrajectoryFiles/restart0000.xyz.gz")
    >>> print(conf.D, conf.N, conf['r'][0])     # Print number of dimensions D, number of particles N and position of first particle
    3 4000 [ 7.197245   6.610052  -4.7467813]

    """
    with gzip.open(filename) as f:
        line1 = f.readline().decode()
        N = int(line1)

        line2 = f.readline().decode()
        meta_data_items = line2.split()
        meta_data = {}
        for item in meta_data_items:
            key, val = item.split("=")
            meta_data[key] = val

        num_types = int(meta_data['numTypes'])
        masses = [float(x) for x in meta_data['mass'].split(',')]
        assert len(masses) == num_types
        if meta_data['ioformat'] == '1':
            lengths = np.array([float(x) for x in meta_data['boxLengths'].split(',')], dtype=np.float32)
        else:
            assert meta_data['ioformat'] == '2'
            sim_box_data = meta_data['sim_box'].split(',')
            sim_box_type = sim_box_data[0]
            sim_box_params = [float(x) for x in sim_box_data[1:]]
            assert sim_box_type == 'RectangularSimulationBox'
            lengths = np.array(sim_box_params)
        # TO DO: handle LeesEdwards sim box
        assert meta_data['columns'].startswith('type,x,y,z,imx,imy,imz')
        has_velocities = (meta_data['columns'].startswith('type,x,y,z,imx,imy,imz,vx,vy,vz'))
        type_array = np.zeros(N, dtype=np.int32)
        r_array = np.zeros((N, 3), dtype=np.float32)
        im_array = np.zeros((N, 3), dtype=np.int32)
        v_array = np.zeros((N, 3), dtype=np.float32)
        m_array = np.ones(N, dtype=np.float32)

        for idx in range(N):
            p_data = f.readline().decode().split()
            ptype = int(p_data[0])
            type_array[idx] = ptype
            r_array[idx, :] = [float(x) for x in p_data[1:4]]
            if not reset_images:
                im_array[idx, :] = [int(x) for x in p_data[4:7]]
            if has_velocities:
                v_array[idx, :] = [float(x) for x in p_data[7:10]]
            m_array[idx] = masses[ptype]

    configuration = Configuration(D=3, compute_flags=compute_flags)
    configuration.simbox = Orthorhombic(3, lengths)
    configuration['r'] = r_array
    configuration['v'] = v_array
    configuration.r_im = im_array
    configuration.ptype = type_array
    configuration['m'] = m_array

    return configuration


def configuration_to_lammps(configuration, timestep=0) -> str:
    """ Convert a configuration to a string formatted as LAMMPS dump file 

    Parameters
    ----------

    configuration : gamdpy.Configuration
        a gamdpy configuration object

    timestep : float
        time at which the configuration is saved

    Returns
    -------

    str
        string formatted as LAMMPS dump file

    Example
    -------

    >>> import gamdpy as gp
    >>> conf = gp.Configuration(D=3)
    >>> conf.make_positions(N=10, rho=1.0)
    >>> lmp_dump = gp.configuration_to_lammps(configuration=conf)

    """
    D = configuration.D
    if D != 3 and D!=2:
        raise ValueError('Only 3D and 2D configurations are supported')
    masses = configuration['m']
    positions = configuration['r']
    image_coordinates = configuration.r_im
    forces = configuration['f']
    velocities = configuration['v']
    ptypes = configuration.ptype
    simulation_box = configuration.simbox.get_lengths()

    # Header
    header = f'ITEM: TIMESTEP\n{timestep:d}\n'
    number_of_atoms = positions.shape[0]
    header += f'ITEM: NUMBER OF ATOMS\n{number_of_atoms:d}\n'
    header += f'ITEM: BOX BOUNDS pp pp pp\n'
    for k in range(D):
        header += f'{-simulation_box[k] / 2:e} {simulation_box[k] / 2:e}\n'
    if D==2:
        header += f'{-1 / 2:e} {1 / 2:e}\n'
    # Atoms
    atom_data = 'ITEM: ATOMS id type mass x y z ix iy iz vx vy vz fx fy fz'
    for i in range(number_of_atoms):
        atom_data += f'\n{i + 1:d} {ptypes[i] + 1:d} {masses[i]:f} '
        for k in range(D):
            atom_data += f'{positions[i, k]:f} '
        if D==2:
            atom_data += f'{0.0:f} '
        for k in range(D):
            atom_data += f'{image_coordinates[i, k]:d} '
        if D==2:
            atom_data += f'{0.0:f} '
        for k in range(D):
            atom_data += f'{velocities[i, k]:f} '
        if D==2:
            atom_data += f'{0.0:f} '
        for k in range(D):
            atom_data += f'{forces[i, k]:f} '
        if D==2:
            atom_data += f'{0.0:f} '
        #atom_data += '\n'
    # Combine header and atom lengths
    lammps_dump = header + atom_data
    return lammps_dump
