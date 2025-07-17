import math
import numpy as np
import numba

@numba.njit()
def calc_nernst_einstein(dR, charges):
    N, D = dR.shape
    res = 0.
    for i in range(N):
        res += charges[i]**2*np.sum(dR[i,:]**2)/N/6
    return res

@numba.njit(parallel=True)
def calc_einstein(dR, charges):
    N, D = dR.shape
    res = 0.
    for i in numba.prange(N):
        for j in range(N):
            res += charges[i]*charges[j]*np.dot(dR[i], dR[j])/N/6
    return res

@numba.njit(parallel=True)
def calc_crossterm(dR, charges):
    N, D = dR.shape
    res = 0.
    for i in numba.prange(N):
        for j in range(i):
            res += charges[i]*charges[j]*np.dot(dR[i], dR[j])/N/3
    return res

def calc_conductivity_(positions, images, charges, simbox, block0, conf_index0, block1, conf_index1, time_index, nernst_einstein, einstein, crossterm):
    """
    Calculate conductivity from conf_index1 in block1 using conf_index0 in block0
    as initial time, and add it at time_index in appropiate arrays.

    
    """
    dR = positions[block1, conf_index1, :, :] - positions[block0, conf_index0, :, :]        # displacements per particle, ignoring change of images 
    dR += (images[block1, conf_index1, :, :] - images[block0, conf_index0, :, :]) * simbox  # add effect of change of images

    nernst_einstein[time_index] += calc_nernst_einstein(dR, charges)
    einstein[time_index] += calc_einstein(dR, charges)
    crossterm[time_index] += calc_crossterm(dR, charges)

    return 


def calc_conductivity(trajectory, first_block, charges):
    """Compute conductivity from a saved trajectory HDF5 file.

    This function processes blocks of saved configurations to evaluate time‐dependent
    dynamical observables, including the mean square displacement (MSD), the non‐Gaussian
    parameter (alpha2), and the self‐intermediate scattering function (Fs), for one or more
    particle types.

    Parameters
    ----------
    trajectory : h5py.File object in the gamdpy style

    first_block : int
        Index of the first block to use as the reference origin.

    qvalues : float or array‐like of shape (num_types,), optional
        Wavevector magnitudes at which to compute the self‐intermediate scattering
        function Fs. If a single float is provided, it is broadcast to all particle types.
        If None, Fs is not computed (remains zero).

    Returns
    -------
    results : dict
        Dictionary containing dynamcal data.

    Examples
    --------
    For command‐line usage, see:
        $ python -m gamdpy.tools.calc_dynamics -h

    Usage within a Python script:

    >>> import gamdpy as gp
    >>> sim = gp.get_default_sim()  # Replace with your simulation object
    >>> for block in sim.run_timeblocks(): pass
    >>> dynamics = gp.calc_dynamics(sim.output, first_block=0, qvalues=7.5)
    >>> dynamics.keys()
    dict_keys(['times', 'msd', 'alpha2', 'qvalues', 'Fs', 'count'])

    """
    attributes = trajectory.attrs
    
    simbox = trajectory['initial_configuration'].attrs['simbox_data'].copy()
    num_blocks, conf_per_block, N, D = trajectory['trajectory_saver/positions'].shape
    blocks = trajectory['trajectory_saver/positions']  # If picking out dataset in inner loop: Very slow!
    images = trajectory['trajectory_saver/images']

    #print(num_types, first_block, num_blocks, conf_per_block, _, N, D, qvalues)
    if first_block > num_blocks - 1:
        print("Warning [calc_dynamics] first_block greater than number of blocks. Remainder will be taken")
    first_block = first_block  % num_blocks # necessary to allow the pythonic idiom of negative indexes

    extra_times = int(math.log2(num_blocks - first_block)) - 1
    total_times = conf_per_block - 1 + extra_times
    count = np.zeros((total_times), dtype=np.int32)
    nernst_einstein = np.zeros(total_times)
    einstein = np.zeros((total_times))
    crossterm = np.zeros((total_times))

    times = attributes['dt'] * 2 ** np.arange(total_times)

    for block in range(first_block, num_blocks):
        for i in range(conf_per_block - 1):
            count[i] += 1
            calc_conductivity_(blocks, images, charges, simbox, block, i + 1, block, 0, i, nernst_einstein, einstein, crossterm)

    # Compute times longer than blocks
    for block in range(first_block, num_blocks):
        for i in range(extra_times):
            index = conf_per_block - 1 + i
            other_block = block + 2 ** (i + 1)
            # print(other_block, end=' ')
            if other_block < num_blocks:
                count[index] += 1
                calc_conductivity_(blocks, images, charges, simbox, other_block, 0, block, 0, index, nernst_einstein, einstein, crossterm)
    
    nernst_einstein /= count
    einstein /= count
    crossterm /= count
    return {'times': times, 'nernst_einstein': nernst_einstein, 'einstein': einstein, 'crossterm': crossterm, 'count': count}
