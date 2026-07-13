import math
from attr import attributes
import numpy as np
import matplotlib.pyplot as plt


def calc_dynamics_(positions, images, ptype, simbox, block0, conf_index0, block1, conf_index1, time_index,  
                   msd, m4d, qvalues, Fs, overlap_distances, Qs, simbox_name='Orthorhombic'):
    """
    Calculate contribution to dynamical properties from conf_index1 in block1 using conf_index0 in block0
    as initial time, and add it at time_index in appropiate arrays.

    TODO: Allow more than one q-value per type
    """
    init_coord = {False: 0, True: 1} [simbox_name == 'LeesEdwards']
    dR = positions[block1, conf_index1, :, init_coord:] - positions[block0, conf_index0, :, init_coord:]
    dR += (images[block1, conf_index1, :, init_coord:] - images[block0, conf_index0, :, init_coord:]) * simbox[init_coord:]

    for i in range(np.max(ptype) + 1):
        dR_type = dR[ptype == i, :]            # D dimensional vector, per particle 
        dR_i_sq = np.sum( dR_type**2, axis=1)  # Length of that vector squared, per particle
        msd[time_index, i] += np.mean(dR_i_sq)
        m4d[time_index, i] += np.mean(dR_i_sq ** 2)
        Fs[time_index, i] += np.mean(np.cos(dR_type*qvalues[i]))
        Qs[time_index, i] += np.mean(dR_i_sq < overlap_distances[i]**2)

    return msd, m4d, Fs, Qs


def calc_dynamics(trajectory, first_block, qvalues=7.5, overlap_distances=0.3, extra_times_method='auto'):
    """Compute dynamical properties from a trajectory in a HDF5 file object.

    This function processes blocks of saved configurations to evaluate time‐dependent
    dynamical observables, the mean square displacement (MSD), the non‐Gaussian
    parameter (alpha2), the self‐intermediate scattering function (Fs), and the self overlap (Qs)

    Parameters
    ----------
    trajectory : h5py.File object in the gamdpy style

    first_block : int
        Index of the first block to use as the reference origin.

    qvalues : float or array‐like of shape (num_types,), optional
        Wavevector magnitudes at which to compute the self‐intermediate scattering
        function Fs. If a single float is provided, it is broadcast to all particle types.
        default: 7.5

    overlap_distances : float or array‐like of shape (num_types,), optional
        Criteria for defining self overlap, Qs : particles are counted if they moved less than 'overlap_distances'
        If a single float is provided, it is broadcast to all particle types.
        Default: 0.3

    extra_times_method : str, optional
        Method for determining extra times beyond the times saved within the timeblocks. Options are:
        - 'log': Use logarithmic spacing. Base of the logarithm is determined from the last three times saved in timeblocks.
        - 'linear': Use linear spacing. The spacing is determined from the last two times saved in timeblocks.
        - 'auto': Automatically determine whether to use linear or logarithmic spacing based on last three times saved in timeblocks. If the last three times are evenly spaced, linear spacing is used; otherwise, logarithmic spacing is used.
        Default: 'auto' 

    Returns
    -------
    results : dict
        Dictionary containing dynamical data. Contains the following keys:
        - 'times': Array of times at which the dynamical properties are evaluated.
        - 'msd': Mean square displacement (MSD) for each particle type.
        - 'alpha2': Non-Gaussian parameter (alpha2) for each particle type
        - 'qvalues': Wavevector magnitudes used for computing Fs.
        - 'Fs': Self-intermediate scattering function (Fs) for each particle type.
        - 'overlap_distances': Overlap distances used for computing Qs.
        - 'Qs': Self overlap (Qs) for each particle type.
        - 'count': Number of pairs of configurations (initial and final) that used to compute the dynamical properties at each time point.
        
    Examples
    --------
    For command‐line usage, see:
        $ python -m gamdpy.tools.calc_dynamics -h

    Usage within a Python script:

    >>> import gamdpy as gp
    >>> sim = gp.get_default_sim()  # Replace with your simulation object
    >>> for block in sim.run_timeblocks(): pass
    >>> dynamics = gp.calc_dynamics(sim.output, first_block=0, qvalues=7.25, overlap_distances=0.3)
    extra_times_method='log', scale_factor=2.000
    0.......
    >>> dynamics.keys()
    dict_keys(['times', 'msd', 'alpha2', 'qvalues', 'Fs', 'overlap_distances', 'Qs', 'count'])

    """
    ptype = trajectory['initial_configuration/ptype'][:].copy()
    attributes = trajectory.attrs
   
    simbox_name = trajectory['initial_configuration'].attrs['simbox_name']
    simbox_data = trajectory['initial_configuration'].attrs['simbox_data'].copy()

    num_types = np.max(ptype) + 1
    
    if isinstance(qvalues, float):
        qvalues = np.ones(num_types)*qvalues
    #print('Calculating Fs using q-values:', qvalues)

    if isinstance(overlap_distances, float):
        overlap_distances = np.ones(num_types)*overlap_distances
    #print('Calculating Qs using overlap_distances:', overlap_distances)
        
    num_blocks, conf_per_block, N, D = trajectory['trajectory/positions'].shape

    if 'steps' in trajectory['trajectory'].keys():
        #print('Using steps from trajectory/steps')
        steps_in_blocks = trajectory['trajectory']['steps'][:]
        assert len(steps_in_blocks) == conf_per_block, "Number of steps in trajectory/steps does not match number of configurations per block"
        steps_per_timeblock = trajectory['trajectory'].attrs['steps_per_timeblock']
        #print(steps_in_blocks)
    else: # trajectory is from before schedulers, so must be log2 schedule
        #print('Using steps from trajectory')
        steps_per_timeblock = conf_per_block
        steps_in_blocks = np.array([0] + [2**k for k in range(conf_per_block-1)]) 
        steps_per_timeblock = steps_in_blocks[-1]
        #print(steps_in_blocks)
 
    blocks = trajectory['trajectory/positions']  # If picking out dataset in inner loop: Very slow!
    images = trajectory['trajectory/images']
    if simbox_name == "Orthorhombic":
        simbox = simbox_data
    elif simbox_name == "LeesEdwards":
        simbox = simbox_data[:D]
    else:
        raise ValueError('Simbox not recognized: ', simbox_name)

    if first_block > num_blocks - 1:
        print("Warning [calc_dynamics] first_block greater than number of blocks. Remainder will be taken")
    first_block = first_block  % num_blocks # necessary to allow the pythonic idiom of negative indexes
    
    if extra_times_method == 'auto':
        if steps_in_blocks[-3] - steps_in_blocks[-2] == steps_in_blocks[-2] - steps_in_blocks[-1]:
            extra_times_method = 'linear'
        else:
            extra_times_method = 'log'

    extra_steps = []
    # determine the extra steps to use for times longer than the blocks
    if extra_times_method == 'log':
        scale_factor = steps_in_blocks[-1]/steps_in_blocks[-2]
        print(f'{extra_times_method=}, {scale_factor=:.3f}')
        this_step = steps_in_blocks[-1]*scale_factor
        while this_step <= num_blocks*steps_per_timeblock/2:
            extra_steps.append(int(this_step))
            this_step *= scale_factor

    if extra_times_method == 'linear':
        extra_timestep = steps_in_blocks[-1] - steps_in_blocks[-2]
        print(f'{extra_times_method=}, {extra_timestep=:d}')
        this_step = steps_in_blocks[-1] + extra_timestep
        while this_step <= num_blocks*steps_per_timeblock/2:
            extra_steps.append(int(this_step))
            this_step += extra_timestep

    # Determine how to find configurations at seperated by each of the extra times
    num_blocks_diff_list = []
    extra_index_list = []
    actual_steps_list = []

    for this_step in extra_steps:
        num_blocks_diff = this_step // steps_per_timeblock
        missing_time = this_step - steps_per_timeblock*num_blocks_diff
        index = 0
        while index < len(steps_in_blocks)-1 and steps_in_blocks[index] < missing_time:
            index += 1
        if extra_times_method == 'log' and index>0:
            if (steps_in_blocks[index]/missing_time) > (missing_time/steps_in_blocks[index-1]):
                index -= 1 # choose the closest index on log-scale,
        if extra_times_method == 'linear' and index>0:
            if (steps_in_blocks[index] - missing_time) > (missing_time - steps_in_blocks[index-1]):
                index -= 1 # choose the closest index on linear-scale,  

        actual_steps = num_blocks_diff*steps_per_timeblock + steps_in_blocks[index]
        # print(this_step, num_blocks_diff, missing_time, steps_in_blocks[index], steps_in_blocks[index], actual_steps)
        
        num_blocks_diff_list.append(num_blocks_diff)
        extra_index_list.append(index)
        actual_steps_list.append(actual_steps)
    
    extra_times = len(extra_steps)
    total_times = conf_per_block - 1 + extra_times
    count = np.zeros((total_times, 1), dtype=np.int32)
    msd = np.zeros((total_times, num_types))
    m4d = np.zeros((total_times, num_types))
    Fs = np.zeros((total_times, num_types))
    Qs = np.zeros((total_times, num_types))

    times = np.zeros(total_times)
    times[0 : conf_per_block-1] = steps_in_blocks[1:]  # in block
    times[conf_per_block -1 :] = np.array(actual_steps_list)
    times *= attributes['dt']

    for block in range(first_block, num_blocks):
        if block % 10 == 0:
            print(block, end='', flush=True)
        else:
            print('.', end='', flush=True)

        # Compute dynamics at times within block
        for i in range(conf_per_block - 1):
            count[i] += 1
            calc_dynamics_(blocks, images, ptype, simbox, block, i + 1, block, 0, i, msd, m4d, qvalues, Fs, overlap_distances, Qs, simbox_name)

        # Compute dynamics at 'ekstra' times, i.e. times beyond what is saved in the timeblock
        for i in range(len(extra_steps)):
             other_block = block + num_blocks_diff_list[i]
             if other_block < num_blocks:
                count[i+conf_per_block-1] += 1
                calc_dynamics_(blocks, images, ptype, simbox,  block, 0, other_block, extra_index_list[i ], i+conf_per_block-1, msd, m4d, qvalues, Fs, overlap_distances, Qs, simbox_name)
    print()

    msd /= count
    m4d /= count
    Fs  /= count
    Qs  /= count

    Deff = D
    if simbox_name == 'LeesEdwards':
        Deff -= 1

    alpha2 = Deff * m4d / ((Deff+2) * msd ** 2) - 1
    return {'times': times, 'msd': msd, 'alpha2': alpha2, 'qvalues':qvalues, 'Fs':Fs, 'overlap_distances':overlap_distances, 'Qs':Qs, 'count': count}


def create_msd_plot(dynamics, figsize=(8, 6)):
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    for dyn in dynamics:
        axs.loglog(dyn['times'], dyn['msd'], '.-', label=dyn['name'])
    axs.set_xlabel('Time')
    axs.set_ylabel('MSD')
    axs.legend()
    return fig, axs

def create_alpha2_plot(dynamics, figsize=(8, 6)):
    fig, axs = plt.subplots(1, 1, figsize=figsize)
    for dyn in dynamics:
        axs.semilogx(dyn['times'], dyn['alpha2'], '.-', label=dyn['name'])
    axs.set_xlabel('Time')
    axs.set_ylabel('alpha2')
    axs.legend()
    return fig, axs


def main(argv: list = None) -> None:
    """ Command line interface for calc_dynamics """
    import sys
    import h5py

    help_message = """gamdpy: calc_dynamics

Calculate and show the mean square displacement (MSD)

Usage: python -m gamdpy.tools.calc_dynamics [options] <input filename> [<input filename> ...]

Options:
    -h, --help      Print this help message and exit.
    -f <int>        First block to use. Default is 0.
    -o <filename>   Output filename. Default is no output.

Example: python -m gamdpy.tools.calc_dynamics -f 4 -o msd.pdf LJ*.h5
    """

    if argv is None:
        argv = sys.argv

    # Print help if '-h' or '--help' is in argv or if no arguments are given
    if '-h' in argv or '--help' in argv or len(argv) == 1:
        print(help_message)
        return

    argv.pop(0)  # remove name

    first_block = 0
    output_filename = ''
    while argv[0][0] == '-':
        if argv[0] == '-f':
            argv.pop(0)  # remove '-f'
            first_block = int(argv.pop(0))  # read and remove parameter
        if argv[0] == '-o':
            argv.pop(0)  # remove '-o'
            output_filename = argv.pop(0)  # read and remove parameter

    # The rest should be filenames...
    dynamics = []
    for filename in argv:
        print(filename, ':', end=' ')
        with h5py.File(filename, "r") as f:
            dynamics.append(calc_dynamics(f, first_block))
            dynamics[-1]['name'] = filename[:-3]

    fig, axs = create_msd_plot(dynamics)
    if not output_filename == '':
        plt.savefig(output_filename)
    plt.show()


if __name__ == '__main__':
    main()
