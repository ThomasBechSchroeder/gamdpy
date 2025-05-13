import math
import numpy as np
import matplotlib.pyplot as plt


def calc_dynamics_(positions, images, ptype, simbox, block0, conf_index0, block1, conf_index1, time_index,  msd, m4d, qvalues=None, Fs=None):
    """
    Calculate contribution to dynamical properties from conf_index1 in block1 using conf_index0 in block0
    as initial time, and add it at time_index in appropiate arrays.

    TODO: Allow more than one q-value per type
    """
    dR = positions[block1, conf_index1, :, :] - positions[block0, conf_index0, :, :]
    dR += (images[block1, conf_index1, :, :] - images[block0, conf_index0, :, :]) * simbox
    for i in range(np.max(ptype) + 1):
        dR_type = dR[ptype == i, :]
        dR_i_sq = np.sum( dR_type**2, axis=1)
        msd[time_index, i] += np.mean(dR_i_sq)
        m4d[time_index, i] += np.mean(dR_i_sq ** 2)
        Fs[time_index, i] += np.mean(np.cos(dR_type*qvalues[i]))

    return msd, m4d, Fs


def calc_dynamics(trajectory, first_block, qvalues=None):
    ptype = trajectory['initial_configuration/ptype'][:].copy()
    attributes = trajectory.attrs
    
    simbox = attributes['simbox_initial'].copy()
    num_types = np.max(ptype) + 1
    if isinstance(qvalues, float):
        qvalues = np.ones(num_types)*qvalues
    num_blocks, conf_per_block, N, D = trajectory['trajectory_saver/positions'].shape
    blocks = trajectory['trajectory_saver/positions']  # If picking out dataset in inner loop: Very slow!
    images = trajectory['trajectory_saver/images']

    #print(num_types, first_block, num_blocks, conf_per_block, _, N, D, qvalues)
    if first_block > num_blocks - 1:
        print("Warning [calc_dynamics] first_block greater than number of blocks. Remainder will be taken")
    first_block = first_block  % num_blocks # necessary to allow the pythonic idiom of negative indexes

    extra_times = int(math.log2(num_blocks - first_block)) - 1
    total_times = conf_per_block - 1 + extra_times
    count = np.zeros((total_times, 1), dtype=np.int32)
    msd = np.zeros((total_times, num_types))
    m4d = np.zeros((total_times, num_types))
    Fs = np.zeros((total_times, num_types))

    times = attributes['dt'] * 2 ** np.arange(total_times)

    for block in range(first_block, num_blocks):
        for i in range(conf_per_block - 1):
            count[i] += 1
            calc_dynamics_(blocks, images, ptype, simbox, block, i + 1, block, 0, i, msd, m4d, qvalues, Fs)

    # Compute times longer than blocks
    for block in range(first_block, num_blocks):
        for i in range(extra_times):
            index = conf_per_block - 1 + i
            other_block = block + 2 ** (i + 1)
            # print(other_block, end=' ')
            if other_block < num_blocks:
                count[index] += 1
                calc_dynamics_(blocks, images, ptype, simbox, other_block, 0, block, 0, index, msd, m4d, qvalues, Fs)

    msd /= count
    m4d /= count
    Fs  /= count
    alpha2 = 3 * m4d / (5 * msd ** 2) - 1
    return {'times': times, 'msd': msd, 'alpha2': alpha2, 'qvalues':qvalues, 'Fs':Fs, 'count': count}


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

    help_message = """rumdpy: calc_dynamics

Calculate and show the mean square displacement (MSD)

Usage: python -m rumdpy.tools.calc_dynamics [options] <input filename> [<input filename> ...]

Options:
    -h, --help      Print this help message and exit.
    -f <int>        First block to use. Default is 0.
    -o <filename>   Output filename. Default is no output.

Example: python -m rumdpy.tools.calc_dynamics -f 4 -o msd.pdf LJ*.h5
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
