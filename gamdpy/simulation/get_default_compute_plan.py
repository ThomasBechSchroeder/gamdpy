import numpy as np
import numba
from numba import cuda
import math
import matplotlib.pyplot as plt
import os

def get_default_compute_plan(configuration):
    """ Return a default compute_plan
    The default compute_plan is a dictionary with a set of parameters specifying how computations are done on the GPU.
    The returned plan depends on the number of particles, and properties of the GPU. The keys of the dictionary are:

    - 'pb': particle per thread block
    - 'tp': threads per particle
    - 'gridsync': Boolean indicating if syncronization should be done by grid.sync() calls
    - 'skin': used when updating nblist
    - 'UtilizeNIII': Boolean indicating if Newton's third law (NIII) should be utilized (see pairpotential_calculator).
    - 'nblist' : 'N squared' (default) or 'linked lists'. Determines algorithm updating nblist

    """
    N = configuration.N

    # Get relevant info about the device. At some point we should be able to deal with no device (GPU) available
    if os.getenv("NUMBA_ENABLE_CUDASIM") != "1":
        # Trying to handle no device (GPU) case
        # NUMBA_ENABLE_CUDASIM environment variable is set to "1" if the cuda simulator is used.
        # See: https://numba.pydata.org/numba-doc/dev/cuda/simulator.html
        device = cuda.get_current_device()

        # Apperently we can't ask the device about how many cores it has, neither in total or per SM (Streaming Processor),
        # so we read the latter from a stored dictionary dependent on the compute capability.
        from gamdpy.cc_cores_per_SM_dict import cc_cores_per_SM_dict
        if device.compute_capability in cc_cores_per_SM_dict:
            cc_cores_per_SM = cc_cores_per_SM_dict[device.compute_capability]
        else:
            print('gamdpy WARNING: Could not find cc_cores_per_SM for this compute_capability. Guessing: 128')
            cc_cores_per_SM = 128
        num_SM = device.MULTIPROCESSOR_COUNT
        num_cc_cores = cc_cores_per_SM * num_SM
        warpsize = device.WARP_SIZE
    else:  # Sets up the behaviour in case the GPU simulator is active and set num_cc_cores = number of threads
        num_SM = 1
        num_cc_cores = numba.get_num_threads()
        warpsize = 1

    # pb: particle per (thread) block
    pb = 512
    while N // pb < 2 * num_SM:  # Performance heuristic
        pb = pb // 2
    if pb < 8:
        pb = 8
    if pb > 256:
        pb = 256

    # tp: threads per particle
    tp = 1
    while N * tp < 2 * num_cc_cores:  # Performance heuristic (conservative)
        tp += 1

    while (pb * tp) % warpsize != 0:  # Number of threads per thread-block should be multiplum of warpsize
        tp += 1

    if tp > 16:
        tp = 16

    # skin: used when updating nblist
    skin = 0.5
    if N > 6 * 1024:
        skin = np.float32( 1.0)  # make the nblist be valid for many steps for large N.

    # UtilizeNIII: Boolean flag indicating if Newton's third law (NIII) should be utilized (see pairpotential_calculator).
    # Utilization of NIII is implemented by using atomic add's to the force array, 
    # so it is inefficient at small system sizes where a lot of conflicts occur.
    UtilizeNIII = True
    if N < 16 * 1024:
        UtilizeNIII = False

    # gridsync: Bolean flag indicating whether synchronization should be done via grid.sync()
    gridsync = True
    if N * tp > 4 * num_cc_cores:  # Heuristic
        gridsync = False

    nblist = 'N squared'
    if N > 6_000:  # Heuristic
        nblist = 'linked lists'
        skin = 0.5

    return {'pb': pb, 'tp': tp, 'skin': skin, 
            'UtilizeNIII': UtilizeNIII, 'gridsync': gridsync, 'nblist': nblist}


