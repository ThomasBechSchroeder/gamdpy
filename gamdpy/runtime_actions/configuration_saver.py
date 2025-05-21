import numpy as np
import numba
import math
from numba import cuda, config

from .runtime_action import RuntimeAction


class TrajectorySaver(RuntimeAction):
    """ 
    Runtime action for saving configurations during timeblock.
    Does logarithmic saving.
    """

    def __init__(self, include_simbox=False, verbose=False) -> None:

        self.include_simbox = include_simbox
        self.num_vectors = 2  # 'r' and 'r_im' (for now!)
        #self.sid = {"r":0, "r_im":1}

    def setup(self, configuration, num_timeblocks: int, steps_per_timeblock: int, output, verbose=False) -> None:
        self.configuration = configuration

        if type(num_timeblocks) != int or num_timeblocks < 0:
            raise ValueError(f'num_timeblocks ({num_timeblocks}) should be non-negative integer.')
        self.num_timeblocks = num_timeblocks
        
        if type(steps_per_timeblock) != int or steps_per_timeblock < 0:
            raise ValueError(f'steps_per_timeblock ({steps_per_timeblock}) should be non-negative integer.')
        self.steps_per_timeblock = steps_per_timeblock

        self.conf_per_block = int(math.log2(self.steps_per_timeblock)) + 2  # Should be user controllable

        # Setup output
        if verbose:
            print(f'Storing results in memory. Expected footprint {self.num_timeblocks * self.conf_per_block * self.num_vectors * self.configuration.N * self.configuration.D * 4 / 1024 / 1024:.2f} MB.')

        if 'trajectory_saver' in output.keys():
            del output['trajectory_saver']
        output.create_group('trajectory_saver')
        output.create_dataset('trajectory_saver/positions', 
                              shape=(self.num_timeblocks, self.conf_per_block, self.configuration.N, self.configuration.D),
                              chunks=(1, 1, self.configuration.N, self.configuration.D), 
                              dtype=np.float32)
        output.create_dataset('trajectory_saver/images', 
                              shape=(self.num_timeblocks, self.conf_per_block, self.configuration.N, self.configuration.D),
                              chunks=(1, 1, self.configuration.N, self.configuration.D), 
                              dtype=np.int32)
        #output.attrs['vectors_names'] = list(self.sid.keys())
        if self.include_simbox:
            if 'sim_box' in output['trajectory_saver'].keys():
                del output['trajectory_saver/sim_box']
            output.create_dataset('trajectory_saver/sim_box', 
                                  shape=(self.num_timeblocks, self.conf_per_block, self.configuration.simbox.len_sim_box_data))

        flag = config.CUDA_LOW_OCCUPANCY_WARNINGS
        config.CUDA_LOW_OCCUPANCY_WARNINGS = False
        self.zero_kernel = self.make_zero_kernel()
        config.CUDA_LOW_OCCUPANCY_WARNINGS = flag

    def get_params(self, configuration, compute_plan):
        self.conf_array = np.zeros((self.conf_per_block, self.num_vectors, self.configuration.N, self.configuration.D),
                                   dtype=np.float32)
        self.d_conf_array = cuda.to_device(self.conf_array)

        if self.include_simbox:
            self.sim_box_output_array = np.zeros((self.conf_per_block, self.configuration.simbox.len_sim_box_data), dtype=np.float32)
            self.d_sim_box_output_array = cuda.to_device(self.sim_box_output_array)
            return (self.d_conf_array, self.d_sim_box_output_array)
        else:
            return (self.d_conf_array,)

    def make_zero_kernel(self):
        # Unpack parameters from configuration and compute_plan
        D, num_part = self.configuration.D, self.configuration.N
        pb = 32
        num_blocks = (num_part - 1) // pb + 1

        def zero_kernel(array):
            Nx, Ny, Nz, Nw = array.shape
            global_id = cuda.grid(1)

            if global_id < Nz:  # particles
                for i in range(Nx):
                    for j in range(Ny):
                        for k in range(Nw):
                            array[i, j, global_id, k] = numba.float32(0.0)

        zero_kernel = cuda.jit(zero_kernel)
        return zero_kernel[num_blocks, pb]

    def update_at_end_of_timeblock(self, timeblock: int, output_reference):
        data = self.d_conf_array.copy_to_host()
        # note that d_conf_array has dimensions (self.conf_per_block, 2, self.configuration.N, self.configuration.D)
        output_reference['trajectory_saver/positions'][timeblock], output_reference['trajectory_saver/images'][timeblock] = data[:, 0], data[:, 1]
        #output['trajectory_saver'][block, :] = self.d_conf_array.copy_to_host()
        if self.include_simbox:
            output_reference['trajectory_saver/sim_box'][timeblock, :] = self.d_sim_box_output_array.copy_to_host()
        self.zero_kernel(self.d_conf_array)

    def get_poststep_kernel(self, configuration, compute_plan, verbose=False):
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']]
        if gridsync:
            def kernel(grid, vectors, scalars, r_im, sim_box, step, conf_saver_params):
                pass
                return
            return cuda.jit(device=gridsync)(kernel)
        else:
            def kernel(grid, vectors, scalars, r_im, sim_box, step, conf_saver_params):
                pass
            return kernel


    def get_prestep_kernel(self, configuration, compute_plan, verbose=False):
        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']]
        num_blocks = (num_part - 1) // pb + 1
        sim_box_array_length = configuration.simbox.len_sim_box_data
        include_simbox = self.include_simbox

        # Unpack indices for scalars to be compiled into kernel  
        r_id, = [configuration.vectors.indices[key] for key in ['r', ]]

        def kernel(grid, vectors, scalars, r_im, sim_box, step, conf_saver_params):
            if include_simbox:
                conf_array, sim_box_output_array = conf_saver_params
            else:
                conf_array, = conf_saver_params

            Flag = False
            if step == 0:
                Flag = True
                save_index = 0
            else:
                b = np.int32(math.log2(np.float32(step)))
                c = 2 ** b
                if step == c:
                    Flag = True
                    save_index = b + 1

            if Flag:
                global_id, my_t = cuda.grid(2)
                if global_id < num_part and my_t == 0:
                    for k in range(D):
                        conf_array[save_index, 0, global_id, k] = vectors[r_id][global_id, k]
                        conf_array[save_index, 1, global_id, k] = np.float32(r_im[global_id, k])
                    if include_simbox and global_id == 0:
                        for k in range(sim_box_array_length):
                            sim_box_output_array[save_index, k] = sim_box[k]
            return

        kernel = cuda.jit(device=gridsync)(kernel)

        if gridsync:
            return kernel  # return device function
        else:
            return kernel[num_blocks, (pb, 1)]  # return kernel, incl. launch parameters
