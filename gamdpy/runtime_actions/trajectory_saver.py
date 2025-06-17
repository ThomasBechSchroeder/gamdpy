import numpy as np
import numba
import math
from numba import cuda, config

from .runtime_action import RuntimeAction


class TimeScheduler():

    def __init__(self, schedule='log2', **kwargs):

        self.schedule = schedule
        self._kwargs = kwargs

    def _setup(self, stepmax):
        """
        This is necessary aside from __init__ because in TrajectorySaver
        `steps_per_timeblock` is initialised only in a `setup` method
        """
        # `stepmax` is by construction the same as `steps_per_timeblock` in TrajectorySaver
        self.stepmax = stepmax

        # keywords specific to selected schedule
        if self.schedule=='log2':
            self.base = self._kwargs.get('base', 2.0)
            self.stepcheck_func = self._get_stepcheck_log2()

        elif self.schedule=='log':
            self.base = self._kwargs.get('base', np.exp(1.0))
            self.stepcheck_func = self._get_stepcheck_log()

        elif self.schedule=='lin':
            self.npoints = self._kwargs.get('npoints', None)
            assert type(self.npoints)==int, 'Invalid number of points'
            self.stepcheck_func = self._get_stepcheck_lin()

        elif self.schedule=='geom':
            self.npoints = self._kwargs.get('npoints', None)
            assert type(self.npoints)==int, 'Invalid number of points'
            self.stepcheck_func = self._get_stepcheck_geom()

        # TODO: "... but never is probably better than RIGHT now"
        # elif self.schedule=='custom':
        #     self.steps = self._kwargs.get('steps', None)
        #     assert self.steps is not None, 'Expected custom list of steps'
        #     try:
        #         self.steps = list(np.array(self.steps, dtype=np.int32))
        #     except:
        #         raise ValueError('Steps must be convertible to list')

    def _get_stepcheck_log2(self):
        def stepcheck(step):
            Flag = False
            save_index = -1 # this is for python calls of the function
            if step == 0:
                Flag = True
                save_index = 0
            else:
                b = np.int32(math.log2(np.float32(step)))
                c = 2 ** b
                if step == c:
                    Flag = True
                    save_index = b + 1
            return Flag, save_index
        return stepcheck

    def _get_stepcheck_log(self):
        base = self.base
        def stepcheck(step):
            Flag = False
            save_index = -1 # this is for python calls of the function
            if step == 0:
                Flag = True
                save_index = 0
            else:
                exponent = np.int32(math.log(np.float32(step)) / math.log(base))
                c = np.int32(base ** exponent)
                if abs(step-c)==1: # this seems to be working
                    Flag = True
                    save_index = exponent
            return Flag, save_index
        return stepcheck

    # TODO
    def _get_stepcheck_lin(self):
        # variables
        def stepcheck(step):
            if step == 0:
                Flag = True
                save_index = 0
            else:
                pass
            return Flag, save_index
        return stepcheck

    # TODO
    def _get_stepcheck_geom(self):
        # variables
        def stepcheck(step):
            if step == 0:
                Flag = True
                save_index = 0
            else:
                pass
            return Flag, save_index
        return stepcheck

    @property
    def savesteps(self):
        try:
            stepmax = self.stepmax
        except AttributeError:
            print('TimeScheduler was not able to setup steps, probably _setup() has not been called yet')
        steps = []
        for step in range(stepmax+1):
            Flag, _ = self.stepcheck_func(step)
            if Flag: steps.append(step)
        if stepmax not in steps: steps.append(stepmax)
        return steps

    @property
    def nsaves(self):
        try:
            return len(self.savesteps)
        except AttributeError:
            print('TimeScheduler was not able to setup steps, probably _setup() has not been called yet')

    # def stepcheck_old(self, step):
    #     Flag = False
    #     if step == 0:
    #         Flag = True
    #         save_index = 0
    #     else:
    #         b = np.int32(math.log2(np.float32(step)))
    #         c = 2 ** b
    #         if step == c:
    #             Flag = True
    #             save_index = b + 1
    #     return Flag, save_index

class TrajectorySaver(RuntimeAction):
    """ 
    Runtime action for saving configurations during timeblock.
    Does logarithmic saving.
    """

    def __init__(self, scheduler=None, include_simbox=False, verbose=False, compression="gzip", compression_opts=4) -> None:
        if isinstance(scheduler, TimeScheduler):
            # in this case the user must have set up the scheduler
            self.time_scheduler = scheduler
        else:
            # fallback option is log, with base 2 handled by the scheduler
            self.time_scheduler = TimeScheduler(schedule='log')

        self.include_simbox = include_simbox
        self.num_vectors = 2  # 'r' and 'r_im' (for now!)
        self.compression = compression
        if self.compression == 'gzip':
            self.compression_opts = compression_opts
        else:
            self.compression_opts = None
        #self.sid = {"r":0, "r_im":1}

        if isinstance(scheduler, TimeScheduler):
            # in this case the user must have set up the scheduler
            self.time_scheduler = scheduler
        else:
            # fallback option is log, with base 2 handled by the scheduler
            self.time_scheduler = TimeScheduler(schedule='log')

    def setup(self, configuration, num_timeblocks: int, steps_per_timeblock: int, output, verbose=False) -> None:
        self.configuration = configuration

        if type(num_timeblocks) != int or num_timeblocks < 0:
            raise ValueError(f'num_timeblocks ({num_timeblocks}) should be non-negative integer.')
        self.num_timeblocks = num_timeblocks
        
        if type(steps_per_timeblock) != int or steps_per_timeblock < 0:
            raise ValueError(f'steps_per_timeblock ({steps_per_timeblock}) should be non-negative integer.')
        self.steps_per_timeblock = steps_per_timeblock

        # pass the number of steps to the scheduler
        # without this line the scheduler does nothing at all
        self.time_scheduler._setup(stepmax=self.steps_per_timeblock)

        # both steps '0' and the last one are already counted by the scheduler
        self.conf_per_block = self.time_scheduler.nsaves# + 1 
        #self.conf_per_block = int(math.log2(self.steps_per_timeblock)) + 2  # Should be user controllable
        
        # Setup output
        if verbose:
            print(f'Storing results in memory. Expected footprint {self.num_timeblocks * self.conf_per_block * self.num_vectors * self.configuration.N * self.configuration.D * 4 / 1024 / 1024:.2f} MB.')

        if 'trajectory_saver' in output.keys():
            del output['trajectory_saver']
        output.create_group('trajectory_saver')

        # Compression has a different syntax depending if is gzip or not because gzip can have also a compression_opts
        # it is possible to use compression=None for not compressing the data
        output.create_dataset('trajectory_saver/positions',
                shape=(self.num_timeblocks, self.conf_per_block, self.configuration.N, self.configuration.D),
                chunks=(1, 1, self.configuration.N, self.configuration.D),
                dtype=np.float32, compression=self.compression, compression_opts=self.compression_opts)
        output.create_dataset('trajectory_saver/images',
                shape=(self.num_timeblocks, self.conf_per_block, self.configuration.N, self.configuration.D),
                chunks=(1, 1, self.configuration.N, self.configuration.D),
                dtype=np.int32,  compression=self.compression, compression_opts=self.compression_opts)
        output['trajectory_saver'].attrs['compression_info'] = f"{self.compression} with opts {self.compression_opts}"

        #output.attrs['vectors_names'] = list(self.sid.keys())
        output.attrs['savesteps'] = self.time_scheduler.savesteps
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

        # get function to check steps in the kernel, already compiled
        stepcheck_function = numba.njit(getattr(self.time_scheduler, 'stepcheck_func'))

        def kernel(grid, vectors, scalars, r_im, sim_box, step, conf_saver_params):
            if include_simbox:
                conf_array, sim_box_output_array = conf_saver_params
            else:
                conf_array, = conf_saver_params

            Flag, save_index = stepcheck_function(step)

            # Flag, save_index = False, 0

            # Flag = False
            # if step == 0:
            #     Flag = True
            #     save_index = 0
            # else:
            #     b = np.int32(math.log2(np.float32(step)))
            #     c = 2 ** b
            #     if step == c:
            #         Flag = True
            #         save_index = b + 1

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
