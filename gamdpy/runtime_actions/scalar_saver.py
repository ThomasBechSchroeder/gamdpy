import numpy as np
import numba
import math
from numba import cuda, config
import h5py

from .runtime_action import RuntimeAction

class ScalarSaver(RuntimeAction):
    """ 
    Runtime action for saving scalar data (such as thermodynamic properties) during a timeblock
    every `steps_between_output` time steps.
    """

    def __init__(self, steps_between_output:int = 16, compute_flags = None, verbose=False, compression="gzip", compression_opts=4) -> None:

        if type(steps_between_output) != int or steps_between_output < 0:
            raise ValueError(f'steps_between_output ({steps_between_output}) should be non-negative integer.')
        self.steps_between_output = steps_between_output

        self.compute_flags = compute_flags
        self.compression = compression
        if self.compression == 'gzip':
            self.compression_opts = compression_opts
        else:
            self.compression_opts = None

    def get_compute_flags(self):
        return self.compute_flags

    def setup(self, configuration, num_timeblocks:int, steps_per_timeblock:int, output, verbose=False) -> None:

        self.configuration = configuration

        if type(num_timeblocks) != int or num_timeblocks < 0:
            raise ValueError(f'num_timeblocks ({num_timeblocks}) should be non-negative integer.')
        self.num_timeblocks = num_timeblocks

        if type(steps_per_timeblock) != int or steps_per_timeblock < 0:
            raise ValueError(f'steps_per_timeblock ({steps_per_timeblock}) should be non-negative integer.')
        self.steps_per_timeblock = steps_per_timeblock

        if self.steps_between_output >= steps_per_timeblock:
            raise ValueError(f'scalar_output ({self.steps_between_output}) must be less than steps_per_timeblock ({steps_per_timeblock})')

        # per block saving of scalars
        compute_flags = configuration.compute_flags
        self.num_scalars = 0
        sid_list = ['U', 'W', 'lapU', 'K', 'Fsq', 'Vol']
        self.sid = {}
        for item in sid_list:
            if compute_flags[item]:
                self.sid[item] = self.num_scalars
                self.num_scalars += 1

        if compute_flags['Ptot']:
            self.sid['Px'] = self.num_scalars
            self.sid['Py'] = self.num_scalars + 1
            self.sid['Pz'] = self.num_scalars + 2
            self.num_scalars += 3

        if compute_flags['stresses']:
            self.sid['Sxy'] = self.num_scalars
            self.num_scalars += 1

        self.scalar_saves_per_block = self.steps_per_timeblock//self.steps_between_output

        # Setup output
        shape = (self.num_timeblocks, self.scalar_saves_per_block, self.num_scalars)
        if 'scalar_saver' in output.keys():
            del output['scalar_saver']
        output.create_group('scalar_saver')

        # Compression has a different syntax depending if is gzip or not because gzip can have also a compression_opts
        # it is possible to use compression=None for not compressing the data
        output.create_dataset('scalar_saver/scalars', shape=shape,
                chunks=(1, self.scalar_saves_per_block, self.num_scalars),
                dtype=np.float32, compression=self.compression, compression_opts=self.compression_opts)
        output['scalar_saver'].attrs['compression_info'] = f"{self.compression} with opts {self.compression_opts}"

        output['scalar_saver'].attrs['steps_between_output'] = self.steps_between_output
        output['scalar_saver'].attrs['scalar_names'] = list(self.sid.keys())

        flag = config.CUDA_LOW_OCCUPANCY_WARNINGS
        config.CUDA_LOW_OCCUPANCY_WARNINGS = False
        self.zero_kernel = self.make_zero_kernel()
        config.CUDA_LOW_OCCUPANCY_WARNINGS = flag

    def make_zero_kernel(self):

        def zero_kernel(array):
            Nx, Ny = array.shape
            #i, j = cuda.grid(2) # doing simple 1 thread kernel for now ...
            for i in range(Nx):
                for j in range(Ny):
                    array[i,j] = numba.float32(0.0)

        zero_kernel = cuda.jit(zero_kernel)
        return zero_kernel[1,1]
     
    def get_params(self, configuration, compute_plan):
        
        self.output_array = np.zeros((self.scalar_saves_per_block, self.num_scalars), dtype=np.float32)
        self.d_output_array = cuda.to_device(self.output_array)
        self.params = (self.steps_between_output, self.d_output_array)
        return self.params
    
    def initialize_before_timeblock(self, timeblock: int, output_reference):
        self.zero_kernel(self.d_output_array)

    def update_at_end_of_timeblock(self,  timeblock: int, output_reference):
        output_reference['scalar_saver/scalars'][timeblock, :] = self.d_output_array.copy_to_host()

    def get_prestep_kernel(self, configuration, compute_plan, verbose=False):
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

    def get_poststep_kernel(self, configuration, compute_plan):
        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        num_blocks = (num_part - 1) // pb + 1
        
        # Below is three alternatives of how to unpack parameters for use in the kernel

#        Alternative 1. Requires renaming of some variables
#        print(configuration.compute_flags)
#        for key in configuration.compute_flags.keys():
#            if key == "stresses" and configuration.compute_flags[key]:
#                sx_id = configuration.vectors.indices['sx']
#            elif configuration.compute_flags[key]:
#                print(key, configuration.compute_flags[key], f'{key}_id', self.sid[key], f'compute_{key}')
#                globals()[f'compute_{key}'] = configuration.compute_flags[key]
#                globals()[f'{key}_id'] = self.sid[key]
#        print(compute_U)


        # Alternative 2. 'None' used as index for scalars that are not there, and therefore shouldn't be used (kernel throws an getitem error)  
        unpacked_compute_flags = [configuration.compute_flags[key] for key in ['U', 'W', 'lapU', 'Fsq', 'K', 'Vol', 'Ptot', 'stresses']]
        compute_u, compute_w, compute_lap, compute_fsq, compute_k, compute_vol, compute_Ptot, compute_stresses = unpacked_compute_flags

        unpacked_scalar_indicies = [self.sid.get(key, None) for key in ['U', 'W', 'lapU', 'Fsq', 'K', 'Vol', 'Px', 'Py', 'Pz', 'Sxy']]
        u_id, w_id, lap_id, fsq_id, k_id, vol_id, Px_id, Py_id, Pz_id, Sxy_id = unpacked_scalar_indicies

        # Original
        #compute_u = configuration.compute_flags['U']
        #compute_w = configuration.compute_flags['W']
        #compute_lap = configuration.compute_flags['lapU']
        #compute_fsq = configuration.compute_flags['Fsq']
        #compute_k = configuration.compute_flags['K']
        #compute_vol = configuration.compute_flags['Vol']
        #compute_Ptot = configuration.compute_flags['Ptot']
        #compute_stresses = configuration.compute_flags['stresses']
        
    
        # Unpack indices for scalars to be compiled into kernel
        #if compute_u:
        #    u_id = self.sid['U']

        #if compute_k:
        #    k_id = self.sid['K']
        #if compute_w:
        #    w_id = self.sid['W']

        #if compute_fsq:
        #    fsq_id = self.sid['Fsq']

        #if compute_lap:
        #    lap_id = self.sid['lapU']

        #if compute_vol:
        #    vol_id = self.sid['Vol']

        #if compute_Ptot:
        #    Px_id = self.sid['Px']
        #    Py_id = self.sid['Py']
        #    Pz_id = self.sid['Pz']

        #if compute_stresses:
        #    Sxy_id = self.sid['Sxy']

        m_id = configuration.sid['m']


        v_id = configuration.vectors.indices['v']
        if compute_stresses:
            sx_id = configuration.vectors.indices['sx']

        volume_function = numba.njit(configuration.simbox.get_volume_function())

        def kernel(grid, vectors, scalars, r_im, sim_box, step, runtime_action_params):
            """     
            """
            steps_between_output, output_array = runtime_action_params # Needs to be compatible with get_params above
            if step%steps_between_output==0:
                save_index = step//steps_between_output
            
                global_id, my_t = cuda.grid(2)
                if global_id < num_part and my_t == 0:
                    if compute_u:
                        cuda.atomic.add(output_array, (save_index, u_id), scalars[global_id][u_id])   # Potential energy
                    if compute_w:
                        cuda.atomic.add(output_array, (save_index, w_id), scalars[global_id][w_id])   # Virial
                    if compute_lap:
                        cuda.atomic.add(output_array, (save_index, lap_id), scalars[global_id][lap_id]) # Laplace
                    if compute_fsq:
                        cuda.atomic.add(output_array, (save_index, fsq_id), scalars[global_id][fsq_id]) # F**2
                    if compute_k:
                        cuda.atomic.add(output_array, (save_index, k_id), scalars[global_id][k_id])   # Kinetic energy

                    # Contribution to total momentum
                    if compute_Ptot or compute_stresses:
                        my_m = scalars[global_id][m_id]
                    if compute_Ptot:
                        cuda.atomic.add(output_array, (save_index, Px_id), my_m*vectors[v_id][global_id][0])
                        cuda.atomic.add(output_array, (save_index, Py_id), my_m*vectors[v_id][global_id][1])
                        cuda.atomic.add(output_array, (save_index, Pz_id), my_m*vectors[v_id][global_id][2])

                    if compute_stresses:
                        # XY component of stress only for now
                        cuda.atomic.add(output_array, (save_index, Sxy_id), vectors[sx_id][global_id][1] -
                                        my_m * vectors[v_id][global_id][0]*vectors[v_id][global_id][1])

                if compute_vol and global_id == 0 and my_t == 0:
                    output_array[save_index][vol_id] = volume_function(sim_box)

            return
        
        kernel = cuda.jit(device=gridsync)(kernel)

        if gridsync:
            return kernel  # return device function
        else:
            return kernel[num_blocks, (pb, 1)]  # return kernel, incl. launch parameters
        
    # Class functions to read data

    def info(h5file):
        h5grp = h5file['scalar_saver']
        str = f"\tscalar_names: {h5grp.attrs['scalar_names']}"
        str += f"\n\tscalars, shape: {h5grp['scalars'].shape}, dtype: {h5grp['scalars'].dtype}"
        return str

    def columns(h5file):
        return list(h5file['scalar_saver'].attrs['scalar_names'])

    def extract(h5file, columns, per_particle=True, first_block=0, last_block=None, subsample=1, function=None):
        _, N, D = h5file['initial_configuration']['vectors'].shape

        h5grp = h5file['scalar_saver']
        scalar_names = list(h5grp.attrs['scalar_names'])

        output = []
        for column in columns: 
            index = scalar_names.index(column)
            data = np.ravel(h5grp['scalars'][first_block:last_block,:,index])[::subsample]
            if per_particle:
                data /= N

            if function:
                data = function(data)
        
            output.append(data)
        return output
        
    def get_times(h5file, first_block=0, last_block=-1, reset_time=True, subsample=1):
        num_timeblock, saves_per_timeblock = h5file['scalar_saver']['scalars'][first_block:last_block,:,0].shape
        times_array = np.arange(0,num_timeblock*saves_per_timeblock, step=subsample)*h5file.attrs['dt']
        return times_array
