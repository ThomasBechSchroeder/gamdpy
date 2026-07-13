import numpy as np
import numba
import math
from numba import cuda, config
import json
import h5py

from .runtime_action import RuntimeAction
from .time_scheduler import Lin

class ScalarSaver(RuntimeAction):
    """
    Runtime action for saving scalar data (such as thermodynamic properties) during a timeblock
    every `steps_between_output` time steps.
    """

    def __init__(self, steps_between_output:int = 0, compute_flags = None, verbose=False, compression="gzip", compression_opts=4) -> None:

        # For now only to put the scheduler information in the h5 file
        self.scheduler = Lin(steps_between=steps_between_output) 

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

    def setup(self, simulation, num_timeblocks:int, steps_per_timeblock:int, output, verbose=False) -> None:

        self.simulation = simulation
        self.configuration = self.simulation.configuration

        if type(num_timeblocks) != int or num_timeblocks < 0:
            raise ValueError(f'num_timeblocks ({num_timeblocks}) should be non-negative integer.')
        self.num_timeblocks = num_timeblocks

        if type(steps_per_timeblock) != int or steps_per_timeblock < 0:
            raise ValueError(f'steps_per_timeblock ({steps_per_timeblock}) should be non-negative integer.')
        self.steps_per_timeblock = steps_per_timeblock

        if self.steps_between_output > steps_per_timeblock:
            raise ValueError(f'scalar_output ({self.steps_between_output}) must be less than or equal to steps_per_timeblock ({steps_per_timeblock})')

        # per block saving of scalars
        compute_flags = self.configuration.compute_flags
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
        if 'scalars' in output.keys():
            del output['scalars']
        output.create_group('scalars')

        # Compression has a different syntax depending if is gzip or not because gzip can have also a compression_opts
        # it is possible to use compression=None for not compressing the data
        output.create_dataset('scalars/scalars', shape=shape,
                chunks=(1, self.scalar_saves_per_block, self.num_scalars),
                dtype=np.float32, compression=self.compression, compression_opts=self.compression_opts)
        output['scalars'].attrs['compression_info'] = f"{self.compression} with opts {self.compression_opts}"
        output['scalars'].attrs['steps_between_output'] = self.steps_between_output # LC: This should be removed because it's above already
        output['scalars'].attrs['scalar_columns'] = list(self.sid.keys())

        # Setup scheduler, and write the relevant information to the h5 file
        self.scheduler.setup(stepmax=self.steps_per_timeblock, ntimeblocks=self.num_timeblocks)
        self.scheduler.info_to_h5(output['scalars'])
        
        # Scheduler info
        #output['scalars'].attrs['scheduler'] = 'Lin' #self.scheduler.__class__.__name__
        #output['scalars'].attrs['scheduler_info'] = json.dumps({'Dt':self.steps_between_output}) #json.dumps(self.scheduler.kwargs)
        #output['scalars'].attrs['scheduler'] = self.scheduler.__class__.__name__
        #output['scalars'].attrs['scheduler_info'] = json.dumps(self.scheduler.kwargs)
        #output['scalars'].create_dataset('steps', data=self.scheduler.steps, dtype=np.int32)

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
        output_reference['scalars/scalars'][timeblock, :] = self.d_output_array.copy_to_host()

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

    def info(h5file: h5py.File) -> str:
        """ Get information about what data was saved by ScalarSaver in a .h5 file

        Parameters
        ----------
        h5file h5py.File: 
            HDF5 file object from which data is read

        Returns
        -------
        str : A string containing information about the available scalars, and the associated shape.

        """

        h5grp = h5file['scalars']
        str = f"\tscalar_columns: {h5grp.attrs['scalar_columns']}"
        str += f"\n\tscalars, shape: {h5grp['scalars'].shape}, dtype: {h5grp['scalars'].dtype}"
        return str

    def columns(h5file: h5py.File) -> list[str]:
        """ Get a list of available data columns saved by ScalarSaver in a .h5 file
        
        Parameters
        ----------
        h5file h5py.File: 
            HDF5 file object from which data is read

        Returns
        -------
        list(str) : A list of available data column keys 
        
        """

        return list(h5file['scalars'].attrs['scalar_columns'])

    def extract(h5file: h5py.File, columns: list[str], per_particle: bool=True, 
                first_block: int=0, last_block: int=None, subsample: int=1, function: callable=None) -> list:
        """ Get a tuple of lists of time series of available data columns saved in a .h5 file
        
        Parameters
        ----------

        h5file : h5py.File
            HDF5 file object from which data will be read

        columns : list[str]
            List of keys for data columns to be extracted, eg. ['U', 'K',]
            
        per_particle : bool
            Bolean flag determining whether date should be returned divided by number of particles (default True)

        first_block : int
            First timeblock to include in returned data (default 0) 
        
        last_block : int or None
            last timeblock to include in returned data 
            (default None, i.e. include last available timeblock)
        
        subsample : int 
            If '2' return every second entry in saved time-series, etc (default 1)  
        
        function : callable  
            function applied to each time series data before returning, e.g.: np.mean (default None)

        Returns
        -------
        list : A list numpy arrays, one for each column asked for - or the results of applying 'function' to these 
        
        """

        _, N, D = h5file['initial_configuration']['vectors'].shape

        h5grp = h5file['scalars']
        scalar_columns = list(h5grp.attrs['scalar_columns'])

        output = []
        for column in columns:
            index = scalar_columns.index(column)
            data = np.ravel(h5grp['scalars'][first_block:last_block,:,index])[::subsample]
            if per_particle:
                data /= N

            if function:
                data = function(data)

            output.append(data)
        return output

    def extract_as_dict(*args, **kwargs):
        """ Get a dictionary of time series of available data columns saved in a .h5 file

        Same as :func:`ScalarSaver.extract`, but returns a dictionary.
        See :func:`ScalarSaver.extract` for more details on the arguments.

        If 'columns' is not specified in kwargs, all available columns are returned.

        """

        if 'columns' not in kwargs.keys():
            h5file = args[0]
            h5grp = h5file['scalars']
            kwargs['columns'] = list(h5grp.attrs['scalar_columns'])
            # print(kwargs['columns'])

        output = ScalarSaver.extract(*args, **kwargs)
        return dict(zip(kwargs['columns'], output))

    def get_times(h5file, first_block=0, last_block=None, reset_time=True, subsample=1):
        """ Get a numpy array with times associated with data columns saved by ScalarSaver in a .h5 file
        
        Parameters
        ----------
        h5file : h5py.File
            HDF5 file object from which data will be read

        first_block : int
            First timeblock to include in returned data (default 0) 
        
        last_block : int ot None
            last timeblock to include in returned data 
            (default None, i.e. include last available timeblock)

        reset_time : bool 
            Bolean flag determining whether the returned times should start from zero (default True)
        
        subsample : int 
            If '2' return every second entry in saved time-series, etc (1) 

        Returns
        -------
        np.array : A numpy arrays with simulation times
        
        Seed also
        ---------

        :function:`ScalarSaver.extract`
        
        """

        num_timeblock, saves_per_timeblock = h5file['scalars/scalars'][first_block:last_block,:,0].shape
        times_array = np.arange(0,num_timeblock*saves_per_timeblock, step=subsample, dtype=float) 
        if not reset_time:
            times_array += first_block * saves_per_timeblock
        times_array *= float(h5file['scalars'].attrs['steps_between_output']) * h5file.attrs['dt']
        return times_array
