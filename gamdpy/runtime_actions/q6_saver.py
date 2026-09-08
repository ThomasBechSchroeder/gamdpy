import numpy as np
import numba
import math
from numba import cuda, config
import h5py

from .runtime_action import RuntimeAction
from .time_scheduler import Lin

class Q6_Saver(RuntimeAction):
    """
    Runtime action for saving scalar data (such as thermodynamic properties) during a timeblock
    every `steps_between_output` time steps.
    """

    def __init__(self, pot_steinhardt_Q6, steps_between_output:int = 0, compute_flags = None, verbose=False, compression="gzip", compression_opts=4) -> None:

        self.pot_steinhardt_Q6 = pot_steinhardt_Q6

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


        self.num_Q6_items = 2 # Q6 and energy
        self.Q6_saves_per_block = self.steps_per_timeblock//self.steps_between_output

        # Setup output
        shape = (self.num_timeblocks, self.Q6_saves_per_block, self.num_Q6_items)
        if 'Q6' in output.keys():
            del output['Q6']
        output.create_group('Q6')

        # Compression has a different syntax depending if is gzip or not because gzip can have also a compression_opts
        # it is possible to use compression=None for not compressing the data
        output.create_dataset('Q6/Q6', shape=shape,
                chunks=(1, self.Q6_saves_per_block, self.num_Q6_items),
                dtype=np.float32, compression=self.compression, compression_opts=self.compression_opts)
        output['Q6'].attrs['compression_info'] = f"{self.compression} with opts {self.compression_opts}"
        output['Q6'].attrs['steps_between_output'] = self.steps_between_output

        # Setup scheduler, and write the relevant information to the h5 file
        self.scheduler.setup(stepmax=self.steps_per_timeblock, ntimeblocks=self.num_timeblocks)
        self.scheduler.info_to_h5(output['Q6'])




    def get_params(self, configuration, compute_plan):

        self.output_array = np.zeros((self.Q6_saves_per_block, self.num_Q6_items), dtype=np.float32)
        self.d_output_array = cuda.to_device(self.output_array)
        self.d_switch_sum_Q6_energy_array = self.pot_steinhardt_Q6.d_switch_sum_Q6_energy
        self.params = (self.steps_between_output, self.d_output_array, self.d_switch_sum_Q6_energy_array)
        return self.params

    def initialize_before_timeblock(self, timeblock: int, output_reference):
        pass
        #self.zero_kernel(self.d_output_array)

    def update_at_end_of_timeblock(self,  timeblock: int, output_reference):
        output_reference['Q6/Q6'][timeblock, :] = self.d_output_array.copy_to_host()

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



        def kernel(grid, vectors, scalars, r_im, sim_box, step, runtime_action_params):
            """
            """
            steps_between_output, output_array, switch_sum_Q6_energy = runtime_action_params # Needs to be compatible with get_params above
            if step%steps_between_output==0:
                save_index = step//steps_between_output

                global_id, my_t = cuda.grid(2)
                if global_id == 0 and my_t == 0:
                    output_array[save_index, 0] = switch_sum_Q6_energy[1]
                    output_array[save_index, 1] = switch_sum_Q6_energy[2]

            return

        kernel = cuda.jit(device=gridsync)(kernel)

        if gridsync:
            return kernel  # return device function
        else:
            return kernel[num_blocks, (pb, 1)]  # return kernel, incl. launch parameters

    # Class functions to read data



    def extract(h5file: h5py.File, first_block: int=0, last_block: int=None, subsample: int=1) -> list:
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

        h5grp = h5file['Q6']

        output = []
        for index in range(2):
            data = np.ravel(h5grp['Q6'][first_block:last_block,:,index])[::subsample]

            output.append(data)
        return output

 

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

        """

        num_timeblock, saves_per_timeblock = h5file['Q6/Q6'][first_block:last_block,:,0].shape
        times_array = np.arange(0,num_timeblock*saves_per_timeblock, step=subsample, dtype=float) 
        if not reset_time:
            times_array += first_block * saves_per_timeblock
        times_array *= float(h5file['Q6'].attrs['steps_between_output']) * h5file.attrs['dt']
        return times_array
