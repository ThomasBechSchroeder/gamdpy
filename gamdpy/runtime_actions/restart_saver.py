import numpy as np
import numba
import math
from numba import cuda, config
import h5py

from .runtime_action import RuntimeAction

class RestartSaver(RuntimeAction):
    """ 
    Runtime action for saving restarts, ie. the current configuration at beginning of every timebock .
    """

    def __init__(self, timeblocks_between_restarts=1) -> None:
        # Later: Give user influence on what and how often is saved
        self.timeblocks_between_restarts = timeblocks_between_restarts

    def setup(self, configuration, num_timeblocks:int, steps_per_timeblock:int, output, verbose=False) -> None:

        self.configuration = configuration

        # Setup output
        if 'restarts' in output.keys():
            del output['restarts']
        grp = output.create_group('restarts')
        grp.attrs['timeblocks_between_restarts'] = self.timeblocks_between_restarts

     
    def get_params(self, configuration, compute_plan):
        return (0,)
    
    def initialize_before_timeblock(self,  timeblock: int, output_reference):
        self.configuration.save(output=output_reference, group_name=f"/restarts/restart{timeblock:04d}", mode="w", include_topology=True)


    def update_at_end_of_timeblock(self,  timeblock: int, output_reference):
        pass

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
