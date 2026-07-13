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

    def setup(self, simulation, num_timeblocks : int, steps_per_timeblock : int, output: h5py.File,
            update_ptype: bool=False, update_topology: bool=False, verbose: bool=False) -> None:

        
        self.simulation = simulation
        self.configuration = simulation.configuration
        self.update_ptype = update_ptype
        self.update_topology = update_topology

        # Setup output
        if 'restarts' in output.keys():
            del output['restarts']
        grp = output.create_group('restarts')
        grp.attrs['timeblocks_between_restarts'] = self.timeblocks_between_restarts

    def get_params(self, configuration, compute_plan):
        return (0,)

    def initialize_before_timeblock(self,  timeblock: int, output_reference):
        group_name_this_restart = f"/restarts/restart{timeblock:04d}"
        self.configuration.save(output=output_reference, group_name=group_name_this_restart, mode="w", 
                                update_ptype=self.update_ptype, use_topology_link=not self.update_topology, verbose=False)
        self.simulation.integrator.save_internal_state(output_reference, group_name_this_restart)

    def update_at_end_of_timeblock(self, timeblock: int, output_reference):
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
