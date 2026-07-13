import numpy as np
import numba
import math
from numba import cuda

# Abstract Base Class and type annotation
from .runtime_action import RuntimeAction
from gamdpy import Configuration


# Could include flags of dimensions to work on
class MomentumReset(RuntimeAction):
    """ 
    Runtime action that sets the total momentum of configuration to zero
    every `steps_between_reset` time step.
    """

    def __init__(self, steps_between_reset: int) -> None:
        if type(steps_between_reset) != int or steps_between_reset < 0:
            raise ValueError(f'steps_between_momentum_reset ({steps_between_reset}) should be non-negative integer.')
        self.steps_between_reset = steps_between_reset

    def setup(self, simulation, num_timeblocks: int, steps_per_timeblock: int, output, verbose=False) -> None:
        pass

    def get_params(self, configuration: Configuration, compute_plan: dict) -> tuple:
        self.total_momentum = np.zeros(configuration.D+1, dtype=np.float32) # Total mass summed in last index of total_momentum
        self.d_total_momentum = cuda.to_device(self.total_momentum)
        return (self.d_total_momentum, ) # return parameters as a tuple

    def get_prestep_kernel(self, configuration: Configuration, compute_plan: dict):

        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']]
        if gridsync:
            def kernel(grid, vectors, scalars, r_im, sim_box, step, momentum_reset_params):
                pass
                return
            return cuda.jit(device=gridsync)(kernel)
        else:
            def kernel(grid, vectors, scalars, r_im, sim_box, step, momentum_reset_params):
                pass
            return kernel

    def get_poststep_kernel(self, configuration: Configuration, compute_plan: dict):

        # Unpack parameters from configuration and compute_plan
        D, num_part = configuration.D, configuration.N
        pb, tp, gridsync = [compute_plan[key] for key in ['pb', 'tp', 'gridsync']] 
        num_blocks = (num_part - 1) // pb + 1

        # Unpack indices for vectors and scalars to be compiled into kernel
        v_id = configuration.vectors.indices['v']
        m_id = configuration.sid['m']

        steps_between_reset = self.steps_between_reset

        def zero_momentum(cm_velocity):
            global_id, my_t = cuda.grid(2)
            if global_id == 0 and my_t == 0:
                for k in range(D+1):
                    cm_velocity[k] = np.float32(0.)
            return
        
        def sum_momentum(vectors, scalars, cm_velocity):
            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                my_m = scalars[global_id][m_id]
                for k in range(D):
                    cuda.atomic.add(cm_velocity, k, my_m * vectors[v_id][global_id][k])
                cuda.atomic.add(cm_velocity, D, my_m) # Total mass summed in last index of cm_velocity
            return

        def shift_velocities(vectors, cm_velocity):
            global_id, my_t = cuda.grid(2)
            if global_id < num_part and my_t == 0:
                for k in range(D):
                    vectors[v_id][global_id,k] -= cm_velocity[k] / cm_velocity[D] 
            return

        zero_momentum = cuda.jit(device=gridsync)(zero_momentum)
        sum_momentum = cuda.jit(device=gridsync)(sum_momentum)
        shift_velocities = cuda.jit(device=gridsync)(shift_velocities)

        if gridsync:
            def kernel(grid, vectors, scalars, r_im, sim_box, step, momentum_reset_params):
                cm_velocity, = momentum_reset_params
                if step%steps_between_reset == 0:
                    zero_momentum(cm_velocity)
                    grid.sync()
                    sum_momentum(vectors, scalars, cm_velocity)
                    grid.sync()
                    shift_velocities(vectors, cm_velocity)
                return
            return cuda.jit(device=gridsync)(kernel)
        else:
            def kernel(grid, vectors, scalars, r_im, sim_box, step, momentum_reset_params):
                cm_velocity, = momentum_reset_params
                if step%steps_between_reset == 0:
                    zero_momentum[1, 1](cm_velocity)
                    sum_momentum[num_blocks, (pb, 1)](vectors, scalars, cm_velocity)
                    shift_velocities[num_blocks, (pb, 1)](vectors, cm_velocity)
                return
            return kernel

