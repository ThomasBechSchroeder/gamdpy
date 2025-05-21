from numba import cuda
from abc import ABC, abstractmethod
from gamdpy import Configuration
from typing import Callable


class RuntimeAction(ABC):
    """
    Abstract Base Class specifying the requirements for a runtime_action, i.e. an action to compiled into to innner MD kernel
    """

    def get_compute_flags(self):
        return None

    def setup(self, configuration: Configuration, num_timeblocks: int, steps_per_timeblock: int, output, verbose=False) -> None:
        pass

    @abstractmethod   
    def get_prestep_kernel(self, configuration: Configuration, compute_plan: dict) -> Callable:
        """
        Get a kernel (or python function depending on compute_plan["gridsync"]) that implements the runtime_action.
        The generated kernel is called after evaluation of interactions, before intergration step is performed, see class Simulation
        """

        pass

    @abstractmethod   
    def get_poststep_kernel(self, configuration: Configuration, compute_plan: dict) -> Callable:
        """
        Get a kernel (or python function depending on compute_plan["gridsync"]) that implements the runtime_action
        The generated kernel is called immediately after evaluation intergration step is performed, see class Simulation
        """

        pass

    @abstractmethod
    def get_params(self, configuration: Configuration, compute_plan: dict) -> tuple :
        """
        Get a tuple with the parameters expected by the associated kernel
        """

        pass

    def update_at_end_of_timeblock(self, timeblock: int, output_reference):
        """
        Method to be called at the end of a timeblock, for e.g. saving data to a file if needed
        """

        pass
   
    def initialize_before_timeblock(self, timeblock: int, output_reference):
        """
        Method to be called before each timeblock 
        """

        pass

def merge_runtime_actions(configuration: Configuration, prestep_kernelA: Callable, poststep_kernelA: Callable, paramsA: tuple, actionB: RuntimeAction, compute_plan: dict) -> tuple[Callable, Callable, tuple] :
    paramsB = actionB.get_params(configuration, compute_plan)
    prestep_kernelB = actionB.get_prestep_kernel(configuration, compute_plan)
    poststep_kernelB = actionB.get_poststep_kernel(configuration, compute_plan)

    if compute_plan['gridsync']:
        # A device function, calling a number of device functions, using gridsync to syncronize
        @cuda.jit( device=compute_plan['gridsync'])
        def prestep_kernel(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params):
            prestep_kernelA(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params[0])
            grid.sync() # Not always necessary !!!
            prestep_kernelB(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params[1])
            return

        @cuda.jit( device=compute_plan['gridsync'])
        def poststep_kernel(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params):
            poststep_kernelA(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params[0])
            grid.sync() # Not always necessary !!!
            poststep_kernelB(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params[1])
            return
        
        return prestep_kernel, poststep_kernel, (paramsA, paramsB, )
    else:
        # Two python function, making several kernel calls to syncronize
        def prestep_kernel(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params):
            prestep_kernelA(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params[0])
            prestep_kernelB(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params[1])
            return

        def poststep_kernel(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params):
            poststep_kernelA(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params[0])
            poststep_kernelB(grid, vectors, scalars, r_im, sim_box, step, runtime_actions_params[1])
            return
        
        return prestep_kernel, poststep_kernel, (paramsA, paramsB, )   

def add_runtime_actions_list(configuration: Configuration, runtime_actions_list: list[RuntimeAction], compute_plan: dict, verbose: bool = False) -> tuple[Callable, Callable, tuple]:

    # Setup first interaction and cuda.jit it if gridsync is used for syncronization
    params = runtime_actions_list[0].get_params(configuration, compute_plan)
    prestep_kernel = runtime_actions_list[0].get_prestep_kernel(configuration, compute_plan)
    poststep_kernel = runtime_actions_list[0].get_poststep_kernel(configuration, compute_plan)

    if compute_plan['gridsync']:
        prestep_kernel: Callable = cuda.jit( device=compute_plan['gridsync'] )(prestep_kernel)
        poststep_kernel: Callable = cuda.jit( device=compute_plan['gridsync'] )(poststep_kernel)

    # Merge in the rest of the runtime_actions (maximum recursion depth might set a maximum for number of interactions)
    for i in range(1, len(runtime_actions_list)):
        prestep_kernel, poststep_kernel, params = merge_runtime_actions(configuration, prestep_kernel, poststep_kernel, params, runtime_actions_list[i], compute_plan)

    return prestep_kernel, poststep_kernel, params
