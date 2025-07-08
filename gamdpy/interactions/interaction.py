import numba
from numba import cuda
from abc import ABC, abstractmethod
from gamdpy import Configuration
from typing import Callable

class Interaction(ABC):
    """
    Abstract Base Class specifying the requirements for an interaction
    """

    @abstractmethod   
    def get_kernel(self, configuration: Configuration, compute_plan: dict, compute_flags: dict[str,bool]) -> Callable:
        """
        Get a kernel (or python function depending on compute_plan["gridsync"]) that implements calculation of the interaction
        """

    @abstractmethod
    def get_params(self, configuration: Configuration, compute_plan: dict) -> tuple:
        """
        Get a tuple with the parameters expected by the associated kernel
        """

    def check_datastructure_validity(self) -> bool:
        """
        Interactions which have an internal data structure (think: PairPotential and its NbList) should overwrite this method 
        with one that checks the validity of it, and throws an error if not valid (Later: repair it and signal rerun of timeblock) 
        """
        return True

def merge_interactions(configuration: Configuration, kernelA: Callable, paramsA: tuple, interactionB: Interaction, compute_plan: dict, compute_flags: dict[str,bool]) -> tuple[Callable, tuple] :
    paramsB = interactionB.get_params(configuration, compute_plan)
    kernelB = interactionB.get_kernel(configuration, compute_plan, compute_flags) 

    if compute_plan['gridsync']:
        # A device function, calling a number of device functions, using gridsync to syncronize
        @cuda.jit( device=compute_plan['gridsync'])
        def interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            kernelA(grid, vectors, scalars, ptype, sim_box, interaction_parameters[0])
            grid.sync() # Not always necessary !!!
            kernelB(grid, vectors, scalars, ptype, sim_box, interaction_parameters[1])
            return
        return interactions, (paramsA, paramsB, )
    else:
        # A python function, making several kernel calls to syncronize  
        def interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            kernelA(0, vectors, scalars, ptype, sim_box, interaction_parameters[0])
            kernelB(0, vectors, scalars, ptype, sim_box, interaction_parameters[1])
            return
        return interactions, (paramsA, paramsB, )


def add_interactions_list(configuration: Configuration, interactions_list: list[Interaction], compute_plan: dict, compute_flags: dict[str,bool], verbose: bool = False) -> tuple[Callable, tuple]:

    # Setup first interaction and cuda.jit it if gridsync is used for syncronization
    params = get_initializer_params(configuration, compute_plan)
    kernel: Callable = get_initializer_kernel(configuration, compute_plan, compute_flags)
    if compute_plan['gridsync']:
        kernel: Callable  = cuda.jit( device=compute_plan['gridsync'] )(kernel)
    
    # Merge in the rest of the interaction (maximum recursion depth might set a maximum for number of interactions)
    for i in range(len(interactions_list)):
        kernel, params = merge_interactions(configuration, kernel, params, interactions_list[i], compute_plan, compute_flags)

    return kernel, params


def get_initializer_params(configuration, compute_plan):
    return (0,)


def get_initializer_kernel(configuration, compute_plan, compute_flags) -> Callable:

    num_cscalars = configuration.num_cscalars
    compute_stresses = compute_flags['stresses']

    # Unpack parameters from configuration and compute_plan
    D, num_part = configuration.D, configuration.N
    pb, tp, gridsync, UtilizeNIII = [compute_plan[key] for key in ['pb', 'tp', 'gridsync', 'UtilizeNIII']] 
    num_blocks = (num_part - 1) // pb + 1  

    # Unpack indices for vectors and scalars to be compiled into kernel
    f_id,  = [configuration.vectors.indices[key] for key in ['f']]

    if compute_stresses:
        sx_id = configuration.vectors.indices['sx']
        if D > 1:
            sy_id = configuration.vectors.indices['sy']
            if D > 2:
                sz_id = configuration.vectors.indices['sz']
                if D > 3:
                    sw_id = configuration.vectors.indices['sw']

    
    def kernel(grid, vectors, scalars, ptype, sim_box, interaction_parameters):

        global_id, my_t = cuda.grid(2)

        if global_id < num_part and my_t==0: 
            for k in range(num_cscalars):
                scalars[global_id, k] = numba.float32(0.0)


        if global_id < num_part and my_t==0: # Initializion of forces moved here to make NewtonIII possible 
            for k in range(D):
                vectors[f_id][global_id, k] = numba.float32(0.0)
                if  compute_stresses:
                    vectors[sx_id][global_id, k] =  numba.float32(0.0)
                    if D > 1:
                        vectors[sy_id][global_id, k] =  numba.float32(0.0)
                        if D > 2:
                            vectors[sz_id][global_id, k] =  numba.float32(0.0)
                            if D > 3:
                                vectors[sw_id][global_id, k] =  numba.float32(0.0)
        return
    

    if gridsync:
        # A device function, calling a number of device functions, using gridsync to syncronize
        return cuda.jit( device=gridsync )(kernel)
    else:
        return cuda.jit( device=gridsync )(kernel)[num_blocks, (pb, tp)]




# Function below not used and will be removed

def add_interactions_list_old(configuration, interactions_list, compute_plan, compute_flags, verbose=True,):
    gridsync = compute_plan['gridsync']
    num_interactions = len(interactions_list)
    assert 0 < num_interactions <= 5
    
    interaction_params_list = []
    for interaction in interactions_list:
        interaction_params_list.append(interaction.get_params(configuration, compute_plan, verbose=verbose))

    i0 = interactions_list[0].get_kernel(configuration, compute_plan, compute_flags, verbose=verbose)
    if num_interactions>1:
        i1 = interactions_list[1].get_kernel(configuration, compute_plan, compute_flags, verbose=verbose)
    if num_interactions>2:
        i2 = interactions_list[2].get_kernel(configuration, compute_plan, compute_flags, verbose=verbose)
    if num_interactions>3:
        i3 = interactions_list[3].get_kernel(configuration, compute_plan, compute_flags, verbose=verbose)
    if num_interactions>4:
        i4 = interactions_list[4].get_kernel(configuration, compute_plan, compute_flags, verbose=verbose)

    if gridsync:
        # A device function, calling a number of device functions, using gridsync to syncronize
        @cuda.jit( device=gridsync )
        def interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            i0(grid, vectors, scalars, ptype, sim_box, interaction_parameters[0])
            if num_interactions>1:
                grid.sync() # Not always necessary !!!
                i1(grid, vectors, scalars, ptype, sim_box, interaction_parameters[1])
            if num_interactions>2:
                grid.sync() # Not always necessary !!!
                i2(grid, vectors, scalars, ptype, sim_box, interaction_parameters[2])
            if num_interactions>3:
                grid.sync() # Not always necessary !!!
                i3(grid, vectors, scalars, ptype, sim_box, interaction_parameters[3])
            if num_interactions>4:
                grid.sync() # Not always necessary !!!
                i4(grid, vectors, scalars, ptype, sim_box, interaction_parameters[4])
            return
        return interactions, tuple(interaction_params_list)

    else:
        # A python function, making several kernel calls to syncronize  
        #@cuda.jit( device=gridsync )
        def interactions(grid, vectors, scalars, ptype, sim_box, interaction_parameters):
            i0(0, vectors, scalars, ptype, sim_box, interaction_parameters[0])
            if num_interactions>1:
                i1(0, vectors, scalars, ptype, sim_box, interaction_parameters[1])
            if num_interactions>2:
                i2(0, vectors, scalars, ptype, sim_box, interaction_parameters[2])
            if num_interactions>3:
                i3(0, vectors, scalars, ptype, sim_box, interaction_parameters[3])
            if num_interactions>4:
                i4(0, vectors, scalars, ptype, sim_box, interaction_parameters[4])
            return
        return interactions, tuple(interaction_params_list)
